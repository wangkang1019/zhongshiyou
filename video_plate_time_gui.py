import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import queue
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from tkinter import font as tkfont
import tkinter as tk

import cv2
import pandas as pd
import requests

#---- 服务配置 ----
DEFAULT_PLATE_URL = "http://localhost:8002/detect"
DEFAULT_PLATE_URLS = DEFAULT_PLATE_URL
DEFAULT_OCR_URL = "http://localhost:8868/predict/ocr_system"

#---- 图像和视频处理配置 ----
PLATE_CROP_SCALE = 1.5
PLATE_CROP_INTERPOLATION = cv2.INTER_CUBIC
DEFAULT_FRAME_INTERVAL_SECONDS = 1.0
DEFAULT_LOG_EVERY = 50
DEFAULT_TIMEOUT = ""
PLATE_CONTAINER_COUNT = 1
PLATE_WORKERS_PER_CONTAINER = 1
PLATE_WORKERS = PLATE_CONTAINER_COUNT * PLATE_WORKERS_PER_CONTAINER
PLATE_MAX_PENDING = PLATE_WORKERS * 2

#---- 车牌修正规则配置 ----
CHONGQING_CONFUSABLE_PREFIXES = set("川沪苏宁湘")
DEFAULT_ENABLE_CHONGQING_CORRECTION = True
TEMPORAL_VOTE_WINDOW = 5

#---- 区域配置 ----
REGIONS_PATH = Path(__file__).with_name("video_plate_time_regions.json")
DEFAULT_REGIONS = {
    "plate": {"x1_ratio": 0.0, "y1_ratio": 0.0, "x2_ratio": 1.0, "y2_ratio": 1.0},
    "time": {"x1_ratio": 975 / 1280, "y1_ratio": 0.0, "x2_ratio": 1215 / 1280, "y2_ratio": 60 / 960},
}

#---- Excel 输出配置 ----
EXCEL_COLUMNS = [
    "plate_no_raw", "plate_no", "plate_confidence", "plate_correction", "rect",
    "plate_color", "color_confidence", "time_text", "time_confidence", "text_region",
    "vedio_path", "video_name",
]


def preview_to_ratio(start, end, display_size, image_size=None):
    display_width, display_height = display_size
    x1, x2 = sorted((max(0, min(display_width, start[0])), max(0, min(display_width, end[0]))))
    y1, y2 = sorted((max(0, min(display_height, start[1])), max(0, min(display_height, end[1]))))
    return {
        "x1_ratio": round(x1 / display_width, 8),
        "y1_ratio": round(y1 / display_height, 8),
        "x2_ratio": round(x2 / display_width, 8),
        "y2_ratio": round(y2 / display_height, 8),
    }


def crop_from_ratio(width, height, region):
    x1 = max(0, min(width - 1, int(width * region["x1_ratio"])))
    y1 = max(0, min(height - 1, int(height * region["y1_ratio"])))
    x2 = max(x1 + 1, min(width, int(width * region["x2_ratio"])))
    y2 = max(y1 + 1, min(height, int(height * region["y2_ratio"])))
    return x1, y1, x2, y2


def scale_box(box, scale):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return tuple(int(round(float(value) * scale)) for value in box)


def save_regions(path, regions):
    Path(path).write_text(json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8")


def load_regions(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = json.loads(json.dumps(DEFAULT_REGIONS))
        for name in ("plate", "time"):
            if isinstance(data.get(name), dict):
                result[name].update({key: float(data[name][key]) for key in result[name] if key in data[name]})
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return json.loads(json.dumps(DEFAULT_REGIONS))


def encode_jpeg(image):
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("无法将图片编码为 JPEG")
    return buffer.tobytes()


def call_plate_service(url, frame, timeout, region):
    x1, y1, x2, y2 = crop_from_ratio(frame.shape[1], frame.shape[0], region)
    plate_crop = frame[y1:y2, x1:x2]
    plate_crop = cv2.resize(plate_crop, None, fx=PLATE_CROP_SCALE, fy=PLATE_CROP_SCALE, interpolation=PLATE_CROP_INTERPOLATION)
    image_bytes = encode_jpeg(plate_crop)
    response = requests.post(url, files={"file": ("plate_crop.jpg", image_bytes, "image/jpeg")}, timeout=timeout)
    response.raise_for_status()
    plates = response.json().get("plates", [])
    for plate in plates:
        rect = plate.get("rect")
        if isinstance(rect, list) and len(rect) == 4:
            plate["rect"] = [
                round(rect[0] / PLATE_CROP_SCALE) + x1,
                round(rect[1] / PLATE_CROP_SCALE) + y1,
                round(rect[2] / PLATE_CROP_SCALE) + x1,
                round(rect[3] / PLATE_CROP_SCALE) + y1,
            ]
    return plates


def order_plate_task_results(results):
    """Restore sampled-frame order after requests complete out of order."""
    return sorted(results, key=lambda item: item["frame_index"])


def normalize_plate_candidate(value):
    if value is None:
        return ""
    text = re.sub(r"\s+", "", str(value).upper())
    return text if re.fullmatch(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}", text) else ""


def correct_plate_rows(rows, enable_chongqing_correction=DEFAULT_ENABLE_CHONGQING_CORRECTION, temporal_window=TEMPORAL_VOTE_WINDOW):
    suffix_groups = {}
    for row in rows:
        raw_text = str(row.get("plate_no_raw", "")).strip()
        raw = normalize_plate_candidate(raw_text)
        row["plate_no_raw"] = raw_text
        row["plate_no"] = ""
        row["plate_correction"] = "格式过滤" if raw_text and not raw else ""
        if not raw:
            continue
        row["plate_no"] = raw
        suffix = raw[1:]
        group = suffix_groups.setdefault(suffix, [])
        group.append(row)

    for suffix, suffix_rows in suffix_groups.items():
        ordered_rows = sorted(suffix_rows, key=lambda row: row.get("_sample_index", row.get("_frame_index", 0)))
        clusters = []
        for row in ordered_rows:
            sample_index = row.get("_sample_index", row.get("_frame_index", 0))
            if not clusters or sample_index - clusters[-1][-1].get("_sample_index", clusters[-1][-1].get("_frame_index", 0)) > temporal_window:
                clusters.append([row])
            else:
                clusters[-1].append(row)

        for group in clusters:
            prefixes = [row["plate_no_raw"][0] for row in group]
            counts = {}
            weighted = {}
            for row, prefix in zip(group, prefixes):
                counts[prefix] = counts.get(prefix, 0) + 1
                weighted[prefix] = weighted.get(prefix, 0.0) + float(row.get("plate_confidence") or 0.0)
            winner = max(counts, key=lambda prefix: (counts[prefix], weighted[prefix]))
            mixed_confusable = len(set(prefixes)) > 1 and all(prefix in CHONGQING_CONFUSABLE_PREFIXES or prefix == "渝" for prefix in prefixes)
            stable_winner = enable_chongqing_correction and counts[winner] >= 2 and counts[winner] / len(group) >= 0.6
            use_chongqing = enable_chongqing_correction and mixed_confusable and not stable_winner
            for row in group:
                raw = row["plate_no_raw"]
                target = winner if stable_winner else ("渝" if use_chongqing else raw[0])
                corrected = target + suffix
                if corrected != raw:
                    row["plate_no"] = corrected
                    row["plate_correction"] = "连续帧首字投票" if stable_winner else "重庆首字先验修正"
    return rows


def normalize_time_text(text):
    digits = "".join(re.findall(r"\d", text or ""))
    if len(digits) < 12:
        return (text or "").strip()
    digits = (digits + "00")[:14] if len(digits) <= 12 else digits[:14]
    try:
        value = datetime(int(digits[:4]), int(digits[4:6]), int(digits[6:8]), int(digits[8:10]), int(digits[10:12]), int(digits[12:14]))
    except ValueError:
        return (text or "").strip()
    return value.strftime("%Y-%m-%d %H:%M:%S")


def parse_video_start_time(video_path):
    match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)", Path(video_path).stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def parse_recognized_datetime(text):
    normalized = normalize_time_text(text)
    try:
        return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def call_ocr_service(url, crop_image, timeout):
    payload = {"images": [base64.b64encode(encode_jpeg(crop_image)).decode("ascii")]}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    items = ((response.json().get("results") or [[]])[0] or [])
    items = [item for item in items if item.get("text")]
    if not items:
        return None
    items.sort(key=lambda item: min((point[0] for point in item.get("text_region", []) if len(point) >= 2), default=0))
    raw = " ".join(str(item["text"]).strip() for item in items)
    return {
        "text": normalize_time_text(raw),
        "confidence": max(float(item.get("confidence", 0) or 0) for item in items),
        "text_region": items[0].get("text_region") if len(items) == 1 else [item.get("text_region") for item in items],
    }


def default_output_path():
    return str(Path.cwd() / f"{datetime.now():%Y%m%d_%H%M%S}_video_plate_time_results.xlsx")


def collect_video_paths(path):
    """Return MP4 files from a file or recursively from a directory."""
    target = Path(path).expanduser()
    if target.is_file():
        return [target] if target.suffix.lower() == ".mp4" else []
    if not target.is_dir():
        return []
    return sorted(
        (item for item in target.rglob("*") if item.is_file() and item.suffix.lower() == ".mp4"),
        key=lambda item: str(item).lower(),
    )


def collect_videos_from_paths(paths_text):
    paths = [part.strip().strip('"') for part in re.split(r"[;\r\n]+", paths_text or "") if part.strip()]
    videos = {}
    for path in paths:
        for video in collect_video_paths(path):
            videos[str(video.resolve()).lower()] = video
    return sorted(videos.values(), key=lambda item: str(item).lower())


def parse_optional_timeout(value):
    return None if not str(value or "").strip() else max(0.001, float(value))


def parse_service_urls(value, fallback=DEFAULT_PLATE_URL):
    urls = [item.strip() for item in re.split(r"[,;\n]+", str(value or "")) if item.strip()]
    return urls or [fallback]


def process_video(video_path, options, log, preview_callback=None):
    rows = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, int(round(fps * options["frame_interval_seconds"]))) if fps > 0 else 1
    video_start = parse_video_start_time(video_path)
    frame_idx = processed = 0
    started = time.time()
    log(
        f"开始: {video_path.name}, fps={fps:.3f}, 每 {options['frame_interval_seconds']} 秒取一帧, "
        f"视频起始时间={video_start or '未从文件名识别'}, OCR校对={'开' if options['enable_ocr'] else '关'}"
    )
    pending = {}
    task_results = []
    plate_urls = options.get("plate_urls") or parse_service_urls(options.get("plate_url"))
    workers = max(1, int(options.get("plate_workers", PLATE_WORKERS)))
    max_pending = max(workers, int(options.get("plate_max_pending", PLATE_MAX_PENDING)))

    def consume(done_futures):
        for future in done_futures:
            task = pending.pop(future)
            try:
                plates = future.result()
            except Exception as exc:
                log(f"{video_path.name}: frame {task['frame_index']} 车牌请求失败: {exc}")
                plates = []
            if preview_callback:
                preview_callback(
                    task["frame"], plates, task["time_info"], task["time_box"],
                    task["frame_index"], str(video_path),
                )
            task_results.append({key: value for key, value in task.items() if key != "frame"} | {"plates": plates})

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="plate") as executor:
            while True:
                while len(pending) >= max_pending:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    consume(done)

                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % interval:
                    continue

                processed += 1
                x1, y1, x2, y2 = crop_from_ratio(frame.shape[1], frame.shape[0], options["time_region"])
                ocr_info = None
                if options["enable_ocr"]:
                    ocr_info = call_ocr_service(options["ocr_url"], frame[y1:y2, x1:x2], options["timeout"])
                timeline_time = video_start + timedelta(seconds=round((frame_idx - 1) / fps)) if video_start and fps > 0 else None
                ocr_time = parse_recognized_datetime(ocr_info.get("text")) if ocr_info else None
                final_time = timeline_time
                if ocr_time and (timeline_time is None or abs((ocr_time - timeline_time).total_seconds()) <= 5):
                    final_time = ocr_time
                time_info = {
                    "text": final_time.strftime("%Y-%m-%d %H:%M:%S") if final_time else "",
                    "confidence": ocr_info.get("confidence") if ocr_info and final_time == ocr_time else None,
                    "text_region": ocr_info.get("text_region", []) if ocr_info else [],
                }
                task = {
                    "frame_index": frame_idx,
                    "sample_index": processed,
                    "frame": frame.copy(),
                    "time_info": time_info,
                    "time_box": (x1, y1, x2, y2),
                }
                future = executor.submit(
                    call_plate_service,
                    plate_urls[(processed - 1) % len(plate_urls)],
                    frame,
                    options["timeout"],
                    options["plate_region"],
                )
                pending[future] = task
                if processed % options["log_every"] == 0:
                    log(
                        f"{video_path.name}: frame {frame_idx}/{total or '?'} "
                        f"submitted={processed}, pending={len(pending)}"
                    )

            if pending:
                done, _ = wait(tuple(pending))
                consume(done)
    finally:
        capture.release()

    for task in order_plate_task_results(task_results):
        plates = task["plates"]
        time_info = task["time_info"]
        if not time_info.get("text"):
            continue
        for plate in plates:
            if not str(plate.get("plate_no", "")).strip():
                continue
            rows.append({
                "plate_no_raw": str(plate.get("plate_no", "")).strip(),
                "plate_no": str(plate.get("plate_no", "")).strip(),
                "plate_confidence": plate.get("confidence"),
                "plate_correction": "",
                "rect": json.dumps(plate.get("rect", []), ensure_ascii=False),
                "plate_color": plate.get("plate_color", ""),
                "color_confidence": plate.get("color_confidence"),
                "time_text": time_info["text"],
                "time_confidence": time_info.get("confidence"),
                "text_region": json.dumps(time_info.get("text_region", []), ensure_ascii=False),
                "vedio_path": str(video_path.parent),
                "video_name": video_path.name,
                "_sample_index": task["sample_index"],
                "_frame_index": task["frame_index"],
            })

    rows.sort(key=lambda row: (row.get("_sample_index", 0), row.get("_frame_index", 0)))
    rows = correct_plate_rows(
        rows,
        options.get("enable_chongqing_correction", DEFAULT_ENABLE_CHONGQING_CORRECTION),
        options.get("temporal_vote_window", TEMPORAL_VOTE_WINDOW),
    )
    corrected_count = sum(1 for row in rows if row.get("plate_correction"))
    log(
        f"完成: {video_path.name}, 处理 {processed} 帧，写入 {len(rows)} 行，"
        f"修正 {corrected_count} 行，用时 {time.time() - started:.1f}s"
    )
    return rows


def write_excel(rows, output_path):
    df = pd.DataFrame(rows, columns=EXCEL_COLUMNS)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        sheet = writer.book["results"]
        for cells in sheet.columns:
            width = max((len(str(cell.value or "")) for cell in cells[:1000]), default=10)
            sheet.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 12), 60)


def read_table(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"不支持的文件格式: {suffix}")


def write_table(df, output_path):
    if Path(output_path).suffix.lower() == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        sheet = writer.book["results"]
        for cells in sheet.columns:
            width = max((len(str(cell.value or "")) for cell in cells[:1000]), default=10)
            sheet.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 12), 60)


def ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def normalize_plate_no(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", "", str(value).upper())
    return text if re.fullmatch(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}", text) else ""


def match_output_path(target_path, output_dir):
    target = Path(target_path)
    suffix = target.suffix if target.suffix.lower() == ".csv" else ".xlsx"
    return Path(output_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{target.stem}{suffix}"


def match_plate_data(source_path, target_path, output_dir, source_time_column="time_text", source_plate_column="plate_no", target_start_column="交易开始时间", target_plate_column="车牌号", progress_callback=None):
    source_df = read_table(source_path)
    target_df = read_table(target_path)
    missing_source = [column for column in (source_time_column, source_plate_column) if column not in source_df.columns]
    missing_target = [column for column in (target_start_column,) if column not in target_df.columns]
    if missing_source:
        raise ValueError(f"源车牌数据缺少列: {', '.join(missing_source)}")
    if missing_target:
        raise ValueError(f"需要处理数据缺少列: {', '.join(missing_target)}")

    source = source_df[[source_time_column, source_plate_column]].copy()
    source["_match_time"] = pd.to_datetime(source[source_time_column], errors="coerce")
    source["_plate_no"] = source[source_plate_column].map(normalize_plate_no)
    source = source.dropna(subset=["_match_time"])
    source = source[source["_plate_no"] != ""]
    source["_match_second"] = source["_match_time"].dt.floor("s")
    start_times = pd.to_datetime(target_df[target_start_column], errors="coerce").dt.floor("s")
    results = []
    total = len(start_times)
    started = time.time()
    for index, start_time in enumerate(start_times, start=1):
        if pd.isna(start_time):
            results.append("")
        else:
            matched = source.loc[source["_match_second"] == start_time, "_plate_no"]
            results.append("/".join(ordered_unique(matched)))
        if progress_callback and (index == 1 or index == total or index % 100 == 0):
            progress_callback(index, total, len(results), started)

    output_df = target_df.copy()
    output_df[target_plate_column] = results
    output_path = match_output_path(target_path, output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_table(output_df, output_path)
    return output_path, len(output_df)


class TkText(ttk.Frame):
    def __init__(self, master, height=10):
        super().__init__(master)
        self.text = tk.Text(self, height=height, wrap="word")
        scrollbar = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def append(self, message):
        self.text.insert("end", message + "\n")
        self.text.see("end")


class VideoTab(ttk.Frame):
    def __init__(self, master, root):
        super().__init__(master, padding=10)
        self.root = root
        self.log_queue = queue.Queue()
        self.video_paths = []
        self.video_root = StringVar()
        self.first_frame = None
        self.preview_photo = None
        self.preview_size = (1, 1)
        self.preview_frame_queue = queue.Queue(maxsize=2)
        self.preview_results = {}
        self.preview_lock = threading.Lock()
        self.preview_current = None
        self.preview_playback_thread = None
        self.preview_stop_event = threading.Event()
        self.selection_mode = None
        self.selection_start = None
        self.selection_rect = None
        self.regions = load_regions(REGIONS_PATH)
        self.plate_url = StringVar(value=DEFAULT_PLATE_URLS)
        self.ocr_url = StringVar(value=DEFAULT_OCR_URL)
        self.output_path = StringVar(value=default_output_path())
        self.timeout = StringVar(value=DEFAULT_TIMEOUT)
        self.frame_interval = DoubleVar(value=DEFAULT_FRAME_INTERVAL_SECONDS)
        self.log_every = IntVar(value=DEFAULT_LOG_EVERY)
        self.playback_speed = DoubleVar(value=10.0)
        self.playback_speed_value = 10.0
        self.playback_speed_lock = threading.Lock()
        self.enable_ocr = BooleanVar(value=False)
        self.enable_chongqing_correction = BooleanVar(value=DEFAULT_ENABLE_CHONGQING_CORRECTION)
        self.running = BooleanVar(value=False)
        self.result_font = tkfont.Font(family="Microsoft YaHei UI", size=10, weight="bold")
        self.build_ui()
        self.playback_speed.trace_add("write", self.on_playback_speed_changed)
        self.after(100, self.flush_logs)
        self.after(50, self.flush_preview)

    def build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(1, weight=1)
        files = ttk.LabelFrame(left, text="视频与输出", padding=8)
        files.pack(fill="x")
        ttk.Button(files, text="选择 MP4 视频", command=self.select_videos).grid(row=0, column=0, sticky="w")
        self.video_label = ttk.Label(files, text="未选择视频")
        self.video_label.grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(files, text="视频路径").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(files, textvariable=self.video_root).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="添加文件夹", command=self.select_video_folder).grid(row=1, column=2)
        ttk.Label(files, text="输出 Excel").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(files, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="选择", command=self.select_output).grid(row=2, column=2)
        files.columnconfigure(1, weight=1)

        services = ttk.LabelFrame(left, text="Docker 服务", padding=8)
        services.pack(fill="x", pady=8)
        ttk.Label(services, text="车牌服务").grid(row=0, column=0, sticky="w")
        ttk.Entry(services, textvariable=self.plate_url, width=72).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(services, text="OCR 服务").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(services, textvariable=self.ocr_url, width=44).grid(row=1, column=1, sticky="ew", padx=8)
        services.columnconfigure(1, weight=1)

        run = ttk.LabelFrame(left, text="处理参数", padding=8)
        run.pack(fill="x")
        for row, (label, var) in enumerate((("间隔秒", self.frame_interval), ("timeout", self.timeout), ("log_every", self.log_every))):
            ttk.Label(run, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(run, textvariable=var, width=14).grid(row=row, column=1, sticky="w", padx=8)
        ttk.Label(run, text="timeout 留空表示不限制请求超时").grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(run, text="启用 OCR 时间校对（默认关闭）", variable=self.enable_ocr).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(run, text="启用重庆车牌首字修正和连续帧投票（默认开启）", variable=self.enable_chongqing_correction).grid(row=4, column=0, columnspan=3, sticky="w")

        playback = ttk.LabelFrame(left, text="Playback", padding=8)
        playback.pack(fill="x", pady=(8, 0))
        ttk.Label(playback, text="speed (1=original)").pack(side="left")
        ttk.Entry(playback, textvariable=self.playback_speed, width=8).pack(side="left", padx=8)
        ttk.Button(playback, text="应用", command=self.apply_playback_speed).pack(side="left")

        area = ttk.LabelFrame(left, text="识别区域", padding=8)
        area.pack(fill="x", pady=8)
        ttk.Button(area, text="框选车牌区域", command=lambda: self.begin_selection("plate")).grid(row=0, column=0, padx=3, pady=3)
        ttk.Button(area, text="框选时间区域", command=lambda: self.begin_selection("time")).grid(row=0, column=1, padx=3, pady=3)
        ttk.Button(area, text="保存区域", command=self.save_area).grid(row=0, column=2, padx=3, pady=3)
        self.area_label = ttk.Label(area, text="已加载默认/上次保存区域")
        self.area_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Button(left, text="开始处理视频", command=self.start_video).pack(anchor="w", pady=8)
        self.video_log = TkText(left, height=13)
        self.video_log.pack(fill="both", expand=True)

        right = ttk.LabelFrame(self, text="第一帧预览：选择模式后点击左上角和右下角", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.preview = tk.Canvas(right, width=640, height=480, background="#202020", cursor="crosshair")
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Button-1>", self.on_preview_click)

    def select_videos(self):
        paths = filedialog.askopenfilenames(title="选择 MP4 视频", filetypes=[("MP4 videos", "*.mp4"), ("All files", "*.*")])
        if not paths:
            return
        self.video_paths = [Path(path) for path in paths]
        self.video_root.set("")
        self.video_label.config(text=f"已选择 {len(self.video_paths)} 个视频")
        self.load_first_frame(self.video_paths[0])

    def select_video_folder(self):
        path = filedialog.askdirectory(title="选择视频路径（递归查找所有 MP4）")
        if not path:
            return
        current = self.video_root.get().strip()
        paths = [part.strip() for part in re.split(r"[;\r\n]+", current) if part.strip()]
        if path not in paths:
            paths.append(path)
        self.video_root.set(";".join(paths))
        self.video_paths = collect_videos_from_paths(self.video_root.get())
        self.video_label.config(text=f"已找到 {len(self.video_paths)} 个视频（{len(paths)} 个路径，含子文件夹）")
        if self.video_paths:
            self.load_first_frame(self.video_paths[0])
        else:
            messagebox.showwarning("提示", "该路径及其子文件夹中没有 MP4 视频")

    def load_first_frame(self, path):
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if ok:
            self.first_frame = frame
            self.render_preview()

    def render_preview(self):
        if self.first_frame is None:
            return
        height, width = self.first_frame.shape[:2]
        scale = min(640 / width, 480 / height, 1.0)
        self.preview_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(self.first_frame, self.preview_size)
        for name, color in (("plate", "#00ff66"), ("time", "#ffcc00")):
            x1, y1, x2, y2 = crop_from_ratio(*self.preview_size, self.regions[name])
            cv2.rectangle(resized, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0) if name == "plate" else (0, 200, 255), 2)
        ok, buffer = cv2.imencode(".png", resized)
        if not ok:
            return
        self.preview_photo = tk.PhotoImage(data=base64.b64encode(buffer).decode("ascii"))
        self.preview.delete("all")
        self.preview.create_image(0, 0, image=self.preview_photo, anchor="nw")

    def queue_preview(self, frame, plates, time_info, time_box, frame_index=None, video_path=None):
        key = (str(video_path or ""), int(frame_index or 0))
        with self.preview_lock:
            self.preview_results[key] = (plates, time_info, time_box)

    def flush_preview(self):
        while True:
            try:
                token, frame_index, frame, time_info, time_box = self.preview_frame_queue.get_nowait()
            except queue.Empty:
                break
            key = (token, frame_index)
            with self.preview_lock:
                result = self.preview_results.pop(key, None)
            self.preview_current = (token, frame_index, frame, time_info, time_box, result)
            self.render_preview_state()
        self.refresh_current_preview()
        self.after(50, self.flush_preview)

    def refresh_current_preview(self):
        if not self.preview_current:
            return
        token, frame_index, frame, time_info, time_box, _ = self.preview_current
        key = (token, frame_index)
        with self.preview_lock:
            result = self.preview_results.pop(key, None)
        if result is not None:
            self.preview_current = (token, frame_index, frame, time_info, time_box, result)
            self.render_preview_state()

    def render_preview_state(self):
        if not self.preview_current:
            return
        _, _, frame, time_info, time_box, result = self.preview_current
        plates = result[0] if result else []
        result_time = result[1] if result else time_info
        result_box = result[2] if result else time_box
        self.render_inference_frame(frame, plates, result_time, result_box)

    def start_preview_playback(self, video_path, interval, speed):
        self.stop_preview_playback()
        self.preview_stop_event.clear()
        thread = threading.Thread(
            target=self.preview_reader,
            args=(Path(video_path), interval, speed),
            daemon=True,
        )
        self.preview_playback_thread = thread
        thread.start()

    def stop_preview_playback(self):
        self.preview_stop_event.set()
        thread = self.preview_playback_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self.preview_playback_thread = None

    def preview_reader(self, video_path, interval, speed):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        sample_interval = max(1, int(round(fps * interval))) if fps > 0 else 1
        video_start = parse_video_start_time(video_path)
        frame_index = 0
        next_display = time.perf_counter()
        try:
            while not self.preview_stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if (frame_index - 1) % sample_interval:
                    continue
                delay = max(0.001, interval / max(0.1, self.get_playback_speed()))
                wait_time = next_display - time.perf_counter()
                if wait_time > 0:
                    self.preview_stop_event.wait(wait_time)
                if self.preview_stop_event.is_set():
                    break
                timeline_time = video_start + timedelta(seconds=round((frame_index - 1) / fps)) if video_start and fps > 0 else None
                time_info = {"text": timeline_time.strftime("%Y-%m-%d %H:%M:%S") if timeline_time else "", "confidence": None, "text_region": []}
                x1, y1, x2, y2 = crop_from_ratio(frame.shape[1], frame.shape[0], self.regions["time"])
                item = (str(video_path), frame_index, frame, time_info, (x1, y1, x2, y2))
                try:
                    self.preview_frame_queue.put(item, timeout=0.2)
                except queue.Full:
                    try:
                        self.preview_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.preview_frame_queue.put_nowait(item)
                    except queue.Full:
                        pass
                next_display = max(next_display + delay, time.perf_counter())
        finally:
            capture.release()

    def wait_preview_playback(self):
        self.stop_preview_playback()

    def render_inference_frame(self, frame, plates, time_info, time_box):
        height, width = frame.shape[:2]
        scale = min(640 / width, 480 / height, 1.0)
        self.preview_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(frame, self.preview_size)
        for plate in plates or []:
            rect = scale_box(plate.get("rect"), scale)
            if rect:
                cv2.rectangle(resized, rect[:2], rect[2:], (0, 255, 0), 2)
        scaled_time_box = scale_box(time_box, scale)
        if scaled_time_box:
            cv2.rectangle(resized, scaled_time_box[:2], scaled_time_box[2:], (0, 200, 255), 2)

        ok, buffer = cv2.imencode(".png", resized)
        if not ok:
            return
        self.preview_photo = tk.PhotoImage(data=base64.b64encode(buffer).decode("ascii"))
        self.preview.delete("all")
        self.preview.create_image(0, 0, image=self.preview_photo, anchor="nw")

        for plate in plates or []:
            rect = scale_box(plate.get("rect"), scale)
            plate_no = str(plate.get("plate_no", "")).strip()
            if rect and plate_no:
                self.draw_result_label(rect[2], rect[3], plate_no, "#00ff66")
        if scaled_time_box and time_info and time_info.get("text"):
            self.draw_result_label(scaled_time_box[2], scaled_time_box[3], str(time_info["text"]), "#ffcc00")

    def draw_result_label(self, x, y, text, color):
        label = str(text)
        width = max(42, min(360, self.result_font.measure(label) + 10))
        x = max(width, min(self.preview_size[0], x))
        y = min(self.preview_size[1] - 4, max(18, y + 18))
        self.preview.create_rectangle(x - width, y - 18, x, y, fill="#111111", outline=color, width=1)
        self.preview.create_text(x - 5, y - 9, text=label, anchor="e", fill=color, font=self.result_font)

    def begin_selection(self, mode):
        if self.first_frame is None:
            messagebox.showwarning("提示", "请先选择视频")
            return
        self.selection_mode = mode
        self.selection_start = None
        self.area_label.config(text=f"当前框选：{('车牌' if mode == 'plate' else '时间')}，请点击左上角和右下角")

    def on_preview_click(self, event):
        if not self.selection_mode or self.first_frame is None:
            return
        x = max(0, min(self.preview_size[0], event.x))
        y = max(0, min(self.preview_size[1], event.y))
        if self.selection_start is None:
            self.selection_start = (x, y)
            self.area_label.config(text="已点击左上角，请点击右下角")
            return
        self.regions[self.selection_mode] = preview_to_ratio(self.selection_start, (x, y), self.preview_size, self.first_frame.shape[1::-1])
        self.selection_mode = None
        self.selection_start = None
        self.area_label.config(text="框选已更新，请点击保存区域")
        self.render_preview()

    def save_area(self):
        save_regions(REGIONS_PATH, self.regions)
        self.area_label.config(text=f"区域已保存：{REGIONS_PATH.name}")

    def select_output(self):
        path = filedialog.asksaveasfilename(title="保存 Excel", defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")], initialfile=Path(self.output_path.get()).name)
        if path:
            self.output_path.set(path)

    def log(self, message):
        self.log_queue.put(message)

    def get_playback_speed(self):
        with self.playback_speed_lock:
            return self.playback_speed_value

    def apply_playback_speed(self):
        self.on_playback_speed_changed()
        self.log(f"playback speed applied: {self.get_playback_speed():g}x")

    def on_playback_speed_changed(self, *_args):
        try:
            speed = max(0.1, float(self.playback_speed.get()))
        except (TypeError, ValueError, tk.TclError):
            return
        with self.playback_speed_lock:
            self.playback_speed_value = speed

    def flush_logs(self):
        while True:
            try:
                self.video_log.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        self.after(100, self.flush_logs)

    def start_video(self):
        if self.running.get():
            return
        if self.video_root.get().strip():
            self.video_paths = collect_videos_from_paths(self.video_root.get())
        if not self.video_paths:
            messagebox.showwarning("提示", "请先选择 MP4 视频，或填写包含视频的文件夹路径")
            return
        self.running.set(True)
        threading.Thread(target=self.video_worker, daemon=True).start()

    def video_worker(self):
        try:
            options = {
                "plate_urls": parse_service_urls(self.plate_url.get()), "ocr_url": self.ocr_url.get().strip(),
                "timeout": parse_optional_timeout(self.timeout.get()),
                "frame_interval_seconds": max(0.001, float(self.frame_interval.get())),
                "log_every": max(1, int(self.log_every.get())),
                "enable_ocr": bool(self.enable_ocr.get()),
                "enable_chongqing_correction": bool(self.enable_chongqing_correction.get()),
                "temporal_vote_window": TEMPORAL_VOTE_WINDOW,
                "plate_workers": PLATE_WORKERS,
                "plate_max_pending": PLATE_MAX_PENDING,
                "plate_region": self.regions["plate"], "time_region": self.regions["time"],
            }
            rows = []
            playback_speed = max(0.1, float(self.playback_speed.get()))
            for path in self.video_paths:
                self.start_preview_playback(path, options["frame_interval_seconds"], playback_speed)
                rows.extend(process_video(path, options, self.log, self.queue_preview))
            output = Path(self.output_path.get())
            output.parent.mkdir(parents=True, exist_ok=True)
            write_excel(rows, output)
            self.log(f"完成: {output}，共 {len(rows)} 行")
            self.root.after(0, lambda: messagebox.showinfo("完成", f"已生成 Excel：\n{output}\n共 {len(rows)} 行"))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self.root.after(0, lambda: messagebox.showerror("错误", str(exc)))
        finally:
            self.root.after(0, lambda: self.running.set(False))


class MatchTab(ttk.Frame):
    def __init__(self, master, root):
        super().__init__(master, padding=12)
        self.root = root
        self.source_path = StringVar()
        self.target_path = StringVar()
        self.output_dir = StringVar(value=str(Path.cwd()))
        self.source_time = StringVar(value="time_text")
        self.source_plate = StringVar(value="plate_no")
        self.target_start = StringVar(value="交易开始时间")
        self.target_plate = StringVar(value="车牌号")
        self.running = BooleanVar(value=False)
        self.log_queue = queue.Queue()
        self.build_ui()
        self.after(100, self.flush_logs)

    def build_ui(self):
        files = ttk.LabelFrame(self, text="输入输出", padding=10)
        files.pack(fill="x")
        for row, label, var, command in ((0, "源车牌数据", self.source_path, self.select_source), (1, "需要处理数据", self.target_path, self.select_target), (2, "处理后数据路径", self.output_dir, self.select_output_dir)):
            ttk.Label(files, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(files, textvariable=var, width=82).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(files, text="选择", command=command).grid(row=row, column=2)
        files.columnconfigure(1, weight=1)
        columns = ttk.LabelFrame(self, text="列名配置", padding=10)
        columns.pack(fill="x", pady=8)
        for idx, (label, var) in enumerate((("源时间列", self.source_time), ("源车牌列", self.source_plate), ("交易开始列", self.target_start), ("写入车牌列", self.target_plate))):
            row, col = idx // 2, (idx % 2) * 2
            ttk.Label(columns, text=label).grid(row=row, column=col, sticky="w", pady=4)
            ttk.Entry(columns, textvariable=var, width=28).grid(row=row, column=col + 1, sticky="w", padx=(8, 18))
        ttk.Button(self, text="开始匹配", command=self.start).pack(anchor="w", pady=8)
        self.log_text = TkText(self, height=18)
        self.log_text.pack(fill="both", expand=True)

    def select_source(self):
        path = filedialog.askopenfilename(title="选择源车牌数据", filetypes=[("Tables", "*.xlsx *.xlsm *.xls *.csv")])
        if path: self.source_path.set(path)

    def select_target(self):
        path = filedialog.askopenfilename(title="选择需要处理数据", filetypes=[("Tables", "*.xlsx *.xlsm *.xls *.csv")])
        if path:
            self.target_path.set(path)
            self.output_dir.set(str(Path(path).parent))

    def select_output_dir(self):
        path = filedialog.askdirectory(title="选择处理后数据路径")
        if path: self.output_dir.set(path)

    def flush_logs(self):
        while True:
            try: self.log_text.append(self.log_queue.get_nowait())
            except queue.Empty: break
        self.after(100, self.flush_logs)

    def log(self, message): self.log_queue.put(message)

    def start(self):
        if self.running.get(): return
        if not self.source_path.get().strip() or not self.target_path.get().strip():
            messagebox.showwarning("提示", "请先选择源车牌数据和需要处理数据")
            return
        self.running.set(True)
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        try:
            def progress(current, total, _matched, started):
                percent = current / total * 100 if total else 100
                self.log(f"进度: {current}/{total} ({percent:.1f}%), 已耗时 {time.time() - started:.1f}s")
            output, count = match_plate_data(self.source_path.get().strip(), self.target_path.get().strip(), self.output_dir.get().strip(), self.source_time.get().strip(), self.source_plate.get().strip(), self.target_start.get().strip(), self.target_plate.get().strip(), progress)
            self.log(f"完成: {output}，处理行数: {count}")
            self.root.after(0, lambda: messagebox.showinfo("完成", f"已生成：\n{output}\n处理行数：{count}"))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self.root.after(0, lambda: messagebox.showerror("错误", str(exc)))
        finally:
            self.root.after(0, lambda: self.running.set(False))


class App:
    def __init__(self, root):
        self.root = root
        root.title("监控视频车牌时间工具")
        root.geometry("1380x820")
        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True)
        tabs.add(VideoTab(tabs, root), text="提取数据")
        tabs.add(MatchTab(tabs, root), text="匹配数据")


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

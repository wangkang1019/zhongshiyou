import base64
import os
import tempfile
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import paddle
except Exception:  # pragma: no cover - reported by /health
    paddle = None

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - reported by /health
    PaddleOCR = None


app = FastAPI(title="PaddleOCR Time Service", version="1.0.0")

_ocr = None
_ocr_lock = threading.Lock()


class OCRRequest(BaseModel):
    images: list[str] = Field(..., description="Base64 encoded images")


def _to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _decode_base64_image(image_base64: str) -> bytes:
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    try:
        return base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


def _get_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr

    with _ocr_lock:
        if _ocr is not None:
            return _ocr
        if PaddleOCR is None:
            raise RuntimeError("paddleocr package is not available")

        kwargs = {
            "lang": os.getenv("PADDLEOCR_LANG", "ch"),
            "device": os.getenv("PADDLEOCR_DEVICE", "gpu"),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        ocr_version = os.getenv("PADDLEOCR_VERSION")
        if ocr_version:
            kwargs["ocr_version"] = ocr_version
        _ocr = PaddleOCR(**kwargs)
        return _ocr


def _extract_items(result: Any) -> list[dict[str, Any]]:
    result = _to_plain(result)
    if isinstance(result, dict):
        data = result.get("res", result)
    else:
        data = result

    if not isinstance(data, dict):
        return []

    texts = data.get("rec_texts") or data.get("texts") or []
    scores = data.get("rec_scores") or data.get("scores") or []
    regions = data.get("rec_polys") or data.get("dt_polys") or data.get("text_regions") or []

    items = []
    for idx, text in enumerate(texts):
        if not text:
            continue
        confidence = scores[idx] if idx < len(scores) else None
        text_region = regions[idx] if idx < len(regions) else None
        items.append(
            {
                "text": str(text),
                "confidence": float(confidence) if confidence is not None else None,
                "text_region": text_region,
            }
        )
    return items


@app.get("/health")
def health():
    cuda_available = False
    paddle_version = None
    if paddle is not None:
        paddle_version = getattr(paddle, "__version__", None)
        try:
            cuda_available = bool(paddle.device.is_compiled_with_cuda())
        except Exception:
            cuda_available = False
    return {
        "status": "ok",
        "model_loaded": _ocr is not None,
        "paddle_available": paddle is not None,
        "paddleocr_available": PaddleOCR is not None,
        "paddle_version": paddle_version,
        "cuda_available": cuda_available,
        "device": os.getenv("PADDLEOCR_DEVICE", "gpu"),
    }


@app.post("/predict/ocr_system")
def predict_ocr_system(payload: OCRRequest):
    ocr = _get_ocr()
    batch_results = []

    for encoded_image in payload.images:
        image_bytes = _decode_base64_image(encoded_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            prediction = ocr.predict(tmp_path)
            image_items = []
            for res in prediction:
                json_result = getattr(res, "json", res)
                image_items.extend(_extract_items(json_result))
            batch_results.append(image_items)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {"status": "0", "msg": "", "results": batch_results}

import os
import threading
from typing import Any, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from detect_plate import detect_Recognition_plate, load_model
from plate_recognition.plate_rec import init_model


DETECT_MODEL_PATH = os.getenv("PLATE_DETECT_MODEL", "weights/plate_detect.pt")
REC_MODEL_PATH = os.getenv("PLATE_REC_MODEL", "weights/plate_rec_color.pth")
IMG_SIZE = int(os.getenv("PLATE_IMG_SIZE", "640"))
IS_COLOR = os.getenv("PLATE_IS_COLOR", "true").lower() not in {"0", "false", "no"}

app = FastAPI(title="Chinese License Plate Recognition Service", version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detect_model: Optional[torch.nn.Module] = None
plate_rec_model: Optional[torch.nn.Module] = None
model_lock = threading.Lock()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cuda_available: bool
    device: str
    detect_model_path: str
    rec_model_path: str
    img_size: int
    is_color: bool


class PlateResult(BaseModel):
    plate_no: str
    confidence: float
    rect: list[int]
    detect_confidence: float
    recognition_confidence: float
    plate_color: str = ""
    color_confidence: float = 0.0


class DetectResponse(BaseModel):
    plates: list[PlateResult]


def load_models() -> None:
    global detect_model, plate_rec_model
    if detect_model is None:
        detect_model = load_model(DETECT_MODEL_PATH, device)
        detect_model.eval()
    if plate_rec_model is None:
        plate_rec_model = init_model(device, REC_MODEL_PATH, is_color=IS_COLOR)
        plate_rec_model.eval()


@app.on_event("startup")
def startup() -> None:
    load_models()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if detect_model is not None and plate_rec_model is not None else "loading",
        model_loaded=detect_model is not None and plate_rec_model is not None,
        cuda_available=torch.cuda.is_available(),
        device=str(device),
        detect_model_path=DETECT_MODEL_PATH,
        rec_model_path=REC_MODEL_PATH,
        img_size=IMG_SIZE,
        is_color=IS_COLOR,
    )


def decode_image(content: bytes) -> np.ndarray:
    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return image


def to_float(value: Any) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(np.mean(value.astype(float)))
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def normalize_result(item: dict[str, Any]) -> PlateResult:
    rec_confidence = to_float(item.get("rec_conf", 0.0))
    detect_confidence = to_float(item.get("detect_conf", 0.0))
    confidence = rec_confidence if rec_confidence > 0 else detect_confidence
    rect = [int(value) for value in item.get("rect", [])]

    return PlateResult(
        plate_no=str(item.get("plate_no", "")),
        confidence=confidence,
        rect=rect,
        detect_confidence=detect_confidence,
        recognition_confidence=rec_confidence,
        plate_color=str(item.get("plate_color", "")),
        color_confidence=to_float(item.get("color_conf", 0.0)) if "color_conf" in item else 0.0,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    content = await file.read()
    image = decode_image(content)
    load_models()

    with model_lock, torch.no_grad():
        assert detect_model is not None
        assert plate_rec_model is not None
        raw_results = detect_Recognition_plate(
            detect_model,
            image,
            device,
            plate_rec_model,
            IMG_SIZE,
            is_color=IS_COLOR,
        )

    plates = [normalize_result(item) for item in raw_results if item.get("plate_no")]
    return DetectResponse(plates=plates)

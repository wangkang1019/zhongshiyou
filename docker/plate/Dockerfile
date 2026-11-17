ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

WORKDIR /app/Chinese_license_plate_detection_recognition

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY docker/plate/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY Chinese_license_plate_detection_recognition/detect_plate.py ./detect_plate.py
COPY Chinese_license_plate_detection_recognition/models ./models
COPY Chinese_license_plate_detection_recognition/utils ./utils
COPY Chinese_license_plate_detection_recognition/plate_recognition ./plate_recognition
COPY Chinese_license_plate_detection_recognition/weights/plate_detect.pt ./weights/plate_detect.pt
COPY Chinese_license_plate_detection_recognition/weights/plate_rec_color.pth ./weights/plate_rec_color.pth
COPY docker/plate/service_api.py ./service_api.py

EXPOSE 8002

CMD ["uvicorn", "service_api:app", "--host", "0.0.0.0", "--port", "8002"]

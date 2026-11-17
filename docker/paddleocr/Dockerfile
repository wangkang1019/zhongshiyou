ARG PADDLE_BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5
FROM ${PADDLE_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEOCR=3.7.0 \
    PADDLEOCR_DEVICE=gpu \
    PADDLEOCR_LANG=ch

WORKDIR /app/PaddleOCR

RUN python -m pip install --upgrade pip setuptools wheel

COPY PaddleOCR/pyproject.toml PaddleOCR/setup.py PaddleOCR/README.md PaddleOCR/LICENSE ./
COPY PaddleOCR/paddleocr ./paddleocr
COPY PaddleOCR/ppocr ./ppocr
COPY PaddleOCR/ppstructure ./ppstructure
COPY PaddleOCR/tools ./tools
COPY PaddleOCR/configs ./configs
COPY PaddleOCR/deploy ./deploy
COPY docker/paddleocr/service_api.py ./service_api.py

RUN python -m pip install --ignore-installed "PyYAML>=6" \
    && python -m pip install --ignore-installed paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && python -m pip install -e . \
    && python -m pip install fastapi "uvicorn[standard]" python-multipart

EXPOSE 8868

CMD ["uvicorn", "service_api:app", "--host", "0.0.0.0", "--port", "8868"]

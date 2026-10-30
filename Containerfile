FROM docker.io/library/python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    fastapi==0.115.12 "uvicorn==0.34.2" numpy==2.2.5 pillow==11.2.1 \
    matplotlib==3.10.1 trimesh==4.6.8 pyrealsense2==2.58.4.10922
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS test
RUN pip install --no-cache-dir pytest==8.3.5 httpx==0.28.1

FROM test AS e2e
RUN pip install --no-cache-dir playwright==1.60.0 \
    && playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/*

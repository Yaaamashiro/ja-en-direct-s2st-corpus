# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-pip \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
        torch==2.7.1 \
        torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app

COPY requirements/runtime.txt /app/requirements/runtime.txt
RUN python3 -m pip install -r /app/requirements/runtime.txt
RUN python3 -c "import pyopenjtalk; print(pyopenjtalk.g2p('テスト', kana=True))"

COPY pyproject.toml /app/pyproject.toml
COPY LICENSE /app/LICENSE
COPY configs /app/configs
COPY src /app/src
COPY tests /app/tests
RUN python3 -m pip install --no-deps /app

ENTRYPOINT ["python3", "-m", "s2st_corpus.cli"]

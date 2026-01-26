FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    sox \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# pip
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch с CUDA
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Код
COPY whisper_pyannote_gpu_v3.py .

# Каталоги для данных
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python3", "whisper_pyannote_gpu_v3.py"]


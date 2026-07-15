ARG CUDA_VERSION="12.4.1"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    python3-dev python3-pip python3.10-venv \
    libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 \
    ffmpeg git git-lfs wget unzip && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support
ARG INDEX_URL="https://download.pytorch.org/whl/cu124"
ARG TORCH_VERSION="2.6.0+cu124"
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL}

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install onnxruntime-gpu

# Clone CodeFormer first (download script places weights under CodeFormer/CodeFormer/weights/)
RUN git lfs install && \
    git clone https://huggingface.co/spaces/sczhou/CodeFormer

# Download all models (cached layer — only re-runs when download script changes)
COPY scripts/download_models.py /tmp/
RUN pip3 install --no-cache-dir tqdm requests && \
    python3 /tmp/download_models.py /app && \
    rm /tmp/download_models.py

# Copy application code
COPY app/ ./app/
COPY examples/ ./examples/

EXPOSE 5000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

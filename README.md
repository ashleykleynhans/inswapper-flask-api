# FaceSwap API

GPU-accelerated face swapping API with 13 face swap models,
CodeFormer restoration, and VRAM-safe serial queue processing.

Powered by [insightface](https://github.com/deepinsight/insightface).

## Disclaimer

This project is intended for educational and research purposes only.
Users are solely responsible for complying with all applicable laws and
regulations regarding the use of face swapping technology. Do not use
this software to create content that violates others' privacy, dignity,
or rights. The author assumes no liability for any misuse.

## Features

- **13 face swap models**: inswapper, simswap, ghost, hififace, hyperswap, blendswap, uniface
- **CodeFormer face restoration**: with RealESRGAN background upscaling
- **Face selector**: gender and age filtering, 7 sort orders
- **Identity blending**: configurable source/target identity weight
- **Configurable mask**: per-edge padding and blur control
- **VRAM-safe queue**: async job queue processes one request at a time
- **Auto-generated API docs**: available at `/docs` (Swagger UI)

## Project Structure

```
├── app/
│   ├── main.py                     # FastAPI app, lifespan, CLI entry point
│   ├── config.py                   # Settings, paths, defaults, logging
│   ├── models/
│   │   ├── requests.py             # Pydantic request model (FaceSwapRequest)
│   │   └── responses.py            # Pydantic response models
│   ├── routes/
│   │   ├── health.py               # GET / health check
│   │   └── faceswap.py             # POST /faceswap, GET status, POST /sync
│   ├── services/
│   │   ├── image_utils.py          # Base64 encode/decode, temp file I/O
│   │   ├── face_analyzer.py        # insightface buffalo_l wrapper
│   │   ├── face_swapper.py         # Enhanced swap pipeline (FaceFusion fork)
│   │   ├── face_swapper_models.py  # 13 model definitions + metadata
│   │   ├── face_selector.py        # Gender/age filter, 7 sort orders
│   │   ├── face_swap_service.py    # Orchestration: process + face_swap
│   │   └── restoration.py          # CodeFormer + RealESRGAN
│   └── job_queue/
│       ├── models.py               # Job dataclass, status lifecycle
│       └── worker.py               # asyncio.Queue + serial worker
├── scripts/
│   └── download_models.py          # Download all ONNX + converter weights
├── tests/
│   ├── conftest.py                     # Shared fixtures, mock ML imports
│   ├── test_config.py                  # Config and Timer tests
│   ├── test_models.py                  # Pydantic request/response validation
│   ├── test_image_utils*.py            # Image encoding, file I/O
│   ├── test_face_selector*.py          # Gender/age filter, 7 sort orders
│   ├── test_face_swapper_models*.py    # Model validation and metadata
│   ├── test_face_swapper_*.py          # Enhanced swap pipeline (mocked ONNX)
│   ├── test_face_swap_service_full.py  # Orchestration validation
│   ├── test_queue_models.py            # Job dataclass and JobStatus enum
│   ├── test_queue_worker*.py           # asyncio.Queue worker (serial, TTL, errors)
│   ├── test_restoration_*.py           # CodeFormer loading and pipeline
│   ├── test_routes.py                  # Health, async/sync endpoints
│   ├── test_main.py                    # App lifespan, error handlers
│   └── test_coverage_gaps.py           # Edge case coverage
├── examples/
│   └── face_swap.py                # Async + sync client examples
├── Dockerfile                      # CUDA 12.4.1, onnxruntime-gpu, uvicorn
├── requirements.txt
├── pytest.ini
└── .coveragerc
```

## Installation

### Clone this repository

```bash
git clone https://github.com/ashleykleynhans/faceswap-api.git
cd faceswap-api
```

### Install the required Python dependencies

#### Linux and Mac

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

#### Windows

```
python3 -m venv .venv
.venv\Scripts\activate
pip3 install -r requirements.txt
```

## Download Models

Three sets of model files are needed before the API can start.

### 1. Face swap models and detection weights

The download script fetches all 13 face swapper ONNX models, 3 embedding
converters, the insightface buffalo_l face detection model, and CodeFormer
restoration weights (~5.3 GB total):

```bash
pip3 install tqdm requests
python3 scripts/download_models.py
```

This places files under:
- `checkpoints/face_swapper/` — 13 swapper models + 3 converter ONNX files
- `checkpoints/models/buffalo_l/` — insightface detection model
- `CodeFormer/CodeFormer/weights/` — codeformer.pth, retinaface, parsing, RealESRGAN

Already-downloaded files are skipped on re-run (checked by file size).

### 2. Clone the CodeFormer repository

[Git Large File Storage](
https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
must be installed before running this command:

```bash
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer
```

### 3. Verify

Start the server and check the health endpoint lists models as available:

```bash
python3 app/main.py
curl http://localhost:8090/
```

## Face Restoration

Face restoration with [CodeFormer](https://github.com/sczhou/CodeFormer) is
enabled by default (`face_restore: true`). It improves output quality at the
cost of additional processing time. RealESRGAN background upscaling is also
enabled by default (`background_enhance: true`).

## Usage

### Start the server

```bash
python3 app/main.py -H 0.0.0.0 -p 8090
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check with available models and queue depth |
| `POST` | `/faceswap` | Submit async face swap (returns 202 with job ID) |
| `GET` | `/faceswap/{job_id}/status` | Poll job status and get result |
| `POST` | `/faceswap/sync` | Synchronous face swap (blocks until complete) |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Async mode example (recommended)

```python
import requests, time

# Submit job
r = requests.post("http://localhost:8090/faceswap", json={
    "source_image": "<base64>",
    "target_image": "<base64>",
    "face_restore": True,
    "face_swapper_model": "inswapper_128",
})
job = r.json()  # {"status": "queued", "job_id": "...", "status_url": "..."}

# Poll for result
while True:
    r = requests.get(f"http://localhost:8090{job['status_url']}")
    data = r.json()
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(1)

# Save result
if data["status"] == "completed":
    image_b64 = data["result"]["image"]
```

### Sync mode example

```python
import requests

r = requests.post("http://localhost:8090/faceswap/sync", json={
    "source_image": "<base64>",
    "target_image": "<base64>",
})
result = r.json()
# result["image"] contains the base64 output
```

## Available Models

| Model | Default Resolution | Source Type |
|-------|-------------------|-------------|
| `inswapper_128` | 512x512 | embedding_projected |
| `inswapper_128_fp16` | 512x512 | embedding_projected |
| `simswap_256` | 1024x1024 | embedding |
| `simswap_unofficial_512` | 512x512 | embedding |
| `ghost_1_256` | 1024x1024 | embedding |
| `ghost_2_256` | 1024x1024 | embedding |
| `ghost_3_256` | 1024x1024 | embedding |
| `hififace_unofficial_256` | 1024x1024 | embedding |
| `hyperswap_1a_256` | 1024x1024 | embedding_norm |
| `hyperswap_1b_256` | 1024x1024 | embedding_norm |
| `hyperswap_1c_256` | 1024x1024 | embedding_norm |
| `blendswap_256` | 1024x1024 | source_face |
| `uniface_256` | 1024x1024 | source_face |

## Docker

```bash
docker build -t inswapper-api .
docker run --gpus all -p 5000:5000 inswapper-api
```

## Benchmarks

These benchmarks are for a source image with a resolution of 960x1280
and a target image with a resolution of 1200x750, upscaled by 1 and
with CodeFormer Face Restoration enabled.

| System                                      | Time Taken    |
|---------------------------------------------|---------------|
| macOS Tahoe 26.5.2 on Apple M5 Pro 16"      | 38.8 seconds  |
| Ubuntu 22.04 LTS on t3a.xlarge AWS instance | 248.1 seconds |
| Ubuntu 22.04 LTS on an A5000 Runpod GPU pod | 14.2 seconds  |
| Windows 10                                  | 103.9 seconds |

Get a [Runpod](https://runpod.io?ref=2xxro4sy) account.

## Acknowledgements

1. [FaceFusion](https://github.com/facefusion/facefusion) for the enhanced swap pipeline.
2. [insightface.ai](https://insightface.ai/) for their powerful face detection and swap models.
3. This codebase is built on top of [FaceFusion](https://github.com/facefusion/facefusion), [inswapper](https://github.com/haofanwang/inswapper), and [CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer).

#!/usr/bin/env python3
"""Download all models locally -- mirror Dockerfile steps.

Ported from runpod-worker-inswapper.
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1]).resolve()
else:
    ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
FACE_SWAPPER = CHECKPOINTS / "face_swapper"
MODELS = CHECKPOINTS / "models"
CODEFORMER = ROOT / "CodeFormer" / "CodeFormer" / "weights"

_BASE = "https://github.com/facefusion/facefusion-assets/releases/download"
MODELS_3_0_0 = f"{_BASE}/models-3.0.0"
MODELS_3_1_0 = f"{_BASE}/models-3.1.0"
MODELS_3_3_0 = f"{_BASE}/models-3.3.0"
MODELS_3_4_0 = f"{_BASE}/models-3.4.0"
CODEFORMER_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0"

DOWNLOADS = [
    # Face swapper models — models-3.0.0
    (FACE_SWAPPER, "blendswap_256.onnx", f"{MODELS_3_0_0}/blendswap_256.onnx"),
    (FACE_SWAPPER, "ghost_1_256.onnx", f"{MODELS_3_0_0}/ghost_1_256.onnx"),
    (FACE_SWAPPER, "ghost_2_256.onnx", f"{MODELS_3_0_0}/ghost_2_256.onnx"),
    (FACE_SWAPPER, "ghost_3_256.onnx", f"{MODELS_3_0_0}/ghost_3_256.onnx"),
    (FACE_SWAPPER, "inswapper_128.onnx",
     "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx?download=true"),
    (FACE_SWAPPER, "inswapper_128_fp16.onnx", f"{MODELS_3_0_0}/inswapper_128_fp16.onnx"),
    (FACE_SWAPPER, "simswap_256.onnx", f"{MODELS_3_0_0}/simswap_256.onnx"),
    (FACE_SWAPPER, "simswap_unofficial_512.onnx", f"{MODELS_3_0_0}/simswap_unofficial_512.onnx"),
    (FACE_SWAPPER, "uniface_256.onnx", f"{MODELS_3_0_0}/uniface_256.onnx"),
    # hififace — models-3.1.0
    (FACE_SWAPPER, "hififace_unofficial_256.onnx", f"{MODELS_3_1_0}/hififace_unofficial_256.onnx"),
    # hyperswap — models-3.3.0
    (FACE_SWAPPER, "hyperswap_1a_256.onnx", f"{MODELS_3_3_0}/hyperswap_1a_256.onnx"),
    (FACE_SWAPPER, "hyperswap_1b_256.onnx", f"{MODELS_3_3_0}/hyperswap_1b_256.onnx"),
    (FACE_SWAPPER, "hyperswap_1c_256.onnx", f"{MODELS_3_3_0}/hyperswap_1c_256.onnx"),
    # Embedding converters — models-3.4.0
    (FACE_SWAPPER, "crossface_ghost.onnx", f"{MODELS_3_4_0}/crossface_ghost.onnx"),
    (FACE_SWAPPER, "crossface_hififace.onnx", f"{MODELS_3_4_0}/crossface_hififace.onnx"),
    (FACE_SWAPPER, "crossface_simswap.onnx", f"{MODELS_3_4_0}/crossface_simswap.onnx"),
    # Insightface face detection
    (MODELS, "buffalo_l.zip",
     "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"),
    # CodeFormer weights
    (CODEFORMER / "CodeFormer", "codeformer.pth", f"{CODEFORMER_URL}/codeformer.pth"),
    (CODEFORMER / "facelib", "detection_Resnet50_Final.pth", f"{CODEFORMER_URL}/detection_Resnet50_Final.pth"),
    (CODEFORMER / "facelib", "parsing_parsenet.pth", f"{CODEFORMER_URL}/parsing_parsenet.pth"),
    (CODEFORMER / "realesrgan", "RealESRGAN_x2plus.pth", f"{CODEFORMER_URL}/RealESRGAN_x2plus.pth"),
]


def _download(dest_dir: Path, filename: str, url: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    with requests.head(url, allow_redirects=True, timeout=30) as head:
        head.raise_for_status()
        expected_size = int(head.headers.get("content-length", 0))

    if dest_path.exists() and expected_size and dest_path.stat().st_size == expected_size:
        tqdm.write(f"✓ {filename}")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with (
        open(dest_path, "wb") as f,
        tqdm(
            total=expected_size, unit="B", unit_scale=True, unit_divisor=1024,
            desc=filename, ascii=" =",
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def _extract_zip(zip_path: Path) -> None:
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists() and any(extract_dir.iterdir()):
        tqdm.write("✓ buffalo_l already extracted")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("done")


def main() -> None:
    print(f"Downloading models to {CHECKPOINTS}\n")
    for dest_dir, filename, url in DOWNLOADS:
        try:
            _download(dest_dir, filename, url)
        except Exception as e:
            tqdm.write(f"✗ {filename}: {e}")

    zip_path = MODELS / "buffalo_l.zip"
    if zip_path.exists():
        _extract_zip(zip_path)

    total = sum(
        (dest_dir / filename).stat().st_size
        for dest_dir, filename, _ in DOWNLOADS
        if (dest_dir / filename).exists()
    )
    print(f"\nDone. Total: {total / (1024 ** 3):.1f} GB")


if __name__ == "__main__":
    main()

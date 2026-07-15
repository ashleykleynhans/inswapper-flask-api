#!/usr/bin/env python3
"""Example client for the InSwapper FastAPI.

Demonstrates both async (queue) and sync modes.
"""

import io
import json
import time
import uuid
import requests
import base64
from PIL import Image

URL = "http://127.0.0.1:8090"

SOURCE_IMAGE = "../data/src.jpg"
TARGET_IMAGE = "../data/target.jpg"
SOURCE_INDEXES = "-1"
TARGET_INDEXES = "-1"
BACKGROUND_ENHANCE = True
FACE_RESTORE = True
FACE_UPSAMPLE = True
UPSCALE = 1
CODEFORMER_FIDELITY = 0.5
OUTPUT_FORMAT = "JPEG"

# Optional: use a different model
# FACE_SWAPPER_MODEL = "simswap_256"
FACE_SWAPPER_MODEL = "inswapper_128"


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and encode to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_result_image(image_b64: str, prefix: str = "result") -> str:
    """Decode base64 image and save to disk."""
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    output_file = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    img.save(output_file, format="JPEG")
    return output_file


def async_mode():
    """Submit a job and poll for the result."""
    print("=== Async mode ===")
    payload = {
        "source_image": encode_image_to_base64(SOURCE_IMAGE),
        "target_image": encode_image_to_base64(TARGET_IMAGE),
        "source_indexes": SOURCE_INDEXES,
        "target_indexes": TARGET_INDEXES,
        "background_enhance": BACKGROUND_ENHANCE,
        "face_restore": FACE_RESTORE,
        "face_upsample": FACE_UPSAMPLE,
        "upscale": UPSCALE,
        "codeformer_fidelity": CODEFORMER_FIDELITY,
        "output_format": OUTPUT_FORMAT,
        "face_swapper_model": FACE_SWAPPER_MODEL,
    }

    # Submit job
    r = requests.post(f"{URL}/faceswap", json=payload)
    print(f"HTTP status: {r.status_code}")
    resp = r.json()
    print(f"Queued: {resp}")

    if r.status_code != 202:
        return

    # Poll for completion
    status_url = resp["status_url"]
    while True:
        r = requests.get(f"{URL}{status_url}")
        data = r.json()
        status = data["status"]
        print(f"  Job status: {status}")

        if status == "completed":
            output = save_result_image(data["result"]["image"], "async")
            print(f"Saved: {output}")
            break
        elif status == "failed":
            print(f"Failed: {data.get('error', 'unknown error')}")
            break

        time.sleep(1)


def sync_mode():
    """Submit a synchronous face swap request."""
    print("=== Sync mode ===")
    payload = {
        "source_image": encode_image_to_base64(SOURCE_IMAGE),
        "target_image": encode_image_to_base64(TARGET_IMAGE),
        "source_indexes": SOURCE_INDEXES,
        "target_indexes": TARGET_INDEXES,
        "background_enhance": BACKGROUND_ENHANCE,
        "face_restore": FACE_RESTORE,
        "face_upsample": FACE_UPSAMPLE,
        "upscale": UPSCALE,
        "codeformer_fidelity": CODEFORMER_FIDELITY,
        "output_format": OUTPUT_FORMAT,
        "face_swapper_model": FACE_SWAPPER_MODEL,
    }

    r = requests.post(f"{URL}/faceswap/sync", json=payload)
    print(f"HTTP status: {r.status_code}")
    resp = r.json()

    if resp["status"] == "ok":
        output = save_result_image(resp["image"], "sync")
        print(f"Saved: {output}")
    else:
        print(f"Error: {resp.get('msg', 'unknown')}")
        print(f"Detail: {resp.get('detail', '')}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "async"
    if mode == "sync":
        sync_mode()
    else:
        async_mode()

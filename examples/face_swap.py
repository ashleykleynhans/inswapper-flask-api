#!/usr/bin/env python3
"""Example client for the FaceSwap API.

Demonstrates both async (queue) and sync modes with elapsed time tracking,
model selection, face selector options, mask controls, and identity blending.
"""

import io
import json
import os
import sys
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

# Model: pick one from the 13 available models
FACE_SWAPPER_MODEL = "inswapper_128"
# FACE_SWAPPER_MODEL = "simswap_256"
# FACE_SWAPPER_MODEL = "ghost_1_256"
# FACE_SWAPPER_MODEL = "blendswap_256"
# FACE_SWAPPER_MODEL = "hififace_unofficial_256"

# Resolution override (None = auto-detect default per model)
FACE_SWAPPER_RESOLUTION = None
# FACE_SWAPPER_RESOLUTION = "1024x1024"  # higher quality, slower

# Identity blend (0.0 = more target identity, 1.0 = more source identity)
FACE_SWAPPER_WEIGHT = 1.0

# Mask controls
FACE_MASK_BLUR = 0.3       # edge softness (0.0 = sharp, 1.0 = very blurry)
FACE_MASK_PADDING = "0,0,0,0"  # top,right,bottom,left percentage inset

# Face selector (target face filtering)
FACE_SELECTOR_MODE = "many"   # "many" = all matching, "one" = best match
FACE_SELECTOR_ORDER = "left-right"  # sort order
# FACE_SELECTOR_ORDER = "best-worst"
# FACE_SELECTOR_GENDER = "female"
# FACE_SELECTOR_AGE_START = 20
# FACE_SELECTOR_AGE_END = 50


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


def build_payload():
    """Build the request payload from module-level settings."""
    return {
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
        "face_swapper_resolution": FACE_SWAPPER_RESOLUTION,
        "face_swapper_weight": FACE_SWAPPER_WEIGHT,
        "face_mask_blur": FACE_MASK_BLUR,
        "face_mask_padding": FACE_MASK_PADDING,
        "face_selector_mode": FACE_SELECTOR_MODE,
        "face_selector_order": FACE_SELECTOR_ORDER,
    }


def async_mode():
    """Submit a job and poll for the result."""
    print("=== Async mode ===")
    print(f"Model: {FACE_SWAPPER_MODEL}  Weight: {FACE_SWAPPER_WEIGHT}  "
          f"Selector: {FACE_SELECTOR_MODE}/{FACE_SELECTOR_ORDER}")

    payload = build_payload()

    # Submit job
    start_time = time.time()
    r = requests.post(f"{URL}/faceswap", json=payload)
    resp = r.json()

    if r.status_code != 202:
        print(f"Error ({r.status_code}): {resp}")
        return

    print(f"Job ID: {resp['job_id']}")

    # Poll for completion
    status_url = resp["status_url"]
    while True:
        r = requests.get(f"{URL}{status_url}")
        data = r.json()
        status = data["status"]
        elapsed = time.time() - start_time
        print(f"  {status} ({elapsed:.1f}s)")

        if status == "completed":
            output = save_result_image(data["result"]["image"], "async")
            print(f"Saved: {output}")
            print(f"Total time: {elapsed:.1f} seconds")
            break
        elif status == "failed":
            print(f"Failed: {data.get('error', 'unknown error')}")
            break

        time.sleep(1)


def sync_mode():
    """Submit a synchronous face swap request."""
    print("=== Sync mode ===")
    print(f"Model: {FACE_SWAPPER_MODEL}  Weight: {FACE_SWAPPER_WEIGHT}  "
          f"Selector: {FACE_SELECTOR_MODE}/{FACE_SELECTOR_ORDER}")

    payload = build_payload()

    start_time = time.time()
    r = requests.post(f"{URL}/faceswap/sync", json=payload)
    elapsed = time.time() - start_time
    resp = r.json()

    if resp["status"] == "ok":
        output = save_result_image(resp["image"], "sync")
        print(f"Saved: {output}")
        print(f"Total time: {elapsed:.1f} seconds")
    else:
        print(f"Error: {resp.get('msg', 'unknown')}")
        print(f"Detail: {resp.get('detail', '')}")


def compare_models(models=None):
    """Run the same swap across multiple models for comparison."""
    if models is None:
        models = [
            "inswapper_128",
            "simswap_256",
            "blendswap_256",
            "ghost_1_256",
        ]

    print("=== Model Comparison ===")
    print(f"Models: {', '.join(models)}")
    print()

    results = {}
    for model in models:
        print(f"--- {model} ---")
        payload = build_payload()
        payload["face_swapper_model"] = model

        start = time.time()
        r = requests.post(f"{URL}/faceswap", json=payload)
        if r.status_code != 202:
            print(f"  Failed to submit: {r.status_code}")
            results[model] = {"status": "error", "time": time.time() - start}
            continue

        job = r.json()
        status_url = job["status_url"]

        while True:
            r = requests.get(f"{URL}{status_url}")
            data = r.json()
            if data["status"] == "completed":
                elapsed = time.time() - start
                results[model] = {"status": "ok", "time": elapsed}
                output = save_result_image(
                    data["result"]["image"], f"compare_{model}",
                )
                print(f"  Saved: {output}")
                print(f"  Time: {elapsed:.1f}s")
                break
            elif data["status"] == "failed":
                elapsed = time.time() - start
                results[model] = {"status": "failed", "time": elapsed}
                print(f"  Failed: {data.get('error', 'unknown')}")
                break
            time.sleep(1)

    # Summary
    print("\n=== Results ===")
    print(f"{'Model':<30} {'Status':<10} {'Time':>8}")
    print("-" * 50)
    for model, info in results.items():
        print(f"{model:<30} {info['status']:<10} {info['time']:>7.1f}s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "async"

    if mode == "sync":
        sync_mode()
    elif mode == "compare":
        models = sys.argv[2:] if len(sys.argv) > 2 else None
        compare_models(models)
    else:
        async_mode()

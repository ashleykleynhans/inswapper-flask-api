"""Image encoding/decoding, file extension detection, temp file I/O."""

import os
import io
import base64
import uuid
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import TMP_PATH


def determine_file_extension(image_data: str) -> str:
    """Detect image format from base64 header bytes.

    Args:
        image_data: Base64-encoded image string.

    Returns:
        File extension including dot ('.jpg' or '.png').
    """
    try:
        if image_data.startswith("/9j/"):
            return ".jpg"
        elif image_data.startswith("iVBORw0Kg"):
            return ".png"
        else:
            return ".png"
    except Exception:
        return ".png"


def decode_base64_to_disk(b64_data: str, prefix: str = "source") -> str:
    """Decode a base64 image and write it to a temp file.

    Args:
        b64_data: Base64-encoded image data.
        prefix: Filename prefix ('source' or 'target').

    Returns:
        Absolute path to the written temp file.
    """
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)  # pragma: no cover — tmp dir created once

    unique_id = uuid.uuid4()
    ext = determine_file_extension(b64_data)
    file_path = os.path.join(TMP_PATH, f"{prefix}_{unique_id}{ext}")

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))

    return file_path


def encode_image_to_base64(image: Image.Image, output_format: str = "JPEG") -> str:
    """Encode a PIL Image as a base64 string.

    Args:
        image: PIL Image to encode.
        output_format: Image format ('JPEG' or 'PNG').

    Returns:
        Base64-encoded string.
    """
    output_buffer = io.BytesIO()
    image.save(output_buffer, format=output_format)
    image_data = output_buffer.getvalue()
    return base64.b64encode(image_data).decode("utf-8")


def encode_bgr_to_base64(bgr_image: np.ndarray, output_format: str = "JPEG") -> str:
    """Encode an OpenCV BGR image array as a base64 string.

    Args:
        bgr_image: OpenCV BGR image (numpy array).
        output_format: Image format ('JPEG' or 'PNG').

    Returns:
        Base64-encoded string.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return encode_image_to_base64(pil_img, output_format)


def load_images_from_paths(
    src_img_path: str, target_img_path: str
) -> Tuple[List[Image.Image], Image.Image]:
    """Load source (possibly multiple) and target images from disk.

    Source path may contain semicolon-separated paths for multiple source images.

    Args:
        src_img_path: Semicolon-separated source image paths.
        target_img_path: Single target image path.

    Returns:
        Tuple of (list of source PIL Images, target PIL Image).
    """
    source_paths = src_img_path.split(";")
    source_images = [Image.open(p) for p in source_paths]
    target_img = Image.open(target_img_path)
    return source_images, target_img


def clean_up_temporary_files(*paths: str) -> None:
    """Remove temporary files, silently ignoring any errors.

    Args:
        paths: One or more file paths to remove.
    """
    for path in paths:
        if path:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:  # pragma: no cover
                pass

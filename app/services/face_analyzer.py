"""Face detection helpers using insightface buffalo_l."""

import logging
from typing import List, Optional

import numpy as np

import insightface

from app.config import CHECKPOINTS_DIR, DEFAULT_DET_SIZE

logger = logging.getLogger(__name__)

# Singleton cache — initialized at startup
FACE_ANALYSER: Optional[insightface.app.FaceAnalysis] = None


def get_face_analyser(
    torch_device: str = "cpu",
    det_size: tuple = DEFAULT_DET_SIZE,
) -> insightface.app.FaceAnalysis:
    """Create (or return cached) insightface FaceAnalysis instance.

    Args:
        torch_device: Device string ('cuda' or 'cpu').
        det_size: Detection size as (width, height) tuple.

    Returns:
        Configured FaceAnalysis instance.
    """
    global FACE_ANALYSER

    if FACE_ANALYSER is not None:
        return FACE_ANALYSER

    providers = (
        ["CUDAExecutionProvider"] if torch_device == "cuda"
        else ["CPUExecutionProvider"]
    )

    logger.info(
        "Initializing face analyser on %s with providers: %s",
        torch_device.upper(), providers,
    )

    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(CHECKPOINTS_DIR),
        providers=providers,
    )
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    FACE_ANALYSER = face_analyser
    return FACE_ANALYSER


def get_one_face(
    face_analyser: insightface.app.FaceAnalysis,
    frame: np.ndarray,
) -> Optional[object]:
    """Return the leftmost face in the frame.

    Args:
        face_analyser: Initialized FaceAnalysis instance.
        frame: BGR image as numpy array.

    Returns:
        Leftmost face object, or None if no faces detected.
    """
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(
    face_analyser: insightface.app.FaceAnalysis,
    frame: np.ndarray,
    min_face_size: float = 0.0,
) -> Optional[List[object]]:
    """Return faces sorted left-to-right, optionally filtered by minimum size.

    Args:
        face_analyser: Initialized FaceAnalysis instance.
        frame: BGR image as numpy array.
        min_face_size: Minimum face size as percentage of image dimension (0-100).

    Returns:
        List of face objects sorted left-to-right, or None.
    """
    try:
        face = face_analyser.get(frame)
        if min_face_size > 0:
            img_height, img_width = frame.shape[:2]
            min_dimension = min(img_width, img_height)
            min_pixels = min_dimension * (min_face_size / 100.0)
            face = [
                f for f in face
                if (f.bbox[2] - f.bbox[0]) >= min_pixels
                or (f.bbox[3] - f.bbox[1]) >= min_pixels
            ]
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

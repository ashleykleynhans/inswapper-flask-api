"""Centralized configuration for the FaceSwap API."""

import os
import sys
import logging
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
FACE_SWAPPER_MODELS_DIR = CHECKPOINTS_DIR / "face_swapper"
MODELS_DIR = CHECKPOINTS_DIR / "models"
CODEFORMER_DIR = BASE_DIR / "CodeFormer"
CODEFORMER_CODE_DIR = CODEFORMER_DIR / "CodeFormer"
TMP_PATH = "/tmp/inswapper"

# Default face swap model (insightface ModelRouter legacy path)
LEGACY_MODEL_PATH = str(CHECKPOINTS_DIR / "face_swapper" / "inswapper_128.onnx")

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "inswapper_128"
DEFAULT_DET_SIZE = (320, 320)
DEFAULT_PORT = 8090
DEFAULT_HOST = "0.0.0.0"
QUEUE_TIMEOUT_SECONDS = 600
JOB_RESULT_TTL_SECONDS = 3600

# Defaults matching the runpod INPUT_SCHEMA
DEFAULTS = {
    "source_indexes": "-1",
    "target_indexes": "-1",
    "background_enhance": True,
    "face_restore": True,
    "face_upsample": True,
    "upscale": 1,
    "codeformer_fidelity": 0.5,
    "output_format": "JPEG",
    "min_face_size": 0.0,
    "face_swapper_model": "inswapper_128",
    "face_swapper_resolution": None,
    "face_swapper_weight": 1.0,
    "face_mask_blur": 0.3,
    "face_mask_padding": "0,0,0,0",
    "face_selector_mode": "many",
    "face_selector_order": "left-right",
    "face_selector_gender": None,
    "face_selector_age_start": None,
    "face_selector_age_end": None,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG

# Mac does not have permission to /var/log
if sys.platform == "linux":
    LOG_PATH = "/var/log/"  # pragma: no cover — Linux-only
else:
    LOG_PATH = ""

LOG_FILE = os.path.join(LOG_PATH, "inswapper.log")


def init_logging() -> None:
    """Configure logging to file and stdout."""
    logging.basicConfig(
        filename=LOG_FILE if LOG_PATH else None,
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=LOG_LEVEL,
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------
class Timer:
    """Simple elapsed-time timer."""

    def __init__(self) -> None:
        self.start = time.time()

    def restart(self) -> None:
        """Reset the timer."""
        self.start = time.time()

    def get_elapsed_time(self) -> float:
        """Return elapsed seconds, rounded to 1 decimal place."""
        end = time.time()
        return round(end - self.start, 1)

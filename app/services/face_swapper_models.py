"""Face swapper model definitions and validation logic.

Ported from runpod-worker-inswapper.
"""

from typing import Dict, List, Tuple

# Model compatibility matrix: model name -> supported resolutions
FACE_SWAPPER_MODEL_SET: Dict[str, List[str]] = {
    "blendswap_256": ["256x256", "384x384", "512x512", "768x768", "1024x1024"],
    "ghost_1_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "ghost_2_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "ghost_3_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hififace_unofficial_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hyperswap_1a_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hyperswap_1b_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hyperswap_1c_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "inswapper_128": ["128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
    "inswapper_128_fp16": ["128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
    "simswap_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "simswap_unofficial_512": ["512x512", "768x768", "1024x1024"],
    "uniface_256": ["256x256", "512x512", "768x768", "1024x1024"],
}

DEFAULT_RESOLUTIONS: Dict[str, str] = {
    "inswapper_128": "512x512",
    "inswapper_128_fp16": "512x512",
    "simswap_unofficial_512": "512x512",
    "default": "1024x1024",
}

MODEL_METADATA: Dict[str, dict] = {
    "blendswap_256": {
        "native_size": (256, 256),
        "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
        "tanh_out": False,
        "source_type": "source_face",
        "source_size": 112,
        "warp_template": "ffhq_512",
    },
    "ghost_1_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding",
        "converter": "crossface_ghost.onnx",
        "warp_template": "arcface_112_v1",
    },
    "ghost_2_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding",
        "converter": "crossface_ghost.onnx",
        "warp_template": "arcface_112_v1",
    },
    "ghost_3_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding",
        "converter": "crossface_ghost.onnx",
        "warp_template": "arcface_112_v1",
    },
    "hififace_unofficial_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding",
        "converter": "crossface_hififace.onnx",
        "warp_template": "mtcnn_512",
    },
    "hyperswap_1a_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding_norm",
        "warp_template": "arcface_128",
    },
    "hyperswap_1b_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding_norm",
        "warp_template": "arcface_128",
    },
    "hyperswap_1c_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "embedding_norm",
        "warp_template": "arcface_128",
    },
    "inswapper_128": {
        "native_size": (128, 128),
        "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
        "tanh_out": False,
        "source_type": "embedding_projected",
        "warp_template": "arcface_128",
    },
    "inswapper_128_fp16": {
        "native_size": (128, 128),
        "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
        "tanh_out": False,
        "source_type": "embedding_projected",
        "warp_template": "arcface_128",
    },
    "simswap_256": {
        "native_size": (256, 256),
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "tanh_out": False,
        "source_type": "embedding",
        "converter": "crossface_simswap.onnx",
        "warp_template": "arcface_128",
    },
    "simswap_unofficial_512": {
        "native_size": (512, 512),
        "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
        "tanh_out": False,
        "source_type": "embedding",
        "converter": "crossface_simswap.onnx",
        "warp_template": "arcface_128",
    },
    "uniface_256": {
        "native_size": (256, 256),
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "tanh_out": True,
        "source_type": "source_face",
        "source_size": 256,
        "warp_template": "ffhq_512",
    },
}


def validate_face_swapper_params(model_name: str, resolution: str) -> None:
    """Validate model and resolution compatibility.

    Args:
        model_name: Face swapper model name.
        resolution: Resolution string (e.g., '512x512').

    Raises:
        ValueError: If model name or resolution is invalid.
    """
    if model_name not in FACE_SWAPPER_MODEL_SET:
        valid_models = ", ".join(sorted(FACE_SWAPPER_MODEL_SET.keys()))
        raise ValueError(
            f"Invalid face_swapper_model: '{model_name}'. "
            f"Valid: {valid_models}"
        )
    if resolution not in FACE_SWAPPER_MODEL_SET[model_name]:
        valid = ", ".join(FACE_SWAPPER_MODEL_SET[model_name])
        raise ValueError(
            f"Model '{model_name}' does not support resolution '{resolution}'. "
            f"Valid: {valid}"
        )


def get_default_resolution(model_name: str) -> str:
    """Get default resolution for a model.

    Args:
        model_name: Face swapper model name.

    Returns:
        Default resolution string.
    """
    return DEFAULT_RESOLUTIONS.get(model_name, DEFAULT_RESOLUTIONS["default"])


def parse_resolution(resolution: str) -> Tuple[int, int]:
    """Parse a resolution string like '512x512' into an (int, int) tuple.

    Args:
        resolution: Resolution string in WIDTHxHEIGHT format.

    Returns:
        Tuple of (width, height).

    Raises:
        ValueError: If format is invalid.
    """
    try:
        parts = resolution.lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid resolution format: '{resolution}'. "
                f"Expected: 'WIDTHxHEIGHT' (e.g. '512x512')"
            )
        return int(parts[0]), int(parts[1])
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid resolution format: '{resolution}'. "
            f"Expected: 'WIDTHxHEIGHT' (e.g. '512x512')"
        ) from e


def get_model_metadata(model_name: str) -> dict:
    """Get preprocessing metadata for a face swapper model.

    Args:
        model_name: Face swapper model name.

    Returns:
        Dictionary with keys: native_size, mean, std, tanh_out, source_type,
        and optionally converter, warp_template, source_size.
    """
    if model_name not in MODEL_METADATA:
        raise KeyError(f"Model metadata not found for: '{model_name}'")
    return MODEL_METADATA[model_name]

"""Face swapping -- forks FaceFusion's pipeline.

Per-model warp templates, estimateAffinePartial2D + RANSAC, box-mask
paste-back, embedding converters, and per-model source preparation.

Ported from runpod-worker-inswapper.
"""

import os
import logging
from typing import Dict, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
from insightface.utils import face_align

from app.services.face_swapper_models import get_model_metadata

logger = logging.getLogger(__name__)

FACE_SWAPPER_MODELS: Dict[str, object] = {}
EMBEDDING_CONVERTERS: Dict[str, object] = {}

WARP_TEMPLATES = {
    "arcface_112_v1": np.array([
        [0.35473214, 0.45658929], [0.64526786, 0.45658929],
        [0.50000000, 0.61154464], [0.37913393, 0.77687500],
        [0.62086607, 0.77687500],
    ]),
    "arcface_128": np.array([
        [0.36167656, 0.40387734], [0.63696719, 0.40235469],
        [0.50019687, 0.56044219], [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ]),
    "ffhq_512": np.array([
        [0.37691676, 0.46864664], [0.62285697, 0.46912813],
        [0.50123859, 0.61331904], [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ]),
    "mtcnn_512": np.array([
        [0.36562865, 0.46733799], [0.63305391, 0.46585885],
        [0.50019127, 0.61942959], [0.39032951, 0.77598822],
        [0.61178945, 0.77476328],
    ]),
}


class _SwapperModel:
    """ONNX face-swapper session + metadata."""

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, None)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) < 2:
            raise ValueError(
                f"Model '{model_path}' has {len(inputs)} input(s), "
                f"expected at least 2"
            )
        self.input_names = [i.name for i in inputs]
        self.output_names = [o.name for o in outputs]
        if not self.output_names:
            raise ValueError(f"Model '{model_path}' has no outputs")
        input_shape = inputs[0].shape
        self.input_size = tuple(input_shape[2:4][::-1])

        # Hyperswap: reversed inputs (embedding first [1,512], image second [1,3,H,W])
        s0 = inputs[0].shape
        s1 = inputs[1].shape
        self.input_swapped = (
            len(s0) == 2 and s0[1] == 512
            and len(s1) >= 3 and s1[1] == 3
        )

        try:
            om = onnx.load(model_path)
            self.emap = (
                numpy_helper.to_array(om.graph.initializer[-1])
                if om.graph.initializer else np.eye(1)
            )
        except Exception:  # pragma: no cover — ONNX model load failure
            self.emap = np.eye(1)

        logger.info(
            "Swapper: %s size=%s swapped=%s",
            os.path.basename(model_path), self.input_size, self.input_swapped,
        )


def _load_embedding_converter(name: str) -> onnxruntime.InferenceSession:
    """Load (or retrieve cached) embedding converter ONNX model."""
    if name not in EMBEDDING_CONVERTERS:
        path = f"checkpoints/face_swapper/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Converter not found: {path}")
        EMBEDDING_CONVERTERS[name] = onnxruntime.InferenceSession(path, None)
    return EMBEDDING_CONVERTERS[name]


def get_face_swapper_model(model_name: str) -> _SwapperModel:
    """Load (or retrieve cached) face swapper ONNX model."""
    if model_name not in FACE_SWAPPER_MODELS:
        path = f"checkpoints/face_swapper/{model_name}.onnx"
        logger.info("Loading model: %s", model_name)
        FACE_SWAPPER_MODELS[model_name] = _SwapperModel(path)
    return FACE_SWAPPER_MODELS[model_name]


# ---------------------------------------------------------------------------
# Warp (FaceFusion: cv2.estimateAffinePartial2D + RANSAC)
# ---------------------------------------------------------------------------


def _warp_face_by_landmark_5(
    frame: np.ndarray,
    landmarks: np.ndarray,
    template: str,
    crop_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp face using 5-point landmarks and affine transform.

    Falls back to insightface's norm_crop2 if RANSAC fails.
    """
    tmpl = WARP_TEMPLATES[template] * np.array(crop_size)
    M = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), tmpl.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=100,
    )[0]
    if M is None:  # pragma: no cover — RANSAC rarely fails
        return face_align.norm_crop2(frame, landmarks, crop_size[0]), np.eye(2, 3)
    return cv2.warpAffine(
        frame, M, crop_size,
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA,
    ), M


# ---------------------------------------------------------------------------
# Paste-back (FaceFusion: box mask + alpha blend)
# ---------------------------------------------------------------------------


def _create_box_mask(
    size: Tuple[int, int],
    blur: float = 0.3,
    pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Create a soft-edged box mask for paste-back blending."""
    w, h = size
    ba = max(int(w * 0.5 * blur) // 2, 1)
    mask = np.ones((h, w), dtype=np.float32)
    mask[:max(ba, int(h * pad[0] / 100)), :] = 0
    mask[-max(ba, int(h * pad[2] / 100)):, :] = 0
    mask[:, :max(ba, int(w * pad[3] / 100))] = 0
    mask[:, -max(ba, int(w * pad[1] / 100)):] = 0
    if int(w * 0.5 * blur) > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), int(w * 0.5 * blur) * 0.25)
    return mask


def _transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply affine transform to points."""
    return cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)


def _calculate_paste_area(
    frame: np.ndarray, crop_frame: np.ndarray, M: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate paste bounding box and projection matrix."""
    th, tw = frame.shape[:2]
    ch, cw = crop_frame.shape[:2]
    inv = cv2.invertAffineTransform(M)
    pts = _transform_points(
        np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]]), inv,
    )
    x1, y1 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [tw, th])
    x2, y2 = np.clip(np.ceil(pts.max(axis=0)).astype(int), 0, [tw, th])
    Pm = inv.copy()
    Pm[0, 2] -= x1
    Pm[1, 2] -= y1
    return np.array([x1, y1, x2, y2]), Pm


def _paste_back(
    frame: np.ndarray,
    crop_frame: np.ndarray,
    crop_mask: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Paste swapped face back into original frame with alpha blending."""
    bbox, Pm = _calculate_paste_area(frame, crop_frame, M)
    x1, y1, x2, y2 = bbox
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:  # pragma: no cover — defensive
        return frame
    im = np.expand_dims(
        cv2.warpAffine(crop_mask, Pm, (pw, ph)).clip(0, 1), -1,
    )
    iv = cv2.warpAffine(
        crop_frame, Pm, (pw, ph), borderMode=cv2.BORDER_REPLICATE,
    )
    out = frame.copy()
    r = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = (r * (1 - im) + iv * im).astype(out.dtype)
    return out


# ---------------------------------------------------------------------------
# Source preparation (FaceFusion: per-model family)
# ---------------------------------------------------------------------------


def _prepare_embedding_projected(
    source_face: object, model: _SwapperModel,
) -> np.ndarray:
    """inswapper: dot(raw embedding, emap) / ||raw embedding||."""
    e = source_face.embedding.reshape((1, -1))
    return np.dot(e, model.emap) / np.linalg.norm(e)


def _prepare_embedding_raw(
    source_face: object, converter_name: str,
) -> np.ndarray:
    """simswap/ghost/hififace: raw embedding -> crossface ONNX -> L2-norm."""
    e = source_face.embedding.reshape((-1, 512))
    c = _load_embedding_converter(converter_name).run(None, {"input": e})[0].ravel()
    return (c / np.linalg.norm(c)).reshape(1, -1)


def _prepare_embedding_norm(source_face: object) -> np.ndarray:
    """hyperswap: use L2-normalized embedding (insightface normed_embedding)."""
    return source_face.normed_embedding.reshape((1, -1))


def _prepare_source_face(
    source_face: object, frame: np.ndarray, source_size: int,
) -> np.ndarray:
    """blendswap/uniface: warp source face to template, no mean/std."""
    src, _ = face_align.norm_crop2(frame, source_face.kps, source_size)
    b = src[:, :, ::-1].astype(np.float32) / 255.0
    return np.expand_dims(b.transpose(2, 0, 1), axis=0).astype(np.float32)


def _balance_embedding(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    weight: float,
    l2_norm_target: bool = True,
) -> np.ndarray:
    """FaceFusion: interpolate source/target identities.

    For hififace, hyperswap, inswapper, simswap: target is L2-normalized.
    For ghost: target is NOT L2-normalized (raw embedding used as-is).
    """
    w = np.interp(weight, [0, 1], [0.35, -0.35]).astype(np.float32)
    tgt = tgt_emb.reshape((1, -1))
    if l2_norm_target:
        n = np.linalg.norm(tgt)
        if n > 0:
            tgt = tgt / n
    return src_emb.reshape((1, -1)) * (1 - w) + tgt * w


# ---------------------------------------------------------------------------
# Target crop preprocessing
# ---------------------------------------------------------------------------


def _prepare_crop_frame(
    frame: np.ndarray, mean: list, std: list,
) -> np.ndarray:
    """Normalize a crop frame for ONNX inference."""
    mn = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    sd = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    f = frame[:, :, ::-1].astype(np.float32) / 255.0
    f = (f - mn) / sd
    return np.expand_dims(f.transpose(2, 0, 1), axis=0).astype(np.float32)


def _normalize_crop_frame(
    frame: np.ndarray, mean: list, std: list, tanh_out: bool,
) -> np.ndarray:
    """Denormalize ONNX output back to uint8 BGR."""
    f = frame[0].transpose(1, 2, 0)
    if tanh_out:
        mn = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        sd = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        f = f * sd + mn
    return (np.clip(f, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def swap_face_enhanced(
    source_face: object,
    target_face: object,
    temp_frame: np.ndarray,
    model: _SwapperModel,
    model_name: str,
    resolution: Tuple[int, int],
    weight: float = 1.0,
    face_mask_blur: float = 0.3,
    face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Enhanced face swap with per-model preprocessing and identity blending.

    Args:
        source_face: insightface Face object for the source identity.
        target_face: insightface Face object for the target face to replace.
        temp_frame: BGR image frame (numpy array).
        model: Loaded _SwapperModel instance.
        model_name: Model name string (e.g., 'inswapper_128').
        resolution: (width, height) tuple for output crop resolution.
        weight: Identity blend weight (0.0-1.0).
        face_mask_blur: Edge softness for mask (0.0-1.0).
        face_mask_padding: Inset as (top, right, bottom, left) percentages.

    Returns:
        BGR image with the swapped face pasted back in.

    Raises:
        TypeError: If model is not a valid _SwapperModel.
        RuntimeError: If ONNX inference fails.
    """
    if not hasattr(model, "session"):
        raise TypeError(f"Expected swapper model, got {type(model).__name__}")

    meta = get_model_metadata(model_name)
    native_size = meta["native_size"]
    mean, std, tanh_out = meta["mean"], meta["std"], meta["tanh_out"]
    source_type = meta["source_type"]
    warp_template = meta.get("warp_template", "arcface_128")
    target_res = resolution[0]

    # --- Target: warp, box mask, resize, normalize ---
    crop_size = (target_res, target_res)
    aimg, M = _warp_face_by_landmark_5(
        temp_frame, target_face.kps, warp_template, crop_size,
    )
    crop_mask = _create_box_mask(
        crop_size, blur=face_mask_blur, pad=face_mask_padding,
    )
    aimg_resized = cv2.resize(aimg, native_size)
    target_blob = _prepare_crop_frame(aimg_resized, mean, std)

    # Ghost models use raw (unnormalized) target embedding for balancing
    l2_norm = not model_name.startswith("ghost_")

    # --- Source: depends on model family ---
    if source_type == "embedding_projected":
        source_input = _prepare_embedding_projected(source_face, model)
        source_input = _balance_embedding(
            source_input, target_face.embedding, weight,
            l2_norm_target=l2_norm,
        )
    elif source_type == "embedding":
        source_input = _prepare_embedding_raw(source_face, meta["converter"])
        source_input = _balance_embedding(
            source_input, target_face.embedding, weight,
            l2_norm_target=l2_norm,
        )
    elif source_type == "embedding_norm":
        source_input = _prepare_embedding_norm(source_face)
        source_input = _balance_embedding(
            source_input, target_face.embedding, weight,
            l2_norm_target=l2_norm,
        )
    elif source_type == "source_face":
        source_input = _prepare_source_face(
            source_face, temp_frame, meta["source_size"],
        )
    else:
        raise ValueError(f"Unknown source_type: '{source_type}'")

    # --- ONNX inference (handle hyperswap's reversed inputs) ---
    if model.input_swapped:  # pragma: no cover — hyperswap models
        fd = {model.input_names[0]: source_input, model.input_names[1]: target_blob}
    else:
        fd = {model.input_names[0]: target_blob, model.input_names[1]: source_input}

    try:
        pred = model.session.run(model.output_names, fd)[0]
    except Exception as e:
        raise RuntimeError(
            f"ONNX inference failed for '{model_name}': {e}",
        ) from e

    # --- Post-process ---
    bgr_fake_native = _normalize_crop_frame(pred, mean, std, tanh_out)
    bgr_fake = cv2.resize(bgr_fake_native, (target_res, target_res))
    result = _paste_back(temp_frame, bgr_fake, crop_mask, M)

    # Weight blend for source_face models
    if source_type == "source_face" and weight < 1.0:
        result = cv2.addWeighted(temp_frame, 1.0 - weight, result, weight, 0)

    return result

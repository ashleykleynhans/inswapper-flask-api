"""Face swap orchestration: process() and face_swap() adapted for FastAPI.

Integrates face detection, face selection, enhanced swap pipeline, and
CodeFormer restoration into a single callable service.
"""

import io
import logging
import traceback
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from app.config import LEGACY_MODEL_PATH
from app.services.face_analyzer import get_face_analyser, get_many_faces
from app.services.face_swapper import get_face_swapper_model, swap_face_enhanced
from app.services.face_swapper_models import (
    validate_face_swapper_params,
    get_default_resolution,
    parse_resolution,
)
from app.services.face_selector import select_faces
from app.services.image_utils import (
    load_images_from_paths,
    encode_image_to_base64,
    clean_up_temporary_files,
)

logger = logging.getLogger(__name__)

# Module-level globals set at startup
_FACE_SWAPPER = None  # legacy insightface inswapper_128
_TORCH_DEVICE = "cpu"
_CODEFORMER_DEVICE = None
_CODEFORMER_NET = None
_UPSAMPLER = None


# ---------------------------------------------------------------------------
# Initialization (called at startup)
# ---------------------------------------------------------------------------


def init_legacy_swapper(model_path: str = LEGACY_MODEL_PATH) -> None:
    """Load the legacy insightface inswapper_128 model.

    Args:
        model_path: Path to inswapper_128.onnx.
    """
    global _FACE_SWAPPER
    import insightface
    _FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
    logger.info("Legacy inswapper_128 loaded from: %s", model_path)


def init_codeformer(
    torch_device: str,
    upsampler,
    codeformer_net,
    codeformer_device,
) -> None:
    """Store CodeFormer components for later use.

    Args:
        torch_device: 'cuda' or 'cpu'.
        upsampler: RealESRGANer instance.
        codeformer_net: Loaded CodeFormer network.
        codeformer_device: Torch device for CodeFormer.
    """
    global _TORCH_DEVICE, _UPSAMPLER, _CODEFORMER_NET, _CODEFORMER_DEVICE
    _TORCH_DEVICE = torch_device
    _UPSAMPLER = upsampler
    _CODEFORMER_NET = codeformer_net
    _CODEFORMER_DEVICE = codeformer_device


# ---------------------------------------------------------------------------
# Low-level swap function
# ---------------------------------------------------------------------------


def _swap_face(
    source_faces: list,
    target_faces: list,
    source_index: int,
    target_index: int,
    temp_frame: np.ndarray,
    face_swap_model=None,
    face_swapper_model_name: str = "inswapper_128",
    face_swapper_resolution=None,
    face_swapper_weight: float = 1.0,
    face_mask_blur: float = 0.3,
    face_mask_padding: tuple = (0, 0, 0, 0),
) -> np.ndarray:
    """Paste source face onto target image.

    Routes to legacy inswapper or enhanced pipeline based on parameters.
    """
    global _FACE_SWAPPER

    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    # Route to legacy swapper for backward compatibility
    if (
        face_swapper_model_name == "inswapper_128"
        and face_swapper_weight == 1.0
        and face_swap_model is None
        and face_swapper_resolution is None
    ):
        return _FACE_SWAPPER.get(temp_frame, target_face, source_face, paste_back=True)

    # Use enhanced swapper
    model = face_swap_model if face_swap_model is not None else _FACE_SWAPPER
    return swap_face_enhanced(
        source_face,
        target_face,
        temp_frame,
        model,
        face_swapper_model_name,
        face_swapper_resolution,
        face_swapper_weight,
        face_mask_blur=face_mask_blur,
        face_mask_padding=face_mask_padding,
    )


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def process(
    source_img: Union[Image.Image, List[Image.Image]],
    target_img: Image.Image,
    source_indexes: str = "-1",
    target_indexes: str = "-1",
    min_face_size: float = 0.0,
    face_swapper_model: str = "inswapper_128",
    face_swapper_resolution: str = None,
    face_swapper_weight: float = 1.0,
    face_mask_blur: float = 0.3,
    face_mask_padding: str = "0,0,0,0",
    face_selector_mode: str = "many",
    face_selector_order: str = "left-right",
    face_selector_gender: str = None,
    face_selector_age_start: int = None,
    face_selector_age_end: int = None,
) -> Image.Image:
    """Perform face swapping on target image using source face(s).

    Args:
        source_img: Single PIL Image or list of PIL Images (one per source).
        target_img: PIL Image to swap faces into.
        source_indexes: Comma-separated source face indexes ("-1" = auto).
        target_indexes: Comma-separated target face indexes ("-1" = auto).
        min_face_size: Minimum face size as percentage of image dimension.
        face_swapper_model: Model name (e.g., 'inswapper_128').
        face_swapper_resolution: Resolution string or None for default.
        face_swapper_weight: Identity blend weight (0.0-1.0).
        face_mask_blur: Edge softness (0.0-1.0).
        face_mask_padding: Inset as "top,right,bottom,left".
        face_selector_mode: 'many' or 'one'.
        face_selector_order: Sort order for target faces.
        face_selector_gender: 'male' or 'female' filter.
        face_selector_age_start: Minimum age filter.
        face_selector_age_end: Maximum age filter.

    Returns:
        PIL Image with swapped faces.

    Raises:
        ValueError: If parameters are invalid.
        Exception: If no faces detected or swap fails.
    """
    # Normalize to list
    if not isinstance(source_img, list):
        source_img = [source_img]

    # Validate face swapper parameters
    if face_swapper_resolution is None:
        face_swapper_resolution = get_default_resolution(face_swapper_model)

    try:
        validate_face_swapper_params(face_swapper_model, face_swapper_resolution)
    except ValueError:
        logger.exception("Validation error")
        raise

    resolution = parse_resolution(face_swapper_resolution)

    # Validate weight range
    if not 0.0 <= face_swapper_weight <= 1.0:
        raise ValueError(
            f"face_swapper_weight must be between 0.0 and 1.0, "
            f"got {face_swapper_weight}"
        )

    # Parse face mask padding string
    mask_padding = tuple(int(x) for x in face_mask_padding.split(","))

    # Validate face selector mode
    if face_selector_mode not in ("many", "one"):
        raise ValueError(
            f"face_selector_mode must be 'many' or 'one', "
            f"got '{face_selector_mode}'"
        )

    # Load face swapper model (lazy)
    face_swap_model = get_face_swapper_model(face_swapper_model)

    # Get face analyser singleton
    face_analyser = get_face_analyser(_TORCH_DEVICE)

    # Convert target to BGR
    try:
        target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    except Exception:
        logger.exception("Failed to convert target image")
        raise

    # Detect and filter target faces
    target_faces = get_many_faces(face_analyser, target_bgr)
    if target_faces:
        target_faces = select_faces(
            target_faces,
            mode=face_selector_mode,
            order=face_selector_order,
            gender=face_selector_gender,
            age_start=face_selector_age_start,
            age_end=face_selector_age_end,
        )

    if target_faces is None or len(target_faces) == 0:
        raise Exception("The target image does not contain any faces!")

    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    import copy
    temp_frame = copy.deepcopy(target_bgr)

    if isinstance(source_img, list) and num_source_images == num_target_faces:
        logger.info(
            "Replacing faces in target from left to right by order",
        )
        for i in range(num_target_faces):
            src_bgr = cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR)
            source_faces = get_many_faces(
                face_analyser, src_bgr, min_face_size,
            )

            if source_faces is None or len(source_faces) == 0:
                raise Exception("No source faces found!")

            temp_frame = _swap_face(
                source_faces, target_faces, i, i, temp_frame,
                face_swap_model, face_swapper_model, resolution,
                face_swapper_weight,
                face_mask_blur=face_mask_blur,
                face_mask_padding=mask_padding,
            )
    elif num_source_images == 1:
        src_bgr = cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR)
        source_faces = get_many_faces(face_analyser, src_bgr, min_face_size)
        num_source_faces = len(source_faces) if source_faces else 0

        logger.info("Source faces: %d, Target faces: %d", num_source_faces, num_target_faces)

        if source_faces is None or num_source_faces == 0:
            raise Exception("No source faces found!")

        if source_indexes == "-1" and target_indexes != "-1":
            logger.info(
                "Replacing specific target face(s) with source face 0",
            )
            tgt_idxs = target_indexes.split(",")
            for target_index in tgt_idxs:
                target_index = int(target_index)
                if target_index >= num_target_faces:
                    raise ValueError(
                        f"Target index {target_index} is out of range. "
                        f"Target image has {num_target_faces} face(s) "
                        f"(indexes 0-{num_target_faces - 1})."
                    )
                temp_frame = _swap_face(
                    source_faces, target_faces, 0, target_index, temp_frame,
                    face_swap_model, face_swapper_model, resolution,
                    face_swapper_weight,
                    face_mask_blur=face_mask_blur,
                    face_mask_padding=mask_padding,
                )
        elif target_indexes == "-1":
            if num_source_faces == 1:
                logger.info("Replacing first target face with source face")
                num_iterations = 1
            elif num_source_faces < num_target_faces:
                logger.info(
                    "Fewer source faces than target, replacing first %d",
                    num_source_faces,
                )
                num_iterations = num_source_faces
            elif num_target_faces < num_source_faces:
                logger.info(
                    "Fewer target faces than source, replacing %d faces",
                    num_target_faces,
                )
                num_iterations = num_target_faces
            else:
                logger.info("Replacing all target faces with source faces")
                num_iterations = num_target_faces

            for i in range(num_iterations):
                source_index = 0 if num_source_faces == 1 else i
                temp_frame = _swap_face(
                    source_faces, target_faces, source_index, i, temp_frame,
                    face_swap_model, face_swapper_model, resolution,
                    face_swapper_weight,
                )
        else:
            logger.info(
                "Replacing specific target face(s) with specific source face(s)",
            )
            src_idxs = source_indexes.split(",")
            tgt_idxs = target_indexes.split(",")

            if len(src_idxs) > num_source_faces:
                raise Exception(
                    "Number of source indexes is greater than the number "
                    "of faces in the source image"
                )
            if len(tgt_idxs) > num_target_faces:
                raise Exception(
                    "Number of target indexes is greater than the number "
                    "of faces in the target image"
                )

            if len(src_idxs) == len(tgt_idxs):
                for si, ti in zip(src_idxs, tgt_idxs):
                    source_index = int(si)
                    target_index = int(ti)

                    if source_index > num_source_faces - 1:
                        raise ValueError(
                            f"Source index {source_index} exceeds "
                            f"available source faces"
                        )
                    if target_index > num_target_faces - 1:
                        raise ValueError(
                            f"Target index {target_index} exceeds "
                            f"available target faces"
                        )

                    temp_frame = _swap_face(
                        source_faces, target_faces,
                        source_index, target_index, temp_frame,
                        face_swap_model, face_swapper_model, resolution,
                        face_swapper_weight,
                    )
    else:
        logger.error("Unsupported face configuration")
        raise Exception("Unsupported face configuration")

    result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    return result_image


def face_swap(
    src_img_path: str,
    target_img_path: str,
    source_indexes: str = "-1",
    target_indexes: str = "-1",
    background_enhance: bool = True,
    face_restore: bool = True,
    face_upsample: bool = True,
    upscale: int = 1,
    codeformer_fidelity: float = 0.5,
    output_format: str = "JPEG",
    min_face_size: float = 0.0,
    face_swapper_model: str = "inswapper_128",
    face_swapper_resolution: str = None,
    face_swapper_weight: float = 1.0,
    face_mask_blur: float = 0.3,
    face_mask_padding: str = "0,0,0,0",
    face_selector_mode: str = "many",
    face_selector_order: str = "left-right",
    face_selector_gender: str = None,
    face_selector_age_start: int = None,
    face_selector_age_end: int = None,
) -> str:
    """Full face swap pipeline: load, swap, restore, encode.

    Args:
        src_img_path: Source image path(s), semicolon-separated for multiple.
        target_img_path: Target image path.
        (All other params match the Pydantic FaceSwapRequest model.)

    Returns:
        Base64-encoded output image string.

    Raises:
        Exception: If any step fails, with traceback in the message.
    """
    try:
        source_imgs, target_img = load_images_from_paths(
            src_img_path, target_img_path,
        )
    except Exception:
        logger.exception("Failed to load images")
        raise

    try:
        logger.info("Performing face swap")
        result_image = process(
            source_imgs,
            target_img,
            source_indexes=source_indexes,
            target_indexes=target_indexes,
            min_face_size=min_face_size,
            face_swapper_model=face_swapper_model,
            face_swapper_resolution=face_swapper_resolution,
            face_swapper_weight=face_swapper_weight,
            face_mask_blur=face_mask_blur,
            face_mask_padding=face_mask_padding,
            face_selector_mode=face_selector_mode,
            face_selector_order=face_selector_order,
            face_selector_gender=face_selector_gender,
            face_selector_age_start=face_selector_age_start,
            face_selector_age_end=face_selector_age_end,
        )
        logger.info("Face swap complete")
    except Exception:
        logger.exception("Face swap failed")
        raise

    if face_restore:
        try:
            from app.services.restoration import face_restoration

            result_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            logger.info("Performing face restoration using CodeFormer")

            result_bgr = face_restoration(
                result_bgr,
                background_enhance,
                face_upsample,
                upscale,
                codeformer_fidelity,
                _UPSAMPLER,
                _CODEFORMER_NET,
                _CODEFORMER_DEVICE,
            )

            logger.info("CodeFormer face restoration completed")
            result_image = Image.fromarray(result_bgr)
        except Exception:
            logger.exception("Face restoration failed")
            raise

    try:
        encoded = encode_image_to_base64(result_image, output_format)
        logger.debug("Output image size: %d characters", len(encoded))
        return encoded
    except Exception:
        logger.exception("Failed to encode output image")
        raise

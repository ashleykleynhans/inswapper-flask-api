"""CodeFormer face restoration with RealESRGAN upscaling.

Ported from runpod-worker-inswapper. Enhanced with graceful GPU error
degradation and automatic upscale limiting for large images.
"""

import os
import logging

import cv2
import torch
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

logger = logging.getLogger(__name__)


def check_ckpts() -> None:
    """Verify CodeFormer weights exist.

    Raises:
        FileNotFoundError: If any required weight files are missing.
    """
    weights = [
        "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth",
        "CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth",
        "CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth",
        "CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
    ]
    missing = [w for w in weights if not os.path.exists(w)]
    if missing:
        raise FileNotFoundError(
            f"CodeFormer weights missing: {', '.join(missing)}. "
            f"Run: python3 scripts/download_models.py"
        )


def set_realesrgan() -> RealESRGANer:
    """Initialize RealESRGAN x2plus upsampler.

    Returns:
        Configured RealESRGANer instance with RRDBNet architecture.
    """
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


def face_restoration(
    img: cv2.Mat,
    background_enhance: bool,
    face_upsample: bool,
    upscale: int,
    codeformer_fidelity: float,
    upsampler: RealESRGANer,
    codeformer_net: torch.nn.Module,
    device: torch.device,
) -> cv2.Mat:
    """Run CodeFormer face restoration on an image.

    Args:
        img: BGR image as numpy array.
        background_enhance: Whether to upscale the background with RealESRGAN.
        face_upsample: Whether to upsample individual faces.
        upscale: Upscale factor (clamped to 1-4 automatically).
        codeformer_fidelity: Fidelity weight (0.0 = highest quality, 1.0 = strict).
        upsampler: RealESRGANer instance.
        codeformer_net: Loaded CodeFormer network.
        device: Torch device.

    Returns:
        Restored RGB image as numpy array.
    """
    try:
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        upscale = int(upscale)
        if upscale > 4:
            upscale = 4
        if upscale > 2 and max(img.shape[:2]) > 1000:
            upscale = 2
        if max(img.shape[:2]) > 1500:
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
        )
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True,
            )
            normalize(
                cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True,
            )
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True,
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                logger.error("Failed inference for CodeFormer: %s", error)
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1),
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if bg_upsampler is not None:
            bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)

        if face_upsample and face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=draw_box,
                face_upsampler=face_upsampler,
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=draw_box,
            )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception:  # pragma: no cover — defensive, re-raised above
        logger.exception("Global exception in face_restoration")
        raise

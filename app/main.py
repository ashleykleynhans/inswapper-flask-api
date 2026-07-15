"""FaceSwap FastAPI application.

Replaces the old Flask app.py with FastAPI, async queue, and the full
enhanced face swap pipeline.
"""

import argparse
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from basicsr.utils.registry import ARCH_REGISTRY

from app.config import (
    BASE_DIR,
    CODEFORMER_CODE_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
    init_logging,
)
from app.routes.health import router as health_router
from app.routes.faceswap import router as faceswap_router
from app.services.face_analyzer import get_face_analyser
from app.services.face_swap_service import (
    init_legacy_swapper,
    init_codeformer,
    face_swap,
)
from app.services.restoration import check_ckpts, set_realesrgan
from app.job_queue.worker import AsyncJobQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue processing callback
# ---------------------------------------------------------------------------


async def _process_job(job_id: str, input_data: dict) -> dict:
    """Process a single face swap job from the queue.

    Args:
        job_id: Unique job identifier.
        input_data: FaceSwapRequest as a dict.

    Returns:
        Dict with 'image' key containing the base64 result.
    """
    from app.services.image_utils import (
        decode_base64_to_disk,
        clean_up_temporary_files,
    )

    source_path = None
    target_path = None

    try:
        source_path = decode_base64_to_disk(input_data["source_image"], "source")
        target_path = decode_base64_to_disk(input_data["target_image"], "target")

        logger.info("[%s] Processing face swap", job_id)

        result_image = face_swap(
            src_img_path=source_path,
            target_img_path=target_path,
            source_indexes=input_data.get("source_indexes", "-1"),
            target_indexes=input_data.get("target_indexes", "-1"),
            background_enhance=input_data.get("background_enhance", True),
            face_restore=input_data.get("face_restore", True),
            face_upsample=input_data.get("face_upsample", True),
            upscale=input_data.get("upscale", 1),
            codeformer_fidelity=input_data.get("codeformer_fidelity", 0.5),
            output_format=input_data.get("output_format", "JPEG"),
            min_face_size=input_data.get("min_face_size", 0.0),
            face_swapper_model=input_data.get("face_swapper_model", "inswapper_128"),
            face_swapper_resolution=input_data.get("face_swapper_resolution"),
            face_swapper_weight=input_data.get("face_swapper_weight", 1.0),
            face_mask_blur=input_data.get("face_mask_blur", 0.3),
            face_mask_padding=input_data.get("face_mask_padding", "0,0,0,0"),
            face_selector_mode=input_data.get("face_selector_mode", "many"),
            face_selector_order=input_data.get("face_selector_order", "left-right"),
            face_selector_gender=input_data.get("face_selector_gender"),
            face_selector_age_start=input_data.get("face_selector_age_start"),
            face_selector_age_end=input_data.get("face_selector_age_end"),
        )

        return {"image": result_image}
    finally:
        clean_up_temporary_files(source_path, target_path)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""
    # --- Startup ---
    init_logging()
    logger.info("Starting FaceSwap API...")

    # Add CodeFormer to sys.path (needed for basicsr/facelib imports)
    codeformer_path = str(CODEFORMER_CODE_DIR)
    if codeformer_path not in sys.path:
        sys.path.insert(0, codeformer_path)

    # Set up torch device
    if torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    logger.info("Torch device: %s", torch_device.upper())

    # Initialize face analyser (buffalo_l)
    get_face_analyser(torch_device)
    logger.info("Face analyser initialized")

    # Initialize legacy inswapper_128
    init_legacy_swapper()
    logger.info("Legacy inswapper_128 loaded")

    # Verify CodeFormer weights
    check_ckpts()
    logger.info("CodeFormer weights verified")

    # Set up RealESRGAN upsampler
    upsampler = set_realesrgan()
    logger.info("RealESRGAN x2plus initialized")

    # Load CodeFormer network
    codeformer_device = torch.device(torch_device)
    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(codeformer_device)

    ckpt_path = os.path.join(
        BASE_DIR, "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth",
    )
    logger.info("Loading CodeFormer model: %s", ckpt_path)
    codeformer_checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(codeformer_checkpoint)
    codeformer_net.eval()
    logger.info("CodeFormer loaded")

    # Initialize face swap service with CodeFormer components
    init_codeformer(torch_device, upsampler, codeformer_net, codeformer_device)

    # Start the async job queue
    queue = AsyncJobQueue(process_fn=_process_job)
    await queue.start()
    app.state.queue = queue
    logger.info("Queue worker started")

    yield

    # --- Shutdown ---
    logger.info("Shutting down...")
    if hasattr(app.state, "queue") and app.state.queue:
        await app.state.queue.stop()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FaceSwap API",
    version="2.0.0",
    description="GPU-accelerated face swapping API with 13 models, "
    "CodeFormer restoration, and VRAM-safe serial queue processing.",
    docs_url="/docs",
    lifespan=lifespan,
)

# Register routes
app.include_router(health_router)
app.include_router(faceswap_router)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "msg": f"{request.url.path} not found",
            "detail": str(exc),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "msg": "Internal Server Error",
            "detail": str(exc),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions."""
    logger.exception("Unhandled exception at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "msg": str(exc),
            "detail": traceback.format_exc(),
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:  # pragma: no cover
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FaceSwap REST API")
    parser.add_argument(
        "-p", "--port",
        help="Port to listen on",
        type=int,
        default=DEFAULT_PORT,
    )
    parser.add_argument(
        "-H", "--host",
        help="Host to bind to",
        default=DEFAULT_HOST,
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    args = get_args()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )

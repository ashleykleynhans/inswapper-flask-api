"""Face swap endpoints: async queue mode and synchronous mode."""

import logging
import traceback

from fastapi import APIRouter, Request, HTTPException

from app.models.requests import FaceSwapRequest
from app.models.responses import (
    JobAcceptedResponse,
    JobStatusResponse,
    SyncFaceSwapResponse,
    ErrorResponse,
)
from app.services.image_utils import (
    decode_base64_to_disk,
    clean_up_temporary_files,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faceswap", tags=["faceswap"])


@router.post("", response_model=JobAcceptedResponse, status_code=202)
async def faceswap_async(
    request: Request,
    payload: FaceSwapRequest,
) -> JobAcceptedResponse:
    """Submit a face swap job to the async queue (VRAM-safe).

    Returns immediately with a job ID. Poll GET /faceswap/{job_id}/status
    to retrieve the result when complete.
    """
    queue = request.app.state.queue
    if queue is None:
        raise HTTPException(
            status_code=503,
            detail="Queue not initialized. Service is starting up.",
        )

    job_id = queue.submit(payload.model_dump())
    return JobAcceptedResponse(
        status="queued",
        job_id=job_id,
        status_url=f"/faceswap/{job_id}/status",
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def faceswap_status(
    request: Request,
    job_id: str,
) -> JobStatusResponse:
    """Get the current status and result of a face swap job."""
    queue = request.app.state.queue
    if queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialized")

    job = queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(**job.to_dict())


@router.post("/sync", response_model=SyncFaceSwapResponse)
async def faceswap_sync(
    request: Request,
    payload: FaceSwapRequest,
) -> SyncFaceSwapResponse:
    """Synchronous face swap (blocks until complete).

    Only safe for single-client scenarios. For production use,
    prefer the async POST /faceswap endpoint.
    """
    from app.services.face_swap_service import face_swap

    source_path = None
    target_path = None

    try:
        # Decode and save images to temp files
        source_path = decode_base64_to_disk(payload.source_image, "source")
        target_path = decode_base64_to_disk(payload.target_image, "target")

        logger.info("Sync face swap: source=%s target=%s", source_path, target_path)

        result_image = face_swap(
            src_img_path=source_path,
            target_img_path=target_path,
            source_indexes=payload.source_indexes,
            target_indexes=payload.target_indexes,
            background_enhance=payload.background_enhance,
            face_restore=payload.face_restore,
            face_upsample=payload.face_upsample,
            upscale=payload.upscale,
            codeformer_fidelity=payload.codeformer_fidelity,
            output_format=payload.output_format,
            min_face_size=payload.min_face_size,
            face_swapper_model=payload.face_swapper_model,
            face_swapper_resolution=payload.face_swapper_resolution,
            face_swapper_weight=payload.face_swapper_weight,
            face_mask_blur=payload.face_mask_blur,
            face_mask_padding=payload.face_mask_padding,
            face_selector_mode=payload.face_selector_mode,
            face_selector_order=payload.face_selector_order,
            face_selector_gender=payload.face_selector_gender,
            face_selector_age_start=payload.face_selector_age_start,
            face_selector_age_end=payload.face_selector_age_end,
        )

        return SyncFaceSwapResponse(
            status="ok",
            image=result_image,
        )
    except Exception as exc:
        logger.exception("Sync face swap failed")
        return SyncFaceSwapResponse(
            status="error",
            msg="Face swap failed",
            detail=str(exc) + "\n" + traceback.format_exc(),
        )
    finally:
        clean_up_temporary_files(source_path, target_path)

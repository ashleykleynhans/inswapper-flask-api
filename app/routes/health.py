"""Health check endpoint."""

import logging

from fastapi import APIRouter, Request

from app.models.responses import HealthResponse
from app.services.face_swapper_models import FACE_SWAPPER_MODEL_SET

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check with available models and queue depth."""
    queue_depth = 0
    if hasattr(request.app.state, "queue") and request.app.state.queue:
        queue_depth = request.app.state.queue.queue_depth

    return HealthResponse(
        status="ok",
        models_available=sorted(FACE_SWAPPER_MODEL_SET.keys()),
        queue_depth=queue_depth,
    )

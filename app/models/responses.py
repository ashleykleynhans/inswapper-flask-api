"""Pydantic response models for the face swap API."""

from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """GET / response."""
    status: str = "ok"
    models_available: list[str] = []
    queue_depth: int = 0


class JobAcceptedResponse(BaseModel):
    """POST /faceswap async mode response."""
    status: str = "queued"
    job_id: str
    status_url: str


class JobStatusResponse(BaseModel):
    """GET /faceswap/{job_id}/status response."""
    status: str  # queued, processing, completed, failed
    job_id: str
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class SyncFaceSwapResponse(BaseModel):
    """POST /faceswap/sync response."""
    status: str  # "ok" or "error"
    image: Optional[str] = None
    msg: Optional[str] = None
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    msg: str
    detail: Optional[str] = None

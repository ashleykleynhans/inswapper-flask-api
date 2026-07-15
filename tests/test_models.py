"""Tests for Pydantic request/response models."""

import pytest
from pydantic import ValidationError

from app.models.requests import FaceSwapRequest
from app.models.responses import (
    HealthResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    SyncFaceSwapResponse,
    ErrorResponse,
)


class TestFaceSwapRequest:
    """Tests for FaceSwapRequest validation."""

    def test_minimal_valid_payload(self):
        req = FaceSwapRequest(
            source_image="/9j/fakejpeg",
            target_image="/9j/fakejpeg",
        )
        assert req.source_image == "/9j/fakejpeg"
        assert req.target_image == "/9j/fakejpeg"
        assert req.source_indexes == "-1"
        assert req.face_swapper_model == "inswapper_128"

    def test_missing_source_image(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(target_image="/9j/fake")

    def test_missing_target_image(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="/9j/fake")

    def test_all_defaults(self):
        req = FaceSwapRequest(
            source_image="src",
            target_image="tgt",
        )
        assert req.background_enhance is True
        assert req.face_restore is True
        assert req.face_upsample is True
        assert req.upscale == 1
        assert req.codeformer_fidelity == 0.5
        assert req.output_format == "JPEG"
        assert req.min_face_size == 0.0
        assert req.face_swapper_model == "inswapper_128"
        assert req.face_swapper_resolution is None
        assert req.face_swapper_weight == 1.0
        assert req.face_mask_blur == 0.3
        assert req.face_mask_padding == "0,0,0,0"
        assert req.face_selector_mode == "many"
        assert req.face_selector_order == "left-right"
        assert req.face_selector_gender is None
        assert req.face_selector_age_start is None
        assert req.face_selector_age_end is None

    def test_upscale_below_min(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", upscale=0)

    def test_upscale_above_max(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", upscale=5)

    def test_codeformer_fidelity_out_of_range(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", codeformer_fidelity=1.5)

    def test_invalid_output_format(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", output_format="GIF")

    def test_invalid_selector_mode(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", face_selector_mode="all")

    def test_invalid_selector_order(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", face_selector_order="random")

    def test_invalid_gender(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", face_selector_gender="other")

    def test_age_start_out_of_range(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", face_selector_age_start=150)

    def test_min_face_size_out_of_range(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", min_face_size=101)

    def test_face_swapper_weight_out_of_range(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", face_swapper_weight=2.0)

    def test_face_mask_blur_out_of_range(self):
        with pytest.raises(ValidationError):
            FaceSwapRequest(source_image="s", target_image="t", face_mask_blur=1.5)

    def test_custom_values(self):
        req = FaceSwapRequest(
            source_image="src",
            target_image="tgt",
            face_swapper_model="simswap_256",
            face_swapper_resolution="512x512",
            face_selector_gender="female",
            face_selector_age_start=20,
            face_selector_age_end=50,
            face_selector_mode="one",
            face_selector_order="best-worst",
        )
        assert req.face_swapper_model == "simswap_256"
        assert req.face_selector_gender == "female"
        assert req.face_selector_age_start == 20


class TestResponses:
    """Tests for response model serialization."""

    def test_health_response(self):
        r = HealthResponse(
            status="ok",
            models_available=["inswapper_128"],
            queue_depth=3,
        )
        d = r.model_dump()
        assert d["status"] == "ok"
        assert d["models_available"] == ["inswapper_128"]
        assert d["queue_depth"] == 3

    def test_job_accepted_response(self):
        r = JobAcceptedResponse(
            status="queued",
            job_id="abc-123",
            status_url="/faceswap/abc-123/status",
        )
        d = r.model_dump()
        assert d["job_id"] == "abc-123"

    def test_job_status_response(self):
        r = JobStatusResponse(
            status="completed",
            job_id="abc-123",
            result={"image": "base64data"},
            created_at="2025-01-01T00:00:00Z",
        )
        d = r.model_dump()
        assert d["status"] == "completed"

    def test_sync_face_swap_response_ok(self):
        r = SyncFaceSwapResponse(status="ok", image="base64img")
        d = r.model_dump()
        assert d["image"] == "base64img"

    def test_sync_face_swap_response_error(self):
        r = SyncFaceSwapResponse(
            status="error",
            msg="Face swap failed",
            detail="no faces found",
        )
        d = r.model_dump()
        assert d["msg"] == "Face swap failed"

    def test_error_response(self):
        r = ErrorResponse(status="error", msg="bad", detail="something")
        d = r.model_dump()
        assert d["detail"] == "something"

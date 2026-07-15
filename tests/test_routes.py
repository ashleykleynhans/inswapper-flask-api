"""Tests for FastAPI route handlers."""

import sys
from contextlib import asynccontextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def mock_all_ml_modules():
    """Mock all heavy ML modules before anything imports app.

    Module-scoped so the mocks persist for all tests in this file.
    """
    heavy = [
        "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
        "torchvision", "torchvision.transforms",
        "torchvision.transforms.functional",
        "cv2", "cv2.dnn",
        "insightface", "insightface.app", "insightface.model_zoo",
        "insightface.utils", "insightface.utils.face_align",
        "onnx", "onnx.numpy_helper", "onnxruntime",
        "basicsr", "basicsr.utils", "basicsr.utils.imwrite",
        "basicsr.utils.img2tensor", "basicsr.utils.tensor2img",
        "basicsr.utils.download_util", "basicsr.utils.realesrgan_utils",
        "basicsr.utils.registry", "basicsr.archs", "basicsr.archs.rrdbnet_arch",
        "facelib", "facelib.utils",
        "facelib.utils.face_restoration_helper",
        "facelib.utils.misc",
    ]
    for mod in heavy:
        sys.modules[mod] = mock.MagicMock()

    sys.modules["torch.cuda"].is_available.return_value = False

    yield


def _make_mock_queue():
    """Create a mock queue."""
    q = mock.MagicMock()
    q.queue_depth = 0
    q.active_jobs = 0
    q.submit.return_value = "test-job-id"

    job = mock.MagicMock()
    job.to_dict.return_value = {
        "status": "completed",
        "job_id": "test-job-id",
        "result": {"image": "base64test"},
        "error": None,
        "created_at": "2025-01-01T00:00:00Z",
        "started_at": "2025-01-01T00:00:01Z",
        "completed_at": "2025-01-01T00:00:10Z",
    }
    q.get_job.return_value = job
    return q


def _make_app(queue):
    """Build a FastAPI TestClient app with no-op lifespan."""
    with (
        mock.patch("app.services.face_analyzer.get_face_analyser"),
        mock.patch("app.services.face_swap_service.init_legacy_swapper"),
        mock.patch("app.services.restoration.check_ckpts", return_value=None),
        mock.patch(
            "app.services.restoration.set_realesrgan",
            return_value=mock.MagicMock(),
        ),
        mock.patch("app.services.face_swap_service.init_codeformer"),
        mock.patch(
            "app.main.AsyncJobQueue.start",
            new_callable=mock.AsyncMock,
        ),
    ):
        # Clear cached app module to get fresh state per test
        for mod in list(sys.modules):
            if mod.startswith("app.") and mod != "app.config":
                sys.modules.pop(mod, None)

        from app.main import app

        @asynccontextmanager
        async def test_lifespan(app):
            app.state.queue = queue
            yield

        app.router.lifespan_context = test_lifespan
        return app


@pytest.fixture
def client():
    """TestClient with mock queue."""
    queue = _make_mock_queue()
    app = _make_app(queue)
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def client_no_queue():
    """TestClient without a queue (state.queue = None)."""
    app = _make_app(None)
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def client_empty_queue():
    """TestClient with queue that returns None for all lookups."""
    queue = _make_mock_queue()
    queue.get_job.return_value = None
    app = _make_app(queue)
    with TestClient(app) as tc:
        yield tc


class TestHealthRoute:
    def test_health_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_available" in data


class TestFaceswapAsyncRoute:
    def test_submit_valid_request(self, client):
        payload = {"source_image": "/9j/fake", "target_image": "/9j/fake"}
        resp = client.post("/faceswap", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert data["job_id"] == "test-job-id"
        assert "status_url" in data

    def test_submit_invalid_missing_source(self, client):
        resp = client.post("/faceswap", json={"target_image": "/9j/fake"})
        assert resp.status_code == 422

    def test_submit_no_queue_returns_503(self, client_no_queue):
        resp = client_no_queue.post(
            "/faceswap",
            json={"source_image": "/9j/fake", "target_image": "/9j/fake"},
        )
        assert resp.status_code == 503


class TestFaceswapStatusRoute:
    def test_status_returns_job(self, client):
        resp = client.get("/faceswap/test-job-id/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_status_not_found(self, client_empty_queue):
        resp = client_empty_queue.get("/faceswap/nonexistent/status")
        assert resp.status_code == 404

    def test_queue_not_initialized(self, client_no_queue):
        resp = client_no_queue.get("/faceswap/test-id/status")
        assert resp.status_code == 503


class TestExceptionHandlers:
    def test_404_on_unknown_route(self, client):
        resp = client.get("/nonexistent-route")
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "error"

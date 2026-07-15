"""Tests for main.py — lifespan, route registration, CLI, error handlers."""

import sys
from contextlib import asynccontextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_all_ml():
    """Mock everything the app imports."""
    heavy = {
        "torch": mock.MagicMock(),
        "torch.nn": mock.MagicMock(),
        "torch.nn.functional": mock.MagicMock(),
        "torch.cuda": mock.MagicMock(),
        "torchvision": mock.MagicMock(),
        "torchvision.transforms": mock.MagicMock(),
        "torchvision.transforms.functional": mock.MagicMock(),
        "cv2": mock.MagicMock(),
        "insightface": mock.MagicMock(),
        "insightface.app": mock.MagicMock(),
        "insightface.model_zoo": mock.MagicMock(),
        "insightface.utils": mock.MagicMock(),
        "insightface.utils.face_align": mock.MagicMock(),
        "onnx": mock.MagicMock(),
        "onnx.numpy_helper": mock.MagicMock(),
        "onnxruntime": mock.MagicMock(),
        "basicsr": mock.MagicMock(),
        "basicsr.utils": mock.MagicMock(),
        "basicsr.utils.imwrite": mock.MagicMock(),
        "basicsr.utils.img2tensor": mock.MagicMock(),
        "basicsr.utils.tensor2img": mock.MagicMock(),
        "basicsr.utils.download_util": mock.MagicMock(),
        "basicsr.utils.realesrgan_utils": mock.MagicMock(),
        "basicsr.utils.registry": mock.MagicMock(),
        "basicsr.archs": mock.MagicMock(),
        "basicsr.archs.rrdbnet_arch": mock.MagicMock(),
        "facelib": mock.MagicMock(),
        "facelib.utils": mock.MagicMock(),
        "facelib.utils.face_restoration_helper": mock.MagicMock(),
        "facelib.utils.misc": mock.MagicMock(),
    }
    for name, mod in heavy.items():
        sys.modules[name] = mod

    sys.modules["torch"].cuda.is_available.return_value = False
    sys.modules["torch"].device.return_value = "cpu"

    # Mock basicsr ARCH_REGISTRY
    sys.modules["basicsr.utils.registry"].ARCH_REGISTRY = mock.MagicMock()
    sys.modules["basicsr.utils.registry"].ARCH_REGISTRY.get.return_value = mock.MagicMock()

    yield


def _make_mock_queue():
    q = mock.MagicMock()
    q.queue_depth = 0
    q.active_jobs = 0
    q.submit.return_value = "test-job-id"
    job = mock.MagicMock()
    job.to_dict.return_value = {
        "status": "completed", "job_id": "test-job-id",
        "result": {"image": "base64test"}, "error": None,
        "created_at": None, "started_at": None, "completed_at": None,
    }
    q.get_job.return_value = job
    return q


@pytest.fixture
def client():
    """TestClient that mocks the full lifespan."""
    # Must mock all the real startup functions
    with (
        mock.patch("app.services.face_analyzer.get_face_analyser"),
        mock.patch("app.services.face_swap_service.init_legacy_swapper"),
        mock.patch("app.services.restoration.check_ckpts", return_value=None),
        mock.patch(
            "app.services.restoration.set_realesrgan",
            return_value=mock.MagicMock(),
        ),
        mock.patch("app.services.face_swap_service.init_codeformer"),
        mock.patch("app.main.AsyncJobQueue.start", new_callable=mock.AsyncMock),
        mock.patch("app.main.init_logging"),
        mock.patch("app.main.torch.load", return_value={"params_ema": {}}),
        mock.patch("builtins.open", mock.mock_open()),
    ):
        # Clear cached modules
        for mod in list(sys.modules):
            if mod.startswith("app.") and mod != "app.config":
                sys.modules.pop(mod, None)

        from app.main import app

        @asynccontextmanager
        async def test_lifespan(app):
            app.state.queue = _make_mock_queue()
            yield

        app.router.lifespan_context = test_lifespan

        with TestClient(app) as tc:
            yield tc


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAsyncFaceswap:
    def test_submit(self, client):
        resp = client.post("/faceswap", json={
            "source_image": "/9j/fake", "target_image": "/9j/fake",
        })
        assert resp.status_code == 202

    def test_validation(self, client):
        resp = client.post("/faceswap", json={"target_image": "/9j/f"})
        assert resp.status_code == 422


class TestStatusEndpoint:
    def test_status(self, client):
        resp = client.get("/faceswap/test-job-id/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


class TestSyncEndpoint:
    def test_sync_success(self, client):
        with mock.patch(
            "app.services.face_swap_service.face_swap",
            return_value="base64result",
        ):
            with mock.patch(
                "app.services.image_utils.decode_base64_to_disk",
                return_value="/tmp/fake_test.jpg",
            ):
                with mock.patch(
                    "app.services.image_utils.clean_up_temporary_files",
                ):
                    resp = client.post("/faceswap/sync", json={
                        "source_image": "/9j/fake",
                        "target_image": "/9j/fake",
                    })
                    assert resp.status_code == 200
                    assert resp.json()["status"] == "ok"

    def test_sync_error(self, client):
        with mock.patch(
            "app.services.face_swap_service.face_swap",
            side_effect=RuntimeError("GPU OOM"),
        ):
            with mock.patch(
                "app.services.image_utils.decode_base64_to_disk",
                return_value="/tmp/fake_test.jpg",
            ):
                with mock.patch(
                    "app.services.image_utils.clean_up_temporary_files",
                ):
                    resp = client.post("/faceswap/sync", json={
                        "source_image": "/9j/fake",
                        "target_image": "/9j/fake",
                    })
                    assert resp.status_code == 200
                    assert resp.json()["status"] == "error"


class TestExceptionHandlers:
    def test_404(self, client):
        resp = client.get("/no-such-route")
        assert resp.status_code == 404
        assert resp.json()["status"] == "error"

    def test_queue_not_initialized_status(self, client):
        # Override queue to None
        client.app.state.queue = None
        resp = client.get("/faceswap/nonexistent/status")
        assert resp.status_code == 503

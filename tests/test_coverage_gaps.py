"""Fill remaining easy coverage gaps."""

import base64
import sys
import os
from unittest import mock

import pytest
import numpy as np


class TestImageUtilsGaps:
    """Coverage for remaining image_utils gaps."""

    def test_decode_base64_jpeg_creates_file(self):
        from app.services.image_utils import decode_base64_to_disk
        from PIL import Image
        import io

        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        path = decode_base64_to_disk(b64, "source")
        assert os.path.exists(path)
        os.remove(path)

    def test_determine_file_extension_exception(self):
        from app.services.image_utils import determine_file_extension
        result = determine_file_extension(None)
        assert result == ".png"


class TestFaceAnalyzerGaps:
    """Coverage for face_analyzer IndexError branch."""

    def test_get_many_faces_index_error(self):
        from app.services.face_analyzer import get_many_faces

        analyser = mock.MagicMock()
        analyser.get.side_effect = IndexError("no faces")

        result = get_many_faces(analyser, np.zeros((100, 100, 3)))
        assert result is None


class TestFaceSelectorSortKey:
    """Coverage for _sort_key branches."""

    def test_sort_key_valid_orders(self):
        from app.services.face_selector import _sort_key

        for order in ("left-right", "right-left", "top-bottom",
                       "small-large", "large-small",
                       "best-worst", "worst-best"):
            key = _sort_key(order)
            assert key is not None

    def test_sort_key_unknown_order(self):
        from app.services.face_selector import _sort_key
        assert _sort_key("unknown") is None


class TestSyncEndpointErrorPath:
    """Coverage for sync endpoint error handling."""

    def test_sync_fails_when_face_swap_raises(self):
        with mock.patch.dict(sys.modules, {
            "torch": mock.MagicMock(),
            "torch.cuda": mock.MagicMock(),
            "cv2": mock.MagicMock(),
            "insightface": mock.MagicMock(),
            "insightface.app": mock.MagicMock(),
            "insightface.model_zoo": mock.MagicMock(),
            "onnx": mock.MagicMock(),
            "onnxruntime": mock.MagicMock(),
            "basicsr": mock.MagicMock(),
            "basicsr.utils": mock.MagicMock(),
            "basicsr.utils.registry": mock.MagicMock(),
            "basicsr.archs": mock.MagicMock(),
            "basicsr.archs.rrdbnet_arch": mock.MagicMock(),
            "basicsr.utils.realesrgan_utils": mock.MagicMock(),
            "facelib": mock.MagicMock(),
            "facelib.utils": mock.MagicMock(),
            "facelib.utils.face_restoration_helper": mock.MagicMock(),
            "facelib.utils.misc": mock.MagicMock(),
            "torchvision": mock.MagicMock(),
            "torchvision.transforms": mock.MagicMock(),
            "torchvision.transforms.functional": mock.MagicMock(),
        }):
            sys.modules["torch.cuda"].is_available.return_value = False

            # Mock face_swap at the RIGHT level — before it's called, but
            # after image decoding. We mock the whole face_swap_service.face_swap
            with mock.patch(
                "app.services.face_swap_service.face_swap",
                side_effect=RuntimeError("GPU out of memory"),
            ):
                with mock.patch("app.services.face_analyzer.get_face_analyser"), \
                     mock.patch("app.services.face_swap_service.init_legacy_swapper"), \
                     mock.patch("app.services.restoration.check_ckpts", return_value=None), \
                     mock.patch("app.services.restoration.set_realesrgan"), \
                     mock.patch("app.services.face_swap_service.init_codeformer"), \
                     mock.patch("app.main.AsyncJobQueue.start", new_callable=mock.AsyncMock):
                    # Mock image IO to avoid real file system calls
                    with mock.patch(
                        "app.services.image_utils.decode_base64_to_disk",
                        return_value="/tmp/fake_test.jpg",
                    ), mock.patch(
                        "app.services.image_utils.clean_up_temporary_files",
                    ), mock.patch(
                        "app.services.face_swap_service.load_images_from_paths",
                        side_effect=RuntimeError("simulated"),
                    ):
                        # Clear app cache
                        for mod in list(sys.modules):
                            if mod.startswith("app."):
                                sys.modules.pop(mod, None)

                        from app.main import app
                        from contextlib import asynccontextmanager

                        @asynccontextmanager
                        async def noop_lifespan(app):
                            app.state.queue = mock.MagicMock()
                            yield

                        app.router.lifespan_context = noop_lifespan

                        from fastapi.testclient import TestClient
                        with TestClient(app) as tc:
                            payload = {
                                "source_image": "/9j/fake",
                                "target_image": "/9j/fake",
                            }
                            resp = tc.post("/faceswap/sync", json=payload)
                            assert resp.status_code == 200
                            data = resp.json()
                            assert data["status"] == "error"
                            assert data["msg"] == "Face swap failed"

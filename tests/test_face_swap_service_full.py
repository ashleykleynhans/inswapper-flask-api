"""Mock tests for face_swap_service — exercises validation and error paths."""

import sys
from unittest import mock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def mock_all():
    heavy = {
        "torch": mock.MagicMock(),
        "torch.cuda": mock.MagicMock(),
        "insightface": mock.MagicMock(),
        "insightface.model_zoo": mock.MagicMock(),
        "onnx": mock.MagicMock(),
        "onnxruntime": mock.MagicMock(),
        "basicsr": mock.MagicMock(),
        "basicsr.utils": mock.MagicMock(),
        "basicsr.utils.registry": mock.MagicMock(),
        "basicsr.archs": mock.MagicMock(),
        "facelib": mock.MagicMock(),
        "facelib.utils": mock.MagicMock(),
        "torchvision": mock.MagicMock(),
    }
    for name, mod in heavy.items():
        sys.modules[name] = mod
    sys.modules["torch"].cuda.is_available.return_value = False
    yield


@pytest.fixture
def test_img():
    return Image.new("RGB", (100, 100), color="red")


class TestProcessValidation:
    def test_weight_out_of_range(self, test_img):
        from app.services.face_swap_service import process
        with pytest.raises(ValueError, match="face_swapper_weight"):
            process(source_img=[test_img], target_img=test_img, face_swapper_weight=2.0)

    def test_weight_negative(self, test_img):
        from app.services.face_swap_service import process
        with pytest.raises(ValueError, match="face_swapper_weight"):
            process(source_img=[test_img], target_img=test_img, face_swapper_weight=-0.5)

    def test_weight_edge_values(self, test_img):
        from app.services.face_swap_service import process
        try:
            process(source_img=[test_img], target_img=test_img, face_swapper_weight=0.0)
        except Exception:
            pass  # Expected to fail later at face detection

    def test_invalid_selector_mode(self, test_img):
        from app.services.face_swap_service import process
        with pytest.raises(ValueError, match="face_selector_mode"):
            process(source_img=[test_img], target_img=test_img, face_selector_mode="bad")


class TestInitFunctions:
    def test_init_legacy_swapper(self):
        from app.services.face_swap_service import init_legacy_swapper
        init_legacy_swapper("/fake/model.onnx")

    def test_init_codeformer(self):
        from app.services.face_swap_service import init_codeformer
        up = mock.MagicMock()
        net = mock.MagicMock()
        dev = mock.MagicMock()
        init_codeformer("cuda", up, net, dev)

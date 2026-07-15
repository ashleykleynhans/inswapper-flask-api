"""Tests for restoration module with mocked ML dependencies."""

import os
import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def mock_ml_modules():
    """Mock heavy ML modules for restoration tests."""
    heavy = {
        "torch": mock.MagicMock(),
        "torch.nn": mock.MagicMock(),
        "torch.nn.functional": mock.MagicMock(),
        "torch.cuda": mock.MagicMock(),
        "torchvision": mock.MagicMock(),
        "torchvision.transforms": mock.MagicMock(),
        "torchvision.transforms.functional": mock.MagicMock(),
        "cv2": mock.MagicMock(),
        "basicsr": mock.MagicMock(),
        "basicsr.utils": mock.MagicMock(),
        "basicsr.utils.imwrite": mock.MagicMock(),
        "basicsr.utils.img2tensor": mock.MagicMock(),
        "basicsr.utils.tensor2img": mock.MagicMock(),
        "basicsr.utils.registry": mock.MagicMock(),
        "basicsr.archs": mock.MagicMock(),
        "basicsr.archs.rrdbnet_arch": mock.MagicMock(),
        "basicsr.utils.realesrgan_utils": mock.MagicMock(),
        "facelib": mock.MagicMock(),
        "facelib.utils": mock.MagicMock(),
        "facelib.utils.face_restoration_helper": mock.MagicMock(),
        "facelib.utils.misc": mock.MagicMock(),
    }
    for name, mod in heavy.items():
        sys.modules[name] = mod

    sys.modules["torch.cuda"].is_available.return_value = False

    yield

    for mod in list(sys.modules):
        if mod.startswith("app.services.restoration"):
            sys.modules.pop(mod, None)


class TestCheckCkpts:
    """Tests for check_ckpts."""

    def test_all_weights_present(self):
        with mock.patch("os.path.exists", return_value=True):
            from app.services.restoration import check_ckpts
            check_ckpts()  # Should not raise

    def test_missing_weights_raises(self):
        with mock.patch("os.path.exists", return_value=False):
            from app.services.restoration import check_ckpts
            with pytest.raises(FileNotFoundError, match="CodeFormer weights missing"):
                check_ckpts()

    def test_partial_missing(self):
        def fake_exists(path):
            return "codeformer.pth" in path

        with mock.patch("os.path.exists", side_effect=fake_exists):
            from app.services.restoration import check_ckpts
            with pytest.raises(FileNotFoundError, match="CodeFormer weights missing"):
                check_ckpts()


class TestSetRealEsrgan:
    """Tests for set_realesrgan."""

    def test_creates_upsampler(self):
        from app.services.restoration import set_realesrgan
        upsampler = set_realesrgan()
        assert upsampler is not None

    def test_creates_upsampler_cpu(self):
        sys.modules["torch.cuda"].is_available.return_value = False
        from app.services.restoration import set_realesrgan
        upsampler = set_realesrgan()
        assert upsampler is not None

    def test_creates_upsampler_gpu(self):
        sys.modules["torch.cuda"].is_available.return_value = True
        from app.services.restoration import set_realesrgan
        upsampler = set_realesrgan()
        assert upsampler is not None

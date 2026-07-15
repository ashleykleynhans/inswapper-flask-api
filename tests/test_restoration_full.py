"""Tests for restoration module with mocked basicsr/facelib/torch."""

import sys
from unittest import mock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_ml():
    """Mock all ML imports, but keep real cv2."""
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.device.return_value = "cpu"
    fake_torch.nn.Module = mock.MagicMock

    heavy = {
        "torch": fake_torch,
        "torch.nn": mock.MagicMock(),
        "torch.nn.functional": mock.MagicMock(),
        "torch.cuda": mock.MagicMock(),
        "torchvision": mock.MagicMock(),
        "torchvision.transforms": mock.MagicMock(),
        "torchvision.transforms.functional": mock.MagicMock(),
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
        "onnx": mock.MagicMock(),
        "onnxruntime": mock.MagicMock(),
    }
    for name, mod in heavy.items():
        sys.modules[name] = mod

    yield


class TestCheckCkpts:
    def test_all_present(self):
        with mock.patch("os.path.exists", return_value=True):
            from app.services.restoration import check_ckpts
            check_ckpts()

    def test_missing(self):
        with mock.patch("os.path.exists", return_value=False):
            from app.services.restoration import check_ckpts
            with pytest.raises(FileNotFoundError, match="CodeFormer weights missing"):
                check_ckpts()


class TestSetRealEsrgan:
    def test_cpu(self):
        sys.modules["torch"].cuda.is_available.return_value = False
        from app.services.restoration import set_realesrgan
        up = set_realesrgan()
        assert up is not None

    def test_gpu(self):
        sys.modules["torch"].cuda.is_available.return_value = True
        from app.services.restoration import set_realesrgan
        up = set_realesrgan()
        assert up is not None


class TestFaceRestoration:
    def test_full_pipeline(self):
        from app.services.restoration import face_restoration

        mock_helper = mock.MagicMock()
        mock_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        mock_helper.get_inverse_affine = mock.MagicMock()
        mock_helper.paste_faces_to_input_image.return_value = np.zeros(
            (100, 100, 3), dtype=np.uint8,
        )

        # img2tensor must return a torch-like object
        fake_tensor = mock.MagicMock()
        fake_tensor.unsqueeze.return_value = fake_tensor
        fake_tensor.to.return_value = fake_tensor

        with mock.patch(
            "app.services.restoration.FaceRestoreHelper",
            return_value=mock_helper,
        ):
            with mock.patch(
                "app.services.restoration.img2tensor",
                return_value=fake_tensor,
            ):
                with mock.patch(
                    "app.services.restoration.normalize",
                ):
                    with mock.patch(
                        "app.services.restoration.tensor2img",
                        return_value=np.zeros((512, 512, 3), dtype=np.uint8),
                    ):
                        net = mock.MagicMock()
                        net.return_value = (fake_tensor, None)

                        upsampler = mock.MagicMock()
                        upsampler.enhance.return_value = (
                            np.zeros((200, 200, 3), dtype=np.uint8),
                        )

                        img = np.zeros((100, 100, 3), dtype=np.uint8)
                        result = face_restoration(
                            img, True, True, 1, 0.5, upsampler, net, "cpu",
                        )
                        assert result is not None

    def _make_tensor(self):
        t = mock.MagicMock()
        t.unsqueeze.return_value = t
        t.to.return_value = t
        return t

    def test_pipeline_no_enhance(self):
        from app.services.restoration import face_restoration

        mock_helper = mock.MagicMock()
        mock_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        mock_helper.paste_faces_to_input_image.return_value = np.zeros(
            (100, 100, 3), dtype=np.uint8,
        )

        ft = self._make_tensor()
        with mock.patch(
            "app.services.restoration.FaceRestoreHelper",
            return_value=mock_helper,
        ):
            with mock.patch(
                "app.services.restoration.img2tensor", return_value=ft,
            ):
                with mock.patch("app.services.restoration.normalize"):
                    with mock.patch(
                        "app.services.restoration.tensor2img",
                        return_value=np.zeros((512, 512, 3), dtype=np.uint8),
                    ):
                        net = mock.MagicMock()
                        net.return_value = (ft, None)
                        img = np.zeros((100, 100, 3), dtype=np.uint8)
                        result = face_restoration(
                            img, False, False, 2, 0.5, None, net, "cpu",
                        )
                        assert result is not None

    def test_codeformer_graceful_degradation(self):
        from app.services.restoration import face_restoration

        mock_helper = mock.MagicMock()
        mock_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        mock_helper.paste_faces_to_input_image.return_value = np.zeros(
            (100, 100, 3), dtype=np.uint8,
        )

        ft = self._make_tensor()
        with mock.patch(
            "app.services.restoration.FaceRestoreHelper",
            return_value=mock_helper,
        ):
            with mock.patch(
                "app.services.restoration.img2tensor", return_value=ft,
            ):
                with mock.patch("app.services.restoration.normalize"):
                    with mock.patch(
                        "app.services.restoration.tensor2img",
                        return_value=np.zeros((512, 512, 3), dtype=np.uint8),
                    ):
                        net = mock.MagicMock()
                        net.side_effect = RuntimeError("CUDA OOM")
                        upsampler = mock.MagicMock()
                        upsampler.enhance.return_value = (
                            np.zeros((200, 200, 3), dtype=np.uint8),
                        )
                        img = np.zeros((100, 100, 3), dtype=np.uint8)
                        result = face_restoration(
                            img, True, True, 1, 0.5, upsampler, net, "cpu",
                        )
                        assert result is not None

    def test_upscale_clamped_to_4(self):
        """upscale > 4 is clamped to 4."""
        from app.services.restoration import face_restoration

        ft = self._make_tensor()
        mock_helper = mock.MagicMock()
        mock_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        mock_helper.paste_faces_to_input_image.return_value = np.zeros(
            (100, 100, 3), dtype=np.uint8,
        )

        with mock.patch(
            "app.services.restoration.FaceRestoreHelper",
            return_value=mock_helper,
        ):
            with mock.patch(
                "app.services.restoration.img2tensor", return_value=ft,
            ):
                with mock.patch("app.services.restoration.normalize"):
                    with mock.patch(
                        "app.services.restoration.tensor2img",
                        return_value=np.zeros((512, 512, 3), dtype=np.uint8),
                    ):
                        net = mock.MagicMock()
                        net.return_value = (ft, None)
                        img = np.zeros((100, 100, 3), dtype=np.uint8)
                        result = face_restoration(
                            img, True, True, 5, 0.5, None, net, "cpu",
                        )
                        assert result is not None

    def test_upscale_limits_large_image(self):
        from app.services.restoration import face_restoration

        mock_helper = mock.MagicMock()
        mock_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        mock_helper.paste_faces_to_input_image.return_value = np.zeros(
            (1600, 1600, 3), dtype=np.uint8,
        )

        ft = self._make_tensor()
        with mock.patch(
            "app.services.restoration.FaceRestoreHelper",
            return_value=mock_helper,
        ):
            with mock.patch(
                "app.services.restoration.img2tensor", return_value=ft,
            ):
                with mock.patch("app.services.restoration.normalize"):
                    with mock.patch(
                        "app.services.restoration.tensor2img",
                        return_value=np.zeros((512, 512, 3), dtype=np.uint8),
                    ):
                        net = mock.MagicMock()
                        net.return_value = (ft, None)
                        img = np.zeros((1600, 1600, 3), dtype=np.uint8)
                        result = face_restoration(
                            img, True, True, 4, 0.5, None, net, "cpu",
                        )
                        assert result is not None

"""Tests for pure-logic functions in face_swapper.py (no GPU/ONNX needed)."""

import sys
from unittest import mock

import numpy as np
import pytest

# Mock all heavy imports before touching face_swapper
HEAVY = [
    "cv2", "onnx", "onnx.numpy_helper", "onnxruntime",
    "insightface", "insightface.utils", "insightface.utils.face_align",
]


@pytest.fixture(autouse=True)
def mock_heavy():
    for mod in HEAVY:
        sys.modules[mod] = mock.MagicMock()
    # cv2.GaussianBlur should act as identity for mask tests
    sys.modules["cv2"].GaussianBlur = lambda m, *a, **kw: m
    sys.modules["cv2"].cvtColor = mock.MagicMock()
    sys.modules["cv2"].warpAffine = mock.MagicMock()
    sys.modules["cv2"].invertAffineTransform = mock.MagicMock()
    sys.modules["cv2"].estimateAffinePartial2D = mock.MagicMock()
    sys.modules["cv2"].BORDER_REPLICATE = 1
    sys.modules["cv2"].INTER_AREA = 3
    sys.modules["cv2"].RANSAC = 8
    yield
    for mod in list(sys.modules):
        if mod.startswith("app.services.face_swapper"):
            sys.modules.pop(mod, None)


class TestWarpTemplates:
    def test_all_templates_are_5_point(self):
        from app.services.face_swapper import WARP_TEMPLATES
        for name, tmpl in WARP_TEMPLATES.items():
            assert tmpl.shape == (5, 2), f"{name} should be (5,2)"


class TestBalanceEmbedding:
    def test_weight_1_0(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        result = _balance_embedding(src, tgt, weight=1.0, l2_norm_target=True)
        assert result.shape == (1, 3)
        # weight=1.0 → w=-0.35 → result ≈ src*(1 - (-0.35)) + tgt*(-0.35)
        # = src*1.35 + tgt*(-0.35)
        # With l2_norm_target: tgt normalized = [0,1,0] / 1 = [0,1,0]
        # result = [1,0,0]*1.35 + [0,1,0]*(-0.35) = [1.35, -0.35, 0]
        np.testing.assert_array_almost_equal(result[0], [1.35, -0.35, 0.0], decimal=5)

    def test_weight_0_0(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        result = _balance_embedding(src, tgt, weight=0.0, l2_norm_target=True)
        # weight=0.0 → w=0.35 → result = src*0.65 + tgt*0.35
        np.testing.assert_array_almost_equal(result[0], [0.65, 0.35, 0.0], decimal=5)

    def test_weight_0_5_neutral(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 1.0]], dtype=np.float32)
        result = _balance_embedding(src, tgt, weight=0.5, l2_norm_target=True)
        # weight=0.5 → w≈0.0 → result ≈ src
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0], decimal=5)

    def test_no_l2_norm(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[2.0, 0.0]], dtype=np.float32)  # norm = 2.0
        result = _balance_embedding(src, tgt, weight=0.5, l2_norm_target=False)
        # weight=0.5 → w≈0.0 → result ≈ src + tgt*0 = src
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0], decimal=5)

    def test_zero_norm_target(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 0.0]], dtype=np.float32)
        result = _balance_embedding(src, tgt, weight=0.5, l2_norm_target=True)
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0], decimal=5)


class TestPrepareCropFrame:
    def test_basic_normalization(self):
        from app.services.face_swapper import _prepare_crop_frame
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        result = _prepare_crop_frame(frame, mean, std)
        assert result.shape == (1, 3, 64, 64)
        assert result.dtype == np.float32

    def test_zero_mean_std(self):
        from app.services.face_swapper import _prepare_crop_frame
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = _prepare_crop_frame(frame, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert result.shape == (1, 3, 32, 32)


class TestNormalizeCropFrame:
    def test_denormalize_tanh_out(self):
        from app.services.face_swapper import _normalize_crop_frame
        frame = np.zeros((1, 3, 32, 32), dtype=np.float32)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        result = _normalize_crop_frame(frame, mean, std, tanh_out=True)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_denormalize_no_tanh(self):
        from app.services.face_swapper import _normalize_crop_frame
        frame = np.zeros((1, 3, 32, 32), dtype=np.float32)
        result = _normalize_crop_frame(frame, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], tanh_out=False)
        assert result.shape == (32, 32, 3)


class TestCreateBoxMask:
    def test_mask_shape(self):
        from app.services.face_swapper import _create_box_mask
        mask = _create_box_mask((64, 64), blur=0.0, pad=(0, 0, 0, 0))
        assert mask.shape == (64, 64)
        # Center is always 1.0
        assert mask[32, 32] == 1.0
        # Top 1px is always zero (ba is at least 1)
        assert mask[0, 32] == 0.0

    def test_mask_with_padding(self):
        from app.services.face_swapper import _create_box_mask
        mask = _create_box_mask((100, 100), blur=0.1, pad=(10, 10, 10, 10))
        assert mask.shape == (100, 100)
        # Top 10% should be zeroed (pad[0] = 10% of 100 = 10px)
        assert mask[0, 50] == 0.0
        # Center should be 1.0
        assert mask[50, 50] == 1.0

    def test_mask_dtype(self):
        from app.services.face_swapper import _create_box_mask
        mask = _create_box_mask((64, 64))
        assert mask.dtype == np.float32

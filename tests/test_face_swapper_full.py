"""Tests for face_swapper module (mocked onnxruntime, cv2, insightface)."""

import sys
import os
import tempfile
from unittest import mock

import cv2 as real_cv2_mod
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_heavy(monkeypatch):
    """Mock onnx, onnxruntime, and insightface at the module level."""
    # onnxruntime
    fake_sess = mock.MagicMock()
    fake_sess.get_inputs.return_value = [
        mock.MagicMock(shape=[1, 3, 128, 128]),
        mock.MagicMock(shape=[1, 512]),
    ]
    fake_sess.get_outputs.return_value = [mock.MagicMock()]
    fake_sess.run.return_value = [np.zeros((1, 3, 128, 128), dtype=np.float32)]

    fake_ort = mock.MagicMock()
    fake_ort.InferenceSession.return_value = fake_sess

    fake_onnx = mock.MagicMock()
    fake_onnx.load.return_value = mock.MagicMock(
        graph=mock.MagicMock(initializer=[1]),
    )
    fake_onnx.numpy_helper = mock.MagicMock()
    fake_onnx.numpy_helper.to_array.return_value = np.eye(512, dtype=np.float32)

    fake_insightface = mock.MagicMock()
    fake_face_align = mock.MagicMock()
    fake_face_align.norm_crop2.return_value = (
        np.zeros((128, 128, 3), dtype=np.uint8),
        np.eye(2, 3, dtype=np.float32),
    )
    fake_insightface.utils = mock.MagicMock()
    fake_insightface.utils.face_align = fake_face_align

    mocks = {
        "onnx": fake_onnx,
        "onnx.numpy_helper": fake_onnx.numpy_helper,
        "onnxruntime": fake_ort,
        "insightface": fake_insightface,
        "insightface.utils": fake_insightface.utils,
        "insightface.utils.face_align": fake_face_align,
    }
    saved = {}
    for name, mod in mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Clear app cache
    for mod in list(sys.modules):
        if mod.startswith("app.services.face_swapper"):
            sys.modules.pop(mod, None)

    # Pre-populate embedding converters cache
    from app.services.face_swapper import EMBEDDING_CONVERTERS
    EMBEDDING_CONVERTERS.clear()
    for conv_name in (
        "crossface_ghost.onnx",
        "crossface_hififace.onnx",
        "crossface_simswap.onnx",
    ):
        conv = mock.MagicMock()
        conv.run.return_value = [np.random.randn(512).astype(np.float32)]
        EMBEDDING_CONVERTERS[conv_name] = conv

    yield

    for name in list(sys.modules):
        if name.startswith("app.services.face_swapper"):
            sys.modules.pop(name, None)
    for name, original in saved.items():
        if original is not None:
            sys.modules[name] = original
        else:
            sys.modules.pop(name, None)


@pytest.fixture
def temp_onnx():
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    yield path
    os.remove(path)


@pytest.fixture
def mock_faces():
    src = mock.MagicMock()
    src.embedding = np.random.randn(512).astype(np.float32)
    src.normed_embedding = src.embedding / np.linalg.norm(src.embedding)
    src.kps = np.random.randn(5, 2).astype(np.float32)

    tgt = mock.MagicMock()
    tgt.embedding = np.random.randn(512).astype(np.float32)
    tgt.normed_embedding = tgt.embedding / np.linalg.norm(tgt.embedding)
    tgt.kps = np.random.randn(5, 2).astype(np.float32)

    return src, tgt


# ---------------------------------------------------------------------------
# Pure-function tests (no cv2 needed)
# ---------------------------------------------------------------------------

class TestWarpTemplates:
    def test_all_5_point(self):
        from app.services.face_swapper import WARP_TEMPLATES
        for name, t in WARP_TEMPLATES.items():
            assert t.shape == (5, 2)


class TestBalanceEmbedding:
    def test_weight_1(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        r = _balance_embedding(src, tgt, 1.0, True)
        np.testing.assert_almost_equal(r[0, 0], 1.35, 4)

    def test_weight_0(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        r = _balance_embedding(src, tgt, 0.0, True)
        np.testing.assert_almost_equal(r[0, 0], 0.65, 4)

    def test_no_l2(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[3.0, 0.0]], dtype=np.float32)
        r = _balance_embedding(src, tgt, 0.5, False)
        np.testing.assert_almost_equal(r[0, 0], 1.0, 4)

    def test_zero_target(self):
        from app.services.face_swapper import _balance_embedding
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 0.0]], dtype=np.float32)
        r = _balance_embedding(src, tgt, 0.5, True)
        np.testing.assert_almost_equal(r[0, 0], 1.0, 4)


class TestPrepareCrop:
    def test_prepare(self):
        from app.services.face_swapper import _prepare_crop_frame
        f = np.ones((64, 64, 3), dtype=np.uint8) * 64
        r = _prepare_crop_frame(f, [0.25, 0.25, 0.25], [0.25, 0.25, 0.25])
        assert r.shape == (1, 3, 64, 64)


class TestNormalizeCrop:
    def test_tanh(self):
        from app.services.face_swapper import _normalize_crop_frame
        p = np.zeros((1, 3, 32, 32), dtype=np.float32)
        r = _normalize_crop_frame(p, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], True)
        assert r.shape == (32, 32, 3)

    def test_no_tanh(self):
        from app.services.face_swapper import _normalize_crop_frame
        p = np.zeros((1, 3, 32, 32), dtype=np.float32)
        r = _normalize_crop_frame(p, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], False)
        assert r.shape == (32, 32, 3)


class TestCreateBoxMask:
    def test_basic(self):
        from app.services.face_swapper import _create_box_mask
        m = _create_box_mask((64, 64), blur=0.0)
        assert m.dtype == np.float32
        assert m[32, 32] == 1.0

    def test_blur(self):
        from app.services.face_swapper import _create_box_mask
        m = _create_box_mask((128, 128), blur=0.5)
        assert m.shape == (128, 128)


# ---------------------------------------------------------------------------
# ONNX/cv2-dependent tests
# ---------------------------------------------------------------------------

class TestSwapperModel:
    def test_create(self, temp_onnx):
        from app.services.face_swapper import _SwapperModel
        m = _SwapperModel(temp_onnx)
        assert m.input_names
        assert m.output_names
        assert not m.input_swapped

    def test_missing(self):
        from app.services.face_swapper import _SwapperModel
        with pytest.raises(FileNotFoundError):
            _SwapperModel("/nonexistent/model.onnx")

    def test_hyperswap_detection(self):
        sess = mock.MagicMock()
        sess.get_inputs.return_value = [
            mock.MagicMock(shape=[1, 512]),
            mock.MagicMock(shape=[1, 3, 256, 256]),
        ]
        sess.get_outputs.return_value = [mock.MagicMock()]
        sys.modules["onnxruntime"].InferenceSession.return_value = sess

        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            from app.services.face_swapper import _SwapperModel
            m = _SwapperModel(path)
            assert m.input_swapped
        finally:
            os.remove(path)

    def test_too_few_inputs(self):
        sess = mock.MagicMock()
        sess.get_inputs.return_value = [mock.MagicMock()]
        sess.get_outputs.return_value = [mock.MagicMock()]
        sys.modules["onnxruntime"].InferenceSession.return_value = sess

        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            from app.services.face_swapper import _SwapperModel
            with pytest.raises(ValueError, match="expected at least 2"):
                _SwapperModel(path)
        finally:
            os.remove(path)

    def test_no_outputs(self):
        sess = mock.MagicMock()
        sess.get_inputs.return_value = [
            mock.MagicMock(shape=[1, 3, 128, 128]),
            mock.MagicMock(shape=[1, 512]),
        ]
        sess.get_outputs.return_value = []
        sys.modules["onnxruntime"].InferenceSession.return_value = sess

        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            from app.services.face_swapper import _SwapperModel
            with pytest.raises(ValueError, match="no outputs"):
                _SwapperModel(path)
        finally:
            os.remove(path)

    def test_empty_initializers(self):
        sys.modules["onnx"].load.return_value = mock.MagicMock(
            graph=mock.MagicMock(initializer=[]),
        )
        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            from app.services.face_swapper import _SwapperModel
            m = _SwapperModel(path)
            np.testing.assert_array_equal(m.emap, np.eye(1))
        finally:
            os.remove(path)


class TestModelCache:
    def test_caches(self):
        from app.services.face_swapper import get_face_swapper_model, FACE_SWAPPER_MODELS
        FACE_SWAPPER_MODELS.clear()
        with mock.patch("os.path.exists", return_value=True):
            m1 = get_face_swapper_model("inswapper_128")
            m2 = get_face_swapper_model("inswapper_128")
            assert m1 is m2


class TestConverterCache:
    def test_loads_and_caches(self):
        from app.services.face_swapper import _load_embedding_converter, EMBEDDING_CONVERTERS
        EMBEDDING_CONVERTERS.clear()
        with mock.patch("os.path.exists", return_value=True):
            c1 = _load_embedding_converter("crossface_ghost.onnx")
            c2 = _load_embedding_converter("crossface_ghost.onnx")
            assert c1 is c2

    def test_missing(self):
        from app.services.face_swapper import _load_embedding_converter
        with mock.patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                _load_embedding_converter("missing.onnx")


class TestWarpFace:
    def test_warps(self):
        from app.services.face_swapper import _warp_face_by_landmark_5
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        lm = np.array([[30, 40], [60, 40], [45, 55], [35, 70], [55, 70]], dtype=np.float32)
        r, _ = _warp_face_by_landmark_5(frame, lm, "arcface_128", (128, 128))
        assert r.shape[0] == 128


class TestPasteBack:
    def test_paste(self):
        from app.services.face_swapper import _paste_back
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = np.ones((128, 128, 3), dtype=np.uint8) * 200
        mask = np.ones((128, 128), dtype=np.float32)
        M = np.eye(2, 3, dtype=np.float32)
        r = _paste_back(frame, crop, mask, M)
        assert r.shape == frame.shape


class TestEmbeddingFuncs:
    def test_projected(self, mock_faces, temp_onnx):
        from app.services.face_swapper import _prepare_embedding_projected, _SwapperModel
        m = _SwapperModel(temp_onnx)
        r = _prepare_embedding_projected(mock_faces[0], m)
        assert r.shape == (1, 512)

    def test_raw(self, mock_faces):
        from app.services.face_swapper import _prepare_embedding_raw, EMBEDDING_CONVERTERS
        EMBEDDING_CONVERTERS.clear()
        conv = mock.MagicMock()
        conv.run.return_value = [mock_faces[0].embedding.reshape(1, -1)]
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch(
                "app.services.face_swapper._load_embedding_converter",
                return_value=conv,
            ):
                r = _prepare_embedding_raw(mock_faces[0], "conv.onnx")
                assert r.shape == (1, 512)

    def test_norm(self, mock_faces):
        from app.services.face_swapper import _prepare_embedding_norm
        r = _prepare_embedding_norm(mock_faces[0])
        assert r.shape == (1, 512)

    def test_source_face(self, mock_faces):
        from app.services.face_swapper import _prepare_source_face
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        r = _prepare_source_face(mock_faces[0], frame, 112)
        assert r.shape[0] == 1


class TestTransformPoints:
    def test_transform(self):
        from app.services.face_swapper import _transform_points
        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        M = np.eye(2, 3, dtype=np.float32)
        r = _transform_points(pts, M)
        assert r.shape == (2, 2)


class TestCalculatePasteArea:
    def test_area(self):
        from app.services.face_swapper import _calculate_paste_area
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = np.zeros((128, 128, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float32)
        bbox, Pm = _calculate_paste_area(frame, crop, M)
        assert len(bbox) == 4


class TestSwapFaceEnhanced:
    def test_inswapper(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "inswapper_128", (512, 512),
        )
        assert r.shape == frame.shape

    def test_simswap(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "simswap_256", (256, 256),
        )
        assert r.shape == frame.shape

    def test_hyperswap(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "hyperswap_1a_256", (256, 256),
        )
        assert r.shape == frame.shape

    def test_blendswap(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "blendswap_256", (256, 256),
        )
        assert r.shape == frame.shape

    def test_blendswap_weighted(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "blendswap_256", (256, 256), weight=0.5,
        )
        assert r.shape == frame.shape

    def test_ghost(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "ghost_1_256", (256, 256),
        )
        assert r.shape == frame.shape

    def test_hififace(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "hififace_unofficial_256", (256, 256),
        )
        assert r.shape == frame.shape

    def test_uniface(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        r = swap_face_enhanced(
            mock_faces[0], mock_faces[1], frame, m,
            "uniface_256", (256, 256),
        )
        assert r.shape == frame.shape

    def test_bad_model_type(self, mock_faces):
        from app.services.face_swapper import swap_face_enhanced
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            swap_face_enhanced(
                mock_faces[0], mock_faces[1], frame,
                "not_a_model", "inswapper_128", (512, 512),
            )

    def test_onnx_error(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel

        bad = mock.MagicMock()
        bad.run.side_effect = RuntimeError("VRAM full")
        bad.get_inputs.return_value = [
            mock.MagicMock(shape=[1, 3, 128, 128]),
            mock.MagicMock(shape=[1, 512]),
        ]
        bad.get_outputs.return_value = [mock.MagicMock()]
        sys.modules["onnxruntime"].InferenceSession.return_value = bad

        m = _SwapperModel(temp_onnx)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="ONNX inference failed"):
            swap_face_enhanced(
                mock_faces[0], mock_faces[1], frame, m,
                "inswapper_128", (512, 512),
            )

    def test_unknown_source_type(self, mock_faces, temp_onnx):
        from app.services.face_swapper import swap_face_enhanced, _SwapperModel
        m = _SwapperModel(temp_onnx)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        with mock.patch(
            "app.services.face_swapper.get_model_metadata",
            return_value={
                "native_size": (128, 128),
                "mean": [0, 0, 0], "std": [1, 1, 1],
                "tanh_out": False,
                "source_type": "unknown_type",
                "warp_template": "arcface_128",
            },
        ):
            with pytest.raises(ValueError, match="Unknown source_type"):
                swap_face_enhanced(
                    mock_faces[0], mock_faces[1], frame, m,
                    "inswapper_128", (128, 128),
                )

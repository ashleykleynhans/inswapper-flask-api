"""Shared pytest fixtures for InSwapper API tests."""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_heavy_imports():
    """Prevent heavy ML imports from loading during test collection."""
    with patch.dict(sys.modules, {
        "basicsr": MagicMock(),
        "basicsr.utils": MagicMock(),
        "basicsr.utils.imwrite": MagicMock(),
        "basicsr.utils.img2tensor": MagicMock(),
        "basicsr.utils.tensor2img": MagicMock(),
        "basicsr.utils.download_util": MagicMock(),
        "basicsr.utils.realesrgan_utils": MagicMock(),
        "basicsr.utils.registry": MagicMock(),
        "basicsr.archs": MagicMock(),
        "basicsr.archs.rrdbnet_arch": MagicMock(),
        "facelib": MagicMock(),
        "facelib.utils": MagicMock(),
        "facelib.utils.face_restoration_helper": MagicMock(),
        "facelib.utils.misc": MagicMock(),
        "torch": MagicMock(),
        "torch.nn": MagicMock(),
        "torch.nn.functional": MagicMock(),
        "torchvision": MagicMock(),
        "torchvision.transforms": MagicMock(),
        "torchvision.transforms.functional": MagicMock(),
        "insightface": MagicMock(),
        "insightface.app": MagicMock(),
        "insightface.model_zoo": MagicMock(),
        "insightface.utils": MagicMock(),
        "insightface.utils.face_align": MagicMock(),
        "onnx": MagicMock(),
        "onnxruntime": MagicMock(),
        "onnx.numpy_helper": MagicMock(),
    }):
        yield


@pytest.fixture
def mock_face_object():
    """Create a mock insightface Face object."""

    class MockBBox:
        def __init__(self):
            self.data = [100.0, 100.0, 200.0, 200.0]

        def __getitem__(self, idx):
            return self.data[idx]

    face = MagicMock()
    face.bbox = MockBBox()
    face.embedding = MagicMock()
    face.normed_embedding = MagicMock()
    face.kps = MagicMock()
    face.gender = 1
    face.age = 30
    face.det_score = 0.95
    return face


@pytest.fixture
def sample_request_payload():
    """Return a minimal valid face swap request payload."""
    return {
        "source_image": "/9j/fakejpegdata",
        "target_image": "/9j/faketargetdata",
    }

"""Tests for image_utils coverage gaps."""

import base64
import tempfile
import os
from unittest import mock

from PIL import Image
import numpy as np

from app.services.image_utils import (
    decode_base64_to_disk,
    encode_image_to_base64,
    encode_bgr_to_base64,
    load_images_from_paths,
    clean_up_temporary_files,
)


class TestDecodeBase64ToDisk:
    """Tests for decode_base64_to_disk."""

    def test_decode_jpeg(self):
        img = Image.new("RGB", (10, 10), color="red")
        buf = __import__("io").BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        path = decode_base64_to_disk(b64, "source")
        assert os.path.exists(path)
        assert "source_" in path
        os.remove(path)

    def test_decode_png(self):
        img = Image.new("RGB", (10, 10), color="blue")
        buf = __import__("io").BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        path = decode_base64_to_disk(b64, "target")
        assert os.path.exists(path)
        assert "target_" in path
        os.remove(path)

    def test_creates_tmp_dir(self):
        import shutil
        from app import config
        # Use a unique tmp path to test creation
        with mock.patch.object(config, "TMP_PATH", "/tmp/inswapper_test_decode"):
            img = Image.new("RGB", (5, 5), color="red")
            buf = __import__("io").BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            path = decode_base64_to_disk(b64, "src")
            assert os.path.exists(path)
            os.remove(path)
            # Clean up the test directory
            try:
                os.rmdir("/tmp/inswapper_test_decode")
            except OSError:
                pass


class TestEncodeBgrToBase64:
    """Tests for encode_bgr_to_base64."""

    def test_encode_bgr_jpeg(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :] = [255, 0, 0]  # BGR blue
        result = encode_bgr_to_base64(bgr, "JPEG")
        assert isinstance(result, str)
        # Re-decode to verify
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_bgr_png(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = encode_bgr_to_base64(bgr, "PNG")
        assert isinstance(result, str)


class TestLoadImagesFromPaths:
    """Tests for load_images_from_paths."""

    def test_single_source(self):
        src = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tgt = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            img = Image.new("RGB", (10, 10), color="red")
            img.save(src.name, format="JPEG")
            img.save(tgt.name, format="JPEG")

            sources, target = load_images_from_paths(src.name, tgt.name)
            assert len(sources) == 1
            assert isinstance(target, Image.Image)
        finally:
            os.remove(src.name)
            os.remove(tgt.name)

    def test_multiple_sources(self):
        src1 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        src2 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tgt = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            img = Image.new("RGB", (10, 10), color="red")
            img.save(src1.name, format="JPEG")
            img.save(src2.name, format="JPEG")
            img.save(tgt.name, format="JPEG")

            src_path = f"{src1.name};{src2.name}"
            sources, target = load_images_from_paths(src_path, tgt.name)
            assert len(sources) == 2
            assert isinstance(target, Image.Image)
        finally:
            os.remove(src1.name)
            os.remove(src2.name)
            os.remove(tgt.name)

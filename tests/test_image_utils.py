"""Tests for image utility functions."""

import os
import base64
import tempfile
from unittest.mock import patch

from PIL import Image
import numpy as np

from app.services.image_utils import (
    determine_file_extension,
    encode_image_to_base64,
    clean_up_temporary_files,
)


class TestDetermineFileExtension:
    """Tests for determine_file_extension."""

    def test_jpeg_header(self):
        assert determine_file_extension("/9j/abc123") == ".jpg"

    def test_png_header(self):
        assert determine_file_extension("iVBORw0Kgabc") == ".png"

    def test_unknown_header(self):
        assert determine_file_extension("unknown") == ".png"

    def test_empty_string(self):
        assert determine_file_extension("") == ".png"


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64."""

    def test_encode_jpeg(self):
        img = Image.new("RGB", (10, 10), color="red")
        result = encode_image_to_base64(img, "JPEG")
        assert isinstance(result, str)
        # Decode back to verify
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_png(self):
        img = Image.new("RGB", (10, 10), color="blue")
        result = encode_image_to_base64(img, "PNG")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


class TestCleanUpTemporaryFiles:
    """Tests for clean_up_temporary_files."""

    def test_removes_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        assert os.path.exists(path)
        clean_up_temporary_files(path)
        assert not os.path.exists(path)

    def test_silent_on_missing_file(self):
        # Should not raise
        clean_up_temporary_files("/nonexistent/path/abc.xyz")

    def test_silent_on_none(self):
        # Should not raise
        clean_up_temporary_files(None)

    def test_handles_multiple_paths(self):
        clean_up_temporary_files(None, "/nonexistent", None)
        # No assertion needed — just verifying no exception

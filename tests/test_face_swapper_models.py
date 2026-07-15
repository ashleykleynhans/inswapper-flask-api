"""Tests for face swapper models — ported from runpod-worker-inswapper."""

import pytest

from app.services.face_swapper_models import (
    FACE_SWAPPER_MODEL_SET,
    validate_face_swapper_params,
    get_default_resolution,
    parse_resolution,
    get_model_metadata,
)


class TestValidateFaceSwapperParams:
    """Tests for validate_face_swapper_params."""

    def test_valid_model_and_resolution(self):
        # Should not raise
        validate_face_swapper_params("inswapper_128", "512x512")

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="Invalid face_swapper_model"):
            validate_face_swapper_params("nonexistent_model", "512x512")

    def test_invalid_resolution_for_model(self):
        with pytest.raises(ValueError, match="does not support resolution"):
            validate_face_swapper_params("simswap_256", "128x128")


class TestGetDefaultResolution:
    """Tests for get_default_resolution."""

    def test_inswapper_default(self):
        assert get_default_resolution("inswapper_128") == "512x512"

    def test_other_model_default(self):
        assert get_default_resolution("blendswap_256") == "1024x1024"

    def test_nonexistent_model_default(self):
        assert get_default_resolution("unknown") == "1024x1024"


class TestParseResolution:
    """Tests for parse_resolution."""

    def test_valid(self):
        assert parse_resolution("512x512") == (512, 512)

    def test_valid_uppercase(self):
        assert parse_resolution("256X256") == (256, 256)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution("512")

    def test_invalid_separator(self):
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution("512-512")

    def test_none_input(self):
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution(None)


class TestGetModelMetadata:
    """Tests for get_model_metadata."""

    def test_inswapper_metadata(self):
        meta = get_model_metadata("inswapper_128")
        assert meta["native_size"] == (128, 128)
        assert meta["source_type"] == "embedding_projected"

    def test_simswap_metadata(self):
        meta = get_model_metadata("simswap_256")
        assert meta["source_type"] == "embedding"

    def test_unknown_model(self):
        with pytest.raises(KeyError):
            get_model_metadata("nonexistent")

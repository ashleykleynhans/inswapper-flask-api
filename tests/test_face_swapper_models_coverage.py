"""Tests for face_swapper_models coverage gaps."""

import pytest

from app.services.face_swapper_models import (
    FACE_SWAPPER_MODEL_SET,
    MODEL_METADATA,
    get_default_resolution,
    parse_resolution,
    get_model_metadata,
)


class TestModelSetIntegrity:
    """Structural tests for model definitions."""

    def test_all_models_have_metadata(self):
        for model in FACE_SWAPPER_MODEL_SET:
            get_model_metadata(model)  # must not raise

    def test_inswapper_fp16_is_in_set(self):
        assert "inswapper_128_fp16" in FACE_SWAPPER_MODEL_SET

    def test_simswap_unofficial_default(self):
        assert get_default_resolution("simswap_unofficial_512") == "512x512"

    def test_parse_resolution_error_on_none(self):
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution(None)

    def test_parse_resolution_error_on_empty(self):
        with pytest.raises(ValueError):
            parse_resolution("")

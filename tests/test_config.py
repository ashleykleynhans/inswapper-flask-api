"""Tests for app config."""

import os
import sys
import logging
from unittest import mock

from app import config


class TestConfigPaths:
    """Tests for config path constants."""

    def test_base_dir_is_absolute(self):
        assert os.path.isabs(config.BASE_DIR)

    def test_checkpoints_dir(self):
        assert config.CHECKPOINTS_DIR.name == "checkpoints"

    def test_face_swapper_models_dir(self):
        assert config.FACE_SWAPPER_MODELS_DIR.name == "face_swapper"

    def test_tmp_path(self):
        assert config.TMP_PATH == "/tmp/inswapper"

    def test_defaults_dict_has_all_keys(self):
        assert config.DEFAULTS["source_indexes"] == "-1"
        assert config.DEFAULTS["face_swapper_model"] == "inswapper_128"
        assert config.DEFAULTS["codeformer_fidelity"] == 0.5


class TestLogging:
    """Tests for logging configuration."""

    def test_init_logging_linux(self):
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch("logging.basicConfig") as bc:
                config.init_logging()
                bc.assert_called_once()

    def test_init_logging_macos(self):
        with mock.patch.object(sys, "platform", "darwin"):
            with mock.patch("logging.basicConfig") as bc:
                config.init_logging()
                bc.assert_called_once()

    def test_log_path_is_defined(self):
        assert isinstance(config.LOG_PATH, str)

    def test_log_file_is_defined(self):
        assert isinstance(config.LOG_FILE, str)
        assert config.LOG_FILE.endswith("inswapper.log")


class TestTimer:
    """Tests for Timer utility."""

    def test_timer_creation(self):
        t = config.Timer()
        assert t.start > 0

    def test_timer_restart(self):
        t = config.Timer()
        original = t.start
        import time
        time.sleep(0.01)
        t.restart()
        assert t.start > original

    def test_timer_elapsed(self):
        t = config.Timer()
        elapsed = t.get_elapsed_time()
        assert elapsed >= 0.0

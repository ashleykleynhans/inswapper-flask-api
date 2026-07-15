"""Tests for face selector — ported from runpod-worker-inswapper."""

import pytest
from unittest.mock import MagicMock

from app.services.face_selector import select_faces


def _make_face(x, y, w, h, gender=1, age=30, det_score=0.9):
    """Create a mock face with the given bbox and attributes."""
    face = MagicMock()
    face.bbox = [float(x), float(y), float(x + w), float(y + h)]
    face.gender = gender
    face.age = age
    face.det_score = det_score
    return face


class TestSelectFaces:
    """Tests for select_faces."""

    def test_empty_list(self):
        assert select_faces([]) == []

    def test_default_left_right_sort(self):
        faces = [
            _make_face(300, 100, 50, 50),
            _make_face(100, 100, 50, 50),
            _make_face(200, 100, 50, 50),
        ]
        result = select_faces(faces)
        assert [f.bbox[0] for f in result] == [100.0, 200.0, 300.0]

    def test_right_left_sort(self):
        faces = [
            _make_face(100, 100, 50, 50),
            _make_face(300, 100, 50, 50),
            _make_face(200, 100, 50, 50),
        ]
        result = select_faces(faces, order="right-left")
        assert [f.bbox[0] for f in result] == [300.0, 200.0, 100.0]

    def test_large_small_sort(self):
        faces = [
            _make_face(100, 100, 30, 30),
            _make_face(200, 100, 60, 60),
            _make_face(300, 100, 40, 40),
        ]
        result = select_faces(faces, order="large-small")
        areas = [
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            for f in result
        ]
        assert areas == sorted(areas, reverse=True)

    def test_gender_filter_male(self):
        faces = [
            _make_face(100, 100, 50, 50, gender=1),  # male
            _make_face(200, 100, 50, 50, gender=0),  # female
        ]
        result = select_faces(faces, gender="male")
        assert len(result) == 1
        assert result[0].bbox[0] == 100.0

    def test_age_filter(self):
        faces = [
            _make_face(100, 100, 50, 50, age=25),
            _make_face(200, 100, 50, 50, age=35),
            _make_face(300, 100, 50, 50, age=45),
        ]
        result = select_faces(faces, age_start=30, age_end=40)
        assert len(result) == 1
        assert result[0].age == 35

    def test_mode_one(self):
        faces = [
            _make_face(200, 100, 50, 50),
            _make_face(100, 100, 50, 50),
        ]
        result = select_faces(faces, mode="one")
        assert len(result) == 1

    def test_best_worst_sort(self):
        faces = [
            _make_face(100, 100, 50, 50, det_score=0.7),
            _make_face(200, 100, 50, 50, det_score=0.95),
            _make_face(300, 100, 50, 50, det_score=0.5),
        ]
        result = select_faces(faces, order="best-worst")
        assert result[0].det_score == 0.95
        assert result[-1].det_score == 0.5

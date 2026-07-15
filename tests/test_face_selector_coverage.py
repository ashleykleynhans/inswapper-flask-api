"""Tests for face_selector — fill remaining coverage gaps."""

from unittest.mock import MagicMock

import pytest

from app.services.face_selector import select_faces


def _make_face(x, y, w, h, gender=1, age=30, det_score=0.9):
    """Create a mock face with the given bbox and attributes."""
    face = MagicMock()
    face.bbox = [float(x), float(y), float(x + w), float(y + h)]
    face.gender = gender
    face.age = age
    face.det_score = det_score
    return face


class TestSelectFacesCoverage:
    """Fill coverage gaps in face_selector."""

    def test_top_bottom_sort(self):
        faces = [
            _make_face(100, 300, 50, 50),
            _make_face(100, 100, 50, 50),
            _make_face(100, 200, 50, 50),
        ]
        result = select_faces(faces, order="top-bottom")
        assert [f.bbox[1] for f in result] == [100.0, 200.0, 300.0]

    def test_small_large_sort(self):
        faces = [
            _make_face(100, 100, 60, 60),
            _make_face(200, 100, 30, 30),
            _make_face(300, 100, 40, 40),
        ]
        result = select_faces(faces, order="small-large")
        areas = [
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            for f in result
        ]
        assert areas == sorted(areas)

    def test_worst_best_sort(self):
        faces = [
            _make_face(100, 100, 50, 50, det_score=0.7),
            _make_face(200, 100, 50, 50, det_score=0.95),
            _make_face(300, 100, 50, 50, det_score=0.5),
        ]
        result = select_faces(faces, order="worst-best")
        assert result[0].det_score == 0.5
        assert result[-1].det_score == 0.95

    def test_gender_filter_female(self):
        faces = [
            _make_face(100, 100, 50, 50, gender=1),  # male
            _make_face(200, 100, 50, 50, gender=0),  # female
        ]
        result = select_faces(faces, gender="female")
        assert len(result) == 1
        assert result[0].bbox[0] == 200.0

    def test_no_gender_filter_keeps_all(self):
        faces = [
            _make_face(100, 100, 50, 50, gender=1),
            _make_face(200, 100, 50, 50, gender=0),
        ]
        result = select_faces(faces)
        assert len(result) == 2

    def test_empty_after_filter(self):
        faces = [
            _make_face(100, 100, 50, 50, gender=1),
        ]
        result = select_faces(faces, gender="female")
        assert result == []

    def test_empty_after_age_filter(self):
        faces = [
            _make_face(100, 100, 50, 50, age=25),
        ]
        result = select_faces(faces, age_start=50)
        assert result == []

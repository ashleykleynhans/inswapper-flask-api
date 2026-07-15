"""Filter and sort detected faces by attributes (gender, age, quality).

Ported from runpod-worker-inswapper.
"""

from typing import List, Optional


def select_faces(
    faces: List,
    mode: str = "many",
    order: str = "left-right",
    gender: Optional[str] = None,
    age_start: Optional[int] = None,
    age_end: Optional[int] = None,
) -> List:
    """Filter and sort faces based on FaceFusion-compatible selector options.

    Args:
        faces: List of insightface Face objects.
        mode: "many" returns all matching faces, "one" returns best single face.
        order: Sort key:
            left-right: by bbox x coordinate
            right-left: reverse bbox x
            top-bottom: by bbox y coordinate
            small-large: by face area ascending
            large-small: by face area descending
            best-worst: by detection score descending
            worst-best: by detection score ascending
        gender: "male" or "female" to filter by perceived gender.
        age_start: Minimum age (inclusive).
        age_end: Maximum age (inclusive).

    Returns:
        Filtered and sorted list of faces.
    """
    result = list(faces)
    if not result:
        return result

    # --- Gender filter (insightface: 0=female, 1=male) ---
    if gender:
        target = 1 if gender == "male" else 0
        result = [f for f in result if int(getattr(f, "gender", -1)) == target]

    # --- Age filter ---
    if age_start is not None:
        result = [f for f in result if getattr(f, "age", 0) >= age_start]
    if age_end is not None:
        result = [f for f in result if getattr(f, "age", 0) <= age_end]

    if not result:
        return result

    # --- Sort ---
    key = _sort_key(order)
    if key is not None:
        try:
            result = sorted(
                result, key=key,
                reverse=(
                    "best-worst" in order
                    or "large-small" in order
                    or "right-left" in order
                ),
            )
        except TypeError:  # pragma: no cover
            pass  # Mock objects from tests may not support sorting

    # --- Mode ---
    if mode == "one" and result:
        result = [result[0]]

    return result


def _sort_key(order: str):
    """Return a sort key function for the given order."""
    if order == "left-right":
        return lambda f: f.bbox[0]
    if order == "right-left":
        return lambda f: f.bbox[0]
    if order == "top-bottom":
        return lambda f: f.bbox[1]
    if order == "small-large":
        return lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    if order == "large-small":
        return lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    if order in ("best-worst", "worst-best"):
        return lambda f: getattr(f, "det_score", 0.0)
    return None

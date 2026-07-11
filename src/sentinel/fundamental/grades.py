"""Letter-grade scale shared by all scorecard rows.

Scores are 0-100; letters follow the usual GPA-style cut points so composite
grades can be computed by averaging numeric scores and mapping back.
"""

from __future__ import annotations

GRADES: tuple[str, ...] = (
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F",
)

_CUTS: tuple[tuple[float, str], ...] = (
    (97, "A+"), (93, "A"), (90, "A-"),
    (87, "B+"), (83, "B"), (80, "B-"),
    (77, "C+"), (73, "C"), (70, "C-"),
    (67, "D+"), (60, "D"),
)


def letter_grade(score: float) -> str:
    for cut, letter in _CUTS:
        if score >= cut:
            return letter
    return "F"


def grade_points(letter: str) -> float:
    """Midpoint score for a letter, for round-tripping composite averages."""
    if letter == "A+":
        return 98.0
    if letter == "F":
        return 50.0
    idx = GRADES.index(letter)
    upper = 100.0 if idx == 0 else _CUTS[idx - 1][0]
    lower = _CUTS[idx][0]
    return (upper + lower) / 2


def notch(letter: str, steps: int) -> str:
    """Shift a grade up (positive) or down (negative) by whole notches."""
    idx = GRADES.index(letter)
    return GRADES[min(max(idx - steps, 0), len(GRADES) - 1)]

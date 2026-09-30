from __future__ import annotations


def expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def update(a: float, b: float, score_a: float, k: float = 24.0) -> tuple[float, float]:
    ea = expected(a, b)
    eb = 1.0 - ea
    return a + k * (score_a - ea), b + k * ((1.0 - score_a) - eb)

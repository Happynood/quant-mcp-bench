from __future__ import annotations

from quantmcp.metrics.stats import bootstrap_ci


def test_ci_basic():
    data = [1.0, 0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.72]
    lo, hi = bootstrap_ci(data, n_resamples=500, confidence=0.95, seed=42)
    assert lo <= sum(data) / len(data) <= hi


def test_ci_single_value():
    lo, hi = bootstrap_ci([1.0], n_resamples=200, confidence=0.95, seed=0)
    assert lo == hi == 1.0


def test_ci_lower_lt_upper():
    data = [0.5, 0.6, 0.7, 0.8, 0.55, 0.65]
    lo, hi = bootstrap_ci(data, n_resamples=200, confidence=0.95, seed=1)
    assert lo <= hi


def test_ci_90_narrower_than_95():
    data = [0.5, 0.6, 0.7, 0.8, 0.55, 0.65, 0.75, 0.85]
    lo95, hi95 = bootstrap_ci(data, n_resamples=500, confidence=0.95, seed=7)
    lo90, hi90 = bootstrap_ci(data, n_resamples=500, confidence=0.90, seed=7)
    assert (hi90 - lo90) <= (hi95 - lo95) + 1e-6


def test_ci_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        bootstrap_ci([], n_resamples=100, confidence=0.95, seed=0)

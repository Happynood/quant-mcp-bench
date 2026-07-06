from __future__ import annotations

import pytest

from quantmcp.metrics.correlation import spearman_correlation


def test_spearman_perfect_positive_correlation():
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_no_correlation_for_symmetric_pattern():
    # y is a "V" shape against x: no monotonic relationship either way.
    assert spearman_correlation([1, 2, 3, 4, 5], [3, 1, 0, 1, 3]) == pytest.approx(0.0)


def test_spearman_handles_ties_via_average_rank():
    # x has a tied pair (rank 2.5 each), y has a tied pair (rank 3.5 each);
    # hand-computed expected rho = 0.5 (see rank-Pearson derivation).
    rho = spearman_correlation([1, 2, 2, 3], [2, 1, 3, 3])
    assert rho == pytest.approx(0.5)


def test_spearman_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman_correlation([1, 2, 3], [1, 2])


def test_spearman_raises_on_fewer_than_two_points():
    with pytest.raises(ValueError):
        spearman_correlation([1], [1])

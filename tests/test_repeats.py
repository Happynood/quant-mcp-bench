from __future__ import annotations

import pytest

from quantmcp.report.repeats import aggregate_repeats


def test_aggregate_repeats_computes_mean_and_range():
    results = [
        {"svr_mcp": 0.8, "tsr": 0.6},
        {"svr_mcp": 1.0, "tsr": 0.7},
        {"svr_mcp": 0.6, "tsr": 0.5},
    ]
    stability = aggregate_repeats(results)
    assert stability.n_repeats == 3
    assert stability.svr_mcp_mean == pytest.approx(0.8)
    assert stability.svr_mcp_min == pytest.approx(0.6)
    assert stability.svr_mcp_max == pytest.approx(1.0)
    assert stability.tsr_mean == pytest.approx(0.6)
    assert stability.tsr_min == pytest.approx(0.5)
    assert stability.tsr_max == pytest.approx(0.7)


def test_aggregate_repeats_single_result_has_zero_range():
    stability = aggregate_repeats([{"svr_mcp": 0.5, "tsr": 0.4}])
    assert stability.n_repeats == 1
    assert stability.svr_mcp_min == stability.svr_mcp_max == pytest.approx(0.5)


def test_aggregate_repeats_raises_on_empty_list():
    with pytest.raises(ValueError):
        aggregate_repeats([])

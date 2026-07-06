from __future__ import annotations

from quantmcp.metrics.deltas import compute_delta, compute_delta_rel, compute_eta


def test_delta_basic():
    d = compute_delta(baseline=0.9, current=0.7)
    assert abs(d - 0.2) < 1e-9


def test_delta_zero():
    assert compute_delta(0.8, 0.8) == 0.0


def test_delta_negative_improvement():
    assert compute_delta(0.7, 0.9) < 0


def test_delta_rel_basic():
    d = compute_delta_rel(baseline=0.8, current=0.6)
    assert d is not None
    assert abs(d - 0.25) < 1e-9


def test_delta_rel_zero_baseline():
    d = compute_delta_rel(baseline=0.0, current=0.5)
    assert d is None


def test_eta_basic():
    eta = compute_eta(fcr=0.8, peak_vram_gb=8.0)
    assert eta is not None
    assert abs(eta - 0.1) < 1e-9


def test_eta_zero_vram():
    assert compute_eta(fcr=0.8, peak_vram_gb=0.0) is None


def test_eta_none_vram():
    assert compute_eta(fcr=0.8, peak_vram_gb=None) is None

from __future__ import annotations

from quantmcp.report.pareto import pareto_front


def test_basic_pareto():
    points = [
        {"model": "A", "fcr": 0.9, "vram_gb": 8.0},
        {"model": "B", "fcr": 0.8, "vram_gb": 6.0},
        {"model": "C", "fcr": 0.7, "vram_gb": 4.0},
        {"model": "D", "fcr": 0.6, "vram_gb": 8.0},
    ]
    front = pareto_front(points, x_key="vram_gb", y_key="fcr", minimize_x=True, maximize_y=True)
    names = {p["model"] for p in front}
    assert "C" in names
    assert "D" not in names


def test_all_on_front():
    points = [
        {"model": "A", "fcr": 0.9, "vram_gb": 10.0},
        {"model": "B", "fcr": 0.7, "vram_gb": 6.0},
        {"model": "C", "fcr": 0.5, "vram_gb": 3.0},
    ]
    front = pareto_front(points, x_key="vram_gb", y_key="fcr", minimize_x=True, maximize_y=True)
    assert len(front) == 3


def test_single_point():
    points = [{"model": "X", "fcr": 0.8, "vram_gb": 8.0}]
    front = pareto_front(points, x_key="vram_gb", y_key="fcr", minimize_x=True, maximize_y=True)
    assert len(front) == 1


def test_empty_input():
    assert pareto_front([], x_key="vram_gb", y_key="fcr") == []

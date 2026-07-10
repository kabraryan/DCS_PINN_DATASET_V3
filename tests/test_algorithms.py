from __future__ import annotations

import numpy as np
import pytest

from benchmark.algorithms import REGISTRY
from benchmark.algorithms.base import Algorithm, AlgorithmError
from benchmark.profile import Dive, reconstruct


def dive(depth=100.0, bt=60.0, at=30.0, did="d1") -> Dive:
    return Dive(did, depth, bt, at, 0.0, "TRIAL")


def test_registry_contains_zhl16c():
    assert "zhl16c" in REGISTRY
    assert isinstance(REGISTRY["zhl16c"], Algorithm)


def test_zhl16c_risk_index_is_finite_and_positive():
    algo = REGISTRY["zhl16c"]
    p = reconstruct(dive(), "staged")
    r = algo.risk_index(p, dive())
    assert np.isfinite(r) and r > 0


def test_zhl16c_risk_index_rises_with_bottom_time():
    algo = REGISTRY["zhl16c"]
    short, long_ = dive(bt=10.0), dive(bt=90.0)
    r_s = algo.risk_index(reconstruct(short, "linear"), short)
    r_l = algo.risk_index(reconstruct(long_, "linear"), long_)
    assert r_l > r_s


def test_zhl16c_deficit_positive_when_ascent_too_fast():
    algo = REGISTRY["zhl16c"]
    d = dive(depth=200.0, bt=40.0, at=1.0)      # nowhere near enough deco
    assert algo.deficit(reconstruct(d, "staged"), d) > 0


def test_zhl16c_deficit_negative_or_zero_when_ascent_generous():
    algo = REGISTRY["zhl16c"]
    d = dive(depth=40.0, bt=10.0, at=200.0)     # far more deco than demanded
    assert algo.deficit(reconstruct(d, "staged"), d) <= 0


def test_zhl16c_deficit_is_a_float_not_none():
    algo = REGISTRY["zhl16c"]
    d = dive()
    assert isinstance(algo.deficit(reconstruct(d, "staged"), d), float)


def test_risk_index_has_variance_over_real_profiles():
    """The 15-line test that would have killed the degenerate bubble model."""
    algo = REGISTRY["zhl16c"]
    vals = []
    for i, (depth, bt, at) in enumerate([
        (60, 20, 5), (80, 30, 8), (100, 40, 12), (120, 25, 20), (140, 35, 30),
        (70, 55, 6), (90, 45, 15), (110, 15, 25), (130, 50, 40), (150, 20, 45),
        (55, 70, 4), (85, 60, 10), (105, 20, 18), (125, 30, 28), (145, 40, 38),
        (65, 25, 7), (95, 35, 14), (115, 45, 22), (135, 55, 33), (155, 65, 50),
    ]):
        d = dive(depth, bt, at, did=f"d{i}")
        vals.append(algo.risk_index(reconstruct(d, "staged"), d))
    assert np.std(vals) > 1e-9, "constant risk_index means the model is degenerate"


def test_unknown_algorithm_raises():
    with pytest.raises(KeyError):
        REGISTRY["does_not_exist"]

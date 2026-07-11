from __future__ import annotations

import numpy as np
import pytest

from benchmark.algorithms import REGISTRY
from benchmark.profile import Dive, reconstruct


def dive(depth=120.0, bt=40.0, at=15.0, did="d1") -> Dive:
    return Dive(did, depth, bt, at, 0.0, "TRIAL")


def test_registered_and_deficit_is_none():
    algo = REGISTRY["ep_bubble"]
    d = dive()
    assert algo.deficit(reconstruct(d, "staged"), d) is None, \
        "EP defines no ceiling; giving it a schedule IS VPM-B"


def test_r_max_never_below_r0_because_of_the_skin():
    algo = REGISTRY["ep_bubble"]
    d = dive(depth=20.0, bt=5.0, at=60.0)     # benign: bubble should not grow
    r = algo.risk_index(reconstruct(d, "staged"), d)
    assert r >= algo.params["r0_um"] - 1e-9


def test_r_max_respects_the_growth_ceiling():
    algo = REGISTRY["ep_bubble"]
    d = dive(depth=280.0, bt=200.0, at=1.0)   # extreme
    assert algo.risk_index(reconstruct(d, "staged"), d) <= algo.params["ceiling_um"] + 1e-6


def test_r_max_rises_with_supersaturation():
    algo = REGISTRY["ep_bubble"]
    mild, severe = dive(depth=60.0, bt=20.0, at=40.0), dive(depth=220.0, bt=90.0, at=3.0)
    assert algo.risk_index(reconstruct(severe, "staged"), severe) > \
           algo.risk_index(reconstruct(mild, "staged"), mild)


def test_risk_index_has_variance_over_20_real_profiles():
    """Correction 11 in 15 lines. Without the skin this is exactly zero."""
    algo = REGISTRY["ep_bubble"]
    vals = []
    for i, (depth, bt, at) in enumerate([
        (60, 20, 5), (80, 30, 8), (100, 40, 12), (120, 25, 20), (140, 35, 30),
        (70, 55, 6), (90, 45, 15), (110, 15, 25), (130, 50, 40), (150, 20, 45),
        (55, 70, 4), (85, 60, 10), (105, 20, 18), (125, 30, 28), (145, 40, 38),
        (65, 25, 7), (95, 35, 14), (115, 45, 22), (135, 55, 33), (155, 65, 50),
    ]):
        d = dive(depth, bt, at, did=f"d{i}")
        vals.append(algo.risk_index(reconstruct(d, "staged"), d))
    assert np.std(vals) > 1e-9, "degenerate: the EP model never grows a bubble"


def test_trajectory_stable_under_tolerance_refinement():
    """Refining rtol must not move the trajectory. Tested at the PRODUCTION r0 (0.7 um)
    on a growing dive, comparing the PRE-saturation trajectory point-by-point.

    An earlier version compared .max() at r0=4 um, where every trajectory saturates
    to exactly the ceiling -- so the relative difference was 0.0 unconditionally and
    the test could not fail. That is the flavour of can't-fail test this project keeps
    finding; here the pre-saturation points genuinely depend on step size.
    """
    from benchmark.algorithms.ep_bubble import integrate_bubble
    ceiling_m = 100e-6
    d = dive(depth=220.0, bt=90.0, at=3.0)      # grows to the ceiling
    p = reconstruct(d, "staged")
    coarse = integrate_bubble(p, r0_m=0.7e-6, rtol=1e-6)
    fine = integrate_bubble(p, r0_m=0.7e-6, rtol=1e-8)

    # Compare only the strictly-pre-ceiling portion: once either hits the ceiling the
    # values are clamped equal and carry no information about step-size sensitivity.
    below = (coarse < ceiling_m * 0.999) & (fine < ceiling_m * 0.999)
    k = int(np.argmax(~below)) if (~below).any() else len(below)
    assert k > 3, "expected a growing, non-degenerate pre-saturation trajectory"
    rel = np.abs(coarse[:k] - fine[:k]) / np.maximum(fine[:k], 1e-12)
    assert rel.max() < 1e-3, f"trajectory unstable under rtol refinement: {rel.max():.2e}"

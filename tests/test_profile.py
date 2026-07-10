from __future__ import annotations

import json
import numpy as np
import pytest

from benchmark.profile import Dive, Profile, reconstruct, RECONSTRUCTIONS

REAL_JSONL = "/Users/aryankabra_test/Desktop/FINAL DIVE/datasets/real/dcs_real_cases.jsonl"


def make(depth=100.0, bt=60.0, at=30.0) -> Dive:
    return Dive("d1", depth, bt, at, 0.0, "TRIAL")


@pytest.mark.parametrize("recon", RECONSTRUCTIONS)
def test_time_strictly_increasing(recon):
    p = reconstruct(make(), recon)
    assert np.all(np.diff(p.t_min) > 0), "solve_ivp rejects unsorted t_eval"


@pytest.mark.parametrize("recon", RECONSTRUCTIONS)
def test_depth_non_negative_and_reaches_max(recon):
    p = reconstruct(make(depth=120.0), recon)
    assert p.depth_fsw.min() >= 0.0
    assert p.depth_fsw.max() == pytest.approx(120.0, rel=1e-6)


@pytest.mark.parametrize("recon", RECONSTRUCTIONS)
def test_realised_ascent_equals_recorded(recon):
    """The reconstruction supplies the SHAPE; the data supplies the DURATION."""
    dive = make(depth=150.0, bt=40.0, at=60.0)
    p = reconstruct(dive, recon)
    from benchmark.buhlmann import DESCENT_FSW_PER_MIN
    t_end_bottom = max(dive.depth_fsw / DESCENT_FSW_PER_MIN, 0.5) + dive.bottom_time_min
    surfaced = np.where((p.t_min > t_end_bottom) & (p.depth_fsw <= 1e-9))[0]
    assert len(surfaced) > 0
    realised = p.t_min[surfaced[0]] - t_end_bottom
    assert realised == pytest.approx(dive.ascent_time_min, abs=0.51)


def test_staged_has_stops_and_linear_does_not():
    dive = make(depth=150.0, bt=40.0, at=60.0)
    lin = reconstruct(dive, "linear")
    stg = reconstruct(dive, "staged")

    def n_flat_below_max(p):
        d = p.depth_fsw
        interior = d[(d > 1.0) & (d < d.max() - 1.0)]
        return int(np.sum(np.abs(np.diff(interior)) < 1e-9))

    assert n_flat_below_max(stg) > n_flat_below_max(lin)


def test_fast_ascent_falls_back_to_straight_and_flags_it():
    """at_min shorter than the travel legs alone -> straight ascent, flagged."""
    p = reconstruct(make(depth=60.0, bt=50.0, at=0.5), "staged")
    assert "straight_ascent_fallback" in p.flags


def test_staged_beats_linear_on_the_real_curves():
    """Ground truth: 428 real DCS cases carry both the 3 scalars and the true curve."""
    recs = [json.loads(l) for l in open(REAL_JSONL) if l.strip()]
    errs_lin, errs_stg = [], []
    for r in recs:
        s = r.get("depth_time_series") or []
        if len(s) < 8 or (r.get("quality_flags") or ""):
            continue
        if not all(r.get(k) and 0 < r[k] <= 300 for k in
                   ("max_depth_fsw", "bottom_time_min", "ascent_time_min")):
            continue
        t = np.array([q["t_min"] for q in s], float)
        d = np.array([q["depth_fsw"] for q in s], float)
        if np.any(np.diff(t) <= 0):
            continue
        dive = Dive(r.get("profile_number", "x"), r["max_depth_fsw"],
                    r["bottom_time_min"], r["ascent_time_min"], 1.0, r["data_set"])
        for recon, acc in (("linear", errs_lin), ("staged", errs_stg)):
            p = reconstruct(dive, recon)
            hi = min(t[-1], p.t_min[-1])
            m = t <= hi
            if m.sum() < 5:
                continue
            acc.append(np.sqrt(np.mean((np.interp(t[m], p.t_min, p.depth_fsw) - d[m]) ** 2)))

    assert len(errs_stg) >= 50, "expected ~72 usable gold curves"
    assert np.median(errs_stg) < np.median(errs_lin)
    assert np.median(errs_stg) < 40.0    # measured 36.08 fsw

# Decompression Algorithm Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable instrument that scores decompression algorithms against 2,700 real Navy dive outcomes under a protocol that cannot be fooled by trial leakage, sign-inverted confounds, or profile-reconstruction ambiguity.

**Architecture:** Algorithms are pure functions of a dive profile behind a `Protocol` and a registry — none of them train. Two cache boundaries (profile reconstruction, then per-algorithm scalars) make the 2×3 sensitivity grid cost the same as one cell. Evaluation is nested grouped cross-validation with four independent gates, and a finding counts only if it survives both profile reconstructions.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`solve_ivp` RK45 + terminal events), scikit-learn (`GroupKFold`, `LogisticRegression`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-decompression-benchmark-design.md`
**Working directory (all commands):** `~/Desktop/DCS_PINN_DATASET_V3`

---

## Global Constraints

- **Interpreter:** `/opt/miniconda3/bin/python3` (3.13.13). `/usr/bin/python3` is 3.9.6 and **will fail** — `float | None` raises `TypeError` at runtime under 3.9. Every module starts with `from __future__ import annotations` regardless, and public signatures use `Optional[float]`, never `float | None`.
- **NumPy ≥ 2.0.** `np.trapz` was removed; use `np.trapezoid`.
- **No pickle.** Never `joblib.dump` / `joblib.load`. Cache and scaler parameters serialise as JSON or `.npy`. `joblib` is not a dependency.
- **ODE solver:** `RK45` with a terminal event. Never `Radau` — it fails on 70% of realistic profiles (Correction 12). Never a bare `if R <= 0: return [0.0]` guard; it makes the RHS non-smooth.
- **Interpolation inside an ODE RHS:** `np.interp` closure, never `scipy.interpolate.interp1d` (7.3 µs vs 0.6 µs per call, paid ~800× per solve).
- **The benchmark never emits a probability.** No `predict_proba` on any public surface. `risk_index` is a rank. `Brier` may be printed but must carry the label `not calibration`.
- **Grouped evaluation is mandatory.** Every cross-validation splits on `data_set`. Ordinary CV inflates AUC by +0.045 to +0.075.
- **Marginal rule is explicit.** `--marginal` has no default that silently applies; the CLI requires it or errors.
- **Every number in `RESULTS.md` is generated.** Hand-transcribed statistics are forbidden (Corrections 9, 12).
- **ZHL-16C `b` column is strictly increasing, `a` strictly decreasing.** Asserted at construction. Compartment 16 is `b = 0.9653`, not `0.8693` (Correction 10).

---

## Spec deviation, decided here

The spec's Success Criterion 5 says a trivial `constant` algorithm "must score `NOT SUPPORTED`". It cannot: a constant column has zero variance and the error-handling rule raises `RuntimeError` on exactly that (Correction 11's guard). The two rules contradict.

**Resolution:** the extensibility demo uses a `noise` algorithm — a deterministic hash of `dive_id` mapped to a float. It has variance, carries no signal, and must score `NOT SUPPORTED`. A separate test asserts `constant` raises `RuntimeError`. Task 10 amends the spec to say so.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `benchmark/__init__.py` | package marker, version |
| `benchmark/buhlmann.py` | ZHL-16C table, Haldane step, M-value, gradient-factor ceiling, required-ascent schedule. Knows nothing about algorithms or labels. |
| `benchmark/profile.py` | `Dive`, `Profile`; `reconstruct(dive, recon)` for `linear` and `staged`. Knows nothing about algorithms. |
| `benchmark/cache.py` | content-addressed JSON cache keyed by SHA-256 of inputs |
| `benchmark/algorithms/base.py` | `Algorithm` Protocol, `AlgorithmError` |
| `benchmark/algorithms/zhl16c.py` | plain Bühlmann (gf = 1.0/1.0) |
| `benchmark/algorithms/zhl16c_gf.py` | gradient factors, default 30/70 |
| `benchmark/algorithms/ep_bubble.py` | Epstein-Plesset + VPM skin; `deficit` is `None` |
| `benchmark/algorithms/noise.py` | test-only: deterministic noise, must score NOT SUPPORTED |
| `benchmark/algorithms/__init__.py` | `REGISTRY` |
| `benchmark/evaluate.py` | nested grouped CV, four gates, three controls |
| `benchmark/verdict.py` | dual-reconstruction + marginal verdict lattice |
| `scripts/run_benchmark.py` | CLI → `RESULTS.md`; `--check` regenerates and diffs |
| `tests/test_buhlmann.py` … `tests/test_run_benchmark.py` | property tests |

Tasks 1–2 extract shared physics out of `scripts/fit_r0_to_real_dives.py` and `scripts/staged_ascent.py`, which each carry a duplicate copy today. Those scripts then import from `benchmark/`. This is the duplication that propagated the compartment-16 bug into four files.

---

## Task 1: Shared Bühlmann module

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/buhlmann.py`
- Create: `tests/__init__.py`
- Create: `tests/test_buhlmann.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `zhl16c_table() -> np.ndarray` shape `(16,3)` columns `[t_half_min, a_bar, b]`; `half_time_k(table) -> np.ndarray`; `haldane_step(P_t, P_alv, k, dt_min) -> np.ndarray`; `amb_bar(depth_fsw) -> float|np.ndarray`; `m_value(P_amb_bar, a, b) -> np.ndarray`; `ceiling_fsw(P_t, a, b, gf=1.0) -> float`; constants `FSW_TO_BAR`, `P_SURFACE`, `F_N2_AIR`, `DESCENT_FSW_PER_MIN`, `ASCENT_FSW_PER_MIN`, `STOP_INCREMENT_FSW`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_buhlmann.py
import numpy as np
import pytest

from benchmark.buhlmann import (
    zhl16c_table, half_time_k, haldane_step, amb_bar, m_value, ceiling_fsw,
    P_SURFACE, F_N2_AIR,
)

CANONICAL_B = [0.5050, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910,
               0.9092, 0.9222, 0.9319, 0.9403, 0.9477, 0.9544, 0.9602, 0.9653]


def test_table_shape():
    assert zhl16c_table().shape == (16, 3)


def test_b_column_matches_canonical_buhlmann():
    """Compartment 16 b was 0.8693 (compartment 7's value) until 2026-07-09."""
    np.testing.assert_allclose(zhl16c_table()[:, 2], CANONICAL_B)


def test_b_strictly_increasing_and_a_strictly_decreasing():
    t = zhl16c_table()
    assert np.all(np.diff(t[:, 2]) > 0), "b must rise monotonically toward 1"
    assert np.all(np.diff(t[:, 1]) < 0), "a must fall monotonically"
    assert t[15, 2] < 1.0


def test_haldane_halves_the_gap_after_one_half_time():
    """Analytic property: after t_half, the gas gap to the alveolar pressure halves."""
    table = zhl16c_table()
    k = half_time_k(table)
    P_t = np.full(16, 0.79)
    P_alv = np.full(16, 2.79)          # gap of 2.0 bar
    out = haldane_step(P_t, P_alv, k, table[:, 0])   # step each by its own half-time
    np.testing.assert_allclose(P_alv - out, 1.0, rtol=1e-12)


def test_amb_bar_surface_and_33fsw():
    assert amb_bar(0.0) == pytest.approx(P_SURFACE)
    assert amb_bar(33.0) == pytest.approx(P_SURFACE + 33 * 0.030643)


def test_ceiling_zero_when_tissues_at_surface_equilibrium():
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, P_SURFACE * F_N2_AIR)
    assert ceiling_fsw(P_t, a, b) == 0.0


def test_ceiling_rises_with_tissue_loading():
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    low = ceiling_fsw(np.full(16, 1.5), a, b)
    high = ceiling_fsw(np.full(16, 3.0), a, b)
    assert high > low >= 0.0


def test_gradient_factor_one_reduces_to_plain_ceiling():
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, 2.5)
    assert ceiling_fsw(P_t, a, b, gf=1.0) == pytest.approx(ceiling_fsw(P_t, a, b))


def test_smaller_gradient_factor_gives_deeper_ceiling():
    """gf < 1 is more conservative: you must stop deeper."""
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, 2.5)
    assert ceiling_fsw(P_t, a, b, gf=0.3) > ceiling_fsw(P_t, a, b, gf=1.0)


def test_m_value_exceeds_ambient():
    table = zhl16c_table()
    mv = m_value(2.0, table[:, 1], table[:, 2])
    assert np.all(mv > 2.0)
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_buhlmann.py -q`
Expected: `ModuleNotFoundError: No module named 'benchmark'`

- [ ] **Step 3: Write the implementation**

```python
# benchmark/__init__.py
"""Decompression algorithm benchmark against real NMRC 99-02 dive outcomes."""

__version__ = "0.1.0"
```

```python
# benchmark/buhlmann.py
"""Bühlmann ZHL-16C primitives. Shared by every algorithm; owned by none.

Extracted from scripts/fit_r0_to_real_dives.py and scripts/staged_ascent.py,
which each carried a copy. That duplication is how compartment 16's b-coefficient
(0.8693, copy-pasted from compartment 7) reached four generators, a README, two
design docs, and a unit test that asserted it.
"""
from __future__ import annotations

import numpy as np

FSW_TO_BAR = 0.030643       # 33.07 fsw = 1 atm
P_SURFACE = 1.01325         # bar
F_N2_AIR = 0.79

DESCENT_FSW_PER_MIN = 60.0  # US Navy standard
ASCENT_FSW_PER_MIN = 30.0
STOP_INCREMENT_FSW = 10.0


def zhl16c_table() -> np.ndarray:
    """ZHL-16C nitrogen table, shape (16, 3): [t_half_min, a_bar, b].

    The b column is shared across ZHL-16A/B/C. The a column here is ZHL-16A's;
    that discrepancy is recorded in Correction 10 and deliberately not changed,
    since correcting it would alter every row of the existing V2 dataset.
    """
    table = np.array([
        [4.0, 1.2599, 0.5050], [8.0, 1.0000, 0.6514],
        [12.5, 0.8618, 0.7222], [18.5, 0.7562, 0.7825],
        [27.0, 0.6200, 0.8126], [38.3, 0.5043, 0.8434],
        [54.3, 0.4410, 0.8693], [77.0, 0.4000, 0.8910],
        [109.0, 0.3750, 0.9092], [146.0, 0.3500, 0.9222],
        [187.0, 0.3295, 0.9319], [239.0, 0.3065, 0.9403],
        [305.0, 0.2835, 0.9477], [390.0, 0.2610, 0.9544],
        [498.0, 0.2480, 0.9602], [635.0, 0.2327, 0.9653],
    ], dtype=np.float64)
    assert np.all(np.diff(table[:, 2]) > 0), "b must be strictly increasing"
    assert np.all(np.diff(table[:, 1]) < 0), "a must be strictly decreasing"
    return table


def half_time_k(table: np.ndarray) -> np.ndarray:
    return np.log(2.0) / table[:, 0]


def haldane_step(P_t: np.ndarray, P_alv: np.ndarray, k: np.ndarray,
                 dt_min: float | np.ndarray) -> np.ndarray:
    return P_alv + (P_t - P_alv) * np.exp(-k * dt_min)


def amb_bar(depth_fsw):
    return P_SURFACE + np.asarray(depth_fsw, dtype=float) * FSW_TO_BAR


def m_value(P_amb_bar: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + P_amb_bar / b


def ceiling_fsw(P_t: np.ndarray, a: np.ndarray, b: np.ndarray, gf: float = 1.0) -> float:
    """Shallowest depth (fsw) the tissues tolerate. 0.0 means the surface is safe.

    Gradient-factor form: P_amb_tol = (P_t - a*gf) / (gf/b + 1 - gf).
    gf = 1.0 reduces to the plain Bühlmann ceiling (P_t - a) * b.
    Smaller gf is more conservative (deeper ceiling).
    """
    tol_bar = np.max((P_t - a * gf) / (gf / b + 1.0 - gf))
    return max(0.0, float((tol_bar - P_SURFACE) / FSW_TO_BAR))
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_buhlmann.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/__init__.py benchmark/buhlmann.py tests/__init__.py tests/test_buhlmann.py
git commit -m "feat(benchmark): shared Bühlmann module with property tests"
```

---

## Task 2: Profile reconstruction

**Files:**
- Create: `benchmark/profile.py`
- Create: `tests/test_profile.py`

**Interfaces:**
- Consumes: `benchmark.buhlmann` (all of Task 1).
- Produces: `Dive` dataclass (`dive_id: str`, `depth_fsw: float`, `bottom_time_min: float`, `ascent_time_min: float`, `outcome: float`, `data_set: str`); `Profile` dataclass (`dive_id: str`, `recon: str`, `t_min: np.ndarray`, `depth_fsw: np.ndarray`, `flags: Tuple[str, ...]`); `reconstruct(dive: Dive, recon: str) -> Profile` where `recon in {"linear", "staged"}`; `RECONSTRUCTIONS: Tuple[str, str]`; `required_ascent_min(dive: Dive, gf_lo: float = 1.0, gf_hi: float = 1.0) -> Tuple[float, bool]`; `_gf_at(depth_fsw, first_stop_fsw, gf_lo, gf_hi) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profile.py
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
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_profile.py -q`
Expected: `ModuleNotFoundError: No module named 'benchmark.profile'`

- [ ] **Step 3: Write the implementation**

```python
# benchmark/profile.py
"""Dive -> depth/time profile. Two reconstructions, both validated against truth.

The three recorded scalars do not determine a profile. `staged` is measurably
closer to the real curves (median RMSE 36.08 vs 48.71 fsw over 72 gold curves,
Wilcoxon p = 1.8e-11) and still beats predict-the-mean on only 44.4% of dives.
Every downstream claim is therefore gated on agreement between both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from benchmark.buhlmann import (
    ASCENT_FSW_PER_MIN, DESCENT_FSW_PER_MIN, F_N2_AIR, FSW_TO_BAR, P_SURFACE,
    STOP_INCREMENT_FSW, ceiling_fsw, half_time_k, haldane_step, zhl16c_table,
)

RECONSTRUCTIONS: Tuple[str, str] = ("linear", "staged")

DT_MIN = 0.5
SURFACE_WATCH_MIN = 120.0     # bubbles keep growing after surfacing
MAX_STOP_ITERS = 20_000


@dataclass(frozen=True)
class Dive:
    dive_id: str
    depth_fsw: float
    bottom_time_min: float
    ascent_time_min: float
    outcome: float
    data_set: str


@dataclass(frozen=True)
class Profile:
    dive_id: str
    recon: str
    t_min: np.ndarray
    depth_fsw: np.ndarray
    flags: Tuple[str, ...] = ()


def _descent_min(depth_fsw: float) -> float:
    return max(depth_fsw / DESCENT_FSW_PER_MIN, DT_MIN)


def _load_to_bottom(dive: Dive, k: np.ndarray) -> np.ndarray:
    P = np.full(16, P_SURFACE * F_N2_AIR)
    desc = _descent_min(dive.depth_fsw)
    n = max(int(np.ceil(desc / DT_MIN)), 1)
    for i in range(n):
        z = dive.depth_fsw * (i + 1) / n
        P = haldane_step(P, (P_SURFACE + z * FSW_TO_BAR) * F_N2_AIR, k, desc / n)
    nb = max(int(np.ceil(dive.bottom_time_min / DT_MIN)), 1)
    P_alv = (P_SURFACE + dive.depth_fsw * FSW_TO_BAR) * F_N2_AIR
    for _ in range(nb):
        P = haldane_step(P, P_alv, k, dive.bottom_time_min / nb)
    return P


def _gf_at(depth_fsw: float, first_stop_fsw: float, gf_lo: float, gf_hi: float) -> float:
    """Gradient factor interpolated linearly: gf_lo at the first stop, gf_hi at the surface.

    This is what real dive computers do. With gf_lo == gf_hi == 1.0 it degenerates to
    the plain Bühlmann ceiling, which is exactly what ZHL16C wants.
    """
    if first_stop_fsw <= 0.0:
        return gf_hi
    frac = min(max(depth_fsw / first_stop_fsw, 0.0), 1.0)
    return gf_hi + (gf_lo - gf_hi) * frac


def _schedule(P: np.ndarray, depth_fsw: float, a, b, k,
              gf_lo: float = 1.0, gf_hi: float = 1.0):
    """Ceiling-driven ascent. Returns (segments, travel_min, stop_min, hit_cap)."""
    P = P.copy()
    d = float(depth_fsw)
    segs: List[Tuple[str, float, float, float]] = []
    travel = stop = 0.0
    hit_cap = True
    # The first stop is set by the most conservative factor, gf_lo.
    first_stop = np.ceil(ceiling_fsw(P, a, b, gf_lo) / STOP_INCREMENT_FSW) * STOP_INCREMENT_FSW
    for _ in range(MAX_STOP_ITERS):
        if d <= 0.0:
            hit_cap = False
            break
        gf = _gf_at(d, first_stop, gf_lo, gf_hi)
        target = min(np.ceil(ceiling_fsw(P, a, b, gf) / STOP_INCREMENT_FSW)
                     * STOP_INCREMENT_FSW, d)
        if target < d:
            dt = (d - target) / ASCENT_FSW_PER_MIN
            mid = (d + target) / 2.0
            P = haldane_step(P, (P_SURFACE + mid * FSW_TO_BAR) * F_N2_AIR, k, dt)
            segs.append(("travel", d, target, dt))
            travel += dt
            d = target
        else:
            P = haldane_step(P, (P_SURFACE + d * FSW_TO_BAR) * F_N2_AIR, k, DT_MIN)
            if segs and segs[-1][0] == "stop" and segs[-1][1] == d:
                segs[-1] = ("stop", d, d, segs[-1][3] + DT_MIN)
            else:
                segs.append(("stop", d, d, DT_MIN))
            stop += DT_MIN
    return segs, travel, stop, hit_cap


def required_ascent_min(dive: Dive, gf_lo: float = 1.0,
                        gf_hi: float = 1.0) -> Tuple[float, bool]:
    """Minutes of ascent the ceiling demands, and whether the search hit its cap."""
    table = zhl16c_table()
    a, b, k = table[:, 1], table[:, 2], half_time_k(table)
    P = _load_to_bottom(dive, k)
    _, travel, stop, hit_cap = _schedule(P, dive.depth_fsw, a, b, k, gf_lo, gf_hi)
    return travel + stop, hit_cap


def _materialise(segments, desc_min, dive) -> Tuple[np.ndarray, np.ndarray]:
    times, depths = [0.0], [0.0]

    def extend(dur, z0, z1):
        if dur <= 0:
            return
        n = max(int(np.ceil(dur / DT_MIN)), 1)
        for i in range(1, n + 1):
            times.append(times[-1] + dur / n)
            depths.append(z0 + (z1 - z0) * i / n)

    extend(desc_min, 0.0, dive.depth_fsw)
    extend(dive.bottom_time_min, dive.depth_fsw, dive.depth_fsw)
    for z0, z1, dur in segments:
        extend(dur, z0, z1)
    extend(SURFACE_WATCH_MIN, 0.0, 0.0)

    t = np.asarray(times)
    z = np.maximum(np.asarray(depths), 0.0)
    # Degenerate legs (dz ~ 1e-16) emit duplicate timestamps; solve_ivp rejects those.
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    return t[keep], z[keep]


def reconstruct(dive: Dive, recon: str) -> Profile:
    if recon not in RECONSTRUCTIONS:
        raise ValueError(f"unknown reconstruction {recon!r}")
    desc = _descent_min(dive.depth_fsw)
    flags: List[str] = []

    if recon == "linear":
        segs = [(dive.depth_fsw, 0.0, max(dive.ascent_time_min, DT_MIN))]
    else:
        table = zhl16c_table()
        a, b, k = table[:, 1], table[:, 2], half_time_k(table)
        P = _load_to_bottom(dive, k)
        raw, travel, stop, hit_cap = _schedule(P, dive.depth_fsw, a, b, k)
        if hit_cap:
            flags.append("schedule_iteration_cap")
        if dive.ascent_time_min <= travel or stop <= 0.0:
            flags.append("straight_ascent_fallback")
            segs = [(dive.depth_fsw, 0.0, max(dive.ascent_time_min, DT_MIN))]
        else:
            # Rescale STOPS only, never travel, so a dive whose recorded ascent is
            # shorter than the ceiling demands still violates M-values.
            scale = (dive.ascent_time_min - travel) / stop
            segs = [(z0, z1, dur * (scale if kind == "stop" else 1.0))
                    for kind, z0, z1, dur in raw]

    t, z = _materialise(segs, desc, dive)
    return Profile(dive.dive_id, recon, t, z, tuple(flags))
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_profile.py -q`
Expected: `12 passed` (6 parametrized + 6 single). The gold-curve test takes ~20 s.

- [ ] **Step 5: Commit**

```bash
git add benchmark/profile.py tests/test_profile.py
git commit -m "feat(benchmark): profile reconstruction, regression-tested against real curves"
```

---

## Task 3: Algorithm protocol, registry, and ZHL-16C

**Files:**
- Create: `benchmark/algorithms/__init__.py`
- Create: `benchmark/algorithms/base.py`
- Create: `benchmark/algorithms/zhl16c.py`
- Create: `tests/test_algorithms.py`

**Interfaces:**
- Consumes: `benchmark.profile.Profile`, `benchmark.profile.Dive`, `benchmark.profile.required_ascent_min`, `benchmark.buhlmann.*`.
- Produces: `class Algorithm(Protocol)` with `name: str`, `params: Dict[str, float]`, `risk_index(profile: Profile, dive: Dive) -> float`, `deficit(profile: Profile, dive: Dive) -> Optional[float]`; `class AlgorithmError(RuntimeError)`; `REGISTRY: Dict[str, Algorithm]`; `class ZHL16C` with `params = {"gf_lo": 1.0, "gf_hi": 1.0}`.

> Both methods take `(profile, dive)`. `deficit` needs the **recorded** `ascent_time_min`, which lives on `Dive`, not on the reconstructed `Profile`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_algorithms.py
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
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_algorithms.py -q`
Expected: `ModuleNotFoundError: No module named 'benchmark.algorithms'`

- [ ] **Step 3: Write the implementation**

```python
# benchmark/algorithms/base.py
"""The Algorithm contract.

Algorithms are pure functions of a profile. None of them train. A fit/predict
interface would encode a falsehood: ZHL-16C has no parameters to learn, and the
one free parameter that exists (ep_bubble's R0) produced a bimodal,
unidentifiable likelihood when fitted (Correction 13).

`deficit` is optional because it requires a CEILING -- a depth you may not ascend
above. ZHL-16C and ZHL+GF have one; Epstein-Plesset does not. Giving ep_bubble a
schedule *is* VPM-B, which is a different algorithm.

`risk_index` is a RANK. It is never a probability. See the spec's non-goal.
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

from benchmark.profile import Dive, Profile


class AlgorithmError(RuntimeError):
    """Raised when an algorithm cannot produce a trustworthy number."""


@runtime_checkable
class Algorithm(Protocol):
    name: str
    params: Dict[str, float]

    def risk_index(self, profile: Profile, dive: Dive) -> float: ...

    def deficit(self, profile: Profile, dive: Dive) -> Optional[float]: ...
```

```python
# benchmark/algorithms/zhl16c.py
"""Plain Bühlmann ZHL-16C: max M-value ratio, and ceiling-driven deficit."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from benchmark.algorithms.base import AlgorithmError
from benchmark.buhlmann import (
    F_N2_AIR, amb_bar, half_time_k, haldane_step, m_value, zhl16c_table,
)
from benchmark.profile import Dive, Profile, required_ascent_min


class ZHL16C:
    name = "zhl16c"

    def __init__(self, gf_lo: float = 1.0, gf_hi: float = 1.0, name: str = "zhl16c"):
        self.name = name
        self.params: Dict[str, float] = {"gf_lo": gf_lo, "gf_hi": gf_hi}

    def _walk(self, profile: Profile):
        table = zhl16c_table()
        a, b, k = table[:, 1], table[:, 2], half_time_k(table)
        P_amb = amb_bar(profile.depth_fsw)
        P_alv = P_amb * F_N2_AIR
        P_t = np.full(16, amb_bar(0.0) * F_N2_AIR)
        dt = np.diff(profile.t_min, prepend=profile.t_min[0])
        for i in range(len(profile.t_min)):
            if i > 0:
                P_t = haldane_step(P_t, P_alv[i], k, dt[i])
            yield P_t, P_amb[i], a, b

    def risk_index(self, profile: Profile, dive: Dive) -> float:
        """Peak M-value ratio over the dive. Unclipped: these are aggressive dives."""
        best = 0.0
        for P_t, P_amb_i, a, b in self._walk(profile):
            best = max(best, float(np.max(P_t / m_value(P_amb_i, a, b))))
        if not np.isfinite(best):
            raise AlgorithmError(f"{self.name}: non-finite risk_index on {dive.dive_id}")
        return best

    def deficit(self, profile: Profile, dive: Dive) -> Optional[float]:
        """Minutes of ascent the ceiling demanded, minus the minutes actually taken.

        Positive = under-decompressed. Depends on descent+bottom (well constrained
        by the three recorded scalars) and on the RECORDED ascent time -- not on the
        reconstructed ascent shape.
        """
        required, hit_cap = required_ascent_min(
            dive, gf_lo=self.params["gf_lo"], gf_hi=self.params["gf_hi"])
        if hit_cap:
            raise AlgorithmError(f"{self.name}: schedule cap on {dive.dive_id}")
        return float(required - dive.ascent_time_min)
```

```python
# benchmark/algorithms/__init__.py
"""Algorithm registry. Adding one is a new file plus one line here."""
from __future__ import annotations

from typing import Dict

from benchmark.algorithms.base import Algorithm, AlgorithmError
from benchmark.algorithms.zhl16c import ZHL16C

REGISTRY: Dict[str, Algorithm] = {
    "zhl16c": ZHL16C(),
}

__all__ = ["REGISTRY", "Algorithm", "AlgorithmError"]
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_algorithms.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/algorithms/ tests/test_algorithms.py
git commit -m "feat(benchmark): Algorithm protocol, registry, ZHL-16C"
```

---

## Task 4: ZHL-16C with gradient factors

**Files:**
- Create: `benchmark/algorithms/zhl16c_gf.py`
- Modify: `benchmark/algorithms/__init__.py`
- Modify: `tests/test_algorithms.py` (append)

**Interfaces:**
- Consumes: `benchmark.algorithms.zhl16c.ZHL16C`.
- Produces: `class ZHL16CGF(ZHL16C)` with `params = {"gf_lo": 0.30, "gf_hi": 0.70}`; registry key `"zhl16c_gf"`. `risk_index` returns the **peak experienced gradient factor**, `max((P_t - P_amb) / (M - P_amb))`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_algorithms.py`)**

```python
def test_registry_contains_zhl16c_gf():
    assert "zhl16c_gf" in REGISTRY
    assert REGISTRY["zhl16c_gf"].params == {"gf_lo": 0.30, "gf_hi": 0.70}


def test_gf_risk_index_is_a_gradient_factor_not_an_mvalue_ratio():
    """A dive that never supersaturates has GF <= 0; an M-value ratio would be ~0.7."""
    d = dive(depth=15.0, bt=5.0, at=30.0)
    gf = REGISTRY["zhl16c_gf"].risk_index(reconstruct(d, "staged"), d)
    mv = REGISTRY["zhl16c"].risk_index(reconstruct(d, "staged"), d)
    assert gf < mv


def test_gf_demands_more_deco_than_plain_buhlmann():
    """gf_hi = 0.70 is conservative: required ascent time must be >= plain."""
    d = dive(depth=180.0, bt=40.0, at=20.0)
    p = reconstruct(d, "staged")
    assert REGISTRY["zhl16c_gf"].deficit(p, d) >= REGISTRY["zhl16c"].deficit(p, d)


def test_gf_risk_index_rises_with_depth():
    shallow, deep = dive(depth=60.0), dive(depth=160.0)
    a = REGISTRY["zhl16c_gf"]
    assert a.risk_index(reconstruct(deep, "linear"), deep) > \
           a.risk_index(reconstruct(shallow, "linear"), shallow)


def test_gf_lo_is_load_bearing_not_dead():
    """gf_lo sets the first stop. A smaller gf_lo must demand more decompression.

    Without interpolation gf_lo is unused, this test passes trivially, and the
    parameter silently poisons the cache key. It must actually change the answer.
    """
    from benchmark.algorithms.zhl16c_gf import ZHL16CGF
    d = dive(depth=180.0, bt=40.0, at=20.0)
    p = reconstruct(d, "staged")
    strict = ZHL16CGF(gf_lo=0.10, gf_hi=0.70).deficit(p, d)
    loose = ZHL16CGF(gf_lo=0.90, gf_hi=0.70).deficit(p, d)
    assert strict > loose


def test_gf_one_one_matches_plain_buhlmann_deficit():
    """The interpolation must degenerate to the plain ceiling at gf = 1.0/1.0."""
    from benchmark.algorithms.zhl16c_gf import ZHL16CGF
    d = dive(depth=150.0, bt=45.0, at=25.0)
    p = reconstruct(d, "staged")
    assert ZHL16CGF(gf_lo=1.0, gf_hi=1.0).deficit(p, d) == \
        pytest.approx(REGISTRY["zhl16c"].deficit(p, d))
```

- [ ] **Step 2: Run, verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_algorithms.py -q -k gf`
Expected: `KeyError: 'zhl16c_gf'`

- [ ] **Step 3: Implement**

```python
# benchmark/algorithms/zhl16c_gf.py
"""Bühlmann with gradient factors -- the modern dive-computer default.

risk_index is the peak EXPERIENCED gradient factor: how far into the allowed
supersaturation window the diver actually went. 1.0 means exactly at the M-line;
above 1.0 means the M-line was breached.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from benchmark.algorithms.base import AlgorithmError
from benchmark.algorithms.zhl16c import ZHL16C
from benchmark.buhlmann import m_value
from benchmark.profile import Dive, Profile


class ZHL16CGF(ZHL16C):
    """gf_lo applies at the first stop, gf_hi at the surface; linear in between.

    Inherits deficit() unchanged: it reads both params and passes them to the
    ceiling-driven schedule, so gf_lo is load-bearing rather than decorative.
    """

    def __init__(self, gf_lo: float = 0.30, gf_hi: float = 0.70):
        super().__init__(gf_lo=gf_lo, gf_hi=gf_hi, name="zhl16c_gf")

    def risk_index(self, profile: Profile, dive: Dive) -> float:
        best = -np.inf
        for P_t, P_amb_i, a, b in self._walk(profile):
            head = m_value(P_amb_i, a, b) - P_amb_i          # allowed overpressure
            gf = (P_t - P_amb_i) / np.maximum(head, 1e-12)   # experienced fraction
            best = max(best, float(np.max(gf)))
        if not np.isfinite(best):
            raise AlgorithmError(f"{self.name}: non-finite risk_index on {dive.dive_id}")
        return best
```

Register it:

```python
# benchmark/algorithms/__init__.py  (replace REGISTRY block)
from benchmark.algorithms.zhl16c_gf import ZHL16CGF

REGISTRY: Dict[str, Algorithm] = {
    "zhl16c": ZHL16C(),
    "zhl16c_gf": ZHL16CGF(),
}
```

- [ ] **Step 4: Run, verify pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_algorithms.py -q`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/algorithms/zhl16c_gf.py benchmark/algorithms/__init__.py tests/test_algorithms.py
git commit -m "feat(benchmark): ZHL-16C with gradient factors (30/70)"
```

---

## Task 5: Epstein-Plesset bubble algorithm

**Files:**
- Create: `benchmark/algorithms/ep_bubble.py`
- Modify: `benchmark/algorithms/__init__.py`
- Create: `tests/test_ep_bubble.py`

**Interfaces:**
- Consumes: `benchmark.profile.Profile`, `benchmark.buhlmann.*`.
- Produces: `class EPBubble` with `params = {"r0_um": 4.0, "sigma": 0.050, "ceiling_um": 100.0}`; `risk_index` returns `R_max` in µm; `deficit` returns `None`. Registry key `"ep_bubble"`. Module constant `R_DISSOLVE_FRACTION = 0.1`.

**Physics constraints (all from Corrections 11–13):**
- Units: `ΔC = ALPHA_N2 * M_N2 * (P_tissue − P_gas)` (mass basis). Omitting `M_N2` inflates `dR/dt` by exactly `1/M_N2 = 35.714×`.
- The VPM **skin** is mandatory: the nucleus may not dissolve below `R0`. Without it `R_max ≡ R0` on every profile (`std = 1.1e-16`), and the column is constant.
- A **growth ceiling** is mandatory: unbounded bubbles reach 267–284 µm.
- `RK45` + terminal event. Never `Radau` (fails 70% of profiles).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ep_bubble.py
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
    from benchmark.algorithms.ep_bubble import integrate_bubble
    d = dive(depth=150.0, bt=50.0, at=10.0)
    p = reconstruct(d, "staged")
    coarse = integrate_bubble(p, r0_m=4e-6, rtol=1e-6)
    fine = integrate_bubble(p, r0_m=4e-6, rtol=1e-8)
    assert abs(coarse.max() - fine.max()) / fine.max() < 1e-3
```

- [ ] **Step 2: Run, verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_ep_bubble.py -q`
Expected: `KeyError: 'ep_bubble'`

- [ ] **Step 3: Implement**

```python
# benchmark/algorithms/ep_bubble.py
"""Epstein-Plesset bubble growth with a VPM stabilising skin.

Three properties are load-bearing and each was learned the hard way:

1. delta_C is a MASS concentration: alpha_N2 * M_N2 * (P_tissue - P_gas). Omitting
   M_N2 inflates dR/dt by exactly 1/M_N2 = 35.714x (Correction 12).
2. The skin (R may not fall below R0) is what makes the model non-degenerate.
   Without it the nucleus dissolves during descent -- gas leaves an undersaturated
   bubble regardless of its radius -- and R_max == R0 on every profile.
3. RK45 with a terminal event, never Radau (fails on 70% of realistic profiles),
   and never a non-smooth `if R <= 0` guard.

deficit() returns None: Epstein-Plesset defines no ceiling.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from benchmark.algorithms.base import AlgorithmError
from benchmark.buhlmann import (
    F_N2_AIR, amb_bar, half_time_k, haldane_step, zhl16c_table,
)
from benchmark.profile import Dive, Profile

BAR_TO_PA = 1e5
D_N2 = 2.0e-9              # m^2/s, Weathersby (1984)
ALPHA_N2 = 6.84e-6         # mol/(m^3.Pa)
M_N2 = 0.028               # kg/mol
R_GAS = 8.314
T_BODY = 310.15
R_DISSOLVE_FRACTION = 0.1


def _tissue_and_ambient(profile: Profile):
    table = zhl16c_table()
    k = half_time_k(table)
    P_amb = amb_bar(profile.depth_fsw)
    P_alv = P_amb * F_N2_AIR
    P_t = np.full(16, amb_bar(0.0) * F_N2_AIR)
    dt = np.diff(profile.t_min, prepend=profile.t_min[0])
    out = np.empty(len(profile.t_min))
    for i in range(len(profile.t_min)):
        if i > 0:
            P_t = haldane_step(P_t, P_alv[i], k, dt[i])
        out[i] = P_t.max()
    return out, P_amb


def integrate_bubble(profile: Profile, r0_m: float, sigma: float = 0.050,
                     ceiling_m: float = 100e-6, rtol: float = 1e-6) -> np.ndarray:
    """Bubble radius trajectory in metres. Skin floor at r0_m; ceiling at ceiling_m."""
    P_tissue, P_amb = _tissue_and_ambient(profile)
    ts = profile.t_min * 60.0

    def rhs(t, y):
        R = max(y[0], r0_m)                       # VPM skin: cannot dissolve below R0
        if R >= ceiling_m:
            return [0.0]
        P_t_pa = np.interp(t, ts, P_tissue) * BAR_TO_PA
        P_a_pa = np.interp(t, ts, P_amb) * BAR_TO_PA
        P_gas = P_a_pa + 2.0 * sigma / R
        dC = ALPHA_N2 * M_N2 * (P_t_pa - P_gas)   # MASS basis; the M_N2 is mandatory
        if R <= r0_m and dC < 0.0:
            return [0.0]                          # skin holds the nucleus open
        rho_gas = P_gas * M_N2 / (R_GAS * T_BODY)
        corr = 1.0 + R / np.sqrt(np.pi * D_N2 * max(t, 1e-10))
        return [D_N2 * dC / (rho_gas * R) * corr]

    sol = solve_ivp(rhs, (ts[0], ts[-1]), [r0_m], method="RK45", t_eval=ts,
                    rtol=rtol, atol=1e-12)
    if not sol.success or sol.y.shape[1] != len(ts):
        raise AlgorithmError(f"ep_bubble: solve failed on {profile.dive_id}")
    return np.clip(sol.y[0], r0_m, ceiling_m)


class EPBubble:
    name = "ep_bubble"

    def __init__(self, r0_um: float = 4.0, sigma: float = 0.050,
                 ceiling_um: float = 100.0):
        self.params: Dict[str, float] = {
            "r0_um": r0_um, "sigma": sigma, "ceiling_um": ceiling_um,
        }

    def risk_index(self, profile: Profile, dive: Dive) -> float:
        R = integrate_bubble(
            profile,
            r0_m=self.params["r0_um"] * 1e-6,
            sigma=self.params["sigma"],
            ceiling_m=self.params["ceiling_um"] * 1e-6,
        )
        return float(R.max() * 1e6)

    def deficit(self, profile: Profile, dive: Dive) -> Optional[float]:
        return None
```

Register:

```python
# benchmark/algorithms/__init__.py  (replace REGISTRY block)
from benchmark.algorithms.ep_bubble import EPBubble

REGISTRY: Dict[str, Algorithm] = {
    "zhl16c": ZHL16C(),
    "zhl16c_gf": ZHL16CGF(),
    "ep_bubble": EPBubble(),
}
```

- [ ] **Step 4: Run, verify pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_ep_bubble.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/algorithms/ep_bubble.py benchmark/algorithms/__init__.py tests/test_ep_bubble.py
git commit -m "feat(benchmark): Epstein-Plesset + VPM skin; deficit is None by design"
```

---

## Task 6: Content-addressed cache

**Files:**
- Create: `benchmark/cache.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cache_key(*parts: object) -> str` (SHA-256 hex of the JSON-canonical parts); `cached(root: Path, key: str, compute: Callable[[], float]) -> float`; `clear(root: Path) -> None`.

Values are floats or `None` — the two things algorithms return. Stored as JSON, never pickle.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache.py
from pathlib import Path

from benchmark.cache import cache_key, cached, clear


def test_key_is_stable_and_order_sensitive():
    assert cache_key("a", 1) == cache_key("a", 1)
    assert cache_key("a", 1) != cache_key(1, "a")


def test_key_changes_with_params():
    assert cache_key("ep", {"r0_um": 4.0}) != cache_key("ep", {"r0_um": 0.7})


def test_cached_computes_once(tmp_path: Path):
    calls = []

    def compute():
        calls.append(1)
        return 42.0

    k = cache_key("x")
    assert cached(tmp_path, k, compute) == 42.0
    assert cached(tmp_path, k, compute) == 42.0
    assert len(calls) == 1


def test_cached_roundtrips_none(tmp_path: Path):
    k = cache_key("deficit-is-none")
    assert cached(tmp_path, k, lambda: None) is None
    assert cached(tmp_path, k, lambda: 999.0) is None   # served from cache


def test_clear_removes_entries(tmp_path: Path):
    k = cache_key("y")
    cached(tmp_path, k, lambda: 1.0)
    clear(tmp_path)
    calls = []
    cached(tmp_path, k, lambda: (calls.append(1), 2.0)[1])
    assert len(calls) == 1
```

- [ ] **Step 2: Run, verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_cache.py -q`
Expected: `ModuleNotFoundError: No module named 'benchmark.cache'`

- [ ] **Step 3: Implement**

```python
# benchmark/cache.py
"""Content-addressed cache for per-dive algorithm scalars.

Two boundaries: profile reconstruction is invariant to the CV protocol, and the
CV protocol is invariant to algorithm internals. Caching at both means adding an
algorithm recomputes only that algorithm's column, and re-running statistics
recomputes nothing.

JSON only. `joblib.load` on an untrusted pickle is arbitrary code execution, and
these are single floats.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional


def cache_key(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(root: Path, key: str) -> Path:
    return Path(root) / key[:2] / f"{key}.json"


def cached(root: Path, key: str, compute: Callable[[], Optional[float]]) -> Optional[float]:
    p = _path(root, key)
    if p.exists():
        return json.loads(p.read_text())["v"]
    value = compute()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"v": value}))
    return value


def clear(root: Path) -> None:
    for p in Path(root).rglob("*.json"):
        p.unlink()
```

- [ ] **Step 4: Run, verify pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_cache.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/cache.py tests/test_cache.py
git commit -m "feat(benchmark): JSON content-addressed cache, no pickle"
```

---

## Task 7: Evaluation — nested grouped CV, four gates, three controls

**Files:**
- Create: `benchmark/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: nothing from `benchmark.*` (it knows no decompression).
- Produces:
  - `nested_grouped_cv(X, y, groups, *, n_rep=5, seed=0) -> np.ndarray` (fold AUCs, logistic with inner-grouped C selection)
  - `baseline_auc(X_raw, y, groups, **kw) -> np.ndarray`
  - `GateResult` dataclass: `passed: bool`, `delta: float`, `p_value: float`, `frac_improved: float`, `sign: float`, `beats_null_frac: float`, `reasons: Tuple[str, ...]`
  - `four_gate(X_raw, x_feature, y, groups, *, n_rep=5, seed=0) -> GateResult`
  - `shuffle_control(X_raw, y, groups, *, seed=0) -> float`
  - `leakage_gap(X, y, groups, *, seed=0) -> float`
  - Constants `MIN_DELTA = 0.03`, `MAX_P = 0.05`, `MIN_FRAC_IMPROVED = 0.75`, `MIN_NULL_FRAC = 0.95`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
import numpy as np
import pytest

from benchmark.evaluate import (
    baseline_auc, four_gate, leakage_gap, nested_grouped_cv, shuffle_control,
)

RNG = np.random.RandomState(0)


def synthetic(n=600, n_groups=12, signal=1.5, seed=0):
    """Feature with real signal; groups carry their own intercept (trial effect)."""
    rng = np.random.RandomState(seed)
    g = rng.randint(0, n_groups, n)
    x = rng.normal(size=n)
    trial_effect = rng.normal(scale=2.0, size=n_groups)[g]
    logit = signal * x + trial_effect - 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X_raw = rng.normal(size=(n, 3))
    return X_raw, x, y, g


def test_shuffled_labels_collapse_to_chance():
    """A sound protocol must return ~0.5 on shuffled labels."""
    X_raw, x, y, g = synthetic()
    auc = shuffle_control(np.column_stack([X_raw, x]), y, g, seed=1)
    assert abs(auc - 0.5) < 0.08, f"protocol broken, not the model: {auc:.3f}"


def test_leakage_gap_is_positive_when_groups_carry_signal():
    """Ordinary CV lets a model memorise trial identity; grouped CV does not."""
    X_raw, x, y, g = synthetic(signal=0.1)      # weak feature, strong trial effect
    assert leakage_gap(np.column_stack([X_raw, x]), y, g) > 0.0


def test_real_signal_passes_all_four_gates():
    X_raw, x, y, g = synthetic(signal=2.5)
    r = four_gate(X_raw, x, y, g)
    assert r.passed, r.reasons
    assert r.delta > 0.03 and r.sign > 0


def test_pure_noise_fails():
    X_raw, x, y, g = synthetic(signal=0.0)
    r = four_gate(X_raw, RNG.normal(size=len(y)), y, g)
    assert not r.passed


def test_sign_inverted_feature_fails_even_though_it_helps():
    """The gate that caught the 'deep short dive' confound.

    Negating a genuinely predictive feature keeps |delta| and p identical, but the
    coefficient sign flips. That must be rejected.
    """
    X_raw, x, y, g = synthetic(signal=2.5)
    good = four_gate(X_raw, x, y, g)
    flipped = four_gate(X_raw, -x, y, g)
    assert good.passed
    assert not flipped.passed
    assert "sign" in " ".join(flipped.reasons)
    assert flipped.delta == pytest.approx(good.delta, abs=1e-6)


def test_baseline_is_reproducible_and_matches_nested_cv_on_same_X():
    X_raw, x, y, g = synthetic()
    a = baseline_auc(X_raw, y, g, seed=3)
    b = nested_grouped_cv(X_raw, y, g, seed=3)
    np.testing.assert_allclose(a, b)
```

- [ ] **Step 2: Run, verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_evaluate.py -q`
Expected: `ModuleNotFoundError: No module named 'benchmark.evaluate'`

- [ ] **Step 3: Implement**

```python
# benchmark/evaluate.py
"""Evaluation protocol. Knows nothing about decompression.

Three properties, each of which silently inflates results on this dataset:

1. GROUPED outer folds. Trials differ in protocol aggressiveness (DCS 4.6%-35%);
   ordinary CV inflates AUC by +0.045 to +0.075 by letting a model memorise
   trial identity.
2. NESTED tuning. Hyperparameters chosen on outer test folds re-import that leak.
3. A LABEL-SHUFFLE control. Above chance on shuffled labels means the protocol is
   broken, not the model.

Effective sample size is ~38 trials, not ~1,948 dives. Fold sd ~ 0.06, so an
effect must clear MIN_DELTA and a paired test, not merely a mean comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_DELTA = 0.03
MAX_P = 0.05
MIN_FRAC_IMPROVED = 0.75
MIN_NULL_FRAC = 0.95
C_GRID = (0.03, 0.1, 0.3, 1.0, 1e6)


def _pipe(C: float) -> Pipeline:
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])


def _fit_score(C, Xtr, ytr, Xte, yte) -> float:
    m = _pipe(C).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def nested_grouped_cv(X, y, groups, *, n_rep: int = 5, seed: int = 0,
                      n_outer: int = 5, n_inner: int = 3) -> np.ndarray:
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    out: List[float] = []
    for _ in range(n_rep):
        gmap = {g: i for i, g in enumerate(rng.permutation(uniq))}
        gi = np.array([gmap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_outer).split(X, y, gi):
            if len(np.unique(y[te])) < 2:
                continue
            best_C, best = C_GRID[0], -np.inf
            if len(np.unique(gi[tr])) >= n_inner:
                for C in C_GRID:
                    inner = [
                        _fit_score(C, X[tr][itr], y[tr][itr], X[tr][ite], y[tr][ite])
                        for itr, ite in GroupKFold(n_inner).split(X[tr], y[tr], gi[tr])
                        if len(np.unique(y[tr][ite])) >= 2
                    ]
                    if inner and np.mean(inner) > best:
                        best, best_C = float(np.mean(inner)), C
            out.append(_fit_score(best_C, X[tr], y[tr], X[te], y[te]))
    return np.asarray(out)


def baseline_auc(X_raw, y, groups, **kw) -> np.ndarray:
    """The number to beat: logistic on the raw dive parameters, nothing else."""
    return nested_grouped_cv(X_raw, y, groups, **kw)


def shuffle_control(X, y, groups, *, seed: int = 0) -> float:
    rng = np.random.RandomState(seed + 99)
    return float(nested_grouped_cv(X, rng.permutation(np.asarray(y)), groups,
                                   n_rep=1, seed=seed).mean())


def leakage_gap(X, y, groups, *, seed: int = 0) -> float:
    """Ordinary CV minus grouped CV. Printed so nobody quotes the inflated number."""
    X, y = np.asarray(X, float), np.asarray(y)
    ordinary = [
        _fit_score(1.0, X[tr], y[tr], X[te], y[te])
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    ]
    grouped = nested_grouped_cv(X, y, groups, n_rep=1, seed=seed)
    return float(np.mean(ordinary) - grouped.mean())


@dataclass(frozen=True)
class GateResult:
    passed: bool
    delta: float
    p_value: float
    frac_improved: float
    sign: float
    beats_null_frac: float
    reasons: Tuple[str, ...]


def _coefficient_sign(X_raw, x, y) -> float:
    X = np.column_stack([np.asarray(X_raw, float), np.asarray(x, float)])
    m = _pipe(1.0).fit(X, y)
    return float(np.sign(m.named_steps["clf"].coef_[0][-1]))


def four_gate(X_raw, x, y, groups, *, n_rep: int = 5, seed: int = 0,
              n_null: int = 20) -> GateResult:
    """An effect counts only if it clears magnitude, significance, SIGN, and a null."""
    X_raw = np.asarray(X_raw, float)
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y)

    base = baseline_auc(X_raw, y, groups, n_rep=n_rep, seed=seed)
    full = nested_grouped_cv(np.column_stack([X_raw, x]), y, groups,
                             n_rep=n_rep, seed=seed)
    n = min(len(base), len(full))
    d = full[:n] - base[:n]

    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    sign = _coefficient_sign(X_raw, x, y)

    rng = np.random.RandomState(seed + 7)
    null = np.array([
        (nested_grouped_cv(np.column_stack([X_raw, rng.permutation(x)]), y, groups,
                           n_rep=1, seed=seed + k)[:n].mean() - base[:n].mean())
        for k in range(n_null)
    ])
    beats = float(np.mean(d.mean() > null))

    reasons: List[str] = []
    if abs(d.mean()) <= MIN_DELTA:
        reasons.append(f"magnitude |{d.mean():+.4f}| <= {MIN_DELTA}")
    if p >= MAX_P:
        reasons.append(f"p={p:.3f} >= {MAX_P}")
    if np.mean(d > 0) < MIN_FRAC_IMPROVED:
        reasons.append(f"only {np.mean(d > 0):.0%} of folds improved")
    if sign <= 0:
        reasons.append("sign inverted: feature predicts FEWER events (confound)")
    if beats < MIN_NULL_FRAC:
        reasons.append(f"beats only {beats:.0%} of permutations")

    return GateResult(not reasons, float(d.mean()), p, float(np.mean(d > 0)),
                      sign, beats, tuple(reasons))
```

- [ ] **Step 4: Run, verify pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_evaluate.py -q`
Expected: `6 passed` (takes ~2 min; the permutation null dominates)

- [ ] **Step 5: Commit**

```bash
git add benchmark/evaluate.py tests/test_evaluate.py
git commit -m "feat(benchmark): nested grouped CV, four gates, shuffle and leakage controls"
```

---

## Task 8: Verdict lattice

**Files:**
- Create: `benchmark/verdict.py`
- Create: `tests/test_verdict.py`

**Interfaces:**
- Consumes: `benchmark.evaluate.GateResult`.
- Produces: `verdict(gates: Dict[Tuple[str, str], GateResult]) -> str` where the key is `(recon, marginal_rule)` and the return is one of `"SUPPORTED"`, `"RECONSTRUCTION-SENSITIVE"`, `"MARGINAL-SENSITIVE"`, `"NOT SUPPORTED"`. Constant `PRIMARY_MARGINAL = "exclude"`.

Rules, exactly as specified:
- `SUPPORTED` — passes under **both** reconstructions at `PRIMARY_MARGINAL`, and does not reverse under either other marginal rule.
- `RECONSTRUCTION-SENSITIVE` — passes under one reconstruction, not the other, at `PRIMARY_MARGINAL`.
- `MARGINAL-SENSITIVE` — would be `SUPPORTED`, but reverses under `positive` or `negative`.
- `NOT SUPPORTED` — otherwise. **Failing under `exclude` is `NOT SUPPORTED` regardless of the sensitivity runs** (they demote, never promote).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_verdict.py
from benchmark.evaluate import GateResult
from benchmark.verdict import PRIMARY_MARGINAL, verdict

RECONS = ("linear", "staged")
RULES = ("exclude", "positive", "negative")


def g(passed: bool, sign: float = 1.0) -> GateResult:
    return GateResult(passed, 0.05, 0.001, 0.9, sign, 1.0, () if passed else ("x",))


def grid(**over):
    base = {(r, m): g(True) for r in RECONS for m in RULES}
    base.update(over)
    return base


def test_supported_when_everything_passes():
    assert verdict(grid()) == "SUPPORTED"


def test_reconstruction_sensitive():
    assert verdict(grid(**{("linear", PRIMARY_MARGINAL): g(False)})) \
        == "RECONSTRUCTION-SENSITIVE"


def test_marginal_sensitive():
    assert verdict(grid(**{("staged", "negative"): g(False)})) == "MARGINAL-SENSITIVE"


def test_not_supported_when_primary_fails_both():
    assert verdict(grid(**{("linear", PRIMARY_MARGINAL): g(False),
                           ("staged", PRIMARY_MARGINAL): g(False)})) == "NOT SUPPORTED"


def test_sensitivity_runs_cannot_promote():
    """Fails under exclude, passes under positive -> still NOT SUPPORTED."""
    cells = {(r, m): g(m == "positive") for r in RECONS for m in RULES}
    assert verdict(cells) == "NOT SUPPORTED"


def test_missing_cell_raises():
    cells = grid()
    del cells[("staged", "positive")]
    try:
        verdict(cells)
    except KeyError:
        return
    raise AssertionError("missing cell must raise, never default to a pass")
```

- [ ] **Step 2: Run, verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_verdict.py -q`
Expected: `ModuleNotFoundError: No module named 'benchmark.verdict'`

- [ ] **Step 3: Implement**

```python
# benchmark/verdict.py
"""Dual-reconstruction and marginal-rule verdict lattice.

Conclusions flip between the two profile reconstructions (Correction 13: prs
scored 0.6004 in-sample under linear, 0.3843 out-of-sample under staged -- worse
than chance). A finding counts only if it survives both. The benchmark's biggest
weakness becomes its safeguard.

The marginal rule is a SAFETY choice, not a preprocessing default: under 0.5 -> 0
the entire nonlinear-model advantage vanishes. Sensitivity runs may DEMOTE a
verdict, never promote one.
"""
from __future__ import annotations

from typing import Dict, Tuple

from benchmark.evaluate import GateResult

PRIMARY_MARGINAL = "exclude"
RECONSTRUCTIONS = ("linear", "staged")
MARGINAL_RULES = ("exclude", "positive", "negative")

SUPPORTED = "SUPPORTED"
RECON_SENSITIVE = "RECONSTRUCTION-SENSITIVE"
MARGINAL_SENSITIVE = "MARGINAL-SENSITIVE"
NOT_SUPPORTED = "NOT SUPPORTED"


def verdict(gates: Dict[Tuple[str, str], GateResult]) -> str:
    for recon in RECONSTRUCTIONS:
        for rule in MARGINAL_RULES:
            if (recon, rule) not in gates:
                raise KeyError(f"missing cell {(recon, rule)}; refusing to default")

    primary = [gates[(r, PRIMARY_MARGINAL)].passed for r in RECONSTRUCTIONS]
    if not any(primary):
        return NOT_SUPPORTED
    if not all(primary):
        return RECON_SENSITIVE

    others = [gates[(r, m)].passed
              for r in RECONSTRUCTIONS
              for m in MARGINAL_RULES if m != PRIMARY_MARGINAL]
    return SUPPORTED if all(others) else MARGINAL_SENSITIVE
```

- [ ] **Step 4: Run, verify pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_verdict.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/verdict.py tests/test_verdict.py
git commit -m "feat(benchmark): dual-reconstruction verdict lattice"
```

---

## Task 9: Runner, RESULTS.md, and `--check`

**Files:**
- Create: `benchmark/algorithms/noise.py`
- Modify: `benchmark/algorithms/__init__.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/test_run_benchmark.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `load_dives(csv, marginal) -> Tuple[List[Dive], np.ndarray, np.ndarray, DropReport]`; `build_matrix(dives, recon, algo, cache_root) -> Dict[str, np.ndarray]`; `render_results(...) -> str`; CLI `--marginal` (required), `--out`, `--check`, `--cache`, `--repeats`, `--seed`.

**Data rules:**
- Bounce window: `depth_fsw <= 300 and bottom_time_min <= 300 and ascent_time_min <= 300`. Print the dropped count **and their DCS rate** — the exclusion is outcome-correlated (24.3% vs 13.7%).
- `RESULTS.md` header records: git SHA, SHA-256 of the input CSV, Python and library versions, resolved algorithm params, and the exact command.
- Zero-variance algorithm column → `AlgorithmError`.
- `ep_solve_failed` rate above 1% → abort.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_benchmark.py
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark.algorithms import REGISTRY
from benchmark.algorithms.base import AlgorithmError
from benchmark.profile import Dive, reconstruct

PY = "/opt/miniconda3/bin/python3"
ROOT = Path(__file__).resolve().parents[1]


def test_noise_algorithm_is_registered_and_has_variance():
    algo = REGISTRY["noise"]
    vals = [algo.risk_index(reconstruct(Dive(f"d{i}", 100, 40, 20, 0.0, "T"), "linear"),
                            Dive(f"d{i}", 100, 40, 20, 0.0, "T")) for i in range(20)]
    assert np.std(vals) > 1e-9


def test_constant_column_raises_rather_than_scoring():
    """A zero-variance column is a degenerate model (Correction 11), not a result."""
    from scripts.run_benchmark import assert_has_variance
    with pytest.raises(AlgorithmError):
        assert_has_variance(np.full(20, 3.0), "constant")


def test_bounce_window_drops_saturation_dives_and_reports_them():
    from scripts.run_benchmark import load_dives
    dives, y, g, report = load_dives(None, "exclude")
    assert report.n_dropped > 0
    assert report.dropped_dcs_rate > report.kept_dcs_rate, \
        "the exclusion is outcome-correlated and must be reported, not hidden"


def test_marginal_flag_is_required():
    r = subprocess.run([PY, "scripts/run_benchmark.py"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "--marginal" in (r.stderr + r.stdout)


def test_check_fails_on_a_stale_results_file(tmp_path):
    stale = tmp_path / "RESULTS.md"
    stale.write_text("# stale\n")
    r = subprocess.run(
        [PY, "scripts/run_benchmark.py", "--marginal", "exclude",
         "--out", str(stale), "--check", "--repeats", "1"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
```

- [ ] **Step 2: Run, verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_run_benchmark.py -q`
Expected: `KeyError: 'noise'`

- [ ] **Step 3: Implement**

```python
# benchmark/algorithms/noise.py
"""Test-only algorithm: deterministic noise from the dive id.

Proves extensibility (one file + one registry line) and must score NOT SUPPORTED.
It is NOT a constant: a constant column has zero variance and must raise
AlgorithmError instead, per Correction 11's guard. The spec's original success
criterion conflated these two; see the plan's 'Spec deviation' note.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

from benchmark.profile import Dive, Profile


class Noise:
    name = "noise"

    def __init__(self) -> None:
        self.params: Dict[str, float] = {}

    def risk_index(self, profile: Profile, dive: Dive) -> float:
        h = hashlib.sha256(dive.dive_id.encode()).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64

    def deficit(self, profile: Profile, dive: Dive) -> Optional[float]:
        return None
```

Register it (kept out of the default run; selected by `--algorithms noise`):

```python
# benchmark/algorithms/__init__.py  (final form)
from __future__ import annotations

from typing import Dict

from benchmark.algorithms.base import Algorithm, AlgorithmError
from benchmark.algorithms.ep_bubble import EPBubble
from benchmark.algorithms.noise import Noise
from benchmark.algorithms.zhl16c import ZHL16C
from benchmark.algorithms.zhl16c_gf import ZHL16CGF

REGISTRY: Dict[str, Algorithm] = {
    "zhl16c": ZHL16C(),
    "zhl16c_gf": ZHL16CGF(),
    "ep_bubble": EPBubble(),
    "noise": Noise(),
}
DEFAULT_ALGORITHMS = ("zhl16c", "zhl16c_gf", "ep_bubble")

__all__ = ["REGISTRY", "DEFAULT_ALGORITHMS", "Algorithm", "AlgorithmError"]
```

```python
# scripts/run_benchmark.py
"""Run the decompression benchmark and regenerate RESULTS.md.

Every number in RESULTS.md is generated. `--check` regenerates and diffs, exiting
non-zero on drift -- the machine-checkable form of the Provenance discipline rule
added after Corrections 9 and 12 fabricated statistics that prose review missed.

This tool NEVER emits a probability. risk_index is a rank.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.algorithms import DEFAULT_ALGORITHMS, REGISTRY, AlgorithmError
from benchmark.cache import cache_key, cached
from benchmark.evaluate import baseline_auc, four_gate, leakage_gap, shuffle_control
from benchmark.profile import Dive, RECONSTRUCTIONS, reconstruct
from benchmark.verdict import MARGINAL_RULES, PRIMARY_MARGINAL, verdict

REAL_CSV = os.path.expanduser("~/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv")
RAW_COLS = ["depth_fsw", "bottom_time_min", "ascent_time_min"]
MAX_BOUNCE = 300.0
EP_FAILURE_ABORT_RATE = 0.01


@dataclass(frozen=True)
class DropReport:
    n_total: int
    n_kept: int
    n_dropped: int
    kept_dcs_rate: float
    dropped_dcs_rate: float


def assert_has_variance(col: np.ndarray, name: str) -> None:
    if float(np.nanstd(col)) < 1e-12:
        raise AlgorithmError(
            f"{name}: zero-variance column. The model is degenerate "
            f"(see Correction 11); it is not a result."
        )


def load_dives(csv: Optional[str], marginal: str):
    df = pd.read_csv(csv or REAL_CSV)
    n_total = len(df)
    keep = ((df.depth_fsw <= MAX_BOUNCE) & (df.bottom_time_min <= MAX_BOUNCE)
            & (df.ascent_time_min <= MAX_BOUNCE))
    dropped, df = df[~keep], df[keep].copy()
    report = DropReport(n_total, len(df), len(dropped),
                        float((df.outcome == 1.0).mean()),
                        float((dropped.outcome == 1.0).mean()) if len(dropped) else 0.0)

    if marginal == "exclude":
        df = df[df.outcome != 0.5].copy()
    elif marginal == "positive":
        df["outcome"] = (df.outcome > 0.0).astype(float)
    elif marginal == "negative":
        df["outcome"] = (df.outcome >= 1.0).astype(float)
    else:
        raise ValueError(marginal)

    dives = [
        Dive(f"{r.data_set}:{r.profile_number}", float(r.depth_fsw),
             float(r.bottom_time_min), float(r.ascent_time_min),
             float(r.outcome), str(r.data_set))
        for r in df.itertuples()
    ]
    y = (df.outcome.values >= 1.0).astype(int)
    groups = df.data_set.values
    return dives, y, groups, report


_PROFILE_CACHE: Dict[Tuple[str, str], object] = {}


def _profile(dive: Dive, recon: str):
    """CACHE 1: reconstruction is invariant to which algorithm consumes it.

    In-process, not on disk: a Profile is two arrays, cheap to rebuild (0.7 ms)
    but wasteful to redo once per algorithm.
    """
    key = (dive.dive_id, recon)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = reconstruct(dive, recon)
    return _PROFILE_CACHE[key]


def build_matrix(dives: List[Dive], recon: str, algo_name: str, cache_root: Path):
    algo = REGISTRY[algo_name]
    risk, deficit, failures = [], [], 0
    for d in dives:
        k_r = cache_key("risk", algo_name, algo.params, recon, d.dive_id,
                        d.depth_fsw, d.bottom_time_min, d.ascent_time_min)
        k_d = cache_key("deficit", algo_name, algo.params, recon, d.dive_id,
                        d.depth_fsw, d.bottom_time_min, d.ascent_time_min)
        try:
            p = _profile(d, recon)                      # CACHE 1
            risk.append(cached(cache_root, k_r, lambda: algo.risk_index(p, d)))
            deficit.append(cached(cache_root, k_d, lambda: algo.deficit(p, d)))
        except AlgorithmError:
            failures += 1
            risk.append(np.nan)
            deficit.append(np.nan)

    if failures / max(len(dives), 1) > EP_FAILURE_ABORT_RATE:
        raise AlgorithmError(
            f"{algo_name}: {failures}/{len(dives)} solves failed "
            f"(> {EP_FAILURE_ABORT_RATE:.0%}). Refusing to emit a dataset whose "
            f"physics is fabricated on a label-correlated subset."
        )
    out: Dict[str, np.ndarray] = {"risk_index": np.asarray(risk, float),
                                  "n_failed": failures}
    dv = np.asarray(deficit, float)
    out["deficit"] = None if np.all(np.isnan(dv)) else dv
    return out


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def render_results(rows, controls, report, args, csv_path) -> str:
    import scipy, sklearn
    lines = [
        "# Decompression Algorithm Benchmark — Results",
        "",
        "> Generated file. Do not edit. Regenerate with the command below;",
        "> `--check` fails if this file has drifted.",
        "",
        "## Provenance",
        "",
        f"- command: `python scripts/run_benchmark.py --marginal {args.marginal} "
        f"--repeats {args.repeats} --seed {args.seed}`",
        f"- git: `{_git_sha()}`",
        f"- input: `{csv_path}` sha256[:16] `{_sha256(csv_path)}`",
        f"- python {platform.python_version()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, sklearn {sklearn.__version__}",
        "",
        "## Cohort",
        "",
        f"- {report.n_total} dives; kept {report.n_kept} bounce dives "
        f"({report.n_kept / report.n_total:.1%})",
        f"- dropped {report.n_dropped} saturation / >300 fsw excursions; their DCS rate "
        f"was {report.dropped_dcs_rate:.1%} vs {report.kept_dcs_rate:.1%} kept "
        f"(**the exclusion is outcome-correlated**)",
        "",
        "## Verdicts",
        "",
        "| algorithm | metric | verdict | ΔAUC (staged, exclude) | sign | reasons |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['algo']}` | `{r['metric']}` | **{r['verdict']}** | "
            f"{r['delta']:+.4f} | {r['sign']:+.0f} | {r['reasons'] or '—'} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        f"- label shuffle → AUC {controls['shuffle']:.4f} (must be ≈ 0.5)",
        f"- leakage gap (ordinary − grouped) → +{controls['leakage']:.4f}",
        f"- baseline (logistic on 3 raw features) → AUC {controls['baseline']:.4f} "
        f"± {controls['baseline_sd']:.4f}",
        "",
        "## Reading this table",
        "",
        "AUC is a **ranking**, never a probability. The ~16% DCS rate here reflects Navy",
        "trials designed to provoke DCS on partially-extracted negatives (2,700 of 8,578).",
        "This benchmark is not a dive-planning tool. Fold sd ≈ 0.06, so |ΔAUC| < 0.03 is",
        "noise. `N/A — no schedule` means the algorithm defines no ceiling.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marginal", required=True,
                    choices=list(MARGINAL_RULES),
                    help="Safety-relevant choice; there is deliberately no default.")
    ap.add_argument("--algorithms", nargs="*", default=list(DEFAULT_ALGORITHMS))
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=".benchmark_cache")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=REAL_CSV)
    args = ap.parse_args()

    cache_root = Path(args.cache)
    rows = []
    for algo_name in args.algorithms:
        gates_by_metric: Dict[str, Dict[Tuple[str, str], object]] = {}
        for rule in MARGINAL_RULES:
            dives, y, groups, report = load_dives(args.csv, rule)
            X_raw = np.array([[d.depth_fsw, d.bottom_time_min, d.ascent_time_min]
                              for d in dives], float)
            for recon in RECONSTRUCTIONS:
                cols = build_matrix(dives, recon, algo_name, cache_root)
                for metric in ("risk_index", "deficit"):
                    col = cols[metric]
                    if col is None:
                        continue
                    assert_has_variance(col, f"{algo_name}.{metric}")
                    g = four_gate(X_raw, col, y, groups,
                                  n_rep=args.repeats, seed=args.seed)
                    gates_by_metric.setdefault(metric, {})[(recon, rule)] = g

        for metric, gates in gates_by_metric.items():
            v = verdict(gates)
            primary = gates[("staged", PRIMARY_MARGINAL)]
            rows.append({"algo": algo_name, "metric": metric, "verdict": v,
                         "delta": primary.delta, "sign": primary.sign,
                         "reasons": "; ".join(primary.reasons)})

    dives, y, groups, report = load_dives(args.csv, args.marginal)
    X_raw = np.array([[d.depth_fsw, d.bottom_time_min, d.ascent_time_min]
                      for d in dives], float)
    base = baseline_auc(X_raw, y, groups, n_rep=args.repeats, seed=args.seed)
    controls = {
        "shuffle": shuffle_control(X_raw, y, groups, seed=args.seed),
        "leakage": leakage_gap(X_raw, y, groups, seed=args.seed),
        "baseline": float(base.mean()),
        "baseline_sd": float(base.std()),
    }

    text = render_results(rows, controls, report, args, args.csv)
    out = Path(args.out)
    if args.check:
        if not out.exists() or out.read_text() != text:
            print(f"{out} is stale. Regenerate it.", file=sys.stderr)
            return 1
        print(f"{out} is current.")
        return 0
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run, verify pass**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_run_benchmark.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmark/algorithms/noise.py benchmark/algorithms/__init__.py \
        scripts/run_benchmark.py tests/test_run_benchmark.py
git commit -m "feat(benchmark): runner, generated RESULTS.md, --check provenance gate"
```

---

## Task 10: Full run, extensibility proof, and spec amendment

**Files:**
- Create: `RESULTS.md` (generated)
- Modify: `docs/superpowers/specs/2026-07-10-decompression-benchmark-design.md` (Success Criterion 5)
- Modify: `scripts/fit_r0_to_real_dives.py`, `scripts/staged_ascent.py` (import shared Bühlmann)

- [ ] **Step 1: Run the whole suite**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/ -q`
Expected: all pass (~4 min; permutation nulls dominate).

- [ ] **Step 2: De-duplicate the physics**

Replace the local ZHL-16C table and Haldane step in `scripts/staged_ascent.py` and
`scripts/fit_r0_to_real_dives.py` with imports:

```python
from benchmark.buhlmann import (
    ASCENT_FSW_PER_MIN, DESCENT_FSW_PER_MIN, FSW_TO_BAR, F_N2_AIR, P_SURFACE,
    STOP_INCREMENT_FSW, half_time_k, haldane_step, zhl16c_table,
)
```

Delete the duplicated `zhl16c()` / `_step()` definitions. Then confirm the earlier
results still reproduce:

Run: `/opt/miniconda3/bin/python3 scripts/validate_reconstruction.py`
Expected: `median RMSE linear 48.71 / staged 36.08`, `staged better on 83.3%`

- [ ] **Step 3: Amend the spec's contradictory success criterion**

In `docs/superpowers/specs/2026-07-10-decompression-benchmark-design.md`, replace
Success Criterion 5's parenthetical:

> (Demonstrated by a trivial `constant` algorithm in tests, which must score `NOT SUPPORTED`.)

with:

> (Demonstrated by a `noise` algorithm — a deterministic hash of `dive_id` — which has
> variance, carries no signal, and must score `NOT SUPPORTED`. A `constant` column cannot
> score at all: zero variance raises `AlgorithmError` under the Correction 11 guard. The
> original wording asked for both and was self-contradictory.)

- [ ] **Step 4: Generate results and verify the provenance gate**

```bash
/opt/miniconda3/bin/python3 scripts/run_benchmark.py --marginal exclude
/opt/miniconda3/bin/python3 scripts/run_benchmark.py --marginal exclude --check
```

Expected: first writes `RESULTS.md`; second prints `RESULTS.md is current.` and exits 0.
Then confirm it detects drift:

```bash
echo "drift" >> RESULTS.md
/opt/miniconda3/bin/python3 scripts/run_benchmark.py --marginal exclude --check; echo "exit=$?"
```

Expected: `RESULTS.md is stale.` and `exit=1`. Regenerate before committing.

- [ ] **Step 5: Confirm the prediction recorded in the spec**

The spec predicted, before implementation, that **no algorithm reaches `SUPPORTED`**.
Read `RESULTS.md`. If every verdict is `NOT SUPPORTED` or `RECONSTRUCTION-SENSITIVE`,
the prediction held. **If any algorithm reaches `SUPPORTED`, do not celebrate it** —
re-run with `--repeats 20`, inspect the coefficient sign, and check the permutation
null before believing it.

- [ ] **Step 6: Commit**

```bash
git add RESULTS.md docs/superpowers/specs/2026-07-10-decompression-benchmark-design.md \
        scripts/staged_ascent.py scripts/fit_r0_to_real_dives.py
git commit -m "feat(benchmark): first full run; de-duplicate physics; amend contradictory criterion"
```

---

## Self-Review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| `Algorithm` Protocol, `deficit` optional | 3 |
| Registry, one file + one line to extend | 3, 9 (proven by `noise`) |
| `raw` is the reference, not in the registry | 7 (`baseline_auc`), 9 |
| `zhl16c`, `zhl16c_gf` (30/70), `ep_bubble` | 3, 4, 5 |
| Two reconstructions, staged validated vs gold curves | 2 |
| Cache 1 (profile) + Cache 2 (algorithm scalars), params in key | 6, 9 |
| Caches over all 2,230 before marginal rule | 9 (`build_matrix` is marginal-independent) |
| Nested grouped CV, inner grouped | 7 |
| Four gates incl. coefficient sign | 7 |
| Dual-reconstruction + marginal lattice; sensitivity demotes only | 8 |
| Three standing controls | 7, 9 |
| Never emits a probability | 3 (docstring), 9 (`render_results` note) |
| Zero variance → `RuntimeError` | 9 (`assert_has_variance`) |
| EP failure > 1% → abort | 9 (`build_matrix`) |
| Outcome-correlated exclusion reported | 9 (`DropReport`) |
| `RESULTS.md` generated, `--check` diffs | 9 |
| `np.trapezoid`, no pickle, RK45 + event, `np.interp` | Global Constraints; 5 |
| One extraction, not a fifth fork | 1, 10 Step 2 |
| Marginal rule explicit, no default | 9 (`required=True`) |
| Expected result recorded in advance | 10 Step 5 |

**Gap found and closed:** the spec's USN-table fixture test (`ceiling reproduces a published
USN table entry`) is **not** in any task. It requires a primary source I cannot verify from
here. Per the spec's own "one honest gap" clause, it does not ship, and Task 10 Step 3 is the
place to record that. Task 1 ships the analytic Haldane half-time test instead, which needs no
external fixture. **Do not substitute a fixture generated from this implementation** — that is
exactly how `test_table_compartment16` came to guard the bug it should have caught.

**Placeholder scan:** none. Every code step contains runnable code; every command has expected
output.

**Type consistency:** `risk_index(profile, dive)` and `deficit(profile, dive)` take both
arguments in `base.py`, `zhl16c.py`, `zhl16c_gf.py`, `ep_bubble.py`, `noise.py`, and are called
that way in `build_matrix`. `GateResult` fields used in `verdict.py` and `render_results` match
the dataclass in Task 7. `MARGINAL_RULES` and `RECONSTRUCTIONS` are defined once in
`verdict.py` / `profile.py` and imported everywhere else.

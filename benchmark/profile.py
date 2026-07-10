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

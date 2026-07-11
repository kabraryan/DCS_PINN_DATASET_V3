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

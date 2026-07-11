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

On r0_um = 0.7: this is the Yount VPM literature value, and it is chosen over the
4.0 that an earlier fit produced -- that fit used a solver bug (unbounded adaptive
RK45 striding over the growth window) and its optimum was an artifact. With the
integrator corrected, the feature is a BINARY THRESHOLD at every r0: the bubble
either never clears the Laplace barrier (stays at the floor r0) or clears it and
runs to the ceiling, with essentially nothing in between (once past the barrier,
2*sigma/R collapses and growth self-accelerates). Larger r0 just lowers the
threshold until every dive saturates -- degenerate at the ceiling. 0.7 um gives
the most balanced floor/ceiling split; even so, across real dives the split
correlates NEGATIVELY with DCS (a proxy for "deep short dive"), which is why the
benchmark does not expect this feature to reach SUPPORTED. See Corrections 12-13.
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
    """Bubble radius trajectory in metres. Skin floor at r0_m; ceiling at ceiling_m.

    Two numerical facts make this correct rather than merely plausible:

    - **max_step is bounded to the forcing grid.** The RHS reads P_tissue and P_amb
      via np.interp on the profile's ~30 s grid, and the skin clamp makes dR/dt = 0
      while the nucleus sits at the floor. An unbounded adaptive RK45 sees that flat
      derivative, takes minutes-long steps, and strides clean over the window where
      supersaturation turns positive -- returning R_max = R0 on dives that actually
      grow (verified: a dive whose true R_max is 100 um reported 4.0 um, and refining
      rtol to 1e-10 did NOT fix it, because it is a step-size problem, not a tolerance
      one). Bounding max_step to the forcing resolution resolves that window.
    - **A terminal event stops integration at the ceiling** instead of a discontinuous
      derivative clamp. Once R clears the Laplace barrier, 2*sigma/R collapses and
      growth is self-accelerating; the event fires cleanly and the trajectory is
      padded at the ceiling for the remaining eval points.
    """
    P_tissue, P_amb = _tissue_and_ambient(profile)
    ts = profile.t_min * 60.0
    max_step = float(np.min(np.diff(ts)))         # bound to the forcing grid

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

    def _hit_ceiling(t, y):
        return y[0] - ceiling_m
    _hit_ceiling.terminal = True
    _hit_ceiling.direction = 1

    sol = solve_ivp(rhs, (ts[0], ts[-1]), [r0_m], method="RK45", t_eval=ts,
                    rtol=rtol, atol=1e-12, max_step=max_step, events=_hit_ceiling)
    if not sol.success:
        raise AlgorithmError(f"ep_bubble: solve failed on {profile.dive_id}")
    R = sol.y[0]
    if len(R) < len(ts):                          # event terminated early at the ceiling
        R = np.concatenate([R, np.full(len(ts) - len(R), ceiling_m)])
    return np.clip(R, r0_m, ceiling_m)


class EPBubble:
    name = "ep_bubble"

    def __init__(self, r0_um: float = 0.7, sigma: float = 0.050,
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

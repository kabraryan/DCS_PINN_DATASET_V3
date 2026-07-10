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

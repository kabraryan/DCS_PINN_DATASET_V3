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

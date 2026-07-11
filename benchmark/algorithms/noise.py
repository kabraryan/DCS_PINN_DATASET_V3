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

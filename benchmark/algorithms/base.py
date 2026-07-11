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

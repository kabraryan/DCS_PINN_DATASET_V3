"""Algorithm registry. Adding one is a new file plus one line here."""
from __future__ import annotations

from typing import Dict

from benchmark.algorithms.base import Algorithm, AlgorithmError
from benchmark.algorithms.ep_bubble import EPBubble
from benchmark.algorithms.zhl16c import ZHL16C
from benchmark.algorithms.zhl16c_gf import ZHL16CGF

REGISTRY: Dict[str, Algorithm] = {
    "zhl16c": ZHL16C(),
    "zhl16c_gf": ZHL16CGF(),
    "ep_bubble": EPBubble(),
}

__all__ = ["REGISTRY", "Algorithm", "AlgorithmError"]

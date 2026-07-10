"""Algorithm registry. Adding one is a new file plus one line here."""
from __future__ import annotations

from typing import Dict

from benchmark.algorithms.base import Algorithm, AlgorithmError
from benchmark.algorithms.zhl16c import ZHL16C

REGISTRY: Dict[str, Algorithm] = {
    "zhl16c": ZHL16C(),
}

__all__ = ["REGISTRY", "Algorithm", "AlgorithmError"]

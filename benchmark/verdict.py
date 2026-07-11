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

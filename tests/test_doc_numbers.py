"""CI guard: the docs' load-bearing numbers must match what the code produces.

This is the durable fix for the project's recurring failure — prose drifting from
code even after a "Provenance discipline" rule was written, because `--check`
guarded only RESULTS.md while the Corrections' inline statistics sat unguarded and
three of them drifted after the EP solver fix.

The slow (ODE-derived) claim is cached by the SHA of its producing source files
(`.doc_numbers_cache.json`, committed), so this test is instant unless the solver
or its harness changes — in which case the claim re-runs, which is exactly when
you want the doc re-verified. See scripts/check_doc_numbers.py.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import check_doc_numbers as cdn  # noqa: E402

REAL_CSV = cdn.REAL_CSV


@pytest.mark.skipif(not os.path.exists(REAL_CSV),
                    reason="real NMRC dataset not present on this machine")
def test_doc_numbers_match_code():
    """Every registered doc number regenerates to the value the doc contains.

    Runs the slow ODE claim too, but it is served from the committed source-hashed
    cache unless the physics changed, so this stays fast in normal CI.
    """
    rc = cdn.run(check=True, include_slow=True)
    assert rc == 0, ("A doc number drifted from the code that produces it. "
                     "Run `python scripts/check_doc_numbers.py --check --slow` to see "
                     "which, then regenerate the doc text (or fix the code).")

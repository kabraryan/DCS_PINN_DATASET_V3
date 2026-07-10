from __future__ import annotations

from benchmark.evaluate import GateResult
from benchmark.verdict import PRIMARY_MARGINAL, verdict

RECONS = ("linear", "staged")
RULES = ("exclude", "positive", "negative")


def g(passed: bool, sign: float = 1.0) -> GateResult:
    return GateResult(passed, 0.05, 0.001, 0.9, sign, 1.0, () if passed else ("x",))


def grid(overrides=None):
    base = {(r, m): g(True) for r in RECONS for m in RULES}
    if overrides:
        base.update(overrides)
    return base


def test_supported_when_everything_passes():
    assert verdict(grid()) == "SUPPORTED"


def test_reconstruction_sensitive():
    assert verdict(grid({("linear", PRIMARY_MARGINAL): g(False)})) \
        == "RECONSTRUCTION-SENSITIVE"


def test_marginal_sensitive():
    assert verdict(grid({("staged", "negative"): g(False)})) == "MARGINAL-SENSITIVE"


def test_not_supported_when_primary_fails_both():
    assert verdict(grid({("linear", PRIMARY_MARGINAL): g(False),
                         ("staged", PRIMARY_MARGINAL): g(False)})) == "NOT SUPPORTED"


def test_sensitivity_runs_cannot_promote():
    """Fails under exclude, passes under positive -> still NOT SUPPORTED."""
    cells = {(r, m): g(m == "positive") for r in RECONS for m in RULES}
    assert verdict(cells) == "NOT SUPPORTED"


def test_missing_cell_raises():
    cells = grid()
    del cells[("staged", "positive")]
    try:
        verdict(cells)
    except KeyError:
        return
    raise AssertionError("missing cell must raise, never default to a pass")

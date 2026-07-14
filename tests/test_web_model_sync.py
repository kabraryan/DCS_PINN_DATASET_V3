"""CI guard: the numbers baked into the web app must match what the code produces.

`decostress_app/index.html` carries the fitted model as literal JS constants --
the scaler stats, the coefficients, the intercept, the cohort quantiles, and the
ZHL-16C table. They were produced by `scripts/export_web_model.py` and pasted in.

That is precisely the shape of Correction 10: compartment 16's b-coefficient was
hand-copied into four files, diverged, and reached a README, two design docs, and
a unit test that asserted the bug. `RESULTS.md --check` and `check_doc_numbers.py`
guard the docs; without this, the web app would be the one unguarded copy -- and
the one a human actually looks at.

So: refit from the real CSV and assert the HTML still agrees. If this fails, the
model changed and the app is serving stale numbers. Re-run:

    python -m scripts.export_web_model
    # then update the constants in decostress_app/index.html
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from benchmark.buhlmann import zhl16c_table

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "decostress_app", "index.html")
REAL_CSV = os.path.expanduser("~/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_CSV),
    reason="real NMRC 99-02 CSV not present; nothing to refit against",
)


def _html() -> str:
    with open(APP) as f:
        return f.read()


def _js_array(name: str, text: str) -> list[float]:
    """Pull `name: [1, 2, 3]` or `const name = [...]` out of the JS."""
    m = re.search(rf"\b{name}\s*[:=]\s*\[([^\]]*)\]", text)
    assert m, f"{name} not found in {APP}"
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", m.group(1))]


def _refit():
    """Refit exactly as scripts/export_web_model.py does."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from scripts.export_web_model import pick_C
    from scripts.run_benchmark import load_dives

    dives, y, groups, _ = load_dives(None, "exclude")
    X = np.array([[d.depth_fsw, d.bottom_time_min, d.ascent_time_min] for d in dives])
    C = pick_C(X, y, groups)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    scores = clf.decision_function(sc.transform(X))
    return sc, clf, scores, len(dives), int(y.sum())


def test_zhl16c_table_matches_the_shared_module():
    """The app's 16 compartments must BE the repo's table -- not a copy that drifted.

    This is the exact check that would have caught Correction 10.
    """
    text = _html()
    m = re.search(r"const ZHL\s*=\s*\[(.*?)\];", text, re.S)
    assert m, "ZHL table not found in the app"
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(1))]
    app_table = np.array(nums, dtype=float).reshape(-1, 3)
    assert app_table.shape == (16, 3), f"expected 16x3, got {app_table.shape}"
    np.testing.assert_allclose(app_table, zhl16c_table(), rtol=0, atol=1e-9)


def test_fitted_coefficients_match_a_fresh_fit():
    text = _html()
    sc, clf, _, _, _ = _refit()

    np.testing.assert_allclose(_js_array("mean", text), sc.mean_, atol=1e-5)
    np.testing.assert_allclose(_js_array("scale", text), sc.scale_, atol=1e-5)
    np.testing.assert_allclose(_js_array("coef", text), clf.coef_[0], atol=1e-5)

    m = re.search(r"intercept:\s*(-?\d+\.\d+)", text)
    assert m, "intercept not found"
    assert float(m.group(1)) == pytest.approx(float(clf.intercept_[0]), abs=1e-5)


def test_cohort_quantiles_match_a_fresh_fit():
    """The percentile the app shows is only meaningful if these are current."""
    text = _html()
    _, _, scores, _, _ = _refit()
    want = np.quantile(scores, np.linspace(0.0, 1.0, 101))
    got = np.array(_js_array("RM_Q", text))
    assert got.shape == want.shape, f"expected 101 quantiles, got {got.shape[0]}"
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_cohort_size_claims_match_the_data():
    """The app tells the user '1,948 real Navy dives, 305 DCS'. It had better be true."""
    text = _html()
    _, _, _, n, n_dcs = _refit()
    m = re.search(r"n:\s*(\d+),\s*nDcs:\s*(\d+)", text)
    assert m, "n / nDcs not found in the app's RM block"
    assert int(m.group(1)) == n, f"app claims n={m.group(1)}, data has {n}"
    assert int(m.group(2)) == n_dcs, f"app claims nDcs={m.group(2)}, data has {n_dcs}"
    # and the prose the user actually reads
    assert f"{n:,}" in text, f"the app's prose should cite {n:,} dives"


def test_app_emits_no_probability_claim():
    """The benchmark's standing rule: never emit a probability. Guard it in the UI too.

    AUC is a ranking. The ~16% DCS rate comes from Navy trials designed to provoke
    DCS; a probability derived from it is meaningless for a real diver.

    The first version of this test listed two exact phrasings and missed a bare
    `P(DCS)` sitting in the chart legend -- the user found it on screen. Match the
    token itself, anywhere it could reach a user's eye, not a list of sentences.
    """
    text = _html()

    # Any P(DCS) that is not inside a source comment saying we don't emit one.
    for m in re.finditer(r"P\(DCS\)", text):
        line = text[text.rfind("\n", 0, m.start()) + 1 : text.find("\n", m.end())]
        assert "never" in line or "NOT " in line or "not a" in line.lower(), (
            f"a literal P(DCS) can reach the user here:\n    {line.strip()}"
        )

    banned = [
        "chance of decompression sickness at",
        "MODELED P(DCS)",
        "probability of decompression sickness",
        "modeled probability",
    ]
    for phrase in banned:
        assert phrase not in text, f"the app must not claim a probability: {phrase!r}"


def test_app_bottom_time_excludes_descent_like_the_training_data():
    """bottom_time is TIME AT DEPTH. The app must not count the descent into it.

    benchmark/profile.py settles the semantics: _materialise() lays out
    descent(desc_min) THEN bottom(bottom_time_min), and _load_to_bottom()
    descends and THEN sits for bottom_time_min. Descent is separate.

    The app first measured bottom time from t=0 -- i.e. including the descent --
    and then fed that into its own loadToBottom(), which descends AGAIN. The
    descent was counted twice, inflating the computed decompression obligation by
    ~1.5 min on a 30 m dive (a 30m/25min dive that honoured its obligation exactly
    still reported a 1.5-min deficit and got voided).

    Caught by sweeping ascent time across the obligation and noticing the deficit
    was not zero at the crossing point.
    """
    text = _html()
    m = re.search(r"const bottom\s*=\s*Math\.max\(([^;]+)\)", text)
    assert m, "could not find the app's bottom-time extraction"
    expr = m.group(1)
    assert "iArrive" in expr, (
        f"bottom time is computed as `{expr.strip()}` -- it must subtract the time "
        "of ARRIVING at the bottom, or the descent is counted twice (once here and "
        "again inside loadToBottom) and the deco obligation is overstated"
    )


def test_dial_is_voided_on_a_decompression_violation():
    """The dial must NOT show a reassuring rank on a dive that skips its deco.

    Found in use: a 45 m / 18 min dive that skipped 19 of its 22.4 mandatory
    decompression minutes -- Bühlmann loading 1.27x the surfacing limit, i.e.
    genuinely bent -- still displayed "42nd percentile, MILDER THAN MOST NAVY
    DIVES" on the headline dial. The refusal existed, but only in the assessment
    text below the fold. A reassuring headline on a bent dive is worse than no
    number at all.

    The model's rank is only meaningful inside the world it was trained on, where
    every diver followed their schedule. Break the schedule and the rank is void.
    """
    text = _html()
    assert "voided" in text, "the app must be able to void the model's rank"
    assert "MODEL VOID" in text, "the voided dial must say so, in the band, unmissably"
    assert "DECO VIOLATION" in text, "the voided dial must name the reason"
    assert "ceilingViolationM" in text, (
        "the app must detect a live ceiling violation mid-dive, not only on surfacing"
    )
    # the dial's value must fall back to the physics when voided
    assert re.search(r"if\(voided\)\{[\s\S]{0,400}?currentStress\(\)", text), (
        "when the model is voided the dial must fall back to Bühlmann loading, "
        "which is still valid -- not keep showing the model's percentile"
    )


def test_ascent_coefficient_is_positive_so_the_obligation_guard_must_exist():
    """The safety-critical invariant.

    The fitted ascent coefficient is POSITIVE: fed a diver-chosen ascent time, the
    model calls a fast ascent safer. It is only safe to display because the app
    scores the decompression OBLIGATION (computed from the tissues), never a number
    the diver picked -- and refuses outright on a dive that skips its obligation.

    If someone ever rewires the app to score `sc.ascent` directly, this fails.
    """
    _, clf, _, _, _ = _refit()
    assert clf.coef_[0][2] > 0, (
        "ascent coefficient is no longer positive -- the inversion this guard "
        "protects against may have changed; re-derive the app's safety argument"
    )
    text = _html()
    assert "requiredAscentMin" in text, "the app must compute the deco obligation"
    assert "violated" in text, "the app must detect a skipped obligation"
    assert "cannot score this dive" in text, (
        "the app must REFUSE to score a dive that skips its obligation -- that dive "
        "is out of distribution and the positive ascent coefficient would call it safer"
    )
    # the model must be scored on the obligation, not the diver's ascent
    m = re.search(r"const scored\s*=\s*\{[^}]*ascent:\s*([A-Za-z_.]+)", text)
    assert m, "could not find what the app feeds the model as `ascent`"
    assert "req" in m.group(1), (
        f"the app feeds `{m.group(1)}` as ascent; it MUST feed the required "
        "obligation, or the model inverts and rewards skipping the safety stop"
    )

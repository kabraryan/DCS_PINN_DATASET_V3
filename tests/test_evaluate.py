from __future__ import annotations

import numpy as np
import pytest

from benchmark.evaluate import (
    baseline_auc, four_gate, leakage_gap, nested_grouped_cv, shuffle_control,
)

RNG = np.random.RandomState(0)


def synthetic(n=600, n_groups=12, signal=1.5, seed=0):
    """Feature with real signal; groups carry their own intercept (trial effect)."""
    rng = np.random.RandomState(seed)
    g = rng.randint(0, n_groups, n)
    x = rng.normal(size=n)
    trial_effect = rng.normal(scale=2.0, size=n_groups)[g]
    logit = signal * x + trial_effect - 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X_raw = rng.normal(size=(n, 3))
    return X_raw, x, y, g


def test_shuffled_labels_collapse_to_chance():
    """A sound protocol must return ~0.5 on shuffled labels."""
    X_raw, x, y, g = synthetic()
    auc = shuffle_control(np.column_stack([X_raw, x]), y, g, seed=1)
    assert abs(auc - 0.5) < 0.08, f"protocol broken, not the model: {auc:.3f}"


def test_leakage_gap_is_positive_when_groups_carry_signal():
    """Ordinary CV lets a model memorise trial identity; grouped CV does not."""
    X_raw, x, y, g = synthetic(signal=0.1)      # weak feature, strong trial effect
    assert leakage_gap(np.column_stack([X_raw, x]), y, g) > 0.0


def test_real_signal_passes_all_four_gates():
    X_raw, x, y, g = synthetic(signal=2.5)
    r = four_gate(X_raw, x, y, g)
    assert r.passed, r.reasons
    assert r.delta > 0.03 and r.sign > 0


def test_pure_noise_fails():
    X_raw, x, y, g = synthetic(signal=0.0)
    r = four_gate(X_raw, RNG.normal(size=len(y)), y, g)
    assert not r.passed


def test_sign_inverted_feature_fails_even_though_it_helps():
    """The gate that caught the 'deep short dive' confound.

    Negating a genuinely predictive feature keeps |delta| and p identical, but the
    coefficient sign flips. That must be rejected.
    """
    X_raw, x, y, g = synthetic(signal=2.5)
    good = four_gate(X_raw, x, y, g)
    flipped = four_gate(X_raw, -x, y, g)
    assert good.passed
    assert not flipped.passed
    assert "sign" in " ".join(flipped.reasons)
    assert flipped.delta == pytest.approx(good.delta, abs=1e-6)


def test_baseline_is_reproducible_and_matches_nested_cv_on_same_X():
    X_raw, x, y, g = synthetic()
    a = baseline_auc(X_raw, y, g, seed=3)
    b = nested_grouped_cv(X_raw, y, g, seed=3)
    np.testing.assert_allclose(a, b)

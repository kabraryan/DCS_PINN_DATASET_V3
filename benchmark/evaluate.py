"""Evaluation protocol. Knows nothing about decompression.

Three properties, each of which silently inflates results on this dataset:

1. GROUPED outer folds. Trials differ in protocol aggressiveness (DCS 4.6%-35%);
   ordinary CV inflates AUC by +0.045 to +0.075 by letting a model memorise
   trial identity.
2. NESTED tuning. Hyperparameters chosen on outer test folds re-import that leak.
3. A LABEL-SHUFFLE control. Above chance on shuffled labels means the protocol is
   broken, not the model.

Effective sample size is ~38 trials, not ~1,948 dives. Fold sd ~ 0.06, so an
effect must clear MIN_DELTA and a paired test, not merely a mean comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_DELTA = 0.03
MAX_P = 0.05
MIN_FRAC_IMPROVED = 0.75
MIN_NULL_FRAC = 0.95
C_GRID = (0.03, 0.1, 0.3, 1.0, 1e6)


def _pipe(C: float) -> Pipeline:
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])


def _fit_score(C, Xtr, ytr, Xte, yte) -> float:
    m = _pipe(C).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def nested_grouped_cv(X, y, groups, *, n_rep: int = 5, seed: int = 0,
                      n_outer: int = 5, n_inner: int = 3) -> np.ndarray:
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    out: List[float] = []
    for _ in range(n_rep):
        gmap = {g: i for i, g in enumerate(rng.permutation(uniq))}
        gi = np.array([gmap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_outer).split(X, y, gi):
            if len(np.unique(y[te])) < 2:
                continue
            best_C, best = C_GRID[0], -np.inf
            if len(np.unique(gi[tr])) >= n_inner:
                for C in C_GRID:
                    inner = [
                        _fit_score(C, X[tr][itr], y[tr][itr], X[tr][ite], y[tr][ite])
                        for itr, ite in GroupKFold(n_inner).split(X[tr], y[tr], gi[tr])
                        if len(np.unique(y[tr][ite])) >= 2
                    ]
                    if inner and np.mean(inner) > best:
                        best, best_C = float(np.mean(inner)), C
            out.append(_fit_score(best_C, X[tr], y[tr], X[te], y[te]))
    return np.asarray(out)


def baseline_auc(X_raw, y, groups, **kw) -> np.ndarray:
    """The number to beat: logistic on the raw dive parameters, nothing else."""
    return nested_grouped_cv(X_raw, y, groups, **kw)


def shuffle_control(X, y, groups, *, seed: int = 0) -> float:
    rng = np.random.RandomState(seed + 99)
    return float(nested_grouped_cv(X, rng.permutation(np.asarray(y)), groups,
                                   n_rep=1, seed=seed).mean())


def leakage_gap(X, y, groups, *, seed: int = 0) -> float:
    """Ordinary CV minus grouped CV. Printed so nobody quotes the inflated number."""
    X, y = np.asarray(X, float), np.asarray(y)
    ordinary = [
        _fit_score(1.0, X[tr], y[tr], X[te], y[te])
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    ]
    grouped = nested_grouped_cv(X, y, groups, n_rep=1, seed=seed)
    return float(np.mean(ordinary) - grouped.mean())


@dataclass(frozen=True)
class GateResult:
    passed: bool
    delta: float
    p_value: float
    frac_improved: float
    sign: float
    beats_null_frac: float
    reasons: Tuple[str, ...]


def _coefficient_sign(X_raw, x, y) -> float:
    X = np.column_stack([np.asarray(X_raw, float), np.asarray(x, float)])
    m = _pipe(1.0).fit(X, y)
    return float(np.sign(m.named_steps["clf"].coef_[0][-1]))


def four_gate(X_raw, x, y, groups, *, n_rep: int = 5, seed: int = 0,
              n_null: int = 20) -> GateResult:
    """An effect counts only if it clears magnitude, significance, SIGN, and a null."""
    X_raw = np.asarray(X_raw, float)
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y)

    base = baseline_auc(X_raw, y, groups, n_rep=n_rep, seed=seed)
    full = nested_grouped_cv(np.column_stack([X_raw, x]), y, groups,
                             n_rep=n_rep, seed=seed)
    n = min(len(base), len(full))
    d = full[:n] - base[:n]

    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    sign = _coefficient_sign(X_raw, x, y)

    rng = np.random.RandomState(seed + 7)
    null = np.array([
        (nested_grouped_cv(np.column_stack([X_raw, rng.permutation(x)]), y, groups,
                           n_rep=1, seed=seed + k)[:n].mean() - base[:n].mean())
        for k in range(n_null)
    ])
    beats = float(np.mean(d.mean() > null))

    reasons: List[str] = []
    if abs(d.mean()) <= MIN_DELTA:
        reasons.append(f"magnitude |{d.mean():+.4f}| <= {MIN_DELTA}")
    if np.isnan(p) or p >= MAX_P:
        reasons.append(f"p={p:.3f} >= {MAX_P}")
    if np.mean(d > 0) < MIN_FRAC_IMPROVED:
        reasons.append(f"only {np.mean(d > 0):.0%} of folds improved")
    if sign <= 0:
        reasons.append("sign inverted: feature predicts FEWER events (confound)")
    if beats < MIN_NULL_FRAC:
        reasons.append(f"beats only {beats:.0%} of permutations")

    return GateResult(not reasons, float(d.mean()), p, float(np.mean(d > 0)),
                      sign, beats, tuple(reasons))

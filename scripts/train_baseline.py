"""Baseline training harness for real DCS dive outcomes (NMRC 99-02).

Establishes the number any physics feature must beat, under an evaluation protocol
that does not lie to you.

Three things this harness enforces, because each of them silently inflates results
on this dataset:

  1. GROUPED evaluation. Trials differ in protocol aggressiveness (DCS rates run
     4.6% to 35%). A model that memorises trial identity scores ~+0.08 AUC higher
     under ordinary CV and generalises nowhere. Outer folds group by `data_set`.
  2. NESTED tuning. Hyperparameters selected on the outer test folds re-import the
     leakage you just removed. The inner loop is also grouped.
  3. A LABEL-SHUFFLE control. If the pipeline reports much above 0.5 on shuffled
     labels, the protocol is broken, not the model.

Effective sample size is ~38 trials, not ~1,948 dives. Fold-to-fold sd is ~0.06,
so AUC differences below ~0.03 are noise. The harness prints paired fold deltas
and a Wilcoxon p-value rather than inviting you to compare two means.

Absolute probabilities are meaningless here: the ~16% DCS rate reflects trials
designed to provoke DCS plus partial negative extraction (2,700 of 8,578 dives).
Brier score is reported for completeness and should not be read as calibration
against any diving population.

Usage:
    python scripts/train_baseline.py                       # raw features, marginals excluded
    python scripts/train_baseline.py --features raw+physics  # adds prs and bubble R_max (slow)
    python scripts/train_baseline.py --marginal positive
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

REAL_CSV = os.path.expanduser('~/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv')
RAW_COLS = ['depth_fsw', 'bottom_time_min', 'ascent_time_min']

# Bounce-dive window; see Correction 13. Saturation exposures and >300 fsw
# excursions are a different physiological regime. The exclusion is
# outcome-correlated (24.3% vs 13.7% DCS) and is reported, not hidden.
MAX_DEPTH_FSW = MAX_BOTTOM_MIN = MAX_ASCENT_MIN = 300.0


# ── model zoo: (name, estimator factory, hyperparameter grid) ────────────────
def baseline_model():
    """Unpenalised logistic on the THREE RAW FEATURES ONLY. The number to beat.

    Held fixed regardless of --features, so adding physics columns is compared
    against the naive dive parameters rather than against itself.
    """
    return ('logistic on 3 raw features [BASELINE]', lambda: Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000))]),
        {'clf__C': [1e6]})


def model_zoo():
    return [
        ('logistic', lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000))]),
         {'clf__C': [1e6]}),

        ('logistic + L2', lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000))]),
         {'clf__C': [0.03, 0.1, 0.3, 1.0]}),

        ('HistGradientBoosting', lambda: HistGradientBoostingClassifier(random_state=0),
         {'max_depth': [2, 3], 'learning_rate': [0.05, 0.1], 'max_iter': [150]}),

        ('RandomForest', lambda: RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
         {'min_samples_leaf': [10, 20, 40]}),
    ]


def _grid(params):
    keys = list(params)
    out = [{}]
    for k in keys:
        out = [dict(d, **{k: v}) for d in out for v in params[k]]
    return out


def _fit_predict(factory, hp, Xtr, ytr, Xte):
    m = factory()
    m.set_params(**hp)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def nested_grouped_cv(X, y, groups, factory, params, n_rep, seed, n_outer=5, n_inner=3):
    """Outer GroupKFold for scoring; inner GroupKFold for hyperparameter choice."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    aucs, aps, briers, chosen = [], [], [], []

    for _ in range(n_rep):
        gmap = {g: i for i, g in enumerate(rng.permutation(uniq))}
        gi = np.array([gmap[g] for g in groups])

        for tr, te in GroupKFold(n_splits=n_outer).split(X, y, gi):
            if len(np.unique(y[te])) < 2:
                continue
            # ── inner loop: pick hyperparameters without touching the test fold
            best_hp, best_score = None, -np.inf
            gi_tr = gi[tr]
            if len(np.unique(gi_tr)) >= n_inner:
                for hp in _grid(params):
                    inner = []
                    for itr, ite in GroupKFold(n_splits=n_inner).split(X[tr], y[tr], gi_tr):
                        if len(np.unique(y[tr][ite])) < 2:
                            continue
                        p = _fit_predict(factory, hp, X[tr][itr], y[tr][itr], X[tr][ite])
                        inner.append(roc_auc_score(y[tr][ite], p))
                    if inner and np.mean(inner) > best_score:
                        best_score, best_hp = np.mean(inner), hp
            if best_hp is None:
                best_hp = _grid(params)[0]

            p = _fit_predict(factory, best_hp, X[tr], y[tr], X[te])
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
            briers.append(brier_score_loss(y[te], p))
            chosen.append(tuple(sorted(best_hp.items())))

    return np.array(aucs), np.array(aps), np.array(briers), chosen


def ungrouped_cv(X, y, factory, params, seed):
    """Ordinary stratified CV -- shown ONLY to quantify trial leakage."""
    hp = _grid(params)[0]
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p = _fit_predict(factory, hp, X[tr], y[tr], X[te])
        out.append(roc_auc_score(y[te], p))
    return np.array(out)


def load(marginal, csv):
    df = pd.read_csv(csv)
    n_all = len(df)
    keep = ((df.depth_fsw <= MAX_DEPTH_FSW) & (df.bottom_time_min <= MAX_BOTTOM_MIN)
            & (df.ascent_time_min <= MAX_ASCENT_MIN))
    dropped = df[~keep]
    df = df[keep].copy()
    print(f'Loaded {n_all} dives; kept {len(df)} bounce dives ({len(df)/n_all:.1%}). '
          f'Dropped {len(dropped)} saturation/deep-excursion '
          f'(DCS {(dropped.outcome==1.0).mean():.1%} vs kept {(df.outcome==1.0).mean():.1%}).')

    if marginal == 'exclude':
        df = df[df.outcome != 0.5].copy()
        rule = 'marginals dropped'
    elif marginal == 'positive':
        df['outcome'] = (df.outcome > 0.0).astype(float)
        rule = 'marginal 0.5 -> 1 (over-warns)'
    else:
        df['outcome'] = (df.outcome >= 1.0).astype(float)
        rule = 'marginal 0.5 -> 0 (under-warns)'
    return df, rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--marginal', choices=['exclude', 'positive', 'negative'], default='exclude',
                    help='How to treat the 0.5 marginal-DCS class. A safety choice, not a default.')
    ap.add_argument('--features', choices=['raw', 'raw+physics'], default='raw')
    ap.add_argument('--ascent', choices=['linear', 'staged'], default='staged')
    ap.add_argument('--r0-um', type=float, default=4.0, help='VPM nucleus radius for the bubble feature')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--csv', default=REAL_CSV)
    args = ap.parse_args()

    df, rule = load(args.marginal, args.csv)
    y = (df.outcome.values >= 1.0).astype(int)
    groups = df.data_set.values
    X_raw = df[RAW_COLS].values.astype(float)
    X = X_raw.copy()
    names = list(RAW_COLS)

    if args.features == 'raw+physics':
        from fit_r0_to_real_dives import buhlmann, bubble_rmax, zhl16c, depth_profile
        from staged_ascent import build_profile as staged_profile
        table = zhl16c()
        print(f'Computing physics features ({args.ascent} ascent, R0={args.r0_um} um)...')
        prs = np.empty(len(df)); rmax = np.empty(len(df))
        R0 = args.r0_um * 1e-6
        for i, (_, r) in enumerate(df.iterrows()):
            if args.ascent == 'staged':
                t, d = staged_profile(r.depth_fsw, r.bottom_time_min, r.ascent_time_min, table)
            else:
                t, d = depth_profile(r.depth_fsw, r.bottom_time_min, r.ascent_time_min)
            Pt, Pa, ratio = buhlmann(t, d, table)
            prs[i] = ratio
            rmax[i] = bubble_rmax(t, Pt, Pa, R0)
        ok = np.isfinite(rmax) & (rmax > 0)
        logr = np.where(ok, np.log(np.maximum(rmax, R0) / R0), 0.0)
        X = np.column_stack([X_raw, prs, logr])
        names += ['physics_risk_score', 'log_bubble_R_max']

    ev = int(y.sum())
    print(f'\nmarginal rule : {rule}')
    print(f'n = {len(df)}, positives = {ev} ({y.mean():.1%}), trials = {df.data_set.nunique()}')
    print(f'features ({len(names)}): {", ".join(names)}')
    print(f'events per predictor = {ev/len(names):.1f}   '
          f'(effective n is ~{df.data_set.nunique()} TRIALS, not {len(df)} dives)')
    print(f'\nNested grouped CV: outer GroupKFold(5) x {args.repeats} repeats, inner GroupKFold(3)\n')

    print(f'{"model":34s} {"AUC":>16s} {"PR-AUC":>9s} {"Brier":>8s} {"min fold":>9s}')
    print('-' * 82)

    # Baseline is ALWAYS the 3 raw dive parameters, whatever --features says.
    b_name, b_factory, b_params = baseline_model()
    base, b_apr, b_bri, _ = nested_grouped_cv(
        X_raw, y, groups, b_factory, b_params, args.repeats, args.seed)
    print(f'{b_name:34s} {base.mean():.4f} +/- {base.std():.4f} '
          f'{b_apr.mean():9.4f} {b_bri.mean():8.4f} {base.min():9.3f}')

    results = {}
    for name, factory, params in model_zoo():
        auc, apr, bri, _ = nested_grouped_cv(
            X, y, groups, factory, params, args.repeats, args.seed)
        results[name] = auc
        tag = f'{name} [{args.features}]'
        print(f'{tag:34s} {auc.mean():.4f} +/- {auc.std():.4f} '
              f'{apr.mean():9.4f} {bri.mean():8.4f} {auc.min():9.3f}')

    print(f'\nPaired fold deltas vs the BASELINE (logistic on {len(RAW_COLS)} raw features):')
    for name, auc in results.items():
        n = min(len(auc), len(base))
        d = auc[:n] - base[:n]
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float('nan')
        verdict = 'real' if (abs(d.mean()) > 0.03 and p < 0.05) else 'NOISE (|d|<0.03 or p>=.05)'
        print(f'  {name:22s} delta {d.mean():+.4f} +/- {d.std():.4f}  '
              f'improved {100*(d>0).mean():5.1f}%  p={p:.3f}  -> {verdict}')
    if args.features == 'raw+physics':
        print('\n  Physics columns earn their place only if a model on `raw+physics` beats the')
        print('  raw baseline by more than ~0.03 AUC. Compare to the same model on `raw`.')

    # ── control 1: trial leakage ────────────────────────────────────────────
    print('\n' + '='*72)
    print('CONTROL 1 — trial leakage: same model, ordinary CV vs grouped CV')
    print('='*72)
    for name, factory, params in model_zoo():
        if name not in ('logistic', 'HistGradientBoosting'):
            continue
        u = ungrouped_cv(X, y, factory, params, args.seed)
        g = results[name]
        print(f'  {name:24s} ordinary {u.mean():.4f}   grouped {g.mean():.4f}   '
              f'phantom +{u.mean()-g.mean():.4f}')
    print('  Ordinary CV lets a model memorise trial identity. Never report it.')

    # ── control 2: shuffled labels must collapse to chance ──────────────────
    print('\n' + '='*72)
    print('CONTROL 2 — label shuffle: a sound protocol must return ~0.5')
    print('='*72)
    rng = np.random.RandomState(args.seed + 99)
    name, factory, params = baseline_model()
    shuf = [nested_grouped_cv(X_raw, rng.permutation(y), groups, factory, params, 1, args.seed + k)[0].mean()
            for k in range(5)]
    shuf = np.array(shuf)
    print(f'  {name} on shuffled labels: AUC {shuf.mean():.4f} +/- {shuf.std():.4f}')
    print(f'  {"OK" if abs(shuf.mean()-0.5) < 0.06 else "PROTOCOL BROKEN"} '
          f'(expected ~0.500)')

    print('\n' + '-'*72)
    print('Read AUC as ranking only. The ~16% DCS rate reflects Navy trials designed to')
    print('provoke DCS plus partial negative extraction (2,700 of 8,578 dives); Brier is')
    print('not calibration against any diving population. Fold sd ~0.06, so differences')
    print('below ~0.03 AUC are noise -- hence the paired test above, not a leaderboard.')


if __name__ == '__main__':
    main()

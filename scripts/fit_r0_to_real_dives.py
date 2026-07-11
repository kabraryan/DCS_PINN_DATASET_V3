"""Fit the VPM nucleus radius R0 to real dive outcomes (NMRC 99-02).

Estimates R0 by profile likelihood against the 2,700 real dives in
`FINAL DIVE/datasets/real/dcs_all_dives.csv`, which contain real non-DCS controls,
then asks the falsifiable question the synthetic dataset cannot: does a bubble
feature improve out-of-sample discrimination over Buhlmann tissue loading alone?

Pipeline per dive:
    (depth_fsw, bottom_time_min, ascent_time_min)
      -> square depth profile (descent 60 fsw/min, bottom, linear ascent, surface watch)
      -> Buhlmann ZHL-16C on air              -> P_tissue(t), P_amb(t), prs
      -> Epstein-Plesset + VPM skin at R0     -> bubble_R_max
      -> logistic(outcome ~ prs + log R_max)  -> log-likelihood at this R0

R0 is chosen to maximise the profile likelihood. Discrimination is evaluated with
GroupKFold by `data_set`, because trials differ in protocol aggressiveness and
pooling them leaks trial identity into the estimate.

Usage:
    python scripts/fit_r0_to_real_dives.py [--marginal exclude|positive|negative]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staged_ascent import build_profile as staged_profile  # noqa: E402

warnings.filterwarnings('ignore')

REAL_CSV = os.path.expanduser('~/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv')

# ── constants ────────────────────────────────────────────────────────────────
# Shared Bühlmann constants come from the single source of truth; only the
# bubble-specific constants below are local. Duplicating the shared ones is how
# compartment 16's b-coefficient once diverged across four files (Correction 10).
from benchmark.buhlmann import (  # noqa: E402
    DESCENT_FSW_PER_MIN, FSW_TO_BAR, F_N2_AIR, P_SURFACE, zhl16c_table,
)

SURFACE_WATCH_MIN   = 120.0    # bubbles keep growing after surfacing
DT_MIN = 0.5

BAR_TO_PA = 1e5
D        = 2.0e-9
ALPHA_N2 = 6.84e-6
SIGMA    = 0.050
M_N2     = 0.028
R_GAS    = 8.314
T_BODY   = 310.15
R_CEILING = 100.0e-6
R_MIN     = 1e-9

# Bounce-dive window. ASAT*/SUREXM are multi-day air saturation exposures and
# SUBX87/UPS290 are >300 fsw excursions: different physiological regimes that a
# single-nucleus bounce model does not represent. Excluded on physiological
# grounds, not on outcome. The exclusion is outcome-correlated (saturation dives
# run ~28% DCS vs ~16% overall), so it is reported, not hidden.
MAX_DEPTH_FSW, MAX_BOTTOM_MIN, MAX_ASCENT_MIN = 300.0, 300.0, 300.0


def zhl16c():
    """Thin wrapper over the shared table (single source of truth, Correction 10)."""
    return zhl16c_table()


def depth_profile(depth_fsw, bt_min, at_min):
    """Square profile: descent, bottom, linear ascent, then surface watch."""
    desc = max(depth_fsw / DESCENT_FSW_PER_MIN, DT_MIN)
    total = desc + bt_min + at_min + SURFACE_WATCH_MIN
    t = np.arange(0.0, total + DT_MIN, DT_MIN)
    d = np.empty_like(t)

    m1 = t < desc
    d[m1] = depth_fsw * t[m1] / desc
    m2 = (t >= desc) & (t < desc + bt_min)
    d[m2] = depth_fsw
    m3 = (t >= desc + bt_min) & (t < desc + bt_min + at_min)
    d[m3] = depth_fsw * (1.0 - (t[m3] - desc - bt_min) / at_min)
    d[t >= desc + bt_min + at_min] = 0.0
    return t, d


def buhlmann(t_min, depth_fsw_series, table):
    """Return P_tissue_max(t) and P_amb(t) in bar, plus raw max M-value ratio."""
    a, b = table[:, 1], table[:, 2]
    k = np.log(2.0) / table[:, 0]
    P_amb = P_SURFACE + depth_fsw_series * FSW_TO_BAR
    P_alv = P_amb * F_N2_AIR

    P_t = np.full(16, P_SURFACE * F_N2_AIR)
    n = len(t_min)
    Pt_ts = np.empty(n)
    max_ratio = 0.0
    dt = np.diff(t_min, prepend=t_min[0])
    decay = None
    for i in range(n):
        if i > 0:
            decay = np.exp(-k * dt[i])
            P_t = P_alv[i] + (P_t - P_alv[i]) * decay
        Pt_ts[i] = P_t.max()
        mv = a + P_amb[i] / b
        max_ratio = max(max_ratio, float(np.max(P_t / mv)))
    return Pt_ts, P_amb, max_ratio


def _rhs(t, y, t_grid_s, Pt, Pa, floor):
    R = max(y[0], floor, R_MIN)
    if R >= R_CEILING:
        return [0.0]
    Pt_pa = np.interp(t, t_grid_s, Pt) * BAR_TO_PA
    Pa_pa = np.interp(t, t_grid_s, Pa) * BAR_TO_PA
    Pg = Pa_pa + 2.0 * SIGMA / R
    dC = ALPHA_N2 * M_N2 * (Pt_pa - Pg)
    if R <= max(floor, R_MIN) and dC < 0.0:
        return [0.0]                              # VPM skin arrests dissolution
    rho = Pg * M_N2 / (R_GAS * T_BODY)
    return [D * dC / (rho * R) * (1.0 + R / np.sqrt(np.pi * D * max(t, 1e-10)))]


def bubble_rmax(t_min, Pt, Pa, R0):
    # max_step bounded to the forcing grid + terminal ceiling event: the skin clamp
    # makes dR/dt = 0 at the floor, and unbounded adaptive RK45 strides over the
    # growth window, reporting R_max = R0 on dives that actually grow. See the
    # ep_bubble.py fix (2026-07-10).
    ts = t_min * 60.0
    max_step = float(np.min(np.diff(ts)))

    def _hit(t, y, *a):
        return y[0] - R_CEILING
    _hit.terminal = True
    _hit.direction = 1

    sol = solve_ivp(_rhs, (ts[0], ts[-1]), [R0], method='RK45', t_eval=ts,
                    args=(ts, Pt, Pa, R0), rtol=1e-6, atol=1e-12,
                    max_step=max_step, events=_hit)
    if not sol.success:
        return np.nan
    R = sol.y[0]
    if len(R) < len(ts):
        R = np.concatenate([R, np.full(len(ts) - len(R), R_CEILING)])
    return float(np.clip(R, R0, R_CEILING).max())


def logistic_ll(beta, X, y):
    z = X @ beta
    return -np.sum(y * z - np.logaddexp(0.0, z))


def fit_logistic(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    b0 = np.zeros(X.shape[1])
    r = minimize(logistic_ll, b0, args=(X, y), method='BFGS')
    return r.x, -r.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--marginal', choices=['exclude', 'positive', 'negative'], default='exclude')
    ap.add_argument('--ascent', choices=['linear', 'staged'], default='staged',
                    help='linear = straight ascent over ascent_time_min; '
                         'staged = Buhlmann-ceiling stops rescaled to ascent_time_min')
    ap.add_argument('--csv', default=REAL_CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n_all = len(df)
    keep = ((df.depth_fsw <= MAX_DEPTH_FSW) & (df.bottom_time_min <= MAX_BOTTOM_MIN)
            & (df.ascent_time_min <= MAX_ASCENT_MIN))
    dropped = df[~keep]
    df = df[keep].copy()

    print(f'Loaded {n_all} real dives; kept {len(df)} bounce dives '
          f'({len(df)/n_all:.1%}). Dropped {len(dropped)} '
          f'(saturation / deep excursion), whose DCS rate was '
          f'{(dropped.outcome==1.0).mean():.1%} vs kept {(df.outcome==1.0).mean():.1%}.')

    if args.marginal == 'exclude':
        df = df[df.outcome != 0.5].copy()
    elif args.marginal == 'positive':
        df['outcome'] = (df.outcome > 0.0).astype(float)
    else:
        df['outcome'] = (df.outcome >= 1.0).astype(float)
    y = (df.outcome.values >= 1.0).astype(float)
    groups = df.data_set.values
    print(f'marginal={args.marginal}: n={len(df)}, positives={int(y.sum())} ({y.mean():.1%}), '
          f'{df.data_set.nunique()} trials\n')

    table = zhl16c()

    # Forcing is independent of R0 -> compute once.
    print(f'Computing Buhlmann forcing ({args.ascent} ascent; independent of R0)...')
    forcing, prs = [], np.empty(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        if args.ascent == 'staged':
            t, d = staged_profile(r.depth_fsw, r.bottom_time_min, r.ascent_time_min, table)
        else:
            t, d = depth_profile(r.depth_fsw, r.bottom_time_min, r.ascent_time_min)
        Pt, Pa, ratio = buhlmann(t, d, table)
        forcing.append((t, Pt, Pa))
        prs[i] = ratio
    print(f'  max M-value ratio: median {np.median(prs):.3f}  p95 {np.percentile(prs,95):.3f}  '
          f'max {prs.max():.3f}')
    auc_prs = roc_auc_score(y, prs)
    print(f'  AUC(prs alone, in-sample) = {auc_prs:.4f}\n')

    # ── profile likelihood over R0 ───────────────────────────────────────────
    grid = np.array([0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]) * 1e-6
    print('Profile likelihood over R0 (logistic: outcome ~ prs + log R_max)')
    print(f'{"R0 (um)":>9} {"grew%":>7} {"logLik":>11} {"beta_bubble":>12} {"AUC in-samp":>12}')
    rows = []
    for R0 in grid:
        rmax = np.array([bubble_rmax(t, Pt, Pa, R0) for (t, Pt, Pa) in forcing])
        ok = np.isfinite(rmax)
        grew = np.mean(rmax[ok] > R0 * 1.0001)
        if rmax[ok].std() < 1e-15:
            print(f'{R0*1e6:9.2f} {grew*100:6.1f}% {"—":>11} {"degenerate":>12} {"—":>12}')
            rows.append((R0, grew, -np.inf, np.nan, np.nan))
            continue
        lr = np.log(rmax[ok] / R0)
        lr = (lr - lr.mean()) / (lr.std() + 1e-30)
        p = (prs[ok] - prs[ok].mean()) / prs[ok].std()
        X = np.column_stack([p, lr])
        beta, ll = fit_logistic(X, y[ok])
        auc = roc_auc_score(y[ok], np.column_stack([np.ones(ok.sum()), X]) @ beta)
        rows.append((R0, grew, ll, beta[2], auc))
        print(f'{R0*1e6:9.2f} {grew*100:6.1f}% {ll:11.2f} {beta[2]:12.3f} {auc:12.4f}')

    rows = [r for r in rows if np.isfinite(r[2])]
    if not rows:
        print('\nAll R0 values degenerate — nothing to fit.'); return
    best = max(rows, key=lambda r: r[2])
    R0_hat, ll_max = best[0], best[2]
    # 95% profile-likelihood CI: 2*(ll_max - ll) <= chi2_{1,0.95} = 3.841
    inside = [r[0] for r in rows if 2*(ll_max - r[2]) <= 3.841]
    print(f'\nR0_hat = {R0_hat*1e6:.2f} um   (max logLik {ll_max:.2f})')
    if inside:
        print(f'95% profile-likelihood CI over the grid: '
              f'[{min(inside)*1e6:.2f}, {max(inside)*1e6:.2f}] um')

    # ── the falsifiable question: out-of-sample lift, grouped by trial ───────
    # A single 5-fold split is far too noisy to adjudicate a ~0.01 AUC effect, and a
    # naive "delta > 0" rule will call noise a win. Use repeated grouped CV, a paired
    # sign test, a permutation null (shuffle the bubble feature), and a sign check on
    # the fitted coefficient -- a feature that "helps" with a physically inverted sign
    # is a confound, not the mechanism.
    print('\n' + '='*72)
    print('Does the bubble feature beat Buhlmann alone OUT OF SAMPLE?')
    print('Repeated GroupKFold by data_set; permutation null; coefficient-sign check')
    print('='*72)

    def cv_delta(rmax_vec, R0, n_rep=10, seed=0):
        ok = np.isfinite(rmax_vec)
        lr = np.log(rmax_vec[ok] / R0)
        Xp, Xb = prs[ok][:, None], np.column_stack([prs[ok], lr])
        yk, gk = y[ok], groups[ok]
        rng = np.random.RandomState(seed)
        uniq = np.unique(gk)
        deltas, betas = [], []
        for _ in range(n_rep):
            perm = rng.permutation(uniq)
            gmap = {g: i for i, g in enumerate(perm)}
            gi = np.array([gmap[g] for g in gk])
            for tr, te in GroupKFold(n_splits=5).split(Xp, yk, gi):
                if len(np.unique(yk[te])) < 2:
                    continue
                aucs = []
                for X in (Xp, Xb):
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-30
                    b, _ = fit_logistic((X[tr]-mu)/sd, yk[tr])
                    s = np.column_stack([np.ones(len(te)), (X[te]-mu)/sd]) @ b
                    aucs.append(roc_auc_score(yk[te], s))
                    if X is Xb:
                        betas.append(b[2])
                deltas.append(aucs[1] - aucs[0])
        return np.array(deltas), np.array(betas)

    for label, R0 in (('literature (Yount VPM)', 0.7e-6), ('profile-likelihood MLE', R0_hat)):
        rmax = np.array([bubble_rmax(t, Pt, Pa, R0) for (t, Pt, Pa) in forcing])
        ok = np.isfinite(rmax)
        if rmax[ok].std() < 1e-15:
            print(f'\n  R0 = {R0*1e6:.2f} um ({label}): degenerate, skipped'); continue
        d, betas = cv_delta(rmax, R0)
        auc_alone = roc_auc_score(y[ok], rmax[ok])
        rho = spearmanr(rmax[ok], prs[ok]).statistic

        print(f'\n  R0 = {R0*1e6:.2f} um  ({label})')
        print(f'    paired delta AUC    : {d.mean():+.4f} +/- {d.std():.4f} over {len(d)} folds')
        print(f'    folds improved      : {100*(d>0).mean():.0f}%')
        print(f'    AUC(R_max alone)    : {auc_alone:.4f}   '
              f'{"<-- INVERTED (bigger bubble => fewer DCS)" if auc_alone < 0.5 else ""}')
        print(f'    mean fitted beta    : {betas.mean():+.3f}   '
              f'{"<-- WRONG SIGN for the mechanism" if betas.mean() < 0 else ""}')
        print(f'    spearman(R_max,prs) : {rho:+.3f}')

        # permutation null: does a shuffled bubble column do as well?
        rng = np.random.RandomState(1)
        null = np.array([cv_delta(rng.permutation(rmax), R0, n_rep=1, seed=k)[0].mean()
                         for k in range(20)])
        print(f'    permutation null    : {null.mean():+.4f} +/- {null.std():.4f}; '
              f'real beats {100*(d.mean()>null).mean():.0f}% of shuffles')

        real_lift = (d.mean() > 0.01 and (d > 0).mean() >= 0.75
                     and betas.mean() > 0 and (d.mean() > null).mean() >= 0.95)
        print(f'    VERDICT: {"adds out-of-sample signal" if real_lift else "NO reliable lift"}')

    # ── ablation: is the physics pipeline better than three raw numbers? ─────
    # A "lift" over a baseline that is itself anti-predictive is not signal. Always
    # report the naive baseline alongside.
    print('\n' + '='*72)
    print('ABLATION — grouped repeated CV AUC (a feature only earns its place here)')
    print('='*72)
    rmax = np.array([bubble_rmax(t, Pt, Pa, R0_hat) for (t, Pt, Pa) in forcing])
    ok = np.isfinite(rmax) & (rmax > 0)
    lr = np.log(rmax[ok] / R0_hat)
    raw = df[['depth_fsw', 'bottom_time_min', 'ascent_time_min']].values[ok]
    yk, gk = y[ok], groups[ok]

    def cv_auc(X, n_rep=10, seed=0):
        rng = np.random.RandomState(seed)
        uniq = np.unique(gk)
        out = []
        for _ in range(n_rep):
            gmap = {g: i for i, g in enumerate(rng.permutation(uniq))}
            gi = np.array([gmap[g] for g in gk])
            for tr, te in GroupKFold(n_splits=5).split(X, yk, gi):
                if len(np.unique(yk[te])) < 2:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-30
                bb, _ = fit_logistic((X[tr]-mu)/sd, yk[tr])
                s = np.column_stack([np.ones(len(te)), (X[te]-mu)/sd]) @ bb
                out.append(roc_auc_score(yk[te], s))
        return np.array(out)

    models = {
        'raw (depth, bottom_time, ascent_time)': raw,
        'prs alone (Buhlmann M-value ratio)':    prs[ok][:, None],
        f'prs + log R_max  (R0={R0_hat*1e6:.1f}um)':  np.column_stack([prs[ok], lr]),
        'raw + prs + log R_max (everything)':    np.column_stack([raw, prs[ok], lr]),
    }
    for name, X in models.items():
        A = cv_auc(X)
        flag = '  <-- WORSE THAN CHANCE' if A.mean() < 0.5 else ''
        print(f'  {name:40s} AUC {A.mean():.4f} +/- {A.std():.4f}{flag}')
    print('\n  If `prs alone` is at or below 0.5, any "lift" from adding a bubble term is')
    print('  repairing a broken baseline, not contributing physics. Compare to `raw`.')

    print('\n' + '-'*72)
    print('Caveats. (1) Absolute DCS rate (~16%) reflects Navy trials designed to provoke')
    print('DCS plus partial negative extraction (2,700 of 8,578 dives); not a population')
    print('rate, so only relative discrimination is meaningful. (2) Air assumed for every')
    print('dive; the report also covers nitrogen-oxygen. (3) No per-diver covariates exist')
    print(f'in this file. (4) Profiles are reconstructed from THREE SCALARS ({args.ascent}')
    print('ascent). Validated against the 428 real depth-time curves: staged is closer')
    print('(median RMSE 36 vs 49 fsw, better on 83% of dives) but still beats a')
    print('predict-the-mean baseline on only 44% -- reconstruction error is large, and')
    print('conclusions about prs FLIP between the two reconstructions. (5) Trials differ')
    print('in protocol aggressiveness; grouping by data_set controls leakage, not')
    print('confounding -- note prs is anti-predictive across trials under staged ascent.')


if __name__ == '__main__':
    main()

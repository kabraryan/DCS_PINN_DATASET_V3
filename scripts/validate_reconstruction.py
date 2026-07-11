"""Which reconstruction matches the REAL depth-time curves: linear or staged?

dcs_real_cases has both the 3 scalars AND the true depth-time series for 428 dives.
Reconstruct from the scalars, compare to truth. This is a ground-truth check that
does not depend on any outcome model.
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/aryankabra_test/Desktop/DCS_PINN_DATASET_V3/scripts')
from fit_r0_to_real_dives import depth_profile, zhl16c
from staged_ascent import build_profile as staged_profile

PATH = '/Users/aryankabra_test/Desktop/FINAL DIVE/datasets/real/dcs_real_cases.jsonl'
table = zhl16c()

recs = [json.loads(l) for l in open(PATH) if l.strip()]
print(f'{len(recs)} real DCS cases with depth-time curves')

def clean(r):
    s = r.get('depth_time_series') or []
    if len(s) < 8: return None
    if (r.get('quality_flags') or ''): return None
    t = np.array([p['t_min'] for p in s], float)
    d = np.array([p['depth_fsw'] for p in s], float)
    if np.any(np.diff(t) <= 0): return None
    if not np.isfinite(t).all() or not np.isfinite(d).all(): return None
    for k in ('max_depth_fsw','bottom_time_min','ascent_time_min'):
        v = r.get(k)
        if v is None or not np.isfinite(v) or v <= 0: return None
    if r['max_depth_fsw'] > 300 or r['bottom_time_min'] > 300 or r['ascent_time_min'] > 300:
        return None
    return t, d, r

usable = [c for c in (clean(r) for r in recs) if c]
print(f'{len(usable)} usable (flag-clean, monotone time, bounce window, >=8 points)\n')

def rmse_against(t_true, d_true, t_rec, d_rec):
    # compare only over the true curve's span, resampling reconstruction onto it
    hi = min(t_true[-1], t_rec[-1])
    m = t_true <= hi
    if m.sum() < 5: return np.nan
    d_i = np.interp(t_true[m], t_rec, d_rec)
    return float(np.sqrt(np.mean((d_i - d_true[m])**2)))

rl, rs, base = [], [], []
for t, d, r in usable:
    D, bt, at = r['max_depth_fsw'], r['bottom_time_min'], r['ascent_time_min']
    tl, dl = depth_profile(D, bt, at)
    ts, ds = staged_profile(D, bt, at, table)
    rl.append(rmse_against(t, d, tl, dl))
    rs.append(rmse_against(t, d, ts, ds))
    base.append(float(np.sqrt(np.mean((np.mean(d) - d)**2))))   # predict-the-mean

rl, rs, base = np.array(rl), np.array(rs), np.array(base)
ok = np.isfinite(rl) & np.isfinite(rs)
rl, rs, base = rl[ok], rs[ok], base[ok]
print(f'n compared: {ok.sum()}')
print(f'  RMSE depth (fsw), predict-the-mean baseline : {np.median(base):8.2f}  (median)')
print(f'  RMSE depth (fsw), LINEAR reconstruction     : {np.median(rl):8.2f}')
print(f'  RMSE depth (fsw), STAGED reconstruction     : {np.median(rs):8.2f}')
print()
print(f'  staged better on {100*np.mean(rs < rl):.1f}% of dives')
print(f'  mean RMSE  linear {rl.mean():.2f}   staged {rs.mean():.2f}')
from scipy.stats import wilcoxon
print(f'  wilcoxon(linear vs staged) p = {wilcoxon(rl, rs).pvalue:.2e}')
print()
better = 'STAGED' if np.median(rs) < np.median(rl) else 'LINEAR'
print(f'  => {better} reconstruction is closer to the real curves.')
print()
print('Both vs baseline: a reconstruction only helps if RMSE < predict-the-mean.')
print(f'  linear beats baseline on {100*np.mean(rl<base):.1f}% of dives')
print(f'  staged beats baseline on {100*np.mean(rs<base):.1f}% of dives')

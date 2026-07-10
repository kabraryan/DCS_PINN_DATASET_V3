"""Reproduce the measurements in Correction 12 of the V3 design spec.

Implements the four candidate nucleation fixes from Correction 11 and reports, for
each, how often a bubble grows beyond its own initial radius across profiles drawn
from V2's seeded sampler.

Result: options 2 and 3 are as degenerate as the unfixed spec. Only the VPM skin
(option 1) produces a bubble that ever grows, and the growth ceiling (option 4) is a
precondition rather than an alternative.

Usage:
    python scripts/verify_nucleation_options.py [--n-profiles 250]

Requires the V2 generator for the Bühlmann forcing. Path is resolved relative to
this repo's sibling projects on the Desktop; override with --v2.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

DEFAULT_V2 = os.path.expanduser('~/Desktop/DCS_DATASET/generate_dcs_dataset_v2.py')

DT_MIN, N_STEPS = 10 / 60, 180
BAR_TO_PA = 1e5
D        = 2.0e-9      # N2 diffusivity in blood (m^2/s)
ALPHA_N2 = 6.84e-6     # molar solubility (mol/(m^3.Pa))
SIGMA    = 0.050       # blood-gas surface tension (N/m)
M_N2     = 0.028
R_GAS    = 8.314
T_BODY   = 310.15
R_CRIT   = 12.0e-6
R_CEILING = 100.0e-6   # option 4
R_MIN    = 1e-9        # numerical floor: keeps 2*sigma/R finite

N2_FRACTIONS = {'air': 0.79, 'nitrox32': 0.68, 'nitrox36': 0.64}
TIME_S = np.arange(N_STEPS) * (DT_MIN * 60.0)


def load_v2(path):
    spec = importlib.util.spec_from_file_location('v2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rhs(t, y, P_tissue, P_amb, skin_floor, ceiling):
    """EP RHS. skin_floor > 0 models a VPM stabilising skin that arrests dissolution."""
    R = max(y[0], skin_floor, R_MIN)
    if ceiling and R >= ceiling:
        return [0.0]

    P_t_pa = np.interp(t, TIME_S, P_tissue) * BAR_TO_PA
    P_a_pa = np.interp(t, TIME_S, P_amb) * BAR_TO_PA
    P_gas_pa = P_a_pa + 2.0 * SIGMA / R

    delta_C = ALPHA_N2 * M_N2 * (P_t_pa - P_gas_pa)     # kg/m^3 (mass basis)
    if R <= max(skin_floor, R_MIN) and delta_C < 0.0:
        return [0.0]                                     # skin holds the nucleus open

    rho_gas = P_gas_pa * M_N2 / (R_GAS * T_BODY)
    correction = 1.0 + R / np.sqrt(np.pi * D * max(t, 1e-10))
    return [D * delta_C / (rho_gas * R) * correction]


def integrate(P_tissue, P_amb, R0, t_start=0.0, skin_floor=0.0, ceiling=None):
    """Return max bubble radius (m), or nan if the solve failed.

    max_step is bounded to the forcing grid: the skin clamp makes dR/dt = 0 at the
    floor, and an unbounded adaptive RK45 strides over the growth window and reports
    R_max = R0 on dives that actually grow. See the ep_bubble.py fix (2026-07-10).
    """
    mask = TIME_S >= t_start
    t_eval = TIME_S[mask]
    if len(t_eval) < 2:
        return R0
    max_step = float(np.min(np.diff(t_eval)))
    kw = {}
    if ceiling:
        def _hit(t, y, *a):
            return y[0] - ceiling
        _hit.terminal = True
        _hit.direction = 1
        kw['events'] = _hit
    sol = solve_ivp(_rhs, (t_eval[0], t_eval[-1]), [R0], method='RK45', t_eval=t_eval,
                    args=(P_tissue, P_amb, skin_floor, ceiling), rtol=1e-6, atol=1e-12,
                    max_step=max_step, **kw)
    if not sol.success:
        return float('nan')
    R = np.clip(sol.y[0], max(skin_floor, R_MIN), ceiling or np.inf)
    return float(R.max())


def build_profiles(v2, n):
    """Bühlmann forcing for n profiles off V2's seeded sampler."""
    table = v2.get_zhl16c_table()
    rng = np.random.RandomState(42)
    trim = [0]
    out = []
    for _ in range(n):
        p = v2.sample_params(rng)
        P_surface = v2.compute_p_surface(p['altitude_m'])
        nf = N2_FRACTIONS[p['gas_name']]
        t_half = v2.compute_effective_half_times(
            table, p['age'], p['body_fat_pct'], p['water_temp_c'],
            p['fitness_name'], p['hydration_name'])
        tissues = v2.preload_tissues(
            p['is_repetitive'], p['surface_interval_min'], p['prior_max_depth_m'],
            t_half, P_surface, nf)
        depth = v2.build_depth_profile(p, trim)

        P_t = tissues.astype(np.float64)
        k = np.log(2) / t_half
        P_tissue = np.zeros(N_STEPS)
        P_amb = np.zeros(N_STEPS)
        for i, d in enumerate(depth):
            pa = float(d) * 0.1 + P_surface
            palv = pa * nf
            P_t = palv + (P_t - palv) * np.exp(-k * DT_MIN)
            P_tissue[i] = P_t.max()
            P_amb[i] = pa

        prs = v2.simulate_dive(depth, tissues, table, t_half, P_surface, nf)['physics_risk_score']

        i_max = int(np.argmax(depth))
        asc = np.where(depth[i_max:] < depth[i_max] - 0.01)[0]
        t_ascent = TIME_S[i_max + asc[0]] if len(asc) else TIME_S[i_max]

        out.append((P_tissue, P_amb, prs, t_ascent))
    return out


def report(label, profiles, R0, at_ascent=False, skin_floor=0.0, ceiling=None):
    prs_all = np.array([p[2] for p in profiles])
    radii = np.array([
        integrate(Pt, Pa, R0,
                  t_start=(t_asc if at_ascent else 0.0),
                  skin_floor=skin_floor, ceiling=ceiling)
        for Pt, Pa, _, t_asc in profiles
    ])
    ok = ~np.isnan(radii)
    um = radii[ok] * 1e6
    prs = prs_all[ok]
    R0_um = R0 * 1e6

    grew = um > R0_um * 1.0001
    rho = spearmanr(um, prs).statistic if um.std() > 1e-12 else float('nan')
    verdict = 'DEGENERATE' if um.std() < 1e-9 else 'alive'
    print(f'{label:44s} grew {grew.mean()*100:5.1f}%  std {um.std():8.3f}  '
          f'max {um.max():9.2f} um  >R_crit {100*np.mean(um > 12):5.1f}%  '
          f'rho(prs) {rho:+.2f}  {verdict}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-profiles', type=int, default=250)
    ap.add_argument('--v2', default=DEFAULT_V2)
    args = ap.parse_args()

    v2 = load_v2(args.v2)
    profiles = build_profiles(v2, args.n_profiles)

    print(f'\nR_max over {args.n_profiles} profiles from V2\'s seeded sampler.')
    print('"grew" = exceeded its OWN R0, so options are compared fairly.\n')

    print('--- No skin: the bubble is free to dissolve ---')
    report('SPEC    R0=0.7um, t=0',            profiles, 0.7e-6)
    report('OPT 2   R0=0.7um, seed at ascent', profiles, 0.7e-6, at_ascent=True)
    report('OPT 3   R0=2um,  t=0',             profiles, 2e-6)
    report('OPT 3   R0=5um,  t=0',             profiles, 5e-6)
    report('OPT 3   R0=10um, t=0',             profiles, 10e-6)
    report('OPT 2+3 R0=5um,  seed at ascent',  profiles, 5e-6, at_ascent=True)

    print('\n--- OPT 1: VPM skin (nucleus cannot dissolve below R0) ---')
    for r in (0.7, 1.0, 2.0, 3.0):
        report(f'OPT 1   R0={r}um, skin, no ceiling', profiles, r*1e-6, skin_floor=r*1e-6)

    print('\n--- OPT 1 + OPT 4: skin plus growth ceiling ---')
    for r in (0.7, 1.0, 2.0, 3.0):
        report(f'OPT 1+4 R0={r}um, skin, ceil 100um', profiles, r*1e-6,
               skin_floor=r*1e-6, ceiling=R_CEILING)

    print('\nConclusion: options 2 and 3 are as degenerate as the unfixed spec.')
    print('Only the skin averts degeneracy; the ceiling is a precondition, not an option.')
    print('See Correction 12 in docs/specs/2026-06-23-pinn-bubble-dynamics-design.md\n')


if __name__ == '__main__':
    main()

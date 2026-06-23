# Patch — one edit to the V3 correction spec

Apply this single change to the "slope inconsistency" item. Leave everything else as written.

## Replace

> **Slope inconsistency:** spec said 18.31, V2 reference says 9.0 → reconciled; running
> code uses 18.31/−18.08, reference doc is outdated.

## With

> **The documented V2 logistic is wrong on its form, not just one constant.** Empirical
> recovery from `dive_profiles_features.csv` (regress `logit(dcs_probability)` on
> `physics_risk_score` for baseline divers, controlling `prs` flexibly) shows three
> documentation errors:
> 1. **Slope/intercept** are ~**18.35 / −18.0**, not the documented 9.0 / −7.5.
> 2. **Undocumented direct terms.** At fixed `prs`, probability still depends on
>    `fitness_level` (≈ −0.32 logit per level) and `hydration_status` (≈ +0.15 logit per
>    level). The reference doc claims these are physics-layer only and never enter the
>    probability layer; the data shows they are in **both** layers (double-counted).
>    Including them drives the residual to ~float precision.
> 3. **Stat drift.** The CSV has 696 profiles at `prs = 1.0`; the README says 905. Other
>    summary means drift slightly too.
>
> **Action:** Do not trust either document for the logistic. **Reconstruct the complete V2
> logistic — every term and constant — from `generate_dcs_dataset_v2.py` source**, then
> validate it against the CSV with the fixed-`prs` regression above. The V3 intercept
> recalibration must operate on this reconstructed model, not the documented one, or it
> bakes the error in.

## Note

This is the only edit. All other items in the correction spec are unaffected because they
do not depend on the logistic coefficients: the PINN removal, Rayleigh-Plesset removal,
isotonic-ordering removal, the fatal-flaw label redesign (regenerate labels from a model
that includes bubble load) and its collinearity caveat, the Henry's-law EP fix, sub-micron
R₀, the corrected `n_critical` formula, the `bubble_terminal_velocity` rename, the
quasi-static Laplace balance, and the honesty-ceiling section.

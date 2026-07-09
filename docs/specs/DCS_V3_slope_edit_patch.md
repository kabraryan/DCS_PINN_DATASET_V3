# Patch — one edit to the V3 correction spec

> **SUPERSEDED (2026-07-09).** This patch is retained only as a record. Its item 3 was
> false: it claimed the CSV holds **696** profiles at `prs = 1.0` and that the README says
> **905**. Neither is true. The CSV holds **490**; the README says **490** and is correct;
> **905** comes from the stale `DCS_Physics_Parameter_Reference.docx`; and **696** appears
> in no file in this project. The corrected text now lives in Correction 9 of
> `2026-06-23-pinn-bubble-dynamics-design.md` — read that, not this. The retraction and the
> verified statistics table there supersede everything below.
>
> Items 1 and 2 below were substantively right (the slope/intercept are 18.31 / −18.08 in
> source, and the fitness/hydration terms really are double-counted), though the regression
> estimates were reported to more precision than they warrant. This patch's own regression
> figures (18.35 / −18.0, −0.32, +0.15) should not be quoted; the exact constants are in
> `generate_dcs_dataset_v2.py:556-589`.

Apply this single change to the "slope inconsistency" item. Leave everything else as written.

## Replace

> **Slope inconsistency:** spec said 18.31, V2 reference says 9.0 → reconciled; running
> code uses 18.31/−18.08, reference doc is outdated.

## With

> **The documented V2 logistic is wrong on its form, not just one constant.** Empirical
> recovery from `dive_profiles_features.csv` (regress `logit(dcs_probability)` on
> `physics_risk_score` for baseline divers, controlling `prs` flexibly) shows two
> documentation errors:
> 1. **Slope/intercept** are **18.31 / −18.08** in source, not the documented 9.0 / −7.5.
> 2. **Undocumented direct terms.** At fixed `prs`, probability still depends on
>    `fitness_level` (source: +0.45 / 0.0 / −0.50 by level) and `hydration_status`
>    (source: 0.0 / +0.15 / +0.30 by level). The reference doc claims these are
>    physics-layer only and never enter the probability layer; the source shows they are in
>    **both** layers (double-counted, as V2's "Fix 7" and "Fix 8"). Including them drives
>    the residual to ~float precision.
> 3. ~~**Stat drift.** The CSV has 696 profiles at `prs = 1.0`; the README says 905.~~
>    **Retracted — both numbers were fabricated.** The CSV has 490 and the README says 490.
>    The stale figure is 905, and it is in the `.docx`, not the README.
>
> **Action:** Do not trust the `.docx` for the logistic. **Reconstruct the complete V2
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

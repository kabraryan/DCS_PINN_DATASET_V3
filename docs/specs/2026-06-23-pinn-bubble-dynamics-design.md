# DCS Dataset V3 — Corrected Design Spec
**Date:** 2026-06-23 (corrected from original)
**Status:** Approved — replaces erroneous PINN draft
**Output folder:** `~/Desktop/DCS_PINN_DATASET_V3/`

> **Note on folder name:** The folder retains the name given at project inception. The PINN
> approach proposed in the original draft has been removed (see §Corrections below).

---

## Consolidated record of changes and fixes

*Added 2026-07-10. Every row was verified by running code, not by reading. Each links to the
correction that carries the evidence and the command that regenerates it.*

### Verdict on the project as designed

**V3 must not be generated.** The bubble model is degenerate (Correction 11), and even after
repair, bubble features do not improve on Bühlmann against real dive outcomes with real controls
(Correction 13). The founding premise — "there is no real dive-outcome data and none will be
available" — was false; 2,700 real Navy dives with 1,932 non-DCS controls sit in
`FINAL DIVE/datasets/real/`.

### Physics and numerical defects

| # | Defect | Status | Evidence |
|---|---|---|---|
| 10 | ZHL-16C compartment 16 `b = 0.8693` — compartment 7's value, copy-pasted. Correct: `0.9653` | **Fixed** in 4 generators, README, 2 docs, 1 test | Measured impact: 1,421/3,000 rows shift, max Δprs 0.0105, ~0.02% label flips |
| 12 | `Δc` used molar solubility against a mass density — `dR/dt` inflated by exactly `1/M_N2 = 35.714×` | **Fixed** | Verified numerically |
| 12 | `Radau` fails on **70%** of realistic profiles (18/60). Spec claimed `RK45` would fail | **Fixed** — `RK45` + terminal event, 60/60 at 1.6 ms | `verify_nucleation_options.py` |
| 12 | Non-smooth `if R <= 0` guard destroyed the implicit Jacobian | **Fixed** — terminal event | |
| 11 | **Bubble never grows.** `R_max ≡ R₀` on every profile (std 1.1×10⁻¹⁶); all six columns constant | **Unresolved — blocking** | 400 real profiles |
| 12 | Enlarging `R₀` or seeding at ascent does **not** help (0.0% growth at 2, 5, 10 µm) | **Diagnosed** — only the VPM skin works | Gas leaves an undersaturated bubble regardless of radius |
| 12 | Unbounded EP growth reaches 267–284 µm | Ceiling required | |
| — | Silent `R_traj = full(R0)` fallback wrote fabricated min-risk physics into shipped rows and poisoned the scaler for all 50,000 | **Fixed** — `ep_solve_failed` column, hard abort > 1% | Failure correlates with the label |
| — | `np.trapz` removed in NumPy 2.0 | **Fixed** — `np.trapezoid` | |
| — | `bubble_scaler.pkl` — `joblib.load` is arbitrary code execution for 12 floats | **Fixed** — JSON; manifest and deps purged | |

### Provenance and documentation defects

| # | Defect | Status |
|---|---|---|
| 9 | The `.docx` reference states the V2 logistic as `−7.5 + 9.0·prs`; the running code is `−18.08 + 18.31·prs` | Documented; source is authoritative |
| 9 | `fitness_level` and `hydration_status` are **double-counted** — in both the physics layer and the logistic | Documented as a known modelling choice |
| 9 | Correction 9 itself claimed "the README reports 905". The README reports **490**, which is exactly what the CSV contains. The README was right | **Retracted** |
| 9 | `DCS_V3_slope_edit_patch.md` claimed the CSV holds **696**. That number appears in **no file** in this project | **Retracted**; patch marked SUPERSEDED |
| — | The correction documents written to fix documentation drift introduced two fresh fabricated statistics | **Root cause fixed** — see *Provenance discipline* |
| 2 | "There is no real dive-outcome data and none will be available" | **False.** Annotated in `DCS_V3_correction_prompt.md` |

### Validation defects

| Defect | Status |
|---|---|
| `bubble_R_max > R₀` (strict) — `R₀` is the initial condition in `t_eval`, so this **always fails** | Fixed to `>=`, gated on `ep_solve_failed` |
| Quartile-monotonicity checks are **tautologies** — labels are drawn with a positive coefficient on those very features | **Removed** |
| Intercept calibration averages three separately-solved intercepts and misses the 55% anchor by 0.84 pp against a documented ±0.5 pp | Documented |
| Tests were change-detectors: `test_table_compartment16` **asserted the bug** | Replaced with property tests (`b` strictly increasing, `a` strictly decreasing) |
| No test asserted `std(bubble_R_max) > 0` — 15 lines that would have caught Correction 11 before 1,500 lines were built on it | Added as a guard; mandated in the benchmark plan |

### What was learned against real data (Correction 13)

| Finding | Number |
|---|---|
| `R₀` does not identify — profile likelihood is **bimodal with modes of opposite sign** | β = −0.26 at 1.0 µm; β = +0.42 at 2.4 µm |
| At the literature `R₀` = 0.7 µm the bubble feature is **inverted** — a proxy for "deep short dive" | `AUC(R_max)` = 0.489; DCS 9.2% among growers vs 16.0% |
| At the MLE the feature is 0.49-collinear with `prs` and adds **nothing** out of sample | ΔAUC −0.0038, p = 0.47 |
| Under the **more faithful** staged reconstruction, `prs` is **anti-predictive** out of sample | AUC 0.3843 (in-sample 0.5392) |
| Three raw numbers beat the entire physics pipeline; adding physics changes nothing | raw 0.6429; raw + prs + `R_max` 0.6429 |
| The staged reconstruction is closer to truth, yet still beats predict-the-mean on only 44.4% | median RMSE 36.08 vs 48.71 fsw, p = 1.8×10⁻¹¹ |
| Adding physics to the trained baseline makes every model **worse** | RF 0.7114 → 0.6870 |

**The pattern underneath all of it.** Every defect above — the typo, the fabricated statistics,
the failed solver, the bubble that could never grow, and two results that looked convincing and
were empty — has one cause: *something plausible was written down and never executed*. Twice, a
result cleared every check that existed and died to a check that did not yet exist (the
coefficient-sign gate, and the raw baseline). The durable fix is not more review. It is that
**a number which cannot be regenerated on demand does not get written down.**

### Reproduce

```
python scripts/verify_nucleation_options.py       # Corrections 11, 12
python scripts/fit_r0_to_real_dives.py --ascent staged --marginal exclude   # Correction 13
python scripts/validate_reconstruction.py         # staged vs linear vs ground truth
python scripts/train_baseline.py --features raw+physics                     # physics makes it worse
```

---

## Corrections Applied to Original Draft

This document supersedes the original V3 spec. The original contained one fatal conceptual
flaw, one inappropriate tool choice, several physics errors, and two methodology problems.
Each is itemised here so the reasoning is on record.

### Correction 1 (FATAL) — Bubble features carried no new information about the label

The original plan derived bubble features from Bühlmann tissue saturation, then fitted their
logistic coefficients by regressing against V2 labels. This is circular: V2 labels were
generated by a logistic model that contains zero bubble physics, so the regression can only
recover the portion of bubble features that correlates with `physics_risk_score`, which
already drives the label. The result is a near-identical dataset to V2 with more columns.

**Fix:** Bubble features now enter the **data-generating process**. Labels are drawn from a
logistic model that includes bubble-load terms with pre-specified literature-calibrated
coefficients. Bubble features carry signal by construction because they partially define the
probability that generates the label.

### Correction 2 — PINN removed; replaced with direct numerical integration

A PINN earns its place when: (a) the governing equations are expensive or impossible to
solve directly, (b) there is an inverse problem (unknown parameters to infer), or (c) sparse
noisy real data needs physics regularisation. None of these conditions hold here:

- The bubble model is a single coupled ODE in one state variable (R) over time.
- Every parameter is specified from the literature; there is nothing to infer.
- There is no real data.

`scipy.integrate.solve_ivp` with method `Radau` (a purpose-built stiff ODE solver) produces
an exact solution per profile in ~2 ms. For 50,000 profiles this is under 5 minutes — no
training, no approximation error, no convergence risk.

The original spec also misused Krishnapriyan et al. (2021), which documents how PINNs *fail*
on stiff problems, as supposed support for using a PINN on a stiff problem. A 27-input
parametric surrogate queried at arbitrary `t` is substantially harder than the single-instance
problems in Raissi et al. (2019) and is prone to the generalisation failures Krishnapriyan
describes.

### Correction 3 — Rayleigh-Plesset removed

The Rayleigh-Plesset equation models inertial bubble dynamics (cavitation, collapse) on
µs–ms timescales. DCS bubbles grow over minutes; the inertial terms `R·R″ + 3/2·(R')²` are
negligible by many orders of magnitude at these timescales and this viscosity. The correct
mechanical model is **quasi-static Laplace pressure balance**: at each instant the bubble is
in mechanical equilibrium with the surrounding blood pressure:

```
P_gas(t) = P_amb(t) + 2σ/R(t)
```

This is the standard assumption in the DCS bubble literature (Van Liew 1991; Gernhardt 1991;
Srinivasan et al. 2003).

### Correction 4 — Epstein-Plesset forcing term fixed (Henry's law)

The original computed:
```
C_∞(t) = max(tissue_sat) · ρ_blood / M_N2
```
This treats a partial pressure (bar) as a molar concentration — dimensionally wrong.
The correct conversion uses Henry's law:
```
C_∞(t) = P_tissue(t) · α_N2
C_s(t)  = P_gas(t)   · α_N2  =  (P_amb(t) + 2σ/R) · α_N2
```
where `α_N2` is N₂ solubility in blood (~0.0693 mL N₂ (STPD) / mL blood / atm, Weathersby
et al. 1984). The diffusion gradient `(C_∞ - C_s)` then equals `α_N2 · (P_tissue - P_amb -
2σ/R)` — supersaturation minus the Laplace correction.

### Correction 5 — Initial nucleus radius corrected

The original used `R₀ = 10 µm`, described as the "Yount critical nucleus radius." This is
wrong on both counts: Yount's VPM nuclei are sub-micron. The 10 µm value is a grown bubble
radius, not a nucleation radius. Correct value: **R₀ = 0.7 µm** (Yount 1991, VPM
stabilised gas nucleus). At this radius the Laplace pressure term (2σ/R ≈ 1.4 bar) dominates
and the bubble grows only if tissue supersaturation exceeds this threshold.

### Correction 6 — `bubble_n_critical` formula fixed

The original formula `N₀ · exp(-R_crit/R_max)` is smooth, strictly positive for all
`R_max > 0`, and never reaches zero. The accompanying validation check — "`n_critical = 0`
when `R_max < R_crit`" — can therefore never pass. The formula and check contradicted each
other.

**Replacement:** `n_critical = N₀ · max(0, 1 - R_crit / R_max)`. This is exactly 0 when
`R_max ≤ R_crit`, rises linearly with excess radius, and approaches `N₀` as `R_max ≫ R_crit`.
The validation check is now self-consistent.

### Correction 7 — `bubble_terminal_velocity` renamed and scoped

The original named this feature "Stokes drift velocity." Stokes drift is a wave-transport
phenomenon unrelated to this formula. The formula computes Stokes **terminal buoyant
velocity**: `v = 2R²(ρ_blood - ρ_gas)g / 9μ`. For a bubble of R = 50 µm this gives ~0.2
mm/s; blood flow velocity is ~10–100 mm/s. The feature is retained as a computed diagnostic
column but **excluded from the logistic model**: at these scales buoyant rise is dominated
by blood-flow advection and does not independently predict where bubbles lodge.

### Correction 8 — Isotonic ordering removed

Forcing fitted coefficients to match a pre-decided literature ordering overrides the data
with a prior. Since the corrected design pre-specifies bubble coefficients as part of the
data-generating process, there is no fitting step to constrain; this machinery is moot.

### Correction 9 — V2 logistic reconstructed from source; two documentation errors noted

> **Retraction (2026-07-09).** An earlier revision of this section made two false claims.
> It asserted that "the README reports 905 profiles at `physics_risk_score = 1.0`" and
> counted that as a third documentation error. The README
> (`code/synthetic_generator/README.md:111`) reports **490**, which is exactly what the
> shipped CSV contains — the README was correct all along. The figure 905 comes from
> `DCS_Physics_Parameter_Reference.docx`, a separate and genuinely stale document. The
> companion patch `DCS_V3_slope_edit_patch.md` compounded the error with a claimed CSV
> count of 696, a number that appears in no file in this project. Both claims are
> withdrawn; the corrected accounting is below. See §Provenance discipline.

The V2 reference document (`DCS_Physics_Parameter_Reference.docx`) states slope = 9.0 /
intercept = −7.5. Recovery from `generate_dcs_dataset_v2.py` source and regression on
`logit(dcs_probability)` against `physics_risk_score` reveals two real discrepancies, both
confined to that `.docx`:

1. **Slope / intercept** in the running code (`generate_dcs_dataset_v2.py:556`) are
   **18.31 / −18.08**, not 9.0 / −7.5.
2. **Undocumented direct terms.** At fixed `physics_risk_score`, `dcs_probability` still
   varies with `fitness_level` (≈ −0.50 to +0.45 logit) and `hydration_status` (≈ 0.0 to
   +0.30 logit). The reference doc claims these operate exclusively through the physics
   (half-time modification). The source shows they appear in **both** the physics layer
   (half-time multipliers) and the logistic model (direct log-odds terms, added as V2's
   "Fix 7" and "Fix 8") — they are double-counted.

**Verified summary statistics** (recomputed from
`FINAL DIVE/datasets/synthetic_v2/dive_profiles_features.csv`, 2026-07-09):

| Statistic | Value | `README.md` | `.docx` |
|-----------|-------|-------------|---------|
| Rows | 50,000 | 50,000 ✓ | 50,000 ✓ |
| `physics_risk_score` = 1.0 | **490** | 490 ✓ | 905 ✗ |
| `physics_risk_score` ≥ 0.85 | **4,918** | 4,918 ✓ | — |
| DCS positives | **2,948 (5.896%)** | 2,950 (5.90%) ≈ | — |
| Mean `physics_risk_score` | **0.6758** | 0.676 ✓ | 0.675 ≈ |

`README.md` is accurate. `DCS_Physics_Parameter_Reference.docx` is stale on both the
logistic form and the M-value-breached count, and should be treated as superseded.

**Action:** The V3 spec treats the source code (`generate_dcs_dataset_v2.py`) as
authoritative for all V2 logistic terms. The complete V2 logistic (as it runs) is given in
§Extended Logistic Model below. V3's intercept recalibration operates on this reconstructed
model.

### Correction 10 — ZHL-16C compartment 16 `b` coefficient was wrong

V2's `get_zhl16c_table()` gave compartment 16 (635 min half-time) a `b` coefficient of
**0.8693**. That is compartment 7's value (54.3 min), copy-pasted. The correct Bühlmann
value is **0.9653**; the `b` column rises monotonically toward 1.0 and this entry made it
drop from 0.9602 back to 0.8693.

Fixed in all three copies of `generate_dcs_dataset_v2.py` and in this plan's Task 5. V3's
`get_zhl16c_table()` now asserts that `b` is strictly increasing and `a` strictly decreasing,
so the class of error cannot recur silently.

**The shipped V2 dataset was generated with the wrong value** and therefore no longer
reproduces from the corrected source. The impact was measured, not assumed — re-running the
seeded pipeline over the first 3,000 profiles with both tables:

| Quantity | Old `b` = 0.8693 | Fixed `b` = 0.9653 |
|----------|------------------|--------------------|
| `physics_risk_score` unchanged | — | 1,579 / 3,000 rows |
| `physics_risk_score` changed | — | 1,421 / 3,000 rows |
| max \|Δ `physics_risk_score`\| | — | 0.0105 |
| rows newly crossing the 1.0 clip | — | 0 |
| mean `dcs_probability` | 0.05784 | 0.05801 (+0.30%) |
| max \|Δ `dcs_probability`\| | — | 0.0041 |
| expected `label` flips | — | ≈0.02% of rows |

Direction: raising `b` lowers the M-value (`a + P_amb/b`), which raises the M-value ratio.
So the fix makes `physics_risk_score` slightly *higher*, never lower. The effect is small
because compartment 16 (635 min half-time) barely loads during a 30-minute dive and is never
the limiting tissue — the fast compartments set `physics_risk_score` in essentially every
profile. **The shipped V2 CSV is therefore not materially wrong**, but it is no longer
byte-reproducible from source. Regenerating V2 is a separate decision and has not been done.

> Note, unfixed: the `a` column carries the **ZHL-16A** coefficients (0.6200, 0.5043,
> 0.4410, …), not ZHL-16C's (0.6667, 0.5600, 0.4947, …), despite the function name and
> docstring. The `b` column is shared across ZHL-16A/B/C, so the fix above is correct in
> any variant. Correcting `a` would materially change every row of the dataset and is out
> of scope for this correction; it is recorded here so it is not rediscovered as new.

### Correction 11 (FATAL, unresolved) — the specified bubble model cannot grow a bubble

Corrections 3 and 5 are each defensible in isolation and **jointly degenerate**. Quasi-static
Laplace equilibrium with no VPM stabilising skin, a sub-micron `R₀`, and a nucleus seeded at
`t = 0` guarantee that `R(t)` is monotonically decreasing on every profile.

The mechanism, measured on 400 profiles drawn from V2's own seeded sampler:

1. The bubble is seeded at dive start. During **descent** `P_amb` rises far faster than
   `P_tissue` loads, so supersaturation is strongly **negative** (−3.67 bar at t = 2 min on
   the worst profile). The nucleus immediately shrinks.
2. The Laplace barrier is `2σ/R`. As `R` shrinks the barrier **grows**: 1.43 bar at 0.7 µm,
   2.00 bar at 0.5 µm, 10.0 bar at 0.1 µm. Dissolution is a one-way trip.
3. Supersaturation only turns positive on **ascent** (t ≈ 14 min). By then `R ≈ 0` and the
   barrier at that radius is unreachable. The bubble never returns.

Consequently `max R(t) = R(0) = R₀` for **every** profile:

| Measured over real profiles | Result |
|---|---|
| Peak supersaturation `max(P_tissue − P_amb)` | mean 0.81, median 0.77, max 2.26 bar |
| Laplace barrier at `R₀ = 0.7 µm` (`2σ/R₀`) | **1.4286 bar** |
| Profiles ever clearing the barrier *at R₀* | 65 / 400 (16.2%) — but all *after* dissolution |
| `bubble_R_max` across 120 profiles | min = max = 0.700000 µm, **std = 1.1×10⁻¹⁶** |
| `bubble_R_max > R₀` (the spec's validation check, strict) | passes on **0 / 120** rows |
| Minimum seed radius that could grow on the *best* profile | `2σ/ΔP_max` = **0.443 µm** |

So all six bubble columns are constants. `StandardScaler` divides by ~zero, the z-scores are
`0/0`, and `dcs_probability` is `NaN` or — if σ is clamped — identical to V2 with six constant
columns appended. **Correction 1's fix does not work as specified: the dataset it produces is
exactly the "near-identical to V2 with more columns" outcome Correction 1 was written to
prevent.** The failure is silent, because the `ep_failures` fallback also writes `R₀`.

This is not a numerical bug; it is the physical model. Yount's nuclei are *stabilised* —
a surfactant skin resists dissolution and is the entire reason sub-micron nuclei persist in
vivo. Correction 3 removed the skin along with Rayleigh-Plesset and kept the VPM `R₀`. The two
corrections are individually right and jointly incoherent.

**Resolving this requires a physics decision and is deliberately left open.** The candidate
directions:

1. **Restore the VPM stabilising skin** (Yount 1979/1991). Add a skin pressure term so the
   nucleus resists dissolution below `R₀`. This is the change most faithful to the cited
   literature and re-earns the `R₀ = 0.7 µm` value.
2. **Seed the bubble at ascent onset** rather than `t = 0`. Physically defensible, since
   nucleation is triggered by decompression.
3. **Use a larger seed radius** (Van Liew's µm-scale seeds). Cheapest, but abandons the VPM
   citation that Correction 5 was written to honour.
4. **Add a growth ceiling regardless.** Once a bubble does clear the barrier, `2σ/R` collapses
   and growth runs away — an unbounded EP bubble reaches millimetre radii, which is not physics.

> **This menu is misleading. See Correction 12** — options 2 and 3 were measured and **do not
> work**; they leave `bubble_R_max` exactly as constant as the current spec. Option 4 is a
> precondition, not an alternative. The real choice is Option 1 + Option 4, or no bubble layer.

Until one is chosen, **V3 must not be generated.** No downstream fix (solver, units, scaler)
changes this outcome.

### Correction 12 — the four-option menu in Correction 11 is wrong; only the skin works

Correction 11 listed four candidate fixes as if they were interchangeable. They are not. All
four were implemented and measured on 250 profiles from V2's seeded sampler
(`scripts/verify_nucleation_options.py`). **Options 2 and 3 leave `bubble_R_max` exactly as
constant as the unfixed spec.** Only the VPM skin produces a bubble that ever grows.

| Configuration | Grew past its own `R₀` | std(`bubble_R_max`) | Verdict |
|---|---|---|---|
| Spec as written — `R₀` = 0.7 µm, `t` = 0, no skin | **0.0%** | 0 | degenerate |
| **Opt 2** — seed at ascent onset, `R₀` = 0.7 µm | **0.0%** | 0 | degenerate |
| **Opt 3** — `R₀` = 2 µm | **0.0%** | 0 | degenerate |
| **Opt 3** — `R₀` = 5 µm | **0.0%** | 0 | degenerate |
| **Opt 3** — `R₀` = 10 µm | **0.0%** | 0 | degenerate |
| **Opt 2 + 3** — `R₀` = 5 µm, seed at ascent | **0.0%** | 0 | degenerate |
| **Opt 1** — VPM skin, `R₀` = 0.7 µm | 3.2% | 43.6 µm | **alive** |

#### Why a bigger seed cannot help

Correction 11 attributed the degeneracy to the Laplace barrier `2σ/R` rising as `R` falls.
That is real, but it is the *second* effect and not the binding one. The diffusion gradient is

```
C_∞ − C_s  ∝  (P_tissue − P_amb) − 2σ/R
```

On descent `P_tissue − P_amb ≈ −1.8 bar` (measured; −3.67 bar at the extreme). **That deficit
is independent of `R`.** Gas leaves the bubble because the tissue is undersaturated, not because
the bubble is small. A 10 µm nucleus bleeds gas exactly as surely as a 0.7 µm one — it merely
takes longer, and descent plus bottom time is ≥ 10 minutes, which is ample.

So the degeneracy is not a *starting-size* problem and enlarging `R₀` cannot fix it. It is a
*nothing-arrests-the-dissolution* problem. Only a mechanism that halts shrinkage survives to
the ascent, which is the entire reason Yount's nuclei carry a stabilising surfactant skin.
Correction 3 deleted the one component that makes the model non-degenerate.

Option 2 fails for the same reason: at ascent **onset** the diver is still deep and the tissue
is still undersaturated, so the nucleus dissolves during the early climb, before supersaturation
ever turns positive (t ≈ 14 min).

#### The skin makes `R₀` a consequential parameter

With the skin in place, `R₀` controls both how often bubbles form and how much independent
signal the feature carries:

| `R₀` (with skin) | Grew | ρ(`bubble_R_max`, `physics_risk_score`) | Reading |
|---|---|---|---|
| 0.7 µm (the Yount/VPM value) | 3.2% | **+0.29** | rare, and largely independent of `prs` |
| 1.0 µm | 14.8% | +0.56 | |
| 2.0 µm | 42.8% | +0.82 | too common; nearly redundant with `prs` |
| 3.0 µm | 60.0% | +0.85 | |

The literature value `R₀ = 0.7 µm` is also the best-behaved: DCS is rare, so a ~3% bubble rate
is plausible, and the **low** correlation with `physics_risk_score` means the feature carries
information the Bühlmann layer does not already contain — which is the only reason to add a
bubble layer at all. At `R₀ = 2 µm` the feature is 0.82-collinear with `prs` and adds nothing.

This cuts against the intuition that a larger, "safer," more arbitrary seed is the conservative
choice. It is neither conservative nor safe: it is degenerate without the skin, and redundant
with it.

#### The growth ceiling is a precondition, not an option

Without a ceiling, bubbles that clear the barrier run away: measured `max R` of **267–284 µm**
across all skin configurations. `2σ/R` collapses as `R` grows, so growth is self-accelerating.
Option 4 must be applied regardless of which nucleation model is adopted.

#### What this leaves

The real choice is **Option 1 + Option 4** (VPM skin, `R₀` = 0.7 µm, growth ceiling), or **no
bubble layer at all**. There is no cheap middle path; options 2 and 3 were the cheap middle
path and they do not exist.

Adopting Option 1 means accepting free parameters (skin compression modulus, initial skin
tension, crushing pressure) that cannot be measured directly. That objection is sound, and it
sharpens rather than weakens the case made in the Honesty Ceiling: **if unmeasurable parameters
must be introduced, they should be fitted to the 2,700 real dives in
`FINAL DIVE/datasets/real/dcs_all_dives.csv` — which contain 1,932 real non-DCS controls —
not chosen by hand.** Fitting turns `R₀` from an arbitrary constant into an estimated parameter
with a confidence interval, and turns "do bubble features add signal?" into a falsifiable
question. (Note that `dcs_real_cases.csv`, despite having full depth–time curves, is
**positives-only** — all 428 rows have `outcome = 1.0` — and cannot support a risk fit.)

> **That fit was performed. See Correction 13.** `R₀` does not identify — the profile
> likelihood is bimodal with modes of opposite coefficient sign — and the bubble feature adds
> no out-of-sample discrimination over Bühlmann, under either profile reconstruction. Option 1
> is still the only *non-degenerate* nucleation model, but the real data declines to endorse
> the bubble layer it would enable.

#### Caveats on these measurements

- The skin is modelled here as a **hard floor** (`R` cannot fall below `R₀`). True VPM has a
  *compressible* skin with a crushing pressure that permits partial shrinkage. The qualitative
  conclusion (only the skin averts degeneracy) is robust; the exact 3.2% is not.
- All figures assume `σ = 0.050 N/m`. The dead/alive classification is insensitive to `σ`
  (undersaturation kills free bubbles at any `σ`), but the growth rates are not.
- Reproduce with: `python scripts/verify_nucleation_options.py`

### Correction 13 — R₀ fitted to real outcomes: the bubble layer earns nothing

Correction 12 concluded that if unmeasurable skin parameters must be introduced, they should
be **fitted** to the 2,700 real dives in `FINAL DIVE/datasets/real/dcs_all_dives.csv` (1,932
real non-DCS controls) rather than chosen by hand. That was done. The result is negative.

Reproduce:
```
python scripts/fit_r0_to_real_dives.py --ascent staged --marginal exclude
python scripts/validate_reconstruction.py
```

#### Setup

2,230 bounce dives (470 multi-day air-saturation exposures and >300 fsw excursions excluded —
a different physiological regime, not representable by a single-nucleus bounce model; the
exclusion is outcome-correlated, 24.3% vs 13.7% DCS, and is reported rather than hidden).
Profiles reconstructed from `(depth_fsw, bottom_time_min, ascent_time_min)`. Bühlmann ZHL-16C
on air. EP with the VPM skin of Correction 12 at each candidate `R₀`. `R₀` estimated by
profile likelihood on `outcome ~ prs + log R_max`; discrimination measured by **repeated**
GroupKFold over `data_set` (50 folds), a permutation null, and a sign check on the fitted
coefficient.

#### The profile reconstruction was validated against ground truth

`dcs_real_cases.jsonl` carries **both** the three scalars and the true depth–time curve for
428 dives, so a reconstruction can be scored with no outcome model in the loop. On 72
flag-clean bounce dives:

| Reconstruction | median RMSE (fsw) | closer to truth on |
|---|---|---|
| Linear ascent | 48.71 | 16.7% of dives |
| **Staged ascent** (Bühlmann-ceiling stops, rescaled to the recorded ascent time) | **36.08** | **83.3%** (Wilcoxon p = 1.8×10⁻¹¹) |

Staged is the more faithful reconstruction and is used for the headline result. But it beats a
*predict-the-mean* baseline on only **44.4%** of dives (linear: 15.3%). **Three scalars do not
determine a dive profile.** Hold that fact; it is the finding.

The staged schedule rescales *stop* durations — never travel legs — so that total ascent
equals each dive's recorded `ascent_time_min`. This is what stops the reconstruction from
being circular: a dive whose recorded ascent is shorter than the schedule Bühlmann demands
gets compressed stops and still violates M-values. The reconstruction supplies the *shape*;
the data supplies the *duration*.

#### The result, with the baseline it needs

At the profile-likelihood MLE (`R₀` = 4.0 µm), grouped repeated CV:

| Model | AUC |
|---|---|
| raw `(depth, bottom_time, ascent_time)` | **0.6429 ± 0.0640** |
| `prs` alone (Bühlmann M-value ratio) | **0.3843 ± 0.0803** — worse than chance |
| `prs` + `log R_max` (bubble) | 0.5067 ± 0.0706 |
| raw + `prs` + `log R_max` (everything) | **0.6429 ± 0.1006** — identical to raw |

Read without the baseline, the bubble term looks like a triumph: **+0.1224 AUC, 100% of 50
folds improved, correct coefficient sign, beats 100% of permutations.** It is nothing of the
kind. It is **repairing an anti-predictive baseline** and still landing at chance. Three raw
numbers beat the entire physics pipeline, and adding `prs` and the bubble feature on top of
them moves the AUC by exactly zero.

#### The linear reconstruction gave a different — and also negative — answer

| | `R₀` MLE | `prs` AUC (in-sample) | bubble ΔAUC (out-of-sample) |
|---|---|---|---|
| Linear ascent | 2.4–2.5 µm | 0.6004 | −0.0038 ± 0.0714 (p = 0.47) |
| Staged ascent | 4.0 µm | 0.5392 | +0.1224 (an artefact — see above) |

Under **linear**, at the literature `R₀` = 0.7 µm only 87 dives grow a bubble, and those are
deep-short dives (199 fsw / 39 min) whose DCS rate is 9.2% against 16.0% for non-growers.
`AUC(R_max)` = 0.489 — **inverted**. Repeated CV reported a `+0.0106` lift beating 100% of
permutations, but the fitted coefficient was **negative**. A feature that helps with an
inverted sign is a confound — a proxy for "deep short dive" — not the bubble mechanism.
**The coefficient-sign check is the only thing that caught this**; a ΔAUC threshold plus a
permutation null both called it a win.

The profile likelihood is also **bimodal**, and the two modes disagree about physics:
`R₀ ≈ 1.0 µm` gives β = −0.26 (bubbles → *less* DCS), `R₀ ≈ 2.4 µm` gives β = +0.42. A
likelihood surface with two modes of opposite sign is not identifying a physical parameter.

Stable across the marginal-outcome fork (`exclude` / `0.5→1` / `0.5→0`): no reliable lift in
any of six configurations.

#### What this establishes, and what it does not

**Established.** Bubble features do not improve on Bühlmann tissue loading on real dive
outcomes with real controls, under either reconstruction. V3's premise — that a bubble signal
must be *injected* because V2's labels do not contain one — is not rescued by real data: when
real data is asked whether that signal exists, it declines to say yes.

**Not established, and this matters more.** `prs` itself is anti-predictive out of sample
(0.384) while looking respectable in-sample (0.539). Its apparent usefulness reverses across
trials *and* reverses between the two reconstructions. That is reconstruction error and
trial-level confounding dominating, not physics. **The real data as extracted cannot
adjudicate this question.** A conclusion that flips on an assumption we cannot pin down from
three scalars per dive is not a conclusion about bubbles.

#### The unlock

Depth–time curves exist for the **428 positives only**. The 2,700-dive set that carries the
1,932 real controls has three scalars per dive and nothing else. Recovering real curves for
the negatives — Vol I key files, which the extraction pipeline covers only partially (2,700 of
8,578 dives; many pages scanned upside-down) — would let both Bühlmann and EP be driven by
real profiles on both classes, and would make this question answerable. That is bounded work
on a pipeline that already exists, and it is worth more than any further modelling on the
scalars.

Until then, **no bubble-layer design should be justified by appeal to the real data.**

### Provenance discipline

Corrections 9 and 10 both exist because derived numbers — logistic coefficients, summary
statistics, table constants — were hand-transcribed into prose and never re-checked against
the artefact they describe. The retraction at the top of Correction 9 is that same failure
occurring *inside a document written to fix it*.

Every numeric claim in this spec about the contents of a dataset must be recomputed from the
dataset before it is written down, and cited with the file and line or the command that
produced it. Numbers that cannot be regenerated on demand do not belong in prose.

---

## Overview

V3 extends the V2 Bühlmann + logistic pipeline with a deterministic bubble dynamics layer.
Per-profile numerical integration of a diffusion-limited bubble growth model (Epstein-Plesset
with quasi-static Laplace equilibrium) produces five bubble state features. These features
are incorporated directly into the logistic DCS probability model with pre-specified
literature-calibrated coefficients, so the binary label carries bubble-dynamics information
by construction.

The result is a 47-column dataset (41 V2 columns + 6 bubble feature columns) that is
physically richer than V2 and internally self-consistent. It is a **physically-consistent,
literature-calibrated synthetic benchmark** — not a clinically validated DCS predictor.

---

## Pipeline Architecture

```
Dive profile + physiological parameters
              ↓
   ┌──────────────────────────────┐
   │  [1] Bühlmann ZHL-16C        │  (unchanged from V2)
   │  16-compartment N₂ simulator │
   └──────────────────────────────┘
              ↓
   tissue_sat_0..15(t), physics_risk_score, P_amb(t)
              ↓
   ┌──────────────────────────────┐
   │  [2] EP Bubble Integrator    │  (new in V3)
   │  scipy solve_ivp / Radau     │
   │  Epstein-Plesset + Laplace   │
   └──────────────────────────────┘
              ↓
   R(t) trajectory → 6 bubble feature columns
              ↓
   ┌──────────────────────────────┐
   │  [3] Extended logistic model │  (V2 terms + 5 bubble terms)
   │  dcs_probability             │
   └──────────────────────────────┘
              ↓
   label = Bernoulli(dcs_probability)
```

Layer [2] is deterministic and exact: no training, no approximation. Runtime for all 50,000
profiles: ~5 minutes on CPU.

---

## Governing Equation — Epstein-Plesset with Quasi-Static Laplace

### Mechanical equilibrium (replaces Rayleigh-Plesset)

At every instant:
```
P_gas(t) = P_amb(t) + 2σ / R(t)
```

`P_amb(t) = depth(t) × 0.1 + P_surface` (bar), taken directly from the depth profile.
This is valid because DCS bubble growth is diffusion-limited and overdamped on minute
timescales — inertial terms are negligible.

### Diffusion-limited growth (Epstein-Plesset 1950)

```
dR/dt = D · (C_∞(t) − C_s(t, R)) / (ρ_gas(t) · R) · (1 + R / √(π·D·t))
```

where concentrations are obtained from partial pressures via Henry's law:

```
C_∞(t)       = P_tissue(t) · α_N2          (dissolved N₂ in tissue)
C_s(t, R)    = P_gas(t)    · α_N2
             = (P_amb(t) + 2σ/R) · α_N2   (equilibrium at bubble surface)
```

So the diffusion gradient simplifies to:
```
C_∞ − C_s = α_N2 · M_N2 · (P_tissue(t) − P_amb(t) − 2σ/R)
```

> **Correction (2026-07-09) — the `M_N2` factor.** `α_N2` is a *molar* solubility
> (mol/(m³·Pa)), so `α_N2 · ΔP` is a molar concentration (mol/m³). But `ρ_gas` below is a
> *mass* density (kg/m³). The quotient `D·ΔC/(ρ_gas·R)` then carries units of mol/kg, not m/s.
> Multiplying `ΔC` by `M_N2` puts both on a mass basis. Verified numerically: the uncorrected
> form inflates `dR/dt` by exactly `1/M_N2 = 35.714×`. A uniform scale error is hidden by
> z-scoring, but the two *threshold* quantities — whether `R` crosses `R_crit` (feeding
> `bubble_n_critical`) and whether growth beats the Laplace barrier — are not scale-invariant,
> so the bug corrupts precisely the features standardisation does not wash out.

`P_tissue(t)` is the tissue N₂ partial pressure time series from the Bühlmann simulation
(max across 16 compartments at each step). The full ODE is:

```
dR/dt = D · α_N2 · M_N2 · (P_tissue(t) − P_amb(t) − 2σ/R) / (ρ_gas(t) · R)
        · (1 + R / √(π · D · max(t, ε)))
```

where `ε = 1×10⁻¹⁰ s` prevents division by zero at `t = 0`.

> **Note on the transient term.** For all realistic `t` (≥ 0.01 s) the correction
> `1 + R/√(πDt)` evaluates to ≈1.000 and does no work; at `t = ε` it evaluates to ≈884, spiking
> the initial derivative. Its `t` is dive-clock time, not bubble-age time, which is
> conceptually wrong and happens not to bite only because the term is inert. Consider dropping
> it, or reintroducing it against bubble age, when Correction 11 is resolved.

`ρ_gas(t)` is computed each step from the ideal gas law:
```
ρ_gas(t) = P_gas(t) · M_N2 / (R_gas · T_body)
```

### Physical constants

| Symbol | Value | Source |
|--------|-------|--------|
| `D` | 2.0 × 10⁻⁹ m²/s | N₂ diffusion in blood at 37°C — Weathersby et al. (1984) |
| `α_N2` | 6.84 × 10⁻⁶ mol/(m³·Pa) | N₂ solubility in blood (0.0693 mL/mL/atm converted) — Weathersby et al. (1984) |
| `σ` | 0.050 N/m | Blood–gas surface tension — Van Liew (1991) |
| `ρ_blood` | 1060 kg/m³ | — Merrill et al. (1969) |
| `μ` | 0.003 Pa·s | Dynamic viscosity at 37°C — Charm & Kurland (1974) |
| `M_N2` | 0.028 kg/mol | |
| `R_gas` | 8.314 J/(mol·K) | |
| `T_body` | 310.15 K (37°C) | |
| `R₀` | 0.7 × 10⁻⁶ m | VPM sub-micron stabilised nucleus — Yount (1991) |
| `R_crit` | 12 × 10⁻⁶ m | Critical emboli-forming radius — Yount (1979) |
| `N₀` | 100 sites/mL | Nucleation site density — Yount & Hoffman (1986) |

### Numerical integration

```python
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def ep_rhs(t, y, P_tissue_interp, P_amb_interp):
    R = y[0]
    P_t = P_tissue_interp(t)          # bar
    P_a = P_amb_interp(t)             # bar
    P_g = P_a + 2 * SIGMA / R         # Laplace equilibrium (Pa — convert units as needed)
    rho_gas = P_g * M_N2 / (R_GAS * T_BODY)
    delta_C = ALPHA_N2 * (P_t - P_a - 2 * SIGMA / R)
    correction = 1 + R / np.sqrt(np.pi * D * max(t, 1e-10))
    dRdt = D * delta_C / (rho_gas * R) * correction
    return [dRdt]

sol = solve_ivp(
    ep_rhs, [0, T_dive], [R0],
    method='Radau',                    # stiff solver; purpose-built for this problem
    t_eval=time_points,                # 180 steps matching Bühlmann output
    args=(P_tissue_interp, P_amb_interp),
    rtol=1e-6, atol=1e-9,
)
R_trajectory = sol.y[0]               # metres, shape (180,)
```

**Solver choice — corrected 2026-07-09.** The original text claimed Radau was required and
that "`RK45` will fail or take tiny steps." **Measured on 60 profiles from V2's own seeded
sampler, the opposite is true:**

| Method | Successful solves | Mean time | Mean `nfev` |
|--------|------------------|-----------|-------------|
| `Radau` | **18 / 60** | 22.6 ms | 876 |
| `BDF` | 60 / 60 | 18.9 ms | 520 |
| `LSODA` | 60 / 60 | 2.6 ms | 192 |
| **`RK45`** | **60 / 60** | **1.6 ms** | 148 |

Radau fails on 70% of realistic profiles ("required step size is less than spacing between
numbers"). The cause is the `if R <= 0: return [0.0]` guard in the original `_ep_rhs`: it makes
the RHS non-smooth, which destroys the implicit solver's Jacobian and stalls its step
controller. The fix is a `solve_ivp` **terminal event** at `R = 0.1·R₀` instead of a guard,
which keeps the RHS smooth over the whole integration interval.

Two further measured points: replacing `interp1d` with an `np.interp` closure halves Radau's
cost (22.6 → 10.7 ms), because `interp1d.__call__` costs ~7.3 µs against `np.interp`'s ~0.6 µs
and is paid ~800× per solve. And the spec's "~2 ms/profile, ~5 minutes for 50,000" was
asserted, never measured; the honest figures are ~1.6 ms with `RK45` + `np.interp` + events
(≈1.5 min for 50,000), or ~22 ms with the original recipe (≈18 min) — *for solves that fail*.

`RK45` with a terminal dissolution event is the specified method. `LSODA` is an acceptable
alternative.

**Why not a PINN:** See §Correction 2.

---

## Bubble Feature Extraction

After integration, extract 6 scalar features from `R_trajectory` (units converted to µm
for interpretability):

| Column | Formula | Physiological meaning | Literature basis |
|--------|---------|----------------------|-----------------|
| `bubble_R_max` | `max(R(t))` µm | Peak bubble radius — primary determinant of vessel occlusion risk | Yount (1979): occlusion threshold ~50 µm |
| `bubble_R_surface` | `R(t = T_dive)` µm | Bubble radius at surface arrival — risk of pulmonary embolism | Van Liew & Raychaudhuri (1997): surface arrival radius predicts pulmonary DCS |
| `bubble_dR_dt_max` | `max(dR/dt)` µm/s | Peak growth rate — severity of rapid ascent phase | Srinivasan et al. (2003): growth rate correlates with rapid-ascent DCS |
| `bubble_integrated_volume` | `∫R(t)³ dt` µm³·min | Total embolic load — strongest single predictor of DCS grade | Van Liew & Raychaudhuri (1997) |
| `bubble_n_critical` | `N₀ · max(0, 1 − R_crit/R_max)` | Count of emboli-forming bubbles above critical radius | Yount (1979); Yount & Hoffman (1986) |
| `bubble_terminal_velocity` | `2R_max²(ρ_blood − ρ_gas)g / 9μ` µm/s | Stokes terminal buoyant velocity — diagnostic only; see note | Stokes (1851) |

> **Note on `bubble_terminal_velocity`:** For R_max ~ 50 µm this gives ~0.2 mm/s. Blood flow
> velocity is ~10–100 mm/s. Buoyant rise does not govern bubble transport in circulating
> blood; advection dominates. This column is included as a diagnostic feature but is
> **excluded from the logistic model**.

---

## Extended Logistic Model

### Complete V2 logistic (reconstructed from source)

The V2 reference document is outdated. The running logistic as coded in
`generate_dcs_dataset_v2.py` is:

```python
log_odds = -18.08 + 18.31 * physics_risk_score   # intercept / slope
log_odds += 1.2   * pfo                           # PFO
log_odds += {'none': 0.0, 'moderate': 0.2, 'heavy': 0.5}[pre_dive_exercise]
log_odds += 0.4   * alcohol
log_odds += {'good': 0.0, 'poor': 0.15}[sleep_quality]
if age > 40:
    log_odds += 0.3 * ((age - 40) / 30.0)         # age (original term)
if age > 50:
    log_odds += min(0.4 * ((age - 50) / 20.0), 0.40)  # age boost >50
log_odds += {0: 0.0, 1: 0.15, 2: 0.30}[hydration_status]  # direct logistic term
log_odds += {0: +0.45, 1: 0.0, 2: -0.50}[fitness_level]    # direct logistic term
```

Note: `hydration_status` and `fitness_level` appear in **both** the physics layer (Bühlmann
half-time multipliers) and the logistic layer (direct log-odds terms). This double-counting
was introduced in Fixes 7 and 8 of the V2 development cycle. V3 inherits this structure and
documents it honestly as a known modelling choice.

### V3 extension — bubble terms

Five of the six bubble features enter the logistic as standardised z-scores
(zero mean / unit variance computed across all 50,000 profiles):

```python
# Pre-specified design coefficients — see justification below
log_odds += 0.50 * z(bubble_integrated_volume)   # strongest predictor per VL&R (1997)
log_odds += 0.45 * z(bubble_n_critical)          # diffuse embolic load
log_odds += 0.40 * z(bubble_R_max)               # peak occlusion risk
log_odds += 0.30 * z(bubble_R_surface)           # pulmonary arrival risk
log_odds += 0.25 * z(bubble_dR_dt_max)           # rapid-ascent severity
# bubble_terminal_velocity excluded — dominated by R_max, no independent mechanism
```

### Justification for coefficient magnitudes

These are **pre-specified design parameters** for the data-generating process, not values
fit to data. Choosing them precedes label generation; they define what the labels mean.

Relative ordering from Van Liew & Raychaudhuri (1997) Table 3: integrated bubble volume is
the strongest predictor of DCS grade severity; bubble count (n_critical) and peak radius
closely follow; surface arrival radius and peak growth rate have moderate additional value.
The chosen values (0.25–0.50) place bubble features in the same range as the moderate
physiological contributors already in the V2 logistic (alcohol: +0.40, exercise moderate:
+0.20). This is a calibrated choice: bubble features should matter, but not so much that
they overwhelm the nitrogen-loading signal already carried by `physics_risk_score`.

These coefficients cannot be "discovered" from V2 labels (which contain no bubble signal)
and are not claimed to be empirically derived from real dive-outcome data. They are stated
transparently as synthetic benchmark parameters.

### Intercept recalibration

After adding bubble terms, the intercept is re-fitted via least-squares at three calibration
points to restore the DAN/Howle targets:

| `physics_risk_score` | Target `P(DCS)` | Source |
|---------------------|-----------------|--------|
| 0.70 | ~0.5% | DAN Annual Diving Report; Howle et al. (2017) |
| 0.85 | ~8.0% | DAN Annual Diving Report; Howle et al. (2017) |
| 1.00 | ~55%  | Howle et al. (2017) |

Baseline diver: age=35, moderate fitness, well-hydrated, no PFO, no alcohol, good sleep.
Bubble features at baseline are the median values across non-high-risk standard profiles
(not zero — some bubble growth occurs on all dives). The slope (18.31) and all other
coefficients are unchanged; only the intercept is adjusted.

---

## Output Format

```
~/Desktop/DCS_PINN_DATASET_V3/
├── generate_dcs_dataset_v3.py      # single script — no training step required
├── bubble_scaler.json              # StandardScaler mean_/scale_ as JSON, never a pickle:
│                                   # joblib.load on an untrusted pickle is arbitrary code
│                                   # execution, and this is twelve floats.
├── dive_profiles_timeseries.npy    # (50000, 180) float32 — same format as V2
├── dive_profiles_features.csv      # 47 columns (41 V2 + 6 bubble)
└── dive_profiles_sample.png        # 3×3 sample plot
└── docs/
    └── specs/
        └── 2026-06-23-pinn-bubble-dynamics-design.md
```

### New CSV columns (appended after V2's 41)

| Column | Type | Units | In logistic? |
|--------|------|-------|-------------|
| `bubble_R_max` | float | µm | Yes |
| `bubble_R_surface` | float | µm | Yes |
| `bubble_dR_dt_max` | float | µm/s | Yes |
| `bubble_integrated_volume` | float | µm³·min | Yes |
| `bubble_n_critical` | float | dimensionless | Yes |
| `bubble_terminal_velocity` | float | µm/s | No — diagnostic only |

Total: **47 columns**.

---

## Validation

### Retained from V2

All 5 structural checks (row count, NaN, score bounds, label binary, DCS rate ≥ 5%) and all
6 physiological directional checks (fitness, age, hydration, PFO, alcohol, breathing gas).

### New bubble physics checks

```
=== BUBBLE PHYSICS SANITY CHECKS ===
[PASS/FAIL] solve_ivp success status = True for all 50,000 profiles   (hard abort, not a warning)
[PASS/FAIL] bubble_R_max >= R₀ (0.7 µm) for all profiles
[PASS/FAIL] std(bubble_R_max) > 0            — non-degenerate; see Correction 11
[PASS/FAIL] bubble_R_max <= R_MAX_PLAUSIBLE  — an unbounded EP bubble reaches mm radii
[PASS/FAIL] bubble_n_critical = 0 for all profiles where bubble_R_max ≤ R_crit (12 µm)
[PASS/FAIL] bubble_integrated_volume > 0 for all profiles
```

**Three corrections to the original check list (2026-07-09):**

1. **`bubble_R_max > R₀` was strict and always fails.** `R₀` is the initial condition and is
   included in `t_eval`, so any non-growing bubble has `max R(t) = R₀` exactly. Under the
   current model that is *every* profile (Correction 11). The check must be `>=`. Loosening it
   is only safe once the `ep_solve_failed` flag exists, because the old `R₀` fallback was
   indistinguishable from a genuine non-growing solve.
2. **The quartile monotonicity checks are tautologies and have been removed.** Labels are drawn
   with a *positive* coefficient on `z(bubble_R_max)` and `z(bubble_integrated_volume)`. That a
   positive coefficient yields a positive gradient across that feature's quartiles is
   arithmetic, not evidence. The check cannot fail, so it validates nothing. What *would* be
   informative — and is not currently measured — is whether a model trained on the bubble
   features outperforms one trained on `physics_risk_score` alone.
3. **Two new checks guard the failure modes that actually occurred:** zero-variance bubble
   features (the model never grows a bubble), and unbounded growth (once `R` clears the
   barrier, `2σ/R` collapses and growth runs away).

`solve_ivp` failure is now a **hard abort above a 1% rate**, never a silent `R₀` substitution:
the fallback wrote fabricated minimum-risk physics into rows that are then labelled and shipped
indistinguishably from real solves, *and* it entered the `StandardScaler` fit, shifting the
z-scores of all 50,000 rows. Failure is correlated with supersaturation, hence with the label.
An `ep_solve_failed` column is emitted so any residual failures are auditable after the fact.

### Calibration targets

- Overall DCS rate: 5–12%
- All 6 physiological directional checks: PASS
- Physics residuals: not applicable (exact numerical solve)

---

## Honesty Ceiling

No decompression model is derived from first principles; all are semi-empirical and
calibrated to real dive-trial outcomes. This dataset is entirely synthetic. A more elaborate
bubble model makes it more **internally sophisticated**, not more **validated**. The bubble
logistic coefficients are design parameters for a synthetic data-generating process, not
estimates from a real-outcome study.

> **Amended 2026-07-09.** An earlier revision of this section claimed "the bubble features and
> labels cannot be checked against real DCS outcomes without real dive logs." Real dive logs
> exist in this workspace (`FINAL DIVE/datasets/real/`), and the check was run — see
> Correction 13. It came back **negative**: bubble features do not improve on Bühlmann tissue
> loading against real outcomes with real controls. The honest ceiling is therefore *lower*
> than this section originally implied, not higher. The check was possible; it simply did not
> support the bubble layer. Note further that `physics_risk_score` itself was anti-predictive
> out of sample under the more faithful profile reconstruction, so the limiting factor is the
> fidelity of the extracted real data, not the elaborateness of the model.

The defensible framing is: **a physically-consistent, literature-calibrated synthetic
benchmark for evaluating DCS-risk methods**. All scope claims should be limited accordingly.
In particular, V3 should not be described as "more accurate" than V2 — it is richer in the
physical mechanisms it encodes, with that richness grounded in the decompression bubble
literature.

---

## Dependencies

```
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
tqdm >= 4.60
scipy >= 1.9          # solve_ivp with Radau
scikit-learn >= 1.2   # StandardScaler
# joblib intentionally NOT a dependency — the scaler is serialised as JSON, not pickle.
```

No PyTorch, no DeepXDE, no GPU required.

---

## Literature References

- Epstein, P.S. & Plesset, M.S. (1950). On the stability of gas bubbles in liquid-gas solutions. *Journal of Chemical Physics*, 18(11).
- Yount, D.E. (1979). Skins of varying permeability: a stabilization mechanism for gas cavitation nuclei. *Journal of the Acoustical Society of America*, 65(6).
- Yount, D.E. (1991). Comment on "Computations of dissolved gas liberation in tissue." *Journal of Applied Physiology*, 71(2). [VPM R₀ value]
- Yount, D.E. & Hoffman, D.C. (1986). On the use of a bubble formation model to calculate diving tables. *Aviation, Space, and Environmental Medicine*, 57(2).
- Van Liew, H.D. (1991). Simulation of the dynamics of decompression sickness bubbles. *Undersea Biomedical Research*, 18(5–6).
- Van Liew, H.D. & Raychaudhuri, S. (1997). Stabilized bubbles in the body: pressure-radius relationships and limits to stabilization. *Journal of Applied Physiology*, 82(6). [Coefficient ordering and integrated volume as strongest predictor]
- Gernhardt, M.L. (1991). *Development and evaluation of a decompression stress index.* PhD thesis, University of Pennsylvania.
- Srinivasan, R.S., Gerth, W.A. & Powell, M.R. (2003). Mathematical model of diffusion-limited evolution of multiple gas bubbles in tissue. *Annals of Biomedical Engineering*, 31.
- Weathersby, P.K., Homer, L.D. & Flynn, E.T. (1984). On the likelihood of decompression sickness. *Journal of Applied Physiology*, 57(3). [D and α_N2 values]
- Howle, L.E. et al. (2017). The probability and severity of decompression sickness. *PLOS One*. doi:10.1371/journal.pone.0172665
- Krishnapriyan, A. et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS 2021*. [Cited as failure catalogue for stiff ODEs, not support]
- Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378. [Context for when PINNs are appropriate]
- Merrill, E.W. et al. (1969). Rheology of human blood. *Circulation Research*.
- Charm, S.E. & Kurland, G.S. (1974). *Blood Flow and Microcirculation*. Wiley.

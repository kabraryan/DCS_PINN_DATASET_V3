# Decompression Algorithm Benchmark — Design Spec

**Date:** 2026-07-10
**Status:** Approved (design), not yet implemented
**Supersedes:** nothing. Complements `docs/specs/2026-06-23-pinn-bubble-dynamics-design.md` (Corrections 1–13).

---

## Purpose

Score decompression algorithms against **real dive outcomes** — 2,700 dives from US Navy report
NMRC 99-02, containing 1,932 real non-DCS controls — under an evaluation protocol that does not
inflate results.

The benchmark answers one question per algorithm: **does it add anything beyond the three raw
dive parameters?** Correction 13 established that Bühlmann's supersaturation score and the
Epstein-Plesset bubble radius both *subtract*. This makes that finding reproducible, extensible
to new algorithms, and hard to fake.

### Explicit non-goal: this is not a dive-planning tool

The benchmark **never emits a probability**. There is no `predict_proba` on the public surface.

The reason is arithmetic, not caution. The best measured ranking on this data is AUC ≈ 0.71.
Recreational DCS incidence is 0.01–0.1% per dive. A warning tuned to catch 80% of DCS cases at
0.05% incidence would flag **52% of all dives**, of which **1 in 1,309** would be a real case —
while the 48% it called safe would still contain 20% of the cases. Even a hypothetical AUC 0.90
model is wrong 414 times in 415 at that base rate. The limiting factor is that DCS is rare, not
that the model is weak.

Further, the ~16% DCS rate in this dataset is an artifact of trials **designed to provoke DCS**
on partially-extracted negatives (2,700 of 8,578). The population is screened, fit, mostly young
military divers doing deep single bounce dives on air, 1944–1997. The file contains **no
per-diver covariates** — no age, temperature, exertion, or PFO. Recreational DCS is driven by
exactly those, plus multi-day repetitive profiles this data does not contain.

Absolute risk transfer from this data to a recreational diver is not a modelling problem. It is
impossible. `Brier` is printed for completeness and labelled *not calibration*.

---

## Scope

### In

| Entry | `risk_index` | `deficit` | Status |
|---|---|---|---|
| `raw` (reference, not an algorithm) | — | — | exists |
| `zhl16c` | max M-value ratio | ✓ ceiling-driven | exists, needs extraction |
| `zhl16c_gf` | surfacing gradient factor | ✓ ceiling-driven | new (~30 lines) |
| `ep_bubble` | `R_max` (EP + VPM skin) | `None` | exists |

Two profile reconstructions: `linear`, `staged`.
Three marginal-outcome rules: `exclude` (primary), `positive` (`0.5→1`), `negative` (`0.5→0`).

### Out (deliberately)

- **RGBM** — proprietary, unspecified in the literature. Not benchmarkable.
- **VPM-B** — the natural next algorithm. Giving `ep_bubble` a schedule *is* VPM-B (a
  critical-radius ceiling with crushing pressure and skin regeneration). It belongs in its own
  addition, not as a free extension.
- **Thalmann LEM / USN probabilistic** — the strongest comparator and the model actually fitted
  to this data, but it is a maximum-likelihood hazard model: a project of its own.
- **Completing the 8,578-dive extraction.** You already hold 419 of the report's 434 DCS cases
  (96% of positives). The missing ~5,878 dives are almost entirely negatives. Discrimination is
  limited by event count, so extraction would add ~15 positives and would not move the ceiling.
  It would tighten variance and fix the base rate; it is not on this critical path.
- **Depth–time curves for negatives.** They do not exist. Vol I Key Files print only scalars
  (`outcome, n_raw, depth_fsw, bottom_time_min, ascent_time_min, T1_min, T2_min`); Vol II has
  curves but is DCS-events-only. The reconstruction error in Correction 13 is a permanent
  ceiling on this source, not a data-recovery task.

---

## Architecture

An algorithm is a **pure function of a profile**. None of them train — ZHL-16C has no parameters
to learn, and the one free parameter that exists (`ep_bubble`'s `R₀`) produced a bimodal,
unidentifiable likelihood when fitted (Correction 13). A `fit`/`predict` interface would encode
that mistake into the type system.

```python
class Algorithm(Protocol):
    name: str
    params: dict                                     # hashed into the cache key
    def risk_index(self, p: Profile) -> float: ...   # required; a RANK, never a probability
    def deficit(self, p: Profile) -> float | None:   # None if the algorithm defines no schedule
```

`deficit` is optional by design. It requires a *ceiling* — a depth you may not ascend above.
ZHL-16C and ZHL+GF have one. `ep_bubble` does not; nothing in Epstein-Plesset says when ascent
is permitted. The harness reports per-metric coverage rather than filling the gap.

`raw` is not in the registry. It is the must-beat feature set, held fixed independent of which
algorithm is under test.

### File map

```
benchmark/
  profile.py          Profile dataclass; reconstruct(dive, recon) -> Profile
  buhlmann.py         ZHL-16C table, Haldane step, M-value, ceiling      [shared, extracted]
  cache.py            content-addressed cache at the two boundaries
  algorithms/
    __init__.py       REGISTRY: name -> Algorithm
    zhl16c.py
    zhl16c_gf.py      gradient factors (gf_lo, gf_hi); default 30/70
    ep_bubble.py      EP + VPM skin; R0 configurable
  evaluate.py         nested grouped CV, four-gate rule, dual-recon gate, controls
scripts/
  run_benchmark.py    CLI -> RESULTS.md; --check regenerates and diffs
tests/
  test_buhlmann.py
  test_algorithms.py
  test_reconstruction.py
  test_evaluate.py
```

### Boundaries

- `profile.py` knows nothing about algorithms.
- `algorithms/*` know nothing about labels, cross-validation, or statistics.
- `evaluate.py` knows nothing about decompression: it consumes a per-dive numeric matrix, an
  outcome vector, and trial groups.

Each is testable alone. ZHL-16C's ceiling can be checked against a published Navy table entry
without touching cross-validation.

### One extraction, not a fifth fork

`buhlmann.py` lifts the ZHL-16C table and Haldane step out of `scripts/fit_r0_to_real_dives.py`
and `scripts/staged_ascent.py`, which each carry a copy today. Those scripts then import it.
Copy-paste duplication is how the compartment-16 `b = 0.8693` bug reached four generators, a
README, two design docs, and a unit test that asserted it (Correction 10).

---

## Data flow

```
dcs_all_dives.csv (2,700)
   │ bounce window: depth/bottom_time/ascent_time ≤ 300      -> 2,230
   │ marginal rule: exclude | positive | negative
   ▼
dives (depth_fsw, bottom_time_min, ascent_time_min, outcome, data_set)
   │
   ├── reconstruct(dive, "linear") ─┐
   └── reconstruct(dive, "staged") ─┤   CACHE 1   key = (dive_id, recon)
                                    ▼
                            Profile(t_min, depth_fsw)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
       zhl16c                  zhl16c_gf                 ep_bubble      CACHE 2
   (risk, deficit)          (risk, deficit)             (risk, None)    key = (dive_id, recon,
          └─────────────────────────┼─────────────────────────┘                 algo, params_hash)
                                    ▼
              per-dive matrix: 2,230 × {algo × metric × recon}
                                    ▼
                              evaluate.py
                 nested grouped CV -> AUC, paired deltas, controls
                                    ▼
                        dual-reconstruction gate
                                    ▼
                              RESULTS.md
```

### Caching

The expensive work is invariant to the CV protocol; the CV protocol is invariant to algorithm
internals. Cutting at those two boundaries means:

- Adding VPM-B recomputes only VPM-B's column.
- Re-running statistics after a CV change recomputes nothing.
- Changing `gf_lo/gf_hi` invalidates only `zhl16c_gf`; changing `R₀` invalidates only
  `ep_bubble`. The `params_hash` in the key is what makes a later gradient-factor sweep cheap
  without a config-file layer.

Full grid: 2 reconstructions × 3 algorithms × 2,230 dives ≈ 13,400 solves, dominated by the EP
ODE at ~1.6 ms/solve (RK45 + `np.interp` + terminal event; Radau fails on 70% of realistic
profiles — Correction 12). Under a minute. Everything downstream of Cache 2 is free.

Both caches are computed over **all 2,230 bounce dives**, before any marginal rule is applied.
The marginal rule selects rows downstream, so switching it re-runs only the statistics — zero
solves. This is why the 2 × 3 sensitivity grid costs the same as a single cell.

**Not cached:** CV split seeds. They must vary across repeats; freezing them would silently
shrink the error bars.

### Provenance

`RESULTS.md` is generated, never hand-edited. Its header records the input CSV's content hash,
the git SHA, resolved parameters, and the exact command. `run_benchmark.py --check` regenerates
and diffs, exiting non-zero on drift. This is the machine-checkable form of the Provenance
discipline rule added to the V3 spec after prose failed to prevent Corrections 9 and 12 from
fabricating statistics.

---

## Metrics and statistical protocol

### Two comparisons; only the second decides

1. **Algorithm scalar alone vs `raw`** — does it capture what the dive parameters capture?
2. **`raw` + algorithm scalar vs `raw` alone** — does it add anything *beyond* them?

(2) is decisive. An algorithm that cannot beat `depth, bottom_time, ascent_time` has not earned
its physics.

### Fixed reference

Unpenalised logistic on the three raw columns: **AUC 0.6419 ± 0.0547** (nested grouped CV,
marginals excluded, n = 1,948, 305 positives, 38 trials). Computed independently of the feature
set under test. `logistic [raw]` must reproduce it at delta exactly `+0.0000` — a harness
self-test.

### Cross-validation

Outer `GroupKFold(5)` by `data_set` × 5 repeats; inner `GroupKFold(3)` for hyperparameters.
Grouping is mandatory: trials differ in protocol aggressiveness (DCS rates 4.6%–35%), and
ordinary CV inflates AUC by +0.045 (logistic) to +0.075 (HistGB) by letting a model memorise
trial identity. Effective sample size is ~38 trials, not 1,948 dives; fold sd ≈ 0.06.

### An effect counts only if it clears all four gates

| Gate | Threshold | Stops |
|---|---|---|
| Magnitude | \|ΔAUC\| > 0.03 | fold noise |
| Paired significance | Wilcoxon p < 0.05 over folds | lucky splits |
| **Coefficient sign** | matches mechanism | **confounds** |
| Permutation null | beats ≥95% of shuffled columns | extra-parameter artifacts |

The sign gate is load-bearing. At `R₀` = 0.7 µm the bubble feature cleared magnitude,
significance, *and* the permutation null — while predicting **fewer** DCS cases when bubbles
grew (`AUC(R_max)` = 0.489, β = −0.25). It was a proxy for "deep short dive." More deficit must
mean more DCS; higher supersaturation must mean more DCS. A sign-inverted win is scored
`NOT SUPPORTED`, never reported as a discovery.

### Dual-reconstruction gate

| Verdict | Condition |
|---|---|
| `SUPPORTED` | all four gates pass under **both** `linear` and `staged` |
| `RECONSTRUCTION-SENSITIVE` | passes under one, not the other |
| `MARGINAL-SENSITIVE` | passes under `exclude`, reverses under `0.5→1` or `0.5→0` |
| `NOT SUPPORTED` | otherwise |

`exclude` is the primary marginal rule; the other two run as sensitivity and can only **demote**
a verdict, never promote one. Consequently an effect that fails under `exclude` is
`NOT SUPPORTED` regardless of how it scores under `0.5→1` or `0.5→0`. `RESULTS.md` prints the
full 2 × 3 grid so a reader sees the disagreement rather than a winner.

The marginal rule is a **safety-relevant choice, never a default**. It is not merely a
sensitivity axis: under `0.5→0` the entire nonlinear-model advantage vanishes (RandomForest
0.6336 vs baseline 0.6315), meaning the nonlinearity trees exploit lives largely in the
marginal cases. The rule must be passed explicitly on the command line.

Rationale: Correction 13 showed conclusions flip between reconstructions (`prs` scored 0.6004
in-sample under linear, 0.3843 out-of-sample under staged — *worse than chance*). Staged is
provably closer to the real curves (median RMSE 36.08 vs 48.71 fsw over 72 gold curves,
Wilcoxon p = 1.8×10⁻¹¹) yet still beats a predict-the-mean baseline on only 44.4% of dives.
Three scalars do not determine a dive profile. The gate makes that weakness the safeguard.

### Standing controls, printed every run

- **Label shuffle** → must return ≈ 0.5 (currently 0.5074). Drift means the protocol broke.
- **Leakage gap** → ordinary CV minus grouped CV, per algorithm. Printed so nobody quotes the
  inflated figure.
- **Permutation null** per algorithm column.

### Reported

AUC (ranking), PR-AUC (positives are 15.7%), paired fold deltas with Wilcoxon p. Brier printed
and labelled *not calibration*.

### Deficit primary, risk index secondary — the gap is a result

`deficit` depends mainly on descent-and-bottom, which the three scalars constrain well, and on
the **recorded** ascent time. `risk_index` depends on the ascent *shape*, which they do not
constrain. If `deficit` yields a stable verdict where `risk_index` flips across reconstructions,
that difference **quantifies what the missing ascent data costs** — the number Correction 13
could only gesture at.

---

## Error handling

Failures are loud. Nothing is silently filled in. Each rule exists because its absence already
cost this project something.

| Condition | Response |
|---|---|
| `solve_ivp` fails on a dive | `ep_solve_failed` column; row excluded from every statistic; **hard abort above 1%** |
| An algorithm's column has zero variance | `RuntimeError`: degenerate model (Correction 11) |
| Staged schedule hits `MAX_STOP_ITERS` | per-dive flag, counted, reported |
| Recorded ascent shorter than travel time | straight-ascent fallback, **counted and printed** |
| Saturation / >300 fsw dives excluded | count **and their DCS rate** printed; the exclusion is outcome-correlated (24.3% vs 13.7%) |
| `RESULTS.md` differs from a fresh run | `--check` exits non-zero |

A failed EP solve once became `R_traj = full(R0)`: fabricated minimum-risk physics, shipped
indistinguishably from real solves, poisoning the scaler for all 50,000 rows — with failure
*correlated to the label*. Prevented by construction here.

---

## Testing

Property tests, not restatements. No test recomputes the implementation. Each would have caught
a real bug from this project.

```
test_buhlmann.py
  b column strictly increasing; a strictly decreasing        <- the 0.8693 bug
  Haldane: after one half-time the gas gap halves            <- analytic
  ceiling reproduces a published USN table entry             <- external ground truth

test_algorithms.py
  risk_index has non-zero variance over 20 real profiles     <- Correction 11 in 15 lines
  EP under constant supersaturation follows sqrt(t) growth   <- analytic solution
  trajectory stable under rtol refinement (halve -> delta < tol)
  monotone supersaturation => monotone R
  deficit is None for ep_bubble, float for zhl16c            <- the optional-metric contract

test_reconstruction.py
  realised ascent time == recorded ascent time
  staged RMSE < linear on the 72 gold curves                 <- regression vs ground truth

test_evaluate.py
  shuffled labels -> AUC ~ 0.5                               <- protocol soundness
  ordinary CV > grouped CV                                   <- leakage is detectable
  logistic[raw] reproduces the baseline at delta 0.0000      <- harness self-test
  a sign-inverted feature scores NOT SUPPORTED               <- the gate that caught the confound
```

The most valuable single test is `risk_index has non-zero variance`. Fifteen lines, a few
seconds, and it would have killed the degenerate bubble model before 1,500 lines of design were
built on it. It runs in CI, not only after a full generation.

### One honest gap

`ceiling reproduces a published USN table entry` requires a hand-transcribed fixture from the US
Navy air decompression tables. The fixture will cite its primary source. **If the entry cannot be
verified against a primary source, the test does not ship and this document is amended to say
so** — rather than shipping a fixture asserting whatever the implementation happens to produce.
That is precisely how `test_table_compartment16` came to guard the bug it should have caught.

---

## Success criteria

1. `pytest` green, including all four control tests in `test_evaluate.py`.
2. `python scripts/run_benchmark.py` regenerates `RESULTS.md` in **under 25 minutes from a cold
   cache and under 30 seconds warm**; `--check` passes on the committed file.

   > **Amended 2026-07-10.** This criterion originally said "under two minutes from a cold
   > cache". That was asserted, never measured, and it contradicts the statistical protocol this
   > same spec mandates. Measured: the 30 gate cells each cost ~2,400 logistic fits (a baseline
   > CV, a full CV, and a **20-permutation null**, each a nested grouped CV of 400 fits) at
   > ~14 ms per fit — **72,000 fits, ≈17 minutes**. The EP ODE (≈21 s) and profile
   > reconstruction (≈5 s) are rounding errors beside it.
   >
   > The permutation null is one of the four gates, not an optional flourish; it is what stops
   > an extra parameter from looking like a discovery. Shrinking it to hit an arbitrary time
   > target would buy speed with exactly the rigour this benchmark exists to supply. The budget
   > moved instead. Warm re-runs are seconds, because both caches sit upstream of the statistics.
3. `RESULTS.md` reports a verdict for every **algorithm-metric pair that exists** × 2
   reconstructions × 3 marginal rules, with the three standing controls. There are **5 pairs**,
   not 8 — `raw` is the reference and carries neither metric, and `ep_bubble` has no `deficit`:

   | pair | |
   |---|---|
   | `zhl16c` × `risk_index` | `zhl16c` × `deficit` |
   | `zhl16c_gf` × `risk_index` | `zhl16c_gf` × `deficit` |
   | `ep_bubble` × `risk_index` | — |

   So **30 verdicts** (5 × 2 × 3). Missing pairs are printed as `N/A — no schedule`, never as a
   blank or a zero.
4. The raw baseline reproduces at AUC 0.6419 ± 0.0547 and `logistic [raw]` delta `+0.0000`.
5. Adding a hypothetical fifth algorithm requires exactly one new file and one registry line,
   with no edit to `evaluate.py`. (Demonstrated by a trivial `constant` algorithm in tests,
   which must score `NOT SUPPORTED`.)

### Expected result, stated in advance

Based on Corrections 12 and 13, the prediction is that **no algorithm reaches `SUPPORTED`**, and
that `zhl16c` and `ep_bubble` land at `NOT SUPPORTED` or `RECONSTRUCTION-SENSITIVE`. Recording
this before implementation is deliberate: it makes a "successful" benchmark run falsifiable
rather than confirmatory. If an algorithm *does* reach `SUPPORTED`, that is a real finding and
the first order of business is to attack it with the sign and permutation gates, not to publish
it.

---

## References

- Corrections 9–13, `docs/specs/2026-06-23-pinn-bubble-dynamics-design.md`
- `scripts/train_baseline.py` — the grouped/nested/controlled harness this generalises
- `scripts/staged_ascent.py`, `scripts/validate_reconstruction.py`
- Bühlmann, A.A. (1984/1995). *Decompression–Decompression Sickness.*
- Yount, D.E. (1979, 1991). Varying-permeability model; stabilised gas nuclei.
- Weathersby, P.K., Homer, L.D. & Flynn, E.T. (1984). On the likelihood of decompression sickness.
- Howle, L.E. et al. (2017). The probability and severity of decompression sickness. *PLOS One.*
- Temple, Ball, Weathersby, Parker & Survanshi (1999). *NMRC 99-02*, Vols I & II.

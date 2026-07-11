# Decompression Algorithm Benchmark — Results

> Generated file. Do not edit. Regenerate with the command below;
> `--check` fails if this file has drifted.

## Provenance

- command: `python scripts/run_benchmark.py --marginal exclude --repeats 5 --seed 0`
- git: `374e5f7`
- input: `/Users/aryankabra_test/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv` sha256[:16] `409d7629d42edfce`
- python 3.13.13, numpy 2.4.6, scipy 1.18.0, sklearn 1.9.0
- 0 solver failures

## Cohort

- 2700 dives; kept 2230 bounce dives (82.6%)
- dropped 470 saturation / >300 fsw excursions; their DCS rate was 24.3% vs 13.7% kept (**the exclusion is outcome-correlated**)

## Verdicts

| algorithm | metric | verdict | ΔAUC (staged, exclude) | sign | reasons |
|---|---|---|---|---|---|
| `zhl16c` | `risk_index` | **NOT SUPPORTED** | +0.0068 | +1 | magnitude |+0.0068| <= 0.03; p=0.201 >= 0.05; beats only 75% of permutations |
| `zhl16c` | `deficit` | **NOT SUPPORTED** | -0.0083 | +1 | magnitude |-0.0083| <= 0.03; p=0.055 >= 0.05; only 36% of folds improved; beats only 0% of permutations |
| `zhl16c_gf` | `risk_index` | **NOT SUPPORTED** | -0.0038 | +1 | magnitude |-0.0038| <= 0.03; p=0.201 >= 0.05; beats only 20% of permutations |
| `zhl16c_gf` | `deficit` | **NOT SUPPORTED** | +0.0010 | +1 | magnitude |+0.0010| <= 0.03; p=0.946 >= 0.05; only 40% of folds improved; beats only 45% of permutations |
| `ep_bubble` | `risk_index` | **NOT SUPPORTED** | -0.0002 | +1 | magnitude |-0.0002| <= 0.03; p=0.861 >= 0.05; only 60% of folds improved; beats only 50% of permutations |

## Controls

- label shuffle → AUC 0.5230 (must be ≈ 0.5)
- leakage gap (ordinary − grouped) → +0.0440
- baseline (logistic on 3 raw features) → AUC 0.6425 ± 0.0532

## Reading this table

AUC is a **ranking**, never a probability. The ~16% DCS rate here reflects Navy
trials designed to provoke DCS on partially-extracted negatives (2,700 of 8,578).
This benchmark is not a dive-planning tool. Fold sd ≈ 0.06, so |ΔAUC| < 0.03 is
noise. `N/A — no schedule` means the algorithm defines no ceiling.

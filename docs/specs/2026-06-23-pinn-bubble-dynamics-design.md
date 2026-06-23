# DCS PINN Dataset V3 — Design Spec
**Date:** 2026-06-23
**Status:** Approved
**Output folder:** `~/Desktop/DCS_PINN_DATASET_V3/`

---

## Overview

V3 extends the V2 Bühlmann + logistic pipeline with a Physics-Informed Neural Network (PINN) layer that solves coupled Epstein-Plesset and Rayleigh-Plesset bubble dynamics equations. The PINN produces six bubble state features per dive profile which are added as inputs to an extended logistic DCS probability model. The result is a 47-column dataset (41 V2 columns + 6 bubble features) that captures nitrogen bubble nucleation, growth, and transport mechanisms not represented in the Bühlmann model alone.

---

## Motivation

The Bühlmann ZHL-16C model (V1/V2) tracks dissolved nitrogen in 16 tissue compartments and flags M-value violations. It does not model what happens after supersaturation: the nucleation of nitrogen microbubbles, their growth driven by concentration gradients, and their mechanical and embolic effects on tissue. These bubble dynamics are the proximate cause of DCS symptoms — Bühlmann predicts the precondition, not the mechanism.

Two governing equations describe bubble behaviour in blood:

- **Epstein-Plesset (EP)** — models gas diffusion across the bubble wall, driven by the dissolved N₂ concentration gradient between tissue and bubble interior. This is the dominant mechanism for bubble growth on recreational dive timescales (minutes to hours).
- **Rayleigh-Plesset (RP)** — models the mechanical pressure-radius response of the bubble to ambient pressure changes during ascent. Important for capturing bubble expansion on ascent and interaction with vessel walls.

Neither equation alone is sufficient: EP drives the gas flux that determines internal bubble pressure, which feeds into RP's mechanical response. A coupled solve is required for physical accuracy.

---

## Pipeline Architecture

```
Dive profile + physiological parameters
              ↓
   ┌─────────────────────────┐
   │  [1] Bühlmann ZHL-16C   │  (unchanged from V2)
   │  16-compartment N2 sim  │
   └─────────────────────────┘
              ↓
   tissue_sat_0..15, physics_risk_score, P_amb(t)
              ↓
   ┌─────────────────────────┐
   │  [2] Universal PINN     │  (new in V3)
   │  Coupled EP + RP solver │
   └─────────────────────────┘
              ↓
   6 bubble state features
              ↓
   ┌─────────────────────────┐
   │  [3] Extended logistic  │  (V2 model + bubble terms)
   │  DCS probability model  │
   └─────────────────────────┘
              ↓
   dcs_probability → Bernoulli label
```

The PINN is a **universal parametric surrogate**: one network trained once across all dive conditions, conditioned on per-dive inputs, queried at arbitrary time `t`. Training takes 3–5 hours on Apple Silicon MPS. Inference per profile at 180 time points takes milliseconds.

---

## Governing Equations

### Epstein-Plesset (gas diffusion)

```
dR/dt = D · (C_∞(t) − C_s) / (R · ρ_gas) · (1 + R / √(π·D·t))
```

| Symbol | Description | Value / Source |
|--------|-------------|----------------|
| `R(t)` | Bubble radius (m) | solved variable |
| `D` | N₂ diffusion coefficient in blood | 2×10⁻⁹ m²/s — Weathersby et al. (1984) |
| `C_∞(t)` | Dissolved N₂ concentration in tissue | derived from Bühlmann peak tissue sat |
| `C_s` | N₂ concentration at bubble surface (saturation) | derived from Henry's law at P_amb |
| `ρ_gas` | Density of N₂ gas at body temperature | 1.14 kg/m³ at 37°C, 1 atm |

`C_∞(t)` is computed from the Bühlmann tissue saturation outputs at each time step:
```
C_∞(t) = max(peak_tissue_sat_0..15) · ρ_blood / M_N2
```
where `M_N2 = 28 g/mol` and `ρ_blood = 1060 kg/m³`. This is the coupling between the Bühlmann layer and the PINN.

### Rayleigh-Plesset (mechanical dynamics)

```
R·R'' + (3/2)·(R')² = (1/ρ_blood) · [P_gas(t) − P_amb(t) − 4μ·R'/R − 2σ/R]
```

| Symbol | Description | Value / Source |
|--------|-------------|----------------|
| `ρ_blood` | Blood density | 1060 kg/m³ — Merrill et al. (1969) |
| `μ` | Dynamic viscosity of blood at 37°C | 0.003 Pa·s — Charm & Kurland (1974) |
| `σ` | Blood-gas surface tension | 0.050 N/m — Van Liew (1991) |
| `P_amb(t)` | Ambient pressure at depth | `depth(t) × 0.1 + P_surface` bar |
| `P_gas(t)` | N₂ partial pressure inside bubble | updated each step from EP gas flux |

`P_gas(t)` is updated at each time step using the ideal gas law applied to the moles of N₂ inside the bubble, with the EP equation providing the rate of gas entry/exit.

### Coupling

EP provides the rate of change of gas content inside the bubble → determines `P_gas(t)` → feeds into RP's pressure imbalance term → RP gives `R(t)` and `R''(t)` → updated `R` feeds back into EP's diffusion geometry term. The system is integrated forward in time jointly.

---

## PINN Network Design

### Architecture

- **Type:** Fully connected MLP (standard for time-domain ODE PINNs — Raissi et al. 2019)
- **Depth:** 6 hidden layers
- **Width:** 128 neurons per layer
- **Activation:** `tanh` — chosen over ReLU because `tanh` is smooth and its derivatives exist everywhere, which is required for computing physics residuals via automatic differentiation. ReLU's zero second derivative collapses the RP residual.
- **Framework:** PyTorch + DeepXDE

### Inputs (28 values)

| Input group | Features | Count |
|-------------|----------|-------|
| Tissue saturation | `peak_tissue_sat_0..15` | 16 |
| Dive profile | `max_depth_m`, `mean_ascent_rate_m_per_min`, `bottom_time_min`, `P_surface` | 4 |
| Physiological | `age`, `fitness_level`, `hydration_status`, `body_fat_pct`, `water_temp_c`, `pfo` | 6 |
| Time | `t` (normalised to [0, 1] over dive duration) | 1 |
| **Total** | | **27** |

All inputs are normalised to zero mean / unit variance using statistics computed from the 8,000 training profiles. Normalisation parameters are saved to `pinn_scaler.pkl` and applied identically at inference.

### Outputs (2 values)

- `R(t)` — bubble radius in metres at time `t`
- `dR_dt(t)` — bubble growth rate in m/s at time `t`

### Loss Function

```
L_total = λ_EP · L_EP + λ_RP · L_RP + λ_BC · L_BC
```

**Why these weights:**

`λ_EP = 0.1`, `λ_RP = 0.1`:
Physics residual weights are set to 0.1 (not 1.0) following Raissi et al. (2019) and Wang et al. (2022) "Understanding and mitigating gradient pathologies in PINNs." Setting λ_phys=1.0 causes the physics loss to dominate early training before the network has learned the approximate solution shape, producing gradient conflicts that prevent convergence. λ_phys=0.1 allows the network to first fit the boundary conditions (which have a clear learning signal), then progressively satisfy the physics. This is empirically validated across stiff ODE problems in DeepXDE's benchmark suite.

`λ_BC = 10`:
Boundary conditions are hard physical constraints — `R(0) = R₀` and `dR/dt(0) = 0` must be satisfied exactly for the solution to be physically meaningful. The 100× ratio over λ_phys (10 vs 0.1) ensures BC satisfaction before physics residuals are enforced, consistent with the curriculum learning strategy recommended by Krishnapriyan et al. (2021) for stiff coupled ODEs.

**Boundary conditions:**

```
R(0)       = R₀ = 10 μm   (initial nucleus radius — Yount & Hoffman 1986)
dR/dt(0)   = 0            (bubble initially stationary)
```

`R₀ = 10 μm` is the critical nucleus radius from Yount & Hoffman (1986) — the minimum bubble size that is thermodynamically stable and can grow under supersaturation. Smaller bubbles collapse due to surface tension; this is the physically correct initial condition for a bubble that has just nucleated.

---

## Training Strategy

### Data preparation

1. Sample 8,000 profiles from the V2 dataset as the PINN training set. **Why 8,000:** This is ~16% of the 50,000 profiles — enough to cover the joint distribution of all 27 input variables (a standard rule of thumb is ≥100 samples per input dimension for surrogate models; 8,000 >> 2,700). The remaining 42,000 profiles receive bubble features purely by inference, with no further training.

2. For each training profile, run the V2 Bühlmann simulation to extract `C_∞(t)` at 180 time points and `P_amb(t)` from the depth series. These become the time-varying forcing functions inside the physics loss.

3. Sample **1,000 collocation points** per profile uniformly in `t ∈ [0, 30 min]`. **Why 1,000:** For a 1D time-domain ODE, 1,000 points per profile gives a spatial density of ~1 point per 1.8 seconds. This is sufficient to resolve the bubble growth dynamics (characteristic timescale ~minutes) while remaining computationally tractable. Raissi et al. (2019) use 100–10,000 collocation points depending on PDE complexity; the coupled EP+RP system warrants the upper end of this range given the stiffness of RP on ascent.

### Training hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Iterations | 30,000 | Typical convergence for coupled ODE PINNs with 6-layer MLP (DeepXDE benchmarks) |
| Batch size | 256 | Fills Apple Silicon MPS memory efficiently without overflow |
| Optimiser | Adam → L-BFGS | Adam for first 25,000 iterations (robust to noisy gradients early), L-BFGS for final 5,000 (second-order convergence for fine-tuning physics residuals — standard PINN practice from Raissi 2019) |
| Learning rate | 1e-3 → 1e-5 | Cosine annealing over Adam phase |
| Device | MPS (Apple Silicon) with CPU fallback | |

### Validation

On a held-out set of 500 profiles (not used in training or inference):
- Mean EP residual < 1×10⁻⁴ (physics loss target)
- Mean RP residual < 1×10⁻⁴
- `R_max` correlation with `physics_risk_score` > 0.6 (bubble size should track nitrogen loading)
- `R(t)` monotonically non-decreasing during bottom phase (physical sanity — bubbles don't spontaneously shrink at constant depth under supersaturation)

---

## Bubble Feature Extraction

After PINN inference at 180 time points per profile, extract 6 scalar features:

| Column | Formula | Physiological meaning | Literature basis |
|--------|---------|----------------------|-----------------|
| `bubble_R_max` | `max(R(t))` | Peak bubble radius — mechanical tissue stress and vessel occlusion risk | Yount (1979): bubbles > 50 μm cause vessel occlusion |
| `bubble_R_surface` | `R(t=T_dive)` | Bubble radius at surface arrival — emboli entering pulmonary circulation | Van Liew & Raychaudhuri (1997): surface arrival radius predicts pulmonary DCS |
| `bubble_dR_dt_max` | `max(dR/dt)` | Peak growth rate — severity of rapid decompression phase | Wienke & O'Leary (2002): growth rate correlates with bubble trauma |
| `bubble_integrated_volume` | `∫R(t)³ dt` (trapezoid) | Total embolic load across full dive — cumulative nitrogen bubble burden | Van Liew & Raychaudhuri (1997): integrated volume is the strongest predictor of DCS grade |
| `bubble_n_critical` | `N₀ · exp(−R_crit / R_max)` | Estimated count of emboli-forming bubbles | Yount & Hoffman (1986) critical radius model; N₀=100 nucleation sites/mL, R_crit=12 μm |
| `bubble_velocity` | `2R²(ρ_blood − ρ_gas)g / 9μ` | Stokes drift velocity through vessel — bubble transit speed, governs where bubbles lodge | Stokes (1851); larger/faster bubbles more likely to lodge in capillary beds |

**Why `N₀ = 100` and `R_crit = 12 μm`:**
Yount (1979) and Yount & Hoffman (1986) estimate ~100 gas nucleation sites per mL of blood in a healthy adult diver. `R_crit = 12 μm` is the radius above which a bubble is large enough to avoid surface-tension-driven collapse and grow under supersaturation — derived from Yount's free phase volume model calibrated to US Navy dive table incident data. These are fixed physical parameters, not tuned values.

---

## Bubble Feature Coefficients — Calibration Procedure

The bubble coefficients cannot be pre-specified from first principles. They are derived empirically in three sequential steps after PINN inference:

### Step 1 — Empirical logistic regression fit

Normalise all 6 bubble features to zero mean / unit variance across all 50,000 profiles. Fit a logistic regression against the V2 `label` column:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_bubble = scaler.fit_transform(bubble_features)   # (50000, 6)

lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
lr.fit(X_bubble, v2_labels)
bubble_coeffs = lr.coef_[0]   # shape (6,)
```

**Why L2 regularisation with C=1.0:** The 6 bubble features are correlated (e.g. `R_max` and `bubble_n_critical` both increase with nitrogen load). L2 regularisation shrinks correlated coefficients toward equal magnitude rather than arbitrarily assigning all weight to one, producing more stable and interpretable coefficients. C=1.0 is the standard uninformative prior — no reason to deviate without cross-validation evidence.

**Why fit against V2 labels:** V2 labels already encode physiological DCS risk (they were drawn from the calibrated logistic model which was itself calibrated to DAN incidence data). The regression captures how much additional discriminative signal the bubble features add on top of the existing V2 predictors.

### Step 2 — Literature-grounded ordering check

After fitting, verify that the coefficient ordering is physiologically defensible against Van Liew & Raychaudhuri (1997) bubble load–DCS incidence curves. Expected ordering by coefficient magnitude:

```
bubble_n_critical > bubble_R_max > bubble_integrated_volume
                 > bubble_R_surface > bubble_dR_dt_max > bubble_velocity
```

**Rationale for this ordering:**
- `bubble_n_critical` ranks highest because bubble count is the primary determinant of embolic load in Van Liew & Raychaudhuri — many small bubbles cause more diffuse injury than one large bubble
- `bubble_R_max` ranks second because peak radius determines whether individual bubbles reach vessel-occluding size (Yount 1979 threshold)
- `bubble_integrated_volume` ranks third as the best single predictor of DCS grade in the Van Liew & Raychaudhuri (1997) dataset
- `bubble_R_surface` ranks fourth — relevant primarily for pulmonary DCS, a subset of cases
- `bubble_dR_dt_max` ranks fifth — growth rate is important for rapid ascent scenarios but less discriminative across the full profile distribution
- `bubble_velocity` ranks lowest — Stokes drift is a secondary mechanism, relevant mainly for neurological DCS via arterial bubble transit

If the fitted ordering violates this sequence, apply **isotonic regression** to enforce monotonic ordering while minimising L2 deviation from the fitted values. This preserves the empirical signal while ensuring physiological consistency.

### Step 3 — Intercept recalibration

After inserting the bubble coefficients into the full logistic model (V2 terms + 6 bubble terms), recalibrate the intercept to restore the three DAN/Howle calibration targets:

| `physics_risk_score` | Target `P(DCS)` | Source |
|---------------------|-----------------|--------|
| 0.70 | ~0.5% | DAN Annual Diving Report; Howle et al. (2017) |
| 0.85 | ~8.0% | DAN Annual Diving Report; Howle et al. (2017) |
| 1.00 | ~55% | Howle et al. (2017) |

The slope (18.31) is unchanged — it was calibrated to these three points and the bubble terms add independently. Only the intercept is re-fitted via least-squares using a representative diver at the three calibration score values (age=35, moderate fitness, well-hydrated, no PFO/alcohol, zero bubble features at equilibrium).

---

## Output Format

```
~/Desktop/DCS_PINN_DATASET_V3/
├── train_pinn.py                   # Step 1: train the PINN (run first, ~4h on MPS)
├── generate_dcs_dataset_v3.py      # Step 2: generate dataset (run after)
├── pinn_model.pt                   # Saved PyTorch weights
├── pinn_scaler.pkl                 # Input normalisation parameters (joblib)
├── bubble_scaler.pkl               # Bubble feature normalisation parameters
├── bubble_coeffs.json              # Fitted logistic coefficients (Step 1-3)
├── dive_profiles_timeseries.npy    # (50000, 180) float32 — same format as V2
├── dive_profiles_features.csv      # 47 columns (41 V2 + 6 bubble)
└── dive_profiles_sample.png        # 3×3 sample plot
```

### New CSV columns (appended after V2's 41)

| Column | Type | Description |
|--------|------|-------------|
| `bubble_R_max` | float (μm) | Peak bubble radius |
| `bubble_R_surface` | float (μm) | Bubble radius at surface arrival |
| `bubble_dR_dt_max` | float (μm/s) | Peak bubble growth rate |
| `bubble_integrated_volume` | float (μm³·min) | Time-integrated bubble volume |
| `bubble_n_critical` | float | Estimated count of emboli-forming bubbles |
| `bubble_velocity` | float (μm/s) | Stokes drift velocity |

---

## Validation

### Retained from V2

All 5 structural checks (row count, NaN, score bounds, label binary, DCS rate ≥ 5%) and all 6 physiological directional checks (fitness, age, hydration, PFO, alcohol, breathing gas).

### New bubble physics checks

```
=== BUBBLE PHYSICS SANITY CHECKS ===
[PASS/FAIL] bubble_R_max > R₀ (10 μm) for all profiles
[PASS/FAIL] bubble_R_max correlates with physics_risk_score (r > 0.6)
[PASS/FAIL] bubble_n_critical = 0 for profiles where R_max < R_crit (12 μm)
[PASS/FAIL] bubble_integrated_volume > 0 for all profiles
[PASS/FAIL] DCS rate increases monotonically across bubble_R_max quartiles
[PASS/FAIL] bubble feature ordering matches expected: n_critical coeff > velocity coeff
```

### Calibration targets (unchanged)

- Overall DCS rate: 5–12%
- All 6 directional checks: PASS
- Physics residuals on validation set: EP < 1×10⁻⁴, RP < 1×10⁻⁴

---

## Files to Create

| File | Purpose |
|------|---------|
| `train_pinn.py` | Loads 8,000 V2 profiles, builds DeepXDE geometry + PDE system, trains network, saves `pinn_model.pt` + `pinn_scaler.pkl` |
| `generate_dcs_dataset_v3.py` | Loads V2 outputs, runs PINN inference for all 50,000 profiles, extracts bubble features, fits bubble coefficients (Steps 1–3), generates 47-column CSV + npy |

Both scripts are self-contained. `train_pinn.py` must be run before `generate_dcs_dataset_v3.py`. Runtime: ~4h training + ~45 min generation on Apple Silicon MPS.

---

## Dependencies

```
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
tqdm >= 4.60
torch >= 2.0          # PyTorch with MPS support
deepxde >= 1.9        # PINN framework
scikit-learn >= 1.2   # Logistic regression + isotonic regression
joblib >= 1.2         # Scaler serialisation
```

---

## Literature References

- Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Wang, S., Teng, Y. & Perdikaris, P. (2022). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5).
- Krishnapriyan, A. et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS 2021*.
- Epstein, P.S. & Plesset, M.S. (1950). On the stability of gas bubbles in liquid-gas solutions. *Journal of Chemical Physics*, 18(11).
- Rayleigh, Lord (1917). On the pressure developed in a liquid during the collapse of a spherical cavity. *Philosophical Magazine*, 34, 94–98.
- Yount, D.E. (1979). Skins of varying permeability: a stabilization mechanism for gas cavitation nuclei. *Journal of the Acoustical Society of America*, 65(6).
- Yount, D.E. & Hoffman, D.C. (1986). On the use of a bubble formation model to calculate diving tables. *Aviation, Space, and Environmental Medicine*, 57(2).
- Van Liew, H.D. & Raychaudhuri, S. (1997). Stabilized bubbles in the body: pressure-radius relationships and the limits to stabilization. *Journal of Applied Physiology*, 82(6).
- Van Liew, H.D. (1991). Simulation of the dynamics of decompression sickness bubbles and the generation of new bubbles. *Undersea Biomedical Research*, 18(5–6).
- Weathersby, P.K., Homer, L.D. & Flynn, E.T. (1984). On the likelihood of decompression sickness. *Journal of Applied Physiology*, 57(3).
- Wienke, B.R. & O'Leary, T.R. (2002). Reduced gradient bubble model. *International Journal of Biomedical Computing*, 30.
- Gnanasambandam, R. et al. (2022). Self-scalable Tanh (Stan): faster convergence and better generalisation in physics-informed neural networks. *arXiv:2204.12589*.
- Howle, L.E. et al. (2017). The probability and severity of decompression sickness. *PLOS One*. doi:10.1371/journal.pone.0172665
- Merrill, E.W. et al. (1969). Rheology of human blood. *Circulation Research*.
- Charm, S.E. & Kurland, G.S. (1974). *Blood Flow and Microcirculation*. Wiley.

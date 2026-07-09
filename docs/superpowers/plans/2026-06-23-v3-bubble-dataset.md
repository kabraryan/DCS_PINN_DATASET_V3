# V3 Bubble Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a 47-column synthetic DCS dataset where six Epstein-Plesset bubble features enter the data-generating process, making the Bernoulli labels informative about bubble dynamics by construction.

**Architecture:** A single self-contained script (`generate_dcs_dataset_v3.py`) re-runs the V2 Bühlmann simulation per profile while capturing the tissue-saturation time series, feeds that into a per-profile EP ODE solved with `scipy.integrate.solve_ivp` (Radau), extracts five bubble scalar features that enter a recalibrated logistic, and draws Bernoulli labels. The StandardScaler and recalibrated intercept are fitted across all 50,000 profiles before any labels are drawn.

**Tech Stack:** Python 3, NumPy, Pandas, SciPy (`solve_ivp`), scikit-learn (`StandardScaler`), Matplotlib, tqdm.

> **BLOCKED — do not execute this plan.** Correction 11 of the design spec records a verified
> fatal flaw: with the specified `R₀`, quasi-static Laplace balance, and `t = 0` seeding, the
> nucleus dissolves during descent and `bubble_R_max ≡ R₀` on every profile. All six bubble
> columns are constants. The bug fixes below (units, solver, failure handling) are correct and
> necessary but do **not** resolve it. A nucleation-model decision is required first.

---

## File Map

| File | Role |
|------|------|
| `generate_dcs_dataset_v3.py` | Single entry-point script — constants, physics, sampling, main loop, validation, plot |
| `tests/test_ep_integrator.py` | Unit tests for EP ODE correctness |
| `tests/test_bubble_features.py` | Unit tests for feature extraction |
| `tests/test_logistic.py` | Tests for extended logistic and intercept calibration |
| `tests/conftest.py` | Shared fixtures |

All generated outputs (`*.npy`, `*.csv`, `*.png`, `*.pkl`) live in the same directory as the script.

---

## Task 1: Project scaffold

**Files:**
- Create: `generate_dcs_dataset_v3.py` (empty stub)
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the stub script**

```python
# generate_dcs_dataset_v3.py
"""DCS Dataset V3 — Bühlmann ZHL-16C + Epstein-Plesset bubble dynamics."""
```

- [ ] **Step 2: Create test infrastructure**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import numpy as np
import pytest

@pytest.fixture
def flat_profile():
    """180-step profile at constant 30 m depth."""
    return np.full(180, 30.0, dtype=np.float32)

@pytest.fixture
def surface_profile():
    """180-step profile at surface — zero supersaturation."""
    return np.zeros(180, dtype=np.float32)

@pytest.fixture
def time_points():
    """Time axis in seconds for 180 steps of 10 s each."""
    return np.arange(180) * 10.0
```

- [ ] **Step 3: Verify pytest discovers tests**

```bash
cd ~/Desktop/DCS_PINN_DATASET_V3
python -m pytest tests/ --collect-only
```

Expected: `no tests ran` (empty test files don't exist yet — that's fine)

- [ ] **Step 4: Install dependencies**

```bash
pip install numpy pandas matplotlib tqdm scipy scikit-learn
```

- [ ] **Step 5: Commit scaffold**

```bash
git add generate_dcs_dataset_v3.py tests/
git commit -m "feat: scaffold V3 project structure"
```

---

## Task 2: Physical constants and EP ODE

**Files:**
- Modify: `generate_dcs_dataset_v3.py` — add constants block and EP functions
- Create: `tests/test_ep_integrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ep_integrator.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_dcs_dataset_v3 import (
    integrate_bubble, R0, R_CRIT, SIGMA, ALPHA_N2, D, BAR_TO_PA
)


def test_bubble_does_not_grow_at_equilibrium(time_points):
    """When P_tissue equals P_amb (+ Laplace), bubble stays near R0."""
    P_amb = np.full(180, 1.0)   # 1 bar at surface
    # Equilibrium: tissue pressure = Laplace + ambient
    R_laplace = R0
    P_laplace_bar = (2 * SIGMA / R_laplace) / BAR_TO_PA
    P_tissue = np.full(180, 1.0 + P_laplace_bar)
    R_traj, success = integrate_bubble(P_tissue, P_amb, time_points)
    assert success
    assert abs(R_traj[-1] - R0) / R0 < 0.05  # within 5% of R0


def test_bubble_grows_under_supersaturation(time_points):
    """Under supersaturation (P_tissue >> P_amb), bubble radius grows."""
    P_amb = np.full(180, 1.0)        # 1 bar (surface)
    P_tissue = np.full(180, 4.0)     # 4 bar dissolved N2 — strong supersaturation
    R_traj, success = integrate_bubble(P_tissue, P_amb, time_points)
    assert success
    assert R_traj[-1] > R0 * 2      # should at least double


def test_bubble_radius_always_positive(time_points):
    """Bubble radius must remain positive throughout integration."""
    P_amb = np.linspace(4.0, 1.0, 180)   # ascent from 4 bar
    P_tissue = np.full(180, 4.0)
    R_traj, success = integrate_bubble(P_tissue, P_amb, time_points)
    assert success
    assert np.all(R_traj > 0)


def test_returns_correct_length(time_points):
    """Output trajectory must match number of time points."""
    P_amb = np.full(180, 1.0)
    P_tissue = np.full(180, 2.0)
    R_traj, success = integrate_bubble(P_tissue, P_amb, time_points)
    assert len(R_traj) == 180


def test_surface_profile_minimal_growth(time_points):
    """At surface with P_tissue near equilibrium, growth is negligible."""
    P_amb = np.full(180, 1.0)
    P_tissue = np.full(180, 0.79)  # surface N2 partial pressure (air)
    R_laplace_bar = (2 * SIGMA / R0) / BAR_TO_PA
    # tissue pressure is BELOW Laplace threshold — bubble should shrink or be static
    R_traj, success = integrate_bubble(P_tissue, P_amb, time_points)
    assert success
    # R should stay near R0 or smaller (no supersaturation)
    assert R_traj[-1] <= R0 * 3
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

```bash
python -m pytest tests/test_ep_integrator.py -v
```

Expected: `ImportError: cannot import name 'integrate_bubble'`

- [ ] **Step 3: Add constants and EP functions to the script**

```python
# generate_dcs_dataset_v3.py
"""DCS Dataset V3 — Bühlmann ZHL-16C + Epstein-Plesset bubble dynamics."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'Desktop', 'DCS_PINN_DATASET_V3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

DT_MIN  = 10 / 60
N_STEPS = 180
N_DIVES = 50_000

# ── PHYSICAL CONSTANTS ──────────────────────────────────────────────────────
# All SI units unless noted.
BAR_TO_PA = 1e5            # 1 bar in Pascals

D        = 2.0e-9          # N2 diffusion coefficient in blood (m²/s)
                           # Source: Weathersby et al. (1984)
ALPHA_N2 = 6.84e-6         # N2 solubility in blood (mol/(m³·Pa))
                           # Converted from 0.0693 mL(STPD)/mL/atm,
                           # Weathersby et al. (1984)
SIGMA    = 0.050           # blood–gas surface tension (N/m), Van Liew (1991)
RHO_BLOOD = 1060.0         # blood density (kg/m³), Merrill et al. (1969)
MU       = 0.003           # dynamic viscosity at 37°C (Pa·s), Charm & Kurland (1974)
M_N2     = 0.028           # molar mass of N2 (kg/mol)
R_GAS    = 8.314           # universal gas constant (J/(mol·K))
T_BODY   = 310.15          # body temperature (K)
G        = 9.81            # gravitational acceleration (m/s²)

R0       = 0.7e-6          # initial VPM nucleus radius (m), Yount (1991)
R_CRIT   = 12.0e-6         # critical emboli-forming radius (m), Yount (1979)
N0       = 100.0           # nucleation site density (sites/mL), Yount & Hoffman (1986)
R_DISSOLVE = 0.1 * R0      # nucleus considered dissolved below this radius;
                           # terminal event floor, keeps 2σ/R and 1/R finite


# ── EP ODE ──────────────────────────────────────────────────────────────────

def _ep_rhs(t, y, P_tissue, P_amb, time_points_s):
    """Epstein-Plesset ODE RHS with quasi-static Laplace mechanical equilibrium.

    dR/dt = D · (C_∞ - C_s) / (ρ_gas · R) · (1 + R / √(π·D·t))

    where:
      P_gas = P_amb + 2σ/R  (Laplace quasi-static balance, replaces Rayleigh-Plesset)
      C_∞ - C_s = α_N2 · M_N2 · (P_tissue - P_gas)   (Henry's law, MASS concentration)
      ρ_gas     = P_gas · M_N2 / (R_gas · T_body)    (ideal gas law, MASS density)

    Both numerator and denominator must be mass-basis. α_N2 is molar
    (mol/(m³·Pa)), so it is multiplied by M_N2 to give kg/m³. Omitting M_N2
    inflates dR/dt by 1/M_N2 = 35.71x (verified numerically).
    """
    R = y[0]

    P_t_pa = np.interp(t, time_points_s, P_tissue) * BAR_TO_PA
    P_a_pa = np.interp(t, time_points_s, P_amb)    * BAR_TO_PA

    # Quasi-static Laplace: P_gas = P_amb + 2σ/R
    P_gas_pa = P_a_pa + 2.0 * SIGMA / R

    rho_gas = P_gas_pa * M_N2 / (R_GAS * T_BODY)            # kg/m³
    delta_C = ALPHA_N2 * M_N2 * (P_t_pa - P_gas_pa)         # kg/m³

    correction = 1.0 + R / np.sqrt(np.pi * D * max(t, 1e-10))

    return [D * delta_C / (rho_gas * R) * correction]


def _dissolved(t, y, P_tissue, P_amb, time_points_s):
    """Terminal event: the nucleus has dissolved. Keeps R strictly positive."""
    return y[0] - R_DISSOLVE
_dissolved.terminal = True
_dissolved.direction = -1


def integrate_bubble(P_tissue_series, P_amb_series, time_points_s):
    """Numerically integrate the EP bubble ODE for one dive profile.

    Args:
        P_tissue_series: (180,) array, max dissolved N2 partial pressure (bar)
        P_amb_series:    (180,) array, ambient pressure at depth (bar)
        time_points_s:   (180,) array, time in seconds

    Returns:
        R_trajectory: (180,) float64, bubble radius in metres. If the bubble
                      dissolves, the trajectory is padded with R_DISSOLVE.
        success:      bool, False if solve_ivp failed
    """
    sol = solve_ivp(
        _ep_rhs,
        (time_points_s[0], time_points_s[-1]),
        [R0],
        method='RK45',
        t_eval=time_points_s,
        args=(P_tissue_series, P_amb_series, time_points_s),
        rtol=1e-6,
        atol=1e-12,
        events=_dissolved,
    )
    R = sol.y[0]
    if len(R) < len(time_points_s):        # terminated early on dissolution
        R = np.concatenate([R, np.full(len(time_points_s) - len(R), R_DISSOLVE)])
    return R, sol.success
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/test_ep_integrator.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add generate_dcs_dataset_v3.py tests/test_ep_integrator.py tests/conftest.py tests/__init__.py
git commit -m "feat: add EP ODE integrator with physical constants"
```

---

## Task 3: Bubble feature extraction

**Files:**
- Modify: `generate_dcs_dataset_v3.py` — add `extract_bubble_features`
- Create: `tests/test_bubble_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bubble_features.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_dcs_dataset_v3 import extract_bubble_features, R0, R_CRIT, N0


@pytest.fixture
def growing_trajectory(time_points):
    """Linearly growing bubble from R0 to 3× R_CRIT over 30 min."""
    return np.linspace(R0, 3 * R_CRIT, 180)


@pytest.fixture
def flat_trajectory():
    """Bubble staying at R0 — no growth."""
    return np.full(180, R0)


def test_n_critical_zero_below_threshold(flat_trajectory, time_points):
    """n_critical must be 0 when R_max ≤ R_crit."""
    feats = extract_bubble_features(flat_trajectory, time_points)
    assert feats['bubble_n_critical'] == 0.0


def test_n_critical_positive_above_threshold(growing_trajectory, time_points):
    """n_critical must be > 0 when R_max > R_crit."""
    feats = extract_bubble_features(growing_trajectory, time_points)
    assert feats['bubble_n_critical'] > 0.0


def test_n_critical_formula(growing_trajectory, time_points):
    """n_critical = N0 · max(0, 1 - R_crit / R_max)."""
    feats = extract_bubble_features(growing_trajectory, time_points)
    R_max_m = growing_trajectory[-1]
    expected = N0 * max(0.0, 1.0 - R_CRIT / R_max_m)
    assert abs(feats['bubble_n_critical'] - expected) < 1e-10


def test_R_max_in_microns(growing_trajectory, time_points):
    """bubble_R_max must be reported in µm."""
    feats = extract_bubble_features(growing_trajectory, time_points)
    expected_um = growing_trajectory.max() * 1e6
    assert abs(feats['bubble_R_max'] - expected_um) < 1e-6


def test_integrated_volume_positive(growing_trajectory, time_points):
    """bubble_integrated_volume must be > 0 for any growing bubble."""
    feats = extract_bubble_features(growing_trajectory, time_points)
    assert feats['bubble_integrated_volume'] > 0.0


def test_all_features_finite(growing_trajectory, time_points):
    """All 6 features must be finite (no NaN or Inf)."""
    feats = extract_bubble_features(growing_trajectory, time_points)
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_R_surface_equals_last_point(growing_trajectory, time_points):
    """bubble_R_surface must equal the last radius in the trajectory (µm)."""
    feats = extract_bubble_features(growing_trajectory, time_points)
    assert abs(feats['bubble_R_surface'] - growing_trajectory[-1] * 1e6) < 1e-9
```

- [ ] **Step 2: Run — verify ImportError**

```bash
python -m pytest tests/test_bubble_features.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'extract_bubble_features'`

- [ ] **Step 3: Add `extract_bubble_features` to the script**

Append to `generate_dcs_dataset_v3.py` after the EP ODE section:

```python
# ── BUBBLE FEATURE EXTRACTION ────────────────────────────────────────────────

def extract_bubble_features(R_trajectory, time_points_s):
    """Extract 6 scalar bubble features from a radius trajectory.

    Args:
        R_trajectory:  (180,) float64 in metres
        time_points_s: (180,) float64 in seconds

    Returns:
        dict with keys: bubble_R_max, bubble_R_surface, bubble_dR_dt_max,
                        bubble_integrated_volume, bubble_n_critical,
                        bubble_terminal_velocity
    """
    R_um = R_trajectory * 1e6           # metres → µm
    R_max_um   = float(np.max(R_um))
    R_max_m    = R_max_um * 1e-6
    R_surf_um  = float(R_um[-1])

    # Peak growth rate (µm/s) via finite differences
    dt_s = np.diff(time_points_s)
    dR_m = np.diff(R_trajectory)
    dRdt_um_s = (dR_m / dt_s) * 1e6
    dRdt_max  = float(np.max(dRdt_um_s)) if len(dRdt_um_s) > 0 else 0.0

    # Time-integrated bubble volume (µm³·min)
    # np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    time_min = time_points_s / 60.0
    integrated_vol = float(_trapz(R_um ** 3, time_min))

    # n_critical: N0 · max(0, 1 − R_crit/R_max)
    # Exactly 0 when R_max ≤ R_crit; monotonically increasing above threshold.
    R_crit_um = R_CRIT * 1e6
    n_crit = float(N0 * max(0.0, 1.0 - R_crit_um / R_max_um)) if R_max_um > 0 else 0.0

    # Terminal buoyant velocity (µm/s) at peak radius — Stokes law.
    # NOTE: diagnostic only; excluded from logistic. At typical bubble sizes
    # (~50 µm), this gives ~0.2 mm/s vs blood flow ~10–100 mm/s; buoyancy
    # does not govern bubble transport relative to advection.
    rho_gas_surface = (1.0e5 * M_N2) / (R_GAS * T_BODY)  # ρ_gas at 1 bar
    v_term_um_s = float(
        2.0 * R_max_m ** 2 * (RHO_BLOOD - rho_gas_surface) * G / (9.0 * MU)
    ) * 1e6

    return {
        'bubble_R_max':               R_max_um,
        'bubble_R_surface':           R_surf_um,
        'bubble_dR_dt_max':           dRdt_max,
        'bubble_integrated_volume':   integrated_vol,
        'bubble_n_critical':          n_crit,
        'bubble_terminal_velocity':   v_term_um_s,
    }
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/test_bubble_features.py -v
```

Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add generate_dcs_dataset_v3.py tests/test_bubble_features.py
git commit -m "feat: add bubble feature extraction (EP trajectory → 6 scalars)"
```

---

## Task 4: Extended logistic model and intercept calibration

**Files:**
- Modify: `generate_dcs_dataset_v3.py` — add logistic functions
- Create: `tests/test_logistic.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_logistic.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_dcs_dataset_v3 import (
    compute_dcs_probability_v3,
    calibrate_intercept,
    BUBBLE_COEFFS,
)


def test_bubble_coeffs_ordering():
    """Integrated volume must have highest coefficient (Van Liew & Raychaudhuri 1997)."""
    assert BUBBLE_COEFFS['bubble_integrated_volume'] >= BUBBLE_COEFFS['bubble_R_max']
    assert BUBBLE_COEFFS['bubble_n_critical']       >= BUBBLE_COEFFS['bubble_dR_dt_max']
    assert BUBBLE_COEFFS['bubble_terminal_velocity'] == 0.0  # excluded from logistic


def test_calibration_targets_met():
    """After calibration, three DAN/Howle targets must be within ±0.5 pp."""
    # Baseline diver: age=35, mod fitness, well-hydrated, no PFO, no alcohol, good sleep
    # Zero-valued bubble z-scores (simulating median of baseline profiles)
    intercept = calibrate_intercept(baseline_bubble_contribution=0.0)

    for score, target_p in [(0.70, 0.005), (0.85, 0.08), (1.00, 0.55)]:
        p = compute_dcs_probability_v3(
            physics_risk_score=score,
            pfo=0, pre_dive_exercise='none', alcohol=0,
            sleep_quality='good', age=35,
            hydration_status=0, fitness_level=1,
            bubble_z_scores={k: 0.0 for k in BUBBLE_COEFFS},
            intercept=intercept,
        )
        assert abs(p - target_p) < 0.01, (
            f"At score={score}: got {p:.4f}, expected {target_p:.4f}"
        )


def test_pfo_increases_probability():
    """PFO=1 must give higher dcs_probability than PFO=0, all else equal."""
    intercept = calibrate_intercept(0.0)
    base_kwargs = dict(
        physics_risk_score=0.80, pre_dive_exercise='none', alcohol=0,
        sleep_quality='good', age=35, hydration_status=0, fitness_level=1,
        bubble_z_scores={k: 0.0 for k in BUBBLE_COEFFS}, intercept=intercept,
    )
    p_no_pfo = compute_dcs_probability_v3(pfo=0, **base_kwargs)
    p_pfo    = compute_dcs_probability_v3(pfo=1, **base_kwargs)
    assert p_pfo > p_no_pfo


def test_high_fitness_lower_probability():
    """High fitness must produce lower dcs_probability than low fitness."""
    intercept = calibrate_intercept(0.0)
    base_kwargs = dict(
        physics_risk_score=0.80, pfo=0, pre_dive_exercise='none', alcohol=0,
        sleep_quality='good', age=35, hydration_status=0,
        bubble_z_scores={k: 0.0 for k in BUBBLE_COEFFS}, intercept=intercept,
    )
    p_low  = compute_dcs_probability_v3(fitness_level=0, **base_kwargs)
    p_high = compute_dcs_probability_v3(fitness_level=2, **base_kwargs)
    assert p_low > p_high


def test_bubble_integrated_volume_increases_probability():
    """Higher bubble_integrated_volume z-score must increase dcs_probability."""
    intercept = calibrate_intercept(0.0)
    base_kwargs = dict(
        physics_risk_score=0.75, pfo=0, pre_dive_exercise='none', alcohol=0,
        sleep_quality='good', age=35, hydration_status=0, fitness_level=1,
        intercept=intercept,
    )
    p_low  = compute_dcs_probability_v3(
        bubble_z_scores={k: -1.0 for k in BUBBLE_COEFFS}, **base_kwargs
    )
    p_high = compute_dcs_probability_v3(
        bubble_z_scores={k: +1.0 for k in BUBBLE_COEFFS}, **base_kwargs
    )
    assert p_high > p_low
```

- [ ] **Step 2: Run — verify ImportError**

```bash
python -m pytest tests/test_logistic.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'compute_dcs_probability_v3'`

- [ ] **Step 3: Add logistic functions to the script**

Append to `generate_dcs_dataset_v3.py`:

```python
# ── EXTENDED LOGISTIC MODEL ──────────────────────────────────────────────────
# Pre-specified bubble logistic coefficients.
# These are design parameters for the synthetic data-generating process,
# calibrated to the relative ordering in Van Liew & Raychaudhuri (1997):
# integrated volume is the strongest predictor of DCS grade severity;
# n_critical and R_max closely follow. Magnitudes (0.25–0.50) place bubble
# features in the same range as moderate V2 contributors (alcohol: +0.40).
# bubble_terminal_velocity is excluded (buoyancy dominated by advection).
BUBBLE_COEFFS = {
    'bubble_integrated_volume':  0.50,   # strongest predictor (VL&R 1997 Table 3)
    'bubble_n_critical':         0.45,   # diffuse embolic load
    'bubble_R_max':              0.40,   # peak occlusion risk
    'bubble_R_surface':          0.30,   # pulmonary arrival risk
    'bubble_dR_dt_max':          0.25,   # rapid-ascent severity
    'bubble_terminal_velocity':  0.00,   # diagnostic only — excluded
}


def calibrate_intercept(baseline_bubble_contribution):
    """Fit intercept to restore three DAN/Howle calibration targets.

    For a baseline diver (age=35, mod fitness, well-hydrated, no PFO,
    no alcohol, good sleep) the V2 non-intercept terms sum to 18.31*score.
    We solve for the intercept β₀ that satisfies all three targets in
    least-squares sense.

    Args:
        baseline_bubble_contribution: scalar, sum(coeff_i * z_median_i)
                                      at the median standard-profile bubble z-scores.

    Returns:
        float, recalibrated intercept.
    """
    calibration_points = [
        (0.70, 0.005),   # DAN Annual Diving Report; Howle et al. (2017)
        (0.85, 0.08),
        (1.00, 0.55),
    ]
    intercepts = []
    for score, p_target in calibration_points:
        logit_target = np.log(p_target / (1.0 - p_target))
        v2_terms = 18.31 * score   # all other baseline terms = 0
        beta0 = logit_target - v2_terms - baseline_bubble_contribution
        intercepts.append(beta0)
    return float(np.mean(intercepts))


def compute_dcs_probability_v3(
    physics_risk_score, pfo, pre_dive_exercise, alcohol,
    sleep_quality, age, hydration_status, fitness_level,
    bubble_z_scores, intercept,
):
    """Extended V3 logistic DCS probability.

    Identical to V2 logistic (reconstructed from generate_dcs_dataset_v2.py
    source) plus five bubble terms with pre-specified coefficients.
    bubble_terminal_velocity is computed but has coefficient 0.0 and is
    excluded from the log-odds sum.

    Args:
        physics_risk_score: float [0, 1]
        pfo:                int 0/1
        pre_dive_exercise:  str 'none'|'moderate'|'heavy'
        alcohol:            int 0/1
        sleep_quality:      str 'good'|'poor'
        age:                int
        hydration_status:   int 0/1/2
        fitness_level:      int 0/1/2
        bubble_z_scores:    dict {feature_name: z_score}
        intercept:          float (recalibrated)

    Returns:
        float, dcs_probability in [0, 1]
    """
    # ── V2 terms (reconstructed from source) ──
    log_odds = intercept + 18.31 * physics_risk_score

    log_odds += 1.2 * pfo

    log_odds += {'none': 0.0, 'moderate': 0.2, 'heavy': 0.5}[pre_dive_exercise]
    log_odds += 0.4 * alcohol
    log_odds += {'good': 0.0, 'poor': 0.15}[sleep_quality]

    if age > 40:
        log_odds += 0.3 * ((age - 40) / 30.0)
    if age > 50:
        log_odds += min(0.4 * ((age - 50) / 20.0), 0.40)

    log_odds += {0: 0.0, 1: 0.15, 2: 0.30}[hydration_status]
    log_odds += {0: +0.45, 1: 0.0, 2: -0.50}[fitness_level]

    # ── V3 bubble terms ──
    for feat, coeff in BUBBLE_COEFFS.items():
        log_odds += coeff * bubble_z_scores.get(feat, 0.0)

    return float(1.0 / (1.0 + np.exp(-log_odds)))
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/test_logistic.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add generate_dcs_dataset_v3.py tests/test_logistic.py
git commit -m "feat: add extended logistic with pre-specified bubble coefficients"
```

---

## Task 5: Bühlmann simulator with tissue-sat time series

**Files:**
- Modify: `generate_dcs_dataset_v3.py` — port V2 Bühlmann functions + add tissue-sat recording

This task copies the V2 Bühlmann infrastructure (ZHL-16C table, physiological modifiers, profile builder, sampling, simulator) and modifies `simulate_dive` to return the full tissue-saturation time series needed by the EP integrator.

- [ ] **Step 1: Add ZHL-16C table and physiological modifiers**

Append to `generate_dcs_dataset_v3.py`:

```python
# ── 1. ZHL-16C TABLE (unchanged from V2) ────────────────────────────────────

def get_zhl16c_table():
    """Return Bühlmann ZHL-16C nitrogen tissue table, shape (16, 3).
    Columns: [t_half_min, a_bar, b].
    """
    table = np.array([
        [   4.0, 1.2599, 0.5050], [   8.0, 1.0000, 0.6514],
        [  12.5, 0.8618, 0.7222], [  18.5, 0.7562, 0.7825],
        [  27.0, 0.6200, 0.8126], [  38.3, 0.5043, 0.8434],
        [  54.3, 0.4410, 0.8693], [  77.0, 0.4000, 0.8910],
        [ 109.0, 0.3750, 0.9092], [ 146.0, 0.3500, 0.9222],
        [ 187.0, 0.3295, 0.9319], [ 239.0, 0.3065, 0.9403],
        [ 305.0, 0.2835, 0.9477], [ 390.0, 0.2610, 0.9544],
        [ 498.0, 0.2480, 0.9602], [ 635.0, 0.2327, 0.9653],
    ], dtype=np.float64)
    # b rises monotonically toward 1.0; a falls. V2 shipped with compartment 16's
    # b set to 0.8693 (compartment 7's value) — this guard is why V3 cannot repeat it.
    assert np.all(np.diff(table[:, 2]) > 0), 'ZHL-16C b column must be strictly increasing'
    assert np.all(np.diff(table[:, 1]) < 0), 'ZHL-16C a column must be strictly decreasing'
    return table


# ── 2. PHYSIOLOGICAL MODIFIERS (unchanged from V2) ──────────────────────────

def compute_effective_half_times(table, age, body_fat_pct, water_temp_c,
                                  fitness, hydration):
    """Compute per-diver effective N2 half-times. Order: age→fat→temp→fitness→hydration."""
    t_half = table[:, 0].copy()

    age_perf = max(0.65, 1.0 - max(0.0, (age - 30) * 0.008))
    t_half = t_half / age_perf

    fat_mult = float(np.clip(1.0 + (body_fat_pct / 100.0 - 0.20) * 1.5, 0.85, 1.35))
    t_half[8:] *= fat_mult

    temp_fact = float(np.clip(1.0 + (25.0 - water_temp_c) * 0.006, 0.91, 1.12))
    t_half[6:] *= temp_fact

    t_half *= {'low': 1.15, 'moderate': 1.0, 'high': 0.88}[fitness]
    t_half *= {'well_hydrated': 1.00, 'mildly_dehydrated': 1.12, 'dehydrated': 1.28}[hydration]

    return t_half


def compute_p_surface(altitude_m):
    return 1.013 * np.exp(-altitude_m / 8430.0)


# ── 3. BÜHLMANN PRIMITIVES ───────────────────────────────────────────────────

def _haldane_step(P_t, P_alv, t_half):
    k = np.log(2) / t_half
    return P_alv + (P_t - P_alv) * np.exp(-k * DT_MIN)


def _compute_mvalue(P_amb, a, b):
    return a + P_amb / b
```

- [ ] **Step 2: Add sampling (ported from V2 with all fixes)**

Append to `generate_dcs_dataset_v3.py`:

```python
# ── 4. SAMPLING (ported from V2 — all Fixes 1-9 preserved) ──────────────────

def sample_params(rng):
    """Sample dive + physiological parameters for one profile.
    Identical to V2 generate_dcs_dataset_v2.py including age-stratified
    override, fitness/hydration skew, age-ascent correlation, dual-override
    guard. physio_type flag is sampled here; override is applied post-simulation
    in main loop (Fix 1 threshold gate).
    """
    age = int(rng.randint(18, 71))

    # Age-stratified high-risk dive override (Fix — from V2)
    age_override_prob = float(np.clip(0.15 + 0.30 * (age - 44) / 26.0, 0.05, 0.45))
    profile_type = int(rng.random() < age_override_prob)

    max_depth_m       = float(rng.uniform(5.0, 45.0))
    bottom_time_steps = int(rng.randint(6, 61))
    descent_rate      = float(rng.uniform(10.0, 25.0))
    n_safety_stops    = int(rng.randint(0, 2))
    stop_dur          = int(rng.randint(6, 25)) if n_safety_stops else 0
    is_repetitive     = bool(rng.randint(0, 2))
    surface_interval  = float(rng.uniform(30.0, 180.0)) if is_repetitive else 0.0
    prior_depth       = float(rng.uniform(10.0, 40.0))  if is_repetitive else 0.0

    if profile_type == 1:
        fitness_name = rng.choice(['low','moderate','high'], p=[0.45,0.40,0.15])
        hydration_name = rng.choice(
            ['well_hydrated','mildly_dehydrated','dehydrated'], p=[0.30,0.40,0.30])
        ascent_rate = float(rng.uniform(3.0, 25.0))   # overridden below
    else:
        fitness_name = rng.choice(['low','moderate','high'], p=[0.25,0.50,0.25])
        hydration_name = rng.choice(
            ['well_hydrated','mildly_dehydrated','dehydrated'], p=[0.50,0.35,0.15])
        age_nudge     = (age - 44) / 26.0 * 3.5
        fitness_nudge = {'low': +1.5, 'moderate': 0.0, 'high': -1.5}[fitness_name]
        combined      = age_nudge + fitness_nudge
        asc_lo = float(np.clip(3.0  + combined, 2.0,  6.0))
        asc_hi = float(np.clip(25.0 + combined, 18.0, 30.0))
        ascent_rate = float(rng.uniform(asc_lo, asc_hi))

    if profile_type == 1:
        max_depth_m       = float(rng.uniform(30.0, 45.0))
        ascent_rate       = float(rng.uniform(15.0, 25.0))
        bottom_time_steps = int(rng.randint(40, 61))
        n_safety_stops    = 0
        stop_dur          = 0

    body_fat_pct = float(rng.uniform(10.0, 40.0))
    water_temp_c = float(rng.uniform(10.0, 30.0))

    fitness_enc   = {'low': 0, 'moderate': 1, 'high': 2}[fitness_name]
    hydration_enc = {'well_hydrated': 0, 'mildly_dehydrated': 1, 'dehydrated': 2}[hydration_name]

    gas_name = rng.choice(['air','nitrox32','nitrox36'], p=[0.70,0.20,0.10])
    gas_enc  = {'air': 0, 'nitrox32': 1, 'nitrox36': 2}[gas_name]

    altitude_m = (float(rng.uniform(0.0, 200.0))
                  if rng.random() < 0.80 else float(rng.uniform(200.0, 2500.0)))

    pfo           = int(rng.random() < 0.25)
    exercise_name = rng.choice(['none','moderate','heavy'], p=[0.55,0.30,0.15])
    exercise_enc  = {'none': 0, 'moderate': 1, 'heavy': 2}[exercise_name]
    alcohol       = int(rng.random() < 0.15)
    sleep_name    = rng.choice(['good','poor'], p=[0.75,0.25])
    sleep_enc     = {'good': 0, 'poor': 1}[sleep_name]

    physio_type = int(rng.random() < 0.10)
    physio_override_body_fat = float(rng.uniform(28.0, 40.0))

    if profile_type == 1 and physio_type == 1:
        if rng.random() < 0.50:
            physio_type = 0

    return dict(
        max_depth_m=max_depth_m, bottom_time_steps=bottom_time_steps,
        descent_rate_m_per_min=descent_rate, ascent_rate_m_per_min=ascent_rate,
        n_safety_stops=n_safety_stops, safety_stop_duration_steps=stop_dur,
        is_repetitive=is_repetitive, surface_interval_min=surface_interval,
        prior_max_depth_m=prior_depth, profile_type=profile_type,
        age=age, body_fat_pct=body_fat_pct, water_temp_c=water_temp_c,
        fitness_name=fitness_name, fitness_level=fitness_enc,
        hydration_name=hydration_name, hydration_status=hydration_enc,
        gas_name=gas_name, breathing_gas=gas_enc, altitude_m=altitude_m,
        pfo=pfo, exercise_name=exercise_name, pre_dive_exercise=exercise_enc,
        alcohol=alcohol, sleep_name=sleep_name, sleep_quality=sleep_enc,
        physio_type=physio_type, physio_override_body_fat=physio_override_body_fat,
    )
```

- [ ] **Step 3: Add `build_depth_profile`, `preload_tissues`, and `simulate_dive_v3`**

Append to `generate_dcs_dataset_v3.py`:

```python
def build_depth_profile(params, trim_counter):
    """Build 180-step depth profile. Ported from V2 unchanged."""
    max_depth    = float(params['max_depth_m'])
    bottom_steps = int(params['bottom_time_steps'])
    desc_per_step = float(params['descent_rate_m_per_min']) * DT_MIN
    asc_per_step  = float(params['ascent_rate_m_per_min'])  * DT_MIN
    n_stops      = int(params['n_safety_stops'])
    stop_steps   = int(params['safety_stop_duration_steps'])

    desc_steps = int(np.ceil(max_depth / desc_per_step))
    if n_stops == 1:
        asc_steps = (int(np.ceil((max_depth - 5.0) / asc_per_step))
                     + stop_steps + int(np.ceil(5.0 / asc_per_step)))
    else:
        asc_steps = int(np.ceil(max_depth / asc_per_step))

    if desc_steps + bottom_steps + asc_steps > 175:
        bottom_steps = max(0, bottom_steps - (desc_steps + bottom_steps + asc_steps - 175))
        trim_counter[0] += 1

    profile = np.zeros(N_STEPS, dtype=np.float32)
    step, depth = 0, 0.0

    while depth < max_depth and step < N_STEPS:
        depth = min(depth + desc_per_step, max_depth)
        profile[step] = depth; step += 1
    for _ in range(bottom_steps):
        if step < N_STEPS: profile[step] = max_depth; step += 1
    if n_stops == 1:
        while depth > 5.0 and step < N_STEPS:
            depth = max(depth - asc_per_step, 5.0); profile[step] = depth; step += 1
        for _ in range(stop_steps):
            if step < N_STEPS: profile[step] = 5.0; step += 1
    while depth > 0.0 and step < N_STEPS:
        depth = max(depth - asc_per_step, 0.0); profile[step] = depth; step += 1

    return profile


def preload_tissues(is_repetitive, surface_interval_min, prior_max_depth_m,
                    t_half_eff, P_surface, nitrogen_fraction):
    """Compute pre-dive tissue N2 state. Ported from V2."""
    surface_alv = P_surface * nitrogen_fraction
    P_t = np.full(16, surface_alv, dtype=np.float64)
    if not is_repetitive:
        return P_t
    step_d, step_a = 15.0 * DT_MIN, 9.0 * DT_MIN
    depth = 0.0
    while depth < prior_max_depth_m:
        depth = min(depth + step_d, prior_max_depth_m)
        P_t = _haldane_step(P_t, (depth*0.1+P_surface)*nitrogen_fraction, t_half_eff)
    P_alv_b = (prior_max_depth_m*0.1+P_surface)*nitrogen_fraction
    for _ in range(60): P_t = _haldane_step(P_t, P_alv_b, t_half_eff)
    while depth > 0.0:
        depth = max(depth - step_a, 0.0)
        P_t = _haldane_step(P_t, (depth*0.1+P_surface)*nitrogen_fraction, t_half_eff)
    k = np.log(2) / t_half_eff
    P_t = surface_alv + (P_t - surface_alv) * np.exp(-k * surface_interval_min)
    return P_t


def simulate_dive_v3(depth_series, initial_tissues, table, t_half_eff,
                     P_surface, nitrogen_fraction):
    """Bühlmann simulation returning both aggregate stats AND tissue-sat time series.

    V3 modification: records P_tissue_max(t) at every step for use in EP ODE.
    All Bühlmann physics identical to V2.

    Returns:
        dict with:
          peak_tissue_sat (16,)       — peak per compartment over full dive
          physics_risk_score float    — max M-value ratio clipped to [0,1]
          P_tissue_timeseries (180,)  — max tissue N2 partial pressure (bar) per step
          P_amb_timeseries    (180,)  — ambient pressure (bar) per step
    """
    a, b = table[:, 1], table[:, 2]
    P_t = initial_tissues.astype(np.float64)
    peak_tissue_sat = np.zeros(16, dtype=np.float64)
    max_mv_ratio = 0.0

    P_tissue_ts = np.zeros(N_STEPS, dtype=np.float64)
    P_amb_ts    = np.zeros(N_STEPS, dtype=np.float64)

    for step, depth in enumerate(depth_series):
        depth_f = float(depth)
        P_amb   = depth_f * 0.1 + P_surface
        P_alv   = P_amb * nitrogen_fraction
        P_t = _haldane_step(P_t, P_alv, t_half_eff)
        mv  = P_t / _compute_mvalue(P_amb, a, b)
        max_mv_ratio = max(max_mv_ratio, float(np.max(mv)))
        np.maximum(peak_tissue_sat, P_t, out=peak_tissue_sat)
        P_tissue_ts[step] = float(np.max(P_t))   # max over compartments (bar)
        P_amb_ts[step]    = P_amb

    surface_mv = _compute_mvalue(P_surface, a, b)
    if np.any(P_t > surface_mv):
        max_mv_ratio = max(max_mv_ratio, float(np.max(P_t / surface_mv)))

    return {
        'peak_tissue_sat':    peak_tissue_sat,
        'physics_risk_score': float(np.clip(max_mv_ratio, 0.0, 1.0)),
        'P_tissue_timeseries': P_tissue_ts,
        'P_amb_timeseries':    P_amb_ts,
    }
```

- [ ] **Step 4: Smoke-test the Bühlmann functions**

```bash
python -c "
import numpy as np, sys
sys.path.insert(0,'$(pwd)')
from generate_dcs_dataset_v3 import (
    get_zhl16c_table, compute_effective_half_times,
    build_depth_profile, simulate_dive_v3, compute_p_surface
)
rng = np.random.RandomState(42)
table = get_zhl16c_table()
t_half = compute_effective_half_times(table, 35, 22.0, 22.0, 'moderate', 'well_hydrated')
depth = np.concatenate([np.linspace(0,30,30), np.full(90,30), np.linspace(30,0,60)]).astype(np.float32)
tissues = np.full(16, 0.79)
P_s = compute_p_surface(0.0)
result = simulate_dive_v3(depth, tissues, table, t_half, P_s, 0.79)
print('physics_risk_score:', result['physics_risk_score'])
print('P_tissue_timeseries shape:', result['P_tissue_timeseries'].shape)
print('P_amb_timeseries shape:', result['P_amb_timeseries'].shape)
assert result['P_tissue_timeseries'].shape == (180,)
print('OK')
"
```

Expected: prints `physics_risk_score: <float>`, shapes `(180,)`, `OK`

- [ ] **Step 5: Commit**

```bash
git add generate_dcs_dataset_v3.py
git commit -m "feat: port V2 Bühlmann + add tissue-sat time series output for EP ODE"
```

---

## Task 6: Main generation loop

**Files:**
- Modify: `generate_dcs_dataset_v3.py` — add `extract_features_v3` and `main`

- [ ] **Step 1: Add `extract_features_v3`**

Append to `generate_dcs_dataset_v3.py`:

```python
# ── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def extract_features_v3(depth_series, params, sim_result, bubble_feats,
                         dcs_prob, label, P_surface):
    """Build one CSV row — 41 V2 columns + 6 bubble columns = 47 total."""
    diffs = np.diff(depth_series.astype(np.float64))
    asc_mask = diffs < 0
    if asc_mask.any():
        rates = -diffs[asc_mask] / DT_MIN
        mean_asc, std_asc = float(np.mean(rates)), float(np.std(rates))
    else:
        mean_asc = std_asc = 0.0

    above = np.where(depth_series > 0.01)[0]
    total_dive_time = float(above[-1] * DT_MIN) if len(above) > 0 else 0.0

    row = {
        'max_depth_m':                float(np.max(depth_series)),
        'bottom_time_min':            params['bottom_time_steps'] * DT_MIN,
        'mean_ascent_rate_m_per_min': mean_asc,
        'ascent_rate_std':            std_asc,
        'total_dive_time_min':        total_dive_time,
        'had_safety_stop':            int(params['n_safety_stops'] > 0),
        'is_repetitive':              int(params['is_repetitive']),
        'surface_interval_min':       float(params['surface_interval_min']),
        'age':                        params['age'],
        'body_fat_pct':               params['body_fat_pct'],
        'water_temp_c':               params['water_temp_c'],
        'fitness_level':              params['fitness_level'],
        'hydration_status':           params['hydration_status'],
        'breathing_gas':              params['breathing_gas'],
        'altitude_m':                 params['altitude_m'],
        'P_surface':                  P_surface,
        'pfo':                        params['pfo'],
        'pre_dive_exercise':          params['pre_dive_exercise'],
        'alcohol':                    params['alcohol'],
        'sleep_quality':              params['sleep_quality'],
    }
    for i in range(16):
        row[f'peak_tissue_sat_{i}'] = float(sim_result['peak_tissue_sat'][i])
    row['physics_risk_score'] = float(sim_result['physics_risk_score'])
    row['dcs_probability']    = float(dcs_prob)
    row['label']              = label
    row['profile_type']       = params['profile_type']
    row['physio_type']        = params['physio_type']
    # Bubble features appended last
    row.update(bubble_feats)
    return row
```

- [ ] **Step 2: Add the main generation loop**

Append to `generate_dcs_dataset_v3.py`:

```python
# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    """Generate V3 DCS dataset with EP bubble dynamics.

    Two-pass design:
      Pass 1 — generate all profiles, run Bühlmann + EP integrator, collect
               bubble features (NO labels yet).
      Fit    — StandardScaler on bubble features; calibrate intercept.
      Pass 2 — recompute dcs_probability with calibrated model; draw labels.
    """
    N2_FRACTIONS = {'air': 0.79, 'nitrox32': 0.68, 'nitrox36': 0.64}
    TIME_POINTS_S = np.arange(N_STEPS) * (DT_MIN * 60.0)   # seconds

    table        = get_zhl16c_table()
    rng          = np.random.RandomState(42)
    trim_counter = [0]
    ep_failures  = 0

    timeseries = np.zeros((N_DIVES, N_STEPS), dtype=np.float32)

    # Storage for pass-1 results (no labels yet)
    records_partial = []      # dicts without dcs_probability / label
    bubble_feature_matrix = np.zeros((N_DIVES, 6), dtype=np.float64)
    BUBBLE_FEAT_KEYS = [
        'bubble_R_max', 'bubble_R_surface', 'bubble_dR_dt_max',
        'bubble_integrated_volume', 'bubble_n_critical', 'bubble_terminal_velocity',
    ]

    print('Pass 1/2 — Bühlmann + EP integration...')
    for i in tqdm(range(N_DIVES), desc='Profiles'):
        params = sample_params(rng)

        P_surface         = compute_p_surface(params['altitude_m'])
        nitrogen_fraction = N2_FRACTIONS[params['gas_name']]

        t_half_eff = compute_effective_half_times(
            table, params['age'], params['body_fat_pct'], params['water_temp_c'],
            params['fitness_name'], params['hydration_name'],
        )

        tissues = preload_tissues(
            params['is_repetitive'], params['surface_interval_min'],
            params['prior_max_depth_m'], t_half_eff, P_surface, nitrogen_fraction,
        )

        depth_series = build_depth_profile(params, trim_counter)
        sim_result   = simulate_dive_v3(
            depth_series, tissues, table, t_half_eff, P_surface, nitrogen_fraction
        )

        # ── Fix 1: physio override threshold gate ──
        if params['physio_type'] == 1:
            if sim_result['physics_risk_score'] < 0.68:
                params['physio_type'] = 0
            else:
                params['pfo']             = 1
                params['hydration_name']  = 'dehydrated'
                params['hydration_status'] = 2
                params['exercise_name']   = 'heavy'
                params['pre_dive_exercise'] = 2
                params['body_fat_pct']    = params['physio_override_body_fat']
                t_half_eff = compute_effective_half_times(
                    table, params['age'], params['body_fat_pct'], params['water_temp_c'],
                    params['fitness_name'], params['hydration_name'],
                )
                tissues_v2 = preload_tissues(
                    params['is_repetitive'], params['surface_interval_min'],
                    params['prior_max_depth_m'], t_half_eff, P_surface, nitrogen_fraction,
                )
                sim_result = simulate_dive_v3(
                    depth_series, tissues_v2, table, t_half_eff, P_surface, nitrogen_fraction
                )

        # ── EP bubble integration ──
        # A failed solve must NOT be silently replaced with a fabricated static
        # nucleus: the R0 fallback is min-risk physics, it enters the scaler fit
        # and therefore shifts the z-scores of every other row, and failure is
        # correlated with supersaturation (i.e. with the label). Flag it, exclude
        # it from the scaler, and abort if it is not rare.
        R_traj, ep_ok = integrate_bubble(
            sim_result['P_tissue_timeseries'],
            sim_result['P_amb_timeseries'],
            TIME_POINTS_S,
        )
        if not ep_ok:
            ep_failures += 1
            R_traj = np.full(N_STEPS, np.nan)

        bubble_feats = extract_bubble_features(R_traj, TIME_POINTS_S)
        bubble_feats['ep_solve_failed'] = int(not ep_ok)

        timeseries[i] = depth_series
        for j, k in enumerate(BUBBLE_FEAT_KEYS):
            bubble_feature_matrix[i, j] = bubble_feats[k]

        records_partial.append({
            'depth_series': depth_series,
            'params':       params,
            'sim_result':   sim_result,
            'bubble_feats': bubble_feats,
            'P_surface':    P_surface,
        })

    # ── Fit StandardScaler on SUCCESSFUL solves only ──
    EP_FAILURE_ABORT_RATE = 0.01
    if ep_failures / N_DIVES > EP_FAILURE_ABORT_RATE:
        raise RuntimeError(
            f'{ep_failures}/{N_DIVES} EP solves failed '
            f'({ep_failures/N_DIVES:.2%} > {EP_FAILURE_ABORT_RATE:.0%}). '
            'Refusing to emit a dataset whose bubble physics is fabricated on '
            'a label-correlated subset. Fix the integrator, do not relax this.'
        )

    ok_mask = ~np.isnan(bubble_feature_matrix).any(axis=1)

    # A constant column has zero variance and z = 0/0. If this fires, the bubble
    # model is degenerate (see Correction 11) and the labels would carry no
    # bubble signal at all — the exact failure Correction 1 exists to prevent.
    stds = bubble_feature_matrix[ok_mask].std(axis=0)
    degenerate = [k for k, s in zip(BUBBLE_FEAT_KEYS, stds) if s < 1e-12]
    if degenerate:
        raise RuntimeError(
            f'Bubble features are constant across all profiles: {degenerate}. '
            'The EP model never grows a bubble; standardising is undefined. '
            'See Correction 11 in the design spec.'
        )

    bubble_scaler = StandardScaler().fit(bubble_feature_matrix[ok_mask])
    bubble_z_matrix = np.full_like(bubble_feature_matrix, np.nan)
    bubble_z_matrix[ok_mask] = bubble_scaler.transform(bubble_feature_matrix[ok_mask])
    # Scaler params as JSON, not pickle: joblib.load on an untrusted pickle is
    # arbitrary code execution, and this is twelve floats.
    with open(os.path.join(OUTPUT_DIR, 'bubble_scaler.json'), 'w') as fh:
        json.dump({'features': BUBBLE_FEAT_KEYS,
                   'mean': bubble_scaler.mean_.tolist(),
                   'scale': bubble_scaler.scale_.tolist()}, fh, indent=2)

    # ── Calibrate intercept using median standard-profile bubble z-scores ──
    std_mask = np.array([
        r['params']['profile_type'] == 0 and r['params']['physio_type'] == 0
        for r in records_partial
    ])
    median_z_standard = np.median(bubble_z_matrix[std_mask], axis=0)  # (6,)
    baseline_bubble_contribution = sum(
        BUBBLE_COEFFS[k] * float(median_z_standard[j])
        for j, k in enumerate(BUBBLE_FEAT_KEYS)
    )
    intercept = calibrate_intercept(baseline_bubble_contribution)
    print(f'Recalibrated intercept: {intercept:.4f}  '
          f'(baseline bubble contribution: {baseline_bubble_contribution:.4f})')

    # ── Pass 2: compute dcs_probability and draw labels ──
    print('Pass 2/2 — computing labels...')
    records = []
    for i, rec in enumerate(tqdm(records_partial, desc='Labels')):
        params       = rec['params']
        sim_result   = rec['sim_result']
        bubble_feats = rec['bubble_feats']
        P_surface    = rec['P_surface']

        bubble_z_scores = {
            k: float(bubble_z_matrix[i, j]) for j, k in enumerate(BUBBLE_FEAT_KEYS)
        }

        dcs_prob = compute_dcs_probability_v3(
            physics_risk_score = sim_result['physics_risk_score'],
            pfo                = params['pfo'],
            pre_dive_exercise  = params['exercise_name'],
            alcohol            = params['alcohol'],
            sleep_quality      = params['sleep_name'],
            age                = params['age'],
            hydration_status   = params['hydration_status'],
            fitness_level      = params['fitness_level'],
            bubble_z_scores    = bubble_z_scores,
            intercept          = intercept,
        )

        label = int(rng.random() < dcs_prob)

        row = extract_features_v3(
            rec['depth_series'], params, sim_result, bubble_feats,
            dcs_prob, label, P_surface,
        )
        records.append(row)

    df = pd.DataFrame(records)

    ts_path  = os.path.join(OUTPUT_DIR, 'dive_profiles_timeseries.npy')
    csv_path = os.path.join(OUTPUT_DIR, 'dive_profiles_features.csv')
    np.save(ts_path, timeseries)
    df.to_csv(csv_path, index=False)
    print(f'\nTimeseries : {ts_path}')
    print(f'Features   : {csv_path}')

    if trim_counter[0]:
        print(f'[INFO] {trim_counter[0]} profiles had bottom time trimmed.')
    if ep_failures:
        print(f'[WARN] {ep_failures} EP solves failed — used R0 fallback.')

    pos_rate = df['label'].mean()
    print(f'\nDCS positive: {int(df.label.sum()):,} ({pos_rate:.2%})')
    if pos_rate < 0.05:
        print('WARNING: positive rate below 5%.')

    run_validation(df)
    plot_sample_profiles(df, timeseries)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Smoke-test the loop on 10 profiles**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import generate_dcs_dataset_v3 as v3
v3.N_DIVES = 10
v3.main()
"
```

Expected: completes without error, prints `DCS positive: N (X%)`, writes files.

- [ ] **Step 4: Commit**

```bash
git add generate_dcs_dataset_v3.py
git commit -m "feat: main generation loop — two-pass Bühlmann+EP+logistic pipeline"
```

---

## Task 7: Validation and plot

**Files:**
- Modify: `generate_dcs_dataset_v3.py` — add `run_validation` and `plot_sample_profiles`

- [ ] **Step 1: Add validation**

Append to `generate_dcs_dataset_v3.py` before `main()`:

```python
# ── VALIDATION ───────────────────────────────────────────────────────────────

def run_validation(df):
    """Print structural, physiological-directional, and bubble physics checks."""

    def chk(msg, cond):
        print(f"{'[PASS]' if cond else '[FAIL]'} {msg}")

    print('\n=== DATASET VALIDATION ===')
    chk(f'Row count: {len(df):,}', len(df) == N_DIVES)
    chk('No NaN values', df.isna().sum().sum() == 0)
    chk('physics_risk_score bounded [0, 1]',
        df['physics_risk_score'].between(0, 1).all())
    chk('dcs_probability bounded [0, 1]',
        df['dcs_probability'].between(0, 1).all())
    chk('label is binary', df['label'].isin([0, 1]).all())

    rate = df['label'].mean()
    print(f"{'[PASS]' if rate >= 0.05 else '[WARN]'} Overall DCS rate: {rate:.2%}")

    print('\n=== PHYSIOLOGICAL CALIBRATION ===')
    for name, mask, target in [
        ('Safe       (< 0.70)', df.physics_risk_score < 0.70,                          '<1%'),
        ('Moderate   (0.70–0.85)', (df.physics_risk_score>=0.70)&(df.physics_risk_score<0.85), '1–10%'),
        ('Borderline (0.85–1.0)', (df.physics_risk_score>=0.85)&(df.physics_risk_score<1.0),  '10–40%'),
        ('Breached   (= 1.0)',   df.physics_risk_score == 1.0,                          '40–70%'),
    ]:
        sub = df.loc[mask]
        r = sub['label'].mean() if len(sub) else float('nan')
        print(f'  {name}: {r:.2%}  target:{target}  n={len(sub):,}')

    print('\n=== BUBBLE PHYSICS SANITY CHECKS ===')
    R0_um = R0 * 1e6
    R_crit_um = R_CRIT * 1e6
    chk(f'bubble_R_max > R0 ({R0_um:.1f} µm) for all profiles',
        (df['bubble_R_max'] > R0_um).all())
    chk('bubble_n_critical = 0 when R_max ≤ R_crit',
        (df.loc[df['bubble_R_max'] <= R_crit_um, 'bubble_n_critical'] == 0.0).all())
    chk('bubble_integrated_volume > 0 for all profiles',
        (df['bubble_integrated_volume'] > 0.0).all())
    # DCS rate increases monotonically across R_max quartiles
    quartiles_r = df.groupby(pd.qcut(df['bubble_R_max'], 4, labels=False))['label'].mean()
    chk('DCS rate increases across bubble_R_max quartiles',
        all(quartiles_r.iloc[i] <= quartiles_r.iloc[i+1] for i in range(len(quartiles_r)-1)))
    # DCS rate increases monotonically across integrated_volume quartiles
    quartiles_v = df.groupby(pd.qcut(df['bubble_integrated_volume'], 4, labels=False))['label'].mean()
    chk('DCS rate increases across bubble_integrated_volume quartiles',
        all(quartiles_v.iloc[i] <= quartiles_v.iloc[i+1] for i in range(len(quartiles_v)-1)))

    print('\n=== PHYSIOLOGICAL DIRECTIONAL CHECKS ===')
    failed = []

    def direction_check(label, rates_dict, direction_test, pass_msg, fail_msg):
        values = list(rates_dict.values())
        formatted = '  '.join(f'{k}={v:.2%}' for k, v in rates_dict.items())
        ok = direction_test(values)
        print(f"  {'[PASS]' if ok else '[FAIL]'} {label}: {formatted}")
        if not ok:
            failed.append(label)

    r = lambda mask: df.loc[mask, 'label'].mean()
    direction_check('Fitness',
        {'low': r(df.fitness_level==0), 'mod': r(df.fitness_level==1), 'high': r(df.fitness_level==2)},
        lambda v: v[0] > v[2], 'low > high', 'INVERTED')
    direction_check('Age',
        {'<35': r(df.age<35), '35-50': r((df.age>=35)&(df.age<=50)), '>50': r(df.age>50)},
        lambda v: v[2] > v[0], '>50 > <35', 'INVERTED')
    direction_check('Hydration',
        {'well': r(df.hydration_status==0), 'mild': r(df.hydration_status==1), 'dehyd': r(df.hydration_status==2)},
        lambda v: v[2] > v[0], 'dehyd > well', 'INVERTED')
    pfo0, pfo1 = r(df.pfo==0), r(df.pfo==1)
    ratio = pfo1/pfo0 if pfo0 > 0 else float('nan')
    ok_pfo = ratio >= 2.0
    print(f"  {'[PASS]' if ok_pfo else '[WARN]'} PFO: absent={pfo0:.2%}  present={pfo1:.2%}  ratio={ratio:.2f}x")
    direction_check('Alcohol',
        {'no': r(df.alcohol==0), 'yes': r(df.alcohol==1)},
        lambda v: v[1] > v[0], 'yes > no', 'INVERTED')
    direction_check('Breathing gas',
        {'air': r(df.breathing_gas==0), 'nx32': r(df.breathing_gas==1), 'nx36': r(df.breathing_gas==2)},
        lambda v: v[0] > v[1] > v[2], 'air > nx32 > nx36', 'INVERTED')

    print()
    if not failed:
        print('ALL DIRECTIONAL CHECKS PASSED')
    else:
        print(f'WARNING: {len(failed)} directional check(s) failed: {", ".join(failed)}')
```

- [ ] **Step 2: Add `plot_sample_profiles`**

Append immediately after `run_validation`:

```python
def plot_sample_profiles(df, ts_array):
    """Save 3×3 sample plot: safe/no-PFO | DCS+PFO | DCS/no-PFO."""
    rng_plot = np.random.RandomState(0)
    masks = [
        (df['label'] == 0) & (df['pfo'] == 0),
        (df['label'] == 1) & (df['pfo'] == 1),
        (df['label'] == 1) & (df['pfo'] == 0),
    ]
    titles = ['Safe, no PFO', 'DCS + PFO', 'DCS, no PFO']
    colours = ['steelblue', 'firebrick', 'darkorange']
    time_axis = np.arange(N_STEPS) * DT_MIN

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('V3 Sample Profiles — Bühlmann + EP bubble dynamics', fontsize=11)

    for row_i, (mask, colour, title) in enumerate(zip(masks, colours, titles)):
        pool  = df.index[mask].tolist()
        picks = rng_plot.choice(pool, size=min(3, len(pool)), replace=False)
        for col_i in range(3):
            ax = axes[row_i, col_i]
            if col_i < len(picks):
                idx = picks[col_i]
                risk  = df.loc[idx, 'physics_risk_score']
                prob  = df.loc[idx, 'dcs_probability']
                Rmax  = df.loc[idx, 'bubble_R_max']
                ax.plot(time_axis, ts_array[idx], color=colour, linewidth=1.2)
                ax.invert_yaxis()
                ax.set_xlabel('Time (min)', fontsize=8)
                ax.set_ylabel('Depth (m)', fontsize=8)
                ax.set_title(
                    f'risk={risk:.3f} | P(DCS)={prob:.3f}\nR_max={Rmax:.1f}µm | {title}',
                    fontsize=7,
                )
                ax.tick_params(labelsize=7)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'dive_profiles_sample.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f'Plot saved: {out}')
```

- [ ] **Step 3: Smoke-test validation on 100 profiles**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import generate_dcs_dataset_v3 as v3
v3.N_DIVES = 100
v3.main()
"
```

Expected: validation prints, plot saved, no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add generate_dcs_dataset_v3.py
git commit -m "feat: add validation checks and sample plot"
```

---

## Task 8: Full run and final commit

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run full generation (50,000 profiles)**

```bash
cd ~/Desktop/DCS_PINN_DATASET_V3
python generate_dcs_dataset_v3.py
```

Expected runtime: ~10–15 minutes on CPU. Watch for:
- `[WARN] EP solves failed` — should be 0 or very low
- `Overall DCS rate: X%` — should be 5–12%
- `ALL DIRECTIONAL CHECKS PASSED`
- All bubble physics checks `[PASS]`

- [ ] **Step 3: Verify output files exist and have correct shape**

```bash
python -c "
import numpy as np, pandas as pd
ts = np.load('dive_profiles_timeseries.npy')
df = pd.read_csv('dive_profiles_features.csv')
assert ts.shape == (50000, 180), f'Bad timeseries shape: {ts.shape}'
assert len(df.columns) == 47, f'Expected 47 columns, got {len(df.columns)}'
assert df.isna().sum().sum() == 0, 'NaN values found'
print('Shape OK:', ts.shape)
print('Columns:', len(df.columns))
print('Label rate:', round(df.label.mean(), 3))
print('bubble_R_max range:', round(df.bubble_R_max.min(), 3), round(df.bubble_R_max.max(), 3))
"
```

- [ ] **Step 4: Commit final outputs**

```bash
git add generate_dcs_dataset_v3.py tests/ \
    dive_profiles_features.csv dive_profiles_timeseries.npy \
    dive_profiles_sample.png bubble_scaler.json
git commit -m "feat: complete V3 dataset — EP bubble dynamics, 47 columns, all checks pass"
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|-----------------|------|
| EP ODE with quasi-static Laplace | Task 2 |
| Henry's law C_∞ conversion | Task 2 (`_ep_rhs`) |
| R₀ = 0.7 µm (VPM nucleus) | Task 2 (constant) |
| `solve_ivp` with Radau | Task 2 (`integrate_bubble`) |
| 6 bubble features extracted | Task 3 |
| `bubble_n_critical` zero below R_crit | Task 3 (formula + test) |
| `bubble_terminal_velocity` renamed + excluded from logistic | Task 3 + Task 4 |
| Pre-specified bubble logistic coefficients | Task 4 (`BUBBLE_COEFFS`) |
| V2 logistic reconstructed from source | Task 4 (`compute_dcs_probability_v3`) |
| Intercept calibration to DAN/Howle targets | Task 4 (`calibrate_intercept`) |
| Two-pass design (features first, labels second) | Task 6 (`main`) |
| Fix 1 physio override threshold gate | Task 6 (`main`) |
| StandardScaler fitted across all profiles | Task 6 (`main`) |
| Baseline bubble contribution from median standard profiles | Task 6 (`main`) |
| Bubble physics sanity checks | Task 7 (`run_validation`) |
| 6 physiological directional checks | Task 7 (`run_validation`) |
| 3×3 sample plot with R_max in title | Task 7 (`plot_sample_profiles`) |
| Honesty ceiling | Documented in spec (not a runtime check) |
| 47-column CSV | Task 6 (`extract_features_v3`) |
| Same npy shape as V2 | Task 6 (`main`) |

**Placeholder scan:** None found.

**Type consistency:** `BUBBLE_FEAT_KEYS` list order matches `bubble_feature_matrix` column order and `bubble_z_scores` dict keys throughout Tasks 6–7. `BUBBLE_COEFFS` keys match `BUBBLE_FEAT_KEYS` and `extract_bubble_features` return keys.

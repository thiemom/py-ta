# py-ta

A Python package for Two-Delay Gamma Flame Transfer Function (FTF) modeling and parameter fitting.

## Overview

This package implements the Two-Delay Gamma FTF model for combustion dynamics analysis. The model captures flame response as a superposition of two physical pathways (equivalence ratio and turbulence) using Gamma-distributed delay kernels, making it suitable for complex FTF data with interference patterns and phase jumps.

## Contents

- **`pyftf.py`** - Two-Delay Gamma FTF model implementation with parameter fitting and grid search capabilities
- **`pyftf_demo.py`** - Comprehensive examples demonstrating model usage with synthetic data generation, fitting, and visualization
  - Includes a fuel-split pilot/main blending demo with I(ω) and T22(ω) sweeps
- **`FTF_fits_references.md`** - Technical reference documentation with equations and usage guidelines
 - **`entropy_loop.py`** - Breathing-mode loop with OU forcing, LBO gate, entropy pockets, and Bake-style conversion (analytic PSD + helpers)
 - **`entropy_loop_demo.py`** - Demo script for the entropy loop (PSD and time-trace synthesis)

## Usage

1. Set up the Python environment:
   ```bash
   ./setup_environment.sh
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run the example:
    ```bash
    python pyftf_demo.py
    ```

4. Run the entropy loop demo:
   ```bash
   python entropy_loop_demo.py
   ```

## Features

- **Two-Delay Gamma Model**: Implements the physically-motivated two-pathway FTF model with Gamma-distributed delays
- **Parameter Fitting**: Robust least-squares fitting with multi-start optimization and regularization
- **Grid Search**: Automated search over shape parameters (m_φ, m_t) to find optimal model configuration
- **T22 Normalization**: Reliable handling of T22 data using temperature ratio T2/T1 with automatic I-domain conversion
- **Multi-Start Optimization**: Avoids local minima through multiple random initializations
- **Comprehensive Examples**: Multiple usage scenarios with synthetic data generation and visualization
 - **Fuel-Split Blending**: Mix pilot and main stages via a logistic split with optional alignment/cross-term

## Model Equation

The Two-Delay Gamma model represents the interaction index as:

```
I(ω) = A_φ · G_φ(ω) − A_t · G_t(ω)
```

where `G_i(ω) = exp(−iω(τ_i + θ_i)) · Γ(m_i) / (Γ(m_i) + (iωτ_i)^m_i)`

**Parameters:**
- `A_φ`, `A_t`: Pathway amplitudes (equivalence ratio and turbulence)
- `τ_φ`, `τ_t`: Characteristic delay times [s]
- `θ_φ`, `θ_t`: Additional phase delays [s]  
- `m_φ`, `m_t`: Shape parameters controlling delay distribution width

## Quick Start

```python
from pyftf import fit_two_delay_gamma, fit_two_delay_gamma_grid

# Basic fitting with known shape parameters
params, info = fit_two_delay_gamma(
    omega, magnitude, phase,
    m_phi=2, m_t=3,
    phase_in_degrees=False
)

# T22 normalization fitting
params, info = fit_two_delay_gamma(
    omega, T22_magnitude, T22_phase,
    m_phi=2, m_t=3,
    normalize=True,
    T_ratio=3.5  # Temperature ratio T2/T1
)

# Grid search to find optimal shape parameters
best, results = fit_two_delay_gamma_grid(
    omega, magnitude, phase,
    mphi_list=range(1, 6),
    mt_list=range(1, 6),
    selection="rmse"
)
```

## Fuel-Split API

Blend pilot and main two-delay stages into a mixed interaction index `I_mix(ω, α)` and project to T22.

```python
from pyftf import FuelSplitConfig, fuel_split_I, T22_from_fuel_split, TwoDelayParams

# Define pilot and main parameters (Two-Delay Gamma per stage)
pilot = TwoDelayParams(A_phi=0.7, A_t=0.4, tau_phi=3.2e-3, r_tau=0.9, theta_phi=1.6e-3, theta_t=0.9e-3)
main  = TwoDelayParams(A_phi=0.45, A_t=0.65, tau_phi=2.3e-3, r_tau=0.82, theta_phi=1.1e-3, theta_t=0.7e-3)

cfg = FuelSplitConfig(
    s0=0.0, s1=5.0, s2=0.0,   # logistic weight for pilot vs main
    dtau_pilot=0.0, dtau_main=0.0,  # optional phase-align shifts
    kappa=0.0, tau_c=0.0            # optional cross-term strength/delay
)

alpha = 0.5  # fuel split in [0, 1]
I_mix = fuel_split_I(omega, alpha, pilot, main, m_pilot_phi=2, m_pilot_t=2, m_main_phi=2, m_main_t=2, cfg=cfg)

# Project to raw T22 with temperature ratio T2/T1
T_ratio = 3.5
T22 = T22_from_fuel_split(omega, alpha, pilot, main, T_ratio, cfg=cfg)
```

Demo script section: `run_fuel_split_example()` in `pyftf_demo.py` generates:

- `fuel_split_I_example.png` (|I_mix| and phase sweeps over α)
- `fuel_split_T22_example.png` (|T22| and phase sweeps over α)

Run the demo after activating the environment:

```bash
source venv/bin/activate
python pyftf_demo.py
```

## Silencing runtime warnings

During optimization, NumPy/SciPy may emit runtime warnings (overflow, divide-by-zero, invalid operations). You can silence these per-call by setting `suppress_warnings=True`.

Default: `suppress_warnings=False` (warnings shown).

```python
# Suppress warnings in a single fit
params, info = fit_two_delay_gamma(
    omega, magnitude, phase,
    m_phi=2, m_t=3,
    suppress_warnings=True
)

# Suppress warnings in grid search
best, results = fit_two_delay_gamma_grid(
    omega, magnitude, phase,
    mphi_list=range(1, 6), mt_list=range(1, 6),
    selection="rmse",
    suppress_warnings=True
)

# With normalization
params, info = fit_two_delay_gamma(
    omega, T22_mag, T22_phase,
    m_phi=2, m_t=3,
    normalize=True, T_ratio=3.5,
    suppress_warnings=True
)
```

## Dependencies

- numpy
- pandas  
- matplotlib
- scipy

All dependencies are listed in `requirements.txt` and installed automatically by the setup script.

## License

See `LICENSE` file for details.
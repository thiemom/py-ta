# py-ta

A Python package for Two-Delay Gamma Flame Transfer Function (FTF) modeling and parameter fitting.

## Overview

This package implements the Two-Delay Gamma FTF model for combustion dynamics analysis. The model captures flame response as a superposition of two physical pathways (equivalence ratio and turbulence) using Gamma-distributed delay kernels, making it suitable for complex FTF data with interference patterns and phase jumps.

## Contents

- **`pyftf.py`** - Two-Delay Gamma FTF model implementation with parameter fitting and grid search capabilities
- **`example_usage.py`** - Comprehensive examples demonstrating model usage with synthetic data generation, fitting, and visualization
- **`FTF_fits_references.md`** - Technical reference documentation with equations and usage guidelines

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
   python example_usage.py
   ```

## Features

- **Two-Delay Gamma Model**: Implements the physically-motivated two-pathway FTF model with Gamma-distributed delays
- **Parameter Fitting**: Robust least-squares fitting with multi-start optimization and regularization
- **Grid Search**: Automated search over shape parameters (m_φ, m_t) to find optimal model configuration
- **T22 Normalization**: Reliable handling of T22 data using temperature ratio T2/T1 with automatic I-domain conversion
- **Multi-Start Optimization**: Avoids local minima through multiple random initializations
- **Comprehensive Examples**: Multiple usage scenarios with synthetic data generation and visualization

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

## Dependencies

- numpy
- pandas  
- matplotlib
- scipy

All dependencies are listed in `requirements.txt` and installed automatically by the setup script.

## License

See `LICENSE` file for details.
# py-ta

A comprehensive Python toolkit for Flame Transfer Function (FTF) modeling, combustion instability analysis, and burner pattern optimization.

## Overview

This package provides both specialized Two-Delay Gamma FTF modeling and a general FTF toolbox with multiple model families, registry-based fitting, and advanced utilities for combustion instability analysis. It includes tools for burner pattern analysis in annular combustors and entropy loop modeling for thermoacoustic systems.

## Contents

### **Core FTF Modeling**
- **`ftf.py`** - Comprehensive FTF modeling toolkit with dual functionality:
  - **Two-Delay Gamma Specialist**: Advanced fitting for two-pathway DTL model
  - **General FTF Toolbox**: 6 model families with registry-based fitting
- **`ftf_demo.py`** - Complete examples including toolbox synthetic demos

### **Combustor Analysis** 
- **`fourier_patterns.py`** - Burner pattern analysis for annular combustors using Fourier decomposition
- **`entropy_loop.py`** - Breathing-mode loop with OU forcing, LBO gate, entropy pockets
- **`entropy_loop_demo.py`** - Demo script for entropy loop (PSD and time-trace synthesis)

### **Documentation**
- **`FTF_fits_references.md`** - Technical reference with equations and usage guidelines
- **`setup_environment.sh`** - Environment setup with direnv support

## Quick Start

### **Environment Setup**
```bash
# Setup environment (creates local venv and installs dependencies)
./setup_environment.sh

# Option 1: Use direnv for automatic activation (recommended)
direnv allow

# Option 2: Manual activation
source venv/bin/activate
```

### **Run Examples**
```bash
# FTF modeling examples (Two-Delay Gamma + Toolbox)
python ftf_demo.py

# Burner pattern analysis
python fourier_patterns.py

# Entropy loop demo
python entropy_loop_demo.py
```

## Features

### **FTF Modeling Capabilities**
- **Two-Delay Gamma Specialist**: Advanced two-pathway DTL model with robust fitting
- **General FTF Toolbox**: 6 model families (diffusion, gamma, dispersion, rational, autoignition, gauss-opposite)
- **Registry-Based Fitting**: Single model selection and pilot→main combinations
- **Multi-Start Optimization**: Robust parameter estimation avoiding local minima
- **Grid Search**: Automated shape parameter optimization (m_φ, m_t)
- **T22 Normalization**: Reliable handling with temperature ratio T2/T1

### **Combustor Analysis Tools**
- **Burner Pattern Analysis**: Fourier decomposition for annular combustor patterns
- **Spatial Coherence**: Mode coupling analysis for instability prediction
- **Fuel-Split Modeling**: Multi-stage combustor blending with phase alignment
- **Entropy Loop Modeling**: Thermoacoustic system dynamics with stochastic forcing

### **Advanced Features**
- **Complex Loss Functions**: Magnitude + phase weighting for robust fitting
- **Synthetic Data Generation**: Comprehensive test cases and validation
- **Visualization**: Automated plot generation for all analysis types
- **Development Environment**: Direnv integration for seamless workflow

## FTF Model Families

### **Two-Delay Gamma (Specialist)**
The core two-pathway model for combustion instability analysis:

```
I(ω) = A_φ · G_φ(ω) − A_t · G_t(ω)
```

where `G_i(ω) = exp(−iω(τ_i + θ_i)) · Γ(m_i) / (Γ(m_i) + (iωτ_i)^m_i)`

**Parameters:**
- `A_φ`, `A_t`: Pathway amplitudes (equivalence ratio vs turbulence)
- `τ_φ`, `τ_t`: Characteristic delay times [s]
- `θ_φ`, `θ_t`: Additional phase delays [s]  
- `m_φ`, `m_t`: Shape parameters controlling delay distribution width

### **General Toolbox Models**
- **Diffusion**: `n·e^{-iωτ}·(1 + iω/ωc)^{-order}` - Transport-limited response
- **Gamma**: `n·(1 + iωθc)^{-ν}` - Lieuwen-style distributed delays
- **Dispersion**: `n·e^{-iωτ}·exp(-(ω/ωd)^k)` - Parmentier mixing/dispersion
- **Rational**: Low-order ARMA-like transfer functions
- **Autoignition**: Zonal superposition (propagation + AI core)
- **Gauss Opposite**: Two-pathway with Gaussian delays and opposite signs

## API Examples

### **Two-Delay Gamma Fitting**
```python
from ftf import fit_two_delay_gamma, fit_two_delay_gamma_grid

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

### **General FTF Toolbox**
```python
from ftf import ftf_n_tau_gauss_opposite, fit_ftf_with_model_choice, REGISTRY

# Direct model evaluation
f = np.linspace(10, 1000, 500)  # Hz
ftf_func = ftf_n_tau_gauss_opposite(n=0.8, tau_conv=4e-3, beta=0.6)
H = ftf_func(f)

# Registry-based model selection
best = fit_ftf_with_model_choice(f, F_meas, candidate_ids=[0,1,2,3,4,5])

# Pilot→main combo fitting
from ftf import build_combo_registry, fit_over_combos
combos = build_combo_registry([0,1,2,4,5])
best_combo = fit_over_combos(f, F_meas, combos)
```

### **Burner Pattern Analysis**
```python
from fourier_patterns import analyze_pattern_effectiveness

# Analyze N=15 burner pattern
pattern = [1,1,2,1,2,1,2,2,1,1,1,1,2,1,1]  # burner types
burner_params = {'n1': 0.8, 'n2': 1.2, 'tau1': 3e-3, 'tau2': 2e-3}
chamber_params = {'R': 0.3, 'c0': 340, 'N': 15}

results = analyze_pattern_effectiveness(pattern, burner_params, chamber_params)
# Returns coupling strengths for azimuthal modes 1-5
```

## Fuel-Split API

Blend pilot and main two-delay stages into a mixed interaction index `I_mix(ω, α)` and project to T22.

```python
from ftf import FuelSplitConfig, fuel_split_I, T22_from_fuel_split, TwoDelayParams

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

## Generated Outputs

Running the demo scripts produces comprehensive analysis plots:

### **FTF Demo (`ftf_demo.py`)**
- `two_delay_gamma_fit_example.png` - Basic Two-Delay Gamma fitting
- `normalization_example.png` - T22 normalization demonstration  
- `fuel_split_I_example.png` - Fuel-split |I_mix| and phase sweeps
- `fuel_split_T22_example.png` - Fuel-split |T22| projections

### **Burner Pattern Analysis (`fourier_patterns.py`)**
- Console output with coupling strengths for azimuthal modes 1-5
- Identifies vulnerable modes and optimal burner placement strategies

### **Entropy Loop (`entropy_loop_demo.py`)**
- Time-domain synthesis with stochastic forcing
- Power spectral density analysis and validation

## Advanced Configuration

### **Warning Suppression**
```python
# Suppress NumPy/SciPy warnings during optimization
params, info = fit_two_delay_gamma(
    omega, magnitude, phase,
    suppress_warnings=True  # Default: False
)
```

### **Custom Model Registration**
```python
# Add custom FTF model to registry
def my_custom_ftf(n, tau, custom_param):
    def F(f):
        return n * np.exp(-1j*2*np.pi*f*tau) * custom_param
    return F

# Register for use with fitting utilities
REGISTRY[6] = ("custom_model", 
               lambda p: my_custom_ftf(p[0], p[1], p[2]),
               ["n", "tau_s", "custom_param"],
               np.array([0.0, 0.001, 0.5]),  # lower bounds
               np.array([2.0, 0.050, 2.0]))  # upper bounds
```

## Dependencies

Core scientific computing stack:
- **numpy** - Numerical computing and array operations
- **scipy** - Optimization and scientific functions  
- **matplotlib** - Plotting and visualization
- **pandas** - Data manipulation (for entropy loop analysis)

All dependencies are listed in `requirements.txt` and installed automatically by `setup_environment.sh`.

## Project Structure

```
py-ta/
├── ftf.py                   # Core FTF modeling toolkit
├── ftf_demo.py              # Comprehensive examples
├── fourier_patterns.py      # Burner pattern analysis
├── entropy_loop.py          # Thermoacoustic dynamics
├── entropy_loop_demo.py     # Entropy loop examples
├── requirements.txt         # Python dependencies
├── setup_environment.sh     # Environment setup script
├── .envrc                   # Direnv configuration (local)
└── FTF_fits_references.md   # Technical documentation
```

## Contributing

This toolkit is designed for combustion instability research and industrial applications. The modular design allows easy extension with new FTF models, fitting algorithms, and analysis tools.

## License

See `LICENSE` file for details.
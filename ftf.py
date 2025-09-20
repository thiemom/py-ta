# -*- coding: utf-8 -*-
r"""
ftf.py — Two-Delay Gamma Flame Transfer Function (FTF) model and fitting

This module implements the Two-Delay Gamma FTF model and provides robust
fitting utilities, including grid search over shape parameters and optional
normalization for T22 data.

Two-Delay Gamma FTF Model
-------------------------
Unicode/plain equations:
    I(ω) = A_φ · G_φ(ω) − A_t · G_t(ω)
where
    G_i(ω) = exp(−i ω (τ_i + θ_i)) · Γ(m_i) / (Γ(m_i) + (i ω τ_i)^{m_i})

LaTeX form:
    I(\omega) = A_\phi\,G_\phi(\omega) - A_t\,G_t(\omega),\quad
    G_i(\omega) = e^{-i\omega(\tau_i+\theta_i)}\,\frac{\Gamma(m_i)}{\Gamma(m_i) + (i\omega\tau_i)^{m_i}}

Meaning
-------
- A_φ, A_t: pathway amplitudes (equivalence ratio and turbulence)
- τ_φ, τ_t: characteristic delays [s]
- θ_φ, θ_t: additional phase delays [s]
- m_φ, m_t: integer shape parameters (delay distribution width)
- Positive ER pathway, negative turbulence pathway → interference patterns

Features
--------
- Superposition of two physical mechanisms (ER and turbulence)
- Gamma-distributed delay kernels per pathway (distributed-delay/DTL)
- Captures interference, magnitude nulls, and phase jumps
- Independent delay distribution parameters per pathway
- Shape parameters control distribution width (higher m → narrower)

Data normalization
------------------
This module supports both domains:
- Direct I(ω) fitting (interaction index domain)
- T22 normalization: I(ω) = (T22(ω) − 1) / (T2/T1 − 1)

Historical context (very brief)
-------------------------------
- Crocco & Cheng (system-theory view; zeros/poles)
- Dowling (thermoacoustic ZPK transfer functions)
- Schuermans & Polifke (industrial low‑order flame models, DTL)
- Lieuwen, Noiray, Polifke (robust ID, stochastic links, modern reviews)

Key references
--------------
- Lieuwen, T. (2012). Unsteady Combustor Physics. CUP. (distributed delays, two-pathway)
- Schuller, T., Durox, D., Candel, S. (2003). Combustion and Flame. (laminar FTF model)
- Noiray, N., Schuermans, B. (2013). IJNLM. (noise-driven Hopf metrics)
- Polifke, W. (2020). Prog. Energy Combust. Sci. (FTF/FDF/DTL system ID)
- Schuller, T., Noiray, N., Poinsot, T., Candel, S. (2020). J. Fluid Mech. (overview & normalization)

Usage
-----
See fit utilities in this module:
- fit_two_delay_gamma(omega, mag, phase, ...)
- fit_two_delay_gamma_grid(omega, mag, phase, ...)

For a complete example, run ftf_demo.py.
"""
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, List, Callable, Optional
from scipy.optimize import least_squares
import warnings
from contextlib import contextmanager

# Two-delay Gamma (DTL) model that:
# 	• Uses integer Gamma shapes m_phi, m_t in N (cascades of first-order lags)
# 	• Re-parameterizes the turbulence delay via a ratio r_tau = tau_t/tau_phi 
# 	  (so tau_t = r_tau * tau_phi), initialized slightly less than 1
# 	• Lets you fit in your preferred domain:
# 	  - default: data are (T2/T1-1) * I(omega) (i.e., T22-1) → no normalization needed
# 	  - normalize=True: pass T2/T1 and the fitter divides by (T2/T1-1) to fit I(omega)
# 	• Uses log-magnitude + wrapped-phase residuals with robust loss

# Model equation:
# I(omega) = A_phi * exp(-i*omega*tau_phi) * (1 + i*omega*theta_phi)^(-m_phi)
#          - A_t * exp(-i*omega*tau_t) * (1 + i*omega*theta_t)^(-m_t)
# where tau_t = r_tau * tau_phi, with r_tau in (0,1)

# If normalize=True, the script converts T22 data to I domain:
# I_meas = (T22 - 1) / (T2/T1 - 1)
# Then fits I directly, which avoids numerical issues in the T22 domain.


# References 
# 	•	Lieuwen, T. Unsteady Combustor Physics, CUP, 2012 — two-pathway decomposition; distributed-delay kernels.
# 	•	Polifke, W. “System identification of combustion dynamics… FTF, FDF and DTL,” Prog. Energy Combust. Sci. 79 (2020) — DTL/Gamma fits & rationale.
# 	•	Schuermans, B.; Bellucci, V.; Paschereit, C.O. “Thermoacoustic modeling and control of a full-scale gas turbine combustor,” Combust. Sci. Tech. 176 (2004) — low-order flame models in practice.
# 	•	Noiray, N.; Schuermans, B. “Deterministic quantities characterizing noise-driven Hopf bifurcations…,” Int. J. Non-Linear Mech. 50 (2013) — links between linear FTF and limit-cycle parameters.
# 	•	Schuller, T.; Noiray, N.; Poinsot, T.; Candel, S. “Dynamics of premixed flames and combustion instabilities,” J. Fluid Mech. 894 (2020) — overview & normalization.


# ----------------------- Model -----------------------
def G_int(omega: np.ndarray, m: int, theta: float, tau: float) -> np.ndarray:
    """Integer-shape Gamma/DTL kernel: exp(-i*omega*tau) * (1 + i*omega*theta)^(-m)."""
    return np.exp(-1j*omega*tau) * (1.0 + 1j*omega*theta)**(-m)

@dataclass
class TwoDelayParams:
    A_phi: float      # >0
    A_t: float        # >0
    tau_phi: float    # >0
    r_tau: float      # in (r_min, 1): tau_t = r_tau * tau_phi
    theta_phi: float  # >0
    theta_t: float    # >0

def I_two_delay(omega: np.ndarray, p: TwoDelayParams, m_phi: int, m_t: int) -> np.ndarray:
    """
    Compute Two-Delay Gamma FTF interaction index I(ω).
    
    Parameters:
    -----------
    omega : array_like
        Angular frequency array [rad/s]
    p : TwoDelayParams
        Model parameters
    m_phi, m_t : int
        Shape parameters for equivalence ratio and turbulence pathways
        
    Returns:
    --------
    I : complex array
        Interaction index I(ω)
    """
    tau_t = p.r_tau * p.tau_phi
    Gp = G_int(omega, m_phi, p.theta_phi, p.tau_phi)
    Gt = G_int(omega, m_t,   p.theta_t,   tau_t)
    return p.A_phi * Gp - p.A_t * Gt

def compute_ftf(omega: np.ndarray, params: TwoDelayParams, m_phi: int = 2, m_t: int = 2) -> np.ndarray:
    """
    Convenience method to compute Two-Delay Gamma FTF from parameters.
    
    Parameters:
    -----------
    omega : array_like
        Angular frequency array [rad/s]
    params : TwoDelayParams
        Fitted or known model parameters
    m_phi : int, default 2
        Shape parameter for equivalence ratio pathway
    m_t : int, default 2
        Shape parameter for turbulence pathway
        
    Returns:
    --------
    ftf : complex array
        Flame Transfer Function I(ω)
        
    Example:
    --------
    >>> omega = np.logspace(0, 3, 100)
    >>> params = TwoDelayParams(A_phi=1.8, A_t=1.2, tau_phi=0.003, 
    ...                         r_tau=0.85, theta_phi=0.012, theta_t=0.018)
    >>> ftf = compute_ftf(omega, params, m_phi=2, m_t=3)
    """
    return I_two_delay(omega, params, m_phi, m_t)

# ----------------------- Utils -----------------------
def unwrap_phase(phi, deg=False):
    phi = np.asarray(phi, float)
    if deg: phi = np.deg2rad(phi)
    return np.unwrap(phi)

def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def softplus(x): return np.log1p(np.exp(x))
def inv_softplus(y):
    y = np.asarray(y, float)
    return np.log(np.expm1(np.maximum(y, 1e-12)))

def s2unit_interval(x):   # R -> (0,1)
    return 1.0/(1.0 + np.exp(-x))
def unit_interval2s(u):   # (0,1) -> R
    u = np.clip(u, 1e-6, 1-1e-6)
    return np.log(u/(1.0-u))

def magphase_to_complex(mag, phase, phase_in_degrees=False):
    phi = unwrap_phase(phase, deg=phase_in_degrees)
    return np.asarray(mag, float) * np.exp(1j*phi)

def as_weight_array(w, n: int) -> np.ndarray:
    """Accept scalar or array weights and return shape [n] array."""
    w = np.asarray(w) if np.ndim(w) else np.array([w], float)
    if w.size == 1:
        return np.full(n, float(w))
    if w.size != n:
        raise ValueError("Weight length must match data length or be scalar.")
    return w.astype(float)

# -------------------- Warning suppression -------------------
@contextmanager
def _maybe_suppress_warnings(enabled: bool):
    """
    Context manager to optionally silence common NumPy/SciPy runtime warnings
    such as overflow, divide-by-zero, and invalid operations during optimization.
    """
    if not enabled:
        yield
        return
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                # Silence SciPy optimize warnings if available
                from scipy.optimize import OptimizeWarning
                warnings.simplefilter('ignore', OptimizeWarning)
            except Exception:
                pass
            yield

# -------------------- Initialization -------------------
def init_two_delay(omega, M_use, phi_unw, m_phi: int, m_t: int) -> TwoDelayParams:
    nlo = max(3, int(0.15*len(omega)))
    S0  = float(np.median(np.clip(M_use[:nlo], 1e-8, None)))
    # bulk delay from low-f phase slope
    A = np.vstack([omega[:nlo], np.ones(nlo)]).T
    slope, _ = np.linalg.lstsq(A, phi_unw[:nlo], rcond=None)[0]
    tau0 = float(np.clip(-slope, 0.0, 1.0/(omega[1]-omega[0] + 1e-9)))
    # crude corner
    target = S0/np.sqrt(2)
    idx = np.where(M_use <= target)[0]
    wc = omega[idx[0]] if idx.size else omega[len(omega)//3]
    theta_phi0 = 1.2/max(wc, 1e-6)
    theta_t0   = 0.8/max(wc, 1e-6)
    A_phi0 = max(S0, 1e-6)
    A_t0   = 0.6 * A_phi0
    return TwoDelayParams(
        A_phi=A_phi0, A_t=A_t0,
        tau_phi=tau0, r_tau=0.9,         # turbulence slightly faster
        theta_phi=theta_phi0, theta_t=theta_t0
    )

# ---------------------- Fitting ------------------------
def pack_uncon(p: TwoDelayParams, r_min=0.4) -> np.ndarray:
    u = (p.r_tau - r_min)/(1.0 - r_min)  # map (r_min,1)->(0,1)
    v = [
        inv_softplus(p.A_phi), inv_softplus(p.A_t),
        inv_softplus(p.tau_phi),
        unit_interval2s(u),                # r_tau
        inv_softplus(p.theta_phi), inv_softplus(p.theta_t)
    ]
    return np.array(v, float)

def unpack_uncon(v: np.ndarray, r_min=0.4) -> TwoDelayParams:
    v = np.asarray(v, float).ravel()
    i=0
    A_phi   = softplus(v[i]); i+=1
    A_t     = softplus(v[i]); i+=1
    tau_phi = softplus(v[i]); i+=1
    u       = s2unit_interval(v[i]); i+=1
    r_tau   = r_min + (1.0 - r_min)*u
    theta_p = softplus(v[i]); i+=1
    theta_t = softplus(v[i]); i+=1
    return TwoDelayParams(A_phi, A_t, tau_phi, r_tau, theta_p, theta_t)

def residuals_joint(x, omega, mag, phase, m_phi, m_t, w_mag_arr, w_phase_arr, r_min, r0, lambda_r):
    """Joint residuals: magnitude + phase + penalty"""
    p = unpack_uncon(x, r_min=r_min)
    I = I_two_delay(omega, p, m_phi, m_t)
    M_use = mag
    phi_unw = phase
    r_mag   = np.sqrt(w_mag_arr)   * (np.log(M_use + 1e-12) - np.log(np.abs(I) + 1e-12))
    r_phase = np.sqrt(w_phase_arr) * (wrap_pi(phi_unw - np.angle(I)))
    # penalty on r_tau about r0 (append as an extra residual)
    r_pen = np.sqrt(lambda_r) * (p.r_tau - r0)
    return np.concatenate([r_mag, r_phase, np.atleast_1d(r_pen)])

def residuals_complex(x, omega, mag, phase, m_phi, m_t, w_mag_arr, w_phase_arr, r_min, r0, lambda_r):
    """Complex residuals: real + imaginary + penalty (better for normalization)"""
    p = unpack_uncon(x, r_min=r_min)
    I_pred = I_two_delay(omega, p, m_phi, m_t)
    I_data = mag * np.exp(1j * phase)
    
    # Complex residual weighted by magnitude and phase weights
    complex_residual = I_data - I_pred
    r_real = np.sqrt(w_mag_arr) * np.real(complex_residual)
    r_imag = np.sqrt(w_phase_arr) * np.imag(complex_residual)
    
    # penalty on r_tau about r0
    r_pen = np.sqrt(lambda_r) * (p.r_tau - r0)
    return np.concatenate([r_real, r_imag, np.atleast_1d(r_pen)])

def _fit_T22_via_I_domain(omega, T22_mag, T22_phase, T_ratio, phase_in_degrees, m_phi, m_t, 
                         w_mag, w_phase, r_min, r0, lambda_r, robust, f_scale, 
                         max_nfev, multi_start, n_starts, use_complex_residual,
                         suppress_warnings: bool = False):
    """
    Internal function: Fit T22 data by converting to I domain, fitting there, then transforming back.
    
    This workaround method converts T22 data to I domain using I = (T22 - 1) / (T_ratio - 1),
    fits the Two-Delay Gamma model in I domain (which is numerically stable), then transforms
    the fitted model back to T22 domain for evaluation.
    
    Parameters:
    -----------
    omega : array
        Angular frequency array [rad/s]
    T22_mag, T22_phase : array
        T22 magnitude and phase data
    T_ratio : float
        Temperature ratio T2/T1
    phase_in_degrees : bool
        Whether phase data is in degrees
    m_phi, m_t : int
        Shape parameters
    w_mag, w_phase : float
        Residual weights
    r_min, r0, lambda_r : float
        Regularization parameters
    robust : str
        Robust loss function
    f_scale : float
        Loss scale parameter
    max_nfev : int
        Maximum function evaluations
    multi_start : bool
        Use multi-start optimization
    n_starts : int
        Number of random starts
    use_complex_residual : bool
        Use complex residuals
        
    Returns:
    --------
    params : TwoDelayParams
        Fitted parameters
    info : dict
        Enhanced fitting information with T22 domain metrics
    """
    
    # Convert T22 → I domain
    T22_complex = magphase_to_complex(T22_mag, T22_phase, phase_in_degrees)
    I_complex = (T22_complex - 1.0) / (T_ratio - 1.0)
    I_mag = np.abs(I_complex)
    I_phase = np.angle(I_complex)
    
    # Fit in I domain (normalize=False)
    p_fit, info_I = fit_two_delay_gamma(
        omega, I_mag, I_phase,
        phase_in_degrees=False,  # Already converted to radians
        m_phi=m_phi, m_t=m_t,
        normalize=False,  # Fit directly in I domain
        w_mag=w_mag, w_phase=w_phase,
        r_min=r_min, r0=r0, lambda_r=lambda_r,
        robust=robust, f_scale=f_scale, max_nfev=max_nfev,
        multi_start=multi_start, n_starts=n_starts,
        use_complex_residual=use_complex_residual,
        use_I_domain_workaround=False,  # Prevent infinite recursion
        suppress_warnings=suppress_warnings
    )
    
    # Generate fitted I and convert to T22
    I_fit = I_two_delay(omega, p_fit, m_phi, m_t)
    T22_fit = 1.0 + (T_ratio - 1.0) * I_fit
    T22_fit_mag = np.abs(T22_fit)
    T22_fit_phase = np.angle(T22_fit)
    
    # Calculate T22 domain metrics
    T22_mag_rmse = np.sqrt(np.mean((T22_fit_mag - T22_mag)**2))
    T22_phase_rmse = np.sqrt(np.mean((T22_fit_phase - T22_phase)**2))
    T22_complex_rmse = np.sqrt(np.mean(np.abs(T22_fit - T22_complex)**2))
    
    # Create enhanced info dict
    info_enhanced = info_I.copy()
    info_enhanced.update({
        'fitted_domain': 'I (converted from T22)',
        'T22_mag_rmse': T22_mag_rmse,
        'T22_phase_rmse': T22_phase_rmse,
        'T22_complex_rmse': T22_complex_rmse,
        'T_ratio': T_ratio,
        'conversion_method': 'I_domain_workaround',
        'I_domain_cost': info_I['cost'],
        'to_T22': lambda omega_eval, T_ratio_eval: 1.0 + (T_ratio_eval - 1.0) * I_two_delay(omega_eval, p_fit, m_phi, m_t)
    })
    
    return p_fit, info_enhanced

def fit_two_delay_gamma(omega, mag, phase, *,
                        phase_in_degrees=False,
                        m_phi: int = 2, m_t: int = 2,
                        normalize: bool = False, T_ratio=None,
                        w_mag: float = 1.0, w_phase: float = 1.0,
                        r_min: float = 0.5, r0: float = 0.9, lambda_r: float = 5e-3,
                        robust: Optional[str] = "soft_l1", f_scale: float = 1.0,
                        max_nfev: int = 50000,
                        multi_start: bool = True, n_starts: int = 5,
                        use_complex_residual: bool = False,
                        use_I_domain_workaround: bool = True,
                        suppress_warnings: bool = False) -> Tuple[TwoDelayParams, Dict]:
    """
    Fit Two-Delay Gamma FTF model parameters to frequency response data.
    
    Parameters:
    -----------
    omega : array_like
        Angular frequency array [rad/s]
    mag : array_like
        Magnitude data
    phase : array_like
        Phase data [rad] (or degrees if phase_in_degrees=True)
    phase_in_degrees : bool, default False
        Whether phase data is in degrees
    m_phi, m_t : int, default 2
        Integer shape parameters for Gamma distributions
    normalize : bool, default False
        Data domain interpretation:
        - False: mag/phase are for (T2/T1 - 1) * I(ω) (i.e., T22 - 1)
        - True: mag/phase are T22 data, requires T_ratio for normalization
    T_ratio : float, optional
        Temperature ratio T2/T1, required when normalize=True
    w_mag, w_phase : float, default 1.0
        Relative weights for magnitude and phase residuals
    r_min : float, default 0.5
        Lower bound for r_tau = tau_t/tau_phi ratio
    r0 : float, default 0.9
        Target value for r_tau regularization
    lambda_r : float, default 5e-3
        Regularization strength for r_tau
    robust : str, default "soft_l1"
        Robust loss function: "linear", "soft_l1", "huber", "cauchy"
    f_scale : float, default 1.0
        Scale parameter for robust loss
    max_nfev : int, default 50000
        Maximum function evaluations
    multi_start : bool, default True
        Use multi-start optimization to avoid local minima
    n_starts : int, default 5
        Number of random starts when multi_start=True
    use_complex_residual : bool, default False
        Use complex residuals instead of separate mag/phase residuals
    use_I_domain_workaround : bool, default True
        For normalize=True, use I domain fitting workaround (recommended)
    suppress_warnings : bool, default False
        If True, suppress NumPy/SciPy runtime warnings during fitting (overflow,
        divide-by-zero, invalid operations, and SciPy Optimize warnings).
        
    Returns:
    --------
    params : TwoDelayParams
        Fitted parameters
    info : dict
        Fitting information including success, cost, RMSE metrics
    """
    # sort
    idx = np.argsort(omega)
    w   = np.asarray(omega, float)[idx]
    M   = np.asarray(mag,   float)[idx]
    ph  = np.asarray(phase, float)[idx]
    n   = len(w)

    if normalize:
        if T_ratio is None:
            raise ValueError("normalize=True requires T_ratio (T2/T1).")
        
        if use_I_domain_workaround:
            # Use I domain workaround to bypass tau_phi optimization issues
            return _fit_T22_via_I_domain(w, M, ph, T_ratio, phase_in_degrees, m_phi, m_t, 
                                       w_mag, w_phase, r_min, r0, lambda_r, robust, f_scale, 
                                       max_nfev, multi_start, n_starts, use_complex_residual,
                                       suppress_warnings=suppress_warnings)
        else:
            # Fixed normalization approach
            Tr = np.asarray(T_ratio, float)
            Tr = np.full_like(w, float(Tr)) if Tr.size==1 else Tr[idx]
            T22_complex = magphase_to_complex(M, ph, phase_in_degrees)   # T22 data
            I_c = (T22_complex - 1.0) / (Tr - 1.0 + 1e-12)  # Correct: I = (T22 - 1) / (T_ratio - 1)
            M_use  = np.abs(I_c)
            phi_unw= unwrap_phase(np.angle(I_c), deg=False)
            fitted_domain = "I (normalized)"
    else:
        M_use  = M
        phi_unw= unwrap_phase(ph, deg=phase_in_degrees)
        fitted_domain = "(T2/T1 - 1) * I"

    # weights
    w_mag_arr   = as_weight_array(w_mag, n)
    w_phase_arr = as_weight_array(w_phase, n)

    if multi_start and n_starts > 1:
        # Multi-start optimization to avoid local minima
        best_sol = None
        best_cost = np.inf
        best_p_hat = None
        
        for start_idx in range(n_starts):
            if start_idx == 0:
                # Use standard initialization for first attempt
                p0 = init_two_delay(w, M_use, phi_unw, m_phi, m_t)
            else:
                # Random initialization for subsequent attempts
                np.random.seed(start_idx + 42)  # Reproducible but different seeds
                # Estimate reasonable ranges from data
                mag_scale = np.mean(M_use)
                p0 = TwoDelayParams(
                    A_phi=mag_scale * np.random.uniform(0.3, 1.5),
                    A_t=mag_scale * np.random.uniform(0.3, 1.5),
                    tau_phi=np.random.uniform(0.001, 0.01),
                    r_tau=np.random.uniform(0.3, 1.0),
                    theta_phi=np.random.uniform(0.005, 0.03),
                    theta_t=np.random.uniform(0.005, 0.03)
                )
            
            x0 = pack_uncon(p0, r_min=r_min)

            # mag-only prefit
            def res_mag_only(x):
                p = unpack_uncon(x, r_min=r_min)
                I = I_two_delay(w, p, m_phi, m_t)
                return np.sqrt(w_mag_arr) * (np.log(M_use+1e-12) - np.log(np.abs(I)+1e-12))
            
            try:
                with _maybe_suppress_warnings(suppress_warnings):
                    x1 = least_squares(res_mag_only, x0, method="trf", loss=robust,
                                       f_scale=f_scale, max_nfev=max_nfev//2).x

                    # joint fit with penalty
                    if use_complex_residual:
                        sol = least_squares(residuals_complex, x1,
                                            args=(w, M_use, phi_unw, m_phi, m_t,
                                                  w_mag_arr, w_phase_arr, r_min, r0, lambda_r),
                                            method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
                    else:
                        sol = least_squares(residuals_joint, x1,
                                            args=(w, M_use, phi_unw, m_phi, m_t,
                                                  w_mag_arr, w_phase_arr, r_min, r0, lambda_r),
                                            method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
                
                if sol.success and sol.cost < best_cost:
                    best_sol = sol
                    best_cost = sol.cost
                    best_p_hat = unpack_uncon(sol.x, r_min=r_min)
                    
            except Exception:
                continue  # Skip failed attempts
        
        if best_sol is None:
            # Fallback to single attempt if all multi-starts failed
            p0 = init_two_delay(w, M_use, phi_unw, m_phi, m_t)
            x0 = pack_uncon(p0, r_min=r_min)
            def res_mag_only(x):
                p = unpack_uncon(x, r_min=r_min)
                I = I_two_delay(w, p, m_phi, m_t)
                return np.sqrt(w_mag_arr) * (np.log(M_use+1e-12) - np.log(np.abs(I)+1e-12))
            with _maybe_suppress_warnings(suppress_warnings):
                x1 = least_squares(res_mag_only, x0, method="trf", loss=robust,
                                   f_scale=f_scale, max_nfev=max_nfev//2).x
                if use_complex_residual:
                    sol = least_squares(residuals_complex, x1,
                                        args=(w, M_use, phi_unw, m_phi, m_t,
                                              w_mag_arr, w_phase_arr, r_min, r0, lambda_r),
                                        method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
                else:
                    sol = least_squares(residuals_joint, x1,
                                        args=(w, M_use, phi_unw, m_phi, m_t,
                                              w_mag_arr, w_phase_arr, r_min, r0, lambda_r),
                                        method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
            p_hat = unpack_uncon(sol.x, r_min=r_min)
        else:
            # Use the best multi-start solution
            sol = best_sol
            p_hat = best_p_hat
    else:
        # Single-start optimization (original behavior)
        p0 = init_two_delay(w, M_use, phi_unw, m_phi, m_t)
        x0 = pack_uncon(p0, r_min=r_min)

        # mag-only prefit
        def res_mag_only(x):
            p = unpack_uncon(x, r_min=r_min)
            I = I_two_delay(w, p, m_phi, m_t)
            return np.sqrt(w_mag_arr) * (np.log(M_use+1e-12) - np.log(np.abs(I)+1e-12))
        with _maybe_suppress_warnings(suppress_warnings):
            x1 = least_squares(res_mag_only, x0, method="trf", loss=robust,
                               f_scale=f_scale, max_nfev=max_nfev//2).x

            # joint fit with penalty
            if use_complex_residual:
                sol = least_squares(residuals_complex, x1,
                                    args=(w, M_use, phi_unw, m_phi, m_t,
                                          w_mag_arr, w_phase_arr, r_min, r0, lambda_r),
                                    method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
            else:
                sol = least_squares(residuals_joint, x1,
                                    args=(w, M_use, phi_unw, m_phi, m_t,
                                          w_mag_arr, w_phase_arr, r_min, r0, lambda_r),
                                    method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
        p_hat = unpack_uncon(sol.x, r_min=r_min)
    
    I_fit = I_two_delay(w, p_hat, m_phi, m_t)

    # plain SSE (no robust) for scoring
    r_mag_plain   = (np.log(M_use+1e-12) - np.log(np.abs(I_fit)+1e-12))
    r_phase_plain = wrap_pi(phi_unw - np.angle(I_fit))
    sse = float(np.sum((np.sqrt(w_mag_arr)*r_mag_plain)**2 + (np.sqrt(w_phase_arr)*r_phase_plain)**2))

    info = {
        "success": bool(sol.success),
        "message": sol.message,
        "nfev": int(sol.nfev),
        "cost": float(sol.cost),
        "fitted_domain": fitted_domain,
        "logmag_rmse": float(np.sqrt(np.mean(r_mag_plain**2))),
        "phase_rmse_rad": float(np.sqrt(np.mean(r_phase_plain**2))),
        "sse": sse,
        "params": p_hat.__dict__
    }

    # reconstructor to T22 if needed
    info["to_T22"] = lambda omega_eval, T_ratio_eval: \
        1.0 + (np.asarray(T_ratio_eval, float) - 1.0) * I_two_delay(np.asarray(omega_eval, float), p_hat, m_phi, m_t)

    return p_hat, info

# ---------------- Grid search over (m_phi, m_t) ----------------
def fit_two_delay_gamma_grid(omega, mag, phase, *,
                             phase_in_degrees=False,
                             mphi_list: Optional[Sequence[int]] = None,
                             mt_list: Optional[Sequence[int]] = None,
                             normalize: bool = False, T_ratio=None,
                             w_mag=1.0, w_phase=1.0,
                             r_min=0.5, r0=0.9, lambda_r=1e-2,
                             robust="soft_l1", f_scale=1.0,
                             selection: str = "rmse",   # "rmse" or "aic"
                             max_nfev=20000,
                             suppress_warnings: bool = False):
    """
    Try multiple integer shapes and pick the best by 'rmse' (default) or 'aic'.
    Defaults: m_phi, m_t in 1..8.
    """
    if mphi_list is None: mphi_list = list(range(1, 9))
    if mt_list   is None: mt_list   = list(range(1, 9))
    results = []
    best = None

    n = len(np.asarray(omega))
    # number of estimated continuous parameters = 6 (A_phi, A_t, tau_phi, r_tau, theta_phi, theta_t)
    k_params = 6

    for mp in mphi_list:
        for mt in mt_list:
            p, info = fit_two_delay_gamma(
                omega, mag, phase,
                phase_in_degrees=phase_in_degrees,
                m_phi=mp, m_t=mt,
                normalize=normalize, T_ratio=T_ratio,
                w_mag=w_mag, w_phase=w_phase,
                r_min=r_min, r0=r0, lambda_r=lambda_r,
                robust=robust, f_scale=f_scale,
                max_nfev=max_nfev,
                suppress_warnings=suppress_warnings
            )
            # score
            if selection.lower() == "aic":
                # AIC with Gaussian residual assumption on concatenated residuals
                # N_eff ~ 2n (mag+phase); use sse reported
                N_eff = 2*n
                aic = 2*k_params + N_eff*np.log((info["sse"]+1e-18)/N_eff)
                score = aic
            else:
                # combined RMSE (simple)
                score = info["logmag_rmse"] + info["phase_rmse_rad"]
            rec = {"m_phi": mp, "m_t": mt, "score": float(score), "info": info, "params": p}
            results.append(rec)
            if (best is None) or (score < best["score"]):
                best = rec

    return best, results

# ---------------- Convenience: make a callable ----------------
def make_I_callable(p: TwoDelayParams, m_phi: int, m_t: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda omega: I_two_delay(np.asarray(omega, float), p, m_phi, m_t)

# Example usage (commented out - see ftf_demo.py for working example):
# Your data: omega [rad/s], mag, phase for the DEFAULT domain (T22 - 1) = (T2/T1 - 1) * I
# If you have |T22| & phase(T22): set normalize=True and pass T_ratio.
#
# best, grid = fit_two_delay_gamma_grid(
#     omega, mag, phase,
#     phase_in_degrees=False,
#     mphi_list=range(1, 9),      # or e.g. [1,2,3,4,6,8]
#     mt_list=range(1, 9),
#     normalize=False,            # default domain is (T22-1)
#     # normalize=True, T_ratio=T2_over_T1,   # <- for raw T22 instead
#     w_mag=1.0, w_phase=1.0,
#     r_min=0.5, r0=0.9, lambda_r=5e-3,      # turbulence delay penalty (tuned)
#     selection="rmse",                      # or "aic"
# )
#
# print("Best shapes:", best["m_phi"], best["m_t"])
# print("Fit RMSE (log-mag, phase):",
#       best["info"]["logmag_rmse"], best["info"]["phase_rmse_rad"])
#

@dataclass
class FuelSplitConfig:
    """
    Configuration for mixing pilot and main flames via a fuel split α ∈ [0,1].
    - Logistic weight parameters (s0, s1, s2) define w_pilot(α) = 1/(1+exp(-(s0+s1 α + s2 α^2)))
    - Small alignment delays dtau_pilot/main shift only phases of each stage
    - Optional cross-term X with strength kappa and delay tau_c
    """
    s0: float = 0.0
    s1: float = 4.0
    s2: float = 0.0
    dtau_pilot: float = 0.0
    dtau_main: float = 0.0
    kappa: float = 0.0
    tau_c: float = 0.0

def _weight_from_split(alpha: float, s0: float, s1: float, s2: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    z = s0 + s1 * a + s2 * (a**2)
    return float(1.0 / (1.0 + np.exp(-z)))

def fuel_split_I(
    omega: np.ndarray,
    fuel_split: float,
    pilot: TwoDelayParams,
    main: TwoDelayParams,
    *,
    m_pilot_phi: int = 2,
    m_pilot_t: int = 2,
    m_main_phi: int = 2,
    m_main_t: int = 2,
    cfg: FuelSplitConfig = FuelSplitConfig()
) -> np.ndarray:
    """
    Mixed interaction-index FTF for a given fuel split α.
    Uses existing `I_two_delay()` to avoid code duplication.
    """
    w_p = _weight_from_split(fuel_split, cfg.s0, cfg.s1, cfg.s2)
    w_m = 1.0 - w_p

    I_p = I_two_delay(omega, pilot, m_pilot_phi, m_pilot_t)
    I_m = I_two_delay(omega, main,  m_main_phi,  m_main_t)

    w = np.asarray(omega, float)
    phase_p = np.exp(-1j * w * cfg.dtau_pilot)
    phase_m = np.exp(-1j * w * cfg.dtau_main)

    P = w_p * phase_p * I_p + w_m * phase_m * I_m
    if cfg.kappa != 0.0:
        X = cfg.kappa * np.exp(-1j * w * cfg.tau_c) * (w_p * I_p) * (w_m * I_m)
        return P + X
    return P

def T22_from_fuel_split(
    omega: np.ndarray,
    fuel_split: float,
    pilot: TwoDelayParams,
    main: TwoDelayParams,
    T_ratio: float,
    *,
    m_pilot_phi: int = 2,
    m_pilot_t: int = 2,
    m_main_phi: int = 2,
    m_main_t: int = 2,
    cfg: FuelSplitConfig = FuelSplitConfig()
) -> np.ndarray:
    """
    Raw burner transfer T22(ω, α) from the mixed interaction index:
        T22 = 1 + (T2/T1 - 1) * I_mix(ω, α)
    """
    I_mix = fuel_split_I(
        omega, fuel_split, pilot, main,
        m_pilot_phi=m_pilot_phi, m_pilot_t=m_pilot_t,
        m_main_phi=m_main_phi, m_main_t=m_main_t,
        cfg=cfg
    )
    return 1.0 + (float(T_ratio) - 1.0) * I_mix

# ===================== FTF Toolbox (from ftf_patch.py) =====================
# The following section integrates additional FTF models and simple fitting
# utilities, while preserving the existing Two-Delay Gamma APIs above.

ComplexTF = Callable[[np.ndarray], np.ndarray]

# -------------------------- FTF Library --------------------------
def ftf_diffusion(n: float = 0.12, tau: float = 6e-3, fc: float = 120.0, order: int = 1) -> ComplexTF:
    """Diffusion-like FTF: n * e^{-i ω τ} * (1 + i ω/ωc)^(-order)"""
    wc = 2*np.pi*fc
    def F(f: np.ndarray) -> np.ndarray:
        w = 2*np.pi*f
        lp = 1.0/(1.0 + 1j*w/wc)
        if order == 2:
            lp = lp**2
        return n * np.exp(-1j*w*tau) * lp
    return F

def ftf_gamma(n: float = 0.8, nu: float = 3.0, theta_c: float = 2e-3) -> ComplexTF:
    """Gamma delay kernel (Lieuwen): n * (1 + i ω θ_c)^(-ν)"""
    def F(f: np.ndarray) -> np.ndarray:
        w = 2*np.pi*f
        return n * (1.0 + 1j*w*theta_c)**(-nu)
    return F

def ftf_dispersion(n: float = 0.3, tau: float = 5e-3, wd: float = 2*np.pi*200.0, k: float = 2.0) -> ComplexTF:
    """Parmentier-style mixing/dispersion: n * e^{-i ω τ} * exp(-(ω/wd)^k)"""
    def F(f: np.ndarray) -> np.ndarray:
        w = 2*np.pi*f
        return n * np.exp(-1j*w*tau) * np.exp(- (w/wd)**k )
    return F

def ftf_rational(num0: float = 0.1, num1: float = 0.0, den1: float = 1e-3, den2: float = 0.0) -> ComplexTF:
    """Low-order rational: (num0 + num1 iω) / (1 + den1 iω + den2 (iω)^2)"""
    def F(f: np.ndarray) -> np.ndarray:
        iw = 1j*2*np.pi*f
        return (num0 + num1*iw) / (1.0 + den1*iw + den2*iw**2)
    return F

def ftf_n_tau_gauss_opposite(
    n: float = 0.5, 
    tau_conv: float = 3e-3, sigma_conv: float = 1e-3,
    tau_turb: float = 2e-3, sigma_turb: float = 0.8e-3,
    beta: float = 0.5
) -> ComplexTF:
    """
    Two-pathway FTF with Gaussian-distributed delays and opposite signs.
    
    H(ω) = n[β E[e^{-iω τ_c}] + (1-β) E[e^{+iω τ_t}]]
    
    where τ_c ~ N(tau_conv, sigma_conv²), τ_t ~ N(tau_turb, sigma_turb²)
    and E[e^{±iωτ}] = exp(±iωμ - 0.5(ωσ)²)
    
    Parameters
    ----------
    n : float
        Overall amplitude
    tau_conv : float
        Mean convective delay [s] (negative contribution)
    sigma_conv : float  
        Std deviation of convective delay [s]
    tau_turb : float
        Mean turbulent delay [s] (positive contribution)
    sigma_turb : float
        Std deviation of turbulent delay [s]
    beta : float
        Weight of convective pathway (0 ≤ β ≤ 1)
    """
    def F(f: np.ndarray) -> np.ndarray:
        ω = 2*np.pi*np.asarray(f, dtype=float)
        conv = np.exp(-1j*ω*tau_conv - 0.5*(ω*sigma_conv)**2)
        turb = np.exp(+1j*ω*tau_turb - 0.5*(ω*sigma_turb)**2)
        return n * (beta*conv + (1.0-beta)*turb)
    return F

# -------- MISO pilot (velocity + phi with an upstream mixing TF) --------
@dataclass
class MISOParams:
    # velocity-path (diffusion-like)
    n_u: float = 0.12
    tau_u: float = 6e-3
    fc_u: float = 120.0
    # equivalence-ratio path (premixed-like)
    n_phi: float = 0.6
    nu_phi: float = 3.0
    theta_phi: float = 2e-3
    # mixing TF between ϕ fluctuations and flame
    tau_mix: float = 2e-3
    wd_mix: float = 2*np.pi*300.0
    k_mix: float = 2.0

def ftf_miso_pilot(params: MISOParams) -> Dict[str, ComplexTF]:
    """Returns dict with Fu (velocity), Fphi (ϕ), and Mix TF."""
    Fu   = ftf_diffusion(n=params.n_u, tau=params.tau_u, fc=params.fc_u, order=1)
    Fphi = ftf_gamma(n=params.n_phi, nu=params.nu_phi, theta_c=params.theta_phi)
    Mix  = ftf_dispersion(n=1.0, tau=params.tau_mix, wd=params.wd_mix, k=params.k_mix)
    return {"Fu": Fu, "Fphi": Fphi, "Mix": Mix}

def evaluate_miso(f: np.ndarray, Fu: ComplexTF, Fphi: ComplexTF, Mix: ComplexTF,
                  u_hat: complex = 1.0, phi_hat: complex = 1.0) -> np.ndarray:
    """Output = Fu*u' + Fphi*Mix*phi'."""
    return Fu(f)*u_hat + Fphi(f)*Mix(f)*phi_hat

# ---------------------- Autoignition (zonal) ----------------------
def ftf_autoignition_zonal(
    w_ai: float = 0.25,
    n_prop: float = 0.5, nu_prop: float = 2.5, theta_prop: float = 2e-3, tau_prop: float = 0.0,
    n_ai: float = 0.4,  nu_ai: float = 1.2, theta_ai: float = 3e-3, tau_ai: float = 2e-3
) -> ComplexTF:
    """
    Zonal superposition: F = (1-w_ai)*F_prop + w_ai*F_AI
    Simplified single-input form; for multi-input see MISO patterns.
    """
    F_prop = lambda f: ftf_gamma(n=n_prop, nu=nu_prop, theta_c=theta_prop)(f) * np.exp(-1j*2*np.pi*f*tau_prop)
    F_ai   = lambda f: ftf_gamma(n=n_ai,  nu=nu_ai,  theta_c=theta_ai)(f)  * np.exp(-1j*2*np.pi*f*tau_ai)
    def F(f: np.ndarray) -> np.ndarray:
        return (1.0 - w_ai)*F_prop(f) + w_ai*F_ai(f)
    return F

# ------------------------- Ordered Mixer -------------------------
def mixer_ordered(F_pilot: ComplexTF, F_main: ComplexTF, alpha: float, beta: float = 1.0, normalized: bool = True) -> ComplexTF:
    """
    Ordered blend pilot→main with fuel split alpha (main share). Returns callable F(f).
    """
    w1 = (1.0 - alpha)**beta
    w2 = (alpha)**beta
    if normalized:
        s = w1 + w2
        if s == 0: s = 1.0
        w1, w2 = w1/s, w2/s
    def F(f: np.ndarray) -> np.ndarray:
        return w1*F_pilot(f) + w2*F_main(f)
    return F

# -------------------------- Registries ---------------------------
ModelFactory = Callable[..., ComplexTF]
RegistryItem = Tuple[str, Callable[[np.ndarray], np.ndarray], List[str], np.ndarray, np.ndarray]

REGISTRY: Dict[int, RegistryItem] = {
    0: ("diffusion_lp1", lambda p: ftf_diffusion(p[0], p[1], p[2], order=1),
        ["n","tau_s","fc_hz"],
        np.array([0.00, 0.0005,   20.0]),
        np.array([2.00, 0.0500, 2000.0])),
    1: ("gamma",        lambda p: ftf_gamma(p[0], p[1], p[2]),
        ["n","nu","theta_c_s"],
        np.array([0.00,  0.5,     0.0003]),
        np.array([2.00,  8.0,     0.0200])),
    2: ("dispersion",   lambda p: ftf_dispersion(p[0], p[1], p[2], p[3]),
        ["n","tau_s","wd_rad","k"],
        np.array([0.00,  0.0005,  2*np.pi* 50.0, 1.0]),
        np.array([2.00,  0.0500,  2*np.pi*3000.0, 4.0])),
    3: ("rational_2p",  lambda p: ftf_rational(p[0], p[1], p[2], p[3]),
        ["num0","num1","den1","den2"],
        np.array([-2.0, -0.05, -0.02,  -1e-5]),
        np.array([ 2.0,  0.05,  0.02,   1e-3])),
    4: ("autoignition_zonal", lambda p: ftf_autoignition_zonal(
            w_ai=p[0],
            n_prop=p[1], nu_prop=p[2], theta_prop=p[3], tau_prop=p[4],
            n_ai=p[5],   nu_ai=p[6],  theta_ai=p[7],  tau_ai=p[8]),
        ["w_ai","prop.n","prop.nu","prop.theta_c_s","prop.tau_s","ai.n","ai.nu","ai.theta_c_s","ai.tau_s"],
        np.array([0.00,  0.00,  0.5,   0.0003,  0.0,    0.00, 0.5,  0.0003,  0.0]),
        np.array([1.00,  2.00,  8.0,   0.0200,  0.050,  2.00, 8.0,  0.0200,  0.050])),
    5: ("gauss_opposite", lambda p: ftf_n_tau_gauss_opposite(
            n=p[0], tau_conv=p[1], sigma_conv=p[2], 
            tau_turb=p[3], sigma_turb=p[4], beta=p[5]),
        ["n","tau_conv_s","sigma_conv_s","tau_turb_s","sigma_turb_s","beta"],
        np.array([0.00,  0.0005,  0.0001,  0.0005,  0.0001,  0.0]),
        np.array([2.00,  0.0200,  0.0050,  0.0200,  0.0050,  1.0])),
}

ComboKey   = Tuple[int,int]
ComboItem  = Tuple[str, Callable[[np.ndarray], np.ndarray], List[str], np.ndarray, np.ndarray]

def build_combo_registry(candidate_ids: List[int], skip_identical: bool = True, beta: float = 1.0) -> Dict[ComboKey, ComboItem]:
    combos: Dict[ComboKey, ComboItem] = {}
    for a in candidate_ids:
        for b in candidate_ids:
            if skip_identical and (a == b):
                continue
            name_a, fa, pa, la, ha = REGISTRY[a]
            name_b, fb, pb, lb, hb = REGISTRY[b]
            pname = [f"pilot.{x}" for x in pa] + [f"main.{x}" for x in pb] + ["alpha"]
            low   = np.concatenate([la, lb, np.array([0.0])])
            high  = np.concatenate([ha, hb, np.array([1.0])])

            def factory_builder(a=a, b=b, fa=fa, fb=fb, beta=beta):
                pa_names = REGISTRY[a][2]
                pb_names = REGISTRY[b][2]
                na, nb = len(pa_names), len(pb_names)
                def factory(pvec: np.ndarray) -> ComplexTF:
                    pa_ = pvec[:na]
                    pb_ = pvec[na:na+nb]
                    alpha = float(pvec[-1])
                    Fp = fa(pa_)
                    Fm = fb(pb_)
                    return mixer_ordered(Fp, Fm, alpha=alpha, beta=beta, normalized=True)
                return factory

            combos[(a,b)] = (f"pilot:{name_a} | main:{name_b}",
                             factory_builder(),
                             pname, low, high)
    return combos

# --------------------------- Fitting -----------------------------
def _unwrap_phase_ftf(z: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(z))

def loss_complex(Fm: np.ndarray, Fd: np.ndarray, w_mag: float = 1.0, w_ph: float = 1.0) -> float:
    if w_ph == 0.0:
        return float(np.mean(np.abs(Fm - Fd)**2))
    mag_m, mag_d = np.abs(Fm), np.abs(Fd)
    ph_m,  ph_d  = _unwrap_phase_ftf(Fm), _unwrap_phase_ftf(Fd)
    mscale = max(np.sqrt(np.mean(mag_d**2)), 1e-12)
    return float(w_mag*np.mean((mag_m - mag_d)**2)/mscale**2 + w_ph*np.mean((ph_m - ph_d)**2)/np.pi**2)

@dataclass
class FitResult:
    key: Tuple[int,int] | None
    name: str
    params: np.ndarray
    param_names: List[str]
    loss: float

def fit_one_model(f: np.ndarray, F_meas: np.ndarray, item: RegistryItem,
                  n_starts: int = 256, seed: int = 0, w_mag: float = 1.0, w_ph: float = 1.0) -> FitResult:
    name, factory, pnames, low, high = item
    rng = np.random.default_rng(seed)
    best_L, best_p = np.inf, None
    P = low.size
    for k in range(n_starts):
        u = (k + rng.random(P)) / n_starts
        p = low + u*(high - low)
        F = factory(p)(f)
        L = loss_complex(F, F_meas, w_mag, w_ph)
        if L < best_L:
            best_L, best_p = L, p
    return FitResult(key=None, name=name, params=best_p, param_names=pnames, loss=best_L)

def fit_ftf_with_model_choice(f: np.ndarray, F_meas: np.ndarray,
                              candidate_ids: List[int] = (0,1,2,3,4),
                              n_starts_per_model: int = 256, seed: int = 42,
                              w_mag: float = 1.0, w_ph: float = 1.0) -> FitResult:
    rng = np.random.default_rng(seed)
    best = None
    for mid in candidate_ids:
        item = REGISTRY[mid]
        fr = fit_one_model(f, F_meas, item,
                           n_starts=n_starts_per_model,
                           seed=int(rng.integers(0, 1_000_000)),
                           w_mag=w_mag, w_ph=w_ph)
        if (best is None) or (fr.loss < best.loss):
            best = fr
    return best

def fit_combo(f: np.ndarray, F_meas: np.ndarray, combo_item: ComboItem,
              n_starts: int = 256, seed: int = 0, w_mag: float = 1.0, w_ph: float = 1.0) -> FitResult:
    name, factory_builder, pnames, low, high = combo_item
    rng = np.random.default_rng(seed)
    bestL, bestp = np.inf, None
    P = low.size
    for k in range(n_starts):
        u = (k + rng.random(P)) / n_starts
        p = low + u*(high - low)
        F = factory_builder(p)(f)
        L = loss_complex(F, F_meas, w_mag, w_ph)
        if L < bestL:
            bestL, bestp = L, p
    return FitResult(key=None, name=name, params=bestp, param_names=pnames, loss=bestL)

def fit_over_combos(f: np.ndarray, F_meas: np.ndarray, combo_registry: Dict[ComboKey, ComboItem],
                    n_starts_per_combo: int = 256, seed: int = 42, w_mag: float = 1.0, w_ph: float = 1.0) -> FitResult:
    rng = np.random.default_rng(seed)
    best = None
    for key, item in combo_registry.items():
        fr = fit_combo(f, F_meas, item,
                       n_starts=n_starts_per_combo,
                       seed=int(rng.integers(0,1_000_000)),
                       w_mag=w_mag, w_ph=w_ph)
        if (best is None) or (fr.loss < best.loss):
            fr.key = key
            best = fr
    return best

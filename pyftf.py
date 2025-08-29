import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
from scipy.optimize import least_squares

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.optimize import least_squares
import matplotlib.pyplot as plt



# -------------------- DTL FTF model --------------------
# Polifke, W. (2020). System identification of combustion dynamics by means of CFD/LES: FTF, FDF and DTL. Progress in Energy and Combustion Science 79: 100845.
# Comprehensive review of flame transfer function identification; shows why single delays fail, and how Gamma/Gaussian distributed delays reproduce measured gain/phase.
# 	•	Schuller, T., Noiray, N., Poinsot, T., Candel, S. (2020). Dynamics of premixed flames and combustion instabilities. J. Fluid Mech. 894, P1.
# Overview of flame dynamics theory; highlights the role of \rho c normalization and convective delay distributions.
# 	•	Lieuwen, T. (2012). Unsteady Combustor Physics. Cambridge University Press.
# Standard reference; derives n–τ, extended n–τ, and motivates distributed-lag generalizations.
# 	•	Noiray, N. & Schuermans, B. (2013). Deterministic quantities characterizing noise driven Hopf bifurcations in gas turbine combustors. International Journal of Non-Linear Mechanics 50, 152–163.
# While focused on noise-driven oscillations, uses distributed-delay style kernels to connect linear and nonlinear response.

@dataclass
class DTLParams:
    """Distributed Time Lag (DTL) FTF parameters.
    
    Args:
        S: Gain scale (dimensionless, > 0)
        tau0: Bulk delay [s] (>= 0)
        thetas: Time widths [s] (J>0 distributed convective lags)
        ks: Shape exponents (dimensionless, J>0)
    """
    S: float
    tau0: float
    thetas: np.ndarray
    ks: np.ndarray

def H_dtl(omega: np.ndarray, p: DTLParams) -> np.ndarray:
    """Distributed Time Lag (DTL) flame transfer function.
    
    H(ω) = S * exp(-i ω τ₀) * ∏_j (1 + i ω θ_j)^(-k_j)
    
    Models distribution of convective delays for smooth low-/band-pass behavior.
    Use when magnitude decays monotonically without rising behavior.
    
    Args:
        omega: Angular frequency [rad/s]
        p: DTL parameters
        
    Returns:
        Complex FTF values
    """
    H = p.S * np.exp(-1j * omega * p.tau0)
    for theta, k in zip(p.thetas, p.ks):
        H *= (1.0 + 1j * omega * theta) ** (-k)
    return H

def calculate_ftf_dtl(omega: np.ndarray, params: DTLParams) -> np.ndarray:
    """Calculate DTL flame transfer function from parameters.
    
    Convenience wrapper for H_dtl() with clear naming.
    
    Args:
        omega: Angular frequency [rad/s]
        params: DTL parameters
        
    Returns:
        Complex FTF values H(ω)
    """
    return H_dtl(omega, params)

# -------------------- Phase handling --------------------

def unwrap_phase(phi: np.ndarray, deg: bool = False) -> np.ndarray:
    """Unwrap phase to remove 2π discontinuities.
    
    Args:
        phi: Phase data [rad] or [deg]
        deg: True if input is in degrees
        
    Returns:
        Unwrapped phase [rad]
    """
    if deg:
        phi = np.deg2rad(phi)
    return np.unwrap(phi)

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap phase to (-π, π] interval.
    
    Args:
        x: Phase values [rad]
        
    Returns:
        Wrapped phase [rad] in (-π, π]
    """
    return (x + np.pi) % (2*np.pi) - np.pi

# -------------------- Parameterization & init --------------------

def pack_params_uncon(p: DTLParams) -> np.ndarray:
    """Pack DTL parameters to unconstrained optimization vector.
    
    Maps physical parameters to R^n using log/inv_softplus transforms
    to enforce positivity constraints during optimization.
    
    Args:
        p: DTL parameters
        
    Returns:
        Unconstrained parameter vector
    """
    def inv_softplus(y):  # approx inverse for y>0
        return np.log(np.expm1(y))
    v = []
    v.append(np.log(p.S))                        # S = exp(sS)
    v.append(inv_softplus(p.tau0 + 1e-12))       # tau0 = softplus(tau0_raw)
    v.extend(inv_softplus(p.thetas + 1e-12))     # theta_j = softplus(...)
    v.extend(inv_softplus(p.ks + 1e-12))         # k_j = softplus(...)
    return np.array(v, dtype=float)

def unpack_params_uncon(v: np.ndarray, J: int) -> DTLParams:
    """Unpack unconstrained vector to DTL parameters.
    
    Inverse of pack_params_uncon. Enforces positivity via exp/softplus.
    
    Args:
        v: Unconstrained parameter vector
        J: Number of poles
        
    Returns:
        DTL parameters with positivity enforced
    """
    def softplus(x): return np.log1p(np.exp(x))
    S = np.exp(v[0])
    tau0 = softplus(v[1])
    thetas = softplus(v[2:2+J])
    ks = softplus(v[2+J:2+2*J])
    return DTLParams(S=S, tau0=tau0, thetas=thetas, ks=ks)

def heuristic_init(omega: np.ndarray, mag: np.ndarray, phase_unw: np.ndarray, J: int) -> DTLParams:
    """Generate initial DTL parameter guess from FTF data.
    
    Uses robust heuristics:
    - S from low-frequency magnitude median
    - tau0 from low-frequency phase slope
    - theta from 3dB corner frequency
    - k ≈ 1 (distributed across poles)
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Magnitude data (linear scale)
        phase_unw: Unwrapped phase [rad]
        J: Number of poles (1 or 2)
        
    Returns:
        Initial DTL parameters
    """
    # low-f region: first 10–20% points
    n = max(3, int(0.15 * len(omega)))
    om_lo = omega[:n]
    mag_lo = mag[:n]
    S0 = np.median(mag_lo.clip(min=1e-6))

    # tau0 from linear fit of phase vs omega near zero
    A = np.vstack([om_lo, np.ones_like(om_lo)]).T
    m_slope, _ = np.linalg.lstsq(A, phase_unw[:n], rcond=None)[0]
    tau0_0 = float(np.clip(-m_slope, 0.0, 0.1/ (omega[1]-omega[0] + 1e-12)))  # keep sane

    # crude corner: ω_c where |H| drops to (1/√2)*low-f median
    target = S0 / np.sqrt(2.0)
    idx = np.where(mag <= target)[0]
    if idx.size > 0:
        w_c = omega[idx[0]]
        theta1 = 1.0 / max(w_c, 1e-6)
    else:
        theta1 = 1.0 / max(omega[len(omega)//3], 1e-6)

    if J == 1:
        thetas = np.array([theta1], float)
        ks = np.array([1.0], float)
    else:
        thetas = np.array([theta1, 3.0*theta1], float)
        ks = np.array([0.8, 0.6], float)

    return DTLParams(S=S0, tau0=max(tau0_0, 0.0), thetas=thetas, ks=ks)

# -------------------- Objective & fitting --------------------

def residuals_uncon(v: np.ndarray,
                    omega: np.ndarray,
                    mag: np.ndarray,
                    phase_unw: np.ndarray,
                    J: int,
                    w_mag: float,
                    w_phase: float) -> np.ndarray:
    p = unpack_params_uncon(v, J)
    Hm = H_dtl(omega, p)
    logmag_res = np.log(mag + 1e-12) - np.log(np.abs(Hm) + 1e-12)
    # phase residual: compare unwrapped measured to model phase, wrap the difference to avoid 2π jumps in residual
    phase_res = wrap_to_pi(phase_unw - np.angle(Hm))
    # weight & concatenate
    return np.concatenate([np.sqrt(w_mag) * logmag_res,
                           np.sqrt(w_phase) * phase_res])

def fit_ftf(omega: np.ndarray,
            mag: np.ndarray,
            phase: np.ndarray,
            *,
            phase_in_degrees: bool = False,
            J: int = 1,
            w_mag: float = 1.0,
            w_phase: float = 1.0,
            robust: str = "soft_l1",
            f_scale: float = 1.0) -> Tuple[DTLParams, Dict]:
    """Fit Distributed Time Lag (DTL) FTF model to frequency response data.
    
    Best for monotonically decaying magnitude without rising behavior.
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Magnitude data (linear scale)
        phase: Phase data [rad] or [deg]
        phase_in_degrees: True if phase input is in degrees
        J: Number of poles (1 or 2)
        w_mag: Magnitude weighting (1.0 = equal weight)
        w_phase: Phase weighting (1.0 = equal weight)
        robust: Loss function ('soft_l1', 'huber', 'linear')
        f_scale: Robust loss scale parameter
        
    Returns:
        Tuple of (fitted_params, info_dict)
    """
    assert J in (1,2), "Supported J=1 or J=2"
    # sort by frequency (just in case)
    idx = np.argsort(omega)
    omega = np.asarray(omega, float)[idx]
    mag   = np.asarray(mag,   float)[idx]
    phase = np.asarray(phase, float)[idx]
    # unwrap phase
    phase_unw = unwrap_phase(phase, deg=phase_in_degrees)

    # init
    p0 = heuristic_init(omega, mag, phase_unw, J)
    x0 = pack_params_uncon(p0)

    # solve
    res = least_squares(residuals_uncon, x0,
                        args=(omega, mag, phase_unw, J, w_mag, w_phase),
                        method="trf", loss=robust, f_scale=f_scale, max_nfev=20000)

    p_hat = unpack_params_uncon(res.x, J)

    # diagnostics
    H_fit = H_dtl(omega, p_hat)
    info = {
        "cost": float(res.cost),
        "success": bool(res.success),
        "message": res.message,
        "nfev": int(res.nfev),
        "logmag_rmse": float(np.sqrt(np.mean((np.log(mag+1e-12)-np.log(np.abs(H_fit)+1e-12))**2))),
        "phase_rmse_rad": float(np.sqrt(np.mean(wrap_to_pi(phase_unw - np.angle(H_fit))**2))),
        "params": {
            "S": float(p_hat.S),
            "tau0": float(p_hat.tau0),
            "thetas": p_hat.thetas.tolist(),
            "ks": p_hat.ks.tolist()
        }
    }
    return p_hat, info

# -------------------- Plot helper --------------------

def plot_fit(omega: np.ndarray, mag: np.ndarray, phase: np.ndarray,
             params: DTLParams, phase_in_degrees=False, title=None):
    """Plot DTL FTF fit comparison.
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Measured magnitude (linear)
        phase: Measured phase [rad] or [deg]
        params: Fitted DTL parameters
        phase_in_degrees: True if phase is in degrees
        title: Plot title
        
    Returns:
        Figure object
    """
    phi_unw = unwrap_phase(phase, deg=phase_in_degrees)
    H_fit = H_dtl(omega, params)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    # magnitude (linear and dB overlay)
    axs[0].plot(omega/(2*np.pi), mag, 'o', ms=4, label='meas')
    axs[0].plot(omega/(2*np.pi), np.abs(H_fit), '-', lw=2, label='fit')
    ax2 = axs[0].twinx()
    ax2.plot(omega/(2*np.pi), 20*np.log10(np.maximum(mag,1e-12)), ':', alpha=0.3)
    ax2.plot(omega/(2*np.pi), 20*np.log10(np.maximum(np.abs(H_fit),1e-12)), ':', alpha=0.8)
    axs[0].set_xlabel('Frequency [Hz]'); axs[0].set_ylabel('|H| (linear)'); ax2.set_ylabel('|H| [dB]')
    axs[0].grid(alpha=0.3); axs[0].legend()

    # phase
    phase_fit = np.angle(H_fit)
    # align by adding multiples of 2π so the curves overlap visually
    delta = phi_unw - phase_fit
    phase_fit_aligned = phase_fit + np.round(delta/(2*np.pi))*2*np.pi
    if phase_in_degrees:
        axs[1].plot(omega/(2*np.pi), np.rad2deg(phi_unw), 'o', ms=4, label='meas (unw)')
        axs[1].plot(omega/(2*np.pi), np.rad2deg(phase_fit_aligned), '-', lw=2, label='fit')
        axs[1].set_ylabel('Phase [deg]')
    else:
        axs[1].plot(omega/(2*np.pi), phi_unw, 'o', ms=4, label='meas (unw)')
        axs[1].plot(omega/(2*np.pi), phase_fit_aligned, '-', lw=2, label='fit')
        axs[1].set_ylabel('Phase [rad]')
    axs[1].set_xlabel('Frequency [Hz]'); axs[1].grid(alpha=0.3); axs[1].legend()
    if title: fig.suptitle(title)
    plt.tight_layout()
    return fig

# ---------------- ZPK-Fractional Fit ----------------
# ZPK-Fractional Fit (Zeros + Poles with fractional exponents)
# 	•	Dowling, A.P. (1995). The calculation of thermoacoustic oscillations. J. Sound Vib. 180, 557–581.
# Introduces transfer-function representations with zeros and poles for combustor elements.
# 	•	Polifke, W. & Lawn, C. (2007). On the low-order modelling of flames for combustion dynamics. In Proceedings of the Summer Program, Center for Turbulence Research.
# Discusses use of zero-pole expansions for flame dynamics models.
# 	•	Freitag, M. (2018). System identification of distributed time delay models for flames. Doctoral thesis, TUM.
# Shows practical fitting of two-pulse DTL and ZPK-like models to FTF data.
# 	•	Schuermans, B., Bellucci, V., Geigle, K., Paschereit, C.O. (2004). Thermoacoustic modeling and control of a full-scale gas turbine combustor. Combust. Sci. Tech. 176, 1169–1197.
# Uses zero–pole transfer functions to represent measured flame dynamics in industrial hardware.
# 	•	Crocco, L. & Cheng, S.I. (1956). Theory of Combustion Instability in Liquid Propellant Rocket Motors. AGARD Combustion Instability Colloquium.
# Early system-theory treatment, placing flame as a general ZPK system between velocity and heat release.

@dataclass
class ZPKFrac:
    """ZPK-Fractional FTF parameters (zeros + poles with fractional exponents).
    
    Args:
        S: Gain scale (dimensionless, > 0)
        tau0: Bulk delay [s] (>= 0)
        z: Zero time constants [s] (R>0, creates rising behavior)
        a: Zero exponents (dimensionless, R>0, phase lead)
        p: Pole time constants [s] (J>0, creates decay)
        k: Pole exponents (dimensionless, J>0, phase lag)
    """
    S: float
    tau0: float
    z: np.ndarray
    a: np.ndarray
    p: np.ndarray
    k: np.ndarray

def H_zpk_frac(omega: np.ndarray, p: ZPKFrac) -> np.ndarray:
    """ZPK-Fractional flame transfer function.
    
    H(ω) = S * exp(-i ω τ₀) * [∏ᵣ (1 + i ω zᵣ)^(aᵣ)] / [∏ⱼ (1 + i ω θⱼ)^(kⱼ)]
    
    Captures FTF with rising-then-falling magnitude via zeros (numerator)
    and poles (denominator). Use when magnitude shows shoulders/humps.
    
    Args:
        omega: Angular frequency [rad/s]
        p: ZPK-Fractional parameters
        
    Returns:
        Complex FTF values
    """
    H = p.S * np.exp(-1j * omega * p.tau0)
    for zr, ar in zip(p.z, p.a):
        H *= (1.0 + 1j * omega * zr) ** (ar)
    for pr, kr in zip(p.p, p.k):
        H /= (1.0 + 1j * omega * pr) ** (kr)
    return H

def calculate_ftf_zpk(omega: np.ndarray, params: ZPKFrac) -> np.ndarray:
    """Calculate ZPK-Fractional flame transfer function from parameters.
    
    Convenience wrapper for H_zpk_frac() with clear naming.
    
    Args:
        omega: Angular frequency [rad/s]
        params: ZPK-Fractional parameters
        
    Returns:
        Complex FTF values H(ω)
    """
    return H_zpk_frac(omega, params)

# ---------------- Utilities (ZPK-specific) ----------------
def softplus_zpk(x):
    """Softplus activation for ZPK model: log(1 + exp(x)). Maps R → R⁺."""
    return np.log1p(np.exp(x))

def inv_softplus_zpk(y):
    """Inverse softplus for ZPK model for y > 0. Maps R⁺ → R.
    
    Args:
        y: Positive values
        
    Returns:
        Inverse softplus values
    """
    y = np.asarray(y, dtype=float)
    return np.log(np.expm1(np.maximum(y, 1e-12)))

def pack_uncon_zpk(params: ZPKFrac) -> np.ndarray:
    """
    Flatten to a 1-D unconstrained vector in this order:
    [log S, inv_sp(tau0), inv_sp(z[0..R-1]), inv_sp(a[0..R-1]),
     inv_sp(p[0..J-1]), inv_sp(k[0..J-1])]
    """
    v = [float(np.log(params.S)), float(inv_softplus_zpk(params.tau0))]
    v.extend(inv_softplus_zpk(np.asarray(params.z, dtype=float)).ravel().tolist())
    v.extend(inv_softplus_zpk(np.asarray(params.a, dtype=float)).ravel().tolist())
    v.extend(inv_softplus_zpk(np.asarray(params.p, dtype=float)).ravel().tolist())
    v.extend(inv_softplus_zpk(np.asarray(params.k, dtype=float)).ravel().tolist())
    return np.array(v, dtype=float)

def unpack_uncon_zpk(v: np.ndarray, R: int, J: int) -> ZPKFrac:
    """
    Inverse of pack_uncon_zpk. Enforces positivity via softplus.
    """
    v = np.asarray(v, dtype=float).ravel()
    i = 0
    S    = np.exp(v[i]); i += 1
    tau0 = softplus_zpk(v[i]); i += 1
    z = softplus_zpk(v[i:i+R]); i += R
    a = softplus_zpk(v[i:i+R]); i += R
    p = softplus_zpk(v[i:i+J]); i += J
    k = softplus_zpk(v[i:i+J]); i += J
    return ZPKFrac(S=float(S), tau0=float(tau0), z=z, a=a, p=p, k=k)

# ---------------- Heuristics ----------------
def init_zpk_frac(omega, mag, phase_unw, R: int, J: int) -> ZPKFrac:
    """Generate initial ZPK-Fractional parameter guess from FTF data.
    
    Uses heuristics to distribute zeros (for rising) and poles (for decay).
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Magnitude data (linear scale)
        phase_unw: Unwrapped phase [rad]
        R: Number of zeros
        J: Number of poles
        
    Returns:
        Initial ZPK-Fractional parameters
    """
    # low-f window
    nlo = max(3, int(0.15*len(omega)))
    S0 = float(np.median(np.clip(mag[:nlo], 1e-8, None)))
    # tau0 from low-f phase slope
    A = np.vstack([omega[:nlo], np.ones(nlo)]).T
    slope, _ = np.linalg.lstsq(A, phase_unw[:nlo], rcond=None)[0]
    tau0_0 = float(np.clip(-slope, 0.0, 1.0/(omega[1]-omega[0]+1e-9)))
    # crude corners
    target = S0/np.sqrt(2)
    idx = np.where(mag <= target)[0]
    w_c = omega[idx[0]] if idx.size else omega[len(omega)//3]
    t0 = 1.0/max(w_c, 1e-6)

    # distribute zeros/poles
    z = t0 * np.linspace(0.4, 1.2, max(1,R))
    a = np.full(R, 0.6)            # gentle lead
    p = t0 * np.linspace(0.8, 2.5, max(1,J))
    k = np.full(J, 1.0)            # 20 dB/dec roll-off each
    return ZPKFrac(S=S0, tau0=tau0_0, z=z, a=a, p=p, k=k)

# ---------------- Residuals & Fit ----------------
def residuals_zpk(v, omega, mag, phi_unw, R, J, w_mag, w_phase):
    """ZPK residual function for optimization.
    
    Args:
        v: Unconstrained parameter vector
        omega: Angular frequency [rad/s]
        mag: Magnitude data (linear)
        phi_unw: Unwrapped phase [rad]
        R: Number of zeros
        J: Number of poles
        w_mag: Magnitude weight
        w_phase: Phase weight
        
    Returns:
        Weighted residual vector
    """
    pars = unpack_uncon_zpk(v, R, J)
    H = H_zpk_frac(omega, pars)
    r_mag  = np.log(mag + 1e-12) - np.log(np.abs(H) + 1e-12)
    r_phase= wrap_to_pi(phi_unw - np.angle(H))
    return np.concatenate([np.sqrt(w_mag)*r_mag, np.sqrt(w_phase)*r_phase])

def fit_ftf_zpk(omega, mag, phase, *,
                phase_in_degrees=False, R=1, J=1,
                w_mag=1.0, w_phase=1.0,
                robust="soft_l1", f_scale=1.0, max_nfev=20000) -> Tuple[ZPKFrac, Dict]:
    """Fit ZPK-Fractional FTF model to frequency response data.
    
    Best for magnitude with rising-then-falling behavior (shoulders/humps).
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Magnitude data (linear scale)
        phase: Phase data [rad] or [deg]
        phase_in_degrees: True if phase input is in degrees
        R: Number of zeros (creates rising behavior)
        J: Number of poles (creates decay)
        w_mag: Magnitude weighting (1.0 = equal weight)
        w_phase: Phase weighting (1.0 = equal weight)
        robust: Loss function ('soft_l1', 'huber', 'linear')
        f_scale: Robust loss scale parameter
        max_nfev: Maximum function evaluations
        
    Returns:
        Tuple of (fitted_params, info_dict)
    """
    # sort & unwrap
    idx = np.argsort(omega)
    w   = np.asarray(omega, float)[idx]
    M   = np.asarray(mag,   float)[idx]
    phi = np.asarray(phase, float)[idx]
    phi_unw = unwrap_phase(phi, deg=phase_in_degrees)

    # init
    p0 = init_zpk_frac(w, M, phi_unw, R, J)
    x0 = pack_uncon_zpk(p0)

    # mag-only prefit (stabilizes)
    def res_mag_only(v): 
        pars = unpack_uncon_zpk(v, R, J)
        H = H_zpk_frac(w, pars)
        return np.sqrt(w_mag)*(np.log(M+1e-12)-np.log(np.abs(H)+1e-12))
    x1 = least_squares(res_mag_only, x0, method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev//2).x

    # joint fit
    sol = least_squares(residuals_zpk, x1, args=(w,M,phi_unw,R,J,w_mag,w_phase),
                        method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
    pars = unpack_uncon_zpk(sol.x, R, J)
    Hf   = H_zpk_frac(w, pars)
    info = {
        "success": bool(sol.success), "message": sol.message, "nfev": int(sol.nfev),
        "cost": float(sol.cost),
        "logmag_rmse": float(np.sqrt(np.mean((np.log(M+1e-12)-np.log(np.abs(Hf)+1e-12))**2))),
        "phase_rmse_rad": float(np.sqrt(np.mean(wrap_to_pi(phi_unw - np.angle(Hf))**2))),
        "params": {"S": float(pars.S), "tau0": float(pars.tau0),
                   "z": pars.z.tolist(), "a": pars.a.tolist(),
                   "p": pars.p.tolist(), "k": pars.k.tolist()}
    }
    return pars, info

# ---------------- Plot ----------------
def plot_zpk_fit(omega, mag, phase, pars: ZPKFrac, phase_in_degrees=False, title=None):
    """Plot ZPK-Fractional FTF fit comparison.
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Measured magnitude (linear)
        phase: Measured phase [rad] or [deg]
        pars: Fitted ZPK-Fractional parameters
        phase_in_degrees: True if phase is in degrees
        title: Plot title
        
    Returns:
        Figure object
    """
    w = np.asarray(omega, float)
    H = H_zpk_frac(w, pars)
    phi_u = unwrap_phase(phase, deg=phase_in_degrees)
    ph_fit = np.angle(H)
    # align wrapped phase visually
    ph_fit += np.round((phi_u - ph_fit)/(2*np.pi))*2*np.pi

    fig, ax = plt.subplots(1,2, figsize=(9,3.8))
    f = w/(2*np.pi)
    ax[0].plot(f, mag, 'o', ms=4, label='meas')
    ax[0].plot(f, np.abs(H), '-', lw=2, label='fit')
    ax[0].set_xlabel('Hz'); ax[0].set_ylabel('|H|'); ax[0].grid(alpha=.3); ax[0].legend()

    if phase_in_degrees:
        ax[1].plot(f, np.rad2deg(phi_u), 'o', ms=4, label='meas')
        ax[1].plot(f, np.rad2deg(ph_fit), '-', lw=2, label='fit')
        ax[1].set_ylabel('Phase [deg]')
    else:
        ax[1].plot(f, phi_u, 'o', ms=4, label='meas')
        ax[1].plot(f, ph_fit, '-', lw=2, label='fit')
        ax[1].set_ylabel('Phase [rad]')
    ax[1].set_xlabel('Hz'); ax[1].grid(alpha=.3); ax[1].legend()
    if title: fig.suptitle(title)
    plt.tight_layout()
    return fig


# -------------------- Example wiring with a DataFrame --------------------
# df must have columns: 'omega' [rad/s], 'mag' [linear], 'phase' [rad or deg].
# If degrees: pass phase_in_degrees=True.

# Example:
# df = pd.read_csv("ftf.csv")
# p_hat, info = fit_ftf(df['omega'].values,
#                       df['mag'].values,
#                       df['phase'].values,
#                       phase_in_degrees=False,   # True if your phase column is degrees
#                       J=1,                      # try J=1, switch to 2 if shape needs it
#                       w_mag=1.0, w_phase=1.0,
#                       robust="soft_l1", f_scale=1.0)
# print(info)
# fig = plot_fit(df['omega'].values, df['mag'].values, df['phase'].values,
#                p_hat, phase_in_degrees=False, title="FTF fit (DTL)")
# plt.show()


# usage example zpk_fit
# omega [rad/s], mag [linear], phase [rad or deg]
# Choose small R,J first (e.g., R=1, J=1). If you see a shoulder that poles alone can’t match, try R=1,J=2.
# pars, info = fit_ftf_zpk(omega, mag, phase, phase_in_degrees=False, R=1, J=2,
#                          w_mag=1.0, w_phase=1.0, robust="soft_l1", f_scale=1.0)
# print(info)
# fig = plot_zpk_fit(omega, mag, phase, pars)
# plt.show()



# -------------------- Two-Pathway FTF Model --------------------
# Two-pathway model decomposes flame response into parallel physical mechanisms:
# 1. Equivalence ratio (φ) pathway: fuel/mixture fluctuations → positive heat release
# 2. Turbulence/velocity pathway: velocity fluctuations → negative heat release
# 
# Mathematical form: I(ω) = A_φ G_φ(ω) - A_t G_t(ω)
# where G_i(ω) = exp(-iωτ_i)(1+iωθ_i)^(-k_i) are Gamma-distributed delay kernels
# 
# References:
# • Lieuwen, T. (2012). Unsteady Combustor Physics. Cambridge University Press.
#   Chapter on distributed delays and multi-pathway decomposition of FTFs.
# • Schuller, T., Durox, D., Candel, S. (2003). A unified model for the prediction 
#   of laminar flame transfer functions. Combust. Flame 134, 21–34.
# • Noiray, N. & Schuermans, B. (2013). Deterministic quantities characterizing 
#   noise driven Hopf bifurcations in gas turbine combustors. Int. J. Non-Linear Mech. 50, 152–163.


@dataclass
class TwoPathParams:
    """Two-pathway FTF parameters.
    
    Models flame response via parallel equivalence ratio and turbulence pathways.
    Based on Lieuwen's decomposition: I(ω) = A_φ G_φ(ω) - A_t G_t(ω)
    
    Args:
        A_phi: Equivalence ratio pathway amplitude (dimensionless, > 0)
        A_t: Turbulence pathway amplitude (dimensionless, > 0)
        tau_phi: ER pathway bulk delay [s] (>= 0)
        tau_t: Turbulence pathway bulk delay [s] (>= 0)
        theta_phi: ER pathway time constant [s] (> 0)
        theta_t: Turbulence pathway time constant [s] (> 0)
        k_phi: ER pathway shape exponent (dimensionless, > 0)
        k_t: Turbulence pathway shape exponent (dimensionless, > 0)
    """
    A_phi: float; A_t: float
    tau_phi: float; tau_t: float
    theta_phi: float; theta_t: float
    k_phi: float; k_t: float

def G_gamma(omega, k, theta, tau):
    """Gamma-distributed delay kernel for two-pathway model.
    
    G(ω) = exp(-i ω τ) * (1 + i ω θ)^(-k)
    
    Args:
        omega: Angular frequency [rad/s]
        k: Shape exponent (dimensionless, > 0)
        theta: Time constant [s] (> 0)
        tau: Bulk delay [s] (>= 0)
        
    Returns:
        Complex transfer function values
    """
    return np.exp(-1j*omega*tau) * (1.0 + 1j*omega*theta)**(-k)

def I_two_path(omega, p: TwoPathParams):
    """Two-pathway interaction index function.
    
    I(ω) = A_φ G_φ(ω) - A_t G_t(ω)
    
    Models flame response as superposition of equivalence ratio (positive)
    and turbulence (negative) pathways with Gamma-distributed delays.
    
    Args:
        omega: Angular frequency [rad/s]
        p: Two-pathway parameters
        
    Returns:
        Complex interaction index values
    """
    return (p.A_phi * G_gamma(omega, p.k_phi, p.theta_phi, p.tau_phi)
            - p.A_t   * G_gamma(omega, p.k_t,   p.theta_t,   p.tau_t))

def calculate_ftf_two_pathway(omega: np.ndarray, params: TwoPathParams, 
                             T_ratio: float = None) -> np.ndarray:
    """Calculate two-pathway flame transfer function from parameters.
    
    Can return either interaction index I(ω) or full FTF T22(ω).
    
    Args:
        omega: Angular frequency [rad/s]
        params: Two-pathway parameters
        T_ratio: Temperature ratio T2/T1. If provided, returns T22(ω) = 1 + (T_ratio-1)*I(ω).
                If None, returns interaction index I(ω) directly.
        
    Returns:
        Complex FTF values: I(ω) if T_ratio=None, else T22(ω)
    """
    I = I_two_path(omega, params)
    if T_ratio is not None:
        return 1.0 + (T_ratio - 1.0) * I
    return I

# -------- Utilities --------
def unwrap_phase_twopath(phi, deg=False):
    phi = np.asarray(phi, float)
    if deg: phi = np.deg2rad(phi)
    return np.unwrap(phi)

def wrap_pi_twopath(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def softplus_twopath(x): return np.log1p(np.exp(x))
def inv_softplus_twopath(y):
    y = np.asarray(y, float)
    return np.log(np.expm1(np.maximum(y, 1e-12)))

def pack_uncon_twopath(p: TwoPathParams) -> np.ndarray:
    v = [
        inv_softplus_twopath(p.A_phi), inv_softplus_twopath(p.A_t),
        inv_softplus_twopath(p.tau_phi), inv_softplus_twopath(p.tau_t),
        inv_softplus_twopath(p.theta_phi), inv_softplus_twopath(p.theta_t),
        inv_softplus_twopath(p.k_phi), inv_softplus_twopath(p.k_t),
    ]
    return np.array(v, float)

def unpack_uncon_twopath(v: np.ndarray) -> TwoPathParams:
    v = np.asarray(v, float).ravel()
    i=0
    A_phi = softplus_twopath(v[i]); i+=1
    A_t   = softplus_twopath(v[i]); i+=1
    tau_p = softplus_twopath(v[i]); i+=1
    tau_t = softplus_twopath(v[i]); i+=1
    th_p  = softplus_twopath(v[i]); i+=1
    th_t  = softplus_twopath(v[i]); i+=1
    k_p   = softplus_twopath(v[i]); i+=1
    k_t   = softplus_twopath(v[i]); i+=1
    return TwoPathParams(A_phi, A_t, tau_p, tau_t, th_p, th_t, k_p, k_t)

def magphase_to_complex(mag, phase, phase_in_degrees=False):
    phi = unwrap_phase_twopath(phase, deg=phase_in_degrees)
    return np.asarray(mag, float) * np.exp(1j*phi)

# -------- Heuristic init --------
def init_two_path(omega, M_use, phi_unw) -> TwoPathParams:
    """Generate initial two-pathway parameter guess from FTF data.
    
    Uses heuristics based on low-frequency behavior and corner frequencies
    to initialize equivalence ratio and turbulence pathway parameters.
    
    Args:
        omega: Angular frequency [rad/s]
        M_use: Magnitude data (linear scale)
        phi_unw: Unwrapped phase [rad]
        
    Returns:
        Initial two-pathway parameters
    """
    # Low-f slope → bulk delay ~ mean convective time
    nlo = max(3, int(0.15*len(omega)))
    A = np.vstack([omega[:nlo], np.ones(nlo)]).T
    slope, _ = np.linalg.lstsq(A, phi_unw[:nlo], rcond=None)[0]
    tau0 = float(np.clip(-slope, 0.0, 1.0/(omega[1]-omega[0] + 1e-9)))

    S0 = float(np.median(np.clip(M_use[:nlo], 1e-8, None)))
    # Gentle cancellation at low f → pick A_phi ≳ A_t, both ~ O(S0)
    A_phi = max(S0, 1e-3)
    A_t   = 0.6 * A_phi

    # Corners from crude -3 dB point
    target = S0/np.sqrt(2)
    idx = np.where(M_use <= target)[0]
    wc = omega[idx[0]] if idx.size else omega[len(omega)//3]
    t_fast = 0.5/max(wc, 1e-6); t_slow = 2.5/max(wc, 1e-6)

    return TwoPathParams(
        A_phi=A_phi, A_t=A_t,
        tau_phi=tau0, tau_t=1.3*tau0,
        theta_phi=t_fast, theta_t=t_slow,
        k_phi=1.0, k_t=1.0
    )

# -------- Residuals --------
def residuals_joint(x, omega, M_use, phi_unw, w_mag, w_phase):
    p = unpack_uncon_twopath(x)
    I = I_two_path(omega, p)
    r_mag  = np.log(M_use + 1e-12) - np.log(np.abs(I) + 1e-12)
    r_phase= wrap_pi_twopath(phi_unw - np.angle(I))
    return np.concatenate([np.sqrt(w_mag)*r_mag, np.sqrt(w_phase)*r_phase])

# -------- Main fit --------
def fit_ftf_two_path(omega, mag, phase, *,
                     phase_in_degrees=False,
                     normalize=True, T_ratio=None,
                     w_mag=1.0, w_phase=1.0,
                     robust="soft_l1", f_scale=1.0,
                     max_nfev=20000) -> Tuple[TwoPathParams, Dict]:
    """Fit two-pathway FTF model to frequency response data.
    
    Models flame response via parallel equivalence ratio and turbulence pathways
    with Gamma-distributed delays. Best for complex interference patterns.
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Magnitude data (linear scale)
        phase: Phase data [rad] or [deg]
        phase_in_degrees: True if phase input is in degrees
        normalize: If True, fits I = (T22-1)/(T_ratio-1); else fits I directly
        T_ratio: Temperature ratio T2/T1 (required if normalize=True)
        w_mag: Magnitude weighting (1.0 = equal weight)
        w_phase: Phase weighting (1.0 = equal weight)
        robust: Loss function ('soft_l1', 'huber', 'linear')
        f_scale: Robust loss scale parameter
        max_nfev: Maximum function evaluations
        
    Returns:
        Tuple of (fitted_params, info_dict)
    """
    idx = np.argsort(omega)
    w   = np.asarray(omega, float)[idx]
    M   = np.asarray(mag,   float)[idx]
    ph  = np.asarray(phase, float)[idx]

    if normalize:
        if T_ratio is None:
            raise ValueError("normalize=True requires T_ratio (T2/T1).")
        Tr = np.asarray(T_ratio, float)
        Tr = np.full_like(w, float(Tr)) if Tr.size==1 else Tr[idx]
        T22 = magphase_to_complex(M, ph, phase_in_degrees)
        I_c = (T22 - 1.0) / (Tr - 1.0 + 1e-12)
        M_use = np.abs(I_c)
        phi_unw = unwrap_phase_twopath(np.angle(I_c), deg=False)
    else:
        M_use = M
        phi_unw = unwrap_phase_twopath(ph, deg=phase_in_degrees)

    p0 = init_two_path(w, M_use, phi_unw)
    x0 = pack_uncon_twopath(p0)

    # Mag-only prefit to stabilize
    def res_mag_only(x):
        p = unpack_uncon_twopath(x); I = I_two_path(w, p)
        return np.sqrt(w_mag) * (np.log(M_use+1e-12)-np.log(np.abs(I)+1e-12))
    x1 = least_squares(res_mag_only, x0, method="trf",
                       loss=robust, f_scale=f_scale, max_nfev=max_nfev//2).x

    # Joint fit
    sol = least_squares(residuals_joint, x1,
                        args=(w, M_use, phi_unw, w_mag, w_phase),
                        method="trf", loss=robust, f_scale=f_scale, max_nfev=max_nfev)
    p_hat = unpack_uncon_twopath(sol.x)
    I_fit = I_two_path(w, p_hat)

    info = {
        "success": bool(sol.success),
        "message": sol.message,
        "nfev": int(sol.nfev),
        "cost": float(sol.cost),
        "fitted_domain": "interaction_index" if normalize else "I_direct",
        "logmag_rmse": float(np.sqrt(np.mean((np.log(M_use+1e-12)-np.log(np.abs(I_fit)+1e-12))**2))),
        "phase_rmse_rad": float(np.sqrt(np.mean(wrap_pi_twopath(phi_unw - np.angle(I_fit))**2))),
        "params": p_hat.__dict__
    }

    # Helper to reconstruct T22 if you fitted I
    if normalize:
        info["to_T22"] = lambda omega_eval, T_ratio_eval: \
            1.0 + (np.asarray(T_ratio_eval, float) - 1.0) * I_two_path(np.asarray(omega_eval, float), p_hat)

    return p_hat, info

# -------- Plot helper --------
def plot_two_path(omega, mag, phase, p: TwoPathParams, *,
                  phase_in_degrees=False, normalize=True, T_ratio=None, title=None):
    """Plot two-pathway FTF fit comparison.
    
    Args:
        omega: Angular frequency [rad/s]
        mag: Measured magnitude (linear)
        phase: Measured phase [rad] or [deg]
        p: Fitted two-pathway parameters
        phase_in_degrees: True if phase is in degrees
        normalize: True if plotting T22, False if plotting I directly
        T_ratio: Temperature ratio T2/T1 (required if normalize=True)
        title: Plot title
        
    Returns:
        Figure object
    """
    w = np.asarray(omega, float); f = w/(2*np.pi)
    if normalize:
        if T_ratio is None: raise ValueError("plot_two_path: need T_ratio when normalize=True.")
        Tr = np.asarray(T_ratio, float)
        I_fit = I_two_path(w, p)
        T22_fit = 1.0 + (Tr - 1.0) * I_fit
        T22_meas = magphase_to_complex(mag, phase, phase_in_degrees)
        phi_u = unwrap_phase_twopath(phase, deg=phase_in_degrees)
        ph_fit = np.angle(T22_fit); ph_fit += np.round((phi_u - ph_fit)/(2*np.pi))*2*np.pi
        fig, ax = plt.subplots(1,2, figsize=(9,3.8))
        ax[0].plot(f, np.abs(T22_meas), 'o', ms=4, label='|T22| meas')
        ax[0].plot(f, np.abs(T22_fit),  '-', lw=2, label='|T22| fit (two-path)')
        ax[0].set_xlabel('Hz'); ax[0].set_ylabel('|T22|'); ax[0].grid(alpha=.3); ax[0].legend()
        show_meas = phi_u if not phase_in_degrees else np.rad2deg(phi_u)
        show_fit  = ph_fit if not phase_in_degrees else np.rad2deg(ph_fit)
        ax[1].plot(f, show_meas, 'o', ms=4, label='phase meas (unw)')
        ax[1].plot(f, show_fit,  '-', lw=2, label='phase fit')
        ax[1].set_xlabel('Hz'); ax[1].set_ylabel('Phase [rad]' if not phase_in_degrees else 'Phase [deg]')
        ax[1].grid(alpha=.3); ax[1].legend()
        if title: fig.suptitle(title); plt.tight_layout()
        return fig
    else:
        I_meas = magphase_to_complex(mag, phase, phase_in_degrees)
        phi_u = unwrap_phase_twopath(np.angle(I_meas), deg=False)
        I_fit = I_two_path(w, p)
        ph_fit = np.angle(I_fit); ph_fit += np.round((phi_u - ph_fit)/(2*np.pi))*2*np.pi
        fig, ax = plt.subplots(1,2, figsize=(9,3.8))
        ax[0].plot(f, np.abs(I_meas), 'o', ms=4, label='|I| meas')
        ax[0].plot(f, np.abs(I_fit),  '-', lw=2, label='|I| fit')
        ax[0].set_xlabel('Hz'); ax[0].set_ylabel('|I|'); ax[0].grid(alpha=.3); ax[0].legend()
        ax[1].plot(f, phi_u, 'o', ms=4, label='phase meas (unw)')
        ax[1].plot(f, ph_fit, '-', lw=2, label='phase fit')
        ax[1].set_xlabel('Hz'); ax[1].set_ylabel('Phase [rad]'); ax[1].grid(alpha=.3); ax[1].legend()
        if title: fig.suptitle(title); plt.tight_layout()
        return fig

# example usage
# Given T22 data (omega [rad/s], |T22|, phase [rad or deg]) and T_ratio = T2/T1
# p_hat, info = fit_ftf_two_path(
#     omega, mag=abs_T22, phase=phase_T22,
#     phase_in_degrees=False,
#     normalize=True, T_ratio=T2_over_T1,   # ← fit in interaction-index domain
#     w_mag=1.0, w_phase=1.0, robust="soft_l1"
# )
# print(info["fitted_domain"], info["logmag_rmse"], info["phase_rmse_rad"])
# fig = plot_two_path(omega, abs_T22, phase_T22, p_hat, normalize=True, T_ratio=T2_over_T1, title="Two-Pathway FTF")
        
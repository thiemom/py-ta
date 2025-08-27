import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
from scipy.optimize import least_squares

# -------------------- DTL FTF model --------------------

@dataclass
class DTLParams:
    S: float           # > 0   (gain scale)
    tau0: float        # >= 0  (bulk delay) [s]
    thetas: np.ndarray # J>0   (time widths) [s]
    ks: np.ndarray     # J>0   (shape exponents) [-]

def H_dtl(omega: np.ndarray, p: DTLParams) -> np.ndarray:
    """
    Complex FTF: H(ω)=S * exp(-i ω τ0) * Π_j (1 + i ω θ_j)^(-k_j)
    """
    H = p.S * np.exp(-1j * omega * p.tau0)
    for theta, k in zip(p.thetas, p.ks):
        H *= (1.0 + 1j * omega * theta) ** (-k)
    return H

# -------------------- Phase handling --------------------

def unwrap_phase(phi: np.ndarray, deg: bool) -> np.ndarray:
    """Return unwrapped phase in radians (input may be rad or deg)."""
    if deg:
        phi = np.deg2rad(phi)
    return np.unwrap(phi)

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap phase difference to (-pi, pi]."""
    return (x + np.pi) % (2*np.pi) - np.pi

# -------------------- Parameterization & init --------------------

def pack_params_uncon(p: DTLParams) -> np.ndarray:
    """
    Map physical params to unconstrained vector via softplus/exponentials
    so we can optimize in R^n and enforce positivity on decode.
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
    """
    Decode unconstrained vector to physical params with positivity.
    """
    def softplus(x): return np.log1p(np.exp(x))
    S = np.exp(v[0])
    tau0 = softplus(v[1])
    thetas = softplus(v[2:2+J])
    ks = softplus(v[2+J:2+2*J])
    return DTLParams(S=S, tau0=tau0, thetas=thetas, ks=ks)

def heuristic_init(omega: np.ndarray, mag: np.ndarray, phase_unw: np.ndarray, J: int) -> DTLParams:
    """
    Simple, robust initial guess:
      - S from low-frequency magnitude median
      - tau0 from low-f phase slope (linear fit of phase vs omega)
      - theta from 3 dB corner (fallback if not found)
      - k ~ 1 (or split across two pulses)
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
    """
    Fit DTL FTF to (omega, |H|, phase).
    Returns (params, info).
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
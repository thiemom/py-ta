# -*- coding: utf-8 -*-
"""
entropy_loop.py - Breathing-mode loop with stochastic LBO gating + entropy pockets

References (brief):
- T. Lieuwen, Unsteady Combustion Physics (low-Mach formulation; opposite-sign pathways).
- B. Schuermans, N. Noiray, M. Bothien, O. Paschereit, W. Polifke - GT thermoacoustic system ID, triggering, intermittency.
- J.-C. Parmentier - convective-dispersive transport kernels for scalar/entropy (dispersion model for pockets).
- Bake-style nozzle entropy-to-acoustic conversion (gain vs M with phase trend).

Model summary:
- Input OU forcing on φ′: S_η(ω) = 2D / (1+ω² t_c²); φ_rms² = D/t_c.
- Flame mapping T′/T0 = ((T_hot − T_cold)/(φ_op·T_hot)) · φ′, gated near LBO.
- Entropy to acoustic: p′ = (γ−1) p0 G_Bake(M) (T′/T0).
- Acoustic back to velocity: u′ = p′/(ρ c); feedback to flame: φ′ = χ u′ + η_φ.
- Simplification: p0/(ρ c) = sqrt(R T0 / γ), so explicit pressure cancels in net gain.
- Add stochastic LBO gate g(t) ∈ {0,1} (Markov switching) and “pocket” bursts at switches.
- Close loop in frequency domain for the OU path; add shot-noise PSD for bursts.

Provides:
- Analytic PSD S_pp(ω) = OU-path/|1−L|² + burst-path (shot noise with Parmentier dispersion).
- Optional time-domain simulator for the gate and burst train (sanity checks).

Quick start:
- OU forcing on φ′ (Ornstein–Uhlenbeck)
- Flame delay + jitter (Lieuwen low-Mach mapping)
- Stochastic LBO gate (random telegraph)
- Entropy pockets at gate switches with Parmentier-like dispersion
- Bake-style entropy-to-acoustic conversion

Analytic PSD:
Use ``S_pp(omega, thermo, flame, ou, bake, gate, gains)`` to get a dict
with keys ``S_ou``, ``S_shot``, and ``S_total``.

Example:
>>> import numpy as np
>>> import entropy_loop as el
>>> f = np.linspace(0.1, 250, 2000)
>>> w = 2*np.pi*f
>>> thermo = el.ThermoProps()
>>> flame = el.FlameMap()
>>> ou = el.OUDrive()
>>> bake = el.BakeModel()
>>> gate = el.GateModel()
>>> gains = el.LoopGains()
>>> S = el.S_pp(w, thermo, flame, ou, bake, gate, gains)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Dict

@dataclass
class ThermoProps:
    """
    Thermodynamic base state used for entropy-to-acoustic conversion.

    Parameters
    ----------
    gamma : float, default 1.35
        Specific heat ratio.
    R : float, default 287.0
        Gas constant [J/(kg·K)].
    T0 : float, default 800.0
        Mean temperature [K].
    p0 : float, default 1.5e6
        Mean pressure [Pa].

    Notes
    -----
    The mapping from temperature (entropy) fluctuations to pressure uses
    (γ−1) p0 G_Bake(M). Convenience accessors for `ρ` and `c`
    are provided for loop-gain scaling.
    """
    gamma: float = 1.35
    R: float = 287.0
    T0: float = 800.0
    p0: float = 1.5e6
    def rho(self) -> float:
        """Return density ρ = p0/(R T0)."""
        return self.p0/(self.R*self.T0)
    def c(self) -> float:
        """Return sound speed c = sqrt(γ R T0)."""
        return np.sqrt(self.gamma*self.R*self.T0)

@dataclass
class FlameMap:
    """
    Lieuwen-style low-Mach mapping from equivalence ratio fluctuations to
    temperature rise with a convective delay and jitter.

    Parameters
    ----------
    T_hot : float, default 2100.0
        Adiabatic flame temperature [K].
    T_cold : float, default 800.0
        Inlet temperature [K].
    phi_op : float, default 0.7
        Operating equivalence ratio.
    tau_res : float, default 8e-3
        Mean residence/response delay [s].
    sigma_tau : float, default 1.5e-3
        Standard deviation of delay jitter [s]. Models dispersion of the flame response.

    Methods
    -------
    K_phi()
        Static gain from φ′ to T′/T0.
    """
    T_hot: float = 2100.0
    T_cold: float = 800.0
    phi_op: float = 0.7
    tau_res: float = 8e-3
    sigma_tau: float = 1.5e-3
    def K_phi(self, T0: Optional[float]=None) -> float:
        """Return mapping coefficient K_φ = (T_hot − T_cold)/(φ_op·T_hot)."""
        return (self.T_hot - self.T_cold)/(self.phi_op*self.T_hot)

@dataclass
class OUDrive:
    """
    Ornstein–Uhlenbeck (OU) forcing on φ′ with PSD S_η(ω) = 2D / (1+ω² t_c²).

    Parameters
    ----------
    D : float, default 0.004
        Diffusion level; sets φ_rms² = D/t_c.
    t_c : float, default 0.12
        Correlation time [s].
    """
    D: float = 0.004
    t_c: float = 0.12

@dataclass
class BakeModel:
    """
    Simplified Bake-style entropy-to-acoustic conversion coefficient G_Bake(M).

    Parameters
    ----------
    M : float, default 0.3
        Nozzle Mach number.
    M0 : float, default 0.6
        Center Mach for Gaussian amplitude trend.
    sigma_M : float, default 0.2
        Width of Gaussian amplitude trend.
    phi_max : float, default 0.5π
        Maximum phase lead (radians) across Mach sweep.

    Notes
    -----
    Returns a complex gain with Gaussian amplitude peak and a phase that
    increases monotonically with M.
    """
    M: float = 0.3
    M0: float = 0.6
    sigma_M: float = 0.2
    phi_max: float = 0.5*np.pi
    def G(self) -> complex:
        """Return complex G(M): Gaussian amplitude with bounded phase ramp."""
        amp = np.exp(-0.5*((self.M - self.M0)/self.sigma_M)**2)
        x = np.clip((self.M - (self.M0 - 2*self.sigma_M))/(4*self.sigma_M), 0.0, 1.0)
        phase = x*self.phi_max
        return amp*np.exp(1j*phase)

@dataclass
class GateModel:
    """
    Random-telegraph gate for LBO-like on/off switching and pocket bursts.

    Parameters
    ----------
    m : float, default 2.0
        Stability margin proxy; larger m favors the ON state.
    lambda0_off, lambda0_on : float, default 2.0
        Baseline hazards [1/s] for OFF→ON and ON→OFF, modulated by `m`.
    beta : float, default 0.9
        Sensitivity of hazards to `m`.
    alpha_T : float, default 0.6
        Burst amplitude coefficient (fraction of DeltaT at switches).
    tau_adv : float, default 0.04
        Advection delay from source to nozzle [s].
    sigma_disp : float, default 0.015
        Temporal dispersion width for pocket kernel [s].

    Methods
    -------
    hazards() : tuple
        Return effective ON/OFF hazards after exponential biasing (λ_off, λ_on).
    p_on() : float
        Stationary ON probability.
    lambda_switch() : float
        Effective switching rate (shot-noise rate for bursts), λ_sw.
    """
    m: float = 2.0
    lambda0_off: float = 2.0
    lambda0_on: float = 2.0
    beta: float = 0.9
    alpha_T: float = 0.6
    tau_adv: float = 0.04
    sigma_disp: float = 0.015
    def hazards(self) -> Tuple[float,float]:
        """Return biased hazards (λ_off, λ_on) given current margin m."""
        lam_off = self.lambda0_off*np.exp(-self.beta*self.m)
        lam_on  = self.lambda0_on *np.exp(+self.beta*self.m)
        return lam_off, lam_on
    def p_on(self) -> float:
        """Return stationary ON probability p_on = λ_on/(λ_off+λ_on)."""
        lo, ln = self.hazards()
        return ln/(lo+ln)
    def lambda_switch(self) -> float:
        """Return effective switching rate λ_sw = (λ_off λ_on)/(λ_off+λ_on)."""
        lo, ln = self.hazards()
        return (lo*ln)/(lo+ln)

@dataclass
class LoopGains:
    """
    Net feedback path scaling for u′→φ′ and mean gating.

    Parameters
    ----------
    chi : float, default 0.8
        Velocity-to-φ′ feedback gain.
    include_mean_gate : bool, default True
        If True, scale the loop by stationary p_on to reflect average coupling.
    """
    chi: float = 0.8
    include_mean_gate: bool = True

def H_flame(omega: np.ndarray, flame: FlameMap) -> np.ndarray:
    """
    Flame frequency response from φ′ to T′/T0 with delay and Gaussian jitter.

    H_flame(ω) = K_φ · exp(−i ω τ_res) · exp(−½ ω² σ_τ²)

    Parameters
    ----------
    omega : array_like
        Angular frequency array [rad/s].
    flame : FlameMap
        Flame mapping parameters.

    Returns
    -------
    ndarray (complex)
        Complex gain from φ′ to T′/T0.
    """
    return flame.K_phi()*np.exp(-1j*omega*flame.tau_res)*np.exp(-0.5*(omega**2)*(flame.sigma_tau**2))

def H_s_to_p(thermo: ThermoProps, bake: BakeModel) -> complex:
    """Return entropy-to-acoustic mapping: (γ−1) p0 G_Bake(M)."""
    return (thermo.gamma - 1.0)*thermo.p0*bake.G()

def L_loop(omega: np.ndarray, thermo: ThermoProps, flame: FlameMap, bake: BakeModel, gains: LoopGains, gate: GateModel) -> np.ndarray:
    """
    Open-loop transfer L(ω) for the breathing mode.

    L(ω) = (χ_eff/(ρc)) · H_s→p · H_flame(ω), with χ_eff optionally scaled by p_on.
    """
    rho_c = thermo.rho()*thermo.c()
    chi_eff = gains.chi*(gate.p_on() if gains.include_mean_gate else 1.0)
    return (chi_eff/rho_c) * H_s_to_p(thermo,bake) * H_flame(omega,flame)

def H_p_phi(omega: np.ndarray, thermo: ThermoProps, flame: FlameMap, bake: BakeModel) -> np.ndarray:
    """Net mapping from φ′ to p′: H_pφ(ω) = H_s→p · H_flame(ω)."""
    return H_s_to_p(thermo,bake) * H_flame(omega,flame)

def S_eta_OU(omega: np.ndarray, ou: OUDrive) -> np.ndarray:
    """OU forcing PSD S_η(ω) = 2D / (1 + (ω t_c)²)."""
    return 2.0*ou.D/(1.0 + (omega*ou.t_c)**2)

def pocket_kernel_hat(omega: np.ndarray, gate: GateModel) -> np.ndarray:
    """
    Frequency response of Parmentier-like pocket dispersion kernel.

    Ψ(ω) = exp(−i ω τ_adv) · exp(−½ ω² σ_disp²)
    """
    return np.exp(-1j*omega*gate.tau_adv)*np.exp(-0.5*(omega**2)*(gate.sigma_disp**2))

def S_burst(omega: np.ndarray, thermo: ThermoProps, bake: BakeModel, gate: GateModel, flame: FlameMap) -> np.ndarray:
    """
    Shot-noise PSD from burst train at gate switches.

    Parameters
    ----------
    omega, thermo, bake, gate, flame : see respective types

    Returns
    -------
    ndarray (float)
        One-sided power spectral density contribution from pocket bursts.
    """
    lam_sw = gate.lambda_switch()
    dT = (flame.T_hot - flame.T_cold)
    a_rms2 = (gate.alpha_T*dT/thermo.T0)**2
    Hs = H_s_to_p(thermo,bake)
    Psi = pocket_kernel_hat(omega, gate)
    return lam_sw * a_rms2 * np.abs(Hs*Psi)**2

def S_pp(omega: np.ndarray,
         thermo: ThermoProps,
         flame: FlameMap,
         ou: OUDrive,
         bake: BakeModel,
         gate: GateModel,
         gains: LoopGains) -> Dict[str,np.ndarray]:
    """
    Analytic pressure PSD decomposition for the breathing-mode loop.

    Returns a dict with keys:
    - "S_ou": OU-forced path through closed loop [Pa²/Hz]
    - "S_shot": Burst (shot-noise) path [Pa²/Hz]
    - "S_total": Sum of both contributions [Pa²/Hz]
    """
    Hpf = H_p_phi(omega, thermo, flame, bake)
    Lw  = L_loop(omega, thermo, flame, bake, gains, gate)
    S_eta = S_eta_OU(omega, ou)
    S_ou = (np.abs(Hpf)**2)*S_eta/np.abs(1.0 - Lw)**2
    S_shot = S_burst(omega, thermo, bake, gate, flame)
    return {"S_ou": S_ou, "S_shot": S_shot, "S_total": S_ou + S_shot}

def S_gg(omega: np.ndarray, gate: GateModel) -> np.ndarray:
    """
    PSD of the random-telegraph gate g(t) with effective switching rate.

    S_gg(ω) = 4 p_on (1−p_on) λ_sw / (ω² + λ_sw²)
    """
    p_on = gate.p_on()
    lam_sw = gate.lambda_switch()
    return 4.0*p_on*(1.0-p_on)*lam_sw/(omega**2 + lam_sw**2)

def simulate_bursts(T: float, dt: float, gate: GateModel, flame: FlameMap, seed: Optional[int]=1234) -> np.ndarray:
    """
    Simulate a pocket-only time trace y(t) ≈ T′/T0 generated at gate switches.

    Each switch injects a Gaussian-shaped pocket with sign depending on ON/OFF.

    Parameters
    ----------
    T : float
        Total duration [s].
    dt : float
        Sample time [s].
    gate : GateModel
        Gate and pocket parameters.
    flame : FlameMap
        Provides ΔT via (T_hot − T_cold).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    ndarray (float)
        Time series of T′/T0 due to pockets alone.
    """
    rng = np.random.default_rng(seed)
    N = int(T/dt)
    y = np.zeros(N, dtype=float)
    g = 1
    lo, ln = gate.hazards()
    t_kernel = np.arange(-5*gate.sigma_disp, 5*gate.sigma_disp+dt, dt)
    psi = np.exp(-0.5*((t_kernel - gate.tau_adv)/gate.sigma_disp)**2)
    psi /= (np.sum(psi) + 1e-16)
    for n in range(N):
        if g == 1:
            if rng.random() < 1.0 - np.exp(-lo*dt):
                g = 0
                a = -gate.alpha_T*(flame.T_hot - flame.T_cold)/flame.T_cold
                i0 = n; i1 = min(N, i0 + len(psi))
                y[i0:i1] += a*psi[:i1-i0]
        else:
            if rng.random() < 1.0 - np.exp(-ln*dt):
                g = 1
                a = +gate.alpha_T*(flame.T_hot - flame.T_cold)/flame.T_cold
                i0 = n; i1 = min(N, i0 + len(psi))
                y[i0:i1] += a*psi[:i1-i0]
    return y
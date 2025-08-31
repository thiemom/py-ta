#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of Two-Delay Gamma FTF model fitting.

This script demonstrates:
1. Basic Two-Delay Gamma FTF model fitting to interaction index I(ω) data
2. Grid search over shape parameters (m_phi, m_t) to find optimal values
3. T22 normalization fitting using temperature ratio T2/T1

The Two-Delay Gamma model represents flame transfer function as:
I(ω) = A_phi * G_phi(ω) - A_t * G_t(ω)

where G_i(ω) = exp(-iωτ_i) * (1 + iωθ_i)^(-m_i) are Gamma distributed time delays
representing equivalence ratio (phi) and turbulence (t) pathways.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import the FTF functions
from ftf import *

def create_synthetic_two_delay_data():
    """
    Create synthetic two-delay gamma FTF data for testing.
    
    Generates interaction index I(ω) data using known parameters,
    representing the (T2/T1 - 1) * I domain (i.e., T22 - 1).
    
    Returns:
    --------
    omega : array
        Angular frequency array [rad/s]
    mag : array
        Magnitude of (T2/T1 - 1) * I(ω)
    phase : array
        Phase of (T2/T1 - 1) * I(ω) [rad]
    true_params : TwoDelayParams
        True parameters used to generate data
    m_phi, m_t : int
        True shape parameters
    """
    # Frequency range
    omega = np.logspace(0, 3, 100)  # 1 to 1000 rad/s
    
    # True parameters for synthetic data
    true_params = TwoDelayParams(
        A_phi=1.8,      # Equivalence ratio pathway amplitude
        A_t=1.2,        # Turbulence pathway amplitude  
        tau_phi=0.003,  # ER pathway delay [s]
        r_tau=0.85,     # Delay ratio (tau_t = r_tau * tau_phi)
        theta_phi=0.012, # ER pathway time constant [s]
        theta_t=0.018   # Turbulence pathway time constant [s]
    )
    
    # Integer shape parameters
    m_phi = 2  # ER pathway shape
    m_t = 3    # Turbulence pathway shape
    
    # Generate synthetic FTF using interaction index
    I_true = I_two_delay(omega, true_params, m_phi, m_t)
    
    # Add some noise
    np.random.seed(42)
    noise_level = 0.04
    mag_noise = 1 + noise_level * np.random.randn(len(omega))
    phase_noise = noise_level * np.random.randn(len(omega))
    
    mag = np.abs(I_true) * mag_noise
    phase = np.angle(I_true) + phase_noise
    
    return omega, mag, phase, true_params, m_phi, m_t

def create_synthetic_T22_data():
    """
    Create synthetic T22 FTF data for testing normalization.
    
    Generates T22 = 1 + (T_ratio - 1) * I(ω) data from known parameters,
    where T_ratio is the temperature ratio T2/T1.
    
    Returns:
    --------
    omega : array
        Angular frequency array [rad/s]
    mag : array
        Magnitude of T22(ω)
    phase : array
        Phase of T22(ω) [rad]
    true_params : TwoDelayParams
        True parameters used to generate data
    m_phi, m_t : int
        True shape parameters
    T_ratio : float
        Temperature ratio T2/T1
    """
    # Use the same parameters as the basic example for consistency
    omega, I_mag, I_phase, true_params, m_phi, m_t = create_synthetic_two_delay_data()
    
    # Temperature ratio
    T_ratio = 3.5  # T2/T1 - moderate ratio
    
    # Convert I to T22 = 1 + (T_ratio - 1) * I
    I_complex = I_mag * np.exp(1j * I_phase)
    T22_complex = 1.0 + (T_ratio - 1.0) * I_complex
    
    # Extract magnitude and phase
    mag = np.abs(T22_complex)
    phase = np.angle(T22_complex)
    
    return omega, mag, phase, true_params, m_phi, m_t, T_ratio

def run_two_delay_example():
    """Run Two-Delay Gamma FTF model example"""
    print("Two-Delay Gamma FTF Model Example")
    print("=" * 40)
    
    # Create synthetic data
    omega, mag, phase, true_params, m_phi_true, m_t_true = create_synthetic_two_delay_data()
    
    print(f"Generated {len(omega)} frequency points")
    print(f"Frequency range: {omega[0]/(2*np.pi):.2f} to {omega[-1]/(2*np.pi):.2f} Hz")
    print(f"True shape parameters: m_phi={m_phi_true}, m_t={m_t_true}")
    
    # Fit the model with single shape parameters
    print(f"\nFitting Two-Delay Gamma model (m_phi={m_phi_true}, m_t={m_t_true})...")
    p_hat, info = fit_two_delay_gamma(
        omega, mag, phase,
        phase_in_degrees=False,
        m_phi=m_phi_true,
        m_t=m_t_true,
        normalize=False,  # Data is already in I(omega) domain
        w_mag=1.0,
        w_phase=1.0,
        robust="soft_l1",
        f_scale=1.0,
        suppress_warnings=True
    )
    
    # Print results
    print(f"\nTwo-Delay Gamma Fit Results:")
    print(f"Success: {info['success']}")
    print(f"Cost: {info['cost']:.6f}")
    print(f"Function evaluations: {info['nfev']}")
    print(f"Log-magnitude RMSE: {info['logmag_rmse']:.6f}")
    print(f"Phase RMSE [rad]: {info['phase_rmse_rad']:.6f}")
    print(f"Fitted domain: {info['fitted_domain']}")
    
    print(f"\nFitted Parameters:")
    print(f"A_phi: {p_hat.A_phi:.4f} (true: {true_params.A_phi:.4f})")
    print(f"A_t: {p_hat.A_t:.4f} (true: {true_params.A_t:.4f})")
    print(f"tau_phi [s]: {p_hat.tau_phi:.6f} (true: {true_params.tau_phi:.6f})")
    print(f"r_tau: {p_hat.r_tau:.4f} (true: {true_params.r_tau:.4f})")
    print(f"tau_t [s]: {p_hat.r_tau * p_hat.tau_phi:.6f} (true: {true_params.r_tau * true_params.tau_phi:.6f})")
    print(f"theta_phi [s]: {p_hat.theta_phi:.6f} (true: {true_params.theta_phi:.6f})")
    print(f"theta_t [s]: {p_hat.theta_t:.6f} (true: {true_params.theta_t:.6f})")
    
    # Create plot
    I_fit = I_two_delay(omega, p_hat, m_phi_true, m_t_true)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Magnitude plot
    axes[0].loglog(omega/(2*np.pi), mag, 'o', ms=4, alpha=0.7, label='Synthetic data')
    axes[0].loglog(omega/(2*np.pi), np.abs(I_fit), '-', lw=2, label='Fitted model')
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('|I(ω)|')
    axes[0].set_title('Magnitude')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Phase plot
    phi_unw = unwrap_phase(phase, deg=False)
    phase_fit = np.angle(I_fit)
    # Align phase for plotting
    phase_fit += np.round((phi_unw - phase_fit)/(2*np.pi))*2*np.pi
    
    axes[1].semilogx(omega/(2*np.pi), phi_unw, 'o', ms=4, alpha=0.7, label='Synthetic data')
    axes[1].semilogx(omega/(2*np.pi), phase_fit, '-', lw=2, label='Fitted model')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Phase [rad]')
    axes[1].set_title('Phase')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'Two-Delay Gamma FTF Fit (m_phi={m_phi_true}, m_t={m_t_true})')
    plt.tight_layout()
    plt.savefig('two_delay_gamma_fit_example.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'two_delay_gamma_fit_example.png'")
    
    return p_hat, info

def run_fuel_split_example():
    """Run fuel-split model example with different pilot/main delays"""
    print("\n\nFuel-Split Example - Pilot/Main blending")
    print("=" * 55)

    # Frequency axis (Hz -> rad/s)
    f = np.logspace(1.8, 2.9, 150)  # ~63 to ~794 Hz
    omega = 2 * np.pi * f

    # Define distinct pilot and main parameters (different delays/time constants)
    pilot = TwoDelayParams(
        A_phi=0.10, A_t=0.40,
        # Make pilot more "pilot-like": longer delay and broader kernel
        tau_phi=5.0e-3, r_tau=0.95,
        theta_phi=2.5e-3, theta_t=1.5e-3
    )
    main = TwoDelayParams(
        A_phi=0.65, A_t=0.65,
        tau_phi=2.3e-3, r_tau=0.82,
        theta_phi=1.1e-3, theta_t=0.7e-3
    )

    # Use different shape parameters per stage
    # Slightly broader by reducing m_phi for pilot
    m_pilot_phi, m_pilot_t = 1, 2
    m_main_phi, m_main_t = 3, 2

    # Fuel split values to visualize
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    cfg = FuelSplitConfig(s0=0.0, s1=5.0, s2=0.0, dtau_pilot=0.0, dtau_main=0.0)

    # Temperature ratio for T22 projection
    T_ratio = 3.5

    # Compute I_mix for different splits
    curves_I = []
    for a in alphas:
        I_mix = fuel_split_I(
            omega, a, pilot, main,
            m_pilot_phi=m_pilot_phi, m_pilot_t=m_pilot_t,
            m_main_phi=m_main_phi, m_main_t=m_main_t,
            cfg=cfg
        )
        curves_I.append((a, I_mix))

    # Plot I magnitude/phase
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for a, I_mix in curves_I:
        axes[0].loglog(f, np.abs(I_mix), label=f"alpha={a:.2f}")
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('|I_mix(ω)|')
    axes[0].set_title('Fuel-split |I_mix| vs frequency')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for a, I_mix in curves_I:
        axes[1].semilogx(f, np.unwrap(np.angle(I_mix)), label=f"alpha={a:.2f}")
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Phase [rad]')
    axes[1].set_title('Fuel-split ∠I_mix vs frequency')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle('Fuel-split model (pilot/main) using Two-Delay Gamma stages')
    plt.tight_layout()
    plt.savefig('fuel_split_I_example.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'fuel_split_I_example.png'")

    # T22 projection: sweep the same alpha set for consistency with I_mix
    curves_T22 = []
    for a in alphas:
        T22 = T22_from_fuel_split(
            omega, a, pilot, main, T_ratio,
            m_pilot_phi=m_pilot_phi, m_pilot_t=m_pilot_t,
            m_main_phi=m_main_phi, m_main_t=m_main_t,
            cfg=cfg
        )
        curves_T22.append((a, T22))

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for a, T22 in curves_T22:
        axes2[0].loglog(f, np.abs(T22), label=f"alpha={a:.2f}")
    axes2[0].set_xlabel('Frequency [Hz]')
    axes2[0].set_ylabel('|T22(ω)|')
    axes2[0].set_title(f'T22 magnitude (T2/T1={T_ratio})')
    axes2[0].grid(alpha=0.3)
    axes2[0].legend()

    for a, T22 in curves_T22:
        axes2[1].semilogx(f, np.unwrap(np.angle(T22)), label=f"alpha={a:.2f}")
    axes2[1].set_xlabel('Frequency [Hz]')
    axes2[1].set_ylabel('Phase [rad]')
    axes2[1].set_title('T22 phase')
    axes2[1].grid(alpha=0.3)
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig('fuel_split_T22_example.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'fuel_split_T22_example.png'")

    return {
        'alphas': alphas,
        'pilot': pilot,
        'main': main,
        'm_shapes': (m_pilot_phi, m_pilot_t, m_main_phi, m_main_t),
        'T_ratio': T_ratio
    }

def run_grid_search_example():
    """Run grid search example to find optimal shape parameters"""
    print("\n\nGrid Search Example - Finding Optimal Shape Parameters")
    print("=" * 60)
    
    # Create synthetic data
    omega, mag, phase, true_params, m_phi_true, m_t_true = create_synthetic_two_delay_data()
    
    print(f"Generated {len(omega)} frequency points")
    print(f"Frequency range: {omega[0]/(2*np.pi):.2f} to {omega[-1]/(2*np.pi):.2f} Hz")
    print(f"True shape parameters: m_phi={m_phi_true}, m_t={m_t_true}")
    
    # Run grid search to find optimal shape parameters
    print("\nRunning grid search over shape parameters...")
    print("Testing m_phi, m_t in range [1, 5]...")
    
    best, results = fit_two_delay_gamma_grid(
        omega, mag, phase,
        phase_in_degrees=False,
        mphi_list=range(1, 6),  # Test m_phi from 1 to 5
        mt_list=range(1, 6),    # Test m_t from 1 to 5
        normalize=False,        # Data is already in I(omega) domain
        w_mag=1.0,
        w_phase=1.0,
        selection="rmse",       # Use RMSE for selection
        max_nfev=10000,         # Reduced for faster grid search
        suppress_warnings=True
    )
    
    # Print results
    print(f"\nGrid Search Results:")
    print(f"Best shape parameters: m_phi={best['m_phi']}, m_t={best['m_t']} (true: m_phi={m_phi_true}, m_t={m_t_true})")
    print(f"Best score (RMSE): {best['score']:.6f}")
    print(f"Success: {best['info']['success']}")
    print(f"Cost: {best['info']['cost']:.6f}")
    print(f"Log-magnitude RMSE: {best['info']['logmag_rmse']:.6f}")
    print(f"Phase RMSE [rad]: {best['info']['phase_rmse_rad']:.6f}")
    
    # Show top 5 results
    print(f"\nTop 5 shape parameter combinations:")
    sorted_results = sorted(results, key=lambda x: x['score'])
    for i, result in enumerate(sorted_results[:5]):
        print(f"  {i+1}. m_phi={result['m_phi']}, m_t={result['m_t']}: score={result['score']:.6f}")
    
    return best, results

def run_normalization_example():
    """Run normalization example with T22 data"""
    print("\n\nNormalization Example - T22 Data")
    print("=" * 40)
    
    # Create synthetic T22 data
    omega, mag, phase, true_params, m_phi_true, m_t_true, T_ratio = create_synthetic_T22_data()
    
    print(f"Generated {len(omega)} frequency points")
    print(f"Frequency range: {omega[0]/(2*np.pi):.2f} to {omega[-1]/(2*np.pi):.2f} Hz")
    print(f"Temperature ratio T2/T1: {T_ratio}")
    print(f"True shape parameters: m_phi={m_phi_true}, m_t={m_t_true}")
    
    # Fit with normalization using true shape parameters
    print(f"\nFitting with normalization (T22 -> I domain) using true shape parameters...")
    p_hat, info = fit_two_delay_gamma(
        omega, mag, phase,
        phase_in_degrees=False,
        m_phi=m_phi_true,
        m_t=m_t_true,
        normalize=True,     # Normalize T22 data to I domain
        T_ratio=T_ratio,    # Temperature ratio for normalization
        w_mag=1.0,
        w_phase=1.0,
        robust="soft_l1",
        f_scale=1.0,
        suppress_warnings=True
    )
    
    # Also test with grid search to compare
    print(f"\nFor comparison, testing grid search over shape parameters...")
    try:
        best, results = fit_two_delay_gamma_grid(
            omega, mag, phase,
            phase_in_degrees=False,
            mphi_list=range(1, 4),
            mt_list=range(1, 4),
            normalize=True,
            T_ratio=T_ratio,
            selection="rmse",
            max_nfev=3000,
            suppress_warnings=True
        )
        print(f"Grid search best: m_phi={best['m_phi']}, m_t={best['m_t']}, score={best['score']:.6f}")
    except Exception as e:
        print(f"Grid search failed: {e}")
        best = None
    
    # Print results
    print(f"\nNormalized Fit Results:")
    print(f"Success: {info['success']}")
    print(f"Cost: {info['cost']:.6f}")
    print(f"Function evaluations: {info['nfev']}")
    print(f"Log-magnitude RMSE: {info['logmag_rmse']:.6f}")
    print(f"Phase RMSE [rad]: {info['phase_rmse_rad']:.6f}")
    print(f"Fitted domain: {info['fitted_domain']}")
    
    print(f"\nFitted Parameters:")
    print(f"A_phi: {p_hat.A_phi:.4f} (true: {true_params.A_phi:.4f})")
    print(f"A_t: {p_hat.A_t:.4f} (true: {true_params.A_t:.4f})")
    print(f"tau_phi [s]: {p_hat.tau_phi:.6f} (true: {true_params.tau_phi:.6f})")
    print(f"r_tau: {p_hat.r_tau:.4f} (true: {true_params.r_tau:.4f})")
    print(f"theta_phi [s]: {p_hat.theta_phi:.6f} (true: {true_params.theta_phi:.6f})")
    print(f"theta_t [s]: {p_hat.theta_t:.6f} (true: {true_params.theta_t:.6f})")
    
    # Test T22 reconstruction
    T22_pred = info["to_T22"](omega, T_ratio)
    
    # Calculate true T22 from true parameters for comparison
    I_true = I_two_delay(omega, true_params, m_phi_true, m_t_true)
    T22_true = 1.0 + (T_ratio - 1.0) * I_true
    
    # Create plot comparing original T22 and reconstructed T22
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Magnitude plot
    axes[0].loglog(omega/(2*np.pi), mag, 'o', ms=4, alpha=0.7, label='Original T22 data')
    axes[0].loglog(omega/(2*np.pi), np.abs(T22_pred), '-', lw=2, label='Fitted T22')
    axes[0].loglog(omega/(2*np.pi), np.abs(T22_true), '--', lw=2, label='True T22 (ideal fit)')
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('|T22|')
    axes[0].set_title('T22 Magnitude')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Phase plot
    phi_unw = unwrap_phase(phase, deg=False)
    phase_pred = np.angle(T22_pred)
    phase_true = np.angle(T22_true)
    # Align phases for plotting
    phase_pred += np.round((phi_unw - phase_pred)/(2*np.pi))*2*np.pi
    phase_true += np.round((phi_unw - phase_true)/(2*np.pi))*2*np.pi
    
    axes[1].semilogx(omega/(2*np.pi), phi_unw, 'o', ms=4, alpha=0.7, label='Original T22 data')
    axes[1].semilogx(omega/(2*np.pi), phase_pred, '-', lw=2, label='Fitted T22')
    axes[1].semilogx(omega/(2*np.pi), phase_true, '--', lw=2, label='True T22 (ideal fit)')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Phase [rad]')
    axes[1].set_title('T22 Phase')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'T22 Normalization Example (T2/T1={T_ratio})')
    plt.tight_layout()
    plt.savefig('normalization_example.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'normalization_example.png'")
    
    return p_hat, info

def main():
    """Main example function - runs Two-Delay Gamma FTF examples"""
    print("Two-Delay Gamma FTF Model Examples")
    print("=" * 50)
    
    # Run examples
    print("Running basic two-delay gamma example...")
    two_delay_params, two_delay_info = run_two_delay_example()
    
    print("\nRunning grid search example...")
    best_grid, grid_results = run_grid_search_example()
    
    print("\nRunning normalization example...")
    norm_params, norm_info = run_normalization_example()

    print("\nRunning fuel-split example...")
    fs_info = run_fuel_split_example()
    
    # Summary comparison
    print("\n\nSummary:")
    print("=" * 20)
    print("Basic fit:")
    print(f"  Success: {two_delay_info['success']}, Cost: {two_delay_info['cost']:.6f}")
    print(f"  Log-mag RMSE: {two_delay_info['logmag_rmse']:.6f}, Phase RMSE: {two_delay_info['phase_rmse_rad']:.6f}")
    
    print("Grid search:")
    print(f"  Best shapes: m_phi={best_grid['m_phi']}, m_t={best_grid['m_t']}")
    print(f"  Success: {best_grid['info']['success']}, Cost: {best_grid['info']['cost']:.6f}")
    print(f"  Log-mag RMSE: {best_grid['info']['logmag_rmse']:.6f}, Phase RMSE: {best_grid['info']['phase_rmse_rad']:.6f}")
    
    print("Normalization:")
    print(f"  Success: {norm_info['success']}, Cost: {norm_info['cost']:.6f}")
    print(f"  Log-mag RMSE: {norm_info['logmag_rmse']:.6f}, Phase RMSE: {norm_info['phase_rmse_rad']:.6f}")
    
    print(f"\nGenerated plots:")
    print(f"  - two_delay_gamma_fit_example.png")
    print(f"  - normalization_example.png")
    print(f"  - fuel_split_I_example.png")
    print(f"  - fuel_split_T22_example.png")
    
    # Show plots only when not running headless (e.g., Agg backend)
    if "agg" not in mpl.get_backend().lower():
        plt.show()
    else:
        print("[Headless] Backend is Agg; skipping plt.show(). Figures saved.")

if __name__ == "__main__":
    main()

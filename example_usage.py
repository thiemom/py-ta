#!/usr/bin/env python3
"""
Example usage of both DTL and ZPK-Fractional FTF models from pyftf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the FTF functions
exec(open('./pyftf.py').read())

def create_synthetic_dtl_data():
    """Create synthetic DTL FTF data for testing"""
    # Frequency range
    omega = np.logspace(0, 3, 100)  # 1 to 1000 rad/s
    
    # True parameters for synthetic data
    true_params = DTLParams(
        S=2.5,
        tau0=0.001,
        thetas=np.array([0.01]),
        ks=np.array([1.2])
    )
    
    # Generate synthetic FTF
    H_true = H_dtl(omega, true_params)
    
    # Add some noise
    np.random.seed(42)
    noise_level = 0.05
    mag_noise = 1 + noise_level * np.random.randn(len(omega))
    phase_noise = noise_level * np.random.randn(len(omega))
    
    mag = np.abs(H_true) * mag_noise
    phase = np.angle(H_true) + phase_noise
    
    return omega, mag, phase, true_params

def create_synthetic_zpk_data():
    """Create synthetic ZPK-Fractional FTF data with rising-then-falling magnitude"""
    # Frequency range
    omega = np.logspace(0, 3, 100)  # 1 to 1000 rad/s
    
    # True parameters for synthetic ZPK data that creates rising-then-falling behavior
    # Zero at low frequency creates rising magnitude, pole at higher frequency creates fall-off
    true_params = ZPKFrac(
        S=0.5,                   # Lower gain to start
        tau0=0.001,              # Small delay
        z=np.array([0.02]),      # Zero time constant (creates rising behavior)
        a=np.array([1.0]),       # Zero exponent (20 dB/dec rise)
        p=np.array([0.005]),     # Pole time constant (creates fall-off)
        k=np.array([1.5])        # Pole exponent (30 dB/dec fall)
    )
    
    # Generate synthetic FTF
    H_true = H_zpk_frac(omega, true_params)
    
    # Add some noise
    np.random.seed(123)
    noise_level = 0.03
    mag_noise = 1 + noise_level * np.random.randn(len(omega))
    phase_noise = noise_level * np.random.randn(len(omega))
    
    mag = np.abs(H_true) * mag_noise
    phase = np.angle(H_true) + phase_noise
    
    return omega, mag, phase, true_params

def run_dtl_example():
    """Run DTL FTF model example"""
    print("DTL FTF Model Example")
    print("=" * 30)
    
    # Create synthetic data
    omega, mag, phase, true_params = create_synthetic_dtl_data()
    
    print(f"Generated {len(omega)} frequency points")
    print(f"Frequency range: {omega[0]/(2*np.pi):.2f} to {omega[-1]/(2*np.pi):.2f} Hz")
    
    # Fit the model
    print("\nFitting DTL model...")
    p_hat, info = fit_ftf(
        omega, mag, phase,
        phase_in_degrees=False,
        J=1,
        w_mag=1.0,
        w_phase=1.0,
        robust="soft_l1",
        f_scale=1.0
    )
    
    # Print results
    print(f"\nDTL Fit Results:")
    print(f"Success: {info['success']}")
    print(f"Cost: {info['cost']:.6f}")
    print(f"Function evaluations: {info['nfev']}")
    print(f"Log-magnitude RMSE: {info['logmag_rmse']:.6f}")
    print(f"Phase RMSE [rad]: {info['phase_rmse_rad']:.6f}")
    
    print(f"\nFitted Parameters:")
    print(f"S (gain): {p_hat.S:.4f} (true: {true_params.S:.4f})")
    print(f"tau0 [s]: {p_hat.tau0:.6f} (true: {true_params.tau0:.6f})")
    print(f"theta [s]: {p_hat.thetas[0]:.6f} (true: {true_params.thetas[0]:.6f})")
    print(f"k [-]: {p_hat.ks[0]:.4f} (true: {true_params.ks[0]:.4f})")
    
    # Create plot
    fig = plot_fit(omega, mag, phase, p_hat, 
                   phase_in_degrees=False, 
                   title="DTL FTF Model Fit Example")
    
    # Save plot
    plt.savefig('dtl_ftf_fit_example.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'dtl_ftf_fit_example.png'")
    
    return p_hat, info

def run_zpk_example():
    """Run ZPK-Fractional FTF model example"""
    print("\n\nZPK-Fractional FTF Model Example")
    print("=" * 40)
    
    # Create synthetic data
    omega, mag, phase, true_params = create_synthetic_zpk_data()
    
    print(f"Generated {len(omega)} frequency points")
    print(f"Frequency range: {omega[0]/(2*np.pi):.2f} to {omega[-1]/(2*np.pi):.2f} Hz")
    
    # Fit the ZPK model
    print("\nFitting ZPK-Fractional model...")
    p_hat, info = fit_ftf_zpk(
        omega, mag, phase,
        phase_in_degrees=False,
        R=1,  # 1 zero
        J=1,  # 1 pole
        w_mag=1.0,
        w_phase=1.0,
        robust="soft_l1",
        f_scale=1.0
    )
    
    # Print results
    print(f"\nZPK Fit Results:")
    print(f"Success: {info['success']}")
    print(f"Cost: {info['cost']:.6f}")
    print(f"Function evaluations: {info['nfev']}")
    print(f"Log-magnitude RMSE: {info['logmag_rmse']:.6f}")
    print(f"Phase RMSE [rad]: {info['phase_rmse_rad']:.6f}")
    
    print(f"\nFitted Parameters:")
    print(f"S (gain): {p_hat.S:.4f} (true: {true_params.S:.4f})")
    print(f"tau0 [s]: {p_hat.tau0:.6f} (true: {true_params.tau0:.6f})")
    print(f"z (zero time const) [s]: {p_hat.z[0]:.6f} (true: {true_params.z[0]:.6f})")
    print(f"a (zero exponent) [-]: {p_hat.a[0]:.4f} (true: {true_params.a[0]:.4f})")
    print(f"p (pole time const) [s]: {p_hat.p[0]:.6f} (true: {true_params.p[0]:.6f})")
    print(f"k (pole exponent) [-]: {p_hat.k[0]:.4f} (true: {true_params.k[0]:.4f})")
    
    # Create plot
    fig = plot_zpk_fit(omega, mag, phase, p_hat,
                       phase_in_degrees=False,
                       title="ZPK-Fractional FTF Model Fit Example")
    
    # Save plot
    plt.savefig('zpk_ftf_fit_example.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'zpk_ftf_fit_example.png'")
    
    return p_hat, info

def test_rising_falling_behavior():
    """Test both models on rising-then-falling magnitude data"""
    print("\n\nTesting Rising-Then-Falling Magnitude Behavior")
    print("=" * 55)
    
    # Create synthetic data with rising-then-falling behavior
    omega, mag, phase, true_zpk_params = create_synthetic_zpk_data()
    
    print(f"Generated {len(omega)} frequency points with rising-then-falling behavior")
    print(f"Frequency range: {omega[0]/(2*np.pi):.2f} to {omega[-1]/(2*np.pi):.2f} Hz")
    
    # Test DTL model (should struggle with rising behavior)
    print("\nTesting DTL model on rising-falling data...")
    dtl_params, dtl_info = fit_ftf(
        omega, mag, phase,
        phase_in_degrees=False,
        J=2,  # Try 2 poles to see if it helps
        w_mag=1.0,
        w_phase=1.0,
        robust="soft_l1",
        f_scale=1.0
    )
    
    print(f"DTL Results:")
    print(f"  Success: {dtl_info['success']}, Cost: {dtl_info['cost']:.6f}")
    print(f"  Log-mag RMSE: {dtl_info['logmag_rmse']:.6f}")
    print(f"  Phase RMSE: {dtl_info['phase_rmse_rad']:.6f}")
    
    # Test ZPK model (should handle rising-falling better)
    print("\nTesting ZPK model on rising-falling data...")
    zpk_params, zpk_info = fit_ftf_zpk(
        omega, mag, phase,
        phase_in_degrees=False,
        R=1,  # 1 zero for rising
        J=1,  # 1 pole for falling
        w_mag=1.0,
        w_phase=1.0,
        robust="soft_l1",
        f_scale=1.0
    )
    
    print(f"ZPK Results:")
    print(f"  Success: {zpk_info['success']}, Cost: {zpk_info['cost']:.6f}")
    print(f"  Log-mag RMSE: {zpk_info['logmag_rmse']:.6f}")
    print(f"  Phase RMSE: {zpk_info['phase_rmse_rad']:.6f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # DTL fit
    H_dtl_fit = H_dtl(omega, dtl_params)
    axes[0,0].loglog(omega/(2*np.pi), mag, 'o', ms=3, alpha=0.7, label='True data')
    axes[0,0].loglog(omega/(2*np.pi), np.abs(H_dtl_fit), '-', lw=2, label='DTL fit')
    axes[0,0].set_xlabel('Frequency [Hz]')
    axes[0,0].set_ylabel('|H|')
    axes[0,0].set_title(f'DTL Model (Cost: {dtl_info["cost"]:.3f})')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # ZPK fit
    H_zpk_fit = H_zpk_frac(omega, zpk_params)
    axes[0,1].loglog(omega/(2*np.pi), mag, 'o', ms=3, alpha=0.7, label='True data')
    axes[0,1].loglog(omega/(2*np.pi), np.abs(H_zpk_fit), '-', lw=2, label='ZPK fit')
    axes[0,1].set_xlabel('Frequency [Hz]')
    axes[0,1].set_ylabel('|H|')
    axes[0,1].set_title(f'ZPK Model (Cost: {zpk_info["cost"]:.3f})')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # Phase comparison
    phi_unw = unwrap_phase(phase, deg=False)
    axes[1,0].semilogx(omega/(2*np.pi), phi_unw, 'o', ms=3, alpha=0.7, label='True data')
    axes[1,0].semilogx(omega/(2*np.pi), np.angle(H_dtl_fit), '-', lw=2, label='DTL fit')
    axes[1,0].set_xlabel('Frequency [Hz]')
    axes[1,0].set_ylabel('Phase [rad]')
    axes[1,0].set_title('DTL Phase')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    axes[1,1].semilogx(omega/(2*np.pi), phi_unw, 'o', ms=3, alpha=0.7, label='True data')
    axes[1,1].semilogx(omega/(2*np.pi), np.angle(H_zpk_fit), '-', lw=2, label='ZPK fit')
    axes[1,1].set_xlabel('Frequency [Hz]')
    axes[1,1].set_ylabel('Phase [rad]')
    axes[1,1].set_title('ZPK Phase')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rising_falling_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved as 'rising_falling_comparison.png'")
    
    return dtl_params, dtl_info, zpk_params, zpk_info

def main():
    """Main example function - runs both DTL and ZPK examples"""
    print("FTF Model Examples - DTL and ZPK-Fractional")
    print("=" * 50)
    
    # Run standard examples
    dtl_params, dtl_info = run_dtl_example()
    zpk_params, zpk_info = run_zpk_example()
    
    # Test rising-falling behavior
    dtl_rf_params, dtl_rf_info, zpk_rf_params, zpk_rf_info = test_rising_falling_behavior()
    
    # Summary comparison
    print("\n\nSummary Comparison:")
    print("=" * 20)
    print("Standard synthetic data:")
    print(f"  DTL Model - Success: {dtl_info['success']}, Cost: {dtl_info['cost']:.6f}")
    print(f"  ZPK Model - Success: {zpk_info['success']}, Cost: {zpk_info['cost']:.6f}")
    print("\nRising-then-falling data:")
    print(f"  DTL Model - Success: {dtl_rf_info['success']}, Cost: {dtl_rf_info['cost']:.6f}")
    print(f"  ZPK Model - Success: {zpk_rf_info['success']}, Cost: {zpk_rf_info['cost']:.6f}")
    
    # Show plots (comment out if running headless)
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example usage of the DTL FTF model from py-ftf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the DTL FTF functions
exec(open('./py-ftf.py').read())

def create_synthetic_data():
    """Create synthetic FTF data for testing"""
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

def main():
    """Main example function"""
    print("DTL FTF Model Example")
    print("=" * 30)
    
    # Create synthetic data
    omega, mag, phase, true_params = create_synthetic_data()
    
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
    print(f"\nFit Results:")
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
    plt.savefig('ftf_fit_example.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'ftf_fit_example.png'")
    
    # Show plot (comment out if running headless)
    plt.show()

if __name__ == "__main__":
    main()

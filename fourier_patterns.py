"""
The code provides three main approaches:
Basic pattern decomposition:
pythonmodes = fourier_decompose_pattern([1,1,1,2,1,2,1,1], max_mode=5)
Weighted by coupling factors:
pythonmodes = fourier_decompose_weighted(pattern, gamma_values, max_mode=5)
Full analysis with burner parameters:
pythonresults = analyze_pattern_effectiveness(pattern, burner_params)
The key insight is in the coupling strength calculation - A_m represents how much the pattern couples to azimuthal mode m. When A_m ≈ 0, that mode sees minimal forcing from the burner pattern.
The find_optimal_pattern() function does brute force optimization to minimize coupling to a target mode, which is essentially what you've been doing manually in Excel.

Fourier Analysis of Burner Patterns in Annular Combustors
Key Physics
Azimuthal combustion instabilities in annular gas turbines arise from coupling between acoustic modes and flame dynamics. The fundamental mechanism involves:

Acoustic pressure oscillations traveling around the annular chamber
Flame response to these oscillations via the flame transfer function (FTF): q'(t) = n·u'(t-τ)
Energy feedback where heat release fluctuations can amplify or damp the acoustic mode

Parmentier et al. (2012) developed a network model treating this as burners connected to an annular duct, leading to the dispersion relation det(∏Tᵢ - I) = 0. Under weak coupling assumptions, this reduces to a simple quadratic equation involving the sum of coupling factors.
However, this summation approach ignores a critical physical reality: spatial phase relationships matter. A burner pattern's effectiveness depends not just on the total number of each burner type, but on how the pattern's spatial structure matches the acoustic mode shapes.
Fourier Decomposition Approach
Each burner pattern can be decomposed as:
Pattern(θ) = ∑ₘ Aₘ e^(imθ)
The key insight is that azimuthal mode m couples primarily to Fourier component Aₘ of the burner pattern. This captures the spatial coherence requirements that Parmentier's weak coupling analysis misses.
For each mode m at frequency ωₘ = mπc₀/L, the coupling strength depends on:

Aₘ: Spatial matching between pattern and mode shape
Flame transfer function: n·e^(jωₘτ) evaluated at the mode frequency
Pattern-mode phase relationship: Determines constructive vs destructive interference

Interpretation Guide
Azimuthal Modes

Mode 0: Uniform (monopole) - affects all burners equally
Mode 1: First azimuthal (~90 Hz for your geometry) - single pressure variation around circumference
Mode 2: Second azimuthal (~180 Hz) - two pressure nodes/antinodes around circumference
Mode m: m pressure nodes/antinodes, frequency = m×90 Hz

Output Parameters
Coupling Strength: Magnitude |Aₘ|

> 1.0: Strong coupling - pattern efficiently drives this mode
0.1-1.0: Moderate coupling - some interaction possible
< 0.1: Weak coupling - pattern poorly matches this mode
≈ 0: Minimal coupling - pattern has nodes at same locations as acoustic mode

Phase: ∠Aₘ

0 to π/2: Pattern leads acoustic mode - potential for instability amplification
π/2 to π: Pattern lags acoustic mode - potential for damping
Phase relationships between different burner types create interference effects

Physical Meaning
For your N=15 system:

Mode frequencies: 90, 180, 270, 360, 450 Hz for modes 1-5
Pattern [1,1,2]×5: Strong coupling expected for modes 5,10,15 (multiples of 5)
Pattern [1,1,2,1,2,1,2,2,1,1,1,1,2,1,1]: More distributed coupling across different modes

Practical Application
Use this analysis to:

Identify vulnerable modes: High coupling strength indicates modes likely to be unstable
Design control patterns: Place burners to minimize coupling to problematic modes
Validate simplified models: Compare Fourier predictions with Parmentier's summation approach
Optimize burner placement: Find patterns that preferentially damp target modes while maintaining combustion performance

The Fourier approach reveals why geometric considerations often dominate over simple burner counting in real combustor design.
"""

import numpy as np
import matplotlib.pyplot as plt

def fourier_decompose_pattern(pattern, max_mode=10):
    """
    pattern: list/array of burner types (e.g., [1,1,1,2,1,2,1,1,...])
    max_mode: maximum azimuthal mode order to compute
    """
    N = len(pattern)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Convert pattern to complex amplitude (could weight by Gamma_i)
    pattern_array = np.array(pattern, dtype=complex)
    
    # Fourier coefficients
    modes = {}
    for m in range(max_mode + 1):
        if m == 0:
            # DC component
            A_m = np.mean(pattern_array)
        else:
            # Complex exponential basis
            basis = np.exp(-1j * m * theta)
            A_m = np.sum(pattern_array * basis) / N
            
        modes[m] = A_m
    
    return modes

def fourier_decompose_weighted(pattern, coupling_factors, max_mode=10):
    """
    pattern: burner type pattern
    coupling_factors: list of Gamma_i for each burner position
    """
    N = len(pattern)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    weighted_pattern = np.array(coupling_factors, dtype=complex)
    
    modes = {}
    for m in range(max_mode + 1):
        if m == 0:
            A_m = np.sum(weighted_pattern) / N
        else:
            basis = np.exp(-1j * m * theta)
            A_m = np.sum(weighted_pattern * basis) / N
            
        modes[m] = A_m
    
    return modes

def pattern_mode_coupling(pattern, burner_params, chamber_params, max_mode=5):
    """
    burner_params: dict with keys 'n', 'tau' containing arrays for each burner type
    chamber_params: dict with 'L' (half-perimeter) and 'c0' (sound speed). For an annular chamber, 'L' is pi * R (acoustic).
    """
    N = len(pattern)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    L = chamber_params['L']
    c0 = chamber_params['c0']
    
    modes = {}
    for m in range(max_mode + 1):
        # Mode-specific reference frequency
        if m == 0:
            omega_m = 0  # DC component
            A_m = sum(burner_params['n'][pattern[i]-1] for i in range(N))
        else:
            omega_m = m * np.pi * c0 / L  # Azimuthal mode frequency
            
            # Build coupling factors for this specific mode
            coupling_factors = []
            for i, btype in enumerate(pattern):
                n_i = burner_params['n'][btype-1]
                tau_i = burner_params['tau'][btype-1]
                
                # Proper FTF evaluation at mode frequency
                gamma_i = n_i * np.exp(1j * omega_m * tau_i)
                coupling_factors.append(gamma_i)
            
            # Fourier decomposition
            basis = np.exp(-1j * m * theta)
            A_m = np.sum(np.array(coupling_factors) * basis)
            
        modes[m] = A_m
    
    return modes

def analyze_pattern_effectiveness(pattern, burner_params, chamber_params):
    """
    Analyze which modes are most/least coupled for given pattern
    """
    modes = pattern_mode_coupling(pattern, burner_params, chamber_params)
    
    results = {}
    for m, A_m in modes.items():
        magnitude = abs(A_m)
        phase = np.angle(A_m)
        results[m] = {
            'magnitude': magnitude,
            'phase': phase,
            'coupling_strength': magnitude / abs(modes[0]) if modes[0] != 0 else 0
        }
    
    return results

def compare_patterns(patterns, labels, burner_params, chamber_params):
    """
    Compare multiple patterns
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for pattern, label in zip(patterns, labels):
        results = analyze_pattern_effectiveness(pattern, burner_params, chamber_params)
        
        modes = list(results.keys())
        magnitudes = [results[m]['magnitude'] for m in modes]
        phases = [results[m]['phase'] for m in modes]
        
        ax1.plot(modes, magnitudes, 'o-', label=label)
        ax2.plot(modes, phases, 's-', label=label)
    
    ax1.set_xlabel('Azimuthal Mode Order')
    ax1.set_ylabel('Coupling Magnitude |A_m|')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Azimuthal Mode Order')
    ax2.set_ylabel('Phase (rad)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def find_optimal_pattern(N, n_type2_burners, burner_params, target_mode):
    """
    Brute force search for pattern that minimizes coupling to target_mode
    """
    from itertools import combinations
    
    best_pattern = None
    min_coupling = float('inf')
    
    # Generate all possible positions for type 2 burners
    for positions in combinations(range(N), n_type2_burners):
        pattern = [1] * N
        for pos in positions:
            pattern[pos] = 2
            
        results = analyze_pattern_effectiveness(pattern, burner_params)
        coupling = results[target_mode]['magnitude']
        
        if coupling < min_coupling:
            min_coupling = coupling
            best_pattern = pattern.copy()
    
    return best_pattern, min_coupling



# Example usage
if __name__ == "__main__":
    # N = 15 burners patterns
    pattern_symmetric = [1,1,2] * 5
    pattern_nonsymmetric = [1,1,2, 1,2,1, 2,2,1, 1,1,1, 1,2,1]
    
    # Chamber parameters (from Parmentier's Table 1)
    chamber_params = {
        'L': 6.59,    # Half perimeter [m]
        'c0': 1191    # Sound speed [m/s]
    }
    
    # Burner parameters
    burner_params = {
        'n': [1.57, 1.57],      # Type 1, Type 2 interaction indices
        'tau': [0.007, 0.012]   # Type 1, Type 2 delays [s]
    }
    
    # Analyze both patterns
    results_sym = analyze_pattern_effectiveness(pattern_symmetric, burner_params, chamber_params)
    results_nonsym = analyze_pattern_effectiveness(pattern_nonsymmetric, burner_params, chamber_params)
    
    # Show mode frequencies
    L, c0 = chamber_params['L'], chamber_params['c0']
    print("Azimuthal mode frequencies:")
    for m in range(1, 6):
        freq = m * c0 / (2 * L)
        print(f"Mode {m}: {freq:.1f} Hz")
    
    print("\nSymmetric pattern coupling strengths:")
    for m in range(6):
        if m in results_sym:
            print(f"Mode {m}: {results_sym[m]['coupling_strength']:.3f}")
    
    print("\nNon-symmetric pattern coupling strengths:")
    for m in range(6):
        if m in results_nonsym:
            print(f"Mode {m}: {results_nonsym[m]['coupling_strength']:.3f}")
    
    # Compare visually
    fig = compare_patterns([pattern_symmetric, pattern_nonsymmetric], 
                          ['Symmetric 112x5', 'Non-symmetric'], 
                          burner_params, chamber_params)
    plt.show()



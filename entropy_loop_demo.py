#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# entropy_loop_demo.py
# Demo for entropy_loop.py with:
# - PSD of analytic model (OU, burst, total)
# - Time-trace synthesis (pocket-only) and FFT analysis:
#     * Linear-average FFT magnitude
#     * Peak-hold FFT magnitude
#
# Notes:
# - "Linear-average" here = average |FFT| across segments (not dB averaging).
# - "Peak-hold" keeps max |FFT| across segments.
# - For power spectra, switch avg_mode="power" below.
#
# Quick start (see module docstring in `entropy_loop.py`):
#
#     import numpy as np
#     import entropy_loop as el
#     f = np.linspace(0.1, 250, 2000)
#     w = 2*np.pi*f
#     thermo = el.ThermoProps(); flame = el.FlameMap()
#     ou = el.OUDrive(); bake = el.BakeModel()
#     gate = el.GateModel(); gains = el.LoopGains()
#     S = el.S_pp(w, thermo, flame, ou, bake, gate, gains)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import entropy_loop as el  # ensure entropy_loop.py is in the same folder

# ---------------- FFT helpers ----------------

def _get_window(n, kind="hann"):
    if kind is None:
        return np.ones(n)
    kind = kind.lower()
    if kind == "hann":
        return np.hanning(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind == "rect":
        return np.ones(n)
    raise ValueError(f"unknown window: {kind}")

def segment_fft(x, fs, nfft=8192, overlap=0.5, window="hann", avg_mode="magnitude"):
    """
    Segment x with given nfft and overlap, return:
      f   : frequency bins [Hz] (one-sided)
      avg : linear-average spectrum (|FFT| or power) across segments
      pkh : peak-hold spectrum (|FFT| or power) across segments

    avg_mode: "magnitude" (|X|), or "power" (|X|^2 / (fs * U)), where
              U is window power normalization (Welch-type).
    """
    x = np.asarray(x, float)
    step = int(nfft * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap too large; step <= 0")
    w = _get_window(nfft, window)
    U = np.sum(w**2)  # window power (for power scaling)
    # number of usable segments
    nseg = 1 + (len(x) - nfft) // step
    if nseg <= 0:
        raise ValueError("time series too short for given nfft/overlap")
    # accumulate
    mag_acc = None
    peak = None
    for i in range(nseg):
        s = i * step
        seg = x[s:s+nfft]
        segw = seg * w
        X = np.fft.rfft(segw, n=nfft)
        if avg_mode == "magnitude":
            val = np.abs(X)
        elif avg_mode == "power":
            # single-sided power spectral density estimate (Welch-like units)
            # Note: here we report power spectrum, not PSD-per-Hz.
            val = (np.abs(X)**2) / (U)
        else:
            raise ValueError("avg_mode must be 'magnitude' or 'power'")
        mag_acc = val if mag_acc is None else (mag_acc + val)
        peak = val if peak is None else np.maximum(peak, val)

    avg = mag_acc / nseg
    f = np.fft.rfftfreq(nfft, d=1/fs)
    # If avg_mode == "power" to get PSD per Hz, divide by fs outside.
    return f, avg, peak

# ---------------- Main demo ----------------

def main():
    parser = argparse.ArgumentParser(description="Entropy loop demo (analytic PSD + time-trace FFT)")
    parser.add_argument("--mode", choices=["binary", "fractional"], default="binary", help="Gate mode for refined LBO gate")
    parser.add_argument("--avg-mode", choices=["magnitude", "power"], default="magnitude", help="FFT averaging mode")
    parser.add_argument("--nfft", type=int, default=8192, help="FFT size per segment")
    parser.add_argument("--overlap", type=float, default=0.5, help="Segment overlap fraction (0..0.9)")
    parser.add_argument("--window", choices=["hann", "hamming", "rect"], default="hann", help="FFT window type")
    parser.add_argument("--T", type=float, default=20.0, help="Time-trace duration [s]")
    parser.add_argument("--dt", type=float, default=2e-4, help="Sample time [s]")
    parser.add_argument("--m", type=float, default=1.8, help="Stability margin proxy")
    parser.add_argument("--uref", type=float, default=5.0, help="Reference velocity scale for gate sensitivity")
    parser.add_argument("--pref", type=float, default=1000.0, help="Reference pressure scale [Pa] for gate sensitivity")
    parser.add_argument("--save-prefix", type=str, default="run", help="Prefix for saved figures")
    args = parser.parse_args()

    # ---- Analytic PSD ----
    fmax = 250.0
    Nf = 2000
    f = np.linspace(0.1, fmax, Nf)  # Hz
    w = 2*np.pi*f

    thermo = el.ThermoProps(gamma=1.35, R=287.0, T0=800.0, p0=1.5e6)
    flame  = el.FlameMap(T_hot=2100.0, T_cold=800.0, phi_op=0.7, tau_res=8e-3, sigma_tau=1.5e-3)
    ou     = el.OUDrive(D=0.004, t_c=0.12)
    bake   = el.BakeModel(M=0.35, M0=0.6, sigma_M=0.2, phi_max=0.5*np.pi)
    gate   = el.GateModel(m=1.8, lambda0_off=2.0, lambda0_on=2.0, beta=0.9, alpha_T=0.6,
                          tau_adv=0.045, sigma_disp=0.018)
    gains  = el.LoopGains(chi=0.8, include_mean_gate=True)

    S = el.S_pp(w, thermo, flame, ou, bake, gate, gains)
    S_total, S_ou, S_shot = S["S_total"], S["S_ou"], S["S_shot"]

    f_peak = f[np.argmax(S_total)]
    print(f"[Analytic] Peak total PSD around ~ {f_peak:.1f} Hz; p_on={gate.p_on():.3f}, λ_sw={gate.lambda_switch():.3f} 1/s")

    plt.figure(figsize=(7,4))
    plt.loglog(f, S_total, label="Total (analytic)")
    plt.loglog(f, S_ou, "--", label="OU path (analytic)")
    plt.loglog(f, S_shot, ":", label="Burst path (analytic)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$S_{pp}$ [Pa$^2$/Hz]")
    plt.title("Analytic PSD: breathing mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.save_prefix}_analytic_psd.png", dpi=150)

    # ---- Time-trace synthesis with refined gate ----
    T = float(args.T)
    dt = float(args.dt)
    fs = 1.0/dt  # Hz

    # Series
    N = int(T/dt)
    m_series = np.full(N, float(args.m))
    u_series = np.zeros(N)
    p_series = np.zeros(N)

    # Refined gate (defaults match prior demo when mode=binary)
    gate_ref = el.GateRefined(
        mode=args.mode,
        lambda0_off=2.0,
        lambda0_on=2.0,
        beta_m_off=0.9,
        beta_m_on=0.9,
        kappa_u=0.0,
        kappa_p=0.0,
        u_ref=float(args.uref),
        p_ref=float(args.pref),
        t_ref_after_off=0.10,
        t_ref_after_on=0.05,
        alpha_T=0.6,
        tau_adv=0.045,
        sigma_disp=0.018,
    )

    g, y_T_over_T0 = el.simulate_pockets_refined(T, dt, gate_ref, flame, m_series, u_series, p_series)
    # Simple mapping to pressure using real(G) to get a real waveform
    p_trace = (thermo.gamma - 1.0) * thermo.p0 * np.real(bake.G()) * y_T_over_T0

    # ---- Linear-average & Peak-hold FFTs ----
    nfft = int(args.nfft)
    overlap = float(args.overlap)
    window = args.window
    avg_mode = args.avg_mode  # or "power"

    f_fft, avg_spec, peak_spec = segment_fft(p_trace, fs, nfft=nfft, overlap=overlap,
                                             window=window, avg_mode=avg_mode)

    # For "power" mode to get PSD per Hz => divide by fs
    if avg_mode == "power":
        avg_plot = avg_spec / fs
        peak_plot = peak_spec / fs
        ylab = "Power spectral density [Pa^2/Hz]"
    else:
        avg_plot = avg_spec
        peak_plot = peak_spec
        ylab = "Linear magnitude [Pa]"

    plt.figure(figsize=(7,4))
    plt.semilogx(f_fft[1:], 20*np.log10(np.maximum(avg_plot[1:], 1e-20)), label="Linear-average FFT")
    plt.semilogx(f_fft[1:], 20*np.log10(np.maximum(peak_plot[1:], 1e-20)), label="Peak-hold FFT", alpha=0.8)
    plt.xlim(0.5, fs/2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Level [dB ref 1 Pa]" if avg_mode=="magnitude" else "Level [dB]")
    plt.title("Time-trace spectra: linear-average vs peak-hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.save_prefix}_time_trace_fft.png", dpi=150)
    # Optional: plot the gate trace for diagnostics
    try:
        t = np.arange(N)*dt
        plt.figure(figsize=(7,2.2))
        plt.plot(t, g, lw=0.8)
        plt.xlabel("Time [s]")
        plt.ylabel("g(t)")
        plt.title(f"Gate trace (mode={args.mode})")
        plt.tight_layout()
        plt.savefig(f"{args.save_prefix}_gate_trace.png", dpi=150)
    except Exception:
        pass
    # Show only when not running headless (e.g., Agg backend)
    if "agg" not in mpl.get_backend().lower():
        plt.show()
    else:
        print("[Headless] Backend is Agg; skipping plt.show(). Figures saved.")

if __name__ == "__main__":
    main()
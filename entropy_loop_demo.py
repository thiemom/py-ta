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
    plt.savefig("analytic_psd.png", dpi=150)

    # ---- Time-trace synthesis (pocket-only) ----
    # Make a pocket-only T'/T0, then map to pressure p'(t) approximately
    T = 20.0     # seconds
    dt = 2e-4    # sample time
    fs = 1.0/dt  # Hz

    y_T_over_T0 = el.simulate_bursts(T, dt, gate, flame)
    # Simple mapping to pressure using real(G) to get a real waveform;
    # if you want phase-correct filtering, apply complex filter in freq domain.
    p_trace = (thermo.gamma - 1.0) * thermo.p0 * np.real(bake.G()) * y_T_over_T0

    # ---- Linear-average & Peak-hold FFTs ----
    nfft = 8192
    overlap = 0.5
    window = "hann"
    avg_mode = "magnitude"  # or "power"

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
    plt.savefig("time_trace_fft.png", dpi=150)
    # Show only when not running headless (e.g., Agg backend)
    if "agg" not in mpl.get_backend().lower():
        plt.show()
    else:
        print("[Headless] Backend is Agg; skipping plt.show(). Figures saved.")

if __name__ == "__main__":
    main()
# Flame Transfer Function (FTF) Fits — Quick Notes

This note summarizes two fit families for flame transfer functions and cites key references.
It uses simple Unicode characters and Markdown math. If your viewer does not render inline
math `$...$`, the plain-text/Unicode forms are still readable.

---

## 1) Distributed Time Lag (DTL) / Fractional Pole Fit

**Equation (Unicode/plain):**  
`H(ω) = S · exp(−i ω τ₀) · ∏ⱼ (1 + i ω θⱼ)^(−kⱼ)`

**Equation (LaTeX):**  
$H(\omega) = S\,e^{-\,i\,\omega\,	au_0}\,\prod_j\left(1 + i\,\omega\,	heta_jight)^{-k_j}$

**Meaning**
- `S` = gain scale (dimensionless)  
- `τ₀` = bulk delay [s]  
- `θⱼ` > 0 = time-widths [s] (distributed convective lags)  
- `kⱼ` > 0 = fractional exponents (shape)

**Features**
- Models a *distribution* of convective delays → smooth low-/band-pass |H| and consistent phase lag.  
- Works well when |H| decays monotonically; avoids brittle single-τ fits.

**When to choose**
- Magnitude shows no pronounced pre-peak rise; phase behaves like a delay with gradual curvature.

---

## 2) ZPK‑Fractional (Zeros + Poles with Fractional Exponents)

**Equation (Unicode/plain):**  
`H(ω) = S · exp(−i ω τ₀) · [∏ᵣ (1 + i ω zᵣ)^(aᵣ)] / [∏ⱼ (1 + i ω θⱼ)^(kⱼ)]`

**Equation (LaTeX):**  
$H(\omega)=S\,e^{-\,i\,\omega\,	au_0}\,\dfrac{\prod_{r}\left(1+i\,\omega\,z_right)^{a_r}}{\prod_{j}\left(1+i\,\omega\,	heta_jight)^{k_j}}$

**Meaning**
- Adds numerator zeros `(zᵣ, aᵣ)` to capture shoulders/rises and phase lead.  
- Denominator poles `(θⱼ, kⱼ)` give decay and lag.

**Features**
- Captures FTF with a mild hump or initial rise; introduces controllable phase lead–lag.

**When to choose**
- Measured |H| rises or has a shoulder; phase shows early lead before lagging.

---

## 3) Two‑Pathway (Lieuwen Decomposition)

**Equation (Unicode/plain):**  
`I(ω) = A_φ · G_φ(ω) − A_t · G_t(ω)`  
where `G_i(ω) = exp(−i ω τ_i) · (1 + i ω θ_i)^(−k_i)`

**Equation (LaTeX):**  
$I(\omega) = A_\phi \, G_\phi(\omega) - A_t \, G_t(\omega)$  
where $G_i(\omega) = e^{-i\omega\tau_i} \left(1 + i\omega\theta_i\right)^{-k_i}$

**Meaning**
- `A_φ` = equivalence ratio pathway amplitude (dimensionless)  
- `A_t` = turbulence/velocity pathway amplitude (dimensionless)  
- `G_φ(ω)`, `G_t(ω)` = Gamma‑distributed delay kernels for each pathway  
- Positive ER pathway, negative turbulence pathway → interference patterns

**Features**
- Models flame response as superposition of two physical mechanisms.  
- Captures complex interference, zero crossings, and phase jumps naturally.  
- Each pathway has independent delay distribution parameters.

**When to choose**
- Complex FTF with interference patterns, magnitude nulls, or phase discontinuities.  
- When physical insight into ER vs. turbulence contributions is desired.

---

## Historical context (very brief)

- **1950s:** Crocco & Cheng — system-theory view of combustion instability (zeros/poles).  
- **1990s:** Dowling — thermoacoustic oscillation models using ZPK transfer functions.  
- **2000s:** Schuermans & Polifke — industrial practice, low‑order flame models, DTL generalizations.  
- **2010s–2020s:** Lieuwen, Noiray, Polifke — robust ID, stochastic/limit-cycle links, modern reviews.

---

## Key References

**DTL / Fractional‑Pole**
- Polifke, W. (2020). *System identification of combustion dynamics by means of CFD/LES: FTF, FDF and DTL*. **Prog. Energy Combust. Sci.**  
- Schuller, T., Noiray, N., Poinsot, T., Candel, S. (2020). *Dynamics of premixed flames and combustion instabilities*. **J. Fluid Mech.**  
- Lieuwen, T. (2012). *Unsteady Combustor Physics*. CUP.  
- Noiray, N. & Schuermans, B. (2013). *Deterministic quantities characterizing noise driven Hopf bifurcations in gas turbine combustors*. **Int. J. Non‑Linear Mech.**

**ZPK‑Fractional**
- Dowling, A.P. (1995). *The calculation of thermoacoustic oscillations*. **J. Sound Vib.**  
- Polifke, W. & Lawn, C. (2007). *Low‑order modelling of flames for combustion dynamics*. CTR Summer Program.  
- Freitag, M. (2018). *System identification of distributed time delay models for flames*. PhD Thesis, TUM.  
- Schuermans, B., Bellucci, V., Geigle, K., Paschereit, C.O. (2004). *Thermoacoustic modeling and control of a full‑scale gas turbine combustor*. **Combust. Sci. Tech.**  
- Crocco, L. & Cheng, S.I. (1956). *Theory of combustion instability in liquid propellant rocket motors*. AGARD.

**Two‑Pathway**
- Lieuwen, T. (2012). *Unsteady Combustor Physics*. CUP. (Chapter on distributed delays and multi‑pathway decomposition)  
- Schuller, T., Durox, D., Candel, S. (2003). *A unified model for the prediction of laminar flame transfer functions*. **Combust. Flame**  
- Noiray, N. & Schuermans, B. (2013). *Deterministic quantities characterizing noise driven Hopf bifurcations in gas turbine combustors*. **Int. J. Non‑Linear Mech.**

---

### Notes on usage
- If Markdown math doesn’t render in your viewer, use the **Unicode/plain** equations above (they are copy‑paste‑ready).  
- For LaTeX documents, copy the **LaTeX** equations directly.  
- If you need a printable PDF, consider converting this Markdown with a LaTeX‑enabled renderer (e.g., Pandoc with MathJax or LaTeX).


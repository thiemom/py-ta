# Flame Transfer Function (FTF) Fits — Quick Notes

This note summarizes the Two-Delay Gamma FTF model implemented in this package and cites key references.
It uses simple Unicode characters and Markdown math. If your viewer does not render inline
math `$...$`, the plain-text/Unicode forms are still readable.

---

## Two-Delay Gamma FTF Model

**Equation (Unicode/plain):**  
`I(ω) = A_φ · G_φ(ω) − A_t · G_t(ω)`  
where `G_i(ω) = exp(−i ω (τ_i + θ_i)) · Γ(m_i) / (Γ(m_i) + (i ω τ_i)^m_i)`

**Equation (LaTeX):**  
$I(\omega) = A_\phi \, G_\phi(\omega) - A_t \, G_t(\omega)$  
where $G_i(\omega) = e^{-i\omega(\tau_i + \theta_i)} \frac{\Gamma(m_i)}{\Gamma(m_i) + (i\omega\tau_i)^{m_i}}$

**Meaning**
- `A_φ` = equivalence ratio pathway amplitude (dimensionless)  
- `A_t` = turbulence/velocity pathway amplitude (dimensionless)  
- `τ_φ`, `τ_t` = characteristic delay times [s] for each pathway
- `θ_φ`, `θ_t` = additional phase delays [s] for each pathway  
- `m_φ`, `m_t` = integer shape parameters controlling delay distribution width
- `G_φ(ω)`, `G_t(ω)` = Gamma-distributed delay kernels for each pathway  
- Positive ER pathway, negative turbulence pathway → interference patterns

**Features**
- Models flame response as superposition of two physical mechanisms (equivalence ratio and turbulence).  
- Uses Gamma distributions for realistic delay spreading in each pathway.
- Captures complex interference, zero crossings, and phase jumps naturally.  
- Each pathway has independent delay distribution parameters.
- Shape parameters `m_φ`, `m_t` control the width of delay distributions (higher = narrower).

**When to choose**
- Complex FTF with interference patterns, magnitude nulls, or phase discontinuities.  
- When physical insight into ER vs. turbulence contributions is desired.
- Suitable for both direct interaction index `I(ω)` fitting and normalized T22 data.

**Data Normalization**
The model can handle both:
- **Direct I(ω) fitting**: When data is already in interaction index domain
- **T22 normalization**: Raw T22 data is converted to I(ω) using `I(ω) = (T22(ω) - 1) / (T2/T1 - 1)`

---

## Historical context (very brief)

- **1950s:** Crocco & Cheng — system-theory view of combustion instability (zeros/poles).  
- **1990s:** Dowling — thermoacoustic oscillation models using ZPK transfer functions.  
- **2000s:** Schuermans & Polifke — industrial practice, low‑order flame models, DTL generalizations.  
- **2010s–2020s:** Lieuwen, Noiray, Polifke — robust ID, stochastic/limit-cycle links, modern reviews.

---

## Key References

**Two-Delay Gamma Model**
- Lieuwen, T. (2012). *Unsteady Combustor Physics*. Cambridge University Press. (Chapter on distributed delays and multi-pathway decomposition)  
- Schuller, T., Durox, D., Candel, S. (2003). *A unified model for the prediction of laminar flame transfer functions*. **Combust. Flame**  
- Noiray, N. & Schuermans, B. (2013). *Deterministic quantities characterizing noise driven Hopf bifurcations in gas turbine combustors*. **Int. J. Non-Linear Mech.**
- Polifke, W. (2020). *System identification of combustion dynamics by means of CFD/LES: FTF, FDF and DTL*. **Prog. Energy Combust. Sci.**  
- Schuller, T., Noiray, N., Poinsot, T., Candel, S. (2020). *Dynamics of premixed flames and combustion instabilities*. **J. Fluid Mech.**

---

### Notes on usage
- If Markdown math doesn’t render in your viewer, use the **Unicode/plain** equations above (they are copy‑paste‑ready).  
- For LaTeX documents, copy the **LaTeX** equations directly.  
- If you need a printable PDF, consider converting this Markdown with a LaTeX‑enabled renderer (e.g., Pandoc with MathJax or LaTeX).


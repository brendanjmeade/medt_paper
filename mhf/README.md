# mhf — mollified half-space, fixed

The **clean semi-numerical build** of the exact-BC mollified Mindlin
half-space triangle-dislocation kernel: just the production path, all known
bugs fixed, nothing legacy.  Created 2026-06-11 from the post-review state
of `mh/` + `mollified_kernel/`.

## What this is

For a triangular dislocation in an elastic half-space (z ≤ 0) with Cortez
mollification `r → √(r² + ε²)`:

- **Direct part** — mollified Kelvin, integrated in closed form over the
  triangle (`analytical_kernels.py`: Van Oosterom solid angle, edge
  antiderivatives, moment recursions).  Accurate on and near the fault.
- **Correction part** — exact-BC complementary Papkovich–Neuber field at
  effective image depth `a = √(src_z² + ε²)` with closed-form coefficients
  (`exact_bc_coeffs.py`).  Its free-surface traction cancels the direct
  part's **pointwise in the source**, so the triangle's BC residual is
  *pure quadrature error with no ε floor* — spectrally convergent, at
  machine precision by n_quad≈16 (tensor, buried) and by graded rules of
  a few hundred points (surface-breaking); measured 4e-5 → 2.6e-16 over
  n_quad 4→16, and 9e-6 → 6e-15 over 120→768 graded points
  (`demo_validation_figure.py`).  Integrated by tensor Gauss
  (`n_quad²` points) or the obs-dependent sinh2D **graded** rule
  (`exact_bc_graded_quad.py`), which keeps small-ε rungs converged for
  surface-breaking faults and falls back to a tensor rule for
  (near-)horizontal triangles.
- **Richardson/Romberg ε-extrapolation** (`eps_extrapolation.py`) — cancels
  the intrinsic O(ε) near-trace error of surface-breaking faults
  (levels=2 → O(ε²)); preserves the exact BC by linearity.
- **JAX twin** (`exact_bc_jax.py`, optional) — jitted/vmapped influence
  matrices, cached per configuration; CPU ≈ 6–10× cutde per pair.

## API

```python
import mhf

# --- single triangle, scalar API -------------------------------------------
u = mhf.disp_hs_mindlin(obs_pts, tri, slip, nu, eps)           # buried fault
f = mhf.field_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=30.0, # near-surface,
                         n_quad=16, graded=True, n_c=20, n_t=16,
                         richardson_levels=2)                  # surface-breaking
f["disp"], f["grad_u"], f["strain"], f["stress"]

# --- grids + multi-triangle batch (one-stop shopping) ----------------------
tris = mhf.make_rect_fault(half_len=5.0, depth=10.0)           # 2-triangle fault
X, Y, obs = mhf.build_grid(half_extent=15.0, n=121, z0=-2.0)   # map-view slice
u, sig = mhf.eval_fields_fault_aware(obs, tris,
                                     slip=(1e-3, 0, 0),        # [s,d,t] in km
                                     eps=1.0, graded=True,
                                     richardson_levels=2)      # u: Richardson
                                                               # sig: blended,
                                                               # ELASTIC (C:ε*
                                                               # subtracted)
Y, Z, obs = mhf.build_grid_section(x0=0.0, y_half=20.0)        # cross-section

K = mhf.jax_kernels(n_quad=8)                                  # JAX (optional)
U = K["disp_matrix"](obs_pts, v1s, v2s, v3s, mu, nu, eps)      # (N,M,3,3)
```

Ready-made figures (3×3 panels: u + 6 stress components, CLI-configurable —
`--slip S D T`, `--eps`, `--graded`, `--richardson-levels`, geometry):

```bash
python -m mhf.demo_mapview                       # horizontal slice
python mhf/demo_xsection.py --graded --richardson-levels 2   # plain-script style works too
```

(Both invocation styles work, from any directory.  On macOS the package
defaults JAX to CPU automatically — jax-metal is unreliable; on CUDA boxes
the GPU is picked up as usual.)

Conventions (cutde-compatible): x East, y North, z Up; free surface z=0;
slip `[strike, dip, tensile]`; normal by right-hand rule on vertex order;
`V_strike = ẑ × V_normal`.  Units scale-free (repo convention km/GPa).

Recommended knobs: ε ≈ fault_width/10; buried faults need no knobs;
surface-breaking faults observed near the trace: `graded=True, n_c=20,
n_t=16, richardson_levels=2` (the fault-aware evaluator keeps the single-ε
value ON the fault and, by default, subtracts the anelastic eigenstress
there — see "Stress readout" below — so on-fault stress is the bounded
elastic value).

## Stress readout: subtract the anelastic eigenstress

A fault slip is an **anelastic** (inelastic / eigen-) strain.  The mollified
kernel therefore returns the *total* stress `C:(ε_el + ε*)`; inside the
~ε-wide smeared fault zone it is dominated by the eigenstress `C:ε*` — the
smeared slip itself.  Its fault-normal profile is the Cortez blob's marginal

    γ(y) = s · (3/4) ε⁴ / (y² + ε²)^{5/2}        (eigenstrain across the zone)

with on-fault slip-sense shear peaking at **(3/4)·μs/ε**, FWHM ≈ 1.13 ε,
equivalent width 4ε/3 — diverging as ε → 0.  That value is NOT the elastic
stress; the genuine (Coulomb-relevant) elastic stress is

    σ_elastic = σ_total − C:ε*,

implemented in `mhf/anelastic.py` (`eigenstress_at_points`: nearest-triangle
assignment, edge-tapered distance so genuine ε-capped tip concentrations at
the patch edges are preserved) and applied **by default** by `eval_fields`
and `eval_fields_fault_aware` (`subtract_anelastic=True`; opt out for kernel
diagnostics only).  The scalar kernels (`field_hs_mindlin` etc.) stay raw
TOTAL by design — they are the kernel oracles.  Off the fault (beyond ~2ε)
the subtraction is a no-op; the corrected on-fault ΔCFS is the bounded,
ε-stable classical signature (stress drop atop the patch, positive lobes
beyond the edges) — see `python -m mhf.demo_eps_sweep` (fig_mhf_epssweep)
and validate group 8.

**Deprecated:** an earlier reading of the unsubtracted total as the physical
stress of a finite-width fault zone (ε = real zone width) is withdrawn — the
1/ε on-fault band is the eigenstrain being counted as stress.  `γ(y)`
survives as the definition of the eigenstrain profile being removed
(`mhf.anelastic.gamma_profile`).

**Moment accounting** (derived + verified numerically to 4 digits):

- Buried faults (top depth ≫ ε): the mollification conserves seismic moment
  **exactly** — the blob is a unit-mass redistribution of potency; far-field
  displacement is ε-invariant (measured deviation ≤ 0.2% even at ε = 2 km
  with the top at 4 km, and that residual is the blob tail reaching z = 0).
- SURFACE-BREAKING faults lose exactly the potency the blob pushes above the
  free surface:  **M₀(ε) = M₀ · (1 − ε/(4D))**  for a fault of down-dip
  extent D with its top at z = 0 (the mean blob mass above the surface,
  ∫₀^∞ Q(h) dh = ε/4, i.e. the moment of a strip of height ε/4 along the
  trace).  Linear in ε — it is the source-side face of the same O(ε)
  near-trace term Richardson cancels: `richardson_levels=2` restores the
  far-field moment to ~1e-4 (measured).  To force classical moment at a
  single ε instead, scale slip by 1/(1 − ε/(4D)).

## Bug fixes included (vs the historical mh/mollified_kernel state)

1. **λ/μ pairing in the DD stiffness contraction** (2026-06-10, critical).
   `T[i,k] = C_kjpq n_j ∂G_ip/∂y_q`: slip pairs with the normal in C's
   first index pair → λ on the `n_k·trace` term, μ on the two `n·∇G`
   terms.  The historical swap was invisible at ν=0.25 (λ=μ) and broke
   everything else (~66% error at ν=0.30).  Fixed here in
   `exact_bc_triangle.py` (inlined contractions), `analytical_kernels.py`,
   `exact_bc_jax.py`.
2. **Graded rule on (near-)horizontal triangles** (2026-06-10, major).
   The depth split returned a zero-point rule, silently dropping the whole
   correction.  `graded_quad` now falls back to a collapsed tensor rule
   whenever the weight-sum contract fails; same fix (static-shape) in the
   JAX twin.
3. Clean-build hardening: `graded_quad` raises for ε≤0; `romberg_weights`
   rejects non-integer levels; observation points with z>0 raise;
   `make_kernels` is lru-cached (no re-jit per call).

## Deliberately not included

- The Apostol `hybrid`/`quadrature` legacy methods (O(ε²)-BC comparators) —
  still in `mollified_kernel/mindlin_{kernels,triangle}.py`, `mh/mindlin.py`.
- The Comninou–Dundurs angular-dislocation shortcut (`mh/green.py`,
  `green_jax.py`, `adv.py`) — fixed 2026-06-10, kept there as comparators.
- The exact-BC *point* kernel reference (`mollified_kernel/
  exact_bc_pointkernel.py`) — not a dependency of the triangle path.
- The future semi-analytic in-plane closure (sees
  `ANALYTIC_CORRECTION_FEASIBILITY.md` at repo root): validated feasible,
  to be added later as an opt-in correction evaluator inside this package.

## Validation

```bash
JAX_PLATFORMS=cpu python -m mhf.validate
```

Eight groups: ε→0 convergence to cutde at ν ∈ {0.25, 0.30, 0.35} (O(ε²));
machine-precision BC (tensor + graded, buried/horizontal/surface-breaking);
stress ≡ Hooke(sym grad); graded ≡ tensor incl. the horizontal contract;
μ-invariance of displacement; Richardson L2 restoring O(ε²) near the trace;
JAX↔numpy parity at ν≠0.25; anelastic eigenstress subtraction (off-fault
no-op, on-fault raw ~1/ε vs corrected bounded, exact wiring identity through
both evaluators, on-fault marginal magnitude (3/4)μs/ε and sign).  The wider
validation suite (oracle ports,
point-kernel checks, Richardson/graded deep tests) lives at the repo root
(`test_exact_bc_*.py`, `test_green_vs_tdhs.py`, `test_exact_bc_regressions.py`).

## Module map

| file | role |
|---|---|
| `mindlin.py` | scalar API: `disp/strain/stress/field_hs_mindlin` |
| `fields.py` | batch API: `eval_fields[_fault_aware]` (ELASTIC stress by default, `subtract_anelastic=True`), grid builders, `make_rect_fault`, `fault_distance` (JAX-fast, numpy fallback; general `[strike, dip, tensile]` slip, per-triangle if desired) |
| `anelastic.py` | eigenstress `C:ε*` of the smeared slip: `eigenstress_at_points`, `subtract_anelastic`, `gamma_profile` (vendored from msd/anelastic.py; identical copy in moss/mhf — edit both) |
| `figures.py` | 3×3-panel field renderers (house style) |
| `demo_mapview.py` | `python -m mhf.demo_mapview` — horizontal-slice figure (elastic stress; `--total` for the raw comparison) |
| `demo_xsection.py` | `python -m mhf.demo_xsection` — across-strike section figure (elastic stress; `--total` for the raw comparison) |
| `demo_sixpanel.py` | `python -m mhf.demo_sixpanel` — u_x / ΔCFS / von Mises, map view + cross-section, elastic stress by default (`--total` for the raw eigenstress-included rendering; uses `mhf.coulomb_stress`, `mhf.von_mises`; colormaps PiYG / RdYlBu / cool) |
| `demo_eps_sweep.py` | `python -m mhf.demo_eps_sweep` — finiteness figure `fig_mhf_epssweep`: raw on-fault ΔCFS peak ∝ 1/ε vs eigenstress-corrected peak bounded (successor of the removed demo_sixpanel_noeigen.py) |
| `demo_eps_animation.py` | `python -m mhf.demo_eps_animation` — renders the six-panel figure over an ε-ladder with FROZEN color limits and assembles an MP4 via ffmpeg: the ELASTIC field converging as ε shrinks (ε-capped tip concentrations sharpen; `--total` reproduces the deprecated intensifying-zone movie) |
| `exact_bc_triangle.py` | triangle kernels: analytic direct + correction quadrature; symbolic correction kernel built at import (~10 s sympy) |
| `exact_bc_coeffs.py` | closed-form PN coefficients in (a, c, ε, ν) |
| `analytical_kernels.py` | closed-form mollified-Kelvin triangle integrals |
| `exact_bc_graded_quad.py` | obs-dependent sinh2D graded correction rule |
| `eps_extrapolation.py` | Romberg weights / ε-ladder / extrapolators |
| `exact_bc_jax.py` | optional JAX twin (`mhf.jax_kernels()`) |
| `validate.py` | `python -m mhf.validate` |

Notes on the fault-aware evaluator (`eval_fields_fault_aware`): displacement
is finite on the fault, so Richardson extrapolation is applied everywhere;
the TOTAL stress is singular there as ε→0 (the eigenstress), so the
Richardson value is blended (by distance to the fault, smoothstep between
`blend_lo·ε` and `blend_hi·ε`) with the regularized single-ε value, and the
anelastic eigenstress at the nominal ε is then subtracted once, after the
blend (default `subtract_anelastic=True`) — figures show the bounded elastic
stress both on and off the fault.  Unlike the old `mh` demo helper, slip is
fully general here: `[strike, dip, tensile]`, one vector or per-triangle.

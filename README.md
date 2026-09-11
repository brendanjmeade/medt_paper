# medt_paper

Code, cached results, and manuscript source to reproduce every figure of

> B. Meade, *Three dimensional non-singular mollified elastic dislocation
> theory for extended width fault zones and inhomogeneous boundary element
> models* (draft, September 2026).

The paper builds a mollified elastic dislocation theory (MEDT) on Cortez-blob
regularized displacement-discontinuity kernels, integrated in closed form over
flat triangles, and uses it for on-fault stress, ε–h decoupling, and
collocation boundary element models with topography and material contrasts.

## Layout

| path | contents |
|---|---|
| `manuscript/` | `main.tex`, `refs.bib`, and `figures/` (the ten figure files the paper includes) |
| `scripts/` | one script per figure, `fig01_…` to `fig10_…`, plus the ensemble computation behind figure 6 and the matplotlib house style |
| `mollified_kernel/` | closed-form mollified DD kernels for a flat triangle (analytic kernels, batched evaluation, quadrature reference); figures 2–8 |
| `mhf/` | mollified half-space (Mindlin) triangle-dislocation package with exact free-surface boundary conditions; figure 9 |
| `topo_inclusion/` | the moss2 mollified boundary element solver (`mbem/` region-graph stack, meshing, benchmark driver, renderer); figure 10 |
| `cache/` | results of the slow computations, so that every figure renders in seconds to minutes (see below) |

All library code is copied verbatim from the research repositories it was
developed in (`moss`, `mhf`, `moss2`); the figure scripts differ from their
originals only in paths, imports and output names (each carries a provenance
comment at the top).

## Environment

Python 3.13 with `numpy`, `scipy`, `matplotlib`; `jax` for figure 9 (a numpy
fallback exists but is about ten times slower); `numba` and the Shewchuk
`triangle` wrapper for the figure 10 *solve* step only (rendering from the
cache needs neither). See `requirements.txt`. LaTeX with `latexmk`, `natbib`,
`changes`, `hyperref`, `cleveref` for the manuscript.

```bash
pip install -r requirements.txt
make PY=python figs        # all figures into manuscript/figures/
make PY=python paper       # manuscript/main.pdf
```

## Figures

| # | file | script | source | compute |
|---|---|---|---|---|
| 1 | `fig_blob_slip_cartoon.pdf` | `fig01_blob_slip_cartoon.py` | closed-form blob marginal and cumulative slip | < 1 s |
| 2 | `fig_triangle_stress_components.pdf` | `fig02_triangle_stress_components.py` | `mollified_kernel` analytic vs Gauss quadrature | ~13 min (Gauss quadrature on a fine grid) |
| 3 | `fig_eps_sweep_stress_field.pdf` | `fig03_eps_sweep_stress_field.py` | `mollified_kernel`, four ε | minutes |
| 4 | `fig_eps_h_decoupling.pdf` | `fig04_eps_h_decoupling.py` | subdivision benchmark; `cache/fig04_eps_h_decoupling.npz` | seconds (cached), `--force` recomputes |
| 5 | `fig_onfault_stress_eps_sweep.pdf` | `fig05_onfault_stress_eps_sweep.py` | analytic stress minus exact finite-triangle eigenstress | seconds |
| 6 | `fig_shape_ensemble.pdf` | `fig06_shape_ensemble.py` (render) and `shape_ensemble_compute.py` (300-triangle sweep) | `cache/fig06_shape_ensemble.npz` | seconds (cached); the sweep is 240k kernel evaluations, run it with `--force` |
| 7 | `fig_near_element_stress.pdf` | `fig07_near_element_stress.py` | analytic CEDT/MEDT vs quadrature near a vertex | seconds |
| 8 | `fig_elliptical_crack_mesh_sensitivity.pdf` | `fig08_elliptical_crack_mesh_sensitivity.py` | 192/775-element elliptical crack fields; `cache/fig08_elliptical_crack_mesh_sensitivity.npz` (21 MB) | seconds (cached), `--force` recomputes |
| 9 | `fig_halfspace_like_displacements.pdf` | `fig09_halfspace_like_displacements.py` | `mhf` half-space kernels, three ε, high-resolution grids | ~3 min with JAX on a laptop; `--fast` for a one-minute preview |
| 10 | `fig_topography_inclusion.pdf` | `fig10_topography_inclusion.py` | `topo_inclusion/` BEM fields; `cache/topo_inclusion_fields_mu10.npz` | seconds (cached); `--solve` reruns the four dense 31k-unknown solves (~20 min, ~16 GB) |

### Notes on provenance

- **Figure 9** in the draft was assembled in Keynote from three frames of the
  mhf ε-ladder animation (`mhf/demo_eps_animation.py`, high-resolution run
  `--eps-max 0.001 --eps-min 2 --n-frames 200 --n 363 --ny 483 --nz 243`),
  cropping the `u_x` column of the six-panel figure. `fig09_…py` produces the
  same panels directly as one vector PDF, with the color limits frozen across
  ε as in the animation. Two things to be aware of: the physics is the exact
  free-surface (Mindlin) half-space solution of a 10 km × 10 km surface-breaking
  fault with 1 m of strike slip, not a boundary element box, so the manuscript
  text describing this figure as a "100 km cube with a fixed base" should be
  revised; and the middle frame of the ladder that was used corresponds to
  ε ≈ 0.3 km, while the caption says 400 m. The script uses the caption's
  values (1 m, 400 m, 2 km) by default; pass `--eps-km` to change them.
- **Figure 10** was computed with the moss2 solver by
  `make_topo_inclusion.py --mu-inc 3` (inclusion shear modulus 3 GPa, host
  30 GPa, ε = 3 km, 10 m of slip) and rendered with
  `render_topo_inclusion_contour.py <npz> "" --smooth`. The wrapper script
  runs exactly those commands. `topo_inclusion/assess_fig06_inclusion.py`
  contains an absolute path to a moss checkout that is only used by an
  optional oracle check, never by the figure.
- The half-space package's own validation (`python -m mhf.validate`) needs
  the optional `cutde` package as a classical reference; no figure does.

## Manuscript

`manuscript/main.tex` is the current draft with the review edits still marked
by the `changes` package (`\usepackage[final]{changes}` accepts them all).
`make paper` runs `latexmk` in `manuscript/`.

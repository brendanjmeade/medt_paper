# Reproduce the figures and the PDF of
#   "Three dimensional non-singular mollified elastic dislocation theory for
#    extended width fault zones and inhomogeneous boundary element models"
#
# Usage (from this directory):
#   make figs                 # all ten figures (cached heavy results are used)
#   make fig06 fig09          # individual figures
#   make paper                # compile manuscript/main.pdf with latexmk
#   make figs-recompute       # also redo the slow computations (see README)
#
# Set PY to the interpreter that has numpy/scipy/matplotlib (+ jax, numba,
# triangle for figures 9-10), e.g.  make PY=~/miniforge3/envs/main/bin/python figs

PY ?= python
export MPLBACKEND = Agg
export JAX_PLATFORMS ?= cpu

FIGDIR := manuscript/figures

FIGS := $(FIGDIR)/fig_blob_slip_cartoon.pdf \
        $(FIGDIR)/fig_triangle_stress_components.pdf \
        $(FIGDIR)/fig_eps_sweep_stress_field.pdf \
        $(FIGDIR)/fig_eps_h_decoupling.pdf \
        $(FIGDIR)/fig_onfault_stress_eps_sweep.pdf \
        $(FIGDIR)/fig_shape_ensemble.pdf \
        $(FIGDIR)/fig_near_element_stress.pdf \
        $(FIGDIR)/fig_elliptical_crack_mesh_sensitivity.pdf \
        $(FIGDIR)/fig_halfspace_like_displacements.pdf \
        $(FIGDIR)/fig_topography_inclusion.pdf

.PHONY: all figs paper clean fig01 fig02 fig03 fig04 fig05 fig06 fig07 fig08 fig09 fig10 figs-recompute

all: figs paper

figs: $(FIGS)

fig01: $(FIGDIR)/fig_blob_slip_cartoon.pdf
fig02: $(FIGDIR)/fig_triangle_stress_components.pdf
fig03: $(FIGDIR)/fig_eps_sweep_stress_field.pdf
fig04: $(FIGDIR)/fig_eps_h_decoupling.pdf
fig05: $(FIGDIR)/fig_onfault_stress_eps_sweep.pdf
fig06: $(FIGDIR)/fig_shape_ensemble.pdf
fig07: $(FIGDIR)/fig_near_element_stress.pdf
fig08: $(FIGDIR)/fig_elliptical_crack_mesh_sensitivity.pdf
fig09: $(FIGDIR)/fig_halfspace_like_displacements.pdf
fig10: $(FIGDIR)/fig_topography_inclusion.pdf

$(FIGDIR)/fig_blob_slip_cartoon.pdf: scripts/fig01_blob_slip_cartoon.py
	$(PY) $<

$(FIGDIR)/fig_triangle_stress_components.pdf: scripts/fig02_triangle_stress_components.py
	$(PY) $<

$(FIGDIR)/fig_eps_sweep_stress_field.pdf: scripts/fig03_eps_sweep_stress_field.py
	$(PY) $<

$(FIGDIR)/fig_eps_h_decoupling.pdf: scripts/fig04_eps_h_decoupling.py cache/fig04_eps_h_decoupling.npz
	$(PY) $<

$(FIGDIR)/fig_onfault_stress_eps_sweep.pdf: scripts/fig05_onfault_stress_eps_sweep.py
	$(PY) $<

# the ensemble sweep (scripts/shape_ensemble_compute.py) is read from the cache;
# delete cache/fig06_shape_ensemble.npz or run it with --force to recompute
$(FIGDIR)/fig_shape_ensemble.pdf: scripts/fig06_shape_ensemble.py scripts/shape_ensemble_compute.py cache/fig06_shape_ensemble.npz
	$(PY) $<

$(FIGDIR)/fig_near_element_stress.pdf: scripts/fig07_near_element_stress.py
	$(PY) $<

$(FIGDIR)/fig_elliptical_crack_mesh_sensitivity.pdf: scripts/fig08_elliptical_crack_mesh_sensitivity.py cache/fig08_elliptical_crack_mesh_sensitivity.npz
	$(PY) $<

# mhf half-space kernels; ~3 min with JAX on a laptop, --fast for a preview
$(FIGDIR)/fig_halfspace_like_displacements.pdf: scripts/fig09_halfspace_like_displacements.py
	$(PY) $<

# renders from the cached BEM fields; add --solve to recompute (~20 min, ~16 GB)
$(FIGDIR)/fig_topography_inclusion.pdf: scripts/fig10_topography_inclusion.py cache/topo_inclusion_fields_mu10.npz
	$(PY) $<

figs-recompute:
	$(PY) scripts/fig04_eps_h_decoupling.py --force
	$(PY) scripts/shape_ensemble_compute.py --force
	$(PY) scripts/fig06_shape_ensemble.py
	$(PY) scripts/fig08_elliptical_crack_mesh_sensitivity.py --force
	$(PY) scripts/fig10_topography_inclusion.py --solve

paper: $(FIGS)
	cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	cd manuscript && latexmk -C main.tex; rm -f main.loc main.soc

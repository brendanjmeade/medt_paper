"""
mhf — mollified half-space, fixed: the clean exact-BC Mindlin build
===================================================================

Self-contained semi-numerical production path for half-space triangular
dislocations with Cortez mollification:

  * analytic mollified-Kelvin direct part (closed form, on-fault accurate);
  * exact-BC Papkovich-Neuber complementary correction at image depth
    a = sqrt(src_z^2 + eps^2) with closed-form coefficients — the
    free-surface traction cancels POINTWISE IN THE SOURCE, so the
    triangle BC residual is pure quadrature error with no eps floor
    (spectrally convergent; machine precision at modest rule orders);
  * correction integrated by tensor Gauss or the obs-dependent sinh2D
    graded rule (with tensor fallback for (near-)horizontal triangles);
  * optional Richardson/Romberg eps-extrapolation for near-trace accuracy
    on surface-breaking faults;
  * anelastic eigenstress subtraction (mhf.anelastic): the batch evaluators
    return the ELASTIC stress by default (sigma_total - C:eps_star), bounded
    on the fault; the scalar kernels return the raw TOTAL stress;
  * optional JAX twin (vmap/jit influence matrices) via mhf.jax_kernels().

Built 2026-06-11 from the post-review state of mh/ + mollified_kernel/,
with all 2026-06-10 review fixes included (lambda/mu stiffness-contraction
pairing; graded-rule horizontal-triangle fallback; see README.md).  The
legacy comparator paths (Apostol hybrid/quadrature methods, the C&D
angular-dislocation shortcut) are deliberately NOT included; they remain
in mh/ and mollified_kernel/.

Conventions (cutde-compatible): x East, y North, z Up; free surface z=0,
material z<=0; slip = [strike, dip, tensile]; triangle normal by
right-hand rule on vertex order; V_strike = z_hat x V_normal.

Validate with:  JAX_PLATFORMS=cpu python -m mhf.validate
"""

from .mindlin import (
    disp_hs_mindlin,
    strain_hs_mindlin,
    stress_hs_mindlin,
    field_hs_mindlin,
)
from .eps_extrapolation import romberg_weights, eps_ladder, combine_samples
from .anelastic import (
    eigenstress_at_points,
    subtract_anelastic,
    gamma_profile,
)
from .fields import (
    eval_fields,
    eval_fields_fault_aware,
    build_grid,
    build_grid_section,
    make_rect_fault,
    fault_distance,
    coulomb_stress,
    von_mises,
    moment_factor,
    q_above_surface,
)

__all__ = [
    "disp_hs_mindlin", "strain_hs_mindlin", "stress_hs_mindlin",
    "field_hs_mindlin", "romberg_weights", "eps_ladder", "combine_samples",
    "eigenstress_at_points", "subtract_anelastic", "gamma_profile",
    "eval_fields", "eval_fields_fault_aware", "build_grid",
    "build_grid_section", "make_rect_fault", "fault_distance",
    "coulomb_stress", "von_mises", "moment_factor", "q_above_surface",
    "jax_kernels",
]


def jax_kernels(n_quad=8, graded=False, n_c=None, n_t=None):
    """Jitted/vmapped JAX kernels (cached): dict with disp_U, stress_H,
    disp_matrix, stress_matrix.  Requires jax; raises ImportError otherwise.
    On Apple Silicon run with JAX_PLATFORMS=cpu (jax-metal is unreliable)."""
    from .exact_bc_jax import make_kernels
    return make_kernels(n_quad=n_quad, graded=graded, n_c=n_c, n_t=n_t)

"""
mhf.mindlin — exact-BC mollified Mindlin half-space triangle dislocation
========================================================================

The clean semi-numerical production path, and nothing else:

  * analytic mollified-Kelvin DIRECT part (closed form, accurate on-fault);
  * exact-BC complementary Papkovich-Neuber CORRECTION (machine-precision
    free-surface traction pointwise in the source), integrated over the
    triangle by Gauss quadrature — tensor rule or obs-dependent sinh2D
    graded rule;
  * optional Richardson/Romberg extrapolation over an eps-ladder for
    near-trace accuracy on SURFACE-BREAKING faults.

Sign / slip conventions match cutde exactly:
  - Coordinate system: East (x), North (y), Up (z)
  - Free surface at z = 0, material at z <= 0
  - Slip ordering: [strike_slip, dip_slip, tensile_slip]
  - Triangle normal: right-hand rule on vertex ordering (v1-v0) x (v2-v0)
  - Strike basis: V_strike = z_hat x V_normal; V_dip = V_normal x V_strike

Units are scale-free; the repo convention is km / GPa / km-slip.

Usage::

    from mhf import disp_hs_mindlin, field_hs_mindlin
    u = disp_hs_mindlin(obs_pts, tri, slip, nu, eps)
    f = field_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=30.0,
                         graded=True, n_c=20, n_t=16, richardson_levels=2)
"""

import numpy as np

from .exact_bc_triangle import (
    disp_dd_kernel as _disp_kernel,
    grad_dd_kernel as _grad_kernel,
)
from .eps_extrapolation import extrapolate_scalarfn, extrapolate_dictfn


def _normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _strike_dip_basis(tri):
    """(V_normal, V_strike, V_dip) for a triangle (3,3), cutde convention."""
    Vnorm = _normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    eY = np.array([0.0, 1.0, 0.0])
    eZ = np.array([0.0, 0.0, 1.0])
    Vstrike = np.cross(eZ, Vnorm)
    if np.linalg.norm(Vstrike) == 0:
        # Horizontal triangle: V_strike = e_Y * sign(Vnorm_z)
        Vstrike = eY * np.sign(Vnorm[2]) if Vnorm[2] != 0 else eY
    Vstrike = _normalize(Vstrike)
    Vdip = np.cross(Vnorm, Vstrike)
    return Vnorm, Vstrike, Vdip


def _slip_to_cartesian(slip_sdt, Vstrike, Vdip, Vnorm):
    """(strike, dip, tensile) -> Cartesian slip vector."""
    s, d, t = slip_sdt
    return s * Vstrike + d * Vdip + t * Vnorm


def _prepare(obs_pts, tri, slip):
    obs_pts = np.asarray(obs_pts, float)
    tri = np.asarray(tri, float)
    slip = np.asarray(slip, float)
    single = obs_pts.ndim == 1
    if single:
        obs_pts = obs_pts[np.newaxis, :]
    if np.any(obs_pts[:, 2] > 0.0):
        raise ValueError(
            "observation point(s) above the free surface (z > 0); the "
            "half-space kernel is defined for z <= 0")
    Vnorm, Vstrike, Vdip = _strike_dip_basis(tri)
    slip_cart = _slip_to_cartesian(slip, Vstrike, Vdip, Vnorm)
    return obs_pts, tri, slip_cart, Vnorm, single


def disp_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=1.0, n_quad=8,
                    richardson_levels=1, richardson_ratio=2.0,
                    graded=False, n_c=None, n_t=None):
    """
    Half-space displacement from a triangular dislocation (exact-BC
    mollified Mindlin kernel).

    Parameters
    ----------
    obs_pts : (N, 3) or (3,) observation points (z <= 0)
    tri     : (3, 3) triangle vertices
    slip    : (3,) [strike_slip, dip_slip, tensile_slip]
    nu      : Poisson's ratio
    eps     : mollification parameter (>= 0; typically ~fault_width/10)
    mu      : shear modulus (displacement is mu-invariant; needed only to
              form lambda internally)
    n_quad  : tensor correction-quadrature order per direction
              (n_quad^2 points; ~8 for values, ~16 for machine-precision BC)
    richardson_levels : 1 = off.  >1 extrapolates over the eps-ladder
              eps/ratio**i, cancelling the intrinsic O(eps) near-trace error
              of SURFACE-BREAKING faults (levels=2 -> O(eps^2)).  Preserves
              the exact free-surface BC.  Pair with graded=True so each
              ladder rung stays quadrature-converged.  Use levels=1 on/very
              near the source triangle (the eps->0 limit is singular there).
    richardson_ratio  : eps-ladder geometric ratio (default 2.0)
    graded  : True -> obs-dependent sinh2D graded correction rule (resolves
              the near-singular correction of surface-breaking faults at
              small eps); requires eps > 0
    n_c, n_t: graded depth / in-plane node counts (default n_quad)

    Returns
    -------
    u : (N, 3) or (3,) displacement
    """
    obs_pts, tri, slip_cart, Vnorm, single = _prepare(obs_pts, tri, slip)
    N = obs_pts.shape[0]

    def _disp_single(e):
        u_out = np.zeros((N, 3))
        for i in range(N):
            U = _disp_kernel(obs_pts[i], tri[0], tri[1], tri[2], Vnorm,
                             mu, nu, e, n_quad, graded=graded, n_c=n_c, n_t=n_t)
            u_out[i] = U @ slip_cart
        return u_out

    if richardson_levels > 1:
        u_out = extrapolate_scalarfn(_disp_single, eps, levels=richardson_levels,
                                     ratio=richardson_ratio)
    else:
        u_out = _disp_single(eps)
    return u_out.squeeze() if single else u_out


def field_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=1.0, n_quad=8,
                     richardson_levels=1, richardson_ratio=2.0,
                     graded=False, n_c=None, n_t=None):
    """
    Displacement, displacement gradient, strain, and stress at obs_pts.

    Same arguments as disp_hs_mindlin.  Each output is extrapolated
    independently when richardson_levels > 1 (exact, by linearity); the
    free-surface BC is preserved (every eps satisfies T_3k(z=0)=0).

    NOTE: 'stress' is the raw TOTAL kernel stress, including the anelastic
    eigenstress C:eps_star inside the ~eps fault zone (a kernel oracle, by
    design).  For presented / Coulomb-relevant elastic stress use
    mhf.eval_fields / mhf.eval_fields_fault_aware (subtract_anelastic=True
    default) or subtract mhf.anelastic.eigenstress_at_points yourself.

    Returns dict:
        'disp'   : (N, 3)    or (3,)
        'grad_u' : (N, 3, 3) or (3, 3)   du_i/dobs_p
        'strain' : (N, 3, 3) or (3, 3)   sym(grad_u)
        'stress' : (N, 3, 3) or (3, 3)   lam tr(e) I + 2 mu e
    """
    obs_pts, tri, slip_cart, Vnorm, single = _prepare(obs_pts, tri, slip)
    N = obs_pts.shape[0]
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

    def _field_single(e):
        u_out = np.zeros((N, 3))
        grad_out = np.zeros((N, 3, 3))
        eps_out = np.zeros((N, 3, 3))
        sig_out = np.zeros((N, 3, 3))
        for i in range(N):
            U = _disp_kernel(obs_pts[i], tri[0], tri[1], tri[2], Vnorm,
                             mu, nu, e, n_quad, graded=graded, n_c=n_c, n_t=n_t)
            Gd = _grad_kernel(obs_pts[i], tri[0], tri[1], tri[2], Vnorm,
                              mu, nu, e, n_quad, graded=graded, n_c=n_c, n_t=n_t)
            u_out[i] = U @ slip_cart
            grad = np.einsum("ipk,k->ip", Gd, slip_cart)
            ee = 0.5 * (grad + grad.T)
            grad_out[i] = grad
            eps_out[i] = ee
            sig_out[i] = lam * np.trace(ee) * np.eye(3) + 2.0 * mu * ee
        return {"disp": u_out, "grad_u": grad_out, "strain": eps_out,
                "stress": sig_out}

    if richardson_levels > 1:
        out = extrapolate_dictfn(_field_single, eps, levels=richardson_levels,
                                 ratio=richardson_ratio)
    else:
        out = _field_single(eps)

    if single:
        return {k: v[0] for k, v in out.items()}
    return out


def strain_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=1.0, n_quad=8,
                      richardson_levels=1, richardson_ratio=2.0,
                      graded=False, n_c=None, n_t=None):
    """Symmetric strain tensor(s); same API as disp_hs_mindlin."""
    return field_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=mu, n_quad=n_quad,
                            richardson_levels=richardson_levels,
                            richardson_ratio=richardson_ratio,
                            graded=graded, n_c=n_c, n_t=n_t)["strain"]


def stress_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=1.0, n_quad=8,
                      richardson_levels=1, richardson_ratio=2.0,
                      graded=False, n_c=None, n_t=None):
    """Symmetric Hooke stress tensor(s); same API as disp_hs_mindlin.
    Raw TOTAL stress (see field_hs_mindlin note on the anelastic term)."""
    return field_hs_mindlin(obs_pts, tri, slip, nu, eps, mu=mu, n_quad=n_quad,
                            richardson_levels=richardson_levels,
                            richardson_ratio=richardson_ratio,
                            graded=graded, n_c=n_c, n_t=n_t)["stress"]

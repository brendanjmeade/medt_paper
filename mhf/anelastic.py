"""Anelastic (eigenstrain) term for mollified-fault stresses.

Vendored/adapted from msd/anelastic.py (msd repo, June 2026); keep in sync
manually — moss-org repos share no package.  This file is IDENTICAL in
mhf/ and moss/mhf/ — edit both.

A fault slip is an *anelastic* (inelastic / eigen-) strain.  When stress is
read off a mollified slip source -- either from the displacement-discontinuity
stress kernel or by differentiating the displacement field and applying
Hooke's law -- the value *inside* the ~eps-wide smeared fault zone is the
TOTAL stress

    sigma_total = C : eps_total = C : (eps_elastic + eps_star),

where ``eps_star`` is the smeared slip itself (the anelastic term).  On the
fault the kernel is dominated by this term, with an on-fault shear that peaks
at ``(3/4) mu s / eps`` and diverges as eps -> 0.  That value is NOT the
elastic stress.  Subtracting the anelastic eigenstress recovers the genuine
elastic (Coulomb-relevant) stress, which is smooth and BOUNDED:

    sigma_elastic = sigma_total - C : eps_star.

SIGN CONVENTION (mhf): the mhf evaluators (``eval_fields``,
``eval_fields_fault_aware``, and the scalar ``field_hs_mindlin``) return
``+sigma_total`` of the slip source directly, so the correction is a plain
subtraction as written above.  (Contrast msd's BEM readout, where the fault
term enters the representation formula as ``-Sdd @ slip`` and removing the
divergent part therefore *adds* ``eigenstress_at_points`` back.)

For pure slip (slip direction ``d`` perpendicular to the fault normal ``n``,
so ``eps_star`` is trace-free) the eigenstress is

    sigma*_ij(x) = mu * s * (d_i n_j + d_j n_i) * rho_eps(perp),
    rho_eps(perp) = (3/4) eps^4 / (perp^2 + eps^2)^(5/2),

i.e. the Cortez blob's fault-normal marginal ``rho_eps`` (which already
integrates the in-plane directions) evaluated at the perpendicular distance to
the fault.  Each observation point is assigned to its NEAREST fault triangle so
the marginal of a multi-triangle planar fault is counted once; the distance is
the perpendicular distance inside the patch footprint and tapers to the edge
distance beyond it, so genuine elastic tip concentrations at the fault edges
are preserved.

Off the zone ``rho_eps ~ (eps/perp)^5`` is negligible, so the subtraction is a
no-op there: only the near-fault band is corrected.

The lambda*trace term is included for completeness so opening (tensile) slip is
handled too; it vanishes for pure shear.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
def _seg_distance(P, A, B):
    """Distance from each point in P (N,3) to segment AB."""
    AB = B - A
    denom = float(AB @ AB) + 1e-300
    t = np.clip((P - A) @ AB / denom, 0.0, 1.0)
    foot = A + t[:, None] * AB
    return np.linalg.norm(P - foot, axis=1)


def point_triangle_distance(P, A, B, C):
    """Unsigned distance from each point in P (N,3) to triangle ABC.

    Returns the perpendicular distance where the in-plane projection lies
    inside the triangle, and the nearest-edge distance otherwise (so the
    value tapers smoothly past the triangle's edges)."""
    P = np.asarray(P, float)
    n = np.cross(B - A, C - A)
    n = n / (np.linalg.norm(n) + 1e-300)
    perp = (P - A) @ n
    proj = P - perp[:, None] * n
    v0, v1, v2 = B - A, C - A, proj - A
    d00 = float(v0 @ v0); d01 = float(v0 @ v1); d11 = float(v1 @ v1)
    d20 = v2 @ v0; d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01 + 1e-300
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    d_edge = np.minimum.reduce([_seg_distance(P, A, B),
                                _seg_distance(P, B, C),
                                _seg_distance(P, C, A)])
    return np.where(inside, np.abs(perp), d_edge)


# ---------------------------------------------------------------------------
def eigenstress_at_points(obs, fault, slip_cart, mu, nu, eps):
    """Anelastic eigenstress sigma*(x) = C:eps_star(x) at obs points (GPa-units).

    Parameters
    ----------
    obs : (N,3) observation points.
    fault : a mesh with ``.vertices`` (Nv,3), ``.triangles`` (Nt,3) and
        ``.normals_and_areas()``, OR a (Nt,3,3) array of triangle vertices
        (the mhf convention).
    slip_cart : Cartesian slip vector.  Either (3,) applied to every fault
        triangle, or (Nt,3) per triangle.  Same length units as the kernel
        slip (e.g. km).
    mu, nu : shear modulus and Poisson's ratio (mu sets the stress units).
    eps : mollification parameter (same length units as the mesh).

    Returns
    -------
    sigma_star : (N,3,3) eigenstress tensor field.  Subtract this from the
        total stress to obtain the elastic stress.
    """
    obs = np.asarray(obs, float)
    N = obs.shape[0]
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    eye = np.eye(3)

    # --- per-triangle vertices and normals ---
    if hasattr(fault, "vertices"):
        verts = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
        normals, _ = fault.normals_and_areas()
        normals = np.asarray(normals, float)
    else:
        verts = np.asarray(fault, float).reshape(-1, 3, 3)
        e1 = verts[:, 1] - verts[:, 0]
        e2 = verts[:, 2] - verts[:, 0]
        cr = np.cross(e1, e2)
        normals = cr / (np.linalg.norm(cr, axis=1, keepdims=True) + 1e-300)
    M = verts.shape[0]

    slip_cart = np.asarray(slip_cart, float)
    if slip_cart.ndim == 1:
        slip_cart = np.broadcast_to(slip_cart, (M, 3))

    # --- per-triangle eigenstress dyad (mu/lam baked in) and obs distances ---
    sig_e = np.zeros((M, 3, 3))     # C:eps_star per unit (s * rho)
    s_arr = np.zeros(M)
    dists = np.empty((N, M))
    for m in range(M):
        A, B, C = verts[m]
        dists[:, m] = point_triangle_distance(obs, A, B, C)
        s = float(np.linalg.norm(slip_cart[m]))
        s_arr[m] = s
        if s <= 0.0:
            continue
        d_hat = slip_cart[m] / s
        n = normals[m] / (np.linalg.norm(normals[m]) + 1e-300)
        e_unit = 0.5 * (np.outer(d_hat, n) + np.outer(n, d_hat))
        sig_e[m] = lam * np.trace(e_unit) * eye + 2.0 * mu * e_unit

    # --- nearest fault triangle per obs point (single-marginal, edge-tapered) ---
    k = np.argmin(dists, axis=1)
    perp = dists[np.arange(N), k]
    rho = 0.75 * eps**4 / (perp**2 + eps**2) ** 2.5
    return (s_arr[k] * rho)[:, None, None] * sig_e[k]


def subtract_anelastic(sigma_total, obs, fault, slip_cart, mu, nu, eps):
    """Return the elastic stress sigma_total - C:eps_star at ``obs``."""
    return np.asarray(sigma_total, float) - eigenstress_at_points(
        obs, fault, slip_cart, mu, nu, eps)


def gamma_profile(perp, slip_mag, eps):
    """Smeared-slip (anelastic) shear-strain profile across the fault zone:
    gamma(y) = s * (3/4) eps^4 / (y^2 + eps^2)^(5/2); integrates to the slip,
    peaks at (3/4) s / eps, FWHM ~ 1.13 eps, equivalent width 4 eps/3."""
    perp = np.asarray(perp, float)
    return slip_mag * 0.75 * eps**4 / (perp**2 + eps**2) ** 2.5

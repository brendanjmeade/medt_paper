"""Compactly supported topographic warps for free-surface TriMeshes.

Topography enters the BEM purely through the mesh: the free surface is
an explicitly meshed FREE_TRACTION patch, and the mollified Kelvin
kernels are full-space — nothing in the solver stack assumes z = 0. A
vertical warp ``z += h(x, y)`` preserves triangle connectivity and can
never flip a winding (the normal's z-sign depends only on the plan-view
coordinates).

Watertightness is the one real constraint: a warped patch must keep
sharing its boundary curves (interface rims, box edges, fault traces)
with the unwarped patches around it. The bump here therefore has EXACT
compact support (bitwise zero outside its cutoff radius, via masking,
not exp underflow), and ``assert_zero_clearance`` provides a loud
pre-solve guard; the RegionModel solid-angle closure validation is the
backstop.
"""

from __future__ import annotations

import numpy as np


def gaussian_bump(center_xy, height, sigma, taper=(2.5, 3.0)):
    """Gaussian bump with C2 compact support.

        h(x, y) = height * exp(-r^2 / (2 sigma^2)) * S(r)

    where S is the quintic smoothstep: 1 for r <= taper[0]*sigma, exactly
    0 for r >= taper[1]*sigma (so h, h', h'' are continuous everywhere
    and h is bitwise 0.0 outside the support).

    Returns a vectorized callable ``h(x, y) -> array`` with attributes
    ``support_radius``, ``center_xy``, ``height``, ``sigma``.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sigma = float(sigma)
    r1 = float(taper[0]) * sigma
    r2 = float(taper[1]) * sigma
    if not 0.0 < r1 < r2:
        raise ValueError("need 0 < taper[0] < taper[1]")

    def h(x, y):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        r = np.hypot(np.atleast_1d(x) - cx,
                     np.atleast_1d(np.asarray(y, dtype=float)) - cy)
        out = np.zeros_like(r)
        m = r < r2                      # exact 0 outside support, by mask
        t = np.clip((r[m] - r1) / (r2 - r1), 0.0, 1.0)
        s = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
        out[m] = height * np.exp(-r[m] ** 2 / (2.0 * sigma ** 2)) * s
        return out[0] if scalar else out

    h.support_radius = r2
    h.center_xy = (cx, cy)
    h.height = float(height)
    h.sigma = sigma
    return h


def apply_topography(mesh, h, taper_depth=None):
    """New TriMesh-compatible mesh with ``z += f(z) * h(x, y)``.

    Triangles and winding are unchanged; normals/areas are recomputed
    on demand by the mesh class from the warped vertices.

    ``taper_depth`` (km): if given, the warp is scaled by the linear
    taper ``f(z) = clip(1 + z / taper_depth, 0, 1)`` — full at z = 0,
    zero at z <= -taper_depth. Use this to warp INTERIOR fault meshes
    consistently with the free surface: topography may cross a fault
    trace provided the fault and the surface patch receive the SAME
    warp along their shared curve (watertightness requires equal warps
    on shared curves, not zero warps). Caveat: a prescribed constant
    slip vector is no longer exactly tangent to the warped fault plane
    (off by up to the local surface slope); project per-element slip
    onto the warped surface if that matters for the application.
    """
    v = mesh.vertices.copy()
    f = 1.0 if taper_depth is None else \
        np.clip(1.0 + v[:, 2] / float(taper_depth), 0.0, 1.0)
    v[:, 2] = v[:, 2] + f * h(v[:, 0], v[:, 1])
    return type(mesh)(vertices=v, triangles=mesh.triangles.copy())


def assert_zero_clearance(h, points_xy, tol=0.0, label=""):
    """Raise unless max|h| <= tol over the given (N, 2) protected points
    (e.g. interface rims, box edges, fault-trace nodes). Returns the max.

    ``tol=0.0`` is the right default for compactly supported bumps: the
    support must clear protected curves entirely, not just approximately.
    """
    p = np.atleast_2d(np.asarray(points_xy, dtype=float))
    worst = float(np.abs(h(p[:, 0], p[:, 1])).max()) if p.size else 0.0
    if worst > tol:
        where = f" at {label}" if label else ""
        raise ValueError(f"topography violates clearance{where}: "
                         f"max|h| = {worst:.3e} > tol = {tol:.3e}")
    return worst

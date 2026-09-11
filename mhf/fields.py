"""Batch field evaluation on observation grids — the one-stop entry point for
working with the mollified half-space pseudo-Green's functions.

Builds map-view / cross-section observation grids, evaluates displacement +
stress from a set of source triangles (summed), with the Richardson and graded
knobs of mhf.mindlin.  Uses the fast JAX influence-matrix kernels when jax is
available (cached jit; CPU ~6-10x cutde per pair) and falls back to the numpy
scalar path otherwise (identical values, slower).

Conventions: East x, North y, Up z; free surface z=0; half-space z<=0;
lengths km, shear modulus GPa; slip [strike, dip, tensile] in km
(1 m = 1e-3 km).  Slip may be one vector for all triangles or per-triangle.
"""
import os
import sys

if sys.platform == "darwin":                 # local jax-metal is unreliable;
    os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CUDA boxes are untouched

import numpy as np

from .mindlin import _strike_dip_basis, field_hs_mindlin
from .eps_extrapolation import combine_samples, eps_ladder
from .anelastic import (eigenstress_at_points,
                        point_triangle_distance as _point_triangle_distance)

# physical defaults (repo conventions)
MU = 30.0            # GPa
NU = 0.25
EPS = 1.0            # km (mollification ~ fault_width/10)
SLIP = (1.0e-3, 0.0, 0.0)   # 1 m strike-slip, in km
N_QUAD = 8

KM_TO_MM = 1.0e6     # displacement km -> mm
GPA_TO_MPA = 1.0e3   # stress GPa -> MPa


# ---------------------------------------------------------------------------
# Observation grids
# ---------------------------------------------------------------------------
def build_grid(half_extent=15.0, n=121, z0=-2.0):
    """Regular x-y map-view grid at depth z0.  Returns (X, Y, obs[N,3]),
    X/Y of shape (n, n)."""
    g = np.linspace(-half_extent, half_extent, n)
    X, Y = np.meshgrid(g, g)
    obs = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z0)])
    return X, Y, obs


def build_grid_section(x0=0.0, y_half=20.0, z_bot=-20.0, ny=161, nz=81):
    """Vertical y-z cross-section grid at x=x0 (z from z_bot up to 0).
    Returns (Y, Z, obs[N,3]) with grid shape (nz, ny)."""
    yy = np.linspace(-y_half, y_half, ny)
    zz = np.linspace(z_bot, 0.0, nz)
    Y, Z = np.meshgrid(yy, zz)
    obs = np.column_stack([np.full(Y.size, x0), Y.ravel(), Z.ravel()])
    return Y, Z, obs


# ---------------------------------------------------------------------------
# Field evaluation
# ---------------------------------------------------------------------------
def _slip_cart_per_tri(tris, slip):
    """Cartesian slip vector per triangle from [strike, dip, tensile] slip.
    `slip` is (3,) applied to every triangle, or (M,3) per triangle."""
    slip = np.asarray(slip, float)
    if slip.ndim == 1:
        slip = np.broadcast_to(slip, (len(tris), 3))
    out = np.empty((len(tris), 3))
    for m, t in enumerate(tris):
        Vn, Vs, Vd = _strike_dip_basis(t)
        out[m] = slip[m, 0] * Vs + slip[m, 1] * Vd + slip[m, 2] * Vn
    return out


def _have_jax():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def _eval_fields_total(obs, tris, slip=SLIP, mu=MU, nu=NU, eps=EPS,
                       n_quad=N_QUAD,
                       richardson_levels=1, richardson_ratio=2.0,
                       graded=False, n_c=None, n_t=None, block=8192):
    """Displacement (N,3) and TOTAL stress (N,3,3) at obs from slip on
    triangles `tris` (M,3,3), summed over triangles.  INTERNAL: the stress is
    the raw kernel value C:(eps_elastic + eps_star) — inside the ~eps fault
    zone it includes the anelastic eigenstress (peak (3/4) mu s / eps,
    diverging as eps -> 0).  Use the public `eval_fields` (which subtracts
    C:eps_star by default) for presented / Coulomb-relevant stress.

    slip : (3,) [strike, dip, tensile] for all triangles, or (M,3) per
           triangle, in km (1 m slip = 1e-3).
    block : obs are evaluated in chunks of this size.  The vmapped JAX
           kernels materialize intermediates ~ (block x M x n_quad^2), so
           this bounds peak memory regardless of grid size (block=8192 keeps
           a worker around ~2 GB; an unchunked 363x363 grid peaked at 33 GB).
    graded : False (tensor rule), True (graded rule for all obs), or
           "auto": obs within 2*diam/n_quad of the fault — where the fixed
           tensor rule cannot resolve the correction's near-singularity and
           produces O(10 MPa) artifacts near surface-breaking traces — use
           the graded rule (n_c=24, n_t=20 unless given); the rest use the
           tensor rule.  Near-fault accuracy at roughly tensor cost.
    richardson_levels > 1 Richardson/Romberg-extrapolates u and sig over the
    eps-ladder (cancels the intrinsic O(eps) near-trace error of
    surface-breaking faults; preserves the exact free-surface BC).  Pair with
    graded=True so each ladder rung stays quadrature-converged.  NOTE: the
    eps->0 limit is singular ON the fault — for grids that touch the fault
    use eval_fields_fault_aware instead.
    """
    obs = np.asarray(obs, float)
    tris = np.asarray(tris, float).reshape(-1, 3, 3)

    if isinstance(graded, str) and graded == "auto":
        diam = max(np.max(np.linalg.norm(t[[1, 2, 0]] - t, axis=1))
                   for t in tris)
        near = fault_distance(obs, tris) < 2.0 * diam / n_quad
        u = np.empty((obs.shape[0], 3))
        sig = np.empty((obs.shape[0], 3, 3))
        for mask, g, kw in [(near, True, dict(n_c=n_c or 24, n_t=n_t or 20)),
                            (~near, False, dict(n_c=n_c, n_t=n_t))]:
            if np.any(mask):
                u[mask], sig[mask] = _eval_fields_total(
                    obs[mask], tris, slip=slip, mu=mu, nu=nu, eps=eps,
                    n_quad=n_quad, richardson_levels=richardson_levels,
                    richardson_ratio=richardson_ratio, graded=g,
                    block=block, **kw)
        return u, sig

    slip_cart = _slip_cart_per_tri(tris, slip)

    if _have_jax():
        import jax.numpy as jnp
        from .exact_bc_jax import make_kernels
        K = make_kernels(n_quad=n_quad, graded=graded, n_c=n_c, n_t=n_t)
        v0, v1, v2 = (jnp.asarray(tris[:, i]) for i in range(3))
        sc = jnp.asarray(slip_cart)
        # peak memory ~ block x M x (quad points per pair): shrink the chunk
        # for the graded rule (~2*n_c*n_t points vs n_quad^2 for tensor)
        if graded:
            npts = 2 * int(n_c or n_quad) * int(n_t or n_quad)
            block = max(256, int(block * n_quad**2 / npts))

        def _single(e):
            u = np.empty((obs.shape[0], 3))
            sig = np.empty((obs.shape[0], 3, 3))
            for lo in range(0, obs.shape[0], block):
                o = jnp.asarray(obs[lo:lo + block])
                dispM = K["disp_matrix"](o, v0, v1, v2, mu, nu, e)
                stressM = K["stress_matrix"](o, v0, v1, v2, mu, nu, e)
                u[lo:lo + block] = np.asarray(
                    jnp.einsum("nmik,mk->ni", dispM, sc))
                sig[lo:lo + block] = np.asarray(
                    jnp.einsum("nMabk,Mk->nab", stressM, sc))
            return u, sig
    else:                                   # numpy fallback: identical, slower
        slip_sdt = np.asarray(slip, float)
        if slip_sdt.ndim == 1:
            slip_sdt = np.broadcast_to(slip_sdt, (len(tris), 3))

        def _single(e):
            u = np.zeros((obs.shape[0], 3))
            sig = np.zeros((obs.shape[0], 3, 3))
            for t, s in zip(tris, slip_sdt):
                f = field_hs_mindlin(obs, t, s, nu, e, mu=mu, n_quad=n_quad,
                                     graded=graded, n_c=n_c, n_t=n_t)
                u += np.atleast_2d(f["disp"])
                sig += f["stress"].reshape(-1, 3, 3)
            return u, sig

    if richardson_levels > 1:
        us, sigs = [], []
        for e in eps_ladder(eps, richardson_levels, richardson_ratio):
            u_e, s_e = _single(e)
            us.append(u_e)
            sigs.append(s_e)
        return (combine_samples(us, richardson_ratio),
                combine_samples(sigs, richardson_ratio))
    return _single(eps)


def eval_fields(obs, tris, slip=SLIP, mu=MU, nu=NU, eps=EPS, n_quad=N_QUAD,
                richardson_levels=1, richardson_ratio=2.0,
                graded=False, n_c=None, n_t=None, block=8192,
                subtract_anelastic=True):
    """Displacement (N,3) and ELASTIC stress (N,3,3) at obs from slip on
    triangles `tris` (M,3,3), summed over triangles.

    Stress: the raw kernel value is the TOTAL stress C:(eps_elastic +
    eps_star); inside the ~eps fault zone it is dominated by the anelastic
    eigenstress C:eps_star (peak (3/4) mu s / eps, diverging as eps -> 0),
    which is NOT the elastic stress.  With `subtract_anelastic=True` (the
    default) the eigenstress at the nominal eps is subtracted, giving the
    bounded, Coulomb-relevant elastic stress; off the fault (beyond ~2 eps)
    the subtraction is a no-op.  Pass `subtract_anelastic=False` only for
    kernel diagnostics.  Displacement is unaffected either way.

    slip : (3,) [strike, dip, tensile] for all triangles, or (M,3) per
           triangle, in km (1 m slip = 1e-3).
    block : obs are evaluated in chunks of this size (see _eval_fields_total).
    graded : False (tensor rule), True (graded rule for all obs), or "auto"
           (graded only within 2*diam/n_quad of the fault).
    richardson_levels > 1 Richardson/Romberg-extrapolates u and sig over the
    eps-ladder.  NOTE: the eps->0 STRESS limit is singular ON the fault even
    after subtraction is applied at the nominal eps — for grids that touch
    the fault use eval_fields_fault_aware instead.
    """
    obs = np.asarray(obs, float)
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    u, sig = _eval_fields_total(obs, tris, slip=slip, mu=mu, nu=nu, eps=eps,
                                n_quad=n_quad,
                                richardson_levels=richardson_levels,
                                richardson_ratio=richardson_ratio,
                                graded=graded, n_c=n_c, n_t=n_t, block=block)
    if subtract_anelastic:
        sig = sig - eigenstress_at_points(obs, tris,
                                          _slip_cart_per_tri(tris, slip),
                                          mu, nu, eps)
    return u, sig


# ---------------------------------------------------------------------------
# Distance to the fault surface (for fault-aware eps-extrapolation)
# ---------------------------------------------------------------------------
# (_point_triangle_distance is imported from .anelastic — single home of the
# edge-tapered point-to-triangle distance.)
def fault_distance(obs, tris):
    """Min Euclidean distance from each obs (N,3) to the triangle set (M,3,3)."""
    obs = np.asarray(obs, float)
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    d = np.full(obs.shape[0], np.inf)
    for t in tris:
        d = np.minimum(d, _point_triangle_distance(obs, t[0], t[1], t[2]))
    return d


def eval_fields_fault_aware(obs, tris, slip=SLIP, mu=MU, nu=NU, eps=EPS,
                            n_quad=N_QUAD, graded=False, n_c=None, n_t=None,
                            richardson_levels=1, richardson_ratio=2.0,
                            blend_lo=1.0, blend_hi=4.0,
                            subtract_anelastic=True):
    """Field eval that is correct BOTH on the fault and off it near the surface.

    * DISPLACEMENT is finite everywhere (on the fault it is the average across
      the slip), so Richardson extrapolation is valid everywhere -> pure
      Richardson for u.
    * STRESS: each ladder rung is a TOTAL stress; on the fault its eps->0
      limit is singular (the anelastic eigenstress ~ 1/eps), so Richardson
      blows up there.  We blend by distance d to the fault surface: the
      regularized single-eps value for d <= blend_lo*eps, the Richardson
      value for d >= blend_hi*eps, smoothstep in between.  Then (default)
      the anelastic eigenstress C:eps_star at the NOMINAL eps is subtracted
      ONCE, after the blend — the on-fault result is the bounded elastic
      stress; off-fault the subtraction is a no-op.  The blend is kept even
      with subtraction: it guards Richardson from amplifying the ~1%-level
      residual of the marginal eigenstress model on exact on-fault grid
      lines.  (Per-rung subtraction + Richardson-everywhere is a possible
      follow-up, gated on a new validate check.)

    Reduces exactly to eval_fields when richardson_levels == 1.
    """
    obs = np.asarray(obs, float)
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    us, ss = [], []
    for e in eps_ladder(eps, richardson_levels, richardson_ratio):
        u_e, s_e = _eval_fields_total(obs, tris, slip=slip, mu=mu, nu=nu,
                                      eps=e, n_quad=n_quad, graded=graded,
                                      n_c=n_c, n_t=n_t)
        us.append(u_e)
        ss.append(s_e)

    def _finish(u, s):
        if subtract_anelastic:
            s = s - eigenstress_at_points(obs, tris,
                                          _slip_cart_per_tri(tris, slip),
                                          mu, nu, eps)
        return u, s

    if richardson_levels == 1:
        return _finish(us[0], ss[0])
    uR = combine_samples(us, richardson_ratio)      # displacement: everywhere
    sR = combine_samples(ss, richardson_ratio)      # stress: off-fault only
    s1 = ss[0]                                      # single-eps stress: on-fault
    d = fault_distance(obs, tris)
    lo, hi = blend_lo * eps, blend_hi * eps
    w = np.clip((d - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)                     # smoothstep
    s = w[:, None, None] * sR + (1.0 - w)[:, None, None] * s1
    return _finish(uR, s)


def fault_plane_segments(tris, axis, value):
    """Intersect each triangle with the plane coord[axis]=value; return segments
    in the other two coords (increasing index order).
      axis=2 (z=z0) -> (x,y) segments;  axis=0 (x=x0) -> (y,z) segments."""
    others = [i for i in range(3) if i != axis]
    segs = []
    for t in np.asarray(tris, float).reshape(-1, 3, 3):
        pts = []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            fa, fb = t[a, axis] - value, t[b, axis] - value
            if fa == 0.0:
                pts.append(t[a, others])
            if fa * fb < 0.0:
                s = fa / (fa - fb)
                pts.append((t[a] + s * (t[b] - t[a]))[others])
        if len(pts) >= 2:
            segs.append(np.array(pts[:2]))
    return segs


def make_rect_fault(half_len=5.0, depth=10.0, top=0.0, y0=0.0):
    """Two conformal triangles tiling the vertical rectangle x in
    [-half_len, half_len], z in [top-depth, top] at y=y0 (strike along x,
    both normals +y so unit strike-slip adds coherently).  top=0 gives a
    surface-breaking fault."""
    A = [-half_len, y0, top]; B = [half_len, y0, top]
    C = [half_len, y0, top - depth]; D = [-half_len, y0, top - depth]
    return np.array([[A, B, C], [A, C, D]])


def q_above_surface(h, eps):
    """Blob mass above the free surface for a unit Cortez source at depth
    h >= 0:  Q(h) = 1/2 - h(3 eps^2 + 2 h^2) / (4 (h^2+eps^2)^(3/2)).
    Q(0) = 1/2;  Q(h) ~ (3/16)(eps/h)^4 for h >> eps."""
    h = np.maximum(np.asarray(h, float), 0.0)
    return 0.5 - h * (3 * eps**2 + 2 * h**2) / (4 * (h**2 + eps**2) ** 1.5)


def moment_factor(tris, eps, n=24):
    """Fraction of the mollified potency RETAINED in the half-space:

        f = 1 - (1/A) * INT_A Q(depth(x)) dA

    (area-weighted over the triangle mesh; Q from q_above_surface).  For a
    surface-breaking rectangle of down-dip extent D this is 1 - eps/(4D) up
    to an O(eps^4/D^3) tail; ~1 for buried faults.  Dividing the slip by f
    restores the classical seismic moment at finite eps (and raises the
    in-zone stresses by the same factor — it trades zone-physics fidelity
    for moment fidelity; see mhf/moment_accounting.tex)."""
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    pts, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (pts + 1.0); w = 0.5 * w
    xi1 = np.repeat(pts, n); u = np.tile(pts, n)
    xi2 = u * (1.0 - xi1)
    ww = np.repeat(w, n) * np.tile(w, n) * (1.0 - xi1)      # sums to 1/2
    lost = 0.0
    area = 0.0
    for v1, v2, v3 in tris:
        a2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        y = ((1.0 - xi1 - xi2)[:, None] * v1 + xi1[:, None] * v2
             + xi2[:, None] * v3)
        lost += a2 * np.sum(ww * q_above_surface(-y[:, 2], eps))   # = INT_T Q dA
        area += 0.5 * a2
    return 1.0 - lost / area


def coulomb_stress(sig, n_hat, s_hat, friction=0.4):
    """Coulomb failure stress change on receiver planes with unit normal
    n_hat and slip direction s_hat (both (3,)):

        dCFS = s_hat . sig . n_hat  +  friction * (n_hat . sig . n_hat)

    (tension-positive normal stress: positive sigma_n = unclamping promotes
    failure).  sig is (N,3,3); returns (N,).

    `sig` must be the ELASTIC stress (the eval_fields/eval_fields_fault_aware
    default, `subtract_anelastic=True`).  The raw TOTAL kernel stress carries
    the anelastic eigenstress C:eps_star inside the ~eps fault zone — an
    on-fault slip-sense shear peaking at (3/4) mu s / eps and diverging as
    eps -> 0.  That term is the smeared slip itself, not a stress available
    to load receiver faults; feeding it to dCFS produces a spurious positive
    band atop the fault that grows as 1/eps.  (An earlier reading of that
    band as the physical stress of a finite-width fault zone is deprecated;
    see mhf.anelastic.)  The subtracted elastic dCFS is bounded and
    eps-stable: stress drop broadside of the patch, positive lobes beyond
    the edges — now including the on-fault pixels."""
    n = np.asarray(n_hat, float); n = n / np.linalg.norm(n)
    s = np.asarray(s_hat, float); s = s / np.linalg.norm(s)
    t = np.einsum("nij,j->ni", sig, n)          # traction on the plane
    return t @ s + friction * (t @ n)


def von_mises(sig):
    """Von Mises equivalent stress sqrt(3/2 * dev(sig):dev(sig)); (N,3,3)->(N,).

    Meaningful as an elastic yield measure only for eigenstress-subtracted
    stress; the raw total's on-fault value is dominated by C:eps_star ~ 1/eps."""
    sig = np.asarray(sig, float)
    dev = sig - (np.trace(sig, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
    return np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))


def report_ranges(u, sig):
    """Print per-component field ranges (mm / MPa) for a sanity check."""
    u = u * KM_TO_MM
    sig = sig * GPA_TO_MPA
    print(f"  |u| max = {np.max(np.linalg.norm(u, axis=1)):.2f} mm  "
          f"(ux,uy,uz max |.| = {np.max(np.abs(u[:, 0])):.1f}, "
          f"{np.max(np.abs(u[:, 1])):.1f}, {np.max(np.abs(u[:, 2])):.1f} mm)")
    comps = [(0, 0, "xx"), (1, 1, "yy"), (2, 2, "zz"),
             (0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]
    s = "  ".join(f"s{nm}={np.max(np.abs(sig[:, i, j])):.2f}" for i, j, nm in comps)
    print(f"  stress max |.| (MPa): {s}")

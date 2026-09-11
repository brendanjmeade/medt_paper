"""JAX port of the exact-BC mollified Mindlin half-space triangle kernel.

GPU-ready, vmap/jit-able re-implementation of the (numpy) exact-BC kernel
(exact_bc_triangle.py), kept SEPARATE from the validated numpy reference.

  G^{eps,HS} = G^{F,eps} (mollified Kelvin direct, analytic) + G^{C,eps}
               (exact-BC complementary correction, quadrature on the smooth part).

Design:
  * Direct part: a faithful jnp transcription of analytical_kernels
    (integrate_all_moments / integrate_DG / integrate_D2G / dd-displacement /
    stress).  For eps>0 every value-guard in the numpy version is provably
    inactive (rho2 = d_perp^2 + z^2 + eps^2 >= eps^2 > 0, R_eps >= eps > 0), so
    they are dropped; the order/edge loops are static (unrolled at trace time);
    the moment dict uses STATIC keys (fine under jit/vmap).
  * Correction part: the symbolic kernels from exact_bc_triangle, lambdified to
    JAX with cse=True (compresses ~20x); the triangle quadrature is reduced
    INSIDE the per-(obs,src) kernel so the n_quad^2 axis is never broadcast
    across N x M pairs.
  * vmap over (obs, src); jit.  Stress/strain are explicit analytic kernels
    (NOT autodiff) for maximum accuracy.

float64 is required (jax_enable_x64); enabled at import.

NOTE on devices: this macOS has a broken jax-metal plugin (it aborts with
VisibleDeviceCount 0).  Run with `JAX_PLATFORMS=cpu` here; on Linux+CUDA JAX
uses the GPU automatically.  The module does NOT force a platform.
"""

import functools

import numpy as np
import sympy as sp
from math import comb

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ====================================================================
# Direct part: jnp transcription of analytical_kernels (eps>0, guards dropped)
# ====================================================================
def _solid_angle(v1, v2, v3, obs):
    r1 = v1 - obs; r2 = v2 - obs; r3 = v3 - obs
    R1 = jnp.linalg.norm(r1); R2 = jnp.linalg.norm(r2); R3 = jnp.linalg.norm(r3)
    numer = jnp.dot(r1, jnp.cross(r2, r3))
    denom = (R1 * R2 * R3 + R3 * jnp.dot(r1, r2)
             + R1 * jnp.dot(r2, r3) + R2 * jnp.dot(r1, r3))
    return 2.0 * jnp.arctan2(numer, denom)


def _J(u, rho2, order):
    """Antiderivative of 1/(u^2+rho2)^(order/2).  order in {1,3,5,7}; eps>0 => rho2>0."""
    R = jnp.sqrt(u**2 + rho2)
    if order == 1:
        return jnp.log(u + R)
    if order == 3:
        return u / (rho2 * R)
    if order == 5:
        return u * (3 * rho2 + 2 * u**2) / (3 * rho2**2 * R**3)
    if order == 7:
        return u * (15 * rho2**2 + 20 * rho2 * u**2 + 8 * u**4) / (15 * rho2**3 * R**5)
    raise ValueError(order)


def _K(u, rho2, order):
    """Antiderivative of u/(u^2+rho2)^(order/2)."""
    R = jnp.sqrt(u**2 + rho2)
    if order == 1:
        return R
    if order == 3:
        return -1.0 / R
    if order == 5:
        return -1.0 / (3.0 * R**3)
    if order == 7:
        return -1.0 / (5.0 * R**5)
    raise ValueError(order)


def integrate_all_moments(v1, v2, v3, obs, eps):
    """jnp version of analytical_kernels.integrate_all_moments (eps>0)."""
    e1 = v2 - v1; e2 = v3 - v1
    normal = jnp.cross(e1, e2)
    area2 = jnp.linalg.norm(normal)
    normal = normal / area2

    z = jnp.dot(obs - v1, normal)
    h2 = z**2 + eps**2
    h = jnp.sqrt(h2)
    obs_proj = obs - z * normal

    ex = e1 / jnp.linalg.norm(e1)
    ey = jnp.cross(normal, ex)

    def to_2d(p):
        d = p - obs_proj
        return jnp.array([jnp.dot(d, ex), jnp.dot(d, ey)])

    p1, p2, p3 = to_2d(v1), to_2d(v2), to_2d(v3)

    obs_eff = obs_proj + h * normal
    Omega = _solid_angle(v1, v2, v3, obs_eff)
    I3 = -Omega / h

    edges = [(p1, p2), (p2, p3), (p3, p1)]
    BN1 = {}; BN2 = {}; BD = {}
    for pa, pb in edges:
        edge_vec = pb - pa
        L = jnp.linalg.norm(edge_vec)
        t_hat = edge_vec / L
        n_out = jnp.array([t_hat[1], -t_hat[0]])
        c1, c2 = n_out[0], n_out[1]
        s1, s2 = t_hat[0], t_hat[1]
        dp = jnp.dot(pa, n_out)
        u_a = jnp.dot(pa, t_hat)
        u_b = jnp.dot(pb, t_hat)
        rho2 = dp**2 + h2

        dJ = {}; dK = {}
        for m in [1, 3, 5]:
            dJ[m] = _J(u_b, rho2, m) - _J(u_a, rho2, m)
            dK[m] = _K(u_b, rho2, m) - _K(u_a, rho2, m)
        dL = {}; dM = {}
        for m in [3, 5]:
            dL[m] = dJ[m - 2] - rho2 * dJ[m]
            dM[m] = dK[m - 2] - rho2 * dK[m]

        def edge_integral(a, b, m):
            total = 0.0
            for i in range(a + 1):
                for j in range(b + 1):
                    coeff = (comb(a, i) * comb(b, j)
                             * dp**(a + b - i - j) * c1**(a - i) * s1**i
                             * c2**(b - j) * s2**j)
                    k = i + j
                    if k == 0:
                        delta = dJ[m]
                    elif k == 1:
                        delta = dK[m]
                    elif k == 2:
                        delta = dL[m]
                    else:
                        delta = dM[m]
                    total = total + coeff * delta
            return total

        for tot_order in range(4):
            for a in range(tot_order + 1):
                b = tot_order - a
                for m in [1, 3, 5]:
                    if tot_order >= 2 and m == 1:
                        continue
                    if tot_order >= 3 and m == 3:
                        continue
                    key = (a, b, m)
                    val = edge_integral(a, b, m)
                    BN1[key] = BN1.get(key, 0.0) + c1 * val
                    BN2[key] = BN2.get(key, 0.0) + c2 * val
                    BD[key] = BD.get(key, 0.0) + dp * val

    M = {}
    E1 = BD.get((0, 0, 1), 0.0); E3 = BD.get((0, 0, 3), 0.0); E5 = BD.get((0, 0, 5), 0.0)
    I1 = E1 - h2 * I3
    I5 = (E3 + I3) / (3 * h2)
    I7 = (E5 + 3 * I5) / (5 * h2)
    M[(0, 0, 1)] = I1; M[(0, 0, 3)] = I3; M[(0, 0, 5)] = I5; M[(0, 0, 7)] = I7

    for n in [3, 5, 7]:
        m = n - 2; cc = -1.0 / (n - 2)
        M[(1, 0, n)] = cc * BN1.get((0, 0, m), 0.0)
        M[(0, 1, n)] = cc * BN2.get((0, 0, m), 0.0)
    for n in [3, 5, 7]:
        m = n - 2; cc = 1.0 / (n - 2)
        M[(2, 0, n)] = cc * (M[(0, 0, m)] - BN1.get((1, 0, m), 0.0))
        M[(1, 1, n)] = -cc * BN1.get((0, 1, m), 0.0)
        M[(0, 2, n)] = cc * (M[(0, 0, m)] - BN2.get((0, 1, m), 0.0))
    for n in [5, 7]:
        m = n - 2; cc = 1.0 / (n - 2)
        if (2, 0, m) in BN1:
            M[(3, 0, n)] = cc * (2 * M[(1, 0, m)] - BN1[(2, 0, m)])
        if (1, 1, m) in BN1:
            M[(2, 1, n)] = cc * (M[(0, 1, m)] - BN1[(1, 1, m)])
        if (0, 2, m) in BN1:
            M[(1, 2, n)] = -cc * BN1[(0, 2, m)]
        if (0, 2, m) in BN2:
            M[(0, 3, n)] = cc * (2 * M[(0, 1, m)] - BN2[(0, 2, m)])
    n = 7; m = 5; cc = 1.0 / 5.0
    if (3, 0, m) in BN1:
        M[(4, 0, n)] = cc * (3 * M[(2, 0, m)] - BN1[(3, 0, m)])
    if (2, 1, m) in BN1:
        M[(3, 1, n)] = cc * (2 * M[(1, 1, m)] - BN1[(2, 1, m)])
    if (1, 2, m) in BN1:
        M[(2, 2, n)] = cc * (M[(0, 2, m)] - BN1[(1, 2, m)])
    if (0, 3, m) in BN1:
        M[(1, 3, n)] = cc * (-BN1[(0, 3, m)])
    if (0, 3, m) in BN2:
        M[(0, 4, n)] = cc * (3 * M[(0, 2, m)] - BN2[(0, 3, m)])

    def gm(a, b, n):
        return M.get((a, b, n), 0.0)

    V = {}
    for n in [3, 5, 7]:
        V[n] = jnp.stack([
            -ex[i] * gm(1, 0, n) - ey[i] * gm(0, 1, n) + z * normal[i] * gm(0, 0, n)
            for i in range(3)])

    comps = [(-1.0, ex, 'x'), (-1.0, ey, 'y'), (z, normal, 'z')]

    def tensor(n, rank):
        out = jnp.zeros((3,) * rank)
        for sel in _index_product(comps, rank):
            a = sum(1 for _, _, lab in sel if lab == 'x')
            b = sum(1 for _, _, lab in sel if lab == 'y')
            coeff = 1.0; zpow = 1.0
            for s, _, lab in sel:
                if lab == 'z':
                    zpow = zpow * s
                else:
                    coeff = coeff * s
            mom = gm(a, b, n)
            outer = sel[0][1]
            for k in range(1, rank):
                outer = jnp.tensordot(outer, sel[k][1], axes=0)
            out = out + (coeff * zpow * mom) * outer
        return out

    T2 = {n: tensor(n, 2) for n in [3, 5, 7]}
    T3 = {n: tensor(n, 3) for n in [5, 7]}
    T4 = {7: tensor(7, 4)}
    return {'ex': ex, 'ey': ey, 'nhat': normal, 'z': z, 'h': h,
            'I': {1: I1, 3: I3, 5: I5, 7: I7}, 'V': V, 'T2': T2, 'T3': T3, 'T4': T4}


def _index_product(comps, rank):
    if rank == 0:
        return [()]
    sub = _index_product(comps, rank - 1)
    return [(c,) + s for c in comps for s in sub]


def integrate_DG(v1, v2, v3, obs, mu, nu, eps):
    mom = integrate_all_moments(v1, v2, v3, obs, eps)
    V3 = mom['V'][3]; V5 = mom['V'][5]; T3_5 = mom['T3'][5]
    C1 = 1.0 / (16.0 * jnp.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cb = 6.0 * (1.0 - nu) * eps**2
    eye = jnp.eye(3)
    G1 = jnp.zeros((3, 3, 3))
    # -c34 dij V3[m] + dim V3[j] + djm V3[i] - 3 T3_5[i,j,m] - cb dij V5[m]
    G1 = -c34 * jnp.einsum('ij,m->ijm', eye, V3)
    G1 = G1 + jnp.einsum('im,j->ijm', eye, V3)
    G1 = G1 + jnp.einsum('jm,i->ijm', eye, V3)
    G1 = G1 - 3.0 * T3_5
    G1 = G1 - cb * jnp.einsum('ij,m->ijm', eye, V5)
    return C1 * G1


def integrate_D2G(v1, v2, v3, obs, mu, nu, eps):
    mom = integrate_all_moments(v1, v2, v3, obs, eps)
    I3 = mom['I'][3]; I5 = mom['I'][5]
    T2_5 = mom['T2'][5]; T2_7 = mom['T2'][7]; T4_7 = mom['T4'][7]
    C1 = 1.0 / (16.0 * jnp.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cb = 2.0 * (1.0 - nu) * eps**2
    eye = jnp.eye(3)
    # Build via einsum of Kronecker deltas (indices r,p,s,q)
    drp_sq = jnp.einsum('rp,sq->rpsq', eye, eye)
    drs_pq = jnp.einsum('rs,pq->rpsq', eye, eye)
    dps_rq = jnp.einsum('ps,rq->rpsq', eye, eye)
    ID2G = (-c34 * drp_sq * I3 + drs_pq * I3 + dps_rq * I3
            + c34 * jnp.einsum('rp,sq->rpsq', eye, T2_5) * 3.0
            - 3.0 * jnp.einsum('rs,pq->rpsq', eye, T2_5)
            - 3.0 * jnp.einsum('ps,rq->rpsq', eye, T2_5)
            - 3.0 * jnp.einsum('rq,ps->rpsq', eye, T2_5)
            - 3.0 * jnp.einsum('pq,rs->rpsq', eye, T2_5)
            - 3.0 * jnp.einsum('sq,rp->rpsq', eye, T2_5)
            + 15.0 * T4_7
            + cb * jnp.einsum('rp,sq->rpsq', eye, (-3.0 * I5 * eye + 15.0 * T2_7)))
    return C1 * ID2G


def _stiffness(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps):
    G1 = integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = _stiffness(mu, nu)
    n = normal
    trace_im = jnp.einsum('imm->i', G1)
    term1 = mu * jnp.einsum('m,ijm->ij', n, G1)
    term2 = lam * jnp.einsum('j,i->ij', n, trace_im)
    term3 = mu * jnp.einsum('k,ikj->ij', n, G1)
    return -(term1 + term2 + term3)


def analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps):
    ID2G = integrate_D2G(v1, v2, v3, obs, mu, nu, eps)
    lam = _stiffness(mu, nu)
    nv = normal
    trace_ID2G = jnp.einsum('rpsp->rs', ID2G)
    ID2G_nq = jnp.einsum('q,risq->ris', nv, ID2G)     # [r,i,s]
    ID2G_np = jnp.einsum('p,rpsi->rsi', nv, ID2G)     # [r,s,i]
    B = (lam * jnp.einsum('k,rs->rsk', nv, trace_ID2G)
         + mu * jnp.transpose(ID2G_nq, (0, 2, 1))     # [r,s,k] from [r,k,s]
         + mu * ID2G_np)
    trace_B = jnp.einsum('rrk->k', B)
    eye = jnp.eye(3)
    H = -(lam * jnp.einsum('mn,k->mnk', eye, trace_B)
          + mu * jnp.transpose(B, (0, 1, 2))           # B[m,n,k]
          + mu * jnp.transpose(B, (1, 0, 2)))          # B[n,m,k]
    return H


# ====================================================================
# Correction part: lambdify the symbolic kernels (exact_bc_triangle) to JAX
# ====================================================================
from . import exact_bc_triangle as _T

_args_sym = _T._args                                     # (x,y,z,c,eps,mu,nu)
# disp source-derivatives DG[i,j,m]: m=0->-d/dx, 1->-d/dy, 2->+d/dc (signs baked in)
_lam_dGc = [sp.lambdify(_args_sym, m, modules="jax", cse=True)
            for m in (_T._dGc_dsx, _T._dGc_dsy, _T._dGc_dsz)]
# mixed 2nd-derivatives for stress, in a fixed order
_D2KEYS = [(_T._x, _T._x), (_T._y, _T._y), (_T._x, _T._y), (_T._x, _T._z),
           (_T._y, _T._z), (_T._x, _T._c), (_T._y, _T._c), (_T._z, _T._c)]
_lam_d2 = [sp.lambdify(_args_sym, _T._d2[k], modules="jax", cse=True) for k in _D2KEYS]


def _tri_quad(n):
    pts, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (pts + 1.0); w = 0.5 * w
    xi1, xi2, ww = [], [], []
    for i in range(n):
        for j in range(n):
            xi1.append(pts[i]); xi2.append(pts[j] * (1.0 - pts[i]))
            ww.append(w[i] * w[j] * (1.0 - pts[i]))
    return jnp.array(xi1), jnp.array(xi2), jnp.array(ww)


def _corr_DG(X, Y, Z, C, eps, mu, nu):
    """Batched DG[q,i,j,m] = d Gcorr_ij/d src_m at quad points (X,Y,Z,C arrays)."""
    cols = [jax.vmap(_lam_dGc[m], in_axes=(0, 0, 0, 0, None, None, None))(
        X, Y, Z, C, eps, mu, nu) for m in range(3)]          # each (Q,3,3)
    return jnp.stack(cols, axis=-1)                          # (Q,3,3,3) [q,i,j,m]


def _corr_DDG(X, Y, Z, C, eps, mu, nu):
    """Batched DDG[q,i,j,p,m] = d^2 Gcorr_ij/(d obs_p d src_m)."""
    d = [jax.vmap(f, in_axes=(0, 0, 0, 0, None, None, None))(X, Y, Z, C, eps, mu, nu)
         for f in _lam_d2]                                   # each (Q,3,3)
    xx, yy, xy, xz, yz, xc, yc, zc = d
    Q = X.shape[0]
    DDG = jnp.zeros((Q, 3, 3, 3, 3))
    DDG = DDG.at[:, :, :, 0, 0].set(-xx); DDG = DDG.at[:, :, :, 0, 1].set(-xy); DDG = DDG.at[:, :, :, 0, 2].set(xc)
    DDG = DDG.at[:, :, :, 1, 0].set(-xy); DDG = DDG.at[:, :, :, 1, 1].set(-yy); DDG = DDG.at[:, :, :, 1, 2].set(yc)
    DDG = DDG.at[:, :, :, 2, 0].set(-xz); DDG = DDG.at[:, :, :, 2, 1].set(-yz); DDG = DDG.at[:, :, :, 2, 2].set(zc)
    return DDG


def _quad_points(v1, v2, v3, xi1, xi2):
    w0 = (1.0 - xi1 - xi2)
    return (w0[:, None] * v1 + xi1[:, None] * v2 + xi2[:, None] * v3)   # (Q,3)


# ====================================================================
# GRADED correction quadrature (obs-dependent sinh2D), static shape for vmap/jit.
# Mirrors mollified_kernel.exact_bc_graded_quad (numpy).  The triangle is ALWAYS
# split into 2 'horizontal-edge' sub-triangles (one may be degenerate -> masked
# to zero weight), so the quad-point count 2*n_c*n_t is static.  Returns physical
# points ys[Q,3] and weights wq[Q] (sum = triangle area).
# ====================================================================
def _point_tri_dist(p, a, b, c):
    """Euclidean distance from point p to triangle (a,b,c), jit/vmap-safe."""
    n = jnp.cross(b - a, c - a)
    nn = jnp.linalg.norm(n) + 1e-300
    perp = (p - a) @ n / nn
    proj = p - perp * n / nn
    v0, v1, v2 = b - a, c - a, proj - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    den = d00 * d11 - d01 * d01 + 1e-300
    v = (d11 * (v2 @ v0) - d01 * (v2 @ v1)) / den
    w = (d00 * (v2 @ v1) - d01 * (v2 @ v0)) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)

    def seg(e0, e1):
        t = jnp.clip((p - e0) @ (e1 - e0)
                     / ((e1 - e0) @ (e1 - e0) + 1e-300), 0.0, 1.0)
        return jnp.linalg.norm(p - (e0 + t * (e1 - e0)))

    d_edge = jnp.minimum(jnp.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return jnp.where(inside, jnp.abs(perp), d_edge)


def _subtri_graded(apex, vq, vr, z_apex, z_h, obs, eps, gt, gw, gs, gws):
    """gt/gw and gs/gws are HALF-size Gauss rules; both directions use
    composite two-panel rules (split at the obs-distance octave in t, and at
    the obs-foot peak s=0 in-plane) — a single Gauss-on-sinh panel puts its
    sparse center on the near-singular peak when the span is large and
    needed ~50 nodes/axis (the 2026-06-11 animation artifact)."""
    tiny = 1e-12
    twoA = jnp.linalg.norm(jnp.cross(vq - apex, vr - vq))
    dzr = z_h - z_apex                                   # signed depth span
    valid = jnp.abs(dzr) > tiny
    dzr_safe = jnp.where(valid, dzr, 1.0)
    dz = jnp.abs(dzr_safe)
    c_lo = jnp.minimum(z_apex, z_h); c_hi = jnp.maximum(z_apex, z_h)
    t_lo = jnp.arcsinh(-c_hi / eps); t_hi = jnp.arcsinh(-c_lo / eps)
    d_obs = _point_tri_dist(obs, apex, vq, vr)
    tspan = t_hi - t_lo
    t_mid = jnp.clip(jnp.arcsinh(d_obs / eps),
                     t_lo + 0.05 * tspan, t_hi - 0.05 * tspan)
    tc = jnp.concatenate([0.5 * (t_mid - t_lo) * gt + 0.5 * (t_mid + t_lo),
                          0.5 * (t_hi - t_mid) * gt + 0.5 * (t_hi + t_mid)])
    wtc = jnp.concatenate([0.5 * (t_mid - t_lo) * gw,
                           0.5 * (t_hi - t_mid) * gw])
    cc = -eps * jnp.sinh(tc)                             # depth nodes (nc,)
    dcw = eps * jnp.cosh(tc) * wtc                       # |dc| weights
    f = (cc - z_apex) / dzr_safe                         # 0 at apex -> 1 at edge
    P1 = apex[None, :] + f[:, None] * (vq - apex)[None, :]       # (nc,3)
    seg = f[:, None] * (vr - vq)[None, :]                        # (nc,3)
    L = jnp.linalg.norm(seg, axis=1)
    L_safe = jnp.where(L > tiny, L, 1.0)
    base_w = (twoA / dz) * f * dcw                       # depth weight (nc,)
    that = seg / L_safe[:, None]
    tau_foot = jnp.sum((obs[None, :] - P1) * that, axis=1) / L_safe
    a = jnp.sqrt(cc * cc + eps * eps)
    perp = obs[None, :] - (P1 + tau_foot[:, None] * seg)
    d_perp = jnp.linalg.norm(perp[:, :2], axis=1)
    w_scale = jnp.maximum(jnp.sqrt(d_perp ** 2 + (a - obs[2]) ** 2), eps) / L_safe
    s_lo = jnp.arcsinh((0.0 - tau_foot) / w_scale)
    s_hi = jnp.arcsinh((1.0 - tau_foot) / w_scale)
    sspan = s_hi - s_lo
    s_mid = jnp.clip(0.0, s_lo + 0.05 * sspan, s_hi - 0.05 * sspan)
    s = jnp.concatenate(
        [0.5 * (s_mid - s_lo)[:, None] * gs[None, :]
         + 0.5 * (s_mid + s_lo)[:, None],
         0.5 * (s_hi - s_mid)[:, None] * gs[None, :]
         + 0.5 * (s_hi + s_mid)[:, None]], axis=1)
    ws = jnp.concatenate(
        [0.5 * (s_mid - s_lo)[:, None] * gws[None, :],
         0.5 * (s_hi - s_mid)[:, None] * gws[None, :]], axis=1)
    tau = tau_foot[:, None] + w_scale[:, None] * jnp.sinh(s)     # (nc,nt)
    dtau = w_scale[:, None] * jnp.cosh(s) * ws
    pts = P1[:, None, :] + tau[:, :, None] * seg[:, None, :]     # (nc,nt,3)
    wts = base_w[:, None] * dtau                                 # (nc,nt)
    wts = jnp.where(valid, wts, 0.0)                             # mask degenerate
    pts = jnp.where(valid, pts, apex[None, None, :])             # keep finite
    nc, nt = wts.shape
    return pts.reshape(nc * nt, 3), wts.reshape(nc * nt)


def _graded_quad_points(obs, v1, v2, v3, eps, gt, gw, gs, gws):
    V = jnp.stack([v1, v2, v3])
    order = jnp.argsort(V[:, 2])
    va = V[order[0]]; vb = V[order[1]]; vc = V[order[2]]
    za, zb, zc = va[2], vb[2], vc[2]
    denom = jnp.where(jnp.abs(zc - za) > 1e-14, zc - za, 1.0)
    vd = va + ((zb - za) / denom) * (vc - va)            # on edge va->vc at z=zb
    p0, w0 = _subtri_graded(va, vb, vd, za, zb, obs, eps, gt, gw, gs, gws)  # lower
    p1, w1 = _subtri_graded(vc, vb, vd, zc, zb, obs, eps, gt, gw, gs, gws)  # upper
    pts = jnp.concatenate([p0, p1], 0)
    wts = jnp.concatenate([w0, w1], 0)
    # (Near-)horizontal fallback, mirroring the numpy graded_quad guard: if the
    # depth split dropped weight (sum(w) != area), select a static-shape
    # collapsed tensor rule instead — for a constant-depth triangle the depth
    # sinh substitution is unnecessary and the correction integrand is smooth.
    area = 0.5 * jnp.linalg.norm(jnp.cross(v2 - v1, v3 - v1))
    nt = gs.shape[0]
    xi1 = jnp.repeat(0.5 * (gt + 1.0), nt)               # (nc2*nt2,)
    uu = jnp.tile(0.5 * (gs + 1.0), gt.shape[0])
    xi2 = uu * (1.0 - xi1)
    tpts = _quad_points(v1, v2, v3, xi1, xi2)            # (nc2*nt2,3)
    twts = (0.25 * jnp.repeat(gw, nt) * jnp.tile(gws, gt.shape[0])
            * (1.0 - xi1) * (2.0 * area))                # sums to area
    # graded path emits 2 sub-tris x (2*nc2)x(2*nt2) points = 8*nc2*nt2:
    # tile the tensor rule to the same static shape with weights/8
    tpts2 = jnp.tile(tpts, (8, 1))
    twts2 = jnp.tile(twts, 8) / 8.0
    bad = jnp.abs(jnp.sum(wts) - area) > 1e-8 * area
    return jnp.where(bad, tpts2, pts), jnp.where(bad, twts2, wts)


# ====================================================================
# Per-(obs,src) kernels — quadrature reduced INSIDE (never broadcast over pairs)
# ====================================================================
def disp_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, xi1, xi2, ww):
    """U[i,k]: u_i(obs) = U[i,k]*slip_k.  Analytic direct + quad correction."""
    U = analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps)
    ys = _quad_points(v1, v2, v3, xi1, xi2)
    X = obs[0] - ys[:, 0]; Y = obs[1] - ys[:, 1]
    Z = jnp.full_like(X, obs[2]); C = ys[:, 2]
    DG = _corr_DG(X, Y, Z, C, eps, mu, nu)                  # (Q,3,3,3)
    lam = _stiffness(mu, nu); n = normal
    tr = jnp.einsum('qimm->qi', DG)
    T = (mu * jnp.einsum('l,qikl->qik', n, DG)
         + mu * jnp.einsum('l,qilk->qik', n, DG)
         + lam * jnp.einsum('k,qi->qik', n, tr))
    area2 = jnp.linalg.norm(jnp.cross(v2 - v1, v3 - v1))
    U = U + jnp.einsum('q,qik->ik', ww * area2, T)
    return U


def stress_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, xi1, xi2, ww):
    """H[m,n,k]: sigma_mn(obs) = H[m,n,k]*slip_k.  Analytic direct stress + quad."""
    H = analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps)
    lam = _stiffness(mu, nu); n = normal
    ys = _quad_points(v1, v2, v3, xi1, xi2)
    X = obs[0] - ys[:, 0]; Y = obs[1] - ys[:, 1]
    Z = jnp.full_like(X, obs[2]); C = ys[:, 2]
    DDG = _corr_DDG(X, Y, Z, C, eps, mu, nu)                # (Q,3,3,3,3)[q,i,j,p,m]
    tr = jnp.einsum('qimpm->qip', DDG)
    S = (mu * jnp.einsum('l,qikpl->qipk', n, DDG)
         + mu * jnp.einsum('l,qilpk->qipk', n, DDG)
         + lam * jnp.einsum('k,qip->qipk', n, tr))
    area2 = jnp.linalg.norm(jnp.cross(v2 - v1, v3 - v1))
    Gdu = jnp.einsum('q,qipk->ipk', ww * area2, S)          # corr grad [i,p,k]
    eye = jnp.eye(3)
    e = 0.5 * (jnp.transpose(Gdu, (2, 0, 1)) + jnp.transpose(Gdu, (2, 1, 0)))   # (k,i,p) sym
    # correction stress per k: lam tr(e_k) I + 2 mu e_k
    tr_e = jnp.einsum('kii->k', e)
    Hcorr = (lam * jnp.einsum('k,mn->mnk', tr_e, eye)
             + 2.0 * mu * jnp.transpose(e, (1, 2, 0)))      # (m,n,k)
    return H + Hcorr


# --- graded-quadrature variants (obs-dependent points, physical weights) ---
def disp_dd_kernel_graded(obs, v1, v2, v3, normal, mu, nu, eps, gt, gw, gs, gws):
    U = analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps)
    ys, wq = _graded_quad_points(obs, v1, v2, v3, eps, gt, gw, gs, gws)
    X = obs[0] - ys[:, 0]; Y = obs[1] - ys[:, 1]
    Z = jnp.full_like(X, obs[2]); C = ys[:, 2]
    DG = _corr_DG(X, Y, Z, C, eps, mu, nu)
    lam = _stiffness(mu, nu); n = normal
    tr = jnp.einsum('qimm->qi', DG)
    T = (mu * jnp.einsum('l,qikl->qik', n, DG)
         + mu * jnp.einsum('l,qilk->qik', n, DG)
         + lam * jnp.einsum('k,qi->qik', n, tr))
    return U + jnp.einsum('q,qik->ik', wq, T)               # wq already physical


def stress_dd_kernel_graded(obs, v1, v2, v3, normal, mu, nu, eps, gt, gw, gs, gws):
    H = analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps)
    lam = _stiffness(mu, nu); n = normal
    ys, wq = _graded_quad_points(obs, v1, v2, v3, eps, gt, gw, gs, gws)
    X = obs[0] - ys[:, 0]; Y = obs[1] - ys[:, 1]
    Z = jnp.full_like(X, obs[2]); C = ys[:, 2]
    DDG = _corr_DDG(X, Y, Z, C, eps, mu, nu)
    tr = jnp.einsum('qimpm->qip', DDG)
    S = (mu * jnp.einsum('l,qikpl->qipk', n, DDG)
         + mu * jnp.einsum('l,qilpk->qipk', n, DDG)
         + lam * jnp.einsum('k,qip->qipk', n, tr))
    Gdu = jnp.einsum('q,qipk->ipk', wq, S)
    eye = jnp.eye(3)
    e = 0.5 * (jnp.transpose(Gdu, (2, 0, 1)) + jnp.transpose(Gdu, (2, 1, 0)))
    tr_e = jnp.einsum('kii->k', e)
    Hcorr = (lam * jnp.einsum('k,mn->mnk', tr_e, eye)
             + 2.0 * mu * jnp.transpose(e, (1, 2, 0)))
    return H + Hcorr


# ====================================================================
# Public entry points: single, and vmapped (obs x src) influence tensors
# ====================================================================
def _normal(v1, v2, v3):
    nrm = jnp.cross(v2 - v1, v3 - v1)
    return nrm / jnp.linalg.norm(nrm)


@functools.lru_cache(maxsize=16)
def make_kernels(n_quad=8, graded=False, n_c=None, n_t=None):
    """Return jitted (disp_U, stress_H) single-pair kernels and vmapped matrix
    builders.  CACHED on (n_quad, graded, n_c, n_t): repeated calls with the
    same settings reuse the compiled kernels (~2 s trace+jit happens once).

    graded=False (default): fixed collapsed tensor-Gauss correction rule of order
      n_quad (obs-independent).  Fast; for buried faults or moderate eps.
    graded=True: obs-dependent sinh2D graded correction rule (n_c depth x n_t
      in-plane nodes, default = n_quad) that resolves the near-singular correction
      integrand of SURFACE-BREAKING faults at small eps -- the enabler so each eps
      on a Richardson/Romberg ladder stays quadrature-converged (see
      mhf.eps_extrapolation, mhf.exact_bc_graded_quad)."""
    if graded:
        nc = int(n_c if n_c is not None else n_quad)
        nt = int(n_t if n_t is not None else n_quad)
        # half-size rules: _subtri_graded builds composite two-panel rules
        _gt, _gw = np.polynomial.legendre.leggauss(max(2, nc // 2))
        _gs, _gws = np.polynomial.legendre.leggauss(max(2, nt // 2))
        gt, gw = jnp.asarray(_gt), jnp.asarray(_gw)
        gs, gws = jnp.asarray(_gs), jnp.asarray(_gws)

        def disp_U(obs, v1, v2, v3, mu, nu, eps):
            return disp_dd_kernel_graded(obs, v1, v2, v3, _normal(v1, v2, v3),
                                         mu, nu, eps, gt, gw, gs, gws)

        def stress_H(obs, v1, v2, v3, mu, nu, eps):
            return stress_dd_kernel_graded(obs, v1, v2, v3, _normal(v1, v2, v3),
                                           mu, nu, eps, gt, gw, gs, gws)
    else:
        xi1, xi2, ww = _tri_quad(n_quad)

        def disp_U(obs, v1, v2, v3, mu, nu, eps):
            return disp_dd_kernel(obs, v1, v2, v3, _normal(v1, v2, v3), mu, nu, eps, xi1, xi2, ww)

        def stress_H(obs, v1, v2, v3, mu, nu, eps):
            return stress_dd_kernel(obs, v1, v2, v3, _normal(v1, v2, v3), mu, nu, eps, xi1, xi2, ww)

    disp_U_j = jax.jit(disp_U)
    stress_H_j = jax.jit(stress_H)
    # influence matrix: every obs (axis0) x every src tri (axis0) -> (Nobs,Msrc,3,3)
    disp_matrix = jax.jit(jax.vmap(jax.vmap(disp_U, in_axes=(None, 0, 0, 0, None, None, None)),
                                   in_axes=(0, None, None, None, None, None, None)))
    stress_matrix = jax.jit(jax.vmap(jax.vmap(stress_H, in_axes=(None, 0, 0, 0, None, None, None)),
                                     in_axes=(0, None, None, None, None, None, None)))
    return {"disp_U": disp_U_j, "stress_H": stress_H_j,
            "disp_matrix": disp_matrix, "stress_matrix": stress_matrix,
            "n_quad": n_quad}


# ====================================================================
# Self-test of the direct part vs the numpy reference
# ====================================================================
def _test_direct():
    from . import analytical_kernels as AK
    rng = np.random.default_rng(0)
    mu, nu = 1.0, 0.25
    worst_disp = worst_str = worst_mom = 0.0
    for _ in range(8):
        v = rng.uniform(-1, 1, (3, 3)); v[:, 2] -= 2.5
        v1, v2, v3 = v
        normal = np.cross(v2 - v1, v3 - v1); normal /= np.linalg.norm(normal)
        obs = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1.5, 0)])
        eps = 0.2
        a = [jnp.asarray(x) for x in (obs, v1, v2, v3)]
        # moments (I7, a tensor)
        mj = integrate_all_moments(a[1], a[2], a[3], a[0], eps)
        mn = AK.integrate_all_moments(v1, v2, v3, obs, eps)
        worst_mom = max(worst_mom, abs(float(mj['I'][7]) - mn['I'][7]),
                        float(np.max(np.abs(np.asarray(mj['T4'][7]) - mn['T4'][7]))))
        Uj = np.asarray(analytical_dd_displacement(a[0], a[1], a[2], a[3],
                                                   jnp.asarray(normal), mu, nu, eps))
        Un = AK.analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps)
        worst_disp = max(worst_disp, np.max(np.abs(Uj - Un)))
        Hj = np.asarray(analytical_stress_kernel(a[0], a[1], a[2], a[3],
                                                 jnp.asarray(normal), mu, nu, eps))
        Hn = AK.analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps)
        worst_str = max(worst_str, np.max(np.abs(Hj - Hn)))
    print(f"  direct moments  vs numpy: {worst_mom:.2e}")
    print(f"  direct DD disp  vs numpy: {worst_disp:.2e}")
    print(f"  direct stress   vs numpy: {worst_str:.2e}")
    return max(worst_disp, worst_str, worst_mom)



"""Obs-dependent GRADED quadrature for the smooth exact-BC complementary
correction over a source triangle.

WHY: for a SURFACE-BREAKING fault the correction integrand has a near-singular
2D structure -- it peaks at the source point closest (in the image sense) to the
observation, with width ~eps in BOTH the depth (src_z -> 0, image depth
a=sqrt(c^2+eps^2) -> eps) and in-plane (src_horizontal -> obs_horizontal)
directions.  A fixed tensor-Gauss rule on the triangle cannot resolve this at
small eps (it stalls).  This module clusters quadrature toward that 2D peak:

  * depth:    substitution c = -eps*sinh(t) (so a=eps*cosh(t) is smooth and the
              eps-scale region near src_z=0 is spread out), Gauss in t.
  * in-plane: along each constant-depth segment, sinh-clustering toward the foot
              of the observation, with width = the image distance at that depth.

It returns PHYSICAL quadrature points ys[Q,3] and weights wq[Q] with
sum_q wq[q] = triangle area, so  INT_T f dA ~ sum_q wq[q] f(ys[q]).  Validated to
~1e-16 against nested adaptive quadrature (see mh/proto_decompose.py) and used as
the enabler so each eps on a Richardson/Romberg ladder is quadrature-converged
at small eps.

The triangle is split at its middle-z vertex into <=2 'horizontal-edge'
sub-triangles (apex + a constant-depth edge), which makes the depth slicing exact
with a factorized area Jacobian.
"""

import numpy as np


def split_horizontal(v1, v2, v3, tiny=1e-12):
    """Split a triangle at the middle-z vertex into <=2 horizontal-edge
    sub-triangles (apex vp, constant-depth edge verts vq, vr at z_h).
    Returns list of (vp, vq, vr, z_apex, z_h)."""
    va, vb, vc = sorted((np.asarray(v1, float), np.asarray(v2, float),
                         np.asarray(v3, float)), key=lambda v: v[2])
    za, zb, zc = va[2], vb[2], vc[2]
    if zc - za < 1e-14:
        return []                                       # degenerate horizontal tri
    f = (zb - za) / (zc - za)
    vd = va + f * (vc - va)                              # point on va->vc at depth zb
    subs = []
    if zb - za > tiny:
        subs.append((va, vb, vd, za, zb))               # lower: apex va (bottom)
    if zc - zb > tiny:
        subs.append((vc, vb, vd, zc, zb))               # upper: apex vc (top)
    return subs


def _flat_tensor_rule(v1, v2, v3, n_c, n_t):
    """Collapsed tensor-Gauss rule on the triangle (weights sum to area).
    Fallback for (near-)horizontal triangles, where the depth sinh
    substitution degenerates: all sources share one depth, the correction
    integrand is smooth there, and a plain tensor rule is accurate."""
    v1, v2, v3 = (np.asarray(v, float) for v in (v1, v2, v3))
    n = max(n_c, n_t)
    pts, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (pts + 1.0); w = 0.5 * w
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    ys, wq = [], []
    for i in range(n):
        for j in range(n):
            xi1 = pts[i]
            xi2 = pts[j] * (1.0 - pts[i])
            ys.append((1.0 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3)
            wq.append(w[i] * w[j] * (1.0 - pts[i]) * area2)
    return np.array(ys), np.array(wq)


def _point_tri_distance(p, a, b, c):
    """Euclidean distance from point p to triangle (a, b, c)."""
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n) + 1e-300
    perp = (p - a) @ n / nn
    proj = p - perp * n / nn
    v0, v1, v2 = b - a, c - a, proj - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    den = d00 * d11 - d01 * d01 + 1e-300
    v = (d11 * (v2 @ v0) - d01 * (v2 @ v1)) / den
    w = (d00 * (v2 @ v1) - d01 * (v2 @ v0)) / den
    if v >= 0 and w >= 0 and v + w <= 1:
        return abs(perp)
    d = np.inf
    for (e0, e1) in ((a, b), (b, c), (c, a)):
        t = np.clip((p - e0) @ (e1 - e0) / ((e1 - e0) @ (e1 - e0) + 1e-300), 0, 1)
        d = min(d, np.linalg.norm(p - (e0 + t * (e1 - e0))))
    return d


def graded_quad(obs, v1, v2, v3, eps, n_c=14, n_t=12):
    """Physical points ys[Q,3] and weights wq[Q] (sum = triangle area) for the
    obs-dependent sinh2D graded rule.  n_c depth nodes x n_t in-plane nodes per
    sub-triangle.  (Near-)horizontal triangles, where the depth split
    degenerates, fall back to a collapsed tensor rule so the weight-sum
    contract always holds.

    The depth integrand has TWO scales: eps (the exact-BC coefficients vary
    through a(c)=sqrt(c^2+eps^2)) and the obs image distance d (the kernel's
    geometric near-singularity).  The depth rule is therefore a COMPOSITE
    two-panel Gauss rule in t (c = -eps*sinh t), split at the d-octave
    t = arcsinh(d/eps) — a single panel needed n_c ~ 50 to converge for obs
    ~100 m from a surface-breaking trace at eps ~ 1 m (the 2026-06-11
    animation artifact); the composite converges at ordinary n_c."""
    if eps <= 0.0:
        raise ValueError(
            f"graded quadrature requires eps > 0 (the depth substitution is "
            f"c = -eps*sinh(t)); got eps={eps!r}.  Use the tensor rule "
            f"(graded=False) for eps = 0.")
    obs = np.asarray(obs, float)
    gt, gw = np.polynomial.legendre.leggauss(max(2, n_c // 2))   # per panel
    gs, gws = np.polynomial.legendre.leggauss(max(2, n_t // 2))  # per panel
    ys, wq = [], []
    for (vp, vq, vr, z_apex, z_h) in split_horizontal(v1, v2, v3):
        twoA = np.linalg.norm(np.cross(vq - vp, vr - vq))
        dz = abs(z_h - z_apex)
        c_lo, c_hi = min(z_apex, z_h), max(z_apex, z_h)  # both <= 0
        t_lo = np.arcsinh(-c_hi / eps)                    # smaller |c| -> smaller t
        t_hi = np.arcsinh(-c_lo / eps)
        d_obs = _point_tri_distance(obs, vp, vq, vr)
        span = t_hi - t_lo
        t_mid = np.clip(np.arcsinh(d_obs / eps),
                        t_lo + 0.05 * span, t_hi - 0.05 * span)
        tc = np.concatenate([0.5 * (t_mid - t_lo) * gt + 0.5 * (t_mid + t_lo),
                             0.5 * (t_hi - t_mid) * gt + 0.5 * (t_hi + t_mid)])
        wtc = np.concatenate([0.5 * (t_mid - t_lo) * gw,
                              0.5 * (t_hi - t_mid) * gw])
        cc = -eps * np.sinh(tc)                           # depth nodes
        dcw = eps * np.cosh(tc) * wtc                     # |dc| weights
        for c, dcwq in zip(cc, dcw):
            f = (c - z_apex) / (z_h - z_apex)             # 0 at apex -> 1 at edge
            P1 = vp + f * (vq - vp)
            seg = f * (vr - vq)                           # segment vector P2-P1
            L = np.linalg.norm(seg)
            if L < 1e-14:
                continue
            base_w = (twoA / dz) * f * dcwq               # depth weight
            that = seg / L
            tau_foot = np.dot(obs - P1, that) / L         # foot param along segment
            a = np.sqrt(c * c + eps * eps)
            perp = obs - (P1 + tau_foot * seg)
            d_perp = np.linalg.norm(perp[:2])             # horiz perp distance
            w_scale = max(np.sqrt(d_perp ** 2 + (a - obs[2]) ** 2), eps) / L
            s_lo = np.arcsinh((0.0 - tau_foot) / w_scale)
            s_hi = np.arcsinh((1.0 - tau_foot) / w_scale)
            # composite two-panel rule split AT the peak s=0 (the obs foot):
            # plain Gauss-on-sinh puts its sparse CENTER on the peak when the
            # span is large (w_scale << 1), needing n_t ~ 50 to converge.
            sspan = s_hi - s_lo
            s_mid = np.clip(0.0, s_lo + 0.05 * sspan, s_hi - 0.05 * sspan)
            s = np.concatenate([0.5 * (s_mid - s_lo) * gs + 0.5 * (s_mid + s_lo),
                                0.5 * (s_hi - s_mid) * gs + 0.5 * (s_hi + s_mid)])
            ws = np.concatenate([0.5 * (s_mid - s_lo) * gws,
                                 0.5 * (s_hi - s_mid) * gws])
            tau = tau_foot + w_scale * np.sinh(s)
            dtau = w_scale * np.cosh(s) * ws
            for tt, wtt in zip(tau, dtau):
                ys.append(P1 + tt * seg)
                wq.append(base_w * wtt)
    area = 0.5 * np.linalg.norm(np.cross(np.asarray(v2, float) - v1,
                                         np.asarray(v3, float) - v1))
    if area <= 0.0:
        return np.zeros((0, 3)), np.zeros(0)                # true zero-area tri
    if not ys or abs(np.sum(wq) - area) > 1e-8 * area:
        # Horizontal / thin-in-z triangle: the depth split dropped weight.
        return _flat_tensor_rule(v1, v2, v3, n_c, n_t)
    return np.array(ys), np.array(wq)


# ---------------------------------------------------------------------------
def _selftest():
    """(1) exact area reproduction; (2) CONVERGENCE to a dense tensor reference for
    a smooth integrand (the graded rule is unbiased -> converges as n grows; the
    near-singular CORRECTION accuracy is validated separately vs nested adaptive
    quad in mh/proto_decompose.py)."""
    v1 = np.array([-5.0, 0.0, 0.0]); v2 = np.array([5.0, 0.0, 0.0])
    v3 = np.array([5.0, 0.0, -10.0])
    obs = np.array([0.0, 2.0, 0.0])
    area_true = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    # dense tensor reference for the smooth test integrand f=1/(|obs-y|^2+1)
    n = 140
    pts, w = np.polynomial.legendre.leggauss(n); pts = 0.5 * (pts + 1); w = 0.5 * w
    a2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    I_ref = 0.0
    for i in range(n):
        for j in range(n):
            xi1 = pts[i]; xi2 = pts[j] * (1 - pts[i])
            ww = w[i] * w[j] * (1 - pts[i]) * a2
            y = (1 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3
            I_ref += ww / (np.sum((obs - y) ** 2) + 1.0)

    ok = True
    for eps in [0.2, 0.05, 0.0125]:
        # area exact at modest n
        ys, wq = graded_quad(obs, v1, v2, v3, eps, 16, 14)
        da = abs(wq.sum() - area_true)
        # convergence of the smooth integrand as n grows
        errs = []
        for nq in (12, 24, 48):
            ys, wq = graded_quad(obs, v1, v2, v3, eps, nq, nq)
            I = np.sum(wq / (np.sum((obs - ys) ** 2, axis=1) + 1.0))
            errs.append(abs(I - I_ref))
        converged = errs[-1] < 1e-9 and errs[-1] < errs[0]
        good = da < 1e-10 and converged
        ok &= good
        print(f"  eps={eps:7.4f}: area err={da:.1e}; smooth-integrand err "
              f"n12->24->48: {errs[0]:.1e} {errs[1]:.1e} {errs[2]:.1e} "
              f"{'OK' if good else 'FAIL'}")
    print("  %s" % ("ALL PASS" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    print("graded_quad self-test (exact area + convergence vs dense tensor):")
    _selftest()

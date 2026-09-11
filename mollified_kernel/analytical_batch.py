"""Vectorized analytical integration of mollified Kelvin kernels over
triangular elements.

The scalar per-(obs, tri) routines in ``analytical_kernels`` are correct but
too slow (~3 ms/pair) for BEM assembly of thousands of elements. This module
computes, for a single source triangle and an array of N observation points,
the analytical moments and kernels in vectorised form with NumPy. Per-triangle
setup (edges, in-plane basis, edge antiderivatives) is scalar; obs-dependent
quantities (z, h, obs_proj, projected vertices, edge integrals, solid angle,
2D moments, 3D tensor moments, and the final kernel) are NumPy arrays of
leading shape ``(N_obs, ...)``.

Two public entry points wrap this into full BEM matrix assemblies:

* :func:`assemble_U_matrix_batch` — Kelvin displacement-from-force kernel
* :func:`assemble_T_matrix_batch` — displacement-from-slip (DD) kernel

Both return ``(3 N_field, 3 N_source)`` matrices matching the legacy
``assemble_BEM_matrices`` layout.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Scalar antiderivatives extended to arrays
# ---------------------------------------------------------------------------

def _antideriv_J_arr(u: np.ndarray, rho2: np.ndarray, order: int) -> np.ndarray:
    """Antiderivative of 1/(u²+ρ²)^(order/2) w.r.t. u, vectorised over obs.

    ``u`` and ``rho2`` have the same shape (typically (N_obs,) or broadcast).
    Returns an array of the same shape.
    """
    R = np.sqrt(u * u + rho2)
    tiny_rho = rho2 < 1e-60
    if order == 1:
        return np.log(u + R + 1e-300)
    if order == 3:
        out = np.where(tiny_rho, 0.0, u / (rho2 * R))
        return out
    if order == 5:
        out = np.where(
            tiny_rho,
            0.0,
            u * (3.0 * rho2 + 2.0 * u * u) / (3.0 * rho2 * rho2 * R ** 3),
        )
        return out
    if order == 7:
        out = np.where(
            tiny_rho,
            0.0,
            u * (15 * rho2 ** 2 + 20 * rho2 * u * u + 8 * u ** 4)
            / (15 * rho2 ** 3 * R ** 5),
        )
        return out
    raise ValueError(f"J order {order} not implemented")


def _antideriv_K_arr(u: np.ndarray, rho2: np.ndarray, order: int) -> np.ndarray:
    """Antiderivative of u/(u²+ρ²)^(order/2) w.r.t. u, vectorised."""
    R = np.sqrt(u * u + rho2)
    if order == 1:
        return R
    if order == 3:
        return -1.0 / np.maximum(R, 1e-30)
    if order == 5:
        return -1.0 / (3.0 * np.maximum(R, 1e-30) ** 3)
    if order == 7:
        return -1.0 / (5.0 * np.maximum(R, 1e-30) ** 5)
    raise ValueError(f"K order {order} not implemented")


# ---------------------------------------------------------------------------
# Solid angle van Oosterom, batched over observation points
# ---------------------------------------------------------------------------

def _solid_angle_batch(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                        obs: np.ndarray) -> np.ndarray:
    """Solid angle subtended by (v1,v2,v3) from each obs (N,3). Returns (N,)."""
    r1 = v1[None, :] - obs
    r2 = v2[None, :] - obs
    r3 = v3[None, :] - obs
    R1 = np.linalg.norm(r1, axis=1)
    R2 = np.linalg.norm(r2, axis=1)
    R3 = np.linalg.norm(r3, axis=1)
    numer = np.einsum("ni,ni->n", r1, np.cross(r2, r3))
    denom = (
        R1 * R2 * R3
        + R3 * np.einsum("ni,ni->n", r1, r2)
        + R1 * np.einsum("ni,ni->n", r2, r3)
        + R2 * np.einsum("ni,ni->n", r1, r3)
    )
    return 2.0 * np.arctan2(numer, denom)


# ---------------------------------------------------------------------------
# Per-triangle moment solver, vectorized over observation points
# ---------------------------------------------------------------------------

def _edge_u_values(pa_x: np.ndarray, pa_y: np.ndarray,
                    pb_x: np.ndarray, pb_y: np.ndarray,
                    t_hat: np.ndarray, n_out: np.ndarray):
    """Edge-local projections of pa, pb onto (t̂, n̂) (all shape (N,))."""
    u_a = pa_x * t_hat[0] + pa_y * t_hat[1]
    u_b = pb_x * t_hat[0] + pb_y * t_hat[1]
    dp = pa_x * n_out[0] + pa_y * n_out[1]
    return u_a, u_b, dp


def integrate_moments_batch(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                             obs: np.ndarray, eps: float,
                             need_orders: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Batched analytical moments for one source triangle and N_obs obs.

    Returns a dict with keys
        ``I``   : {n: (N,) array}  ∫∫ 1/R^n dA
        ``V``   : {n: (N,3) array} ∫∫ d_i/R^n dA
        ``T2``  : {n: (N,3,3) array} ∫∫ d_i d_j/R^n dA
        ``T3``  : {n: (N,3,3,3) array} ∫∫ d_i d_j d_k/R^n dA
        ``ex``, ``ey``, ``nhat`` : triangle in-plane basis, each (3,)
        ``z``, ``h``             : per-obs arrays of shape (N,)

    ``need_orders`` lists which R^n exponents the caller will consume so we
    can skip expensive branches. Orders available: 1 (I only), 3 (V, T2),
    5 (T3). T3 at order 5 is needed for the DD displacement kernel.
    """
    obs = np.asarray(obs, dtype=float)
    if obs.ndim == 1:
        obs = obs[None, :]
    N = obs.shape[0]

    e1 = v2 - v1
    e2 = v3 - v1
    normal = np.cross(e1, e2)
    area2 = np.linalg.norm(normal)
    if area2 < 1e-30:
        # Degenerate triangle: zero integrals
        zeros_I = np.zeros(N)
        zeros_V = np.zeros((N, 3))
        zeros_T2 = np.zeros((N, 3, 3))
        zeros_T3 = np.zeros((N, 3, 3, 3))
        return {
            "I": {n: zeros_I.copy() for n in (1, 3, 5, 7)},
            "V": {n: zeros_V.copy() for n in (3, 5, 7)},
            "T2": {n: zeros_T2.copy() for n in (3, 5, 7)},
            "T3": {n: zeros_T3.copy() for n in (5, 7)},
            "ex": np.array([1.0, 0.0, 0.0]),
            "ey": np.array([0.0, 1.0, 0.0]),
            "nhat": np.array([0.0, 0.0, 1.0]),
            "z": np.zeros(N),
            "h": np.full(N, float(eps)),
        }

    normal = normal / area2

    # Per-obs: z, h, obs_proj
    z = (obs - v1[None, :]) @ normal            # (N,)
    h2 = z * z + eps * eps                      # (N,)
    h = np.sqrt(h2)                             # (N,)
    obs_proj = obs - z[:, None] * normal[None, :]  # (N,3)

    # In-plane basis (shared across obs)
    ex = e1 / np.linalg.norm(e1)
    ey = np.cross(normal, ex)

    # Project triangle vertices into in-plane 2D relative to obs_proj
    def to_2d(p):
        d = p[None, :] - obs_proj        # (N,3)
        return d @ ex, d @ ey             # two (N,) arrays
    p1x, p1y = to_2d(v1)
    p2x, p2y = to_2d(v2)
    p3x, p3y = to_2d(v3)

    edges = [
        (p1x, p1y, p2x, p2y),
        (p2x, p2y, p3x, p3y),
        (p3x, p3y, p1x, p1y),
    ]

    # Solid-angle-based I3 (per obs)
    obs_eff = obs_proj + h[:, None] * normal[None, :]  # (N,3)
    Omega = _solid_angle_batch(v1, v2, v3, obs_eff)    # (N,)
    I3 = -Omega / h                                      # (N,)

    # Boundary integrals accumulators, arrays indexed by (a,b,m) scalar dict.
    # We will only compute BD/BN combos actually needed for moments up to
    # (a+b)=3 at m=5.  Order ≤ 1 at m=1 is enough for I1 and first-order
    # moments; higher orders at m=3,5 feed T2 and T3.
    BN1: dict[tuple[int, int, int], np.ndarray] = {}
    BN2: dict[tuple[int, int, int], np.ndarray] = {}
    BD: dict[tuple[int, int, int], np.ndarray] = {}

    # Precompute which (a,b,m) are needed
    need_abm: set[tuple[int, int, int]] = set()
    # base integrals: (0,0,m) for m=1,3,5 (always)
    for m in (1, 3, 5):
        need_abm.add((0, 0, m))
    if 3 in need_orders:
        # first-order moments at m=3,5 via BN at m=1,3
        for m in (1, 3):
            need_abm.add((0, 0, m))
        # second-order moments at m=3 via BN at m=1 with order 1
        for ab in ((1, 0), (0, 1)):
            need_abm.add((*ab, 1))
    if 5 in need_orders:
        # first-order moments at m=5,7 via BN at m=3,5
        for ab in ((0, 0),):
            need_abm.update({(*ab, m) for m in (3, 5)})
        # second-order moments at m=5 via BN at m=3 with order 1
        for ab in ((1, 0), (0, 1)):
            need_abm.add((*ab, 3))
        # third-order moments at m=5 via BN at m=3 with order 2
        for ab in ((2, 0), (1, 1), (0, 2)):
            need_abm.add((*ab, 3))

    for pa_x, pa_y, pb_x, pb_y in edges:
        edge_vec_x = pb_x - pa_x
        edge_vec_y = pb_y - pa_y
        L = np.sqrt(edge_vec_x * edge_vec_x + edge_vec_y * edge_vec_y)
        valid = L > 1e-30
        L_safe = np.where(valid, L, 1.0)
        t_hat_x = edge_vec_x / L_safe
        t_hat_y = edge_vec_y / L_safe
        n_out_x = t_hat_y
        n_out_y = -t_hat_x

        dp = pa_x * n_out_x + pa_y * n_out_y
        u_a = pa_x * t_hat_x + pa_y * t_hat_y
        u_b = pb_x * t_hat_x + pb_y * t_hat_y
        rho2 = dp * dp + h2

        # 1D antiderivatives at u_a, u_b
        dJ = {}
        dK = {}
        for m in (1, 3, 5):
            dJ[m] = _antideriv_J_arr(u_b, rho2, m) - _antideriv_J_arr(u_a, rho2, m)
            dK[m] = _antideriv_K_arr(u_b, rho2, m) - _antideriv_K_arr(u_a, rho2, m)
        # Derived antiderivatives: ∫u²/R^m = dJ[m-2] - ρ² dJ[m]
        dL = {m: dJ[m - 2] - rho2 * dJ[m] for m in (3, 5)}
        # ∫u³/R^m = dK[m-2] - ρ² dK[m]
        dM = {m: dK[m - 2] - rho2 * dK[m] for m in (3, 5)}

        def edge_integral(a: int, b: int, m: int) -> np.ndarray:
            # Expand (c₁ u + s₁ dp?)... we mirror the scalar derivation
            # exactly but with NumPy arrays.
            # The integrand is ξ₁^a ξ₂^b / R^m du, expanded in (u, dp):
            #   ξ₁ = c₁ dp + s₁ u,   ξ₂ = c₂ dp + s₂ u
            # with c₁=n_out_x, c₂=n_out_y, s₁=t_hat_x, s₂=t_hat_y.
            total = np.zeros(N)
            for i in range(a + 1):
                for j in range(b + 1):
                    coeff = (
                        _comb(a, i) * _comb(b, j)
                        * dp ** (a + b - i - j)
                        * n_out_x ** (a - i) * t_hat_x ** i
                        * n_out_y ** (b - j) * t_hat_y ** j
                    )
                    k = i + j
                    if k == 0:
                        delta = dJ[m]
                    elif k == 1:
                        delta = dK[m]
                    elif k == 2:
                        delta = dL[m]
                    elif k == 3:
                        delta = dM[m]
                    else:
                        raise ValueError(f"u^{k} not supported")
                    total = total + coeff * delta
            return total

        for a, b, m in need_abm:
            val = edge_integral(a, b, m)
            val = np.where(valid, val, 0.0)
            BN1[(a, b, m)] = BN1.get((a, b, m), 0.0) + n_out_x * val
            BN2[(a, b, m)] = BN2.get((a, b, m), 0.0) + n_out_y * val
            BD[(a, b, m)] = BD.get((a, b, m), 0.0) + dp * val

    # ---- Base integrals ----
    E1 = BD.get((0, 0, 1), np.zeros(N))
    E3 = BD.get((0, 0, 3), np.zeros(N))
    E5 = BD.get((0, 0, 5), np.zeros(N))

    I1 = E1 - h2 * I3
    # Guard against h→0 (only relevant when obs lies on the triangle plane
    # with ε→0 simultaneously — mollification keeps h ≥ ε).
    I5 = (E3 + I3) / (3.0 * h2)
    I7 = (E5 + 3.0 * I5) / (5.0 * h2)

    M: dict[tuple[int, int, int], np.ndarray] = {
        (0, 0, 1): I1,
        (0, 0, 3): I3,
        (0, 0, 5): I5,
        (0, 0, 7): I7,
    }

    # ---- First-order 2D moments at n=3,5,7 ----
    for n in (3, 5, 7):
        m = n - 2
        if m not in (1, 3, 5):
            continue
        c = -1.0 / (n - 2)
        M[(1, 0, n)] = c * BN1.get((0, 0, m), np.zeros(N))
        M[(0, 1, n)] = c * BN2.get((0, 0, m), np.zeros(N))

    # ---- Second-order 2D moments at n=3,5,7 ----
    for n in (3, 5, 7):
        m = n - 2
        c = 1.0 / (n - 2)
        M[(2, 0, n)] = c * (M.get((0, 0, m), np.zeros(N)) - BN1.get((1, 0, m), np.zeros(N)))
        M[(1, 1, n)] = -c * BN1.get((0, 1, m), np.zeros(N))
        M[(0, 2, n)] = c * (M.get((0, 0, m), np.zeros(N)) - BN2.get((0, 1, m), np.zeros(N)))

    # ---- Third-order 2D moments at n=5,7 (needed for T3[5]) ----
    for n in (5, 7):
        m = n - 2
        c = 1.0 / (n - 2)
        if (2, 0, m) in BN1:
            M[(3, 0, n)] = c * (2.0 * M[(1, 0, m)] - BN1[(2, 0, m)])
        if (1, 1, m) in BN1:
            M[(2, 1, n)] = c * (M[(0, 1, m)] - BN1[(1, 1, m)])
        if (0, 2, m) in BN1:
            M[(1, 2, n)] = -c * BN1[(0, 2, m)]
        if (0, 2, m) in BN2:
            M[(0, 3, n)] = c * (2.0 * M[(0, 1, m)] - BN2[(0, 2, m)])

    # ------------------------------------------------------------------
    # Transform 2D moments to 3D tensor moments
    # d_i = -ξ₁ ex_i - ξ₂ ey_i + z n̂_i
    # ------------------------------------------------------------------
    V = {}
    for n in (3, 5, 7):
        m10 = M.get((1, 0, n), np.zeros(N))     # (N,)
        m01 = M.get((0, 1, n), np.zeros(N))     # (N,)
        m00 = M.get((0, 0, n), np.zeros(N))     # (N,)
        Vn = (
            -ex[None, :] * m10[:, None]
            - ey[None, :] * m01[:, None]
            + normal[None, :] * (z * m00)[:, None]
        )
        V[n] = Vn  # (N,3)

    T2 = {}
    for n in (3, 5, 7):
        m20 = M.get((2, 0, n), np.zeros(N))
        m11 = M.get((1, 1, n), np.zeros(N))
        m02 = M.get((0, 2, n), np.zeros(N))
        m10 = M.get((1, 0, n), np.zeros(N))
        m01 = M.get((0, 1, n), np.zeros(N))
        m00 = M.get((0, 0, n), np.zeros(N))
        # Expand d_i d_j over {ex, ey, n̂} basis
        # d_i = -ξ₁ ex_i - ξ₂ ey_i + z n̂_i
        E_ex = ex[None, :] * ex[None, :][:, :, None].swapaxes(0, 1)  # not used directly
        # Compute T2_ij = Σ over basis vector products
        ex_e = ex[None, :]       # (1,3)
        ey_e = ey[None, :]
        n_e = normal[None, :]
        # ξ₁² coefficient: ex_i ex_j  times M[2,0,n]
        t20 = ex_e[:, :, None] * ex_e[:, None, :] * m20[:, None, None]   # broadcasts to (N,3,3)
        t11 = (ex_e[:, :, None] * ey_e[:, None, :]
               + ey_e[:, :, None] * ex_e[:, None, :]) * m11[:, None, None]
        t02 = ey_e[:, :, None] * ey_e[:, None, :] * m02[:, None, None]
        # -z * (ex_i n̂_j + n̂_i ex_j) * M[1,0,n]
        zv = z[:, None, None]
        t10 = -zv * (ex_e[:, :, None] * n_e[:, None, :]
                     + n_e[:, :, None] * ex_e[:, None, :]) * m10[:, None, None]
        t01 = -zv * (ey_e[:, :, None] * n_e[:, None, :]
                     + n_e[:, :, None] * ey_e[:, None, :]) * m01[:, None, None]
        # z² n̂_i n̂_j * M[0,0,n]
        t00 = (z ** 2)[:, None, None] * (n_e[:, :, None] * n_e[:, None, :]) * m00[:, None, None]
        T2[n] = t20 + t11 + t02 + t10 + t01 + t00

    # T3 at n=5,7 — needed for DD displacement kernel (uses T3[5])
    T3: dict[int, np.ndarray] = {}
    if 5 in need_orders or 7 in need_orders:
        components = [(-1.0, ex, "xi1"), (-1.0, ey, "xi2"), ("z", normal, "z")]
        orders_needed = tuple(n for n in (5, 7) if n in need_orders)
        for n in orders_needed:
            Tn = np.zeros((N, 3, 3, 3))
            for si, bi, li in components:
                for sj, bj, lj in components:
                    for sk, bk, lk in components:
                        a = sum(1 for l in (li, lj, lk) if l == "xi1")
                        b = sum(1 for l in (li, lj, lk) if l == "xi2")
                        # coeff: product of (s or z) over component.
                        # For ξ components, factor is s (= -1). For z component,
                        # factor is z (the per-obs scalar).
                        sign = 1.0
                        z_power = 1
                        for s, l in ((si, li), (sj, lj), (sk, lk)):
                            if l == "z":
                                z_power += 1  # actual z multiplication below
                            else:
                                sign *= s
                        # number of z factors = z_power - 1 (we init at 1)
                        n_z = z_power - 1
                        moment = M.get((a, b, n), None)
                        if moment is None:
                            continue
                        coeff_per_obs = sign * (z ** n_z)   # (N,) if n_z>0 else scalar
                        term = np.einsum(
                            "i,j,k,n->nijk",
                            bi, bj, bk,
                            coeff_per_obs * moment if n_z > 0 else (sign * moment),
                        )
                        Tn = Tn + term
            T3[n] = Tn

    return {
        "I": {1: I1, 3: I3, 5: I5, 7: I7},
        "V": V,
        "T2": T2,
        "T3": T3,
        "ex": ex,
        "ey": ey,
        "nhat": normal,
        "z": z,
        "h": h,
    }


def _comb(n: int, k: int) -> int:
    """Small binomial coefficient (n ≤ 4 here)."""
    from math import comb
    return comb(n, k)


# ---------------------------------------------------------------------------
# Kernel matrices
# ---------------------------------------------------------------------------

def kelvin_G_batch(v1, v2, v3, obs: np.ndarray, mu: float, nu: float,
                    eps: float) -> np.ndarray:
    """Integrated Kelvin displacement Green function G over one triangle.

    Uses the Galerkin/Cortez-convolved form
        G_ij = C1 [(3-4ν) δ_ij/R + d_i d_j/R^3 + 2(1-ν) ε² δ_ij/R^3].

    Returns G of shape ``(N_obs, 3, 3)`` such that
    ``u_i(obs) = G[obs, i, j] * f_j`` where ``f`` is the constant force
    per unit area applied over the triangle.
    """
    mom = integrate_moments_batch(v1, v2, v3, obs, eps, need_orders=(1, 3))
    I1 = mom["I"][1]
    I3 = mom["I"][3]
    T23 = mom["T2"][3]
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps * eps
    eye3 = np.eye(3)
    diag_coeff = c34 * I1 + c_blob * I3       # (N,)
    return C1 * (diag_coeff[:, None, None] * eye3[None, :, :] + T23)


def dd_displacement_batch(v1, v2, v3, normal: np.ndarray, obs: np.ndarray,
                            mu: float, nu: float, eps: float) -> np.ndarray:
    """Integrated displacement-discontinuity displacement kernel.

    Uses the Galerkin/Cortez-convolved gradient
        ∂G_ik/∂x_m = C1 [-c34 δ_ik d_m/R^3 + δ_im d_k/R^3 + δ_km d_i/R^3
                          - 3 d_i d_k d_m/R^5 - 6(1-ν) ε² δ_ik d_m/R^5].

    Returns U of shape ``(N_obs, 3, 3)`` such that ``u_i(obs) = U[obs, i, j] * Δu_j``.
    """
    mom = integrate_moments_batch(v1, v2, v3, obs, eps, need_orders=(3, 5))
    V3 = mom["V"][3]       # (N,3)
    V5 = mom["V"][5]       # (N,3)
    T35 = mom["T3"][5]     # (N,3,3,3)

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob_dg = 6.0 * (1.0 - nu) * eps * eps
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

    # G1[obs, i, k, m] = C1 [ -c34 δ_ik V3[obs,m] + δ_im V3[obs,k]
    #                          + δ_km V3[obs,i] - 3 T3_5[obs,i,k,m]
    #                          - 6(1-ν) ε² δ_ik V5[obs,m] ]
    eye3 = np.eye(3)
    G1 = np.zeros((obs.shape[0], 3, 3, 3))
    G1 += -c34 * np.einsum("ik,nm->nikm", eye3, V3)
    G1 += np.einsum("im,nk->nikm", eye3, V3)
    G1 += np.einsum("km,ni->nikm", eye3, V3)
    G1 += -3.0 * T35
    G1 += -c_blob_dg * np.einsum("ik,nm->nikm", eye3, V5)
    G1 *= C1

    n = normal
    # U[i, j] = -[ lam * Σ_m n_m G1[i,j,m] + mu * n_j * trace_im
    #             + mu * Σ_k n_k G1[i,k,j] ]
    # trace_im = Σ_m G1[i,m,m]
    trace_im = np.einsum("nimm->ni", G1)
    term1 = lam * np.einsum("m,nijm->nij", n, G1)
    term2 = mu * np.einsum("j,ni->nij", n, trace_im)
    term3 = mu * np.einsum("k,nikj->nij", n, G1)
    U = -(term1 + term2 + term3)
    return U


# ---------------------------------------------------------------------------
# Full BEM-matrix assemblies
# ---------------------------------------------------------------------------

def _place_blocks(block_list_s: np.ndarray) -> np.ndarray:
    """Vectorized reshape of (N_s, N_f, 3, 3) into (3 N_f, 3 N_s) BEM layout.

    ``block_list_s[s, i, :, :]`` is the 3×3 kernel for (field i, source s).
    The output places row block i (rows 3i..3i+3) and column block s (cols
    3s..3s+3) with that 3×3 matrix, matching the legacy layout used by
    ``assemble_BEM_matrices``.
    """
    N_s, N_f = block_list_s.shape[:2]
    # (N_s, N_f, 3, 3) -> (N_f, 3, N_s, 3) -> (3 N_f, 3 N_s)
    return (
        block_list_s.transpose(1, 2, 0, 3)   # (N_f, 3, N_s, 3)
        .reshape(3 * N_f, 3 * N_s)
    )


def assemble_U_matrix_batch(x_field: np.ndarray, tri_verts: np.ndarray,
                              mu: float, nu: float, eps: float) -> np.ndarray:
    """Assemble (3 N_f, 3 N_s) matrix for kernel="U" using analytical integ.

    ``tri_verts`` has shape ``(N_s, 3, 3)`` where [s,k,:] is vertex k of
    source triangle s.
    """
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    blocks = np.empty((N_s, N_f, 3, 3))
    for s in range(N_s):
        v1, v2, v3 = tri_verts[s, 0], tri_verts[s, 1], tri_verts[s, 2]
        blocks[s] = kelvin_G_batch(v1, v2, v3, x_field, mu, nu, eps)
    return _place_blocks(blocks)


def assemble_T_matrix_batch(x_field: np.ndarray, tri_verts: np.ndarray,
                              normals_source: np.ndarray,
                              mu: float, nu: float, eps: float) -> np.ndarray:
    """Assemble (3 N_f, 3 N_s) matrix for kernel="T"."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    blocks = np.empty((N_s, N_f, 3, 3))
    for s in range(N_s):
        v1, v2, v3 = tri_verts[s, 0], tri_verts[s, 1], tri_verts[s, 2]
        blocks[s] = dd_displacement_batch(v1, v2, v3, normals_source[s],
                                           x_field, mu, nu, eps)
    return _place_blocks(blocks)

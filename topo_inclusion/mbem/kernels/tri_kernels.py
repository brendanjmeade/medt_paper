"""Numba pair-level analytical integration of mollified Kelvin kernels.

Scalar port of ``mollified_kernel/analytical_batch.py`` restructured to
emit GEOMETRY-ONLY basis blocks (material coefficients are applied later
by :mod:`mbem.kernels.basis`):

U (Kelvin force->displacement) kernel over a source triangle::

    G(mu, nu) = g1*B1 + g2*B2 + g3*B3
    B1_ij = I1 * delta_ij            (I1 = integral of 1/R_eps dA)
    B2_ij = eps^2 * I3 * delta_ij    (I3 = integral of 1/R_eps^3 dA)
    B3_ij = T2[3]_ij                 (integral of d_i d_j / R_eps^3 dA)

T (slip->displacement, DD) kernel::

    U(mu, lam) = c1*L[P1] + c2*L[P2] + c3*L[P3]
               + c4*M[P1] + c5*M[P2] + c6*M[P3]
    P1_ikm = delta_ik V3_m
    P2_ikm = delta_im V3_k + delta_km V3_i - 3*T3[5]_ikm
    P3_ikm = eps^2 * delta_ik V5_m
    L[P]_ij = sum_m n_m P_ijm
    M[P]_ij = n_j * sum_m P_imm + sum_k n_k P_ikj

where V3/V5 are first moments of d/R^3, d/R^5 and T3[5] the third moment
of d d d / R^5, all integrated analytically over the source triangle
with the Cortez mollification baked into R_eps = sqrt(r^2 + eps^2).
eps is PER SOURCE ELEMENT (an (N_src,) array at the assembly level).

The numerical guards (1e-300 in the log, 1e-30/1e-60 floors, degenerate
triangle/edge skips) replicate analytical_batch.py exactly so the two
paths agree to machine precision.

Matrix layout matches the legacy ``assemble_BEM_matrices``: row block
3*f..3*f+3 = field element f, column block 3*s..3*s+3 = source element s.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------
# Edge antiderivatives (orders 1 and 3 are all the U/T kernels consume)
# ---------------------------------------------------------------------

@njit(cache=True, inline="always")
def _J1(u, rho2):
    return np.log(u + np.sqrt(u * u + rho2) + 1e-300)


@njit(cache=True, inline="always")
def _J3(u, rho2):
    if rho2 < 1e-60:
        return 0.0
    return u / (rho2 * np.sqrt(u * u + rho2))


@njit(cache=True, inline="always")
def _K1(u, rho2):
    return np.sqrt(u * u + rho2)


@njit(cache=True, inline="always")
def _K3(u, rho2):
    R = np.sqrt(u * u + rho2)
    if R < 1e-30:
        R = 1e-30
    return -1.0 / R


@njit(cache=True, inline="always")
def _solid_angle(v1, v2, v3, ox, oy, oz):
    """van Oosterom signed solid angle of triangle (v1,v2,v3) from obs."""
    r1x = v1[0] - ox; r1y = v1[1] - oy; r1z = v1[2] - oz
    r2x = v2[0] - ox; r2y = v2[1] - oy; r2z = v2[2] - oz
    r3x = v3[0] - ox; r3y = v3[1] - oy; r3z = v3[2] - oz
    R1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    R2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    R3 = np.sqrt(r3x * r3x + r3y * r3y + r3z * r3z)
    cx = r2y * r3z - r2z * r3y
    cy = r2z * r3x - r2x * r3z
    cz = r2x * r3y - r2y * r3x
    numer = r1x * cx + r1y * cy + r1z * cz
    d12 = r1x * r2x + r1y * r2y + r1z * r2z
    d23 = r2x * r3x + r2y * r3y + r2z * r3z
    d13 = r1x * r3x + r1y * r3y + r1z * r3z
    denom = R1 * R2 * R3 + R3 * d12 + R1 * d23 + R2 * d13
    return 2.0 * np.arctan2(numer, denom)


@njit(cache=True, inline="always")
def _edge_int(a, b, dp, nx, ny, tx, ty, dJ, dK, dL):
    """Edge integral of xi1^a xi2^b / R^m along one edge.

    Mirrors analytical_batch.edge_integral: the integrand is expanded in
    the edge-local frame xi1 = nx*dp + tx*u, xi2 = ny*dp + ty*u; dJ/dK/dL
    are the order-m antiderivative differences for u^0, u^1, u^2.
    a + b <= 2 here, so binomials are 1 or 2 and u-powers stop at 2.
    """
    total = 0.0
    for i in range(a + 1):
        for j in range(b + 1):
            ca = 1.0
            if a == 2 and i == 1:
                ca = 2.0
            cb = 1.0
            if b == 2 and j == 1:
                cb = 2.0
            coeff = (ca * cb
                     * dp ** (a + b - i - j)
                     * nx ** (a - i) * tx ** i
                     * ny ** (b - j) * ty ** j)
            k = i + j
            if k == 0:
                delta = dJ
            elif k == 1:
                delta = dK
            else:
                delta = dL
            total += coeff * delta
    return total


# ---------------------------------------------------------------------
# Per-source-triangle frame setup
# ---------------------------------------------------------------------

@njit(cache=True)
def _tri_frame(tv, ex, ey, nhat):
    """In-plane basis of source triangle tv (3,3). Returns False if degenerate."""
    e1x = tv[1, 0] - tv[0, 0]; e1y = tv[1, 1] - tv[0, 1]; e1z = tv[1, 2] - tv[0, 2]
    e2x = tv[2, 0] - tv[0, 0]; e2y = tv[2, 1] - tv[0, 1]; e2z = tv[2, 2] - tv[0, 2]
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    area2 = np.sqrt(nx * nx + ny * ny + nz * nz)
    if area2 < 1e-30:
        return False
    nhat[0] = nx / area2; nhat[1] = ny / area2; nhat[2] = nz / area2
    e1n = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    ex[0] = e1x / e1n; ex[1] = e1y / e1n; ex[2] = e1z / e1n
    ey[0] = nhat[1] * ex[2] - nhat[2] * ex[1]
    ey[1] = nhat[2] * ex[0] - nhat[0] * ex[2]
    ey[2] = nhat[0] * ex[1] - nhat[1] * ex[0]
    return True


# ---------------------------------------------------------------------
# U-kernel basis blocks for one (source triangle, obs) pair
# ---------------------------------------------------------------------

@njit(cache=True)
def _u_basis_pair(tv, ex, ey, nhat, obs, eps, out):
    """Write the 3 U-kernel basis 3x3 blocks into out (3,3,3): [b,i,j]."""
    # Per-obs frame quantities
    dx = obs[0] - tv[0, 0]; dy = obs[1] - tv[0, 1]; dz = obs[2] - tv[0, 2]
    z = dx * nhat[0] + dy * nhat[1] + dz * nhat[2]
    h2 = z * z + eps * eps
    h = np.sqrt(h2)
    opx = obs[0] - z * nhat[0]
    opy = obs[1] - z * nhat[1]
    opz = obs[2] - z * nhat[2]

    # Projected triangle vertices in the (ex, ey) plane relative to obs_proj
    px = np.empty(3); py = np.empty(3)
    for k in range(3):
        wx = tv[k, 0] - opx; wy = tv[k, 1] - opy; wz = tv[k, 2] - opz
        px[k] = wx * ex[0] + wy * ex[1] + wz * ex[2]
        py[k] = wx * ey[0] + wy * ey[1] + wz * ey[2]

    # I3 from the solid angle at the effective (lifted) observation point
    oex = opx + h * nhat[0]; oey = opy + h * nhat[1]; oez = opz + h * nhat[2]
    Omega = _solid_angle(tv[0], tv[1], tv[2], oex, oey, oez)
    I3 = -Omega / h

    # Edge loop: accumulate BD(0,0,1), BN(0,0,1), BN(1,0,1), BN(0,1,1)
    BD001 = 0.0
    BN1_001 = 0.0; BN2_001 = 0.0
    BN1_101 = 0.0; BN2_101 = 0.0
    BN1_011 = 0.0; BN2_011 = 0.0
    for e in range(3):
        ax = px[e]; ay = py[e]
        bx = px[(e + 1) % 3]; by = py[(e + 1) % 3]
        evx = bx - ax; evy = by - ay
        L = np.sqrt(evx * evx + evy * evy)
        if L <= 1e-30:
            continue
        tx = evx / L; ty = evy / L
        nx = ty; ny = -tx
        dp = ax * nx + ay * ny
        ua = ax * tx + ay * ty
        ub = bx * tx + by * ty
        rho2 = dp * dp + h2
        dJ1 = _J1(ub, rho2) - _J1(ua, rho2)
        dK1 = _K1(ub, rho2) - _K1(ua, rho2)

        v00 = _edge_int(0, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)
        v10 = _edge_int(1, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)
        v01 = _edge_int(0, 1, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)

        BD001 += dp * v00
        BN1_001 += nx * v00; BN2_001 += ny * v00
        BN1_101 += nx * v10; BN2_101 += ny * v10
        BN1_011 += nx * v01; BN2_011 += ny * v01

    I1 = BD001 - h2 * I3

    # 2D moments at n=3 (recursion constants: first order c=-1, second c=+1)
    m10 = -BN1_001
    m01 = -BN2_001
    m20 = I1 - BN1_101
    m11 = -BN1_011
    m02 = I1 - BN2_011

    # B1 = I1*delta, B2 = eps^2*I3*delta
    for i in range(3):
        for j in range(3):
            out[0, i, j] = 0.0
            out[1, i, j] = 0.0
            out[2, i, j] = 0.0
        out[0, i, i] = I1
        out[1, i, i] = eps * eps * I3

    # B3 = T2[3]: expand d_i d_j over the (ex, ey, nhat) frame,
    # d = -xi1*ex - xi2*ey + z*nhat
    for i in range(3):
        for j in range(3):
            out[2, i, j] = (
                ex[i] * ex[j] * m20
                + (ex[i] * ey[j] + ey[i] * ex[j]) * m11
                + ey[i] * ey[j] * m02
                - z * (ex[i] * nhat[j] + nhat[i] * ex[j]) * m10
                - z * (ey[i] * nhat[j] + nhat[i] * ey[j]) * m01
                + z * z * nhat[i] * nhat[j] * I3
            )


# ---------------------------------------------------------------------
# T-kernel basis blocks for one (source triangle, obs) pair
# ---------------------------------------------------------------------

@njit(cache=True)
def _t_basis_pair(tv, ex, ey, nhat, nsrc, obs, eps, out, T35, P):
    """Write the 6 T-kernel basis 3x3 blocks into out (6,3,3).

    Order: [L[P1], L[P2], L[P3], M[P1], M[P2], M[P3]].
    ``T35`` (3,3,3) and ``P`` (3,3,3) are caller-provided scratch.
    """
    dx = obs[0] - tv[0, 0]; dy = obs[1] - tv[0, 1]; dz = obs[2] - tv[0, 2]
    z = dx * nhat[0] + dy * nhat[1] + dz * nhat[2]
    h2 = z * z + eps * eps
    h = np.sqrt(h2)
    opx = obs[0] - z * nhat[0]
    opy = obs[1] - z * nhat[1]
    opz = obs[2] - z * nhat[2]

    px = np.empty(3); py = np.empty(3)
    for k in range(3):
        wx = tv[k, 0] - opx; wy = tv[k, 1] - opy; wz = tv[k, 2] - opz
        px[k] = wx * ex[0] + wy * ex[1] + wz * ex[2]
        py[k] = wx * ey[0] + wy * ey[1] + wz * ey[2]

    oex = opx + h * nhat[0]; oey = opy + h * nhat[1]; oez = opz + h * nhat[2]
    Omega = _solid_angle(tv[0], tv[1], tv[2], oex, oey, oez)
    I3 = -Omega / h

    # Edge accumulators: m=1 needs (0,0); m=3 needs (0,0),(1,0),(0,1),
    # (2,0),(1,1),(0,2) (BD only for (0,0,3); BN1/BN2 as consumed below).
    BN1_001 = 0.0; BN2_001 = 0.0
    BD003 = 0.0
    BN1_003 = 0.0; BN2_003 = 0.0
    BN1_103 = 0.0
    BN1_013 = 0.0; BN2_013 = 0.0
    BN1_203 = 0.0
    BN1_113 = 0.0
    BN1_023 = 0.0; BN2_023 = 0.0
    for e in range(3):
        ax = px[e]; ay = py[e]
        bx = px[(e + 1) % 3]; by = py[(e + 1) % 3]
        evx = bx - ax; evy = by - ay
        L = np.sqrt(evx * evx + evy * evy)
        if L <= 1e-30:
            continue
        tx = evx / L; ty = evy / L
        nx = ty; ny = -tx
        dp = ax * nx + ay * ny
        ua = ax * tx + ay * ty
        ub = bx * tx + by * ty
        rho2 = dp * dp + h2
        dJ1 = _J1(ub, rho2) - _J1(ua, rho2)
        dK1 = _K1(ub, rho2) - _K1(ua, rho2)
        dJ3 = _J3(ub, rho2) - _J3(ua, rho2)
        dK3 = _K3(ub, rho2) - _K3(ua, rho2)
        dL3 = dJ1 - rho2 * dJ3

        v00m1 = _edge_int(0, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)
        v00 = _edge_int(0, 0, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v10 = _edge_int(1, 0, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v01 = _edge_int(0, 1, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v20 = _edge_int(2, 0, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v11 = _edge_int(1, 1, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v02 = _edge_int(0, 2, dp, nx, ny, tx, ty, dJ3, dK3, dL3)

        BN1_001 += nx * v00m1; BN2_001 += ny * v00m1
        BD003 += dp * v00
        BN1_003 += nx * v00; BN2_003 += ny * v00
        BN1_103 += nx * v10
        BN1_013 += nx * v01; BN2_013 += ny * v01
        BN1_203 += nx * v20
        BN1_113 += nx * v11
        BN1_023 += nx * v02; BN2_023 += ny * v02

    I5 = (BD003 + I3) / (3.0 * h2)

    # 2D moments. First order: M(.,n) = -1/(n-2) * BN(.,n-2).
    m10_3 = -BN1_001
    m01_3 = -BN2_001
    m10_5 = -BN1_003 / 3.0
    m01_5 = -BN2_003 / 3.0
    # Second order at n=5: c = 1/3
    m20_5 = (I3 - BN1_103) / 3.0
    m11_5 = -BN1_013 / 3.0
    m02_5 = (I3 - BN2_013) / 3.0
    # Third order at n=5: c = 1/3
    m30_5 = (2.0 * m10_3 - BN1_203) / 3.0
    m21_5 = (m01_3 - BN1_113) / 3.0
    m12_5 = -BN1_023 / 3.0
    m03_5 = (2.0 * m01_3 - BN2_023) / 3.0

    # First 3D moments V3, V5: d = -xi1*ex - xi2*ey + z*nhat
    V3x = -ex[0] * m10_3 - ey[0] * m01_3 + nhat[0] * z * I3
    V3y = -ex[1] * m10_3 - ey[1] * m01_3 + nhat[1] * z * I3
    V3z = -ex[2] * m10_3 - ey[2] * m01_3 + nhat[2] * z * I3
    V5x = -ex[0] * m10_5 - ey[0] * m01_5 + nhat[0] * z * I5
    V5y = -ex[1] * m10_5 - ey[1] * m01_5 + nhat[1] * z * I5
    V5z = -ex[2] * m10_5 - ey[2] * m01_5 + nhat[2] * z * I5

    # Third 3D moment T3[5]: sum over component triples of the frame.
    # Component c: 0 -> (-1, ex), 1 -> (-1, ey), 2 -> (z, nhat); for a
    # triple with a slots of xi1, b of xi2, and the rest z, the scalar
    # factor is (-1)^(a+b) * z^(#z) * M(a, b, 5).
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T35[i, j, k] = 0.0
    for c1 in range(3):
        for c2 in range(3):
            for c3 in range(3):
                a = (1 if c1 == 0 else 0) + (1 if c2 == 0 else 0) + (1 if c3 == 0 else 0)
                b = (1 if c1 == 1 else 0) + (1 if c2 == 1 else 0) + (1 if c3 == 1 else 0)
                nz = 3 - a - b
                if a == 3:
                    mom = m30_5
                elif a == 2 and b == 1:
                    mom = m21_5
                elif a == 2 and b == 0:
                    mom = m20_5
                elif a == 1 and b == 2:
                    mom = m12_5
                elif a == 1 and b == 1:
                    mom = m11_5
                elif a == 1 and b == 0:
                    mom = m10_5
                elif a == 0 and b == 3:
                    mom = m03_5
                elif a == 0 and b == 2:
                    mom = m02_5
                elif a == 0 and b == 1:
                    mom = m01_5
                else:
                    mom = I5
                sign = 1.0
                if (a + b) % 2 == 1:
                    sign = -1.0
                coeff = sign * z ** nz * mom
                if coeff == 0.0:
                    continue
                b1 = ex if c1 == 0 else (ey if c1 == 1 else nhat)
                b2 = ex if c2 == 0 else (ey if c2 == 1 else nhat)
                b3 = ex if c3 == 0 else (ey if c3 == 1 else nhat)
                for i in range(3):
                    ci = coeff * b1[i]
                    for j in range(3):
                        cij = ci * b2[j]
                        for k in range(3):
                            T35[i, j, k] += cij * b3[k]

    # ---- P tensors and their L / M contractions ----
    # L[P]_ij = sum_m n_m P[i,j,m]
    # M[P]_ij = n_j * sum_m P[i,m,m] + sum_k n_k P[i,k,j]
    e2 = eps * eps
    V3 = (V3x, V3y, V3z)
    V5 = (V5x, V5y, V5z)

    # P1[i,k,m] = delta_ik V3_m
    for i in range(3):
        for k in range(3):
            for m in range(3):
                P[i, k, m] = V3[m] if i == k else 0.0
    _contract_LM(P, nsrc, out, 0, 3)

    # P2[i,k,m] = delta_im V3_k + delta_km V3_i - 3*T35[i,k,m]
    for i in range(3):
        for k in range(3):
            for m in range(3):
                val = -3.0 * T35[i, k, m]
                if i == m:
                    val += V3[k]
                if k == m:
                    val += V3[i]
                P[i, k, m] = val
    _contract_LM(P, nsrc, out, 1, 4)

    # P3[i,k,m] = eps^2 * delta_ik V5_m
    for i in range(3):
        for k in range(3):
            for m in range(3):
                P[i, k, m] = e2 * V5[m] if i == k else 0.0
    _contract_LM(P, nsrc, out, 2, 5)


@njit(cache=True, inline="always")
def _contract_LM(P, n, out, idx_L, idx_M):
    """L and M contractions of P (3,3,3) with source normal n into out."""
    for i in range(3):
        tr = P[i, 0, 0] + P[i, 1, 1] + P[i, 2, 2]
        for j in range(3):
            out[idx_L, i, j] = (P[i, j, 0] * n[0] + P[i, j, 1] * n[1]
                                + P[i, j, 2] * n[2])
            out[idx_M, i, j] = (n[j] * tr
                                + n[0] * P[i, 0, j] + n[1] * P[i, 1, j]
                                + n[2] * P[i, 2, j])


# ---------------------------------------------------------------------
# Full-matrix basis assembly (parallel over source triangles)
# ---------------------------------------------------------------------

@njit(cache=True, parallel=True)
def u_basis_matrices(x_field, tri_verts, eps_arr):
    """U-kernel basis stack, shape (3, 3*N_f, 3*N_s)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3, 3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _u_basis_pair(tri_verts[s], ex, ey, nhat, x_field[f], eps, blk)
            for b in range(3):
                for i in range(3):
                    for j in range(3):
                        out[b, 3 * f + i, 3 * s + j] = blk[b, i, j]
    return out


@njit(cache=True, parallel=True)
def t_basis_matrices(x_field, tri_verts, normals, eps_arr):
    """T-kernel basis stack, shape (6, 3*N_f, 3*N_s)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((6, 3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _t_basis_pair(tri_verts[s], ex, ey, nhat, normals[s],
                          x_field[f], eps, blk, T35, P)
            for b in range(6):
                for i in range(3):
                    for j in range(3):
                        out[b, 3 * f + i, 3 * s + j] = blk[b, i, j]
    return out


@njit(cache=True, parallel=True)
def u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3):
    """U-kernel matrix with coefficients applied in-loop (no basis storage)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _u_basis_pair(tri_verts[s], ex, ey, nhat, x_field[f], eps, blk)
            for i in range(3):
                for j in range(3):
                    out[3 * f + i, 3 * s + j] = (g1 * blk[0, i, j]
                                                 + g2 * blk[1, i, j]
                                                 + g3 * blk[2, i, j])
    return out


@njit(cache=True, parallel=True)
def t_matrix_direct(x_field, tri_verts, normals, eps_arr,
                    c1, c2, c3, c4, c5, c6):
    """T-kernel matrix with coefficients applied in-loop (no basis storage)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _t_basis_pair(tri_verts[s], ex, ey, nhat, normals[s],
                          x_field[f], eps, blk, T35, P)
            for i in range(3):
                for j in range(3):
                    out[3 * f + i, 3 * s + j] = (
                        c1 * blk[0, i, j] + c2 * blk[1, i, j]
                        + c3 * blk[2, i, j] + c4 * blk[3, i, j]
                        + c5 * blk[4, i, j] + c6 * blk[5, i, j])
    return out

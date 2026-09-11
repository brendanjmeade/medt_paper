#!/usr/bin/env python3
"""
analytical_kernels.py — Fully analytical mollified elastic kernels
===================================================================

Analytically integrate the mollified (Cortez-style) displacement
discontinuity stress kernel over a flat triangular element.

R_eps = sqrt(|x' - obs|^2 + eps^2)

Strategy:
  1. Project obs onto triangle plane.  h = sqrt(z^2 + eps^2).
  2. Van Oosterom solid angle -> I3 = -Omega/h.
  3. Divergence theorem edge sums + recursion -> I1, I5, I7.
  4. Higher moment integrals (∫∫ ξ^a ξ^b / R^n dA) via
     repeated divergence theorem with edge antiderivatives.
  5. Transform 2D moments to 3D tensor moments.
  6. Contract with elastic constants -> DD stress kernel.
"""

import numpy as np
from math import comb


# ====================================================================
# 1D antiderivatives for edge integrals
# ====================================================================

def solid_angle_van_oosterom(v1, v2, v3, obs):
    """Solid angle of triangle (v1,v2,v3) as seen from obs."""
    r1 = v1 - obs
    r2 = v2 - obs
    r3 = v3 - obs
    R1 = np.linalg.norm(r1)
    R2 = np.linalg.norm(r2)
    R3 = np.linalg.norm(r3)
    numer = np.dot(r1, np.cross(r2, r3))
    denom = (R1 * R2 * R3
             + R3 * np.dot(r1, r2)
             + R1 * np.dot(r2, r3)
             + R2 * np.dot(r1, r3))
    return 2.0 * np.arctan2(numer, denom)


def _antideriv_J(u, rho2, order):
    """
    Antiderivative of 1/(u^2+rho^2)^(n/2) w.r.t. u.
    Orders: 1, 3, 5, 7.
    """
    R = np.sqrt(u**2 + rho2)
    if order == 1:
        return np.log(u + R + 1e-300)
    elif order == 3:
        if rho2 < 1e-60:
            return 0.0
        return u / (rho2 * R)
    elif order == 5:
        if rho2 < 1e-60:
            return 0.0
        return u * (3*rho2 + 2*u**2) / (3 * rho2**2 * R**3)
    elif order == 7:
        if rho2 < 1e-60:
            return 0.0
        return u * (15*rho2**2 + 20*rho2*u**2 + 8*u**4) / (15 * rho2**3 * R**5)
    else:
        raise ValueError(f"J order {order} not implemented")


def _antideriv_K(u, rho2, order):
    """
    Antiderivative of u/(u^2+rho^2)^(n/2) w.r.t. u.

    K_1(u) = R
    K_3(u) = -1/R
    K_5(u) = -1/(3*R^3)
    K_7(u) = -1/(5*R^5)
    """
    R = np.sqrt(u**2 + rho2)
    if order == 1:
        return R
    elif order == 3:
        if R < 1e-30:
            return 0.0
        return -1.0 / R
    elif order == 5:
        if R < 1e-30:
            return 0.0
        return -1.0 / (3.0 * R**3)
    elif order == 7:
        if R < 1e-30:
            return 0.0
        return -1.0 / (5.0 * R**5)
    else:
        raise ValueError(f"K order {order} not implemented")


# ====================================================================
# Base potential integrals (unchanged from before)
# ====================================================================

def integrate_over_triangle(v1, v2, v3, obs, eps):
    """
    Analytically compute base potential integrals I_n = ∫∫ 1/R_eps^n dA.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    normal = np.cross(e1, e2)
    area2 = np.linalg.norm(normal)
    if area2 < 1e-30:
        return {'I1': 0, 'I3': 0, 'I5': 0, 'I7': 0}
    normal = normal / area2

    z = np.dot(obs - v1, normal)
    h = np.sqrt(z**2 + eps**2)
    obs_proj = obs - z * normal

    obs_eff = obs_proj + h * normal
    Omega = solid_angle_van_oosterom(v1, v2, v3, obs_eff)
    I3 = -Omega / h

    ex = e1 / np.linalg.norm(e1)
    ey = np.cross(normal, ex)

    def to_2d(p):
        d = p - obs_proj
        return np.array([np.dot(d, ex), np.dot(d, ey)])

    p1, p2, p3 = to_2d(v1), to_2d(v2), to_2d(v3)
    edges = [(p1, p2), (p2, p3), (p3, p1)]

    E1 = E3 = E5 = 0.0
    for pa, pb in edges:
        edge_vec = pb - pa
        L = np.linalg.norm(edge_vec)
        if L < 1e-30:
            continue
        t_hat = edge_vec / L
        n_out = np.array([t_hat[1], -t_hat[0]])
        d_perp = np.dot(pa, n_out)
        u_a = np.dot(pa, t_hat)
        u_b = np.dot(pb, t_hat)
        rho2 = d_perp**2 + h**2

        for order, key in [(1, 'E1'), (3, 'E3'), (5, 'E5')]:
            val = d_perp * (_antideriv_J(u_b, rho2, order)
                           - _antideriv_J(u_a, rho2, order))
            if key == 'E1':
                E1 += val
            elif key == 'E3':
                E3 += val
            elif key == 'E5':
                E5 += val

    h2 = h**2
    I1 = E1 - h2 * I3
    I5 = (E3 + I3) / (3 * h2) if h2 > 1e-60 else 0.0
    I7 = (E5 + 3 * I5) / (5 * h2) if h2 > 1e-60 else 0.0

    return {'I1': I1, 'I3': I3, 'I5': I5, 'I7': I7,
            'Omega': Omega, 'z': z, 'h': h}


# ====================================================================
# All moment integrals via edge-by-edge divergence theorem
# ====================================================================

def integrate_all_moments(v1, v2, v3, obs, eps):
    """
    Compute ALL moment integrals needed for displacement & stress kernels.

    2D moments: M2d[(a,b,n)] = ∫∫_T ξ₁^a ξ₂^b / R_eps^n dA
    3D tensors: I[n], V3d[n], T2[n], T3[n], T4[n]

    where ξ₁, ξ₂ are in-plane coords centered at obs_proj,
    d = obs - y = -ξ₁ ê₁ - ξ₂ ê₂ + z n̂,
    and R = sqrt(ξ₁² + ξ₂² + h²), h = sqrt(z² + ε²).

    Returns
    -------
    dict with:
      ex, ey, nhat : (3,) in-plane and normal basis vectors
      z, h         : signed height and effective height
      I            : {n: scalar} base integrals, n = 1,3,5,7
      V            : {n: (3,)} vector moments ∫∫ d_i/R^n dA
      T2           : {n: (3,3)} tensor ∫∫ d_i d_j/R^n dA
      T3           : {n: (3,3,3)} tensor ∫∫ d_i d_j d_k/R^n dA
      T4           : {n: (3,3,3,3)} tensor ∫∫ d_i d_j d_k d_l/R^n dA
    """
    # --- Setup: triangle geometry ---
    e1 = v2 - v1
    e2 = v3 - v1
    normal = np.cross(e1, e2)
    area2 = np.linalg.norm(normal)
    if area2 < 1e-30:
        return None
    normal = normal / area2

    z = np.dot(obs - v1, normal)
    h2 = z**2 + eps**2
    h = np.sqrt(h2)
    obs_proj = obs - z * normal

    ex = e1 / np.linalg.norm(e1)
    ey = np.cross(normal, ex)

    def to_2d(p):
        d = p - obs_proj
        return np.array([np.dot(d, ex), np.dot(d, ey)])

    p1, p2, p3 = to_2d(v1), to_2d(v2), to_2d(v3)

    # --- I3 from solid angle ---
    obs_eff = obs_proj + h * normal
    Omega = solid_angle_van_oosterom(v1, v2, v3, obs_eff)
    I3 = -Omega / h

    # --- Edge loop: compute boundary integrals ---
    edges = [(p1, p2), (p2, p3), (p3, p1)]

    # BN1[(a,b,m)] = Σ_edges n̂₁ ∫ ξ₁^a ξ₂^b / R^m du
    # BN2[(a,b,m)] = Σ_edges n̂₂ ∫ ξ₁^a ξ₂^b / R^m du
    # BD[(a,b,m)]  = Σ_edges d_perp ∫ ξ₁^a ξ₂^b / R^m du
    BN1 = {}
    BN2 = {}
    BD = {}

    for pa, pb in edges:
        edge_vec = pb - pa
        L = np.linalg.norm(edge_vec)
        if L < 1e-30:
            continue

        t_hat = edge_vec / L
        n_out = np.array([t_hat[1], -t_hat[0]])

        c1, c2 = n_out[0], n_out[1]
        s1, s2 = t_hat[0], t_hat[1]
        dp = np.dot(pa, n_out)
        u_a = np.dot(pa, t_hat)
        u_b = np.dot(pb, t_hat)
        rho2 = dp**2 + h2

        # Compute 1D antiderivative differences
        dJ = {}
        dK = {}
        for m in [1, 3, 5]:
            dJ[m] = (_antideriv_J(u_b, rho2, m)
                     - _antideriv_J(u_a, rho2, m))
            dK[m] = (_antideriv_K(u_b, rho2, m)
                     - _antideriv_K(u_a, rho2, m))

        # Derived antiderivatives: ∫u²/R^m = J_{m-2} - ρ²J_m
        dL = {}
        dM = {}
        for m in [3, 5]:
            dL[m] = dJ[m-2] - rho2 * dJ[m]
            dM[m] = dK[m-2] - rho2 * dK[m]

        def edge_integral(a, b, m):
            """∫_{u_a}^{u_b} ξ₁^a ξ₂^b / R^m du"""
            total = 0.0
            for i in range(a + 1):
                for j in range(b + 1):
                    coeff = (comb(a, i) * comb(b, j)
                             * dp**(a + b - i - j)
                             * c1**(a - i) * s1**i
                             * c2**(b - j) * s2**j)
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
                    total += coeff * delta
            return total

        # Compute all needed boundary integrals
        # Need (a,b) combos up to order 3 at m=1,3,5
        # and order 0 at m=1,3,5 for base integrals
        for tot_order in range(4):  # 0,1,2,3
            for a in range(tot_order + 1):
                b = tot_order - a
                for m in [1, 3, 5]:
                    # Skip higher-order moments at low m if not needed
                    # We need: moments up to order 0 at all m (base integrals)
                    #          moments up to order 1 at m=1,3,5 (for first-order)
                    #          moments up to order 2 at m=3,5 (for second-order)
                    #          moments up to order 3 at m=5 (for third/fourth-order)
                    if tot_order >= 2 and m == 1:
                        continue  # 2nd+ order not needed at m=1
                    if tot_order >= 3 and m == 3:
                        continue  # 3rd+ order not needed at m=3

                    key = (a, b, m)
                    val = edge_integral(a, b, m)
                    BN1[key] = BN1.get(key, 0.0) + c1 * val
                    BN2[key] = BN2.get(key, 0.0) + c2 * val
                    BD[key] = BD.get(key, 0.0) + dp * val

    # ================================================================
    # Build 2D area integrals from boundary integrals
    # ================================================================
    M = {}

    # --- Step 0: Base integrals ---
    E1 = BD.get((0, 0, 1), 0.0)
    E3 = BD.get((0, 0, 3), 0.0)
    E5 = BD.get((0, 0, 5), 0.0)

    I1 = E1 - h2 * I3
    I5 = (E3 + I3) / (3 * h2) if h2 > 1e-60 else 0.0
    I7 = (E5 + 3 * I5) / (5 * h2) if h2 > 1e-60 else 0.0

    M[(0, 0, 1)] = I1
    M[(0, 0, 3)] = I3
    M[(0, 0, 5)] = I5
    M[(0, 0, 7)] = I7

    # --- Step 1: First-order 2D moments ---
    # M^{(1,0)}_n = -1/(n-2) * BN1[(0,0,n-2)]
    # M^{(0,1)}_n = -1/(n-2) * BN2[(0,0,n-2)]
    for n in [3, 5, 7]:
        m = n - 2
        c = -1.0 / (n - 2)
        M[(1, 0, n)] = c * BN1.get((0, 0, m), 0.0)
        M[(0, 1, n)] = c * BN2.get((0, 0, m), 0.0)

    # --- Step 2: Second-order 2D moments ---
    # M^{(2,0)}_n = [I_{n-2} - BN1[(1,0,n-2)]] / (n-2)
    # M^{(1,1)}_n = -1/(n-2) * BN1[(0,1,n-2)]
    # M^{(0,2)}_n = [I_{n-2} - BN2[(0,1,n-2)]] / (n-2)
    # n=3 is needed for the integrated Kelvin G (force) kernel.
    for n in [3, 5, 7]:
        m = n - 2
        c = 1.0 / (n - 2)
        M[(2, 0, n)] = c * (M[(0, 0, m)] - BN1.get((1, 0, m), 0.0))
        M[(1, 1, n)] = -c * BN1.get((0, 1, m), 0.0)
        M[(0, 2, n)] = c * (M[(0, 0, m)] - BN2.get((0, 1, m), 0.0))

    # --- Step 3: Third-order 2D moments (for displacement kernel) ---
    # M^{(3,0)}_n = [2*M^{(1,0)}_{n-2} - BN1[(2,0,n-2)]] / (n-2)
    # M^{(2,1)}_n = [M^{(0,1)}_{n-2} - BN1[(1,1,n-2)]] / (n-2)
    # M^{(1,2)}_n = -1/(n-2) * BN1[(0,2,n-2)]
    # M^{(0,3)}_n = [2*M^{(0,1)}_{n-2} - BN2[(0,2,n-2)]] / (n-2)
    for n in [5, 7]:
        m = n - 2
        c = 1.0 / (n - 2)
        if (2, 0, m) in BN1:
            M[(3, 0, n)] = c * (2 * M[(1, 0, m)] - BN1[(2, 0, m)])
        if (1, 1, m) in BN1:
            M[(2, 1, n)] = c * (M[(0, 1, m)] - BN1[(1, 1, m)])
        if (0, 2, m) in BN1:
            M[(1, 2, n)] = -c * BN1[(0, 2, m)]
        if (0, 2, m) in BN2:
            M[(0, 3, n)] = c * (2 * M[(0, 1, m)] - BN2[(0, 2, m)])

    # --- Step 4: Fourth-order 2D moments (for stress kernel at R^7) ---
    # M^{(4,0)}_7 = [3*M^{(2,0)}_5 - BN1[(3,0,5)]] / 5
    # M^{(3,1)}_7 = [2*M^{(1,1)}_5 - BN1[(2,1,5)]] / 5
    # M^{(2,2)}_7 = [M^{(0,2)}_5 - BN1[(1,2,5)]] / 5
    # M^{(1,3)}_7 = -1/5 * BN1[(0,3,5)]
    # M^{(0,4)}_7 = [3*M^{(0,2)}_5 - BN2[(0,3,5)]] / 5
    n = 7
    m = 5
    c = 1.0 / 5.0
    if (3, 0, m) in BN1:
        M[(4, 0, n)] = c * (3 * M[(2, 0, m)] - BN1[(3, 0, m)])
    if (2, 1, m) in BN1:
        M[(3, 1, n)] = c * (2 * M[(1, 1, m)] - BN1[(2, 1, m)])
    if (1, 2, m) in BN1:
        M[(2, 2, n)] = c * (M[(0, 2, m)] - BN1[(1, 2, m)])
    if (0, 3, m) in BN1:
        M[(1, 3, n)] = c * (-BN1[(0, 3, m)])
    if (0, 3, m) in BN2:
        M[(0, 4, n)] = c * (3 * M[(0, 2, m)] - BN2[(0, 3, m)])

    # ================================================================
    # Transform 2D moments to 3D tensor moments
    # ================================================================
    # d_i = -ξ₁ ex_i - ξ₂ ey_i + z nhat_i
    # Basis vectors for decomposition:
    basis = [ex, ey, normal]  # [ê₁, ê₂, n̂]
    signs = [-1.0, -1.0, 1.0]  # d = -ξ₁ê₁ - ξ₂ê₂ + z n̂
    coords = [None, None, z]   # ξ₁, ξ₂ are integrated; z is constant

    def get_2d_moment(a, b, n):
        """Get M^{(a,b)}_n, returning 0 if not computed."""
        return M.get((a, b, n), 0.0)

    # --- 3D vector moments V[n][i] = ∫∫ d_i / R^n dA ---
    V = {}
    for n in [3, 5, 7]:
        V[n] = np.zeros(3)
        for i in range(3):
            # d_i = -ξ₁ ex_i - ξ₂ ey_i + z n̂_i
            V[n][i] = (-ex[i] * get_2d_moment(1, 0, n)
                       - ey[i] * get_2d_moment(0, 1, n)
                       + z * normal[i] * get_2d_moment(0, 0, n))

    # --- 3D 2nd-order tensor T2[n][i,j] = ∫∫ d_i d_j / R^n dA ---
    # n=3 is required for the integrated Kelvin G (force) kernel.
    T2 = {}
    for n in [3, 5, 7]:
        T2[n] = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                # d_i d_j expanded:
                val = 0.0
                # ξ₁² term
                val += ex[i] * ex[j] * get_2d_moment(2, 0, n)
                # ξ₁ξ₂ terms
                val += (ex[i] * ey[j] + ey[i] * ex[j]) * get_2d_moment(1, 1, n)
                # ξ₂² term
                val += ey[i] * ey[j] * get_2d_moment(0, 2, n)
                # ξ₁·z terms
                val += -z * (ex[i] * normal[j] + normal[i] * ex[j]) * get_2d_moment(1, 0, n)
                # ξ₂·z terms
                val += -z * (ey[i] * normal[j] + normal[i] * ey[j]) * get_2d_moment(0, 1, n)
                # z² term
                val += z**2 * normal[i] * normal[j] * get_2d_moment(0, 0, n)
                T2[n][i, j] = val

    # --- 3D 3rd-order tensor T3[n][i,j,k] = ∫∫ d_i d_j d_k / R^n dA ---
    T3 = {}
    for n in [5, 7]:
        T3[n] = np.zeros((3, 3, 3))
        # d_i = Σ_A s_A b_A_i * ξ_A  (A runs over ξ₁, ξ₂, z)
        # Use components: (-ξ₁, ex), (-ξ₂, ey), (z, n̂)
        components = [(-1, ex, 'xi1'), (-1, ey, 'xi2'), (z, normal, 'z')]
        for si, bi, li in components:
            for sj, bj, lj in components:
                for sk, bk, lk in components:
                    # Powers of ξ₁, ξ₂
                    a = sum(1 for l in [li, lj, lk] if l == 'xi1')
                    b = sum(1 for l in [li, lj, lk] if l == 'xi2')
                    c = sum(1 for l in [li, lj, lk] if l == 'z')
                    # Coefficient
                    sign = 1.0
                    z_power = 1.0
                    for s, l in [(si, li), (sj, lj), (sk, lk)]:
                        if l == 'z':
                            z_power *= s  # z factor
                        else:
                            sign *= s  # -1 factors from ξ terms
                    coeff = sign * z_power
                    moment = get_2d_moment(a, b, n)
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                T3[n][i, j, k] += coeff * bi[i] * bj[j] * bk[k] * moment

    # --- 3D 4th-order tensor T4[7][i,j,k,l] = ∫∫ d_i d_j d_k d_l / R^7 dA ---
    T4 = {}
    n = 7
    T4[n] = np.zeros((3, 3, 3, 3))
    components = [(-1, ex, 'xi1'), (-1, ey, 'xi2'), (z, normal, 'z')]
    for si, bi, li in components:
        for sj, bj, lj in components:
            for sk, bk, lk in components:
                for sl, bl, ll in components:
                    a = sum(1 for l_ in [li, lj, lk, ll] if l_ == 'xi1')
                    b = sum(1 for l_ in [li, lj, lk, ll] if l_ == 'xi2')
                    sign = 1.0
                    z_power = 1.0
                    for s, l_ in [(si, li), (sj, lj), (sk, lk), (sl, ll)]:
                        if l_ == 'z':
                            z_power *= s
                        else:
                            sign *= s
                    coeff = sign * z_power
                    moment = get_2d_moment(a, b, n)
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for l in range(3):
                                    T4[n][i, j, k, l] += (
                                        coeff * bi[i] * bj[j] * bk[k] * bl[l] * moment
                                    )

    return {
        'ex': ex, 'ey': ey, 'nhat': normal,
        'z': z, 'h': h,
        'I': {1: I1, 3: I3, 5: I5, 7: I7},
        'V': V,       # V[n][i]
        'T2': T2,     # T2[n][i,j]
        'T3': T3,     # T3[n][i,j,k]
        'T4': T4,     # T4[n][i,j,k,l]
        'M2d': M,     # raw 2D moments for debugging
    }


# ====================================================================
# Integrated D2G tensor (analytical)
# ====================================================================

def integrate_D2G(v1, v2, v3, obs, mu, nu, eps):
    """
    Analytically integrate D2G[r,p,s,q] = d²G_rp/(d d_s d d_q)
    over a triangular element.

    Uses the Galerkin/Cortez-convolved form
        G_rp^eps = C1 [(3-4ν) δ_rp/R + d_r d_p/R^3 + 2(1-ν) ε² δ_rp/R^3].

    Returns (3,3,3,3) tensor.
    """
    mom = integrate_all_moments(v1, v2, v3, obs, eps)
    if mom is None:
        return np.zeros((3, 3, 3, 3))

    I3 = mom['I'][3]
    I5 = mom['I'][5]
    T2_5 = mom['T2'][5]    # ∫∫ d_i d_j / R^5 dA
    T2_7 = mom['T2'][7]    # ∫∫ d_i d_j / R^7 dA
    T4_7 = mom['T4'][7]    # ∫∫ d_i d_j d_k d_l / R^7 dA

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps**2

    # D2G[r,p,s,q] = C1 * {
    #   -c34*(r==p) * [(s==q)/R^3 - 3*d_s*d_q/R^5]
    #   + (r==s)*[(p==q)/R^3 - 3*d_p*d_q/R^5]
    #   + (p==s)*[(r==q)/R^3 - 3*d_r*d_q/R^5]
    #   - 3*(r==q)*d_p*d_s/R^5
    #   - 3*d_r*(p==q)*d_s/R^5
    #   - 3*d_r*d_p*(s==q)/R^5
    #   + 15*d_r*d_p*d_s*d_q/R^7
    #   + 2(1-ν)ε² (r==p) [-3 (s==q)/R^5 + 15 d_s d_q/R^7]   # <- Cortez-blob term
    # }
    #
    # Integrated:
    ID2G = np.zeros((3, 3, 3, 3))
    for r in range(3):
        for p in range(3):
            for s in range(3):
                for q in range(3):
                    val = 0.0

                    # Terms proportional to I3 (1/R^3 integrated)
                    val += -c34 * (r == p) * (s == q) * I3
                    val += (r == s) * (p == q) * I3
                    val += (p == s) * (r == q) * I3

                    # Terms proportional to T2_5 (d_a*d_b/R^5 integrated)
                    val += c34 * (r == p) * 3.0 * T2_5[s, q]
                    val += -(r == s) * 3.0 * T2_5[p, q]
                    val += -(p == s) * 3.0 * T2_5[r, q]
                    val += -3.0 * (r == q) * T2_5[p, s]
                    val += -3.0 * (p == q) * T2_5[r, s]
                    val += -3.0 * (s == q) * T2_5[r, p]

                    # Terms proportional to T4_7 (d_r*d_p*d_s*d_q/R^7)
                    val += 15.0 * T4_7[r, p, s, q]

                    # Cortez-blob term: 2(1-ν)ε² (r==p) * [-3(s==q)/R^5 + 15 d_s d_q/R^7]
                    val += c_blob * (r == p) * (
                        -3.0 * (s == q) * I5 + 15.0 * T2_7[s, q]
                    )

                    ID2G[r, p, s, q] = C1 * val

    return ID2G


# ====================================================================
# Full DD stress kernel (analytical integration)
# ====================================================================

def analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps):
    """
    Analytically integrated DD stress kernel over a triangular element.

    H[m,n,k] = ∫∫_T K[m,n,k](obs, y, normal) dA(y)

    such that: sigma_mn(obs) = H[m,n,k] * Delta_u_k

    Uses the representation:
        K_mn,k = -C_mnrs * C_kjpq * nu_j * D2G[r,p,s,q]

    Parameters
    ----------
    obs    : (3,) observation point
    v1,v2,v3 : (3,) triangle vertices
    normal : (3,) unit normal at the element
    mu, nu : elastic parameters
    eps    : mollification parameter

    Returns
    -------
    H : (3,3,3) integrated stress kernel
    """
    ID2G = integrate_D2G(v1, v2, v3, obs, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n_vec = normal

    # Step 1: B[r,s,k] = C_kjpq * n_j * ID2G[r,p,s,q]
    # C_kjpq = lam*d_kj*d_pq + mu*(d_kp*d_jq + d_kq*d_jp)
    trace_ID2G = np.zeros((3, 3))
    for r in range(3):
        for s in range(3):
            trace_ID2G[r, s] = sum(ID2G[r, p, s, p] for p in range(3))

    ID2G_nq = np.zeros((3, 3, 3))
    ID2G_np = np.zeros((3, 3, 3))
    for r in range(3):
        for i in range(3):
            for s in range(3):
                ID2G_nq[r, i, s] = sum(n_vec[q] * ID2G[r, i, s, q] for q in range(3))
                ID2G_np[r, s, i] = sum(n_vec[p] * ID2G[r, p, s, i] for p in range(3))

    B = np.zeros((3, 3, 3))
    for k in range(3):
        B[:, :, k] = (lam * n_vec[k] * trace_ID2G
                       + mu * ID2G_nq[:, k, :]
                       + mu * ID2G_np[:, :, k])

    # Step 2: H[m,n,k] = -C_mnrs * B[r,s,k]
    trace_B = np.array([sum(B[r, r, k] for r in range(3)) for k in range(3)])

    H = np.zeros((3, 3, 3))
    for k in range(3):
        for m in range(3):
            for nn in range(3):
                H[m, nn, k] = -(lam * (m == nn) * trace_B[k]
                                 + mu * (B[m, nn, k] + B[nn, m, k]))
    return H


# ====================================================================
# Integrated mollified Kelvin gradient ∂G/∂x  (used by both
# DD displacement and Kelvin force-stress kernels)
# ====================================================================

def integrate_DG(v1, v2, v3, obs, mu, nu, eps):
    """
    Analytically integrate G1[i,j,m] = ∂G_ij/∂x_m  over a triangle.

    Uses the Galerkin/Cortez-convolved form
        G_ij^eps = C1 [(3-4ν) δ_ij/R + d_i d_j/R^3 + 2(1-ν) ε² δ_ij/R^3].

    With d = obs - y and R = sqrt(|d|^2 + eps^2):

        ∂G_ij/∂x_m = C1 * [
              -(3-4ν) δ_ij d_m / R^3
              + δ_im d_j / R^3
              + δ_jm d_i / R^3
              - 3 d_i d_j d_m / R^5
              - 6(1-ν) ε² δ_ij d_m / R^5     # <- Cortez-blob contribution
        ]

    Integrated:

        G1[i,j,m] = C1 * [
              -(3-4ν) δ_ij V_3[m]
              + δ_im V_3[j]
              + δ_jm V_3[i]
              - 3 T3_5[i,j,m]
              - 6(1-ν) ε² δ_ij V_5[m]
        ]

    Returns (3,3,3) tensor.
    """
    mom = integrate_all_moments(v1, v2, v3, obs, eps)
    if mom is None:
        return np.zeros((3, 3, 3))

    V3 = mom['V'][3]      # ∫∫ d_i / R^3 dA
    V5 = mom['V'][5]      # ∫∫ d_i / R^5 dA
    T3_5 = mom['T3'][5]   # ∫∫ d_i d_j d_k / R^5 dA

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob_dg = 6.0 * (1.0 - nu) * eps**2

    G1 = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                G1[i, j, m] = C1 * (
                    -c34 * (i == j) * V3[m]
                    + (i == m) * V3[j]
                    + (j == m) * V3[i]
                    - 3.0 * T3_5[i, j, m]
                    - c_blob_dg * (i == j) * V5[m]
                )
    return G1


# ====================================================================
# Integrated mollified Kelvin G (force/single-layer) kernel
# ====================================================================

def analytical_kelvin_G(obs, v1, v2, v3, mu, nu, eps):
    """
    Analytically integrate the mollified Kelvin displacement Green's
    function G_ij over a triangular element.

        G_ij(d, eps) = 1/(16π μ (1-ν)) * [
            (3 - 4ν) δ_ij / R_eps
            + d_i d_j / R_eps^3
            + 2(1-ν) ε² δ_ij / R_eps^3        # <- Cortez-blob contribution
        ]

    where d = obs - y and R_eps = sqrt(|d|^2 + eps^2). The result is
    the displacement at `obs` due to a unit constant force per unit
    area applied over the triangle (single-layer / force kernel):

        u_i(obs) = G_int[i,j] * f_j

    Parameters
    ----------
    obs    : (3,) observation point
    v1,v2,v3 : (3,) triangle vertices
    mu, nu : elastic parameters
    eps    : mollification parameter

    Returns
    -------
    G_int : (3, 3) integrated Kelvin displacement kernel
    """
    mom = integrate_all_moments(v1, v2, v3, obs, eps)
    if mom is None:
        return np.zeros((3, 3))

    I1 = mom['I'][1]          # ∫∫ 1/R_eps dA
    I3 = mom['I'][3]          # ∫∫ 1/R_eps^3 dA
    T2_3 = mom['T2'][3]       # ∫∫ d_i d_j / R_eps^3 dA

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps**2

    G_int = C1 * ((c34 * I1 + c_blob * I3) * np.eye(3) + T2_3)
    return G_int


# ====================================================================
# Integrated DD displacement kernel (slip → displacement)
# ====================================================================

def analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps):
    """
    Analytically integrated displacement kernel for a triangular
    displacement discontinuity (DD) element.

    U[i,j] gives the i-th displacement at obs due to a unit constant
    slip vector Δu_j on the triangle:

        u_i(obs) = U[i,j] * Δu_j

    Derivation (slip index pairs with the NORMAL in the first index pair of C):
        u_i(x) = ∫∫ Δu_j C_jkpq n_k (∂G_ip/∂y_q) dS(y)
               = -∫∫ Δu_j C_jkpq n_k (∂G_ip/∂x_q) dS  [∂_y = -∂_x]
               = -C_jkpq n_k G1[i,p,q]  ·  Δu_j

    Expanding C_jkpq = λ δ_jk δ_pq + μ (δ_jp δ_kq + δ_jq δ_kp):

        U[i,j] = -[ μ Σ_m n_m G1[i,j,m]
                  + λ n_j Σ_m G1[i,m,m]
                  + μ Σ_k n_k G1[i,k,j] ]

    Returns (3,3) tensor.
    """
    G1 = integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal

    U = np.zeros((3, 3))
    for i in range(3):
        # trace term: Σ_m G1[i,m,m]
        trace_im = sum(G1[i, m, m] for m in range(3))
        for j in range(3):
            term1 = mu * sum(n[m] * G1[i, j, m] for m in range(3))
            term2 = lam * n[j] * trace_im
            term3 = mu * sum(n[k] * G1[i, k, j] for k in range(3))
            U[i, j] = -(term1 + term2 + term3)
    return U


# ====================================================================
# Integrated mollified Kelvin force-stress kernel
# ====================================================================

def analytical_kelvin_stress(obs, v1, v2, v3, mu, nu, eps):
    """
    Analytically integrated stress kernel for a triangular constant-
    force element (single-layer / Kelvin force kernel).

    S[i,j,k] gives the i,j stress at obs due to a unit constant force
    in direction k applied per unit area over the triangle:

        sigma_ij(obs) = S[i,j,k] * f_k

    Derivation:
        sigma_ij(x) for a point force f_k at y has displacement
        u_a(x) = G_ak(x,y) f_k. Stress is
        sigma_ij = C_ijab ε_ab(u) = C_ijab ∂G_ak/∂x_b · f_k

        Integrated over the triangle:
            S[i,j,k] = C_ijab ∫∫ ∂G_ak/∂x_b dA
                     = C_ijab G1[a,k,b]
                     = λ δ_ij Σ_a G1[a,k,a]
                       + μ G1[i,k,j]
                       + μ G1[j,k,i]

    Returns (3,3,3) tensor.
    """
    G1 = integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

    S = np.zeros((3, 3, 3))
    for k in range(3):
        # trace_k = Σ_a G1[a,k,a]
        trace_k = sum(G1[a, k, a] for a in range(3))
        for i in range(3):
            for j in range(3):
                S[i, j, k] = (lam * (i == j) * trace_k
                              + mu * G1[i, k, j]
                              + mu * G1[j, k, i])
    return S


# ====================================================================
# Numerical integration for validation
# ====================================================================

def integrate_numerical(v1, v2, v3, obs, eps, n_quad=20):
    """Numerically integrate 1/R_eps^n over triangle."""
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts

    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    I1 = I3 = I5 = I7 = 0.0

    for i in range(n_quad):
        for j in range(n_quad):
            xi1 = pts[i]
            xi2 = pts[j] * (1.0 - pts[i])
            w = wts[i] * wts[j] * (1.0 - pts[i])

            x = (1 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3
            d = obs - x
            r2 = np.dot(d, d)
            re2 = r2 + eps**2
            re = np.sqrt(re2)
            I1 += w / re * area2
            I3 += w / (re * re2) * area2
            I5 += w / (re * re2**2) * area2
            I7 += w / (re * re2**3) * area2

    return {'I1': I1, 'I3': I3, 'I5': I5, 'I7': I7}


def integrate_moments_numerical(v1, v2, v3, obs, eps, n_quad=20):
    """Numerically integrate moment integrals for validation."""
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts

    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    # Accumulators
    V3 = np.zeros(3)
    V5 = np.zeros(3)
    V7 = np.zeros(3)
    T2_5 = np.zeros((3, 3))
    T2_7 = np.zeros((3, 3))
    T4_7 = np.zeros((3, 3, 3, 3))

    for i in range(n_quad):
        for j in range(n_quad):
            xi1 = pts[i]
            xi2 = pts[j] * (1.0 - pts[i])
            w = wts[i] * wts[j] * (1.0 - pts[i])

            x = (1 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3
            d = obs - x
            r2 = np.dot(d, d)
            re2 = r2 + eps**2
            re = np.sqrt(re2)

            ire3 = 1.0 / (re * re2)
            ire5 = ire3 / re2
            ire7 = ire5 / re2

            wA = w * area2
            V3 += wA * d * ire3
            V5 += wA * d * ire5
            V7 += wA * d * ire7

            dd = np.outer(d, d)
            T2_5 += wA * dd * ire5
            T2_7 += wA * dd * ire7

            dddd = np.einsum('i,j,k,l->ijkl', d, d, d, d)
            T4_7 += wA * dddd * ire7

    return {'V3': V3, 'V5': V5, 'V7': V7,
            'T2_5': T2_5, 'T2_7': T2_7, 'T4_7': T4_7}


# ====================================================================
# Validation
# ====================================================================

def validate_base_integrals():
    """Compare analytical vs numerical base integrals."""
    print("=" * 70)
    print("VALIDATION 1: Base potential integrals")
    print("=" * 70)

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    test_cases = [
        ("Far field",           np.array([0.5, 0.3, 2.0]),  0.1),
        ("Near field",          np.array([0.3, 0.2, 0.5]),  0.1),
        ("On plane, mollified", np.array([0.3, 0.2, 0.0]),  0.2),
        ("Centroid, eps=0.1",   np.array([1/3, 1/3, 0.0]),  0.1),
        ("Outside triangle",    np.array([2.0, 2.0, 0.3]),  0.1),
        ("Below plane",         np.array([0.5, 0.5, -1.0]), 0.1),
        ("At vertex",           np.array([0.0, 0.0, 0.0]),  0.2),
        ("On edge",             np.array([0.5, 0.0, 0.0]),  0.2),
    ]

    all_ok = True
    for name, obs, eps in test_cases:
        ana = integrate_over_triangle(v1, v2, v3, obs, eps)
        num = integrate_numerical(v1, v2, v3, obs, eps, n_quad=30)

        print(f"\n  {name}: obs={obs}, eps={eps}")
        for key in ['I1', 'I3', 'I5', 'I7']:
            a = ana[key]
            n = num[key]
            err = abs(a - n) / abs(n) if abs(n) > 1e-12 else abs(a - n)
            ok = err < 1e-3
            if not ok:
                all_ok = False
            print(f"    {key}: ana={a:14.8e} num={n:14.8e} err={err:.2e} {'OK' if ok else '**FAIL**'}")

    return all_ok


def validate_moment_integrals():
    """Compare analytical vs numerical moment integrals."""
    print("\n" + "=" * 70)
    print("VALIDATION 2: Moment integrals (vector & tensor)")
    print("=" * 70)

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    test_cases = [
        ("Far field",           np.array([0.5, 0.3, 2.0]),  0.1),
        ("Near field",          np.array([0.3, 0.2, 0.5]),  0.1),
        ("On plane, mollified", np.array([0.3, 0.2, 0.0]),  0.2),
        ("Centroid, eps=0.1",   np.array([1/3, 1/3, 0.0]),  0.1),
        ("Outside triangle",    np.array([2.0, 2.0, 0.3]),  0.1),
        ("Below plane",         np.array([0.5, 0.5, -1.0]), 0.1),
    ]

    all_ok = True
    for name, obs, eps in test_cases:
        ana = integrate_all_moments(v1, v2, v3, obs, eps)
        num = integrate_moments_numerical(v1, v2, v3, obs, eps, n_quad=30)

        print(f"\n  {name}: obs={obs}, eps={eps}")

        # Check V3 (vector moment at R^3)
        for label, ana_val, num_val in [
            ('V3', ana['V'][3], num['V3']),
            ('V5', ana['V'][5], num['V5']),
            ('V7', ana['V'][7], num['V7']),
        ]:
            err = np.linalg.norm(ana_val - num_val)
            ref = np.linalg.norm(num_val)
            rel = err / ref if ref > 1e-12 else err
            ok = rel < 1e-3
            if not ok:
                all_ok = False
            print(f"    {label}: |err|={err:.2e} |ref|={ref:.2e} rel={rel:.2e} {'OK' if ok else '**FAIL**'}")

        # Check T2_5, T2_7
        for label, ana_val, num_val in [
            ('T2_5', ana['T2'][5], num['T2_5']),
            ('T2_7', ana['T2'][7], num['T2_7']),
        ]:
            err = np.linalg.norm(ana_val - num_val)
            ref = np.linalg.norm(num_val)
            rel = err / ref if ref > 1e-12 else err
            ok = rel < 1e-3
            if not ok:
                all_ok = False
            print(f"    {label}: |err|={err:.2e} |ref|={ref:.2e} rel={rel:.2e} {'OK' if ok else '**FAIL**'}")

        # Check T4_7
        err = np.linalg.norm(ana['T4'][7] - num['T4_7'])
        ref = np.linalg.norm(num['T4_7'])
        rel = err / ref if ref > 1e-12 else err
        ok = rel < 1e-3
        if not ok:
            all_ok = False
        print(f"    T4_7: |err|={err:.2e} |ref|={ref:.2e} rel={rel:.2e} {'OK' if ok else '**FAIL**'}")

    return all_ok


def kelvin_dG_pointwise(d, mu, nu, eps):
    """
    Pointwise first derivative of the mollified Kelvin G:
        DG[i,j,m] = ∂G_ij/∂x_m  evaluated at d = obs - source.

    Uses the Galerkin/Cortez-convolved form of G^eps so the gradient
    includes the -6(1-ν)ε² δ_ij d_m/R^5 contribution.

    Returns (3,3,3) tensor. Used by the numerical references for
    validating analytical_dd_displacement and analytical_kelvin_stress.
    """
    r2 = d[0]**2 + d[1]**2 + d[2]**2
    re2 = r2 + eps**2
    re = np.sqrt(re2)
    ire3 = 1.0 / (re * re2)
    ire5 = ire3 / re2

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob_dg = 6.0 * (1.0 - nu) * eps**2

    DG = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                DG[i, j, m] = C1 * (
                    -c34 * (i == j) * d[m] * ire3
                    + (i == m) * d[j] * ire3
                    + (j == m) * d[i] * ire3
                    - 3.0 * d[i] * d[j] * d[m] * ire5
                    - c_blob_dg * (i == j) * d[m] * ire5
                )
    return DG


def integrate_dd_displacement_numerical(obs, v1, v2, v3, normal,
                                         mu, nu, eps, n_quad=20):
    """Numerical reference for analytical_dd_displacement."""
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts

    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

    U = np.zeros((3, 3))
    for ii in range(n_quad):
        for jj in range(n_quad):
            xi1 = pts[ii]
            xi2 = pts[jj] * (1.0 - pts[ii])
            w = wts[ii] * wts[jj] * (1.0 - pts[ii])

            y = (1 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3
            d = obs - y
            DG = kelvin_dG_pointwise(d, mu, nu, eps)

            # u_i per unit Δu_j contribution at this quad point:
            #   -[ λ Σ_m n_m DG[i,j,m]
            #    + μ n_j Σ_m DG[i,m,m]
            #    + μ Σ_k n_k DG[i,k,j] ]
            for i in range(3):
                trace_im = sum(DG[i, m, m] for m in range(3))
                for j in range(3):
                    t1 = lam * sum(normal[m] * DG[i, j, m] for m in range(3))
                    t2 = mu * normal[j] * trace_im
                    t3 = mu * sum(normal[k] * DG[i, k, j] for k in range(3))
                    U[i, j] -= w * area2 * (t1 + t2 + t3)
    return U


def integrate_kelvin_stress_numerical(obs, v1, v2, v3, mu, nu, eps, n_quad=20):
    """Numerical reference for analytical_kelvin_stress."""
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts

    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

    S = np.zeros((3, 3, 3))
    for ii in range(n_quad):
        for jj in range(n_quad):
            xi1 = pts[ii]
            xi2 = pts[jj] * (1.0 - pts[ii])
            w = wts[ii] * wts[jj] * (1.0 - pts[ii])

            y = (1 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3
            d = obs - y
            DG = kelvin_dG_pointwise(d, mu, nu, eps)

            # S[i,j,k] = λ δ_ij Σ_a DG[a,k,a] + μ DG[i,k,j] + μ DG[j,k,i]
            for k in range(3):
                trace_k = sum(DG[a, k, a] for a in range(3))
                for i in range(3):
                    for j in range(3):
                        S[i, j, k] += w * area2 * (
                            lam * (i == j) * trace_k
                            + mu * DG[i, k, j]
                            + mu * DG[j, k, i]
                        )
    return S


def integrate_kelvin_G_numerical(obs, v1, v2, v3, mu, nu, eps, n_quad=20):
    """
    Numerically integrate the mollified Kelvin G_ij over a triangle
    via a tensor-product Gauss-Legendre rule on the unit triangle.

    Uses the Galerkin/Cortez-convolved form including the
    2(1-ν) ε² δ_ij/R^3 term.

    Reference for validating analytical_kelvin_G.
    """
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts

    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps**2

    G_int = np.zeros((3, 3))
    I3x3 = np.eye(3)

    for i in range(n_quad):
        for j in range(n_quad):
            xi1 = pts[i]
            xi2 = pts[j] * (1.0 - pts[i])
            w = wts[i] * wts[j] * (1.0 - pts[i])

            y = (1 - xi1 - xi2) * v1 + xi1 * v2 + xi2 * v3
            d = obs - y
            r2 = np.dot(d, d)
            re2 = r2 + eps**2
            re = np.sqrt(re2)
            re3 = re * re2

            G = C1 * (
                c34 / re * I3x3
                + np.outer(d, d) / re3
                + (c_blob / re3) * I3x3
            )
            G_int += w * area2 * G

    return G_int


def validate_kelvin_kernel():
    """Compare analytical vs numerical integrated Kelvin G kernel."""
    print("\n" + "=" * 70)
    print("VALIDATION 4: Integrated mollified Kelvin G (force kernel)")
    print("=" * 70)

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    mu = 1.0
    nu = 0.25

    test_cases = [
        ("Far field",           np.array([0.5, 0.3, 2.0]),  0.1),
        ("Near field",          np.array([0.3, 0.2, 0.5]),  0.1),
        ("On plane, mollified", np.array([0.3, 0.2, 0.0]),  0.2),
        ("Centroid, eps=0.1",   np.array([1/3, 1/3, 0.0]),  0.1),
        ("Outside triangle",    np.array([2.0, 2.0, 0.3]),  0.1),
        ("Below plane",         np.array([0.5, 0.5, -1.0]), 0.1),
        ("Self at centroid",    np.array([1/3, 1/3, 0.0]),  0.05),
    ]

    all_ok = True
    for name, obs, eps in test_cases:
        G_ana = analytical_kelvin_G(obs, v1, v2, v3, mu, nu, eps)
        G_num = integrate_kelvin_G_numerical(obs, v1, v2, v3, mu, nu, eps, n_quad=30)

        err = np.linalg.norm(G_ana - G_num)
        ref = np.linalg.norm(G_num)
        rel = err / ref if ref > 1e-12 else err
        ok = rel < 1e-3
        if not ok:
            all_ok = False

        print(f"\n  {name}: obs={obs}, eps={eps}")
        print(f"    |G_ana|={np.linalg.norm(G_ana):.6e}  "
              f"|G_num|={np.linalg.norm(G_num):.6e}")
        print(f"    |diff|={err:.2e}  rel={rel:.2e}  "
              f"{'OK' if ok else '**FAIL**'}")

        # Show diagonal components and one off-diagonal
        for ii, jj, lbl in [(0, 0, 'G11'), (1, 1, 'G22'),
                            (2, 2, 'G33'), (0, 2, 'G13')]:
            a = G_ana[ii, jj]
            n = G_num[ii, jj]
            ce = abs(a - n) / abs(n) if abs(n) > 1e-12 else abs(a - n)
            print(f"      {lbl}: ana={a:12.6e} num={n:12.6e} err={ce:.2e}")

    return all_ok


def validate_dd_displacement():
    """Compare analytical vs numerical DD displacement kernel."""
    print("\n" + "=" * 70)
    print("VALIDATION 5: Integrated DD displacement kernel")
    print("=" * 70)

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    mu = 1.0
    nu = 0.25

    test_cases = [
        ("Far field",           np.array([0.5, 0.3, 2.0]),  0.1),
        ("Near field",          np.array([0.3, 0.2, 0.5]),  0.1),
        ("On plane, mollified", np.array([0.3, 0.2, 0.0]),  0.2),
        ("Outside triangle",    np.array([2.0, 2.0, 0.3]),  0.1),
        ("Below plane",         np.array([0.5, 0.5, -1.0]), 0.1),
    ]

    all_ok = True
    for name, obs, eps in test_cases:
        U_ana = analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps)
        U_num = integrate_dd_displacement_numerical(
            obs, v1, v2, v3, normal, mu, nu, eps, n_quad=30
        )

        err = np.linalg.norm(U_ana - U_num)
        ref = np.linalg.norm(U_num)
        rel = err / ref if ref > 1e-12 else err
        ok = rel < 1e-3
        if not ok:
            all_ok = False

        print(f"\n  {name}: obs={obs}, eps={eps}")
        print(f"    |U_ana|={np.linalg.norm(U_ana):.6e}  "
              f"|U_num|={np.linalg.norm(U_num):.6e}")
        print(f"    |diff|={err:.2e}  rel={rel:.2e}  "
              f"{'OK' if ok else '**FAIL**'}")

        for ii, jj, lbl in [(0, 0, 'U₁/Δu₁'), (1, 1, 'U₂/Δu₂'),
                            (2, 2, 'U₃/Δu₃'), (0, 2, 'U₁/Δu₃')]:
            a = U_ana[ii, jj]
            n = U_num[ii, jj]
            ce = abs(a - n) / abs(n) if abs(n) > 1e-12 else abs(a - n)
            print(f"      {lbl}: ana={a:12.6e} num={n:12.6e} err={ce:.2e}")

    return all_ok


def validate_kelvin_stress_kernel():
    """Compare analytical vs numerical Kelvin force-stress kernel."""
    print("\n" + "=" * 70)
    print("VALIDATION 6: Integrated Kelvin force-stress kernel")
    print("=" * 70)

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    mu = 1.0
    nu = 0.25

    test_cases = [
        ("Far field",           np.array([0.5, 0.3, 2.0]),  0.1),
        ("Near field",          np.array([0.3, 0.2, 0.5]),  0.1),
        ("On plane, mollified", np.array([0.3, 0.2, 0.0]),  0.2),
        ("Outside triangle",    np.array([2.0, 2.0, 0.3]),  0.1),
        ("Below plane",         np.array([0.5, 0.5, -1.0]), 0.1),
    ]

    all_ok = True
    for name, obs, eps in test_cases:
        S_ana = analytical_kelvin_stress(obs, v1, v2, v3, mu, nu, eps)
        S_num = integrate_kelvin_stress_numerical(
            obs, v1, v2, v3, mu, nu, eps, n_quad=30
        )

        err = np.linalg.norm(S_ana - S_num)
        ref = np.linalg.norm(S_num)
        rel = err / ref if ref > 1e-12 else err
        ok = rel < 1e-3
        if not ok:
            all_ok = False

        print(f"\n  {name}: obs={obs}, eps={eps}")
        print(f"    |S_ana|={np.linalg.norm(S_ana):.6e}  "
              f"|S_num|={np.linalg.norm(S_num):.6e}")
        print(f"    |diff|={err:.2e}  rel={rel:.2e}  "
              f"{'OK' if ok else '**FAIL**'}")

        for ii, jj, kk, lbl in [(0, 0, 0, 'σ₁₁/f₁'),
                                (2, 2, 2, 'σ₃₃/f₃'),
                                (0, 2, 0, 'σ₁₃/f₁')]:
            a = S_ana[ii, jj, kk]
            n = S_num[ii, jj, kk]
            ce = abs(a - n) / abs(n) if abs(n) > 1e-12 else abs(a - n)
            print(f"      {lbl}: ana={a:12.6e} num={n:12.6e} err={ce:.2e}")

    return all_ok


def validate_stress_kernel():
    """Compare analytical vs numerical DD stress kernel."""
    print("\n" + "=" * 70)
    print("VALIDATION 3: Full DD stress kernel")
    print("=" * 70)

    # Import numerical kernel for comparison
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mollified_elastic_kernels import integrate_stress_kernel

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    mu = 1.0
    nu = 0.25

    test_cases = [
        ("Far field",           np.array([0.5, 0.3, 2.0]),  0.1),
        ("Near field",          np.array([0.3, 0.2, 0.5]),  0.1),
        ("On plane, mollified", np.array([0.3, 0.2, 0.0]),  0.2),
        ("Centroid, eps=0.1",   np.array([1/3, 1/3, 0.0]),  0.1),
        ("Outside triangle",    np.array([2.0, 2.0, 0.3]),  0.1),
        ("Self-stress at centroid", np.array([1/3, 1/3, 0.0]), 0.05),
    ]

    all_ok = True
    for name, obs, eps in test_cases:
        H_ana = analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps)
        H_num = integrate_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=20)

        err = np.linalg.norm(H_ana - H_num)
        ref = np.linalg.norm(H_num)
        rel = err / ref if ref > 1e-12 else err
        ok = rel < 1e-2
        if not ok:
            all_ok = False

        print(f"\n  {name}: obs={obs}, eps={eps}")
        print(f"    |H_ana|={np.linalg.norm(H_ana):.6e}  |H_num|={np.linalg.norm(H_num):.6e}")
        print(f"    |diff|={err:.2e}  rel={rel:.2e}  {'OK' if ok else '**FAIL**'}")

        # Show component-wise for key stress components
        labels = [(0,2,0, 'σ₁₃/Δu₁'), (1,2,1, 'σ₂₃/Δu₂'), (2,2,2, 'σ₃₃/Δu₃')]
        for m, nn, k, lbl in labels:
            a = H_ana[m, nn, k]
            n = H_num[m, nn, k]
            ce = abs(a - n) / abs(n) if abs(n) > 1e-12 else abs(a - n)
            print(f"      {lbl}: ana={a:12.6e} num={n:12.6e} err={ce:.2e}")

    return all_ok


def validate():
    """Run all validations."""
    ok1 = validate_base_integrals()
    ok2 = validate_moment_integrals()
    ok3 = validate_stress_kernel()
    ok4 = validate_kelvin_kernel()
    ok5 = validate_dd_displacement()
    ok6 = validate_kelvin_stress_kernel()

    print("\n" + "=" * 70)
    print(f"  FINAL: Base={'PASS' if ok1 else 'FAIL'}  "
          f"Moments={'PASS' if ok2 else 'FAIL'}  "
          f"DDStress={'PASS' if ok3 else 'FAIL'}  "
          f"KelvinG={'PASS' if ok4 else 'FAIL'}  "
          f"DDDisp={'PASS' if ok5 else 'FAIL'}  "
          f"KelvinStress={'PASS' if ok6 else 'FAIL'}")
    print("=" * 70)
    return ok1 and ok2 and ok3 and ok4 and ok5 and ok6


if __name__ == '__main__':
    validate()

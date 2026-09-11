#!/usr/bin/env python3
"""
mollified_elastic_kernels.py
============================

3D Mollified Elastic Kernels for Quasi-Static Boundary Element Methods

Implements Cortez-style blob regularization of the Kelvin (elastic
fundamental) solution for 3D isotropic elasticity. The singular
displacement-discontinuity stress kernel (O(1/r^3)) is replaced with
a smooth, bounded function via:  r -> r_eps = sqrt(r^2 + eps^2)

This enables direct numerical quadrature of ALL boundary integrals,
including the hypersingular self-interaction terms.

The naive replacement r -> r_eps corresponds to the blob function:
    phi_eps(r) = 3*eps^2 / (4*pi*(r^2 + eps^2)^(5/2))
which integrates to unity and converges to delta(r) as eps -> 0.

Reference:
    Cortez R (2001) The Method of Regularized Stokeslets.
    SIAM J. Sci. Comput. 23(4):1204-1225.
    DOI: 10.1137/S106482750038146X

Usage:
    python mollified_elastic_kernels.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ========================================================================
# SECTION 1: Regularized Kelvin Solution and Stress Kernel
# ========================================================================


def kelvin_d2G(d, mu, nu, eps):
    """
    Second derivative tensor of the regularized Kelvin solution.

    G_rp^eps = C1 * [(3-4*nu)*delta_rp/r_eps
                     + d_r*d_p/r_eps^3
                     + 2*(1-nu)*eps^2*delta_rp/r_eps^3]

    The third term is the Cortez-blob convolution contribution: it makes
    G^eps the exact elastostatic displacement of a point force smeared by
    phi^(C)(R) = 15*eps^4 / (8*pi*r_eps^7), so that the regularized
    Cauchy-Navier equation L_ij G^eps_jk = -delta_ik phi^(C) holds exactly.

    Returns D2G[r,p,s,q] = d^2 G_rp / (d d_s  d d_q)
    where d = y - x (observation minus source), r_eps = sqrt(|d|^2 + eps^2).

    This is the core tensor for computing the stress kernel.
    Singular: O(1/r^5) terms.  Regularized: O(1/eps^5), bounded.

    Parameters
    ----------
    d   : (3,) vector from source to observation point
    mu  : shear modulus
    nu  : Poisson's ratio
    eps : mollification parameter (>0 for regularized, 0 for singular)

    Returns
    -------
    D2G : (3,3,3,3) tensor
    """
    r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
    re2 = r2 + eps**2
    re = np.sqrt(re2)
    ire3 = 1.0 / (re * re2)
    ire5 = ire3 / re2
    ire7 = ire5 / re2

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps**2

    # Build tensor component by component (clear, correct)
    D2G = np.zeros((3, 3, 3, 3))
    for r in range(3):
        for p in range(3):
            for s in range(3):
                for q in range(3):
                    D2G[r, p, s, q] = C1 * (
                        # From d^2/dd_s dd_q of [(3-4nu)*delta_rp / r_eps]
                        -c34 * (r == p) * ((s == q) * ire3 - 3.0 * d[s] * d[q] * ire5)
                        # From d^2/dd_s dd_q of [d_r * d_p / r_eps^3]
                        + (r == s) * ((p == q) * ire3 - 3.0 * d[p] * d[q] * ire5)
                        + (p == s) * ((r == q) * ire3 - 3.0 * d[r] * d[q] * ire5)
                        - 3.0
                        * (
                            (r == q) * d[p] * d[s]
                            + d[r] * (p == q) * d[s]
                            + d[r] * d[p] * (s == q)
                        )
                        * ire5
                        + 15.0 * d[r] * d[p] * d[s] * d[q] * ire7
                        # From d^2/dd_s dd_q of [2(1-nu)*eps^2*delta_rp / r_eps^3]
                        + c_blob
                        * (r == p)
                        * (-3.0 * (s == q) * ire5 + 15.0 * d[s] * d[q] * ire7)
                    )
    return D2G


def dd_stress_kernel(y, x, normal, mu, nu, eps):
    """
    Displacement discontinuity stress kernel (regularized).

    Computes K[m,n,k] such that the stress at observation point y
    due to a point displacement discontinuity Delta_u at source x
    on a surface with outward normal is:

        sigma_mn(y) = K[m,n,k] * Delta_u_k

    From the representation theorem:
        K_mn,k = -C_mnrs * C_kjpq * nu_j * D2G[r,p,s,q]

    where C_ijkl is the isotropic stiffness tensor.

    Parameters
    ----------
    y      : (3,) observation point
    x      : (3,) source point on fault surface
    normal : (3,) outward unit normal at source
    mu     : shear modulus
    nu     : Poisson's ratio
    eps    : mollification parameter

    Returns
    -------
    K : (3,3,3) tensor.  K[m,n,k] = sigma_mn per unit slip in direction k.
    """
    d = y - x
    D2G = kelvin_d2G(d, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal

    # Efficient contraction in 3 steps:
    #
    # Step 1: B[r,s,k] = sum_{j,p,q} C_kjpq * n_j * D2G[r,p,s,q]
    #
    # For isotropic C_kjpq = lam*d_kj*d_pq + mu*(d_kp*d_jq + d_kq*d_jp):
    #   B[r,s,k] = lam*n_k*trace_D2G[r,s] + mu*D2G_nq[r,k,s] + mu*D2G_np[r,s,k]

    trace_D2G = np.zeros((3, 3))
    for r in range(3):
        for s in range(3):
            trace_D2G[r, s] = sum(D2G[r, p, s, p] for p in range(3))

    D2G_nq = np.zeros((3, 3, 3))  # D2G contracted with normal on 4th index
    D2G_np = np.zeros((3, 3, 3))  # D2G contracted with normal on 2nd index
    for r in range(3):
        for i in range(3):
            for s in range(3):
                D2G_nq[r, i, s] = sum(n[q] * D2G[r, i, s, q] for q in range(3))
                D2G_np[r, s, i] = sum(n[p] * D2G[r, p, s, i] for p in range(3))

    B = np.zeros((3, 3, 3))
    for k in range(3):
        B[:, :, k] = (
            lam * n[k] * trace_D2G + mu * D2G_nq[:, k, :] + mu * D2G_np[:, :, k]
        )

    # Step 2: K[m,n,k] = -C_mnrs * B[r,s,k]
    # For isotropic C_mnrs = lam*d_mn*d_rs + mu*(d_mr*d_ns + d_ms*d_nr):
    #   K[m,n,k] = -(lam*(m==n)*trace_B[k] + mu*(B[m,n,k] + B[n,m,k]))

    trace_B = np.array([sum(B[r, r, k] for r in range(3)) for k in range(3)])

    K = np.zeros((3, 3, 3))
    for k in range(3):
        for m in range(3):
            for nn in range(3):
                K[m, nn, k] = -(
                    lam * (m == nn) * trace_B[k] + mu * (B[m, nn, k] + B[nn, m, k])
                )
    return K


# ========================================================================
# SECTION 2: Triangular Element Integration
# ========================================================================


def triangle_quadrature(n_per_side):
    """
    Gaussian quadrature over reference triangle with vertices
    (0,0), (1,0), (0,1).

    Uses collapsed tensor product rule: maps unit square to triangle
    via xi2 -> xi2*(1-xi1), with Jacobian (1-xi1).

    Parameters
    ----------
    n_per_side : number of Gauss points per direction (total = n^2)

    Returns
    -------
    xi1, xi2, weights : arrays of length n_per_side^2
    """
    pts, wts = np.polynomial.legendre.leggauss(n_per_side)
    pts = 0.5 * (pts + 1.0)  # Map [-1,1] -> [0,1]
    wts = 0.5 * wts

    xi1 = np.zeros(n_per_side**2)
    xi2 = np.zeros(n_per_side**2)
    weights = np.zeros(n_per_side**2)

    idx = 0
    for i in range(n_per_side):
        for j in range(n_per_side):
            xi1[idx] = pts[i]
            xi2[idx] = pts[j] * (1.0 - pts[i])
            weights[idx] = wts[i] * wts[j] * (1.0 - pts[i])
            idx += 1

    return xi1, xi2, weights


def integrate_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad):
    """
    Integrate the DD stress kernel over a triangular element.

    H[m,n,k] = integral_triangle K[m,n,k](obs, x, normal) dA(x)

    The physical triangle has vertices v1, v2, v3.
    Mapping from reference: x = (1-xi1-xi2)*v1 + xi1*v2 + xi2*v3

    Parameters
    ----------
    obs    : (3,) observation point
    v1,v2,v3 : (3,) triangle vertices
    normal : (3,) unit normal
    mu,nu  : elastic parameters
    eps    : mollification parameter
    n_quad : quadrature order per side (total points = n_quad^2)

    Returns
    -------
    H : (3,3,3) integrated stress kernel
        sigma_mn = H[m,n,k] * Delta_u_k
    """
    xi1, xi2, wts = triangle_quadrature(n_quad)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))  # = 2 * area

    H = np.zeros((3, 3, 3))
    for i in range(len(wts)):
        # Physical coordinates of quadrature point
        x = (1.0 - xi1[i] - xi2[i]) * v1 + xi1[i] * v2 + xi2[i] * v3
        K = dd_stress_kernel(obs, x, normal, mu, nu, eps)
        H += wts[i] * K * area2

    return H


# ========================================================================
# SECTION 3: Validation and Demonstration
# ========================================================================


def make_test_triangle(scale=1.0):
    """
    Create an equilateral triangle in the x1-x2 plane centered at origin.
    Returns vertices and unit normal.
    """
    v1 = scale * np.array([0.0, 2.0 / np.sqrt(3.0), 0.0])
    v2 = scale * np.array([-1.0, -1.0 / np.sqrt(3.0), 0.0])
    v3 = scale * np.array([1.0, -1.0 / np.sqrt(3.0), 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    return v1, v2, v3, normal


def test_external_point(mu, nu):
    """
    Test 1: Regularized kernel convergence at an external point.
    As eps -> 0, the regularized integral should approach the singular one.
    We use a very small eps as the 'reference' solution.
    """
    print("=" * 70)
    print("TEST 1: External point convergence (eps -> 0)")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    obs = np.array([0.5, 0.3, 2.0])  # Well above the element
    n_quad = 8

    # Reference: use very small eps
    eps_ref = 1e-10
    H_ref = integrate_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps_ref, n_quad)

    # A representative component for display
    ref_val = H_ref[2, 2, 0]  # sigma_33 due to slip in x1 direction
    print(f"\n  Reference (eps=1e-10): sigma_33 kernel = {ref_val:.10e}")
    print(f"\n  {'eps':>12s}  {'sigma_33 kernel':>18s}  {'relative error':>16s}")
    print(f"  {'-' * 12}  {'-' * 18}  {'-' * 16}")

    for eps in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        H = integrate_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
        val = H[2, 2, 0]
        err = abs(val - ref_val) / (abs(ref_val) + 1e-30)
        print(f"  {eps:12.4e}  {val:18.10e}  {err:16.8e}")

    print()


def test_self_interaction(mu, nu):
    """
    Test 2: Self-interaction at the element centroid.
    This is the key test — the hypersingular integral that is normally
    undefined becomes a regular, finite integral with mollification.
    """
    print("=" * 70)
    print("TEST 2: Self-interaction (observation at element centroid)")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    centroid = (v1 + v2 + v3) / 3.0

    print(f"\n  Triangle vertices:")
    print(f"    v1 = {v1}")
    print(f"    v2 = {v2}")
    print(f"    v3 = {v3}")
    print(f"  Centroid = {centroid}")
    print(f"  Normal   = {normal}")

    print(f"\n  Self-stress tensor H[m,n,k] for various eps values:")
    print(f"  (showing sigma_12 component due to strike-slip Δu_1)\n")

    n_quad = 12

    eps_values = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    results = {}

    print(f"  {'eps':>10s}  {'H[0,2,0]':>14s}  {'H[1,2,1]':>14s}  {'H[2,2,2]':>14s}")
    print(
        f"  {'':>10s}  {'(sig13/du1)':>14s}  {'(sig23/du2)':>14s}  {'(sig33/du3)':>14s}"
    )
    print(f"  {'-' * 10}  {'-' * 14}  {'-' * 14}  {'-' * 14}")

    for eps in eps_values:
        H = integrate_stress_kernel(centroid, v1, v2, v3, normal, mu, nu, eps, n_quad)
        results[eps] = H
        print(
            f"  {eps:10.4f}  {H[0, 2, 0]:14.6e}  {H[1, 2, 1]:14.6e}  {H[2, 2, 2]:14.6e}"
        )

    print()
    return results


def test_quadrature_convergence(mu, nu):
    """
    Test 3: Quadrature convergence for the self-interaction.
    For fixed eps, the integrand is smooth, so we expect rapid convergence.
    """
    print("=" * 70)
    print("TEST 3: Quadrature convergence for self-interaction (fixed eps)")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    centroid = (v1 + v2 + v3) / 3.0
    eps = 0.3

    print(f"\n  eps = {eps}")
    print(f"\n  {'n_quad':>8s}  {'n_points':>10s}  {'H[0,2,0]':>16s}  {'change':>14s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 16}  {'-' * 14}")

    prev = None
    for n_quad in [4, 6, 8, 10, 12, 16, 20]:
        H = integrate_stress_kernel(centroid, v1, v2, v3, normal, mu, nu, eps, n_quad)
        val = H[0, 2, 0]
        n_pts = n_quad**2
        if prev is not None:
            change = abs(val - prev)
            print(f"  {n_quad:8d}  {n_pts:10d}  {val:16.10e}  {change:14.2e}")
        else:
            print(f"  {n_quad:8d}  {n_pts:10d}  {val:16.10e}  {'---':>14s}")
        prev = val

    print()


def test_richardson_extrapolation(mu, nu):
    """
    Test 4: Richardson extrapolation on eps.

    If the regularization error is O(eps^2), then evaluating at
    eps and eps/2 and extrapolating gives O(eps^4) accuracy.

    sigma(eps) = sigma_true + a2*eps^2 + a4*eps^4 + ...

    Richardson order 1: (4*sigma(eps/2) - sigma(eps)) / 3  -> O(eps^4)
    Richardson order 2: eliminates eps^4 term -> O(eps^6)
    """
    print("=" * 70)
    print("TEST 4: Richardson extrapolation on eps")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    centroid = (v1 + v2 + v3) / 3.0
    n_quad = 16

    # Compute at a geometric sequence of eps values
    eps0 = 0.8
    n_levels = 7
    eps_vals = [eps0 / 2**k for k in range(n_levels)]

    print(f"\n  Computing self-interaction at {n_levels} eps values...")
    raw_results = []
    for eps in eps_vals:
        H = integrate_stress_kernel(centroid, v1, v2, v3, normal, mu, nu, eps, n_quad)
        raw_results.append(H[0, 2, 0])  # Track sigma_13 for strike-slip

    print(f"\n  Raw values (sigma_13 component — shear restoring stress):")
    print(f"  {'eps':>12s}  {'H[0,2,0]':>18s}")
    print(f"  {'-' * 12}  {'-' * 18}")
    for i, eps in enumerate(eps_vals):
        print(f"  {eps:12.6f}  {raw_results[i]:18.10e}")

    # Richardson extrapolation (assuming O(eps^2) leading error)
    print(f"\n  Richardson extrapolation (assuming O(eps^2) error):")

    # Level 0: raw values
    table = [raw_results[:]]

    # Level 1: eliminate eps^2 term
    for level in range(1, min(4, n_levels)):
        prev = table[level - 1]
        factor = 4**level  # For eps^(2*level) elimination
        new = []
        for i in range(len(prev) - 1):
            new.append((factor * prev[i + 1] - prev[i]) / (factor - 1))
        table.append(new)

    print(f"\n  Richardson table (diagonal = best estimates):")
    print(f"  {'Level':>7s}  {'Value':>18s}  {'Change from prev':>18s}")
    print(f"  {'-' * 7}  {'-' * 18}  {'-' * 18}")
    for level in range(len(table)):
        if len(table[level]) > 0:
            val = table[level][-1]  # Best estimate at this level
            if level > 0 and len(table[level - 1]) > 0:
                prev_val = table[level - 1][-1]
                change = abs(val - prev_val)
                print(f"  {level:7d}  {val:18.10e}  {change:18.10e}")
            else:
                print(f"  {level:7d}  {val:18.10e}  {'---':>18s}")

    # Print full Richardson table
    print(f"\n  Full Richardson table:")
    header = "  {:>12s}".format("eps")
    for level in range(len(table)):
        header += f"  {'R' + str(level):>16s}"
    print(header)
    print("  " + "-" * (14 + 18 * len(table)))

    for i in range(len(raw_results)):
        line = f"  {eps_vals[i]:12.6f}"
        for level in range(len(table)):
            if i < len(table[level]):
                line += f"  {table[level][i]:16.10e}"
            else:
                line += f"  {'':>16s}"
        print(line)

    print()
    return table


def test_off_plane_profile(mu, nu):
    """
    Test 5: Stress profile approaching the element from above.
    Shows how the mollified stress smoothly transitions through
    the element plane, vs. the singular stress which diverges.
    """
    print("=" * 70)
    print("TEST 5: Stress profile along normal through centroid")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    centroid = (v1 + v2 + v3) / 3.0
    n_quad = 12

    heights = np.concatenate(
        [
            -np.logspace(0, -2, 20)[::-1],
            [0.0],
            np.logspace(-2, 0, 20),
        ]
    )

    eps_vals = [0.5, 0.2, 0.1, 0.05]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for eps in eps_vals:
        stress_13 = []
        stress_33_open = []
        for h in heights:
            obs = centroid + h * normal
            H = integrate_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
            stress_13.append(H[0, 2, 0])  # sigma_13 from strike-slip
            stress_33_open.append(H[2, 2, 2])  # sigma_33 from opening

        axes[0].plot(heights, stress_13, "-", label=f"eps={eps}", linewidth=1.5)
        axes[1].plot(heights, stress_33_open, "-", label=f"eps={eps}", linewidth=1.5)

    for ax, title in zip(
        axes, ["sigma_13 (shear from strike-slip)", "sigma_33 (normal from opening)"]
    ):
        ax.set_xlabel("Height above element (x3)")
        ax.set_ylabel("Stress kernel value")
        ax.set_title(f"{title} per unit strike-slip")
        ax.legend(fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Mollified DD stress kernel along normal through centroid\n"
        "(unit equilateral triangle, mu=1, nu=0.25)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "stress_profile.png"), dpi=150)
    print("\n  Saved stress_profile.png")
    plt.close()


def test_convergence_plot(mu, nu):
    """
    Test 6: Convergence plot — self-interaction error vs eps.
    Uses Richardson-extrapolated value as 'truth'.
    """
    print("=" * 70)
    print("TEST 6: Convergence rate plot")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    centroid = (v1 + v2 + v3) / 3.0
    n_quad = 16

    # Compute reference via Richardson extrapolation from small eps
    eps_ref_vals = [0.01 / 2**k for k in range(5)]
    ref_raw = []
    for eps in eps_ref_vals:
        H = integrate_stress_kernel(centroid, v1, v2, v3, normal, mu, nu, eps, n_quad)
        ref_raw.append(H[0, 2, 0])  # sigma_13 from strike-slip

    # Two levels of Richardson
    r1 = [(4 * ref_raw[i + 1] - ref_raw[i]) / 3 for i in range(len(ref_raw) - 1)]
    r2 = [(16 * r1[i + 1] - r1[i]) / 15 for i in range(len(r1) - 1)]
    ref_val = r2[-1]
    print(f"\n  Reference value (Richardson-extrapolated): {ref_val:.12e}")

    # Compute at various eps
    eps_test = np.logspace(-2, 0, 25)
    errors = []
    values = []
    for eps in eps_test:
        H = integrate_stress_kernel(centroid, v1, v2, v3, normal, mu, nu, eps, n_quad)
        val = H[0, 2, 0]
        values.append(val)
        errors.append(abs(val - ref_val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Error vs eps
    ax1.loglog(eps_test, errors, "bo-", markersize=4, label="Regularization error")
    ax1.loglog(
        eps_test,
        0.5 * eps_test**2 * errors[0] / eps_test[0] ** 2,
        "r--",
        alpha=0.7,
        label=r"$O(\epsilon^2)$ reference slope",
    )
    ax1.set_xlabel(r"$\epsilon$ (mollification parameter)")
    ax1.set_ylabel("|H_eps - H_true|")
    ax1.set_title("Self-interaction error vs mollification parameter")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Value vs eps
    ax2.semilogx(eps_test, values, "bo-", markersize=4, label="Mollified value")
    ax2.axhline(y=ref_val, color="r", linestyle="--", label="Extrapolated true value")
    ax2.set_xlabel(r"$\epsilon$ (mollification parameter)")
    ax2.set_ylabel("H[0,2,0] (sigma_13 kernel)")
    ax2.set_title("Self-interaction kernel value vs epsilon")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Convergence of mollified self-interaction integral\n"
        "(equilateral triangle, centroid, mu=1, nu=0.25)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "convergence_plot.png"), dpi=150
    )
    print("  Saved convergence_plot.png")
    plt.close()


def print_full_self_stress_tensor(mu, nu, eps=0.2):
    """
    Print the complete self-interaction stress kernel tensor
    for a specific eps value.
    """
    print("=" * 70)
    print(f"FULL SELF-STRESS TENSOR (eps = {eps})")
    print("=" * 70)

    v1, v2, v3, normal = make_test_triangle()
    centroid = (v1 + v2 + v3) / 3.0
    n_quad = 14

    H = integrate_stress_kernel(centroid, v1, v2, v3, normal, mu, nu, eps, n_quad)

    slip_labels = ["Δu₁ (strike)", "Δu₂ (dip)", "Δu₃ (opening)"]
    stress_labels = ["σ₁₁", "σ₁₂", "σ₁₃", "σ₂₁", "σ₂₂", "σ₂₃", "σ₃₁", "σ₃₂", "σ₃₃"]

    for k in range(3):
        print(f"\n  Slip direction: {slip_labels[k]}")
        print(f"  {'':>6s}", end="")
        for n in range(3):
            print(f"  {'x' + str(n + 1):>14s}", end="")
        print()
        for m in range(3):
            print(f"  {'x' + str(m + 1):>6s}", end="")
            for n in range(3):
                print(f"  {H[m, n, k]:14.6e}", end="")
            print()

    # Verify symmetry of stress tensor
    print(f"\n  Stress tensor symmetry check (max |H[m,n,k] - H[n,m,k]|):")
    for k in range(3):
        max_asym = max(abs(H[m, n, k] - H[n, m, k]) for m in range(3) for n in range(3))
        print(f"    Slip direction {k + 1}: {max_asym:.2e}")

    print()


# ========================================================================
# MAIN
# ========================================================================


def main():
    """Run all demonstrations."""
    mu = 1.0  # Shear modulus (normalized)
    nu = 0.25  # Poisson's ratio

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  MOLLIFIED 3D ELASTIC KERNELS — Cortez-style Regularization        ║")
    print("║  Quasi-static elasticity, displacement discontinuity formulation   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(
        f"\n  Material: mu = {mu}, nu = {nu}, lambda = {2 * mu * nu / (1 - 2 * nu):.4f}"
    )
    print(f"  Element:  unit equilateral triangle in x1-x2 plane")
    print(f"  Normal:   [0, 0, 1] (x3 direction)\n")

    # Run tests
    test_external_point(mu, nu)
    test_self_interaction(mu, nu)
    test_quadrature_convergence(mu, nu)
    test_richardson_extrapolation(mu, nu)
    print_full_self_stress_tensor(mu, nu, eps=0.0001)
    test_off_plane_profile(mu, nu)
    test_convergence_plot(mu, nu)

    print("=" * 70)
    print("All tests completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

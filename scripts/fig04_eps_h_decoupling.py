"""Figure 4 — ε is decoupled from h with analytic per-triangle integration.

Verifies the elasticity analog of the central claim of Ferranti & Cortez
(2024), Sec. 1.3: with quadrature-based regularized kernels the
parameter ε must be tied to the discretization scale h (typically ε ~ h)
because the per-element quadrature error scales as a positive power of
(h/ε); whereas with *analytic* per-triangle integration there is no
such coupling and ε can be made independently small.

Test problem
------------
A unit-edge equilateral triangle (the "master" element) lies in the z = 0
plane, carries uniform strike-slip Δu = (1, 0, 0), and is observed at a
point just above its centroid, P = (0, 0, 0.05 L). We compute the full
3×3 stress tensor σ at P by

    σ_full(ε) = K_int(P; T, n, ε) Δu

via the closed-form analytic integral over T (`analytical_stress_kernel`).
Because T is flat and the slip is uniform, this value is the *exact*
integral of the (mollified) point kernel for any ε > 0.

We then subdivide T into N² congruent sub-triangles (edge h = L / N) and
sum per-sub-triangle contributions in two ways: analytic and Gauss
quadrature. The analytic sum recovers σ_full(ε) up to floating-point
round-off, *regardless of N or ε*; the quadrature sum carries an
additional error that depends on (h, ε, n_q).

Panels (side by side, legends inside at center left)
------
Left:  fixed h = L/8: error vs ε for analytic and several quadrature
       orders.
Right: fixed ε: error vs N (= L/h) at four values of ε for both
       methods.
"""
# medt_paper: copied from moss/manuscript/scripts/fig04_eps_h_decoupling.py; only paths, imports and
# output names were changed (figures go to manuscript/figures/, caches to cache/).

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _paper_style import set_paper_style  # noqa: E402
from mollified_kernel.analytical_kernels import (  # noqa: E402
    analytical_stress_kernel,
)
from mollified_kernel.mollified_elastic_kernels import (  # noqa: E402
    integrate_stress_kernel,
)

set_paper_style()
# this figure uses the standard matplotlib sans-serif face
import matplotlib as mpl
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})
np.seterr(divide="ignore", invalid="ignore")


def equilateral_triangle(L: float = 1.0):
    h = L * np.sqrt(3.0) / 2.0
    centroid_y = h / 3.0
    v1 = np.array([0.0, h - centroid_y, 0.0])
    v2 = np.array([-L / 2.0, -centroid_y, 0.0])
    v3 = np.array([L / 2.0, -centroid_y, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    return v1, v2, v3, n


def subdivide(v1, v2, v3, N):
    e2 = v2 - v1
    e3 = v3 - v1
    h = 1.0 / N
    for i in range(N):
        for j in range(N - i):
            p00 = v1 + i * h * e2 + j * h * e3
            p10 = v1 + (i + 1) * h * e2 + j * h * e3
            p01 = v1 + i * h * e2 + (j + 1) * h * e3
            yield (p00, p10, p01)
            if i + j < N - 1:
                p11 = v1 + (i + 1) * h * e2 + (j + 1) * h * e3
                yield (p10, p11, p01)


def stress_sum_analytic(P, tris, n_vec, mu, nu, eps):
    delta_u = np.array([1.0, 0.0, 0.0])
    H = np.zeros((3, 3, 3))
    for (a, b, c) in tris:
        H += analytical_stress_kernel(P, a, b, c, n_vec, mu, nu, eps)
    return np.einsum("mnk,k->mn", H, delta_u)


def stress_sum_quad(P, tris, n_vec, mu, nu, eps, n_quad):
    delta_u = np.array([1.0, 0.0, 0.0])
    H = np.zeros((3, 3, 3))
    for (a, b, c) in tris:
        H += integrate_stress_kernel(P, a, b, c, n_vec, mu, nu, eps, n_quad)
    return np.einsum("mnk,k->mn", H, delta_u)


CACHE = os.path.join(ROOT, "cache", "fig04_eps_h_decoupling.npz")


def main():
    L = 1.0
    mu, nu = 1.0, 0.25
    v1, v2, v3, n_vec = equilateral_triangle(L)
    P = np.array([0.0, 0.0, 0.05 * L])

    # Doubled-resolution sweeps:
    eps_grid = np.logspace(-3.0, 0.5, 48) * L     # was 24 points
    N_grid = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]   # was [1,2,4,8,16,32]
    quad_orders = [2, 4, 8]

    if os.path.exists(CACHE) and "--force" not in sys.argv:
        d = np.load(CACHE)
        eps_grid = d["eps_grid"]
        N_grid = list(d["N_grid"])
        err_analytic = d["err_analytic"]
        err_quad = {q: d[f"err_quad_{q}"] for q in quad_orders}
    else:
        print("computing reference σ_full(ε) ...")
        sig_full = np.zeros((len(eps_grid), 3, 3))
        for ie, eps in enumerate(eps_grid):
            sig_full[ie] = stress_sum_analytic(P, [(v1, v2, v3)], n_vec, mu, nu, eps)
        norm_full = np.array([np.linalg.norm(s) for s in sig_full])

        print("computing subdivision sums ...")
        err_analytic = np.zeros((len(N_grid), len(eps_grid)))
        err_quad = {q: np.zeros((len(N_grid), len(eps_grid))) for q in quad_orders}

        for iN, N in enumerate(N_grid):
            tris = list(subdivide(v1, v2, v3, N))
            n_tris = len(tris)
            print(f"  N={N}: {n_tris} sub-triangles")
            for ie, eps in enumerate(eps_grid):
                sa = stress_sum_analytic(P, tris, n_vec, mu, nu, eps)
                err_analytic[iN, ie] = (
                    np.linalg.norm(sa - sig_full[ie]) / norm_full[ie]
                )
                for q in quad_orders:
                    sq = stress_sum_quad(P, tris, n_vec, mu, nu, eps, q)
                    err_quad[q][iN, ie] = (
                        np.linalg.norm(sq - sig_full[ie]) / norm_full[ie]
                    )

        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, eps_grid=eps_grid, N_grid=np.asarray(N_grid),
                 err_analytic=err_analytic,
                 **{f"err_quad_{q}": err_quad[q] for q in quad_orders})

    # ---- Plot ----------------------------------------------------------
    # Side-by-side panels; each legend sits inside its own subplot at
    # center left, so no panel letters are needed (referred to in the
    # text as the left and right panels).
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.13,
                         wspace=0.28)

    floor = 1e-17

    # Upper panel: error vs ε at fixed N = 8.
    ax = axes[0]
    iN_a = N_grid.index(8)
    ax.loglog(eps_grid / L, np.maximum(err_analytic[iN_a], floor),
              "-o", color="black", lw=0.5, ms=1.875,
              label=r"analytic")
    quad_colors = {2: "#2ca02c", 4: "#1f77b4", 8: "#9467bd"}
    for q in quad_orders:
        ax.loglog(eps_grid / L, np.maximum(err_quad[q][iN_a], floor),
                   "-o", color=quad_colors[q], lw=0.5, ms=1.875,
                   label=rf"quadrature, $n_q={q}$")
    ax.set_xlabel(r"$\varepsilon \; / \; L$")
    ax.set_ylabel(r"relative error in $\sigma$")
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-18, 1e2)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticks([1e-18, 1e-14, 1e-10, 1e-6, 1e-2, 1e2])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_box_aspect(1)
    ax.legend(loc="center left", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="0.3", fontsize=7)

    # Right panel: error vs N at four ε.
    ax = axes[1]
    eps_plot = [1.0, 0.1, 0.01, 0.001]
    palette = ["#000000", "#1f77b4", "#d62728", "#ff7f0e"]
    inv_h = np.array(N_grid)
    for c, eps in zip(palette, eps_plot):
        ie = int(np.argmin(np.abs(eps_grid - eps * L)))
        ax.loglog(inv_h, np.maximum(err_analytic[:, ie], floor),
                   "-o", color=c, lw=0.5, ms=1.875,
                   label=rf"analytic, $\varepsilon \; / \; L = {eps:g}$")
        ax.loglog(inv_h, np.maximum(err_quad[4][:, ie], floor),
                   "--o", color=c, lw=0.5, ms=1.875, alpha=0.8,
                   label=rf"quadrature $n_q=4$, $\varepsilon \; / \; L = {eps:g}$")
    ax.set_xlabel(r"$N = L \; / \; h$")
    ax.set_ylabel(r"relative error in $\sigma$")
    ax.set_xlim(1, 64)
    ax.set_ylim(1e-18, 1e2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([1e-18, 1e-14, 1e-10, 1e-6, 1e-2, 1e2])
    ax.set_box_aspect(1)
    ax.legend(loc="center left", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="0.3", fontsize=7)

    out_dir = os.path.join(ROOT, "manuscript", "figures")
    fig.savefig(os.path.join(out_dir, "fig_eps_h_decoupling.pdf"))
    plt.close(fig)
    print("wrote manuscript/figures/fig_eps_h_decoupling.pdf")


if __name__ == "__main__":
    main()

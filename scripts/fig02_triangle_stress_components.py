"""Figure 2 — Stress field over a triangle: three integration strategies.

A unit-edge equilateral triangle in the z = 0 plane is loaded with unit
right-lateral strike-slip Δu = (1, 0, 0). On a horizontal evaluation
plane z = z₀ (slightly above the triangle) we compute the six
independent stress components on a grid by THREE methods, all displayed
with shared symmetric color limits per component:

  Row a (analytic, ε=0)        — closed-form integral of the SINGULAR
                                  kernel over the triangle. The kernel
                                  is hypersingular pointwise but the
                                  integrated field is bounded for any
                                  off-plane observation (the singular
                                  contribution at the source point is
                                  carried by an absolutely convergent
                                  solid-angle term + edge antideriva-
                                  tives). The closed-form is exact at
                                  ε=0.

  Row b (quadrature, ε=0)      — Gauss quadrature of the singular
                                  kernel, which is the textbook approach
                                  practitioners reach for first. Where
                                  the evaluation plane is near the
                                  triangle the kernel scales as 1/R⁵ and
                                  no fixed-order quadrature can resolve
                                  it; the panel shows the resulting
                                  speckle / blow-up.

  Row c (analytic, ε=0.1 L)    — closed-form integral of the *mollified*
                                  kernel. Smooth and bounded everywhere
                                  by construction.

Rows a and c look qualitatively identical (both bounded, both close to
the same limit field); the difference is what happens at ε exactly
zero — analytic integration handles it cleanly, quadrature does not.
"""
# medt_paper: copied from moss/manuscript/scripts/fig02_triangle_field.py; only paths, imports and
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

from _paper_style import (  # noqa: E402
    set_paper_style, text_size,
)
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


def main():
    L = 1.0
    mu, nu = 1.0, 0.25
    v1, v2, v3, n_vec = equilateral_triangle(L)

    n_grid = 201
    z0 = 0.05 * L
    eps_moll = 0.1 * L
    n_quad_singular = 8

    xs = np.linspace(-2.0 * L, 2.0 * L, n_grid)
    ys = np.linspace(-2.0 * L, 2.0 * L, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # Three rows, six stress components.
    sig_anal0 = np.zeros((6, n_grid, n_grid))   # row a: analytic, eps=0
    sig_quad0 = np.zeros((6, n_grid, n_grid))   # row b: quadrature, eps=0
    sig_moll = np.zeros((6, n_grid, n_grid))    # row c: analytic, eps=eps_moll

    delta_u = np.array([1.0, 0.0, 0.0])
    cmp_idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    cmp_lbl = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{zz}$",
               r"$\sigma_{xy}$", r"$\sigma_{xz}$", r"$\sigma_{yz}$"]

    print(f"Building {n_grid}x{n_grid} field for three methods ...")
    for j in range(n_grid):
        for i in range(n_grid):
            obs = np.array([X[j, i], Y[j, i], z0])
            # (a) analytic, eps = 0 (singular kernel, bounded by analytic
            # integration for off-plane observation)
            H_a0 = analytical_stress_kernel(
                obs, v1, v2, v3, n_vec, mu, nu, 0.0
            )
            # (b) Gauss quadrature, eps = 0 (singular)
            H_q0 = integrate_stress_kernel(
                obs, v1, v2, v3, n_vec, mu, nu, 0.0, n_quad_singular
            )
            # (c) analytic, eps = eps_moll (mollified)
            H_m = analytical_stress_kernel(
                obs, v1, v2, v3, n_vec, mu, nu, eps_moll
            )
            sa0 = np.einsum("mnk,k->mn", H_a0, delta_u)
            sq0 = np.einsum("mnk,k->mn", H_q0, delta_u)
            sm = np.einsum("mnk,k->mn", H_m, delta_u)
            for k, (a, b) in enumerate(cmp_idx):
                sig_anal0[k, j, i] = sa0[a, b]
                sig_quad0[k, j, i] = sq0[a, b]
                sig_moll[k, j, i] = sm[a, b]
        if j % 10 == 0:
            print(f"  row {j} / {n_grid}")

    # ---- Plot ----------------------------------------------------------
    fig, axes = plt.subplots(3, 6, figsize=text_size(4.6),
                              sharex=True, sharey=True,
                              gridspec_kw=dict(wspace=0.12, hspace=0.12))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08)

    tri_x = [v1[0], v2[0], v3[0], v1[0]]
    tri_y = [v1[1], v2[1], v3[1], v1[1]]

    row_data = [sig_anal0, sig_quad0, sig_moll]
    row_labels = [
        r"analytic, $\varepsilon = 0$",
        r"quadrature, $\varepsilon = 0$",
        rf"analytic, $\varepsilon \; / \; L = {eps_moll:g}$",
    ]

    for k in range(6):
        # vmax set by the mollified row (the well-behaved limit).
        vmax = float(np.nanpercentile(np.abs(sig_moll[k]), 99))
        vmax = max(vmax, 1e-6)

        for r, data in enumerate(row_data):
            ax = axes[r, k]
            ax.pcolormesh(
                X / L, Y / L, data[k],
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                shading="nearest", rasterized=True,
            )
            ax.plot([x / L for x in tri_x], [y / L for y in tri_y],
                     color="k", lw=0.6)
            ax.set_aspect("equal")
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])

            if r == 0:
                ax.set_title(cmp_lbl[k], fontsize=9)
            if k == 0:
                ax.set_ylabel(row_labels[r] + "\n" + r"$y \; / \; L$",
                                fontsize=7.5)
            if r == 2:
                ax.set_xlabel(r"$x \; / \; L$")

    out_dir = os.path.join(ROOT, "manuscript", "figures")
    fig.savefig(os.path.join(out_dir, "fig_triangle_stress_components.pdf"))
    plt.close(fig)
    print("wrote manuscript/figures/fig_triangle_stress_components.pdf")


if __name__ == "__main__":
    main()

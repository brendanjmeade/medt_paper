"""Figure 3 — Mollified DD stress field at decreasing ε.

Same geometry as Figure 2: a unit-edge equilateral triangle in the z = 0
plane loaded with unit right-lateral strike-slip Δu = (1, 0, 0). The six
independent stress components are evaluated on a horizontal plane z = z0
just above the triangle using *only* the mollified closed-form kernel,
and the regularization parameter is swept ε/L ∈ {1, 0.1, 0.01, 0.001}.

The point: as ε shrinks the analytic mollified field converges pointwise
to the (formally singular) limit, but remains *bounded* on the entire
evaluation plane — there is no near-element blow-up, in marked contrast
to the top row of Figure 2. The closed-form moment integrals depend on ε
only through the effective height h_eff = √(z² + ε²) of the in-plane
edge antiderivatives, which is well-defined for any ε > 0.
"""
# medt_paper: copied from moss/manuscript/scripts/fig03_eps_sweep.py; only paths, imports and
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

from _paper_style import set_paper_style, text_size  # noqa: E402
from mollified_kernel.analytical_kernels import (  # noqa: E402
    analytical_stress_kernel,
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
    eps_list = [1.0, 0.1, 0.01, 0.001]

    xs = np.linspace(-2.0 * L, 2.0 * L, n_grid)
    ys = np.linspace(-2.0 * L, 2.0 * L, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    delta_u = np.array([1.0, 0.0, 0.0])
    cmp_idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    cmp_lbl = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{zz}$",
               r"$\sigma_{xy}$", r"$\sigma_{xz}$", r"$\sigma_{yz}$"]

    n_eps = len(eps_list)
    sig = np.zeros((n_eps, 6, n_grid, n_grid))

    for e, eps in enumerate(eps_list):
        print(f"eps/L = {eps:g} ...")
        for j in range(n_grid):
            for i in range(n_grid):
                obs = np.array([X[j, i], Y[j, i], z0])
                H = analytical_stress_kernel(
                    obs, v1, v2, v3, n_vec, mu, nu, eps * L
                )
                s = np.einsum("mnk,k->mn", H, delta_u)
                for k, (a, b) in enumerate(cmp_idx):
                    sig[e, k, j, i] = s[a, b]
            if j % 20 == 0:
                print(f"  row {j} / {n_grid}")

    # Use the smallest-ε field (the "limit") to set a per-component vmax,
    # so all ε panels share a comparable color scale that does not depend
    # on the ε that happens to dominate.
    fig, axes = plt.subplots(n_eps, 6, figsize=text_size(5.6),
                              sharex=True, sharey=True,
                              gridspec_kw=dict(wspace=0.10, hspace=0.10))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)

    tri_x = [v1[0], v2[0], v3[0], v1[0]]
    tri_y = [v1[1], v2[1], v3[1], v1[1]]

    for k in range(6):
        # Common vmax across all ε rows for this component, set by the
        # smallest-ε field (the most "singular" limit we reach).
        vmax = float(np.nanpercentile(np.abs(sig[-1, k]), 99.5))
        vmax = max(vmax, 1e-6)

        for e, eps in enumerate(eps_list):
            ax = axes[e, k]
            ax.pcolormesh(
                X / L, Y / L, sig[e, k],
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

            if e == 0:
                ax.set_title(cmp_lbl[k], fontsize=9)
            if k == 0:
                ax.set_ylabel(rf"$\varepsilon \; / \; L = {eps:g}$" + "\n"
                                + r"$y \; / \; L$", fontsize=8)
            if e == n_eps - 1:
                ax.set_xlabel(r"$x \; / \; L$")

    out_dir = os.path.join(ROOT, "manuscript", "figures")
    fig.savefig(os.path.join(out_dir, "fig_eps_sweep_stress_field.pdf"))
    plt.close(fig)
    print("wrote manuscript/figures/fig_eps_sweep_stress_field.pdf")


if __name__ == "__main__":
    main()

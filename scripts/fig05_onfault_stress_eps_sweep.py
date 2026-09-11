"""Figure — On-fault elastic stress versus the mollification scale,
for the elementary unit-triangle source of Figures 1-4.

Unit equilateral triangle (edge L = 1) in the z = 0 plane, unit
strike slip Du = (1, 0, 0), mu = 1, nu = 1/4.  We evaluate sigma_xz
at the triangle centroid ON the fault plane and at fixed off-fault
stand-offs z0 / L in {1/4, 1, 10}, sweeping eps / L over three
decades:

  - on-fault ELASTIC (eigenstress subtracted): bounded and eps-stable,
    converging to a finite limit;
  - off-fault ELASTIC at each z0: eps-independent once eps < z0 (at
    z0 = 10 L the value changes by < 1% even at eps = L).

Only the elastic curves are plotted; the totals (which track
(3/4) mu s / eps on the fault and diverge as eps -> 0) are computed
and printed for verification but deliberately not shown.

The eigenstress is the exact finite-triangle blob convolution
sigma*_xz = mu s (delta_T * phi_eps), with the Cortez blob
phi_eps(r) = 15 eps^4 / (8 pi (r^2 + eps^2)^(7/2)) whose fault-normal
marginal is gamma_eps(z) = (3/4) eps^4 / (z^2 + eps^2)^(5/2).  The
radial part of the convolution is analytic; the boundary correction
is an angular integral over the distance to the triangle edge,
evaluated with 4000 rays (converged to ~1e-12 for this geometry).
"""
# medt_paper: copied from moss/manuscript/scripts/fig14_onfault_eps_sweep.py; only paths, imports and
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

set_paper_style()
# this figure uses the standard matplotlib sans-serif face
import matplotlib as mpl
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})
np.seterr(divide="ignore", invalid="ignore")

MU, NU = 1.0, 0.25
Z_ON = 1.0e-9        # tiny offset for the on-plane evaluation
Z_OFFS = [0.25, 1.0, 10.0]      # off-fault stand-offs z0 / L


def equilateral_triangle(L: float = 1.0):
    h = L * np.sqrt(3.0) / 2.0
    centroid_y = h / 3.0
    v1 = np.array([0.0, h - centroid_y, 0.0])
    v2 = np.array([-L / 2.0, -centroid_y, 0.0])
    v3 = np.array([L / 2.0, -centroid_y, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    return v1, v2, v3, n


def boundary_distance(verts, thetas):
    """Distance from the centroid (origin) to the triangle boundary
    along each ray direction theta."""
    P = np.full(len(thetas), np.inf)
    c, s = np.cos(thetas), np.sin(thetas)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ax, ay = verts[a][0], verts[a][1]
        bx, by = verts[b][0], verts[b][1]
        ex, ey = bx - ax, by - ay
        # solve t * (c, s) = (ax, ay) + u * (ex, ey), 0 <= u <= 1, t > 0
        det = c * (-ey) - s * (-ex)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * (-ey) - ay * (-ex)) / det
            u = (c * ay - s * ax) / det
        ok = (np.abs(det) > 1e-14) & (t > 0) & (u >= -1e-12) & (u <= 1 + 1e-12)
        P[ok] = np.minimum(P[ok], t[ok])
    return P


def eigenstress_xz(eps, z, P_theta):
    """Exact sigma*_xz = mu * s * (delta_T * phi_eps) at height z above
    the centroid: infinite-plane marginal minus the boundary term."""
    h2 = z * z + eps * eps
    marginal = 0.75 * eps ** 4 / h2 ** 2.5
    # boundary correction: (3 eps^4 / 8 pi) * int (P^2 + h2)^(-5/2) dtheta
    corr = (3.0 * eps ** 4 / (8.0 * np.pi)) * np.mean(
        (P_theta ** 2 + h2) ** -2.5) * 2.0 * np.pi
    return MU * 1.0 * (marginal - corr)


def main():
    v1, v2, v3, n_vec = equilateral_triangle(1.0)
    verts = [v1, v2, v3]
    delta_u = np.array([1.0, 0.0, 0.0])
    thetas = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)
    P_theta = boundary_distance(verts, thetas)

    eps_grid = np.logspace(-3.0, 0.0, 200)
    heights = [Z_ON] + Z_OFFS
    tot = {z: np.zeros_like(eps_grid) for z in heights}
    el = {z: np.zeros_like(eps_grid) for z in heights}

    for k, eps in enumerate(eps_grid):
        for z in heights:
            obs = np.array([0.0, 0.0, z])
            H = analytical_stress_kernel(obs, v1, v2, v3, n_vec,
                                         MU, NU, eps)
            s_tot = np.einsum("mnk,k->mn", H, delta_u)[0, 2]
            s_star = eigenstress_xz(eps, z, P_theta)
            tot[z][k] = s_tot
            el[z][k] = s_tot - s_star
        print(f"eps={eps:8.4f}: on-fault total {tot[Z_ON][k]:+10.4f} "
              f"(3/4/eps = {0.75 / eps:9.3f}), elastic {el[Z_ON][k]:+8.4f}; "
              + "; ".join(f"z0={z:g}: el {el[z][k]:+.3e}"
                          for z in Z_OFFS))

    # ---- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.13)

    ax.loglog(eps_grid, np.abs(el[Z_ON]), "-", color="#d62728", lw=0.5,
              label=r"$z_0 = 0$")
    off_style = [(0.25, "#1f77b4", r"$z_0 = L \; / \; 4$"),
                 (1.0, "#2ca02c", r"$z_0 = L$"),
                 (10.0, "#9467bd", r"$z_0 = 10 L$")]
    for z, color, label in off_style:
        ax.loglog(eps_grid, np.abs(el[z]), "-", color=color, lw=0.5,
                  label=label)

    ax.set_xlabel(r"$\varepsilon \; / \; L$")
    ax.set_ylabel(r"$|\sigma_{xz}|$ at the centroid")
    ax.set_xlim(1e-3, 1e0)
    ax.set_ylim(1e-5, 1e1)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticks([1e-5, 1e-3, 1e-1, 1e1])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_box_aspect(1)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="0.3", fontsize=6.5)

    out_dir = os.path.join(ROOT, "manuscript", "figures")
    fig.savefig(os.path.join(out_dir, "fig_onfault_stress_eps_sweep.pdf"))
    plt.close(fig)
    print("wrote fig_onfault_stress_eps_sweep.pdf")


if __name__ == "__main__":
    main()

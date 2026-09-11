"""Figure — The "CEDT cliff": kernel-level breakdown demo.

CEDT = classical elastic dislocation theory (the exact singular
analytic triangle-dislocation solution); the curves here are computed
with this repo's own analytic kernels at eps = 0.

Single source equilateral triangle (edge $L=1$) in the $z=0$ plane,
unit slip $\\Delta u=(1,0,0)$. Walk an observation point in-plane
toward one of the source's vertices, parameterised by stand-off
$d/L\\in[10^{-4},\\,2]$. This is exactly the BEM matrix-entry
geometry: an off-diagonal collocation point approaching the source
triangle's boundary. We plot $\\|\\sigma\\|_F$ at the observation
point for five methods:

  - CEDT analytic singular ($\\varepsilon=0$): finite for any
    $d>0$ but grows like ${\\sim}\\,1/d$ near the boundary; the BEM
    matrix entry diverges in the $d\\to 0$ limit.
  - Gauss quadrature of the singular kernel at $n_q\\in\\{4,8,16\\}$:
    fixed-order nodes can't resolve the $1/R^5$ peak when an
    integration point happens to lie close to the observation; the
    result oscillates between drastic over- and undershoots, with no
    quadrature order rescuing it.
  - mollified analytic at $\\varepsilon/L=1/3$: bounded everywhere by
    construction, saturating at the smoothed limit as $d\\to 0$.

The figure makes the central BEM message visible at one glance: the
CEDT entry blows up at the near-element distances that are
exactly where BEM collocation lives; quadrature is unusable; only
the mollified closed form is uniformly valid across all blocks of
the matrix.
"""
# medt_paper: copied from moss/manuscript/scripts/fig10_cedt_cliff.py; only paths, imports and
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
    set_paper_style, text_size, panel_letter,
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
np.seterr(divide="ignore", invalid="ignore", over="ignore")


def _equilateral_triangle(L: float = 1.0):
    h = L * np.sqrt(3.0) / 2.0
    centroid_y = h / 3.0
    v1 = np.array([0.0, h - centroid_y, 0.0])
    v2 = np.array([-L / 2.0, -centroid_y, 0.0])
    v3 = np.array([L / 2.0, -centroid_y, 0.0])
    return v1, v2, v3, np.array([0.0, 0.0, 1.0])


def main():
    L = 1.0
    mu, nu = 1.0, 0.25
    eps_moll = L / 3.0
    v1, v2, v3, n_vec = _equilateral_triangle(L)
    slip = np.array([1.0, 0.0, 0.0])

    # Observation walks in-plane toward the source's right vertex
    # (L/2, -h/3, 0). A 1e-6·L off-plane offset keeps the analytical
    # formula well-defined (ε=0 with strictly in-plane obs needs a
    # special finite-part branch).
    h_tri = L * np.sqrt(3.0) / 2.0
    y_obs = -h_tri / 3.0
    z_off = 1.0e-6 * L
    ds = np.logspace(-4, np.log10(2.0), 90)         # stand-off d/L

    def _evaluate_one(method, n_q=None):
        out = np.zeros(len(ds))
        for i, d in enumerate(ds):
            obs = np.array([L / 2.0 + d, y_obs, z_off])
            try:
                if method == "analytic":
                    H = analytical_stress_kernel(
                        obs, v1, v2, v3, n_vec, mu, nu, 0.0,
                    )
                elif method == "mollified":
                    H = analytical_stress_kernel(
                        obs, v1, v2, v3, n_vec, mu, nu, eps_moll,
                    )
                elif method == "quadrature":
                    H = integrate_stress_kernel(
                        obs, v1, v2, v3, n_vec, mu, nu, 0.0, n_q,
                    )
                else:
                    raise ValueError(method)
                sig = np.einsum("mnk,k->mn", H, slip)
                out[i] = np.linalg.norm(sig)
            except Exception:
                out[i] = np.nan
        return out

    print("computing analytic singular ...")
    sig_an = _evaluate_one("analytic")
    print("computing mollified ...")
    sig_mo = _evaluate_one("mollified")
    sig_q = {}
    for n_q in (4, 8, 16):
        print(f"computing quadrature n_q={n_q} ...")
        sig_q[n_q] = _evaluate_one("quadrature", n_q=n_q)

    fig, ax = plt.subplots(figsize=(3.22, 3.08))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.12)

    # CEDT analytic singular, black
    ax.loglog(ds, sig_an, "-", color="#000000", lw=0.5,
              label=r"CEDT analytic ($\varepsilon=0$)")
    # quadrature curves: cool palette
    quad_colors = {4: "#9467bd", 8: "#1f77b4", 16: "#2ca02c"}
    for n_q, c in quad_colors.items():
        ax.loglog(ds, sig_q[n_q], "-", color=c, lw=0.5,
                  label=rf"Gauss quadrature, $n_q={n_q}$")
    # mollified analytic, warm
    ax.loglog(ds, sig_mo, "-", color="#d62728", lw=0.5,
              label=r"MEDT analytic ($\varepsilon \; / \; L = 1 \; / \; 3$)")

    # Reference 1/d line for visual comparison with the analytic singular tail.
    d_ref = np.array([1e-3, 1.0])
    ref_idx = int(np.argmin(np.abs(ds - 0.01)))
    if np.isfinite(sig_an[ref_idx]) and sig_an[ref_idx] > 0:
        c_ref = sig_an[ref_idx] * 0.01
        ax.loglog(d_ref, c_ref / d_ref, "--",
                  color="0.5", lw=0.5, label=r"$\propto 1 \; / \; d_{\mathrm{s}}$")

    ax.set_xlabel(r"$d_{\mathrm{s}} \; / \; L$")
    ax.set_ylabel(r"$\|\sigma\|_F$ at observation point")
    ax.set_xlim(1e-4, 1e1)
    ax.set_ylim(1e-3, 1e4)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_box_aspect(1)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False)

    out_dir = os.path.join(ROOT, "manuscript", "figures")
    fig.savefig(os.path.join(out_dir, "fig_near_element_stress.pdf"))
    plt.close(fig)
    print("wrote manuscript/figures/fig_near_element_stress.pdf")


if __name__ == "__main__":
    main()

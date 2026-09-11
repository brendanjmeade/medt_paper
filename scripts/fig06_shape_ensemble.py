"""Figure — Shape sensitivity of the elastic stress at all four
stand-offs of the single-triangle sweep (fig14): the 300-shape random
ensemble drawn as individual lines, tinted per stand-off.

Same data as fig15_shape_ensemble (300 random equal-area triangles,
q >= 0.05, four stand-offs z0 / L in {0, 1/4, 1, 10}, elastic sigma_xz
at the centroid with the exact finite-triangle eigenstress subtracted).
Rendering: every ensemble curve is drawn as a thin, semi-transparent
line in a lighter shade of its stand-off's reference color (light red
for z0 = 0, light blue for L / 4, light green for L, light purple for
10 L), with the four full-color equilateral curves of fig14 on top.
Compared with the uniform-gray cloud of fig15, the hue keys each
ensemble line to its stand-off where the clouds overlap; compared with
the 95% bands of fig16, the individual sign-change dips stay visible.

Reads _cache/fig15_shape_ensemble.npz; if the cache is missing, the
fig15 computation is run first (which also rebuilds the fig15 figure).
"""
# medt_paper: copied from moss/manuscript/scripts/fig18_shape_ensemble_lines.py; only paths, imports and
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
import shape_ensemble_compute as ens  # noqa: E402

set_paper_style()
# this figure uses the standard matplotlib sans-serif face
import matplotlib as mpl
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})
np.seterr(divide="ignore", invalid="ignore")

EQ_STYLE = [("#d62728", r"$z_0 = 0$"),
            ("#1f77b4", r"$z_0 = L \; / \; 4$"),
            ("#2ca02c", r"$z_0 = L$"),
            ("#9467bd", r"$z_0 = 10 L$")]
TINT = 0.5          # fraction of the distance from full color to white
ALPHA = 0.15
LW_ENS = 0.35


def lighten(color, f=TINT):
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(1.0 - f * (1.0 - rgb))


def report(el_all, el_eq, eps_grid, heights):
    n_tri = el_all.shape[0]
    print(f"ensemble spread over {n_tri} shapes at eps = "
          f"{eps_grid[0]:.0e}:")
    for j, z in enumerate(heights):
        band = el_all[:, j, 0]
        dev = np.max(np.abs(band - el_eq[j, 0]) / np.abs(el_eq[j, 0]))
        print(f"  z0={z:g}: [{band.min():+.3e}, {band.max():+.3e}]  "
              f"(equilateral {el_eq[j, 0]:+.3e}, max dev {dev:.2%})")


def render(el_all, el_eq, eps_grid, out_stem):
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.13)

    n_tri = el_all.shape[0]
    for j, (color, _) in enumerate(EQ_STYLE):
        tint = lighten(color)
        for i in range(n_tri):
            ax.loglog(eps_grid, np.abs(el_all[i, j]), "-",
                      color=tint, lw=LW_ENS, alpha=ALPHA)
    for j, (color, label) in enumerate(EQ_STYLE):
        ax.loglog(eps_grid, np.abs(el_eq[j]), "-", color=color, lw=1.0,
                  label=label)

    ax.set_xlabel(r"$\varepsilon \; / \; L$")
    ax.set_ylabel(r"$|\sigma_{xz}|$")
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
    fig.savefig(os.path.join(out_dir, out_stem + ".pdf"))
    plt.close(fig)
    print(f"wrote {out_stem}.pdf")


def main():
    if not os.path.exists(ens.CACHE):
        print("fig15 cache missing -- running the ensemble sweep first ...")
        ens.main()
    d = np.load(ens.CACHE)
    report(d["el_all"], d["el_eq"], d["eps_grid"], d["heights"])
    render(d["el_all"], d["el_eq"], d["eps_grid"],
           "fig_shape_ensemble")


if __name__ == "__main__":
    main()

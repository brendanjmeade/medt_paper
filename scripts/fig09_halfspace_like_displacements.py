"""Figure 9 -- fault-parallel displacement u_x of a mollified strike-slip fault
in an elastic half-space, in map view (top row) and across-strike
cross-section (bottom row), for three mollification scales eps.

The paper's figure was assembled (in Keynote) from three frames of the mhf
eps-ladder animation (mhf/demo_eps_animation.py, high-resolution run:
--eps-max 0.001 --eps-min 2 --n-frames 200 --n 363 --ny 483 --nz 243),
cropping the u_x column of the six-panel figure.  This script produces the
same panels directly, as a single vector PDF, with the u_x color limits
frozen across the three eps values exactly as in the animation (99th
percentile of |u_x| over all panels).

Physics: exact-boundary-condition mollified Mindlin (half-space) triangle
dislocation kernels from the mhf package -- a 10 km x 10 km vertical
rectangular fault (two triangles) breaking the free surface at y = 0 with
1 m of strike slip, nu = 1/4.  Map view at z = -2 km; section at x = 0.

Run (from the package root; JAX recommended, numpy fallback is much slower):
    python scripts/fig09_halfspace_like_displacements.py
    python scripts/fig09_halfspace_like_displacements.py --fast   # coarse grids
    python scripts/fig09_halfspace_like_displacements.py --eps-km 0.001 0.4 2.0
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mhf.demo_sixpanel import add_common_args, compute_fields  # noqa: E402
from mhf.figures import _fmt  # noqa: E402  (also applies the house rcParams)

CMAP = "PiYG"
N_BANDS = 7


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="u_x map view + cross-section for three eps (Figure 9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)
    p.add_argument("--eps-km", type=float, nargs="+", default=[0.001, 0.4, 2.0],
                   help="mollification scales (km), one column each")
    p.add_argument("--fast", action="store_true",
                   help="coarse grids (n=121, ny=161, nz=81) for a quick look")
    p.add_argument("--out", default=os.path.join(ROOT, "manuscript", "figures",
                                                 "fig_halfspace_like_displacements"),
                   help="output stem (PDF)")
    args = p.parse_args(argv)
    if not args.fast:                       # the paper's high-resolution grids
        args.n, args.ny, args.nz = 363, 483, 243
    return args


def _len_str(km):
    """Math-mode length label with upright units: 1\\,m, 400\\,m, 2\\,km."""
    return (rf"{km:.4g}\,\mathrm{{km}}" if km >= 1.0
            else rf"{km * 1e3:.4g}\,\mathrm{{m}}")


def _panel(ax, A, B, data, vmax, overlays):
    levels = np.linspace(-vmax, vmax, N_BANDS + 1)
    im = ax.contourf(A, B, data, levels=levels, cmap=CMAP, extend="both")
    for seg in overlays:
        ax.plot(seg[:, 0], seg[:, 1], "k-", lw=1.0)
    ax.set_aspect("equal")
    ax.tick_params(direction="out", length=3, width=0.8)
    return im


def main(argv=None):
    args = _parse_args(argv)
    eps_list = list(args.eps_km)
    print(f"Figure 9: u_x for eps = {eps_list} km; grids n={args.n}, "
          f"ny={args.ny}, nz={args.nz}", flush=True)
    fields = []
    for eps in eps_list:
        F = compute_fields(args, eps)
        fields.append(F)
        print(f"  eps={eps:g} km: |u_x| max map {np.abs(F['top'][0]).max():.1f} mm, "
              f"section {np.abs(F['bot'][0]).max():.1f} mm", flush=True)

    # frozen color limit across eps: 99th percentile of |u_x| over all panels
    data = np.concatenate([np.abs(F[row][0]).ravel()
                           for F in fields for row in ("top", "bot")])
    vmax = float(np.percentile(data, 99.0))
    print(f"  frozen u_x limit: +/-{vmax:.3g} mm", flush=True)

    h = args.half_extent
    fig, axes = plt.subplots(2, len(eps_list), figsize=(3.4 * len(eps_list), 6.0),
                             gridspec_kw=dict(wspace=0.35, hspace=0.35,
                                              height_ratios=[1.0, 0.62]))
    axes = np.atleast_2d(axes)
    im = None
    for j, (eps, F) in enumerate(zip(eps_list, fields)):
        ax = axes[0, j]
        im = _panel(ax, F["X"], F["Y"], F["top"][0], vmax, F["ov_top"])
        ax.set_title(rf"$u_x$ (mm), $\varepsilon = {_len_str(eps)}$", pad=3)
        ax.set_xlim(-h, h); ax.set_ylim(-h, h)
        ax.set_xticks([-int(h), 0, int(h)]); ax.set_yticks([-int(h), 0, int(h)])
        ax.set_xlabel(r"$x$ (km)")
        ax = axes[1, j]
        _panel(ax, F["Yc"], F["Zc"], F["bot"][0], vmax, F["ov_bot"])
        ax.set_xlim(-F["y_half"], F["y_half"]); ax.set_ylim(F["z_bot"], 0.0)
        ax.set_xticks([-int(F["y_half"]), 0, int(F["y_half"])])
        ax.set_yticks([int(F["z_bot"]), int(round(F["z_bot"] / 2)), 0])
        ax.set_xlabel(r"$y$ (km)")
    axes[0, 0].set_ylabel(r"$y$ (km)")
    axes[1, 0].set_ylabel(r"$z$ (km)")
    assert im is not None, "no panels were drawn"
    for row in range(2):
        cax = axes[row, -1].inset_axes([1.06, 0.0, 0.05, 1.0])
        cb = fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax])
        cb.ax.set_yticklabels([_fmt(-vmax), "0", _fmt(vmax)])
        cb.ax.tick_params(labelsize=7)

    out = args.out + ".pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()

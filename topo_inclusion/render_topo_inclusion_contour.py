"""Contourf rendering of the topography/inclusion decomposition.

Same content as render_topo_inclusion12.py, but the per-triangle BEM
surface displacements are interpolated onto a regular 500 x 500 grid
and drawn with filled contours instead of flat-shaded triangles.

Layout (columns u_x, u_y, u_z; every residual against the SAME plain
half-space reference u(flat, hom)):

  row 1  full model            u(topo, het)
  row 2  inclusion only        u(flat, het) - u(flat, hom)
  row 3  topography only       u(topo, hom) - u(flat, hom)
  row 4  topography+inclusion  u(topo, het) - u(flat, hom)

NOTE on the fault trace: the displacement field is genuinely
DISCONTINUOUS across the surface-breaking fault (x = 0, |y| <= 100);
linear interpolation onto the grid smears the jump over about one grid
cell (0.8 km at 500 x 500), which contourf renders as a tight band of
contours along the trace. That is a faithful presentation, not an
artifact to fix.

Smoothing: --smooth applies one pass of vertex averaging to the
per-triangle values before gridding (each vertex takes the mean of its
adjacent triangles; triangles take the mean of their vertices). This
suppresses the element-scale roughness that is most visible in the
small-amplitude u_z panels. On the host surface the averaging is done
SEPARATELY on each side of the fault plane so the genuine slip
discontinuity is not smeared (the side split extends beyond the fault
ends along x = 0, where the field is continuous — the only cost there
is slightly less smoothing along that line). The output suffix gains
"_smooth" automatically.

Interpolation: "linear" (default) is piecewise-linear on a Delaunay
triangulation of the BEM triangle centroids — faithful to the
piecewise-constant BEM solution, and it honestly exposes the coarse
regions of the model (large far-field triangles render as faceted
contours). "cubic" (Clough-Tocher C1) gives smoother contours but can
overshoot near sharp gradients — most visibly ringing along the fault
discontinuity — so treat it as presentation polish, not added accuracy.

Usage:
    python benchmarks/render_topo_inclusion_contour.py \
        [npz] [suffix] [--method linear|cubic]
Defaults: the on-fault cache, suffix "_onfault", linear.
Output (repo root): fig_topo_inclusion_contour{suffix}.png/.pdf
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbem.topography import gaussian_bump  # noqa: E402

NPZ_DEFAULT = (pathlib.Path(__file__).parent
               / "topo_inclusion_fields_mu10_onfault.npz")

INC_X, INC_Y, INC_R = -100.0, 100.0, 75.0
NGRID = 500
EXTENT = (-200.0, 200.0)


def _vertex_smooth(verts, tris, values, split_x_trace=False):
    """One vertex-averaging pass over per-triangle values.

    ``split_x_trace``: average each side of x = 0 separately so a
    displacement jump across the fault plane is preserved.
    """
    cent_x = verts[tris].mean(axis=1)[:, 0]
    sides = ([cent_x < 0.0, cent_x >= 0.0] if split_x_trace
             else [np.ones(tris.shape[0], dtype=bool)])
    out = np.empty_like(values)
    for sel in sides:
        vert_sum = np.zeros((verts.shape[0], values.shape[1]))
        vert_cnt = np.zeros(verts.shape[0])
        for k in range(3):
            np.add.at(vert_sum, tris[sel, k], values[sel])
            np.add.at(vert_cnt, tris[sel, k], 1)
        vert_val = vert_sum / np.maximum(vert_cnt, 1)[:, None]
        out[sel] = vert_val[tris[sel]].mean(axis=1)
    return out


def _grid_field(centroids, values, gx, gy, method="linear"):
    """Interpolate per-triangle values onto the grid, with a
    nearest-neighbour fallback for hull-edge NaNs."""
    pts = centroids[:, :2]
    g = griddata(pts, values, (gx, gy), method=method)
    nan = np.isnan(g)
    if nan.any():
        g[nan] = griddata(pts, values, (gx[nan], gy[nan]),
                          method="nearest")
    return g


def _style(ax, gx, gy, h_grid):
    ax.plot([0.0, 0.0], [-100.0, 100.0], "k-", lw=0.8)
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(INC_X + INC_R * np.cos(th), INC_Y + INC_R * np.sin(th),
            "k--", lw=0.7)
    if h_grid.max() > 0.5:
        ax.contour(gx, gy, h_grid, levels=[0.5, 1.0, 1.5],
                   colors="0.35", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(*EXTENT)
    ax.set_ylim(*EXTENT)
    ax.set_xticks([-200, 0, 200])
    ax.set_yticks([-200, 0, 200])
    ax.tick_params(direction="out", length=3, width=0.8)


def main(npz_path=NPZ_DEFAULT, suffix="_onfault",
         method="linear", smooth=False):
    d = np.load(npz_path)
    if smooth:
        suffix = suffix + "_smooth"
    mu_inc = float(d["mu_inc"]) if "mu_inc" in d.files else None

    cent = np.vstack([
        d["host_top_vertices"][d["host_top_triangles"]].mean(axis=1),
        d["inclusion_top_vertices"][d["inclusion_top_triangles"]]
        .mean(axis=1)])
    axis = np.linspace(*EXTENT, NGRID)
    gx, gy = np.meshgrid(axis, axis)

    bump = gaussian_bump(tuple(d["bump_center"]), float(d["bump_height"]),
                         float(d["bump_sigma"]))
    h_grid = bump(gx.ravel(), gy.ravel()).reshape(gx.shape)

    def field(surface, state):        # (N_tri_total, 3) in mm
        uh = d[f"u_host_top_{surface}_{state}"] * 1e6
        ui = d[f"u_inclusion_top_{surface}_{state}"] * 1e6
        if smooth:
            uh = _vertex_smooth(d["host_top_vertices"],
                                d["host_top_triangles"], uh,
                                split_x_trace=True)
            ui = _vertex_smooth(d["inclusion_top_vertices"],
                                d["inclusion_top_triangles"], ui)
        return np.vstack([uh, ui])

    th = field("topo", "het")
    tm = field("topo", "hom")
    fh = field("flat", "het")
    fm = field("flat", "hom")         # the half-space reference

    rows = [
        ("full model", th),
        ("inclusion only", fh - fm),
        ("topography only", tm - fm),
        ("topography + inclusion", th - fm),
    ]

    comp = [r"u_x", r"u_y", r"u_z"]
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.2),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.20, hspace=0.16))

    for r, (label, U) in enumerate(rows):
        for k in range(3):
            ax = axes[r, k]
            g = _grid_field(cent, U[:, k], gx, gy,
                            method=method)
            vmax = max(float(np.percentile(np.abs(U[:, k]), 99.0)), 0.5)
            levels = np.linspace(-vmax, vmax, 21)
            cf = ax.contourf(gx, gy, g, levels=levels, cmap="RdBu_r",
                             extend="both")
            _style(ax, gx, gy, h_grid)
            prefix = rf"${comp[k]}$" if r == 0 else rf"$\Delta {comp[k]}$"
            ax.set_title(f"{prefix} (mm)", fontsize=9)
            cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04,
                              shrink=0.8)
            cb.ax.tick_params(labelsize=7)
            cb.set_ticks([-vmax, 0, vmax])
            cb.set_ticklabels([f"{-vmax:.0f}", "0", f"{vmax:.0f}"])
        axes[r, 0].set_ylabel(f"{label}\n\n$y$ (km)", fontsize=9)
    for k in range(3):
        axes[3, k].set_xlabel(r"$x$ (km)")
    for ax, letter in zip(axes.flat, "abcdefghijkl"):
        ax.text(0.95, 0.95, letter, transform=ax.transAxes,
                ha="right", va="top")

    if mu_inc is not None:
        fig.suptitle(
            rf"strike-slip fault, $\mu_{{\rm inc}} = \mu/{30/mu_inc:.0f}$ "
            rf"inclusion, 2 km hill — {NGRID}$\times${NGRID} gridded, "
            rf"filled contours", y=0.995, fontsize=10)

    for ext in ("png", "pdf"):
        fname = ROOT / f"fig_topo_inclusion_contour{suffix}.{ext}"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", type=pathlib.Path,
                    default=NPZ_DEFAULT)
    ap.add_argument("suffix", nargs="?", default="_onfault")
    ap.add_argument("--method", choices=("linear", "cubic"),
                    default="linear")
    ap.add_argument("--smooth", action="store_true",
                    help="vertex-average per-triangle values before "
                         "gridding (fault jump preserved)")
    a = ap.parse_args()
    main(a.npz, suffix=a.suffix, method=a.method, smooth=a.smooth)

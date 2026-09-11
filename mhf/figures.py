"""3x3-panel field figures (3 displacement + 6 unique stress components) in the
repo house style: full box, sparse edge ticks, no grid, RdBu_r symmetric about
zero, per-panel colorbar, top-right panel letters, equal aspect, shared outer
labels.  Used by mhf.demo_mapview / mhf.demo_xsection."""
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .fields import KM_TO_MM, GPA_TO_MPA, fault_plane_segments

mpl.rcParams.update({
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavusans",
})


def _fmt(v):
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2g}"


def _make_panels(u, sig, d0, d1):
    """The 9 (field, label) panels (disp mm, stress MPa) on a (d0,d1) grid."""
    u = u.reshape(d0, d1, 3) * KM_TO_MM
    sig = sig.reshape(d0, d1, 3, 3) * GPA_TO_MPA
    return [
        (u[:, :, 0], r"$u_x$ (mm)"), (u[:, :, 1], r"$u_y$ (mm)"),
        (u[:, :, 2], r"$u_z$ (mm)"),
        (sig[:, :, 0, 0], r"$\sigma_{xx}$ (MPa)"),
        (sig[:, :, 1, 1], r"$\sigma_{yy}$ (MPa)"),
        (sig[:, :, 2, 2], r"$\sigma_{zz}$ (MPa)"),
        (sig[:, :, 0, 1], r"$\sigma_{xy}$ (MPa)"),
        (sig[:, :, 0, 2], r"$\sigma_{xz}$ (MPa)"),
        (sig[:, :, 1, 2], r"$\sigma_{yz}$ (MPa)"),
    ]


def _render_9panels(A, B, panels, overlays, xlabel, ylabel, a_lim, b_lim,
                    a_ticks, b_ticks, title, outstem, figsize):
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.32, hspace=0.18))
    for k, (ax, (data, label)) in enumerate(zip(axes.flat, panels)):
        vmax = float(np.percentile(np.abs(data), 99.0))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.max(np.abs(data))) or 1.0
        im = ax.pcolormesh(A, B, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="nearest", rasterized=True)
        for seg in overlays:                  # fault <-> plane intersection
            ax.plot(seg[:, 0], seg[:, 1], "k-", lw=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(*a_lim); ax.set_ylim(*b_lim)
        ax.set_xticks(a_ticks); ax.set_yticks(b_ticks)
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.set_title(label, pad=3)
        ax.text(0.96, 0.95, "abcdefghi"[k], transform=ax.transAxes, ha="right",
                va="top", fontsize=9,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                          alpha=0.7))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          ticks=[-vmax, 0, vmax])
        cb.ax.set_yticklabels([_fmt(-vmax), "0", _fmt(vmax)])
        cb.ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10.5, y=0.985)
    here = os.path.dirname(os.path.abspath(__file__))
    for ext in ("png", "pdf"):
        out = os.path.join(here, f"{outstem}.{ext}")
        fig.savefig(out)
        print(f"  wrote {out}")
    plt.close(fig)


def plot_9panel(X, Y, u, sig, tris, z0, title, outstem, half_extent=15.0):
    """Map-view (x-y slice at z0) 3x3 figure: row 1 u; rows 2-3 stress."""
    ny, nx = X.shape
    panels = _make_panels(u, sig, ny, nx)
    overlays = fault_plane_segments(tris, 2, z0)         # z-plane -> (x,y)
    lim = half_extent
    t = [-int(lim), 0, int(lim)]
    _render_9panels(X, Y, panels, overlays, r"$x$ (km)", r"$y$ (km)",
                    (-lim, lim), (-lim, lim), t, t, title, outstem, (8.0, 8.2))


def plot_9panel_section(Y, Z, u, sig, tris, x0, title, outstem,
                        y_half=20.0, z_bot=-20.0):
    """Vertical cross-section (y-z at x=x0) 3x3 figure; free surface z=0 on top."""
    nz, ny = Y.shape
    panels = _make_panels(u, sig, nz, ny)
    overlays = fault_plane_segments(tris, 0, x0)         # x-plane -> (y,z)
    yt = [-int(y_half), 0, int(y_half)]
    zt = [int(z_bot), int(round(z_bot / 2)), 0]
    _render_9panels(Y, Z, panels, overlays, r"$y$ (km)", r"$z$ (km)",
                    (-y_half, y_half), (z_bot, 0.0), yt, zt, title, outstem,
                    (9.5, 6.2))

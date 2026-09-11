"""Figure — The stress field of the mesh: CEDT vs mollified on a
conformally tessellated elliptical crack, at two mesh resolutions.

An elliptical crack (semi-axes 0.95 x 0.45, z = 0 plane) carries the
smooth slip taper

    s(x, y) = (1 - r^2)^{3/2},   r^2 = (x/a)^2 + (y/b)^2,

which has zero gradient at the rim, so the physical stress field of
the slip distribution has NO rim ring: its structure lives in a broad
mid-annulus band.  The crack is tessellated conformally (boundary
nodes on the rim, no element crosses it) at edge h = 0.125 and at
h/2, and the slip is sampled piecewise-constant per element — exactly
what a collocation BEM does.  Every internal edge between elements
with different sampled slip is a real edge dislocation.

Panels (all maps share one log color scale; observation plane
z0 = 0.02 above the crack):
  (a, b) CEDT (eps = 0) on mesh h and on mesh h/2: the field is an
      image of the triangulation — a 1/r line singularity glows along
      every internal edge, and REFINING THE MESH CHANGES THE FIELD'S
      STRUCTURE, which now traces the finer web.
  (c, d) Mollified elastic at fixed eps = h/3 (h the coarse edge;
      smooth-slip eigenstress subtracted, since z0 < eps) on the same
      two meshes: the field is an image of the slip distribution and
      is unchanged by the refinement.
Stress is computed from the batched analytic DD displacement kernel
(in-plane derivatives from the grid, z-derivative from two planes at
z0 +/- dz).  Fields are cached in _cache/fig11_mesh_image_v2.npz;
delete it (or pass --force) to recompute.
"""
# medt_paper: copied from moss/manuscript/scripts/fig11_mesh_image.py; only paths, imports and
# output names were changed (figures go to manuscript/figures/, caches to cache/).

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _paper_style import (  # noqa: E402
    set_paper_style, panel_letter,
)
from mollified_kernel.analytical_batch import dd_displacement_batch  # noqa: E402
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

CACHE = os.path.join(ROOT, "cache", "fig08_elliptical_crack_mesh_sensitivity.npz")

MU, NU = 1.0, 0.25
LAM = 2.0 * MU * NU / (1.0 - 2.0 * NU)
H_TRI = 0.125                  # coarse triangle edge
Z0 = 0.02                      # observation height above the crack plane
DZ = 0.004                     # z-derivative offset
EPS_MOLL = H_TRI / 3.0         # fixed physical smoothing for both meshes
A_TAPER, B_TAPER = 0.95, 0.45  # crack semi-axes


def tessellate(h_tri):
    """Conformal triangulation of the elliptical crack at edge h_tri:
    boundary nodes lie exactly on the rim, so no element crosses it.
    Interior nodes on a hexagonal lattice; Delaunay of the (convex)
    ellipse keeps every triangle inside."""
    from scipy.spatial import Delaunay
    a, bb = A_TAPER, B_TAPER
    th = np.linspace(0.0, 2.0 * np.pi, 4001)
    dx = -a * np.sin(th)
    dy = bb * np.cos(th)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1])
                                           * np.diff(th))])
    n_b = int(round(arc[-1] / h_tri))
    th_b = np.interp(np.linspace(0.0, arc[-1], n_b, endpoint=False),
                     arc, th)
    bound = np.column_stack([a * np.cos(th_b), bb * np.sin(th_b)])
    pts = [bound]
    dy_row = h_tri * np.sqrt(3.0) / 2.0
    j = 0
    y = -bb
    while y <= bb:
        x0 = -a + (0.5 * h_tri if j % 2 else 0.0)
        xs_row = np.arange(x0, a + 1e-9, h_tri)
        keep = ((xs_row / (a - 0.55 * h_tri)) ** 2
                + (y / (bb - 0.55 * h_tri)) ** 2) < 1.0
        if keep.any():
            pts.append(np.column_stack([xs_row[keep],
                                        np.full(keep.sum(), y)]))
        j += 1
        y += dy_row
    pts2d = np.vstack(pts)
    tri = Delaunay(pts2d)
    verts = np.column_stack([pts2d, np.zeros(len(pts2d))])
    return verts, tri.simplices.copy()


def slip(x, y):
    """Smooth taper with zero gradient at the rim: no physical rim
    ring, so the mollified field's structure is decoupled from the
    CEDT web's brightest features."""
    r2 = (x / A_TAPER) ** 2 + (y / B_TAPER) ** 2
    return np.maximum(0.0, 1.0 - r2) ** 1.5


def u_field(pts, verts, tris, b, eps):
    u = np.zeros((pts.shape[0], 3))
    n_vec = np.array([0.0, 0.0, 1.0])
    for s, tri in enumerate(tris):
        if b[s] == 0.0:
            continue
        U = dd_displacement_batch(verts[tri[0]], verts[tri[1]],
                                  verts[tri[2]], n_vec, pts, MU, NU, eps)
        u += U[:, :, 0] * b[s]
    return u


def sigma_tensor(xg, yg, verts, tris, b, eps):
    """Full sigma tensor on the grid at z = Z0 via grid differencing
    (x, y) and two-plane differencing (z)."""
    X, Y = np.meshgrid(xg, yg)
    flat = np.column_stack([X.ravel(), Y.ravel()])
    grad = np.zeros((X.shape[0], X.shape[1], 3, 3))
    u_mid = None
    for dz, w in ((-DZ, -1.0), (0.0, 0.0), (+DZ, +1.0)):
        pts = np.column_stack([flat, np.full(len(flat), Z0 + dz)])
        u = u_field(pts, verts, tris, b, eps).reshape(X.shape + (3,))
        if w == 0.0:
            u_mid = u
        else:
            grad[..., 2] += w * u / (2.0 * DZ)
    dy = yg[1] - yg[0]
    dx = xg[1] - xg[0]
    for i in range(3):
        gy, gx = np.gradient(u_mid[..., i], dy, dx)
        grad[..., i, 0] = gx
        grad[..., i, 1] = gy
    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    tr = np.trace(strain, axis1=-2, axis2=-1)
    return LAM * tr[..., None, None] * np.eye(3) + 2.0 * MU * strain


def gamma_marginal(z, eps):
    """Cortez fault-normal marginal of the smeared slip: integrates
    to 1 over z, peak (3/4)/eps at z = 0."""
    return 0.75 * eps ** 4 / (z ** 2 + eps ** 2) ** 2.5


def elastic_frob(sig, xg, yg, eps):
    """|sigma_el|_F: subtract the smooth-slip eigenstress (xz/zx)."""
    X, Y = np.meshgrid(xg, yg)
    star = MU * slip(X, Y) * gamma_marginal(Z0, eps)
    s = sig.copy()
    s[..., 0, 2] -= star
    s[..., 2, 0] -= star
    return np.sqrt(np.sum(s ** 2, axis=(-2, -1)))


def main(force: bool = False):
    meshes = {}
    for tag, h in (("h", H_TRI), ("h2", 0.5 * H_TRI)):
        verts, tris = tessellate(h)
        cen = verts[tris].mean(axis=1)
        meshes[tag] = (verts, tris, cen, slip(cen[:, 0], cen[:, 1]))
        print(f"mesh {tag}: {len(tris)} triangles")

    xg = np.linspace(-1.15, 1.15, 361)
    yg = np.linspace(-0.65, 0.65, 201)

    if not force and os.path.exists(CACHE):
        d = np.load(CACHE)
        sig = {k: d[k] for k in ("cedt_h", "cedt_h2", "moll_h", "moll_h2")}
    else:
        sig = {}
        for tag in ("h", "h2"):
            verts, tris, cen, b = meshes[tag]
            print(f"computing CEDT field on mesh {tag} ...")
            sig[f"cedt_{tag}"] = sigma_tensor(xg, yg, verts, tris, b, 0.0)
            print(f"computing mollified field on mesh {tag} ...")
            sig[f"moll_{tag}"] = sigma_tensor(xg, yg, verts, tris, b,
                                              EPS_MOLL)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, **sig)

    F = {"cedt_h": np.sqrt(np.sum(sig["cedt_h"] ** 2, axis=(-2, -1))),
         "cedt_h2": np.sqrt(np.sum(sig["cedt_h2"] ** 2, axis=(-2, -1))),
         "moll_h": elastic_frob(sig["moll_h"], xg, yg, EPS_MOLL),
         "moll_h2": elastic_frob(sig["moll_h2"], xg, yg, EPS_MOLL)}

    # refinement-invariance metrics (quoted in the caption)
    X, Y = np.meshgrid(xg, yg)
    inside = (X / A_TAPER) ** 2 + (Y / B_TAPER) ** 2 < 1.0
    rel = lambda A, B: (np.sqrt(np.mean((A[inside] - B[inside]) ** 2))
                        / np.sqrt(np.mean(B[inside] ** 2)))
    print(f"mollified h vs h/2 relative RMS change: "
          f"{rel(F['moll_h2'], F['moll_h']):.3f}")
    print(f"CEDT      h vs h/2 relative RMS change: "
          f"{rel(F['cedt_h2'], F['cedt_h']):.3f}")

    print(f"CEDT h field:  max {np.nanmax(F['cedt_h']):.2f}, "
          f"99.9pct {np.nanpercentile(F['cedt_h'], 99.9):.2f}")
    print(f"moll h field:  max {np.nanmax(F['moll_h']):.2f}")
    # ---- Plot ----------------------------------------------------------
    fig = plt.figure(figsize=(8.6, 4.6))
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[1.25, 1.25, 0.09],
                          wspace=0.22, hspace=0.14,
                          left=0.07, right=0.95, top=0.93, bottom=0.11)
    ax_maps = [[fig.add_subplot(gs[r, c]) for c in range(2)]
               for r in range(2)]
    cax = fig.add_subplot(gs[:, 2])

    vmax = float(np.nanpercentile(F["cedt_h"], 99.9))
    norm = LogNorm(vmin=vmax / 1.0e4, vmax=vmax)
    order = [("cedt_h", 0, 0, "a"), ("cedt_h2", 0, 1, "b"),
             ("moll_h", 1, 0, "c"), ("moll_h2", 1, 1, "d")]
    for key, r, c, letter in order:
        ax = ax_maps[r][c]
        pm = ax.pcolormesh(xg, yg, F[key], norm=norm, cmap="magma",
                           rasterized=True, shading="auto")
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(direction="out", length=3, width=0.8)
        panel_letter(ax, letter)
        ax.texts[-1].set_fontweight("normal")
        if r == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$x$")
        if c == 1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$y$")
    ax_maps[0][0].set_title(r"mesh $h$", fontsize=9)
    ax_maps[0][1].set_title(r"mesh $h/2$", fontsize=9)
    ax_maps[0][0].text(-0.24, 0.5, r"CEDT ($\varepsilon = 0$)",
                       transform=ax_maps[0][0].transAxes, rotation=90,
                       va="center", ha="center", fontsize=9)
    ax_maps[1][0].text(-0.24, 0.5,
                       r"MEDT ($\varepsilon = h/3$), elastic",
                       transform=ax_maps[1][0].transAxes, rotation=90,
                       va="center", ha="center", fontsize=9)

    # shrink the colorbar axes to 70% height, centered
    pos = cax.get_position()
    cax.set_position([pos.x0 + 0.55 * pos.width,
                      pos.y0 + 0.15 * pos.height,
                      0.4 * pos.width, 0.7 * pos.height])
    cb = fig.colorbar(pm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("right")
    cb.set_label(r"$\|\sigma\|_F$ at $z_0 = 0.02$")

    out_dir = os.path.join(ROOT, "manuscript", "figures")
    fig.savefig(os.path.join(out_dir, "fig_elliptical_crack_mesh_sensitivity.pdf"), dpi=300)
    plt.close(fig)
    print("wrote manuscript/figures/fig_elliptical_crack_mesh_sensitivity.pdf")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

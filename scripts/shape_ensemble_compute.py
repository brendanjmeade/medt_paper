"""Figure — Shape sensitivity of the elastic stress at all four
stand-offs of the single-triangle sweep (fig14): a random ensemble of
triangle shapes drawn as a gray cloud behind the equilateral curves.

The four colored curves repeat fig14 exactly: unit equilateral triangle
(edge L = 1) in the z = 0 plane, unit strike slip Du = (1, 0, 0),
mu = 1, nu = 1/4, elastic (eigenstress-subtracted) sigma_xz at the
centroid on the fault plane (z0 = 0) and at stand-offs z0 / L in
{1/4, 1, 10}, swept over eps / L.  Behind them, the same four sweeps
are repeated for 300 random triangles (vertices uniform in the unit
square, near-degenerate shapes with quality

    q = 4 sqrt(3) A / (a^2 + b^2 + c^2)      (1 equilateral -> 0 sliver)

below 0.05 rejected), each translated so its centroid is at the origin
and rescaled to the area of the unit equilateral triangle, and drawn as
transparent light-gray lines.  Equal area and slip means equal potency,
so the z0 = 10 L band collapses onto the equilateral curve (the far
field cannot see shape), while the on-fault band spreads by a factor of
several: shape sensitivity decays with stand-off.

The eigenstress is the exact finite-triangle blob convolution of fig14
(infinite-plane marginal minus an angular boundary integral over the
centroid-to-edge distance P(theta), 8000 rays), valid at any height.

Printed diagnostics back the caption: equilateral parity with fig14 at
all four heights, per-band small-eps spread across the ensemble, the
z0 = 10 L collapse, on-fault plateau-extent versus d_min statistics,
and a ray-doubling convergence check on the lowest-quality member.

The 240k analytic kernel evaluations are parallelized over triangles
with ProcessPoolExecutor; results are cached in
_cache/fig15_shape_ensemble.npz (pass --force to recompute).
"""
# medt_paper: copied from moss/manuscript/scripts/fig15_shape_ensemble.py (the ensemble computation; the paper figure is rendered by fig06_shape_ensemble.py); only paths, imports and
# output names were changed (figures go to manuscript/figures/, caches to cache/).

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
HEIGHTS = [Z_ON, 0.25, 1.0, 10.0]
AREA_REF = np.sqrt(3.0) / 4.0   # area of the unit equilateral triangle
N_TRI = 300
Q_MIN = 0.05
N_RAYS = 8000
EPS_GRID = np.logspace(-3.0, 0.0, 200)

CACHE = os.path.join(ROOT, "cache", "fig06_shape_ensemble.npz")


def equilateral_triangle(L: float = 1.0):
    h = L * np.sqrt(3.0) / 2.0
    centroid_y = h / 3.0
    v1 = np.array([0.0, h - centroid_y, 0.0])
    v2 = np.array([-L / 2.0, -centroid_y, 0.0])
    v3 = np.array([L / 2.0, -centroid_y, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    return v1, v2, v3, n


def triangle_quality(verts):
    """q = 4 sqrt(3) A / (a^2 + b^2 + c^2): 1 equilateral, -> 0 sliver."""
    v1, v2, v3 = verts
    e1, e2 = v2 - v1, v3 - v1
    area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
    ssq = (np.sum((v2 - v1) ** 2) + np.sum((v3 - v2) ** 2)
           + np.sum((v1 - v3) ** 2))
    return 4.0 * np.sqrt(3.0) * area / ssq


def sample_ensemble(rng, n_tri, q_min):
    """Random triangles: vertices uniform in the unit square, CCW order,
    centroid at the origin, area rescaled to AREA_REF."""
    tris = []
    while len(tris) < n_tri:
        pts = rng.uniform(0.0, 1.0, size=(3, 2))
        e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
        signed2 = e1[0] * e2[1] - e1[1] * e2[0]
        if signed2 < 0.0:                    # enforce CCW (normal +z)
            pts = pts[[0, 2, 1]]
            signed2 = -signed2
        area = 0.5 * signed2
        if area < 1e-6:
            continue
        pts = pts - pts.mean(axis=0)         # centroid -> origin
        pts = pts * np.sqrt(AREA_REF / area)  # area -> AREA_REF
        verts = [np.array([p[0], p[1], 0.0]) for p in pts]
        if triangle_quality(verts) < q_min:
            continue
        tris.append(verts)
    return tris


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


def elastic_sweep(verts, n_rays=N_RAYS):
    """Elastic sigma_xz at the centroid for every height in HEIGHTS
    over the eps sweep; returns ((n_heights, n_eps), d_min)."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    P_theta = boundary_distance(verts, thetas)
    n_vec = np.array([0.0, 0.0, 1.0])
    delta_u = np.array([1.0, 0.0, 0.0])
    el = np.zeros((len(HEIGHTS), len(EPS_GRID)))
    for k, eps in enumerate(EPS_GRID):
        for j, z in enumerate(HEIGHTS):
            obs = np.array([0.0, 0.0, z])
            H = analytical_stress_kernel(obs, verts[0], verts[1], verts[2],
                                         n_vec, MU, NU, eps)
            s_tot = np.einsum("mnk,k->mn", H, delta_u)[0, 2]
            el[j, k] = s_tot - eigenstress_xz(eps, z, P_theta)
    return el, float(np.min(P_theta))


def _worker(verts_arr):
    verts = [verts_arr[0], verts_arr[1], verts_arr[2]]
    return elastic_sweep(verts)


def main():
    if os.path.exists(CACHE) and "--force" not in sys.argv:
        d = np.load(CACHE)
        quality = d["quality"]
        d_min = d["d_min"]
        el_all = d["el_all"]                 # (N_TRI, n_heights, n_eps)
        el_eq = d["el_eq"]                   # (n_heights, n_eps)
        ray_conv = float(d["ray_conv"])
    else:
        rng = np.random.default_rng(0)
        tris = sample_ensemble(rng, N_TRI, Q_MIN)
        quality = np.array([triangle_quality(v) for v in tris])
        print(f"sampled {N_TRI} triangles: q in "
              f"[{quality.min():.3f}, {quality.max():.3f}]", flush=True)

        el_all = np.zeros((N_TRI, len(HEIGHTS), len(EPS_GRID)))
        d_min = np.zeros(N_TRI)
        n_workers = max(1, (os.cpu_count() or 4) - 2)
        print(f"sweeping {N_TRI} triangles x {len(HEIGHTS)} heights x "
              f"{len(EPS_GRID)} eps on {n_workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            args = [np.array(v) for v in tris]
            for i, (el, dm) in enumerate(pool.map(_worker, args,
                                                  chunksize=4)):
                el_all[i] = el
                d_min[i] = dm
                if i % 25 == 0:
                    print(f"  triangle {i} / {N_TRI}  (q={quality[i]:.3f}, "
                          f"on-fault plateau={el[0, 0]:+.4f})", flush=True)

        # equilateral reference through the identical pipeline
        v1, v2, v3, _ = equilateral_triangle(1.0)
        el_eq, _ = elastic_sweep([v1, v2, v3])

        # ray-doubling convergence check on the lowest-quality member
        iw = int(np.argmin(quality))
        el_w2, _ = elastic_sweep(tris[iw], n_rays=2 * N_RAYS)
        ray_conv = float(np.max(np.abs(el_w2 - el_all[iw]))
                         / np.max(np.abs(el_all[iw])))

        verts_all = np.array([np.array(v) for v in tris])
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, verts_all=verts_all, quality=quality, d_min=d_min,
                 el_all=el_all, el_eq=el_eq, eps_grid=EPS_GRID,
                 heights=np.asarray(HEIGHTS), ray_conv=ray_conv)

    # ---- Diagnostics ---------------------------------------------------
    print("\nequilateral parity with fig14 at eps=1e-3 "
          "(expect -1.6711, -0.368, +3.1e-2, +5.7e-5):")
    for j, z in enumerate(HEIGHTS):
        print(f"  z0={z:g}: {el_eq[j, 0]:+.4e}")

    print("ensemble spread at eps=1e-3 (min / median / max, and max "
          "relative deviation from the equilateral):")
    for j, z in enumerate(HEIGHTS):
        band = el_all[:, j, 0]
        dev = np.max(np.abs(band - el_eq[j, 0]) / np.abs(el_eq[j, 0]))
        print(f"  z0={z:g}: {band.min():+.3e} / "
              f"{np.median(band):+.3e} / {band.max():+.3e}   "
              f"max dev {dev:.2%}")

    plateau = el_all[:, 0, 0]
    eps_10 = np.full(N_TRI, np.nan)
    for i in range(N_TRI):
        dep = (np.abs(el_all[i, 0] - plateau[i])
               > 0.1 * np.abs(plateau[i]))
        if dep.any():
            eps_10[i] = EPS_GRID[int(np.argmax(dep))]
    ratio = eps_10 / d_min
    print(f"on-fault plateau: [{plateau.min():+.4f}, {plateau.max():+.4f}]"
          f"  (equilateral {el_eq[0, 0]:+.4f})")
    print(f"eps_10 / d_min: min {np.nanmin(ratio):.3f}, "
          f"median {np.nanmedian(ratio):.3f}, max {np.nanmax(ratio):.3f}")
    ok = ~np.isnan(eps_10)
    if ok.sum() > 2:
        r = np.corrcoef(np.log(eps_10[ok]), np.log(d_min[ok]))[0, 1]
        print(f"corr(log eps_10, log d_min) = {r:.3f} over {ok.sum()} tris")
    print(f"ray-doubling convergence (worst q): {ray_conv:.2e}")

    # ---- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.13)

    for i in range(N_TRI):
        for j in range(len(HEIGHTS)):
            ax.loglog(EPS_GRID, np.abs(el_all[i, j]), "-",
                      color="0.6", lw=0.35, alpha=0.15)

    eq_style = [("#d62728", r"$z_0 = 0$"),
                ("#1f77b4", r"$z_0 = L \; / \; 4$"),
                ("#2ca02c", r"$z_0 = L$"),
                ("#9467bd", r"$z_0 = 10 L$")]
    for j, (color, label) in enumerate(eq_style):
        ax.loglog(EPS_GRID, np.abs(el_eq[j]), "-", color=color, lw=0.5,
                  label=label)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.6", lw=0.75,
                          label="random shapes"))
    labels.append("random shapes")

    ax.set_xlabel(r"$\varepsilon \; / \; L$")
    ax.set_ylabel(r"$|\sigma_{xz}|$ at the centroid")
    ax.set_xlim(1e-3, 1e0)
    ax.set_ylim(1e-5, 1e1)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticks([1e-5, 1e-3, 1e-1, 1e1])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_box_aspect(1)
    ax.legend(handles, labels, loc="lower right", frameon=True,
              framealpha=1.0, facecolor="white", edgecolor="0.3",
              fontsize=6.5)

    out_dir = os.path.join(ROOT, "cache")
    fig.savefig(os.path.join(out_dir, "fig06_shape_ensemble_gray_preview.pdf"))
    plt.close(fig)
    print("wrote cache/fig06_shape_ensemble_gray_preview.pdf")


if __name__ == "__main__":
    main()

"""VALIDATION FIGURE: the three quantitative claims of the exact-BC
construction, computed fresh (not hardcoded) from the same checks as
``python -m mhf.validate``:

  (a) eps -> 0 convergence of the assembled half-space displacement to
      the classical (cutde) triangular-dislocation solution at
      O(eps^2), across Poisson's ratios (the nu sweep guards the
      lambda-mu stiffness pairing, invisible at nu = 1/4);
  (b) free-surface traction residual versus quadrature order: buried
      triangles sit at machine precision INDEPENDENT of the rule
      (the pointwise cancellation), while the surface-breaking graded
      residual is quadrature-limited and converges spectrally;
  (c) near-trace stress error versus eps for a surface-breaking fault:
      a single evaluation is intrinsically O(eps), and two-level
      Richardson extrapolation restores better than O(eps^2).

Writes manuscript/figures/fig_mhf_validation.{png,pdf}.

Run:
    JAX_PLATFORMS=cpu python -m mhf.demo_validation_figure
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

try:
    from .mindlin import disp_hs_mindlin, field_hs_mindlin, \
        _strike_dip_basis
    from .exact_bc_triangle import stress_dd_kernel
    from . import figures  # noqa: F401  (house rcParams)
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from mhf.mindlin import disp_hs_mindlin, field_hs_mindlin, \
        _strike_dip_basis
    from mhf.exact_bc_triangle import stress_dd_kernel
    from mhf import figures  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "manuscript", "figures")

TRI_BURIED = np.array([[0.0, -1.0, -2.5], [0.0, 1.2, -2.2],
                       [0.0, 0.1, -0.9]])
TRI_SURF = np.array([[0.0, -1.0, -2.0], [0.0, 1.0, -2.0],
                     [0.0, 0.0, 0.0]])

NU_SWEEP = (0.25, 0.30, 0.35)
NU_COLORS = {0.25: "#1f5fa6", 0.30: "#2a7f4f", 0.35: "#c1272d"}


def panel_a_data():
    """eps->0 convergence to cutde at interior obs, per nu."""
    import cutde.halfspace as hs
    obs = np.array([0.8, 0.4, -1.2])
    slip = np.array([1.0, 0.0, 0.0])
    eps_list = np.geomspace(0.4, 0.025, 5)
    out = {}
    for nu in NU_SWEEP:
        uref = hs.disp(obs[None, :], TRI_BURIED[None, :, :],
                       slip[None, :], nu)[0]
        errs = [np.abs(disp_hs_mindlin(obs, TRI_BURIED, [1, 0, 0], nu, e)
                       - uref).max() for e in eps_list]
        out[nu] = (eps_list, np.array(errs))
    return out


def panel_b_data():
    """Free-surface traction residual max|T_3k(z=0)| vs rule order."""
    obs = np.array([0.4, 0.3, 0.0])
    # buried, tensor rule: pointwise cancellation -> machine precision
    # at EVERY order
    n_buried = _strike_dip_basis(TRI_BURIED)[0]
    nq_list = [4, 8, 16, 24, 32]
    buried = []
    for nq in nq_list:
        H = stress_dd_kernel(obs, *TRI_BURIED, n_buried, 1.0, 0.30, 0.1,
                             nq)
        buried.append(np.abs(H[:, 2, :]).max())
    # surface-breaking, graded rule: quadrature-limited, spectral
    n_surf = _strike_dip_basis(TRI_SURF)[0]
    rules = [(12, 10), (16, 12), (20, 16), (24, 20), (28, 24), (32, 24)]
    surf = []
    for n_c, n_t in rules:
        H = stress_dd_kernel(obs, *TRI_SURF, n_surf, 1.0, 0.30, 0.1, 16,
                             graded=True, n_c=n_c, n_t=n_t)
        surf.append(np.abs(H[:, 2, :]).max())
    return (np.array(nq_list, float), np.array(buried),
            np.array([r[0] * r[1] for r in rules], float), np.array(surf))


def panel_c_data():
    """Near-trace stress error vs eps, Richardson levels 1 and 2."""
    import cutde.halfspace as hs
    T1 = np.array([[-5.0, 0.0, -10.0], [5.0, 0.0, -10.0],
                   [5.0, 0.0, 0.0]])
    T2 = np.array([[-5.0, 0.0, -10.0], [5.0, 0.0, 0.0],
                   [-5.0, 0.0, 0.0]])
    obs = np.array([0.0, 2.0, 0.0])
    slip = np.array([1.0, 0.0, 0.0])
    nu, mu = 0.25, 1.0
    e6 = np.zeros(6)
    for T in (T1, T2):
        e6 += np.asarray(hs.strain(obs[None].astype(float),
                                   T[None].astype(float), slip[None],
                                   nu))[0]
    sref_xy = float(np.asarray(hs.strain_to_stress(e6[None], mu, nu))
                    [0][3])
    eps_list = np.array([1.0, 0.5, 0.25])
    out = {}
    for levels in (1, 2):
        errs = []
        for eps in eps_list:
            sig = np.zeros((3, 3))
            for T in (T1, T2):
                f = field_hs_mindlin(obs, T, slip, nu, eps, mu=mu,
                                     n_quad=16, graded=True, n_c=20,
                                     n_t=16, richardson_levels=levels)
                sig += f["stress"]
            errs.append(abs(sig[0, 1] - sref_xy))
        out[levels] = (eps_list, np.array(errs))
    return out


def main():
    a = panel_a_data()
    b_nq, b_buried, b_pts, b_surf = panel_b_data()
    c = panel_c_data()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(11.4, 4.0))

    # (a) eps->0 convergence, nu sweep, slope-2 reference
    for nu in NU_SWEEP:
        e, err = a[nu]
        ax0.loglog(e, err, "o-", ms=4, lw=1.2, color=NU_COLORS[nu],
                   label=rf"$\nu={nu:g}$")
    e_ref = np.array([0.4, 0.025])
    ref0 = a[0.25][1][0]
    ax0.loglog(e_ref, ref0 * (e_ref / 0.4) ** 2, "--", color="0.55",
               lw=0.9)
    ax0.text(0.07, 2.5e-5, r"$\propto \varepsilon^{2}$", color="0.4",
             fontsize=8)
    ax0.set_xlabel(r"$\varepsilon$")
    ax0.set_ylabel(r"max $|u - u_{\mathrm{cutde}}|$")
    ax0.set_xticks([0.025, 0.1, 0.4])
    ax0.set_xticklabels(["0.025", "0.1", "0.4"])
    ax0.set_yticks([1e-5, 1e-4, 1e-3])
    ax0.minorticks_off()
    ax0.legend(frameon=False, fontsize=8, loc="upper left")

    # (b) BC residual vs quadrature points
    ax1.loglog(b_nq ** 2, b_buried, "o-", ms=4, lw=1.2, color="#1f5fa6",
               label="buried, tensor")
    ax1.loglog(b_pts, b_surf, "s-", ms=4, lw=1.2, color="#c1272d",
               label="surface-breaking, graded")
    ax1.axhline(1e-15, color="0.6", lw=0.6, ls=":")
    ax1.text(20, 2.2e-15, "machine precision", color="0.4", fontsize=7.5)
    ax1.set_xlabel(r"quadrature points")
    ax1.set_ylabel(r"max $|T_{3k}(z{=}0)|$")
    ax1.set_ylim(1e-17, 1e-4)
    ax1.set_yticks([1e-16, 1e-10, 1e-4])
    ax1.set_xticks([16, 128, 1024])
    ax1.set_xticklabels(["16", "128", "1024"])
    ax1.minorticks_off()
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # (c) Richardson near-trace rates
    marks = {1: ("o-", "#c1272d", "single evaluation"),
             2: ("s-", "#1f5fa6", "Richardson, 2 levels")}
    for levels in (1, 2):
        e, err = c[levels]
        m, col, lab = marks[levels]
        ax2.loglog(e, err, m, ms=4, lw=1.2, color=col, label=lab)
    ax2.loglog([1.0, 0.25], c[1][1][0] * np.array([1.0, 0.25]), "--",
               color="0.55", lw=0.9)
    ax2.text(0.27, 1.1e-2, r"$\propto \varepsilon$", color="0.4",
             fontsize=8)
    ax2.loglog([1.0, 0.25], c[2][1][0] * np.array([1.0, 0.25]) ** 2,
               ":", color="0.55", lw=0.9)
    ax2.text(0.27, 2.5e-4, r"$\propto \varepsilon^{2}$", color="0.4",
             fontsize=8)
    ax2.set_xlabel(r"$\varepsilon$ (km)")
    ax2.set_ylabel(r"$|\sigma_{xy} - \sigma_{xy}^{\mathrm{cutde}}|$")
    ax2.set_xticks([0.25, 0.5, 1.0])
    ax2.set_xticklabels(["0.25", "0.5", "1"])
    ax2.set_yticks([1e-4, 1e-3, 1e-2])
    ax2.minorticks_off()
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    for k, ax in enumerate((ax0, ax1, ax2)):
        ax.set_box_aspect(1)
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.text(0.06, 0.06, "abc"[k], transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"fig_mhf_validation.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)

    # printed gates for the manuscript text
    for nu in NU_SWEEP:
        e, err = a[nu]
        rate = np.mean(np.log(err[:-1] / err[1:])
                       / np.log(e[:-1] / e[1:]))
        print(f"  (a) nu={nu:g}: mean order {rate:.2f}")
    print(f"  (b) buried residual range "
          f"{b_buried.min():.1e}-{b_buried.max():.1e}; "
          f"surface-breaking {b_surf[0]:.1e} -> {b_surf[-1]:.1e}")
    for levels in (1, 2):
        e, err = c[levels]
        rate = np.mean(np.log(err[:-1] / err[1:])
                       / np.log(e[:-1] / e[1:]))
        print(f"  (c) levels={levels}: mean order {rate:.2f}")


if __name__ == "__main__":
    main()

"""EPS-SWEEP demo: the anelastic eigenstress subtraction keeps the on-fault
elastic stress finite as eps -> 0.

On a fault-normal profile through the fault's interior centroid, the raw
TOTAL |dCFS| peak grows ~ (3/4) mu s / eps (the anelastic eigenstress
C:eps_star of the smeared slip), while the corrected ELASTIC |dCFS| peak —
the library default, `subtract_anelastic=True` — stays bounded and
eps-stable.  Two panels:

  (a) peak |dCFS| vs eps, log-log: raw ~ 1/eps vs corrected flat;
  (b) fault-normal dCFS profiles: raw spikes (off-scale) vs bounded
      corrected curves.

This is the successor of the former demo_sixpanel_noeigen.py: the
subtraction now lives in mhf.anelastic and is the default of
eval_fields / eval_fields_fault_aware, so the six-panel elastic figure is
just `python -m mhf.demo_sixpanel`; this demo keeps the finiteness evidence.

Run:
    python -m mhf.demo_eps_sweep
    python -m mhf.demo_eps_sweep --sweep-eps 2 1 0.5 0.25 0.125 --graded
"""
import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

try:                                   # python -m mhf.demo_eps_sweep
    from .fields import (eval_fields_fault_aware, make_rect_fault,
                         coulomb_stress, moment_factor, _slip_cart_per_tri,
                         _strike_dip_basis, GPA_TO_MPA, MU, EPS)
    from .anelastic import eigenstress_at_points
    from .demo_sixpanel import add_common_args
    from . import figures  # noqa: F401  (applies the house rcParams)
except ImportError:                    # python mhf/demo_eps_sweep.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mhf.fields import (eval_fields_fault_aware, make_rect_fault,
                            coulomb_stress, moment_factor, _slip_cart_per_tri,
                            _strike_dip_basis, GPA_TO_MPA, MU, EPS)
    from mhf.anelastic import eigenstress_at_points
    from mhf.demo_sixpanel import add_common_args
    from mhf import figures  # noqa: F401

RAW_C = "#c1272d"    # warm  -> raw TOTAL (eigenstress-included) stress
COR_C = "#1f5fa6"    # cool  -> corrected (elastic) stress


# ---------------------------------------------------------------------------
# eps-sweep: peak on-fault |dCFS| vs eps, raw vs corrected (finiteness demo)
# ---------------------------------------------------------------------------
def run_eps_sweep(args, eps_list):
    """On a fault-normal profile through the interior centroid, evaluate the
    raw (TOTAL, eigenstress-included) and corrected (elastic) dCFS over an eps
    ladder.  Returns (yy, eps_arr, rows, prof_raw, prof_corr) and prints a
    table."""
    tris = make_rect_fault(half_len=args.half_len, depth=args.fault_depth)
    slip_km0 = np.asarray(args.slip, float) * 1e-3
    cm = getattr(args, "conserve_moment", False)
    n_hat = _strike_dip_basis(tris[0])[0]
    g = "auto" if getattr(args, "graded_auto", False) else args.graded

    # fault-normal profile (vary y) through the interior centroid x=0,
    # z=-fault_depth/2 -> away from the edges, so the corrected curve is the
    # bounded interior stress drop, not the (genuine, eps-capped) tip field.
    z_mid = -0.5 * args.fault_depth
    yy = np.linspace(-3.0, 3.0, 401)
    obs = np.column_stack([np.zeros_like(yy), yy, np.full_like(yy, z_mid)])

    rows, prof_raw, prof_corr = [], [], []
    for eps in eps_list:
        slip_km = slip_km0 / moment_factor(tris, eps) if cm else slip_km0
        slip_cart_all = _slip_cart_per_tri(tris, slip_km)
        s_hat = slip_cart_all[0]
        kw = dict(slip=slip_km, nu=args.nu, eps=eps, n_quad=args.n_quad,
                  graded=g, richardson_levels=args.richardson_levels)
        # one evaluation per eps: raw total, then the library subtraction
        # (identical to subtract_anelastic=True — the wiring identity is
        # gated to 1e-12 by validate group 8)
        _, sig_raw = eval_fields_fault_aware(obs, tris,
                                             subtract_anelastic=False, **kw)
        sig_cor = sig_raw - eigenstress_at_points(obs, tris, slip_cart_all,
                                                  MU, args.nu, eps)
        d_raw = coulomb_stress(sig_raw, n_hat, s_hat, args.friction) * GPA_TO_MPA
        d_cor = coulomb_stress(sig_cor, n_hat, s_hat, args.friction) * GPA_TO_MPA
        pk_raw, pk_cor = float(np.max(np.abs(d_raw))), float(np.max(np.abs(d_cor)))
        rows.append((eps, pk_raw, pk_cor, pk_raw / pk_cor))
        prof_raw.append(d_raw)
        prof_corr.append(d_cor)

    print("\n  eps-sweep: peak |dCFS| (MPa) on a fault-normal profile through "
          "the interior centroid")
    print(f"    {'eps (km)':>9} {'raw':>11} {'corrected':>11} {'raw/corr':>9}")
    for eps, pr, pc, ra in rows:
        print(f"    {eps:9.4g} {pr:11.4g} {pc:11.4g} {ra:9.3g}")
    print("    raw peak ~ (3/4) mu s / eps  ->  grows ~1/eps;   "
          "corrected peak  ->  bounded (elastic stress drop)\n")
    return yy, np.asarray(eps_list, float), rows, prof_raw, prof_corr


def render_eps_sweep(args, yy, eps_arr, rows, prof_raw, prof_corr, stem,
                     exts=("png", "pdf")):
    """Two-panel finiteness figure: (a) peak |dCFS| vs eps on log-log (raw ~1/eps,
    corrected flat); (b) fault-normal dCFS profiles (raw spikes vs bounded
    corrected curves)."""
    eps_arr = np.asarray(eps_arr, float)
    pk_raw = np.array([r[1] for r in rows])
    pk_cor = np.array([r[2] for r in rows])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 4.3))

    # (a) peak |dCFS| vs eps -------------------------------------------------
    ax0.loglog(eps_arr, pk_raw, "o-", color=RAW_C, lw=1.4, ms=5,
               label="raw TOTAL (with eigenstress)")
    ax0.loglog(eps_arr, pk_cor, "s-", color=COR_C, lw=1.4, ms=5,
               label="corrected (elastic)")
    ref = pk_raw[np.argmax(eps_arr)] * (eps_arr.max() / eps_arr)   # slope -1
    ax0.loglog(eps_arr, ref, "--", color="0.55", lw=0.9)
    ax0.text(eps_arr.min(), ref.max(), r"$\propto 1/\varepsilon$",
             color="0.4", ha="left", va="bottom", fontsize=8)
    ax0.set_xlabel(r"$\varepsilon$ (km)")
    ax0.set_ylabel(r"peak $|\Delta\mathrm{CFS}|$ (MPa)")
    ax0.set_xticks([eps_arr.min(), np.sqrt(eps_arr.min() * eps_arr.max()),
                    eps_arr.max()])
    lo = 10 ** math.floor(math.log10(min(pk_cor.min(), pk_raw.min())))
    hi = 10 ** math.ceil(math.log10(pk_raw.max()))
    ax0.set_yticks([10 ** k for k in
                    range(int(round(math.log10(lo))), int(round(math.log10(hi))) + 1)])
    for ax in (ax0,):
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
    ax0.set_box_aspect(1)
    ax0.legend(frameon=False, fontsize=8, loc="lower left")
    ax0.tick_params(direction="out", length=3, width=0.8)

    # (b) fault-normal dCFS profiles ----------------------------------------
    for k in range(len(eps_arr)):
        ax1.plot(yy, prof_raw[k], "-", color=RAW_C, lw=1.0, alpha=0.85)
        ax1.plot(yy, prof_corr[k], "-", color=COR_C, lw=1.3)
    ax1.axhline(0.0, color="0.6", lw=0.6)
    # focus the y-window on the (bounded) corrected curves; raw spikes clip out
    cor_all = np.concatenate(prof_corr)
    c_lo, c_hi = float(cor_all.min()), float(cor_all.max())
    span = max(c_hi - c_lo, 1e-9)
    ax1.set_ylim(c_lo - 0.35 * span, c_hi + 1.1 * span)   # headroom: raw enters top
    ax1.set_xlim(-3, 3)
    ax1.set_xticks([-3, 0, 3])
    ax1.set_xlabel(r"$y$ (km, fault-normal)")
    ax1.set_ylabel(r"$\Delta\mathrm{CFS}$ (MPa)")
    ax1.set_box_aspect(1)
    handles = [Line2D([0], [0], color=RAW_C, lw=1.2),
               Line2D([0], [0], color=COR_C, lw=1.4)]
    ax1.legend(handles, ["raw TOTAL (with eigenstress)", "corrected (elastic)"],
               frameon=False, fontsize=8, loc="lower right")
    ax1.text(0.5, 0.97, r"raw spike $\to$ off-scale ($\propto 1/\varepsilon$)",
             transform=ax1.transAxes, ha="center", va="top", fontsize=7.5,
             color=RAW_C)
    ax1.tick_params(direction="out", length=3, width=0.8)

    for k, ax in enumerate((ax0, ax1)):
        ax.text(0.96, 0.96, "ab"[k], transform=ax.transAxes, ha="right",
                va="top", fontsize=9)
    fig.suptitle(r"Eigenstress-corrected $\Delta$CFS stays finite as "
                 r"$\varepsilon\to0$ (interior on-fault)", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outs = []
    for ext in exts:
        out = f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        outs.append(out)
    plt.close(fig)
    return outs


# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Eps-sweep finiteness demo: raw TOTAL on-fault dCFS grows "
                     "~1/eps while the eigenstress-corrected elastic dCFS stays "
                     "bounded."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)
    p.add_argument("--eps", "--epsilon", dest="eps", type=float, default=EPS,
                   help="Cortez mollification parameter epsilon (km); unused "
                        "by the sweep itself (kept for add_common_args parity)")
    p.add_argument("--sweep-eps", type=float, nargs="+",
                   default=[2.0, 1.0, 0.5, 0.25, 0.125],
                   help="epsilon ladder (km) for the finiteness sweep")
    p.add_argument("--sweep-out", default="fig_mhf_epssweep",
                   help="eps-sweep output stem (PNG+PDF), written in mhf/")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    print(f"Eps-sweep finiteness demo: fault {2*args.half_len:g} x "
          f"{args.fault_depth:g} km at y=0, slip={args.slip} m, "
          f"ladder={args.sweep_eps} km")
    here = os.path.dirname(os.path.abspath(__file__))
    yy, eps_arr, rows, pr, pc = run_eps_sweep(args, args.sweep_eps)
    for out in render_eps_sweep(args, yy, eps_arr, rows, pr, pc,
                                os.path.join(here, args.sweep_out)):
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

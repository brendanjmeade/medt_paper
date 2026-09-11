"""SIX-PANEL demo: x-displacement, Coulomb failure stress, and von Mises
stress for a vertical strike-slip fault — map view (top row) and across-strike
cross-section (bottom row).

  columns:  u_x (mm, PiYG)  |  dCFS (MPa, RdYlBu)  |  sigma_vM (MPa, cool, log)
  rows:     map view (x-y slice at z0)  /  cross-section (y-z plane at x=0)

dCFS = s.sig.n + f*(n.sig.n) on receiver planes parallel to the source fault
with the source slip sense (f = --friction).  Stress panels show the ELASTIC
stress by default: the anelastic eigenstress C:eps_star (the smeared slip
itself, on-fault peak (3/4) mu s/eps, diverging as eps->0) is subtracted, so
the on-fault dCFS is the bounded, eps-stable classical signature — stress
drop broadside (including on-fault pixels), positive lobes beyond the patch
edges.  Pass --total to plot the raw TOTAL stress (eigenstress included;
deprecated fault-zone reading, comparison only — see mhf.anelastic).

The compute/render split (compute_fields / render_sixpanel) is reused by
mhf.demo_eps_animation, which renders frames over an eps-ladder with FROZEN
color limits.

Run:
    python mhf/demo_sixpanel.py
    python -m mhf.demo_sixpanel --graded --richardson-levels 2 --n-quad 16
    python mhf/demo_sixpanel.py --slip 0 1 0 --friction 0.6   # dip-slip
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:                                   # python -m mhf.demo_sixpanel
    from .fields import (build_grid, build_grid_section,
                         eval_fields_fault_aware, make_rect_fault,
                         fault_plane_segments, coulomb_stress, von_mises,
                         moment_factor, _slip_cart_per_tri, _strike_dip_basis,
                         KM_TO_MM, GPA_TO_MPA, NU, EPS)
    from .figures import _fmt          # also applies the house rcParams
except ImportError:                    # python mhf/demo_sixpanel.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mhf.fields import (build_grid, build_grid_section,
                            eval_fields_fault_aware, make_rect_fault,
                            fault_plane_segments, coulomb_stress, von_mises,
                            moment_factor, _slip_cart_per_tri,
                            _strike_dip_basis, KM_TO_MM, GPA_TO_MPA, NU, EPS)
    from mhf.figures import _fmt

X0 = 0.0      # along-strike position of the cross-section

CMAPS = ["PiYG", "RdYlBu_r", "cool"]   # u_x | dCFS | von Mises (per column)
KINDS = ["div", "div", "log"]          # (RdYlBu_r: red = positive dCFS = promoting)
N_BANDS = 7                            # filled contourf bands per panel


def add_common_args(p):
    """Geometry / physics / accuracy options shared with the animation demo."""
    p.add_argument("--half-extent", type=float, default=15.0,
                   help="map half-extent (km)")
    p.add_argument("--n", type=int, default=121, help="map grid points per axis")
    p.add_argument("--z0", type=float, default=-2.0,
                   help="map slice depth (km, <= 0)")
    p.add_argument("--width", type=float, default=40.0,
                   help="section width (km); y in [-w/2, w/2]")
    p.add_argument("--depth", type=float, default=20.0,
                   help="section depth (km); z in [-depth, 0]")
    p.add_argument("--ny", type=int, default=161, help="section points in y")
    p.add_argument("--nz", type=int, default=81, help="section points in z")
    p.add_argument("--half-len", type=float, default=5.0,
                   help="fault half-length along strike (km)")
    p.add_argument("--fault-depth", type=float, default=10.0,
                   help="fault bottom depth (km)")
    p.add_argument("--slip", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   metavar=("S", "D", "T"),
                   help="slip [strike dip tensile] in metres")
    p.add_argument("--friction", type=float, default=0.4,
                   help="friction coefficient f in dCFS")
    p.add_argument("--nu", type=float, default=NU, help="Poisson's ratio")
    p.add_argument("--n-quad", type=int, default=8,
                   help="correction-quadrature order (>=16 near the trace)")
    p.add_argument("--graded", action="store_true",
                   help="graded correction quadrature (surface-breaking faults)")
    p.add_argument("--graded-auto", action="store_true",
                   help="graded rule only for obs near the fault (within "
                        "2*diam/n_quad), tensor elsewhere: removes the "
                        "near-trace tensor-rule artifact at ~tensor cost")
    p.add_argument("--richardson-levels", type=int, default=1,
                   help="eps-extrapolation levels (off-fault; use WITH --graded)")
    p.add_argument("--conserve-moment", action="store_true",
                   help="rescale slip by 1/moment_factor(tris, eps) so the "
                        "classical seismic moment is preserved at finite eps "
                        "(surface-breaking faults lose M0*eps/(4D) of potency "
                        "above the free surface; see mhf/moment_accounting.tex). "
                        "NOTE: also raises in-zone stresses by the same factor — "
                        "trades zone-physics fidelity for moment fidelity.")
    p.add_argument("--total", dest="subtract_anelastic", action="store_false",
                   help="plot the raw TOTAL stress (anelastic eigenstress "
                        "C:eps_star included; on-fault ~ (3/4) mu s/eps, "
                        "diverging as eps->0).  Deprecated fault-zone reading, "
                        "for comparison only.  Default: subtract, plot the "
                        "elastic stress.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Six-panel figure: u_x, Coulomb failure stress, and von "
                     "Mises stress in map view (top) and cross-section "
                     "(bottom) for a vertical strike-slip fault."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)
    p.add_argument("--eps", "--epsilon", dest="eps", type=float, default=EPS,
                   help="Cortez mollification parameter epsilon (km); "
                        "typically ~fault_width/10")
    p.add_argument("--out", default="fig_mhf_sixpanel",
                   help="output filename stem (PNG+PDF), written in mhf/")
    return p.parse_args(argv)


def compute_fields(args, eps):
    """Evaluate the six panel fields at one eps.  Returns a dict with grids,
    panel data (top/bot = [u_x mm, dCFS MPa, sigma_vM MPa]), and overlays."""
    tris = make_rect_fault(half_len=args.half_len, depth=args.fault_depth)
    slip_km = np.asarray(args.slip, float) * 1e-3
    if getattr(args, "conserve_moment", False):
        slip_km = slip_km / moment_factor(tris, eps)
    n_hat = _strike_dip_basis(tris[0])[0]                  # receiver normal
    s_hat = _slip_cart_per_tri(tris, slip_km)[0]           # slip sense
    g = "auto" if getattr(args, "graded_auto", False) else args.graded
    kw = dict(slip=slip_km, nu=args.nu, eps=eps, n_quad=args.n_quad,
              graded=g, richardson_levels=args.richardson_levels,
              subtract_anelastic=getattr(args, "subtract_anelastic", True))

    X, Y, obs_m = build_grid(half_extent=args.half_extent, n=args.n, z0=args.z0)
    u_m, sig_m = eval_fields_fault_aware(obs_m, tris, **kw)
    y_half, z_bot = args.width / 2.0, -abs(args.depth)
    Yc, Zc, obs_c = build_grid_section(x0=X0, y_half=y_half, z_bot=z_bot,
                                       ny=args.ny, nz=args.nz)
    u_c, sig_c = eval_fields_fault_aware(obs_c, tris, **kw)

    def _row(u, sig, shape):
        ux = (u[:, 0] * KM_TO_MM).reshape(shape)
        cfs = (coulomb_stress(sig, n_hat, s_hat, args.friction)
               * GPA_TO_MPA).reshape(shape)
        vm = (von_mises(sig) * GPA_TO_MPA).reshape(shape)
        return [ux, cfs, vm]

    return dict(X=X, Y=Y, Yc=Yc, Zc=Zc,
                top=_row(u_m, sig_m, X.shape), bot=_row(u_c, sig_c, Yc.shape),
                ov_top=fault_plane_segments(tris, 2, args.z0),
                ov_bot=fault_plane_segments(tris, 0, X0),
                y_half=y_half, z_bot=z_bot)


def panel_limits(fields_list, log_pct=None):
    """Shared per-column color limits over one or more compute_fields results:
    'div' columns -> 99th-percentile |.| over all frames/rows; 'log' column ->
    4 decades below the global max, or below the log_pct percentile when set.
    Pass log_pct (e.g. 99) for animations over wide eps ladders: the on-fault
    pixels grow ~1/eps and a max-based scale would flatten everything else."""
    lims = []
    for j, kind in enumerate(KINDS):
        data = np.concatenate(
            [np.abs(f[row][j]).ravel() for f in fields_list
             for row in ("top", "bot")])
        if kind == "div":
            vmax = float(np.percentile(data, 99.0))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(np.max(data)) or 1.0
            lims.append(vmax)
        else:
            if log_pct is None:
                vmax = float(np.max(data)) or 1.0
            else:
                vmax = float(np.percentile(data[data > 0], log_pct)) or 1.0
            lims.append((vmax * 1e-4, vmax))
    return lims


def _panel(ax, fig, A, B, data, kind, cmap, lim, overlays):
    if kind == "div":
        vmax = lim
        levels = np.linspace(-vmax, vmax, N_BANDS + 1)
        im = ax.contourf(A, B, data, levels=levels, cmap=cmap, extend="both")
        ticks, labels = [-vmax, 0, vmax], [_fmt(-vmax), "0", _fmt(vmax)]
    else:
        vmin, vmax = lim
        levels = np.geomspace(vmin, vmax, N_BANDS + 1)
        im = ax.contourf(A, B, np.maximum(data, vmin), levels=levels,
                         cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                         extend="both")
        ticks = [vmin, np.sqrt(vmin * vmax), vmax]
        labels = [_fmt(t) for t in ticks]
    for seg in overlays:
        ax.plot(seg[:, 0], seg[:, 1], "k-", lw=1.0)
    ax.set_aspect("equal")
    ax.tick_params(direction="out", length=3, width=0.8)
    # inset_axes is a child of ax -> the bar matches the RENDERED axes height
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(labels)
    cb.ax.tick_params(labelsize=7)


def render_sixpanel(args, F, eps, outpath_stem, limits=None, exts=("png", "pdf")):
    """Render one six-panel figure from compute_fields output F.  limits=None
    autoscales (single figure); pass panel_limits(...) output for frozen scales
    (animation frames)."""
    if limits is None:
        limits = panel_limits([F])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4),
                             gridspec_kw=dict(wspace=0.42, hspace=0.30,
                                              height_ratios=[1.0, 0.62]))
    titles = [r"$u_x$ (mm)", rf"$\Delta$CFS (MPa), $f={args.friction:g}$",
              r"$\sigma_{vM}$ (MPa)"]
    h = args.half_extent
    for j in range(3):
        ax = axes[0, j]
        _panel(ax, fig, F["X"], F["Y"], F["top"][j], KINDS[j], CMAPS[j],
               limits[j], F["ov_top"])
        ax.set_title(titles[j], pad=3)
        ax.set_xlim(-h, h); ax.set_ylim(-h, h)
        ax.set_xticks([-int(h), 0, int(h)]); ax.set_yticks([-int(h), 0, int(h)])
        ax.set_xlabel(r"$x$ (km)")
        ax = axes[1, j]
        _panel(ax, fig, F["Yc"], F["Zc"], F["bot"][j], KINDS[j], CMAPS[j],
               limits[j], F["ov_bot"])
        ax.set_xlim(-F["y_half"], F["y_half"]); ax.set_ylim(F["z_bot"], 0.0)
        ax.set_xticks([-int(F["y_half"]), 0, int(F["y_half"])])
        ax.set_yticks([int(F["z_bot"]), int(round(F["z_bot"] / 2)), 0])
        ax.set_xlabel(r"$y$ (km)")
    axes[0, 0].set_ylabel(r"$y$ (km)")
    axes[1, 0].set_ylabel(r"$z$ (km)")
    for k, ax in enumerate(axes.flat):
        ax.text(0.96, 0.95, "abcdef"[k], transform=ax.transAxes, ha="right",
                va="top", fontsize=9,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                          alpha=0.7))
    s, d, t = args.slip

    def _len_str(km):                       # adaptive units: km above 100 m
        return f"{km:.4g}" + r"\,\mathrm{km}" if km >= 0.1 else \
               f"{km * 1e3:.4g}" + r"\,\mathrm{m}"

    cm_note = ""
    if getattr(args, "conserve_moment", False):
        tris = make_rect_fault(half_len=args.half_len, depth=args.fault_depth)
        cm_note = (r", moment-conserving slip $\times"
                   rf"{1.0 / moment_factor(tris, eps):.3f}$")
    stress_tag = (r"elastic stress ($C\!:\!\varepsilon^*$ subtracted)"
                  if getattr(args, "subtract_anelastic", True)
                  else r"TOTAL stress (incl. $C\!:\!\varepsilon^*$)")
    fig.suptitle(
        r"Mollified half-space: map view (top, $z=%g$ km) and cross-section "
        r"(bottom, $x=0$)" % args.z0 + "\n"
        rf"$\nu={args.nu:g}$, slip $(s,d,t)=({s:g},{d:g},{t:g})$ m, "
        rf"$\varepsilon={_len_str(eps)}$, " + stress_tag + cm_note,
        fontsize=10.5, y=0.99)
    outs = []
    for ext in exts:
        out = f"{outpath_stem}.{ext}"
        fig.savefig(out)
        outs.append(out)
    plt.close(fig)
    return outs


def main(argv=None):
    args = _parse_args(argv)
    print(f"Six-panel (u_x, dCFS f={args.friction:g}, von Mises): fault "
          f"{2*args.half_len:g} x {args.fault_depth:g} km at y=0, "
          f"slip={args.slip} m, eps={args.eps:g} km")
    F = compute_fields(args, args.eps)
    here = os.path.dirname(os.path.abspath(__file__))
    for out in render_sixpanel(args, F, args.eps,
                               os.path.join(here, args.out)):
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

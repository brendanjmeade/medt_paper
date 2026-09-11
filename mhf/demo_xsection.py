"""CROSS-SECTION demo: mollified half-space fields of a vertical strike-slip
fault on the vertical y-z plane at x=0 (across strike, through the fault
midpoint) — 3x3 panels (u_x, u_y, u_z + 6 unique stress components).

The fault is a surface-breaking vertical rectangle striking along x (trace at
y=0), tiled by two conformal triangles, with uniform [strike, dip, tensile]
slip (default 1 m strike-slip).  In the section it appears as a vertical line
at y=0 from the free surface down to z=-fault_depth.

Run:
    python -m mhf.demo_xsection
    python -m mhf.demo_xsection --graded --richardson-levels 2 --n-quad 16
    python -m mhf.demo_xsection --width 30 --depth 15 --fault-depth 8 --eps 1.5
"""
import argparse

import numpy as np

try:                                   # python -m mhf.demo_xsection
    from .fields import (build_grid_section, eval_fields_fault_aware,
                         make_rect_fault, report_ranges, NU, EPS)
    from .figures import plot_9panel_section
except ImportError:                    # python mhf/demo_xsection.py
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mhf.fields import (build_grid_section, eval_fields_fault_aware,
                            make_rect_fault, report_ranges, NU, EPS)
    from mhf.figures import plot_9panel_section

X0 = 0.0      # along-strike position of the section


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Vertical cross-section (y-z at x=0) of the mollified "
                     "half-space fields of a vertical strike-slip fault."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--width", type=float, default=40.0,
                   help="full across-fault width (km); y in [-w/2, w/2]")
    p.add_argument("--depth", type=float, default=20.0,
                   help="section depth (km); z in [-depth, 0]")
    p.add_argument("--ny", type=int, default=161,
                   help="observation points across fault (y)")
    p.add_argument("--nz", type=int, default=81,
                   help="observation points in depth (z)")
    p.add_argument("--half-len", type=float, default=5.0,
                   help="fault half-length along strike (km)")
    p.add_argument("--fault-depth", type=float, default=10.0,
                   help="fault bottom depth (km); fault spans z in [-d, 0]")
    p.add_argument("--slip", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   metavar=("S", "D", "T"),
                   help="slip [strike dip tensile] in metres")
    p.add_argument("--eps", "--epsilon", dest="eps", type=float, default=EPS,
                   help="Cortez mollification parameter epsilon (km); "
                        "typically ~fault_width/10")
    p.add_argument("--nu", type=float, default=NU, help="Poisson's ratio")
    p.add_argument("--n-quad", type=int, default=16,
                   help="correction-quadrature order (>=16 resolves the "
                        "near-singular integrand at the surface-breaking trace)")
    p.add_argument("--graded", action="store_true",
                   help="obs-dependent graded correction quadrature "
                        "(recommended for surface-breaking faults at small eps)")
    p.add_argument("--richardson-levels", type=int, default=1,
                   help="eps-extrapolation levels (1=off; 2 -> O(eps^2) near "
                        "the trace; applied off-fault only — on the fault the "
                        "regularized single-eps value is kept). Use WITH --graded.")
    p.add_argument("--total", dest="subtract_anelastic", action="store_false",
                   help="plot the raw TOTAL stress (anelastic eigenstress "
                        "C:eps_star included; on-fault ~ (3/4) mu s/eps). "
                        "Default: subtract, plot the elastic stress.")
    p.add_argument("--out", default="fig_mhf_xsection",
                   help="output filename stem (PNG+PDF), written in mhf/")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    y_half = args.width / 2.0
    z_bot = -abs(args.depth)
    tris = make_rect_fault(half_len=args.half_len, depth=args.fault_depth)
    slip_km = np.asarray(args.slip, float) * 1e-3
    if args.fault_depth > abs(args.depth):
        print(f"  note: fault_depth ({args.fault_depth} km) exceeds section "
              f"depth ({abs(args.depth)} km); the fault line is clipped.")
    print(f"Cross-section at x=0  (width={args.width} km, "
          f"depth={abs(args.depth)} km, ny={args.ny}, nz={args.nz}, "
          f"fault_depth={args.fault_depth} km, slip={args.slip} m, "
          f"eps={args.eps:g} km):")
    Y, Z, obs = build_grid_section(x0=X0, y_half=y_half, z_bot=z_bot,
                                   ny=args.ny, nz=args.nz)
    if args.richardson_levels > 1 or args.graded:
        print(f"  near-surface accuracy: graded={args.graded}, "
              f"richardson_levels={args.richardson_levels}")
    u, sig = eval_fields_fault_aware(
        obs, tris, slip=slip_km, nu=args.nu, eps=args.eps, n_quad=args.n_quad,
        graded=args.graded, richardson_levels=args.richardson_levels,
        subtract_anelastic=args.subtract_anelastic)
    report_ranges(u, sig)
    s, d, t = args.slip
    stress_tag = (r"elastic stress ($C\!:\!\varepsilon^*$ subtracted)"
                  if args.subtract_anelastic
                  else r"TOTAL stress (incl. $C\!:\!\varepsilon^*$)")
    title = (r"Mollified half-space: vertical fault, "
             r"cross-section across strike at $x=0$"
             "\n"
             rf"$\nu={args.nu:g}$, $\varepsilon={args.eps:g}$ km, "
             rf"slip $(s,d,t)=({s:g},{d:g},{t:g})$ m "
             rf"(fault $y=0$, $z\in[{-args.fault_depth:g},0]$), " + stress_tag)
    plot_9panel_section(Y, Z, u, sig, tris, X0, title, args.out,
                        y_half=y_half, z_bot=z_bot)


if __name__ == "__main__":
    main()

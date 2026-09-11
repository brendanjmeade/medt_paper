"""MAP-VIEW demo: mollified half-space fields of a vertical strike-slip fault
on a horizontal slice — 3x3 panels (u_x, u_y, u_z + 6 unique stress components).

The fault is a surface-breaking vertical rectangle striking along x (trace at
y=0), tiled by two conformal triangles, with uniform [strike, dip, tensile]
slip (default 1 m strike-slip).

Run:
    python -m mhf.demo_mapview
    python -m mhf.demo_mapview --z0 -0.0 --eps 1.0 --graded --richardson-levels 2
    python -m mhf.demo_mapview --slip 0 1 0 --fault-depth 8     # dip-slip
"""
import argparse

import numpy as np

try:                                   # python -m mhf.demo_mapview
    from .fields import (build_grid, eval_fields_fault_aware, make_rect_fault,
                         report_ranges, NU, EPS)
    from .figures import plot_9panel
except ImportError:                    # python mhf/demo_mapview.py
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mhf.fields import (build_grid, eval_fields_fault_aware,
                            make_rect_fault, report_ranges, NU, EPS)
    from mhf.figures import plot_9panel


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Map view (x-y slice) of the mollified half-space fields "
                     "of a vertical strike-slip fault."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--half-extent", type=float, default=15.0,
                   help="map half-extent (km); x,y in [-h, h]")
    p.add_argument("--n", type=int, default=121, help="grid points per axis")
    p.add_argument("--z0", type=float, default=-2.0,
                   help="slice depth (km, <= 0); 0 = free surface")
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
    p.add_argument("--n-quad", type=int, default=8,
                   help="correction-quadrature order (>=16 near the trace)")
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
    p.add_argument("--out", default="fig_mhf_mapview",
                   help="output filename stem (PNG+PDF), written in mhf/")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    tris = make_rect_fault(half_len=args.half_len, depth=args.fault_depth)
    slip_km = np.asarray(args.slip, float) * 1e-3
    print(f"Map view at z={args.z0} km  (fault {2*args.half_len:g} x "
          f"{args.fault_depth:g} km at y=0, slip={args.slip} m, "
          f"eps={args.eps:g} km):")
    X, Y, obs = build_grid(half_extent=args.half_extent, n=args.n, z0=args.z0)
    u, sig = eval_fields_fault_aware(
        obs, tris, slip=slip_km, nu=args.nu, eps=args.eps, n_quad=args.n_quad,
        graded=args.graded, richardson_levels=args.richardson_levels,
        subtract_anelastic=args.subtract_anelastic)
    report_ranges(u, sig)
    s, d, t = args.slip
    stress_tag = (r"elastic stress ($C\!:\!\varepsilon^*$ subtracted)"
                  if args.subtract_anelastic
                  else r"TOTAL stress (incl. $C\!:\!\varepsilon^*$)")
    title = (r"Mollified half-space: vertical fault, map view"
             "\n"
             rf"slice $z={args.z0:g}$ km, $\nu={args.nu:g}$, "
             rf"$\varepsilon={args.eps:g}$ km, "
             rf"slip $(s,d,t)=({s:g},{d:g},{t:g})$ m, " + stress_tag)
    plot_9panel(X, Y, u, sig, tris, args.z0, title, args.out,
                half_extent=args.half_extent)


if __name__ == "__main__":
    main()

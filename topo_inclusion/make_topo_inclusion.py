"""Topography + inclusion showcase: solve the four-state decomposition.

The fig06 soft-inclusion model (host + mu_host/100 cylindrical inclusion
+ surface-breaking strike-slip fault) gains a compactly supported
Gaussian hill east of the fault:

    h(x,y) = 2 km * exp(-r^2 / (2 * 30^2)) * S(r),  r about (100, 0),
    S = quintic taper on [75, 90] km, EXACTLY zero beyond 90 km

so the topography never touches the inclusion (support clears the
inclusion rim by >=125 km, the box edges by 100 km, and the fault trace
by 10 km — the trace itself stays flat).

Four solves on the SAME refined triangulation (warped vs flat, het vs
homogeneous inclusion), all with the calibrated formulation:

    (topo, het)   (topo, hom)   (flat, het)   (flat, hom)

enabling same-connectivity decompositions:
    inclusion effect  Du_inc  = u(topo,het) - u(topo,hom)
    topography effect Du_topo = u(topo,het) - u(flat,het)

Caches fields to benchmarks/topo_inclusion_fields.npz for
benchmarks/render_topo_inclusion.py.

Run from the moss2 repo root:  python benchmarks/make_topo_inclusion.py
"""

import gc
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from inclusion_mesh import (_circle_boundary, _rectangle_boundary,
                            make_inclusion_geometry)
from local_box_mesh_eq import make_vertical_fault_eq

from assess_fig06_inclusion import MAT_HOST, build_model
from mbem.backends.dense import AssembledDense
from mbem.model import generate_system
from mbem.topography import (apply_topography, assert_zero_clearance,
                             gaussian_bump)

EPS = 3.0
BUMP_CENTER = (100.0, 0.0)
BUMP_HEIGHT = 2.0          # km of relief
BUMP_SIGMA = 30.0          # km
BUMP_EDGE = 9.0            # km, target triangle edge inside the support
FAULT_DEPTH = 20.0         # km (taper depth for fault-conforming warps)
OUT = pathlib.Path(__file__).parent / "topo_inclusion_fields.npz"


def build(bump_center=BUMP_CENTER, bump_sigma=BUMP_SIGMA,
          bump_height=BUMP_HEIGHT):
    """fig06 geometry + refinement disk under the bump + the warp.

    If the bump's support reaches the fault trace, the fault mesh is
    warped consistently (same h, depth-tapered) so its top edge conforms
    to the deformed free surface — topography may overlap the fault.
    The inclusion rim and box edges must always stay clear.
    """
    bump = gaussian_bump(bump_center, bump_height, bump_sigma,
                         taper=(2.5, 3.0))
    fault_trace = np.array([[0.0, -100.0], [0.0, 100.0]])
    meshes = make_inclusion_geometry(
        x_range=(-200.0, 200.0), y_range=(-200.0, 200.0), z_bottom=-200.0,
        inclusion_center_xy=(-100.0, 100.0), inclusion_radius=75.0,
        inclusion_depth=50.0,
        target_edge_inclusion=6.0, target_edge_top=9.0,
        target_edge_far=80.0, target_edge_side=80.0,
        fault_trace=fault_trace, fault_edge=4.5, host_top_max_edge=20.0,
        top_refine_disks=[(bump_center, bump.support_radius, BUMP_EDGE,
                           0.5 * BUMP_EDGE ** 2)])
    fault_mesh, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=200.0, depth_range=(-20.0, 0.0), target_edge=4.5)

    # Hard watertightness guards (curves shared with UNWARPED patches).
    rim, _ = _circle_boundary((-100.0, 100.0), 75.0, 6.0)
    rect = _rectangle_boundary((-200.0, 200.0), (-200.0, 200.0), 9.0)
    for label, pts in (("inclusion rim", rim), ("box edge", rect)):
        assert_zero_clearance(bump, pts, tol=0.0, label=label)

    # Fault: equal-warp rule. If topography reaches the trace, warp the
    # fault mesh with the same h (full at z=0, zero at the fault bottom).
    trace = fault_mesh.vertices[np.isclose(fault_mesh.vertices[:, 2], 0.0),
                                :2]
    h_trace = float(np.abs(bump(trace[:, 0], trace[:, 1])).max())
    if h_trace > 0.0:
        fault_topo = apply_topography(fault_mesh, bump,
                                      taper_depth=FAULT_DEPTH)
        print(f"  bump overlaps fault trace (max h on trace "
              f"{h_trace*1000:.0f} m): fault mesh warped consistently",
              flush=True)
    else:
        fault_topo = fault_mesh

    host_top_flat = meshes["host_top"]
    host_top_topo = apply_topography(host_top_flat, bump)
    return (meshes, host_top_flat, host_top_topo, fault_mesh, fault_topo,
            s_hat, bump)


def main(mu_inc: float = 0.3, lam_inc: float | None = None,
         out: pathlib.Path = OUT, bump_center=BUMP_CENTER,
         bump_sigma=BUMP_SIGMA, bump_height=BUMP_HEIGHT):
    import mollified_bem as mb
    mat_inc = mb.ElasticMaterial(mu=mu_inc,
                                 lam=mu_inc if lam_inc is None else lam_inc)
    (meshes, top_flat, top_topo, fault_flat, fault_topo,
     s_hat, bump) = build(bump_center, bump_sigma, bump_height)
    for k, m in meshes.items():
        print(f"  {k:16s}: {m.n_triangles:5d} tri", flush=True)
    print(f"  bump: H={bump.height} km, sigma={bump.sigma} km, "
          f"support r={bump.support_radius} km at {bump.center_xy}")
    print(f"  inclusion: mu={mat_inc.mu} GPa (host 30)", flush=True)

    fields = {}
    conds = {}
    for surface, host_top, fault_mesh in (("topo", top_topo, fault_topo),
                                          ("flat", top_flat, fault_flat)):
        meshes["host_top"] = host_top
        model = build_model(meshes, fault_mesh, s_hat, mat_inc=mat_inc)
        system = generate_system(model)
        print(f"[{surface}] unknowns: {system.layout.n_unknowns}",
              flush=True)

        asm = AssembledDense(system, EPS, "direct", jump="calibrated")
        sol = asm.solve()
        conds[f"{surface}_het"] = asm.cond_estimate
        for p in ("host_top", "inclusion_top"):
            fields[f"u_{p}_{surface}_het"] = sol[f"u:{p}"]
        print(f"[{surface}] het: cond {asm.cond_estimate:.3e}", flush=True)

        asm.A = None
        asm._lu = None
        gc.collect()
        asm_h = asm.rebuild_for_materials({"inclusion": MAT_HOST})
        del asm
        gc.collect()
        sol_h = asm_h.solve()
        conds[f"{surface}_hom"] = asm_h.cond_estimate
        for p in ("host_top", "inclusion_top"):
            fields[f"u_{p}_{surface}_hom"] = sol_h[f"u:{p}"]
        print(f"[{surface}] hom: cond {asm_h.cond_estimate:.3e}", flush=True)
        del asm_h, sol, sol_h, model, system
        gc.collect()

    v = top_flat.vertices
    np.savez(
        out,
        host_top_vertices=v,
        host_top_triangles=top_flat.triangles,
        host_top_h=bump(v[:, 0], v[:, 1]),
        inclusion_top_vertices=meshes["inclusion_top"].vertices,
        inclusion_top_triangles=meshes["inclusion_top"].triangles,
        bump_center=np.array(bump.center_xy),
        bump_height=bump.height, bump_sigma=bump.sigma,
        bump_support=bump.support_radius,
        mu_inc=mat_inc.mu, lam_inc=mat_inc.lam,
        **fields,
        **{f"cond_{k}": val for k, val in conds.items()},
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-inc", type=float, default=0.3,
                    help="inclusion shear modulus, GPa (host is 30)")
    ap.add_argument("--lam-inc", type=float, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--bump-center", type=float, nargs=2,
                    default=list(BUMP_CENTER), metavar=("X", "Y"))
    ap.add_argument("--bump-sigma", type=float, default=BUMP_SIGMA)
    ap.add_argument("--bump-height", type=float, default=BUMP_HEIGHT)
    a = ap.parse_args()
    main(mu_inc=a.mu_inc, lam_inc=a.lam_inc, out=a.out,
         bump_center=tuple(a.bump_center), bump_sigma=a.bump_sigma,
         bump_height=a.bump_height)

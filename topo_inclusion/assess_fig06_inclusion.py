"""Assess fig06 (soft-inclusion showcase) against the resonance finding.

The moss-repo figure (manuscript/figures/fig06_showcase_heterogeneous)
solves a host + soft cylindrical inclusion (mu_inc = mu_host/100 = 0.3
GPa) with the half-jump formulation. Extreme contrast + interface
creases (cylinder wall meets the free surface and the bottom disk) is
precisely the risk profile of the spurious-resonance defect.

This script, at a reduced (~10k DOF) resolution:
  1. rebuilds the same geometry via the (verbatim-copied) inclusion_mesh
     generator and expresses it as an mbem RegionModel;
  2. validates the region core against the ORIGINAL moss solver
     (static_box_two_region.solve_two_region_box) as an oracle on this
     topology — half-jump mode must reproduce it;
  3. measures jump-identity violations and per-solve condition numbers;
  4. runs the figure quantity Delta u = u(het) - u(homo-ref) with the
     half-jump and calibrated formulations and quantifies the bias;
  5. caches fields for the corrected-figure renderer.

Run from the moss2 repo root:  python benchmarks/assess_fig06_inclusion.py
"""

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MOSS_SCRIPTS = "/Users/meade/Desktop/moss/manuscript/scripts"

import mollified_bem as mb
from inclusion_mesh import make_inclusion_geometry
from local_box_mesh_eq import make_vertical_fault_eq

from mbem.backends.dense import AssembledDense
from mbem.kernels import basis as kb
from mbem.model import BCType, Patch, Region, RegionModel, generate_system

EPS = 4.0
MAT_HOST = mb.ElasticMaterial(mu=30.0, lam=30.0)
MAT_INC = mb.ElasticMaterial(mu=0.3, lam=0.3)     # mu_host / 100
OUT = pathlib.Path(__file__).parent / "fig06_assessment.npz"


def build():
    fault_trace = np.array([[0.0, -100.0], [0.0, 100.0]])
    meshes = make_inclusion_geometry(
        x_range=(-200.0, 200.0), y_range=(-200.0, 200.0), z_bottom=-200.0,
        inclusion_center_xy=(-100.0, 100.0), inclusion_radius=75.0,
        inclusion_depth=50.0,
        target_edge_inclusion=10.0, target_edge_top=15.0,
        target_edge_far=100.0, target_edge_side=100.0,
        fault_trace=fault_trace, fault_edge=7.5, host_top_max_edge=30.0)
    fault_mesh, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=200.0, depth_range=(-20.0, 0.0), target_edge=7.5)
    return meshes, fault_mesh, n_hat, s_hat


def build_model(meshes, fault_mesh, s_hat, mat_inc=None):
    mat_inc = MAT_INC if mat_inc is None else mat_inc
    host_top = Patch("host_top", meshes["host_top"], BCType.FREE_TRACTION)
    host_sides = Patch("host_sides", meshes["host_sides"],
                       BCType.FREE_TRACTION)
    host_base = Patch("host_base", meshes["host_base"],
                      BCType.PRESCRIBED_DISPLACEMENT)
    inc_top = Patch("inclusion_top", meshes["inclusion_top"],
                    BCType.FREE_TRACTION)
    int_side = Patch("interface_side", meshes["interface_side"],
                     BCType.INTERFACE)
    int_bot = Patch("interface_bot", meshes["interface_bot"],
                    BCType.INTERFACE)
    fault = Patch("fault", fault_mesh, BCType.FAULT,
                  value=np.broadcast_to(0.01 * np.asarray(s_hat),
                                        (fault_mesh.n_triangles, 3)))
    host = Region("host", MAT_HOST,
                  [host_top, host_sides, int_side, int_bot, host_base],
                  probe_point=np.array([120.0, -120.0, -100.0]),
                  faults=[fault])
    inclusion = Region("inclusion", mat_inc, [inc_top, int_side, int_bot],
                       probe_point=np.array([-100.0, 100.0, -25.0]))
    return RegionModel([host, inclusion])


def rowsum_violations(model):
    print("=== jump-identity violations (eps = %.1f) ===" % EPS, flush=True)
    worst = 0.0
    for region in model.regions:
        for q in region.patches:
            xq = q.mesh.centroids()
            total = np.zeros((xq.shape[0], 3, 3))
            for p in region.patches:
                sigma = model.orientation(region, p)
                H = kb.assemble_t_matrix(
                    xq, p.mesh, region.material,
                    kb.as_eps_array(EPS, p.n_triangles))
                total += sigma * H.reshape(xq.shape[0], 3, p.n_triangles,
                                           3).sum(axis=2)
            total += 0.5 * np.eye(3)[None, :, :]
            viol = np.sqrt((total ** 2).sum(axis=(1, 2)))
            worst = max(worst, float(viol.max()))
            print(f"  {region.name:9s} @ {q.name:14s}: "
                  f"mean {viol.mean():.3e}  max {viol.max():.3e}",
                  flush=True)
    return worst


def oracle_check(meshes, fault_mesh, n_hat, s_hat, sol_half):
    """mbem half-jump vs the original moss solver on this topology."""
    sys.path.insert(0, MOSS_SCRIPTS)
    from static_box_two_region import solve_two_region_box
    t0 = time.time()
    ref = solve_two_region_box(meshes, fault_mesh, n_hat, s_hat, 0.01,
                               MAT_HOST, MAT_INC, eps=EPS, verbose=False)
    print(f"  (moss legacy solver: {time.time()-t0:.0f} s)", flush=True)
    pairs = [("u:host_top", ref.u_host_top),
             ("u:host_sides", ref.u_host_sides),
             ("t:host_base", ref.t_host_base),
             ("u:inclusion_top", ref.u_inc_top),
             ("u:interface_side", ref.u_int_side),
             ("t:interface_side", ref.t_int_side),
             ("u:interface_bot", ref.u_int_bot),
             ("t:interface_bot", ref.t_int_bot)]
    worst = 0.0
    for name, r in pairs:
        err = np.max(np.abs(sol_half[name] - r)) / np.max(np.abs(r))
        worst = max(worst, err)
        print(f"  oracle parity {name:18s}: {err:.3e}", flush=True)
    return worst


def main():
    meshes, fault_mesh, n_hat, s_hat = build()
    for k, m in meshes.items():
        print(f"  {k:16s}: {m.n_triangles:5d} tri")
    print(f"  fault           : {fault_mesh.n_triangles:5d} tri")

    model = build_model(meshes, fault_mesh, s_hat)
    print("sigma table:", model.sigma_table())
    system = generate_system(model)
    print(f"unknowns: {system.layout.n_unknowns}")

    worst_viol = rowsum_violations(model)

    results = {}
    conds = {}
    basis_cache: dict = {}
    for jump in ("half", "calibrated"):
        t0 = time.time()
        asm = AssembledDense(system, EPS, "basis", jump=jump,
                             _basis_cache=basis_cache)
        sol_het = asm.solve()
        cond_het = asm.cond_estimate
        asm_h = asm.rebuild_for_materials({"inclusion": MAT_HOST})
        sol_hom = asm_h.solve()
        cond_hom = asm_h.cond_estimate
        print(f"{jump}: het cond {cond_het:.3e}, homo cond {cond_hom:.3e} "
              f"({time.time()-t0:.0f} s)", flush=True)
        results[jump] = (sol_het, sol_hom)
        conds[jump] = (cond_het, cond_hom)

    print("=== oracle parity (half-jump vs moss legacy solver) ===")
    worst_parity = oracle_check(meshes, fault_mesh, n_hat, s_hat,
                                results["half"][0])

    # --- the figure quantity: Delta u on the two top surfaces ---
    print("=== half vs calibrated, figure quantities ===")
    summary = {}
    for patch in ("host_top", "inclusion_top"):
        u_h = {j: results[j][0][f"u:{patch}"] for j in results}
        d_u = {j: results[j][0][f"u:{patch}"] - results[j][1][f"u:{patch}"]
               for j in results}
        scale_u = np.max(np.abs(u_h["calibrated"]))
        scale_d = np.max(np.abs(d_u["calibrated"]))
        err_u = np.max(np.abs(u_h["half"] - u_h["calibrated"])) / scale_u
        err_d = np.max(np.abs(d_u["half"] - d_u["calibrated"])) / scale_d
        print(f"  {patch:14s}: raw-u rel diff {err_u:.3e}   "
              f"Delta-u rel diff {err_d:.3e}")
        summary[patch] = (err_u, err_d)

    np.savez(
        OUT,
        host_top_vertices=meshes["host_top"].vertices,
        host_top_triangles=meshes["host_top"].triangles,
        inclusion_top_vertices=meshes["inclusion_top"].vertices,
        inclusion_top_triangles=meshes["inclusion_top"].triangles,
        worst_violation=worst_viol,
        worst_parity=worst_parity,
        cond_half_het=conds["half"][0], cond_half_hom=conds["half"][1],
        cond_cal_het=conds["calibrated"][0],
        cond_cal_hom=conds["calibrated"][1],
        **{f"u_{p}_{j}_{c}": results[j][0 if c == "het" else 1][f"u:{p}"]
           for p in ("host_top", "inclusion_top")
           for j in ("half", "calibrated")
           for c in ("het", "hom")},
    )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

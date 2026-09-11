"""Block Gauss-Seidel preconditioner over coupled interface super-blocks.

The diagonal super-block of an interface patch is the LOCAL 2x2
transmission system in its (u, t) unknowns,

    [ 1/2 I + sA*H^A_pp    -sA*G^A_pp ]
    [ 1/2 I + sB*H^B_pp    -sB*G^B_pp ]

which is well-posed even where a first-kind G block alone is nearly
singular (it discretizes the locally well-posed two-sided transmission
problem) and contrast-robust. Non-interface slots form their own
super-blocks (second-kind 1/2 I + sigma H for u-slots; -sigma G for a
prescribed-displacement t-slot). One forward GS sweep in slot order —
the region graph of layered models is a path, so the sweep approximates
chain elimination.

Diagonal solves form a two-rung ladder:
  * size <= max_dense : exact dense assembly (fast numba kernels) + LU;
  * larger            : HODLR solver at loose tolerance over the
                        element-interleaved local system (u,t per
                        element), scalable in memory and build time.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .. import defaults
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from ..model.core import BCType
from .hodlr import HodlrSolver


class _SBEvaluator:
    """Dense element-subset evaluator of one super-block's local matrix.

    Local layout is ELEMENT-INTERLEAVED: d DOFs per element, where the
    first 3 are the patch's u-row/col and (for interfaces) the next 3
    are the t-row/col. ``concat_perm`` maps the super-block's
    slot-concatenated layout (used by the GS sweep) to this interleaved
    layout.
    """

    def __init__(self, assembled, slots, terms):
        patch = slots[0].patch
        self.centroids = np.ascontiguousarray(patch.mesh.centroids())
        tri_verts, normals = kb._source_arrays(patch.mesh)
        self.tri_verts = tri_verts
        self.normals = normals
        self.eps = assembled.eps_for(patch)
        self.d = 3 * len(slots)
        self.n_elems = patch.n_triangles

        kind_off = {s.kind: 3 * idx for idx, s in enumerate(slots)}
        self.recipes = []
        for t in terms:
            mat = assembled.materials[t.region.name]
            self.recipes.append((
                t.kernel,
                kb.t_coeffs(mat.mu, mat.lam) if t.kernel == "H"
                else kb.u_coeffs(mat.mu, mat.lam),
                t.scale,
                kind_off[t.row.kind],
                kind_off[t.col.kind],
                t.diag_half,
            ))

        # concat -> interleaved permutation (length d * n_elems):
        # interleaved[d*e + off + a] = concat[3*n*idx(off) + 3*e + a]
        n = self.n_elems
        perm = np.empty(self.d * n, dtype=int)
        for idx in range(len(slots)):
            for a in range(3):
                perm[np.arange(n) * self.d + 3 * idx + a] = \
                    3 * n * idx + 3 * np.arange(n) + a
        self.concat_perm = perm

    def eval_block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        nr, nc = len(rows), len(cols)
        out = np.zeros((self.d * nr, self.d * nc))
        xf = self.centroids[rows]
        tv = self.tri_verts[cols]
        nv = self.normals[cols]
        ee = self.eps[cols]
        same = rows[:, None] == cols[None, :]
        for kernel, coeffs, scale, r_off, c_off, diag_half in self.recipes:
            if kernel == "H":
                blk = tk.t_matrix_direct(xf, tv, nv, ee, *coeffs)
            else:
                blk = tk.u_matrix_direct(xf, tv, ee, *coeffs)
            for a in range(3):
                for b in range(3):
                    out[r_off + a::self.d, c_off + b::self.d] += \
                        scale * blk[a::3, b::3]
            if diag_half:
                ii, jj = np.where(same)
                for a in range(3):
                    out[self.d * ii + r_off + a,
                        self.d * jj + c_off + a] += 0.5
        return out


class _SuperBlock:
    def __init__(self, slots):
        self.slots = list(slots)
        self.global_idx = np.concatenate([
            np.arange(s.offset, s.stop) for s in self.slots])
        self.size = int(self.global_idx.size)
        self.local_offset = {}
        off = 0
        for s in self.slots:
            self.local_offset[s.name] = off
            off += s.size
        self.solve_fn = None
        self.lower_terms = []   # terms with col in an earlier super-block


def _term_coeffs(term, materials):
    mat = materials[term.region.name]
    if term.kernel == "H":
        return kb.t_coeffs(mat.mu, mat.lam)
    return kb.u_coeffs(mat.mu, mat.lam)


class BlockGaussSeidel:
    """Right preconditioner z = M(r) for an AssembledH system."""

    def __init__(self, assembled,
                 max_dense: int = defaults.MAX_DENSE_PRECOND_DOF,
                 hodlr_tol: float = defaults.HODLR_PRECOND_TOL,
                 verbose: bool = False):
        self.asm = assembled
        layout = assembled.layout

        # ---- group slots into super-blocks ----
        groups = []
        seen = set()
        for slot in layout.slots:
            if id(slot.patch) in seen:
                continue
            if slot.patch.bc is BCType.INTERFACE:
                groups.append([layout.slot(slot.patch, "u"),
                               layout.slot(slot.patch, "t")])
                seen.add(id(slot.patch))
            else:
                groups.append([slot])
        self.sbs = [_SuperBlock(g) for g in groups]

        sb_of_slot = {}
        for k, sb in enumerate(self.sbs):
            for s in sb.slots:
                sb_of_slot[s.name] = k

        sb_terms = [[] for _ in self.sbs]
        for term in assembled.system.terms:
            kr = sb_of_slot[term.row.name]
            kc = sb_of_slot[term.col.name]
            if kr == kc:
                sb_terms[kr].append(term)
            elif kc < kr:
                self.sbs[kr].lower_terms.append(term)

        # ---- diagonal solves: dense LU rung or HODLR rung ----
        for k, sb in enumerate(self.sbs):
            ev = _SBEvaluator(assembled, sb.slots, sb_terms[k])
            if sb.size <= max_dense:
                all_elems = np.arange(ev.n_elems)
                D_inter = ev.eval_block(all_elems, all_elems)
                lu = lu_factor(D_inter)
                perm = ev.concat_perm

                def solve_fn(r, lu=lu, perm=perm):
                    return _permuted_lu_solve(lu, r, perm)

                sb.solve_fn = solve_fn
                if verbose:
                    print(f"  SB {[s.name for s in sb.slots]}: dense LU "
                          f"({sb.size} DOFs)")
            else:
                hod = HodlrSolver(ev.eval_block, ev.centroids, ev.d,
                                  tol=hodlr_tol,
                                  leaf_elems=defaults.HODLR_LEAF_ELEMS)
                perm = ev.concat_perm

                def solve_fn(r, hod=hod, perm=perm):
                    return _permuted_solver(hod.solve, r, perm)

                sb.solve_fn = solve_fn
                if verbose:
                    print(f"  SB {[s.name for s in sb.slots]}: HODLR "
                          f"({sb.size} DOFs, tol {hodlr_tol:g}); "
                          f"{hod.rank_summary()}")

    def __call__(self, r: np.ndarray) -> np.ndarray:
        z = np.zeros_like(r)
        for sb in self.sbs:
            rk = r[sb.global_idx].astype(z.dtype, copy=True)
            for term in sb.lower_terms:
                pair = self.asm.pair_for(term)
                coeffs = _term_coeffs(term, self.asm.materials)
                xseg = z[term.col.offset:term.col.stop]
                contrib = term.scale * pair.matvec(coeffs, xseg)
                r0 = sb.local_offset[term.row.name]
                rk[r0:r0 + term.row.size] -= contrib
            z[sb.global_idx] = sb.solve_fn(rk)
        return z


def _permuted_lu_solve(lu, r_concat, perm):
    """Solve the interleaved-layout LU for a concat-layout RHS."""
    r_inter = np.empty_like(r_concat)
    r_inter[np.arange(perm.size)] = r_concat[perm]
    x_inter = lu_solve(lu, r_inter)
    x_concat = np.empty_like(x_inter)
    x_concat[perm] = x_inter
    return x_concat


def _permuted_solver(solve, r_concat, perm):
    r_inter = r_concat[perm].copy()
    x_inter = solve(r_inter)
    x_concat = np.empty_like(x_inter)
    x_concat[perm] = x_inter
    return x_concat

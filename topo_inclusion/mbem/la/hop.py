"""Compressed pair operators with material-basis storage.

A ``PairCompressed`` represents ALL basis matrices of one
(field mesh, source mesh, kernel) pair in block-compressed form:
admissible blocks as per-basis low-rank factors (see :mod:`.aca`),
near-field leaf blocks as exact dense basis stacks. A material enters
only through its coefficient vector: the per-material view combines the
factors (with one QR+SVD re-truncation per low-rank block) and is
cached, so the SAME object serves every region material and every
Laplace sample.
"""

from __future__ import annotations

import numpy as np

from .. import defaults
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from .aca import BasisLR, BlockEvalCache, compress_block, recompress
from .cluster import build_cluster_tree, build_partition


def _dof_idx(elems: np.ndarray) -> np.ndarray:
    return (3 * elems[:, None] + np.arange(3)[None, :]).ravel()


class _BasisEval:
    """Subset evaluators for one (field, source, kernel) pair."""

    def __init__(self, field_mesh, source_mesh, kernel: str, eps_arr):
        self.x_field = np.ascontiguousarray(field_mesh.centroids())
        tri_verts, normals = kb._source_arrays(source_mesh)
        self.tri_verts = tri_verts
        self.normals = normals
        self.eps = eps_arr
        self.kernel = kernel
        self.n_basis = 6 if kernel == "H" else 3

    def stack(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(B, 3*len(rows), 3*len(cols)) exact basis sub-stack."""
        xf = self.x_field[rows]
        tv = self.tri_verts[cols]
        ee = self.eps[cols]
        if self.kernel == "H":
            return tk.t_basis_matrices(xf, tv, self.normals[cols], ee)
        return tk.u_basis_matrices(xf, tv, ee)


class PairCompressed:
    def __init__(self, field_mesh, source_mesh, kernel: str, eps_arr,
                 tol: float = defaults.BLOCK_COMPRESSION_TOL,
                 min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                 eta: float = defaults.ADMISSIBILITY_ETA,
                 max_admissible: int = 4096,
                 tree_cache: dict | None = None):
        self.eval = _BasisEval(field_mesh, source_mesh, kernel, eps_arr)
        self.n_field = field_mesh.n_triangles
        self.n_source = source_mesh.n_triangles
        self.n_basis = self.eval.n_basis
        self.shape = (3 * self.n_field, 3 * self.n_source)
        self.tol = tol

        tree_cache = {} if tree_cache is None else tree_cache

        def _tree(mesh):
            key = id(mesh)
            t = tree_cache.get(key)
            if t is None:
                t = build_cluster_tree(
                    np.ascontiguousarray(mesh.centroids()), min_leaf)
                tree_cache[key] = t
            return t

        part = build_partition(_tree(field_mesh), _tree(source_mesh),
                               eta=eta, max_admissible=max_admissible)

        rng = np.random.default_rng(12345)
        self.blocks = []         # (row_dofs, col_dofs, BasisLR | ndarray)
        self.n_lowrank = 0
        self.n_dense = 0
        self.n_fallback = 0

        for rows, cols in part.admissible:
            cache = BlockEvalCache(self.eval.stack, rows, cols)
            lr = compress_block(cache, self.n_basis, tol=tol, rng=rng)
            self.n_fallback += 1 if lr.fallback else 0
            self.blocks.append((_dof_idx(rows), _dof_idx(cols), lr))
            self.n_lowrank += 1

        for rows, cols in part.dense:
            stack = self.eval.stack(rows, cols)
            self.blocks.append((_dof_idx(rows), _dof_idx(cols), stack))
            self.n_dense += 1

        self._views: dict[bytes, list] = {}

    # -- material views -----------------------------------------------

    def _view(self, coeffs: np.ndarray) -> list:
        c = np.asarray(coeffs)
        key = c.tobytes()
        view = self._views.get(key)
        if view is None:
            view = []
            for rdofs, cdofs, payload in self.blocks:
                if isinstance(payload, BasisLR):
                    U_cat = np.hstack([c[b] * payload.U[b]
                                       for b in range(self.n_basis)])
                    V_cat = np.hstack(payload.V)
                    if np.iscomplexobj(U_cat):
                        # complex coeffs: keep concatenated factors
                        # (recompress is real-QR based; rank cost is
                        # acceptable for the Laplace sweep)
                        view.append((rdofs, cdofs, U_cat, V_cat))
                    else:
                        U, V = recompress(U_cat, V_cat, self.tol)
                        view.append((rdofs, cdofs, U, V))
                else:
                    M_eff = np.tensordot(c, payload, axes=1)
                    view.append((rdofs, cdofs, M_eff, None))
            self._views[key] = view
        return view

    # -- operations -----------------------------------------------------

    def matvec(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        dtype = np.result_type(np.asarray(coeffs).dtype, x.dtype, np.float64)
        y = np.zeros(self.shape[0], dtype=dtype)
        for rdofs, cdofs, A, V in self._view(coeffs):
            if V is None:
                y[rdofs] += A @ x[cdofs]
            else:
                y[rdofs] += A @ (V.T @ x[cdofs])
        return y

    def to_dense(self, coeffs: np.ndarray) -> np.ndarray:
        M = np.zeros(self.shape,
                     dtype=np.result_type(np.asarray(coeffs).dtype,
                                          np.float64))
        for rdofs, cdofs, A, V in self._view(coeffs):
            if V is None:
                M[np.ix_(rdofs, cdofs)] += A
            else:
                M[np.ix_(rdofs, cdofs)] += A @ V.T
        return M

    # -- stats ------------------------------------------------------

    def nbytes(self) -> int:
        total = 0
        for _, _, payload in self.blocks:
            total += payload.nbytes() if isinstance(payload, BasisLR) \
                else payload.nbytes
        return total

    def dense_equivalent_bytes(self) -> int:
        """Bytes of ONE dense material matrix (what a solver would hold)."""
        return self.shape[0] * self.shape[1] * 8

    def view_nbytes(self, coeffs: np.ndarray) -> int:
        """Bytes of the per-material working set (recombined view)."""
        total = 0
        for _, _, A, V in self._view(coeffs):
            total += A.nbytes + (V.nbytes if V is not None else 0)
        return total

    def summary(self) -> str:
        ranks = [r for _, _, p in self.blocks if isinstance(p, BasisLR)
                 for r in p.ranks]
        return (f"PairCompressed {self.shape} x{self.n_basis} bases: "
                f"{self.n_lowrank} low-rank blocks "
                f"({self.n_fallback} w/ fallback, "
                f"avg basis rank {np.mean(ranks) if ranks else 0:.1f}), "
                f"{self.n_dense} dense leaf, "
                f"{self.nbytes()/1e6:.1f} MB vs {self.n_basis} x "
                f"{self.dense_equivalent_bytes()/1e6:.1f} MB dense-basis")

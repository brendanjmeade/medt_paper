"""HODLR factorization/solver via recursive Sherman-Morrison-Woodbury.

Classic O(N log^2 N) HODLR direct solver (Ambikasaran & Darve style):
at every node the matrix splits as

    A = [[A11, U12 V12^T], [U21 V21^T, A22]]
      = D + U W           D = blkdiag(A11, A22)

with the off-diagonal blocks compressed by ACA (d-DOF element-block
pivots, sampled-residual stopping). The factorization recursively
prepares, per node,

    Z = D^{-1} U          (child solves on k RHS columns)
    C = I + W Z           (small (k12+k21) capacitance, LU-factored)

and a solve applies  x = y - Z C^{-1} (W y)  with  y = D^{-1} b.

At loose tolerance (~1e-2) this is the scalable preconditioner rung for
super-blocks beyond the dense-LU cap; at tight tolerance (~1e-7) it is
a moderate-size direct solver. Build cost is dominated by the ACA
kernel evaluations and the recursive child solves.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .aca import BlockEvalCache, aca_single, _svd_keep
from .cluster import build_cluster_tree


class _Leaf:
    __slots__ = ("elems", "lu")

    def __init__(self, elems, lu):
        self.elems = elems
        self.lu = lu

    def solve(self, b):
        return lu_solve(self.lu, b)


class _Node:
    __slots__ = ("n1", "left", "right", "Z1", "Z2", "V12", "V21", "C_lu")

    def __init__(self, left, right, n1, Z1, Z2, V12, V21, C_lu):
        self.left = left
        self.right = right
        self.n1 = n1
        self.Z1 = Z1        # (n1, k12) = A11^{-1} U12
        self.Z2 = Z2        # (n2, k21) = A22^{-1} U21
        self.V12 = V12      # (n2, k12)
        self.V21 = V21      # (n1, k21)
        self.C_lu = C_lu

    def solve(self, b):
        y1 = self.left.solve(b[:self.n1])
        y2 = self.right.solve(b[self.n1:])
        t = np.concatenate([self.V12.T @ y2, self.V21.T @ y1], axis=0)
        s = lu_solve(self.C_lu, t)
        k12 = self.V12.shape[1]
        y1 -= self.Z1 @ s[:k12]
        y2 -= self.Z2 @ s[k12:]
        return np.concatenate([y1, y2], axis=0)


class HodlrSolver:
    """HODLR approximate inverse of a dense-evaluable element-block matrix.

    ``eval_block(rows, cols)`` -> (d*len(rows), d*len(cols)) exact
    sub-matrix for ELEMENT index arrays (original element ids).
    """

    def __init__(self, eval_block, centroids: np.ndarray, d: int,
                 tol: float = 1e-2, leaf_elems: int = 96,
                 rng=None):
        self.eval_block = eval_block
        self.d = d
        self.tol = tol
        self.rng = np.random.default_rng(7) if rng is None else rng

        tree = build_cluster_tree(centroids, min_leaf=leaf_elems)
        self.perm_elems = self._collect_order(tree)
        # DOF permutation: original -> position in HODLR ordering
        n = centroids.shape[0]
        self.n_dofs = d * n
        dof_perm = (d * self.perm_elems[:, None]
                    + np.arange(d)[None, :]).ravel()
        self.dof_perm = dof_perm                       # hodlr_pos -> orig dof
        self.inv_perm = np.empty_like(dof_perm)
        self.inv_perm[dof_perm] = np.arange(self.n_dofs)
        self.rank_profile: list[tuple[int, int]] = []  # (block size, rank)

        self.root = self._build(tree)

    @staticmethod
    def _collect_order(tree) -> np.ndarray:
        out = []

        def rec(node):
            if node.is_leaf:
                out.append(node.indices)
            else:
                rec(node.left)
                rec(node.right)

        rec(tree)
        return np.concatenate(out)

    # -- construction -------------------------------------------------

    def _lowrank(self, rows: np.ndarray, cols: np.ndarray):
        """Compress eval_block[rows, cols] (ordered subsets) by ACA."""
        n_rows, n_cols = len(rows), len(cols)

        def stack_fn(r, c):
            return self.eval_block(r, c)[None, :, :]

        cache = BlockEvalCache(stack_fn, rows, cols, d=self.d)
        srows, scols, stop_exact = cache.sample(self.rng, 10)
        uv = aca_single(cache, 0, n_rows, n_cols, self.tol,
                        srows, scols, stop_exact[0])
        if uv is None:
            M = self.eval_block(rows, cols)
            u, s, vt = np.linalg.svd(M, full_matrices=False)
            keep = _svd_keep(s, self.tol)
            uv = (u[:, :keep] * s[:keep], vt[:keep, :].T)
        return uv

    def _build(self, node):
        elems = node.indices
        if node.is_leaf:
            M = self.eval_block(elems, elems)
            return _Leaf(elems, lu_factor(M))

        e1 = self._collect_order(node.left)
        e2 = self._collect_order(node.right)
        U12, V12 = self._lowrank(e1, e2)
        U21, V21 = self._lowrank(e2, e1)
        self.rank_profile.append((self.d * len(e1), U12.shape[1]))
        self.rank_profile.append((self.d * len(e2), U21.shape[1]))

        left = self._build(node.left)
        right = self._build(node.right)

        Z1 = _solve_node(left, U12)
        Z2 = _solve_node(right, U21)
        k12, k21 = U12.shape[1], U21.shape[1]
        C = np.eye(k12 + k21)
        C[:k12, k12:] += V12.T @ Z2
        C[k12:, :k12] += V21.T @ Z1
        return _Node(left, right, self.d * len(e1), Z1, Z2, V12, V21,
                     lu_factor(C))

    # -- application ----------------------------------------------------

    def solve(self, b: np.ndarray) -> np.ndarray:
        one_d = b.ndim == 1
        bp = b[self.dof_perm]              # bp[pos] = b[orig dof at pos]
        if one_d:
            bp = bp[:, None]
        xp = self.root.solve(bp)
        out = np.empty_like(xp)
        out[self.dof_perm, :] = xp         # out[orig dof] = xp[pos]
        return out[:, 0] if one_d else out

    def max_rank(self) -> int:
        return max((r for _, r in self.rank_profile), default=0)

    def rank_summary(self) -> str:
        if not self.rank_profile:
            return "HODLR: single leaf (dense)"
        tops = sorted(self.rank_profile, reverse=True)[:4]
        return ("HODLR ranks (size, rank): "
                + ", ".join(f"({s}, {r})" for s, r in tops))


def _solve_node(node, B: np.ndarray) -> np.ndarray:
    return node.solve(B)

"""Block ACA compression of material-basis matrix stacks.

Each admissible block stores the basis matrices SEPARATELY in low rank:

    A_b ~ U_b @ V_b.T          b = 1..B   (B = 3 for U-kernel, 6 for T)

Per-basis storage beats compressing the stacked matrix because the basis
matrices do not share row/column spaces (a stacked factorization pays up
to a Bx rank penalty — measured, not hypothetical). The per-material
combined block sum_b c_b U_b V_b^T is re-truncated once per material by
QR+SVD and cached by the operator layer, so matvec rank stays at the
combination's own epsilon-rank.

Pivoting is 3x3 ELEMENT-block (scalar ACA on interleaved xyz DOFs is
what plateaued in the legacy code, hmatrix.py:493). Kernel evaluations
go through a shared row/column cache: one evaluation yields the rows of
ALL bases, so compressing 6 bases costs the same kernel work as one.
Convergence uses the legacy Frobenius-increment heuristic plus a
randomized verification per basis; failures fall back to exact dense
evaluation + per-basis SVD (correctness guaranteed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import defaults


@dataclass
class BasisLR:
    """Per-basis low-rank factors: A_b ~ U[b] @ V[b].T (ragged ranks)."""
    U: list          # B arrays (3nr, k_b)
    V: list          # B arrays (3nc, k_b)
    fallback: int = 0   # how many bases came from the dense+SVD fallback

    @property
    def ranks(self) -> list:
        return [u.shape[1] for u in self.U]

    def nbytes(self) -> int:
        return sum(u.nbytes + v.nbytes for u, v in zip(self.U, self.V))


class BlockEvalCache:
    """Caches exact basis rows/cols/samples for one admissible block.

    ``stack_fn(rows, cols)`` -> (B, d*len(rows), d*len(cols)) where d is
    the DOFs per element (3 for kernel bases; 6 for coupled (u,t)
    transmission blocks used by the HODLR rung).
    """

    def __init__(self, stack_fn, rows: np.ndarray, cols: np.ndarray,
                 d: int = 3):
        self.stack_fn = stack_fn
        self.rows = rows
        self.cols = cols
        self.d = d
        self._row: dict[int, np.ndarray] = {}
        self._col: dict[int, np.ndarray] = {}
        self.n_row_evals = 0
        self.n_col_evals = 0

    def row(self, b: int, i: int) -> np.ndarray:
        """(d, d*nc) exact rows of element i for basis b."""
        blk = self._row.get(i)
        if blk is None:
            blk = self.stack_fn(self.rows[i:i + 1], self.cols)
            self._row[i] = blk
            self.n_row_evals += 1
        return blk[b]

    def col(self, b: int, j: int) -> np.ndarray:
        """(d*nr, d) exact cols of element j for basis b."""
        blk = self._col.get(j)
        if blk is None:
            blk = self.stack_fn(self.rows, self.cols[j:j + 1])
            self._col[j] = blk
            self.n_col_evals += 1
        return blk[b]

    def dense(self) -> np.ndarray:
        return self.stack_fn(self.rows, self.cols)

    def sample(self, rng, m: int):
        r = rng.choice(len(self.rows), size=min(m, len(self.rows)),
                       replace=False)
        c = rng.choice(len(self.cols), size=min(m, len(self.cols)),
                       replace=False)
        return r, c, self.stack_fn(self.rows[r], self.cols[c])


def _pinv3(P: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    u, s, vt = np.linalg.svd(P)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros_like(P.T)
    s_inv = np.where(s > rcond * s[0], 1.0 / np.where(s == 0, 1.0, s), 0.0)
    return (vt.T * s_inv) @ u.T


def aca_single(cache: BlockEvalCache, b: int, n_rows: int, n_cols: int,
               tol: float, srows: np.ndarray, scols: np.ndarray,
               sample_exact: np.ndarray):
    """Element-block ACA of basis ``b``. Returns (U, V) or None.

    Stopping is based on the TRUE residual tracked on a persistent
    random sample (``sample_exact``: exact entries at srows x scols) —
    the Frobenius-increment heuristic alone plateaus on this kernel
    (the failure mode recorded at hmatrix.py:493). The increment check
    remains only as a cheap accelerator for an early exit test.
    """
    d = cache.d
    nc3 = d * n_cols
    rsel = (d * srows[:, None] + np.arange(d)[None, :]).ravel()
    csel = (d * scols[:, None] + np.arange(d)[None, :]).ravel()
    R_sample = sample_exact.copy()
    sample_norm = np.linalg.norm(sample_exact)
    stop_tol = 0.5 * tol

    U_parts: list[np.ndarray] = []
    V_parts: list[np.ndarray] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    i_pivot = 0
    converged = False

    max_steps = max(8, int(min(n_rows, n_cols)
                           * defaults.ACA_MAX_RANK_FRACTION))
    for _ in range(min(max_steps, n_rows, n_cols)):
        used_rows.add(i_pivot)
        R_row = cache.row(b, i_pivot).copy()          # (d, d*nc)
        r0 = d * i_pivot
        for Up, Vp in zip(U_parts, V_parts):
            R_row -= Up[r0:r0 + d, :] @ Vp.T

        scores = np.sqrt(np.sum(R_row.reshape(d, n_cols, d) ** 2,
                                axis=(0, 2)))
        for j in used_cols:
            scores[j] = -1.0
        j_pivot = int(np.argmax(scores))
        if scores[j_pivot] <= 1e-300:
            converged = True
            break
        used_cols.add(j_pivot)

        C_col = cache.col(b, j_pivot).copy()          # (d*nr, d)
        c0 = d * j_pivot
        for Up, Vp in zip(U_parts, V_parts):
            C_col -= Up @ Vp[c0:c0 + d, :].T

        P = R_row[:, c0:c0 + d]                       # (d, d)
        V_new = (_pinv3(P) @ R_row).T                 # (d*nc, d)
        U_new = C_col                                 # (d*nr, d)

        U_parts.append(U_new)
        V_parts.append(V_new)

        # True-residual tracking on the persistent sample
        R_sample -= U_new[rsel, :] @ V_new[csel, :].T
        if sample_norm == 0.0 or \
                np.linalg.norm(R_sample) < stop_tol * sample_norm:
            converged = True
            break

        rscores = np.sqrt(np.sum(U_new.reshape(n_rows, d, d) ** 2,
                                 axis=(1, 2)))
        for i in used_rows:
            rscores[i] = -1.0
        i_pivot = int(np.argmax(rscores))
        if rscores[i_pivot] <= 0.0:
            converged = True
            break

    if not converged:
        return None      # full-rank sweep without convergence

    if not U_parts:
        return (np.zeros((d * n_rows, 1)), np.zeros((nc3, 1)))

    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    return recompress(U, V, 0.5 * tol)


def recompress(U: np.ndarray, V: np.ndarray, tol: float):
    """Truncate U @ V.T via thin QR of both factors and an SVD."""
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V)
    u, s, vt = np.linalg.svd(Ru @ Rv.T, full_matrices=False)
    keep = _svd_keep(s, tol)
    return Qu @ (u[:, :keep] * s[:keep]), Qv @ vt[:keep, :].T


def _svd_keep(s: np.ndarray, tol: float) -> int:
    """Smallest k with ||s[k:]|| <= tol * s[0]."""
    if s.size == 0 or s[0] == 0.0:
        return 1
    tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
    keep = int(np.searchsorted(-tail, -tol * s[0]))
    return max(1, min(keep, s.size))


def compress_block(cache: BlockEvalCache, n_basis: int,
                   tol: float = defaults.BLOCK_COMPRESSION_TOL,
                   verify_factor: float = 3.0,
                   rng=None) -> BasisLR:
    """Compress all basis matrices of one admissible block.

    Per-basis ACA with a shared kernel-evaluation cache. The stopping
    sample and the verification sample are INDEPENDENT random draws;
    any basis whose verification error exceeds verify_factor * tol
    falls back to exact dense evaluation + SVD.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_rows = len(cache.rows)
    n_cols = len(cache.cols)

    Us: list = [None] * n_basis
    Vs: list = [None] * n_basis
    n_fallback = 0

    d = cache.d
    srows, scols, stop_exact = cache.sample(rng, 10)
    vrows, vcols, verify_exact = cache.sample(rng, 10)
    vrsel = (d * vrows[:, None] + np.arange(d)[None, :]).ravel()
    vcsel = (d * vcols[:, None] + np.arange(d)[None, :]).ravel()

    dense = None
    for b in range(n_basis):
        uv = aca_single(cache, b, n_rows, n_cols, tol,
                        srows, scols, stop_exact[b])
        if uv is not None:
            U, V = uv
            approx = U[vrsel] @ V[vcsel].T
            denom = np.linalg.norm(verify_exact[b])
            err = np.linalg.norm(approx - verify_exact[b])
            if denom > 0:
                err /= denom
            if err > verify_factor * tol:
                uv = None
        if uv is None:
            if dense is None:
                dense = cache.dense()
            u, s, vt = np.linalg.svd(dense[b], full_matrices=False)
            keep = _svd_keep(s, tol)
            uv = (u[:, :keep] * s[:keep], vt[:keep, :].T)
            n_fallback += 1
        Us[b], Vs[b] = uv

    return BasisLR(U=Us, V=Vs, fallback=n_fallback)

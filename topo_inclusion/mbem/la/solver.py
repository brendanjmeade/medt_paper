"""Flexible right-preconditioned GMRES with honest reporting.

Replaces the bare ``scipy.sparse.linalg.gmres(rtol=1e-10, ...)`` calls of
the legacy stack. Differences that matter:

* RIGHT preconditioning — the iteration's residual is the residual of
  the actual system, and convergence is confirmed against the TRUE
  relative residual ||b - A x|| / ||b|| before reporting success.
* Flexible (FGMRES) — the preconditioner may change between iterations
  (inner iterative solves, rebuilt ladder rungs).
* Stagnation detection — if the residual fails to improve by
  ``stagnation_factor`` over ``stagnation_window`` iterations, the solve
  stops and says so instead of burning maxiter.
* Every solve returns a ``SolveReport``; nothing is silently swallowed.

Supports real and complex systems (dtype follows the inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import defaults


@dataclass
class SolveReport:
    converged: bool
    iterations: int
    true_relres: float
    arnoldi_relres: float
    stagnated: bool = False
    restarts: int = 0
    residual_history: list = field(default_factory=list)

    def __str__(self):
        status = "converged" if self.converged else (
            "STAGNATED" if self.stagnated else "NOT CONVERGED")
        return (f"FGMRES {status}: {self.iterations} iters, "
                f"true relres {self.true_relres:.3e}")


def fgmres(A, b, M=None, x0=None,
           rtol: float = defaults.GMRES_RTOL,
           restart: int = defaults.GMRES_RESTART,
           maxiter: int = defaults.GMRES_MAXITER,
           stagnation_window: int = 100,
           stagnation_factor: float = 10.0,
           callback=None):
    """Solve A x = b. Returns (x, SolveReport).

    ``A`` and ``M`` are callables v -> A@v / M@v (or objects with a
    ``matvec``/``__matmul__``); ``M`` approximates A^{-1} (right
    preconditioner).
    """
    A_mv = _as_matvec(A)
    M_mv = _as_matvec(M) if M is not None else (lambda v: v)

    b = np.asarray(b)
    n = b.shape[0]
    dtype = np.result_type(b.dtype, np.float64)
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return np.zeros(n, dtype=dtype), SolveReport(True, 0, 0.0, 0.0)

    x = np.zeros(n, dtype=dtype) if x0 is None else np.array(x0, dtype=dtype)

    history: list[float] = []
    total_iters = 0
    restarts = 0
    best_res = np.inf
    best_res_at = 0
    stagnated = False

    while total_iters < maxiter and not stagnated:
        r = b - A_mv(x)
        beta = np.linalg.norm(r)
        arnoldi_res = beta / b_norm
        if arnoldi_res < rtol:
            break

        m = min(restart, maxiter - total_iters)
        V = np.empty((m + 1, n), dtype=dtype)
        Z = np.empty((m, n), dtype=dtype)
        H = np.zeros((m + 1, m), dtype=dtype)
        cs = np.zeros(m, dtype=dtype)
        sn = np.zeros(m, dtype=dtype)
        g = np.zeros(m + 1, dtype=dtype)
        V[0] = r / beta
        g[0] = beta

        j_done = 0
        for j in range(m):
            Z[j] = M_mv(V[j])
            w = A_mv(Z[j])
            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = np.vdot(V[i], w)
                w -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j].real > 1e-300:
                V[j + 1] = w / H[j + 1, j]

            # Apply accumulated Givens rotations, then form a new one
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -np.conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            denom = np.sqrt(np.abs(H[j, j]) ** 2 + np.abs(H[j + 1, j]) ** 2)
            if denom == 0.0:
                j_done = j + 1
                break
            cs[j] = np.abs(H[j, j]) / denom if np.abs(H[j, j]) > 0 else 0.0
            if np.abs(H[j, j]) > 0:
                phase = H[j, j] / np.abs(H[j, j])
                sn[j] = phase * np.conj(H[j + 1, j]) / denom
            else:
                sn[j] = 1.0
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -np.conj(sn[j]) * g[j]
            g[j] = cs[j] * g[j]

            total_iters += 1
            j_done = j + 1
            arnoldi_res = float(np.abs(g[j + 1])) / b_norm
            history.append(arnoldi_res)
            if callback is not None:
                callback(total_iters, arnoldi_res)

            # Stagnation bookkeeping
            if arnoldi_res < best_res / stagnation_factor:
                best_res = arnoldi_res
                best_res_at = total_iters
            elif total_iters - best_res_at >= stagnation_window:
                stagnated = True
                break

            if arnoldi_res < rtol or total_iters >= maxiter:
                break

        # Form the restart/final iterate
        if j_done > 0:
            y = _solve_upper(H[:j_done, :j_done], g[:j_done])
            x = x + Z[:j_done].T @ y
        restarts += 1

        if stagnated:
            break

    true_relres = float(np.linalg.norm(b - A_mv(x)) / b_norm)
    arnoldi_final = history[-1] if history else true_relres
    converged = true_relres < rtol
    return x, SolveReport(
        converged=converged,
        iterations=total_iters,
        true_relres=true_relres,
        arnoldi_relres=float(arnoldi_final),
        stagnated=stagnated and not converged,
        restarts=restarts - 1,
        residual_history=history,
    )


def _solve_upper(R, g):
    """Back-substitution for the small upper-triangular least-squares system."""
    m = R.shape[0]
    y = np.zeros(m, dtype=R.dtype)
    for i in range(m - 1, -1, -1):
        s = g[i] - R[i, i + 1:] @ y[i + 1:]
        y[i] = s / R[i, i]
    return y


def _as_matvec(A):
    if A is None:
        return None
    if callable(A) and not hasattr(A, "matvec"):
        return A
    if hasattr(A, "matvec"):
        return A.matvec
    return lambda v: A @ v

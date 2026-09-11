"""Equilibration for the coupled [u; t] BEM block systems.

The legacy systems mix displacement unknowns (km) and traction unknowns
(GPa) with H blocks that are O(1) and G blocks that scale like
L_char/mu — without any equilibration (none exists in the legacy code).
Two composable stages:

1. Physics (slot) scaling — column scale per unknown slot: u-slots get
   1, t-slots get mu_char/L_char... inverted so the scaled traction
   variable is t * L_char / mu_char, restoring displacement units. This
   is s-aware in the Laplace pipeline (mu_char tracks mu_tilde(s)).
2. Ruiz iterative equilibration of the (already physics-scaled) matrix,
   capped to a configurable range so scaling never fights the physics.

Scaled system: As = Dr @ A @ Dc, bs = Dr @ b, x = Dc @ y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def slot_scales(layout, L_char: float) -> np.ndarray:
    """Per-DOF column scale Dc (diagonal) for the physics stage.

    u-slots: 1. t-slots: L_char / mu_char, with mu_char the harmonic
    mean of |mu| over the regions incident to the patch (an interface
    traction couples both sides; a base traction has one region).
    """
    model = layout.model
    d = np.ones(layout.n_unknowns)
    for slot in layout.slots:
        if slot.kind != "t":
            continue
        regs = [r for r in model.regions
                if any(p is slot.patch for p in r.patches)]
        mus = np.array([abs(r.material.mu) for r in regs], dtype=float)
        mu_char = len(mus) / np.sum(1.0 / mus)
        d[slot.offset:slot.stop] = L_char / mu_char
    return d


def ruiz_scaling(A: np.ndarray, n_iter: int = 8,
                 cap: float = 1e6) -> tuple[np.ndarray, np.ndarray]:
    """Ruiz iterative row/column equilibration (infinity norm).

    Returns (dr, dc) such that diag(dr) @ A @ diag(dc) has rows and
    columns with unit max-norm (approximately). Scales are capped to
    [1/cap, cap] relative to 1 to avoid pathological amplification.
    """
    n, m = A.shape
    dr = np.ones(n)
    dc = np.ones(m)
    B = A.copy()
    for _ in range(n_iter):
        r = np.sqrt(np.max(np.abs(B), axis=1))
        c = np.sqrt(np.max(np.abs(B), axis=0))
        r[r == 0] = 1.0
        c[c == 0] = 1.0
        dr /= r
        dc /= c
        B = A * dr[:, None] * dc[None, :]
    np.clip(dr, 1.0 / cap, cap, out=dr)
    np.clip(dc, 1.0 / cap, cap, out=dc)
    return dr, dc


@dataclass
class ScaledSystem:
    """Equilibrated dense system with transparent solve mapping."""
    A_scaled: np.ndarray
    b_scaled: np.ndarray
    dr: np.ndarray
    dc: np.ndarray

    @classmethod
    def from_dense(cls, A: np.ndarray, b: np.ndarray,
                   col_physics: np.ndarray | None = None,
                   ruiz_iters: int = 8) -> "ScaledSystem":
        dc0 = np.ones(A.shape[1]) if col_physics is None else col_physics
        A1 = A * dc0[None, :]
        dr, dc1 = ruiz_scaling(A1, n_iter=ruiz_iters)
        dc = dc0 * dc1
        As = A * dr[:, None] * dc[None, :]
        return cls(A_scaled=As, b_scaled=b * dr, dr=dr, dc=dc)

    def unscale(self, y: np.ndarray) -> np.ndarray:
        return self.dc * y

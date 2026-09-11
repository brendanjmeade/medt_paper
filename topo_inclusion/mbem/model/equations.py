"""Block-equation generation from a RegionModel.

THE sign rule (single formula replacing every hand-written assembler):

For region R with material m(R), collocation patch q in dR, and
sigma(R,p) = +1 iff patch p's stored normals point out of R:

    A[row(R,q), u_p] += sigma(R,p) * H^{m(R)}_{qp} + 1/2 * delta_{qp} * I
    A[row(R,q), t_p] += -sigma(R,p) * G^{m(R)}_{qp}
    b[row(R,q)]      -= sum_{f in faults(R)} H^{m(R)}_{qf} @ slip_f
                        (+ prescribed-value columns moved to the RHS
                         with their LHS coefficients)

H is the T-kernel (slip/displacement -> displacement) influence matrix
assembled with the patch's stored normals; G is the U-kernel; the 1/2 I
collocation jump multiplies the single-valued u and never flips. The
shared interface traction unknown is t_p = sigma_stored * n_stored, so
region R sees sigma(R,p) * t_p — hence the -sigma on G.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import BCType, Patch, Region, RegionModel
from .layout import Slot, UnknownLayout


@dataclass(frozen=True)
class BlockTerm:
    row: Slot
    col: Slot
    kernel: str                  # "H" (T-kernel) | "G" (U-kernel)
    field_patch: Patch
    source_patch: Patch
    region: Region               # material provider
    scale: float                 # +-1 from the sigma rule
    diag_half: bool = False      # add +1/2 I on this block (q == p, u column)


@dataclass(frozen=True)
class RhsTerm:
    """b[row] += scale * K^{m(region)}_{q,source} @ vector."""
    row: Slot
    kernel: str
    field_patch: Patch
    source_patch: Patch
    region: Region
    scale: float
    vector: np.ndarray           # flattened known value (3*N_source,)
    add_half_of_vector: bool = False   # also b[row] += scale_half * vector
    scale_half: float = 0.0


@dataclass
class BlockSystem:
    model: RegionModel
    layout: UnknownLayout
    terms: list[BlockTerm] = field(default_factory=list)
    rhs_terms: list[RhsTerm] = field(default_factory=list)

    def mesh_pairs(self) -> set:
        return {(id(t.field_patch), id(t.source_patch), t.kernel)
                for t in self.terms}


def generate_system(model: RegionModel) -> BlockSystem:
    layout = UnknownLayout(model)
    system = BlockSystem(model=model, layout=layout)

    for region in model.regions:
        for q in region.patches:
            row = layout.row_slot(region, q)

            for p in region.patches:
                sigma = float(model.orientation(region, p))

                # ---- u_p term: sigma * H + (1/2) delta_qp I ----
                if layout.has_slot(p, "u"):
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(p, "u"), kernel="H",
                        field_patch=q, source_patch=p, region=region,
                        scale=sigma, diag_half=(p is q)))
                else:
                    # u_p prescribed: move (sigma H + 1/2 delta_qp I) @ u_bar
                    u_bar = p.value_array().ravel()
                    if np.any(u_bar):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel="H", field_patch=q,
                            source_patch=p, region=region, scale=-sigma,
                            vector=u_bar,
                            add_half_of_vector=(p is q), scale_half=-0.5))

                # ---- t_p term: -sigma * G ----
                if layout.has_slot(p, "t"):
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(p, "t"), kernel="G",
                        field_patch=q, source_patch=p, region=region,
                        scale=-sigma))
                else:
                    # t_p prescribed: move -sigma G @ t_bar to RHS (+sigma)
                    t_bar = p.value_array().ravel()
                    if np.any(t_bar):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel="G", field_patch=q,
                            source_patch=p, region=region, scale=sigma,
                            vector=t_bar))

            # ---- fault sources of this region (no sigma) ----
            for f in region.faults:
                slip = f.value_array().ravel()
                if np.any(slip):
                    system.rhs_terms.append(RhsTerm(
                        row=row, kernel="H", field_patch=q, source_patch=f,
                        region=region, scale=-1.0, vector=slip))

    return system

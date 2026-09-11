"""Deterministic unknown layout for a RegionModel.

Slot emission rule (reproduces the legacy orderings exactly):
regions in model order; within a region, patches in declaration order,
skipping patches already emitted; FREE_TRACTION emits its u-slot,
PRESCRIBED_DISPLACEMENT emits its t-slot, INTERFACE emits u then t.

Row assignment: the BIE of region R collocated on patch q lands on
  * q's u-slot if q is FREE_TRACTION,
  * q's t-slot if q is PRESCRIBED_DISPLACEMENT,
  * for INTERFACE: the first incident region (model order) takes the
    u-slot row, the second takes the t-slot row — keeping the
    second-kind (1/2 I + sigma H) blocks on the u-row diagonals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import BCType, Patch, Region, RegionModel


@dataclass(frozen=True)
class Slot:
    patch: Patch
    kind: str          # "u" | "t"
    offset: int
    size: int          # 3 * n_triangles

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.patch.name}"

    @property
    def stop(self) -> int:
        return self.offset + self.size


class UnknownLayout:
    def __init__(self, model: RegionModel):
        self.model = model
        self.slots: list[Slot] = []
        self._by_key: dict[tuple[int, str], Slot] = {}

        offset = 0
        emitted: set[int] = set()
        for region in model.regions:
            for patch in region.patches:
                if id(patch) in emitted:
                    continue
                emitted.add(id(patch))
                kinds = {
                    BCType.FREE_TRACTION: ("u",),
                    BCType.PRESCRIBED_DISPLACEMENT: ("t",),
                    BCType.INTERFACE: ("u", "t"),
                }[patch.bc]
                for kind in kinds:
                    slot = Slot(patch, kind, offset, 3 * patch.n_triangles)
                    self.slots.append(slot)
                    self._by_key[(id(patch), kind)] = slot
                    offset += slot.size
        self.n_unknowns = offset

    def slot(self, patch: Patch, kind: str) -> Slot:
        return self._by_key[(id(patch), kind)]

    def has_slot(self, patch: Patch, kind: str) -> bool:
        return (id(patch), kind) in self._by_key

    def row_slot(self, region: Region, patch: Patch) -> Slot:
        if patch.bc is BCType.FREE_TRACTION:
            return self.slot(patch, "u")
        if patch.bc is BCType.PRESCRIBED_DISPLACEMENT:
            return self.slot(patch, "t")
        if patch.bc is BCType.INTERFACE:
            first, second = self.model.interface_regions(patch)
            if region is first:
                return self.slot(patch, "u")
            if region is second:
                return self.slot(patch, "t")
            raise ValueError(f"region '{region.name}' is not incident to "
                             f"interface '{patch.name}'")
        raise ValueError(f"no collocation row for bc {patch.bc}")

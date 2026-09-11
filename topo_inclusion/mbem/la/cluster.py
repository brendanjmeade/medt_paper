"""Cluster trees and admissibility-driven block partitions.

Clean reimplementation of the legacy ``hmatrix.py`` clustering (which is
frozen as an oracle): binary principal-axis bisection over element
centroids, bounding-sphere eta-admissibility, and a recursive partition
of a (field, source) mesh pair into admissible (low-rank) and inadmissible
leaf (dense) blocks. All indices are ELEMENT indices; DOF interleaving
(3 per element) is the storage layer's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import defaults


class ClusterNode:
    """Cluster with an axis-aligned bounding box.

    AABBs (not bounding spheres) are essential here: bounding spheres of
    spherical-cap clusters on concentric shells overlap massively and
    report zero separation, so admissibility never triggers — measured,
    not hypothetical. Boxes capture the radial gap.
    """

    __slots__ = ("indices", "lo", "hi", "left", "right")

    def __init__(self, indices: np.ndarray, centroids: np.ndarray):
        self.indices = indices
        pts = centroids[indices]
        self.lo = pts.min(axis=0)
        self.hi = pts.max(axis=0)
        self.left = None
        self.right = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def n(self) -> int:
        return len(self.indices)

    def diameter(self) -> float:
        return float(np.linalg.norm(self.hi - self.lo))

    def distance_to(self, other: "ClusterNode") -> float:
        gap = np.maximum(0.0, np.maximum(self.lo - other.hi,
                                         other.lo - self.hi))
        return float(np.linalg.norm(gap))


def build_cluster_tree(centroids: np.ndarray,
                       min_leaf: int = defaults.CLUSTER_MIN_LEAF
                       ) -> ClusterNode:
    def _build(idx):
        node = ClusterNode(idx, centroids)
        if len(idx) <= min_leaf:
            return node
        pts = centroids[idx]
        mean = pts.mean(axis=0)
        cov = (pts - mean).T @ (pts - mean)
        _, vecs = np.linalg.eigh(cov)
        proj = (pts - mean) @ vecs[:, -1]
        median = np.median(proj)
        left_mask = proj < median
        if left_mask.sum() == 0 or (~left_mask).sum() == 0:
            return node
        node.left = _build(idx[left_mask])
        node.right = _build(idx[~left_mask])
        return node

    return _build(np.arange(centroids.shape[0]))


def is_admissible(cf: ClusterNode, cs: ClusterNode,
                  eta: float = defaults.ADMISSIBILITY_ETA) -> bool:
    return min(cf.diameter(), cs.diameter()) < eta * cf.distance_to(cs)


@dataclass
class BlockPartition:
    """Element-index block lists for one mesh pair."""
    admissible: list = field(default_factory=list)    # (rows, cols)
    dense: list = field(default_factory=list)         # (rows, cols)

    @property
    def n_blocks(self) -> int:
        return len(self.admissible) + len(self.dense)


def build_partition(tree_f: ClusterNode, tree_s: ClusterNode,
                    eta: float = defaults.ADMISSIBILITY_ETA,
                    max_admissible: int = 4096,
                    min_aca: int = defaults.ACA_MIN_BLOCK) -> BlockPartition:
    """Partition the (field x source) interaction into blocks.

    ``max_admissible`` caps admissible block side length (elements) so a
    single ACA never spans the whole mesh at the top of the tree.
    ``min_aca`` routes admissible blocks too small for certifiable cross
    approximation to the dense list instead.
    """
    part = BlockPartition()

    def _descend(cf: ClusterNode, cs: ClusterNode):
        oversize = cf.n > max_admissible or cs.n > max_admissible
        if is_admissible(cf, cs, eta) and not oversize:
            if min(cf.n, cs.n) < min_aca:
                part.dense.append((cf.indices, cs.indices))
            else:
                part.admissible.append((cf.indices, cs.indices))
        elif cf.is_leaf and cs.is_leaf:
            part.dense.append((cf.indices, cs.indices))
        else:
            for child_f in ([cf.left, cf.right] if not cf.is_leaf else [cf]):
                for child_s in ([cs.left, cs.right] if not cs.is_leaf
                                else [cs]):
                    _descend(child_f, child_s)

    _descend(tree_f, tree_s)
    return part

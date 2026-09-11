"""Compressed backend: block-compressed operators + preconditioned FGMRES.

Same BlockSystem contract as the dense backend, but every (field,
source, kernel) pair is stored ONCE as a material-basis PairCompressed;
materials only enter through coefficient vectors. ``rebuild_for_materials``
is therefore nearly free (the compressed geometry is shared), and the
preconditioner refactorizes only its small dense diagonal blocks.
"""

from __future__ import annotations

import numpy as np

from .. import defaults
from ..kernels import basis as kb
from ..la.hop import PairCompressed
from ..la.preconditioner import BlockGaussSeidel
from ..la.solver import fgmres
from ..model.equations import BlockSystem


class HBackend:
    def __init__(self, tol: float = defaults.BLOCK_COMPRESSION_TOL,
                 min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                 eta: float = defaults.ADMISSIBILITY_ETA,
                 max_admissible: int = 4096,
                 verbose: bool = False):
        self.opts = dict(tol=tol, min_leaf=min_leaf, eta=eta,
                         max_admissible=max_admissible)
        self.verbose = verbose

    def assemble(self, system: BlockSystem, eps) -> "AssembledH":
        return AssembledH(system, eps, self.opts, self.verbose)


def _coeffs(kernel: str, mat) -> np.ndarray:
    return kb.t_coeffs(mat.mu, mat.lam) if kernel == "H" \
        else kb.u_coeffs(mat.mu, mat.lam)


class AssembledH:
    def __init__(self, system: BlockSystem, eps, opts: dict, verbose: bool,
                 _shared=None):
        self.system = system
        self.layout = system.layout
        self.eps = eps
        self.opts = opts
        self.verbose = verbose
        self.materials = {r.name: r.material for r in system.model.regions}

        if _shared is None:
            self._pairs: dict = {}
            self._tree_cache: dict = {}
            pair_keys = {(id(t.field_patch), id(t.source_patch), t.kernel):
                         (t.field_patch, t.source_patch, t.kernel)
                         for t in system.terms}
            for rt in system.rhs_terms:
                pair_keys[(id(rt.field_patch), id(rt.source_patch),
                           rt.kernel)] = (rt.field_patch, rt.source_patch,
                                          rt.kernel)
            for key, (fp, sp, kern) in pair_keys.items():
                pc = PairCompressed(
                    fp.mesh, sp.mesh, kern, self.eps_for(sp),
                    tree_cache=self._tree_cache, **opts)
                if verbose:
                    print(f"  {fp.name} <- {sp.name} [{kern}]: "
                          f"{pc.summary()}")
                self._pairs[key] = pc
        else:
            self._pairs, self._tree_cache = _shared

        self.b = self._build_rhs()
        self._precond = None

    # -- helpers --------------------------------------------------------

    def eps_for(self, patch) -> np.ndarray:
        e = self.eps[patch.name] if isinstance(self.eps, dict) else self.eps
        return kb.as_eps_array(e, patch.n_triangles)

    def pair_for(self, term) -> PairCompressed:
        return self._pairs[(id(term.field_patch), id(term.source_patch),
                            term.kernel)]

    def _build_rhs(self) -> np.ndarray:
        b = np.zeros(self.layout.n_unknowns)
        for rt in self.system.rhs_terms:
            pair = self.pair_for(rt)
            mat = self.materials[rt.region.name]
            contrib = rt.scale * pair.matvec(_coeffs(rt.kernel, mat),
                                             rt.vector)
            b[rt.row.offset:rt.row.stop] += contrib
            if rt.add_half_of_vector:
                b[rt.row.offset:rt.row.stop] += rt.scale_half * rt.vector
        return b

    # -- operator ---------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for term in self.system.terms:
            pair = self.pair_for(term)
            mat = self.materials[term.region.name]
            seg = x[term.col.offset:term.col.stop]
            y[term.row.offset:term.row.stop] += \
                term.scale * pair.matvec(_coeffs(term.kernel, mat), seg)
            if term.diag_half:
                y[term.row.offset:term.row.stop] += 0.5 * seg
        return y

    def to_dense(self) -> np.ndarray:
        n = self.layout.n_unknowns
        A = np.zeros((n, n))
        for term in self.system.terms:
            pair = self.pair_for(term)
            mat = self.materials[term.region.name]
            blk = pair.to_dense(_coeffs(term.kernel, mat))
            A[term.row.offset:term.row.stop,
              term.col.offset:term.col.stop] += term.scale * blk
            if term.diag_half:
                idx = np.arange(term.row.size)
                A[term.row.offset + idx, term.col.offset + idx] += 0.5
        return A

    # -- solve -------------------------------------------------------

    def solve(self, rtol: float = defaults.GMRES_RTOL,
              restart: int = defaults.GMRES_RESTART,
              maxiter: int = defaults.GMRES_MAXITER,
              x0: np.ndarray | None = None,
              precond_max_dense: int = defaults.MAX_DENSE_PRECOND_DOF):
        """Preconditioned FGMRES. Returns (slot dict, SolveReport)."""
        if self._precond is None:
            self._precond = BlockGaussSeidel(self, max_dense=precond_max_dense,
                                             verbose=self.verbose)
        x, report = fgmres(self.matvec, self.b, M=self._precond, x0=x0,
                           rtol=rtol, restart=restart, maxiter=maxiter)
        if self.verbose:
            print(f"  {report}")
        out = {s.name: x[s.offset:s.stop].reshape(-1, 3)
               for s in self.layout.slots}
        return out, report

    # -- viscoelastic hook --------------------------------------------

    def rebuild_for_materials(self, material_map: dict) -> "AssembledH":
        new = AssembledH(self.system, self.eps, self.opts, self.verbose,
                         _shared=(self._pairs, self._tree_cache))
        for name, mat in material_map.items():
            if name not in new.materials:
                raise KeyError(f"unknown region '{name}'")
            new.materials[name] = mat
        new.b = new._build_rhs()
        return new

    # -- stats -------------------------------------------------------

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self._pairs.values())

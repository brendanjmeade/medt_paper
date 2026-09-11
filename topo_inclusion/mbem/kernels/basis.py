"""Material-basis assembly API.

The mollified Kelvin U/T influence matrices decompose exactly as

    M(material) = sum_k c_k(mu, lam) * B_k

with GEOMETRY-ONLY basis matrices B_k (3 for the U kernel, 6 for the T
kernel; eps^2 is baked into the relevant B_k). Assemble the basis once
per (field mesh, source mesh, eps) and recombine for every region
material and every Laplace sample — including complex mu_tilde(s).

BINDING RULES (from the approved plan's cross-review):
  * coefficients are computed from (mu, lam) directly — NEVER via a
    1/(1-2nu) intermediate (it amplifies catastrophically at the fluid
    limit nu -> 1/2);
  * eps is an (N_src,) per-source-element array everywhere; a scalar is
    promoted to a constant array (== legacy global-eps behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tri_kernels as tk


# ---------------------------------------------------------------------
# Material coefficient functions (complex-safe)
# ---------------------------------------------------------------------

def _mu_lam(material=None, mu=None, lam=None):
    if material is not None:
        return material.mu, material.lam
    if mu is None or lam is None:
        raise ValueError("provide either material or (mu, lam)")
    return mu, lam


def u_coeffs(mu, lam) -> np.ndarray:
    """Coefficients (3,) for the U-kernel basis [I1*d, eps^2*I3*d, T2[3]]."""
    nu = lam / (2.0 * (lam + mu))
    one_minus_nu = (lam + 2.0 * mu) / (2.0 * (lam + mu))
    C1 = 1.0 / (16.0 * np.pi * mu * one_minus_nu)
    c34 = 3.0 - 4.0 * nu
    g1 = c34 * C1
    g2 = 1.0 / (8.0 * np.pi * mu)          # = 2*(1-nu)*C1, the (1-nu) cancels
    g3 = C1
    return np.array([g1, g2, g3])


def t_coeffs(mu, lam) -> np.ndarray:
    """Coefficients (6,) for the T-kernel basis [L1, L2, L3, M1, M2, M3]."""
    nu = lam / (2.0 * (lam + mu))
    one_minus_nu = (lam + 2.0 * mu) / (2.0 * (lam + mu))
    C1 = 1.0 / (16.0 * np.pi * mu * one_minus_nu)
    c34 = 3.0 - 4.0 * nu
    cL = lam * C1
    cM = mu * C1                            # = 1/(16*pi*(1-nu)), mu cancels
    blob = 6.0 * one_minus_nu
    return np.array([c34 * cL, -cL, blob * cL,
                     c34 * cM, -cM, blob * cM])


# ---------------------------------------------------------------------
# Mesh-input normalization
# ---------------------------------------------------------------------

def _source_arrays(mesh_source):
    tri_verts = mesh_source.vertices[mesh_source.triangles]
    normals, _ = mesh_source.normals_and_areas()
    return np.ascontiguousarray(tri_verts), np.ascontiguousarray(normals)


def _field_points(mesh_or_points):
    if hasattr(mesh_or_points, "centroids"):
        return np.ascontiguousarray(mesh_or_points.centroids())
    return np.ascontiguousarray(np.asarray(mesh_or_points, dtype=float))


def as_eps_array(eps, n_source: int) -> np.ndarray:
    """Promote scalar eps to (N_src,); validate array length otherwise."""
    arr = np.asarray(eps, dtype=float)
    if arr.ndim == 0:
        return np.full(n_source, float(arr))
    if arr.shape != (n_source,):
        raise ValueError(f"eps array shape {arr.shape} != ({n_source},)")
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------
# Basis containers
# ---------------------------------------------------------------------

@dataclass
class UBasis:
    """Geometry-only U-kernel basis stack, shape (3, 3*N_f, 3*N_s)."""
    stack: np.ndarray

    def combine(self, material=None, mu=None, lam=None) -> np.ndarray:
        c = u_coeffs(*_mu_lam(material, mu, lam))
        return np.tensordot(c, self.stack, axes=1)

    def nbytes(self) -> int:
        return self.stack.nbytes


@dataclass
class TBasis:
    """Geometry-only T-kernel basis stack, shape (6, 3*N_f, 3*N_s)."""
    stack: np.ndarray

    def combine(self, material=None, mu=None, lam=None) -> np.ndarray:
        c = t_coeffs(*_mu_lam(material, mu, lam))
        return np.tensordot(c, self.stack, axes=1)

    def nbytes(self) -> int:
        return self.stack.nbytes


# ---------------------------------------------------------------------
# Assembly entry points
# ---------------------------------------------------------------------

def assemble_u_basis(mesh_field, mesh_source, eps) -> UBasis:
    x_field = _field_points(mesh_field)
    tri_verts, _ = _source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    return UBasis(tk.u_basis_matrices(x_field, tri_verts, eps_arr))


def assemble_t_basis(mesh_field, mesh_source, eps) -> TBasis:
    x_field = _field_points(mesh_field)
    tri_verts, normals = _source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    return TBasis(tk.t_basis_matrices(x_field, tri_verts, normals, eps_arr))


def assemble_u_matrix(mesh_field, mesh_source, material, eps) -> np.ndarray:
    """One-shot real-material U matrix (coefficients applied in-loop)."""
    x_field = _field_points(mesh_field)
    tri_verts, _ = _source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    g1, g2, g3 = u_coeffs(material.mu, material.lam)
    return tk.u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3)


def assemble_t_matrix(mesh_field, mesh_source, material, eps) -> np.ndarray:
    """One-shot real-material T matrix (coefficients applied in-loop)."""
    x_field = _field_points(mesh_field)
    tri_verts, normals = _source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    c = t_coeffs(material.mu, material.lam)
    return tk.t_matrix_direct(x_field, tri_verts, normals, eps_arr, *c)

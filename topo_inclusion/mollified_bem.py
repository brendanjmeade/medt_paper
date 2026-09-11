"""
Multi-domain BEM with Mollified Elastic Kernels on the Sphere

This module implements:
  1. Mollified (regularized) Kelvin displacement and traction kernels
  2. Icosphere mesh generation for the Earth's free surface
  3. Triangle mesh utilities (normals, areas, centroids)
  4. BEM matrix assembly with free-surface and interface conditions
  5. Fault source as displacement discontinuity
  6. H-matrix compression (placeholder for future)

The mollified kernels replace the singular 1/r Kelvin solution with a
smooth approximation using r_eps = sqrt(r^2 + eps^2), eliminating all
singular and hypersingular integrals.

Author: Brendan Meade & Claude
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.linalg import lu_factor, lu_solve


# ============================================================
# Material properties
# ============================================================

@dataclass
class ElasticMaterial:
    """Isotropic elastic material."""
    mu: float    # shear modulus (GPa)
    lam: float   # first Lamé parameter (GPa)

    @property
    def nu(self):
        """Poisson's ratio."""
        return self.lam / (2.0 * (self.lam + self.mu))

    @property
    def E(self):
        """Young's modulus."""
        return self.mu * (3.0 * self.lam + 2.0 * self.mu) / (self.lam + self.mu)


# ============================================================
# Mollified Kelvin kernels
# ============================================================

def kelvin_U_mollified(r_vec, mu, nu, eps):
    """Mollified Kelvin displacement Green's function.

    The classical Kelvin solution for displacement at x due to a
    point force at x' in an infinite isotropic elastic medium is:

        U_ij(r) = 1/(16 pi mu (1-nu)) * [
            (3 - 4nu) delta_ij / r  +  r_i r_j / r^3
        ]

    where r = x - x', r = |r|.

    The mollified version replaces:
        1/r   -> 1/r_eps           where r_eps = sqrt(r^2 + eps^2)
        1/r^3 -> 1/r_eps^3

    This is equivalent to convolving the point force with a smooth
    blob of width eps, which is the exact solution to Navier's equation
    with a distributed load (the regularized Kelvinlet approach).

    Args:
        r_vec: (3,) or (N, 3) displacement vectors (x - x')
        mu: shear modulus
        nu: Poisson's ratio
        eps: mollification parameter

    Returns:
        U: (3, 3) or (N, 3, 3) displacement kernel
    """
    r_vec = np.atleast_2d(r_vec)  # (N, 3)
    N = r_vec.shape[0]

    r2 = np.sum(r_vec**2, axis=1)  # (N,)
    r_eps = np.sqrt(r2 + eps**2)   # (N,)

    prefac = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))

    a = (3.0 - 4.0 * nu)

    # Vectorized: U_ij = prefac * (a * dij / r_eps + ri*rj / r_eps^3)
    inv_r_eps = 1.0 / r_eps     # (N,)
    inv_r_eps3 = inv_r_eps**3   # (N,)

    # Outer product r_i * r_j: (N, 3, 3)
    rr = r_vec[:, :, None] * r_vec[:, None, :]

    # Identity part: a * dij / r_eps
    U = prefac * (a * np.eye(3)[None, :, :] * inv_r_eps[:, None, None]
                  + rr * inv_r_eps3[:, None, None])

    return U


def kelvin_T_mollified(r_vec, n_vec, mu, nu, eps):
    """Mollified Kelvin traction kernel.

    The classical traction kernel T_ij(r, n) gives the j-th traction
    component at a point with outward normal n, due to a unit point
    force in the i-th direction applied at the origin:

        T_ij(r, n) = -1/(8 pi (1-nu) r^2) * [
            dr/dn * ((1 - 2nu) delta_ij + 3 r_i r_j / r^2)
            + (1 - 2nu)(n_i r_j - n_j r_i) / r^2
        ]

    where dr/dn = (r . n) / r.

    The mollified version replaces powers of 1/r with 1/r_eps.
    Specifically:
        1/r^2 -> 1/r_eps^2
        1/r^3 -> 1/r_eps^3  (for the dr/dn terms)
        r_i/r -> r_i/r_eps   (unit vector regularized)

    The key insight: since T involves derivatives of U, and U is
    already mollified, T inherits the smoothness. The mollified T
    is the exact traction from the mollified displacement field.

    We compute T from the stress of the mollified U field using
    Hooke's law and then contracting with n.

    Args:
        r_vec: (N, 3) displacement vectors
        n_vec: (N, 3) or (3,) outward normal at field point
        mu: shear modulus
        nu: Poisson's ratio
        eps: mollification parameter

    Returns:
        T: (N, 3, 3) traction kernel T_ij
           T_ij * f_j = traction_i due to force f at source
    """
    r_vec = np.atleast_2d(r_vec)  # (N, 3)
    n_vec = np.atleast_2d(n_vec)  # (N, 3) or (1, 3)
    if n_vec.shape[0] == 1:
        n_vec = np.broadcast_to(n_vec, r_vec.shape)
    N = r_vec.shape[0]

    r2 = np.sum(r_vec**2, axis=1)             # (N,)
    r_eps2 = r2 + eps**2                        # (N,)
    r_eps = np.sqrt(r_eps2)                     # (N,)
    r_eps3 = r_eps * r_eps2                     # (N,) = r_eps^3
    r_eps5 = r_eps3 * r_eps2                    # (N,) = r_eps^5

    # r dot n
    r_dot_n = np.sum(r_vec * n_vec, axis=1)  # (N,)

    prefac = -1.0 / (8.0 * np.pi * (1.0 - nu))

    # Vectorized T kernel
    coeff_12nu = 1.0 - 2.0 * nu

    # Outer product r_i * r_j: (N, 3, 3)
    rr = r_vec[:, :, None] * r_vec[:, None, :]

    # Term 1: r_dot_n * [(1-2nu) dij / r_eps^3 + 3 ri rj / r_eps^5]
    term1 = (r_dot_n[:, None, None] *
             (coeff_12nu * np.eye(3)[None, :, :] / r_eps3[:, None, None]
              + 3.0 * rr / r_eps5[:, None, None]))

    # Term 2: (1-2nu)(ni rj - nj ri) / r_eps^3
    nr = n_vec[:, :, None] * r_vec[:, None, :]  # ni * rj
    rn = r_vec[:, :, None] * n_vec[:, None, :]  # ri * nj
    term2 = coeff_12nu * (nr - rn) / r_eps3[:, None, None]

    T = prefac * (term1 + term2)

    return T


# ============================================================
# Mesh data structures
# ============================================================

@dataclass
class TriMesh:
    """Triangle surface mesh."""
    vertices: np.ndarray   # (n_verts, 3)
    triangles: np.ndarray  # (n_tris, 3) integer indices

    @property
    def n_triangles(self):
        return self.triangles.shape[0]

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    def centroids(self):
        """Triangle centroids."""
        v = self.vertices[self.triangles]  # (n_tri, 3, 3)
        return v.mean(axis=1)  # (n_tri, 3)

    def normals_and_areas(self):
        """Outward normals and areas for each triangle."""
        v = self.vertices[self.triangles]  # (n_tri, 3, 3)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        cross = np.cross(e1, e2)
        area2 = np.linalg.norm(cross, axis=1)  # 2 * area
        normals = cross / area2[:, None]
        areas = 0.5 * area2
        return normals, areas

    def ensure_outward_normals(self, center=np.array([0.0, 0.0, 0.0])):
        """Flip triangles so normals point away from center."""
        centroids = self.centroids()
        normals, _ = self.normals_and_areas()
        outward = centroids - center
        dots = np.sum(normals * outward, axis=1)
        flip = dots < 0
        self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]


# ============================================================
# Sphere mesh generation (icosphere)
# ============================================================

def make_icosphere(radius=6371.0, n_refine=2):
    """Generate an icosphere by subdividing an icosahedron.

    Args:
        radius: sphere radius (km)
        n_refine: number of subdivision levels
            0 -> 20 triangles (very coarse)
            1 -> 80 triangles
            2 -> 320 triangles
            3 -> 1280 triangles
            4 -> 5120 triangles

    Returns:
        TriMesh
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # 12 vertices of icosahedron
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)

    # Normalize to unit sphere
    verts /= np.linalg.norm(verts[0])

    # 20 faces of icosahedron
    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)

    # Subdivide
    for _ in range(n_refine):
        verts, faces = _subdivide(verts, faces)

    # Scale to radius
    verts *= radius

    mesh = TriMesh(vertices=verts, triangles=faces)
    mesh.ensure_outward_normals()
    return mesh


def _subdivide(verts, faces):
    """Subdivide each triangle into 4 by adding edge midpoints."""
    edge_map = {}
    new_verts = list(verts)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_map:
            return edge_map[key]
        mid = 0.5 * (verts[i] + verts[j])
        mid /= np.linalg.norm(mid)  # project to unit sphere
        idx = len(new_verts)
        new_verts.append(mid)
        edge_map[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca],
        ])

    return np.array(new_verts), np.array(new_faces, dtype=int)


def _subdivide_selective(verts, faces, mask):
    """Subdivide only marked triangles into 4, keeping others intact.

    Conforming closure is NOT done here — call _conforming_closure() first
    to expand the mask to avoid T-junctions.

    Args:
        verts: (N_v, 3) vertices on the unit sphere
        faces: (N_f, 3) triangle indices
        mask: (N_f,) bool — True for triangles to subdivide

    Returns:
        new_verts, new_faces (on unit sphere)
    """
    edge_map = {}
    new_verts = list(verts)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_map:
            return edge_map[key]
        mid = 0.5 * (verts[i] + verts[j])
        nrm = np.linalg.norm(mid)
        if nrm > 0:
            mid /= nrm  # project to unit sphere
        idx = len(new_verts)
        new_verts.append(mid)
        edge_map[key] = idx
        return idx

    new_faces = []
    for fi, tri in enumerate(faces):
        if mask[fi]:
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            ])
        else:
            new_faces.append(list(tri))

    return np.array(new_verts), np.array(new_faces, dtype=int)


def _conforming_closure(faces, mask):
    """Expand refinement mask to avoid T-junctions (non-conforming edges).

    Any triangle sharing an edge with a marked triangle must also be marked.
    Repeat until stable.

    Args:
        faces: (N_f, 3) triangle indices
        mask: (N_f,) bool — initial refinement marks

    Returns:
        expanded mask (N_f,) bool
    """
    mask = mask.copy()
    N_f = len(faces)

    # Build edge -> triangle adjacency
    edge_to_tris = {}
    for fi in range(N_f):
        tri = faces[fi]
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = (min(e), max(e))
            if key not in edge_to_tris:
                edge_to_tris[key] = []
            edge_to_tris[key].append(fi)

    # Iterate until stable
    changed = True
    while changed:
        changed = False
        for fi in range(N_f):
            if not mask[fi]:
                continue
            tri = faces[fi]
            for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                key = (min(e), max(e))
                for fj in edge_to_tris[key]:
                    if not mask[fj]:
                        mask[fj] = True
                        changed = True
    return mask


def make_adaptive_icosphere(radius=6371.0, n_base=2, refine_center=None,
                            refine_radius=3000.0, n_extra=3, taper=0.6):
    """Generate an adaptively refined icosphere.

    Starts with a coarse base mesh and selectively refines triangles near
    a specified center point. Uses conforming closure to prevent T-junctions.

    Args:
        radius: sphere radius (km)
        n_base: base refinement level (2 -> 320 elements)
        refine_center: (3,) Cartesian point to refine around (km)
        refine_radius: initial distance threshold for refinement (km)
        n_extra: number of additional local refinement passes
        taper: factor to shrink refine_radius each pass

    Returns:
        TriMesh with adaptive resolution
    """
    if refine_center is None:
        refine_center = np.array([radius, 0.0, 0.0])

    # Build base icosphere on unit sphere, then refine selectively
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])

    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)

    # Uniform refinement to base level
    for _ in range(n_base):
        verts, faces = _subdivide(verts, faces)

    # Normalize refine_center to unit sphere for angular distance
    rc_hat = refine_center / np.linalg.norm(refine_center)

    # Selective refinement passes
    cur_radius = refine_radius
    for level in range(n_extra):
        # Compute centroids on unit sphere
        v_tri = verts[faces]  # (N_f, 3, 3) on unit sphere
        centroids = v_tri.mean(axis=1)  # (N_f, 3)
        # Normalize centroids to unit sphere
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_hat = centroids / c_norms

        # Great-circle (angular) distance in km
        dots = np.clip(np.sum(centroids_hat * rc_hat, axis=1), -1.0, 1.0)
        dist = np.arccos(dots) * radius  # km

        mask = dist < cur_radius
        n_marked = np.sum(mask)
        if n_marked == 0:
            break

        # Note: conforming closure is NOT needed for constant-element BEM
        # since there is no inter-element continuity requirement.
        # T-junctions are harmless for piecewise-constant approximation.

        print(f"  Adaptive level {level}: {n_marked}/{len(faces)} "
              f"elements refined (radius={cur_radius:.0f} km)")

        verts, faces = _subdivide_selective(verts, faces, mask)
        cur_radius *= taper

    # Scale to physical radius
    verts *= radius

    mesh = TriMesh(vertices=verts, triangles=faces)
    mesh.ensure_outward_normals()
    return mesh


# ============================================================
# Flat fault mesh generation
# ============================================================

def make_fault_mesh(center, strike, dip, length, width, n_along=4, n_down=4):
    """Generate a rectangular fault mesh.

    Strike and dip are defined in the local tangent plane at the fault
    center on the sphere:
      - strike: azimuth CW from local north (degrees)
      - dip: angle from horizontal (degrees), measured in the plane
        perpendicular to strike (right-hand rule: dip direction is
        90 degrees CW from strike when viewed from above)

    Args:
        center: (3,) center of fault in Cartesian coords (km)
        strike: strike angle (degrees, CW from local north)
        dip: dip angle (degrees, from horizontal)
        length: along-strike length (km)
        width: down-dip width (km)
        n_along: elements along strike
        n_down: elements down dip

    Returns:
        TriMesh for the fault surface, fault_normal, strike_direction
    """
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)

    # Local tangent plane basis at fault center
    r_hat = center / np.linalg.norm(center)  # local up (radial)

    # Local east and north on the sphere
    # east = z_hat x r_hat (projected), north = r_hat x east
    z_hat = np.array([0.0, 0.0, 1.0])
    local_east = np.cross(z_hat, r_hat)
    east_norm = np.linalg.norm(local_east)
    if east_norm < 1e-10:
        # At poles, use y-hat as reference
        local_east = np.array([0.0, 1.0, 0.0])
    else:
        local_east /= east_norm
    local_north = np.cross(r_hat, local_east)
    local_north /= np.linalg.norm(local_north)

    # Strike direction in local tangent plane (CW from north)
    s_hat = np.cos(strike_rad) * local_north + np.sin(strike_rad) * local_east

    # Horizontal dip direction (90 deg CW from strike, looking down)
    horiz_dip = np.sin(strike_rad) * local_north - np.cos(strike_rad) * local_east
    # Wait: 90 CW from strike. If strike points north, dip direction points east.
    # Rotation CW by 90: (cos, sin) -> (sin, -cos) ... let me redo.
    # CW rotation of s_hat by 90 degrees in the tangent plane:
    horiz_dip = np.cross(r_hat, s_hat)  # perpendicular to strike, in tangent plane
    # This gives the direction 90 CCW from strike looking from outside.
    # Convention: dip direction is to the RIGHT of the strike direction.
    # cross(up, strike) points LEFT of strike, so negate:
    horiz_dip = -np.cross(r_hat, s_hat)
    horiz_dip /= np.linalg.norm(horiz_dip)

    # Down-dip direction: rotate from horizontal dip into -up by dip angle
    d_hat = np.cos(dip_rad) * horiz_dip - np.sin(dip_rad) * r_hat

    # Normal (right-hand rule: s x d)
    n_hat = np.cross(s_hat, d_hat)
    n_hat /= np.linalg.norm(n_hat)

    # Grid of vertices
    ss = np.linspace(-length/2, length/2, n_along + 1)
    dd = np.linspace(-width/2, width/2, n_down + 1)

    verts = []
    for d_val in dd:
        for s_val in ss:
            v = center + s_val * s_hat + d_val * d_hat
            verts.append(v)
    verts = np.array(verts)

    # Triangulate
    faces = []
    nx = n_along + 1
    for j in range(n_down):
        for i in range(n_along):
            v00 = j * nx + i
            v10 = j * nx + i + 1
            v01 = (j+1) * nx + i
            v11 = (j+1) * nx + i + 1
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])
    faces = np.array(faces, dtype=int)

    mesh = TriMesh(vertices=verts, triangles=faces)
    return mesh, n_hat, s_hat


# ============================================================
# BEM matrix assembly
# ============================================================

def assemble_BEM_matrices(mesh_field, mesh_source, material, eps,
                          kernel="U"):
    """Assemble BEM influence matrix using analytical triangle integration.

    For each field element i (centroid-collocated) and source element j,
    compute the exact analytical integral of the mollified kernel over
    source element j at the field centroid. Uses the per-triangle
    analytical integrators from ``mollified_kernel.analytical_kernels``
    (Kelvin G for kernel="U", displacement-discontinuity for kernel="T"),
    vectorized over observation points for each source triangle.

    Returns:
        Matrix of shape (3*N_field, 3*N_source)
    """
    from mollified_kernel.analytical_batch import (
        assemble_U_matrix_batch,
        assemble_T_matrix_batch,
    )

    mu = material.mu
    nu = material.nu

    x_field = mesh_field.centroids()
    n_source, _ = mesh_source.normals_and_areas()
    tri_verts = mesh_source.vertices[mesh_source.triangles]  # (N_s, 3, 3)

    if kernel == "U":
        return assemble_U_matrix_batch(x_field, tri_verts, mu, nu, eps)
    elif kernel == "T":
        return assemble_T_matrix_batch(x_field, tri_verts, n_source,
                                       mu, nu, eps)
    raise ValueError(f"Unknown kernel: {kernel}")


# ============================================================
# Single-region BEM solver (homogeneous sphere + fault)
# ============================================================

def solve_homogeneous_sphere(sphere_mesh, fault_mesh, fault_normal,
                              fault_slip_vector, slip_magnitude,
                              material, eps, verbose=True):
    """Solve for displacement on a homogeneous sphere with a fault.

    Physics:
    - The sphere surface has free-surface BC: traction = 0
    - The fault imposes a displacement discontinuity

    BEM formulation (indirect method using single-layer potential):
    We represent the solution as displacement due to an unknown
    traction distribution φ on the free surface, plus the known
    displacement discontinuity on the fault:

        u(x) = ∫_surf U(x, x') · φ(x') dS' + u_fault(x)

    where u_fault(x) is the displacement from the fault slip (computed
    using the displacement discontinuity kernel).

    The free-surface condition t(x) = 0 on the sphere gives:

        ∫_surf T(x, x') · φ(x') dS' + t_fault(x) = 0    for x on surf

    Wait -- this isn't quite right for an indirect formulation.

    Let me use the DIRECT BEM formulation instead:

    For a point x on the boundary of a domain with volume Ω:

        c(x) u(x) = ∫_∂Ω [U(x,x') t(x') - T(x,x') u(x')] dS'
                     + ∫_fault [U(x,x') Δt(x') - T(x,x') Δu(x')] dS'

    For a free surface: t(x') = 0 on the sphere surface.
    For the fault: Δu is prescribed (the slip), Δt = 0 (no opening).

    So:
        c(x) u(x) + ∫_surf T(x,x') u(x') dS' = -∫_fault T(x,x') Δu(x') dS'

    This gives us: [c·I + H] u_surf = -H_fault · Δu_fault

    where H is the T-kernel matrix (surface to surface) and
    H_fault is the T-kernel matrix (fault to surface).

    For exterior problems (we're outside looking in, but actually
    for a solid sphere we're inside), c = 0.5 I for smooth boundaries.

    Args:
        sphere_mesh: TriMesh for the free surface
        fault_mesh: TriMesh for the fault
        fault_normal: (3,) fault normal vector
        fault_slip_vector: (3,) slip direction
        slip_magnitude: scalar slip amount (km)
        material: ElasticMaterial
        eps: mollification parameter
        verbose: print progress

    Returns:
        u_surf: (N_surf, 3) displacement on the sphere surface
        u_fault: (N_fault, 3) displacement at fault centroids
    """
    N_surf = sphere_mesh.n_triangles
    N_fault = fault_mesh.n_triangles

    if verbose:
        print(f"Sphere elements: {N_surf}")
        print(f"Fault elements:  {N_fault}")
        print(f"System size:     {3*N_surf} x {3*N_surf}")

    # Prescribed slip (displacement discontinuity) on fault
    delta_u = np.zeros((N_fault, 3))
    for i in range(N_fault):
        delta_u[i] = slip_magnitude * fault_slip_vector

    # Assemble T-kernel matrix: surface to surface
    if verbose: print("Assembling H (surface-surface T-kernel)...")
    H_ss = assemble_BEM_matrices(sphere_mesh, sphere_mesh, material, eps,
                                  kernel="T")

    # Add c*I term (c = 0.5 for smooth boundary of solid domain)
    # The solid interior "sees" the boundary from inside
    c_val = 0.5
    for i in range(N_surf):
        for d in range(3):
            H_ss[3*i+d, 3*i+d] += c_val

    # Assemble T-kernel matrix: fault to surface
    if verbose: print("Assembling H (fault-surface T-kernel)...")
    H_fs = assemble_BEM_matrices(sphere_mesh, fault_mesh, material, eps,
                                  kernel="T")

    # RHS: -H_fault · Δu
    delta_u_flat = delta_u.ravel()  # (3*N_fault,)
    rhs = -H_fs @ delta_u_flat

    # Solve: (c*I + H_ss) u_surf = rhs
    if verbose: print("Solving linear system...")
    lu, piv = lu_factor(H_ss)
    u_surf_flat = lu_solve((lu, piv), rhs)
    u_surf = u_surf_flat.reshape((N_surf, 3))

    if verbose:
        u_max = np.max(np.linalg.norm(u_surf, axis=1))
        print(f"Max surface displacement: {u_max:.6e} km")

    # Compute displacement at fault centroids (for diagnostics)
    if verbose: print("Computing fault displacement...")
    # u at fault = contribution from surface tractions (=0) + fault slip
    # Since t=0 on surface, u at interior points comes from:
    #   u(x) = ∫_surf U(x,x') t(x') dS' - ∫_surf T(x,x') u(x') dS'
    #          + fault contribution
    # With t=0: u(x) = -∫_surf T(x,x') u(x') dS' + u_fault_direct(x)

    # T-kernel: surface to fault evaluation points
    H_sf = assemble_BEM_matrices(fault_mesh, sphere_mesh, material, eps,
                                  kernel="T")
    # U-kernel: fault to fault (self-contribution from slip)
    # For displacement from the DD: u_dd(x) = -∫_fault T(x,x') Δu(x') dS'
    H_ff = assemble_BEM_matrices(fault_mesh, fault_mesh, material, eps,
                                  kernel="T")

    # Note: for interior points (not on boundary), c = 1 (full interior point)
    u_fault_flat = -H_sf @ u_surf_flat - H_ff @ delta_u_flat
    u_fault = u_fault_flat.reshape((N_fault, 3))

    return u_surf, u_fault


# ============================================================
# Two-region BEM solver (sphere with material interface)
# ============================================================

def solve_two_region_sphere(sphere_mesh, interface_mesh, fault_mesh,
                             fault_normal, fault_slip_vector, slip_magnitude,
                             material_outer, material_inner, eps,
                             verbose=True):
    """Solve for displacement with two material regions.

    Region 1 (outer): between free surface and interface
    Region 2 (inner): inside the interface

    The fault is in Region 1 (outer region).

    Unknowns:
    - u on free surface (N_surf * 3)
    - u on interface (N_int * 3)
    - t on interface (N_int * 3)
    (t = 0 on free surface, so not an unknown there)

    Equations (with interface mesh normals n_I pointing outward from basin):

    1. Free surface BIE from Region 1 (outward on interface = -n_I):
       (0.5I + H^1_ss) u_s - H^1_si u_i + G^1_si t_i = -H^1_sf Δu_f

    2. Interface BIE from Region 1:
       H^1_is u_s + (0.5I - H^1_ii) u_i + G^1_ii t_i = -H^1_if Δu_f

    3. Interface BIE from Region 2 (outward on interface = +n_I):
       (0.5I + H^2_ii) u_i - G^2_ii t_i = 0

    Sign conventions:
    - Interface traction unknown: t_i = σ · n_I (w.r.t. mesh normals)
    - Region 1 outward normal at interface = -n_I
    - Region 2 outward normal at interface = +n_I
    - T(x,y,-n_I) = -T(x,y,n_I), explaining the sign flips in Eq 1 & 2

    Assembled as a block system and solved directly.

    Args:
        sphere_mesh: free surface mesh
        interface_mesh: material interface mesh (normal pointing inward)
        fault_mesh: fault mesh (in outer region)
        fault_normal: fault normal
        fault_slip_vector: slip direction
        slip_magnitude: slip magnitude
        material_outer: ElasticMaterial for outer region
        material_inner: ElasticMaterial for inner region
        eps: mollification parameter
        verbose: print progress

    Returns:
        u_surf: displacement on free surface
        u_int: displacement on interface
        t_int: traction on interface
    """
    N_s = sphere_mesh.n_triangles    # free surface
    N_i = interface_mesh.n_triangles  # interface
    N_f = fault_mesh.n_triangles      # fault

    N_unk = 3 * N_s + 3 * N_i + 3 * N_i  # u_s, u_i, t_i
    if verbose:
        print(f"Surface elements:   {N_s}")
        print(f"Interface elements: {N_i}")
        print(f"Fault elements:     {N_f}")
        print(f"Total unknowns:     {N_unk}")

    # Prescribed fault slip
    delta_u = np.tile(slip_magnitude * fault_slip_vector, (N_f, 1))
    delta_u_flat = delta_u.ravel()

    mu1, nu1 = material_outer.mu, material_outer.nu
    mu2, nu2 = material_inner.mu, material_inner.nu

    # ---- Region 1 matrices (outer material) ----
    if verbose: print("Assembling Region 1 (outer) matrices...")

    # T-kernel matrices (Region 1 material)
    H1_ss = assemble_BEM_matrices(sphere_mesh, sphere_mesh, material_outer, eps, "T")
    H1_si = assemble_BEM_matrices(sphere_mesh, interface_mesh, material_outer, eps, "T")
    H1_is = assemble_BEM_matrices(interface_mesh, sphere_mesh, material_outer, eps, "T")
    H1_ii = assemble_BEM_matrices(interface_mesh, interface_mesh, material_outer, eps, "T")

    # U-kernel matrices (for traction unknowns on interface)
    G1_si = assemble_BEM_matrices(sphere_mesh, interface_mesh, material_outer, eps, "U")
    G1_ii = assemble_BEM_matrices(interface_mesh, interface_mesh, material_outer, eps, "U")

    # Fault influence on surface and interface
    H1_sf = assemble_BEM_matrices(sphere_mesh, fault_mesh, material_outer, eps, "T")
    H1_if = assemble_BEM_matrices(interface_mesh, fault_mesh, material_outer, eps, "T")

    # ---- Region 2 matrices (inner material) ----
    if verbose: print("Assembling Region 2 (inner) matrices...")

    H2_ii = assemble_BEM_matrices(interface_mesh, interface_mesh, material_inner, eps, "T")
    G2_ii = assemble_BEM_matrices(interface_mesh, interface_mesh, material_inner, eps, "U")

    # ---- Assemble global system ----
    if verbose: print("Assembling global system...")

    # Block indices
    s0, s1 = 0, 3*N_s                    # u_surf
    i0, i1 = 3*N_s, 3*N_s + 3*N_i       # u_interface
    t0, t1 = 3*N_s + 3*N_i, N_unk       # t_interface

    A = np.zeros((N_unk, N_unk))
    b = np.zeros(N_unk)

    # === SIGN CONVENTIONS ===
    # Interface mesh normals (n_I) point outward from basin (Region 2).
    # Region 1 (outer) outward normal at interface = -n_I.
    # Region 2 (inner) outward normal at interface = +n_I.
    #
    # All H matrices are assembled with T-kernel using source normals = n_I
    # (the interface mesh normals). For Region 1's BIE at the interface,
    # T(x,y,-n_I) = -T(x,y,n_I) = -H, so interface T-terms get negated.
    # Region 1's interface traction: t^{out}_{Ω1} = σ·(-n_I) = -t_I.
    #
    # Eq 1: Free surface BIE from Region 1:
    #   (0.5I + H_ss) u_s - H_si u_i + G_si t_i = -H_sf Δu
    A[s0:s1, s0:s1] = H1_ss
    for k in range(3*N_s):
        A[s0+k, s0+k] += 0.5
    A[s0:s1, i0:i1] = -H1_si       # negated: Region 1 outward = -n_I
    A[s0:s1, t0:t1] = G1_si         # positive: t^{out}_{Ω1} = -t_I
    b[s0:s1] = -H1_sf @ delta_u_flat

    # Eq 2: Interface BIE from Region 1:
    #   H_is u_s + (0.5I - H_ii^{(1)}) u_i + G_ii^{(1)} t_i = -H_if Δu
    A[i0:i1, s0:s1] = H1_is
    A[i0:i1, i0:i1] = -H1_ii        # negated: Region 1 outward = -n_I
    for k in range(3*N_i):
        A[i0+k, i0+k] += 0.5        # c = +0.5 from Region 1 interior
    A[i0:i1, t0:t1] = G1_ii         # positive: t^{out}_{Ω1} = -t_I
    b[i0:i1] = -H1_if @ delta_u_flat

    # Eq 3: Interface BIE from Region 2:
    #   (0.5I + H_ii^{(2)}) u_i - G_ii^{(2)} t_i = 0
    # Region 2's outward normal = n_I (mesh normals), so no sign flip.
    # c = +0.5 from Region 2 interior.
    A[i0+3*N_i:t1, i0:i1] = H2_ii
    for k in range(3*N_i):
        A[i0+3*N_i+k, i0+k] += 0.5  # c = +0.5
    A[i0+3*N_i:t1, t0:t1] = -G2_ii  # negative on LHS
    # RHS = 0 (no sources in Region 2)

    # Solve
    if verbose: print(f"Solving {N_unk}x{N_unk} system...")
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)

    u_surf = x[s0:s1].reshape((N_s, 3))
    u_int = x[i0:i1].reshape((N_i, 3))
    t_int = x[t0:t1].reshape((N_i, 3))

    if verbose:
        print(f"Max surface displacement:   {np.max(np.linalg.norm(u_surf, axis=1)):.6e} km")
        print(f"Max interface displacement:  {np.max(np.linalg.norm(u_int, axis=1)):.6e} km")
        print(f"Max interface traction:      {np.max(np.linalg.norm(t_int, axis=1)):.6e} GPa")

    return u_surf, u_int, t_int


# ============================================================
# Three-region sphere (e.g. crust / asthenosphere / lower mantle)
# ============================================================

def solve_three_region_sphere(sphere_mesh, interface1_mesh, interface2_mesh,
                               fault_mesh, fault_normal, fault_slip_vector,
                               slip_magnitude, material_1, material_2,
                               material_3, eps, verbose=True):
    """Solve for displacement with three material regions on a sphere.

    Region 1 (outermost): between free surface and interface 1
    Region 2 (middle):    between interface 1 and interface 2
    Region 3 (innermost): inside interface 2

    The fault is in Region 1 (the outermost region).

    Interface normals convention: both interface meshes have outward-pointing
    normals (radially outward from the Earth center), i.e. n_I1 and n_I2
    both point outward.

    Unknowns (5 blocks):
    - u_s:   displacement on free surface          (3 * N_s)
    - u_i1:  displacement on interface 1            (3 * N_i1)
    - t_i1:  traction on interface 1 (σ·n_I1)      (3 * N_i1)
    - u_i2:  displacement on interface 2            (3 * N_i2)
    - t_i2:  traction on interface 2 (σ·n_I2)      (3 * N_i2)

    Block system (5 equations):
    Eq 1 — Region 1 BIE at surface:
      (½+H¹_ss) u_s - H¹_si1 u_i1 + G¹_si1 t_i1 = -H¹_sf Δu

    Eq 2 — Region 1 BIE at interface 1:
      H¹_i1s u_s + (½-H¹_i1i1) u_i1 + G¹_i1i1 t_i1 = -H¹_i1f Δu

    Eq 3 — Region 2 BIE at interface 1:
      (½+H²_i1i1) u_i1 - G²_i1i1 t_i1 - H²_i1i2 u_i2 + G²_i1i2 t_i2 = 0

    Eq 4 — Region 2 BIE at interface 2:
      H²_i2i1 u_i1 - G²_i2i1 t_i1 + (½-H²_i2i2) u_i2 + G²_i2i2 t_i2 = 0

    Eq 5 — Region 3 BIE at interface 2:
      (½+H³_i2i2) u_i2 - G³_i2i2 t_i2 = 0

    Args:
        sphere_mesh: free surface mesh
        interface1_mesh: outer interface (e.g. base of crust), normals outward
        interface2_mesh: inner interface (e.g. base of asthenosphere), normals outward
        fault_mesh: fault in Region 1
        fault_normal, fault_slip_vector, slip_magnitude: fault parameters
        material_1: ElasticMaterial for Region 1 (outermost)
        material_2: ElasticMaterial for Region 2 (middle)
        material_3: ElasticMaterial for Region 3 (innermost)
        eps: mollification parameter
    """
    N_s  = sphere_mesh.n_triangles
    N_i1 = interface1_mesh.n_triangles
    N_i2 = interface2_mesh.n_triangles
    N_f  = fault_mesh.n_triangles

    n3s  = 3 * N_s
    n3i1 = 3 * N_i1
    n3i2 = 3 * N_i2
    N_unk = n3s + 2 * n3i1 + 2 * n3i2

    if verbose:
        print(f"Surface:    {N_s} elem")
        print(f"Interface1: {N_i1} elem")
        print(f"Interface2: {N_i2} elem")
        print(f"Fault:      {N_f} elem")
        print(f"Unknowns:   {N_unk}")

    delta_u = np.tile(slip_magnitude * fault_slip_vector, (N_f, 1))
    delta_u_flat = delta_u.ravel()

    # ---- Region 1 matrices (outermost material) ----
    if verbose: print("Assembling Region 1 matrices ...")
    H1_ss   = assemble_BEM_matrices(sphere_mesh,     sphere_mesh,     material_1, eps, "T")
    H1_si1  = assemble_BEM_matrices(sphere_mesh,     interface1_mesh, material_1, eps, "T")
    H1_i1s  = assemble_BEM_matrices(interface1_mesh,  sphere_mesh,     material_1, eps, "T")
    H1_i1i1 = assemble_BEM_matrices(interface1_mesh,  interface1_mesh, material_1, eps, "T")
    G1_si1  = assemble_BEM_matrices(sphere_mesh,     interface1_mesh, material_1, eps, "U")
    G1_i1i1 = assemble_BEM_matrices(interface1_mesh,  interface1_mesh, material_1, eps, "U")
    H1_sf   = assemble_BEM_matrices(sphere_mesh,     fault_mesh,      material_1, eps, "T")
    H1_i1f  = assemble_BEM_matrices(interface1_mesh,  fault_mesh,      material_1, eps, "T")

    # ---- Region 2 matrices (middle material) ----
    if verbose: print("Assembling Region 2 matrices ...")
    H2_i1i1 = assemble_BEM_matrices(interface1_mesh, interface1_mesh, material_2, eps, "T")
    H2_i1i2 = assemble_BEM_matrices(interface1_mesh, interface2_mesh, material_2, eps, "T")
    H2_i2i1 = assemble_BEM_matrices(interface2_mesh, interface1_mesh, material_2, eps, "T")
    H2_i2i2 = assemble_BEM_matrices(interface2_mesh, interface2_mesh, material_2, eps, "T")
    G2_i1i1 = assemble_BEM_matrices(interface1_mesh, interface1_mesh, material_2, eps, "U")
    G2_i1i2 = assemble_BEM_matrices(interface1_mesh, interface2_mesh, material_2, eps, "U")
    G2_i2i1 = assemble_BEM_matrices(interface2_mesh, interface1_mesh, material_2, eps, "U")
    G2_i2i2 = assemble_BEM_matrices(interface2_mesh, interface2_mesh, material_2, eps, "U")

    # ---- Region 3 matrices (innermost material) ----
    if verbose: print("Assembling Region 3 matrices ...")
    H3_i2i2 = assemble_BEM_matrices(interface2_mesh, interface2_mesh, material_3, eps, "T")
    G3_i2i2 = assemble_BEM_matrices(interface2_mesh, interface2_mesh, material_3, eps, "U")

    # ---- Assemble global 5-block system ----
    if verbose: print("Assembling global system ...")

    # Block offsets
    s0  = 0;                 s1  = n3s
    i10 = s1;                i11 = s1 + n3i1
    t10 = i11;               t11 = i11 + n3i1
    i20 = t11;               i21 = t11 + n3i2
    t20 = i21;               t21 = i21 + n3i2

    A = np.zeros((N_unk, N_unk))
    b = np.zeros(N_unk)

    # Eq 1: Region 1 BIE at surface
    A[s0:s1, s0:s1]   = H1_ss
    for k in range(n3s):
        A[s0+k, s0+k] += 0.5
    A[s0:s1, i10:i11] = -H1_si1
    A[s0:s1, t10:t11] = G1_si1
    b[s0:s1]           = -H1_sf @ delta_u_flat

    # Eq 2: Region 1 BIE at interface 1
    A[i10:i11, s0:s1]   = H1_i1s
    A[i10:i11, i10:i11] = -H1_i1i1
    for k in range(n3i1):
        A[i10+k, i10+k] += 0.5
    A[i10:i11, t10:t11] = G1_i1i1
    b[i10:i11]           = -H1_i1f @ delta_u_flat

    # Eq 3: Region 2 BIE at interface 1
    A[t10:t11, i10:i11] = H2_i1i1
    for k in range(n3i1):
        A[t10+k, i10+k] += 0.5
    A[t10:t11, t10:t11] = -G2_i1i1
    A[t10:t11, i20:i21] = -H2_i1i2
    A[t10:t11, t20:t21] = G2_i1i2

    # Eq 4: Region 2 BIE at interface 2
    A[i20:i21, i10:i11] = H2_i2i1
    A[i20:i21, t10:t11] = -G2_i2i1
    A[i20:i21, i20:i21] = -H2_i2i2
    for k in range(n3i2):
        A[i20+k, i20+k] += 0.5
    A[i20:i21, t20:t21] = G2_i2i2

    # Eq 5: Region 3 BIE at interface 2
    A[t20:t21, i20:i21] = H3_i2i2
    for k in range(n3i2):
        A[t20+k, i20+k] += 0.5
    A[t20:t21, t20:t21] = -G3_i2i2

    # Solve
    if verbose: print(f"Solving {N_unk}x{N_unk} system ...")
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)

    u_surf  = x[s0:s1].reshape((N_s, 3))
    u_int1  = x[i10:i11].reshape((N_i1, 3))
    t_int1  = x[t10:t11].reshape((N_i1, 3))
    u_int2  = x[i20:i21].reshape((N_i2, 3))
    t_int2  = x[t20:t21].reshape((N_i2, 3))

    if verbose:
        print(f"Max surface displacement:    {np.max(np.linalg.norm(u_surf, axis=1)):.6e} km")
        print(f"Max interface1 displacement: {np.max(np.linalg.norm(u_int1, axis=1)):.6e} km")
        print(f"Max interface2 displacement: {np.max(np.linalg.norm(u_int2, axis=1)):.6e} km")

    return u_surf, u_int1, t_int1, u_int2, t_int2


# ============================================================
# Evaluation at arbitrary interior points
# ============================================================

def evaluate_displacement(eval_points, sphere_mesh, u_surf,
                           fault_mesh, fault_slip, material, eps):
    """Evaluate displacement at arbitrary interior points.

    u(x) = -∫_surf T(x,x') u(x') dS' - ∫_fault T(x,x') Δu(x') dS'

    (since t = 0 on free surface, the U-kernel term vanishes)

    Args:
        eval_points: (N_eval, 3) evaluation coordinates
        sphere_mesh: free surface mesh
        u_surf: (N_surf, 3) surface displacement (solved)
        fault_mesh: fault mesh
        fault_slip: (N_fault, 3) prescribed slip
        material: ElasticMaterial
        eps: mollification parameter

    Returns:
        u_eval: (N_eval, 3) displacement at evaluation points
    """
    # Create a dummy mesh for evaluation points
    # We use the T-kernel from surface/fault to eval points
    N_eval = eval_points.shape[0]
    N_surf = sphere_mesh.n_triangles
    N_fault = fault_mesh.n_triangles

    # We need T-kernel evaluated at eval_points, which requires normals
    # at eval_points. But for interior points, we're just evaluating the
    # representation formula — we need the surface T-kernel (with surface
    # normals), not eval-point normals.

    # T(x, x') uses the normal at x' (the source/boundary point).
    # So we need: for each eval point, sum T(x_eval, x'_j, n_j) * u_j * dA_j

    x_eval = eval_points
    x_surf = sphere_mesh.centroids()
    n_surf, a_surf = sphere_mesh.normals_and_areas()

    mu = material.mu
    nu = material.nu

    u_eval = np.zeros((N_eval, 3))

    # Surface contribution
    for j in range(N_surf):
        r_vec = x_eval - x_surf[j]  # (N_eval, 3)
        T_j = kelvin_T_mollified(r_vec, n_surf[j], mu, nu, eps)  # (N_eval, 3, 3)
        # u contribution: -T_ij * u_j * dA
        for i in range(N_eval):
            u_eval[i] -= T_j[i] @ u_surf[j] * a_surf[j]

    # Fault contribution
    x_fault = fault_mesh.centroids()
    n_fault, a_fault = fault_mesh.normals_and_areas()

    for j in range(N_fault):
        r_vec = x_eval - x_fault[j]
        T_j = kelvin_T_mollified(r_vec, n_fault[j], mu, nu, eps)
        for i in range(N_eval):
            u_eval[i] -= T_j[i] @ fault_slip[j] * a_fault[j]

    return u_eval


# ============================================================
# H-matrix accelerated solvers
# ============================================================

def solve_homogeneous_sphere_hmatrix(sphere_mesh, fault_mesh, fault_normal,
                                      fault_slip_vector, slip_magnitude,
                                      material, eps,
                                      eta=2.0, aca_tol=1e-6, gmres_tol=1e-10,
                                      verbose=True):
    """Homogeneous sphere solver using H-matrix compression + GMRES.

    Same physics as solve_homogeneous_sphere but uses compressed
    matrices for O(N log N) memory and iterative solver.
    """
    from hmatrix import assemble_hmatrix, HMatrix, BlockHMatrixOperator
    from scipy.sparse.linalg import gmres, LinearOperator

    N_surf = sphere_mesh.n_triangles
    N_fault = fault_mesh.n_triangles

    if verbose:
        print(f"Sphere elements: {N_surf}")
        print(f"Fault elements:  {N_fault}")
        print(f"System size:     {3*N_surf} x {3*N_surf} (H-matrix)")

    delta_u = np.tile(slip_magnitude * fault_slip_vector, (N_fault, 1))
    delta_u_flat = delta_u.ravel()

    # Assemble H-matrices
    if verbose: print("Assembling H_ss (H-matrix, T-kernel)...")
    H_ss = assemble_hmatrix(sphere_mesh, sphere_mesh, material, eps, "T",
                            eta=eta, aca_tol=aca_tol, verbose=verbose)

    if verbose: print("Assembling H_fs (H-matrix, T-kernel)...")
    H_fs = assemble_hmatrix(sphere_mesh, fault_mesh, material, eps, "T",
                            eta=eta, aca_tol=aca_tol, verbose=verbose)

    # RHS
    rhs = np.zeros(3 * N_surf)
    # H_fs matvec for RHS
    rhs[:] = -H_fs.matvec(delta_u_flat)

    # System operator: (0.5*I + H_ss) x = rhs
    n = 3 * N_surf

    def mv(v):
        return 0.5 * v + H_ss.matvec(v)

    A_op = LinearOperator((n, n), matvec=mv, dtype=np.float64)

    # GMRES solve
    if verbose: print("Solving with GMRES...")
    iters = [0]

    def callback(rk):
        iters[0] += 1

    u_surf_flat, info = gmres(A_op, rhs, rtol=gmres_tol, restart=200,
                              maxiter=500, callback=callback,
                              callback_type='pr_norm')
    if verbose:
        print(f"GMRES: {iters[0]} iterations, info={info}")

    u_surf = u_surf_flat.reshape((N_surf, 3))

    if verbose:
        u_max = np.max(np.linalg.norm(u_surf, axis=1))
        print(f"Max surface displacement: {u_max:.6e} km")

    return u_surf, None  # no fault displacement computed


def solve_two_region_sphere_hmatrix(sphere_mesh, interface_mesh, fault_mesh,
                                     fault_normal, fault_slip_vector,
                                     slip_magnitude,
                                     material_outer, material_inner, eps,
                                     eta=2.0, aca_tol=1e-6, gmres_tol=1e-10,
                                     verbose=True):
    """Two-region sphere solver using H-matrix compression + GMRES.

    Same physics and sign conventions as solve_two_region_sphere but
    uses H-matrix compressed matrices for reduced memory.
    """
    from hmatrix import assemble_hmatrix, HMatrix, BlockHMatrixOperator
    from scipy.sparse.linalg import gmres

    N_s = sphere_mesh.n_triangles
    N_i = interface_mesh.n_triangles
    N_f = fault_mesh.n_triangles
    N_unk = 3 * N_s + 3 * N_i + 3 * N_i

    if verbose:
        print(f"Surface elements:   {N_s}")
        print(f"Interface elements: {N_i}")
        print(f"Fault elements:     {N_f}")
        print(f"Total unknowns:     {N_unk} (H-matrix)")

    delta_u = np.tile(slip_magnitude * fault_slip_vector, (N_f, 1))
    delta_u_flat = delta_u.ravel()

    # ---- Region 1 matrices (outer material, H-matrix) ----
    if verbose: print("Assembling Region 1 (outer) H-matrices...")
    H1_ss = assemble_hmatrix(sphere_mesh, sphere_mesh, material_outer, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)
    H1_si = assemble_hmatrix(sphere_mesh, interface_mesh, material_outer, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)
    H1_is = assemble_hmatrix(interface_mesh, sphere_mesh, material_outer, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)
    H1_ii = assemble_hmatrix(interface_mesh, interface_mesh, material_outer, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)

    G1_si = assemble_hmatrix(sphere_mesh, interface_mesh, material_outer, eps,
                             "U", eta=eta, aca_tol=aca_tol, verbose=verbose)
    G1_ii = assemble_hmatrix(interface_mesh, interface_mesh, material_outer, eps,
                             "U", eta=eta, aca_tol=aca_tol, verbose=verbose)

    H1_sf = assemble_hmatrix(sphere_mesh, fault_mesh, material_outer, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)
    H1_if = assemble_hmatrix(interface_mesh, fault_mesh, material_outer, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)

    # ---- Region 2 matrices (inner material, H-matrix) ----
    if verbose: print("Assembling Region 2 (inner) H-matrices...")
    H2_ii = assemble_hmatrix(interface_mesh, interface_mesh, material_inner, eps,
                             "T", eta=eta, aca_tol=aca_tol, verbose=verbose)
    G2_ii = assemble_hmatrix(interface_mesh, interface_mesh, material_inner, eps,
                             "U", eta=eta, aca_tol=aca_tol, verbose=verbose)

    # ---- RHS ----
    b = np.zeros(N_unk)
    s0, s1 = 0, 3 * N_s
    i0, i1 = 3 * N_s, 3 * N_s + 3 * N_i
    t0, t1 = 3 * N_s + 3 * N_i, N_unk

    b[s0:s1] = -H1_sf.matvec(delta_u_flat)
    b[i0:i1] = -H1_if.matvec(delta_u_flat)
    # b[t0:t1] = 0 (no sources in Region 2)

    # ---- Block matvec ----
    # Same sign conventions as the dense solver:
    # Row 0 (surf):  (0.5I + H1_ss) u_s - H1_si u_i + G1_si t_i
    # Row 1 (iface): H1_is u_s + (0.5I - H1_ii) u_i + G1_ii t_i
    # Row 2 (iface2): (0.5I + H2_ii) u_i - G2_ii t_i

    def block_matvec(v):
        u_s = v[s0:s1]
        u_i = v[i0:i1]
        t_i = v[t0:t1]
        y = np.zeros(N_unk)

        # Eq 1
        y[s0:s1] = (0.5 * u_s + H1_ss.matvec(u_s)
                     - H1_si.matvec(u_i)
                     + G1_si.matvec(t_i))
        # Eq 2
        y[i0:i1] = (H1_is.matvec(u_s)
                     + 0.5 * u_i - H1_ii.matvec(u_i)
                     + G1_ii.matvec(t_i))
        # Eq 3
        y[t0:t1] = (0.5 * u_i + H2_ii.matvec(u_i)
                     - G2_ii.matvec(t_i))
        return y

    from scipy.sparse.linalg import LinearOperator
    A_op = LinearOperator((N_unk, N_unk), matvec=block_matvec,
                          dtype=np.float64)

    # ---- Block-diagonal preconditioner ----
    # For the sphere block (0.5I + H_ss): the 0.5I diagonal dominance
    # makes it well-conditioned, so use simple 2*I (inverse of 0.5I).
    # For interface blocks: form exact dense preconditioners since they're
    # smaller (3*N_i × 3*N_i) and the material contrast makes them stiff.
    if verbose: print("Building block-diagonal preconditioner...")

    # Threshold for forming exact dense preconditioner
    MAX_DENSE_PRECOND = 8000  # max DOFs for dense factorization

    # Block 0: (0.5I + H1_ss) — use approximate preconditioner if too large
    if 3 * N_s <= MAX_DENSE_PRECOND:
        D0 = H1_ss.to_dense()
        for k in range(3 * N_s):
            D0[k, k] += 0.5
        lu0, piv0 = lu_factor(D0)
        del D0
        precond_block0 = lambda r: lu_solve((lu0, piv0), r)
    else:
        # 0.5I dominates → use 2*I as approximate inverse
        if verbose:
            print(f"  Block 0 ({3*N_s} DOFs): using 2*I preconditioner")
        precond_block0 = lambda r: 2.0 * r

    # Block 1: (0.5I - H1_ii) — exact dense
    D1 = -H1_ii.to_dense()
    for k in range(3 * N_i):
        D1[k, k] += 0.5
    lu1, piv1 = lu_factor(D1)
    del D1

    # Block 2: -G2_ii — exact dense
    D2 = -G2_ii.to_dense()
    lu2, piv2 = lu_factor(D2)
    del D2

    def precond(r):
        """Block-diagonal preconditioner."""
        z = np.zeros(N_unk)
        z[s0:s1] = precond_block0(r[s0:s1])
        z[i0:i1] = lu_solve((lu1, piv1), r[i0:i1])
        z[t0:t1] = lu_solve((lu2, piv2), r[t0:t1])
        return z

    M_op = LinearOperator((N_unk, N_unk), matvec=precond, dtype=np.float64)

    # GMRES solve with preconditioner
    if verbose: print(f"Solving {N_unk}-DOF system with preconditioned GMRES...")
    iters = [0]

    def callback(rk):
        iters[0] += 1
        if verbose and iters[0] % 20 == 0:
            print(f"  GMRES iter {iters[0]}: residual = {rk:.3e}")

    x, info = gmres(A_op, b, rtol=gmres_tol, restart=300,
                    maxiter=2000, callback=callback,
                    callback_type='pr_norm', M=M_op)

    if verbose:
        print(f"GMRES: {iters[0]} iterations, info={info}")

    u_surf = x[s0:s1].reshape((N_s, 3))
    u_int = x[i0:i1].reshape((N_i, 3))
    t_int = x[t0:t1].reshape((N_i, 3))

    if verbose:
        print(f"Max surface displacement:   {np.max(np.linalg.norm(u_surf, axis=1)):.6e} km")
        print(f"Max interface displacement:  {np.max(np.linalg.norm(u_int, axis=1)):.6e} km")
        print(f"Max interface traction:      {np.max(np.linalg.norm(t_int, axis=1)):.6e} GPa")

    return u_surf, u_int, t_int

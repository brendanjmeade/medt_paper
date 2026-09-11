"""Equilateral-triangle mesh generators for the local box model.

Uses Shewchuk's ``triangle`` library to produce Delaunay triangulations
with a minimum-angle quality constraint, giving near-equilateral
triangles. Symmetric under x→−x and y→−y when the input domain is.

The API matches :mod:`local_box_mesh` so existing callers keep working.
"""
from __future__ import annotations

import numpy as np
import triangle as tr

from mollified_bem import TriMesh


__all__ = [
    "triangulate_rectangle",
    "triangulate_top_with_fault_trace",
    "make_rectangular_patch_eq",
    "make_top_patch_with_fault",
    "make_vertical_panel_eq",
    "make_layered_box_eq",
    "make_layered_box_with_fault_graded",
    "make_vertical_fault_eq",
]


def triangulate_rectangle(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    target_edge: float,
    min_angle_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a rectangle with near-equilateral cells.

    Returns ``(vertices_xy, triangles)`` — ``vertices_xy`` is (N, 2),
    ``triangles`` is (M, 3) indexing into ``vertices_xy``.

    ``target_edge`` sets the characteristic edge length. ``min_angle_deg``
    caps the smallest interior angle; 30° is the standard
    equilateral-ish setting (Shewchuk's guarantee).

    The PSLG boundary is sampled at exactly ``target_edge`` spacing so
    mirror partners align, and segment endpoints are placed at the four
    corners to keep the rectangle outline intact.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    if target_edge <= 0:
        raise ValueError("target_edge must be positive")

    def edge_points(a: float, b: float, edge: float) -> np.ndarray:
        n = max(int(round((b - a) / edge)), 1)
        return np.linspace(a, b, n + 1)

    xs = edge_points(x0, x1, target_edge)
    ys = edge_points(y0, y1, target_edge)

    # Boundary vertices, walking CCW: bottom, right, top, left (skip repeats).
    bottom = np.column_stack([xs, np.full(xs.size, y0)])
    right = np.column_stack([np.full(ys.size - 2, x1), ys[1:-1]])
    top = np.column_stack([xs[::-1], np.full(xs.size, y1)])
    left = np.column_stack([np.full(ys.size - 2, x0), ys[-2:0:-1]])
    bverts = np.vstack([bottom, right, top, left])
    n_b = bverts.shape[0]
    segments = np.column_stack([np.arange(n_b), np.roll(np.arange(n_b), -1)])

    # Triangle's area constraint: equilateral triangle with edge e has
    # area = sqrt(3)/4 * e^2 ≈ 0.433 e^2.
    max_area = 0.5 * target_edge * target_edge
    switches = f"pq{min_angle_deg:.0f}a{max_area:.6f}Q"  # p=PSLG, Q=quiet

    pslg = {
        "vertices": bverts,
        "segments": segments,
    }
    mesh = tr.triangulate(pslg, switches)
    return np.asarray(mesh["vertices"]), np.asarray(mesh["triangles"])


def _symmetrize_triangulation(verts_xy: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Force the triangulation to be invariant under x→−x and y→−y.

    Takes the 4-fold symmetric closure of the vertex set: for each
    vertex (x, y), generate (±x, ±y), dedupe within a small tolerance,
    and re-triangulate. This guarantees that mirror partners exist for
    every element, which is necessary for BEM solves of fault problems
    with those symmetries to produce artifact-free vertical motion.
    """
    pts = [verts_xy]
    pts.append(verts_xy * np.array([[-1.0, 1.0]]))
    pts.append(verts_xy * np.array([[1.0, -1.0]]))
    pts.append(verts_xy * np.array([[-1.0, -1.0]]))
    all_pts = np.vstack(pts)

    # Dedupe with a small tolerance
    scale = max(float(np.max(np.abs(all_pts))), 1.0)
    tol = 1e-7 * scale
    # sort lexicographically then collapse neighbours closer than tol
    order = np.lexsort((all_pts[:, 1], all_pts[:, 0]))
    sorted_pts = all_pts[order]
    keep = np.ones(sorted_pts.shape[0], dtype=bool)
    for i in range(1, sorted_pts.shape[0]):
        if np.all(np.abs(sorted_pts[i] - sorted_pts[i - 1]) < tol):
            keep[i] = False
    verts_sym = sorted_pts[keep]

    # Re-triangulate with Delaunay (no quality refinement — just use the node set)
    mesh = tr.triangulate({"vertices": verts_sym}, "Q")
    return np.asarray(mesh["vertices"]), np.asarray(mesh["triangles"])


def triangulate_top_with_fault_trace(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    fault_trace: np.ndarray,
    edge_fault: float,
    edge_far: float,
    near_field_radius: float | None = None,
    edge_near: float | None = None,
    min_angle_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-surface triangulation with an exact fault-trace edge + graded size.

    Parameters
    ----------
    fault_trace : (K, 2) array
        Vertex positions that together span the fault's surface trace. The
        returned triangulation has triangle edges that exactly follow this
        polyline.
    edge_fault : float
        Target spacing between fault-trace vertices (sets the dense
        near-fault element size).
    edge_far : float
        Target spacing along the outer box boundary (sets the coarse
        far-field element size).
    near_field_radius : float | None
        If given, builds a "transition" square region of this half-width
        around the fault, bounded by segments and refined to ``edge_near``
        element size. This gives a two-level graded mesh: fine along the
        fault, medium in the transition region, coarse in the far field.
    edge_near : float | None
        Element size in the transition region. Ignored if
        ``near_field_radius`` is None.

    Notes
    -----
    Grading is achieved by (a) setting dense vertex spacing along the
    fault trace and sparse spacing along the outer boundary — Triangle's
    ``-q`` quality refinement interpolates between them; and (b) optionally
    enclosing a near-field region with segments and applying a regional
    area constraint via the ``-A`` switch. T-junctions between the
    boundary of adjacent PSLG regions are fine for piecewise-constant
    BEM.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    if edge_fault <= 0 or edge_far <= 0:
        raise ValueError("edge_fault and edge_far must be positive")

    def edge_points(a: float, b: float, edge: float) -> np.ndarray:
        n = max(int(round((b - a) / edge)), 1)
        return np.linspace(a, b, n + 1)

    # --- outer boundary, CCW starting at bottom-left corner ---
    xs = edge_points(x0, x1, edge_far)
    ys = edge_points(y0, y1, edge_far)
    b_bot = np.column_stack([xs, np.full(xs.size, y0)])
    b_right = np.column_stack([np.full(ys.size - 2, x1), ys[1:-1]])
    b_top = np.column_stack([xs[::-1], np.full(xs.size, y1)])
    b_left = np.column_stack([np.full(ys.size - 2, x0), ys[-2:0:-1]])
    outer_xy = np.vstack([b_bot, b_right, b_top, b_left])
    n_outer = outer_xy.shape[0]
    outer_seg = np.column_stack([np.arange(n_outer), np.roll(np.arange(n_outer), -1)])

    # --- fault-trace polyline ---
    fault_xy = np.asarray(fault_trace, dtype=float)
    if fault_xy.ndim != 2 or fault_xy.shape[1] != 2:
        raise ValueError("fault_trace must be shape (K, 2)")
    # Resample the polyline at edge_fault spacing so the triangulation
    # naturally seeds dense vertices along the fault.
    seg_lens = np.linalg.norm(np.diff(fault_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum[-1]
    n_ft = max(int(round(total_len / edge_fault)), 2) + 1
    s_new = np.linspace(0, total_len, n_ft)
    ft_xy = np.column_stack([
        np.interp(s_new, cum, fault_xy[:, 0]),
        np.interp(s_new, cum, fault_xy[:, 1]),
    ])
    n_fault = ft_xy.shape[0]
    fault_off = n_outer
    fault_seg = np.column_stack([
        fault_off + np.arange(n_fault - 1),
        fault_off + np.arange(1, n_fault),
    ])

    verts_list = [outer_xy, ft_xy]
    segs_list = [outer_seg, fault_seg]
    holes: list[list[float]] = []
    regions_list: list[list[float]] = []

    # --- optional transition region around fault ---
    if near_field_radius is not None:
        if edge_near is None:
            raise ValueError("edge_near must be given if near_field_radius is set")
        # Build a closed rectangle enclosing the fault trace with the given
        # half-width padding. Segments along its boundary get dense spacing
        # so the "near" region is correctly refined.
        ft_xmin, ft_xmax = ft_xy[:, 0].min(), ft_xy[:, 0].max()
        ft_ymin, ft_ymax = ft_xy[:, 1].min(), ft_xy[:, 1].max()
        R = near_field_radius
        rx0 = max(ft_xmin - R, x0 + edge_far * 0.01)
        rx1 = min(ft_xmax + R, x1 - edge_far * 0.01)
        ry0 = max(ft_ymin - R, y0 + edge_far * 0.01)
        ry1 = min(ft_ymax + R, y1 - edge_far * 0.01)
        xs_in = edge_points(rx0, rx1, edge_near)
        ys_in = edge_points(ry0, ry1, edge_near)
        i_bot = np.column_stack([xs_in, np.full(xs_in.size, ry0)])
        i_right = np.column_stack([np.full(ys_in.size - 2, rx1), ys_in[1:-1]])
        i_top = np.column_stack([xs_in[::-1], np.full(xs_in.size, ry1)])
        i_left = np.column_stack([np.full(ys_in.size - 2, rx0), ys_in[-2:0:-1]])
        inner_xy = np.vstack([i_bot, i_right, i_top, i_left])
        n_inner = inner_xy.shape[0]
        inner_off = sum(v.shape[0] for v in verts_list)
        inner_seg = np.column_stack([
            inner_off + np.arange(n_inner),
            inner_off + np.roll(np.arange(n_inner), -1),
        ])
        verts_list.append(inner_xy)
        segs_list.append(inner_seg)

        # Regions:
        # - "far": any point strictly outside inner rectangle, inside outer.
        # - "near": any point inside inner rectangle, off the fault trace.
        # - attribute 1 → far, attribute 2 → near
        # Use a small x offset to sit clearly off x=0 (the fault).
        near_pt_x = 0.5 * (rx0 + rx1) + 0.5 * edge_near
        near_pt_y = 0.5 * (ry0 + ry1)
        far_pt_x = 0.5 * (x0 + rx0)
        far_pt_y = 0.5 * (y0 + ry0)
        area_near = 0.5 * edge_near * edge_near
        area_far = 0.5 * edge_far * edge_far
        regions_list.append([far_pt_x, far_pt_y, 1.0, area_far])
        regions_list.append([near_pt_x, near_pt_y, 2.0, area_near])

    verts_all = np.vstack(verts_list)
    segs_all = np.vstack(segs_list)

    pslg = {"vertices": verts_all, "segments": segs_all}
    if regions_list:
        pslg["regions"] = np.asarray(regions_list, dtype=float)
        switches = f"pq{min_angle_deg:.0f}AaQ"
    else:
        # No regional constraint — use a single global area based on
        # edge_far; Triangle's -q refinement shrinks triangles near the
        # dense fault vertices automatically.
        area_far = 0.5 * edge_far * edge_far
        switches = f"pq{min_angle_deg:.0f}a{area_far:.6f}Q"

    mesh = tr.triangulate(pslg, switches)
    return np.asarray(mesh["vertices"]), np.asarray(mesh["triangles"])


def make_top_patch_with_fault(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_level: float,
    fault_trace: np.ndarray,
    edge_fault: float,
    edge_far: float,
    near_field_radius: float | None = None,
    edge_near: float | None = None,
    normal_up: bool = True,
    min_angle_deg: float = 30.0,
) -> TriMesh:
    """Top-surface TriMesh that's exactly aligned with the fault trace and
    denser in the near-field than in the far-field."""
    verts_xy, tris = triangulate_top_with_fault_trace(
        x_range, y_range, fault_trace, edge_fault, edge_far,
        near_field_radius, edge_near, min_angle_deg,
    )
    verts = np.column_stack([verts_xy, np.full(verts_xy.shape[0], z_level)])
    mesh = TriMesh(vertices=verts, triangles=tris)
    normals, _ = mesh.normals_and_areas()
    want = np.array([0.0, 0.0, +1.0 if normal_up else -1.0])
    needs_flip = normals @ want < 0
    if needs_flip.any():
        tris = mesh.triangles.copy()
        tris[needs_flip] = tris[needs_flip][:, [0, 2, 1]]
        mesh = TriMesh(vertices=mesh.vertices, triangles=tris)
    return mesh


def make_layered_box_with_fault_graded(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_interfaces: tuple[float, float],
    z_bottom: float,
    fault_trace: np.ndarray,
    edge_fault: float,
    edge_near: float,
    edge_far: float,
    edge_interface_near: float | None = None,
    edge_interface_far: float | None = None,
    edge_side: float = 50.0,
    near_field_radius: float = 200.0,
) -> dict:
    """Layered-box mesh with fault-aligned, graded free-surface mesh.

    The top surface has triangle edges coinciding with ``fault_trace`` and
    a two-level graded mesh (``edge_fault`` / ``edge_near`` / ``edge_far``).
    The upper interface (layer 1 / layer 2 boundary) is also graded but
    doesn't include the fault trace (the fault is above this interface).
    Lower interface, base, and sides use uniform-quality Delaunay meshes.
    """
    if edge_interface_near is None:
        edge_interface_near = 1.2 * edge_near
    if edge_interface_far is None:
        edge_interface_far = 1.2 * edge_far

    z1, z2 = z_interfaces
    if not (0 > z1 > z2 > z_bottom):
        raise ValueError("Expected 0 > z_interfaces[0] > z_interfaces[1] > z_bottom")

    top = make_top_patch_with_fault(
        x_range, y_range, 0.0, fault_trace,
        edge_fault=edge_fault,
        edge_far=edge_far,
        near_field_radius=near_field_radius,
        edge_near=edge_near,
        normal_up=True,
    )

    base = make_rectangular_patch_eq(
        x_range, y_range, z_bottom,
        target_edge=edge_far,
        normal_up=False,
    )

    # Upper interface: graded but no fault-trace segment (fault is in layer 1 only).
    interface1 = make_top_patch_with_fault(
        x_range, y_range, z1,
        fault_trace=fault_trace,  # still include as a refinement hint
        edge_fault=edge_interface_near,
        edge_far=edge_interface_far,
        near_field_radius=near_field_radius,
        edge_near=edge_interface_near,
        normal_up=True,
    )

    interface2 = make_rectangular_patch_eq(
        x_range, y_range, z2,
        target_edge=edge_far,
        normal_up=True,
    )

    layer_bounds = {1: (z1, 0.0), 2: (z2, z1), 3: (z_bottom, z2)}
    sides: dict[int, TriMesh] = {}
    for layer, (z_lo, z_hi) in layer_bounds.items():
        panels = [
            make_vertical_panel_eq("x", x_range[1], y_range, (z_lo, z_hi), edge_side, +1),
            make_vertical_panel_eq("x", x_range[0], y_range, (z_lo, z_hi), edge_side, -1),
            make_vertical_panel_eq("y", y_range[1], x_range, (z_lo, z_hi), edge_side, +1),
            make_vertical_panel_eq("y", y_range[0], x_range, (z_lo, z_hi), edge_side, -1),
        ]
        sides[layer] = _concatenate_meshes(panels)

    return {
        "top": top,
        "base": base,
        "sides": sides,
        "interfaces": {1: interface1, 2: interface2},
    }


def make_rectangular_patch_eq(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_level: float,
    target_edge: float,
    normal_up: bool = True,
    min_angle_deg: float = 30.0,
) -> TriMesh:
    """Near-equilateral triangulation of a flat rectangle at fixed z.

    The Delaunay refinement inside Triangle distributes triangle
    orientations nearly isotropically, so the BEM solution does not pick
    up a preferred-direction mesh bias the way a fixed-diagonal tensor
    grid does. We do NOT force exact 4-fold mirror symmetry on the node
    set here — re-triangulating a mirror-closed node set creates
    sliver triangles (min angle → 1°) and destroys BEM accuracy far
    more than the residual asymmetry costs.
    """
    verts_xy, tris = triangulate_rectangle(x_range, y_range, target_edge, min_angle_deg)
    verts = np.column_stack([verts_xy, np.full(verts_xy.shape[0], z_level)])
    mesh = TriMesh(vertices=verts, triangles=tris)
    normals, _ = mesh.normals_and_areas()
    want = np.array([0.0, 0.0, +1.0 if normal_up else -1.0])
    # Flip any triangles whose computed normal points the wrong way.
    needs_flip = normals @ want < 0
    if needs_flip.any():
        tris = mesh.triangles.copy()
        tris[needs_flip] = tris[needs_flip][:, [0, 2, 1]]
        mesh = TriMesh(vertices=mesh.vertices, triangles=tris)
    return mesh


def make_vertical_panel_eq(
    axis: str,
    fixed_coord: float,
    tangent_range: tuple[float, float],
    z_range: tuple[float, float],
    target_edge: float,
    outward_sign: int,
    min_angle_deg: float = 30.0,
) -> TriMesh:
    """Near-equilateral triangulation of a vertical box-side panel."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    verts_2d, tris = triangulate_rectangle(
        tangent_range, z_range, target_edge, min_angle_deg,
    )
    # No 4-fold symmetry enforcement for a vertical wall (asymmetric in z).

    # Lift 2D (t, z) coords into 3D (x, y, z).
    t_arr = verts_2d[:, 0]
    z_arr = verts_2d[:, 1]
    if axis == "x":
        verts = np.column_stack([np.full(t_arr.size, fixed_coord), t_arr, z_arr])
    else:
        verts = np.column_stack([t_arr, np.full(t_arr.size, fixed_coord), z_arr])

    mesh = TriMesh(vertices=verts, triangles=tris)
    normals, _ = mesh.normals_and_areas()
    want = np.zeros(3)
    want[0 if axis == "x" else 1] = float(outward_sign)
    needs_flip = normals @ want < 0
    if needs_flip.any():
        tris = mesh.triangles.copy()
        tris[needs_flip] = tris[needs_flip][:, [0, 2, 1]]
        mesh = TriMesh(vertices=mesh.vertices, triangles=tris)
    return mesh


def _concatenate_meshes(meshes: list[TriMesh]) -> TriMesh:
    verts_list = []
    tris_list = []
    offset = 0
    for m in meshes:
        verts_list.append(m.vertices)
        tris_list.append(m.triangles + offset)
        offset += m.n_vertices
    return TriMesh(
        vertices=np.vstack(verts_list),
        triangles=np.vstack(tris_list),
    )


def make_layered_box_eq(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_interfaces: tuple[float, float],
    z_bottom: float,
    edge_top: float,
    edge_interface: float,
    edge_side: float,
) -> dict:
    """Equilateral-mesh version of :func:`local_box_mesh.make_layered_box`."""
    z1, z2 = z_interfaces
    if not (0 > z1 > z2 > z_bottom):
        raise ValueError("Expected 0 > z_interfaces[0] > z_interfaces[1] > z_bottom")

    top = make_rectangular_patch_eq(
        x_range, y_range, 0.0, edge_top, normal_up=True,
    )
    base = make_rectangular_patch_eq(
        x_range, y_range, z_bottom, edge_top, normal_up=False,
    )
    interface1 = make_rectangular_patch_eq(
        x_range, y_range, z1, edge_interface, normal_up=True,
    )
    interface2 = make_rectangular_patch_eq(
        x_range, y_range, z2, edge_interface, normal_up=True,
    )

    layer_bounds = {1: (z1, 0.0), 2: (z2, z1), 3: (z_bottom, z2)}
    sides: dict[int, TriMesh] = {}
    for layer, (z_lo, z_hi) in layer_bounds.items():
        panels = [
            make_vertical_panel_eq("x", x_range[1], y_range, (z_lo, z_hi), edge_side, +1),
            make_vertical_panel_eq("x", x_range[0], y_range, (z_lo, z_hi), edge_side, -1),
            make_vertical_panel_eq("y", y_range[1], x_range, (z_lo, z_hi), edge_side, +1),
            make_vertical_panel_eq("y", y_range[0], x_range, (z_lo, z_hi), edge_side, -1),
        ]
        sides[layer] = _concatenate_meshes(panels)

    return {
        "top": top,
        "base": base,
        "sides": sides,
        "interfaces": {1: interface1, 2: interface2},
    }


def make_vertical_fault_eq(
    strike_length: float,
    depth_range: tuple[float, float],
    target_edge: float,
    center_xy: tuple[float, float] = (0.0, 0.0),
    strike_azimuth_deg: float = 0.0,
    top_edge_y_trace: np.ndarray | None = None,
) -> tuple[TriMesh, np.ndarray, np.ndarray]:
    """Equilateral-mesh vertical fault panel.

    If ``top_edge_y_trace`` is given, the top edge (z = z_top, typically 0)
    is constrained to pass through those along-strike coordinates. This
    matches the fault-trace vertices used in a fault-aligned top-surface
    mesh, so every fault-mesh triangle edge at z = 0 coincides with an
    edge of the top-surface mesh.
    """
    z_bot, z_top = depth_range
    if not (z_bot < z_top <= 0):
        raise ValueError("Expected z_bot < z_top ≤ 0 in depth_range")

    s_range = (-strike_length / 2, strike_length / 2)
    if top_edge_y_trace is None:
        verts_2d, tris = triangulate_rectangle(s_range, (z_bot, z_top), target_edge)
    else:
        # Build PSLG where the top edge has exactly the requested x-coords
        # (x-coords in the 2D (s, z) plane here = y-coords along strike).
        s_top = np.asarray(top_edge_y_trace, dtype=float).copy()
        s_top = np.sort(s_top)
        if s_top[0] > s_range[0]:
            s_top = np.concatenate([[s_range[0]], s_top])
        if s_top[-1] < s_range[1]:
            s_top = np.concatenate([s_top, [s_range[1]]])
        # Bottom / side edges use target_edge spacing
        def edge_points(a, b, edge):
            n = max(int(round((b - a) / edge)), 1)
            return np.linspace(a, b, n + 1)
        s_bot = edge_points(s_range[0], s_range[1], target_edge)
        z_left = edge_points(z_bot, z_top, target_edge)
        z_right = z_left.copy()
        # Walk CCW: bottom (L→R), right (bot→top), top (R→L), left (top→bot)
        b_bot = np.column_stack([s_bot, np.full(s_bot.size, z_bot)])
        b_right = np.column_stack([np.full(z_right.size - 2, s_range[1]), z_right[1:-1]])
        b_top = np.column_stack([s_top[::-1], np.full(s_top.size, z_top)])
        b_left = np.column_stack([np.full(z_left.size - 2, s_range[0]), z_left[-2:0:-1]])
        verts = np.vstack([b_bot, b_right, b_top, b_left])
        n = verts.shape[0]
        seg = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])
        area = 0.5 * target_edge * target_edge
        mesh = tr.triangulate({"vertices": verts, "segments": seg}, f"pq30a{area:.6f}Q")
        verts_2d, tris = np.asarray(mesh["vertices"]), np.asarray(mesh["triangles"])

    theta = np.radians(strike_azimuth_deg)
    s_hat = np.array([np.sin(theta), np.cos(theta), 0.0])
    n_hat = np.array([np.cos(theta), -np.sin(theta), 0.0])

    s_arr = verts_2d[:, 0]
    z_arr = verts_2d[:, 1]
    verts = np.column_stack([
        center_xy[0] + s_arr * s_hat[0],
        center_xy[1] + s_arr * s_hat[1],
        z_arr,
    ])

    mesh = TriMesh(vertices=verts, triangles=tris)
    normals, _ = mesh.normals_and_areas()
    needs_flip = normals @ n_hat < 0
    if needs_flip.any():
        tris = mesh.triangles.copy()
        tris[needs_flip] = tris[needs_flip][:, [0, 2, 1]]
        mesh = TriMesh(vertices=mesh.vertices, triangles=tris)

    return mesh, n_hat, s_hat

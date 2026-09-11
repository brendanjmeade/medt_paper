"""Mesh utilities for the heterogeneous half-space showcase.

Builds a conforming pair of TriMesh boundaries for a host + cylindrical
inclusion BEM problem:

* Host outer boundary: free top with a circular hole at the inclusion's
  surface footprint, four vertical free sides, clamped base, and the
  *interface* (cylinder side wall + bottom disk of the inclusion),
  oriented with normals as stored on the inclusion side. The host's
  collocation rows reference the interface with sign-flipped normals.
* Inclusion outer boundary: free disk top filling the host's hole, plus
  the same interface (cylinder side + bottom disk).

The fault (a vertical strike-slip rectangle in the host region) is
provided unchanged from `local_box_mesh_eq.make_vertical_fault_eq`.

This module returns a dict of TriMeshes plus an integer mapping from
surface-name to an outward-normal vector (per surface), so the BEM
solver can flip signs cleanly.
"""

from __future__ import annotations

import numpy as np
import triangle as tr

from local_box_mesh_eq import (
    make_rectangular_patch_eq,
    make_vertical_panel_eq,
    _concatenate_meshes,
)
from mollified_bem import TriMesh


def _circle_boundary(center_xy, R, target_edge):
    """N evenly spaced points on a circle of radius R.

    Returns (N, 2) array of points and the integer N.
    """
    cx, cy = center_xy
    circumference = 2.0 * np.pi * R
    N = max(int(round(circumference / target_edge)), 12)
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    pts = np.column_stack([cx + R * np.cos(theta), cy + R * np.sin(theta)])
    return pts, N


def _rectangle_boundary(x_range, y_range, target_edge):
    """Discretized rectangular boundary, returned in CCW order so the
    interior of the rectangle lies to the left as we walk it.
    """
    x0, x1 = x_range
    y0, y1 = y_range

    def along(a, b, e):
        n = max(int(round((b - a) / e)), 1)
        return np.linspace(a, b, n + 1)

    xs = along(x0, x1, target_edge)
    ys = along(y0, y1, target_edge)
    bot = np.column_stack([xs, np.full(xs.size, y0)])
    right = np.column_stack([np.full(ys.size - 2, x1), ys[1:-1]])
    top = np.column_stack([xs[::-1], np.full(xs.size, y1)])
    left = np.column_stack([np.full(ys.size - 2, x0), ys[-2:0:-1]])
    return np.vstack([bot, right, top, left])


def _disk_with_boundary(circle_pts, target_edge, z_level, normal_up=True):
    """Triangulate a disk whose boundary is exactly `circle_pts` (in CCW
    order), at given z_level. Returns a TriMesh with normals as requested.
    """
    n = circle_pts.shape[0]
    seg = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])
    area = 0.5 * target_edge ** 2
    mesh = tr.triangulate(
        {"vertices": circle_pts, "segments": seg},
        f"pq30a{area:.6f}Q",
    )
    verts2 = np.asarray(mesh["vertices"])
    tris = np.asarray(mesh["triangles"])
    verts = np.column_stack([verts2, np.full(verts2.shape[0], z_level)])
    tm = TriMesh(vertices=verts, triangles=tris)
    n_vec, _ = tm.normals_and_areas()
    want = np.array([0.0, 0.0, 1.0 if normal_up else -1.0])
    flip = n_vec @ want < 0
    if flip.any():
        t = tm.triangles.copy()
        t[flip] = t[flip][:, [0, 2, 1]]
        tm = TriMesh(vertices=tm.vertices, triangles=t)
    return tm


def _annulus_top(rect_outer, circle_inner_pts, hole_center, target_edge,
                 max_area=None, fault_trace=None, fault_edge=None,
                 refine_disks=None):
    """Triangulate (rectangle minus disk) at z = 0, normals +z.

    If `max_area` is None, no uniform area constraint is applied — the
    mesh grades naturally from the (fine) inner-circle boundary to the
    (coarse) outer-rectangle boundary based on each PSLG's segment
    density. Pass a number to force a uniform-or-finer maximum.

    If `fault_trace` is given as an (M, 2) polyline, the polyline is
    inserted as an additional internal segment so that triangle edges
    follow it exactly. `fault_edge` (km) sets the dense node spacing
    along the fault.

    `refine_disks` is an optional list of ((cx, cy), radius, ring_edge,
    max_area_inside): each entry adds a circular ring of PSLG segments
    plus a Triangle `regions` row, locally capping triangle areas inside
    the disk (used e.g. to resolve a topographic bump). With both a
    numeric global area cap and regional caps, Triangle applies the
    smaller per triangle.
    """
    seg_o = np.column_stack([np.arange(rect_outer.shape[0]),
                             np.roll(np.arange(rect_outer.shape[0]), -1)])
    n_o = rect_outer.shape[0]
    seg_i = np.column_stack([np.arange(circle_inner_pts.shape[0]),
                             np.roll(np.arange(circle_inner_pts.shape[0]), -1)]) + n_o
    verts_list = [rect_outer, circle_inner_pts]
    seg_list = [seg_o, seg_i]
    n_vert_running = n_o + circle_inner_pts.shape[0]

    if fault_trace is not None:
        ft = np.asarray(fault_trace, dtype=float)
        if fault_edge is None:
            fault_edge = target_edge
        seg_lens = np.linalg.norm(np.diff(ft, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum[-1]
        n_ft = max(int(round(total_len / fault_edge)), 2) + 1
        s_new = np.linspace(0, total_len, n_ft)
        ft_xy = np.column_stack([
            np.interp(s_new, cum, ft[:, 0]),
            np.interp(s_new, cum, ft[:, 1]),
        ])
        seg_f = np.column_stack([
            n_vert_running + np.arange(n_ft - 1),
            n_vert_running + np.arange(1, n_ft),
        ])
        verts_list.append(ft_xy)
        seg_list.append(seg_f)
        n_vert_running += n_ft

    regions = []
    if refine_disks:
        for (dcx, dcy), rad, ring_edge, a_in in refine_disks:
            ring_pts, n_ring = _circle_boundary((dcx, dcy), rad, ring_edge)
            seg_r = np.column_stack([
                np.arange(n_ring),
                np.roll(np.arange(n_ring), -1),
            ]) + n_vert_running
            verts_list.append(ring_pts)
            seg_list.append(seg_r)
            n_vert_running += n_ring
            regions.append([dcx, dcy, 1.0, a_in])

    verts = np.vstack(verts_list)
    seg = np.vstack(seg_list)
    flags = "pq30" if max_area is None else f"pq30a{max_area:.6f}"
    if regions:
        flags += "a"   # bare 'a': apply per-region max areas from 'regions';
                       # combined with a numeric 'a', the smaller cap wins
    flags += "Q"
    pslg = {"vertices": verts, "segments": seg, "holes": [list(hole_center)]}
    if regions:
        pslg["regions"] = np.asarray(regions, dtype=float)
    mesh = tr.triangulate(pslg, flags)
    verts2 = np.asarray(mesh["vertices"])
    tris = np.asarray(mesh["triangles"])
    verts3 = np.column_stack([verts2, np.zeros(verts2.shape[0])])
    tm = TriMesh(vertices=verts3, triangles=tris)
    n_vec, _ = tm.normals_and_areas()
    flip = n_vec @ np.array([0.0, 0.0, 1.0]) < 0
    if flip.any():
        t = tm.triangles.copy()
        t[flip] = t[flip][:, [0, 2, 1]]
        tm = TriMesh(vertices=tm.vertices, triangles=t)
    return tm


def _cylinder_side(circle_pts, z_top, z_bot, m_depth, center_xy):
    """Vertical cylinder side wall as a structured triangulation; outward
    normals point radially outward from `center_xy`.

    Layout: M+1 z-levels × N nodes/level. For each (i, k) we lay down two
    triangles spanning the quad ((i,k), (i+1,k), (i+1,k+1), (i,k+1)).
    """
    N = circle_pts.shape[0]
    z_levels = np.linspace(z_top, z_bot, m_depth + 1)
    verts = np.zeros(((m_depth + 1) * N, 3))
    for k, z in enumerate(z_levels):
        verts[k * N:(k + 1) * N, 0:2] = circle_pts
        verts[k * N:(k + 1) * N, 2] = z

    tris = []
    for k in range(m_depth):
        for i in range(N):
            a = k * N + i
            b = k * N + (i + 1) % N
            c = (k + 1) * N + (i + 1) % N
            d = (k + 1) * N + i
            # (a, b, c) and (a, c, d), checked for outward normal below.
            tris.append([a, b, c])
            tris.append([a, c, d])
    tris = np.asarray(tris, dtype=np.int64)

    tm = TriMesh(vertices=verts, triangles=tris)
    # Outward normal is radial; check the first triangle's normal vs the
    # radial vector at its centroid and flip uniformly if needed.
    n_vec, _ = tm.normals_and_areas()
    centroids = verts[tris].mean(axis=1)
    cx, cy = center_xy
    radial = centroids[:, 0:2] - np.array([[cx, cy]])
    radial_len = np.linalg.norm(radial, axis=1)
    radial_hat = np.column_stack([
        radial[:, 0] / np.maximum(radial_len, 1e-30),
        radial[:, 1] / np.maximum(radial_len, 1e-30),
        np.zeros_like(radial_len),
    ])
    # Triangles whose normal points "inward" (negative dot with radial)
    # need flipping.
    dot = (n_vec * radial_hat).sum(axis=1)
    flip = dot < 0
    if flip.any():
        t = tm.triangles.copy()
        t[flip] = t[flip][:, [0, 2, 1]]
        tm = TriMesh(vertices=tm.vertices, triangles=t)
    return tm


def make_inclusion_geometry(
    x_range=(-500.0, 500.0),
    y_range=(-500.0, 500.0),
    z_bottom=-200.0,
    inclusion_center_xy=(-100.0, 0.0),
    inclusion_radius=75.0,
    inclusion_depth=10.0,
    target_edge_inclusion=20.0,
    target_edge_top=40.0,
    target_edge_far=60.0,
    target_edge_side=60.0,
    fault_trace=None,
    fault_edge=None,
    host_top_max_edge=None,
    top_refine_disks=None,
):
    """Build the full set of meshes for a host (rectangular box with a
    cylindrical inclusion footprint) + cylindrical inclusion problem.

    Returns a dict with keys
        host_top, host_sides, host_base
        inclusion_top, interface_side, interface_bot

    and a per-key outward normal direction the mesh was *constructed* to
    have. Interface meshes have normals pointing *outward from the
    inclusion* (i.e. from the inclusion into the host).
    """
    cx, cy = inclusion_center_xy
    R = inclusion_radius
    D = inclusion_depth

    # Boundary nodes of the inclusion top circle — shared between
    # host_top (as the inner boundary of the rectangular hole) and
    # inclusion_top (as the disk's outer boundary). The same set is also
    # the top edge of interface_side.
    circle_top, _N_circ = _circle_boundary((cx, cy), R, target_edge_inclusion)

    # The bottom-disk boundary is the same circle in (x, y) but at z=-D.
    # Since interface_side uses a structured uniform triangulation around
    # the circle, the bottom-disk PSLG must use the same N_circ points
    # in the same order.

    # Inclusion top free disk (z = 0, normals +z).
    inclusion_top = _disk_with_boundary(
        circle_top, target_edge_inclusion, z_level=0.0, normal_up=True,
    )

    # Host top with circular hole (z = 0, normals +z). The outer
    # rectangle is segmented at the coarser `target_edge_top` while the
    # inner circle has already been segmented at the finer
    # `target_edge_inclusion`. With no uniform area constraint the
    # interior triangles grade naturally from fine-near-the-disk to
    # coarse-near-the-box, ensuring the host triangles immediately
    # outside the disk match the inclusion triangles immediately inside
    # — which the BEM formulation requires for the (fictitious-at-equal-
    # materials) interface to behave correctly.
    rect_outer = _rectangle_boundary(x_range, y_range, target_edge_top)
    # Reverse the inner circle to be CW so the hole is correctly oriented
    # (triangle library wants outer CCW + holes punched by hole points).
    host_top_max_area = (
        None if host_top_max_edge is None
        else 0.5 * host_top_max_edge ** 2
    )
    host_top = _annulus_top(rect_outer, circle_top[::-1].copy(),
                            hole_center=(cx, cy),
                            target_edge=target_edge_inclusion,
                            max_area=host_top_max_area,
                            fault_trace=fault_trace,
                            fault_edge=fault_edge,
                            refine_disks=top_refine_disks)

    # Cylindrical interface side (radial outward normals). We require
    # at least two depth layers regardless of D/edge ratio so the
    # cylinder is not represented as a single ring of triangles --
    # a single ring leaves the bottom edge with very-different-density
    # neighbours which destabilises the BEM.
    m_depth = max(int(round(D / target_edge_inclusion)), 2)
    interface_side = _cylinder_side(
        circle_top, z_top=0.0, z_bot=-D, m_depth=m_depth,
        center_xy=(cx, cy),
    )

    # Inclusion bottom disk at z = -D, normals -z (outward FROM inclusion
    # is downward, since the inclusion sits above z = -D).
    interface_bot = _disk_with_boundary(
        circle_top, target_edge_inclusion, z_level=-D, normal_up=False,
    )

    # Host base at z = z_bottom, normals -z (downward, outward from host).
    host_base = make_rectangular_patch_eq(
        x_range, y_range, z_bottom, target_edge_far, normal_up=False,
    )

    # Host sides (4 vertical panels), normals outward.
    panels = [
        make_vertical_panel_eq("x", x_range[1], y_range, (z_bottom, 0.0),
                               target_edge_side, +1),
        make_vertical_panel_eq("x", x_range[0], y_range, (z_bottom, 0.0),
                               target_edge_side, -1),
        make_vertical_panel_eq("y", y_range[1], x_range, (z_bottom, 0.0),
                               target_edge_side, +1),
        make_vertical_panel_eq("y", y_range[0], x_range, (z_bottom, 0.0),
                               target_edge_side, -1),
    ]
    host_sides = _concatenate_meshes(panels)

    return {
        "host_top": host_top,
        "host_sides": host_sides,
        "host_base": host_base,
        "inclusion_top": inclusion_top,
        "interface_side": interface_side,
        "interface_bot": interface_bot,
    }


if __name__ == "__main__":
    meshes = make_inclusion_geometry()
    for k, m in meshes.items():
        print(f"  {k:18s}: {m.n_triangles:6d} tri, {m.vertices.shape[0]:6d} v")

"""3D visualization utilities for graphs with spatial embeddings.

This module provides functions for rendering NetworkX graphs in 3D space,
including support for face polygons (triangles and quads), depth-based layering,
and customizable node/edge styling.
"""

import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from typing import Dict, List, Tuple, Optional, Any
from graph2grav import technical_helpers as th


def fit_plane_and_order_polygon(points, tol_plane=5e-2):
    """Best-fit plane to k points and order them cyclically.

    Fits a plane to 3 or 4 points using SVD, validates planarity within tolerance,
    and orders the points cyclically by angle in the plane's coordinate system.

    Args:
        points: Array-like of shape (k, 3) where k is typically 3 or 4.
        tol_plane: Maximum perpendicular distance from fitted plane (default 5e-2).
                   Looser tolerance handles depth-induced z-offsets.

    Returns:
        Tuple of (ok, ordered_pts) where:
            - ok (bool): True if points are sufficiently planar
            - ordered_pts (np.ndarray or None): Points ordered cyclically in the plane,
              or None if planarity check fails
    """
    P = np.asarray(points, float)
    ctr = P.mean(axis=0)
    Q = P - ctr
    # principal components
    _, S, Vt = np.linalg.svd(Q, full_matrices=False)
    nrm = Vt[-1]
    max_dist = np.max(np.abs(Q @ nrm))
    if not np.isfinite(max_dist) or max_dist > tol_plane:
        return False, None
    u = Vt[0]
    v = np.cross(nrm, u)
    v /= np.linalg.norm(v)
    uv = np.stack([Q @ u, Q @ v], axis=1)
    ang = np.arctan2(uv[:, 1], uv[:, 0])
    order = np.argsort(ang)
    return True, P[order]


def polygons_for_faces(G, pos3d, include_quads=True, tol_plane=5e-2):
    """Extract face polygons from graph triangles and chordless 4-cycles.

    Returns list of ordered polygon vertices for rendering graph faces in 3D.
    Uses planarity checking to filter out non-planar cycles.

    Args:
        G: NetworkX graph
        pos3d: Dictionary mapping node -> 3D position np.array([x, y, z])
        include_quads: Whether to include chordless 4-cycles in addition to triangles
        tol_plane: Planarity tolerance passed to fit_plane_and_order_polygon

    Returns:
        List of np.array(k, 3) polygons, where k is 3 for triangles or 4 for quads
    """
    polys = []
    # triangles
    for tri in th.triangles_nodes(G):
        pts = [pos3d[n] for n in tri]
        ok, poly = fit_plane_and_order_polygon(pts, tol_plane)
        if ok:
            polys.append(poly)
    # 4-cycles (chordless only)
    if include_quads:
        for cyc4 in th.chordless_4cycles_nodes(G):
            pts = [pos3d[n] for n in cyc4]
            ok, poly = fit_plane_and_order_polygon(pts, tol_plane)
            if ok:
                polys.append(poly)
    return polys


def get_depth_from_n(n, graph):
    """Helper to get the depth of a node from graph attributes.

    Args:
        n: Node identifier
        graph: NetworkX graph with 'depth' node attribute

    Returns:
        Depth value from graph.nodes[n]["depth"]
    """
    return graph.nodes[n]["depth"]


def draw_graph_3d(
    ax,
    G,
    pos3d,
    inner_colors,
    outer_colors,
    labels,
    node_size_outer=10,
    node_size_inner=5,
    edge_width=1,
    reverse=False,
    show_faces=True,
    face_alpha=0.25,
    face_color="#7aa6ff",
    tol_plane=1e-2,
    use_wormdepth=True,
    zshift=0,
):
    """Draw a 3D graph with depth-based layering and optional face polygons.

    Renders nodes with two-color styling (inner/outer circles), edges, and
    face polygons. Uses depth information for z-ordering to create proper
    visual layering.

    Args:
        ax: matplotlib Axes3D object
        G: NetworkX graph
        pos3d: Dict mapping node -> 3D position array [x, y, z]
        inner_colors: List/array of colors for inner node circles (same order as G.nodes())
        outer_colors: List/array of colors for outer node circles (same order as G.nodes())
        labels: Dict mapping node -> label string, or None for no labels
        node_size_outer: Size of outer node circle (default 80)
        node_size_inner: Size of inner node circle (default 40)
        edge_width: Line width for edges (default 1.5)
        reverse: If True, reverse depth ordering for layering (default False)
        show_faces: Whether to render face polygons (default True)
        face_alpha: Transparency of face polygons (default 0.25)
        face_color: Color of face polygons (default "#7aa6ff")
        tol_plane: Planarity tolerance for face detection (default 1e-2)
        use_wormdepth: If True, use "wormdepth" attribute; else use "depth" (default True)
        zshift: Offset for z-order values (default 0)

    Notes:
        - Nodes should have either 'wormdepth' or 'depth' attribute
        - Uses depth-based z-ordering: deeper nodes rendered on top or bottom based on 'reverse'
        - Faces are extracted from triangles and chordless 4-cycles
    """
    try:
        if use_wormdepth:
            depth_of = {n: G.nodes[n]["wormdepth"] for n in G.nodes()}
        else:
            depth_of = {n: get_depth_from_n(n, G) for n in G.nodes()}
    except Exception:
        depth_of = {n: 0 for n in G.nodes()}

    depths_sorted = sorted(set(depth_of.values()))
    order = list(reversed(depths_sorted)) if reverse else depths_sorted

    base, step = 10, 3
    zorder_of_depth = {d: base + step * i + zshift for i, d in enumerate(order)}
    zorder_of_edge = {d: zorder_of_depth[d] - 1 + zshift for d in depths_sorted}
    zorder_of_face = {d: zorder_of_depth[d] - 0.25 + zshift for d in depths_sorted}

    inner_color_of = {n: c for n, c in zip(G.nodes(), inner_colors)}
    outer_color_of = {n: c for n, c in zip(G.nodes(), outer_colors)}

    # -------- faces (triangles + chordless quads, unique) --------
    if show_faces:
        polys = polygons_for_faces(G, pos3d, include_quads=True, tol_plane=5e-2)
        # layer by min node depth in the polygon
        try:
            if use_wormdepth:
                depth_of = {n: G.nodes[n]["wormdepth"] for n in G.nodes()}
            else:
                depth_of = {n: get_depth_from_n(n, G) for n in G.nodes()}
        except Exception:
            depth_of = {n: 0 for n in G.nodes()}
        zdepth = {tuple(pos3d[n]): depth_of[n] for n in G.nodes()}  # exact mapping
        faces_by_depth = {d: [] for d in depths_sorted}
        for poly in polys:
            d_face = min(zdepth[tuple(p)] for p in poly)
            faces_by_depth[d_face].append(poly)

        for d in depths_sorted:
            if faces_by_depth[d]:
                pc = Poly3DCollection(
                    faces_by_depth[d],
                    facecolors=face_color,
                    edgecolors="none",
                    alpha=face_alpha,
                    zorder=zorder_of_face[d],
                )
                ax.add_collection3d(pc)

    # -------- edges --------
    edges_by_depth = {d: [] for d in depths_sorted}
    for u, v in G.edges():
        de = round((depth_of[u] + depth_of[v]) / 2)
        edges_by_depth[de].append([pos3d[u], pos3d[v]])
    for d in depths_sorted:
        segs = edges_by_depth[d]
        if segs:
            lc = Line3DCollection(
                np.asarray(segs),
                linewidths=edge_width,
                color="black",
                alpha=0.5,
                zorder=zorder_of_edge[d],
            )
            ax.add_collection3d(lc)

    # -------- nodes (depthshade off for crisp layering) --------
    for d in depths_sorted:
        nodes_d = [n for n in G.nodes() if depth_of[n] == d]
        if not nodes_d:
            continue
        xyz = np.array([pos3d[n] for n in nodes_d])
        outer_cols = [outer_color_of[n] for n in nodes_d]
        inner_cols = [inner_color_of[n] for n in nodes_d]
        z_nd = zorder_of_depth[d]
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=node_size_outer,
            c=outer_cols,
            depthshade=False,
            alpha=0.8,
            zorder=z_nd,
        )
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=node_size_inner,
            c=inner_cols,
            depthshade=False,
            alpha=0.8,
            zorder=z_nd + 0.5,
        )

    if labels is not None:
        for n, (x, y, z) in pos3d.items():
            text = labels.get(n, str(n)) if isinstance(labels, dict) else str(n)
            if text:
                ax.text(
                    x,
                    y,
                    z,
                    text,
                    color="black",
                    zorder=zorder_of_depth[depth_of[n]] + 1,
                )

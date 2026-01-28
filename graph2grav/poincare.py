"""Poincaré disk hyperbolic geometry utilities.

This module provides functions for working with the Poincaré disk model of
hyperbolic geometry, including:
- Geodesic computations (arcs, lengths, extensions)
- Hyperbolic distance calculations
- Reflections across geodesics
- Tessellation generation and drawing
- Ryu-Takayanagi surface visualization

The Poincaré disk represents the hyperbolic plane as the interior of the
unit disk, where geodesics are circular arcs perpendicular to the boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


# =============================================================================
# Core Geodesic Computations
# =============================================================================

def get_geodesic_circle(
    z1: complex,
    z2: complex,
    eps: float = 1e-12
) -> tuple[complex | None, float | None]:
    """Get the center and radius of the geodesic arc through z1 and z2.

    In the Poincaré disk, geodesics are either:
    1. Circular arcs orthogonal to the boundary (|center|² = radius² + 1)
    2. Diameters through the origin (when z1 and z2 are collinear with origin)

    Args:
        z1, z2: Complex numbers representing points in the Poincaré disk
        eps: Tolerance for detecting the diameter case

    Returns:
        (center, radius) of the geodesic circle, or (None, None) for diameters
    """
    # det = cross product, measures collinearity with origin
    det = z1.real * z2.imag - z1.imag * z2.real

    # Check for collinearity with origin (diameter case)
    if abs(det) < eps:
        return None, None

    # For a circle orthogonal to the unit circle: |c|² = R² + 1
    # Combined with |z - c|² = R², we get: Re(z * conj(c)) = (|z|² + 1) / 2
    # This gives two linear equations in (cx, cy) from z1 and z2.
    b1 = (abs(z1)**2 + 1.0) / 2.0
    b2 = (abs(z2)**2 + 1.0) / 2.0

    cx = (b1 * z2.imag - b2 * z1.imag) / det
    cy = (z1.real * b2 - z2.real * b1) / det

    c = complex(cx, cy)
    R2 = abs(c)**2 - 1.0

    # Check for invalid circle (radius too small implies numerical error)
    if R2 <= 0:
        return None, None

    R = np.sqrt(R2)

    # If radius is huge, treat as diameter to avoid float precision loss.
    # A circle with R > 1000 in the unit disk is visually indistinguishable from a line.
    if R > 1000.0:
        return None, None

    return c, R


def geodesic_arc_points(
    z1: complex,
    z2: complex,
    n_points: int = 100,
    eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Compute points along the geodesic arc from z1 to z2 in the Poincaré disk.

    Returns the arc that stays inside the unit disk (the shorter hyperbolic path).

    Args:
        z1, z2: Complex numbers representing endpoints in the Poincaré disk
        n_points: Number of points to sample along the arc
        eps: Tolerance for detecting the diameter case

    Returns:
        (xs, ys): Arrays of x and y coordinates along the geodesic
    """
    center, radius = get_geodesic_circle(z1, z2, eps=eps)

    if center is None:
        # Diameter case - straight line through origin
        xs = np.linspace(z1.real, z2.real, n_points)
        ys = np.linspace(z1.imag, z2.imag, n_points)
        return xs, ys

    # Circular arc case
    a1 = np.angle(z1 - center)
    a2 = np.angle(z2 - center)

    # Normalize angle difference to (-π, π]
    d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi

    # Alternative arc is the long way around
    d_alt = d - 2 * np.pi * np.sign(d) if abs(d) > eps else 2 * np.pi

    def midpoint_for(delta):
        am = a1 + delta / 2
        return center + radius * np.exp(1j * am)

    m1 = midpoint_for(d)
    m2 = midpoint_for(d_alt)

    # Pick arc whose midpoint is inside disk (with small margin)
    delta = d if abs(m1) < 1 - 1e-10 else d_alt

    ts = np.linspace(0, 1, n_points)
    angles = a1 + delta * ts
    pts = center + radius * np.exp(1j * angles)

    return pts.real, pts.imag


def geodesic_arc_points_boundary(
    theta1: float,
    theta2: float,
    n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Compute geodesic arc points connecting two angles on the boundary.

    Convenience wrapper for geodesic_arc_points when endpoints are on the
    unit circle (useful for Ryu-Takayanagi surfaces).

    Args:
        theta1, theta2: Angles (in radians) of boundary points
        n_points: Number of points to sample

    Returns:
        (xs, ys): Arrays of coordinates along the geodesic
    """
    # Normalize angles to [0, 2π)
    theta1 = theta1 % (2 * np.pi)
    theta2 = theta2 % (2 * np.pi)

    # Boundary points
    P1 = np.exp(1j * theta1)
    P2 = np.exp(1j * theta2)

    # Angular difference (choose representation in [-π, π])
    diff = theta2 - theta1
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    # Check if geodesic is approximately a diameter (antipodal points)
    if np.abs(np.abs(diff) - np.pi) < 1e-10:
        return np.array([P1.real, P2.real]), np.array([P1.imag, P2.imag])

    # Midpoint angle and half-angle
    theta_m = theta1 + diff / 2
    delta = diff / 2

    # Center of geodesic circle (outside the unit disk)
    center_dist = 1.0 / np.cos(delta)
    C = center_dist * np.exp(1j * theta_m)
    r = np.abs(np.tan(delta))

    # Angles of P1 and P2 relative to center C
    alpha1 = np.arctan2(P1.imag - C.imag, P1.real - C.real)
    alpha2 = np.arctan2(P2.imag - C.imag, P2.real - C.real)

    # Determine arc direction to stay inside the disk
    d_alpha = alpha2 - alpha1
    if d_alpha > np.pi:
        d_alpha -= 2 * np.pi
    elif d_alpha < -np.pi:
        d_alpha += 2 * np.pi

    angles = np.linspace(alpha1, alpha1 + d_alpha, n_points)
    arc_x = C.real + r * np.cos(angles)
    arc_y = C.imag + r * np.sin(angles)

    return arc_x, arc_y


def extend_geodesic_to_boundary(
    z1: complex,
    z2: complex,
    eps: float = 1e-10
) -> complex:
    """Find where the geodesic through z1 and z2 intersects the boundary.

    Returns the intersection point on the unit circle on the side of z2
    (i.e., extending from z1 through z2 to the boundary).

    Args:
        z1, z2: Points in the Poincaré disk
        eps: Tolerance for numerical comparisons

    Returns:
        Complex number on the unit circle (|result| = 1)
    """
    x1, y1 = z1.real, z1.imag
    x2, y2 = z2.real, z2.imag

    # Check if nearly collinear with origin (geodesic is a diameter)
    cross = x1 * y2 - x2 * y1
    if abs(cross) < eps:
        direction = z2 - z1
        if abs(direction) < eps:
            return z2 / abs(z2) if abs(z2) > 0 else complex(1, 0)
        direction = direction / abs(direction)
        # Return point on boundary in direction from z1 toward z2
        sign = 1 if (z2 - z1).real * direction.real + (z2 - z1).imag * direction.imag > 0 else -1
        return sign * direction

    # General case: find the circle perpendicular to unit circle passing through z1, z2
    center, radius = get_geodesic_circle(z1, z2, eps=eps)

    if center is None:
        direction = z2 - z1
        if abs(direction) < eps:
            return z2 / abs(z2) if abs(z2) > 0 else complex(1, 0)
        direction = direction / abs(direction)
        return direction

    d = abs(center)
    if d < eps:
        return z2 / abs(z2) if abs(z2) > 0 else complex(1, 0)

    # Find intersections of geodesic circle with unit circle
    # Distance from origin to the chord connecting intersections
    h = (d*d + 1 - radius*radius) / (2 * d)

    if abs(h) > 1:
        return z2 / abs(z2) if abs(z2) > 0 else complex(1, 0)

    half_chord = np.sqrt(max(0, 1 - h*h))
    to_center = center / d
    closest = h * to_center
    perp = complex(-to_center.imag, to_center.real)

    int1 = closest + half_chord * perp
    int2 = closest - half_chord * perp

    # Return the intersection closer to z2
    return int1 if abs(int1 - z2) < abs(int2 - z2) else int2


# =============================================================================
# Hyperbolic Distance
# =============================================================================

def hyperbolic_distance(z1: complex, z2: complex) -> float:
    """Compute the hyperbolic distance between two points in the Poincaré disk.

    The hyperbolic distance formula is:
        d(z1, z2) = 2 * arctanh(|z1 - z2| / |1 - conj(z1)*z2|)

    Args:
        z1, z2: Complex numbers in the unit disk (|z| < 1)

    Returns:
        Hyperbolic distance (returns inf if either point is on/outside boundary)
    """
    if abs(z1) >= 1 or abs(z2) >= 1:
        return np.inf

    numerator = abs(z1 - z2)
    denominator = abs(1 - np.conj(z1) * z2)

    if denominator < 1e-12:
        return np.inf

    ratio = numerator / denominator
    if ratio >= 1:
        return np.inf

    return 2 * np.arctanh(ratio)


def geodesic_length_boundary_points(
    theta1: float,
    theta2: float,
    radius: float = 0.999
) -> float:
    """Compute the length of a geodesic between two points near the boundary.

    Since points on the boundary have infinite hyperbolic distance, this
    function uses points slightly inside the disk (at the given radius).

    Args:
        theta1, theta2: Angles (in radians) of boundary points
        radius: Radius at which to evaluate (< 1, default 0.999)

    Returns:
        Hyperbolic distance between the points
    """
    z1 = radius * complex(np.cos(theta1), np.sin(theta1))
    z2 = radius * complex(np.cos(theta2), np.sin(theta2))
    return hyperbolic_distance(z1, z2)


# =============================================================================
# Reflections (Isometries)
# =============================================================================

def reflect_across_geodesic(z: complex, center: complex, radius: float) -> complex:
    """Reflect a point across a geodesic circle.

    In the Poincaré disk, reflection across a geodesic (circle orthogonal to
    the boundary) is given by inversion in that circle:
        z' = center + radius² / conj(z - center)

    Args:
        z: Point to reflect
        center: Center of the geodesic circle
        radius: Radius of the geodesic circle

    Returns:
        Reflected point
    """
    return center + radius**2 / np.conj(z - center)


def reflect_across_diameter(z: complex, direction: complex) -> complex:
    """Reflect a point across a diameter (geodesic through the origin).

    For a diameter with direction e^{iθ}, reflection is:
        z' = e^{2iθ} * conj(z)

    Args:
        z: Point to reflect
        direction: Complex number giving the direction of the diameter

    Returns:
        Reflected point
    """
    d = direction / abs(direction)
    return d**2 * np.conj(z)


# =============================================================================
# Möbius Transformations
# =============================================================================

def apply_mobius(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a Möbius transform defined by a 2x2 matrix to complex points.

    The matrix [[a, b], [c, d]] corresponds to f(z) = (az + b) / (cz + d).

    Args:
        matrix: 2x2 numpy array defining the Möbius transform
        points: Array of complex numbers

    Returns:
        Transformed points
    """
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    zs = np.array(points)
    return (a * zs + b) / (c * zs + d)


def get_reflection_matrix(center: complex, radius: float) -> np.ndarray:
    """Get the 2x2 matrix representing reflection across a geodesic circle.

    The reflection z' = center + radius²/conj(z - center) for a circle
    orthogonal to the unit disk is an anti-holomorphic Möbius transform.

    Args:
        center: Center of the geodesic circle
        radius: Radius of the geodesic circle

    Returns:
        2x2 complex numpy array
    """
    return np.array([
        [center, -1.0],
        [1.0, -np.conj(center)]
    ], dtype=complex)


# =============================================================================
# Tessellation
# =============================================================================

def compute_poincare_circumradius(p: int, q: int) -> float:
    """Compute the Poincaré disk radius of vertices for a {p, q} tessellation.

    For a {p, q} tessellation (p-gons meeting q at each vertex), the
    hyperbolic circumradius R satisfies: cosh(R) = cot(π/p) * cot(π/q)
    The Poincaré disk radius is then r = tanh(R/2).

    Args:
        p: Number of sides of each polygon
        q: Number of polygons meeting at each vertex

    Returns:
        Poincaré disk radius of polygon vertices
    """
    cosh_R = 1.0 / (np.tan(np.pi / p) * np.tan(np.pi / q))
    R = np.arccosh(cosh_R)
    return np.tanh(R / 2)


def generate_tessellation(
    p: int,
    q: int,
    max_depth: int = 4,
    max_radius: float = 1.0
) -> list[tuple[list[complex], int]]:
    """Generate tiles for a {p, q} hyperbolic tessellation of the Poincaré disk.

    Uses matrix-based Möbius transforms for numerical stability. Each tile is
    represented by a transformation matrix, avoiding accumulated floating-point
    errors from repeated vertex reflections.

    Args:
        p: Number of sides of each polygon
        q: Number of polygons meeting at each vertex
        max_depth: Maximum recursion depth for tile generation
        max_radius: Only include tiles whose centroid is within this radius

    Returns:
        List of (vertices, parity) where vertices is a list of complex numbers
        and parity is 0 or 1 for checkerboard 2-coloring
    """
    tiles, _, _ = generate_tessellation_with_transforms(p, q, max_depth, max_radius)
    return [(t['vertices'], t['parity']) for t in tiles]


def generate_tessellation_with_transforms(
    p: int,
    q: int,
    max_depth: int = 4,
    max_radius: float = 1.0
) -> tuple[list[dict], list[np.ndarray], np.ndarray]:
    """Generate tiles for a {p, q} tessellation, returning full transform info.

    This version returns the Möbius transform matrix for each tile, enabling
    generation of "ghost" tiles beyond the truncation boundary.

    Args:
        p: Number of sides of each polygon
        q: Number of polygons meeting at each vertex
        max_depth: Maximum recursion depth
        max_radius: Only include tiles whose centroid is within this radius

    Returns:
        Tuple of:
        - List of tile dicts with keys: 'vertices', 'parity', 'depth', 'matrix', 'arrival_idx'
        - List of generator matrices (reflections across base polygon edges)
        - Base vertices array
    """
    if (p - 2) * (q - 2) <= 4:
        raise ValueError(f"{{p, q}} = {{{p}, {q}}} is not hyperbolic. Need (p-2)(q-2) > 4.")

    # Setup base polygon
    r = compute_poincare_circumradius(p, q)
    angles = 2 * np.pi * np.arange(p) / p
    base_vertices = r * np.exp(1j * angles)

    # Pre-calculate generator matrices (reflections across each edge)
    generators = []
    for i in range(p):
        z1 = base_vertices[i]
        z2 = base_vertices[(i + 1) % p]
        C, R = get_geodesic_circle(z1, z2)
        if C is None:
            raise ValueError("Edge was a diameter - not expected for valid {p,q}")
        mat = get_reflection_matrix(C, R)
        generators.append(mat)

    # BFS with matrices
    initial_matrix = np.eye(2, dtype=complex)
    queue = [(initial_matrix, 0, -1)]

    tiles: list[dict] = []
    seen_centroids: list[complex] = []
    TOLERANCE = 1e-4

    while queue:
        M_curr, depth, arrival_idx = queue.pop(0)
        is_odd = (depth % 2 != 0)

        if is_odd:
            verts = apply_mobius(M_curr, np.conj(base_vertices))
            current_verts = list(verts[::-1])
        else:
            verts = apply_mobius(M_curr, base_vertices)
            current_verts = list(verts)

        centroid = np.mean(current_verts)
        if abs(centroid) > max_radius:
            continue

        if any(abs(centroid - c) < TOLERANCE for c in seen_centroids):
            continue

        seen_centroids.append(centroid)
        tiles.append({
            'vertices': current_verts,
            'parity': 0,  # Placeholder
            'depth': depth,
            'matrix': M_curr.copy(),
            'arrival_idx': arrival_idx
        })

        if depth >= max_depth:
            continue

        for i, G in enumerate(generators):
            if i == arrival_idx:
                continue
            if is_odd:
                M_next = M_curr @ np.conj(G)
            else:
                M_next = M_curr @ G
            queue.append((M_next, depth + 1, i))

    # Fix parities using graph-based 2-coloring
    n_tiles = len(tiles)
    if n_tiles == 0:
        return tiles, generators, base_vertices

    # Build adjacency from shared edges
    edge_to_tiles: dict[tuple, list[int]] = {}
    for idx, tile in enumerate(tiles):
        vertices = tile['vertices']
        for i in range(len(vertices)):
            z1 = vertices[i]
            z2 = vertices[(i + 1) % len(vertices)]
            edge_key = tuple(sorted([(round(z1.real, 6), round(z1.imag, 6)),
                                    (round(z2.real, 6), round(z2.imag, 6))]))
            if edge_key not in edge_to_tiles:
                edge_to_tiles[edge_key] = []
            edge_to_tiles[edge_key].append(idx)

    adjacency: list[list[int]] = [[] for _ in range(n_tiles)]
    for tile_indices in edge_to_tiles.values():
        if len(tile_indices) == 2:
            i, j = tile_indices
            adjacency[i].append(j)
            adjacency[j].append(i)

    # BFS 2-coloring
    parities = [-1] * n_tiles
    parities[0] = 0
    bfs_queue = [0]
    while bfs_queue:
        current = bfs_queue.pop(0)
        current_parity = parities[current]
        for neighbor in adjacency[current]:
            if parities[neighbor] == -1:
                parities[neighbor] = 1 - current_parity
                bfs_queue.append(neighbor)

    for idx in range(n_tiles):
        tiles[idx]['parity'] = parities[idx]

    return tiles, generators, base_vertices


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_geodesic(
    ax: "plt.Axes",
    theta1: float,
    theta2: float,
    **kwargs
) -> None:
    """Draw a geodesic on the Poincaré disk connecting boundary points.

    Args:
        ax: Matplotlib axes
        theta1, theta2: Angles (in radians) of boundary endpoints
        **kwargs: Passed to ax.plot()
    """
    arc_x, arc_y = geodesic_arc_points_boundary(theta1, theta2)
    ax.plot(arc_x, arc_y, **kwargs)


def draw_tessellation(
    ax: "plt.Axes",
    p: int = 4,
    q: int = 6,
    max_depth: int = 5,
    color0: str = "white",
    color1: str = "lightgray",
    edge_color: str = "gray",
    edge_alpha: float = 0.3,
    offset: tuple[float, float] = (0.0, 0.0),
    points_per_edge: int = 30,
    extend_to_boundary: bool = True
) -> None:
    """Draw a {p, q} hyperbolic tessellation on the Poincaré disk.

    Edges are drawn as geodesic arcs (circular arcs perpendicular to the boundary).
    When extend_to_boundary=True, boundary edges are extended to the unit circle
    and ideal wedges fill the boundary region with proper checkerboard coloring.

    Args:
        ax: Matplotlib axes
        p: Number of sides per polygon
        q: Number of polygons meeting at each vertex
        max_depth: Maximum recursion depth for tile generation
        color0, color1: Colors for checkerboard tiles
        edge_color: Color for tile edges
        edge_alpha: Alpha for tile edges
        offset: (x, y) offset for the entire tiling
        points_per_edge: Number of points for each curved edge
        extend_to_boundary: If True, extend outer edges to the unit circle
    """
    ox, oy = offset

    tiles, generators, base_vertices = generate_tessellation_with_transforms(
        p, q, max_depth=max_depth
    )

    if not tiles:
        return

    # Build edge->tile map to identify boundary edges
    edge_to_tile_info: dict[tuple, list[dict]] = {}
    for tile in tiles:
        vertices = tile['vertices']
        for i in range(len(vertices)):
            z1 = vertices[i]
            z2 = vertices[(i + 1) % len(vertices)]
            edge_key = tuple(sorted([(round(z1.real, 6), round(z1.imag, 6)),
                                    (round(z2.real, 6), round(z2.imag, 6))]))
            if edge_key not in edge_to_tile_info:
                edge_to_tile_info[edge_key] = []
            edge_to_tile_info[edge_key].append({
                'tile': tile,
                'edge_idx': i,
                'z1': z1,
                'z2': z2
            })

    # 1. Draw Ghost Tiles as Ideal Wedges (boundary region)
    if extend_to_boundary:
        for edge_key, tile_list in edge_to_tile_info.items():
            if len(tile_list) == 1:
                info = tile_list[0]
                parent_tile = info['tile']
                z1_shared, z2_shared = info['z1'], info['z2']
                edge_idx = info['edge_idx']

                parent_parity = parent_tile['parity']
                if parent_parity not in (0, 1):
                    continue
                ghost_parity = 1 - parent_parity
                wedge_color = color0 if ghost_parity == 0 else color1

                # Generate the ghost tile (next layer)
                M_parent = parent_tile['matrix']
                parent_depth = parent_tile['depth']
                is_odd = (parent_depth % 2 != 0)
                G = generators[edge_idx]

                if is_odd:
                    M_ghost = M_parent @ np.conj(G)
                else:
                    M_ghost = M_parent @ G

                ghost_depth = parent_depth + 1
                ghost_is_odd = (ghost_depth % 2 != 0)

                if ghost_is_odd:
                    verts = apply_mobius(M_ghost, np.conj(base_vertices))
                    ghost_verts = list(verts[::-1])
                else:
                    verts = apply_mobius(M_ghost, base_vertices)
                    ghost_verts = list(verts)

                # Find indices of shared vertices in ghost tile
                def get_idx(pt, verts):
                    dists = [abs(v - pt) for v in verts]
                    return int(np.argmin(dists))

                idx1 = get_idx(z1_shared, ghost_verts)
                idx2 = get_idx(z2_shared, ghost_verts)
                n_ghost = len(ghost_verts)

                # Get neighbors in the ghost tile
                prev_1 = ghost_verts[(idx1 - 1) % n_ghost]
                next_1 = ghost_verts[(idx1 + 1) % n_ghost]
                outer_1 = prev_1 if abs(next_1 - z2_shared) < 1e-4 else next_1

                prev_2 = ghost_verts[(idx2 - 1) % n_ghost]
                next_2 = ghost_verts[(idx2 + 1) % n_ghost]
                outer_2 = prev_2 if abs(next_2 - z1_shared) < 1e-4 else next_2

                # Extend the "side" edges to the boundary
                p_ext_1 = extend_geodesic_to_boundary(z1_shared, outer_1)
                p_ext_2 = extend_geodesic_to_boundary(z2_shared, outer_2)

                # Construct the wedge polygon
                pts_real: list[float] = []
                pts_imag: list[float] = []

                # Shared edge (z1 -> z2)
                xs, ys = geodesic_arc_points(z1_shared, z2_shared, n_points=points_per_edge)
                pts_real.extend(xs)
                pts_imag.extend(ys)

                # Side edge 2 (z2 -> p_ext_2)
                xs, ys = geodesic_arc_points(z2_shared, p_ext_2, n_points=points_per_edge)
                pts_real.extend(xs)
                pts_imag.extend(ys)

                # Boundary Arc (p_ext_2 -> p_ext_1)
                angle_a = np.angle(p_ext_2)
                angle_b = np.angle(p_ext_1)
                diff = angle_b - angle_a
                if diff > np.pi:
                    angle_b -= 2 * np.pi
                elif diff < -np.pi:
                    angle_b += 2 * np.pi

                bs = np.linspace(angle_a, angle_b, points_per_edge)
                pts_real.extend(np.cos(bs).tolist())
                pts_imag.extend(np.sin(bs).tolist())

                # Side edge 1 (p_ext_1 -> z1)
                xs, ys = geodesic_arc_points(p_ext_1, z1_shared, n_points=points_per_edge)
                pts_real.extend(xs)
                pts_imag.extend(ys)

                # Apply offsets
                pts_real = [x + ox for x in pts_real]
                pts_imag = [y + oy for y in pts_imag]

                ax.fill(pts_real, pts_imag, color=wedge_color, edgecolor='none',
                       linewidth=0, alpha=1, zorder=-3)

    # 2. Draw Real Tiles (Internal)
    for tile in tiles:
        vertices = tile['vertices']
        parity = tile['parity']
        all_xs = []
        all_ys = []

        for i in range(len(vertices)):
            z1 = vertices[i]
            z2 = vertices[(i + 1) % len(vertices)]
            arc_xs, arc_ys = geodesic_arc_points(z1, z2, n_points=points_per_edge)
            all_xs.extend(arc_xs[:-1])
            all_ys.extend(arc_ys[:-1])

        all_xs = [x + ox for x in all_xs]
        all_ys = [y + oy for y in all_ys]

        tile_color = color0 if parity == 0 else color1
        ax.fill(all_xs, all_ys, color=tile_color, edgecolor='none',
               linewidth=0, alpha=1, zorder=-1)

    # 3. Draw Edges
    edges_drawn = set()
    for tile in tiles:
        vertices = tile['vertices']
        for i in range(len(vertices)):
            z1 = vertices[i]
            z2 = vertices[(i + 1) % len(vertices)]

            edge_key = tuple(sorted([(round(z1.real, 6), round(z1.imag, 6)),
                                     (round(z2.real, 6), round(z2.imag, 6))]))
            if edge_key in edges_drawn:
                continue
            edges_drawn.add(edge_key)

            if extend_to_boundary:
                z1_use = extend_geodesic_to_boundary(z2, z1)
                z2_use = extend_geodesic_to_boundary(z1, z2)
            else:
                z1_use = z1
                z2_use = z2

            arc_xs, arc_ys = geodesic_arc_points(z1_use, z2_use, n_points=points_per_edge)
            arc_xs = [x + ox for x in arc_xs]
            arc_ys = [y + oy for y in arc_ys]
            ax.plot(arc_xs, arc_ys, color=edge_color, alpha=edge_alpha, linewidth=0.5)


def fill_entanglement_wedge(
    ax: "plt.Axes",
    theta1: float,
    theta2: float,
    fill_color: str = "pink",
    alpha: float = 0.5,
    zorder: int = 0,
    offset: tuple[float, float] = (0.0, 0.0)
) -> None:
    """Fill the entanglement wedge between a geodesic and the boundary arc.

    Args:
        ax: Matplotlib axes
        theta1, theta2: Angles defining the boundary region
        fill_color: Color for the wedge fill
        alpha: Transparency
        zorder: Drawing order
        offset: (x, y) offset
    """
    ox, oy = offset

    # Get geodesic arc
    geodesic_x, geodesic_y = geodesic_arc_points_boundary(theta1, theta2, n_points=100)

    # Boundary arc from theta2 back to theta1
    boundary_angles = np.linspace(theta2, theta1, 100)
    boundary_x = np.cos(boundary_angles)
    boundary_y = np.sin(boundary_angles)

    # Combine into closed polygon
    wedge_x = np.concatenate([geodesic_x, boundary_x]) + ox
    wedge_y = np.concatenate([geodesic_y, boundary_y]) + oy

    ax.fill(wedge_x, wedge_y, color=fill_color, alpha=alpha, zorder=zorder)


# =============================================================================
# Utility Functions
# =============================================================================

def order_vertices_ccw(vertices: list[complex]) -> list[complex]:
    """Order polygon vertices counter-clockwise around their centroid.

    Args:
        vertices: List of complex numbers representing vertices

    Returns:
        Vertices reordered counter-clockwise
    """
    verts = np.array(vertices, dtype=complex)
    centroid = np.mean(verts)
    angles = np.angle(verts - centroid)
    order = np.argsort(angles)
    return list(verts[order])


def is_between_short_arc(
    theta: np.ndarray | float,
    alpha: float,
    beta: float
) -> np.ndarray | bool:
    """Check if angle(s) theta lie on the short arc from alpha to beta.

    Args:
        theta: Angle(s) to check
        alpha, beta: Arc endpoints

    Returns:
        Boolean (array) indicating if theta is between alpha and beta
    """
    def angdiff(x, y):
        return (x - y + np.pi) % (2 * np.pi) - np.pi

    d_ab = angdiff(beta, alpha)
    d_at = angdiff(theta, alpha)

    return (np.abs(d_ab) < np.pi) & (d_ab * d_at >= 0) & (np.abs(d_at) <= np.abs(d_ab))

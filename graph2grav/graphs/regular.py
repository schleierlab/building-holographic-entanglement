"""Regular tiling graph generation for spherical, Euclidean, and hyperbolic patches.

This module provides a small, curated set of regular tilings exposed through a
single {p, q} interface. The returned graphs are labeled for use with the
quench-and-measure workflow in :mod:`graph2grav.gaussian` and
:mod:`graph2grav.analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Iterable

import networkx as nx
import numpy as np
from pydantic import ValidationError

from .hyperbolic import (
    generate_hyperbolic_tiling_with_hypertiling,
)
from .validation import check_labels


SUPPORTED_TILINGS: tuple[tuple[int, int], ...] = (
    (3, 3),
    (4, 3),
    (3, 4),
    (5, 3),
    (3, 5),
    (3, 6),
    (4, 4),
    (6, 3),
    (3, 8),
    (4, 5),
)


def _is_supported_tiling(p: int, q: int) -> bool:
    if (p, q) in SUPPORTED_TILINGS:
        return True
    return p >= 3 and q >= 3 and Fraction(1, p) + Fraction(1, q) < Fraction(1, 2)


@dataclass(frozen=True)
class RegularTilingSpec:
    """Specification for a regular tiling patch."""

    p: int
    q: int
    depth: int
    boundary_shells: int = 1
    center: object | None = None
    centering: str | None = None
    geometry: str | None = None

    @property
    def schlafli_symbol(self) -> str:
        return f"{{{self.p},{self.q}}}"


def supported_regular_tilings() -> list[tuple[int, int]]:
    """Return the curated display set of supported Schlafli symbols."""

    return list(SUPPORTED_TILINGS)


def infer_geometry(p: int, q: int) -> str:
    """Infer geometry from the Schlafli symbol."""

    lhs = Fraction(1, p) + Fraction(1, q)
    half = Fraction(1, 2)
    if lhs > half:
        return "spherical"
    if lhs == half:
        return "euclidean"
    return "hyperbolic"


def generate_regular_patch(spec: RegularTilingSpec, validate: bool = True) -> nx.Graph:
    """Generate a labeled patch of a supported regular tiling.

    Args:
        spec: Tiling specification.
        validate: If True, validate graph/node labels with Pydantic.

    Returns:
        A graph with integer node labels and the attributes required by the
        Gaussian-state analysis code.
    """

    if spec.depth < 1:
        raise ValueError("RegularTilingSpec.depth must be at least 1.")
    if spec.boundary_shells < 1:
        raise ValueError("RegularTilingSpec.boundary_shells must be at least 1.")

    requested = (spec.p, spec.q)
    if not _is_supported_tiling(*requested):
        raise NotImplementedError(
            f"{spec.schlafli_symbol} is not supported. "
            f"Supported curated tilings: {supported_regular_tilings()}, "
            "plus all hyperbolic {p,q} with p,q >= 3 and 1/p + 1/q < 1/2."
        )

    geometry = infer_geometry(spec.p, spec.q)
    if spec.geometry is not None and spec.geometry != geometry:
        raise ValueError(
            f"Geometry mismatch for {spec.schlafli_symbol}: inferred {geometry!r}, "
            f"got {spec.geometry!r}."
        )

    if geometry == "spherical":
        graph, center_2d, resolved_centering, resolved_truncation = _generate_spherical_patch(spec)
    elif geometry == "euclidean":
        graph, center_2d, resolved_centering, resolved_truncation = _generate_euclidean_patch(spec)
    else:
        graph, center_2d, resolved_centering, resolved_truncation = _generate_hyperbolic_patch(spec)

    label_circular_boundary_from_embedding(
        graph,
        center=center_2d,
        boundary_shells=spec.boundary_shells,
    )

    graph.graph["number_of_boundaries"] = 1
    graph.graph["periodic"] = True
    graph.graph["subdivided"] = False
    graph.graph["schlafli_symbol"] = spec.schlafli_symbol
    graph.graph["p"] = spec.p
    graph.graph["q"] = spec.q
    graph.graph["geometry"] = geometry
    graph.graph["boundary_shells"] = spec.boundary_shells
    graph.graph["depth"] = spec.depth
    graph.graph["center"] = tuple(float(v) for v in center_2d)
    graph.graph["centering"] = resolved_centering
    graph.graph["truncation"] = resolved_truncation

    graph_attrs_ref = graph.graph
    for node in graph.nodes:
        graph.nodes[node]["graph"] = graph_attrs_ref

    if validate:
        try:
            check_labels(graph)
        except ValidationError as exc:
            raise ValueError(
                f"Generated graph for {spec.schlafli_symbol} failed validation."
            ) from exc

    return graph


def decorate_regular_patch(
    graph: nx.Graph,
    *,
    bulk_weight: float = 1.0,
    boundary_weight: float = 1.0,
    validate: bool = True,
) -> nx.Graph:
    """Decorate a regular patch with signed edge modes and dangling probes.

    Every original edge ``u-v`` is replaced by ``u-s-v`` with weights
    ``(+bulk_weight, -bulk_weight)``. Original perimeter nodes become measured
    frontier modes, and one retained probe is attached to each frontier mode.
    This is the regular-tiling analogue of ``subdivided_tree_decoration_1``.
    """
    if graph.graph.get("number_of_boundaries", 1) != 1:
        raise ValueError("regular-patch decoration requires exactly one boundary")
    if not graph.graph.get("periodic", False):
        raise ValueError("regular-patch decoration requires a periodic boundary")
    if bulk_weight == 0 or boundary_weight == 0:
        raise ValueError("decoration weights must be nonzero")

    decorated = graph.copy()
    original_edges = list(decorated.edges())
    boundary_nodes = sorted(
        [node for node, data in decorated.nodes(data=True) if data.get("is_boundary")],
        key=lambda node: decorated.nodes[node]["position_in_boundary"],
    )
    if not boundary_nodes:
        raise ValueError("regular patch has no labeled boundary")
    next_node = max(decorated.nodes) + 1 if decorated.number_of_nodes() else 0

    for node, data in decorated.nodes(data=True):
        data["tier"] = "primary"
        data["region"] = "bulk"
        was_boundary = bool(data.get("is_boundary"))
        data["was_boundary"] = was_boundary
        if was_boundary:
            data["original_position_in_boundary"] = int(data["position_in_boundary"])
            data["original_boundary_angle"] = float(
                data.get(
                    "boundary_angle",
                    np.angle(complex(*np.asarray(data["coord"], dtype=float))),
                )
            )
        data["is_boundary"] = False
        data["is_ancilla"] = True
        for boundary_index in range(10):
            data[f"b{boundary_index}"] = False

    for node_u, node_v in original_edges:
        coord_u = np.asarray(decorated.nodes[node_u]["coord"], dtype=float)
        coord_v = np.asarray(decorated.nodes[node_v]["coord"], dtype=float)
        midpoint = tuple(((coord_u + coord_v) / 2).tolist())
        decorated.add_node(
            next_node,
            coord=midpoint,
            tier="secondary",
            region="bulk",
            is_ancilla=True,
            is_boundary=False,
            b0=False,
        )
        for boundary_index in range(1, 10):
            decorated.nodes[next_node][f"b{boundary_index}"] = False
        decorated.remove_edge(node_u, node_v)
        decorated.add_edge(node_u, next_node, mweight=float(bulk_weight))
        decorated.add_edge(node_v, next_node, mweight=-float(bulk_weight))
        next_node += 1

    for frontier_node in boundary_nodes:
        data = decorated.nodes[frontier_node]
        position = int(data["original_position_in_boundary"])
        angle = float(data["original_boundary_angle"])
        coord = np.asarray(data["coord"], dtype=float)
        radius = float(np.linalg.norm(coord))
        direction = np.array([1.0, 0.0]) if radius < 1e-12 else coord / radius
        radial_gap = max(0.08, 0.7 * (1.0 - radius))
        probe_coord = tuple((direction * min(0.985, radius + radial_gap)).tolist())
        decorated.add_node(
            next_node,
            coord=probe_coord,
            tier="primary",
            region="probe",
            is_ancilla=False,
            is_boundary=True,
            b0=True,
            boundary_index=0,
            position_in_boundary=position,
            boundary_angle=angle,
            probe_parent=frontier_node,
        )
        decorated.add_edge(frontier_node, next_node, mweight=float(boundary_weight))
        next_node += 1

    decorated.graph = dict(graph.graph)
    decorated.graph["subdivided"] = True
    decorated.graph["decorated_from_regular"] = True
    decorated.graph["decorated_bulk_weight"] = float(bulk_weight)
    decorated.graph["decorated_boundary_weight"] = float(boundary_weight)
    decorated.graph["decorated_boundary_mode"] = "subdivided_bulk_plus_dangling_probes"
    decorated = nx.convert_node_labels_to_integers(
        decorated, ordering="sorted", label_attribute="original_node"
    )
    graph_attrs_ref = decorated.graph
    for node in decorated.nodes:
        decorated.nodes[node]["graph"] = graph_attrs_ref

    if validate:
        check_labels(decorated)
    return decorated


def label_circular_boundary_from_embedding(
    graph: nx.Graph,
    *,
    center: complex | tuple[float, float] | list[float] | np.ndarray = (0.0, 0.0),
    boundary_shells: int = 1,
) -> None:
    """Label the outer shell of a patch as a cyclic boundary.

    When face-incidence data is available, the exposed cut is the set of edges
    incident to exactly one retained face. Otherwise, deficient-degree nodes
    seed the boundary. Additional shells are included by graph distance from
    that exposed cut. A one-shell perimeter is ordered by walking its cycle;
    thicker boundaries fall back to polar-angle ordering.
    """

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot label boundary on an empty graph.")
    if boundary_shells < 1:
        raise ValueError("boundary_shells must be at least 1.")

    center_xy = _coerce_point_2d(center)
    ambient_degree = graph.graph.get("ambient_degree")
    if ambient_degree is None:
        ambient_degree = max(data.get("ambient_degree", graph.degree(node)) for node, data in graph.nodes(data=True))

    perimeter_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("face_incidence") == 1
    ]
    if perimeter_edges:
        exposed_boundary = sorted({node for edge in perimeter_edges for node in edge})
    else:
        exposed_boundary = [
            node
            for node, data in graph.nodes(data=True)
            if graph.degree(node) < data.get("ambient_degree", ambient_degree)
        ]

    if not exposed_boundary:
        radii = {
            node: np.hypot(
                _coerce_point_2d(data["coord"])[0] - center_xy[0],
                _coerce_point_2d(data["coord"])[1] - center_xy[1],
            )
            for node, data in graph.nodes(data=True)
        }
        max_radius = max(radii.values())
        exposed_boundary = [node for node, radius in radii.items() if np.isclose(radius, max_radius)]

    shell_distances = nx.multi_source_dijkstra_path_length(graph, exposed_boundary, weight=None)
    boundary_nodes = {
        node for node, dist in shell_distances.items() if dist < boundary_shells
    }

    def angle_ccw_from_east(node: int) -> float:
        x, y = _coerce_point_2d(graph.nodes[node]["coord"])
        theta = float(np.arctan2(y - center_xy[1], x - center_xy[0]))
        return float(np.mod(theta, 2 * np.pi))

    def radius(node: int) -> float:
        x, y = _coerce_point_2d(graph.nodes[node]["coord"])
        return float(np.hypot(x - center_xy[0], y - center_xy[1]))

    if boundary_shells == 1 and perimeter_edges:
        perimeter = nx.Graph(perimeter_edges)
        if not nx.is_connected(perimeter) or any(
            perimeter.degree(node) != 2 for node in perimeter.nodes
        ):
            raise ValueError("Exposed face edges do not form a single boundary cycle.")
        ordered_boundary = _order_boundary_cycle(
            perimeter,
            graph=graph,
            center=center_xy,
        )
    else:
        ordered_boundary = sorted(
            boundary_nodes,
            key=lambda node: (angle_ccw_from_east(node), -radius(node)),
        )

    for node, data in graph.nodes(data=True):
        is_boundary = node in boundary_nodes
        data["is_boundary"] = is_boundary
        data["is_ancilla"] = not is_boundary
        for boundary_flag in range(10):
            data.pop(f"b{boundary_flag}", None)
        data["b0"] = is_boundary
        data.pop("boundary_index", None)
        data.pop("position_in_boundary", None)
        if is_boundary:
            data["boundary_index"] = 0

    for idx, node in enumerate(ordered_boundary):
        graph.nodes[node]["position_in_boundary"] = idx
        graph.nodes[node]["boundary_angle"] = angle_ccw_from_east(node)


def _order_boundary_cycle(
    perimeter: nx.Graph,
    *,
    graph: nx.Graph,
    center: tuple[float, float],
) -> list[int]:
    """Walk a degree-two perimeter counterclockwise, starting nearest east."""

    center_xy = np.asarray(center, dtype=float)

    def point(node: int) -> np.ndarray:
        return np.asarray(_coerce_point_2d(graph.nodes[node]["coord"]), dtype=float)

    def angle(node: int) -> float:
        delta = point(node) - center_xy
        return float(np.mod(np.arctan2(delta[1], delta[0]), 2 * np.pi))

    start = min(perimeter.nodes, key=lambda node: (angle(node), -np.linalg.norm(point(node) - center_xy)))

    def walk(first_neighbor: int) -> list[int]:
        ordered = [start]
        previous = start
        current = first_neighbor
        while current != start:
            ordered.append(current)
            next_nodes = [node for node in perimeter.neighbors(current) if node != previous]
            if len(next_nodes) != 1:
                raise ValueError("Could not walk the boundary cycle unambiguously.")
            previous, current = current, next_nodes[0]
        return ordered

    neighbors = list(perimeter.neighbors(start))
    candidate = walk(neighbors[0])
    coords = np.asarray([point(node) for node in candidate])
    signed_area = 0.5 * np.sum(
        coords[:, 0] * np.roll(coords[:, 1], -1)
        - coords[:, 1] * np.roll(coords[:, 0], -1)
    )
    return candidate if signed_area > 0 else walk(neighbors[1])


def _generate_spherical_patch(
    spec: RegularTilingSpec,
) -> tuple[nx.Graph, tuple[float, float], str, str]:
    graph, coords = _build_spherical_ambient_graph(spec.p, spec.q)
    center_node, center_vector = _resolve_spherical_center(graph, coords, spec.center)

    shell_nodes = nx.single_source_shortest_path_length(graph, center_node, cutoff=spec.depth)
    patch_nodes = sorted(shell_nodes.keys())
    patch = graph.subgraph(patch_nodes).copy()

    projected = _project_to_tangent_plane(coords, center_vector)
    for node in patch.nodes:
        patch.nodes[node]["coord"] = tuple(float(v) for v in projected[node])
        patch.nodes[node]["ambient_degree"] = spec.q

    patch = nx.convert_node_labels_to_integers(patch, ordering="sorted", label_attribute="original_node")
    patch.graph["ambient_degree"] = spec.q
    return patch, (0.0, 0.0), "vertex", "graph_shells"


def _generate_euclidean_patch(
    spec: RegularTilingSpec,
) -> tuple[nx.Graph, tuple[float, float], str, str]:
    if (spec.p, spec.q) == (4, 4):
        ambient = _build_square_lattice(max(6, 4 * spec.depth + 6))
    elif (spec.p, spec.q) == (3, 6):
        ambient = _build_triangular_lattice(max(6, 4 * spec.depth + 6))
    elif (spec.p, spec.q) == (6, 3):
        ambient = _build_hexagonal_lattice(max(4, 3 * spec.depth + 6))
    else:
        raise NotImplementedError(f"Unsupported Euclidean tiling {spec.schlafli_symbol}.")

    center_xy, resolved_centering = _resolve_planar_center(
        ambient,
        spec.center,
        spec.centering,
        tiling=(spec.p, spec.q),
    )

    if (spec.p, spec.q) == (6, 3) and resolved_centering == "face":
        patch, center_xy = _extract_hexagonal_face_shell_patch(
            ambient,
            center_xy,
            depth=spec.depth,
        )
        resolved_truncation = "face_shells"
    else:
        radius = float(spec.depth) + 0.35
        patch = _extract_radius_patch(ambient, center_xy, radius)
        resolved_truncation = "euclidean_radius"

    for _, data in patch.nodes(data=True):
        data["ambient_degree"] = spec.q

    patch = nx.convert_node_labels_to_integers(patch, ordering="sorted", label_attribute="original_node")
    patch.graph["ambient_degree"] = spec.q
    return patch, center_xy, resolved_centering, resolved_truncation


def _generate_hyperbolic_patch(
    spec: RegularTilingSpec,
) -> tuple[nx.Graph, tuple[float, float], str, str]:
    resolved_centering = spec.centering or "vertex"
    if resolved_centering not in {"vertex", "face"}:
        raise ValueError("Hyperbolic centering must be 'vertex' or 'face'.")
    hypertiling_center = "cell" if resolved_centering == "face" else "vertex"
    graph = generate_hyperbolic_tiling_with_hypertiling(
        spec.p,
        spec.q,
        depth=spec.depth,
        center=hypertiling_center,
    )
    for _, data in graph.nodes(data=True):
        coord = data["coord"]
        data["coord"] = (float(coord.real), float(coord.imag))
        data["ambient_degree"] = spec.q
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted", label_attribute="original_node")
    graph.graph["ambient_degree"] = spec.q
    center_xy = _coerce_point_2d(spec.center if spec.center is not None else (0.0, 0.0))
    return graph, center_xy, resolved_centering, "hypertiling_layers"


def _build_spherical_ambient_graph(p: int, q: int) -> tuple[nx.Graph, dict[int, np.ndarray]]:
    coords = _platonic_coordinates(p, q)
    graph = nx.Graph()
    for idx, coord in coords.items():
        graph.add_node(idx, coord3=np.asarray(coord, dtype=float), ambient_degree=q)

    edge_pairs = _edges_from_unit_coordinates(coords.values(), degree=q)
    for i, j in edge_pairs:
        graph.add_edge(i, j)

    return graph, {idx: np.asarray(coord, dtype=float) for idx, coord in coords.items()}


def _platonic_coordinates(p: int, q: int) -> dict[int, np.ndarray]:
    phi = (1 + np.sqrt(5.0)) / 2.0

    if (p, q) == (3, 3):
        raw = [
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
        ]
    elif (p, q) == (4, 3):
        raw = list(product((-1, 1), repeat=3))
    elif (p, q) == (3, 4):
        raw = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
    elif (p, q) == (5, 3):
        raw = list(product((-1, 1), repeat=3))
        raw += [
            (0, 1 / phi, phi), (0, 1 / phi, -phi), (0, -1 / phi, phi), (0, -1 / phi, -phi),
            (1 / phi, phi, 0), (1 / phi, -phi, 0), (-1 / phi, phi, 0), (-1 / phi, -phi, 0),
            (phi, 0, 1 / phi), (phi, 0, -1 / phi), (-phi, 0, 1 / phi), (-phi, 0, -1 / phi),
        ]
    elif (p, q) == (3, 5):
        raw = [
            (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi),
            (1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1),
        ]
    else:
        raise NotImplementedError(f"Unsupported spherical tiling {{{p},{q}}}.")

    coords: dict[int, np.ndarray] = {}
    for idx, coord in enumerate(raw):
        vec = np.asarray(coord, dtype=float)
        vec = vec / np.linalg.norm(vec)
        coords[idx] = vec
    return coords


def _edges_from_unit_coordinates(coords: Iterable[np.ndarray], degree: int) -> list[tuple[int, int]]:
    coord_list = [np.asarray(coord, dtype=float) for coord in coords]
    distances = {}
    min_distance = None
    for i in range(len(coord_list)):
        for j in range(i + 1, len(coord_list)):
            dist = float(np.linalg.norm(coord_list[i] - coord_list[j]))
            distances[(i, j)] = dist
            if min_distance is None or dist < min_distance:
                min_distance = dist

    assert min_distance is not None
    tolerance = min_distance * 1.05
    edges = [(i, j) for (i, j), dist in distances.items() if dist <= tolerance]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(coord_list)))
    graph.add_edges_from(edges)
    if not all(graph.degree(node) == degree for node in graph.nodes):
        raise ValueError("Failed to reconstruct the expected Platonic-solid adjacency.")
    return edges


def _project_to_tangent_plane(
    coords: dict[int, np.ndarray],
    center_vector: np.ndarray,
) -> dict[int, tuple[float, float]]:
    c = center_vector / np.linalg.norm(center_vector)
    reference = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(c, reference)
    u /= np.linalg.norm(u)
    v = np.cross(c, u)
    v /= np.linalg.norm(v)

    projected = {}
    for node, coord in coords.items():
        projected[node] = (float(np.dot(coord, u)), float(np.dot(coord, v)))
    return projected


def _resolve_spherical_center(
    graph: nx.Graph,
    coords: dict[int, np.ndarray],
    center: object | None,
) -> tuple[int, np.ndarray]:
    if center is None:
        center_node = 0
        return center_node, coords[center_node]
    if isinstance(center, int):
        if center not in graph:
            raise ValueError(f"Center node {center} not found in spherical graph.")
        return center, coords[center]

    center_vec = np.asarray(center, dtype=float)
    if center_vec.shape != (3,):
        raise ValueError("Spherical center must be a node id or a 3-vector.")
    center_vec = center_vec / np.linalg.norm(center_vec)
    center_node = max(coords, key=lambda node: float(np.dot(coords[node], center_vec)))
    return center_node, center_vec


def _build_square_lattice(span: int) -> nx.Graph:
    graph = nx.Graph()
    for x in range(-span, span + 1):
        for y in range(-span, span + 1):
            graph.add_node((x, y), coord=(float(x), float(y)), ambient_degree=4)
    for x in range(-span, span + 1):
        for y in range(-span, span + 1):
            if x < span:
                graph.add_edge((x, y), (x + 1, y))
            if y < span:
                graph.add_edge((x, y), (x, y + 1))
    return graph


def _build_triangular_lattice(span: int) -> nx.Graph:
    graph = nx.Graph()
    root_three_over_two = np.sqrt(3.0) / 2.0
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            x = i + 0.5 * j
            y = root_three_over_two * j
            graph.add_node((i, j), coord=(float(x), float(y)), ambient_degree=6)

    directions = [(1, 0), (0, 1), (1, -1)]
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            for di, dj in directions:
                neighbor = (i + di, j + dj)
                if neighbor in graph:
                    graph.add_edge((i, j), neighbor)
    return graph


def _build_hexagonal_lattice(span: int) -> nx.Graph:
    ambient = nx.hexagonal_lattice_graph(span, span)
    graph = nx.Graph()
    for node, data in ambient.nodes(data=True):
        x, y = data["pos"]
        graph.add_node(node, coord=(float(x), float(y)), ambient_degree=3)
    graph.add_edges_from(ambient.edges())
    return graph


def _extract_radius_patch(
    ambient: nx.Graph,
    center_xy: tuple[float, float],
    radius: float,
) -> nx.Graph:
    center_node = min(
        ambient.nodes,
        key=lambda node: _distance(_coerce_point_2d(ambient.nodes[node]["coord"]), center_xy),
    )
    selected = {
        node for node, data in ambient.nodes(data=True)
        if _distance(_coerce_point_2d(data["coord"]), center_xy) <= radius
    }
    patch = ambient.subgraph(selected).copy()
    component = nx.node_connected_component(patch, center_node)
    return patch.subgraph(component).copy()


def _extract_hexagonal_face_shell_patch(
    ambient: nx.Graph,
    center_xy: tuple[float, float],
    *,
    depth: int,
) -> tuple[nx.Graph, tuple[float, float]]:
    bounded_faces = _bounded_faces_with_centers(ambient, face_size=6)
    if not bounded_faces:
        raise ValueError("Could not identify bounded hexagon faces for {6,3} truncation.")

    target = np.asarray(center_xy, dtype=float)
    center_face_idx = min(
        range(len(bounded_faces)),
        key=lambda idx: float(
            np.linalg.norm(np.asarray(bounded_faces[idx][1], dtype=float) - target)
        ),
    )

    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(range(len(bounded_faces)))
    face_vertex_sets = [set(face) for face, _ in bounded_faces]
    for i in range(len(bounded_faces)):
        for j in range(i + 1, len(bounded_faces)):
            if len(face_vertex_sets[i] & face_vertex_sets[j]) == 2:
                dual_graph.add_edge(i, j)

    face_shell_radius = max(depth - 2, 0)
    selected_face_indices = nx.single_source_shortest_path_length(
        dual_graph,
        center_face_idx,
        cutoff=face_shell_radius,
    )
    selected_vertices = set()
    for face_idx in selected_face_indices:
        selected_vertices.update(bounded_faces[face_idx][0])

    patch = ambient.subgraph(selected_vertices).copy()
    patch.graph["face_shell_radius"] = face_shell_radius
    center_face_xy = bounded_faces[center_face_idx][1]
    return patch, center_face_xy


def _resolve_planar_center(
    graph: nx.Graph,
    center: object | None,
    centering: str | None,
    *,
    tiling: tuple[int, int],
) -> tuple[tuple[float, float], str]:
    target = _planar_target_center(graph)

    if center is None:
        resolved_centering = _resolve_default_centering(centering, tiling)
        if resolved_centering == "vertex":
            return _nearest_planar_node_center(graph, target), resolved_centering
        if resolved_centering == "face":
            return _resolve_planar_face_center(graph, face_size=tiling[0], target=target), resolved_centering
        raise ValueError(f"Unsupported centering mode {resolved_centering!r}.")

    if centering is not None and centering not in {"vertex", "face"}:
        raise ValueError("Euclidean centering must be 'vertex' or 'face'.")
    return _coerce_point_2d(center), "explicit"


def _resolve_default_centering(centering: str | None, tiling: tuple[int, int]) -> str:
    if centering is None:
        return "face" if tiling == (6, 3) else "vertex"
    if centering not in {"vertex", "face"}:
        raise ValueError("Euclidean centering must be 'vertex' or 'face'.")
    return centering


def _planar_target_center(graph: nx.Graph) -> tuple[float, float]:
    coords = np.array([data["coord"] for _, data in graph.nodes(data=True)], dtype=float)
    centroid = coords.mean(axis=0)
    return (float(centroid[0]), float(centroid[1]))


def _nearest_planar_node_center(
    graph: nx.Graph,
    target: tuple[float, float],
) -> tuple[float, float]:
    target_array = np.asarray(target, dtype=float)
    node = min(
        graph.nodes,
        key=lambda node_id: float(
            np.linalg.norm(np.asarray(graph.nodes[node_id]["coord"], dtype=float) - target_array)
        ),
    )
    return _coerce_point_2d(graph.nodes[node]["coord"])


def _resolve_planar_face_center(
    graph: nx.Graph,
    *,
    face_size: int,
    target: tuple[float, float],
) -> tuple[float, float]:
    face_centers = _bounded_face_centers(graph, face_size=face_size)
    if not face_centers:
        raise ValueError("Could not identify a bounded face for planar face-centering.")

    target_array = np.asarray(target, dtype=float)
    return min(
        face_centers,
        key=lambda point: float(np.linalg.norm(np.asarray(point, dtype=float) - target_array)),
    )


def _bounded_face_centers(
    graph: nx.Graph,
    *,
    face_size: int,
) -> list[tuple[float, float]]:
    return [center for _, center in _bounded_faces_with_centers(graph, face_size=face_size)]


def _bounded_faces_with_centers(
    graph: nx.Graph,
    *,
    face_size: int,
) -> list[tuple[tuple[object, ...], tuple[float, float]]]:
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise ValueError("Planar face-centering requires a planar graph.")

    seen_directed_edges: set[tuple[object, object]] = set()
    faces_with_centers: list[tuple[tuple[object, ...], tuple[float, float]]] = []
    seen_faces: set[frozenset[object]] = set()
    for u in embedding:
        for v in embedding[u]:
            if (u, v) in seen_directed_edges:
                continue

            face = embedding.traverse_face(u, v)
            for idx in range(len(face)):
                a = face[idx]
                b = face[(idx + 1) % len(face)]
                seen_directed_edges.add((a, b))

            if len(face) != face_size:
                continue

            face_key = frozenset(face)
            if face_key in seen_faces:
                continue
            seen_faces.add(face_key)

            pts = np.array([graph.nodes[node]["coord"] for node in face], dtype=float)
            center = pts.mean(axis=0)
            faces_with_centers.append((tuple(face), (float(center[0]), float(center[1]))))

    return faces_with_centers


def _coerce_point_2d(point: complex | tuple[float, float] | list[float] | np.ndarray) -> tuple[float, float]:
    if isinstance(point, complex):
        return (float(point.real), float(point.imag))
    arr = np.asarray(point, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Expected a 2D point, got shape {arr.shape}.")
    return (float(arr[0]), float(arr[1]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

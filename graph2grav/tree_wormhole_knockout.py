from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import tomllib
from contextlib import redirect_stdout

import networkx as nx
import numpy as np

from . import gaussian
from .analysis import Analysis
from .graphs.regular import RegularTilingSpec, decorate_regular_patch, generate_regular_patch
from .graphs.standard import (
    crosslinked_tree,
    make_tree_periodic,
    subdivided_tree_decoration_1,
    tree_node_rowindex,
)
from . import technical_helpers


SYMEIG_WARNING_TEXT = "Warning: symeigs less than 1/2:"


@dataclass(frozen=True)
class TreeWormholeConfig:
    graph_kind: str
    couplingtime: float = 1.0
    presqueeze: float = 0.2
    renyi2: bool = False
    use_decorated: bool = False
    mera_depth: int = 4
    mera_remove_nodes: bool = False
    wormhole_start_depth: int = 4
    wormhole_end_depth: int = 6
    wormhole_neck_length: int = 2
    wormhole_onesided: bool = True
    hyperbolic_p: int = 3
    hyperbolic_q: int = 7
    hyperbolic_depth: int = 3
    hyperbolic_boundary_shells: int = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_wormhole_config() -> dict:
    config_path = _repo_root() / "global_figure_settings.toml"
    config = tomllib.loads(config_path.read_text())
    config["wormhole-figure"]["wormhole"]["start_depth"] = int(config["wormhole-figure"]["wormhole"]["start_depth"])
    config["wormhole-figure"]["wormhole"]["end_depth"] = int(config["wormhole-figure"]["wormhole"]["end_depth"])
    config["wormhole-figure"]["wormhole"]["neck_length"] = int(config["wormhole-figure"]["wormhole"]["neck_length"])
    return config


def _ensure_graph_refs(graph: nx.Graph) -> nx.Graph:
    graph_attrs_ref = graph.graph
    for node in graph.nodes:
        graph.nodes[node]["graph"] = graph_attrs_ref
    return graph


def _assign_mera_coords(graph: nx.Graph) -> nx.Graph:
    depth_values = [int(graph.nodes[node]["depth"]) for node in graph.nodes if graph.nodes[node].get("depth") is not None]
    depth_max = max(depth_values) if depth_values else 0
    hyperbolic_spacing = 0.5
    for node in graph.nodes:
        data = graph.nodes[node]
        if data.get("tier") == "secondary":
            continue
        if data.get("region") == "probe":
            position = int(data.get("position_in_boundary", 0))
            number_of_boundary_nodes = max(
                1,
                sum(
                    1
                    for _, other in graph.nodes(data=True)
                    if other.get("is_boundary", False)
                ),
            )
            angular_spacing = 2.0 * np.pi / number_of_boundary_nodes
            theta = 2.0 * np.pi * position / number_of_boundary_nodes + angular_spacing / 2.0
            # Keep dangling probes visibly outside the crown ring in the
            # decorated MERA view so the probe legs do not collapse.
            radius = 1.22
        else:
            depth = int(data["depth"])
            row = int(tree_node_rowindex(int(node)))
            number_of_nodes_at_depth = 2 ** depth if depth > 0 else 1
            euclidean_r = hyperbolic_spacing * depth
            max_euclidean_r = hyperbolic_spacing * depth_max
            if max_euclidean_r <= 0:
                radius = 0.0
            else:
                radius = float(np.tanh(euclidean_r / 2.0) / np.tanh(max_euclidean_r / 2.0))
            angular_spacing = 2.0 * np.pi / number_of_nodes_at_depth
            theta = 2.0 * np.pi * row / number_of_nodes_at_depth + angular_spacing / 2.0
            if depth == depth_max:
                radius *= 1.05
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        data["coord"] = (float(x), float(y))
    for node, data in graph.nodes(data=True):
        if data.get("tier") != "secondary":
            continue
        neighbor_coords = [
            np.asarray(graph.nodes[neighbor]["coord"], dtype=float)
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor].get("coord") is not None
        ]
        if neighbor_coords:
            midpoint = np.mean(neighbor_coords, axis=0)
            data["coord"] = (float(midpoint[0]), float(midpoint[1]))
    graph.graph["visual_depth_max"] = depth_max
    return graph


def _decorate_regular_graph(graph: nx.Graph, *, bulk_weight: float = 1.0, boundary_weight: float = 1.0) -> nx.Graph:
    return decorate_regular_patch(
        graph,
        bulk_weight=bulk_weight,
        boundary_weight=boundary_weight,
    )


def build_mera_graph(config: TreeWormholeConfig) -> nx.Graph:
    if bool(config.use_decorated):
        graph = subdivided_tree_decoration_1(
            int(config.mera_depth),
            draw=False,
            validate=False,
        )
    else:
        graph = make_tree_periodic(
            crosslinked_tree(int(config.mera_depth)),
            remove_nodes=bool(config.mera_remove_nodes),
        )
    graph = _assign_mera_coords(graph)
    graph.graph["knockout_case"] = "mera"
    graph.graph["decorated_mode"] = bool(config.use_decorated)
    return _ensure_graph_refs(graph)


def build_wormhole_graph(config: TreeWormholeConfig) -> nx.Graph:
    from figure_helpers.wormhole import decorated_wormhole, undecorated_wormhole

    figure_config = _default_wormhole_config()
    figure_config["wormhole-figure"]["wormhole"]["start_depth"] = int(config.wormhole_start_depth)
    figure_config["wormhole-figure"]["wormhole"]["end_depth"] = int(config.wormhole_end_depth)
    figure_config["wormhole-figure"]["wormhole"]["neck_length"] = int(config.wormhole_neck_length)

    wormhole_builder = decorated_wormhole if bool(config.use_decorated) else undecorated_wormhole
    graph = wormhole_builder(
        plot=False,
        config=figure_config,
        onesided=bool(config.wormhole_onesided),
    )
    for node in graph.nodes:
        position = np.asarray(graph.nodes[node]["position_in_3D"], dtype=float)
        graph.nodes[node]["coord"] = (float(position[0]), float(position[2]))
    graph.graph["knockout_case"] = "wormhole"
    graph.graph["wormhole_onesided"] = bool(config.wormhole_onesided)
    graph.graph["decorated_mode"] = bool(config.use_decorated)
    return _ensure_graph_refs(graph)


def build_hyperbolic_graph(config: TreeWormholeConfig) -> nx.Graph:
    spec = RegularTilingSpec(
        p=int(config.hyperbolic_p),
        q=int(config.hyperbolic_q),
        depth=int(config.hyperbolic_depth),
        boundary_shells=int(config.hyperbolic_boundary_shells),
        center=None,
        centering=None,
    )
    graph = generate_regular_patch(spec)
    if bool(config.use_decorated):
        graph = _decorate_regular_graph(graph)
    graph.graph["knockout_case"] = "hyperbolic"
    graph.graph["decorated_mode"] = bool(config.use_decorated)
    return _ensure_graph_refs(graph)


def build_graph(config: TreeWormholeConfig) -> nx.Graph:
    kind = str(config.graph_kind).lower()
    if kind == "mera":
        return build_mera_graph(config)
    if kind == "wormhole":
        return build_wormhole_graph(config)
    if kind == "hyperbolic":
        return build_hyperbolic_graph(config)
    raise ValueError(f"Unsupported graph_kind: {config.graph_kind}")


def build_analysis(config: TreeWormholeConfig) -> tuple[nx.Graph, Analysis]:
    graph = build_graph(config)
    analysis = Analysis(
        graph,
        couplingtime=float(config.couplingtime),
        global_presqueeze=float(config.presqueeze),
        is_decorated=bool(config.use_decorated),
    )
    return graph, analysis


def boundary_after_measurement_payload(
    graph: nx.Graph,
    *,
    couplingtime: float,
    presqueeze: float,
) -> dict:
    if bool(graph.graph.get("decorated_mode", False)):
        covariance = gaussian.covariance_from_weighted_quench(
            graph,
            couplingtime=float(couplingtime),
            global_presqueeze=float(presqueeze),
        )
    else:
        covariance = gaussian.covariance_from_unweighted_quench(
            graph,
            couplingtime=float(couplingtime),
            bulk_presqueeze=float(presqueeze),
            boundary_presqueeze=float(presqueeze),
        )
    kept_indices = gaussian.indices(graph, "b0")
    boundary_position_indices = gaussian.indices(graph, "b0")
    ancilla_momentum_indices = gaussian.momentum_indices(graph, "is_ancilla")
    boundary_after = gaussian.get_measured(
        covariance,
        boundary_position_indices,
        ancilla_momentum_indices,
    )
    boundary_nodes_map, unique_sorted_positions = technical_helpers.get_boundary_nodes_by_position(
        graph,
        kept_indices=kept_indices,
    )
    boundary_length = len(gaussian.position(boundary_after))
    return {
        "boundary_after": boundary_after,
        "boundary_nodes_map": boundary_nodes_map,
        "unique_sorted_positions": unique_sorted_positions,
        "boundary_length": int(boundary_length),
    }


def compact_relabel_graph(graph: nx.Graph) -> nx.Graph:
    mapping = {int(node): idx for idx, node in enumerate(sorted(int(node) for node in graph.nodes()))}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    return _ensure_graph_refs(relabeled)


def domain_boundary_nodes(graph: nx.Graph) -> list[int]:
    kind = str(graph.graph.get("knockout_case", "")).lower()
    if kind in {"mera", "hyperbolic"}:
        nodes = [node for node, data in graph.nodes(data=True) if bool(data.get("is_boundary", False))]
        return sorted(nodes, key=lambda node: int(graph.nodes[node]["position_in_boundary"]))
    if kind == "wormhole":
        nodes = [
            node
            for node, data in graph.nodes(data=True)
            if bool(data.get("is_boundary", False)) and bool(data.get("b0", False))
        ]
        return sorted(nodes, key=lambda node: int(graph.nodes[node]["position_in_boundary"]))
    raise ValueError(f"Unsupported knockout_case: {graph.graph.get('knockout_case')}")


def all_boundary_nodes(graph: nx.Graph) -> list[int]:
    nodes = [node for node, data in graph.nodes(data=True) if bool(data.get("is_boundary", False))]
    return sorted(nodes, key=lambda node: (int(graph.nodes[node].get("position_in_boundary", -1)), int(node)))


def normalize_boundary_positions(boundary_count: int, positions: list[int] | tuple[int, ...]) -> list[int]:
    boundary_count = int(boundary_count)
    if boundary_count <= 0:
        return []
    return sorted({int(position) % boundary_count for position in positions})


def selected_boundary_positions(boundary_count: int, region_length: int) -> list[int]:
    boundary_count = int(boundary_count)
    region_length = max(0, min(int(region_length), boundary_count))
    return list(range(region_length))


def periodic_interval_positions(boundary_count: int, start: int, length: int) -> list[int]:
    boundary_count = int(boundary_count)
    if boundary_count <= 0:
        return []
    return [int((int(start) + offset) % boundary_count) for offset in range(max(0, int(length)))]


def two_interval_positions(boundary_count: int, l1: int, l2: int, gap: int, *, start: int = 0) -> list[int]:
    first = periodic_interval_positions(boundary_count, int(start), int(l1))
    second_start = int(start) + int(l1) + int(gap)
    second = periodic_interval_positions(boundary_count, int(second_start), int(l2))
    return normalize_boundary_positions(boundary_count, first + second)


def selected_and_complement_boundary_nodes(graph: nx.Graph, region_length: int) -> tuple[list[int], list[int]]:
    domain_nodes = domain_boundary_nodes(graph)
    selected_positions = set(selected_boundary_positions(len(domain_nodes), region_length))
    selected = []
    for node in domain_nodes:
        if int(graph.nodes[node]["position_in_boundary"]) in selected_positions:
            selected.append(int(node))
    selected_set = set(selected)
    complement = [int(node) for node in all_boundary_nodes(graph) if int(node) not in selected_set]
    return selected, complement


def boundary_nodes_for_positions(graph: nx.Graph, positions: list[int] | tuple[int, ...]) -> list[int]:
    domain_nodes = domain_boundary_nodes(graph)
    selected_positions = set(normalize_boundary_positions(len(domain_nodes), positions))
    return [
        int(node)
        for node in domain_nodes
        if int(graph.nodes[node]["position_in_boundary"]) in selected_positions
    ]


def selected_and_complement_boundary_nodes_for_positions(
    graph: nx.Graph,
    positions: list[int] | tuple[int, ...],
) -> tuple[list[int], list[int]]:
    selected = boundary_nodes_for_positions(graph, positions)
    selected_set = set(int(node) for node in selected)
    complement = [int(node) for node in all_boundary_nodes(graph) if int(node) not in selected_set]
    return selected, complement


def entropy_for_region_length(analysis: Analysis, region_length: int, *, renyi2: bool = False) -> float:
    boundary_positions = [
        int(position)
        for position in selected_boundary_positions(int(analysis.boundary_length), int(region_length))
        if int(position) in analysis.boundary_nodes_map
    ]
    return float(
        gaussian.calculate_boundary_entropy(
            gaussian.to_inter(analysis.boundary_after_bulk_measurement),
            boundary_positions,
            analysis.boundary_nodes_map,
            renyi=bool(renyi2),
            alpha=2,
        )
    )


def entropy_for_region_length_from_payload(payload: dict, region_length: int, *, renyi2: bool = False) -> float:
    boundary_length = int(payload["boundary_length"])
    boundary_positions = [
        int(position)
        for position in selected_boundary_positions(boundary_length, int(region_length))
        if int(position) in payload["boundary_nodes_map"]
    ]
    return float(
        gaussian.calculate_boundary_entropy(
            gaussian.to_inter(payload["boundary_after"]),
            boundary_positions,
            payload["boundary_nodes_map"],
            renyi=bool(renyi2),
            alpha=2,
        )
    )


def entropy_for_boundary_positions_from_payload(
    payload: dict,
    positions: list[int] | tuple[int, ...],
    *,
    renyi2: bool = False,
) -> float:
    boundary_length = int(payload["boundary_length"])
    boundary_positions = [
        int(position)
        for position in normalize_boundary_positions(boundary_length, positions)
        if int(position) in payload["boundary_nodes_map"]
    ]
    return float(
        gaussian.calculate_boundary_entropy(
            gaussian.to_inter(payload["boundary_after"]),
            boundary_positions,
            payload["boundary_nodes_map"],
            renyi=bool(renyi2),
            alpha=2,
        )
    )


def _run_with_symeig_warning_capture(func, /, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    captured = buffer.getvalue()
    if captured:
        print(captured, end="")
    return result, (SYMEIG_WARNING_TEXT in captured)


def node_support_separator_details(
    graph: nx.Graph,
    region_length: int,
    *,
    forbid_selected_boundary_support: bool = False,
) -> dict:
    selected_boundary_nodes, complement_boundary_nodes = selected_and_complement_boundary_nodes(graph, region_length)
    return node_support_separator_details_for_boundary_nodes(
        graph,
        selected_boundary_nodes=selected_boundary_nodes,
        complement_boundary_nodes=complement_boundary_nodes,
        forbid_selected_boundary_support=forbid_selected_boundary_support,
    )


def node_support_separator_details_for_positions(
    graph: nx.Graph,
    positions: list[int] | tuple[int, ...],
    *,
    forbid_selected_boundary_support: bool = False,
) -> dict:
    selected_boundary_nodes, complement_boundary_nodes = selected_and_complement_boundary_nodes_for_positions(graph, positions)
    return node_support_separator_details_for_boundary_nodes(
        graph,
        selected_boundary_nodes=selected_boundary_nodes,
        complement_boundary_nodes=complement_boundary_nodes,
        forbid_selected_boundary_support=forbid_selected_boundary_support,
    )


def node_support_separator_details_for_boundary_nodes(
    graph: nx.Graph,
    *,
    selected_boundary_nodes: list[int] | tuple[int, ...],
    complement_boundary_nodes: list[int] | tuple[int, ...],
    forbid_selected_boundary_support: bool = False,
) -> dict:
    selected_boundary_nodes = set(int(node) for node in selected_boundary_nodes)
    complement_boundary_nodes = set(int(node) for node in complement_boundary_nodes)

    source = "__source__"
    sink = "__sink__"
    big = 1_000_000.0
    split = nx.DiGraph()
    split.add_node(source)
    split.add_node(sink)

    def node_in(node: int):
        return ("in", int(node))

    def node_out(node: int):
        return ("out", int(node))

    for node in graph.nodes():
        capacity = big if (forbid_selected_boundary_support and int(node) in selected_boundary_nodes) else 1.0
        split.add_edge(node_in(node), node_out(node), capacity=capacity)

    for left, right in graph.edges():
        split.add_edge(node_out(left), node_in(right), capacity=big)
        split.add_edge(node_out(right), node_in(left), capacity=big)

    for node in selected_boundary_nodes:
        split.add_edge(source, node_in(node), capacity=big)
    for node in complement_boundary_nodes:
        split.add_edge(node_out(node), sink, capacity=big)

    _, partition = nx.minimum_cut(split, source, sink, capacity="capacity")
    source_side, _ = partition
    support_nodes = sorted(
        int(node)
        for node in graph.nodes()
        if node_in(node) in source_side and node_out(node) not in source_side
    )

    cut_edges = []
    support_set = set(support_nodes)
    for node in support_nodes:
        for neighbor in graph.neighbors(node):
            if int(neighbor) in support_set:
                continue
            cut_edges.append(tuple(sorted((int(node), int(neighbor)))))
    cut_edges = sorted(set(cut_edges))

    return {
        "support_count": int(len(support_nodes)),
        "support_nodes": support_nodes,
        "cut_edges": cut_edges,
        "selected_boundary_nodes": sorted(selected_boundary_nodes),
        "complement_boundary_nodes": sorted(complement_boundary_nodes),
        "domain_boundary_nodes": [int(node) for node in domain_boundary_nodes(graph)],
    }


def knockout_map_for_boundary_positions(
    config: TreeWormholeConfig,
    *,
    positions: list[int] | tuple[int, ...],
    nodes_to_knock: list[int] | None = None,
) -> dict:
    graph = build_graph(config)
    base_payload = boundary_after_measurement_payload(
        graph,
        couplingtime=float(config.couplingtime),
        presqueeze=float(config.presqueeze),
    )
    base_entropy, base_warning = _run_with_symeig_warning_capture(
        entropy_for_boundary_positions_from_payload,
        base_payload,
        positions,
        renyi2=bool(config.renyi2),
    )

    if nodes_to_knock is None:
        nodes_to_knock = sorted(int(node) for node in graph.nodes())
    else:
        nodes_to_knock = [int(node) for node in nodes_to_knock]

    knockout_map = {}
    symeig_warning_occurred = bool(base_warning)
    for node in nodes_to_knock:
        knockout_graph = graph.copy()
        knockout_graph.remove_node(int(node))
        knockout_graph = compact_relabel_graph(knockout_graph)
        knockout_payload = boundary_after_measurement_payload(
            knockout_graph,
            couplingtime=float(config.couplingtime),
            presqueeze=float(config.presqueeze),
        )
        knockout_entropy, knockout_warning = _run_with_symeig_warning_capture(
            entropy_for_boundary_positions_from_payload,
            knockout_payload,
            positions,
            renyi2=bool(config.renyi2),
        )
        symeig_warning_occurred = bool(symeig_warning_occurred or knockout_warning)
        knockout_map[int(node)] = float(base_entropy - knockout_entropy)

    return {
        "graph": graph,
        "analysis": None,
        "base_entropy": float(base_entropy),
        "positions": normalize_boundary_positions(len(domain_boundary_nodes(graph)), positions),
        "knockout_map": knockout_map,
        "symeig_warning_occurred": bool(symeig_warning_occurred),
    }


def knockout_map_for_region(
    config: TreeWormholeConfig,
    *,
    region_length: int,
    nodes_to_knock: list[int] | None = None,
) -> dict:
    graph = build_graph(config)
    base_payload = boundary_after_measurement_payload(
        graph,
        couplingtime=float(config.couplingtime),
        presqueeze=float(config.presqueeze),
    )
    base_entropy = entropy_for_region_length_from_payload(base_payload, region_length, renyi2=bool(config.renyi2))

    if nodes_to_knock is None:
        nodes_to_knock = sorted(int(node) for node in graph.nodes())
    else:
        nodes_to_knock = [int(node) for node in nodes_to_knock]

    knockout_map = {}
    for node in nodes_to_knock:
        knockout_graph = graph.copy()
        knockout_graph.remove_node(int(node))
        knockout_graph = compact_relabel_graph(knockout_graph)
        knockout_payload = boundary_after_measurement_payload(
            knockout_graph,
            couplingtime=float(config.couplingtime),
            presqueeze=float(config.presqueeze),
        )
        knockout_entropy = entropy_for_region_length_from_payload(
            knockout_payload,
            region_length,
            renyi2=bool(config.renyi2),
        )
        knockout_map[int(node)] = float(base_entropy - knockout_entropy)

    return {
        "graph": graph,
        "analysis": None,
        "base_entropy": float(base_entropy),
        "region_length": int(region_length),
        "knockout_map": knockout_map,
    }


def sweep_two_interval_gap(
    config: TreeWormholeConfig,
    *,
    l1: int,
    l2: int,
    start: int = 0,
    gaps: list[int] | tuple[int, ...] | None = None,
    forbid_selected_boundary_support: bool = False,
) -> dict:
    graph = build_graph(config)
    payload = boundary_after_measurement_payload(
        graph,
        couplingtime=float(config.couplingtime),
        presqueeze=float(config.presqueeze),
    )
    boundary_count = len(domain_boundary_nodes(graph))
    max_gap = boundary_count - int(l1) - int(l2)
    if max_gap < 0:
        raise ValueError(f"Interval lengths ({l1}, {l2}) are too large for boundary count {boundary_count}.")

    if gaps is None:
        gap_values = np.arange(max_gap + 1, dtype=int)
    else:
        gap_values = np.asarray(
            sorted({max(0, min(int(value), max_gap)) for value in gaps}),
            dtype=int,
        )

    selected_positions_by_gap = [
        two_interval_positions(boundary_count, int(l1), int(l2), int(gap), start=int(start))
        for gap in gap_values
    ]
    symeig_warning_occurred = False
    entropy_values = []
    for positions in selected_positions_by_gap:
        entropy_value, entropy_warning = _run_with_symeig_warning_capture(
            entropy_for_boundary_positions_from_payload,
            payload,
            positions,
            renyi2=bool(config.renyi2),
        )
        entropy_values.append(float(entropy_value))
        symeig_warning_occurred = bool(symeig_warning_occurred or entropy_warning)
    entropies = np.asarray(entropy_values, dtype=float)
    s_a1, s_a1_warning = _run_with_symeig_warning_capture(
        entropy_for_boundary_positions_from_payload,
        payload,
        periodic_interval_positions(boundary_count, int(start), int(l1)),
        renyi2=bool(config.renyi2),
    )
    symeig_warning_occurred = bool(symeig_warning_occurred or s_a1_warning)
    s_a2_values = []
    for gap in gap_values:
        s_a2_value, s_a2_warning = _run_with_symeig_warning_capture(
            entropy_for_boundary_positions_from_payload,
            payload,
            periodic_interval_positions(boundary_count, int(start) + int(l1) + int(gap), int(l2)),
            renyi2=bool(config.renyi2),
        )
        s_a2_values.append(float(s_a2_value))
        symeig_warning_occurred = bool(symeig_warning_occurred or s_a2_warning)
    s_a2 = np.asarray(s_a2_values, dtype=float)
    node_support_details = [
        node_support_separator_details_for_positions(
            graph,
            positions,
            forbid_selected_boundary_support=forbid_selected_boundary_support,
        )
        for positions in selected_positions_by_gap
    ]
    node_support_counts = np.asarray([details["support_count"] for details in node_support_details], dtype=int)
    return {
        "graph": graph,
        "payload": payload,
        "boundary_count": int(boundary_count),
        "l1": int(l1),
        "l2": int(l2),
        "start": int(start),
        "gaps": gap_values,
        "selected_positions_by_gap": selected_positions_by_gap,
        "entropies": entropies,
        "s_a1": float(s_a1),
        "s_a2": s_a2,
        "mutual_information": float(s_a1) + s_a2 - entropies,
        "node_support_counts": node_support_counts,
        "node_support_details": node_support_details,
        "symeig_warning_occurred": bool(symeig_warning_occurred),
    }


def sweep_graph(config: TreeWormholeConfig, *, region_lengths: list[int] | None = None) -> dict:
    graph, analysis = build_analysis(config)
    boundary_count = int(analysis.boundary_length)
    if region_lengths is None:
        region_sizes = np.arange(boundary_count + 1, dtype=int)
    else:
        region_sizes = np.asarray(sorted({max(0, min(int(v), boundary_count)) for v in region_lengths}), dtype=int)
    entropies = np.asarray(
        [entropy_for_region_length(analysis, int(region_length), renyi2=bool(config.renyi2)) for region_length in region_sizes],
        dtype=float,
    )
    node_support_details = [node_support_separator_details(graph, int(region_length)) for region_length in region_sizes]
    node_support_counts = np.asarray([details["support_count"] for details in node_support_details], dtype=int)
    return {
        "graph": graph,
        "analysis": analysis,
        "region_sizes": region_sizes,
        "entropies": entropies,
        "node_support_counts": node_support_counts,
        "node_support_details": node_support_details,
    }

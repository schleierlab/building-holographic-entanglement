"""Hyperbolic geometry and hyperbolic tiling graph generation.

This module provides functions for:
- Hyperbolic geometry operations in the Poincare disk model
- Generation of regular hyperbolic `{p,q}` tilings
- The original vertex-growth `{3,q}` generator as a reference implementation
"""

import networkx as nx
import numpy as np
import collections
import math
import warnings
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy
from pydantic import ValidationError

# Import from validation module
from .validation import check_labels


# --- Hyperbolic Geometry Helpers (Poincaré Disk) ---

def poincare_dist(z1: complex, z2: complex) -> float:
    """Calculate hyperbolic distance between two points in Poincaré disk."""
    abs_z1_sq = abs(z1)**2
    abs_z2_sq = abs(z2)**2
    abs_diff_sq = abs(z1 - z2)**2
    boundary_epsilon = 1e-9
    if abs_z1_sq >= 1.0 - boundary_epsilon or abs_z2_sq >= 1.0 - boundary_epsilon:
        if abs(abs_z1_sq - 1.0) < boundary_epsilon and abs(abs_z2_sq - 1.0) < boundary_epsilon and abs_diff_sq < boundary_epsilon**2:
             return 0.0
        return float('inf')
    denom = (1 - abs_z1_sq) * (1 - abs_z2_sq)
    if abs(denom) < 1e-15:
        return float('inf') if abs_diff_sq > 1e-9 else 0.0
    arg = 1 + 2 * abs_diff_sq / denom
    if arg < 1.0:
        if arg > 1.0 - 1e-12: return 0.0
        return float('nan')
    dist = np.arccosh(arg)
    return dist if dist > 1e-12 else 0.0

def mobius_transform(z: complex, a: complex, b: complex, c: complex, d: complex) -> complex:
    """Applies Mobius transform (az+b)/(cz+d)."""
    denominator = (c * z + d)
    if abs(denominator) < 1e-15:
         inf_point = complex(1e16, 0)
         return inf_point * (a*z+b) if abs(a*z+b) > 1e-9 else inf_point
    return (a * z + b) / denominator

def isometry_translate_origin_to_w(z: complex, w: complex) -> complex:
    """Isometry mapping origin to w. Maps z."""
    if abs(w) >= 1.0 - 1e-12: raise ValueError(f"Translation target w={w} must be strictly inside the unit disk.")
    return mobius_transform(z, 1, w, w.conjugate(), 1)

def isometry_translate_w_to_origin(z: complex, w: complex) -> complex:
    """Isometry mapping w to origin. Maps z."""
    if abs(w) >= 1.0 - 1e-12: raise ValueError(f"Translation source w={w} must be strictly inside the unit disk.")
    return mobius_transform(z, 1, -w, -w.conjugate(), 1)

def find_neighbor_coords(center_coord: complex, parent_coord: Optional[complex], edge_dist: float, num_neighbors: int) -> List[complex]:
    """
    Calculates coordinates of neighbors around center_coord in Poincaré disk.
    Uses parent_coord (if provided) to orient the neighbors correctly.
    """
    neighbors = []
    r = math.tanh(edge_dist / 2.0)
    if r < 1e-12: raise ValueError("Edge distance is too small, leading to r=0.")

    if abs(center_coord) < 1e-9: # Center is origin
        angle_step = 2 * math.pi / num_neighbors
        for i in range(num_neighbors):
            angle = i * angle_step
            neighbor = r * complex(math.cos(angle), math.sin(angle))
            neighbors.append(neighbor)
    else:
        if parent_coord is None: raise ValueError("Parent coordinate required.")
        if abs(center_coord - parent_coord) < 1e-9: raise ValueError("Center and parent too close.")
        T = lambda z: isometry_translate_w_to_origin(z, center_coord)
        T_inv = lambda z: isometry_translate_origin_to_w(z, center_coord)
        parent_transformed = T(parent_coord)
        parent_transformed_angle = np.angle(parent_transformed)
        angle_step = 2 * math.pi / num_neighbors
        for i in range(num_neighbors):
            angle = parent_transformed_angle + i * angle_step # Use corrected angle
            neighbor_at_origin = r * complex(math.cos(angle), math.sin(angle))
            neighbor = T_inv(neighbor_at_origin)
            neighbors.append(neighbor)
    return neighbors


# --- Hyperbolic Tiling Generation Functions ---


def _coordinate_key(point: complex, *, decimals: int = 9) -> tuple[float, float]:
    return (round(float(point.real), decimals), round(float(point.imag), decimals))


def generate_hyperbolic_tiling_with_hypertiling(
    p: int,
    q: int,
    depth: int,
    *,
    center: str = "vertex",
) -> nx.Graph:
    """Generate a regular hyperbolic vertex graph using ``hypertiling``.

    ``hypertiling`` constructs the regular polygon cells and their Poincare-disk
    coordinates. This adapter merges coincident polygon vertices and records
    edge face incidence so the finite patch's outer perimeter can be identified
    without geometric or degree heuristics.
    """

    if not isinstance(p, int) or not isinstance(q, int) or p < 3 or q < 3:
        raise ValueError("p and q must be integers greater than or equal to 3.")
    if 1.0 / p + 1.0 / q >= 0.5:
        raise ValueError(f"{{{p},{q}}} is not a hyperbolic regular tiling.")
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer.")
    if center not in {"vertex", "cell"}:
        raise ValueError("center must be either 'vertex' or 'cell'.")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Failed to import numba.*",
                module="hypertiling.check_numba",
            )
            import hypertiling
    except ImportError as exc:
        raise ImportError(
            "Hyperbolic regular patches require the 'hypertiling' package."
        ) from exc

    tiling = hypertiling.HyperbolicTiling(p, q, depth, center=center)
    graph = nx.Graph()
    vertex_ids: dict[tuple[float, float], int] = {}

    for polygon in tiling:
        polygon_vertices = tuple(complex(point) for point in polygon[1:])
        if len(polygon_vertices) != p:
            raise RuntimeError(
                f"hypertiling returned {len(polygon_vertices)} vertices for a {p}-gon."
            )

        face_vertices = []
        for point in polygon_vertices:
            key = _coordinate_key(point, decimals=10)
            node = vertex_ids.get(key)
            if node is None:
                node = len(vertex_ids)
                vertex_ids[key] = node
                graph.add_node(
                    node,
                    coord=point,
                    ambient_degree=q,
                )
            face_vertices.append(node)

        for index, node in enumerate(face_vertices):
            neighbor = face_vertices[(index + 1) % p]
            if graph.has_edge(node, neighbor):
                graph.edges[node, neighbor]["face_incidence"] += 1
            else:
                graph.add_edge(node, neighbor, face_incidence=1)

    graph.graph["ambient_degree"] = q
    graph.graph["generator"] = "hypertiling_srs"
    graph.graph["hypertiling_version"] = getattr(hypertiling, "__version__", "unknown")
    graph.graph["hypertiling_cell_count"] = len(tiling)
    graph.graph["hypertiling_center"] = center
    graph.graph["schlafli_symbol"] = f"{{{p},{q}}}"
    return graph


def pentagonal_tiling(depth, validate=True):
    """
    Create a graph approximating an order-5 square tiling {4,5} up to a certain depth.
    Each vertex should have degree 5, and minimal cycles should be squares.
    Nodes are added layer by layer (BFS). Attempts to connect nodes to form cycles
    approximating squares, aiming for degree 5, and avoiding triangles.

    Note: This uses heuristics and may not perfectly represent the {4,5} tiling,
    especially regarding consistent square formation and degree regularity at the edges.
    The function name remains `pentagonal_tiling` as requested, but implements {4,5}.
    """
    G = nx.Graph()
    if depth < 0:
        raise ValueError("Depth must be non-negative")

    if depth == 0:
        G.add_node(0, depth=0)
        nodes_at_depth = {0: {0}}
        node_counter = 1
    else:
        G.add_node(0, depth=0)
        queue = collections.deque([(0, 0)]) # node, depth
        nodes_at_depth = {0: {0}}
        node_counter = 1
        visited_for_expansion = {0} # Nodes whose children have been initially added
        max_nodes = 10000 # Safety limit

        # --- Build outwards (BFS-like expansion) and connect ---
        processed_nodes = set()
        while queue:
            if node_counter > max_nodes:
                print(f"Warning: Node limit ({max_nodes}) reached during generation.")
                break

            u, d = queue.popleft()
            processed_nodes.add(u)

            if d >= depth:
                continue

            # --- Add Children ---
            # Aim to add enough children to potentially reach degree 5 later.
            # Root needs 5 neighbors. Others need 4 more (excluding parent).
            num_new_children_target = 5 if d == 0 else 4

            children_added_this_step = []
            for _ in range(num_new_children_target):
                # Stop adding if parent node u already reached target degree
                if G.degree(u) >= 5: break
                if node_counter > max_nodes: break

                v = node_counter
                G.add_node(v, depth=d + 1)
                G.add_edge(u, v)
                children_added_this_step.append(v)

                if d + 1 not in nodes_at_depth:
                    nodes_at_depth[d + 1] = set()
                nodes_at_depth[d + 1].add(v)

                # Add to queue for next layer's expansion if within depth
                if v not in visited_for_expansion and (d + 1) < depth:
                     queue.append((v, d + 1))
                     visited_for_expansion.add(v)
                node_counter += 1
            if node_counter > max_nodes: break

            # --- Connect New Children (Form Squares Heuristically) ---
            # Each child v needs degree 5: 1 parent (u), 2 siblings, 2 cousins.

            # 1. Connect siblings linearly (provides 2 neighbors for inner children)
            num_children = len(children_added_this_step)
            for i in range(num_children):
                child = children_added_this_step[i]
                if G.degree(child) >= 5: continue

                # Connect to previous sibling
                if i > 0:
                    prev_sibling = children_added_this_step[i-1]
                    if G.degree(prev_sibling) < 5 and not G.has_edge(child, prev_sibling):
                         # Check for triangle (child, prev_sibling, u) - impossible here
                         G.add_edge(child, prev_sibling)

                # Connect to next sibling
                if i < num_children - 1:
                     next_sibling = children_added_this_step[i+1]
                     # Check degree of child again as it might have changed
                     if G.degree(child) < 5 and G.degree(next_sibling) < 5 and not G.has_edge(child, next_sibling):
                          G.add_edge(child, next_sibling)

            # 2. Connect to cousins (children of parent's siblings)
            if d >= 0: # Need a parent to have siblings
                neighbors_of_u = list(G.neighbors(u))
                parent = next((n for n in neighbors_of_u if G.nodes[n]['depth'] == d - 1), None)

                if parent is not None:
                    # Find siblings of u
                    grandparent = next((gp for gp in G.neighbors(parent) if G.nodes[gp]['depth'] == d - 2), None)
                    siblings_of_u = {s for s in G.neighbors(parent) if G.nodes[s]['depth'] == d and s != u} if parent is not None else set()

                    # Find potential partners: children of siblings of u (cousins)
                    potential_partners = set()
                    for s in siblings_of_u:
                        potential_partners.update(p for p in G.neighbors(s) if G.nodes[p]['depth'] == d + 1)

                    # Create a list of available partners (degree < 5)
                    available_partners = list(p for p in potential_partners if G.degree(p) < 5)
                    partner_pool_idx = 0 # Index for iterating through available_partners pool

                    for child in children_added_this_step:
                        needed_connections = 5 - G.degree(child)
                        connections_made_to_cousins = 0
                        current_partner_idx = partner_pool_idx # Start searching from where the last child left off? Or reset?
                                                        # Let's try continuing, might distribute connections better.

                        search_attempts = 0 # Limit search to avoid infinite loops in dense areas
                        max_search_attempts = len(available_partners) * 2

                        while connections_made_to_cousins < needed_connections and search_attempts < max_search_attempts:
                            if not available_partners: break # No more partners

                            current_partner_idx = current_partner_idx % len(available_partners) # Wrap around index
                            partner = available_partners[current_partner_idx]
                            search_attempts += 1

                            # Check validity
                            if not G.has_edge(child, partner) and G.degree(partner) < 5:
                                # Check for triangle: child-partner-common_neighbor
                                common_neighbors = list(nx.common_neighbors(G, child, partner))
                                if not common_neighbors:
                                    G.add_edge(child, partner)
                                    connections_made_to_cousins += 1
                                    needed_connections -= 1 # Update needed connections
                                    # Check if partner is now full
                                    if G.degree(partner) >= 5:
                                        available_partners.pop(current_partner_idx)
                                        # Index effectively moved forward, no need to increment current_partner_idx here
                                        if not available_partners: break # Pool empty
                                        # Ensure index stays valid after pop
                                        current_partner_idx = current_partner_idx % len(available_partners)
                                        continue # Skip incrementing index
                                    # else: partner still available
                                else: # Triangle found, skip this partner
                                    pass # Just move to next index
                            # else: Edge exists or partner full, skip

                            current_partner_idx += 1 # Move to next potential partner in the pool

                        # Update the main pool index for the next child to start from
                        partner_pool_idx = current_partner_idx

    # --- Final Degree Check & Cleanup (Optional) ---
    # Could add a pass here, but risky.

    # --- Determine boundary and Labeling ---
    # (Keep existing labeling logic, it should adapt)
    if not G.nodes:
        actual_max_depth = -1
        boundary_nodes = set()
    else:
        actual_max_depth = 0
        for node in G.nodes:
            node_depth = G.nodes[node].get('depth', -1)
            if node_depth > actual_max_depth:
                actual_max_depth = node_depth

        boundary_nodes = {n for n in G.nodes if G.nodes[n].get('depth') == actual_max_depth}
        if not boundary_nodes and actual_max_depth > 0:
            actual_max_depth -= 1
            boundary_nodes = {n for n in G.nodes if G.nodes[n].get('depth') == actual_max_depth}

    boundary_node_list = sorted(list(boundary_nodes))
    boundary_pos_map = {node: i for i, node in enumerate(boundary_node_list)}

    for node in G.nodes():
        node_data = G.nodes[node]
        node_depth = node_data.get('depth', 0)
        is_boundary = node in boundary_nodes
        node_data['is_ancilla'] = not is_boundary
        node_data['is_boundary'] = is_boundary
        node_data['b0'] = is_boundary

        if is_boundary:
            node_data['boundary_index'] = 0
            node_data['position_in_boundary'] = boundary_pos_map.get(node, -1)

        if not is_boundary:
             for i in range(10):
                 node_data.setdefault(f'b{i}', False)
                 if node_data[f'b{i}'] is True: node_data[f'b{i}'] = False
        else:
             correct_boundary_index = node_data.get('boundary_index', 0)
             for i in range(10):
                 node_data.setdefault(f'b{i}', False)
                 node_data[f'b{i}'] = (i == correct_boundary_index)

    G.graph["number_of_boundaries"] = 1 if boundary_nodes else 0
    G.graph["max_depth"] = actual_max_depth
    G.graph["periodic"] = False
    G.graph["subdivided"] = False

    graph_attrs_ref = G.graph
    for node in G.nodes():
        G.nodes[node]["graph"] = graph_attrs_ref

    if validate:
        try:
            check_labels(G)
        except ValidationError as e:
            try:
                error_info = e.errors()[0]
                loc = error_info.get('loc', ['unknown'])[0]
                msg = error_info.get('msg', 'Unknown validation error')
                print(f"Validation failed for {__name__}(depth={depth}):") # Use function name
                print(f"Node {loc} failed: {msg}")
            except (IndexError, KeyError, AttributeError):
                 print(f"Validation failed with complex error: {e}")
            print("Warning: Graph generated, but failed validation. Structure or labeling may be incorrect.")
        except Exception as e:
            print(f"An unexpected error occurred during validation: {e}")

    return G


def generate_order5_square_tiling(depth):
    """Generate {4,5} tiling using group theory approach with relators.

    {4,5}-tiling: squares (order‑4) meeting 5 around each vertex.
    Uses word reduction in the group presentation.
    """
    # {4,5}-tiling: squares (order‑4) meeting 5 around each vertex
    p, q = 4, 5
    gens = ['s','S','t','T']
    inv = {'s':'S','S':'s','t':'T','T':'t'}

    # precompute all relator‐words we want to kill
    relators = []
    # s^4 and its inverse (same for S)
    relators += [['s']*p, ['S']*p]
    # t^5 and its inverse (same for T)
    relators += [['t']*q, ['T']*q]
    # (s t)^2, (t s)^2, and their inverses
    relators += [
        ['s','t']*2,
        ['t','s']*2,
        ['S','T']*2,
        ['T','S']*2,
    ]

    def reduce_word(word):
        changed = True
        while changed:
            changed = False
            # 1) cancel a A or A a
            for i in range(len(word)-1):
                if inv[word[i]] == word[i+1]:
                    del word[i:i+2]
                    changed = True
                    break
            if changed:
                continue

            # 2) apply any relator
            for rel in relators:
                L = len(rel)
                for i in range(len(word)-L+1):
                    if word[i:i+L] == rel:
                        del word[i:i+L]
                        changed = True
                        break
                if changed:
                    break

        return tuple(word)

    G = nx.Graph()
    identity = ()
    G.add_node(identity)
    visited = {identity}
    frontier = {identity}

    for _ in range(depth):
        new_frontier = set()
        for g in frontier:
            for gen in gens:
                w = list(g) + [gen]
                r = reduce_word(w)
                G.add_node(r)
                G.add_edge(g, r)
                if r not in visited:
                    visited.add(r)
                    new_frontier.add(r)
        frontier = new_frontier

    return G


def _hyperbolic_edge_distance(p: int, q: int) -> float:
    """Return the hyperbolic edge length for a regular `{p,q}` tiling."""

    cosh_d_val = (
        math.cos(2.0 * math.pi / p) + math.cos(math.pi / q) ** 2
    ) / (math.sin(math.pi / q) ** 2)
    if cosh_d_val < 1.0 - 1e-12:
        raise ValueError(f"Invalid edge distance for {{{p},{q}}}: cosh(d)={cosh_d_val}")
    return math.acosh(max(1.0, cosh_d_val))


def generate_hyperbolic_tiling_3_q_geometric_with_raw(
    q_value: int,
    depth: int,
    epsilon: float = 1e-5,
    merge_last_layer: bool = True,
    prune_last_layer: bool = True,
    merge_degree_one_boundary: bool = True,
) -> Tuple[Optional[nx.Graph], List[Dict]]:
    """
    Generate a `{3,q}` hyperbolic tiling vertex graph in the Poincare disk.

    This is retained as the original reference implementation. The production
    regular-tiling API uses :func:`generate_hyperbolic_tiling_with_hypertiling`.
    The construction mirrors the existing `{3,8}` implementation: breadth-first
    growth from the origin, geometric vertex identification by hyperbolic
    proximity, optional pruning of the outer layer, and optional merging of
    dangling degree-1 boundary nodes.
    """

    if q_value < 7:
        raise ValueError("{3,q} is hyperbolic only for q >= 7.")
    if depth < 0:
        raise ValueError("Depth cannot be negative")
    if depth == 0:
        graph = nx.Graph()
        graph.add_node(0, depth=0, coord=complex(0, 0))
        return graph, []
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy required.")
        return None, []

    graph = nx.Graph()
    node_count = 0
    node_info: Dict[int, Dict[str, Any]] = {}
    coords_list: List[Tuple[complex, int]] = []
    raw_neighbor_points: List[Dict] = []
    processed_edges = set()

    def add_edge_geo(u_id, v_id):
        if u_id == v_id:
            return
        if u_id not in graph or v_id not in graph:
            return
        edge_tuple = tuple(sorted((u_id, v_id)))
        if edge_tuple not in processed_edges:
            graph.add_edge(u_id, v_id)
            processed_edges.add(edge_tuple)

    try:
        edge_dist = _hyperbolic_edge_distance(3, q_value)
    except ValueError as exc:
        print(f"Error calc edge dist for {{3,{q_value}}}: {exc}")
        return None, []
    if edge_dist < 1e-9:
        print("Warning: Edge dist near zero.")
        return None, []

    origin_id = node_count
    node_count += 1
    origin_coord = complex(0, 0)
    node_info[origin_id] = {"id": origin_id, "depth": 0, "parent": -1, "coord": origin_coord}
    coords_list.append((origin_coord, origin_id))
    graph.add_node(origin_id, depth=0, coord=origin_coord)
    frontier = collections.deque([origin_id])
    processed_count = 0
    nodes_created_last_layer = 0

    print(
        f"Starting Geometric BFS for {{3,{q_value}}} "
        f"(MergeLastLayer={merge_last_layer}, PruneLastLayer={prune_last_layer}) "
        f"up to depth {depth}..."
    )

    while frontier:
        u_id = frontier.popleft()
        if u_id not in node_info:
            continue
        u_data = node_info[u_id]
        u_coord = u_data["coord"]
        u_depth = u_data["depth"]
        u_parent_id = u_data["parent"]
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count} nodes...")
        if u_depth >= depth:
            continue

        parent_coord = node_info[u_parent_id]["coord"] if u_parent_id != -1 else None

        try:
            potential_neighbor_coords = find_neighbor_coords(
                u_coord,
                parent_coord,
                edge_dist,
                q_value,
            )
        except Exception as exc:
            print(f"Error finding neighbors for {u_id}: {exc}")
            continue

        for v_raw in potential_neighbor_coords:
            raw_neighbor_points.append(
                {
                    "parent_id": u_id,
                    "parent_coord": u_coord,
                    "parent_depth": u_depth,
                    "raw_coord": v_raw,
                }
            )

        parent_candidate_coord = None
        if parent_coord is not None:
            min_dist_to_parent = float("inf")
            for coord in potential_neighbor_coords:
                if abs(coord) < 1.0:
                    dist = poincare_dist(coord, parent_coord)
                    if not math.isnan(dist) and dist < min_dist_to_parent:
                        min_dist_to_parent = dist
                        parent_candidate_coord = coord

        for v_coord_raw in potential_neighbor_coords:
            if (
                parent_candidate_coord is not None
                and poincare_dist(v_coord_raw, parent_candidate_coord) < epsilon
            ):
                continue

            if abs(v_coord_raw) >= 1.0:
                if abs(v_coord_raw) < 1.0 + 1e-9:
                    v_coord_raw = v_coord_raw / abs(v_coord_raw) * (1.0 - 1e-12)
                else:
                    continue

            min_dist_found = float("inf")
            closest_existing_id = -1
            for existing_coord, existing_id in coords_list:
                if existing_id == u_id:
                    continue
                dist_check = poincare_dist(v_coord_raw, existing_coord)
                if not math.isnan(dist_check) and dist_check < min_dist_found:
                    min_dist_found = dist_check
                    closest_existing_id = existing_id

            should_merge = closest_existing_id != -1 and min_dist_found < epsilon
            v_potential_depth = u_depth + 1
            allow_merge = merge_last_layer or (v_potential_depth < depth)

            if should_merge and allow_merge:
                add_edge_geo(u_id, closest_existing_id)
            else:
                v_id = node_count
                node_count += 1
                if not allow_merge and v_potential_depth == depth and should_merge:
                    nodes_created_last_layer += 1
                node_info[v_id] = {
                    "id": v_id,
                    "depth": v_potential_depth,
                    "parent": u_id,
                    "coord": v_coord_raw,
                }
                coords_list.append((v_coord_raw, v_id))
                graph.add_node(v_id, depth=v_potential_depth, coord=v_coord_raw)
                add_edge_geo(u_id, v_id)
                if v_potential_depth < depth:
                    frontier.append(v_id)

    print(
        f"\nGeometric BFS complete. Nodes before pruning: {graph.number_of_nodes()}, "
        f"Edges: {graph.number_of_edges()}"
    )
    if not merge_last_layer:
        print(f"Debug: Created {nodes_created_last_layer} distinct nodes at max depth instead of merging.")

    nodes_at_max_depth = []
    if depth >= 0:
        nodes_at_max_depth = [
            nid
            for nid, data in node_info.items()
            if nid in graph and data.get("depth") == depth
        ]

    if prune_last_layer and nodes_at_max_depth:
        print(f"Pruning {len(nodes_at_max_depth)} nodes at depth {depth}...")
        graph.remove_nodes_from(nodes_at_max_depth)
        print(
            f"Graph after pruning: Nodes: {graph.number_of_nodes()}, "
            f"Edges: {graph.number_of_edges()}"
        )

    if merge_degree_one_boundary:
        print("Attempting to merge degree-1 boundary nodes...")
        neighbors_of_degree_one = collections.defaultdict(list)
        for node in list(graph.nodes()):
            if graph.degree(node) == 1:
                if node not in graph:
                    continue
                neighbor = list(graph.neighbors(node))[0]
                neighbors_of_degree_one[neighbor].append(node)

        nodes_merged_count = 0
        for neighbor, degree_one_nodes in neighbors_of_degree_one.items():
            if len(degree_one_nodes) > 1:
                node_to_keep = min(degree_one_nodes)
                nodes_to_merge = [n for n in degree_one_nodes if n != node_to_keep]
                for node_to_merge in nodes_to_merge:
                    if node_to_merge in graph:
                        graph.remove_node(node_to_merge)
                        node_info.pop(node_to_merge, None)
                        coords_list = [
                            (coord, nid) for coord, nid in coords_list if nid != node_to_merge
                        ]
                        nodes_merged_count += 1

        if nodes_merged_count > 0:
            print(f"Merged {nodes_merged_count} degree-1 nodes.")
            print(
                f"Graph after merging degree-1 nodes: Nodes: {graph.number_of_nodes()}, "
                f"Edges: {graph.number_of_edges()}"
            )

    final_nodes = list(graph.nodes())
    if not final_nodes:
        actual_max_depth = -1
        boundary_nodes = set()
    else:
        depths = [node_info[nid]["depth"] for nid in final_nodes if nid in node_info]
        actual_max_depth = max(depths) if depths else -1
        boundary_nodes = {
            nid
            for nid in final_nodes
            if nid in node_info
            and node_info[nid]["depth"] == actual_max_depth
            and graph.degree(nid) < q_value
        }

    boundary_node_list = sorted(boundary_nodes)
    boundary_pos_map = {node: i for i, node in enumerate(boundary_node_list)}

    for node in final_nodes:
        if node not in node_info:
            continue
        node_data = graph.nodes[node]
        node_depth = node_info[node].get("depth", -1)
        is_boundary = node in boundary_nodes

        node_data["is_ancilla"] = not is_boundary
        node_data["is_boundary"] = is_boundary
        if is_boundary:
            node_data["boundary_index"] = 0
            node_data["position_in_boundary"] = boundary_pos_map.get(node, -1)
            node_data["b0"] = True
            for i in range(1, 10):
                node_data[f"b{i}"] = False
        else:
            for i in range(10):
                node_data[f"b{i}"] = False

        node_data.setdefault("depth", node_depth)
        node_data.setdefault("coord", node_info[node].get("coord", None))

    graph = nx.convert_node_labels_to_integers(
        graph,
        ordering="sorted",
        label_attribute="original_node",
    )

    graph.graph["number_of_boundaries"] = 1 if boundary_nodes else 0
    graph.graph["max_depth"] = actual_max_depth
    graph.graph["periodic"] = False
    graph.graph["subdivided"] = False
    graph.graph["schlafli_symbol"] = f"{{3,{q_value}}}"

    graph_attrs_ref = graph.graph
    for node in graph.nodes:
        graph.nodes[node]["graph"] = graph_attrs_ref

    return graph, raw_neighbor_points


def generate_hyperbolic_tiling_3_8_geometric_with_raw(
    depth: int,
    epsilon: float = 1e-5,
    merge_last_layer: bool = True,
    prune_last_layer: bool = True,
    merge_degree_one_boundary: bool = True,
) -> Tuple[Optional[nx.Graph], List[Dict]]:
    """Backward-compatible wrapper for the `{3,8}` geometric generator."""

    return generate_hyperbolic_tiling_3_q_geometric_with_raw(
        8,
        depth=depth,
        epsilon=epsilon,
        merge_last_layer=merge_last_layer,
        prune_last_layer=prune_last_layer,
        merge_degree_one_boundary=merge_degree_one_boundary,
    )

"""Hyperbolic geometry and hyperbolic tiling graph generation.

This module provides functions for:
- Hyperbolic geometry operations in the Poincaré disk model
- Generation of hyperbolic tilings: {4,5} and {3,8}
"""

import networkx as nx
import numpy as np
import collections
import math
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


def generate_hyperbolic_tiling_3_8_geometric_with_raw(
    depth: int,
    epsilon: float = 1e-5,
    merge_last_layer: bool = True,
    prune_last_layer: bool = True,
    merge_degree_one_boundary: bool = True # New parameter
    ) -> Tuple[Optional[nx.Graph], List[Dict]]:
    """
    Generates the {3,8} tiling vertex graph using geometric construction
    and also returns the list of raw potential neighbor coordinates calculated.
    Includes option to disable vertex merging for the final layer,
    option to prune nodes at the final depth layer, and option to merge
    degree-1 boundary nodes connected to the same neighbor.

    Args:
        depth: Max depth from origin node 0.
        epsilon: Tolerance for considering two points identical via distance check.
        merge_last_layer: If False, vertices at max depth are not merged.
        prune_last_layer: If True, removes all nodes at the maximum specified depth
                          from the final graph.
        merge_degree_one_boundary: If True, merges degree-1 nodes that share the
                                   same neighbor after pruning.

    Returns:
        A tuple containing:
        - NetworkX Graph object (or None on error).
        - List of dictionaries for raw points:
          {'parent_id': int, 'parent_coord': complex, 'parent_depth': int, 'raw_coord': complex}
    """
    if depth < 0: raise ValueError("Depth cannot be negative")
    if depth == 0:
        G = nx.Graph(); G.add_node(0, depth=0, coord=complex(0,0)); return G, []
    try: import numpy as np
    except ImportError: print("Error: numpy required."); return None, []

    G = nx.Graph()
    node_count = 0
    # node_info stores metadata collected during BFS
    node_info: Dict[int, Dict[str, Any]] = {}
    # coords_list stores (coord, id) pairs for proximity checks
    coords_list: List[Tuple[complex, int]] = []
    # raw_neighbor_points stores dicts about calculated points before merging
    raw_neighbor_points = []
    # processed_edges stores tuples of sorted node IDs for edges added
    processed_edges = set()

    def add_edge_geo(u_id, v_id):
        """Helper to add edges if they don't exist, u != v, and nodes exist."""
        if u_id == v_id: return
        # Ensure nodes exist before adding edge
        if u_id not in G or v_id not in G: return
        edge_tuple = tuple(sorted((u_id, v_id)))
        if edge_tuple not in processed_edges:
            G.add_edge(u_id, v_id)
            processed_edges.add(edge_tuple)

    # --- Calculate edge distance for {3,8} ---
    p, q = 3, 8
    edge_dist = -1.0
    try:
        # print("\n--- Calculating Edge Distance for {3,8} ---")
        # Formula: cosh(d) = (cos(2pi/p) + cos^2(pi/q)) / sin^2(pi/q)
        # Theoretical value for {3,8} is 1 + sqrt(2)
        cosh_d_val = 1.0 + math.sqrt(2.0)
        # print(f"Using theoretical cosh(d) = 1 + sqrt(2) = {cosh_d_val:.8f}")
        if cosh_d_val < 1.0 - 1e-12: raise ValueError(f"cosh(d) < 1 calculated")
        edge_dist = math.acosh(max(1.0, cosh_d_val))
        # print(f"Hyperbolic edge distance d = acosh(cosh(d)) = {edge_dist:.8f}")
        # r_euclidean = math.tanh(edge_dist / 2.0)
        # print(f"Euclidean radius r = tanh(d/2) = {r_euclidean:.8f}")
        # print("---------------------------------\n")
    except ValueError as e: print(f"Error calc edge dist: {e}"); return None, []
    if edge_dist < 1e-9: print("Warning: Edge dist near zero."); return None, []
    # --- End Distance Calculation ---


    # BFS Initialization
    origin_id = node_count; node_count += 1
    origin_coord = complex(0, 0)
    node_info[origin_id] = {'id': origin_id, 'depth': 0, 'parent': -1, 'coord': origin_coord}
    coords_list.append((origin_coord, origin_id))
    G.add_node(origin_id, depth=0, coord=origin_coord)
    q = collections.deque([origin_id])
    processed_count = 0
    nodes_created_last_layer = 0 # Debug counter
    print(f"Starting Geometric BFS for {{3,8}} (MergeLastLayer={merge_last_layer}, PruneLastLayer={prune_last_layer}) up to depth {depth}...")

    # BFS Loop
    while q:
        u_id = q.popleft()
        if u_id not in node_info: continue
        u_data = node_info[u_id]; u_coord = u_data['coord']; u_depth = u_data['depth']; u_parent_id = u_data['parent']
        processed_count += 1
        if processed_count % 100 == 0: print(f"  Processed {processed_count} nodes...")
        # Stop generating *from* nodes at max depth
        if u_depth >= depth: continue

        parent_coord = node_info[u_parent_id]['coord'] if u_parent_id != -1 else None

        # Find coordinates of the q=8 potential neighbors
        num_neighbors_q = 8
        try:
             potential_neighbor_coords = find_neighbor_coords(u_coord, parent_coord, edge_dist, num_neighbors_q)
        except Exception as e: print(f"Error finding neighbors for {u_id}: {e}"); continue

        # Store Raw Coordinates
        for v_raw in potential_neighbor_coords:
             raw_neighbor_points.append({
                 'parent_id': u_id, 'parent_coord': u_coord,
                 'parent_depth': u_depth, 'raw_coord': v_raw
             })

        # Identify parent candidate among raw coords
        parent_candidate_coord = None
        if parent_coord is not None:
            min_dist_to_parent = float('inf')
            for coord in potential_neighbor_coords:
                 if abs(coord) < 1.0:
                     dist = poincare_dist(coord, parent_coord)
                     if not math.isnan(dist) and dist < min_dist_to_parent:
                          min_dist_to_parent = dist; parent_candidate_coord = coord

        # Process the potential non-parent neighbors
        for v_coord_raw in potential_neighbor_coords:
             # Skip the coordinate identified as the parent candidate
             if parent_candidate_coord is not None and poincare_dist(v_coord_raw, parent_candidate_coord) < epsilon:
                 continue
             # Check validity and pull back if slightly outside boundary
             if abs(v_coord_raw) >= 1.0:
                 if abs(v_coord_raw) < 1.0 + 1e-9: v_coord_raw = v_coord_raw / abs(v_coord_raw) * (1.0 - 1e-12)
                 else: continue

             # Vertex Identification (Proximity Check)
             v_id = -1; found_existing = False; min_dist_found = float('inf'); closest_existing_id = -1
             # Check proximity against ALL existing nodes except u_id itself
             for existing_coord, existing_id in coords_list:
                 if existing_id == u_id: continue
                 dist_check = poincare_dist(v_coord_raw, existing_coord)
                 if not math.isnan(dist_check) and dist_check < min_dist_found:
                      min_dist_found = dist_check; closest_existing_id = existing_id

             should_merge = (closest_existing_id != -1 and min_dist_found < epsilon)
             v_potential_depth = u_depth + 1
             allow_merge = merge_last_layer or (v_potential_depth < depth)

             # Add Node/Edge
             if should_merge and allow_merge:
                  v_id = closest_existing_id
                  add_edge_geo(u_id, v_id)
             else:
                  # Create new node
                  v_id = node_count; node_count += 1
                  if not allow_merge and v_potential_depth == depth:
                       if should_merge:
                           # print(f"Debug: Creating new node {v_id} at max depth {depth} instead of merging with {closest_existing_id} (dist {min_dist_found:.2e}). Parent={u_id}.")
                           nodes_created_last_layer += 1
                  node_info[v_id] = {'id': v_id, 'depth': v_potential_depth, 'parent': u_id, 'coord': v_coord_raw}
                  coords_list.append((v_coord_raw, v_id))
                  G.add_node(v_id, depth=v_potential_depth, coord=v_coord_raw)
                  add_edge_geo(u_id, v_id)
                  # Add to queue only if the *newly created* node is within depth limit for further exploration
                  if v_potential_depth < depth:
                      q.append(v_id)


    print(f"\nGeometric BFS complete. Nodes before pruning: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if not merge_last_layer: print(f"Debug: Created {nodes_created_last_layer} distinct nodes at max depth instead of merging.")

    # --- Prune Last Layer (Optional) ---
    nodes_at_max_depth = [] # Keep track even if not pruning, for attribute setting
    if depth >= 0: # Find nodes at max depth using node_info
         nodes_at_max_depth = [
             nid for nid, data in node_info.items()
             if nid in G and data.get('depth') == depth # Check nid exists in G too
         ]

    if prune_last_layer and nodes_at_max_depth:
        print(f"Pruning {len(nodes_at_max_depth)} nodes at depth {depth}...")
        G.remove_nodes_from(nodes_at_max_depth)
        print(f"Graph after pruning: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    # --- End Pruning ---

    # Reindex sites after pruning. Reindexing is necessary because we later use the site indices to build matrices
    # and perform calculations. This ensures that the indices are consistent and sequential.




    # --- Merge Degree-1 Boundary Nodes (Optional) ---
    if merge_degree_one_boundary:
        print("Attempting to merge degree-1 boundary nodes...")
        nodes_to_remove = []
        neighbors_of_degree_one = collections.defaultdict(list)

        # Identify degree-1 nodes and group by neighbor
        for node in list(G.nodes()): # Iterate over a copy of node list
            if G.degree(node) == 1:
                # Check if node still exists (might have been removed in a previous merge)
                if node not in G: continue
                neighbor = list(G.neighbors(node))[0]
                neighbors_of_degree_one[neighbor].append(node)

        # Perform merging
        nodes_merged_count = 0
        for neighbor, degree_one_nodes in neighbors_of_degree_one.items():
            if len(degree_one_nodes) > 1:
                # Keep the first node (e.g., lowest ID if sorted, or just the first encountered)
                node_to_keep = min(degree_one_nodes) # Keep the one with the smallest ID
                nodes_to_merge = [n for n in degree_one_nodes if n != node_to_keep]

                # Remove the merged nodes from graph and node_info
                for node_to_merge in nodes_to_merge:
                    if node_to_merge in G:
                        G.remove_node(node_to_merge)
                        if node_to_merge in node_info:
                            del node_info[node_to_merge]
                        # Also remove from coords_list if necessary (though less critical after BFS)
                        coords_list = [(c, nid) for c, nid in coords_list if nid != node_to_merge]
                        nodes_merged_count += 1
                        nodes_to_remove.append(node_to_merge) # Keep track if needed

        if nodes_merged_count > 0:
            print(f"Merged {nodes_merged_count} degree-1 nodes.")
            print(f"Graph after merging degree-1 nodes: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    # --- End Merge Degree-1 ---



    # --- Determine boundary and Labeling (AFTER pruning and optional merging) ---
    final_nodes = list(G.nodes()) # Get nodes present after potential pruning/merging
    if not final_nodes:
        actual_max_depth = -1
        boundary_nodes = set()
    else:
        # Recalculate max depth based on remaining nodes
        depths = [node_info[nid]['depth'] for nid in final_nodes if nid in node_info]
        actual_max_depth = max(depths) if depths else -1
        boundary_nodes = {nid for nid in final_nodes if nid in node_info and node_info[nid]['depth'] == actual_max_depth and G.degree(nid) == 4
                          }

    # Sort boundary nodes for consistent position assignment (e.g., by node ID)
    boundary_node_list = sorted(list(boundary_nodes))
    boundary_pos_map = {node: i for i, node in enumerate(boundary_node_list)}

    for node in final_nodes:
        if node not in node_info: continue # Skip if info missing (shouldn't happen ideally)
        node_data = G.nodes[node] # Get node data dict from graph
        node_depth = node_info[node].get('depth', -1)
        # node degree must be 4 to be boundary
        is_boundary = node in boundary_nodes

        # Basic labels
        node_data['is_ancilla'] = not is_boundary
        node_data['is_boundary'] = is_boundary

        # Boundary-specific labels
        if is_boundary:
            node_data['boundary_index'] = 0
            node_data['position_in_boundary'] = boundary_pos_map.get(node, -1) # Assign position
            node_data['b0'] = True
            for i in range(1, 10): # Ensure others are False
                 node_data[f'b{i}'] = False
        else: # Ancilla node
            for i in range(10): # Ensure all bX are False
                 node_data[f'b{i}'] = False

        # Add coordinate and depth from node_info if not already present (should be)
        node_data.setdefault('depth', node_depth)
        node_data.setdefault('coord', node_info[node].get('coord', None))

    nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=False)

    print(G.nodes)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    H.graph = deepcopy(G.graph)

    G = H

    print(H.nodes)


    # Add graph-level attributes
    G.graph["number_of_boundaries"] = 1 if boundary_nodes else 0
    G.graph["max_depth"] = actual_max_depth
    G.graph["periodic"] = False # Hyperbolic tilings are not periodic in this sense
    G.graph["subdivided"] = False

    # Add reference to graph attributes in each node
    graph_attrs_ref = G.graph
    for node in final_nodes:
        if node in G: # Check node still exists
             G.nodes[node]["graph"] = graph_attrs_ref
    # --- End Labeling ---


    # Add attributes to remaining nodes just before returning
    # final_nodes = list(G.nodes()) # Get nodes present after potential pruning
    # nx.set_node_attributes(G, {nid: node_info[nid]['depth'] for nid in final_nodes if nid in node_info}, "depth")
    # nx.set_node_attributes(G, {nid: node_info[nid]['coord'] for nid in final_nodes if nid in node_info}, "coord")
    # ^^^ This is now handled within the labeling loop ^^^

    return G, raw_neighbor_points

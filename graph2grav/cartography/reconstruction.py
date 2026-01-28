"""Bulk reconstruction and entanglement wedge analysis."""

import numpy as np
import networkx as nx
from graph2grav import gaussian
from graph2grav.cartography.subsystems import extract_subsystem_covariance


def add_bulk_node(graph, bulk_position, coupling=1.0, region='probe'):
    """Add a bulk probe node to the graph as a boundary node in group b1.

    Creates a new graph with an additional "bulk probe" node connected
    to a specified position in the original graph. The probe is added as
    a boundary node in a separate boundary group (b1) to allow easy
    computation of mutual information between the main boundary (b0)
    and the bulk probe (b1).

    Args:
        graph: NetworkX graph (typically a MERA graph)
        bulk_position: Node index in original graph to connect bulk node to
        coupling: Edge weight for connection to bulk node (default 1.0)
        region: Region attribute for the bulk node (default 'probe')

    Returns:
        graph_extended: New graph with bulk node added
        bulk_node_id: ID of the newly added bulk node

    Example:
        >>> G = crosslinked_tree(5)
        >>> G_ext, bulk_id = add_bulk_node(G, bulk_position=10, coupling=1.0)
        >>> G_ext.nodes[bulk_id]['is_boundary']
        True
        >>> G_ext.nodes[bulk_id]['boundary_index']
        1

    Notes:
        - The bulk probe is a BOUNDARY node (not ancilla)
        - It belongs to boundary group b1 (main boundary is b0)
        - This allows measuring out all ancilla while keeping the probe
        - MI can then be computed between b0 and b1
    """
    graph_extended = graph.copy()

    # New bulk node ID is one past the last node
    bulk_node_id = max(graph.nodes()) + 1

    # Add b1=False to all existing nodes (they're in b0, not b1)
    for node in graph_extended.nodes():
        if 'b1' not in graph_extended.nodes[node]:
            graph_extended.nodes[node]['b1'] = False

    # Add bulk probe node as boundary node in group b1
    graph_extended.add_node(
        bulk_node_id,
        is_boundary=True,      # It's a boundary node
        is_ancilla=False,      # NOT an ancilla node
        is_bulk_probe=True,    # Special flag for identification
        b0=False,              # Not in main boundary group
        b1=True,               # In probe boundary group
        boundary_index=1,      # Belongs to boundary 1
        position_in_boundary=0, # First (and only) node in b1
        region=region,
        graph=graph_extended.graph  # Reference to graph dict
    )

    # Update graph-level attributes
    # Now we have 2 boundaries: b0 (original) and b1 (probe)
    graph_extended.graph['number_of_boundaries'] = 2

    # Add edge to specified position
    # For weighted graphs, include mweight
    if len(list(graph.edges(data=True))) > 0 and 'mweight' in list(graph.edges(data=True))[0][2]:
        graph_extended.add_edge(bulk_position, bulk_node_id, mweight=coupling)
    else:
        graph_extended.add_edge(bulk_position, bulk_node_id, weight=coupling)

    return graph_extended, bulk_node_id


def get_post_measurement_covariance(graph_with_bulk, bulk_node_id,
                                     use_weighted=False,
                                     couplingtime=1.0,
                                     bulk_presqueeze=1.0,
                                     boundary_presqueeze=1.0,
                                     global_presqueeze=1.0
                                     ):
    """Generate covariance and measure out ancilla (keeping all boundary nodes).

    Args:
        graph_with_bulk: Graph with bulk probe node added (probe is a boundary node in b1)
        bulk_node_id: ID of the bulk probe node
        use_weighted: If True, use weighted quench; else unweighted (default False)
        couplingtime: Time parameter for evolution (default 1.0)
        bulk_presqueeze: Initial squeezing for bulk nodes (default 1.0)
        boundary_presqueeze: Initial squeezing for boundary (default 1.0)

    Returns:
        cov_post: Covariance matrix after measurement (all boundary nodes including probe)
        kept_indices: Original indices of kept nodes (sorted)
        bulk_index_in_kept: Index of bulk probe in the kept indices list

    Notes:
        - Bulk probe is a boundary node in group b1, so it's automatically kept
        - Measures out all ancilla nodes
        - Returns covariance in sequential ordering
    """
    # Generate full covariance using appropriate quench
    if use_weighted:
        cov_full = gaussian.covariance_from_weighted_quench(
            graph_with_bulk,
            couplingtime=couplingtime,
            global_presqueeze=global_presqueeze
        )
    else:
        cov_full = gaussian.covariance_from_unweighted_quench(
            graph_with_bulk,
            couplingtime=couplingtime,
            bulk_presqueeze=bulk_presqueeze,
            boundary_presqueeze=boundary_presqueeze
        )

    # Debug: Check if full covariance is physical before measurement
    N_full = cov_full.shape[0] // 2
    Omega_full = np.block([[np.zeros((N_full, N_full)), np.eye(N_full)],
                          [-np.eye(N_full), np.zeros((N_full, N_full))]])
    symeigs_full = gaussian.symeigvals(cov_full, Omega_full) / 2
    min_symeig_full = np.min(symeigs_full)
    if min_symeig_full < 0.5 - 1e-10:
        import warnings
        warnings.warn(f"Covariance BEFORE measurement is already unphysical! Min symeig: {min_symeig_full:.6f}")

    # Perform homodyne measurement using gaussian helper functions
    # Following the pattern from 01_introduction.py

    # Get node lists for reference
    b0_nodes = sorted(gaussian.nodes_from_boolean_label(graph_with_bulk, "b0"))
    b1_nodes = sorted(gaussian.nodes_from_boolean_label(graph_with_bulk, "b1"))
    kept_node_ids = sorted(b0_nodes + b1_nodes)

    # Verify bulk probe is in b1
    assert bulk_node_id in b1_nodes, "Bulk probe should be in boundary group b1"

    # Perform measurement: keep all boundary DOFs, measure momentum of ancilla
    # NOTE: Sorting indices to maintain canonical (q₁,...,qₙ,p₁,...,pₙ) ordering.
    # Downstream code assumes this ordering when computing symplectic eigenvalues.
    # Unsorted indices would require using a permuted symplectic form Ω' = PΩP^T.
    kept_indices_unsorted = gaussian.indices(graph_with_bulk, "b0") + gaussian.indices(graph_with_bulk, "b1")
    kept_indices_sorted = sorted(kept_indices_unsorted)
    measure_indices = gaussian.momentum_indices(graph_with_bulk, "is_ancilla")

    cov_post = gaussian.get_measured(
        cov_full,
        kept_indices_sorted,
        measure_indices
    )

    # After measurement, nodes are reindexed 0, 1, 2, ... in the order they appear in kept_node_ids
    # Find where the bulk node appears in this new indexing
    bulk_index_in_kept = kept_node_ids.index(bulk_node_id)

    return cov_post, kept_node_ids, bulk_index_in_kept


def compute_bulk_entropies(cov_post, region_indices, bulk_index, N_total):
    """Compute all entropies for bulk reconstruction analysis.

    Computes S_b, S_R, S_{R+b}, S_{R^c}, S_{R^c+b}, and MI(R:b).

    Args:
        cov_post: Post-measurement covariance (2N x 2N) in sequential ordering
        region_indices: Indices of boundary region R (list of ints)
        bulk_index: Index of bulk probe node in the covariance matrix
        N_total: Total number of nodes in post-measurement system (boundary + bulk)

    Returns:
        dict with keys:
            - 'S_b': Entropy of bulk node alone
            - 'S_R': Entropy of region R
            - 'S_R_b': Joint entropy of R and bulk
            - 'S_Rc': Entropy of complement region
            - 'S_Rc_b': Joint entropy of complement and bulk
            - 'MI_R_b': Mutual information I(R:b)

    Notes:
        - Assumes cov_post is in sequential ordering
        - All indices should be in range [0, N_total)
    """
    def make_omega(N):
        """Create symplectic form for N modes in sequential ordering."""
        return np.block([[np.zeros((N, N)), np.eye(N)],
                        [-np.eye(N), np.zeros((N, N))]])

    # S_b: Entropy of bulk node alone
    cov_b = extract_subsystem_covariance(cov_post, [bulk_index], ordering='seq')
    S_b = gaussian.entropy(cov_b, make_omega(1))

    # S_R: Entropy of region R
    cov_R = extract_subsystem_covariance(cov_post, region_indices, ordering='seq')
    S_R = gaussian.entropy(cov_R, make_omega(len(region_indices)))

    # S_{R+b}: Joint entropy of R and bulk
    region_plus_bulk = list(region_indices) + [bulk_index]
    cov_R_b = extract_subsystem_covariance(cov_post, region_plus_bulk, ordering='seq')
    S_R_b = gaussian.entropy(cov_R_b, make_omega(len(region_plus_bulk)))

    # Complement region R^c
    all_boundary_indices = list(range(N_total))
    all_boundary_indices.remove(bulk_index)  # All except bulk
    Rc_indices = [i for i in all_boundary_indices if i not in region_indices]

    # S_{R^c}: Entropy of complement
    cov_Rc = extract_subsystem_covariance(cov_post, Rc_indices, ordering='seq')
    S_Rc = gaussian.entropy(cov_Rc, make_omega(len(Rc_indices)))

    # S_{R^c+b}: Joint entropy of complement and bulk
    Rc_plus_bulk = Rc_indices + [bulk_index]
    cov_Rc_b = extract_subsystem_covariance(cov_post, Rc_plus_bulk, ordering='seq')
    S_Rc_b = gaussian.entropy(cov_Rc_b, make_omega(len(Rc_plus_bulk)))

    # Mutual information
    MI_R_b = S_R + S_b - S_R_b

    # Handle numerical errors (MI should be non-negative)
    if MI_R_b < 0 and MI_R_b > -1e-10:
        MI_R_b = 0.0

    return {
        'S_b': S_b,
        'S_R': S_R,
        'S_R_b': S_R_b,
        'S_Rc': S_Rc,
        'S_Rc_b': S_Rc_b,
        'MI_R_b': MI_R_b
    }


def sweep_bulk_reconstruction(graph, bulk_positions, region_sizes,
                               use_weighted=False,
                               coupling=1.0,
                               couplingtime=1.0,
                               bulk_presqueeze=1.0,
                               boundary_presqueeze=1.0,
                               global_presqueeze=1.0,
                               verbose=True):
    """Sweep over bulk positions and boundary region sizes.

    Computes mutual information and entropies for many configurations
    to map out the entanglement wedge structure.

    Args:
        graph: Base MERA graph (before adding bulk node)
        bulk_positions: List of node indices to test as bulk positions
        region_sizes: List of boundary region sizes to test
        use_weighted: Use weighted vs unweighted quench (default False)
        coupling: Bulk node coupling strength (default 1.0)
        couplingtime: Evolution time parameter (default 1.0)
        bulk_presqueeze: Initial bulk squeezing (default 1.0)
        boundary_presqueeze: Initial boundary squeezing (default 1.0)
        verbose: Show progress bar (default True)

    Returns:
        list of dicts, each containing:
            - 'bulk_position': Node index where bulk was added
            - 'region_size': Size of boundary region R
            - 'S_b', 'S_R', 'S_R_b', 'S_Rc', 'S_Rc_b', 'MI_R_b': Entropies

    Example:
        >>> G = crosslinked_tree(5)
        >>> G = make_tree_periodic(G, remove_nodes=True)
        >>> bulk_pos = list(range(10))  # Test first 10 nodes
        >>> region_sizes = [1, 2, 3, 4]
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
    """
    results = []

    # Get boundary nodes from original graph
    boundary_nodes = sorted([n for n, d in graph.nodes(data=True)
                            if d.get('is_boundary', False)])
    N_boundary = len(boundary_nodes)

    # Progress bar if requested
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(bulk_positions, desc="Bulk positions")
        except ImportError:
            print("Install tqdm for progress bars: pip install tqdm")
            iterator = bulk_positions
    else:
        iterator = bulk_positions

    for bulk_pos in iterator:
        # Add bulk node
        graph_ext, bulk_id = add_bulk_node(graph, bulk_pos, coupling=coupling)

        # Generate post-measurement covariance
        cov_post, kept_indices, bulk_idx_kept = get_post_measurement_covariance(
            graph_ext, bulk_id,
            use_weighted=use_weighted,
            couplingtime=couplingtime,
            bulk_presqueeze=bulk_presqueeze,
            boundary_presqueeze=boundary_presqueeze,
            global_presqueeze=global_presqueeze
        )

        # Map original boundary node IDs to indices in kept array
        # kept_indices = sorted(boundary_nodes + [bulk_id])
        # So boundary nodes are at indices 0, 1, ..., N_boundary-1
        # and bulk is at index N_boundary (if bulk_id > all boundary nodes)
        # But this depends on sorting!

        # More robust: create mapping
        node_to_idx = {node: idx for idx, node in enumerate(kept_indices)}
        boundary_indices_in_kept = [node_to_idx[bn] for bn in boundary_nodes]

        # Sweep over region sizes
        for region_size in region_sizes:
            if region_size > N_boundary:
                continue  # Skip if region too large

            # Define region R as first region_size boundary nodes
            # (can also randomize or use different strategies)
            region_R_nodes = boundary_nodes[:region_size]
            region_R_indices = [node_to_idx[n] for n in region_R_nodes]

            # Compute entropies
            entropies = compute_bulk_entropies(
                cov_post,
                region_R_indices,
                bulk_idx_kept,
                len(kept_indices)
            )

            # Store results
            result = {
                'bulk_position': bulk_pos,
                'region_size': region_size,
                **entropies
            }
            results.append(result)

    return results

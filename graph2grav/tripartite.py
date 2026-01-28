"""Tripartite information and entanglement inequality calculations.

This module provides functions for calculating tripartite information and
analyzing entanglement inequalities in Gaussian states on graphs.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

from .technical_helpers import get_lists, get_boundary_nodes_by_position
from .technical_helpers import generate_non_overlapping_regions_v2
from .gaussian import entropy, Omega_interwoven, get_measured, calculate_boundary_entropy

def get_tripartite_information(
    C: np.ndarray,
    A: List[int],
    B: List[int],
    C_regions: List[int],
) -> float:
    """Calculate the tripartite information I3(A:B:C) for three regions.

    Parameters
    ----------
    C : np.ndarray
        The covariance matrix
    A : List[int]
        List of indices for region A
    B : List[int]
        List of indices for region B
    C_regions : List[int]
        List of indices for region C
    imeasured : List[int], optional
        List of measured mode indices, by default None

    Returns
    -------
    float
        The tripartite information I3(A:B:C)
    """

    # Calculate entanglement entropies
    S_A = calculate_single_entropy(C, A)
    S_B = calculate_single_entropy(C, B)
    S_C = calculate_single_entropy(C, C_regions)
    S_AB = calculate_single_entropy(C, A + B)
    S_AC = calculate_single_entropy(C, A + C_regions)
    S_BC = calculate_single_entropy(C, B + C_regions)
    S_ABC = calculate_single_entropy(C, A + B + C_regions)

    # Calculate tripartite information
    I3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

    return I3

def plot_EE_inequalities(
    C: np.ndarray,
    n_regions: int = 3,
    ax: Optional[plt.Axes] = None
) -> None:
    """Plot entanglement entropy inequalities for a given graph.

    Parameters
    ----------
    C : nx.Graph
        The covariance matrix of the graph
    n_regions : int, optional
        Number of regions to analyze, by default 3
    imeasured : List[int], optional
        List of measured mode indices, by default None
    ax : Optional[plt.Axes], optional
        Matplotlib axes to plot on, by default None
    """

    # Get all possible region combinations
    region_triples = get_lists(n_regions)

    # Calculate entanglement inequalities for each combination
    violations = []
    for A, B, C_regions in region_triples:
        # Calculate smoothed entanglement entropies
        S_A = calculate_single_entropy(C, A)
        S_B = calculate_single_entropy(C, B)
        S_C = calculate_single_entropy(C, C_regions)
        S_AB = calculate_single_entropy(C, A + B)
        S_AC = calculate_single_entropy(C, A + C_regions)
        S_BC = calculate_single_entropy(C, B + C_regions)
        S_ABC = calculate_single_entropy(C, A + B + C_regions)

        # Calculate monogamy of mutual information
        MMI = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC
        violations.append(MMI)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram of violations
    ax.hist(violations, bins=20, alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', label='MMI = 0')
    ax.set_xlabel('Monogamy of Mutual Information')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Distribution of Entanglement Inequality Violations')

def calculate_single_entropy(cov_matrix_interspersed, region_indices):
    """Calculates the entanglement entropy for a given region.

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order (q1, p1, q2, p2, ...).
        region_indices: A list of site indices (0 to L-1) defining the region.

    Returns:
        The entanglement entropy of the region.
    """
    if not region_indices:
        return 0.0 # Entropy of empty region is 0

    # Convert site indices to phase space indices (interspersed order)
    phase_space_indices = []
    for i in sorted(region_indices):
        phase_space_indices.extend([2 * i, 2 * i + 1])

    sub_matrix = cov_matrix_interspersed[np.ix_(phase_space_indices, phase_space_indices)]
    sub_omega = Omega_interwoven(len(region_indices))

    return entropy(sub_matrix, sub_omega, val_if_invalid=np.nan)


def calculate_tripartite_information(cov_matrix_interspersed, A, B, C):
    """Calculates the tripartite information I3 for three regions A, B, C.

    I3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order.
        A: List of site indices for region A.
        B: List of site indices for region B.
        C: List of site indices for region C.

    Returns:
        The tripartite information I3.
    """
    S_A = calculate_single_entropy(cov_matrix_interspersed, A)
    S_B = calculate_single_entropy(cov_matrix_interspersed, B)
    S_C = calculate_single_entropy(cov_matrix_interspersed, C)

    # Note: Region indices must be combined and sorted for sub_matrix selection,
    # but calculate_single_entropy handles sorting internally.
    S_AB = calculate_single_entropy(cov_matrix_interspersed, A + B)
    S_BC = calculate_single_entropy(cov_matrix_interspersed, B + C)
    S_AC = calculate_single_entropy(cov_matrix_interspersed, A + C)

    S_ABC = calculate_single_entropy(cov_matrix_interspersed, A + B + C)

    # Handle potential NaN results from entropy calculation if sub_matrix was invalid
    entropies = [S_A, S_B, S_C, S_AB, S_BC, S_AC, S_ABC]
    if np.isnan(entropies).any():
        return np.nan

    I3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    return I3


def calculate_all_tripartite(cov_matrix_interspersed, mingap: int = 0):
    """Calculates I3 for all combinations of three non-overlapping contiguous regions.

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order.
        mingap: The minimum required gap between the end of one region
                and the start of the next (cyclically). Must be >= 0.

    Returns:
        A list of tuples: [( (A, B, C), I3_value ), ... ]
    """
    L = cov_matrix_interspersed.shape[0] // 2
    if L < 3:
        return [] # Need at least 3 sites for 3 regions

    region_triples = generate_non_overlapping_regions_v2(L, mingap=mingap)

    results = []
    for A, B, C in region_triples:
        # Ensure regions are sorted lists internally if needed, though calculate_single_entropy handles it.
        # The generate_non_overlapping_regions_v2 function should already return sorted lists within tuples.
        I3 = calculate_tripartite_information(cov_matrix_interspersed, A, B, C)
        results.append(((A, B, C), I3))

    return results

# --- New Functions for Boundary Regions ---


def calculate_tripartite_information_boundary(
        cov_matrix_interspersed: np.ndarray, A_pos: List[int], B_pos: List[int], C_pos: List[int], boundary_nodes_map: Dict[int, List[int]], average_over_two=True
        ) -> float:
    """Calculates the tripartite information I3 for three boundary regions A, B, C defined by positions.

    I3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order.
        A_pos: List of boundary position indices for region A.
        B_pos: List of boundary position indices for region B.
        C_pos: List of boundary position indices for region C.
        boundary_nodes_map: A dictionary mapping boundary position -> list of phase space node indices.

    Returns:
        The tripartite information I3 for the boundary regions.
    """

    if average_over_two:
        # Average over the two possible orderings of A and B
        I3_A_B = calculate_tripartite_information(cov_matrix_interspersed, A_pos, B_pos, C_pos)
        I3_A_B_shifted = calculate_tripartite_information(
            np.roll(cov_matrix_interspersed, (2, 2), axis=(0, 1)), A_pos, B_pos, C_pos)
        return (I3_A_B + I3_A_B_shifted) / 2

    S_A = calculate_boundary_entropy(cov_matrix_interspersed, A_pos, boundary_nodes_map)
    S_B = calculate_boundary_entropy(cov_matrix_interspersed, B_pos, boundary_nodes_map)
    S_C = calculate_boundary_entropy(cov_matrix_interspersed, C_pos, boundary_nodes_map)

    S_AB = calculate_boundary_entropy(cov_matrix_interspersed, A_pos + B_pos, boundary_nodes_map)
    S_BC = calculate_boundary_entropy(cov_matrix_interspersed, B_pos + C_pos, boundary_nodes_map)
    S_AC = calculate_boundary_entropy(cov_matrix_interspersed, A_pos + C_pos, boundary_nodes_map)

    S_ABC = calculate_boundary_entropy(cov_matrix_interspersed, A_pos + B_pos + C_pos, boundary_nodes_map)

    # Handle potential NaN results
    entropies = [S_A, S_B, S_C, S_AB, S_BC, S_AC, S_ABC]
    if np.isnan(entropies).any():
        return np.nan
    
    if np.any(np.array(entropies) < 0):
        raise ValueError("Entropies must be non-negative.")

    I3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    return I3

def calculate_all_tripartite_boundary(
    graph: nx.Graph,
    cov_matrix_interspersed: np.ndarray,
    kept_indices: Optional[List[int]] = None,
    mingap: int = 0,
    use_tqdm: Optional[bool] = True,
    average_over_two: bool = False,
    assume_rotational_invariance: bool = False,
    boundary_nodes_map = None
) -> List[Tuple[Tuple[List[int], List[int], List[int]], float]]:
    """Calculates I3 for all combinations of three non-overlapping contiguous boundary regions.

    Regions are defined by boundary positions.
    Handles cases where the covariance matrix has been reduced by measurement.

    Args:
        graph: The networkx graph containing original boundary node information.
        cov_matrix_interspersed: The full *or measured* covariance matrix in interspersed order.
                                 If measured, its indices must correspond to `kept_indices`.
        kept_indices: Optional list of the original phase space indices that remain
                      in `cov_matrix_interspersed` after measurement.
                      If None, assumes `cov_matrix_interspersed` is the full matrix.
        mingap: The minimum required gap between the end of one boundary region
                and the start of the next (cyclically, based on boundary positions).
                Must be >= 0.
        use_tqdm: Whether to display a progress bar.

    Returns:
        A list of tuples: [ ( (A_pos, B_pos, C_pos), I3_value ), ... ]
        where A_pos, B_pos, C_pos are lists of boundary position indices.
    """
    # Get boundary node mapping (relative to kept_indices if provided) and unique sorted positions
    if boundary_nodes_map is None:
        boundary_nodes_map, unique_sorted_positions = get_boundary_nodes_by_position(graph, kept_indices)

    L_boundary = len(unique_sorted_positions)
    # Check if enough positions exist considering the minimum gap
    min_required_positions = 3 * (mingap + 1) # 3 regions of length 1 + 3 gaps of length mingap
    if L_boundary < min_required_positions:
        print(f"Warning: Need at least {min_required_positions} unique boundary positions for mingap={mingap}, found {L_boundary}.")
        return []

    # Map the generated region indices (0 to L_boundary-1) back to actual boundary positions
    position_map = {i: pos for i, pos in enumerate(unique_sorted_positions)}

    # Generate combinations using indices 0 to L_boundary-1, applying mingap
    region_index_triples = generate_non_overlapping_regions_v2(
        L_boundary, mingap=mingap, assume_rotational_invariance=assume_rotational_invariance)

    results = []
    iterable = tqdm(region_index_triples) if use_tqdm else region_index_triples
    for A_idx, B_idx, C_idx in iterable:
        # Convert indices back to actual boundary positions
        A_pos = [position_map[i] for i in A_idx]
        B_pos = [position_map[i] for i in B_idx]
        C_pos = [position_map[i] for i in C_idx]

        # Pass the (potentially measured) covariance matrix and the relative boundary_nodes_map
        I3 = calculate_tripartite_information_boundary(cov_matrix_interspersed, A_pos, B_pos, C_pos, boundary_nodes_map, average_over_two=average_over_two)
        results.append(((A_pos, B_pos, C_pos), I3))

    return results
"""gaussian.py: Operations producing and manipulating Gaussian states."""
import networkx as nx
import numpy as np  
from scipy.linalg import sqrtm, eigh, inv
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Dict
from tqdm import tqdm

def critical_hamiltonian(q: float, graph: nx.Graph) -> np.ndarray:
    """From a graph and mass q, return the near-critical Hamiltonian that embeds the Laplacian in the position block.

    q is the mass of the particles, such that q=0 is the critical point.
    """

    Laplacian = nx.laplacian_matrix(graph).todense()
    N = len(graph.nodes) # number of nodes

    h = np.block([
        [Laplacian, np.zeros((N, N))],
        [np.zeros((N, N)), np.zeros((N, N))],
    ]) + q * np.eye(2*N)

    return h

def critical_covariance(q: float, graph: nx.Graph) -> np.ndarray:
    """From a graph and a mass q, return the covariance matrix of the near-critical Gaussian state.

    The ability to directly calculate the covariance relies on the Hamiltonian matrix h being block diagonal into a position and momentum block,
    which `critical_hamiltonian` satisfies.

    See Schuch, N., Cirac, J.I. & Wolf, M.M. Quantum States on Harmonic Lattices. Commun. Math. Phys. 267, 65–92 (2006) for details,
    in particular Eq. 8.
    """

    h = critical_hamiltonian(q, graph)
    # N = len(graph.nodes) # number of nodes
    N =  graph.number_of_nodes()

    h_qq = h[:N, :N]
    assert np.allclose(h[N:, :N], 0), "Gc_pq must be zero matrix"
    assert np.allclose(h[:N, N:], 0), "Gc_qp must be zero matrix"
    h_pp = h[N:, N:]


    X = np.linalg.inv(sqrtm(h_qq)) @ sqrtm(sqrtm(h_qq) @ h_pp @ sqrtm(h_qq)) @ np.linalg.inv(sqrtm(h_qq))

    Vc = np.real(np.block([
        [X, np.zeros((N, N))],
        [np.zeros((N, N)), np.linalg.inv(X)]
    ]))

    return Vc


def covariance_from_weighted_quench(G, couplingtime=1., bulk_presqueeze=1., global_presqueeze=1.):
    """
    Compute the covariance matrix from a weighted quench based on the weighted
    adjacency matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Each node should have a 'region' attribute that is either
        'bulk' or something else. Each edge should have an 'mweight' attribute
        representing its weight.
    couplingtime : float, optional
        Scaling factor for the adjacency matrix, representing the coupling strength
        or time. Default is 1.
    bulk_presqueeze : float, optional
        Factor to squeeze the initial position of bulk nodes.
        Default is 1, which is no squeezing. Note: Using this option is
        not advised as its benefits are unclear.
    Returns
    -------
    numpy.ndarray
        The computed covariance matrix after the quench transformation.
    Notes
    -----
    Using the bulk_presqueeze option is not advised, because it is not clear
    if it is helpful.
    """
    A = np.array(nx.adjacency_matrix(G, weight="mweight").todense())
    nodes = np.array(G.nodes)

    # initial_position_covariance = np.array([
    #     bulk_presqueeze if G.nodes[n]["region"] == "bulk"
    #     else 1  for n in nodes
    # ])
    squeezing = np.array([global_presqueeze]*len(nodes))

    # 12/18/25: Changed definition of squeezing to match convention in paper
    initial_covariance = np.diag(np.concatenate(
        [1/squeezing, squeezing]
    ))

    A = couplingtime*A
    I = np.eye(A.shape[0])
    Z = np.zeros(A.shape)

    S = np.block([[I, Z], [-A, I]])
    C = S @ initial_covariance @ S.T
    return C

def covariance_from_unweighted_quench(G, couplingtime=1, bulk_presqueeze=1, boundary_presqueeze=1):
    """
    Compute the covariance matrix from a weighted quench based on the weighted
    adjacency matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph. Each node should have a 'region' attribute that is either
        'bulk' or something else. Each edge should have an 'mweight' attribute
        representing its weight.
    couplingtime : float, optional
        Scaling factor for the adjacency matrix, representing the coupling strength
        or time. Default is 1.
    bulk_presqueeze : float, optional
        Factor to squeeze the initial position of bulk nodes.
        Default is 1, which is no squeezing. Note: Using this option is
        not advised as its benefits are unclear.
    Returns
    -------
    numpy.ndarray
        The computed covariance matrix after the quench transformation.
    Notes
    -----
    Using the bulk_presqueeze option is not advised, because it is not clear
    if it is helpful.
    """
    A = np.array(nx.adjacency_matrix(G).todense())

    squeezing = np.array([
        bulk_presqueeze if G.nodes[n]["is_ancilla"]
        else (boundary_presqueeze if G.nodes[n]["is_boundary"] else 1)  for n in G.nodes
    ])

    # 12/18/2025: reversing meaning of squeezing parameters to match paper 
    initial_covariance = np.diag(np.concatenate(
        [1/squeezing, squeezing]
    ))

    A = couplingtime*A
    I = np.eye(A.shape[0])
    Z = np.zeros(A.shape)

    S = np.block([[I, Z], [-A, I]])
    C = S @ initial_covariance @ S.T
    return C


def to_seq(C):
    """Convert a covariance matrix C from interspersed order (q1, p1, q2, p2, ...) to sequential order (q1, q2, ..., p1, p2, ...)."""
    indices = np.arange(C.shape[0])
    even = list(indices[0::2])
    odd = list(indices[1::2])
    C = C[:, even+odd]
    C = C[even+odd, :]
    return C

def to_inter(C):
    """Convert a covariance matrix C from sequential order (q1, q2, ..., p1, p2, ...) to interspersed order (q1, p1, q2, p2, ...)."""
    nn = C.shape[0]
    indices = np.arange(nn)
    firsthalf = list(indices[:nn//2])
    secondhalf = list(indices[nn//2:])
    intersperse = [i for j in zip(firsthalf, secondhalf) for i in j]
    C = C[:, intersperse]
    C = C[intersperse, :]
    return C

def Omega_sequential(N): # formerly Omega_from_stack
    """The Omega with stacked, qs and ps are separated."""
    return np.kron([[0, 1], [-1, 0]], np.eye(N))

def Omega_interwoven(nsites):
    """With interwoven is easier to slice by region."""
    return np.kron(np.eye(nsites), [[0, 1], [-1, 0]])

def symeigvals(C, Omega):
    """Calculate the symplectic eigenvalues."""
    symeigs = np.abs(np.linalg.eigvals(C @ Omega))
    return symeigs

def entropy_from_symeigs(symeigs, val_if_invalid=np.nan):
    """ Calculate von Neumann entropy from the symplectic eigenvalues. Might be a factor of two off; dividing by two to compensate. Also now diving the symplectic eigenvalues by 2."""
    s = [(((se+1/2)*np.log(se+1/2) - (se - 1/2)*np.log(se-1/2)) if se > 1/2 else val_if_invalid) for se in np.array(symeigs) ]
    
    badmask = np.array(symeigs) < 1/2-0.00001
    if np.any(badmask):
        print("Warning: symeigs less than 1/2:")
        print("- number of symeigs less than 1/2:", np.count_nonzero(badmask))
        print("- symeigs less than 1/2", np.array(symeigs)[badmask])


    return np.nansum(s)/2

def entropy(C, Omega, val_if_invalid=np.nan, norm=1):
    """ Calculate von Neumann entropy from a covariance matrix and a symplectic form.

    I wouldn't recommend using val_if_invalid, but it can be helpful if the symplectic eigenvalue
    should be 1/2 but is slightly below due to numerical errors. In that case, val_if_invalid should be set to zero.

    Another option is to add the identity to the covariance matrix until it is valid. This is equivalent to adding
    thermal noise to the system, if I recall correctly.
    """
    symeigs = symeigvals(C, Omega) / (norm * 2)
    return entropy_from_symeigs(symeigs, val_if_invalid)



def nodes_from_boolean_label(graph, label):
    """Return list of nodes where the given boolean attribute is True.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with boolean node attributes.
    label : str
        Name of boolean node attribute to filter by.

    Returns
    -------
    list
        Node IDs where graph.nodes[n][label] is True.
    """
    return [n for n in graph.nodes if graph.nodes[n][label]]

def boundary_nodes(graph):
    """Return list of boundary nodes (where is_boundary is True)."""
    return nodes_from_boolean_label(graph, "is_boundary")



def position_indices_from_nodes(nodes, graph=None, N=None):
    """Return position quadrature indices for given nodes.

    In sequential ordering (q₁,...,qₙ,p₁,...,pₙ), position indices equal node IDs.

    Parameters
    ----------
    nodes : list
        List of node IDs.
    graph : networkx.Graph, optional
        Graph (unused, kept for API consistency).
    N : int, optional
        Total number of nodes (unused, kept for API consistency).

    Returns
    -------
    list
        Position indices (same as input nodes).
    """
    return nodes

def momentum_indices_from_nodes(nodes, graph=None, N=None):
    """
    This function returns the momentum indices for the given nodes in the graph.
    
    Parameters:
    nodes (list): List of node indices for which momentum indices are to be calculated.
    graph (networkx.Graph, optional): The graph object containing the nodes. Default is None.
    N (int, optional): Total number of nodes in the graph. Default is None.
    
    Returns:
    list: List of momentum indices corresponding to the given nodes.
    
    Raises:
    ValueError: If neither graph nor N is specified.
    """
    if graph:
        N = len(graph.nodes)
    elif N is None:
        raise ValueError("Either graph or N must be specified")

    return [n+N for n in nodes]

def indices_from_nodes(nodes, graph=None, N=None):
    """Return both position and momentum indices for given nodes.

    Parameters
    ----------
    nodes : list
        List of node IDs.
    graph : networkx.Graph, optional
        Graph containing the nodes.
    N : int, optional
        Total number of nodes (if graph not provided).

    Returns
    -------
    list
        Position indices followed by momentum indices.
    """
    return list(position_indices_from_nodes(nodes, graph, N)) + list(momentum_indices_from_nodes(nodes, graph, N))

def indices(graph, label):
    """Return position and momentum indices for nodes with given boolean label.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with boolean node attributes.
    label : str
        Name of boolean node attribute.

    Returns
    -------
    list
        Position indices followed by momentum indices for labeled nodes.
    """
    return indices_from_nodes(nodes_from_boolean_label(graph, label), graph)

def momentum_indices(graph, label):
    """Return momentum indices for nodes with given boolean label.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with boolean node attributes.
    label : str
        Name of boolean node attribute.

    Returns
    -------
    list
        Momentum indices for labeled nodes.
    """
    return momentum_indices_from_nodes(nodes_from_boolean_label(graph, label), graph)

def position_indices(graph, label):
    """Return position indices for nodes with given boolean label.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with boolean node attributes.
    label : str
        Name of boolean node attribute.

    Returns
    -------
    list
        Position indices for labeled nodes.
    """
    return position_indices_from_nodes(nodes_from_boolean_label(graph, label), graph)

def mesh(graph, label, label2=None):
    """
    This function returns the position mesh indices for the given labels in the graph.
    
    Parameters:
    graph (networkx.Graph): The graph object containing the nodes.
    label (str): The label to filter nodes for the first dimension of the mesh.
    label2 (str, optional): The label to filter nodes for the second dimension of the mesh. Default is None.
    
    Returns:
    tuple: A tuple of arrays representing the position mesh indices.
    """
    return np.ix_(
        indices(graph, label),
        indices(graph, label if label2 is None else label2)
    )

# todo handle interspersed case

def position_mesh(graph, label, label2=None):
    """
    This function returns the position mesh indices for the given labels in the graph.
    
    Parameters:
    graph (networkx.Graph): The graph object containing the nodes.
    label (str): The label to filter nodes for the first dimension of the mesh.
    label2 (str, optional): The label to filter nodes for the second dimension of the mesh. Default is None.
    
    Returns:
    tuple: A tuple of arrays representing the position mesh indices.
    """
    return np.ix_(
        position_indices_from_nodes(nodes_from_boolean_label(graph, label), graph),
        position_indices_from_nodes(nodes_from_boolean_label(graph, label if label2 is None else label2), graph)
    )

def momentum_mesh(graph, label, label2=None):
    """
    This function returns the momentum mesh indices for the given labels in the graph.

    Parameters:
    graph (networkx.Graph): The graph object containing the nodes.
    label (str): The label to filter nodes for the first dimension of the mesh.
    label2 (str, optional): The label to filter nodes for the second dimension of the mesh. Default is None.

    Returns:
    tuple: A tuple of arrays representing the momentum mesh indices.
    """
    return np.ix_(
        momentum_indices_from_nodes(nodes_from_boolean_label(graph, label), graph),
        momentum_indices_from_nodes(nodes_from_boolean_label(graph, label if label2 is None else label2), graph)
    )   




# todo rework to be based on labels
def projective_measure(V1, V2, V12, d=1000):
    """Projective measurement onto a state with squeezing d."""
    Dd = np.diag([1/d, d]*(np.shape(V2)[0]//2))
    return to_seq(V1 - V12.T @ np.linalg.pinv(V2+ Dd @ Dd ) @ V12)

def homodyne_measure(V1, V2, V12):
    """Homodyne measurement on quadratures specified by V2, leaving behind an impacted V1."""
    return V1 - V12 @ np.linalg.pinv(V2) @ V12.T

def get_measured(C, remainingindices, measuredindices):
    """Perform homodyne measurement on specified DOFs, returning reduced covariance.

    Parameters
    ----------
    C : numpy.ndarray
        Full covariance matrix before measurement.
    remainingindices : list
        Indices of DOFs to keep (not measured).
    measuredindices : list
        Indices of DOFs to measure out.

    Returns
    -------
    numpy.ndarray
        Reduced covariance matrix for remaining DOFs after measurement.
    """
    V1 = C[np.ix_(remainingindices, remainingindices)]
    V2 = C[np.ix_(measuredindices, measuredindices)]
    V12 = C[np.ix_(remainingindices, measuredindices)]
    return homodyne_measure(V1, V2, V12)


def topleft(A):
    """Extract top-left quadrant of a matrix (position-position block)."""
    return A[:, :len(A)//2][:len(A)//2, :]

position = topleft

def bottomright(A):
    """Extract bottom-right quadrant of a matrix (momentum-momentum block)."""
    return A[:, len(A)//2:][len(A)//2:, :]

momentum = bottomright


def get_EE(C, include_starting_zero=True, average_over_two=False):
    """Compute entanglement entropies for contiguous boundary regions of increasing length.

    Parameters
    ----------
    C : numpy.ndarray
        Covariance matrix in sequential ordering.
    include_starting_zero : bool, optional
        If True, prepend 0 to entropy list (S=0 for empty region). Default True.
    average_over_two : bool, optional
        If True, average entropies over two shifted configurations to account for
        boundary non-translation-invariance. Default False.

    Returns
    -------
    list
        Entanglement entropies for regions of length 0, 1, 2, ..., N/2.
    """
    if average_over_two:
        # roll the matrix
        C2 = to_inter(C)
        C2 = np.roll(C2, shift=(2, 2), axis=(0, 1))
        C2 = to_seq(C2)

        
        return list((
            np.array(get_EE(C, include_starting_zero=include_starting_zero, average_over_two=False))
            + np.array(get_EE(C2, include_starting_zero=include_starting_zero, average_over_two=False))
        ) / 2)

    # reordering the covariance matrix such that position and momentum are interleaved
    # makes it easier to index a region of the covariance matrix
    iC = to_inter(C)

    entropies = []

    halflen = len(iC)//2

    for length in np.arange(0, halflen)+1:
        
        region = np.arange(2*length)

        Omega = Omega_interwoven(length)

        entropies.append(entropy(iC[region, :][:, region], Omega, val_if_invalid=np.nan))

    if include_starting_zero:
        return [0] + entropies
    
    
    # The boundaries are not quite translation invariant, so we average over two

    return entropies

def renyi2_entropy(C):
    """Calculate the Renyi-2 entropy from the covariance matrix."""
    return np.linalg.slogdet(C)[1] / 2

def renyi_entropy(C, Omega, alpha=2, val_if_invalid=np.nan):
    """Currently we are assuming a norm of 1."""
    if alpha == 2:
        return renyi2_entropy(C)
    elif alpha == 1:
        return entropy(C, Omega, val_if_invalid=val_if_invalid)
    else:
        """See Serafini equation 3.96 and surrounding text for a reference on the math.
           The fomrulas we use are a little different because we use natural logarithms
           and we put the normalization factor in a different spot. What's important
           is that it matches with both von Neumann entropy and Renyi 2 entropy. This does.
        """
        # factors of two are confusing here, but I think this is correct for a norm of 1.
        symplectic_eigenvalues_with_duplicates = symeigvals(C, Omega) / 2

        # symplectic eigenvalues come in pairs, so we need to take every second element
        symplectic_eigenvalues = np.sort(symplectic_eigenvalues_with_duplicates)[::2]
        
        internal = 1 / ((symplectic_eigenvalues + 1/2)**alpha - (symplectic_eigenvalues - 1/2)**alpha)
        renyi_entropies = 1 / (1 - alpha) * np.sum(np.log(internal))


        return renyi_entropies


def calculate_symplectic_eigenvalues(cov_matrix_interspersed: np.ndarray, boundary_positions: List[int], boundary_nodes_map: Dict[int, List[int]], norm = 1) -> float:
    """Calculates the entanglement entropy for a region defined by boundary positions.

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order.
        boundary_positions: A list of boundary position indices defining the region.
        boundary_nodes_map: A dictionary mapping boundary position -> list of phase space node indices.

    Returns:
        The entanglement entropy of the boundary region.
    """
    if boundary_positions is None or len(boundary_positions) == 0:
        return 0.0 # Entropy of empty region is 0

    # Collect all phase space indices corresponding to the given boundary positions
    mode_indices = []
    for pos in boundary_positions:
        if pos in boundary_nodes_map:
            mode_indices.extend(boundary_nodes_map[pos])
        else:
            # Optional: Warn if a requested position doesn't exist in the map
            print(f"Warning: Boundary position {pos} not found in boundary_nodes_map.")

    if not mode_indices:
        return 0.0 # No nodes found for the given positions
    
    mode_indices = np.array(mode_indices)

    num_modes = len(mode_indices)
    
    phase_space_indices = list(np.concatenate([2* mode_indices, 2*mode_indices + 1]))

    # Sort the combined phase space indices
    phase_space_indices.sort()

    # Extract the sub-matrix
    sub_matrix = cov_matrix_interspersed[np.ix_(phase_space_indices, phase_space_indices)]

    # print(boundary_positions)
    # print(sub_matrix.shape)

    # print(phase_space_indices)
    # print(sub_matrix.shape)

    # Calculate the number of *sites* (pairs of q,p) in the sub-region
    if len(phase_space_indices) % 2 != 0:
        # This shouldn't happen if nodes always come in q,p pairs
        print("Warning: Odd number of phase space indices found for boundary region.")
        return np.nan # Or handle error appropriately

    sub_omega = Omega_interwoven(num_modes)

    symeigs = symeigvals(sub_matrix, sub_omega) / (norm * 2)

    # it returns two copies of each symplectic eigenvalue. We don't want that.

    sorted_symeigs = np.sort(symeigs)
    every_other_symeig = sorted_symeigs[::2]
    
    return every_other_symeig
    

def calculate_boundary_entropy(cov_matrix_interspersed: np.ndarray, boundary_positions: List[int], boundary_nodes_map: Dict[int, List[int]], renyi=False, alpha=2) -> float:
    """Calculates the entanglement entropy for a region defined by boundary positions.

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order.
        boundary_positions: A list of boundary position indices defining the region.
        boundary_nodes_map: A dictionary mapping boundary position -> list of phase space node indices.

    Returns:
        The entanglement entropy of the boundary region.
    """
    if boundary_positions is None or len(boundary_positions) == 0:
        return 0.0 # Entropy of empty region is 0

    # Collect all phase space indices corresponding to the given boundary positions
    mode_indices = []
    for pos in boundary_positions:
        if pos in boundary_nodes_map:
            mode_indices.extend(boundary_nodes_map[pos])
        else:
            # Optional: Warn if a requested position doesn't exist in the map
            print(f"Warning: Boundary position {pos} not found in boundary_nodes_map.")

    if not mode_indices:
        return 0.0 # No nodes found for the given positions
    
    mode_indices = np.array(mode_indices)

    num_modes = len(mode_indices)
    
    phase_space_indices = list(np.concatenate([2* mode_indices, 2*mode_indices + 1]))

    # Sort the combined phase space indices
    phase_space_indices.sort()

    # Extract the sub-matrix
    sub_matrix = cov_matrix_interspersed[np.ix_(phase_space_indices, phase_space_indices)]

    # print(boundary_positions)
    # print(sub_matrix.shape)

    # print(phase_space_indices)
    # print(sub_matrix.shape)

    # Calculate the number of *sites* (pairs of q,p) in the sub-region
    if len(phase_space_indices) % 2 != 0:
        # This shouldn't happen if nodes always come in q,p pairs
        print("Warning: Odd number of phase space indices found for boundary region.")
        return np.nan # Or handle error appropriately

    sub_omega = Omega_interwoven(num_modes)

    return entropy(sub_matrix, sub_omega) if not renyi else renyi_entropy(sub_matrix, sub_omega, alpha=alpha) # , val_if_invalid=np.nan)

def calculate_boundary_submatrix(cov_matrix_interspersed: np.ndarray, boundary_positions: List[int], boundary_nodes_map: Dict[int, List[int]]) -> float:
    """Calculates the entanglement entropy for a region defined by boundary positions.

    Args:
        cov_matrix_interspersed: The full covariance matrix in interspersed order.
        boundary_positions: A list of boundary position indices defining the region.
        boundary_nodes_map: A dictionary mapping boundary position -> list of phase space node indices.

    Returns:
        The entanglement entropy of the boundary region.
    """
    if boundary_positions is None or len(boundary_positions) == 0:
        return 0.0 # Entropy of empty region is 0

    # Collect all phase space indices corresponding to the given boundary positions
    mode_indices = []
    for pos in boundary_positions:
        if pos in boundary_nodes_map:
            mode_indices.extend(boundary_nodes_map[pos])
        else:
            # Optional: Warn if a requested position doesn't exist in the map
            raise ValueError(f"Boundary position {pos} not found in boundary_nodes_map.")

    if not mode_indices:
        return 0.0 # No nodes found for the given positions
    
    mode_indices = np.array(mode_indices)

    num_modes = len(mode_indices)
    
    phase_space_indices = list(np.concatenate([2* mode_indices, 2*mode_indices + 1]))

    # Sort the combined phase space indices
    phase_space_indices.sort()

    # Extract the sub-matrix
    sub_matrix = cov_matrix_interspersed[np.ix_(phase_space_indices, phase_space_indices)]

    # print(boundary_positions)
    # print(sub_matrix.shape)

    # print(phase_space_indices)
    # print(sub_matrix.shape)

    # Calculate the number of *sites* (pairs of q,p) in the sub-region
    if len(phase_space_indices) % 2 != 0:
        # This shouldn't happen if nodes always come in q,p pairs
        print("Warning: Odd number of phase space indices found for boundary region.")
        return np.nan # Or handle error appropriately

    sub_omega = Omega_interwoven(num_modes)

    return sub_matrix, sub_omega

def boundary_get_symplectic_eigenvalues_by_length(C, boundary_nodes_map):
    """Calculate entanglement entropy for a region defined by boundary positions."""

    C_interpersed = to_inter(C)

    symeigs_by_length = []

    max_position = max(boundary_nodes_map.keys())

    # print(list(boundary_nodes_map.keys()))
    # print(boundary_nodes_map)

    # print(C_interpersed.shape)
    # print(max_position)

    for length in np.arange(max_position+1):

        symeigs_by_length.append(
            calculate_symplectic_eigenvalues(C_interpersed, np.arange(length+1), boundary_nodes_map)
        )

    return symeigs_by_length

def boundary_get_EE_by_length(C, boundary_nodes_map, average_over_two=False, average_over_half=False,  average_over_all=False, probe_leg_length=2, renyi=False):
    """Calculate entanglement entropy for a region defined by boundary positions."""

    if average_over_all:
        iC = to_inter(C)
        entropies = []
        for shift in tqdm(range(len(C) // 2)):
            shifted_C = np.roll(iC, shift=(shift*2, shift*2), axis=(0, 1))
            shifted_C = to_seq(shifted_C)

            entropies.append(
                np.array(boundary_get_EE_by_length(shifted_C, boundary_nodes_map, average_over_two=False, average_over_all=False, renyi=renyi))
            )
        
        return np.mean(entropies, axis=0)

    elif average_over_half:
        iC = to_inter(C)
        entropies = []
        for shift in tqdm(range(len(C) // (2 * probe_leg_length))):
            shifted_C = np.roll(iC, shift=(shift*probe_leg_length, shift*probe_leg_length), axis=(0, 1))
            shifted_C = to_seq(shifted_C)

            entropies.append(
                np.array(boundary_get_EE_by_length(shifted_C, boundary_nodes_map, average_over_two=False, average_over_all=False, renyi=renyi))
            )
        
        return np.mean(entropies, axis=0)




    # this option is not really physically justified, it's not like shifting by two leaves the system invariant.
    elif average_over_two:

        # roll the matrix
        C2 = to_inter(C)
        C2 = np.roll(C2, shift=(probe_leg_length, probe_leg_length), axis=(0, 1))
        C2 = to_seq(C2)

        # plt.figure()
        # plt.imshow(np.abs(C), norm=mpl.colors.LogNorm())
        # plt.figure()
        # plt.imshow(np.abs(C2), norm=mpl.colors.LogNorm())
        
        return list((
            np.array(boundary_get_EE_by_length(C, boundary_nodes_map, average_over_two=False, renyi=renyi))
            + np.array(boundary_get_EE_by_length(C2, boundary_nodes_map, average_over_two=False,renyi=renyi))
        ) / 2)

    # plt.imshow(np.abs(iC), norm=mpl.colors.LogNorm())

    C_interpersed = to_inter(C)

    entropies = []

    max_position = max(boundary_nodes_map.keys())

    # print(list(boundary_nodes_map.keys()))
    # print(boundary_nodes_map)

    # print(C_interpersed.shape)
    # print(max_position)

    for length in np.arange(max_position+1):

        entropies.append(
            calculate_boundary_entropy(C_interpersed, np.arange(length+1), boundary_nodes_map, renyi=renyi)
        )

    return [0] + entropies


def williamson_transform(gamma):
    """
    Compute the Williamson transform W and symplectic eigenvalues sigma
    for a symmetric positive-definite covariance matrix gamma_A.
    Implements eqs. (37)-(41) of Coser et al. (2017).
    
    Parameters
    ----------
    gamma : (2N, 2N) ndarray
        Covariance matrix (symmetric, positive definite).
        
    Returns
    -------
    W : (2N, 2N) ndarray
        Symplectic matrix satisfying gamma = W.T @ (D⊕D) @ W.
    sigma : (N,) ndarray
        Symplectic eigenvalues.
    """
    # Check dimensions
    dim = gamma.shape[0]
    assert gamma.shape == (dim, dim) and dim % 2 == 0
    N = dim // 2
    
    # Standard symplectic form J
    J = np.block([
        [np.zeros((N, N)), np.eye(N)],
        [-np.eye(N),       np.zeros((N, N))]
    ])
    
    # Step 1: sqrt of gamma
    sqrt_gamma = sqrtm(gamma)
    
    # Step 2: gamma_hat = sqrt(gamma) @ J @ sqrt(gamma)
    gamma_hat = sqrt_gamma @ J @ sqrt_gamma
    
    # Step 3: |gamma_hat| = sqrtm(gamma_hat @ gamma_hat.T)
    abs_gamma_hat = sqrtm(gamma_hat @ gamma_hat.T)
    
    # Step 4: diagonalize |gamma_hat|
    w, Q = eigh(abs_gamma_hat)
    
    # Sort eigenvalues descending
    idx = np.argsort(w)[::-1]
    w = w[idx]
    Q = Q[:, idx]
    
    # Symplectic eigenvalues: half the sorted entries
    sigma = w[::2]
    
    # Step 5: build D^{-1/2} ⊕ D^{-1/2}
    inv_sqrt_sigma = 1.0 / np.sqrt(sigma)
    D_inv_sqrt = np.diag(np.concatenate([inv_sqrt_sigma, inv_sqrt_sigma]))
    
    # Step 6: W = (D^{-1/2}⊕D^{-1/2}) @ Q.T @ sqrt_gamma
    W = D_inv_sqrt @ Q.T @ sqrt_gamma
    
    return W, sigma

def compute_K_from_W(W):
    """
    Compute the orthogonal factor K from the symplectic matrix W
    via polar decomposition: W = K @ E_R.
    """
    M = W.T @ W
    M_inv_sqrt = inv(sqrtm(M))
    K = W @ M_inv_sqrt
    return K

def mode_participation_from_K(K):
    """
    Compute the mode participation matrix p_{k,i} from the symplectic-orthogonal K.
    Implements eq. (47) of Coser et al. (2017).
    """
    N = K.shape[0] // 2
    U = K[:N, :N]
    Y = K[:N, N:]
    Z = K[N:, :N]
    V = K[N:, N:]
    p = 0.5 * (U**2 + Y**2 + Z**2 + V**2)
    return p

# Example usage:
#   W, sigma = williamson_transform(gamma_A)
#   K = compute_K_from_W(W)
#   p = mode_participation_from_K(K)

def compute_mode_participation(gamma):
    """
    Compute the mode participation matrix from the covariance matrix gamma. Requires covariance matrix to be sequential.
    
    Parameters
    ----------
    gamma : (2N, 2N) ndarray
        Covariance matrix (symmetric, positive definite).
        
    Returns
    -------
    p : (2N, 2N) ndarray
        Mode participation matrix.
    """
    W, sigma = williamson_transform(gamma)
    K = compute_K_from_W(W)
    p = mode_participation_from_K(K)
    
    return p, sigma

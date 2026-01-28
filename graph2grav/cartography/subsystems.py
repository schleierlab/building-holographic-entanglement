"""Utilities for extracting subsystem covariances and computing mutual information."""

import numpy as np
from graph2grav.gaussian import entropy, to_seq, to_inter


def extract_subsystem_covariance(cov, indices, ordering='seq'):
    """Extract subsystem covariance matrix for specified modes.

    Given a full system covariance matrix, extract the covariance matrix
    for a subsystem consisting of the modes specified by indices.

    Args:
        cov: Full covariance matrix (2N x 2N) where N is number of modes
        indices: List or array of mode indices to extract (0-indexed, range 0 to N-1)
        ordering: Covariance matrix ordering, either:
                  'seq' - sequential (q1, q2, ..., qN, p1, p2, ..., pN) [default]
                  'inter' - interspersed (q1, p1, q2, p2, ...)

    Returns:
        Subsystem covariance matrix (2M x 2M) where M = len(indices)
        Same ordering as input.

    Example:
        >>> # 3-mode system, extract modes 0 and 2
        >>> cov_full = np.eye(6)  # 2*3 = 6 dimensional
        >>> indices = [0, 2]
        >>> cov_sub = extract_subsystem_covariance(cov_full, indices)
        >>> cov_sub.shape
        (4, 4)  # 2*2 = 4 dimensional

    Notes:
        - For sequential ordering: positions are indices [0, N), momenta are [N, 2N)
        - For interspersed ordering: mode i has position at 2i, momentum at 2i+1
        - The subsystem maintains the same ordering structure as the input
    """
    # Validate inputs
    full_size = cov.shape[0]
    if full_size % 2 != 0:
        raise ValueError("Covariance matrix must have even dimensions")

    N = full_size // 2  # Number of modes in full system
    indices = np.array(indices, dtype=int)

    if np.any(indices < 0) or np.any(indices >= N):
        raise ValueError(f"Indices must be between 0 and {N-1}")

    M = len(indices)  # Number of modes in subsystem

    # Extract based on ordering
    if ordering == 'seq':
        # Sequential: [q1, q2, ..., qN, p1, p2, ..., pN]
        # Position indices: indices
        # Momentum indices: indices + N
        pos_indices = indices
        mom_indices = indices + N
        all_indices = np.concatenate([pos_indices, mom_indices])

    elif ordering == 'inter':
        # Interspersed: [q1, p1, q2, p2, ...]
        # Mode i has position at 2i, momentum at 2i+1
        all_indices = np.zeros(2 * M, dtype=int)
        for i, idx in enumerate(indices):
            all_indices[2*i] = 2*idx      # position
            all_indices[2*i + 1] = 2*idx + 1  # momentum
    else:
        raise ValueError(f"Unknown ordering '{ordering}'. Use 'seq' or 'inter'.")

    # Extract subsystem
    subsystem_cov = cov[np.ix_(all_indices, all_indices)]

    return subsystem_cov


def mutual_information(cov, indices_A, indices_B, Omega=None, ordering='seq'):
    """Compute mutual information I(A:B) between two subsystems.

    Mutual information quantifies correlations between subsystems A and B:
        I(A:B) = S_A + S_B - S_{AB}

    where S_A, S_B are individual entropies and S_{AB} is joint entropy.

    Args:
        cov: Full covariance matrix (2N x 2N)
        indices_A: Indices of subsystem A
        indices_B: Indices of subsystem B
        Omega: Symplectic form. If None, uses standard form for given ordering.
        ordering: Covariance matrix ordering ('seq' or 'inter')

    Returns:
        Mutual information I(A:B) (non-negative real number)

    Example:
        >>> # Compute MI between modes 0,1 and modes 2,3
        >>> cov = get_some_covariance(N=4)
        >>> MI = mutual_information(cov, [0, 1], [2, 3])

    Notes:
        - I(A:B) ≥ 0 (non-negative)
        - I(A:B) = 0 iff A and B are uncorrelated
        - I(A:B) = I(B:A) (symmetric)
        - For subsystems A, B:  indices must be disjoint
    """
    indices_A = np.array(indices_A, dtype=int)
    indices_B = np.array(indices_B, dtype=int)

    # Check for overlap
    if len(np.intersect1d(indices_A, indices_B)) > 0:
        raise ValueError("Subsystems A and B must be disjoint (no overlapping indices)")

    # Get subsystem covariances
    cov_A = extract_subsystem_covariance(cov, indices_A, ordering=ordering)
    cov_B = extract_subsystem_covariance(cov, indices_B, ordering=ordering)

    # Joint subsystem
    indices_AB = np.concatenate([indices_A, indices_B])
    cov_AB = extract_subsystem_covariance(cov, indices_AB, ordering=ordering)

    # Create symplectic forms if not provided
    if Omega is None:
        N_A = len(indices_A)
        N_B = len(indices_B)
        N_AB = N_A + N_B

        if ordering == 'seq':
            # Sequential ordering: Omega = [[0, I], [-I, 0]]
            Omega_A = np.block([[np.zeros((N_A, N_A)), np.eye(N_A)],
                               [-np.eye(N_A), np.zeros((N_A, N_A))]])
            Omega_B = np.block([[np.zeros((N_B, N_B)), np.eye(N_B)],
                               [-np.eye(N_B), np.zeros((N_B, N_B))]])
            Omega_AB = np.block([[np.zeros((N_AB, N_AB)), np.eye(N_AB)],
                                [-np.eye(N_AB), np.zeros((N_AB, N_AB))]])
        else:  # interspersed
            # Interspersed: Omega = block diagonal [[0, 1], [-1, 0]] for each mode
            def make_omega_inter(N):
                """Construct symplectic form Ω for N modes in interspersed ordering."""
                omega_block = np.array([[0, 1], [-1, 0]])
                return np.kron(np.eye(N), omega_block)

            Omega_A = make_omega_inter(N_A)
            Omega_B = make_omega_inter(N_B)
            Omega_AB = make_omega_inter(N_AB)
    else:
        # Use provided Omega (must extract subsystem symplectic forms)
        # This is more complex - for now, assume None or handle externally
        raise NotImplementedError("Custom Omega not yet supported. Use None for automatic.")

    # Compute entropies
    S_A = entropy(cov_A, Omega_A)
    S_B = entropy(cov_B, Omega_B)
    S_AB = entropy(cov_AB, Omega_AB)

    # Mutual information
    MI = S_A + S_B - S_AB

    # Handle numerical errors (MI should be non-negative)
    if MI < 0 and MI > -1e-10:
        MI = 0.0
    elif MI < -1e-10:
        import warnings
        warnings.warn(f"Mutual information is negative: {MI}. This may indicate numerical issues.")

    return MI

"""Gaussian vector fixed-point equation diagnostics.

This module implements the covariance-matrix version of the VFPE diagnostics
used by Li, Lin, and McGreevy.  Public functions use the conventional Gaussian
covariance normalization ``C_ij = <{R_i, R_j}> / 2`` and sequential phase-space
ordering ``(q_1, ..., q_N, p_1, ..., p_N)``.

``graph2grav.gaussian.Analysis`` uses twice this covariance normalization.  Use
``boundary_covariance_for_vfpe`` to reorder and rescale an analysis result
before applying these diagnostics.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np
import scipy.linalg as la


def symplectic_form(n_modes: int) -> np.ndarray:
    """Return the symplectic form in sequential ``(q..., p...)`` ordering."""
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    eye = np.eye(n_modes)
    zero = np.zeros((n_modes, n_modes))
    return np.block([[zero, eye], [-eye, zero]])


def _validate_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if covariance.shape[0] % 2:
        raise ValueError("covariance dimension must be even")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values")
    return 0.5 * (covariance + covariance.T)


def restrict_covariance(covariance: np.ndarray, modes: Sequence[int]) -> np.ndarray:
    """Restrict a covariance matrix to the selected modes."""
    covariance = _validate_covariance(covariance)
    n_modes = covariance.shape[0] // 2
    modes = [int(mode) for mode in modes]
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates")
    if any(mode < 0 or mode >= n_modes for mode in modes):
        raise ValueError("mode index is outside the covariance")
    indices = modes + [mode + n_modes for mode in modes]
    return covariance[np.ix_(indices, indices)]


def symplectic_eigenvalues(covariance: np.ndarray) -> np.ndarray:
    """Return the positive symplectic eigenvalues of a covariance matrix."""
    covariance = _validate_covariance(covariance)
    n_modes = covariance.shape[0] // 2
    eigs = np.linalg.eigvals(1j * symplectic_form(n_modes) @ covariance)
    positive = np.sort(np.real(eigs[np.real(eigs) > 1e-10]))
    if positive.size != n_modes:
        # The spectrum is +/- nu with each absolute value appearing twice.
        absolute = np.sort(np.abs(eigs))
        positive = np.real_if_close(absolute[::2]).astype(float)
    return positive


def gaussian_entropy(covariance: np.ndarray, *, pure_mode_floor: float = 1e-12) -> float:
    """Return the von Neumann entropy of a zero-mean Gaussian state."""
    if pure_mode_floor <= 0:
        raise ValueError("pure_mode_floor must be positive")
    occupation = np.maximum(symplectic_eigenvalues(covariance) - 0.5, pure_mode_floor)
    entropy = (occupation + 1) * np.log(occupation + 1) - occupation * np.log(occupation)
    return float(np.sum(entropy))


def _symmetric_matrix_function(matrix: np.ndarray, function) -> np.ndarray:
    values, vectors = la.eigh(0.5 * (matrix + matrix.T))
    return (vectors * function(values)) @ vectors.T


def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    return _symmetric_matrix_function(matrix, np.sqrt)


def _inverse_symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    return _symmetric_matrix_function(matrix, lambda values: 1 / np.sqrt(values))


def _entanglement_kernel(
    symplectic_eigenvalues_: np.ndarray,
    *,
    pure_mode_floor: float,
) -> np.ndarray:
    """Return ``2 nu arccoth(2 nu)`` with stable near-pure handling."""
    delta = np.maximum(symplectic_eigenvalues_ - 0.5, pure_mode_floor)
    return (0.5 + delta) * (np.log1p(delta) - np.log(delta))


def modular_hamiltonian_from_xp(
    x: np.ndarray,
    p: np.ndarray,
    *,
    pure_mode_floor: float = 1e-12,
) -> np.ndarray:
    """Return the modular quadratic form for a covariance with no qp block."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if x.shape != p.shape or x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x and p must be square matrices with the same shape")
    if pure_mode_floor <= 0:
        raise ValueError("pure_mode_floor must be positive")

    sqrt_x = _symmetric_sqrt(x)
    inv_sqrt_x = _inverse_symmetric_sqrt(x)
    nu2, vectors = la.eigh(0.5 * (sqrt_x @ p @ sqrt_x + sqrt_x @ p.T @ sqrt_x))
    kernel = _entanglement_kernel(
        np.sqrt(np.maximum(nu2, 0)), pure_mode_floor=pure_mode_floor
    )
    hq = inv_sqrt_x @ ((vectors * kernel) @ vectors.T) @ inv_sqrt_x

    sqrt_p = _symmetric_sqrt(p)
    inv_sqrt_p = _inverse_symmetric_sqrt(p)
    nu2, vectors = la.eigh(0.5 * (sqrt_p @ x @ sqrt_p + sqrt_p @ x.T @ sqrt_p))
    kernel = _entanglement_kernel(
        np.sqrt(np.maximum(nu2, 0)), pure_mode_floor=pure_mode_floor
    )
    hp = inv_sqrt_p @ ((vectors * kernel) @ vectors.T) @ inv_sqrt_p

    zero = np.zeros_like(hq)
    h = np.block([[hq, zero], [zero, hp]])
    return 0.5 * (h + h.T)


def _williamson_modular_hamiltonian(
    covariance: np.ndarray,
    *,
    pure_mode_floor: float,
) -> np.ndarray:
    """Compute a general real modular form through Williamson modes."""
    n_modes = covariance.shape[0] // 2
    values, vectors = la.eigh(covariance)
    if np.min(values) <= 0:
        raise ValueError("covariance must be positive definite")
    sqrt_covariance = (vectors * np.sqrt(values)) @ vectors.T

    antisymmetric = sqrt_covariance @ symplectic_form(n_modes) @ sqrt_covariance
    schur, orthogonal = la.schur(antisymmetric, output="real")
    symplectic_eigenvalues_ = np.empty(n_modes)
    for mode in range(n_modes):
        row = 2 * mode
        if schur[row, row + 1] < 0:
            orthogonal[:, [row, row + 1]] = orthogonal[:, [row + 1, row]]
        symplectic_eigenvalues_[mode] = abs(schur[row, row + 1])

    repeated_nu = np.repeat(symplectic_eigenvalues_, 2)
    williamson = sqrt_covariance @ orthogonal @ np.diag(1 / np.sqrt(repeated_nu))
    regulated_nu = np.maximum(symplectic_eigenvalues_, 0.5 + pure_mode_floor)
    energies = np.log((regulated_nu + 0.5) / (regulated_nu - 0.5))
    inverse = la.inv(williamson)
    h = inverse.T @ np.diag(np.repeat(energies, 2)) @ inverse
    return 0.5 * (h + h.T)


def modular_hamiltonian_matrix(
    covariance: np.ndarray,
    *,
    pure_mode_floor: float = 1e-12,
    qp_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return ``h`` for ``K = R.T @ h @ R / 2 + const``.

    A stable eigendecomposition is used when the covariance has no qp block.
    A real Williamson decomposition is used otherwise.
    """
    covariance = _validate_covariance(covariance)
    n_modes = covariance.shape[0] // 2
    x = covariance[:n_modes, :n_modes]
    p = covariance[n_modes:, n_modes:]
    qp = covariance[:n_modes, n_modes:]
    if np.max(np.abs(qp)) <= qp_tolerance:
        return modular_hamiltonian_from_xp(x, p, pure_mode_floor=pure_mode_floor)

    return _williamson_modular_hamiltonian(
        covariance, pure_mode_floor=pure_mode_floor
    )


def embed_quadratic(local_h: np.ndarray, modes: Sequence[int], n_modes: int) -> np.ndarray:
    """Embed a local quadratic form into a full phase space."""
    modes = [int(mode) for mode in modes]
    if local_h.shape != (2 * len(modes), 2 * len(modes)):
        raise ValueError("local_h shape does not match modes")
    local_indices = modes + [mode + n_modes for mode in modes]
    full_h = np.zeros((2 * n_modes, 2 * n_modes))
    full_h[np.ix_(local_indices, local_indices)] = local_h
    return full_h


def quadratic_expectation(h: np.ndarray, covariance: np.ndarray) -> float:
    """Return the expectation of the quadratic part of a modular operator."""
    return float(0.5 * np.trace(h @ covariance))


def quadratic_variance(h: np.ndarray, covariance: np.ndarray) -> float:
    """Return the variance of ``R.T @ h @ R / 2`` in a Gaussian state."""
    covariance = _validate_covariance(covariance)
    n_modes = covariance.shape[0] // 2
    gamma = covariance + 0.5j * symplectic_form(n_modes)
    variance = 0.25 * np.einsum("ij,kl,ik,jl->", h, h, gamma, gamma)
    variance += 0.25 * np.einsum("ij,kl,il,jk->", h, h, gamma, gamma)
    if abs(np.imag(variance)) > 1e-6 * max(1.0, abs(np.real(variance))):
        raise ValueError("quadratic variance has a non-negligible imaginary part")
    return float(np.real(variance))


def contiguous_modes(start: int, length: int, n_sites: int) -> list[int]:
    """Return a periodic contiguous interval of mode indices."""
    if length < 1 or length > n_sites:
        raise ValueError("length must be between 1 and n_sites")
    return [(start + offset) % n_sites for offset in range(length)]


def circular_cross_ratio(lengths: tuple[int, int, int], n_sites: int) -> float:
    """Return the circular continuum cross-ratio for adjacent A, B, and C."""
    a_len, b_len, c_len = lengths
    if min(lengths) < 1 or a_len + b_len + c_len >= n_sites:
        raise ValueError("A, B, and C must be nonempty and leave a nonempty complement")
    numerator = np.sin(np.pi * a_len / n_sites) * np.sin(np.pi * c_len / n_sites)
    denominator = np.sin(np.pi * (a_len + b_len) / n_sites) * np.sin(
        np.pi * (b_len + c_len) / n_sites
    )
    return float(numerator / denominator)


def balanced_four_arc_lengths(n_sites: int) -> tuple[int, int, int, int]:
    """Split a circle into four nearly equal nonempty integer arcs."""
    if n_sites < 4:
        raise ValueError("at least four sites are required")
    base, remainder = divmod(n_sites, 4)
    lengths = [base] * 4
    for offset in range(remainder):
        lengths[3 - offset] += 1
    return tuple(lengths)  # type: ignore[return-value]


def _region_entropy(
    covariance: np.ndarray,
    modes: Sequence[int],
    *,
    pure_mode_floor: float,
    cache: MutableMapping[tuple[int, ...], float] | None,
) -> float:
    key = tuple(modes)
    if cache is not None and key in cache:
        return cache[key]
    value = gaussian_entropy(
        restrict_covariance(covariance, modes), pure_mode_floor=pure_mode_floor
    )
    if cache is not None:
        cache[key] = value
    return value


def modular_term(
    covariance: np.ndarray,
    modes: Sequence[int],
    *,
    pure_mode_floor: float = 1e-12,
    cache: MutableMapping[tuple[int, ...], np.ndarray] | None = None,
) -> np.ndarray:
    """Return an embedded region modular-Hamiltonian quadratic form."""
    key = tuple(modes)
    if cache is not None and key in cache:
        return cache[key]
    n_modes = covariance.shape[0] // 2
    local_covariance = restrict_covariance(covariance, modes)
    local_h = modular_hamiltonian_matrix(
        local_covariance, pure_mode_floor=pure_mode_floor
    )
    value = embed_quadratic(local_h, modes, n_modes)
    if cache is not None:
        cache[key] = value
    return value


def _vfpe_regions(
    *, a_start: int, lengths: tuple[int, int, int], n_sites: int
) -> dict[str, list[int]]:
    a_len, b_len, c_len = lengths
    circular_cross_ratio(lengths, n_sites)
    a = contiguous_modes(a_start, a_len, n_sites)
    b = contiguous_modes(a_start + a_len, b_len, n_sites)
    c = contiguous_modes(a_start + a_len + b_len, c_len, n_sites)
    return {"a": a, "b": b, "c": c, "ab": a + b, "bc": b + c, "abc": a + b + c}


def vfpe_operator_from_covariance(
    covariance: np.ndarray,
    *,
    a_start: int,
    lengths: tuple[int, int, int],
    eta: float | None = None,
    pure_mode_floor: float = 1e-12,
    modular_cache: MutableMapping[tuple[int, ...], np.ndarray] | None = None,
) -> np.ndarray:
    """Assemble the local VFPE operator ``K_Delta`` for adjacent A, B, C."""
    covariance = _validate_covariance(covariance)
    n_sites = covariance.shape[0] // 2
    regions = _vfpe_regions(a_start=a_start, lengths=lengths, n_sites=n_sites)
    eta = circular_cross_ratio(lengths, n_sites) if eta is None else float(eta)
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie between zero and one")

    terms = {
        name: modular_term(
            covariance,
            modes,
            pure_mode_floor=pure_mode_floor,
            cache=modular_cache,
        )
        for name, modes in regions.items()
    }
    delta = terms["ab"] + terms["bc"] - terms["a"] - terms["c"]
    cmi = terms["ab"] + terms["bc"] - terms["b"] - terms["abc"]
    return eta * delta + (1 - eta) * cmi


def vfpe_entropy_expectation(
    covariance: np.ndarray,
    *,
    a_start: int,
    lengths: tuple[int, int, int],
    eta: float | None = None,
    pure_mode_floor: float = 1e-12,
    entropy_cache: MutableMapping[tuple[int, ...], float] | None = None,
) -> float:
    """Return ``<K_Delta>`` from entropies, including modular constants."""
    covariance = _validate_covariance(covariance)
    n_sites = covariance.shape[0] // 2
    regions = _vfpe_regions(a_start=a_start, lengths=lengths, n_sites=n_sites)
    eta = circular_cross_ratio(lengths, n_sites) if eta is None else float(eta)
    entropies = {
        name: _region_entropy(
            covariance,
            modes,
            pure_mode_floor=pure_mode_floor,
            cache=entropy_cache,
        )
        for name, modes in regions.items()
    }
    delta = entropies["ab"] + entropies["bc"] - entropies["a"] - entropies["c"]
    cmi = entropies["ab"] + entropies["bc"] - entropies["b"] - entropies["abc"]
    return float(eta * delta + (1 - eta) * cmi)


def vfpe_residual_from_covariance(
    covariance: np.ndarray,
    *,
    a_start: int,
    lengths: tuple[int, int, int],
    eta: float | None = None,
    pure_mode_floor: float = 1e-12,
    normalize: bool = True,
    modular_cache: MutableMapping[tuple[int, ...], np.ndarray] | None = None,
    entropy_cache: MutableMapping[tuple[int, ...], float] | None = None,
) -> dict[str, float]:
    """Return variance-based local VFPE diagnostics.

    ``raw_residual`` is the standard deviation of ``K_Delta``.  The preferred
    dimensionless quantity, ``relative_residual``, divides it by the full
    entropy expectation, which includes the additive modular constants.
    """
    covariance = _validate_covariance(covariance)
    n_sites = covariance.shape[0] // 2
    eta_value = circular_cross_ratio(lengths, n_sites) if eta is None else float(eta)
    operator = vfpe_operator_from_covariance(
        covariance,
        a_start=a_start,
        lengths=lengths,
        eta=eta_value,
        pure_mode_floor=pure_mode_floor,
        modular_cache=modular_cache,
    )
    quadratic_mean = quadratic_expectation(operator, covariance)
    variance = max(0.0, quadratic_variance(operator, covariance))
    raw_residual = float(np.sqrt(variance))
    entropy_expectation = vfpe_entropy_expectation(
        covariance,
        a_start=a_start,
        lengths=lengths,
        eta=eta_value,
        pure_mode_floor=pure_mode_floor,
        entropy_cache=entropy_cache,
    )
    relative = (
        raw_residual / abs(entropy_expectation)
        if abs(entropy_expectation) > 1e-14
        else float("nan")
    )
    return {
        "eta": eta_value,
        "quadratic_mean": quadratic_mean,
        "entropy_expectation": entropy_expectation,
        "variance": variance,
        "raw_residual": raw_residual,
        "relative_residual": relative,
        "residual": relative if normalize else raw_residual,
    }


def scan_vfpe_origins(
    covariance: np.ndarray,
    *,
    lengths: tuple[int, int, int],
    starts: Sequence[int] | None = None,
    eta: float | None = None,
    pure_mode_floor: float = 1e-12,
) -> list[dict[str, float]]:
    """Evaluate the local VFPE residual around all requested circle origins."""
    covariance = _validate_covariance(covariance)
    n_sites = covariance.shape[0] // 2
    starts = range(n_sites) if starts is None else starts
    modular_cache: dict[tuple[int, ...], np.ndarray] = {}
    entropy_cache: dict[tuple[int, ...], float] = {}
    results = []
    for start in starts:
        result = vfpe_residual_from_covariance(
            covariance,
            a_start=int(start),
            lengths=lengths,
            eta=eta,
            pure_mode_floor=pure_mode_floor,
            modular_cache=modular_cache,
            entropy_cache=entropy_cache,
        )
        result["a_start"] = int(start)
        results.append(result)
    return results


def c_delta_operator_from_covariance(
    covariance: np.ndarray,
    *,
    start: int = 0,
    pure_mode_floor: float = 1e-12,
    modular_cache: MutableMapping[tuple[int, ...], np.ndarray] | None = None,
) -> np.ndarray:
    """Return the four-equal-arc ``c_hat_Delta`` quadratic operator."""
    covariance = _validate_covariance(covariance)
    n_sites = covariance.shape[0] // 2
    if n_sites % 4:
        raise ValueError("c_hat_Delta requires a boundary size divisible by four")
    arc_length = n_sites // 4
    arcs = [contiguous_modes(start + i * arc_length, arc_length, n_sites) for i in range(4)]
    operator = np.zeros_like(covariance)
    for index, arc in enumerate(arcs):
        next_arc = arcs[(index + 1) % 4]
        operator += modular_term(
            covariance,
            arc + next_arc,
            pure_mode_floor=pure_mode_floor,
            cache=modular_cache,
        )
        operator -= modular_term(
            covariance,
            next_arc,
            pure_mode_floor=pure_mode_floor,
            cache=modular_cache,
        )
    return 3 * operator / (2 * np.log(2))


def c_delta_entropy_expectation(
    covariance: np.ndarray,
    *,
    start: int = 0,
    pure_mode_floor: float = 1e-12,
    entropy_cache: MutableMapping[tuple[int, ...], float] | None = None,
) -> float:
    """Return the entropy expectation of the four-arc ``c_hat_Delta``."""
    covariance = _validate_covariance(covariance)
    n_sites = covariance.shape[0] // 2
    if n_sites % 4:
        raise ValueError("c_hat_Delta requires a boundary size divisible by four")
    arc_length = n_sites // 4
    arcs = [contiguous_modes(start + i * arc_length, arc_length, n_sites) for i in range(4)]
    value = 0.0
    for index, arc in enumerate(arcs):
        next_arc = arcs[(index + 1) % 4]
        value += _region_entropy(
            covariance,
            arc + next_arc,
            pure_mode_floor=pure_mode_floor,
            cache=entropy_cache,
        )
        value -= _region_entropy(
            covariance,
            next_arc,
            pure_mode_floor=pure_mode_floor,
            cache=entropy_cache,
        )
    return float(3 * value / (2 * np.log(2)))


def c_delta_residual_from_covariance(
    covariance: np.ndarray,
    *,
    start: int = 0,
    pure_mode_floor: float = 1e-12,
) -> dict[str, float]:
    """Return the mean and standard deviation of the paper's ``c_hat_Delta``."""
    covariance = _validate_covariance(covariance)
    operator = c_delta_operator_from_covariance(
        covariance, start=start, pure_mode_floor=pure_mode_floor
    )
    quadratic_mean = quadratic_expectation(operator, covariance)
    variance = max(0.0, quadratic_variance(operator, covariance))
    raw_residual = float(np.sqrt(variance))
    entropy_expectation = c_delta_entropy_expectation(
        covariance, start=start, pure_mode_floor=pure_mode_floor
    )
    relative = (
        raw_residual / abs(entropy_expectation)
        if abs(entropy_expectation) > 1e-14
        else float("nan")
    )
    return {
        "quadratic_mean": quadratic_mean,
        "entropy_expectation": entropy_expectation,
        "variance": variance,
        "raw_residual": raw_residual,
        "relative_residual": relative,
    }


def _convert_boundary_covariance(
    graph: Any,
    raw_covariance: np.ndarray,
    *,
    boundary_label: str = "b0",
    validate_physical: bool = True,
) -> np.ndarray:
    if not graph.graph.get("periodic", False):
        raise ValueError("VFPE boundary intervals require a periodic graph")
    if graph.graph.get("number_of_boundaries", 1) != 1:
        raise ValueError("VFPE adapter currently supports exactly one boundary")

    boundary_nodes = [
        node for node, attributes in graph.nodes(data=True) if attributes.get(boundary_label, False)
    ]
    positions = [graph.nodes[node].get("position_in_boundary") for node in boundary_nodes]
    if any(position is None for position in positions):
        raise ValueError("every boundary node needs position_in_boundary")
    if sorted(positions) != list(range(len(boundary_nodes))):
        raise ValueError("boundary positions must be unique and contiguous from zero")

    raw = np.asarray(raw_covariance, dtype=float)
    boundary_length = len(boundary_nodes)
    if raw.shape != (2 * boundary_length, 2 * boundary_length):
        raise ValueError("analysis boundary covariance does not match the graph boundary")
    order = np.argsort(positions)
    phase_order = np.concatenate([order, order + boundary_length])
    covariance = raw[np.ix_(phase_order, phase_order)] / 2
    covariance = 0.5 * (covariance + covariance.T)

    if validate_physical:
        minimum = float(np.min(symplectic_eigenvalues(covariance)))
        if minimum < 0.5 - 1e-7:
            raise ValueError(
                f"converted boundary covariance is unphysical: min symplectic eigenvalue={minimum}"
            )
    return covariance


def boundary_covariance_for_vfpe(
    analysis: Any,
    *,
    boundary_label: str = "b0",
    validate_physical: bool = True,
) -> np.ndarray:
    """Convert an ``Analysis`` boundary state to VFPE conventions.

    Boundary modes are reordered by ``position_in_boundary`` and the
    graph2grav covariance is divided by two. The VFPE checks currently assume
    one periodic boundary with exactly one node at every boundary position.
    """
    return _convert_boundary_covariance(
        analysis.graph,
        analysis.boundary_after_bulk_measurement,
        boundary_label=boundary_label,
        validate_physical=validate_physical,
    )


def boundary_covariance_for_vfpe_from_graph(
    graph: Any,
    *,
    couplingtime: float = 1.0,
    global_presqueeze: float = 1.0,
    is_decorated: bool = False,
    boundary_label: str = "b0",
    validate_physical: bool = True,
) -> np.ndarray:
    """Prepare a boundary covariance without running entropy analysis.

    This is the efficient path for parameter sweeps. It performs the same
    quench and bulk-momentum measurement as ``Analysis`` and then converts the
    retained covariance to conventional VFPE normalization.
    """
    from graph2grav import gaussian

    if is_decorated:
        full_covariance = gaussian.covariance_from_weighted_quench(
            graph,
            couplingtime=couplingtime,
            global_presqueeze=global_presqueeze,
        )
    else:
        full_covariance = gaussian.covariance_from_unweighted_quench(
            graph,
            couplingtime=couplingtime,
            bulk_presqueeze=global_presqueeze,
            boundary_presqueeze=global_presqueeze,
        )
    boundary_indices = gaussian.indices(graph, boundary_label)
    ancilla_momenta = gaussian.momentum_indices(graph, "is_ancilla")
    boundary_covariance = gaussian.get_measured(
        full_covariance,
        boundary_indices,
        ancilla_momenta,
    )
    return _convert_boundary_covariance(
        graph,
        boundary_covariance,
        boundary_label=boundary_label,
        validate_physical=validate_physical,
    )

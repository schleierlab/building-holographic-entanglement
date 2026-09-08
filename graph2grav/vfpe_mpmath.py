"""Slow arbitrary-precision reference calculations for Gaussian VFPE checks.

This module mirrors the block-diagonal ``X/P`` path in :mod:`graph2grav.vfpe`
without changing the production NumPy implementation.  It is intended for
small graphs and precision checks, not parameter sweeps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mpmath as mp


def _mpf(value: Any) -> mp.mpf:
    """Convert existing scalar data without importing binary-float noise."""
    if isinstance(value, mp.mpf):
        return value
    return mp.mpf(str(value))


def _symmetrize(matrix: mp.matrix) -> mp.matrix:
    return (matrix + matrix.T) / 2


def _submatrix(
    matrix: mp.matrix,
    rows: Sequence[int],
    columns: Sequence[int] | None = None,
) -> mp.matrix:
    columns = rows if columns is None else columns
    return mp.matrix([[matrix[row, column] for column in columns] for row in rows])


def _block_diagonal(first: mp.matrix, second: mp.matrix) -> mp.matrix:
    result = mp.matrix(first.rows + second.rows, first.cols + second.cols)
    for row in range(first.rows):
        for column in range(first.cols):
            result[row, column] = first[row, column]
    for row in range(second.rows):
        for column in range(second.cols):
            result[first.rows + row, first.cols + column] = second[row, column]
    return result


def _weighted_adjacency(graph: Any, *, weight: str | None) -> tuple[mp.matrix, list[Any]]:
    nodes = list(graph.nodes)
    node_indices = {node: index for index, node in enumerate(nodes)}
    adjacency = mp.matrix(len(nodes), len(nodes))
    for left, right, attributes in graph.edges(data=True):
        value = _mpf(attributes.get(weight, 1.0) if weight is not None else 1.0)
        left_index = node_indices[left]
        right_index = node_indices[right]
        adjacency[left_index, right_index] += value
        if left_index != right_index:
            adjacency[right_index, left_index] += value
    return adjacency, nodes


def quench_covariance_mpmath(
    graph: Any,
    *,
    couplingtime: float | str | mp.mpf = 1.0,
    global_presqueeze: float | str | mp.mpf = 1.0,
    is_decorated: bool = False,
    dps: int = 60,
) -> mp.matrix:
    """Return the full quench covariance using arbitrary-precision arithmetic."""
    if dps < 15:
        raise ValueError("dps must be at least 15")
    with mp.workdps(dps):
        adjacency, _ = _weighted_adjacency(
            graph, weight="mweight" if is_decorated else None
        )
        n_modes = adjacency.rows
        mu = _mpf(global_presqueeze)
        time = _mpf(couplingtime)
        if mu <= 0:
            raise ValueError("global_presqueeze must be positive")

        identity = mp.eye(n_modes)
        scaled_adjacency = time * adjacency
        position = identity / mu
        momentum = mu * identity
        top_right = -(position * scaled_adjacency.T)
        bottom_left = -(scaled_adjacency * position)
        bottom_right = scaled_adjacency * position * scaled_adjacency.T + momentum

        covariance = mp.matrix(2 * n_modes, 2 * n_modes)
        blocks = (
            (position, top_right),
            (bottom_left, bottom_right),
        )
        for block_row, row_blocks in enumerate(blocks):
            for block_column, block in enumerate(row_blocks):
                for row in range(n_modes):
                    for column in range(n_modes):
                        covariance[
                            block_row * n_modes + row,
                            block_column * n_modes + column,
                        ] = block[row, column]
        return _symmetrize(covariance)


def homodyne_measure_mpmath(
    covariance: mp.matrix,
    remaining_indices: Sequence[int],
    measured_indices: Sequence[int],
) -> mp.matrix:
    """Apply the same Schur complement as ``gaussian.get_measured``."""
    remaining = [int(index) for index in remaining_indices]
    measured = [int(index) for index in measured_indices]
    retained = _submatrix(covariance, remaining)
    measured_covariance = _submatrix(covariance, measured)
    cross = _submatrix(covariance, remaining, measured)
    conditional = retained - cross * (measured_covariance**-1) * cross.T
    return _symmetrize(conditional)


def boundary_covariance_for_vfpe_mpmath(
    graph: Any,
    *,
    couplingtime: float | str | mp.mpf = 1.0,
    global_presqueeze: float | str | mp.mpf = 1.0,
    is_decorated: bool = False,
    boundary_label: str = "b0",
    dps: int = 60,
) -> mp.matrix:
    """Prepare and homodyne-measure a boundary covariance in VFPE units."""
    with mp.workdps(dps):
        covariance = quench_covariance_mpmath(
            graph,
            couplingtime=couplingtime,
            global_presqueeze=global_presqueeze,
            is_decorated=is_decorated,
            dps=dps,
        )
        nodes = list(graph.nodes)
        node_indices = {node: index for index, node in enumerate(nodes)}
        n_modes = len(nodes)
        boundary_nodes = [
            node for node in nodes if graph.nodes[node].get(boundary_label, False)
        ]
        ancilla_nodes = [
            node for node in nodes if graph.nodes[node].get("is_ancilla", False)
        ]
        positions = [
            graph.nodes[node].get("position_in_boundary") for node in boundary_nodes
        ]
        if sorted(positions) != list(range(len(boundary_nodes))):
            raise ValueError("boundary positions must be unique and contiguous from zero")

        boundary_q = [node_indices[node] for node in boundary_nodes]
        boundary_p = [index + n_modes for index in boundary_q]
        ancilla_p = [node_indices[node] + n_modes for node in ancilla_nodes]
        boundary_covariance = homodyne_measure_mpmath(
            covariance, boundary_q + boundary_p, ancilla_p
        )

        order = sorted(range(len(boundary_nodes)), key=positions.__getitem__)
        boundary_length = len(boundary_nodes)
        phase_order = order + [index + boundary_length for index in order]
        return _submatrix(boundary_covariance, phase_order) / 2


def decorated_boundary_covariance_for_vfpe_mpmath(
    graph: Any,
    *,
    couplingtime: float | str | mp.mpf = 1.0,
    global_presqueeze: float | str | mp.mpf = 1.0,
    boundary_label: str = "b0",
    dps: int = 60,
) -> mp.matrix:
    """Prepare a decorated boundary covariance without full-system matrices.

    A decorated graph is bipartite: primary bulk nodes connect to secondary
    bulk nodes and boundary probes.  Eliminating the measured momenta gives

    ``V_boundary = diag(W^-1 / mu, mu W) / 2``

    where ``W = I + alpha Cb.T (I + alpha Cs Cs.T)^-1 Cb``,
    ``alpha = (couplingtime / mu)^2``, and ``Cs``/``Cb`` are the weighted
    primary-to-secondary and primary-to-boundary adjacency blocks.
    """
    if dps < 15:
        raise ValueError("dps must be at least 15")
    with mp.workdps(dps):
        mu = _mpf(global_presqueeze)
        time = _mpf(couplingtime)
        if mu <= 0:
            raise ValueError("global_presqueeze must be positive")

        primary = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("is_ancilla", False)
            and attributes.get("tier") == "primary"
        ]
        secondary = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("is_ancilla", False)
            and attributes.get("tier") == "secondary"
        ]
        boundary = sorted(
            (
                node
                for node, attributes in graph.nodes(data=True)
                if attributes.get(boundary_label, False)
            ),
            key=lambda node: graph.nodes[node]["position_in_boundary"],
        )
        expected_nodes = set(primary) | set(secondary) | set(boundary)
        if expected_nodes != set(graph.nodes):
            raise ValueError(
                "decorated mpmath path requires primary, secondary, or boundary nodes"
            )
        if [graph.nodes[node]["position_in_boundary"] for node in boundary] != list(
            range(len(boundary))
        ):
            raise ValueError("boundary positions must be unique and contiguous from zero")

        primary_index = {node: index for index, node in enumerate(primary)}
        secondary_index = {node: index for index, node in enumerate(secondary)}
        boundary_index = {node: index for index, node in enumerate(boundary)}
        primary_secondary = mp.matrix(len(primary), len(secondary))
        primary_boundary = mp.matrix(len(primary), len(boundary))
        for left, right, attributes in graph.edges(data=True):
            if right in primary_index:
                left, right = right, left
            if left not in primary_index:
                raise ValueError("every decorated edge must touch a primary node")
            value = _mpf(attributes.get("mweight", 1.0))
            if right in secondary_index:
                primary_secondary[
                    primary_index[left], secondary_index[right]
                ] += value
            elif right in boundary_index:
                primary_boundary[
                    primary_index[left], boundary_index[right]
                ] += value
            else:
                raise ValueError(
                    "decorated edges must connect primary-secondary or primary-boundary"
                )

        alpha = (time / mu) ** 2
        primary_kernel = (
            mp.eye(len(primary))
            + alpha * primary_secondary * primary_secondary.T
        )
        transmission = (
            primary_boundary.T * (primary_kernel**-1) * primary_boundary
        )
        boundary_kernel = mp.eye(len(boundary)) + alpha * transmission
        x = (boundary_kernel**-1) / (2 * mu)
        p = mu * boundary_kernel / 2
        return _block_diagonal(_symmetrize(x), _symmetrize(p))


def _symmetric_matrix_function(matrix: mp.matrix, function) -> mp.matrix:
    values, vectors = mp.eigsy(_symmetrize(matrix))
    transformed = mp.diag([function(values[index]) for index in range(values.rows)])
    return _symmetrize(vectors * transformed * vectors.T)


def _symmetric_sqrt(matrix: mp.matrix) -> mp.matrix:
    return _symmetric_matrix_function(matrix, mp.sqrt)


def _inverse_symmetric_sqrt(matrix: mp.matrix) -> mp.matrix:
    return _symmetric_matrix_function(matrix, lambda value: 1 / mp.sqrt(value))


def _entanglement_kernel(
    symplectic_eigenvalues: Sequence[mp.mpf], *, pure_mode_floor: mp.mpf
) -> list[mp.mpf]:
    result = []
    for eigenvalue in symplectic_eigenvalues:
        delta = max(eigenvalue - mp.mpf("0.5"), pure_mode_floor)
        result.append((mp.mpf("0.5") + delta) * (mp.log1p(delta) - mp.log(delta)))
    return result


def modular_hamiltonian_from_xp_mpmath(
    x: mp.matrix,
    p: mp.matrix,
    *,
    pure_mode_floor: float | str | mp.mpf = "1e-12",
) -> mp.matrix:
    """Return the arbitrary-precision modular quadratic form for zero ``qp``."""
    if x.rows != x.cols or p.rows != p.cols or x.rows != p.rows:
        raise ValueError("x and p must be square matrices with the same shape")
    floor = _mpf(pure_mode_floor)
    if floor <= 0:
        raise ValueError("pure_mode_floor must be positive")

    sqrt_x = _symmetric_sqrt(x)
    inv_sqrt_x = _inverse_symmetric_sqrt(x)
    nu2_x, vectors_x = mp.eigsy(_symmetrize(sqrt_x * p * sqrt_x))
    kernel_x = _entanglement_kernel(
        [mp.sqrt(max(nu2_x[index], mp.mpf("0"))) for index in range(nu2_x.rows)],
        pure_mode_floor=floor,
    )
    hq = inv_sqrt_x * vectors_x * mp.diag(kernel_x) * vectors_x.T * inv_sqrt_x

    sqrt_p = _symmetric_sqrt(p)
    inv_sqrt_p = _inverse_symmetric_sqrt(p)
    nu2_p, vectors_p = mp.eigsy(_symmetrize(sqrt_p * x * sqrt_p))
    kernel_p = _entanglement_kernel(
        [mp.sqrt(max(nu2_p[index], mp.mpf("0"))) for index in range(nu2_p.rows)],
        pure_mode_floor=floor,
    )
    hp = inv_sqrt_p * vectors_p * mp.diag(kernel_p) * vectors_p.T * inv_sqrt_p
    return _block_diagonal(_symmetrize(hq), _symmetrize(hp))


def _xp_blocks(covariance: mp.matrix) -> tuple[mp.matrix, mp.matrix]:
    if covariance.rows != covariance.cols or covariance.rows % 2:
        raise ValueError("covariance must be square with even dimension")
    n_modes = covariance.rows // 2
    qp = _submatrix(covariance, range(n_modes), range(n_modes, 2 * n_modes))
    maximum_qp = max(
        (
            abs(qp[row, column])
            for row in range(n_modes)
            for column in range(n_modes)
        ),
        default=mp.mpf("0"),
    )
    if maximum_qp > mp.sqrt(mp.eps):
        raise ValueError("mpmath reference path currently requires a zero qp block")
    return (
        _submatrix(covariance, range(n_modes)),
        _submatrix(covariance, range(n_modes, 2 * n_modes)),
    )


def symplectic_eigenvalues_mpmath(covariance: mp.matrix) -> list[mp.mpf]:
    """Return symplectic eigenvalues for a block-diagonal covariance."""
    x, p = _xp_blocks(covariance)
    sqrt_x = _symmetric_sqrt(x)
    values, _ = mp.eigsy(_symmetrize(sqrt_x * p * sqrt_x))
    return [
        mp.sqrt(max(values[index], mp.mpf("0"))) for index in range(values.rows)
    ]


def gaussian_entropy_mpmath(
    covariance: mp.matrix,
    *,
    pure_mode_floor: float | str | mp.mpf = "1e-12",
) -> mp.mpf:
    floor = _mpf(pure_mode_floor)
    terms = []
    for eigenvalue in symplectic_eigenvalues_mpmath(covariance):
        occupation = max(eigenvalue - mp.mpf("0.5"), floor)
        terms.append(
            (occupation + 1) * mp.log(occupation + 1)
            - occupation * mp.log(occupation)
        )
    return mp.fsum(terms)


def restrict_covariance_mpmath(
    covariance: mp.matrix, modes: Sequence[int]
) -> mp.matrix:
    n_modes = covariance.rows // 2
    selected = [int(mode) for mode in modes]
    indices = selected + [mode + n_modes for mode in selected]
    return _submatrix(covariance, indices)


def _embed_quadratic(
    local_h: mp.matrix, modes: Sequence[int], n_modes: int
) -> mp.matrix:
    selected = [int(mode) for mode in modes]
    local_indices = selected + [mode + n_modes for mode in selected]
    result = mp.matrix(2 * n_modes, 2 * n_modes)
    for local_row, full_row in enumerate(local_indices):
        for local_column, full_column in enumerate(local_indices):
            result[full_row, full_column] = local_h[local_row, local_column]
    return result


def _contiguous_modes(start: int, length: int, n_sites: int) -> list[int]:
    return [(start + offset) % n_sites for offset in range(length)]


def _regions(
    *, a_start: int, lengths: tuple[int, int, int], n_sites: int
) -> dict[str, list[int]]:
    a_length, b_length, c_length = lengths
    if min(lengths) < 1 or sum(lengths) >= n_sites:
        raise ValueError("A, B, and C must be nonempty and leave a nonempty complement")
    a = _contiguous_modes(a_start, a_length, n_sites)
    b = _contiguous_modes(a_start + a_length, b_length, n_sites)
    c = _contiguous_modes(a_start + a_length + b_length, c_length, n_sites)
    return {"a": a, "b": b, "c": c, "ab": a + b, "bc": b + c, "abc": a + b + c}


def _circular_cross_ratio(lengths: tuple[int, int, int], n_sites: int) -> mp.mpf:
    a_length, b_length, c_length = lengths
    return (
        mp.sin(mp.pi * a_length / n_sites)
        * mp.sin(mp.pi * c_length / n_sites)
        / (
            mp.sin(mp.pi * (a_length + b_length) / n_sites)
            * mp.sin(mp.pi * (b_length + c_length) / n_sites)
        )
    )


def _modular_term(
    covariance: mp.matrix,
    modes: Sequence[int],
    *,
    pure_mode_floor: mp.mpf,
) -> mp.matrix:
    local_covariance = restrict_covariance_mpmath(covariance, modes)
    x, p = _xp_blocks(local_covariance)
    local_h = modular_hamiltonian_from_xp_mpmath(
        x, p, pure_mode_floor=pure_mode_floor
    )
    return _embed_quadratic(local_h, modes, covariance.rows // 2)


def vfpe_operator_mpmath(
    covariance: mp.matrix,
    *,
    a_start: int,
    lengths: tuple[int, int, int],
    eta: float | str | mp.mpf | None = None,
    pure_mode_floor: float | str | mp.mpf = "1e-12",
) -> mp.matrix:
    """Assemble the arbitrary-precision ``K_Delta`` quadratic form."""
    n_sites = covariance.rows // 2
    regions = _regions(a_start=a_start, lengths=lengths, n_sites=n_sites)
    eta_value = _circular_cross_ratio(lengths, n_sites) if eta is None else _mpf(eta)
    floor = _mpf(pure_mode_floor)
    terms = {
        name: _modular_term(covariance, modes, pure_mode_floor=floor)
        for name, modes in regions.items()
    }
    delta = terms["ab"] + terms["bc"] - terms["a"] - terms["c"]
    cmi = terms["ab"] + terms["bc"] - terms["b"] - terms["abc"]
    return eta_value * delta + (1 - eta_value) * cmi


def _quadratic_expectation(h: mp.matrix, covariance: mp.matrix) -> mp.mpf:
    return mp.fsum(
        h[row, column] * covariance[column, row]
        for row in range(h.rows)
        for column in range(h.cols)
    ) / 2


def _symplectic_form(n_modes: int) -> mp.matrix:
    omega = mp.matrix(2 * n_modes, 2 * n_modes)
    for mode in range(n_modes):
        omega[mode, mode + n_modes] = 1
        omega[mode + n_modes, mode] = -1
    return omega


def _quadratic_variance(h: mp.matrix, covariance: mp.matrix) -> mp.mpf:
    omega = _symplectic_form(covariance.rows // 2)
    classical_product = h * covariance
    quantum_product = h * omega
    classical_square = classical_product * classical_product
    quantum_square = quantum_product * quantum_product
    classical = mp.fsum(
        classical_square[index, index] for index in range(h.rows)
    ) / 2
    quantum = mp.fsum(
        quantum_square[index, index] for index in range(h.rows)
    ) / 8
    return classical + quantum


def _vfpe_entropy_expectation(
    covariance: mp.matrix,
    *,
    a_start: int,
    lengths: tuple[int, int, int],
    eta: mp.mpf,
    pure_mode_floor: mp.mpf,
) -> mp.mpf:
    regions = _regions(
        a_start=a_start, lengths=lengths, n_sites=covariance.rows // 2
    )
    entropies = {
        name: gaussian_entropy_mpmath(
            restrict_covariance_mpmath(covariance, modes),
            pure_mode_floor=pure_mode_floor,
        )
        for name, modes in regions.items()
    }
    delta = mp.fsum(
        [entropies["ab"], entropies["bc"], -entropies["a"], -entropies["c"]]
    )
    cmi = mp.fsum(
        [entropies["ab"], entropies["bc"], -entropies["b"], -entropies["abc"]]
    )
    return eta * delta + (1 - eta) * cmi


def vfpe_residual_from_covariance_mpmath(
    covariance: mp.matrix,
    *,
    a_start: int,
    lengths: tuple[int, int, int],
    eta: float | str | mp.mpf | None = None,
    pure_mode_floor: float | str | mp.mpf = "1e-12",
    dps: int = 60,
) -> dict[str, mp.mpf]:
    """Return high-precision VFPE diagnostics for a zero-``qp`` covariance."""
    with mp.workdps(dps):
        floor = _mpf(pure_mode_floor)
        eta_value = (
            _circular_cross_ratio(lengths, covariance.rows // 2)
            if eta is None
            else _mpf(eta)
        )
        operator = vfpe_operator_mpmath(
            covariance,
            a_start=a_start,
            lengths=lengths,
            eta=eta_value,
            pure_mode_floor=floor,
        )
        quadratic_mean = _quadratic_expectation(operator, covariance)
        variance = max(_quadratic_variance(operator, covariance), mp.mpf("0"))
        raw_residual = mp.sqrt(variance)
        entropy_expectation = _vfpe_entropy_expectation(
            covariance,
            a_start=a_start,
            lengths=lengths,
            eta=eta_value,
            pure_mode_floor=floor,
        )
        relative_residual = raw_residual / abs(entropy_expectation)
        return {
            "eta": eta_value,
            "quadratic_mean": quadratic_mean,
            "entropy_expectation": entropy_expectation,
            "variance": variance,
            "raw_residual": raw_residual,
            "relative_residual": relative_residual,
            "residual": relative_residual,
        }

"""Smoke tests for supplemental regular-graph and VFPE figure support."""

from __future__ import annotations

import numpy as np
import networkx as nx

from graph2grav.graphs import (
    RegularTilingSpec,
    decorate_regular_patch,
    generate_regular_patch,
)
from graph2grav.graphs.standard import subdivided_tree_decoration_1
from graph2grav.vfpe import (
    balanced_four_arc_lengths,
    boundary_covariance_for_vfpe_from_graph,
    c_delta_residual_from_covariance,
    symplectic_eigenvalues,
    vfpe_residual_from_covariance,
)


def test_regular_hyperbolic_patches_have_ordered_boundaries():
    for q_value in (7, 8):
        graph = generate_regular_patch(RegularTilingSpec(3, q_value, depth=2))
        boundary = [
            node for node, data in graph.nodes(data=True) if data.get("is_boundary")
        ]
        positions = sorted(graph.nodes[node]["position_in_boundary"] for node in boundary)

        assert boundary
        assert positions == list(range(len(boundary)))
        assert graph.graph["schlafli_symbol"] == f"{{3,{q_value}}}"
        assert graph.graph["periodic"] is True


def test_regular_patch_decoration_preserves_boundary_length():
    graph = generate_regular_patch(RegularTilingSpec(3, 7, depth=2))
    decorated = decorate_regular_patch(graph)

    original_boundary = sum(
        data.get("is_boundary", False) for _, data in graph.nodes(data=True)
    )
    decorated_boundary = sum(
        data.get("is_boundary", False) for _, data in decorated.nodes(data=True)
    )

    assert decorated_boundary == original_boundary
    assert decorated.graph["subdivided"] is True
    assert decorated.graph["decorated_from_regular"] is True


def test_decorated_mera_vfpe_smoke():
    graph = subdivided_tree_decoration_1(3, draw=False)
    covariance = boundary_covariance_for_vfpe_from_graph(
        graph,
        couplingtime=1.0,
        global_presqueeze=0.1,
        is_decorated=True,
    )
    arcs = balanced_four_arc_lengths(covariance.shape[0] // 2)
    vfpe = vfpe_residual_from_covariance(covariance, a_start=0, lengths=arcs[:3])
    c_delta = c_delta_residual_from_covariance(covariance)

    assert np.min(symplectic_eigenvalues(covariance)) >= 0.5 - 1e-9
    assert np.isfinite(vfpe["raw_residual"])
    assert np.isfinite(vfpe["relative_residual"])
    assert np.isfinite(c_delta["entropy_expectation"])


def test_decorated_mera_pre_decoration_hole_preserves_boundary():
    graph = subdivided_tree_decoration_1(
        5,
        draw=False,
        validate=False,
        pre_remove_up_to_depth=2,
    )
    boundary_count = sum(data.get("is_boundary", False) for _, data in graph.nodes(data=True))
    primary_bulk_depths = [
        data.get("depth")
        for _, data in graph.nodes(data=True)
        if data.get("tier") == "primary" and not data.get("is_boundary")
    ]

    assert boundary_count == 2**5
    assert nx.is_connected(graph)
    assert min(primary_bulk_depths) > 2

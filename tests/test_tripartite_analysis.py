from __future__ import annotations

import numpy as np

from graph2grav import tree_wormhole_knockout as twk
from graph2grav.tripartite_analysis import (
    BoundaryRegions,
    BoundaryTriplet,
    _sample_uniform_three_part_composition,
    evaluate_tripartite_dataset,
    evaluate_five_region_cyclic_dataset,
    five_region_cyclic_margin,
    five_region_cyclic_terms,
    rt_geodesic_pairing_for_region,
    sample_boundary_regions,
    sample_boundary_triplets,
    tripartite_from_entropy,
)


def test_tripartite_from_entropy_combines_seven_regions():
    triplet = BoundaryTriplet(a=(0,), b=(2,), c=(4,))
    entropy_by_region = {
        (0,): 1.0,
        (2,): 2.0,
        (4,): 3.0,
        (0, 2): 4.0,
        (0, 4): 5.0,
        (2, 4): 6.0,
        (0, 2, 4): 7.0,
    }

    assert tripartite_from_entropy(entropy_by_region, triplet) == -2.0


def test_sample_boundary_triplets_are_disjoint_and_reproducible():
    first = sample_boundary_triplets(16, n_samples=12, max_region_size=3, seed=4)
    second = sample_boundary_triplets(16, n_samples=12, max_region_size=3, seed=4)

    assert first == second
    for triplet in first:
        union = set(triplet.a) | set(triplet.b) | set(triplet.c)
        assert len(union) == len(triplet.a) + len(triplet.b) + len(triplet.c)


def test_gap_extra_sampler_can_populate_all_three_gaps():
    rng = np.random.default_rng(12)
    samples = [_sample_uniform_three_part_composition(12, rng) for _ in range(200)]

    assert all(np.sum(sample) == 12 for sample in samples)
    assert any(np.all(sample > 0) for sample in samples)


def test_sample_boundary_regions_are_disjoint_and_reproducible():
    first = sample_boundary_regions(30, n_regions=5, n_samples=10, max_region_size=3, seed=4)
    second = sample_boundary_regions(30, n_regions=5, n_samples=10, max_region_size=3, seed=4)

    assert first == second
    for sample in first:
        assert len(sample.regions) == 5
        union = set().union(*(set(region) for region in sample.regions))
        assert len(union) == sum(len(region) for region in sample.regions)


def test_five_region_cyclic_margin_uses_equation_4_12_terms():
    sample = BoundaryRegions(regions=((0,), (2,), (4,), (6,), (8,)))
    terms = five_region_cyclic_terms(sample)
    entropies = {term: 0.0 for term in terms}
    for term in ("abc", "bcd", "cde", "dea", "eab"):
        entropies[term] = 2.0
    for term in ("bc", "cd", "de", "ea", "ab", "abcde"):
        entropies[term] = 1.0

    assert five_region_cyclic_margin(entropies) == 4.0


def test_rt_geodesic_pairing_returns_minimizing_pairs():
    pairing = rt_geodesic_pairing_for_region((1, 2, 5, 6), boundary_length=12)

    assert len(pairing["cut_coordinates"]) == 4
    assert len(pairing["pairs"]) == 2
    pair_cost = sum(
        np.log((12 / np.pi) * np.sin(np.pi * min(abs(left - right), 12 - abs(left - right)) / 12))
        for left, right in pairing["pairs"]
    )
    assert np.isclose(pair_cost, pairing["log_chord_sum"])


def test_evaluate_tripartite_dataset_can_use_hyperbolic_embedding():
    config = twk.TreeWormholeConfig(
        graph_kind="hyperbolic",
        hyperbolic_p=3,
        hyperbolic_q=7,
        hyperbolic_depth=2,
        hyperbolic_boundary_shells=1,
        use_decorated=False,
        couplingtime=1.0,
        presqueeze=0.2,
    )
    graph = twk.build_graph(config)
    payload = twk.boundary_after_measurement_payload(
        graph,
        couplingtime=config.couplingtime,
        presqueeze=config.presqueeze,
    )
    triplets = sample_boundary_triplets(payload["boundary_length"], n_samples=3, max_region_size=2, seed=2)
    result = evaluate_tripartite_dataset(
        payload,
        triplets,
        central_charge=1.0,
        graph=graph,
        rt_geometry="hyperbolic_embedding",
    )

    assert result["rt_geometry"] == "hyperbolic_embedding"
    assert result["rt_coefficient"] == 1.0 / 6.0
    assert np.all(np.isfinite(result["rt_i3"]))


def test_evaluate_tripartite_dataset_smoke():
    config = twk.TreeWormholeConfig(
        graph_kind="mera",
        mera_depth=3,
        use_decorated=False,
        couplingtime=1.0,
        presqueeze=0.2,
    )
    graph = twk.build_graph(config)
    payload = twk.boundary_after_measurement_payload(
        graph,
        couplingtime=config.couplingtime,
        presqueeze=config.presqueeze,
    )
    triplets = sample_boundary_triplets(payload["boundary_length"], n_samples=6, max_region_size=2, seed=1)
    result = evaluate_tripartite_dataset(payload, triplets, central_charge=1.0)

    assert len(result["rows"]) == 6
    assert result["rt_coefficient"] == 1.0 / 3.0
    assert np.all(np.isfinite(result["state_i3"]))
    assert np.all(np.isfinite(result["rt_i3"]))
    assert result["endpoint_convention"] == "midpoint"


def test_evaluate_five_region_cyclic_dataset_smoke():
    config = twk.TreeWormholeConfig(
        graph_kind="mera",
        mera_depth=4,
        use_decorated=False,
        couplingtime=1.0,
        presqueeze=0.2,
    )
    graph = twk.build_graph(config)
    payload = twk.boundary_after_measurement_payload(
        graph,
        couplingtime=config.couplingtime,
        presqueeze=config.presqueeze,
    )
    samples = sample_boundary_regions(payload["boundary_length"], n_regions=5, n_samples=4, max_region_size=2, seed=1)
    result = evaluate_five_region_cyclic_dataset(payload, samples, central_charge=1.0)

    assert len(result["rows"]) == 4
    assert result["rt_coefficient"] == 1.0 / 3.0
    assert np.all(np.isfinite(result["state_margin"]))
    assert np.all(np.isfinite(result["rt_margin"]))

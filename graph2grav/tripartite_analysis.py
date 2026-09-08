"""Tripartite-information diagnostics for boundary graph states.

The routines here compare Gaussian-state tripartite information against an RT
proxy.  For a boundary region X, the RT entropy proxy is the minimum
noncrossing pairing of the cut points of X.  Uniform boundary systems use the
periodic CFT logarithm with coefficient c / 3.  Hyperbolic graph systems may
instead use the embedded Poincare-disk boundary-edge cut points, with geodesic
lengths weighted by c / 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from graph2grav import tree_wormhole_knockout as twk
from graph2grav.graphs.hyperbolic import isometry_translate_origin_to_w, isometry_translate_w_to_origin
from graph2grav.poincare import hyperbolic_distance


@dataclass(frozen=True)
class BoundaryTriplet:
    """Three disjoint cyclic boundary regions."""

    a: tuple[int, ...]
    b: tuple[int, ...]
    c: tuple[int, ...]


@dataclass(frozen=True)
class BoundaryRegions:
    """Ordered disjoint cyclic boundary regions."""

    regions: tuple[tuple[int, ...], ...]


def cyclic_interval(start: int, length: int, boundary_length: int) -> tuple[int, ...]:
    """Return a cyclic interval of boundary positions."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if boundary_length <= 0:
        raise ValueError("boundary_length must be positive")
    return tuple((int(start) + offset) % int(boundary_length) for offset in range(int(length)))


def _sample_uniform_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from non-negative integer tuples summing to total."""
    total = int(total)
    parts = int(parts)
    if total < 0:
        raise ValueError("total must be non-negative")
    if parts <= 0:
        raise ValueError("parts must be positive")
    if parts == 1:
        return np.asarray((total,), dtype=int)

    bar_positions = np.sort(rng.choice(total + parts - 1, size=parts - 1, replace=False))
    counts = []
    previous = -1
    for position in bar_positions:
        counts.append(int(position - previous - 1))
        previous = int(position)
    counts.append(int(total + parts - 1 - previous - 1))
    return np.asarray(counts, dtype=int)


def _sample_uniform_three_part_composition(total: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from non-negative integer triples summing to total."""
    return _sample_uniform_composition(total, 3, rng)


def sample_boundary_triplets(
    boundary_length: int,
    *,
    n_samples: int = 600,
    min_region_size: int = 1,
    max_region_size: int | None = None,
    min_gap: int = 1,
    seed: int = 0,
) -> list[BoundaryTriplet]:
    """Sample non-overlapping cyclic triplets with nonempty gaps.

    The construction samples three region lengths and three separating gaps
    around the boundary circle.  After assigning the minimum gap, the remaining
    sites are distributed uniformly over all non-negative three-gap triples.
    """
    boundary_length = int(boundary_length)
    min_region_size = int(min_region_size)
    min_gap = int(min_gap)
    if boundary_length < 3 * (min_region_size + min_gap):
        raise ValueError("boundary_length is too small for the requested triplets")
    if max_region_size is None:
        max_region_size = max(min_region_size, boundary_length // 5)
    max_region_size = int(max_region_size)
    if max_region_size < min_region_size:
        raise ValueError("max_region_size must be at least min_region_size")

    rng = np.random.default_rng(seed)
    triplets: set[BoundaryTriplet] = set()
    max_attempts = max(1000, int(n_samples) * 200)
    attempts = 0
    while len(triplets) < int(n_samples) and attempts < max_attempts:
        attempts += 1
        lengths = rng.integers(min_region_size, max_region_size + 1, size=3)
        base_gaps = np.full(3, min_gap, dtype=int)
        used = int(np.sum(lengths) + np.sum(base_gaps))
        if used > boundary_length:
            continue
        gaps = base_gaps + _sample_uniform_three_part_composition(boundary_length - used, rng)
        start = int(rng.integers(0, boundary_length))

        a = cyclic_interval(start, int(lengths[0]), boundary_length)
        b_start = start + int(lengths[0]) + int(gaps[0])
        b = cyclic_interval(b_start, int(lengths[1]), boundary_length)
        c_start = b_start + int(lengths[1]) + int(gaps[1])
        c = cyclic_interval(c_start, int(lengths[2]), boundary_length)
        if len(set(a) | set(b) | set(c)) != int(np.sum(lengths)):
            continue
        triplets.add(BoundaryTriplet(a=a, b=b, c=c))

    if len(triplets) < int(n_samples):
        raise RuntimeError(f"only sampled {len(triplets)} triplets after {attempts} attempts")
    return sorted(triplets, key=lambda triplet: (triplet.a, triplet.b, triplet.c))


def sample_boundary_regions(
    boundary_length: int,
    *,
    n_regions: int,
    n_samples: int = 600,
    min_region_size: int = 1,
    max_region_size: int | None = None,
    min_gap: int = 1,
    seed: int = 0,
) -> list[BoundaryRegions]:
    """Sample ordered disjoint cyclic intervals with uniformly sampled gap tuples."""
    boundary_length = int(boundary_length)
    n_regions = int(n_regions)
    min_region_size = int(min_region_size)
    min_gap = int(min_gap)
    if n_regions <= 0:
        raise ValueError("n_regions must be positive")
    if boundary_length < n_regions * (min_region_size + min_gap):
        raise ValueError("boundary_length is too small for the requested regions")
    if max_region_size is None:
        max_region_size = max(min_region_size, boundary_length // (2 * n_regions + 1))
    max_region_size = int(max_region_size)
    if max_region_size < min_region_size:
        raise ValueError("max_region_size must be at least min_region_size")

    rng = np.random.default_rng(seed)
    samples: set[BoundaryRegions] = set()
    max_attempts = max(1000, int(n_samples) * 200)
    attempts = 0
    while len(samples) < int(n_samples) and attempts < max_attempts:
        attempts += 1
        lengths = rng.integers(min_region_size, max_region_size + 1, size=n_regions)
        base_gaps = np.full(n_regions, min_gap, dtype=int)
        used = int(np.sum(lengths) + np.sum(base_gaps))
        if used > boundary_length:
            continue
        gaps = base_gaps + _sample_uniform_composition(boundary_length - used, n_regions, rng)

        start = int(rng.integers(0, boundary_length))
        regions = []
        cursor = start
        for length, gap in zip(lengths, gaps):
            regions.append(cyclic_interval(cursor, int(length), boundary_length))
            cursor += int(length) + int(gap)
        if len(set().union(*(set(region) for region in regions))) != int(np.sum(lengths)):
            continue
        samples.add(BoundaryRegions(regions=tuple(regions)))

    if len(samples) < int(n_samples):
        raise RuntimeError(f"only sampled {len(samples)} region sets after {attempts} attempts")
    return sorted(samples, key=lambda sample: sample.regions)


def _region_key(positions: Iterable[int], boundary_length: int) -> tuple[int, ...]:
    return tuple(sorted({int(position) % int(boundary_length) for position in positions}))


def _union(*regions: tuple[int, ...]) -> tuple[int, ...]:
    values = set()
    for region in regions:
        values.update(int(position) for position in region)
    return tuple(sorted(values))


EndpointConvention = Literal["midpoint"]
RTGeometry = Literal["uniform_circle", "hyperbolic_embedding"]


def _boundary_cut_coordinates(
    positions: tuple[int, ...],
    boundary_length: int,
    *,
    endpoint_convention: EndpointConvention = "midpoint",
) -> tuple[float, ...]:
    selected = set(_region_key(positions, boundary_length))
    coordinates = []
    for edge in range(boundary_length):
        left_selected = edge in selected
        right_selected = ((edge + 1) % boundary_length) in selected
        if left_selected == right_selected:
            continue
        if endpoint_convention != "midpoint":
            raise ValueError(f"unknown endpoint convention {endpoint_convention!r}")
        offset = 0.5
        coordinates.append(float(edge) + offset)
    return tuple(coordinates)


def _boundary_nodes(graph) -> list:
    return sorted(
        [node for node, data in graph.nodes(data=True) if data.get("is_boundary")],
        key=lambda node: graph.nodes[node]["position_in_boundary"],
    )


def _coord_as_complex(graph, node) -> complex:
    coord = np.asarray(graph.nodes[node]["coord"], dtype=float)
    return complex(float(coord[0]), float(coord[1]))


def _hyperbolic_geodesic_fraction(z1: complex, z2: complex, fraction: float) -> complex:
    if abs(z1 - z2) < 1e-14:
        return z1
    translated_z2 = isometry_translate_w_to_origin(z2, z1)
    radius = abs(translated_z2)
    if radius < 1e-14:
        point_at_origin = 0j
    else:
        point_radius = np.tanh(float(fraction) * np.arctanh(radius))
        point_at_origin = point_radius * translated_z2 / radius
    return isometry_translate_origin_to_w(point_at_origin, z1)


def _boundary_cut_points_from_embedding(
    graph,
    positions: tuple[int, ...],
    boundary_length: int,
    *,
    endpoint_convention: EndpointConvention = "midpoint",
) -> tuple[complex, ...]:
    boundary = _boundary_nodes(graph)
    if len(boundary) != int(boundary_length):
        raise ValueError("graph boundary size does not match payload boundary_length")
    selected = set(_region_key(positions, boundary_length))
    points = []
    for edge in range(boundary_length):
        left_selected = edge in selected
        right_selected = ((edge + 1) % boundary_length) in selected
        if left_selected == right_selected:
            continue
        if endpoint_convention != "midpoint":
            raise ValueError(f"unknown endpoint convention {endpoint_convention!r}")
        fraction = 0.5
        left = boundary[edge]
        right = boundary[(edge + 1) % boundary_length]
        points.append(_hyperbolic_geodesic_fraction(_coord_as_complex(graph, left), _coord_as_complex(graph, right), fraction))
    return tuple(points)


def _periodic_log_chord(boundary_length: int, left_coordinate: float, right_coordinate: float) -> float:
    separation = abs(float(left_coordinate) - float(right_coordinate))
    separation = min(separation, float(boundary_length) - separation)
    chord = (float(boundary_length) / np.pi) * np.sin(np.pi * separation / float(boundary_length))
    return float(np.log(chord))


def _minimum_noncrossing_pairing_log_chord(cut_coordinates: tuple[float, ...], boundary_length: int) -> float:
    cost, _ = _minimum_noncrossing_pairing(
        cut_coordinates,
        boundary_length,
        lambda left, right: _periodic_log_chord(boundary_length, float(left), float(right)),
    )
    return cost


def _minimum_noncrossing_pairing(
    cut_points: tuple,
    boundary_length: int,
    distance,
) -> tuple[float, tuple[tuple[int, int], ...]]:
    if not cut_points:
        return 0.0, ()
    if len(cut_points) % 2:
        raise ValueError("RT cut points must come in pairs")

    distances = {
        (i, j): float(distance(cut_points[i], cut_points[j]))
        for i in range(len(cut_points))
        for j in range(i + 1, len(cut_points))
    }
    memo: dict[tuple[int, ...], tuple[float, tuple[tuple[int, int], ...]]] = {}

    def solve(indices: tuple[int, ...]) -> tuple[float, tuple[tuple[int, int], ...]]:
        if not indices:
            return 0.0, ()
        if indices in memo:
            return memo[indices]
        first = indices[0]
        best = float("inf")
        best_pairs: tuple[tuple[int, int], ...] = ()
        for partner_offset in range(1, len(indices), 2):
            partner = indices[partner_offset]
            inside = indices[1:partner_offset]
            outside = indices[partner_offset + 1 :]
            low, high = sorted((first, partner))
            inside_cost, inside_pairs = solve(inside)
            outside_cost, outside_pairs = solve(outside)
            cost = distances[(low, high)] + inside_cost + outside_cost
            if cost < best:
                best = cost
                best_pairs = ((first, partner),) + inside_pairs + outside_pairs
        memo[indices] = best, best_pairs
        return memo[indices]

    cost, pairs = solve(tuple(range(len(cut_points))))
    return float(cost), tuple((int(left), int(right)) for left, right in pairs)


def rt_geodesic_pairing_for_region(
    positions: tuple[int, ...],
    *,
    boundary_length: int,
    endpoint_convention: EndpointConvention = "midpoint",
    graph=None,
    rt_geometry: RTGeometry = "uniform_circle",
) -> dict:
    """Return the cut coordinates and minimizing noncrossing RT pairing for a region."""
    key = _region_key(positions, boundary_length)
    if not key or len(key) == boundary_length:
        return {"region": key, "cut_points": (), "cut_coordinates": (), "pair_indices": (), "pairs": (), "weight_sum": 0.0, "log_chord_sum": 0.0}

    if rt_geometry == "uniform_circle":
        cut_points = _boundary_cut_coordinates(
            key,
            boundary_length,
            endpoint_convention=endpoint_convention,
        )
        weight_sum, pair_indices = _minimum_noncrossing_pairing(
            cut_points,
            boundary_length,
            lambda left, right: _periodic_log_chord(boundary_length, float(left), float(right)),
        )
    elif rt_geometry == "hyperbolic_embedding":
        if graph is None:
            raise ValueError("graph is required for hyperbolic_embedding RT geometry")
        cut_points = _boundary_cut_points_from_embedding(
            graph,
            key,
            boundary_length,
            endpoint_convention=endpoint_convention,
        )
        weight_sum, pair_indices = _minimum_noncrossing_pairing(
            cut_points,
            boundary_length,
            lambda left, right: hyperbolic_distance(complex(left), complex(right)),
        )
    else:
        raise ValueError(f"unknown RT geometry {rt_geometry!r}")
    pairs = tuple((cut_points[left], cut_points[right]) for left, right in pair_indices)
    return {
        "region": key,
        "cut_points": cut_points,
        "cut_coordinates": cut_points,
        "pair_indices": pair_indices,
        "pairs": pairs,
        "weight_sum": float(weight_sum),
        "log_chord_sum": float(weight_sum),
    }


def tripartite_from_entropy(entropy_by_region: dict[tuple[int, ...], float], triplet: BoundaryTriplet) -> float:
    """Combine seven region entropies into I3(A:B:C)."""
    a, b, c = triplet.a, triplet.b, triplet.c
    ab = _union(a, b)
    ac = _union(a, c)
    bc = _union(b, c)
    abc = _union(a, b, c)
    return float(
        entropy_by_region[a]
        + entropy_by_region[b]
        + entropy_by_region[c]
        - entropy_by_region[ab]
        - entropy_by_region[ac]
        - entropy_by_region[bc]
        + entropy_by_region[abc]
    )


FIVE_REGION_CYCLIC_POSITIVE_TERMS = ("abc", "bcd", "cde", "dea", "eab")
FIVE_REGION_CYCLIC_NEGATIVE_TERMS = ("bc", "cd", "de", "ea", "ab", "abcde")


def _cyclic_region_union(regions: tuple[tuple[int, ...], ...], start: int, length: int) -> tuple[int, ...]:
    return _union(*(regions[(int(start) + offset) % len(regions)] for offset in range(int(length))))


def five_region_cyclic_terms(sample: BoundaryRegions) -> dict[str, tuple[int, ...]]:
    """Return the entropy regions entering the five-party cyclic inequality."""
    if len(sample.regions) != 5:
        raise ValueError("five-region cyclic inequality requires exactly five regions")
    names = ("a", "b", "c", "d", "e")
    terms = {name: sample.regions[index] for index, name in enumerate(names)}
    terms.update(
        {
            "ab": _cyclic_region_union(sample.regions, 0, 2),
            "bc": _cyclic_region_union(sample.regions, 1, 2),
            "cd": _cyclic_region_union(sample.regions, 2, 2),
            "de": _cyclic_region_union(sample.regions, 3, 2),
            "ea": _cyclic_region_union(sample.regions, 4, 2),
            "abc": _cyclic_region_union(sample.regions, 0, 3),
            "bcd": _cyclic_region_union(sample.regions, 1, 3),
            "cde": _cyclic_region_union(sample.regions, 2, 3),
            "dea": _cyclic_region_union(sample.regions, 3, 3),
            "eab": _cyclic_region_union(sample.regions, 4, 3),
            "abcde": _union(*sample.regions),
        }
    )
    return terms


def five_region_cyclic_margin(entropy_by_term: dict[str, float]) -> float:
    """Return LHS - RHS for the five-party cyclic holographic entropy inequality."""
    return float(
        sum(float(entropy_by_term[term]) for term in FIVE_REGION_CYCLIC_POSITIVE_TERMS)
        - sum(float(entropy_by_term[term]) for term in FIVE_REGION_CYCLIC_NEGATIVE_TERMS)
    )


def _state_entropy(
    payload: dict,
    positions: tuple[int, ...],
    *,
    boundary_length: int,
    cache: dict[tuple[int, ...], float],
    renyi2: bool = False,
) -> float:
    key = _region_key(positions, boundary_length)
    if key not in cache:
        if not key or len(key) == boundary_length:
            cache[key] = 0.0
        else:
            cache[key] = twk.entropy_for_boundary_positions_from_payload(payload, key, renyi2=renyi2)
    return cache[key]


def _rt_entropy_cft(
    positions: tuple[int, ...],
    *,
    boundary_length: int,
    central_charge: float,
    cache: dict[tuple[int, ...], float],
    endpoint_convention: EndpointConvention = "midpoint",
    graph=None,
    rt_geometry: RTGeometry = "uniform_circle",
) -> float:
    key = _region_key(positions, boundary_length)
    if key not in cache:
        if not key or len(key) == boundary_length:
            cache[key] = 0.0
        else:
            pairing = rt_geodesic_pairing_for_region(
                key,
                boundary_length=boundary_length,
                endpoint_convention=endpoint_convention,
                graph=graph,
                rt_geometry=rt_geometry,
            )
            coefficient = float(central_charge) / (6.0 if rt_geometry == "hyperbolic_embedding" else 3.0)
            cache[key] = coefficient * float(pairing["weight_sum"])
    return cache[key]


def evaluate_five_region_cyclic_dataset(
    payload: dict,
    samples: list[BoundaryRegions],
    *,
    central_charge: float,
    graph=None,
    rt_geometry: RTGeometry = "uniform_circle",
    renyi2: bool = False,
) -> dict:
    """Evaluate the five-party cyclic entropy inequality on sampled boundary regions."""
    boundary_length = int(payload["boundary_length"])
    state_entropy_cache: dict[tuple[int, ...], float] = {}
    rt_entropy_cache: dict[tuple[int, ...], float] = {}
    rows = []

    for sample in samples:
        terms = five_region_cyclic_terms(sample)
        state_entropies = {
            term: _state_entropy(
                payload,
                positions,
                boundary_length=boundary_length,
                cache=state_entropy_cache,
                renyi2=renyi2,
            )
            for term, positions in terms.items()
        }
        rt_entropies = {
            term: _rt_entropy_cft(
                positions,
                boundary_length=boundary_length,
                central_charge=central_charge,
                cache=rt_entropy_cache,
                endpoint_convention="midpoint",
                graph=graph,
                rt_geometry=rt_geometry,
            )
            for term, positions in terms.items()
        }
        state_margin = five_region_cyclic_margin(state_entropies)
        rt_margin = five_region_cyclic_margin(rt_entropies)
        rows.append(
            {
                "sample": sample,
                "state_margin": float(state_margin),
                "rt_margin": float(rt_margin),
                "state_violation": float(max(-state_margin, 0.0)),
                "rt_violation": float(max(-rt_margin, 0.0)),
                "state_entropies": state_entropies,
                "rt_entropies": rt_entropies,
            }
        )

    state_margin = np.asarray([row["state_margin"] for row in rows], dtype=float)
    rt_margin = np.asarray([row["rt_margin"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "central_charge": float(central_charge),
        "rt_geometry": rt_geometry,
        "rt_coefficient": float(central_charge) / (6.0 if rt_geometry == "hyperbolic_embedding" else 3.0),
        "state_margin": state_margin,
        "rt_margin": rt_margin,
        "delta_margin": state_margin - rt_margin,
        "state_violation": np.maximum(-state_margin, 0.0),
        "rt_violation": np.maximum(-rt_margin, 0.0),
        "max_state_violation": float(np.nanmax(np.maximum(-state_margin, 0.0))),
        "state_violation_fraction": float(np.mean(state_margin < 0.0)),
        "max_rt_violation": float(np.nanmax(np.maximum(-rt_margin, 0.0))),
        "rt_violation_fraction": float(np.mean(rt_margin < 0.0)),
        "positive_terms": FIVE_REGION_CYCLIC_POSITIVE_TERMS,
        "negative_terms": FIVE_REGION_CYCLIC_NEGATIVE_TERMS,
        "state_entropy_cache": state_entropy_cache,
        "rt_entropy_cache": rt_entropy_cache,
    }


def evaluate_tripartite_dataset(
    payload: dict,
    triplets: list[BoundaryTriplet],
    *,
    central_charge: float,
    graph=None,
    rt_geometry: RTGeometry = "uniform_circle",
    renyi2: bool = False,
) -> dict:
    """Evaluate state I3, CFT/RT I3, and MMI diagnostics for triplets."""
    boundary_length = int(payload["boundary_length"])
    state_entropy_cache: dict[tuple[int, ...], float] = {}
    rt_entropy_cache: dict[tuple[int, ...], float] = {}
    rows = []

    for triplet in triplets:
        regions = {
            "a": triplet.a,
            "b": triplet.b,
            "c": triplet.c,
            "ab": _union(triplet.a, triplet.b),
            "ac": _union(triplet.a, triplet.c),
            "bc": _union(triplet.b, triplet.c),
            "abc": _union(triplet.a, triplet.b, triplet.c),
        }
        state_entropies = {
            region: _state_entropy(
                payload,
                positions,
                boundary_length=boundary_length,
                cache=state_entropy_cache,
                renyi2=renyi2,
            )
            for region, positions in regions.items()
        }
        rt_entropies = {
            region: _rt_entropy_cft(
                positions,
                boundary_length=boundary_length,
                central_charge=central_charge,
                cache=rt_entropy_cache,
                endpoint_convention="midpoint",
                graph=graph,
                rt_geometry=rt_geometry,
            )
            for region, positions in regions.items()
        }
        state_i3 = (
            state_entropies["a"]
            + state_entropies["b"]
            + state_entropies["c"]
            - state_entropies["ab"]
            - state_entropies["ac"]
            - state_entropies["bc"]
            + state_entropies["abc"]
        )
        rt_i3 = (
            rt_entropies["a"]
            + rt_entropies["b"]
            + rt_entropies["c"]
            - rt_entropies["ab"]
            - rt_entropies["ac"]
            - rt_entropies["bc"]
            + rt_entropies["abc"]
        )
        rows.append(
            {
                "triplet": triplet,
                "state_i3": float(state_i3),
                "rt_i3": float(rt_i3),
                "state_entropies": state_entropies,
                "rt_entropies": rt_entropies,
            }
        )

    for row in rows:
        row["delta_i3"] = float(row["state_i3"] - row["rt_i3"])
        row["state_i3_positive"] = float(max(row["state_i3"], 0.0))

    state_i3 = np.asarray([row["state_i3"] for row in rows], dtype=float)
    rt_i3 = np.asarray([row["rt_i3"] for row in rows], dtype=float)
    delta_i3 = np.asarray([row["delta_i3"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "central_charge": float(central_charge),
        "rt_geometry": rt_geometry,
        "rt_coefficient": float(central_charge) / (6.0 if rt_geometry == "hyperbolic_embedding" else 3.0),
        "epsilon_mmi": float(np.nanmax(np.maximum(state_i3, 0.0))),
        "positive_fraction": float(np.mean(state_i3 > 0.0)),
        "state_i3": state_i3,
        "rt_i3": rt_i3,
        "delta_i3": delta_i3,
        "endpoint_convention": "midpoint",
        "state_entropy_cache": state_entropy_cache,
        "rt_entropy_cache": rt_entropy_cache,
    }

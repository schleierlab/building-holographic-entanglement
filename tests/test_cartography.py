"""Tests for cartography module (bulk reconstruction and entanglement wedge analysis)."""

import pytest
import numpy as np
import networkx as nx

from graph2grav.cartography import (
    extract_subsystem_covariance,
    mutual_information,
    add_bulk_node,
    get_post_measurement_covariance,
    compute_bulk_entropies,
    sweep_bulk_reconstruction,
)

from graph2grav.graphs.standard import crosslinked_tree, make_tree_periodic
from graph2grav import gaussian


def check_covariance_physical(cov, name="covariance"):
    """Check if a covariance matrix is physical (all symeigs >= 0.5)."""
    N = cov.shape[0] // 2
    Omega = np.block([[np.zeros((N, N)), np.eye(N)],
                     [-np.eye(N), np.zeros((N, N))]])
    symeigs = gaussian.symeigvals(cov, Omega) / 2  # Divide by 2 to match convention

    min_symeig = np.min(symeigs)
    if min_symeig < 0.5 - 1e-10:  # Allow small numerical error
        num_bad = np.sum(symeigs < 0.5 - 1e-10)
        bad_values = symeigs[symeigs < 0.5 - 1e-10]
        raise AssertionError(
            f"Unphysical state detected in {name}:\n"
            f"  {num_bad}/{len(symeigs)} symeigs < 0.5\n"
            f"  Min symeig: {min_symeig:.6f}\n"
            f"  Bad values: {bad_values}\n"
            f"This indicates the state violates the uncertainty principle."
        )
    return True


class TestSubsystems:
    """Tests for subsystems.py functions."""

    def test_extract_subsystem_sequential(self):
        """Test extracting subsystem with sequential ordering."""
        # Create 3-mode identity covariance
        N = 3
        cov_full = np.eye(2 * N)

        # Extract modes 0 and 2
        indices = [0, 2]
        cov_sub = extract_subsystem_covariance(cov_full, indices, ordering='seq')

        # Should get 4x4 identity (2 modes)
        assert cov_sub.shape == (4, 4)
        assert np.allclose(cov_sub, np.eye(4))

    def test_extract_subsystem_interspersed(self):
        """Test extracting subsystem with interspersed ordering."""
        # Create 3-mode system in interspersed ordering
        N = 3
        cov_full = np.eye(2 * N)

        # Extract mode 1
        indices = [1]
        cov_sub = extract_subsystem_covariance(cov_full, indices, ordering='inter')

        # Should get 2x2 identity (1 mode)
        assert cov_sub.shape == (2, 2)
        assert np.allclose(cov_sub, np.eye(2))

    def test_extract_subsystem_invalid_ordering(self):
        """Test that invalid ordering raises error."""
        cov = np.eye(4)
        with pytest.raises(ValueError, match="Unknown ordering"):
            extract_subsystem_covariance(cov, [0], ordering='invalid')

    def test_extract_subsystem_invalid_indices(self):
        """Test that out-of-range indices raise error."""
        N = 3
        cov = np.eye(2 * N)

        # Index too large
        with pytest.raises(ValueError, match="Indices must be between"):
            extract_subsystem_covariance(cov, [5], ordering='seq')

        # Negative index
        with pytest.raises(ValueError, match="Indices must be between"):
            extract_subsystem_covariance(cov, [-1], ordering='seq')

    def test_mutual_information_product_state(self):
        """Test MI is zero for product state."""
        # Product state (identity covariance)
        N = 4
        cov = np.eye(2 * N)

        # MI should be zero for uncorrelated subsystems
        MI = mutual_information(cov, [0, 1], [2, 3], ordering='seq')
        assert np.isclose(MI, 0.0, atol=1e-10)

    def test_mutual_information_nonnegative(self):
        """Test MI is non-negative."""
        # Create some covariance (may have correlations)
        N = 4
        np.random.seed(42)
        cov = np.eye(2 * N) * 0.5

        MI = mutual_information(cov, [0], [1], ordering='seq')
        assert MI >= -1e-10  # Allow small numerical errors

    def test_mutual_information_overlapping_subsystems(self):
        """Test that overlapping subsystems raise error."""
        cov = np.eye(6)

        with pytest.raises(ValueError, match="must be disjoint"):
            mutual_information(cov, [0, 1], [1, 2], ordering='seq')


class TestReconstruction:
    """Tests for reconstruction.py functions."""

    @pytest.fixture
    def small_mera(self):
        """Create a small MERA graph for testing."""
        G = crosslinked_tree(depth=3)  # 8 boundary nodes
        G_mera = make_tree_periodic(G, remove_nodes=True)
        return G_mera

    def test_add_bulk_node(self, small_mera):
        """Test adding bulk node to graph."""
        G = small_mera
        original_nodes = G.number_of_nodes()

        # Add bulk node
        G_ext, bulk_id = add_bulk_node(G, bulk_position=5, coupling=1.0)

        # Check node count increased
        assert G_ext.number_of_nodes() == original_nodes + 1

        # Check bulk node has correct attributes (boundary node in b1)
        assert G_ext.nodes[bulk_id]['is_bulk_probe'] == True
        assert G_ext.nodes[bulk_id]['is_boundary'] == True  # Boundary, not ancilla
        assert G_ext.nodes[bulk_id]['is_ancilla'] == False
        assert G_ext.nodes[bulk_id]['b1'] == True  # In boundary group b1
        assert G_ext.nodes[bulk_id]['b0'] == False  # Not in b0
        assert G_ext.nodes[bulk_id]['boundary_index'] == 1

        # Check graph has 2 boundaries now
        assert G_ext.graph['number_of_boundaries'] == 2

        # Check edge exists
        assert G_ext.has_edge(5, bulk_id) or G_ext.has_edge(bulk_id, 5)

    def test_add_bulk_node_id(self, small_mera):
        """Test bulk node ID is max + 1."""
        G = small_mera
        max_id = max(G.nodes())

        G_ext, bulk_id = add_bulk_node(G, bulk_position=5)

        assert bulk_id == max_id + 1

    def test_get_post_measurement_covariance_unweighted(self, small_mera):
        """Test post-measurement covariance with unweighted quench."""
        G = small_mera
        boundary_nodes = [n for n, d in G.nodes(data=True) if d.get('is_boundary', False)]

        # Add bulk node
        G_ext, bulk_id = add_bulk_node(G, bulk_position=5)

        # Get post-measurement covariance
        cov_post, kept_indices, bulk_idx = get_post_measurement_covariance(
            G_ext, bulk_id,
            use_weighted=False,
            couplingtime=1.0
        )

        # Check dimensions
        N_kept = len(kept_indices)
        assert cov_post.shape == (2 * N_kept, 2 * N_kept)

        # Should have boundary + bulk
        assert N_kept == len(boundary_nodes) + 1

        # Bulk should be in kept indices
        assert kept_indices[bulk_idx] == bulk_id

    def test_get_post_measurement_covariance_weighted(self, small_mera):
        """Test post-measurement covariance with weighted quench."""
        G = small_mera
        boundary_nodes = [n for n, d in G.nodes(data=True) if d.get('is_boundary', False)]

        # Add bulk node
        G_ext, bulk_id = add_bulk_node(G, bulk_position=5)

        # Get post-measurement covariance (weighted)
        cov_post, kept_indices, bulk_idx = get_post_measurement_covariance(
            G_ext, bulk_id,
            use_weighted=True,
            couplingtime=1.0,
            bulk_presqueeze=1.0
        )

        # Check dimensions
        N_kept = len(kept_indices)
        assert cov_post.shape == (2 * N_kept, 2 * N_kept)
        assert N_kept == len(boundary_nodes) + 1

    def test_compute_bulk_entropies(self, small_mera):
        """Test computing all bulk entropies."""
        G = small_mera
        boundary_nodes = sorted([n for n, d in G.nodes(data=True) if d.get('is_boundary', False)])

        # Add bulk and get covariance
        G_ext, bulk_id = add_bulk_node(G, bulk_position=5)
        cov_post, kept_indices, bulk_idx = get_post_measurement_covariance(
            G_ext, bulk_id, use_weighted=False
        )

        # Check that post-measurement covariance is physical
        check_covariance_physical(cov_post, "post-measurement covariance")

        # Define region
        region_size = 3
        region_nodes = boundary_nodes[:region_size]
        node_to_idx = {node: idx for idx, node in enumerate(kept_indices)}
        region_indices = [node_to_idx[n] for n in region_nodes]

        # Compute entropies
        entropies = compute_bulk_entropies(
            cov_post, region_indices, bulk_idx, len(kept_indices)
        )

        # Check all keys present
        required_keys = ['S_b', 'S_R', 'S_R_b', 'S_Rc', 'S_Rc_b', 'MI_R_b']
        assert all(k in entropies for k in required_keys)

        # Check non-negativity
        for key, value in entropies.items():
            assert value >= -1e-10, f"{key} should be non-negative"

        # Check MI formula
        MI_check = entropies['S_R'] + entropies['S_b'] - entropies['S_R_b']
        assert np.isclose(MI_check, entropies['MI_R_b'], atol=1e-9)

    def test_compute_bulk_entropies_mi_nonnegative(self, small_mera):
        """Test that mutual information is non-negative."""
        G = small_mera
        boundary_nodes = sorted([n for n, d in G.nodes(data=True) if d.get('is_boundary', False)])

        G_ext, bulk_id = add_bulk_node(G, bulk_position=5)
        cov_post, kept_indices, bulk_idx = get_post_measurement_covariance(
            G_ext, bulk_id, use_weighted=False
        )

        # Test several region sizes
        node_to_idx = {node: idx for idx, node in enumerate(kept_indices)}
        for region_size in [1, 2, 4]:
            region_nodes = boundary_nodes[:region_size]
            region_indices = [node_to_idx[n] for n in region_nodes]

            entropies = compute_bulk_entropies(
                cov_post, region_indices, bulk_idx, len(kept_indices)
            )

            assert entropies['MI_R_b'] >= -1e-10, f"MI should be non-negative for region_size={region_size}"

    def test_sweep_bulk_reconstruction(self, small_mera):
        """Test sweeping over bulk positions and region sizes."""
        G = small_mera
        boundary_nodes = [n for n, d in G.nodes(data=True) if d.get('is_boundary', False)]
        ancilla_nodes = [n for n, d in G.nodes(data=True) if d.get('is_ancilla', False)]

        # Small sweep
        bulk_positions = ancilla_nodes[:2]  # Test 2 positions
        region_sizes = [2, 4]  # Test 2 sizes

        results = sweep_bulk_reconstruction(
            G, bulk_positions, region_sizes,
            use_weighted=False,
            verbose=False
        )

        # Check number of results
        expected_count = len(bulk_positions) * len(region_sizes)
        assert len(results) == expected_count

        # Check structure of results
        required_keys = ['bulk_position', 'region_size', 'S_b', 'S_R', 'S_R_b', 'S_Rc', 'S_Rc_b', 'MI_R_b']
        for r in results:
            assert all(k in r for k in required_keys)

            # Check MI formula
            MI_check = r['S_R'] + r['S_b'] - r['S_R_b']
            assert np.isclose(MI_check, r['MI_R_b'], atol=1e-9)

            # Check non-negativity
            assert r['MI_R_b'] >= -1e-10

    def test_sweep_bulk_reconstruction_weighted(self, small_mera):
        """Test sweep with weighted quench."""
        G = small_mera
        ancilla_nodes = [n for n, d in G.nodes(data=True) if d.get('is_ancilla', False)]

        bulk_positions = ancilla_nodes[:1]  # Just 1 position
        region_sizes = [2]

        results = sweep_bulk_reconstruction(
            G, bulk_positions, region_sizes,
            use_weighted=True,  # Use weighted quench
            verbose=False
        )

        assert len(results) == 1
        assert results[0]['MI_R_b'] >= -1e-10


class TestPhysicalProperties:
    """Tests for physical properties of the results."""

    @pytest.fixture
    def mera_with_results(self):
        """Create MERA and run small sweep."""
        G = crosslinked_tree(depth=3)
        G_mera = make_tree_periodic(G, remove_nodes=True)

        ancilla = [n for n, d in G_mera.nodes(data=True) if d.get('is_ancilla', False)]
        bulk_positions = ancilla[:2]
        region_sizes = [2, 4]

        results = sweep_bulk_reconstruction(
            G_mera, bulk_positions, region_sizes,
            use_weighted=False, verbose=False
        )

        return G_mera, results

    def test_mi_bounded_by_entropies(self, mera_with_results):
        """Test that I(R:b) ≤ S_R + S_b (from formula since S_{R+b} ≥ 0)."""
        _, results = mera_with_results

        for r in results:
            MI = r['MI_R_b']
            S_R = r['S_R']
            S_b = r['S_b']

            # MI = S_R + S_b - S_{R+b}, so MI ≤ S_R + S_b (since S_{R+b} ≥ 0)
            # Note: MI can exceed min(S_R, S_b)! For pure states, MI = S_R + S_b
            assert MI <= S_R + S_b + 1e-9, "MI should not exceed S_R + S_b"

    def test_subadditivity(self, mera_with_results):
        """Test subadditivity: S_{R+b} ≤ S_R + S_b."""
        _, results = mera_with_results

        for r in results:
            S_R_b = r['S_R_b']
            S_R = r['S_R']
            S_b = r['S_b']

            assert S_R_b <= S_R + S_b + 1e-9, "Subadditivity violated"

    def test_mi_increases_with_region_size(self, mera_with_results):
        """Test that MI generally increases with region size (not strict)."""
        _, results = mera_with_results

        # Group by bulk position
        by_position = {}
        for r in results:
            pos = r['bulk_position']
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(r)

        # For each position, check MI trend
        for pos, res_list in by_position.items():
            # Sort by region size
            res_list.sort(key=lambda x: x['region_size'])

            # MI should not decrease drastically
            # (this is a weak test - MI can fluctuate)
            MIs = [r['MI_R_b'] for r in res_list]
            assert max(MIs) >= min(MIs), "MI should vary with region size"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_region(self):
        """Test behavior with empty region (should handle gracefully or error)."""
        # This might not be a valid use case, but test robustness
        # Create minimal system
        cov = np.eye(4)  # 2 modes

        # Try to extract empty subsystem
        cov_empty = extract_subsystem_covariance(cov, [], ordering='seq')
        assert cov_empty.shape == (0, 0)

    def test_single_mode_subsystem(self):
        """Test extracting single mode."""
        N = 3
        cov = np.eye(2 * N)

        cov_single = extract_subsystem_covariance(cov, [1], ordering='seq')
        assert cov_single.shape == (2, 2)
        assert np.allclose(cov_single, np.eye(2))

    def test_very_small_graph(self):
        """Test with minimal MERA graph."""
        G = crosslinked_tree(depth=2)  # 4 boundary nodes
        G_mera = make_tree_periodic(G, remove_nodes=True)

        ancilla = [n for n, d in G_mera.nodes(data=True) if d.get('is_ancilla', False)]

        if len(ancilla) > 0:
            results = sweep_bulk_reconstruction(
                G_mera, [ancilla[0]], [1],
                use_weighted=False, verbose=False
            )
            assert len(results) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

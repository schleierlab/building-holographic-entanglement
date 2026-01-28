"""Cartography module for bulk reconstruction and entanglement wedge analysis.

This module provides tools for:
- Adding bulk nodes to boundary systems post-measurement
- Computing mutual information between bulk and boundary regions
- Analyzing entanglement wedge structure
- Sweeping bulk configurations for entanglement wedge mapping
"""

from graph2grav.cartography.subsystems import (
    extract_subsystem_covariance,
    mutual_information,
)

from graph2grav.cartography.reconstruction import (
    add_bulk_node,
    get_post_measurement_covariance,
    compute_bulk_entropies,
    sweep_bulk_reconstruction,
)

from graph2grav.cartography.plotting import (
    plot_mutual_info_heatmap,
    plot_mutual_info_vs_region_size,
    plot_entropy_components,
    plot_wedge_boundary,
    plot_bulk_entropy_comparison,
)

__all__ = [
    # Subsystems
    "extract_subsystem_covariance",
    "mutual_information",
    # Reconstruction
    "add_bulk_node",
    "get_post_measurement_covariance",
    "compute_bulk_entropies",
    "sweep_bulk_reconstruction",
    # Plotting
    "plot_mutual_info_heatmap",
    "plot_mutual_info_vs_region_size",
    "plot_entropy_components",
    "plot_wedge_boundary",
    "plot_bulk_entropy_comparison",
]

"""Helper functions for supplementary scaling figures."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import graph2grav.graphs as graphs
from graph2grav.graphs.standard import make_tree_periodic
from graph2grav.analysis import Analysis
from graph2grav import analytic_entropies

import graph2grav.plotting as plotting


def plot_entanglement_entropy_scaling_with_depth(ax: plt.Axes, depths: list[int], squeezing=1/10, coupling_time=1, config=None, is_decorated: bool = False):  # pyright: ignore
    """Plot entanglement entropy S(ℓ) vs region size for multiple graph depths.

    Args:
        ax: Matplotlib axes to plot on
        depths: List of tree depths to compare (e.g., [4, 5, 6])
        squeezing: Squeezing parameter
        coupling_time: Coupling time parameter
        config: Configuration dictionary
        is_decorated: If True, use decorated (weighted) graphs
    """
    assert config is not None, "You must provide a config."

    # Use fixed squeezing and coupling time from config

    colors = [plotting.color_for_depth(depth) for depth in depths]

    for depth, color in zip(depths, colors):
        if is_decorated:
            graph = graphs.subdivided_tree_decoration_1(depth, validate=False, draw=False)
        else:
            ctree = graphs.crosslinked_tree(depth)
            graph = make_tree_periodic(ctree, remove_nodes=False)

        analysis = Analysis(
            graph,
            couplingtime=coupling_time,
            global_presqueeze=squeezing,
            is_decorated=is_decorated,
        )

        # Plot data points (region_sizes[1:-1])
        ax.plot(
            analysis.region_sizes,
            analysis.boundary_entropies,
            'o',
            color=color,
            label=f"$L = {2**depth}$",
            markersize=4,
            alpha=0.8
        )

        # Fit central charge and plot CFT theory line
        analysis.fit_central_charge()
        fit_xs = np.linspace(1, analysis.boundary_length - 1, 100)
        fit_ys = analytic_entropies.periodic_CFT_entropy(
            fit_xs, analysis.boundary_length,
            central_charge=analysis.central_charge,
            offset=analysis.cft_entropy_offset
        )
        ax.plot(fit_xs, fit_ys, '--', color='black', linewidth=1, alpha=0.8)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S$")


def plot_entanglement_entropy_scaling_with_squeezing(ax: plt.Axes, squeezing_values: list[float], config=None, is_decorated: bool = False):  # pyright: ignore
    """Plot entanglement entropy S(ℓ) vs region size for multiple squeezing values.

    Args:
        ax: Matplotlib axes to plot on
        squeezing_values: List of squeezing values to compare
        config: Configuration dictionary
        is_decorated: If True, use decorated (weighted) graphs
    """
    assert config is not None, "You must provide a config."

    # Use fixed depth and coupling time from config
    depth = 5 # config["figure-1"]["graph"]["depth"]
    coupling_time = config["figure-1"]["graph"]["coupling_time"]

    if is_decorated:
        graph = graphs.subdivided_tree_decoration_1(depth, validate=False, draw=False)
    else:
        ctree = graphs.crosslinked_tree(depth)
        graph = make_tree_periodic(ctree, remove_nodes=False)

    # Use a different color progression for squeezing (plasma colormap)
    cmap = plt.cm.plasma  # pyright: ignore
    n = len(squeezing_values)
    colors = [cmap(0.6 * i / (n - 1)) if n > 1 else cmap(0.5) for i in range(n)]

    for squeezing, color in zip(squeezing_values, colors):
        analysis = Analysis(
            graph,
            couplingtime=coupling_time,
            global_presqueeze=squeezing,
            is_decorated=is_decorated,
        )

        ax.plot(
            analysis.region_sizes,
            analysis.boundary_entropies,
            'o',
            color=color,
            label=rf"$\mu$ = {squeezing:.2g}",
            markersize=4,
            alpha=0.8
        )

        # Fit central charge and plot CFT theory line
        analysis.fit_central_charge()
        fit_xs = np.linspace(1, analysis.boundary_length - 1, 100)
        fit_ys = analytic_entropies.periodic_CFT_entropy(
            fit_xs, analysis.boundary_length,
            central_charge=analysis.central_charge,
            offset=analysis.cft_entropy_offset
        )
        ax.plot(fit_xs, fit_ys, '--', color='black', linewidth=1, alpha=0.8)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S$")
    # ax.legend(fontsize=8, ncol=2)


def plot_central_charge_versus_squeezing(ax: plt.Axes, squeezing_values, depths: list[int], config=None, is_decorated: bool = False):  # pyright: ignore
    """Plot fitted central charge c vs squeezing parameter μ for multiple system sizes.

    Args:
        ax: Matplotlib axes to plot on
        squeezing_values: Array or list of squeezing values to sweep
        depths: List of tree depths to compare (should match LHS plot)
        config: Configuration dictionary
        is_decorated: If True, use decorated graphs; if False, use undecorated
    """
    assert config is not None, "You must provide a config."

    coupling_time = config["figure-1"]["graph"]["coupling_time"]
    colors = [plotting.color_for_depth(depth) for depth in depths]

    for depth, color in zip(depths, colors):
        if is_decorated:
            graph = graphs.subdivided_tree_decoration_1(depth, validate=False, draw=False)
        else:
            ctree = graphs.crosslinked_tree(depth)
            graph = make_tree_periodic(ctree, remove_nodes=False)

        central_charges = []
        for squeezing in squeezing_values:
            analysis = Analysis(
                graph,
                couplingtime=coupling_time,
                global_presqueeze=squeezing,
                is_decorated=is_decorated,
            )
            analysis.fit_central_charge()
            central_charges.append(analysis.central_charge)

        ax.plot(
            squeezing_values,
            central_charges,
            '-',
            color=color,
            linewidth=1.5,
            label=f'$L = {2**depth}$'
        )

    # Add reference line at c=1 (free boson CFT)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='c = 1')

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$c$")
    ax.set_xscale('log')
    # ax.legend(fontsize=8)

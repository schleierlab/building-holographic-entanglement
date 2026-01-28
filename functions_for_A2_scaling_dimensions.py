"""Functions for A2_scaling_dimensions.ipynb - Scaling dimension dependence on squeezing."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import graph2grav.graphs as graphs
from graph2grav.graphs.standard import make_tree_periodic
from graph2grav.analysis import Analysis
from graph2grav.cartography.subsystems import mutual_information

import graph2grav.plotting as plotting


def fit_mutual_info_scaling_dimension(
    analysis: Analysis,
    fraction_to_exclude: float = 1/3,
) -> float:
    """Fit the scaling dimension from mutual information decay.

    MI between sites 0 and d decays as 1/d^(2*Delta_MI).

    Args:
        analysis: Analysis object with computed boundary covariance
        fraction_to_exclude: Fraction of data to exclude at edges

    Returns:
        Fitted scaling dimension for mutual information
    """
    boundary_cov = analysis.boundary_after_bulk_measurement
    L = analysis.boundary_length

    # Compute MI vs distance
    distances = []
    mi_values = []

    for d in range(1, L // 2 + 1):
        mi = mutual_information(boundary_cov, [0], [d], ordering='seq')
        distances.append(d)
        mi_values.append(mi)

    distances = np.array(distances)
    mi_values = np.array(mi_values)

    # Convert to chord distance
    chord_dists = np.sin(np.pi * distances / L)

    # Fit in log-log space, excluding edges
    length = len(chord_dists)
    start = max(1, int(length * fraction_to_exclude))
    end = length - start

    log_x = np.log(chord_dists[start:end])
    log_mi = np.log(mi_values[start:end])

    # MI ~ 1/d^(2*Delta), so log(MI) = const - 2*Delta*log(d)
    slope, _ = np.polyfit(log_x, log_mi, 1)
    scaling_dimension = -slope / 2

    return scaling_dimension


def compute_scaling_dimensions_vs_squeezing(
    squeezing_values: np.ndarray,
    depth: int = 5,
    coupling_time: float = 1.0,
    fraction_to_exclude_x: float = 1/3,
    fraction_to_exclude_p: float = 1/3,
    fraction_to_exclude_mi: float = 1/3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute position, momentum, and MI scaling dimensions as a function of squeezing.

    For decorated graphs, the scaling dimensions Δ_x and Δ_p depend on the
    initial squeezing parameter μ:
    - Small μ (strong squeezing): Δ_x → 1, Δ_p → 0
    - Large μ (weak squeezing): Δ_x → 1/2, Δ_p → 1/2

    Args:
        squeezing_values: Array of squeezing values to sweep
        depth: Tree depth for the decorated graph
        coupling_time: Quench coupling time
        fraction_to_exclude_x: Fraction of data to exclude at edges when fitting x
        fraction_to_exclude_p: Fraction of data to exclude at edges when fitting p
        fraction_to_exclude_mi: Fraction of data to exclude at edges when fitting MI

    Returns:
        Tuple of (squeezing_values, delta_x, delta_p, delta_mi) where:
        - delta_x: Position scaling dimensions
        - delta_p: Momentum scaling dimensions
        - delta_mi: Mutual information scaling dimensions
    """
    delta_x = []
    delta_p = []
    delta_mi = []

    for mu in squeezing_values:
        graph = graphs.subdivided_tree_decoration_1(depth, validate=False, draw=False)

        analysis = Analysis(
            graph,
            couplingtime=coupling_time,
            global_presqueeze=mu,
            is_decorated=True,
        )

        # Fit scaling dimensions with independent exclusion fractions
        analysis.fit_position_scaling_dimension(fraction_to_exclude=fraction_to_exclude_x)
        analysis.fit_momentum_scaling_dimension(fraction_to_exclude=fraction_to_exclude_p)

        delta_x.append(analysis.position_scaling_dimension)
        delta_p.append(analysis.momentum_scaling_dimension)

        # Fit MI scaling dimension
        mi_delta = fit_mutual_info_scaling_dimension(analysis, fraction_to_exclude_mi)
        delta_mi.append(mi_delta)

    return squeezing_values, np.array(delta_x), np.array(delta_p), np.array(delta_mi)


def plot_scaling_dimensions_vs_squeezing(
    ax: plt.Axes,
    squeezing_values: np.ndarray | None = None,
    depth: int = 5,
    coupling_time: float = 1.0,
    fraction_to_exclude_x: float = 1/3,
    fraction_to_exclude_p: float = 1/3,
    fraction_to_exclude_mi: float = 1/3,
    show_theory_lines: bool = True,
) -> None:
    """Plot position, momentum, and MI scaling dimensions vs squeezing parameter.

    Args:
        ax: Matplotlib axes to plot on
        squeezing_values: Array of squeezing values (if None, uses default range)
        depth: Tree depth for the decorated graph
        coupling_time: Quench coupling time
        fraction_to_exclude_x: Fraction of data to exclude when fitting x
        fraction_to_exclude_p: Fraction of data to exclude when fitting p
        fraction_to_exclude_mi: Fraction of data to exclude when fitting MI
        show_theory_lines: If True, show reference lines at Δ = 0, 1/2, 1
    """
    if squeezing_values is None:
        squeezing_values = np.geomspace(0.01, 5, 30)

    _, delta_x, delta_p, delta_mi = compute_scaling_dimensions_vs_squeezing(
        squeezing_values, depth, coupling_time,
        fraction_to_exclude_x, fraction_to_exclude_p, fraction_to_exclude_mi
    )

    colors = plotting.default_colors

    ax.semilogx(
        squeezing_values, delta_x,
        'o', color=colors[0], markersize=6, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\Delta_x$'
    )

    ax.semilogx(
        squeezing_values, delta_p,
        's', color=colors[1], markersize=6, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\Delta_p$'
    )

    # Plot 1/2 of MI scaling dimension since we plot sqrt(MI) ~ 1/d^(Delta_I)
    ax.semilogx(
        squeezing_values, delta_mi / 2,
        '^', color=colors[2], markersize=6, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\Delta_I / 2$'
    )

    if show_theory_lines:
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5,
                   label=r'$\Delta = 1$')

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\Delta$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_scaling_dimension_sum(
    ax: plt.Axes,
    squeezing_values: np.ndarray | None = None,
    depth: int = 5,
    coupling_time: float = 1.0,
    fraction_to_exclude: float = 1/3,
) -> None:
    """Plot the sum Δ_x + Δ_p vs squeezing parameter.

    For a free boson CFT, this sum should equal 1.

    Args:
        ax: Matplotlib axes to plot on
        squeezing_values: Array of squeezing values
        depth: Tree depth for the decorated graph
        coupling_time: Quench coupling time
        fraction_to_exclude: Fraction of data to exclude when fitting
    """
    if squeezing_values is None:
        squeezing_values = np.logspace(-2, 1, 30)

    _, delta_x, delta_p, _ = compute_scaling_dimensions_vs_squeezing(
        squeezing_values, depth, coupling_time, fraction_to_exclude
    )

    colors = plotting.default_colors

    ax.semilogx(
        squeezing_values, delta_x + delta_p,
        '-o', color=colors[2], markersize=4, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\Delta_x + \Delta_p$'
    )

    # Reference line at 1 (free boson CFT prediction)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.6,
               label=r'$\Delta_x + \Delta_p = 1$')

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\Delta_x + \Delta_p$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_scaling_dimensions_multiple_depths(
    ax: plt.Axes,
    depths: list[int] | None = None,
    squeezing_values: np.ndarray | None = None,
    coupling_time: float = 1.0,
    fraction_to_exclude: float = 1/3,
    which: str = 'x',
) -> None:
    """Plot scaling dimension vs squeezing for multiple system sizes.

    Args:
        ax: Matplotlib axes to plot on
        depths: List of tree depths to compare
        squeezing_values: Array of squeezing values
        coupling_time: Quench coupling time
        fraction_to_exclude: Fraction of data to exclude when fitting
        which: 'x' for position, 'p' for momentum, 'I' for MI scaling dimension
    """
    if depths is None:
        depths = [4, 5, 6]
    if squeezing_values is None:
        squeezing_values = np.logspace(-2, 1, 20)

    colors = plotting.default_colors

    for i, depth in enumerate(depths):
        _, delta_x, delta_p, delta_mi = compute_scaling_dimensions_vs_squeezing(
            squeezing_values, depth, coupling_time, fraction_to_exclude
        )

        if which == 'x':
            delta = delta_x
            symbol = r'\Delta_x'
        elif which == 'p':
            delta = delta_p
            symbol = r'\Delta_p'
        else:
            # Divide by 2 since we plot sqrt(MI) ~ 1/d^(Delta_I)
            delta = delta_mi / 2
            symbol = r'\Delta_I'

        ax.semilogx(
            squeezing_values, delta,
            '-o', color=colors[i], markersize=4, alpha=0.9,
            linewidth=1, markeredgewidth=0,
            label=rf'$L = {2**depth}$'
        )

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(rf'${symbol}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def compute_mi_scaling_dimension_vs_squeezing(
    squeezing_values: np.ndarray,
    depth: int = 5,
    coupling_time: float = 1.0,
    fraction_to_exclude: float = 1/3,
    is_decorated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MI scaling dimension as a function of squeezing for decorated or undecorated graphs.

    Args:
        squeezing_values: Array of squeezing values to sweep
        depth: Tree depth
        coupling_time: Quench coupling time
        fraction_to_exclude: Fraction of data to exclude at edges when fitting
        is_decorated: If True, use decorated graph; if False, use undecorated

    Returns:
        Tuple of (squeezing_values, delta_mi)
    """
    delta_mi = []

    for mu in squeezing_values:
        if is_decorated:
            graph = graphs.subdivided_tree_decoration_1(depth, validate=False, draw=False)
        else:
            ctree = graphs.crosslinked_tree(depth)
            graph = make_tree_periodic(ctree, remove_nodes=False)

        analysis = Analysis(
            graph,
            couplingtime=coupling_time,
            global_presqueeze=mu,
            is_decorated=is_decorated,
            fraction_to_exclude_for_scaling_fit=fraction_to_exclude,
        )

        # Fit MI scaling dimension
        mi_delta = fit_mutual_info_scaling_dimension(analysis, fraction_to_exclude)
        delta_mi.append(mi_delta)

    return squeezing_values, np.array(delta_mi)


def plot_mi_scaling_dimension_comparison(
    ax: plt.Axes,
    squeezing_values: np.ndarray | None = None,
    depth: int = 5,
    coupling_time: float = 1.0,
    fraction_to_exclude: float = 1/3,
    show_theory_line: bool = True,
) -> None:
    """Plot MI scaling dimension vs squeezing for decorated and undecorated graphs.

    Args:
        ax: Matplotlib axes to plot on
        squeezing_values: Array of squeezing values (if None, uses default range)
        depth: Tree depth
        coupling_time: Quench coupling time
        fraction_to_exclude: Fraction of data to exclude when fitting
        show_theory_line: If True, show reference line at Δ = 1
    """
    if squeezing_values is None:
        squeezing_values = np.geomspace(0.01, 5, 25)

    colors = plotting.default_colors

    # Compute for decorated
    _, delta_mi_decorated = compute_mi_scaling_dimension_vs_squeezing(
        squeezing_values, depth, coupling_time, fraction_to_exclude, is_decorated=True
    )

    # Compute for undecorated
    _, delta_mi_undecorated = compute_mi_scaling_dimension_vs_squeezing(
        squeezing_values, depth, coupling_time, fraction_to_exclude, is_decorated=False
    )

    # Plot Δ_I / 2 since sqrt(MI) ~ 1/d^(Δ_I/2)
    ax.semilogx(
        squeezing_values, delta_mi_decorated / 2,
        '^', color=colors[2], markersize=6, alpha=0.9,
        markeredgewidth=0,
        label='Decorated'
    )

    ax.semilogx(
        squeezing_values, delta_mi_undecorated / 2,
        'P', color=colors[3], markersize=6, alpha=0.9,
        markeredgewidth=0,
        label='Undecorated'
    )

    if show_theory_line:
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5,
                   label=r'$\Delta_I / 2 = 1$')

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\Delta_I / 2$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_mi_scaling_dimension_multiple_depths(
    ax: plt.Axes,
    depths: list[int] | None = None,
    squeezing_values: np.ndarray | None = None,
    coupling_time: float = 1.0,
    fraction_to_exclude: float = 1/3,
    show_theory_line: bool = True,
) -> None:
    """Plot MI scaling dimension vs squeezing for multiple depths.

    Different depths get different colors. Decorated vs undecorated
    are distinguished by marker style (triangles vs plus signs).

    Args:
        ax: Matplotlib axes to plot on
        depths: List of tree depths to compare
        squeezing_values: Array of squeezing values (if None, uses default range)
        coupling_time: Quench coupling time
        fraction_to_exclude: Fraction of data to exclude when fitting
        show_theory_line: If True, show reference line at Δ = 1
    """
    if depths is None:
        depths = [5, 6, 7]
    if squeezing_values is None:
        squeezing_values = np.geomspace(0.01, 5, 20)

    colors = plotting.default_colors

    for i, depth in enumerate(depths):
        color = colors[i]
        L = 2**depth

        # Compute for decorated
        _, delta_mi_decorated = compute_mi_scaling_dimension_vs_squeezing(
            squeezing_values, depth, coupling_time, fraction_to_exclude, is_decorated=True
        )

        # Compute for undecorated
        _, delta_mi_undecorated = compute_mi_scaling_dimension_vs_squeezing(
            squeezing_values, depth, coupling_time, fraction_to_exclude, is_decorated=False
        )

        # Plot Δ_I / 2 since sqrt(MI) ~ 1/d^(Δ_I/2)
        # Decorated: triangles, Undecorated: plus signs
        ax.semilogx(
            squeezing_values, delta_mi_decorated / 2,
            '^', color=color, markersize=6, alpha=0.9,
            markeredgewidth=0,
            label=rf'Decorated $L={L}$'
        )

        ax.semilogx(
            squeezing_values, delta_mi_undecorated / 2,
            'P', color=color, markersize=6, alpha=0.9,
            markeredgewidth=0.5, markeredgecolor=color,
            markerfacecolor='white',
            label=rf'Undecorated $L={L}$'
        )

    if show_theory_line:
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5,
                   label=r'$\Delta_I / 2 = 1$')

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\Delta_I / 2$')
    # place legend below
    ax.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    ax.grid(True, alpha=0.3)

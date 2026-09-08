"""Helper functions for supplementary correlation figures."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import graph2grav.graphs as graphs
from graph2grav.graphs.standard import make_tree_periodic
from graph2grav.analysis import Analysis, format_with_uncertainty
from graph2grav.cartography.subsystems import mutual_information

import graph2grav.plotting as plotting


def compute_correlations_vs_distance(
    boundary_cov: np.ndarray,
    boundary_length: int,
    absolute_value: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute XX and PP correlations starting from site 0 going around the circle.

    Correlations are normalized by geometric mean of variances.

    Args:
        boundary_cov: Boundary covariance matrix in sequential ordering
                     (q1, ..., qN, p1, ..., pN)
        boundary_length: Number of boundary sites L

    Returns:
        Tuple of (distances, xx_corr, pp_corr) where:
        - distances: Array of site separations (1, 2, ..., L//2)
        - xx_corr: |<x_0 x_d>| / sqrt(<x_0^2><x_d^2>)
        - pp_corr: |<p_0 p_d>| / sqrt(<p_0^2><p_d^2>)
    """
    L = boundary_length
    N = L  # number of modes

    # Extract position and momentum blocks (sequential ordering)
    xx_block = boundary_cov[:N, :N]
    pp_block = boundary_cov[N:, N:]

    # Compute correlations from site 0 to sites 1, 2, ..., L//2
    distances = []
    xx_corr = []
    pp_corr = []

    for r in range(1, L // 2 + 1):
        # Normalize by geometric mean of variances
        xx_norm = xx_block[0, r] / np.sqrt(xx_block[0, 0] * xx_block[r, r])
        pp_norm = pp_block[0, r] / np.sqrt(pp_block[0, 0] * pp_block[r, r])

        if absolute_value:
            xx_norm = np.abs(xx_norm)
            pp_norm = np.abs(pp_norm)

        distances.append(r)
        xx_corr.append(xx_norm)
        pp_corr.append(pp_norm)

    return np.array(distances), np.array(xx_corr), np.array(pp_corr)


def compute_mutual_info_vs_distance(
    boundary_cov: np.ndarray,
    boundary_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mutual information between single sites as a function of distance.

    Args:
        boundary_cov: Boundary covariance matrix in sequential ordering
        boundary_length: Number of boundary sites L

    Returns:
        Tuple of (distances, mutual_info) where distances excludes 0
        (MI with self is not meaningful).
    """
    L = boundary_length

    distances = []
    mi_values = []

    for r in range(1, L // 2 + 1):
        # Mutual information between site 0 and site r
        mi = mutual_information(boundary_cov, [0], [r], ordering='seq')
        distances.append(r)
        mi_values.append(mi)

    return np.array(distances), np.array(mi_values)


def chord_distance(r: np.ndarray, L: int) -> np.ndarray:
    """Convert lattice distance to chord distance sin(pi * r / L).

    This is the natural distance measure for periodic boundaries,
    mapping to the chord length on the boundary circle.
    """
    return np.sin(np.pi * r / L)


def plot_correlations_and_mutual_info(
    ax: plt.Axes,
    depth: int = 5,
    squeezing: float = 0.1,
    coupling_time: float = 1.0,
    is_decorated: bool = False,
    show_theory_lines: bool = True,
    symlog: bool = False,
    fit_x: bool = False,
    fit_p: bool = False,
    fit_mi: bool = False,
    fraction_to_exclude_x: float = 1/3,
    fraction_to_exclude_p: float = 1/3,
    fraction_to_exclude_mi: float = 0.0,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    """Plot XX, PP correlations and mutual information vs chord distance.

    Creates a log-log plot showing:
    - XX correlations (position-position)
    - PP correlations (momentum-momentum)
    - Mutual information between single sites

    Args:
        ax: Matplotlib axes to plot on
        depth: Tree depth (boundary length = 2^depth)
        squeezing: Global presqueezing parameter
        coupling_time: Quench coupling time
        is_decorated: If True, use decorated graphs
        show_theory_lines: If True, add 1/d^2 reference line
        symlog: If True, use symlog scale for y-axis
        fit_x: If True, fit x correlations and show scaling dimension in legend
        fit_p: If True, fit p correlations and show scaling dimension in legend
        fit_mi: If True, fit MI and show scaling dimension in legend
        fraction_to_exclude_x: Fraction of data to exclude at edges when fitting x
        fraction_to_exclude_p: Fraction of data to exclude at edges when fitting p
        fraction_to_exclude_mi: Fraction of data to exclude at edges when fitting MI

    Returns:
        Tuple of ((delta_x, sigma_x), (delta_p, sigma_p), (delta_mi, sigma_mi))
        where each is a (value, uncertainty) pair. Returns None if not fitted.
        Uncertainties are standard errors from least-squares covariance matrix.
    """
    # Create graph and run analysis
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

    # Get boundary covariance in sequential ordering
    boundary_cov = analysis.boundary_after_bulk_measurement
    L = analysis.boundary_length

    # Compute correlations
    dists, xx_corr, pp_corr = compute_correlations_vs_distance(boundary_cov, L, absolute_value=not symlog)
    mi_dists, mi_vals = compute_mutual_info_vs_distance(boundary_cov, L)

    # Convert to chord distance
    chord_dists = chord_distance(dists, L)
    mi_chord_dists = chord_distance(mi_dists, L)

    # Plot correlations and MI bound
    colors = plotting.default_colors
    plotfunc = ax.semilogy if symlog else ax.loglog

    # Track fitted scaling dimensions and uncertainties
    delta_x = None
    delta_p = None
    delta_mi = None
    sigma_x = None
    sigma_p = None
    sigma_mi = None

    # Fit x correlations if requested
    if fit_x:
        length = len(chord_dists)
        start = max(1, int(length * fraction_to_exclude_x))
        end = length - start if fraction_to_exclude_x > 0 else length

        log_x_fit = np.log(chord_dists[start:end])
        log_y_fit = np.log(xx_corr[start:end])
        coeffs_x, cov_x = np.polyfit(log_x_fit, log_y_fit, 1, cov=True)
        slope_x, intercept_x = coeffs_x
        sigma_slope_x = np.sqrt(cov_x[0, 0])
        delta_x = -slope_x / 2  # Correlations decay as 1/d^(2*Delta)
        sigma_x = sigma_slope_x / 2  # Propagate uncertainty

    # Fit p correlations if requested
    if fit_p:
        length = len(chord_dists)
        start = max(1, int(length * fraction_to_exclude_p))
        end = length - start if fraction_to_exclude_p > 0 else length

        log_x_fit = np.log(chord_dists[start:end])
        log_y_fit = np.log(pp_corr[start:end])
        coeffs_p, cov_p = np.polyfit(log_x_fit, log_y_fit, 1, cov=True)
        slope_p, intercept_p = coeffs_p
        sigma_slope_p = np.sqrt(cov_p[0, 0])
        delta_p = -slope_p / 2  # Correlations decay as 1/d^(2*Delta)
        sigma_p = sigma_slope_p / 2  # Propagate uncertainty

    # XX correlations - circles
    plotfunc(
        chord_dists, xx_corr,
        'o', color=colors[0], markersize=4, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\mathrm{Corr}(x_0, x_d)$' if fit_x else r'$\frac{\langle x_0 x_d \rangle}{\sqrt{\langle x_0^2 \rangle \langle x_d^2 \rangle}}$'
    )

    # PP correlations - squares
    plotfunc(
        chord_dists, pp_corr,
        's', color=colors[1], markersize=4, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\mathrm{Corr}(p_0, p_d)$' if fit_p else r'$\frac{\langle p_0 p_d \rangle}{\sqrt{\langle p_0^2 \rangle \langle p_d^2 \rangle}}$'
    )

    # sqrt(2 * Mutual information) - the proper bound on correlations
    plotfunc(
        mi_chord_dists, np.sqrt(2  * mi_vals),
        '^', color=colors[2], markersize=4, alpha=0.9,
        linewidth=1, markeredgewidth=0,
        label=r'$\sqrt{2I(0:d)}$' if fit_mi else r'$\sqrt{2 I(0:d)}$'
    )

    x_theory = np.linspace(chord_dists[0], chord_dists[-1], 100)

    # Add fit lines for x and p if requested (with scaling dimension in legend)
    if fit_x:
        y_fit_x = np.exp(intercept_x) * x_theory**slope_x
        plotfunc(
            x_theory, y_fit_x,
            '--', color=colors[0], linewidth=1, alpha=0.6,
            label=rf'$\Delta_x = {format_with_uncertainty(delta_x, sigma_x)}$'
        )

    if fit_p:
        y_fit_p = np.exp(intercept_p) * x_theory**slope_p
        plotfunc(
            x_theory, y_fit_p,
            '--', color=colors[1], linewidth=1, alpha=0.6,
            label=rf'$\Delta_p = {format_with_uncertainty(delta_p, sigma_p)}$'
        )

    if show_theory_lines:
        # Correlations decay as 1/r^2
        # Fit amplitude from data (use middle of range)
        mid_idx = len(chord_dists) // 3
        amp = xx_corr[mid_idx] * chord_dists[mid_idx]**2

        plotfunc(
            x_theory, amp / x_theory**2,
            '--', color='black', linewidth=1, alpha=0.6,
            label=r'$\propto 1/d^2$'
        )

    if fit_mi:
        # Fit MI power law: I ~ d^(-2*Delta_I)
        length = len(mi_chord_dists)
        start = max(1, int(length * fraction_to_exclude_mi))
        end = length - start if fraction_to_exclude_mi > 0 else length

        log_x = np.log(mi_chord_dists[start:end])
        log_y = np.log(mi_vals[start:end])  # Fit MI directly

        coeffs_mi, cov_mi = np.polyfit(log_x, log_y, 1, cov=True)
        slope_mi, intercept_mi = coeffs_mi
        sigma_slope_mi = np.sqrt(cov_mi[0, 0])
        delta_mi = -slope_mi / 2  # MI ~ d^(-2*Delta_I), so slope = -2*Delta_I
        sigma_mi = sigma_slope_mi / 2
        # For sqrt(2*MI) plot, divide exponent by 2
        y_fit = np.sqrt(2 * np.exp(intercept_mi)) * x_theory**(slope_mi / 2)

        plotfunc(
            x_theory, y_fit,
            '--', color=colors[2], linewidth=1, alpha=0.6,
            label=rf'$\Delta_I = {format_with_uncertainty(delta_mi, sigma_mi)}$'
        )
    if symlog:
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.set_xscale('log')

    ax.set_xlabel(r'$\sin(\pi d / L)$')
    # ax.set_ylabel('Correlation / Mutual Information')
    # ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Return (value, uncertainty) tuples or None
    result_x = (delta_x, sigma_x) if delta_x is not None else None
    result_p = (delta_p, sigma_p) if delta_p is not None else None
    result_mi = (delta_mi, sigma_mi) if delta_mi is not None else None
    return result_x, result_p, result_mi


def plot_mutual_info_multiple_squeezing(
    ax: plt.Axes,
    depth: int = 5,
    squeezing_values: list[float] | None = None,
    coupling_time: float = 1.0,
    is_decorated: bool = False,
) -> None:
    """Plot mutual information vs distance for multiple squeezing values.

    Args:
        ax: Matplotlib axes to plot on
        depth: Tree depth
        squeezing_values: List of squeezing values to compare
        coupling_time: Quench coupling time
        is_decorated: If True, use decorated graphs
    """
    if squeezing_values is None:
        squeezing_values = [1, 0.2, 0.1, 0.05]

    # Use plasma colormap for squeezing values
    cmap = plt.cm.plasma
    n = len(squeezing_values)
    colors = [cmap(0.7 * i / (n - 1)) if n > 1 else cmap(0.5) for i in range(n)]

    for squeezing, color in zip(squeezing_values, colors):
        # Create graph and run analysis
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

        boundary_cov = analysis.boundary_after_bulk_measurement
        L = analysis.boundary_length

        mi_dists, mi_vals = compute_mutual_info_vs_distance(boundary_cov, L)
        mi_chord_dists = chord_distance(mi_dists, L)

        ax.loglog(
            mi_chord_dists, np.sqrt(2 * mi_vals),
            '-o', color=color, markersize=4, alpha=0.9,
            linewidth=1, markeredgewidth=0,
            label=rf'$\mu = {squeezing:.2g}$'
        )

    ax.set_xlabel(r'Chord distance $\sin(\pi d / L)$')
    ax.set_ylabel(r'$\sqrt{2 I(0:d)}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')


def plot_mutual_info_with_fit(
    ax: plt.Axes,
    depth: int = 5,
    squeezing: float = 0.1,
    coupling_time: float = 1.0,
    is_decorated: bool = False,
    fraction_to_exclude: float = 1/3,
) -> float:
    """Plot mutual information vs distance with power-law fit overlaid.

    Shows sqrt(2*MI) vs chord distance with the fitted power-law line
    to verify the scaling dimension fit is working.

    Args:
        ax: Matplotlib axes to plot on
        depth: Tree depth (boundary length = 2^depth)
        squeezing: Global presqueezing parameter
        coupling_time: Quench coupling time
        is_decorated: If True, use decorated graphs
        fraction_to_exclude: Fraction of data to exclude at edges when fitting

    Returns:
        Fitted scaling dimension (Delta_I / 2, i.e. the exponent for sqrt(MI))
    """
    # Create graph and run analysis
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

    # Get boundary covariance in sequential ordering
    boundary_cov = analysis.boundary_after_bulk_measurement
    L = analysis.boundary_length

    # Compute MI vs distance
    mi_dists, mi_vals = compute_mutual_info_vs_distance(boundary_cov, L)
    mi_chord_dists = chord_distance(mi_dists, L)

    # sqrt(2 * MI) is what we plot
    sqrt_mi = np.sqrt(2 * mi_vals)

    # Fit in log-log space, excluding edges
    length = len(mi_chord_dists)
    start = max(1, int(length * fraction_to_exclude))
    end = length - start

    log_x = np.log(mi_chord_dists[start:end])
    log_y = np.log(sqrt_mi[start:end])

    # sqrt(MI) ~ 1/d^(Delta_I/2), so log(sqrt(MI)) = const - (Delta_I/2)*log(d)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    scaling_dim = -slope  # Delta_I / 2

    colors = plotting.default_colors

    # Plot data
    ax.loglog(
        mi_chord_dists, sqrt_mi,
        'o', color=colors[2], markersize=5, alpha=0.9,
        markeredgewidth=0,
        label=r'$\sqrt{2 I(0:d)}$'
    )

    # Plot fit line
    x_fit = np.linspace(mi_chord_dists[0], mi_chord_dists[-1], 100)
    y_fit = np.exp(intercept) * x_fit**slope

    ax.loglog(
        x_fit, y_fit,
        '--', color='black', linewidth=1.5, alpha=0.7,
        label=rf'Fit: $\propto 1/d^{{{scaling_dim:.2f}}}$'
    )

    # Mark the fit region
    ax.axvline(mi_chord_dists[start], color='gray', linestyle=':', alpha=0.4)
    ax.axvline(mi_chord_dists[end-1], color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel(r'$\sin(\pi d / L)$')
    ax.set_ylabel(r'$\sqrt{2 I(0:d)}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    return scaling_dim

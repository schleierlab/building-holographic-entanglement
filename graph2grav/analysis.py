"""Unified Analysis class for graph entanglement analysis.

This module provides a single Analysis class that works for both decorated
(weighted) and undecorated (unweighted) graphs.
"""

import colorsys
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgb, to_rgba

from graph2grav import gaussian
from graph2grav import technical_helpers
from graph2grav import analytic_entropies


def format_with_uncertainty(value: float, uncertainty: float) -> str:
    """Format a value with uncertainty in scientific notation with parentheses style.

    Automatically determines significant figures from uncertainty (1 sig fig).

    Examples:
        format_with_uncertainty(0.5012, 0.003) -> "5.0(3) \\times 10^{-1}"
        format_with_uncertainty(1.653e-3, 0.021e-3) -> "1.65(2) \\times 10^{-3}"

    Args:
        value: The measured value
        uncertainty: The standard error

    Returns:
        LaTeX-formatted string with value(uncertainty) \\times 10^{exponent}
    """
    if uncertainty <= 0 or not np.isfinite(uncertainty):
        return f"{value:.2e}"

    # Determine order of magnitude of uncertainty (1 sig fig)
    unc_exponent = math.floor(math.log10(abs(uncertainty)))
    round_to = unc_exponent  # Round to 1 significant figure

    # Round uncertainty and value to same precision
    unc_rounded = round(uncertainty, -round_to)
    val_rounded = round(value, -round_to)

    # Determine exponent for scientific notation (based on value)
    if val_rounded == 0:
        val_exponent = 0
    else:
        val_exponent = math.floor(math.log10(abs(val_rounded)))

    # Express mantissa and uncertainty in terms of the value's exponent
    mantissa = val_rounded / (10 ** val_exponent)
    unc_scaled = unc_rounded / (10 ** val_exponent)

    # Number of decimal places needed for mantissa
    decimal_places = max(0, val_exponent - round_to)

    # Format the uncertainty as integer (last digits)
    unc_last_digits = round(unc_scaled * (10 ** decimal_places))

    # Build the string
    mantissa_str = f"{mantissa:.{decimal_places}f}"

    if val_exponent == 0:
        return f"{mantissa_str}({unc_last_digits:.0f})"
    else:
        return f"{mantissa_str}({unc_last_digits:.0f}) \\times 10^{{{val_exponent}}}"


class Analysis:
    """Analyze entanglement properties of a graph after quench and measurement protocol.

    This unified class handles both decorated (weighted) and undecorated (unweighted)
    graphs. Use `is_decorated=True` for graphs with edge weights (mweight attribute).

    Attributes:
        graph: The input NetworkX graph
        C: Full covariance matrix after quench
        boundary_after_bulk_measurement: Covariance of boundary after measuring bulk momenta
        boundary_entropies: Entanglement entropies by region size
        central_charge: Fitted central charge (None until fit_central_charge() is called)
    """

    def __init__(
        self,
        graph: nx.Graph,
        couplingtime: float = 1.0,
        global_presqueeze: float = 1.0,
        is_decorated: bool = False,
        config: dict | None = None,
        fraction_to_exclude_for_scaling_fit: float = 1/3,
    ):
        """Initialize analysis for a graph.

        Args:
            graph: NetworkX graph with proper node attributes (is_ancilla, is_boundary, b0, etc.)
            couplingtime: Duration of the XX interaction quench
            global_presqueeze: Initial squeezing applied to all nodes
            is_decorated: If True, use weighted quench (for decorated graphs with mweight);
                         if False, use unweighted quench
            config: Optional configuration dict for plotting (contains color schemes, etc.)
            fraction_to_exclude_for_scaling_fit: Fraction of data to exclude at edges
                when fitting scaling dimensions (default 1/3)
        """
        self.graph = graph
        self.couplingtime = couplingtime
        self.global_presqueeze = global_presqueeze
        self.is_decorated = is_decorated
        self.config = config
        self.fraction_to_exclude_for_scaling_fit = fraction_to_exclude_for_scaling_fit

        # Compute covariance matrix using appropriate quench type
        if is_decorated:
            self.C = gaussian.covariance_from_weighted_quench(
                graph,
                couplingtime=couplingtime,
                global_presqueeze=global_presqueeze
            )
        else:
            # For undecorated, pass global_presqueeze to both bulk and boundary
            self.C = gaussian.covariance_from_unweighted_quench(
                graph,
                couplingtime=couplingtime,
                bulk_presqueeze=global_presqueeze,
                boundary_presqueeze=global_presqueeze
            )

        # Extract boundary information
        self.boundary_initial = np.abs(self.C)[gaussian.mesh(graph, "b0", "b0")]
        self.kept_indices = gaussian.indices(graph, "b0")

        self.boundary_before_measurement = self.C[np.ix_(self.kept_indices, self.kept_indices)]
        self.boundary_after_bulk_measurement = gaussian.get_measured(
            self.C,
            gaussian.indices(graph, "b0"),
            gaussian.momentum_indices(graph, "is_ancilla")
        )
        self.boundary_after_bulk_pos_measurement = gaussian.get_measured(
            self.C,
            gaussian.indices(graph, "b0"),
            gaussian.position_indices(graph, "is_ancilla")
        )

        self.boundary_length = len(gaussian.position(self.boundary_after_bulk_measurement))
        self.periodic = True  # graph.graph["periodic"]

        # Boundary node mapping for entropy calculations
        self.boundary_nodes_map, self.unique_sorted_positions = \
            technical_helpers.get_boundary_nodes_by_position(graph, kept_indices=self.kept_indices)

        # Compute entropies by region size
        self.boundary_entropies = gaussian.boundary_get_EE_by_length(
            self.boundary_after_bulk_measurement, self.boundary_nodes_map
        )
        self.boundary_entropies_pos = gaussian.boundary_get_EE_by_length(
            self.boundary_after_bulk_pos_measurement, self.boundary_nodes_map
        )

        self.region_sizes = np.arange(len(self.boundary_entropies))

        # Initialize fit results to None (computed on demand)
        self.central_charge: float | None = None
        self.central_charge_stderr: float | None = None
        self.cft_entropy_offset: float | None = None
        self.cft_entropy_offset_stderr: float | None = None
        self.position_scaling_dimension: float | None = None
        self.position_scaling_dimension_stderr: float | None = None
        self.position_normalization: float | None = None
        self.momentum_scaling_dimension: float | None = None
        self.momentum_scaling_dimension_stderr: float | None = None
        self.momentum_normalization: float | None = None

    @property
    def xs(self):
        """X-coordinates for plotting (sin-transformed for periodic boundaries)."""
        if self.periodic:
            return np.sin(np.pi * np.arange(self.boundary_length) / self.boundary_length)
        else:
            return np.arange(self.boundary_length)

    @property
    def linear_xs(self):
        """Linear x-coordinates (site indices)."""
        return np.arange(self.boundary_length)

    @property
    def xlabel(self):
        """X-axis label for plots."""
        if self.periodic:
            return r"$\sin(\pi j / L)$"
        else:
            return r"Site $j$"

    # -------------------------------------------------------------------------
    # Fitting methods (opt-in, not called automatically)
    # -------------------------------------------------------------------------

    def fit_central_charge(self, print_fit: bool = False) -> tuple[float, float]:
        """Fit the central charge using the entanglement entropy.

        Stores results in self.central_charge, self.central_charge_stderr,
        self.cft_entropy_offset, and self.cft_entropy_offset_stderr.

        Args:
            print_fit: If True, print the fitted values

        Returns:
            Tuple of (central_charge, offset)
        """
        from scipy.optimize import curve_fit

        def cft_entropy(length, c, offset=0):
            return analytic_entropies.periodic_CFT_entropy(
                length, self.boundary_length, central_charge=c, offset=offset
            )

        popt, pcov = curve_fit(
            cft_entropy,
            self.region_sizes[1:-1],
            self.boundary_entropies[1:-1],
            p0=[1, 0]
        )

        self.central_charge = popt[0]
        self.cft_entropy_offset = popt[1]
        # Standard errors are square roots of diagonal of covariance matrix
        self.central_charge_stderr = np.sqrt(pcov[0, 0])
        self.cft_entropy_offset_stderr = np.sqrt(pcov[1, 1])

        if print_fit:
            print(f"Fitted central charge: {popt[0]} ± {self.central_charge_stderr}")
            print(f"Fitted offset: {popt[1]} ± {self.cft_entropy_offset_stderr}")

        return popt[0], popt[1]

    def fit_position_scaling_dimension(
        self,
        fraction_to_exclude: float | None = None
    ) -> tuple[float, float]:
        """Fit the scaling dimension from position covariance data.

        Stores results in self.position_scaling_dimension, self.position_scaling_dimension_stderr,
        and self.position_normalization.

        Args:
            fraction_to_exclude: Fraction of data to exclude at edges.
                If None, uses self.fraction_to_exclude_for_scaling_fit

        Returns:
            Tuple of (scaling_dimension, normalization)
        """
        if fraction_to_exclude is None:
            fraction_to_exclude = self.fraction_to_exclude_for_scaling_fit

        length = len(self.xs)
        start = max(1, int(length * fraction_to_exclude))
        end = length - start

        log_xs = np.log(self.xs[start:end])
        log_cov = np.log(np.abs(gaussian.position(self.boundary_after_bulk_measurement)[0, start:end]))

        coeffs, cov = np.polyfit(log_xs, log_cov, 1, cov=True)
        slope, intercept = coeffs
        slope_stderr = np.sqrt(cov[0, 0])
        scaling_dimension = -slope / 2
        scaling_dimension_stderr = slope_stderr / 2  # Propagate uncertainty
        normalization = np.exp(intercept)

        self.position_scaling_dimension = scaling_dimension
        self.position_scaling_dimension_stderr = scaling_dimension_stderr
        self.position_normalization = normalization

        return scaling_dimension, normalization

    def fit_momentum_scaling_dimension(
        self,
        fraction_to_exclude: float | None = None
    ) -> tuple[float, float]:
        """Fit the scaling dimension from momentum covariance data.

        Stores results in self.momentum_scaling_dimension, self.momentum_scaling_dimension_stderr,
        and self.momentum_normalization.

        Args:
            fraction_to_exclude: Fraction of data to exclude at edges.
                If None, uses self.fraction_to_exclude_for_scaling_fit

        Returns:
            Tuple of (scaling_dimension, normalization)
        """
        if fraction_to_exclude is None:
            fraction_to_exclude = self.fraction_to_exclude_for_scaling_fit

        length = len(self.xs)
        start = max(1, int(length * fraction_to_exclude))
        end = length - start

        log_xs = np.log(self.xs[start:end])
        log_cov = np.log(np.abs(gaussian.momentum(self.boundary_after_bulk_measurement)[0, start:end]))

        coeffs, cov = np.polyfit(log_xs, log_cov, 1, cov=True)
        slope, intercept = coeffs
        slope_stderr = np.sqrt(cov[0, 0])
        scaling_dimension = -slope / 2
        scaling_dimension_stderr = slope_stderr / 2  # Propagate uncertainty
        normalization = np.exp(intercept)

        self.momentum_scaling_dimension = scaling_dimension
        self.momentum_scaling_dimension_stderr = scaling_dimension_stderr
        self.momentum_normalization = normalization

        return scaling_dimension, normalization

    def get_renyi_entropies_across_alpha(self, alphas, region_size: int):
        """Compute the Renyi entropies for a given region size across a range of alpha values.

        Args:
            alphas: Array of Renyi indices
            region_size: Size of the boundary region

        Returns:
            List of Renyi entropies for each alpha
        """
        interC = gaussian.to_inter(self.boundary_after_bulk_measurement)

        return [
            gaussian.calculate_boundary_entropy(
                interC, np.arange(region_size), self.boundary_nodes_map,
                renyi=True, alpha=alpha
            )
            for alpha in alphas
        ]

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------

    def plot_entanglement_entropy(
        self,
        ax: plt.Axes,
        include_ends: bool = False,
        zorder: int | None = None,
        markersize: float | None = None,
        alpha: float = 0.3,
        color: str | None = None,
        lighten_amount: float = 0.5,
        use_simple_markers: bool = False,
        config: dict | None = None,
    ) -> plt.Axes:
        """Plot entanglement entropy vs region size.

        Args:
            ax: Matplotlib axes to plot on
            include_ends: If True, include region sizes 0 and L
            zorder: Z-order for plot elements
            markersize: Marker size
            alpha: Marker face alpha (or full marker alpha if use_simple_markers=True)
            color: Override color (if None, uses config)
            lighten_amount: Amount to lighten marker edge color (0=no change, 1=white)
            use_simple_markers: If True, use simple marker style with alpha on full marker
                (intro style); if False, use markerfacecolor with alpha + lightened edge
                (wormhole style)
            config: Config dict (if None, uses self.config)

        Returns:
            The axes object
        """
        cfg = config if config is not None else self.config
        assert cfg is not None, "config must not be None"

        region_sizes = self.region_sizes if include_ends else self.region_sizes[1:-1]
        boundary_entropies = self.boundary_entropies if include_ends else self.boundary_entropies[1:-1]

        if color is None:
            color = cfg["graph"]["nodes"]["selected_boundary"]["ryu_takayanagi"]["color"]

        if use_simple_markers:
            # Simple style (used by intro): alpha applies to full marker
            ax.plot(
                region_sizes, boundary_entropies, "o",
                alpha=alpha,
                color=color,
                zorder=zorder,
                markersize=markersize,
            )
        else:
            # Wormhole style: alpha on face only, lightened edge
            def lighten_keep_hue(c, amount=0.3):
                """Lighten a color while keeping its hue."""
                r, g, b = to_rgb(c)
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                l = l + (1 - l) * amount
                r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
                return (r2, g2, b2)

            markeredgecolor = lighten_keep_hue(color, amount=lighten_amount)

            ax.plot(
                region_sizes, boundary_entropies, "o",
                markerfacecolor=to_rgba(color, alpha=alpha),
                markeredgecolor=markeredgecolor,
                zorder=zorder,
                markersize=markersize,
                markeredgewidth=0.5,
            )

        return ax

    def plot_momentum_covariance(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot momentum covariance vs distance.

        Args:
            ax: Matplotlib axes (if None, creates new figure)

        Returns:
            The axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        nvals = len(self.xs)
        ax.loglog(
            self.xs, gaussian.momentum(self.boundary_initial)[0, :nvals],
            "o", alpha=0.7, label="Before measurement"
        )
        ax.loglog(
            self.xs, np.abs(gaussian.momentum(self.boundary_after_bulk_measurement)[0, :nvals]),
            "o", alpha=0.7, label="After measurement"
        )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(r"cov($p_0$, $p_j$)")
        ax.legend()

        return ax

    def plot_position_covariance(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot position covariance vs distance.

        Args:
            ax: Matplotlib axes (if None, creates new figure)

        Returns:
            The axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.loglog(
            self.xs, gaussian.position(self.boundary_initial)[0, :],
            "o", alpha=0.7, label="Before measurement"
        )
        ax.loglog(
            self.xs, np.abs(gaussian.position(self.boundary_after_bulk_measurement)[0, :]),
            "o", alpha=0.7, label="After measurement"
        )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(r"cov($x_0$, $x_j$)")
        ax.legend()

        return ax

    def plot_position_covariance_v2(self, cpower: float = 1, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot position covariance with power compensation.

        Args:
            cpower: Compensation power
            ax: Matplotlib axes (if None, creates new figure)

        Returns:
            The axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        xlen = len(self.xs)
        chosen_xs = self.xs[1:(xlen//2):2]
        chosen_xs = chosen_xs / np.max(chosen_xs)

        chosen_data = (
            gaussian.position(self.boundary_after_bulk_measurement)[0, 1:(xlen//2):2]
            * self.xs[1:(xlen//2):2]**cpower
        )
        ax.scatter(
            chosen_xs, chosen_data, alpha=0.7, label="After measurement",
            c=plt.cm.viridis(np.linspace(0, 1, len(chosen_xs)))  # pyright: ignore
        )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(r"cov($x_0$, $x_j$)")
        ax.set_title(f"Compensation Power = {cpower}")
        ax.legend()

        return ax

    def plot_position_powerlaw(
        self,
        ax: plt.Axes,
        scaling_dimension: float | None = None,
        label: str | None = None,
        color: str | None = None,
    ) -> None:
        """Plot power-law fit for position covariance.

        Args:
            ax: Matplotlib axes
            scaling_dimension: Override scaling dimension (if None, uses fitted value)
            label: Plot label
            color: Line color
        """
        if scaling_dimension is None:
            scaling_dimension = self.position_scaling_dimension
        if scaling_dimension is None:
            raise ValueError("No scaling dimension available. Call fit_position_scaling_dimension() first.")
        if self.position_normalization is None:
            raise ValueError("No normalization available. Call fit_position_scaling_dimension() first.")

        ax.loglog(
            self.xs[1:],
            self.position_normalization * self.xs[1:] ** (-2 * scaling_dimension),
            label=label,
            color=color
        )

    def plot_momentum_powerlaw(
        self,
        ax: plt.Axes,
        scaling_dimension: float | None = None,
        label: str | None = None,
        color: str | None = None,
    ) -> None:
        """Plot power-law fit for momentum covariance.

        Args:
            ax: Matplotlib axes
            scaling_dimension: Override scaling dimension (if None, uses fitted value)
            label: Plot label
            color: Line color
        """
        if scaling_dimension is None:
            scaling_dimension = self.momentum_scaling_dimension
        if scaling_dimension is None:
            raise ValueError("No scaling dimension available. Call fit_momentum_scaling_dimension() first.")
        if self.momentum_normalization is None:
            raise ValueError("No normalization available. Call fit_momentum_scaling_dimension() first.")

        ax.loglog(
            self.xs[1:],
            self.momentum_normalization * self.xs[1:] ** (-2 * scaling_dimension),
            label=label,
            color=color
        )

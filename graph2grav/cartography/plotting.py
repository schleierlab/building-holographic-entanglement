"""Visualization tools for bulk reconstruction and entanglement wedge analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx


def plot_mutual_info_heatmap(results, ax=None, cmap='viridis', **kwargs):
    """Plot 2D heatmap of mutual information I(R:b).

    Creates a heatmap showing how mutual information between boundary region R
    and bulk node b varies with region size and bulk position.

    Args:
        results: List of dicts from sweep_bulk_reconstruction(), each containing:
                 - 'bulk_position': Node index where bulk was added
                 - 'region_size': Size of boundary region R
                 - 'MI_R_b': Mutual information I(R:b)
        ax: Matplotlib axes to plot on (creates new figure if None)
        cmap: Colormap name (default 'viridis')
        **kwargs: Additional arguments passed to imshow

    Returns:
        fig, ax, im: Matplotlib figure, axes, and image objects

    Example:
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
        >>> fig, ax, im = plot_mutual_info_heatmap(results)
        >>> plt.colorbar(im, ax=ax, label='I(R:b)')
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Extract unique bulk positions and region sizes
    bulk_positions = sorted(set(r['bulk_position'] for r in results))
    region_sizes = sorted(set(r['region_size'] for r in results))

    # Create 2D array for MI values
    MI_array = np.zeros((len(bulk_positions), len(region_sizes)))

    # Fill array
    for r in results:
        i = bulk_positions.index(r['bulk_position'])
        j = region_sizes.index(r['region_size'])
        MI_array[i, j] = r['MI_R_b']

    # Plot heatmap
    im = ax.imshow(MI_array, aspect='auto', origin='lower',
                   cmap=cmap, **kwargs)

    # Set tick labels
    ax.set_xticks(range(len(region_sizes)))
    ax.set_xticklabels(region_sizes)
    ax.set_yticks(range(len(bulk_positions)))
    ax.set_yticklabels(bulk_positions)

    # Labels
    ax.set_xlabel('Region Size |R|', fontsize=12)
    ax.set_ylabel('Bulk Position', fontsize=12)
    ax.set_title('Mutual Information I(R:b)', fontsize=14)

    plt.colorbar(im, ax=ax, label='I(R:b)')


    return fig, ax, im


def plot_wedge_boundary(graph, results, region_size, ax=None,
                       threshold=0.5, node_size=300,
                       cmap='coolwarm', pos=None, config=None, **kwargs):
    """Plot graph with bulk nodes colored by mutual information.

    Visualizes which bulk nodes are in the entanglement wedge of a boundary
    region by coloring nodes according to their mutual information with the region.

    Args:
        graph: Base graph (before bulk node addition)
        results: List of dicts from sweep_bulk_reconstruction()
        region_size: Which region size to visualize
        ax: Matplotlib axes (creates new figure if None)
        threshold: MI threshold for "in wedge" (default 0.5)
        node_size: Size of nodes in plot (default 300, ignored if config provided)
        cmap: Colormap for MI values (default 'coolwarm')
        pos: Node positions dict (uses spring_layout if None)
        config: Configuration dict from loadconfig (optional, for consistent styling)
        **kwargs: Additional arguments passed to draw_networkx

    Returns:
        fig, ax: Matplotlib figure and axes

    Example:
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
        >>> fig, ax = plot_wedge_boundary(G, results, region_size=4, config=config)
        >>> plt.show()

    Notes:
        - Boundary nodes are drawn as squares
        - Bulk nodes are colored by MI value (warm = high MI)
        - Nodes with MI > threshold are considered "in the wedge"
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Filter results for this region size
    region_results = [r for r in results if r['region_size'] == region_size]

    if len(region_results) == 0:
        raise ValueError(f"No results found for region_size={region_size}")

    # Create dict mapping bulk_position -> MI value
    MI_by_position = {r['bulk_position']: r['MI_R_b']
                      for r in region_results}

    # Get node positions
    if pos is None:
        pos = nx.spring_layout(graph, seed=42)

    # Separate boundary and ancilla nodes
    boundary_nodes = [n for n, d in graph.nodes(data=True)
                      if d.get('is_boundary', False)]
    ancilla_nodes = [n for n, d in graph.nodes(data=True)
                     if d.get('is_ancilla', False)]

    # Get MI values for ancilla nodes (bulk positions)
    MI_values = [MI_by_position.get(n, 0.0) for n in ancilla_nodes]

    # Normalize MI values for colormap
    if len(MI_values) > 0 and max(MI_values) > 0:
        norm = Normalize(vmin=0, vmax=max(MI_values))
    else:
        norm = Normalize(vmin=0, vmax=1)

    # Use config colors if provided
    if config is not None:
        linewidth = config["graph"]["linewidth"]
        boundary_outer_color = config["graph"]["nodes"]["boundary"]["outer"]["color"]
        boundary_inner_color = config["graph"]["nodes"]["boundary"]["inner"]["color"]
        boundary_size = config["graph"]["nodes"]["boundary"]["outer"]["size"]

        # Draw edges
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, width=linewidth)

        # Draw boundary nodes with double circle
        nx.draw_networkx_nodes(graph, pos, nodelist=boundary_nodes,
                              node_color=boundary_outer_color,
                              node_shape='s',
                              node_size=boundary_size*2,
                              ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=boundary_nodes,
                              node_color=boundary_inner_color,
                              node_shape='s',
                              node_size=boundary_size,
                              ax=ax,
                              label='Boundary')
    else:
        # Draw edges
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)

        # Draw boundary nodes (squares)
        nx.draw_networkx_nodes(graph, pos, nodelist=boundary_nodes,
                              node_color='lightblue',
                              node_shape='s',
                              node_size=node_size,
                              ax=ax,
                              label='Boundary')

    # Draw ancilla nodes (circles) colored by MI
    if len(ancilla_nodes) > 0:
        ancilla_size = config["graph"]["nodes"]["ancilla"]["outer"]["size"]*1.5 if config else node_size
        nx.draw_networkx_nodes(graph, pos, nodelist=ancilla_nodes,
                              node_color=MI_values,
                              node_shape='o',
                              node_size=ancilla_size,
                              cmap=cmap,
                              vmin=norm.vmin,
                              vmax=norm.vmax,
                              ax=ax,
                              edgecolors='black',
                              linewidths=1,
                              label='Bulk')

    # Draw labels if configured
    if config is None or config["graph"]["which_nodes_to_label"] != "none":
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='I(R:b)')

    # Title
    ax.set_title(f'Entanglement Wedge (Region Size = {region_size})')
    ax.legend(loc='best')
    ax.axis('off')

    return fig, ax


def plot_mutual_info_vs_region_size(results, bulk_position, ax=None, **kwargs):
    """Plot mutual information I(R:b) vs region size for a fixed bulk position.

    Shows how mutual information between boundary region R and bulk node b
    varies with the size of the boundary region for a fixed bulk node position.

    Args:
        results: List of dicts from sweep_bulk_reconstruction()
        bulk_position: Which bulk position to analyze
        ax: Matplotlib axes (creates new figure if None)
        **kwargs: Additional arguments passed to plot

    Returns:
        fig, ax: Matplotlib figure and axes

    Example:
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
        >>> fig, ax = plot_mutual_info_vs_region_size(results, bulk_position=10)
        >>> plt.show()

    Notes:
        - Plots only I(R:b) vs region size
        - Use plot_entropy_components() to see all entropy terms
        - I(R:b) characterizes entanglement wedge structure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Filter results for this bulk position
    bulk_results = [r for r in results
                    if r['bulk_position'] == bulk_position]

    if len(bulk_results) == 0:
        raise ValueError(f"No results found for bulk_position={bulk_position}")

    # Sort by region size
    bulk_results = sorted(bulk_results, key=lambda r: r['region_size'])

    # Extract data
    region_sizes = [r['region_size'] for r in bulk_results]
    MI_R_b = [r['MI_R_b'] for r in bulk_results]

    # Plot
    ax.plot(region_sizes, MI_R_b, 'o-', label='$I(R:b)$',
           linewidth=2.5, markersize=8, color='red', alpha=0.2)

    # Labels and formatting
    ax.set_xlabel('Region Size |R|', fontsize=12)
    ax.set_ylabel('Mutual Information $I(R:b)$', fontsize=12)
    ax.set_title(f'Mutual Information vs Region Size (Bulk Position = {bulk_position})',
                fontsize=14)
    # ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_bulk_entropy_only(results, bulk_position, ax=None, **kwargs):
    """Plot bulk node entropy S_b vs region size for a fixed bulk position.

    Shows how the entropy of the bulk node alone varies with the size
    of the boundary region R for a fixed bulk node position.

    Args:
        results: List of dicts from sweep_bulk_reconstruction()
        bulk_position: Which bulk position to analyze
        ax: Matplotlib axes (creates new figure if None)
        **kwargs: Additional arguments passed to plot

    Returns:
        fig, ax: Matplotlib figure and axes

    Example:
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
        >>> fig, ax = plot_bulk_entropy_only(results, bulk_position=10)
        >>> plt.show()

    Notes:
        - Plots only S_b (bulk entropy) vs region size
        - Useful for understanding bulk node entanglement structure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Filter results for this bulk position
    bulk_results = [r for r in results
                    if r['bulk_position'] == bulk_position]

    if len(bulk_results) == 0:
        raise ValueError(f"No results found for bulk_position={bulk_position}")

    # Sort by region size
    bulk_results = sorted(bulk_results, key=lambda r: r['region_size'])

    # Extract data
    region_sizes = [r['region_size'] for r in bulk_results]
    S_b = [r['S_b'] for r in bulk_results]

    # Plot
    ax.plot(region_sizes, S_b, 'o-', label='$S_b$ (bulk)',
           linewidth=2, markersize=6)

    # Labels and formatting
    ax.set_xlabel('Region Size |R|', fontsize=12)
    ax.set_ylabel('Bulk Entropy $S_b$', fontsize=12)
    ax.set_title(f'Bulk Entropy vs Region Size (Bulk Position = {bulk_position})',
                fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_entropy_components(results, bulk_position, ax=None,
                           plot_mutual_info_only=False, **kwargs):
    """Plot entropy components with region size for a fixed bulk position.

    Shows how various entropies (S_R, S_b, S_{R+b}, etc.) vary with the size
    of the boundary region R for a fixed bulk node position.

    Args:
        results: List of dicts from sweep_bulk_reconstruction()
        bulk_position: Which bulk position to analyze
        ax: Matplotlib axes (creates new figure if None)
        plot_mutual_info_only: If True, only plot I(R:b); if False, plot all components (default False)
        **kwargs: Additional arguments passed to plot

    Returns:
        fig, ax: Matplotlib figure and axes

    Example:
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
        >>> fig, ax = plot_entropy_components(results, bulk_position=10)
        >>> plt.show()

    Notes:
        - Plots S_R (region), S_b (bulk), S_{R+b} (joint), and I(R:b)
        - Useful for comparing to holographic predictions
        - S_R should grow as region size increases
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Filter results for this bulk position
    bulk_results = [r for r in results
                    if r['bulk_position'] == bulk_position]

    if len(bulk_results) == 0:
        raise ValueError(f"No results found for bulk_position={bulk_position}")

    # Sort by region size
    bulk_results = sorted(bulk_results, key=lambda r: r['region_size'])

    # Extract data
    region_sizes = [r['region_size'] for r in bulk_results]
    S_R = [r['S_R'] for r in bulk_results]
    S_b = [r['S_b'] for r in bulk_results]
    S_R_b = [r['S_R_b'] for r in bulk_results]
    MI_R_b = [r['MI_R_b'] for r in bulk_results]

    # Plot
    if not plot_mutual_info_only:
        ax.plot(region_sizes, S_R, 'o-', label='$S_R$ (region)',
               linewidth=2, markersize=6)
        ax.plot(region_sizes, S_b, 's--', label='$S_b$ (bulk)',
               linewidth=2, markersize=6)
        ax.plot(region_sizes, S_R_b, '^-.', label='$S_{R+b}$ (joint)',
               linewidth=2, markersize=6)
        ax.plot(region_sizes, MI_R_b, 'd-', label='$I(R:b)$ (mutual info)',
               linewidth=2.5, markersize=6, color='red')
    else:
        ax.plot(region_sizes, MI_R_b, 'o-', label='$I(R:b)$',
               linewidth=2.5, markersize=8, color='red')

    # Labels and formatting
    ax.set_xlabel('Region Size |R|', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title(f'Entropy Components (Bulk Position = {bulk_position})',
                fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_bulk_entropy_comparison(results, region_size, ax=None,
                                  normalize=False, **kwargs):
    """Compare bulk node entropies across different positions.

    Shows S_b (entropy of bulk node alone) for all bulk positions,
    useful for understanding which bulk nodes are more entangled.

    Args:
        results: List of dicts from sweep_bulk_reconstruction()
        region_size: Which region size to use for comparison
        ax: Matplotlib axes (creates new figure if None)
        normalize: If True, normalize mutual info by S_b (default False)
        **kwargs: Additional arguments passed to bar plot

    Returns:
        fig, ax: Matplotlib figure and axes

    Example:
        >>> results = sweep_bulk_reconstruction(G, bulk_pos, region_sizes)
        >>> fig, ax = plot_bulk_entropy_comparison(results, region_size=4)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Filter for this region size
    region_results = [r for r in results if r['region_size'] == region_size]

    if len(region_results) == 0:
        raise ValueError(f"No results found for region_size={region_size}")

    # Sort by bulk position
    region_results = sorted(region_results, key=lambda r: r['bulk_position'])

    # Extract data
    bulk_positions = [r['bulk_position'] for r in region_results]
    S_b_values = [r['S_b'] for r in region_results]
    MI_values = [r['MI_R_b'] for r in region_results]

    if normalize:
        # Normalized mutual info: I(R:b) / S_b (should be ≤ 1)
        plot_values = [MI / S_b if S_b > 1e-10 else 0.0
                      for MI, S_b in zip(MI_values, S_b_values)]
        ylabel = 'Normalized Mutual Info: $I(R:b) / S_b$'
        title_suffix = '(Normalized)'
    else:
        plot_values = MI_values
        ylabel = 'Mutual Information $I(R:b)$'
        title_suffix = ''

    # Bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(bulk_positions)))
    ax.bar(range(len(bulk_positions)), plot_values, color=colors, **kwargs)

    # Formatting
    ax.set_xticks(range(len(bulk_positions)))
    ax.set_xticklabels(bulk_positions, rotation=45 if len(bulk_positions) > 20 else 0)
    ax.set_xlabel('Bulk Position', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Bulk-Boundary Correlations (|R| = {region_size}) {title_suffix}',
                fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    return fig, ax

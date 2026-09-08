from copy import  deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx

import graph2grav.technical_helpers as technical_helpers
import graph2grav.graphs as graphs
from graph2grav.graphs.standard import make_tree_periodic
import graph2grav.gaussian as gaussian
import graph2grav.analytic_entropies as analytic_entropies
from graph2grav.analysis import Analysis, format_with_uncertainty
from graph2grav import poincare

from matplotlib.ticker import FuncFormatter, ScalarFormatter

config_to_use = None

def set_config_for_introduction(config):
    global config_to_use
    config_to_use = config


def _figure_1_params(config, figure_params=None):
    if figure_params is not None:
        return figure_params
    if "figure-1" not in config:
        raise ValueError("Figure 1 parameters now live in 01_introduction.ipynb; pass figure_params.")
    figure_config = config["figure-1"]
    return {
        "graph_depth": figure_config["graph"]["depth"],
        "coupling_time": figure_config["graph"]["coupling_time"],
        "squeezing": figure_config["graph"]["squeezing"],
        "region_size_for_renyi_entropies": figure_config["entropies"]["region_size_for_renyi_entropies"],
    }


def plot_protocol(axs, use_poincare_layout=True, label=False, config=None, figure_params=None):
    """This function plots the quench-and-measure protocol."""

    if config is None:
        global config_to_use
        config = config_to_use
    
    assert config is not None, "You must provide a config either as an argument or by calling set_config_for_introduction first."
    figure_params = _figure_1_params(config, figure_params)

    axs = np.array(axs)
    axs_flat = axs.flatten()

    if len(axs_flat) == 2:
        show_displacement_and_final = False
    elif len(axs_flat) == 4:
        show_displacement_and_final = True
    else:
        raise ValueError("axs must have either 2 or 4 subplots.")

    for ax in axs_flat:

        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    
    # axs_flat[-1].set_facecolor("#EEFFE7")   

    extra = 0

    if show_displacement_and_final:
        border = patches.Rectangle(
            (-extra, -extra), 1 + 2*extra, 1 + 2*extra,
            transform=axs_flat[-1].transAxes,
            facecolor="none",
            edgecolor="#42DD0086",
            linewidth=1.75,
            zorder=10,
            clip_on=False
        )
        axs_flat[-1].add_patch(border)

    depth = figure_params["graph_depth"]

    ctree: nx.Graph = graphs.crosslinked_tree(depth) # networkx graph object
    ptree = make_tree_periodic(ctree, remove_nodes=False)

    post_measurement_graph = deepcopy(ptree)

    for edge in post_measurement_graph.edges:
        if ptree.nodes[edge[0]]["is_ancilla"] or ptree.nodes[edge[1]]["is_ancilla"]:
            post_measurement_graph.remove_edge(*edge)

    # kamada kawai layout
    if use_poincare_layout:
        hyperbolic_spacing = 0.5 # config["graph"]["nodes"]["hyperbolic_spacing"]
        pos = {}
        max_depth = max([ptree.nodes[n]["depth"] for n in ptree.nodes])

        for n in ptree.nodes:
            euclidean_r = hyperbolic_spacing * ptree.nodes[n]["depth"]
            max_euclidean_r = hyperbolic_spacing * max_depth


            r = np.tanh(euclidean_r / 2)  / np.tanh(max_euclidean_r / 2)

            # print(f"Node {n}: depth={ptree.nodes[n]['depth']}, euclidean_r={euclidean_r}, r={r}, number of nodes at depth={2 ** ptree.nodes[n]['depth']}")

            number_of_nodes_at_depth = 2 ** ptree.nodes[n]["depth"]
            angular_spacing = 2 * np.pi / number_of_nodes_at_depth
            theta = 2 * np.pi * graphs.standard.tree_node_rowindex(n) / number_of_nodes_at_depth + angular_spacing / 2
            if ptree.nodes[n]["depth"] == max_depth:
                r = r * 1.1
            pos[n] = (r * np.cos(theta), r * np.sin(theta))


    else:
        pos = nx.kamada_kawai_layout(ptree)


    inner_premeasure = [
        config["graph"]["nodes"]["boundary"]["inner"]["color"]
        if ptree.nodes[n]["is_boundary"]
        else config["graph"]["nodes"]["ancilla"]["inner"]["color"]
        for n in ptree.nodes
    ]
    outer_premeasure = [
        config["graph"]["nodes"]["boundary"]["outer"]["color"]
        if ptree.nodes[n]["is_boundary"]
        else config["graph"]["nodes"]["ancilla"]["outer"]["color"]
        for n in ptree.nodes
    ]

    inner_postmeasure = [
        config["graph"]["nodes"]["boundary"]["inner"]["color"]
        if ptree.nodes[n]["is_boundary"]
        else config["graph"]["nodes"]["post_measurement"]["inner"]["color"]
        for n in ptree.nodes
    ]

    outer_postmeasure = [
        config["graph"]["nodes"]["boundary"]["outer"]["color"]
        if ptree.nodes[n]["is_boundary"]
        else config["graph"]["nodes"]["post_measurement"]["outer"]["color"]
        for n in ptree.nodes
    ]

    node_names_by_set_to_label = {
        "all": None,
        "just_boundary": {n: ptree.nodes[n]["position_in_boundary"] if ptree.nodes[n]["is_boundary"] else "" for n in ptree.nodes},
        "none": {n: "" for n in ptree.nodes}
    }

    post_measurement_plot_indices = [2, 3]

    for n, ax in enumerate(axs_flat):



        if n in post_measurement_plot_indices:
            graph_to_draw = post_measurement_graph
            outer_colors = outer_postmeasure
            inner_colors = inner_postmeasure
        else:
            graph_to_draw = post_measurement_graph if n == 1 else ptree # now we might not draw the edges in the measurement graph
            outer_colors = outer_premeasure
            inner_colors = inner_premeasure



        nx.draw_networkx_nodes(graph_to_draw, pos , ax=ax, node_color=outer_colors, node_size=config["graph"]["nodes"]["outer"]["size"]/1.5)
        nx.draw_networkx_nodes(graph_to_draw, pos , ax=ax, node_color=inner_colors, node_size=config["graph"]["nodes"]["inner"]["size"]/1.5)

        nx.draw_networkx_edges(graph_to_draw, pos, ax=ax, edgelist=graph_to_draw.edges, width=1, alpha=0.7)
        nx.draw_networkx_labels(graph_to_draw, pos, ax=ax, labels=node_names_by_set_to_label[config["graph"]["which_nodes_to_label"]], font_color="black")

    # above and to the left
    # x = 0.15
    # y = 0.85

    # centered above each
    x = 0.5
    y = 0.95

    if label:
        axs_flat[0].set_title("Quench", x=x, y=y)
        axs_flat[1].set_title("Measure Bulk", x=x, y=y)
        if show_displacement_and_final:
            axs_flat[2].set_title("Displace", x=x, y=y)
            axs_flat[3].set_title("Final State", x=x, y=y, color="#31A200")


    circle = plt.Circle( # pyright: ignore
        (0, 0), config["operations"]["measure"]["size"], color=config["operations"]["measure"]["color"], fill=True, linewidth=0,
        alpha=config["operations"]["measure"]["alpha"],
        zorder = 10
    )
    axs_flat[1].add_artist(circle)

    # draw measurement circle on the Measure subplot

    if show_displacement_and_final:

        # draw displacement square on the Displace subplot

        displace_ax = axs[1, 0]

        for node in post_measurement_graph.nodes:
            if post_measurement_graph.nodes[node]["is_boundary"]:
                x, y = pos[node]

                size = config["operations"]["displace"]["size"]
                circle = mpl.patches.Circle( # pyright: ignore
                    (x, y), size/2,
                    color=config["operations"]["displace"]["color"],
                    alpha=config["operations"]["displace"]["alpha"],
                    linewidth=0, fill=True,
                    zorder=10
                )
                displace_ax.add_patch(circle)

    # Draw arrows between subplots in Z pattern: (0,0) -> (0,1) -> (1,0) -> (1,1)
    fig = axs_flat[0].get_figure()
    arrow_style = mpl.patches.ArrowStyle("-|>", head_length=0.3, head_width=0.2)
    arrow_color = "#000000"
    arrow_lw = 2.5

    # Arrow connections: (from_ax, to_ax, from_side, to_side)
    # from_side/to_side: 'right', 'left', 'top', 'bottom' or corner positions
    connections = [
        (axs_flat[0], axs_flat[1], (1.0, 0.5), (0.0, 0.5)),   # right of (0,0) to left of (0,1)
    ]
    if show_displacement_and_final:
        connections +=[
            (axs_flat[1], axs_flat[2], (0.05, 0.15), (0.95, 0.85)),   # bottom of (0,1) to top of (1,0)
            (axs_flat[2], axs_flat[3], (1.0, 0.5), (0.0, 0.5)),   # right of (1,0) to left of (1,1)
        ]

    for from_ax, to_ax, from_pos, to_pos in connections:
        arrow = mpl.patches.ConnectionPatch(
            xyA=from_pos, xyB=to_pos,
            coordsA="axes fraction", coordsB="axes fraction",
            axesA=from_ax, axesB=to_ax,
            arrowstyle=arrow_style,
            color=arrow_color,
            linewidth=arrow_lw,
            zorder=100,
            capstyle="butt",
            joinstyle="miter"
        )
        fig.add_artist(arrow)
    
    for ax in axs_flat:
        # set xlim and ylim
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

# Analysis class is now imported from graph2grav.analysis

def plot_entanglement_and_renyi_entropies(
    axs: list[plt.Axes],
    show_squeezing_instead=True,
    config=None,
    export_csv: str | None = None,
    figure_params=None,
): # pyright: ignore
    """Plot entanglement and Renyi entropies.

    Args:
        axs: List of matplotlib axes [entropy_ax, renyi_ax]
        show_squeezing_instead: If True, vary squeezing; else vary coupling time
        config: Configuration dictionary
        export_csv: If provided, export data to this CSV file path
    """
    if config is None:
        global config_to_use

        config = config_to_use

    assert config is not None, "You must provide a config either as an argument or by calling set_config_for_introduction first."
    figure_params = _figure_1_params(config, figure_params)

    # Data collection for CSV export
    csv_data = {
        'entanglement_entropy': {},
        'renyi_entropies': []
    }

    depth = figure_params["graph_depth"]

    ctree: nx.Graph = graphs.crosslinked_tree(depth) # networkx graph object
    ptree = make_tree_periodic(ctree, remove_nodes=False)


    axs_flat = np.array(axs).flatten()

    analysis = Analysis(
        ptree,
        couplingtime=figure_params["coupling_time"],
        global_presqueeze=figure_params["squeezing"],
        config=config,
    )

    print("Using squeezing of mu = ", figure_params["squeezing"])


    analysis.fit_central_charge()
    analysis.plot_entanglement_entropy(axs_flat[0], include_ends=True, zorder=5, markersize=4.5, alpha=0.8, use_simple_markers=True, config=config)

    # Print central charge with uncertainty
    print(f"Fitted central charge: c = {format_with_uncertainty(analysis.central_charge, analysis.central_charge_stderr)}")
    print(f"CFT entropy offset: {format_with_uncertainty(analysis.cft_entropy_offset, analysis.cft_entropy_offset_stderr)}")

    # Collect entanglement entropy data for CSV
    csv_data['entanglement_entropy'] = {
        'region_sizes': list(analysis.region_sizes),
        'entropies': list(analysis.boundary_entropies),
        'central_charge_fit': analysis.central_charge,
        'central_charge_stderr': analysis.central_charge_stderr,
        'offset_fit': analysis.cft_entropy_offset,
        'offset_stderr': analysis.cft_entropy_offset_stderr,
        'boundary_length': analysis.boundary_length,
        'coupling_time': figure_params["coupling_time"],
        'squeezing': figure_params["squeezing"],
    }

    target_region_size = figure_params["region_size_for_renyi_entropies"]

    fine_lengths = np.linspace(1, analysis.boundary_length-1, 1000)

    # plot entanglement entropy
    axs_flat[0].plot(
        fine_lengths,
        analytic_entropies.periodic_CFT_entropy(fine_lengths, analysis.boundary_length, central_charge=analysis.central_charge, offset=analysis.cft_entropy_offset),
        "--",
        # label=rf"CFT formula (fit: $c = {analysis.central_charge:.2f}$, offset: {analysis.cft_entropy_offset:.2f})",
        color="black", # config["graph"]["nodes"]["selected_boundary"]["cft_prediction"]["color"],

        zorder=2
    )

    axs_flat[0].set_ylabel(r"$S$")
    axs_flat[0].set_xlabel(r"$\ell$")
    axs_flat[0].set_xticks([0, 4, 8, 12, 16])
    # axs_flat[0].legend()


    # plot renyi entropies

    ctree: nx.Graph = graphs.crosslinked_tree(depth) # networkx graph object
    ptree = make_tree_periodic(ctree, remove_nodes=False)


    geometric_mean_coupling_time = figure_params["coupling_time"]
    geometric_mean_squeezing = figure_params["squeezing"]
    factor = 5

    if show_squeezing_instead:
        min_squeezing=  geometric_mean_squeezing / factor
        max_squeezing = geometric_mean_squeezing * factor
        squeezing_values = np.geomspace(min_squeezing, max_squeezing, 5)
        coupling_times = np.ones_like(squeezing_values)

        value_name = r"Squeezing Factor $\mu$"
        values = squeezing_values

    else: # otherwise do coupling times
        min_coupling_time = geometric_mean_coupling_time / factor
        max_coupling_time = geometric_mean_coupling_time * factor
        coupling_times = np.geomspace(min_coupling_time, max_coupling_time, 5)
        squeezing_values = np.ones_like(coupling_times)

        value_name = "Coupling Time"
        values = coupling_times
    
    if show_squeezing_instead:
        print(f"Squeezing values for Renyi entropy plot: {squeezing_values}")

    # cmap = plt.cm.viridis
    colors = ["#2d1e5e", "#e52d67", "#f3d656"]

    # 2. Create the colormap
    # 'from_list' automatically interpolates these colors evenly
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_pink_center", colors)



    for coupling_time, squeezing in zip(coupling_times, squeezing_values):

        analysis = Analysis(ptree, couplingtime=coupling_time, global_presqueeze=squeezing)
        alphas = np.linspace(0.1, 5, 100)
        renyi_entropies = np.array(analysis.get_renyi_entropies_across_alpha(alphas, region_size=target_region_size))

        closest_to_1_index = np.argmin(np.abs(alphas - 1))

        normalized_renyi_entropies = renyi_entropies / renyi_entropies[closest_to_1_index]

        # Collect Renyi entropy data for CSV
        csv_data['renyi_entropies'].append({
            'coupling_time': coupling_time,
            'squeezing': squeezing,
            'region_size': target_region_size,
            'alphas': alphas.tolist(),
            'renyi_entropies': renyi_entropies.tolist(),
            'normalized_renyi_entropies': normalized_renyi_entropies.tolist(),
            'S_EE': renyi_entropies[closest_to_1_index],
        })

        if show_squeezing_instead:
            color = cmap((np.log(squeezing) - np.log(min_squeezing)) / (np.log(max_squeezing) - np.log(min_squeezing)))
        else:
            color = cmap((np.log(coupling_time) - np.log(min_coupling_time)) / (np.log(max_coupling_time) - np.log(min_coupling_time)))

        axs_flat[1].plot(
            alphas,
            normalized_renyi_entropies,
            label=r"Rényi-$\alpha$ Entropy (from simulation)",
            color=color,
            alpha=1
        )
    

    # We are no longer highlighting these point
    # # plot a marker for entanglement entropy
    # axs_flat[1].plot(1, analysis.get_renyi_entropies_across_alpha([1], region_size=target_region_size)[0], "s", color="tab:green", label=r"Entanglement Entropy ($\alpha = 1$)", markersize=8, alpha=1)
    # # plot a marker for Rényi-2 entropy
    # axs_flat[1].plot(2, analysis.get_renyi_entropies_across_alpha([2], region_size=target_region_size)[0], "^", color="tab:purple", label=r"Rényi-2 Entropy ($\alpha = 2$)", markersize=8, alpha=1
    # axs_flat[1].legend()

    axs_flat[1].set_xlabel(r"$\alpha$")
    axs_flat[1].set_ylabel(r"$S_\alpha / S$")


    # we are normalizing to 1
    renyi = 1 # analysis.get_renyi_entropies_across_alpha([1], region_size=target_region_size)[0]

    axs_flat[1].plot(alphas, (1 + 1 / alphas) * renyi / 2, label="Guideline: $S_\\alpha = (1 + 1/\\alpha) S$", color="black", linestyle="--")

    axs_flat[0].set_xlabel(r"$\ell$")
    axs_flat[1].set_ylim(0, 2)

    # create a logarithmic cbar for axs_flat[1] showing coupling time
    if show_squeezing_instead:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(vmin=min_squeezing, vmax=max_squeezing))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(vmin=min_coupling_time, vmax=max_coupling_time))

    cbar = plt.colorbar(sm, ax=axs_flat[1], pad=0.05, shrink=0.7, aspect=10)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))  

    if show_squeezing_instead:
        cbar.ax.set_title(r"$\mu$")
    else:
        cbar.set_label("Coupling Time")

    # remove small ticks on colorbar
    cbar.ax.minorticks_off()

    # fig.tight_layout()

    # Export to CSV if requested
    if export_csv:
        import csv
        import os

        # Create directory if needed
        os.makedirs(os.path.dirname(export_csv) if os.path.dirname(export_csv) else '.', exist_ok=True)

        # Export entanglement entropy data
        ee_path = export_csv.replace('.csv', '_entanglement_entropy.csv')
        with open(ee_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# Entanglement Entropy Data'])
            writer.writerow([f'# boundary_length={csv_data["entanglement_entropy"]["boundary_length"]}'])
            writer.writerow([f'# coupling_time={csv_data["entanglement_entropy"]["coupling_time"]}'])
            writer.writerow([f'# squeezing={csv_data["entanglement_entropy"]["squeezing"]}'])
            writer.writerow([f'# central_charge_fit={csv_data["entanglement_entropy"]["central_charge_fit"]}'])
            writer.writerow([f'# central_charge_stderr={csv_data["entanglement_entropy"]["central_charge_stderr"]}'])
            writer.writerow([f'# offset_fit={csv_data["entanglement_entropy"]["offset_fit"]}'])
            writer.writerow([f'# offset_stderr={csv_data["entanglement_entropy"]["offset_stderr"]}'])
            writer.writerow(['region_size', 'entanglement_entropy'])
            for L, S in zip(csv_data['entanglement_entropy']['region_sizes'],
                           csv_data['entanglement_entropy']['entropies']):
                writer.writerow([L, S])
        print(f"Exported entanglement entropy to: {ee_path}")

        # Export Renyi entropy data
        renyi_path = export_csv.replace('.csv', '_renyi_entropy.csv')
        with open(renyi_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# Renyi Entropy Data'])
            writer.writerow([f'# region_size={target_region_size}'])
            # Header with squeezing values
            header = ['alpha'] + [f'squeezing={d["squeezing"]:.4f}' for d in csv_data['renyi_entropies']]
            header_norm = ['alpha'] + [f'squeezing={d["squeezing"]:.4f}_normalized' for d in csv_data['renyi_entropies']]
            writer.writerow(header)
            # Data rows
            alphas = csv_data['renyi_entropies'][0]['alphas']
            for i, alpha in enumerate(alphas):
                row = [alpha] + [d['renyi_entropies'][i] for d in csv_data['renyi_entropies']]
                writer.writerow(row)
        print(f"Exported Renyi entropy to: {renyi_path}")

        # Also export normalized Renyi entropies
        renyi_norm_path = export_csv.replace('.csv', '_renyi_entropy_normalized.csv')
        with open(renyi_norm_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# Normalized Renyi Entropy Data (S_alpha / S_EE)'])
            writer.writerow([f'# region_size={target_region_size}'])
            header = ['alpha'] + [f'squeezing={d["squeezing"]:.4f}' for d in csv_data['renyi_entropies']]
            writer.writerow(header)
            alphas = csv_data['renyi_entropies'][0]['alphas']
            for i, alpha in enumerate(alphas):
                row = [alpha] + [d['normalized_renyi_entropies'][i] for d in csv_data['renyi_entropies']]
                writer.writerow(row)
        print(f"Exported normalized Renyi entropy to: {renyi_norm_path}")
    
    return analysis


# =============================================================================
# Poincaré Geometry Functions (delegated to graph2grav.poincare module)
# =============================================================================

# Re-export functions from poincare module for backwards compatibility
get_poincare_geodesic_points = poincare.geodesic_arc_points_boundary
draw_poincare_geodesic = poincare.draw_geodesic
compute_poincare_circumradius = poincare.compute_poincare_circumradius
get_geodesic_circle = poincare.get_geodesic_circle
reflect_point_geodesic = poincare.reflect_across_geodesic
reflect_point_diameter = poincare.reflect_across_diameter
order_vertices_ccw = poincare.order_vertices_ccw
apply_mobius = poincare.apply_mobius
get_reflection_matrix = poincare.get_reflection_matrix
generate_poincare_tessellation_with_transforms = poincare.generate_tessellation_with_transforms
generate_poincare_tessellation = poincare.generate_tessellation
get_geodesic_arc_points = poincare.geodesic_arc_points
extend_geodesic_to_boundary = poincare.extend_geodesic_to_boundary


def draw_boundary_arc(ax: plt.Axes, z1: complex, z2: complex, n_points: int = 50,  # pyright: ignore
                      offset: tuple[float, float] = (0.0, 0.0), **kwargs):
    """Draw an arc along the unit circle from z1 to z2 (shortest path)."""
    ox, oy = offset
    angle1 = np.angle(z1)
    angle2 = np.angle(z2)

    # Go the short way
    diff = angle2 - angle1
    if diff > np.pi:
        angle2 -= 2 * np.pi
    elif diff < -np.pi:
        angle2 += 2 * np.pi

    angles = np.linspace(angle1, angle2, n_points)
    xs = np.cos(angles) + ox
    ys = np.sin(angles) + oy
    ax.plot(xs, ys, **kwargs)


def draw_filled_boundary_wedge(ax: plt.Axes, z1_boundary: complex, z2_boundary: complex,  # pyright: ignore
                                z1_inner: complex, z2_inner: complex, n_points: int = 30,
                                offset: tuple[float, float] = (0.0, 0.0), **kwargs):
    """Draw a filled wedge between boundary arc and a geodesic arc."""
    ox, oy = offset

    angle1 = np.angle(z1_boundary)
    angle2 = np.angle(z2_boundary)
    diff = angle2 - angle1
    if diff > np.pi:
        angle2 -= 2 * np.pi
    elif diff < -np.pi:
        angle2 += 2 * np.pi

    boundary_angles = np.linspace(angle1, angle2, n_points)
    boundary_xs = np.cos(boundary_angles)
    boundary_ys = np.sin(boundary_angles)

    inner_xs, inner_ys = get_geodesic_arc_points(z2_inner, z1_inner, n_points=n_points)

    all_xs = list(boundary_xs) + list(inner_xs)
    all_ys = list(boundary_ys) + list(inner_ys)

    all_xs = [x + ox for x in all_xs]
    all_ys = [y + oy for y in all_ys]

    ax.fill(all_xs, all_ys, **kwargs)


def draw_poincare_tiling(ax: plt.Axes, p: int = 4, q: int = 6, max_depth: int = 5,  # pyright: ignore
                          color0: str = "white", color1: str = "lightgray",
                          edge_color: str = "gray", edge_alpha: float = 0.3,
                          offset: tuple[float, float] = (0.0, 0.0),
                          points_per_edge: int = 30,
                          extend_to_boundary: bool = True):
    """Draw a {p, q} hyperbolic tessellation on the Poincaré disk."""
    poincare.draw_tessellation(
        ax, p=p, q=q, max_depth=max_depth,
        color0=color0, color1=color1,
        edge_color=edge_color, edge_alpha=edge_alpha,
        offset=offset, points_per_edge=points_per_edge,
        extend_to_boundary=extend_to_boundary
    )


# Use poincare module for is_between_short_arc
is_between_short_arc = poincare.is_between_short_arc

def draw_ryu_takayanagi(
        ax: plt.Axes, theta1: float = -np.pi/4.5, theta2: float = np.pi/4.5,
        bulk_style: str = "circles", offset: tuple[float, float] = (0.0, 0.0),
        show_bulk_nodes=False, config=None, tiling_p: int = 4, tiling_q: int = 6,
        tiling_depth: int = 5, show_ryu_takayanagi=True): # pyright: ignore
    """ Draws a schematic of the Ryu-Takayanagi surface on the Poincare disk.

    Args:
        ax: Matplotlib axes to draw on
        theta1: Angle of first boundary endpoint (radians)
        theta2: Angle of second boundary endpoint (radians)
        bulk_style: "circles" for concentric circles, "geodesics" for bulk geodesics,
                   "tiling" for hyperbolic tessellation, "none" for no bulk structure
        offset: (x_offset, y_offset) tuple to shift the entire diagram
        config: Configuration dictionary (optional)
        tiling_p: Number of sides per polygon for tiling (default 4)
        tiling_q: Number of polygons meeting at each vertex for tiling (default 6)
        tiling_depth: Maximum recursion depth for tiling generation (default 5)
    """
    if config is None:
        global config_to_use

        config = config_to_use

    assert config is not None, "You must provide a config either as an argument or by calling set_config_for_introduction first."

    hyperbolic_spacing = config["graph"]["nodes"]["hyperbolic_spacing"]
    n = np.arange(0, 5)
    r_n = np.tanh(n * hyperbolic_spacing / 2)
    max_log_nodes = n[-1]  # = 4, so 2^4 = 16 boundary nodes

    ox, oy = offset

    # Draw filled disk
    circle = plt.Circle( # pyright: ignore
        (ox, oy), 1, color="lightgray", fill=None, linewidth=1.5, alpha=0.5,
        zorder = 1
    )
    ax.add_artist(circle)

    # Draw boundary circle
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = np.cos(theta) + ox
    y = np.sin(theta) + oy
    ax.plot(x, y, color="black", linewidth=1.5, zorder=5)

    if show_arc := False:

        # draw circular arc outside main circle to show the boundary region
        # use square endcaps
        angular_spacing_at_max_nodes = 2 * np.pi / (2 ** max_log_nodes)
        theta_outside = np.linspace(theta1, theta2, 100)

        boundary_arc_radius = 1.2
        x_outside = boundary_arc_radius * np.cos(theta_outside) + ox
        y_outside = boundary_arc_radius * np.sin(theta_outside) + oy
        ax.plot(x_outside, y_outside, # color="black",
                linewidth=1.5, zorder=5, solid_capstyle="projecting",
                color=config["graph"]["nodes"]["selected_boundary"]["outer"]["color"])
        
        # add tick marks along the arc
        tick_thetas = np.arange(theta1, theta2, angular_spacing_at_max_nodes)
        tick_x = boundary_arc_radius * np.cos(tick_thetas) + ox
        tick_y = boundary_arc_radius * np.sin(tick_thetas) + oy

        from matplotlib.collections import LineCollection


        tick_len = 0.2  # length in data units; tweak to taste
        half = tick_len / 2

        segs = []
        for t in tick_thetas:
            cx = boundary_arc_radius * np.cos(t) + ox
            cy = boundary_arc_radius * np.sin(t) + oy

            # radial direction (perpendicular to circular arc)
            dx = np.cos(t)
            dy = np.sin(t)

            segs.append([
                (cx - half * dx, cy - half * dy),
                (cx + half * dx, cy + half * dy),
            ])

        lc = LineCollection(
            segs,
            colors=config["graph"]["nodes"]["selected_boundary"]["outer"]["color"],
            linewidths=1.5,
            zorder=6,
        )
        ax.add_collection(lc)

        # add explicit square markers at the two ends
        ax.plot(
            [x_outside[0], x_outside[-1]],
            [y_outside[0], y_outside[-1]],
            linestyle="None", marker="s",
            # color="black",
            zorder=6,
            markersize=6,  # tweak to taste
            markeredgewidth=1.5,
            markeredgecolor=config["graph"]["nodes"]["selected_boundary"]["outer"]["color"],
            color= config["graph"]["nodes"]["selected_boundary"]["inner"]["color"]
        )


    if show_ryu_takayanagi:
        # Draw the Ryu-Takayanagi geodesic
        rt_x, rt_y = get_poincare_geodesic_points(theta1, theta2, n_points=100)
        ax.plot(rt_x + ox, rt_y + oy,
            color=config["graph"]["nodes"]["selected_boundary"]["ryu_takayanagi"]["color"],
            linewidth=2, zorder=10
        )

        # Fill the entanglement wedge (region between geodesic and boundary)
        geodesic_x, geodesic_y = get_poincare_geodesic_points(theta1, theta2, n_points=100)
        # Boundary arc from theta2 back to theta1 (closing the region)
        boundary_angles = np.linspace(theta2, theta1, 100)
        boundary_x = np.cos(boundary_angles)
        boundary_y = np.sin(boundary_angles)
        # Combine geodesic and boundary arc into closed polygon
        wedge_x = np.concatenate([geodesic_x, boundary_x]) + ox
        wedge_y = np.concatenate([geodesic_y, boundary_y]) + oy
        ax.fill(wedge_x, wedge_y, color="pink", alpha=0.5, zorder=0)


    # Draw bulk structure (circles or geodesics)
    if bulk_style == "circles":
        for log_nodes, radius in zip(n, r_n):
            circle = plt.Circle( # pyright: ignore
                (ox, oy), radius, color="black", fill=False, linewidth=1, alpha=0.3,
                zorder = 2
            )
            ax.add_artist(circle)
    elif bulk_style == "geodesics":
        # Draw geodesics at midpoints between boundary nodes
        # Boundary nodes are at angles: (k + 0.5) * 2π / 2^max_log_nodes for k = 0, 1, ..., 2^max_log_nodes - 1
        # Midpoints are at: k * 2π / 2^max_log_nodes for k = 0, 1, ..., 2^max_log_nodes - 1
        num_boundary = 2 ** max_log_nodes
        midpoint_angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)

        # Draw geodesic between every pair of midpoints
        for i in range(num_boundary):
            for j in range(i + 1, num_boundary):
                gx, gy = get_poincare_geodesic_points(midpoint_angles[i], midpoint_angles[j])
                ax.plot(gx + ox, gy + oy,
                    color=config["graph"]["nodes"]["boundary"]["other_geodesics"]["color"], linewidth=1, alpha=0.15, zorder=2)
    elif bulk_style == "tiling":
        # Draw hyperbolic tessellation with alternating colors
        draw_poincare_tiling(ax, p=tiling_p, q=tiling_q, max_depth=tiling_depth,
                            extend_to_boundary=False,
                             color0="white", color1="lightgray",
                             edge_color="gray", edge_alpha=0.3,
                             offset=offset)
    elif bulk_style == "none":
        pass
    else:
        raise ValueError(f"Unknown bulk_style: {bulk_style}")

    # Draw nodes at each layer
    for log_nodes, radius in zip(n, r_n):
        num_nodes = 2**log_nodes
        angular_spacing = 2 * np.pi / num_nodes
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False) + angular_spacing / 2

        is_last_layer = (log_nodes == n[-1])
        if is_last_layer:
            radius = 1 # jank but I want it here
            x_nodes = radius * np.cos(angles) + ox
            y_nodes = radius * np.sin(angles) + oy
            # For tiling and geodesics styles, don't highlight selected region
            inside_selected_region = np.zeros(len(angles), dtype=bool) if not show_ryu_takayanagi else is_between_short_arc(angles, theta1, theta2)
            ax.plot(
                x_nodes[inside_selected_region], y_nodes[inside_selected_region], 'o', color=config["graph"]["nodes"]["selected_boundary"]["outer"]["color"],
                markersize=5, alpha=1, zorder=10
            )
            ax.plot(
                x_nodes[inside_selected_region], y_nodes[inside_selected_region], 'o', color=config["graph"]["nodes"]["selected_boundary"]["inner"]["color"],
                markersize=3, alpha=1, zorder=20
            )

            ax.plot(
                x_nodes[~inside_selected_region], y_nodes[~inside_selected_region], 'o', color=config["graph"]["nodes"]["boundary"]["outer"]["color"],
                markersize=5, alpha=1, zorder=10
            )
            ax.plot(
                x_nodes[~inside_selected_region], y_nodes[~inside_selected_region], 'o', color=config["graph"]["nodes"]["boundary"]["inner"]["color"],
                markersize=3, alpha=1, zorder=20
            )
        elif show_bulk_nodes:
            x_nodes = radius * np.cos(angles) + ox
            y_nodes = radius * np.sin(angles) + oy
            # Skip bulk nodes for tiling style since tessellation provides visual structure
            ax.plot(
                x_nodes, y_nodes, 'o', color=config["graph"]["nodes"]["post_measurement"]["outer"]["color"],
                markersize=5, alpha=1, zorder=10
            )
            ax.plot(
                x_nodes, y_nodes, 'o', color=config["graph"]["nodes"]["post_measurement"]["inner"]["color"],
                markersize=3, alpha=1, zorder=20
            )
    
    ax.set_aspect('equal')
    ax.axis('off')

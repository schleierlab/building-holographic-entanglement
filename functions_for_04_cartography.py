"""Helper functions for 05_decorated_cartography.ipynb visualization."""

import numpy as np
import colorsys
import matplotlib.colors as mcolors

from graph2grav import poincare

# Re-export functions from poincare module for backwards compatibility
poincare_geodesic_arc = poincare.geodesic_arc_points
hyperbolic_distance = poincare.hyperbolic_distance
geodesic_length_boundary_points = poincare.geodesic_length_boundary_points


def draw_graph_edges_with_decoration(ax, graph, pos, edge_color='gray', edge_alpha=0.3,
                                      edge_linewidth=1.0, dot_color='black', dot_size=10,
                                      zorder=1):
    """Draw graph edges with small black dots at midpoints to indicate decoration."""
    midpoints_x = []
    midpoints_y = []

    for u, v in graph.edges():
        if u in pos and v in pos:
            p1 = pos[u]
            p2 = pos[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                   color=edge_color, alpha=edge_alpha,
                   linewidth=edge_linewidth, zorder=zorder)
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            midpoints_x.append(mid_x)
            midpoints_y.append(mid_y)

    if midpoints_x:
        ax.scatter(midpoints_x, midpoints_y, c=dot_color, s=dot_size,
                  zorder=zorder+1, edgecolors='none')


def fill_entanglement_wedge_discontinuous(ax, boundary_nodes, regions, rt_config,
                                          fill_color='pink', alpha=0.5, zorder=0):
    """Fill the entanglement wedge for discontinuous regions."""
    if len(regions) == 1:
        theta1, theta2 = regions[0]
        z1 = complex(np.cos(theta1), np.sin(theta1))
        z2 = complex(np.cos(theta2), np.sin(theta2))
        geodesic_x, geodesic_y = poincare_geodesic_arc(z1, z2, n_points=100)
        diff = theta1 - theta2
        while diff > 0:
            diff -= 2 * np.pi
        while diff < -2 * np.pi:
            diff += 2 * np.pi
        boundary_angles = np.linspace(theta2, theta2 + diff, 100)
        wedge_x = np.concatenate([geodesic_x, np.cos(boundary_angles)])
        wedge_y = np.concatenate([geodesic_y, np.sin(boundary_angles)])
        ax.fill(wedge_x, wedge_y, color=fill_color, alpha=alpha, zorder=zorder)

    elif len(regions) == 2:
        (theta1_A, theta2_A) = regions[0]
        (theta1_B, theta2_B) = regions[1]

        if rt_config == 'connect_regions':
            z_A1, z_A2 = complex(np.cos(theta1_A), np.sin(theta1_A)), complex(np.cos(theta2_A), np.sin(theta2_A))
            z_B1, z_B2 = complex(np.cos(theta1_B), np.sin(theta1_B)), complex(np.cos(theta2_B), np.sin(theta2_B))
            geo1_x, geo1_y = poincare_geodesic_arc(z_A2, z_B1, n_points=100)
            geo2_x, geo2_y = poincare_geodesic_arc(z_B2, z_A1, n_points=100)
            diff_A = theta2_A - theta1_A
            while diff_A < 0: diff_A += 2 * np.pi
            while diff_A > 2 * np.pi: diff_A -= 2 * np.pi
            diff_B = theta2_B - theta1_B
            while diff_B < 0: diff_B += 2 * np.pi
            while diff_B > 2 * np.pi: diff_B -= 2 * np.pi
            boundary_A = np.linspace(theta1_A, theta1_A + diff_A, 50)
            boundary_B = np.linspace(theta1_B, theta1_B + diff_B, 50)
            wedge_x = np.concatenate([np.cos(boundary_A), geo1_x, np.cos(boundary_B), geo2_x])
            wedge_y = np.concatenate([np.sin(boundary_A), geo1_y, np.sin(boundary_B), geo2_y])
            ax.fill(wedge_x, wedge_y, color=fill_color, alpha=alpha, zorder=zorder)
        else:
            for (theta1, theta2) in regions:
                z1, z2 = complex(np.cos(theta1), np.sin(theta1)), complex(np.cos(theta2), np.sin(theta2))
                geodesic_x, geodesic_y = poincare_geodesic_arc(z1, z2, n_points=100)
                diff = theta1 - theta2
                while diff > 0: diff -= 2 * np.pi
                while diff < -2 * np.pi: diff += 2 * np.pi
                boundary_angles = np.linspace(theta2, theta2 + diff, 100)
                wedge_x = np.concatenate([geodesic_x, np.cos(boundary_angles)])
                wedge_y = np.concatenate([geodesic_y, np.sin(boundary_angles)])
                ax.fill(wedge_x, wedge_y, color=fill_color, alpha=alpha, zorder=zorder)


def draw_rt_surface_discontinuous(ax, boundary_nodes, regions,
                                  color='cyan', linewidth=3, linestyle='-', alpha=0.9, zorder=10,
                                  debug=False):
    """Draw the RT surface for discontinuous regions, choosing minimal length configuration.

    For 2 regions with 4 boundary points [p0, p1, p2, p3] in CCW order where:
      - Region A spans p0 -> p1 (CCW)
      - Region B spans p2 -> p3 (CCW)

    The two RT options are:
      - Option 1 (enclose each): geodesic(p0, p1) + geodesic(p2, p3) -> disconnected wedge
      - Option 2 (connect across): geodesic(p1, p2) + geodesic(p3, p0) -> connected wedge
    """
    if len(regions) == 1:
        theta1, theta2 = regions[0]
        z1, z2 = complex(np.cos(theta1), np.sin(theta1)), complex(np.cos(theta2), np.sin(theta2))
        xs, ys = poincare_geodesic_arc(z1, z2, n_points=100)
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder, solid_capstyle='round')
        return 'single'

    elif len(regions) == 2:
        (theta1_A, theta2_A), (theta1_B, theta2_B) = regions
        # p0 = start of A, p1 = end of A, p2 = start of B, p3 = end of B
        p0 = complex(np.cos(theta1_A), np.sin(theta1_A))
        p1 = complex(np.cos(theta2_A), np.sin(theta2_A))
        p2 = complex(np.cos(theta1_B), np.sin(theta1_B))
        p3 = complex(np.cos(theta2_B), np.sin(theta2_B))

        # Option 1: enclose each region (disconnected wedge)
        # Geodesics from p0->p1 and p2->p3
        len_p0_p1 = geodesic_length_boundary_points(theta1_A, theta2_A)
        len_p2_p3 = geodesic_length_boundary_points(theta1_B, theta2_B)
        length_enclose = len_p0_p1 + len_p2_p3

        # Option 2: connect across (connected wedge)
        # Geodesics from p1->p2 and p3->p0
        len_p1_p2 = geodesic_length_boundary_points(theta2_A, theta1_B)
        len_p3_p0 = geodesic_length_boundary_points(theta2_B, theta1_A)
        length_connect = len_p1_p2 + len_p3_p0

        if debug:
            print(f"  Region A: theta = ({theta1_A:.3f}, {theta2_A:.3f}) rad = ({np.degrees(theta1_A):.1f}°, {np.degrees(theta2_A):.1f}°)")
            print(f"  Region B: theta = ({theta1_B:.3f}, {theta2_B:.3f}) rad = ({np.degrees(theta1_B):.1f}°, {np.degrees(theta2_B):.1f}°)")
            print(f"  4 boundary points (CCW on unit circle):")
            print(f"    p0 (start A) at {np.degrees(theta1_A):.1f}°: ({p0.real:.3f}, {p0.imag:.3f})")
            print(f"    p1 (end A)   at {np.degrees(theta2_A):.1f}°: ({p1.real:.3f}, {p1.imag:.3f})")
            print(f"    p2 (start B) at {np.degrees(theta1_B):.1f}°: ({p2.real:.3f}, {p2.imag:.3f})")
            print(f"    p3 (end B)   at {np.degrees(theta2_B):.1f}°: ({p3.real:.3f}, {p3.imag:.3f})")
            # Angular gaps between regions (should be CCW order: p0 -> p1 -> p2 -> p3 -> p0)
            gap_A = (theta2_A - theta1_A) % (2*np.pi)  # region A angular span
            gap_B = (theta2_B - theta1_B) % (2*np.pi)  # region B angular span
            gap_1to2 = (theta1_B - theta2_A) % (2*np.pi)  # gap from end of A to start of B
            gap_3to0 = (theta1_A - theta2_B) % (2*np.pi)  # gap from end of B to start of A (wraparound)
            print(f"  Angular spans: A={np.degrees(gap_A):.1f}°, gap1→2={np.degrees(gap_1to2):.1f}°, B={np.degrees(gap_B):.1f}°, gap3→0={np.degrees(gap_3to0):.1f}°")
            print(f"  Total: {np.degrees(gap_A + gap_1to2 + gap_B + gap_3to0):.1f}° (should be 360°)")
            print(f"  Option 1 (enclose each): d(p0,p1) + d(p2,p3) = {np.degrees(len_p0_p1):.1f}° + {np.degrees(len_p2_p3):.1f}° = {np.degrees(length_enclose):.1f}°")
            print(f"  Option 2 (connect across): d(p1,p2) + d(p3,p0) = {np.degrees(len_p1_p2):.1f}° + {np.degrees(len_p3_p0):.1f}° = {np.degrees(length_connect):.1f}°")

        if length_enclose <= length_connect:
            # Draw geodesics enclosing each region separately
            for z_start, z_end in [(p0, p1), (p2, p3)]:
                xs, ys = poincare_geodesic_arc(z_start, z_end, n_points=100)
                ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder, solid_capstyle='round')
            if debug:
                print(f"  -> Chose: enclose_each (disconnected wedge)")
            return 'enclose_each'
        else:
            # Draw geodesics connecting across the gaps
            for z_start, z_end in [(p1, p2), (p3, p0)]:
                xs, ys = poincare_geodesic_arc(z_start, z_end, n_points=100)
                ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder, solid_capstyle='round')
            if debug:
                print(f"  -> Chose: connect_across (connected wedge)")
            return 'connect_across'
    return None


def fill_entanglement_wedge(ax, pos, boundary_nodes, selected_boundary, fill_color='pink', alpha=0.5, zorder=0):
    """Fill the entanglement wedge for a contiguous region."""
    if len(selected_boundary) == 0 or len(selected_boundary) == len(boundary_nodes):
        return None
    p1, p2 = pos[selected_boundary[0]], pos[selected_boundary[-1]]
    angular_spacing = 2 * np.pi / len(boundary_nodes)
    angular_offset = angular_spacing / 2
    theta1 = np.arctan2(p1[1], p1[0]) - angular_offset
    theta2 = np.arctan2(p2[1], p2[0]) + angular_offset
    z1, z2 = complex(np.cos(theta1), np.sin(theta1)), complex(np.cos(theta2), np.sin(theta2))
    geodesic_x, geodesic_y = poincare_geodesic_arc(z1, z2, n_points=100)
    diff = theta1 - theta2
    while diff > 0: diff -= 2 * np.pi
    while diff < -2 * np.pi: diff += 2 * np.pi
    boundary_angles = np.linspace(theta2, theta2 + diff, 100)
    wedge_x = np.concatenate([geodesic_x, np.cos(boundary_angles)])
    wedge_y = np.concatenate([geodesic_y, np.sin(boundary_angles)])
    ax.fill(wedge_x, wedge_y, color=fill_color, alpha=alpha, zorder=zorder)


def draw_rt_surface(ax, pos, boundary_nodes, selected_boundary, color='cyan', linewidth=3, linestyle='-', alpha=0.9, zorder=10):
    """Draw the Ryu-Takayanagi surface for a contiguous boundary region."""
    if len(selected_boundary) == 0 or len(selected_boundary) == len(boundary_nodes):
        return None
    p1, p2 = pos[selected_boundary[0]], pos[selected_boundary[-1]]
    angular_spacing = 2 * np.pi / len(boundary_nodes)
    angular_offset = angular_spacing / 2
    theta1 = np.arctan2(p1[1], p1[0]) - angular_offset
    theta2 = np.arctan2(p2[1], p2[0]) + angular_offset
    z1, z2 = complex(np.cos(theta1), np.sin(theta1)), complex(np.cos(theta2), np.sin(theta2))
    xs, ys = poincare_geodesic_arc(z1, z2, n_points=100)
    line, = ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder, solid_capstyle='round')
    return line


def invert_lightness(color, max_value=0.25):
    """Invert lightness of a color for contrast."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l_inv = 1.0 - l
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_inv, s)
    return (min(r2, max_value), min(g2, max_value), min(b2, max_value))


def build_radial_layout(G, boundary_nodes, hyperbolic_spacing=0.4):
    """Build radial layout for decorated graph in Poincaré disk.

    Args:
        G: NetworkX graph with depth and tier attributes
        boundary_nodes: List of boundary node IDs
        hyperbolic_spacing: Spacing parameter for hyperbolic radial coordinate

    Returns:
        pos: Dict mapping node ID to (x, y) position
    """
    pos = {}
    n_boundary = len(boundary_nodes)

    for node in G.nodes():
        node_data = G.nodes[node]
        node_depth = node_data.get('depth')

        if node_data.get('is_boundary'):
            boundary_pos = node_data.get('position_in_boundary', 0)
            angular_position = 2 * np.pi * boundary_pos / n_boundary + np.pi / n_boundary
            radius = 1
        elif node_depth is not None:
            nodes_at_depth = [n for n in G.nodes()
                            if G.nodes[n].get('depth') == node_depth
                            and G.nodes[n].get("tier") == "primary"
                            and not G.nodes[n].get('is_boundary')]
            nodes_at_depth = sorted(nodes_at_depth)
            if node in nodes_at_depth:
                node_idx = nodes_at_depth.index(node)
                n_at_depth = len(nodes_at_depth)
                angular_position = 2 * np.pi * node_idx / n_at_depth + np.pi / n_at_depth
            else:
                angular_position = 0
            euclidean_r = hyperbolic_spacing * node_depth
            radius = np.tanh(euclidean_r / 2)
        else:
            continue

        x = radius * np.cos(angular_position)
        y = radius * np.sin(angular_position)
        pos[node] = np.array([x, y])

    return pos


def compute_complement_regions(region_spec, n_boundary):
    """Compute complement of a discontinuous region specification.

    Args:
        region_spec: List of (start, end) tuples defining regions
        n_boundary: Total number of boundary nodes

    Returns:
        List of (start, end) tuples for complement regions
    """
    all_selected = set()
    for (start, end) in region_spec:
        all_selected.update(range(start, end))

    complement_indices = sorted([i for i in range(n_boundary) if i not in all_selected])

    # Find contiguous runs in complement
    complement_regions = []
    if complement_indices:
        run_start = complement_indices[0]
        prev = complement_indices[0]
        for i in complement_indices[1:] + [None]:
            if i is None or i != prev + 1:
                complement_regions.append((run_start, prev + 1))
                if i is not None:
                    run_start = i
            prev = i if i is not None else prev

    return complement_regions

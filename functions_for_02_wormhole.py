import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import colorsys

from copy import deepcopy
import networkx as nx

from graph2grav import analytic_entropies
from graph2grav.plotting_3d import draw_graph_3d, get_depth_from_n

import graph2grav.graphs as graphs
import graph2grav.gaussian as gaussian
import graph2grav.technical_helpers as th
from graph2grav.analysis import Analysis


def UndecoratedAnalysis(graph: nx.Graph, couplingtime=1, bulk_presqueeze=1, boundary_presqueeze=1, config=None):
    """Backwards-compatible wrapper for undecorated analysis.

    .. deprecated::
        Use `Analysis(graph, is_decorated=False, ...)` directly instead.

    Note: bulk_presqueeze and boundary_presqueeze are now unified into global_presqueeze.
    If they differ, bulk_presqueeze takes precedence.
    """
    warnings.warn(
        "UndecoratedAnalysis() is deprecated. Use Analysis(graph, is_decorated=False, ...) directly.",
        DeprecationWarning,
        stacklevel=2
    )
    presqueeze = bulk_presqueeze if bulk_presqueeze == boundary_presqueeze else bulk_presqueeze
    analysis = Analysis(
        graph,
        couplingtime=couplingtime,
        global_presqueeze=presqueeze,
        is_decorated=False,
        config=config,
    )
    analysis.fit_central_charge()
    return analysis


def DecoratedAnalysis(graph: nx.Graph, couplingtime=1., global_presqueeze=1., config=None):
    """Backwards-compatible wrapper for decorated analysis.

    .. deprecated::
        Use `Analysis(graph, is_decorated=True, ...)` directly instead.
    """
    warnings.warn(
        "DecoratedAnalysis() is deprecated. Use Analysis(graph, is_decorated=True, ...) directly.",
        DeprecationWarning,
        stacklevel=2
    )
    analysis = Analysis(
        graph,
        couplingtime=couplingtime,
        global_presqueeze=global_presqueeze,
        is_decorated=True,
        config=config,
    )
    analysis.fit_central_charge()
    return analysis


def undecorated_wormhole(ax=None, plot=True, config=None, onesided=False):
    """Instead of overlaying two graphs with transparency, build a real wormhole with a neck."""

    if config is None:
        raise ValueError("config must not be none")

    start_depth = config["wormhole-figure"]["wormhole"]["start_depth"]
    end_depth = config["wormhole-figure"]["wormhole"]["end_depth"]
    neck_length = config["wormhole-figure"]["wormhole"]["neck_length"]
    mouth_length = end_depth - start_depth # the mouth are the are varying depth regions on either side of the neck

    ptree: nx.Graph = graphs.crosslinked_tree_with_hole(end_depth, start_depth-1)
    ptree2 = deepcopy(ptree)

    # One wormhole mouth starts at 0 wormdepth
    max_depth = max(get_depth_from_n(n, ptree) for n in ptree.nodes)

    for node in ptree.nodes:
        ptree.nodes[node]["wormdepth"] = max_depth - ptree.nodes[node]["depth"]

        if onesided:
            ptree.nodes[node]["b1"] = False
    


    # count how many nodes are in the start_depth layer
    neck_edge_number_of_nodes = sum(1 for n in ptree.nodes if get_depth_from_n(n, ptree) == start_depth)
    # make rectangular square lattice for the neck
    neck_lattice = nx.grid_2d_graph(neck_edge_number_of_nodes, neck_length)

    # draw edges to make circular boundary conditions
    for i in range(neck_length):
        neck_lattice.add_edge((0, i), (neck_edge_number_of_nodes-1, i))


    # The neck starts at the max wormdepth of ptree and continues from there
    max_wormdepth_ptree = max(ptree.nodes[n]["wormdepth"] for n in ptree.nodes)

    for node in neck_lattice.nodes():
        neck_lattice.nodes[node]["wormdepth"] = max_wormdepth_ptree + node[1]
        neck_lattice.nodes[node]["is_boundary"] = False  # mark neck nodes as non-boundary
        neck_lattice.nodes[node]["b0"] = False  # mark neck nodes as non-boundary
        if onesided:
            neck_lattice.nodes[node]["b1"] = False  # mark neck nodes as non-boundary
        neck_lattice.nodes[node]["is_ancilla"] = True  # mark neck nodes as non-boundary
        neck_lattice.nodes[node]["is_neck"] = True

    # ptree2 becomes the second mouth
    minimum_depth_ptree2 = min(get_depth_from_n(n, ptree2) for n in ptree2.nodes)
    for node in ptree2.nodes:
        ptree2.nodes[node]["wormdepth"] = max_wormdepth_ptree + neck_length + ptree2.nodes[node]["depth"]  - minimum_depth_ptree2 - 1

        if onesided:
            if ptree2.nodes[node]["is_boundary"]:
                ptree2.nodes[node]["b1"] = True
                ptree2.nodes[node]["b0"] = False
            else:
                ptree2.nodes[node]["b1"] = False
                ptree2.nodes[node]["b0"] = False

    



    # glue neck to ptree and ptree2 appropriately
    nx.relabel_nodes(neck_lattice, dict_neck:={node: (node[0], neck_lattice.nodes[node]["wormdepth"]) for node in neck_lattice.nodes()}, copy=False)
    nx.relabel_nodes(ptree, dict_ptree:={node: (node, ptree.nodes[node]["wormdepth"]) for node in ptree.nodes()}, copy=False)
    nx.relabel_nodes(ptree2, dict_ptree2:={node: (node, ptree2.nodes[node]["wormdepth"]) for node in ptree2.nodes()}, copy=False)

    # compose graphs to glue them together
    full_wormhole_graph = nx.compose_all([ptree, neck_lattice, ptree2])

    for node in full_wormhole_graph.nodes():
        # if is_neck attribute is not there, set it to False
        if "is_neck" not in full_wormhole_graph.nodes[node]:
            full_wormhole_graph.nodes[node]["is_neck"] = False

    # pos3D = nx.kamada_kawai_layout(full_wormhole_graph, dim=3)
    wormdepths = []
    for node in full_wormhole_graph.nodes():
        wormdepth = full_wormhole_graph.nodes[node]["wormdepth"]
        wormdepths.append(wormdepth)
    


    max_radius = end_depth 

    radii = np.zeros_like(wormdepths, dtype=float)


    scale = 0.75

    for i, wormdepth in enumerate(np.unique(wormdepths)):
        if wormdepth < mouth_length:
            # first mouth
            radii[i] = max_radius - wormdepth / scale
        elif wormdepth < mouth_length + neck_length: # we are in the neck then
            # second mouth
            radii[i] = max_radius - mouth_length / scale
        else: # we are in the second mouth
            # neck region
            radii[i] = max_radius - mouth_length /scale + (wormdepth - mouth_length - neck_length + 1) / scale
        

    numbers_of_nodes_per_wormdepth = {wd: sum(1 for n in full_wormhole_graph.nodes() if full_wormhole_graph.nodes[n]["wormdepth"] == wd) for wd in np.unique(wormdepths)}
    smallest_node_label_per_wormdepth = {wd: min(n[0] for n in full_wormhole_graph.nodes() if full_wormhole_graph.nodes[n]["wormdepth"] == wd) for wd in np.unique(wormdepths)}




    # def xy_from_node_label(node):
    #     wormdepth = full_wormhole_graph.nodes[node]["wormdepth"]

    #     pass

    pos3D = {}

    # def wormdepth_is_in_neck(wormdepth):
    #     return mouth_length <= wormdepth < mouth_length + neck_length


    for node in full_wormhole_graph.nodes():
        wormdepth = full_wormhole_graph.nodes[node]["wormdepth"]

        numbers_of_nodes = numbers_of_nodes_per_wormdepth[wormdepth]
        smallest_node_label = smallest_node_label_per_wormdepth[wormdepth]
        index_in_layer = node[0] - smallest_node_label

        angular_spacing = 2 * np.pi / numbers_of_nodes

        angular_position = 2 * np.pi * index_in_layer / numbers_of_nodes + angular_spacing / 2

        x = radii[wormdepth] * np.cos(angular_position)
        y = radii[wormdepth] * np.sin(angular_position)



        pos3D[node] = np.array([x, y, 2* wormdepth])




    color_mode = "before_measurement"  # or "before_measurement"

    inner_colors = []
    outer_colors = []

    if color_mode == "before_measurement":
        inner_colors = [
            config["graph"]["nodes"]["neck"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["boundary"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["ancilla"]["inner"]["color"]
            for n in full_wormhole_graph.nodes
        ]
        outer_colors = [
            config["graph"]["nodes"]["neck"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["boundary"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["ancilla"]["outer"]["color"]
            for n in full_wormhole_graph.nodes
        ]
    elif color_mode == "after_measurement":
        inner_colors = [
            config["graph"]["nodes"]["post_measurement"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["boundary"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["post_measurement"]["inner"]["color"]
            for n in full_wormhole_graph.nodes
        ]
        outer_colors = [
            config["graph"]["nodes"]["post_measurement"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["boundary"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["post_measurement"]["outer"]["color"]
            for n in full_wormhole_graph.nodes
        ]


    label_type = "none"
    labels = None

    if label_type == "node":
        labels = [node for node in full_wormhole_graph.nodes()]
    elif label_type == "wormdepth":
        labels = {n: str(ptree.nodes[n]["wormdepth"]) for n in full_wormhole_graph.nodes()}
    elif label_type == "none":
        labels = {n: "" for n in full_wormhole_graph.nodes()}
    elif label_type == "position_in_boundary":
        labels = {
            n: f"{full_wormhole_graph.nodes[n]['position_in_boundary']}"
            if full_wormhole_graph.nodes[n].get("is_boundary", False)
            else ""
            for n in full_wormhole_graph.nodes()
        }
    
    
    

    # draw the full wormhole graph using pos3D

    if plot:
        if ax is None:
            fig = plt.figure(figsize=(config["wormhole-figure"]["width"]*2, config["wormhole-figure"]["height"]*2))
            ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        # ax.set_axis_off()
        draw_graph_3d(
            ax, full_wormhole_graph, pos3D, inner_colors, outer_colors, labels, reverse=False, face_alpha=0.5, show_faces=False
        )



        all_pts = np.array(list(pos3D.values()))
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        ranges = maxs - mins
        pad = 0.05 * ranges.max() if np.isfinite(ranges.max()) and ranges.max() > 0 else 0.05
        ax.set_xlim(mins[0]-pad, maxs[0]+pad)
        ax.set_ylim(mins[1]-pad, maxs[1]+pad)
        ax.set_zlim(mins[2]-pad, maxs[2]+pad)
        # Equal aspect in 3D
        # ax.set_box_aspect((ranges[0] + 2*pad, ranges[1] + 2*pad, 2 *(ranges[2] + 2*pad)))

        ax.view_init(
            azim=config["wormhole-figure"]["view"]["azim"],
            elev=config["wormhole-figure"]["view"]["elev"],
            roll=config["wormhole-figure"]["view"]["roll"]
        )
        ax.set_proj_type(config["wormhole-figure"]["view"]["projection"])
    
    # relabel nodes to be from 0 to N-1
    mapping = {n: i for i, n in enumerate(full_wormhole_graph.nodes())}
    full_wormhole_graph = nx.relabel_nodes(full_wormhole_graph, mapping)

    # update pos3D accordingly
    pos3D = {mapping[n]: pos3D[n] for n in pos3D.keys()}

    for node in full_wormhole_graph.nodes():
        full_wormhole_graph.nodes[node]["position_in_3D"] = pos3D[node]
    
    return full_wormhole_graph

def decorated_wormhole(ax=None, plot=True, config=None, onesided=False):
    """Instead of overlaying two graphs with transparency, build a real wormhole with a neck.

    Args:
        ax: Matplotlib 3D axes to plot on
        plot: Whether to plot the graph
        config: Configuration dictionary
        onesided: If True, first mouth is b0 and second mouth is b1 (separate boundaries)
    """

    if config is None:
        raise ValueError("config must not be none")

    start_depth = config["wormhole-figure"]["wormhole"]["start_depth"]
    end_depth = config["wormhole-figure"]["wormhole"]["end_depth"]
    neck_length = config["wormhole-figure"]["wormhole"]["neck_length"]
    mouth_length = end_depth - start_depth # the mouth are the are varying depth regions on either side of the neck

    ptree: nx.Graph = graphs.crosslinked_tree_with_hole(end_depth, start_depth-1)
    ptree2 = deepcopy(ptree)

    # One wormhole mouth starts at 0 wormdepth
    max_depth = max(get_depth_from_n(n, ptree) for n in ptree.nodes)

    for node in ptree.nodes:
        ptree.nodes[node]["wormdepth"] = max_depth - ptree.nodes[node]["depth"]

        if onesided:
            ptree.nodes[node]["b1"] = False

    # count how many nodes are in the start_depth layer
    neck_edge_number_of_nodes = sum(1 for n in ptree.nodes if get_depth_from_n(n, ptree) == start_depth)
    # make rectangular square lattice for the neck
    neck_lattice = nx.grid_2d_graph(neck_edge_number_of_nodes, neck_length)

    # draw edges to make circular boundary conditions
    for i in range(neck_length):
        neck_lattice.add_edge((0, i), (neck_edge_number_of_nodes-1, i))


    # The neck starts at the max wormdepth of ptree and continues from there
    max_wormdepth_ptree = max(ptree.nodes[n]["wormdepth"] for n in ptree.nodes)

    for node in neck_lattice.nodes():
        neck_lattice.nodes[node]["wormdepth"] = max_wormdepth_ptree + node[1]
        neck_lattice.nodes[node]["is_boundary"] = False  # mark neck nodes as non-boundary
        neck_lattice.nodes[node]["b0"] = False  # mark neck nodes as non-boundary
        if onesided:
            neck_lattice.nodes[node]["b1"] = False  # mark neck nodes as non-boundary
        neck_lattice.nodes[node]["is_ancilla"] = True  # mark neck nodes as non-boundary
        neck_lattice.nodes[node]["is_neck"] = True
        neck_lattice.nodes[node]["tier"] = "primary"

    # ptree2 becomes the second mouth
    minimum_depth_ptree2 = min(get_depth_from_n(n, ptree2) for n in ptree2.nodes)
    for node in ptree2.nodes:
        ptree2.nodes[node]["wormdepth"] = max_wormdepth_ptree + neck_length + ptree2.nodes[node]["depth"]  - minimum_depth_ptree2 - 1

        if onesided:
            if ptree2.nodes[node]["is_boundary"]:
                ptree2.nodes[node]["b1"] = True
                ptree2.nodes[node]["b0"] = False
            else:
                ptree2.nodes[node]["b1"] = False
                ptree2.nodes[node]["b0"] = False

    # glue neck to ptree and ptree2 appropriately
    nx.relabel_nodes(neck_lattice, dict_neck:={node: (node[0], neck_lattice.nodes[node]["wormdepth"]) for node in neck_lattice.nodes()}, copy=False)
    nx.relabel_nodes(ptree, dict_ptree:={node: (node, ptree.nodes[node]["wormdepth"]) for node in ptree.nodes()}, copy=False)
    nx.relabel_nodes(ptree2, dict_ptree2:={node: (node, ptree2.nodes[node]["wormdepth"]) for node in ptree2.nodes()}, copy=False)

    # compose graphs to glue them together
    full_wormhole_graph = nx.compose_all([ptree, neck_lattice, ptree2])

    for node in full_wormhole_graph.nodes():
        # if is_neck attribute is not there, set it to False
        if "is_neck" not in full_wormhole_graph.nodes[node]:
            full_wormhole_graph.nodes[node]["is_neck"] = False

    # pos3D = nx.kamada_kawai_layout(full_wormhole_graph, dim=3)
    wormdepths = []
    for node in full_wormhole_graph.nodes():
        wormdepth = full_wormhole_graph.nodes[node]["wormdepth"]
        wormdepths.append(wormdepth)
    


    max_radius = end_depth 

    radii = np.zeros_like(wormdepths, dtype=float)


    scale = 0.75

    for i, wormdepth in enumerate(np.unique(wormdepths)):
        if wormdepth < mouth_length:
            # first mouth
            radii[i] = max_radius - wormdepth / scale
        elif wormdepth < mouth_length + neck_length: # we are in the neck then
            # second mouth
            radii[i] = max_radius - mouth_length / scale
        else: # we are in the second mouth
            # neck region
            radii[i] = max_radius - mouth_length /scale + (wormdepth - mouth_length - neck_length + 1) / scale
        

    numbers_of_nodes_per_wormdepth = {wd: sum(1 for n in full_wormhole_graph.nodes() if full_wormhole_graph.nodes[n]["wormdepth"] == wd) for wd in np.unique(wormdepths)}
    smallest_node_label_per_wormdepth = {wd: min(n[0] for n in full_wormhole_graph.nodes() if full_wormhole_graph.nodes[n]["wormdepth"] == wd) for wd in np.unique(wormdepths)}





    pos3D = {}

    # def wormdepth_is_in_neck(wormdepth):
    #     return mouth_length <= wormdepth < mouth_length + neck_length


    for node in full_wormhole_graph.nodes():
        wormdepth = full_wormhole_graph.nodes[node]["wormdepth"]

        numbers_of_nodes = numbers_of_nodes_per_wormdepth[wormdepth]
        smallest_node_label = smallest_node_label_per_wormdepth[wormdepth]
        index_in_layer = node[0] - smallest_node_label

        angular_spacing = 2 * np.pi / numbers_of_nodes

        angular_position = 2 * np.pi * index_in_layer / numbers_of_nodes + angular_spacing / 2

        x = radii[wormdepth] * np.cos(angular_position)
        y = radii[wormdepth] * np.sin(angular_position)



        pos3D[node] = np.array([x, y, wormdepth])
    
    # Here is where we want to decorate!

    edges = list(full_wormhole_graph.edges())

    for edge in edges:
        node1, node2 = edge
        wormdepth1 = full_wormhole_graph.nodes[node1]["wormdepth"]
        wormdepth2 = full_wormhole_graph.nodes[node2]["wormdepth"]

        new_wormdepth = (wormdepth1 + wormdepth2) / 2

        new_position = (pos3D[node1] + pos3D[node2]) / 2
        # add new node
        new_node_name = (f"intermediate_{node1}_{node2}", new_wormdepth)
        full_wormhole_graph.add_node(new_node_name)
        full_wormhole_graph.nodes[new_node_name]["wormdepth"] = new_wormdepth
        full_wormhole_graph.nodes[new_node_name]["is_ancilla"] = True
        full_wormhole_graph.nodes[new_node_name]["is_boundary"] = False
        full_wormhole_graph.nodes[new_node_name]["tier"] = "secondary"
        full_wormhole_graph.nodes[new_node_name]["b0"] = False
        full_wormhole_graph.nodes[new_node_name]["b1"] = False
        full_wormhole_graph.nodes[new_node_name]["is_neck"] = False
        pos3D[new_node_name] = new_position
        # remove old edge
        full_wormhole_graph.remove_edge(node1, node2)
        # add edges to new node
        full_wormhole_graph.add_edge(node1, new_node_name, mweight = 1.0)
        full_wormhole_graph.add_edge(node2, new_node_name, mweight = -1.0)

    if add_probes := True: 
        # Make new boundary by attaching probe nodes to all boundary nodes,
        # then setting old boundary nodes to non-boundary
        boundary_nodes = [n for n in full_wormhole_graph.nodes() if full_wormhole_graph.nodes[n].get("is_boundary", False)]

        min_wormdepth = min(full_wormhole_graph.nodes[n]["wormdepth"] for n in full_wormhole_graph.nodes())
        max_wormdepth = max(full_wormhole_graph.nodes[n]["wormdepth"] for n in full_wormhole_graph.nodes())

        for bnode in boundary_nodes:
            wormdepth = full_wormhole_graph.nodes[bnode]["wormdepth"]
            new_node_name = (f"probe_{bnode}", wormdepth)
            full_wormhole_graph.add_node(new_node_name)
            full_wormhole_graph.nodes[new_node_name]["wormdepth"] = wormdepth
            full_wormhole_graph.nodes[new_node_name]["is_ancilla"] = False
            full_wormhole_graph.nodes[new_node_name]["is_boundary"] = True
            full_wormhole_graph.nodes[new_node_name]["tier"] = "primary"
            full_wormhole_graph.nodes[new_node_name]["is_neck"] = False
            full_wormhole_graph.nodes[new_node_name]["position_in_boundary"] = full_wormhole_graph.nodes[bnode]["position_in_boundary"]

            # Inherit boundary assignment from old boundary node
            if onesided:
                # Check which boundary the old node belonged to
                old_b0 = full_wormhole_graph.nodes[bnode].get("b0", False)
                old_b1 = full_wormhole_graph.nodes[bnode].get("b1", False)
                full_wormhole_graph.nodes[new_node_name]["b0"] = old_b0
                full_wormhole_graph.nodes[new_node_name]["b1"] = old_b1
            else:
                full_wormhole_graph.nodes[new_node_name]["b0"] = True
                full_wormhole_graph.nodes[new_node_name]["b1"] = False

            # if at minimal wormdepth, set wormdepth to be one less than min to avoid overlap
            # if at maximal wormdepth, set wormdepth to be one more than max to avoid overlap
            wormdepth = full_wormhole_graph.nodes[bnode]["wormdepth"]
            if wormdepth == min_wormdepth:
                full_wormhole_graph.nodes[new_node_name]["wormdepth"] = min_wormdepth - 1
            elif wormdepth == max_wormdepth:
                full_wormhole_graph.nodes[new_node_name]["wormdepth"] = max_wormdepth + 1

            wormdepth = full_wormhole_graph.nodes[new_node_name]["wormdepth"]
            # set position accordingly

            old_position = pos3D[bnode]
            pos3D[new_node_name] = np.array([old_position[0], old_position[1], wormdepth])

            # connect new probe node to old boundary node
            full_wormhole_graph.add_edge(bnode, new_node_name, mweight = 1.0)
            # set old boundary node to non-boundary
            full_wormhole_graph.nodes[bnode]["is_boundary"] = False
            full_wormhole_graph.nodes[bnode]["is_ancilla"] = True
            full_wormhole_graph.nodes[bnode]["b0"] = False
            if onesided:
                full_wormhole_graph.nodes[bnode]["b1"] = False



    color_mode = "before_measurement"  # or "before_measurement"

    inner_colors = []
    outer_colors = []

    if color_mode == "before_measurement":
        inner_colors = [
            config["graph"]["nodes"]["neck"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["ancilla_secondary"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["tier"] == "secondary"
            else config["graph"]["nodes"]["boundary"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["ancilla"]["inner"]["color"]
            for n in full_wormhole_graph.nodes
        ]
        outer_colors = [
            config["graph"]["nodes"]["neck"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["ancilla_secondary"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["tier"] == "secondary"
            else config["graph"]["nodes"]["boundary"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["ancilla"]["outer"]["color"]
            for n in full_wormhole_graph.nodes
        ]
    elif color_mode == "after_measurement":
        inner_colors = [
            config["graph"]["nodes"]["post_measurement"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["boundary"]["inner"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["post_measurement"]["inner"]["color"]
            for n in full_wormhole_graph.nodes
        ]
        outer_colors = [
            config["graph"]["nodes"]["post_measurement"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_neck"]
            else config["graph"]["nodes"]["boundary"]["outer"]["color"]
            if full_wormhole_graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["post_measurement"]["outer"]["color"]
            for n in full_wormhole_graph.nodes
        ]


    label_type = "none"
    labels = None

    if label_type == "node":
        labels = [node for node in full_wormhole_graph.nodes()]
    elif label_type == "wormdepth":
        labels = {n: str(full_wormhole_graph.nodes[n]["wormdepth"]) for n in full_wormhole_graph.nodes()}
    elif label_type == "none":
        labels = {n: "" for n in full_wormhole_graph.nodes()}
    elif label_type == "position_in_boundary":
        labels = {
            n: f"{full_wormhole_graph.nodes[n]['position_in_boundary']}"
            if full_wormhole_graph.nodes[n].get("is_boundary", False)
            else ""
            for n in full_wormhole_graph.nodes()
        }
    
    
    

    # draw the full wormhole graph using pos3D

    if plot:
        if ax is None:
            fig = plt.figure(figsize=(config["wormhole-figure"]["width"], config["wormhole-figure"]["height"]))
            ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        # ax.set_axis_off()
        draw_graph_3d(
            ax, full_wormhole_graph, pos3D, inner_colors, outer_colors, labels, reverse=False, face_alpha=0.5, show_faces=False
        )



        all_pts = np.array(list(pos3D.values()))
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        ranges = maxs - mins
        pad = 0.05 * ranges.max() if np.isfinite(ranges.max()) and ranges.max() > 0 else 1.0
        ax.set_xlim(mins[0]-pad, maxs[0]+pad)
        ax.set_ylim(mins[1]-pad, maxs[1]+pad)
        ax.set_zlim(mins[2]-pad, maxs[2]+pad)

        # Equal aspect in 3D
        # ax.set_box_aspect((ranges[0] + 2*pad, ranges[1] + 2*pad, 2 *(ranges[2] + 2*pad)))

        ax.view_init(
            azim=config["wormhole-figure"]["view"]["azim"],
            elev=config["wormhole-figure"]["view"]["elev"],
            roll=config["wormhole-figure"]["view"]["roll"]
        )
        ax.set_proj_type(config["wormhole-figure"]["view"]["projection"])
    
    # relabel nodes to be from 0 to N-1
    mapping = {n: i for i, n in enumerate(full_wormhole_graph.nodes())}
    full_wormhole_graph = nx.relabel_nodes(full_wormhole_graph, mapping)

    # update pos3D accordingly
    pos3D = {mapping[n]: pos3D[n] for n in pos3D.keys()}

    for node in full_wormhole_graph.nodes():
        full_wormhole_graph.nodes[node]["position_in_3D"] = pos3D[node]
    
    return full_wormhole_graph
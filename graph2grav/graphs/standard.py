"""Standard graph construction functions for tree-based and subdivided graphs.

This module provides functions for creating:
- Binary trees with crosslinks
- Subdivided trees with probe nodes
- Ring graphs with subdivisions

The following "tree" functions also work on graphs that are indexed like binary trees.

It's not important that the graph is actually a tree. For example, in the crosslinked
tree case, the graph has extra edges that make the graph contain cycles, but the functions are
still useful for ascertaining node properties.

Similarly, the functions will not be accurate
but perhaps have extra edges that make the graph contain cycles.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

# Import from validation module
from .validation import check_labels


# --- Tree Helper Functions ---

def tree_node_depth(n):
    """Depth of a node in a tree-based graph. The root has depth 0."""
    return np.floor(np.log2(n+1))

def tree_node_rowindex(n):
    """Position of a node along a row in a tree-based graph."""
    return int(n - 2**tree_node_depth(n) + 1)

# Note: it is not recommended to use this.
# Use tree.nodes[node]['is_boundary'] instead.
def tree_is_boundary_node(n, maxdepth):
    return tree_node_depth(n) == maxdepth


# --- Tree Construction Functions ---

def crosslinked_tree(depth, validate=True):
    """
    Create a tree with crosslinks between nodes at the same depth.
    """

    tree = nx.balanced_tree(2, depth)
    # for i in range(depth):
    #     for j in range(2**i):
    #         tree.add_edge(2**i + j, 2**i + j + 2**i)

    for edge in tree.edges:
        tree.edges[edge]["is_last_layer"] = False
    
    tree.nodes[0]["depth"] = 0

    for layer in range(1, depth+1):
        start = 2**layer - 1
        end = 2**(layer+1) - 2
        for i in range(start, end):
            tree.add_edge(i, i+1)
            tree.nodes[i]["depth"] = layer
            tree.nodes[i+1]["depth"] = layer

            if layer == depth:
                # label edge as last layer
                tree.edges[i, i+1]["is_last_layer"] = True
            else:
                tree.edges[i, i+1]["is_last_layer"] = False

    for node in tree.nodes:
        mynode = tree.nodes[node]
        if tree_node_depth(node) < depth:
            mynode['is_ancilla'] = True
            mynode['is_boundary'] = False
            mynode['b0'] = False
        else:
            mynode['is_ancilla'] = False
            mynode['is_boundary'] = True
            mynode['b0'] = True

            # boundary information
            mynode['boundary_index'] = 0
            mynode['position_in_boundary'] = tree_node_rowindex(node)

    tree.graph["number_of_boundaries"] = 1
    tree.graph["max_depth"] = depth
    tree.graph["periodic"] = False
    tree.graph["subdivided"] = False

    for node in tree.nodes:
        tree.nodes[node]["graph"] = tree.graph

    if validate:
        check_labels(tree)

    return tree

def crosslinked_tree_with_hole(depth, hole_depth=None, periodic=True, validate=True):
    """
    Create a tree with crosslinks between nodes at the same depth.
    """

    if hole_depth is not None and hole_depth < 0:
        hole_depth = None

    tree = nx.balanced_tree(2, depth)
    # for i in range(depth):
    #     for j in range(2**i):
    #         tree.add_edge(2**i + j, 2**i + j + 2**i)

    for edge in tree.edges:
        tree.edges[edge]["is_last_layer"] = False

    for layer in range(1, depth+1):
        start = 2**layer - 1
        end = 2**(layer+1) - 2
        for i in range(start, end):
            tree.add_edge(i, i+1)
            tree.nodes[i]["depth"] = layer
            tree.nodes[i+1]["depth"] = layer
            if layer == depth:
                # label edge as last layer
                tree.edges[i, i+1]["is_last_layer"] = True
            else:
                tree.edges[i, i+1]["is_last_layer"] = False

    for node in tree.nodes:
        mynode = tree.nodes[node]
        mynode["tier"] = "primary"
        if tree_node_depth(node) < depth:
            mynode['is_ancilla'] = True
            mynode['is_boundary'] = False
            mynode['b0'] = False
        else:
            mynode['is_ancilla'] = False
            mynode['is_boundary'] = True
            mynode['b0'] = True

            # boundary information
            mynode['boundary_index'] = 0
            mynode['position_in_boundary'] = tree_node_rowindex(node)

    tree.graph["number_of_boundaries"] = 1
    tree.graph["max_depth"] = depth
    tree.graph["periodic"] = False
    tree.graph["subdivided"] = False

    for node in tree.nodes:
        tree.nodes[node]["graph"] = tree.graph

    if periodic:
        for d in range(1, tree.graph["max_depth"] + 1):
            tree.add_edge(2**d - 1, 2**(d+1) - 2)

            if d == tree.graph["max_depth"]:
                # label edge as last layer
                tree.edges[2**d - 1, 2**(d+1) - 2]["is_last_layer"] = True
            else:
                tree.edges[2**d - 1, 2**(d+1) - 2]["is_last_layer"] = False

    if hole_depth is not None:
        remove_list = []

        for node in tree.nodes:
            if tree_node_depth(node) <= hole_depth:
                remove_list.append(node)

        for node in remove_list:
            tree.remove_node(node)

        nx.relabel_nodes(tree, {n: m for m, n in enumerate(tree.nodes)}, copy=False)

        H = nx.Graph()
        H.add_nodes_from(sorted(tree.nodes(data=True)))
        H.add_edges_from(tree.edges(data=True))
        H.graph = deepcopy(tree.graph)

        tree = H

    if validate:
        check_labels(tree)

    return tree


def tree_pos(depth):
    tree = nx.balanced_tree(2, depth)
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot') # for plotting

    return pos

def make_tree_periodic(G, remove_nodes=True):
    Gp = deepcopy(G)
    # add in periodic boundary conditions
    for d in range(1, Gp.graph["max_depth"] + 1):
        Gp.add_edge(2**d - 1, 2**(d+1) - 2)
        if d == Gp.graph["max_depth"]:
            # label edge as last layer
            Gp.edges[2**d - 1, 2**(d+1) - 2]["is_last_layer"] = True
        else:
            Gp.edges[2**d - 1, 2**(d+1) - 2]["is_last_layer"] = False

    if remove_nodes:
        Gp.remove_node(0)
        Gp.remove_node(1)
        Gp.remove_node(2)

    # Otherwise we run into indexing issues
    nx.relabel_nodes(Gp, {n: m for m, n in enumerate(Gp.nodes)}, copy=False)

    H = nx.Graph()
    H.add_nodes_from(sorted(Gp.nodes(data=True)))
    H.add_edges_from(Gp.edges(data=True))
    H.graph = deepcopy(Gp.graph)

    Gp = H

    Gp.graph["periodic"] = True


    return Gp


# --- Subdivision Functions ---

def subdivided_tree(
        maxdepth, burnlist=[], draw=True,
        bulk_weight=1,
        probe_weight=1, remove_crown=False,
        only_subdivide_boundary=False,
        validate=True
    ):
    """
    Create a tree with crosslinks between nodes at the same depth, and subdivide the edges.
    Each edge spawns two edges with opposite `mweight` and a new node in between.

    The follwing attributes are added to the nodes:

    "tier":
        - "primary" for initial nodes, before subdividing
        - "secondary" for nodes that are created by subdividing edges

    "region":
        - "bulk" for nodes that are within the bulk
          - aka "b0"
        - "probe" for the nodes within each pair appended to the bulk

    "is_ancilla":
        - True for nodes that are ancilla (i.e. are bulk)
        - False for nodes that are not ancilla

    "is_boundary":
        - True for nodes that are boundary (i.e. are probes)
        - False for nodes that are not boundary
    """
    G: nx.Graph = nx.balanced_tree(2, maxdepth)

    # old way
    nx.set_node_attributes(G, "primary", "tier")
    nx.set_node_attributes(G, "bulk", "region")

    # new way
    nx.set_node_attributes(G, True, "is_ancilla")
    nx.set_node_attributes(G, False, "is_boundary")
    nx.set_node_attributes(G, False, "b0")


    crosslinks = []

    periodic = True

    if use_crosslinks := True:
        for layer in range(1, maxdepth+1):
            start = 2**layer - 1
            end = 2**(layer+1) - 2

            if periodic:
                G.add_edge(start, end)
                crosslinks.append((start, end))

            for i in range(start, end):
                G.add_edge(i, i+1)
                crosslinks.append((i, i+1))

    lastrow_start = 2**maxdepth - 1
    lastrow_end = 2 **(maxdepth+1) - 1

    probe_edges = []

    double = True

    # Add probe nodes
    for index in range(lastrow_end - lastrow_start):
        G.add_node(
            lastrow_end + index, tier="primary", region="probe",
            is_ancilla=False, is_boundary=True, b0=True,
            boundary_index=0, position_in_boundary=index
        )
        G.add_edge(lastrow_start + index, lastrow_end + index)
        probe_edges.append((lastrow_start + index, lastrow_end + index))

    if remove_crown:
        # removing these nodes makes the final graph cleaner.
        for node in [0, 1, 2]:
            G.remove_node(node)

        G = nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=True)

    if pre_remove_up_to_depth is not None:
        remove_list = [
            node
            for node, data in G.nodes(data=True)
            if data.get("depth") is not None and data["depth"] <= pre_remove_up_to_depth
        ]
        G.remove_nodes_from(remove_list)
        G = nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=True)

    G0 = G.copy()
    next_node_number = G.number_of_nodes()

    nx.set_edge_attributes(G, 1, "mweight")

    for edge in list(G.edges):
        # G.add_edge(edge[0], f"{edge[0]}-{edge[1]}", mweight=1)
        # G.add_edge(edge[1], f"{edge[0]}-{edge[1]}", mweight=-1)


        is_edge_node = False

        if G.nodes[edge[0]]["region"] == G.nodes[edge[1]]["region"] == "bulk":

            if only_subdivide_boundary:
                continue

            # If we are subdividing an edge between two bulk nodes
            # print(f"Subdividing edge {edge} between two bulk nodes")
            region = "bulk"
            mweight = bulk_weight

        elif G.nodes[edge[0]]["region"] == G.nodes[edge[1]]["region"] == "probe":
            # print(f"Subdividing edge {edge} between two probe nodes")
            region = "probe"
            mweight = probe_weight

            # maybe should raise error here
            raise ValueError("Probe nodes should not be connected")
        else:
            # print(f"Subdividing edge {edge} between a bulk and a probe node")
            region = "probe"
            mweight = probe_weight

            is_edge_node = True

            if G.nodes[edge[0]]["region"] == "probe":
                probe_node = G.nodes[edge[0]]
            else:
                probe_node = G.nodes[edge[1]]

            # we link the position in boundary to the probe position in boundary
            # we will have to write more code to make this work properly,
            # and maybe this is not the right thing to do.
            position_in_boundary = probe_node["position_in_boundary"]

        G.add_node(
            next_node_number,
            tier="secondary",
            region=region,
            is_ancilla=(region == "bulk"),
            is_boundary=(region == "probe"),
            boundary_index=0,
        )

        if is_edge_node:
            G.nodes[next_node_number]["position_in_boundary"] = position_in_boundary
            G.nodes[next_node_number]["b0"] = True
        else:
            G.nodes[next_node_number]["b0"] = False


        G.add_edge(edge[0], next_node_number, mweight=mweight)
        G.add_edge(edge[1], next_node_number, mweight=-mweight)
        G.remove_edge(edge[0], edge[1])

        next_node_number += 1

    probe_edges = []


    N = G.number_of_nodes()
    boundary_start = lastrow_end + 1
    boundary_end = N

    # make hole

    if burn:= True:
        # burnlist = np.arange(0, 7)
        for node in burnlist:
            # if connected node is an intermediate node, remove it
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if G.nodes[neighbor]["ntype"] == "intermediate":
                    G.remove_node(neighbor)
            G.remove_node(node)
        # G.remove_nodes_from(burnlist)

    G = nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=True)

    # Outdated code to add shortctuts

    # if add_shortcut := False:
    #     G.add_node(last:=G.number_of_nodes(), ntype="intermediate", btype="bulk2")
    #     G.add_edge(last, 0, mweight=1)
    #     G.add_edge(last, 8, mweight=-1)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    G = H

    if draw:
        draw_subdivided_graph(G)

    G.graph["number_of_boundaries"] = 1
    G.graph["periodic"] = True
    G.graph["subdivided"] = True


    for node in G.nodes:
        G.nodes[node]["graph"] = G.graph

    if validate:
        check_labels(G)

    return G # , G0

def subdivided_tree_decoration_1(
        maxdepth, burnlist=[], draw=True,
        bulk_weight=1,
        probe_weight=1, remove_crown=False,
        only_subdivide_boundary=False,
        validate=True,
        remove_up_to_depth=None,
        pre_remove_up_to_depth=None
    ):
    """
    Create a tree with crosslinks between nodes at the same depth, and subdivide the edges.
    Each edge spawns two edges with opposite `mweight` and a new node in between.

    In decoration style 1, we only use one probe node per probe leg.

    The follwing attributes are added to the nodes:

    "tier":
        - "primary" for initial nodes, before subdividing
        - "secondary" for nodes that are created by subdividing edges

    "region":
        - "bulk" for nodes that are within the bulk
          - aka "b0"
        - "probe" for the nodes within each pair appended to the bulk

    "is_ancilla":
        - True for nodes that are ancilla (i.e. are bulk)
        - False for nodes that are not ancilla

    "is_boundary":
        - True for nodes that are boundary (i.e. are probes)
        - False for nodes that are not boundary
    """
    G: nx.Graph = nx.balanced_tree(2, maxdepth)

    # old way
    nx.set_node_attributes(G, "primary", "tier")
    nx.set_node_attributes(G, "bulk", "region")

    # new way
    nx.set_node_attributes(G, True, "is_ancilla")
    nx.set_node_attributes(G, False, "is_boundary")
    nx.set_node_attributes(G, False, "b0")

    for node in G.nodes:
        G.nodes[node]["depth"] = tree_node_depth(node)
        G.nodes[node]["position_in_layer"] = tree_node_rowindex(node)

    for edge in G.edges:
        G.edges[edge]["depth"] = min(G.nodes[edge[0]]["depth"], G.nodes[edge[1]]["depth"])


    crosslinks = []

    periodic = True

    if use_crosslinks := True:
        for layer in range(1, maxdepth+1):
            start = 2**layer - 1
            end = 2**(layer+1) - 2

            if periodic:
                G.add_edge(start, end, depth=layer)
                crosslinks.append((start, end))

            for i in range(start, end):
                G.add_edge(i, i+1, depth=layer)
                crosslinks.append((i, i+1))

    lastrow_start = 2**maxdepth - 1
    lastrow_end = 2 **(maxdepth+1) - 1

    probe_edges = []

    main_graph_size = G.number_of_nodes()

    # Add probe nodes
    for index in range(lastrow_end - lastrow_start):
        G.add_node(
            lastrow_end + index, tier="primary", region="probe",
            is_ancilla=False, is_boundary=True, b0=True,
            boundary_index=0, position_in_boundary=index,
            depth=None
        )
        G.add_edge(lastrow_start + index, lastrow_end + index, depth=None)
        probe_edges.append((lastrow_start + index, lastrow_end + index))

    if remove_crown:
        # removing these nodes makes the final graph cleaner.
        for node in [0, 1, 2]:
            G.remove_node(node)

        G = nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=True)

        main_graph_size = main_graph_size - 3

    if pre_remove_up_to_depth is not None:
        remove_list = [
            node
            for node, data in G.nodes(data=True)
            if data.get("depth") is not None and data["depth"] <= pre_remove_up_to_depth
        ]
        G.remove_nodes_from(remove_list)
        nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=False)

    G0 = G.copy()
    next_node_number = G.number_of_nodes()

    nx.set_edge_attributes(G, 1, "mweight")

    for edge in list(G.edges):
        # G.add_edge(edge[0], f"{edge[0]}-{edge[1]}", mweight=1)
        # G.add_edge(edge[1], f"{edge[0]}-{edge[1]}", mweight=-1)


        is_edge_node = False

        if G.nodes[edge[0]]["region"] == G.nodes[edge[1]]["region"] == "bulk":

            if only_subdivide_boundary:
                continue

            # If we are subdividing an edge between two bulk nodes
            # print(f"Subdividing edge {edge} between two bulk nodes")
            region = "bulk"
            mweight = bulk_weight

        elif G.nodes[edge[0]]["region"] == G.nodes[edge[1]]["region"] == "probe":
            # print(f"Subdividing edge {edge} between two probe nodes")
            region = "probe"
            mweight = probe_weight

            G.edges[edge]["mweight"] = probe_weight

            # maybe should raise error here
            raise ValueError("Probe nodes should not be connected")
        else:
            # Otherwise, we are connecting a bulk node to a probe node

            # in decoration style 1, we only use one probe node per probe leg

            # Probe weight sets how strongly the probe node is connected to the bulk node
            G.edges[edge]["mweight"] = probe_weight

            continue

        # The rest assumes you are a connector node between nodes of the main graph
        node_depth = min(G.nodes[edge[0]]["depth"], G.nodes[edge[1]]["depth"])

        G.add_node(
            next_node_number,
            tier="secondary",
            region=region,
            is_ancilla=(region == "bulk"),
            is_boundary=(region == "probe"),
            boundary_index=0,
            depth=node_depth
        )

        G.nodes[next_node_number]["b0"] = False


        G.add_edge(edge[0], next_node_number, mweight=mweight, depth=node_depth)
        G.add_edge(edge[1], next_node_number, mweight=-mweight, depth=node_depth)
        G.remove_edge(edge[0], edge[1])

        next_node_number += 1

    probe_edges = []


    N = G.number_of_nodes()
    boundary_start = lastrow_end + 1
    boundary_end = N

    # make hole

    if burn:= True:
        # burnlist = np.arange(0, 7)
        for node in burnlist:
            # if connected node is an intermediate node, remove it
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if G.nodes[neighbor]["ntype"] == "intermediate":
                    G.remove_node(neighbor)
            G.remove_node(node)
        # G.remove_nodes_from(burnlist)

    if remove_up_to_depth is not None:
        remove_list = []

        for node in G.nodes:
            if "depth" not in G.nodes[node]:
                print(f"Node {node} does not have a depth attribute ({G.nodes[node]}). Skipping.")
                continue

            if G.nodes[node]["depth"] is not None and G.nodes[node]["depth"] <= remove_up_to_depth:
                remove_list.append(node)

        for node in remove_list:
            G.remove_node(node)

        remove_edge_list = []

        for edge in G.edges:
            if "depth" not in G.edges[edge]:
                print(f"Edge {edge} does not have a depth attribute ({G.edges[edge]}). Skipping.")
                continue

            if G.edges[edge]["depth"] is not None and G.edges[edge]["depth"] <= remove_up_to_depth:
                remove_edge_list.append(edge)

        for edge in remove_edge_list:
            G.remove_edge(*edge)

    G = nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=True)

    # Outdated code to add shortctuts

    # if add_shortcut := False:
    #     G.add_node(last:=G.number_of_nodes(), ntype="intermediate", btype="bulk2")
    #     G.add_edge(last, 0, mweight=1)
    #     G.add_edge(last, 8, mweight=-1)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    G = H

    if draw:
        draw_subdivided_graph(G)

    G.graph["number_of_boundaries"] = 1
    G.graph["periodic"] = True
    G.graph["subdivided"] = True
    G.graph["depth"] = maxdepth
    G.graph["main_graph_size"] = main_graph_size


    for node in G.nodes:
        G.nodes[node]["graph"] = G.graph

    if validate:
        check_labels(G)

    return G # , G0


# --- Visualization Functions ---

def draw_subdivided_graph(G: nx.Graph, ax=None, node_size=500, font_size=8):
    """
    Draw a subdivided tree with nodes colored by their type.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    color_map = [get_standard_node_color(G, n) for n in G.nodes]
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G, pos, ax, with_labels=True,
        node_size=node_size, node_color=color_map,
        font_size=font_size,
        edge_color=["red" if G.edges[edge]["mweight"] > 0 else "blue" for edge in G.edges]
    )


def get_standard_node_color(G: nx.Graph, node):
    """
    Get the color of a node in a subdivided tree based on its type.

    For use with subdivided tree and subdivided ring.
    """
    if G.nodes[node]["tier"] == "primary":
        if G.nodes[node]["region"] == "probe":
            return "pink"

        # otherwise it's a bulk primary node
        return "lightblue"
    else:
        if G.nodes[node]["region"] == "probe":
            return "lightgreen"

        # otherwise it's a bulk secondary node
        return "yellow"


# --- Ring Construction ---

def subdividedring(ringlength=6, shortcuts=[(0, 3)], draw=True, probe_weight=0.00001):
    G: nx.Graph = nx.cycle_graph(ringlength)

    nx.set_node_attributes(G, "main", "ntype")
    nx.set_node_attributes(G, "bulk", "btype")


    lastrow_start = 0
    lastrow_end = ringlength

    probe_edges = []

    double = True

    for i in range(lastrow_end - lastrow_start):
        G.add_node(lastrow_end + i, ntype="main", btype="probe")
        G.add_edge(lastrow_start + i, lastrow_end + i)
        probe_edges.append((lastrow_start + i, lastrow_end + i))

    G0 = G.copy()
    count = G.number_of_nodes()

    nx.set_edge_attributes(G, 1, "mweight")

    for edge in list(G.edges):

        if G.nodes[edge[0]]["btype"] == "bulk" and G.nodes[edge[1]]["btype"] == "bulk":
            if not double:
                continue
            btype="bulk2"
            mweight = 1
        elif G.nodes[edge[0]]["btype"] == "probe" and G.nodes[edge[1]]["btype"] == "probe":
            btype="probe"
            mweight = probe_weight
        else:
            btype="edge"
            mweight = probe_weight

        G.add_node(count, tier="secondary", btype=btype)

        G.add_edge(edge[0], count, mweight=mweight)
        G.add_edge(edge[1], count, mweight=-mweight)
        G.remove_edge(edge[0], edge[1])

        count += 1


    index = 0

    probe_edges = []


    N = G.number_of_nodes()

    nx.relabel_nodes(G, {n: m for m, n in enumerate(G.nodes)}, copy=False)

    index = G.number_of_nodes()
    for shortcut in shortcuts:
        G.add_node(index, ntype="intermediate", btype="bulk2")
        G.add_edge(index, shortcut[0], mweight=1)
        G.add_edge(index, shortcut[1], mweight=-1)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    G = H


    if draw:
        draw_subdivided_graph(G)

    return G, G0

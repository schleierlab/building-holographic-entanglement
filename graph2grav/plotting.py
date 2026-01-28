#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:05:23 2021

@author: philipp and avikar
"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
from datetime import datetime
from cycler import cycler
from dataclasses import dataclass

import networkx as nx

import numpy as np


def evenbounds(data):
    """Return the bounds for data that are symmetric around zero."""
    return {
        "vmin": -np.max(np.abs(data)),
        "vmax": np.max(np.abs(data)),
    }


@dataclass
class StanfordColors:
    paloalto = "#175e54"
    paloverde = "#279989"
    olive = "#8F993E"
    bay = "#6FA287"
    sky = "#4298B5"
    lagunita = "#007C92"
    poppy="#E98300"
    spirited="#E04F39"
    illuminating="#FEDD5C"
    plum="#620059"
    light_sky = "#67afd2"
    light_plum = "#734675"
    light_poppy = "#F9A44A"
    light_archway = "#766253"
    archway =  "#5D4B3C"
    cardinal = "#8C1515"
    light_cardinal = "#B83A4B"
    stone = "#7F7776"
    dark_bay = "#417865"
    dark_sky = "#016895"
    dark_poppy = "#D1660F"

sc = StanfordColors()

colors = [sc.light_cardinal, sc.paloalto, sc.plum, sc.lagunita]
stanford_cycle = cycler(color=colors)

default_red = "#c8140c"
default_blue = "#295da6"
default_gold = "#ec8d0d"
default_purple = "#762f83"
default_green = "#23c333"
default_pink = "#dda0ab"

default_colors = [
    default_red,  # red
    default_blue,  # blue
    default_gold,  # gold
    default_purple,  # purple
    default_green,  # green
    default_pink,
]

helvetica_font = {"family": "HelveticaNeue"}

default_cycle = cycler(color=default_colors)
rcParams["axes.prop_cycle"] = default_cycle


def save_as_png(fig, name, location="PNGs", include_time=True, **kwargs):
    """Save figure as PNG with optional timestamp.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Base filename (without extension).
    location : str, optional
        Directory to save in. Default "PNGs".
    include_time : bool, optional
        Include timestamp in filename. Default True.
    **kwargs
        Additional arguments passed to fig.savefig().
    """
    if not os.path.isdir(location):
        os.mkdir(location)
    current_time = datetime.now()
    time_str = current_time.strftime("%y-%m-%d_%H%M%S") if include_time else ""
    fig.savefig(os.path.join(location, f"{name}{time_str}.png"), format="png", **kwargs)


def save_as_pdf(fig, name, location="PDFs", include_time=True, **kwargs):
    """Save figure as PDF with optional timestamp.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Base filename (without extension).
    location : str, optional
        Directory to save in. Default "PDFs".
    include_time : bool, optional
        Include timestamp in filename. Default True.
    **kwargs
        Additional arguments passed to fig.savefig().
    """
    if not os.path.isdir(location):
        os.mkdir(location)
    current_time = datetime.now()
    time_str = current_time.strftime("%y-%m-%d_%H%M%S") if include_time else ""
    fig.savefig(os.path.join(location, f"{name}{time_str}.pdf"), format="pdf", **kwargs)

def plot_graph_from_config(
    graph, config, ax=None, post_measurement=False,
    displace_boundary=False, show_measurement_circle=False
):
    """The protocol comprises the left column of Figure 1."""


    assert config is not None, "You must provide a config."

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(config["default-figure"]["width"], config["default-figure"]["height"]))
    

    ax.set_axis_off()

    if post_measurement:
        # measurement removes entanglement between ancilla and the boundary nodes
        edge_list = list(graph.edges)

        for edge in edge_list:
            if graph.nodes[edge[0]]["is_ancilla"] or graph.nodes[edge[1]]["is_ancilla"]:
                graph.remove_edge(*edge)

    # kamada kawai layout
    pos = nx.kamada_kawai_layout(graph)

    if post_measurement:
        inner_colors = [
            config["graph"]["nodes"]["boundary"]["inner"]["color"]
            if graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["post_measurement"]["inner"]["color"]
            for n in graph.nodes
        ]

        outer_colors = [
            config["graph"]["nodes"]["boundary"]["outer"]["color"]
            if graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["post_measurement"]["outer"]["color"]
            for n in graph.nodes
        ]
    else:
        inner_colors = [
            config["graph"]["nodes"]["boundary"]["inner"]["color"]
            if graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["ancilla"]["inner"]["color"]
            for n in graph.nodes
        ]
        outer_colors = [
            config["graph"]["nodes"]["boundary"]["outer"]["color"]
            if graph.nodes[n]["is_boundary"]
            else config["graph"]["nodes"]["ancilla"]["outer"]["color"]
            for n in graph.nodes
        ]

    node_names_by_set_to_label = {
        "all": None,
        "just_boundary": {n: graph.nodes[n]["position_in_boundary"] if graph.nodes[n]["is_boundary"] else "" for n in graph.nodes},
        "none": {n: "" for n in graph.nodes}
    }





    nx.draw_networkx_nodes(graph, pos , ax=ax, node_color=outer_colors, node_size=200)
    nx.draw_networkx_nodes(graph, pos , ax=ax, node_color=inner_colors, node_size=100)

    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=graph.edges, width=2)
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=node_names_by_set_to_label[config["graph"]["which_nodes_to_label"]], font_color="black")

    # draw measurement circle on the Measure subplot

    if show_measurement_circle:
        circle = plt.Circle( # pyright: ignore
            (0, 0), 0.8, color=config["operations"]["measure"]["color"], fill=True, linewidth=2, alpha=config["operations"]["measure"]["alpha"],
            zorder = 10
        )
        ax.add_artist(circle)

    # draw displacement square on the Displace subplot
    if displace_boundary:
        for node in graph.nodes:
            if graph.nodes[node]["is_boundary"]:
                x, y = pos[node]

                size = 0.4 # maybe add to settings
                circle = mpl.patches.Circle( # pyright: ignore
                    (x, y), size/2,
                    color=config["operations"]["displace"]["color"],
                    alpha=config["operations"]["displace"]["alpha"],
                    linewidth=0, fill=True,
                    zorder=10
                )
                ax.add_patch(circle)
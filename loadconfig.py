import tomllib
import matplotlib.pyplot as plt

import graph2grav.plotting

with open("global_figure_settings.toml", "rb") as f:
    config = tomllib.load(f)

def reload():
    """Reload the global figure configuration."""
    global config
    with open("global_figure_settings.toml", "rb") as f:
        config = tomllib.load(f)

    plt.rc('xtick', labelsize=config["plt-runconfig"]["xtick"]["labelsize"])
    plt.rc('ytick', labelsize=config["plt-runconfig"]["ytick"]["labelsize"])
    plt.rc('axes', labelsize=config["plt-runconfig"]["axes"]["labelsize"])
    plt.rc('axes', titlesize=config["plt-runconfig"]["axes"]["titlesize"])
    plt.rc('legend', fontsize=config["plt-runconfig"]["legend"]["fontsize"])
    plt.rc('figure', titlesize=config["plt-runconfig"]["figure"]["titlesize"])
    plt.rc('font', family=config["plt-runconfig"]["font"]["family"])

    plt.rc('xtick', direction=config["plt-runconfig"]["xtick"]["direction"])
    plt.rc('ytick', direction=config["plt-runconfig"]["ytick"]["direction"])

    graph2grav.plotting.configure_depth_palette(config)

    return config

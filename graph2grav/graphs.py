"""Graph construction and validation (backward compatibility module).

This module maintains backward compatibility by re-exporting everything from
the graphs submodule. New code should import directly from graph2grav.graphs
or its submodules (graphs.standard, graphs.hyperbolic, graphs.validation).

DEPRECATED: This file exists only for backward compatibility.
            Use `from graph2grav.graphs import ...` instead.
"""

# Re-export everything from the graphs submodule for backward compatibility
from .graphs import *  # noqa: F401, F403
from .graphs import __all__  # noqa: F401

# Also make submodules accessible
from . import graphs  # noqa: F401

"""Graph construction and validation for graph2grav.

This submodule provides functions for creating various graph structures
used in holographic quantum state preparation.

Most commonly used functions are exposed at the top level.
Less common functions can be accessed via submodules:
  - graphs.standard.*
  - graphs.hyperbolic.*
  - graphs.validation.*
"""

# Import commonly used functions for convenient access
from .standard import (
    crosslinked_tree,
    crosslinked_tree_with_hole,
    subdivided_tree_decoration_1,
)

from .hyperbolic import (
    generate_hyperbolic_tiling_3_8_geometric_with_raw,
)

from .validation import (
    check_labels,
)

# Make submodules accessible
from . import standard
from . import hyperbolic
from . import validation

__all__ = [
    # Commonly used functions
    'crosslinked_tree',
    'crosslinked_tree_with_hole',
    'subdivided_tree_decoration_1',
    'generate_hyperbolic_tiling_3_8_geometric_with_raw',
    'check_labels',
    # Submodules
    'standard',
    'hyperbolic',
    'validation',
]

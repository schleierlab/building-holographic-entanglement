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
    generate_hyperbolic_tiling_with_hypertiling,
    generate_hyperbolic_tiling_3_q_geometric_with_raw,
    generate_hyperbolic_tiling_3_8_geometric_with_raw,
)

from .regular import (
    RegularTilingSpec,
    decorate_regular_patch,
    generate_regular_patch,
    label_circular_boundary_from_embedding,
    supported_regular_tilings,
)

from .validation import (
    check_labels,
)

# Make submodules accessible
from . import standard
from . import hyperbolic
from . import regular
from . import validation

__all__ = [
    # Commonly used functions
    'crosslinked_tree',
    'crosslinked_tree_with_hole',
    'subdivided_tree_decoration_1',
    'generate_hyperbolic_tiling_with_hypertiling',
    'generate_hyperbolic_tiling_3_q_geometric_with_raw',
    'generate_hyperbolic_tiling_3_8_geometric_with_raw',
    'RegularTilingSpec',
    'decorate_regular_patch',
    'generate_regular_patch',
    'label_circular_boundary_from_embedding',
    'supported_regular_tilings',
    'check_labels',
    # Submodules
    'standard',
    'hyperbolic',
    'regular',
    'validation',
]

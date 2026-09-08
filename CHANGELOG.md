# Changelog

## Unreleased: Supplemental Figure Port

### Added

- Added supplemental notebooks `A3` through `A9` for the new figure set.
- Added generated supplemental figure PDFs for `A3` through `A9`.
- Added regular tiling support in `graph2grav/graphs/regular.py`, including regular patch generation, boundary labeling, and decoration of regular patches.
- Added `graph2grav/tree_wormhole_knockout.py` for shared MERA, decorated MERA, and hyperbolic graph construction used by the new supplemental notebooks.
- Added VFPE utilities in `graph2grav/vfpe.py` and high-precision companion routines in `graph2grav/vfpe_mpmath.py`.
- Added `tests/test_regular_and_vfpe.py` covering regular graph generation and VFPE support.
- Added A3-A9 notebooks to `scripts/regenerate_figures.py`.

### Changed

- Updated depth styling to use a shared depth palette from `global_figure_settings.toml`.
- Updated A0 scaling plots to use the shared depth color helper.
- Updated graph plotting/config helpers to support the new supplemental figure styling.
- Updated `.gitignore` to ignore generated notebook caches, Matplotlib cache files, notebook checkpoints, test cache, and temporary rendered previews.

### Supplemental Figures

- `supplementary-notebooks/S08_regular_graphs.ipynb`
  - Builds regular `{3,7}` and `{3,8}` graph examples.
  - Plots entropy curves with periodic-CFT fits.
  - Adds scaled bulk cut-distance diagnostic labeled as scaled `d_bulk`.
  - Places fitted central charge beside the CFT fit label.

- `supplementary-notebooks/S06_boundary_mutual_information.ipynb`
  - Ports multi-interval mutual information diagnostics.
  - Plots mutual information versus boundary separation and circular cross-ratio `eta`.
  - Uses the circular sine cross-ratio appropriate for a periodic boundary.

- `supplementary-notebooks/S07_decorated_boundary_mutual_information.ipynb`
  - Adds decorated MERA and decorated `{3,7}` mutual-information diagnostics.
  - Removes small-cross-ratio fit overlays from the decorated plots.
  - Adds an early graph-check plot with separate colors for boundary, decoration, and original nodes.
  - Labels an example pair of intervals `A_1` and `A_2` on the decorated graph check.

- `supplementary-notebooks/S09_decorated_mera_vfpe.ipynb`
  - Adds decorated MERA VFPE diagnostics.
  - Uses the raw VFPE residual `sigma(K_Delta)` rather than the normalized residual.
  - Uses the McGreevy-paper-style `c_Delta` estimator from the four-arc `c_hat_Delta` operator.
  - Condenses the main figure to a one-row, three-panel layout.

- `A6_undecorated_mera_vfpe.ipynb`
  - Adds undecorated MERA VFPE companion diagnostics.
  - Uses consistent MERA depth notation `R_b`.

- `A7_decorated_mera_hole.ipynb`
  - Adds decorated MERA pre-decoration hole diagnostics.
  - Uses `R_b` for MERA depth and `R_h` for hole cutoff notation.

- `A8_fixed_thickness_decorated_shells.ipynb`
  - Adds fixed-thickness decorated shell diagnostics.
  - Replaces `d_max` notation with `R_b`.

- `supplementary-notebooks/S03_strong_squeezing_lattice_entropy.ipynb`
  - Adds a single-panel strong-squeezing entropy curve illustrating lattice-scale/piecewise behavior.

### Notes

- The cache directories under `outputs/` and rendered PNG previews under `tmp/` are generated artifacts and are now ignored.
- The new figure notebooks currently include executed outputs for visual inspection.

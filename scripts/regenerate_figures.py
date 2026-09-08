#!/usr/bin/env python
"""Regenerate all figures by executing figure source files.

Usage:
    python scripts/regenerate_figures.py           # Run all figure sources
    python scripts/regenerate_figures.py --dry-run # Show what would run
    python scripts/regenerate_figures.py 01 S03    # Run specific notebooks (by prefix)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Figure sources in execution order
FIGURE_SOURCES = [
    "01_introduction.ipynb",
    "02_wormhole.ipynb",
    "03_decorated_graph.ipynb",
    "04_cartography.ipynb",
    "supplementary-notebooks/S01_illustration_of_decoration.ipynb",
    "supplementary-notebooks/S02_regular_graphs.ipynb",
    "supplementary-notebooks/S03_scaling.ipynb",
    "supplementary-notebooks/S04_rt_crossover_sharpness.ipynb",
    "supplementary-notebooks/S05_strong_squeezing_lattice_entropy.ipynb",
    "supplementary-notebooks/S06_correlations.ipynb",
    "supplementary-notebooks/S07_scaling_dimensions.ipynb",
    "supplementary-notebooks/S08_boundary_mutual_information.ipynb",
    "supplementary-notebooks/S09_decorated_boundary_mutual_information.ipynb",
    "supplementary-notebooks/S10_tripartite_mmi_combined.ipynb",
    "supplementary-notebooks/S11_decorated_mera_vfpe.ipynb",
]


def normalize_prefix(prefix: str) -> str:
    """Normalize source prefixes for matching FIGURE_SOURCES entries."""
    lower = prefix.lower()
    if lower.startswith(("s", "a")):
        try:
            return f"S{int(prefix[1:]):02d}"
        except ValueError:
            return prefix
    return prefix


def run_python_script(script: str, timeout: int = 600) -> bool:
    """Execute a Python figure script. Returns True on success."""
    path = ROOT / script
    if not path.exists():
        print(f"  [SKIP] {script} not found")
        return False

    print(f"  [RUN]  {script}...", end=" ", flush=True)

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
    )

    if result.returncode == 0:
        print("OK")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print("FAILED")
        print(result.stderr)
        return False


def run_notebook(notebook: str, timeout: int = 600) -> bool:
    """Execute a notebook in place. Returns True on success."""
    path = ROOT / notebook
    if not path.exists():
        print(f"  [SKIP] {notebook} not found")
        return False

    print(f"  [RUN]  {notebook}...", end=" ", flush=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("OK")
        return True
    else:
        print("FAILED")
        print(result.stderr)
        return False


def run_figure_source(source: str, timeout: int = 600) -> bool:
    """Execute a notebook or Python figure source."""
    if source.endswith(".py"):
        return run_python_script(source, timeout=timeout)
    return run_notebook(source, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description="Regenerate figures from notebooks and scripts")
    parser.add_argument(
        "prefixes",
        nargs="*",
        help="Figure source prefixes to run (e.g., '01' '03' 'S01'). Runs all if not specified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per notebook in seconds (default: 600)",
    )
    parser.add_argument(
        "--no-hash-update",
        action="store_true",
        help="Skip updating figure hashes after running",
    )
    args = parser.parse_args()

    # Filter notebooks by prefix if specified
    if args.prefixes:
        prefixes = [normalize_prefix(prefix) for prefix in args.prefixes]
        sources = [
            source for source in FIGURE_SOURCES
            if any(Path(source).name.startswith(p) for p in prefixes)
        ]
        if not sources:
            print(f"No figure sources match prefixes: {args.prefixes}")
            print(f"Available: {FIGURE_SOURCES}")
            sys.exit(1)
    else:
        sources = FIGURE_SOURCES

    print(f"Figure sources to run: {len(sources)}")
    for source in sources:
        print(f"  - {source}")

    if args.dry_run:
        print("\n[DRY RUN] No figure sources executed.")
        return

    # Capture baseline hashes before running
    print("\nCapturing baseline hashes...")
    from graph2grav.figure_hashing import compare_hashes, capture_all_hashes

    baseline = capture_all_hashes()

    # Run figure sources
    print(f"\nExecuting {len(sources)} figure sources...")
    results = {}
    for source in sources:
        results[source] = run_figure_source(source, timeout=args.timeout)

    # Summary
    succeeded = sum(results.values())
    failed = len(results) - succeeded

    print(f"\nResults: {succeeded} succeeded, {failed} failed")

    if failed > 0:
        print("\nFailed notebooks:")
        for nb, ok in results.items():
            if not ok:
                print(f"  - {nb}")

    # Update and compare hashes
    if not args.no_hash_update:
        print("\nUpdating figure hashes...")
        capture_all_hashes()

        print("\nComparing figures to baseline...")
        comparison = compare_hashes()

        if comparison["changed"]:
            print(f"\nChanged figures ({len(comparison['changed'])}):")
            for name, old_hash, new_hash in comparison["changed"]:
                print(f"  - {name}")
        else:
            print("\nNo figures changed.")

        if comparison["new"]:
            print(f"\nNew figures ({len(comparison['new'])}):")
            for name in comparison["new"]:
                print(f"  - {name}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

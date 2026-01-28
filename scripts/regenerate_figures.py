#!/usr/bin/env python
"""Regenerate all figures by executing notebooks.

Usage:
    python scripts/regenerate_figures.py           # Run all notebooks
    python scripts/regenerate_figures.py --dry-run # Show what would run
    python scripts/regenerate_figures.py 01 03     # Run specific notebooks (by prefix)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Notebooks in execution order
NOTEBOOKS = [
    "01_introduction.ipynb",
    "02_wormhole.ipynb",
    "03_decorated_graph.ipynb",
    "04_cartography.ipynb",
    "A0_scaling.ipynb",
    "A1_correlations.ipynb",
    "A2_scaling_dimensions.ipynb",
]


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


def main():
    parser = argparse.ArgumentParser(description="Regenerate figures from notebooks")
    parser.add_argument(
        "prefixes",
        nargs="*",
        help="Notebook prefixes to run (e.g., '01' '03' 'A1'). Runs all if not specified.",
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
        notebooks = [
            nb for nb in NOTEBOOKS
            if any(nb.startswith(p) for p in args.prefixes)
        ]
        if not notebooks:
            print(f"No notebooks match prefixes: {args.prefixes}")
            print(f"Available: {NOTEBOOKS}")
            sys.exit(1)
    else:
        notebooks = NOTEBOOKS

    print(f"Notebooks to run: {len(notebooks)}")
    for nb in notebooks:
        print(f"  - {nb}")

    if args.dry_run:
        print("\n[DRY RUN] No notebooks executed.")
        return

    # Capture baseline hashes before running
    print("\nCapturing baseline hashes...")
    from graph2grav.figure_hashing import compare_hashes, capture_all_hashes

    baseline = capture_all_hashes()

    # Run notebooks
    print(f"\nExecuting {len(notebooks)} notebooks...")
    results = {}
    for nb in notebooks:
        results[nb] = run_notebook(nb, timeout=args.timeout)

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

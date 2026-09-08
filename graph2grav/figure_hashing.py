"""Figure hashing utilities for tracking changes to generated figures.

This module provides tools to compute hashes of figures (both file content and
matplotlib data) and compare them across code changes to detect regressions.

Two files are maintained:
- `.hashes.json`: Current hashes for all figures (always reflects latest state)
- `.hash_history.json`: Timestamped history of changes (preserves previous states)

History is only logged when data_hash or size actually changes, not when just
the file_hash changes (which happens due to PDF metadata timestamps).

Usage:
    # Capture baseline hashes
    python -m graph2grav.figure_hashing capture

    # Compare current figures against baseline
    python -m graph2grav.figure_hashing compare

    # View hash history
    python -m graph2grav.figure_hashing history
    python -m graph2grav.figure_hashing history figure_1.pdf

    # In notebooks, save figures with automatic hash tracking
    from graph2grav.figure_hashing import save_figure_with_hash
    save_figure_with_hash(fig, "figures/figure_1.pdf")
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


# Default paths
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent / "figures"
DEFAULT_HASH_DB = DEFAULT_FIGURES_DIR / ".hashes.json"
DEFAULT_HISTORY_DB = DEFAULT_FIGURES_DIR / ".hash_history.json"


def _display_path(path: str | Path) -> str:
    """Return a local, non-absolute path for user-facing log messages."""
    path = Path(path)
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent]
    resolved = path.resolve()

    for base in candidates:
        try:
            return str(resolved.relative_to(base.resolve()))
        except ValueError:
            pass

    return path.name


def compute_file_hash(filepath: str | Path, short: bool = True) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to the file to hash
        short: If True, return truncated 16-character hash

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    full_hash = hasher.hexdigest()
    return full_hash[:16] if short else full_hash


def compute_figure_data_hash(fig: "matplotlib.figure.Figure", short: bool = True) -> str:
    """Compute hash of numerical data in a matplotlib figure.

    This hash captures the actual data being plotted (line coordinates,
    axis limits, etc.) rather than rendering details. Useful for detecting
    changes in computed values even when file format changes.

    Args:
        fig: Matplotlib Figure object
        short: If True, return truncated 16-character hash

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()

    for ax in fig.get_axes():
        # Hash axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        hasher.update(f"xlim:{xlim[0]:.10e},{xlim[1]:.10e}".encode())
        hasher.update(f"ylim:{ylim[0]:.10e},{ylim[1]:.10e}".encode())

        # Hash line data
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if hasattr(xdata, 'tobytes'):
                hasher.update(xdata.tobytes())
            else:
                hasher.update(np.array(xdata).tobytes())
            if hasattr(ydata, 'tobytes'):
                hasher.update(ydata.tobytes())
            else:
                hasher.update(np.array(ydata).tobytes())

        # Hash scatter/collection data
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if hasattr(offsets, 'tobytes'):
                hasher.update(offsets.tobytes())
            elif hasattr(offsets, 'data'):
                hasher.update(np.asarray(offsets).tobytes())

        # Hash image data
        for image in ax.images:
            arr = image.get_array()
            if arr is not None:
                hasher.update(np.asarray(arr).tobytes())

    full_hash = hasher.hexdigest()
    return full_hash[:16] if short else full_hash


def load_hash_db(hash_db: str | Path = DEFAULT_HASH_DB) -> dict:
    """Load the hash database from disk.

    Args:
        hash_db: Path to the hash database JSON file

    Returns:
        Dictionary mapping figure names to their hash info
    """
    hash_db = Path(hash_db)
    if hash_db.exists():
        with open(hash_db) as f:
            return json.load(f)
    return {}


def save_hash_db(db: dict, hash_db: str | Path = DEFAULT_HASH_DB) -> None:
    """Save the hash database to disk.

    Args:
        db: Dictionary mapping figure names to their hash info
        hash_db: Path to the hash database JSON file
    """
    hash_db = Path(hash_db)
    hash_db.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_db, 'w') as f:
        json.dump(db, f, indent=2, sort_keys=True)


def load_history_db(history_db: str | Path = DEFAULT_HISTORY_DB) -> dict:
    """Load the hash history database from disk.

    Args:
        history_db: Path to the history database JSON file

    Returns:
        Dictionary mapping figure names to list of historical hash entries
    """
    history_db = Path(history_db)
    if history_db.exists():
        with open(history_db) as f:
            return json.load(f)
    return {}


def save_history_db(db: dict, history_db: str | Path = DEFAULT_HISTORY_DB) -> None:
    """Save the hash history database to disk.

    Args:
        db: Dictionary mapping figure names to their history
        history_db: Path to the history database JSON file
    """
    history_db = Path(history_db)
    history_db.parent.mkdir(parents=True, exist_ok=True)
    with open(history_db, 'w') as f:
        json.dump(db, f, indent=2, sort_keys=True)


def log_hash_change(
    figure_name: str,
    hashes: dict,
    history_db: str | Path = DEFAULT_HISTORY_DB,
    max_entries_per_file: int = 100
) -> None:
    """Log a hash change to the history database.

    Only logs if the data_hash or size has changed from the most recent entry.

    Args:
        figure_name: Name of the figure file
        hashes: Dictionary with 'file_hash', 'data_hash', and 'size'
        history_db: Path to the history database JSON file
        max_entries_per_file: Maximum history entries to keep per file
    """
    history = load_history_db(history_db)

    # Create entry with timestamp
    entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'data_hash': hashes.get('data_hash'),
        'file_hash': hashes.get('file_hash'),
        'size': hashes.get('size')
    }

    if figure_name not in history:
        history[figure_name] = []

    # Check if this is actually a change (compare data_hash and size, not file_hash)
    # File hash changes due to PDF metadata, but data_hash reflects actual content
    if history[figure_name]:
        last_entry = history[figure_name][-1]
        if (last_entry.get('data_hash') == entry['data_hash'] and
            last_entry.get('size') == entry['size']):
            # No meaningful change, skip logging
            return

    # Add new entry
    history[figure_name].append(entry)

    # Trim to max entries
    if len(history[figure_name]) > max_entries_per_file:
        history[figure_name] = history[figure_name][-max_entries_per_file:]

    save_history_db(history, history_db)


def save_figure_with_hash(
    fig: "matplotlib.figure.Figure",
    filepath: str | Path,
    hash_db: str | Path = DEFAULT_HASH_DB,
    history_db: str | Path = DEFAULT_HISTORY_DB,
    verbose: bool = True,
    **savefig_kwargs
) -> dict:
    """Save a figure and update the hash database.

    Args:
        fig: Matplotlib Figure object
        filepath: Path to save the figure
        hash_db: Path to the hash database JSON file
        history_db: Path to the history database JSON file
        verbose: If True, print where the figure and hash records were saved
        **savefig_kwargs: Additional arguments passed to fig.savefig()

    Returns:
        Dictionary with hash information for this figure
    """
    filepath = Path(filepath)

    # Save the figure
    fig.savefig(filepath, **savefig_kwargs)

    # Compute hashes
    hashes = {
        'file_hash': compute_file_hash(filepath),
        'data_hash': compute_figure_data_hash(fig),
        'size': os.path.getsize(filepath)
    }

    # Log to history (only if data_hash or size changed)
    log_hash_change(filepath.name, hashes, history_db)

    # Update current database
    db = load_hash_db(hash_db)
    db[filepath.name] = hashes
    save_hash_db(db, hash_db)

    if verbose:
        print(
            f"Saved figure to {_display_path(filepath)}. "
            f"Hash info updated in {_display_path(hash_db)}; history in {_display_path(history_db)}. "
            f"file_hash={hashes['file_hash']}, data_hash={hashes['data_hash']}, size={hashes['size']} bytes."
        )

    return hashes


def capture_all_hashes(
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    hash_db: str | Path = DEFAULT_HASH_DB,
    history_db: str | Path = DEFAULT_HISTORY_DB,
    extensions: tuple[str, ...] = ('.pdf', '.png', '.svg')
) -> dict:
    """Capture file hashes for all figures in a directory.

    Note: This only captures file hashes, not data hashes (which require
    the matplotlib Figure objects). For data hashes, use save_figure_with_hash
    when generating figures.

    Args:
        figures_dir: Directory containing figures
        hash_db: Path to the hash database JSON file
        history_db: Path to the history database JSON file
        extensions: File extensions to include

    Returns:
        Dictionary mapping figure names to their hash info
    """
    figures_dir = Path(figures_dir)
    db = load_hash_db(hash_db)

    for filepath in figures_dir.iterdir():
        if filepath.suffix.lower() in extensions:
            # Update file hash (preserve data_hash if it exists)
            existing = db.get(filepath.name, {})
            hashes = {
                'file_hash': compute_file_hash(filepath),
                'data_hash': existing.get('data_hash'),  # Preserve existing
                'size': os.path.getsize(filepath)
            }
            db[filepath.name] = hashes

            # Log to history if this is a change
            log_hash_change(filepath.name, hashes, history_db)

    save_hash_db(db, hash_db)
    return db


def compare_hashes(
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    hash_db: str | Path = DEFAULT_HASH_DB,
    extensions: tuple[str, ...] = ('.pdf', '.png', '.svg')
) -> dict:
    """Compare current figure hashes against the baseline.

    Args:
        figures_dir: Directory containing figures
        hash_db: Path to the hash database JSON file
        extensions: File extensions to check

    Returns:
        Dictionary with comparison results:
        - 'unchanged': List of figures with matching hashes
        - 'changed': List of (name, old_hash, new_hash) for changed figures
        - 'new': List of figures not in baseline
        - 'missing': List of figures in baseline but not on disk
    """
    figures_dir = Path(figures_dir)
    db = load_hash_db(hash_db)

    results = {
        'unchanged': [],
        'changed': [],
        'new': [],
        'missing': []
    }

    # Find all current figures
    current_figures = {
        f.name for f in figures_dir.iterdir()
        if f.suffix.lower() in extensions
    }

    # Check figures in baseline
    for name, info in db.items():
        filepath = figures_dir / name
        if not filepath.exists():
            results['missing'].append(name)
            continue

        current_hash = compute_file_hash(filepath)
        if current_hash == info.get('file_hash'):
            results['unchanged'].append(name)
        else:
            results['changed'].append((name, info.get('file_hash'), current_hash))

    # Check for new figures
    for name in current_figures:
        if name not in db:
            results['new'].append(name)

    return results


def print_comparison_report(results: dict) -> None:
    """Print a formatted comparison report.

    Args:
        results: Dictionary from compare_hashes()
    """
    print("\n" + "=" * 60)
    print("FIGURE HASH COMPARISON REPORT")
    print("=" * 60)

    if results['unchanged']:
        print(f"\n✓ Unchanged ({len(results['unchanged'])} figures):")
        for name in sorted(results['unchanged']):
            print(f"  - {name}")

    if results['changed']:
        print(f"\n✗ Changed ({len(results['changed'])} figures):")
        for name, old_hash, new_hash in sorted(results['changed']):
            print(f"  - {name}")
            print(f"      old: {old_hash}")
            print(f"      new: {new_hash}")

    if results['new']:
        print(f"\n? New ({len(results['new'])} figures):")
        for name in sorted(results['new']):
            print(f"  - {name}")

    if results['missing']:
        print(f"\n! Missing ({len(results['missing'])} figures):")
        for name in sorted(results['missing']):
            print(f"  - {name}")

    print("\n" + "-" * 60)
    total = (len(results['unchanged']) + len(results['changed']) +
             len(results['new']) + len(results['missing']))
    print(f"Total: {total} figures tracked")

    if results['changed']:
        print("⚠️  Some figures have changed!")
    elif results['missing']:
        print("⚠️  Some figures are missing!")
    else:
        print("✓ All figures match baseline")

    print("=" * 60 + "\n")


def print_history_report(
    figure_name: str | None = None,
    history_db: str | Path = DEFAULT_HISTORY_DB,
    limit: int = 10
) -> None:
    """Print history for one or all figures.

    Args:
        figure_name: Name of a specific figure, or None for all
        history_db: Path to the history database JSON file
        limit: Maximum entries to show per figure
    """
    history = load_history_db(history_db)

    if not history:
        print("No history recorded yet.")
        print("Run notebooks with save_figure_with_hash() to start tracking.")
        return

    print("\n" + "=" * 70)
    print("FIGURE HASH HISTORY")
    print("=" * 70)

    figures_to_show = [figure_name] if figure_name else sorted(history.keys())

    for name in figures_to_show:
        if name not in history:
            print(f"\n{name}: No history found")
            continue

        entries = history[name]
        print(f"\n{name} ({len(entries)} entries):")
        print("-" * 60)

        # Show most recent entries (up to limit)
        for entry in entries[-limit:]:
            ts = entry.get('timestamp', 'unknown')
            data_hash = entry.get('data_hash', 'N/A')
            size = entry.get('size', 'N/A')
            if data_hash and len(data_hash) > 8:
                data_hash = data_hash[:8] + '...'
            print(f"  {ts}  data={data_hash}  size={size}")

    print("\n" + "=" * 70 + "\n")


def main():
    """Command-line interface for figure hashing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m graph2grav.figure_hashing [capture|compare|history]")
        print("")
        print("Commands:")
        print("  capture           - Capture current figure hashes as baseline")
        print("  compare           - Compare current figures against baseline")
        print("  history [name]    - Show hash history (optionally for one figure)")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'capture':
        print(f"Capturing hashes from {DEFAULT_FIGURES_DIR}...")
        db = capture_all_hashes()
        print(f"Captured hashes for {len(db)} figures")
        print(f"Saved to {DEFAULT_HASH_DB}")
        print(f"History logged to {DEFAULT_HISTORY_DB}")

    elif command == 'compare':
        print(f"Comparing figures in {DEFAULT_FIGURES_DIR}...")
        results = compare_hashes()
        print_comparison_report(results)

        # Exit with error code if changes detected
        if results['changed'] or results['missing']:
            sys.exit(1)

    elif command == 'history':
        figure_name = sys.argv[2] if len(sys.argv) > 2 else None
        print_history_report(figure_name)

    else:
        print(f"Unknown command: {command}")
        print("Use 'capture', 'compare', or 'history'")
        sys.exit(1)


if __name__ == '__main__':
    main()

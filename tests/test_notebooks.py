"""Test that all notebooks execute without errors.

This test runs all Jupyter notebooks in the repository to ensure they work
with the current codebase. Notebooks are executed but NOT modified.
"""

import sys
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def normalize_pattern(pattern):
    """Convert a pattern like '3', 'a1', 'A01' to a glob pattern.

    Examples:
        '3' -> '03_*.ipynb'
        '03' -> '03_*.ipynb'
        'a1' -> 'A01_*.ipynb'
        'A1' -> 'A01_*.ipynb'
    """
    pattern = pattern.lower()

    # Check if it's an appendix notebook (starts with 'a')
    if pattern.startswith('a'):
        # Remove 'a' prefix and get the number
        num_str = pattern[1:]
        try:
            num = int(num_str)
            return f"A{num:02d}_*.ipynb"
        except ValueError:
            # If can't parse, return pattern as-is with wildcard
            return f"{pattern}*.ipynb"
    else:
        # It's a main notebook number
        try:
            num = int(pattern)
            return f"{num:02d}_*.ipynb"
        except ValueError:
            # If can't parse, return pattern as-is with wildcard
            return f"{pattern}*.ipynb"


def find_notebooks(pattern=None):
    """Find notebook files in the repository root.

    Args:
        pattern: Optional pattern like '3', 'a1', etc. to match specific notebooks.
                 If None, returns all notebooks.
    """
    if pattern:
        glob_pattern = normalize_pattern(pattern)
        notebooks = sorted(repo_root.glob(glob_pattern))
    else:
        notebooks = sorted(repo_root.glob("*.ipynb"))

    # Filter out checkpoint files
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]
    return notebooks


def execute_notebook(notebook_path, show_progress=True):
    """Execute a notebook without saving changes.

    Uses nbconvert's ExecutePreprocessor to execute the notebook in memory.
    Returns True if successful, False otherwise.
    """
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from tqdm import tqdm
    except ImportError as e:
        missing_libs = []
        if 'nbformat' in str(e):
            missing_libs.append('nbformat')
        if 'nbconvert' in str(e):
            missing_libs.append('nbconvert')
        if 'tqdm' in str(e):
            missing_libs.append('tqdm')
        return False, f"Error: Required libraries not installed. Run: pip install {' '.join(missing_libs)}\nDetails: {e}"

    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Build a map of cell indices to their section headers
        cell_headers = {}
        current_header = "Initialization"

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown':
                # Extract first header from markdown cell
                lines = cell.source.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('#'):
                        # Extract header text (remove # symbols and clean up)
                        header_text = line.lstrip('#').strip()
                        if header_text:
                            current_header = header_text
                            break
            cell_headers[idx] = current_header

        # Count code cells for progress bar
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        total_cells = len(code_cells)

        # Create preprocessor with timeout
        ep = ExecutePreprocessor(timeout=300, kernel_name='python3')

        # Create progress bar if requested
        if show_progress and total_cells > 0:
            # Track code cell counter separately
            code_cell_counter = [0]  # Use list to allow modification in nested function

            pbar = tqdm(total=total_cells, desc=f"  {current_header[:40]}", leave=False,
                       bar_format='{desc}: {n_fmt}/{total_fmt} [{bar}] {percentage:3.0f}%')

            # Monkey-patch preprocess_cell to update progress
            original_preprocess_cell = ep.preprocess_cell

            def preprocess_cell_with_progress(cell, resources, cell_index):
                # Update progress bar description with current section
                if cell_index in cell_headers:
                    header = cell_headers[cell_index]
                    # Truncate long headers
                    if len(header) > 40:
                        header = header[:37] + "..."
                    pbar.set_description(f"  {header}")

                result = original_preprocess_cell(cell, resources, cell_index)

                if cell.cell_type == 'code':
                    code_cell_counter[0] += 1
                    pbar.update(1)

                return result

            ep.preprocess_cell = preprocess_cell_with_progress

        # Execute the notebook (in memory, doesn't modify file)
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})

        if show_progress and total_cells > 0:
            pbar.close()

        return True, None

    except TimeoutError:
        return False, "Timeout: Notebook took longer than 5 minutes to execute"
    except Exception as e:
        # Extract full error information including traceback
        import traceback
        error_msg = f"{str(e)}\n\nFull traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"

        # If it's a CellExecutionError, include the cell info
        if hasattr(e, 'traceback'):
            error_msg = f"Cell execution error:\n{e.traceback}"

        return False, error_msg


def test_all_notebooks(pattern=None):
    """Test that all notebooks execute successfully.

    Args:
        pattern: Optional pattern to match specific notebooks (e.g., '3', 'a1').
    """
    print("=" * 70)
    if pattern:
        print(f"Testing notebook matching pattern: {pattern}")
    else:
        print("Testing notebook execution...")
    print("=" * 70)

    notebooks = find_notebooks(pattern)

    if not notebooks:
        if pattern:
            print(f"No notebooks found matching pattern: {pattern}")
            print(f"  Looking for: {normalize_pattern(pattern)}")
        else:
            print("No notebooks found in repository root")
        return 0

    print(f"\nFound {len(notebooks)} notebook(s) to test:\n")

    results = {}

    for notebook in notebooks:
        notebook_name = notebook.name
        print(f"Executing: {notebook_name}...")

        success, error = execute_notebook(notebook, show_progress=True)
        results[notebook_name] = (success, error)

        if success:
            print(f"  ✓ {notebook_name} completed successfully")
        else:
            print(f"  ✗ {notebook_name} failed")
            if error:
                # Print error with indentation
                print(f"\n  Error in {notebook_name}:")
                print("  " + "-" * 66)
                for line in str(error).split('\n'):
                    print(f"  {line}")
                print("  " + "-" * 66 + "\n")

    # Summary
    print("\n" + "=" * 70)
    successful = sum(1 for success, _ in results.values() if success)
    total = len(results)

    print(f"Results: {successful}/{total} notebooks executed successfully\n")

    if successful < total:
        print("Failed notebooks:")
        for name, (success, error) in results.items():
            if not success:
                print(f"  ✗ {name}")
                if error and "pip install" in error:
                    print(f"    {error.split('Details:')[0].strip()}")
        return 1
    else:
        print("✓ All notebooks executed successfully!")
        return 0


def test_notebooks_quick_check():
    """Quick test that just checks if notebooks can be found and dependencies are available."""
    print("=" * 70)
    print("Quick notebook check (dependency verification)...")
    print("=" * 70)

    # Check if required libraries are available
    try:
        import nbformat
        import nbconvert
        from nbconvert.preprocessors import ExecutePreprocessor
        import tqdm
        print("✓ All required libraries available")
        print(f"  nbconvert version: {nbconvert.__version__}")
        print(f"  nbformat version: {nbformat.__version__}")
        print(f"  tqdm version: {tqdm.__version__}")
    except ImportError as e:
        print("✗ Required libraries not installed")
        print(f"  Install with: pip install nbconvert nbformat tqdm")
        print(f"  Error: {e}")
        return 1

    # Check for notebooks
    notebooks = find_notebooks()
    if notebooks:
        print(f"✓ Found {len(notebooks)} notebook(s)")
        for nb in notebooks:
            print(f"  - {nb.name}")
    else:
        print("⚠ No notebooks found in repository root")

    print("\n" + "=" * 70)
    print("To run full notebook tests: python tests/test_notebooks.py")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    # Check if --quick flag is provided
    if "--quick" in sys.argv:
        sys.exit(test_notebooks_quick_check())

    # Check for pattern argument (any non-flag argument)
    pattern = None
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            pattern = arg
            break

    sys.exit(test_all_notebooks(pattern))

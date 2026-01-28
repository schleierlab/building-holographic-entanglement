"""Test that all modules can be imported without errors.

This test catches missing imports, undefined functions, and syntax errors.
"""

import sys
from pathlib import Path
import importlib

# Add parent directory to path to import graph2grav
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


# Known external dependencies
EXTERNAL_DEPS = {
    'networkx', 'numpy', 'scipy', 'matplotlib', 'mpmath',
    'tqdm', 'pydantic', 'pygraphviz'
}


def categorize_import_error(error):
    """Categorize import errors into dependency vs code issues."""
    error_str = str(error)

    # Check if it's a missing external dependency
    for dep in EXTERNAL_DEPS:
        if f"No module named '{dep}'" in error_str:
            return "missing_dependency", dep
        if f"No module named \"{dep}\"" in error_str:
            return "missing_dependency", dep

    # Check for missing internal imports
    if "cannot import name" in error_str:
        return "code_issue", error_str

    # Check for attribute errors (function doesn't exist)
    if isinstance(error, AttributeError):
        return "code_issue", error_str

    # Other import errors are likely code issues
    if "ImportError" in str(type(error)):
        return "code_issue", error_str

    return "unknown", error_str


def test_module_import(module_name):
    """Test importing a single module and categorize any errors."""
    try:
        importlib.import_module(module_name)
        return "success", None, None
    except Exception as e:
        category, detail = categorize_import_error(e)
        return category, detail, e


def discover_modules():
    """Automatically discover all Python modules in graph2grav package."""
    graph2grav_dir = repo_root / "graph2grav"
    modules = ["graph2grav"]  # Start with the package itself

    # Find all .py files in the graph2grav directory
    for py_file in sorted(graph2grav_dir.glob("*.py")):
        # Skip __init__.py and any private/test files
        if py_file.name.startswith("_") or py_file.name.startswith("test_"):
            continue

        # Convert filename to module name (remove .py extension)
        module_name = py_file.stem
        modules.append(f"graph2grav.{module_name}")

    return modules


def run_all_tests():
    """Run all import tests and report results."""
    print("=" * 70)
    print("Testing module imports...")
    print("=" * 70)

    modules = discover_modules()
    print(f"\nDiscovered {len(modules)} modules to test\n")

    results = {}
    missing_deps = set()
    code_issues = []

    for module in modules:
        status, detail, error = test_module_import(module)
        results[module] = (status, detail, error)

        if status == "missing_dependency":
            missing_deps.add(detail)
        elif status == "code_issue":
            code_issues.append((module, detail))

    # Print results
    print("\nImport Status:")
    print("-" * 70)
    for module, (status, detail, error) in results.items():
        if status == "success":
            print(f"✓ {module}")
        elif status == "missing_dependency":
            print(f"⚠ {module} (missing dependency: {detail})")
        elif status == "code_issue":
            print(f"✗ {module}")
            print(f"  Issue: {detail}")
        else:
            print(f"? {module}")
            print(f"  Unknown error: {detail}")

    # Summary
    print("\n" + "=" * 70)
    success_count = sum(1 for _, (s, _, _) in results.items() if s == "success")
    dep_count = len([m for m, (s, _, _) in results.items() if s == "missing_dependency"])
    issue_count = len(code_issues)

    print(f"Summary: {success_count}/{len(modules)} modules imported successfully")

    if missing_deps:
        print(f"\n⚠ Missing dependencies: {', '.join(sorted(missing_deps))}")
        print("  (Install with: pip install " + " ".join(sorted(missing_deps)) + ")")

    if code_issues:
        print(f"\n✗ {issue_count} code issue(s) found:")
        for module, detail in code_issues:
            print(f"  - {module}: {detail}")
        return 1

    if success_count == len(modules):
        print("\n✓ All modules imported successfully!")
        return 0
    elif issue_count == 0:
        print("\n⚠ All failures are due to missing dependencies (not code issues)")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

"""Every routes entry script disables bytecode writes BEFORE it imports the package.

Python writes ``.pyc`` files into a ``__pycache__`` beside each source module it imports when the
cache is missing or stale. For an admission flight that is a write into a source tree the
run-root scan and the source-cache check do not cover (sol, 2026-10-08: a ``__pycache__`` junctioned
into a pinned origin flight would receive them at import time). Two scripts already set
``sys.dont_write_bytecode = True``; this makes it the rule for every entry script, so a new one
cannot be added without it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
ENTRY_SCRIPTS = sorted(SCRIPTS.glob("her2_routes_*.py"))


def _first_package_import_line(tree):
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("smallAntibodyGen"):
            lines.append(node.lineno)
        elif isinstance(node, ast.Import):
            lines.extend(node.lineno for alias in node.names if alias.name.startswith("smallAntibodyGen"))
    return min(lines) if lines else None


def _bytecode_switch_line(tree):
    for node in tree.body:                                   # module level only
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and ast.unparse(node.targets[0]) == "sys.dont_write_bytecode"
                and isinstance(node.value, ast.Constant) and node.value.value is True):
            return node.lineno
    return None


def test_the_entry_scripts_exist():
    assert len(ENTRY_SCRIPTS) >= 10, ENTRY_SCRIPTS


@pytest.mark.parametrize("script", ENTRY_SCRIPTS, ids=lambda path: path.name)
def test_a_script_that_imports_the_package_disables_bytecode_first(script):
    tree = ast.parse(script.read_text(encoding="utf-8"))
    first_import = _first_package_import_line(tree)
    if first_import is None:
        pytest.skip(f"{script.name} does not import the package")
    switch = _bytecode_switch_line(tree)
    assert switch is not None, (
        f"{script.name} imports smallAntibodyGen (line {first_import}) without setting "
        "sys.dont_write_bytecode = True at module level")
    assert switch < first_import, (
        f"{script.name} sets sys.dont_write_bytecode at line {switch}, after its first package "
        f"import at line {first_import}; the modules imported before it already wrote bytecode")

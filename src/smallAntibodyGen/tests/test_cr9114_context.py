"""Guard against joining absent polymer residues during context preparation."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def context_script():
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    spec = importlib.util.spec_from_file_location("cr9114_context", scripts / "prepare_cr9114_5cjq_context.py")
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(scripts))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def test_missing_internal_polymer_positions_remain_chain_breaks(context_script):
    assert context_script.contiguous_runs([5, 6, 7, 10, 11, 20]) == [[5, 6, 7], [10, 11], [20]]


@pytest.mark.parametrize("indices", [[], [1, 1], [2, 1], [1, 3, 2]])
def test_ambiguous_polymer_order_is_rejected(context_script, indices):
    with pytest.raises(ValueError, match="unique and increasing"):
        context_script.contiguous_runs(indices)


def test_existing_output_is_not_overwritten(context_script, tmp_path):
    pytest.importorskip("biotite")
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        context_script.prepare(tmp_path, tmp_path)
    assert sentinel.read_text() == "preserve"

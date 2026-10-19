"""The campaign source identity must cover every Python file under its pinned locations.

``inputs.source_identity`` globs ``*.py`` non-recursively.  That is complete only while
the pinned directories have no nested subpackage; this test enumerates recursively and
compares, so a nested module (or a new flat one that the glob would miss for any reason)
fails here instead of silently escaping the identity that checkpoints record.
"""
from __future__ import annotations

from pathlib import Path

from smallAntibodyGen.experiments.trastuzumab_poc import inputs
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest, file_digest

PINNED_DIRECTORIES = ("experiments/trastuzumab_poc", "models/antibody_antigen")
PINNED_FILES = ("models/dplm_policy.py",)
FUNCTIONAL_MODULES = ("functional_data.py", "functional_protocol.py", "functional_train.py",
                      "functional_checkpoint.py", "post_budget.py")


def package_root() -> Path:
    return Path(inputs.__file__).resolve().parents[2]


def recursive_inventory() -> set:
    package = package_root()
    found = {path.relative_to(package).as_posix()
             for relative in PINNED_DIRECTORIES
             for path in (package / relative).rglob("*.py")}
    found.update(PINNED_FILES)
    return found


def test_source_identity_lists_exactly_the_python_files_found_recursively():
    files = inputs.source_identity()["files"]
    found = recursive_inventory()
    assert not found - set(files), f"python files missing from source_identity: {sorted(found - set(files))}"
    assert not set(files) - found, f"source_identity lists files that do not exist: {sorted(set(files) - found)}"


def test_the_functional_modules_are_inventoried():
    files = inputs.source_identity()["files"]
    for name in FUNCTIONAL_MODULES:
        assert f"experiments/trastuzumab_poc/{name}" in files
    assert recursive_inventory() >= {f"experiments/trastuzumab_poc/{name}" for name in FUNCTIONAL_MODULES}


def test_inventory_digests_are_the_current_file_bytes():
    identity = inputs.source_identity()
    package = package_root()
    for relative, recorded in identity["files"].items():
        assert file_digest(package / relative) == recorded, relative
    assert identity["sha256"] == digest(identity["files"])
    assert identity["schema"] == "trastuzumab-source/1"

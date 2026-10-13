from __future__ import annotations

import ast
import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from smallAntibodyGen.experiments.interaction_qualification import config as config_module
from smallAntibodyGen.experiments.interaction_qualification.config import SchemaError, load_config, parse_config

GOOD = {
    "schema": "aa-qualification/1",
    "qualification_id": "mint-pilot",
    "max_tokens": 1024,
    "ridge_alpha": 1.0,
    "evaluate_test": False,
    "diagnostics": {
        "seed": 0,
        "partitions": ["development"],
        "mask_counts": [1, 2, 3, "full"],
        "geometries": ["suffix", "subset"],
        "max_examples": 32,
        "max_forwards": 20000,
        "local_gap": {"enabled": True, "max_pairs": 32, "kappa": 2.0},
    },
}


def doc(path=(), value=None, remove=False):
    out = copy.deepcopy(GOOD)
    if path:
        node = out
        for key in path[:-1]:
            node = node[key]
        if remove:
            del node[path[-1]]
        else:
            node[path[-1]] = value
    return out


def parsed(document):
    return parse_config(json.dumps(document))


def test_config_module_is_pure_stdlib():
    tree = ast.parse(Path(config_module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module or "")
    assert names and all(name.split(".")[0] in sys.stdlib_module_names for name in names), names


def test_the_reference_config_parses_to_immutable_typed_values():
    config = parsed(GOOD)
    assert config.qualification_id == "mint-pilot"
    assert config.max_tokens == 1024 and config.ridge_alpha == 1.0 and config.evaluate_test is False
    diag = config.diagnostics
    assert diag.seed == 0 and diag.partitions == ("development",)
    assert diag.mask_counts == (1, 2, 3, "full") and diag.geometries == ("suffix", "subset")
    assert diag.max_examples == 32 and diag.max_forwards == 20000
    assert diag.local_gap.enabled is True and diag.local_gap.max_pairs == 32 and diag.local_gap.kappa == 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_tokens = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        diag.seed = 1


def test_as_dict_round_trips_and_the_hash_ignores_key_order():
    config = parsed(GOOD)
    assert config.as_dict() == GOOD
    assert parsed(config.as_dict()) == config
    reordered = json.dumps(GOOD, sort_keys=True)
    assert parse_config(reordered).content_sha256() == config.content_sha256()
    changed = parsed(doc(("ridge_alpha",), 2.0))
    assert changed.content_sha256() != config.content_sha256()


def test_integer_alpha_and_kappa_are_accepted_as_reals_but_stored_as_floats():
    config = parsed(doc(("ridge_alpha",), 2))
    assert config.ridge_alpha == 2.0 and isinstance(config.ridge_alpha, float)
    kappa = parsed(doc(("diagnostics", "local_gap", "kappa"), 3)).diagnostics.local_gap.kappa
    assert kappa == 3.0 and isinstance(kappa, float)


def test_test_partition_diagnostics_need_explicit_test_evaluation():
    wants_test = doc(("diagnostics", "partitions"), ["development", "test"])
    with pytest.raises(SchemaError, match="test"):
        parsed(wants_test)
    wants_test["evaluate_test"] = True
    assert parsed(wants_test).diagnostics.partitions == ("development", "test")
    # evaluating test without asking for test diagnostics is fine and changes nothing else
    assert parsed(doc(("evaluate_test",), True)).diagnostics.partitions == ("development",)


BAD_VALUES = [
    (("schema",), "aa-qualification/2"),
    (("qualification_id",), ""),
    (("qualification_id",), 7),
    (("qualification_id",), " padded"),
    (("max_tokens",), 0),
    (("max_tokens",), -5),
    (("max_tokens",), True),
    (("max_tokens",), 1024.0),
    (("max_tokens",), "1024"),
    (("ridge_alpha",), 0),
    (("ridge_alpha",), -1.0),
    (("ridge_alpha",), True),
    (("ridge_alpha",), "1.0"),
    (("ridge_alpha",), None),
    (("evaluate_test",), 0),
    (("evaluate_test",), "false"),
    (("evaluate_test",), None),
    (("diagnostics", "seed"), -1),
    (("diagnostics", "seed"), True),
    (("diagnostics", "seed"), 0.0),
    (("diagnostics", "max_examples"), 0),
    (("diagnostics", "max_examples"), 1.5),
    (("diagnostics", "max_forwards"), 0),
    (("diagnostics", "max_forwards"), True),
    (("diagnostics", "partitions"), []),
    (("diagnostics", "partitions"), "development"),
    (("diagnostics", "partitions"), ["development", "development"]),
    (("diagnostics", "partitions"), ["validation"]),
    (("diagnostics", "partitions"), [["development"]]),
    (("diagnostics", "geometries"), []),
    (("diagnostics", "geometries"), ["prefix"]),
    (("diagnostics", "geometries"), ["suffix", "suffix"]),
    (("diagnostics", "mask_counts"), []),
    (("diagnostics", "mask_counts"), [1, 1]),
    (("diagnostics", "mask_counts"), ["full", "full"]),
    (("diagnostics", "mask_counts"), [0]),
    (("diagnostics", "mask_counts"), [-1]),
    (("diagnostics", "mask_counts"), [True]),
    (("diagnostics", "mask_counts"), [1.5]),
    (("diagnostics", "mask_counts"), [2.0]),
    (("diagnostics", "mask_counts"), ["all"]),
    (("diagnostics", "mask_counts"), [None]),
    (("diagnostics", "local_gap", "enabled"), 1),
    (("diagnostics", "local_gap", "enabled"), "true"),
    (("diagnostics", "local_gap", "max_pairs"), 0),
    (("diagnostics", "local_gap", "max_pairs"), True),
    (("diagnostics", "local_gap", "kappa"), 0),
    (("diagnostics", "local_gap", "kappa"), -2.0),
    (("diagnostics", "local_gap", "kappa"), True),
    (("diagnostics", "local_gap", "kappa"), "2.0"),
    (("diagnostics",), []),
    (("diagnostics", "local_gap"), True),
]


@pytest.mark.parametrize("path,value", BAD_VALUES, ids=[f"{'.'.join(p)}={v!r}" for p, v in BAD_VALUES])
def test_bad_values_are_rejected(path, value):
    with pytest.raises(SchemaError):
        parsed(doc(path, value))


@pytest.mark.parametrize(
    "path",
    [
        ("schema",),
        ("qualification_id",),
        ("max_tokens",),
        ("ridge_alpha",),
        ("evaluate_test",),
        ("diagnostics",),
        ("diagnostics", "seed"),
        ("diagnostics", "partitions"),
        ("diagnostics", "mask_counts"),
        ("diagnostics", "geometries"),
        ("diagnostics", "max_examples"),
        ("diagnostics", "max_forwards"),
        ("diagnostics", "local_gap"),
        ("diagnostics", "local_gap", "enabled"),
        ("diagnostics", "local_gap", "max_pairs"),
        ("diagnostics", "local_gap", "kappa"),
    ],
    ids=".".join,
)
def test_every_key_is_required(path):
    with pytest.raises(SchemaError, match="missing"):
        parsed(doc(path, remove=True))


@pytest.mark.parametrize(
    "path,key",
    [((), "data_path"), ((), "checkpoint"), (("diagnostics",), "device"), (("diagnostics", "local_gap"), "mode")],
)
def test_the_schema_is_closed_so_paths_and_overrides_have_no_place(path, key):
    document = doc()
    node = document
    for step in path:
        node = node[step]
    node[key] = "x"
    with pytest.raises(SchemaError, match="unexpected"):
        parsed(document)


def test_a_non_object_root_is_rejected():
    for text in ("[]", "1", "null", '"x"'):
        with pytest.raises(SchemaError):
            parse_config(text)


def test_strict_json_refuses_duplicate_keys_and_non_finite_numbers():
    text = json.dumps(GOOD)
    with pytest.raises(SchemaError, match="duplicate"):
        parse_config(text[:-1] + ', "max_tokens": 2048}')
    with pytest.raises(SchemaError, match="duplicate"):
        parse_config(text.replace('"seed": 0', '"seed": 0, "seed": 1'))
    for token in ("NaN", "Infinity", "-Infinity", "1e999"):
        with pytest.raises(SchemaError):
            parse_config(text.replace('"ridge_alpha": 1.0', f'"ridge_alpha": {token}'))
    with pytest.raises(SchemaError):
        parse_config("{not json")


def test_load_config_reads_exactly_the_given_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(json.dumps(GOOD).encode("utf-8"))
    assert load_config(path) == parsed(GOOD)
    with pytest.raises(SchemaError, match="cannot read"):
        load_config(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(SchemaError, match="UTF-8"):
        load_config(bad)

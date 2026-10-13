from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smallAntibodyGen.experiments.interaction_qualification import config as config_module
from smallAntibodyGen.experiments.interaction_qualification import records as rec
from smallAntibodyGen.experiments.interaction_qualification.config import SchemaError
from smallAntibodyGen.tests import aa_qualification_world as world

INTERNAL_ALLOWED = {
    "smallAntibodyGen.models.antibody_antigen.views",
    "smallAntibodyGen.experiments.interaction_qualification.config",
}


def imported_names(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module or "")
    return names


# --- imports ---------------------------------------------------------------------------------


def test_records_import_only_the_standard_library_views_and_config():
    for name in imported_names(rec):
        assert name.split(".")[0] in sys.stdlib_module_names or name in INTERNAL_ALLOWED, name


def test_admission_modules_load_with_nothing_but_the_standard_library(tmp_path):
    """Run with ``-S`` (no site-packages): importing config and records must not need torch/numpy/scipy."""
    src_root = Path(config_module.__file__).parents[3]
    code = (
        "import sys\n"
        "import smallAntibodyGen.experiments.interaction_qualification.config\n"
        "import smallAntibodyGen.experiments.interaction_qualification.records\n"
        "loaded = {'torch', 'numpy', 'scipy', 'pandas'} & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ, PYTHONPATH=str(src_root), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=str(tmp_path), env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- one record ------------------------------------------------------------------------------


def base_row(**changes):
    item = world.row(
        "r1", heavy="ACD", light="EF", antigen="GHI", value=7, region=[["H", 0], ["H", 2], ["L", 1]]
    )
    item.update(changes)
    return item


def test_a_valid_record_is_normalized_and_immutable():
    record = rec.parse_record(base_row())
    assert record.pkd == 7.0 and isinstance(record.pkd, float)
    assert record.light == "EF"
    assert record.region == (("H", 0), ("H", 2), ("L", 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.pkd = 1.0
    assert rec.parse_record(record.to_row()) == record
    assert rec.parse_record(json.loads(json.dumps(record.to_row()))).content_sha256 == record.content_sha256


def test_light_may_be_null_and_region_may_be_null_or_absent():
    heavy_only = base_row(light=None, region=None)
    record = rec.parse_record(heavy_only)
    assert record.light is None and record.region is None
    absent = copy.deepcopy(heavy_only)
    del absent["region"]
    assert rec.parse_record(absent).content_sha256 == record.content_sha256


def test_content_hash_binds_values_but_not_spelling():
    reference = rec.parse_record(base_row())
    assert rec.parse_record(base_row(measurement={"kind": "exact_pkd", "value": 7.0, "measured": True})) == reference
    assert (
        rec.parse_record(base_row(measurement={"kind": "exact_pkd", "value": 7.0, "measured": True})).content_sha256
        == reference.content_sha256
    )
    for changes in (
        {"measurement": {"kind": "exact_pkd", "value": 7.1, "measured": True}},
        {"record_id": "r2"},
        {"heavy": "ACE"},
        {"source": "study2"},
        {"region": [["H", 0]]},
    ):
        assert rec.parse_record(base_row(**changes)).content_sha256 != reference.content_sha256


BAD_ROWS = {
    "wrong schema": {"schema": "aa-record/2"},
    "empty heavy": {"heavy": ""},
    "lowercase heavy": {"heavy": "acd"},
    "non-canonical heavy": {"heavy": "ACX"},
    "heavy not a string": {"heavy": 5},
    "null heavy": {"heavy": None},
    "empty light": {"light": ""},
    "lowercase light": {"light": "ef"},
    "numeric light": {"light": 1},
    "non-canonical antigen": {"antigen": "GHB"},
    "empty antigen": {"antigen": ""},
    "null antigen": {"antigen": None},
    "empty record id": {"record_id": ""},
    "numeric record id": {"record_id": 1},
    "padded source": {"source": " study1"},
    "empty assay": {"assay": ""},
    "empty antibody group": {"antibody_group": ""},
    "null antigen group": {"antigen_group": None},
    "unknown partition": {"partition": "validation"},
    "other assay kind": {"measurement": {"kind": "kd", "value": 7.3, "measured": True}},
    "string value": {"measurement": {"kind": "exact_pkd", "value": "7.3", "measured": True}},
    "bool value": {"measurement": {"kind": "exact_pkd", "value": True, "measured": True}},
    "null value": {"measurement": {"kind": "exact_pkd", "value": None, "measured": True}},
    "measured false": {"measurement": {"kind": "exact_pkd", "value": 7.3, "measured": False}},
    "measured string": {"measurement": {"kind": "exact_pkd", "value": 7.3, "measured": "true"}},
    "measured integer": {"measurement": {"kind": "exact_pkd", "value": 7.3, "measured": 1}},
    "measurement extra key": {"measurement": {"kind": "exact_pkd", "value": 7.3, "measured": True, "unit": "M"}},
    "measurement missing key": {"measurement": {"kind": "exact_pkd", "value": 7.3}},
    "measurement not an object": {"measurement": 7.3},
    "extra field": {"censored": False},
    "empty region": {"region": []},
    "duplicate region coordinate": {"region": [["H", 0], ["H", 0]]},
    "unsorted region": {"region": [["H", 2], ["H", 0]]},
    "light before heavy": {"region": [["L", 0], ["H", 0]]},
    "antigen in region": {"region": [["Ag", 0]]},
    "heavy coordinate out of bounds": {"region": [["H", 3]]},
    "light coordinate out of bounds": {"region": [["L", 2]]},
    "float coordinate index": {"region": [["H", 1.0]]},
    "bool coordinate index": {"region": [["H", True]]},
    "negative coordinate index": {"region": [["H", -1]]},
    "unknown region role": {"region": [["X", 0]]},
    "short region pair": {"region": [["H"]]},
    "region object": {"region": {"H": [0]}},
    "region of strings": {"region": ["H0"]},
}


@pytest.mark.parametrize("changes", list(BAD_ROWS.values()), ids=list(BAD_ROWS))
def test_closed_record_schema_rejects(changes):
    with pytest.raises(SchemaError):
        rec.parse_record(base_row(**changes))


@pytest.mark.parametrize("key", rec._RECORD_KEYS)
def test_every_record_key_but_region_is_required(key):
    item = base_row()
    del item[key]
    with pytest.raises(SchemaError, match="missing"):
        rec.parse_record(item)


def test_a_light_coordinate_is_refused_when_there_is_no_light_chain():
    with pytest.raises(SchemaError, match="outside chain L"):
        rec.parse_record(base_row(light=None, region=[["L", 0]]))


def test_direct_construction_validates_like_parsing():
    with pytest.raises(SchemaError):
        rec.Record("r1", "ACD", "", "GHI", "g", "a", "train", "s", "a", 7.0)
    with pytest.raises(SchemaError):
        rec.Record("r1", "ACD", None, "GHI", "g", "a", "train", "s", "a", True)
    with pytest.raises(SchemaError):
        rec.Record("r1", "ACD", None, "GHI", "g", "a", "train", "s", "a", 7.0, region=[("H", 0)])


# --- JSONL -----------------------------------------------------------------------------------


def test_an_empty_file_is_an_empty_panel_and_the_final_newline_is_optional():
    assert rec.parse_records(b"") == ()
    line = json.dumps(base_row())
    assert len(rec.parse_records(line.encode("utf-8"))) == 1
    assert len(rec.parse_records((line + "\n").encode("utf-8"))) == 1
    assert len(rec.parse_records((line + "\r\n" + line.replace("r1", "r2") + "\r\n").encode("utf-8"))) == 2


def test_blank_lines_and_bad_rows_reject_the_whole_file_with_a_line_number():
    line = json.dumps(base_row())
    other = json.dumps(base_row(record_id="r2"))
    with pytest.raises(SchemaError, match="line 2"):
        rec.parse_records((line + "\n\n" + other + "\n").encode("utf-8"))
    with pytest.raises(SchemaError, match="line 1"):
        rec.parse_records(b"\n")
    with pytest.raises(SchemaError, match="line 2"):
        rec.parse_records((line + "\n" + json.dumps(base_row(heavy="ACX")) + "\n").encode("utf-8"))
    with pytest.raises(SchemaError, match="line 1"):
        rec.parse_records(b"[1, 2]\n")


def test_jsonl_is_strict_about_duplicate_keys_and_non_finite_numbers():
    line = json.dumps(base_row())
    with pytest.raises(SchemaError, match="duplicate"):
        rec.parse_records((line[:-1] + ', "assay": "assay2"}\n').encode("utf-8"))
    for token in ("NaN", "Infinity", "-Infinity", "1e999"):
        with pytest.raises(SchemaError):
            rec.parse_records(line.replace('"value": 7', f'"value": {token}').encode("utf-8"))
    with pytest.raises(SchemaError, match="UTF-8"):
        rec.parse_records(b"\xff\xfe\n")


# --- identities ------------------------------------------------------------------------------


def test_fingerprints_frame_chain_roles_and_lengths():
    assert rec.antibody_fingerprint("AC", "D") != rec.antibody_fingerprint("A", "CD")
    assert rec.pair_fingerprint("AC", "D", "EFG") != rec.pair_fingerprint("A", "CD", "EFG")
    assert rec.pair_fingerprint("AC", None, "DEFG") != rec.pair_fingerprint("ACD", None, "EFG")
    assert rec.pair_fingerprint("AC", None, "DE") != rec.pair_fingerprint("AC", "D", "E")
    assert rec.antibody_fingerprint("ACD", None) != rec.antibody_fingerprint("ACD", "A")
    assert rec.antibody_fingerprint("ACD", None) == rec.antibody_fingerprint("ACD", None)
    assert rec.antigen_fingerprint("EFG") != rec.antigen_fingerprint("EFH")


def test_fingerprints_ignore_every_identifier_and_value():
    first = rec.parse_record(base_row())
    renamed = rec.parse_record(
        base_row(record_id="zzz", source="s9", assay="a9", antibody_group="g1", antigen_group="g2", partition="test")
    )
    assert (first.antibody_fingerprint, first.antigen_fingerprint, first.pair_fingerprint) == (
        renamed.antibody_fingerprint,
        renamed.antigen_fingerprint,
        renamed.pair_fingerprint,
    )


# --- manifest --------------------------------------------------------------------------------


def manifest_doc(**changes):
    item = {
        "schema": "aa-manifest/1",
        "records_file": "panel.jsonl",
        "records_sha256": "0" * 64,
        "sources": {"study1": {"citation": "synthetic citation", "assays": {"assay1": "synthetic exact pKD"}}},
        "split_regime": "both",
        "split_basis": "synthetic assignment",
    }
    item.update(changes)
    return item


def test_a_valid_manifest_parses():
    manifest = rec.parse_manifest(json.dumps(manifest_doc()))
    assert manifest.records_file == "panel.jsonl" and manifest.split_regime == "both"
    assert manifest.declared_assays() == {("study1", "assay1"): "synthetic exact pKD"}
    assert manifest.sources[0].citation == "synthetic citation"


BAD_MANIFESTS = {
    "wrong schema": {"schema": "aa-manifest/2"},
    "empty records file": {"records_file": ""},
    "numeric records file": {"records_file": 3},
    "uppercase hash": {"records_sha256": "A" * 64},
    "short hash": {"records_sha256": "0" * 63},
    "non-hex hash": {"records_sha256": "g" * 64},
    "hash with newline": {"records_sha256": "0" * 64 + "\n"},
    "sources as a list": {"sources": []},
    "source without citation": {"sources": {"study1": {"assays": {"assay1": "d"}}}},
    "source with extra key": {"sources": {"study1": {"citation": "c", "assays": {"assay1": "d"}, "url": "u"}}},
    "blank citation": {"sources": {"study1": {"citation": " ", "assays": {"assay1": "d"}}}},
    "source without assays": {"sources": {"study1": {"citation": "c", "assays": {}}}},
    "blank assay description": {"sources": {"study1": {"citation": "c", "assays": {"assay1": ""}}}},
    "unknown regime": {"split_regime": "random"},
    "blank split basis": {"split_basis": "  "},
    "extra key": {"dataset_dir": "somewhere"},
}


@pytest.mark.parametrize("changes", list(BAD_MANIFESTS.values()), ids=list(BAD_MANIFESTS))
def test_closed_manifest_schema_rejects(changes):
    with pytest.raises(SchemaError):
        rec.parse_manifest(json.dumps(manifest_doc(**changes)))


@pytest.mark.parametrize("key", rec._MANIFEST_KEYS)
def test_every_manifest_key_is_required(key):
    item = manifest_doc()
    del item[key]
    with pytest.raises(SchemaError, match="missing"):
        rec.parse_manifest(json.dumps(item))


def test_manifest_json_is_strict():
    text = json.dumps(manifest_doc())
    with pytest.raises(SchemaError, match="duplicate"):
        rec.parse_manifest(text[:-1] + ', "split_regime": "both"}')
    with pytest.raises(SchemaError):
        rec.parse_manifest(text.replace('"both"', "NaN"))


# --- loading from disk -----------------------------------------------------------------------


def small_rows():
    return world.grid("train", ["ACD", "ACE"], ["EFG"]) + world.grid("development", ["ACF"], ["EFH"])


def test_load_manifest_reads_the_declared_file_next_to_the_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "decoy.jsonl").write_bytes(b"this is never read\n")
    path = world.write_panel(bundle, small_rows())
    panel = rec.load_manifest(path)
    assert len(panel.records) == 3
    assert panel.records_file_sha256 == hashlib.sha256((bundle / "panel.jsonl").read_bytes()).hexdigest()
    assert panel.manifest.split_regime == "both"


def test_load_manifest_accepts_an_absolute_records_path(tmp_path):
    rows = small_rows()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    data = world.jsonl_bytes(rows)
    (elsewhere / "data.jsonl").write_bytes(data)
    manifest = world.manifest_dict(
        rows, hashlib.sha256(data).hexdigest(), records_file=str(elsewhere / "data.jsonl")
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(json.dumps(manifest).encode("utf-8"))
    assert len(rec.load_manifest(path).records) == 3


def test_the_file_hash_is_checked_before_the_file_is_parsed(tmp_path):
    path = world.write_panel(tmp_path, small_rows())
    (tmp_path / "panel.jsonl").write_bytes(b"this is not json\n")
    with pytest.raises(SchemaError, match="records_sha256") as info:
        rec.load_manifest(path)
    assert "invalid JSON" not in str(info.value)


def test_a_changed_byte_is_a_hash_mismatch(tmp_path):
    path = world.write_panel(tmp_path, small_rows())
    data = (tmp_path / "panel.jsonl").read_bytes()
    (tmp_path / "panel.jsonl").write_bytes(data.replace(b"7.0", b"7.5", 1))
    with pytest.raises(SchemaError, match="records_sha256"):
        rec.load_manifest(path)


def test_a_missing_records_file_or_manifest_is_a_schema_error(tmp_path):
    path = world.write_panel(tmp_path, small_rows())
    (tmp_path / "panel.jsonl").unlink()
    with pytest.raises(SchemaError, match="cannot read"):
        rec.load_manifest(path)
    with pytest.raises(SchemaError, match="cannot read"):
        rec.load_manifest(tmp_path / "absent.json")


# --- panel: order, duplicates, provenance ----------------------------------------------------


def mixed_rows():
    return (
        world.grid("train", ["ACD", "ACE", "ACF"], ["EFG", "EFH"])
        + world.grid("development", ["ACG"], ["EFI", "EFK"])
        + world.grid("test", ["ACH"], ["EFL"])
    )


def test_panel_order_ignores_row_order_and_record_ids():
    rows = mixed_rows()
    forward = world.panel_of(rows)
    backward = world.panel_of(rows[::-1])
    assert [r.record_id for r in forward.records] == [r.record_id for r in backward.records]
    assert forward.content_sha256 == backward.content_sha256
    assert forward.records_file_sha256 != backward.records_file_sha256
    renamed = [dict(item, record_id=f"id-{len(rows) - i:03d}") for i, item in enumerate(rows)]
    other = world.panel_of(renamed)
    assert [r.pair_fingerprint for r in other.records] == [r.pair_fingerprint for r in forward.records]
    assert [r.record_id for r in other.records] != sorted(r.record_id for r in other.records)


def test_panel_is_sorted_by_partition_then_fingerprints_source_assay():
    panel = world.panel_of(mixed_rows())
    rank = {"train": 0, "development": 1, "test": 2}
    keys = [
        (rank[r.partition], r.antibody_fingerprint, r.antigen_fingerprint, r.source, r.assay) for r in panel.records
    ]
    assert keys == sorted(keys)
    assert [r.partition for r in panel.records] == ["train"] * 6 + ["development"] * 2 + ["test"]
    assert [r.record_id for r in panel.partition_records("development")] == [
        r.record_id for r in panel.records if r.partition == "development"
    ]
    with pytest.raises(ValueError):
        panel.partition_records("validation")


def test_content_hash_moves_with_a_value_but_not_with_row_order():
    rows = mixed_rows()
    changed = copy.deepcopy(rows)
    changed[0]["measurement"]["value"] += 0.5
    assert world.panel_of(changed).content_sha256 != world.panel_of(rows).content_sha256


def test_record_ids_must_be_unique():
    rows = [world.row("same", heavy="ACD"), world.row("same", heavy="ACE")]
    with pytest.raises(SchemaError, match="duplicate record_id"):
        world.panel_of(rows)


def test_the_same_pair_source_and_assay_is_a_duplicate_even_with_a_new_id_or_partition():
    first = world.row("a", partition="train")
    with pytest.raises(SchemaError, match="duplicate measurement"):
        world.panel_of([first, world.row("b", partition="train", value=1.0)])
    with pytest.raises(SchemaError, match="duplicate measurement"):
        world.panel_of([first, world.row("b", partition="development")])


def test_distinct_sources_and_assays_may_measure_one_pair_and_each_counts_once():
    rows = [
        world.row("a", assay="assay1"),
        world.row("b", assay="assay2"),
        world.row("c", source="study2", assay="assay1"),
    ]
    panel = world.panel_of(rows)
    support = rec.support_counts(panel.records)
    assert support["overall"]["n_records"] == 3 and support["overall"]["unique_pairs"] == 1
    assert "weight 1" in rec.support_report(panel)["weighting"]


def test_every_used_source_and_assay_needs_a_provenance_declaration():
    rows = [world.row("a")]
    other = world.declarations([world.row("x", source="study9", assay="assay9")])
    with pytest.raises(SchemaError, match="provenance"):
        world.panel_of(rows, sources=other)
    half = world.declarations(rows)
    with pytest.raises(SchemaError, match="provenance"):
        world.panel_of(rows + [world.row("b", heavy="ACE", assay="assay2")], sources=half)


# --- split audit -----------------------------------------------------------------------------

REGIMES = ("antibody_holdout", "target_holdout", "both")


def accepts(regime, rows):
    return len(world.panel_of(rows, regime=regime).records) == len(rows)


@pytest.mark.parametrize("regime", REGIMES)
def test_an_exact_pair_may_not_straddle_partitions_in_any_regime(regime):
    rows = [world.row("a", partition="train", assay="assay1"), world.row("b", partition="development", assay="assay2")]
    with pytest.raises(SchemaError, match="exact pair"):
        world.panel_of(rows, regime=regime)


def test_antibody_holdout_separates_antibody_sequences_even_when_groups_are_renamed():
    shared = [
        world.row("a", heavy="ACD", antigen="EFG", partition="train"),
        world.row("b", heavy="ACD", antigen="EFH", partition="development"),
    ]
    for regime in ("antibody_holdout", "both"):
        with pytest.raises(SchemaError, match="antibody sequence"):
            world.panel_of(shared, regime=regime)
    assert accepts("target_holdout", shared)
    renamed = copy.deepcopy(shared)
    renamed[1]["antibody_group"] = "another-name"
    for regime in REGIMES:
        with pytest.raises(SchemaError, match="antibody sequence"):
            world.panel_of(renamed, regime=regime)


def test_antibody_holdout_separates_a_declared_antibody_group_across_different_sequences():
    rows = [
        world.row("a", heavy="ACD", antigen="EFG", partition="train", antibody_group="family"),
        world.row("b", heavy="ACE", antigen="EFH", partition="development", antibody_group="family"),
    ]
    for regime in ("antibody_holdout", "both"):
        with pytest.raises(SchemaError, match="antibody_group"):
            world.panel_of(rows, regime=regime)
    assert accepts("target_holdout", rows)


def test_target_holdout_separates_antigen_sequences_even_when_groups_are_renamed():
    shared = [
        world.row("a", heavy="ACD", antigen="EFG", partition="train"),
        world.row("b", heavy="ACE", antigen="EFG", partition="development"),
    ]
    for regime in ("target_holdout", "both"):
        with pytest.raises(SchemaError, match="antigen sequence"):
            world.panel_of(shared, regime=regime)
    assert accepts("antibody_holdout", shared)
    renamed = copy.deepcopy(shared)
    renamed[1]["antigen_group"] = "another-name"
    for regime in REGIMES:
        with pytest.raises(SchemaError, match="antigen sequence"):
            world.panel_of(renamed, regime=regime)


def test_target_holdout_separates_a_declared_antigen_group_across_different_sequences():
    rows = [
        world.row("a", heavy="ACD", antigen="EFG", partition="train", antigen_group="family"),
        world.row("b", heavy="ACE", antigen="EFH", partition="development", antigen_group="family"),
    ]
    for regime in ("target_holdout", "both"):
        with pytest.raises(SchemaError, match="antigen_group"):
            world.panel_of(rows, regime=regime)
    assert accepts("antibody_holdout", rows)


@pytest.mark.parametrize("regime", REGIMES)
def test_one_sequence_maps_to_one_declared_group_in_every_regime(regime):
    antibody = [
        world.row("a", heavy="ACD", antigen="EFG", antibody_group="g1"),
        world.row("b", heavy="ACD", antigen="EFH", antibody_group="g2"),
    ]
    with pytest.raises(SchemaError, match="antibody sequence"):
        world.panel_of(antibody, regime=regime)
    antigen = [
        world.row("a", heavy="ACD", antigen="EFG", antigen_group="g1"),
        world.row("b", heavy="ACE", antigen="EFG", antigen_group="g2"),
    ]
    with pytest.raises(SchemaError, match="antigen sequence"):
        world.panel_of(antigen, regime=regime)


@pytest.mark.parametrize("regime", REGIMES)
def test_one_group_may_hold_several_sequences_inside_one_partition(regime):
    rows = [
        world.row("a", heavy="ACD", antigen="EFG", antibody_group="family", antigen_group="target"),
        world.row("b", heavy="ACE", antigen="EFH", antibody_group="family", antigen_group="target"),
    ]
    assert accepts(regime, rows)


def test_heavy_only_and_paired_antibodies_are_different_sequences():
    rows = [
        world.row("a", heavy="ACD", light=None, antigen="EFG", partition="train"),
        world.row("b", heavy="ACD", light="EF", antigen="EFH", partition="development"),
    ]
    assert accepts("antibody_holdout", rows)
    assert rec.antibody_fingerprint("ACD", None) != rec.antibody_fingerprint("ACD", "EF")


def test_both_regime_accepts_a_clean_panel():
    assert accepts("both", mixed_rows())


# --- support counts and the report ------------------------------------------------------------


def test_complete_2x2_blocks_match_a_brute_force_count():
    full = [(a, g) for a in "12" for g in "12"]
    assert rec.complete_2x2_blocks(full) == 1
    assert rec.complete_2x2_blocks(full[:3]) == 0
    assert rec.complete_2x2_blocks([(a, g) for a in "12" for g in "123"]) == 3
    assert rec.complete_2x2_blocks([(a, g) for a in "123" for g in "123"]) == 9
    assert rec.complete_2x2_blocks([("1", g) for g in "123"]) == 0
    assert rec.complete_2x2_blocks([("1", "1"), ("2", "2"), ("3", "3")]) == 0
    assert rec.complete_2x2_blocks([]) == 0
    cells = {("1", "1"), ("1", "2"), ("1", "3"), ("2", "1"), ("2", "2"), ("3", "2"), ("3", "3"), ("4", "1"), ("4", "3")}
    antibodies = sorted({a for a, _ in cells})
    antigens = sorted({g for _, g in cells})
    expected = sum(
        all((a, g) in cells for a in pair_a for g in pair_g)
        for pair_a in itertools.combinations(antibodies, 2)
        for pair_g in itertools.combinations(antigens, 2)
    )
    assert expected > 0
    assert rec.complete_2x2_blocks(cells) == expected
    assert rec.complete_2x2_blocks({(g, a) for a, g in cells}) == expected


def support_panel():
    rows = world.grid("train", ["ACD", "ACE"], ["EFG", "EFH"])
    rows[0]["region"] = [["H", 0]]
    rows += [
        world.row("d1", heavy="ACF", antigen="EFI", partition="development", assay="assay1"),
        world.row("d2", heavy="ACG", antigen="EFI", partition="development", assay="assay1"),
        world.row("d3", heavy="ACF", antigen="EFK", partition="development", assay="assay2"),
        world.row("d4", heavy="ACG", antigen="EFK", partition="development", assay="assay2"),
    ]
    return world.panel_of(rows)


def test_support_counts_are_per_partition_source_and_assay_with_real_2x2_blocks():
    support = rec.support_counts(support_panel().records)
    assert support["declared_measured"] == 8
    assert support["overall"] == {
        "n_records": 8,
        "unique_antibodies": 4,
        "unique_antigens": 4,
        "unique_pairs": 8,
        "missing_region": 7,
    }
    assert support["by_partition"]["train"]["n_records"] == 4
    assert support["by_partition"]["train"]["missing_region"] == 3
    assert support["by_partition"]["development"]["unique_pairs"] == 4
    assert support["by_partition"]["test"]["n_records"] == 0
    groups = {(g["partition"], g["source"], g["assay"]): g for g in support["groups"]}
    assert [(g["partition"], g["assay"]) for g in support["groups"]] == [
        ("train", "assay1"),
        ("development", "assay1"),
        ("development", "assay2"),
    ]
    assert groups[("train", "study1", "assay1")]["complete_2x2_blocks"] == 1
    # Each development assay holds a 2x1 slice; pooled over assays they would form a 2x2 block that does not exist.
    assert groups[("development", "study1", "assay1")]["complete_2x2_blocks"] == 0
    assert groups[("development", "study1", "assay2")]["complete_2x2_blocks"] == 0
    assert groups[("development", "study1", "assay1")]["unique_antigens"] == 1


def test_the_report_is_json_serializable_and_never_claims_what_it_did_not_audit():
    panel = support_panel()
    report = rec.support_report(panel)
    json.dumps(report, allow_nan=False)
    assert report["split_regime"] == "both" and report["split_basis_status"] == "declared"
    assert report["split_basis"] == panel.manifest.split_basis
    checks = report["split_checks"]
    assert checks["homology_separation"] == "not_audited"
    assert checks["upstream_pretraining_exclusion"] == "not_audited"
    assert checks["exact_pair_disjoint_across_partitions"] == "enforced"
    assert checks["antibody_sequence_and_group_disjoint"] == "enforced"
    assert checks["antigen_sequence_and_group_disjoint"] == "enforced"
    assert report["measured_flag"] == "declared_not_verified"
    assert report["scientific_qualification"] == "not_assessed"
    assert "specificity" in report and report["no_support"] is False


def test_the_report_marks_axes_a_regime_does_not_require():
    antibody_only = rec.support_report(world.panel_of(mixed_rows(), regime="antibody_holdout"))
    assert antibody_only["split_checks"]["antigen_sequence_and_group_disjoint"] == "not_required_by_regime"
    target_only = rec.support_report(world.panel_of(mixed_rows(), regime="target_holdout"))
    assert target_only["split_checks"]["antibody_sequence_and_group_disjoint"] == "not_required_by_regime"


def test_the_report_lists_original_ids_in_deterministic_order_for_the_identity_join():
    rows = mixed_rows()
    report = rec.support_report(world.panel_of(rows))
    reverse = rec.support_report(world.panel_of(rows[::-1]))
    assert [e["record_id"] for e in report["records"]] == [e["record_id"] for e in reverse["records"]]
    assert {e["record_id"] for e in report["records"]} == {item["record_id"] for item in rows}
    first = report["records"][0]
    assert set(first) == {
        "record_id",
        "partition",
        "source",
        "assay",
        "antibody_fingerprint",
        "antigen_fingerprint",
        "pair_fingerprint",
        "content_sha256",
        "has_region",
    }


def test_unused_declarations_are_reported_not_hidden():
    rows = [world.row("a")]
    sources = world.declarations(rows)
    sources["study1"]["assays"]["assay-unused"] = "declared but not used"
    report = rec.support_report(world.panel_of(rows, sources=sources))
    assert report["unused_declarations"] == [{"source": "study1", "assay": "assay-unused"}]


def test_an_empty_panel_is_valid_admission_with_no_support_and_never_qualifies_science():
    panel = world.panel_of([])
    assert panel.records == ()
    report = rec.support_report(panel)
    assert report["no_support"] is True
    assert report["scientific_qualification"] == "not_assessed"
    assert report["support"]["overall"]["n_records"] == 0
    assert report["support"]["groups"] == []
    assert report["records"] == []

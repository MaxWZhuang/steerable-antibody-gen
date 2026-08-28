"""
Tests for `scripts/audit_target_identity.py` (J02 measurement, Ruling 4 input).

Synthetic fixtures only; nothing here reads the real corpus. What is pinned is
that each reported quantity means what the report claims it means, because the
audit's whole value is that its numbers are quotable as evidence.

The two leakage axes are tested separately and against each other: a corpus can
be perfectly target-disjoint and still be antibody-leaky, and conflating the two
is what made a target-keyed split look like it protected a stage that predicts
the antibody.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def audit_module(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "audit_target_identity.py"
    spec = importlib.util.spec_from_file_location("audit_target_identity", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preparer(audit_module):
    pytest.importorskip("pyarrow")
    return audit_module.load_preparer()


def _record(
    *,
    record_id: str,
    split: str,
    heavy: str,
    hcdr3: str,
    antigen: str,
    uniprot: str = "",
    pdb: str = "",
    name: str = "",
    is_strong_binder: bool = True,
    binder_label: int | None = 1,
) -> dict:
    return {
        "record_id": record_id,
        "split": split,
        "sequence_heavy": heavy,
        "heavy_variable_aa": heavy,
        "cdr3_aa_heavy": hcdr3,
        "sequence_antigen": antigen,
        "target_uniprot": uniprot,
        "target_pdb": pdb,
        "target_name": name,
        "target_key": uniprot or pdb or name,
        "is_strong_binder": is_strong_binder,
        "binder_label": binder_label,
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "corpus.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def test_two_accessions_on_one_row_merge_into_one_component(
    audit_module, preparer, tmp_path: Path
):
    """The alias merge is the whole point of canonicalization: one biological
    target written two ways must not be counted as two targets."""
    records = [
        _record(
            record_id="a", split="train", heavy="QVQ", hcdr3="ARDY",
            antigen="MKT", uniprot="P11111", pdb="1ABC",
        ),
        _record(
            record_id="b", split="val", heavy="QVK", hcdr3="ARWY",
            antigen="MKT", pdb="1ABC",
        ),
    ]
    report = audit_module.audit(_write(tmp_path, records), preparer)
    assert report["concentration"]["distinct_canonical_targets"] == 1
    # Two legacy keys, one canonical component -> exactly the fusion being counted.
    assert report["components_merging_multiple_legacy_keys"]["count"] == 1


def test_an_alias_merge_can_reveal_a_straddle_the_legacy_key_hid(
    audit_module, preparer, tmp_path: Path
):
    """
    The legacy key reports zero straddle by construction -- the split was BUILT
    on it. Canonical identity is what can disagree, and a disagreement is the
    finding, so it must be visible in the report rather than averaged away.
    """
    records = [
        _record(
            record_id="a", split="train", heavy="QVQ", hcdr3="ARDY",
            antigen="MKT", uniprot="P11111", pdb="1ABC",
        ),
        _record(
            record_id="b", split="val", heavy="EVK", hcdr3="ARWY",
            antigen="MKT", pdb="1ABC",
        ),
    ]
    report = audit_module.audit(_write(tmp_path, records), preparer)
    assert report["legacy_target_key"]["straddling_groups"] == 0
    assert report["canonical"]["straddling_groups"] == 1
    assert report["per_split_leakage"]["val"]["rows_on_a_target_seen_in_another_split"] == 1


def test_unrelated_targets_do_not_merge(audit_module, preparer, tmp_path: Path):
    """The audit must not report leakage that is an artifact of over-merging."""
    records = [
        _record(
            record_id="a", split="train", heavy="QVQ", hcdr3="ARDY",
            antigen="MKT", uniprot="P11111",
        ),
        _record(
            record_id="b", split="val", heavy="EVK", hcdr3="ARWY",
            antigen="WWW", uniprot="P22222",
        ),
    ]
    report = audit_module.audit(_write(tmp_path, records), preparer)
    assert report["concentration"]["distinct_canonical_targets"] == 2
    assert report["canonical"]["straddling_groups"] == 0
    assert report["per_split_leakage"]["val"]["rows_on_a_target_seen_in_another_split"] == 0


def test_target_disjoint_can_still_be_antibody_leaky(
    audit_module, preparer, tmp_path: Path
):
    """
    THE point of the antibody-side axis. Both rows below are on different
    targets, so every target-side number is clean -- and the val row's HCDR3 is
    nonetheless verbatim in train, which is what a stage that reconstructs HCDR3
    can memorize. A split can be perfectly target-disjoint and still not support
    a generalization claim about the antibody.
    """
    records = [
        _record(
            record_id="a", split="train", heavy="QVQLVES", hcdr3="ARDYW",
            antigen="MKT", uniprot="P11111",
        ),
        _record(
            record_id="b", split="val", heavy="QVQLVES", hcdr3="ARDYW",
            antigen="WWW", uniprot="P22222",
        ),
    ]
    report = audit_module.audit(_write(tmp_path, records), preparer)

    assert report["canonical"]["straddling_groups"] == 0
    assert report["per_split_leakage"]["val"]["rows_on_a_target_seen_in_another_split"] == 0

    antibody = report["antibody_side_leakage"]["all_rows"]
    assert antibody["val_rows"] == 1
    assert antibody["val_rows_whose_heavy_chain_is_in_train"] == 1
    assert antibody["val_rows_whose_hcdr3_is_in_train"] == 1
    assert antibody["hcdr3_leak_fraction"] == 1.0


def test_populations_are_gated_the_way_the_stages_gate_them(
    audit_module, preparer, tmp_path: Path
):
    """
    Stage 4 gates on `is_strong_binder`, stage 3 on the binary `binder_label`.
    Reporting only whole-file leakage hides that a stage's own val population can
    leak at a very different rate -- which is exactly what happened here.
    """
    records = [
        _record(
            record_id="t1", split="train", heavy="QVQ", hcdr3="ARDY", antigen="MKT",
            uniprot="P11111", is_strong_binder=True, binder_label=1,
        ),
        # Val row inside BOTH populations.
        _record(
            record_id="v1", split="val", heavy="QVQ", hcdr3="ARDY", antigen="WWW",
            uniprot="P22222", is_strong_binder=True, binder_label=1,
        ),
        # Val row in the binary population only: not a strong binder.
        _record(
            record_id="v2", split="val", heavy="EEE", hcdr3="AAAA", antigen="YYY",
            uniprot="P33333", is_strong_binder=False, binder_label=0,
        ),
        # Val row in neither: fuzzy/KD row with no binary label, not strong.
        _record(
            record_id="v3", split="val", heavy="FFF", hcdr3="CCCC", antigen="ZZZ",
            uniprot="P44444", is_strong_binder=False, binder_label=None,
        ),
    ]
    report = audit_module.audit(_write(tmp_path, records), preparer)

    assert report["per_population"]["stage4_strong_binders"]["rows_by_split"]["val"] == 1
    assert report["per_population"]["stage3_binary_labeled"]["rows_by_split"]["val"] == 2
    assert report["antibody_side_leakage"]["all_rows"]["val_rows"] == 3


def test_labels_never_influence_the_identity_partition(
    audit_module, preparer, tmp_path: Path
):
    """
    Supervision must not decide which rows are grouped, or the split becomes a
    function of the labels it is meant to hold out. Flipping every label must
    leave the canonical partition byte-identical.
    """
    base = [
        _record(
            record_id="a", split="train", heavy="QVQ", hcdr3="ARDY",
            antigen="MKT", uniprot="P11111", is_strong_binder=True, binder_label=1,
        ),
        _record(
            record_id="b", split="val", heavy="EVK", hcdr3="ARWY",
            antigen="MKT", uniprot="P11111", is_strong_binder=True, binder_label=1,
        ),
    ]
    flipped = [
        {**record, "is_strong_binder": False, "binder_label": 0} for record in base
    ]

    report_a = audit_module.audit(_write(tmp_path / "a", base), preparer)
    report_b = audit_module.audit(_write(tmp_path / "b", flipped), preparer)
    assert report_a["canonical"] == report_b["canonical"]
    assert report_a["concentration"]["top_targets"] == report_b["concentration"]["top_targets"]


@pytest.fixture(autouse=True)
def _make_subdirs(tmp_path: Path):
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)

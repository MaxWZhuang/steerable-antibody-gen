"""Failures that would silently change the benchmark-to-structure correspondence."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import pytest

pytest.importorskip("biotite")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/audit_cr9114_5cjq_mapping.py"
spec = importlib.util.spec_from_file_location("cr9114_mapping_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("data", [b"\x00", b"\x00" + struct.pack(">I", 10) + b"abc"])
def test_truncated_snapgene_is_rejected(data):
    with pytest.raises(ValueError, match="Truncated SnapGene"):
        list(audit.snapgene_packets(data))


def test_upstream_source_is_not_executed():
    source = b"raise RuntimeError('must not run')\nbinary_dict_list = [{'GCT': '0', 'TGT': '1'}]"
    assert audit.literal_assignment(source, "binary_dict_list") == [{"GCT": "0", "TGT": "1"}]


def test_reversed_binary_state_fails_even_with_correct_residue_pair():
    germline, somatic = "A" * 16, "C" * 16
    labels = {(i, "A", "C") for i in range(16)}
    binary = [{"GCT": "0", "TGT": "1"} for _ in range(16)]
    binary[7] = {"GCT": "1", "TGT": "0"}
    with pytest.raises(ValueError, match="binary direction"):
        audit.verify_alleles(germline, somatic, binary, list(map(str, range(16))), labels)


def test_fragment_position_disagreement_fails():
    labels = {(i, "A", "C") for i in range(16)}
    labels.remove((2, "A", "C"))
    labels.add((16, "A", "C"))
    with pytest.raises(ValueError, match="fragment labels disagree"):
        audit.verify_alleles("A" * 16, "C" * 16,
                             [{"GCT": "0", "TGT": "1"}] * 16,
                             list(map(str, range(16))), labels)


def synthetic_vh():
    from biotite.sequence import ProteinSequence

    sequence = list("A" * 121)
    sequence[45] = "E"
    template = sequence.copy()
    template[23], template[45] = "S", "D"
    polymer, observed = [], {}
    for i, aa in enumerate(template):
        # Adjacent sequence positions intentionally share a residue number but
        # differ in insertion code. Dropping the code makes the mapping wrong.
        number, insertion = (83, "A") if i == 83 else (i + 1, "")
        residue = ProteinSequence.convert_letter_1to3(aa)
        polymer.append({"pdb_strand_id": "H", "seq_id": str(i + 1),
                        "mon_id": residue, "pdb_seq_num": str(number),
                        "pdb_ins_code": insertion or "."})
        observed[("H", number, insertion)] = [
            {"auth_atom_id": atom, "auth_comp_id": residue} for atom in ("N", "CA", "C")
        ]
    return "".join(sequence), polymer, observed


def test_insertion_code_is_part_of_residue_identity():
    sequence, polymer, observed = synthetic_vh()
    correspondence = audit.map_vh(sequence, polymer, observed)
    assert correspondence[82]["author_res_id"] == correspondence[83]["author_res_id"] == 83
    assert correspondence[82]["insertion_code"] == ""
    assert correspondence[83]["insertion_code"] == "A"


def test_missing_inserted_residue_backbone_is_not_borrowed_from_neighbor():
    sequence, polymer, observed = synthetic_vh()
    observed[("H", 83, "A")].pop()
    with pytest.raises(ValueError, match="Missing/duplicate backbone"):
        audit.map_vh(sequence, polymer, observed)


def test_unexpected_template_mismatch_is_rejected():
    sequence, polymer, observed = synthetic_vh()
    sequence = "V" + sequence[1:]
    with pytest.raises(ValueError, match="unique ungapped VH match"):
        audit.map_vh(sequence, polymer, observed)


def test_changed_source_bytes_are_rejected(tmp_path):
    import hashlib
    import json

    source = tmp_path / "source.bin"
    source.write_bytes(b"old")
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"files": {"example": {
        "relative_path": "source.bin", "size_bytes": 3,
        "sha256": hashlib.sha256(b"old").hexdigest(),
    }}}))
    source.write_bytes(b"new")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        audit.read_pinned_sources(tmp_path, manifest)

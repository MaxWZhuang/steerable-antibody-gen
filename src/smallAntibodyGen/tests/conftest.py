from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def script_path(project_root: Path) -> Path:
    return project_root / "scripts" / "prepare_oas.py"


@pytest.fixture
def tokenizer():
    from smallAntibodyGen.tokenizer import AminoAcidTokenizer
    return AminoAcidTokenizer()


@pytest.fixture
def heavy_seq() -> str:
    return (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRF"
        "TISRDNSKNTLYLQMNSLRAEDTAVYYCAKNDILVGYSAFDYWGQGTLVTVSS"
    )


@pytest.fixture
def heavy_cdr3() -> str:
    return "AKNDILVGYSAFDY"


@pytest.fixture
def light_seq() -> str:
    return (
        "DIQMTQSPSSLSASVGDRVTITCQASQDINNYLNWYQQKPGKAPKLLIYYTSRLHSGVPSRFSGSGSGTDFT"
        "LTISSLQPEDFATYYCQQYNSYPWTFGQGTKVEIK"
    )


@pytest.fixture
def light_cdr3() -> str:
    return "QQYNSYPWT"


@pytest.fixture
def make_oas_row():
    """
    Return a helper that creates a minimal OAS-like row dictionary.
    """
    def _make(
        *,
        sequence_alignment_aa: str = "",
        cdr3_aa: str = "",
        locus: str = "H",
        productive: str = "T",
        vj_in_frame: str = "T",
        stop_codon: str = "F",
        v_frameshift: str = "F",
        complete_vdj: str = "T",
        redundancy: int = 1,
        v_call: str = "IGHV1-1*01",
        d_call: str = "IGHD1-1*01",
        j_call: str = "IGHJ4*02",
        v_sequence_alignment_aa: str = "",
        sequence_aa: str = "",
        sequence: str = "",
        fwr1_aa: str = "",
        cdr1_aa: str = "",
        fwr2_aa: str = "",
        cdr2_aa: str = "",
        fwr3_aa: str = "",
        fwr4_aa: str = "",
        extra: Dict | None = None,
    ) -> Dict:
        row = {
            "sequence": sequence,
            "locus": locus,
            "stop_codon": stop_codon,
            "vj_in_frame": vj_in_frame,
            "v_frameshift": v_frameshift,
            "productive": productive,
            "rev_comp": "F",
            "complete_vdj": complete_vdj,
            "v_call": v_call,
            "d_call": d_call,
            "j_call": j_call,
            "sequence_alignment_aa": sequence_alignment_aa,
            "v_sequence_alignment_aa": v_sequence_alignment_aa,
            "sequence_aa": sequence_aa,
            "fwr1_aa": fwr1_aa,
            "cdr1_aa": cdr1_aa,
            "fwr2_aa": fwr2_aa,
            "cdr2_aa": cdr2_aa,
            "fwr3_aa": fwr3_aa,
            "fwr4_aa": fwr4_aa,
            "cdr3_aa": cdr3_aa,
            "Redundancy": redundancy,
        }
        if extra:
            row.update(extra)
        return row

    return _make


@pytest.fixture
def write_oas_data_unit():
    """
    Return a helper that writes a tiny OAS-style .csv.gz file.

    Supports:
    - plain metadata JSON line
    - CSV-quoted metadata JSON line
    - no metadata line
    - configurable delimiter
    """
    def _write(
        path: Path,
        rows: List[Dict],
        *,
        metadata: Dict | None = None,
        quoted_metadata: bool = True,
        delimiter: str = ",",
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            if metadata is not None:
                if quoted_metadata:
                    payload = json.dumps(metadata).replace('"', '""')
                    f.write(f'"{payload}"\n')
                else:
                    f.write(json.dumps(metadata) + "\n")

            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return path

    return _write


@pytest.fixture
def write_processed_jsonl_gz():
    def _write(path: Path, records: List[Dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    return _write
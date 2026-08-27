"""Regression tests for two corpus-level defects in the paired OAS path.

1. **Split leakage.** The paired train/val split was keyed on the full
   ``(heavy, light)`` pair, so one heavy chain observed with several cognate lights
   was scattered across BOTH splits -- leaking the exact HCDR3 modeling target into
   validation. On the shipped corpus 1,406 heavy sequences occurred in both splits
   and 6.3% of val rows had a byte-identical heavy chain in train, so stage-2 val
   loss (which selects ``best.pt``) was partly measuring memorization. The unpaired
   path never had this defect: it keys the split on ``f"{locus}:{variable_aa}"``.

2. **Silent truncation.** ``prepare_oas.py`` bounds heavy and light independently
   (up to 345 tokens) and writes ``token_length`` unclamped, while the collator
   hard-truncates to ``cfg.max_length``. At the shipped ``max_length: 192``, 99.97%
   of paired rows overflow and 99.77% lose their LIGHT CDR3 entirely. Nothing
   surfaced it -- Python dedupes the tokenizer's UserWarning. The fix is a loud
   preflight, because whether to raise ``max_length``, tighten the corpus bounds, or
   accept the loss is a research decision, not something the trainer should pick.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from smallAntibodyGen.data.MLMCollator import OASRecord


def _load_script(project_root: Path, name: str):
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _paired_row(heavy_seq, heavy_cdr3, light_seq, light_cdr3, idx: int) -> dict:
    return {
        "sequence_id_heavy": f"heavy-{idx}",
        "sequence_alignment_aa_heavy": heavy_seq,
        "v_sequence_alignment_aa_heavy": heavy_seq,
        "cdr3_aa_heavy": heavy_cdr3,
        "locus_heavy": "H",
        "productive_heavy": "T", "vj_in_frame_heavy": "T", "stop_codon_heavy": "F",
        "v_frameshift_heavy": "F", "complete_vdj_heavy": "T",
        "v_call_heavy": "IGHV1-1*01", "d_call_heavy": "IGHD1-1*01", "j_call_heavy": "IGHJ4*02",
        "sequence_id_light": f"light-{idx}",
        "sequence_alignment_aa_light": light_seq,
        "v_sequence_alignment_aa_light": light_seq,
        "cdr3_aa_light": light_cdr3,
        "locus_light": "K",
        "productive_light": "T", "vj_in_frame_light": "T", "stop_codon_light": "F",
        "v_frameshift_light": "F", "complete_vdj_light": "T",
        "v_call_light": "IGKV1-1*01", "d_call_light": "", "j_call_light": "IGKJ1*01",
        "Redundancy": 1,
    }


def test_one_heavy_chain_never_straddles_the_paired_split(
    tmp_path: Path, script_path: Path, write_oas_data_unit,
    heavy_seq, heavy_cdr3, light_seq, light_cdr3,
):
    """A heavy chain seen with many cognate lights must land in ONE split."""
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    # One heavy chain, 40 distinct light chains. Under the pair-keyed split this
    # scattered across train and val.
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for idx in range(40):
        # Vary TWO positions so all 40 light chains are genuinely distinct;
        # dedup keys on the full pair, so a repeated light would be dropped.
        light = (
            light_seq[:20]
            + alphabet[idx % 20]
            + alphabet[(idx // 20) % 20]
            + light_seq[22:]
        )
        rows.append(_paired_row(heavy_seq, heavy_cdr3, light, light_cdr3, idx))
    assert len({r["sequence_alignment_aa_light"] for r in rows}) == 40
    write_oas_data_unit(
        raw_dir / "many_lights.csv.gz", rows,
        metadata={"Chain": "Paired"}, quoted_metadata=True,
    )
    subprocess.run(
        [sys.executable, str(script_path), "--input-dir", str(raw_dir),
         "--output-dir", str(out_dir), "--stats-output", str(stats_path)],
        check=True, capture_output=True, text=True,
    )

    records = _load_jsonl_gz(out_dir / "oas_paired.jsonl.gz")
    assert len(records) == 40, "all 40 distinct pairs must survive dedup"

    splits_by_heavy: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_heavy[record["sequence_heavy"]].add(record["split"])

    straddling = {h: s for h, s in splits_by_heavy.items() if len(s) > 1}
    assert not straddling, (
        f"{len(straddling)} heavy chain(s) appear in both splits: "
        "the HCDR3 target leaks into validation"
    )


def test_distinct_pairs_are_still_all_kept(
    tmp_path: Path, script_path: Path, write_oas_data_unit,
    heavy_seq, heavy_cdr3, light_seq, light_cdr3,
):
    """Dedup must stay keyed on the full pair -- the split fix must not merge rows.

    A distinct (heavy, light) pair IS a distinct training example, so changing the
    SPLIT key must not change what is kept.
    """
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    rows = []
    for idx in range(5):
        light = light_seq[:20] + "ACDEF"[idx] + light_seq[21:]
        rows.append(_paired_row(heavy_seq, heavy_cdr3, light, light_cdr3, idx))
    # ...plus an exact duplicate of the first pair, which MUST be dropped.
    rows.append(_paired_row(heavy_seq, heavy_cdr3, light_seq[:20] + "A" + light_seq[21:], light_cdr3, 99))
    write_oas_data_unit(
        raw_dir / "pairs.csv.gz", rows, metadata={"Chain": "Paired"}, quoted_metadata=True,
    )
    subprocess.run(
        [sys.executable, str(script_path), "--input-dir", str(raw_dir),
         "--output-dir", str(out_dir), "--stats-output", str(tmp_path / "s.json")],
        check=True, capture_output=True, text=True,
    )
    records = _load_jsonl_gz(out_dir / "oas_paired.jsonl.gz")
    assert len(records) == 5, "the exact-duplicate pair must be deduped, the 5 distinct kept"


# --------------------------------------------------------------------------- #
# The truncation preflight.
# --------------------------------------------------------------------------- #
class _Records:
    def __init__(self, records):
        self.records = records


def _paired_record(heavy_len: int, light_len: int, light_cdr3_end: int) -> OASRecord:
    heavy = "A" * heavy_len
    light = "C" * light_len
    return OASRecord(
        sequence=heavy, locus="PAIRED", chain_group="paired", split="train",
        length=heavy_len + light_len, token_length=heavy_len + light_len + 5,
        sequence_heavy=heavy, sequence_light=light,
        heavy_locus="IGH", light_locus="IGK", is_paired=True,
        cdr3_start_aa_light=light_cdr3_end - 9, cdr3_end_aa_light=light_cdr3_end,
    )


def test_preflight_counts_overflow_and_lost_light_cdr3(project_root: Path):
    mlm_train = _load_script(project_root, "mlm_train")
    # heavy 180 + light 160 + 5 specials = 345 tokens, far past 192.
    dataset = _Records([_paired_record(180, 160, 150) for _ in range(10)])

    counts = mlm_train.summarize_length_truncation(dataset, max_length=192)
    assert counts["total"] == 10
    assert counts["overflow"] == 10
    assert counts["worst_overflow"] == 345 - 192
    # light CDR3 ends at token 2 + 180 + 2 + 150 = 334 > 192
    assert counts["lost_light_cdr3"] == 10

    message = mlm_train.format_length_truncation_warning(counts, 192, "train")
    assert message is not None
    assert "TRUNCATED" in message
    assert "LIGHT CDR3" in message
    assert "CDR-L3" in message


def test_preflight_is_silent_when_nothing_overflows(project_root: Path):
    """It must not cry wolf on a corpus that fits -- otherwise it gets ignored."""
    mlm_train = _load_script(project_root, "mlm_train")
    dataset = _Records([_paired_record(60, 50, 45) for _ in range(10)])

    counts = mlm_train.summarize_length_truncation(dataset, max_length=192)
    assert counts["overflow"] == 0
    assert counts["lost_light_cdr3"] == 0
    assert mlm_train.format_length_truncation_warning(counts, 192, "train") is None


def test_preflight_reports_lost_heavy_cdr3_for_single_chain_records(project_root: Path):
    mlm_train = _load_script(project_root, "mlm_train")
    heavy = "A" * 300
    dataset = _Records([
        OASRecord(
            sequence=heavy, locus="IGH", chain_group="heavy", split="train",
            length=len(heavy), token_length=len(heavy) + 3,
            cdr3_start_aa=280, cdr3_end_aa=294,
        )
    ])
    counts = mlm_train.summarize_length_truncation(dataset, max_length=192)
    assert counts["overflow"] == 1
    assert counts["lost_heavy_cdr3"] == 1
    message = mlm_train.format_length_truncation_warning(counts, 192, "train")
    assert "HEAVY CDR3" in message

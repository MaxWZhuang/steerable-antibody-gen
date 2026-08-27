"""Tests for the encoder-context census (`scripts/context_length_census.py`).

Why these tests are shaped this way
-----------------------------------
The census exists because the repository reconstructs the encoder's token layout
in several places and only one of them -- ``AminoAcidTokenizer`` -- is ground
truth. A census that re-derived the layout a fifth time would inherit exactly the
defect it is meant to detect. So the load-bearing test here is not "does the
census produce plausible numbers"; it is
:func:`test_census_antibody_length_matches_the_real_tokenizer` and its siblings,
which pin the census against ``tokenizer.encode_*`` on handcrafted rows.

The boundary cases matter for the same reason. ``encode_*`` truncates with a hard
``tokens[:max_length]`` and then OVERWRITES the last surviving token with
``[EOS]``, so "fits" and "keeps its last residue" are not the same predicate. A
row at exactly ``max_length`` fits; a CDR3 ending exactly at token index
``max_length`` inside an OVERFLOWING row does not keep its last residue. The
census reports both (``lost_*_cdr3`` and ``clipped_*_cdr3``).
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pytest

from smallAntibodyGen.data.lengths import (
    heavy_cdr3_token_end,
    light_cdr3_token_end,
    paired_token_length,
    single_chain_token_length,
)
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


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


@pytest.fixture
def census(project_root: Path):
    return _load_script(project_root, "context_length_census")


def _write_jsonl_gz(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _paired_row(
    *,
    heavy_len: int,
    light_len: int,
    light_cdr3_end: int | None = None,
    heavy_cdr3_end: int | None = None,
    split: str = "train",
    token_length: int | None = None,
) -> dict:
    heavy = "A" * heavy_len
    light = "C" * light_len
    row = {
        "sequence": heavy,
        "sequence_heavy": heavy,
        "sequence_light": light,
        "heavy_locus": "IGH",
        "light_locus": "IGK",
        "locus": "PAIRED",
        "chain_group": "paired",
        "is_paired": True,
        "split": split,
        "length": heavy_len + light_len,
        "cdr3_end_aa_heavy": heavy_cdr3_end,
        "cdr3_end_aa_light": light_cdr3_end,
    }
    if token_length is not None:
        row["token_length"] = token_length
    return row


def _heavy_only_row(
    *,
    heavy_len: int,
    cdr3_end: int | None = None,
    split: str = "train",
) -> dict:
    heavy = "A" * heavy_len
    return {
        "sequence": heavy,
        "locus": "IGH",
        "chain_group": "heavy",
        "split": split,
        "length": heavy_len,
        "cdr3_end_aa": cdr3_end,
    }


def _run(census, tmp_path: Path, rows: list[dict], candidates: list[int], **kwargs) -> dict:
    data_path = _write_jsonl_gz(tmp_path / "corpus.jsonl.gz", rows)
    tokenizer = AminoAcidTokenizer()
    return census.run_census(
        data_path=data_path,
        candidates=sorted(set(candidates)),
        antigen_candidates=sorted(set(candidates)),
        tokenizer=tokenizer,
        antigen_tokenizer=census.build_antigen_tokenizer(
            antigen_encoder_type="scratch",
            tokenizer=tokenizer,
            esm_model_name="unused",
        ),
        **kwargs,
    )


def _population(result: dict, split: str, record_type: str) -> dict:
    matches = [
        p for p in result["populations"]
        if p["split"] == split and p["record_type"] == record_type
    ]
    assert len(matches) == 1, f"expected one {split}/{record_type} population, got {matches}"
    return matches[0]


def _candidate(population: dict, max_length: int) -> dict:
    matches = [c for c in population["antibody_candidates"] if c["max_length"] == max_length]
    assert len(matches) == 1
    return matches[0]


# --------------------------------------------------------------------------- #
# The load-bearing test: agreement with the real tokenizer.
# --------------------------------------------------------------------------- #
def test_census_antibody_length_matches_the_real_tokenizer(census):
    """
    The census must read the layout off ``encode_*``, not re-derive it.

    ``summarize_length_truncation`` is an arithmetic RECONSTRUCTION of the
    tokenizer's layout, and so is ``OASSequenceDataset._load``'s ``token_length``
    fallback. Neither ever calls the tokenizer, so neither can notice if the
    tokenizer changes. This census can, and this test is why.
    """
    tokenizer = AminoAcidTokenizer()
    rows = [
        _paired_row(heavy_len=118, light_len=107, light_cdr3_end=98),
        _heavy_only_row(heavy_len=121, cdr3_end=110),
        # An unknown residue must still cost exactly one token ([UNK]).
        {"sequence": "ACDJZ", "locus": "IGH", "chain_group": "heavy", "split": "train"},
    ]
    for row in rows:
        record = census.parse_length_record(row)
        measured = census.encode_antibody_length(record, tokenizer)
        if record.heavy and record.light:
            expected = tokenizer.encode_paired_sequences(
                record.heavy, record.light,
                heavy_locus=record.heavy_locus, light_locus=record.light_locus,
                max_length=None,
            )
        else:
            expected = tokenizer.encode_sequence(
                record.heavy, locus=record.heavy_locus, max_length=None
            )
        assert measured == len(expected)


def test_arithmetic_reconstruction_agrees_with_the_tokenizer_on_prepared_layouts(census):
    """
    Pin the heuristic-vs-tokenizer agreement the census reports.

    ``prepare_antibody_antigen.py`` never writes ``token_length``, so
    ``OASSequenceDataset._load`` falls back to ``len(heavy) + len(light) + 5`` /
    ``len(sequence) + 3``. That heuristic is a separate copy of the token layout.
    On the layouts the producers emit it agrees with the tokenizer exactly -- and
    if it ever stops agreeing, this test is what says so.
    """
    tokenizer = AminoAcidTokenizer()
    for row in (
        _paired_row(heavy_len=118, light_len=107),
        _heavy_only_row(heavy_len=121),
        _paired_row(heavy_len=1, light_len=1),
    ):
        record = census.parse_length_record(row)
        assert census.reconstructed_antibody_length(record) == census.encode_antibody_length(
            record, tokenizer
        )

    # The one input on which they must NOT be assumed to agree: a sequence
    # carrying characters the tokenizer maps to [UNK] still costs one token each,
    # so agreement holds -- but agreement is a fact about this tokenizer, not an
    # invariant of the format. Assert it explicitly rather than by omission.
    unknown = census.parse_length_record(
        {"sequence": "ACDJZ", "locus": "IGH", "chain_group": "heavy", "split": "train"}
    )
    assert census.reconstructed_antibody_length(unknown) == single_chain_token_length(5)
    assert census.encode_antibody_length(unknown, tokenizer) == single_chain_token_length(5)


def test_census_reports_stored_and_reconstructed_disagreement(census, tmp_path: Path):
    """A stale stored ``token_length`` must be surfaced, not trusted."""
    rows = [
        # 15 + 12 + 5 = 32 tokens, but the producer wrote 99.
        _paired_row(heavy_len=15, light_len=12, token_length=99),
        _paired_row(heavy_len=15, light_len=12, token_length=32),
    ]
    result = _run(census, tmp_path, rows, [32])
    agreement = _population(result, "train", "paired")["layout_agreement"]
    assert agreement["rows_with_stored_token_length"] == 2
    assert agreement["stored_vs_tokenizer_mismatches"] == 1
    assert agreement["stored_vs_tokenizer_max_abs_delta"] == 99 - 32
    # The arithmetic reconstruction does not read the stored field, so it agrees.
    assert agreement["reconstruction_vs_tokenizer_mismatches"] == 0


# --------------------------------------------------------------------------- #
# Boundaries: exactly max_length, and one token above.
# --------------------------------------------------------------------------- #
def test_row_at_exactly_max_length_does_not_overflow(census, tmp_path: Path):
    # [CLS] [IGH] 15xA [SEP] [IGK] 12xC [EOS] = 32 tokens.
    assert paired_token_length(15, 12) == 32
    result = _run(census, tmp_path, [_paired_row(heavy_len=15, light_len=12)], [32])
    population = _population(result, "train", "paired")
    assert population["antibody_stream"]["max"] == 32
    row = _candidate(population, 32)
    assert row["overflow"] == 0
    assert row["worst_overflow"] == 0


def test_row_one_token_above_max_length_overflows_by_one(census, tmp_path: Path):
    assert paired_token_length(15, 13) == 33
    result = _run(census, tmp_path, [_paired_row(heavy_len=15, light_len=13)], [32])
    row = _candidate(_population(result, "train", "paired"), 32)
    assert row["overflow"] == 1
    assert row["overflow_fraction"] == 1.0
    assert row["worst_overflow"] == 1


def test_boundary_matches_whether_the_tokenizer_actually_truncates(census, tmp_path: Path):
    """
    The overflow predicate must be the tokenizer's, not one token off it.

    ``encode_*`` truncates iff ``len(tokens) > max_length``. Pinning the census
    against that directly is cheaper than reasoning about the off-by-one.
    """
    tokenizer = AminoAcidTokenizer()
    for heavy_len, light_len, expect_overflow in ((15, 11, False), (15, 12, False), (15, 13, True)):
        row = _paired_row(heavy_len=heavy_len, light_len=light_len)
        natural = len(
            tokenizer.encode_paired_sequences("A" * heavy_len, "C" * light_len, max_length=None)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            truncated = len(
                tokenizer.encode_paired_sequences("A" * heavy_len, "C" * light_len, max_length=32)
            )
        # Two independent readings of "did the tokenizer truncate": the shortened
        # output, and the UserWarning the corpus-scale run never gets to see.
        assert (truncated < natural) is expect_overflow
        assert bool(caught) is expect_overflow

        result = _run(census, tmp_path, [row], [32])
        assert _candidate(_population(result, "train", "paired"), 32)["overflow"] == int(
            expect_overflow
        )


# --------------------------------------------------------------------------- #
# Paired light-CDR3 offsets.
# --------------------------------------------------------------------------- #
def test_paired_light_cdr3_offset_is_measured_past_the_sep_and_chain_token(
    census, tmp_path: Path
):
    """
    The light chain starts four tokens after the heavy chain's last residue.

    ``[CLS] [IGH] heavy... [SEP] [IGK] light... [EOS]``: a light CDR3 ending at
    amino-acid index ``e`` of a pair whose heavy chain is ``n`` long ends at token
    index ``2 + n + 2 + e``. Getting this wrong is the difference between "the
    pairing objective still sees CDR-L3" and "it does not".
    """
    heavy_len = 15
    # 2 + 15 + 2 + 13 = 32: ends exactly at the window edge.
    assert light_cdr3_token_end(heavy_len, 13) == 32
    result = _run(
        census, tmp_path,
        [_paired_row(heavy_len=heavy_len, light_len=40, light_cdr3_end=13)],
        [32],
    )
    population = _population(result, "train", "paired")
    assert population["rows_with_light_cdr3"] == 1
    row = _candidate(population, 32)
    # The row overflows (2+15+2+40+1 = 60 tokens), so its final surviving token is
    # forced to [EOS] -- the CDR3's last residue is clipped even though its end
    # index is not strictly past the window.
    assert row["overflow"] == 1
    assert row["lost_light_cdr3"] == 0
    assert row["clipped_light_cdr3"] == 1

    # One amino acid further out and the CDR3 is genuinely past the window.
    result = _run(
        census, tmp_path,
        [_paired_row(heavy_len=heavy_len, light_len=40, light_cdr3_end=14)],
        [32],
    )
    row = _candidate(_population(result, "train", "paired"), 32)
    assert row["lost_light_cdr3"] == 1
    assert row["lost_light_cdr3_fraction"] == 1.0


def test_light_cdr3_inside_the_window_is_not_counted_as_lost(census, tmp_path: Path):
    result = _run(
        census, tmp_path,
        [_paired_row(heavy_len=15, light_len=40, light_cdr3_end=5)],
        [64],
    )
    row = _candidate(_population(result, "train", "paired"), 64)
    assert row["overflow"] == 0
    assert row["lost_light_cdr3"] == 0
    assert row["clipped_light_cdr3"] == 0


# --------------------------------------------------------------------------- #
# Heavy-only rows.
# --------------------------------------------------------------------------- #
def test_heavy_only_rows_are_classified_and_counted_separately(census, tmp_path: Path):
    rows = [
        _heavy_only_row(heavy_len=29, cdr3_end=25),
        _paired_row(heavy_len=15, light_len=12),
    ]
    result = _run(census, tmp_path, rows, [32])
    heavy = _population(result, "train", "single_heavy")
    assert heavy["rows"] == 1
    assert heavy["antibody_stream"]["max"] == single_chain_token_length(29) == 32
    assert _population(result, "train", "paired")["rows"] == 1


def test_heavy_only_row_reports_lost_heavy_cdr3(census, tmp_path: Path):
    """A heavy-only row falls back to the unsuffixed ``cdr3_end_aa``."""
    # 2 + 31 = 33 > 32.
    assert heavy_cdr3_token_end(31) == 33
    result = _run(census, tmp_path, [_heavy_only_row(heavy_len=100, cdr3_end=31)], [32])
    population = _population(result, "train", "single_heavy")
    assert population["rows_with_heavy_cdr3"] == 1
    row = _candidate(population, 32)
    assert row["overflow"] == 1
    assert row["lost_heavy_cdr3"] == 1
    assert row["lost_light_cdr3"] == 0


def test_single_light_rows_are_their_own_population(census, tmp_path: Path):
    rows = [{
        "sequence": "C" * 20, "locus": "IGK", "chain_group": "light",
        "split": "train", "length": 20,
    }]
    result = _run(census, tmp_path, rows, [32])
    assert _population(result, "train", "single_light")["rows"] == 1


# --------------------------------------------------------------------------- #
# The antigen stream. NOTE: no antibody-antigen corpus exists locally, so this
# path is exercised only by these fixtures.
# --------------------------------------------------------------------------- #
def test_antigen_stream_is_reported_separately_from_the_antibody_stream(
    census, tmp_path: Path
):
    """
    The dual-stream model encodes the antigen separately, so the census must too.

    On ``antigen_encoder_type='scratch'`` the collator's effective antigen cap is
    the ANTIBODY ``max_length`` -- ``antigen_max_length`` is ignored entirely --
    while ``prepare_antibody_antigen.py`` admits antigens up to 2048 aa. Reporting
    the two streams under one number would hide exactly that.
    """
    antigen = "D" * 300
    rows = [{
        "sequence": "A" * 118, "sequence_heavy": "A" * 118, "sequence_light": "C" * 107,
        "sequence_antigen": antigen, "heavy_locus": "IGH", "light_locus": None,
        "locus": "PAIRED_ANTIGEN", "chain_group": "paired_antigen",
        "is_paired": True, "split": "train", "cdr3_end_aa_heavy": 110,
    }]
    result = _run(census, tmp_path, rows, [192, 256])
    population = _population(result, "train", "antibody_antigen")

    assert population["antibody_stream"]["max"] == paired_token_length(118, 107) == 230
    # [CLS] [OTHER_CHAIN] 300 residues [EOS]
    assert population["antigen_stream"]["max"] == single_chain_token_length(300) == 303

    antigen_at_192 = next(
        c for c in population["antigen_candidates"] if c["max_length"] == 192
    )
    assert antigen_at_192["overflow"] == 1
    assert antigen_at_192["worst_overflow"] == 303 - 192
    # The antibody stream fits at 256 while the antigen still does not: one number
    # for both streams would have said "fine".
    antibody_at_256 = _candidate(population, 256)
    assert antibody_at_256["overflow"] == 0
    antigen_at_256 = next(
        c for c in population["antigen_candidates"] if c["max_length"] == 256
    )
    assert antigen_at_256["overflow"] == 1


def test_antigen_length_matches_the_production_antigen_tokenizer(census):
    tokenizer = AminoAcidTokenizer()
    adapter = census.build_antigen_tokenizer(
        antigen_encoder_type="scratch", tokenizer=tokenizer, esm_model_name="unused"
    )
    antigen = "DEFGH" * 40
    assert len(adapter.encode(antigen, census.UNCAPPED_MAX_LENGTH)) == len(
        tokenizer.encode_sequence(antigen, locus=None, max_length=None)
    )


def test_populations_without_an_antigen_report_a_null_antigen_stream(
    census, tmp_path: Path
):
    result = _run(census, tmp_path, [_paired_row(heavy_len=15, light_len=12)], [32])
    population = _population(result, "train", "paired")
    assert population["antigen_stream"] is None
    assert population["antigen_candidates"] is None


# --------------------------------------------------------------------------- #
# Determinism and the CLI contract.
# --------------------------------------------------------------------------- #
def test_output_json_is_byte_identical_across_runs(census, tmp_path: Path):
    rows = [
        _paired_row(heavy_len=15, light_len=12, split="val"),
        _heavy_only_row(heavy_len=29, cdr3_end=25),
        _paired_row(heavy_len=20, light_len=20, light_cdr3_end=14),
        {"sequence": "C" * 20, "locus": "IGL", "chain_group": "light", "split": "val"},
    ]
    data_path = _write_jsonl_gz(tmp_path / "corpus.jsonl.gz", rows)
    outputs = []
    for i in range(2):
        out = tmp_path / f"census-{i}.json"
        census.main([
            "--data-path", str(data_path),
            "--candidate-max-length", "512", "192", "256",
            "--output-json", str(out),
        ])
        outputs.append(out.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]

    payload = json.loads(outputs[0])
    # Candidates are sorted, not left in the order they were typed.
    assert payload["candidate_max_lengths"] == [192, 256, 512]
    for population in payload["populations"]:
        assert [c["max_length"] for c in population["antibody_candidates"]] == [192, 256, 512]
    # Populations are ordered split-major, then by the fixed RECORD_TYPES order.
    keys = [(p["split"], p["record_type"]) for p in payload["populations"]]
    assert keys == [
        ("train", "single_heavy"),
        ("train", "paired"),
        ("val", "single_light"),
        ("val", "paired"),
    ]


def test_splits_filter_restricts_the_census(census, tmp_path: Path):
    rows = [
        _paired_row(heavy_len=15, light_len=12, split="train"),
        _paired_row(heavy_len=15, light_len=12, split="val"),
    ]
    result = _run(census, tmp_path, rows, [32], splits=["train"])
    assert result["rows_censused"] == 1
    assert result["rows_skipped_by_split_filter"] == 1
    assert [p["split"] for p in result["populations"]] == ["train"]


def test_cli_rejects_non_positive_candidates(census, tmp_path: Path):
    data_path = _write_jsonl_gz(
        tmp_path / "corpus.jsonl.gz", [_paired_row(heavy_len=15, light_len=12)]
    )
    with pytest.raises(SystemExit):
        census.main(["--data-path", str(data_path), "--candidate-max-length", "0"])


def test_census_does_not_import_the_training_script(project_root: Path):
    """
    The census must not depend on ``scripts/mlm_train.py``.

    Importing it would drag in torch, the whole config surface, and a module that
    another agent is editing -- for two pure functions. The shared arithmetic lives
    in ``smallAntibodyGen.data.lengths`` precisely so this import is unnecessary.
    """
    source = (project_root.parents[1] / "scripts" / "context_length_census.py").read_text(
        encoding="utf-8"
    )
    assert "mlm_train" not in source


def test_census_does_not_instantiate_the_eager_dataset(project_root: Path):
    """
    ``OASSequenceDataset`` appends an entire split to ``self.records``. A
    full-corpus census through it would hold the corpus in memory; the census
    streams instead.
    """
    source = (project_root.parents[1] / "scripts" / "context_length_census.py").read_text(
        encoding="utf-8"
    )
    assert "OASSequenceDataset(" not in source


# --------------------------------------------------------------------------- #
# The extraction into data/lengths.py must be behavior-preserving.
# --------------------------------------------------------------------------- #
def test_mlm_train_still_exposes_the_moved_truncation_helpers(project_root: Path):
    """
    The production preflight and the six existing regression tests reach these
    through the ``mlm_train`` namespace. Moving them must leave that name bound to
    the very same function objects, not to copies.
    """
    from smallAntibodyGen.data import lengths

    mlm_train = _load_script(project_root, "mlm_train")
    assert mlm_train.summarize_length_truncation is lengths.summarize_length_truncation
    assert mlm_train.format_length_truncation_warning is lengths.format_length_truncation_warning

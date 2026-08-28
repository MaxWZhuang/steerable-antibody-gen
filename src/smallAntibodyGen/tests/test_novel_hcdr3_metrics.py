"""
Tests for the AB-11 novel-HCDR3 validation partition.

The measured problem: 78.60% of stage-4 validation rows have a heavy CDR3 that
appears verbatim in training, so `hcdr3_span_exact_match` as reported is ~79%
reachable by recall. The mitigation reports the same metrics a second time over
the rows where that is not true.

What must hold:

- with the knob off, the metric key set is EXACTLY what it was before, so no
  existing run's metrics.jsonl changes shape;
- with it on, the partition is real -- a memorized row counts toward the
  unpartitioned metric and not the novel one;
- the denominator travels with the metric, because ~382 rows is small enough
  that the number is uninterpretable without its n.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from smallAntibodyGen.tests.test_train_infra import load_mlm_train_module


@pytest.fixture
def mlm_train(project_root: Path):
    return load_mlm_train_module(project_root)


def _batch(mlm_train, tokenizer, spans: list[str], predict: list[str]):
    """
    Build a minimal batch whose HCDR3 spans are `spans` and whose argmax
    predictions are `predict`. Span layout is [CLS] + span, so start=1.
    """
    width = len(spans[0])
    batch_size = len(spans)
    seq_len = width + 2
    vocab = len(tokenizer.vocab)

    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    logits = torch.zeros(batch_size, seq_len, vocab)
    target_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for row, (truth, guess) in enumerate(zip(spans, predict)):
        for offset, (true_ch, guess_ch) in enumerate(zip(truth, guess)):
            pos = 1 + offset
            labels[row, pos] = tokenizer.vocab.index(true_ch)
            target_mask[row, pos] = 1
            logits[row, pos, tokenizer.vocab.index(guess_ch)] = 10.0

    return dict(
        logits=logits,
        labels=labels,
        hcdr3_target_mask=target_mask,
        hcdr3_token_start=torch.full((batch_size,), 1, dtype=torch.long),
        hcdr3_token_end=torch.full((batch_size,), 1 + width, dtype=torch.long),
        hcdr3_valid_mask=torch.ones(batch_size, dtype=torch.bool),
    )


def test_knob_off_leaves_the_metric_key_set_untouched(mlm_train, tokenizer):
    """An unmodified run must not gain keys; metrics.jsonl shape is a contract."""
    batch = _batch(mlm_train, tokenizer, ["AAAA"], ["AAAA"])
    counts = mlm_train.hcdr3_metric_counts(**batch)
    assert not any(key.startswith("hcdr3_novel") for key in counts)

    metrics = mlm_train.finalize_hcdr3_metrics(counts)
    assert set(metrics) == {
        "hcdr3_token_acc",
        "hcdr3_span_exact_match",
        "hcdr3_target_tokens",
        "hcdr3_valid_spans",
    }


def test_supplying_only_one_half_of_the_partition_is_a_no_op(mlm_train, tokenizer):
    """
    Both arguments are required together. If a reference without an
    `id_to_token` silently enabled the partition, every span would decode to
    nothing, match no reference entry, and the whole validation set would be
    reported as novel -- the metric would look clean precisely when it is not.
    """
    batch = _batch(mlm_train, tokenizer, ["AAAA"], ["AAAA"])
    only_reference = mlm_train.hcdr3_metric_counts(
        **batch, novel_reference=frozenset({"AAAA"})
    )
    only_vocab = mlm_train.hcdr3_metric_counts(**batch, id_to_token=list(tokenizer.vocab))
    assert not any(key.startswith("hcdr3_novel") for key in only_reference)
    assert not any(key.startswith("hcdr3_novel") for key in only_vocab)


def test_a_memorized_row_is_excluded_from_the_novel_partition(mlm_train, tokenizer):
    """
    The partition, stated directly. Two rows, both predicted perfectly. One's
    HCDR3 is in the training reference, the other's is not. The unpartitioned
    metric sees both; the novel metric must see only the second.
    """
    batch = _batch(mlm_train, tokenizer, ["AAAA", "CCCC"], ["AAAA", "CCCC"])
    counts = mlm_train.hcdr3_metric_counts(
        **batch,
        novel_reference=frozenset({"AAAA"}),
        id_to_token=list(tokenizer.vocab),
    )
    assert counts["hcdr3_valid_spans"] == 2
    assert counts["hcdr3_novel_valid_spans"] == 1
    assert counts["hcdr3_exact_matches"] == 2
    assert counts["hcdr3_novel_exact_matches"] == 1


def test_the_partition_can_disagree_with_the_headline_number(mlm_train, tokenizer):
    """
    The case the whole mitigation exists for: perfect on the memorized row, wrong
    on the unseen one. The headline metric reads 0.5 while the honest number is
    0.0, and only the partition makes that visible.
    """
    batch = _batch(mlm_train, tokenizer, ["AAAA", "CCCC"], ["AAAA", "DDDD"])
    counts = mlm_train.hcdr3_metric_counts(
        **batch,
        novel_reference=frozenset({"AAAA"}),
        id_to_token=list(tokenizer.vocab),
    )
    metrics = mlm_train.finalize_hcdr3_metrics(counts)
    assert metrics["hcdr3_span_exact_match"] == 0.5
    assert metrics["hcdr3_novel_span_exact_match"] == 0.0
    assert metrics["hcdr3_novel_valid_spans"] == 1.0


def test_the_denominator_is_reported_beside_the_metric(mlm_train, tokenizer):
    """
    On the real corpus the novel subset is ~382 rows against 1,785. A rate that
    small is only interpretable next to its n, so the count is a first-class
    metric rather than something a reader has to reconstruct.
    """
    batch = _batch(mlm_train, tokenizer, ["AAAA", "CCCC", "EEEE"], ["AAAA", "CCCC", "EEEE"])
    counts = mlm_train.hcdr3_metric_counts(
        **batch,
        novel_reference=frozenset({"AAAA"}),
        id_to_token=list(tokenizer.vocab),
    )
    metrics = mlm_train.finalize_hcdr3_metrics(counts)
    assert metrics["hcdr3_novel_valid_spans"] == 2.0
    assert metrics["hcdr3_novel_target_tokens"] == 8.0


def test_an_empty_novel_subset_reports_nan_not_a_silent_perfect_score(
    mlm_train, tokenizer
):
    """
    If every validation row is memorized the novel metric has no denominator.
    NaN says "not measured"; 0.0 or 1.0 would both be read as a result.
    """
    batch = _batch(mlm_train, tokenizer, ["AAAA"], ["AAAA"])
    counts = mlm_train.hcdr3_metric_counts(
        **batch,
        novel_reference=frozenset({"AAAA"}),
        id_to_token=list(tokenizer.vocab),
    )
    metrics = mlm_train.finalize_hcdr3_metrics(counts)
    assert counts["hcdr3_novel_valid_spans"] == 0
    assert metrics["hcdr3_novel_span_exact_match"] != metrics["hcdr3_novel_span_exact_match"]


def test_config_default_is_off_and_the_flag_turns_it_on(mlm_train, tmp_path: Path):
    """Class-K discipline: the default reproduces the historical behavior."""
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    default = mlm_train.parse_args(["--data-path", str(data_path)])
    assert default.report_novel_hcdr3_metrics is False

    enabled = mlm_train.parse_args(
        ["--data-path", str(data_path), "--report-novel-hcdr3-metrics"]
    )
    assert enabled.report_novel_hcdr3_metrics is True

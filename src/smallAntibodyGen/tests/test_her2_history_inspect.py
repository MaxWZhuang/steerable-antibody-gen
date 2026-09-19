"""Recovering the original histories: windows, cycles, the reconstruction, the refusal.

The reconstruction test asserts *properties* of the recovered batch means -- that
one cycle partitions the chosen population, and that the seed actually selects the
partition -- rather than recomputing the permutation the implementation uses,
which would only check that the code agrees with itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_history as history
from smallAntibodyGen.experiments.her2_runtime import sha256

CHOSEN = 12
BATCH_PAIRS = 4
PAIRING_SEED = 40521842


def dpo_history(updates=6, *, pairs=BATCH_PAIRS):
    """A synthetic DPO history in the shape the original loop wrote."""
    rows = []
    per_cycle = CHOSEN // BATCH_PAIRS
    for update in range(1, updates + 1):
        position = update - 1
        rows.append({"update": update, "cycle": position // per_cycle,
                     "batch": position % per_cycle,
                     "pairs": pairs,
                     "loss": 0.7 - 0.05 * update,
                     "mean_margin": 0.1 * update,
                     "accuracy": min(1.0, 0.5 + 0.1 * update),
                     "mean_policy_chosen": -30.0 - 0.5 * update,
                     "mean_policy_rejected": -35.0 - update,
                     "gradient_norm": 0.3,
                     "learning_rate": 1e-5 * min(1.0, (update + 1) / 4),
                     "update_gpu_seconds": 0.15,
                     "cumulative_gpu_seconds": 0.15 * update})
    return rows


def write_run(root, name, *, method="dpo", seed=20260918, updates=6, budgets=True,
              batch_sequences=2 * BATCH_PAIRS, cache_values=None, parent_sha="a" * 64):
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = dpo_history(updates)
    if method != "dpo":
        dropped = ("mean_margin", "accuracy", "mean_policy_chosen", "mean_policy_rejected")
        rows = [dict({k: v for k, v in row.items() if k not in dropped}, nll_per_residue=1.5)
                for row in rows]
    document = {"method": method, "seed": seed, "updates": updates, "history": rows,
                "batch_sequences": batch_sequences,
                "identity": {"parent_sha256": parent_sha, "method": method, "seed": seed},
                "gpu_seconds": 0.15 * updates, "precharged_gpu_seconds": 68.8,
                "budgets": {"180.0": {"val_pair_metrics": {
                    "chosen_nll_per_residue": 3.197, "mean_chosen_sum_log_probability": -31.97,
                    "pairs": 25722}}} if budgets else {}}
    if method == "dpo":
        document["pairing"] = {"batch_pairs": batch_sequences // 2}
    if cache_values is not None:
        document["reference_cache"] = write_cache(directory, cache_values, parent_sha=parent_sha)
    path = directory / "summary.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def write_cache(directory, values, *, parent_sha="a" * 64):
    """A frozen reference cache with the identity sidecar the original writes beside it."""
    from smallAntibodyGen.experiments.her2_preferences import array_digest
    values = np.asarray(values, dtype=np.float64)
    np.save(directory / "reference_cache.npy", values)
    identity = {"schema_version": "her2-reference-cache/1", "checkpoint_sha256": parent_sha,
                "config_sha256": "c" * 64, "scaffold_prefix_sha256": "d" * 64,
                "probability_convention": "sum_log_probability_over_10_core_positions_20way_renormalized",
                "core_order_sha256": "e" * 64, "rows": int(values.size),
                "values_sha256": array_digest(values)}
    (directory / "reference_cache.json").write_text(json.dumps({
        "schema_version": "her2-reference-cache/1", "identity": identity,
        "creation_gpu_seconds": 68.8}), encoding="utf-8")
    return {"identity": identity, "rows": int(values.size), "creation_gpu_seconds": 68.8,
            "fresh_parity": {"max_absolute_difference": 1e-7}}


# ---------------------------------------------------------------------------
# windows and cycles
# ---------------------------------------------------------------------------

def test_windows_are_means_over_the_declared_inclusive_ranges():
    rows = dpo_history(12)
    window = history.window_statistics(rows, "mean_margin", start=1, end=3)
    assert window["updates"] == 3
    assert window["mean"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert window["min"] == pytest.approx(0.1) and window["max"] == pytest.approx(0.3)


def test_an_empty_window_or_an_absent_field_is_reported_not_skipped():
    rows = dpo_history(3)
    assert history.window_statistics(rows, "mean_margin", start=100, end=500)["updates"] == 0
    assert "note" in history.window_statistics(rows, "nll_per_residue", start=1, end=3)


def test_the_window_summary_covers_every_declared_range_and_the_tail():
    summary = history.window_summary(dpo_history(600), history.DPO_FIELDS)
    assert set(summary) == {f"{a}-{b}" for a, b in history.DEFAULT_WINDOWS} | {"last_100"}
    assert summary["101-500"]["mean_margin"]["updates"] == 400
    assert summary["last_100"]["mean_margin"]["start"] == 501


def test_cycle_boundaries_record_where_each_pass_over_the_population_began():
    boundaries = history.cycle_boundaries(dpo_history(9))
    assert sorted(boundaries, key=int) == ["0", "1", "2"]
    assert boundaries["0"] == {"first_update": 1, "last_update": 3, "updates": 3,
                               "first_cumulative_gpu_seconds": pytest.approx(0.15),
                               "last_cumulative_gpu_seconds": pytest.approx(0.45)}


def test_the_original_learning_rate_is_relabelled_as_the_next_one(tmp_path):
    path = write_run(tmp_path, "dpo_seed20260918")
    document = history.summarize_run(path)
    assert document["learning_rate"]["actually"] == "learning_rate_next"
    assert "steps the scheduler before reading it" in document["learning_rate"]["note"]
    assert document["learning_rate"]["first"] == pytest.approx(1e-5 * 0.5)


def test_a_summary_whose_update_count_disagrees_with_its_history_is_refused(tmp_path):
    path = write_run(tmp_path, "dpo_seed20260918", updates=6)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["updates"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="history rows"):
        history.summarize_run(path)


def test_the_first_fixed_validation_measurement_is_labelled_as_the_first(tmp_path):
    path = write_run(tmp_path, "dpo_seed20260918")
    first = history.summarize_run(path)["fixed_validation_first_measurement"]
    assert first["target_gpu_seconds"] == 180.0
    assert first["chosen_nll_per_residue"] == pytest.approx(3.197)
    assert "no onset time can be read off it" in first["note"]


# ---------------------------------------------------------------------------
# the reconstruction
# ---------------------------------------------------------------------------

def test_one_cycle_of_reconstructed_batches_partitions_the_chosen_population():
    """Each cycle visits every chosen row once, so the batch means average to the whole."""
    # Powers of two: distinct subsets have distinct sums, so equal batch means would
    # mean the batches are the same rows rather than a numerical coincidence.
    values = 2.0 ** np.arange(CHOSEN)
    rows = history.reconstruct_batch_chosen_drop(
        dpo_history(3), values, chosen_count=CHOSEN, pairing_seed=PAIRING_SEED,
        batch_pairs=BATCH_PAIRS)
    assert [row["batch"] for row in rows] == [0, 1, 2]
    assert all(row["pairs"] == BATCH_PAIRS for row in rows)
    means = np.array([row["reference_mean_chosen"] for row in rows])
    assert means.mean() == pytest.approx(values.mean()), "the three batches partition the cycle"
    assert len(set(means.tolist())) == 3, "and they are different batches, not the same rows"


def test_the_reconstruction_depends_on_the_declared_pairing_seed():
    values = 2.0 ** np.arange(CHOSEN)
    mine = history.reconstruct_batch_chosen_drop(
        dpo_history(3), values, chosen_count=CHOSEN, pairing_seed=PAIRING_SEED,
        batch_pairs=BATCH_PAIRS)
    other = history.reconstruct_batch_chosen_drop(
        dpo_history(3), values, chosen_count=CHOSEN, pairing_seed=PAIRING_SEED + 1,
        batch_pairs=BATCH_PAIRS)
    assert [row["reference_mean_chosen"] for row in mine] != \
        [row["reference_mean_chosen"] for row in other]


def test_the_drop_is_the_reference_minus_the_policy_and_positive_means_worse():
    values = np.full(CHOSEN, -30.0)
    rows = history.reconstruct_batch_chosen_drop(
        dpo_history(3), values, chosen_count=CHOSEN, pairing_seed=PAIRING_SEED,
        batch_pairs=BATCH_PAIRS)
    # The synthetic policy means are -30.5, -31.0, -31.5 against a flat -30 reference.
    assert [row["batch_mean_chosen_drop"] for row in rows] == pytest.approx([0.5, 1.0, 1.5])


def test_the_first_update_above_the_threshold_is_reported_with_the_window_means():
    values = np.full(CHOSEN, -30.0)
    rows = history.reconstruct_batch_chosen_drop(
        dpo_history(6), values, chosen_count=CHOSEN, pairing_seed=PAIRING_SEED,
        batch_pairs=BATCH_PAIRS)
    summary = history.summarize_batch_chosen_drop(rows, threshold=1.0)
    assert summary["available"] is True
    assert summary["first_update_above_threshold"] == 3, "update 2 is exactly 1.0 and does not"
    assert summary["windows"]["1-10"]["mean_batch_chosen_drop"] == pytest.approx(
        np.mean([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    assert "does not date the first crossing of any gate" in summary["note"]


def test_a_missing_reference_cache_is_reported_unavailable_not_estimated(tmp_path):
    path = write_run(tmp_path, "dpo_seed20260918")
    document = history.summarize_run(path, reference_cache=tmp_path / "absent.npy",
                                     pairing_seed_base=20260924, batch_pairs=BATCH_PAIRS,
                                     chosen_count=CHOSEN)
    assert document["batch_chosen_drop"]["available"] is False
    assert "nothing is estimated in its place" in document["batch_chosen_drop"]["reason"]


def test_a_continued_sft_run_reports_its_sequence_batch_with_pairs_inapplicable(tmp_path):
    """CX-32: those summaries record batch_sequences; "recorded batch None" read as missing."""
    path = write_run(tmp_path, "continued_sft_seed20260918", method="continued_sft")
    document = history.summarize_run(path, pairing_seed_base=20260924, batch_pairs=BATCH_PAIRS,
                                     chosen_count=CHOSEN)
    assert document["batch"]["batch_sequences"] == 2 * BATCH_PAIRS
    assert document["batch"]["batch_pairs"] is None
    assert document["batch"]["pairs_applicable"] is False
    assert "chosen positives only" in document["batch"]["pairs_note"]
    assert document["batch_chosen_drop"]["available"] is False
    assert "no pairing" in document["batch_chosen_drop"]["reason"]
    # The declared pair batch of the current config does not enter an SFT record.
    assert "recorded_batch_pairs" not in document["batch"]


def test_a_dpo_run_reports_its_batch_even_with_no_cache_on_disk(tmp_path):
    path = write_run(tmp_path, "dpo_seed20260918")
    document = history.summarize_run(path, reference_cache=tmp_path / "absent.npy",
                                     pairing_seed_base=20260924, batch_pairs=BATCH_PAIRS,
                                     chosen_count=CHOSEN)
    assert document["batch"]["batch_pairs"] == BATCH_PAIRS
    assert document["batch_chosen_drop"]["available"] is False


def test_a_reference_cache_shorter_than_the_population_is_refused():
    with pytest.raises(ValueError, match="fewer than the"):
        history.reconstruct_batch_chosen_drop(dpo_history(1), np.zeros(3), chosen_count=CHOSEN,
                                              pairing_seed=PAIRING_SEED, batch_pairs=BATCH_PAIRS)


# ---------------------------------------------------------------------------
# what cannot be recovered
# ---------------------------------------------------------------------------

def test_the_mean_margin_does_not_determine_the_mean_coefficient():
    """Why the missing-diagnostics record exists, demonstrated on the arithmetic itself.

    Two per-pair margin distributions with the SAME mean give materially different
    mean DPO coefficients, because ``sigma`` is nonlinear. Anything reconstructed
    from the stored ``mean_margin`` alone would be a fabrication.
    """
    beta = 0.5
    tight = np.full(8, 2.0)
    spread = np.concatenate([np.full(4, -6.0), np.full(4, 10.0)])
    assert tight.mean() == pytest.approx(spread.mean())

    def mean_coefficient(margins):
        return float(np.mean(beta / (1.0 + np.exp(beta * margins))))

    # Both values written out independently: sigma(-beta * m) at m = 2 for the tight
    # batch, and the average of the two saturating ends for the spread one.
    tight_mean, spread_mean = mean_coefficient(tight), mean_coefficient(spread)
    assert tight_mean == pytest.approx(0.5 / (1 + np.exp(1.0)))
    assert tight_mean == pytest.approx(0.13447071068499755)
    assert spread_mean == pytest.approx(0.25 * (1 / (1 + np.exp(-3.0)) + 1 / (1 + np.exp(5.0))))
    assert spread_mean == pytest.approx(0.2398167444366795)
    # 78% higher on identical mean margins. Not a rounding artifact, and not 2x
    # either: the claim is that the mean margin does not determine this number.
    assert spread_mean / tight_mean == pytest.approx(1.7834, abs=1e-3)
    assert spread_mean - tight_mean == pytest.approx(0.1053460337516819)
    record = history.MISSING_DIAGNOSTICS["per_pair_coefficient_distribution"]
    assert record["recoverable"] is False
    assert "mean_margin only" in record["stored"]
    assert "no distribution is reconstructed from the mean" in record["not_done"]


def test_the_missing_early_validation_measurement_is_named_as_missing():
    record = history.MISSING_DIAGNOSTICS["fixed_validation_likelihood_before_180s"]
    assert record["recoverable"] is False
    assert "bracketed, not dated" in record["not_done"]


# ---------------------------------------------------------------------------
# the whole campaign, read-only
# ---------------------------------------------------------------------------

def snapshot(root):
    """Content hashes keyed by POSIX-relative path, so the keys read the same on Windows."""
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def test_inspecting_the_original_campaign_writes_nothing_under_its_output_root(tmp_path):
    original = tmp_path / "her2_posttrain_20260918" / "continuation"
    runs = {}
    for method in ("dpo", "continued_sft"):
        for seed in (20260918, 20260919):
            name = f"{method}_seed{seed}"
            runs[name] = write_run(original, name, method=method, seed=seed)
    before = snapshot(original)
    document = history.inspect_original_campaign(runs, root=original)
    after = snapshot(original)
    assert before == after, "the original campaign's artifacts are read-only inputs"
    assert set(document["runs"]) == set(runs)
    assert set(document["inputs"]) == {f"{name}::summary" for name in runs}
    assert all(entry["sha256"] == before[f"{name}/summary.json"]
               for name, entry in ((key.split("::")[0], value)
                                   for key, value in document["inputs"].items()))
    assert "writes nothing under its output root" in document["read_only"]


def test_the_inputs_a_later_stage_rechecks_include_the_reference_cache_and_its_sidecar(tmp_path):
    """The reconstruction is a claim about those cache values, so they are inputs too."""
    original = tmp_path / "continuation"
    values = np.full(CHOSEN, -30.0)
    path = write_run(original, "dpo_seed20260918", cache_values=values)
    document = history.inspect_original_campaign(
        {"dpo_seed20260918": path}, root=original,
        reference={"cache_paths": {"dpo_seed20260918": original / "dpo_seed20260918"
                                   / "reference_cache.npy"},
                   "pairing_seed_base": 20260924, "batch_pairs": BATCH_PAIRS,
                   "chosen_count": CHOSEN})
    assert set(document["inputs"]) == {"dpo_seed20260918::summary",
                                       "dpo_seed20260918::reference_cache",
                                       "dpo_seed20260918::reference_cache_sidecar"}
    for entry in document["inputs"].values():
        assert Path(entry["path"]).is_file()
        assert entry["sha256"] == sha256(entry["path"])
    assert document["runs"]["dpo_seed20260918"]["batch_chosen_drop"]["available"] is True


def test_the_reconstruction_verifies_the_cache_against_what_the_run_recorded(tmp_path):
    """Re-hashing an array against its own sidecar proves only that the file is itself."""
    original = tmp_path / "continuation"
    values = np.full(CHOSEN, -30.0)
    path = write_run(original, "dpo_seed20260918", cache_values=values)
    cache = original / "dpo_seed20260918" / "reference_cache.npy"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert history.verify_reference_provenance(cache, summary)[1]["rows"] == CHOSEN

    # A cache produced by a different parent checkpoint, sidecar and all.
    other = write_cache(original / "dpo_seed20260918", values, parent_sha="b" * 64)
    with pytest.raises(ValueError, match="does not carry the identity this run recorded"):
        history.verify_reference_provenance(cache, summary)
    assert other["identity"]["checkpoint_sha256"] == "b" * 64

    # And a cache whose bytes moved after the sidecar was written.
    summary["reference_cache"] = other
    np.save(cache, values + 1.0)
    with pytest.raises(ValueError, match="do not match the digest"):
        history.verify_reference_provenance(cache, summary)


def test_a_missing_identity_sidecar_is_not_a_frozen_reference(tmp_path):
    original = tmp_path / "continuation"
    path = write_run(original, "dpo_seed20260918", cache_values=np.full(CHOSEN, -30.0))
    (original / "dpo_seed20260918" / "reference_cache.json").unlink()
    summary = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="is missing"):
        history.verify_reference_provenance(
            original / "dpo_seed20260918" / "reference_cache.npy", summary)


def test_the_batch_size_comes_from_the_run_not_from_the_current_config(tmp_path):
    """A config default is not evidence about what a finished run did."""
    original = tmp_path / "continuation"
    path = write_run(original, "dpo_seed20260918", batch_sequences=2 * BATCH_PAIRS)
    summary = json.loads(path.read_text(encoding="utf-8"))
    recorded = history.recorded_batch_size(path, summary, batch_pairs=BATCH_PAIRS)
    assert recorded["batch_pairs"] == BATCH_PAIRS
    assert recorded["batch_sequences"] == 2 * BATCH_PAIRS
    with pytest.raises(ValueError, match="is not evidence about what that run did"):
        history.recorded_batch_size(path, summary, batch_pairs=BATCH_PAIRS * 2)
    summary.pop("batch_sequences")
    with pytest.raises(ValueError, match="does not record the batch size"):
        history.recorded_batch_size(path, summary)


def test_reconstructing_at_the_wrong_batch_size_is_refused_by_the_recorded_pair_counts(tmp_path):
    """The rows would be real rows, just not the ones that update trained on."""
    values = 2.0 ** np.arange(CHOSEN)
    with pytest.raises(ValueError, match="not the one that run trained at"):
        history.reconstruct_batch_chosen_drop(
            dpo_history(3), values, chosen_count=CHOSEN, pairing_seed=PAIRING_SEED,
            batch_pairs=BATCH_PAIRS + 1)


def test_the_run_summary_reconstructs_at_its_own_recorded_batch_size(tmp_path):
    original = tmp_path / "continuation"
    path = write_run(original, "dpo_seed20260918", cache_values=np.full(CHOSEN, -30.0))
    document = history.summarize_run(
        path, reference_cache=original / "dpo_seed20260918" / "reference_cache.npy",
        pairing_seed_base=20260924, batch_pairs=BATCH_PAIRS, chosen_count=CHOSEN)
    assert document["batch"]["batch_pairs"] == BATCH_PAIRS
    assert document["batch_chosen_drop"]["batch_pairs"] == BATCH_PAIRS
    assert document["reference_cache"]["matches_summary_identity"] is True
    with pytest.raises(ValueError, match="is not evidence about what that run did"):
        history.summarize_run(
            path, reference_cache=original / "dpo_seed20260918" / "reference_cache.npy",
            pairing_seed_base=20260924, batch_pairs=BATCH_PAIRS * 2, chosen_count=CHOSEN)


def test_each_cycle_carries_its_own_regime_and_the_next_update_learning_rate(tmp_path):
    """Per-cycle batch-mean aggregates, under names that say which rate was recorded."""
    cycles = history.cycle_summary(dpo_history(9), history.DPO_FIELDS)
    assert sorted(cycles, key=int) == ["0", "1", "2"]
    first = cycles["0"]
    assert first["first_update"] == 1 and first["last_update"] == 3
    assert set(first["metrics"]) >= {"loss", "mean_margin", "accuracy", "mean_policy_chosen",
                                     "mean_policy_rejected", "gradient_norm",
                                     "learning_rate_next"}
    assert first["metrics"]["mean_margin"]["mean"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert first["metrics"]["accuracy"]["mean"] == pytest.approx((0.6 + 0.7 + 0.8) / 3)
    assert first["learning_rate_next"]["first"] == pytest.approx(1e-5 * 0.5)
    assert "NEXT update" in first["learning_rate_next"]["note"]
    assert "learning_rate" not in first["metrics"], "relabelled, not reported under both names"
    # The fixed validation gate is a different measurement on a fixed population.
    assert "regime" in first["aggregate_note"] and "batches change" in first["aggregate_note"]


def test_the_recovered_report_carries_each_input_hash_and_its_update_count(tmp_path):
    path = write_run(tmp_path, "dpo_seed20260918", updates=6)
    document = history.summarize_run(path)
    assert document["summary"]["sha256"] == sha256(path)
    assert document["updates"] == 6 and document["history_rows"] == 6
    assert document["method"] == "dpo" and document["seed"] == 20260918
    assert document["budgets_reached"] == ["180.0"]
    assert document["missing_diagnostics"] is history.MISSING_DIAGNOSTICS


def test_a_continued_sft_history_is_summarized_over_its_own_fields(tmp_path):
    path = write_run(tmp_path, "continued_sft_seed20260918", method="continued_sft")
    document = history.summarize_run(path)
    assert document["windows"]["1-10"]["nll_per_residue"]["updates"] == 6
    assert "mean_margin" not in document["windows"]["1-10"]

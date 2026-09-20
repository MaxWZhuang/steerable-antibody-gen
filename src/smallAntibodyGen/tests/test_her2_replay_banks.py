"""Bank generation, the frozen teacher cache, and the refusals that keep it honest.

A cache that is transposed, re-sorted, built from other weights or silently
renormalized still produces a loss. Each test here is one of those, plus the two
properties that are easy to state and easy to lose: duplicates are the sampling
law, and accidental overlap between two independently sampled banks is reported
rather than removed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_replay_banks as banks
from smallAntibodyGen.experiments import her2_support_paths as paths


def synthetic_cache(rows=6, seed=3):
    """A well-formed cache and the cores it describes, in the production dtypes."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, 10, 20, generator=generator)
    logs = torch.log_softmax(logits, dim=-1)
    index = np.random.default_rng(seed).integers(0, 20, size=(rows, 10)).astype(np.int8)
    return (logs.exp().numpy().astype(np.float32), logs.numpy().astype(np.float32), index)


def test_the_cached_logs_reproduce_the_sequence_density_of_each_draw():
    probabilities, logs, index = synthetic_cache()
    derived = banks.cached_sequence_log_probability(logs, index)
    manual = np.array([sum(float(logs[row, position, index[row, position]])
                           for position in range(10)) for row in range(index.shape[0])])
    assert np.allclose(derived, manual, atol=1e-6)


def test_the_cache_cross_check_catches_a_cache_built_from_other_weights():
    probabilities, logs, index = synthetic_cache()
    sampler = banks.cached_sequence_log_probability(logs, index)
    block = banks.validate_teacher_cache(probabilities, logs, index,
                                         sampler_sum_log_probability=sampler,
                                         label="probe")
    assert block["sampler_cross_check"]["max_abs_error"] < 1e-5
    other_probabilities, other_logs, _ = synthetic_cache(seed=9)
    with pytest.raises(ValueError, match="cached teacher logs versus the sampler"):
        banks.validate_teacher_cache(other_probabilities, other_logs, index,
                                     sampler_sum_log_probability=sampler, label="probe")


def test_a_reordered_cache_is_refused_by_the_cross_check():
    """Same numbers, different row order: still a different artifact."""
    probabilities, logs, index = synthetic_cache()
    sampler = banks.cached_sequence_log_probability(logs, index)
    shuffled = np.ascontiguousarray(logs[::-1])
    with pytest.raises(ValueError, match="cached teacher logs versus the sampler"):
        banks.validate_teacher_cache(np.ascontiguousarray(probabilities[::-1]), shuffled, index,
                                     sampler_sum_log_probability=sampler, label="probe")


def test_a_transposed_cache_is_refused_before_anything_is_computed():
    probabilities, logs, index = synthetic_cache()
    with pytest.raises(ValueError, match="expected"):
        banks.validate_teacher_cache(probabilities.transpose(0, 2, 1),
                                     logs.transpose(0, 2, 1), index,
                                     sampler_sum_log_probability=np.zeros(6), label="probe")


def test_cached_sequence_log_probability_refuses_a_mismatched_block():
    probabilities, logs, index = synthetic_cache(rows=6)
    with pytest.raises(ValueError, match="expected"):
        banks.cached_sequence_log_probability(logs[:3], index)


def test_overlap_is_measured_reported_and_never_removed():
    replay_index = np.array([[0] * 10, [1] * 10, [2] * 10], dtype=np.int8)
    monitor_index = np.array([[1] * 10, [3] * 10, [1] * 10], dtype=np.int8)
    block = banks.bank_overlap(replay_index, monitor_index)
    assert block["shared_unique_cores"] == 1
    assert block["monitor_draws_also_in_replay"] == 2
    assert block["action"] == "none"
    assert "sampling law" in block["note"]


def test_probe_rows_are_deterministic_and_bounded():
    first = banks.probe_rows(1000, count=8, seed=777000501)
    second = banks.probe_rows(1000, count=8, seed=777000501)
    assert np.array_equal(first, second) and first.size == 8
    assert banks.probe_rows(4, count=8, seed=1).size == 4


def test_bank_identity_binds_the_parent_the_seed_and_the_order():
    probabilities, logs, index = synthetic_cache()
    bank = banks.Bank(role="replay", parent_seed=20260918, parent_id="parent::x",
                      parent_state_sha256="a" * 64, draw_seed=777000101, temperature=1.0,
                      index=index, sampler_sum_log_probability=np.zeros(index.shape[0]))
    identity = banks.bank_identity(bank, campaign_id="c", freeze_commit="deadbeef")
    assert identity["order_sha256"] == paths.array_digest(index)
    assert identity["draw_seed"] == 777000101 and identity["freeze_commit"] == "deadbeef"
    document = bank.document()
    assert document["duplicates_retained"] is True
    assert document["unique_cores"] + document["duplicate_draws"] == document["rows"]


def test_an_unknown_bank_role_is_refused():
    with pytest.raises(ValueError, match="Unknown bank role"):
        banks.draw_bank(None, role="validation", parent_seed=1, parent_id="x",
                        parent_state_sha256="y", rows=1, draw_seed=1)


# ---------------------------------------------------------------------------
# the inherited diversity reference
# ---------------------------------------------------------------------------

def write_draws(path, cores, *, ordered=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    order = range(len(cores)) if ordered else reversed(range(len(cores)))
    lines = ["draw_index,core"] + [f"{i},{core}" for i, core in zip(order, cores)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths.sha256_file(path)


def test_the_inherited_reference_is_accepted_only_at_its_published_digest(tmp_path):
    cores = ["ACDEFGHIKL", "ACDEFGHIKM", "ACDEFGHIKL"]
    target = tmp_path / "parent_draws_policy_sft_seed20260918.csv"
    digest = write_draws(target, cores)
    index, record = banks.historical_reference_document(
        target, expected_sha256=digest, expected_rows=3, label="inherited")
    assert index.shape == (3, 10)
    assert record["role"] == "inherited_diversity_reference"
    assert "preservation diagnostics" in record["not_used_for"]
    with pytest.raises(ValueError, match="published audit input"):
        banks.historical_reference_document(target, expected_sha256="0" * 64, expected_rows=3,
                                            label="inherited")


def test_the_inherited_reference_refuses_a_wrong_row_count_or_a_re_sorted_file(tmp_path):
    cores = ["ACDEFGHIKL", "ACDEFGHIKM"]
    target = tmp_path / "draws.csv"
    digest = write_draws(target, cores)
    with pytest.raises(ValueError, match="rows, expected 3"):
        banks.historical_reference_document(target, expected_sha256=digest, expected_rows=3,
                                            label="inherited")
    scrambled = tmp_path / "scrambled.csv"
    digest = write_draws(scrambled, cores, ordered=False)
    with pytest.raises(ValueError, match="draw_index is not the exact"):
        banks.historical_reference_document(scrambled, expected_sha256=digest, expected_rows=2,
                                            label="inherited")


def test_a_missing_inherited_reference_is_named_not_worked_around(tmp_path):
    with pytest.raises(ValueError, match="is missing at"):
        banks.historical_reference_document(tmp_path / "absent.csv", expected_sha256="0" * 64,
                                            expected_rows=1, label="inherited")


# ---------------------------------------------------------------------------
# the banks manifest is bound to the freeze that declared the rule
# ---------------------------------------------------------------------------

def test_the_banks_manifest_names_the_freeze_it_was_generated_under():
    manifest = banks.banks_manifest({"seed1::replay_bank": {"rows": 10}}, campaign_id="c",
                                    freeze_commit="abc", freeze_sha256="def",
                                    overlap={}, timings={})
    assert manifest["bank_count"] == 1
    assert banks.require_banks_bound(manifest, freeze_commit="abc", freeze_sha256="def")
    assert "before these bytes existed" in manifest["ordering_note"]


def test_banks_from_another_freeze_are_refused():
    manifest = banks.banks_manifest({}, campaign_id="c", freeze_commit="abc",
                                    freeze_sha256="def", overlap={}, timings={})
    with pytest.raises(ValueError, match="bound to the specification"):
        banks.require_banks_bound(manifest, freeze_commit="xyz", freeze_sha256="def")


# ---------------------------------------------------------------------------
# the generated bytes are checked against the manifest, not against themselves
# ---------------------------------------------------------------------------

def generated_banks(tmp_path, seeds=(20260918,)):
    """A run directory holding the artifacts one banks stage would have produced.

    The identity sidecars are here beside their arrays because the stage writes
    them and nothing can read the arrays without them: ``load_parent_reference``
    and ``load_reference_cache`` refuse a missing identity document, and
    ``read_shard`` refuses a container whose completion record is absent.
    """
    entries = {}

    def produce(key, relative, *, role, **fields):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative.encode())
        entries[key] = {"file": relative, "sha256": paths.sha256_file(target), "role": role,
                        **fields}

    for seed in seeds:
        for role in ("replay", "monitor"):
            produce(f"seed{seed}::{role}_bank", f"banks/seed{seed}/{role}/bank.npz",
                    role=role, order_sha256="0" * 64)
            produce(f"seed{seed}::{role}_bank{banks.SHARD_RECORD_SUFFIX}",
                    f"banks/seed{seed}/{role}/bank{paths.SHARD_RECORD}",
                    role=f"{role}_bank_record")
        for stem, suffix, key in (("parent_validation_reference", ".npz",
                                   "parent_validation_reference"),
                                  ("ipo_reference_cache", ".npy", "ipo_reference_cache")):
            produce(f"seed{seed}::{key}", f"banks/seed{seed}/{stem}{suffix}", role=key)
            produce(f"seed{seed}::{key}{banks.SIDECAR_SUFFIX}",
                    f"banks/seed{seed}/{stem}.json", role=f"{key}_sidecar")
    manifest = banks.banks_manifest(entries, campaign_id="c", freeze_commit="abc",
                                    freeze_sha256="def", overlap={}, timings={})
    return manifest, (lambda logical: tmp_path / logical)


def test_every_role_seed_and_sidecar_is_verified_against_the_completed_manifest(tmp_path):
    manifest, resolve = generated_banks(tmp_path)
    report = banks.verify_bank_artifacts(
        manifest, resolve=resolve, expected_keys=banks.expected_bank_keys([20260918]))
    assert report["artifacts_checked"] == 8, "four arrays and the four documents beside them"
    assert all(row["matches"] for row in report["artifacts"])
    checked = {row["artifact"] for row in report["artifacts"]}
    assert "seed20260918::parent_validation_reference_sidecar" in checked
    assert "seed20260918::replay_bank_record" in checked


def test_a_deleted_identity_sidecar_is_caught_rather_than_silently_uncovered(tmp_path):
    """The claim "every sidecar was re-hashed" is only true if they are named.

    They were not: the manifest held four array entries and the loop over it could
    not see a deleted ``parent_validation_reference.json`` at all, while the
    report said it had checked one.
    """
    manifest, resolve = generated_banks(tmp_path)
    (tmp_path / "banks/seed20260918/parent_validation_reference.json").unlink()
    with pytest.raises(ValueError, match="absent from the run directory"):
        banks.verify_bank_artifacts(manifest, resolve=resolve,
                                    expected_keys=banks.expected_bank_keys([20260918]))


def test_an_edited_shard_record_beside_an_intact_container_is_caught(tmp_path):
    manifest, resolve = generated_banks(tmp_path)
    target = tmp_path / f"banks/seed20260918/replay/bank{paths.SHARD_RECORD}"
    target.write_bytes(b"a coherently rewritten completion record")
    with pytest.raises(ValueError, match="bytes differ from the completed banks manifest"):
        banks.verify_bank_artifacts(manifest, resolve=resolve,
                                    expected_keys=banks.expected_bank_keys([20260918]))


def test_a_deleted_bank_artifact_is_a_problem_rather_than_one_fewer_row(tmp_path):
    manifest, resolve = generated_banks(tmp_path)
    (tmp_path / "banks/seed20260918/monitor/bank.npz").unlink()
    with pytest.raises(ValueError, match="absent from the run directory"):
        banks.verify_bank_artifacts(manifest, resolve=resolve,
                                    expected_keys=banks.expected_bank_keys([20260918]))


def test_a_coherently_rewritten_bank_still_fails_the_saved_authority(tmp_path):
    """The failure a self-validating shard cannot see: regenerate both sides."""
    manifest, resolve = generated_banks(tmp_path)
    target = tmp_path / "banks/seed20260918/replay/bank.npz"
    target.write_bytes(b"regenerated, internally perfectly consistent")
    with pytest.raises(ValueError, match="bytes differ from the completed banks manifest"):
        banks.verify_bank_artifacts(manifest, resolve=resolve,
                                    expected_keys=banks.expected_bank_keys([20260918]))


def test_an_artifact_the_declared_grid_requires_and_the_manifest_never_recorded_is_missing(
        tmp_path):
    manifest, resolve = generated_banks(tmp_path, seeds=(20260918,))
    with pytest.raises(ValueError, match="never recorded it"):
        banks.verify_bank_artifacts(
            manifest, resolve=resolve,
            expected_keys=banks.expected_bank_keys([20260918, 20260919]))


def test_the_expected_keys_follow_the_declared_grid_and_the_declared_tasks():
    with_ipo = banks.expected_bank_keys([1, 2])
    assert len(with_ipo) == 16, "per seed: two banks, two reference arrays, four documents"
    assert "seed1::ipo_reference_cache" in with_ipo
    assert "seed1::ipo_reference_cache_sidecar" in with_ipo
    assert "seed1::parent_validation_reference_sidecar" in with_ipo
    assert "seed2::monitor_bank_record" in with_ipo
    without = banks.expected_bank_keys([1, 2], include_ipo_reference=False)
    assert len(without) == 12 and all("ipo_reference" not in key for key in without)

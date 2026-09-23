"""Bank roles, the selection refusal, and seed lineages that are reproducible.

``may_influence_selection`` is an object the code consults, not a comment. The
final 50k audit bank is the one artifact whose whole value comes from having
been drawn after the reported checkpoints were named, and a single early read
turns it into a development bank.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_banks as banks
from smallAntibodyGen.experiments import her2_replay_streams as streams
from smallAntibodyGen.tests.her2_nf_support import TinyPolicy, tiny_cores


def test_every_declared_role_exists_with_its_size_and_permission():
    assert set(banks.BANK_ROLES) == {"replay", "development_preservation",
                                     "final_preservation", "q_generation_screen",
                                     "q_generation_finalist"}
    assert banks.BANK_ROLES["replay"].rows == 100_000
    assert banks.BANK_ROLES["development_preservation"].rows == 10_000
    assert banks.BANK_ROLES["final_preservation"].rows == 50_000
    assert banks.BANK_ROLES["q_generation_finalist"].rows == 50_000
    allowed = {name for name, spec in banks.BANK_ROLES.items()
               if spec.may_influence_selection}
    assert allowed == {"replay", "development_preservation"}


def test_the_audit_bank_refuses_to_inform_a_selection_before_the_freeze():
    with pytest.raises(ValueError, match="may not influence selection"):
        banks.require_may_influence_selection("final_preservation", where="checkpoint choice")
    with pytest.raises(ValueError, match="names no checkpoints"):
        banks.require_may_influence_selection("final_preservation",
                                              freeze_record={"named_checkpoints": []},
                                              where="checkpoint choice")
    assert banks.require_may_influence_selection(
        "final_preservation", freeze_record={"named_checkpoints": [{"name": "a"}]},
        where="audit") is True
    assert banks.require_may_influence_selection("development_preservation",
                                                 where="pilot") is True


def test_seed_lineages_are_reproducible_and_outside_the_historical_band():
    first = banks.seed_lineage("replay_purge_20260918", spawn_key=(1, 20260918))
    second = banks.seed_lineage("replay_purge_20260918", spawn_key=(1, 20260918))
    assert first["seed"] == second["seed"]
    low, high = streams.HISTORICAL_SEED_BAND
    assert not (low <= first["seed"] <= high)
    assert first["entropy"] == banks.SEED_ENTROPY
    assert "not an independence test" in first["independence_note"]
    other = banks.seed_lineage("replay_matched_20260918", spawn_key=(1, 20260919))
    assert other["seed"] != first["seed"]


def test_the_seed_table_refuses_a_purpose_collision():
    table = banks.seed_table({"replay": [1, 0], "monitor": [2, 0]})
    assert set(table) == {"replay", "monitor"}
    with pytest.raises(ValueError, match="more than one purpose"):
        banks.seed_table({"replay": [1, 0], "duplicate": [1, 0]})


def test_streams_are_domain_separated_by_role_regime_parent_model_and_repeat():
    """Two banks differing in ONE of those must not share a sampling stream."""
    registry = banks.StreamRegistry()
    base = {"domain": "q_generation_finalist", "role": "q_generation_finalist",
            "regime": "purge", "parent": "parent::seed20260918", "model": "IPO_0@u1000",
            "repeat": 0}
    seeds = {"base": registry.stream(**base)["seed"]}
    for field, value in (("role", "q_generation_screen"), ("regime", "matched"),
                         ("parent", "parent::seed20260919"), ("model", "IPO_FKL@u1000"),
                         ("repeat", 1)):
        seeds[field] = registry.stream(**dict(base, **{field: value}))["seed"]
    assert len(set(seeds.values())) == len(seeds), seeds
    # The same descriptor twice is the same stream, not a new one.
    assert registry.stream(**base)["seed"] == seeds["base"]
    assert registry.document()["count"] == 6
    assert "not an independence test" in registry.document()["independence_note"]


def test_a_stream_registry_refuses_an_unknown_domain():
    with pytest.raises(ValueError, match="unknown stream domain"):
        banks.StreamRegistry().stream(domain="vibes", role="x")


def test_the_screen_and_the_finalist_repeat_zero_do_not_collide():
    """Both used ``spawn_key=(5, repeat, key(name))`` and differed in nothing."""
    registry = banks.StreamRegistry()
    screen = registry.stream(domain="q_generation_screen", role="q_generation_screen",
                             model="IPO_0@u1000_seed20260918", repeat=0)
    finalist = registry.stream(domain="q_generation_finalist",
                               role="q_generation_finalist",
                               model="IPO_0@u1000_seed20260918", repeat=0)
    assert screen["seed"] != finalist["seed"]


def test_the_name_key_is_wide_and_is_never_the_process_hash():
    """A five-digit modulus over forty model names has a real collision chance."""
    key = banks.text_key("A_IPO_FKL@u1000_seed20260918")
    assert len(key) == 3 and all(0 <= value < 2 ** 32 for value in key)
    assert banks.text_key("x") == banks.text_key("x")
    assert banks.text_key("x") != banks.text_key("y")


def test_a_persisted_bank_refuses_a_wrong_parent_role_or_seed(tmp_path):
    policy = TinyPolicy(seed=21)
    lineage = banks.seed_lineage("persist", spawn_key=(9, 21))
    bank, document = banks.draw_bank(policy, role="development_preservation",
                                     parent_id="parent::seed1", parent_state_sha256="aaa",
                                     seed_lineage_record=lineage, rows=32, batch_size=16)
    target = tmp_path / "bank.npz"
    banks.save_bank(target, bank, document)
    reloaded = banks.load_bank(target, role="development_preservation",
                               parent_id="parent::seed1", parent_state_sha256="aaa",
                               draw_seed=lineage["seed"])
    assert np.array_equal(reloaded["index"], bank.index)
    assert np.allclose(reloaded["parent_scores"], bank.sum_log_probability)
    with pytest.raises(ValueError, match="role is"):
        banks.load_bank(target, role="replay", parent_id="parent::seed1",
                        parent_state_sha256="aaa")
    with pytest.raises(ValueError, match="belongs to"):
        banks.load_bank(target, role="development_preservation", parent_id="parent::seed2",
                        parent_state_sha256="aaa")
    with pytest.raises(ValueError, match="different distribution"):
        banks.load_bank(target, role="development_preservation", parent_id="parent::seed1",
                        parent_state_sha256="bbb")
    with pytest.raises(ValueError, match="drawn at seed"):
        banks.load_bank(target, role="development_preservation", parent_id="parent::seed1",
                        parent_state_sha256="aaa", draw_seed=1)
    assert banks.load_bank(tmp_path / "absent.npz", role="replay", parent_id="p",
                           parent_state_sha256="a") is None


def test_a_persisted_bank_refuses_changed_scores_under_intact_cores(tmp_path):
    """Reproduced: the loader checked ``index_sha256`` and returned the cached
    probabilities unchecked, so an altered score array loaded cleanly. Those
    scores are the parent reference the preservation terms subtract from and the
    numbers the bounded calibration decides on."""
    from smallAntibodyGen.experiments import her2_support_paths as paths

    policy = TinyPolicy(seed=23)
    lineage = banks.seed_lineage("tamper", spawn_key=(9, 23))
    bank, document = banks.draw_bank(policy, role="development_preservation",
                                     parent_id="parent::seed1", parent_state_sha256="aaa",
                                     seed_lineage_record=lineage, rows=32, batch_size=16)
    target = tmp_path / "bank.npz"
    banks.save_bank(target, bank, document)

    arrays = dict(paths.read_arrays(target))
    scores = np.array(arrays["parent_sum_log_probability"], dtype=np.float64)
    scores[0] -= 5.0
    arrays["parent_sum_log_probability"] = scores
    paths.write_arrays(target, arrays)

    with pytest.raises(ValueError, match="changed probability array"):
        banks.load_bank(target, role="development_preservation", parent_id="parent::seed1",
                        parent_state_sha256="aaa")

    # A bank carrying no score digest is unverifiable, which is refused for the
    # same reason rather than accepted on its core indices alone.
    paths.write_arrays(target, {"core_index": np.asarray(bank.index),
                                "parent_sum_log_probability":
                                    np.asarray(bank.sum_log_probability, dtype=np.float64)})
    paths.write_json(target.with_suffix(".json"),
                     {key: value for key, value in document.items()
                      if key != "sum_log_probability_sha256"})
    with pytest.raises(ValueError, match="records no sum_log_probability_sha256"):
        banks.load_bank(target, role="development_preservation", parent_id="parent::seed1",
                        parent_state_sha256="aaa")


def test_persisted_conditionals_round_trip_at_the_declared_float64(tmp_path):
    """The config budgets conditionals at float64 and forbids a diagnostic
    downcast, and the document digests the float64 in-memory array. Writing
    float32 made that digest describe bytes nobody stored, so it could never
    check anything."""
    from smallAntibodyGen.experiments import her2_support_paths as paths

    policy = TinyPolicy(seed=29)
    lineage = banks.seed_lineage("conditionals", spawn_key=(9, 29))
    bank, document = banks.draw_bank(policy, role="q_generation_screen",
                                     parent_id="model::a", parent_state_sha256="aaa",
                                     seed_lineage_record=lineage, rows=16, batch_size=8,
                                     retain_conditionals=True, conditional_batch=8)
    assert bank.conditionals is not None
    target = tmp_path / "bank.npz"
    banks.save_bank(target, bank, document)

    with np.load(target) as handle:
        assert handle["log_conditionals"].dtype == np.float64

    reloaded = banks.load_bank(target, role="q_generation_screen", parent_id="model::a",
                               parent_state_sha256="aaa")
    assert np.array_equal(reloaded["log_conditionals"], np.asarray(bank.conditionals))
    assert (paths.array_digest(reloaded["log_conditionals"])
            == document["conditionals"]["sha256"]), "the recorded digest must verify the bytes"

    arrays = dict(paths.read_arrays(target))
    altered = np.array(arrays["log_conditionals"], dtype=np.float64)
    altered[0, 0, 0] -= 0.25
    arrays["log_conditionals"] = altered
    paths.write_arrays(target, arrays)
    with pytest.raises(ValueError, match="never evaluated"):
        banks.load_bank(target, role="q_generation_screen", parent_id="model::a",
                        parent_state_sha256="aaa")

    # A described block that is not stored is refused too, rather than read as absent.
    paths.write_arrays(target, {"core_index": np.asarray(bank.index),
                                "parent_sum_log_probability":
                                    np.asarray(bank.sum_log_probability, dtype=np.float64)})
    with pytest.raises(ValueError, match="does not store"):
        banks.load_bank(target, role="q_generation_screen", parent_id="model::a",
                        parent_state_sha256="aaa")


def test_a_drawn_bank_proves_sampler_scorer_parity_and_records_its_identity():
    policy = TinyPolicy(seed=1)
    lineage = banks.seed_lineage("probe", spawn_key=(9, 9))
    bank, document = banks.draw_bank(policy, role="q_generation_screen",
                                     parent_id="parent::tiny",
                                     parent_state_sha256="abc", seed_lineage_record=lineage,
                                     rows=256, batch_size=64)
    assert bank.rows == 256
    assert document["parity"]["max_abs_error"] < 1e-9
    assert document["role"] == "q_generation_screen"
    assert document["role_spec"]["may_influence_selection"] is False
    assert document["duplicates_retained"] is True
    assert document["unique_cores"] <= 256
    assert document["draw_seed"] == lineage["seed"]


def test_conditionals_are_retained_only_when_asked_and_are_hashed():
    policy = TinyPolicy(seed=2)
    lineage = banks.seed_lineage("probe2", spawn_key=(9, 10))
    plain, plain_document = banks.draw_bank(policy, role="q_generation_screen",
                                            parent_id="p", parent_state_sha256="a",
                                            seed_lineage_record=lineage, rows=32, batch_size=16)
    assert plain.conditionals is None and "conditionals" not in plain_document
    kept, document = banks.draw_bank(policy, role="q_generation_finalist", parent_id="p",
                                     parent_state_sha256="a", seed_lineage_record=lineage,
                                     rows=32, batch_size=16, retain_conditionals=True,
                                     conditional_batch=8)
    assert kept.conditionals.shape == (32, 10, 20)
    assert len(document["conditionals"]["sha256"]) == 64
    assert "REALIZED prefix" in document["conditionals"]["content"]


def test_an_unknown_role_is_refused():
    policy = TinyPolicy(seed=3)
    with pytest.raises(ValueError, match="Unknown bank role"):
        banks.draw_bank(policy, role="whatever", parent_id="p", parent_state_sha256="a",
                        seed_lineage_record=banks.seed_lineage("x", spawn_key=(0, 1)), rows=4)


def test_bank_overlap_is_reported_and_not_removed():
    first = tiny_cores(50, seed=4)
    second = np.vstack([first[:5], tiny_cores(45, seed=5)])
    block = banks.bank_overlap(first, second)
    assert block["shared_unique_cores"] >= 5
    assert block["action"] == "none"
    assert "changing the sampling law" in block["note"] or "sampling law" in block["note"]


def test_the_manifest_carries_roles_digests_and_the_selection_rule():
    policy = TinyPolicy(seed=6)
    lineage = banks.seed_lineage("manifest", spawn_key=(9, 11))
    _, document = banks.draw_bank(policy, role="development_preservation", parent_id="p",
                                  parent_state_sha256="a", seed_lineage_record=lineage,
                                  rows=16, batch_size=8)
    manifest = banks.banks_manifest({"dev::p": document}, campaign_id="nf",
                                    source_snapshot_sha256="s" * 64)
    assert manifest["roles_present"] == ["development_preservation"]
    assert set(manifest["role_table"]) == set(banks.BANK_ROLES)
    assert "refused by require_may_influence_selection" in manifest["selection_rule"]
    assert "never across different parent distributions" in manifest["reuse_rule"]


def test_an_external_artifact_is_adopted_as_a_reference_not_as_a_measurement(tmp_path):
    target = tmp_path / "fresh_preservation_u1000.npz"
    np.savez(target, values=np.zeros(10))
    from smallAntibodyGen.experiments import her2_support_paths as paths
    digest = paths.sha256_file(target)
    document = banks.adopt_external_artifact(target, expected_sha256=digest,
                                             label="historical fresh bank")
    assert document["status"] == "external reference"
    assert "no number from this artifact is reported as a measurement" in document["usage"]
    assert "cannot be redrawn from a seed" in document["limitation"]

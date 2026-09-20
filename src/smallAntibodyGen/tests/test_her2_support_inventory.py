"""Two checkpoint schemas, one manifest authority, and coverage that counts verification.

The adapters are exercised against real ``torch.save`` payloads of a tiny module,
because the failures that matter here -- a schema silently downgraded, a state
loaded at the wrong dtype, a diagnostic snapshot whose identity lives one level
down, a manifest digest that no longer matches the bytes -- are all things that
succeed quietly if the reader is lenient.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from smallAntibodyGen.experiments import her2_support_inventory as inventory  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths          # noqa: E402
from smallAntibodyGen.experiments.her2_policy import POLICY_SCHEMA, state_digest  # noqa: E402
from smallAntibodyGen.experiments.her2_guarded_trajectory import TRAJECTORY_SCHEMA  # noqa: E402


def container():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))


@pytest.fixture
def reader():
    return inventory.CheckpointReader(container)


def save_policy(path, module, **document):
    payload = {"schema_version": POLICY_SCHEMA, "state": module.state_dict(),
               "state_sha256": state_digest(module), "kind": "policy", **document}
    torch.save(payload, path)
    return paths.sha256_file(path)


def save_trajectory(path, module, **document):
    payload = {"schema_version": TRAJECTORY_SCHEMA, "state": module.state_dict(),
               "kind": inventory.ROLE_LAST_PASSING, **document}
    torch.save(payload, path)
    return paths.sha256_file(path)


# ---------------------------------------------------------------------------
# both adapters preserve state; neither downgrades a schema
# ---------------------------------------------------------------------------

def test_the_policy_adapter_reproduces_the_recorded_state_digest(tmp_path, reader):
    module = container()
    target = tmp_path / "epoch_3.pt"
    digest = save_policy(target, module, seed=20260918, epoch=3,
                         identity={"seed": 20260918, "arm": {"arm_id": "sft"}})
    payload = reader.read_policy(target, logical="root/epoch_3.pt", expected_file_sha256=digest)
    assert payload.state_digest_recorded == payload.state_digest_audit_computed
    assert payload.state_digest_audit_computed == state_digest(module)
    assert payload.schema_version == POLICY_SCHEMA
    # The payload document must not overwrite the row's campaign-relative path.
    assert "logical_path" not in payload.document()
    assert payload.document()["verified_logical_name"] == "root/epoch_3.pt"


def test_the_diagnostic_adapter_records_no_state_digest_and_invents_none(tmp_path, reader):
    module = container()
    target = tmp_path / "last_passing.pt"
    digest = save_trajectory(target, module, update=51,
                             metadata={"identity": {"seed": 1, "arm": {"arm_id": "dpo"}}})
    payload = reader.read_trajectory(target, logical="root/last_passing.pt",
                                     expected_file_sha256=digest)
    assert payload.state_digest_recorded is None
    assert "records no state digest" in payload.state_digest_recorded_reason
    assert payload.state_digest_audit_computed == state_digest(module)


def test_each_reader_refuses_the_other_schema_instead_of_coercing(tmp_path, reader):
    module = container()
    policy_path, trajectory_path = tmp_path / "p.pt", tmp_path / "t.pt"
    policy_digest = save_policy(policy_path, module)
    trajectory_digest = save_trajectory(trajectory_path, module,
                                        metadata={"identity": {"seed": 1}})
    with pytest.raises(ValueError, match="does not downgrade"):
        reader.read_trajectory(policy_path, logical="root/p.pt",
                               expected_file_sha256=policy_digest)
    with pytest.raises(ValueError, match="does not downgrade"):
        reader.read_policy(trajectory_path, logical="root/t.pt",
                           expected_file_sha256=trajectory_digest)


def test_the_file_is_hashed_before_it_is_opened(tmp_path, reader):
    module = container()
    target = tmp_path / "p.pt"
    save_policy(target, module)
    with pytest.raises(ValueError, match="does not match the manifest digest"):
        reader.read_policy(target, logical="root/p.pt", expected_file_sha256="0" * 64)
    with pytest.raises(ValueError, match="trusted sha256"):
        reader.read_policy(target, logical="root/p.pt", expected_file_sha256=None)


def test_a_missing_state_is_reported_not_substituted(tmp_path, reader):
    with pytest.raises(ValueError, match="missing at the resolved location"):
        reader.read_policy(tmp_path / "absent.pt", logical="root/absent.pt",
                           expected_file_sha256="a" * 64)


def test_a_tampered_state_digest_fails_the_policy_adapter(tmp_path, reader):
    module = container()
    target = tmp_path / "p.pt"
    payload = {"schema_version": POLICY_SCHEMA, "state": module.state_dict(),
               "state_sha256": "b" * 64, "kind": "policy"}
    torch.save(payload, target)
    with pytest.raises(ValueError, match="not reproduced by a strict reload"):
        reader.read_policy(target, logical="root/p.pt",
                           expected_file_sha256=paths.sha256_file(target))


def test_a_reduced_precision_state_is_rejected_rather_than_cast(tmp_path, reader):
    """load_state_dict would cast float16 into float32 and score different weights."""
    module = container()
    state = {name: tensor.half() for name, tensor in module.state_dict().items()}
    target = tmp_path / "half.pt"
    torch.save({"schema_version": POLICY_SCHEMA, "state": state, "state_sha256": "c" * 64,
                "kind": "policy"}, target)
    with pytest.raises(ValueError, match="load_state_dict would cast it silently"):
        reader.read_policy(target, logical="root/half.pt",
                           expected_file_sha256=paths.sha256_file(target))


def test_a_nonfinite_tensor_is_a_failure_to_report(tmp_path, reader):
    module = container()
    state = dict(module.state_dict())
    key = sorted(state)[0]
    broken = state[key].clone()
    broken.view(-1)[0] = float("nan")
    state[key] = broken
    target = tmp_path / "nan.pt"
    torch.save({"schema_version": POLICY_SCHEMA, "state": state, "state_sha256": "d" * 64,
                "kind": "policy"}, target)
    with pytest.raises(ValueError, match="nonfinite"):
        reader.read_policy(target, logical="root/nan.pt",
                           expected_file_sha256=paths.sha256_file(target))


def test_an_unexpected_tensor_set_fails_before_any_digest_is_computed(tmp_path, reader):
    module = container()
    state = dict(module.state_dict())
    state.pop(sorted(state)[0])
    target = tmp_path / "short.pt"
    torch.save({"schema_version": POLICY_SCHEMA, "state": state, "state_sha256": "e" * 64,
                "kind": "policy"}, target)
    with pytest.raises(ValueError, match="missing tensors"):
        reader.read_policy(target, logical="root/short.pt",
                           expected_file_sha256=paths.sha256_file(target))


# ---------------------------------------------------------------------------
# identity lives in a schema-specific place, and the parent hash is per seed
# ---------------------------------------------------------------------------

def payload_for(schema, metadata, kind="policy"):
    return inventory.CheckpointPayload(
        logical="root/x.pt", file_sha256="f" * 64, schema_version=schema, kind=kind,
        payload_keys=("state",), metadata=metadata, tensors={},
        state_digest_recorded=None, state_digest_recorded_reason=None,
        state_digest_audit_computed="0" * 64)


def test_a_diagnostic_identity_is_found_one_level_down_not_declared_absent():
    nested = payload_for(TRAJECTORY_SCHEMA, {"metadata": {"identity": {"seed": 7}}, "update": 3},
                         kind=inventory.ROLE_LAST_PASSING)
    assert inventory.payload_identity(nested) == {"seed": 7}
    flat = payload_for(POLICY_SCHEMA, {"identity": {"seed": 7}})
    assert inventory.payload_identity(flat) == {"seed": 7}


def test_an_absent_identity_is_a_provenance_failure():
    with pytest.raises(ValueError, match="identity"):
        inventory.payload_identity(payload_for(POLICY_SCHEMA, {"seed": 7}))


def test_the_guarded_parent_hash_is_the_seed_entry_not_the_selection_manifest():
    identity = {"inherited": {"base_selection_sha256": "1" * 64,
                              "parents": {"20260918": {"sha256": "2" * 64},
                                          "20260919": {"sha256": "3" * 64}}}}
    assert inventory.guarded_parent_sha256(identity, 20260918) == "2" * 64
    assert inventory.guarded_parent_sha256(identity, 20260919) == "3" * 64
    assert inventory.guarded_parent_sha256({"inherited": {"base_selection_sha256": "1" * 64}},
                                           20260918) is None


#: One payload per role, shaped like the real campaign states, with the contract the
#: enumerator builds for it. Every field below is present in all 159 native payloads
#: of that role, which is why it is required rather than compared-if-present.
NATIVE_PAYLOADS = {
    inventory.ROLE_PARENT: (
        lambda: payload_for(POLICY_SCHEMA,
                            {"identity": {"seed": 20260918, "arm": "sft"}, "epoch": 3}),
        {"arm_id": "sft", "epoch": 3, "seed": 20260918}),
    inventory.ROLE_ENDPOINT: (
        lambda: payload_for(POLICY_SCHEMA, {
            "identity": {"inherited": {"parents": {"20260918": {"sha256": "a" * 64}}},
                         "stage": 1},
            "seed": 20260918, "arm": {"arm_id": "continued_sft", "objective": "continued_sft",
                                      "stage": 1},
            "budget_seconds": 180.0, "progress": {"updates": 1411}}),
        {"arm_id": "continued_sft", "budget_seconds": 180.0, "parent_sha256": "a" * 64,
         "seed": 20260918, "stage": 1, "update": 1411}),
    inventory.ROLE_LAST_PASSING: (
        lambda: payload_for(TRAJECTORY_SCHEMA, {
            "metadata": {"identity": {"inherited": {"parents": {"20260918": {"sha256": "a" * 64}}},
                                      "arm": {"arm_id": "dpo_beta0p1", "stage": 1}},
                         "seed": 20260918, "stage": 1},
            "update": 26}, kind=inventory.ROLE_LAST_PASSING),
        {"arm_id": "dpo_beta0p1", "parent_sha256": "a" * 64, "seed": 20260918, "stage": 1,
         "update": 26}),
    inventory.ROLE_FAILED_STATE: (
        lambda: payload_for(TRAJECTORY_SCHEMA, {
            "metadata": {"identity": {"inherited": {"parents": {"20260918": {"sha256": "a" * 64}}},
                                      "arm": {"arm_id": "dpo_beta0p1", "stage": 1}},
                         "seed": 20260918, "stage": 1},
            "update": 51}, kind=inventory.ROLE_FAILED_STATE),
        {"arm_id": "dpo_beta0p1", "parent_sha256": "a" * 64, "seed": 20260918, "stage": 1,
         "update": 51}),
    inventory.ROLE_V1_ENDPOINT: (
        lambda: payload_for(POLICY_SCHEMA, {
            "identity": {"parent_sha256": "a" * 64}, "seed": 20260918, "method": "continued_sft",
            "budget_seconds": 180.0, "progress": {"updates": 1335}}),
        {"budget_seconds": 180.0, "method": "continued_sft", "parent_sha256": "a" * 64,
         "seed": 20260918, "update": 1335}),
}


@pytest.mark.parametrize("role", sorted(NATIVE_PAYLOADS))
def test_every_role_declares_the_bindings_its_real_payloads_all_carry(role):
    """The enumerator's contract has to cover the required set, and it has to pass."""
    build, contract = NATIVE_PAYLOADS[role]
    assert set(inventory.REQUIRED_BINDINGS[role]) <= set(contract)
    checked = inventory.validate_payload_contract(build(), contract, role=role,
                                                  logical="root/x.pt")
    for field in inventory.REQUIRED_BINDINGS[role]:
        assert checked["checked"][field]["matched"] is True
    assert checked["required_bindings"] == list(inventory.REQUIRED_BINDINGS[role])


@pytest.mark.parametrize("role", sorted(NATIVE_PAYLOADS))
def test_a_required_binding_moved_into_the_optional_tier_is_refused(role):
    """The exact shape the previous revision shipped: contract {} plus compare-if-present."""
    build, contract = NATIVE_PAYLOADS[role]
    with pytest.raises(ValueError, match="must bind"):
        inventory.validate_payload_contract(build(), {}, role=role, logical="root/x.pt",
                                            compare_if_present=contract)


@pytest.mark.parametrize("role", sorted(NATIVE_PAYLOADS))
def test_a_contract_that_drops_one_required_field_is_refused(role):
    build, contract = NATIVE_PAYLOADS[role]
    for field in inventory.REQUIRED_BINDINGS[role]:
        reduced = {key: value for key, value in contract.items() if key != field}
        with pytest.raises(ValueError, match=f"must bind.*{field}"):
            inventory.validate_payload_contract(build(), reduced, role=role, logical="root/x.pt")


@pytest.mark.parametrize("role", sorted(NATIVE_PAYLOADS))
def test_a_payload_missing_a_required_field_is_a_provenance_gap(role):
    """Absence in the payload, not in the manifest: the checkpoint cannot confirm itself."""
    build, contract = NATIVE_PAYLOADS[role]
    payload = build()
    stripped = inventory.CheckpointPayload(
        logical=payload.logical, file_sha256=payload.file_sha256,
        schema_version=payload.schema_version, kind=payload.kind,
        payload_keys=payload.payload_keys,
        metadata={"identity": {"note": "nothing this contract names"}}
        if payload.schema_version == POLICY_SCHEMA else
        {"metadata": {"identity": {"note": "nothing this contract names"}}},
        tensors={}, state_digest_recorded=None, state_digest_recorded_reason=None,
        state_digest_audit_computed="0" * 64)
    with pytest.raises(ValueError, match="provenance gap|confirmed none|never relabelled"):
        inventory.validate_payload_contract(stripped, contract, role=role, logical="root/x.pt")


def test_the_repro_that_accepted_an_absent_seed_and_epoch_now_fails():
    """CX-22's exact reproduction: one matching field must not carry the whole identity."""
    payload = payload_for(POLICY_SCHEMA, {"identity": {"arm": "sft"}})
    with pytest.raises(ValueError, match="must bind"):
        inventory.validate_payload_contract(
            payload, {}, role=inventory.ROLE_PARENT, logical="probe",
            compare_if_present={"epoch": 3, "seed": 20260918, "arm_id": "sft"})


def test_the_contract_fails_a_payload_bound_to_a_different_parent():
    build, contract = NATIVE_PAYLOADS[inventory.ROLE_LAST_PASSING]
    with pytest.raises(ValueError, match="wrong checkpoint"):
        inventory.validate_payload_contract(
            build(), dict(contract, parent_sha256="b" * 64),
            role=inventory.ROLE_LAST_PASSING, logical="root/x.pt")


def test_a_field_the_writer_never_recorded_is_named_not_invented():
    build, contract = NATIVE_PAYLOADS[inventory.ROLE_PARENT]
    checked = inventory.validate_payload_contract(
        build(), contract, role=inventory.ROLE_PARENT, logical="root/x.pt",
        compare_if_present={"budget_seconds": 600.0})
    assert checked["checked"]["seed"]["matched"] is True
    assert "budget_seconds" in checked["not_recorded_by_payload"]


def test_a_payload_confirming_nothing_at_all_is_a_provenance_failure():
    payload = payload_for(POLICY_SCHEMA, {"identity": {"note": "no fields"}})
    with pytest.raises(ValueError, match="carries no|confirmed none of the declared bindings"):
        inventory.validate_payload_contract(
            payload, {"arm_id": "sft", "epoch": 3, "seed": 5}, role=inventory.ROLE_PARENT,
            logical="root/x.pt")


def test_a_failure_snapshot_cannot_be_filed_as_a_last_passing_state():
    build, contract = NATIVE_PAYLOADS[inventory.ROLE_FAILED_STATE]
    with pytest.raises(ValueError, match="never relabelled"):
        inventory.validate_payload_contract(
            build(), dict(contract, update=51), role=inventory.ROLE_LAST_PASSING,
            logical="root/x.pt")


# ---------------------------------------------------------------------------
# enumeration: paths, reuse aliases and recovered counts
# ---------------------------------------------------------------------------

#: Shaped like the real ``base_selection.json``: the table is keyed by RUN name
#: (``sft_seed1``) while the selected parent is named ``policy_sft_seed1``, and each
#: run's entry is a mapping of epoch to row. A lookup on the selected name finds
#: nothing at all, which is how three parents came to publish null counts.
BASE_SELECTION = {
    "selected": {
        "policy_sft_seed1": {"checkpoint": "outputs/camp/policy_sft_seed1/epoch_3.pt",
                             "epoch": 3, "kind": "policy", "sha256": "a" * 64,
                             "val_positive_nll_per_residue": 1.4919637766143667}},
    "checkpoint_table": {
        "sft_seed1": {
            "1": {"epoch": 1, "steps": 942, "exposures": 120504,
                  "checkpoint": "outputs/camp/policy_sft_seed1/epoch_1.pt",
                  "checkpoint_sha256": "1" * 64, "state_sha256": "2" * 64},
            "3": {"epoch": 3, "steps": 2826, "exposures": 361512,
                  "checkpoint": "outputs/camp/policy_sft_seed1/epoch_3.pt",
                  "checkpoint_sha256": "a" * 64, "state_sha256": "e" * 64},
            "5": {"epoch": 5, "steps": 4710, "exposures": 602520,
                  "checkpoint": "outputs/camp/policy_sft_seed1/epoch_5.pt",
                  "checkpoint_sha256": "5" * 64, "state_sha256": "6" * 64}},
        "scratch_seed1": {
            "3": {"epoch": 3, "steps": 3, "exposures": 4,
                  "checkpoint": "outputs/camp/policy_scratch_seed1/epoch_3.pt",
                  "checkpoint_sha256": "7" * 64}}}}


def test_a_parent_row_keeps_one_path_convention_so_the_root_is_not_doubled():
    records = inventory.parent_records(BASE_SELECTION, seeds=[1], logical_root="outputs/camp",
                                       anchor="camp")
    record = records[0]
    assert record["logical_path"] == "policy_sft_seed1/epoch_3.pt"
    assert record["root"] == "outputs/camp"
    # resolving the row under its root must not produce outputs/camp/outputs/camp/...
    resolved = paths.resolve_under("/tmp/local/camp", record["logical_path"])
    assert "camp" not in str(resolved.parent.name)


def test_parent_counts_are_found_under_the_run_key_not_the_selected_name():
    """The join is on the checkpoint's own digest, so the key naming cannot break it."""
    record = inventory.parent_records(BASE_SELECTION, seeds=[1], logical_root="outputs/camp",
                                      anchor="camp")[0]
    assert record["updates"] == 2826 and record["exposures"] == 361512
    assert record["updates_reason"] is None and record["exposures_reason"] is None
    assert record["progress_binding"]["table_key"] == "sft_seed1"
    assert record["progress_binding"]["matched_by"] == "checkpoint_sha256"
    assert record["progress_binding"]["epoch"] == 3
    # the selection table's own state digest becomes a second witness for the payload
    assert record["state_sha256_recorded_by_manifest"] == "e" * 64


def test_a_neighbouring_epoch_row_is_never_substituted():
    selection = {"selected": BASE_SELECTION["selected"],
                 "checkpoint_table": {"sft_seed1": {
                     "1": BASE_SELECTION["checkpoint_table"]["sft_seed1"]["1"],
                     "5": BASE_SELECTION["checkpoint_table"]["sft_seed1"]["5"]}}}
    record = inventory.parent_records(selection, seeds=[1], logical_root="outputs/camp",
                                      anchor="camp")[0]
    assert record["updates"] is None and record["exposures"] is None
    assert "neighbouring epoch" in record["updates_reason"]
    assert "['1', '5']" in record["updates_reason"]


def test_an_epoch_row_describing_other_bytes_is_not_this_parents_row():
    """Same epoch, different checkpoint digest: a different set of weights."""
    table = {"sft_seed1": {"3": dict(BASE_SELECTION["checkpoint_table"]["sft_seed1"]["3"],
                                     checkpoint_sha256="9" * 64)}}
    record = inventory.parent_records({"selected": BASE_SELECTION["selected"],
                                       "checkpoint_table": table},
                                      seeds=[1], logical_root="outputs/camp", anchor="camp")[0]
    assert record["updates"] is None
    assert record["progress_binding"] is None


def test_a_row_without_a_digest_is_matched_on_the_recorded_checkpoint_path():
    row = {"epoch": 3, "steps": 2826, "exposures": 361512,
           "checkpoint": "outputs/camp/policy_sft_seed1/epoch_3.pt"}
    updates, exposures, reason, binding = inventory.parent_progress(
        {"checkpoint_table": {"sft_seed1": {"3": row}}}, "policy_sft_seed1", 3,
        checkpoint="outputs/camp/policy_sft_seed1/epoch_3.pt", checkpoint_sha256="a" * 64)
    assert (updates, exposures, reason) == (2826, 361512, None)
    assert binding["matched_by"] == "checkpoint"


def test_a_nested_block_inside_a_flat_row_is_not_mistaken_for_an_epoch_row():
    """``{"exposures": {...}}`` inside a row is data, not a table keyed by epoch."""
    table = {"sft_seed1": {"epoch": 3, "steps": 2826, "exposures": {"sequences": 361512},
                           "checkpoint_sha256": "a" * 64}}
    rows = inventory.checkpoint_table_rows(table)
    assert rows == [("sft_seed1", None, table["sft_seed1"])]


def endpoint(name, seed=1, stage=1, budget=600.0, **extra):
    return dict({"name": name, "trajectory": name, "checkpoint":
                 f"C:\\runs\\camp\\stage{stage}\\{name}\\budget{int(budget)}.pt",
                 "checkpoint_sha256": "b" * 64, "arm_id": "dpo_beta0p1", "objective": "dpo",
                 "coefficients": {"beta": 0.1}, "seed": seed, "nominal_budget": budget,
                 "checkpoint_binding": {"state_sha256": "c" * 64, "updates": 40},
                 "parent_kl": {"mean": 0.5}, "generation": {"unique_fraction": 0.97}}, **extra)


def native_reused_control(stage=1, seed=1, budget=600.0, arm_id="continued_sft", **extra):
    """A reused-control entry shaped like the real campaign's: no name, no digest."""
    return dict({"reused_from_stage": stage, "arm_id": arm_id, "seed": seed,
                 "nominal_budget": budget, "objective": "continued_sft", "coefficients": {},
                 "reached": True, "gate_passed": True, "diversity_eligible": True,
                 "gate_evidence": {"D": 0.087, "update": 1411},
                 "val_auroc": 0.9799, "chosen_nll_per_residue": 1.5009}, **extra)


def test_reused_controls_are_resolved_by_their_endpoint_tuple_not_by_a_name():
    """The real entries carry no name and no checkpoint_sha256; only the tuple."""
    records = inventory.guarded_endpoint_records(
        {1: {"endpoints": [endpoint("continued_sft_seed1", arm_id="continued_sft")],
             "reused_controls": []}},
        anchor="camp", logical_root="outputs/camp")
    reuse = inventory.reused_control_aliases(
        {1: {"reused_controls": []},
         2: {"reused_controls": [native_reused_control()]},
         3: {"reused_controls": [native_reused_control()]}},
        records)
    assert reuse["attached"] == 2 and reuse["unmatched"] == []
    assert [alias["stage"] for alias in records[0]["aliases"]] == [2, 3]
    assert records[0]["aliases"][0]["reused_from_stage"] == 1
    assert records[0]["aliases"][0]["source_manifest"] == "validation/stage2_endpoints.json"
    assert reuse["resolved_by"] == list(inventory.REUSED_CONTROL_REFERENCE)


def test_every_declared_budget_of_a_reused_control_resolves_separately():
    """Three seeds x three budgets x two later stages is eighteen declarations."""
    endpoints = [endpoint(f"continued_sft_seed{seed}_budget{int(budget)}", seed=seed,
                          budget=budget, arm_id="continued_sft",
                          trajectory=f"continued_sft_seed{seed}")
                 for seed in (1, 2, 3) for budget in (180.0, 360.0, 600.0)]
    records = inventory.guarded_endpoint_records({1: {"endpoints": endpoints}},
                                                 anchor="camp", logical_root="outputs/camp")
    declarations = [native_reused_control(seed=seed, budget=budget)
                    for seed in (1, 2, 3) for budget in (180.0, 360.0, 600.0)]
    reuse = inventory.reused_control_aliases(
        {2: {"reused_controls": declarations}, 3: {"reused_controls": declarations}}, records)
    assert reuse["attached"] == 18 and reuse["unmatched"] == []


def test_a_reused_control_naming_no_reached_endpoint_is_unmatched_not_attached():
    records = inventory.guarded_endpoint_records(
        {1: {"endpoints": [endpoint("continued_sft_seed1", arm_id="continued_sft")]}},
        anchor="camp", logical_root="outputs/camp")
    reuse = inventory.reused_control_aliases(
        {2: {"reused_controls": [native_reused_control(budget=1800.0)]}}, records)
    assert reuse["attached"] == 0
    assert reuse["unmatched"][0]["reference"]["nominal_budget"] == 1800.0
    assert "no reached endpoint" in reuse["unmatched"][0]["reason"]


def test_a_reused_control_missing_a_reference_field_says_which_one():
    records = inventory.guarded_endpoint_records(
        {1: {"endpoints": [endpoint("continued_sft_seed1", arm_id="continued_sft")]}},
        anchor="camp", logical_root="outputs/camp")
    entry = native_reused_control()
    entry.pop("nominal_budget")
    reuse = inventory.reused_control_aliases({2: {"reused_controls": [entry]}}, records)
    assert reuse["attached"] == 0
    assert "nominal_budget" in reuse["unmatched"][0]["reason"]


def test_a_reused_control_pointing_at_different_bytes_is_not_an_alias():
    records = inventory.guarded_endpoint_records(
        {1: {"endpoints": [endpoint("continued_sft_seed1", arm_id="continued_sft")]}},
        anchor="camp", logical_root="outputs/camp")
    with pytest.raises(ValueError, match="different checkpoint digest"):
        inventory.reused_control_aliases(
            {2: {"reused_controls": [native_reused_control(checkpoint_sha256="9" * 64)]}}, records)


def test_a_reused_controls_mapping_is_also_accepted():
    records = inventory.guarded_endpoint_records(
        {1: {"endpoints": [endpoint("continued_sft_seed1", arm_id="continued_sft")]}},
        anchor="camp", logical_root="outputs/camp")
    reuse = inventory.reused_control_aliases(
        {2: {"reused_controls": {"stage1::continued_sft_seed1": native_reused_control()}}},
        records)
    assert reuse["attached"] == 1


def test_an_unreadable_reused_controls_shape_is_refused():
    with pytest.raises(ValueError, match="reused_controls is a"):
        inventory._reused_control_entries("continued_sft_seed1")


def test_endpoint_rows_carry_their_historical_metrics_and_update_count():
    records = inventory.guarded_endpoint_records(
        {1: {"endpoints": [endpoint("dpo_seed1", **{"checkpoint_binding": {}, "updates": 61})]}},
        anchor="camp", logical_root="outputs/camp")
    record = records[0]
    assert record["updates"] == 61, "the top-level count is used when the binding omits it"
    assert record["historical_metrics_source"]["generation"]["unique_fraction"] == 0.97
    assert record["logical_path"] == "stage1/dpo_seed1/budget600.pt"


V1_SUMMARIES = {"dpo_seed1": {"method": "dpo", "seed": 1, "budgets": {
    "180.0": {"checkpoint": "outputs/camp/continuation/dpo_seed1/budget180.pt",
              "checkpoint_sha256": "b" * 64, "updates": 12},
    "600.0": {"checkpoint": "outputs/camp/continuation/dpo_seed1/budget600.pt",
              "checkpoint_sha256": "c" * 64, "updates": 40}}}}

#: One global document, keyed by run AND budget, exactly like the real campaign's
#: ``continuation/validation_records.json``. It also holds parent and zero-shot rows.
V1_VALIDATION = {
    "policy_sft_seed1": {"name": "policy_sft_seed1", "role": "parent"},
    "piggen_zeroshot": {"name": "piggen_zeroshot", "role": "baseline"},
    "dpo_seed1_budget180": {
        "name": "dpo_seed1_budget180", "method": "dpo", "seed": 1, "budget_seconds": 180.0,
        "checkpoint": "outputs/camp/continuation/dpo_seed1/budget180.pt",
        "sha256": "b" * 64,
        "val_metrics": {"auroc": 0.9798}, "val_pair_metrics": {"pair_accuracy": 0.9715},
        "val_positive_nll_per_residue": 1.4989,
        "generation": {"unique_fraction": 0.9765},
        "diversity": {"eligible": True, "checks": {"unique_fraction": {"observed": 0.9765}}}}}


def test_v1_rows_join_the_one_global_validation_document_by_run_and_budget():
    records = inventory.v1_endpoint_records(V1_SUMMARIES, logical_root="outputs/camp",
                                            parents_by_seed={1: "a" * 64}, anchor="camp",
                                            validation_records=V1_VALIDATION)
    by_budget = {record["nominal_budget_gpu_seconds"]: record for record in records}
    joined = by_budget[180.0]["historical_metrics_source"]
    assert by_budget[180.0]["logical_path"] == "continuation/dpo_seed1/budget180.pt"
    assert joined["key"] == "dpo_seed1_budget180"
    assert joined["val_metrics"]["auroc"] == 0.9798
    assert joined["val_pair_metrics"]["pair_accuracy"] == 0.9715
    assert joined["generation"]["unique_fraction"] == 0.9765
    assert joined["diversity_eligible"] is True
    assert by_budget[180.0]["diversity_eligible"] is True
    # the row confirms itself against the endpoint's own checkpoint identity
    assert "sha256" in joined["confirmed_by"] and "checkpoint" in joined["confirmed_by"]
    # fields this row genuinely does not carry are named, not left blank
    assert "parent_kl" in joined["not_recorded_by_row"]
    assert joined["parent_kl"] is None


def test_a_budget_with_no_row_is_a_named_absence_not_a_neighbouring_budget():
    records = inventory.v1_endpoint_records(V1_SUMMARIES, logical_root="outputs/camp",
                                            parents_by_seed={1: "a" * 64}, anchor="camp",
                                            validation_records=V1_VALIDATION)
    missing = [record for record in records
               if record["nominal_budget_gpu_seconds"] == 600.0][0]
    reason = missing["historical_metrics_source"]["reason"]
    assert "dpo_seed1_budget600" in reason and "not substituted" in reason


def test_a_validation_row_that_describes_other_weights_is_a_conflict_not_a_column():
    document = {"dpo_seed1_budget180": dict(V1_VALIDATION["dpo_seed1_budget180"],
                                            sha256="9" * 64)}
    with pytest.raises(ValueError, match="disagrees with the endpoint"):
        inventory.v1_endpoint_records(V1_SUMMARIES, logical_root="outputs/camp",
                                      parents_by_seed={1: "a" * 64}, anchor="camp",
                                      validation_records=document)


def test_a_per_run_lookup_finds_nothing_which_is_how_the_columns_emptied():
    """The shape the previous revision expected: a per-run file with a records list."""
    per_run = {"dpo_seed1": {"records": [{"budget_seconds": 180.0, "val_metrics": {"auroc": 1.0}}]}}
    records = inventory.v1_endpoint_records(V1_SUMMARIES, logical_root="outputs/camp",
                                            parents_by_seed={1: "a" * 64}, anchor="camp",
                                            validation_records=per_run)
    assert all(record["historical_metrics_source"].get("val_metrics") is None
               for record in records)
    assert all("holds no" in record["historical_metrics_source"]["reason"] for record in records)


# ---------------------------------------------------------------------------
# deduplication and coverage
# ---------------------------------------------------------------------------

class Bank:
    file_sha256 = "bank"
    order_sha256 = "order"
    parent_id = "parent::p"


def test_two_identical_states_are_one_computation_and_every_alias_is_kept():
    bank = Bank()
    rows = [{"id": "b", "state_digest_audit_computed": "s", "parent_id": "parent::p"},
            {"id": "a", "state_digest_audit_computed": "s", "parent_id": "parent::p"},
            {"id": "c", "state_digest_audit_computed": "t", "parent_id": "parent::p"}]
    for row in rows:
        row["dedup_key"] = inventory.dedup_key(row, bank=bank)
    assignment, summary = inventory.deduplicate(rows)
    assert summary["distinct_computations"] == 2
    assert assignment["a"]["is_primary"] and assignment["b"]["alias_of"] == "a"
    assert assignment["c"]["is_primary"]


def test_a_different_bank_or_row_order_is_a_different_computation():
    class Other(Bank):
        order_sha256 = "different"
    row = {"id": "a", "state_digest_audit_computed": "s", "parent_id": "parent::p"}
    assert inventory.dedup_key(row, bank=Bank()) != inventory.dedup_key(row, bank=Other())


def test_an_unverified_row_is_a_coverage_gap_even_though_the_row_exists():
    records = [{"id": "parent::p", "role": inventory.ROLE_PARENT,
                "status": inventory.STATUS_FAILED}]
    coverage = inventory.coverage_table(
        records, expected={"total": 1, "by_role": {inventory.ROLE_PARENT: 1}})
    assert coverage["complete"] is False
    kinds = {entry["kind"] for entry in coverage["shortfalls"]}
    assert kinds == {"verification"}
    assert coverage["verified_total"] == 0 and coverage["total"] == 1


def test_a_fully_verified_population_is_complete():
    records = [{"id": "parent::p", "role": inventory.ROLE_PARENT,
                "status": inventory.STATUS_VERIFIED}]
    coverage = inventory.coverage_table(
        records, expected={"total": 1, "by_role": {inventory.ROLE_PARENT: 1}})
    assert coverage["complete"] and coverage["shortfalls"] == []


def test_a_requirement_that_was_never_evaluated_cannot_be_satisfied():
    completion = inventory.audit_completion({"inventory_coverage": {"satisfied": True}})
    assert completion["complete"] is False
    assert "ches_populations" in completion["unmet"]
    assert completion["requirements"]["scored_computations"]["detail"].startswith(
        "this requirement was never evaluated")


def test_every_satisfied_requirement_completes_the_audit():
    completion = inventory.audit_completion(
        {name: {"satisfied": True} for name in inventory.COMPLETION_REQUIREMENTS})
    assert completion["complete"] and completion["unmet"] == []


# ---------------------------------------------------------------------------
# banks, journals and source provenance
# ---------------------------------------------------------------------------

def write_bank(path, cores, index_column=None):
    rows = ["draw_index,core"]
    for position, core in enumerate(cores):
        rows.append(f"{position if index_column is None else index_column[position]},{core}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return paths.sha256_file(path)


def reference_for(path, digest, **overrides):
    return dict({"path": f"validation/{path.name}", "sha256": digest, "draws": 2,
                 "temperature": 1.0, "draw_seed": 5, "name": "policy_sft_seed1",
                 "parent_sha256": "a" * 64, "seed": 1}, **overrides)


def test_a_bank_is_verified_against_every_claim_its_document_makes(tmp_path):
    target = tmp_path / "bank.csv"
    digest = write_bank(target, ["ACDEFGHIKL", "ACDEFGHIKL"])
    bank = inventory.verify_parent_bank(
        reference_for(target, digest), path=target, expected_rows=2, expected_temperature=1.0,
        parent_sha256="a" * 64, parent_id="parent::policy_sft_seed1")
    assert bank.rows == 2 and bank.document()["unique_cores"] == 1
    assert bank.document()["duplicates_retained"] is True


def test_a_reordered_bank_is_a_different_measurement(tmp_path):
    target = tmp_path / "bank.csv"
    digest = write_bank(target, ["ACDEFGHIKL", "ACDEFGHIKM"], index_column=[1, 0])
    with pytest.raises(ValueError, match="exact 0..1 order"):
        inventory.verify_parent_bank(
            reference_for(target, digest), path=target, expected_rows=2,
            expected_temperature=1.0, parent_sha256="a" * 64, parent_id="parent::p")


def test_a_bank_bound_to_another_parent_or_temperature_is_refused(tmp_path):
    target = tmp_path / "bank.csv"
    digest = write_bank(target, ["ACDEFGHIKL", "ACDEFGHIKM"])
    with pytest.raises(ValueError, match="is bound to parent"):
        inventory.verify_parent_bank(
            reference_for(target, digest), path=target, expected_rows=2,
            expected_temperature=1.0, parent_sha256="z" * 64, parent_id="parent::p")
    with pytest.raises(ValueError, match="different temperature"):
        inventory.verify_parent_bank(
            reference_for(target, digest, temperature=0.8), path=target, expected_rows=2,
            expected_temperature=1.0, parent_sha256="a" * 64, parent_id="parent::p")


def test_a_bank_whose_bytes_changed_is_refused(tmp_path):
    target = tmp_path / "bank.csv"
    digest = write_bank(target, ["ACDEFGHIKL", "ACDEFGHIKM"])
    write_bank(target, ["ACDEFGHIKL", "ACDEFGHIKW"])
    with pytest.raises(ValueError, match="hashes"):
        inventory.verify_parent_bank(
            reference_for(target, digest), path=target, expected_rows=2,
            expected_temperature=1.0, parent_sha256="a" * 64, parent_id="parent::p")


def test_stages_that_disagree_about_a_bank_fail_closed():
    stages = {1: {"parent_draw_references": {"1": {"path": "a.csv", "sha256": "x",
                                                   "draw_seed": 5, "temperature": 1.0,
                                                   "parent_sha256": "a"}}},
              2: {"parent_draw_references": {"1": {"path": "a.csv", "sha256": "y",
                                                   "draw_seed": 5, "temperature": 1.0,
                                                   "parent_sha256": "a"}}}}
    with pytest.raises(ValueError, match="disagree about the parent draw banks"):
        inventory.cross_stage_bank_consistency(stages)


def test_exposures_are_recovered_only_at_the_exact_update(tmp_path):
    journal = tmp_path / "updates.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in [
        {"update": 50, "exposures": {"pairs": 800}},
        {"update": 51, "exposures": {"pairs": 816}}]) + "\n", encoding="utf-8")
    assert inventory.exposures_at_update(journal, 51) == ({"pairs": 816}, None)
    values, reason = inventory.exposures_at_update(journal, 52)
    assert values is None and "not substituted" in reason
    values, reason = inventory.exposures_at_update(tmp_path / "absent.jsonl", 51)
    assert values is None and "absent" in reason
    values, reason = inventory.exposures_at_update(journal, None)
    assert values is None and "no update number" in reason


def test_an_archive_that_does_not_match_the_launch_manifest_fails_closed(tmp_path):
    snapshot = tmp_path / "source_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "scripts/train.py").write_bytes(b"archived")
    manifest = {"source_files": {"scripts/train.py": {"sha256": paths.sha256_bytes(b"other")}}}
    with pytest.raises(ValueError, match="do not match the launch manifest"):
        inventory.verify_source_snapshot(manifest, snapshot, repository_root=tmp_path)


def test_working_tree_drift_is_reported_but_does_not_fail(tmp_path):
    snapshot = tmp_path / "source_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "scripts/train.py").write_bytes(b"archived")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts/train.py").write_bytes(b"edited since")
    manifest = {"source_files": {"scripts/train.py": {"sha256": paths.sha256_bytes(b"archived")}}}
    report = inventory.verify_source_snapshot(manifest, snapshot, repository_root=tmp_path)
    assert report["archived_verified"] == 1
    assert report["current_drift"] == ["scripts/train.py"]
    assert report["archive_mismatched"] == []


def test_the_audits_own_source_identity_is_separate_and_must_exist(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_bytes(b"code")
    digests = inventory.audit_source_identity(tmp_path, ["src/module.py"])
    assert digests == {"src/module.py": paths.sha256_bytes(b"code")}
    with pytest.raises(ValueError, match="missing"):
        inventory.audit_source_identity(tmp_path, ["src/absent.py"])

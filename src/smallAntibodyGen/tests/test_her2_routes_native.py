"""I1, I8, I9, I18, I20: the native-weight checks. OPT-IN, run by Codex.

Skipped unless ``HER2_ROUTES_ARTIFACT_ROOT`` names a directory holding the pinned
releases, so the ordinary suite stays offline and fixture-free. Codex runs these
against the real bytes:

    $env:HER2_ROUTES_ARTIFACT_ROOT = "F:/smallAntibodyGen/her2_generative_routes_20260927"
    ./.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_routes_native.py -q

These are the checks that cannot be faked with a synthetic model: strict load, the
parameter count, the tied-head identity, the bit-exact zero-initialised parity, batch
isolation, cache-free revision agreement, and resume equivalence.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ARTIFACT_ROOT = os.environ.get("HER2_ROUTES_ARTIFACT_ROOT")
pytestmark = pytest.mark.skipif(
    not ARTIFACT_ROOT or not Path(ARTIFACT_ROOT).is_dir(),
    reason="set HER2_ROUTES_ARTIFACT_ROOT to the pinned artifact root to run the native checks")


@pytest.fixture(scope="module")
def prepared():
    from smallAntibodyGen.experiments import her2_routes_env as env
    record = env.prepare_environment(ARTIFACT_ROOT)
    env.require_environment_ordering(record, label="native tests")
    return record


@pytest.fixture(scope="module")
def condition_spec():
    from smallAntibodyGen.experiments.her2_routes_conditions import ConditionSpec
    return ConditionSpec(assays=("alphabind_her2_validation", "engelhart2022_s1"))


@pytest.fixture(scope="module")
def native(prepared):
    from smallAntibodyGen.experiments.her2_routes_env import (
        NATIVE_PACKAGE_ROOT,
        NATIVE_WEIGHTS_ROOT,
    )
    from smallAntibodyGen.models.ablang2_policy import load_pinned_ablang2
    root = Path(ARTIFACT_ROOT)
    return load_pinned_ablang2(root / NATIVE_WEIGHTS_ROOT,
                               package_root=root / NATIVE_PACKAGE_ROOT, device="cpu")


# --- I20: the native backbone contract ------------------------------------

def test_i20_strict_load_and_the_measured_parameter_count(native):
    from smallAntibodyGen.models.ablang2_policy import ABLANG2_PARAMETERS
    assert native.load_record["strict"] is True
    assert native.load_record["weights_only"] is True
    assert native.load_record["parameters"] == ABLANG2_PARAMETERS
    assert native.hyperparameters["hidden_embed_size"] == 480
    assert native.hyperparameters["n_encoder_blocks"] == 12
    assert native.hyperparameters["n_attn_heads"] == 20
    assert native.hyperparameters["a_fn"] == "swiglu"


def test_i20_hparams_rename_is_mapped_and_the_dropped_sentinels_are_verified(native):
    mapping = native.load_record["mapping"]
    assert mapping["renamed"] == {"pad_tkn": "padding_tkn"}
    assert set(mapping["verified_and_dropped"]) == {"start_tkn", "end_tkn", "sep_tkn"}
    assert mapping["verified_and_dropped"]["sep_tkn"] == 25
    assert "``**hparams`` raises" in mapping["rule"]


def test_i20_the_output_projection_is_tied_by_object_identity(native):
    import torch
    model = native.model
    assert model.AbHead.weights is model.AbRep.aa_embed_layer.weight
    assert torch.equal(model.AbHead.weights, model.AbRep.aa_embed_layer.weight)
    # One tensor under two state_dict keys; named_parameters de-duplicates it.
    keys = native.load_record["tied_state_dict_keys"]
    assert len(keys) >= 2
    assert "de-duplicates" in native.load_record["tie_note"]


def test_i20_the_double_attention_scaling_is_retained(native):
    import math
    from smallAntibodyGen.models.ablang2_policy import attention_scaling_record
    record = attention_scaling_record(native)
    assert record["head_dim"] == 24
    assert record["query_scale"] == pytest.approx(24 ** -0.5)
    assert record["score_divisor"] == pytest.approx(math.sqrt(24))
    # The effective scale is 1/24, not 1/sqrt(24). Retained, because the released
    # weights were trained under it.
    assert record["effective_scale"] == pytest.approx(1.0 / 24.0)
    assert record["single_scaling_would_be"] != pytest.approx(record["effective_scale"])


def test_i20_external_source_digests_cover_every_loaded_module(native):
    digests = native.load_record["native_source_sha256"]
    for name in ("ablang2/models/ablang2/ablang.py",
                 "ablang2/models/ablang2/encoderblock.py",
                 "ablang2/models/ablang2/tokenizers.py",
                 "ablang2/models/ablang2/vocab.py"):
        assert name in digests and len(digests[name]) == 64


def test_the_pinned_package_init_is_hashed_and_not_executed(prepared):
    import sys
    from smallAntibodyGen.experiments.her2_routes_env import NATIVE_PACKAGE_ROOT
    from smallAntibodyGen.models.ablang2_policy import load_native_modules
    block = load_native_modules(Path(ARTIFACT_ROOT) / NATIVE_PACKAGE_ROOT)
    assert "hashed as part of the frozen identity" in block["note"]
    # The synthetic parents have a __path__ and carry no executed module body.
    assert getattr(sys.modules["ablang2"], "__path__", None)
    assert not hasattr(sys.modules["ablang2"], "pretrained")


# --- I1: zero-initialised equivalence -------------------------------------

def test_i1_masked_editor_at_zero_init_is_bit_identical_to_native(native, condition_spec):
    import torch
    from smallAntibodyGen.experiments.her2_routes_conditions import Condition
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    editor = ConditionedAbLang2(native, condition_spec)
    tokens = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 11, 12, 22]], dtype=torch.long)
    indices = [list(condition_spec.encode(
        Condition(family="affinity", bin="high", assay="engelhart2022_s1")))]
    record = editor.native_parity(tokens, condition_indices=indices)
    assert record["tables_zero_initialised"]
    assert record["bit_identical"], record
    assert record["max_abs_difference"] == 0.0


def test_i1_disabling_conditioning_restores_the_native_module(native, condition_spec):
    import torch
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    editor = ConditionedAbLang2(native, condition_spec)
    tokens = torch.tensor([[0, 14, 15, 22, 25, 0, 11, 22]], dtype=torch.long)
    with torch.no_grad():
        before = editor.native(tokens).clone()
    editor.disable_conditioning()
    with torch.no_grad():
        after = editor.native(tokens)
    assert torch.equal(before, after)


def test_i1_added_tables_do_not_grow_the_vocabulary(native, condition_spec):
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    editor = ConditionedAbLang2(native, condition_spec)
    report = editor.parameter_report()
    assert report["vocabulary_grown"] is False
    assert report["total"] == report["native"] + report["added"]
    assert "old tensor" in report["vocabulary_note"]


def test_i1_causal_route_vocabulary_growth_copies_every_row_exactly(prepared,
                                                                   condition_spec):
    from smallAntibodyGen.experiments import her2_routes_spec as spec
    from smallAntibodyGen.models.her2_routes_causal import ConditionedCausalInfiller
    repository = Path(__file__).resolve().parents[3]
    raw = repository / "data" / "raw" / "her2_functional_20260918"
    if not raw.is_dir():
        pytest.skip("the pinned HER2 release is not present")
    infiller = ConditionedCausalInfiller.from_release(raw, condition_spec, device="cpu")
    assert infiller.growth["rows_copied_exactly"] is True
    assert infiller.growth["old_vocab_size"] == 26
    assert infiller.growth["new_vocab_size"] == 28
    report = infiller.parameter_report()
    assert report["total"] == report["native"] + report["added"]
    assert report["all_additions_zero_initialised_at_construction"]
    assert isinstance(spec.ROUTES_SCHEMA, str)


def test_i1_the_two_appended_columns_cannot_move_the_renormalised_conditional(
        prepared, condition_spec):
    from smallAntibodyGen.experiments import her2_data
    from smallAntibodyGen.experiments.her2_routes_serial import (
        causal_serialization,
        her2_construct,
    )
    from smallAntibodyGen.models.her2_routes_causal import ConditionedCausalInfiller
    repository = Path(__file__).resolve().parents[3]
    raw = repository / "data" / "raw" / "her2_functional_20260918"
    if not raw.is_dir():
        pytest.skip("the pinned HER2 release is not present")
    infiller = ConditionedCausalInfiller.from_release(raw, condition_spec, device="cpu")
    scaffold = her2_data.load_scaffold(raw)
    construct = her2_construct(scaffold, her2_data.WT_CORE, record_id="i1")
    serialization = causal_serialization(construct,
                                         residues=construct.residues(construct.sites))
    report = infiller.vocabulary_invariance([serialization.tokens],
                                            [serialization.query_slots],
                                            pointers=[serialization.pointers])
    # Mathematically identical; the residual is fp32 rounding in a differently shaped
    # matmul, compared against the repository's own MEASURED tolerance.
    assert report["max_abs_error"] <= report["atol"] + report["rtol"] * 1e2
    assert "excluded from the renormalisation" in report["claim"]


def test_a_frozen_reference_carries_its_own_conditioning(native, condition_spec):
    """The check that a zero-initialised parity test cannot make.

    The conditioning hook used to be a closure over the wrapper. ``copy.deepcopy``
    copies the hooked module but treats a function as atomic, so a copied
    reference's hook still read the ORIGINAL wrapper's injection: identical state
    dicts, and logits that differed by 5.7955 once the tables were nonzero. Run
    with PERTURBED tables or this proves nothing, and the assertion below says so.
    """
    import copy

    import torch
    from smallAntibodyGen.experiments.her2_routes_conditions import Condition
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    editor = ConditionedAbLang2(native, condition_spec)
    tokens = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 11, 12, 22]], dtype=torch.long)
    indices = [list(condition_spec.encode(
        Condition(family="affinity", bin="high", assay="engelhart2022_s1")))]
    with torch.no_grad():
        row = condition_spec.encode(Condition(family="affinity", bin="high",
                                              assay="engelhart2022_s1"))[1]
        editor.condition_bin.weight[row] = torch.linspace(-0.5, 0.5,
                                                          editor.condition_bin.weight.shape[1])
    reference = editor.frozen_reference()
    record = editor.reference_parity(reference, tokens, condition_indices=indices)
    assert record["meaningful"], "the tables are still zero; this check would pass anyway"
    assert record["max_abs_difference"] == 0.0
    # A plain deepcopy is safe too, and the copy's hook is its own.
    clone = copy.deepcopy(editor)
    assert clone._hooked_block is not editor._hooked_block
    assert editor.reference_parity(clone, tokens,
                                   condition_indices=indices)["max_abs_difference"] == 0.0
    # Unhooking the copy leaves the ORIGINAL conditioned. The naive handle copy did not.
    clone.disable_conditioning()
    with torch.no_grad():
        conditioned = editor.forward(tokens, condition_indices=indices)
        unconditioned = clone.forward(tokens)
    assert not torch.allclose(conditioned, unconditioned)
    assert editor._handle is not None
    # The frozen reference is frozen.
    assert all(not parameter.requires_grad for parameter in reference.parameters())


def test_the_conditioned_editor_places_every_table_on_one_device(prepared, condition_spec):
    from smallAntibodyGen.experiments.her2_routes_env import (
        NATIVE_PACKAGE_ROOT,
        NATIVE_WEIGHTS_ROOT,
    )
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    root = Path(ARTIFACT_ROOT)
    editor = ConditionedAbLang2.from_release(root / NATIVE_WEIGHTS_ROOT, condition_spec,
                                             package_root=root / NATIVE_PACKAGE_ROOT,
                                             device="cpu")
    assert editor.assert_device_consistency()["device"] == "cpu"


def test_the_old_token_pathway_is_unchanged_by_the_vocabulary_growth(prepared,
                                                                    condition_spec):
    """The real I1 parity: the untouched released tables against the grown ones.

    Not a slice of one model, and not two different conditionals. The retained
    pre-growth embedding and head are swapped back in and the same old-token input
    is run through both.
    """
    from smallAntibodyGen.experiments import her2_data
    from smallAntibodyGen.experiments.her2_routes_serial import (
        causal_serialization,
        her2_construct,
    )
    from smallAntibodyGen.models.her2_routes_causal import ConditionedCausalInfiller
    repository = Path(__file__).resolve().parents[3]
    raw = repository / "data" / "raw" / "her2_functional_20260918"
    if not raw.is_dir():
        pytest.skip("the pinned HER2 release is not present")
    infiller = ConditionedCausalInfiller.from_release(raw, condition_spec, device="cpu")
    assert infiller.assert_device_consistency()["device"] == "cpu"
    scaffold = her2_data.load_scaffold(raw)
    construct = her2_construct(scaffold, her2_data.WT_CORE, record_id="parity")
    serialization = causal_serialization(construct,
                                         residues=construct.residues(construct.sites))
    old_tokens = [[23, *[infiller.vocabulary[residue]
                        for residue in construct.heavy + construct.light], 24]]
    record = infiller.native_token_parity(old_tokens)
    assert record["bit_identical"], record
    assert record["max_abs_difference"] == 0.0
    assert record["old_vocab_size"] == 26 and record["new_vocab_size"] == 28
    # The probe refuses a token the released vocabulary does not contain, so it
    # cannot be pointed at the new ids and report a vacuous pass.
    with pytest.raises(ValueError, match="not in the released"):
        infiller.native_token_parity([list(serialization.tokens)])


def test_every_prefix_pointer_is_reinjected_at_every_decode_step(prepared, condition_spec):
    """Teacher forcing injects k pointers; a decode that injects one is a different model.

    At zero initialisation both paths agree whatever the decoder does, so the
    pointer table is perturbed first. The measured disagreement of the
    one-pointer decoder was 1.47135 nats per step.
    """
    import torch
    from smallAntibodyGen.experiments import her2_data
    from smallAntibodyGen.experiments.her2_routes_conditions import BIN_NULL, Condition
    from smallAntibodyGen.experiments.her2_routes_serial import her2_construct
    from smallAntibodyGen.models.her2_routes_causal import ConditionedCausalInfiller
    repository = Path(__file__).resolve().parents[3]
    raw = repository / "data" / "raw" / "her2_functional_20260918"
    if not raw.is_dir():
        pytest.skip("the pinned HER2 release is not present")
    infiller = ConditionedCausalInfiller.from_release(raw, condition_spec, device="cpu")
    scaffold = her2_data.load_scaffold(raw)
    construct = her2_construct(scaffold, her2_data.WT_CORE, record_id="pointer")
    condition = Condition(family="affinity", bin=BIN_NULL, assay="engelhart2022_s1")
    with torch.no_grad():
        generator = torch.Generator().manual_seed(919)
        infiller.pointer.weight.copy_(torch.randn(infiller.pointer.weight.shape,
                                                  generator=generator) * 0.15)
    record = infiller.step_score_parity(construct, order=construct.sites,
                                        condition=condition,
                                        residues=construct.residues(construct.sites))
    assert record["pointer_table_nonzero"] and record["meaningful"]
    assert record["steps"] == len(construct.sites)
    assert record["max_abs_difference"] < 1e-4, record["per_step_max"]
    # EVERY step agrees, not just the first: the first position agreed to 8e-7 even
    # under the defect, and only later steps diverged.
    assert all(value < 1e-4 for value in record["per_step_max"])
    # A randomised emission order has the same property.
    import numpy as np
    from smallAntibodyGen.experiments.her2_routes_serial import emission_order
    shuffled = emission_order(construct.sites, order="random",
                              rng=np.random.default_rng(7))
    other = infiller.step_score_parity(construct, order=shuffled, condition=condition,
                                       residues=construct.residues(shuffled))
    assert other["max_abs_difference"] < 1e-4, other["per_step_max"]


def test_a_sampled_fill_can_be_rescored_and_agrees(prepared, condition_spec):
    """The recorded per-step density must be the density of the draw.

    With nonzero pointers this is the end-to-end version of the parity above: the
    decoder's own reported log probabilities are compared against a teacher-forced
    rescore of exactly what it emitted.
    """
    import numpy as np
    import torch
    from smallAntibodyGen.experiments import her2_data
    from smallAntibodyGen.experiments.her2_routes_conditions import BIN_NULL, Condition
    from smallAntibodyGen.experiments.her2_routes_serial import (
        causal_serialization,
        her2_construct,
    )
    from smallAntibodyGen.models.her2_routes_causal import ConditionedCausalInfiller
    repository = Path(__file__).resolve().parents[3]
    raw = repository / "data" / "raw" / "her2_functional_20260918"
    if not raw.is_dir():
        pytest.skip("the pinned HER2 release is not present")
    infiller = ConditionedCausalInfiller.from_release(raw, condition_spec, device="cpu")
    with torch.no_grad():
        generator = torch.Generator().manual_seed(919)
        infiller.pointer.weight.copy_(torch.randn(infiller.pointer.weight.shape,
                                                  generator=generator) * 0.15)
    scaffold = her2_data.load_scaffold(raw)
    construct = her2_construct(scaffold, her2_data.WT_CORE, record_id="fill")
    condition = Condition(family="affinity", bin=BIN_NULL, assay="engelhart2022_s1")
    residues, per_step = infiller.fill(construct, order=construct.sites,
                                       condition=condition,
                                       rng=torch.Generator().manual_seed(3))
    assert len(residues) == len(construct.sites)
    filled = construct.with_residues(construct.sites, residues)
    serialization = causal_serialization(filled, order=construct.sites,
                                         residues=residues)
    with torch.no_grad():
        rescored = infiller.canonical_log_probs(
            [serialization.tokens], [serialization.query_slots],
            pointers=[serialization.pointers],
            condition_indices=[list(condition_spec.encode(condition))])[0]
    targets = torch.as_tensor(serialization.targets, dtype=torch.long)
    picked = rescored.gather(1, targets.unsqueeze(-1)).squeeze(-1).numpy()
    assert np.allclose(per_step, picked, atol=1e-4), (per_step, picked)


# --- I8: batch and packing isolation --------------------------------------

def test_i8_a_prediction_agrees_alone_and_in_a_padded_batch(native, condition_spec):
    import torch
    from smallAntibodyGen.experiments.her2_routes_serial import ABLANG_PAD_ID
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    editor = ConditionedAbLang2(native, condition_spec)
    short = [0, 14, 15, 23, 22, 25, 0, 11, 22]
    long = [0, 14, 15, 23, 22, 25, 0, 11, 12, 13, 22]
    slots = [[3]]
    with torch.no_grad():
        alone = editor.canonical_log_probs(torch.tensor([short]), slots)
        padded = torch.tensor([short + [ABLANG_PAD_ID] * (len(long) - len(short)), long])
        together = editor.canonical_log_probs(padded, [[3], [3]])
    assert torch.allclose(alone[0, 0], together[0, 0], atol=1e-5), \
        "the padded batch changed a prediction; padding is not masked correctly"


def test_i8_changing_another_packed_example_cannot_change_a_prediction(native,
                                                                      condition_spec):
    import torch
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    editor = ConditionedAbLang2(native, condition_spec)
    first = [0, 14, 15, 23, 22, 25, 0, 11, 22]
    second = [0, 16, 17, 23, 22, 25, 0, 12, 22]
    other = [0, 19, 20, 23, 22, 25, 0, 13, 22]
    with torch.no_grad():
        left = editor.canonical_log_probs(torch.tensor([first, second]), [[3], [3]])
        right = editor.canonical_log_probs(torch.tensor([first, other]), [[3], [3]])
    assert torch.allclose(left[0, 0], right[0, 0], atol=0, rtol=0)


# --- I9: revision and caching ---------------------------------------------

def test_i9_a_revised_prediction_agrees_with_a_fresh_full_forward(prepared,
                                                                 condition_spec):
    import torch
    from smallAntibodyGen.experiments import her2_data
    from smallAntibodyGen.experiments.her2_routes_serial import (
        causal_serialization,
        her2_construct,
    )
    from smallAntibodyGen.models.her2_routes_causal import ConditionedCausalInfiller
    repository = Path(__file__).resolve().parents[3]
    raw = repository / "data" / "raw" / "her2_functional_20260918"
    if not raw.is_dir():
        pytest.skip("the pinned HER2 release is not present")
    infiller = ConditionedCausalInfiller.from_release(raw, condition_spec, device="cpu")
    scaffold = her2_data.load_scaffold(raw)
    construct = her2_construct(scaffold, her2_data.WT_CORE, record_id="i9")
    sites = construct.sites
    full = causal_serialization(construct, residues=construct.residues(sites))
    with torch.no_grad():
        teacher_forced = infiller.canonical_log_probs(
            [full.tokens], [full.query_slots], pointers=[full.pointers])
        # The same conditionals, recomputed one step at a time with NO cache reuse.
        stepwise = []
        tokens = list(full.tokens[:full.context_length + 1])
        for step in range(len(sites)):
            slot = full.context_length + step
            block = infiller.canonical_log_probs([tokens], [[slot]],
                                                 pointers=[[full.pointers[step]]])
            stepwise.append(block[0, 0])
            tokens.append(full.tokens[full.context_length + 1 + step])
    for step in range(len(sites)):
        assert torch.allclose(teacher_forced[0, step], stepwise[step], atol=1e-5), \
            f"step {step}: one causal forward disagrees with step-by-step recomputation"


# --- I18: resume equivalence ----------------------------------------------

def test_activation_checkpointing_preserves_native_keys_outputs_and_gradients(native, condition_spec):
    import copy
    import torch
    from smallAntibodyGen.models.ablang2_policy import ConditionedAbLang2
    plain = ConditionedAbLang2(copy.deepcopy(native), condition_spec)
    checked = ConditionedAbLang2(copy.deepcopy(native), condition_spec,
                                 activation_checkpointing=True)
    with torch.no_grad():
        plain.condition_bin.weight.add_(0.1)
    assert set(plain.state_dict()) == set(checked.state_dict())
    checked.load_state_dict(plain.state_dict(), strict=True)
    tokens = [[0, 14, 23, 22, 25, 0, 11, 22]]
    condition = [[0, 1, 0]]
    plain.train()
    checked.train()
    first = plain(tokens, condition_indices=condition)
    second = checked(tokens, condition_indices=condition)
    assert torch.equal(first, second)
    first.square().mean().backward()
    second.square().mean().backward()
    first_parameters = list(plain.parameters())
    second_parameters = list(checked.parameters())
    assert len(first_parameters) == len(second_parameters)
    for left, right in zip(first_parameters, second_parameters):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-5)
    checked.eval()
    reference = checked.frozen_reference()
    assert checked.reference_parity(reference, tokens, condition_indices=condition)["max_abs_difference"] == 0

def test_i18_a_resumed_short_run_matches_its_uninterrupted_counterpart(prepared, tmp_path):
    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    from smallAntibodyGen.experiments import her2_routes_spec as spec
    repository = Path(__file__).resolve().parents[3]
    config = repository / "configs" / "experiments" / "her2_generative_routes.json"
    context = spec.resolve_context(repository, config_path=config,
                                   artifact_root=ARTIFACT_ROOT,
                                   run_root=tmp_path / "flight")
    services = campaign.build_services(context, device="cpu")
    result = services.smoke("piggen_causal_infiller", directory=tmp_path / "smoke",
                            updates=3)
    assert result["passed"], result["failures"]
    assert result["detail"]["resume_reaches_same_step"]
    assert result["detail"]["resume_restored_exposure"]
    assert result["detail"]["negative_cases_raise"]


def test_the_gpu_profile_uses_three_signals(prepared, tmp_path):
    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    from smallAntibodyGen.experiments import her2_routes_spec as spec
    repository = Path(__file__).resolve().parents[3]
    config = repository / "configs" / "experiments" / "her2_generative_routes.json"
    context = spec.resolve_context(repository, config_path=config,
                                   artifact_root=ARTIFACT_ROOT,
                                   run_root=tmp_path / "flight")
    services = campaign.build_services(context, device="cpu")
    route = "piggen_causal_infiller"
    # The legacy config requests local interactions (quartets), so a microbatch must be even and
    # at least 4; 1 and 2 are rejected before any model is built. Profile the shape that runs.
    profile = services.profile(route, microbatches=(4,),
                               views=services.smoke_views(route, rows=2))
    assert "spills to system RAM" in profile["three_signal_rule"]
    for row in profile["rows"]:
        assert set(row["signals"]) == {"driver_free_above_floor",
                                       "allocator_peak_within_budget",
                                       "step_time_not_degraded"}


@pytest.mark.parametrize("route", ["ablang2_masked_editor", "piggen_causal_infiller"])
def test_runtime_batch_generation_is_seed_blind_and_matches_final_rescore(prepared, tmp_path, route):
    import numpy as np
    import torch
    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    from smallAntibodyGen.experiments import her2_routes_spec as spec
    from smallAntibodyGen.experiments import her2_routes_generate as generate
    from smallAntibodyGen.experiments.her2_routes_conditions import Condition
    from smallAntibodyGen.experiments.her2_routes_serial import her2_core_edit_map
    repository = Path(__file__).resolve().parents[3]
    context = spec.resolve_context(repository,
        config_path=repository / 'configs/experiments/her2_generative_routes.json',
        artifact_root=ARTIFACT_ROOT, run_root=tmp_path / route)
    services = campaign.build_services(context, device='cpu')
    model = services.build_route(route)
    model.eval()
    with torch.no_grad():
        rng = torch.Generator().manual_seed(915)
        for name, parameter in model.named_parameters():
            if 'condition' in name or 'pointer' in name:
                parameter.copy_(torch.randn(parameter.shape, generator=rng) * 0.1)
    condition = Condition(family='affinity', bin='high', assay=context.config['her2']['assay'])
    options = dict(draws=2, sites=her2_core_edit_map(), condition=condition,
                   strategy=generate.ORDER_DECLARED, seed=91, batch_size=2)
    left = generate.fill_batch(services, route, model, **options, seed_window='AAAAAAAAAA')
    right = generate.fill_batch(services, route, model, **options, seed_window='WWWWWWWWWW')
    assert left.windows == right.windows
    np.testing.assert_allclose(left.log_probabilities, right.log_probabilities, atol=1e-5)
    with torch.no_grad():
        scores = services.window_scores(route, model, left.windows, condition=condition,
                                         batch_size=2).numpy()
    np.testing.assert_allclose(left.log_probabilities, scores, atol=1e-4)
    revised = generate.fill_batch(services, route, model, **options, remask_rounds=1)
    with torch.no_grad():
        scores = services.window_scores(route, model, revised.windows, condition=condition,
                                         batch_size=2).numpy()
    np.testing.assert_allclose(revised.log_probabilities, scores, atol=1e-4)
    assert revised.diagnostics['terminal_distribution_log_probability'] is None

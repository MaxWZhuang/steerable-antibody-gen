"""The native DPLM trunk, LoRA, conditions, diffusion and decoding -- on tiny REAL torch models.

Everything here runs on CPU in fp32 against a two-layer, 32-wide trunk with the released
state-dict key names. What that does and does not establish, stated so the file cannot be
read as more than it is:

* the trunk's arithmetic is cross-checked against the INSTALLED ``transformers`` ESM
  implementation on identical random weights and identical tensors (eval mode, no network).
  That is evidence about the semantics of the re-implementation. It is **not** a numerical
  claim about the pretrained checkpoints: no pretrained weights are used anywhere here;
* the strict loader is exercised against a synthetic checkpoint written in the released
  layout (legacy ``inv_freq`` keys, the unused contact head), not against the real file;
* the "low-level roundtrip" tests prove save/load semantics of the code, not the quality of
  any trained model.

No test in this file asserts on source text.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_routes_dplm_pins as pins_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_routes_conditions import ConditionSpec
from smallAntibodyGen.models import dplm_policy as dplm

HEAVY = "ACDEFGHIKLMNPQRSTVWY"
LIGHT = "YWVTSRQPNMLKIHGFEDCA"
CANONICAL_NATIVE = set(dplm.DPLM_CANONICAL_IDS)


def tiny_config(**overrides):
    base = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                intermediate_size=64, max_position_embeddings=64, hidden_dropout_prob=0.0)
    base.update(overrides)
    return dplm.DPLMConfig(**base)


SPEC = ConditionSpec(assays=("a1", "a2"))


def make_model(*, lora=None, checkpointing=False, seed=1, std=0.3, diffusion=None,
               storage_dtype=torch.float32, config=None):
    config = config or tiny_config()
    native = dplm.NativeDPLM.random_init(config, seed=seed, std=std, storage_dtype=storage_dtype)
    torch.manual_seed(seed)
    return dplm.ConditionedDPLM(native, SPEC, lora=lora, activation_checkpointing=checkpointing,
                                diffusion=diffusion)


def construct(heavy=HEAVY, light=LIGHT, sites=((serial.HEAVY, 3), (serial.HEAVY, 4),
                                               (serial.HEAVY, 5)), record="a"):
    return serial.Construct(heavy=heavy, light=light, sites=tuple(sites), source="synthetic",
                            record_id=record)


def public_batch(constructs=None, *, masked=None):
    """Public (AbLang-id) tokens and query slots for equal-length constructs."""
    constructs = constructs or [construct(record="a"), construct(record="b")]
    rows, slots = [], []
    for item in constructs:
        serialization = serial.ablang_serialization(item, masked_sites=masked or item.sites,
                                                    order=item.sites)
        rows.append(serialization.tokens)
        slots.append(serialization.query_slots)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(slots, dtype=torch.long)


CONDITIONS = torch.tensor([[0, 0, 0], [1, 2, 1]], dtype=torch.long)


def randomise_tables(model, std=0.1, seed=5):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for table in model._tables():
            table.weight.copy_(torch.randn(table.weight.shape, generator=generator) * std)


# ---------------------------------------------------------------------------
# vocabulary, canonical table, boundary
# ---------------------------------------------------------------------------

def test_vocabulary_table_reproduces_the_pinned_vocab_bytes():
    """The 33-token table, joined as the released file is, has the hub's git blob id.

    ``6b946952...`` is the blob id the hub API lists for BOTH variants' ``vocab.txt`` and 93
    bytes is its size (61 characters + 32 newlines: LF-separated, no trailing newline).
    """
    text = "\n".join(dplm.DPLM_VOCAB)
    assert len(dplm.DPLM_VOCAB) == dplm.DPLM_VOCAB_SIZE == 33
    assert len(text.encode("ascii")) == 93
    assert dplm.parse_vocab(text) == dplm.DPLM_VOCAB
    import hashlib
    blob = hashlib.sha1(b"blob 93\0" + text.encode("ascii")).hexdigest()
    assert blob == "6b946952cc35537226f07fd70957ee2f848880d2"
    assert dplm.DPLM_VOCAB[dplm.DPLM_CLS_ID] == "<cls>"
    assert dplm.DPLM_VOCAB[dplm.DPLM_PAD_ID] == "<pad>"
    assert dplm.DPLM_VOCAB[dplm.DPLM_EOS_ID] == "<eos>"
    assert dplm.DPLM_VOCAB[dplm.DPLM_X_ID] == "X"
    assert dplm.DPLM_VOCAB[dplm.DPLM_MASK_ID] == "<mask>"


def test_the_explicit_canonical_table_is_the_vocabulary_in_canonical_order():
    from smallAntibodyGen.experiments.her2_data import CANONICAL
    assert len(dplm.DPLM_CANONICAL_IDS) == 20 == len(set(dplm.DPLM_CANONICAL_IDS))
    assert "".join(dplm.DPLM_VOCAB[i] for i in dplm.DPLM_CANONICAL_IDS) == CANONICAL
    # B U Z O . - X and the specials are never canonical
    for token in ("B", "U", "Z", "O", ".", "-", "X", "<mask>", "<pad>", "<cls>", "<eos>",
                  "<unk>", "<null_1>"):
        assert dplm.DPLM_VOCAB.index(token) not in CANONICAL_NATIVE


def test_the_boundary_translation_agrees_with_an_independent_native_construction():
    paired = construct()
    heavy_only = construct(light="", sites=((serial.HEAVY, 2),))
    for item in (paired, heavy_only):
        for masked in ((), item.sites):
            public = serial.ablang_token_ids(item, masked_sites=masked)
            native = dplm.dplm_token_ids(item, masked_sites=masked)
            assert tuple(dplm.ablang_to_dplm(public).tolist()) == native


def test_paired_layout_and_positions_equal_the_public_positions():
    item = construct()
    tokens = dplm.dplm_token_ids(item)
    h, l = len(item.heavy), len(item.light)
    assert tokens[0] == dplm.DPLM_CLS_ID
    assert tokens[1 + h:3 + h] == (dplm.DPLM_EOS_ID, dplm.DPLM_EOS_ID)
    assert tokens[3 + h] == dplm.DPLM_CLS_ID and tokens[-1] == dplm.DPLM_EOS_ID
    assert len(tokens) == h + l + 5
    solo = construct(light="", sites=((serial.HEAVY, 2),))
    tail = dplm.dplm_token_ids(solo)
    assert tail[-2:] == (dplm.DPLM_EOS_ID, dplm.DPLM_EOS_ID) and len(tail) == len(solo.heavy) + 3
    for source in (item, solo, construct(sites=((serial.HEAVY, 0), (serial.LIGHT, 4)))):
        assert dplm.dplm_positions(source) == serial.ablang_positions(source)
    serialization = dplm.dplm_serialization(item, masked_sites=item.sites, order=item.sites)
    assert all(serialization.tokens[slot] == dplm.DPLM_MASK_ID
               for slot in serialization.query_slots)
    with pytest.raises(ValueError, match="masked"):
        dplm.dplm_serialization(item, masked_sites=(), order=item.sites)


def test_a_padded_batch_translates_to_native_pad_never_to_mask():
    pad = serial.ABLANG_PAD_ID
    assert pad == 21 and dplm.DPLM_PAD_ID == 1 and dplm.DPLM_MASK_ID == 32
    public = torch.tensor([[0, 14, 15, 22, 25, pad, pad], [0, 14, 23, 22, 25, 0, 22]])
    native = dplm.ablang_to_dplm(public)
    assert native[0, -2:].tolist() == [dplm.DPLM_PAD_ID] * 2
    assert not bool((native[0] == dplm.DPLM_MASK_ID).any())
    assert native[1, 2].item() == dplm.DPLM_MASK_ID
    # AbLang M is public id 1; a native pad of 1 must not have been what padding became
    assert dplm.DPLM_RESIDUE_IDS["A"] != dplm.DPLM_PAD_ID


def test_ids_with_no_native_meaning_are_refused():
    with pytest.raises(ValueError, match="no native DPLM meaning"):
        dplm.ablang_to_dplm(torch.tensor([[0, serial.ABLANG_UNKNOWN_ID, 22]]))
    with pytest.raises(ValueError, match="must lie in"):
        dplm.ablang_to_dplm(torch.tensor([[0, 26, 22]]))
    with pytest.raises(ValueError, match="must lie in"):
        dplm.ablang_to_dplm(torch.tensor([[0, -1, 22]]))


def test_position_budget_is_validated_and_nothing_is_truncated():
    model = make_model(config=tiny_config(max_position_embeddings=16))
    long_row = torch.tensor([[0] + [14] * 20 + [22]])
    with pytest.raises(ValueError, match="exceeds the model's 16 positions"):
        model.canonical_log_probs(long_row, torch.tensor([[1]]))


# ---------------------------------------------------------------------------
# the trunk against the installed transformers ESM
# ---------------------------------------------------------------------------

def _hf_reference(config, *, std=0.3, seed=3):
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.EsmConfig(
        vocab_size=33, hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        position_embedding_type="rotary", token_dropout=True, mask_token_id=32,
        pad_token_id=1, emb_layer_norm_before=False, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, layer_norm_eps=config.layer_norm_eps)
    torch.manual_seed(seed)
    reference = transformers.EsmForMaskedLM(hf_config).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0.0, std)
        reference.lm_head.decoder.weight.copy_(reference.esm.embeddings.word_embeddings.weight)
    return reference


def test_tiny_trunk_matches_the_installed_transformers_esm_in_eval_mode():
    """Same random weights, same tensors, eval mode, no network: logits agree.

    Covers pre-scaled queries before rotary, both LayerNorms, ESM's GELU, the token-dropout
    rescale (the batch contains ``<mask>`` and, in one row, padding) and the tied head.
    """
    config = tiny_config()
    reference = _hf_reference(config)
    state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    state["lm_head.decoder.weight"] = state["esm.embeddings.word_embeddings.weight"].clone()
    ours = dplm.DPLMEncoder(config).eval()
    mine = ours.state_dict()
    assert set(mine) <= set(state), sorted(set(mine) - set(state))
    ours.load_state_dict({key: state[key] for key in mine}, strict=True)
    ids = torch.tensor([[0, 5, 6, 32, 7, 2, 2, 0, 8, 32, 9, 2],
                        [0, 5, 32, 7, 2, 2, 0, 8, 9, 2, 1, 1]])
    plain = torch.tensor([[0, 5, 6, 10, 7, 2, 2, 0, 8, 11, 9, 2],
                          [0, 5, 12, 7, 2, 2, 0, 8, 9, 2, 1, 1]])
    for batch in (ids, plain):
        attention = batch.ne(1)
        with torch.no_grad():
            theirs = reference(input_ids=batch, attention_mask=attention.long()).logits
            got = ours(batch)["logits"]
        assert float(got.abs().max()) > 0.05, "logits too small for the comparison to mean anything"
        assert torch.allclose(got[attention], theirs[attention], atol=1e-4, rtol=1e-4), (
            float((got[attention] - theirs[attention]).abs().max()))


def test_padding_does_not_change_a_row_and_rotary_positions_are_per_row():
    model = make_model()
    row = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 14, 22]])
    padded = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 14, 22, 21, 21, 21]])
    slots = torch.tensor([[3]])
    with torch.no_grad():
        alone = model.canonical_log_probs(row, slots)
        together = model.canonical_log_probs(padded, slots)
    assert torch.allclose(alone, together, atol=1e-5)


def test_the_expected_inventory_is_the_constructed_models_state_dict():
    config = tiny_config()
    model = dplm.DPLMEncoder(config)
    shapes = dplm.expected_state_shapes(config)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == shapes
    assert model.lm_head.decoder.weight is model.esm.embeddings.word_embeddings.weight


# ---------------------------------------------------------------------------
# the strict inventory and the loader
# ---------------------------------------------------------------------------

def released_layout(config, *, std=0.3, seed=2):
    """A raw checkpoint in the released layout: learned keys + legacy buffers + contact head."""
    model = dplm.DPLMEncoder(config)
    torch.manual_seed(seed)
    model.initialize_weights(std)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    inv = dplm.rotary_inv_freq(config.head_dim, config.rope_theta)
    for key in dplm.legacy_inv_freq_keys(config):
        state[key] = inv.clone()
    for key, shape in dplm.contact_head_shapes(config).items():
        state[key] = torch.randn(shape)
    # the released HF checkpoints also keep an unused learned absolute-position table
    state[dplm.POSITION_EMBEDDINGS_KEY] = torch.randn(config.max_position_embeddings,
                                                      config.hidden_size)
    return model, state


def test_the_released_layout_splits_into_the_learned_keys_and_validated_drops():
    config = tiny_config()
    model, state = released_layout(config)
    clean, record = dplm.split_checkpoint_state(state, config)
    assert set(clean) == set(dplm.expected_state_shapes(config))
    assert dplm.POSITION_EMBEDDINGS_KEY not in clean
    assert record["dropped"]["position_embeddings"]["shape"] == [
        config.max_position_embeddings, config.hidden_size]
    assert record["tied_output_equals_embedding"] is True
    assert "legacy_inv_freq" not in record["dropped"]
    assert record["preserved"]["rotary_inv_freq"]["keys"] == config.num_hidden_layers
    assert record["preserved"]["rotary_inv_freq"]["encoding"] == "fp16-rounded-analytical" \
        or record["preserved"]["rotary_inv_freq"]["max_relative_error_vs_analytical"] < 1e-6
    assert set(record["dropped"]["contact_head"]["keys"]) == set(dplm.contact_head_shapes(config))


@pytest.mark.parametrize("damage, message", [
    ("missing", "missing 1 expected key"),
    ("unexpected", "outside the expected inventory"),
    ("prefix_unexpected", "outside the expected inventory"),
    ("wrong_shape", "has shape"),
    ("untied", "tied output projection differs"),
    ("bad_inv_freq", "disagrees with the analytical rotary frequencies"),
    ("bad_inv_freq_shape", "has shape"),
    ("bad_contact_shape", "has shape"),
    ("partial_contact", "contact head is partial"),
    ("bad_position_ids", "is not arange"),
    ("not_tensor", "flat mapping of tensors"),
    ("bad_position_embeddings_shape", "has shape"),
    ("near_miss_position_key", "outside the expected inventory"),
])
def test_the_strict_inventory_refuses(damage, message):
    config = tiny_config()
    _model, state = released_layout(config)
    if damage == "missing":
        del state["esm.encoder.layer.1.output.dense.bias"]
    elif damage == "unexpected":
        state["esm.encoder.layer.0.attention.self.surprise"] = torch.zeros(1)
    elif damage == "prefix_unexpected":
        state["esm.pooler.dense.weight"] = torch.zeros(32, 32)
    elif damage == "wrong_shape":
        state["esm.encoder.layer.0.intermediate.dense.weight"] = torch.zeros(65, 32)
    elif damage == "untied":
        state["lm_head.decoder.weight"] = state["lm_head.decoder.weight"] + 1.0
    elif damage == "bad_inv_freq":
        key = dplm.legacy_inv_freq_keys(config)[0]
        state[key] = state[key] * 1.01
    elif damage == "bad_inv_freq_shape":
        state[dplm.legacy_inv_freq_keys(config)[0]] = torch.ones(3)
    elif damage == "bad_contact_shape":
        state["esm.contact_head.regression.weight"] = torch.zeros(1, 3)
    elif damage == "partial_contact":
        del state["esm.contact_head.regression.bias"]
    elif damage == "bad_position_ids":
        state["esm.embeddings.position_ids"] = torch.arange(64).expand(1, -1) + 1
    elif damage == "not_tensor":
        state["esm.encoder.emb_layer_norm_after.bias"] = [0.0]
    elif damage == "bad_position_embeddings_shape":
        state[dplm.POSITION_EMBEDDINGS_KEY] = torch.zeros(3, config.hidden_size)
    elif damage == "near_miss_position_key":
        state["esm.embeddings.position_embeddings.bias"] = torch.zeros(config.hidden_size)
    with pytest.raises(ValueError, match=message):
        dplm.split_checkpoint_state(state, config)


def test_position_ids_that_equal_arange_are_accepted_then_dropped():
    config = tiny_config()
    _model, state = released_layout(config)
    state["esm.embeddings.position_ids"] = torch.arange(64).expand(1, -1).clone()
    clean, record = dplm.split_checkpoint_state(state, config)
    assert "esm.embeddings.position_ids" not in clean
    assert "position_ids" in record["dropped"]


def fp16_layout(config):
    """The released layout whose per-layer inv_freq are the fp16-rounded analytical values."""
    model, state = released_layout(config)
    rounded = dplm.rotary_inv_freq(config.head_dim, config.rope_theta).half().float()
    for key in dplm.legacy_inv_freq_keys(config):
        state[key] = rounded.clone()
    return model, state, rounded


def test_fp16_rounded_frequencies_are_validated_and_preserved_not_recomputed(tmp_path):
    config = tiny_config()
    source, state, rounded = fp16_layout(config)
    analytical = dplm.rotary_inv_freq(config.head_dim, config.rope_theta)
    assert not torch.equal(rounded, analytical), "the fixture must be a real quantization"
    _clean, record = dplm.split_checkpoint_state(state, config)
    assert record["preserved"]["rotary_inv_freq"]["encoding"] == "fp16-rounded-analytical"
    directory, pins = write_variant(tmp_path, config, state)
    native = dplm.load_pinned_dplm(directory, pins=pins, variant="tiny_dplm")
    assert torch.equal(native.model.rotary_inv_freq, rounded)
    assert "rotary_inv_freq" not in native.model.state_dict()
    assert native.model.rotary_inv_freq.dtype == torch.float32
    # survives a storage-dtype cast and a frozen-reference copy bit for bit
    assert torch.equal(copy.deepcopy(native.model).rotary_inv_freq, rounded)
    assert torch.equal(copy.deepcopy(native.model).to(torch.float16).rotary_inv_freq, rounded)
    long_ids = torch.tensor([[0] + [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 5 + [2]])  # 52 tokens
    plain = dplm.DPLMEncoder(config).eval()
    plain.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        saved_logits = native.model(long_ids)["logits"]
        analytic_logits = plain(long_ids)["logits"]
    assert float((saved_logits - analytic_logits).abs().max()) > 1e-6, "quantization is invisible"


def test_saved_quantized_frequencies_reach_the_forward_and_match_hf_with_the_same_values():
    config = tiny_config()
    reference = _hf_reference(config)
    rounded = dplm.rotary_inv_freq(config.head_dim, config.rope_theta).half().float()
    hf_buffers = [buffer for name, buffer in reference.named_buffers() if name.endswith("inv_freq")]
    if not hf_buffers:
        pytest.skip("the installed transformers ESM keeps no inv_freq buffer to overwrite")
    for buffer in hf_buffers:
        assert buffer.shape == rounded.shape
        buffer.copy_(rounded)
    state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    state["lm_head.decoder.weight"] = state["esm.embeddings.word_embeddings.weight"].clone()
    ours = dplm.DPLMEncoder(config).eval()
    ours.load_state_dict({key: state[key] for key in ours.state_dict()}, strict=True)
    ours.set_rotary_inv_freq(rounded)
    ids = torch.tensor([[0] + [5, 6, 32, 7, 8, 9, 10, 11, 12, 13] * 5 + [2]])
    attention = ids.ne(1)
    with torch.no_grad():
        theirs = reference(input_ids=ids, attention_mask=attention.long()).logits
        got = ours(ids)["logits"]
        analytic = dplm.DPLMEncoder(config).eval()
        analytic.load_state_dict(ours.state_dict(), strict=True)
        default = analytic(ids)["logits"]
    assert torch.allclose(got, theirs, atol=1e-4, rtol=1e-4)
    assert float((default - theirs).abs().max()) > float((got - theirs).abs().max())


@pytest.mark.parametrize("damage, message", [
    ("partial", "of 2 per-layer inv_freq buffers"),
    ("layers_disagree", "disagrees with the other layers"),
    ("bf16_rounded", "disagrees with the analytical rotary frequencies"),
])
def test_bad_or_partial_saved_frequencies_are_refused(damage, message):
    config = tiny_config()
    _source, state, rounded = fp16_layout(config)
    keys = dplm.legacy_inv_freq_keys(config)
    if damage == "partial":
        del state[keys[1]]
    elif damage == "layers_disagree":
        # a different valid encoding in one layer: analytical vs fp16-rounded
        state[keys[1]] = dplm.rotary_inv_freq(config.head_dim, config.rope_theta)
    elif damage == "bf16_rounded":
        analytical = dplm.rotary_inv_freq(config.head_dim, config.rope_theta)
        for key in keys:
            state[key] = analytical.to(torch.bfloat16).float()
    with pytest.raises(ValueError, match=message):
        dplm.split_checkpoint_state(state, config)


def config_document(config, **overrides):
    document = {
        "_name_or_path": "tiny", "architectures": ["EsmForDPLM"],
        "attention_probs_dropout_prob": 0.0, "classifier_dropout": None,
        "dplm_type": "dplm_esm", "emb_layer_norm_before": False, "esmfold_config": None,
        "hidden_act": "gelu", "hidden_dropout_prob": config.hidden_dropout_prob,
        "hidden_size": config.hidden_size, "initializer_range": 0.02,
        "intermediate_size": config.intermediate_size, "is_folding_model": False,
        "layer_norm_eps": config.layer_norm_eps, "mask_token_id": 32,
        "max_position_embeddings": config.max_position_embeddings, "model_type": "esm",
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers, "pad_token_id": 1,
        "position_embedding_type": "rotary", "token_dropout": True, "torch_dtype": "float32",
        "transformers_version": "4.39.2", "use_cache": True, "vocab_list": None,
        "vocab_size": 33}
    document.update(overrides)
    return document


def write_variant(root, config, state, *, vocab_tokens=None, document=None,
                  architecture_hidden=None, variant="tiny_dplm"):
    """Write config/vocab/weights under ``root/models/tiny`` and return matching pins."""
    directory = root / "models" / "tiny"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(document or config_document(config)), encoding="utf-8")
    (directory / "vocab.txt").write_text("\n".join(vocab_tokens or dplm.DPLM_VOCAB),
                                         encoding="ascii")
    torch.save(state, directory / "pytorch_model.bin")
    files = {}
    for name in pins_lib.FILES:
        target = directory / name
        block = {"bytes": target.stat().st_size, "sha256": paths.sha256_file(target)}
        if name == "vocab.txt":
            block["git_blob_sha1"] = pins_lib.git_blob_sha1(target)
        files[name] = block
    architecture = {"hidden_size": architecture_hidden or config.hidden_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "num_attention_heads": config.num_attention_heads,
                    "intermediate_size": config.intermediate_size, "vocab_size": 33,
                    "max_position_embeddings": config.max_position_embeddings,
                    "layer_norm_eps": config.layer_norm_eps, "rope_theta": 10000.0}
    pins = {"schema": pins_lib.PINS_SCHEMA, "variants": {variant: {
        "repo": "local/tiny", "revision": "0" * 40, "logical_dir": "models/tiny",
        "files": files, "architecture": architecture}}}
    return directory, pins


def test_a_pinned_checkpoint_loads_strictly_and_reproduces_the_source_logits(tmp_path):
    config = tiny_config()
    source, state = released_layout(config)
    directory, pins = write_variant(tmp_path, config, state)
    native = dplm.load_pinned_dplm(directory, pins=pins, variant="tiny_dplm")
    assert native.load_record["strict"] and native.load_record["weights_only"]
    assert native.load_record["inventory"]["preserved"]["rotary_inv_freq"]["keys"] == 2
    assert native.identity["revision"] == "0" * 40
    assert native.identity["pins_sha256"] == pins_lib.pins_digest(pins)
    assert native.model.lm_head.decoder.weight is native.model.esm.embeddings.word_embeddings.weight
    assert not native.model.training
    ids = torch.tensor([[0, 5, 6, 32, 7, 2, 2, 0, 8, 9, 2]])
    with torch.no_grad():
        assert torch.equal(native.model(ids)["logits"], source.eval()(ids)["logits"])
    # nothing was left random: the loaded weights ARE the file's
    for key, value in native.model.state_dict().items():
        assert torch.equal(value, source.state_dict()[key])


def test_fp16_storage_computes_in_fp32_and_keeps_its_dtype(tmp_path):
    config = tiny_config()
    source, state = released_layout(config)
    directory, pins = write_variant(tmp_path, config, state)
    native = dplm.load_pinned_dplm(directory, pins=pins, variant="tiny_dplm",
                                   storage_dtype=torch.float16)
    assert {p.dtype for p in native.model.parameters()} == {torch.float16}
    rounded = dplm.DPLMEncoder(config)
    rounded.load_state_dict({k: v.half().float() for k, v in source.state_dict().items()})
    ids = torch.tensor([[0, 5, 6, 32, 7, 2, 2, 0, 8, 9, 2]])
    with torch.no_grad():
        got = native.model(ids)["logits"]
        assert got.dtype == torch.float32
        assert torch.allclose(got, rounded.eval()(ids)["logits"], atol=1e-5)


@pytest.mark.parametrize("damage", ["config_bytes", "weights_bytes", "missing_weights",
                                    "null_digest", "vocab_order", "config_disagrees_with_pins",
                                    "unexpected_key", "missing_key"])
def test_the_loader_fails_closed(tmp_path, damage):
    config = tiny_config()
    _source, state = released_layout(config)
    if damage == "unexpected_key":
        state["esm.mystery"] = torch.zeros(2)
    if damage == "missing_key":
        del state["lm_head.bias"]
    vocab = None
    if damage == "vocab_order":
        vocab = list(dplm.DPLM_VOCAB)
        vocab[4], vocab[5] = vocab[5], vocab[4]
    hidden = 48 if damage == "config_disagrees_with_pins" else None
    directory, pins = write_variant(tmp_path, config, state, vocab_tokens=vocab,
                                    architecture_hidden=hidden)
    entry = pins["variants"]["tiny_dplm"]
    if damage == "config_bytes":
        (directory / "config.json").write_text("{}", encoding="utf-8")
    elif damage == "weights_bytes":
        with (directory / "pytorch_model.bin").open("r+b") as stream:
            stream.seek(-8, 2)
            stream.write(b"\xff" * 8)
    elif damage == "missing_weights":
        (directory / "pytorch_model.bin").unlink()
    elif damage == "null_digest":
        entry["files"]["pytorch_model.bin"]["sha256"] = None
    with pytest.raises(ValueError):
        dplm.load_pinned_dplm(directory, pins=pins, variant="tiny_dplm")


def test_the_pins_file_refuses_a_placeholder_and_a_short_revision():
    pins = json.loads(json.dumps({"schema": pins_lib.PINS_SCHEMA, "variants": {"v": {
        "repo": "r", "revision": "0" * 40, "logical_dir": "models/v", "architecture": {},
        "files": {name: {"bytes": 10, "sha256": "0" * 64} for name in pins_lib.FILES}}}}))
    pins_lib.validate_pins(pins)
    broken = copy.deepcopy(pins)
    broken["variants"]["v"]["files"]["config.json"]["sha256"] = None
    with pytest.raises(ValueError, match="pins no digest"):
        pins_lib.validate_pins(broken)
    broken = copy.deepcopy(pins)
    broken["variants"]["v"]["revision"] = "abc"
    with pytest.raises(ValueError, match="full commit id"):
        pins_lib.validate_pins(broken)


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

def test_zero_condition_tables_are_a_bit_identical_no_op():
    model = make_model()
    tokens, _ = public_batch()
    parity = model.native_parity(tokens)
    assert parity["tables_zero_initialised"] and parity["bit_identical"]
    assert parity["max_abs_difference"] == 0.0


def test_perturbed_condition_tables_move_the_logits_and_receive_gradients():
    model = make_model()
    tokens, slots = public_batch()
    with torch.no_grad():
        before = model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS).clone()
    randomise_tables(model)
    with torch.no_grad():
        after = model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)
    assert float((after - before).abs().max()) > 1e-4
    # a row's condition is per row: the two rows differ only in their conditions
    same = model.canonical_log_probs(tokens[:1].repeat(2, 1), slots[:1].repeat(2, 1),
                                     condition_indices=CONDITIONS)
    assert float((same[0] - same[1]).abs().max()) > 1e-5


def test_condition_gradients_are_nonzero_from_zero_initialisation():
    model = make_model()
    tokens, slots = public_batch()
    loss = -model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)[..., 0].sum()
    loss.backward()
    for name, table in (("family", model.condition_family), ("bin", model.condition_bin),
                        ("assay", model.condition_assay)):
        assert table.weight.grad is not None
        assert float(table.weight.grad.abs().sum()) > 0.0, f"the {name} table got no gradient"


def test_conditions_are_refused_when_disabled_and_a_bad_shape_is_refused():
    model = make_model()
    tokens, slots = public_batch()
    with pytest.raises(ValueError, match=r"\(B, 3\)"):
        model.canonical_log_probs(tokens, slots, condition_indices=torch.zeros(2, 2))
    model.use_conditions = False
    with pytest.raises(ValueError, match="conditioning is disabled"):
        model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)
    model.canonical_log_probs(tokens, slots)              # no conditions requested: fine


def test_representation_pools_real_tokens_only_and_checks_the_mask():
    model = make_model()
    row = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 14, 22]])
    padded = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 14, 22, 21, 21]])
    with torch.no_grad():
        one = model.representation(row)
        two = model.representation(padded, attention_mask=padded.ne(21))
    assert one.shape == (1, 32) and torch.allclose(one, two, atol=1e-5)
    with pytest.raises(ValueError, match="disagrees with the padding"):
        model.representation(padded, attention_mask=torch.ones_like(padded))


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

LORA = {"rank": 4, "alpha": 8, "dropout": 0.0}


def test_lora_wraps_the_six_exact_targets_of_every_layer_exactly_once():
    model = make_model(lora=LORA)
    layers = model.config.num_hidden_layers
    wrapped = set(model.native.lora_modules())
    expected = {f"esm.encoder.layer.{i}.{target}" for i in range(layers)
                for target in dplm.LORA_TARGETS}
    assert wrapped == expected and len(model.lora_names) == 6 * layers == len(expected)
    # nothing else carries an adapter: not the LM head, not a LayerNorm, not the embedding
    for name, module in model.native.named_modules():
        if isinstance(module, dplm.DPLMLinear) and name not in expected:
            assert module.lora_A is None, name
    with pytest.raises(ValueError, match="already carries"):
        model.native.get_submodule("esm.encoder.layer.0.attention.self.query").attach_lora(
            rank=4, alpha=8, dropout=0.0)


def test_lora_rank_is_configurable_and_counted():
    small = make_model(lora={"rank": 2, "alpha": 4, "dropout": 0.0}).parameter_report()
    large = make_model(lora={"rank": 8, "alpha": 16, "dropout": 0.0}).parameter_report()
    assert large["lora"] == 4 * small["lora"] and small["lora"] > 0
    assert small["lora_rank"] == 2 and large["lora_rank"] == 8
    assert small["trainable"] == small["lora"] + small["condition_tables"]
    assert small["frozen"] == small["base"]


def test_a_fresh_adapter_is_an_exact_no_op_and_a_trained_one_is_not():
    model = make_model(lora=LORA)
    tokens, slots = public_batch()
    parity = model.lora_noop_parity(tokens, condition_indices=CONDITIONS)
    assert parity["lora_B_all_zero"] and parity["max_abs_difference"] == 0.0
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_B.normal_(0.0, 0.2)
    moved = model.lora_noop_parity(tokens, condition_indices=CONDITIONS)
    assert moved["max_abs_difference"] > 1e-4 and not moved["meaningful"]


def test_lora_freezes_the_base_and_updates_only_adapters_and_conditions():
    model = make_model(lora=LORA)
    tokens, slots = public_batch()
    base_names = set(model.base_state_keys())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == set(model.adapter_state_keys())
    assert not (trainable & base_names)
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = -model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)[
            ..., 0].sum()
        loss.backward()
        optimizer.step()
    after = model.state_dict()
    for key in base_names:
        assert torch.equal(snapshot[key], after[key]), f"frozen base tensor {key} moved"
    moved_adapters = [k for k in model.adapter_state_keys() if "lora_B" in k
                      and not torch.equal(snapshot[k], after[k])]
    assert len(moved_adapters) == len(model.lora_names)
    assert not torch.equal(snapshot["condition_bin.weight"], after["condition_bin.weight"])


def test_fp16_base_with_fp32_adapters_trains_and_keeps_its_precision():
    model = make_model(lora=LORA, storage_dtype=torch.float16)
    policy = model.dtype_policy()
    assert policy["frozen_or_base_storage"] == ["torch.float16"]
    assert policy["trainable"] == ["torch.float32"] and policy["compute"] == "torch.float32"
    tokens, slots = public_batch()
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_B.normal_(0.0, 0.2)
    loss = -model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)[..., 0].sum()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(g.dtype == torch.float32 and bool(torch.isfinite(g).all())
                         for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert model.dtype_policy() == policy
    model.to("cpu")
    assert model.dtype_policy() == policy, ".to(device) must not change any parameter dtype"


def test_activation_checkpointing_gives_the_same_gradients_as_the_plain_pass():
    plain = make_model(lora=LORA, checkpointing=False)
    checked = make_model(lora=LORA, checkpointing=True)
    randomise_tables(plain)
    with torch.no_grad():
        for module in plain.native.lora_modules().values():
            module.lora_B.normal_(0.0, 0.2)
    checked.load_state_dict(plain.state_dict())
    tokens, slots = public_batch()
    grads = []
    for model in (plain, checked):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = -model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)[
            ..., 0].sum()
        loss.backward()
        grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                      if p.requires_grad})
    assert set(grads[0]) == set(grads[1]) and grads[0]
    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], atol=1e-6, rtol=1e-5), name
    a_grad = [g for n, g in grads[1].items() if "lora_A" in n]
    assert any(float(g.abs().sum()) > 0 for g in a_grad), "checkpointing lost the adapter gradient"


def _scoring_model(use_lora):
    lora = dict(LORA, dropout=0.1) if use_lora else None
    model = make_model(lora=lora, checkpointing=True,
                       config=tiny_config(hidden_dropout_prob=0.1))
    randomise_tables(model)
    if use_lora:
        with torch.no_grad():
            for module in model.native.lora_modules().values():
                module.lora_B.normal_(0.0, 0.1)
    return model.train()


def _repeated_scoring_loss(score, tokens, slots, indices, *, seed=321):
    """Three scorings of edited inputs, then one loss. ``score`` is the path under test."""
    torch.manual_seed(seed)
    terms = []
    for step in range(3):
        current = tokens.clone()
        current[:, 1] = 1 + step
        terms.append(score(current, slots, indices)[:, :, step].sum())
    return torch.stack(terms).log_softmax(0)[1].neg()


def _reference_score(model):
    """Oracle: the same scoring without the outer checkpoint (layer checkpoints retained)."""
    return lambda tokens, slots, indices: model._score_native(
        model._public(tokens), torch.as_tensor(slots, dtype=torch.long), indices)


def _trainable_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


@pytest.mark.parametrize("use_lora", [False, True])
def test_whole_scoring_recompute_matches_the_uncheckpointed_reference(use_lora):
    model = _scoring_model(use_lora)
    tokens, slots = public_batch()
    runs = []
    for score in (_reference_score(model),
                  lambda t, s, c: model.canonical_log_probs(t, s, condition_indices=c)):
        model.zero_grad(set_to_none=True)
        loss = _repeated_scoring_loss(score, tokens, slots, CONDITIONS)
        loss.backward()
        runs.append((loss.detach(), _trainable_grads(model)))
    (ref_loss, ref_grads), (loss, grads) = runs
    assert torch.allclose(ref_loss, loss, atol=1e-6, rtol=1e-6)
    assert ref_grads.keys() == grads.keys()
    assert any(n.startswith("condition_") for n in grads), "table gradients were lost"
    assert any(float(g.abs().sum()) > 0 for n, g in grads.items() if n.startswith("condition_"))
    if use_lora:
        assert any("lora_B" in n and float(g.abs().sum()) > 0 for n, g in grads.items())
    for name in ref_grads:
        assert torch.allclose(ref_grads[name], grads[name], atol=2e-5, rtol=2e-5), name


@pytest.mark.parametrize("use_lora", [False, True])
def test_recompute_survives_caller_buffer_mutation_between_scoring_and_backward(use_lora):
    model = _scoring_model(use_lora)
    tokens, slots = public_batch()
    tokens_np, slots_np, indices_np = tokens.numpy().copy(), slots.numpy().copy(), \
        CONDITIONS.numpy().copy()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(9)
    terms = []
    for step in range(3):
        current = tokens_np.copy()
        current[:, 1] = 1 + step
        terms.append(model.canonical_log_probs(
            current, slots_np, condition_indices=indices_np)[:, :, step].sum())
        current[:] = tokens_np[:, ::-1]          # the caller reuses its buffers afterwards
    loss = torch.stack(terms).log_softmax(0)[1].neg()
    slots_np[:] = slots_np[:, ::-1]
    indices_np[:] = indices_np[::-1]
    loss.backward()
    grads = _trainable_grads(model)

    model.zero_grad(set_to_none=True)
    torch.manual_seed(9)
    reference = _repeated_scoring_loss(
        _reference_score(model), tokens, slots, CONDITIONS, seed=9)
    reference.backward()
    expected = _trainable_grads(model)
    assert torch.allclose(loss.detach(), reference.detach(), atol=1e-6, rtol=1e-6)
    assert grads.keys() == expected.keys()
    for name in expected:
        assert torch.allclose(expected[name], grads[name], atol=2e-5, rtol=2e-5), name
    # the mutated buffers would have given a different answer, so the copy is what saved it
    assert not np.array_equal(slots_np, slots.numpy()) \
        or not np.array_equal(indices_np, CONDITIONS.numpy())


@pytest.mark.parametrize("use_lora", [False, True])
def test_whole_scoring_recompute_reduces_retained_non_parameter_storage(use_lora):
    model = _scoring_model(use_lora)
    tokens, slots = public_batch()
    parameters = {p.untyped_storage().data_ptr() for p in model.parameters()}
    retained = []
    for score in (_reference_score(model),
                  lambda t, s, c: model.canonical_log_probs(t, s, condition_indices=c)):
        seen = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameters:
                seen[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = _repeated_scoring_loss(score, tokens, slots, CONDITIONS)
        loss.backward()
        retained.append(sum(seen.values()))
    before, after = retained
    assert after < before / 2, (before, after)


def test_eval_and_no_grad_scoring_do_not_use_the_recompute_boundary(monkeypatch):
    model = _scoring_model(False)
    tokens, slots = public_batch()

    def refuse(*args, **kwargs):
        raise AssertionError("the scoring checkpoint must apply only when training with grad")

    monkeypatch.setattr(dplm.torch.utils.checkpoint, "checkpoint", refuse)
    model.eval()
    with_grad = model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)
    with torch.no_grad():
        no_grad = model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS)
    assert torch.equal(with_grad, no_grad)
    model.train()
    with torch.no_grad():                       # training mode but no grad: still no boundary
        assert model.canonical_log_probs(
            tokens, slots, condition_indices=CONDITIONS).shape == with_grad.shape


# ---------------------------------------------------------------------------
# diffusion corruption
# ---------------------------------------------------------------------------

def test_q_sample_masks_at_the_declared_rate_and_never_touches_specials_or_immutable():
    rows, length, timesteps = 2000, 50, 500
    x0 = torch.full((rows, length), dplm.DPLM_RESIDUE_IDS["A"], dtype=torch.long)
    x0[:, 0] = dplm.DPLM_CLS_ID
    x0[:, 24] = dplm.DPLM_EOS_ID
    x0[:1000, 45:] = dplm.DPLM_PAD_ID
    immutable = torch.zeros(rows, length, dtype=torch.bool)
    immutable[:, 10:20] = True
    maskable = dplm.non_special_mask(x0, immutable=immutable)
    assert not bool(maskable[:, 0].any()) and not bool(maskable[:, 10:20].any())
    assert not bool(maskable[:1000, 45:].any())
    generator = torch.Generator().manual_seed(0)
    uniform = torch.rand(rows, length, generator=generator)
    t = torch.full((rows,), 100, dtype=torch.long)
    out = dplm.q_sample(x0, t, maskable, uniform, num_timesteps=timesteps)
    mask = out["mask_mask"]
    assert not bool((mask & ~maskable).any())
    assert bool((out["x_t"][mask] == dplm.DPLM_MASK_ID).all())
    assert bool(torch.equal(out["x_t"][~mask], x0[~mask]))
    fraction = float(mask[maskable].float().mean())
    assert fraction == pytest.approx(100 / 500, abs=0.01)
    everything = dplm.q_sample(x0, torch.full((rows,), timesteps), maskable, uniform,
                               num_timesteps=timesteps)
    assert bool((everything["mask_mask"] == maskable).all())


def test_diffusion_weights_match_the_upstream_definitions():
    t = torch.tensor([1, 250, 500])
    assert torch.equal(dplm.diffusion_weights(t, num_timesteps=500, weighting="constant"),
                       torch.ones(3))
    linear = dplm.diffusion_weights(t, num_timesteps=500, weighting="linear")
    assert linear.tolist() == pytest.approx([1.0, 251 / 500, 1 / 500])
    with pytest.raises(ValueError):
        dplm.diffusion_weights(t, num_timesteps=500, weighting="cosine")


def unmasked_public_rows():
    items = [construct(record="a"), construct(record="b", heavy=HEAVY[::-1])]
    return torch.tensor([serial.ablang_token_ids(item, masked_sites=()) for item in items])


def test_diffusion_loss_is_finite_reproducible_and_reaches_the_trunk():
    settings = {"num_timesteps": 4, "weighting": "constant", "decode_steps": 3}
    model = make_model(diffusion=settings)
    tokens = unmasked_public_rows()
    loss, diagnostics = model.diffusion_loss(tokens, rng=np.random.default_rng(7),
                                             condition_indices=CONDITIONS)
    again, _ = model.diffusion_loss(tokens, rng=np.random.default_rng(7),
                                    condition_indices=CONDITIONS)
    other, _ = model.diffusion_loss(tokens, rng=np.random.default_rng(8),
                                    condition_indices=CONDITIONS)
    assert torch.isfinite(loss) and float(loss) > 0
    assert float(loss) == float(again), "the same generator state must corrupt identically"
    assert float(loss) != float(other)
    assert diagnostics["active_targets"] > 0 and diagnostics["num_timesteps"] == 4
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.native.parameters())
    assert float(model.condition_bin.weight.grad.abs().sum()) > 0


def test_diffusion_loss_refuses_an_input_that_already_holds_mask():
    model = make_model()
    tokens, _ = public_batch()
    with pytest.raises(ValueError, match="already holds <mask>"):
        model.diffusion_loss(tokens, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# iterative reparameterized decoding
# ---------------------------------------------------------------------------

def test_decoding_is_canonical_only_with_an_immutable_scaffold_and_counts_its_forwards():
    model = make_model(diffusion={"num_timesteps": 8, "weighting": "constant",
                                  "decode_steps": 6})
    tokens, slots = public_batch()
    generator = torch.Generator().manual_seed(11)
    result = model.iterative_decode(tokens, slots, condition_indices=CONDITIONS,
                                    generator=generator, keep_history=True)
    native_in = dplm.ablang_to_dplm(tokens)
    rows = torch.arange(2)[:, None].expand_as(slots)
    scaffold = torch.ones_like(native_in, dtype=torch.bool)
    scaffold[rows, slots] = False
    assert torch.equal(result.native_tokens[scaffold], native_in[scaffold])
    decoded = result.native_tokens[rows, slots]
    assert set(decoded.flatten().tolist()) <= CANONICAL_NATIVE
    assert (result.canonical >= 0).all() and (result.canonical < 20).all()
    assert result.forward_calls == 6 == result.steps == len(result.history)
    for snapshot in result.history:
        assert torch.equal(snapshot[scaffold], native_in[scaffold])
        design = snapshot[~scaffold]
        assert set(design.tolist()) <= CANONICAL_NATIVE | {dplm.DPLM_MASK_ID}
    document = result.document()
    assert document["trajectory_logprob"] is None
    assert "normalized log probability" in document["trajectory_logprob_note"]
    again = model.iterative_decode(tokens, slots, condition_indices=CONDITIONS,
                                   generator=torch.Generator().manual_seed(11))
    assert torch.equal(again.native_tokens, result.native_tokens)


def test_decoding_restores_the_callers_train_mode_and_uses_no_dropout():
    model = make_model(config=tiny_config(hidden_dropout_prob=0.3))
    tokens, slots = public_batch()
    model.train()
    first = model.iterative_decode(tokens, slots, generator=torch.Generator().manual_seed(1),
                                   steps=3)
    assert model.training, "decoding must restore the caller's mode"
    second = model.iterative_decode(tokens, slots, generator=torch.Generator().manual_seed(1),
                                    steps=3)
    assert torch.equal(first.native_tokens, second.native_tokens)


def _mode_map(model):
    """Every module's own ``training`` flag, by name: the state a decode must leave unchanged."""
    return {name: module.training for name, module in model.named_modules()}


def _adapter_flags(model):
    return {name: module.lora_enabled for name, module in model.native.lora_modules().items()}


def _decode_once(model, tokens, slots, *, seed=1):
    return model.iterative_decode(tokens, slots, condition_indices=CONDITIONS,
                                  generator=torch.Generator().manual_seed(seed), steps=3)


@pytest.mark.parametrize("root_training", [True, False])
def test_decoding_restores_every_modules_own_mode_and_not_the_roots(root_training):
    """A mixed model (root and a child in opposite modes) must come back exactly as it was.

    ``self.train(root_flag)`` would write the root's flag into every descendant, so the child
    held in the other mode would silently change. The decode itself still runs with dropout off:
    the tokens equal those of a model that was in eval mode throughout.
    """
    config = tiny_config(hidden_dropout_prob=0.3)
    reference = make_model(lora=LORA, config=config)
    reference.eval()
    tokens, slots = public_batch()
    expected = _decode_once(reference, tokens, slots)

    model = make_model(lora=LORA, config=config)
    model.train(root_training)
    model.native.train(not root_training)
    before, adapters = _mode_map(model), _adapter_flags(model)
    assert before[""] is root_training and before["native"] is (not root_training)
    assert len(set(before.values())) == 2, "the fixture really is a mixed-mode model"
    result = _decode_once(model, tokens, slots)
    assert _mode_map(model) == before
    assert _adapter_flags(model) == adapters
    assert torch.equal(result.native_tokens, expected.native_tokens), (
        "the decode must run with dropout off whatever mode the caller left")


@pytest.mark.parametrize("root_training", [True, False])
def test_decoding_restores_every_modules_own_mode_when_the_decode_raises(root_training,
                                                                         monkeypatch):
    model = make_model(lora=LORA, config=tiny_config(hidden_dropout_prob=0.3))
    tokens, slots = public_batch()
    model.train(root_training)
    model.native.train(not root_training)
    before = _mode_map(model)
    inside = {}

    def boom(*args, **kwargs):
        inside.update(_mode_map(model))
        raise RuntimeError("the sampler failed")
    monkeypatch.setattr(dplm, "iterative_reparam_decode", boom)
    with pytest.raises(RuntimeError, match="the sampler failed"):
        _decode_once(model, tokens, slots)
    assert inside and not any(inside.values()), "the decode ran with every module in eval mode"
    assert _mode_map(model) == before


def test_decoding_restores_a_single_module_that_the_caller_left_in_a_different_mode():
    model = make_model(lora=LORA)
    tokens, slots = public_batch()
    model.eval()
    target = next(iter(model.native.lora_modules().values())).lora_dropout
    target.train()
    before = _mode_map(model)
    assert sum(before.values()) == 1
    _decode_once(model, tokens, slots)
    assert _mode_map(model) == before


def test_lora_disabled_restores_each_modules_own_flag_on_every_exit():
    model = make_model(lora=LORA)
    modules = model.native.lora_modules()
    names = sorted(modules)
    assert len(names) >= 2
    modules[names[0]].lora_enabled = False       # one adapter deliberately off before entry
    before = _adapter_flags(model)
    assert set(before.values()) == {True, False}

    def all_off():
        return not any(_adapter_flags(model).values())

    with model.lora_disabled():
        assert all_off()
    assert _adapter_flags(model) == before, "a normal exit must not re-enable the module that was off"

    with pytest.raises(RuntimeError, match="boom"):
        with model.lora_disabled():
            assert all_off()
            raise RuntimeError("boom")
    assert _adapter_flags(model) == before

    with model.lora_disabled():
        with model.lora_disabled():
            assert all_off()
        assert all_off(), "an inner exit must leave the outer context disabled"
    assert _adapter_flags(model) == before

    with model.lora_disabled():
        with pytest.raises(RuntimeError, match="inner"):
            with model.lora_disabled():
                raise RuntimeError("inner")
        assert all_off(), "an inner exception must leave the outer context disabled"
    assert _adapter_flags(model) == before


def test_a_nested_lora_disabled_keeps_the_logits_unadapted_until_the_outer_exit():
    model = make_model(lora=LORA)
    model.eval()
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_B.normal_(0.0, 0.2)
    tokens, _slots = public_batch()
    native = model._public(tokens)
    with torch.no_grad():
        adapted = model.logits_native_ids(native, condition_indices=CONDITIONS)
        with model.lora_disabled():
            base = model.logits_native_ids(native, condition_indices=CONDITIONS)
            with model.lora_disabled():
                pass
            after_inner = model.logits_native_ids(native, condition_indices=CONDITIONS)
        restored = model.logits_native_ids(native, condition_indices=CONDITIONS)
    assert not torch.equal(adapted, base), "the fixture's adapters must actually matter"
    assert torch.equal(base, after_inner)
    assert torch.equal(adapted, restored)


@pytest.mark.parametrize("strategy", ["reparam-uncond-deterministic-linear",
                                      "reparam-cond-deterministic-cosine",
                                      "reparam-uncond-stochastic1.0-linear",
                                      "reparam-cond-stochastic2.0-cosine"])
def test_hostile_logits_cannot_produce_a_non_canonical_token_or_move_the_scaffold(strategy):
    """Logits that overwhelmingly prefer X, B, U, Z, O, '.', '-', <unk>, <mask> and every special."""
    generator = torch.Generator().manual_seed(3)
    tokens = torch.tensor([[0, 5, 6, 32, 32, 32, 7, 2, 2, 0, 8, 9, 2],
                           [0, 5, 6, 32, 32, 32, 7, 2, 2, 0, 8, 9, 2]])
    immutable = torch.ones_like(tokens, dtype=torch.bool)
    immutable[:, 3:6] = False
    hostile = [0, 1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    def logits_fn(current):
        logits = torch.randn(current.shape[0], current.shape[1], 33, generator=generator)
        logits[..., hostile] += 60.0
        return logits
    result = dplm.iterative_reparam_decode(
        logits_fn, tokens, immutable, steps=5, strategy=strategy,
        generator=torch.Generator().manual_seed(9),
        canonical_ids=torch.tensor(dplm.DPLM_CANONICAL_IDS), keep_history=True)
    assert torch.equal(result.native_tokens[immutable], tokens[immutable])
    assert set(result.native_tokens[:, 3:6].flatten().tolist()) <= CANONICAL_NATIVE
    assert result.forward_calls == 5


def test_a_design_slot_that_is_not_mask_is_refused_and_bad_strategies_are_refused():
    model = make_model()
    tokens, slots = public_batch()
    revealed = tokens.clone()
    revealed[0, slots[0, 0]] = serial.ABLANG_RESIDUE_IDS["A"]
    with pytest.raises(ValueError, match="must hold <mask>"):
        model.iterative_decode(revealed, slots)
    with pytest.raises(ValueError, match="Unsupported decoding strategy"):
        dplm.iterative_reparam_decode(
            lambda t: torch.zeros(t.shape[0], t.shape[1], 33), dplm.ablang_to_dplm(tokens),
            torch.zeros_like(tokens, dtype=torch.bool), steps=2, strategy="argmax",
            canonical_ids=torch.tensor(dplm.DPLM_CANONICAL_IDS))
    with pytest.raises(ValueError):
        dplm.DiffusionSettings(decoding_strategy="reparam-uncond-deterministic-quadratic")


def test_decode_result_places_residues_at_the_declared_sites():
    model = make_model(diffusion={"num_timesteps": 8, "weighting": "constant",
                                  "decode_steps": 4})
    item = construct(sites=((serial.HEAVY, 3), (serial.HEAVY, 4), (serial.LIGHT, 2)))
    tokens, slots = public_batch([item, item])
    result = model.iterative_decode(tokens, slots, generator=torch.Generator().manual_seed(2))
    decoded = result.canonical.tolist()
    assert len(decoded[0]) == 3
    native = result.native_tokens
    for row in range(2):
        for column, slot in enumerate(slots[row].tolist()):
            assert dplm.NATIVE_TO_CANONICAL[int(native[row, slot])] == decoded[row][column]


# ---------------------------------------------------------------------------
# checkpoint protocol
# ---------------------------------------------------------------------------

def state_bundle(seed=1):
    return checkpoints.TrainingState(
        mask_rng=np.random.default_rng([seed, 1]), sampler_rng=np.random.default_rng([seed, 2]),
        decode_rng=np.random.default_rng([seed, 3]))


IDENTITY = checkpoints.training_identity(
    route="dplm_650m_lora_masked", arm="full", seed=1, config_sha256="c", snapshot_sha256="s",
    external_sha256="e", condition_sha256="v", split_sha256="p", native_sha256="n")


def test_a_native_identity_changes_the_training_identity_and_absence_leaves_it_alone():
    without = checkpoints.training_identity(
        route="r", arm="a", seed=1, config_sha256="c", snapshot_sha256="s", external_sha256="e",
        condition_sha256="v", split_sha256="p")
    assert "native_sha256" not in without
    one = checkpoints.training_identity(
        route="r", arm="a", seed=1, config_sha256="c", snapshot_sha256="s", external_sha256="e",
        condition_sha256="v", split_sha256="p", native_sha256="n1")
    two = checkpoints.training_identity(
        route="r", arm="a", seed=1, config_sha256="c", snapshot_sha256="s", external_sha256="e",
        condition_sha256="v", split_sha256="p", native_sha256="n2")
    assert one["identity_sha256"] != two["identity_sha256"] != without["identity_sha256"]
    with pytest.raises(ValueError, match="native_sha256"):
        checkpoints.require_identity(one, two, path="x")


def test_lora_checkpoint_holds_exactly_the_adapter_keys_and_round_trips_strictly(tmp_path):
    model = make_model(lora=LORA)
    randomise_tables(model)
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_A.normal_(0.0, 0.3)
            module.lora_B.normal_(0.0, 0.2)
    state, meta = model.checkpoint_state()
    assert set(state) == set(model.adapter_state_keys())
    assert not set(state) & set(model.base_state_keys())
    assert meta["format"] == dplm.FORMAT_ADAPTER and meta["base_identity"] == model.base_identity
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state_bundle(), identity=IDENTITY)
    payload = torch.load(str(target), map_location="cpu", weights_only=False)
    assert set(payload["model"]) == set(state) and payload["model_format"]["format"] == \
        dplm.FORMAT_ADAPTER
    fresh = make_model(lora=LORA)
    before = fresh.dtype_policy()
    checkpoints.restore(target, model=fresh, heads=None, optimizer=None, scheduler=None,
                        state=state_bundle(2), identity=IDENTITY)
    for key in model.adapter_state_keys():
        assert torch.equal(model.state_dict()[key], fresh.state_dict()[key]), key
    assert fresh.dtype_policy() == before
    tokens, slots = public_batch()
    with torch.no_grad():
        assert torch.equal(model.canonical_log_probs(tokens, slots, condition_indices=CONDITIONS),
                           fresh.canonical_log_probs(tokens, slots,
                                                     condition_indices=CONDITIONS))
    info = checkpoints.load_inference(target, model=make_model(lora=LORA), heads=None,
                                      identity=IDENTITY)
    assert info["restored"] == ["model", "heads"]


def test_an_adapter_checkpoint_is_refused_for_a_different_base_or_settings_or_keys():
    model = make_model(lora=LORA)
    state, meta = model.checkpoint_state()
    other_base = make_model(lora=LORA, seed=99)
    assert other_base.base_identity != model.base_identity
    with pytest.raises(ValueError, match="different base"):
        other_base.load_checkpoint_state(state, meta)
    other_rank = make_model(lora={"rank": 8, "alpha": 8, "dropout": 0.0})
    with pytest.raises(ValueError, match="LoRA settings differ"):
        other_rank.load_checkpoint_state(state, meta)
    target = make_model(lora=LORA)
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="keys differ"):
        target.load_checkpoint_state(missing, meta)
    extra = dict(state, **{"native.esm.encoder.layer.0.attention.self.query.weight":
                           torch.zeros(32, 32)})
    with pytest.raises(ValueError, match="keys differ"):
        target.load_checkpoint_state(extra, meta)
    with pytest.raises(ValueError, match="different diffusion|Diffusion settings differ"):
        make_model(lora=LORA, diffusion={"num_timesteps": 7}).load_checkpoint_state(state, meta)
    with pytest.raises(ValueError, match="is 'dplm-adapter-only/1' and this model is"):
        make_model().load_checkpoint_state(state, meta)
    with pytest.raises(ValueError, match="no model_format"):
        target.load_checkpoint_state(state, None)
    with pytest.raises(ValueError, match="strictly"):
        target.load_checkpoint_state(state, meta, strict=False)


def test_a_plain_state_dict_checkpoint_is_refused_by_a_protocol_model(tmp_path):
    model = make_model(lora=LORA)
    plain = torch.nn.Linear(2, 2)
    target = checkpoints.save(tmp_path / "plain.pt", model=plain, heads=None, optimizer=None,
                              scheduler=None, state=state_bundle(), identity=IDENTITY)
    with pytest.raises(ValueError, match="no model_format"):
        checkpoints.restore(target, model=model, heads=None, optimizer=None, scheduler=None,
                            state=state_bundle(2), identity=IDENTITY)
    adapter = checkpoints.save(tmp_path / "adapter.pt", model=model, heads=None,
                               optimizer=None, scheduler=None, state=state_bundle(),
                               identity=IDENTITY)
    with pytest.raises(ValueError, match="checkpoint protocol"):
        checkpoints.restore(adapter, model=torch.nn.Linear(2, 2), heads=None, optimizer=None,
                            scheduler=None, state=state_bundle(2), identity=IDENTITY)


def test_full_finetune_checkpoint_round_trips_the_whole_state_strictly():
    model = make_model()
    randomise_tables(model)
    state, meta = model.checkpoint_state()
    assert meta["format"] == dplm.FORMAT_FULL and meta["lora"] is None
    assert set(state) == set(model.state_dict())
    fresh = make_model(seed=1)
    with torch.no_grad():
        for parameter in fresh.native.parameters():
            parameter.add_(1.0)
    fresh.load_checkpoint_state(state, meta)
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key]), key
    assert fresh.native.lm_head.decoder.weight is fresh.native.esm.embeddings.word_embeddings.weight
    broken = dict(state)
    broken.pop("condition_bin.weight")
    with pytest.raises(ValueError, match="keys differ"):
        fresh.load_checkpoint_state(broken, meta)


def test_full_checkpoint_with_conflicting_tied_copies_is_refused_before_any_mutation():
    model = make_model()
    state, meta = model.checkpoint_state()
    state = {key: value.clone() for key, value in state.items()}
    state["native.lm_head.decoder.weight"] += 1.0
    fresh = make_model(seed=1)
    before = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="tied output projection differs"):
        fresh.load_checkpoint_state(state, meta)
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} moved during a refused load"
    wrong_shape = {key: value.clone() for key, value in model.checkpoint_state()[0].items()}
    wrong_shape["condition_bin.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="has shape"):
        fresh.load_checkpoint_state(wrong_shape, meta)


def test_an_adapter_over_a_different_base_precision_is_refused():
    model = make_model(lora=LORA)
    state, meta = model.checkpoint_state()
    other = make_model(lora=LORA)
    other.base_identity = dict(other.base_identity, storage_dtype="torch.float16")
    with pytest.raises(ValueError, match="different base"):
        other.load_checkpoint_state(state, meta)


def test_frozen_reference_shares_lora_base_storage_but_not_adapters_or_parameters():
    model = make_model(lora=LORA)
    reference = model.frozen_reference()
    mine = dict(model.native.named_parameters())
    theirs = dict(reference.native.named_parameters())
    base_keys = [k for k, p in mine.items() if not p.requires_grad]
    assert base_keys
    for key in base_keys:
        assert theirs[key] is not mine[key], "the wrapper must be its own Parameter"
        assert theirs[key].data_ptr() == mine[key].data_ptr(), f"{key} storage was duplicated"
        assert not theirs[key].requires_grad
    for key, parameter in mine.items():
        if parameter.requires_grad:
            assert theirs[key].data_ptr() != parameter.data_ptr(), f"{key} adapter is shared"
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_B.add_(1.0)
        model.condition_bin.weight.add_(1.0)
    assert not torch.equal(reference.condition_bin.weight, model.condition_bin.weight)
    assert all(bool(torch.all(m.lora_B == 0)) for m in reference.native.lora_modules().values())


def test_full_tune_reference_has_independent_weights():
    model = make_model()
    reference = model.frozen_reference()
    with torch.no_grad():
        model.native.esm.embeddings.word_embeddings.weight.add_(1.0)
    assert not torch.equal(reference.native.esm.embeddings.word_embeddings.weight,
                           model.native.esm.embeddings.word_embeddings.weight)


def test_frozen_reference_is_independent_and_matches_under_nonzero_tables():
    model = make_model(lora=LORA)
    randomise_tables(model)
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_B.normal_(0.0, 0.2)
    reference = model.frozen_reference()
    tokens, _ = public_batch()
    parity = model.reference_parity(reference, tokens, condition_indices=CONDITIONS)
    assert parity["meaningful"] and parity["max_abs_difference"] == 0.0
    assert not any(p.requires_grad for p in reference.parameters())
    assert reference.native is not model.native
    with torch.no_grad():
        model.condition_bin.weight.add_(1.0)
    assert not torch.equal(reference.condition_bin.weight, model.condition_bin.weight)
    report = model.parameter_report()
    assert report["total"] == report["base"] + report["condition_tables"] + report["lora"]
    assert model.assert_device_consistency()["device"] == "cpu"

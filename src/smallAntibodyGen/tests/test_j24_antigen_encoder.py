"""
Contract tests for J24, the antigen-encoder comparison.

J24 asks: given the same pretrained antibody model, data, supervision,
initialization, and 1024-token antigen crop, does frozen ESM-2 produce stronger
antigen-dependent policy behavior than the scratch antigen encoder?

Every test here defends the "same ... except the encoder" clause. None of them
measures a result -- J24 has not run, and cannot until a promoted stage-2
checkpoint and the approved inner-development assets exist. What they pin is that
if it does run, a difference is attributable to the encoder.

The three failure modes they exist for are all silent:

1. the arms see different antigen RESIDUES, because their tokenizers spend
   different numbers of special tokens inside the same budget;
2. the arms start from different FUSION weights, because the two antigen
   encoders consume different amounts of init RNG;
3. a STALE cache entry is reused across a change that should have invalidated it.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.antigen_tokenization import build_antigen_tokenizer
from smallAntibodyGen.experiments import antigen_residues as res
from smallAntibodyGen.experiments import init_parity
from smallAntibodyGen.experiments.antigen_cache import (
    AntigenCacheKey,
    FrozenAntigenCache,
    sequence_digest,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig
from smallAntibodyGen.tokenizer import AminoAcidTokenizer

ANTIGEN = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(400))


def _config(**overrides) -> MLMConfig:
    base = dict(
        vocab_size=35,
        pad_token_id=0,
        max_length=64,
        antigen_max_length=128,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
    )
    base.update(overrides)
    return MLMConfig(**base)


@pytest.fixture
def scratch_tokenizer():
    return build_antigen_tokenizer("scratch", AminoAcidTokenizer(), "unused")


@pytest.fixture
def esm_tokenizer():
    pytest.importorskip("transformers", reason="optional 'esm' extra not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return build_antigen_tokenizer(
                "esm", AminoAcidTokenizer(), "facebook/esm2_t6_8M_UR50D"
            )
        except OSError:
            pytest.skip("ESM tokenizer unavailable (offline)")


# --------------------------------------------------------------------------- #
# 1. Same residues
# --------------------------------------------------------------------------- #
def test_the_two_tokenizers_really_do_spend_different_overheads(
    scratch_tokenizer, esm_tokenizer
):
    """
    The premise, re-measured rather than quoted.

    If this ever stops being true the cropping below becomes unnecessary, and a
    test asserting a difference that no longer exists is better than silently
    keeping machinery nobody can justify.
    """
    scratch_overhead = res.special_token_overhead(scratch_tokenizer, 512)
    esm_overhead = res.special_token_overhead(esm_tokenizer, 512)
    assert scratch_overhead == 3, "scratch: [CLS] [OTHER_CHAIN] ... [EOS]"
    assert esm_overhead == 2, "esm: <cls> ... <eos>"
    assert scratch_overhead != esm_overhead


def test_uncorrected_budgets_would_hand_one_arm_an_extra_residue(
    scratch_tokenizer, esm_tokenizer
):
    """
    The defect this module prevents, stated as a number.

    At the J24 budget of 1024 the ESM arm would see 1022 residues and the scratch
    arm 1021 -- one extra residue, in the same arm, on every antigen.
    """
    assert res.residue_capacity(scratch_tokenizer, 1024) == 1021
    assert res.residue_capacity(esm_tokenizer, 1024) == 1022


def test_common_budget_takes_the_minimum_and_reports_the_sacrifice(
    scratch_tokenizer, esm_tokenizer
):
    """The shared budget is the most constrained arm's, and what the other arm
    gave up is reported rather than hidden."""
    budget = res.common_residue_budget(
        {"scratch": scratch_tokenizer, "esm": esm_tokenizer}, 1024
    )
    assert budget.residues == 1021
    assert budget.per_arm_capacity == {"scratch": 1021, "esm": 1022}
    assert budget.sacrificed == {"scratch": 0, "esm": 1}


def test_both_arms_encode_byte_identical_residues(scratch_tokenizer, esm_tokenizer):
    """
    The invariant J24 depends on: after cropping, the two arms carry the same
    residue count. Their token ids differ -- different vocabularies -- which is
    why the count is derived from each tokenizer's own measured overhead rather
    than compared directly.
    """
    tokenizers = {"scratch": scratch_tokenizer, "esm": esm_tokenizer}
    budget = res.common_residue_budget(tokenizers, 128)
    encoded = res.encode_arms_identically(ANTIGEN, tokenizers, budget)

    counts = {
        name: res.retained_residue_count(ids, tokenizers[name], budget.token_budget)
        for name, ids in encoded.items()
    }
    assert counts["scratch"] == counts["esm"] == budget.residues
    # And neither arm overflowed its budget.
    assert all(len(ids) <= budget.token_budget for ids in encoded.values())


def test_a_short_antigen_is_untouched_by_cropping(scratch_tokenizer, esm_tokenizer):
    """Cropping must bind only when the antigen actually exceeds the budget;
    otherwise it would silently shorten most of the corpus."""
    tokenizers = {"scratch": scratch_tokenizer, "esm": esm_tokenizer}
    budget = res.common_residue_budget(tokenizers, 128)
    short = "ACDEFGHIKL"
    assert res.crop_antigen(short, budget.residues) == short


def test_a_tokenizer_that_is_not_one_token_per_residue_is_rejected():
    """
    Cropping equalizes arms only if token count is affine in residue count. A
    tokenizer that merges residues into subwords would break that, and the
    failure would look like a small unexplained arm difference rather than an
    error.
    """

    class Subwordish:
        pad_id = 0

        def encode(self, sequence, max_length):
            # Two residues per token: token count is not residue count + c.
            return [1] + [7] * ((len(sequence) + 1) // 2) + [2]

    with pytest.raises(res.AntigenTokenizationError, match="one token per residue"):
        res.special_token_overhead(Subwordish(), 512)


# --------------------------------------------------------------------------- #
# 2. Same starting weights
# --------------------------------------------------------------------------- #
def test_naive_seeding_does_not_give_the_arms_matching_fusion_weights():
    """
    Why `init_parity` exists at all.

    Both arms are built under the same seed, and their shared fusion weights
    still differ, because the scratch antigen encoder consumes a different amount
    of init RNG than an ESM encoder would. Here the effect is produced with two
    scratch models of different antigen DEPTH, which exercises the same mechanism
    without requiring the ESM download.
    """
    torch.manual_seed(1234)
    left = AntibodyAntigenCrossAttention(_config(n_layers=2))
    torch.manual_seed(1234)
    right = AntibodyAntigenCrossAttention(_config(n_layers=3))

    with pytest.raises(AssertionError):
        init_parity.assert_shared_parameters_match(left, right)


def test_reinitialization_makes_shared_parameters_bit_identical():
    """The correction: after the per-module pass, the shared parameters match
    regardless of what the antigen encoder consumed."""
    left_cfg, right_cfg = _config(n_layers=2), _config(n_layers=3)
    torch.manual_seed(1234)
    left = AntibodyAntigenCrossAttention(left_cfg)
    torch.manual_seed(9999)  # deliberately different: the pass must not depend on it
    right = AntibodyAntigenCrossAttention(right_cfg)

    init_parity.reinitialize_shared_modules(left, left_cfg, seed=7)
    init_parity.reinitialize_shared_modules(right, right_cfg, seed=7)

    init_parity.assert_shared_parameters_match(left, right)


def test_reinitialization_leaves_the_encoders_alone():
    """
    The pass must NOT equalize the thing under test. If it reached into the
    antigen encoder the comparison would be between two identical models.
    """
    cfg = _config()
    torch.manual_seed(3)
    model = AntibodyAntigenCrossAttention(cfg)
    before = {
        k: v.clone() for k, v in model.antigen_encoder.state_dict().items()
    }
    init_parity.reinitialize_shared_modules(model, cfg, seed=7)
    after = model.antigen_encoder.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"antigen encoder changed at {key}"


def test_reinitialization_restores_the_global_rng_state():
    """
    Fixing an arm asymmetry must not introduce a different one. If this pass left
    the RNG advanced, the two arms would diverge in data order and dropout
    instead of in fusion weights.
    """
    cfg = _config()
    model = AntibodyAntigenCrossAttention(cfg)
    torch.manual_seed(55)
    state_before = torch.get_rng_state()
    init_parity.reinitialize_shared_modules(model, cfg, seed=7)
    assert torch.equal(state_before, torch.get_rng_state())


def test_module_seeds_do_not_shift_when_the_shared_list_grows():
    """
    Seeds are hashed per module NAME, not derived from position, so adding a
    module to SHARED_MODULE_NAMES cannot silently change the values of the
    modules listed after it -- which would invalidate every arm recorded before
    the addition.
    """
    a = init_parity._module_seed(7, "fusion_mlp")
    b = init_parity._module_seed(7, "fusion_mlp")
    c = init_parity._module_seed(7, "lm_head")
    assert a == b
    assert a != c


def test_the_shared_module_list_covers_what_the_model_actually_builds():
    """
    Guard against drift: a module added to the dual-stream model must be
    consciously classified as shared or arm-specific. Otherwise a new head would
    quietly sit outside the parity guarantee.
    """
    cfg = _config()
    model = AntibodyAntigenCrossAttention(cfg)
    built = {name for name, _ in model.named_children()}
    classified = (
        set(init_parity.SHARED_MODULE_NAMES)
        | set(init_parity.ARM_SPECIFIC_MODULE_NAMES)
        # The antibody stream is warm-started from stage 2 in BOTH arms, so it is
        # neither freshly initialized nor arm-specific.
        | {"antibody_encoder"}
        # Dropout has no parameters.
        | {"fusion_dropout"}
    )
    unclassified = built - classified
    assert not unclassified, (
        f"modules not classified for J24 parity: {sorted(unclassified)}. "
        "Add each to SHARED_MODULE_NAMES or ARM_SPECIFIC_MODULE_NAMES."
    )


# --------------------------------------------------------------------------- #
# 3. The cache cannot go stale
# --------------------------------------------------------------------------- #
def _key(**overrides) -> AntigenCacheKey:
    base = dict(
        esm_model_name="facebook/esm2_t6_8M_UR50D",
        tokenizer_signature="tok-abc",
        token_budget=1024,
        residue_budget=1021,
        sequence_sha256=sequence_digest(ANTIGEN),
        dtype="torch.float32",
    )
    base.update(overrides)
    return AntigenCacheKey(**base)


@pytest.mark.parametrize(
    "field,value",
    [
        ("esm_model_name", "facebook/esm2_t12_35M_UR50D"),
        ("tokenizer_signature", "tok-different"),
        ("token_budget", 512),
        ("residue_budget", 1022),
        ("sequence_sha256", sequence_digest("ACDEF")),
        ("dtype", "torch.float16"),
        ("format_version", "antigen-embedding-cache/2"),
    ],
)
def test_every_key_field_invalidates_the_entry(tmp_path: Path, field, value):
    """
    Each field is in the key because changing it alone changes the tensor. A
    field that did not invalidate would be the stale-cache bug: a number computed
    under different conditions, returned as if current.
    """
    cache = FrozenAntigenCache(tmp_path)
    original = _key()
    cache.put(original, torch.ones(4, 8))
    assert cache.get(original) is not None
    assert cache.get(_key(**{field: value})) is None


def test_a_hit_returns_the_same_tensor(tmp_path: Path):
    cache = FrozenAntigenCache(tmp_path)
    encoding = torch.randn(3, 8)
    key = _key()
    cache.put(key, encoding)
    assert torch.equal(cache.get(key), encoding)


def test_the_cache_survives_a_new_process(tmp_path: Path):
    """Disk backing is the point: the corpus has ~3,176 distinct antigens and the
    backbone is frozen, so a cache that dies with the process buys nothing."""
    key = _key()
    encoding = torch.randn(3, 8)
    FrozenAntigenCache(tmp_path).put(key, encoding)

    fresh = FrozenAntigenCache(tmp_path)  # no shared memory state
    assert torch.equal(fresh.get(key), encoding)


def test_a_file_whose_stored_key_disagrees_is_a_miss(tmp_path: Path):
    """
    The key is stored beside the tensor and re-checked. A hand-copied or
    collided file must not be indistinguishable from a hit -- the cache exists to
    be trustworthy, not merely fast.
    """
    cache = FrozenAntigenCache(tmp_path)
    key = _key()
    cache.put(key, torch.ones(2, 2))
    path = tmp_path / f"{key.digest()}.pt"
    torch.save({"key": "some-other-digest", "encoding": torch.zeros(2, 2)}, path)

    assert FrozenAntigenCache(tmp_path).get(key) is None


def test_get_or_compute_computes_once(tmp_path: Path):
    cache = FrozenAntigenCache(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return torch.ones(2, 2)

    key = _key()
    first = cache.get_or_compute(key, compute)
    second = cache.get_or_compute(key, compute)
    assert len(calls) == 1
    assert torch.equal(first, second)


def test_cache_reports_its_own_cost(tmp_path: Path):
    """J24's report has a cache-cost column; the cache has to supply it."""
    cache = FrozenAntigenCache(tmp_path)
    key = _key()
    cache.get(key)
    cache.put(key, torch.ones(2, 2))
    cache.get(key)
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["entries"] == 1


def test_cached_and_uncached_encodings_agree(tmp_path: Path):
    """
    Cached/uncached parity, on the real frozen encoder.

    The cache is only legitimate if it returns exactly what recomputation would.
    A frozen backbone in eval mode is a deterministic function of its input, so
    the tolerance here is exact equality rather than a fudge factor -- anything
    looser would hide the failure this test exists to catch.
    """
    pytest.importorskip("transformers", reason="optional 'esm' extra not installed")
    from smallAntibodyGen.models.esm_antigen_encoder import ESMAntigenEncoder

    cfg = _config(antigen_encoder_type="esm", antigen_max_length=64)
    try:
        encoder = ESMAntigenEncoder(cfg)
    except OSError:
        pytest.skip("ESM weights unavailable (offline)")
    encoder.eval()

    ids = torch.randint(4, 20, (1, 24))
    mask = torch.ones_like(ids)

    # Cache the HIDDEN STATES only. `forward` returns `(hidden, attention_mask)`,
    # and the mask is a function of the input ids rather than of the frozen
    # backbone -- caching it would store a second copy of something already known
    # and give it an independent chance to go stale.
    def encode():
        with torch.no_grad():
            hidden, _mask = encoder(ids, mask)
        return hidden

    uncached = encode()

    cache = FrozenAntigenCache(tmp_path)
    key = _key(token_budget=64, residue_budget=62)
    stored = cache.get_or_compute(key, encode)      # miss -> computes
    replayed = cache.get_or_compute(key, encode)    # hit  -> must not recompute

    assert cache.stats() == {
        "hits": 1, "misses": 1, "lookups": 2, "hit_rate": 0.5, "entries": 1
    }
    assert torch.equal(stored, uncached), "cached value differs from a fresh encode"
    assert torch.equal(replayed, uncached), "replayed value differs from a fresh encode"

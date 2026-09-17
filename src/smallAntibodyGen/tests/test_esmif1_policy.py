"""Pins the constrained fixed-geometry editing policy.

The policy in `smallAntibodyGen.models.esmif1_policy` can be silently wrong in
ways that no shape assertion would catch, so the tests below are built around
those failure modes rather than around coverage:

- **the wrong decision index**, which would score residue ``p`` against the
  logits for ``p+1`` and still produce a well-formed normalized distribution;
- **the wrong support**, i.e. normalizing over the full 35-token vocabulary or
  leaking probability into forced positions;
- **an incremental-cache hole**, the bug upstream's own `sample` ships: skipping
  the decoder call at an already-known position means the token before it is
  never ingested, so sampling and scoring silently describe different processes;
- **a dead gradient**, from `no_grad` or a detached tensor on the decoder path.

The toy decoder here *emulates upstream's incremental semantics* -- with an
`incremental_state` it ingests only the last token -- so the cache-hole test is
causal rather than a comment. `test_skipping_a_forced_position_changes_the_score`
demonstrates the divergence explicitly, which is what gives the agreement tests
their teeth.

Everything except the final section runs on the toy modules with no optional
dependency. The device section is skipped without CUDA. The final section builds
a **real** `GVPTransformerModel` from a hand-written args namespace at ~32 dims:
it needs the `esm-if1` extra but never a checkpoint, so it downloads nothing. It
goes through the `upstream_stack` fixture, which installs the compatibility layer
*before* any `esm` import and restores the process-global state it changes.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from smallAntibodyGen.models.esmif1_policy import (
    CANONICAL_RESIDUES,
    ESMIF1_TOKENS,
    ConstrainedEditPolicy,
    ConstrainedEditSpace,
    EditableSite,
    ESMIF1Alphabet,
    FixedGeometry,
    PREFIX_TOKEN,
)


BIOTITE_AVAILABLE = importlib.util.find_spec("biotite") is not None
PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None
ESM_AVAILABLE = importlib.util.find_spec("esm") is not None
ESM_IF1_STACK = BIOTITE_AVAILABLE and PYG_AVAILABLE and ESM_AVAILABLE
CUDA_AVAILABLE = torch.cuda.is_available()

VOCAB = len(ESMIF1_TOKENS)
CHANNELS = 4

# Ten canonical residues with forced positions before the first site (0, 1),
# between sites (3, 4, 6, 7) and after the last one (9). The gaps are what make
# a cache hole observable.
CONTEXT = "ACDEFGHIKL"
SITES = (
    EditableSite(2, ("D", "N")),
    EditableSite(5, ("G", "S")),
    EditableSite(8, ("K", "R")),
)

CONTEXT_16 = "ACDEFGHIKLMNPQRSTVWY"
SITES_16 = tuple(
    EditableSite(position, (CONTEXT_16[position], "A"))
    for position in range(2, 18)
)


# --------------------------------------------------------------------------
# Toy backbone
# --------------------------------------------------------------------------

class _ToyEncoder(nn.Module):
    """A deterministic, geometry-only encoder that counts its calls.

    Mirrors the one property the caching argument rests on: the encoder is a
    function of coordinates, padding mask and confidence, and never of the
    sequence.
    """

    def __init__(self, channels: int = CHANNELS, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, channels, dtype=dtype))
        self.calls = 0

    def forward(self, coords, padding_mask, confidence, return_all_hiddens=False):
        self.calls += 1
        finite = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        summary = finite.sum(dim=(-2, -1))                       # (B, T)
        states = summary.unsqueeze(-1) * self.scale + confidence.unsqueeze(-1)
        return {
            "encoder_out": [states.transpose(0, 1)],             # (T, B, C)
            "encoder_padding_mask": [padding_mask],
            "encoder_embedding": [],
            "encoder_states": [],
        }


class _ToyDecoder(nn.Module):
    """A causal decoder that emulates upstream's incremental cache semantics.

    Teacher-forced, the logits at position ``t`` are the running sum of a learned
    per-token effect over prefix tokens ``0..t`` plus a projection of the encoder
    states. Incrementally, only the *last* token of each call is ingested --
    exactly `transformer_decoder.py:162-165` plus
    `esm/multihead_attention.py:286-323`. The two paths agree if and only if the
    caller presents every prefix token exactly once, in order.
    """

    def __init__(
        self,
        vocab: int = VOCAB,
        channels: int = CHANNELS,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.token_effect = nn.Parameter(
            torch.randn(vocab, vocab, generator=generator, dtype=dtype) * 0.3
        )
        self.enc_proj = nn.Linear(channels, vocab, bias=False, dtype=dtype)
        self.dropout_module = nn.Dropout(0.5)
        self.saw_training: bool | None = None
        self.ingested_stream: list[torch.Tensor] = []

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
    ):
        self.saw_training = self.training
        context = self.enc_proj(encoder_out["encoder_out"][0].mean(dim=0))   # (B, V)
        if incremental_state is None:
            running = self.token_effect[prev_output_tokens].cumsum(dim=1)    # (B, T, V)
            logits = running + context.unsqueeze(1)
        else:
            state = incremental_state.setdefault("toy", {})
            last = prev_output_tokens[:, -1]
            self.ingested_stream.append(last.clone())
            running = state.get("running")
            step = self.token_effect[last]
            state["running"] = step if running is None else running + step
            logits = (state["running"] + context).unsqueeze(1)               # (B, 1, V)
        return logits.transpose(1, 2), {}                                    # B x V x T


class _ToyModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def _toy_policy(
    context: str = CONTEXT, sites: tuple[EditableSite, ...] = SITES
) -> ConstrainedEditPolicy:
    model = _ToyModel(_ToyEncoder(), _ToyDecoder())
    return ConstrainedEditPolicy(model, ConstrainedEditSpace(context, sites))


def _toy_coords(num_residues: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        num_residues, 3, 3, generator=generator, dtype=torch.float64
    ) * 4.0


@pytest.fixture
def policy() -> ConstrainedEditPolicy:
    return _toy_policy()


@pytest.fixture
def geometry(policy: ConstrainedEditPolicy) -> FixedGeometry:
    return policy.encode_geometry(_toy_coords(len(policy.space.context)))


# --------------------------------------------------------------------------
# Native alphabet
# --------------------------------------------------------------------------

def test_token_table_reproduces_the_native_layout():
    alphabet = ESMIF1Alphabet.native()

    assert len(alphabet) == 35
    assert ESMIF1_TOKENS[:4] == ("<null_0>", "<pad>", "<eos>", "<unk>")
    assert ESMIF1_TOKENS[31:] == ("<null_1>", "<mask>", "<cath>", "<af2>")
    assert alphabet.padding_idx == 1
    assert alphabet.mask_idx == 32
    assert alphabet.prefix_idx == alphabet.index_of(PREFIX_TOKEN) == 33
    # The 20 canonical residues are the first 20 standard tokens, at 4..23.
    assert set(ESMIF1_TOKENS[4:24]) == set(CANONICAL_RESIDUES)
    for token in CANONICAL_RESIDUES:
        assert 4 <= alphabet.index_of(token) <= 23


def test_alphabet_rejects_a_table_that_is_not_the_native_one():
    shuffled = ("X",) + ESMIF1_TOKENS[1:]

    with pytest.raises(ValueError, match="does not match the native"):
        ESMIF1Alphabet(shuffled)
    with pytest.raises(ValueError, match="does not match the native"):
        ESMIF1Alphabet(ESMIF1_TOKENS[:-1])


def test_alphabet_matching_rejects_a_shifted_special_index():
    """A table that matches but whose derived indices do not is the near-miss
    that would corrupt every score without raising."""
    class _Impostor:
        all_toks = ESMIF1_TOKENS
        padding_idx = 0

    with pytest.raises(ValueError, match="padding_idx"):
        ESMIF1Alphabet.matching(_Impostor())

    with pytest.raises(ValueError, match="all_toks"):
        ESMIF1Alphabet.matching(object())


def test_alphabet_refuses_to_silently_encode_unknown_characters():
    alphabet = ESMIF1Alphabet.native()

    with pytest.raises(ValueError, match="not a native"):
        alphabet.encode("AC*D")
    with pytest.raises(ValueError, match="outside the native table"):
        alphabet.token_at(len(ESMIF1_TOKENS))


# --------------------------------------------------------------------------
# Edit space
# --------------------------------------------------------------------------

def test_space_sorts_sites_and_reads_its_own_genotype():
    space = ConstrainedEditSpace(CONTEXT, tuple(reversed(SITES)))

    assert space.positions == (2, 5, 8)
    assert space.num_sites == 3
    assert space.size == 8
    assert space.context_alleles == (0, 0, 0)
    assert space.sequence_for((0, 0, 0)) == CONTEXT
    assert space.sequence_for((1, 0, 1)) == "ACNEFGHIRL"
    assert space.alleles_for("ACNEFGHIRL") == (1, 0, 1)


def test_enumeration_is_exhaustive_and_ordered():
    space = ConstrainedEditSpace(CONTEXT, SITES)

    sequences = space.enumerate_sequences()

    assert len(sequences) == len(set(sequences)) == 8
    assert space.enumerate_alleles()[0] == (0, 0, 0)
    assert space.enumerate_alleles()[1] == (0, 0, 1)   # last site varies fastest
    assert space.enumerate_alleles()[-1] == (1, 1, 1)
    for sequence in sequences:
        assert space.validate_candidate(sequence) == sequence


def test_sixteen_site_space_round_trips_every_genotype_it_is_asked_for():
    space = ConstrainedEditSpace(CONTEXT_16, SITES_16)

    assert space.num_sites == 16
    assert space.size == 65_536
    for alleles in [(0,) * 16, (1,) * 16, tuple(i % 2 for i in range(16))]:
        assert space.alleles_for(space.sequence_for(alleles)) == alleles
    with pytest.raises(ValueError, match="above max_sequences"):
        space.enumerate_sequences(max_sequences=1024)


@pytest.mark.parametrize(
    "position, alleles, message",
    [
        (2.5, ("D", "N"), "must be an integer"),
        (-1, ("D", "N"), "non-negative"),
        (2, ("D",), "exactly two"),
        (2, ("D", "N", "S"), "exactly two"),
        (2, ("D", "D"), "duplicate alleles"),
        (2, ("D", "X"), "canonical residues"),
        (2, "DN", "not the string"),
    ],
)
def test_editable_site_rejects_an_invalid_support(position, alleles, message):
    with pytest.raises(ValueError, match=message):
        EditableSite(position, alleles)


@pytest.mark.parametrize(
    "alleles", [("A", ""), ("", "A"), ("A", "AC"), ("AC", "DE"), ("A", "AA")]
)
def test_editable_site_rejects_an_empty_or_multi_character_allele(alleles):
    """`allele in CANONICAL_RESIDUES` is substring membership, so "" and "AC"
    both pass it. Either would be written straight into the sequence, changing
    its length at that site and breaking the fixed-length contract."""
    with pytest.raises(ValueError, match="exactly one canonical residue"):
        EditableSite(0, alleles)


@pytest.mark.parametrize(
    "context, sites, message",
    [
        ("", SITES, "non-empty"),
        ("AC*EFGHIKL", SITES, "not a standard"),
        (CONTEXT, (), "at least one editable site"),
        (CONTEXT, (EditableSite(2, ("D", "N")), EditableSite(2, ("D", "S"))),
         "duplicate editable site"),
        (CONTEXT, (EditableSite(99, ("D", "N")),), "outside the context"),
        (CONTEXT, (EditableSite(0, ("D", "N")),), "must itself be a member"),
    ],
)
def test_space_rejects_an_invalid_declaration(context, sites, message):
    with pytest.raises(ValueError, match=message):
        ConstrainedEditSpace(context, sites)


@pytest.mark.parametrize(
    "candidate, message",
    [
        ("ACDEFGHIK", "does not match the context length"),
        ("ACDEFGHIKLM", "does not match the context length"),
        ("AWDEFGHIKL", "immutable position 1 changed"),
        ("ACWEFGHIKL", "outside the support"),
    ],
)
def test_space_rejects_an_invalid_candidate(candidate, message):
    space = ConstrainedEditSpace(CONTEXT, SITES)

    with pytest.raises(ValueError, match=message):
        space.alleles_for(candidate)


# --------------------------------------------------------------------------
# Prefixes and decision indices
# --------------------------------------------------------------------------

def test_prefix_starts_with_cath_and_drops_the_last_residue(policy):
    alphabet = ESMIF1Alphabet.native()
    candidate = "ACNEFGHIRL"

    prefix = policy.prefix_tokens(candidate)

    # Upstream's `tokens[:, :-1]`: column p is the decision for residue p.
    assert prefix.shape == (1, len(CONTEXT))
    assert int(prefix[0, 0]) == alphabet.index_of(PREFIX_TOKEN)
    assert prefix[0, 1:].tolist() == alphabet.encode(candidate[:-1])


def test_every_forced_residue_stays_in_the_prefix(policy):
    alphabet = ESMIF1Alphabet.native()
    editable = set(policy.space.positions)

    prefix = policy.prefix_tokens(policy.space.enumerate_sequences())

    for row, candidate in enumerate(policy.space.enumerate_sequences()):
        for position in range(len(CONTEXT) - 1):
            expected = alphabet.index_of(candidate[position])
            assert int(prefix[row, position + 1]) == expected
            if position not in editable:
                assert int(prefix[row, position + 1]) == alphabet.index_of(
                    CONTEXT[position]
                )


# --------------------------------------------------------------------------
# The probability contract
# --------------------------------------------------------------------------

def test_enumerated_probabilities_sum_to_one(policy, geometry):
    log_probs = policy.log_prob(policy.space.enumerate_sequences(), geometry)

    assert log_probs.shape == (8,)
    assert torch.logsumexp(log_probs, dim=0).abs() < 1e-12


@pytest.mark.parametrize(
    "context, sites", [(CONTEXT, SITES), (CONTEXT_16, SITES_16)]
)
def test_every_site_is_normalized_over_its_two_alleles(context, sites):
    local = _toy_policy(context, sites)
    local_geometry = local.encode_geometry(_toy_coords(len(context)))

    site_log_probs = local.site_log_probabilities(context, local_geometry)

    assert site_log_probs.shape == (1, len(sites), 2)
    assert torch.logsumexp(site_log_probs, dim=-1).abs().max() < 1e-12


def test_log_prob_is_the_sum_of_the_chosen_site_log_probabilities(policy, geometry):
    alphabet = ESMIF1Alphabet.native()
    candidates = ["ACNEFGHIRL", CONTEXT, "ACDEFSHIRL"]
    alleles = torch.tensor([policy.space.alleles_for(c) for c in candidates])

    logits = policy.native_logits(candidates, geometry)

    # Recomputed from raw logits without touching the module's own helpers.
    manual = torch.zeros(len(candidates), dtype=logits.dtype)
    for index, site in enumerate(policy.space.sites):
        allowed = [alphabet.index_of(a) for a in site.alleles]
        restricted = torch.log_softmax(logits[:, allowed, site.position], dim=-1)
        manual = manual + restricted[
            torch.arange(len(candidates)), alleles[:, index]
        ]
    assert torch.allclose(policy.log_prob(candidates, geometry), manual, atol=1e-12)


def test_forced_positions_and_disallowed_residues_contribute_nothing(policy):
    """Perturbing anything the contract says is irrelevant must change nothing:
    forced decision columns, and the 33 vocabulary rows outside each support."""
    alphabet = ESMIF1Alphabet.native()
    torch.manual_seed(0)
    logits = torch.randn(2, VOCAB, len(CONTEXT), dtype=torch.float64)
    alleles = torch.tensor([[0, 1, 0], [1, 1, 1]])
    baseline = policy.log_prob_from_logits(logits, alleles)

    perturbed = logits.clone()
    forced = [p for p in range(len(CONTEXT)) if p not in policy.space.positions]
    perturbed[:, :, forced] += 7.5
    for site in policy.space.sites:
        allowed = {alphabet.index_of(a) for a in site.alleles}
        blocked = [v for v in range(VOCAB) if v not in allowed]
        perturbed[:, blocked, site.position] += 11.0

    assert torch.equal(policy.log_prob_from_logits(perturbed, alleles), baseline)


def test_log_prob_gradient_is_onehot_minus_softmax_on_the_allowed_rows(policy):
    """One assertion pins the decision index, the support restriction,
    temperature 1, and the zero contribution of every forced position."""
    alphabet = ESMIF1Alphabet.native()
    torch.manual_seed(1)
    logits = torch.randn(
        2, VOCAB, len(CONTEXT), dtype=torch.float64, requires_grad=True
    )
    alleles = torch.tensor([[0, 1, 0], [1, 0, 1]])

    policy.log_prob_from_logits(logits, alleles).sum().backward()

    expected = torch.zeros_like(logits)
    for row in range(2):
        for index, site in enumerate(policy.space.sites):
            allowed = [alphabet.index_of(a) for a in site.alleles]
            probabilities = torch.softmax(
                logits[row, allowed, site.position].detach(), dim=0
            )
            onehot = torch.zeros(2, dtype=torch.float64)
            onehot[int(alleles[row, index])] = 1.0
            expected[row, allowed, site.position] = onehot - probabilities
    assert torch.allclose(logits.grad, expected, atol=1e-12)


def test_preference_gradient_matches_central_finite_differences(policy, geometry):
    """An actual preference objective -- the logistic loss on a margin measured
    against a detached reference -- differentiated against central differences.

    The coordinates checked are the three the analytic gradient itself says
    matter most, and each is asserted nonzero *before* it is compared. Choosing
    coordinates by hand instead lands on disallowed vocabulary rows, where the
    derivative is exactly zero and the comparison passes while testing nothing.
    """
    winner, loser = "ACNEFGHIRL", CONTEXT
    parameter = policy.model.decoder.token_effect
    with torch.no_grad():
        reference = policy.log_prob([winner, loser], geometry)
    assert not reference.requires_grad

    def loss() -> torch.Tensor:
        log_probs = policy.log_prob([winner, loser], geometry)
        margin = (log_probs[0] - reference[0]) - (log_probs[1] - reference[1])
        return -F.logsigmoid(2.0 * margin)

    analytic = torch.autograd.grad(loss(), parameter)[0]

    assert analytic.abs().max() > 1e-6
    columns = analytic.shape[1]
    largest = analytic.abs().flatten().topk(3).indices.tolist()
    epsilon = 1e-6
    for row, column in [(index // columns, index % columns) for index in largest]:
        assert abs(float(analytic[row, column])) > 1e-6
        with torch.no_grad():
            parameter[row, column] += epsilon
            plus = float(loss())
            parameter[row, column] -= 2 * epsilon
            minus = float(loss())
            parameter[row, column] += epsilon
        numeric = (plus - minus) / (2 * epsilon)
        assert abs(numeric - float(analytic[row, column])) < 1e-6


def test_preference_steps_raise_the_margin_and_keep_the_total_mass_at_one(
    policy, geometry
):
    winner, loser = "ACNEFGHIRL", CONTEXT

    def margin() -> float:
        with torch.no_grad():
            log_probs = policy.log_prob([winner, loser], geometry)
        return float(log_probs[0] - log_probs[1])

    before = margin()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        log_probs = policy.log_prob([winner, loser], geometry)
        (-(log_probs[0] - log_probs[1])).backward()
        optimizer.step()

    assert margin() > before
    with torch.no_grad():
        total = torch.logsumexp(
            policy.log_prob(policy.space.enumerate_sequences(), geometry), dim=0
        )
    assert total.abs() < 1e-12


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def test_sampler_log_probability_matches_teacher_forcing(policy, geometry):
    sample = policy.sample(geometry, num_samples=6,
                           generator=torch.Generator().manual_seed(7))

    rescored = policy.log_prob(sample.sequences, geometry)

    assert sample.log_probability.shape == (6,)
    assert torch.allclose(sample.log_probability, rescored, atol=1e-12)
    for sequence, alleles in zip(sample.sequences, sample.alleles):
        assert policy.space.alleles_for(sequence) == alleles


def test_sampler_ingests_every_prefix_token_in_order(policy, geometry):
    decoder = policy.model.decoder
    decoder.ingested_stream.clear()

    sample = policy.sample(geometry, num_samples=3,
                           generator=torch.Generator().manual_seed(1))

    last_position = policy.space.sites[-1].position
    stream = torch.stack(decoder.ingested_stream, dim=1)
    expected = policy.prefix_tokens(sample.sequences)[:, : last_position + 1]
    assert stream.shape == (3, last_position + 1)
    assert torch.equal(stream, expected)


def test_skipping_a_forced_position_changes_the_score(policy, geometry):
    """Gives the agreement tests their teeth: the upstream `continue` at a known
    position (`gvp_transformer.py:122-124`) is reproduced here and shown to
    diverge from teacher forcing on the same sequence."""
    candidate = "ACNEFGHIRL"

    honest = _incremental_log_prob(policy, geometry, candidate, skip_forced=False)
    holed = _incremental_log_prob(policy, geometry, candidate, skip_forced=True)
    teacher_forced = float(policy.log_prob(candidate, geometry)[0].detach())

    assert abs(honest - teacher_forced) < 1e-12
    assert abs(holed - teacher_forced) > 1e-3


def _incremental_log_prob(
    policy: ConstrainedEditPolicy,
    geometry: FixedGeometry,
    sequence: str,
    *,
    skip_forced: bool,
) -> float:
    """Score one candidate through the incremental path, optionally with the
    upstream cache hole."""
    space = policy.space
    last_position = space.sites[-1].position
    site_of = {p: j for j, p in enumerate(space.positions)}
    alleles = space.alleles_for(sequence)
    tokens = policy.prefix_tokens(sequence)[:, : last_position + 1]
    state: dict = {}
    total = 0.0
    policy.model.eval()
    with torch.no_grad():
        for step in range(last_position + 1):
            if skip_forced and step not in site_of:
                continue
            logits, _ = policy.model.decoder(
                tokens[:, : step + 1],
                encoder_out=geometry.decoder_encoder_out(1),
                incremental_state=state,
            )
            if step in site_of:
                index = site_of[step]
                allowed = policy.allele_token_ids[index]
                restricted = torch.log_softmax(
                    logits[:, :, -1].index_select(1, allowed), dim=-1
                )
                total += float(restricted[0, alleles[index]])
    return total


def test_sampled_sequences_preserve_every_immutable_residue(policy, geometry):
    sample = policy.sample(geometry, num_samples=32,
                           generator=torch.Generator().manual_seed(4))

    editable = set(policy.space.positions)
    for sequence in sample.sequences:
        assert len(sequence) == len(CONTEXT)
        for position, (got, fixed) in enumerate(zip(sequence, CONTEXT)):
            if position not in editable:
                assert got == fixed
        for site in policy.space.sites:
            assert sequence[site.position] in site.alleles


def test_sixteen_site_sampling_agrees_with_teacher_forcing():
    local = _toy_policy(CONTEXT_16, SITES_16)
    local_geometry = local.encode_geometry(_toy_coords(len(CONTEXT_16)))

    sample = local.sample(local_geometry, num_samples=4,
                          generator=torch.Generator().manual_seed(5))

    assert torch.allclose(
        sample.log_probability, local.log_prob(sample.sequences, local_geometry),
        atol=1e-12,
    )


def test_sampling_is_reproducible_from_a_seeded_generator(policy, geometry):
    first = policy.sample(geometry, num_samples=8,
                          generator=torch.Generator().manual_seed(12))
    second = policy.sample(geometry, num_samples=8,
                           generator=torch.Generator().manual_seed(12))

    assert first.sequences == second.sequences
    assert torch.equal(first.log_probability, second.log_probability)


def test_sampler_rejects_a_non_positive_batch(policy, geometry):
    with pytest.raises(ValueError, match="num_samples must be positive"):
        policy.sample(geometry, num_samples=0)


# --------------------------------------------------------------------------
# Gradients, dropout, and the cached geometry
# --------------------------------------------------------------------------

def test_scoring_runs_with_dropout_off_and_restores_training_flags(policy, geometry):
    policy.train()
    policy.model.decoder.dropout_module.eval()   # a deliberately mixed state

    policy.log_prob(CONTEXT, geometry)

    assert policy.model.decoder.saw_training is False
    assert policy.model.decoder.training is True
    assert policy.model.decoder.dropout_module.training is False
    assert policy.training is True


def test_encoder_is_frozen_while_the_decoder_receives_gradient(policy, geometry):
    assert all(not p.requires_grad for p in policy.model.encoder.parameters())

    policy.log_prob(CONTEXT, geometry).sum().backward()

    assert policy.model.encoder.scale.grad is None
    assert policy.model.decoder.token_effect.grad is not None
    assert policy.model.decoder.token_effect.grad.abs().sum() > 0
    # The cached geometry still feeds the decoder; only the encoder is detached.
    assert policy.model.decoder.enc_proj.weight.grad is not None


def test_geometry_is_encoded_once_and_reused(policy):
    coords = _toy_coords(len(CONTEXT))

    geometry = policy.encode_geometry(coords)
    policy.log_prob(CONTEXT, geometry)
    policy.log_prob(policy.space.enumerate_sequences(), geometry)
    policy.sample(geometry, num_samples=2, generator=torch.Generator().manual_seed(0))

    assert policy.model.encoder.calls == 1
    assert geometry.num_residues == len(CONTEXT)
    assert geometry.encoder_out.shape == (len(CONTEXT) + 2, 1, CHANNELS)
    assert not geometry.encoder_out.requires_grad
    assert len(geometry.digest) == 64
    assert policy.encode_geometry(coords).digest == geometry.digest


def test_geometry_batching_is_a_broadcast_not_a_copy(geometry):
    batched = geometry.decoder_encoder_out(5)

    assert batched["encoder_out"][0].shape[1] == 5
    assert batched["encoder_padding_mask"][0].shape[0] == 5
    assert batched["encoder_out"][0].data_ptr() == geometry.encoder_out.data_ptr()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        geometry.decoder_encoder_out(0)


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this box")
def test_policy_built_on_an_already_moved_model_scores_and_backprops():
    """`ConstrainedEditPolicy(model.cuda(), space)` is the natural call order and
    must work: site buffers built on the CPU would fail inside `index_select`
    with a message about neither the policy nor the fix."""
    model = _ToyModel(_ToyEncoder(), _ToyDecoder()).cuda()
    space = ConstrainedEditSpace(CONTEXT, SITES)

    policy = ConstrainedEditPolicy(model, space)

    assert policy.site_positions.device.type == "cuda"
    assert policy.allele_token_ids.device.type == "cuda"
    geometry = policy.encode_geometry(_toy_coords(len(CONTEXT)))
    assert geometry.device.type == "cuda"
    with torch.no_grad():
        total = torch.logsumexp(
            policy.log_prob(space.enumerate_sequences(), geometry), dim=0
        )
    assert total.abs() < 1e-12
    policy.log_prob(CONTEXT, geometry).sum().backward()
    gradient = policy.model.decoder.token_effect.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert policy.model.encoder.scale.grad is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this box")
def test_cuda_sampling_matches_teacher_forced_rescoring():
    model = _ToyModel(_ToyEncoder(), _ToyDecoder()).cuda()
    policy = ConstrainedEditPolicy(model, ConstrainedEditSpace(CONTEXT, SITES))
    geometry = policy.encode_geometry(_toy_coords(len(CONTEXT)))

    sample = policy.sample(
        geometry, num_samples=6,
        generator=torch.Generator(device="cuda").manual_seed(7),
    )

    assert sample.log_probability.device.type == "cuda"
    # The exact agreement is pinned on the CPU at 1e-12; the looser bound here
    # covers CUDA reassociating the toy decoder's running sum, nothing else.
    assert torch.allclose(
        sample.log_probability, policy.log_prob(sample.sequences, geometry),
        atol=1e-10,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this box")
def test_moving_the_whole_policy_carries_its_buffers():
    """The buffers are non-persistent, but they must still follow `.to`."""
    policy = _toy_policy()

    moved = policy.to("cuda")

    assert moved is policy
    assert policy.site_positions.device.type == "cuda"
    assert policy.allele_token_ids.device.type == "cuda"
    geometry = policy.encode_geometry(_toy_coords(len(CONTEXT)))
    assert torch.isfinite(policy.log_prob(CONTEXT, geometry)).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this box")
def test_moving_only_the_backbone_is_refused_by_name():
    """Moving `policy.model` alone leaves the buffers behind. That is a policy
    bug, so it is named as one rather than surfacing as a device error from
    somewhere inside `index_select`."""
    policy = _toy_policy()
    policy.model.cuda()

    geometry = policy.encode_geometry(_toy_coords(len(CONTEXT)))
    with pytest.raises(ValueError, match=r"policy\.to\(device\)"):
        policy.log_prob(CONTEXT, geometry)


# --------------------------------------------------------------------------
# Rejections on the policy surface
# --------------------------------------------------------------------------

def test_policy_rejects_a_model_or_space_of_the_wrong_shape():
    space = ConstrainedEditSpace(CONTEXT, SITES)

    class _NoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = _ToyDecoder()

    class _NoDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _ToyEncoder()

    # `encoder` is checked first, so a model missing both is named by that.
    with pytest.raises(TypeError, match="has no `encoder`"):
        ConstrainedEditPolicy(nn.Sequential(), space)
    with pytest.raises(TypeError, match="has no `encoder`"):
        ConstrainedEditPolicy(_NoEncoder(), space)
    with pytest.raises(TypeError, match="has no `decoder`"):
        ConstrainedEditPolicy(_NoDecoder(), space)
    with pytest.raises(TypeError, match="must be a ConstrainedEditSpace"):
        ConstrainedEditPolicy(_ToyModel(_ToyEncoder(), _ToyDecoder()), "ACDE")


def test_policy_rejects_a_mismatched_alphabet():
    model = _ToyModel(_ToyEncoder(), _ToyDecoder())
    space = ConstrainedEditSpace(CONTEXT, SITES)

    class _TruncatedAlphabet:
        all_toks = ESMIF1_TOKENS[:-1]

    with pytest.raises(ValueError, match="does not match the native"):
        ConstrainedEditPolicy(model, space, alphabet=_TruncatedAlphabet())
    # A bare token table is not an alphabet: it is rejected for the absent
    # attribute, not for its contents.
    with pytest.raises(ValueError, match="all_toks"):
        ConstrainedEditPolicy(model, space, alphabet=ESMIF1_TOKENS)


def test_policy_rejects_a_model_whose_own_dictionary_is_not_native():
    class _WrongDictionary:
        all_toks = ("A", "C")

    decoder = _ToyDecoder()
    decoder.dictionary = _WrongDictionary()

    with pytest.raises(ValueError, match="does not match the native"):
        ConstrainedEditPolicy(
            _ToyModel(_ToyEncoder(), decoder), ConstrainedEditSpace(CONTEXT, SITES)
        )


def test_scoring_rejects_an_empty_batch(policy, geometry):
    with pytest.raises(ValueError, match="empty candidate batch"):
        policy.log_prob([], geometry)


def test_scoring_rejects_a_geometry_that_is_too_short(policy):
    short = policy.encode_geometry(_toy_coords(len(CONTEXT) - 1))

    with pytest.raises(ValueError, match="never longer"):
        policy.log_prob(CONTEXT, short)


def test_scoring_accepts_a_geometry_longer_than_the_context(policy):
    """Upstream's multichain route scores a short target chain against
    complex-length coordinates. The inequality is checked; no residue mapping is
    invented."""
    longer = policy.encode_geometry(_toy_coords(len(CONTEXT) + 6))

    assert torch.isfinite(policy.log_prob(CONTEXT, longer)).all()


def test_scoring_rejects_a_geometry_of_the_wrong_dtype(policy):
    geometry = policy.encode_geometry(_toy_coords(len(CONTEXT)))
    recast = FixedGeometry(
        encoder_out=geometry.encoder_out.to(torch.float32),
        encoder_padding_mask=geometry.encoder_padding_mask,
        num_residues=geometry.num_residues,
        digest=geometry.digest,
    )

    with pytest.raises(ValueError, match="re-encode after re-casting"):
        policy.log_prob(CONTEXT, recast)


def test_scoring_rejects_something_that_is_not_a_geometry(policy):
    with pytest.raises(TypeError, match="must be a FixedGeometry"):
        policy.log_prob(CONTEXT, torch.zeros(3))


@pytest.mark.parametrize(
    "coords, confidence, message",
    [
        (torch.zeros(4, 3), None, r"shaped \(L, 3, 3\)"),
        (torch.zeros(1, 4, 3, 3), None, r"shaped \(L, 3, 3\)"),
        (torch.zeros(4, 3, 4), None, r"shaped \(L, 3, 3\)"),
        (torch.zeros(4, 3, 3), [1.0, 1.0], "coordinates cover 4 residues"),
        (torch.zeros(4, 3, 3), [[1.0]] * 4, "scalar or a 1-D sequence"),
        (torch.zeros(4, 3, 3), [1.0, 1.0, 1.0, -1.0], r"must lie in \[0, 1\]"),
        (torch.zeros(4, 3, 3), [1.0, 1.0, 1.0, 1.5], r"must lie in \[0, 1\]"),
        (torch.zeros(4, 3, 3), [1.0, 1.0, 1.0, float("nan")], "contains NaN"),
    ],
)
def test_encode_geometry_rejects_invalid_inputs(policy, coords, confidence, message):
    with pytest.raises(ValueError, match=message):
        policy.encode_geometry(coords, confidence)


def test_encode_geometry_accepts_a_scalar_confidence(policy):
    geometry = policy.encode_geometry(_toy_coords(len(CONTEXT)), 0.5)

    assert geometry.num_residues == len(CONTEXT)


def test_log_prob_from_logits_rejects_a_foreign_vocabulary(policy):
    with pytest.raises(ValueError, match="vocabulary entries"):
        policy.log_prob_from_logits(
            torch.zeros(1, 20, len(CONTEXT), dtype=torch.float64),
            torch.tensor([[0, 0, 0]]),
        )
    with pytest.raises(ValueError, match="last editable site"):
        policy.log_prob_from_logits(
            torch.zeros(1, VOCAB, 4, dtype=torch.float64), torch.tensor([[0, 0, 0]])
        )
    with pytest.raises(ValueError, match=r"shaped \(batch, 3\)"):
        policy.log_prob_from_logits(
            torch.zeros(1, VOCAB, len(CONTEXT), dtype=torch.float64),
            torch.tensor([[0, 0]]),
        )
    with pytest.raises(ValueError, match=r"must lie in \{0, 1\}"):
        policy.log_prob_from_logits(
            torch.zeros(1, VOCAB, len(CONTEXT), dtype=torch.float64),
            torch.tensor([[0, 2, 0]]),
        )


@pytest.mark.parametrize(
    "alleles, message",
    [
        ([[0.7, 0.0, 0.0]], "whole numbers"),
        (torch.tensor([[0.0, 1.5, 1.0]]), "whole numbers"),
        (torch.tensor([[-0.5, 0.0, 1.0]]), "whole numbers"),
        (torch.tensor([[0.0, float("nan"), 1.0]]), "NaN or infinity"),
        (torch.tensor([[0.0, float("inf"), 1.0]]), "NaN or infinity"),
        (torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.complex64), "complex"),
        ("010", "nested sequence"),
    ],
)
def test_log_prob_from_logits_rejects_a_genotype_that_is_not_integral(
    policy, alleles, message
):
    """`torch.as_tensor(x, dtype=torch.long)` truncates toward zero, so an
    allele index of 0.7 would score genotype 0 and return a confident, wrong
    number. Both the range check and the site count pass it."""
    logits = torch.zeros(1, VOCAB, len(CONTEXT), dtype=torch.float64)

    with pytest.raises(ValueError, match=message):
        policy.log_prob_from_logits(logits, alleles)


@pytest.mark.parametrize(
    "alleles",
    [
        [[0, 1, 0], [1, 1, 0]],
        torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.int32),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[False, True, False], [True, True, False]]),
    ],
)
def test_log_prob_from_logits_accepts_every_exactly_integral_genotype(policy, alleles):
    """The rejection above must not cost the legitimate spellings: a nested
    list, any integer dtype, an exactly-integral float, and bool."""
    torch.manual_seed(3)
    logits = torch.randn(2, VOCAB, len(CONTEXT), dtype=torch.float64)
    expected = policy.log_prob_from_logits(
        logits, torch.tensor([[0, 1, 0], [1, 1, 0]])
    )

    assert torch.equal(policy.log_prob_from_logits(logits, alleles), expected)


# --------------------------------------------------------------------------
# The base install must not need the optional stack
# --------------------------------------------------------------------------

def test_module_imports_and_scores_without_the_optional_esm_stack(monkeypatch):
    """Pinned by BLOCKING the imports rather than by relying on their absence,
    so the contract is asserted on every box instead of only on CI."""
    blocked = {"esm", "biotite", "torch_geometric", "torch_scatter"}

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in blocked:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    for cached in [n for n in sys.modules if n.split(".")[0] in blocked]:
        monkeypatch.delitem(sys.modules, cached)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
    monkeypatch.delitem(
        sys.modules, "smallAntibodyGen.models.esmif1_policy", raising=False
    )

    module = importlib.import_module("smallAntibodyGen.models.esmif1_policy")

    assert module.ESMIF1_TOKENS == ESMIF1_TOKENS
    space = module.ConstrainedEditSpace(
        CONTEXT, (module.EditableSite(2, ("D", "N")), module.EditableSite(5, ("G", "S")))
    )
    local = module.ConstrainedEditPolicy(_ToyModel(_ToyEncoder(), _ToyDecoder()), space)
    local_geometry = local.encode_geometry(_toy_coords(len(CONTEXT)))
    assert torch.logsumexp(
        local.log_prob(space.enumerate_sequences(), local_geometry), dim=0
    ).abs() < 1e-12


# --------------------------------------------------------------------------
# Optional: the real upstream modules, built from scratch, no download
# --------------------------------------------------------------------------

@pytest.fixture
def upstream_stack():
    """Install the ESM compatibility layer, then undo its process-global effects.

    `install()` writes a `torch_scatter` shim into `sys.modules` and aliases two
    biotite names. Both outlive the test that caused them, and
    `test_esmif1_compat.py::test_install_defers_to_an_importable_but_unimported_torch_scatter`
    asserts on exactly that state -- so leaving it behind makes a test in another
    file pass or fail on file order alone. Installing here rather than inside
    `_upstream_model` also means the tests below may import `esm` at the top of
    their bodies and still pass when run one at a time.
    """
    from smallAntibodyGen.esmif1_compat import install

    had_scatter = "torch_scatter" in sys.modules
    previous_scatter = sys.modules.get("torch_scatter")
    saved: list[tuple[object, str, object]] = []
    if BIOTITE_AVAILABLE:
        import biotite.structure as bs
        from biotite.structure.io import pdbx

        for owner, name in ((bs, "filter_backbone"), (pdbx, "PDBxFile")):
            saved.append((owner, name, getattr(owner, name, None)))

    install()
    try:
        yield
    finally:
        if had_scatter:
            sys.modules["torch_scatter"] = previous_scatter
        else:
            sys.modules.pop("torch_scatter", None)
        for owner, name, value in saved:
            if value is None:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, value)


def _upstream_model():
    """A ~32-dim `GVPTransformerModel` with random weights.

    Structurally identical to the released checkpoint's architecture and driven
    through the same code paths; only the weights differ, so it verifies wiring
    and never a biological result. No network access. Requires the
    `upstream_stack` fixture to have run.
    """
    from esm.data import Alphabet
    from esm.inverse_folding.gvp_transformer import GVPTransformerModel

    args = argparse.Namespace(
        dropout=0.1,
        attention_dropout=0.0,
        encoder_embed_dim=32,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_embed_dim=64,
        decoder_embed_dim=32,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_embed_dim=64,
        gvp_top_k_neighbors=6,
        gvp_num_encoder_layers=1,
        gvp_dropout=0.0,
        gvp_node_hidden_dim_scalar=16,
        gvp_node_hidden_dim_vector=8,
        gvp_edge_hidden_dim_scalar=8,
        gvp_edge_hidden_dim_vector=2,
    )
    alphabet = Alphabet.from_architecture("invariant_gvp")
    torch.manual_seed(0)
    model = GVPTransformerModel(args, alphabet).eval()
    return model, alphabet


def _upstream_coords(num_residues: int) -> torch.Tensor:
    """A plausible extended backbone: consecutive CA atoms ~3.8 A apart."""
    generator = torch.Generator().manual_seed(2)
    offsets = torch.tensor(
        [[-1.2, 0.4, 0.0], [0.0, 0.0, 0.0], [1.3, 0.5, 0.0]], dtype=torch.float32
    )
    centres = torch.arange(num_residues, dtype=torch.float32).unsqueeze(-1) * torch.tensor(
        [3.8, 0.0, 0.0]
    )
    jitter = torch.randn(
        num_residues, 3, 3, generator=generator, dtype=torch.float32
    ) * 0.1
    return centres.unsqueeze(1) + offsets.unsqueeze(0) + jitter


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_experiment_feature_path_matches_native_policy_and_has_embedding_gradients(upstream_stack):
    from types import SimpleNamespace
    from smallAntibodyGen.experiments.regularization import decoder_statistics, off_diagonal_cosine

    model, alphabet = _upstream_model()
    space = ConstrainedEditSpace(CONTEXT, SITES)
    policy = ConstrainedEditPolicy(model, space, alphabet=alphabet)
    geometry = policy.encode_geometry(_upstream_coords(len(CONTEXT)))
    bound = SimpleNamespace(policy=policy, space=space, geometry=geometry)
    sequences = space.enumerate_sequences()
    scores, z = decoder_statistics(model.decoder, bound, sequences)
    expected = policy.log_prob(sequences, geometry)
    torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-6)
    weight = model.decoder.output_projection.weight
    actual_grad, = torch.autograd.grad(scores.sum(), weight, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.sum(), weight)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)
    off_diagonal_cosine(z).backward()
    gradients = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in model.encoder.parameters())


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_local_token_table_matches_the_upstream_alphabet(upstream_stack):
    """The one place the local replica could drift from upstream."""
    from esm.data import Alphabet

    upstream = Alphabet.from_architecture("invariant_gvp")

    assert tuple(upstream.all_toks) == ESMIF1_TOKENS
    assert ESMIF1Alphabet.matching(upstream).prefix_idx == upstream.get_idx("<cath>")


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_prefix_tokens_match_the_coord_batch_converter(upstream_stack):
    from esm.inverse_folding.util import CoordBatchConverter

    model, alphabet = _upstream_model()
    space = ConstrainedEditSpace(CONTEXT, SITES)
    policy = ConstrainedEditPolicy(model, space, alphabet=alphabet)
    coords = _upstream_coords(len(CONTEXT))
    candidate = "ACNEFGHIRL"

    _, _, _, tokens, _ = CoordBatchConverter(alphabet)(
        [(coords.numpy(), None, candidate)]
    )

    assert torch.equal(policy.prefix_tokens(candidate), tokens[:, :-1])


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
@pytest.mark.parametrize("nan_atom", [None, (3, 0), (4, 1)])
def test_cached_encoding_matches_the_upstream_converter_and_encoder(
    upstream_stack, nan_atom
):
    """Pins the coordinate batching -- including the two different NaN
    conventions: a NaN N atom becomes *padding*, a NaN CA only clears
    `coord_mask`."""
    from esm.inverse_folding.util import CoordBatchConverter

    model, alphabet = _upstream_model()
    policy = ConstrainedEditPolicy(
        model, ConstrainedEditSpace(CONTEXT, SITES), alphabet=alphabet
    )
    coords = _upstream_coords(len(CONTEXT))
    if nan_atom is not None:
        coords[nan_atom[0], nan_atom[1], :] = float("nan")

    geometry = policy.encode_geometry(coords)

    batch_coords, confidence, _, _, padding_mask = CoordBatchConverter(alphabet)(
        [(coords.numpy(), None, None)]
    )
    with torch.no_grad():
        expected = model.encoder(
            batch_coords, padding_mask, confidence, return_all_hiddens=False
        )
    assert torch.equal(geometry.encoder_padding_mask, expected["encoder_padding_mask"][0])
    torch.testing.assert_close(
        geometry.encoder_out, expected["encoder_out"][0],
        rtol=1e-4, atol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cached_geometry_reproduces_an_uncached_upstream_forward(
    upstream_stack, batch_size
):
    """The parity claim: caching the encoding changes no logit.

    Compared with `assert_close`, not bit-for-bit: the tolerance covers float32
    reassociation between the two paths, nothing about the contract.
    """
    from esm.inverse_folding.util import CoordBatchConverter

    model, alphabet = _upstream_model()
    space = ConstrainedEditSpace(CONTEXT, SITES)
    policy = ConstrainedEditPolicy(model, space, alphabet=alphabet)
    coords = _upstream_coords(len(CONTEXT))
    candidates = space.enumerate_sequences()[:batch_size]

    got = policy.native_logits(candidates, policy.encode_geometry(coords))

    batch_coords, confidence, _, tokens, padding_mask = CoordBatchConverter(alphabet)(
        [(coords.numpy(), None, candidate) for candidate in candidates]
    )
    with torch.no_grad():
        expected, _ = model.forward(
            batch_coords, padding_mask, confidence, tokens[:, :-1]
        )
    assert got.shape == (batch_size, len(ESMIF1_TOKENS), len(CONTEXT))
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_upstream_mass_is_one_and_sampling_agrees_with_scoring(upstream_stack):
    model, alphabet = _upstream_model()
    space = ConstrainedEditSpace(CONTEXT, SITES)
    policy = ConstrainedEditPolicy(model, space, alphabet=alphabet)
    geometry = policy.encode_geometry(_upstream_coords(len(CONTEXT)))

    with torch.no_grad():
        total = torch.logsumexp(
            policy.log_prob(space.enumerate_sequences(), geometry), dim=0
        )
    sample = policy.sample(geometry, num_samples=4,
                           generator=torch.Generator().manual_seed(0))

    assert total.abs() < 1e-5
    torch.testing.assert_close(
        sample.log_probability, policy.log_prob(sample.sequences, geometry),
        rtol=1e-4, atol=1e-4,
    )

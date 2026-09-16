# Constrained fixed-geometry editing policy over ESM-IF1

**Date:** 2026-09-15

**Status:** implemented as `smallAntibodyGen.models.esmif1_policy`, pinned by
`src/smallAntibodyGen/tests/test_esmif1_policy.py`. This is decoder mechanics
only. No weights have been loaded, no structure has been declared, no benchmark
site has been mapped onto a residue index, and nothing has been trained. The
milestone it implements is the one named in
[the recommendation](../reference/fixed-target-posttraining-recommendation.md)
§"Exact probability contract for the finite benchmark", under the
[Decision 0003 clarification](decisions/0003-pretrained-conditioned-policy.md#scope-clarification--2026-09-14).

## The contract

For a declared fixed backbone geometry `C`, a fixed context sequence, and a set
of editable sites each carrying exactly two allowed canonical residues:

```text
log q_theta(y | C) = sum over editable sites t:
    log softmax(logits_theta(prefix_t, C)[allowed_alleles_t])[chosen_allele_t]
```

- **Native everything.** The alphabet is the 35-token `invariant_gvp` table, the
  prefix token is `<cath>`, the decision order is ESM-IF1's own autoregressive
  order, and the logits are the native decoder head's. Nothing is re-headed,
  re-tokenized, or re-ordered.
- **Forced residues stay in the prefix and contribute zero.** `prefix_t` is
  `<cath>` followed by residues `0 .. t-1` of the *candidate*, so every immutable
  residue before `t` is present and conditions the decision. It is never itself a
  term in the sum: under this process its probability is one.
- **Support and temperature.** Each site normalizes over its own two allele
  tokens at temperature 1. There is no nucleus, top-k, or rejection knob to set,
  so a policy and a frozen reference cannot silently diverge in their process.
- **Allele index order is the caller's.** `EditableSite(position, (a, b))` makes
  `a` index 0 and `b` index 1. No germline/somatic convention is imposed.

`log q` is a **normalized constrained editing process**. It is neither the native
sequence likelihood nor the native model conditioned probabilistically on the
fixed residues, and values are comparable only within one `ConstrainedEditSpace`
and one geometry. Enumerated over the whole space it sums to one; that is what
`test_enumerated_probabilities_sum_to_one` asserts.

## API

| Object | Role |
|---|---|
| `EditableSite(position, alleles)` | One editable position, exactly two single-character canonical residues |
| `ConstrainedEditSpace(context, sites)` | The fixed context plus its sites; `sequence_for`, `alleles_for`, `enumerate_sequences` |
| `FixedGeometry` | A detached, reusable encoding of one declared geometry |
| `ConstrainedEditPolicy(model, space, alphabet=None)` | `encode_geometry`, `prefix_tokens`, `native_logits`, `site_log_probabilities`, `log_prob`, `log_prob_from_logits`, `sample` |
| `ConstrainedSample` | Sampled sequences, genotypes, and the summed action log-probability |
| `ESMIF1Alphabet` | The native token table, replicated locally; `matching` refuses any other |

Site counts are not fixed: the toy tests run 3 sites, the 16-site path is
exercised by `test_sixteen_site_space_round_trips_every_genotype_it_is_asked_for`
and `test_sixteen_site_sampling_agrees_with_teacher_forcing`. Enumeration is
guarded by `max_sequences`, because a 16-site space is 65,536 candidates.

```python
from smallAntibodyGen.models.esmif1_policy import (
    ConstrainedEditPolicy, ConstrainedEditSpace, EditableSite,
)

space = ConstrainedEditSpace(
    context="ACDEFGHIKL",                       # the parent, a member of the space
    sites=(EditableSite(2, ("D", "N")), EditableSite(5, ("G", "S"))),
)
policy = ConstrainedEditPolicy(model, space, alphabet=alphabet)   # GVPTransformerModel
geometry = policy.encode_geometry(coords)       # (L, 3, 3) N/CA/C, encoded once

log_q = policy.log_prob(["ACNEFGHIKL", "ACDEFSHIKL"], geometry)   # (2,), differentiable
log_q.sum().backward()                                            # decoder receives gradient

sample = policy.sample(geometry, num_samples=8)
assert torch.allclose(sample.log_probability, policy.log_prob(sample.sequences, geometry))
```

## Semantics that are easy to get wrong

**Decision index.** Upstream scores `prev = tokens[:, :-1]` against
`target = tokens[:, 1:]` (`esm/inverse_folding/util.py:113-118`) and returns
`B x V x T` (`transformer_decoder.py:125`). `append_eos` is False for
`invariant_gvp`, so the decision for residue `p` is column `p`. An off-by-one
here still produces a well-formed normalized distribution over the right support,
which is why `test_prefix_tokens_match_the_coord_batch_converter` compares
against `CoordBatchConverter` directly.

**Caching the encoding is exact, not an approximation.** The encoder consumes
coordinates, a padding mask, and confidence; its token input is a constant
`<mask>`/`<pad>` pattern derived from the padding mask
(`gvp_transformer_encoder.py:83-87`). It never sees the sequence, so one detached
encoding reused across candidates is identical to re-encoding per candidate.
`test_cached_geometry_reproduces_an_uncached_upstream_forward` checks that
against a full uncached `model.forward` on the real upstream model.

**Differentiability.** Scoring runs with dropout off (every submodule's training
flag restored afterwards) and **without** `torch.no_grad`. The encoder is frozen
by `requires_grad_(False)` on its parameters and its output is detached at the
cache boundary; the decoder, including its cross-attention onto the cached
states, stays trainable. `sample` is the one path under `no_grad`, and returns a
detached log-probability: the intended training shape is sample-then-rescore.

**The sampler processes every forced token.** Under `incremental_state` the
decoder ingests only the last token (`transformer_decoder.py:162-165`) and the
attention module appends exactly that one key/value
(`esm/multihead_attention.py:286-323`). Upstream's `sample` skips the decoder
call at an already-known position (`gvp_transformer.py:122-124`), so the token
before it is never ingested and later steps attend over a key/value set with
holes. `ConstrainedEditPolicy.sample` calls the decoder at every prefix position
and only *samples* at editable ones.
`test_skipping_a_forced_position_changes_the_score` reproduces the upstream skip
and shows it diverging from teacher forcing, which is what gives the agreement
tests their teeth.

**Rejections.** Invalid site positions and supports (`EditableSite`); a context
that is not itself a member of the space, duplicate sites, out-of-range sites,
and non-standard context tokens (`ConstrainedEditSpace`); a candidate of the
wrong length, one that changed an immutable residue, or one holding a residue
outside a site's support (`alleles_for`); a non-native alphabet or model
dictionary; a geometry shorter than the context, or cached on the wrong
device/dtype; a genotype that is not an exact integer in `{0, 1}`. Each is
refused rather than coerced. The policy's site buffers are built on the
backbone's device, and moving `policy.model` alone — leaving them behind — is
named as a policy error rather than surfacing from inside `index_select`.

## Limitations

- **Fixed geometry is an approximation.** One encoding is reused for every
  candidate. That is exact with respect to the encoder's inputs; it is *not*
  evidence that each variant adopts that conformation.
- **No structure, no mapping, no weights.** Coordinates are an argument.
  Benchmark residue numbering, chain selection, and PDB numbering are not
  implemented here and are not implied by `EditableSite.position`, which is a
  plain 0-based index into the context string.
- **Missing-coordinate convention is undeclared.** Upstream derives the padding
  mask from the N atom's x coordinate alone, so a NaN N becomes padding while a
  NaN CA only clears `coord_mask`. `encode_geometry` reproduces both branches and
  documents them; it does not decide which one "missing residue" should mean.
- **Geometry longer than the context is permitted** (upstream's multichain route
  scores a short chain against complex-length coordinates) but no residue
  correspondence between the two is checked, because none is declared.
- **Single chain in, single chain decoded.** Multichain packing is the caller's
  responsibility.
- **`FixedGeometry` is a process-local cache, not a portable artifact.** It
  carries a content digest but no provenance for weights, structure, or chain
  selection, and it must be re-encoded after any move, re-cast, or encoder
  change.
- **Not run against released weights.** The optional tests build a ~32-dim
  `GVPTransformerModel` with random weights, so they verify wiring and never a
  biological result. Weight loading on this box is a separate, earlier record
  ([training-box evidence](../reference/evidence/esm-if1-training-box-2026-09-14.json));
  nothing in this milestone re-ran it.

## Pending gates

These remain open and none of them is addressed here:

1. **Declared structural input:** pin the structure, chain selection, residue
   mapping, missing-coordinate convention, and the relation between the
   structural antigen and the measured H1 target.
2. **Artifact pinning:** exact source/weight hashes for
   `esm_if1_gvp4_t16_142M_UR50`, plus upstream score parity against the released
   checkpoint rather than a randomly initialized model.
3. **Device and throughput:** measured memory and throughput for a backward pass
   at the real sequence length on the chosen training device.
4. **Reference policy and objective:** a frozen reference copy, the DPO/SFT
   losses themselves, and the four-arm training plan. This module supplies
   `log q` for both sides of `beta * log(q / q_reference)` and nothing else.
5. **Development-side headroom gate** before committing to the training budget.

Promotion requires all of the above. Implementing the probability contract is not
checkpoint promotion.

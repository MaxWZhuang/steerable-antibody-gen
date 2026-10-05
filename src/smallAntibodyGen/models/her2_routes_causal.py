"""Route 2: contextual autoregressive infilling on the pinned p-IgGen backbone.

``her2_policy.CorePolicy`` reads a 99-token prefix -- start plus ``VH[:98]`` --
and predicts ten residues. That interface cannot express this campaign's task: it
omits FR4, the entire light chain, and any notion of *which* site is being
predicted. This module is the paired-context replacement. It keeps everything
``CorePolicy`` got right (strict native load, dropout zero, the 20-way
renormalised categorical, ranking by the mean rather than the sum, no sentinel
able to terminate a decode) and adds three things:

* **Two real tokens, at the next free ids.** ``[MASK]`` = 26 marks a site whose
  residue is hidden in the visible context; ``[SEP]`` = 27 separates the context
  from the emission block. Both embedding rows and both output rows are
  **zero-initialised**, and neither id is ever in the sampling support, so neither
  can be emitted. ``eos_token_id: 2`` is arginine in this release and decoding
  runs a fixed number of steps, so an internal R still cannot stop it.
* **A zero-initialised query pointer.** At emission step ``t`` the input is the
  residue emitted at ``t-1`` (or the separator) plus ``pointer(serialised position
  of the site being predicted now)``. A pointer beats a marker token on a specific
  failure mode: a marker in the vocabulary is emittable by the output head and
  would need masking out, whereas a pointer cannot be emitted at all. It is also
  a function of coordinates only, which is what makes the no-leakage test in
  :mod:`test_her2_routes_alignment` a real test rather than an inspection.
* **Zero-initialised condition tables**, the same three-table product Route 1
  uses, so "quality conditioning helped" cannot mean "one route had a richer
  conditioning vocabulary".

Because every addition is zero-initialised, the model at construction reproduces
the released backbone's conditional distribution over the twenty canonical
residues **exactly in mathematics** -- the two new output columns are excluded
from the renormalisation -- and to within the repository's own measured fp32
tolerance numerically. :meth:`ConditionedCausalInfiller.native_parity` measures
it with ``her2_policy.compare_sum_log_probabilities`` and records the error.
"""
from __future__ import annotations

import numpy as np
import torch

from ..experiments.her2_data import CANONICAL
from ..experiments.her2_policy import (
    SUM_LOG_PROBABILITY_ATOL,
    SUM_LOG_PROBABILITY_RTOL,
    compare_sum_log_probabilities,
    load_pinned_model,
    load_vocab,
)
from ..experiments.her2_routes_serial import (
    PIGGEN_EXTENDED_VOCAB_SIZE,
    PIGGEN_MASK_ID,
    PIGGEN_SEP_ID,
    PIGGEN_VOCAB_SIZE,
    bind_piggen_vocabulary,
    causal_serialization,
)
from ..experiments.her2_runtime import require

#: Released backbone parameter count, from ``her2_policy.PIGGEN_PARAMETERS``.
PIGGEN_PARAMETERS = 22097408


def grow_vocabulary(model, *, target=PIGGEN_EXTENDED_VOCAB_SIZE):
    """Extend ``embed_in`` and ``embed_out`` to ``target`` rows, copying exactly.

    The new rows are zeros in both tables. Every original row is copied with
    ``torch.equal`` verified afterwards, because "the checkpoint loaded and the
    logits look plausible" is not evidence that row 7 is still tyrosine.

    Returns the record I1 asserts against.
    """
    hidden = int(model.config.hidden_size)
    old = int(model.config.vocab_size)
    require(old == PIGGEN_VOCAB_SIZE,
            f"Expected the released {PIGGEN_VOCAB_SIZE}-entry vocabulary, found {old}")
    require(int(target) > old, f"Cannot shrink the vocabulary from {old} to {target}")

    embed_in = model.gpt_neox.embed_in
    grown_in = torch.nn.Embedding(int(target), hidden)
    with torch.no_grad():
        torch.nn.init.zeros_(grown_in.weight)
        grown_in.weight[:old].copy_(embed_in.weight)
    grown_in = grown_in.to(embed_in.weight.device, dtype=embed_in.weight.dtype)

    head = model.embed_out
    require(head.bias is None, "The released head has no bias; a bias would need copying too")
    grown_out = torch.nn.Linear(hidden, int(target), bias=False)
    with torch.no_grad():
        torch.nn.init.zeros_(grown_out.weight)
        grown_out.weight[:old].copy_(head.weight)
    grown_out = grown_out.to(head.weight.device, dtype=head.weight.dtype)

    exact = (bool(torch.equal(grown_in.weight[:old].cpu(), embed_in.weight.detach().cpu()))
             and bool(torch.equal(grown_out.weight[:old].cpu(), head.weight.detach().cpu())))
    require(exact,
            "A copied vocabulary row differs from the released one. This is not a tolerance "
            "question: the rows are copied, so they are equal or the copy is wrong.")
    # The untouched released tables are retained OUTSIDE the module tree (a plain
    # dict, so nn.Module does not register them and the parameter count is
    # unaffected) purely so ``native_token_parity`` can run the real pre-growth
    # pathway rather than a slice of the post-growth one.
    original = {"embed_in": embed_in, "embed_out": head}
    model.gpt_neox.embed_in = grown_in
    model.embed_out = grown_out
    model.config.vocab_size = int(target)
    return {"old_vocab_size": old, "new_vocab_size": int(target),
            "original_tables": original,
            "rows_copied_exactly": True,
            "new_rows": {"mask": PIGGEN_MASK_ID, "separator": PIGGEN_SEP_ID},
            "new_rows_initialisation": "zeros in both the input embedding and the output head",
            "added_parameters": 2 * (int(target) - old) * hidden,
            "tie_note": ("GPTNeoX keeps embed_in and embed_out as separate tensors in this "
                         "release, so both are grown and both are verified. A tied head would "
                         "need a single growth and a different assertion.")}


class ConditionedCausalInfiller(torch.nn.Module):
    """The paired-context AR infiller: pinned trunk, pointer, conditions, fixed steps."""

    def __init__(self, model, vocabulary, condition_spec, *, max_positions=400,
                 native_parameters=PIGGEN_PARAMETERS):
        super().__init__()
        config = model.config
        require(float(getattr(config, "hidden_dropout", 0.0)) == 0.0
                and float(getattr(config, "attention_dropout", 0.0)) == 0.0,
                "This route trains and scores at dropout zero, so a prediction agrees alone and "
                "in a padded batch (I8). A nonzero dropout breaks that property silently.")
        require(int(config.max_position_embeddings) == int(max_positions),
                f"The released position budget is {config.max_position_embeddings} and this model "
                f"was constructed for {max_positions}")
        self.model = model
        self.vocabulary = dict(vocabulary)
        self.condition_spec = condition_spec
        self.max_positions = int(max_positions)
        hidden = int(config.hidden_size)
        sizes = condition_spec.sizes
        self.pointer = torch.nn.Embedding(self.max_positions, hidden)
        self.condition_family = torch.nn.Embedding(sizes["family"], hidden)
        self.condition_bin = torch.nn.Embedding(sizes["bin"], hidden)
        self.condition_assay = torch.nn.Embedding(sizes["assay"], hidden)
        for table in (self.pointer, self.condition_family, self.condition_bin,
                      self.condition_assay):
            torch.nn.init.zeros_(table.weight)
        self.register_buffer("canonical_ids", torch.tensor(
            [int(self.vocabulary[residue]) for residue in CANONICAL], dtype=torch.long),
            persistent=False)
        self.native_parameters = int(native_parameters)
        self._added = (2 * (int(config.vocab_size) - PIGGEN_VOCAB_SIZE) * hidden
                       + sum(int(t.weight.numel()) for t in
                             (self.pointer, self.condition_family, self.condition_bin,
                              self.condition_assay)))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_release(cls, raw_root, condition_spec, *, device="cpu",
                     attn_implementation="sdpa"):
        """Strict-load the release, grow the vocabulary, bind the residue table.

        The pointer and condition tables are constructed on the default device, so
        moving only the trunk leaves them on the CPU and the first ``index_select``
        raises -- on the CUDA path only, which is exactly where a test that never
        ran on a GPU would miss it. The whole wrapper is moved and the placement is
        asserted.
        """
        model = load_pinned_model(raw_root, device=device,
                                  attn_implementation=attn_implementation)
        vocabulary = bind_piggen_vocabulary(load_vocab(raw_root))
        growth = grow_vocabulary(model)
        infiller = cls(model, vocabulary, condition_spec)
        infiller.native_tables = {key: value.to(device)
                                  for key, value in growth.pop("original_tables").items()}
        infiller.growth = growth
        infiller = infiller.to(device)
        infiller.assert_device_consistency(label=f"from_release(device={device!r})")
        return infiller

    def assert_device_consistency(self, *, label="ConditionedCausalInfiller"):
        """Every parameter and buffer on one device. Measured, not assumed."""
        devices = sorted({str(p.device) for p in self.parameters()}
                         | {str(b.device) for b in self.buffers()})
        require(len(devices) == 1,
                f"{label}: parameters and buffers span {devices}. A trunk on CUDA with the "
                "pointer and condition tables on CPU fails at the first index_select.")
        stray = sorted(name for name, table in dict(getattr(self, "native_tables", {})).items()
                       if str(table.weight.device) != devices[0])
        require(not stray,
                f"{label}: the retained pre-growth table(s) {stray} are on another device than "
                "the model, so the native-parity probe would compare across devices")
        return {"device": devices[0]}

    def parameter_report(self):
        """Native, added and total counts, computed from the modules that exist."""
        total = int(sum(p.numel() for p in self.parameters()))
        expected = self.native_parameters + self._added
        require(total == expected,
                f"The infiller holds {total} parameters and the released trunk plus the declared "
                f"additions is {expected}. An unexpected count means a table was shared, "
                "duplicated or silently resized, and the comparison it feeds is not the declared "
                "architecture.")
        return {"native": self.native_parameters, "added": int(self._added), "total": total,
                "breakdown": {"vocabulary_rows": int(
                    2 * (int(self.model.config.vocab_size) - PIGGEN_VOCAB_SIZE)
                    * int(self.model.config.hidden_size)),
                    "pointer": int(self.pointer.weight.numel()),
                    "conditions": int(self.condition_family.weight.numel()
                                      + self.condition_bin.weight.numel()
                                      + self.condition_assay.weight.numel())},
                "all_additions_zero_initialised_at_construction": True}

    # -- forward -----------------------------------------------------------
    @property
    def device(self):
        return self.pointer.weight.device

    def condition_vector(self, condition_indices):
        indices = torch.as_tensor(condition_indices, dtype=torch.long, device=self.device)
        require(indices.dim() == 2 and indices.shape[1] == 3,
                f"Condition indices must be (B, 3), got {tuple(indices.shape)}")
        return (self.condition_family(indices[:, 0]) + self.condition_bin(indices[:, 1])
                + self.condition_assay(indices[:, 2])).unsqueeze(1)

    def input_embeddings(self, tokens, *, query_slots=None, pointers=None,
                         condition_indices=None):
        """``inputs_embeds`` with the pointer scattered onto the query slots.

        The trunk is then called through its own ``forward(inputs_embeds=...)``, so
        no part of the released attention, rotary or MLP path is reimplemented
        here. The pointer lands *only* on the query slots: adding it to the context
        positions would make the visible background depend on the emission order.
        """
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        require(tokens.dim() == 2, f"Tokens must be (B, T), got {tuple(tokens.shape)}")
        embeddings = self.model.gpt_neox.embed_in(tokens)
        if condition_indices is not None:
            embeddings = embeddings + self.condition_vector(condition_indices)
        if pointers is not None:
            require(query_slots is not None, "Pointers need the query slots they attach to")
            slots = torch.as_tensor(query_slots, dtype=torch.long, device=self.device)
            values = torch.as_tensor(pointers, dtype=torch.long, device=self.device)
            require(slots.shape == values.shape,
                    f"{tuple(slots.shape)} query slots and {tuple(values.shape)} pointers")
            require(bool((values < self.max_positions).all()),
                    "A pointer names a serialised position outside the model's position budget")
            embeddings = embeddings.clone()
            rows = torch.arange(tokens.shape[0], device=self.device).unsqueeze(1)
            embeddings[rows, slots] = embeddings[rows, slots] + self.pointer(values)
        return embeddings

    def forward(self, tokens, *, query_slots=None, pointers=None, condition_indices=None,
                attention_mask=None):
        embeddings = self.input_embeddings(tokens, query_slots=query_slots, pointers=pointers,
                                           condition_indices=condition_indices)
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask, device=embeddings.device)
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask,
                          use_cache=False).logits

    def representation(self, tokens, *, attention_mask=None, condition_indices=None,
                       query_slots=None, pointers=None):
        """``(B, hidden)`` mean-pooled final-layer state. The head's input.

        Captured with a temporary forward hook on the last GPTNeoX layer, for the
        same reason as the masked route: the hook reads whatever the layer emits
        and cannot be wrong about the module's return signature, and the two routes
        then supply the head with the same kind of object.

        Pooled over the real tokens only. The AR serialisation is right-padded with
        the mask id, so averaging the padding would make a head's prediction depend
        on the longest construct in its batch.
        """
        captured = {}

        def hook(_module, _args, output):
            captured["state"] = output[0] if isinstance(output, tuple) else output

        handle = self.model.gpt_neox.layers[-1].register_forward_hook(hook)
        try:
            self.forward(tokens, query_slots=query_slots, pointers=pointers,
                         condition_indices=condition_indices, attention_mask=attention_mask)
        finally:
            handle.remove()
        require("state" in captured,
                "The last GPTNeoX layer emitted nothing during the forward, so there is no "
                "representation to pool. The layer layout changed and the head would otherwise "
                "train on whatever was in the buffer.")
        state = captured["state"]
        require(state.dim() == 3,
                f"Expected (B, T, hidden) from the final layer, got {tuple(state.shape)}")
        if attention_mask is None:
            return state.mean(dim=1)
        weights = torch.as_tensor(attention_mask, dtype=state.dtype,
                                  device=state.device).unsqueeze(-1)
        return (state * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    def canonical_log_probs(self, tokens, query_slots, *, pointers=None,
                            condition_indices=None, attention_mask=None):
        """``(B, k, 20)`` log probabilities of the canonical residues at each query.

        Renormalised over the twenty canonical residues, which is what excludes
        ``[MASK]``, ``[SEP]``, ``<PAD>``, ``B``, ``Z`` and the three sentinels from
        the support identically in training, scoring and sampling.
        """
        logits = self.forward(tokens, query_slots=query_slots, pointers=pointers,
                              condition_indices=condition_indices,
                              attention_mask=attention_mask)
        slots = torch.as_tensor(query_slots, dtype=torch.long, device=logits.device)
        gathered = logits.gather(1, slots.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
        return torch.log_softmax(gathered[..., self.canonical_ids].float(), dim=-1)

    def window_log_probs(self, batch):
        """``(B, k)`` log probability of each realised residue under the declared order.

        ``batch`` is a :class:`~smallAntibodyGen.experiments.her2_routes_data.
        CausalBatch`. One causal forward produces every conditional, because
        position ``context_length + t`` attends only to the context and to
        ``r_{<t}``. That is exact, not an approximation, and I3 asserts the
        alignment with a synthetic unique-label sequence.
        """
        log_probs = self.canonical_log_probs(
            batch.tokens, batch.query_slots, pointers=batch.pointers,
            condition_indices=batch.condition_indices,
            attention_mask=batch.attention_mask)
        targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
        require(targets.shape == log_probs.shape[:2],
                f"{tuple(targets.shape)} targets against {tuple(log_probs.shape[:2])} queries")
        picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        if batch.target_mask is None:
            return picked
        mask = torch.as_tensor(batch.target_mask, dtype=picked.dtype, device=picked.device)
        return picked * mask

    # -- decoding ----------------------------------------------------------
    @torch.no_grad()
    def fill(self, construct, *, order, condition, rng, temperature=1.0):
        """Fill an edit map one site at a time in ``order``. Exactly ``k`` steps.

        Returns ``(residues, per-step log probabilities at temperature 1)``. The
        recorded log probability is always the temperature-1 density, so a draw can
        be rescored and must agree.

        **Every pointer emitted so far is re-injected at every step.** Teacher
        forcing scatters a pointer onto all ``k`` query slots in one forward, so a
        decoder that injected only the newest one would be sampling from a
        different conditional than the one it was trained on and the one it is
        rescored under. At random pointer weights (std 0.15, seed 919) that gap
        measured 1.47135 nats per step -- invisible at zero initialisation, which
        is why the parity test has to run with pointers that have moved.
        """
        require(float(temperature) > 0, "Temperature must be positive")
        indices = [self.condition_spec.encode(condition)]
        serialization = causal_serialization(construct, order=order,
                                             max_positions=self.max_positions)
        tokens = list(serialization.tokens)
        slots, pointers = [], []
        emitted, per_step = [], []
        for step in range(len(order)):
            slots.append(serialization.context_length + step)
            pointers.append(int(serialization.pointers[step]))
            log_probs = self.canonical_log_probs(
                [tokens], [slots], pointers=[pointers], condition_indices=indices)[0, -1]
            scaled = torch.softmax(log_probs / float(temperature), dim=-1)
            draw = int(torch.multinomial(scaled, 1, generator=rng).item())
            per_step.append(float(log_probs[draw]))
            emitted.append(CANONICAL[draw])
            tokens.append(int(self.vocabulary[CANONICAL[draw]]))
        return "".join(emitted), np.asarray(per_step, dtype=np.float64)

    @torch.no_grad()
    def step_score_parity(self, construct, *, order, condition, residues, label="I3"):
        """Per-step decode scores must equal the teacher-forced ones. Measured.

        Replays ``residues`` through the incremental decode path and compares
        against the single teacher-forced forward, position by position. Run it
        with NONZERO pointer weights: at zero initialisation both paths agree
        trivially and the check is vacuous.
        """
        indices = [self.condition_spec.encode(condition)]
        serialization = causal_serialization(construct, order=order, residues=residues,
                                             max_positions=self.max_positions)
        forced = self.canonical_log_probs(
            [serialization.tokens], [serialization.query_slots],
            pointers=[serialization.pointers], condition_indices=indices)[0]
        tokens = list(serialization.tokens[:serialization.context_length + 1])
        slots, pointers, incremental = [], [], []
        for step, residue in enumerate(residues):
            slots.append(serialization.context_length + step)
            pointers.append(int(serialization.pointers[step]))
            row = self.canonical_log_probs([tokens], [slots], pointers=[pointers],
                                           condition_indices=indices)[0, -1]
            incremental.append(row)
            tokens.append(int(self.vocabulary[residue]))
        stacked = torch.stack(incremental)
        difference = (stacked.double() - forced.double()).abs()
        pointer_moved = bool(float(self.pointer.weight.abs().sum()) > 0)
        return {"label": label, "steps": int(stacked.shape[0]),
                "max_abs_difference": float(difference.max()),
                "per_step_max": [float(value) for value in difference.max(dim=-1).values],
                "pointer_table_nonzero": pointer_moved,
                "meaningful": pointer_moved,
                "claim": ("the incremental decode and the single teacher-forced forward are the "
                          "same conditional. They are only the same if every prefix pointer is "
                          "re-injected at every step.")}

    # -- parity ------------------------------------------------------------
    @torch.no_grad()
    def native_parity(self, reference_policy, cores, *, scaffold, label="route2_native_parity"):
        """Compare this model's core log probabilities against ``CorePolicy``'s.

        The comparison is the *scientific* quantity: the sum over the ten HER2 core
        positions of the 20-way renormalised conditional, computed by the released
        prefix-only interface and by this paired-context one at zero-initialised
        additions. They are not the same computation -- this one sees FR4 and the
        light chain -- so the two numbers are recorded and the **difference is
        reported as a measurement**, not asserted to be zero.

        The parity claim I1 needs is :meth:`native_token_parity`, which swaps the
        untouched released tables back in and compares the old-token logits
        directly. :meth:`vocabulary_invariance` is the weaker companion check on
        the renormalisation.
        """
        reference = reference_policy.score(np.asarray(cores))["sum_log_probability"]
        from ..experiments.her2_routes_serial import her2_construct
        mine = []
        for row, core in enumerate(cores):
            construct = her2_construct(scaffold, "".join(CANONICAL[int(i)] for i in core)
                                       if not isinstance(core, str) else core,
                                       record_id=f"parity{row}")
            serialization = causal_serialization(
                construct, residues=construct.residues(construct.sites),
                max_positions=self.max_positions)
            log_probs = self.canonical_log_probs(
                [serialization.tokens], [serialization.query_slots],
                pointers=[serialization.pointers])[0]
            targets = torch.as_tensor(serialization.targets, dtype=torch.long,
                                      device=log_probs.device)
            mine.append(float(log_probs.gather(1, targets.unsqueeze(-1)).sum()))
        difference = np.asarray(mine) - np.asarray(reference, dtype=np.float64)
        return {"rows": int(len(mine)),
                "prefix_only_mean": float(np.mean(reference)),
                "paired_context_mean": float(np.mean(mine)),
                "mean_difference": float(np.mean(difference)),
                "max_abs_difference": float(np.max(np.abs(difference))),
                "label": label,
                "interpretation": ("these are two DIFFERENT conditionals -- the released interface "
                                   "reads start+VH[:98] and this one reads the whole paired "
                                   "construct with the window masked -- so the difference is a "
                                   "measurement of how much the added context changes the released "
                                   "backbone's belief, not a parity failure. The parity claim this "
                                   "campaign asserts is vocabulary_invariance below.")}

    @torch.no_grad()
    def native_token_parity(self, tokens, *, tolerance=0.0, label="I1"):
        """The released 26-vocabulary pathway, before and after the 26->28 growth.

        This is the parity claim I1 actually asks for, and it is a different
        comparison from both :meth:`native_parity` (two different conditionals) and
        :meth:`vocabulary_invariance` (two slices of the *same* grown trunk). Here
        the **untouched released tables** are swapped back in, the same old-token
        input is run through both, and the first 26 logit columns are compared.

        The rows are copied, the two appended rows are zero, and every input token
        is below 26, so the two computations are the same arithmetic in a different
        matrix shape; the measured difference on the pinned release is 0. The
        conditioning and pointer additions are not applied on either side -- this
        probe is about the vocabulary growth alone.
        """
        require(getattr(self, "native_tables", None),
                "The pre-growth tables were not retained, so this probe cannot run the real "
                "native pathway. Construct through from_release, which keeps them.")
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        require(int(ids.max()) < PIGGEN_VOCAB_SIZE,
                f"This probe compares the OLD-token pathway; token id {int(ids.max())} is not in "
                f"the released {PIGGEN_VOCAB_SIZE}-entry vocabulary")
        grown_in, grown_out = self.model.gpt_neox.embed_in, self.model.embed_out
        after = self.model(input_ids=ids, use_cache=False).logits[..., :PIGGEN_VOCAB_SIZE]
        try:
            self.model.gpt_neox.embed_in = self.native_tables["embed_in"]
            self.model.embed_out = self.native_tables["embed_out"]
            before = self.model(input_ids=ids, use_cache=False).logits
        finally:
            self.model.gpt_neox.embed_in = grown_in
            self.model.embed_out = grown_out
        require(before.shape[-1] == PIGGEN_VOCAB_SIZE,
                f"The retained native head emits {before.shape[-1]} columns, not "
                f"{PIGGEN_VOCAB_SIZE}")
        difference = float((after.double() - before.double()).abs().max())
        require(difference <= float(tolerance),
                f"{label}: the old-token logits moved by {difference} when the vocabulary grew "
                f"from {PIGGEN_VOCAB_SIZE} to {int(self.model.config.vocab_size)}. The rows are "
                "copied and the new rows are zero, so a nonzero difference is a copy error or a "
                "BLAS re-tiling; either way it is investigated, not widened.")
        return {"label": label, "max_abs_difference": difference,
                "tolerance": float(tolerance), "bit_identical": difference == 0.0,
                "old_vocab_size": PIGGEN_VOCAB_SIZE,
                "new_vocab_size": int(self.model.config.vocab_size),
                "compares": ("the untouched released embedding and head against the grown ones "
                             "on the same old-token input. Not a slice of one model.")}

    @torch.no_grad()
    def vocabulary_invariance(self, tokens, query_slots, *, pointers=None, label="I1"):
        """Assert the two appended columns cannot change the 20-way conditional.

        Computed twice: once over the grown 28-column head and once with the head
        sliced back to its original 26 columns. Both are renormalised over the same
        twenty canonical residues, so they are mathematically identical and differ
        only by fp32 rounding in a differently shaped GEMM -- which is exactly why
        this is compared with the repository's own measured sum-log-probability
        tolerance instead of asserted bit-equal.

        This compares two slices of ONE trunk and is therefore silent about the
        growth itself; :meth:`native_token_parity` is the probe that runs the real
        pre-growth pathway.
        """
        grown = self.canonical_log_probs(tokens, query_slots, pointers=pointers)
        head = self.model.embed_out
        original = torch.nn.Linear(head.in_features, PIGGEN_VOCAB_SIZE, bias=False)
        original = original.to(head.weight.device, dtype=head.weight.dtype)
        with torch.no_grad():
            original.weight.copy_(head.weight[:PIGGEN_VOCAB_SIZE])
        try:
            self.model.embed_out = original
            sliced = self.canonical_log_probs(tokens, query_slots, pointers=pointers)
        finally:
            self.model.embed_out = head
        report = compare_sum_log_probabilities(
            grown.sum(dim=1).flatten().double().cpu().numpy(),
            sliced.sum(dim=1).flatten().double().cpu().numpy(),
            label=f"{label}: grown vs original-width head",
            atol=SUM_LOG_PROBABILITY_ATOL, rtol=SUM_LOG_PROBABILITY_RTOL)
        return dict(report, claim=("the appended mask and separator columns are excluded from the "
                                   "renormalisation, so they cannot move the distribution. The "
                                   "residual above is fp32 rounding in a 28-column versus "
                                   "26-column matmul."))

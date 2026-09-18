"""p-IgGen core policy: strict native load, shared-prefix scoring, sampling, SFT.

The policy is the pinned p-IgGen causal LM (`GPTNeoXForCausalLM`, 4 layers, 8
heads, hidden 768) reading a **fixed 99-token prefix** -- the start token plus
VH[:98], ending ...YYCSR -- and predicting the ten editable HCDR3 core residues.
There is no antigen encoder, no FR4 and no light chain in the context, and no
claim that one is there.

Four properties are enforced here rather than left to convention:

* **The prefix cache is differentiable and single-use.** ``batch_repeat_interleave``
  mutates the cache in place and the second forward *appends* the core positions
  to that same object (length 99 -> 108). So the cache is built inside each call
  and never returned, never hoisted, never reused, and never detached during
  training. Reusing it is the entire bug class this design removes; a probe
  measured the cached path against ordinary full teacher forcing at 6.2e-6
  (logits) and 3.5e-6 (all-parameter gradients), and :meth:`CorePolicy.full_logits`
  keeps that comparison available forever.
* **Dropout must be zero.** A shared prefix applies *one* dropout mask across the
  whole batch where full teacher forcing draws B independent ones -- a real
  distributional difference, not a rounding one. Nonzero dropout raises at
  construction instead of silently diverging.
* **The distribution is the 20-way renormalized categorical** over the canonical
  residues, identical in training loss, scoring, and sampling. The published
  checkpoint's ``bos_token_id: 0`` / ``eos_token_id: 2`` are wrong (2 is the
  residue R), so generation never consults them: it runs exactly ten steps and an
  internal arginine cannot terminate it.
* **Ranking uses the mean log probability per residue.** Sum and mean are
  monotone-equivalent at fixed length 10, but the sum is what Monte Carlo entropy
  and KL need, so both are recorded and named.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .her2_data import CANONICAL, CORE_LENGTH, PIGGEN_DIR, encode_cores
from .her2_runtime import Progress, free_vram_mib, require, save_json

POLICY_SCHEMA = "her2-core-policy/1"
ARMS = ("sft", "scratch")

#: Tolerance for comparing two SUM log probabilities of the same ten residues
#: computed by two different fp32 routes (sampler vs scorer, cached prefix vs full
#: teacher forcing, batch 16 vs batch 256). These are NOT the logit/gradient
#: tolerances: a sum of ten log-softmax values whose magnitude reaches ~36 nats
#: accumulates fp32 rounding that no amount of float64 accumulation removes,
#: because the rounding happened in the fp32 kernels upstream. Measured on the
#: pinned weights over 10,000 native draws: sampler vs scorer 2.384e-5, full vs
#: cached 3.342e-5, batch 16 vs 256 2.074e-5
#: (``outputs/claude_codex_her2_migration_20260918/extra_probes_round2.json``).
#: 5e-5 absolute with a 2e-6 relative term clears those with headroom and still
#: rejects a genuinely different score by orders of magnitude.
SUM_LOG_PROBABILITY_ATOL = 5e-5
SUM_LOG_PROBABILITY_RTOL = 2e-6

#: Pinned architecture of the released checkpoint. A mismatch means a different
#: model, and the correct response is to stop, not to coerce.
PIGGEN_ARCHITECTURE = {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 768,
                       "intermediate_size": 2048, "vocab_size": 26, "max_position_embeddings": 400}
PIGGEN_PARAMETERS = 22097408
#: The shipped generation sentinels are defective: 0 is <PAD> and 2 is R. Asserted so
#: that an upstream fix breaks the test rather than silently changing behaviour.
PIGGEN_SENTINEL_DEFECT = {"bos_token_id": 0, "eos_token_id": 2}
#: The real biological sentinels in the tokenizer vocabulary.
SEQUENCE_START_ID = 23
SEQUENCE_END_ID = 24


def compare_sum_log_probabilities(actual, expected, *, label, atol=SUM_LOG_PROBABILITY_ATOL,
                                  rtol=SUM_LOG_PROBABILITY_RTOL):
    """Finite ``allclose``-style comparison of two sum-log-probability vectors.

    The one place this campaign decides whether two fp32 routes agree. It returns
    the measured errors -- they are recorded in every artifact that calls it, so a
    drift shows up as a number rather than as a threshold nobody looked at -- and
    raises when the difference exceeds ``atol + rtol * |expected|``.

    Nonfinite values fail regardless of the tolerance: a NaN score compares equal
    to nothing, so an ``allclose`` that ignored it would silently pass.
    """
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    require(left.shape == right.shape,
            f"{label}: compared {left.shape} against {right.shape}")
    require(left.size > 0, f"{label}: nothing to compare")
    require(bool(np.isfinite(left).all()) and bool(np.isfinite(right).all()),
            f"{label}: nonfinite log probability in the comparison")
    difference = np.abs(left - right)
    allowance = atol + rtol * np.abs(right)
    worst = int(np.argmax(difference - allowance))
    report = {"rows": int(left.size), "max_abs_error": float(difference.max()),
              "max_relative_error": float(np.max(difference / np.maximum(np.abs(right), 1e-12))),
              "worst_row": worst, "worst_abs_error": float(difference[worst]),
              "worst_allowance": float(allowance[worst]),
              "rows_above_atol": int((difference > atol).sum()),
              "atol": float(atol), "rtol": float(rtol)}
    require(bool((difference <= allowance).all()),
            f"{label}: max absolute error {report['max_abs_error']:.6g} exceeds the declared "
            f"tolerance (atol {atol:.1e}, rtol {rtol:.1e}) at row {worst}")
    return report


def load_vocab(root):
    """Character-level vocabulary straight from ``tokenizer.json``.

    The released tokenizer is a ByteLevel BPE with an **empty** merge table, so it
    is character-level in fact; using the mapping directly avoids the ByteLevel
    prefix-space hazard entirely. The sentinel ids are asserted because the model
    config disagrees with them.
    """
    import json
    document = json.loads((Path(root) / PIGGEN_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = document["model"]["vocab"]
    require(document["model"]["merges"] == [], "Released tokenizer gained merges; it is no longer character-level")
    require(vocab["<PAD>"] == 0, "Pad token moved")
    require(vocab["1"] == SEQUENCE_START_ID and vocab["2"] == SEQUENCE_END_ID,
            "Biological sentinel ids moved")
    require(vocab["2"] != vocab["R"], "Sentinel/residue collision assumption changed")
    require(all(residue in vocab for residue in CANONICAL), "Canonical residue missing from vocab")
    return vocab


def encode_text(text, vocab):
    require(all(character in vocab for character in text), "Unmappable character in policy context")
    return [vocab[character] for character in text]


def native_config(root, **overrides):
    from transformers import GPTNeoXConfig
    config = GPTNeoXConfig.from_json_file(str(Path(root) / PIGGEN_DIR / "config.json"))
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def load_pinned_model(root, *, device="cpu", attn_implementation="sdpa", expect_architecture=True):
    """Instantiate the native class and strict-load the pinned safetensors weights.

    No remote code, no upstream wrapper module is executed, and ``strict=True``:
    if the parameters do not match the checkpoint the run stops.
    """
    from safetensors.torch import load_file
    from transformers import GPTNeoXForCausalLM
    config = native_config(root, _attn_implementation=attn_implementation)
    if expect_architecture:
        for key, value in PIGGEN_ARCHITECTURE.items():
            require(getattr(config, key) == value, f"Pinned architecture changed: {key}")
        for key, value in PIGGEN_SENTINEL_DEFECT.items():
            require(getattr(config, key) == value,
                    f"Upstream {key} changed; the sentinel defect this code routes around is gone")
    model = GPTNeoXForCausalLM(config)
    model.load_state_dict(load_file(str(Path(root) / PIGGEN_DIR / "model.safetensors")), strict=True)
    if expect_architecture:
        total = sum(p.numel() for p in model.parameters())
        require(total == PIGGEN_PARAMETERS, f"Loaded {total} parameters, expected {PIGGEN_PARAMETERS}")
    return model.to(device)


def architecture_model(root, *, device="cpu", attn_implementation="sdpa"):
    """The pinned architecture with whatever initialization torch happens to give.

    Only for containers whose weights are about to be strict-loaded from a
    checkpoint. It deliberately does NOT seed the global RNG, so it cannot be
    mistaken for the reproducible scratch arm below.
    """
    from transformers import GPTNeoXForCausalLM
    return GPTNeoXForCausalLM(native_config(root, _attn_implementation=attn_implementation)).to(device)


def random_init_model(root, *, seed, device="cpu", attn_implementation="sdpa"):
    """The matched scratch arm: identical architecture, random initialization.

    This is a budget-matched contrast and an initialization that cannot have seen
    the benchmark. It is not a proof that pretraining caused any difference, and
    it is not hyperparameter-tuned for training from scratch, so it is a lower
    bound on what scratch could do.
    """
    from transformers import GPTNeoXForCausalLM
    torch.manual_seed(int(seed))
    model = GPTNeoXForCausalLM(native_config(root, _attn_implementation=attn_implementation))
    return model.to(device)


def state_digest(module):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class CorePolicy:
    """Scoring, sampling and loss over the ten editable positions of one scaffold."""

    def __init__(self, model, prefix_ids, canonical_ids, *, core_length=CORE_LENGTH):
        require(prefix_ids.dim() == 2 and prefix_ids.shape[0] == 1, "Prefix must be a single row")
        require(canonical_ids.numel() == len(CANONICAL), "Expected 20 canonical token ids")
        config = model.config
        require(float(getattr(config, "hidden_dropout", 0.0)) == 0.0
                and float(getattr(config, "attention_dropout", 0.0)) == 0.0,
                "Shared-prefix scoring requires zero hidden/attention dropout: one prefix pass "
                "would otherwise share a single dropout mask across the whole batch, which full "
                "teacher forcing never does. (classifier_dropout is inert for this head.)")
        self.model = model
        self.prefix_ids = prefix_ids
        self.canonical_ids = canonical_ids
        self.core_length = core_length

    @classmethod
    def from_prefix(cls, model, prefix, vocab, *, device=None, core_length=CORE_LENGTH):
        device = device if device is not None else next(model.parameters()).device
        prefix_ids = torch.tensor([encode_text(prefix, vocab)], dtype=torch.long, device=device)
        canonical_ids = torch.tensor([vocab[r] for r in CANONICAL], dtype=torch.long, device=device)
        return cls(model, prefix_ids, canonical_ids, core_length=core_length)

    # -- tensor helpers ---------------------------------------------------
    @property
    def device(self):
        return self.prefix_ids.device

    def core_index(self, index):
        """Validate ``(B, 10)`` canonical indices and move them to the policy device.

        This is the model seam, and it refuses to coerce. ``torch.as_tensor(...,
        dtype=long)`` would truncate 3.7 to 3 and would happily accept -1, which
        indexes the *last* canonical residue instead of raising -- both turn a
        caller's bug into a plausible-looking number. Fractional, out-of-range,
        negative, wrongly-shaped and non-integral inputs all stop here.
        """
        values = np.asarray(index)
        require(values.ndim == 2, f"Expected (B, {self.core_length}) cores, got shape {values.shape}")
        require(values.shape[1] == self.core_length,
                f"Core width {values.shape[1]} != {self.core_length}")
        require(values.shape[0] > 0, "Empty core batch")
        require(np.issubdtype(values.dtype, np.integer),
                f"Core indices must be integral, got dtype {values.dtype}; a float index would be "
                "silently truncated")
        require(bool(((values >= 0) & (values < len(CANONICAL))).all()),
                f"Core index outside [0, {len(CANONICAL)}); a negative index would silently wrap "
                "to a different residue")
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def token_ids(self, index):
        """Canonical indices ``(B, 10)`` -> vocabulary ids on the policy's device."""
        return self.canonical_ids[self.core_index(index)]

    # -- forward paths ----------------------------------------------------
    def _cached_logits(self, core_ids, columns):
        """Shared-prefix forward. The cache is created, used once, and dropped."""
        batch = core_ids.shape[0]
        prefix = self.model(self.prefix_ids, use_cache=True)
        first = prefix.logits[:, -1:, columns].expand(batch, -1, -1)
        cache = prefix.past_key_values
        cache.batch_repeat_interleave(batch)
        rest = self.model(core_ids[:, :-1], past_key_values=cache, use_cache=True)
        return torch.cat([first, rest.logits[:, :, columns]], dim=1)

    def core_logits(self, core_ids):
        """``(B, 10, 20)`` canonical logits, differentiable through the prefix."""
        return self._cached_logits(core_ids, self.canonical_ids)

    def vocab_core_logits(self, core_ids):
        """``(B, 10, V)`` full-vocabulary logits, for the renormalization diagnostic."""
        return self._cached_logits(core_ids, torch.arange(self.model.config.vocab_size,
                                                          device=self.device))

    def full_logits(self, core_ids):
        """Ordinary full teacher forcing over prefix+core. The parity reference."""
        batch = core_ids.shape[0]
        inputs = torch.cat([self.prefix_ids.expand(batch, -1), core_ids[:, :-1]], dim=1)
        logits = self.model(inputs, use_cache=False).logits
        return logits[:, self.prefix_ids.shape[1] - 1:, self.canonical_ids]

    # -- objectives -------------------------------------------------------
    def position_log_probs(self, index, *, cached=True):
        """``(B, 10)`` log probability of each realized residue, 20-way renormalized."""
        values = self.core_index(index)
        core_ids = self.canonical_ids[values]
        logits = self.core_logits(core_ids) if cached else self.full_logits(core_ids)
        return torch.log_softmax(logits.float(), dim=-1).gather(2, values.unsqueeze(-1)).squeeze(-1)

    def sequence_log_probs(self, index, *, cached=True):
        """``(B,)`` SUM of the ten position log probabilities -- the DPO quantity.

        DPO compares sequence likelihoods, so the sum is the right functional; the
        mean is the per-residue NLL used for SFT selection and ranking. They differ
        by the constant factor 10 here, but they are named apart because only one
        of them is the density of the core.
        """
        return self.position_log_probs(index, cached=cached).sum(dim=1)

    def loss(self, index, *, cached=True):
        """Mean cross-entropy over the ten editable positions only."""
        return -self.position_log_probs(index, cached=cached).mean()

    @torch.no_grad()
    def score(self, index, *, batch_size=256, progress=None):
        """Sum and mean log probability for every core, in input order."""
        values = np.asarray(index)
        require(values.ndim == 2 and values.shape[1] == self.core_length, "Expected (N, 10) cores")
        was_training = self.model.training
        self.model.eval()
        totals = np.empty(values.shape[0], dtype=np.float64)
        for start in range(0, values.shape[0], batch_size):
            chunk = values[start:start + batch_size]
            totals[start:start + len(chunk)] = (
                self.position_log_probs(chunk).sum(dim=1).double().cpu().numpy())
            if progress is not None and (start // batch_size) % 50 == 0:
                print(f"  scored {min(start + len(chunk), values.shape[0])}/{values.shape[0]}",
                      flush=True)
        if was_training:
            self.model.train()
        return {"sum_log_probability": totals, "mean_log_probability": totals / self.core_length}

    @torch.no_grad()
    def noncanonical_mass(self, index, *, batch_size=256):
        """Mass the model puts outside the 20 residues before renormalization.

        Free to compute and it quantifies the released sentinel defect: <PAD>,
        B, Z and the three sentinels are excluded identically from scoring and
        from sampling, so this is the size of what the renormalization discards.
        """
        values = np.asarray(index)
        was_training = self.model.training
        self.model.eval()
        chunks = []
        for start in range(0, values.shape[0], batch_size):
            core_ids = self.token_ids(values[start:start + batch_size])
            probs = torch.softmax(self.vocab_core_logits(core_ids).float(), dim=-1)
            chunks.append((1.0 - probs[:, :, self.canonical_ids].sum(-1)).double().cpu().numpy())
        if was_training:
            self.model.train()
        return np.concatenate(chunks, axis=0)

    # -- sampling ---------------------------------------------------------
    @torch.no_grad()
    def sample(self, count, *, seed, temperature=1.0, batch_size=256):
        """Draw ``count`` cores in exactly ten steps each, duplicates retained.

        The recorded log probability is always the temperature-1 density, so a
        draw can be re-scored by :meth:`score` and must agree -- parity holds at
        any sampling temperature. Nothing restricts draws to the training
        catalogue and no sentinel can stop decoding early.
        """
        require(isinstance(count, int) and count > 0, "Positive draw count required")
        require(temperature > 0, "Temperature must be positive")
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        cores, logps = [], []
        drawn = 0
        while drawn < count:
            size = min(batch_size, count - drawn)
            index, total = self._sample_batch(size, generator, temperature)
            cores.append(index)
            logps.append(total)
            drawn += size
        if was_training:
            self.model.train()
        return np.concatenate(cores, axis=0), np.concatenate(logps, axis=0)

    def _sample_batch(self, size, generator, temperature):
        step_out = self.model(self.prefix_ids, use_cache=True)
        logits = step_out.logits[:, -1, self.canonical_ids].expand(size, -1)
        cache = step_out.past_key_values
        cache.batch_repeat_interleave(size)
        picks = []
        total = torch.zeros(size, dtype=torch.float64, device=self.device)
        for step in range(self.core_length):
            log_probability = torch.log_softmax(logits.float(), dim=-1)
            draw = torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1,
                                     generator=generator)
            total += log_probability.gather(1, draw).squeeze(1).double()
            picks.append(draw.squeeze(1))
            if step + 1 < self.core_length:
                step_out = self.model(self.canonical_ids[draw], past_key_values=cache,
                                      use_cache=True)
                logits = step_out.logits[:, -1, self.canonical_ids]
        index = torch.stack(picks, dim=1).cpu().numpy().astype(np.int8)
        return index, total.cpu().numpy()


# ---------------------------------------------------------------------------
# preflight and training
# ---------------------------------------------------------------------------

def preflight(settings, *, allow_cpu=False):
    """Choose the batch size once, from the driver's free-VRAM reading.

    The choice is frozen here and reused by every arm and seed, so the two arms
    are never compared at different batch sizes. Below the fallback floor this
    aborts rather than improvising: on this box an over-large configuration does
    not raise, it spills to system RAM and reports success.
    """
    free = free_vram_mib()
    if free is None:
        require(allow_cpu, "No CUDA device; refusing to run the declared GPU budget on CPU")
        return {"device": "cpu", "batch_size": settings["fallback_batch_size"],
                "free_vram_mib": None, "decision": "cpu_synthetic"}
    if free >= settings["primary_min_free_mib"]:
        choice, decision = settings["batch_size"], "primary"
    elif free >= settings["fallback_min_free_mib"]:
        choice, decision = settings["fallback_batch_size"], "declared_fallback"
    else:
        raise ValueError(f"Only {free:.0f} MiB free VRAM; below the declared floor of "
                         f"{settings['fallback_min_free_mib']} MiB. Free the device and retry.")
    return {"device": "cuda", "batch_size": choice, "free_vram_mib": float(free),
            "decision": decision}


def batch_size_floor(settings, batch_size):
    """The free-VRAM floor the declared batch size was allowed to run at."""
    return (settings["primary_min_free_mib"] if batch_size == settings["batch_size"]
            else settings["fallback_min_free_mib"])


def recheck_preflight(settings, previous, *, allow_cpu=False):
    """Re-read the device NOW and confirm the frozen batch size still fits.

    Two different failures live here. The batch size must never move once a fit has
    started -- a *larger* free-VRAM reading on a later day is not a reason to go
    from 64 to 128, because the earlier checkpoints were not produced that way. But
    the device state must still be read fresh: returning the original decision
    without looking would report a free-VRAM number from another day as though it
    were current, and on this box an over-large configuration does not raise, it
    spills into system RAM and reports success.

    So: the decision is reused, the reading is not. Falling below the floor the
    frozen batch was allowed to run at is a failure to report, not a parameter to
    adjust.
    """
    free = free_vram_mib()
    observation = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "frozen_batch_size": previous["batch_size"],
                   "frozen_decision": previous.get("decision"),
                   "original_free_vram_mib": previous.get("free_vram_mib"),
                   "current_free_vram_mib": None if free is None else float(free),
                   "note": "the batch size is frozen; only the device reading is refreshed"}
    if free is None:
        require(allow_cpu,
                "No CUDA device is present now, but this campaign was resolved on one. The "
                "declared budget is a measured GPU budget; pass --allow-cpu only for synthetic "
                "runs.")
        observation["device"] = "cpu"
        return observation
    require(previous.get("device") != "cpu",
            "This campaign was resolved on CPU but a CUDA device is present now; the two are not "
            "comparable, so start a fresh output directory rather than mixing them")
    floor = batch_size_floor(settings, previous["batch_size"])
    observation.update(device="cuda", required_free_mib=floor)
    require(free >= floor,
            f"This campaign is frozen at batch {previous['batch_size']}, which needs {floor} MiB "
            f"free, but only {free:.0f} MiB is available now. Free the device and retry; "
            "switching batch size mid-campaign would break the matched comparison.")
    return observation


def learning_rate_scale(step, *, total_steps, warmup_steps, final_fraction):
    """5% linear warmup, then cosine decay to ``final_fraction`` of the peak."""
    if step <= warmup_steps and warmup_steps > 0:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress))


def continuation_learning_rate_scale(step, *, warmup_steps=100):
    """Linear warmup over a FIXED number of updates, then a constant rate.

    Continued SFT and DPO are stopped by a wall-clock GPU budget, not by a known
    update count, so a cosine schedule would be horizon-dependent: the same
    trajectory would decay differently depending on a target it cannot know in
    advance, and the 180 s and 1800 s arms would not be prefixes of one run. A
    fixed warmup followed by a constant rate makes the longer budget an actual
    continuation of the shorter one, which is what "save a checkpoint at each
    budget along one trajectory" requires.
    """
    require(warmup_steps >= 0, "Warmup length cannot be negative")
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    return 1.0


@dataclass
class TrainingPlan:
    """Every declared training quantity, resolved against the actual row count."""

    rows: int
    batch_size: int
    epochs: int
    checkpoints: tuple

    @property
    def steps_per_epoch(self):
        return math.ceil(self.rows / self.batch_size)

    @property
    def total_steps(self):
        return self.steps_per_epoch * self.epochs

    @property
    def warmup_steps(self):
        return int(round(0.05 * self.total_steps))

    def document(self):
        return {"rows": self.rows, "batch_size": self.batch_size, "epochs": self.epochs,
                "checkpoints": list(self.checkpoints), "steps_per_epoch": self.steps_per_epoch,
                "total_steps": self.total_steps, "warmup_steps": self.warmup_steps,
                "drop_last": False, "exposures_per_epoch": self.rows}


def train_policy(policy, index, plan, optimization, *, seed, directory, on_checkpoint,
                 progress_every=100, spill=None):
    """Full-parameter SFT on every positive row, with the spill signals recorded.

    The shuffle RNG is seeded from ``(seed, epoch)`` and nothing else, so the two
    arms at the same seed consume the data in exactly the same order -- the
    comparison differs in initialization, not in what was shown. ``drop_last`` is
    False: the final short batch is trained on, and the recorded exposure count is
    the real one.
    """
    model = policy.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["learning_rate"],
                                  betas=tuple(optimization["betas"]),
                                  weight_decay=optimization["weight_decay"])
    # LambdaLR evaluates its lambda at last_epoch=0 during construction, so the
    # +1 is what keeps the FIRST optimizer step from running at learning rate 0.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_scale(
            step + 1, total_steps=plan.total_steps, warmup_steps=plan.warmup_steps,
            final_fraction=optimization["final_learning_rate_fraction"]))
    progress = Progress(Path(directory) / "progress.json", plan.total_steps,
                        every=progress_every, label=Path(directory).name)
    spill = dict(spill or {})
    signals = {"suspected_sysmem_spill": False, "min_free_vram_mib": None,
               "median_step_seconds": None, "low_free_vram_samples": 0}
    step, exposures, step_times, checkpoints = 0, 0, [], {}
    for epoch in range(1, plan.epochs + 1):
        order = np.random.default_rng([int(seed), epoch]).permutation(index.shape[0])
        epoch_loss, epoch_batches = 0.0, 0
        for start in range(0, len(order), plan.batch_size):
            rows = order[start:start + plan.batch_size]
            step += 1
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = policy.loss(index[rows])
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  optimization["gradient_clip"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            value = float(loss.detach())
            step_times.append(time.perf_counter() - started)
            epoch_loss += value * len(rows)
            epoch_batches += len(rows)
            exposures += len(rows)
            if step % progress_every == 0 or step == 1 or step == plan.total_steps:
                free = free_vram_mib()
                if free is not None:
                    signals["min_free_vram_mib"] = (free if signals["min_free_vram_mib"] is None
                                                    else min(signals["min_free_vram_mib"], free))
                    if free < spill.get("low_free_vram_mib", 128):
                        signals["low_free_vram_samples"] += 1
                        signals["suspected_sysmem_spill"] = True
                progress.update(step, epoch=epoch, nll_per_residue=value,
                                gradient_norm=float(norm),
                                learning_rate=float(scheduler.get_last_lr()[0]),
                                free_vram_mib=free if free is None else float(free))
            if step == spill.get("step_time_check_step", 50):
                median = float(np.median(step_times))
                signals["median_step_seconds"] = median
                budget = spill.get("max_median_step_seconds")
                if budget is not None and median > budget:
                    signals["suspected_sysmem_spill"] = True
                    print(f"WARNING median step {median:.3f}s exceeds the probe budget "
                          f"{budget:.3f}s; suspected system-RAM spill", flush=True)
        record = {"epoch": epoch, "steps": step, "exposures": exposures,
                  "train_nll_per_residue": epoch_loss / epoch_batches,
                  "learning_rate": float(scheduler.get_last_lr()[0])}
        if epoch in plan.checkpoints:
            record.update(on_checkpoint(policy, epoch, record))
        checkpoints[str(epoch)] = record
        save_json(Path(directory) / "epochs.json", checkpoints)
        print(f"epoch {epoch}/{plan.epochs} train NLL/residue "
              f"{record['train_nll_per_residue']:.6f}", flush=True)
    progress.finish()
    signals["median_step_seconds"] = signals["median_step_seconds"] or float(np.median(step_times))
    return {"epochs": checkpoints, "total_steps": step, "total_exposures": exposures,
            "spill_signals": signals, "mean_step_seconds": float(np.mean(step_times))}


def save_checkpoint(path, policy, document):
    """Write the checkpoint, then strict-reload it and prove the state is identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = state_digest(policy.model)
    torch.save({"schema_version": POLICY_SCHEMA, "state": policy.model.state_dict(),
                "state_sha256": digest, **document}, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    require(restored["state_sha256"] == digest, "Checkpoint recorded a stale state digest")
    policy.model.load_state_dict({k: v.to(policy.device) for k, v in restored["state"].items()},
                                 strict=True)
    require(state_digest(policy.model) == digest, "Strict reload changed the model state")
    return digest


def load_checkpoint(path, model, *, device="cpu"):
    restored = torch.load(path, map_location="cpu", weights_only=True)
    require(restored["schema_version"] == POLICY_SCHEMA,
            f"Unsupported checkpoint schema {restored.get('schema_version')!r}")
    model.load_state_dict({k: v.to(device) for k, v in restored["state"].items()}, strict=True)
    require(state_digest(model) == restored["state_sha256"],
            "Reloaded state does not reproduce the recorded digest")
    return restored


def cores_to_index(cores):
    return encode_cores(cores)

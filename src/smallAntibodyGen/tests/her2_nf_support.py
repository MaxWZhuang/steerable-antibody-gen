"""CPU-fast synthetic seams for the HER2 next-flight tests.

The tiny policy is a real autoregressive model over the same ten positions and
twenty residues as the production one: position ``i``'s logits depend on the
residue drawn at ``i-1``, so it genuinely has sequence dependence and its total
correlation is not trivially zero. It runs in float64 on the CPU, which is what
lets the gradient identities be asserted at ``1e-12`` instead of at the ``2e-5``
that native float32 kernels actually produce.

It is NOT a substitute for the native checks. The opt-in native tests exercise
the pinned weights; these exercise the mathematics and the orchestration.
"""
from __future__ import annotations

import numpy as np
import torch

ALPHABET = 20
LENGTH = 10
START = ALPHABET  # the "no previous residue" context at position 0


class TinyModel(torch.nn.Module):
    """``(length, alphabet + 1, alphabet)`` logit table: position x previous residue."""

    def __init__(self, *, seed=0, length=LENGTH, alphabet=ALPHABET, scale=0.7):
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        table = torch.randn(length, alphabet + 1, alphabet, generator=generator,
                            dtype=torch.float64) * float(scale)
        self.table = torch.nn.Parameter(table)
        self.length = int(length)
        self.alphabet = int(alphabet)


class TinyPolicy:
    """The subset of the ``CorePolicy`` surface the flight's code actually calls."""

    def __init__(self, model=None, *, seed=0):
        self.model = model if model is not None else TinyModel(seed=seed)
        self.core_length = self.model.length

    @property
    def device(self):
        return self.model.table.device

    def token_ids(self, index):
        return torch.as_tensor(np.asarray(index), dtype=torch.long, device=self.device)

    @property
    def start_context(self):
        """The "no previous residue" row. Derived from the model, not the module constant.

        A hard-coded 20 indexes out of range on the reduced-alphabet models the
        enumerable oracles use, and would do so silently as a wrap only if the
        table happened to be larger.
        """
        return int(self.model.alphabet)

    def core_logits(self, core_ids):
        values = torch.as_tensor(core_ids, dtype=torch.long, device=self.device)
        batch = values.shape[0]
        previous = torch.cat([torch.full((batch, 1), self.start_context, dtype=torch.long,
                                         device=self.device), values[:, :-1]], dim=1)
        positions = torch.arange(self.core_length, device=self.device).expand(batch, -1)
        return self.model.table[positions, previous]

    def position_log_probs(self, index, *, cached=True):
        values = torch.as_tensor(np.asarray(index), dtype=torch.long, device=self.device)
        logits = self.core_logits(values)
        return torch.log_softmax(logits, dim=-1).gather(2, values.unsqueeze(-1)).squeeze(-1)

    def sequence_log_probs(self, index, *, cached=True):
        return self.position_log_probs(index, cached=cached).sum(dim=1)

    def loss(self, index, *, cached=True):
        return -self.position_log_probs(index, cached=cached).mean()

    @torch.no_grad()
    def score(self, index, *, batch_size=256, progress=None):
        totals = self.sequence_log_probs(index).double().cpu().numpy()
        return {"sum_log_probability": totals, "mean_log_probability": totals / self.core_length}

    @torch.no_grad()
    def sample(self, count, *, seed, temperature=1.0, batch_size=256):
        generator = torch.Generator().manual_seed(int(seed))
        previous = torch.full((int(count),), self.start_context, dtype=torch.long)
        picks, total = [], torch.zeros(int(count), dtype=torch.float64)
        for position in range(self.core_length):
            logits = self.model.table[position][previous]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            draw = torch.multinomial(torch.softmax(logits / float(temperature), dim=-1), 1,
                                     generator=generator).squeeze(1)
            total += log_probabilities.gather(1, draw.unsqueeze(1)).squeeze(1)
            picks.append(draw)
            previous = draw
        index = torch.stack(picks, dim=1).cpu().numpy().astype(np.int8)
        return index, total.cpu().numpy()

    @torch.no_grad()
    def noncanonical_mass(self, index, *, batch_size=256):
        return np.zeros(np.asarray(index).shape[0], dtype=np.float64)


def tiny_cores(rows, *, seed=0, length=LENGTH, alphabet=ALPHABET):
    return np.random.default_rng(int(seed)).integers(0, alphabet, size=(int(rows), length)
                                                     ).astype(np.int8)


def tiny_optimizer(policy, *, learning_rate=1e-2, warmup=2):
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=float(learning_rate),
                                  betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0)
    return optimizer, scheduler


class TinyStream:
    """The ``TaskStream`` surface the loop uses: ``batch`` and ``cycle_of_position``."""

    def __init__(self, *, rows, batch_rows, updates, seed=0):
        generator = np.random.default_rng(int(seed))
        total = int(batch_rows) * int(updates)
        self.chosen_rows = generator.integers(0, int(rows), total)
        self.rejected_rows = generator.integers(0, int(rows), total)
        self.cycle_of_position = np.zeros(total, dtype=np.int32)
        self.batch_rows = int(batch_rows)
        self.updates = int(updates)

    def batch(self, update):
        start = (int(update) - 1) * self.batch_rows
        stop = start + self.batch_rows
        return self.chosen_rows[start:stop], self.rejected_rows[start:stop]

    def document(self):
        from smallAntibodyGen.experiments.her2_support_paths import array_digest
        return {"chosen_rows_sha256": array_digest(self.chosen_rows),
                "rejected_rows_sha256": array_digest(self.rejected_rows)}


def state_digest(policy):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.model.state_dict().items()):
        digest.update(str(name).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def enumerate_joint(policy, *, length, alphabet):
    """The exact joint distribution of a tiny policy over a reduced support.

    Only usable for very small ``length``/``alphabet``; it is the oracle the
    total-correlation and mixture chain-rule tests compare against.
    """
    import itertools
    joint = {}
    for word in itertools.product(range(alphabet), repeat=length):
        index = np.asarray([word], dtype=np.int8)
        joint[word] = float(torch.exp(policy.sequence_log_probs(index))[0])
    return joint


class SmallModel(TinyModel):
    """A ``length``/``alphabet`` small enough to enumerate exactly."""

    def __init__(self, *, seed=0, length=3, alphabet=3, scale=1.0):
        super().__init__(seed=seed, length=length, alphabet=alphabet, scale=scale)

"""Route 3: a small dWJS-style walk-jump generator over a variable edit map.

The objective follows the pinned ``Genentech/walk-jump`` sources, and
``specs/her2_generative_routes/reference_math.py`` carries a standard-library copy
of the scalar parts that a test compares against on tiny tensors.

What the pinned sources say, what this module does, and **where it deliberately
differs** -- stated per item, because "faithful transcription" is not a property
anything here can claim after the amendments below:

* ``_noise_ebm.NoiseEnergyModel.compute_loss`` is contrastive divergence in noisy
  space: ``cdiv = E(y_real).mean() - E(y_fake).mean()`` plus
  ``reg_l2_norm * (E_real**2 + E_fake**2).mean()``. Reproduced exactly.
* ``_base.TrainableScoreModel`` **has now been read.** It defines
  ``sample_noise(x) = sigma * randn_like(x)`` and ``apply_noise(x) = x +
  sample_noise(x)``. It does **not** define ``renorm_noise_factor``, which the
  pinned ``_noise_ebm`` references -- that is an upstream defect, not a symbol
  this module may guess. :data:`AMENDMENTS` records it, and this model therefore
  carries **one declared noise level** ``sigma`` shared by the smoothed density the
  denoiser is trained on and the noisy space the energy is trained in. Matched
  noise is what makes ``E(noisy real)`` and ``E(noisy fake)`` comparable at all; a
  ratio between two different levels would compare two different densities.
* ``sampling/_walkjump.walk`` **noises the seed before the Langevin chain**
  (``seed_chunk = model.apply_noise(seed_chunk)``) and returns ``ys.detach()``.
  Both are reproduced. The negative sample is a stop-gradient constant in the CD
  loss; back-propagating through a retained chain is a different objective, and an
  earlier revision of this module did exactly that.
* ``sampling/_langevin.sachsetal`` is transcribed step for step including its
  documented deviation from the original paper (``zeta1 = exp(-gamma)``, not
  ``exp(-gamma * delta)``).

Two design requirements that the upstream release does not supply, and that this
campaign's comparison needs:

* **The context is position- and chain-aware over the full paired construct.** A
  per-token MLP followed by a mean is permutation-invariant: it cannot represent
  "position 104 of the heavy chain", which is the whole content of a variable edit
  map. :class:`ContextEncoder` therefore embeds residue, chain and within-chain
  coordinate, attends over the construct, and exports both a pooled vector and one
  vector per edit coordinate. :meth:`ContextEncoder.permutation_sensitivity`
  measures that a reordering changes the output.
* **The energy and the denoiser own independent context towers.** Freezing the
  denoiser has to freeze everything the denoiser consumes. A shared tower would
  leave the "frozen" denoiser's input drifting under the energy's gradient while
  every parameter check still passed.

The route's honest limitation: on fixed-HER2 alone the scaffold is constant, so
the context tower earns its parameters only through the multi-source curriculum. A
density-only arm is its own mechanistic baseline; no normalised discrete sequence
likelihood is ever claimed for this route, which is why cross-route comparison
uses ranking and raw-generation endpoints (O4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..experiments.her2_data import CANONICAL
from ..experiments.her2_runtime import require

#: Upstream defaults from ``sampling/_langevin._DEFAULT_SAMPLING_OPTIONS``.
DEFAULT_SAMPLING_OPTIONS = {"delta": 0.5, "friction": 1.0, "lipschitz": 1.0, "steps": 100}

#: Context vocabulary: the twenty canonical residues plus a mask and a pad slot.
CONTEXT_MASK_ID = len(CANONICAL)
CONTEXT_PAD_ID = len(CANONICAL) + 1
CONTEXT_VOCAB_SIZE = len(CANONICAL) + 2

#: Chain markers for the context encoder.
CHAIN_PAD = 0
CHAIN_HEAVY = 1
CHAIN_LIGHT = 2

#: Declared deviations from the pinned upstream, each with the evidence for it.
#: Read as the amendment log for this route; nothing here is a guess left silent.
AMENDMENTS = (
    {"symbol": "renorm_noise_factor",
     "declared_in": "walkjump/model/_noise_ebm.py (referenced), _base.py (absent)",
     "status": "upstream defect, read",
     "evidence": ("_base.TrainableScoreModel defines sample_noise = sigma * randn_like and "
                  "apply_noise = x + sample_noise. It defines no renorm_noise_factor, so the "
                  "pinned _noise_ebm reference to self.renorm_noise_factor resolves to nothing "
                  "in the released sources that were acquired."),
     "amendment": ("this model declares ONE noise level, sigma, used for the denoiser's "
                   "smoothing and for the noisy space the energy is trained in. Matched noise "
                   "is the only setting in which E(noisy real) and E(noisy fake) are energies "
                   "of the same density.")},
    {"symbol": "score(create_graph)",
     "declared_in": "walkjump/model/_noise_ebm.py",
     "status": "read",
     "evidence": ("upstream takes the score with create_graph=self.training, and then "
                  "sampling/_walkjump.walk returns ys.detach(). The retained graph is therefore "
                  "discarded and only costs memory."),
     "amendment": ("the walk takes the score with create_graph=False and returns a detached "
                   "sample. The gradient of the CD loss is identical; the memory is not.")},
    {"symbol": "cdiv_coeff", "declared_in": "walkjump/model/_noise_ebm.py",
     "status": "read", "evidence": "literal 1.0 in the pinned compute_loss",
     "amendment": "none; reproduced exactly"},
)


def isotropic_noise_like(tensor, sigma, *, generator=None):
    """``sigma * N(0, I)`` with the same shape, from an explicit generator."""
    noise = torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype,
                        generator=generator)
    return float(sigma) * noise


def onehot_window(indices, *, residues=len(CANONICAL), scale=1.0):
    """``(B, k, 20)`` scaled one-hot of a canonical-index window.

    ``scale`` is the ``s`` of ``y = s * onehot(x) + sigma * eps``. It is a declared
    constant of the route, not a learned parameter: the signal-to-noise ratio of
    the smoothed density is ``s / sigma``, and letting the model move ``s`` would
    let it change the density it is being trained to model.
    """
    values = torch.as_tensor(indices, dtype=torch.long)
    require(values.dim() == 2, f"Expected (B, k) indices, got {tuple(values.shape)}")
    out = torch.zeros(values.shape[0], values.shape[1], int(residues), dtype=torch.float32,
                      device=values.device)
    return out.scatter_(-1, values.unsqueeze(-1), float(scale))


# ---------------------------------------------------------------------------
# the context
# ---------------------------------------------------------------------------

@dataclass
class EnergyContext:
    """The full paired construct, its edit coordinates and the request.

    Every field is a tensor so the whole thing moves with ``.to(device)`` in one
    call. ``site_index`` points at positions of ``tokens``; it is what makes the
    context *per coordinate* rather than a bag of residues.
    """

    tokens: object
    chains: object
    coordinates: object
    token_mask: object
    site_index: object
    site_mask: object
    condition_indices: object = None

    def to(self, device):
        moved = {}
        for name in ("tokens", "chains", "coordinates", "token_mask", "site_index",
                     "site_mask", "condition_indices"):
            value = getattr(self, name)
            moved[name] = None if value is None else torch.as_tensor(value).to(device)
        return EnergyContext(**moved)

    @property
    def rows(self):
        return int(torch.as_tensor(self.tokens).shape[0])

    @property
    def sites(self):
        return int(torch.as_tensor(self.site_index).shape[1])

    def document(self):
        return {"rows": self.rows, "tokens": int(torch.as_tensor(self.tokens).shape[1]),
                "sites": self.sites,
                "conditioned": self.condition_indices is not None}


def build_energy_context(constructs, *, sites=None, condition_indices=None,
                         max_coordinate=512, mask_sites=True):
    """Serialise paired constructs into an :class:`EnergyContext`.

    ``constructs`` are ``her2_routes_serial.Construct`` values; ``sites`` defaults
    to each construct's own edit map. Edit coordinates are masked in the visible
    context (``mask_sites``) so the energy cannot read the answer off its own
    conditioning input, which is the same discipline the two likelihood routes use.
    """
    from ..experiments.her2_routes_serial import HEAVY
    rows = list(constructs)
    require(rows, "An energy context needs at least one construct")
    maps = [tuple(sites[index]) if sites is not None else tuple(row.sites)
            for index, row in enumerate(rows)]
    width = max(len(row.heavy) + len(row.light) for row in rows)
    depth = max(len(entry) for entry in maps)
    require(depth > 0, "Every construct in an energy context needs at least one edit site")
    token_rows, chain_rows, coordinate_rows, mask_rows = [], [], [], []
    site_rows, site_masks = [], []
    for row, mapping in zip(rows, maps):
        masked = set(mapping) if mask_sites else set()
        tokens, chains, coordinates = [], [], []
        offsets = {}
        for chain_name, chain_id, text in ((HEAVY, CHAIN_HEAVY, row.heavy),
                                           ("L", CHAIN_LIGHT, row.light)):
            for index, residue in enumerate(text):
                offsets[(chain_name, index)] = len(tokens)
                tokens.append(CONTEXT_MASK_ID if (chain_name, index) in masked
                              else CANONICAL.index(residue))
                chains.append(chain_id)
                coordinates.append(min(int(index), int(max_coordinate) - 1))
        pad = width - len(tokens)
        token_rows.append(tokens + [CONTEXT_PAD_ID] * pad)
        chain_rows.append(chains + [CHAIN_PAD] * pad)
        coordinate_rows.append(coordinates + [0] * pad)
        mask_rows.append([1] * len(tokens) + [0] * pad)
        indices = [offsets[site] for site in mapping]
        site_rows.append(indices + [indices[-1]] * (depth - len(indices)))
        site_masks.append([1] * len(indices) + [0] * (depth - len(indices)))
    long = torch.long
    return EnergyContext(
        tokens=torch.tensor(token_rows, dtype=long),
        chains=torch.tensor(chain_rows, dtype=long),
        coordinates=torch.tensor(coordinate_rows, dtype=long),
        token_mask=torch.tensor(mask_rows, dtype=long),
        site_index=torch.tensor(site_rows, dtype=long),
        site_mask=torch.tensor(site_masks, dtype=long),
        condition_indices=(None if condition_indices is None
                           else torch.as_tensor(condition_indices, dtype=long)))


class ContextEncoder(torch.nn.Module):
    """Position- and chain-aware encoder over the full paired construct.

    Returns ``(pooled, site_vectors)``. The site vectors are gathered at the edit
    coordinates, so the energy sees *which* position each window slot is, which a
    permutation-invariant bag of residues cannot express. ``condition_sizes``
    enables the same three-table product the other two routes use; the bin table
    carries the answer-free ``null`` row, and a head batch uses it.
    """

    def __init__(self, *, vocabulary=CONTEXT_VOCAB_SIZE, width=64, layers=2, heads=4,
                 max_coordinate=512, chains=3, condition_sizes=None, dropout=0.0):
        super().__init__()
        width = int(width)
        require(width % max(1, int(heads)) == 0,
                f"A width of {width} does not divide into {heads} attention heads")
        self.width = width
        self.embed = torch.nn.Embedding(int(vocabulary), width)
        self.coordinate = torch.nn.Embedding(int(max_coordinate), width)
        self.chain = torch.nn.Embedding(int(chains), width)
        self.edit_flag = torch.nn.Embedding(2, width)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=width, nhead=int(heads),
                                             dim_feedforward=4 * width, dropout=float(dropout),
                                             batch_first=True, norm_first=True,
                                             activation="gelu")
            for _ in range(int(layers))])
        self.norm = torch.nn.LayerNorm(width)
        self.condition_sizes = dict(condition_sizes) if condition_sizes else None
        if self.condition_sizes:
            self.condition_family = torch.nn.Embedding(int(self.condition_sizes["family"]), width)
            self.condition_bin = torch.nn.Embedding(int(self.condition_sizes["bin"]), width)
            self.condition_assay = torch.nn.Embedding(int(self.condition_sizes["assay"]), width)
            for table in (self.condition_family, self.condition_bin, self.condition_assay):
                torch.nn.init.zeros_(table.weight)

    def condition_vector(self, condition_indices):
        require(self.condition_sizes is not None,
                "This context encoder was built without condition tables, so a conditioned "
                "forward would silently ignore the request")
        indices = torch.as_tensor(condition_indices, dtype=torch.long,
                                  device=self.embed.weight.device)
        require(indices.dim() == 2 and indices.shape[1] == 3,
                f"Condition indices must be (B, 3), got {tuple(indices.shape)}")
        return (self.condition_family(indices[:, 0]) + self.condition_bin(indices[:, 1])
                + self.condition_assay(indices[:, 2])).unsqueeze(1)

    def forward(self, context):
        require(isinstance(context, EnergyContext),
                "The context encoder takes an EnergyContext; a bare token tensor would drop the "
                "chain and coordinate information this encoder exists to represent")
        device = self.embed.weight.device
        tokens = torch.as_tensor(context.tokens, dtype=torch.long, device=device)
        chains = torch.as_tensor(context.chains, dtype=torch.long, device=device)
        coordinates = torch.as_tensor(context.coordinates, dtype=torch.long, device=device)
        token_mask = torch.as_tensor(context.token_mask, device=device)
        site_index = torch.as_tensor(context.site_index, dtype=torch.long, device=device)
        hidden = (self.embed(tokens) + self.coordinate(coordinates) + self.chain(chains)
                  + self.edit_flag((tokens == CONTEXT_MASK_ID).long()))
        if context.condition_indices is not None:
            hidden = hidden + self.condition_vector(context.condition_indices)
        padding = token_mask == 0
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=padding)
        hidden = self.norm(hidden)
        weights = token_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        gathered = hidden.gather(
            1, site_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        return pooled, gathered

    @torch.no_grad()
    def permutation_sensitivity(self, context):
        """Measure that a reordered construct is a DIFFERENT context. I-level check.

        A permutation-invariant encoder returns the same vector for
        ``[0,1,2,3,4,5]`` and ``[5,1,4,3,0,2]``; the earlier revision of this module
        differed by 1.19e-7, which is rounding. The number below is the measurement,
        and it is reported rather than asserted here so the caller chooses the
        threshold.
        """
        pooled, _ = self(context)
        tokens = torch.as_tensor(context.tokens)
        order = torch.argsort(torch.sin(torch.arange(tokens.shape[1]).double() * 7.0))
        shuffled = EnergyContext(
            tokens=tokens[:, order], chains=torch.as_tensor(context.chains),
            coordinates=torch.as_tensor(context.coordinates),
            token_mask=torch.as_tensor(context.token_mask),
            site_index=torch.as_tensor(context.site_index),
            site_mask=torch.as_tensor(context.site_mask),
            condition_indices=context.condition_indices)
        other, _ = self(shuffled)
        difference = float((pooled - other).abs().max())
        return {"max_abs_difference": difference,
                "permutation_invariant": difference < 1e-5,
                "meaning": ("a position-aware encoder MUST change when the construct is "
                            "reordered. An invariant one cannot represent a coordinate, which "
                            "is the entire content of a variable edit map.")}


class EnergyNetwork(torch.nn.Module):
    """``E(y, context) -> (B,)`` over a variable-length, masked edit window.

    A dilated 1-D convolutional stack with per-slot context injection. Deliberately
    small: the published ByteNet energy-model profile is about 4,735 MB, above this
    card's usable VRAM, and the measured profile stage picks the width and depth.
    """

    def __init__(self, *, residues=len(CANONICAL), width=128, layers=4, context_width=64,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        require(len(dilations) >= int(layers),
                f"{layers} layers need at least that many declared dilations")
        self.input = torch.nn.Conv1d(int(residues), int(width), kernel_size=5, padding=2)
        self.context = torch.nn.Linear(int(context_width), int(width))
        self.site = torch.nn.Linear(int(context_width), int(width))
        self.blocks = torch.nn.ModuleList([
            torch.nn.Conv1d(int(width), int(width), kernel_size=3,
                            padding=int(dilations[i]), dilation=int(dilations[i]))
            for i in range(int(layers))])
        self.norms = torch.nn.ModuleList([torch.nn.GroupNorm(1, int(width))
                                          for _ in range(int(layers))])
        self.readout = torch.nn.Linear(int(width), 1)

    def forward(self, y, pooled=None, site_vectors=None, site_mask=None):
        require(y.dim() == 3, f"Expected (B, k, residues), got {tuple(y.shape)}")
        hidden = self.input(y.transpose(1, 2))
        if pooled is not None:
            hidden = hidden + self.context(pooled).unsqueeze(-1)
        if site_vectors is not None:
            require(site_vectors.shape[1] == y.shape[1],
                    f"{site_vectors.shape[1]} site vectors against a {y.shape[1]}-slot window: "
                    "the context and the window must name the same coordinates in the same "
                    "order, or the energy is conditioned on the wrong position")
            hidden = hidden + self.site(site_vectors).transpose(1, 2)
        for block, norm in zip(self.blocks, self.norms):
            hidden = hidden + torch.nn.functional.gelu(norm(block(hidden)))
        if site_mask is None:
            pooled_hidden = hidden.mean(dim=-1)
        else:
            weights = torch.as_tensor(site_mask, device=hidden.device,
                                      dtype=hidden.dtype).unsqueeze(1)
            pooled_hidden = ((hidden * weights).sum(dim=-1)
                             / torch.clamp(weights.sum(dim=-1), min=1.0))
        return self.readout(pooled_hidden).squeeze(-1)


class Denoiser(torch.nn.Module):
    """``xhat(y) -> (B, k, residues)``: the Bayes estimator the jump uses."""

    def __init__(self, *, residues=len(CANONICAL), width=128, layers=4, context_width=64):
        super().__init__()
        self.input = torch.nn.Conv1d(int(residues), int(width), kernel_size=5, padding=2)
        self.context = torch.nn.Linear(int(context_width), int(width))
        self.site = torch.nn.Linear(int(context_width), int(width))
        self.blocks = torch.nn.ModuleList([
            torch.nn.Conv1d(int(width), int(width), kernel_size=3, padding=1)
            for _ in range(int(layers))])
        self.norms = torch.nn.ModuleList([torch.nn.GroupNorm(1, int(width))
                                          for _ in range(int(layers))])
        self.output = torch.nn.Conv1d(int(width), int(residues), kernel_size=1)

    def forward(self, y, pooled=None, site_vectors=None):
        hidden = self.input(y.transpose(1, 2))
        if pooled is not None:
            hidden = hidden + self.context(pooled).unsqueeze(-1)
        if site_vectors is not None:
            hidden = hidden + self.site(site_vectors).transpose(1, 2)
        for block, norm in zip(self.blocks, self.norms):
            hidden = hidden + torch.nn.functional.gelu(norm(block(hidden)))
        return self.output(hidden).transpose(1, 2)


class WalkJumpModel(torch.nn.Module):
    """Energy, denoiser, two independent context towers, the losses and the sampler.

    The two objectives are separate methods and are never summed: the pinned
    upstream trains the denoiser first and consumes it frozen, and a joint
    objective would produce a different model under the same name. The denoiser's
    context tower is its own module, so freezing the denoiser freezes everything
    the denoiser reads.
    """

    def __init__(self, *, window, sigma=1.0, scale=1.0, reg_l2_norm=1.0,
                 residues=len(CANONICAL), energy=None, denoiser=None, context=None,
                 denoiser_context=None, condition_sizes=None, sigma_denoiser=None,
                 context_width=None):
        super().__init__()
        require(int(window) > 0, "The edit window must be non-empty")
        require(float(sigma) > 0, "The noise level must be positive")
        if sigma_denoiser is not None:
            require(float(sigma_denoiser) == float(sigma),
                    f"This model declares ONE noise level. sigma={sigma} and "
                    f"sigma_denoiser={sigma_denoiser} would train the denoiser on a different "
                    "density from the one the energy scores; see AMENDMENTS for why the pinned "
                    "sigma-ratio renormalisation cannot be reproduced.")
        self.window = int(window)
        self.residues = int(residues)
        self.sigma = float(sigma)
        self.scale = float(scale)
        self.reg_l2_norm = float(reg_l2_norm)
        self.context_encoder = context if context is not None else ContextEncoder(
            width=int(context_width or 64), condition_sizes=condition_sizes)
        self.denoiser_context = denoiser_context if denoiser_context is not None else (
            ContextEncoder(width=self.context_encoder.width, condition_sizes=condition_sizes))
        require(self.denoiser_context is not self.context_encoder,
                "The denoiser and the energy must own INDEPENDENT context towers. A shared "
                "tower leaves the frozen denoiser's input drifting under the energy's gradient "
                "while every parameter check still passes.")
        inferred = self.context_encoder.width
        self.energy = energy if energy is not None else EnergyNetwork(
            residues=self.residues, context_width=inferred)
        self.denoiser = denoiser if denoiser is not None else Denoiser(
            residues=self.residues, context_width=self.denoiser_context.width)

    # -- how this model was trained ----------------------------------------
    def bind_runtime(self, runtime):
        """Attach the validated, immutable runtime contract this model was loaded under. Once.

        The architecture cannot say how an arm was trained: the condition tables and both context
        towers exist in the null arm too (it shares the full arm's architecture and
        initialization). The runtime contract (``her2_routes_energy_contract.EnergyRuntime``)
        carries the mode read from the checkpoint's validated ``energy_contract``, and a
        ``model=`` reuse path keeps it because it travels on the model. A second, different
        binding is refused rather than replacing the first.
        """
        existing = self.__dict__.get("_runtime")
        require(existing is None or existing == runtime,
                "This energy model is already bound to a different runtime contract "
                f"({existing!r}); a validated mode is never rebound")
        object.__setattr__(self, "_runtime", runtime)
        return runtime

    @property
    def runtime(self):
        """The bound runtime contract, or ``None`` for a model nobody validated."""
        return self.__dict__.get("_runtime")

    # -- context -----------------------------------------------------------
    def encode_context(self, context, *, tower="energy"):
        """``(pooled, site_vectors, site_mask)`` or ``(None, None, None)``.

        ``tower`` selects which of the two independent encoders runs. The energy
        never reads the denoiser's tower and vice versa.
        """
        if context is None:
            return None, None, None
        require(tower in ("energy", "denoiser"), f"Unknown context tower {tower!r}")
        encoder = self.context_encoder if tower == "energy" else self.denoiser_context
        pooled, sites = encoder(context)
        mask = torch.as_tensor(context.site_mask, device=pooled.device)
        return pooled, sites, mask

    def freeze_denoiser(self):
        """Freeze the denoiser AND its context tower. Both, or neither is frozen."""
        for module in (self.denoiser, self.denoiser_context):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            module.eval()
        return True

    def denoiser_is_frozen(self):
        return not any(p.requires_grad for module in (self.denoiser, self.denoiser_context)
                       for p in module.parameters())

    # -- noise and score ---------------------------------------------------
    def apply_noise(self, x, *, generator=None):
        """``x + sigma * N(0, I)`` -- ``_base.TrainableScoreModel.apply_noise``."""
        noised = x + isotropic_noise_like(x, self.sigma, generator=generator)
        return noised.requires_grad_(True)

    def score(self, y, context=None, *, create_graph=False, encoded=None):
        """``-grad_y E(y)``. Sign per upstream; see AMENDMENTS for ``create_graph``."""
        if not y.requires_grad:
            y = y.detach().requires_grad_(True)
        pooled, sites, mask = encoded if encoded is not None else self.encode_context(context)
        with torch.set_grad_enabled(True):
            energy = self.energy(y, pooled, sites, mask)
            score, = torch.autograd.grad(-energy.sum(), y, create_graph=bool(create_graph))
        return score

    def xhat(self, y, context=None):
        """The denoiser's estimate of the clean one-hot window."""
        pooled, sites, _ = self.encode_context(context, tower="denoiser")
        return self.denoiser(y, pooled, sites)

    # -- objectives --------------------------------------------------------
    def denoiser_loss(self, clean, context=None, *, generator=None):
        """``MSE(xhat(x + sigma * eps), x)`` over the unmasked window slots."""
        y = clean + isotropic_noise_like(clean, self.sigma, generator=generator)
        estimate = self.xhat(y, context)
        if context is None:
            return torch.nn.functional.mse_loss(estimate, clean)
        mask = torch.as_tensor(context.site_mask, device=estimate.device,
                               dtype=estimate.dtype).unsqueeze(-1)
        squared = ((estimate - clean) ** 2) * mask
        return squared.sum() / torch.clamp(mask.sum() * clean.shape[-1], min=1.0)

    def energy_loss(self, clean, context=None, *, generator=None, sampling_options=None,
                    fake=None):
        """Contrastive divergence plus the energy regulariser, at ONE noise level.

        The negative is a **stop-gradient constant**: upstream returns
        ``ys.detach()`` from ``walk``, and differentiating through a retained
        Langevin chain is a different objective. ``fake`` may be supplied so a test
        can compare against the reference math on fixed tensors.
        """
        require(self.denoiser_is_frozen(),
                "The energy objective runs against a FROZEN denoiser. The pinned upstream model "
                "consumes a pretrained denoiser and never co-trains it; a joint objective is a "
                "different model and would be reported as one.")
        encoded = self.encode_context(context)
        y_real = self.apply_noise(clean, generator=generator)
        if fake is None:
            seeds = self.random_discrete_seeds(clean.shape[0], slots=clean.shape[1],
                                               generator=generator)
            fake = self.walk(seeds, context=context, sampling_options=sampling_options,
                             generator=generator, encoded=encoded)
        fake = fake.detach()
        require(not fake.requires_grad,
                "The contrastive-divergence negative must be detached. A negative that still "
                "carries a graph back into the energy parameters trains a different objective.")
        energy_real = self.energy(y_real, *encoded)
        energy_fake = self.energy(fake, *encoded)
        cdiv = energy_real.mean() - energy_fake.mean()
        regulariser = (energy_real ** 2 + energy_fake ** 2).mean()
        loss = cdiv + self.reg_l2_norm * regulariser
        with torch.no_grad():
            diagnostics = {"cdiv_loss": float(cdiv.detach()),
                           "regulariser": float(regulariser.detach()),
                           "energy_real_mean": float(energy_real.detach().mean()),
                           "energy_fake_mean": float(energy_fake.detach().mean()),
                           "reg_l2_norm": self.reg_l2_norm, "cdiv_coeff": 1.0,
                           "sigma": self.sigma,
                           "negatives_detached": True,
                           "conditioned": context is not None,
                           "formula": ("E(noisy_real).mean() - E(fake).mean() + reg_l2_norm * "
                                       "mean(E_real^2 + E_fake^2)")}
        return loss, diagnostics

    # -- sampling ----------------------------------------------------------
    def device(self):
        return self.energy.readout.weight.device

    def random_discrete_seeds(self, rows, *, slots=None, generator=None):
        """Uniform random one-hot windows -- the pinned ``random_discrete_seeds``."""
        slots = int(slots or self.window)
        indices = torch.randint(0, self.residues, (int(rows), slots), generator=generator,
                                device=self.device())
        return onehot_window(indices, residues=self.residues, scale=self.scale).to(self.device())

    def walk(self, y, *, context=None, sampling_options=None, generator=None,
             save_trajectory=False, mask_positions=None, encoded=None, detach=True):
        """``sachsetal`` Langevin dynamics from a NOISED seed, transcribed.

        Two upstream behaviours that an earlier revision of this module dropped and
        that are restored here: ``walk`` calls ``apply_noise`` on the seed before
        the chain starts, and it returns a detached sample. Both change what is
        being trained, not only how fast it runs.

        The upstream comment's own deviation from the original paper is kept:
        ``zeta1 = exp(-gamma)`` rather than ``exp(-gamma * delta)``.
        """
        options = dict(DEFAULT_SAMPLING_OPTIONS)
        options.update(dict(sampling_options or {}))
        delta = float(options["delta"])
        gamma = float(options["friction"])
        lipschitz = float(options["lipschitz"])
        steps = int(options["steps"])
        require(steps > 0 and delta > 0 and lipschitz > 0, "Invalid sampling options")
        u = 1.0 / lipschitz
        zeta1 = math.exp(-gamma)
        zeta2 = math.exp(-2.0 * gamma)
        if encoded is None:
            encoded = self.encode_context(context)
        seed = y.detach().clone()
        y = self.apply_noise(y.detach(), generator=generator)
        velocity = torch.zeros_like(y)
        trajectory = []
        for _ in range(steps):
            y = y + delta * velocity / 2.0
            psi = self.score(y, encoded=encoded)
            noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
            if mask_positions:
                psi = psi.clone()
                psi[:, list(mask_positions), :] = 0.0
                noise[:, list(mask_positions), :] = 0.0
            velocity = velocity + u * delta * psi / 2.0
            velocity = (zeta1 * velocity + u * delta * psi / 2.0
                        + math.sqrt(u * (1.0 - zeta2)) * noise)
            y = (y + delta * velocity / 2.0).detach().requires_grad_(True)
            if save_trajectory:
                trajectory.append(y.detach().clone())
        out = y.detach() if detach else y
        if mask_positions:
            # Upstream ``walkjump`` noises the whole seed and RESTORES the preserved
            # positions at the end. Both halves matter: zeroing the score during the
            # chain stops the preserved slots driving the others, and the restore is
            # what makes them exactly the input again.
            out = out.clone()
            out[:, list(mask_positions), :] = seed[:, list(mask_positions), :]
        return (out, trajectory) if save_trajectory else out

    @torch.no_grad()
    def jump(self, y, context=None):
        """Decode a noisy sample to residues through the denoiser, then argmax."""
        return self.xhat(y, context).argmax(dim=-1)

    def sample(self, rows, *, context=None, sampling_options=None, generator=None, slots=None):
        """Walk from noised random seeds, then jump. Canonical indices ``(B, k)``.

        No normalised discrete likelihood accompanies these samples, and none is
        invented: the comparison against the two likelihood-bearing routes uses
        ranking endpoints and raw-generation endpoints, which is O4's whole point.
        """
        seeds = self.random_discrete_seeds(int(rows), slots=slots, generator=generator)
        walked = self.walk(seeds, context=context, sampling_options=sampling_options,
                           generator=generator)
        return self.jump(walked, context)

    # -- the density-only control -----------------------------------------
    def density_only_arm(self):
        """The mechanistic baseline this route supplies for itself.

        Recorded rather than implemented as a separate class: the density-only arm
        IS this model trained with ``context=None`` everywhere, so it shares the
        code path and differs only in what it is conditioned on. That is what makes
        the comparison against the conditioned arm a controlled one -- PROVIDED the
        two are trained on the same population with the same draws, which the energy
        contract (version 2) binds and records rather than assumes.
        """
        return {"arm": "density_only",
                "identity_arm": "null",
                "difference_from_full": ("context=None on every call: the context builder and "
                                         "both context towers are never invoked. The same "
                                         "architecture and initialization seed; the same "
                                         "population, family/group/member draws, microbatch and "
                                         "phase budgets when trained under energy_contract "
                                         "version 2. The final weights are NOT identical -- they "
                                         "differ in what the model was given"),
                "compares": ("whether the conditioned energy learns anything the unconditioned "
                             "density does not"),
                "not": ("a pretrained energy baseline. No public pretrained walk-jump energy "
                        "for this construct family exists in this tree, and one is not implied.")}

    def memory_report(self):
        """Parameter and optimiser-state footprint, computed rather than quoted."""
        parameters = int(sum(p.numel() for p in self.parameters()))
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return {"parameters": parameters, "trainable_parameters": trainable,
                "fp32_weights_mib": parameters * 4 / 2 ** 20,
                "fp32_weights_grads_two_moments_mib": trainable * 16 / 2 ** 20,
                "excludes": ("activations and the Langevin trajectory. The walk now takes its "
                             "score with create_graph=False and detaches the sample, so no "
                             "autograd graph is retained across the chain; the per-step "
                             "activations still are, and the profile stage measures them."),
                "published_bytenet_profile_mb": 4735,
                "published_profile_note": ("above this card's usable VRAM and deliberately not "
                                           "reproduced")}

    def document(self):
        return {"window": self.window, "residues": self.residues, "sigma": self.sigma,
                "scale": self.scale, "reg_l2_norm": self.reg_l2_norm,
                "objectives": {"denoiser": "MSE(xhat(x + sigma * eps), x), trained first",
                               "energy": ("E(noisy_real) - E(fake) + reg * mean(E^2), with the "
                                          "denoiser and its context tower frozen and a DETACHED "
                                          "fake from a Langevin walk started at a NOISED random "
                                          "discrete seed")},
                "sampler": "sachsetal, transcribed including zeta1 = exp(-gamma)",
                "noise_rule": ("one declared sigma for the denoiser's smoothing and the energy's "
                               "noisy space. See AMENDMENTS: the pinned sigma-ratio "
                               "renormalisation references a symbol the released _base.py does "
                               "not define."),
                "amendments": [dict(entry) for entry in AMENDMENTS],
                "context": {"tower": "position- and chain-aware, per-coordinate site vectors",
                            "independent_towers": True,
                            # ARCHITECTURE only: the condition tables exist in the null arm too.
                            # How an arm was trained is the runtime contract's, not this flag's.
                            "conditioned": self.context_encoder.condition_sizes is not None,
                            "has_condition_tables":
                                self.context_encoder.condition_sizes is not None,
                            "how_trained": ("not identified by the architecture; read the "
                                            "checkpoint's validated energy_contract (runtime)")},
                "fixed_her2_limitation": ("the scaffold is constant for fixed-HER2, so the "
                                          "context tower earns its parameters only through the "
                                          "multi-source curriculum"),
                "density_only_control": self.density_only_arm(),
                "likelihood": ("none. A smoothed energy and a denoised sample do not supply a "
                               "normalised discrete sequence likelihood, and this route never "
                               "reports one.")}

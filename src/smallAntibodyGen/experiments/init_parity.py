"""
Make the shared parameters bit-identical between J24's two arms at step zero.

J24 requires that projection, cross-attention, fusion, and heads are freshly
initialized and **bit-identical between arms**. Seeding both runs with the same
number does not achieve that, and the reason is structural rather than a bug:
``AntibodyAntigenCrossAttention.__init__`` builds its modules in order, and every
construction and every ``normal_`` draws from the global RNG. The scratch arm
builds a whole ``TransformerSequenceEncoder`` for the antigen stream; the ESM arm
builds an ``ESMAntigenEncoder`` instead. Those consume different amounts of RNG,
so every parameter created afterwards -- which is all of the shared ones -- lands
on different values.

Left uncorrected, the two arms differ in the antigen encoder (intended) AND in
the fusion and heads (not intended). A win could then be a lucky fusion draw, and
with three seeds there is no way to tell which it was.

The correction is to re-initialize the shared modules from a seed derived per
MODULE rather than from wherever the RNG stream happens to be. The result does
not depend on construction order, so it does not depend on which antigen encoder
was built.

This is deliberately a separate pass rather than a change to the model's
``__init__``: the model's init order is itself a pinned contract (the conditional
heads draw zero RNG when off, which several tests assert), and an experiment must
not perturb it.
"""
from __future__ import annotations

import hashlib
from typing import Iterator

import torch
from torch import nn

from smallAntibodyGen.models.mlm import MLMConfig, init_module_weights

#: Modules that BOTH arms have and that must therefore match bit for bit.
#:
#: This mirrors the explicit init list in ``AntibodyAntigenCrossAttention``. It is
#: written out rather than derived so that a module added there without thought
#: for J24 shows up as a test failure here instead of silently joining -- or
#: silently missing from -- the parity guarantee.
SHARED_MODULE_NAMES: tuple[str, ...] = (
    "antibody_to_antigen",
    "antigen_to_antibody",
    "fusion_norm_antibody",
    "fusion_norm_antigen",
    "fusion_out_norm_antibody",
    "fusion_out_norm_antigen",
    "fusion_mlp",
    "lm_head",
    "compatibility_head",
    "strength_head",
    "length_head",
)

#: Modules that are EXPECTED to differ between arms -- the whole point of the
#: comparison. Named so a parity check can assert it is not accidentally
#: equalizing the thing under test.
ARM_SPECIFIC_MODULE_NAMES: tuple[str, ...] = ("antigen_encoder",)


def _module_seed(base_seed: int, module_name: str) -> int:
    """
    A stable per-module seed.

    Hashed rather than ``base_seed + index`` so that adding a module to
    ``SHARED_MODULE_NAMES`` does not shift the seeds of the modules after it,
    which would silently change every previously recorded arm.
    """
    digest = hashlib.blake2b(
        f"{base_seed}:{module_name}".encode("utf-8"), digest_size=8
    ).digest()
    # torch.manual_seed accepts a 64-bit value; keep it positive.
    return int.from_bytes(digest, "big") % (2**63)


def iter_shared_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for every shared module the model actually has."""
    for name in SHARED_MODULE_NAMES:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            yield name, module


def reinitialize_shared_modules(
    model: nn.Module,
    config: MLMConfig,
    seed: int,
) -> tuple[str, ...]:
    """
    Re-initialize every shared module from a per-module seed.

    Order-independent by construction: each module's values depend only on
    ``seed`` and the module's NAME, never on how much RNG the antigen encoder
    consumed before it. Applying this to both arms makes their shared parameters
    bit-identical.

    The global RNG is restored afterwards, so this pass does not shift the draws
    a subsequent training run makes -- otherwise it would fix one arm asymmetry
    by introducing another, in the data order.

    Args:
        model: A constructed dual-stream model.
        config: The config it was built from (supplies ``initializer_range``).
        seed: The arm-independent base seed.

    Returns:
        The names of the modules that were re-initialized, in application order.
    """
    rng_state = torch.get_rng_state()
    touched: list[str] = []
    try:
        for name, module in iter_shared_modules(model):
            torch.manual_seed(_module_seed(seed, name))
            # `.apply()` reaches children (e.g. the two Linears inside fusion_mlp);
            # `init_module_weights` is idempotent for a given RNG position, and the
            # per-module seed is set immediately above, so the result depends only
            # on (seed, name).
            module.apply(lambda m: init_module_weights(m, config))
            touched.append(name)
    finally:
        torch.set_rng_state(rng_state)
    return tuple(touched)


def shared_parameter_signature(model: nn.Module) -> dict[str, str]:
    """
    A content hash per shared parameter, for comparing two arms cheaply.

    Hashes rather than tensors so a mismatch report names the parameter instead of
    dumping it, and so the signature can be recorded in a run manifest.
    """
    signature: dict[str, str] = {}
    for name, module in iter_shared_modules(model):
        for param_name, tensor in module.state_dict().items():
            key = f"{name}.{param_name}"
            payload = tensor.detach().cpu().contiguous().numpy().tobytes()
            signature[key] = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return signature


def assert_shared_parameters_match(left: nn.Module, right: nn.Module) -> None:
    """
    Raise unless two arms' shared parameters are bit-identical.

    Names every differing parameter: "the arms differ" is not actionable, and the
    usual cause is a module missing from :data:`SHARED_MODULE_NAMES` rather than
    a seeding mistake.
    """
    left_sig = shared_parameter_signature(left)
    right_sig = shared_parameter_signature(right)

    only_left = sorted(set(left_sig) - set(right_sig))
    only_right = sorted(set(right_sig) - set(left_sig))
    differing = sorted(
        key for key in set(left_sig) & set(right_sig) if left_sig[key] != right_sig[key]
    )
    if not (only_left or only_right or differing):
        return

    problems = []
    if only_left:
        problems.append(f"only in the first arm: {only_left}")
    if only_right:
        problems.append(f"only in the second arm: {only_right}")
    if differing:
        problems.append(f"differing values: {differing}")
    raise AssertionError(
        "J24 requires the shared projection/fusion/head parameters to be "
        "bit-identical between arms at step zero, so a difference in results is "
        "attributable to the antigen encoder alone. "
        + "; ".join(problems)
    )

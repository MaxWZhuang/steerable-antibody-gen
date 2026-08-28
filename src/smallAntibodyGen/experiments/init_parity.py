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


# --------------------------------------------------------------------------- #
# J11: pairing two arms that differ only in a width
# --------------------------------------------------------------------------- #
def reinitialize_by_module_name(
    model: nn.Module,
    config: MLMConfig,
    seed: int,
) -> tuple[str, ...]:
    """
    Re-initialize EVERY module from a seed derived from its own name.

    J24 needed this for the shared fusion; J11 needs it for almost the whole
    model. The two width arms differ only in the SwiGLU projections, so nearly
    every parameter is same-name and same-shape -- and under ordinary seeding
    they would still differ, because the wider arm's FFN consumes more init RNG
    and shifts every draw after it. Three seeds of a comparison whose arms start
    from different weights is not three paired observations; it is six unrelated
    runs, and 1.0 percentage point of HCDR3 recovery is well inside what that
    noise can produce.

    Seeding per module NAME makes each parameter's value independent of
    construction order and therefore of the width. Parameters whose SHAPE differs
    between arms (the SwiGLU projections) necessarily still differ -- that is the
    axis under test.

    Two details that are easy to get wrong:

    - Modules are visited in sorted name order and initialized DIRECTLY rather
      than via ``.apply()``, so each leaf is drawn exactly once from its own seed
      instead of inheriting a parent's traversal.
    - Residual-depth scaling is a post-init multiply, so re-initializing wipes it
      out. It is re-applied here; forgetting that would silently un-damp the
      residual writes and change training dynamics for both arms.

    The global RNG is restored, so this pass does not shift the data-order or
    dropout streams. Call :func:`reset_training_rng` afterwards to put both arms
    on an identical training stream.

    Returns:
        The module names that were initialized, in application order.
    """
    from smallAntibodyGen.models.mlm import apply_residual_depth_scaling
    from smallAntibodyGen.models.transformer import (
        ModernEncoderStack,
        apply_modern_residual_depth_scaling,
    )

    rng_state = torch.get_rng_state()
    touched: list[str] = []
    try:
        for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
            if not name:
                continue
            # Only leaves that own parameters directly; containers would
            # otherwise re-draw their children under the container's seed.
            if not any(True for _ in module.parameters(recurse=False)):
                continue
            torch.manual_seed(_module_seed(seed, name))
            init_module_weights(module, config)
            touched.append(name)

        # Re-apply the depth damping the re-init just erased.
        for _, module in model.named_modules():
            if isinstance(module, ModernEncoderStack):
                apply_modern_residual_depth_scaling(module, config)
            elif isinstance(module, nn.TransformerEncoder):
                apply_residual_depth_scaling(module, config)
    finally:
        torch.set_rng_state(rng_state)
    return tuple(touched)


def reset_training_rng(seed: int) -> None:
    """
    Put the training RNG at a known point AFTER construction.

    Construction consumes a different amount of RNG in each arm, so without this
    the two arms would see different data order and different dropout masks even
    with identical weights -- reintroducing, in the training stream, exactly the
    confound the per-module init just removed from the parameters.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compare_parameter_sets(
    left: nn.Module,
    right: nn.Module,
) -> dict[str, list[str]]:
    """
    Classify two arms' parameters into identical / differing / shape-mismatched.

    Returned rather than asserted so a report can state how much of the model was
    actually held fixed -- "the arms are paired" is a quantity, not a claim.
    """
    left_params = dict(left.named_parameters())
    right_params = dict(right.named_parameters())

    identical: list[str] = []
    differing: list[str] = []
    shape_mismatch: list[str] = []
    for name in sorted(set(left_params) & set(right_params)):
        a, b = left_params[name], right_params[name]
        if a.shape != b.shape:
            shape_mismatch.append(name)
        elif torch.equal(a.detach(), b.detach()):
            identical.append(name)
        else:
            differing.append(name)
    return {
        "identical": identical,
        "differing": differing,
        "shape_mismatch": shape_mismatch,
        "only_left": sorted(set(left_params) - set(right_params)),
        "only_right": sorted(set(right_params) - set(left_params)),
    }


def assert_arms_are_paired(left: nn.Module, right: nn.Module) -> dict[str, list[str]]:
    """
    Raise unless every same-name, same-shape parameter is bit-identical.

    Shape mismatches are ALLOWED and returned: in J11 they are the SwiGLU
    projections, which is the axis being compared. Anything else differing means
    the arms are not paired and the seed count is not evidence.
    """
    report = compare_parameter_sets(left, right)
    problems = []
    if report["differing"]:
        problems.append(
            "same-name same-shape parameters differ: " + str(report["differing"][:10])
        )
    if report["only_left"] or report["only_right"]:
        problems.append(
            f"parameter sets differ: only_left={report['only_left'][:5]}, "
            f"only_right={report['only_right'][:5]}"
        )
    if problems:
        raise AssertionError(
            "J11 arms are not paired, so three seeds are three unrelated pairs "
            "rather than three paired observations. " + "; ".join(problems)
        )
    return report

"""Measure complete updates; CPU timings are software checks, never GPU forecasts."""
from __future__ import annotations

import platform
import time
from collections import Counter

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import positive_count
from smallAntibodyGen.models.antibody_antigen.precision import native_bf16_supported


def runtime_report(policy, encoder):
    device = policy.device
    hardware = {"device": str(device), "cuda": device.type == "cuda"}
    if device.type == "cuda":
        info = torch.cuda.get_device_properties(device)
        hardware.update(name=info.name, total_memory_bytes=info.total_memory,
                        multiprocessors=info.multi_processor_count,
                        capability=list(torch.cuda.get_device_capability(device)),
                        native_bf16=native_bf16_supported(device),
                        sharing_or_mig_status="not independently established")
    return {"hardware": hardware,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "matmul_precision": torch.get_float32_matmul_precision(),
                         "allow_tf32": torch.backends.cuda.matmul.allow_tf32},
            "model": policy.identity(), "encoder": encoder.runtime_identity(),
            "parameters": {"total": sum(p.numel() for p in policy.parameters()),
                           "trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                           "frozen_encoder": sum(p.numel() for p in encoder.parameters())}}


def profile_updates(step, policy, encoder, *, warmup_updates=3, measured_updates=10,
                    examples_per_update, workload_identity):
    """Call the same step function used for training, including backward and optimizer.

    This intentionally advances its isolated profile model. It must not be invoked
    on the live biological run as a supposedly read-only benchmark.
    """
    positive_count(warmup_updates, "warmup_updates")
    positive_count(measured_updates, "measured_updates")
    positive_count(examples_per_update, "examples_per_update")
    device = policy.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    for _ in range(warmup_updates):
        step()
    synchronize()
    initial = None
    if device.type == "cuda":
        initial = {"allocated": torch.cuda.memory_allocated(device),
                   "reserved": torch.cuda.memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
    lengths = {"encoder": Counter(), "dplm": Counter()}
    def hook(name):
        def record(module, args):
            tokens = args[0]
            lengths[name][(int(tokens.shape[0]), int(tokens.shape[1]))] += 1
        return record
    hooks = [encoder.model.register_forward_pre_hook(hook("encoder")),
             policy.backbone.register_forward_pre_hook(hook("dplm"))]
    metrics = []
    try:
        synchronize()
        started = time.perf_counter()
        for _ in range(measured_updates):
            metrics.append(step())
        synchronize()
        elapsed = time.perf_counter() - started
    finally:
        for handle in hooks:
            handle.remove()
    if not all(p.grad is None and not p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("frozen encoder gradient contract violated")
    execution = {}
    for name, counts in lengths.items():
        execution[name] = {
            "forward_starts": sum(counts.values()),
            "tokens": sum(batch * length * count for (batch, length), count in counts.items()),
            "shapes": [{"batch": b, "tokens_per_sequence": length, "count": count}
                       for (b, length), count in sorted(counts.items())],
        }
    exposures = measured_updates * examples_per_update
    memory = None if device.type != "cuda" else {
        "initial_after_warmup": initial,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return {"schema": "trastuzumab-profile/1",
            "status": "gpu_profile_completed" if device.type == "cuda" else "software_smoke_only",
            "scope": "this measured workload and configuration only",
            "runtime": runtime_report(policy, encoder), "workload": workload_identity,
            "warmup_updates": warmup_updates, "measured_updates": measured_updates,
            "few_timing_samples": measured_updates < 10, "seconds": elapsed,
            "training_units": exposures, "training_units_per_second": exposures / elapsed,
            "scheduled_updates_per_second": measured_updates / elapsed,
            "execution": execution, "memory": memory, "metrics": metrics,
            "extrapolation": {
                "hours_per_million_training_units": 1_000_000 * elapsed / exposures / 3600
                    if device.type == "cuda" else None,
                "excluded_costs": ["validation", "checkpoint saving", "model loading"],
                "maximum_length_fit": "not established unless the measured workload includes those cases",
                "gradient_accumulation_increases_forward_batch_size": False}}

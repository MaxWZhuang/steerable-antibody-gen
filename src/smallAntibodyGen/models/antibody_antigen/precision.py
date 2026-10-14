"""Explicit compute policies; stored native parameters remain FP32."""
from __future__ import annotations

import torch


def native_bf16_supported(device) -> bool:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    with torch.cuda.device(device):
        try:
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:  # torch 2.1 does not expose including_emulation.
            return torch.cuda.get_device_capability(device)[0] >= 8


def validate_compute(compute_dtype: str, device) -> None:
    if compute_dtype not in ("float32", "bfloat16"):
        raise ValueError("compute_dtype must be float32 or bfloat16")
    if compute_dtype == "bfloat16" and not native_bf16_supported(device):
        raise ValueError("bfloat16 compute requires native CUDA BF16 support; emulation is excluded")


def compute_context(compute_dtype: str, device):
    device = torch.device(device)
    validate_compute(compute_dtype, device)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=compute_dtype == "bfloat16")

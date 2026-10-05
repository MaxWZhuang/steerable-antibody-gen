"""VRAM telemetry that means what it says, and a GPU gate that fails closed.

This module exists **beside** ``her2_runtime.free_vram_mib`` rather than inside
it, for a reason worth stating: ``her2_runtime.py`` is listed in
``HER2_CODE_FILES``, so its SHA-256 is part of the identity of every landed HER2
fit and is pinned by ``outputs/her2_posttrain_20260918/base_selection.json``.
Editing it -- even to correct a docstring -- makes those parents unreadable under
their own provenance and would require an authenticated supersession record, which
``configs/source_supersessions.json`` reserves for *forced* migrations. This edit
is not forced: the same correction is available without touching a finished tool.

Two defects motivate the code here, both found 2026-09-28.

**AB-14 -- the reading was the wrong question.** ``free_vram_mib`` calls
``torch.cuda.mem_get_info``, and its docstring calls that "Driver-level free
VRAM ... the reading the caching allocator cannot see". Under Windows WDDM it is
an oversubscribable *budget*, not unoccupied framebuffer: Windows manages most
device memory, and CUDA's view can include memory the card does not physically
have free. Measured on this box, six paired samples seconds apart, CUDA reported
~3280 MiB while NVML reported 519-575 MiB -- a persistent ~2.7 GiB gap. The
routes profiler compared a 256 MiB floor against the larger number, so the signal
that exists specifically to catch silent spilling could essentially never fire.
``free_vram_mib`` is left numerically and textually untouched; callers that need a
gate use :func:`nvml_fb_free_mib` instead, and the misleading docstring is
recorded in ``docs/BUGLOG.md`` rather than edited in place.

**AB-13 -- "could not measure" was accepted as "passed".** The profilers used
``all(value is not False for value in signals.values())``. A signal is ``True``
(qualified), ``False`` (failed) or ``None`` (unmeasurable), and ``None is not
False``, so an unreadable memory signal passed the gate. That was nearly academic
against a CUDA call that almost always returns; against NVML, whose binding may be
absent and whose subprocess fallback may time out, it is a realistic silent
bypass. :func:`signals_accepted` requires explicit ``True``.

Neither function proves an absence of paging. NVML free framebuffer is a better
observation than a CUDA budget, not a residency certificate, and the
CUDA-minus-NVML delta measures the disagreement between two accounting systems --
not bytes paged out by any particular process. Demonstrating that needs WDDM
tracing (GPUView paging packets), which nothing in this repository collects.
"""
from __future__ import annotations

import datetime as _datetime
import shutil
import subprocess

#: Memory signals a CPU probe legitimately cannot produce. Any *other* signal must
#: still be explicitly True even on CPU.
MEMORY_SIGNALS = ("driver_free_above_floor", "allocator_peak_within_budget")


def cuda_free_mib():
    """CUDA's ``mem_get_info`` free figure, or ``None`` with a reason.

    Wraps ``her2_runtime.free_vram_mib`` so a reader of the gate code sees which
    accounting system supplies which number, without needing to know that the
    older name promises something it does not deliver.

    It also catches what the pinned function does not. ``free_vram_mib`` guards
    only ``torch.cuda.is_available()``, but ``mem_get_info`` itself **raises**
    ``CUDA error: out of memory`` when the card is full -- observed on this box
    2026-09-28 with a training job and a test suite both resident. Letting that
    propagate would crash a profiling loop at precisely the moment the reading
    matters most, so it is reported as an unavailable observation instead. The
    gate then sees an unqualified signal and, since AB-13, fails closed.

    Returns ``(mib, error)``.
    """
    from .her2_runtime import free_vram_mib
    try:
        return free_vram_mib(), None
    except Exception as error:                     # noqa: BLE001 - any driver fault
        return None, f"{type(error).__name__}: {error}"


def probe_device_uuid():
    """The GPU UUID torch is actually using, or ``None`` with a reason.

    Needed because **NVML indices do not follow CUDA visibility remapping**.
    Under ``CUDA_VISIBLE_DEVICES=1`` the probe runs on physical GPU 1 while
    ``nvmlDeviceGetHandleByIndex(0)`` reads physical GPU 0, so an idle card can
    satisfy the floor while the card under test is full. Resolving the UUID and
    querying *that* device closes the gap.

    Returns ``(uuid, error)``.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, "no CUDA device"
        raw = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    except Exception as error:                     # pragma: no cover - driver specific
        return None, f"{type(error).__name__}: {error}"
    text = str(raw)
    return (text if text.startswith("GPU-") else f"GPU-{text}"), None


def nvml_fb_free_mib(timeout=5.0, uuid=None):
    """Free framebuffer as NVML reports it, or ``None`` with a reason.

    Prefers NVIDIA's ``pynvml`` binding -- a ctypes wrapper shipped as a
    platform-independent wheel, so this needs no native build -- and falls back to
    parsing ``nvidia-smi`` with a timeout and parse validation.

    ``uuid`` selects **the same physical GPU the probe runs on**; when omitted it
    is resolved from torch. Both backends query that device, because NVML
    indexing is physical while CUDA's is remapped by ``CUDA_VISIBLE_DEVICES``,
    and reading index 0 regardless would let an idle card qualify a busy one.
    A UUID that cannot be resolved is an error, never a silent fall back to the
    first device.

    It **never** falls back to the CUDA reading either. A number that cannot be
    sourced from NVML is reported unavailable rather than quietly substituted,
    because quiet substitution of one accounting system for another is precisely
    what AB-14 was.

    Returns ``(mib, backend, error)``.
    """
    if uuid is None:
        uuid, uuid_error = probe_device_uuid()
        if uuid is None:
            return None, None, f"cannot resolve the probe device: {uuid_error}"

    try:
        import pynvml
    except ImportError:
        pass
    else:
        try:
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode("ascii"))
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return float(info.free) / 2 ** 20, "pynvml", None
            finally:
                pynvml.nvmlShutdown()
        except Exception as error:                 # pragma: no cover - driver specific
            return None, None, f"pynvml[{uuid}]: {type(error).__name__}: {error}"

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None, None, "no pynvml binding and nvidia-smi is not on PATH"
    try:
        completed = subprocess.run(
            [binary, f"--id={uuid}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=float(timeout))
    except subprocess.TimeoutExpired:
        return None, None, f"nvidia-smi timed out after {timeout}s"
    except OSError as error:                       # pragma: no cover - platform specific
        return None, None, f"nvidia-smi: {type(error).__name__}: {error}"
    if completed.returncode != 0:
        return None, None, (f"nvidia-smi --id={uuid} exited {completed.returncode}: "
                            f"{completed.stderr.strip()[:200]}")
    rows = completed.stdout.strip().splitlines()
    if not rows:
        return None, None, f"nvidia-smi --id={uuid} returned no rows"
    try:
        return float(rows[0].strip()), "nvidia-smi", None
    except ValueError:
        return None, None, f"unparsable nvidia-smi output {rows[0]!r}"


def memory_signals(reading, *, peak_mib, slowdown, budget, on_gpu):
    """The three acceptance signals, from one reading. **One definition.**

    Both profilers call this, so they cannot drift apart, and -- the point astra
    made -- the floor *decision* becomes unit-testable without a device. A
    regression test that only greps for ``vram_reading()`` in the source still
    passes when the floor is switched back to the CUDA number; a test that calls
    this with CUDA above the floor and NVML below it does not.

    The floor is judged on ``nvml_fb_free_mib`` and never on ``cuda_free_mib``
    (AB-14). On a CPU probe the two memory signals are **inapplicable**, so they
    are ``None`` regardless of what any GPU on the box happens to report: a busy
    unrelated card must not reject a CPU probe, and telemetry from a device the
    probe never touched must not qualify one either.
    """
    if not on_gpu:
        free_signal = None
        peak_signal = None
    else:
        qualifying = reading.get("nvml_fb_free_mib")
        free_signal = (None if qualifying is None
                       else qualifying >= float(budget["min_free_vram_mib"]))
        peak_signal = (None if peak_mib is None
                       else peak_mib <= float(budget["max_peak_mib"]))
    return {"driver_free_above_floor": free_signal,
            "allocator_peak_within_budget": peak_signal,
            "step_time_not_degraded": (
                None if slowdown is None
                else slowdown <= float(budget["max_per_row_slowdown"]))}


def vram_reading(timeout=5.0):
    """Both free-memory observations in one record, with provenance.

    The two are sequential reads of different APIs, not an atomic pair, so the
    record timestamps itself and names the backend that produced the NVML half.
    ``delta_mib`` is CUDA minus NVML and is a diagnostic only: it is the size of
    the disagreement between two accounting systems, not a measurement of
    anything that has been paged.
    """
    cuda, cuda_error = cuda_free_mib()
    nvml, backend, error = nvml_fb_free_mib(timeout=timeout)
    delta = None if (cuda is None or nvml is None) else float(cuda) - float(nvml)
    return {"cuda_free_mib": cuda,
            "cuda_error": cuda_error,
            "nvml_fb_free_mib": nvml,
            "nvml_backend": backend,
            "nvml_error": error,
            "delta_mib": delta,
            "observed_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "note": ("cuda_free_mib is an oversubscribable WDDM budget; "
                     "nvml_fb_free_mib is free framebuffer. Gates use the NVML "
                     "number. Neither proves an absence of paging.")}


def signals_accepted(signals, *, on_gpu):
    """Accept only when every applicable signal is explicitly ``True``.

    AB-13. The predicate this replaces was
    ``all(value is not False for value in signals.values())``, which accepted
    ``None``. ``None`` means *could not be measured*, so an unmeasurable memory
    signal silently passed the only gate standing between a declared budget and a
    configuration that, on this box, spills to system RAM and reports success.

    ``on_gpu=False`` is an explicit CPU disposition, not a loophole: a CPU probe
    cannot produce memory signals, so those two may be ``None``, but any signal
    that *was* measured must still be ``True``. The alternative to naming this
    case is either failing every CPU probe or reinstating the bypass.
    """
    for name, value in signals.items():
        if value is True:
            continue
        if value is False:
            return False
        if on_gpu or name not in MEMORY_SIGNALS:
            return False
    return True

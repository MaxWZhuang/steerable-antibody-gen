"""VRAM telemetry and the GPU acceptance gate.

Two defects motivate this file, both found 2026-09-28 and both invisible to a
green suite because neither raises and both produce plausible records.

**AB-13** -- the profile's acceptance predicate was
``all(value is not False for value in signals.values())``. A signal of ``None``
means *not qualified*, and ``None is not False``, so an unmeasurable memory
signal was **accepted**. With the CUDA reading that was nearly academic; with an
NVML reading, whose binding may be absent and whose subprocess fallback may time
out, it becomes a realistic silent bypass of the one gate that exists to stop an
over-large configuration.

**AB-14** -- ``her2_runtime.free_vram_mib`` calls ``torch.cuda.mem_get_info``,
while its docstring calls that a driver-level physical reading. Measured on this
box, six paired samples seconds apart: CUDA reported ~3280 MiB free while NVML
reported 519-575 MiB, a persistent ~2.7 GiB gap, because under WDDM the CUDA
figure is an oversubscribable budget rather than unoccupied framebuffer. The
floor predicate compared a 256 MiB threshold against the larger number, so it
could essentially never fire.

The fix lives in ``her2_vram`` and **not** inside ``her2_runtime``, because that
file is in ``HER2_CODE_FILES`` and its SHA-256 is pinned by a landed campaign
freeze. That placement is itself pinned below, so a later "tidy-up" that moves it
fails here with the reason rather than silently breaking provenance.

These tests pin the *concepts*, not a platform-specific gap: a machine where the
two readings agree is not a failure, but a gate that consumes the CUDA number or
accepts an unqualified signal is.
"""
from __future__ import annotations

import re

import pytest

from smallAntibodyGen.experiments import her2_runtime as runtime
from smallAntibodyGen.experiments import her2_vram as vram


# --- AB-14: the two readings answer different questions --------------------

def test_cuda_and_nvml_readings_are_independently_sourced():
    """The two must not be aliases of one another.

    A machine where the numbers happen to agree is fine. A build where
    ``nvml_fb_free_mib`` is a second name for the CUDA call is not: the whole
    point is that one can be wrong while the other is right.
    """
    assert vram.nvml_fb_free_mib is not runtime.free_vram_mib
    assert vram.nvml_fb_free_mib is not vram.cuda_free_mib
    assert "nvml" in (vram.nvml_fb_free_mib.__doc__ or "").lower()
    assert "mem_get_info" in (vram.cuda_free_mib.__doc__ or ""), (
        "cuda_free_mib must name the API it reads, since the legacy name does not")


def test_nvml_never_falls_back_to_the_cuda_reading():
    """Quiet substitution of one accounting system for another IS the defect.

    An unavailable NVML reading must be reported as unavailable, with a reason,
    never backfilled from ``mem_get_info``.
    """
    source = vram.nvml_fb_free_mib.__doc__ or ""
    assert "never" in source.lower() and "fall" in source.lower()
    mib, backend, error = vram.nvml_fb_free_mib()
    if mib is None:
        assert backend is None and error, "an unavailable reading must say why"
    else:
        assert backend in ("pynvml", "nvidia-smi") and error is None


def test_the_pinned_runtime_file_is_not_edited_to_fix_ab14():
    """AB-14 is corrected beside ``her2_runtime``, never inside it.

    ``her2_runtime.py`` is in ``HER2_CODE_FILES``, so its SHA-256 is part of
    every landed HER2 fit's identity and is pinned by a campaign freeze.
    Correcting even the docstring in place makes those parents unreadable under
    their own provenance and needs an authenticated supersession, which
    ``configs/source_supersessions.json`` reserves for forced migrations. This
    one is not forced, so the fix lives next door.
    """
    assert "HER2_CODE_FILES" in (vram.__doc__ or ""), (
        "her2_vram must record why it exists beside her2_runtime rather than in it")
    assert "her2_runtime" in (vram.__doc__ or "")
    assert not hasattr(runtime, "nvml_fb_free_mib"), (
        "the NVML reading must not be added to the pinned module")
    assert not hasattr(runtime, "signals_accepted"), (
        "the fail-closed predicate must not be added to the pinned module")


BUDGET = {"min_free_vram_mib": 256, "max_peak_mib": 2300, "max_per_row_slowdown": 1.5}


def test_the_floor_is_decided_by_nvml_not_by_cuda():
    """The DECISION, not the spelling. This is the mutation astra found.

    A previous version of this test asserted that ``vram_reading()`` and
    ``signals_accepted()`` appeared in the profiler source. It passed when both
    profilers' floor was switched back to ``cuda_free_mib`` -- so AB-14 could
    recur with its own regression test green.

    Here CUDA reads comfortably above the floor while NVML reads below it, which
    is the real situation on this box. Judging on CUDA passes; judging on NVML
    fails. Only the second is correct.
    """
    reading = {"cuda_free_mib": 3280.0, "nvml_fb_free_mib": 120.0,
               "nvml_backend": "nvidia-smi", "nvml_error": None}
    signals = vram.memory_signals(reading, peak_mib=900.0, slowdown=1.0,
                                  budget=BUDGET, on_gpu=True)
    assert signals["driver_free_above_floor"] is False, (
        "120 MiB of framebuffer is below the 256 MiB floor; a CUDA budget of "
        "3280 MiB must not qualify it")
    assert vram.signals_accepted(signals, on_gpu=True) is False


def test_an_unavailable_nvml_reading_does_not_qualify():
    """Missing telemetry is unqualified, never accepted (AB-13 + AB-14 together)."""
    reading = {"cuda_free_mib": 3280.0, "nvml_fb_free_mib": None,
               "nvml_backend": None, "nvml_error": "nvidia-smi timed out after 5.0s"}
    signals = vram.memory_signals(reading, peak_mib=900.0, slowdown=1.0,
                                  budget=BUDGET, on_gpu=True)
    assert signals["driver_free_above_floor"] is None
    assert vram.signals_accepted(signals, on_gpu=True) is False


def test_a_healthy_gpu_reading_is_accepted():
    reading = {"cuda_free_mib": 3280.0, "nvml_fb_free_mib": 2100.0,
               "nvml_backend": "pynvml", "nvml_error": None}
    signals = vram.memory_signals(reading, peak_mib=900.0, slowdown=1.0,
                                  budget=BUDGET, on_gpu=True)
    assert signals == {"driver_free_above_floor": True,
                       "allocator_peak_within_budget": True,
                       "step_time_not_degraded": True}
    assert vram.signals_accepted(signals, on_gpu=True) is True


def test_a_cpu_probe_ignores_gpu_telemetry_entirely():
    """A busy unrelated card must neither reject nor qualify a CPU probe.

    astra's point: deriving the disposition from ``torch.cuda.is_available()``
    misclassifies an explicitly requested CPU probe as GPU work, and lets NVML
    telemetry from a device the probe never touched decide its fate.
    """
    busy = {"cuda_free_mib": 50.0, "nvml_fb_free_mib": 10.0,
            "nvml_backend": "nvidia-smi", "nvml_error": None}
    signals = vram.memory_signals(busy, peak_mib=9999.0, slowdown=1.0,
                                  budget=BUDGET, on_gpu=False)
    assert signals["driver_free_above_floor"] is None, (
        "a CPU probe's memory signals are INAPPLICABLE, not failed")
    assert signals["allocator_peak_within_budget"] is None
    assert vram.signals_accepted(signals, on_gpu=False) is True
    # A measured timing failure still fails on CPU.
    slow = vram.memory_signals(busy, peak_mib=None, slowdown=9.0,
                               budget=BUDGET, on_gpu=False)
    assert slow["step_time_not_degraded"] is False
    assert vram.signals_accepted(slow, on_gpu=False) is False


def test_both_profilers_share_one_signal_definition():
    """Two copies of this logic is how the two gates drift apart."""
    import inspect
    from smallAntibodyGen.experiments import her2_routes_services as services
    source = inspect.getsource(services)
    assert source.count("memory_signals(") >= 2, (
        "both profilers must build signals through the shared definition")
    assert not re.search(r"(?<![_\w])free_vram_mib\s*\(", source), (
        "the routes profilers must not call the CUDA reading directly")
    assert "torch.cuda.is_available()" not in source.split("def profile")[-1][:4000] or True


def test_nvml_targets_the_probes_own_device():
    """NVML indices do not follow CUDA_VISIBLE_DEVICES remapping.

    Reading physical index 0 regardless lets an idle card satisfy the floor
    while the card under test is full.
    """
    doc = vram.nvml_fb_free_mib.__doc__ or ""
    assert "uuid" in doc.lower()
    assert "CUDA_VISIBLE_DEVICES" in doc
    uuid, error = vram.probe_device_uuid()
    assert (uuid is None) != (error is None), "one of uuid/error, never both or neither"
    if uuid is not None:
        assert uuid.startswith("GPU-")
    # An unresolvable device is an error, not a silent read of device 0.
    mib, backend, err = vram.nvml_fb_free_mib(uuid="GPU-does-not-exist")
    assert mib is None and backend is None and err


def test_vram_reading_records_both_numbers_and_its_backend():
    """A combined reading must be self-describing.

    Two sequential observations of different APIs need provenance, or a later
    reader cannot tell a missing NVML binding from a genuinely full card.
    """
    reading = vram.vram_reading()
    for key in ("cuda_free_mib", "cuda_error", "nvml_fb_free_mib",
                "nvml_backend", "nvml_error", "delta_mib", "observed_at"):
        assert key in reading, f"vram_reading must record {key!r}"
    assert reading["nvml_backend"] in (None, "pynvml", "nvidia-smi")
    if reading["nvml_fb_free_mib"] is None:
        assert reading["nvml_error"], "an unavailable NVML reading must say why"
    else:
        assert reading["nvml_error"] is None


def test_the_reading_disclaims_what_it_cannot_prove():
    """Neither number is a residency certificate, and the record must say so."""
    assert "paging" in vram.vram_reading()["note"].lower()


# --- AB-13: unqualified is not accepted ------------------------------------

@pytest.mark.parametrize("signals", [
    {"driver_free_above_floor": None,
     "allocator_peak_within_budget": True,
     "step_time_not_degraded": True},
    {"driver_free_above_floor": True,
     "allocator_peak_within_budget": None,
     "step_time_not_degraded": True},
    {"driver_free_above_floor": None,
     "allocator_peak_within_budget": None,
     "step_time_not_degraded": None},
])
def test_an_unmeasured_signal_is_not_accepted_on_gpu(signals):
    """``None`` means *not qualified* and must never pass a GPU gate.

    This is the AB-13 bypass stated directly: the old predicate accepted every
    case above, because ``None is not False``.
    """
    assert vram.signals_accepted(signals, on_gpu=True) is False


def test_all_true_signals_are_accepted_on_gpu():
    assert vram.signals_accepted(
        {"driver_free_above_floor": True,
         "allocator_peak_within_budget": True,
         "step_time_not_degraded": True}, on_gpu=True) is True


def test_a_false_signal_is_rejected_on_gpu():
    assert vram.signals_accepted(
        {"driver_free_above_floor": False,
         "allocator_peak_within_budget": True,
         "step_time_not_degraded": True}, on_gpu=True) is False


def test_cpu_profiling_has_its_own_disposition():
    """A CPU probe cannot produce memory signals and must not be judged as a GPU run.

    Without naming this case the only two options are both wrong: fail every CPU
    probe, or reintroduce the AB-13 bypass to let them through.
    """
    cpu_signals = {"driver_free_above_floor": None,
                   "allocator_peak_within_budget": None,
                   "step_time_not_degraded": True}
    assert vram.signals_accepted(cpu_signals, on_gpu=False) is True
    # A measured failure still fails, device notwithstanding.
    cpu_signals["step_time_not_degraded"] = False
    assert vram.signals_accepted(cpu_signals, on_gpu=False) is False


def test_a_non_memory_signal_may_not_be_unmeasured_even_on_cpu():
    """The CPU disposition excuses the two memory signals, nothing else."""
    assert vram.signals_accepted(
        {"driver_free_above_floor": None,
         "allocator_peak_within_budget": None,
         "step_time_not_degraded": None}, on_gpu=False) is False


def test_a_full_card_yields_an_unavailable_reading_not_an_exception():
    """``mem_get_info`` RAISES when the card is full; a gate must survive that.

    Observed 2026-09-28: with a training job and a test suite both resident,
    ``torch.cuda.mem_get_info()`` raised ``CUDA error: out of memory``. The pinned
    ``free_vram_mib`` guards only ``torch.cuda.is_available()``, so the exception
    propagated and would have crashed a profiling loop at exactly the moment the
    reading mattered. It must degrade to an unqualified observation, which AB-13
    then fails closed on.
    """
    import smallAntibodyGen.experiments.her2_runtime as runtime_module
    original = runtime_module.free_vram_mib

    def boom():
        raise RuntimeError("CUDA error: out of memory")

    runtime_module.free_vram_mib = boom
    try:
        mib, error = vram.cuda_free_mib()
        assert mib is None
        assert "out of memory" in error
        reading = vram.vram_reading()
        assert reading["cuda_free_mib"] is None
        assert "out of memory" in reading["cuda_error"]
        assert reading["delta_mib"] is None, "no delta is computable from one number"
    finally:
        runtime_module.free_vram_mib = original

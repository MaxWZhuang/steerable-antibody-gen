"""Checkpoints, the RNG bundle, the cadence policy and the identity refusal.

A resume that restores the weights and nothing else produces a different
trajectory from the uninterrupted run while passing every weight comparison. So
the state this module writes is the whole of it, and the resume asserts each piece
is present rather than filling a missing one with a fresh default.

Four defects this module exists to fix, each of which was real:

* **``best_metric = inf``** made the run record unserialisable. A short run with no
  evaluation carried an infinity into ``canonical_json`` and the whole training
  record was lost at the end of a completed fit. ``None`` is the "no evaluation
  yet" value, and it round-trips.
* **the exposure ledger was checked for and never restored.** A resumed arm
  restarted every counter at zero, and the matched-exposure comparison between
  control arms then compared a resumed arm's partial count with an uninterrupted
  arm's full one.
* **the CUDA and Python RNG states were absent.** Dropout, multinomial sampling
  and the Langevin noise all draw from the device generator; restoring only the
  CPU generator diverges as soon as any of them runs.
* **``LambdaLR`` overwrote the plateau reduction on the next step.** The plateau
  handler multiplied ``param_group["lr"]`` and the scheduler recomputed it from the
  base learning rate one step later, so the recorded reduction never took effect.
  Here the plateau count is a factor *inside* the lambda, so the two schedules
  compose instead of fighting.

The cadence is the other half. Writing a full Adam state every update is
several hundred megabytes per step of I/O; the policy is last on a declared
interval, best on improvement, and declared sparse milestones, with the disk cost
computed rather than discovered at the last trajectory.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import her2_routes_data as data_lib
from . import her2_support_paths as paths
from .her2_runtime import require

CHECKPOINT_SCHEMA = "her2-routes-checkpoint/3"

#: Keys a resume refuses to proceed without. Each one is a silent divergence.
REQUIRED_KEYS = ("model", "heads", "optimizer", "scheduler", "rng", "exposure", "step",
                 "identity", "sampler_position")

#: Scalars :func:`restore` reads with ``int(...)``/``float(...)`` that are NOT in
#: :data:`REQUIRED_KEYS`. ``REQUIRED_KEYS`` is also the ``restored`` list a resume reports, so the
#: two sets stay separate; :func:`require_restorable_metadata` checks both.
RESTORE_SCALAR_KEYS = ("best_metric", "best_step", "plateau_reductions",
                       "plateau_bad_evaluations")
#: Every counter ``restore`` converts with ``int(...)``: nonnegative integers, never bools.
RESTORE_COUNTER_KEYS = ("step", "sampler_position", "best_step", "plateau_reductions",
                        "plateau_bad_evaluations")

_BOOLEAN = (bool, np.bool_)


def _is_counter(value):
    """A nonnegative integer that is not a bool: ``int(True)`` would resume at 1."""
    return (isinstance(value, numbers.Integral) and not isinstance(value, _BOOLEAN)
            and int(value) >= 0)


def _is_finite_real(value):
    """A finite real number that is not a bool. An int too large for a float is not finite here."""
    if not isinstance(value, numbers.Real) or isinstance(value, _BOOLEAN):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _is_byte_state(state):
    return (isinstance(state, torch.Tensor) and state.dtype == torch.uint8 and state.dim() == 1
            and state.numel() > 0)


def _preflight(label, check):
    """Run ``check`` on an independent generator; any failure is a refusal naming ``label``."""
    try:
        check()
    except Exception as error:
        require(False, f"The checkpoint's state for {label} is not restorable "
                       f"({type(error).__name__}: {error})")


def _generator_device_kind(generator):
    """``"cpu"``/``"cuda"``/other: the device type of a named torch generator."""
    return str(getattr(getattr(generator, "device", None), "type", "cpu"))


@dataclass
class RngBundle:
    """Every generator a step consumes, captured together.

    ``numpy`` generators are captured by their bit-generator state, torch by its
    tensor state, and CUDA per device. A bundle missing one of them restores a run
    that is identical in its weights and different in its masking.
    """

    torch_cpu: object = None
    torch_cuda: object = None
    numpy: dict = field(default_factory=dict)
    python_random: object = None
    torch_generators: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, generators):
        import random
        return cls(torch_cpu=torch.get_rng_state(),
                   # initialized-only: asking an uninitialized CUDA runtime for its RNG state
                   # creates the context, and a CPU process has no CUDA stream to record
                   torch_cuda=(torch.cuda.get_rng_state_all()
                               if torch.cuda.is_initialized() else None),
                   numpy={str(name): generator.bit_generator.state
                          for name, generator in dict(generators).items()
                          if not isinstance(generator, torch.Generator)},
                   python_random=random.getstate(),
                   torch_generators={str(name): generator.get_state()
                                     for name, generator in dict(generators).items()
                                     if isinstance(generator, torch.Generator)})

    def _validated_cuda(self):
        """The per-device CUDA states to restore (``[]`` for none), or a refusal.

        Full resume only. The destination's CUDA state is read with ``is_initialized`` alone, so
        validating never creates a device context in a CPU process. An initialized destination
        needs a nonempty per-device set of exactly its device count; an uninitialized one can
        only take an empty/absent set, because a recorded CUDA stream cannot be restored there
        and dropping it would resume a different trajectory. ``load_inference`` is the
        weights-only path for a CPU box.
        """
        initialized = torch.cuda.is_initialized()
        stored = self.torch_cuda
        if stored is None:
            require(not initialized,
                    "The checkpoint records no CUDA generator state but this process has "
                    "initialized CUDA. A resume that left the device generators where they are "
                    "would diverge from the uninterrupted run (dropout, sampling and Langevin "
                    "noise draw from them).")
            return []
        require(isinstance(stored, (list, tuple)),
                f"The checkpoint's CUDA generator state is a {type(stored).__name__}, not a list "
                "of per-device byte states")
        for index, state in enumerate(stored):
            require(_is_byte_state(state),
                    f"The checkpoint's CUDA generator state for device {index} is not a 1-D "
                    "uint8 tensor")
        if not initialized:
            require(len(stored) == 0,
                    f"The checkpoint holds {len(stored)} CUDA generator state(s) but CUDA is not "
                    "initialized in this process. A full resume would drop them and diverge from "
                    "the uninterrupted run; construct the CUDA training model before restoring, "
                    "or use load_inference for weights-only use on a CPU box.")
            return []
        live = torch.cuda.device_count()
        require(len(stored) > 0,
                "The checkpoint holds no CUDA generator state but this process has initialized "
                "CUDA; it is not a complete resume of a GPU run")
        require(len(stored) == live,
                f"The checkpoint holds {len(stored)} CUDA generator state(s) and this box has "
                f"{live}. A resume that re-seeded the device generator would diverge from the "
                "uninterrupted run while every weight check passed.")
        return list(stored)

    def restore(self, generators):
        import random
        generators = dict(generators)
        require(self.python_random is not None, "Checkpoint is missing the Python random state")
        require(self.torch_cpu is not None,
                "Checkpoint is missing the torch CPU random state. A resume that kept the live "
                "CPU generator would diverge from the uninterrupted run.")
        require(_is_byte_state(self.torch_cpu),
                "The checkpoint's torch CPU random state is not a nonempty 1-D uint8 tensor")
        cuda_states = self._validated_cuda()
        require(isinstance(self.numpy, dict) and isinstance(self.torch_generators, dict),
                "The checkpoint's named generator states must be mappings")
        ambiguous = sorted(set(self.numpy) & set(self.torch_generators))
        require(not ambiguous,
                f"The checkpoint records the generator name(s) {ambiguous} as both NumPy and torch "
                "generators; refusing to guess which state belongs to which")
        missing = sorted(set(generators) - set(self.numpy) - set(self.torch_generators))
        require(not missing,
                f"The checkpoint carries no state for the generator(s) {missing}. A resume that "
                "re-seeded the masking or the sampler would produce a different trajectory from "
                "the uninterrupted run.")
        for name, generator in generators.items():
            name = str(name)
            if isinstance(generator, torch.Generator):
                require(name in self.torch_generators,
                        f"Generator {name!r} is a torch generator here but the checkpoint "
                        "recorded it as a NumPy generator; refusing to restore a different kind")
                require(_is_byte_state(self.torch_generators[name]),
                        f"The checkpoint's state for torch generator {name!r} is not a nonempty "
                        "1-D uint8 tensor")
            else:
                require(name in self.numpy,
                        f"Generator {name!r} is a NumPy generator here but the checkpoint "
                        "recorded it as a torch generator; refusing to restore a different kind")
                state = self.numpy[name]
                require(isinstance(state, dict) and state,
                        f"The checkpoint's state for NumPy generator {name!r} is not a nonempty "
                        "bit-generator state mapping")
                kind = type(generator.bit_generator).__name__
                require(state.get("bit_generator") == kind,
                        f"The checkpoint's state for NumPy generator {name!r} is for bit generator "
                        f"{state.get('bit_generator')!r}, not {kind!r}")
        # Content preflight on independent explicitly-CPU generators (no global draw, no CUDA
        # generator is ever created), so a malformed CPU payload is refused before any global
        # generator moves. CUDA payloads (global and named) have a different byte layout that a
        # CPU generator cannot interpret; they get the structural, presence and topology checks
        # above only, and arbitrary malformed CUDA bytes cannot be fully validated on CPU.
        _preflight("Python random", lambda: random.Random().setstate(self.python_random))
        _preflight("torch CPU",
                   lambda: torch.Generator(device="cpu").set_state(self.torch_cpu.cpu()))
        for name, generator in generators.items():
            name = str(name)
            if isinstance(generator, torch.Generator):
                kind = _generator_device_kind(generator)
                require(kind in ("cpu", "cuda"),
                        f"Torch generator {name!r} is on a {kind!r} device; only CPU and CUDA "
                        "generators can be restored")
                if kind == "cpu":
                    _preflight(f"torch generator {name!r}",
                               lambda n=name: torch.Generator(device="cpu").set_state(
                                   self.torch_generators[n].cpu()))
                else:
                    require(torch.cuda.is_initialized() and cuda_states,
                            f"Torch generator {name!r} is a CUDA generator but the checkpoint "
                            "or this process has no initialized CUDA state; a CUDA generator is "
                            "not restorable from a CPU-only record")
            else:
                def check(n=name, g=generator):
                    clone = type(g.bit_generator)()
                    clone.state = self.numpy[n]
                _preflight(f"NumPy generator {name!r}", check)
        # presence, kind, CPU contents and CUDA structure/topology are checked: only now touch the
        # global generators. A CUDA setter failing after this point is not rolled back.
        random.setstate(self.python_random)
        torch.set_rng_state(self.torch_cpu.cpu())
        if cuda_states:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        for name, generator in generators.items():
            if isinstance(generator, torch.Generator):
                generator.set_state(self.torch_generators[str(name)].cpu())
            else:
                generator.bit_generator.state = self.numpy[str(name)]
        return True

    def document(self):
        return {"torch_cpu": self.torch_cpu is not None,
                "torch_cuda_devices": (0 if self.torch_cuda is None else len(self.torch_cuda)),
                "numpy_generators": sorted(self.numpy),
                "python_random": self.python_random is not None,
                "torch_generators": sorted(self.torch_generators)}


@dataclass
class TrainingState:
    """Everything a resume must restore. Missing any of it is a silent divergence."""

    step: int = 0
    best_metric: object = None
    best_step: int = 0
    plateau_reductions: int = 0
    plateau_bad_evaluations: int = 0
    sampler_position: int = 0
    mask_rng: object = None
    sampler_rng: object = None
    decode_rng: object = None
    energy_rng: object = None
    ledger: object = field(default_factory=data_lib.ExposureLedger)
    energy_buffer: object = None
    energy_phase: str = ""
    history: list = field(default_factory=list)

    def generators(self):
        out = {}
        for name in ("mask_rng", "sampler_rng", "decode_rng", "energy_rng"):
            generator = getattr(self, name)
            if generator is not None:
                out[name] = generator
        return out

    def improved(self, metric, *, delta):
        """Is ``metric`` a real improvement? ``None`` means nothing has been measured.

        Written out because ``inf`` as the initial best both broke serialisation and
        made the *first* evaluation an automatic improvement regardless of its
        value, which then wrote a ``best.pt`` nobody had compared against anything.
        """
        if metric is None:
            return False
        if self.best_metric is None:
            return True
        return float(metric) < float(self.best_metric) - float(delta)

    def document(self):
        return {"step": int(self.step),
                "best_metric": (None if self.best_metric is None else float(self.best_metric)),
                "best_step": int(self.best_step),
                "plateau_reductions": int(self.plateau_reductions),
                "plateau_bad_evaluations": int(self.plateau_bad_evaluations),
                "sampler_position": int(self.sampler_position),
                "energy_phase": self.energy_phase or None,
                "exposure": self.ledger.document()}


def training_identity(*, route, arm, seed, config_sha256, snapshot_sha256, external_sha256,
                      condition_sha256, split_sha256, schedule_sha256=None, native_sha256=None):
    """What a checkpoint was produced by. A resume against a different one is refused.

    Data identity is in the list on purpose. Weights that resumed under a changed
    source closure, a changed condition vocabulary or a changed split are weights
    whose recorded provenance is wrong, and nothing in the tensors says so.

    ``native_sha256`` binds a native pretrained base: the pins digest, the base file
    digests, the LoRA settings, the optimizer policy and the diffusion settings. It is
    present in the record only when given, so the three original routes' identities --
    and every checkpoint already written under them -- are unchanged.
    """
    record = {"route": str(route), "arm": str(arm), "seed": int(seed),
              "config_sha256": config_sha256, "snapshot_sha256": snapshot_sha256,
              "external_sha256": external_sha256,
              "condition_vocabulary_sha256": condition_sha256,
              "split_sha256": split_sha256, "schedule_sha256": schedule_sha256}
    if native_sha256 is not None:
        record["native_sha256"] = native_sha256
    record["identity_sha256"] = paths.digest_document(record)
    return record


def require_identity(recorded, current, *, path):
    """Refuse a resume whose identity differs, naming the fields that moved."""
    recorded = dict(recorded or {})
    current = dict(current or {})
    differing = sorted(key for key in set(recorded) | set(current)
                       if key != "identity_sha256" and recorded.get(key) != current.get(key))
    require(not differing,
            f"{path} was produced under a different identity: {differing} changed. Resuming "
            "would attribute the recorded numbers to code, sources or a split that did not "
            "produce them. Declare a new revision; the superseded root stays read-only.")
    return True


def require_restorable_metadata(payload, *, path, generators):
    """Refuse an obviously unrestorable FULL checkpoint from its CPU metadata alone.

    :func:`restore` finds these defects only after the caller has built a model and seeded the
    global generators. This reads the payload and nothing else: no model, optimizer, directory or
    generator is created or changed, and no CUDA state is consulted (a resume builds its device
    model first, so the process's CUDA state is only meaningful to ``restore``). ``generators``
    maps each generator name the resuming run will restore to ``"numpy"`` or ``"torch"``.

    Checked: the schema, the required keys, the scalar fields ``restore`` indexes after loading
    state (``best_metric`` is ``None`` or a finite real; ``step``, ``sampler_position``,
    ``best_step`` and both plateau counters are nonnegative integers and never bools, which
    ``int(...)`` would silently coerce), that the RNG state is an :class:`RngBundle` holding the
    Python and torch CPU states (restorable on independent generators), the structure of any
    recorded CUDA states, and that every named generator is present as the right kind with a
    well-formed state. NOT checked: anything that needs the live objects (parameter names and
    shapes, optimizer tensors, a named generator's content), which ``restore`` still does.
    """
    import random
    require(isinstance(payload, dict), f"{path} is not a checkpoint payload")
    require(payload.get("schema_version") == CHECKPOINT_SCHEMA,
            f"{path} is a {payload.get('schema_version')!r} checkpoint, not {CHECKPOINT_SCHEMA}")
    for key in REQUIRED_KEYS:
        require(key in payload,
                f"{path} is missing {key!r}. A resume that restored the weights and re-seeded the "
                "masking would diverge from the uninterrupted run while passing every weight "
                "check, so a partial checkpoint is refused rather than completed with defaults.")
    for key in RESTORE_SCALAR_KEYS:
        require(key in payload,
                f"{path} is missing {key!r}. restore reads it after loading state, so a "
                "checkpoint without it would fail midway through a resume; it is refused here, "
                "before anything is built.")
    require(payload["best_metric"] is None or _is_finite_real(payload["best_metric"]),
            f"{path}: 'best_metric' is {payload['best_metric']!r} "
            f"({type(payload['best_metric']).__name__}); it must be None (no evaluation yet) or a "
            "finite real number, not a bool, NaN or an infinity.")
    for key in RESTORE_COUNTER_KEYS:
        require(_is_counter(payload[key]),
                f"{path}: {key!r} is {payload[key]!r} ({type(payload[key]).__name__}); it must be "
                "a nonnegative integer and not a bool, because restore converts it with int() and "
                "would resume a different trajectory.")
    bundle = payload["rng"]
    require(isinstance(bundle, RngBundle),
            f"{path} carries a {type(bundle).__name__} where the RNG bundle belongs")
    require(bundle.python_random is not None,
            f"{path}: the checkpoint is missing the Python random state")
    require(bundle.torch_cpu is not None,
            f"{path}: the checkpoint is missing the torch CPU random state. A resume that kept "
            "the live CPU generator would diverge from the uninterrupted run.")
    require(_is_byte_state(bundle.torch_cpu),
            f"{path}: the checkpoint's torch CPU random state is not a nonempty 1-D uint8 tensor")
    cuda = bundle.torch_cuda
    require(cuda is None or (isinstance(cuda, (list, tuple))
                             and all(_is_byte_state(state) for state in cuda)),
            f"{path}: the checkpoint's CUDA generator state is not a list of 1-D uint8 tensors")
    require(isinstance(bundle.numpy, dict) and isinstance(bundle.torch_generators, dict),
            f"{path}: the checkpoint's named generator states must be mappings")
    ambiguous = sorted(set(bundle.numpy) & set(bundle.torch_generators))
    require(not ambiguous,
            f"{path} records the generator name(s) {ambiguous} as both NumPy and torch "
            "generators; refusing to guess which state belongs to which")
    wanted = {str(name): kind for name, kind in dict(generators).items()}
    require(all(kind in ("numpy", "torch") for kind in wanted.values()),
            f"the generator kinds must be 'numpy' or 'torch', got {wanted}")
    missing = sorted(set(wanted) - set(bundle.numpy) - set(bundle.torch_generators))
    require(not missing,
            f"{path}: the checkpoint carries no state for the generator(s) {missing}. A resume "
            "that re-seeded the masking or the sampler would produce a different trajectory from "
            "the uninterrupted run.")
    numpy_kind = type(np.random.default_rng(0).bit_generator).__name__
    for name, kind in wanted.items():
        if kind == "torch":
            require(name in bundle.torch_generators,
                    f"{path}: generator {name!r} is a torch generator here but the checkpoint "
                    "recorded it as a NumPy generator; refusing to restore a different kind")
            require(_is_byte_state(bundle.torch_generators[name]),
                    f"{path}: the checkpoint's state for torch generator {name!r} is not a "
                    "nonempty 1-D uint8 tensor")
        else:
            require(name in bundle.numpy,
                    f"{path}: generator {name!r} is a NumPy generator here but the checkpoint "
                    "recorded it as a torch generator; refusing to restore a different kind")
            state = bundle.numpy[name]
            require(isinstance(state, dict) and state,
                    f"{path}: the checkpoint's state for NumPy generator {name!r} is not a "
                    "nonempty bit-generator state mapping")
            require(state.get("bit_generator") == numpy_kind,
                    f"{path}: the checkpoint's state for NumPy generator {name!r} is for bit "
                    f"generator {state.get('bit_generator')!r}, not {numpy_kind!r}")
    _preflight("Python random", lambda: random.Random().setstate(bundle.python_random))
    _preflight("torch CPU",
               lambda: torch.Generator(device="cpu").set_state(bundle.torch_cpu.cpu()))
    return True


def _model_payload(model):
    """``(state, meta)``. ``state_dict`` unless the model declares the explicit protocol.

    A model that stores only its adapters says so through ``checkpoint_state`` /
    ``load_checkpoint_state``. ``state_dict`` itself is never overridden, so a nested
    module cannot silently drop keys; the protocol is opt-in and carries the base identity.
    """
    if hasattr(model, "checkpoint_state"):
        return model.checkpoint_state()
    return model.state_dict(), None


def _load_model(model, state, meta, *, strict, path):
    if hasattr(model, "load_checkpoint_state"):
        require(meta is not None,
                f"{path} carries no model_format block, but this model restores through the "
                "explicit checkpoint protocol; a plain state_dict is not accepted for it.")
        return model.load_checkpoint_state(state, meta, strict=bool(strict))
    require(meta is None,
            f"{path} was written by a model with a {meta.get('format') if meta else None!r} "
            "checkpoint protocol and this model has none; it will not be loaded as a plain "
            "state_dict.")
    return model.load_state_dict(state, strict=bool(strict))


def save(path, *, model, heads, optimizer, scheduler, state, identity, extra=None):
    """Atomic write of the whole state. An exclusive temporary then replace, never in place."""
    model_state, model_format = _model_payload(model)
    payload = {"schema_version": CHECKPOINT_SCHEMA, "step": int(state.step),
               "model": model_state,
               "heads": None if heads is None else heads.state_dict(),
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "scheduler": None if scheduler is None else scheduler.state_dict(),
               "rng": RngBundle.capture(state.generators()),
               "exposure": state.ledger.state(),
               "best_metric": (None if state.best_metric is None else float(state.best_metric)),
               "best_step": int(state.best_step),
               "plateau_reductions": int(state.plateau_reductions),
               "plateau_bad_evaluations": int(state.plateau_bad_evaluations),
               "sampler_position": int(state.sampler_position),
               "energy_buffer": state.energy_buffer,
               "energy_phase": state.energy_phase,
               "history": list(state.history),
               "identity": dict(identity)}
    if model_format is not None:
        payload["model_format"] = model_format
    if extra:
        payload.update(dict(extra))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # An exclusively created temporary, never ``<name>.tmp``: a hard link or symbolic link planted
    # at a fixed temporary name would be written through, truncating a file in another flight.
    with paths.atomic_open(target, "wb") as stream:
        torch.save(payload, stream)
    return target


def restore(path, *, model, heads, optimizer, scheduler, state, identity, strict=True):
    """Restore the whole state, or refuse. Returns the state and the load record."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    require(payload.get("schema_version") == CHECKPOINT_SCHEMA,
            f"{path} is a {payload.get('schema_version')!r} checkpoint, not {CHECKPOINT_SCHEMA}")
    for key in REQUIRED_KEYS:
        require(key in payload,
                f"{path} is missing {key!r}. A resume that restored the weights and re-seeded the "
                "masking would diverge from the uninterrupted run while passing every weight "
                "check, so a partial checkpoint is refused rather than completed with defaults.")
    if identity is not None:
        require_identity(payload["identity"], identity, path=path)
    _load_model(model, payload["model"], payload.get("model_format"), strict=strict, path=path)
    if heads is not None:
        require(payload["heads"] is not None,
                f"{path} holds no observation-head state and this run has heads. Restoring the "
                "trunk alone would resume with a freshly initialised head on a trained trunk.")
        heads.load_state_dict(payload["heads"], strict=bool(strict))
    if optimizer is not None:
        require(payload["optimizer"] is not None, f"{path} has no optimizer state for training resume")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        require(payload["scheduler"] is not None, f"{path} has no scheduler state for training resume")
        scheduler.load_state_dict(payload["scheduler"])
    state.step = int(payload["step"])
    state.best_metric = (None if payload["best_metric"] is None
                         else float(payload["best_metric"]))
    state.best_step = int(payload["best_step"])
    state.plateau_reductions = int(payload["plateau_reductions"])
    state.plateau_bad_evaluations = int(payload["plateau_bad_evaluations"])
    state.sampler_position = int(payload["sampler_position"])
    state.energy_buffer = payload.get("energy_buffer")
    state.energy_phase = str(payload.get("energy_phase") or "")
    state.history = list(payload.get("history") or [])
    state.ledger = data_lib.ExposureLedger.restore(payload["exposure"])
    bundle = payload["rng"]
    require(isinstance(bundle, RngBundle),
            f"{path} carries a {type(bundle).__name__} where the RNG bundle belongs")
    bundle.restore(state.generators())
    return state, {"path": str(path), "step": state.step,
                   "restored": list(REQUIRED_KEYS),
                   "rng": bundle.document(),
                   "exposure_rows": int(state.ledger.totals["microbatch_rows"]),
                   "rule": ("weights, heads, optimizer, scheduler, every generator, the sampler "
                            "position and the exposure ledger. A missing piece is refused, not "
                            "defaulted.")}


def load_inference(path, *, model, heads, identity, payload=None):
    """Strict weights-only inference load; leave all training and caller RNG state alone."""
    if payload is None:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    require(payload.get("schema_version") == CHECKPOINT_SCHEMA,
            f"{path} has an incompatible checkpoint schema")
    require_identity(payload["identity"], identity, path=path)
    _load_model(model, payload["model"], payload.get("model_format"), strict=True, path=path)
    if heads is not None:
        require(payload.get("heads") is not None, f"{path} has no trained head state")
        heads.load_state_dict(payload["heads"], strict=True)
    return {"path": str(path), "step": int(payload["step"]), "trained": True,
            "training_identity": payload["identity"], "restored": ["model", "heads"],
            "rng_restored": False}


@dataclass
class Cadence:
    """Which checkpoints are written and how often. The disk cost is computed.

    ``save_every`` is the load-bearing number. Writing a full Adam state every
    update is hundreds of megabytes of I/O per step and it dominates the step time
    of a small model; writing it once per interval bounds the replayed work to that
    interval instead.
    """

    save_every: int = 100
    milestones: tuple = ()
    keep_best: bool = True
    bytes_per_checkpoint: int = 0

    def is_milestone(self, step):
        return int(step) in {int(value) for value in self.milestones}

    def should_save_last(self, step, *, total):
        step, total = int(step), int(total)
        return step >= total or step % max(1, int(self.save_every)) == 0

    def retained(self):
        return int(self.keep_best) + 1 + len(tuple(self.milestones))

    def document(self, *, disk_budget_bytes=None):
        total = int(self.bytes_per_checkpoint) * self.retained()
        record = {"save_every": int(self.save_every),
                  "milestones": [int(value) for value in self.milestones],
                  "keep_best": bool(self.keep_best),
                  "checkpoints_retained": self.retained(),
                  "estimated_bytes": total, "estimated_gib": total / 1024 ** 3,
                  "rule": ("last on the declared interval, best on improvement, declared sparse "
                           "milestones. A full optimizer copy per update is hundreds of megabytes "
                           "of I/O per step and it dominates a small model's step time.")}
        if disk_budget_bytes is not None:
            record["disk_budget_gib"] = int(disk_budget_bytes) / 1024 ** 3
            record["fits"] = total <= int(disk_budget_bytes)
            require(record["fits"],
                    f"The retention policy needs {total / 1024 ** 3:.1f} GiB against a declared "
                    f"budget of {int(disk_budget_bytes) / 1024 ** 3:.1f} GiB. Reduce the "
                    "milestones deliberately rather than discovering it at the last trajectory.")
        return record


def warmup_plateau_lambda(state, *, warmup, plateau_factor):
    """One multiplier that composes warm-up with the plateau reductions.

    The plateau count is read from ``state`` **inside** the lambda, so a reduction
    survives the next ``scheduler.step()``. Multiplying ``param_group["lr"]``
    directly and leaving ``LambdaLR`` to recompute from the base rate is how a
    recorded reduction silently never happened.
    """
    warmup = max(1, int(warmup))
    factor = float(plateau_factor)

    def schedule(step):
        return min(1.0, (int(step) + 1) / warmup) * (factor ** int(state.plateau_reductions))
    return schedule


def resumed_trajectory_report(uninterrupted, resumed, *, tolerance=0.0):
    """Do an uninterrupted run and a resumed one agree step for step? Measured.

    Compares the per-step loss and gradient-norm histories from the step the resume
    began at. An equality check on the final weights alone passes when the resumed
    run took a different path to a similar place, which is the case a re-seeded
    mask RNG produces.

    A pass needs genuine coverage: the resumed history must be a nonempty, uninterrupted
    suffix of the reference that ends at the reference's last recorded step (a missing
    reference prefix is fine -- a resume starts part-way -- but an interior reference step the
    resumed run skipped, or a resumed step the reference never took, is not). Duplicate or
    non-integer steps, missing or non-finite ``loss`` / ``gradient_norm`` and a non-finite or
    negative tolerance are invalid evidence and yield a failed report with the reasons in
    ``problems``; nothing non-finite reaches the returned (JSON) document.
    """
    problems = []
    tolerance_value = _finite_float(tolerance)
    tolerance_ok = tolerance_value is not None and tolerance_value >= 0.0
    if not tolerance_ok:
        problems.append(f"tolerance {_short_repr(tolerance)} is not a finite non-negative number")
        tolerance_value = 0.0
    by_step = []
    for label, history in (("uninterrupted", uninterrupted), ("resumed", resumed)):
        table, found = _trajectory_table(history, label)
        by_step.append(table)
        problems.extend(found)
    reference, tail = by_step
    steps = sorted(set(reference) & set(tail))
    if not reference:
        problems.append("the uninterrupted history holds no usable step")
    if not tail:
        problems.append("the resumed history holds no usable step")
    if reference and tail:
        foreign = sorted(set(tail) - set(reference))
        if foreign:
            problems.append(f"resumed steps {foreign[:5]} are not in the uninterrupted history")
        ordered = sorted(reference)
        expected = [step for step in ordered if step >= min(tail)]
        if sorted(tail) != expected:
            problems.append(
                f"the resumed steps {sorted(tail)[:5]}... are not the uninterrupted suffix "
                f"{expected[:5]}... ending at step {ordered[-1]}; a resumed run must cover every "
                "reference step from where it began through the last one")
    worst, offenders, any_overflow = 0.0, [], False
    if not problems:
        for step in steps:
            for field_name in ("loss", "gradient_norm"):
                a, b = reference[step][field_name], tail[step][field_name]
                difference = abs(a - b)
                overflow = not math.isfinite(difference)
                if overflow:
                    # the true difference is not representable: say so, never invent a number
                    any_overflow = True
                    difference = None
                else:
                    worst = max(worst, difference)
                if overflow or difference > tolerance_value:
                    offenders.append({"step": step, "field": field_name, "left": a, "right": b,
                                      "difference": difference,
                                      "difference_overflow": overflow})
    return {"compared_steps": len(steps) if not problems else 0,
            "max_abs_difference": None if any_overflow else worst,
            "difference_overflow": any_overflow,
            "tolerance": tolerance_value if tolerance_ok else None,
            "identical": not problems and not offenders,
            "offenders": offenders[:10], "problems": problems[:10],
            "asserts": ("the resumed trajectory, not only the resumed weights. A run that "
                        "re-seeded its masking reaches a similar place by a different path and a "
                        "final-weight comparison does not see it.")}


def _finite_float(value):
    """``float(value)`` once, or ``None`` if not a real number, unrepresentable or non-finite."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return converted if math.isfinite(converted) else None


def _short_repr(value):
    """A bounded repr: an enormous int must not blow up (or bloat) the problem text."""
    try:
        text = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    return text if len(text) <= 60 else f"<{type(value).__name__} {text[:20]}...>"


def _trajectory_table(history, label):
    """``({step: {loss, gradient_norm}}, problems)`` for one history, refusing bad evidence."""
    problems, table, seen = [], {}, set()
    if not isinstance(history, (list, tuple)):
        return {}, [f"the {label} history is a {type(history).__name__}, not a list of entries"]
    for position, entry in enumerate(history):
        step = entry.get("step") if isinstance(entry, dict) else None
        if not isinstance(step, numbers.Integral) or isinstance(step, bool):
            problems.append(f"{label} entry {position} has no integer step ({step!r})")
            continue
        step = int(step)
        if step in seen:
            problems.append(f"{label} history repeats step {step}")
            continue
        seen.add(step)
        values = {}
        for field_name in ("loss", "gradient_norm"):
            value = entry.get(field_name)
            converted = _finite_float(value)
            if converted is None:
                problems.append(f"{label} step {step} has no finite {field_name} "
                                f"({_short_repr(value)})")
                break
            values[field_name] = converted
        else:
            table[step] = values
    return table, problems

"""The concrete runtime: model construction, the training loop, profiling, resume.

Imported lazily by ``her2_routes_campaign.build_services`` so orchestration stays
testable without torch. The loop is deliberately plain -- one optimizer, gradient
accumulation to the declared effective batch, plateau learning-rate reductions, best
and last checkpoints plus declared milestones -- and the interesting parts are the
things it refuses to do:

* **It does not infer that a configuration fits from the absence of an OOM.** This
  box spills CUDA allocations to system RAM: a 5.6 GB configuration returned
  ``ok=True`` on a 4 GB card. So :meth:`FlightServices.profile` records the driver's
  free-VRAM reading **and** the median step time **and** the allocator's peak, and a
  configuration is accepted only when all three agree.
* **It does not report a forecast it did not measure.** Every per-unit cost comes
  from a measured microbatch on this box, and the exposure ledger's six counters are
  multiplied separately -- there is no path that turns "20,000 updates" into an
  exposure count.
* **It does not resume a changed source closure.** Every stage re-verifies the
  repository snapshot and the external source table before it does anything
  expensive.
* **It does not restore a partial state.** A resume that restored weights but
  re-seeded the mask RNG would diverge from the uninterrupted run while passing
  every weight check, so the checkpoint carries the sampler, mask RNG, scheduler,
  energy buffer and exposure ledger, and the resume asserts each is present.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import her2_routes_admission as admission
from . import her2_routes_capabilities as capabilities
from . import her2_routes_checkpoints as checkpoints
from . import her2_routes_controls as controls
from . import her2_routes_data as data_lib
from . import her2_routes_energy_contract as energy_contract
from . import her2_routes_lineage as lineage
from . import her2_routes_objectives as objectives_lib
from . import her2_routes_spec as spec
from . import her2_routes_tasks as tasks_lib
from . import her2_support_paths as paths
from .her2_routes_conditions import BIN_NULL, Condition
from .her2_routes_env import (
    NATIVE_PACKAGE_ROOT,
    NATIVE_WEIGHTS_ROOT,
    assert_no_home_cache,
)
from .her2_routes_serial import (
    ABLANG_PAD_ID,
    PIGGEN_MASK_ID,
    her2_core_edit_map,
)
from .her2_runtime import require
from .her2_vram import memory_signals, signals_accepted, vram_reading

ROUTE_MASKED = "ablang2_masked_editor"
ROUTE_CAUSAL = "piggen_causal_infiller"
ROUTE_ENERGY = "dwjs_energy"
ROUTES = (ROUTE_MASKED, ROUTE_CAUSAL, ROUTE_ENERGY)

#: The native pretrained DPLM routes. They are OPTIONAL: ``ROUTES`` is unchanged, and a
#: config lists one in ``routes.enabled`` only through an explicit opt-in revision (the
#: current three-route config does not). They speak the pipeline's public (AbLang) token
#: ids and translate once, inside the model.
ROUTE_DPLM_150M = "dplm_150m_masked"
ROUTE_DPLM_650M_LORA = "dplm_650m_lora_masked"
OPTIONAL_ROUTES = (ROUTE_DPLM_150M, ROUTE_DPLM_650M_LORA)
DPLM_ROUTES = OPTIONAL_ROUTES
KNOWN_ROUTES = ROUTES + OPTIONAL_ROUTES
#: The routes whose model is a masked editor over the public AbLang serialisation.
MASKED_LIKE_ROUTES = (ROUTE_MASKED,) + DPLM_ROUTES

#: Keys an ``optimizer`` block may carry. Anything else is refused rather than ignored.
ADAFACTOR_KEYS = ("name", "learning_rate", "relative_step", "scale_parameter", "warmup_init",
                  "beta1", "eps", "clip_threshold", "decay_rate", "weight_decay")
ADAMW_KEYS = ("name", "learning_rate", "betas", "weight_decay")

#: The training state and the checkpoint machinery live in
#: ``her2_routes_checkpoints`` so the resume-correctness tests can exercise them
#: without a model. Re-exported because the callers in this module and the tests
#: both name it.
TrainingState = checkpoints.TrainingState

#: The generators an energy resume restores, by kind: ``fresh_state``'s three NumPy generators
#: plus the energy route's torch generator that ``train_energy`` attaches. The resume checks a
#: checkpoint's RNG bundle against this BEFORE a state exists (``checkpoints.
#: require_restorable_metadata``); a test pins it to the generators a real state carries.
ENERGY_RNG_INVENTORY = {"mask_rng": "numpy", "sampler_rng": "numpy", "decode_rng": "numpy",
                        "energy_rng": "torch"}


class ShapeUnavailable(ValueError):
    """A microbatch size the enrolled pools cannot fill with representative rows.

    Raised by :meth:`FlightServices.profile_draws`. The profile records the candidate as rejected
    with this reason and carries on to the next one; it never profiles a different shape under
    the requested size's name. ``notes`` carries what the draw learned on the way.
    """

    def __init__(self, message, notes=None):
        super().__init__(message)
        self.notes = dict(notes or {})


def assign_summary_bin(fitted, row):
    """The frozen quality cut applied to one construct summary, or ``None``.

    A summary with an exact scalar value (every replicate an exact finite point) is binned by
    that point. Any other summary -- censored, mixed-censored, or with room around its value -- is
    binned by its WHOLE replicate interval through :func:`~.her2_routes_conditions.assign_bin`,
    which answers only when the interval lies inside one bin. The earlier call passed the
    uncensored-only mean of a mixed row as its value, so a row whose floor replicate straddled a
    cut was placed by the replicates that happened to be above it.
    """
    from .her2_routes_conditions import assign_bin
    exact = row.exact_value
    if exact is not None:
        return assign_bin(fitted, value=exact)
    low, high = row.interval
    if low is None and high is None:
        return None
    for bound in (low, high):
        if bound is not None and not math.isfinite(float(bound)):
            return None
    return assign_bin(fitted, interval=(low, high))


def exact_value_groups(rows):
    """``{exact value: [rows]}``: the rows a SCALAR ordering may use, grouped by their value.

    The enrichment ranking chooses its pairs by scalar order, so only a row with an exact
    value -- every replicate an exact finite point -- can supply one. ``row.value`` is the mean of
    the uncensored replicates, which for a floor replicate plus two points is a number the assay
    never produced. A row without an exact value is not in the result, and it is NOT dropped from
    the head pool: its per-replicate intervals still train the likelihood.
    """
    groups = {}
    for row in rows:
        exact = row.exact_value
        if exact is not None and np.isfinite(exact):
            groups.setdefault(float(exact), []).append(row)
    return groups


def fit_condition_bins(summaries, *, minimum_support, fit):
    """``(thresholds, bins, support)``: frozen quality cuts fitted on TRAIN exact values.

    Per ``(family, assay, parent)`` population: the cuts are fitted on the EXACT scalar values of
    the TRAIN rows (``fit`` is ``fit_bin_thresholds``). ``row.value`` is the uncensored-only mean,
    which for a construct with a floor replicate is a number the assay never produced and would
    move a cut. The rows left out of a fit are counted by their own reason, per population, in
    ``support`` beside the cuts they did not shape.

    The cuts are fitted on TRAIN rows and then APPLIED to every row of the population, which is the
    correct direction: a held-out row may be binned by a rule it did not help define. A row with an
    exact value is binned by that point; a censored, mixed or nondegenerate row is binned by its
    WHOLE interval, and only when the interval fits inside one bin
    (:func:`assign_summary_bin`). Such rows are never dropped from the head and whole-window pools:
    an unbinnable one trains at ``bin=null``.
    """
    thresholds, bins, support, by_population = {}, {}, {}, {}
    for row in summaries:
        if not row.is_training or row.value is None:
            continue
        by_population.setdefault((row.family, row.assay, row.parent), []).append(row)
    for (family, assay, parent), group in sorted(by_population.items()):
        exact = [row.exact_value for row in group if row.exact_value is not None]
        excluded = {}
        for row in group:
            if row.exact_value is None:
                excluded[row.exact_status] = excluded.get(row.exact_status, 0) + 1
        fitted = fit(exact, family=family, assay=assay, parent=parent, scale=group[0].scale,
                     minimum_support=int(minimum_support))
        key = f"{family}::{assay}::{parent}"
        thresholds[key] = None if fitted is None else fitted.document()
        support[key] = {"train_rows": len(group), "exact_rows_fitted": len(exact),
                        "excluded_by_exact_status": dict(sorted(excluded.items())),
                        "fitted": fitted is not None}
        if fitted is None:
            continue
        for row in summaries:
            if (row.family, row.assay, row.parent) != (family, assay, parent):
                continue
            name = assign_summary_bin(fitted, row)
            if name is not None:
                bins[row.population_key] = name
    return thresholds, bins, support


HER2_MEASURED_SOURCE = "alphabind_her2_validation"


def _digest_lines(lines):
    """SHA-256 over newline-joined lines, in the order given."""
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


#: What :func:`contrast_input_binding` binds. Bump it when that set changes, so a digest written
#: under an older definition can never be mistaken for one under the current definition.
CONTRAST_INPUT_DIGEST_VERSION = "contrast-pool-input/1"


def _digest_scalar(value):
    """Exact text for one number or ``None``: ``float.hex`` round-trips every bit."""
    if value is None:
        return "None"
    try:
        return float(value).hex()
    except (TypeError, ValueError):
        return repr(value)


def _contrast_row_record(row):
    """Everything about one summary that a contrast enumeration, or a stream built on its endpoints,
    can read -- complete, never truncated.

    ``ConstructSummary.document()`` is NOT this record: it cuts the identity at 32 characters,
    reports the sites as counts and keeps six replicates, so two different pools can share a
    document. Here the full heavy and light chains, every site coordinate of the construct, the
    region and the mutated sites, and every replicate's value, bounds, censoring, uncertainty, id
    and label are all present.
    """
    construct = row.construct
    return (
        construct.heavy, construct.light,
        tuple((str(chain), int(index)) for chain, index in construct.sites),
        str(construct.source), str(construct.record_id),
        str(row.study), str(row.assay), str(row.target), str(row.parent), str(row.family),
        str(row.kind), str(row.scale), str(row.source), str(row.partition),
        repr(row.ordinal_class),
        repr(tuple(tuple(site) for site in row.region)),
        repr(tuple(tuple(site) for site in row.mutated_sites)),
        tuple((_digest_scalar(item.value), _digest_scalar(item.lower),
               _digest_scalar(item.upper), str(item.censoring), _digest_scalar(item.uncertainty),
               repr(item.replicate), str(item.measurement_id)) for item in row.measurements),
        tuple(sorted((str(key), repr(value)) for key, value in dict(row.raw).items())))


def acquired_population_keys(subset):
    """The validated ``{population_key}`` of an acquired subset of summaries.

    A population key is ``(full heavy|light identity, canonical study, canonical assay, target)``:
    the unit one :class:`ConstructSummary` is. Anything that is not four non-empty text parts, or
    that names the same unit twice, is refused -- a duplicate would make the declared label count
    and the number of selected units disagree.
    """
    keys = set()
    for row in subset:
        key = tuple(row.population_key)
        require(len(key) == 4 and all(isinstance(part, str) and part for part in key),
                f"a population key is (identity, study, assay, target), four non-empty strings; "
                f"got {key!r}")
        require(key not in keys,
                f"the subset names the population {key[1:]} of one construct twice; an acquired "
                "unit is one (construct, study, assay, target) summary")
        keys.add(key)
    require(keys, "an acquired subset needs at least one summary")
    return keys


def contrast_input_binding(rows, *, settings, parents):
    """The versioned digest of EVERYTHING a contrast enumeration was given.

    ``rows`` is the already-materialised, ORDERED pool. The order is bound because the enumeration
    consumes one seeded generator group by group in pool order, so the same rows in another order
    are a different draw; the digest therefore does not pretend to be order invariant. ``settings``
    is the complete set of enumeration arguments (caps, seed, flags) and ``parents`` the declared
    parent-sequence map, both of which change what is enumerated.
    """
    digest = hashlib.sha256()
    digest.update(CONTRAST_INPUT_DIGEST_VERSION.encode("utf-8"))
    digest.update(b"\n")
    digest.update(repr(sorted((str(key), repr(value)) for key, value in dict(settings).items()))
                  .encode("utf-8"))
    digest.update(b"\n")
    digest.update(repr(sorted((str(name), repr(tuple(pair))) for name, pair in
                              dict(parents).items())).encode("utf-8"))
    digest.update(b"\n")
    count = 0
    for row in rows:
        digest.update(repr(_contrast_row_record(row)).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return {"version": CONTRAST_INPUT_DIGEST_VERSION, "sha256": digest.hexdigest(),
            "records": count, "settings": dict(settings),
            "order_sensitive": True, "parent_sequences": len(dict(parents))}


def her2_training_population(summaries, closure):
    """``(eligible, counts)``: the HER2 TRAINING constructs that clear the closure.

    ``eligible`` is ``[(window, summary), ...]`` in summary order and is independent of whether a
    summary has an exact value -- a censored or mixed-censored construct still trains the head
    and whole-window tasks, so it belongs to the adaptation population. The exact quantitative
    subset is the ones whose ``exact_value`` is not ``None``; ``counts`` says how many of each,
    how the rest are excluded (``excluded_by_exact_status``) and what the split and the closure
    removed, so nothing leaves the population without being counted.
    """
    eligible = []
    counts = {"source_rows": 0, "not_training_partition": 0, "closure_refused": 0,
              "eligible": 0, "exact": 0, "excluded_by_exact_status": {}}
    for row in summaries:
        if row.source != HER2_MEASURED_SOURCE:
            continue
        counts["source_rows"] += 1
        if not row.is_training:
            counts["not_training_partition"] += 1
            continue
        window = row.construct.residues(row.construct.sites)
        if closure.violates(window=window, identity=row.identity) is not None:
            counts["closure_refused"] += 1
            continue
        eligible.append((window, row))
        if row.exact_value is None:
            status = row.exact_status
            counts["excluded_by_exact_status"][status] = \
                counts["excluded_by_exact_status"].get(status, 0) + 1
        else:
            counts["exact"] += 1
    counts["eligible"] = len(eligible)
    counts["excluded_by_exact_status"] = dict(sorted(counts["excluded_by_exact_status"].items()))
    return eligible, counts


def _json_safe(node):
    """A JSON-compatible copy: numpy coerced, a nonfinite float becomes ``None`` (never NaN)."""
    if isinstance(node, dict):
        return {str(key): _json_safe(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_json_safe(value) for value in node]
    if isinstance(node, np.ndarray):
        return _json_safe(node.tolist())
    if isinstance(node, np.generic):
        return _json_safe(node.item())
    if isinstance(node, float):
        return node if math.isfinite(node) else None
    return node


class FlightServices:
    """Model construction, measured profiling, the loop, and the smoke path."""

    def __init__(self, context, *, device="cuda", source_cache=None):
        self.context = context
        self.config = context.config
        self.device = str(device)
        #: ``None`` keeps the shared ``<artifact>/cache/sources``. See :meth:`source_cache_root`.
        self._source_cache = None if source_cache is None else Path(source_cache)
        if self.config.get("runtime", {}).get("cpu_threads"):
            torch.set_num_threads(int(self.config["runtime"]["cpu_threads"]))
        if self.config.get("runtime", {}).get("deterministic_algorithms", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        #: The config's list. It sizes the condition tables for the preflight
        #: probes only; production models are built after ``sources`` from the
        #: vocabulary the enrolled training partitions actually produce.
        self.declared_condition_spec = context.condition_spec()
        self._cache = {}
        vocabulary_path = self.context.path("condition_vocabulary.json")
        if vocabulary_path.is_file():
            from .her2_routes_conditions import ConditionSpec
            self._cache["condition_spec"] = ConditionSpec.from_enrolled(
                paths.read_json(vocabulary_path),
                bin_dropout=float(self.config["conditions"]["bin_dropout"]),
                assay_dropout=float(self.config["conditions"]["assay_dropout"]))

    def source_cache_root(self):
        """Where the source readers keep their parsed-table caches. The one definition.

        By default the shared ``<artifact>/cache/sources`` every flight on this artifact root
        uses. Several readers rewrite their cache non-atomically when a digest, parser revision
        or preamble disagrees, so a caller that must not write into a live flight's inputs
        constructs the services with ``source_cache=<its own scratch directory>`` and every
        reader follows. The default path is spelled here and nowhere else: a copy at a call
        site is how one reader would be redirected while another kept writing to the shared
        directory.
        """
        root = (self._source_cache if self._source_cache is not None
                else self.context.artifact("cache/sources"))
        if spec.ADMISSION_PHASE_KEY in self.config:
            # The one place an admission flight writes OUTSIDE its run root: a cache miss rewrites
            # ``rows.tsv`` / ``index.json`` in place. A cache that resolves into a pinned origin, or
            # holds a link, a junction or a multiply-linked table, is refused before any reader sees
            # it, so a clean run root alone is not mistaken for a safe flight.
            admission.require_disjoint(root, spec.admission_pins(self.config))
            admission.require_clean_tree(root, label="the source cache")
        return root

    @property
    def condition_spec(self):
        """The frozen condition spec: derived if sources have been read, else declared.

        A single flat list in a config is a second source of truth about what the
        data contains, and it drifts -- the Li releases carry a per-row ``Assay``
        column and CR9114 carries two antigens, so the identities the stream
        produces are a property of the files. The derived vocabulary is frozen
        once, recorded in the freeze, and a later run whose derivation disagrees is
        refused rather than resized.
        """
        return self._cache.get("condition_spec") or self.declared_condition_spec

    # -- roots and inputs --------------------------------------------------
    def native_roots(self):
        return {"package": self.context.artifact(NATIVE_PACKAGE_ROOT),
                "weights": self.context.artifact(NATIVE_WEIGHTS_ROOT)}

    def scaffold(self):
        if "scaffold" not in self._cache:
            from . import her2_data
            root = self.context.repository(self.config["roots"]["her2_raw"])
            self._cache["scaffold"] = her2_data.load_scaffold(root)
        return self._cache["scaffold"]

    # -- native DPLM -------------------------------------------------------
    @staticmethod
    def is_dplm(route):
        return route in DPLM_ROUTES

    def route_block(self, route):
        return dict(self.config["routes"].get(route) or {})

    def dplm_pins(self):
        """The pins document. Read from the repository, cached, never from the network."""
        if "dplm_pins" not in self._cache:
            from . import her2_routes_dplm_pins as pins_lib
            block = self.config.get("dplm")
            require(block and block.get("pins"),
                    "A DPLM route is enabled but the config has no dplm.pins entry")
            self._cache["dplm_pins"] = pins_lib.load_pins(self.context.repository(block["pins"]))
        return self._cache["dplm_pins"]

    def uses_diffusion_reconstruction(self, route):
        """Does this route train reconstruction with sampled-timestep corruption?"""
        if not self.is_dplm(route):
            return False
        setting = str((self.route_block(route).get("objective") or {}).get("reconstruction",
                                                                          "mask_fraction"))
        require(setting in ("diffusion", "mask_fraction"),
                f"routes.{route}.objective.reconstruction must be 'diffusion' or 'mask_fraction'")
        return setting == "diffusion"

    def supports_diffusion_decoding(self, route):
        return self.is_dplm(route)

    def native_identity(self, route):
        """The digest binding a native base, LoRA settings, optimizer and diffusion settings.

        ``None`` for the three original routes, so their identities are unchanged. For a DPLM
        route it folds in the pins digest and the pinned base digests: changing the pins, the
        LoRA rank, the optimizer policy or a diffusion setting changes the identity, and a
        resume against the old checkpoint is refused.
        """
        if not self.is_dplm(route):
            return None
        from . import her2_routes_dplm_pins as pins_lib
        block = self.route_block(route)
        pins = self.dplm_pins()
        entry = pins_lib.variant_entry(pins, block["variant"])
        document = {"variant": block["variant"], "repo": entry["repo"],
                    "revision": entry["revision"],
                    "files": {name: dict(entry["files"][name]) for name in pins_lib.FILES},
                    "pins_sha256": pins_lib.pins_digest(pins),
                    "lora": block.get("lora"), "optimizer": block.get("optimizer"),
                    "diffusion": block.get("diffusion"),
                    "objective": block.get("objective"),
                    "activation_checkpointing": bool(block.get("activation_checkpointing", False)),
                    "dropout": block.get("dropout"), "base_dtype": block.get("base_dtype", "auto"),
                    "resolved_storage_dtype": str(self.dplm_storage_dtype(route)),
                    "max_positions": block.get("max_positions")}
        return paths.digest_document(document)

    def dplm_storage_dtype(self, route):
        """The resting dtype of the frozen base. ``auto`` is fp16 for LoRA on CUDA, else fp32.

        ``fp16`` is an explicit request (LoRA only) and holds on any device, so an adapter
        trained over an fp16 base can be re-opened on CPU without silently becoming fp32.
        The result is part of the base identity, so a mismatched adapter is refused.
        """
        block = self.route_block(route)
        base_dtype = str(block.get("base_dtype", "auto"))
        require(base_dtype in ("auto", "fp32", "fp16"),
                f"routes.{route}.base_dtype must be auto, fp32 or fp16")
        require(base_dtype != "fp16" or block.get("lora") is not None,
                f"routes.{route}.base_dtype fp16 needs a LoRA route: a full-tune base trains in fp32")
        if base_dtype == "fp16" or (block.get("lora") is not None and base_dtype == "auto"
                                    and self.device.startswith("cuda")):
            return torch.float16
        return torch.float32

    def _build_dplm(self, route):
        from ..models import dplm_policy
        block = self.route_block(route)
        pins = self.dplm_pins()
        from . import her2_routes_dplm_pins as pins_lib
        entry = pins_lib.variant_entry(pins, block["variant"])
        lora = block.get("lora")
        storage = self.dplm_storage_dtype(route)
        native = dplm_policy.load_pinned_dplm(
            self.context.artifact(entry["logical_dir"]), pins=pins, variant=block["variant"],
            device=self.device, storage_dtype=storage, dropout=block.get("dropout"))
        require(int(block.get("max_positions", 512)) <= int(native.config.max_position_embeddings),
                f"routes.{route}.max_positions exceeds the model's "
                f"{native.config.max_position_embeddings} positions")
        model = dplm_policy.ConditionedDPLM(
            native, self.condition_spec, lora=lora,
            activation_checkpointing=bool(block.get("activation_checkpointing", False)),
            diffusion=block.get("diffusion"))
        return model.to(self.device)

    # -- checkpoint lineage ------------------------------------------------
    def run_document(self, name):
        target = self.context.path(f"{name}.json")
        return paths.read_json(target) if target.is_file() else None

    def checkpoint_lineage(self, stage, route):
        """The checkpoint ``stage`` consumes, resolved explicitly. See ``her2_routes_lineage``.

        The energy route has no priors or post-training path (its teachers and preference
        arms are defined over token likelihoods), so for it the later stages read the
        bake-off best, and the record says so.
        """
        effective = stage
        if route == ROUTE_ENERGY and stage in lineage.FINAL_STAGES + ("posttrain",):
            effective = "controls"
        record = lineage.resolve_lineage(
            effective, route=route, path_for=self.context.path,
            record_for=lambda name: read_stage_record_safe(self.context, name),
            document_for=self.run_document)
        if effective != stage:
            record = dict(record, stage=stage, reason=(
                record["reason"] + "; the energy route has no post-training checkpoint"))
        return record

    def require_checkpoint(self, path, *, stage, route, role):
        """A named checkpoint that must exist. The backbone comparator is passed as ``None``."""
        return lineage.require_selected_checkpoint(path, stage=stage, route=route, role=role)

    # -- optimizers --------------------------------------------------------
    def route_learning_rate(self, route):
        """The route's own learning rate if its optimizer block names one, else the global."""
        block = dict(self.route_block(route).get("optimizer") or {})
        if "learning_rate" in block:
            return float(block["learning_rate"])
        return float(self.config["optimization"]["learning_rate"])

    def build_optimizer(self, route, parameters, *, lr, **adam_options):
        """The one place a language-route optimizer is built (train, profile, distil, posttrain).

        A route with no ``optimizer`` block gets EXACTLY the AdamW it always got, with the
        caller's own keyword arguments, so the three original routes are unchanged. A block
        may name ``adamw`` or ``adafactor``. Adafactor is ``transformers.optimization.
        Adafactor`` with fixed-LR options -- ``relative_step``, ``scale_parameter`` and
        ``warmup_init`` all False -- because ``torch.optim.Adafactor`` is relative-step and
        parameter-scaled and would fight this loop's warm-up/plateau schedule. If it cannot
        be imported the run stops; it does not fall back to Adam, which would silently
        forfeit the memory bound the route was chosen for.
        """
        block = dict(self.route_block(route).get("optimizer") or {})
        if not block:
            return torch.optim.AdamW(parameters, lr=float(lr), **adam_options)
        name = str(block.get("name", "adamw")).lower()
        if name == "adamw":
            unknown = sorted(set(block) - set(ADAMW_KEYS))
            require(not unknown, f"routes.{route}.optimizer: unknown AdamW key(s) {unknown}")
            options = dict(adam_options)
            if "betas" in block:
                options["betas"] = tuple(block["betas"])
            if "weight_decay" in block:
                options["weight_decay"] = float(block["weight_decay"])
            return torch.optim.AdamW(parameters, lr=float(lr), **options)
        require(name == "adafactor", f"routes.{route}.optimizer.name {name!r} is not adamw/adafactor")
        unknown = sorted(set(block) - set(ADAFACTOR_KEYS))
        require(not unknown, f"routes.{route}.optimizer: unknown Adafactor key(s) {unknown}")
        for flag in ("relative_step", "scale_parameter", "warmup_init"):
            require(block.get(flag, False) is False,
                    f"routes.{route}.optimizer.{flag} must be false: this loop drives a fixed "
                    "learning rate through its own warm-up and plateau schedule")
        try:
            from transformers.optimization import Adafactor
        except ImportError as error:
            raise ValueError(
                f"routes.{route} names the Adafactor optimizer but transformers.optimization."
                "Adafactor is not importable. The run stops rather than substituting Adam, "
                "which would forfeit the optimizer-state memory bound this route relies on.") from error
        return Adafactor(
            parameters, lr=float(lr), eps=tuple(block.get("eps", (1e-30, 1e-3))),
            clip_threshold=float(block.get("clip_threshold", 1.0)),
            decay_rate=float(block.get("decay_rate", -0.8)), beta1=block.get("beta1"),
            weight_decay=float(block.get("weight_decay", 0.0)), scale_parameter=False,
            relative_step=False, warmup_init=False)

    # -- construction ------------------------------------------------------
    def build_route(self, route, *, seed=0):
        require(route in KNOWN_ROUTES, f"Unknown route {route!r}")
        torch.manual_seed(int(seed))
        if route in DPLM_ROUTES:
            return self._build_dplm(route)
        if route == ROUTE_MASKED:
            from ..models.ablang2_policy import ConditionedAbLang2, load_pinned_ablang2
            roots = self.native_roots()
            native = load_pinned_ablang2(roots["weights"], package_root=roots["package"],
                                        device=self.device)
            model = ConditionedAbLang2(native, self.condition_spec,
                activation_checkpointing=bool(self.config["routes"][route].get("activation_checkpointing", False)))
            return model.to(self.device)
        if route == ROUTE_CAUSAL:
            from ..models.her2_routes_causal import ConditionedCausalInfiller
            root = self.context.repository(self.config["roots"]["her2_raw"])
            return ConditionedCausalInfiller.from_release(
                root, self.condition_spec, device=self.device).to(self.device)
        from ..models.her2_routes_energy import WalkJumpModel
        block = self.config["routes"][ROUTE_ENERGY]
        model = WalkJumpModel(window=int(block["window"]),
                              sigma=float(block["sigma"]),
                              scale=float(block["scale"]),
                              reg_l2_norm=float(block["reg_l2_norm"]),
                              context_width=int(block["context_width"]),
                              condition_sizes=self.condition_spec.sizes)
        return model.to(self.device)

    def build_heads(self, route, model):
        from ..models.her2_routes_heads import ObservationHeads
        if route in DPLM_ROUTES:
            hidden = int(model.hidden_size)
        else:
            hidden = (int(model.hyperparameters["hidden_embed_size"]) if route == ROUTE_MASKED
                      else int(model.model.config.hidden_size) if route == ROUTE_CAUSAL
                      else int(self.config["routes"][ROUTE_ENERGY]["context_width"]))
        return ObservationHeads(hidden, assays=self.condition_spec.assay_order).to(self.device)

    def pad_id(self, route):
        """The PUBLIC pad id. The DPLM routes keep the pipeline's AbLang pad (21).

        Collation and ``window_log_probs`` speak public ids; a native pad of 1 here would
        collide with AbLang ``M``. The one translation to native ids is inside the model.
        """
        return ABLANG_PAD_ID if route in MASKED_LIKE_ROUTES else PIGGEN_MASK_ID

    # -- one forward -------------------------------------------------------
    def window_log_probs(self, route, model, batch):
        """``(B, k)`` declared-order sequential conditionals for scoring a joint."""
        capabilities.require_token_likelihood(route, "FlightServices.window_log_probs")
        if route == ROUTE_CAUSAL:
            return model.window_log_probs(batch)
        from .her2_routes_serial import ablang_canonical_ids
        tokens = np.array(batch.tokens, copy=True)
        slots, targets = np.asarray(batch.query_slots), np.asarray(batch.targets)
        active = (np.ones_like(targets, dtype=bool) if batch.target_mask is None
                  else np.asarray(batch.target_mask, dtype=bool))
        native = np.asarray(ablang_canonical_ids())
        values = []
        for step in range(slots.shape[1]):
            log_probs = model.canonical_log_probs(
                tokens.copy(), slots[:, step:step + 1], condition_indices=batch.condition_indices)[:, 0]
            target = torch.as_tensor(targets[:, step], dtype=torch.long, device=log_probs.device)
            mask = torch.as_tensor(active[:, step], dtype=log_probs.dtype, device=log_probs.device)
            values.append(log_probs.gather(1, target[:, None])[:, 0] * mask)
            rows = np.flatnonzero(active[:, step])
            tokens[rows, slots[rows, step]] = native[targets[rows, step]]
        return torch.stack(values, dim=1)

    def canonical_log_probs(self, route, model, batch):
        capabilities.require_token_likelihood(route, "FlightServices.canonical_log_probs")
        if route == ROUTE_CAUSAL:
            return model.canonical_log_probs(batch.tokens, batch.query_slots,
                                             pointers=batch.pointers,
                                             condition_indices=batch.condition_indices,
                                             attention_mask=batch.attention_mask)
        return model.canonical_log_probs(batch.tokens, batch.query_slots,
                                         condition_indices=batch.condition_indices)

    def representation(self, route, model, batch):
        """``(B, hidden)`` trunk representation of a head batch. The head's input.

        The heads read the trunk, so their gradient reaches the shared encoder. A
        head trained on a detached representation measures the head, which is
        exactly the confound the ``head_only`` control arm exists to separate -- and
        it cannot separate it if the full arm's head is also decoupled.
        """
        capabilities.require_token_likelihood(route, "FlightServices.representation")
        return model.representation(batch.tokens, attention_mask=batch.attention_mask,
                                    condition_indices=batch.condition_indices)

    def energy_representation(self, model, summaries):
        """``(B, width)`` pooled context vector for the energy route's heads.

        Only a model that was trained WITH a context has one to pool: the null arm is given none, so
        asking it for this vector is refused rather than answered from a tower it never trained.
        """
        runtime = energy_contract.require_runtime(model, where="energy_representation")
        require(runtime.conditioned,
                "energy_representation: the null arm is given no context, so it has no pooled "
                "context vector; the context builder and both towers are never invoked for it")
        pooled, _sites, _mask = model.encode_context(self.energy_context(summaries,
                                                                        answer_free=True))
        return pooled

    def joint_support(self, route, model, summary, *, sites, condition, residues=20,
                      first_residues=None):
        """The sequentially revealed two-site joint. ``(log_joint, forwards, max TV)``.

        ``log_joint[a, b] = log q(r1 = a) + log q(r2 = b | r1 = a)``. The second
        conditional is scored **after** the first residue has been revealed in the
        input, which is the only construction that can express a coupling: two
        conditionals taken from one doubly masked forward multiply to a normalised
        distribution over 400 outcomes that is independent by construction, passes
        every normalisation check, and cannot move an interaction term.

        One batched forward per route. The masked editor needs ``1 + residues``
        rows -- the doubly masked row for the first conditional and one row per
        revealed first residue -- and the autoregressive route needs ``residues``,
        because a teacher-forced pass already yields both conditionals and the first
        one is identical across rows.
        """
        capabilities.require_token_likelihood(route, "FlightServices.joint_support")
        from .her2_data import CANONICAL
        from .her2_routes_serial import ablang_serialization, causal_serialization
        first_indices = tuple(range(int(residues))) if first_residues is None else tuple(
            sorted(set(int(i) for i in first_residues)))
        require(first_indices and all(0 <= i < int(residues) for i in first_indices),
                "Invalid first-residue support for sequential joint")
        first, second = sites
        construct = data_lib.with_sites(summary.construct, (first, second))
        rows = []
        if route == ROUTE_CAUSAL:
            for index in first_indices:
                residue = CANONICAL[index]
                revealed = construct.with_residues((first,), residue)
                rows.append((revealed, causal_serialization(
                    revealed, order=(first, second),
                    residues=revealed.residues((first, second)),
                    max_positions=self._positions(route)), condition))
            batch = self.collate(route, rows)
            log_probs = self.canonical_log_probs(route, model, batch)
            first_conditional = log_probs[0, 0]
            second_conditionals = log_probs[:, 1]
            forwards = len(rows)
        else:
            rows.append((construct, ablang_serialization(
                construct, masked_sites=(first, second), order=(first,)), condition))
            for index in first_indices:
                residue = CANONICAL[index]
                revealed = construct.with_residues((first,), residue)
                rows.append((revealed, ablang_serialization(
                    revealed, masked_sites=(second,), order=(second,)), condition))
            batch = self.collate(route, rows)
            log_probs = self.canonical_log_probs(route, model, batch)
            first_conditional = log_probs[0, 0]
            second_conditionals = log_probs[1:, 0]
            forwards = len(rows)
        require(second_conditionals.shape[0] == len(first_indices),
                f"The joint needs {len(first_indices)} revealed conditionals, got "
                f"{second_conditionals.shape[0]}")
        index_tensor = torch.as_tensor(first_indices, dtype=torch.long, device=first_conditional.device)
        selected_joint = first_conditional.index_select(0, index_tensor).unsqueeze(-1) + second_conditionals
        log_joint = selected_joint.new_full((int(residues), int(residues)), float("-inf")).index_copy(
            0, index_tensor, selected_joint)
        with torch.no_grad():
            probabilities = second_conditionals.detach().exp()
            worst = 0.0
            for left in range(len(first_indices)):
                for right in range(left + 1, len(first_indices)):
                    distance = float(0.5 * (probabilities[left]
                                            - probabilities[right]).abs().sum())
                    worst = max(worst, distance)
        return log_joint, forwards, worst

    # -- profiling ---------------------------------------------------------
    def profile_draws(self, route, size, *, seed=20260927):
        """Real draws of every scheduled task at the WORST shape each one has.

        The earlier profiler asked for eight smoke views, got four, reported eight
        and divided the timing by eight -- and it measured ``masked_nll`` on a
        ten-site HER2 window, which is the cheapest thing the loop ever does. What
        the loop actually runs is a two-site coupled contrast with a sequentially
        revealed 400-way joint, a head batch over a full unmasked paired construct,
        and a whole-window task over the widest common region any enrolled source
        declares. Those are what this profiles.

        Every draw holds EXACTLY ``size`` rows. The earlier local draw replaced the batch with one
        quartet -- four rows -- whatever size was asked for, so a size-8 measurement timed and
        sized half the batch it was named for. The local draw is now ``size`` endpoint rows of
        representative contrasts (quartets first when the recipe requests interactions, then
        pairs); a task the pools cannot fill to ``size`` rows is not profiled under that size:
        :class:`ShapeUnavailable` names it, and the caller records the candidate as rejected.
        """
        size = data_lib.positive_integer(size, "profile microbatch")
        rng = np.random.default_rng([int(seed), 5])
        stream = self.curriculum_stream(route, seed=int(seed), updates=64, microbatch_override=size)
        pools = self.training_pools()
        draws, notes, unavailable = {}, {}, {}
        widest = None
        for row in pools["summaries"][data_lib.FAMILY_HER2] + \
                pools["summaries"][data_lib.FAMILY_OTHER]:
            if row.has_edit_map and (widest is None or len(row.region) > len(widest.region)):
                widest = row
        tile = int(self.config["curriculum"].get("max_query_sites", 0)) or None
        if widest is None:
            unavailable[data_lib.TASK_WHOLE] = "no enrolled measured row declares an edit region"
        else:
            views = data_lib.whole_window_views(
                [widest] * size, condition_spec=self.condition_spec,
                bins=dict(self._cache.get("bins") or {}), route=self._kind(route),
                max_positions=self._positions(route), tile=tile, rng=rng, sample_tile=True)
            if len(views) == size:
                draws[data_lib.TASK_WHOLE] = tasks_lib.TaskDraw(
                    task=data_lib.TASK_WHOLE,
                    family=(data_lib.FAMILY_HER2 if widest.source == "alphabind_her2_validation"
                            else data_lib.FAMILY_OTHER), views=views,
                    summaries=(widest,) * len(views), source=widest.source)
                notes["widest_region_sites"] = len(widest.region)
            else:
                unavailable[data_lib.TASK_WHOLE] = (
                    f"the widest region formed {len(views)} view(s) for {size} requested rows")
        for task in (data_lib.TASK_LOCAL, data_lib.TASK_HEADS, data_lib.TASK_RECONSTRUCTION):
            if task == data_lib.TASK_LOCAL:
                draw, reason = self._profile_local_draw(pools, route, size, int(seed), notes)
            else:
                draw = stream(1, 0, rng, task_override=task,
                              family_override=(data_lib.FAMILY_REPLAY
                                               if task == data_lib.TASK_RECONSTRUCTION
                                               else data_lib.FAMILY_OTHER))
                reason = None if draw is not None and draw.task == task else (
                    "no draw of this task could be formed")
            if draw is not None and reason is None and draw.rows() != size:
                reason = f"the stream formed {draw.rows()} row(s) for {size} requested"
            if reason is not None or draw is None:
                unavailable[task] = reason or "no draw of this task could be formed"
                continue
            draws[task] = draw
        if unavailable or set(draws) != set(data_lib.TASKS):
            notes["unavailable"] = dict(sorted(unavailable.items()))
            raise ShapeUnavailable(
                f"microbatch {size} cannot be profiled at its own shape: {dict(unavailable)}. "
                "A task is reported unprofiled at this size rather than measured at a different "
                "one.", notes=notes)
        return draws, notes

    def _profile_local_draw(self, pools, route, size, seed, notes):
        """``(TaskDraw | None, reason)``: ``size`` endpoint rows of representative contrasts.

        Quartets first (four rows, the worst local shape: their joint costs ``1 + 20`` scored rows)
        when the declared recipe requests interactions, then pairs for the remainder. Candidates
        are used in a fixed cycle, so a pool with a single quartet still yields a size-8 batch (two
        copies, recorded in ``notes``); a pool that cannot tile ``size`` at all returns a reason.

        EVERY available family is searched, in order, for a pool that carries the recipe -- the
        first non-empty pool is not enough: it may hold only pairs while a later family holds the
        quartets. When the joint term runs (``interaction_weight > 0``) a representative quartet is
        one ``local_outcome`` really fits (:func:`~.her2_routes_data.joint_executable`: numeric,
        with an interaction -- an enrichment quartet is ordered, never fitted). A recipe that
        requests quartets and finds none is UNAVAILABLE; it is never profiled with pairs, because
        a profile of the cheaper computation certifies a shape the run does not execute. A coupled
        recipe with weight zero profiles its quartet rows without the joint and says so in
        ``notes``.
        """
        recipe = data_lib.local_shape_recipe(self.config["curriculum"])
        need = data_lib.local_joint_requirement(self.config["curriculum"])
        found, inventory = [], []
        for family in (data_lib.FAMILY_OTHER, data_lib.FAMILY_HER2):
            members = pools["summaries"][family]
            if not members:
                continue
            bank = self.contrast_pool(
                members, seed=seed, cache_key=(family, seed, "full", False, len(members)))
            if not bank:
                continue
            coupled = [c for c in bank if c.kind == data_lib.CONTRAST_COUPLED]
            pairs = [c for c in bank if c.kind != data_lib.CONTRAST_COUPLED]
            quartets = ([c for c in coupled if data_lib.joint_executable(c)]
                        if need["joint_required"] else coupled)
            found.append((family, quartets, pairs))
            inventory.append(f"{family}: {len(coupled)} coupled ({len(quartets)} usable), "
                             f"{len(pairs)} pair(s)")
        if not found:
            return None, "no family holds a contrast pool to profile the local task from"
        wants = need["coupled_rows_required"]
        if wants and not any(quartets for _family, quartets, _pairs in found):
            reason = ("the declared local recipe requests quartets"
                      + (" and fits their joint (interaction_weight "
                         f"{need['interaction_weight']:g})" if need["joint_required"] else "")
                      + " but no available pool holds a "
                      + ("quartet that executes it (a numeric coupled contrast with an "
                         "interaction; enrichment-scale quartets are ordered, not fitted)"
                         if need["joint_required"] else "coupled contrast")
                      + f" [{'; '.join(inventory)}]. The local task is not profiled with pairs: "
                        "that is a cheaper computation than the one the run executes.")
            notes["local_quartet_unavailable"] = reason
            return None, reason
        for family, quartets, pairs in found:
            quartets = quartets if wants else []
            picked, rows, quartet_at, pair_at = [], 0, 0, 0
            while rows < size:
                remaining = size - rows
                if quartets and remaining >= 4:
                    choice = quartets[quartet_at % len(quartets)]
                    quartet_at += 1
                elif pairs and remaining >= 2:
                    choice = pairs[pair_at % len(pairs)]
                    pair_at += 1
                else:
                    break
                picked.append(choice)
                rows += len(choice.alternatives())
            # A recipe that requests quartets is profiled with at least one: a size too small for
            # a quartet that fills with pairs has timed something else.
            if rows != size or (wants and not quartet_at):
                continue
            if wants and not need["joint_required"]:
                notes["local_joint_not_executed"] = (
                    "the recipe requests coupled quartets but interaction_weight is "
                    f"{need['interaction_weight']:g} (max_joint_per_update "
                    f"{need['max_joint_per_update']}): their rows are scored and no sequential "
                    "joint runs, so this profiles that cheaper declared computation")
            if (quartet_at > len(quartets) > 0) or (pair_at > len(pairs) > 0):
                notes["local_contrasts_repeated"] = (
                    "the pool held fewer distinct contrasts than the batch needs; a contrast "
                    "appears more than once so the profiled batch has the requested number of "
                    "rows")
            notes["local_draw_family"] = family
            return tasks_lib.TaskDraw(
                task=data_lib.TASK_LOCAL, family=family,
                views=self.contrast_views(picked, route=route), contrasts=tuple(picked),
                summaries=tuple(row for contrast in picked for row in contrast.alternatives()),
                source=picked[0].left.source), None
        return None, (f"the contrast pools [{'; '.join(inventory)}] cannot tile {size} endpoint "
                      f"rows of the {recipe['mode']} local recipe")

    def profile(self, route, *, microbatches=(1, 2, 4, 8), steps=4, draws=None, views=None):
        """Measure step time, allocator peak and driver free VRAM per task and microbatch.

        Three signals, because one is not enough on this box. A configuration is
        accepted only when the driver reading stays above the declared floor, the
        allocator peak fits the declared budget, and the median step time does not
        exceed the declared multiple of the smallest microbatch's per-row time. The
        third is what catches a spill that the first two miss.

        The profiled computation is the **real objective set**, heads included, so
        the accepted microbatch is a statement about the configuration that runs.
        """
        budget = self.config["profile"]
        # Every CUDA call below is gated on this. A CPU profile on a CUDA-capable host must
        # stay CPU-only: it must not reset, read or synchronise a device it never used.
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        # Candidate shapes are judged BEFORE any model is built. A candidate must be a positive
        # integer that divides the effective batch exactly and can carry the declared local recipe
        # (an even microbatch of at least four when the recipe requests interactions). The ones
        # that cannot are recorded as rejected, with the reason, and profiling continues with the
        # rest -- they are not rounded to a nearby size.
        # The RAW declared value, judged before anything is built: ``int(64.5)`` is 64, and a
        # profile that accepted a microbatch against a truncated effective batch would certify a
        # divisibility the run does not have. Negative, fractional, boolean and missing values
        # are refused here, on the injected-draws path as well.
        effective = data_lib.positive_integer(
            self.config["optimization"].get("effective_batch"), "optimization.effective_batch")
        curriculum = self.config.get("curriculum")
        requested, candidates, rejected = list(microbatches), [], []
        for size in requested:
            try:
                data_lib.validate_microbatch(size, effective_batch=effective, curriculum=curriculum,
                                             label="profile candidate microbatch")
            except ValueError as error:
                rejected.append({"microbatch": size if isinstance(size, int) else str(size),
                                 "kind": "incompatible_shape", "reason": str(error)})
                continue
            candidates.append(int(size))
        require(candidates,
                "Every candidate microbatch was rejected before any model was built, so there is "
                "nothing to profile: "
                + "; ".join(f"{entry['microbatch']!r}: {entry['reason']}" for entry in rejected))
        if draws is None:
            self.training_pools()
        model = self.build_route(route)
        heads = self.build_heads(route, model)
        # Production train() runs in training mode. load_pinned_dplm returns the native child in
        # eval mode, and per-layer checkpoints and dropout follow native.training, so a profile
        # that skips this measures a different (cheaper, dropout-free) computation.
        model.train()
        heads.train()
        native = getattr(model, "native", None)
        training_mode = {"model": bool(model.training),
                         "native": None if native is None else bool(native.training),
                         "heads": bool(heads.training)}
        device_total_mib = (torch.cuda.get_device_properties(self.device).total_memory / 2 ** 20
                            if use_cuda else None)
        parameters = [p for p in list(model.parameters()) + list(heads.parameters())
                      if p.requires_grad]
        optimizer = self.build_optimizer(route, parameters, lr=float(budget["probe_lr"]))
        settings = self.task_settings()
        clip = float(self.config["optimization"]["gradient_clip"])
        rows, notes = [], {}
        baseline_per_row = {}
        for size in candidates:
            if draws is not None:
                drawn, note = draws, {}
            else:
                try:
                    drawn, note = self.profile_draws(route, int(size))
                except ShapeUnavailable as error:
                    notes.update(error.notes)
                    rejected.append({"microbatch": int(size), "kind": "shape_unavailable",
                                     "reason": str(error)})
                    continue
            notes.update(note)
            for task, draw in sorted(drawn.items()):
                if use_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                timings, failure, row_count = [], None, draw.rows()
                detail, dispatched = None, None
                observed_joint = []                    # per executed step: (available, trained)
                completed = 0                          # finite, optimizer-applied steps
                rng = np.random.default_rng([int(budget.get("probe_seed", 7)), 1])
                if row_count != int(size):
                    # The draw is not the shape it is named for. Timing it would report the cost
                    # of ``row_count`` rows as the cost of ``size`` rows, so it is not run: the
                    # row is a failure that names both numbers and can never be accepted.
                    failure = (f"shape_mismatch: the {task} draw holds {row_count} row(s) for a "
                               f"requested microbatch of {int(size)}")
                    detail = {"requested": int(size), "formed": int(row_count)}
                try:
                    for index in range(int(steps) if failure is None else 0):
                        started = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        outcome = tasks_lib.dispatch(self, route, model, heads, draw,
                                                     arm="full", settings=settings, rng=rng)
                        dispatched = {"term": outcome.term,
                                      "scheduled_task": outcome.scheduled_task,
                                      "rows": int(outcome.rows),
                                      "diagnostic_keys": sorted(outcome.diagnostics)}
                        if task == data_lib.TASK_LOCAL:
                            joint = dict(outcome.diagnostics.get("interaction") or {})
                            observed_joint.append((joint.get("quartets_available"),
                                                   joint.get("quartets_trained")))
                        outcome.loss.backward()
                        # The same clip production applies, so the profiled step is the
                        # production step. A non-finite loss or gradient never reaches the
                        # optimizer: the step is skipped and the row is a failure.
                        norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
                        loss_value = outcome.loss.detach()
                        if not bool(torch.isfinite(loss_value) & torch.isfinite(norm)):
                            loss_finite = bool(torch.isfinite(loss_value))
                            failure = "nonfinite_loss" if not loss_finite else "nonfinite_gradient"
                            detail = {"step": int(index),
                                      "loss": self._json_scalar(loss_value),
                                      "gradient_norm": self._json_scalar(norm)}
                            break
                        optimizer.step()
                        completed += 1
                        if use_cuda:
                            torch.cuda.synchronize()
                        if index > 0:                # drop the warm-up step
                            timings.append(time.perf_counter() - started)
                except torch.cuda.OutOfMemoryError as error:   # pragma: no cover - device specific
                    failure = f"OutOfMemoryError: {error}"
                median = None if not timings else float(np.median(timings))
                per_row = None if median is None or row_count == 0 else median / row_count
                if task not in baseline_per_row and per_row is not None:
                    baseline_per_row[task] = per_row
                peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if use_cuda else None)
                reserved = (torch.cuda.max_memory_reserved() / 2 ** 20 if use_cuda else None)
                # AB-14: the floor is judged on the NVML framebuffer reading,
                # for the device this probe actually ran on. driver_free_mib
                # keeps the legacy CUDA number so older records stay comparable,
                # but it is telemetry now, not the gate.
                on_gpu = self.device.startswith("cuda")
                vram = vram_reading() if use_cuda else {
                    "cuda_free_mib": None, "cuda_error": None, "nvml_fb_free_mib": None,
                    "nvml_backend": None, "nvml_error": None, "delta_mib": None,
                    "observed_at": None,
                    "note": "a CPU probe reads no GPU memory, whatever the host has"}
                free = vram["cuda_free_mib"]
                baseline = baseline_per_row.get(task)
                slowdown = (None if per_row is None or baseline in (None, 0)
                            else per_row / baseline)
                signals = memory_signals(vram, peak_mib=peak, slowdown=slowdown,
                                         budget=budget, on_gpu=on_gpu)
                # What the local task actually EXECUTED, read off the dispatched outcomes rather
                # than the draw: pairs also carry an ``interaction`` diagnostic, so that key says
                # nothing about whether a joint ran. The minimum over the steps is recorded, so one
                # step that skipped the joint cannot hide behind the others.
                interaction_evidence = None
                if task == data_lib.TASK_LOCAL:
                    contrasts = tuple(draw.contrasts or ())
                    complete = bool(observed_joint) and all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for pair in observed_joint for value in pair)
                    interaction_evidence = {
                        "coupled_contrasts": sum(
                            1 for contrast in contrasts
                            if contrast.kind == data_lib.CONTRAST_COUPLED),
                        "joint_executable_contrasts": sum(
                            1 for contrast in contrasts if data_lib.joint_executable(contrast)),
                        "quartets_available": (min(pair[0] for pair in observed_joint)
                                               if complete else None),
                        "quartets_trained": (min(pair[1] for pair in observed_joint)
                                             if complete else None),
                        "steps_observed": len(observed_joint),
                        "interaction_weight": float(settings["interaction_weight"]),
                        "max_joint_per_update": int(settings["max_joint_per_update"])}
                rows.append({"microbatch": int(size), "task": task,
                             "requested_microbatch": int(size),
                             "rows_measured": int(row_count),
                             "completed_steps": int(completed), "steps_requested": int(steps),
                             "median_step_seconds": median,
                             "seconds_per_row": per_row, "per_row_slowdown": slowdown,
                             "peak_allocated_mib": peak, "peak_reserved_mib": reserved,
                             "driver_free_mib": free, "vram": vram, "failure": failure,
                             "failure_detail": detail, "dispatched": dispatched,
                             "interaction_evidence": interaction_evidence,
                             "probe_device": self.device,
                             "signals": signals,
                             "accepted": bool(failure is None and signals_accepted(
                                 signals, on_gpu=on_gpu))})
        # A microbatch is accepted only when EVERY profiled task accepted it. The
        # loop schedules all of them, so accepting a size on the cheapest one is how
        # a configuration that fits the reconstruction task gets chosen for a run
        # whose expensive task does not fit.
        by_size = {}
        for row in rows:
            by_size.setdefault(row["microbatch"], []).append(row)
        accepted_sizes = sorted(size for size, block in by_size.items()
                                if block and all(row["accepted"] for row in block))
        chosen = max(accepted_sizes, default=None)
        del optimizer, model, heads
        if use_cuda:
            torch.cuda.empty_cache()
        return {"route": route, "rows": rows, "chosen_microbatch": chosen,
                "accepted_microbatches": accepted_sizes,
                "tasks_profiled": sorted({row["task"] for row in rows}),
                "candidates_requested": [size if isinstance(size, int) else str(size)
                                         for size in requested],
                "rejected_shapes": rejected,
                "notes": notes,
                "local_joint_requirement": data_lib.local_joint_requirement(curriculum),
                "training_mode": training_mode,
                "device_total_mib": device_total_mib,
                # ``chosen`` divides the effective batch exactly: every candidate was validated
                # before the model was built, so this is an exact quotient and never a rounding.
                "accumulation_steps": (None if chosen is None else effective // int(chosen)),
                "effective_batch": effective,
                "acceptance_rule": ("a microbatch is accepted only when every profiled task "
                                    "accepted it, and a task row is accepted only when its draw "
                                    "held exactly the candidate's number of rows. The loop "
                                    "schedules all of them."),
                "three_signal_rule": (
                    "driver free VRAM AND allocator peak AND per-row step time. On this box CUDA "
                    "spills to system RAM instead of raising: a 5.6 GB configuration once "
                    "returned ok=True on a 4 GB card, so the absence of an OOM is not evidence "
                    "that a configuration fits."),
                "measured_on": {"cuda": bool(use_cuda),
                                "device": (torch.cuda.get_device_name(0) if use_cuda else None)}}

    @staticmethod
    def _json_scalar(value):
        """A float, or a small marker dict for NaN/inf, so an evidence record stays valid JSON."""
        number = float(value)
        if math.isfinite(number):
            return number
        return {"nonfinite": "nan" if math.isnan(number) else ("+inf" if number > 0 else "-inf")}

    def selected_shapes(self):
        """The microbatch and accumulation the PROFILE accepted, or the declared ones.

        The profile stage chose a microbatch and the training loop read the config's,
        so the measurement and the run could disagree without anything saying so.
        This is the one place either is read, and it records which it used.

        Nothing is cached. The result was once remembered for the life of the object, so a
        fallback to the declared shapes taken before ``profile.json`` existed outlived the file
        appearing, and a re-measured profile was ignored. Reading the one small file each time
        is cheap; the record names the file and its digest, so a run says which profile it used.

        What is returned is validated, so a shape that cannot run is refused here rather than
        rounded by the loop: a positive integer microbatch that divides the effective batch
        EXACTLY (the accumulation is the exact quotient, never a floor), and one that can carry
        the declared local recipe -- an even microbatch of at least four when the recipe requests
        interactions, an even one of at least two when it is pair-only. That is the same
        requirement for every control arm, which all share this curriculum.
        """
        optimization = self.config["optimization"]
        curriculum = self.config.get("curriculum")
        declared_microbatch = optimization["microbatch"]
        declared_accumulation = optimization["accumulation"]
        effective = optimization.get("effective_batch")
        if effective is None:
            effective = (data_lib.positive_integer(declared_microbatch, "optimization.microbatch")
                         * data_lib.positive_integer(declared_accumulation,
                                                      "optimization.accumulation"))
        provenance = {
            "profile_path": str(self.context.path(spec.PROFILE_JSON)), "profile_present": False,
            "profile_sha256": None, "profile_chosen_microbatch": {},
            "declared": {"microbatch": declared_microbatch, "accumulation": declared_accumulation,
                         "effective_batch": effective}}
        target = self.context.path(spec.PROFILE_JSON)
        chosen = {}
        if target.is_file():
            profiles = paths.read_json(target)
            chosen = {str(name): block["chosen_microbatch"] for name, block in profiles.items()
                      if isinstance(block, dict) and block.get("chosen_microbatch")}
            provenance.update(profile_present=True, profile_sha256=paths.sha256_file(target),
                              profile_chosen_microbatch=dict(chosen))
        if chosen:
            for name, value in chosen.items():
                data_lib.positive_integer(value, f"the profile's chosen microbatch for {name}")
            microbatch = min(int(value) for value in chosen.values())
            checked = data_lib.validate_microbatch(
                microbatch, effective_batch=effective, curriculum=curriculum,
                label="the profile-accepted microbatch")
            shapes = {"microbatch": checked["microbatch"], "accumulation": checked["accumulation"],
                      "source": ("the smallest microbatch the profile accepted across the enabled "
                                 "routes, with the accumulation set to hold the declared effective "
                                 "batch exactly. The smallest, because one configuration runs all "
                                 "of them.")}
        else:
            checked = data_lib.validate_microbatch(
                declared_microbatch, effective_batch=effective, accumulation=declared_accumulation,
                curriculum=curriculum, label="the declared microbatch")
            shapes = {"microbatch": checked["microbatch"], "accumulation": checked["accumulation"],
                      "source": ("the config's declared shapes; no profile record carrying an "
                                 "accepted microbatch was found")}
        shapes["effective_batch"] = shapes["microbatch"] * shapes["accumulation"]
        shapes["local_recipe"] = checked["local_recipe"]["mode"]
        shapes["provenance"] = provenance
        return shapes

    def forecast(self, profiles, *, schedule, ledger_sample):
        """Multiply measured per-unit costs by the declared schedule.

        The per-row cost is a **share-weighted mean over the profiled tasks**, not the
        cheapest one. The loop schedules four tasks at the declared shares and they
        differ in cost by an order of magnitude -- a coupled contrast with a
        sequentially revealed joint against a reconstruction batch -- so forecasting
        from the minimum understates the flight by roughly the ratio between them.

        Row presentations and student forwards are multiplied separately from a
        MEASURED forwards-per-row ratio. This method does not derive one counter from
        another, and that is the only claim it makes about them.
        """
        require(profiles, "A forecast needs at least one measured profile")
        sample = dict(ledger_sample)
        shapes = self.selected_shapes()
        rows_per_update = int(shapes["effective_batch"])
        forwards_per_row = float(sample["forwards_per_row"] or 1.0)
        shares = dict(self.config["curriculum"]["shares"])
        out = {}
        for route, profile in dict(profiles).items():
            accepted = [row for row in profile["rows"] if row["accepted"]
                        and row["seconds_per_row"] is not None]
            require(accepted,
                    f"{route}: no microbatch passed all three signals, so there is no measured "
                    "per-row cost to forecast from. An unmeasured cost is not extrapolated.")
            chosen = profile.get("chosen_microbatch")
            at_chosen = [row for row in accepted if row["microbatch"] == chosen] or accepted
            if route == ROUTE_ENERGY:
                block = self.config["routes"][route]
                costs = [row["median_step_seconds"] for row in at_chosen
                         if row["walk_steps"] == int(block["walk_steps"])]
                require(costs and profile.get("denoiser_seconds_per_update") is not None,
                        "Both energy phases need a measured cost at the declared walk length")
                phase_costs = {"denoiser": float(profile["denoiser_seconds_per_update"]),
                               "energy": float(np.mean(costs))}
                phase_updates = {"denoiser": int(block["denoiser_updates"]),
                                 "energy": int(block["energy_updates"])}
                seconds = 2 * sum(phase_costs[p] * phase_updates[p] for p in phase_costs)
                out[route] = {"hours": seconds / 3600, "seconds": seconds,
                              "microbatch": chosen,
                              "optimizer_updates": 2 * sum(phase_updates.values()),
                              "row_presentations": 2 * sum(phase_updates.values()) * chosen,
                              "per_phase_seconds_per_update": phase_costs,
                              "per_arm_phase_updates": phase_updates,
                              "arms": [energy_contract.ARM_FULL, energy_contract.ARM_NULL],
                              "basis": ("Each measured phase cost, including its walk, times its "
                                        "updates; both arms. Both arms train on the same "
                                        "population, and the cost is measured WITH the context "
                                        "in the graph, so it is the conditioned arm's figure "
                                        "applied to both (the null arm builds no context and "
                                        "was not timed separately)."),
                              "excludes": ["evaluation", "generation", "checkpoint I/O"]}
                continue
            by_task = {}
            for row in at_chosen:
                by_task.setdefault(row.get("task") or "unknown", []).append(
                    float(row["seconds_per_row"]))
            costs = {task: float(np.mean(values)) for task, values in by_task.items()}
            weight_total = sum(float(shares.get(task, 0.0)) for task in costs) or None
            if weight_total:
                per_row = sum(costs[task] * float(shares.get(task, 0.0))
                              for task in costs) / weight_total
            else:
                per_row = float(np.mean(list(costs.values())))
            updates = int(schedule[route]["updates"])
            rows = updates * rows_per_update
            forwards = int(round(rows * forwards_per_row))
            seconds = per_row * rows
            out[route] = {
                "measured_seconds_per_row": per_row,
                "per_task_seconds_per_row": costs,
                "task_shares_used": {task: float(shares.get(task, 0.0)) for task in costs},
                "microbatch": chosen,
                "optimizer_updates": updates,
                "row_presentations": rows,
                "student_forwards": forwards,
                "forwards_per_row_measured": forwards_per_row,
                "seconds": seconds, "hours": seconds / 3600.0,
                "accounting": (
                    f"{updates} updates x {rows_per_update} effective batch = {rows} ROW "
                    f"PRESENTATIONS. Student forwards are {forwards}, which is larger because "
                    "preference pairs and local contrasts each consume two forwards. Quoting "
                    f"{rows} as 'exposures' would understate the compute by "
                    f"{forwards - rows} forwards."),
                "per_row_basis": ("the share-weighted mean over the profiled tasks at the chosen "
                                  "microbatch. The minimum would be the reconstruction task's, "
                                  "which is the cheapest thing the loop runs."),
                "excludes": ["evaluation passes", "bank generation", "checkpoint I/O",
                             "the per-step activations of the Langevin walk (energy route)",
                             *(["diffusion decoding: decode_steps denoiser forwards per draw "
                                "batch, plus a fixed-order rescoring of each window "
                                "(reported by the decoding stage's own accounting)"]
                               if self.is_dplm(route) else [])]}
        out = self.forecast_imports(out)
        total = sum(block["hours"] for block in out.values())
        return {"per_route": out, "total_hours": total,
                "reserve_hours": float(self.config["runtime"]["reserve_hours"]),
                "with_reserve_hours": total + float(self.config["runtime"]["reserve_hours"]),
                "soft_cap_hours_per_route": float(self.config["runtime"]["soft_cap_hours"]),
                "drop_order": list(self.config["runtime"]["drop_order"]),
                "basis": ("measured per-row costs on this box times the declared schedule. It is "
                          "reported whether or not it fits, and it never relabels an incomplete "
                          "flight as complete.")}

    def forecast_imports(self, per_route):
        """Count an admitted route as an IMPORT: zero hours here, its would-be cost kept beside it.

        Forecasting a finished route's full training schedule would put days of GPU time into the
        total of a flight that trains none of it. A config without the admission phase is returned
        unchanged.
        """
        policy = self.admission_policy()
        if policy is None:
            return per_route
        out = {}
        for route, block in per_route.items():
            if route not in policy["admitted_routes"]:
                out[route] = block
                continue
            out[route] = {"imported": True, "hours": 0.0, "seconds": 0.0, "optimizer_updates": 0,
                          "row_presentations": 0, "student_forwards": 0,
                          "microbatch": block.get("microbatch"),
                          "hours_if_trained_here": block.get("hours"),
                          "basis": ("admitted from a finished flight: its checkpoint is verified "
                                    "and copied, and no update runs in this flight"),
                          "excludes": ["the origin flight's training time"]}
        return out

    # -- collation ---------------------------------------------------------
    def collate(self, route, views):
        capabilities.require_token_likelihood(route, "FlightServices.collate")
        if route == ROUTE_CAUSAL:
            return data_lib.collate_causal(views, pad_id=self.pad_id(route),
                                           condition_spec=self.condition_spec)
        return data_lib.collate_masked(views, pad_id=self.pad_id(route),
                                       condition_spec=self.condition_spec)

    # -- the loop ----------------------------------------------------------
    def fresh_state(self, seed):
        """A training state with every generator this loop consumes, seeded apart."""
        return TrainingState(mask_rng=np.random.default_rng([int(seed), 1]),
                             sampler_rng=np.random.default_rng([int(seed), 2]),
                             decode_rng=np.random.default_rng([int(seed), 3]))

    def training_identity(self, route, *, arm, seed, schedule_sha256=None, pinned=None):
        """The identity a checkpoint is bound to. A resume against another is refused.

        ``pinned`` is an admission pin (``her2_routes_admission``) and is ``None`` everywhere
        except the admission phase's scoring of an imported checkpoint. It replaces EXACTLY the
        three fields in ``admission.OVERRIDDEN_IDENTITY_FIELDS`` with the origin flight's pinned
        values; route, arm, seed, vocabulary, split, schedule and native base stay this root's own.
        """
        marker = self.context.path(spec.EXTERNAL_MARKER)
        snapshot = self.context.path(spec.FREEZE_MARKER)
        vocabulary = self.context.path("condition_vocabulary.json")
        split = self.context.path(spec.SPLIT_FREEZE)
        fields = {"config_sha256": paths.sha256_file(self.context.config_path),
                  "snapshot_sha256": (paths.read_json(snapshot)["snapshot_sha256"]
                                      if snapshot.is_file() else None),
                  "external_sha256": (paths.read_json(marker)["external_sha256"]
                                      if marker.is_file() else None)}
        if pinned is not None:
            fields.update(admission.pinned_identity_fields(pinned))
        return checkpoints.training_identity(
            route=route, arm=arm, seed=int(seed), **fields,
            condition_sha256=(paths.sha256_file(vocabulary) if vocabulary.is_file() else None),
            split_sha256=(paths.digest_document(paths.read_json(split))
                          if split.is_file() else None),
            schedule_sha256=schedule_sha256, native_sha256=self.native_identity(route))

    # -- admission (opt-in; see her2_routes_admission) ---------------------
    def admission_policy(self):
        """The admission-phase policy of this flight's config, or ``None``. Cached."""
        if "admission_policy" not in self._cache:
            phase = spec.active_phase(self.config)
            self._cache["admission_policy"] = phase if spec.is_admission_phase(phase) else None
        return self._cache["admission_policy"]

    def admitted_pin(self, route, checkpoint):
        """The pin for ``checkpoint`` iff it is exactly this root's admitted ``best.pt`` of ``route``.

        ``None`` for every other path, route or config, which is what keeps the foreign-identity
        branch unreachable for anything that was not admitted: another file in the same directory,
        the same bytes under another name, and every route of a flight without the phase.
        """
        policy = self.admission_policy()
        if policy is None or checkpoint is None or route not in policy["admitted_routes"]:
            return None
        target = self.context.path("bakeoff", route, "best.pt")
        if Path(checkpoint).resolve() != target.resolve():
            return None
        return spec.admission_pins(self.config)[route]

    def require_admitted_load(self, route, checkpoint, pin, payload):
        """Re-verify an admitted checkpoint against its committed pin. Run on EVERY load."""
        problems = admission.loadable_problems(route, checkpoint, pin, live_config=self.config)
        problems.extend(admission.record_payload_problems(
            Path(checkpoint).parent / admission.ADMISSION_FILE, payload))
        if int(payload.get("step", -1)) != int(pin["checkpoint_step"]):
            problems.append(f"the payload is step {payload.get('step')!r}, the pin says "
                            f"{pin['checkpoint_step']}")
        require(not problems,
                f"{checkpoint} is not the admitted {route} checkpoint: " + "; ".join(problems))

    def bakeoff_record(self, route):
        """What the gate needs of a route's bakeoff: ``best`` and whether it was trained here."""
        directory = self.context.path("bakeoff", route)
        marker = directory / admission.ADMISSION_FILE
        if marker.is_file():
            record = admission.read_record(marker)
            pin = spec.admission_pins(self.config).get(route)
            require(pin is not None and record.get("pin") == dict(pin),
                    f"{marker}: the record's embedded pin differs from the pin committed in the "
                    "config")
            return {"best": record["best"], "admitted": True, "trained": False}
        return paths.read_json(directory / "run.json")

    def require_not_admitted(self, directory):
        """The guard every training entry point passes through before it writes anything.

        Training into a directory that holds an admitted checkpoint would overwrite it. Under the
        admission phase the directory is ALSO resolved and checked against EVERY pinned origin
        flight: a junction at ``bakeoff/<route>`` (or at its ``density_only`` child) that points
        into a finished flight has no ``admission.json`` to find, so the marker check alone lets
        the trainer write progress and checkpoints into that flight. The same check protects
        ``admit_route``; this is the one for the routes that are trained here.
        """
        directory = Path(directory)
        marker = directory / admission.ADMISSION_FILE
        require(not marker.exists(),
                f"{directory} holds an admitted checkpoint ({admission.ADMISSION_FILE}); "
                "admission never trains over or removes it")
        if spec.ADMISSION_PHASE_KEY in self.config:
            admission.require_disjoint(directory, spec.admission_pins(self.config))

    def admit_route(self, route, pin):
        """Verify one finished route against its pin, copy its ``best.pt`` and write the record.

        Order matters: the origin flight (config, markers, external table, vocabulary, split,
        checkpoint bytes) is compared with this root first; the checkpoint's own stored identity
        is then compared with the identity this root would expect for it (this root's live values
        plus the three pinned ones, never the checkpoint's own claim); only then is anything
        written. The origin flight is only read, and a destination that is, sits in or contains it
        is refused first.
        """
        directory = self.context.path("bakeoff", route)
        # The run root AND the resolved bakeoff directory, each against EVERY pinned origin (plus
        # the pin handed in): a junction at ``bakeoff/<route>`` that points into the origin passes
        # the first check, lets ``copy_verified`` accept the origin's own file as already copied,
        # and would then write the record into a finished flight; one that points into ANOTHER
        # route's origin passes a check made against this route's pin alone.
        every_pin = {**spec.admission_pins(self.config), route: pin}
        admission.require_disjoint(self.context.run_root, every_pin)
        admission.require_disjoint(directory, every_pin)
        problems = admission.origin_problems(pin, route, live=admission.live_values(self.context))
        require(not problems, f"{route} cannot be admitted: " + "; ".join(problems))
        source = admission.origin_checkpoint_path(pin, route)
        payload = torch.load(str(source), map_location="cpu", mmap=True, weights_only=False)
        expected = self.training_identity(route, arm=pin["arm"], seed=pin["seed"],
                                          schedule_sha256=pin["schedule_sha256"], pinned=pin)
        checkpoints.require_identity(payload["identity"], expected, path=source)
        require(int(payload["step"]) == int(pin["checkpoint_step"]),
                f"{source} is step {payload['step']}, the pin says {pin['checkpoint_step']}")
        best = {"best_step": payload.get("best_step"), "best_metric": payload.get("best_metric")}
        destination = admission.copy_verified(source, directory / "best.pt",
                                              sha256=pin["checkpoint_sha256"],
                                              nbytes=pin["checkpoint_bytes"])
        record = admission.build_record(route, pin, origin_root=pin["origin_flight"],
                                        destination=destination, best=best)
        record_path = directory / admission.ADMISSION_FILE
        admission.write_record(record_path, record)
        return {"route": route, "admitted": True, "trained": False, "selected_on": "development",
                "rule": "best", "selector": pin["selector"], "step": int(pin["checkpoint_step"]),
                "best": best, "checkpoint": str(destination), "sha256": pin["checkpoint_sha256"],
                "record": str(record_path), "origin_flight": pin["origin_flight"]}

    def cadence(self, *, milestones=()):
        storage = self.config["storage"]
        optimization = self.config["optimization"]
        return checkpoints.Cadence(
            save_every=int(optimization.get("save_every", 100)),
            milestones=tuple(int(value) for value in milestones),
            bytes_per_checkpoint=int(storage["bytes_per_checkpoint"]))

    def task_settings(self):
        block = self.config["curriculum"]
        return {"kappa": float(block["kappa"]),
                "preference_weight": float(block.get("preference_weight", 1.0)),
                "local_reconstruction_weight": float(
                    block.get("local_reconstruction_weight", 0.25)),
                "interaction_weight": float(block.get("interaction_weight", 0.5)),
                "max_joint_per_update": int(block.get("max_joint_per_update", 1))}

    def train(self, route, *, stream, updates, seed, directory, arm="full", heartbeat=None,
              resume=True, milestones=(), evaluate=None):
        """Train one route. Returns the run record; checkpoints go under ``directory``.

        ``stream(step, micro, rng)`` yields a :class:`~smallAntibodyGen.experiments.
        her2_routes_tasks.TaskDraw`. The loop owns the optimizer, the accumulation,
        the composed warm-up/plateau schedule, the ledger and the checkpoints; the
        stream owns which task and which family each microbatch draws from, and
        :func:`her2_routes_tasks.dispatch` owns which objective the arm runs on it.

        The coefficient is looked up under the name of the objective that **ran**,
        not the task that was scheduled. That is what stops a control arm from
        reporting a reconstruction loss under the name of a head loss: an arm that
        removes the head objective trains reconstruction on the identical rows, so
        the row presentations match exactly and only the objective differs.

        This is the token-likelihood loop. The energy route is refused here, before any bank,
        model, optimizer or directory exists; it trains through :meth:`train_energy`.
        """
        capabilities.require_token_likelihood(route, "FlightServices.train")
        directory = Path(directory)
        self.require_not_admitted(directory)
        policy = spec.training_phase(self.config)
        if policy is not None:
            # Central: nothing under the phase trains with the legacy selector or none at all,
            # and nothing outside the phase's routes trains.
            require(route in policy["routes"],
                    f"the {policy['phase_id']} phase trains only {policy['routes']}")
            require(evaluate is not None and getattr(evaluate, "selection_metric", None)
                    == policy["selection_metric"],
                    f"the {policy['phase_id']} phase selects checkpoints with "
                    f"{policy['selection_metric']!r}; pass services.selection_metric(route)")
        # The shape is validated BEFORE any model is built, for every arm alike: the arms share this
        # curriculum, so a microbatch that cannot carry its local recipe is a failure of all of
        # them and is raised once, here, with nothing loaded.
        shapes = self.selected_shapes()
        directory.mkdir(parents=True, exist_ok=True)
        from ..models.her2_routes_heads import HeadAccumulator
        coefficients = controls.arm_coefficients(arm, base=self.config["curriculum"]["weights"])
        model = self.build_route(route, seed=seed)
        heads = self.build_heads(route, model)
        parameters = [p for p in list(model.parameters()) + list(heads.parameters())
                      if p.requires_grad]
        optimization = self.config["optimization"]
        optimizer = self.build_optimizer(route, parameters, lr=self.route_learning_rate(route),
                                         betas=tuple(optimization["betas"]),
                                         weight_decay=float(optimization["weight_decay"]))
        state = self.fresh_state(seed)
        # The plateau reduction is a factor INSIDE the lambda, so a cut survives the
        # next scheduler.step(). Multiplying param_group["lr"] and leaving LambdaLR
        # to recompute from the base rate is how a recorded reduction never happened.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, checkpoints.warmup_plateau_lambda(
                state, warmup=int(optimization["warmup"]),
                plateau_factor=float(optimization["plateau_factor"])))
        identity = self.training_identity(route, arm=arm, seed=seed)
        checkpoint = directory / "last.pt"
        resume_record = None
        if resume and checkpoint.is_file():
            state, resume_record = checkpoints.restore(
                checkpoint, model=model, heads=heads, optimizer=optimizer, scheduler=scheduler,
                state=state, identity=identity)
            scheduler.lr_lambdas = [checkpoints.warmup_plateau_lambda(
                state, warmup=int(optimization["warmup"]),
                plateau_factor=float(optimization["plateau_factor"]))]
        accumulation = int(shapes["accumulation"])
        cadence = self.cadence(milestones=milestones)
        settings = self.task_settings()
        accumulator = HeadAccumulator()
        progress = paths.StageProgress(directory / "progress.json", stage=f"train::{route}::{arm}",
                                       total=int(updates), every=25)
        progress.start()
        model.train()
        heads.train()
        try:
            while state.step < int(updates):
                state.step += 1
                optimizer.zero_grad(set_to_none=True)
                totals, terms, drawn = 0.0, {}, []
                for micro in range(accumulation):
                    draw = stream(state.step, micro, state.sampler_rng)
                    state.sampler_position += 1
                    if draw is None or draw.rows() == 0:
                        continue
                    outcome = tasks_lib.dispatch(
                        self, route, model, heads, draw, arm=arm, settings=settings,
                        rng=state.mask_rng, accumulator=accumulator)
                    weight = float(coefficients.get(outcome.term, 0.0))
                    if weight > 0.0:
                        scaled = weight * outcome.loss / accumulation
                        scaled.backward()
                        totals += float(scaled.detach())
                    for name, value in outcome.terms.items():
                        terms[name] = terms.get(name, 0.0) + float(value)
                    # The exposure is recorded whether or not the coefficient is
                    # nonzero, because the rows were drawn and presented either way
                    # and the matched-exposure comparison is about rows. It is recorded
                    # through the outcome's per-source contributions, so a batch drawn
                    # across several sources is attributed row by row and the globals
                    # and the breakdowns come from the same numbers.
                    state.ledger.record_outcome(task=draw.task, outcome=outcome,
                                                update=micro == 0)
                    drawn.append({"task": draw.task, "term": outcome.term,
                                  "coefficient": weight, "family": draw.family,
                                  "rows": outcome.rows})
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(optimization["gradient_clip"]), error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                entry = {"step": state.step, "loss": totals, "terms": dict(terms),
                         "gradient_norm": float(norm),
                         "learning_rate": float(scheduler.get_last_lr()[0]),
                         "microbatches": drawn}
                if evaluate is not None and (state.step % int(optimization["eval_every"]) == 0
                                             or state.step == int(updates)):
                    metric = evaluate(model, heads, state.step)
                    metric = None if metric is None else float(metric)
                    entry["development_metric"] = metric
                    diagnostics = getattr(evaluate, "last_diagnostics", None)
                    if diagnostics is not None:
                        # A selector that reports more than one number keeps the rest with the
                        # step, so it survives the checkpoint history and a resume.
                        entry["selection_diagnostics"] = _json_safe(diagnostics)
                    state = self._plateau(state, metric, model=model, heads=heads,
                                          optimizer=optimizer, scheduler=scheduler,
                                          directory=directory, identity=identity)
                state.history.append(entry)
                progress.advance(f"step {state.step}", completed=state.step)
                if heartbeat is not None:
                    heartbeat.beat(phase="train", route=route, arm=arm, step=state.step,
                                   loss=totals)
                if cadence.is_milestone(state.step):
                    checkpoints.save(directory / f"step{state.step}.pt", model=model,
                                     heads=heads, optimizer=optimizer, scheduler=scheduler,
                                     state=state, identity=identity)
                if cadence.should_save_last(state.step, total=int(updates)):
                    checkpoints.save(checkpoint, model=model, heads=heads, optimizer=optimizer,
                                     scheduler=scheduler, state=state, identity=identity)
            checkpoints.save(checkpoint, model=model, heads=heads, optimizer=optimizer,
                             scheduler=scheduler, state=state, identity=identity)
            progress.finish()
        except BaseException as error:                    # noqa: BLE001 - recorded, re-raised
            progress.finish("failed", error=f"{type(error).__name__}: {error}")
            raise
        record = {"route": route, "arm": arm, "seed": int(seed), "updates": int(state.step),
                  "history": list(state.history), "state": state.document(),
                  "coefficients": coefficients,
                  "resume": resume_record,
                  "identity": dict(identity),
                  "shapes": dict(shapes),
                  "cadence": cadence.document(
                      disk_budget_bytes=int(self.config["storage"]["min_free_bytes"])),
                  "task_settings": dict(settings),
                  "objectives_wired": {
                      data_lib.TASK_RECONSTRUCTION: (
                          "masked negative log likelihood over randomly masked positions of an "
                          "unlabelled paired repertoire row at bin=null"),
                      data_lib.TASK_WHOLE: (
                          "masked negative log likelihood over the declared COMMON region under "
                          "the assigned quality bin, plus softplus(-(log q(x | assigned) - "
                          "log q(x | contrast))) on the same row and the same targets"),
                      data_lib.TASK_LOCAL: (
                          "squared error between the decoder's ABSOLUTE conditional log odds and "
                          "kappa * measured margin, plus retained reconstruction of both "
                          "alternatives, plus the sequentially revealed 400-way joint fitted to "
                          "the measured four-cell quartet distribution where one is complete"),
                      data_lib.TASK_HEADS: (
                          "ObservationHeads on the full unmasked variant at bin=null: cumulative "
                          "logit for the sorted display bins, interval-censored Gaussian PER "
                          "REPLICATE with that replicate's own censoring and the combined "
                          "observation scale, and a within-assay ranking loss for enrichment")},
                  "arm_objectives": dict(tasks_lib.ARM_OBJECTIVES[arm]),
                  "head_accumulator": accumulator.document(),
                  "exposure": state.ledger.document(),
                  "declared_family_mix": dict(self.config["curriculum"]["family_mix"]),
                  "best": {"metric": state.best_metric, "step": state.best_step},
                  "plateau_reductions": state.plateau_reductions}
        if policy is not None:
            record["training_phase"] = {
                "phase_id": policy["phase_id"], "policy_sha256": policy["policy_sha256"],
                "admission": policy["admission"], "selection_metric": policy["selection_metric"],
                "head_diagnostics": policy["head_diagnostics"],
                "note": ("training-only: the selector is not the legacy development metric and "
                         "this run is not competition admission")}
        paths.write_json(directory / "run.json", record)
        return record

    def _plateau(self, state, metric, *, model, heads, optimizer, scheduler, directory,
                 identity):
        """Best-checkpoint selection and the learning-rate reduction. One place."""
        optimization = self.config["optimization"]
        if state.improved(metric, delta=float(optimization["plateau_delta"])):
            state.best_metric, state.best_step = float(metric), state.step
            state.plateau_bad_evaluations = 0
            checkpoints.save(directory / "best.pt", model=model, heads=heads,
                             optimizer=optimizer, scheduler=scheduler, state=state,
                             identity=identity)
            return state
        if metric is None:
            return state
        if state.plateau_reductions >= int(optimization["max_plateau_reductions"]):
            return state
        state.plateau_bad_evaluations += 1
        if state.plateau_bad_evaluations < int(optimization.get("plateau_patience_evaluations", 4)):
            return state
        state.plateau_reductions += 1
        state.plateau_bad_evaluations = 0
        return state

    # -- native evidence ---------------------------------------------------
    def native_evidence(self, route, *, home_cache_baseline=None):
        """Parity, parameter counts and gradient reach for one route, measured.

        Each probe here runs against a model whose added tables have been moved
        **off** zero where the property under test is otherwise vacuous. A
        conditioning-parity check at zero-initialised tables passes whatever the
        seam does, and a decode-parity check at a zero pointer table passes
        whatever the decoder injects; those were the two checks that missed real
        defects, so they are run twice -- at zero and after a seeded perturbation.
        """
        model = self.build_route(route)
        record = {"route": route}
        gradient_context = None
        if route == ROUTE_MASKED:
            from ..models.ablang2_policy import attention_scaling_record
            tokens = torch.tensor([[0, 14, 15, 22, 25, 0, 14, 22]], dtype=torch.long,
                                  device=self.device)
            conditions = [[0, 0, 0]]
            record["parity_at_zero"] = model.native_parity(tokens)
            record["parameters"] = model.parameter_report()
            record["attention_scaling"] = attention_scaling_record(
                type("N", (), {"blocks": list(model.native.AbRep.encoder_blocks)})())
            record["load"] = dict(model.load_record)
            record["device"] = model.assert_device_consistency()
            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(919)
                perturbation = torch.randn(model.condition_bin.weight.shape,
                                           generator=generator).to(
                                               model.condition_bin.weight.device) * 0.1
                model.condition_bin.weight.add_(perturbation)
            record["reference_parity_nonzero_tables"] = model.reference_parity(
                model.frozen_reference(), tokens, condition_indices=conditions)
            with torch.no_grad():
                model.condition_bin.weight.zero_()
        elif route in DPLM_ROUTES:
            # Public (AbLang-id) tokens, translated once inside the model: <VH>|<VL> with one
            # masked residue at slot 3.
            tokens = torch.tensor([[0, 14, 15, 23, 22, 25, 0, 14, 22],
                                   [0, 14, 15, 23, 22, 25, 0, 14, 22]], dtype=torch.long,
                                  device=self.device)
            conditions = [[0, 0, 0], [0, 0, 0]]
            record["parity_at_zero"] = model.native_parity(tokens)
            record["parameters"] = model.parameter_report()
            record["load"] = dict(model.load_record)
            record["device"] = model.assert_device_consistency()
            record["dtype_policy"] = model.dtype_policy()
            record["diffusion_settings"] = model.diffusion.document()
            if model.is_lora:
                record["lora_noop_parity"] = model.lora_noop_parity(tokens)
                record["lora"] = model.lora_settings.document()
            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(919)
                perturbation = torch.randn(model.condition_bin.weight.shape,
                                           generator=generator).to(
                                               model.condition_bin.weight.device) * 0.1
                model.condition_bin.weight.add_(perturbation)
            record["reference_parity_nonzero_tables"] = model.reference_parity(
                model.frozen_reference(), tokens, condition_indices=conditions)
            with torch.no_grad():
                model.condition_bin.weight.zero_()
            model.eval()
            probe = model.iterative_decode(
                tokens, [[3], [3]], condition_indices=conditions,
                generator=torch.Generator(device=self.device).manual_seed(919), steps=3)
            model.train()
            record["decode_probe"] = {**probe.document(), "slots": [3],
                                      "decoded_canonical": probe.canonical.tolist()}
        elif route == ROUTE_CAUSAL:
            from .her2_routes_conditions import BIN_NULL, Condition
            from .her2_routes_serial import causal_serialization, her2_construct
            construct = her2_construct(self.scaffold(), "WGGDGFYAMD", record_id="parity")
            serialization = causal_serialization(
                construct, residues=construct.residues(construct.sites))
            record["parameters"] = model.parameter_report()
            record["device"] = model.assert_device_consistency()
            record["native_token_parity"] = model.native_token_parity(
                [[23, *[model.vocabulary[residue] for residue in
                         construct.heavy + construct.light], 24]])
            record["vocabulary_invariance"] = model.vocabulary_invariance(
                [serialization.tokens], [serialization.query_slots],
                pointers=[serialization.pointers])
            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(919)
                model.pointer.weight.copy_(torch.randn(
                    model.pointer.weight.shape, generator=generator).to(
                        model.pointer.weight.device) * 0.15)
            record["step_score_parity_nonzero_pointer"] = model.step_score_parity(
                construct, order=construct.sites,
                condition=Condition(family="affinity", bin=BIN_NULL, assay="null"),
                residues=construct.residues(construct.sites))
            with torch.no_grad():
                model.pointer.weight.zero_()
            record["growth"] = dict(getattr(model, "growth", {}))
        else:
            from ..models.her2_routes_energy import build_energy_context
            from .her2_routes_serial import her2_construct
            construct = her2_construct(self.scaffold(), "WGGDGFYAMD", record_id="energy")
            context = build_energy_context([construct]).to(self.device)
            record["context_permutation"] = model.context_encoder.permutation_sensitivity(
                context)
            gradient_context = build_energy_context(
                [construct, construct],
                condition_indices=[[0, 0, 0], [0, 0, 0]]).to(self.device)
            model.freeze_denoiser()
            record["memory"] = model.memory_report()
            record["contract"] = model.document()
        heads = None if route == ROUTE_ENERGY else self.build_heads(route, model)
        record["gradient"] = self.gradient_reach(route, model, context=gradient_context,
                                                 heads=heads)
        record["home_cache"] = assert_no_home_cache(baseline=home_cache_baseline)
        return record

    def gradient_reach(self, route, model, *, context=None, heads=None):
        """Does the loss reach the trunk, the heads and the added tables? Measured.

        The energy route is measured WITH its context, because the question this
        answers for that route is whether the context tower is in the graph at all.
        An unconditioned probe reports every context parameter as dead and the
        reader cannot tell that from a defect.

        When ``heads`` is supplied the observation-head objective is added, and its
        parameters are reported separately: a head trained on a detached
        representation still has gradients of its own, so "the head has gradients" and
        "the head's gradient reaches the trunk" are two findings and this records
        both.
        """
        if route == ROUTE_ENERGY:
            clean = torch.zeros(2, model.window, model.residues, device=self.device)
            clean[:, :, 0] = 1.0
            model.freeze_denoiser()
            loss, _ = model.energy_loss(clean, context, fake=clean + 0.1)
        else:
            views = self.smoke_views(route, rows=2)
            batch = self.collate(route, views)
            log_probs = self.canonical_log_probs(route, model, batch)
            targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
            mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
            loss, _ = objectives_lib.masked_nll(log_probs, targets, mask)
        head_record = None
        if heads is not None and route != ROUTE_ENERGY:
            head_record = self.head_gradient_probe(route, model, heads)
            loss = loss + head_record.pop("loss")
        model.zero_grad(set_to_none=True)
        if heads is not None:
            heads.zero_grad(set_to_none=True)
        loss.backward()
        reached, dead, frozen = [], [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                # A frozen base (the LoRA route) has no gradient by construction; counting
                # it as "dead" would bury the adapters and tables that should have one.
                frozen.append(name)
            elif parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0:
                dead.append(name)
            else:
                reached.append(name)
        head_reached = []
        if heads is not None:
            head_reached = [name for name, parameter in heads.named_parameters()
                            if parameter.grad is not None
                            and float(parameter.grad.abs().sum()) > 0.0]
        return {"parameters_with_gradient": len(reached), "parameters_without": len(dead),
                "without_gradient": dead[:20],
                **({"frozen_parameters": len(frozen)} if frozen else {}),
                "trunk_reached": any("encoder" in name or "layers" in name or "blocks" in name
                                     for name in reached),
                "head_parameters_with_gradient": len(head_reached),
                "head_probe": head_record,
                "head_gradient_reaches_trunk": (None if head_record is None
                                                else bool(head_record["trunk_reached"])),
                "expected_zero": ("the zero-initialised condition and pointer tables receive zero "
                                  "gradient only if the batch never uses them; a table that is "
                                  "used and still has no gradient is a defect")}

    def head_gradient_probe(self, route, model, heads):
        """Does the observation-head loss reach the TRUNK? Measured on a real head batch.

        Built from the smoke constructs with a synthetic point measurement, because
        this probe runs in preflight before any outcome has been opened. What it tests
        is the wiring -- representation, head, gradient path -- not a measured value.
        """
        summaries = []
        for construct, serialization, _condition in self.smoke_views(route, rows=2):
            summaries.append(data_lib.ConstructSummary(
                construct=construct, study="probe", assay=self.condition_spec.assay_order[0],
                target="probe", parent="probe::parent", family="affinity", kind="point",
                measurements=(data_lib.ReplicateMeasurement(
                    value=8.0, lower=8.0, upper=8.0, censoring="none", replicate=1,
                    measurement_id="probe"),),
                region=tuple(construct.sites), partition="train", scale="pKD",
                source="probe"))
            del serialization
        views = data_lib.head_views(summaries, condition_spec=self.condition_spec,
                                    route=self._kind(route),
                                    max_positions=self._positions(route))
        batch = data_lib.collate_heads(views, pad_id=self.pad_id(route),
                                       condition_spec=self.condition_spec,
                                       assay_index=self.assay_index())
        outcome = tasks_lib.head_outcome(self, route, model, heads, batch)
        trunk = torch.autograd.grad(
            outcome.loss, [p for p in model.parameters() if p.requires_grad],
            retain_graph=True, allow_unused=True)
        reached = sum(1 for grad in trunk
                      if grad is not None and float(grad.abs().sum()) > 0.0)
        return {"loss": outcome.loss, "eligible_rows": outcome.diagnostics["eligible_rows"],
                "trunk_parameters_reached": reached, "trunk_reached": reached > 0,
                "measurement": "synthetic; this probe tests the wiring, not a measured value",
                "why": ("a head trained on a DETACHED representation still has gradients of its "
                        "own, so 'the head has gradients' and 'the head's gradient reaches the "
                        "trunk' are two findings")}

    # -- views and smoke ---------------------------------------------------
    def smoke_views(self, route, *, rows=4):
        """A handful of real HER2 views. Real scaffold, real serialisation, tiny count."""
        from .her2_routes_serial import (
            ablang_serialization,
            causal_serialization,
            her2_construct,
        )
        cores = ["WGGDGFYAMD", "WGGDGFYAMA", "WAGDGFYAMD", "WGGDGFYAMV"][:int(rows)]
        condition = Condition(family="affinity", bin=BIN_NULL, assay="null")
        views = []
        for index, core in enumerate(cores):
            construct = her2_construct(self.scaffold(), core, record_id=f"smoke{index}")
            sites = her2_core_edit_map()
            serialization = (causal_serialization(construct, order=sites,
                                                  residues=construct.residues(sites))
                             if route == ROUTE_CAUSAL
                             else ablang_serialization(construct, order=sites))
            views.append((construct, serialization, condition))
        return tuple(views)

    def smoke(self, route, *, directory, updates=None, seed=20260927):
        """A miniature end-to-end run against the production code, plus the negatives.

        Real model, real optimizer, real objectives, real journals, tiny sizes. It
        runs the **production curriculum stream**, so all four tasks and all three
        families are exercised, and it asserts that the resumed trajectory matches
        the uninterrupted one step for step rather than only reaching the same step
        count -- a resume that re-seeded its masking reaches a similar place by a
        different path and a step-count check does not see it.
        """
        directory = Path(directory)
        updates = int(updates or self.config["smoke"]["updates"])
        if route == ROUTE_ENERGY:
            return self.smoke_energy(directory=directory, seed=seed, steps=updates)
        stream = self.curriculum_stream(route, seed=seed, updates=updates)
        first = self.train(route, stream=stream, updates=int(updates), seed=seed,
                           directory=directory / "first", resume=False,
                           evaluate=self.selection_metric(route))
        interrupted = self.train(route, stream=self.curriculum_stream(
            route, seed=seed, updates=updates), updates=max(1, int(updates) - 1),
            seed=seed, directory=directory / "resumed", resume=False,
            evaluate=self.selection_metric(route))
        resumed = self.train(route, stream=self.curriculum_stream(
            route, seed=seed, updates=updates), updates=int(updates), seed=seed,
            directory=directory / "resumed", resume=True,
            evaluate=self.selection_metric(route))
        trajectory = checkpoints.resumed_trajectory_report(
            first["history"], resumed["history"],
            tolerance=float(self.config["smoke"].get("trajectory_tolerance", 1e-4)))
        terms = sorted({name for entry in first["history"] for name in entry["terms"]})
        tasks_seen = sorted({micro["task"] for entry in first["history"]
                             for micro in entry["microbatches"]})
        objectives_seen = sorted({micro["term"] for entry in first["history"]
                                  for micro in entry["microbatches"]})
        checks = {
            "trained": first["updates"] == int(updates),
            "resume_reaches_same_step": resumed["updates"] == first["updates"],
            "resume_restored_exposure": (resumed["exposure"]["totals"]["microbatch_rows"]
                                         == first["exposure"]["totals"]["microbatch_rows"]),
            "resumed_trajectory_matches": trajectory["identical"],
            "interrupted_stopped_short": interrupted["updates"] < int(updates),
            "loss_finite": all(math.isfinite(entry["loss"]) for entry in first["history"]),
            "gradient_nonzero": any(entry["gradient_norm"] > 0 for entry in first["history"]),
            "more_than_one_objective_ran": len(objectives_seen) > 1,
            "run_record_serialises": self._serialises(first),
        }
        negatives = {}
        try:
            data_lib.require_answer_free([Condition(family="affinity", bin="high",
                                                    assay="null")])
            negatives["answer_free_guard"] = "DID NOT RAISE"
        except ValueError:
            negatives["answer_free_guard"] = "raised as required"
        try:
            controls.require_matched_exposure({"a": {"totals": {"microbatch_rows": 10,
                                                                "student_forwards": 10}},
                                               "b": {"totals": {"microbatch_rows": 8,
                                                                "student_forwards": 8}}})
            negatives["matched_exposure_guard"] = "DID NOT RAISE"
        except ValueError:
            negatives["matched_exposure_guard"] = "raised as required"
        try:
            checkpoints.require_identity({"config_sha256": "a"}, {"config_sha256": "b"},
                                         path="<smoke>")
            negatives["identity_guard"] = "DID NOT RAISE"
        except ValueError:
            negatives["identity_guard"] = "raised as required"
        checks["negative_cases_raise"] = all(value == "raised as required"
                                             for value in negatives.values())
        failures = sorted(name for name, value in checks.items() if not value)
        return {"route": route, "checks": len(checks), "passed": not failures,
                "failures": failures, "detail": checks, "negatives": negatives,
                "trajectory": trajectory,
                "observed": {"tasks": tasks_seen, "objectives": objectives_seen,
                             "loss_terms": terms,
                             "family_shares": first["exposure"].get(
                                 "realised_family_shares")},
                "updates": int(updates)}

    def _serialises(self, record):
        """Would this run record survive the stage writer? ``best_metric`` broke it once."""
        try:
            paths.canonical_json(record)
        except (TypeError, ValueError):
            return False
        return True

    def smoke_energy_resume(self, *, directory, seed=20260927, steps=2):
        """A miniature two-phase energy fit, interrupted and resumed. Trajectory compared.

        Reported as a pass or a failure rather than under ``not_checked``: the route
        has a resume now, so there is something for a resume check to exercise.
        """
        directory = Path(directory)
        # The smoke is the conditioned arm on the full declared population, stated rather than
        # defaulted: the population and the conditioning are separate, explicit arguments.
        arm = {"population": energy_contract.POPULATION_ALL_ELIGIBLE, "conditioned": True}
        full = self.train_energy(directory=directory / "full", seed=seed, resume=False,
                                 updates=int(steps) * 2, **arm)
        partial = self.train_energy(directory=directory / "resumed", seed=seed, resume=False,
                                    updates=int(steps) * 2, stop_after_updates=int(steps), **arm)
        resumed = self.train_energy(directory=directory / "resumed", seed=seed, resume=True,
                                    updates=int(steps) * 2, **arm)
        trajectories = {
            phase: checkpoints.resumed_trajectory_report(
                full["phases"][phase]["history"], resumed["phases"][phase]["history"],
                tolerance=1e-5)
            for phase in ("denoiser", "energy")}
        checks = {
            "phases_complete": resumed["phase_updates"] == full["phase_updates"],
            "resumed_further_than_partial": (sum(resumed["phase_updates"].values())
                                             > sum(partial["phase_updates"].values())),
            "resume_record_present": resumed["resume"] is not None,
            "exposure_restored": (resumed["exposure"]["totals"]["microbatch_rows"]
                                  >= partial["exposure"]["totals"]["microbatch_rows"]),
            "trajectories_identical": all(row["identical"] and row["compared_steps"] > 0
                                          for row in trajectories.values()),
            # The draws themselves, not only the losses: a resume that reset or skipped the trace
            # would show other rows under the same weights.
            "exposure_trace_continued": resumed["exposure_trace"] == full["exposure_trace"],
        }
        failures = sorted(name for name, value in checks.items() if not value)
        return {"passed": not failures, "failures": failures, "detail": checks,
                "trajectories": trajectories,
                "full": full["phase_updates"], "partial": partial["phase_updates"],
                "resumed": resumed["phase_updates"]}

    def smoke_energy(self, *, directory, seed=20260927, steps=3):
        """A miniature real two-phase energy fit, with the frozen-denoiser check.

        The generic smoke path collates token views and asks the model for
        ``canonical_log_probs``; the energy route has neither, so running it there
        would be testing an interface this route does not have. What IS tested is
        what the route claims: a finite denoiser objective, a finite contrastive
        objective against a detached negative, gradient reaching the energy AND its
        context tower, and no gradient reaching the frozen denoiser or its own
        tower.
        """
        from ..models.her2_routes_energy import onehot_window
        from .her2_routes_serial import canonical_index
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        block = self.config["routes"][ROUTE_ENERGY]
        model = self.build_route(ROUTE_ENERGY, seed=seed)
        summaries = [row for row in self.eligible_summaries()
                     if len(row.construct.sites) == int(block["window"])][:2]
        require(summaries,
                f"No measured construct has a {block['window']}-site edit map for the energy "
                "smoke")
        windows = torch.as_tensor(
            [[canonical_index(row.construct.residue_at(site)) for site in row.construct.sites]
             for row in summaries], dtype=torch.long)
        clean = onehot_window(windows, residues=model.residues,
                             scale=model.scale).to(self.device)
        context = self.energy_context(summaries)
        generator = self.energy_generator(seed)
        denoiser_parameters = [p for name, p in model.named_parameters()
                               if name.startswith("denoiser")]
        optimizer = torch.optim.AdamW(denoiser_parameters, lr=1e-4)
        denoiser_losses = []
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = model.denoiser_loss(clean, context, generator=generator)
            loss.backward()
            optimizer.step()
            denoiser_losses.append(float(loss.detach()))
        denoiser_context_moved = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.denoiser_context.parameters())
        model.freeze_denoiser()
        energy_parameters = [p for name, p in model.named_parameters()
                             if not name.startswith("denoiser")]
        optimizer = torch.optim.AdamW(energy_parameters, lr=1e-4)
        energy_losses, diagnostics = [], {}
        model.zero_grad(set_to_none=True)
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = model.energy_loss(
                clean, context, generator=generator,
                sampling_options={"steps": int(block["profile_walk_steps"][0])})
            loss.backward()
            optimizer.step()
            energy_losses.append(float(loss.detach()))
        checks = {
            "denoiser_loss_finite": all(math.isfinite(value) for value in denoiser_losses),
            "denoiser_context_trained": denoiser_context_moved,
            "energy_loss_finite": all(math.isfinite(value) for value in energy_losses),
            "negative_detached": bool(diagnostics.get("negatives_detached")),
            "conditioned": bool(diagnostics.get("conditioned")),
            "energy_context_reached": any(
                p.grad is not None and float(p.grad.abs().sum()) > 0
                for p in model.context_encoder.parameters()),
            "frozen_denoiser_untouched": all(
                p.grad is None or float(p.grad.abs().sum()) == 0.0
                for module in (model.denoiser, model.denoiser_context)
                for p in module.parameters()),
        }
        resume = self.smoke_energy_resume(directory=directory / "resume", seed=seed,
                                          steps=max(1, int(steps) - 1))
        checks["resume_matches"] = bool(resume["passed"])
        failures = sorted(name for name, value in checks.items() if not value)
        paths.write_json(directory / "smoke_energy.json", {
            "route": ROUTE_ENERGY, "detail": checks,
            "denoiser_losses": denoiser_losses, "energy_losses": energy_losses,
            "resume": resume,
            "diagnostics": {key: value for key, value in diagnostics.items()
                            if not isinstance(value, dict)}})
        return {"route": ROUTE_ENERGY, "checks": len(checks), "passed": not failures,
                "failures": failures, "detail": checks, "negatives": {},
                "resume": resume}

    def profile_energy(self):
        """Profile the energy route across microbatch AND walk length, with context.

        Measured separately because the Langevin walk's cost scales with its step
        count -- each step is a forward plus a score backward through the energy --
        and because the route has no token-view interface for the generic profiler
        to collate. The context tower is in the graph, since profiling the
        unconditioned path would understate the configuration that actually runs.
        """
        from ..models.her2_routes_energy import onehot_window
        block = self.config["routes"][ROUTE_ENERGY]
        budget = self.config["profile"]
        model = self.build_route(ROUTE_ENERGY)
        model.freeze_denoiser()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=float(budget["probe_lr"]))
        from dataclasses import replace
        summaries = self.energy_rows()
        widest = max(summaries, key=lambda row: (len(row.region),
                     len(row.construct.heavy) + len(row.construct.light)))
        sites = data_lib._tiles(widest.region, int(self.config["curriculum"]["max_query_sites"]))[0]
        widest = replace(widest, construct=data_lib.with_sites(widest.construct, sites), region=sites)
        rows, baselines = [], {}
        for size in tuple(budget["energy_microbatches"]):
            for walk_steps in tuple(block["profile_walk_steps"]):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                indices = torch.randint(0, model.residues, (int(size), len(sites)))
                clean = onehot_window(indices, residues=model.residues,
                                     scale=model.scale).to(self.device)
                # Profiled WITH the context tower in the graph. The unconditioned
                # path is a different, cheaper computation, and a profile of it
                # would understate the memory of the configuration that runs.
                context_rows = [widest] * int(size)
                context = self.energy_context(context_rows)
                timings, failure = [], None
                try:
                    for index in range(3):
                        started = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        loss, _ = model.energy_loss(
                            clean, context, sampling_options={"steps": int(walk_steps)})
                        loss.backward()
                        optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if index > 0:
                            timings.append(time.perf_counter() - started)
                except torch.cuda.OutOfMemoryError as error:   # pragma: no cover
                    failure = f"OutOfMemoryError: {error}"
                median = None if not timings else float(np.median(timings))
                per_row = None if median is None else median / int(size)
                if walk_steps not in baselines and per_row is not None:
                    baselines[walk_steps] = per_row
                baseline = baselines.get(walk_steps)
                peak = (torch.cuda.max_memory_allocated() / 2 ** 20
                        if torch.cuda.is_available() else None)
                # AB-14: see the note in the language-route profiler above.
                on_gpu = self.device.startswith("cuda")
                vram = vram_reading()
                free = vram["cuda_free_mib"]
                slowdown = None if per_row is None or not baseline else per_row / baseline
                signals = memory_signals(vram, peak_mib=peak, slowdown=slowdown,
                                         budget=budget, on_gpu=on_gpu)
                rows.append({"microbatch": int(size), "walk_steps": int(walk_steps),
                             "median_step_seconds": median, "seconds_per_row": per_row,
                             "peak_allocated_mib": peak, "driver_free_mib": free,
                             "vram": vram, "probe_device": self.device,
                             "per_row_slowdown": slowdown, "failure": failure,
                             "signals": signals,
                             "accepted": bool(failure is None and signals_accepted(
                                 signals, on_gpu=on_gpu))})
        accepted = [row for row in rows if row["accepted"]
                    and row["walk_steps"] == int(block["walk_steps"])]
        chosen_size = max((row["microbatch"] for row in accepted), default=None)
        denoiser_seconds = None
        if chosen_size is not None:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("denoiser"))
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=float(budget["probe_lr"]))
            indices = torch.randint(0, model.residues, (chosen_size, len(sites)))
            clean = onehot_window(indices, residues=model.residues,
                                 scale=model.scale).to(self.device)
            context = self.energy_context([widest] * chosen_size)
            timings = []
            for index in range(3):
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                model.denoiser_loss(clean, context).backward()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if index:
                    timings.append(time.perf_counter() - started)
            denoiser_seconds = float(np.median(timings))
        del optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"route": ROUTE_ENERGY, "rows": rows,
                "chosen_microbatch": chosen_size,
                "denoiser_seconds_per_update": denoiser_seconds,
                "chosen_walk_steps": max((row["walk_steps"] for row in accepted), default=None),
                "accumulation_steps": None,
                "effective_batch": max((row["microbatch"] for row in accepted), default=None),
                "query_sites": len(sites),
                "profiled_assay": widest.assay,
                "three_signal_rule": "as for the other routes",
                "walk_graph_note": ("the walk takes its score with create_graph=False and "
                                    "detaches each step, so no autograd graph is retained across "
                                    "the chain -- upstream's create_graph=self.training built one "
                                    "and then discarded it at ys.detach(). Per-step activations "
                                    "and the score's own backward still cost time proportional to "
                                    "the walk length, which is what the walk_steps sweep above "
                                    "measures. The published ~4,735 MB ByteNet profile is above "
                                    "this card and is not reproduced."),
                "profiled_with_context": True}

    def ledger_sample(self, *, attempts=48):
        """Measure ``forwards_per_row`` on the REAL stream. Never a tabulated constant.

        Runs the production curriculum stream and the production objectives on a
        small number of microbatches, and reads the forward count each one actually
        performed off the outcome. An earlier revision wrote the ratio into a table
        -- ``local_effects: 2`` -- which is the arithmetic it was supposed to be
        checking: the whole-window task's contrast forward and the coupled quartet's
        twenty-one scored rows are both in the loop and neither was in the table.
        """
        ledger = data_lib.ExposureLedger()
        route = self.config["routes"]["enabled"][0]
        if route == ROUTE_ENERGY:
            route = next((name for name in self.config["routes"]["enabled"]
                          if name != ROUTE_ENERGY), route)
        model = self.build_route(route)
        heads = self.build_heads(route, model)
        stream = self.curriculum_stream(route, seed=int(self.config["split"]["seed"]),
                                        updates=int(attempts))
        settings = self.task_settings()
        rng = np.random.default_rng([int(self.config["split"]["seed"]), 41])
        observed, seen = {}, set()
        # Counting rows, targets and forwards needs no derivative: eval mode (the
        # loaded native is already eval, so layer checkpoints are off anyway) and no
        # autograd graph, so this bookkeeping cannot cost training-sized memory.
        model.eval()
        heads.eval()
        with torch.no_grad():
            for step in range(1, int(attempts) + 1):
                draw = stream(step, 0, rng)
                if draw is None or draw.rows() == 0:
                    continue
                outcome = tasks_lib.dispatch(self, route, model, heads, draw, arm="full",
                                             settings=settings, rng=rng)
                ledger.record_outcome(task=draw.task, outcome=outcome, update=True)
                observed.setdefault(draw.task, []).append(
                    outcome.forwards / max(1, outcome.rows))
                seen.add(draw.task)
                if len(seen) == len(data_lib.TASKS):
                    break
        del model, heads
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        document = ledger.document()
        document["measured_from"] = {task: float(np.mean(values))
                                     for task, values in sorted(observed.items())}
        document["tasks_observed"] = sorted(seen)
        document["tasks_not_observed"] = sorted(set(data_lib.TASKS) - seen)
        document["basis"] = ("forwards per row read off the real objectives on the real stream. "
                             "The ratio is measured because the whole-window contrast pass and the "
                             "coupled quartet's twenty-one scored rows are both in the loop and "
                             "neither appears in an arithmetic table.")
        return document

    # -- sources and split -------------------------------------------------
    def source_paths(self):
        """Resolved single-file paths for every enrolled source. Multi-member sources
        are resolved by :meth:`source_members` instead, and are skipped here rather
        than raising: a source declared as a list of members has no one path."""
        block = self.config["sources"]
        out = {}
        for name, entry in sorted(dict(block["paths"]).items()):
            logical = entry.get("logical")
            if not logical:
                continue
            root = entry.get("root", "artifact")
            out[name] = (self.context.artifact(logical) if root == "artifact"
                         else self.context.repository(logical))
        return out

    def build_sources(self, *, retain_observations=False):
        """Load and qualify every enrolled source, AFTER the sequence-only freeze.

        Ordering is the substance of this method, not a detail of it. The frozen
        split is read first and its train window set gates the HER2 loader, so the
        reserved measurements are never converted into values. Bin thresholds are
        then fitted over the **training** rows of each ``(family, assay, parent)``
        population only: a threshold fitted over every observation puts the
        held-out rows' own values into the definition of the bin they are later
        conditioned on.
        """
        from . import her2_routes_sources as sources
        from . import her2_routes_split as split_lib
        from .her2_routes_conditions import fit_bin_thresholds
        split = self.split()
        allowed_windows = frozenset(
            core for core, assignment in split.assignment.items()
            if assignment in (split_lib.TRAIN, split_lib.DEVELOPMENT))
        require(allowed_windows,
                "The frozen split assigns no window to train or development, so there is "
                "nothing to open outcomes for")
        paths_by_name = self.source_paths()
        loaded, groups = {}, {}
        scaffold = self.scaffold()
        sites = her2_core_edit_map()
        cache_root = self.source_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)

        def add(name, block):
            loaded[name] = block
            groups[name] = block.get("observations") or ()

        window_partitions = {core: assignment for core, assignment in split.assignment.items()
                             if assignment in (split_lib.TRAIN, split_lib.DEVELOPMENT)}
        if "alphabind_her2_validation" in paths_by_name:
            add("alphabind_her2_validation", sources.load_alphaseq_observations(
                paths_by_name["alphabind_her2_validation"],
                source="alphabind_her2_validation", scaffold=scaffold, sites=sites,
                allowed_windows=allowed_windows, window_partitions=window_partitions))
        for name in ("alphabind_aab_pp489", "alphabind_pembrolizumab", "alphabind_vhh72"):
            members = self.source_members(name)
            if members:
                add(name, sources.load_alphabind_nonher2(members, source=name))
        # A curated summary of a study whose PRIMARY release is also enrolled is
        # not a second experiment. Engelhart 2022 and the first Li release are one
        # AlphaSeq experiment; the curated FLAb Petersen table is a per-pair
        # summary of the barcode sheets. Enrolling both counts one study twice in
        # every mixture share and in every threshold fit, and the de-duplication
        # key cannot see it because the two arrive with different replicate and
        # assay structure. So the supersession is explicit and recorded.
        supersessions = {}
        for curated, primary in (("engelhart2022_s1", "li2023_s1_a"),
                                 ("petersen2024_ha", "petersen2024_primary")):
            if curated in paths_by_name and primary in paths_by_name:
                supersessions[curated] = {
                    "superseded_by": primary,
                    "study": sources.canonical_study(curated),
                    "reason": ("the primary release of the same study is enrolled. Loading both "
                               "double-counts one experiment.")}
        for name, loader in (("engelhart2022_s1", sources.load_engelhart),
                             ("koenig2017_g6", sources.load_koenig),
                             ("petersen2024_ha", sources.load_petersen_flab)):
            if name in paths_by_name and name not in supersessions:
                kwargs = ({"historical_path": paths_by_name.get("koenig2017_historical")}
                          if name == "koenig2017_g6" else {})
                add(name, loader(paths_by_name[name], **kwargs))
        li_targets = tuple(self.config["sources"].get("li_targets") or ()) or None
        # The first release carries the three MIT_* parent-control rows and the
        # second carries none, so the first is loaded first and its controls are
        # handed to the second through an explicit cross-release table. Without that
        # handoff the second release's two families have no background at all, and
        # an earlier revision gave them a placeholder edit site at H0.
        li_parents = {}
        for name, preamble in (("li2023_s1_a", 0),
                               ("li2023_s1_b", sources.LI2_PREAMBLE_LINES)):
            if name in paths_by_name:
                block = sources.load_li(paths_by_name[name], source=name, preamble=preamble,
                                        cache=cache_root, targets=li_targets,
                                        parents=dict(li_parents))
                add(name, block)
                li_parents.update(sources.li_parent_export(block["table"]))
        if "adams2017_fluorescein" in paths_by_name:
            add("adams2017_fluorescein",
                sources.load_adams_affinity(paths_by_name["adams2017_fluorescein"]))
        for name, column in (("adams2017_expression", "Expression [ER]"),
                             ("koenig2017_expression", "enrichment ratio")):
            if name in paths_by_name:
                add(name, sources.load_expression(paths_by_name[name], source=name,
                                                  column=column))
        if "adams2017_primary" in paths_by_name and loaded.get("adams2017_fluorescein"):
            # The curated table is still LOADED: it is the only source of the fixed
            # light chain, which the replicate files do not carry. Its measurements
            # are then withdrawn from the training groups, because the primary rows
            # are the same measurements with their replicate structure and their
            # correctly scaled uncertainty intact.
            light = sources.derive_adams_light(loaded["adams2017_fluorescein"]["observations"])
            add("adams2017_primary", sources.load_adams_primary(
                paths_by_name["adams2017_primary"], light=light))
            groups["adams2017_fluorescein"] = ()
            supersessions["adams2017_fluorescein"] = {
                "superseded_by": "adams2017_primary",
                "study": sources.canonical_study("adams2017_fluorescein"),
                "retained_for": "the fixed light chain, which the replicate files do not carry",
                "reason": ("the primary replicate tables are the same measurements with three "
                           "biological replicates and a correctly scaled sigma. The curated "
                           "summary's rows would be a fourth, averaged copy.")}
        if "petersen2024_parents" in paths_by_name and "petersen2024_primary" in paths_by_name:
            parents = sources.load_petersen_parents(paths_by_name["petersen2024_parents"])
            if "petersen2024_coding_parents" in paths_by_name:
                coding = sources.load_petersen_coding_parents(paths_by_name["petersen2024_coding_parents"])
                for key, sequence in coding["parents"].items():
                    require(key not in parents["parents"] or parents["parents"][key] == sequence,
                            f"Petersen parent exports disagree for {key}")
                    parents["parents"][key] = sequence
                parents["coding_export"] = {k: v for k, v in coding.items() if k != "parents"}
            pooled = paths_by_name.get("petersen2024_pooled")
            ha_path = pooled or paths_by_name["petersen2024_primary"]
            ha_sheets = ("Figure 4c",) if pooled else sources.PETERSEN_BARCODE_SHEETS
            add("petersen2024_primary", sources.load_petersen_primary(
                ha_path, parents=parents["parents"], sheets=ha_sheets))
            loaded["petersen2024_primary"]["parent_export"] = {
                key: value for key, value in parents.items() if key != "parents"}
            if pooled:
                supersessions["petersen2024_ha_barcode_fits"] = {
                    "superseded_by": "published Figure 4c variant fits with two biological replicates",
                    "reason": "barcode fits are components of the pooled fit, not extra biological replicates"}
            if "petersen2024_coding_parents" in paths_by_name:
                add("petersen2024_s1", sources.load_petersen_primary(
                    paths_by_name["petersen2024_primary"], parents=parents["parents"],
                    sheets=("4A8_CC121_combined",), target="SARS-CoV-2 S1"))
                supersessions["petersen2024_s1_barcode_fits"] = {
                    "superseded_by": "4A8_CC121_combined",
                    "reason": "published pooled variant fit replaces its constituent barcode fits"}
            if "petersen2024_ha" in loaded:
                loaded["petersen2024_primary"]["join_audit"] = sources.petersen_join_audit(
                    loaded["petersen2024_primary"]["observations"],
                    loaded["petersen2024_ha"]["observations"])
        if "cr9114" in paths_by_name and "cr9114_prepared" in paths_by_name:
            space = sources.CR9114EditSpace.from_prepared(paths_by_name["cr9114_prepared"])
            allowed_genotypes, genotype_splits = None, None
            if "cr9114_whitelist" in paths_by_name:
                pins = dict(self.config["sources"].get("cr9114_pins") or {})
                whitelist = split_lib.cr9114_eligibility_whitelist(
                    paths_by_name["cr9114_whitelist"],
                    expected_total=pins.get("eligible"),
                    expected_sha256=pins.get("eligible_file_sha256"))
                allowed_genotypes = frozenset(whitelist["eligible"])
                # The whitelist's own split column. Without it every CR9114 row is a
                # training row, including the 11,750 development genotypes, and the
                # train-only threshold claim is false for the largest landscape here.
                genotype_splits = dict(whitelist["eligible"])
                self._cache["cr9114_whitelist"] = whitelist
                for expected, key in ((pins.get("eligible_train"), "train"),
                                      (pins.get("eligible_development"), "development")):
                    if expected is None:
                        continue
                    observed = sum(1 for value in genotype_splits.values() if value == key)
                    require(observed == int(expected),
                            f"The whitelist assigns {observed} genotype(s) to {key!r} and the pin "
                            f"says {expected}. A partition count that moved is a changed release, "
                            "not a rounding difference.")
            for antigen, name in (("h3", "cr9114_h3"), ("fluB", "cr9114_flub")):
                add(name, sources.load_cr9114(paths_by_name["cr9114"], antigen=antigen,
                                              space=space,
                                              allowed_genotypes=allowed_genotypes,
                                              genotype_splits=genotype_splits))
        closure = sources.observation_closure(groups)
        observations = closure["observations"]
        heldouts = dict(self.config["sources"].get("withheld") or {})
        observations, heldout_record = sources.apply_declared_heldouts(
            observations, withheld_parents=tuple(heldouts.get("parents") or ()),
            withheld_studies=tuple(heldouts.get("studies") or ()))
        summaries = data_lib.summarise_observations(observations)
        display = self.display_pool()
        vocabulary = sources.derive_assay_vocabulary(
            observations,
            additional_training_counts=({display.assay: len(display)}
                                        if display is not None else {}))
        vocabulary_check = self.freeze_condition_vocabulary(vocabulary)
        parent_sequences = {}
        for name, block in sorted(loaded.items()):
            parent = block.get("parent") or {}
            if parent.get("heavy"):
                parent_sequences[f"{name}::parental"] = (parent["heavy"], parent.get("light") or "")
                parent_sequences[f"{name}::wild_type"] = (parent["heavy"],
                                                          parent.get("light") or "")
        parent_sequences.update(li_parents)
        self._cache["parent_sequences"] = parent_sequences
        train_summaries = [row for row in summaries if row.is_training]
        partition_counts = {}
        for row in summaries:
            partition_counts[row.partition] = partition_counts.get(row.partition, 0) + 1
        thresholds, bins, threshold_support = fit_condition_bins(
            summaries, minimum_support=int(self.config["conditions"]["minimum_bin_support"]),
            fit=fit_bin_thresholds)
        oas = self.paired_oas()
        display = self.display_pool()
        inventory = sources.source_inventory(loaded)
        inventory.update(closure={key: value for key, value in closure.items()
                                  if key != "observations"},
                         bin_thresholds=thresholds,
                         bin_threshold_support=threshold_support,
                         bins_assigned=len(bins),
                         summaries=len(summaries),
                         training_summaries=len(train_summaries),
                         partitions=dict(sorted(partition_counts.items())),
                         declared_heldouts=heldout_record,
                         assay_vocabulary=dict(vocabulary, assays=list(vocabulary["assays"])),
                         condition_vocabulary=vocabulary_check,
                         supersessions=dict(sorted(supersessions.items())),
                         measurement_kinds=data_lib.measurement_kinds(observations),
                         region_geometry={name: block.get("region") for name, block in
                                          sorted(loaded.items()) if block.get("region")},
                         paired_oas={key: value for key, value in oas.items()
                                     if key not in ("constructs", "by_split")},
                         paired_oas_by_split={name: len(value) for name, value
                                              in dict(oas["by_split"]).items()},
                         display_library=(None if display is None else
                                          {**display.document(),
                                           **dict(self._cache.get("display_record") or {})}),
                         parent_sequences=sorted(parent_sequences),
                         ordering=("the sequence-only split is frozen FIRST and its train "
                                   "window set gates the HER2 outcome read, so no reserved "
                                   "measurement is materialised by this stage"),
                         partition_rule=("every row carries the partition its own source "
                                         "declares: the frozen window assignment for HER2, the "
                                         "eligibility whitelist's split column for CR9114, and a "
                                         "declared parent/study heldout for the transfer panel. "
                                         "An absent declaration is train, and the counts above "
                                         "are what makes that statement checkable."),
                         threshold_rule=("thirds of the TRAIN-ONLY EXACT-scalar distribution of one "
                                         "(family, assay, parent) population, frozen, with an "
                                         "explicit tie rule and a minimum support; rows that are "
                                         "not exact points are counted per population in "
                                         "bin_threshold_support and shape no cut. A bin is "
                                         "assigned only when the whole measurement-or-censoring "
                                         "interval fits inside it."))
        if retain_observations:
            self._cache["observations"] = observations
        self._cache["summaries"] = summaries
        self._cache["training_summaries"] = tuple(train_summaries)
        self._cache["bins"] = bins
        self._cache["assay_vocabulary"] = vocabulary
        self._cache["source_inventory"] = inventory
        return inventory

    def source_members(self, name):
        """Resolved paths for a source declared as a LIST of members, or ``()``.

        The non-HER2 AlphaSeq populations arrive as a fine-tuning file and a
        validation file that are one population measured in two batches, so the
        config names both and the loader reads them together.
        """
        entry = dict(self.config["sources"]["paths"]).get(name)
        if not entry:
            return ()
        logicals = entry.get("logicals") or ([entry["logical"]] if entry.get("logical") else [])
        root = entry.get("root", "artifact")
        out = []
        for logical in logicals:
            target = (self.context.artifact(logical) if root == "artifact"
                      else self.context.repository(logical))
            if target.exists():
                out.append(target)
        return tuple(out)

    def observations(self):
        if "observations" not in self._cache:
            self.build_sources(retain_observations=True)
        return self._cache["observations"]

    def summaries(self):
        if "summaries" not in self._cache:
            self.build_sources()
        return self._cache["summaries"]

    def assay_vocabulary(self):
        if "assay_vocabulary" not in self._cache:
            self.build_sources()
        return self._cache["assay_vocabulary"]

    def freeze_condition_vocabulary(self, vocabulary):
        """Derive the condition spec from the enrolled data, then pin it.

        Written to ``condition_vocabulary.json`` on the first pass and compared
        against on every later one. A change in the derived vocabulary changes the
        embedding table shape, which would make every existing checkpoint
        unloadable; the campaign refuses it and names a new revision instead of
        resizing a table under a resumed run.
        """
        from .her2_routes_conditions import ConditionSpec
        block = self.config["conditions"]
        derived = list(vocabulary["assays"])
        require(derived, "The enrolled sources produced no assay identity to condition on")
        record_path = self.context.path("condition_vocabulary.json")
        record = {"schema_version": spec.ROUTES_SCHEMA, "record_kind": "condition_vocabulary",
                  "assays": derived, "counts": dict(vocabulary["counts"]),
                  "declared_in_config": sorted(self.declared_condition_spec.assays),
                  "declared_but_never_produced": sorted(
                      set(self.declared_condition_spec.assays) - set(derived)),
                  "produced_but_not_declared": sorted(
                      set(derived) - set(self.declared_condition_spec.assays)),
                  "rule": ("the vocabulary is DERIVED from the enrolled training partitions and "
                           "pinned here. The config's list sizes the preflight probes only; a "
                           "name it declares that the data never produces would keep an "
                           "untrained embedding row, and one the data produces that it omits "
                           "would raise inside the training loop.")}
        if record_path.is_file():
            previous = paths.read_json(record_path)
            require(list(previous.get("assays") or []) == derived
                    and previous.get("counts") == record["counts"],
                    "The derived condition vocabulary changed between runs: "
                    f"{sorted(set(previous.get('assays') or []) ^ set(derived))}. That changes "
                    "the condition embedding shape and makes every existing checkpoint "
                    "unloadable. Declare a new revision; do not resize a table under a resume.")
            record = previous
        else:
            paths.write_json(record_path, record)
        self._cache["condition_spec"] = ConditionSpec.from_enrolled(
            vocabulary, bin_dropout=float(block["bin_dropout"]),
            assay_dropout=float(block["assay_dropout"]))
        return record

    def assay_index(self):
        return {name: position for position, name in enumerate(self.condition_spec.assay_order)}

    def freeze_hard_split(self, *, persist=True):
        """Sequence-only split of the AlphaBind population against the old library.

        Runs before any outcome is opened and persists the result: a freeze that
        lives only in memory is re-derived on the next process, and a re-derivation
        that disagrees is indistinguishable from the original unless the record is
        on disk to compare against. ``persist=False`` is for a dry run that prints
        the table and writes nothing -- it still compares against an existing
        record, because a dry run that cannot see a disagreement is not a preview.
        """
        from . import her2_data
        from . import her2_routes_sources as sources
        from . import her2_routes_split as split_lib
        manifest, _ = spec.read_acquisition_manifest(self.context)
        target = self.source_paths()["alphabind_her2_validation"]
        read = sources.load_alphaseq_sequences(target)
        scaffold = self.scaffold()
        sites = her2_core_edit_map()
        parent = sources.derive_alphaseq_parent(read["sequences"])
        windows, rejected = [], {}
        for sequence in read["sequences"]:
            ok, reason = sources.alphaseq_compatibility(sequence, parent=parent,
                                                        scaffold=scaffold, sites=sites)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            parts = sources.split_alphaseq_construct(sequence)
            windows.append(parts[0][sites[0][1]:sites[-1][1] + 1])
        require(windows, f"No compatible AlphaBind window survived: {rejected}")
        root = self.context.repository(self.config["roots"]["her2_raw"])
        # Novelty is measured against the FULL old display library. The old TEST
        # split's outcomes stay sealed forever; only its sequences are read, which
        # is what "novel to the parents" has to mean -- a parent selected on the
        # old validation split has seen those windows too.
        library, library_parts = [], {}
        for name in ("train", "val"):
            frame = her2_data.load_split(root, name)
            values = [str(value) for value in frame.seq.tolist()]
            library_parts[name] = len(values)
            library.extend(values)
        # ``test_sequences`` is the display test's one sanctioned reader: a
        # ``usecols=["seq"]`` read that cannot return a bin. The sealed outcomes
        # stay sealed; the sequences are what novelty is defined against.
        test_cores = [str(value) for value in her2_data.test_sequences(root).tolist()]
        library_parts["test_sequences_only"] = len(test_cores)
        library.extend(test_cores)
        require(library, f"No old display library sequence was readable: {library_parts}")
        settings = dict(self.config["split"])
        result = split_lib.build_hard_split(candidates=windows, old_library=library,
                                            settings=settings)
        certificate = split_lib.split_certificate(
            result, rho=float(settings["assumed_rho"]),
            target=float(settings["ambition_target"]))
        record_path = self.context.path(spec.SPLIT_FREEZE)
        existing = paths.read_json(record_path) if record_path.is_file() else None
        freeze = split_lib.freeze_split(
            result, sequence_digest=split_lib.sequence_digest(windows),
            algorithm_digest=split_lib.algorithm_digest(settings),
            acquisition_manifest=manifest, previous=existing)
        freeze["old_library"] = dict(library_parts)
        freeze["old_library_note"] = ("sequences only. The old display TEST split's outcomes are "
                                      "sealed permanently; reading its sequences for a novelty "
                                      "check opens nothing.")
        self.persist_split_freeze(record_path, freeze, existing=existing, persist=persist)
        self._cache["split"] = result
        self._cache["closure"] = split_lib.ExclusionClosure.build(
            reserved_windows=result.reserved,
            development_windows=result.development,
            reserved_identities=tuple(
                self.scaffold_construct(core).identity()
                for core in (*result.reserved, *result.development)),
            radius=int(settings["purge_radius"]),
            extra={"alphabind_her2_full": sources.SOURCES["alphabind_her2_full"].
                   reason_not_enrolled})
        return {"certificate": certificate, "freeze": freeze,
                "parent": parent, "rejected": rejected,
                "compatible_windows": len(windows),
                "unique_windows": len(set(windows)),
                "old_library": dict(library_parts)}

    @staticmethod
    def persist_split_freeze(record_path, freeze, *, existing, persist=True):
        """Write the freeze record the FIRST time only. ``True`` if it wrote.

        A second freeze that agrees with the record on disk comes back stamped ``reconfirmed``
        (``freeze_split``), and rewriting it with that stamp changes the record's digest, which every
        checkpoint identity binds (``split_sha256``). ``split()`` re-derives the split in every FRESH
        process (a resume, or a stage launched separately), so rewriting here silently moved the
        digest of a pinned split and orphaned every checkpoint written under it, with the split's
        content untouched. The stamp survives in the returned record; the file keeps its bytes.
        A record that DISAGREES never reaches this point: ``freeze_split`` refuses it first.
        """
        if not persist or existing is not None:
            return False
        paths.write_json(record_path, freeze)
        return True

    def split(self):
        if "split" not in self._cache:
            self.freeze_hard_split()
        return self._cache["split"]

    def closure(self):
        if "closure" not in self._cache:
            self.freeze_hard_split()
        return self._cache["closure"]

    def scaffold_construct(self, core):
        from .her2_routes_serial import her2_construct
        return her2_construct(self.scaffold(), core)

    def closure_window(self, construct):
        """Apply HER2 radius rules only within the fixed HER2 scaffold's edit space."""
        if "closure_parent" not in self._cache:
            self._cache["closure_parent"] = self.scaffold_construct(self.config["her2"]["wild_type_core"])
        parent = self._cache["closure_parent"]
        sites = her2_core_edit_map()
        if len(construct.heavy) != len(parent.heavy) or construct.light != parent.light:
            return None
        mutable = {index for chain, index in sites if chain == "H"}
        if any(a != b for index, (a, b) in enumerate(zip(construct.heavy, parent.heavy))
               if index not in mutable):
            return None
        return construct.residues(sites)

    # -- the curriculum stream ---------------------------------------------
    def eligible_summaries(self):
        """Measured constructs that clear the exclusion closure, at its radius.

        The closure is consulted once here and again per consumer at draw time.
        The first pass is what makes the stream cheap; the second is what makes
        it correct if a later caller hands the stream a different pool.
        """
        if "eligible" in self._cache:
            return self._cache["eligible"]
        closure = self.closure()
        eligible, refused = [], 0
        for row in self.summaries():
            window = self.closure_window(row.construct)
            if closure.violates(window=window, identity=row.identity) is None:
                eligible.append(row)
            else:
                refused += 1
        require(eligible,
                "Every measured construct was excluded by the closure; there is nothing to "
                "train on")
        self._cache["eligible"] = tuple(eligible)
        self._cache["eligible_refused"] = refused
        return self._cache["eligible"]

    def training_pools(self):
        """The measured training rows, partitioned by declared family. Cached.

        Three pools, because the declared ``family_mix`` allocates across three
        populations and a single pooled list makes the realised mixture whatever the
        row counts happen to be -- and the row counts are dominated by whichever
        source published the most variants. The ordinal display pool is held as
        arrays and materialised per draw, so enrolling all 367,042 rows costs a few
        megabytes rather than a few hundred.
        """
        if "pools" in self._cache:
            return self._cache["pools"]
        closure = self.closure()
        eligible = [row for row in self.eligible_summaries() if row.is_training]
        her2, other = [], []
        for row in eligible:
            (her2 if row.study == "alphabind_her2" or row.source == "alphabind_her2_validation"
             else other).append(row)
        display = self.display_pool()
        pools = {data_lib.FAMILY_HER2: tuple(her2), data_lib.FAMILY_OTHER: tuple(other),
                 data_lib.FAMILY_REPLAY: tuple(self.paired_oas_constructs())}
        self._cache["pools"] = {"summaries": pools, "display": display,
                                "closure_radius": int(closure.radius),
                                "counts": {name: len(value) for name, value in pools.items()},
                                "display_rows": 0 if display is None else len(display),
                                "rule": ("the HER2 measured population, every other measured "
                                         "landscape, and the unlabelled paired repertoire. The "
                                         "declared family mix draws across these three.")}
        return self._cache["pools"]

    def display_pool(self):
        """The sorted-bin display library as an :class:`OrdinalPool`, or ``None``."""
        if "display_pool" in self._cache:
            return self._cache["display_pool"]
        block = dict(self.config["sources"].get("her2_display") or {})
        if not block.get("enrolled", True):
            self._cache["display_pool"] = None
            return None
        root = self.context.repository(self.config["roots"]["her2_raw"])
        record = data_lib.build_display_pool(
            root, scaffold=self.scaffold(), split=str(block.get("split", "train")),
            closure=self.closure(),
            positives_only=bool(block.get("positives_only", False)))
        self._cache["display_record"] = {key: value for key, value in record.items()
                                         if key != "pool"}
        self._cache["display_pool"] = record["pool"]
        return record["pool"]

    def contrast_pool(self, eligible, *, seed, cache_key=None):
        """The bounded contrast enumeration over the training rows, cached on what it was GIVEN.

        The enumeration is the most expensive thing in a stream's construction -- one
        masked-context bucketing pass per varying column of every group -- and the bake-off, the
        control arms, the confirmation seeds and the learning-curve fractions all build a stream,
        so an identical request is answered from the cache.

        "Identical" is decided HERE, from the inputs, and not by the caller's label. A caller key
        used to be the whole test, so a key that records only a family, a seed, an arm and a COUNT
        handed a different same-size subset the first subset's endpoints, and a changed label
        reused the old margins. The pool is materialised once and a versioned digest of the actual
        ordered records (full chains, every site, every replicate's supervision, the metadata a
        contrast reads), the complete enumeration settings and seed, and the parent-sequence map
        is bound into the cache slot. ``cache_key`` stays as the caller's namespace; a mismatching
        digest is a miss, so profile, E8, curriculum and learning-curve callers are all covered.
        ``cache_key=None`` still means "do not cache" (the refresh epochs each use a new seed).
        """
        rows = tuple(eligible)          # one materialisation: a generator is consumed by the digest
        block = self.config["curriculum"]
        parents = self.parent_sequences()
        settings = {
            "max_sites": int(block["max_contrast_sites"]),
            "max_per_background": int(block.get("max_contrasts_per_background", 64)),
            "max_per_context": int(block.get("max_contrasts_per_context", 32)),
            "max_quartets_per_background": int(block.get("max_quartets_per_background", 8)),
            "seed": int(seed), "require_same_assay": True, "training_only": True}
        binding = contrast_input_binding(rows, settings=settings, parents=parents)
        slot = None if cache_key is None else (cache_key, binding["sha256"])
        if slot is not None:
            cached = (self._cache.get("contrast_pools") or {}).get(slot)
            if cached is not None:
                self._cache["contrast_inventory"] = cached["inventory"]
                return cached["contrasts"]
        record = data_lib.build_local_contrasts(
            rows, max_sites=settings["max_sites"],
            max_per_background=settings["max_per_background"],
            max_per_context=settings["max_per_context"],
            max_quartets_per_background=settings["max_quartets_per_background"],
            parents=parents, seed=settings["seed"],
            require_same_assay=settings["require_same_assay"],
            training_only=settings["training_only"])
        inventory = {key: value for key, value in record.items() if key != "contrasts"}
        inventory["input_binding"] = binding
        self._cache["contrast_inventory"] = inventory
        if slot is not None:
            self._cache.setdefault("contrast_pools", {})[slot] = {
                "contrasts": record["contrasts"], "inventory": inventory}
        return record["contrasts"]

    def parent_sequences(self):
        """``{parent name: (heavy, light)}`` for every background a loader resolved.

        Used by the contrast builder to decide which residue an edit is measured
        *against*. Without it the reference is the group's per-column mode, which is
        a good default and is not the declared background of a scan whose parent is
        published.
        """
        return dict(self._cache.get("parent_sequences") or {})

    def curriculum_stream(self, route, *, seed, arm="full", permutation=None, updates=None,
                          microbatch_override=None):
        """``stream(step, micro, rng) -> TaskDraw`` over the four tasks and three families.

        Every window that reaches a view passes the exclusion closure first, for
        the consumer the task corresponds to, **at the declared radius**. A
        reserved core and everything within two edits of one therefore cannot
        enter as a reconstruction target, a contrast endpoint, a whole-window
        target or a head row.

        The draw carries *both* the head batch and the masked views of the same
        constructs. That is what makes the matched-exposure comparison possible: a
        control arm without a head objective trains reconstruction on the identical
        rows, so the row presentations agree exactly and the arms differ only in
        which objective those rows fed.

        The family assignment is per **microbatch**, not per update: with sixteen
        accumulation steps an update-level assignment would give one family the
        whole effective batch, and the realised mixture would be the declared one at
        sixteen times the variance.
        """
        capabilities.require_token_likelihood(route, "FlightServices.curriculum_stream")
        block = self.config["curriculum"]
        mixture = data_lib.TaskMixture(shares=dict(block["shares"]),
                                      family_mix=dict(block["family_mix"]))
        total = int(updates or self.config["routes"][route]["updates"])
        shapes = self.selected_shapes()
        accumulation = int(shapes["accumulation"])
        # An explicit microbatch (the profiler's candidates) is judged against the same recipe as
        # the selected one: positive integer, and able to carry the declared local recipe. A
        # requested 1 used to emit a pair (2 rows) and a requested 3 a pair (2 rows), silently.
        microbatch = int(shapes["microbatch"]) if microbatch_override is None else \
            data_lib.positive_integer(microbatch_override, "microbatch_override")
        data_lib.validate_microbatch(microbatch, curriculum=block, label="the stream microbatch")
        schedule = mixture.joint_schedule(total, seed=int(seed), microbatches=accumulation)
        closure = self.closure()
        pools = self.training_pools()
        bins = dict(self._cache.get("bins") or {})
        by_family = {name: list(value) for name, value in pools["summaries"].items()}
        if permutation is not None:
            for name in (data_lib.FAMILY_HER2, data_lib.FAMILY_OTHER):
                by_family[name] = list(permutation(by_family[name]))
                for row in by_family[name]:
                    donor = row.raw.get("outcome_donor_key")
                    if donor is not None:
                        bins.pop(row.population_key, None)
                        if tuple(donor) in self._cache.get("bins", {}):
                            bins[row.population_key] = self._cache["bins"][tuple(donor)]
        display = pools["display"]
        if display is not None and arm == controls.ARM_OUTCOME_PERMUTED:
            from dataclasses import replace
            classes = np.asarray(display.classes).copy()
            distance = np.asarray([sum(a != b for a, b in zip(core, self.config["her2"]["wild_type_core"]))
                                   for core in display.cores])
            permutation_rng = np.random.default_rng([int(seed), 97])
            for value in np.unique(distance):
                rows = np.flatnonzero(distance == value)
                classes[rows] = classes[rows][permutation_rng.permutation(len(rows))]
            display = replace(display, classes=classes)
        whole_pools = {name: tuple(row for row in rows if row.has_edit_map)
                       for name, rows in by_family.items() if name != data_lib.FAMILY_REPLAY}
        head_pools, enrichment_values = {}, {}
        for name, rows in by_family.items():
            if name == data_lib.FAMILY_REPLAY:
                continue
            groups = {}
            for row in rows:
                groups.setdefault(row.assay, []).append(row)
            head_pools[name] = {}
            for assay, group in groups.items():
                if group[0].kind == "enrichment":
                    # The ranking pairs are chosen by SCALAR order, so only rows with an exact
                    # value can supply one: ``row.value`` is the uncensored-only mean of a mixed
                    # row, a number the assay never produced. The rows stay in the head pool
                    # below -- their per-replicate intervals still train the likelihood.
                    by_value = exact_value_groups(group)
                    if len(by_value) < 2:
                        continue
                    enrichment_values[assay] = tuple(tuple(v) for v in by_value.values())
                head_pools[name][assay] = tuple(group)
        contrasts_by_family = {}
        for name in (data_lib.FAMILY_HER2, data_lib.FAMILY_OTHER):
            # This key is only the caller's namespace (family, seed, arm, permuted?, count).
            # What makes a cache hit CORRECT is the digest of the pool's actual contents that
            # ``contrast_pool`` binds itself: a same-count different subset, a changed label or a
            # permuted outcome is a miss whatever this key says.
            contrasts_by_family[name] = self.contrast_pool(
                by_family[name], seed=int(seed),
                cache_key=(name, int(seed), str(arm), permutation is not None,
                           len(by_family[name]))) if by_family[name] else ()
        def permute_contrast_targets(pools, epoch):
            if arm != controls.ARM_BACKGROUND_PERMUTED:
                return pools
            changed, audits = {}, {}
            for name, pool in pools.items():
                changed[name], audits[name] = controls.permute_background_effects(
                    pool, seed=int(seed) + int(epoch))
            self._cache["background_permutation_audit"] = audits
            return changed

        contrasts_by_family = permute_contrast_targets(contrasts_by_family, 0)
        contrast_epoch = 0
        quartet_pools = {name: tuple(c for c in pool if c.kind == data_lib.CONTRAST_COUPLED)
                         for name, pool in contrasts_by_family.items()}
        pair_pools = {name: tuple(c for c in pool if c.kind != data_lib.CONTRAST_COUPLED)
                     for name, pool in contrasts_by_family.items()}
        assay_index = self.assay_index()
        tile = int(block.get("max_query_sites", 0)) or None
        self._cache["stream_inventory"] = {
            "family_counts": {name: len(value) for name, value in by_family.items()},
            "contrast_counts": {name: len(value) for name, value in contrasts_by_family.items()},
            "display_rows": 0 if display is None else len(display),
            "declared_family_mix": dict(block["family_mix"]),
            "declared_task_shares": dict(block["shares"])}

        def _draw(pool, rng, count):
            return [pool[int(index)] for index in rng.integers(0, len(pool),
                                                               min(count, len(pool)))]

        def _measured_family(name, rng):
            """The family to draw a measured task from, falling back with a record."""
            if name == data_lib.FAMILY_REPLAY:
                name = data_lib.FAMILY_HER2
            if by_family.get(name) or (name == data_lib.FAMILY_HER2 and display is not None):
                return name
            other = data_lib.FAMILY_OTHER if name == data_lib.FAMILY_HER2 \
                else data_lib.FAMILY_HER2
            return other if by_family.get(other) else name

        def _head_draw(family, rng):
            """Balance HER2 likelihood kinds and draw within a uniformly chosen assay."""
            quantitative = head_pools.get(family) or {}
            if family == data_lib.FAMILY_HER2 and display is not None:
                weight = float(block.get("her2_head_display_probability", 0.5))
                if not quantitative or float(rng.random()) < weight:
                    indices = rng.integers(0, len(display), microbatch)
                    return display.materialise(indices, region=her2_core_edit_map())
            if not quantitative:
                return ()
            assays = tuple(quantitative)
            assay = assays[int(rng.integers(len(assays)))]
            if assay in enrichment_values:
                values = enrichment_values[assay]
                first, second = rng.choice(len(values), size=2, replace=False)
                pair = _draw(values[int(first)], rng, 1) + _draw(values[int(second)], rng, 1)
                return tuple(pair + _draw(quantitative[assay], rng, max(0, microbatch - 2)))
            return tuple(_draw(quantitative[assay], rng, microbatch))

        def stream(step, micro, rng, *, task_override=None, family_override=None):
            nonlocal contrast_epoch, contrasts_by_family, quartet_pools, pair_pools
            view_rng = np.random.default_rng([int(seed), int(step), int(micro), 43])
            position = ((int(step) - 1) * accumulation + int(micro)) % len(schedule)
            task, family = schedule[position]
            task = task_override or task
            family = family_override or family
            if task == data_lib.TASK_RECONSTRUCTION:
                pool = by_family[family]
                if not pool:
                    return None
                chosen = _draw(pool, rng, microbatch)
                if family != data_lib.FAMILY_REPLAY:
                    chosen = [row.construct for row in chosen]
                for construct in chosen:
                    closure.require_allowed(consumer="reconstruction_target",
                                            window=self.closure_window(construct),
                                            identity=construct.identity())
                views = data_lib.reconstruction_views(
                    chosen, rng=rng, mask_fraction=float(block["mask_fraction"]),
                    condition_spec=self.condition_spec, family="display",
                    route=self._kind(route), max_positions=self._positions(route))
                return tasks_lib.TaskDraw(task=data_lib.TASK_RECONSTRUCTION,
                                          family=family, views=views,
                                          source=chosen[0].source)
            family = _measured_family(family, rng)
            if task == data_lib.TASK_LOCAL:
                epoch = (int(step) - 1) // max(1, int(block.get("contrast_refresh_updates", 2500)))
                if epoch != contrast_epoch:
                    contrasts_by_family = {name: self.contrast_pool(
                        by_family[name], seed=int(seed) + epoch, cache_key=None)
                        for name in (data_lib.FAMILY_HER2, data_lib.FAMILY_OTHER)}
                    contrasts_by_family = permute_contrast_targets(contrasts_by_family, epoch)
                    quartet_pools = {name: tuple(c for c in pool if c.kind == data_lib.CONTRAST_COUPLED)
                                     for name, pool in contrasts_by_family.items()}
                    pair_pools = {name: tuple(c for c in pool if c.kind != data_lib.CONTRAST_COUPLED)
                                 for name, pool in contrasts_by_family.items()}
                    contrast_epoch = epoch
                pool = contrasts_by_family.get(family) or ()
                if not pool:
                    return None
                quartets, pairs = quartet_pools.get(family) or (), pair_pools.get(family) or ()
                # ``microbatch`` was validated against the declared recipe above, so this loop
                # tiles it exactly: pairs are two rows, quartets four, and the size is even.
                picked, remaining = [], microbatch
                while remaining >= 2:
                    use_quartet = bool(quartets and remaining >= 4 and
                        (not pairs or rng.random() < float(block.get("local_quartet_probability", 0.25))))
                    candidates = quartets if use_quartet else pairs
                    if not candidates:
                        break
                    chosen = candidates[int(rng.integers(len(candidates)))]
                    picked.append(chosen)
                    remaining -= len(chosen.alternatives())
                require(picked, "A local task has no pair or fitting quartet")
                for contrast in picked:
                    for row in contrast.alternatives():
                        closure.require_allowed(
                            consumer="local_contrast_endpoint",
                            window=self.closure_window(row.construct),
                            identity=row.construct.identity())
                views = self.contrast_views(picked, route=route)
                # The emitted local view count IS the validated microbatch. A pool that cannot
                # tile it (no pair left for a two-row remainder, say) must say so: a shorter
                # batch would be presented as the declared shape and every arm's matched rows
                # would silently differ from the effective batch.
                require(len(views) == microbatch,
                        f"The local stream emitted {len(views)} endpoint row(s) for a validated "
                        f"microbatch of {microbatch}: the {family} contrast pool "
                        f"({len(quartets)} quartet(s), {len(pairs)} pair(s)) cannot tile it.")
                return tasks_lib.TaskDraw(task=task, family=family, views=views,
                                          contrasts=tuple(picked),
                                          summaries=tuple(row for contrast in picked
                                                          for row in contrast.alternatives()),
                                          source=picked[0].left.source)
            if task == data_lib.TASK_HEADS:
                chosen = _head_draw(family, rng)
                if not chosen:
                    return None
                for row in chosen:
                    closure.require_allowed(
                        consumer="observation_head_row",
                        window=self.closure_window(row.construct),
                        identity=row.identity)
                head_views = data_lib.head_views(
                    chosen, condition_spec=self.condition_spec, route=self._kind(route),
                    max_positions=self._positions(route), rng=view_rng)
                head_batch = data_lib.collate_heads(
                    head_views, pad_id=self.pad_id(route), condition_spec=self.condition_spec,
                    assay_index=assay_index)
                # The same constructs as masked views, so an arm without the head
                # objective presents the identical rows under reconstruction.
                views = data_lib.whole_window_views(
                    [row for row in chosen if row.has_edit_map],
                    condition_spec=self.condition_spec, bins=bins, route=self._kind(route),
                    max_positions=self._positions(route), answer_free=True, tile=tile,
                    rng=view_rng, sample_tile=True)
                missing_region = [row.construct for row in chosen if not row.has_edit_map]
                if missing_region:
                    views += data_lib.reconstruction_views(
                        missing_region, rng=view_rng, mask_fraction=float(block["mask_fraction"]),
                        condition_spec=self.condition_spec, family=chosen[0].family,
                        route=self._kind(route), max_positions=self._positions(route))
                require(len(views) == len(chosen), "Head controls must present the same number of constructs")
                return tasks_lib.TaskDraw(task=task, family=family, views=views,
                                          head_batch=head_batch, summaries=tuple(chosen),
                                          source=chosen[0].source)
            pool = whole_pools.get(family) or ()
            if not pool:
                return None
            chosen = _draw(pool, rng, microbatch)
            for row in chosen:
                closure.require_allowed(
                    consumer="whole_window_target",
                    window=self.closure_window(row.construct),
                    identity=row.identity)
            views = data_lib.whole_window_views(
                chosen, condition_spec=self.condition_spec, bins=bins, route=self._kind(route),
                max_positions=self._positions(route), tile=tile, rng=view_rng, sample_tile=True,
                progressive_probability=float(block.get("masked_progressive_probability", 0.5)))
            if not views:
                return None
            return tasks_lib.TaskDraw(task=task, family=family, views=views,
                                      summaries=tuple(chosen), source=chosen[0].source)
        return stream

    def subset_stream(self, route, subset, *, seed, updates):
        """A curriculum stream restricted to a declared subset of measured rows.

        E13's input. The restriction is applied to the measured pools only: the
        unlabelled replay pool is the same at every data fraction, because the
        question is how much *measured* data the recipe needs.

        The unit selected is the **population key** -- full heavy|light identity, canonical study,
        canonical assay and target -- which is exactly what ``summarise_observations`` groups into
        one :class:`ConstructSummary`. Selecting by ``row.identity`` alone admitted every other
        assay and target measured on the same full construct, so a chosen assay brought its
        neighbours along. Every replicate of a chosen summary travels with it (the summary holds
        them all), and a chosen key the training pools do not hold is refused rather than trained
        on a smaller set than was declared.
        """
        keys = acquired_population_keys(subset)
        pools = self.training_pools()["summaries"]
        available = {row.population_key
                     for name in (data_lib.FAMILY_HER2, data_lib.FAMILY_OTHER)
                     for row in pools[name]}
        missing = sorted(keys - available)
        require(not missing,
                f"{len(missing)} selected population key(s) are not in the measured training "
                f"pools, e.g. {[key[1:] for key in missing[:2]]}; a subset stream would train on "
                "fewer constructs than the subset declares")

        def restrict(rows):
            return [row for row in rows if row.population_key in keys]
        return self.curriculum_stream(route, seed=int(seed), updates=int(updates),
                                      permutation=restrict)

    def withheld_marker(self, row):
        """``parent:<name>`` / ``study:<name>`` when this row was declared withheld."""
        if row.is_training:
            return None
        heldouts = dict(self.config["sources"].get("withheld") or {})
        if row.parent in set(heldouts.get("parents") or ()):
            return f"parent:{row.parent}"
        if row.study in set(heldouts.get("studies") or ()) or \
                row.source in set(heldouts.get("studies") or ()):
            return f"study:{row.study}"
        return None

    def summary_scores(self, route, model, rows, *, batch_size=16):
        """Mean window log probability of each measured construct, over its own region.

        Used by the transfer panel, where the rows are not HER2 cores and cannot be
        scored through the fixed scaffold. A row without a region is skipped by the
        caller; here an empty region would produce a view with no query.
        """
        from .her2_routes_conditions import BIN_HIGH, Condition
        from .her2_routes_serial import ablang_serialization, causal_serialization
        views = []
        for row in rows:
            require(row.has_edit_map,
                    f"{row.identity[:16]} has no declared region, so it has no window to score. "
                    "Rows without a shared support train the heads and are not scored here.")
            condition = Condition(family=row.family, bin=BIN_HIGH,
                                  assay=row.assay if row.assay in self.condition_spec.assays else "null")
            construct = row.construct
            serialization = (
                causal_serialization(construct, order=construct.sites,
                                     residues=construct.residues(construct.sites),
                                     max_positions=self._positions(route))
                if route == ROUTE_CAUSAL
                else ablang_serialization(construct, order=construct.sites))
            views.append((construct, serialization, condition))
        scores = []
        with torch.no_grad():
            for start in range(0, len(views), int(batch_size)):
                chunk = views[start:start + int(batch_size)]
                batch = self.collate(route, chunk)
                totals = self.window_log_probs(route, model, batch).sum(dim=1)
                for view, value in zip(chunk, totals):
                    scores.append(float(value) / max(1, len(view[1].order)))
        return scores

    def reversal_pairs(self, route, checkpoint):
        """E5's input: oriented substitution pairs on the development panel, scored.

        Both endpoints carry their measured outcome and their distance profile, which
        is what makes the exact grouping possible. The panel is small and the counts
        are reported, because the audited HER2 panel had 0-3 repeated development
        groups and that is what support-gates this panel rather than a relaxation.
        """
        rows = self.development_panel_outcomes()
        outcomes = {row["window"]: float(row["value"]) for row in rows}
        windows = [row["window"] for row in rows]
        scores = dict(zip(windows, self.score_panel(route, checkpoint, rows)))
        wild = self.config["her2"]["wild_type_core"]
        # Distance to the nearest ELIGIBLE ADAPTATION sequence, exact and chunked. The earlier
        # profile measured it against the other development windows -- the held-out panel, which
        # no adaptation consumer ever sees -- and a radius query's "-1: nothing within the radius"
        # is not a distance, so neither is used here.
        nearest = dict(zip(windows, controls.nearest_hamming(windows, self.adaptation_bank())))
        profiles = {window: {"to_wild_type": controls.hamming(window, wild),
                             "to_nearest_training": nearest[window]} for window in windows}
        pairs = []
        for left in range(len(windows)):
            for right in range(left + 1, len(windows)):
                a, b = windows[left], windows[right]
                if controls.oriented_substitution_group(a, b) is None:
                    continue
                pairs.append({
                    "left_window": a, "right_window": b,
                    "left_profile": dict(profiles[a]), "right_profile": dict(profiles[b]),
                    "outcome_delta": outcomes[a] - outcomes[b],
                    "score_delta": scores[a] - scores[b]})
        return pairs

    def _kind(self, route):
        if route == ROUTE_CAUSAL:
            from .her2_policy import load_vocab
            from .her2_routes_serial import bind_piggen_vocabulary, PIGGEN_ID_OF_RESIDUE
            if not PIGGEN_ID_OF_RESIDUE:
                bind_piggen_vocabulary(load_vocab(self.context.repository(self.config["roots"]["her2_raw"])))
        return "causal" if route == ROUTE_CAUSAL else "masked"

    def _positions(self, route):
        return int(self.config["routes"][route].get("max_positions", 400))

    def contrast_views(self, contrasts, *, route):
        """Every alternative of each contrast, masked at the UNION of changed sites.

        A coupled quartet presents four alternatives -- background, A, B and AB --
        against one masked background, not two. Presenting only the two singles
        would leave the interaction, which is the whole reason the quartet is
        enumerated, with nothing to be computed from.
        """
        from .her2_routes_serial import ablang_serialization, causal_serialization
        from .her2_routes_conditions import Condition
        views = []
        for contrast in contrasts:
            for record in contrast.alternatives():
                construct = data_lib.with_sites(record.construct, contrast.sites)
                condition = Condition(family=record.family, bin=BIN_NULL, assay=record.assay)
                serialization = (
                    causal_serialization(construct, order=contrast.sites,
                                         residues=construct.residues(contrast.sites),
                                         max_positions=self._positions(route))
                    if route == ROUTE_CAUSAL
                    else ablang_serialization(construct, masked_sites=contrast.sites,
                                              order=contrast.sites))
                views.append((construct, serialization, condition))
        return tuple(views)

    def contrast_payload(self, contrasts):
        """The numeric side of a contrast batch: margins, interactions, row spans.

        Returned alongside the views so the objective has the measured numbers
        and not only the masked sequences. ``row_span`` says which rows of the
        collated batch belong to which contrast, because a quartet contributes
        four rows and a single pair two.
        """
        import numpy as np
        spans, margins, interactions, kinds, assays = [], [], [], [], []
        cursor = 0
        for contrast in contrasts:
            width = len(contrast.alternatives())
            spans.append((cursor, cursor + width))
            cursor += width
            margins.append(float("nan") if contrast.margin is None else float(contrast.margin))
            interactions.append(float("nan") if contrast.interaction is None
                                else float(contrast.interaction))
            kinds.append(contrast.kind)
            assays.append(contrast.assay)
        return {"row_span": tuple(spans), "rows": cursor,
                "margin": np.asarray(margins, dtype=np.float64),
                "interaction": np.asarray(interactions, dtype=np.float64),
                "kind": tuple(kinds), "assay": tuple(assays),
                "with_margin": int(np.isfinite(np.asarray(margins, dtype=np.float64)).sum()),
                "with_interaction": int(np.isfinite(
                    np.asarray(interactions, dtype=np.float64)).sum())}

    def paired_oas(self):
        """The prepared paired repertoire, reservoir-sampled per split and cached."""
        if "oas_record" in self._cache:
            return self._cache["oas_record"]
        from . import her2_routes_sources as sources
        block = self.config["curriculum"]
        root = self.context.repository(self.config["roots"]["paired_oas"])
        cache = self.source_cache_root()
        cache.mkdir(parents=True, exist_ok=True)
        record = sources.load_paired_oas(
            root, limit=int(block["oas_rows"]), cache=cache,
            splits=tuple(block.get("oas_splits") or ("train", "val")),
            seed=int(self.config["split"]["seed"]))
        self._cache["oas_record"] = record
        return record

    def paired_oas_constructs(self, *, split="train"):
        """Training-split paired constructs only. The other splits stay held out.

        The held-out repertoire split is what an unseen-study retention claim is
        measured on, and it is not a training pool. An earlier reader took the first
        rows of whichever file sorted first, which had neither property.
        """
        record = self.paired_oas()
        by_split = dict(record["by_split"])
        pool = by_split.get(str(split)) or ()
        require(pool,
                f"The paired OAS release yielded no {split!r} row. Its splits are "
                f"{sorted(by_split)}; the row field is {record['member']}'s own 'split'.")
        closure = self.closure()
        return tuple(construct for construct in pool if closure.violates(
            window=self.closure_window(construct), identity=construct.identity()) is None)

    # -- the bake-off ------------------------------------------------------
    def run_bakeoff(self, *, heartbeat=None):
        """Train every enabled route to its declared budget. One record per route."""
        # Under the DPLM training phase the pinned split is verified here too, before the first
        # update of either route, so a direct call cannot skip the stage-level check.
        phase = spec.active_phase(self.config)
        admission_policy = phase if spec.is_admission_phase(phase) else None
        policy = None if admission_policy is not None else phase
        spec.verify_training_phase_split(self.context, phase)
        pins = spec.admission_pins(self.config) if admission_policy is not None else {}
        if admission_policy is not None:
            # A direct call must not skip the stage-level check (see _require_admission_ready).
            admission.require_clean_run_root(self.context.run_root)
        records, remaining = {}, []
        for route in self.config["routes"]["enabled"]:
            block = self.config["routes"][route]
            directory = self.context.path("bakeoff", route)
            if admission_policy is not None:
                if route in admission_policy["admitted_routes"]:
                    # Imported, not trained: the pinned best.pt of a finished flight.
                    records[route] = self.admit_route(route, pins[route])
                    continue
                self.require_not_admitted(directory)
            if route == ROUTE_ENERGY:
                # The energy route's budget is two named phases, so its terminality
                # is the sum of both rather than a single ``updates`` figure.
                # The two arms are the SAME experiment except for the conditioning: one declared
                # population, seed, family/group/member ordering, tile rule, microbatch and phase
                # budgets (the energy contract binds and records all of it), with the context given
                # to one and withheld (context=None) from the other.
                shared = {"population": energy_contract.POPULATION_ALL_ELIGIBLE,
                          "seed": int(block["seed"]), "heartbeat": heartbeat}
                records[route] = self.train_energy(directory=directory, conditioned=True, **shared)
                baseline = self.train_energy(directory=directory / "density_only",
                                             conditioned=False, **shared)
                matched = energy_contract.require_matched_controls(records[route], baseline)
                records[route]["density_only_baseline"] = {
                    "checkpoint": str(directory / "density_only" / "best.pt"),
                    "arm": energy_contract.ARM_NULL,
                    "phase_updates": baseline["phase_updates"],
                    "exposure": baseline["exposure"], "identity": baseline["identity"],
                    "energy_contract_sha256": baseline["energy_contract_sha256"],
                    "exposure_trace": baseline["exposure_trace"],
                    "matched_to_full": matched}
                expected = int(block["denoiser_updates"]) + int(block["energy_updates"])
                if records[route]["updates"] < expected:
                    remaining.append(route)
                continue
            records[route] = self.train(
                route, stream=self.curriculum_stream(route, seed=int(block["seed"]),
                                                     updates=int(block["updates"])),
                updates=int(block["updates"]), seed=int(block["seed"]), directory=directory,
                heartbeat=heartbeat, milestones=tuple(block.get("milestones") or ()),
                evaluate=self.selection_metric(route))
            if records[route]["updates"] < int(block["updates"]):
                remaining.append(route)
        result = {"routes": records, "completed": sorted(set(records) - set(remaining)),
                  "remaining": remaining, "all_terminal": not remaining}
        if policy is not None:
            result["training_phase"] = {"phase_id": policy["phase_id"],
                                        "policy_sha256": policy["policy_sha256"],
                                        "admission": policy["admission"],
                                        "selection_metric": policy["selection_metric"]}
        if admission_policy is not None:
            result["admission_phase"] = {"phase_id": admission_policy["phase_id"],
                                         "policy_sha256": admission_policy["policy_sha256"],
                                         "pins_sha256": admission_policy["pins_sha256"],
                                         "admission": admission_policy["admission"]}
        return result

    def energy_context(self, summaries, *, answer_free=False, rng=None, requested_bin=None):
        """Build the conditional context for a batch of energy-route rows.

        The energy route is conditional or it is the density-only control; there is
        no third thing. An earlier revision never passed a context at all, so the
        context tower's parameters were allocated, reported in the memory table and
        never reached by a gradient -- and the route's conditional claim rested on
        code that did not run.

        Only the conditioned arm calls this. The null arm passes ``context=None`` in training,
        scoring and sampling and never builds one; the runtime contract bound to a loaded model
        (``energy_contract.EnergyRuntime``) is what says which arm a model is.
        """
        from ..models.her2_routes_energy import build_energy_context
        from .her2_routes_conditions import Condition
        bins = dict(self._cache.get("bins") or {})
        conditions = []
        for row in summaries:
            name = requested_bin if requested_bin is not None else (
                BIN_NULL if answer_free else bins.get(row.population_key, BIN_NULL))
            condition = Condition(family=row.family, bin=name, assay=row.assay)
            if rng is not None:
                condition = self.condition_spec.apply_dropout(condition, rng)
            conditions.append(list(self.condition_spec.encode(condition)))
        return build_energy_context([row.construct for row in summaries],
                                    condition_indices=conditions).to(self.device)

    def energy_generator(self, seed):
        """A generator on the device the tensors live on. A CPU one cannot drive CUDA.

        ``torch.randn(..., device="cuda", generator=<cpu generator>)`` raises, and an
        earlier revision passed a CPU generator into a CUDA noise draw -- which meant
        the energy route had never run on the device it was profiled for.
        """
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def energy_rows(self, *, her2_only=False):
        """Measured training constructs with declared regions, across all enrolled lengths."""
        rows = [record for record in self.eligible_summaries()
                if record.is_training and record.has_edit_map
                and (not her2_only or record.source == energy_contract.HER2_SOURCE)]
        require(rows,
                "No eligible measured training region for the energy route")
        return tuple(rows)

    def energy_run_shape(self, *, updates=None):
        """The microbatch and phase budgets a run will ACTUALLY use, resolved once.

        The microbatch is the profile's chosen one for this route when ``gpu_profile.json`` has
        it and the declared one otherwise; the budgets are the declared phase updates unless an
        explicit ``updates`` replaces both (the smoke). All three are validated positive integers.
        Only ``None`` (or an absent profile entry or key) means "not specified": a supplied ``0``,
        ``False`` or empty string is an invalid size and is refused, never read as absent -- it
        would otherwise start the full declared budgets or fall back to the declared microbatch
        under a request that named something else.
        This is called before an identity is built or a checkpoint is read, because the numbers it
        returns go into the energy contract: a ``gpu_profile.json`` that changed between an
        interruption and its resume must change the contract and refuse the resume, not silently
        change the experiment. ``source`` and ``profile`` are provenance only and are not hashed.
        """
        block = self.config["routes"][ROUTE_ENERGY]
        profile_path = self.context.path(spec.PROFILE_JSON)
        entry, profile_sha256 = {}, None
        if profile_path.is_file():
            document = paths.read_json(profile_path)
            require(isinstance(document, dict),
                    f"{profile_path} is not a mapping of route profiles")
            entry = document.get(ROUTE_ENERGY)
            entry = {} if entry is None else entry
            require(isinstance(entry, dict),
                    f"{profile_path}: the {ROUTE_ENERGY} profile is not a mapping")
            profile_sha256 = paths.sha256_file(profile_path)
        chosen = entry.get("chosen_microbatch")
        if chosen is not None:
            microbatch = data_lib.positive_integer(chosen, "the energy profile's chosen_microbatch")
            source = "profile"
        else:
            microbatch = data_lib.positive_integer(block["microbatch"],
                                                   f"routes.{ROUTE_ENERGY}.microbatch")
            source = "declared"
        override = None if updates is None else data_lib.positive_integer(updates, "updates")
        budget = {phase: data_lib.positive_integer(
                      override if override is not None else block[key],
                      f"routes.{ROUTE_ENERGY}.{key}")
                  for phase, key in ((energy_contract.PHASE_DENOISER, "denoiser_updates"),
                                     (energy_contract.PHASE_ENERGY, "energy_updates"))}
        return {"microbatch": microbatch, "microbatch_source": source, "budget": budget,
                "budget_source": ("the explicit updates argument, for both phases"
                                  if override is not None else "the declared phase updates"),
                "profile": {"path": str(profile_path), "present": profile_path.is_file(),
                            "sha256": profile_sha256}}

    def energy_training_plan(self, *, seed, population, conditioned, updates=None):
        """Everything a run is bound to, resolved BEFORE any model, directory or state exists.

        Returns the version-2 energy contract, the checkpoint identity whose ``schedule_sha256``
        is that contract's hash, the canonical cohort the sampler's random indices address, and the
        resolved shape. ``population`` and ``conditioned`` are separate and both required: the
        population says which rows, ``conditioned`` says whether the model is given a context,
        and neither implies the other. The contract's ``common_sha256`` excludes only the
        conditioning, so a full and a null run over the same population share it.
        """
        from ..models.her2_routes_energy import DEFAULT_SAMPLING_OPTIONS
        seed = int(seed)
        arm = energy_contract.arm_for(conditioned)
        energy_contract.require_population(population)
        block = self.config["routes"][ROUTE_ENERGY]
        shape = self.energy_run_shape(updates=updates)
        rows = self.energy_rows(her2_only=population == energy_contract.POPULATION_HER2_ONLY)
        groups = energy_contract.canonical_groups(rows)
        cohort = energy_contract.cohort_document(groups)
        walk_steps = data_lib.positive_integer(block["walk_steps"],
                                               f"routes.{ROUTE_ENERGY}.walk_steps")
        max_query_sites = int(self.config["curriculum"].get("max_query_sites", 48))
        optimization = self.config["optimization"]
        assignment = None
        if conditioned:
            assignment = energy_contract.condition_assignment_document(
                groups, dict(self._cache.get("bins") or {}), default_bin=BIN_NULL,
                spec_document=self.condition_spec.document())
        device_type = str(self.device).split(":")[0]
        sampling = {
            "seed": seed,
            "sampler_generator": ("numpy default_rng([seed, 2]): the family, group, member and "
                                  "tile draws, and nothing else"),
            "dropout_generator": ("numpy default_rng([seed, 1]) (mask_rng): condition dropout "
                                  "only. Conditioning may consume it differently, so it is never "
                                  "the sampler's generator"),
            "noise_generators": {"device": device_type, "denoiser_seed": seed,
                                 "energy_seed": seed + 1},
            "family_law": energy_contract.family_law(sorted(groups)),
            "draw_order": "family (when both exist), group, members, tile -- all from the sampler",
            "group_rule": "uniform over the chosen family's canonical groups",
            "member_rule": ("microbatch members drawn uniformly WITH replacement within the "
                            "group, as indices into its canonical member order"),
            "tile_rule": {"max_query_sites": max_query_sites,
                          "rule": ("the group's region is cut into consecutive tiles of at most "
                                   "max_query_sites sites and one tile is drawn uniformly per "
                                   "step; every member is presented at that tile's sites")},
            "microbatch": shape["microbatch"]}
        model_block = {key: block[key] for key in ("window", "sigma", "scale", "reg_l2_norm",
                                                    "context_width")}
        initialization = {
            "seed": seed,
            "rule": "build_route(seed) seeds torch before the model is constructed",
            "architecture": {"window": int(model_block["window"]),
                             "sigma": float(model_block["sigma"]),
                             "scale": float(model_block["scale"]),
                             "reg_l2_norm": float(model_block["reg_l2_norm"]),
                             "context_width": int(model_block["context_width"]),
                             "condition_sizes": {name: int(size) for name, size in
                                                 self.condition_spec.sizes.items()}},
            "condition_tables_built": True,
            "same_for_both_arms": ("the null arm builds the same condition tables and context "
                                   "towers from the same seed; it is never given a context")}
        phases = {
            "order": list(energy_contract.PHASES), "budget": dict(shape["budget"]),
            "walk": {**DEFAULT_SAMPLING_OPTIONS, "steps": walk_steps},
            "optimizer": {"name": "AdamW", "learning_rate": float(optimization["learning_rate"]),
                          "denoiser_phase": ("parameters named denoiser*: the denoiser and its "
                                             "own context tower"),
                          "energy_phase": ("every remaining parameter that requires a gradient, "
                                           "with the denoiser and its tower frozen")},
            "gradient_clip": float(optimization["gradient_clip"])}
        contract = energy_contract.build_contract(
            arm=arm, population=population, cohort=cohort, sampling=sampling,
            initialization=initialization, phases=phases,
            forward_ledger=energy_contract.forward_ledger_document(walk_steps),
            condition_assignment=assignment)
        identity = self.training_identity(
            ROUTE_ENERGY, arm=arm, seed=seed, schedule_sha256=contract["energy_contract_sha256"])
        return {"arm": arm, "contract": contract, "identity": identity, "groups": groups,
                "shape": shape, "budget": dict(shape["budget"]), "walk_steps": walk_steps,
                "max_query_sites": max_query_sites, "rows": cohort["rows"],
                "microbatch": shape["microbatch"]}

    @staticmethod
    def _require_no_legacy_energy_artifacts(directory, names):
        """Never overwrite a checkpoint written before energy_contract version 2.

        A fresh start into a directory that still holds one would replace the only copy of a
        result that cannot be reproduced under the new contract. The artifact is left untouched.
        """
        for name in names:
            target = Path(directory) / name
            if target.is_file():
                payload = torch.load(str(target), map_location="cpu", weights_only=False)
                if not (isinstance(payload, dict) and payload.get("energy_contract") is not None):
                    raise energy_contract.EnergyCheckpointIncompatible(
                        f"{target} {energy_contract.LEGACY_REFUSAL}")

    def train_energy(self, *, directory, seed, population, conditioned, heartbeat=None,
                     resume=True, updates=None, stop_after_updates=None):
        """Denoiser first, then the frozen-denoiser energy objective. Two phases, resumable.

        ``population`` (:data:`~.her2_routes_energy_contract.POPULATIONS`) and ``conditioned``
        are separate, explicit and required. The earlier signature coupled them -- the conditioned
        arm took every eligible row and the density-only arm only the HER2 rows -- so the two
        arms differed in data as well as in context. A full and a null run with the same
        ``population``, ``seed`` and ``updates`` now share the cohort, the initialization, the
        family/group/member ordering, the tile rule, the microbatch and the phase budgets; the
        null run passes ``context=None`` and never invokes the context builder.

        What this adds over the earlier revision, item by item, because each one was
        a defect rather than a refinement:

        * the sampler draws from the **training** rows only. It drew from every
          eligible row, so the density included the development panel it was later
          evaluated on;
        * the noise generator is on the **model's device**. A CPU generator cannot
          drive a CUDA ``randn``, so the conditioned path had never run on the GPU;
        * the phases **resume**. It restarted from scratch on every call and saved
          only the model at the very end, so an interrupted fit lost everything and
          a repeated call silently retrained;
        * the checkpoint carries the optimizer, both phases' step counts, the RNG
          bundle and the exposure ledger, and it is written on the declared cadence;
        * the checkpoint is bound to an ``energy_contract`` (version 2): the population and
          its cohort, the conditioning, the draw law, the actual seed and microbatch, the phase
          budgets and the condition assignment, hashed into ``identity.schedule_sha256``. A resume
          recomputes it and compares it BEFORE a directory, a model, a generator or an optimizer
          exists; a checkpoint with none (written before version 2), an unrecognised phase or a
          changed contract is refused, and the artifact is left untouched;
        * every step is appended to a cumulative :class:`~.her2_routes_energy_contract.
          ExposureTrace`, saved with the checkpoint and continued on resume. A resume also checks
          that trace's arithmetic (rows, site exposures) and the checkpoint's exposure ledger
          against the contract, and the checkpoint's schema, required keys and RNG inventory, all
          before anything is built.

        ``updates`` and ``stop_after_updates`` are positive integers when supplied; only ``None``
        means "not specified". A supplied ``0`` is refused, not read as the full budgets or as a
        run that stops after one step.
        """
        from dataclasses import replace
        from ..models.her2_routes_energy import onehot_window
        from .her2_routes_serial import canonical_index
        directory = Path(directory)
        self.require_not_admitted(directory)
        stop = (None if stop_after_updates is None
                else data_lib.positive_integer(stop_after_updates, "stop_after_updates"))
        # 1. Resolve and bind everything the run depends on. No model, directory or state yet.
        plan = self.energy_training_plan(seed=seed, population=population,
                                         conditioned=conditioned, updates=updates)
        contract, identity = plan["contract"], plan["identity"]
        budget, microbatch = plan["budget"], plan["microbatch"]
        groups, present = plan["groups"], tuple(sorted(plan["groups"]))
        walk_steps = plan["walk_steps"]
        checkpoint = directory / "last.pt"
        # 2. Validate the checkpoint on CPU metadata, still before anything is mutated.
        resume_payload, position = None, None
        if resume and checkpoint.is_file():
            resume_payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            position = energy_contract.require_resumable(resume_payload, contract=contract,
                                                         path=checkpoint)
            checkpoints.require_identity(resume_payload.get("identity"), identity,
                                         path=checkpoint)
            # The schema, the required keys and the RNG bundle's inventory are read here, from the
            # payload alone, so a checkpoint ``restore`` would refuse is refused before the model
            # is built and the global generators are seeded.
            checkpoints.require_restorable_metadata(resume_payload, path=checkpoint,
                                                    generators=ENERGY_RNG_INVENTORY)
            self._require_no_legacy_energy_artifacts(directory, ("best.pt",))
        else:
            self._require_no_legacy_energy_artifacts(directory, ("last.pt", "best.pt"))
        # 3. Only now: the directory, the model, the state and the restore.
        directory.mkdir(parents=True, exist_ok=True)
        model = self.build_route(ROUTE_ENERGY, seed=seed)
        state = self.fresh_state(seed)
        state.energy_rng = self.energy_generator(int(seed))
        trace = energy_contract.ExposureTrace()
        resume_record = None
        if resume_payload is not None:
            state, resume_record = checkpoints.restore(
                checkpoint, model=model, heads=None, optimizer=None, scheduler=None,
                state=state, identity=identity)
            trace = position["trace"]

        def extras(phase_name, **more):
            return {"energy_phase": phase_name, "phase_budget": dict(budget),
                    "energy_contract": contract, "energy_trace": trace.state(), **more}
        cadence = self.cadence()
        phases, done = {}, {"denoiser": 0, "energy": 0}
        if state.energy_phase == "energy":
            done["denoiser"] = budget["denoiser"]
            done["energy"] = int(state.step)
        elif state.energy_phase == "denoiser":
            done["denoiser"] = int(state.step)
        for phase in ("denoiser", "energy"):
            if phase == "denoiser" and state.energy_phase == "energy":
                phases[phase] = {"updates": done[phase], "resumed_completed_phase": True}
                continue
            continuing_phase = state.energy_phase == phase
            parameters = [p for name, p in model.named_parameters()
                          if name.startswith("denoiser") == (phase == "denoiser")]
            if phase == "energy":
                # The denoiser AND its own context tower are frozen here. A shared
                # tower would leave the frozen denoiser's input drifting under the
                # energy's gradient while every parameter check still passed.
                model.freeze_denoiser()
                parameters = [p for p in parameters if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters,
                                          lr=float(self.config["optimization"]["learning_rate"]))
            if resume_payload is not None and continuing_phase:
                # Validated above, before anything was built: the phase is recognised, the budgets
                # are exactly this run's and the optimizer state is present.
                optimizer.load_state_dict(resume_payload["optimizer"])
            if not continuing_phase:
                state.energy_rng = self.energy_generator(int(seed) + (0 if phase == "denoiser" else 1))
                state.history = []
            state.energy_phase = phase
            state.step = int(done[phase])
            generator = state.energy_rng
            history = list(state.history)
            total = int(budget[phase])
            progress = paths.StageProgress(directory / f"progress_{phase}.json",
                                           stage=f"energy::{phase}", total=total, every=25)
            progress.start()
            while state.step < total:
                state.step += 1
                # Only a family that exists is drawn, and ONLY the sampler's generator is consumed
                # here: conditioning dropout below draws from mask_rng, so a conditioned and a null
                # run take identical draws from the same seed.
                family = energy_contract.draw_family(present, state.sampler_rng)
                group = int(state.sampler_rng.integers(len(groups[family])))
                pool = groups[family][group][1]
                members = [int(index) for index in
                           state.sampler_rng.integers(0, len(pool), microbatch)]
                chosen = [pool[index] for index in members]
                tiles = data_lib._tiles(chosen[0].region, plan["max_query_sites"])
                tile = int(state.sampler_rng.integers(len(tiles)))
                sites = tiles[tile]
                chosen = [replace(row, construct=data_lib.with_sites(row.construct, sites), region=sites)
                          for row in chosen]
                windows = [[canonical_index(row.construct.residue_at(site)) for site in sites] for row in chosen]
                clean = onehot_window(torch.as_tensor(windows), residues=model.residues,
                                     scale=model.scale).to(self.device)
                # The null arm is given NO context: nothing is built, no tower is invoked.
                context = (self.energy_context(chosen, rng=state.mask_rng)
                           if conditioned else None)
                optimizer.zero_grad(set_to_none=True)
                if phase == "denoiser":
                    loss = model.denoiser_loss(clean, context, generator=generator)
                    diagnostics = {"objective": "MSE(xhat(x + sigma * eps), x)",
                                   "conditioned": context is not None}
                else:
                    loss, diagnostics = model.energy_loss(
                        clean, context, generator=generator,
                        sampling_options={"steps": walk_steps})
                require(bool(torch.isfinite(loss)),
                        f"Nonfinite {phase} loss at step {state.step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(self.config["optimization"]["gradient_clip"]),
                    error_if_nonfinite=True)
                optimizer.step()
                entry = {"step": state.step, "phase": phase, "loss": float(loss.detach()),
                         "gradient_norm": float(norm),
                         **{key: value for key, value in diagnostics.items()
                            if not isinstance(value, dict)}}
                history.append(entry)
                state.history = history
                # Attributed row by row: a pool is one (assay, region), not one source, and its
                # first row's source is not the source of the others. ``forwards`` is ROW-FORWARDS
                # per row: 1 in the denoiser phase, walk_steps + 2 in the energy phase (the walk,
                # then the real and the fake evaluation). It is not a count of batched calls.
                state.ledger.record(
                    task=data_lib.TASK_WHOLE, update=True,
                    contributions=data_lib.contributions_by_source(
                        sources=[row.source for row in chosen],
                        identities=[row.identity for row in chosen],
                        forwards=energy_contract.forwards_per_row(phase, walk_steps),
                        targets=len(sites), measured=True,
                        family=(data_lib.FAMILY_HER2 if family == energy_contract.FAMILY_HER2
                                else data_lib.FAMILY_OTHER)))
                # The same step, as the draws themselves: it advances with the weights and the
                # ledger, so every checkpoint saved below holds the trace of exactly its steps.
                trace.record(phase=phase, step=state.step, family=family, group=group, tile=tile,
                             sites=sites, members=members,
                             identities=[row.identity for row in chosen])
                progress.advance(f"{phase} {state.step}", completed=state.step)
                if heartbeat is not None:
                    heartbeat.beat(phase=f"energy::{phase}", step=state.step,
                                   loss=float(loss.detach()))
                if cadence.should_save_last(state.step, total=total):
                    checkpoints.save(checkpoint, model=model, heads=None, optimizer=optimizer,
                                     scheduler=None, state=state, identity=identity,
                                     extra=extras(phase))
                completed = (done["denoiser"] if phase == "energy" else 0) + state.step
                if stop is not None and completed >= stop:
                    break
            checkpoints.save(checkpoint, model=model, heads=None, optimizer=optimizer,
                             scheduler=None, state=state, identity=identity,
                             extra=extras(phase))
            progress.finish()
            phases[phase] = {"updates": int(state.step), "history": history,
                             "final_loss": history[-1]["loss"] if history else None}
            done[phase] = int(state.step)
            if state.step < total or (stop is not None and sum(done.values()) >= stop):
                break
            state.history = []
        if done == budget:
            checkpoints.save(directory / "best.pt", model=model, heads=None, optimizer=None,
                         scheduler=None, state=state, identity=identity,
                         extra=extras("energy",
                                      selection=("the terminal energy checkpoint. This route has "
                                                 "no normalised likelihood, so there is no "
                                                 "development log probability to select a best "
                                                 "step on; the selection rule is stated rather "
                                                 "than implied.")))
        record = {"route": ROUTE_ENERGY, "seed": int(seed),
                  "updates": sum(done.values()), "phase_updates": dict(done),
                  "arm": plan["arm"], "population": population, "conditioned": bool(conditioned),
                  "phases": phases, "contract": model.document(),
                  "energy_contract": contract,
                  "energy_contract_sha256": contract["energy_contract_sha256"],
                  "common_sha256": contract["common_sha256"],
                  "exposure_trace": trace.document(),
                  "shape": plan["shape"],
                  "memory": model.memory_report(),
                  "resume": resume_record,
                  "identity": dict(identity),
                  "training_rows": plan["rows"],
                  "forward_ledger": contract["forward_ledger"],
                  "matching": ("population, seed, cohort ordering, tile rule, microbatch and phase "
                               "budgets are bound in the energy contract and shared (common_sha256) "
                               "by a full and a null run; the architecture and initialization "
                               "match, the final weights do not"),
                  "exposure": state.ledger.document(),
                  "heldout_rule": ("the sampler draws from the TRAINING partition only. An "
                                   "earlier revision drew from every eligible row, so the learned "
                                   "density included the development panel."),
                  "phase_order": ("the denoiser trains first and is FROZEN while the energy "
                                  "trains, which is the pinned upstream contract. A joint "
                                  "objective would be a different model.")}
        paths.write_json(directory / "run.json", record)
        return record

    # -- development metric and gate ---------------------------------------
    def development_metric(self, route):
        """Select on measured affinity ordering; keep absolute likelihood diagnostic."""
        from . import her2_routes_eval as eval_lib
        panel = None

        def metric(model, heads, step):
            nonlocal panel
            if panel is None:
                panel = self.development_panel_outcomes()
            model.eval()
            with torch.no_grad():
                scores = self.score_panel(route, None, panel, model=model, heads=heads, batch_size=4)
            rho = eval_lib.spearman(scores, [row["value"] for row in panel])
            require(rho is not None and math.isfinite(rho), "Development affinity ranking is undefined")
            self._cache["last_development_metric"] = {
                "step": int(step), "spearman": rho, "mean_log_probability": float(np.mean(scores)),
                "rows": len(panel)}
            model.train()
            return -float(rho)
        return metric

    def selection_metric(self, route):
        """The callback ``train`` evaluates for checkpoint selection and the plateau schedule.

        A config with no ``training_phase`` gets exactly ``development_metric`` (the legacy
        selector, unchanged). Under the explicit DPLM training phase it is the HR17 training
        selector instead, which is a different statistic and is labelled as one.
        """
        policy = spec.training_phase(self.config)
        if policy is None:
            return self.development_metric(route)
        require(route in policy["routes"] and self.is_dplm(route),
                f"the {policy['phase_id']} training phase selects only {policy['routes']}, not "
                f"{route!r}")
        return self.hr17_training_metric(route)

    def hr17_training_metric(self, route):
        """A ``metric(model, heads, step)`` callback for the DPLM training phase.

        Independent of ``score_panel`` / ``development_metric``. ``last_diagnostics`` holds the
        JSON-compatible record of the most recent call, which ``train`` stores beside the step.
        """
        panel = None

        def metric(model, heads, step):
            nonlocal panel
            if panel is None:
                panel = self.development_panel_outcomes()
            diagnostics = self.hr17_selection(route, model, heads, panel, step=step)
            metric.last_diagnostics = diagnostics
            self._cache["last_selection_diagnostics"] = {
                key: diagnostics[key] for key in ("step", "metric", "spearman_ratio", "rows")}
            return diagnostics["metric"]
        metric.last_diagnostics = None
        metric.selection_metric = spec.SELECTION_METRIC_TRAINING
        return metric

    def _rng_devices(self):
        return ([torch.cuda.current_device()]
                if self.device.startswith("cuda") and torch.cuda.is_available() else [])

    def hr17_selection_terms(self, route, model, heads, panel, *, batch_size=4):
        """Per-row fixed-path mean log probability of the window under BIN_HIGH and BIN_NULL.

        One pair of views per panel row: the same construct, the same initial serialization,
        the same declared site order and targets, the same affinity family and the same HER2
        assay -- only the requested bin differs. The divisor is the actual number of fixed sites
        (ten). No diffusion sample likelihood is involved: the fixed declared-order path is
        scored under each condition.

        Model AND heads are put in eval mode under ``no_grad`` with the forked RNG, so no
        gradient, weight or generator state changes, and every module's incoming mode is
        restored in ``finally`` even if a forward raises.
        """
        from . import her2_data
        from .her2_routes_conditions import BIN_HIGH
        from .her2_routes_serial import ablang_serialization, her2_construct
        require(self.is_dplm(route), f"the HR17 training selector is defined for the DPLM routes, "
                                     f"not {route!r}")
        sites = her2_core_edit_map()
        require(len(sites) == 10, f"the fixed HER2 window has {len(sites)} sites, not ten")
        require(panel is not None and len(panel) > 0, "The HR17 selection panel is empty")
        assay = self.config["her2"]["assay"]
        high = Condition(family="affinity", bin=BIN_HIGH, assay=assay)
        null = Condition(family="affinity", bin=BIN_NULL, assay=assay)
        scaffold = self.scaffold()
        shared = []
        for index, row in enumerate(panel):
            require(isinstance(row, dict) and "window" in row and "value" in row,
                    f"HR17 panel row {index} is malformed: {row!r}")
            window = row["window"]
            require(isinstance(window, str) and len(window) == len(sites)
                    and set(window) <= set(her2_data.CANONICAL),
                    f"HR17 panel row {index} window {window!r} is not {len(sites)} canonical "
                    "residues")
            construct = her2_construct(scaffold, window, record_id=f"panel{index}")
            shared.append((construct, ablang_serialization(construct, order=sites)))
        snapshot = [(module, module.training) for root in (model, heads) if root is not None
                    for module in root.modules()]
        highs, nulls = [], []
        try:
            model.eval()
            if heads is not None:
                heads.eval()
            with torch.random.fork_rng(devices=self._rng_devices()), torch.no_grad():
                for start in range(0, len(shared), int(batch_size)):
                    chunk = shared[start:start + int(batch_size)]
                    batch_high = self.collate(route, [(c, s, high) for c, s in chunk])
                    batch_null = self.collate(route, [(c, s, null) for c, s in chunk])
                    for name in ("tokens", "query_slots", "targets", "attention_mask",
                                 "target_mask"):
                        require(np.array_equal(getattr(batch_high, name), getattr(batch_null, name)),
                                f"the high and null batches differ in {name}: only the bin may "
                                "differ between the two terms")
                    for batch, sink in ((batch_high, highs), (batch_null, nulls)):
                        log_probs = self.window_log_probs(route, model, batch)
                        sink.extend(float(value) / len(sites) for value in log_probs.sum(dim=1))
        finally:
            for module, flag in snapshot:
                module.training = flag
        return {"high": highs, "null": nulls, "divisor": len(sites), "assay": assay}

    def hr17_selection(self, route, model, heads, panel, *, step, batch_size=4):
        """The HR17 training selector: negative Spearman of (high - null) against measured pKD.

        The primary number is the mean fixed-path log high-versus-null ratio per panel row,
        ranked against the already-qualified outcome, sign-flipped so lower is better (the loop
        minimises). A tied constant ratio, a nonfinite value or an undefined correlation is
        refused rather than turned into a number; there is no all-zero fallback. The high-only and
        null-only correlations are diagnostics and may be ``None`` when undefined -- never NaN.
        """
        from . import her2_routes_eval as eval_lib
        terms = self.hr17_selection_terms(route, model, heads, panel, batch_size=batch_size)
        high = np.asarray(terms["high"], dtype=np.float64)
        null = np.asarray(terms["null"], dtype=np.float64)
        require(high.shape == null.shape == (len(panel),),
                f"HR17 scored {high.shape} high and {null.shape} null rows for {len(panel)} rows")
        require(bool(np.isfinite(high).all() and np.isfinite(null).all()),
                "HR17 produced a nonfinite window log probability")
        ratio = high - null
        outcomes = [float(row["value"]) for row in panel]
        rho = eval_lib.spearman(ratio, outcomes)
        require(rho is not None and math.isfinite(rho),
                "The HR17 high-minus-null ranking is undefined (a constant ratio or too few rows)")

        def diagnostic(scores):
            value = eval_lib.spearman(scores, outcomes)
            return None if value is None or not math.isfinite(value) else float(value)
        return _json_safe({
            "step": int(step), "metric_name": spec.SELECTION_METRIC_TRAINING,
            "metric": -float(rho), "spearman_ratio": float(rho),
            "spearman_high": diagnostic(high), "spearman_null": diagnostic(null),
            "mean_high": float(high.mean()), "mean_null": float(null.mean()),
            "mean_ratio": float(ratio.mean()), "rows": len(panel),
            "divisor": int(terms["divisor"]), "assay": terms["assay"],
            "high": high.tolist(), "null": null.tolist(), "ratio": ratio.tolist(),
            "note": ("training-time selector only: mean fixed-path high-minus-null log "
                     "probability, negative Spearman versus the qualified outcome. Not the "
                     "legacy development metric and not competition admission.")})

    def load_for_scoring(self, route, checkpoint, *, seed=0, arm=None):
        """Build a model and load a checkpoint into it for scoring. Identity checked.

        ``checkpoint=None`` is the frozen zero-shot backbone: the same architecture
        with the zero-initialised conditioning and no training. It is a real
        comparator rather than the absence of one, so it goes through this same
        function and is labelled.

        **The energy route** (``dwjs_energy``) has no pretrained backbone, so ``checkpoint=None``
        is a RANDOMLY INITIALISED model and is labelled as one, with an explicit mode (``arm``,
        ``full`` unless the caller names ``null``). A checkpoint is validated BEFORE the model is
        built: its version-2 energy contract is recovered, its hash RECOMPUTED and compared with
        ``identity.schedule_sha256`` (the stored hash is never copied into the expected identity),
        the arm and the conditioning mode must agree, and an explicit ``arm`` that contradicts the
        checkpoint is refused. The validated mode is then bound, immutably, to the returned model,
        so any ``model=`` reuse path keeps it. A checkpoint with no energy contract (written before
        version 2) is refused with an incompatibility error and left untouched.
        """
        absent = (f"{checkpoint} is absent. Scoring the zero-shot backbone under the name of a "
                  "trained arm is the one substitution this function refuses; pass None for the "
                  "backbone explicitly.")
        present = checkpoint is not None and Path(checkpoint).is_file()
        payload = binding = None
        if route == ROUTE_ENERGY and checkpoint is not None:
            require(present, absent)
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            binding = energy_contract.inference_binding(payload, path=checkpoint, arm=arm)
        with torch.random.fork_rng():
            model = self.build_route(route, seed=seed)
            heads = self.build_heads(route, model)
        record = {"checkpoint": None if checkpoint is None else str(checkpoint),
                  "trained": False}
        if present:
            # The energy route's checkpoints carry no head state: its two phases train
            # the denoiser and the energy, and its observation heads read the context
            # tower rather than a token trunk. Asking for head state that was never
            # written would refuse a valid checkpoint.
            if payload is None:
                payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            trained = payload["identity"]
            pin = self.admitted_pin(route, checkpoint)
            if pin is not None:
                # An imported checkpoint: arm, seed and schedule come from the committed pin, NOT
                # from the payload, and three identity fields come from the origin flight. The
                # file's hash, the step and the live route block are re-verified every time.
                require(arm is None or str(arm) == str(pin["arm"]),
                        f"{checkpoint} is admitted as the {pin['arm']!r} arm but the {arm!r} arm was "
                        "requested; scoring one arm's weights under the other's name is refused")
                self.require_admitted_load(route, checkpoint, pin, payload)
                expected = self.training_identity(
                    route, arm=pin["arm"], seed=pin["seed"],
                    schedule_sha256=pin["schedule_sha256"], pinned=pin)
            elif binding is None:
                expected = self.training_identity(
                    route, arm=trained["arm"] if arm is None else arm, seed=int(trained["seed"]),
                    schedule_sha256=trained.get("schedule_sha256"))
            else:
                stored, terminal = binding
                expected = self.training_identity(
                    route, arm=stored["arm"], seed=int(trained["seed"]),
                    schedule_sha256=energy_contract.contract_digest(stored))
            record = checkpoints.load_inference(
                Path(checkpoint), model=model, heads=None if route == ROUTE_ENERGY else heads,
                identity=expected, payload=payload)
            if binding is not None:
                runtime = energy_contract.checkpoint_runtime(
                    stored, path=checkpoint, step=record["step"], terminal=terminal)
                model.bind_runtime(runtime)
                record = {**record, "energy_runtime": runtime.document()}
        elif checkpoint is not None:
            require(False, absent)
        elif route == ROUTE_ENERGY:
            runtime = energy_contract.checkpoint_free_runtime(
                energy_contract.ARM_FULL if arm is None else arm)
            model.bind_runtime(runtime)
            record = {**record, "energy_runtime": runtime.document(),
                      "initialization": runtime.initialization,
                      "label": ("a randomly initialised, untrained energy model. There is no "
                                "pretrained walk-jump energy for this construct family, so this "
                                "is NOT a frozen zero-shot backbone; its mode is stated, not "
                                "inferred")}
        model.eval()
        heads.eval()
        return model, heads, record

    def development_views(self, route):
        """The frozen development panel's views. Reserved windows cannot appear here.

        The check is that the panel does not intersect the RESERVED set, not that its
        windows clear the exclusion closure. They cannot clear it: the closure guards
        the development windows too, precisely so no *training* consumer touches them,
        and evaluation is not a training consumer. Asking the closure to allow them
        raised on every evaluation, which is what this check replaces -- and the
        property that actually matters here is the one asserted.
        """
        from .her2_routes_conditions import Condition
        from .her2_routes_serial import (
            ablang_serialization,
            causal_serialization,
            her2_construct,
        )
        split = self.split()
        scaffold = self.scaffold()
        sites = her2_core_edit_map()
        reserved = set(split.reserved)
        leaked = sorted(set(split.development) & reserved)
        require(not leaked,
                f"{len(leaked)} window(s) are in BOTH the development panel and the reserved "
                f"test, e.g. {leaked[:3]}. Evaluating on a reserved core is the one thing the "
                "frozen split exists to prevent, and it is checked here rather than assumed from "
                "the certificate.")
        views = []
        for index, window in enumerate(split.development):
            construct = her2_construct(scaffold, window, record_id=f"dev{index}")
            condition = Condition(family="affinity", bin=BIN_NULL, assay="null")
            serialization = (causal_serialization(construct, order=sites,
                                                  residues=construct.residues(sites),
                                                  max_positions=self._positions(route))
                             if route == ROUTE_CAUSAL
                             else ablang_serialization(construct, order=sites))
            views.append((construct, serialization, condition))
        require(views, "The development panel produced no views")
        return tuple(views)

    def evaluate_route_gate(self):
        """G1 on the development panel: ranking and raw generation, Pareto, predeclared."""
        from . import her2_routes_eval as eval_lib
        gates = self.config["gates"]
        decisions, scores = {}, {}
        for route in self.config["routes"]["enabled"]:
            block = self.arm_scores(route)
            scores[route] = block
            decisions[route] = eval_lib.gate_decision(
                route=route, ranking=block["ranking"], generation=block["generation"],
                backbone_ranking=block["backbone_ranking"],
                distance_ranking=block["distance_ranking"],
                comparators=block["comparators"], delta=float(gates["delta"]),
                margins=dict(gates["margins"]))
        advancing = sorted(route for route, block in decisions.items() if block["advances"])
        winner = (max(advancing, key=lambda route: scores[route]["ranking"])
                  if advancing else None)
        return {"decisions": decisions, "scores": scores, "advancing": advancing,
                "winner": winner,
                "spr152": ("reported in the report stage beside this table and never gated on"),
                "criterion": "predeclared Pareto; see her2_routes_eval.gate_decision"}

    def independent_assessor(self):
        """A predictor of display quality that is not the route. Fitted once, cached.

        Independent in parameters and in objective -- an additive site model fitted
        by the inherited baseline code on the display training split -- and NOT
        independent in data, which is stated beside every number it produces. It is
        what "independently assessed predicted quality" means here; a route scoring
        its own draws is a measurement of its own prior.
        """
        if "assessor" in self._cache:
            return self._cache["assessor"]
        from . import her2_baselines as baselines
        from . import her2_data
        root = self.context.repository(self.config["roots"]["her2_raw"])
        train = self.display_pool()
        require(train is not None, "The independent assessor needs the closure-filtered display pool")
        index = her2_data.encode_cores(list(train.cores))
        targets = np.asarray(train.classes, dtype=np.int64)
        model, optimization = baselines.fit_additive_linear(index, targets)

        def score(windows):
            return [float(value) for value in
                    model.positive_scores(her2_data.encode_cores(list(windows)))]
        self._cache["assessor"] = {
            "score": score, "fit": {"parameters": optimization["parameters"],
                                    "objective_evaluations":
                                        optimization["objective_evaluations"],
                                    "train_cross_entropy":
                                        optimization["train_cross_entropy"]},
            "independence": ("independent in parameters and objective, NOT in data: it is fitted "
                             "on the same display library the policy's positives came from")}
        return self._cache["assessor"]

    def generation_endpoint(self, route, checkpoint, *, draws=None, seed=None):
        """The raw-generation endpoint: draws assessed independently, against the backbone.

        A measured quantity, unlike the earlier endpoint, which was the negated
        development NLL compared against a threshold of zero -- a positive loss
        negated is always below zero, so that gate could never pass.

        The route draws ``draws`` windows and so does its own frozen zero-shot
        backbone, both are scored by :meth:`independent_assessor`, and the endpoint
        is the difference of means. Novelty and duplicate rates are reported beside
        it, because a generator that reproduces its training set scores well on any
        assessor fitted on that set.

        **The energy route's comparator keeps the trained arm.** Its "backbone" is a randomly
        initialised model, and that comparator is built in the TRAINED model's mode (a null
        checkpoint's comparator is a null model, which is never given a context and never calls a
        context builder) and from the trained checkpoint's recorded initialization seed, so it is
        the model's own starting point. Loading it with the defaults would compare a null
        checkpoint with a conditioned random model and report a conditioning change as if it were
        the training gain. ``seed`` here is the SAMPLING seed and is a separate parameter.
        """
        from . import her2_routes_generate as generate
        from .her2_routes_conditions import BIN_HIGH, Condition
        block = self.config["gates"]
        draws = int(draws or block.get("generation_draws", 256))
        seed = int(seed if seed is not None else self.config["decoding"]["seed"])
        condition = Condition(family="affinity", bin=BIN_HIGH,
                              assay=self.config["her2"]["assay"])
        assessor = self.independent_assessor()
        sites = her2_core_edit_map()
        out = {}
        trained_arm, initialization_seed = None, 0
        seed_source = "the default initialization seed: the trained arm is itself checkpoint-free"
        for label, target in (("trained", checkpoint), ("backbone", None)):
            if route == ROUTE_ENERGY and label == "backbone":
                model, heads, load = self.load_for_scoring(
                    route, None, seed=initialization_seed, arm=trained_arm)
            else:
                model, heads, load = self.load_for_scoring(route, target)
            if route == ROUTE_ENERGY:
                # The mode is the loaded model's validated runtime contract. Context rows are only
                # needed (and only read) when the model was trained WITH a context.
                runtime = energy_contract.require_runtime(
                    model, where=f"generation_endpoint ({label})",
                    arm=trained_arm if label == "backbone" else None)
                if label == "trained":
                    trained_arm = runtime.arm
                    recorded = load.get("training_identity")
                    if recorded:
                        initialization_seed = int(recorded.get("seed", initialization_seed))
                        seed_source = "the trained checkpoint's recorded identity seed"
                rows = list(self.energy_rows(her2_only=True)) if runtime.conditioned else []
                result = generate.energy_generate(
                    self, model, draws=draws, context_rows=rows, seed=seed,
                    batch_size=int(self.config["routes"][ROUTE_ENERGY]["microbatch"]),
                    sampling_options={"steps": int(
                        self.config["routes"][ROUTE_ENERGY]["walk_steps"])},
                    mode=runtime.mode)
            else:
                # A route may declare the procedure the gate's raw draws use; the default is the
                # declared fill order. The DPLM routes declare their native iterative decoder,
                # and the trained and backbone arms use the SAME one.
                result = generate.fill_batch(
                    self, route, model, draws=draws, sites=sites, condition=condition,
                    strategy=str(self.route_block(route).get("gate_generation_strategy",
                                                             "declared")),
                    seed=seed, batch_size=int(self.selected_shapes()["microbatch"]) * 4)
            scores = assessor["score"](result.windows)
            unique = sorted(set(result.windows))
            out[label] = {"draws": len(result.windows), "unique": len(unique),
                          "duplicate_rate": 1.0 - len(unique) / max(1, len(result.windows)),
                          "assessed_mean": float(np.mean(scores)) if scores else None,
                          "mean_log_probability": result.document()["mean_log_probability"],
                          "accounting": result.accounting(), "checkpoint": load}
            if route == ROUTE_ENERGY and label == "backbone":
                out[label]["comparator"] = {
                    "arm": runtime.arm, "mode": runtime.mode,
                    "initialization": runtime.initialization,
                    "initialization_seed": initialization_seed,
                    "initialization_seed_source": seed_source,
                    "sampling_seed": seed}
            del model, heads
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        trained = out["trained"]["assessed_mean"]
        backbone = out["backbone"]["assessed_mean"]
        return {"value": (None if trained is None or backbone is None
                          else float(trained - backbone)),
                "arms": out, "assessor": assessor["fit"],
                "assessor_independence": assessor["independence"],
                "definition": ("mean independently assessed predicted quality of the route's own "
                               "raw draws minus the same for its frozen zero-shot backbone. "
                               "Novelty and duplicate rates are beside it because a generator that "
                               "reproduces its training set scores well on an assessor fitted on "
                               "that set."),
                **({"backbone_note": ("the energy route has no pretrained backbone: its "
                                      "'backbone' arm is a RANDOMLY INITIALISED, untrained energy "
                                      "model (see arms.backbone.checkpoint), not a frozen "
                                      "zero-shot model, built in the TRAINED arm's conditioning "
                                      "mode and from its recorded initialization seed (see "
                                      "arms.backbone.comparator), so the difference does not "
                                      "include a change of conditioning")}
                   if route == ROUTE_ENERGY else {}),
                "not_measured": ("nothing here has been experimentally assayed. An off-catalogue "
                                 "draw is an in-silico candidate.")}

    def arm_scores(self, route):
        """Ranking and generation endpoints for one route, plus its two controls.

        The backbone comparator is the **frozen zero-shot** model -- the same object
        with conditioning disabled and no training -- so "the route improved" is
        measured against what it started from rather than against nothing.
        """
        from . import her2_routes_controls as controls_lib
        from . import her2_routes_eval as eval_lib
        directory = self.context.path("bakeoff", route)
        record = self.bakeoff_record(route)
        panel = self.development_panel_outcomes()
        checkpoint = self.require_checkpoint(
            directory / "best.pt", stage="route_gate", route=route,
            role="bake-off best checkpoint")
        outcomes = [row["value"] for row in panel]
        def required(name, value):
            # An undefined correlation or absent endpoint is missing evidence, not 0.0;
            # only a genuine finite zero may reach the gate as zero.
            return eval_lib.finite_scalar(value, f"arm_scores[{route}] {name}")
        def ranking(name, scores):
            # Validated as soon as the scores exist, so an invalid ranking stops the arm
            # before the next model pass and, above all, before generation.
            try:
                rho = eval_lib.spearman(scores, outcomes)
            except ValueError as error:
                raise ValueError(f"arm_scores[{route}] {name}: {error}") from error
            return required(name, rho)
        ranked = {}
        ranked["ranking"] = ranking("ranking", self.score_panel(route, checkpoint, panel))
        ranked["backbone_ranking"] = ranking("backbone_ranking",
                                             self.score_panel(route, None, panel))
        ranked["distance_ranking"] = ranking("distance_ranking", controls_lib.distance_only_baseline(
            [row["window"] for row in panel], wild_type=self.config["her2"]["wild_type_core"]))
        comparators = self.comparator_scores(panel)
        comparator_values = comparators.get("scores") if isinstance(comparators, dict) else None
        require(comparator_values,
                f"arm_scores[{route}] comparator scores are missing; the gate needs them")
        for name, value in comparator_values.items():
            required(f"comparator {name}", value)
        generation = self.generation_endpoint(route, checkpoint)
        return {**ranked,
                "generation": required("generation", generation.get("value")),
                "generation_detail": generation,
                # Only the SCALAR comparator scores reach the gate. The fit record is
                # a nested dict, and passing one to float() is a TypeError the gate
                # would raise on rather than a comparison it would make.
                "comparators": dict(comparators["scores"]),
                "comparator_detail": comparators,
                "selected_checkpoint": None if checkpoint is None else str(checkpoint),
                "best": record.get("best"),
                "panel": len(panel),
                "backbone_is": (
                    "a RANDOMLY INITIALISED, untrained energy model: this route has no pretrained "
                    "backbone, so its comparator is the seed's initial weights, not a frozen "
                    "zero-shot model"
                    if route == ROUTE_ENERGY else
                    "the frozen zero-shot model: the same architecture with the "
                    "zero-initialised conditioning and no training")}

    def development_panel_outcomes(self):
        """The development windows with their EXACT outcomes. Reserved stays closed.

        The outcome of a window is the mean of its summaries' :attr:`exact_value` -- the mean of
        replicates that are all exact finite points. ``summary.value`` is the mean of the
        uncensored replicates, which for a construct with a floor replicate and two points is a
        number the assay never produced, and every ranking, correlation and matching on this panel
        would inherit it. A summary that is not an exact point is excluded and counted by its own
        reason (:meth:`development_panel_accounting`); the frozen candidate membership, the window
        identities and their order are the split's and are untouched.
        """
        split = self.split()
        development = set(split.development)
        by_window, excluded, measured = {}, {}, set()
        for record in self.summaries():
            if record.source != "alphabind_her2_validation":
                continue
            window = record.construct.residues(record.construct.sites)
            if window not in development:
                continue
            measured.add(window)
            exact = record.exact_value
            if exact is None:
                excluded[record.exact_status] = excluded.get(record.exact_status, 0) + 1
                continue
            by_window.setdefault(window, []).append(exact)
        rows = [{"window": window, "value": float(np.mean(by_window[window]))}
                for window in split.development if window in by_window]
        self._cache["development_panel_accounting"] = {
            "frozen_development_windows": len(development),
            "windows_with_a_measured_summary": len(measured),
            "windows_with_an_exact_outcome": len(rows),
            "windows_without_an_exact_outcome": len(measured) - len(rows),
            "summaries_excluded_by_exact_status": dict(sorted(excluded.items())),
            "rule": ("the outcome is the mean exact_value of a window's summaries; a summary that "
                     "is censored, mixed-censored, non-finite or has room around its value is "
                     "excluded and counted, and membership is the frozen split's")}
        require(rows,
                "No development window carries an exact outcome. The panel exists but is "
                "unlabelled, so no ranking endpoint can be computed on it: "
                f"{self._cache['development_panel_accounting']}")
        return rows

    def development_panel_accounting(self):
        """What :meth:`development_panel_outcomes` kept and excluded, and why."""
        if "development_panel_accounting" not in self._cache:
            self.development_panel_outcomes()
        return dict(self._cache.get("development_panel_accounting")
                    or {"note": "this panel supplied no accounting"})

    def score_panel(self, route, checkpoint, panel, *, batch_size=16, model=None,
                    heads=None, arm=None, seed=0, requested_bin=None):
        """Mean window log probability under the requested bin, per row.

        ``requested_bin=None`` is the high-affinity bin, which is what this method scored
        before the keyword existed, byte for byte. Any other known bin (``null`` in particular)
        goes through the same collation and serialization for the route, so a high-versus-null
        comparison for any route is two calls of one code path rather than a second copy of the
        encoding. The energy route has no bin; asking it for one is refused, not ignored.

        The MEAN, not the raw sum: the repository's ranking convention, because the
        campaign compares across windows of different lengths and the sum grows with
        length. Batched, because a 100-row panel scored one row at a time is 100
        forwards of one row each and the confirmation stage scores it per seed.

        ``model``/``heads`` let a caller reuse a loaded pair across several panels
        instead of reloading the trunk for each one.

        ``arm`` is optional. For the likelihood routes ``None`` means ``full``, exactly as the
        default was. For the energy route ``None`` means "whatever the checkpoint says": the arm
        and conditioning mode come from its validated energy contract (or from the runtime bound
        to a ``model=`` you pass), and a named ``arm`` that contradicts them is refused rather than
        scoring the null arm's weights as the full arm. The null arm is scored with no context.
        """
        from .her2_routes_conditions import BIN_HIGH, Condition
        from .her2_routes_serial import (
            ablang_serialization,
            causal_serialization,
            her2_construct,
        )
        # Validated before anything is loaded: a refused request must cost nothing.
        if route == ROUTE_ENERGY:
            require(requested_bin is None,
                    "The energy route has no bin: it scores -E(clean window, context), so a "
                    f"requested bin of {requested_bin!r} would be silently ignored")
        condition = Condition(family="affinity", bin=requested_bin or BIN_HIGH,
                              assay=self.config["her2"]["assay"])
        owned = model is None
        if owned:
            model, heads, _record = self.load_for_scoring(
                route, checkpoint, seed=seed,
                arm=arm if (arm is not None or route == ROUTE_ENERGY) else "full")
        if route == ROUTE_ENERGY:
            scores = self.energy_panel_scores(model, panel, arm=arm)
            if owned:
                del model, heads
            return scores
        scaffold = self.scaffold()
        sites = her2_core_edit_map()
        views = []
        for index, row in enumerate(panel):
            construct = her2_construct(scaffold, row["window"], record_id=f"panel{index}")
            serialization = (causal_serialization(construct, order=sites,
                                                  residues=construct.residues(sites),
                                                  max_positions=self._positions(route))
                             if route == ROUTE_CAUSAL
                             else ablang_serialization(construct, order=sites))
            views.append((construct, serialization, condition))
        scores = []
        with torch.no_grad():
            for start in range(0, len(views), int(batch_size)):
                batch = self.collate(route, views[start:start + int(batch_size)])
                log_probs = self.window_log_probs(route, model, batch)
                scores.extend(float(value) / len(sites) for value in log_probs.sum(dim=1))
        if owned:
            del model, heads
        return scores

    def energy_panel_scores(self, model, panel, *, batch_size=16, arm=None):
        """``-E(clean window, context)`` per panel row. The energy route's ranking score.

        Negative energy, on the clean one-hot window with the model's own context. It
        is **not** a normalised sequence likelihood and is never reported as one: the
        route's cross-route comparison is by ranking and raw generation, which is why
        this is a ranking score with a declared definition rather than a log
        probability with a borrowed name. An earlier revision called
        ``canonical_log_probs`` on this model, which it does not have.

        The conditioning mode is the model's validated runtime contract, not its architecture:
        the null arm owns condition tables and context towers it was never given a context for.
        A model with no runtime contract is refused, a named ``arm`` that contradicts it is
        refused, and the null arm is scored with ``context=None`` -- neither the context builder
        nor either tower is invoked, and its energy reads only the window.
        """
        from ..models.her2_routes_energy import onehot_window
        from .her2_routes_serial import canonical_index, her2_construct
        runtime = energy_contract.require_runtime(model, where="energy_panel_scores", arm=arm)
        scaffold = self.scaffold()
        sites = her2_core_edit_map()
        rows, scores = [], []
        for index, row in enumerate(panel):
            construct = her2_construct(scaffold, row["window"], record_id=f"energy{index}")
            rows.append(data_lib.ConstructSummary(
                construct=construct, study="alphabind_her2", assay=self.config["her2"]["assay"],
                target="HER2", parent="alphabind_her2_validation::parental", family="affinity",
                kind="point", measurements=(), region=tuple(sites), partition="development",
                scale="pKD", source="alphabind_her2_validation"))
        with torch.no_grad():
            for start in range(0, len(rows), int(batch_size)):
                chunk = rows[start:start + int(batch_size)]
                windows = torch.as_tensor(
                    [[canonical_index(row.construct.residue_at(site)) for site in sites]
                     for row in chunk], dtype=torch.long)
                clean = onehot_window(windows, residues=model.residues,
                                     scale=model.scale).to(self.device)
                context = (self.energy_context(chunk, answer_free=True)
                           if runtime.conditioned else None)
                pooled, site_vectors, mask = model.encode_context(context)
                energy = model.energy(clean, pooled, site_vectors, mask)
                scores.extend(float(-value) for value in energy)
        return scores

    def comparator_scores(self, panel):
        """The inherited discriminative comparators, on the same panel."""
        from . import her2_baselines as baselines
        from . import her2_data
        root = self.context.repository(self.config["roots"]["her2_raw"])
        train = her2_data.load_split(root, "train")
        index = her2_data.encode_cores(train.seq)
        targets = baselines.class_targets(train["class"])
        panel_index = her2_data.encode_cores([row["window"] for row in panel])
        # fit_additive_linear returns (model, optimization record); the ranking score
        # is positive_scores, i.e. P(high). Verified against the source rather than
        # guessed, because a `.scores` attribute does not exist on this class.
        additive, optimization = baselines.fit_additive_linear(index, targets)
        from . import her2_routes_eval as eval_lib
        outcomes = [row["value"] for row in panel]
        # ``scores`` holds ONLY scalars, and it is the only key the gate reads. The
        # earlier record mixed a nested fit block in beside the scores and the gate's
        # ``max(float(value) for value in comparators.values())`` raised on it.
        def required(name, scores):
            # An undefined comparator correlation is missing evidence, never 0.0.
            try:
                rho = eval_lib.spearman(scores, outcomes)
            except ValueError as error:
                raise ValueError(f"comparator_scores {name}: {error}") from error
            return eval_lib.finite_scalar(rho, f"comparator_scores {name} Spearman")
        return {"scores": {"additive_linear": required(
                    "additive_linear", additive.positive_scores(panel_index)),
                    "negative_wt_distance": required(
                        "negative_wt_distance", baselines.negative_wt_distance(panel_index))},
                "fit": {"parameters": int(optimization["parameters"]),
                        "objective_evaluations": int(optimization["objective_evaluations"]),
                        "train_cross_entropy": float(optimization["train_cross_entropy"])},
                "frozen_backbone_note": ("the frozen zero-shot backbone is reported by arm_scores "
                                         "as backbone_ranking, not here: it is the route's own "
                                         "starting point rather than a discriminative comparator"),
                "proxy_label": ("these comparators are fit on the same display library the "
                                "policy's positives came from, so they are independent in "
                                "parameters and objective and NOT in data"),
                "no_generator_probability": ("a discriminative comparator has no raw-generation "
                                             "distribution. None is invented for it, and the gate "
                                             "does not require the decoder to out-rank it.")}

    # -- T2 controls -------------------------------------------------------
    def run_controls(self, *, heartbeat=None):
        """Train the matched-exposure control arms and evaluate E17 and E19.

        Every arm draws the **same** rows from the same seeded stream: the arm table
        in ``her2_routes_tasks`` decides which objective those rows feed, and an
        excluded objective is replaced by reconstruction on the identical batch
        rather than skipped. That is what makes ``require_matched_exposure`` a check
        that can pass -- an arm that skipped its excluded task would present fewer
        rows, and the comparison would confound the removed objective with a shorter
        schedule.
        """
        from . import her2_routes_controls as controls_lib
        from . import her2_routes_eval as eval_lib
        from . import her2_routes_report as report_lib
        # The selected route is resolved first and refused if it has no token likelihood: the
        # matched-exposure arms are token-objective ablations, so nothing below is defined for it.
        gate = read_stage_record_safe(self.context, "route_gate")
        route = (gate or {}).get("winner") or self.config["routes"]["enabled"][0]
        capabilities.require_token_likelihood(route, "FlightServices.run_controls")
        # Every arm trains on this one curriculum, so the shape is validated once and for all of
        # them BEFORE any arm builds a model: a microbatch that cannot carry the declared local
        # recipe (an even size of at least four when interactions are requested) is not a
        # property of one arm, and discovering it on the third would leave two trained for nothing.
        shapes = self.selected_shapes()
        block = self.config["routes"][route]
        updates = int(self.config["controls"]["updates"])
        arms, ledgers = {}, {}
        for arm in (controls_lib.ARM_FULL, *self.config["controls"]["arms"]):
            permutation = None
            if arm == controls_lib.ARM_OUTCOME_PERMUTED:
                permutation = self.outcome_permutation(seed=int(block["seed"]))
            directory = self.context.path("controls", arm)
            record = self.train(
                route, stream=self.curriculum_stream(route, seed=int(block["seed"]), arm=arm,
                                                     permutation=permutation, updates=updates),
                updates=updates, seed=int(block["seed"]), directory=directory, arm=arm,
                heartbeat=heartbeat, evaluate=self.development_metric(route))
            arms[arm] = record
            ledgers[arm] = record["exposure"]
        matched = controls_lib.require_matched_exposure(ledgers)
        panel = self.development_panel_outcomes()
        outcomes = [row["value"] for row in panel]
        scores = {}
        for arm in arms:
            best = self.require_checkpoint(
                self.context.path("controls", arm, "best.pt"), stage="controls", route=route,
                role=f"control arm {arm!r} best checkpoint")
            scores[arm] = self.score_panel(route, best, panel, arm=arm, seed=int(block["seed"]))
        # An undefined correlation (a constant score, or fewer than two rows) is ``None`` and stays
        # ``None``: ``or 0.0`` turned "no ordering" into a correlation of zero, which a control
        # arm then "failed to beat" or "matched" by arithmetic.
        rankings = {arm: eval_lib.spearman(values, outcomes) for arm, values in scores.items()}
        full_best = self.require_checkpoint(
            self.context.path("controls", controls_lib.ARM_FULL, "best.pt"), stage="controls",
            route=route, role="control full-arm best checkpoint")
        e19 = report_lib.e19_shortcuts(
            self, route, full_best, margins=self.config["controls"]["margins"],
            training_bank=self.training_label_bank(),
            adaptation_bank={"windows": list(self.adaptation_bank()), "values": None,
                             "provenance": self.adaptation_bank_record()})
        margins = dict(self.config["controls"]["margins"])
        failed, failure_reasons = controls_lib.control_gate(
            rankings, required=self.config["controls"]["must_not_match"],
            margin=float(margins["control"]), e19=e19)
        objectives = {arm: record["arm_objectives"] for arm, record in arms.items()}
        return {"arms": {arm: {key: value for key, value in record.items()
                               if key != "history"}
                         for arm, record in arms.items()},
                "rankings": rankings, "matched_exposure": matched,
                "arm_objectives": objectives,
                "e17": {"arm_rankings": rankings,
                        "reading": ("the full arm must beat the reconstruction-only, head-only and "
                                    "both permuted arms by the declared margin. A head-only arm "
                                    "that matches the full arm means the gain is in the head, "
                                    "not the decoder.")},
                "e19": e19, "failed": failed, "failure_reasons": failure_reasons,
                "survives": not failed, "shapes": shapes,
                "exposure_note": ("rows are matched exactly; student forwards legitimately differ, "
                                  "because an arm running reconstruction on a contrast batch makes "
                                  "no second scored pass. Both are reported.")}

    def _permute_summaries(self, rows, *, keys, seed):
        """Replace each summary's measurements with a within-stratum permuted point.

        Operates on :class:`ConstructSummary` rows, because that is what the stream
        draws. Each permuted row keeps its construct, its region and its partition and
        carries **one** point measurement at the permuted value: the permutation
        destroys the sequence-to-outcome correspondence, and preserving a replicate
        structure whose values came from another construct would be a fiction about
        that construct's replicates.
        """
        from dataclasses import replace
        rows = tuple(rows)
        require(len(rows) == len(keys), "One permutation stratum is required per construct")
        groups = {}
        for index, (row, key) in enumerate(zip(rows, keys)):
            groups.setdefault((str(key), row.assay, row.parent, row.kind), []).append(index)
        donors = np.arange(len(rows))
        rng = np.random.default_rng(int(seed))
        for key in sorted(groups):
            members = np.asarray(groups[key])
            donors[members] = members[rng.permutation(len(members))]
        out = []
        for row, index in zip(rows, donors):
            donor = rows[int(index)]
            out.append(replace(row, measurements=donor.measurements, ordinal_class=donor.ordinal_class,
                               raw={**row.raw, "outcome_donor_key": donor.population_key}))
        self._cache.setdefault("permutations", {})[str(seed)] = {
            "rows": len(rows), "bundles_moved": int(np.count_nonzero(donors != np.arange(len(rows)))),
            "strata": len(groups), "preserved": "complete replicate, uncertainty, censoring and ordinal bundles"}
        return tuple(out)

    def outcome_permutation(self, *, seed):
        """Shuffle outcomes WITHIN assay and distance strata. Marginals survive.

        Within-stratum, so the outcome's marginal distribution and its relationship to
        edit distance both survive and only the sequence-outcome correspondence is
        destroyed. An arm that beat a globally shuffled control would only have
        learned about distance.
        """
        from . import her2_routes_controls as controls_lib

        def permute(rows):
            keys = [f"{row.assay}::"
                    f"{controls_lib.distance_stratum(len(row.mutated_sites)) if row.raw.get('mutation_map_known') else 'unknown'}"
                    for row in rows]
            return self._permute_summaries(rows, keys=keys, seed=int(seed))
        return permute

    def background_permutation(self, *, seed):
        """Shuffle outcomes within canonically oriented substitution groups.

        The group key is the row's own substitution set in canonical orientation, so
        the substitution marginals survive and only the background dependence is
        destroyed. An orientation-dependent key would put the same substitution in two
        groups depending on which member was read first.
        """
        def permute(rows):
            keys = []
            for row in rows:
                sites = tuple(sorted(row.mutated_sites)) or tuple(sorted(row.region))[:1]
                signature = ";".join(
                    f"{chain}{int(index)}:{row.construct.residue_at((chain, int(index)))}"
                    for chain, index in sites)
                keys.append(f"{row.assay}::{len(sites)}::{signature}")
            return self._permute_summaries(rows, keys=keys, seed=int(seed) + 1)
        return permute

    # -- T3 priors ---------------------------------------------------------
    def her2_training_bank(self):
        """``(windows, values)`` for the eligible measured HER2 TRAINING rows with an EXACT value.

        The teacher's training set, the distillation bank and the nearest-labelled-neighbour
        baseline all come from here, so none can contain a development or reserved core: the
        partition is the frozen split's own assignment and the closure is consulted at its radius.

        This is the EXACT QUANTITATIVE bank: ``exact_value``, not ``value``, because a numeric
        target must be a number the assay produced, and a construct with one censored and two exact
        replicates has an uncensored-only mean that no assay produced. It is NOT the adaptation
        population -- a censored construct still trains the head and whole-window tasks, so the
        geometry questions are answered by :meth:`adaptation_bank`. What this bank kept and
        excluded is in ``_cache['her2_training_bank_record']`` (:meth:`training_label_bank`).
        """
        eligible, counts = her2_training_population(self.summaries(), self.closure())
        exact = [(window, float(row.exact_value)) for window, row in eligible
                 if row.exact_value is not None]
        windows = [window for window, _value in exact]
        values = [value for _window, value in exact]
        require(windows,
                "The teacher has no eligible measured HER2 training bank. Every row was held out "
                "by the frozen split, refused by the closure at its radius, or not an exact "
                f"scalar measurement ({counts}).")
        record = {
            "kind": "exact_labelled_training_bank", "rows": len(windows),
            "sha256": _digest_lines(f"{window}\t{value!r}" for window, value in sorted(exact)),
            "counts": counts,
            "label_semantics": ("exact_value of the construct: the mean of replicates that are all "
                                "exact finite points. Never the uncensored-only mean, never a "
                                "display class."),
            "population": ("HER2 (alphabind_her2_validation) TRAINING partition, closure-filtered, "
                           "exact scalar rows only. Not the adaptation population and not a "
                           "measure of realised exposure.")}
        cache = getattr(self, "_cache", None)
        if isinstance(cache, dict):
            cache["her2_training_bank_record"] = record
        return windows, values

    def training_label_bank(self):
        """``{"windows", "values", "provenance"}``: the labelled bank E19's baseline reads."""
        windows, values = self.her2_training_bank()
        return {"windows": windows, "values": values,
                "provenance": dict(self._cache["her2_training_bank_record"])}

    def window_scores(self, route, model, windows, *, condition, batch_size=32, sites=None,
                      base=None):
        """``(N,)`` summed window log probability for raw cores. Differentiable.

        Used by the distillation objective, so it is deliberately not wrapped in
        ``no_grad``: the student's parameters have to be in the graph for a
        distillation step to be a step.

        ``sites`` (default: the ten-residue HER2 core) scores a partial, light-chain or
        mixed site set instead: each window is written into the wild-type scaffold at those
        sites, in the supplied order, and scored in that order. ``base`` (default: the
        wild-type core) is the ten-residue core that fills every NON-scored core position,
        so a decode from a seed window is rescored on the scaffold it was decoded on.
        """
        capabilities.require_token_likelihood(route, "FlightServices.window_scores")
        from .her2_routes_serial import (
            ablang_serialization,
            causal_serialization,
            her2_construct,
        )
        scaffold = self.scaffold()
        custom = sites is not None
        sites = tuple(sites) if custom else her2_core_edit_map()
        wild = base or self.config["her2"]["wild_type_core"]
        views = []
        for index, window in enumerate(windows):
            if custom:
                require(len(window) == len(sites),
                        f"A window of {len(window)} residues does not fill {len(sites)} sites")
                construct = data_lib.with_sites(
                    her2_construct(scaffold, wild, record_id=f"score{index}"),
                    sites).with_residues(sites, window)
            else:
                construct = her2_construct(scaffold, window, record_id=f"score{index}")
            serialization = (causal_serialization(construct, order=sites,
                                                  residues=construct.residues(sites),
                                                  max_positions=self._positions(route))
                             if route == ROUTE_CAUSAL
                             else ablang_serialization(construct, order=sites))
            views.append((construct, serialization, condition))
        chunks = []
        for start in range(0, len(views), int(batch_size)):
            batch = self.collate(route, views[start:start + int(batch_size)])
            chunks.append(self.window_log_probs(route, model, batch).sum(dim=1))
        return torch.cat(chunks, dim=0)

    def run_priors(self, *, heartbeat=None):
        """Fit each candidate teacher, qualify it against both controls, then distil.

        The order is the substance. A teacher is **fitted** on the measured training
        rows, **qualified** on the development panel against a shuffled-outcome
        teacher and a position-only projection of itself, and only then does the
        student take an optimizer step under it. An earlier revision set
        ``distilled = qualified`` and performed no update at all, so the flag was the
        distillation.

        Both uses of a qualified teacher are run and reported separately: the
        distillation (student weights move) and the rescoring (they do not). "The
        teacher helped" is otherwise two mechanisms under one name.
        """
        from . import her2_routes_eval as eval_lib
        from . import her2_routes_teachers as teachers_lib
        # The selected route first: a teacher is distilled into a token-likelihood student, so a
        # route without one is refused before the development panel or the training bank is built.
        gate = read_stage_record_safe(self.context, "route_gate")
        route = (gate or {}).get("winner") or self.config["routes"]["enabled"][0]
        capabilities.require_token_likelihood(route, "FlightServices.run_priors")
        block = self.config["priors"]
        margin = float(block["qualification_margin"])
        panel = self.development_panel_outcomes()
        outcomes = [row["value"] for row in panel]
        windows = [row["window"] for row in panel]
        bank, values = self.her2_training_bank()
        wild = self.config["her2"]["wild_type_core"]
        teachers, skipped, distillations = {}, {}, {}
        declared = dict(block["teachers"])
        for name, entry in sorted(declared.items()):
            if name == teachers_lib.TEACHER_STRUCTURAL:
                provenance = teachers_lib.structural_teacher_provenance(
                    entry.get("checkpoint"), allow_read=bool(entry.get("allow_read")))
                if not (provenance["present"] and entry.get("available")):
                    skipped[name] = {**provenance,
                                     "policy": ("recorded with its measured prerequisite state. "
                                                "No placeholder implementation is substituted "
                                                "and no generic 'unavailable' flag is used.")}
                    continue
            if not entry.get("available"):
                skipped[name] = {"reason": entry.get("reason") or "not available in this tree",
                                 "policy": ("recorded as skipped with its prerequisite evidence; "
                                            "no placeholder implementation is substituted")}
                continue
            if name == teachers_lib.TEACHER_NEIGHBOUR:
                scores = self.neighbour_teacher_scores(windows, bank, values,
                                                       radius=int(block["radius"]),
                                                       wild_type=wild)
                decision = teachers_lib.qualify(
                    name=name, real=scores["real"], shuffled=scores["shuffled"],
                    position_only=scores["position_only"], outcomes=outcomes, margin=margin,
                    spearman=eval_lib.spearman)
                decision["degeneracy"] = scores["degeneracy"]
                teachers[name] = decision
                if not decision["qualified"]:
                    skipped[name] = {"reason": "did not clear both controls", "measured": decision}
                continue
            require(name == teachers_lib.TEACHER_FUNCTIONAL,
                    f"No teacher implementation named {name!r}. Declare it available only when "
                    "there is a fitted predictor behind the name.")
            fit = teachers_lib.fit_functional_teacher(
                bank, values, seed=int(block["shuffle_seed"]),
                updates=int(block.get("teacher_updates", 400)),
                learning_rate=float(block.get("teacher_learning_rate", 0.01)),
                rank=int(block.get("teacher_rank", 16)))
            control = teachers_lib.fit_functional_teacher(
                bank, values, seed=int(block["shuffle_seed"]),
                updates=int(block.get("teacher_updates", 400)),
                learning_rate=float(block.get("teacher_learning_rate", 0.01)),
                rank=int(block.get("teacher_rank", 16)), shuffle_outcomes=True)
            real = fit.score(windows)
            shuffled = control.score(windows)
            position_only = teachers_lib.position_only_scores(windows, wild_type=wild,
                                                              teacher=fit)
            decision = teachers_lib.qualify(
                name=name, real=real, shuffled=shuffled, position_only=position_only,
                outcomes=outcomes, margin=margin, spearman=eval_lib.spearman)
            decision["fit"] = fit.document()
            decision["shuffled_fit"] = control.document()
            decision["matched_exposure"] = {
                "teacher_rows": fit.rows, "control_rows": control.rows,
                "teacher_updates": len(fit.history), "control_updates": len(control.history),
                "matched": (fit.rows == control.rows
                            and len(fit.history) == len(control.history))}
            teachers[name] = decision
            if not decision["qualified"]:
                skipped[name] = {"reason": "did not clear both controls on development",
                                 "measured": decision}
                continue
            distillations[name] = self.distil_teacher(route, fit, bank=bank,
                                                      panel=panel, heartbeat=heartbeat)
        return {"teachers": teachers, "skipped": skipped, "distillations": distillations,
                "route": route,
                "bank": {"rows": len(bank), "partition": "training only, closure-filtered"},
                "cost_warning": ("AbMAP's default 100 mutation augmentations per candidate make "
                                 "5,000 candidates cost 505,000 backbone evaluations. Teachers are "
                                 "cached in separate jobs and charged to the forecast."),
                "order": ("fit on training rows, qualify on development against BOTH controls, "
                          "then take student optimizer steps. A flag set beside a qualification "
                          "is not a distillation."),
                "agreement_caveat": "agreement with a teacher is not independent biological validation"}

    def neighbour_teacher_scores(self, windows, bank, values, *, radius, wild_type):
        """The neighbour-mean teacher and its controls, plus its measured degeneracy.

        Kept because the plan names it, and reported with the reason it cannot work
        on this panel: every panel core is at least the declared novelty separation
        from the whole old display library, so a radius-2 neighbour query returns an
        empty set and the score collapses to the bank mean -- one constant for every
        candidate, which has no ranking at all. The degeneracy is counted rather than
        described.
        """
        from . import her2_routes_controls as controls_lib
        rng = np.random.default_rng(int(self.config["priors"]["shuffle_seed"]))
        shuffled_values = list(np.asarray(values)[rng.permutation(len(values))])

        def score(queries, bank_values):
            out, empty = [], 0
            for window in queries:
                near = [value for member, value in zip(bank, bank_values)
                        if controls_lib.hamming(window, member) <= int(radius)]
                if not near:
                    empty += 1
                out.append(float(np.mean(near)) if near else float(np.mean(bank_values)))
            return out, empty
        real, empty = score(windows, values)
        shuffled, _ = score(windows, shuffled_values)
        position_only, _ = score([wild_type] * len(windows), values)
        distinct = len(set(round(value, 12) for value in real))
        return {"real": real, "shuffled": shuffled, "position_only": position_only,
                "degeneracy": {"queries": len(windows), "with_no_neighbour": empty,
                               "distinct_scores": distinct,
                               "constant": distinct <= 1,
                               "reading": ("a radius-{} query against a panel separated by at "
                                           "least the declared novelty distance returns nothing, "
                                           "so the score is the bank mean for every candidate. "
                                           "That is not a teacher.").format(int(radius))}}

    def distil_teacher(self, route, fit, *, bank, panel, heartbeat=None):
        """Real student updates under a qualified teacher, with before/after ranking."""
        capabilities.require_token_likelihood(route, "FlightServices.distil_teacher")
        from . import her2_routes_eval as eval_lib
        from . import her2_routes_teachers as teachers_lib
        from .her2_routes_conditions import BIN_HIGH, Condition
        block = self.config["priors"]
        start = self.checkpoint_lineage("priors", route)
        model, heads, load = self.load_for_scoring(
            route, start["path"], seed=int(self.config["routes"][route]["seed"]))
        outcomes = [row["value"] for row in panel]
        before = eval_lib.spearman(self.score_panel(route, None, panel, model=model,
                                                    heads=heads), outcomes)
        condition = Condition(family="affinity", bin=BIN_HIGH,
                              assay=self.config["her2"]["assay"])
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.build_optimizer(
            route, parameters, lr=float(block.get("distillation_learning_rate", 1e-5)))
        rng = np.random.default_rng([int(block["shuffle_seed"]), 9])
        model.train()
        record = teachers_lib.distil(
            self, route, model=model, heads=heads, teacher=fit, bank=list(bank),
            optimizer=optimizer, updates=int(block.get("distillation_updates", 200)),
            batch_size=int(block.get("distillation_batch", 16)),
            temperature=float(block.get("distillation_temperature", 0.5)),
            rng=rng, condition=condition)
        model.eval()
        after_scores = self.score_panel(route, None, panel, model=model, heads=heads)
        after = eval_lib.spearman(after_scores, outcomes)
        directory = self.context.path("priors", route)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints.save(directory / "distilled.pt", model=model, heads=heads,
                         optimizer=optimizer, scheduler=None,
                         state=self.fresh_state(int(self.config["routes"][route]["seed"])),
                         identity=self.training_identity(
                             route, arm="distilled",
                             seed=int(self.config["routes"][route]["seed"])))
        rescoring = teachers_lib.rescore(
            [row["window"] for row in panel],
            student_scores=self.score_panel(route, None, panel, model=model, heads=heads),
            teacher=fit, weight=float(block.get("rescore_weight", 1.0)))
        rescored = eval_lib.spearman(rescoring["combined"], outcomes)
        del model, heads
        if heartbeat is not None:
            heartbeat.beat(phase="priors", route=route, distilled=record["distilled"])
        return {"student": load, "distillation": record, "lineage": start,
                "development_spearman": {"before": before, "after": after,
                                         "delta": (None if before is None or after is None
                                                   else after - before)},
                "rescoring": {"weight": rescoring["weight"], "spearman": rescored,
                              "delta": (None if before is None or rescored is None
                                        else rescored - before)},
                "checkpoint": str(directory / "distilled.pt"),
                "separation": ("distillation moved the student's weights; rescoring did not. Both "
                               "are reported, because one number for 'the teacher helped' cannot "
                               "say which mechanism produced it.")}

    # -- T4 post-training --------------------------------------------------
    def pair_views(self, route, pairs, *, condition=None):
        """Chosen-then-rejected rows for a pair batch, masked at the differing site.

        Adjacent rows, chosen first: the convention the T4 losses read, and the one that
        makes ``left[0::2]`` the chosen half. Both members are masked at the union of the
        changed coordinates, so the pair's two inputs are byte-identical.

        Each pair is scored under the condition its OWN metadata names -- its family and
        its assay, at the high bin -- through ``her2_routes_pairs.pair_condition``.
        ``condition`` may be an explicit :class:`Condition`, accepted only if it agrees
        with every pair's family and assay; a pair with no assay or family, an explicit
        condition for another assay, or an unsupported named mode raises. There is no HER2
        default and no fallback to the dropped-assay ``null`` slot: a pair whose assay is
        not enrolled fails in ``condition_spec.encode`` at collation, loudly.
        """
        from . import her2_routes_pairs as pairs_lib
        from .her2_routes_serial import ablang_serialization, causal_serialization
        views = []
        for pair in pairs:
            resolved = pairs_lib.pair_condition(pair, condition)
            sites = tuple(pair["sites"])
            for row in (pair["chosen"], pair["rejected"]):
                construct = data_lib.with_sites(row.construct, sites)
                serialization = (
                    causal_serialization(construct, order=sites,
                                         residues=construct.residues(sites),
                                         max_positions=self._positions(route))
                    if route == ROUTE_CAUSAL
                    else ablang_serialization(construct, masked_sites=sites, order=sites))
                views.append((construct, serialization, resolved))
        return tuple(views)

    def run_posttrain(self, *, heartbeat=None):
        """Run the declared methods as real optimisations from one selected start.

        Each arm starts from the **same** selected checkpoint, holds its **own**
        frozen reference, draws from its **own** pair bank under the declared reuse
        cap, takes real optimizer steps, and is scored on development before and
        after. An earlier revision drew pairs, returned the design metadata and
        performed no backward pass at all, so the stage recorded four methods and
        ran none.

        **Contract version 2** (``posttrain.contract_version``). The contract, the method
        list and the pair pool are validated before any checkpoint is read or model built,
        so a stale configuration fails in milliseconds. Pairs carry their own assay,
        target, family and scale and are scored under their own condition; the pool is
        pooled across assays unless ``posttrain.population`` names an explicit filter, and
        every arm is reported per assay. Every arm draws from the **same** pair pool with the
        same draw seeds and reuse cap, which the record proves with a pool digest and a
        per-arm draw-schedule digest. Those digests are structural; the measured values and
        winners the arms were trained against have their own (``pair_targets_sha256`` and a
        per-arm target schedule), and the checkpoint identity's ``schedule_sha256`` binds
        pool, targets and both schedules, so changed training targets are a different
        identity. Pairs are drawn background-first (``PairBank``). The policy is scored in eval
        mode with gradients ON (deterministic, differentiable); the reference is frozen, eval
        and ``no_grad``. The design metadata quotes the configured ``winner_likelihood``, the
        value the preference arms execute.

        Conditional supervised fitting is the first arm because the measured failure
        is quantitative: the completed campaign's preference arms moved display
        ranking and left affinity ordering untouched, and that table is carried in
        ``her2_routes_posttrain.PRIOR_EVIDENCE`` so this stage cannot forget it.
        """
        from . import her2_routes_eval as eval_lib
        from . import her2_routes_pairs as pairs_lib
        from . import her2_routes_posttrain as posttrain
        block = self.config["posttrain"]
        version = posttrain.require_contract(block)
        methods = posttrain.resolve_methods(block["methods"])
        gate = read_stage_record_safe(self.context, "route_gate")
        route = (gate or {}).get("winner") or self.config["routes"]["enabled"][0]
        # Every post-training arm scores a token likelihood, so a route without one is refused here:
        # after the pure contract validation, before a checkpoint, a pair bank or a model.
        capabilities.require_token_likelihood(route, "FlightServices.run_posttrain")
        seed = int(self.config["routes"][route]["seed"])
        start = self.checkpoint_lineage("posttrain", route)
        selected = Path(start["path"])
        pairs = self.measured_pairs()
        posttrain.require_measured_pairs(pairs)
        pairs_lib.require_pair_metadata(pairs)
        pool_digest = pairs_lib.pair_pool_digest(pairs)
        targets_digest = pairs_lib.pair_targets_digest(pairs)
        # The configured coefficient, read once: the preference loss applies it and the design
        # metadata quotes it, so the record cannot state one value while another was executed.
        winner_likelihood = float(block.get("winner_likelihood", 0.0))
        pool_record = dict(self._cache.get("pair_inventory") or {})
        panel = self.development_panel_outcomes()
        outcomes = [row["value"] for row in panel]
        results, reference_digests, schedules, target_schedules = {}, {}, {}, {}
        for method in methods:
            model, heads, load = self.load_for_scoring(route, selected, seed=seed)
            # Each arm gets an INDEPENDENT initial state and an independent bank over
            # the SAME pool with the SAME draw seeds and reuse cap. A shared bank means
            # the fourth arm trains on whatever the first three left, and the comparison
            # is then between four different datasets; matched seeds are what make the
            # arms differ in their objective alone.
            bank = posttrain.PairBank(pairs=tuple(pairs), reuse_cap=int(block["reuse_cap"]))
            reference = posttrain.freeze_reference(model)
            reference_digests[method] = reference["sha256"]
            if route in DPLM_ROUTES:
                # one base load: the reference shares the frozen LoRA base storage and carries
                # its own adapters/conditions (a full-tune route deep-copies, so no saving there)
                frozen = model.frozen_reference()
            else:
                frozen = self.load_for_scoring(route, selected, seed=seed)[0]
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
            frozen.eval()
            posttrain.require_deterministic_scoring(frozen, label=f"{method} reference",
                                                    gradients=False)
            before = eval_lib.spearman(self.score_panel(route, None, panel, model=model,
                                                        heads=heads), outcomes)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = self.build_optimizer(
                route, parameters, lr=float(block.get("learning_rate", 1e-6)))
            directory = self.context.path("posttrain", method)
            directory.mkdir(parents=True, exist_ok=True)
            progress = paths.StageProgress(directory / "progress.json",
                                           stage=f"posttrain::{method}",
                                           total=int(block["updates"]), every=10)
            progress.start()
            history, draws, exhausted = [], [], None
            drawn_identities, drawn_targets = [], []
            exposure = posttrain.AssayExposure()
            batch_size = max(1, int(block.get("pairs_per_batch", 8)))
            kappa = float(self.config["curriculum"]["kappa"])
            # The policy is scored in EVAL mode with gradients ON, not train mode: the two
            # halves of a pair are separate forwards of one masked input, and dropout would
            # make them two different conditionals. Every module's training flag is restored
            # afterwards. The optimizer still steps; parameter_movement below is the proof.
            with posttrain.deterministic_scoring(model):
                posttrain.require_deterministic_scoring(model, label=f"{method} policy")
                for step in range(1, int(block["updates"]) + 1):
                    try:
                        drawn, draw_record = bank.draw(batch_size, seed=int(block["seed"]) + step)
                    except ValueError as error:
                        # Every measured pair has reached the reuse cap. That is a real end
                        # of the arm's data, not a crash: generating more candidates does
                        # not create more measurements, so the arm stops here and the step
                        # it stopped at is recorded.
                        exhausted = str(error)
                        break
                    if not drawn:
                        exhausted = "the bank returned no pair under the reuse cap"
                        break
                    draws.append(draw_record)
                    drawn_identities.append([pair["pair_digest"] for pair in drawn])
                    drawn_targets.append([pair["target_digest"] for pair in drawn])
                    views = self.pair_views(route, drawn)
                    batch = self.collate(route, views)
                    log_probs = self.canonical_log_probs(route, model, batch)
                    targets = torch.as_tensor(batch.targets, dtype=torch.long,
                                              device=log_probs.device)
                    mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
                    posttrain.require_one_site_rows(mask, label=f"{method} batch")
                    picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                    window = (picked * mask.to(picked.dtype)).sum(dim=1)
                    chosen, rejected = window[0::2], window[1::2]
                    margins = torch.as_tensor([float(pair["margin"]) for pair in drawn],
                                              dtype=window.dtype, device=window.device)
                    assays = [pair["assay"] for pair in drawn]
                    if method == posttrain.METHOD_WEIGHTED_CONDITIONAL_NLL:
                        loss, diagnostics = posttrain.weighted_conditional_nll_loss(
                            log_probs, targets, mask, deltas=margins, kappa=kappa,
                            assays=assays)
                    elif method == posttrain.METHOD_MARGIN_MATCH:
                        loss, diagnostics = posttrain.margin_match_loss(
                            chosen, rejected, margins, kappa=kappa)
                    else:
                        with torch.no_grad():
                            reference_log = self.canonical_log_probs(route, frozen, batch)
                            reference_picked = reference_log.gather(
                                2, targets.unsqueeze(-1)).squeeze(-1)
                            reference_window = (reference_picked
                                                * mask.to(reference_picked.dtype)).sum(dim=1)
                        coefficients = {"tau": float(block.get("tau", posttrain.IPO_TAU)),
                                        "beta": float(block.get("beta", 0.1)),
                                        "window": len(drawn[0]["sites"])}
                        loss, diagnostics = posttrain.preference_loss(
                            method, policy_chosen=chosen, policy_rejected=rejected,
                            reference_chosen=reference_window[0::2].detach(),
                            reference_rejected=reference_window[1::2].detach(),
                            coefficients=coefficients,
                            winner_likelihood=winner_likelihood)
                    exposure.add(assays, masses=posttrain.alternative_mass(
                        chosen.detach(), rejected.detach()).tolist())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, float(self.config["optimization"]["gradient_clip"]),
                        error_if_nonfinite=True)
                    optimizer.step()
                    history.append({"step": step, "loss": float(loss.detach()),
                                    "gradient_norm": float(norm), "pairs": len(drawn),
                                    **{key: value for key, value in dict(diagnostics).items()
                                       if isinstance(value, (int, float, str, bool))}})
                    progress.advance(f"{method} {step}", completed=step)
                    if heartbeat is not None:
                        heartbeat.beat(phase="posttrain", method=method, step=step,
                                       loss=float(loss.detach()))
            progress.finish()
            model.eval()
            schedules[method] = posttrain.schedule_digest(drawn_identities)
            target_schedules[method] = posttrain.schedule_digest(drawn_targets)
            training_targets = posttrain.training_targets_digest(
                pair_pool_sha256=pool_digest, pair_targets_sha256=targets_digest,
                draw_schedule_sha256=schedules[method],
                target_schedule_sha256=target_schedules[method])
            after = eval_lib.spearman(self.score_panel(route, None, panel, model=model,
                                                       heads=heads), outcomes)
            state = self.fresh_state(seed)
            state.step = len(history)
            # The arm string carries the contract version, so weights trained under the
            # version-1 objectives cannot be loaded or resumed as version 2 and the reverse.
            # The identity's schedule field carries the training-targets digest -- pool, measured
            # targets, and both draw schedules -- so a changed measured value, winner or
            # provenance is a different identity even when every structural pair is unchanged.
            checkpoints.save(directory / "last.pt", model=model, heads=heads,
                             optimizer=optimizer, scheduler=None, state=state,
                             identity=self.training_identity(
                                 route, arm=posttrain.arm_identity(method), seed=seed,
                                 schedule_sha256=training_targets),
                             extra={"posttrain": {"contract_version": version, "method": method,
                                                  "pairs_version": pairs_lib.PAIRS_VERSION,
                                                  "pair_pool_sha256": pool_digest,
                                                  "pair_targets_sha256": targets_digest,
                                                  "draw_schedule_sha256": schedules[method],
                                                  "target_schedule_sha256":
                                                      target_schedules[method],
                                                  "training_targets_sha256": training_targets,
                                                  "winner_likelihood": winner_likelihood}})
            moved = posttrain.parameter_movement(reference["state"], model.state_dict())
            results[method] = {
                "contract_version": version, "objective": posttrain.OBJECTIVES[method],
                "updates": len(history), "history": history,
                "requested_updates": int(block["updates"]),
                "stopped_early": exhausted,
                "pairs_drawn": sum(record["drawn"] for record in draws),
                "draw_records": draws[:5], "pair_bank": bank.document(),
                "pair_pool_sha256": pool_digest,
                "pair_targets_sha256": targets_digest,
                "draw_schedule_sha256": schedules[method],
                "target_schedule_sha256": target_schedules[method],
                "training_targets_sha256": training_targets,
                "assay_exposure": exposure.document(),
                "scoring": "eval mode, gradients on; reference frozen eval under no_grad",
                "reference_sha256": reference["sha256"],
                "development_spearman": {"before": before, "after": after,
                                         "delta": (None if before is None or after is None
                                                   else after - before)},
                "parameter_movement": moved,
                "start": load,
                "design": posttrain.design(methods=[method],
                                           reuse_cap=int(block["reuse_cap"]),
                                           winner_likelihood=winner_likelihood)}
            del model, heads, frozen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ranked = sorted((name for name, block_ in results.items()
                         if block_["development_spearman"]["after"] is not None),
                        key=lambda name: -results[name]["development_spearman"]["after"])
        best_method = ranked[0] if ranked else None
        return {"methods": results, "route": route, "contract_version": version,
                "reference_sha256": reference_digests,
                "pairs": {"version": pairs_lib.PAIRS_VERSION, "pool_sha256": pool_digest,
                          "targets_sha256": targets_digest, "pool": pool_record},
                "matched_draws": {"draw_schedule_sha256": dict(schedules),
                                  "target_schedule_sha256": dict(target_schedules),
                                  "identical_across_arms": (len(set(schedules.values())) <= 1
                                                            and len(set(target_schedules.values()))
                                                            <= 1)},
                "selected_start": str(selected), "lineage": start,
                "selection": {"ranked": ranked, "best": best_method,
                              "criterion": "development Spearman after the arm's own updates"},
                "selected_method": best_method,
                "selected_checkpoint": (None if best_method is None else
                                        str(self.context.path("posttrain", best_method,
                                                              "last.pt"))),
                "independence": ("each arm starts from the same selected checkpoint with its OWN "
                                 "frozen reference and its OWN pair bank under the declared reuse "
                                 "cap. A shared bank would leave the last arm training on what the "
                                 "first three did not use."),
                "prior_evidence": posttrain.PRIOR_EVIDENCE}

    def measured_pairs(self, *, population=None, allow_empty=False):
        """Qualified one-site measured pairs, each carrying its own assay, target and family.

        Built from the bounded contrast enumeration over construct **summaries** (one
        differing coordinate, so a pair is a preference about that coordinate) and
        qualified by ``her2_routes_pairs.build_measured_pairs``: exact finite uncensored
        pKD replicates on both endpoints (``summary.kind`` is not consulted, so Adams'
        interval-kind points survive and a mixed-censored construct does not), both contrast
        classes, ties counted, both endpoints checked against the closure at its radius --
        excluding only the chosen one would leave a reserved neighbour in the pool as the
        rejected half of every pair it appears in -- and an assay that is not enrolled in
        the condition vocabulary refused rather than relabelled ``null``.

        The pool is **pooled across assays**. ``population`` (or, if omitted,
        ``posttrain.population``) restricts it to an explicit ``{assay, target, family}``
        filter; there is no implicit HER2 restriction, and what fell outside a filter is
        counted. ``allow_empty`` returns ``[]`` for an empty selection instead of raising,
        for the evaluation panel that reports its own unsupported reason.

        Both members' measured values travel with the pair, because the likelihood and
        log-odds arms fit the decoder to the measured difference and cannot do that from a
        sign. Each pair also carries a ``target_digest`` over those values, the winner and the
        measurement identities; a structural pair measured two conflicting ways raises here
        instead of keeping the first one enumerated. The qualified pool is cached; its counts
        are in ``_cache['pair_inventory']``.
        """
        from . import her2_routes_data as data
        from . import her2_routes_pairs as pairs_lib
        from .her2_routes_conditions import ASSAY_NULL
        pool = self._cache.get("pair_pool")
        if pool is None:
            block = self.config["curriculum"]
            record = data.build_local_contrasts(
                [row for row in self.eligible_summaries() if row.is_training], max_sites=1,
                max_per_background=int(block.get("max_contrasts_per_background", 64)),
                max_per_context=int(block.get("max_contrasts_per_context", 32)),
                parents=self.parent_sequences())
            closure = self.closure()

            def violates(row):
                window = row.construct.residues(row.construct.sites)
                return closure.violates(window=window, identity=row.identity) is not None

            pool = pairs_lib.build_measured_pairs(
                record["contrasts"], violates=violates,
                enrolled_assays=[name for name in self.condition_spec.assay_order
                                 if name != ASSAY_NULL])
            pool["contrast_inventory"] = {key: value for key, value in record.items()
                                          if key != "contrasts"}
            self._cache["pair_pool"] = pool
        requested = (population if population is not None
                     else dict(self.config.get("posttrain") or {}).get("population"))
        selected = pairs_lib.select_population(pool["pairs"], requested)
        self._cache["pair_inventory"] = {
            "version": pairs_lib.PAIRS_VERSION, "population": selected["population"],
            "available": selected["available"], "selected": selected["selected"],
            "outside_population": selected["outside_population"],
            "counts": pool["counts"], "contrast_inventory": pool["contrast_inventory"]}
        if not selected["pairs"]:
            require(allow_empty,
                    "No eligible measured preference pair survived qualification, the closure "
                    f"and the population filter {selected['population']}. Refusals: "
                    f"{pool['counts']['excluded']}. Both endpoints must be exact finite pKD "
                    "training rows, clear the radius, and carry an enrolled assay.")
            return []
        return list(selected["pairs"])

    # -- T5 decoding -------------------------------------------------------
    def adaptation_bank(self):
        """The ELIGIBLE measured adaptation windows. Not the development panel.

        The earlier decoding stage used ``development_panel_outcomes`` as both the
        training medoid source and the nearest-training bank, so "distance to
        training" was distance to the held-out panel and the frozen medoids were the
        panel's own members. Both are the opposite of what the names say.

        This is the ADAPTATION POPULATION: every HER2 training sequence the model may adapt on,
        enumerated independently of whether its value is an exact scalar (a censored or
        mixed-censored construct still trains the head and whole-window tasks, so it is
        geometry the model has seen), plus the closure-filtered display cores. It is a set of
        sequences, not of labels, and it is not the realised exposure of any arm. The exact
        quantitative bank is :meth:`her2_training_bank`. The sorted tuple is cached with a record
        of its count, digest and semantics (:meth:`adaptation_bank_record`).
        """
        if "adaptation_bank" in self._cache:
            return self._cache["adaptation_bank"]
        eligible, counts = her2_training_population(self.summaries(), self.closure())
        measured = {window for window, _row in eligible}
        display = self.display_pool()
        displayed = {str(core) for core in display.cores} if display is not None else set()
        windows = tuple(sorted(measured | displayed))
        require(windows,
                "The adaptation bank is empty: no HER2 training row and no display core survived "
                f"the split and the closure ({counts}).")
        self._cache["adaptation_bank_record"] = {
            "kind": "eligible_adaptation_bank", "windows": len(windows),
            "sha256": _digest_lines(windows),
            "measured_her2_windows": len(measured),
            "display_windows": len(displayed),
            "display_only_windows": len(displayed - measured),
            "measured_without_exact_value": counts["eligible"] - counts["exact"],
            "counts": counts,
            "semantics": ("sequence geometry only: HER2 TRAINING windows (labelled or not, "
                          "closure-filtered) plus closure-filtered display cores. It carries no "
                          "labels, so a display class can never become an affinity, and it is the "
                          "ELIGIBLE adaptation population, not the realised exposure of any arm.")}
        self._cache["adaptation_bank"] = windows
        return windows

    def adaptation_bank_record(self):
        """Count, digest and source semantics of :meth:`adaptation_bank`."""
        self.adaptation_bank()
        return dict(self._cache["adaptation_bank_record"])

    def run_decoding(self, *, heartbeat=None):
        """Compare the fill strategies on the same weights at the declared draw counts.

        Real batched generation: the strategies differ in the **order** each draw is
        filled in, and ``easy_first``/``hard_first`` read the model's own conditional
        entropy at every step. The earlier implementation passed ``rng.random()`` as
        the confidence, which made the two a single random order under two labels.

        The clustering is frozen once from the pooled candidates of every arm at the
        MAXIMUM draw count, then reused for every arm and every nested prefix, and the
        nearest-training bank is the eligible adaptation bank rather than the
        development panel.
        """
        from . import her2_routes_decoding as decoding
        from . import her2_routes_eval as eval_lib
        from . import her2_routes_generate as generate
        from .her2_routes_conditions import BIN_HIGH, Condition
        block = self.config["decoding"]
        gate = read_stage_record_safe(self.context, "route_gate")
        route = (gate or {}).get("winner") or self.config["routes"]["enabled"][0]
        sites = her2_core_edit_map()
        condition = Condition(family="affinity", bin=BIN_HIGH,
                              assay=self.config["her2"]["assay"])
        draws = int(block["draws"])
        prefixes = [int(value) for value in block["draw_prefixes"]]
        require(max(prefixes) <= draws,
                f"The declared prefixes {prefixes} ask for more draws than the {draws} this stage "
                "generates. The shorter count is the measured one and is not padded.")
        bank = self.adaptation_bank()
        assessor = self.independent_assessor()
        source = self.checkpoint_lineage("decoding", route)
        checkpoint = Path(source["path"])
        model, heads, load = self.load_for_scoring(route, checkpoint)
        batch_size = max(1, int(self.selected_shapes()["microbatch"]) * 4)
        per_strategy, per_strategy_skipped = {}, {}
        if route == ROUTE_ENERGY:
            # The mode is the loaded checkpoint's validated runtime contract; context rows are
            # read only for a model that was trained with a context.
            runtime = energy_contract.require_runtime(model, where="run_decoding")
            rows = list(self.energy_rows(her2_only=True)) if runtime.conditioned else []
            result = generate.energy_generate(
                self, model, draws=draws, context_rows=rows, seed=int(block["seed"]),
                batch_size=int(self.config["routes"][ROUTE_ENERGY]["microbatch"]),
                sampling_options={"steps": int(
                    self.config["routes"][ROUTE_ENERGY]["walk_steps"])},
                mode=runtime.mode)
            per_strategy["walk_jump"] = result
        else:
            for strategy in block["strategies"]:
                if strategy == generate.ORDER_DIFFUSION and not self.supports_diffusion_decoding(route):
                    # Skipped, and recorded as skipped: the strategy is defined for the
                    # diffusion routes only and the AR and AbLang routes have no such decoder.
                    per_strategy_skipped[strategy] = (
                        f"{route} has no diffusion decoder; the strategy is defined for the "
                        "native DPLM routes")
                    continue
                result = generate.fill_batch(
                    self, route, model, draws=draws, sites=sites, condition=condition,
                    strategy=strategy, seed=int(block["seed"]), batch_size=batch_size,
                    remask_rounds=(int(block.get("remask_rounds", 0))
                                   if strategy == generate.ORDER_REMASK else 0),
                    remask_fraction=float(block.get("remask_fraction", 0.3)))
                per_strategy[strategy] = result
                if heartbeat is not None:
                    heartbeat.beat(phase="decoding", strategy=strategy,
                                   draws=len(result.windows))
        del model, heads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clustering = eval_lib.freeze_clustering(
            {name: result.windows for name, result in per_strategy.items()},
            radius=int(block["cluster_radius"]),
            separation=int(block["cluster_separation"]),
            assigned=eval_lib.training_medoids(
                bank, separation=int(block["cluster_separation"])))
        reports = {}
        for name, result in per_strategy.items():
            assessed = assessor["score"](sorted(set(result.windows)))
            reports[name] = {
                "accounting": result.accounting(),
                "draw_record": result.document(),
                "generation": eval_lib.generation_report(
                    result.windows, training=bank, catalogue=bank, clustering=clustering,
                    wild_type=self.config["her2"]["wild_type_core"],
                    draw_prefixes=prefixes),
                "independent_quality": {
                    "unique": len(assessed),
                    "mean": float(np.mean(assessed)) if assessed else None,
                    "assessor": assessor["fit"],
                    "independence": assessor["independence"]}}
        ranked = sorted((name for name, record in reports.items()
                         if record["independent_quality"]["mean"] is not None),
                        key=lambda name: -reports[name]["independent_quality"]["mean"])
        allocation = decoding.two_site_allocation_example()
        return {"route": route, "checkpoint": load, "lineage": source,
                "strategies": reports, "draw_prefixes": prefixes,
                "strategies_skipped": per_strategy_skipped,
                "clustering": clustering.document(),
                "bank": {"windows": len(bank),
                         "source": ("the eligible measured adaptation bank: training partition, "
                                    "closure-filtered. NOT the development panel.")},
                "ranked_by_independent_quality": ranked,
                "functionals": decoding.FUNCTIONALS,
                "allocation_example": allocation,
                "planner": self.planner_comparison(route, checkpoint),
                "reporting_functional": ("the fixed declared-order joint at temperature 1. A "
                                         "remasking trajectory has no closed-form density and its "
                                         "reported log probability is the fixed-order density of "
                                         "the final window, labelled as such.")}

    def planner_comparison(self, route, checkpoint, *, draws=None):
        """Planner-weighted reconstruction against unweighted, at matched exposure.

        The I19 identity is the reason this is a comparison rather than a claim: at
        planner weight zero the weighted loss and its gradients reproduce ordinary
        training exactly, so a nonzero weight's effect is measurable against that
        exact baseline on the same batch.
        """
        if route == ROUTE_ENERGY:
            return {"supported": False,
                    "reason": ("the energy route has no per-target conditional to weight. The "
                               "planner is defined for the two likelihood routes.")}
        weights = [float(value) for value in
                   (self.config["decoding"].get("planner_weights") or (0.0, 0.5))]
        panel = self.development_views(route)
        model, heads, _load = self.load_for_scoring(route, checkpoint)
        rows = []
        with torch.no_grad():
            batch = self.collate(route, panel[:min(len(panel), int(draws or 16))])
            log_probs = self.canonical_log_probs(route, model, batch)
            targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
            mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
            picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            for weight in weights:
                derived, record = objectives_lib.planner_weights(
                    log_probs, mask, weight=weight, mode="easy_first")
                active = torch.clamp(derived.sum(), min=1e-12)
                loss = -(picked * derived).sum() / active
                rows.append({"weight": weight, "loss": float(loss),
                             "identical_to_ordinary_training":
                                 record["identical_to_ordinary_training"]})
        del model, heads
        return {"supported": True, "rows": rows,
                "zero_weight_identity": ("at weight 0 the derived weights are exactly the mask, so "
                                         "the weighted loss reproduces ordinary training. Any "
                                         "difference at weight 0 is a defect in the weighting, "
                                         "not an effect of planning.")}

    # -- T6 confirmation ---------------------------------------------------
    def frozen_recipe(self):
        """The recipe the confirmation seeds train. Assembled from the selections made.

        This is what "three seeds of the FROZEN recipe" has to mean: the route the
        gate chose, the control arm that survived, the teacher decision the priors
        stage reached, the post-training method its own development comparison
        selected, the decoding strategy the decoding stage ranked first, and the
        best-step rule. An earlier revision retrained the basic curriculum and
        ignored all six, so the confirmation confirmed a specification that no
        selection had produced.
        """
        gate = read_stage_record_safe(self.context, "route_gate") or {}
        priors = read_stage_record_safe(self.context, "priors") or {}
        posttrain = read_stage_record_safe(self.context, "posttrain") or {}
        decoding = read_stage_record_safe(self.context, "decoding") or {}
        controls_record = read_stage_record_safe(self.context, "controls") or {}
        route = gate.get("winner") or self.config["routes"]["enabled"][0]
        posttrain_file = self.context.path("posttrain.json")
        decoding_file = self.context.path("decoding.json")
        posttrain_block = paths.read_json(posttrain_file) if posttrain_file.is_file() else {}
        decoding_block = paths.read_json(decoding_file) if decoding_file.is_file() else {}
        # A stage record that says completed needs its document and selection: a missing
        # file is never read as "nothing was selected". Named values are kept verbatim so an
        # unsupported method/strategy still reaches require_replayable and is refused there.
        if posttrain.get("status") == "completed":
            lineage.completed_document(posttrain_block or None, stage="confirmation_freeze",
                                       name="posttrain", route=route)
            lineage.selected_method(posttrain_block, posttrain)
        if decoding.get("status") == "completed":
            lineage.completed_document(decoding_block or None, stage="confirmation_freeze",
                                       name="decoding", route=route)
            if not (decoding_block.get("ranked_by_independent_quality") or []):
                raise lineage.LineageError(
                    "decoding.json ranks no strategy although the decoding stage completed; the "
                    "recipe's decoding strategy is not guessed.")
        # HR-01: the per-teacher decisions are in priors.json (a mapping). The priors STAGE
        # RECORD's ``teachers`` is a list of names and cannot be read as one.
        priors_document = self.run_document("priors") if priors else None
        if priors and priors_document is None:
            raise lineage.LineageError(
                "The priors stage record exists but priors.json (the per-teacher decisions) is "
                "absent, so the recipe's teacher decisions cannot be read. They are not "
                "guessed from the record's list of names.")
        teachers = ({} if priors_document is None
                    else lineage.teacher_decisions(priors_document))
        recipe = {
            "route": route,
            "arm": "full",
            "updates": int(self.config["confirmation"]["updates"]),
            "task_settings": self.task_settings(),
            "shapes": self.selected_shapes(),
            "curriculum": {"shares": dict(self.config["curriculum"]["shares"]),
                           "family_mix": dict(self.config["curriculum"]["family_mix"]),
                           "weights": dict(self.config["curriculum"]["weights"])},
            "teachers": teachers,
            "posttrain_method": (posttrain_block.get("selection") or {}).get("best"),
            "decoding_strategy": (decoding_block.get("ranked_by_independent_quality")
                                  or [None])[0],
            "best_step_rule": ("the best development metric with the declared plateau delta; "
                               "checkpoint best.pt"),
            "controls_survived": bool(controls_record.get("survives")),
            "selections_read": {"route_gate": bool(gate), "controls": bool(controls_record),
                                "priors": bool(priors), "posttrain": bool(posttrain),
                                "decoding": bool(decoding)}}
        recipe["sha256"] = paths.digest_document(
            {key: value for key, value in recipe.items() if key != "sha256"})
        return recipe

    def run_confirmation(self, *, heartbeat=None):
        """Three seeds of the frozen recipe, then the single reserved opening."""
        from . import her2_routes_eval as eval_lib
        block = self.config["confirmation"]
        # The recipe the confirmation_freeze stage froze is CONSUMED and identity-checked; it
        # is never re-assembled from the current stage records. And it is refused, before
        # any training or opening, when it names something this training would not apply.
        recipe = lineage.consume_frozen_recipe(
            self.context.path("frozen_recipe.json"),
            freeze_record=read_stage_record_safe(self.context, "confirmation_freeze"))
        plan = lineage.require_replayable(recipe, where="confirm")
        route = recipe["route"]
        # Three seeds of the frozen recipe are three runs of the token-likelihood loop; a route
        # with none is refused before the first seed's directory, stream or model exists.
        capabilities.require_token_likelihood(route, "FlightServices.run_confirmation")
        applied = {"route": route, "base_curriculum": True, "teachers": [],
                   "posttrain_method": None, "decoding_strategy": None,
                   "replay_plan": plan,
                   "note": ("what the training actually applied. It equals the recipe only "
                            "because the recipe named nothing beyond the route and the base "
                            "curriculum; anything else was refused above.")}
        seeds, records = [int(value) for value in block["seeds"]], {}
        require(len(seeds) >= 3,
                f"The confirmation declares {len(seeds)} seed(s). Three is the declared minimum, "
                "and seed variation cannot be separated from sampling uncertainty below it.")
        for seed in seeds:
            directory = self.context.path("confirm", f"seed{seed}")
            records[seed] = self.train(
                route, stream=self.curriculum_stream(route, seed=seed,
                                                     updates=int(block["updates"])),
                updates=int(block["updates"]), seed=seed, directory=directory,
                heartbeat=heartbeat, evaluate=self.development_metric(route))
        complete = all(records[seed]["updates"] >= int(block["updates"]) for seed in seeds)
        split = self.split()
        n = len(split.reserved)
        if not complete:
            return {"seeds": {str(seed): records[seed]["updates"] for seed in seeds},
                    "complete": False, "reserved_opened": False, "reserved_n": n,
                    "recipe": recipe, "applied": applied,
                    "status": ("RUNNING. The confirmation seeds have not finished, so the reserved "
                               "test stays closed and nothing is reported as complete."),
                    "power": eval_lib.panel_record("E6", population=n, supported=False,
                                                   reason="the confirmation seeds are still "
                                                          "running and the panel is unopened")}
        if n < int(self.config["split"]["minimum_reserved"]):
            return {"seeds": {str(seed): records[seed]["updates"] for seed in seeds},
                    "complete": True, "reserved_opened": False, "reserved_n": n,
                    "recipe": recipe, "applied": applied,
                    "status": (f"G4 does not hold: {n} reserved cores against a declared minimum "
                               f"of {self.config['split']['minimum_reserved']}. The endpoint is "
                               "reported as underpowered with its actual n and the arithmetic."),
                    "power": eval_lib.panel_record(
                        "E6", population=n, supported=False,
                        reason=f"underpowered at n={n}; see split_certificate.json")}
        opened = self.open_reserved(route, seeds=seeds, recipe=recipe)
        return {"seeds": {str(seed): records[seed]["updates"] for seed in seeds},
                "complete": True, "reserved_opened": True, "reserved_n": n,
                "recipe": recipe, "applied": applied, "reserved": opened,
                "status": "the reserved test was opened once, after every gate held"}

    def reserved_guard_path(self):
        return self.context.path("reserved_opening.json")

    def mint_reserved_unlock(self, recipe):
        """Persist a once-only guard, then mint the single-use unlock. In that order.

        The order is the guarantee. A guard written after the read would let a crash
        between the two leave the panel openable again, and "opened once" has to be a
        property of the filesystem rather than of whoever ran the stage.
        """
        from . import her2_routes_sources as sources
        guard = self.reserved_guard_path()
        require(not guard.exists(),
                f"{guard} already exists: the reserved panel has been opened. Its record names "
                "when and under which frozen selection. A second opening is a second look at the "
                "same held-out data and there is no flag that permits it.")
        record = {"schema_version": spec.ROUTES_SCHEMA, "record_kind": "reserved_opening",
                  "frozen_selection_sha256": recipe["sha256"],
                  "recipe": dict(recipe), "opened_at": paths.utc_now(),
                  "owner": dict(controls_owner()),
                  "rule": ("written BEFORE the first reserved outcome is read. A guard written "
                           "afterwards leaves a crash between the two with the panel still "
                           "openable.")}
        paths.write_json(guard, record)
        return sources.ReservedUnlock(
            reason="the single declared opening of the frozen reserved panel",
            frozen_selection_sha256=recipe["sha256"],
            minted_by=str(dict(controls_owner()).get("host", "unknown")))

    def reserved_panel(self, *, unlock):
        """The reserved windows with their outcomes, read through the single-use unlock.

        The outcomes are parsed here and nowhere else: the cached observations were
        loaded with the reserved windows gated OUT, so a reserved panel taken from
        that cache is empty. Spending the unlock on a fresh, unrestricted read is what
        makes the opening a real one, and the unlock refuses a second spend.
        """
        from . import her2_routes_sources as sources
        split = self.split()
        target = self.source_paths()["alphabind_her2_validation"]
        block = sources.load_alphaseq_observations(
            target, source="alphabind_her2_validation", scaffold=self.scaffold(),
            sites=her2_core_edit_map(), allowed_windows=None, reserved_unlock=unlock)
        reserved = set(split.reserved)
        # Every replicate of a reserved window is summarised first, and the construct-level value
        # is then qualified: a raw ``Observation`` carries one replicate and no ``exact_value``,
        # and the mean of whichever replicates survived is not a measurement. A summary that is
        # not an exact point is excluded and counted; membership is the frozen reserved set's.
        records = [record for record in block["observations"]
                   if record.construct.residues(record.construct.sites) in reserved]
        summaries = data_lib.summarise_observations(records)
        by_window, excluded = {}, {}
        for summary in summaries:
            exact = summary.exact_value
            if exact is None:
                excluded[summary.exact_status] = excluded.get(summary.exact_status, 0) + 1
                continue
            window = summary.construct.residues(summary.construct.sites)
            by_window.setdefault(window, []).append(float(exact))
        panel = [{"window": window, "value": float(np.mean(by_window[window]))}
                 for window in split.reserved if window in by_window]
        require(panel,
                "No reserved window carries an exact outcome in the unrestricted read. The "
                "panel was frozen from the same file's sequences, so an empty intersection means "
                f"the file changed or every summary was excluded ({excluded}).")
        return panel, {"unlock": block["reserved_unlock"],
                       "window_gate": block["window_gate"],
                       "rows_in_file": block["rows"],
                       "reserved_observations": len(records),
                       "reserved_summaries": len(summaries),
                       "summaries_excluded_by_exact_status": dict(sorted(excluded.items())),
                       "reserved_windows_with_outcomes": len(panel)}

    def open_reserved(self, route, *, seeds, recipe):
        """Score the reserved panel once. Paired delta-Spearman against the control.

        Three corrections over the earlier revision, all of them load-bearing:

        * the panel comes from a **fresh unrestricted read** through the single-use
          unlock. The cached observations were loaded with the reserved windows gated
          out, so the earlier reader found nothing and the endpoint was computed over
          an empty panel;
        * the guard is persisted **before** the read, so "opened once" is a property
          of the filesystem;
        * the statistic is the bootstrapped difference of **Spearman** correlations
          over sequence neighbourhoods, not the bootstrapped mean of raw log-score
          differences. An arbitrary additive offset in one arm's scores moves the
          latter and moves no ordering.
        """
        from . import her2_routes_eval as eval_lib
        # Every seed's checkpoint is required BEFORE the once-only guard is written: a
        # missing file discovered afterwards would leave the panel "opened" with nothing scored.
        selected = {seed: self.require_checkpoint(
            self.context.path("confirm", f"seed{seed}", "best.pt"), stage="confirm",
            route=route, role=f"confirmation seed {seed} best checkpoint") for seed in seeds}
        unlock = self.mint_reserved_unlock(recipe)
        panel, read_record = self.reserved_panel(unlock=unlock)
        outcomes = [row["value"] for row in panel]
        windows = [row["window"] for row in panel]
        groups = self.neighbourhood_labels(windows)
        # ``None`` here is the declared frozen zero-shot comparator, passed explicitly.
        control = self.score_panel(route, None, panel)
        per_seed, per_seed_scores = {}, {}
        for seed in seeds:
            scores = self.score_panel(route, selected[seed], panel, seed=seed)
            per_seed_scores[str(seed)] = scores
            per_seed[str(seed)] = eval_lib.spearman(scores, outcomes)
        averaged = [float(np.mean(column)) for column in zip(*per_seed_scores.values())]
        paired = eval_lib.paired_delta_spearman(
            averaged, control, outcomes, groups=groups,
            seed=int(self.config["confirmation"]["bootstrap_seed"]))
        values = [value for value in per_seed.values() if value is not None]
        return {"panel": len(panel), "per_seed_spearman": per_seed,
                "control_spearman": eval_lib.spearman(control, outcomes),
                "paired_delta_spearman": paired,
                "read": read_record,
                "guard": str(self.reserved_guard_path()),
                "opened_once": True,
                "neighbourhoods": len(set(groups)),
                "seed_variation": {"seeds": len(seeds),
                                   "mean": float(np.mean(values)) if values else None,
                                   "spread": (None if len(values) < 2
                                              else max(values) - min(values)),
                                   "separate_from": ("the within-seed neighbourhood bootstrap "
                                                     "above. Seeds are models; neighbourhoods are "
                                                     "rows.")},
                "caveat": ("reserved means unopened BY US. This is public antibody sequence data "
                           "and upstream pretraining exposure for both backbones is unresolved.")}

    def neighbourhood_labels(self, windows, *, radius=None):
        """Sequence-neighbourhood labels: the bootstrap unit. Not a selection batch.

        Greedy cover at the declared radius, so two windows within it share a label
        and travel together through every resample. Selection batches are explicitly
        not this: their measured diameters were 6-10 edits and they were built by a
        separation rule, not by proximity.
        """
        from . import her2_routes_split as split_lib
        radius = int(radius if radius is not None
                     else self.config["confirmation"].get("neighbourhood_radius", 2))
        medoids, _ = split_lib.greedy_medoids(windows, radius=radius)
        index = split_lib.HammingIndex(medoids, radius=radius)
        labels = []
        for window in windows:
            best, chosen = None, None
            for medoid in medoids:
                distance = controls.hamming(window, medoid)
                if distance <= radius and (best is None or distance < best
                                           or (distance == best and medoid < chosen)):
                    best, chosen = distance, medoid
            labels.append(chosen or window)
        del index
        return labels

    # -- the report --------------------------------------------------------
    def build_report(self, *, heartbeat=None):
        """Compute E1-E22. Every panel returns a number or the DATA fact blocking it."""
        from . import her2_routes_controls as controls_lib
        from . import her2_routes_eval as eval_lib
        from . import her2_routes_report as report_lib
        gate = read_stage_record_safe(self.context, "route_gate") or {}
        route = gate.get("winner") or self.config["routes"]["enabled"][0]
        source = self.checkpoint_lineage("report", route)
        checkpoint = Path(source["path"])
        stage_records = {name: (read_stage_record_safe(self.context, name) or {})
                         for name in ("acquire", "preflight", "split", "sources", "profile",
                                      "smoke", "freeze", "bakeoff", "route_gate", "controls",
                                      "priors", "posttrain", "decoding", "confirm")}
        stages = {name: record.get("status", "not_run")
                  for name, record in stage_records.items()}
        blocks = {}
        for name in ("bakeoff", "route_gate", "controls", "priors", "posttrain", "decoding",
                     "confirmation"):
            target = self.context.path(f"{name}.json")
            if target.is_file():
                blocks[name] = paths.read_json(target)
        panels, unsupported = {}, []
        for name, reason in eval_lib.UNSUPPORTED.items():
            panels[name] = report_lib.panel(name, population=0, supported=False, reason=reason)
            unsupported.append(name)
        panel_rows = self.development_panel_outcomes()
        outcomes = [row["value"] for row in panel_rows]
        # E1-E4, E6, E9, E10, E14, E17, E18, E21, E22 from the records that exist.
        panels["E1"] = self.panel_prior_retention(route, checkpoint)
        panels["E2"] = self.panel_conditional_completion(route, checkpoint)
        scores = self.score_panel(route, checkpoint, panel_rows)
        panels["E3"] = report_lib.panel(
            "E3", population=len(panel_rows),
            value=eval_lib.spearman(scores, outcomes),
            assay=self.config["her2"]["assay"],
            extra={"comparators": self.comparator_scores(panel_rows),
                   "panel_accounting": self.development_panel_accounting()})
        panels["E4"] = self.panel_local_ordering(route, checkpoint)
        panels["E5"] = report_lib.e5_reversals(self, route, checkpoint)
        panels["E6"] = self.panel_quantitative(blocks.get("confirmation"), panel_rows)
        panels["E7"] = self.panel_combination(route, checkpoint)
        panels["E8"] = self.panel_interaction()
        panels["E9"] = self.panel_raw_generation(blocks.get("decoding"))
        panels["E10"] = self.panel_catalogue(blocks.get("decoding"))
        panels["E11"] = self.panel_structural(blocks.get("priors"))
        panels["E13"] = (report_lib.panel(
            "E13", population=0, supported=False,
            reason=("the learning-curve fits are a separate, declared stage budget and this run "
                    "did not enable confirmation.learning_curve; the code path is "
                    "her2_routes_report.e13_learning_curve"))
            if not bool(self.config["confirmation"].get("learning_curve")) else
            report_lib.e13_learning_curve(
                self, route, updates=int(self.config["confirmation"]["learning_curve_updates"]),
                seed=int(self.config["confirmation"]["seeds"][0]), heartbeat=heartbeat))
        panels["E14"] = self.panel_repeatability(blocks.get("confirmation"))
        panels["E15"] = report_lib.e15_transfer(self, route, checkpoint)
        gates = self.config["gates"]
        if (gates.get("readout_affinity_tolerance") is not None
                or gates.get("readout_expression_tolerance") is not None):
            # Separate units, separate numbers: a half-declared pair is refused by the panel.
            tolerances = {"affinity_tolerance": gates.get("readout_affinity_tolerance"),
                          "expression_tolerance": gates.get("readout_expression_tolerance")}
        else:
            # The legacy single number, broadcast to both readouts -- and recorded as a broadcast.
            tolerances = {"tolerance": float(gates.get("readout_tolerance", 0.25))}
        panels["E16"] = report_lib.e16_readouts(
            self, assay_pairs=gates.get("readout_assay_pairs"), **tolerances)
        panels["E17"] = self.panel_learning_from_context(blocks.get("controls"))
        panels["E18"] = self.panel_teacher(blocks.get("priors"))
        panels["E19"] = report_lib.e19_shortcuts(
            self, route, checkpoint, margins=self.config["controls"]["margins"],
            training_bank=self.training_label_bank(),
            adaptation_bank={"windows": list(self.adaptation_bank()), "values": None,
                             "provenance": self.adaptation_bank_record()})
        panels["E21"] = self.panel_local_versus_global(blocks.get("controls"))
        panels["E22"] = self.panel_mechanism(blocks.get("decoding"))
        unsupported = sorted(name for name, record in panels.items()
                             if isinstance(record, dict) and not record.get("supported", True))
        diagnosis = report_lib.no_gain_diagnosis(
            panels=panels, controls=blocks.get("controls") or {},
            exposure=(blocks.get("bakeoff") or {}).get("routes", {}),
            learning_curve=(panels.get("E13") or {}).get("curve"))
        return report_lib.assemble(
            panels={**panels, "stage_blocks": blocks}, stages=stages,
            unsupported=unsupported,
            caveats=[
                "upstream pretraining exposure for p-IgGen and AbLang2 is UNRESOLVED; the "
                "local 1.94M-record OAS probe finds zero hits for the wild-type core and also "
                "fails to find trastuzumab itself, so the negative is underpowered",
                "the display test split is sealed permanently and was never opened",
                "the CR9114 H1 partition was never read; the protected genotypes are the "
                "complement of the pinned eligibility whitelist, whose own selection on H1 "
                "eligibility is inherited by every transfer claim from that population",
                "SPR-152 is a diagnostic and never entered selection",
                "no antibody generated by this project has been experimentally assayed",
                "batch diameters were 6-10 edits; selection batches are not families and not "
                "bootstrap units"],
            extra={"route": route, "lineage": source,
                   "shortcut_context": controls_lib.shortcut_context(),
                   "no_gain_diagnosis": diagnosis,
                   "source_inventory": {
                       key: value for key, value in
                       dict(self._cache.get("source_inventory") or {}).items()
                       if key != "sources"}})

    # -- individual panels --------------------------------------------------
    def panel_prior_retention(self, route, checkpoint):
        """E1: held-out repertoire likelihood under the trained model and the backbone."""
        from . import her2_routes_report as report_lib
        if not capabilities.supports_token_likelihood(route):
            # Refused before the repertoire is read or a model loaded; no retention number is made
            # up for a route that has no per-token likelihood to retain.
            return report_lib.panel(
                "E1", population=0, supported=False,
                reason=("the energy route has no per-token likelihood: it supplies an energy and a "
                        "walk-jump sampler, so its likelihood on a held-out repertoire is "
                        "undefined and none is reported"))
        record = self.paired_oas()
        held = dict(record["by_split"]).get("val") or ()
        if not held:
            return report_lib.panel(
                "E1", population=0, supported=False,
                reason=("the prepared paired release declares no held-out split, so there is no "
                        "unseen repertoire to measure retention on. Its splits are "
                        f"{sorted(dict(record['by_split']))}."))
        rows = list(held)[:int(self.config["gates"].get("retention_rows", 128))]
        rng = np.random.default_rng([int(self.config["split"]["seed"]), 31])
        views = data_lib.reconstruction_views(
            rows, rng=rng, mask_fraction=float(self.config["curriculum"]["mask_fraction"]),
            condition_spec=self.condition_spec, family="display", route=self._kind(route),
            max_positions=self._positions(route))
        out = {}
        for label, target in (("trained", checkpoint), ("backbone", None)):
            model, heads, _load = self.load_for_scoring(route, target)
            with torch.no_grad():
                total, count = 0.0, 0
                for start in range(0, len(views), 8):
                    batch = self.collate(route, views[start:start + 8])
                    log_probs = self.window_log_probs(route, model, batch)
                    total += float(log_probs.sum())
                    count += int(np.asarray(batch.target_mask).sum())
            out[label] = None if count == 0 else total / count
            del model, heads
        delta = (None if out["trained"] is None or out["backbone"] is None
                 else out["trained"] - out["backbone"])
        return report_lib.panel("E1", population=len(rows), value=delta,
                                extra={"mean_log_probability_per_target": out,
                                       "split": "val",
                                       "reading": ("negative means the trained model lost prior "
                                                   "likelihood on an unseen repertoire relative "
                                                   "to its own starting point")})

    def panel_conditional_completion(self, route, checkpoint):
        """E2: does the quality condition move the window likelihood? Measured, both ways."""
        from . import her2_routes_report as report_lib
        from .her2_routes_conditions import BIN_HIGH, BIN_LOW, Condition
        from .her2_routes_serial import ablang_serialization, causal_serialization, her2_construct
        rows = self.development_panel_outcomes()
        if not rows:
            return report_lib.panel("E2", population=0, supported=False,
                                    reason="the development panel carries no qualified outcome")
        if route == ROUTE_ENERGY:
            return report_lib.panel(
                "E2", population=len(rows), supported=False,
                reason=("the energy route has no per-window conditional to compare across quality "
                        "requests: it supplies an energy, not a normalised likelihood, and this "
                        "panel is the difference of two window log probabilities. Its conditional "
                        "claim is measured by the density-only control instead."))
        scaffold, sites = self.scaffold(), her2_core_edit_map()
        model, heads, _load = self.load_for_scoring(route, checkpoint)
        scores = {}
        for name in (BIN_HIGH, BIN_LOW, BIN_NULL):
            condition = Condition(family="affinity", bin=name,
                                  assay=self.config["her2"]["assay"])
            views = []
            for index, row in enumerate(rows):
                construct = her2_construct(scaffold, row["window"], record_id=f"e2{index}")
                serialization = (causal_serialization(
                    construct, order=sites, residues=construct.residues(sites),
                    max_positions=self._positions(route)) if route == ROUTE_CAUSAL
                    else ablang_serialization(construct, order=sites))
                views.append((construct, serialization, condition))
            with torch.no_grad():
                totals = []
                for start in range(0, len(views), 16):
                    batch = self.collate(route, views[start:start + 16])
                    totals.extend(float(value) for value in
                                  self.window_log_probs(route, model, batch).sum(dim=1))
            scores[name] = totals
        del model, heads
        outcomes = [row["value"] for row in rows]
        high = np.asarray(scores[BIN_HIGH]) - np.asarray(scores[BIN_LOW])
        return report_lib.panel(
            "E2", population=len(rows),
            value=float(np.corrcoef(high, np.asarray(outcomes))[0, 1])
            if len(rows) > 2 and np.std(high) > 0 else None,
            extra={"mean_high_minus_low": float(np.mean(high)),
                   "mean_high_minus_null": float(np.mean(
                       np.asarray(scores[BIN_HIGH]) - np.asarray(scores[BIN_NULL]))),
                   "reading": ("the value is the correlation between (high minus low) window log "
                               "probability and the measured outcome. Zero means the condition "
                               "moves the likelihood in a way unrelated to the measurement.")})

    def panel_local_ordering(self, route, checkpoint):
        """E4: does the decoder order two measured alternatives correctly? Concordance.

        The population is requested EXPLICITLY: the exact-pKD measured pairs of the HER2
        affinity assay (``her2.assay``, family ``affinity``) -- the population this panel
        advertises -- rather than whatever the pool's enumeration order happened to put
        first. Each pair is scored under the condition its own metadata names, which for this
        population is the HER2 affinity condition. The record states the pairs version, the
        population, the qualification counts and the sampling rule (a content-addressed
        subsample of ``gates.local_pairs`` pairs, independent of enumeration order).
        """
        from . import her2_routes_pairs as pairs_lib
        from . import her2_routes_report as report_lib
        if route == ROUTE_ENERGY:
            return report_lib.panel(
                "E4", population=0, supported=False,
                reason=("the local ordering panel scores two alternatives at one masked "
                        "background through a conditional, which the energy route does not "
                        "supply. Its ranking endpoint is the negative energy and is reported by "
                        "E3."))
        population = {"assay": self.config["her2"]["assay"], "family": "affinity"}
        pool = self.measured_pairs(population=population, allow_empty=True)
        inventory = dict(self._cache.get("pair_inventory") or {})
        if not pool:
            return report_lib.panel(
                "E4", population=0, supported=False,
                reason=("no exact finite pKD measured alternative pair of the requested "
                        f"population {population} survived qualification and the closure; "
                        f"refusals: {(inventory.get('counts') or {}).get('excluded')}"))
        pairs, sampling = pairs_lib.sample_pairs(
            pool, int(self.config["gates"].get("local_pairs", 256)))
        model, heads, _load = self.load_for_scoring(route, checkpoint)
        correct, ties = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(pairs), 16):
                chunk = pairs[start:start + 16]
                batch = self.collate(route, self.pair_views(route, chunk))
                window = self.window_log_probs(route, model, batch).sum(dim=1)
                for index in range(len(chunk)):
                    difference = float(window[2 * index] - window[2 * index + 1])
                    if difference == 0.0:
                        correct += 0.5
                        ties += 1
                    elif difference > 0:
                        correct += 1.0
        del model, heads
        return report_lib.panel(
            "E4", population=len(pairs), value=correct / len(pairs),
            extra={"score_ties": ties, "pairs_version": pairs_lib.PAIRS_VERSION,
                   "requested_population": population, "sampling": sampling,
                   "pool": {"available": inventory.get("available"),
                            "selected": inventory.get("selected"),
                            "outside_population": inventory.get("outside_population"),
                            "qualification": (inventory.get("counts") or {})},
                   "definition": ("fraction of exact-pKD measured alternative pairs the "
                                  "decoder assigns the higher window log probability to the "
                                  "better-measured member. A model-score tie earns half "
                                  "credit; measured ties are excluded and counted.")})

    def panel_quantitative(self, confirmation, panel_rows):
        """E6: the reserved quantitative endpoint, or the reason it is closed."""
        from . import her2_routes_report as report_lib
        block = dict(confirmation or {})
        reserved = dict(block.get("reserved") or {})
        if not reserved:
            return report_lib.panel(
                "E6", population=len(panel_rows), supported=False,
                reason=("the reserved panel is unopened. Its endpoint is reported only after the "
                        "confirmation seeds complete and every gate holds; the status is in "
                        f"confirmation.json ({block.get('status') or 'not run'})."))
        paired = dict(reserved.get("paired_delta_spearman") or {})
        return report_lib.panel(
            "E6", population=int(reserved.get("panel") or 0),
            value=paired.get("delta"), interval=paired.get("interval"),
            assay=self.config["her2"]["assay"],
            extra={"per_seed": reserved.get("per_seed_spearman"),
                   "control": reserved.get("control_spearman"),
                   "seed_variation": reserved.get("seed_variation"),
                   "neighbourhoods": paired.get("neighbourhoods")})

    def panel_combination(self, route, checkpoint):
        """E7: does the model order combinations it never saw? Distance-stratified."""
        from . import her2_routes_report as report_lib
        rows = [row for row in self.development_panel_outcomes()
                if controls.hamming(row["window"],
                                    self.config["her2"]["wild_type_core"]) >= 2]
        if len(rows) < 4:
            return report_lib.panel(
                "E7", population=len(rows), supported=False,
                reason=(f"only {len(rows)} development core carries two or more substitutions, so "
                        "there is no combination stratum to measure generalisation on."))
        from . import her2_routes_eval as eval_lib
        scores = self.score_panel(route, checkpoint, rows)
        outcomes = [row["value"] for row in rows]
        strata = [controls.distance_stratum(controls.hamming(
            row["window"], self.config["her2"]["wild_type_core"])) for row in rows]
        return report_lib.panel(
            "E7", population=len(rows), value=eval_lib.spearman(scores, outcomes),
            distances=[controls.hamming(row["window"],
                                        self.config["her2"]["wild_type_core"]) for row in rows],
            extra={"stratified": controls.stratified_concordance(scores, outcomes, strata)})

    def panel_interaction(self):
        """E8, DATA side: the interactions of the MEASURED TRAINING quartets. A count, then a number.

        This is a training-data interaction diagnostic. It takes no model predictions: the
        quartets are enumerated from the enrolled training partitions and the value is the mean
        absolute interaction of the measured numbers, a property of the data. It is not a model's
        interaction accuracy, and nothing here scores one. Whether a model reproduces the measured
        interactions is a separate, predictive endpoint that this API does not evaluate; the record
        says so in ``predictive_endpoint`` instead of folding it into the data reason.
        """
        from . import her2_routes_report as report_lib
        from .her2_routes_sources import SCALE_PKD
        rows = [row for row in self.eligible_summaries() if row.is_training]
        contrasts = self.contrast_pool(rows, seed=int(self.config["split"]["seed"]),
                                       cache_key=("E8", int(self.config["split"]["seed"]),
                                                  len(rows)))
        inventory = dict(self._cache.get("contrast_inventory") or {})
        quartets = int(dict(inventory.get("by_kind") or {}).get(
            data_lib.CONTRAST_COUPLED, 0))
        diagnostic = {
            "evidence_kind": "training_data_interaction_diagnostic",
            "model_predictions": None,
            "predictive_endpoint": {
                "status": "not_evaluated",
                "scope": ("this API receives no model predictions, so no model interaction "
                          "accuracy, rescue or epistasis recovery is computed or implied here"),
                "reading": ("the value below describes the measured training quartets only; a "
                            "predictive claim needs a model's scores on held-out quartets, which "
                            "is a different panel input")},
            "population_scope": "measured training quartets, not held-out data"}
        if quartets == 0:
            return report_lib.panel(
                "E8", population=0, supported=False,
                reason=("no complete measured quartet -- background, both singles and the double "
                        "in one assay -- exists in the enrolled training partitions, so no "
                        "numerical interaction is computable. The enumeration's available count "
                        "is in the source inventory."),
                extra=diagnostic)
        # An interaction is a difference of differences ON ONE SCALE. A pKD interaction and an
        # enrichment-ratio interaction are numbers in different units, so they are summarised per
        # scale and never pooled into one mean. The primary number is the pKD affinity quartets'.
        by_scale = {}
        for contrast in contrasts:
            if contrast.kind == data_lib.CONTRAST_COUPLED and contrast.interaction is not None:
                by_scale.setdefault(contrast.scale or "unspecified", []).append(
                    float(contrast.interaction))
        scales = {scale: {"quartets": len(values),
                          "mean_absolute_interaction": float(np.mean(np.abs(values))),
                          "max_absolute_interaction": float(np.max(np.abs(values)))}
                  for scale, values in sorted(by_scale.items())}
        interactions = by_scale.get(SCALE_PKD, [])
        if not interactions:
            return report_lib.panel(
                "E8", population=0, supported=False,
                reason=("no complete measured quartet on the pKD affinity scale exists in the "
                        "enrolled training partitions, so no numerical affinity interaction is "
                        "computable. Quartets on other scales are reported per scale beside this "
                        "and are not pooled with pKD: they are in different units."),
                extra={**diagnostic, "quartets": quartets, "by_scale": scales,
                       "primary_scale": SCALE_PKD})
        return report_lib.panel(
            "E8", population=len(interactions),
            value=float(np.mean(np.abs(interactions))),
            extra={**diagnostic, "quartets": quartets, "primary_scale": SCALE_PKD,
                   "by_scale": scales,
                   "mean_absolute_interaction": float(np.mean(np.abs(interactions))),
                   "max_absolute_interaction": float(np.max(np.abs(interactions))),
                   "pooling_rule": ("the value is the pKD affinity quartets' only. Other scales "
                                    "are listed in by_scale and never averaged into it."),
                   "definition": "(AB - bg) - ((A - bg) + (B - bg)) within one assay"})

    def panel_raw_generation(self, decoding):
        """E9: raw draws, novelty and duplicate rates at the declared prefixes."""
        from . import her2_routes_report as report_lib
        block = dict(decoding or {})
        strategies = dict(block.get("strategies") or {})
        if not strategies:
            return report_lib.panel("E9", population=0, supported=False,
                                    reason="the decoding stage has not run, so no raw draw exists")
        rows = {}
        for name, record in sorted(strategies.items()):
            report = dict(record.get("generation") or {})
            rows[name] = {"prefixes": report.get("rows"),
                          "accounting": record.get("accounting"),
                          "independent_quality": record.get("independent_quality")}
        largest = max((row for record in strategies.values()
                       for row in (dict(record.get("generation") or {}).get("rows") or [])),
                      key=lambda row: row["draws"], default=None)
        return report_lib.panel(
            "E9", population=int(largest["draws"]) if largest else 0,
            value=(None if largest is None else largest.get("novelty_rate")),
            extra={"by_strategy": rows, "ranked": block.get("ranked_by_independent_quality"),
                   "labels": ("an off-catalogue draw is an in-silico candidate. Nothing generated "
                              "here has been experimentally assayed.")})

    def panel_catalogue(self, decoding):
        """E10: closed-catalogue recovery. Allocation among known sequences, labelled."""
        from . import her2_routes_report as report_lib
        block = dict(decoding or {})
        strategies = dict(block.get("strategies") or {})
        if not strategies:
            return report_lib.panel("E10", population=0, supported=False,
                                    reason="the decoding stage has not run")
        rates = {}
        for name, record in sorted(strategies.items()):
            rows = dict(record.get("generation") or {}).get("rows") or []
            if rows:
                rates[name] = rows[-1].get("catalogue_recovery_rate")
        usable = [value for value in rates.values() if value is not None]
        return report_lib.panel(
            "E10", population=int((block.get("bank") or {}).get("windows") or 0),
            value=float(max(usable)) if usable else None,
            extra={"by_strategy": rates,
                   "meaning": ("closed-catalogue recovery measures allocation among KNOWN "
                               "sequences. It is not novel-design success.")})

    def panel_repeatability(self, confirmation):
        """E14: seed spread of the frozen recipe, separate from row uncertainty."""
        from . import her2_routes_report as report_lib
        block = dict(confirmation or {})
        reserved = dict(block.get("reserved") or {})
        variation = dict(reserved.get("seed_variation") or {})
        if not variation:
            return report_lib.panel(
                "E14", population=len(block.get("seeds") or {}), supported=False,
                reason=("the reserved panel is unopened, so there is no per-seed endpoint to take "
                        "a spread of. The development-side seed table is in confirmation.json."))
        return report_lib.panel("E14", population=int(variation.get("seeds") or 0),
                                value=variation.get("spread"),
                                extra={**variation,
                                       "separate_from": ("the within-seed neighbourhood bootstrap. "
                                                         "Rows are not extra models.")})

    def panel_learning_from_context(self, controls_block):
        """E17: the full arm against each matched control. The gate's own table."""
        from . import her2_routes_report as report_lib
        block = dict(controls_block or {})
        rankings = dict(block.get("rankings") or {})
        if not rankings:
            return report_lib.panel("E17", population=0, supported=False,
                                    reason="the controls stage has not run")
        full = rankings.get(controls.ARM_FULL)
        deltas = {arm: (None if full is None or value is None else full - value)
                  for arm, value in sorted(rankings.items()) if arm != controls.ARM_FULL}
        usable = [value for value in deltas.values() if value is not None]
        return report_lib.panel(
            "E17", population=len(rankings), value=float(min(usable)) if usable else None,
            extra={"rankings": rankings, "deltas": deltas,
                   "matched_exposure": block.get("matched_exposure"),
                   "value_is": "the SMALLEST margin over any control arm",
                   "reading": ("a head-only arm that matches the full arm means the gain is in "
                               "the head, not the decoder.")})

    def panel_structural(self, priors):
        """E11: the structural teacher's measured contribution, or its recorded state.

        The blocking reason is read from the priors record rather than asserted, because
        this teacher is **obtainable**: a local ESM-IF1 checkpoint exists and may be read
        explicitly. "Not in this tree" would be false, and "not implemented" would be a
        placeholder; what is true is a prerequisite state, and it is reported.
        """
        from . import her2_routes_report as report_lib
        from . import her2_routes_teachers as teachers_lib
        block = dict(priors or {})
        name = teachers_lib.TEACHER_STRUCTURAL
        qualified = dict(block.get("teachers") or {}).get(name)
        distilled = dict(block.get("distillations") or {}).get(name)
        if distilled:
            return report_lib.panel(
                "E11", population=1,
                value=(distilled.get("development_spearman") or {}).get("delta"),
                extra={"qualification": qualified, "distillation": distilled})
        skipped = dict(block.get("skipped") or {}).get(name) or {}
        reason = str(skipped.get("reason") or "")
        detail = (f"the structural teacher was not distilled. Its recorded prerequisite state: "
                  f"present={skipped.get('present')}, read={skipped.get('read')}, "
                  f"gate={skipped.get('gate') or reason or 'not declared available'}. "
                  "It is obtainable rather than absent, and it enters only after its shuffled "
                  "and position-only controls pass on development.")
        return report_lib.panel("E11", population=0, supported=False, reason=detail,
                                extra={"recorded_state": skipped,
                                       "qualification": qualified})

    def panel_teacher(self, priors):
        """E18: the teacher's measured contribution, distillation against rescoring."""
        from . import her2_routes_report as report_lib
        block = dict(priors or {})
        distillations = dict(block.get("distillations") or {})
        if not distillations:
            return report_lib.panel(
                "E18", population=0, supported=False,
                reason=("no teacher cleared both controls on development, so none was distilled. "
                        "The measured qualification table and the skip reasons are in "
                        "priors.json."))
        rows = {}
        for name, record in sorted(distillations.items()):
            rows[name] = {"distillation_delta": (record.get("development_spearman") or {}
                                                 ).get("delta"),
                          "rescoring_delta": (record.get("rescoring") or {}).get("delta"),
                          "parameters_moved": (record.get("distillation") or {}
                                               ).get("max_abs_parameter_change")}
        usable = [row["distillation_delta"] for row in rows.values()
                  if row["distillation_delta"] is not None]
        return report_lib.panel(
            "E18", population=len(rows), value=float(max(usable)) if usable else None,
            extra={"by_teacher": rows,
                   "separation": ("distillation moves the student; rescoring does not. Both are "
                                  "reported because one number cannot say which mechanism "
                                  "produced a change."),
                   "caveat": "agreement with a teacher is not independent biological validation"})

    def panel_local_versus_global(self, controls_block):
        """E21: the local-only arm against the full arm. The whole-window task's value."""
        from . import her2_routes_report as report_lib
        rankings = dict(dict(controls_block or {}).get("rankings") or {})
        full = rankings.get(controls.ARM_FULL)
        local = rankings.get(controls.ARM_LOCAL_ONLY)
        if full is None or local is None:
            return report_lib.panel(
                "E21", population=len(rankings), supported=False,
                reason=("the local-only control arm has not run, so the whole-window task's "
                        "contribution is not separable."))
        return report_lib.panel("E21", population=len(rankings), value=float(full - local),
                                extra={"full": full, "local_only": local,
                                       "reading": ("positive means the whole-window task "
                                                   "contributed beyond the local contrasts")})

    def panel_mechanism(self, decoding):
        """E22: does the fill order change the samples? The generative-mechanism panel."""
        from . import her2_routes_report as report_lib
        strategies = dict(dict(decoding or {}).get("strategies") or {})
        if len(strategies) < 2:
            return report_lib.panel(
                "E22", population=len(strategies), supported=False,
                reason=("fewer than two fill strategies produced draws, so there is no mechanism "
                        "comparison. The energy route has one sampler and is compared by raw "
                        "generation only."))
        quality = {name: (record.get("independent_quality") or {}).get("mean")
                   for name, record in sorted(strategies.items())}
        usable = [value for value in quality.values() if value is not None]
        return report_lib.panel(
            "E22", population=len(strategies),
            value=(None if len(usable) < 2 else float(max(usable) - min(usable))),
            extra={"by_strategy": quality,
                   "occupancy": {name: (dict(record.get("generation") or {}).get("rows")
                                        or [{}])[-1].get("occupancy")
                                 for name, record in sorted(strategies.items())},
                   "reading": ("the spread of independently assessed quality across fill orders. "
                               "Zero means the order does not change what is produced.")})


def read_stage_record_safe(context, stage):
    target = context.path("stages", f"{stage}.json")
    return paths.read_json(target) if target.is_file() else None


def controls_owner():
    """Who minted a reserved unlock. Reused from the landed harness's owner identity."""
    from .her2_replay_campaign import owner_identity
    return owner_identity()

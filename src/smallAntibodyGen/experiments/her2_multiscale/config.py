"""The pilot's configuration: a closed schema bound to the base routes config by digest.

Stdlib only, so the command line can read and refuse a config before ``prepare_environment`` and
long before torch is imported.

The pilot config declares only what the pilot adds: which arm, how many updates, the seed, how often
a local draw is replaced and with which distances, the reservoir bound, the diagnostic bounds and the
split counts the restored membership must reproduce. Everything else -- the route block, the
optimizer (Adafactor, never substituted), learning rate, curriculum weights, shares, family mix,
microbatch and accumulation, the split settings and the source list -- is the BASE config's, which
the pilot names by path and SHA-256. A field this schema does not list is refused rather than
ignored: a stray ``routes`` or ``optimizer`` key would otherwise look like an override that nothing
reads.

Numbers are strict. JSON ``NaN``/``Infinity`` are refused at parse time, a ``bool`` is not an
integer, ``1000.0`` is not an update count, and a probability outside ``[0, 1]`` or a non-positive
weight is an error, not a clamp.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

from ..her2_runtime import require

PILOT_SCHEMA = "her2-multiscale-pilot/1"

#: The only route the pilot trains: native DPLM 150M. A missing pinned weight is fatal.
ROUTE = "dplm_150m_masked"

#: arm -> (mask geometry, pair-distance scheme). The four arms of one pilot differ in these two
#: choices only; every arm keeps the base reconstruction, heads, measured sources and legacy quartet
#: draws.
ARM_DESIGN = {"baseline": ("suffix", "d1"), "masks": ("subset", "d1"),
              "pairs": ("suffix", "d123"), "combined": ("subset", "d123")}
ARMS = tuple(ARM_DESIGN)
DISTANCES = (1, 2, 3)

TOP_LEVEL = ("schema_version", "pilot_id", "base", "route", "arm", "updates", "seed", "local",
             "inputs", "diagnostics")
OPTIONAL_TOP_LEVEL = ("note",)
MAX_UPDATES = 20000
SPLIT_COUNT_KEYS = ("candidates", "development", "reserved", "excluded", "retained_for_training")

#: How the quality bins of the frozen sources are treated. ``current_exact_training`` re-fits every
#: bin threshold with the CURRENT source code (``fit_condition_bins``: the cuts come from the TRAIN
#: rows' exact points only) over the frozen split and the same source bytes, and freezes what that
#: produced inside this pilot. The production flight's thresholds were fitted on ``row.value``, the
#: mean of a summary's uncensored replicates, over summaries that also held censored replicates (the
#: exact-only rule now excludes them), so the two differ; the difference is reported, never
#: ignored, and nothing here claims to replay the historical thresholds.
BIN_POLICIES = ("current_exact_training",)


def _refuse_constant(constant):
    raise ValueError(f"{constant} is not a valid JSON number; the pilot config takes finite "
                     "numbers only")


def _no_duplicate_keys(pairs):
    keys = [key for key, _ in pairs]
    duplicated = sorted({key for key in keys if keys.count(key) > 1})
    require(not duplicated, f"duplicate key(s) {duplicated} in one JSON object")
    return dict(pairs)


def read_json_strict(path):
    """Parse JSON refusing ``NaN``/``Infinity`` and a repeated key in one object."""
    path = Path(path)
    require(path.is_file(), f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_refuse_constant,
                      object_pairs_hook=_no_duplicate_keys)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed(block, name, keys, *, optional=()):
    require(isinstance(block, dict), f"{name} must be an object, got {type(block).__name__}")
    unknown = sorted(set(block) - set(keys) - set(optional))
    missing = sorted(set(keys) - set(block))
    require(not unknown, f"{name} has unrecognised field(s) {unknown}; the schema is closed, "
                         "and a field nothing reads is refused rather than ignored")
    require(not missing, f"{name} is missing field(s) {missing}")
    return block


def _integer(value, name, *, minimum, maximum):
    require(isinstance(value, Integral) and not isinstance(value, bool),
            f"{name} must be an integer, got {value!r}")
    require(minimum <= int(value) <= maximum,
            f"{name} must be between {minimum} and {maximum}, got {value}")
    return int(value)


def _real(value, name, *, low, high, low_open=False):
    require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"{name} must be a finite number, got {value!r}")
    number = float(value)
    require(number > low if low_open else number >= low, f"{name} must exceed {low}, got {value}")
    require(number <= high, f"{name} must be at most {high}, got {value}")
    return number


def _text(value, name):
    require(isinstance(value, str) and value.strip(), f"{name} must be a non-empty string")
    return value


def _digest(value, name):
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{name} must be a lowercase 64-character SHA-256 hex digest, got {value!r}")
    return value


def _inside(root, relative, name):
    text = _text(relative, name)
    require(not os.path.isabs(text), f"{name} must be relative to the code root, got {text!r}")
    base = os.path.normpath(os.path.abspath(str(root)))
    target = os.path.normpath(os.path.join(base, text))
    require(target.startswith(base + os.sep), f"{name} {text!r} escapes the code root")
    return Path(target)


@dataclass(frozen=True)
class PilotConfig:
    """The validated pilot declaration. ``document`` is the config exactly as parsed."""

    path: Path
    document: dict
    sha256: str
    pilot_id: str
    base_path: Path
    base_sha256: str
    arm: str
    updates: int
    seed: int
    replacement_probability: float
    declared_distance_weights: dict
    reservoir_capacity: int
    diagnostics: dict
    expected_counts: dict
    bin_policy: str

    route = ROUTE

    @property
    def geometry(self):
        return ARM_DESIGN[self.arm][0]

    @property
    def distance_scheme(self):
        return ARM_DESIGN[self.arm][1]

    @property
    def distance_weights(self):
        """``{distance: probability}`` this arm draws pair distances with. Never renormalised away.

        ``d1`` arms draw distance 1 always. ``d123`` arms use the declared weights, normalised to
        sum to one; a distance with no weight is not silently skipped (the schema requires all
        three), and the pool refuses to start if a weighted distance has no pair.
        """
        if self.distance_scheme == "d1":
            return {1: 1.0}
        total = sum(self.declared_distance_weights.values())
        return {distance: weight / total
                for distance, weight in sorted(self.declared_distance_weights.items())}

    def with_arm(self, arm):
        require(arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}")
        return replace(self, arm=arm, document=dict(self.document, arm=arm))

    def document_record(self):
        return {"path": str(self.path), "sha256": self.sha256, "arm": self.arm,
                "geometry": self.geometry, "distance_scheme": self.distance_scheme,
                "distance_weights": {str(key): value for key, value in
                                     self.distance_weights.items()},
                "document": self.document}


def load_pilot_config(path, *, code_root):
    """Read ``path``, refuse anything the closed schema does not list, bind the base config."""
    path = Path(path)
    document = read_json_strict(path)
    _closed(document, "the pilot config", TOP_LEVEL, optional=OPTIONAL_TOP_LEVEL)
    require(document["schema_version"] == PILOT_SCHEMA,
            f"schema_version is {document['schema_version']!r}, not {PILOT_SCHEMA!r}")
    require(document["route"] == ROUTE,
            f"route must be {ROUTE!r}: the pilot trains the native DPLM 150M route only, got "
            f"{document['route']!r}")
    arm = document["arm"]
    require(arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}")
    base = _closed(document["base"], "base", ("config", "sha256"))
    base_path = _inside(code_root, base["config"], "base.config")
    require(base_path.is_file(), f"the base config {base_path} is missing")
    base_sha256 = _digest(base["sha256"], "base.sha256")
    observed = file_sha256(base_path)
    require(observed == base_sha256,
            f"the base config {base_path.name} hashes to {observed}, the pilot binds "
            f"{base_sha256}. The pilot trains with the base's route, optimizer, weights and "
            "shapes, so a base that changed is a different pilot")
    local = _closed(document["local"], "local",
                    ("replacement_probability", "distance_weights", "reservoir_capacity"))
    weights = _closed(local["distance_weights"], "local.distance_weights",
                      tuple(str(distance) for distance in DISTANCES))
    declared = {int(key): _real(value, f"local.distance_weights.{key}", low=0.0, high=1e9,
                                low_open=True) for key, value in weights.items()}
    diagnostics = _closed(document["diagnostics"], "diagnostics",
                          ("batch_size", "completion_rows", "completion_counts", "order_rows",
                           "fit_pairs_per_distance"))
    counts = diagnostics["completion_counts"]
    require(isinstance(counts, list) and counts, "diagnostics.completion_counts must be a list")
    cleaned_counts = tuple(_integer(item, "a diagnostics.completion_counts entry", minimum=1,
                                    maximum=10) for item in counts)
    require(list(cleaned_counts) == sorted(set(cleaned_counts)),
            "diagnostics.completion_counts must be strictly increasing")
    inputs = _closed(document["inputs"], "inputs", ("expected_counts", "bin_policy"))
    bin_policy = inputs["bin_policy"]
    require(isinstance(bin_policy, str) and bin_policy in BIN_POLICIES,
            f"inputs.bin_policy must be one of {BIN_POLICIES}, got {bin_policy!r}")
    expected = _closed(inputs["expected_counts"], "inputs.expected_counts", SPLIT_COUNT_KEYS)
    return PilotConfig(
        path=path, document=document, sha256=file_sha256(path),
        pilot_id=_text(document["pilot_id"], "pilot_id"), base_path=base_path,
        base_sha256=base_sha256, arm=arm,
        updates=_integer(document["updates"], "updates", minimum=1, maximum=MAX_UPDATES),
        seed=_integer(document["seed"], "seed", minimum=0, maximum=2 ** 31 - 1),
        replacement_probability=_real(local["replacement_probability"],
                                      "local.replacement_probability", low=0.0, high=1.0),
        declared_distance_weights=declared,
        reservoir_capacity=_integer(local["reservoir_capacity"], "local.reservoir_capacity",
                                    minimum=1, maximum=65536),
        diagnostics={
            "batch_size": _integer(diagnostics["batch_size"], "diagnostics.batch_size",
                                   minimum=1, maximum=64),
            "completion_rows": _integer(diagnostics["completion_rows"],
                                        "diagnostics.completion_rows", minimum=1, maximum=200),
            "completion_counts": list(cleaned_counts),
            "order_rows": _integer(diagnostics["order_rows"], "diagnostics.order_rows",
                                   minimum=1, maximum=200),
            "fit_pairs_per_distance": _integer(diagnostics["fit_pairs_per_distance"],
                                               "diagnostics.fit_pairs_per_distance", minimum=1,
                                               maximum=512)},
        expected_counts={key: _integer(expected[key], f"inputs.expected_counts.{key}", minimum=0,
                                       maximum=10 ** 7) for key in SPLIT_COUNT_KEYS},
        bin_policy=bin_policy)

"""Verified reuse of the completed campaign's artifacts. Nothing here is rebuilt.

A reused historical arm is only a factorial cell if this flight consumes the
*same* bytes the old one did. Re-deriving a stream that happens to hash equal is
evidence about the derivation rule; it is not the saved replay order, the saved
reference cache or the saved bank, and three of those four were being
regenerated rather than loaded.

Everything in here is read-only and every loader refuses rather than falls back:

* :func:`load_checkpoint` accepts the schemas that actually exist on disk --
  ``her2-core-policy/1`` for parents and this flight's own endpoints,
  ``her2-parent-replay-endpoint/1`` for the completed campaign's -- and
  strict-loads with a state-digest check. It never relaxes ``strict=True`` and
  never coerces a parameter into place.
* :func:`load_task_stream` and :func:`load_replay_bank` go through
  ``her2_support_paths.read_shard``, which re-derives every recorded dtype,
  shape, row order, array digest and container hash before returning anything.
* :func:`load_reference_cache` verifies the cached values against their recorded
  digest, the core order against the population this flight built, the scaffold
  prefix against the one it loaded, and the parent checkpoint against the one it
  is about to train from. The historical ``config_sha256`` names the old
  campaign's configuration by construction and is reported, not asserted equal.

A wrong parent, a wrong population or a changed archive is a failure here, where
it is one line, rather than a plausible number forty hours later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import her2_support_paths as paths
from .her2_nf_contract import NF_SCHEMA
from .her2_policy import POLICY_SCHEMA
from .her2_nf_storage import load_cpu, state_digest, state_dict_digest
from .her2_preferences import array_digest as reference_values_digest, core_digest
from .her2_runtime import require

#: Checkpoint schemas this flight knows how to load, and what each one is.
HISTORICAL_ENDPOINT_SCHEMA = "her2-parent-replay-endpoint/1"
CNN_SCHEMA = "her2-cnn/1"
CHECKPOINT_SCHEMAS = {
    POLICY_SCHEMA: "a parent SFT checkpoint or one of this flight's own endpoints",
    HISTORICAL_ENDPOINT_SCHEMA: "an endpoint of the completed parent-replay campaign",
}

#: The reference cache covers the chosen population followed by the rejected one.
REFERENCE_CACHE_SCHEMA = "her2-reference-cache/1"


def load_checkpoint(path, model, *, device="cpu"):
    """Strict, schema-aware, digest-checked load of any checkpoint this flight reads.

    ``her2_policy.load_checkpoint`` accepts only ``her2-core-policy/1``, so every
    historical endpoint -- the whole reused half of Block A, the continued-SFT
    controls, and every checkpoint the finalist audit names -- was unreadable.
    This accepts the schemas that exist and keeps both guarantees that matter:
    ``strict=True`` on the parameter set, and the recorded state digest
    re-derived from the loaded weights. A model whose parameters do not match is
    a retrain, never a coercion.
    """
    import torch

    target = Path(path)
    require(target.is_file(), f"{target} is not readable; no checkpoint is substituted for it")
    restored = load_cpu(target, weights_only=True)
    schema = restored.get("schema_version")
    require(schema in CHECKPOINT_SCHEMAS,
            f"{target} carries checkpoint schema {schema!r}. This flight reads "
            f"{sorted(CHECKPOINT_SCHEMAS)}; an unknown schema is refused rather than guessed at.")
    require("state" in restored, f"{target} carries no 'state' tensors")
    require(restored.get("state_sha256") is not None,
            f"{target} has no verifiable state digest")
    require(state_dict_digest(restored["state"]) == restored["state_sha256"],
            f"{target} has corrupt checkpoint tensors")
    model.load_state_dict(restored["state"], strict=True)
    observed = state_digest(model)
    recorded = restored.get("state_sha256")
    require(recorded is None or observed == recorded,
            f"{target}: the loaded weights digest to {observed} and the checkpoint records "
            f"{recorded}. A checkpoint that does not reproduce its own digest is not trained "
            "from.")
    return {"path": str(target), "schema_version": schema,
            "kind": CHECKPOINT_SCHEMAS[schema],
            "state_sha256": observed, "file_sha256": paths.sha256_file(target),
            "update": restored.get("update"), "exposures": restored.get("exposures"),
            "metadata": restored.get("metadata"), "loaded_strict": True}


def checkpoint_metadata(path):
    """Read a checkpoint's header without instantiating a model."""
    import torch

    target = Path(path)
    require(target.is_file(), f"{target} is not readable")
    restored = load_cpu(target, weights_only=True)
    require(state_dict_digest(restored["state"]) == restored.get("state_sha256"),
            f"{target} has corrupt checkpoint tensors")
    return {"path": str(target), "schema_version": restored.get("schema_version"),
            "state_sha256": restored.get("state_sha256"), "update": restored.get("update"),
            "exposures": restored.get("exposures"),
            "file_sha256": paths.sha256_file(target)}


# ---------------------------------------------------------------------------
# the saved task stream, including its saved replay order
# ---------------------------------------------------------------------------

class SavedTaskStream:
    """The completed campaign's stream, read from disk rather than re-derived.

    It presents the same surface as ``her2_replay_streams.TaskStream`` -- the
    loop only calls ``batch`` and reads ``cycle_of_position`` -- so a reused arm
    and a fresh one go through exactly the same training code.
    """

    def __init__(self, *, seed, chosen_rows, rejected_rows, cycle_of_position, replay_order,
                 batch_rows, record):
        self.seed = int(seed)
        self.chosen_rows = np.asarray(chosen_rows, dtype=np.int64)
        self.rejected_rows = np.asarray(rejected_rows, dtype=np.int64)
        self.cycle_of_position = np.asarray(cycle_of_position)
        self.replay_order = np.asarray(replay_order, dtype=np.int64)
        self.batch_rows = int(batch_rows)
        self.record = dict(record)
        require(self.chosen_rows.size == self.rejected_rows.size == self.cycle_of_position.size,
                "the saved stream's three position arrays disagree about their length")

    @property
    def exposures(self):
        return int(self.chosen_rows.size)

    @property
    def updates(self):
        return self.exposures // self.batch_rows

    def batch(self, update):
        require(1 <= int(update) <= self.updates,
                f"Update {update!r} is outside 1..{self.updates} for this saved stream")
        start = (int(update) - 1) * self.batch_rows
        return (self.chosen_rows[start:start + self.batch_rows],
                self.rejected_rows[start:start + self.batch_rows])

    def document(self):
        identity = dict(self.record.get("identity") or {})
        return {"source": "saved", "seed": self.seed, "exposures": self.exposures,
                "updates": self.updates, "batch_rows": self.batch_rows,
                "pairing_seed": identity.get("pairing_seed"),
                "chosen_rows_sha256": paths.array_digest(self.chosen_rows),
                "rejected_rows_sha256": paths.array_digest(self.rejected_rows),
                "cycle_of_position_sha256": paths.array_digest(self.cycle_of_position),
                "replay_order_sha256": paths.array_digest(self.replay_order),
                "basis": ("the arrays the completed campaign trained on, verified by read_shard "
                          "against their recorded dtypes, shapes, order and content digests. "
                          "They are not re-derived and assumed equal.")}


def load_task_stream(historical_root, seed, *, batch_rows):
    """The saved chosen/rejected/cycle/replay-order arrays for one parent seed."""
    directory = Path(historical_root) / "streams" / f"seed{int(seed)}"
    arrays, record = paths.read_shard(directory, "task_stream")
    missing = sorted({"chosen_rows", "rejected_rows", "cycle_of_position", "replay_order"}
                     - set(arrays))
    require(not missing, f"{directory}: the saved task stream is missing {missing}")
    return SavedTaskStream(seed=seed, chosen_rows=arrays["chosen_rows"],
                           rejected_rows=arrays["rejected_rows"],
                           cycle_of_position=arrays["cycle_of_position"],
                           replay_order=arrays["replay_order"], batch_rows=batch_rows,
                           record=record)


def compare_stream(saved, derived):
    """Recorded versus re-derived digests, key by key, with no verdict implied."""
    recorded = dict(saved.record.get("identity") or {})
    keys = ("chosen_rows_sha256", "rejected_rows_sha256", "cycle_of_position_sha256",
            "pairing_seed", "exposures")
    comparison = {key: {"recorded": recorded.get(key), "derived": derived.get(key),
                        "matches": recorded.get(key) == derived.get(key)} for key in keys}
    return {"comparison": comparison,
            "all_match": all(entry["matches"] for entry in comparison.values()),
            "replay_order_sha256": paths.array_digest(saved.replay_order),
            "meaning": ("agreement shows the derivation rule is unchanged. The arrays this "
                        "flight actually trains on are the SAVED ones either way.")}


# ---------------------------------------------------------------------------
# the saved banks
# ---------------------------------------------------------------------------

def load_replay_bank(historical_root, seed, *, role="replay", parent_state_sha256=None,
                     expected_rows=None):
    """One saved parent bank: draws, parent scores and the frozen teacher cache.

    ``role`` is ``replay`` (100k, training) or ``monitor`` (10k independent
    draws). They are different populations and are never substituted for one
    another: the monitor bank is the development-preservation bank the pilots
    and the in-fit diagnostics read, and pointing those at the first rows of the
    training bank would measure preservation on rows the arm trained on.
    """
    directory = Path(historical_root) / "banks" / f"seed{int(seed)}" / str(role)
    arrays, record = paths.read_shard(directory, "bank")
    index = np.asarray(arrays["core_index"])
    if expected_rows is not None:
        require(index.shape[0] == int(expected_rows),
                f"{directory}: the saved {role} bank holds {index.shape[0]} rows and "
                f"{expected_rows} were declared")
    if parent_state_sha256 is not None:
        recorded = (record.get("identity") or {}).get("parent_state_sha256")
        require(recorded == parent_state_sha256,
                f"{directory}: this bank was drawn from parent state {recorded} and the policy "
                f"about to read it is {parent_state_sha256}. A replay bank from another parent "
                "silently redefines every preservation number computed with it.")
    return {"role": str(role), "directory": str(directory), "index": index,
            "parent_scores": np.asarray(arrays["parent_sum_log_probability"], dtype=np.float64),
            "sampler_scores": np.asarray(arrays["sampler_sum_log_probability"], dtype=np.float64),
            "teacher_probabilities_array": np.asarray(arrays["teacher_probabilities"]),
            "teacher_log_probabilities_array": np.asarray(arrays["teacher_log_probabilities"]),
            "record": record, "rows": int(index.shape[0]),
            "source": "saved",
            "basis": ("the completed campaign's own bank, verified by read_shard against its "
                      "recorded dtypes, shapes, order and content digests")}


def load_reference_cache(historical_root, seed, *, population, scaffold_prefix,
                         parent_checkpoint_sha256=None):
    """The saved frozen parent scores for the chosen and rejected populations.

    The cache covers the chosen rows followed by the rejected rows, in that
    order, so the core-order digest is checked against exactly that
    concatenation of the population this flight built. A cache whose order digest
    disagrees describes different rows, and every IPO margin computed against it
    would be a margin against the wrong reference.

    ``config_sha256`` in the saved identity is the completed campaign's
    configuration digest, which is a different file from this flight's by
    construction. It is reported rather than asserted equal; the things that
    would change the NUMBERS -- the parent weights, the row order, the prefix and
    the probability convention -- are all checked.
    """
    array_path = Path(historical_root) / "banks" / f"seed{int(seed)}" / "ipo_reference_cache.npy"
    sidecar = array_path.with_suffix(".json")
    require(array_path.is_file() and sidecar.is_file(),
            f"{array_path} (or its sidecar) is not readable; the saved reference cache is not "
            "substituted by a rescore, because a rescore is a different artifact")
    document = paths.read_json(sidecar)
    require(document.get("schema_version") == REFERENCE_CACHE_SCHEMA,
            f"{sidecar}: unsupported reference-cache schema {document.get('schema_version')!r}")
    identity = dict(document["identity"])
    values = np.load(array_path)
    # Version 1 hashes contiguous float64 score bytes. Shard-array hashes also
    # frame dtype and shape, and are a different digest convention.
    require(reference_values_digest(values) == identity.get("values_sha256"),
            f"{array_path}: the cached values do not match the digest recorded with them")
    index = np.concatenate([population.chosen_index, population.rejected_index])
    require(int(values.size) == int(index.shape[0]),
            f"{array_path}: the cache holds {values.size} values and this flight's chosen+rejected "
            f"population is {index.shape[0]} rows")
    observed_order = core_digest(index)
    require(observed_order == identity.get("core_order_sha256"),
            f"{array_path}: the cache was built over core order {identity.get('core_order_sha256')}"
            f" and this flight's population is {observed_order}. These are different rows.")
    import hashlib
    prefix_digest = hashlib.sha256(str(scaffold_prefix).encode()).hexdigest()
    require(prefix_digest == identity.get("scaffold_prefix_sha256"),
            f"{array_path}: the cache was built under a different conditioning prompt")
    if parent_checkpoint_sha256 is not None:
        require(identity.get("checkpoint_sha256") == parent_checkpoint_sha256,
                f"{array_path}: the cache was built from checkpoint "
                f"{identity.get('checkpoint_sha256')} and this arm's parent is "
                f"{parent_checkpoint_sha256}")
    split = int(population.chosen_index.shape[0])
    values = np.asarray(values, dtype=np.float64)
    return {"chosen": values[:split], "rejected": values[split:], "identity": identity,
            "source": "saved", "path": str(array_path), "rows": int(values.size),
            "config_sha256_note": ("the saved config_sha256 names the completed campaign's "
                                   "configuration. It is reported, not asserted equal to this "
                                   "flight's; everything that would change the cached numbers is "
                                   "checked."),
            "basis": "the completed campaign's own frozen reference scores"}


def load_endpoint_evaluation(historical_root, trajectory, update):
    """A historical endpoint's saved validation/generation arrays, if they exist."""
    # The same convention as load_monitor_scores: a path relative to the
    # historical root, never a bare name to which a segment is re-added here.
    directory = (Path(historical_root) / str(trajectory) / "endpoints"
                 / f"update{int(update)}")
    if not (directory / "evaluation.complete.json").is_file():
        return None
    arrays, record = paths.read_shard(directory, "evaluation")
    return {"trajectory": str(trajectory), "update": int(update),
            "validation_mean_log_probability":
                np.asarray(arrays["validation_mean_log_probability"], dtype=np.float64),
            "generation_core_index": np.asarray(arrays["generation_core_index"]),
            "generation_sum_log_probability":
                np.asarray(arrays["generation_sum_log_probability"], dtype=np.float64),
            "record": record,
            "density_note": ("validation_mean_log_probability is ell/10. Multiply by 10 before "
                             "using it as a density; the ranking is unchanged either way.")}


def load_monitor_scores(historical_root, trajectory, *, check=1, update=25):
    """One saved monitor-score vector, for the update-25 parity comparison.

    ``trajectory`` is a path RELATIVE TO THE HISTORICAL ROOT -- the form the
    config declares (``trajectories/ipo_lambda0_seed{seed}``) and the form the
    parity check already uses to open the same arm's ``updates.jsonl``. Adding a
    ``trajectories`` segment here doubled it, so every saved vector read as
    absent and the parity gate passed on the task-loss scalars alone while its
    record stated that the score vector had been compared.
    """
    target = (Path(historical_root) / str(trajectory) / "monitor_scores"
              / f"check_{int(check):05d}_update{int(update)}.npz")
    if not target.is_file():
        return None
    arrays = paths.read_arrays(target)
    return {"path": str(target), "arrays": {key: np.asarray(value)
                                            for key, value in arrays.items()}}


def load_historical_cnn(path, *, expected_sha256=None):
    """A hash-verified historical CNN checkpoint, as state plus its own provenance."""
    import torch

    target = Path(path)
    require(target.is_file(), f"{target} is not readable; the historical CNN is not re-fitted")
    digest = paths.sha256_file(target)
    require(expected_sha256 is None or digest == expected_sha256,
            f"{target} hashes {digest} and the record expects {expected_sha256}")
    restored = torch.load(target, map_location="cpu", weights_only=True)
    require(restored.get("schema_version") == CNN_SCHEMA,
            f"{target} carries schema {restored.get('schema_version')!r}, not {CNN_SCHEMA!r}")
    return {"path": str(target), "file_sha256": digest, "state": restored["state"],
            "identity": restored.get("identity"), "report": restored.get("report"),
            "selection_provenance": ("the completed campaign's own selection, on its own "
                                     "population. It is reused with that provenance, not "
                                     "reselected here.")}


def reuse_record(**blocks):
    """One artifact-level statement of what was loaded rather than rebuilt."""
    return {"schema_version": NF_SCHEMA, "record_kind": "historical_reuse",
            **{key: value for key, value in blocks.items()},
            "rule": ("every entry above was READ from the completed campaign and verified. "
                     "Nothing under the historical root is written, and no artifact is "
                     "re-derived and then described as reused.")}

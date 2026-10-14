"""Strict, outcome-free query and benchmark transport contracts."""
from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.objectives import whole_core_scores
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import EndpointConstruct
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.views import mask_complete


def canonical_bytes(value):
    """Match benchmark JSON hashing, including normalization of signed zero."""
    def clean(x):
        if isinstance(x, float):
            if not math.isfinite(x):
                raise ValueError("JSON values must be finite")
            return 0.0 if x == 0 else x
        if isinstance(x, dict):
            if any(not isinstance(k, str) for k in x):
                raise TypeError("JSON object keys must be strings")
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if x is None or isinstance(x, (str, int, bool)):
            return x
        raise TypeError(f"unsupported JSON value {type(x).__name__}")
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_keys(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{where}: expected exactly {sorted(keys)}")


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))
    canonical_bytes(value)
    return value


def positive_count(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def seed_value(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


def text_value(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True)
class Query:
    case_id: str
    core: str
    construct_id: str


@dataclass(frozen=True)
class QueryPanel:
    protocol_id: str
    assay_id: str
    heavy_prefix: str
    heavy_suffix: str
    light: str
    antigen: str
    queries: tuple[Query, ...]

    def __post_init__(self):
        object.__setattr__(self, "queries", tuple(self.queries))
        text_value(self.protocol_id, "protocol_id")
        text_value(self.assay_id, "assay_id")
        text_value(self.heavy_prefix, "heavy_prefix")
        text_value(self.heavy_suffix, "heavy_suffix")
        self.construct("A" * 10)
        if not self.queries or len({q.case_id for q in self.queries}) != len(self.queries):
            raise ValueError("queries must have nonempty, unique case IDs")
        for query in self.queries:
            text_value(query.case_id, "case_id")
            self.construct(query.core)
            if query.construct_id != self.construct_id(query.core):
                raise ValueError("query construct_id does not match literal scaffold and core")

    def construct(self, core):
        return EndpointConstruct(self.heavy_prefix, core, self.heavy_suffix, self.light, self.antigen)

    @property
    def scaffold_id(self):
        return digest({"kind": "scaffold", "segments": [self.heavy_prefix, self.heavy_suffix, self.light]})

    def construct_id(self, core):
        return digest({"kind": "construct", "scaffold": self.scaffold_id, "core": core})

    @property
    def antigen_sha256(self):
        return hashlib.sha256(self.antigen.encode("ascii")).hexdigest()

    def document(self):
        return {"schema": "trastuzumab-queries/1", **asdict(self)}

    @classmethod
    def from_document(cls, document):
        require_keys(document, ("schema", "protocol_id", "assay_id", "heavy_prefix",
                               "heavy_suffix", "light", "antigen", "queries"), "query panel")
        if document["schema"] != "trastuzumab-queries/1":
            raise ValueError("unsupported query schema")
        queries = []
        if not isinstance(document["queries"], list):
            raise ValueError("queries must be a JSON list")
        for row in document["queries"]:
            require_keys(row, ("case_id", "core", "construct_id"), "query")
            queries.append(Query(**row))
        values = {k: v for k, v in document.items() if k not in ("schema", "queries")}
        return cls(**values, queries=tuple(queries))

    def masked(self, core, arm):
        if arm not in ("paired", "abonly"):
            raise ValueError("arm must be paired or abonly")
        c = self.construct(core)
        sequences = {"H": c.full_heavy(), "L": c.light}
        if arm == "paired":
            sequences["Ag"] = c.Ag
        coords = tuple(("H", len(c.heavy_prefix) + i) for i in range(10))
        return mask_complete(sequences, coords)


@contextmanager
def evaluating(policy):
    training = policy.training
    try:
        policy.eval()
        with torch.no_grad():
            yield
    finally:
        policy.train(training)


def inference_metadata(policy, encoder, panel, checkpoint_identity, arm):
    canonical_bytes(checkpoint_identity)
    if encoder.encoder_identity != policy.encoder_identity:
        raise ValueError("encoder and policy identity differ")
    view, _ = panel.masked("A" * 10, arm)
    provider = DynamicContextProvider(encoder, view)
    policy.preflight([view.without_ag()], [provider])
    return {"schema": "trastuzumab-inference/1", "checkpoint": checkpoint_identity,
            "model": policy.identity(), "encoder_runtime": encoder.runtime_identity(),
            "queries_sha256": digest(panel.document()), "scaffold_id": panel.scaffold_id,
            "assay_id": panel.assay_id, "antigen_sha256": panel.antigen_sha256,
            "context_arm": arm, "query_roles_or_outcomes_available": False}


def score_queries(policy, encoder, panel, *, checkpoint_identity, arm,
                  order_seed=0, orders=4, scorer_id="whole-core"):
    seed_value(order_seed)
    positive_count(orders, "orders")
    text_value(scorer_id, "scorer_id")
    metadata = inference_metadata(policy, encoder, panel, checkpoint_identity, arm)
    base, _ = panel.masked("A" * 10, arm)
    coords = tuple(base.hidden_sites())
    rng = random.Random(order_seed)
    schedules = [tuple(rng.sample(coords, len(coords))) for _ in range(orders)]
    metadata["readout"] = {"name": "whole_core_mean_sequential_log_probability",
                           "canonical_residues": 20, "order_seed": order_seed,
                           "orders": [[list(c) for c in order] for order in schedules],
                           "averaging": "arithmetic mean over complete orders; nats per site",
                           "is_sampler_likelihood": False, "is_affinity": False}
    cache = {}
    with evaluating(policy):
        for core in sorted({q.core for q in panel.queries}):
            try:
                combined, targets = panel.masked(core, arm)
                provider = DynamicContextProvider(encoder, combined)
                values = [whole_core_scores(policy, provider, combined.without_ag(), [targets],
                                            order, state_microbatch=1)[0] for order in schedules]
                score = float(torch.stack(values).mean().cpu())
                if not math.isfinite(score):
                    raise FloatingPointError("nonfinite core score")
                cache[core] = (score, None)
            except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
                cache[core] = (None, f"{type(error).__name__}: {error}")
    result = {"schema": "her2-benchmark/1", "record_kind": "score_table",
              "scorer_id": scorer_id, "readout": "whole_core_mean_sequential_log_probability",
              "orientation": "higher_is_better",
              "provenance": {"queries": digest(panel.document()), "metadata": digest(metadata)},
              "scores": [{"case_id": q.case_id, "score": cache[q.core][0], "failure": cache[q.core][1]}
                         for q in sorted(panel.queries, key=lambda q: q.case_id)]}
    return result, metadata


def validate_generation_request(panel, *, arm_id, sampling_seeds, attempts_per_seed,
                                steps=20, strategy="reparam-uncond-deterministic-linear",
                                reference_core=None, editable_count=10):
    """Validate an export request before loading models or opening checkpoints."""
    positive_count(attempts_per_seed, "attempts_per_seed")
    positive_count(steps, "steps")
    positive_count(editable_count, "editable_count")
    if editable_count > 10:
        raise ValueError("editable_count cannot exceed core width 10")
    if reference_core is None:
        raise ValueError("generation requires an explicit canonical reference_core")
    panel.construct(reference_core)
    text_value(arm_id, "arm_id")
    seeds = tuple(sorted(seed_value(s) for s in sampling_seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("sampling seeds must be nonempty and unique")
    # Validate native strategy before recording any attempts.
    from smallAntibodyGen.models.dplm_policy import DiffusionSettings
    if not isinstance(strategy, str) or strategy != strategy.strip():
        raise ValueError("decoding strategy must be a canonical string without whitespace")
    DiffusionSettings(decode_steps=steps, decoding_strategy=strategy)
    selection = strategy.split("-")[2]
    if selection.startswith("stochastic"):
        scale = float(selection[len("stochastic"):])
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("stochastic decoding scale must be finite and nonnegative")
    return seeds


def generate_attempts(policy, encoder, panel, *, checkpoint_identity, arm,
                      arm_id, sampling_seeds, attempts_per_seed, steps=20,
                      strategy="reparam-uncond-deterministic-linear",
                      reference_core=None, editable_count=10):
    seeds = validate_generation_request(panel, arm_id=arm_id, sampling_seeds=sampling_seeds,
                                        attempts_per_seed=attempts_per_seed, steps=steps,
                                        strategy=strategy, reference_core=reference_core,
                                        editable_count=editable_count)
    metadata = inference_metadata(policy, encoder, panel, checkpoint_identity, arm)
    metadata["generation"] = {"arm_id": arm_id, "sampling_seeds": list(seeds),
                              "attempts_per_seed": attempts_per_seed, "steps": steps,
                              "strategy": strategy, "rng": "sha256-seed-index/1",
                              "selection": "all raw attempts; no redraws or deduplication",
                              "reference_core": reference_core, "editable_count": editable_count,
                              "edit_budget_semantics": "uniform subset of editable positions; substitutions are not forced",
                              "temperature": 1.0, "tokens": "canonical20",
                              "position_rule": "sha256-seed-index-purpose uniform subset"}
    construct = panel.construct(reference_core)
    sequences = {"H": construct.full_heavy(), "L": construct.light}
    if arm == "paired":
        sequences["Ag"] = construct.Ag
    coords = tuple(("H", len(construct.heavy_prefix) + i) for i in range(10))
    attempts = []
    with evaluating(policy):
        for seed in seeds:
            for index in range(attempts_per_seed):
                # Same per-attempt stochastic stream across model arms.
                keyed = int(digest({"seed": seed, "index": index, "purpose": "generation"})[:16], 16) % (2**63)
                generator = torch.Generator(device=policy.device).manual_seed(keyed)
                position_seed = int(digest({"seed": seed, "index": index, "purpose": "edit_positions"})[:16], 16)
                editable = tuple(sorted(random.Random(position_seed).sample(range(10), editable_count)))
                combined, _ = mask_complete(sequences, [coords[i] for i in editable])
                query = combined.without_ag()
                provider = DynamicContextProvider(encoder, combined)
                core, failure = None, None
                try:
                    result = policy.generate(query, provider, steps=steps, strategy=strategy, generator=generator)
                    core = "".join(result.token(c) for c in coords)
                except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
                    failure = f"{type(error).__name__}: {error}"
                attempts.append({"attempt_id": f"{panel.protocol_id}/{arm_id}/{seed}/{index}",
                                 "arm": arm_id, "seed": seed, "index": index,
                                 "scaffold_id": panel.scaffold_id, "raw_core": core,
                                 "failure_reason": failure,
                                 "source": {"metadata_sha256": digest(metadata), "steps": steps,
                                            "rng_seed": keyed, "reference_core": reference_core,
                                            "editable_positions": list(editable),
                                            "hamming_distance": None if core is None else sum(a != b for a, b in zip(core, reference_core))}})
    payload = {"protocol_id": panel.protocol_id, "scaffold_id": panel.scaffold_id,
               "budgets": {arm_id: {str(seed): attempts_per_seed for seed in seeds}},
               "attempts": attempts}
    return {**payload, "attempt_set_sha256": digest(payload)}, metadata


def write_export(directory, *, payload_name, payload, metadata):
    """Publish a new export only when its manifest is written last."""
    if payload_name not in ("scores.json", "attempts.json"):
        raise ValueError("unsupported payload filename")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, document in ((payload_name, payload), ("metadata.json", metadata)):
        content = canonical_bytes(document) + b"\n"
        path = directory / name
        with path.open("xb") as stream:
            stream.write(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    manifest = {"schema": "trastuzumab-export/1", "files": files,
                "checkpoint": metadata["checkpoint"], "queries_sha256": metadata["queries_sha256"]}
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(canonical_bytes(manifest) + b"\n")
    return manifest

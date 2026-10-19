"""Preregistered downstream study (M4): an outcome-free plan, strict export validation, final-horizon selection.

The world is real wherever that is cheap: actual tiny ``StageTrainer`` runs (a direct-release arm, and
parents exported from actual midtraining with and without the detached functional arm), actual
checkpoints, the existing ``load_post`` loader for the opaque label pins, and score exports written by the
actual ``artifacts.score_queries`` and ``artifacts.write_export``.  Only the per-core numbers are
controlled (``artifacts.whole_core_scores`` is replaced by a table) so that every metric has a known
answer that is derived here independently of ``study.py``.
"""
from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import math
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import artifacts, cli, inputs, study
from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    Query, QueryPanel, canonical_bytes, digest, file_digest, read_json, score_queries, write_export,
)
from smallAntibodyGen.experiments.trastuzumab_poc.parent import export_parent, load_parent
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest, select_budget,
)
from smallAntibodyGen.experiments.trastuzumab_poc.runner import StageTrainer, TrainingConfig
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import qualify_transfer
from smallAntibodyGen.tests.test_trastuzumab_functional_data import imported_names
from smallAntibodyGen.tests.test_trastuzumab_functional_runner import (
    SMALL_VALUES, build_trainer, fresh_models, functional_run, small_world, training_config,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_transfer import FAMILIES, SOURCE, post_fusion

ROOT = Path(__file__).resolve().parents[3]
BUDGET_SEED = "study-budget"
SIZES = (2, 4)
RATES = (0.001, 0.003)
SEEDS = (3, 5)
BOUNDARIES = (1, 2)
PROSPECTIVE_RATE = 0.003
TEMPLATE = {
    "stage": "post", "arm": "paired", "total_updates": 2, "examples_per_update": 1, "weight_decay": 0.01,
    "gradient_clip": 1.0, "schedule": "constant", "reconstruction_weight": 1.0, "ordering_weight": 1.0,
    "ordering_margin": 0.1, "minimum_ordering_endpoint_coverage": 0.0, "span_weight": 0.0,
    "full_region_weight": 0.0}
QUERY = {"protocol_id": "synthetic-protocol", "scorer_id": "tiny-whole-core", "context_arm": "paired",
         "order_seed": 7, "orders": 2}
POOL_CORES = ("ACDEFGHIKL", "LMNPQRSTVW", "AAAAAAAAAA", "CCCCCCCCCC")
DEV_IDS = ("d0", "d1", "d2", "d3")
DEV_CORES = dict(zip(DEV_IDS, ("DDDDDDDDDD", "EEEEEEEEEE", "FFFFFFFFFF", "GGGGGGGGGG")))
# d1 and d2 are tied: exactly five of the six pairs are comparable.
DEV_VALUES = {"d0": 10.0, "d1": 20.0, "d2": 20.0, "d3": 40.0}
# Score vectors over (d0, d1, d2, d3) and their ordinal concordance, computed by hand over the five pairs.
LEVELS = {
    "perfect": ((1.0, 2.0, 3.0, 4.0), 1.0),
    "ignored_tie": ((1.0, 3.0, 2.0, 4.0), 1.0),     # the tied measured pair is excluded, whatever its order
    "tie": ((1.0, 1.0, 3.0, 4.0), 0.9),              # one prediction tie: half credit
    "swap": ((2.0, 1.0, 3.0, 4.0), 0.8),
    "worse": ((3.0, 1.0, 2.0, 4.0), 0.6),
    "reversed": ((4.0, 3.0, 2.0, 1.0), 0.0),
}
A_STAGE = "mid-denoising-only"
D_STAGE = "mid-functional-detached"
# Final-horizon level of (arm, budget) as a function of (rate index, origin index, seed index).
FINAL = {
    ("release", 2): lambda r, o, s: "swap" if r == 0 else ("perfect", "reversed")[s],
    ("release", 4): lambda r, o, s: "worse" if r == 0 else "swap",
    (A_STAGE, 2): lambda r, o, s: ("worse", "perfect")[o] if r == 0 else ("perfect", "swap")[o],
    (A_STAGE, 4): lambda r, o, s: "perfect" if r == 0 else "swap",
    (D_STAGE, 2): lambda r, o, s: "tie",
    (D_STAGE, 4): lambda r, o, s: "swap" if r == 0 else "perfect",
}
# (arm, budget) -> (selected rate, mean final metric), derived by hand from FINAL and LEVELS.
EXPECTED = {("release", 2): (0.001, 0.8), ("release", 4): (0.003, 0.8),
            (A_STAGE, 2): (0.003, 0.9), (A_STAGE, 4): (0.001, 1.0),
            (D_STAGE, 2): (0.001, 0.9), (D_STAGE, 4): (0.003, 1.0)}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def write(path, document):
    path.write_bytes(canonical_bytes(document))
    return path


def plain(value):
    return json.loads(canonical_bytes(value))


def hexn(number):
    return f"{number:064x}"


def put(container, keys, value):
    for key in keys[:-1]:
        container = container[key]
    container[keys[-1]] = value


def pretty(document):
    """A hand-edited-looking file: UTF-8 BOM, indentation, trailing newline."""
    return b"\xef\xbb\xbf" + json.dumps(document, indent=2).encode("utf-8") + b"\n"


def sha256_of(data):
    return hashlib.sha256(data).hexdigest()


def reseen(checkpoint, **lists):
    """Replace observed-ID lists of a recorded checkpoint and keep their ``unique`` counts matching, so that a
    test violates only the rule it names."""
    for key, values in lists.items():
        checkpoint["progress"]["seen"][key] = list(values)
        checkpoint["progress"]["unique"][key] = len(values)


def replaced_after_first_read(patch, path, replacement):
    """The file is replaced on disk right after its FIRST read-open: that reader is served the original bytes
    and every later reader sees ``replacement``.  Returns a counter dict (``reads`` is 1 once it happened)."""
    real_open, target, state = Path.open, Path(path).resolve(), {"reads": 0}

    def opener(self, mode="r", *args, **kwargs):
        if "r" in mode and "+" not in mode and state["reads"] == 0 and self.resolve() == target:
            state["reads"] += 1
            with real_open(self, "rb") as stream:
                original = stream.read()
            target.write_bytes(replacement)
            if "b" in mode:
                return io.BytesIO(original)
            return io.StringIO(original.decode(kwargs.get("encoding") or "utf-8"))
        return real_open(self, mode, *args, **kwargs)

    patch.setattr(Path, "open", opener)
    return state


def reference_concordance(scores, measured):
    """The definition, pair by pair: tied measured pairs excluded, tied scores count one half."""
    total = concordant = 0.0
    for (s1, m1), (s2, m2) in itertools.combinations(list(zip(scores, measured)), 2):
        if m1 == m2:
            continue
        total += 1
        product = (s1 - s2) * (m1 - m2)
        concordant += 1.0 if product > 0 else 0.5 if s1 == s2 else 0.0
    return None if not total else concordant / total


# ---------------------------------------------------------------------------
# Documents: manifest, specification, development labels
# ---------------------------------------------------------------------------

def manifest_document(*, antigen="HIK", prefix="AC", suffix="DE", dev_assay="SPR", dev_ids=DEV_IDS):
    def entry(endpoint_id, core, role, assay="SPR"):
        return {"endpoint_id": endpoint_id, "role": role, "neighbourhood_group": f"group-{endpoint_id}",
                "assay_id": assay, "endpoint_semantics": "synthetic ordering",
                "construct": {"heavy_prefix": prefix, "core": core, "heavy_suffix": suffix, "light": "FG",
                              "Ag": antigen}}
    entries = [entry(f"p{i}", core, "post_pool") for i, core in enumerate(POOL_CORES)]
    entries += [entry(eid, DEV_CORES[eid], "development", dev_assay) for eid in dev_ids]
    entries.append(entry("r0", "HHHHHHHHHH", "reserved"))
    return {"schema": "trastuzumab-endpoints/1", "entries": entries}


def manifest_from(document):
    return EndpointManifest(tuple(inputs.record(EndpointEntry, row, {
        "construct": lambda x: inputs.record(EndpointConstruct, x)}) for row in document["entries"]))


def make_spec(*, pins, identity, arms):
    return {
        "schema": study.SPEC_SCHEMA, "study_id": "synthetic-study",
        "development": {"assay_id": "SPR", "orientation": "higher_better"}, "query": dict(QUERY),
        "recipe": {"training": dict(TEMPLATE), "fusion": asdict(post_fusion()), "boundaries": list(BOUNDARIES),
                   "learning_rates": list(RATES), "prospective_learning_rate": PROSPECTIVE_RATE,
                   "post_seeds": list(SEEDS)},
        "budgets": {"budget_seed": BUDGET_SEED, "entries": pins}, "identity": identity, "arms": arms}


def synthetic_spec():
    """A valid specification over opaque, well-formed hashes: enough for every outcome-free check."""
    encoder = {"source_sha256": hexn(1), "checkpoint_sha256": hexn(2), "extraction_layer": "layer_1",
               "precision": "float32", "tokenizer": "synthetic-test-only", "eval_mode": "true"}
    identity = {
        "release_base_sha256": hexn(10), "source_sha256": hexn(11),
        "encoder": {"identity": dict(encoder),
                    "runtime": {"encoder": dict(encoder), "storage_dtype": "float32", "compute_dtype": "float32",
                                "native_batch_size": 1, "max_tokens": 128, "normalization_implementations": []}},
        "execution": {"hardware": {"device": "cpu", "cuda": False}, "software": {"python": "3", "torch": "2"},
                      "parameters": {"total": 1, "trainable": 1, "frozen_encoder": 1}}}

    def origin(index, base):
        return {"origin_id": f"mid-{index}", "parent_sha256": hexn(base), "parent_checkpoint_sha256": hexn(base + 1),
                "parent_provenance_sha256": hexn(base + 2)}

    arms = [{"arm_id": "release", "kind": "release", "functional_mode": None, "origins": [{"origin_id": "release"}]},
            {"arm_id": A_STAGE, "kind": "mid", "functional_mode": None, "origins": [origin(0, 100), origin(1, 110)]},
            {"arm_id": D_STAGE, "kind": "mid", "functional_mode": "detached", "origins": [origin(0, 200)]}]
    pins = [{"budget_n": n, "selected_data_sha256": hexn(300 + n), "selected_labels_sha256": hexn(400 + n)}
            for n in SIZES]
    return make_spec(pins=pins, identity=identity, arms=arms)


SYNTHETIC_MANIFEST_SHA = hexn(7)


def synthetic_plan(spec=None, document=None):
    return study.build_plan(synthetic_spec() if spec is None else spec,
                            manifest_from(manifest_document() if document is None else document),
                            SYNTHETIC_MANIFEST_SHA)


def reseal(plan):
    plan["plan_sha256"] = digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def labels_document(plan, values=None, **changes):
    values = DEV_VALUES if values is None else values
    development = plan["development"]
    document = {"schema": study.LABELS_SCHEMA, "plan_sha256": plan["plan_sha256"],
                "assay_id": development["assay_id"], "orientation": development["orientation"],
                "labels": [{"endpoint_id": eid, "value": value, "orientation": development["orientation"],
                            "assay_id": development["assay_id"]} for eid, value in sorted(values.items())]}
    document.update(changes)
    return document


def labels_opened():
    raise AssertionError("the development labels were opened before every export validated")


# ---------------------------------------------------------------------------
# The world: real trainers, real checkpoints, real score exports
# ---------------------------------------------------------------------------

def table_scorer(by_core):
    """Stands in for ``whole_core_scores``: one controlled number (or failure) per core."""
    def scoring(policy, provider, view, targets, order, **kwargs):
        value = by_core["".join(residue for _, residue in targets[0].items)]
        if isinstance(value, BaseException):
            raise value
        return torch.tensor([value])
    return scoring


def export(world, trial, boundary, directory, *, vector, panel=None, provenance=None, arm="paired",
           order_seed=QUERY["order_seed"], orders=QUERY["orders"], scorer_id=QUERY["scorer_id"],
           edit_metadata=None, edit_payload=None, overrides=None):
    """Write one export through the actual ``score_queries`` and ``write_export``; returns the manifest hash."""
    panel = world.panel if panel is None else panel
    provenance = copy.deepcopy(world.provenance[(trial["trial_id"], boundary)] if provenance is None else provenance)
    by_core = {world.dev_cores[eid]: value for eid, value in zip(DEV_IDS, vector)}
    by_core.update(overrides or {})
    scorer = world.scorers[(trial["arm_id"], trial["origin_id"])]
    with mock.patch.object(artifacts, "whole_core_scores", table_scorer(by_core)):
        payload, metadata = score_queries(scorer, world.encoder, panel, checkpoint_identity=provenance, arm=arm,
                                          order_seed=order_seed, orders=orders, scorer_id=scorer_id)
    if edit_metadata is not None:
        edit_metadata(metadata)
        payload["provenance"]["metadata"] = digest(metadata)
    if edit_payload is not None:
        edit_payload(payload)
    write_export(directory, payload_name="scores.json", payload=payload, metadata=metadata)
    return file_digest(directory / "manifest.json")


def level_of(trial, boundary):
    rate, seed = RATES.index(trial["learning_rate"]), SEEDS.index(trial["post_seed"])
    origin = 0 if trial["origin_id"] == "release" else int(trial["origin_id"].rsplit("-", 1)[1])
    arm, n = trial["arm_id"], trial["budget_n"]
    if boundary == 1:
        # The prospective rate leads at the first boundary of one cell and loses by the final horizon.
        if (arm, n) == ("release", 2):
            return "worse" if rate == 0 else "perfect"
        return "swap"
    return FINAL[(arm, n)](rate, origin, seed)


def build_world(directory):
    manifest_path = write(directory / "endpoints.json", manifest_document())
    manifest = inputs.load_endpoint_manifest(manifest_path)
    manifest_sha = file_digest(manifest_path)
    # The opaque label pins come from the existing loader, as authorized label preparation would obtain them.
    restricted, data_identity, pins = {}, {}, []
    for n in SIZES:
        budget = select_budget(manifest, n, BUDGET_SEED)
        labels = {"schema": "trastuzumab-selected-labels/1", "selection_digest": budget.selection_digest,
                  "measurements": [{"endpoint_id": eid, "value": 1.0 + index, "orientation": "higher_better",
                                    "assay_id": "SPR"} for index, eid in enumerate(budget.selected_ids)]}
        labels_path = write(directory / f"labels-{n}.json", labels)
        restricted[n], data_identity[n] = inputs.load_post(manifest_path, labels_path, budget_n=n,
                                                           budget_seed=BUDGET_SEED)
        pins.append({"budget_n": n, "selected_data_sha256": data_identity[n]["selected_data"],
                     "selected_labels_sha256": data_identity[n]["selected_labels_sha256"]})
    source = inputs.source_identity()

    # Actual midtraining, with and without the detached functional arm, and the parents exported from it.
    _, _, population, qualified = small_world()
    _, encoder = fresh_models("lora")
    parents = {}
    for arm_id, mode, seeds in ((A_STAGE, None, (11, 12)), (D_STAGE, "detached", (13,))):
        for index, seed in enumerate(seeds):
            certificate = qualify_transfer(population, manifest, functional=None if mode is None else qualified,
                                           **FAMILIES)
            functional = None if mode is None else functional_run(qualified, SMALL_VALUES["ordered"],
                                                                  gradient_mode=mode)
            policy, _ = fresh_models("lora")
            mid = build_trainer(policy, encoder, population, training_config(total_updates=2, seed=seed),
                                functional=functional, certificate=certificate)
            mid.run()
            saved = ck.save_checkpoint(mid, directory / f"mid-{arm_id}-{index}.pt")
            loaded, _ = fresh_models("lora")
            provenance = ck.load_inference_checkpoint(loaded, saved["path"], encoder=encoder,
                                                      expected_sha256=saved["sha256"], source_identity=SOURCE)
            path = directory / f"parent-{arm_id}-{index}.pt"
            parents[(arm_id, f"mid-{index}")] = SimpleNamespace(
                path=path, sha=export_parent(loaded, path, provenance=provenance)["sha256"],
                checkpoint_sha=saved["sha256"], provenance_sha=digest(provenance))

    def make_policy(origin):
        if origin is None:
            return fresh_models("lora")[0]
        return load_parent(origin.path, expected_sha256=origin.sha, encoder_identity=encoder.encoder_identity,
                           fusion_config=post_fusion(), device="cpu", seed=5)

    def initialization(policy, origin):
        if origin is None:
            return {"kind": "release", "base": policy.base_identity}
        return {"kind": "parent", "sha256": origin.sha, "base": policy.base_identity}

    def post_trainer(origin, n, rate, seed):
        policy = make_policy(origin)
        config = TrainingConfig(**TEMPLATE, learning_rate=rate, seed=seed)
        return StageTrainer(policy, encoder, config, post_data=restricted[n], data_identity=data_identity[n],
                            source_identity=source, initialization_identity=initialization(policy, origin))

    # The declared execution contract and encoder come from an actual run's recorded identity.
    probe = post_trainer(None, SIZES[0], RATES[0], SEEDS[0]).identity
    identity = plain({
        "release_base_sha256": digest(probe["model"]["base"]), "source_sha256": source["sha256"],
        "encoder": {"identity": probe["model"]["encoder"], "runtime": probe["encoder_runtime"]},
        "execution": {key: probe["runtime"][key] for key in ("hardware", "software", "parameters")}})
    arms = [{"arm_id": "release", "kind": "release", "functional_mode": None, "origins": [{"origin_id": "release"}]}]
    for arm_id, mode in ((A_STAGE, None), (D_STAGE, "detached")):
        arms.append({"arm_id": arm_id, "kind": "mid", "functional_mode": mode, "origins": [
            {"origin_id": origin_id, "parent_sha256": p.sha, "parent_checkpoint_sha256": p.checkpoint_sha,
             "parent_provenance_sha256": p.provenance_sha}
            for (name, origin_id), p in sorted(parents.items()) if name == arm_id]})
    spec = make_spec(pins=pins, identity=identity, arms=arms)
    plan = study.build_plan(spec, manifest, manifest_sha)

    world = SimpleNamespace(
        dir=directory, manifest=manifest, manifest_sha=manifest_sha, spec=spec, plan=plan, encoder=encoder,
        spec_document={**spec, "endpoint_manifest_path": "endpoints.json"}, parents=parents, source=source,
        panel=QueryPanel.from_document(plan["queries"]), dev_cores=dict(DEV_CORES), provenance={}, scorers={})
    write(directory / "spec.json", world.spec_document)

    # One actual post run per trial; a checkpoint at every declared boundary.
    for trial in plan["trials"]:
        origin = parents.get((trial["arm_id"], trial["origin_id"]))
        trainer = post_trainer(origin, trial["budget_n"], trial["learning_rate"], trial["post_seed"])
        recorded = plain({key: trainer.identity["runtime"][key] for key in ("hardware", "software", "parameters")})
        assert recorded == identity["execution"], "every tiny arm must share one execution contract"
        for boundary in BOUNDARIES:
            trainer.run(stop_after_updates=boundary)
            saved = ck.save_checkpoint(trainer, directory / "checkpoints" / f"{trial['trial_id'][:20]}-{boundary}.pt")
            world.provenance[(trial["trial_id"], boundary)] = {
                "checkpoint_sha256": saved["sha256"], "training": copy.deepcopy(trainer.identity),
                "progress": copy.deepcopy(trainer.progress())}
        # A scoring policy of the same identity as the trained one, as the ``score`` command would load.
        if (trial["arm_id"], trial["origin_id"]) not in world.scorers:
            world.scorers[(trial["arm_id"], trial["origin_id"])] = make_policy(origin)
    # The default results index: every trial at every boundary has its own actual export.
    records = []
    for index, trial in enumerate(plan["trials"]):
        for boundary in BOUNDARIES:
            name = f"exports/t{index:03d}-b{boundary}"
            sha = export(world, trial, boundary, directory / name, vector=LEVELS[level_of(trial, boundary)][0])
            records.append({"trial_id": trial["trial_id"], "boundary": boundary, "status": "exported",
                            "export_path": name, "manifest_sha256": sha})
    world.results = {"schema": study.RESULTS_SCHEMA, "plan_sha256": plan["plan_sha256"], "records": records}
    write(directory / "results.json", world.results)
    return world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return build_world(tmp_path_factory.mktemp("study"))
    finally:
        torch.set_num_threads(previous)


class Study:
    """A mutable copy of the default results index plus the exports one test rewrites."""

    def __init__(self, world, tmp_path, plan=None):
        self.world, self.tmp, self.plan = world, tmp_path, world.plan if plan is None else plan
        self.results = copy.deepcopy(world.results)
        self.results["plan_sha256"] = self.plan["plan_sha256"]
        self.rewrites = 0

    def trial(self, arm, origin, n, rate, seed):
        for trial in self.plan["trials"]:
            if (trial["arm_id"], trial["origin_id"], trial["budget_n"], trial["learning_rate"],
                    trial["post_seed"]) == (arm, origin, n, rate, seed):
                return trial
        raise KeyError((arm, origin, n, rate, seed))

    def row(self, trial, boundary):
        return next(r for r in self.results["records"]
                    if (r["trial_id"], r["boundary"]) == (trial["trial_id"], boundary))

    def rewrite(self, trial, boundary, **changes):
        self.rewrites += 1
        directory = self.tmp / f"rewrite-{self.rewrites}"
        vector = changes.pop("vector", LEVELS[level_of(trial, boundary)][0])
        sha = export(self.world, trial, boundary, directory, vector=vector, **changes)
        self.row(trial, boundary).update(export_path=str(directory), manifest_sha256=sha)
        return directory

    def fail(self, trial, boundary, reason="CUDA out of memory in the post run"):
        row = self.row(trial, boundary)
        row.clear()
        row.update(trial_id=trial["trial_id"], boundary=boundary, status="failed", reason=reason)

    def select(self, document=None, loader=None):
        document = labels_document(self.plan) if document is None else document
        return study.select_study(
            self.plan, self.world.manifest, self.results, loader or (lambda: (document, hexn(2))),
            base=self.world.dir, files={"plan": hexn(1), "manifest": self.world.manifest_sha, "results": hexn(3)})


def final_mean(plan, arm, n, rate):
    trials = [t for t in plan["trials"] if (t["arm_id"], t["budget_n"], t["learning_rate"]) == (arm, n, rate)]
    return math.fsum(LEVELS[level_of(t, plan["boundaries"][-1])][1] for t in trials) / len(trials)


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LEVELS))
def test_the_level_table_is_the_pairwise_definition(name):
    vector, expected = LEVELS[name]
    measured = [DEV_VALUES[eid] for eid in DEV_IDS]
    assert reference_concordance(vector, measured) == pytest.approx(expected)
    result = study.ordinal_concordance(vector, measured)
    assert result["value"] == pytest.approx(expected) and result["comparable_pairs"] == 5


def test_measured_ties_are_excluded_and_prediction_ties_earn_half_credit():
    measured = [DEV_VALUES[eid] for eid in DEV_IDS]
    tie = study.ordinal_concordance(LEVELS["tie"][0], measured)
    assert (tie["comparable_pairs"], tie["concordant"], tie["tied_predictions"], tie["discordant"]) == (5, 4, 1, 0)
    assert tie["value"] == (2 * 4 + 1) / (2 * 5)
    flat = study.ordinal_concordance((5.0, 5.0, 5.0, 5.0), measured)
    assert flat["value"] == 0.5 and flat["tied_predictions"] == 5      # an uninformative score is exactly chance
    swapped_tied_pair = study.ordinal_concordance((1.0, 9.0, 2.0, 10.0), measured)
    assert swapped_tied_pair["value"] == 1.0 and swapped_tied_pair["comparable_pairs"] == 5


def test_no_comparable_pairs_is_undefined_not_zero_or_one():
    result = study.ordinal_concordance((1.0, 2.0, 3.0), (7.0, 7.0, 7.0))
    assert result["value"] is None and result["comparable_pairs"] == 0
    assert study.ordinal_concordance((), ())["value"] is None
    assert study.ordinal_concordance((1.0,), (1.0,))["value"] is None


@pytest.mark.parametrize("seed", range(12))
def test_the_fast_metric_matches_the_quadratic_definition_on_random_tied_data(seed):
    rng = random.Random(seed)
    size = rng.randrange(2, 14)
    scores = [float(rng.randrange(0, 5)) for _ in range(size)]
    measured = [float(rng.randrange(0, 4)) for _ in range(size)]
    expected = reference_concordance(scores, measured)
    actual = study.ordinal_concordance(scores, measured)["value"]
    assert actual == (pytest.approx(expected) if expected is not None else None)


def test_lower_better_values_are_oriented_before_the_comparison():
    assert study.oriented(5.0, "higher_better") == 5.0 and study.oriented(5.0, "lower_better") == -5.0
    raw = [100.0 - DEV_VALUES[eid] for eid in DEV_IDS]
    values = [study.oriented(v, "lower_better") for v in raw]
    assert study.ordinal_concordance(LEVELS["swap"][0], values)["value"] == pytest.approx(0.8)


@pytest.mark.parametrize("scores,values", (((1.0, float("nan")), (1.0, 2.0)), ((1.0, 2.0), (True, 2.0)),
                                           ((1.0,), (1.0, 2.0))))
def test_the_metric_validates_its_inputs(scores, values):
    with pytest.raises(ValueError):
        study.ordinal_concordance(scores, values)


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------

SPEC_PROBLEMS = {
    "schema": lambda d: d.update(schema="trastuzumab-study-spec/2"),
    "extra_key": lambda d: d.update(outcome=1.0),
    "missing_key": lambda d: d.pop("arms"),
    "missing_pin": lambda d: d["budgets"]["entries"][0].pop("selected_labels_sha256"),
    "missing_data_pin": lambda d: d["budgets"]["entries"][1].pop("selected_data_sha256"),
    "malformed_pin": lambda d: d["budgets"]["entries"][0].update(selected_data_sha256="not-a-hash"),
    "uppercase_pin": lambda d: d["budgets"]["entries"][0].update(selected_labels_sha256="A" * 64),
    "budgets_not_increasing": lambda d: d["budgets"]["entries"].reverse(),
    "duplicate_budget": lambda d: d["budgets"]["entries"][1].update(budget_n=2),
    "zero_budget": lambda d: d["budgets"]["entries"][0].update(budget_n=0),
    "bool_budget": lambda d: d["budgets"]["entries"][0].update(budget_n=True),
    "empty_budgets": lambda d: d["budgets"].update(entries=[]),
    "empty_budget_seed": lambda d: d["budgets"].update(budget_seed=""),
    "rates_not_increasing": lambda d: d["recipe"].update(learning_rates=[0.003, 0.001]),
    "duplicate_rates": lambda d: d["recipe"].update(learning_rates=[0.001, 0.001]),
    "negative_rate": lambda d: d["recipe"].update(learning_rates=[-1.0, 0.001]),
    "bool_rate": lambda d: d["recipe"].update(learning_rates=[True, 0.003]),
    "prospective_off_grid": lambda d: d["recipe"].update(prospective_learning_rate=0.002),
    "template_carries_a_rate": lambda d: d["recipe"]["training"].update(learning_rate=0.1),
    "template_carries_a_seed": lambda d: d["recipe"]["training"].update(seed=0),
    "template_missing_field": lambda d: d["recipe"]["training"].pop("weight_decay"),
    "template_midtraining": lambda d: d["recipe"]["training"].update(stage="mid"),
    "template_abonly": lambda d: d["recipe"]["training"].update(arm="abonly"),
    "final_boundary_is_not_the_horizon": lambda d: d["recipe"].update(boundaries=[1]),
    "boundary_beyond_the_horizon": lambda d: d["recipe"].update(boundaries=[1, 2, 3]),
    "boundaries_not_increasing": lambda d: d["recipe"].update(boundaries=[2, 1]),
    "zero_boundary": lambda d: d["recipe"].update(boundaries=[0, 2]),
    "duplicate_seeds": lambda d: d["recipe"].update(post_seeds=[3, 3]),
    "negative_seed": lambda d: d["recipe"].update(post_seeds=[-1, 3]),
    "full_update_fusion": lambda d: d["recipe"]["fusion"].update(update_mode="full"),
    "fusion_missing_field": lambda d: d["recipe"]["fusion"].pop("lora_rank"),
    "encoder_runtime_names_another_encoder": lambda d: d["identity"]["encoder"]["runtime"]["encoder"].update(
        tokenizer="other"),
    "encoder_dtype_differs_from_fusion": lambda d: d["identity"]["encoder"]["runtime"].update(
        compute_dtype="bfloat16"),
    "execution_missing_section": lambda d: d["identity"]["execution"].pop("software"),
    "empty_hardware": lambda d: d["identity"]["execution"].update(hardware={}),
    "malformed_source": lambda d: d["identity"].update(source_sha256="x"),
    "malformed_release_base": lambda d: d["identity"].update(release_base_sha256="x"),
    "one_arm_is_no_comparison": lambda d: d.update(arms=d["arms"][:1]),
    "duplicate_arm_id": lambda d: d["arms"][1].update(arm_id="release"),
    "release_with_two_origins": lambda d: d["arms"][0]["origins"].append({"origin_id": "second"}),
    "release_with_a_mode": lambda d: d["arms"][0].update(functional_mode="detached"),
    "unknown_mode": lambda d: d["arms"][1].update(functional_mode="stacked"),
    "unknown_kind": lambda d: d["arms"][1].update(kind="post"),
    "mid_origin_without_hashes": lambda d: d["arms"][1]["origins"].__setitem__(0, {"origin_id": "mid-0"}),
    "mid_arm_without_origins": lambda d: d["arms"][1].update(origins=[]),
    "duplicate_origin_id": lambda d: d["arms"][1]["origins"][1].update(origin_id="mid-0"),
    "parent_shared_across_arms": lambda d: d["arms"][2]["origins"][0].update(
        parent_sha256=d["arms"][1]["origins"][0]["parent_sha256"]),
    "parent_equals_its_checkpoint": lambda d: d["arms"][1]["origins"][0].update(
        parent_checkpoint_sha256=d["arms"][1]["origins"][0]["parent_sha256"]),
    "two_release_arms": lambda d: d["arms"].append(
        {"arm_id": "release-2", "kind": "release", "functional_mode": None, "origins": [{"origin_id": "r"}]}),
    "bad_orientation": lambda d: d["development"].update(orientation="bigger"),
    "bad_context_arm": lambda d: d["query"].update(context_arm="ab"),
    "bool_order_seed": lambda d: d["query"].update(order_seed=True),
    "zero_orders": lambda d: d["query"].update(orders=0),
}


@pytest.mark.parametrize("problem", sorted(SPEC_PROBLEMS))
def test_the_specification_is_strict(problem):
    document = copy.deepcopy(synthetic_spec())
    SPEC_PROBLEMS[problem](document)
    with pytest.raises((ValueError, TypeError)):
        study.normalize_spec(document)


def test_the_synthetic_specification_is_valid_and_normalization_is_idempotent():
    spec = study.normalize_spec(synthetic_spec())
    assert study.normalize_spec(spec) == spec
    assert [b["budget_n"] for b in spec["budgets"]["entries"]] == list(SIZES)


def test_normalization_never_aliases_its_input():
    document = synthetic_spec()
    spec = study.normalize_spec(document)
    document["recipe"]["training"]["weight_decay"] = 9.0
    document["arms"][1]["origins"][0]["origin_id"] = "changed"
    assert spec["recipe"]["training"]["weight_decay"] == TEMPLATE["weight_decay"]
    assert spec["arms"][1]["origins"][0]["origin_id"] == "mid-0"


def test_the_example_specification_parses_under_the_current_schema():
    path = ROOT / "configs" / "trastuzumab_lowresource" / "study" / "study_spec.json"
    spec, manifest_path = study.parse_spec(read_json(path))
    assert manifest_path == "REPLACE/endpoints.json"
    assert [a["arm_id"] for a in spec["arms"]] == ["release", "mid-denoising-only", "mid-functional-detached",
                                                  "mid-functional-shared"]
    assert spec["recipe"]["boundaries"][-1] == spec["recipe"]["training"]["total_updates"]
    assert spec["recipe"]["prospective_learning_rate"] in spec["recipe"]["learning_rates"]
    # Nothing but the specification is read: the path is resolved, never opened.
    resolved = study.load_spec(path)[1]
    assert resolved.endswith("endpoints.json") and "REPLACE" in resolved


# ---------------------------------------------------------------------------
# The plan: outcome-free, derived, rederivable
# ---------------------------------------------------------------------------

def test_the_panel_is_every_development_entry_with_case_id_equal_to_endpoint_id():
    plan = synthetic_plan()
    queries = plan["queries"]["queries"]
    assert [q["case_id"] for q in queries] == list(DEV_IDS) == plan["development"]["endpoint_ids"]
    assert [q["core"] for q in queries] == [DEV_CORES[eid] for eid in DEV_IDS]
    assert "r0" not in plan["development"]["endpoint_ids"] and not {"p0", "p1", "p2", "p3"} & set(DEV_IDS)
    panel = QueryPanel.from_document(plan["queries"])
    assert all(q.construct_id == panel.construct_id(q.core) for q in panel.queries)
    assert plan["scaffold_id"] == panel.scaffold_id and plan["antigen_sha256"] == panel.antigen_sha256
    assert plan["queries_sha256"] == digest(panel.document())
    assert plan["development"]["endpoint_semantics"] == "synthetic ordering"


def test_the_readout_is_the_literal_readout_with_independently_derived_orders():
    plan = synthetic_plan()
    panel = QueryPanel.from_document(plan["queries"])
    coords = tuple(("H", len(panel.heavy_prefix) + i) for i in range(10))
    rng = random.Random(QUERY["order_seed"])
    orders = [[list(c) for c in rng.sample(coords, 10)] for _ in range(QUERY["orders"])]
    readout = plan["readout"]
    assert readout["orders"] == orders and len(orders) == QUERY["orders"]
    assert readout["name"] == "whole_core_mean_sequential_log_probability" and readout["canonical_residues"] == 20
    assert readout["order_seed"] == QUERY["order_seed"]
    assert readout["averaging"] == "arithmetic mean over complete orders; nats per site"
    assert readout["is_sampler_likelihood"] is False and readout["is_affinity"] is False
    assert set(readout) == {"name", "canonical_residues", "order_seed", "orders", "averaging",
                            "is_sampler_likelihood", "is_affinity"}
    first = len(panel.heavy_prefix)
    assert all(sorted(order) == [["H", first + i] for i in range(10)] for order in readout["orders"])


def test_the_inventory_is_every_real_origin_budget_rate_and_seed_with_deterministic_ids():
    plan = synthetic_plan()
    trials = plan["trials"]
    # release: 1 origin; denoising-only: 2 real parent replicates; detached: 1.  2 budgets x 2 rates x 2 seeds each.
    assert len(trials) == (1 + 2 + 1) * 2 * 2 * 2 == plan["counts"]["trials"]
    assert plan["counts"]["exports"] == len(trials) * len(BOUNDARIES)
    assert len({t["trial_id"] for t in trials}) == len(trials)
    for trial in trials:
        coordinates = {k: v for k, v in trial.items() if k != "trial_id"}
        assert trial["trial_id"] == digest(coordinates)
    by_arm = {}
    for trial in trials:
        by_arm.setdefault(trial["arm_id"], set()).add(trial["origin_id"])
    assert by_arm == {"release": {"release"}, A_STAGE: {"mid-0", "mid-1"}, D_STAGE: {"mid-0"}}
    assert synthetic_plan()["trials"] == trials and synthetic_plan() == plan
    assert plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def test_the_plan_binds_budgets_pins_manifest_and_the_unique_label_union():
    plan = synthetic_plan()
    manifest = manifest_from(manifest_document())
    for budget in plan["budgets"]:
        canonical = select_budget(manifest, budget["budget_n"], BUDGET_SEED)
        assert budget["identity"]["selected_ids"] == list(canonical.selected_ids)
        assert budget["identity"]["selection_digest"] == canonical.selection_digest
        assert budget["selected_data_sha256"] == hexn(300 + budget["budget_n"])
        assert budget["selected_labels_sha256"] == hexn(400 + budget["budget_n"])
    small, large = (b["identity"]["selected_ids"] for b in plan["budgets"])
    assert large[:len(small)] == small                                       # nested prefixes of one seeded order
    assert plan["manifest"]["file_sha256"] == SYNTHETIC_MANIFEST_SHA
    assert plan["manifest"]["digest"] == manifest.manifest_digest()
    assert plan["manifest"]["roles"] == {"post_pool": 4, "development": 4, "reserved": 1}
    counts = plan["counts"]
    assert (counts["largest_fitting_budget"], counts["development_labels"], counts["unique_label_union"]) == (4, 4, 8)


def test_a_different_pin_budget_seed_or_manifest_gives_a_different_plan_but_the_same_trial_ids():
    base = synthetic_plan()
    spec = synthetic_spec()
    spec["budgets"]["entries"][0]["selected_data_sha256"] = hexn(999)
    repinned = synthetic_plan(spec)
    assert repinned["plan_sha256"] != base["plan_sha256"]
    assert [t["trial_id"] for t in repinned["trials"]] == [t["trial_id"] for t in base["trials"]]
    spec = synthetic_spec()
    spec["budgets"]["budget_seed"] = "another-seed"
    assert synthetic_plan(spec)["plan_sha256"] != base["plan_sha256"]
    other = study.build_plan(synthetic_spec(), manifest_from(manifest_document()), hexn(8))
    assert other["plan_sha256"] != base["plan_sha256"]


def test_the_planner_never_opens_labels_models_or_checkpoints(tmp_path, monkeypatch):
    spec = copy.deepcopy(synthetic_spec())
    write(tmp_path / "endpoints.json", manifest_document())
    write(tmp_path / "spec.json", {**spec, "endpoint_manifest_path": "endpoints.json"})

    def boom(*args, **kwargs):
        raise AssertionError("the planner opened a label, a model or a checkpoint")

    for owner, name in ((inputs, "load_post"), (artifacts, "score_queries"), (cli, "models"),
                        (cli, "load_inference_checkpoint"), (ck, "read_checkpoint"), (torch, "load")):
        monkeypatch.setattr(owner, name, boom)
    monkeypatch.setattr(StageTrainer, "__init__", boom)
    plan = study.plan_study(tmp_path / "spec.json")
    assert plan == study.build_plan(spec, manifest_from(manifest_document()), file_digest(tmp_path / "endpoints.json"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["endpoints.json", "spec.json"]


def test_the_study_module_cannot_open_fitting_labels_models_or_the_benchmark_package():
    forbidden = {"load_post", "load_functional_measurements", "load_restricted_measurements", "RestrictedPostData",
                 "MeasuredEndpoint", "build_models", "StageTrainer", "resume_checkpoint", "load_inference_checkpoint",
                 "load_parent", "score_queries", "generate_attempts", "export_parent"}
    names = imported_names(study)
    assert not [name for name in names if name.rsplit(".", 1)[-1] in forbidden]
    assert not [name for name in names if "benchmark" in name]


def test_a_full_antigen_mutation_keeps_every_construct_id_but_changes_the_panel_and_the_plan():
    base = synthetic_plan()
    mutated = synthetic_plan(document=manifest_document(antigen="HIL"))
    assert [q["construct_id"] for q in mutated["queries"]["queries"]] == [
        q["construct_id"] for q in base["queries"]["queries"]]              # construct_id excludes the antigen
    assert mutated["scaffold_id"] == base["scaffold_id"]
    assert mutated["antigen_sha256"] != base["antigen_sha256"]
    assert mutated["queries_sha256"] != base["queries_sha256"]
    assert mutated["plan_sha256"] != base["plan_sha256"]


@pytest.mark.parametrize("problem,document", (
    ("empty_prefix", manifest_document(prefix="")),
    ("empty_suffix", manifest_document(suffix="")),
    ("one_development_endpoint", manifest_document(dev_ids=("d0",))),
    ("no_development_endpoint", manifest_document(dev_ids=())),
    ("another_development_assay", manifest_document(dev_assay="OTHER")),
))
def test_the_panel_needs_a_nonempty_scaffold_and_one_development_assay(problem, document):
    spec = synthetic_spec()
    if problem == "another_development_assay":
        document = copy.deepcopy(document)
        document["entries"][4]["assay_id"] = "SPR"                           # one development entry differs
    with pytest.raises(ValueError):
        synthetic_plan(spec, document)


def test_a_budget_larger_than_the_post_pool_is_refused():
    spec = synthetic_spec()
    spec["budgets"]["entries"][1].update(budget_n=5)
    with pytest.raises(ValueError, match="exceeds post_pool"):
        synthetic_plan(spec)


PLAN_TAMPERS = {
    "dropped_trial": lambda p: p["trials"].pop(),
    "duplicated_trial": lambda p: p["trials"].append(copy.deepcopy(p["trials"][0])),
    "extra_trial": lambda p: p["trials"].append({**p["trials"][0], "trial_id": hexn(5), "post_seed": 99}),
    "renamed_trial": lambda p: p["trials"][0].update(trial_id=hexn(5)),
    "edited_order": lambda p: p["readout"]["orders"][0].reverse(),
    "fewer_orders": lambda p: p["readout"]["orders"].pop(),
    "edited_budget_ids": lambda p: p["budgets"][1]["identity"]["selected_ids"].reverse(),
    "edited_pin": lambda p: p["budgets"][0].update(selected_data_sha256=hexn(5)),
    "edited_panel_core": lambda p: p["queries"]["queries"][0].update(core="KKKKKKKKKK"),
    "edited_antigen": lambda p: p["queries"].update(antigen="HIL"),
    "edited_boundaries": lambda p: p["boundaries"].pop(0),
    "edited_counts": lambda p: p["counts"].update(trials=1),
    "edited_development": lambda p: p["development"]["endpoint_ids"].append("r0"),
    "edited_spec_rate": lambda p: p["spec"]["recipe"]["learning_rates"].append(0.01),
    "edited_manifest_binding": lambda p: p["manifest"].update(file_sha256=hexn(5)),
}


@pytest.mark.parametrize("name", sorted(PLAN_TAMPERS))
def test_a_resealed_tampered_plan_is_not_the_plan_its_specification_derives(name):
    plan = synthetic_plan()
    manifest = manifest_from(manifest_document())
    assert study.validate_plan(plan, manifest, SYNTHETIC_MANIFEST_SHA) == plan
    PLAN_TAMPERS[name](plan)
    with pytest.raises(ValueError, match="digest mismatch"):
        study.validate_plan(plan, manifest, SYNTHETIC_MANIFEST_SHA)
    reseal(plan)                      # an author who recomputes the digest is still caught by rederivation
    with pytest.raises(ValueError):
        study.validate_plan(plan, manifest, SYNTHETIC_MANIFEST_SHA)


def test_a_plan_is_bound_to_the_manifest_file_it_was_derived_from():
    plan = synthetic_plan()
    manifest = manifest_from(manifest_document())
    with pytest.raises(ValueError, match="derive"):
        study.validate_plan(plan, manifest, hexn(8))
    with pytest.raises(ValueError, match="derive"):
        study.validate_plan(plan, manifest_from(manifest_document(antigen="HIL")), SYNTHETIC_MANIFEST_SHA)


# ---------------------------------------------------------------------------
# The world is actual: lineage, pins, exports
# ---------------------------------------------------------------------------

def test_the_world_is_made_of_actual_runs_with_distinct_real_parents(world):
    plan = world.plan
    assert plan["counts"]["trials"] == 32 and plan["counts"]["exports"] == 64
    assert len({p.sha for p in world.parents.values()}) == len(world.parents) == 3
    for (arm_id, origin_id), parent in world.parents.items():
        assert len({parent.sha, parent.checkpoint_sha, parent.provenance_sha}) == 3
    trial = plan["trials"][0]
    identity = world.provenance[(trial["trial_id"], 2)]["training"]
    assert identity["schema"] == "trastuzumab-training/2" and identity["selection"] == "fixed-last"
    assert identity["config"]["stage"] == "post" and identity["functional"] is None
    assert identity["input"]["selected_data"] == next(
        b["selected_data_sha256"] for b in plan["budgets"] if b["budget_n"] == trial["budget_n"])
    # The recorded exports' readout is exactly the planned one: score_queries wrote it, the planner derived it.
    metadata = read_json(world.dir / world.results["records"][0]["export_path"] / "metadata.json")
    assert metadata["readout"] == plan["readout"]


def test_the_planned_panel_scores_exactly_like_the_actual_exporter(world):
    plan = world.plan
    trial = plan["trials"][0]
    with mock.patch.object(artifacts, "whole_core_scores", table_scorer({c: 1.0 for c in DEV_CORES.values()})):
        payload, metadata = score_queries(world.scorers[(trial["arm_id"], trial["origin_id"])], world.encoder,
                                          world.panel, checkpoint_identity=world.provenance[(trial["trial_id"], 1)],
                                          arm="paired", order_seed=QUERY["order_seed"], orders=QUERY["orders"],
                                          scorer_id=QUERY["scorer_id"])
    assert metadata["readout"] == plan["readout"]
    assert metadata["queries_sha256"] == plan["queries_sha256"] and payload["provenance"]["queries"] == plan["queries_sha256"]
    assert [row["case_id"] for row in payload["scores"]] == plan["development"]["endpoint_ids"]


# ---------------------------------------------------------------------------
# Selection with known answers
# ---------------------------------------------------------------------------

def test_selection_at_the_final_horizon_aggregates_every_origin_and_seed_with_known_answers(world, tmp_path):
    scenario = Study(world, tmp_path)
    result = scenario.select()
    assert result["schema"] == study.SELECTION_SCHEMA
    assert result["result_sha256"] == digest({k: v for k, v in result.items() if k != "result_sha256"})
    assert len(result["selected"]) == len(EXPECTED) == 6
    for entry in result["selected"]:
        rate, mean = EXPECTED[(entry["arm_id"], entry["budget_n"])]
        assert entry["selected_learning_rate"] == rate
        assert entry["mean_final_metric"] == pytest.approx(mean)
        assert entry["final_boundary"] == 2 and entry["reason"] is None
    # The independent oracle agrees with the table, candidate by candidate.
    for candidate in result["candidates"]:
        expected = final_mean(world.plan, candidate["arm_id"], candidate["budget_n"], candidate["learning_rate"])
        assert candidate["eligible"] and candidate["mean_final_metric"] == pytest.approx(expected)
        assert candidate["valid_final_trials"] == candidate["planned_trials"]
    # Real replicates only: the release arm has its seeds, a mid arm has parents x seeds.
    planned = {(c["arm_id"], c["budget_n"], c["learning_rate"]): c["planned_trials"] for c in result["candidates"]}
    assert {arm: {v for (a, _, _), v in planned.items() if a == arm} for arm in ("release", A_STAGE, D_STAGE)} == {
        "release": {2}, A_STAGE: {4}, D_STAGE: {2}}


def test_an_exact_tie_chooses_the_smallest_learning_rate(world, tmp_path):
    result = Study(world, tmp_path).select()
    tied = [c for c in result["candidates"] if c["arm_id"] == D_STAGE and c["budget_n"] == 2]
    assert len(tied) == 2 and tied[0]["mean_final_metric"] == tied[1]["mean_final_metric"]
    chosen = next(s for s in result["selected"] if s["arm_id"] == D_STAGE and s["budget_n"] == 2)
    assert chosen["selected_learning_rate"] == min(RATES)


def test_an_exact_tie_of_different_rational_decompositions_is_a_tie_not_a_float_bit_win(world, tmp_path):
    """Release, n=2, two real replicates per rate, four distinct measured values (six comparable pairs).

    The smaller rate has concordant counts [1, 4] and the larger [0, 5]: both means are exactly 5/12, from
    different fractions.  Their floating means differ in the last bit (0.41666666666666663 against
    0.4166666666666667), which made the larger rate win a tie the rule says goes to the smaller one.
    """
    assert math.fsum([1 / 6, 4 / 6]) / 2 < math.fsum([0 / 6, 5 / 6]) / 2        # a real floating-point misordering
    scenario = Study(world, tmp_path)
    vectors = {(0.001, 3): (4.0, 3.0, 1.0, 2.0), (0.001, 5): (2.0, 1.0, 4.0, 3.0),
               (0.003, 3): (4.0, 3.0, 2.0, 1.0), (0.003, 5): (1.0, 2.0, 4.0, 3.0)}
    for (rate, seed), vector in vectors.items():
        scenario.rewrite(scenario.trial("release", "release", 2, rate, seed), 2, vector=vector)
    result = scenario.select(labels_document(scenario.plan, {"d0": 10.0, "d1": 20.0, "d2": 30.0, "d3": 40.0}))
    concordant = {}
    for trial in scenario.plan["trials"]:
        if (trial["arm_id"], trial["budget_n"]) == ("release", 2):
            pairs = next(e for e in result["evaluations"]
                         if (e["trial_id"], e["boundary"]) == (trial["trial_id"], 2))["pairs"]
            assert pairs["comparable_pairs"] == 6 and pairs["tied_predictions"] == 0
            concordant.setdefault(trial["learning_rate"], []).append(pairs["concordant"])
    assert {rate: sorted(counts) for rate, counts in concordant.items()} == {0.001: [1, 4], 0.003: [0, 5]}
    candidates = {c["learning_rate"]: c for c in result["candidates"] if (c["arm_id"], c["budget_n"]) == ("release", 2)}
    assert candidates[0.001]["mean_final_metric"] == candidates[0.003]["mean_final_metric"] == 5 / 12
    chosen = next(s for s in result["selected"] if (s["arm_id"], s["budget_n"]) == ("release", 2))
    assert chosen["selected_learning_rate"] == 0.001 and chosen["mean_final_metric"] == 5 / 12
    # The curves use the same exact calculation: the two final points agree.
    cell = next(c for c in result["curves"] if (c["arm_id"], c["budget_n"]) == ("release", 2))
    assert cell["retrospective_selected"]["learning_rate"] == 0.001
    assert cell["prospective_fixed"]["learning_rate"] == PROSPECTIVE_RATE
    assert cell["retrospective_selected"]["points"][-1]["mean_metric"] == 5 / 12
    assert cell["prospective_fixed"]["points"][-1]["mean_metric"] == 5 / 12


def test_aggregation_beats_a_single_lucky_seed_or_parent(world, tmp_path):
    result = Study(world, tmp_path).select()
    # release, n=2: rate 0.003 owns the best single checkpoint (perfect) but also the worst; the mean decides.
    lucky = [c for c in result["candidates"] if c["arm_id"] == "release" and c["budget_n"] == 2]
    assert {c["learning_rate"]: round(c["mean_final_metric"], 6) for c in lucky} == {0.001: 0.8, 0.003: 0.5}
    release = next(s for s in result["selected"] if s["arm_id"] == "release" and s["budget_n"] == 2)
    assert release["selected_learning_rate"] == 0.001
    # denoising-only, n=2: parent 0 alone prefers 0.003 (1.0 vs 0.6) and parent 1 alone prefers 0.001 (1.0 vs 0.8);
    # the pooled mean over both real parents and both seeds, 0.9 against 0.8, decides.
    pooled = next(s for s in result["selected"] if s["arm_id"] == A_STAGE and s["budget_n"] == 2)
    assert pooled["selected_learning_rate"] == 0.003 and pooled["mean_final_metric"] == pytest.approx(0.9)


def test_every_winning_origin_and_seed_checkpoint_is_kept(world, tmp_path):
    result = Study(world, tmp_path).select()
    for entry in result["selected"]:
        winners = entry["winning_trials"]
        planned = [t for t in world.plan["trials"] if (t["arm_id"], t["budget_n"], t["learning_rate"]) == (
            entry["arm_id"], entry["budget_n"], entry["selected_learning_rate"])]
        assert {w["trial_id"] for w in winners} == {t["trial_id"] for t in planned}
        for winner in winners:
            assert winner["checkpoint_sha256"] == world.provenance[(winner["trial_id"], 2)]["checkpoint_sha256"]
            assert winner["metric"] == pytest.approx(LEVELS[level_of(next(
                t for t in planned if t["trial_id"] == winner["trial_id"]), 2)][1])
    noaux = next(s for s in result["selected"] if s["arm_id"] == A_STAGE and s["budget_n"] == 2)
    assert {(w["origin_id"], w["post_seed"]) for w in noaux["winning_trials"]} == {
        (o, s) for o in ("mid-0", "mid-1") for s in SEEDS}


def test_the_final_winner_can_differ_from_the_intermediate_winner_and_the_two_curves_stay_separate(world, tmp_path):
    result = Study(world, tmp_path).select()
    cell = next(c for c in result["curves"] if c["arm_id"] == "release" and c["budget_n"] == 2)
    retrospective, prospective = cell["retrospective_selected"], cell["prospective_fixed"]
    assert retrospective["learning_rate"] == 0.001 and prospective["learning_rate"] == PROSPECTIVE_RATE == 0.003
    assert retrospective["label"].startswith("RETROSPECTIVE") and prospective["label"].startswith("PROSPECTIVE")
    by_boundary = lambda curve: {p["boundary"]: p["mean_metric"] for p in curve["points"]}
    assert by_boundary(retrospective) == pytest.approx({1: 0.6, 2: 0.8})      # the selected rate was behind at boundary 1
    assert by_boundary(prospective) == pytest.approx({1: 1.0, 2: 0.5})        # the fixed rate led, then lost
    for curve in (retrospective, prospective):
        assert [p["boundary"] for p in curve["points"]] == list(BOUNDARIES)
        assert all(p["planned"] == 2 and p["valid"] == 2 for p in curve["points"])
        assert {m["post_seed"] for m in curve["points"][0]["members"]} == set(SEEDS)
    assert cell["retrospective_selected"] != cell["prospective_fixed"]


def test_label_accounting_counts_the_exact_unique_union_and_keeps_broad_supervision_separate(world, tmp_path):
    accounting = Study(world, tmp_path).select()["label_accounting"]
    assert accounting["fitting_labels_by_budget"] == {"2": 2, "4": 4}
    assert accounting["unique_fitting_labels"] == 4            # nested: the largest budget contains the smaller
    assert accounting["development_labels"] == 4 and accounting["unique_label_union"] == 8
    assert accounting["development_labels_used_in_training"] == 0
    broad = {(b["arm_id"], b["origin_id"]): b for b in accounting["broad_supervision"]}
    assert set(broad) == {(A_STAGE, "mid-0"), (A_STAGE, "mid-1"), (D_STAGE, "mid-0")}      # not the release arm
    assert broad[(A_STAGE, "mid-0")]["functional_labels"] == 0 and broad[(A_STAGE, "mid-0")]["functional_mode"] is None
    assert broad[(D_STAGE, "mid-0")]["functional_mode"] == "detached"
    assert broad[(D_STAGE, "mid-0")]["functional_labels"] >= 1
    assert accounting["unique_label_union"] == accounting["unique_fitting_labels"] + accounting["development_labels"]
    assert {entry["status"] for entry in broad.values()} == {"verified"}
    assert {key: entry["declared_functional_mode"] for key, entry in broad.items()} == {
        (A_STAGE, "mid-0"): None, (A_STAGE, "mid-1"): None, (D_STAGE, "mid-0"): "detached"}


def broad_entries(result):
    return {(b["arm_id"], b["origin_id"]): b for b in result["label_accounting"]["broad_supervision"]}


def fail_arm(scenario, arm_id, keep=()):
    """Mark every record of an arm as an explicit failure, except the (trial id, boundary) pairs in ``keep``."""
    for trial in scenario.plan["trials"]:
        if trial["arm_id"] == arm_id:
            for boundary in BOUNDARIES:
                if (trial["trial_id"], boundary) not in keep:
                    scenario.fail(trial, boundary)


def test_an_origin_whose_records_all_failed_is_unavailable_not_absent_and_not_zero(world, tmp_path):
    baseline = broad_entries(Study(world, tmp_path / "baseline").select())
    assert baseline[(D_STAGE, "mid-0")]["status"] == "verified" and baseline[(D_STAGE, "mid-0")]["functional_labels"] >= 1
    scenario = Study(world, tmp_path / "failed")
    fail_arm(scenario, D_STAGE)
    entries = broad_entries(scenario.select())
    assert set(entries) == set(baseline) == {(A_STAGE, "mid-0"), (A_STAGE, "mid-1"), (D_STAGE, "mid-0")}
    lost = entries[(D_STAGE, "mid-0")]
    assert lost["status"] == "unavailable" and lost["reason"]
    assert lost["declared_functional_mode"] == "detached"                 # the declaration stays visible
    assert lost["functional_mode"] is None and lost["functional_labels"] is None      # unknown, never an inferred zero
    assert lost["functional_selection_digest"] is None
    for key in ((A_STAGE, "mid-0"), (A_STAGE, "mid-1")):
        assert entries[key] == baseline[key] and entries[key]["status"] == "verified"
    assert not [arm_id for arm_id, _ in entries if arm_id == "release"]          # the release arm has no broad stage


def test_a_partly_failed_origin_is_still_verified_by_the_exports_that_remain(world, tmp_path):
    baseline = broad_entries(Study(world, tmp_path / "baseline").select())
    scenario = Study(world, tmp_path / "partial")
    kept = next(t for t in scenario.plan["trials"] if t["arm_id"] == D_STAGE)
    fail_arm(scenario, D_STAGE, keep={(kept["trial_id"], 1)})
    result = scenario.select()
    assert result["counts"]["failed_exports"] == 15 and result["counts"]["scored"] == 49
    assert broad_entries(result)[(D_STAGE, "mid-0")] == baseline[(D_STAGE, "mid-0")]
    assert broad_entries(result)[(D_STAGE, "mid-0")]["status"] == "verified"


def test_a_study_whose_every_record_failed_still_declares_every_origin_as_unavailable(world, tmp_path):
    scenario = Study(world, tmp_path)
    for trial in scenario.plan["trials"]:
        for boundary in BOUNDARIES:
            scenario.fail(trial, boundary)
    result = scenario.select()
    entries = broad_entries(result)
    assert set(entries) == {(A_STAGE, "mid-0"), (A_STAGE, "mid-1"), (D_STAGE, "mid-0")}
    assert {entry["status"] for entry in entries.values()} == {"unavailable"}
    assert all(entry["functional_labels"] is None and entry["functional_selection_digest"] is None
               and entry["functional_mode"] is None for entry in entries.values())
    assert {key: entry["declared_functional_mode"] for key, entry in entries.items()} == {
        (A_STAGE, "mid-0"): None, (A_STAGE, "mid-1"): None, (D_STAGE, "mid-0"): "detached"}
    assert result["counts"]["failed_exports"] == 64 and result["counts"]["scored"] == 0
    assert all(s["selected_learning_rate"] is None for s in result["selected"])


def test_repeated_evaluation_opportunities_are_counted(world, tmp_path):
    result = Study(world, tmp_path).select()
    opportunities = result["opportunities"]
    assert opportunities["selection_decisions"] == 6 and opportunities["candidate_learning_rates"] == 12
    assert opportunities["final_horizon_scorings"] == 32 and opportunities["all_boundary_scorings"] == 64
    assert result["counts"] == {"trials": 32, "exports": 64, "failed_exports": 0, "invalid_scores": 0, "scored": 64}
    assert len(result["evaluations"]) == 64


def test_the_artifact_binds_the_plan_results_and_labels_and_leaks_no_label_value(world, tmp_path):
    scenario = Study(world, tmp_path)
    document = labels_document(scenario.plan)
    result = scenario.select(document)
    inputs_ = result["inputs"]
    assert inputs_["plan_sha256"] == world.plan["plan_sha256"]
    assert inputs_["results_digest"] == digest(scenario.results)
    assert inputs_["development_labels_digest"] == digest(document)
    assert (inputs_["plan_file_sha256"], inputs_["manifest_file_sha256"], inputs_["results_file_sha256"]) == (
        hexn(1), world.manifest_sha, hexn(3))
    text = json.dumps(result)
    assert all(f'"value": {v}' not in text for v in DEV_VALUES.values())
    assert result["metric"]["name"] == "development_ordinal_concordance"
    assert "smallest learning rate" in result["selection_rule"]["tie_rule"]


def test_orientation_lower_better_selects_the_same_winners(world, tmp_path):
    spec = copy.deepcopy(world.spec)
    spec["development"]["orientation"] = "lower_better"
    plan = study.build_plan(spec, world.manifest, world.manifest_sha)
    values = {eid: 100.0 - v for eid, v in DEV_VALUES.items()}
    result = Study(world, tmp_path, plan=plan).select(labels_document(plan, values))
    for entry in result["selected"]:
        assert entry["selected_learning_rate"] == EXPECTED[(entry["arm_id"], entry["budget_n"])][0]


# ---------------------------------------------------------------------------
# Failures: explicit, retained, and never rescued by an intermediate checkpoint
# ---------------------------------------------------------------------------

def test_an_explicit_final_failure_makes_that_rate_ineligible_and_is_not_rescued_by_the_first_boundary(world, tmp_path):
    scenario = Study(world, tmp_path)
    failed = scenario.trial("release", "release", 2, 0.001, 3)
    scenario.fail(failed, 2)
    result = scenario.select()
    candidate = next(c for c in result["candidates"] if (c["arm_id"], c["budget_n"], c["learning_rate"]) == (
        "release", 2, 0.001))
    assert not candidate["eligible"] and candidate["mean_final_metric"] is None
    assert (candidate["planned_trials"], candidate["valid_final_trials"]) == (2, 1)    # denominator is not reduced
    assert candidate["ineligible"][0]["status"] == "export_failed" and "CUDA out of memory" in candidate["ineligible"][0]["reason"]
    chosen = next(s for s in result["selected"] if (s["arm_id"], s["budget_n"]) == ("release", 2))
    assert chosen["selected_learning_rate"] == 0.003 and chosen["mean_final_metric"] == pytest.approx(0.5)
    # The failed trial's first boundary scored fine, yet it cannot rescue the failed final candidate.
    assert next(e for e in result["evaluations"] if (e["trial_id"], e["boundary"]) == (failed["trial_id"], 1))[
        "status"] == "scored"
    assert result["counts"]["failed_exports"] == 1 and result["counts"]["scored"] == 63


def test_an_intermediate_failure_is_retained_in_the_curve_without_disqualifying_the_final_candidate(world, tmp_path):
    scenario = Study(world, tmp_path)
    failed = scenario.trial("release", "release", 2, 0.001, 5)
    scenario.fail(failed, 1, "node lost before the first boundary export")
    result = scenario.select()
    chosen = next(s for s in result["selected"] if (s["arm_id"], s["budget_n"]) == ("release", 2))
    assert chosen["selected_learning_rate"] == 0.001 and chosen["mean_final_metric"] == pytest.approx(0.8)
    curve = next(c for c in result["curves"] if (c["arm_id"], c["budget_n"]) == ("release", 2))[
        "retrospective_selected"]
    first, final = curve["points"]
    assert (first["planned"], first["valid"], first["mean_metric"]) == (2, 1, None)   # no mean over a surviving subset
    member = next(m for m in first["members"] if m["trial_id"] == failed["trial_id"])
    assert member["status"] == "export_failed" and "node lost" in member["reason"] and member["metric"] is None
    assert final["valid"] == 2 and final["mean_metric"] == pytest.approx(0.8)


@pytest.mark.parametrize("injected", ("exception", "nonfinite"))
def test_one_failed_query_invalidates_the_whole_candidate(world, tmp_path, injected):
    scenario = Study(world, tmp_path)
    trial = scenario.trial("release", "release", 2, 0.001, 3)
    broken = FloatingPointError("synthetic failure") if injected == "exception" else float("nan")
    scenario.rewrite(trial, 2, overrides={DEV_CORES["d1"]: broken})
    exported = read_json(scenario.tmp / "rewrite-1" / "scores.json")
    failing = [row for row in exported["scores"] if row["failure"]]
    assert len(failing) == 1 and failing[0]["score"] is None and failing[0]["case_id"] == "d1"
    result = scenario.select()
    evaluation = next(e for e in result["evaluations"] if (e["trial_id"], e["boundary"]) == (trial["trial_id"], 2))
    assert evaluation["status"] == "invalid_scores" and evaluation["metric"] is None
    assert [f["case_id"] for f in evaluation["failures"]] == ["d1"]
    candidate = next(c for c in result["candidates"] if (c["arm_id"], c["budget_n"], c["learning_rate"]) == (
        "release", 2, 0.001))
    assert not candidate["eligible"] and candidate["valid_final_trials"] == 1       # the three good queries are not scored
    assert next(s for s in result["selected"] if (s["arm_id"], s["budget_n"]) == ("release", 2))[
        "selected_learning_rate"] == 0.003
    assert result["counts"]["invalid_scores"] == 1


def test_a_cell_with_no_eligible_rate_selects_nothing_and_leaves_the_others_alone(world, tmp_path):
    scenario = Study(world, tmp_path)
    for rate in RATES:
        scenario.fail(scenario.trial(D_STAGE, "mid-0", 2, rate, 3), 2)
    result = scenario.select()
    cell = next(s for s in result["selected"] if (s["arm_id"], s["budget_n"]) == (D_STAGE, 2))
    assert cell["selected_learning_rate"] is None and cell["winning_trials"] == [] and cell["reason"]
    assert next(c for c in result["curves"] if (c["arm_id"], c["budget_n"]) == (D_STAGE, 2))[
        "retrospective_selected"] is None
    others = {(s["arm_id"], s["budget_n"]): s["selected_learning_rate"] for s in result["selected"]}
    assert others[(D_STAGE, 4)] == 0.003 and others[("release", 2)] == 0.001


def test_the_fixed_rate_curve_includes_failures_even_when_the_final_candidate_failed(world, tmp_path):
    scenario = Study(world, tmp_path)
    scenario.fail(scenario.trial("release", "release", 2, PROSPECTIVE_RATE, 5), 2)
    prospective = next(c for c in scenario.select()["curves"] if (c["arm_id"], c["budget_n"]) == ("release", 2))[
        "prospective_fixed"]
    final = prospective["points"][-1]
    assert (final["planned"], final["valid"], final["mean_metric"]) == (2, 1, None)
    assert [m["status"] for m in final["members"]].count("export_failed") == 1


# ---------------------------------------------------------------------------
# Everything is validated before a development label is opened
# ---------------------------------------------------------------------------

def test_inventory_problems_block_selection_before_the_labels_are_opened(world, tmp_path):
    def check(change, message):
        scenario = Study(world, tmp_path)
        change(scenario)
        with pytest.raises(ValueError, match=message):
            scenario.select(loader=labels_opened)

    check(lambda s: s.results["records"].pop(0), "1 missing")
    check(lambda s: s.results["records"].append(copy.deepcopy(s.results["records"][3])), "1 duplicated")
    check(lambda s: s.results["records"].append(
        {"trial_id": "unplanned", "boundary": 1, "status": "failed", "reason": "r"}), "1 unplanned")
    check(lambda s: s.results["records"][0].update(boundary=3), "not the planned inventory")
    check(lambda s: s.results.update(plan_sha256=hexn(9)), "different plan")
    check(lambda s: s.results.update(schema="trastuzumab-study-results/2"), "unsupported")
    check(lambda s: s.results.update(extra=1), "expected exactly")
    check(lambda s: s.results["records"].clear(), "64 missing")


@pytest.mark.parametrize("change", (
    lambda r: r.update(status="skipped"),
    lambda r: r.update(boundary=True),
    lambda r: r.update(unexpected=1),
    lambda r: r.pop("export_path"),
    lambda r: r.update(manifest_sha256="short"),
    lambda r: r.update(export_path=""),
))
def test_a_malformed_results_record_blocks_selection(world, tmp_path, change):
    scenario = Study(world, tmp_path)
    change(scenario.results["records"][0])
    with pytest.raises(ValueError):
        scenario.select(loader=labels_opened)


def test_a_failure_record_needs_a_reason(world, tmp_path):
    scenario = Study(world, tmp_path)
    scenario.fail(scenario.plan["trials"][0], 1, reason="")
    with pytest.raises(ValueError, match="failure reason"):
        scenario.select(loader=labels_opened)


def test_a_missing_export_directory_blocks_selection(world, tmp_path):
    scenario = Study(world, tmp_path)
    scenario.results["records"][0]["export_path"] = "exports/never-written"
    with pytest.raises(ValueError, match="does not exist"):
        scenario.select(loader=labels_opened)


def test_changed_export_bytes_and_stray_files_are_detected(world, tmp_path):
    scenario = Study(world, tmp_path)
    first = scenario.plan["trials"][0]
    original = world.dir / scenario.row(first, 1)["export_path"]

    def copied(name):
        target = tmp_path / name
        shutil.copytree(original, target)
        scenario.row(first, 1)["export_path"] = str(target)
        return target

    target = copied("scores-edited")
    (target / "scores.json").write_bytes((target / "scores.json").read_bytes()[:-1] + b" \n")
    with pytest.raises(ValueError, match="recorded SHA256 and size"):
        scenario.select(loader=labels_opened)

    target = copied("metadata-edited")
    (target / "metadata.json").write_bytes((target / "metadata.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="recorded SHA256 and size"):
        scenario.select(loader=labels_opened)

    target = copied("manifest-edited")
    (target / "manifest.json").write_bytes((target / "manifest.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="recorded SHA256 and size"):
        scenario.select(loader=labels_opened)

    target = copied("stray")
    (target / "notes.txt").write_text("not part of an export")
    with pytest.raises(ValueError, match="an export holds exactly"):
        scenario.select(loader=labels_opened)

    target = copied("incomplete")
    (target / "metadata.json").unlink()
    with pytest.raises(ValueError, match="an export holds exactly"):
        scenario.select(loader=labels_opened)

    scenario.row(first, 1)["export_path"] = str(original)
    scenario.row(first, 1)["manifest_sha256"] = hexn(9)
    with pytest.raises(ValueError, match="recorded SHA256 and size"):
        scenario.select(loader=labels_opened)


def drifted_panel(panel, change):
    if change == "core":
        queries = list(panel.queries)
        queries[0] = Query(queries[0].case_id, "KKKKKKKKKK", panel.construct_id("KKKKKKKKKK"))
        return replace(panel, queries=tuple(queries)), {"KKKKKKKKKK": 1.0}
    if change == "antigen":
        return replace(panel, antigen="HIL"), {}
    if change == "assay":
        return replace(panel, assay_id="OTHER"), {}
    if change == "protocol":
        return replace(panel, protocol_id="another-protocol"), {}
    prefix = "AA"                                                       # a new scaffold needs new construct ids
    scaffold = digest({"kind": "scaffold", "segments": [prefix, panel.heavy_suffix, panel.light]})
    queries = tuple(Query(q.case_id, q.core, digest({"kind": "construct", "scaffold": scaffold, "core": q.core}))
                    for q in panel.queries)
    return QueryPanel(panel.protocol_id, panel.assay_id, prefix, panel.heavy_suffix, panel.light, panel.antigen,
                      queries), {}


@pytest.mark.parametrize("change", ("core", "antigen", "assay", "protocol", "scaffold"))
def test_a_drifted_panel_is_rejected_even_when_every_construct_id_survives(world, tmp_path, change):
    scenario = Study(world, tmp_path)
    trial = scenario.plan["trials"][0]
    panel, overrides = drifted_panel(world.panel, change)
    if change == "antigen":
        assert [q.construct_id for q in panel.queries] == [q.construct_id for q in world.panel.queries]
    scenario.rewrite(trial, 1, panel=panel, overrides=overrides)
    with pytest.raises(ValueError, match="queries_sha256 differs from the plan"):
        scenario.select(loader=labels_opened)


@pytest.mark.parametrize("change,message", (
    ({"orders": 3}, "readout differs from the plan"),
    ({"order_seed": 8}, "readout differs from the plan"),
    ({"arm": "abonly"}, "context_arm differs from the plan"),
    ({"scorer_id": "another-scorer"}, "not the declared score table"),
    ({"edit_metadata": lambda m: m["readout"].update(is_affinity=True)}, "readout differs from the plan"),
    ({"edit_metadata": lambda m: m.update(query_roles_or_outcomes_available=True)}, "no outcomes were available"),
    ({"edit_metadata": lambda m: m.update(scaffold_id=hexn(9))}, "scaffold_id differs from the plan"),
    ({"edit_metadata": lambda m: m.update(antigen_sha256=hexn(9))}, "antigen_sha256 differs from the plan"),
    ({"edit_metadata": lambda m: m.update(schema="trastuzumab-inference/2")}, "schema differs from the plan"),
    ({"edit_metadata": lambda m: m.pop("assay_id")}, "expected exactly"),
    ({"edit_metadata": lambda m: m["model"].update(serialization="another")}, "different model or encoder"),
    ({"edit_metadata": lambda m: m["encoder_runtime"].update(max_tokens=9)}, "different model or encoder"),
    ({"edit_payload": lambda p: p["scores"].pop()}, "not exactly the declared development endpoints"),
    ({"edit_payload": lambda p: p["scores"].append(dict(p["scores"][0]))}, "not exactly the declared development endpoints"),
    ({"edit_payload": lambda p: p["scores"].append({"case_id": "r0", "score": 1.0, "failure": None})},
     "not exactly the declared development endpoints"),
    ({"edit_payload": lambda p: p["scores"][0].update(case_id="p0")}, "not exactly the declared development endpoints"),
    ({"edit_payload": lambda p: p["scores"][0].update(score=True)}, "finite number"),
    ({"edit_payload": lambda p: p["scores"][0].update(score="1.0")}, "finite number"),
    ({"edit_payload": lambda p: p["scores"][0].update(score=None, failure="")}, "nonempty failure reason"),
    ({"edit_payload": lambda p: p["scores"][0].update(failure="boom")}, "nonempty failure reason"),
    ({"edit_payload": lambda p: p["provenance"].update(metadata=hexn(9))}, "do not link"),
    ({"edit_payload": lambda p: p.update(record_kind="attempt_table")}, "not the declared score table"),
    ({"edit_payload": lambda p: p.update(orientation="lower_is_better")}, "not the declared score table"),
    ({"edit_payload": lambda p: p.update(readout="mean_log_probability")}, "not the declared score table"),
    ({"edit_payload": lambda p: p.update(extra=1)}, "expected exactly"),
))
def test_export_drift_is_rejected_before_the_labels_are_opened(world, tmp_path, change, message):
    scenario = Study(world, tmp_path)
    scenario.rewrite(scenario.plan["trials"][0], 1, **change)
    with pytest.raises(ValueError, match=message):
        scenario.select(loader=labels_opened)


RELEASE_N2 = ("release", "release", 2, 0.001, 3)
RELEASE_N4 = ("release", "release", 4, 0.001, 3)
MID_N2 = (A_STAGE, "mid-0", 2, 0.001, 3)
PROVENANCE_DRIFT = {
    "learning_rate": (lambda p: put(p, ("training", "config", "learning_rate"), 0.5), "declared post recipe", RELEASE_N2),
    "seed": (lambda p: put(p, ("training", "config", "seed"), 99), "declared post recipe", RELEASE_N2),
    "horizon": (lambda p: put(p, ("training", "config", "total_updates"), 9), "declared post recipe", RELEASE_N2),
    "ordering_weight": (lambda p: put(p, ("training", "config", "ordering_weight"), 0.0), "declared post recipe",
                        RELEASE_N2),
    "schema": (lambda p: put(p, ("training", "schema"), "trastuzumab-training/1"), "current schema", RELEASE_N2),
    "selection": (lambda p: put(p, ("training", "selection"), "best-on-development"), "current schema", RELEASE_N2),
    "fusion": (lambda p: put(p, ("training", "model", "fusion", "lora_rank"), 7), "common fusion config", RELEASE_N2),
    "encoder_identity": (lambda p: put(p, ("training", "model", "encoder", "tokenizer"), "other"),
                         "encoder identity or runtime", RELEASE_N2),
    "encoder_runtime": (lambda p: put(p, ("training", "encoder_runtime", "max_tokens"), 77),
                        "encoder identity or runtime", RELEASE_N2),
    "runtime_model": (lambda p: put(p, ("training", "runtime", "model", "serialization"), "other"),
                      "different model or encoder", RELEASE_N2),
    "runtime_encoder": (lambda p: put(p, ("training", "runtime", "encoder", "max_tokens"), 1),
                        "different model or encoder", RELEASE_N2),
    "hardware": (lambda p: put(p, ("training", "runtime", "hardware", "device"), "cuda:1"), "execution", RELEASE_N2),
    "software": (lambda p: put(p, ("training", "runtime", "software", "torch"), "0.0.0"), "execution", RELEASE_N2),
    "parameter_counts": (lambda p: put(p, ("training", "runtime", "parameters", "trainable"), 1), "execution",
                         RELEASE_N2),
    "source": (lambda p: put(p, ("training", "source", "sha256"), hexn(9)), "declared source", RELEASE_N2),
    "source_files": (lambda p: put(p, ("training", "source", "files", "experiments/trastuzumab_poc/study.py"), hexn(9)),
                     "declared source", RELEASE_N2),
    "selected_data_pin": (lambda p: put(p, ("training", "input", "selected_data"), hexn(9)), "selected_data SHA256",
                          RELEASE_N2),
    "selected_data_in_data": (lambda p: put(p, ("training", "data", "selected_data"), hexn(9)), "selected_data SHA256",
                              RELEASE_N2),
    "selected_labels_pin": (lambda p: put(p, ("training", "input", "selected_labels_sha256"), hexn(9)),
                            "selected_labels_sha256", RELEASE_N2),
    "endpoint_manifest_file": (lambda p: put(p, ("training", "input", "endpoint_manifest_sha256"), hexn(9)),
                               "different endpoint manifest", RELEASE_N2),
    "endpoint_manifest_digest": (lambda p: put(p, ("training", "input", "manifest"), hexn(9)),
                                 "different endpoint manifest", RELEASE_N2),
    "current_budget_order": (lambda p: p["training"]["input"]["budget"]["selected_ids"].reverse(),
                             "canonical selection", RELEASE_N4),
    "current_budget_other_size": (lambda p: put(p, ("training", "input", "budget"),
                                                {**p["training"]["input"]["budget"], "budget_n": 1,
                                                 "selected_ids": p["training"]["input"]["budget"]["selected_ids"][:1]}),
                                  "canonical selection", RELEASE_N4),
    "available_budget": (lambda p: put(p, ("progress", "fitting_endpoints_available"), 3), "fitting_endpoints_available",
                         RELEASE_N4),
    "available_budget_float": (lambda p: put(p, ("progress", "fitting_endpoints_available"), 4.0),
                               "fitting_endpoints_available", RELEASE_N4),
    "boundary": (lambda p: put(p, ("progress", "next_update"), 1), "next_update", RELEASE_N2),
    "selection_labels": (lambda p: put(p, ("progress", "selection_labels"), 1), "selection_labels", RELEASE_N2),
    # Internally consistent (union, subsets, counts), so only the available-budget subset rule can reject it.
    "seen_outside_budget": (lambda p: reseen(p, endpoints=["r0"], reconstruction_endpoints=["r0"],
                                             ordering_endpoints=[], informative_ordering_endpoints=[]),
                            "outside its available fitting budget", RELEASE_N2),
    "functional_identity": (lambda p: put(p, ("training", "functional"), {"schema": "x"}),
                            "no functional supervision", RELEASE_N2),
    "functional_progress": (lambda p: put(p, ("progress", "functional"), {"counters": {}}),
                            "no functional supervision", RELEASE_N2),
    "malformed_checkpoint_hash": (lambda p: put(p, ("checkpoint_sha256",), "xyz"), "lowercase SHA-256", RELEASE_N2),
    "missing_progress_field": (lambda p: p["progress"].pop("selection_labels"), "checkpoint progress: expected exactly",
                               RELEASE_N2),
    "extra_progress_field": (lambda p: p["progress"].update(note=1), "checkpoint progress: expected exactly",
                             RELEASE_N2),
    "missing_training_field": (lambda p: p["training"].pop("ordering_readout"),
                               "checkpoint training identity: expected exactly", RELEASE_N2),
    "extra_training_field": (lambda p: p["training"].update(note=1), "checkpoint training identity: expected exactly",
                             RELEASE_N2),
    # The fixed recipe the trainer writes and ``checkpoint._validate`` enforces.
    "optimizer_kind": (lambda p: put(p, ("training", "optimizer", "kind"), "SGD"), "fixed AdamW recipe", RELEASE_N2),
    "optimizer_betas": (lambda p: put(p, ("training", "optimizer", "betas"), [0.9, 0.99]), "fixed AdamW recipe",
                        RELEASE_N2),
    "optimizer_flag_as_integer": (lambda p: put(p, ("training", "optimizer", "foreach"), 0),     # 0 == False in Python
                                  "fixed AdamW recipe", RELEASE_N2),
    "random_algorithm": (lambda p: put(p, ("training", "rng"), "mersenne/1"), "fixed AdamW recipe", RELEASE_N2),
    "ordering_readout": (lambda p: put(p, ("training", "ordering_readout"), "dropout-enabled/1"),
                         "fixed AdamW recipe", RELEASE_N2),
    # Contradictory progress records (supplied-record consistency; these are mutated, not real training leaks).
    "examples_counter_zero": (lambda p: put(p, ("progress", "counters", "examples"), 0), "counters are inconsistent",
                              RELEASE_N2),
    "pair_draws_counter": (lambda p: put(p, ("progress", "counters", "pair_draws"), 1), "counters are inconsistent",
                           RELEASE_N2),
    "informative_pairs_counter": (lambda p: p["progress"]["counters"].update(
        informative_pairs=p["progress"]["counters"]["informative_pairs"] + 1), "counters are inconsistent", RELEASE_N2),
    "teacher_forced_counter": (lambda p: put(p, ("progress", "counters", "teacher_forced_states"), 1),
                               "counters are inconsistent", RELEASE_N2),
    "counter_as_float": (lambda p: put(p, ("progress", "counters", "pair_draws"), 2.0), "nonnegative integer",
                         RELEASE_N2),
    "counter_as_boolean": (lambda p: put(p, ("progress", "counters", "ties"), False), "nonnegative integer",
                           RELEASE_N2),
    "counter_negative": (lambda p: put(p, ("progress", "counters", "ties"), -1), "nonnegative integer", RELEASE_N2),
    "counter_missing": (lambda p: p["progress"]["counters"].pop("ties"), "checkpoint counters: expected exactly",
                        RELEASE_N2),
    "duplicate_seen_endpoint": (lambda p: p["progress"]["seen"]["endpoints"].insert(
        0, p["progress"]["seen"]["endpoints"][0]), "sorted list of unique nonempty IDs", RELEASE_N2),
    "empty_seen_id": (lambda p: reseen(p, records=[""]), "sorted list of unique nonempty IDs", RELEASE_N2),
    # The former acceptance fixture: only seen.endpoints is shortened, so unique, the union and the subsets disagree.
    "seen_endpoints_truncated_alone": (
        lambda p: put(p, ("progress", "seen", "endpoints"), p["progress"]["seen"]["endpoints"][:-1]),
        "unique count of endpoints", RELEASE_N4),
    "unique_count_inflated": (
        lambda p: put(p, ("progress", "unique", "endpoints"), p["progress"]["unique"]["endpoints"] + 1),
        "unique count of endpoints", RELEASE_N2),
    "broad_record_exposure": (lambda p: reseen(p, records=["pair-1"]), "only endpoint exposure", RELEASE_N2),
    # The union is valid and inside the budget, but one measurement endpoint is not.
    "ordering_endpoint_outside_budget": (
        lambda p: reseen(p, ordering_endpoints=sorted(set(p["progress"]["seen"]["ordering_endpoints"]) | {"r0"})),
        "reconstruction and measurement exposure", RELEASE_N2),
    "informative_endpoint_not_ordered": (
        lambda p: reseen(p, informative_ordering_endpoints=sorted(
            set(p["progress"]["seen"]["informative_ordering_endpoints"]) | {"r0"})),
        "reconstruction and measurement exposure", RELEASE_N2),
    "union_missing_a_reconstruction_endpoint": (
        lambda p: reseen(p, endpoints=[], reconstruction_endpoints=p["progress"]["seen"]["reconstruction_endpoints"]),
        "reconstruction and measurement exposure", RELEASE_N2),
    "inherited_exposure_on_release": (lambda p: put(p, ("progress", "inherited_exposure"),
                                                    {"parent_progress": {}, "external_declaration": None}),
                                      "inherited exposure", RELEASE_N2),
    "release_with_certificate": (lambda p: put(p, ("training", "transfer_certificate"), {"x": 1}),
                                 "no broad transfer certificate", RELEASE_N2),
    "release_inherits_nothing_else": (lambda p: put(p, ("training", "initialization", "kind"), "parent"),
                                      "release initialization", RELEASE_N2),
    "parent_exposure_dropped": (lambda p: put(p, ("progress", "inherited_exposure"), None), "inherited exposure", MID_N2),
    "parent_exposure_declared": (lambda p: put(p, ("progress", "inherited_exposure", "external_declaration"),
                                               {"labels": 1}), "inherited exposure", MID_N2),
    "parent_certificate_dropped": (lambda p: put(p, ("training", "transfer_certificate"), None),
                                   "transfer certificate", MID_N2),
}


@pytest.mark.parametrize("name", sorted(PROVENANCE_DRIFT))
def test_checkpoint_provenance_drift_is_rejected_before_the_labels_are_opened(world, tmp_path, name):
    editor, message, coordinates = PROVENANCE_DRIFT[name]
    scenario = Study(world, tmp_path)
    trial = scenario.trial(*coordinates)
    # The export is scored from the GENUINE checkpoint identity (the exporter refuses a drifted encoder runtime and
    # would hide the selector under test).  The recorded identity is edited afterwards, and the metadata, the
    # payload's metadata link, the manifest's checkpoint copy and every byte hash are then written consistently.
    scenario.rewrite(trial, 2, edit_metadata=lambda metadata: editor(metadata["checkpoint"]))
    with pytest.raises(ValueError, match=message):
        scenario.select(loader=labels_opened)


@pytest.mark.parametrize("field, spelling", [
    ("examples_per_update", True),       # True == 1 == the declared examples_per_update
    ("examples_per_update", 1.0),        # 1.0 == 1, but the trainer takes only an integer count
    ("gradient_clip", True),             # True == 1.0 == the declared gradient_clip
    ("ordering_weight", True)])
def test_a_recorded_config_dict_equal_to_the_declaration_but_invalid_for_the_trainer_is_rejected(
        world, tmp_path, field, spelling):
    """Python dict equality reads ``True`` as ``1`` and ``1.0`` as ``1``.  Both boundaries record the SAME malformed
    identity, so the check that a trial's boundaries agree cannot be what rejects it."""
    scenario = Study(world, tmp_path)
    trial = scenario.trial(*RELEASE_N2)
    assert trial["learning_rate"] == 0.001 and TEMPLATE[field] == 1 and TEMPLATE[field] == spelling
    for boundary in BOUNDARIES:
        scenario.rewrite(trial, boundary, edit_metadata=lambda metadata: put(
            metadata["checkpoint"], ("training", "config", field), spelling))
    with pytest.raises(ValueError, match=f"not a valid training config: {field} must"):
        scenario.select(loader=labels_opened)


def test_a_valid_integer_spelling_of_a_float_setting_is_still_the_declared_recipe(world, tmp_path):
    scenario = Study(world, tmp_path)
    trial = scenario.trial(*RELEASE_N2)
    for boundary in BOUNDARIES:
        scenario.rewrite(trial, boundary, edit_metadata=lambda metadata: put(
            metadata["checkpoint"], ("training", "config", "gradient_clip"), 1))
    rows = [e for e in scenario.select()["evaluations"] if e["trial_id"] == trial["trial_id"]]
    assert len(rows) == len(BOUNDARIES) and all(row["status"] == "scored" for row in rows)


def test_the_same_available_budget_with_fewer_endpoints_seen_is_comparable(world, tmp_path):
    """Genuine checkpoints: after one update at most three of the four available endpoints were observed
    (one reconstruction example plus one ordered pair), so a smaller seen set under the same available budget is
    what the trainer really records.  Nothing requires every available endpoint to be seen."""
    first = [t for t in world.plan["trials"] if t["budget_n"] == 4]
    assert first
    recorded = {t["trial_id"]: world.provenance[(t["trial_id"], 1)]["progress"] for t in first}
    assert all(p["fitting_endpoints_available"] == 4 and 1 <= len(p["seen"]["endpoints"]) <= 3
               for p in recorded.values())
    result = Study(world, tmp_path).select()
    rows = [e for e in result["evaluations"] if e["trial_id"] in recorded and e["boundary"] == 1]
    assert len(rows) == len(first)
    for row in rows:
        assert row["status"] == "scored" and row["available_endpoints"] == 4
        assert row["seen_endpoints"] == len(recorded[row["trial_id"]]["seen"]["endpoints"]) < row["available_endpoints"]
    assert all(e["seen_endpoints"] <= e["available_endpoints"] for e in result["evaluations"]
               if e["available_endpoints"] is not None)


def test_an_internally_consistent_smaller_exposure_is_accepted(world, tmp_path):
    scenario = Study(world, tmp_path)
    trial = scenario.trial(*RELEASE_N4)
    genuine = world.provenance[(trial["trial_id"], 2)]["progress"]["seen"]
    assert len(genuine["endpoints"]) >= 1
    gone = genuine["endpoints"][0]

    def forget(metadata):
        # Remove one endpoint from every list that holds it: the union, the subsets and the counts stay consistent.
        reseen(metadata["checkpoint"], **{key: [x for x in genuine[key] if x != gone]
                                           for key in ("endpoints", "reconstruction_endpoints", "ordering_endpoints",
                                                       "informative_ordering_endpoints")})

    scenario.rewrite(trial, 2, edit_metadata=forget)
    result = scenario.select()
    row = next(e for e in result["evaluations"] if (e["trial_id"], e["boundary"]) == (trial["trial_id"], 2))
    assert row["status"] == "scored" and row["available_endpoints"] == 4
    assert row["seen_endpoints"] == len(genuine["endpoints"]) - 1


def test_the_same_seen_endpoints_with_another_available_budget_are_rejected(world, tmp_path):
    scenario = Study(world, tmp_path)
    trial = scenario.trial(*RELEASE_N4)
    provenance = copy.deepcopy(world.provenance[(trial["trial_id"], 2)])
    provenance["progress"]["fitting_endpoints_available"] = 2          # a smaller budget that saw the same endpoints
    scenario.rewrite(trial, 2, provenance=provenance)
    with pytest.raises(ValueError, match="fitting_endpoints_available"):
        scenario.select(loader=labels_opened)


def test_another_trials_export_cannot_stand_in_for_this_one(world, tmp_path):
    scenario = Study(world, tmp_path)
    target, donor = scenario.trial(*RELEASE_N2), scenario.trial("release", "release", 2, 0.001, 5)
    row, other = scenario.row(target, 2), scenario.row(donor, 2)
    row.update(export_path=other["export_path"], manifest_sha256=other["manifest_sha256"])
    with pytest.raises(ValueError, match="declared post recipe"):
        scenario.select(loader=labels_opened)
    scenario = Study(world, tmp_path)
    scenario.row(target, 2).update(export_path=scenario.row(target, 1)["export_path"],
                                   manifest_sha256=scenario.row(target, 1)["manifest_sha256"])
    with pytest.raises(ValueError, match="next_update"):
        scenario.select(loader=labels_opened)


def test_an_export_from_another_parent_replicate_is_rejected(world, tmp_path):
    scenario = Study(world, tmp_path)
    mine, other = scenario.trial(A_STAGE, "mid-0", 2, 0.001, 3), scenario.trial(A_STAGE, "mid-1", 2, 0.001, 3)
    scenario.row(mine, 1).update(export_path=scenario.row(other, 1)["export_path"],
                                 manifest_sha256=scenario.row(other, 1)["manifest_sha256"])
    with pytest.raises(ValueError, match="parent other than origin 'mid-0'"):
        scenario.select(loader=labels_opened)


def test_checkpoint_hashes_must_be_unique_and_a_post_checkpoint_is_never_a_parent(world, tmp_path):
    # Exports are immutable (``write_export`` refuses an existing directory), so each independent scenario
    # writes its rewrites into its own subdirectory instead of restarting ``rewrite-1`` in a shared one.
    scenario = Study(world, tmp_path / "duplicate-hash")
    first, second = scenario.plan["trials"][:2]
    stolen = copy.deepcopy(world.provenance[(second["trial_id"], 2)])
    stolen["checkpoint_sha256"] = world.provenance[(first["trial_id"], 2)]["checkpoint_sha256"]
    scenario.rewrite(second, 2, provenance=stolen)
    with pytest.raises(ValueError, match="same checkpoint SHA256"):
        scenario.select(loader=labels_opened)
    scenario = Study(world, tmp_path / "posing-as-parent")
    parent = next(iter(world.parents.values()))
    posing = copy.deepcopy(world.provenance[(first["trial_id"], 2)])
    posing["checkpoint_sha256"] = parent.sha
    scenario.rewrite(first, 2, provenance=posing)
    with pytest.raises(ValueError, match="never a parent"):
        scenario.select(loader=labels_opened)


def test_a_trials_boundaries_must_record_one_training_identity(world, tmp_path):
    scenario = Study(world, tmp_path)
    trial = scenario.plan["trials"][0]
    provenance = copy.deepcopy(world.provenance[(trial["trial_id"], 2)])
    provenance["training"]["parameter_names"] = provenance["training"]["parameter_names"][:-1]
    scenario.rewrite(trial, 2, provenance=provenance)
    with pytest.raises(ValueError, match="different training identities"):
        scenario.select(loader=labels_opened)


def edit_bases(provenance, change):
    """Apply one edit to every copy of the base identity a post checkpoint records."""
    training = provenance["training"]
    for base in (training["model"]["base"], training["initialization"]["base"], training["runtime"]["model"]["base"]):
        change(base)


def replanned(world, arm_id, origin_id, **declared):
    spec = copy.deepcopy(world.spec)
    arm = next(a for a in spec["arms"] if a["arm_id"] == arm_id)
    arm["origins"][[o["origin_id"] for o in arm["origins"]].index(origin_id)].update(declared)
    return spec, study.build_plan(spec, world.manifest, world.manifest_sha)


LINEAGE = {
    "post_trained_parent": (lambda b: put(b, ("parent_provenance", "training", "config", "stage"), "post")),
    "incomplete_parent": (lambda b: put(b, ("parent_provenance", "progress", "next_update"), 1)),
    "parent_with_fitting_exposure": (lambda b: put(b, ("parent_provenance", "progress", "fitting_endpoints_available"), 1)),
    "parent_with_selection_labels": (lambda b: put(b, ("parent_provenance", "progress", "selection_labels"), 1)),
    "parent_selected_on_development": (lambda b: put(b, ("parent_provenance", "training", "selection"), "best")),
    "parent_with_inherited_exposure": (lambda b: put(b, ("parent_provenance", "progress", "inherited_exposure"), {"x": 1})),
}


@pytest.mark.parametrize("name", sorted(LINEAGE))
def test_a_post_trained_or_measured_parent_is_never_accepted_as_a_midtraining_origin(world, tmp_path, name):
    trial = Study(world, tmp_path).trial(*MID_N2)
    provenance = copy.deepcopy(world.provenance[(trial["trial_id"], 1)])
    edit_bases(provenance, LINEAGE[name])
    # The declared origin hash is updated to the edited provenance, so the hash does not hide the defect.
    declared = digest(provenance["training"]["model"]["base"]["parent_provenance"])
    _, plan = replanned(world, A_STAGE, "mid-0", parent_provenance_sha256=declared)
    scenario = Study(world, tmp_path, plan=plan)
    scenario.rewrite(trial, 1, provenance=provenance)
    with pytest.raises(ValueError, match="completed, unmeasured midtraining parent"):
        scenario.select(loader=labels_opened)


def test_a_release_checkpoint_cannot_hide_a_materialized_parent(world, tmp_path):
    scenario = Study(world, tmp_path)
    trial = scenario.trial(*RELEASE_N2)
    provenance = copy.deepcopy(world.provenance[(trial["trial_id"], 1)])
    edit_bases(provenance, lambda base: base.update(variant="materialized-parent"))
    scenario.rewrite(trial, 1, provenance=provenance)
    with pytest.raises(ValueError, match="cannot hide a materialized parent"):
        scenario.select(loader=labels_opened)


def test_a_release_checkpoint_must_come_from_the_declared_backbone(world, tmp_path):
    spec = copy.deepcopy(world.spec)
    spec["identity"]["release_base_sha256"] = hexn(9)
    scenario = Study(world, tmp_path, plan=study.build_plan(spec, world.manifest, world.manifest_sha))
    with pytest.raises(ValueError, match="declared release backbone"):
        scenario.select(loader=labels_opened)


@pytest.mark.parametrize("declared,message", (
    ({"parent_sha256": hexn(999)}, "parent other than origin 'mid-0'"),
    ({"parent_checkpoint_sha256": hexn(999)}, "parent provenance is not the one declared"),
    ({"parent_provenance_sha256": hexn(999)}, "parent provenance is not the one declared"),
))
def test_each_origin_must_be_the_declared_real_parent(world, tmp_path, declared, message):
    _, plan = replanned(world, A_STAGE, "mid-0", **declared)
    with pytest.raises(ValueError, match=message):
        Study(world, tmp_path, plan=plan).select(loader=labels_opened)


@pytest.mark.parametrize("arm_id,mode", ((A_STAGE, "shared"), (A_STAGE, "detached"), (D_STAGE, None), (D_STAGE, "shared")))
def test_a_declared_functional_mode_must_be_the_one_the_parent_trained_with(world, tmp_path, arm_id, mode):
    spec = copy.deepcopy(world.spec)
    next(a for a in spec["arms"] if a["arm_id"] == arm_id)["functional_mode"] = mode
    scenario = Study(world, tmp_path, plan=study.build_plan(spec, world.manifest, world.manifest_sha))
    with pytest.raises(ValueError, match="declares functional_mode"):
        scenario.select(loader=labels_opened)


@pytest.mark.parametrize("edit", (
    lambda s: s["recipe"]["training"].update(examples_per_update=2),
    lambda s: s["recipe"]["fusion"].update(lora_rank=3),
    lambda s: s["identity"].update(source_sha256=hexn(9)),
    lambda s: [s["identity"]["encoder"]["identity"].update(tokenizer="other"),
                s["identity"]["encoder"]["runtime"]["encoder"].update(tokenizer="other")],
    lambda s: s["identity"]["execution"]["hardware"].update(device="cuda"),
    lambda s: s["query"].update(orders=3),
    lambda s: s["query"].update(order_seed=8),
    lambda s: s["query"].update(scorer_id="other"),
    lambda s: s["query"].update(protocol_id="other"),
    lambda s: s["query"].update(context_arm="abonly"),
))
def test_a_changed_declaration_rejects_every_export_recorded_under_the_old_one(world, tmp_path, edit):
    spec = copy.deepcopy(world.spec)
    edit(spec)
    plan = study.build_plan(spec, world.manifest, world.manifest_sha)         # each declaration is itself valid
    with pytest.raises(ValueError):
        Study(world, tmp_path, plan=plan).select(loader=labels_opened)


def test_pins_declared_for_other_labels_reject_every_export_of_that_budget(world, tmp_path):
    for field in ("selected_data_sha256", "selected_labels_sha256"):
        spec = copy.deepcopy(world.spec)
        spec["budgets"]["entries"][0][field] = hexn(9)
        scenario = Study(world, tmp_path, plan=study.build_plan(spec, world.manifest, world.manifest_sha))
        with pytest.raises(ValueError, match=field.replace("_sha256", "")):
            scenario.select(loader=labels_opened)


def test_a_different_manifest_file_hash_rejects_the_plan_before_any_export_is_read(world, tmp_path):
    scenario = Study(world, tmp_path)
    scenario.results["records"][0]["export_path"] = "exports/never-read"
    with pytest.raises(ValueError, match="derive"):
        study.select_study(scenario.plan, world.manifest, scenario.results, labels_opened, base=world.dir,
                           files={"plan": hexn(1), "manifest": hexn(8), "results": hexn(3)})


# ---------------------------------------------------------------------------
# The development labels
# ---------------------------------------------------------------------------

LABEL_PROBLEMS = {
    "schema": (lambda d, p: d.update(schema="trastuzumab-study-development-labels/2"), "unsupported"),
    "other_plan": (lambda d, p: d.update(plan_sha256=hexn(9)), "different plan"),
    "other_assay": (lambda d, p: d.update(assay_id="OTHER"), "another assay or orientation"),
    "other_orientation": (lambda d, p: d.update(orientation="lower_better"), "another assay or orientation"),
    "mixed_orientation": (lambda d, p: d["labels"][0].update(orientation="lower_better"), "one assay and one orientation"),
    "label_in_another_assay": (lambda d, p: d["labels"][0].update(assay_id="OTHER"), "one assay and one orientation"),
    "duplicate_label": (lambda d, p: d["labels"].append(dict(d["labels"][0])), "duplicate"),
    "missing_label": (lambda d, p: d["labels"].pop(), "missing"),
    "fitting_label_smuggled_in": (lambda d, p: d["labels"].append(
        {"endpoint_id": "p0", "value": 1.0, "orientation": "higher_better", "assay_id": "SPR"}), "extra"),
    "reserved_label_smuggled_in": (lambda d, p: d["labels"].append(
        {"endpoint_id": "r0", "value": 1.0, "orientation": "higher_better", "assay_id": "SPR"}), "extra"),
    "unknown_field": (lambda d, p: d["labels"][0].update(note="x"), "unknown or missing"),
    "boolean_value": (lambda d, p: d["labels"][0].update(value=True), "numeric"),
    "extra_top_level": (lambda d, p: d.update(split="reserved"), "expected exactly"),
    "labels_not_a_list": (lambda d, p: d.update(labels={}), "must be a list"),
}


@pytest.mark.parametrize("name", sorted(LABEL_PROBLEMS))
def test_the_development_label_document_is_exact(world, tmp_path, name):
    scenario = Study(world, tmp_path)
    document = labels_document(scenario.plan)
    change, message = LABEL_PROBLEMS[name]
    change(document, scenario.plan)
    with pytest.raises((ValueError, TypeError), match=message):
        scenario.select(document)


def test_a_development_set_with_no_comparable_pair_is_refused(world, tmp_path):
    scenario = Study(world, tmp_path)
    with pytest.raises(ValueError, match="undefined"):
        scenario.select(labels_document(scenario.plan, {eid: 5.0 for eid in DEV_IDS}))


def test_the_labels_are_read_exactly_once_and_only_after_everything_validated(world, tmp_path):
    scenario = Study(world, tmp_path)
    opened = []
    document = labels_document(scenario.plan)

    def loader():
        opened.append(True)
        return document, hexn(2)

    scenario.select(loader=loader)
    assert opened == [True]


# ---------------------------------------------------------------------------
# One read per file: a recorded hash names the bytes that were parsed
# ---------------------------------------------------------------------------

def test_the_snapshot_reader_accepts_a_bom_and_pretty_printing_and_hashes_what_it_parsed(tmp_path):
    document = {"b": [1, 2.5, None, "café"], "a": {"c": "x"}}
    data = pretty(document)
    path = tmp_path / "doc.json"
    path.write_bytes(data)
    assert study.read_snapshot(path) == (document, sha256_of(data))
    assert study.read_snapshot(path) == (read_json(path), file_digest(path))          # what it replaces, for one read


@pytest.mark.parametrize("data,message", (
    (b'{"a": 1, "a": 2}', "duplicate JSON key"),
    (b'{"a": NaN}', "nonfinite JSON"),
    (b'{"a": -Infinity}', "nonfinite JSON"),
    (b'{"a": 1e999}', "finite"),
))
def test_the_snapshot_reader_keeps_the_strict_json_rules(tmp_path, data, message):
    path = tmp_path / "doc.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        study.read_snapshot(path)
    with pytest.raises(ValueError, match=message):
        read_json(path)                                                              # the reader it mirrors agrees


@pytest.mark.parametrize("change,message", (
    (lambda d: d.update(schema="trastuzumab-endpoints/2"), "unsupported endpoint schema"),
    (lambda d: d.update(entries={}), "expected a sequence"),
    (lambda d: d.update(extra=1), "expected exactly"),
    (lambda d: d["entries"][0].update(extra=1), "unknown or missing fields"),
    (lambda d: d["entries"][0]["construct"].update(extra=1), "unknown or missing fields"),
))
def test_the_manifest_snapshot_keeps_the_existing_loaders_contract(tmp_path, change, message):
    path = tmp_path / "endpoints.json"
    path.write_bytes(pretty(manifest_document()))
    manifest, sha256 = study._manifest_snapshot(path)
    assert sha256 == file_digest(path)
    assert manifest.manifest_digest() == inputs.load_endpoint_manifest(path).manifest_digest()
    document = manifest_document()
    change(document)
    path.write_bytes(canonical_bytes(document))
    with pytest.raises(ValueError, match=message):
        study._manifest_snapshot(path)
    with pytest.raises(ValueError, match=message):
        inputs.load_endpoint_manifest(path)


def test_the_planner_hashes_the_manifest_bytes_it_parsed_when_the_file_is_replaced_after_reading(
        world, tmp_path, monkeypatch):
    original, replacement = pretty(manifest_document()), canonical_bytes(manifest_document(antigen="HIL"))
    manifest_path = tmp_path / "endpoints.json"
    manifest_path.write_bytes(original)
    write(tmp_path / "spec.json", world.spec_document)
    with monkeypatch.context() as patch:
        state = replaced_after_first_read(patch, manifest_path, replacement)
        plan = study.plan_study(tmp_path / "spec.json")
    assert state["reads"] == 1 and manifest_path.read_bytes() == replacement          # the replacement did happen
    assert plan == study.build_plan(world.spec, world.manifest, sha256_of(original))
    assert plan["manifest"]["file_sha256"] == sha256_of(original) != sha256_of(replacement)
    assert plan["antigen_sha256"] == world.plan["antigen_sha256"]             # the panel is the parsed manifest's


FILE_FIELDS = {"plan": "plan_file_sha256", "manifest": "manifest_file_sha256", "results": "results_file_sha256",
               "labels": "development_labels_file_sha256"}


@pytest.mark.parametrize("replaced", sorted(FILE_FIELDS))
def test_selection_binds_the_bytes_it_parsed_when_a_file_is_replaced_after_reading(
        world, tmp_path, monkeypatch, replaced):
    results = copy.deepcopy(world.results)
    for row in results["records"]:
        row["export_path"] = str(world.dir / row["export_path"])
    labels = labels_document(world.plan)
    # Pretty-printed with a BOM wherever the checkpoints do not pin the exact manifest bytes.
    originals = {"plan": pretty(world.plan), "manifest": (world.dir / "endpoints.json").read_bytes(),
                 "results": pretty(results), "labels": pretty(labels)}
    paths = {name: tmp_path / f"{name}.json" for name in originals}
    for name, data in originals.items():
        paths[name].write_bytes(data)
    replacement = b'{"replaced": true}\n'
    with monkeypatch.context() as patch:
        state = replaced_after_first_read(patch, paths[replaced], replacement)
        result = study.select_study_files(paths["plan"], paths["manifest"], paths["results"], paths["labels"])
    assert state["reads"] == 1 and paths[replaced].read_bytes() == replacement        # the replacement did happen
    for name, field in FILE_FIELDS.items():
        assert result["inputs"][field] == sha256_of(originals[name]), name
    assert result["inputs"]["results_digest"] == digest(results)
    assert result["inputs"]["development_labels_digest"] == digest(labels)
    assert {(s["arm_id"], s["budget_n"]): s["selected_learning_rate"] for s in result["selected"]} == {
        key: value[0] for key, value in EXPECTED.items()}


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def run(*args):
    cli.main([str(arg) for arg in args])


def forbid(patch):
    def boom(*args, **kwargs):
        raise AssertionError("opened a model, a checkpoint or a fitting label")

    for owner, name in ((cli, "models"), (cli, "load_inference_checkpoint"), (inputs, "load_post"),
                        (artifacts, "score_queries")):
        patch.setattr(owner, name, boom)


def test_cli_plans_without_labels_or_models_and_refuses_to_overwrite(world, tmp_path, monkeypatch):
    shutil.copy(world.dir / "endpoints.json", tmp_path / "endpoints.json")
    write(tmp_path / "spec.json", world.spec_document)
    with monkeypatch.context() as patch:
        forbid(patch)
        run("plan-study", "--spec", tmp_path / "spec.json", "--output", tmp_path / "plan.json")
        before = (tmp_path / "plan.json").read_bytes()
        with pytest.raises(ValueError, match="output already exists"):
            run("plan-study", "--spec", tmp_path / "spec.json", "--output", tmp_path / "plan.json")
        assert (tmp_path / "plan.json").read_bytes() == before
    plan = read_json(tmp_path / "plan.json")
    assert plan == world.plan and study.validate_plan(plan, world.manifest, world.manifest_sha) == plan
    assert before == canonical_bytes(plan) + b"\n"
    assert plan["manifest"]["file_sha256"] == file_digest(tmp_path / "endpoints.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["endpoints.json", "plan.json", "spec.json"]


def test_cli_planning_failure_writes_nothing(world, tmp_path, monkeypatch):
    document = copy.deepcopy(world.spec_document)
    document["budgets"]["entries"][1].pop("selected_data_sha256")
    write(tmp_path / "endpoints.json", manifest_document())
    write(tmp_path / "spec.json", document)
    with monkeypatch.context() as patch:
        forbid(patch)
        with pytest.raises(ValueError, match="expected exactly"):
            run("plan-study", "--spec", tmp_path / "spec.json", "--output", tmp_path / "plan.json")
    assert not (tmp_path / "plan.json").exists()


def test_cli_selects_after_validating_and_binds_the_files_it_read(world, tmp_path, monkeypatch):
    plan_path = write(tmp_path / "plan.json", world.plan)
    labels_path = write(tmp_path / "labels.json", labels_document(world.plan))
    arguments = ("--plan", plan_path, "--manifest", world.dir / "endpoints.json",
                 "--results", world.dir / "results.json", "--labels", labels_path)
    with monkeypatch.context() as patch:
        forbid(patch)
        run("select-study", *arguments, "--output", tmp_path / "selection.json")
        with pytest.raises(ValueError, match="output already exists"):
            run("select-study", *arguments, "--output", tmp_path / "selection.json")
    result = read_json(tmp_path / "selection.json")
    assert result["schema"] == study.SELECTION_SCHEMA
    assert result["result_sha256"] == digest({k: v for k, v in result.items() if k != "result_sha256"})
    bound = result["inputs"]
    assert bound["plan_sha256"] == world.plan["plan_sha256"] and bound["plan_file_sha256"] == file_digest(plan_path)
    assert bound["manifest_file_sha256"] == world.manifest_sha
    assert bound["results_file_sha256"] == file_digest(world.dir / "results.json")
    assert bound["development_labels_file_sha256"] == file_digest(labels_path)
    assert {(s["arm_id"], s["budget_n"]): s["selected_learning_rate"] for s in result["selected"]} == {
        key: value[0] for key, value in EXPECTED.items()}
    assert (tmp_path / "selection.json").read_bytes() == canonical_bytes(result) + b"\n"


def test_cli_selection_failure_never_opens_the_labels_and_never_writes(world, tmp_path):
    plan_path = write(tmp_path / "plan.json", world.plan)
    damaged = copy.deepcopy(world.results)
    damaged["records"].pop(0)
    results_path = write(tmp_path / "results.json", damaged)
    with pytest.raises(ValueError, match="not the planned inventory"):
        run("select-study", "--plan", plan_path, "--manifest", world.dir / "endpoints.json",
            "--results", results_path, "--labels", tmp_path / "must-not-open.json",
            "--output", tmp_path / "selection.json")
    assert not (tmp_path / "selection.json").exists()


def test_cli_invalid_labels_write_nothing(world, tmp_path):
    plan_path = write(tmp_path / "plan.json", world.plan)
    labels_path = write(tmp_path / "labels.json", labels_document(world.plan, plan_sha256=hexn(9)))
    with pytest.raises(ValueError, match="different plan"):
        run("select-study", "--plan", plan_path, "--manifest", world.dir / "endpoints.json",
            "--results", world.dir / "results.json", "--labels", labels_path, "--output", tmp_path / "selection.json")
    assert not (tmp_path / "selection.json").exists()


def test_cli_parser_keeps_the_existing_commands_and_requires_every_study_flag():
    parser = cli.parser()
    assert parser.parse_args(["plan-study", "--spec", "s", "--output", "o"]).command == "plan-study"
    args = parser.parse_args(["select-study", "--plan", "p", "--manifest", "m", "--results", "r",
                              "--labels", "l", "--output", "o"])
    assert (args.plan, args.manifest, args.results, args.labels) == ("p", "m", "r", "l")
    with pytest.raises(SystemExit):
        parser.parse_args(["select-study", "--plan", "p", "--manifest", "m", "--results", "r", "--output", "o"])
    with pytest.raises(SystemExit):
        parser.parse_args(["plan-study", "--spec", "s"])
    assert parser.parse_args(["plan-retention", "--spec", "s", "--output", "o"]).command == "plan-retention"
    assert parser.parse_args(["select-budget", "--manifest", "m", "--budget", "3", "--budget-seed", "x",
                              "--output", "o"]).budget == 3

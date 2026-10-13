"""A small synthetic HER2 world on disk, for the pilot's input and runner tests.

Not a test module (nothing here is collected). It writes, under a temporary directory:

* a fixed scaffold (patched in for ``her2_data.load_scaffold``) and an AlphaBind-shaped CSV with two
  replicate rows per core: sixteen TRAIN cores forming a four-site binary lattice (so there are
  pairs at Hamming distance 1, 2 and 3 and complete quartets), two DEVELOPMENT cores, one RESERVED
  core and two cores EXCLUDED for lying within the purge radius of a held-out core;
* the frozen split record those sets imply, a frozen condition vocabulary and a frozen source
  inventory. The inventory is derived by the REAL ``FrozenInputServices`` (``adopt`` with the
  restored split and closure, the vocabulary installed, then ``build_sources``) over this world and
  saved BEFORE ``load_inputs`` is ever called; production has no bootstrap mode and the fixture
  never writes an empty inventory for it to run under;
* a base routes config derived from the REAL ``her2_generative_routes_dplm_train.json`` (so the
  Adafactor route block, the curriculum and the closed ``training_phase`` schema are the real ones,
  re-pinned to this world's digests), a pilot config bound to it by digest, tiny DPLM pins and the
  acquisition manifests ``verify_source_identities`` reads;
* the one seam of the real runtime that cannot run here: ``dplm_policy.load_pinned_dplm`` is replaced
  by a recorder that returns a real ``NativeDPLM.random_init`` (32 wide, 2 layers, 4 heads, 256
  positions). Nothing else is faked: collation, the DPLM conditioning wrapper, the sequential joint,
  the objectives, the heads and Adafactor all run for real.

Two things are also patched because they read the user's disk: the paired OAS release (a handful of
synthetic constructs carrying the real record's fields) and ``her2_data.load_scaffold``.

The synthetic CODE root holds only configs until :meth:`World.install_code` copies the pilot's real
static source closure into it, so ``runner.code_identity`` hashes real bytes and refuses a root
without them.
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch

from smallAntibodyGen.experiments import her2_data
from smallAntibodyGen.experiments import her2_replay_spec as replay_spec
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_sources as sources
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_routes_split as split_lib
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_multiscale import config as config_lib
from smallAntibodyGen.experiments.her2_multiscale import inputs as inputs_lib
from smallAntibodyGen.experiments.her2_multiscale import membership as membership_lib
from smallAntibodyGen.experiments.her2_multiscale import roots as roots_lib
from smallAntibodyGen.models import dplm_policy as dplm

REPO = Path(__file__).resolve().parents[3]
REAL_BASE = REPO / "configs" / "experiments" / "her2_generative_routes_dplm_train.json"
REAL_PILOT = REPO / "configs" / "experiments" / "her2_multiscale_pilot.json"

ASSAY = "alphabind_her2:affinity:HER2"
WILD = "WGGDGFYAMD"
LATTICE = {0: "A", 2: "S", 4: "T", 6: "C"}
DEVELOPMENT = ("CCCCCCCCCC", "DDDDDDDDDD")
RESERVED = ("EEEEEEEEEE",)
EXCLUDED = ("CCCCCCCCCA", "EEEEEEEEAK")
SOURCE_LOGICAL = "data/alphabind/her2.csv"
ROUTE = config_lib.ROUTE
CLOSURE_ENTRY = "scripts/her2_multiscale_pilot.py"


def fake_scaffold():
    heavy = "A" * 98 + WILD + "YWGQGTLVTVSS"
    assert len(heavy) == her2_data.HEAVY_LENGTH
    return her2_data.Scaffold(heavy=heavy, light="D" * her2_data.LIGHT_LENGTH)


def lattice_cores():
    cores = []
    for bits in itertools.product((0, 1), repeat=len(LATTICE)):
        core = list(WILD)
        for (offset, alternative), bit in zip(sorted(LATTICE.items()), bits):
            if bit:
                core[offset] = alternative
        cores.append("".join(core))
    return cores


def tiny_native_config():
    return dplm.DPLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                           intermediate_size=64, max_position_embeddings=256,
                           hidden_dropout_prob=0.0)


def write_alphabind(path, scaffold, cores):
    """Two replicate rows per core: ``alphaseq_affinity = 9 - pKD`` (log10 KD/nM)."""
    rows, counter = [], 0
    for index, core in enumerate(cores):
        for replicate in range(2):
            sequence = scaffold.with_core(core) + sources.ALPHASEQ_LINKER + scaffold.light + "R"
            assert len(sequence) == sources.ALPHASEQ_CONSTRUCT_LENGTH
            rows.append({"id": f"row{counter}", "mata_description": "synthetic",
                         "mata_sequence": sequence,
                         "alphaseq_affinity": repr(9.0 - (7.0 + 0.1 * index + 0.02 * replicate))})
            counter += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def split_record(settings, candidates):
    return {"schema_version": split_lib.SPLIT_SCHEMA, "record_kind": "hard_split_freeze",
            "settings": dict(settings),
            "counts": {"candidates": len(candidates), "development": len(DEVELOPMENT),
                       "reserved": len(RESERVED), "retained_for_training": len(lattice_cores()),
                       "purged_within_radius": len(EXCLUDED), "excluded": len(EXCLUDED)},
            "sequence_sha256": split_lib.sequence_digest(candidates), "algorithm_sha256": "0" * 64,
            "development": list(DEVELOPMENT), "reserved": list(RESERVED),
            "excluded": {"within_2_edits_of_a_held_out_core": sorted(EXCLUDED)},
            "selection": {"note": "synthetic"}}


def write_pins(code):
    files = {name: {"bytes": 10, "sha256": "0" * 64} for name in
             ("config.json", "vocab.txt", "pytorch_model.bin")}
    document = {"schema": "her2-routes-dplm-pins/1", "variants": {
        variant: {"repo": f"local/{variant}", "revision": "a" * 40,
                  "logical_dir": f"models/{variant}", "files": copy.deepcopy(files),
                  "architecture": {}} for variant in ("dplm_150m", "dplm_650m")}}
    target = code / "configs" / "experiments" / "dplm_pins.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document), encoding="utf-8")
    return files


def write_manifests(artifacts, csv_path, pin_files):
    data = artifacts / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "acquisition_manifest.json").write_text(json.dumps({"files": [{
        "path": SOURCE_LOGICAL, "sha256": paths.sha256_file(csv_path),
        "bytes": csv_path.stat().st_size}]}), encoding="utf-8")
    (data / "dplm_acquisition_manifest.json").write_text(json.dumps({"files": [
        {"path": f"models/dplm_150m/{name}", "sha256": block["sha256"], "bytes": block["bytes"]}
        for name, block in pin_files.items()]}), encoding="utf-8")


def install_code(code_root):
    """Copy the real static source closure of the pilot's entry script into ``code_root``.

    The closure is computed from this repository (``her2_replay_spec.source_closure``) and copied
    file by file, so a code root built here holds the real bytes the identity is computed over.
    """
    closure = replay_spec.source_closure(REPO, [CLOSURE_ENTRY])
    for name in closure["files"]:
        target = Path(code_root) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    return closure


OAS_SHA256 = "ab" * 32


def paired_oas_record():
    """A paired-OAS release record with the fields the real ``load_paired_oas`` returns."""
    constructs = tuple(serial.Construct(
        heavy=("ACDEFGHIKLMNPQRSTVWY" * 3)[index:index + 40], light="YWVTSRQPNMLKIHGFEDCA" * 2,
        sites=(("H", 3),), source="paired_oas", record_id=f"oas{index}") for index in range(4))
    return {"constructs": constructs, "by_split": {"train": constructs, "val": ()},
            "rows_in_file": 4, "rows_seen_by_split": {"train": 4, "val": 0},
            "other_splits_present": {}, "excluded": {}, "studies": 2,
            "top_studies": {"synthetic_a": 2, "synthetic_b": 2}, "limit_per_split": 4,
            "splits": ["train", "val"], "seed": 7, "member": "synthetic.jsonl.gz",
            "sha256": OAS_SHA256, "sampling": "synthetic", "outcome": "none"}


def prepared_environment():
    """The record ``prepare_environment`` returns, as it looks once ordering has been respected."""
    return {"artifact_root_configured": True, "cache_variables": {}, "sys_path_inserted": [],
            "declared_paths_absent": [], "imported_before_this_call": [],
            "already_prepared_before_scientific_imports": False, "offline": True}


@dataclass
class World:
    root: Path
    code: Path
    data: Path
    artifacts: Path
    frozen: Path
    base_path: Path
    pilot_path: Path
    scaffold: object
    candidates: tuple
    settings: dict
    record: dict
    policy: dict
    loads: list = field(default_factory=list)

    @property
    def counts(self):
        return {"candidates": len(self.candidates), "development": len(DEVELOPMENT),
                "reserved": len(RESERVED), "excluded": len(EXCLUDED),
                "retained_for_training": len(lattice_cores())}

    def run_root(self, name="run"):
        parent = self.root / "runs"
        parent.mkdir(parents=True, exist_ok=True)
        return parent / name

    def declared_inputs(self):
        """The data and historical locations this world's base config declares, resolved."""
        base = json.loads(self.base_path.read_text(encoding="utf-8"))
        return roots_lib.declared_input_roots(base, data_repo_root=self.data)

    def reserve(self, name="run"):
        return roots_lib.reserve_run_root(
            self.run_root(name), code_root=self.code, data_repo_root=self.data,
            artifact_root=self.artifacts, frozen_inputs=self.frozen,
            declared_inputs=self.declared_inputs())

    def install_code(self):
        """Copy the pilot's REAL static source closure into the synthetic code root.

        ``runner.code_identity`` hashes the closure of ``scripts/her2_multiscale_pilot.py`` from the
        code root it is given and refuses a root that lacks it, so a run over this world needs the
        real bytes there.
        """
        return install_code(self.code)

    def derive_inventory(self, name="inventory-source"):
        """Write ``source_inventory.json`` from the real services, over this world, without a pilot.

        Exactly the stages of ``load_inputs`` that produce an inventory: the frozen vocabulary is
        installed into a fresh run root, the split is restored from its record and its closure
        built, ``FrozenInputServices.adopt`` installs both, and the inherited ``build_sources``
        derives the inventory. Nothing is compared, so there is no empty-inventory mode to bypass.
        """
        pilot = self.pilot()
        context = self.context(self.reserve(name), pilot)
        vocabulary = self.frozen / inputs_lib.VOCABULARY_FILE
        inputs_lib.install_frozen_vocabulary(context, {"vocabulary": {
            "path": str(vocabulary), "file_sha256": paths.sha256_file(vocabulary)}})
        services = inputs_lib.FrozenInputServices(context, device="cpu",
                                                  source_cache=context.roots.source_cache)
        scaffold = services.scaffold()
        candidates = membership_lib.candidate_windows(
            services.source_paths()[inputs_lib.HER2_SOURCE], scaffold=scaffold)
        membership = membership_lib.restore_membership(
            self.record, record_sha256=paths.digest_document(self.record),
            candidates=candidates["windows"],
            expected_split_sha256=self.policy["expected_split_sha256"],
            expected_sequence_sha256=self.policy["expected_sequence_sha256"],
            split_settings=context.config["split"], expected_counts=pilot.expected_counts)
        services.adopt(membership, membership_lib.build_closure(membership, scaffold=scaffold))
        inventory = inputs_lib.plain(services.build_sources(retain_observations=True))
        paths.write_json(self.frozen / "source_inventory.json", inventory)
        return inventory

    def pilot(self, path=None):
        return config_lib.load_pilot_config(path or self.pilot_path, code_root=self.code)

    def write_pilot(self, name, mutate):
        document = json.loads(self.pilot_path.read_text(encoding="utf-8"))
        mutate(document)
        target = self.code / "configs" / "experiments" / f"{name}.json"
        target.write_text(json.dumps(document), encoding="utf-8")
        return target

    def context(self, roots, pilot):
        base = spec.load_config(pilot.base_path)
        return roots_lib.PilotContext(
            config=inputs_lib.narrowed_config(base), config_path=pilot.base_path, roots=roots,
            revision="multiscale_pilot", campaign_id=pilot.pilot_id)

    def load(self, name="run", *, pilot=None, policy=None):
        """``(loaded inputs, context)`` over a freshly reserved run root."""
        pilot = pilot or self.pilot()
        roots = self.reserve(name)
        context = self.context(roots, pilot)
        loaded = inputs_lib.load_inputs(pilot, context, policy=policy or self.policy,
                                        frozen_dir=self.frozen, device="cpu")
        return loaded, context


def build_world(tmp_path, monkeypatch, *, quartet_probability=0.25, curriculum=None):
    root = Path(tmp_path)
    code, data, artifacts, frozen = (root / name for name in ("code", "data", "artifacts", "frozen"))
    for directory in (code, data, artifacts, frozen):
        directory.mkdir(parents=True, exist_ok=True)
    scaffold = fake_scaffold()
    monkeypatch.setattr(her2_data, "load_scaffold", lambda _root: scaffold)
    monkeypatch.setattr(inputs_lib.FrozenInputServices, "paired_oas",
                        lambda self: paired_oas_record())
    cores = lattice_cores() + list(DEVELOPMENT) + list(RESERVED) + list(EXCLUDED)
    csv_path = artifacts / SOURCE_LOGICAL
    write_alphabind(csv_path, scaffold, cores)
    pin_files = write_pins(code)
    write_manifests(artifacts, csv_path, pin_files)

    base = json.loads(REAL_BASE.read_text(encoding="utf-8"))
    candidates = tuple(sorted(cores))
    record = split_record(base["split"], candidates)
    digest, sequence = paths.digest_document(record), record["sequence_sha256"]
    paths.write_json(frozen / spec.SPLIT_FREEZE, record)
    monkeypatch.setattr(spec, "TRAINING_PHASE_SPLIT_SHA256", digest)
    monkeypatch.setattr(spec, "TRAINING_PHASE_SEQUENCE_SHA256", sequence)
    base["training_phase"]["expected_split_sha256"] = digest
    base["training_phase"]["expected_sequence_sha256"] = sequence
    base["artifacts"]["root"] = str(artifacts)
    block = base["routes"][ROUTE]
    block.update(activation_checkpointing=False, dropout=0.0, max_positions=256)
    block["diffusion"] = dict(block["diffusion"], num_timesteps=8, decode_steps=4)
    base["conditions"].update(assays=[ASSAY], minimum_bin_support=4)
    base["sources"] = {"paths": {inputs_lib.HER2_SOURCE: {"root": "artifact",
                                                          "logical": SOURCE_LOGICAL}},
                       "her2_display": {"enrolled": False}, "withheld": {}}
    base["roots"] = {"her2_raw": "data/raw/her2", "paired_oas": "data/oas", "historical": {}}
    base["her2"]["assay"] = ASSAY
    base["optimization"].update(microbatch=4, accumulation=2, effective_batch=8, warmup=2)
    base["curriculum"].update(local_quartet_probability=float(quartet_probability),
                              **dict(curriculum or {}))
    base["runtime"] = {}
    base_path = code / "configs" / "experiments" / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")

    pilot_document = json.loads(REAL_PILOT.read_text(encoding="utf-8"))
    pilot_document["base"] = {"config": "configs/experiments/base.json",
                              "sha256": config_lib.file_sha256(base_path)}
    pilot_document["updates"] = 4
    pilot_document["local"]["reservoir_capacity"] = 64
    pilot_document["local"]["replacement_probability"] = 1.0
    pilot_document["inputs"]["expected_counts"] = {
        "candidates": len(candidates), "development": len(DEVELOPMENT),
        "reserved": len(RESERVED), "excluded": len(EXCLUDED),
        "retained_for_training": len(lattice_cores())}
    pilot_document["diagnostics"] = {"batch_size": 2, "completion_rows": 2,
                                     "completion_counts": [1, 4, 10], "order_rows": 2,
                                     "fit_pairs_per_distance": 4}
    pilot_path = code / "configs" / "experiments" / "pilot.json"
    pilot_path.write_text(json.dumps(pilot_document), encoding="utf-8")

    world = World(root=root, code=code, data=data, artifacts=artifacts, frozen=frozen,
                  base_path=base_path, pilot_path=pilot_path, scaffold=scaffold,
                  candidates=candidates, settings=dict(base["split"]), record=record,
                  policy={"expected_split_sha256": digest, "expected_sequence_sha256": sequence})

    def recording_loader(weights_dir, *, pins, variant, device="cpu", storage_dtype=torch.float32,
                         dropout=None):
        world.loads.append({"variant": variant, "directory": Path(weights_dir)})
        return dplm.NativeDPLM.random_init(tiny_native_config(), seed=3, std=0.3, label=variant,
                                           storage_dtype=storage_dtype)
    monkeypatch.setattr(dplm, "load_pinned_dplm", recording_loader)

    # The vocabulary and the inventory are what a production pass over this world records. The
    # vocabulary's counts are the train-partition observation rows per assay; the inventory is
    # derived by the real services (``World.derive_inventory``) and saved before ``load_inputs``
    # is ever called.
    train_rows = 2 * len(lattice_cores())
    paths.write_json(frozen / "condition_vocabulary.json", {
        "schema_version": spec.ROUTES_SCHEMA, "record_kind": "condition_vocabulary",
        "assays": [ASSAY], "counts": {ASSAY: train_rows}})
    world.derive_inventory()
    assert world.loads == [], "deriving the inventory must not build a model"
    return world


def cycling_stream(base, accumulation):
    """The real curriculum stream, with the task and family cycled so every task is exercised.

    reconstruction (replay), local (HER2), whole window (HER2), heads (HER2), repeating. The rows,
    constructs, conditions and contrasts are the real stream's; only the schedule is fixed.
    """
    from smallAntibodyGen.experiments import her2_routes_data as data_lib
    plan = [(data_lib.TASK_RECONSTRUCTION, data_lib.FAMILY_REPLAY),
            (data_lib.TASK_LOCAL, data_lib.FAMILY_HER2),
            (data_lib.TASK_WHOLE, data_lib.FAMILY_HER2),
            (data_lib.TASK_HEADS, data_lib.FAMILY_HER2)]

    def stream(step, micro, rng):
        task, family = plan[((int(step) - 1) * int(accumulation) + int(micro)) % len(plan)]
        return base(step, micro, rng, task_override=task, family_override=family)
    return stream

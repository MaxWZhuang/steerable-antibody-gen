"""The acquired-only frozen-feature comparison matrix, its cache format and its CLI.

Tests ``experiments/functional_matrix.py`` and ``scripts/functional_matrix.py``: the versioned feature
cache (strict JSON header + numeric NPZ), the exact roster join, the acquired-only readout maps
(hand-computed), the frozen plan, the freeze index that is written, reloaded and verified before any
evaluation outcome is opened, the paired group bootstrap and the output-safety rules. Everything is
NumPy on CPU with tiny synthetic fixtures; nothing imports a native model. Expected values come from
hand derivations and independent NumPy computations, not from re-running the implementation. What
this establishes is that the machinery computes what it says; it says nothing about any
representation, readout, native model or biology, and no test requires a representation to win.

No test asserts on source text.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smallAntibodyGen.experiments import functional_baseline as fb
from smallAntibodyGen.experiments import functional_matrix as fm

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "functional_matrix.py"
SRC = Path(__file__).resolve().parents[2]

ALPHABET, LENGTH = "ACDE", 4
LABELLED, UNLABELLED, HELD_OUT = 24, 2, 8
TOTAL = LABELLED + UNLABELLED + HELD_OUT
CANDIDATE_HEADER = "candidate_id,sequence,group_id,split,prior_log_weight"


# ---------------------------------------------------------------------------
# helpers: a small synthetic world, its files, and runners
# ---------------------------------------------------------------------------

def load_cli():
    spec = importlib.util.spec_from_file_location("functional_matrix_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_free():
    return fb.SupervisionHistory(status="label_free", exposure_status="none")


def explicit_space(prefix, width, supervision=None):
    names = tuple(f"{prefix}{i}" for i in range(width))
    return fb.FeatureSpace(kind="explicit", names=names, schema={"columns": list(names)},
                           history=("explicit synthetic columns",),
                           supervision=supervision or label_free())


def identity_document():
    return {"kind": "complete_sequence", "alphabet": ALPHABET, "length": LENGTH,
            "construct_description": "one chain of four residues; the whole sequence"}


def extractor_document():
    return {"kind": "synthetic", "checkpoint_sha256": None, "source_sha256": None,
            "description": "synthetic test features"}


def activation_document():
    return {"location": "synthetic", "pooling": "none", "token_map": "residue letters"}


def conditioning_document(count, *, family=None):
    return {"candidate_masking": "none", "answer_free": True,
            "family": {"recorded": False} if family is None
            else {"recorded": True, "values": family},
            "assay": {"recorded": True, "values": ["synthetic-assay"] * count},
            "quality": {"recorded": True, "values": [None] * count}}


def manifest_document(**overrides):
    document = {
        "schema": fm.MANIFEST_SCHEMA, "target": "T", "assay": "A",
        "utility": {"units": "u", "scale": "linear", "higher_is_better": True,
                    "useful_threshold": 0.0},
        "grouping_rationale": "training candidates are singleton groups; held-out groups are pairs",
        "candidates": {"identity_kind": "complete_sequence", "alphabet": ALPHABET,
                       "length": LENGTH},
        "prior": {"kind": "score_normalized", "semantics": "random scores, one structural zero",
                  "provenance": {"status": "label_free",
                                 "sequence_only_exposure": {"status": "none"}}}}
    document.update(overrides)
    return document


def plan_document(**overrides):
    document = {
        "schema": fm.PLAN_SCHEMA, "representations": ["alpha", "beta"],
        "maps": [{"id": "linear", "kind": "linear", "center": True, "scale": "std",
                  "trace_normalize": True},
                 {"id": "rbf", "kind": "rbf", "center": True, "scale": "std",
                  "trace_normalize": True,
                  "bandwidth": {"rule": "median_heuristic", "multiplier": 1.0}}],
        "label_budgets": [6, 12], "seeds": [0, 1],
        "baseline": {"tau": 1.0, "ridge_alpha": 0.5, "sir_proposal_draws": 64, "final_draws": 16,
                     "dpo_pressures": [0.5, 1.0]},
        "reference": {"representation": "alpha", "map": "linear"},
        "bootstrap": {"replicates": 40, "seed": 0, "confidence": 0.9, "min_groups": 3}}
    document.update(overrides)
    return document


def small_plan(**overrides):
    """One representation, both maps, one budget, one seed: two cells."""
    return plan_document(representations=["alpha"], label_budgets=[6], seeds=[0], **overrides)


def write_csv(path, header, rows):
    Path(path).write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""),
                          encoding="utf-8")
    return Path(path)


def prior_text(value):
    return "-inf" if value == -math.inf else repr(float(value))


class World:
    """34 candidates: 24 labelled and 2 unlabelled training rows (singleton groups) and 8 held-out
    rows (4 groups of 2), with features for two informative-or-not representations."""

    def __init__(self, seed=11, pinned=None):
        rng = np.random.default_rng(seed)
        pinned = dict(pinned or {})
        pool = [s for s in ("".join(t) for t in itertools.product(ALPHABET, repeat=LENGTH))
                if s not in pinned.values()]
        chosen = iter([pool[int(i)] for i in rng.choice(len(pool), size=TOTAL - len(pinned),
                                                        replace=False)])
        self.sequences = [pinned[i] if i in pinned else next(chosen) for i in range(TOTAL)]
        self.ids = [f"c{i:02d}" for i in range(TOTAL)]
        self.splits = ["train"] * (LABELLED + UNLABELLED) + ["eval"] * HELD_OUT
        self.groups = ([f"gt{i:02d}" for i in range(LABELLED + UNLABELLED)]
                       + [f"ge{i // 2}" for i in range(HELD_OUT)])
        self.prior = [float(x) for x in rng.normal(size=TOTAL)]
        self.prior[LABELLED + UNLABELLED + 1] = -math.inf
        self.features = {"alpha": rng.normal(size=(TOTAL, 4)), "beta": rng.normal(size=(TOTAL, 3))}
        weights = rng.normal(size=4)
        truth = self.features["alpha"] @ weights + 0.05 * rng.normal(size=TOTAL)
        self.utility = {self.ids[i]: float(truth[i]) for i in range(TOTAL)}

    @property
    def labelled_ids(self):
        return self.ids[:LABELLED]

    @property
    def eval_ids(self):
        return self.ids[LABELLED + UNLABELLED:]

    def eval_truth(self):
        return {ident: self.utility[ident] for ident in self.eval_ids}

    def variant(self, *, features=None, utility=None):
        other = copy.deepcopy(self)
        for name, matrix in (features or {}).items():
            other.features[name] = np.array(matrix, dtype=np.float64)
        if utility is not None:
            other.utility = dict(utility)
        return other

    def cache_args(self, rep, order=None, supervision=None):
        order = list(range(TOTAL)) if order is None else list(order)
        matrix = self.features[rep]
        return dict(
            representation_id=rep, ids=[self.ids[i] for i in order],
            sequences=[self.sequences[i] for i in order], groups=[self.groups[i] for i in order],
            splits=[self.splits[i] for i in order], features=matrix[order],
            space=explicit_space(rep[:2], matrix.shape[1], supervision),
            identity=identity_document(), extractor=extractor_document(),
            activation=activation_document(), conditioning=conditioning_document(len(order)),
            dtype="float64")

    def write(self, root, *, permute=False, plan=None, manifest=None, supervision=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        order, cache_order = list(range(TOTAL)), list(range(TOTAL))
        if permute:
            order = np.random.default_rng(5).permutation(TOTAL).tolist()
            cache_order = np.random.default_rng(6).permutation(TOTAL).tolist()
        candidates = write_csv(root / "candidates.csv", CANDIDATE_HEADER, [
            f"{self.ids[i]},{self.sequences[i]},{self.groups[i]},{self.splits[i]},"
            f"{prior_text(self.prior[i])}" for i in order])
        train = write_csv(root / "train_outcomes.csv", "candidate_id,utility", [
            f"{self.ids[i]},{self.utility[self.ids[i]]!r}" for i in order if i < LABELLED])
        evaluation = write_csv(root / "eval_outcomes.csv", "candidate_id,utility", [
            f"{self.ids[i]},{self.utility[self.ids[i]]!r}" for i in order
            if i >= LABELLED + UNLABELLED])
        manifest_path = fb.write_json(root / "matrix_manifest.json", manifest or manifest_document())
        plan_path = fb.write_json(root / "matrix_plan.json", plan or plan_document())
        caches, npz = {}, {}
        for rep in self.features:
            header, npz_path = root / f"{rep}.cache.json", root / f"{rep}.cache.npz"
            fm.write_feature_cache(header, npz_path, overwrite=True,
                                   **self.cache_args(rep, cache_order, (supervision or {}).get(rep)))
            caches[rep], npz[rep] = header, npz_path
        return SimpleNamespace(root=root, candidates=candidates, train=train, eval=evaluation,
                               manifest=manifest_path, plan=plan_path, caches=caches, npz=npz)


def run(paths, out, *, overwrite=False, **replace):
    reps = fm.load_plan(paths.plan).representations
    kwargs = dict(candidates=paths.candidates, train_outcomes=paths.train,
                  eval_outcomes=paths.eval, manifest=paths.manifest, plan=paths.plan,
                  caches={rep: paths.caches[rep] for rep in reps},
                  cache_npz={rep: paths.npz[rep] for rep in reps})
    kwargs.update(replace)
    return fm.run_matrix(output_dir=out, overwrite=overwrite, **kwargs)


def acquired_expected(offered, budget, seed):
    """The documented rule, written independently: the first ``budget`` of one seeded permutation."""
    return np.sort(np.random.default_rng([seed, 1]).permutation(offered)[:budget]).tolist()


def edit_header(header_path, mutate):
    document = json.loads(Path(header_path).read_text(encoding="utf-8"))
    mutate(document)
    Path(header_path).write_text(json.dumps(document), encoding="utf-8")


def refresh_file_hash(header_path, npz_path):
    digest = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()
    edit_header(header_path, lambda doc: doc["storage"].__setitem__("file_sha256", digest))


def fresh_cache(tmp_path, rep="alpha", world=None, **overrides):
    world = world or World()
    header, npz = Path(tmp_path) / f"{rep}.cache.json", Path(tmp_path) / f"{rep}.cache.npz"
    args = world.cache_args(rep)
    args.update(overrides)
    fm.write_feature_cache(header, npz, **args)
    return header, npz, world


def tree_bytes(root, names):
    return {name: (Path(root) / name).read_bytes() for name in names}


def pre_evaluation_names(out):
    out = Path(out)
    names = sorted(f"{fm.CELLS_DIR}/{item.name}" for item in (out / fm.CELLS_DIR).iterdir())
    return names + [fm.INDEX_NAME, fm.INPUTS_NAME]


@pytest.fixture(scope="module")
def matrix_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("matrix")
    world = World()
    paths = world.write(root / "in")
    out = root / "out"
    summary = run(paths, out)
    return SimpleNamespace(world=world, paths=paths, out=out, summary=summary, root=root)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("small")
    world = World()
    paths = world.write(root / "in", plan=small_plan())
    out = root / "out"
    summary = run(paths, out)
    return SimpleNamespace(world=world, paths=paths, out=out, summary=summary, root=root)


# ---------------------------------------------------------------------------
# the feature cache: strict JSON header + numeric NPZ
# ---------------------------------------------------------------------------

def test_a_cache_round_trips_as_strict_json_plus_a_plain_numeric_npz(tmp_path):
    world = World()
    order = np.random.default_rng(3).permutation(TOTAL).tolist()
    header, npz = tmp_path / "a.cache.json", tmp_path / "a.cache.npz"
    args = world.cache_args("alpha", order)
    args["dtype"] = "float32"
    fm.write_feature_cache(header, npz, **args)
    document = json.loads(header.read_text(encoding="utf-8"))
    assert document["schema"] == fm.CACHE_SCHEMA and "prior" not in document
    with np.load(npz, allow_pickle=False) as archive:
        assert archive.files == ["features"]
        stored = archive["features"]
    assert stored.dtype == np.float32 and stored.shape == (TOTAL, 4)
    cache = fm.load_feature_cache(header)
    assert cache.ids == tuple(sorted(world.ids))
    canonical = world.features["alpha"].astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(cache.features, canonical)
    assert cache.features.dtype == np.float64 and not cache.features.flags.writeable
    assert cache.cache_sha256 == document["cache_sha256"]
    assert cache.sequences == tuple(world.sequences)      # ids are c00.. so canonical = original


def test_the_canonical_digest_ignores_row_order_but_the_raw_file_digest_does_not(tmp_path):
    world = World()
    a_header, a_npz, _ = fresh_cache(tmp_path / "a", world=world)
    order = np.random.default_rng(4).permutation(TOTAL).tolist()
    b_header, b_npz = tmp_path / "b" / "alpha.cache.json", tmp_path / "b" / "alpha.cache.npz"
    fm.write_feature_cache(b_header, b_npz, **world.cache_args("alpha", order))
    first, second = fm.load_feature_cache(a_header), fm.load_feature_cache(b_header)
    assert first.cache_sha256 == second.cache_sha256
    assert first.ids == second.ids
    np.testing.assert_array_equal(first.features, second.features)
    assert first.raw["file_sha256"] != second.raw["file_sha256"]
    assert first.raw["header_sha256"] != second.raw["header_sha256"]


def _cross_groups(doc):
    splits = doc["rows"]["splits"]
    train, held = splits.index("train"), splits.index("eval")
    doc["rows"]["groups"][train] = doc["rows"]["groups"][held]


HEADER_REFUSALS = [
    ("duplicate_id", lambda d: d["rows"]["ids"].__setitem__(1, d["rows"]["ids"][0]),
     "repeats a candidate id"),
    ("duplicate_sequence",
     lambda d: d["rows"]["sequences"].__setitem__(1, d["rows"]["sequences"][0]),
     "same physical sequence"),
    ("group_crossing", _cross_groups, "appear in both"),
    ("bad_split", lambda d: d["rows"]["splits"].__setitem__(0, "val"), "must be 'train' or 'eval'"),
    ("an_edited_window_is_not_a_construct",
     lambda d: d["rows"]["sequences"].__setitem__(0, "CD"), "full-construct length"),
    ("residue_outside_alphabet",
     lambda d: d["rows"]["sequences"].__setitem__(0, "W" + d["rows"]["sequences"][0][1:]),
     "outside the alphabet"),
    ("lowercase_sequence",
     lambda d: d["rows"]["sequences"].__setitem__(0, d["rows"]["sequences"][0].lower()),
     "not canonical"),
    ("heavy_light_identity", lambda d: d["identity"].update(kind="heavy_light"),
     "unsupported identity kind"),
    ("a_cache_carries_a_prior", lambda d: d.update(prior={"kind": "uniform"}),
     "cannot carry a prior"),
    ("unknown_key", lambda d: d.update(extra=1), "unknown key"),
    ("quality_is_an_outcome_bin",
     lambda d: d["conditioning"].update(
         quality={"recorded": True, "values": ["high"] + [None] * (len(d["rows"]["ids"]) - 1)}),
     "non-null values"),
    ("quality_is_absent", lambda d: d["conditioning"].update(quality={"recorded": False}),
     "EXPLICIT null"),
    ("candidates_were_masked", lambda d: d["conditioning"].update(candidate_masking="hcdr3"),
     "UNMASKED"),
    ("conditioning_not_answer_free", lambda d: d["conditioning"].update(answer_free=False),
     "answer_free must be true"),
    ("unsupported_dtype", lambda d: d["storage"].update(dtype="float16"), "storage.dtype"),
    ("npz_traversal", lambda d: d["storage"].update(file="../alpha.cache.npz"),
     "plain file name"),
    ("npz_drive_letter", lambda d: d["storage"].update(file="C:/alpha.cache.npz"),
     "plain file name"),
    ("npz_in_a_subdirectory", lambda d: d["storage"].update(file="sub/alpha.cache.npz"),
     "plain file name"),
    ("header_shape_disagrees", lambda d: d["storage"].update(shape=[1, 1]), "storage.shape"),
    ("feature_names_disagree_with_dimension", lambda d: d["feature_space"]["names"].pop(),
     "storage.shape"),
    ("native_without_hashes", lambda d: d["extractor"].update(kind="native_checkpoint"),
     "must bind BOTH"),
    ("malformed_hash", lambda d: d["extractor"].update(checkpoint_sha256="abc"), "64-character"),
    ("wrong_schema", lambda d: d.update(schema="functional-feature-cache/2"),
     "not a functional-feature-cache/1"),
    ("unsafe_representation_id", lambda d: d.update(representation_id="../evil"),
     "representation_id"),
]


@pytest.mark.parametrize("name,mutate,pattern", HEADER_REFUSALS,
                         ids=[item[0] for item in HEADER_REFUSALS])
def test_a_header_that_breaks_the_cache_contract_is_refused(tmp_path, name, mutate, pattern):
    header, _, _ = fresh_cache(tmp_path)
    edit_header(header, mutate)
    with pytest.raises(ValueError, match=pattern):
        fm.load_feature_cache(header)


def test_a_replaced_npz_or_an_edited_header_makes_the_cache_stale(tmp_path):
    header, npz, world = fresh_cache(tmp_path / "npz")
    np.savez(npz, features=world.features["alpha"] + 1.0)
    with pytest.raises(ValueError, match="STALE"):
        fm.load_feature_cache(header)
    header, npz, world = fresh_cache(tmp_path / "header")

    def rename_a_group(doc):
        doc["rows"]["groups"][doc["rows"]["splits"].index("train")] = "renamed-group"
    edit_header(header, rename_a_group)
    with pytest.raises(ValueError, match="cache_sha256"):
        fm.load_feature_cache(header)
    header, npz, world = fresh_cache(tmp_path / "content")
    edit_header(header, lambda doc: doc["storage"].update(content_sha256="0" * 64))
    with pytest.raises(ValueError, match="content_sha256"):
        fm.load_feature_cache(header)
    header, npz, world = fresh_cache(tmp_path / "edited_array")
    np.savez(npz, features=world.features["alpha"] * 1.0000001)
    refresh_file_hash(header, npz)
    with pytest.raises(ValueError, match="content_sha256"):
        fm.load_feature_cache(header)


def _object_array(world):
    return np.array(world.features["alpha"].tolist(), dtype=object)


@pytest.mark.parametrize("name,build,pattern", [
    ("nan", lambda w: np.where(np.arange(TOTAL * 4).reshape(TOTAL, 4) == 5, np.nan,
                               w.features["alpha"]), "NaN or infinity"),
    ("infinity", lambda w: np.where(np.arange(TOTAL * 4).reshape(TOTAL, 4) == 7, np.inf,
                                    w.features["alpha"]), "NaN or infinity"),
    ("dtype", lambda w: w.features["alpha"].astype(np.float32), "dtype"),
    ("shape", lambda w: w.features["alpha"][:-1], "shape"),
    ("object_dtype", _object_array, "pickle"),
])
def test_an_npz_with_consistent_hashes_but_bad_numeric_content_is_refused(tmp_path, name, build,
                                                                         pattern):
    header, npz, world = fresh_cache(tmp_path)
    np.savez(npz, features=build(world))
    refresh_file_hash(header, npz)
    with pytest.raises(ValueError, match=pattern):
        fm.load_feature_cache(header)


def test_an_npz_with_an_extra_member_or_no_archive_is_refused(tmp_path):
    header, npz, world = fresh_cache(tmp_path / "extra")
    np.savez(npz, features=world.features["alpha"], other=world.features["alpha"])
    refresh_file_hash(header, npz)
    with pytest.raises(ValueError, match="exactly one array"):
        fm.load_feature_cache(header)
    header, npz, world = fresh_cache(tmp_path / "garbage")
    npz.write_bytes(b"this is not a zip archive")
    refresh_file_hash(header, npz)
    with pytest.raises(ValueError, match="pickle|cannot be read"):
        fm.load_feature_cache(header)


def test_the_writer_refuses_bad_inputs_before_writing_anything(tmp_path):
    world = World()
    args = world.cache_args("alpha")
    header, npz = tmp_path / "w" / "x.cache.json", tmp_path / "w" / "x.cache.npz"
    bad_nan = {**args, "features": np.where(np.arange(TOTAL * 4).reshape(TOTAL, 4) == 3, np.nan,
                                            args["features"])}
    with pytest.raises(ValueError, match="finite"):
        fm.write_feature_cache(header, npz, **bad_nan)
    with pytest.raises(ValueError, match="overflowed"):
        fm.write_feature_cache(header, npz, **{**args, "dtype": "float32",
                                               "features": args["features"] * 1e300})
    with pytest.raises(ValueError, match="must be \\("):
        fm.write_feature_cache(header, npz, **{**args, "features": args["features"][:, :2]})
    with pytest.raises(ValueError, match="full-construct length"):
        fm.write_feature_cache(header, npz, **{**args, "sequences": ["CD"] * TOTAL})
    with pytest.raises(ValueError, match="repeats a candidate id"):
        fm.write_feature_cache(header, npz, **{**args, "ids": ["c00"] * TOTAL})
    with pytest.raises(ValueError, match="header's directory"):
        fm.write_feature_cache(header, tmp_path / "elsewhere" / "x.cache.npz", **args)
    assert not header.exists() and not npz.exists(), "a refusal must be write-free"
    fm.write_feature_cache(header, npz, **args)
    with pytest.raises(ValueError, match="already exists"):
        fm.write_feature_cache(header, npz, **args)
    fm.write_feature_cache(header, npz, overwrite=True, **args)


def test_conditioning_distinguishes_an_explicit_null_from_an_absent_field(tmp_path):
    world = World()
    absent = {**world.cache_args("alpha"), "conditioning": conditioning_document(TOTAL)}
    explicit = {**world.cache_args("alpha"),
                "conditioning": conditioning_document(TOTAL, family=[None] * TOTAL)}
    valued = {**world.cache_args("alpha"),
              "conditioning": conditioning_document(TOTAL, family=["fam-a"] * (TOTAL - 1) + [None])}
    loaded = {}
    for name, args in (("absent", absent), ("explicit", explicit), ("valued", valued)):
        header = tmp_path / name / "x.cache.json"
        fm.write_feature_cache(header, tmp_path / name / "x.cache.npz", **args)
        loaded[name] = fm.load_feature_cache(header)
    assert loaded["absent"].conditioning_summary["family"] == {"state": "not_recorded"}
    assert loaded["explicit"].conditioning_summary["family"] == {"state": "explicit_null",
                                                                  "rows": TOTAL}
    assert loaded["valued"].conditioning_summary["family"] == {
        "state": "values", "rows": TOTAL, "null_rows": 1, "distinct_values": 1}
    for cache in loaded.values():
        assert cache.conditioning_summary["quality"]["state"] == "explicit_null"
        assert cache.conditioning_summary["assay"]["state"] == "values"
    assert len({cache.cache_sha256 for cache in loaded.values()}) == 3, \
        "absence, explicit null and a value are different statements and must hash differently"


# ---------------------------------------------------------------------------
# the roster join: exact, on the full construct, no convenient intersection
# ---------------------------------------------------------------------------

def roster_for(world, tmp_path):
    paths = world.write(tmp_path / "roster")
    manifest = fm.load_matrix_manifest(paths.manifest)
    table = fb.read_candidates(paths.candidates, alphabet=manifest.alphabet, length=manifest.length)
    return manifest, table


def written_cache(tmp_path, args, name="c"):
    header, npz = tmp_path / name / "x.cache.json", tmp_path / name / "x.cache.npz"
    fm.write_feature_cache(header, npz, **args)
    return fm.load_feature_cache(header)


def unused_sequence(world):
    return next(s for s in ("".join(t) for t in itertools.product(ALPHABET, repeat=LENGTH))
                if s not in world.sequences)


def test_a_complete_matching_cache_joins(tmp_path):
    world = World()
    manifest, table = roster_for(world, tmp_path)
    cache = written_cache(tmp_path, world.cache_args("alpha", np.random.default_rng(2)
                                                      .permutation(TOTAL).tolist()))
    assert fm.check_cache_against_roster(cache, table, manifest) is True


@pytest.mark.parametrize("case,pattern", [
    ("missing", "lacks"), ("extra", "outside the roster"),
    ("swapped_sequences", "different physical sequence"), ("group", "different group"),
    ("split", "different split")])
def test_the_join_refuses_missing_extra_and_mismatched_rows(tmp_path, case, pattern):
    world = World()
    manifest, table = roster_for(world, tmp_path)
    args = world.cache_args("alpha")
    if case == "missing":
        args = {**args, **{k: v[:-1] for k, v in args.items()
                           if k in ("ids", "sequences", "groups", "splits", "features")}}
        # the per-row conditioning must follow the rows, or the cache itself is malformed and the
        # roster join this case is about is never reached
        args["conditioning"] = conditioning_document(len(args["ids"]))
    elif case == "extra":
        args = {**args, "ids": args["ids"] + ["zz99"],
                "sequences": args["sequences"] + [unused_sequence(world)],
                "groups": args["groups"] + ["gz"], "splits": args["splits"] + ["train"],
                "features": np.vstack([args["features"], np.zeros((1, 4))])}
        args["conditioning"] = conditioning_document(len(args["ids"]))
    elif case == "swapped_sequences":
        sequences = list(args["sequences"])
        sequences[0], sequences[1] = sequences[1], sequences[0]
        args = {**args, "sequences": sequences}
    elif case == "group":
        groups = list(args["groups"])
        groups[0] = "some-other-group"
        args = {**args, "groups": groups}
    else:
        splits = list(args["splits"])
        splits[0] = "eval"                      # c00's singleton group stays valid in the cache
        args = {**args, "splits": splits}
    cache = written_cache(tmp_path, args)
    with pytest.raises(ValueError, match=pattern):
        fm.check_cache_against_roster(cache, table, manifest)


def test_conditioning_that_does_not_list_one_entry_per_row_is_still_refused(tmp_path):
    # the fixtures above must supply per-row conditioning of the right length; the validation that
    # made them fail stays strict
    args = World().cache_args("alpha")
    short = {**args, **{k: v[:-1] for k, v in args.items()
                        if k in ("ids", "sequences", "groups", "splits", "features")}}
    assert len(short["conditioning"]["assay"]["values"]) == TOTAL > len(short["ids"])
    with pytest.raises(ValueError, match="one entry per row"):
        fm.write_feature_cache(tmp_path / "x.cache.json", tmp_path / "x.cache.npz", **short)
    assert not list(tmp_path.iterdir()), "a refusal must be write-free"


def test_one_edited_window_on_two_different_full_constructs_cannot_misjoin(tmp_path):
    # c00 = A|CD|A and c01 = D|CD|E share the window CD (positions 1-2) and differ outside it.
    world = World(pinned={0: "ACDA", 1: "DCDE"})
    manifest, table = roster_for(world, tmp_path)
    args = world.cache_args("alpha")
    sequences, features = list(args["sequences"]), args["features"].copy()
    sequences[0], sequences[1] = sequences[1], sequences[0]
    features[[0, 1]] = features[[1, 0]]          # a window-keyed join would accept this swap
    cache = written_cache(tmp_path, {**args, "sequences": sequences, "features": features})
    with pytest.raises(ValueError, match="different physical sequence") as caught:
        fm.check_cache_against_roster(cache, table, manifest)
    assert "c00" in str(caught.value)
    paths = world.write(tmp_path / "swapped", plan=small_plan())
    fm.write_feature_cache(paths.caches["alpha"], paths.npz["alpha"], overwrite=True,
                           **{**args, "sequences": sequences, "features": features})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="different physical sequence"):
        run(paths, out)
    assert not out.exists(), "a refused join must write nothing"


def test_the_cache_and_the_manifest_must_agree_on_the_declared_identity(tmp_path):
    world = World()
    _, table = roster_for(world, tmp_path)
    cache = written_cache(tmp_path, world.cache_args("alpha"))
    wider = fm.parse_matrix_manifest(manifest_document(
        candidates={"identity_kind": "complete_sequence", "alphabet": "ACDEF", "length": LENGTH}))
    with pytest.raises(ValueError, match="declares alphabet"):
        fm.check_cache_against_roster(cache, table, wider)
    longer = fm.parse_matrix_manifest(manifest_document(
        candidates={"identity_kind": "complete_sequence", "alphabet": ALPHABET,
                    "length": LENGTH + 1}))
    with pytest.raises(ValueError, match="declares alphabet"):
        fm.check_cache_against_roster(cache, table, longer)


def test_a_cache_for_another_representation_than_the_plan_names_is_refused(tmp_path):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    with pytest.raises(ValueError, match="not the planned"):
        run(paths, tmp_path / "out", caches={"alpha": paths.caches["beta"]}, cache_npz={})
    with pytest.raises(ValueError, match="exactly the planned"):
        run(paths, tmp_path / "out2", caches={"alpha": paths.caches["alpha"],
                                              "beta": paths.caches["beta"]})


MANIFEST_REFUSALS = [
    ("schema", lambda d: d.update(schema="x"), "not a"),
    ("unknown_key", lambda d: d.update(extra=1), "unknown key"),
    ("missing_prior", lambda d: d.pop("prior"), "missing"),
    ("heavy_light", lambda d: d["candidates"].update(identity_kind="heavy_light"),
     "unsupported identity kind"),
    ("higher_is_better_false", lambda d: d["utility"].update(higher_is_better=False),
     "higher_is_better"),
    ("missing_units", lambda d: d["utility"].pop("units"), "missing"),
    ("repeated_alphabet_letter", lambda d: d["candidates"].update(alphabet="AA"), "distinct"),
    ("unknown_prior_kind", lambda d: d["prior"].update(kind="mystery"), "prior kind"),
]


@pytest.mark.parametrize("name,mutate,pattern", MANIFEST_REFUSALS,
                         ids=[item[0] for item in MANIFEST_REFUSALS])
def test_the_matrix_manifest_is_strict_and_infers_nothing(name, mutate, pattern):
    document = manifest_document()
    mutate(document)
    with pytest.raises(ValueError, match=pattern):
        fm.parse_matrix_manifest(document)
    assert fm.parse_matrix_manifest(manifest_document()).length == LENGTH


# ---------------------------------------------------------------------------
# the frozen plan and the safe cell names
# ---------------------------------------------------------------------------

PLAN_REFUSALS = [
    ("unknown_key", lambda d: d.update(tune=True), "unknown key"),
    ("baseline_unknown_option", lambda d: d["baseline"].update(learning_rate=1.0), "unknown key"),
    ("baseline_without_tau", lambda d: d["baseline"].pop("tau"), "missing"),
    ("declared_sir_proposal", lambda d: d["baseline"].update(sir_proposal="declared"),
     "declared"),
    ("seed_smuggled_into_baseline", lambda d: d["baseline"].update(seed=3), "unknown key"),
    ("too_few_bootstrap_replicates", lambda d: d["bootstrap"].update(replicates=3),
     "bootstrap replicates"),
    ("confidence_out_of_range", lambda d: d["bootstrap"].update(confidence=1.5), "confidence"),
    ("reference_map_not_planned", lambda d: d["reference"].update(map="nope"), "planned map"),
    ("reference_representation_not_planned",
     lambda d: d["reference"].update(representation="zzz"), "planned representation"),
    ("repeated_budget", lambda d: d.update(label_budgets=[6, 6]), "repeats a budget"),
    ("zero_budget", lambda d: d.update(label_budgets=[0]), "label budget"),
    ("empty_seeds", lambda d: d.update(seeds=[]), "non-empty"),
    ("negative_seed", lambda d: d.update(seeds=[-1]), "seed"),
    ("path_fragment_as_representation", lambda d: d.update(representations=["alpha", "a/b"]),
     "representation id"),
    ("ids_that_differ_only_by_case", lambda d: d.update(representations=["alpha", "Alpha"]),
     "ignoring letter case"),
    ("rbf_without_bandwidth", lambda d: d["maps"][1].pop("bandwidth"), "bandwidth"),
    ("linear_with_a_bandwidth",
     lambda d: d["maps"][0].update(bandwidth={"rule": "fixed", "value": 1.0}), "no bandwidth"),
    ("tuned_bandwidth_rule", lambda d: d["maps"][1]["bandwidth"].update(rule="cv_tuned"),
     "bandwidth rule"),
    ("zero_multiplier", lambda d: d["maps"][1]["bandwidth"].update(multiplier=0), "positive"),
    ("unknown_scale_mode", lambda d: d["maps"][0].update(scale="unit"), "map scale"),
    ("unsafe_map_id", lambda d: d["maps"][0].update(id="has space"), "map id"),
]


@pytest.mark.parametrize("name,mutate,pattern", PLAN_REFUSALS,
                         ids=[item[0] for item in PLAN_REFUSALS])
def test_a_plan_with_an_unsafe_or_tuned_setting_is_refused(name, mutate, pattern):
    document = plan_document()
    mutate(document)
    with pytest.raises(ValueError, match=pattern):
        fm.parse_plan(document)


def test_a_plan_is_canonical_and_round_trips_through_its_own_document():
    shuffled = plan_document(representations=["beta", "alpha"], label_budgets=[12, 6, None],
                             seeds=[1, 0])
    shuffled["maps"] = list(reversed(shuffled["maps"]))
    plan = fm.parse_plan(shuffled)
    assert plan.representations == ("alpha", "beta") and plan.seeds == (0, 1)
    assert plan.label_budgets == (6, 12, None)
    assert [spec.id for spec in plan.maps] == ["linear", "rbf"]
    again = fm.parse_plan(plan.document())
    assert again.document() == plan.document() and again.fingerprint() == plan.fingerprint()
    assert fm.parse_plan(plan_document()).fingerprint() != plan.fingerprint()
    assert plan.baseline["tau"] == 1.0 and plan.baseline["dpo_pressures"] == [0.5, 1.0]


def test_cell_file_names_come_from_validated_ids_and_a_hash_never_an_arbitrary_fragment():
    plan = fm.parse_plan(plan_document(label_budgets=[6, None]))
    keys = fm.enumerate_cells(plan)
    assert len(keys) == 2 * 2 * 2 * 2
    names = [key.filename() for key in keys]
    assert len(set(names)) == len(names) and len({key.cell_id for key in keys}) == len(keys)
    for name in names:
        assert os.path.basename(name) == name and "/" not in name and "\\" not in name
        assert name.startswith("cell_") and name.endswith(".json") and " " not in name
        assert re_hash(name)
    assert fm.CellKey("alpha", "linear", None, 3).cell_id == "alpha|linear|ball|s3"


def re_hash(name):
    stem = name[:-len(".json")]
    tail = stem.rsplit("_", 1)[1]
    return len(tail) == 12 and all(c in "0123456789abcdef" for c in tail)


# ---------------------------------------------------------------------------
# the readout maps, by hand
# ---------------------------------------------------------------------------

def rbf_spec(*, center=False, scale="none", trace=False, bandwidth=None):
    return fm.MapSpec(id="r", kind="rbf", center=center, scale=scale, trace_normalize=trace,
                      bandwidth=bandwidth or {"rule": "median_heuristic", "multiplier": 1.0})


def linear_spec(*, center=True, scale="none", trace=False):
    return fm.MapSpec(id="l", kind="linear", center=center, scale=scale, trace_normalize=trace)


def test_the_rbf_map_entries_centers_and_bandwidth_match_a_hand_computation():
    # centers (0,0) and (3,4): one pairwise distance, 5, so sigma = 5 and phi = exp(-d^2 / 50).
    readout = fm.fit_readout(rbf_spec(), [[0.0, 0.0], [3.0, 4.0]], ridge_alpha=1.0)
    assert readout.centers.tolist() == [[0.0, 0.0], [3.0, 4.0]]
    assert readout.mean.tolist() == [0.0, 0.0] and readout.scale.tolist() == [1.0, 1.0]
    assert readout.bandwidth["median_pairwise_distance"] == pytest.approx(5.0)
    assert readout.bandwidth["sigma"] == pytest.approx(5.0)
    mapped = readout.transform([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    half, two = math.exp(-0.5), math.exp(-2.0)
    np.testing.assert_allclose(mapped, [[1.0, half], [half, 1.0], [two, half]], rtol=1e-12)
    assert readout.output_dimension == 2 and readout.input_dimension == 2


def test_rbf_centering_scaling_and_the_bandwidth_rules_match_a_hand_computation():
    # rows 0 and 2: mean 1, population std 1, standardised centers -1 and +1 (distance 2).
    # x = 5 standardises to 4: squared distances 25 and 9.
    readout = fm.fit_readout(rbf_spec(center=True, scale="std"), [[0.0], [2.0]], ridge_alpha=1.0)
    assert readout.mean.tolist() == [1.0] and readout.scale.tolist() == [1.0]
    assert readout.centers.tolist() == [[-1.0], [1.0]]
    assert readout.bandwidth["sigma"] == pytest.approx(2.0)
    np.testing.assert_allclose(readout.transform([[5.0]]),
                               [[math.exp(-25 / 8), math.exp(-9 / 8)]], rtol=1e-12)
    half = fm.fit_readout(rbf_spec(center=True, scale="std",
                                   bandwidth={"rule": "median_heuristic", "multiplier": 0.5}),
                          [[0.0], [2.0]], ridge_alpha=1.0)
    np.testing.assert_allclose(half.transform([[5.0]]),
                               [[math.exp(-25 / 2), math.exp(-9 / 2)]], rtol=1e-12)
    fixed = fm.fit_readout(rbf_spec(center=True, scale="std",
                                    bandwidth={"rule": "fixed", "value": 3.0}),
                           [[0.0], [2.0]], ridge_alpha=1.0)
    assert fixed.bandwidth["median_pairwise_distance"] is None
    np.testing.assert_allclose(fixed.transform([[5.0]]),
                               [[math.exp(-25 / 18), math.exp(-9 / 18)]], rtol=1e-12)


def test_trace_normalisation_sets_the_mean_squared_centred_norm_to_one_by_hand():
    # design [[1, a], [a, 1]], a = exp(-1/2): centred squares sum to (1 - a)^2, so the factor is
    # sqrt(2) / (1 - a) and the centred acquired design then has mean squared norm 1.
    spec = rbf_spec(trace=True)
    readout = fm.fit_readout(spec, [[0.0, 0.0], [3.0, 4.0]], ridge_alpha=1.0)
    a = math.exp(-0.5)
    assert readout.trace_factor == pytest.approx(math.sqrt(2.0) / (1.0 - a), rel=1e-12)
    mapped = readout.transform([[0.0, 0.0], [3.0, 4.0]])
    centred = mapped - mapped.mean(axis=0)
    assert float((centred ** 2).sum() / 2) == pytest.approx(1.0, rel=1e-12)
    plain = fm.fit_readout(rbf_spec(trace=False), [[0.0, 0.0], [3.0, 4.0]], ridge_alpha=1.0)
    assert plain.trace_factor == 1.0


def test_dimension_rank_and_effective_ridge_df_match_hand_values():
    two = fm.fit_readout(linear_spec(), [[1.0], [-1.0]], ridge_alpha=1.0)
    assert two.output_dimension == 1 and two.capacity["design_rank"] == 1
    assert two.capacity["effective_ridge_df"] == pytest.approx(2.0 / 3.0)
    # X'X = diag(8, 2): df = 8/(8+2) + 2/(2+2) = 1.3 for alpha = 2.
    rows = [[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    plain = fm.fit_readout(linear_spec(), rows, ridge_alpha=2.0)
    assert plain.capacity["design_rank"] == 2
    assert plain.capacity["effective_ridge_df"] == pytest.approx(1.3)
    # trace normalisation divides the spectrum by 2.5: s^2 = 3.2 and 0.8
    scaled = fm.fit_readout(linear_spec(trace=True), rows, ridge_alpha=2.0)
    assert scaled.capacity["effective_ridge_df"] == pytest.approx(3.2 / 5.2 + 0.8 / 2.8)
    # the same alpha shrinks differently: trace normalisation does not equalise geometry
    assert scaled.capacity["effective_ridge_df"] < plain.capacity["effective_ridge_df"]


def test_constant_columns_are_kept_explicitly_and_contribute_exactly_zero():
    rows = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    readout = fm.fit_readout(linear_spec(scale="std"), rows, ridge_alpha=1.0)
    assert readout.constant_columns == (1,) and readout.output_dimension == 2
    np.testing.assert_allclose(readout.scale, [math.sqrt(2.0 / 3.0), 1.0])
    assert readout.mean.tolist() == [2.0, 5.0]
    assert np.all(readout.transform(rows)[:, 1] == 0.0)
    np.testing.assert_allclose(readout.transform([[2.0, 9.0]]), [[0.0, 4.0]])
    assert readout.capacity["design_rank"] == 1


def test_all_identical_acquired_rows_leave_a_degenerate_design_and_no_median_bandwidth():
    rows = [[1.0, 2.0]] * 3
    linear = fm.fit_readout(linear_spec(scale="std", trace=True), rows, ridge_alpha=1.0)
    assert linear.degenerate_design is True and linear.trace_factor == 1.0
    assert linear.capacity["effective_ridge_df"] == 0.0 and linear.capacity["design_rank"] == 0
    assert np.all(linear.transform(rows) == 0.0)
    np.testing.assert_allclose(linear.transform([[2.0, 2.0]]), [[1.0, 0.0]])
    with pytest.raises(fm.CellUnsupported, match="identical"):
        fm.fit_readout(rbf_spec(), rows, ridge_alpha=1.0)
    fixed = fm.fit_readout(rbf_spec(bandwidth={"rule": "fixed", "value": 1.0}), rows,
                           ridge_alpha=1.0)
    assert fixed.output_dimension == 1 and fixed.duplicate_acquired_rows == 2
    assert fixed.degenerate_design is True


def test_exact_duplicate_acquired_rows_give_one_rbf_center_and_are_counted():
    readout = fm.fit_readout(rbf_spec(), [[0.0], [0.0], [2.0], [4.0]], ridge_alpha=1.0)
    assert readout.center_positions == (0, 2, 3) and readout.duplicate_acquired_rows == 1
    assert readout.output_dimension == 3
    assert readout.centers.tolist() == [[0.0], [2.0], [4.0]]
    assert readout.bandwidth["sigma"] == pytest.approx(2.0)      # median of {2, 4, 2}
    assert readout.bandwidth["positive_pairs"] == 3


def test_constancy_is_exact_so_a_tiny_but_real_variation_is_mapped_and_fitted_not_discarded():
    # [0, 1e-13] varies, so it is NOT constant however small its units: mean 5e-14, population std
    # 5e-14, standardised values -1 and +1. A tolerance-classified "constant" would have produced
    # a recorded zero coordinate next to nonzero mapped values and nonzero ridge weights.
    rows = [[0.0], [1e-13]]
    standardised = fm.fit_readout(linear_spec(scale="std", trace=True), rows, ridge_alpha=1.0)
    assert standardised.constant_columns == () and standardised.near_constant_columns == (0,)
    np.testing.assert_allclose(standardised.mean, [5e-14], rtol=1e-12)
    np.testing.assert_allclose(standardised.scale, [5e-14], rtol=1e-12)
    assert standardised.trace_factor == pytest.approx(1.0, rel=1e-9)
    np.testing.assert_allclose(standardised.transform(rows), [[-1.0], [1.0]], rtol=1e-9)
    assert standardised.capacity["design_rank"] == 1
    assert (standardised.capacity["exactly_constant_columns"],
            standardised.capacity["near_constant_columns"]) == (0, 1)
    state = standardised.state_document()
    assert state["constant_columns"] == [] and state["near_constant_columns"] == [0]
    # unscaled, the same direction is still real: trace normalisation is the only rescaling, by the
    # hand value 1 / sqrt(mean squared centred norm) = 1 / 5e-14, and no row is special-cased
    raw = fm.fit_readout(linear_spec(scale="none", trace=True), rows, ridge_alpha=1.0)
    assert raw.trace_factor == pytest.approx(2e13, rel=1e-9)
    np.testing.assert_allclose(raw.transform(rows), [[-1.0], [1.0]], rtol=1e-9)
    # and the ridge the baseline fits on it genuinely uses it: w = x'yc / (x'x + alpha) = 1 / 2.01
    model = fb.fit_reward(raw.transform(rows), [0.0, 1.0], explicit_space("x", 1), alpha=0.01)
    assert model.weights[0] == pytest.approx(1.0 / 2.01, rel=1e-6)


def test_only_bit_identical_acquired_values_make_a_column_constant():
    rows = [[1.0, 5.0, 2.0], [2.0, 5.0, 2.0 + 2.0 ** -50], [3.0, 5.0, 2.0]]
    readout = fm.fit_readout(linear_spec(scale="std"), rows, ridge_alpha=1.0)
    assert readout.constant_columns == (1,)
    assert readout.near_constant_columns == (2,)            # varies by 2^-50: listed, never dropped
    assert readout.scale[1] == 1.0 and readout.scale[2] > 0.0 and readout.scale[2] != 1.0
    assert np.all(readout.transform(rows)[:, 1] == 0.0)
    assert readout.capacity["design_rank"] == 2
    # a signed zero is still the same value
    signed = fm.fit_readout(linear_spec(), [[0.0, 1.0], [-0.0, 2.0]], ridge_alpha=1.0)
    assert signed.constant_columns == (0,)


@pytest.mark.parametrize("center,scale,trace", list(itertools.product((True, False),
                                                                      ("none", "std"),
                                                                      (True, False))))
def test_a_stable_zero_design_has_rank_zero_df_zero_and_factor_one(center, scale, trace):
    # 0.1 + 0.1 + 0.1 is 0.30000000000000004, so its mean is 0.10000000000000002, not 0.1; a plain
    # mean-centred residue (about 1e-17) used to be amplified by trace normalisation into a factor
    # of 7e16 and a phantom rank-1 direction. Identical rows must centre to EXACTLY zero.
    same = [[0.1], [0.1], [0.1]]
    linear = fm.fit_readout(linear_spec(center=center, scale=scale, trace=trace), same,
                            ridge_alpha=1.0)
    assert linear.constant_columns == (0,) and linear.degenerate_design is True
    assert linear.centered_variance == 0.0 and linear.trace_factor == 1.0
    assert linear.capacity["design_rank"] == 0 and linear.capacity["effective_ridge_df"] == 0.0
    mapped = linear.transform(same)
    assert np.all(mapped == mapped[0]), "identical rows must stay identical after the map"
    assert mapped[0, 0] == (0.0 if center else 0.1)
    fixed = rbf_spec(center=center, scale=scale, trace=trace,
                     bandwidth={"rule": "fixed", "value": 1.0})
    kernel = fm.fit_readout(fixed, same, ridge_alpha=1.0)
    assert kernel.degenerate_design is True and kernel.output_dimension == 1
    assert kernel.centered_variance == 0.0 and kernel.trace_factor == 1.0
    assert kernel.capacity["design_rank"] == 0 and kernel.capacity["effective_ridge_df"] == 0.0
    assert np.all(kernel.transform(same) == 1.0)             # exp(0): the unscaled constant design


def test_an_ordinary_nonconstant_design_keeps_its_hand_derived_rank_df_and_factor():
    # x = 0.1, 0.3, 0.5, uncentred and unscaled: centred -0.2, 0, +0.2, so the mean squared centred
    # norm is 0.08 / 3 and g = sqrt(37.5). With three rows the trace-normalised squared singular
    # values sum to 3, and the design has rank 1, so s^2 = 3 and df = 3 / (3 + alpha) = 0.75.
    readout = fm.fit_readout(linear_spec(center=False, scale="none", trace=True),
                             [[0.1], [0.3], [0.5]], ridge_alpha=1.0)
    assert readout.degenerate_design is False and readout.constant_columns == ()
    assert readout.centered_variance == pytest.approx(0.08 / 3.0, rel=1e-12)
    assert readout.trace_factor == pytest.approx(math.sqrt(37.5), rel=1e-12)
    assert readout.capacity["design_rank"] == 1
    assert readout.capacity["effective_ridge_df"] == pytest.approx(0.75, rel=1e-12)
    np.testing.assert_allclose(readout.transform([[0.1], [0.5]]),
                               [[0.1 * math.sqrt(37.5)], [0.5 * math.sqrt(37.5)]], rtol=1e-12)


def stored_rows_df(rows, alpha):
    """The effective ridge df of ``rows``, written independently of the module under test: anchored
    centring (relative to the first row), then the singular values of what remains."""
    rows = np.asarray(rows, dtype=np.float64)
    shifted = rows - rows[0]
    squares = np.linalg.svd(shifted - shifted.mean(axis=0), compute_uv=False) ** 2
    return float(np.sum(squares / (squares + alpha)))


OFFSET = (1.0, 1.0 + 2.0 ** -50, 1.0)


def test_capacity_describes_the_actual_transformed_rows_when_the_trace_factor_rounds():
    """A variation of 2^-50 on an offset of 1: g ~ 2.4e15 and g * (1 + 2^-50) rounds to g + 2, not
    g + 2.12. The stored rows are [A, A + 2, A] (centred squares 8/3, df 8/11), whereas the design
    before the multiplication would have claimed a unit variance (squares 3, df 3/4)."""
    rows = [[value] for value in OFFSET]
    readout = fm.fit_readout(linear_spec(center=False, scale="none", trace=True), rows,
                             ridge_alpha=1.0)
    mapped = readout.transform(rows)
    assert (mapped[1] - mapped[0]).tolist() == [2.0] and (mapped[2] - mapped[0]).tolist() == [0.0]
    assert readout.trace_factor == pytest.approx(3.0 * 2.0 ** 50 / math.sqrt(2.0), rel=1e-12)
    assert readout.centered_variance == pytest.approx(2.0 ** -100 * 2.0 / 9.0, rel=1e-9)
    assert readout.capacity["design_rank"] == 1
    assert readout.capacity["effective_ridge_df"] == pytest.approx(8.0 / 11.0, rel=1e-9)
    assert readout.capacity["effective_ridge_df"] == pytest.approx(stored_rows_df(mapped, 1.0),
                                                                   rel=1e-12)
    assert readout.realized_variance == pytest.approx(8.0 / 9.0, rel=1e-9)
    assert readout.realized_variance != pytest.approx(1.0, rel=1e-3), (
        "the realized variance is recorded as it is, not promised to be one")
    trace = readout.state_document()["trace_normalization"]
    assert trace["realized_centered_acquired_variance_after"] == readout.realized_variance
    assert trace["centered_acquired_variance_before"] == readout.centered_variance
    assert "not promised to be exactly 1" in trace["realized_variance_note"]


@pytest.mark.parametrize("kind", ["linear", "rbf"])
@pytest.mark.parametrize("center,scale,trace", list(itertools.product((True, False),
                                                                      ("none", "std"),
                                                                      (True, False))))
def test_reported_df_equals_the_df_of_the_rows_the_transform_produces(kind, center, scale, trace):
    """The invariant ``verify_freeze`` enforces, asserted at its source for every map option."""
    rows = np.random.default_rng(7).normal(size=(7, 3))
    if kind == "linear":
        spec = linear_spec(center=center, scale=scale, trace=trace)
    else:
        spec = rbf_spec(center=center, scale=scale, trace=trace)
    readout = fm.fit_readout(spec, rows, ridge_alpha=0.7)
    capacity = readout.capacity
    assert capacity["effective_ridge_df"] == pytest.approx(
        stored_rows_df(readout.transform(rows), 0.7), rel=1e-12)
    assert 0 <= capacity["design_rank"] <= min(7, capacity["output_dimension"])
    if trace:
        assert readout.realized_variance == pytest.approx(1.0, rel=1e-9), (
            "a well-conditioned design keeps its unit variance")


def offset_world():
    """The acquired rows (budget 3, seed 0) carry ``OFFSET``; every other candidate holds 1.0."""
    world = World()
    offset = np.ones((TOTAL, 1))
    acquired = fb.acquire_labels(LABELLED, 3, 0).tolist()
    for position, value in zip(acquired, OFFSET):
        offset[position, 0] = value
    world.features["offset"] = offset
    return world


def offset_plan():
    return plan_document(
        representations=["offset"], label_budgets=[3], seeds=[0],
        maps=[{"id": "offset_linear", "kind": "linear", "center": False, "scale": "none",
               "trace_normalize": True}],
        baseline={"tau": 1.0, "ridge_alpha": 1.0, "sir_proposal_draws": 64, "final_draws": 16,
                  "dpo_pressures": [0.5, 1.0]},
        reference={"representation": "offset", "map": "offset_linear"})


def test_a_genuine_ill_conditioned_acquisition_freezes_verifies_and_evaluates(tmp_path):
    """The run that used to refuse its own freeze: valid inputs, one acquired variation of 2^-50."""
    paths = offset_world().write(tmp_path / "in", plan=offset_plan())
    out = tmp_path / "out"
    summary = run(paths, out)
    assert summary["status_counts"] == {"cells": 1, "ok": 1, "unsupported": 0, "failed": 0}
    verified = fm.verify_freeze(out)                    # reloaded and re-verified from disk
    envelope = verified["envelopes"]["offset|offset_linear|b3|s0"]
    stored = np.array(envelope["artifact"]["training"]["features"], dtype=np.float64)
    assert stored.shape == (3, 1)
    assert (stored[:, 0] - stored[0, 0]).tolist() == [0.0, 2.0, 0.0]
    capacity = envelope["capacity"]
    assert capacity["design_rank"] == 1
    assert capacity["effective_ridge_df"] == pytest.approx(8.0 / 11.0, rel=1e-9)
    assert capacity["effective_ridge_df"] == pytest.approx(stored_rows_df(stored, 1.0), rel=1e-9)
    trace = envelope["artifact"]["features"]["space"]["schema"]["transform"][
        "trace_normalization"]
    assert trace["realized_centered_acquired_variance_after"] == pytest.approx(8.0 / 9.0, rel=1e-9)
    assert (out / fm.RESULTS_NAME).is_file()


def test_the_capacity_check_still_refuses_a_doctored_df_on_the_ill_conditioned_freeze(tmp_path):
    """The verification tolerance is untouched: the repair changed what is REPORTED, not what passes."""
    paths = offset_world().write(tmp_path / "in", plan=offset_plan())
    out = tmp_path / "out"
    run(paths, out)
    cell = fm.CellKey("offset", "offset_linear", 3, 0).filename()
    honest = fb.read_json(out / fm.CELLS_DIR / cell)["capacity"]
    rehash_envelope(out, cell, forge_capacity(
        effective_ridge_df=honest["effective_ridge_df"] + 0.05))
    with pytest.raises(ValueError, match="effective ridge df"):
        fm.verify_freeze(out)


def test_a_readout_refuses_a_dense_system_above_the_baseline_bound_before_mapping_anything(
        monkeypatch):
    monkeypatch.setattr(fb, "MAX_DENSE_SYSTEM_ELEMENTS", 9)
    fits = fm.fit_readout(linear_spec(), [[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]], ridge_alpha=1.0)
    assert fits.output_dimension == 3                        # 3 x 3 = 9 elements: at the bound
    with pytest.raises(ValueError, match="4 x 4 = 16 elements"):
        fm.fit_readout(linear_spec(), np.ones((2, 4)), ridge_alpha=1.0)
    rows = np.arange(8.0).reshape(4, 2)
    assert fm.fit_readout(rbf_spec(), rows[:3], ridge_alpha=1.0).output_dimension == 3
    with pytest.raises(ValueError, match="RBF readout.*4 x 4 = 16 elements"):
        fm.fit_readout(rbf_spec(), rows, ridge_alpha=1.0)    # four distinct centers: a 4 x 4 system
    # a duplicate row is not a new center: four rows, three centers, still within the bound
    assert fm.fit_readout(rbf_spec(), rows[[0, 0, 1, 2]], ridge_alpha=1.0).output_dimension == 3


def test_a_cell_above_the_dense_bound_is_recorded_failed_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "MAX_DENSE_SYSTEM_ELEMENTS", 9)       # alpha is 4 wide, RBF 6 wide
    paths = World().write(tmp_path / "in", plan=small_plan())
    summary = run(paths, tmp_path / "out")
    assert summary["status_counts"] == {"cells": 2, "ok": 0, "unsupported": 0, "failed": 2}
    for item in summary["cells"]:
        assert item["status"] == "failed" and item["reason"].startswith("ValueError:")
        assert "dense ridge Gram system" in item["reason"] and "elements" in item["reason"]
    assert fm.verify_freeze(tmp_path / "out")["index"]["counts"]["failed"] == 2


def test_a_readout_refuses_malformed_input():
    readout = fm.fit_readout(linear_spec(), [[1.0, 2.0], [3.0, 4.0]], ridge_alpha=1.0)
    with pytest.raises(ValueError, match="readout input"):
        readout.transform([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="finite"):
        readout.transform([[1.0, np.nan]])
    with pytest.raises(ValueError, match="finite"):
        fm.fit_readout(linear_spec(), [[1.0, np.inf]], ridge_alpha=1.0)
    with pytest.raises(ValueError, match="ridge_alpha"):
        fm.fit_readout(linear_spec(), [[1.0]], ridge_alpha=0.0)


# ---------------------------------------------------------------------------
# the matrix: counts, pairing, provenance, acquired-only transforms
# ---------------------------------------------------------------------------

def test_every_planned_cell_reports_exact_counts_and_paired_nested_acquisitions(matrix_run):
    assert matrix_run.summary["status_counts"] == {"cells": 16, "ok": 16, "unsupported": 0,
                                                   "failed": 0}
    verified = fm.verify_freeze(matrix_run.out)
    chosen = {}
    for key in verified["keys"]:
        artifact = verified["envelopes"][key.cell_id]["artifact"]
        accounting = artifact["provenance"]["label_accounting"]
        counts = artifact["provenance"]["split_counts"]
        assert accounting["offered_labels"] == LABELLED
        assert accounting["acquired_labels"] == key.label_budget
        assert accounting["unlabelled_train_candidates"] == UNLABELLED
        assert counts["train_candidates_offered"] == LABELLED + UNLABELLED
        assert counts["train_labelled_acquired"] == key.label_budget
        assert counts["evaluation_candidates"] == HELD_OUT
        assert artifact["support"]["ids"] == sorted(matrix_run.world.eval_ids)
        expected = [f"c{i:02d}" for i in acquired_expected(LABELLED, key.label_budget, key.seed)]
        assert artifact["training"]["ids"] == expected
        chosen[(key.representation, key.map_id, key.label_budget, key.seed)] = set(expected)
    for rep, map_id, seed in itertools.product(("alpha", "beta"), ("linear", "rbf"), (0, 1)):
        assert chosen[(rep, map_id, 6, seed)] < chosen[(rep, map_id, 12, seed)]


def test_the_derived_space_records_the_transform_acquired_ids_and_raw_digests(matrix_run):
    verified = fm.verify_freeze(matrix_run.out)
    alpha = fm.load_feature_cache(matrix_run.paths.caches["alpha"])
    for map_id in ("linear", "rbf"):
        artifact = verified["envelopes"][f"alpha|{map_id}|b6|s0"]["artifact"]
        schema = artifact["features"]["space"]["schema"]
        ids = artifact["training"]["ids"]
        rows = [matrix_run.world.ids.index(ident) for ident in ids]
        assert schema["acquired"]["ids"] == ids and schema["acquired"]["budget"] == 6
        assert schema["acquired"]["offered_labelled_count"] == LABELLED
        assert schema["provenance"]["canonical_cache_sha256"] == alpha.cache_sha256
        assert schema["provenance"]["unlabelled_covariates_used"] is False
        assert schema["provenance"]["unacquired_labelled_covariates_used"] is False
        assert schema["transform"]["mean"] == pytest.approx(
            matrix_run.world.features["alpha"][rows].mean(axis=0).tolist())
        assert schema["transform"]["scale"] == pytest.approx(
            matrix_run.world.features["alpha"][rows].std(axis=0).tolist())
        assert schema["applied_to"] == {"offered_rows": LABELLED, "held_out_rows": HELD_OUT,
                                        "note": schema["applied_to"]["note"]}
        assert schema["preprocessing_cost"]["transform"]["rows"] == LABELLED + HELD_OUT
        capacity = schema["capacity"]
        assert 0.0 < capacity["effective_ridge_df"] <= capacity["output_dimension"]
        expected_dimension = 4 if map_id == "linear" else 6
        assert capacity["output_dimension"] == expected_dimension
        assert len(artifact["fitted"]["reward"]["weights"]) == expected_dimension
    rbf = verified["envelopes"]["alpha|rbf|b6|s0"]["artifact"]["features"]["space"]["schema"]
    assert rbf["center_ids"] == rbf["acquired"]["ids"]


def _cell_artifacts(out):
    return {cell_id: envelope["artifact"]
            for cell_id, envelope in fm.verify_freeze(out)["envelopes"].items()
            if envelope["status"] == "ok"}


def test_unacquired_and_heldout_features_cannot_change_the_fit_but_change_the_provenance(tmp_path):
    world = World()
    paths = world.write(tmp_path / "a", plan=small_plan())
    run(paths, tmp_path / "out_a")
    base = _cell_artifacts(tmp_path / "out_a")
    acquired = set(next(iter(base.values()))["training"]["ids"])
    unacquired = next(ident for ident in world.labelled_ids if ident not in acquired)
    touched = [world.ids.index(unacquired), LABELLED, LABELLED + UNLABELLED]   # + unlabelled + held-out
    features = world.features["alpha"].copy()
    features[touched] += 5.0
    changed = world.variant(features={"alpha": features})
    run(changed.write(tmp_path / "b", plan=small_plan()), tmp_path / "out_b")
    other = _cell_artifacts(tmp_path / "out_b")
    for cell_id, left in base.items():
        right = other[cell_id]
        left_schema, right_schema = (a["features"]["space"]["schema"] for a in (left, right))
        assert left_schema["transform"] == right_schema["transform"], cell_id
        assert left["fitted"]["reward"] == right["fitted"]["reward"], cell_id
        assert left["training"]["features"] == right["training"]["features"], cell_id
        assert (left_schema["provenance"]["canonical_cache_sha256"]
                != right_schema["provenance"]["canonical_cache_sha256"]), cell_id
        assert left["hashes"]["artifact_sha256"] != right["hashes"]["artifact_sha256"], cell_id
        left_train = left["provenance"]["input_binding"]["components"]["train"]["features_sha256"]
        right_train = right["provenance"]["input_binding"]["components"]["train"]["features_sha256"]
        assert left_train != right_train, "an unacquired offered row is bound into the provenance"
        assert left["predictions"]["utility"][1:] == right["predictions"]["utility"][1:]
        assert left["predictions"]["utility"][0] != right["predictions"]["utility"][0]


def test_changing_an_acquired_feature_row_changes_the_transform_and_the_fit(tmp_path):
    world = World()
    paths = world.write(tmp_path / "a", plan=small_plan())
    run(paths, tmp_path / "out_a")
    base = _cell_artifacts(tmp_path / "out_a")
    acquired = next(iter(base.values()))["training"]["ids"]
    features = world.features["alpha"].copy()
    features[world.ids.index(acquired[0])] += 5.0
    run(world.variant(features={"alpha": features}).write(tmp_path / "b", plan=small_plan()),
        tmp_path / "out_b")
    other = _cell_artifacts(tmp_path / "out_b")
    linear_a = base["alpha|linear|b6|s0"]["features"]["space"]["schema"]["transform"]
    linear_b = other["alpha|linear|b6|s0"]["features"]["space"]["schema"]["transform"]
    assert linear_a["mean"] != linear_b["mean"] and linear_a["scale"] != linear_b["scale"]
    rbf_a = base["alpha|rbf|b6|s0"]["features"]["space"]["schema"]["transform"]
    rbf_b = other["alpha|rbf|b6|s0"]["features"]["space"]["schema"]["transform"]
    assert rbf_a["centers"]["centers_sha256"] != rbf_b["centers"]["centers_sha256"]
    for cell_id in base:
        assert base[cell_id]["fitted"]["reward"]["weights"] \
            != other[cell_id]["fitted"]["reward"]["weights"]


def test_permuting_every_row_order_changes_no_frozen_byte(matrix_run, tmp_path):
    permuted = matrix_run.world.write(tmp_path / "permuted", permute=True)
    assert permuted.candidates.read_bytes() != matrix_run.paths.candidates.read_bytes()
    assert fm.load_feature_cache(permuted.caches["alpha"]).raw["file_sha256"] \
        != fm.load_feature_cache(matrix_run.paths.caches["alpha"]).raw["file_sha256"]
    run(permuted, tmp_path / "out")
    names = pre_evaluation_names(matrix_run.out)[:-1]            # inputs.json holds raw digests
    assert names == pre_evaluation_names(tmp_path / "out")[:-1]
    assert tree_bytes(matrix_run.out, names + [fm.RESULTS_NAME]) \
        == tree_bytes(tmp_path / "out", names + [fm.RESULTS_NAME])
    assert (matrix_run.out / fm.INPUTS_NAME).read_bytes() != (tmp_path / "out" / fm.INPUTS_NAME).read_bytes()


def test_changing_the_evaluation_outcomes_changes_no_pre_evaluation_file(small_run, tmp_path):
    flipped = write_csv(tmp_path / "eval_flipped.csv", "candidate_id,utility", [
        f"{ident},{-value!r}" for ident, value in small_run.world.eval_truth().items()])
    run(small_run.paths, tmp_path / "out", eval_outcomes=flipped)
    names = pre_evaluation_names(small_run.out)
    assert names == pre_evaluation_names(tmp_path / "out")
    assert tree_bytes(small_run.out, names) == tree_bytes(tmp_path / "out", names)
    assert (small_run.out / fm.RESULTS_NAME).read_bytes() \
        != (tmp_path / "out" / fm.RESULTS_NAME).read_bytes()
    left = fb.read_json(small_run.out / fm.RESULTS_NAME)
    right = fb.read_json(tmp_path / "out" / fm.RESULTS_NAME)
    assert left["index_sha256"] == right["index_sha256"]
    assert left["cells"] != right["cells"]


def test_the_evaluation_file_is_opened_only_after_every_cell_and_the_index_are_verified(
        tmp_path, monkeypatch):
    world = World().variant()
    world.features["flat"] = np.ones((TOTAL, 2))
    plan = plan_document(representations=["alpha", "flat"], label_budgets=[6, 100], seeds=[0])
    paths = world.write(tmp_path / "in", plan=plan)
    out = tmp_path / "out"
    events, snapshots = [], []
    real_open, real_read_text, real_write_text = Path.open, Path.read_text, Path.write_text

    def recording(kind, real):
        def wrapper(self, *args, **kwargs):
            events.append((kind, Path(self)))
            return real(self, *args, **kwargs)
        return wrapper
    monkeypatch.setattr(Path, "open", recording("open", real_open))
    monkeypatch.setattr(Path, "read_text", recording("read_text", real_read_text))
    monkeypatch.setattr(Path, "write_text", recording("write_text", real_write_text))
    original = fb.read_outcomes

    def spy(path, *, label):
        if Path(path).name == paths.eval.name:
            snapshots.append({
                "cells": sorted(item.name for item in (out / fm.CELLS_DIR).iterdir()),
                "counts": fm.verify_freeze(out)["index"]["counts"]})
        return original(path, label=label)
    monkeypatch.setattr(fb, "read_outcomes", spy)
    summary = run(paths, out)
    monkeypatch.undo()
    assert len(snapshots) == 1
    assert len(snapshots[0]["cells"]) == 8, "every cell, unsupported ones included, was on disk"
    assert snapshots[0]["counts"] == {"cells": 8, "ok": 3, "unsupported": 5, "failed": 0}
    assert summary["status_counts"] == snapshots[0]["counts"]
    eval_events = [i for i, (_, path) in enumerate(events) if path.name == paths.eval.name]
    assert eval_events, "the evaluation file was never read"
    first_eval = eval_events[0]
    index_written = max(i for i, (_, path) in enumerate(events)
                        if path.name == fm.INDEX_NAME + ".tmp")
    index_reloaded = min(i for i, (kind, path) in enumerate(events)
                         if path.name == fm.INDEX_NAME and i > index_written)
    assert index_written < index_reloaded < first_eval
    for item in (out / fm.CELLS_DIR).iterdir():
        reloaded = [i for i, (_, path) in enumerate(events) if path.name == item.name]
        assert reloaded and min(reloaded) < first_eval, f"{item.name} was not reloaded first"


def test_unsupported_cells_keep_an_explicit_reason_and_stay_in_the_index_and_the_report(tmp_path):
    world = World()
    world.features["flat"] = np.ones((TOTAL, 2))
    plan = plan_document(representations=["alpha", "flat"], label_budgets=[6, 100], seeds=[0])
    paths = world.write(tmp_path / "in", plan=plan)
    summary = run(paths, tmp_path / "out")
    assert summary["status_counts"] == {"cells": 8, "ok": 3, "unsupported": 5, "failed": 0}
    reasons = {item["cell_id"]: item for item in summary["cells"]}
    for cell_id in ("alpha|linear|b100|s0", "alpha|rbf|b100|s0", "flat|linear|b100|s0",
                    "flat|rbf|b100|s0"):
        assert reasons[cell_id]["status"] == "unsupported" and "exceeds" in reasons[cell_id]["reason"]
    assert reasons["flat|rbf|b6|s0"]["status"] == "unsupported"
    assert "identical" in reasons["flat|rbf|b6|s0"]["reason"]
    assert reasons["flat|linear|b6|s0"]["status"] == "ok"      # degenerate but numerically defined
    index = fb.read_json(tmp_path / "out" / fm.INDEX_NAME)
    assert [entry["status"] for entry in index["cells"]].count("unsupported") == 5
    results = fb.read_json(tmp_path / "out" / fm.RESULTS_NAME)
    by_id = {item["cell_id"]: item for item in results["cells"]}
    assert len(by_id) == 8 and by_id["flat|rbf|b6|s0"]["evaluator_report"] is None
    assert by_id["flat|rbf|b6|s0"]["comparison_to_reference"]["status"] == "cell_unavailable"
    assert by_id["alpha|rbf|b6|s0"]["comparison_to_reference"]["status"] == "ok"
    assert by_id["flat|linear|b6|s0"]["comparison_to_reference"]["status"] == "ok"
    assert by_id["alpha|linear|b100|s0"]["comparison_to_reference"]["status"] == "is_reference"
    flat = fm.verify_freeze(tmp_path / "out")["envelopes"]["flat|linear|b6|s0"]["artifact"]
    assert flat["features"]["space"]["schema"]["transform"]["trace_normalization"][
        "degenerate_acquired_design"] is True
    assert flat["fitted"]["reward"]["weights"] == [0.0, 0.0]


def test_a_failed_cell_is_recorded_visibly_and_the_cli_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fb, "MAX_PAIR_MATRIX_ELEMENTS", 10)
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    summary = run(paths, tmp_path / "out")
    assert summary["status_counts"] == {"cells": 2, "ok": 0, "unsupported": 0, "failed": 2}
    for item in summary["cells"]:
        assert item["status"] == "failed" and item["reason"].startswith("ValueError:")
        assert "pair-difference matrix" in item["reason"]
    verified = fm.verify_freeze(tmp_path / "out")
    assert verified["index"]["counts"]["failed"] == 2
    results = fb.read_json(tmp_path / "out" / fm.RESULTS_NAME)
    assert [item["status"] for item in results["cells"]] == ["failed", "failed"]
    cli = load_cli()
    argv = ["run", "--candidates", str(paths.candidates), "--train-outcomes", str(paths.train),
            "--eval-outcomes", str(paths.eval), "--manifest", str(paths.manifest),
            "--plan", str(paths.plan), "--cache", f"alpha={paths.caches['alpha']}",
            "--output-dir", str(tmp_path / "cli_out")]
    assert cli.main(argv) == 1
    assert json.loads(capsys.readouterr().out)["status_counts"]["failed"] == 2


def test_a_missing_or_incomplete_evaluation_file_leaves_a_verified_freeze_that_never_saw_it(
        tmp_path, small_run):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    with pytest.raises(ValueError, match="does not exist"):
        run(paths, tmp_path / "out_missing", eval_outcomes=tmp_path / "nope.csv")
    verified = fm.verify_freeze(tmp_path / "out_missing")
    assert verified["index"]["counts"]["ok"] == 2
    assert not (tmp_path / "out_missing" / fm.RESULTS_NAME).exists()
    rows = [f"{ident},{value!r}" for ident, value in list(world.eval_truth().items())[:-1]]
    short = write_csv(tmp_path / "short.csv", "candidate_id,utility", rows)
    with pytest.raises(ValueError, match="missing"):
        run(paths, tmp_path / "out_short", eval_outcomes=short)
    assert fm.verify_freeze(tmp_path / "out_short")["index"]["counts"]["ok"] == 2
    assert not (tmp_path / "out_short" / fm.RESULTS_NAME).exists()
    # the complete verified freeze is what the safe public evaluator needs: score it afterwards
    # with the real outcomes and get exactly the report a one-shot run produces
    for left_over in ("out_missing", "out_short"):
        again = fm.evaluate_matrix(tmp_path / left_over, paths.eval)
        assert again["status_counts"]["ok"] == 2
        assert (tmp_path / left_over / fm.RESULTS_NAME).read_bytes() \
            == (small_run.out / fm.RESULTS_NAME).read_bytes()
        assert (tmp_path / left_over / fm.INDEX_NAME).read_bytes() \
            == (small_run.out / fm.INDEX_NAME).read_bytes()


# ---------------------------------------------------------------------------
# evaluation: the baseline evaluator, DPO in the mapped basis, constant outcomes
# ---------------------------------------------------------------------------

def test_every_policy_value_regret_and_rank_comes_from_the_baseline_evaluator(matrix_run):
    results = fb.read_json(matrix_run.out / fm.RESULTS_NAME)
    verified = fm.verify_freeze(matrix_run.out)
    by_id = {item["cell_id"]: item for item in results["cells"]}
    assert not any(key in results for key in ("winner", "best_cell", "ranking", "headline"))
    assert "No winning cell" in results["no_winner_selected"]
    for cell_id, envelope in verified["envelopes"].items():
        expected = fb.evaluate_frozen(envelope["artifact"], matrix_run.world.eval_truth())
        assert by_id[cell_id]["evaluator_report"] == expected, cell_id
        assert expected["identities"]["all_hold"] is True
        assert by_id[cell_id]["readout"]["rmse"] == pytest.approx(
            expected["prediction_ranking"]["rmse"])
    assert by_id["alpha|linear|b12|s0"]["evaluator_report"]["oracle"]["oracle_regret"][
        "value"] == pytest.approx(0.0, abs=1e-9)


def test_comparisons_are_paired_with_the_declared_reference_per_budget_and_seed(matrix_run):
    results = fb.read_json(matrix_run.out / fm.RESULTS_NAME)
    by_id = {item["cell_id"]: item for item in results["cells"]}
    statuses = {}
    for item in results["cells"]:
        statuses.setdefault(item["comparison_to_reference"]["status"], []).append(item["cell_id"])
    assert sorted(statuses["is_reference"]) == sorted(
        f"alpha|linear|b{b}|s{s}" for b in (6, 12) for s in (0, 1))
    assert len(statuses["ok"]) == 12
    for cell_id in statuses["ok"]:
        item = by_id[cell_id]
        comparison = item["comparison_to_reference"]
        reference = f"alpha|linear|b{item['key']['label_budget']}|s{item['key']['seed']}"
        assert comparison["reference_cell"] == reference and comparison["paired_acquisition"] is True
        assert comparison["n_groups"] == 4 and comparison["interval"][0] <= comparison["interval"][1]
        assert comparison["mse_cell"] == pytest.approx(item["readout"]["mse"])
        assert comparison["mse_difference"] == pytest.approx(
            item["readout"]["mse"] - by_id[reference]["readout"]["mse"])
        assert item["readout"]["acquired_ids_sha256"] \
            == by_id[reference]["readout"]["acquired_ids_sha256"]
        assert "seed" in comparison["seed_note"] and "pooled" in comparison["seed_note"]


def test_a_panel_with_too_few_support_groups_gets_a_null_interval_not_a_made_up_one(tmp_path):
    world = World()
    plan = small_plan()
    plan["bootstrap"]["min_groups"] = 5            # the held-out panel has only 4 groups
    paths = world.write(tmp_path / "in", plan=plan)
    run(paths, tmp_path / "out")
    results = fb.read_json(tmp_path / "out" / fm.RESULTS_NAME)
    comparison = {item["cell_id"]: item["comparison_to_reference"] for item in results["cells"]}
    assert comparison["alpha|rbf|b6|s0"]["status"] == "insufficient_groups"
    assert comparison["alpha|rbf|b6|s0"]["interval"] is None
    assert math.isfinite(comparison["alpha|rbf|b6|s0"]["mse_difference"])


def test_strict_pairs_give_a_real_dpo_update_and_the_energy_is_linear_in_the_mapped_basis(
        small_run):
    artifacts = _cell_artifacts(small_run.out)
    world = small_run.world
    for map_id, spec in (("linear", linear_spec(scale="std", trace=True)),
                         ("rbf", rbf_spec(center=True, scale="std", trace=True))):
        artifact = artifacts[f"alpha|{map_id}|b6|s0"]
        dpo = artifact["fitted"]["dpo"]
        assert dpo["status"] == "fit"
        theta = np.array(dpo["theta"])
        assert float(np.linalg.norm(theta)) > 1e-3
        assert dpo["diagnostics"]["updates"] >= 1 and dpo["diagnostics"]["loss_decreased"] is True
        assert dpo["diagnostics"]["final_loss"] < dpo["diagnostics"]["initial_loss"]
        assert theta.shape[0] == artifact["features"]["space"]["schema"]["capacity"][
            "output_dimension"]
        rows = [world.ids.index(ident) for ident in artifact["training"]["ids"]]
        readout = fm.fit_readout(spec, world.features["alpha"][rows], ridge_alpha=0.5)
        support = [world.ids.index(ident) for ident in artifact["support"]["ids"]]
        mapped = readout.transform(world.features["alpha"][support])
        np.testing.assert_allclose(artifact["predictions"]["dpo_energy"], mapped @ theta,
                                   rtol=1e-9, atol=1e-12)


def test_constant_training_outcomes_leave_no_dpo_pair_but_keep_a_numeric_evaluation(tmp_path):
    world = World()
    flat = world.variant(utility={**world.utility,
                                  **{ident: 1.0 for ident in world.labelled_ids}})
    paths = flat.write(tmp_path / "in", plan=small_plan())
    run(paths, tmp_path / "out", eval_outcomes=paths.eval)
    verified = fm.verify_freeze(tmp_path / "out")
    results = fb.read_json(tmp_path / "out" / fm.RESULTS_NAME)
    for item in results["cells"]:
        artifact = verified["envelopes"][item["cell_id"]]["artifact"]
        assert artifact["fitted"]["dpo"]["status"] == "unsupported"
        assert all(weight == 0.0 for weight in artifact["fitted"]["reward"]["weights"])
        report = item["evaluator_report"]
        assert report["dpo"]["status"] == "unsupported"
        assert math.isfinite(report["prediction_ranking"]["rmse"])
        assert report["prediction_ranking"]["all_candidates"]["spearman"]["value"] is None
        assert report["policies"]["ridge_tilt"]["expected_utility"] is not None
        assert item["readout"]["mse"] > 0.0


@pytest.mark.parametrize("supervision,expected", [
    (None, "eligible"),
    (fb.SupervisionHistory(status="unknown"), "unavailable"),
    (fb.SupervisionHistory(status="documented", documented=(
        {"assay": "A", "target": "T", "label_count": 2146, "label_ids": None},)), "ineligible"),
])
def test_feature_history_decides_the_clean_low_n_verdict_and_unknown_stays_unavailable(
        tmp_path, supervision, expected):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan(),
                        supervision={"alpha": supervision} if supervision else None)
    run(paths, tmp_path / "out")
    results = fb.read_json(tmp_path / "out" / fm.RESULTS_NAME)
    for item in results["cells"]:
        assert item["evaluator_report"]["clean_low_n"]["status"] == expected
    index = fb.read_json(tmp_path / "out" / fm.INDEX_NAME)
    assert index["representations"]["alpha"]["feature_supervision"]["status"] == (
        supervision.status if supervision else "label_free")


def test_the_roster_refuses_a_training_outcome_on_the_evaluation_split(tmp_path):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    rows = paths.train.read_text(encoding="utf-8").splitlines()
    rows.append(f"{world.eval_ids[0]},0.5")
    paths.train.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation-split"):
        run(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# the paired group bootstrap
# ---------------------------------------------------------------------------

def test_a_constant_difference_has_a_degenerate_interval_at_that_difference():
    groups = ["a", "a", "b", "c", "c", "c", "d"]
    reference = np.array([0.2, 0.4, 0.1, 0.3, 0.3, 0.9, 0.5])
    result = fm.paired_group_bootstrap(reference + 0.5, reference, groups, replicates=50, seed=1,
                                       confidence=0.9, min_groups=3)
    assert result["status"] == "ok" and result["n_groups"] == 4 and result["n_candidates"] == 7
    assert result["mse_difference"] == pytest.approx(0.5)
    assert result["interval"][0] == pytest.approx(0.5, abs=1e-12)
    assert result["interval"][1] == pytest.approx(0.5, abs=1e-12)
    assert result["mse_cell"] - result["mse_reference"] == pytest.approx(0.5)


def test_the_interval_lies_among_the_group_means_and_swapping_the_cells_negates_it():
    rng = np.random.default_rng(7)
    groups = [f"g{i // 3}" for i in range(18)]
    cell, reference = rng.random(18), rng.random(18)
    result = fm.paired_group_bootstrap(cell, reference, groups, replicates=200, seed=3,
                                       confidence=0.8, min_groups=3, stream=(7, 0, 6))
    means = [float((cell - reference)[i * 3:(i + 1) * 3].mean()) for i in range(6)]
    low, high = result["interval"]
    assert min(means) - 1e-12 <= low <= high <= max(means) + 1e-12
    swapped = fm.paired_group_bootstrap(reference, cell, groups, replicates=200, seed=3,
                                        confidence=0.8, min_groups=3, stream=(7, 0, 6))
    assert swapped["mse_difference"] == pytest.approx(-result["mse_difference"])
    np.testing.assert_allclose(swapped["interval"], [-high, -low], atol=1e-12)
    again = fm.paired_group_bootstrap(cell, reference, groups, replicates=200, seed=3,
                                      confidence=0.8, min_groups=3, stream=(7, 0, 6))
    assert again == result


def test_too_few_groups_make_the_interval_null_with_a_reason():
    result = fm.paired_group_bootstrap([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], ["a", "a", "b"],
                                       replicates=50, seed=0, confidence=0.9, min_groups=3)
    assert result["status"] == "insufficient_groups" and result["interval"] is None
    assert "min_groups=3" in result["reason"]
    assert result["mse_difference"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="align"):
        fm.paired_group_bootstrap([1.0, 2.0], [1.0], ["a", "b"], replicates=50, seed=0,
                                  confidence=0.9, min_groups=2)


# ---------------------------------------------------------------------------
# output safety, strict JSON and tamper refusal
# ---------------------------------------------------------------------------

ALIAS_INPUTS = {
    "candidates": lambda p: p.candidates, "train_outcomes": lambda p: p.train,
    "eval_outcomes": lambda p: p.eval, "manifest": lambda p: p.manifest, "plan": lambda p: p.plan,
    "cache_header": lambda p: p.caches["alpha"], "cache_npz": lambda p: p.npz["alpha"]}


def alias_replacement(which, target):
    if which == "cache_header":
        return {"caches": {"alpha": target}}
    if which == "cache_npz":
        return {"cache_npz": {"alpha": target}}
    return {which: target}


@pytest.mark.parametrize("which", sorted(ALIAS_INPUTS))
@pytest.mark.parametrize("alias_name", ["index", "index_tmp", "results", "inputs", "cell",
                                        "cell_tmp"])
def test_an_input_aliasing_any_output_or_staging_file_is_refused_even_with_overwrite(
        tmp_path, which, alias_name):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    cell = fm.enumerate_cells(fm.load_plan(paths.plan))[0].filename()
    names = {"index": fm.INDEX_NAME, "index_tmp": fm.INDEX_NAME + ".tmp",
             "results": fm.RESULTS_NAME, "inputs": fm.INPUTS_NAME,
             "cell": f"{fm.CELLS_DIR}/{cell}", "cell_tmp": f"{fm.CELLS_DIR}/{cell}.tmp"}
    out = tmp_path / "out"
    victim = out / names[alias_name]
    victim.parent.mkdir(parents=True)
    victim.write_bytes(ALIAS_INPUTS[which](paths).read_bytes())
    original = victim.read_bytes()
    listing = sorted(str(p.relative_to(out)) for p in out.rglob("*"))
    with pytest.raises(ValueError, match="resolves to the same file"):
        run(paths, out, overwrite=True, **alias_replacement(which, victim))
    assert victim.read_bytes() == original, "an input was destroyed"
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*")) == listing


def test_a_dotdot_spelling_and_a_hard_link_of_an_output_are_still_aliases(tmp_path):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    out = tmp_path / "out"
    out.mkdir()
    victim = out / fm.INDEX_NAME
    victim.write_bytes(paths.eval.read_bytes())
    original = victim.read_bytes()
    with pytest.raises(ValueError, match="resolves to the same file"):
        run(paths, out, overwrite=True, eval_outcomes=out / ".." / "out" / fm.INDEX_NAME)
    link = tmp_path / "in" / "linked_eval.csv"
    try:
        os.link(victim, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are unavailable on this filesystem")
    with pytest.raises(ValueError, match="resolves to the same file"):
        run(paths, out, overwrite=True, eval_outcomes=link)
    assert victim.read_bytes() == original


def test_existing_outputs_need_overwrite_stray_cells_are_never_cleaned_and_inputs_are_untouched(
        tmp_path):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    inputs = [paths.candidates, paths.train, paths.eval, paths.manifest, paths.plan,
              paths.caches["alpha"], paths.npz["alpha"]]
    before = {path: path.read_bytes() for path in inputs}
    out = tmp_path / "out"
    first = run(paths, out)
    with pytest.raises(ValueError, match="already exists"):
        run(paths, out)
    again = run(paths, out, overwrite=True)
    assert again["index_sha256"] == first["index_sha256"]
    stray = out / fm.CELLS_DIR / "old_cell.json"
    stray.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not produce"):
        run(paths, out, overwrite=True)
    assert stray.read_text(encoding="utf-8") == "{}"
    with pytest.raises(ValueError, match="does not list"):
        fm.verify_freeze(out)
    assert {path: path.read_bytes() for path in inputs} == before
    with pytest.raises(ValueError, match="not a directory"):
        run(paths, paths.candidates, overwrite=True)


def test_nested_duplicate_json_keys_are_refused_in_every_input_document(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text('{"schema": "x", "baseline": {"tau": 1, "tau": 2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        fm.load_plan(plan)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"utility": {"units": "a", "units": "b"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        fm.load_matrix_manifest(manifest)
    header, _, _ = fresh_cache(tmp_path / "cache")
    text = header.read_text(encoding="utf-8").rstrip()
    assert text.endswith("}")
    header.write_text(text[:-1] + ', "storage": {"file": "a.npz", "file": "b.npz"}}',
                      encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        fm.load_feature_cache(header)
    nan = tmp_path / "nan.json"
    nan.write_text('{"tau": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        fm.load_plan(nan)


def test_cache_ids_and_paths_outside_the_plan_are_refused_before_any_write(tmp_path):
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="key must be"):
        run(paths, out, caches={"../evil": paths.caches["alpha"]})
    with pytest.raises(ValueError, match="without a cache"):
        run(paths, out, cache_npz={"alpha": paths.npz["alpha"], "ghost": paths.npz["alpha"]})
    with pytest.raises(ValueError, match="does not exist"):
        run(paths, out, cache_npz={"alpha": tmp_path / "missing.npz"})
    assert not out.exists()


def rehash_chain(out, cell_name, mutate_artifact):
    """Mutate an artifact and re-derive EVERY hash above it: a consistent forgery."""
    path = Path(out) / fm.CELLS_DIR / cell_name
    envelope = fb.read_json(path)
    mutate_artifact(envelope["artifact"])
    envelope["artifact"]["hashes"] = fb._section_hashes(envelope["artifact"])
    fb.write_json(path, envelope)
    index = fb.read_json(Path(out) / fm.INDEX_NAME)
    entry = next(item for item in index["cells"] if item["file"].endswith(cell_name))
    entry["cell_sha256"] = fb.digest(envelope)
    entry["artifact_sha256"] = envelope["artifact"]["hashes"]["artifact_sha256"]
    del index["index_sha256"]
    index["index_sha256"] = fb.digest(index)
    fb.write_json(Path(out) / fm.INDEX_NAME, index)


def test_tampering_with_any_frozen_part_is_refused(small_run, tmp_path):
    names = [item.name for item in sorted((small_run.out / fm.CELLS_DIR).iterdir())]
    assert fm.verify_freeze(small_run.out)["index"]["counts"]["ok"] == 2

    def copy_out(name):
        target = tmp_path / name
        shutil.copytree(small_run.out, target)
        return target
    edited = copy_out("cell_edit")
    path = edited / fm.CELLS_DIR / names[0]
    path.write_text(path.read_text(encoding="utf-8").replace('"status": "ok"', '"status": "failed"',
                                                              1), encoding="utf-8")
    with pytest.raises(ValueError, match="modified after the freeze"):
        fm.verify_freeze(edited)
    swapped = copy_out("cell_swap")
    (swapped / fm.CELLS_DIR / names[0]).write_bytes((swapped / fm.CELLS_DIR / names[1]).read_bytes())
    with pytest.raises(ValueError, match="modified after the freeze"):
        fm.verify_freeze(swapped)
    index_edit = copy_out("index_edit")
    document = fb.read_json(index_edit / fm.INDEX_NAME)
    document["claims"]["scope"] = "something else"
    fb.write_json(index_edit / fm.INDEX_NAME, document)
    with pytest.raises(ValueError, match="index_sha256"):
        fm.verify_freeze(index_edit)
    plan_edit = copy_out("plan_edit")
    document = fb.read_json(plan_edit / fm.INDEX_NAME)
    document["plan"]["seeds"] = [0, 5]
    fb.write_json(plan_edit / fm.INDEX_NAME, document)
    with pytest.raises(ValueError, match="index_sha256"):
        fm.verify_freeze(plan_edit)
    missing = copy_out("cell_missing")
    (missing / fm.CELLS_DIR / names[0]).unlink()
    with pytest.raises(ValueError, match="is missing"):
        fm.verify_freeze(missing)
    forged = copy_out("forged_predictions")
    rehash_chain(forged, names[0], lambda art: art["predictions"]["utility"].__setitem__(
        0, art["predictions"]["utility"][0] + 0.5))
    with pytest.raises(ValueError, match="exact tilt"):
        fm.verify_freeze(forged)
    rebound = copy_out("forged_support")
    rehash_chain(rebound, names[0], lambda art: art["support"]["groups"].__setitem__(0, "evil"))
    with pytest.raises(ValueError, match="common support"):
        fm.verify_freeze(rebound)
    assert fm.verify_freeze(small_run.out)["index"]["counts"]["ok"] == 2


# ---------------------------------------------------------------------------
# a cell's artifact must be ITS OWN: representation, map, cache and capacity bound to the key
# ---------------------------------------------------------------------------

def cell_file(rep, map_id, budget=6, seed=0):
    return fm.CellKey(rep, map_id, budget, seed).filename()


def copy_freeze(source, target, *, drop_report=False):
    shutil.copytree(source, target)
    if drop_report:
        (Path(target) / fm.RESULTS_NAME).unlink()
    return Path(target)


def rehash_envelope(out, cell_name, mutate_envelope):
    """Mutate a whole cell envelope (the artifact may be replaced outright) and re-derive EVERY hash
    above the change: the artifact's section hashes, the cell digest, the index entry and the index.
    The result is internally hash-consistent, so only a cross-check can refuse it."""
    path = Path(out) / fm.CELLS_DIR / cell_name
    envelope = fb.read_json(path)
    mutate_envelope(envelope)
    if envelope["artifact"] is not None:
        envelope["artifact"]["hashes"] = fb._section_hashes(envelope["artifact"])
    fb.write_json(path, envelope)
    index = fb.read_json(Path(out) / fm.INDEX_NAME)
    entry = next(item for item in index["cells"] if item["file"].endswith(cell_name))
    entry["cell_sha256"] = fb.digest(envelope)
    if envelope["artifact"] is not None:
        entry["artifact_sha256"] = envelope["artifact"]["hashes"]["artifact_sha256"]
    index.pop("index_sha256")
    index["index_sha256"] = fb.digest(index)
    fb.write_json(Path(out) / fm.INDEX_NAME, index)


def rehash_index(out, mutate_index):
    path = Path(out) / fm.INDEX_NAME
    index = fb.read_json(path)
    mutate_index(index)
    index.pop("index_sha256")
    index["index_sha256"] = fb.digest(index)
    fb.write_json(path, index)


def forge_capacity(**changes):
    """Doctor the readout capacity inside the artifact AND in the envelope, then re-derive the
    artifact's own input binding, so ``verify_artifact`` is satisfied and only the cell-level
    cross-checks can notice."""
    def mutate(envelope):
        artifact = envelope["artifact"]
        for target in (artifact["features"]["space"]["schema"]["capacity"], envelope["capacity"]):
            target.update(changes)
        binding = artifact["provenance"]["input_binding"]
        binding["components"]["space_sha256"] = fb.digest(artifact["features"]["space"])
        binding["problem_sha256"] = fb.digest(binding["components"])
    return mutate


@pytest.fixture(scope="module")
def pair_run(tmp_path_factory):
    """Two representations x two maps at one budget and seed: four cells that share the roster, the
    acquisition and the configuration, so another cell's artifact is a plausible substitute."""
    root = tmp_path_factory.mktemp("pair")
    world = World()
    paths = world.write(root / "in", plan=plan_document(label_budgets=[6], seeds=[0]))
    out = root / "out"
    summary = run(paths, out)
    assert summary["status_counts"] == {"cells": 4, "ok": 4, "unsupported": 0, "failed": 0}
    return SimpleNamespace(world=world, paths=paths, out=out, summary=summary, root=root)


SUBSTITUTIONS = [
    ("another_representation", ("alpha", "linear"), ("beta", "linear")),
    ("another_map", ("alpha", "linear"), ("alpha", "rbf")),
    ("another_representation_and_map", ("alpha", "rbf"), ("beta", "linear")),
]


@pytest.mark.parametrize("name,victim,donor", SUBSTITUTIONS, ids=[item[0] for item in SUBSTITUTIONS])
def test_another_cells_valid_artifact_cannot_replace_a_cell_even_with_every_hash_recomputed(
        pair_run, tmp_path, name, victim, donor):
    target = copy_freeze(pair_run.out, tmp_path / "frozen")
    assert fm.verify_freeze(target)["index"]["counts"]["ok"] == 4
    donated = fb.read_json(target / fm.CELLS_DIR / cell_file(*donor))
    assert fb.verify_artifact(donated["artifact"]) is True       # a perfectly valid artifact...
    rehash_envelope(target, cell_file(*victim), lambda envelope: envelope.update(
        artifact=copy.deepcopy(donated["artifact"]), capacity=copy.deepcopy(donated["capacity"])))
    with pytest.raises(ValueError, match="another representation or readout map"):
        fm.verify_freeze(target)                                  # ...under the wrong key
    with pytest.raises(ValueError, match="another representation or readout map"):
        fm.evaluate_matrix(target, pair_run.paths.eval, overwrite=True)
    assert fm.verify_freeze(pair_run.out)["index"]["counts"]["ok"] == 4


def test_an_artifact_fitted_on_another_cache_of_the_same_representation_is_refused(pair_run,
                                                                                    tmp_path):
    features = pair_run.world.features["alpha"].copy()
    features[LABELLED + UNLABELLED] += 5.0       # a held-out row: roster, acquisition and plan agree
    other = pair_run.world.variant(features={"alpha": features})
    other_paths = other.write(tmp_path / "other_in", plan=plan_document(label_budgets=[6], seeds=[0]))
    run(other_paths, tmp_path / "other_out")
    donated = fb.read_json(tmp_path / "other_out" / fm.CELLS_DIR / cell_file("alpha", "linear"))
    assert fb.verify_artifact(donated["artifact"]) is True
    target = copy_freeze(pair_run.out, tmp_path / "frozen")
    rehash_envelope(target, cell_file("alpha", "linear"), lambda envelope: envelope.update(
        artifact=donated["artifact"], capacity=donated["capacity"]))
    with pytest.raises(ValueError, match="different cache"):
        fm.verify_freeze(target)
    # the converse: the INDEX claims another cache than every artifact was fitted on
    converse = copy_freeze(pair_run.out, tmp_path / "converse")
    rehash_index(converse, lambda index: index["representations"]["alpha"].update(
        cache_sha256="0" * 64))
    with pytest.raises(ValueError, match="different cache"):
        fm.verify_freeze(converse)


def test_the_index_must_summarise_exactly_the_planned_representations(pair_run, tmp_path):
    missing = copy_freeze(pair_run.out, tmp_path / "missing")
    rehash_index(missing, lambda index: index["representations"].pop("beta"))
    with pytest.raises(ValueError, match="exactly the planned representations"):
        fm.verify_freeze(missing)
    relabelled = copy_freeze(pair_run.out, tmp_path / "relabelled")
    rehash_index(relabelled, lambda index: index["representations"]["beta"].update(
        representation_id="alpha"))
    with pytest.raises(ValueError, match="describes representation"):
        fm.verify_freeze(relabelled)


def test_a_doctored_capacity_is_refused_in_the_cell_and_inside_the_artifact(pair_run, tmp_path):
    victim = cell_file("alpha", "rbf")
    honest = fb.read_json(pair_run.out / fm.CELLS_DIR / victim)["capacity"]
    envelope_only = copy_freeze(pair_run.out, tmp_path / "envelope_only")
    rehash_envelope(envelope_only, victim, lambda envelope: envelope["capacity"].update(
        effective_ridge_df=honest["effective_ridge_df"] + 1.0))
    with pytest.raises(ValueError, match="reported capacity differs"):
        fm.verify_freeze(envelope_only)
    both = copy_freeze(pair_run.out, tmp_path / "both")
    rehash_envelope(both, victim, forge_capacity(
        effective_ridge_df=honest["effective_ridge_df"] + 1.0))
    with pytest.raises(ValueError, match="effective ridge df"):
        fm.verify_freeze(both)
    widened = copy_freeze(pair_run.out, tmp_path / "widened")
    rehash_envelope(widened, victim, forge_capacity(
        output_dimension=honest["output_dimension"] + 1))
    with pytest.raises(ValueError, match="dimensions"):
        fm.verify_freeze(widened)
    assert fm.verify_freeze(pair_run.out)["index"]["counts"]["ok"] == 4


def test_a_forged_acquisition_in_the_readout_record_is_refused(pair_run, tmp_path):
    forged = copy_freeze(pair_run.out, tmp_path / "forged")

    def move_the_seed(envelope):
        schema = envelope["artifact"]["features"]["space"]["schema"]
        schema["acquired"]["seed"] = 99
        binding = envelope["artifact"]["provenance"]["input_binding"]
        binding["components"]["space_sha256"] = fb.digest(envelope["artifact"]["features"]["space"])
        binding["problem_sha256"] = fb.digest(binding["components"])
    rehash_envelope(forged, cell_file("alpha", "linear"), move_the_seed)
    with pytest.raises(ValueError, match="acquisition or applied-row counts"):
        fm.verify_freeze(forged)


# ---------------------------------------------------------------------------
# the public evaluator verifies the disk itself and protects its own report
# ---------------------------------------------------------------------------

def test_the_public_evaluator_verifies_the_current_disk_not_a_stale_dictionary(small_run, tmp_path,
                                                                               monkeypatch):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    stale = fm.verify_freeze(target)                  # a perfectly valid dictionary, taken now...
    with pytest.raises(TypeError):                    # ...and there is no way to hand it in
        fm.evaluate_matrix(target, small_run.paths.eval, verified=stale)
    victim = next(iter(sorted((target / fm.CELLS_DIR).iterdir())))
    victim.write_text(victim.read_text(encoding="utf-8").replace('"status": "ok"',
                                                                 '"status": "failed"', 1),
                      encoding="utf-8")               # ...then the disk changes
    opened = []
    original = fb.read_outcomes

    def spy(path, *, label):
        opened.append(Path(path))
        return original(path, label=label)
    monkeypatch.setattr(fb, "read_outcomes", spy)
    with pytest.raises(ValueError, match="modified after the freeze"):
        fm.evaluate_matrix(target, small_run.paths.eval)
    assert opened == [], "the evaluation outcomes were opened against a freeze that does not verify"
    assert not (target / fm.RESULTS_NAME).exists()
    with pytest.raises(ValueError, match="does not exist"):       # nor does it travel to a directory
        fm.evaluate_matrix(tmp_path / "elsewhere", small_run.paths.eval)
    assert opened == []


@pytest.mark.parametrize("alias_name", ["results", "results_tmp"])
def test_the_public_evaluator_refuses_an_evaluation_file_that_aliases_its_report(
        small_run, tmp_path, alias_name):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    victim = target / {"results": fm.RESULTS_NAME, "results_tmp": fm.RESULTS_NAME + ".tmp"}[alias_name]
    victim.write_bytes(small_run.paths.eval.read_bytes())
    original = victim.read_bytes()
    listing = sorted(str(item.relative_to(target)) for item in target.rglob("*"))
    with pytest.raises(ValueError, match="resolves to the same file"):
        fm.evaluate_matrix(target, victim, overwrite=True)      # even with overwrite
    assert victim.read_bytes() == original, "the evaluation input was replaced by the report"
    assert sorted(str(item.relative_to(target)) for item in target.rglob("*")) == listing


def test_the_public_evaluator_sees_a_dotdot_spelling_and_a_hard_link_as_the_report(small_run,
                                                                                    tmp_path):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    victim = target / fm.RESULTS_NAME
    victim.write_bytes(small_run.paths.eval.read_bytes())
    original = victim.read_bytes()
    with pytest.raises(ValueError, match="resolves to the same file"):
        fm.evaluate_matrix(target, target / ".." / target.name / fm.RESULTS_NAME, overwrite=True)
    link = tmp_path / "linked_eval.csv"
    try:
        os.link(victim, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are unavailable on this filesystem")
    with pytest.raises(ValueError, match="resolves to the same file"):
        fm.evaluate_matrix(target, link, overwrite=True)
    assert victim.read_bytes() == original and link.read_bytes() == original


def test_the_public_evaluator_never_treats_a_frozen_file_as_outcomes_or_overwrites_it(small_run,
                                                                                      tmp_path):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    frozen = {path: path.read_bytes() for path in target.rglob("*") if path.is_file()}
    cell = next(iter(sorted((target / fm.CELLS_DIR).iterdir())))
    for name in (fm.INDEX_NAME, fm.INPUTS_NAME, f"{fm.CELLS_DIR}/{cell.name}"):
        with pytest.raises(ValueError):
            fm.evaluate_matrix(target, target / name, overwrite=True)
    assert {path: path.read_bytes() for path in target.rglob("*") if path.is_file()} == frozen
    assert not (target / fm.RESULTS_NAME).exists()


def spy_on_outcomes(monkeypatch):
    """Record every path ``read_outcomes`` is asked to open; the list is the evidence."""
    opened, original = [], fb.read_outcomes

    def spy(path, *, label):
        opened.append(Path(path))
        return original(path, label=label)
    monkeypatch.setattr(fb, "read_outcomes", spy)
    return opened


def snapshot_files(root):
    return {path: path.read_bytes() for path in Path(root).rglob("*") if path.is_file()}


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("which", ["index", "inputs", "cell"])
def test_a_frozen_file_given_as_the_evaluation_file_is_refused_by_name_before_it_is_opened(
        small_run, tmp_path, monkeypatch, which, overwrite):
    """The frozen files used to sit on the SAME input side as the evaluation file, so passing one
    was only noticed when the CSV parse failed, after it had been opened as outcomes."""
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    cell = sorted((target / fm.CELLS_DIR).iterdir())[0].name
    victim = target / {"index": fm.INDEX_NAME, "inputs": fm.INPUTS_NAME,
                       "cell": f"{fm.CELLS_DIR}/{cell}"}[which]
    before = snapshot_files(target)
    opened = spy_on_outcomes(monkeypatch)
    with pytest.raises(ValueError, match="resolves to the same file as the"):
        fm.evaluate_matrix(target, victim, overwrite=overwrite)
    assert opened == [], "a frozen file was opened as an outcome table"
    assert snapshot_files(target) == before
    assert not (target / fm.RESULTS_NAME).exists()
    assert not (target / (fm.RESULTS_NAME + ".tmp")).exists()


def test_a_hard_link_or_dotdot_spelling_of_a_frozen_file_is_still_that_frozen_file(
        small_run, tmp_path, monkeypatch):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    victim = target / fm.INDEX_NAME
    before = snapshot_files(target)
    opened = spy_on_outcomes(monkeypatch)
    with pytest.raises(ValueError, match="resolves to the same file as the"):
        fm.evaluate_matrix(target, target / ".." / target.name / fm.INDEX_NAME)
    link = tmp_path / "linked_eval.csv"
    try:
        os.link(victim, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are unavailable on this filesystem")
    with pytest.raises(ValueError, match="resolves to the same file as the"):
        fm.evaluate_matrix(target, link, overwrite=True)
    assert opened == [] and link.read_bytes() == before[victim]
    assert snapshot_files(target) == before


STAGING_CASES = [(False, False), (False, True), (True, True)]      # (final report exists, overwrite)


@pytest.mark.parametrize("final_report,overwrite", STAGING_CASES)
@pytest.mark.parametrize("linked", [False, True])
def test_a_pre_existing_report_staging_file_is_never_truncated_with_or_without_overwrite(
        small_run, tmp_path, monkeypatch, linked, final_report, overwrite):
    """MATERIAL DATA LOSS, reproduced: ``matrix_evaluation_report.json.tmp`` hard-linked to a real
    feature-cache header was overwritten by the default evaluator and the header became report JSON.
    A pre-existing staging file is not known to be owned, so it is refused even with overwrite."""
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=not final_report)
    header = tmp_path / "recorded.cache.json"
    header.write_bytes(small_run.paths.caches["alpha"].read_bytes())
    staging = target / (fm.RESULTS_NAME + ".tmp")
    if linked:
        try:
            os.link(header, staging)                  # the staging name IS the cache header
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("hard links are unavailable on this filesystem")
    else:
        staging.write_bytes(b"sentinel bytes that must survive")
    sentinel, header_bytes = staging.read_bytes(), header.read_bytes()
    before = snapshot_files(target)
    opened = spy_on_outcomes(monkeypatch)
    with pytest.raises(ValueError, match="never truncated or deleted"):
        fm.evaluate_matrix(target, small_run.paths.eval, overwrite=overwrite)
    assert opened == [], "the outcomes were opened although the evaluation could not be written"
    assert staging.read_bytes() == sentinel and header.read_bytes() == header_bytes
    assert snapshot_files(target) == before
    assert (target / fm.RESULTS_NAME).exists() is final_report


@pytest.mark.parametrize("overwrite", [False, True])
def test_run_matrix_refuses_an_existing_report_staging_file_before_it_writes_anything(
        small_run, tmp_path, overwrite):
    out = tmp_path / "out"
    out.mkdir()
    staging = out / (fm.RESULTS_NAME + ".tmp")
    staging.write_bytes(b"sentinel bytes that must survive")
    with pytest.raises(ValueError, match="never truncated or deleted"):
        run(small_run.paths, out, overwrite=overwrite)
    assert sorted(item.name for item in out.iterdir()) == [staging.name]
    assert staging.read_bytes() == b"sentinel bytes that must survive"


def test_the_evaluator_stages_through_a_file_it_creates_and_leaves_none_behind(small_run,
                                                                              tmp_path):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    report, staging = target / fm.RESULTS_NAME, target / (fm.RESULTS_NAME + ".tmp")
    results = fm.evaluate_matrix(target, small_run.paths.eval)
    assert report.is_file() and not staging.exists()
    assert fb.read_json(report)["index_sha256"] == results["index_sha256"]
    assert report.read_bytes() == (small_run.out / fm.RESULTS_NAME).read_bytes(), (
        "the exclusive staging path writes the same bytes as the pipeline's own report")


def test_a_failed_report_write_removes_only_the_staging_file_it_created(small_run, tmp_path,
                                                                        monkeypatch):
    target = copy_freeze(small_run.out, tmp_path / "frozen", drop_report=True)
    report, staging = target / fm.RESULTS_NAME, target / (fm.RESULTS_NAME + ".tmp")

    def refuse(self, destination):
        raise OSError("replace refused")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", refuse)
        with pytest.raises(OSError, match="replace refused"):
            fm.evaluate_matrix(target, small_run.paths.eval)
    assert not staging.exists() and not report.exists(), "its own staging file was left behind"
    assert fm.evaluate_matrix(target, small_run.paths.eval)["status_counts"]["ok"] == 2


def test_the_public_evaluator_replaces_an_existing_report_only_with_overwrite(small_run, tmp_path):
    target = copy_freeze(small_run.out, tmp_path / "frozen")              # keeps its report
    report = target / fm.RESULTS_NAME
    before = report.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        fm.evaluate_matrix(target, small_run.paths.eval)
    assert report.read_bytes() == before
    again = fm.evaluate_matrix(target, small_run.paths.eval, overwrite=True)
    assert report.read_bytes() == before and again["index_sha256"] == fb.read_json(
        target / fm.INDEX_NAME)["index_sha256"]
    flipped = write_csv(tmp_path / "flipped.csv", "candidate_id,utility", [
        f"{ident},{-value!r}" for ident, value in small_run.world.eval_truth().items()])
    fm.evaluate_matrix(target, flipped, overwrite=True)
    assert report.read_bytes() != before
    report.unlink()
    report.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        fm.evaluate_matrix(target, small_run.paths.eval, overwrite=True)


def test_run_matrix_hands_its_own_overwrite_authority_to_the_evaluator(small_run, tmp_path):
    out = tmp_path / "out"
    first = run(small_run.paths, out)
    report = out / fm.RESULTS_NAME
    before = report.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        run(small_run.paths, out)
    assert report.read_bytes() == before
    assert run(small_run.paths, out, overwrite=True)["index_sha256"] == first["index_sha256"]
    assert report.read_bytes() == before


def test_the_cli_evaluates_a_freeze_that_run_left_unscored(tmp_path, capsys):
    cli = load_cli()
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    out, later = tmp_path / "out", tmp_path / "later_eval.csv"
    with pytest.raises(ValueError, match="does not exist"):
        run(paths, out, eval_outcomes=later)
    assert fm.verify_freeze(out)["index"]["counts"]["ok"] == 2
    assert not (out / fm.RESULTS_NAME).exists()
    shutil.copyfile(paths.eval, later)
    argv = ["evaluate", "--output-dir", str(out), "--eval-outcomes", str(later)]
    assert cli.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status_counts"] == {"cells": 2, "ok": 2, "unsupported": 0, "failed": 0}
    assert (out / fm.RESULTS_NAME).is_file()
    assert cli.main(argv) == 2
    assert "already exists" in capsys.readouterr().err
    assert cli.main(argv + ["--overwrite"]) == 0
    capsys.readouterr()
    aliased = ["evaluate", "--output-dir", str(out), "--eval-outcomes", str(out / fm.RESULTS_NAME),
               "--overwrite"]
    assert cli.main(aliased) == 2
    assert "resolves to the same file" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the cache writer: four files, four DIFFERENT physical files
# ---------------------------------------------------------------------------

def _file_system_folds_case(directory):
    probe = Path(directory) / "CaseProbe.tmp"
    probe.write_text("probe", encoding="utf-8")
    try:
        return (Path(directory) / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


def _listing(directory):
    return sorted((str(item.relative_to(directory)), item.read_bytes() if item.is_file() else None)
                  for item in Path(directory).rglob("*"))


def test_the_writer_refuses_file_names_that_differ_only_by_letter_case_before_writing(tmp_path):
    if not _file_system_folds_case(tmp_path):
        pytest.skip("this file system distinguishes letter case, so A.npz and a.npz differ")
    args = World().cache_args("alpha")
    folder = tmp_path / "w"
    folder.mkdir()
    with pytest.raises(ValueError, match="same physical file"):
        fm.write_feature_cache(folder / "A.npz", folder / "a.npz", **args)
    assert _listing(folder) == [], "a refusal must be write-free"
    precious = folder / "A.npz"
    precious.write_bytes(b"an existing binary that must survive")
    before = _listing(folder)
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="same physical file"):
            fm.write_feature_cache(folder / "A.npz", folder / "a.npz", overwrite=overwrite, **args)
        assert _listing(folder) == before
    assert precious.read_bytes() == b"an existing binary that must survive"


def test_the_writer_refuses_a_header_that_is_the_npz_or_either_staging_file(tmp_path):
    args = World().cache_args("alpha")
    folder = tmp_path / "w"
    folder.mkdir()
    for header, npz in ((folder / "x.npz", folder / "x.npz"),
                        (folder / "x.cache.npz.tmp", folder / "x.cache.npz"),
                        (folder / "y.cache.json", folder / "y.cache.json.tmp")):
        with pytest.raises(ValueError, match="same physical file|plain file name"):
            fm.write_feature_cache(header, npz, **args)
        with pytest.raises(ValueError, match="same physical file|plain file name"):
            fm.write_feature_cache(header, npz, overwrite=True, **args)
    assert _listing(folder) == [], "a refusal must be write-free"


def test_the_writer_refuses_a_hard_linked_header_and_npz_and_leaves_both_untouched(tmp_path):
    args = World().cache_args("alpha")
    folder = tmp_path / "w"
    header, npz = folder / "x.cache.json", folder / "x.cache.npz"
    fm.write_feature_cache(header, npz, **args)
    header.unlink()
    try:
        os.link(npz, header)                          # the header IS the NPZ, under another name
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are unavailable on this filesystem")
    before = _listing(folder)
    assert dict(before)["x.cache.json"] == dict(before)["x.cache.npz"]
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="same physical file"):
            fm.write_feature_cache(header, npz, overwrite=overwrite, **args)
        assert _listing(folder) == before, "a refusal must leave both names untouched"


def test_an_existing_staging_file_is_never_silently_replaced(tmp_path):
    args = World().cache_args("alpha")
    folder = tmp_path / "w"
    folder.mkdir()
    stale = folder / "x.cache.npz.tmp"
    stale.write_bytes(b"someone else's file")
    with pytest.raises(ValueError, match="already exists"):
        fm.write_feature_cache(folder / "x.cache.json", folder / "x.cache.npz", **args)
    assert stale.read_bytes() == b"someone else's file"
    assert sorted(item.name for item in folder.iterdir()) == ["x.cache.npz.tmp"]
    fm.write_feature_cache(folder / "x.cache.json", folder / "x.cache.npz", overwrite=True, **args)
    assert fm.load_feature_cache(folder / "x.cache.json").ids == tuple(sorted(World().ids))


def test_no_native_model_or_gpu_library_is_imported_by_the_module():
    code = ("import sys; import smallAntibodyGen.experiments.functional_matrix as m; "
            "bad = [n for n in ('torch', 'transformers', 'esm', 'tensorflow', 'jax') "
            "if n in sys.modules]; assert not bad, bad; print('clean')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(SRC)})
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# ---------------------------------------------------------------------------
# the CLI and the selftest
# ---------------------------------------------------------------------------

def test_the_cli_help_lists_every_subcommand_and_the_cache_flag(capsys):
    cli = load_cli()
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    for word in ("selftest", "run", "evaluate", "verify-cache", "acquired", "never"):
        assert word in text
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    assert "--cache" in capsys.readouterr().out


def test_the_cli_runs_the_matrix_on_explicit_paths_and_prints_strict_json(tmp_path, capsys):
    cli = load_cli()
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    argv = ["run", "--candidates", str(paths.candidates), "--train-outcomes", str(paths.train),
            "--eval-outcomes", str(paths.eval), "--manifest", str(paths.manifest),
            "--plan", str(paths.plan), "--cache", f"alpha={paths.caches['alpha']}",
            "--cache-npz", f"alpha={paths.npz['alpha']}", "--output-dir", str(tmp_path / "out")]
    assert cli.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status_counts"] == {"cells": 2, "ok": 2, "unsupported": 0, "failed": 0}
    assert (tmp_path / "out" / fm.RESULTS_NAME).is_file()
    assert cli.main(argv) == 2
    assert "already exists" in capsys.readouterr().err


def test_the_cli_reports_malformed_arguments_and_aliases_with_exit_2(tmp_path, capsys):
    cli = load_cli()
    world = World()
    paths = world.write(tmp_path / "in", plan=small_plan())
    out = tmp_path / "out"
    base = ["run", "--candidates", str(paths.candidates), "--train-outcomes", str(paths.train),
            "--eval-outcomes", str(paths.eval), "--manifest", str(paths.manifest),
            "--plan", str(paths.plan), "--output-dir", str(out)]
    assert cli.main(base + ["--cache", "alpha"]) == 2
    assert "ID=PATH" in capsys.readouterr().err
    assert cli.main(base + ["--cache", f"alpha={paths.caches['alpha']}",
                            "--cache", f"alpha={paths.caches['alpha']}"]) == 2
    assert "twice" in capsys.readouterr().err
    out.mkdir()
    (out / fm.INDEX_NAME).write_bytes(paths.eval.read_bytes())
    aliased = [item if item != str(paths.eval) else str(out / fm.INDEX_NAME) for item in base]
    assert cli.main(aliased + ["--cache", f"alpha={paths.caches['alpha']}", "--overwrite"]) == 2
    assert "resolves to the same file" in capsys.readouterr().err


def test_the_cli_verifies_a_cache_and_refuses_a_tampered_one(tmp_path, capsys):
    cli = load_cli()
    header, npz, world = fresh_cache(tmp_path)
    assert cli.main(["verify-cache", "--header", str(header)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["rows"] == TOTAL and printed["summary"]["feature_dimension"] == 4
    assert printed["summary"]["conditioning"]["quality"]["state"] == "explicit_null"
    np.savez(npz, features=world.features["alpha"] + 1.0)
    assert cli.main(["verify-cache", "--header", str(header)]) == 2
    assert "STALE" in capsys.readouterr().err


def test_the_selftest_passes_every_named_check_and_writes_a_strict_report(tmp_path, capsys):
    cli = load_cli()
    out = tmp_path / "selftest"
    # unrelated data that already sits under the output directory, at the very name the selftest
    # used to delete recursively: it must survive untouched
    precious = out / "matrix_tampered"
    (precious / "nested").mkdir(parents=True)
    (precious / "nested" / "keep.txt").write_text("not the selftest's to delete", encoding="utf-8")
    (precious / "top.txt").write_text("also precious", encoding="utf-8")
    before = _listing(precious)
    assert cli.main(["selftest", "--output-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "selftest PASSED" in printed and "!!" not in printed
    assert _listing(precious) == before and precious.is_dir()
    report = fb.read_json(out / fm.SELFTEST_REPORT_NAME)
    fixture = Path(report["tampered_fixture"])
    assert fixture.is_dir() and fixture != precious and fixture.parent == out
    assert (fixture / "matrix" / fm.INDEX_NAME).is_file(), "the owned fixture was left in place"
    assert report["schema"] == fm.SELFTEST_SCHEMA and report["passed"] is True
    names = {check["name"] for check in report["checks"]}
    for expected in ("matrix::every_planned_cell_is_ok_and_listed",
                     "matrix::evaluation_outcomes_cannot_change_any_pre_evaluation_file",
                     "matrix::row_order_permutation_changes_no_cell_index_or_result_byte",
                     "matrix::acquisitions_are_paired_across_representations_and_maps",
                     "matrix::a_tampered_cell_is_refused"):
        assert expected in names
    assert all(check["holds"] for check in report["checks"])
    assert report["summary"]["status_counts"]["ok"] == 24
    for rep in fm.SELFTEST_REPRESENTATIONS:
        assert fm.load_feature_cache(out / "synthetic" / "caches" / f"{rep}.cache.json")

"""HER2 / trastuzumab fixed-scaffold data access, integrity gates and neighbour search.

What this benchmark is, stated once so no consumer has to infer it: one antibody
lineage, one target, **ten** editable HCDR3 positions inside a fixed heavy
framework and a fixed light chain. The published labels are sorted binding bins
(``high`` vs the lower ``mid``/``low`` bins), not KD values, and nothing here
invents one. It is a local library benchmark, not de novo antibody design.

Three guards live in this module because they are data-flow properties, not
policy:

1. **Test labels.** :func:`load_split` refuses ``test`` unless handed a
   :class:`SelectionUnlock` that :func:`read_selection_freeze` minted from a
   **final**-stage freeze whose every referenced checkpoint hash verified. A
   ``SelectionUnlock`` is also minted for the intermediate initial-SFT freeze,
   because post-training has to verify that one too -- so the stage is checked
   again at the access boundary (:func:`_require_unlock`), not only where the
   token is issued. Overlap diagnostics that
   legitimately need test *sequences* call :func:`test_sequences`, which reads the
   ``seq`` column alone -- the labels are never materialized.
2. **Assay outcomes.** :func:`workbook_table` reads name/heavy/light/design-label
   columns and refuses the KD column under the same unlock rule, so the cohort can
   be built and frozen without anybody seeing an outcome.
3. **Sequence integrity.** Every loader re-derives ``edit_distance`` and the
   class/label correspondence rather than trusting the file, and every scaffold
   offset is an asserted constant (:data:`CORE_START`, :data:`PREFIX_LENGTH`), never
   a substring search performed at generation time.
"""
from __future__ import annotations

import itertools
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from .her2_runtime import require, sha256

#: Raw root, outside Git. Pinned by ``source_manifest.json`` inside it.
RAW_ROOT = "data/raw/her2_functional_20260918"
SPLIT_DIR = ("buzz/data/affinity_data/her2/her2_aff_large/processed/remove_overlap/"
             "random_split/0.7_0.15_0.15")
SUBMITTED_CSV = "buzz/data/spr/spr_submitted.csv"
WORKBOOK_XLSX = "buzz/data/spr/spr_results.xlsx"
PIGGEN_DIR = "piggen"

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
CORE_LENGTH = 10
#: 3-class order used by every classifier head and probability table in this study.
CLASS_ORDER = ("low", "mid", "high")
POSITIVE_CLASS = "high"

WT_CORE = "WGGDGFYAMD"
#: The heavy chain reads ...YYC | SR | <core> | Y | WGQGTLVTVSS. The policy prefix is
#: the start token plus VH[:98], i.e. it ENDS in ...YYCSR and excludes FR4 and all of
#: VL. These are asserted constants; a mismatch is a changed release, not a fixable
#: offset.
ANCHOR_LEFT = "SR"
ANCHOR_RIGHT = "Y"
ANCHOR_START = 96
CORE_START = 98
HEAVY_LENGTH = 120
LIGHT_LENGTH = 107
START_TOKEN = "1"
PREFIX_LENGTH = 99

SPLITS = ("train", "val", "test")
SPLIT_ROWS = {"train": 367042, "val": 78652, "test": 78652}
SPLIT_CLASS_COUNTS = {
    "train": {"high": 120504, "mid": 133008, "low": 113530},
    "val": {"high": 25822, "mid": 28502, "low": 24328},
    "test": {"high": 25823, "mid": 28501, "low": 24328},
}
LIBRARY_TOTAL = 524346

DESIGN_METHODS = ("ablang_all", "ablang_one", "blosum", "esm_one", "protein_mpnn")
#: Workbook columns. Only ``KD`` is an outcome; the rest are metadata and may be read
#: at any time. Sheet1, header on row 16, data from row 17.
WORKBOOK_HEADER_ROW = 16
WORKBOOK_COLUMNS = {"kd_molar": "E", "name": "N", "heavy": "P", "light": "W", "design_label": "AD"}
OUTCOME_COLUMNS = frozenset({"E"})

OUTCOME_QUANTITATIVE = "quantitative"
OUTCOME_NON_BINDING = "non_binding"
OUTCOME_BINDING_UNQUANTIFIED = "binding_unquantified"
OUTCOME_MISSING = "missing"
OUTCOME_UNSUPPORTED = "unsupported"

def _normalize_outcome(text):
    """Upper-case with all whitespace removed. Token tables are stored this way too.

    Storing the tables raw and normalizing only the input is how ``NO BINDING``
    became unmatched: the comparison string had its space stripped while the
    token kept one. Normalizing both sides at definition time removes the class.
    """
    return "".join(str(text).split()).upper()


_NON_BINDING_TOKENS = frozenset(_normalize_outcome(t) for t in
                                ("N.B.", "NB", "N.B", "NO BINDING", "NON-BINDER", "NO BINDER"))
_BINDING_UNQUANTIFIED_TOKENS = frozenset(_normalize_outcome(t) for t in ("I.C.", "IC", "I.C"))

SELECTION_SCHEMA = "her2-selection/1"
#: Stage markers. Only ``final`` may unlock reserved labels; the intermediate
#: initial-SFT selection is a real, hash-verified freeze that deliberately does not.
SELECTION_STAGE_BASE = "initial_sft_selection"
SELECTION_STAGE_FINAL = "final"
SELECTION_STAGES = (SELECTION_STAGE_BASE, SELECTION_STAGE_FINAL)


class Her2GuardError(RuntimeError):
    """A reserved outcome was requested without a verified frozen selection."""


# ---------------------------------------------------------------------------
# core encoding and packed codes
# ---------------------------------------------------------------------------

_CODE_WEIGHTS = np.uint64(1) << (np.uint64(5) * np.arange(CORE_LENGTH, dtype=np.uint64))


def encode_cores(sequences):
    """``(N, 10)`` int8 canonical indices. Every row is validated on its own.

    Row-wise validation is the load-bearing part. Concatenating first and
    checking only the total length lets a 9-mer and an 11-mer cancel out: the
    reshape then succeeds and silently splits residues across the sequence
    boundary, producing two cores that appear in neither input. So length, type
    and alphabet are checked per row, before anything is joined.
    """
    seqs = list(sequences)
    require(len(seqs) > 0, "No cores to encode")
    for position, core in enumerate(seqs):
        require(isinstance(core, str),
                f"Core {position} is {type(core).__name__}, not a string")
        require(len(core) == CORE_LENGTH,
                f"Core {position} has length {len(core)}, expected exactly {CORE_LENGTH}")
        require(core.isascii(), f"Core {position} carries a non-ASCII character")
    table = np.full(256, -1, dtype=np.int8)
    for index, residue in enumerate(CANONICAL):
        table[ord(residue)] = index
    raw = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    index = table[raw].reshape(len(seqs), CORE_LENGTH)
    bad = np.flatnonzero((index < 0).any(axis=1))
    require(bad.size == 0,
            f"Non-canonical residue in HER2 core(s) at row(s) {bad[:5].tolist()}: "
            f"{[seqs[i] for i in bad[:5]]}")
    return index


def decode_cores(index):
    """Inverse of :func:`encode_cores`; the round trip is exact."""
    values = np.asarray(index)
    require(values.ndim == 2 and values.shape[1] == CORE_LENGTH, "Expected (N, 10) core indices")
    require(bool(((values >= 0) & (values < len(CANONICAL))).all()), "Core index out of range")
    letters = np.array(list(CANONICAL))
    return ["".join(row) for row in letters[values]]


def pack_codes(index, drop_positions=()):
    """Pack a core into one uint64, 5 bits per site, optionally deleting sites.

    Ten sites x 5 bits = 50 bits, so the packing is lossless and equality of two
    codes is equality of the kept residues -- there is no hash collision to test
    for. Deleted sites contribute zero, which is exactly the "ignore this
    position" semantics the neighbour search needs.
    """
    values = np.asarray(index, dtype=np.int64)
    require(values.ndim == 2 and values.shape[1] == CORE_LENGTH, "Expected (N, 10) core indices")
    weights = _CODE_WEIGHTS.copy()
    for position in drop_positions:
        require(0 <= position < CORE_LENGTH, "Deletion position outside the core")
        weights[position] = np.uint64(0)
    return (values.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def hamming_to(index, reference):
    """Per-row Hamming distance to one reference core."""
    values = np.asarray(index)
    target = np.asarray(reference)
    require(values.ndim == 2 and target.shape == (values.shape[1],), "Reference core shape")
    return (values != target[None, :]).sum(axis=1).astype(np.int64)


def support_counts(index):
    """``(10, 20)`` residue counts, i.e. which residues the data actually supports."""
    values = np.asarray(index)
    counts = np.zeros((CORE_LENGTH, len(CANONICAL)), dtype=np.int64)
    for position in range(CORE_LENGTH):
        counts[position] = np.bincount(values[:, position], minlength=len(CANONICAL))
    return counts


@dataclass(frozen=True)
class NeighbourLookup:
    """Exact nearest-training-neighbour result, with the fallback made visible."""

    distance: np.ndarray          # exact Hamming distance, or -1 beyond max_distance
    neighbour_count: np.ndarray   # how many training rows sit at that exact distance
    label_mean: np.ndarray        # mean label over ALL nearest neighbours; NaN if none
    prior: float

    @property
    def score(self):
        """The baseline score: nearest-neighbour label mean, else the class prior."""
        return np.where(self.distance >= 0, np.nan_to_num(self.label_mean, nan=self.prior),
                        self.prior)

    def strata(self):
        """Labels ``0``/``1``/``2``/``>=3`` for proximity-stratified reporting."""
        return np.where(self.distance < 0, ">=3", self.distance.astype(str))


def nearest_training_labels(query_index, train_index, train_labels, *, max_distance=2, prior=None):
    """Exact nearest training neighbour by deletion masks over packed 5-bit codes.

    Why this is exact, and why nothing is double counted: masks are applied in
    increasing size ``k``, and a query is removed from the pending set as soon as
    it matches at some ``k``. So every hit found at size ``k`` is a pair at
    distance *exactly* ``k`` -- a closer pair would have been resolved earlier --
    and a pair at distance exactly ``k`` agrees on all but those ``k`` sites, so it
    matches exactly one mask of size ``k``: the one deleting precisely the
    differing positions. Each neighbour is therefore counted once, and the label
    mean is over *all* tied nearest neighbours rather than an arbitrary one.

    ``prior`` is the training positive rate and is the declared fallback when no
    neighbour exists within ``max_distance``. There is no k-sweep and nothing here
    is tuned.
    """
    queries = np.asarray(query_index)
    train = np.asarray(train_index)
    labels = np.asarray(train_labels, dtype=np.float64)
    require(train.ndim == 2 and train.shape[1] == CORE_LENGTH, "Expected (N, 10) training cores")
    require(labels.shape == (train.shape[0],), "One label per training core")
    require(bool(np.isfinite(labels).all()), "Training labels must be finite")
    require(isinstance(max_distance, int) and 0 <= max_distance <= 3, "Unsupported max_distance")
    prior = float(labels.mean()) if prior is None else float(prior)

    total = queries.shape[0]
    distance = np.full(total, -1, dtype=np.int64)
    counts = np.zeros(total, dtype=np.int64)
    means = np.full(total, np.nan, dtype=np.float64)
    pending = np.arange(total)
    for size in range(max_distance + 1):
        if pending.size == 0:
            break
        hit_sum = np.zeros(pending.size, dtype=np.float64)
        hit_count = np.zeros(pending.size, dtype=np.int64)
        for positions in itertools.combinations(range(CORE_LENGTH), size):
            train_codes = pack_codes(train, positions)
            order = np.argsort(train_codes, kind="stable")
            ordered = train_codes[order]
            cumulative = np.concatenate([[0.0], np.cumsum(labels[order])])
            probe = pack_codes(queries[pending], positions)
            left = np.searchsorted(ordered, probe, side="left")
            right = np.searchsorted(ordered, probe, side="right")
            hit_sum += cumulative[right] - cumulative[left]
            hit_count += right - left
        found = hit_count > 0
        rows = pending[found]
        distance[rows] = size
        counts[rows] = hit_count[found]
        means[rows] = hit_sum[found] / hit_count[found]
        pending = pending[~found]
    return NeighbourLookup(distance=distance, neighbour_count=counts, label_mean=means, prior=prior)


def mean_pairwise_hamming(index):
    """Exact mean pairwise Hamming over all N(N-1)/2 pairs from one-hot counts.

    Aggregating counts per site gives the identical-pair total per site, so the
    exact mean needs no O(N^2) distance matrix.
    """
    values = np.asarray(index)
    n = values.shape[0]
    require(n >= 2, "Mean pairwise distance needs at least two sequences")
    counts = support_counts(values).astype(np.float64)
    mismatched = (n * n - (counts ** 2).sum(axis=1)).sum()
    return float(mismatched / (n * (n - 1)))


def site_entropy(index):
    """Per-position entropy in nats over the realized residue frequencies."""
    counts = support_counts(np.asarray(index)).astype(np.float64)
    frequency = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(frequency > 0, frequency * np.log(frequency), 0.0)
    return (-terms.sum(axis=1)).tolist()


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionUnlock:
    """Evidence that a frozen selection was verified before any outcome was read.

    Minted only by :func:`read_selection_freeze`, and it carries the ``stage`` of
    the freeze it came from -- a base-stage token is evidence that the initial-SFT
    freeze verified, and nothing else. Only a ``final``-stage token passes
    :func:`_require_unlock`.

    This is a workflow and schema check, not a security token: a caller holding
    the module can obviously construct one. What it does buy is that the ordinary
    code path cannot reach a reserved label without a file on disk naming every
    artifact, and that the artifacts hash to what the file says.
    """

    freeze_path: Path
    freeze_sha256: str
    config_sha256: str
    stage: str
    verified: tuple


def read_selection_freeze(path, *, root, expected_config_sha256=None,
                          expected_stage=SELECTION_STAGE_FINAL, expected_selected=None,
                          expected_code_digests=None, expected_source_digests=None):
    """Load a frozen selection, verify every artifact it names, and mint the unlock.

    Three refusals that the earlier version did not make:

    * an **empty** ``selected`` block no longer mints an unlock -- it verified
      nothing, so it proved nothing;
    * ``expected_stage`` separates the intermediate initial-SFT selection (which
      starts post-training) from the ``final`` freeze (which is the only one that
      may open reserved labels);
    * ``expected_selected``, when given, must match the freeze's key set
      **exactly** -- a freeze that is merely the right size, or that quietly drops
      a failed arm, is rejected rather than evaluated.
    """
    path = Path(path)
    require(path.is_file(), f"No frozen selection at {path}; run the training stage first")
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema_version") == SELECTION_SCHEMA,
            f"Unsupported selection schema {document.get('schema_version')!r}")
    stage = document.get("stage")
    require(stage in SELECTION_STAGES, f"Frozen selection has no known stage marker: {stage!r}")
    require(stage == expected_stage,
            f"This freeze is stage {stage!r}; {expected_stage!r} is required here. The "
            "initial-SFT selection starts post-training and never unlocks reserved labels.")
    if expected_config_sha256 is not None:
        require(document["config_sha256"] == expected_config_sha256,
                "Frozen selection was produced by a different run configuration")
    if expected_code_digests is not None:
        require(document.get("code_digests") == expected_code_digests,
                "Frozen selection was produced by different scientific code; re-freeze against "
                "the current sources rather than evaluating a stale identity")
    if expected_source_digests is not None:
        require(document.get("source_digests") == expected_source_digests,
                "Frozen selection was produced against different pinned sources")
    selected = document.get("selected") or {}
    require(isinstance(selected, dict) and selected,
            "Frozen selection names no artifacts; an empty selection verifies nothing")
    if expected_selected is not None:
        expected_names = set(expected_selected)
        actual = set(selected)
        require(actual == expected_names,
                f"Frozen selection key set mismatch. Missing: {sorted(expected_names - actual)}; "
                f"unexpected: {sorted(actual - expected_names)}")
    verified = []
    for name, entry in sorted(selected.items()):
        target = Path(root) / entry["checkpoint"]
        require(target.is_file(), f"Selected checkpoint is missing: {target}")
        actual = sha256(target)
        require(actual == entry["sha256"],
                f"Selected checkpoint {name} changed on disk: {actual} != {entry['sha256']}")
        verified.append(name)
    return document, SelectionUnlock(freeze_path=path, freeze_sha256=sha256(path),
                                     config_sha256=document["config_sha256"], stage=stage,
                                     verified=tuple(verified))


def _require_unlock(unlock, what):
    """The access boundary. A token is not enough; it has to be the FINAL one.

    ``read_selection_freeze`` mints a token for the intermediate initial-SFT freeze
    too, because post-training needs that freeze verified. Checking only the token's
    *type* here meant the ordinary intermediate token -- obtainable by any stage
    that starts a continuation -- reached the test and workbook loaders. The stage
    is therefore re-checked at the point of access, not only at the point of issue.
    """
    if not isinstance(unlock, SelectionUnlock):
        raise Her2GuardError(
            f"{what} may only be read through a verified SelectionUnlock. Build the frozen "
            "selection first; auditing these values earlier is exactly the leak the freeze "
            "exists to prevent.")
    if unlock.stage != SELECTION_STAGE_FINAL:
        raise Her2GuardError(
            f"{what} needs a {SELECTION_STAGE_FINAL!r}-stage unlock; this one is "
            f"{unlock.stage!r}. The initial-SFT selection starts post-training and never opens "
            "reserved labels -- the final freeze, written after every continuation budget has "
            "been evaluated on validation, is the only one that does.")


def split_path(root, split):
    require(split in SPLITS, f"Unknown split {split!r}")
    return Path(root) / SPLIT_DIR / f"{split}.csv"


def load_split(root, split, *, unlock=None, expect_counts=True):
    """Load one published split and re-derive every column that can be re-derived.

    ``class``/``label`` agreement and ``edit_distance`` are recomputed rather than
    trusted, so a corrupted or re-sorted file fails here instead of quietly
    becoming a training signal.
    """
    if split == "test":
        _require_unlock(unlock, "Reserved test labels")
    frame = pd.read_csv(split_path(root, split), dtype={"seq": str, "class": str},
                        keep_default_na=False)
    require(list(frame.columns) == ["seq", "class", "label", "edit_distance"],
            f"Unexpected columns in {split}.csv: {list(frame.columns)}")
    require(frame.seq.is_unique, f"Duplicate cores inside {split}.csv")
    require(set(frame["class"]) <= set(CLASS_ORDER), f"Unknown class label in {split}.csv")
    index = encode_cores(frame.seq)
    require(bool((frame.label.to_numpy() == (frame["class"] == POSITIVE_CLASS).to_numpy()).all()),
            f"label disagrees with class in {split}.csv")
    require(bool((hamming_to(index, encode_cores([WT_CORE])[0])
                  == frame.edit_distance.to_numpy()).all()),
            f"edit_distance disagrees with Hamming distance to {WT_CORE} in {split}.csv")
    if expect_counts:
        require(len(frame) == SPLIT_ROWS[split],
                f"{split}.csv has {len(frame)} rows, expected {SPLIT_ROWS[split]}")
        observed = frame["class"].value_counts().to_dict()
        require(observed == SPLIT_CLASS_COUNTS[split],
                f"{split}.csv class counts {observed} differ from the audited release")
    return frame


def test_sequences(root):
    """Test-split cores with no labels read at all -- ``usecols`` keeps them off the wire.

    Overlap and novelty diagnostics need to know whether a generated core exists
    in the test split. They do not need to know its bin, and this function cannot
    tell them.
    """
    frame = pd.read_csv(split_path(root, "test"), usecols=["seq"], dtype={"seq": str},
                        keep_default_na=False)
    require(list(frame.columns) == ["seq"], "test sequence read pulled a non-sequence column")
    return frame.seq


def audit_split(frame, split):
    index = encode_cores(frame.seq)
    distance = hamming_to(index, encode_cores([WT_CORE])[0])
    positives = frame["class"] == POSITIVE_CLASS
    return {
        "rows": int(len(frame)), "unique_cores": int(frame.seq.nunique()),
        "class_counts": {k: int(v) for k, v in frame["class"].value_counts().items()},
        "positive_rate": float(positives.mean()),
        "core_lengths": sorted({len(s) for s in frame.seq}),
        "residues_observed": "".join(sorted({c for s in frame.seq for c in s})),
        "sites_with_full_20_support": int((support_counts(index) > 0).all(axis=1).sum()),
        "wt_distance_counts": {str(int(d)): int(c) for d, c in
                               zip(*np.unique(distance, return_counts=True))},
        "positive_rate_by_wt_distance": {
            str(int(d)): float(positives.to_numpy()[distance == d].mean())
            for d in np.unique(distance)},
        "split": split,
    }


# ---------------------------------------------------------------------------
# fixed scaffold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scaffold:
    """The one heavy/light pair every row in this benchmark shares."""

    heavy: str
    light: str

    @property
    def prefix(self):
        """Start token + VH[:98]. Ends ...YYCSR; carries no FR4 and no light chain."""
        return START_TOKEN + self.heavy[:CORE_START]

    @property
    def suffix(self):
        return self.heavy[CORE_START + CORE_LENGTH:]

    def with_core(self, core):
        require(len(core) == CORE_LENGTH, "Core length must match the fixed scaffold")
        return self.heavy[:CORE_START] + core + self.suffix


def load_scaffold(root):
    """Read the fixed trastuzumab VH/VL from row 0 of the submitted design table.

    Every offset is asserted against the released sequence. The prefix is derived
    from :data:`CORE_START`, not from ``str.index``, so a changed release fails
    loudly instead of silently shifting the editable window.
    """
    frame = read_submitted(root)
    row = frame.iloc[0]
    heavy, light = str(row["H"]), str(row["L"])
    require(len(heavy) == HEAVY_LENGTH and len(light) == LIGHT_LENGTH,
            "Fixed scaffold chain lengths changed")
    require(heavy[ANCHOR_START:CORE_START] == ANCHOR_LEFT, "Left HCDR3 anchor moved")
    require(heavy[CORE_START:CORE_START + CORE_LENGTH] == WT_CORE, "Wild-type core moved")
    require(heavy[CORE_START + CORE_LENGTH] == ANCHOR_RIGHT, "Right HCDR3 anchor moved")
    require(heavy.count(ANCHOR_LEFT + WT_CORE + ANCHOR_RIGHT) == 1,
            "The SR+core+Y motif is not unique in the heavy chain")
    require(str(row["H3"]) == WT_CORE, "Row 0 H3 column is not the wild-type core")
    scaffold = Scaffold(heavy=heavy, light=light)
    require(len(scaffold.prefix) == PREFIX_LENGTH, "Policy prefix length changed")
    require(scaffold.prefix.endswith("YYC" + ANCHOR_LEFT), "Policy prefix no longer ends ...YYCSR")
    return scaffold


def read_submitted(root):
    frame = pd.read_csv(Path(root) / SUBMITTED_CSV, dtype=str, keep_default_na=False)
    require(list(frame.columns) == ["name", "plate", "H", "L", "H_opt", "L_opt", "Label", "H3"],
            f"Unexpected columns in {SUBMITTED_CSV}: {list(frame.columns)}")
    require(frame.name.is_unique, "Submitted design names are not unique")
    return frame


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def read_downloaded_manifest(root):
    """The retrieval record written next to the data. Read only; never rewritten."""
    document = json.loads((Path(root) / "source_manifest.json").read_text(encoding="utf-8"))
    require(document["files"], "Downloaded source manifest lists no files")
    return document


def read_piggen_manifest(root):
    return json.loads((Path(root) / PIGGEN_DIR / "manifest.json").read_text(encoding="utf-8"))


def verify_downloaded_files(root, repository_root):
    """Re-hash every pinned download. Returns one problem string per mismatch."""
    problems = []
    for entry in read_downloaded_manifest(root)["files"]:
        target = Path(repository_root) / entry["local_path"]
        if not target.is_file():
            problems.append(f"{entry['local_path']}: missing")
            continue
        size = target.stat().st_size
        if size != entry["bytes"]:
            problems.append(f"{entry['local_path']}: {size} bytes != manifest {entry['bytes']}")
        digest = sha256(target)
        if digest != entry["sha256"]:
            problems.append(f"{entry['local_path']}: sha256 {digest} != manifest {entry['sha256']}")
    for entry in read_piggen_manifest(root)["files"]:
        target = Path(repository_root) / entry["path"]
        if not target.is_file():
            problems.append(f"{entry['path']}: missing")
            continue
        if target.stat().st_size != entry["bytes"]:
            problems.append(f"{entry['path']}: size != manifest {entry['bytes']}")
        if sha256(target) != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 != manifest {entry['sha256']}")
    return tuple(problems)


def verify_tracked_manifests(repository_root, manifest_paths, raw_root):
    """Re-hash every file named by the TRACKED manifests against what is on disk.

    :func:`verify_downloaded_files` checks the untracked retrieval record, which
    lives next to the data and could in principle be rewritten by whatever wrote
    the data. The tracked manifests under ``specs/benchmarks/`` are the ones in
    version control, so they are the ones a reviewer can diff -- and until this
    existed only the audit script checked them, which meant a fit could consume
    files the committed manifests did not describe.
    """
    from ..benchmarks import provenance as prov
    problems = []
    for relative in manifest_paths:
        document = prov.load_manifest_document(Path(repository_root) / relative)
        manifest = document.validated()
        for problem in prov.verify_manifest_files(manifest, raw_root):
            problems.append(f"{relative}: {problem}")
    return tuple(problems)


def verify_all_sources(raw_root, repository_root, manifest_paths):
    """Both source-of-truth checks, in the one call every entry point makes."""
    return (tuple(verify_downloaded_files(raw_root, repository_root))
            + tuple(verify_tracked_manifests(repository_root, manifest_paths, raw_root)))


def source_digests(repository_root, raw_root, manifest_paths):
    """Every pinned input's hash, in the one shape every stage compares.

    Built once here because the stages have to compare *the same* dictionary: the
    training run recorded one construction and the continuation another, so the
    source lineage could not be checked across the freeze at all. Tracked manifests
    are re-hashed from disk; the data and weight hashes come from the retrieval
    records, which :func:`verify_all_sources` has already validated against the
    files themselves.
    """
    digests = {relative: sha256(Path(repository_root) / relative) for relative in manifest_paths}
    for entry in read_downloaded_manifest(raw_root)["files"]:
        digests[entry["local_path"]] = entry["sha256"]
    for entry in read_piggen_manifest(raw_root)["files"]:
        digests[entry["path"]] = entry["sha256"]
    return digests


def labelled_lookup(frames):
    """``core -> (split, class)`` for the splits whose labels the caller may read.

    Keeping the split beside the class is what makes a per-split conditional hit
    rate expressible. Pooling them is how a memorized training positive gets
    counted as evidence of held-out generalization.
    """
    split_of, class_of = {}, {}
    for name, frame in frames.items():
        for core, label in zip(frame.seq, frame["class"]):
            require(core not in split_of, f"Core {core} appears in two splits")
            split_of[core], class_of[core] = name, label
    return split_of, class_of


# ---------------------------------------------------------------------------
# independent assay workbook (stdlib .xlsx reader)
# ---------------------------------------------------------------------------

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")
_SUPPORTED_CELL_TYPES = frozenset({None, "n", "s", "str", "inlineStr"})


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root:
        parts = []
        for child in item:
            if _tag(child) == "t":
                parts.append(child.text or "")
            elif _tag(child) == "r":
                parts.extend(grandchild.text or "" for grandchild in child
                             if _tag(grandchild) == "t")
        values.append("".join(parts))
    return values


def _first_sheet_target(archive):
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find(f"{_MAIN_NS}sheets")
    require(sheets is not None and len(sheets), "Workbook declares no sheets")
    sheet = sheets[0]
    relationship = sheet.get(f"{_REL_NS}id")
    rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for entry in rels:
        if entry.get("Id") == relationship:
            target = entry.get("Target")
            return sheet.get("name"), ("xl/" + target if not target.startswith("/")
                                       else target.lstrip("/"))
    raise ValueError(f"Workbook relationship {relationship!r} has no target")


def read_workbook(path, columns, *, unlock=None):
    """Read the requested columns of the first sheet with the standard library only.

    ``openpyxl`` is not installed and is not needed: an .xlsx file is a ZIP of XML.
    Columns are named by letter and everything else is skipped, which is also how
    the outcome column stays unread -- asking for it without a verified unlock
    raises here rather than deeper in the analysis.
    """
    wanted = {str(c).upper() for c in columns}
    require(wanted, "No workbook columns requested")
    if wanted & OUTCOME_COLUMNS:
        _require_unlock(unlock, f"Workbook outcome columns {sorted(wanted & OUTCOME_COLUMNS)}")
    rows = {}
    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        sheet_name, target = _first_sheet_target(archive)
        with archive.open(target) as stream:
            for _, element in ElementTree.iterparse(stream, events=("end",)):
                if _tag(element) != "c":
                    continue
                reference = element.get("r") or ""
                match = _CELL_REF.match(reference)
                if match is None or match.group(1) not in wanted:
                    element.clear()
                    continue
                kind = element.get("t")
                require(kind in _SUPPORTED_CELL_TYPES,
                        f"Unsupported cell type {kind!r} at {reference}; refusing to guess")
                if kind == "inlineStr":
                    node = element.find(f"{_MAIN_NS}is")
                    text = "".join(t.text or "" for t in node.iter(f"{_MAIN_NS}t")) if node is not None else ""
                else:
                    node = element.find(f"{_MAIN_NS}v")
                    text = "" if node is None or node.text is None else node.text
                    if kind == "s" and text != "":
                        text = strings[int(text)]
                rows.setdefault(int(match.group(2)), {})[match.group(1)] = text
                element.clear()
    return sheet_name, rows


def workbook_table(root, *, unlock=None, include_outcome=False):
    """Design metadata from the assay workbook; the KD column only under unlock."""
    columns = dict(WORKBOOK_COLUMNS)
    if not include_outcome:
        columns.pop("kd_molar")
    sheet_name, rows = read_workbook(Path(root) / WORKBOOK_XLSX, columns.values(), unlock=unlock)
    header = {name: rows.get(WORKBOOK_HEADER_ROW, {}).get(letter, "")
              for name, letter in columns.items()}
    records = []
    for number in sorted(rows):
        if number <= WORKBOOK_HEADER_ROW:
            continue
        cells = rows[number]
        record = {"workbook_row": number}
        record.update({name: cells.get(letter, "") for name, letter in columns.items()})
        if any(str(record[name]).strip() for name in columns):
            records.append(record)
    frame = pd.DataFrame(records)
    require(len(frame) > 0, "Workbook carried no data rows below the header")
    return {"sheet_name": sheet_name, "header": header, "table": frame}


def classify_outcome(value):
    """Map one workbook KD cell to (outcome class, KD in molar or None).

    ``N.B.`` is non-binding and carries no value. ``I.C.`` means binding was
    observed but not quantified: it is a binary positive with **no** KD, never a
    fabricated number and never a negative. Anything else non-empty is reported
    as unsupported rather than guessed.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return OUTCOME_MISSING, None
    upper = _normalize_outcome(text)
    if upper in _NON_BINDING_TOKENS:
        return OUTCOME_NON_BINDING, None
    if upper in _BINDING_UNQUANTIFIED_TOKENS:
        return OUTCOME_BINDING_UNQUANTIFIED, None
    try:
        number = float(text)
    except ValueError:
        return OUTCOME_UNSUPPORTED, None
    if not np.isfinite(number) or number <= 0:
        return OUTCOME_UNSUPPORTED, None
    return OUTCOME_QUANTITATIVE, number


def assay_cohort(root, scaffold, library_cores, *, unlock=None, include_outcome=False):
    """Build the independent-assay cohort and record every exclusion by count.

    Membership rules, fixed before any outcome is read: identical heavy prefix,
    heavy suffix and light chain as the training scaffold; a canonical 10-mer
    core; one of the five design methods; and **no** exact core overlap with any
    published Buzz split, because a design the policy was trained on is not an
    independent test of it. Repeated cores are collapsed so a control replicated
    across plates does not become pseudoreplication.
    """
    loaded = workbook_table(root, unlock=unlock, include_outcome=include_outcome)
    frame = loaded["table"].copy()
    submitted = read_submitted(root)
    merged = frame.merge(submitted[["name", "H", "L", "Label", "H3"]], on="name", how="left",
                         validate="one_to_one")
    require(bool(merged["H"].notna().all()), "A workbook row has no matching submitted design")
    require(bool((merged.H == merged.heavy).all()) and bool((merged.L == merged.light).all()),
            "Workbook heavy/light disagree with the submitted design table")
    require(bool((merged.Label == merged.design_label).all()),
            "Workbook design label disagrees with the submitted design table")

    same_scaffold = ((merged.heavy.str.len() == HEAVY_LENGTH)
                     & (merged.light == scaffold.light)
                     & merged.heavy.str[:CORE_START].eq(scaffold.heavy[:CORE_START])
                     & merged.heavy.str[CORE_START + CORE_LENGTH:].eq(scaffold.suffix))
    merged["core"] = np.where(same_scaffold,
                              merged.heavy.str[CORE_START:CORE_START + CORE_LENGTH], "")
    canonical = merged.core.apply(lambda c: len(c) == CORE_LENGTH and set(c) <= set(CANONICAL))
    merged["same_scaffold"] = same_scaffold & canonical
    require(bool((merged.loc[merged.same_scaffold & merged.H3.ne(""), "core"]
                  == merged.loc[merged.same_scaffold & merged.H3.ne(""), "H3"]).all()),
            "Core derived from the heavy chain disagrees with the released H3 column")

    library = set(library_cores)
    merged["in_library"] = merged.core.isin(library) & merged.same_scaffold
    merged["is_design_method"] = merged.design_label.isin(DESIGN_METHODS)
    primary = merged[merged.same_scaffold & merged.is_design_method & ~merged.in_library]
    deduped = primary.drop_duplicates(subset="core", keep="first")
    controls = merged[merged.same_scaffold & ~merged.is_design_method]
    counts = {
        "workbook_rows": int(len(merged)),
        "design_label_counts": {k: int(v) for k, v in merged.design_label.value_counts().items()},
        "same_scaffold_rows": int(merged.same_scaffold.sum()),
        "unsupported_scaffold_rows": int((~merged.same_scaffold).sum()),
        "unsupported_scaffold_labels": {k: int(v) for k, v in
                                        merged.loc[~merged.same_scaffold, "design_label"]
                                        .value_counts().items()},
        "design_method_rows": int((merged.same_scaffold & merged.is_design_method).sum()),
        "library_overlap_rows": int((merged.in_library & merged.is_design_method).sum()),
        "library_overlap_cores": sorted(set(merged.loc[merged.in_library, "core"])),
        "primary_rows": int(len(primary)),
        "primary_unique_cores": int(len(deduped)),
        "primary_duplicate_rows": int(len(primary) - len(deduped)),
        "control_rows": int(len(controls)),
        "control_labels": {k: int(v) for k, v in controls.design_label.value_counts().items()},
    }
    return {"all": merged, "primary": deduped.reset_index(drop=True),
            "controls": controls.reset_index(drop=True), "counts": counts,
            "header": loaded["header"], "sheet_name": loaded["sheet_name"]}

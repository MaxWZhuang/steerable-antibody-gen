"""HER2 data-layer tests: guards, integrity, exact neighbour search, workbook parsing.

Everything here is synthetic except two tests that read the committed manifests and
(when the untracked raw root happens to be present) cross-check their hashes.
Nothing downloads.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from smallAntibodyGen.benchmarks import provenance as prov
from smallAntibodyGen.experiments import her2_data as data

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def hamming(left, right):
    return sum(a != b for a, b in zip(left, right))


def core_rows(seed, count):
    rng = np.random.default_rng(seed)
    letters = np.array(list(data.CANONICAL))
    seen, rows = set(), []
    while len(rows) < count:
        core = "".join(rng.choice(letters, size=data.CORE_LENGTH))
        if core in seen:
            continue
        seen.add(core)
        rows.append(core)
    return rows


def write_split(path, rows):
    """rows: iterable of (core, class). label and edit_distance are derived correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["seq,class,label,edit_distance"]
    for core, name in rows:
        lines.append(f"{core},{name},{int(name == 'high')},{hamming(core, data.WT_CORE)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SYNTHETIC_HEAVY = ("A" * 93 + "YYC" + data.ANCHOR_LEFT + data.WT_CORE + "YWGQGTLVTVSS")
SYNTHETIC_LIGHT = "D" * data.LIGHT_LENGTH


def write_submitted(path, designs):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name,plate,H,L,H_opt,L_opt,Label,H3"]
    for name, heavy, light, label, h3 in designs:
        lines.append(f"{name},1,{heavy},{light},,,{label},{h3}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_xlsx(path, rows, *, sheet_name="Sheet1"):
    """Minimal .xlsx: one sheet, shared strings for text, raw numbers otherwise."""
    strings, lookup = [], {}
    cells = {}
    for number, columns in sorted(rows.items()):
        parts = []
        for letter, value in sorted(columns.items()):
            if isinstance(value, (int, float)):
                parts.append(f'<c r="{letter}{number}"><v>{value}</v></c>')
            else:
                if value not in lookup:
                    lookup[value] = len(strings)
                    strings.append(value)
                parts.append(f'<c r="{letter}{number}" t="s"><v>{lookup[value]}</v></c>')
        cells[number] = f'<row r="{number}">{"".join(parts)}</row>'
    sheet = (f'<worksheet xmlns="{MAIN_NS}"><sheetData>'
             + "".join(cells[n] for n in sorted(cells)) + "</sheetData></worksheet>")
    shared = (f'<sst xmlns="{MAIN_NS}" count="{len(strings)}" uniqueCount="{len(strings)}">'
              + "".join(f"<si><t>{value}</t></si>" for value in strings) + "</sst>")
    workbook = (f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}"><sheets>'
                f'<sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = (f'<Relationships xmlns="{REL_NS}"><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml" Type="worksheet"/></Relationships>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


@pytest.fixture
def raw_root(tmp_path):
    """A miniature stand-in for the pinned raw root: splits, designs, workbook."""
    root = tmp_path / "raw"
    cores = core_rows(20260918, 30)
    train = [(core, ["high", "mid", "low"][i % 3]) for i, core in enumerate(cores[:18])]
    val = [(core, ["high", "mid", "low"][i % 3]) for i, core in enumerate(cores[18:24])]
    test = [(core, ["high", "mid", "low"][i % 3]) for i, core in enumerate(cores[24:])]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_split(root / data.SPLIT_DIR / f"{name}.csv", rows)
    designs = [("IgG_1", SYNTHETIC_HEAVY, SYNTHETIC_LIGHT, "Positive Control", data.WT_CORE)]
    workbook = {16: {"E": "HER2 KD (M)", "N": "IgG", "P": "H", "W": "L", "AD": "Label"},
                17: {"E": 1e-9, "N": "IgG_1", "P": SYNTHETIC_HEAVY, "W": SYNTHETIC_LIGHT,
                     "AD": "Positive Control"}}
    designed = core_rows(20260919, 4)
    for offset, core in enumerate(designed, start=2):
        heavy = SYNTHETIC_HEAVY[:data.CORE_START] + core + SYNTHETIC_HEAVY[data.CORE_START + 10:]
        method = data.DESIGN_METHODS[offset % len(data.DESIGN_METHODS)]
        designs.append((f"IgG_{offset}", heavy, SYNTHETIC_LIGHT, method, core))
        workbook[16 + offset] = {"E": ["N.B.", "I.C.", 2.5e-8, ""][offset % 4],
                                 "N": f"IgG_{offset}", "P": heavy, "W": SYNTHETIC_LIGHT,
                                 "AD": method}
    # one design that duplicates a training core, so the overlap exclusion has work to do
    overlap_heavy = (SYNTHETIC_HEAVY[:data.CORE_START] + train[0][0]
                     + SYNTHETIC_HEAVY[data.CORE_START + 10:])
    designs.append(("IgG_6", overlap_heavy, SYNTHETIC_LIGHT, "blosum", train[0][0]))
    workbook[22] = {"E": 3.1e-9, "N": "IgG_6", "P": overlap_heavy, "W": SYNTHETIC_LIGHT,
                    "AD": "blosum"}
    # one row on an unsupported scaffold (short heavy), which must be excluded with a count
    designs.append(("IgG_7", SYNTHETIC_HEAVY[:-1], SYNTHETIC_LIGHT, "ablang_all_len9", ""))
    workbook[23] = {"E": "N.B.", "N": "IgG_7", "P": SYNTHETIC_HEAVY[:-1], "W": SYNTHETIC_LIGHT,
                    "AD": "ablang_all_len9"}
    write_submitted(root / data.SUBMITTED_CSV, designs)
    write_xlsx(root / data.WORKBOOK_XLSX, workbook)
    return root


# ---------------------------------------------------------------------------
# T9  fixed scaffold offsets are asserted constants
# ---------------------------------------------------------------------------

def test_scaffold_offsets_are_constants_not_searches(raw_root):
    scaffold = data.load_scaffold(raw_root)
    assert len(scaffold.prefix) == data.PREFIX_LENGTH == 99
    assert scaffold.prefix.startswith(data.START_TOKEN)
    assert scaffold.prefix.endswith("YYC" + data.ANCHOR_LEFT)
    assert scaffold.heavy[data.CORE_START:data.CORE_START + 10] == data.WT_CORE
    assert scaffold.heavy[data.CORE_START + 10] == data.ANCHOR_RIGHT
    assert scaffold.with_core("ACDEFGHIKL") == scaffold.heavy.replace(data.WT_CORE, "ACDEFGHIKL")


def test_scaffold_rejects_a_moved_core(raw_root, tmp_path):
    shifted = "A" + SYNTHETIC_HEAVY[:-1]
    write_submitted(raw_root / data.SUBMITTED_CSV,
                    [("IgG_1", shifted, SYNTHETIC_LIGHT, "Positive Control", data.WT_CORE)])
    with pytest.raises(ValueError):
        data.load_scaffold(raw_root)


# ---------------------------------------------------------------------------
# T10/T11  split integrity is re-derived, not trusted
# ---------------------------------------------------------------------------

def test_load_split_rederives_label_and_distance(raw_root):
    frame = data.load_split(raw_root, "train", expect_counts=False)
    assert list(frame.columns) == ["seq", "class", "label", "edit_distance"]
    assert frame.seq.str.len().unique().tolist() == [data.CORE_LENGTH]
    assert set(frame["class"]) <= set(data.CLASS_ORDER)


@pytest.mark.parametrize("corruption", ["label", "distance", "residue", "length", "duplicate"])
def test_load_split_rejects_corruption(raw_root, corruption):
    path = data.split_path(raw_root, "train")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    core, name, label, distance = lines[1].split(",")
    if corruption == "label":
        lines[1] = f"{core},{name},{1 - int(label)},{distance}"
    elif corruption == "distance":
        lines[1] = f"{core},{name},{label},{int(distance) + 1}"
    elif corruption == "residue":
        lines[1] = f"{'X' + core[1:]},{name},{label},{distance}"
    elif corruption == "length":
        lines[1] = f"{core + 'A'},{name},{label},{distance}"
    else:
        lines.append(lines[1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_split(raw_root, "train", expect_counts=False)


def test_expected_counts_gate_production_loads(raw_root):
    """The synthetic fixture is tiny; the production path demands the audited counts."""
    with pytest.raises(ValueError, match="rows, expected"):
        data.load_split(raw_root, "train", expect_counts=True)


# ---------------------------------------------------------------------------
# T12  reserved labels and assay outcomes need a verified unlock
# ---------------------------------------------------------------------------

def test_test_split_labels_are_guarded(raw_root):
    with pytest.raises(data.Her2GuardError):
        data.load_split(raw_root, "test", expect_counts=False)
    with pytest.raises(data.Her2GuardError):
        data.load_split(raw_root, "test", unlock=True, expect_counts=False)


def test_test_sequences_never_materializes_a_label(raw_root):
    sequences = data.test_sequences(raw_root)
    assert sequences.name == "seq"
    assert not hasattr(sequences, "class")
    assert all(len(core) == data.CORE_LENGTH for core in sequences)


def test_workbook_outcome_column_is_guarded(raw_root):
    with pytest.raises(data.Her2GuardError):
        data.workbook_table(raw_root, include_outcome=True)
    metadata = data.workbook_table(raw_root)
    assert "kd_molar" not in metadata["table"].columns
    assert set(metadata["table"].columns) == {"workbook_row", "name", "heavy", "light",
                                              "design_label"}


def make_freeze(tmp_path, checkpoint, *, config_sha256="abc", stage=data.SELECTION_STAGE_FINAL,
                selected=None, name="selection_frozen.json"):
    payload = {"schema_version": data.SELECTION_SCHEMA, "stage": stage,
               "config_sha256": config_sha256,
               "selected": {"policy_a": {"checkpoint": checkpoint.name,
                                         "sha256": prov.sha256_file(checkpoint)}}
               if selected is None else selected}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_selection_freeze_mints_the_only_unlock(raw_root, tmp_path):
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint)
    document, unlock = data.read_selection_freeze(freeze, root=tmp_path, expected_config_sha256="abc")
    assert isinstance(unlock, data.SelectionUnlock)
    assert unlock.verified == ("policy_a",)
    assert unlock.stage == data.SELECTION_STAGE_FINAL
    frame = data.load_split(raw_root, "test", unlock=unlock, expect_counts=False)
    assert len(frame) > 0
    assert document["schema_version"] == data.SELECTION_SCHEMA


def test_selection_freeze_rejects_an_empty_selection(tmp_path):
    """An empty selection verified nothing, so it cannot be evidence of anything."""
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint, selected={})
    with pytest.raises(ValueError, match="names no artifacts"):
        data.read_selection_freeze(freeze, root=tmp_path)


def test_initial_selection_cannot_unlock_reserved_labels(raw_root, tmp_path):
    """The intermediate freeze starts post-training; it is not a test-label key.

    The earlier version of this test stopped at the reader: it checked that the
    default reader rejects a base-stage freeze, then obtained a legitimate
    base-stage token and asserted nothing about it. That token reached the
    downstream loaders, which is the thing that mattered. So the token is now
    carried to the actual access boundaries -- the reserved test labels and the
    workbook outcome column -- and both must refuse it.
    """
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint, stage=data.SELECTION_STAGE_BASE)
    with pytest.raises(ValueError, match="stage"):
        data.read_selection_freeze(freeze, root=tmp_path)
    document, unlock = data.read_selection_freeze(
        freeze, root=tmp_path, expected_stage=data.SELECTION_STAGE_BASE)
    assert unlock.stage == data.SELECTION_STAGE_BASE
    assert document["stage"] == data.SELECTION_STAGE_BASE

    with pytest.raises(data.Her2GuardError, match="final"):
        data.load_split(raw_root, "test", unlock=unlock, expect_counts=False)
    with pytest.raises(data.Her2GuardError, match="final"):
        data.workbook_table(raw_root, unlock=unlock, include_outcome=True)
    with pytest.raises(data.Her2GuardError, match="final"):
        data.assay_cohort(raw_root, data.load_scaffold(raw_root), set(), unlock=unlock,
                          include_outcome=True)
    # The same freeze marked `final` is the only thing that opens them.
    final = make_freeze(tmp_path, checkpoint, stage=data.SELECTION_STAGE_FINAL,
                        name="selection_final.json")
    _, opened = data.read_selection_freeze(final, root=tmp_path)
    assert len(data.load_split(raw_root, "test", unlock=opened, expect_counts=False)) > 0
    assert "kd_molar" in data.workbook_table(raw_root, unlock=opened,
                                             include_outcome=True)["table"].columns


def test_selection_freeze_requires_the_exact_expected_key_set(tmp_path):
    """A freeze that quietly drops an arm is rejected, not silently evaluated."""
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="key set mismatch"):
        data.read_selection_freeze(freeze, root=tmp_path,
                                   expected_selected=["policy_a", "policy_b"])
    _, unlock = data.read_selection_freeze(freeze, root=tmp_path, expected_selected=["policy_a"])
    assert unlock.verified == ("policy_a",)


def test_selection_freeze_rejects_stale_code_digests(tmp_path):
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="different scientific code"):
        data.read_selection_freeze(freeze, root=tmp_path,
                                   expected_code_digests={"scripts/train_her2.py": "deadbeef"})


def test_selection_freeze_rejects_a_missing_stage(tmp_path):
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    path = tmp_path / "selection_frozen.json"
    path.write_text(json.dumps({"schema_version": data.SELECTION_SCHEMA, "config_sha256": "abc",
                                "selected": {"policy_a": {"checkpoint": checkpoint.name,
                                                          "sha256": prov.sha256_file(checkpoint)}}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="no known stage marker"):
        data.read_selection_freeze(path, root=tmp_path)


def test_selection_freeze_rejects_different_pinned_sources(tmp_path):
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint)
    document = json.loads(freeze.read_text(encoding="utf-8"))
    document["source_digests"] = {"train.csv": "original"}
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="different pinned sources"):
        data.read_selection_freeze(freeze, root=tmp_path,
                                   expected_source_digests={"train.csv": "replacement"})
    data.read_selection_freeze(freeze, root=tmp_path,
                               expected_source_digests=document["source_digests"])


def test_selection_freeze_rejects_a_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint)
    checkpoint.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="changed on disk"):
        data.read_selection_freeze(freeze, root=tmp_path)


def test_selection_freeze_rejects_a_different_config(tmp_path):
    checkpoint = tmp_path / "epoch_3.pt"
    checkpoint.write_bytes(b"weights")
    freeze = make_freeze(tmp_path, checkpoint, config_sha256="abc")
    with pytest.raises(ValueError, match="different run configuration"):
        data.read_selection_freeze(freeze, root=tmp_path, expected_config_sha256="def")


# ---------------------------------------------------------------------------
# T13  neighbour search against brute force, including ties
# ---------------------------------------------------------------------------

def brute_force(query, train, labels, max_distance):
    distances = (query[:, None, :] != train[None, :, :]).sum(axis=2)
    out = []
    for row in distances:
        best = row.min()
        if best > max_distance:
            out.append((-1, 0, None))
            continue
        tied = labels[row == best]
        out.append((int(best), int(tied.size), float(tied.mean())))
    return out


def test_nearest_training_matches_brute_force_with_ties():
    rng = np.random.default_rng(7)
    train = rng.integers(0, 20, size=(400, data.CORE_LENGTH))
    query = rng.integers(0, 20, size=(120, data.CORE_LENGTH))
    # query[0] is train[0] with one site changed, so train[0] is ALREADY a
    # Hamming-1 neighbour. Planting train[1] and train[2] at distance 1 as well
    # gives three tied nearest neighbours, not two -- the earlier expectation of
    # two forgot the row the query was derived from. Brute force below is the
    # arbiter; these explicit assertions just pin the tie-averaging case.
    query[0] = train[0]
    query[0][3] = (train[0][3] + 1) % 20
    train[1] = query[0].copy()
    train[1][7] = (query[0][7] + 1) % 20
    train[2] = query[0].copy()
    train[2][5] = (query[0][5] + 1) % 20
    query[1] = train[3]
    labels = rng.integers(0, 2, size=400).astype(np.float64)
    labels[0], labels[1], labels[2] = 1.0, 0.0, 1.0
    lookup = data.nearest_training_labels(query, train, labels, max_distance=2)
    expected = brute_force(query, train, labels, 2)
    for position, (distance, count, mean) in enumerate(expected):
        assert int(lookup.distance[position]) == distance, position
        assert int(lookup.neighbour_count[position]) == count, position
        if mean is None:
            assert np.isnan(lookup.label_mean[position])
        else:
            assert lookup.label_mean[position] == pytest.approx(mean)
    assert int(lookup.distance[0]) == 1
    assert int(lookup.neighbour_count[0]) == 3
    assert lookup.label_mean[0] == pytest.approx(2 / 3)
    assert int(lookup.distance[1]) == 0


def test_nearest_training_falls_back_to_the_prior_and_labels_the_stratum():
    train = np.zeros((1, data.CORE_LENGTH), dtype=np.int64)
    query = np.full((1, data.CORE_LENGTH), 5, dtype=np.int64)
    lookup = data.nearest_training_labels(query, train, np.array([1.0]), max_distance=2,
                                          prior=0.25)
    assert int(lookup.distance[0]) == -1
    assert int(lookup.neighbour_count[0]) == 0
    assert lookup.score[0] == pytest.approx(0.25)
    assert lookup.strata().tolist() == [">=3"]


def test_encode_cores_rejects_mismatched_lengths_that_would_cancel_out():
    """A 9-mer and an 11-mer used to reshape into two cores that were never input."""
    with pytest.raises(ValueError, match="length 9"):
        data.encode_cores(["A" * 9, "C" * 11])
    with pytest.raises(ValueError, match="length 11"):
        data.encode_cores(["A" * 11])


@pytest.mark.parametrize("bad,match", [
    ([None], "not a string"),
    ([b"ACDEFGHIKL"], "not a string"),
    (["ACDEFGHIK"], "length 9"),
    (["ACDEFGHIKLM"], "length 11"),
    (["ACDEFGHIKX"], "Non-canonical"),
    (["ACDEFGHIKÅ"], "non-ASCII"),
    ([], "No cores"),
])
def test_encode_cores_validates_every_row(bad, match):
    with pytest.raises(ValueError, match=match):
        data.encode_cores(bad)


def test_encode_cores_reports_the_offending_row():
    with pytest.raises(ValueError, match=r"row\(s\) \[2\]"):
        data.encode_cores(["ACDEFGHIKL", "ACDEFGHIKL", "ACDEFGHIKB"])


def test_no_binding_alias_is_classified():
    """The advertised 'NO BINDING' alias matched nothing before the tokens were normalized."""
    for text in ("NO BINDING", "no binding", " No  Binding ", "N.B.", "nb"):
        assert data.classify_outcome(text)[0] == data.OUTCOME_NON_BINDING, text


def test_packed_codes_are_lossless_and_mask_exactly():
    index = data.encode_cores(["ACDEFGHIKL", "ACDEFGHIKM"])
    assert data.pack_codes(index)[0] != data.pack_codes(index)[1]
    assert data.pack_codes(index, (9,))[0] == data.pack_codes(index, (9,))[1]
    assert data.decode_cores(index) == ["ACDEFGHIKL", "ACDEFGHIKM"]


def test_mean_pairwise_hamming_matches_the_explicit_sum():
    rng = np.random.default_rng(3)
    index = rng.integers(0, 20, size=(40, data.CORE_LENGTH))
    explicit = [(index[i] != index[j]).sum() for i in range(40) for j in range(i + 1, 40)]
    assert data.mean_pairwise_hamming(index) == pytest.approx(float(np.mean(explicit)))


# ---------------------------------------------------------------------------
# workbook parsing and outcome classification
# ---------------------------------------------------------------------------

def test_workbook_parser_reads_shared_strings_and_numbers(raw_root):
    sheet, rows = data.read_workbook(raw_root / data.WORKBOOK_XLSX, ["N", "AD"])
    assert sheet == "Sheet1"
    assert rows[data.WORKBOOK_HEADER_ROW]["N"] == "IgG"
    assert rows[17]["N"] == "IgG_1"
    assert "P" not in rows[17], "columns that were not requested must not be materialized"


def test_workbook_parser_refuses_an_unsupported_cell_type(tmp_path):
    path = tmp_path / "odd.xlsx"
    write_xlsx(path, {1: {"A": "x"}})
    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/worksheets/sheet1.xml"] = (
        f'<worksheet xmlns="{MAIN_NS}"><sheetData><row r="1">'
        '<c r="A1" t="e"><v>#REF!</v></c></row></sheetData></worksheet>').encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="Unsupported cell type"):
        data.read_workbook(path, ["A"])


@pytest.mark.parametrize("value,expected", [
    ("1.2e-9", data.OUTCOME_QUANTITATIVE), (2.5e-8, data.OUTCOME_QUANTITATIVE),
    ("N.B.", data.OUTCOME_NON_BINDING), (" n.b. ", data.OUTCOME_NON_BINDING),
    ("I.C.", data.OUTCOME_BINDING_UNQUANTIFIED), ("", data.OUTCOME_MISSING),
    ("   ", data.OUTCOME_MISSING), ("weak", data.OUTCOME_UNSUPPORTED),
    ("-1e-9", data.OUTCOME_UNSUPPORTED),
])
def test_outcome_classification_never_guesses(value, expected):
    outcome, kd = data.classify_outcome(value)
    assert outcome == expected
    assert (kd is not None) == (expected == data.OUTCOME_QUANTITATIVE)
    if expected == data.OUTCOME_BINDING_UNQUANTIFIED:
        assert kd is None, "I.C. must never carry a fabricated KD"


def test_assay_cohort_excludes_library_overlap_and_unsupported_scaffolds(raw_root):
    scaffold = data.load_scaffold(raw_root)
    train = data.load_split(raw_root, "train", expect_counts=False)
    cohort = data.assay_cohort(raw_root, scaffold, set(train.seq))
    counts = cohort["counts"]
    assert counts["unsupported_scaffold_rows"] == 1
    assert counts["library_overlap_rows"] == 1
    assert counts["library_overlap_cores"] == [train.seq.iloc[0]]
    assert counts["primary_rows"] == counts["design_method_rows"] - counts["library_overlap_rows"]
    assert set(cohort["primary"].design_label) <= set(data.DESIGN_METHODS)
    assert cohort["primary"].core.is_unique
    assert set(cohort["controls"].design_label) == {"Positive Control"}
    assert "kd_molar" not in cohort["primary"].columns


def test_assay_cohort_deduplicates_repeated_cores(raw_root):
    """A control replicated across plates must not become two independent rows."""
    scaffold = data.load_scaffold(raw_root)
    designs = []
    for line in (raw_root / data.SUBMITTED_CSV).read_text(encoding="utf-8").strip().split("\n")[1:]:
        fields = line.split(",")
        designs.append((fields[0], fields[2], fields[3], fields[6], fields[7]))
    repeat = [d for d in designs if d[3] in data.DESIGN_METHODS][0]
    designs.append(("IgG_repeat", repeat[1], repeat[2], repeat[3], repeat[4]))
    write_submitted(raw_root / data.SUBMITTED_CSV, designs)
    sheet, rows = data.read_workbook(raw_root / data.WORKBOOK_XLSX, ["N", "P", "W", "AD"])
    rows[max(rows) + 1] = {"N": "IgG_repeat", "P": repeat[1], "W": repeat[2], "AD": repeat[3]}
    write_xlsx(raw_root / data.WORKBOOK_XLSX, rows)
    cohort = data.assay_cohort(raw_root, scaffold, set())
    assert cohort["counts"]["primary_duplicate_rows"] == 1
    assert cohort["primary"].core.is_unique


# ---------------------------------------------------------------------------
# T14/T15  committed manifests, and the cross-check against the retrieval record
# ---------------------------------------------------------------------------

HER2_MANIFESTS = ("buzz_her2_affinity", "absci_denovo_her2", "piggen_backbone")


@pytest.fixture
def repository_root(project_root: Path) -> Path:
    return project_root.parents[1]


def test_committed_her2_manifests_validate(repository_root):
    for stem in HER2_MANIFESTS:
        manifest = prov.load_manifest_document(
            repository_root / "specs" / "benchmarks" / f"{stem}.json").validated()
        assert manifest.files
        assert all(entry.relative_path.count("..") == 0 for entry in manifest.files)


def test_committed_manifests_agree_with_the_retrieval_record(repository_root):
    """Tracked hashes must equal the untracked record's hashes, byte for byte."""
    root = repository_root / data.RAW_ROOT
    if not (root / "source_manifest.json").is_file():
        pytest.skip("raw root is not present on this machine")
    pinned = {}
    for entry in data.read_downloaded_manifest(root)["files"]:
        pinned[Path(entry["local_path"]).relative_to(data.RAW_ROOT).as_posix()] = entry["sha256"]
    for entry in data.read_piggen_manifest(root)["files"]:
        pinned[Path(entry["path"]).relative_to(data.RAW_ROOT).as_posix()] = entry["sha256"]
    for stem in HER2_MANIFESTS:
        manifest = prov.load_manifest_document(
            repository_root / "specs" / "benchmarks" / f"{stem}.json").validated()
        for entry in manifest.files:
            assert entry.relative_path in pinned, f"{stem}: {entry.relative_path} is unpinned"
            assert entry.sha256 == pinned[entry.relative_path], f"{stem}: {entry.relative_path}"
        assert prov.verify_manifest_files(manifest, root) == ()

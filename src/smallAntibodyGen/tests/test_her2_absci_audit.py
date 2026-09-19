"""The Absci audit: KD cells stay what they are, and nothing gets cropped into support.

Most of these run on small synthetic frames so the counts are checkable by eye.
Two tests read the pinned release itself and are skipped when the git-ignored raw
root is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_absci_audit as absci

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/her2_functional_20260918"
IN_SUPPORT = "SR" + "ACDEFGHIKL" + "Y"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# KD cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell,expected,value", [
    ("0.94", absci.KD_FINITE, 0.94),
    (" 1.2 ", absci.KD_FINITE, 1.2),
    ("", absci.KD_BLANK, None),
    ("   ", absci.KD_BLANK, None),
    ("NB", absci.KD_NON_BINDING, None),
    ("N.B.", absci.KD_NON_BINDING, None),
    ("no binding", absci.KD_NON_BINDING, None),
    ("I.C.", absci.KD_BINDING_UNQUANTIFIED, None),
    ("N/A", absci.KD_NOT_AVAILABLE, None),
    ("<0.1", absci.KD_CENSORED, None),
    (">1000", absci.KD_CENSORED, None),
    ("≤0.5", absci.KD_CENSORED, None),
    ("6.2 ± 4.9", absci.KD_UNSUPPORTED, None),
    ("abc", absci.KD_UNSUPPORTED, None),
    ("-1", absci.KD_UNSUPPORTED, None),
    ("0", absci.KD_UNSUPPORTED, None),
])
def test_every_cell_class_is_named_and_none_becomes_a_number(cell, expected, value):
    kind, number, _ = absci.classify_kd(cell)
    assert kind == expected
    assert number == value
    if expected != absci.KD_FINITE:
        assert number is None, "a censored, absent or unquantified cell is not a KD"


def test_a_zero_width_space_is_recorded_rather_than_swallowed():
    kind, number, contaminated = absci.classify_kd("1.9​")
    assert kind == absci.KD_FINITE and number == pytest.approx(1.9)
    assert contaminated is True
    blank_kind, _, blank_contaminated = absci.classify_kd("​")
    assert blank_kind == absci.KD_BLANK and blank_contaminated is True


def test_a_column_report_counts_classes_and_the_contamination():
    report, values = absci.classify_column(["0.5", "", "NB", "I.C.", "1.0​", "<3"])
    assert report["classes"] == {absci.KD_BINDING_UNQUANTIFIED: 1, absci.KD_BLANK: 1,
                                 absci.KD_CENSORED: 1, absci.KD_FINITE: 2,
                                 absci.KD_NON_BINDING: 1}
    assert report["finite_positive"] == 2
    assert report["zero_width_space_cells"] == 1
    assert values == [0.5, None, None, None, 1.0, None]


# ---------------------------------------------------------------------------
# support, and the cropping that does not happen
# ---------------------------------------------------------------------------

def test_only_an_already_in_support_hcdr3_yields_a_core():
    assert absci.supported_core(IN_SUPPORT) == "ACDEFGHIKL"
    assert absci.supported_core("SRACDEFGHIKY") is None, "12-mer: cropping is not performed"
    assert absci.supported_core("SRACDEFGHIKLMNY") is None, "15-mer"
    assert absci.supported_core("ARACDEFGHIKLY") is None, "wrong left anchor"
    assert absci.supported_core("SRACDEFGHIKLW") is None, "wrong right anchor"
    assert absci.supported_core("SRACDEFGHIKBY") is None, "non-canonical residue"
    assert absci.supported_core("") is None


def test_the_wild_type_hcdr3_is_in_support_by_construction():
    assert absci.supported_core(absci.TRASTUZUMAB_HCDR3) == "WGGDGFYAMD"
    assert absci.SUPPORTED_HCDR3_LENGTH == 13


def test_a_file_without_framework_columns_reports_an_upper_bound(tmp_path):
    path = write_csv(tmp_path / "zero-shot.csv", ["HCDR3", "KD (nM)"],
                     [[IN_SUPPORT, "0.9"], ["ARYYYGFYYFDY", "1.2"],
                      ["TRYFFNGWYYFDV", "1.7"]])
    report = absci.audit_file(path, name="zero_shot")
    assert report["support"]["cdr_context_verified"] is False
    assert report["support"]["upper_bound"] is True
    assert "UPPER BOUND" in report["support"]["reason"]
    assert report["support"]["compatible_rows"] == 1
    assert report["hcdr3"]["length_histogram"] == {"12": 1, "13": 2}
    assert report["kd"]["finite_positive"] == 3
    assert report["hcdr3"]["cdr_context"]["unique"] is None, "not computable, so not zero"


def test_a_binders_only_file_cannot_establish_discrimination(tmp_path):
    path = write_csv(tmp_path / "zero-shot.csv", ["HCDR3", "KD (nM)"], [[IN_SUPPORT, "0.9"]])
    report = absci.audit_file(path, name="zero_shot")
    assert report["binder_flag"]["discrimination_possible"] is False
    # No binder column: the row is UNKNOWN, not a labelled positive and not a negative.
    assert report["binder_flag"]["true"] is None and report["binder_flag"]["false"] is None
    assert report["binder_flag"]["unknown"] == 1
    assert report["binder_flag"]["finite_kd_rows"] == 1
    # CX-32: the file may be described as the published binder set it is. What it
    # may not do is yield a ranking statistic computed over labels it never carried.
    note = report["binder_flag"]["note"]
    assert "published as a binder set" in note
    assert "no discrimination statistic" in note
    assert "average precision is 1" not in note
    # A finite KD is a measurement with error, not an exact number.
    assert "not as an exact or noiseless value" in report["kd"]["precision_note"]


def test_framework_columns_turn_the_upper_bound_into_a_verified_cdr_context(tmp_path):
    other = "SR" + "ACDEFGHIKM" + "Y"
    path = write_csv(
        tmp_path / "controls.csv", ["HCDR1", "HCDR2", "HCDR3", "KD (nM)", "Binder"],
        [[absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, IN_SUPPORT, "0.56", "True"],
         [absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, other, "", "False"],
         ["GFNIKDTA", absci.TRASTUZUMAB_HCDR2, "SR" + "ACDEFGHIKN" + "Y", "0.7", "True"],
         [absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, "ARYYYGFYYFDY", "0.8", "True"]])
    report = absci.audit_file(path, name="controls", binder_column="Binder")
    assert report["support"]["cdr_context_verified"] is True
    assert report["support"]["upper_bound"] is False
    assert report["support"]["length_and_anchor_compatible_rows"] == 3
    assert report["support"]["fixed_framework_rows"] == 3
    # In support AND on the fixed CDR context: the third row fails H1, the fourth fails length.
    assert report["support"]["compatible_rows"] == 2
    assert report["support"]["compatible_kd_classes"] == {absci.KD_BLANK: 1, absci.KD_FINITE: 1}
    assert report["binder_flag"]["discrimination_possible"] is True
    assert report["binder_flag"]["true"] == 3 and report["binder_flag"]["false"] == 1
    # H1 and H2 matching is CDR-context compatibility. These files carry no VH/VL
    # sequence, so nothing in them proves the rest of the scaffold.
    assert report["support"]["scaffold_proven"] is False
    assert "no VH/VL sequence" in report["support"]["scaffold_note"]


def test_the_compatible_subset_counts_the_explicit_binder_column_not_the_blank_kds(tmp_path):
    """A blank KD is a missing number. The only thing that says "did not bind" is the flag."""
    path = write_csv(
        tmp_path / "controls.csv", ["HCDR1", "HCDR2", "HCDR3", "KD (nM)", "Binder"],
        [[absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, IN_SUPPORT, "0.56", "True"],
         [absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, "SR" + "ACDEFGHIKM" + "Y", "",
          "False"],
         [absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, "SR" + "ACDEFGHIKN" + "Y", "", ""]])
    report = absci.audit_file(path, name="controls", binder_column="Binder")
    assert report["support"]["compatible_rows"] == 3
    assert report["support"]["compatible_kd_classes"] == {absci.KD_BLANK: 2, absci.KD_FINITE: 1}
    # Two blank KDs: one explicitly not a binder, one with nothing said about it.
    assert report["support"]["compatible_binder_flags"] == {"true": 1, "false": 1, "unknown": 1}
    assert "not a negative" in report["support"]["binder_flag_note"]
    assert absci.classify_binder("") == "unknown"
    assert absci.classify_binder("TRUE") == "true" and absci.classify_binder(" false ") == "false"


def test_a_repeated_hcdr3_is_not_a_repeated_antibody(tmp_path):
    """Same HCDR3 on a different HCDR1: one duplicate by HCDR3, none by CDR context."""
    path = write_csv(
        tmp_path / "controls.csv", ["HCDR1", "HCDR2", "HCDR3", "KD (nM)"],
        [[absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, IN_SUPPORT, "0.5"],
         ["GFNIKDTA", absci.TRASTUZUMAB_HCDR2, IN_SUPPORT, "0.6"]])
    report = absci.audit_file(path, name="controls")
    assert report["hcdr3"]["unique"] == 1 and report["hcdr3"]["duplicate_rows"] == 1
    assert report["hcdr3"]["cdr_context"]["unique"] == 2
    assert report["hcdr3"]["cdr_context"]["duplicate_rows"] == 0
    assert "not a repeated antibody" in report["hcdr3"]["duplicate_note"]


def test_finite_and_nonbinding_compatible_rows_are_counted_apart_from_the_overlap(tmp_path):
    """38 compatible is not 38 measurements, and 27 finite is not 26 independent ones."""
    overlapping = "SR" + "ACDEFGHIKM" + "Y"
    path = write_csv(
        tmp_path / "controls.csv", ["HCDR1", "HCDR2", "HCDR3", "KD (nM)", "Binder"],
        [[absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, IN_SUPPORT, "0.56", "True"],
         [absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, overlapping, "0.7", "True"],
         [absci.TRASTUZUMAB_HCDR1, absci.TRASTUZUMAB_HCDR2, "SR" + "ACDEFGHIKN" + "Y", "NB",
          "False"]])
    report = absci.audit_file(path, name="controls", binder_column="Binder",
                              library_cores={"train": {"ACDEFGHIKM"}, "val": set(), "test": set()})
    overlap = report["library_overlap"]
    assert overlap["train"]["unique_cores_in_split"] == 1
    assert overlap["finite_kd_compatible_cores"] == 2
    assert overlap["independent_finite_kd_cores"] == ["ACDEFGHIKL"]
    assert "ACDEFGHIKN" in overlap["independent_of_every_split"], "non-binding, but independent"
    assert report["support"]["compatible_kd_classes"] == {absci.KD_FINITE: 2,
                                                          absci.KD_NON_BINDING: 1}
    # The independent subset carries its own class and flag counts, never a pooled total.
    assert overlap["independent_rows"] == 2
    assert overlap["independent_kd_classes"] == {absci.KD_FINITE: 1, absci.KD_NON_BINDING: 1}
    assert overlap["independent_binder_flags"] == {"true": 1, "false": 1, "unknown": 0}
    assert "never pooled" in overlap["counting_note"]


def test_duplicate_hcdr3_rows_are_counted_not_silently_deduped(tmp_path):
    path = write_csv(tmp_path / "controls.csv", ["HCDR3", "KD (nM)"],
                     [[IN_SUPPORT, "0.5"], [IN_SUPPORT, "0.6"], ["ARYYYGFYYFDY", "1.0"]])
    report = absci.audit_file(path, name="controls")
    assert report["rows"] == 3
    assert report["hcdr3"]["unique"] == 2
    assert report["hcdr3"]["duplicate_rows"] == 1


def test_counts_are_not_additive_across_files(tmp_path):
    binders = write_csv(tmp_path / "binders.csv", ["HCDR3", "KD (nM)"],
                        [[IN_SUPPORT, "0.5"], ["ARYYYGFYYFDY", "1.0"]])
    controls = write_csv(tmp_path / "controls.csv", ["HCDR3", "KD (nM)"],
                         [[IN_SUPPORT, "0.5"], ["TRYFFNGWYYFDV", "2.0"],
                          ["ARYYYGFYYFDY", "1.0"]])
    overlap = absci.cross_file_overlap({"binders": {"path": str(binders)},
                                        "controls": {"path": str(controls)}})
    assert overlap["additive"] is False
    assert overlap["pairwise"]["binders|controls"]["shared_unique_hcdr3"] == 2
    assert overlap["pairwise"]["binders|controls"]["left_fraction"] == pytest.approx(1.0)
    assert "count measurements twice" in overlap["note"]


def test_the_module_exposes_no_route_from_an_out_of_support_row_to_a_core(tmp_path):
    """The 152 finite Buzz SPR rows cannot be topped up by pooling incompatible rows."""
    path = write_csv(tmp_path / "zero-shot.csv", ["HCDR3", "KD (nM)"],
                     [["ARYYYGFYYFDY", "1.2"], ["TRYFFNGWYYFDV", "0.9"]])
    report = absci.audit_file(path, name="zero_shot")
    assert report["compatible_cores"] == []
    assert report["support"]["cropping"].startswith("not performed")


# ---------------------------------------------------------------------------
# the pinned release itself
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (RAW_ROOT / "absci/zero-shot-binders.csv").is_file(),
                    reason="the pinned Absci files live under the git-ignored raw root")
def test_the_pinned_files_match_the_counts_the_tracked_manifest_records():
    """The manifest's recorded audit is re-derived from the bytes, not restated."""
    manifest = json.loads((ROOT / "specs/benchmarks/absci_denovo_her2.json").read_text(
        encoding="utf-8"))
    zero_shot = absci.audit_file(RAW_ROOT / absci.FILES["zero_shot_binders"],
                                 name="zero_shot_binders")
    controls = absci.audit_file(RAW_ROOT / absci.FILES["spr_controls"], name="spr_controls",
                                binder_column="Binder")
    assert zero_shot["hcdr3"]["unique"] == 422
    assert zero_shot["kd"]["finite_positive"] == zero_shot["rows"] == 422
    assert zero_shot["support"]["compatible_rows"] == 4
    assert zero_shot["support"]["upper_bound"] is True
    assert controls["rows"] == 1855 and controls["hcdr3"]["unique"] == 1829
    assert controls["binder_flag"]["true"] == 758 and controls["binder_flag"]["false"] == 1097
    assert controls["kd"]["classes"][absci.KD_FINITE] == 758
    assert controls["kd"]["classes"][absci.KD_BLANK] == 1097
    # 26 repeated HCDR3s, zero repeated CDR contexts: those are different designs.
    assert controls["hcdr3"]["duplicate_rows"] == 26
    assert controls["hcdr3"]["cdr_context"]["duplicate_rows"] == 0
    # The 38 CDR-context-compatible control rows: 27 with a finite KD, 11 explicitly
    # flagged as non-binders. The 11 are negatives because the column says so.
    assert controls["support"]["compatible_rows"] == 38
    assert controls["support"]["compatible_kd_classes"][absci.KD_FINITE] == 27
    flags = controls["support"]["compatible_binder_flags"]
    assert flags["false"] == 11
    assert flags["true"] + flags["false"] + flags["unknown"] == 38
    assert "422 unique HCDR3s" in manifest["notes"]
    assert "Absci Corporation (2023)" in manifest["license"]


@pytest.mark.skipif(not (RAW_ROOT / "absci/zero-shot-binders.csv").is_file(),
                    reason="the pinned Absci files live under the git-ignored raw root")
def test_the_audit_verifies_the_pinned_bytes_before_reading_them():
    document = absci.audit(ROOT, RAW_ROOT, library_cores=None)
    assert document["attribution"] == "Absci Corporation (2023)"
    assert document["role"]["primary_cohort"] is False
    assert document["role"]["merged_cohort"] is None
    assert document["cross_file"]["additive"] is False
    assert set(document["files"]) == set(absci.FILES)
    # Every binder HCDR3 also appears in the controls file, which is why the row
    # counts across files are not independent observations.
    shared = document["cross_file"]["pairwise"]["spr_controls|zero_shot_binders"]
    assert shared["shared_unique_hcdr3"] == 422

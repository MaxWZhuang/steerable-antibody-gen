"""I15, I16: assay direction, censoring, identity closure and the H1 column guard.

Tiny synthetic files written into ``tmp_path``. No F: access, no large fixtures. The
CR9114 guard is tested against a synthetic CSV carrying the real column names, because
the failure it prevents is structural and not data-dependent.
"""
from __future__ import annotations

import csv
import io
import json
import math
import zipfile

import pytest

from smallAntibodyGen.experiments import her2_routes_sources as sources
from smallAntibodyGen.experiments.her2_routes_conditions import (
    BIN_HIGH,
    BIN_LOW,
    assign_bin,
    fit_bin_thresholds,
)

HEAVY = "ACDEFGHIKLMNPQRSTVWY"
LIGHT = "YWVTSRQPNMLKIHGFEDCA"


def write_csv(path, rows, *, preamble=()):
    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in preamble:
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_zip(path, name, rows, *, extra_members=()):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, buffer.getvalue())
        for member in extra_members:
            archive.writestr(member, "resource fork\n")
    return path


# --- I16: direction and censoring -----------------------------------------

def test_i16_smaller_kd_is_better_on_every_scale():
    # 1 nM is pKD 9, 1000 nM is pKD 6. Hand-checked, in one place.
    assert sources.pkd_from_log10_nm(0.0) == pytest.approx(9.0)
    assert sources.pkd_from_log10_nm(3.0) == pytest.approx(6.0)
    assert sources.pkd_from_molar(1e-9) == pytest.approx(9.0)
    assert sources.pkd_from_log10_nm(0.0) > sources.pkd_from_log10_nm(1.0)
    with pytest.raises(ValueError, match="not a measurement"):
        sources.pkd_from_molar(0.0)


def test_i16_adams_censors_the_tight_tail_on_the_right(tmp_path):
    rows = [
        {"Kd (Tite-Seq) [M]": "1e-7", "heavy": HEAVY, "light": LIGHT, "format": "Fab",
         "fitness": "0.5"},
        {"Kd (Tite-Seq) [M]": "1e-11", "heavy": "W" + HEAVY[1:], "light": LIGHT,
         "format": "Fab", "fitness": "0.9"},
        {"Kd (Tite-Seq) [M]": "1e-3", "heavy": "V" + HEAVY[1:], "light": LIGHT,
         "format": "Fab", "fitness": "0.1"},
        {"Kd (Tite-Seq) [M]": "", "heavy": "Y" + HEAVY[1:], "light": LIGHT, "format": "Fab",
         "fitness": ""},
    ]
    block = sources.load_adams_affinity(write_csv(tmp_path / "adams.csv", rows))
    by_censoring = {record.censoring: record for record in block["observations"]}
    assert set(by_censoring) == {"none", "right", "left"}
    # 1e-11 M is TIGHTER than the window, so it is right-censored in pKD.
    right = by_censoring["right"]
    assert right.interval[0] == pytest.approx(9.5)
    assert right.interval[1] is None
    assert right.value is None
    left = by_censoring["left"]
    assert left.interval[0] is None
    assert left.interval[1] == pytest.approx(5.0)
    assert by_censoring["none"].value == pytest.approx(7.0)
    # A blank cell is a missing number and is counted, not made a negative.
    assert block["missing_affinity"] == 1
    assert "scFv, not Fab" in right.qualification["format_qualification"]


def test_i16_no_universal_thousand_limit(tmp_path):
    rows = [{"heavy": HEAVY, "light": LIGHT, "KD": "2490", "fitness": "0.1"},
            {"heavy": "W" + HEAVY[1:], "light": LIGHT, "KD": "5", "fitness": "0.9"}]
    block = sources.load_petersen_flab(write_csv(tmp_path / "petersen.csv", rows))
    assert block["values_above_1000_nm"] == 1
    # 2490 nM is enrolled as a measurement, not censored at 1000.
    values = sorted(record.value for record in block["observations"])
    assert values[0] == pytest.approx(9.0 - 3.3961, abs=1e-3)
    assert all(record.censoring == "none" for record in block["observations"])


#: The synthetic CDR annotations. Each one occurs exactly once in its chain, which is
#: what ``locate_cdrs`` requires: a CDR boundary is an annotation, and a string that
#: matches twice does not identify a span.
LI_CDRS = {"CDRH1": "CDE", "CDRH2": "HIK", "CDRH3": "RST",
           "CDRL1": "WVT", "CDRL2": "QPN", "CDRL3": "IHG"}


def _li_row(poi, replicate, affinity, *, heavy=HEAVY, light=LIGHT, target="S1", assay="1",
            annotate=True):
    row = {name: "" for name in sources.LI_HEADER}
    row.update(POI=poi, Target=target, Assay=assay, Replicate=replicate,
               Pred_affinity=affinity, HC=heavy, LC=light)
    if annotate:
        for name, text in LI_CDRS.items():
            chain = heavy if name.startswith("CDRH") else light
            # The annotation must be a span of THIS row's own chain, so a variant whose
            # substitution lands inside a CDR carries the mutated span.
            reference = HEAVY if name.startswith("CDRH") else LIGHT
            start = reference.index(text)
            row[name] = chain[start:start + len(text)]
    return row


def test_i16_replicates_are_kept_as_fields_not_as_extra_constructs(tmp_path):
    rows = [_li_row("AAYL49_a", "1", "1.0"), _li_row("AAYL49_a", "2", "1.2")]
    block = sources.load_li(write_csv(tmp_path / "li1.csv", rows), source="li2023_s1_a")
    # Two replicate ROWS of one construct: ONE aggregated observation carrying both
    # values. The replicate structure survives as a field; what does not survive is
    # a second Python object per replicate, which at 1.26 million rows is the shape
    # that does not fit.
    assert len(block["observations"]) == 1
    record = block["observations"][0]
    assert record.qualification["replicate_support"] == 2
    assert [entry["log10_kd_nm"] for entry in record.raw["replicates"]] == pytest.approx([1.0, 1.2])
    assert record.uncertainty is not None
    closure = sources.observation_closure({"li2023_s1_a": block["observations"]})
    assert closure["kept"] == 1
    assert closure["unique_constructs"] == 1
    assert closure["duplicates"] == 0


def test_the_dedup_key_names_the_study_not_the_publication_alias(tmp_path):
    """Engelhart 2022 and the first Li release are one AlphaSeq experiment.

    Two things follow, and both are checked. The canonical study is shared, so the
    two publications are one population for threshold fitting and one weight in the
    mixture. And the de-duplication key is built from that study rather than from
    the source name, so an identical row arriving under a second alias collapses.

    What the key alone cannot do is collapse a curated SUMMARY against the primary
    rows it summarises -- those arrive with different assay and replicate structure
    on purpose. That is handled by an explicit supersession at enrolment, which is
    recorded rather than inferred.
    """
    assert sources.canonical_study("engelhart2022_s1") == sources.canonical_study("li2023_s1_a")
    # The second Li release IS a different study and is not collapsed into it.
    assert sources.canonical_study("li2023_s1_b") != sources.canonical_study("li2023_s1_a")
    # Adams' curated summary and its primary replicate tables are one study.
    assert (sources.canonical_study("adams2017_fluorescein")
            == sources.canonical_study("adams2017_primary"))
    rows = [{"Pred_affinity": "2.0", "heavy": HEAVY, "light": LIGHT, "fitness": "0.3"}]
    block = sources.load_engelhart(write_zip(tmp_path / "e.csv.zip", "e.csv", rows))
    record = block["observations"][0]
    assert record.study == "engelhart2022_alphaseq_s1"
    assert record.observation_identity[1] == record.study
    assert record.source not in record.observation_identity
    # The assay identity carries the study, the family and the target, so an
    # affinity and an expression readout of one study cannot collide.
    assert record.assay_key == "engelhart2022_alphaseq_s1:affinity:SARS-CoV-2 spike HR2 (S2) peptide"


def test_li_families_get_their_own_declared_parent_not_a_modal_one():
    """AAYL49/50 share MIT_14; 51 is MIT_91 and 52 is MIT_95. Four POIs, three parents."""
    assert sources.li_parent_of("AAYL49_00001") == "MIT_14_HL_scFV"
    assert sources.li_parent_of("AAYL50_00001") == "MIT_14_HL_scFV"
    assert sources.li_parent_of("AAYL51_00001") == "MIT_91_LH_scFV"
    assert sources.li_parent_of("AAYL52_00001") == "MIT_95_HL_scFV"
    assert sources.li_parent_of("MIT_14_HL_scFV") == "MIT_14_HL_scFV"
    # An unknown family gets NO parent rather than the modal one.
    assert sources.li_parent_of("SOMETHING_ELSE") is None


def test_li_parents_are_declared_and_the_region_is_the_family_common_one(tmp_path):
    """The parent decides the row's MUTATIONS; the annotations decide its REGION.

    Both are needed and they are different objects. The parent is what a measured
    single-site effect is measured against, so it must be the family's own control
    rather than a modal sequence across families. The region is the family's six
    annotated CDR spans, shared by every member -- a mutation-dependent region gives a
    single mutant a one-site window and a triple mutant a three-site one, so their
    window log probabilities are not comparable and the width leaks the edit count.
    """
    parent_heavy, parent_light = HEAVY, LIGHT
    variant_heavy = HEAVY.replace("CDE", "CWE")      # one substitution inside CDRH1
    rows = [_li_row("MIT_91_LH_scFV", "1", "1.0", heavy=parent_heavy, light=parent_light),
            _li_row("AAYL51_0001", "1", "1.5", heavy=variant_heavy, light=parent_light)]
    block = sources.load_li(write_csv(tmp_path / "li_families.csv", rows),
                            source="li2023_s1_a")
    by_poi = {record.raw["POI"]: record for record in block["observations"]}
    variant = by_poi["AAYL51_0001"]
    assert variant.parent == "MIT_91_LH_scFV"
    assert variant.qualification["parent_control"] == "MIT_91_LH_scFV"
    # One substitution against ITS OWN parent, carried separately from the region.
    assert variant.raw["mutated_sites"] == [["H", HEAVY.index("CDE") + 1]]
    # The region is the six annotated spans: 3 residues each, heavy then light.
    assert len(variant.construct.sites) == 18
    assert variant.construct.sites == by_poi["MIT_91_LH_scFV"].construct.sites, (
        "every member of a scaffold family shares one region, or the window task has a "
        "different support for every row")
    assert "never a modal sequence across families" in variant.qualification["parent_rule"]
    assert "shared by every member" in variant.qualification["region_rule"]
    assert block["inventory"]["scaffold_families"] == 1


def test_a_poi_whose_annotations_do_not_locate_is_counted_not_guessed(tmp_path):
    rows = [_li_row("AAYL49_a", "1", "1.0", annotate=False)]
    block = sources.load_li(write_csv(tmp_path / "li_bad.csv", rows), source="li2023_s1_a")
    assert block["poi_without_annotation"] == 1
    record = block["observations"][0]
    assert record.construct.sites == (), "no region is invented for an unlocatable POI"
    assert "did not locate uniquely" in record.qualification["unresolved"]
    assert block["inventory"]["annotation_failures"]


def test_the_second_release_imports_the_first_releases_parent_control(tmp_path):
    """Li2 publishes no control row, and its two families are Li1's MIT_14."""
    first = [_li_row("MIT_14_HL_scFV", "1", "1.0")]
    block = sources.load_li(write_csv(tmp_path / "li1_parents.csv", first),
                            source="li2023_s1_a")
    exported = sources.li_parent_export(block["table"])
    assert exported == {"MIT_14_HL_scFV": (HEAVY, LIGHT)}
    variant_heavy = HEAVY.replace("CDE", "CWE")
    second = [_li_row("14H_En_HC_1", "1.0", "1.4", heavy=variant_heavy)]
    other = sources.load_li(write_csv(tmp_path / "li2_rows.csv", second),
                            source="li2023_s1_b", parents=exported)
    record = other["observations"][0]
    assert record.parent == "MIT_14_HL_scFV"
    assert record.raw["mutated_sites"] == [["H", HEAVY.index("CDE") + 1]]
    assert other["parent"]["imported_controls"] == ["MIT_14_HL_scFV"]
    assert other["parent"]["resolved"] is True
    # A 1.0-style replicate identifier is replicate one, not "not recorded".
    assert record.raw["replicates"][0]["replicate"] == 1


def test_i16_petersen_interval_qualification_rejects_and_does_not_repair():
    assert sources.qualify_interval(5.0, 4.0, 6.0)[0] == (4.0, 6.0)
    assert sources.qualify_interval(5.0, 0.0, 6.0) == (None, "a bound is at or below the "
                                                             "physical floor")
    assert sources.qualify_interval(5.0, 6.0, 4.0) == (None, "bounds inverted")
    assert sources.qualify_interval(9.0, 4.0, 6.0) == (None, "bounds do not enclose the estimate")
    assert sources.qualify_interval(None, 4.0, 6.0) == (None, "no point estimate")


# --- the column-name trap -------------------------------------------------

def test_pred_affinity_is_enrolled_as_measured(tmp_path):
    rows = [{"Pred_affinity": "2.0", "heavy": HEAVY, "light": LIGHT, "fitness": "0.3"},
            {"Pred_affinity": "1.0", "heavy": "W" + HEAVY[1:], "light": LIGHT,
             "fitness": "0.8"}]
    block = sources.load_engelhart(write_zip(tmp_path / "engelhart.csv.zip", "e.csv", rows))
    assert len(block["observations"]) == 2
    record = block["observations"][0]
    assert record.family == "affinity"
    assert record.qualification["raw_units"] == "log10(KD/nM)"
    assert "not a trained model's prediction" in record.qualification[
        "column_name_is_misleading"]
    assert record.value == pytest.approx(sources.pkd_from_log10_nm(
        record.raw["Pred_affinity"]))


def test_koenig_binding_is_quarantined_rather_than_given_an_invented_direction(tmp_path):
    """An unsigned quantity trains nothing ordered.

    The release's four KD-named columns are transforms of one enrichment-derived
    number whose mapping AND direction are unestablished. An earlier revision
    enrolled it with higher-is-better semantics; a ranking loss on an unsigned
    quantity trains the reverse of the truth half the time, which is worse than
    not training on it. The raw columns are preserved and the row carries no value.
    """
    rows = [{"heavy": HEAVY, "light": LIGHT, "-ln(KD (nM) )": "-2.3", "KD (nM)": "10",
             "KD (M)": "1e-8", "-log( KD (M) )": "8", "fitness": "0.5"},
            {"heavy": "W" + HEAVY[1:], "light": LIGHT, "-ln(KD (nM) )": "-4.6",
             "KD (nM)": "100", "KD (M)": "1e-7", "-log( KD (M) )": "7", "fitness": "0.2"}]
    rows.append(dict(rows[0], heavy=HEAVY[:1] + "W" + HEAVY[2:]))
    block = sources.load_koenig(write_csv(tmp_path / "koenig.csv", rows))
    assert block["quarantined_binding_rows"] == 3
    for record in block["observations"]:
        assert record.family == "enrichment"
        assert record.kind == sources.KIND_ENRICHMENT
        assert record.scale == sources.SCALE_ENRICHMENT
        assert record.value is None, "a quarantined row carries no trainable value"
        assert record.raw["-ln(KD (nM) )"] is not None, "the raw column is preserved"
        assert "PHAGE ENRICHMENT" in record.qualification["curation_mapping"]
        assert record.qualification["quarantined"] is True
        assert record.qualification["trains"].startswith("nothing")
        assert "DIRECTION" in record.qualification["unresolved"]
        assert record.qualification["direction_evidence"][
            "pkd_vs_qualified_expression_spearman"] == pytest.approx(-0.499)
        assert "not about the experiment" in record.qualification[
            "copying_a_curation_does_not_establish_units"]
    # The internal consistency of the four transforms is measured, not assumed.
    assert block["consistency"]["rows_checked"] == 3
    assert block["consistency"]["max_abs_error_neg_ln_nm"] < 0.01
    # A flag alone cannot establish the sign. The historical fitness comparison can.
    with pytest.raises(ValueError, match="independently pinned"):
        sources.load_koenig(write_csv(tmp_path / "unsigned.csv", rows), quarantine_binding=False)
    historical = [{"heavy": r["heavy"], "light": r["light"], "fitness": r["KD (nM)"]}
                  for r in rows]
    lifted = sources.load_koenig(write_csv(tmp_path / "koenig2.csv", rows),
                                 historical_path=write_csv(tmp_path / "historical.csv", historical))
    assert all(record.value is not None for record in lifted["observations"])
    assert lifted["observations"][1].value > lifted["observations"][0].value


def test_expression_is_not_converted_into_an_affinity(tmp_path):
    rows = [{"Expression [ER]": "1.4", "heavy": HEAVY, "light": LIGHT, "format": "scFv",
             "fitness": "0.2"}]
    block = sources.load_expression(write_csv(tmp_path / "exp.csv", rows),
                                    source="adams2017_expression", column="Expression [ER]")
    record = block["observations"][0]
    assert record.kind == sources.KIND_ENRICHMENT
    assert "different functional readout" in record.qualification["is_not_an_affinity"]


# --- zip and preamble handling --------------------------------------------

def test_macos_resource_forks_are_filtered_and_two_csvs_are_refused(tmp_path):
    rows = [{"Pred_affinity": "2.0", "heavy": HEAVY, "light": LIGHT, "fitness": "0.3"}]
    target = write_zip(tmp_path / "ok.csv.zip", "real.csv", rows,
                       extra_members=("__MACOSX/._real.csv", "._real.csv"))
    name, read = sources.zipped_csv_rows(target)
    assert name == "real.csv" and len(read) == 1
    with zipfile.ZipFile(tmp_path / "two.csv.zip", "w") as archive:
        archive.writestr("a.csv", "x\n1\n")
        archive.writestr("b.csv", "x\n2\n")
    with pytest.raises(ValueError, match="candidate CSV file"):
        sources.zipped_csv_rows(tmp_path / "two.csv.zip")


def test_li2_preamble_is_asserted_before_it_is_skipped(tmp_path):
    rows = [{name: "" for name in sources.LI_HEADER}]
    rows[0].update(POI="p", HC=HEAVY, LC=LIGHT, Pred_affinity="1.0", Target="S1",
                   Assay="alphaseq", Replicate="1")
    preamble = [f"note {index}" for index in range(sources.LI2_PREAMBLE_LINES)]
    good = write_csv(tmp_path / "li2.csv", rows, preamble=preamble)
    block = sources.load_li(good, source="li2023_s1_b",
                            preamble=sources.LI2_PREAMBLE_LINES)
    assert len(block["observations"]) == 1
    assert block["preamble_lines"] == sources.LI2_PREAMBLE_LINES
    # A wrong declared preamble is refused rather than silently discarding a data row.
    with pytest.raises(ValueError, match="preamble"):
        sources.load_li(good, source="li2023_s1_b", preamble=3)


def test_a_preamble_line_that_is_the_header_is_refused(tmp_path):
    rows = [{name: "" for name in sources.LI_HEADER}]
    rows[0].update(POI="p", HC=HEAVY, LC=LIGHT, Pred_affinity="1.0")
    bad = write_csv(tmp_path / "li2bad.csv", rows, preamble=["POI,Sequence"] * 2)
    with pytest.raises(ValueError, match="begins with"):
        sources.load_li(bad, source="li2023_s1_b", preamble=2)


# --- I15: the CR9114 H1 guard --------------------------------------------

def _cr9114_space(tmp_path, *, sites=16):
    """A prepared manifest with a real decoded sequence and a real edit space."""
    germline = ("A" * sites) + "GGGG"
    document = {"manifest": {
        "decoded_sequence": germline,
        "edit_space": {"sites": [
            {"site_id": f"s{index}", "sequence_index": index,
             "allowed_residues": ["A", "V"]} for index in range(sites)]}}}
    path = tmp_path / "cr9114.prepared.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return sources.CR9114EditSpace.from_prepared(path)


def test_cr9114_requires_the_real_edit_space_and_has_no_placeholder(tmp_path):
    """No fallback sequence. A fallback maps 65,536 genotypes onto one construct."""
    path = _cr9114_csv(tmp_path)
    with pytest.raises(ValueError, match="needs a CR9114EditSpace"):
        sources.load_cr9114(path, antigen="h3")
    with pytest.raises(ValueError, match="prepared manifest .* is missing"):
        sources.CR9114EditSpace.from_prepared(tmp_path / "absent.json")
    space = _cr9114_space(tmp_path)
    assert space.length == 16
    # Bit i selects the somatic residue at site i, and the germline is bit 0.
    assert space.sequence_for("0" * 16) == space.germline
    assert space.sequence_for("1" + "0" * 15)[0] == "V"
    assert space.sequence_for("1" * 16).count("V") == 16
    assert space.edit_sites() == tuple(("H", index) for index in range(16))
    with pytest.raises(ValueError, match="against a 16-site edit space"):
        space.sequence_for("01")
    block = sources.load_cr9114(path, antigen="h3", space=space)
    heavies = {record.construct.heavy for record in block["observations"]}
    assert len(heavies) == 3, "distinct genotypes, distinct sequences"
    assert len(block["observations"]) == 9, "three real replicate likelihoods per genotype"
    assert all(len(record.construct.sites) == 16 for record in block["observations"])
    assert block["edit_space"]["sites"] == 16


def test_cr9114_only_enrols_whitelisted_genotypes_when_one_is_supplied(tmp_path):
    path = _cr9114_csv(tmp_path)
    space = _cr9114_space(tmp_path)
    everything = sources.load_cr9114(path, antigen="h3", space=space)
    gated = sources.load_cr9114(path, antigen="h3", space=space,
                                allowed_genotypes={"1" * 16})
    assert len(everything["observations"]) == 9
    assert len(gated["observations"]) == 3
    assert gated["excluded"]["not_in_allowed_genotypes"] == 2
    assert gated["genotype_gate"] == "the pinned eligibility whitelist"
    assert everything["genotype_gate"] == "UNRESTRICTED"


def _cr9114_csv(tmp_path):
    header = (["genotype"] + [f"h1_{part}" for part in ("repa", "repb", "repc", "mean", "sem")]
              + [f"h3_{part}" for part in ("repa", "repb", "repc", "mean", "sem")]
              + [f"fluB_{part}" for part in ("repa", "repb", "repc", "mean", "sem")]
              + [f"pos{index}" for index in range(1, 17)] + ["som_mut"])
    # Row 0 sits entirely AT OR BELOW the floor, so it must come back left-censored.
    # Rows 1 and 2 are above it, so they must come back uncensored. Mixing them is the
    # point: a fixture where nothing is at the floor cannot test the floor.
    replicates = {"0" * 16: (6.0, 5.9, 6.0), "1" + "0" * 15: (7.0, 7.1, 7.2),
                  "1" * 16: (8.0, 8.1, 8.2)}
    rows = []
    for genotype, values in replicates.items():
        row = dict.fromkeys(header, "")
        row["genotype"] = genotype
        for part, value in (("repa", 9.0), ("repb", 9.1), ("repc", 9.2)):
            row[f"h1_{part}"] = str(value)
        for part, value in zip(("repa", "repb", "repc"), values):
            row[f"h3_{part}"] = str(value)
            row[f"fluB_{part}"] = str(value - 0.5)
        row["h3_mean"] = str(sum(values) / 3.0)
        row["h3_sem"] = "0.1"
        row["fluB_mean"] = str(sum(values) / 3.0 - 0.5)
        row["fluB_sem"] = "0.1"
        for position, bit in enumerate(genotype, start=1):
            row[f"pos{position}"] = bit
        row["som_mut"] = str(genotype.count("1"))
        rows.append(row)
    return write_csv(tmp_path / "cr9114.csv", rows)


def test_i15_h1_columns_are_structurally_unreadable(tmp_path):
    path = _cr9114_csv(tmp_path)
    block = sources.load_cr9114(path, antigen="h3", space=_cr9114_space(tmp_path))
    assert not set(block["columns_read"]) & sources.CR9114_RESERVED_COLUMNS
    assert sorted(block["reserved_columns_refused"]) == sorted(sources.CR9114_RESERVED_COLUMNS)
    for record in block["observations"]:
        assert record.qualification["h1_never_read"]
        assert "h1_repa" not in record.raw
    with pytest.raises(ValueError, match="Unsupported antigen partition"):
        sources.load_cr9114(path, antigen="h1", space=_cr9114_space(tmp_path))


def test_i15_h3_floor_is_left_censoring_not_a_weak_target(tmp_path):
    block = sources.load_cr9114(_cr9114_csv(tmp_path), antigen="h3",
                                space=_cr9114_space(tmp_path),
                                floor=sources.CR9114_H3_FLUB_FLOOR)
    censored = [record for record in block["observations"] if record.censoring == "left"]
    uncensored = [record for record in block["observations"] if record.censoring == "none"]
    assert len(censored) == 3
    assert len(uncensored) == 6
    for record in censored:
        assert record.value is None
        assert record.interval == (None, sources.CR9114_H3_FLUB_FLOOR)
        assert "LEFT-censored" in record.qualification["floor_semantics"]
    for record in uncensored:
        assert record.value is not None and record.value > sources.CR9114_H3_FLUB_FLOOR
        assert record.uncertainty is None  # this is a replicate, not its pooled SEM
        assert record.value == record.raw["replicate_value"]


def test_i15_h3_and_flub_are_never_pooled(tmp_path):
    path = _cr9114_csv(tmp_path)
    space = _cr9114_space(tmp_path)
    h3 = sources.load_cr9114(path, antigen="h3", space=space)
    flub = sources.load_cr9114(path, antigen="fluB", space=space)
    assert {record.target for record in h3["observations"]} == {"H3 HA"}
    assert {record.target for record in flub["observations"]} == {"influenza B HA"}
    assert h3["columns_read"] != flub["columns_read"]
    # Two antigens are two assay identities, so they get separate observation scales
    # and separate bin thresholds instead of one pooled population.
    assert ({record.assay_key for record in h3["observations"]}
            != {record.assay_key for record in flub["observations"]})


def test_a_partially_censored_row_is_not_averaged_into_an_exact_point(tmp_path):
    """One replicate at the floor says 'no tighter than this'. It is not a number.

    Averaging it in produces a point estimate the assay did not produce, biased
    towards the floor by exactly the censored replicate's weight.
    """
    header = (["genotype"] + [f"h3_{part}" for part in ("repa", "repb", "repc", "mean", "sem")])
    rows = [dict.fromkeys(header, "")]
    rows[0].update(genotype="1" * 16, h3_repa="6.0", h3_repb="8.0", h3_repc="8.2",
                   h3_mean="7.4", h3_sem="0.7")
    path = write_csv(tmp_path / "partial.csv", rows)
    block = sources.load_cr9114(path, antigen="h3", space=_cr9114_space(tmp_path))
    records = block["observations"]
    assert len(records) == 3
    assert records[0].censoring == "left" and records[0].interval == (None, 6.0)
    assert records[0].value is None
    assert [row.value for row in records[1:]] == pytest.approx([8.0, 8.2])
    assert len({row.observation_identity for row in records}) == 3


def test_cr9114_zero_allele_comes_from_allowed_order_not_the_mature_structure(tmp_path):
    path = tmp_path / "mature.json"
    path.write_text(json.dumps({"manifest": {
        "decoded_sequence": "VVGG", "edit_space": {"sites": [
            {"site_id": "one", "sequence_index": 0, "allowed_residues": ["A", "V"]},
            {"site_id": "two", "sequence_index": 1, "allowed_residues": ["C", "V"]}]}}}),
        encoding="utf-8")
    space = sources.CR9114EditSpace.from_prepared(path)
    assert space.sequence_for("00") == "ACGG"
    assert space.sequence_for("11") == "VVGG"
    assert space.sequence_for("01") == "AVGG"


# --- AlphaSeq compatibility ----------------------------------------------

def _alphaseq_sequence(core, *, linker="G" * 15, light=None, heavy_prefix=None):
    heavy = (heavy_prefix or "A" * 98) + core + "Y" * (120 - 98 - len(core))
    return heavy + linker + (light or "C" * 107) + "R"


class _Scaffold:
    def __init__(self, heavy, light):
        self.heavy, self.light = heavy, light


def test_alphaseq_construct_offsets_are_asserted_constants():
    sequence = _alphaseq_sequence("WGGDGFYAMD")
    assert len(sequence) == sources.ALPHASEQ_CONSTRUCT_LENGTH
    parts = sources.split_alphaseq_construct(sequence)
    assert parts is not None
    heavy, linker, light, terminal = parts
    assert len(heavy) == 120 and len(linker) == 15 and len(light) == 107
    assert terminal == "R"
    # A missing terminal R is not a compatible construct.
    assert sources.split_alphaseq_construct(sequence[:-1] + "A") is None
    assert sources.split_alphaseq_construct(sequence[:-1]) is None


def test_alphaseq_parent_is_derived_and_a_mixture_is_refused():
    sequences = [_alphaseq_sequence("WGGDGFYAMD") for _ in range(9)]
    sequences.append(_alphaseq_sequence("WGGDGFYAMD", linker="S" * 15))
    parent = sources.derive_alphaseq_parent(sequences)
    assert parent["linker"] == "G" * 15
    assert parent["shares"]["linker"] == pytest.approx(0.9)
    mixture = ([_alphaseq_sequence("WGGDGFYAMD") for _ in range(5)]
               + [_alphaseq_sequence("WGGDGFYAMD", linker="S" * 15) for _ in range(5)])
    with pytest.raises(ValueError, match="not one parental scaffold"):
        sources.derive_alphaseq_parent(mixture)


def test_alphaseq_compatibility_refuses_core_only_inclusion():
    from smallAntibodyGen.experiments.her2_routes_serial import her2_core_edit_map
    sites = her2_core_edit_map()
    parent = {"linker": "G" * 15, "light": "C" * 107}
    scaffold = _Scaffold(heavy="A" * 98 + "WGGDGFYAMD" + "Y" * 12, light="C" * 107)
    ok, reason = sources.alphaseq_compatibility(
        _alphaseq_sequence("WAGDGFYAMD"), parent=parent, scaffold=scaffold, sites=sites)
    assert ok and reason == "compatible"
    # The right ten-residue window but a different framework: refused.
    bad, reason = sources.alphaseq_compatibility(
        _alphaseq_sequence("WGGDGFYAMD", heavy_prefix="W" + "A" * 97),
        parent=parent, scaffold=scaffold, sites=sites)
    assert not bad and "heavy framework position 0" in reason
    # A different light chain: refused, and no light chain is invented.
    bad, reason = sources.alphaseq_compatibility(
        _alphaseq_sequence("WGGDGFYAMD", light="D" * 107),
        parent=parent, scaffold=scaffold, sites=sites)
    assert not bad and "VL differs" in reason


def test_sequence_only_read_does_not_materialise_the_outcome(tmp_path):
    rows = [{"mata_description": "d1", "matalpha_description": "t",
             "alphaseq_affinity": "1.5", "mata_sequence": _alphaseq_sequence("WGGDGFYAMD"),
             "predicted_affinity": "1.4", "edit_distance": "1", "parental_mean_aff": "1.0",
             "parental_aff_margin": "0.1", "avg_off_target_affinity": "3.0", "model": "m",
             "id": "1"}]
    block = sources.load_alphaseq_sequences(write_csv(tmp_path / "ab.csv", rows))
    assert set(block) == {"sequences", "descriptions", "rows", "sha256", "columns_parsed",
                          "outcome_columns_present_but_unread", "rule"}
    assert "alphaseq_affinity" in block["outcome_columns_present_but_unread"]
    assert "1.5" not in str(block["sequences"])
    # The restriction is in the PARSER, not in a later discard: only the sequence and
    # the descriptor are converted into values at all.
    assert block["columns_parsed"] == ["mata_sequence", "mata_description"]
    assert not set(block["columns_parsed"]) & sources.ALPHASEQ_OUTCOME_COLUMNS


def test_the_outcome_read_needs_either_a_window_gate_or_a_single_use_unlock(tmp_path):
    """Loading HER2 outcomes without the frozen window set reads the reserved panel."""
    from smallAntibodyGen.experiments.her2_routes_serial import her2_core_edit_map
    core = "WGGDGFYAMD"
    rows = [{"mata_description": "d1", "matalpha_description": "t",
             "alphaseq_affinity": "1.5", "mata_sequence": _alphaseq_sequence(core),
             "predicted_affinity": "1.4", "edit_distance": "1", "parental_mean_aff": "1.0",
             "parental_aff_margin": "0.1", "avg_off_target_affinity": "3.0", "model": "m",
             "id": "1"}]
    path = write_csv(tmp_path / "ab.csv", rows)
    scaffold = _Scaffold(heavy="A" * 98 + core + "Y" * 12, light="C" * 107)
    sites = her2_core_edit_map()
    parent = {"linker": "G" * 15, "light": "C" * 107}
    with pytest.raises(ValueError, match="reads the reserved panel too"):
        sources.load_alphaseq_observations(path, source="alphabind_her2_validation",
                                           scaffold=scaffold, sites=sites, parent=parent)
    gated = sources.load_alphaseq_observations(
        path, source="alphabind_her2_validation", scaffold=scaffold, sites=sites,
        parent=parent, allowed_windows={core})
    assert len(gated["observations"]) == 1
    assert gated["window_gate"] == "frozen train/development windows only"
    withheld = sources.load_alphaseq_observations(
        path, source="alphabind_her2_validation", scaffold=scaffold, sites=sites,
        parent=parent, allowed_windows={"AAAAAAAAAA"})
    assert withheld["observations"] == ()
    assert withheld["excluded"]["not in the frozen train/development window set"] == 1
    # The unlock is the single declared opening and refuses a second use.
    unlock = sources.ReservedUnlock(reason="the one declared reserved opening",
                                    frozen_selection_sha256="abc", minted_by="test")
    opened = sources.load_alphaseq_observations(
        path, source="alphabind_her2_validation", scaffold=scaffold, sites=sites,
        parent=parent, reserved_unlock=unlock)
    assert opened["window_gate"] == "UNRESTRICTED"
    assert opened["reserved_unlock"]["minted_by"] == "test"
    with pytest.raises(ValueError, match="already been spent"):
        sources.load_alphaseq_observations(path, source="alphabind_her2_validation",
                                           scaffold=scaffold, sites=sites, parent=parent,
                                           reserved_unlock=unlock)


# --- bins -----------------------------------------------------------------

def test_bins_are_thirds_with_a_fixed_tie_rule_and_a_support_floor():
    values = list(range(90))
    thresholds = fit_bin_thresholds(values, family="affinity", assay="a", scale="pKD",
                                    minimum_support=30)
    assert thresholds is not None
    assert assign_bin(thresholds, value=0) == BIN_LOW
    assert assign_bin(thresholds, value=89) == BIN_HIGH
    # A value exactly at the lower cut falls in the LOWER bin.
    assert assign_bin(thresholds, value=thresholds.lower) == BIN_LOW
    # Below the support floor no thresholds exist at all.
    assert fit_bin_thresholds([1, 2, 3], family="affinity", assay="a", scale="pKD",
                              minimum_support=30) is None
    # A degenerate distribution has no thirds, and that is reported rather than nudged.
    assert fit_bin_thresholds([5.0] * 100, family="affinity", assay="a",
                              scale="pKD", minimum_support=30) is None


def test_a_straddling_censored_interval_gets_no_bin():
    thresholds = fit_bin_thresholds(list(range(90)), family="affinity", assay="a",
                                    scale="pKD", minimum_support=30)
    assert assign_bin(thresholds, interval=(None, 89.0)) is None
    assert assign_bin(thresholds, interval=(0.0, 1.0)) == BIN_LOW
    assert assign_bin(thresholds, interval=(88.0, None)) == BIN_HIGH


# --- the Adams primary replicates ----------------------------------------

def test_adams_wild_type_windows_are_asserted_against_the_backbone():
    heavy = sources.adams_wild_type_heavy()
    assert len(heavy) == 117
    assert heavy[27:37] == sources.ADAMS_WT_CDR1 == "TFSDYWMNWV"
    assert heavy[99:109] == sources.ADAMS_WT_CDR3 == "GSYYGMDYWG"
    built = sources.adams_construct_heavy("TFGDYWMNWV", "GSYYGMDYWG")
    assert len(built) == 117
    assert built[27:37] == "TFGDYWMNWV"
    # Everything outside the two windows is the wild-type backbone, unchanged.
    assert built[:27] == heavy[:27] and built[37:99] == heavy[37:99]
    assert built[109:] == heavy[109:]
    with pytest.raises(ValueError, match="does not fill positions"):
        sources.adams_construct_heavy("SHORT", sources.ADAMS_WT_CDR3)


def _adams_replicate_rows(*, kd="1e-7", sigma="0.0227"):
    """Three wild-type-SYNONYMOUS rows plus three mutants.

    The three wild-type rows carry the same translated windows and different
    nucleotide barcodes, which is what the release's synonymous copies of the
    parent are and what the per-replicate expression normaliser is taken over.
    """
    wild = {"CDR1H_AA": sources.ADAMS_WT_CDR1, "CDR3H_AA": sources.ADAMS_WT_CDR3,
            "fit_KD": kd, "fit_KD_sigma": sigma, "expression": "10.0",
            "CDR1H": "acg", "CDR3H": "tgt", "unsorted": "100",
            "fit_fraction": "0.1", "fit_saturation": "27.0"}
    synonymous = [dict(wild, CDR1H="acc", expression="9.0"),
                  dict(wild, CDR1H="aca", expression="11.0")]
    variant = dict(wild, CDR1H_AA="TFGDYWMNWV", fit_KD="1e-6", expression="5.0")
    tight = dict(wild, CDR3H_AA="GSYYGMDYWA", fit_KD="1e-11")
    weak = dict(wild, CDR3H_AA="GSYYGMDYWV", fit_KD="1e-3")
    return [wild, *synonymous, variant, tight, weak]


def test_adams_primary_keeps_replicates_and_rescales_sigma_to_pkd(tmp_path):
    """Three replicates, and a sigma in natural-log KD divided by ln(10).

    ``KD_fit_log_poiss.py`` computes the fit variance in natural-log KD. Carrying
    the reported sigma unconverted states every Adams uncertainty 2.303x too wide,
    and nothing downstream would look wrong.
    """
    root = tmp_path / "adams2017"
    (root / "data").mkdir(parents=True)
    for number in (1, 2, 3):
        write_csv(root / "data" / f"replicate_{number}.csv", _adams_replicate_rows())
    block = sources.load_adams_primary(root, light="C" * 112)
    assert set(block["per_replicate"]) == {"replicate_1", "replicate_2", "replicate_3"}
    by_replicate = {}
    for record in block["observations"]:
        by_replicate.setdefault(record.replicate, []).append(record)
    assert sorted(by_replicate) == [1, 2, 3]
    record = by_replicate[1][0]
    assert len(record.construct.heavy) == 117 and len(record.construct.light) == 112
    assert len(record.construct.sites) == 20, "the two ten-residue windows"
    # 0.0227 natural-log sigma is 0.00986 in pKD, not 0.0227.
    assert record.uncertainty == pytest.approx(0.0227 / sources.ADAMS_SIGMA_LOG_BASE, rel=1e-6)
    assert record.raw["fit_KD_sigma_natural_log"] == pytest.approx(0.0227)
    assert "ln(10)" in record.qualification["sigma_units"]
    # Doubly censored, with the tight tail on the RIGHT in pKD.
    censoring = {row.censoring for row in by_replicate[1]}
    assert censoring == {"none", "right", "left"}
    right = next(row for row in by_replicate[1] if row.censoring == "right")
    assert right.interval == (sources.ADAMS_RIGHT_CENSOR_PKD, None) and right.value is None
    # Expression is normalised by this replicate's wild-type-synonymous rows.
    normaliser = block["per_replicate"]["replicate_1"]["normaliser"]
    assert normaliser["wild_type_synonymous_rows"] == 3
    assert normaliser["normaliser"] == pytest.approx(10.0)
    variant = next(row for row in by_replicate[1]
                   if row.raw["CDR1H_AA"] == "TFGDYWMNWV")
    assert variant.raw["expression_normalised"] == pytest.approx(0.5)
    # Same canonical study as the curated summary, so the closure cannot count both.
    assert record.study == sources.canonical_study("adams2017_fluorescein")


def test_adams_light_chain_comes_from_the_curated_table_and_is_validated(tmp_path):
    rows = [{"Kd (Tite-Seq) [M]": "1e-7", "heavy": "A" * 117, "light": "C" * 112,
             "format": "Fab", "fitness": "0.5"}]
    curated = sources.load_adams_affinity(write_csv(tmp_path / "adams.csv", rows))
    assert sources.derive_adams_light(curated["observations"]) == "C" * 112
    mixed = rows + [{"Kd (Tite-Seq) [M]": "1e-7", "heavy": "A" * 117, "light": "D" * 112,
                     "format": "Fab", "fitness": "0.5"}]
    two = sources.load_adams_affinity(write_csv(tmp_path / "adams2.csv", mixed))
    with pytest.raises(ValueError, match="distinct light chains"):
        sources.derive_adams_light(two["observations"])


# --- the Petersen parent export ------------------------------------------

def test_petersen_parents_resolve_a_variant_specification(tmp_path):
    rows = [{"Antibody": "222-1C06", "Chain": "VH", "Sequence": "EVQLVESGGD"},
            {"Antibody": "222-1C06", "Chain": "VL", "Sequence": "EIVMTQSPAT"}]
    block = sources.load_petersen_parents(write_csv(tmp_path / "parents.csv", rows))
    assert block["paired"] == ["222-1C06"]
    assert "Whitehead" in block["provenance"]
    resolved = sources.resolve_petersen_variant("222-1C06>VH:E1Q|222-1C06>VL:I2L",
                                                block["parents"])
    heavy, light, sites = resolved
    assert heavy == "QVQLVESGGD" and light == "ELVMTQSPAT"
    assert sites == (("H", 0), ("L", 1))
    # A stated original residue that does not match the parent is refused, not applied.
    assert sources.resolve_petersen_variant("222-1C06>VH:A1Q", block["parents"]) is None
    # An unknown parent is unresolved rather than attached to a guessed background.
    assert sources.resolve_petersen_variant("nobody>VH:E1Q", block["parents"]) is None


# --- the source table ----------------------------------------------------


def test_petersen_actual_semicolon_dna_grammar_and_stop_codon_rejection():
    parents = {("p", "VH"): "EVQLVESGGD", ("p", "VL"): "EIVMTQSPAT"}
    resolved = sources.resolve_petersen_variant(
        "p>VH:E1Q-CAA;V2V-GTC;L4A-GCC|p>VL:I2L-CTG", parents)
    assert resolved == ("QVQAVESGGD", "ELVMTQSPAT", (("H", 0), ("H", 3), ("L", 1)))
    assert sources.resolve_petersen_variant("p>VH:E1*-TAG|p>VL:WT", parents) is None
    assert sources.resolve_petersen_variant("p>VH:unparsed|p>VL:WT", parents) is None


def test_adams_synonymous_barcodes_are_measurements_not_duplicate_publications(tmp_path):
    root = tmp_path / "adams"
    (root / "data").mkdir(parents=True)
    write_csv(root / "data" / "replicate_1.csv", _adams_replicate_rows())
    block = sources.load_adams_primary(root, light="C" * 112, replicates=(1,))
    closed = sources.observation_closure({"a": block["observations"], "copy": block["observations"]})
    assert closed["kept"] == 6 and closed["duplicates"] == 6
    assert closed["unique_constructs"] == 4


def test_petersen_fit_ci_is_uncertainty_and_n_is_not_a_biological_replicate(monkeypatch, tmp_path):
    path = tmp_path / "p.xlsx"
    path.write_bytes(b"fixture")
    rows = [{"Variant": "p>VH:E1Q-CAA|p>VL:WT", "Kd": 10.0,
             "95% CI low": 5.0, "95% CI high": 20.0, "Barcode": "b1", "Ab": "p", "n": 10},
            {"Variant": "p>VH:WT|p>VL:WT", "Kd": 20.0,
             "95% CI low": 0.0, "95% CI high": 10.0, "Barcode": "b2", "Ab": "p", "n": 9}]
    monkeypatch.setattr(sources, "read_xlsx_sheet", lambda *args: rows)
    parents = {("p", "VH"): "EVQLVESGGD", ("p", "VL"): "EIVMTQSPAT"}
    block = sources.load_petersen_primary(path, parents=parents, sheets=("s",))
    mutant, wt = block["observations"]
    assert mutant.construct.sites == wt.construct.sites == (("H", 0),)
    assert mutant.interval == (8.0, 8.0) and mutant.censoring == "none"
    assert mutant.uncertainty == pytest.approx((math.log10(20) - math.log10(5)) / 3.92)
    assert mutant.raw["barcode"] == "b1" and mutant.replicate is None
    assert wt.raw["ci_nM"] == [0.0, 10.0] and wt.uncertainty is None
    assert wt.raw["number_of_concentrations"] == 9

def test_source_table_agrees_with_the_tracked_qualification_config(project_root):
    config = json.loads(
        (project_root.parents[1] / "configs" / "experiments"
         / "her2_generative_routes_sources.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == "her2-generative-routes-sources/1"
    assert sources.CR9114_H3_FLUB_FLOOR == config["sources"]["cr9114"]["floor_pkd"]
    assert sorted(sources.CR9114_RESERVED_COLUMNS) == sorted(
        config["sources"]["cr9114"]["reserved_columns"])
    assert sources.LI2_PREAMBLE_LINES == config["sources"]["li2023_s1"]["files"][
        "affinity2"]["preamble_lines"]
    window = config["sources"]["adams2017_fluorescein"]["detection_window_molar"]
    assert sources.ADAMS_WINDOW_MOLAR[0] == pytest.approx(window[0])
    assert sources.ADAMS_WINDOW_MOLAR[1] == pytest.approx(window[1])
    assert sources.SOURCES["alphabind_her2_full"].enrolled is False


def test_observation_closure_collapses_a_republished_row_not_a_replicate(tmp_path):
    rows = [{"Pred_affinity": "2.0", "heavy": HEAVY, "light": LIGHT, "fitness": "0.3"}]
    block = sources.load_engelhart(write_zip(tmp_path / "e.csv.zip", "e.csv", rows))
    doubled = sources.observation_closure({"engelhart2022_s1": block["observations"],
                                           "republished": block["observations"]})
    assert doubled["kept"] == 1
    assert doubled["duplicates"] == 1

"""Shared common regions, CDR annotation locating, and the non-HER2 chain boundary.

Three loader defects are pinned, each measured against the real released sequences
rather than a synthetic stand-in.

**A mutation-dependent edit map is not a region.** A single mutant got a one-site
window and a triple mutant a three-site window, so their window log probabilities
were not comparable, a whole-window preference between them compared two different
quantities, and the width of the window leaked how many edits the row carried.
:func:`common_region` computes the positions that vary across a whole group and
assigns the same one to every member.

**A CDR boundary is an annotation.** The earlier reader dropped the six CDR columns
as "derivable from HC/LC", which is false -- nothing in a bare heavy chain says where
CDRH2 ends -- and then had no region available but each row's own mutations. The
sequences and annotations here are the released MIT_14, MIT_91 and MIT_95 parent
controls, so :func:`locate_cdrs` is checked against the spans the release declares.

**A fixed offset is not a chain boundary.** The non-HER2 AlphaSeq releases carry
insertions, deletions and framework substitutions, so the length-243 offsets of the
trastuzumab population reassign the boundary on any indel row. The linker is located
instead, and a construct with no linker is heavy-only -- which is what the nanobody
release is.
"""
from __future__ import annotations

import pytest

from smallAntibodyGen.experiments import her2_routes_split as split_lib
from smallAntibodyGen.experiments import her2_routes_sources as sources
from smallAntibodyGen.experiments.her2_routes_serial import HEAVY, LIGHT, Construct

#: The released MIT_14 parent control: heavy, light and its six CDR annotations.
MIT14 = {
    "HC": ("EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTST"
           "NTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSS"),
    "LC": ("DVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTD"
           "FTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK"),
    "CDRH1": "GFTLNSYGIS", "CDRH2": "VIYSDGRRTFYGDSVK", "CDRH3": "GRAAGTFDS",
    "CDRL1": "KSSQSVLYESRNKNSVA", "CDRL2": "WASTRES", "CDRL3": "QQYHRLPLS"}

#: The released MIT_95 parent control. A different scaffold family: different lengths,
#: different CDR spans, and therefore a different common region.
MIT95 = {
    "HC": ("EVQLVESGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTST"
           "STAYMELRSLRSDDTAVYYCARVGRGVIDHWGQGTLVTVSS"),
    "LC": ("SSELTQDPAVSVALGQTVRITCEGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITG"
           "AQAEDEADYYCSSRDSSGFQVFFGAGTKLTVL"),
    "CDRH1": "GYTFTSYGIS", "CDRH2": "WISAYNGNTNYAQKLQ", "CDRH3": "VGRGVIDH",
    "CDRL1": "EGDSLRYYYAN", "CDRL2": "GKNNRPS", "CDRL3": "SSRDSSGFQVF"}

LINKER = sources.ALPHASEQ_LINKER


# --- CDR annotation locating -----------------------------------------------

def test_the_released_annotations_locate_uniquely_and_give_the_declared_spans():
    sites, signature = sources.locate_cdrs(MIT14["HC"], MIT14["LC"], MIT14)
    assert sites is not None, signature
    heavy_sites = [index for chain, index in sites if chain == HEAVY]
    light_sites = [index for chain, index in sites if chain == LIGHT]
    # The release's own declared coordinates for MIT_14: heavy 25-34, 49-64, 98-106 and
    # light 23-39, 55-61, 94-102. Derived from the annotations, not assumed.
    assert heavy_sites == list(range(25, 35)) + list(range(49, 65)) + list(range(98, 107))
    assert light_sites == list(range(23, 40)) + list(range(55, 62)) + list(range(94, 103))
    assert len(sites) == 68
    assert signature.count("|") == 1
    assert "*" * 10 in signature, "the located spans are masked out of the framework signature"


def test_two_scaffold_families_have_different_signatures_and_regions():
    left_sites, left_signature = sources.locate_cdrs(MIT14["HC"], MIT14["LC"], MIT14)
    right_sites, right_signature = sources.locate_cdrs(MIT95["HC"], MIT95["LC"], MIT95)
    assert left_signature != right_signature
    assert left_sites != right_sites
    # MIT_95's light chain is shorter and its CDR spans sit elsewhere, which is exactly
    # why one pooled modal parent across families produces a meaningless edit map.
    assert len(MIT95["LC"]) != len(MIT14["LC"])


def test_an_annotation_that_does_not_occur_is_refused_not_guessed():
    broken = dict(MIT14, CDRH3="WWWWWWWWWW")
    sites, reason = sources.locate_cdrs(MIT14["HC"], MIT14["LC"], broken)
    assert sites is None and "CDRH3 does not occur" in reason


def test_an_annotation_that_occurs_twice_is_refused():
    heavy = "AAAA" + "GGG" + "AAAA" + "GGG" + "AAAA"
    annotations = dict(MIT14, CDRH1="GGG")
    sites, reason = sources.locate_cdrs(heavy, MIT14["LC"], annotations)
    assert sites is None and "occurs more than once" in reason


def test_an_empty_annotation_is_refused():
    sites, reason = sources.locate_cdrs(MIT14["HC"], MIT14["LC"], dict(MIT14, CDRL2=""))
    assert sites is None and "empty" in reason


# --- the cross-release parent table ---------------------------------------

def test_the_second_release_families_resolve_to_the_first_releases_parent():
    """14H and 14L are MIT_14, supported by the framework signature AND the design names."""
    assert sources.li_parent_of("14H_En_HC_25005", source="li2023_s1_b") == "MIT_14_HL_scFV"
    assert sources.li_parent_of("14L_GP_Gibbs_28422", source="li2023_s1_b") == "MIT_14_HL_scFV"
    # The first release's own four families keep their own three parents.
    assert sources.li_parent_of("AAYL49_28", source="li2023_s1_a") == "MIT_14_HL_scFV"
    assert sources.li_parent_of("AAYL50_1332", source="li2023_s1_a") == "MIT_14_HL_scFV"
    assert sources.li_parent_of("AAYL51_3753", source="li2023_s1_a") == "MIT_91_LH_scFV"
    assert sources.li_parent_of("AAYL52_12105", source="li2023_s1_a") == "MIT_95_HL_scFV"
    assert sources.li_parent_of("MIT_14_HL_scFV", source="li2023_s1_a") == "MIT_14_HL_scFV"
    assert sources.li_parent_of("UNKNOWN_1", source="li2023_s1_a") is None


def test_a_replicate_identifier_of_one_point_zero_is_replicate_one():
    """The two releases disagree on the column's type and ``isdigit`` is false for 1.0."""
    assert sources._replicate_id("1") == 1
    assert sources._replicate_id("1.0") == 1
    assert sources._replicate_id(2.0) == 2
    assert sources._replicate_id("") == -1
    assert sources._replicate_id("nan") == -1
    assert sources._replicate_id("1.5") == -1


# --- the shared common region ---------------------------------------------

def observation(heavy, light, *, value=8.0, source="probe"):
    return sources.Observation(
        construct=Construct(heavy=heavy, light=light, sites=(), source=source,
                            record_id=heavy[:6]),
        source=source, family=sources.FAMILY_AFFINITY, kind=sources.KIND_POINT,
        target="t", assay=source, parent=f"{source}::parental", scale=sources.SCALE_PKD,
        value=value, interval=(value, value))


def test_the_common_region_is_the_groups_varying_positions_not_the_rows_own():
    heavy = "ACDEFGHIKL"
    light = "KLIHGFEDCA"
    rows = [observation(heavy, light),
            observation("WCDEFGHIKL", light),
            observation("WCDEFGHIKW", light),
            observation(heavy, "KLIHGFEDCW")]
    region, geometry = sources.common_region(rows, {"heavy": heavy, "light": light})
    assert region == ((HEAVY, 0), (HEAVY, 9), (LIGHT, 9))
    assert geometry["varying"] == 3 and geometry["dropped"] == 0
    enriched, unalignable, record = sources._assign_sites(
        rows, {"heavy": heavy, "light": light})
    assert unalignable == 0
    # Every member gets the SAME region, and its own mutations are carried separately.
    assert {row.construct.sites for row in enriched} == {region}
    single = next(row for row in enriched if row.construct.heavy == "WCDEFGHIKL")
    assert single.raw["mutated_sites"] == [[HEAVY, 0]]
    triple = next(row for row in enriched if row.construct.heavy == "WCDEFGHIKW")
    assert triple.raw["mutated_sites"] == [[HEAVY, 0], [HEAVY, 9]]
    assert record["head_only_rows"] == 0


def test_an_indel_row_keeps_its_outcome_and_loses_its_region():
    heavy = "ACDEFGHIKL"
    light = "KLIHGFEDCA"
    rows = [observation(heavy, light), observation("WCDEFGHIKL", light),
            observation("ACDEFGHIK", light)]
    enriched, unalignable, record = sources._assign_sites(
        rows, {"heavy": heavy, "light": light})
    assert unalignable == 1
    short = next(row for row in enriched if len(row.construct.heavy) == 9)
    assert short.construct.sites == (), (
        "an indel needs an alignment decision this campaign has not made, so the row trains "
        "the outcome heads and not the window tasks")
    assert short.raw["mutated_sites"] == []
    assert record["head_only_rows"] == 1
    # The population is NOT truncated: the row keeps its measured value.
    assert short.value == pytest.approx(8.0)


def test_a_region_wider_than_the_budget_is_capped_with_the_dropped_count():
    heavy = "ACDEFGHIKL"
    rows = [observation(heavy, ""), observation("WWWWWWWWWW", ""),
            observation("WWWWWWWWWA", "")]
    region, geometry = sources.common_region(rows, {"heavy": heavy, "light": ""},
                                            maximum=4)
    assert len(region) == 4
    assert geometry["dropped"] == 6 and geometry["varying"] == 10


# --- the non-HER2 chain boundary ------------------------------------------

def test_the_linker_locates_the_chain_boundary_in_a_paired_construct():
    heavy, light = "QVQLVESGGG", "DVQMTQSPSS"
    parts = sources.split_alphaseq_scfv(heavy + LINKER + light)
    assert parts == (heavy, light, "paired_scfv")


def test_a_construct_with_no_linker_is_heavy_only_and_no_light_chain_is_invented():
    parts = sources.split_alphaseq_scfv("QVQLQESGGGLVQAGGSLRLSCAAS")
    assert parts == ("QVQLQESGGGLVQAGGSLRLSCAAS", "", "heavy_only")


def test_two_linkers_do_not_identify_a_boundary_and_are_refused():
    assert sources.split_alphaseq_scfv("AAA" + LINKER + "BBB" + LINKER + "CCC") is None


def test_the_declared_non_her2_orientations_match_the_release_table():
    assert sources.ALPHABIND_NONHER2["alphabind_vhh72"]["orientation"] == "heavy_only"
    for name in ("alphabind_aab_pp489", "alphabind_pembrolizumab"):
        assert sources.ALPHABIND_NONHER2[name]["orientation"] == "paired_scfv"
    targets = {block["target"] for block in sources.ALPHABIND_NONHER2.values()}
    assert targets == {"TIGIT", "PD-1", "SARS-CoV-2 RBD"}


# --- the closure across edit spaces ---------------------------------------

def test_the_radius_rule_does_not_compare_two_different_edit_spaces():
    """A 16-site genotype and a 10-mer HER2 core have no Hamming distance.

    Asking for one raised inside ``encode_windows``, which turned every non-HER2 row
    into a crash rather than an allowed one. The radius rule protects the HER2 panel
    and can say nothing about a different edit space; those rows are protected by
    their own source's declared partition instead.
    """
    closure = split_lib.ExclusionClosure.build(
        reserved_windows=["WGGDGFYAMD"], development_windows=["AAAAAAAAAA"], radius=2)
    assert "within 1 edit(s)" in closure.violates(window="WGGDGFYAMA")
    for other_space in ("0000000000000001", "A" * 68, "ACD"):
        assert closure.nearest_held_out(other_space) == -1
        assert closure.violates(window=other_space) is None
        assert closure.require_allowed(consumer="observation_head_row", window=other_space)
    assert closure.document()["guarded_window_length"] == 10
    assert "different edit spaces" in closure.document()["length_rule"]


def test_a_row_with_no_region_is_allowed_and_not_a_crash():
    closure = split_lib.ExclusionClosure.build(reserved_windows=["WGGDGFYAMD"], radius=2)
    assert closure.violates(window="") is None


# --- declared heldouts ----------------------------------------------------

def test_a_declared_withheld_parent_moves_its_rows_out_of_training():
    rows = [observation("ACDEFGHIKL", "KLIHGFEDCA", source="probe"),
            observation("WCDEFGHIKL", "KLIHGFEDCA", source="other")]
    moved, record = sources.apply_declared_heldouts(
        rows, withheld_parents=["probe::parental"])
    assert record["rows_moved"]["parent"] == 1
    held = [row for row in moved if row.raw.get("partition") == "development"]
    assert len(held) == 1 and held[0].raw["withheld_as"] == "parent"
    assert [row for row in moved if not row.raw.get("partition")], "the rest are untouched"


def test_a_declared_withheld_study_moves_every_row_of_it():
    rows = [observation("ACDEFGHIKL", "KLIHGFEDCA", source="probe"),
            observation("WCDEFGHIKL", "KLIHGFEDCA", source="probe")]
    moved, record = sources.apply_declared_heldouts(rows, withheld_studies=["probe"])
    assert record["rows_moved"]["study"] == 2
    assert all(row.raw["partition"] == "development" for row in moved)
    assert "chosen after the fit" in record["rule"]


# --- the paired OAS keys --------------------------------------------------

def test_the_oas_loader_reads_the_keys_the_release_actually_carries(tmp_path):
    """The earlier reader asked for ``heavy``/``light`` and enrolled zero rows."""
    import gzip
    import json
    root = tmp_path / "oas_paired_all"
    root.mkdir()
    target = root / sources.OAS_MEMBER
    records = []
    for index in range(40):
        records.append({"sequence_heavy": "ACDEFGHIKL", "sequence_light": "KLIHGFEDCA",
                        "split": "train" if index % 4 else "val",
                        "pair_id": f"p{index}", "source_file": "f.csv",
                        "metadata": {"author": "a", "run": f"r{index % 3}",
                                     "subject": "s"}})
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    # A decoy: the duplicate population. Globbing the directory would double every row.
    (root / "oas_all.jsonl.gz").write_bytes(target.read_bytes())
    block = sources.load_paired_oas(root, limit=5, splits=("train", "val"), seed=1)
    assert block["rows_in_file"] == 40
    assert set(block["by_split"]) == {"train", "val"}
    assert len(block["by_split"]["train"]) == 5, "the cap is per split"
    assert len(block["by_split"]["val"]) == 5
    assert block["rows_seen_by_split"]["train"] == 30
    # author/run/subject is retained as a study identity. How many of the three runs
    # land in a ten-row reservoir sample is a property of the sample, so what is
    # asserted is the identity FORM rather than a count the seed happens to produce.
    assert 1 <= block["studies"] <= 3
    assert all(name.count("/") == 2 for name in block["top_studies"])
    assert block["member"] == sources.OAS_MEMBER
    assert "reservoir per split" in block["sampling"]


def test_the_oas_cache_is_keyed_on_the_parser_and_the_cap(tmp_path):
    import gzip
    import json
    root = tmp_path / "oas_paired_all"
    root.mkdir()
    with gzip.open(root / sources.OAS_MEMBER, "wt", encoding="utf-8") as stream:
        for index in range(10):
            stream.write(json.dumps({"sequence_heavy": "ACDEFGHIKL",
                                     "sequence_light": "KLIHGFEDCA", "split": "train",
                                     "pair_id": f"p{index}",
                                     "metadata": {"author": "a"}}) + "\n")
    cache = tmp_path / "cache"
    first = sources.load_paired_oas(root, limit=4, splits=("train",), cache=cache, seed=1)
    second = sources.load_paired_oas(root, limit=4, splits=("train",), cache=cache, seed=1)
    assert second.get("reused_cache") is True
    assert [c.identity() for c in second["constructs"]] == \
        [c.identity() for c in first["constructs"]]
    # A different cap is a different sample, so the cache must not answer for it.
    third = sources.load_paired_oas(root, limit=2, splits=("train",), cache=cache, seed=1)
    assert third.get("reused_cache") is not True

"""Fixture-only tests for the J05a benchmark provenance schemas.

Nothing here touches the network or any real download. The committed manifests
under ``specs/benchmarks/`` are read as *text fixtures*: they are tracked files,
and the point of reading them is to prove that an unfilled manifest is rejected
by the same validator that would accept an owner-approved one.

The three properties worth protecting are:

1. an unfilled ``TODO(owner)`` field can never be mistaken for an approved value,
2. a UniProt full-length sequence can never be read as the assayed construct, and
3. a non-binding or censored outcome can never be read as a point measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from smallAntibodyGen.benchmarks import provenance as prov


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def manifest_dir(project_root: Path) -> Path:
    return project_root.parents[1] / "specs" / "benchmarks"


@pytest.fixture
def approved_manifest_dict() -> dict:
    """A structurally complete, owner-approved manifest (synthetic)."""
    return {
        "schema_version": prov.SCHEMA_VERSION,
        "dataset_name": "fixture_landscape",
        "release_version": "v1.2.0",
        "source_url": "https://example.invalid/fixture/v1.2.0",
        "license": "CC-BY-4.0",
        "retrieval_date": "2026-01-31",
        "files": [
            {
                "relative_path": "fixture/genotypes.csv",
                "size_bytes": 12,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "fixture/measurements.csv",
                "size_bytes": 34,
                "sha256": "b" * 64,
            },
        ],
        "candidate_source_url": None,
        "candidate_source_url_verified": False,
        "plan_assertions": [],
        "owner_decisions": [],
        "notes": "",
    }


@pytest.fixture
def exact_construct() -> prov.ExactAssayedConstruct:
    return prov.ExactAssayedConstruct(
        sequence="MKTAYIAKQRQISFVKSHFSRQ",
        source_kind=prov.CONSTRUCT_SOURCE_RELEASE_SUPPLEMENTARY_FILE,
        source_locator="elife-71393-supp1.xlsx#sheet=constructs",
    )


# --------------------------------------------------------------------------
# 1. manifest schema
# --------------------------------------------------------------------------

def test_validate_source_manifest_accepts_a_complete_manifest(approved_manifest_dict):
    manifest = prov.validate_source_manifest(approved_manifest_dict)
    assert manifest.dataset_name == "fixture_landscape"
    assert manifest.release_version == "v1.2.0"
    assert len(manifest.files) == 2
    assert manifest.files[0].relative_path == "fixture/genotypes.csv"


def test_validate_source_manifest_rejects_the_owner_sentinel(approved_manifest_dict):
    approved_manifest_dict["license"] = prov.TODO_OWNER
    with pytest.raises(prov.UnsuppliedOwnerDecisionError) as exc:
        prov.validate_source_manifest(approved_manifest_dict)
    assert "license" in str(exc.value)


def test_owner_sentinel_is_rejected_wherever_it_is_nested(approved_manifest_dict):
    approved_manifest_dict["files"][1]["sha256"] = prov.TODO_OWNER
    with pytest.raises(prov.UnsuppliedOwnerDecisionError) as exc:
        prov.validate_source_manifest(approved_manifest_dict)
    assert "files[1].sha256" in str(exc.value)


def test_validate_source_manifest_rejects_missing_and_unknown_keys(approved_manifest_dict):
    dropped = dict(approved_manifest_dict)
    dropped.pop("license")
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(dropped)

    extra = dict(approved_manifest_dict)
    extra["licence"] = "CC-BY-4.0"
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(extra)


@pytest.mark.parametrize(
    "bad_hash",
    ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65],
)
def test_validate_source_manifest_rejects_malformed_hashes(approved_manifest_dict, bad_hash):
    approved_manifest_dict["files"][0]["sha256"] = bad_hash
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


@pytest.mark.parametrize(
    "bad_path",
    ["/absolute/path.csv", "../escape.csv", "nested/../escape.csv", "", "trailing/"],
)
def test_validate_source_manifest_rejects_unsafe_relative_paths(approved_manifest_dict, bad_path):
    approved_manifest_dict["files"][0]["relative_path"] = bad_path
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


def test_validate_source_manifest_rejects_duplicate_paths(approved_manifest_dict):
    approved_manifest_dict["files"][1]["relative_path"] = (
        approved_manifest_dict["files"][0]["relative_path"]
    )
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


def test_validate_source_manifest_rejects_empty_file_list(approved_manifest_dict):
    approved_manifest_dict["files"] = []
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


@pytest.mark.parametrize("bad_size", [-1, 1.5, "12", True])
def test_validate_source_manifest_rejects_bad_sizes(approved_manifest_dict, bad_size):
    approved_manifest_dict["files"][0]["size_bytes"] = bad_size
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


@pytest.mark.parametrize("bad_date", ["31-01-2026", "2026-1-31", "2026-13-01", "today", ""])
def test_validate_source_manifest_rejects_non_iso_dates(approved_manifest_dict, bad_date):
    approved_manifest_dict["retrieval_date"] = bad_date
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


def test_validator_rejects_rather_than_repairs(approved_manifest_dict):
    """Whitespace and case are not silently normalized away."""
    approved_manifest_dict["files"][0]["sha256"] = " " + "a" * 64
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


def test_candidate_url_must_be_marked_unverified(approved_manifest_dict):
    approved_manifest_dict["candidate_source_url"] = "https://example.invalid/paper"
    approved_manifest_dict["candidate_source_url_verified"] = True
    with pytest.raises(prov.ManifestValidationError):
        prov.validate_source_manifest(approved_manifest_dict)


# --------------------------------------------------------------------------
# 2. structural parse vs. strict validation
# --------------------------------------------------------------------------

def test_parse_reports_unsupplied_fields_without_raising(approved_manifest_dict):
    approved_manifest_dict["release_version"] = prov.TODO_OWNER
    approved_manifest_dict["license"] = prov.TODO_OWNER
    doc = prov.parse_manifest_document(approved_manifest_dict)
    assert doc.is_approved is False
    assert doc.unsupplied_fields == ("license", "release_version")


def test_parse_of_a_complete_manifest_is_approved(approved_manifest_dict):
    doc = prov.parse_manifest_document(approved_manifest_dict)
    assert doc.is_approved is True
    assert doc.unsupplied_fields == ()
    assert doc.validated().dataset_name == "fixture_landscape"


def test_parse_still_rejects_structural_damage(approved_manifest_dict):
    approved_manifest_dict.pop("files")
    with pytest.raises(prov.ManifestValidationError):
        prov.parse_manifest_document(approved_manifest_dict)


# --------------------------------------------------------------------------
# 3. committed manifest templates
# --------------------------------------------------------------------------

def test_every_committed_manifest_parses_structurally(manifest_dir):
    paths = sorted(manifest_dir.glob("*.json"))
    assert [p.name for p in paths] == [
        "avida_hil6.json",
        "cr9114_cr6261_landscape.json",
        "open_alphaseq.json",
    ]
    for path in paths:
        doc = prov.load_manifest_document(path)
        assert doc.dataset_name == path.stem


def test_every_committed_manifest_is_unapproved_and_rejected(manifest_dir):
    for path in sorted(manifest_dir.glob("*.json")):
        doc = prov.load_manifest_document(path)
        assert doc.is_approved is False, f"{path.name} looks approved but the owner has not signed off"
        assert doc.unsupplied_fields, path.name
        with pytest.raises(prov.UnsuppliedOwnerDecisionError):
            prov.validate_source_manifest(doc.raw)


def test_committed_manifests_record_no_invented_hashes(manifest_dir):
    for path in sorted(manifest_dir.glob("*.json")):
        doc = prov.load_manifest_document(path)
        assert doc.raw["files"] == [], f"{path.name} must not carry hashes nobody computed"


def test_committed_manifests_mark_candidate_urls_unverified(manifest_dir):
    for path in sorted(manifest_dir.glob("*.json")):
        doc = prov.load_manifest_document(path)
        assert doc.raw["candidate_source_url_verified"] is False
        assert doc.raw["source_url"] == prov.TODO_OWNER


def test_committed_manifests_are_in_canonical_form(manifest_dir):
    """Sorted keys, two-space indent, trailing newline -- byte-stable on rewrite."""
    for path in sorted(manifest_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == prov.dumps_manifest(raw)


def test_committed_manifests_list_open_owner_decisions(manifest_dir):
    for path in sorted(manifest_dir.glob("*.json")):
        doc = prov.load_manifest_document(path)
        decisions = doc.raw["owner_decisions"]
        assert decisions, path.name
        for decision in decisions:
            assert set(decision) == {"key", "question", "status"}
            assert decision["status"] == "unsupplied"


# --------------------------------------------------------------------------
# 4. hashing helpers (tiny temp files, never a real download)
# --------------------------------------------------------------------------

def test_sha256_file_is_chunked_and_matches_hashlib(tmp_path):
    import hashlib

    payload = b"benchmark-bytes" * 1000
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert prov.sha256_file(target, chunk_size=64) == hashlib.sha256(payload).hexdigest()


def test_file_entry_for_builds_a_manifest_entry(tmp_path):
    root = tmp_path / "raw"
    (root / "sub").mkdir(parents=True)
    target = root / "sub" / "a.csv"
    target.write_bytes(b"abc")
    entry = prov.file_entry_for(root, target)
    assert entry.relative_path == "sub/a.csv"
    assert entry.size_bytes == 3
    assert entry.sha256 == prov.sha256_file(target)


def test_verify_manifest_files_reports_every_mismatch(tmp_path, approved_manifest_dict):
    root = tmp_path / "raw"
    (root / "fixture").mkdir(parents=True)
    (root / "fixture" / "genotypes.csv").write_bytes(b"abc")
    # measurements.csv is deliberately absent.
    approved_manifest_dict["files"][0]["size_bytes"] = 3
    manifest = prov.validate_source_manifest(approved_manifest_dict)
    problems = prov.verify_manifest_files(manifest, root)
    joined = " | ".join(problems)
    assert "fixture/genotypes.csv" in joined and "sha256" in joined
    assert "fixture/measurements.csv" in joined and "missing" in joined


# --------------------------------------------------------------------------
# 5. assayed construct: no silent UniProt substitution
# --------------------------------------------------------------------------

def test_exact_construct_exposes_its_sequence(exact_construct):
    assert exact_construct.kind == prov.CONSTRUCT_KIND_EXACT
    assert prov.assayed_construct_sequence(exact_construct) == exact_construct.sequence
    assert prov.require_assayed_construct_sequence(exact_construct)


def test_exact_construct_rejects_a_non_authoritative_source():
    with pytest.raises(prov.ConstructProvenanceError) as exc:
        prov.ExactAssayedConstruct(
            sequence="MKTAYIAK",
            source_kind="uniprot_full_length",
            source_locator="P0DTC2",
        )
    assert "uniprot" in str(exc.value).lower()


def test_registered_null_construct_has_no_sequence_attribute_at_all():
    null = prov.RegisteredNullConstruct(
        reason_code=prov.NULL_REASON_CONSTRUCT_NOT_RELEASED,
        detail="eLife 71393 supplementary files list genotypes, not the assayed HA construct.",
    )
    assert null.kind == prov.CONSTRUCT_KIND_REGISTERED_NULL
    assert not hasattr(null, "sequence")
    with pytest.raises(AttributeError):
        _ = null.sequence  # type: ignore[attr-defined]
    assert prov.assayed_construct_sequence(null) is None
    with pytest.raises(prov.RegisteredNullError):
        prov.require_assayed_construct_sequence(null)


def test_registered_null_requires_a_known_reason_and_detail():
    with pytest.raises(prov.ConstructProvenanceError):
        prov.RegisteredNullConstruct(reason_code="because", detail="x")
    with pytest.raises(prov.ConstructProvenanceError):
        prov.RegisteredNullConstruct(
            reason_code=prov.NULL_REASON_CONSTRUCT_NOT_RELEASED, detail="  "
        )


def test_a_reference_sequence_is_not_a_construct_and_cannot_stand_in(exact_construct):
    reference = prov.ReferenceSequence(
        sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
        role=prov.REFERENCE_ROLE_UNIPROT_FULL_LENGTH,
        accession="P03452",
    )
    # It carries a sequence, but it is a different type with a different accessor.
    assert reference.sequence
    with pytest.raises(TypeError):
        prov.assayed_construct_sequence(reference)  # type: ignore[arg-type]

    with pytest.raises(prov.ConstructProvenanceError):
        prov.AntigenProvenance(antigen_id="HA", construct=reference)  # type: ignore[arg-type]


def test_reference_sequences_travel_beside_a_registered_null_without_filling_it():
    null = prov.RegisteredNullConstruct(
        reason_code=prov.NULL_REASON_CONSTRUCT_NOT_RELEASED,
        detail="not released",
    )
    antigen = prov.AntigenProvenance(
        antigen_id="HA",
        construct=null,
        reference_sequences=(
            prov.ReferenceSequence(
                sequence="MKTAYIAK",
                role=prov.REFERENCE_ROLE_UNIPROT_FULL_LENGTH,
                accession="P03452",
            ),
        ),
    )
    assert antigen.has_assayed_construct is False
    assert antigen.assayed_sequence_or_none() is None
    assert antigen.registered_null_reason() == prov.NULL_REASON_CONSTRUCT_NOT_RELEASED
    # The reference is reachable only under its own name.
    assert antigen.reference_sequences[0].role == prov.REFERENCE_ROLE_UNIPROT_FULL_LENGTH


def test_reference_role_must_not_claim_to_be_the_construct():
    with pytest.raises(prov.ConstructProvenanceError):
        prov.ReferenceSequence(
            sequence="MKT",
            role=prov.CONSTRUCT_SOURCE_RELEASE_SUPPLEMENTARY_FILE,
            accession="x",
        )


def test_construct_sequence_must_be_plausible_residues():
    with pytest.raises(prov.ConstructProvenanceError):
        prov.ExactAssayedConstruct(
            sequence="MKT-AYI",
            source_kind=prov.CONSTRUCT_SOURCE_RELEASE_SUPPLEMENTARY_FILE,
            source_locator="f",
        )
    with pytest.raises(prov.ConstructProvenanceError):
        prov.ExactAssayedConstruct(
            sequence="",
            source_kind=prov.CONSTRUCT_SOURCE_RELEASE_SUPPLEMENTARY_FILE,
            source_locator="f",
        )


# --------------------------------------------------------------------------
# 6. epitope masks
# --------------------------------------------------------------------------

def test_epitope_mask_requires_a_cited_source(exact_construct):
    with pytest.raises(prov.EpitopeProvenanceError):
        prov.EpitopeMask(residue_positions=(1, 2), position_basis=prov.POSITION_BASIS_CONSTRUCT)


def test_epitope_mask_accepts_experimental_and_structural_evidence(exact_construct):
    mask = prov.EpitopeMask(
        residue_positions=(5, 3, 9),
        position_basis=prov.POSITION_BASIS_CONSTRUCT,
        evidence=prov.EvidenceCitation(
            kind=prov.EVIDENCE_KIND_STRUCTURAL,
            identifier="PDB 4FQI",
            description="contact residues within 4.5 A",
        ),
    )
    assert mask.residue_positions == (3, 5, 9)  # sorted, deduplicated, deterministic


def test_epitope_evidence_rejects_an_uncited_kind():
    with pytest.raises(prov.EpitopeProvenanceError):
        prov.EvidenceCitation(kind="guessed", identifier="x", description="y")
    with pytest.raises(prov.EpitopeProvenanceError):
        prov.EvidenceCitation(kind=prov.EVIDENCE_KIND_EXPERIMENTAL, identifier="", description="y")


def test_epitope_mask_is_provenance_only_and_not_a_model_input_mode(exact_construct):
    """The mask must not carry encoding/tokenization fields; it stays annotation."""
    mask = prov.EpitopeMask(
        residue_positions=(1,),
        position_basis=prov.POSITION_BASIS_CONSTRUCT,
        evidence=prov.EvidenceCitation(
            kind=prov.EVIDENCE_KIND_EXPERIMENTAL,
            identifier="doi:10.0000/x",
            description="alanine scan",
        ),
    )
    field_names = {f for f in mask.__dataclass_fields__}
    assert field_names == {"residue_positions", "position_basis", "evidence"}
    antigen = prov.AntigenProvenance(
        antigen_id="HA", construct=exact_construct, epitope_masks=(mask,)
    )
    # Adding a mask changes nothing about the construct the model would condition on.
    assert antigen.assayed_sequence_or_none() == exact_construct.sequence


def test_epitope_mask_positions_must_be_positive_and_in_range(exact_construct):
    evidence = prov.EvidenceCitation(
        kind=prov.EVIDENCE_KIND_EXPERIMENTAL, identifier="doi:x", description="d"
    )
    with pytest.raises(prov.EpitopeProvenanceError):
        prov.EpitopeMask(
            residue_positions=(0,), position_basis=prov.POSITION_BASIS_CONSTRUCT, evidence=evidence
        )
    long_mask = prov.EpitopeMask(
        residue_positions=(999,), position_basis=prov.POSITION_BASIS_CONSTRUCT, evidence=evidence
    )
    with pytest.raises(prov.EpitopeProvenanceError):
        prov.AntigenProvenance(
            antigen_id="HA", construct=exact_construct, epitope_masks=(long_mask,)
        )


def test_epitope_mask_against_a_registered_null_may_not_use_construct_numbering():
    evidence = prov.EvidenceCitation(
        kind=prov.EVIDENCE_KIND_STRUCTURAL, identifier="PDB 1XYZ", description="d"
    )
    null = prov.RegisteredNullConstruct(
        reason_code=prov.NULL_REASON_CONSTRUCT_NOT_RELEASED, detail="not released"
    )
    with pytest.raises(prov.EpitopeProvenanceError):
        prov.AntigenProvenance(
            antigen_id="HA",
            construct=null,
            epitope_masks=(
                prov.EpitopeMask(
                    residue_positions=(4,),
                    position_basis=prov.POSITION_BASIS_CONSTRUCT,
                    evidence=evidence,
                ),
            ),
        )
    # Reference numbering is fine: it names a sequence that actually exists.
    antigen = prov.AntigenProvenance(
        antigen_id="HA",
        construct=null,
        epitope_masks=(
            prov.EpitopeMask(
                residue_positions=(4,),
                position_basis=prov.POSITION_BASIS_UNIPROT,
                evidence=evidence,
            ),
        ),
    )
    assert antigen.has_assayed_construct is False


# --------------------------------------------------------------------------
# 7. censoring
# --------------------------------------------------------------------------

def test_point_measurement_round_trips():
    m = prov.Measurement.point(2.5)
    assert m.censoring == prov.CENSORING_NONE
    assert m.value == 2.5
    assert m.is_censored is False


def test_non_binding_is_representable_and_carries_no_value():
    m = prov.Measurement.non_binding(detection_limit=1000.0)
    assert m.censoring == prov.CENSORING_NON_BINDING
    assert m.value is None
    assert m.is_censored is True
    assert m.is_non_binding is True


def test_non_binding_may_not_carry_a_fake_kd():
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring=prov.CENSORING_NON_BINDING, value=1e-6)


def test_left_and_right_censoring_require_the_matching_bound():
    left = prov.Measurement.left_censored(upper_bound=0.1)
    assert left.value is None and left.upper_bound == 0.1 and left.lower_bound is None
    right = prov.Measurement.right_censored(lower_bound=1000.0)
    assert right.value is None and right.lower_bound == 1000.0

    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring=prov.CENSORING_LEFT, lower_bound=1.0)
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring=prov.CENSORING_RIGHT, upper_bound=1.0)


def test_interval_requires_both_bounds_in_order():
    interval = prov.Measurement.interval(1.0, 2.0)
    assert interval.lower_bound == 1.0 and interval.upper_bound == 2.0
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement.interval(2.0, 1.0)
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring=prov.CENSORING_INTERVAL, lower_bound=1.0)


def test_a_point_measurement_may_not_smuggle_bounds():
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring=prov.CENSORING_NONE, value=1.0, upper_bound=2.0)
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring=prov.CENSORING_NONE)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_measurements_are_rejected(bad):
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement.point(bad)


def test_unknown_censoring_mode_is_rejected():
    with pytest.raises(prov.MeasurementValidationError):
        prov.Measurement(censoring="maybe", value=1.0)


# --------------------------------------------------------------------------
# 8. replicates
# --------------------------------------------------------------------------

def _spec(direction: str = prov.DIRECTION_HIGHER_IS_STRONGER) -> prov.AssaySpec:
    return prov.AssaySpec(
        assay_name="fixture_assay", quantity="neg_log10_kd", unit="log10_M", direction=direction
    )


def test_replicates_are_retained_individually():
    measurement = prov.AssayMeasurement(
        assay=_spec(),
        replicates=(
            prov.Replicate("rep1", prov.Measurement.point(9.0)),
            prov.Replicate("rep2", prov.Measurement.point(9.4)),
            prov.Replicate("rep3", prov.Measurement.non_binding()),
        ),
    )
    assert len(measurement.replicates) == 3
    assert [r.replicate_id for r in measurement.replicates] == ["rep1", "rep2", "rep3"]
    assert measurement.has_censored_replicate is True
    # No aggregate is computed or required by the schema.
    assert not hasattr(measurement, "mean")


def test_replicate_ids_must_be_unique_and_non_empty():
    with pytest.raises(prov.MeasurementValidationError):
        prov.AssayMeasurement(
            assay=_spec(),
            replicates=(
                prov.Replicate("rep1", prov.Measurement.point(1.0)),
                prov.Replicate("rep1", prov.Measurement.point(2.0)),
            ),
        )
    with pytest.raises(prov.MeasurementValidationError):
        prov.AssayMeasurement(
            assay=_spec(), replicates=(prov.Replicate("", prov.Measurement.point(1.0)),)
        )


def test_at_least_one_replicate_is_required():
    with pytest.raises(prov.MeasurementValidationError):
        prov.AssayMeasurement(assay=_spec(), replicates=())


def test_a_source_published_aggregate_must_say_so_and_stand_alone():
    aggregate = prov.Replicate(
        "source_aggregate",
        prov.Measurement.point(9.1),
        source_aggregate=True,
        aggregation_method="mean_of_3_unreleased_replicates",
    )
    measurement = prov.AssayMeasurement(assay=_spec(), replicates=(aggregate,))
    assert measurement.is_source_aggregate_only is True

    with pytest.raises(prov.MeasurementValidationError):
        prov.Replicate("x", prov.Measurement.point(1.0), source_aggregate=True)

    with pytest.raises(prov.MeasurementValidationError):
        prov.AssayMeasurement(
            assay=_spec(),
            replicates=(aggregate, prov.Replicate("rep1", prov.Measurement.point(9.0))),
        )


# --------------------------------------------------------------------------
# 9. assay direction normalized once
# --------------------------------------------------------------------------

def test_direction_must_be_declared_and_known():
    with pytest.raises(prov.MeasurementValidationError):
        prov.AssaySpec(assay_name="a", quantity="q", unit="u", direction="lower_better")


def test_higher_is_stronger_passes_through_and_retains_the_raw_value():
    spec = _spec(prov.DIRECTION_HIGHER_IS_STRONGER)
    normalized = prov.normalize_measurement(prov.Measurement.point(9.0), spec.direction)
    assert normalized.strength == 9.0
    assert normalized.raw.value == 9.0
    assert normalized.raw_direction == prov.DIRECTION_HIGHER_IS_STRONGER


def test_lower_is_stronger_is_negated_once_and_the_raw_value_survives():
    spec = _spec(prov.DIRECTION_LOWER_IS_STRONGER)
    normalized = prov.normalize_measurement(prov.Measurement.point(3.2), spec.direction)
    assert normalized.strength == -3.2
    assert normalized.raw.value == 3.2
    assert normalized.raw_direction == prov.DIRECTION_LOWER_IS_STRONGER
    # Normalizing is idempotent in the sense that the normalized view is already
    # higher-is-stronger; re-normalizing it is a programming error.
    with pytest.raises(prov.MeasurementValidationError):
        prov.normalize_measurement(normalized, spec.direction)  # type: ignore[arg-type]


def test_direction_flip_swaps_the_censoring_side():
    """A right-censored KD (weak) is a left-censored strength (weak). Sides must swap."""
    raw = prov.Measurement.right_censored(lower_bound=1000.0)  # KD > 1000 nM
    normalized = prov.normalize_measurement(raw, prov.DIRECTION_LOWER_IS_STRONGER)
    assert normalized.censoring == prov.CENSORING_LEFT
    assert normalized.upper_bound == -1000.0
    assert normalized.lower_bound is None
    assert normalized.strength is None


def test_direction_flip_swaps_interval_bounds():
    raw = prov.Measurement.interval(1.0, 4.0)
    normalized = prov.normalize_measurement(raw, prov.DIRECTION_LOWER_IS_STRONGER)
    assert (normalized.lower_bound, normalized.upper_bound) == (-4.0, -1.0)


def test_non_binding_stays_non_binding_under_either_direction():
    for direction in (prov.DIRECTION_HIGHER_IS_STRONGER, prov.DIRECTION_LOWER_IS_STRONGER):
        normalized = prov.normalize_measurement(prov.Measurement.non_binding(), direction)
        assert normalized.censoring == prov.CENSORING_NON_BINDING
        assert normalized.strength is None
        assert normalized.is_non_binding is True


def test_normalized_replicate_strengths_are_ordered_like_the_replicates():
    measurement = prov.AssayMeasurement(
        assay=_spec(prov.DIRECTION_LOWER_IS_STRONGER),
        replicates=(
            prov.Replicate("rep1", prov.Measurement.point(1.0)),
            prov.Replicate("rep2", prov.Measurement.point(5.0)),
        ),
    )
    strengths = [n.strength for n in measurement.normalized()]
    assert strengths == [-1.0, -5.0]


# --------------------------------------------------------------------------
# 10. assay provenance record
# --------------------------------------------------------------------------

def _record(antigen: prov.AntigenProvenance) -> prov.AssayProvenanceRecord:
    return prov.AssayProvenanceRecord(
        record_id="r0",
        dataset_name="fixture_landscape",
        binder_id="b0",
        binder_sequence="EVQLVESGGG",
        antigen=antigen,
        measurements=(
            prov.AssayMeasurement(
                assay=_spec(), replicates=(prov.Replicate("rep1", prov.Measurement.point(8.0)),)
            ),
        ),
    )


def test_record_distinguishes_registered_null_from_a_real_sequence(exact_construct):
    with_sequence = _record(prov.AntigenProvenance(antigen_id="HA", construct=exact_construct))
    assert with_sequence.conditioning_status == prov.CONDITIONING_EXACT_CONSTRUCT
    assert with_sequence.antigen.assayed_sequence_or_none() == exact_construct.sequence

    without = _record(
        prov.AntigenProvenance(
            antigen_id="HA",
            construct=prov.RegisteredNullConstruct(
                reason_code=prov.NULL_REASON_CONSTRUCT_NOT_RELEASED, detail="not released"
            ),
        )
    )
    assert without.conditioning_status == prov.CONDITIONING_REGISTERED_NULL
    assert without.antigen.assayed_sequence_or_none() is None
    assert without.antigen.registered_null_reason() == prov.NULL_REASON_CONSTRUCT_NOT_RELEASED


def test_record_requires_at_least_one_measurement(exact_construct):
    with pytest.raises(prov.MeasurementValidationError):
        prov.AssayProvenanceRecord(
            record_id="r0",
            dataset_name="d",
            binder_id="b",
            binder_sequence="EVQ",
            antigen=prov.AntigenProvenance(antigen_id="HA", construct=exact_construct),
            measurements=(),
        )


def test_record_rejects_a_bare_string_antigen(exact_construct):
    with pytest.raises(prov.ConstructProvenanceError):
        prov.AssayProvenanceRecord(
            record_id="r0",
            dataset_name="d",
            binder_id="b",
            binder_sequence="EVQ",
            antigen="MKTAYIAK",  # type: ignore[arg-type]
            measurements=(
                prov.AssayMeasurement(
                    assay=_spec(),
                    replicates=(prov.Replicate("rep1", prov.Measurement.point(1.0)),),
                ),
            ),
        )


def test_record_is_json_round_trippable_and_deterministic(exact_construct):
    record = _record(prov.AntigenProvenance(antigen_id="HA", construct=exact_construct))
    encoded = prov.record_to_json_dict(record)
    assert prov.dumps_manifest(encoded) == prov.dumps_manifest(
        prov.record_to_json_dict(prov.record_from_json_dict(encoded))
    )
    assert prov.record_from_json_dict(encoded) == record


def test_json_round_trip_preserves_a_registered_null_reason():
    record = _record(
        prov.AntigenProvenance(
            antigen_id="HA",
            construct=prov.RegisteredNullConstruct(
                reason_code=prov.NULL_REASON_AMBIGUOUS_RELEASE, detail="two candidate constructs"
            ),
        )
    )
    restored = prov.record_from_json_dict(prov.record_to_json_dict(record))
    assert restored.conditioning_status == prov.CONDITIONING_REGISTERED_NULL
    assert restored.antigen.registered_null_reason() == prov.NULL_REASON_AMBIGUOUS_RELEASE


def test_json_round_trip_preserves_censoring_and_replicates():
    record = prov.AssayProvenanceRecord(
        record_id="r1",
        dataset_name="d",
        binder_id="b",
        binder_sequence="EVQ",
        antigen=prov.AntigenProvenance(
            antigen_id="HA",
            construct=prov.RegisteredNullConstruct(
                reason_code=prov.NULL_REASON_CONSTRUCT_NOT_RELEASED, detail="x"
            ),
        ),
        measurements=(
            prov.AssayMeasurement(
                assay=_spec(prov.DIRECTION_LOWER_IS_STRONGER),
                replicates=(
                    prov.Replicate("rep1", prov.Measurement.right_censored(lower_bound=1000.0)),
                    prov.Replicate("rep2", prov.Measurement.non_binding(detection_limit=1000.0)),
                ),
            ),
        ),
    )
    restored = prov.record_from_json_dict(prov.record_to_json_dict(record))
    assert restored == record
    assert restored.measurements[0].replicates[1].measurement.is_non_binding is True


def test_module_never_imports_a_network_client():
    import inspect

    source = inspect.getsource(prov)
    for forbidden in ("urllib", "requests", "http.client", "socket", "urlopen"):
        assert forbidden not in source, f"provenance.py must not reach the network ({forbidden})"

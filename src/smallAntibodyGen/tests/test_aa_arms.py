from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments.interaction_qualification import arms as arms_module
from smallAntibodyGen.experiments.interaction_qualification import records as rec
from smallAntibodyGen.experiments.interaction_qualification.arms import (
    ARM_NAMES,
    ArmFeatures,
    ArmMismatchError,
    FullViewMemo,
    admit_by_joint_length,
    arm_report,
    assemble_matrix,
    common_ids,
    extract_arm_features,
    feature_views,
    joint_serialized_length,
    planned_feature_forwards,
)
from smallAntibodyGen.models.antibody_antigen import views as views_module
from smallAntibodyGen.models.antibody_antigen.denoising import N_CANONICAL, BackendOutput
from smallAntibodyGen.models.antibody_antigen.views import VisibleView, mask_complete
from smallAntibodyGen.tests import aa_qualification_world as world
from smallAntibodyGen.tests.aa_world import RecordingBackend, ToyBackend


class RampBackend:
    """Row i of a view gets features [i, n_rows]: pooled means have closed forms and depend on the view's size."""

    def forward(self, view):
        coords = view.coords()
        n = len(coords)
        return BackendOutput(
            coords=coords,
            raw_logits=torch.zeros(n, N_CANONICAL),
            canonical_mass=torch.full((n,), 0.9),
            features=torch.tensor([[float(i), float(n)] for i in range(n)], dtype=torch.float32),
            mask_ratio=len(view.hidden_sites()) / n,
        )


class GradModeBackend:
    def __init__(self, inner):
        self.inner = inner
        self.grad_enabled = []

    def forward(self, view):
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.inner.forward(view)


def one(record_id, heavy, light, antigen, **kwargs):
    return rec.parse_record(world.row(record_id, heavy=heavy, light=light, antigen=antigen, **kwargs))


# --- views and length admission --------------------------------------------------------------


def test_feature_views_are_unmasked_and_built_from_sequences_only():
    paired = feature_views(one("a", "ACD", "EF", "GHIK"))
    assert paired.joint.roles == ("H", "L", "Ag") and paired.antibody.roles == ("H", "L")
    assert paired.antigen.roles == ("Ag",)
    assert paired.joint.chain("H") == ("A", "C", "D") and paired.joint.chain("L") == ("E", "F")
    assert paired.antibody.chains == paired.joint.chains[:2] and paired.antigen.chains == paired.joint.chains[2:]
    heavy_only = feature_views(one("b", "ACD", None, "GHIK"))
    assert heavy_only.joint.roles == ("H", "Ag") and heavy_only.antibody.roles == ("H",)
    for views in (paired, heavy_only):
        for view in (views.joint, views.antibody, views.antigen):
            assert view.hidden_sites() == ()
            assert {f.name for f in dataclasses.fields(view)} == {"roles", "chains"}


def test_joint_serialized_length_counts_residues_plus_two_specials_per_chain():
    paired, heavy_only = one("a", "ACD", "EF", "GHIK"), one("b", "ACD", None, "GHIK")
    assert joint_serialized_length(paired) == 9 + 2 * 3
    assert joint_serialized_length(heavy_only) == 7 + 2 * 2
    for record in (paired, heavy_only):
        chains = feature_views(record).joint.chains
        assert joint_serialized_length(record) == sum(len(chain) + 2 for chain in chains)


def test_length_admission_is_one_arm_independent_cap_and_counts_what_it_drops():
    records = [
        one("short", "ACD", None, "GHIK"),
        one("edge", "ACD", "EF", "GHIK"),
        one("long", "ACDEFG", "HIKLMN", "PQRSTVWY"),
    ]
    admission = admit_by_joint_length(records, 15)
    assert [r.record_id for r in admission.eligible] == ["short", "edge"]
    assert admission.excluded == (("long", 26),) and admission.max_tokens == 15
    tighter = admit_by_joint_length(records, 14)
    assert [r.record_id for r in tighter.eligible] == ["short"]
    assert tighter.excluded == (("edge", 15), ("long", 26))
    assert admit_by_joint_length(records, 26).excluded == ()
    for bad in (0, -1, True, 1.5, "15"):
        with pytest.raises(ValueError):
            admit_by_joint_length(records, bad)


# --- the full-view memo ----------------------------------------------------------------------


def test_the_memo_refuses_masked_views_before_any_backend_call():
    spy = RecordingBackend(RampBackend())
    memo = FullViewMemo(spy)
    masked, _ = mask_complete({"H": "ACD", "Ag": "GHIK"}, [("H", 1)])
    with pytest.raises(ValueError, match="masked"):
        memo.pooled(masked)
    with pytest.raises(TypeError):
        memo.pooled("not a view")
    assert spy.views == [] and len(memo) == 0 and memo.forward_calls == 0


def test_each_distinct_visible_content_is_forwarded_exactly_once():
    records = [
        one("a", "ACD", "EF", "GHIK"),
        one("b", "ACD", "EF", "GHIL"),  # same antibody, other antigen
        one("c", "ACD", "EF", "GHIK", source="other", assay="other", value=1.0),  # same sequences as "a"
    ]
    spy = RecordingBackend(RampBackend())
    memo = FullViewMemo(spy)
    assert planned_feature_forwards(records) == 5
    extract_arm_features(records, memo)
    # joint(GHIK), joint(GHIL), antibody-only(ACD+EF) once, antigen-only(GHIK), antigen-only(GHIL)
    assert len(spy.views) == memo.forward_calls == len(memo) == 5
    assert len({view.content_digest() for view in spy.views}) == 5
    extract_arm_features(records, memo)
    assert len(spy.views) == 5
    assert planned_feature_forwards(records, memo) == 0


def test_pooled_features_match_closed_forms_and_the_arm_layout():
    arms = extract_arm_features(
        [one("p", "ACD", "EF", "GHIK"), one("h", "ACD", None, "GHIK")], FullViewMemo(RampBackend())
    )
    # paired: joint rows 0..8 (n=9): antibody rows 0..4 -> mean 2, antigen rows 5..8 -> mean 6.5; sub-views have their own n
    np.testing.assert_allclose(arms["joint"].vectors["p"], [2.0, 9.0, 6.5, 9.0])
    np.testing.assert_allclose(arms["antibody_only"].vectors["p"], [2.0, 5.0])
    np.testing.assert_allclose(arms["independent"].vectors["p"], [2.0, 5.0, 1.5, 4.0])
    # heavy-only: joint rows 0..6 (n=7): antibody rows 0..2 -> mean 1, antigen rows 3..6 -> mean 4.5
    np.testing.assert_allclose(arms["joint"].vectors["h"], [1.0, 7.0, 4.5, 7.0])
    np.testing.assert_allclose(arms["antibody_only"].vectors["h"], [1.0, 3.0])
    np.testing.assert_allclose(arms["independent"].vectors["h"], [1.0, 3.0, 1.5, 4.0])
    assert [arms[name].dimension for name in ARM_NAMES] == [4, 2, 4]
    assert all(arms[name].arm == name for name in ARM_NAMES)


def test_the_memo_keeps_only_pooled_cpu_vectors_and_runs_under_no_grad():
    spy = GradModeBackend(ToyBackend(seed=0, dim=8))
    memo = FullViewMemo(spy)
    extract_arm_features([one("a", "ACD", "EF", "GHIK"), one("b", "ACE", None, "GHIL")], memo)
    assert len(spy.grad_enabled) == memo.forward_calls == len(memo) == 6
    assert not any(spy.grad_enabled)
    assert set(vars(memo)) == {"_backend", "_entries", "forward_calls"}
    absent = 0
    for pooled in memo.entries.values():
        for vector in (pooled.antibody, pooled.antigen):
            if vector is None:
                absent += 1
                continue
            assert vector.shape == (8,) and vector.dtype == torch.float64
            assert vector.device.type == "cpu" and not vector.requires_grad
    # 2 joint views hold both vectors; each of the 2 antibody-only views lacks an antigen vector and each of the
    # 2 antigen-only views lacks an antibody vector
    assert absent == 4
    with pytest.raises(TypeError):
        memo.entries["x"] = None


def test_a_backend_failure_leaves_nothing_cached_or_counted():
    class NonFinite:
        def forward(self, view):
            good = RampBackend().forward(view)
            return BackendOutput(
                coords=good.coords,
                raw_logits=good.raw_logits,
                canonical_mass=good.canonical_mass,
                features=torch.full_like(good.features, float("nan")),
                mask_ratio=good.mask_ratio,
            )

    memo = FullViewMemo(NonFinite())
    view = feature_views(one("a", "ACD", None, "GHIK")).joint
    with pytest.raises(ValueError, match="non-finite"):
        memo.pooled(view)
    assert memo.forward_calls == 0 and len(memo) == 0


# --- what the backend can see ----------------------------------------------------------------


def test_the_backend_sees_only_visible_views_and_the_train_partition_goes_first():
    records = [
        one("dev", "ACE", None, "GHIL", partition="development"),
        one(
            "trn",
            "ACD",
            None,
            "GHIK",
            partition="train",
            source="secret-source",
            assay="secret-assay",
            antibody_group="secret-group",
            value=3.21,
        ),
    ]
    spy = RecordingBackend(RampBackend())
    extract_arm_features(records, FullViewMemo(spy))
    assert len(spy.views) == 6
    for view in spy.views:
        assert type(view) is VisibleView
        text = repr(view)
        for secret in ("trn", "dev", "secret-source", "secret-assay", "secret-group", "3.21"):
            assert secret not in text
    train_views = feature_views(records[1])
    train_digests = {v.content_digest() for v in (train_views.joint, train_views.antibody, train_views.antigen)}
    assert {view.content_digest() for view in spy.views[:3]} == train_digests


def test_heldout_rows_and_row_order_cannot_change_train_features():
    train = [one("t0", "ACD", "EF", "GHIK"), one("t1", "ACDEF", None, "GHIKLMN")]
    heldout = [
        one("d0", "ACDEFGHIKLMNPQ", "RSTVWY", "GHIKLMNPQRSTVWYACDEFG", partition="development", value=1.0),
        one("d1", "AC", "D", "E", partition="test", value=9.0),
    ]
    alone = extract_arm_features(train, FullViewMemo(ToyBackend(seed=3)))
    together = extract_arm_features(train + heldout, FullViewMemo(ToyBackend(seed=3)))
    shuffled = extract_arm_features((train + heldout)[::-1], FullViewMemo(ToyBackend(seed=3)))
    for arm in ARM_NAMES:
        for record in train:
            reference = alone[arm].vectors[record.record_id]
            assert np.array_equal(reference, together[arm].vectors[record.record_id])
            assert np.array_equal(reference, shuffled[arm].vectors[record.record_id])


def test_identifiers_and_labels_cannot_change_features():
    first = [one("x1", "ACD", "EF", "GHIK"), one("x2", "ACE", None, "GHIL")]
    second = [
        one(
            "y9",
            "ACD",
            "EF",
            "GHIK",
            source="s2",
            assay="a2",
            value=1.5,
            antibody_group="gg",
            antigen_group="hh",
            partition="development",
            region=[["H", 0]],
        ),
        one("y8", "ACE", None, "GHIL", source="s3", assay="a3", value=9.9, partition="test"),
    ]
    one_features = extract_arm_features(first, FullViewMemo(ToyBackend(seed=5)))
    two_features = extract_arm_features(second, FullViewMemo(ToyBackend(seed=5)))
    for arm in ARM_NAMES:
        assert np.array_equal(
            assemble_matrix(one_features[arm], ["x1", "x2"]), assemble_matrix(two_features[arm], ["y9", "y8"])
        )


def test_the_independent_arm_reuses_the_antibody_only_vector_and_differs_from_the_joint_one():
    arms = extract_arm_features([one("a", "ACD", "EF", "GHIK")], FullViewMemo(ToyBackend(seed=1, dim=8)))
    joint, antibody, independent = (arms[name].vectors["a"] for name in ARM_NAMES)
    assert [v.shape for v in (joint, antibody, independent)] == [(16,), (8,), (16,)]
    assert np.array_equal(independent[:8], antibody)
    assert not np.allclose(joint[:8], antibody)


# --- feature maps keyed by record id -----------------------------------------------------------


def test_arms_are_joined_by_record_id_whatever_their_insertion_order():
    joint = ArmFeatures("joint", {"b": [1.0, 2.0], "a": [3.0, 4.0]})
    independent = ArmFeatures("independent", {"a": [1.0, 1.0], "b": [2.0, 2.0]})
    antibody = ArmFeatures("antibody_only", {"b": [0.25], "a": [0.5]})
    arms = {"joint": joint, "independent": independent, "antibody_only": antibody}
    assert common_ids(arms) == ("a", "b")
    np.testing.assert_array_equal(assemble_matrix(joint, ["a", "b"]), [[3.0, 4.0], [1.0, 2.0]])
    np.testing.assert_array_equal(assemble_matrix(joint, ["b", "a"]), [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(assemble_matrix(independent, ["a", "b"]), [[1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_array_equal(assemble_matrix(antibody, ["b", "a"]), [[0.25], [0.5]])
    assert assemble_matrix(joint, []).shape == (0, 2)


def test_missing_extra_or_repeated_identities_are_errors_not_intersections():
    joint = ArmFeatures("joint", {"a": [3.0, 4.0], "b": [1.0, 2.0]})
    independent = ArmFeatures("independent", {"a": [1.0, 1.0], "b": [2.0, 2.0]})
    missing = ArmFeatures("antibody_only", {"a": [0.5]})
    extra = ArmFeatures("antibody_only", {"a": [0.5], "b": [0.25], "c": [1.0]})
    for broken in (missing, extra):
        with pytest.raises(ArmMismatchError, match="same record ids"):
            common_ids({"joint": joint, "independent": independent, "antibody_only": broken})
    with pytest.raises(ArmMismatchError, match="holds features"):
        common_ids({"joint": independent})
    with pytest.raises(ValueError):
        common_ids({})
    with pytest.raises(ArmMismatchError, match="no features"):
        assemble_matrix(joint, ["a", "zzz"])
    with pytest.raises(ArmMismatchError, match="unique"):
        assemble_matrix(joint, ["a", "a"])
    with pytest.raises(TypeError):
        assemble_matrix(joint, "ab")


def test_duplicate_record_ids_cannot_be_extracted():
    records = [one("same", "ACD", None, "GHIK"), one("same", "ACE", None, "GHIL")]
    with pytest.raises(ValueError, match="unique"):
        extract_arm_features(records, FullViewMemo(RampBackend()))


def test_arm_features_validate_copy_and_freeze_their_vectors():
    for bad in ({"a": [1.0], "b": [1.0, 2.0]}, {"a": [float("nan")]}, {"a": []}, {"a": [[1.0]]}, {"": [1.0]}):
        with pytest.raises(ValueError):
            ArmFeatures("joint", bad)
    with pytest.raises(ValueError):
        ArmFeatures("mystery", {"a": [1.0]})
    source = np.array([1.0, 2.0])
    features = ArmFeatures("joint", {"a": source})
    source[0] = 99.0
    assert features.vectors["a"][0] == 1.0
    with pytest.raises(ValueError):
        features.vectors["a"][0] = 5.0
    with pytest.raises(TypeError):
        features.vectors["z"] = source
    empty = extract_arm_features([], FullViewMemo(RampBackend()))
    assert common_ids(empty) == () and empty["joint"].dimension == 0


def test_the_arm_report_states_widths_and_that_arms_are_not_capacity_matched():
    arms = extract_arm_features([one("a", "ACD", "EF", "GHIK")], FullViewMemo(RampBackend()))
    report = arm_report(arms)
    assert {name: entry["width"] for name, entry in report["arms"].items()} == {
        "joint": 4,
        "antibody_only": 2,
        "independent": 4,
    }
    assert all(entry["n_records"] == 1 and entry["description"] for entry in report["arms"].values())
    assert "capacity" in report["note"]


# --- layering ---------------------------------------------------------------------------------


def imported_names(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module or "")
    return names


def test_arms_import_nothing_from_tests_or_scripts_and_models_never_import_experiments():
    for name in imported_names(arms_module.__file__):
        assert name.split(".")[0] != "scripts" and not name.startswith("smallAntibodyGen.tests"), name
    for path in Path(views_module.__file__).parent.glob("*.py"):
        assert not any(name.startswith("smallAntibodyGen.experiments") for name in imported_names(path)), path.name

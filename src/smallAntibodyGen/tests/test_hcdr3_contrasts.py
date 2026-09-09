"""Scientific contracts for the frozen ranking/sensitivity benchmark."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from smallAntibodyGen.evaluation.contrasts import (
    build_manifest, concordance, heavy_context, load_json, save_json, verify_manifest,
)
from smallAntibodyGen.evaluation.contrast_exposure import audit_exposure, stage_eligible, variant_identity
from smallAntibodyGen.evaluation.contrast_scoring import score_manifest
from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller
from smallAntibodyGen.tokenizer import AminoAcidTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from hcdr3_contrast_compare import compare_scores


def row(cdr3="AA", value=1.0, *, target="one", light="", prefix="MCC", antigen="AAAA", **extra):
    result = dict(record_id=f"{target}/{prefix}/{cdr3}/{value}/{light}", target=target,
                  sequence_heavy=prefix + cdr3 + "WG", sequence_light=light,
                  sequence_antigen=antigen, cdr3_aa_heavy=cdr3,
                  cdr3_start_aa_heavy=len(prefix), cdr3_end_aa_heavy=len(prefix) + len(cdr3),
                  dataset="alphaseq", affinity_type="alphaseq", processed_measurement_float=value,
                  split="train", is_strong_binder=False)
    result.update(extra)
    return result


def freeze(rows):
    return build_manifest(rows, canonicalize=lambda r: r["target"], provenance={},
                          protocol={"targets": ["one", "two", "pdb:3gbm", "pdb:3gbn"], "seed": 42,
                                    "assays": [{"dataset": "alphaseq", "affinity_type": "alphaseq",
                                                "direction": "lower"}]})


def two_targets():
    return [row(), row("GG", 2.0), row(target="two", antigen="DDDDD"),
            row("GG", 2.0, target="two", antigen="DDDDD")]


def test_order_independent_manifest_and_fixed_context_boundaries(tmp_path):
    rows = two_targets() + [row("YY", 3.0, light="QQ"), row("YY", 3.0, antigen="E"),
                            row("YY", 3.0, source_url="different-assay-source")]
    m = freeze(rows)
    assert m == freeze(list(reversed(rows)))
    assert m["summary"]["groups"] == 2
    assert m["summary"]["rows"] == 4
    assert m["summary"]["all_outside_strong_binder_gate"] == 2
    assert len({g["fold"] for g in m["groups"]}) == 1  # background shared across targets
    assert m["claims"]["historical_holdout_established"] is False
    assert m["claims"]["antigen_conditioning_correctness_established"] is False
    path = tmp_path / "manifest.json.gz"
    save_json(path, m)
    original_bytes = path.read_bytes()
    save_json(path, m)
    assert path.read_bytes() == original_bytes and load_json(path) == m
    tampered = deepcopy(m)
    tampered["groups"][0]["variants"][0]["measurement"] = 90
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(tampered)
    with pytest.raises(FileExistsError):
        save_json(path, tampered)


def test_quantitative_repeats_do_not_silently_select_a_measurement():
    rows = two_targets() + [row(value=8)]
    m = freeze(rows)
    assert m["summary"]["groups"] == 1
    assert m["exclusions"]["rows_in_group_with_conflicting_repeat_measurements"] == 3
    m = freeze(two_targets() + [row(record_id="repeat")])
    g = next(g for g in m["groups"] if g["target"] == "one")
    assert len(g["variants"]) == 2
    assert len(next(v for v in g["variants"] if v["hcdr3"] == "AA")["observation_ids"]) == 2


def test_invalid_binary_and_nonfinite_rows_are_not_graded_variants():
    bad = [row("YY", True), row("YY", float("nan")), row("YY", 4, affinity_type="bool"),
           row("YY", 4, cdr3_start_aa_heavy=0), row("XX", 5)]
    m = freeze(two_targets() + bad)
    assert m["summary"]["rows"] == 4
    old = row()
    old.update(cdr3_start_aa=3, cdr3_end_aa=5, cdr3_aa="AA", cdr3_start_aa_heavy=None)
    assert heavy_context(old)[3] == "AA"
    with pytest.raises(ValueError, match="no groups"):
        freeze([row()])


def test_full_and_graded_ha_censuses_reconcile_shared_structure_rows():
    rows = two_targets()
    for target, loop in [("pdb:3gbm", "CC"), ("pdb:3gbn", "DD")]:
        rows += [row(loop, 7, target=target, affinity_type="-log kd", prefix=loop),
                 row("SS", 1, target=target, affinity_type="bool")]
    m = freeze(rows)
    assert m["cohort_reconciliation"]["pdb:3gbm"]["rows"] == 2
    assert m["cohort_reconciliation"]["pdb:3gbm"]["graded_rows"] == 1
    assert m["cohort_reconciliation"]["ha_shared"]["antibodies"] == 1
    assert m["cohort_reconciliation"]["ha_shared"]["graded_antibodies"] == 0


@pytest.mark.parametrize("direction,scores,expected", [
    ("lower", [3., 2., 1.], 1.), ("higher", [3., 2., 1.], 0.),
    ("higher", [2., 2., 2.], .5),
])
def test_affinity_direction_and_score_ties(direction, scores, expected):
    assert concordance([1., 2., 3.], scores, direction=direction)["concordance"] == expected
    assert concordance([1., 1.], [2., 3.], direction=direction)["comparable_pairs"] == 0


class InspectableModel(torch.nn.Module):
    def __init__(self, tokenizer, antigen_sensitive=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.antigen_sensitive = antigen_sensitive
        self.inputs = []

    def forward(self, antibody_input_ids, antibody_attention_mask, antigen_input_ids, antigen_attention_mask):
        self.inputs.extend(antibody_input_ids.detach().cpu().tolist())
        n, length = antibody_input_ids.shape
        logits = torch.zeros(n, length, len(self.tokenizer.token_to_id))
        bias = antigen_input_ids[:, 2].float() / 10 if self.antigen_sensitive else torch.ones(n)
        logits[:, :, self.tokenizer.token_to_id["A"]] = bias[:, None]
        logits[:, :, self.tokenizer.token_to_id["G"]] = -bias[:, None]
        return logits, torch.zeros(n, 2)


@pytest.mark.parametrize("sensitive", [False, True])
def test_scorer_masks_all_candidates_and_labels_swaps_only_as_sensitivity(sensitive):
    m = freeze(two_targets())
    tokenizer = AminoAcidTokenizer()
    model = InspectableModel(tokenizer, antigen_sensitive=sensitive)
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=32, device=torch.device("cpu"))
    result = score_manifest(m, infiller, fold="all", batch_size=3, progress=lambda _: None)
    assert len(model.inputs) == 4  # two targets x two conditions, shared by both variants
    assert all(ids[5:7] == [tokenizer.mask_id, tokenizer.mask_id] for ids in model.inputs)
    for g in result["groups"]:
        assert g["native_ranking"]["concordance"] == 1
        shift = g["antigen_sensitivity"]["variant_dependent_shift_range"]
        assert (shift > 0) is sensitive
        assert g["antigen_sensitivity"]["correctness"] == "unmeasured"
        assert "substituted_ranking" not in g
    assert result["claims"]["antigen_correctness_established"] is False
    repeated = score_manifest(m, infiller, fold="all", batch_size=3, progress=lambda _: None)
    assert result == repeated


@pytest.mark.parametrize("scoring_mode", ["full_span", "pll"])
def test_overflow_excludes_whole_group_and_keeps_other_targets(scoring_mode):
    rows = [row(), row("G" * 40, 2), row(target="two", antigen="DDDDD"),
            row("GG", 2, target="two", antigen="DDDDD")]
    tokenizer = AminoAcidTokenizer()
    infiller = FixedLengthHCDR3Infiller(InspectableModel(tokenizer), tokenizer, max_length=16,
                                      device=torch.device("cpu"))
    result = score_manifest(freeze(rows), infiller, fold="all", progress=lambda _: None,
                            scoring_mode=scoring_mode)
    assert result["per_target"]["one"]["scored_groups"] == 0
    assert result["per_target"]["two"]["scored_groups"] == 1
    assert result["excluded"][0]["reason"] == "complete_antibody_exceeds_checkpoint_context"


class NeighborSensitiveModel(InspectableModel):
    def forward(self, antibody_input_ids, antibody_attention_mask, antigen_input_ids, antigen_attention_mask):
        logits, other = super().forward(antibody_input_ids, antibody_attention_mask,
                                        antigen_input_ids, antigen_attention_mask)
        a, g = [self.tokenizer.token_to_id[aa] for aa in "AG"]
        # Fixtures have no A/G in the prefix; suffix WG contributes one G.
        bias = (antibody_input_ids == a).sum(1) - (antibody_input_ids == g).sum(1) + 1
        logits.zero_()
        logits[:, :, a] = bias[:, None].float()
        return logits, other


def test_pll_hides_scored_residue_and_uses_other_candidate_residues():
    tokenizer = AminoAcidTokenizer()
    model = NeighborSensitiveModel(tokenizer)
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=32,
                                      device=torch.device("cpu"))
    manifest = freeze(two_targets())
    result = score_manifest(manifest, infiller, fold="all", scoring_mode="pll",
                            batch_size=3, progress=lambda _: None)
    a, g = [tokenizer.token_to_id[aa] for aa in "AG"]
    expected_inputs = {(tokenizer.mask_id, a), (a, tokenizer.mask_id),
                       (tokenizer.mask_id, g), (g, tokenizer.mask_id)}
    assert len(model.inputs) == 16  # groups x conditions x variants x residues
    assert {tuple(ids[5:7]) for ids in model.inputs} == expected_inputs
    for group in result["groups"]:
        assert group["native_ranking"]["concordance"] == 1
        for variant in group["variants"]:
            logits = torch.zeros(20)
            logits[0] = 1 if variant["hcdr3"] == "AA" else -1
            expected = torch.log_softmax(logits, 0)[0 if variant["hcdr3"] == "AA" else 1].item()
            assert variant["scores"]["native"] == pytest.approx(expected)
            assert variant["scores"]["substituted_unmeasured"] == pytest.approx(expected)
    serial = score_manifest(manifest, infiller, fold="all", scoring_mode="pll",
                            batch_size=1, progress=lambda _: None)
    assert serial["groups"] == result["groups"]
    full = score_manifest(manifest, infiller, fold="all", progress=lambda _: None)
    assert all(group["native_ranking"]["concordance"] == .5 for group in full["groups"])


def test_one_residue_pll_equals_full_span():
    tokenizer = AminoAcidTokenizer()
    infiller = FixedLengthHCDR3Infiller(InspectableModel(tokenizer, antigen_sensitive=True),
                                      tokenizer, max_length=32, device=torch.device("cpu"))
    manifest = freeze([row("A"), row("G", 2)])
    runs = [score_manifest(manifest, infiller, fold="all", scoring_mode=mode,
                           progress=lambda _: None) for mode in ("full_span", "pll")]
    assert runs[0]["groups"] == runs[1]["groups"]


def comparison_fixture():
    tokenizer = AminoAcidTokenizer()
    infiller = FixedLengthHCDR3Infiller(NeighborSensitiveModel(tokenizer), tokenizer,
                                      max_length=32, device=torch.device("cpu"))
    manifest = freeze(two_targets())
    runs = [score_manifest(manifest, infiller, fold="all", scoring_mode=mode,
                           progress=lambda _: None) for mode in ("full_span", "pll")]
    for run in runs:
        run["provenance"] = {"checkpoint_sha256": "same-checkpoint",
                             "exposure_sha256": "same-exposure", "reconstructed_train_config": {}}
    return runs[0], runs[1], manifest


def test_paired_comparison_detects_context_gain_without_antigen_gain():
    full, pll, manifest = comparison_fixture()
    report = compare_scores(full, pll, manifest, n_resamples=20)
    assert report["pll_minus_full_span"]["estimate"] == .5
    assert report["pll_minus_full_span"]["low"] == .5
    assert report["change_in_native_minus_substituted"]["estimate"] == 0
    assert report["leave_largest_group_out_difference"]["estimate"] is None
    assert report["claims"]["antigen_conditioning_correctness_established"] is False


@pytest.mark.parametrize("change", ["checkpoint", "variant", "group", "measurement", "exclusion"])
def test_paired_comparison_refuses_unmatched_runs(change):
    full, pll, manifest = comparison_fixture()
    if change == "checkpoint":
        pll["provenance"]["checkpoint_sha256"] = "other-checkpoint"
    elif change == "variant":
        pll["groups"][0]["variants"].pop()
    elif change == "group":
        pll["groups"].pop()
    elif change == "measurement":
        pll["groups"][0]["variants"][0]["measurement"] += 1
    else:
        pll["excluded"].append({"group_id": "extra"})
    with pytest.raises(ValueError):
        compare_scores(full, pll, manifest, n_resamples=20)


def test_audit_keeps_assignment_eligibility_and_ancestor_contact_separate(tmp_path):
    m = freeze(two_targets())
    ancestor = [row(prefix="CCC", antigen="", chain_group="heavy"),
                row("GG", 2, prefix="YYY", antigen="", chain_group="light")]
    datasets = [ancestor, two_targets()]
    stages = []
    for i, (records, name) in enumerate(zip(datasets, ("base", "antigen_hcdr3_infill_refine"))):
        path = tmp_path / f"stage{i}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf8")
        stages.append({"corpus": str(path), "config": {"training_stage": name}})
    audit = audit_exposure(m, stages, progress=lambda _: None)
    g = m["groups"][0]
    a, b = g["variants"]
    a_flags = audit["variants"][variant_identity(g, a)]["stages"]
    b_flags = audit["variants"][variant_identity(g, b)]["stages"]
    assert "training:exact_hcdr3" in a_flags["0"]
    assert "0" not in b_flags  # a light-chain CDR3 is not an ancestral HCDR3
    assert a_flags["1"] == ["assigned_training"]
    assert audit["claims"]["historical_holdout_established"] is False
    assert audit["claims"]["fuzzy_hcdr3_neighborhood"].startswith("unmeasured")
    cfg = {"training_stage": "antigen_real_label_refine", "include_strength_rows": True}
    assert not stage_eligible(dict(affinity_strength_quantile=True), cfg)
    assert stage_eligible(dict(affinity_strength_quantile=.5), cfg)


def test_stage4_eligibility_uses_reader_fallback_without_benchmark_span_filter():
    cfg = {"training_stage": "antigen_hcdr3_infill_refine"}
    legacy = row(affinity_type="bool", binder_label=1)
    del legacy["is_strong_binder"]
    assert stage_eligible(legacy, cfg)
    legacy["is_strong_binder"] = False
    assert not stage_eligible(legacy, cfg)  # stored reader flag wins
    legacy["is_strong_binder"] = True
    legacy["cdr3_start_aa_heavy"], legacy["cdr3_end_aa_heavy"] = 99, 101
    assert heavy_context(legacy) is None
    assert stage_eligible(legacy, cfg)  # trainer checks span length, not sequence equality

"""
Tests for J24's comparison command and promotion rule.

The rule is lexicographic on purpose, and most of these tests are about what it
must REFUSE. A comparison harness that always produces a winner is the failure
mode: it converts "we could not tell" into "ESM won", and nothing downstream ever
revisits it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cmp_mod(project_root: Path):
    script = project_root.parents[1] / "scripts" / "compare_antigen_encoder_arms.py"
    spec = importlib.util.spec_from_file_location("compare_antigen_encoder_arms", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arm(**overrides):
    base = dict(
        policy_response_correct_vs_swap=0.10,
        policy_response_noise_band=0.02,
        compatibility_auprc=0.70,
        compatibility_calibration_error=0.05,
        avida_inner_dev_mutant_response=0.30,
        seed_variance=0.01,
        throughput_sequences_per_second=100.0,
        cache_build_seconds=0.0,
        seeds=3,
    )
    base.update(overrides)
    return base


MARGINS = dict(auprc_margin=0.02, calibration_margin=0.02, indistinguishable_within=0.01)


# --------------------------------------------------------------------------- #
# Validation happens BEFORE any metric is read
# --------------------------------------------------------------------------- #
def test_the_checked_in_pair_is_a_valid_one_axis_comparison(cmp_mod, project_root: Path):
    """The shipped configs must actually satisfy the design they document."""
    d = project_root.parents[1] / "configs/experiments/antigen_encoder"
    problems = cmp_mod.validate_paired_configs(
        cmp_mod.load_config(d / "arm_scratch.yaml"),
        cmp_mod.load_config(d / "arm_esm.yaml"),
    )
    assert problems == []


def test_a_second_differing_key_invalidates_the_comparison(cmp_mod):
    """
    The one-axis rule, mechanically enforced. A drifted learning rate between
    arms turns an encoder comparison into an encoder-and-schedule comparison, and
    nothing in the metrics would reveal it.
    """
    left = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2.pt",
            "antigen_max_length": 1024, "learning_rate": 3e-5,
            "antigen_encoder": {"type": "scratch"}}
    right = dict(left)
    right["learning_rate"] = 1e-4
    right["antigen_encoder"] = {"type": "esm", "finetune": "frozen",
                                "antigen_max_length": 1024}
    problems = cmp_mod.validate_paired_configs(left, right)
    assert any("outside the antigen encoder" in p for p in problems)


def test_different_init_checkpoints_are_rejected(cmp_mod):
    """Both arms must be rooted at the SAME stage-2 checkpoint, or the ESM arm
    inherits fusion and head weights fitted to its competitor's features."""
    left = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2_a.pt",
            "antigen_max_length": 1024, "antigen_encoder": {"type": "scratch"}}
    right = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2_b.pt",
             "antigen_max_length": 1024,
             "antigen_encoder": {"type": "esm", "finetune": "frozen",
                                 "antigen_max_length": 1024}}
    problems = cmp_mod.validate_paired_configs(left, right)
    assert any("same stage-2 checkpoint" in p for p in problems)


def test_mismatched_antigen_budgets_are_rejected(cmp_mod):
    """That would be a context-length comparison wearing an encoder comparison's
    name -- the exact confound the pre-AB-07 ESM ablation had to declare."""
    left = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2.pt",
            "antigen_max_length": 1024, "antigen_encoder": {"type": "scratch"}}
    right = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2.pt",
             "antigen_max_length": 512,
             "antigen_encoder": {"type": "esm", "finetune": "frozen",
                                 "antigen_max_length": 512}}
    problems = cmp_mod.validate_paired_configs(left, right)
    assert any("context-length comparison" in p for p in problems)


def test_lora_in_the_esm_arm_is_rejected(cmp_mod):
    """J24 adds no adaptation; LoRA is a later one-axis arm and would confound
    'pretrained representation' with 'adapted representation'."""
    left = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2.pt",
            "antigen_max_length": 1024, "antigen_encoder": {"type": "scratch"}}
    right = {"training_stage": "antigen_real_label_refine", "init_checkpoint": "s2.pt",
             "antigen_max_length": 1024,
             "antigen_encoder": {"type": "esm", "finetune": "lora",
                                 "antigen_max_length": 1024}}
    assert any("no LoRA" in p for p in cmp_mod.validate_paired_configs(left, right))


# --------------------------------------------------------------------------- #
# The promotion rule
# --------------------------------------------------------------------------- #
def test_fewer_than_three_seeds_is_refused(cmp_mod):
    """A single-seed difference is not distinguishable from initialization luck."""
    with pytest.raises(cmp_mod.ComparisonError, match="at least 3"):
        cmp_mod.decide({"scratch": _arm(seeds=1), "esm": _arm(seeds=1)}, **MARGINS)


def test_a_missing_metric_is_refused_not_defaulted(cmp_mod):
    """Treating an absent metric as zero silently decides the comparison."""
    incomplete = _arm()
    del incomplete["cache_build_seconds"]
    with pytest.raises(cmp_mod.ComparisonError, match="missing metrics"):
        cmp_mod.decide({"scratch": _arm(), "esm": incomplete}, **MARGINS)


def test_a_response_inside_the_noise_band_fails_gate_one(cmp_mod):
    """
    'Outside the band', not 'greater than zero'. The permutation control measures
    what zero looks like; a response inside it is indistinguishable from no
    antigen dependence at all.
    """
    decision = cmp_mod.decide(
        {
            "scratch": _arm(policy_response_correct_vs_swap=0.01,
                            policy_response_noise_band=0.02),
            "esm": _arm(policy_response_correct_vs_swap=0.015,
                        policy_response_noise_band=0.02),
        },
        **MARGINS,
    )
    assert decision["promoted"] is None
    assert "promote neither" in decision["reason"]


def test_a_classifier_only_win_cannot_be_promoted(cmp_mod):
    """
    The veto. The ESM arm has a clearly better compatibility AUPRC and no
    token-policy response. J24 selects a sensor for GENERATION, so a better
    classifier that does not move the policy is not a better sensor.
    """
    decision = cmp_mod.decide(
        {
            "scratch": _arm(policy_response_correct_vs_swap=0.005,
                            policy_response_noise_band=0.02,
                            compatibility_auprc=0.70),
            "esm": _arm(policy_response_correct_vs_swap=0.005,
                        policy_response_noise_band=0.02,
                        compatibility_auprc=0.85),
        },
        **MARGINS,
    )
    assert decision["promoted"] is None
    assert decision["evaluations"]["esm"]["classifier_only_improvement"] is True


def test_an_auprc_regression_beyond_the_margin_fails_gate_two(cmp_mod):
    decision = cmp_mod.decide(
        {"scratch": _arm(), "esm": _arm(compatibility_auprc=0.60)}, **MARGINS
    )
    assert decision["promoted"] == "scratch"
    assert decision["evaluations"]["esm"]["passed"] is False
    assert "AUPRC fell" in decision["evaluations"]["esm"]["gate_2_no_regression"]["detail"]


def test_a_calibration_regression_beyond_the_margin_fails_gate_two(cmp_mod):
    decision = cmp_mod.decide(
        {"scratch": _arm(), "esm": _arm(compatibility_calibration_error=0.20)}, **MARGINS
    )
    assert decision["evaluations"]["esm"]["passed"] is False
    assert "calibration error rose" in (
        decision["evaluations"]["esm"]["gate_2_no_regression"]["detail"]
    )


def test_a_tie_keeps_scratch(cmp_mod):
    """
    Indistinguishable within measured noise -> keep the simpler dependency. ESM
    is a whole extra model to download, cache, version, and explain; a tie does
    not buy that.
    """
    decision = cmp_mod.decide(
        {
            "scratch": _arm(avida_inner_dev_mutant_response=0.300),
            "esm": _arm(avida_inner_dev_mutant_response=0.305),
        },
        **MARGINS,
    )
    assert decision["promoted"] == "scratch"
    assert "simpler dependency" in decision["reason"]


def test_a_clear_esm_win_is_promoted(cmp_mod):
    """The rule must be able to say yes, or it is not a decision procedure."""
    decision = cmp_mod.decide(
        {
            "scratch": _arm(avida_inner_dev_mutant_response=0.30),
            "esm": _arm(avida_inner_dev_mutant_response=0.55),
        },
        **MARGINS,
    )
    assert decision["promoted"] == "esm"
    assert decision["ranking"] == ["esm", "scratch"]


def test_ranking_is_lexicographic_not_blended(cmp_mod):
    """
    A large throughput advantage must NOT buy a worse AVIDa response. Blending
    would let the cheap criterion pay for the expensive one, which is exactly the
    trade J24 must not make.
    """
    ranking = cmp_mod.rank_passing_arms(
        {
            "esm": _arm(avida_inner_dev_mutant_response=0.30,
                        throughput_sequences_per_second=10_000.0),
            "scratch": _arm(avida_inner_dev_mutant_response=0.50,
                            throughput_sequences_per_second=1.0),
        }
    )
    assert ranking[0] == "scratch", "throughput must not outrank AVIDa response"


def test_seed_variance_breaks_a_tie_toward_the_stabler_arm(cmp_mod):
    ranking = cmp_mod.rank_passing_arms(
        {
            "esm": _arm(avida_inner_dev_mutant_response=0.4, seed_variance=0.05),
            "scratch": _arm(avida_inner_dev_mutant_response=0.4, seed_variance=0.01),
        }
    )
    assert ranking[0] == "scratch"


# --------------------------------------------------------------------------- #
# CLI behavior
# --------------------------------------------------------------------------- #
def test_validate_only_needs_no_results(cmp_mod):
    assert cmp_mod.main(["--validate-only"]) == 0


def test_results_without_predeclared_margins_is_refused(cmp_mod, tmp_path, capsys):
    """
    A margin chosen after seeing results is not a margin. The command refuses
    rather than supplying a default, because a default here would be exactly that
    post-hoc threshold wearing a respectable name.
    """
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"scratch": _arm(), "esm": _arm()}), encoding="utf-8")
    assert cmp_mod.main(["--results", str(results)]) == 1
    assert "predeclared inputs" in capsys.readouterr().err


def test_end_to_end_writes_the_predeclared_schema(cmp_mod, tmp_path):
    results = tmp_path / "r.json"
    results.write_text(
        json.dumps(
            {
                "scratch": _arm(avida_inner_dev_mutant_response=0.30),
                "esm": _arm(avida_inner_dev_mutant_response=0.55),
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    assert (
        cmp_mod.main(
            [
                "--results", str(results),
                "--auprc-margin", "0.02",
                "--calibration-margin", "0.02",
                "--indistinguishable-within", "0.01",
                "--output-json", str(out),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == cmp_mod.SCHEMA_VERSION
    assert report["decision"]["promoted"] == "esm"
    # The report must carry the margins it was judged under, or a later reader
    # cannot tell whether they were predeclared or fitted.
    assert report["margins"]["auprc"] == 0.02
    assert "sensor" in report["scope"]

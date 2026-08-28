#!/usr/bin/env python
"""
Compare J24's two antigen-encoder arms and apply the promotion rule.

J24 selects the antigen SENSOR. It is not the target-specific claim; a later
experiment tests whether the chosen sensor helps generate better held-out HCDR3
choices for a specific target against controlled contrast antigens.

Two jobs, deliberately in one command so neither can be skipped:

1. **Validate the comparison before reading any metric.** The arms must differ in
   exactly the antigen encoder. This checks that from the configs themselves
   rather than trusting that someone kept them in step, and it refuses to score a
   comparison that is not one-axis. Running it with ``--validate-only`` is the
   pre-flight; it needs no results.

2. **Apply the promotion rule lexicographically**, in the owner's order, without
   blending anything into a single score. A blended score lets a large throughput
   win pay for an absent policy response, which is exactly the trade J24 must not
   make.

The rule (specs/experiments/j24_antigen_encoder.md is authoritative):

  Gate 1  positive correct-antigen vs matched-swap HCDR3 policy response,
          beyond the null/permutation noise band.
  Gate 2  no regression in compatibility calibration or held-out AUPRC beyond a
          predeclared margin.
  Rank    among passing arms: AVIDa inner-development mutant response, seed
          variance, throughput, cache cost.
  Veto    a classifier-only improvement cannot win when token-policy response is
          absent.
  Ties    neither passing -> promote neither. Indistinguishable within measured
          noise -> keep scratch, the simpler dependency.

Thresholds are inputs, not defaults: every margin comes from the predeclared
report and none is invented here. An unset margin is an error rather than a
guess, because a threshold chosen after seeing results is not a threshold.

Usage::

    python scripts/compare_antigen_encoder_arms.py --validate-only
    python scripts/compare_antigen_encoder_arms.py --results j24_results.json \\
        --output-json outputs/j24-comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "j24-antigen-encoder-comparison/1"

#: The two keys allowed to differ between the paired configs.
#:
#: ``antigen_encoder`` is the axis under test. ``output_dir`` must differ or the
#: arms would overwrite each other -- and Rule 2 forbids sharing an output dir
#: anyway. Anything else differing means the comparison is not one-axis.
ALLOWED_CONFIG_DIFFERENCES = frozenset({"antigen_encoder", "output_dir"})

#: Metrics a passing arm must supply. Listed so a missing metric is reported as
#: missing rather than silently treated as a zero or a loss.
REQUIRED_ARM_METRICS = (
    "policy_response_correct_vs_swap",
    "policy_response_noise_band",
    "compatibility_auprc",
    "compatibility_calibration_error",
    "avida_inner_dev_mutant_response",
    "seed_variance",
    "throughput_sequences_per_second",
    "cache_build_seconds",
    "seeds",
    # Both counts, because the arms differ in what TRAINS as well as in what the
    # encoder represents: the scratch antigen encoder is trainable, the ESM
    # backbone is frozen. Reporting only totals would hide that; reporting only
    # trainable would hide the ESM arm's inference cost.
    "total_parameters",
    "trainable_parameters",
)


class ComparisonError(RuntimeError):
    """The comparison is not valid; no metric should be read."""


# --------------------------------------------------------------------------- #
# 1. Validate the comparison
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def config_differences(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    """Every top-level key whose value differs between the two arms."""
    return {k for k in set(left) | set(right) if left.get(k) != right.get(k)}


def validate_paired_configs(scratch: dict[str, Any], esm: dict[str, Any]) -> list[str]:
    """
    Return the reasons this pair is not a valid one-axis comparison.

    Empty list means valid. Returned rather than raised so the caller can report
    every problem at once -- fixing one config difference only to discover the
    next is how a pilot gets run three times.
    """
    problems: list[str] = []

    unexpected = config_differences(scratch, esm) - ALLOWED_CONFIG_DIFFERENCES
    if unexpected:
        problems.append(
            f"configs differ outside the antigen encoder: {sorted(unexpected)}. "
            "J24 requires data rows, ordering, masks, replay, optimizer steps, and "
            "supervision to be identical."
        )

    if scratch.get("init_checkpoint") != esm.get("init_checkpoint"):
        problems.append("arms do not start from the same stage-2 checkpoint")
    if not str(scratch.get("init_checkpoint", "")):
        problems.append("no init_checkpoint: both arms must be stage-2 rooted")

    for name, cfg in (("scratch", scratch), ("esm", esm)):
        stage = cfg.get("training_stage")
        if stage != "antigen_real_label_refine":
            problems.append(
                f"{name} arm runs training_stage={stage!r}; J24 compares encoders at "
                "stage 3, before the canonical run"
            )

    scratch_block = scratch.get("antigen_encoder") or {}
    esm_block = esm.get("antigen_encoder") or {}
    if scratch_block.get("type") != "scratch" or esm_block.get("type") != "esm":
        problems.append("the two arms are not one scratch and one esm")
    if esm_block.get("finetune") != "frozen":
        problems.append(
            f"esm arm finetune={esm_block.get('finetune')!r}; J24 adds no LoRA -- "
            "adaptation is a later one-axis arm"
        )

    # The antigen budget must match, or the comparison is partly about context.
    scratch_budget = scratch.get("antigen_max_length")
    esm_budget = esm_block.get("antigen_max_length", esm.get("antigen_max_length"))
    if scratch_budget != esm_budget:
        problems.append(
            f"antigen budgets differ (scratch={scratch_budget}, esm={esm_budget}); "
            "that is a context-length comparison, not an encoder comparison"
        )
    return problems


# --------------------------------------------------------------------------- #
# 2. The promotion rule
# --------------------------------------------------------------------------- #
def _missing_metrics(arm: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_ARM_METRICS if key not in arm]


def gate_policy_response(arm: dict[str, Any]) -> tuple[bool, str]:
    """
    Gate 1: correct-antigen vs matched-swap HCDR3 policy response must be
    positive and outside the null/permutation noise band.

    "Outside the band" rather than "> 0": the band IS the measurement of what
    zero looks like here, and a response inside it is indistinguishable from the
    permutation control.
    """
    response = float(arm["policy_response_correct_vs_swap"])
    band = float(arm["policy_response_noise_band"])
    if band < 0:
        raise ComparisonError("policy_response_noise_band must be non-negative")
    if response > band:
        return True, f"response {response:.4g} exceeds the noise band {band:.4g}"
    return False, (
        f"response {response:.4g} is within the null/permutation noise band "
        f"{band:.4g}; indistinguishable from no antigen dependence"
    )


def gate_no_regression(
    arm: dict[str, Any],
    baseline: dict[str, Any],
    auprc_margin: float,
    calibration_margin: float,
) -> tuple[bool, str]:
    """
    Gate 2: no regression in compatibility calibration or held-out AUPRC beyond
    the predeclared margins.
    """
    auprc_drop = float(baseline["compatibility_auprc"]) - float(arm["compatibility_auprc"])
    calibration_rise = float(arm["compatibility_calibration_error"]) - float(
        baseline["compatibility_calibration_error"]
    )
    reasons = []
    if auprc_drop > auprc_margin:
        reasons.append(f"AUPRC fell {auprc_drop:.4g} (margin {auprc_margin:.4g})")
    if calibration_rise > calibration_margin:
        reasons.append(
            f"calibration error rose {calibration_rise:.4g} "
            f"(margin {calibration_margin:.4g})"
        )
    if reasons:
        return False, "; ".join(reasons)
    return True, (
        f"AUPRC change {-auprc_drop:+.4g}, calibration change {-calibration_rise:+.4g}, "
        "both within margin"
    )


def rank_passing_arms(arms: dict[str, dict[str, Any]]) -> list[str]:
    """
    Order passing arms by the tie-breakers, in the owner's order.

    Lexicographic, not weighted: each criterion is consulted only when the
    previous one ties. AVIDa response and throughput are maximized; seed variance
    and cache cost are minimized.
    """
    def key(name: str):
        arm = arms[name]
        return (
            -float(arm["avida_inner_dev_mutant_response"]),
            float(arm["seed_variance"]),
            -float(arm["throughput_sequences_per_second"]),
            float(arm["cache_build_seconds"]),
        )

    return sorted(arms, key=key)


def decide(
    arms: dict[str, dict[str, Any]],
    *,
    auprc_margin: float,
    calibration_margin: float,
    indistinguishable_within: float,
) -> dict[str, Any]:
    """
    Apply the whole rule and return the predeclared report body.

    Every gate result is recorded, including for arms that fail early, so the
    report shows WHY an arm lost rather than only that it did.
    """
    if "scratch" not in arms:
        raise ComparisonError("the scratch arm is the baseline and must be present")

    for name, arm in arms.items():
        missing = _missing_metrics(arm)
        if missing:
            raise ComparisonError(f"arm {name!r} is missing metrics: {missing}")
        seeds = int(arm["seeds"])
        if seeds < 3:
            raise ComparisonError(
                f"arm {name!r} reports {seeds} seeds; J24 requires at least 3, "
                "because a single-seed difference is not distinguishable from "
                "initialization luck"
            )

    baseline = arms["scratch"]
    evaluations: dict[str, Any] = {}
    passing: dict[str, dict[str, Any]] = {}

    for name, arm in arms.items():
        policy_ok, policy_why = gate_policy_response(arm)
        regression_ok, regression_why = gate_no_regression(
            arm, baseline, auprc_margin, calibration_margin
        )
        # The veto, stated explicitly rather than left implicit in gate 1: an arm
        # with a better classifier and no token-policy response does not win.
        classifier_only = (
            not policy_ok
            and float(arm["compatibility_auprc"]) > float(baseline["compatibility_auprc"])
        )
        evaluations[name] = {
            "gate_1_policy_response": {"passed": policy_ok, "detail": policy_why},
            "gate_2_no_regression": {"passed": regression_ok, "detail": regression_why},
            "classifier_only_improvement": classifier_only,
            "passed": policy_ok and regression_ok,
        }
        if policy_ok and regression_ok:
            passing[name] = arm

    if not passing:
        return {
            "promoted": None,
            "reason": (
                "no arm cleared both gates; promote neither. A sensor that does not "
                "produce measurable antigen-dependent policy behavior is not a sensor."
            ),
            "evaluations": evaluations,
            "ranking": [],
        }

    ranking = rank_passing_arms(passing)
    winner = ranking[0]

    if len(ranking) > 1:
        gap = abs(
            float(passing[ranking[0]]["avida_inner_dev_mutant_response"])
            - float(passing[ranking[1]]["avida_inner_dev_mutant_response"])
        )
        if gap <= indistinguishable_within:
            return {
                "promoted": "scratch",
                "reason": (
                    f"arms are indistinguishable within measured noise (gap {gap:.4g} "
                    f"<= {indistinguishable_within:.4g}); keep scratch as the simpler "
                    "dependency"
                ),
                "evaluations": evaluations,
                "ranking": ranking,
            }

    return {
        "promoted": winner,
        "reason": f"{winner} cleared both gates and leads the lexicographic ranking",
        "evaluations": evaluations,
        "ranking": ranking,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    default_dir = Path(__file__).resolve().parents[1] / "configs/experiments/antigen_encoder"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--scratch-config", type=Path, default=default_dir / "arm_scratch.yaml")
    parser.add_argument("--esm-config", type=Path, default=default_dir / "arm_esm.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the pair is a one-axis comparison and stop. Needs no results.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help='JSON: {"scratch": {...metrics...}, "esm": {...}}',
    )
    parser.add_argument(
        "--auprc-margin",
        type=float,
        default=None,
        help="Predeclared allowed AUPRC regression. Required with --results.",
    )
    parser.add_argument(
        "--calibration-margin",
        type=float,
        default=None,
        help="Predeclared allowed calibration-error increase. Required with --results.",
    )
    parser.add_argument(
        "--indistinguishable-within",
        type=float,
        default=None,
        help="AVIDa-response gap below which the arms are a tie. Required with --results.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    scratch_cfg = load_config(args.scratch_config)
    esm_cfg = load_config(args.esm_config)
    problems = validate_paired_configs(scratch_cfg, esm_cfg)

    if problems:
        print("J24 comparison is NOT valid:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRefusing to read metrics from an invalid comparison.",
            file=sys.stderr,
        )
        return 1

    print("J24 pair is a valid one-axis comparison:")
    print(f"  scratch: {args.scratch_config}")
    print(f"  esm    : {args.esm_config}")
    print(f"  differing keys: {sorted(config_differences(scratch_cfg, esm_cfg))}")

    if args.validate_only or args.results is None:
        if args.results is None and not args.validate_only:
            print("\nNo --results supplied; validation only. J24 has not run.")
        return 0

    for name, value in (
        ("--auprc-margin", args.auprc_margin),
        ("--calibration-margin", args.calibration_margin),
        ("--indistinguishable-within", args.indistinguishable_within),
    ):
        if value is None:
            print(
                f"\nrefusing: {name} is required with --results. Margins are "
                "predeclared inputs; one chosen after seeing results is not a "
                "threshold.",
                file=sys.stderr,
            )
            return 1

    arms = json.loads(args.results.read_text(encoding="utf-8"))
    try:
        decision = decide(
            arms,
            auprc_margin=args.auprc_margin,
            calibration_margin=args.calibration_margin,
            indistinguishable_within=args.indistinguishable_within,
        )
    except ComparisonError as exc:
        print(f"\nrefusing: {exc}", file=sys.stderr)
        return 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "question": (
            "Given the same pretrained antibody model, data, supervision, "
            "initialization, and 1024-token antigen crop, does frozen ESM-2 produce "
            "stronger antigen-dependent policy behavior than the scratch antigen "
            "encoder?"
        ),
        "scope": (
            "J24 selects the antigen sensor. It does not test the target-specific "
            "claim, which is a later experiment."
        ),
        "claim_limit": (
            "The arms differ in training regime as well as representation: the "
            "scratch antigen encoder trains while the ESM backbone is frozen. The "
            "result therefore says which ENCODER PACKAGE works better under the "
            "intended training regime. It does not isolate the effect of "
            "pretraining alone -- that would need a trainable-ESM or "
            "frozen-scratch arm, which J24 deliberately does not run."
        ),
        "margins": {
            "auprc": args.auprc_margin,
            "calibration": args.calibration_margin,
            "indistinguishable_within": args.indistinguishable_within,
        },
        "configs": {
            "scratch": args.scratch_config.as_posix(),
            "esm": args.esm_config.as_posix(),
            "differing_keys": sorted(config_differences(scratch_cfg, esm_cfg)),
        },
        "arms": arms,
        "decision": decision,
    }

    print(f"\npromoted: {decision['promoted']}")
    print(f"reason:   {decision['reason']}")
    for name, evaluation in decision["evaluations"].items():
        print(f"  [{name}] passed={evaluation['passed']}")
        print(f"      gate 1 policy response: {evaluation['gate_1_policy_response']['detail']}")
        print(f"      gate 2 no regression  : {evaluation['gate_2_no_regression']['detail']}")
        if evaluation["classifier_only_improvement"]:
            print("      NOTE: classifier-only improvement -- cannot win under the veto")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

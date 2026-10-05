#!/usr/bin/env python
"""Finite-support learning baseline: a runnable CPU check of fit -> allocate -> evaluate.

NumPy only. No production data discovery, no native models, no GPU: every input and output path
is explicit. This is a finite-support learning/optimisation baseline with a synthetic
qualification. It is not native generation and not a claim of biological improvement; the
normalised restriction of a declared reference to a finite pool is the experiment, not the native
generator's distribution.

Three stages with one-way dependence:

  1. fit       ridge reward predictor and a bounded linear-energy DPO policy, from the TRAINING
               labels only (a seeded label budget is drawn without looking at evaluation outcomes);
  2. allocate  exact learned tilt, DPO tilts at several pressures and a finite SIR approximation
               over the held-out candidates, from FROZEN predictions. No outcome argument exists;
  3. evaluate  the hashed frozen artifact is written, reloaded and verified FIRST; only then is the
               evaluation-outcome file opened. A missing outcome fails (nothing is renormalised
               onto an observed subset) and a hindsight oracle is computed here and only here.

Subcommands:

  selftest        run the hand-computed analytic cases, a planted interaction experiment (additive
                  predictor versus interaction ridge, with ablation / donor-patch / rescue
                  interventions and a permutation null) and the CSV pipeline on synthetic data;
                  write selftest_report.json plus the synthetic inputs and artifacts. Exit 0 only
                  if every named check holds. The planted toy is a FIT/INTERVENTION qualification
                  on a replicated full factorial (8 unique feature contexts, 24 training rows)
                  with the interaction coordinate ab supplied and evaluation on the same feature
                  cells; it is NOT unseen-context generalisation.
  finite-support  run the pipeline on explicit CSV/JSON files and write frozen_artifact.json and
                  evaluation_report.json.

Input contracts (exact columns; an extra column is refused rather than ignored):

  candidates CSV         candidate_id, sequence, group_id, split ("train" or "eval"),
                         prior_log_weight (a finite log weight, or the EXPLICIT spelling -inf for
                         a structural zero; an overflowing literal such as -1e999 is refused).
                         Train-split rows with no training outcome are accepted as unlabelled
                         training candidates and are bound into the artifact by digest.
  training outcomes CSV  candidate_id, utility
  evaluation outcomes    candidate_id, utility   (must cover EVERY eval candidate)
  manifest JSON          target, assay, utility {units, scale, higher_is_better (must be true),
                         useful_threshold (fixed, in utility units)}, grouping_rationale,
                         features {kind "sequence_onehot", alphabet, length, provenance},
                         prior {kind uniform | joint_log_probability | score_normalized, semantics,
                         provenance}. Provenance is label_free, documented (assay, target,
                         label_count, ideally label_ids) or unknown.

All utilities must belong to one comparable target and assay; no conversion is inferred from a raw
KD. Provenance is declared, not verified: the artifact hash proves the artifact was not edited, not
that the manifest's statements are true or that the inputs were truthful, and an unknown or
overlapping label history makes the clean low-N / acquisition-replay verdict unavailable or false.
The manifest JSON is read strictly: a duplicate key at any nesting level is refused.

Label accounting. A label is identified by (target, assay, label id). Declared supervision uses
(acquired, feature history, prior history) are reported per component and never summed into a label
count; the unique same-target total is exact only when every component lists its label ids and no
history is unknown, otherwise it is a bound or unavailable. Declared validation labels are a
separate cost. Repeated label uses by the DPO optimiser are counted as uses, not acquisition.
--label-budget is NESTED for a fixed roster and seed (the first N of one seeded permutation), so a
smaller budget acquires a subset of a larger one.

Output safety. Any input path (candidates, training outcomes, evaluation outcomes, manifest) that
resolves to an output (frozen_artifact.json, evaluation_report.json or their .tmp staging files) is
refused before anything is written, even with --overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import functional_baseline as fb  # noqa: E402

_DEFAULTS = fb.BaselineConfig(tau=1.0)


def _formatter(prog):
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=34, width=100)


def build_parser():
    parser = argparse.ArgumentParser(prog="functional_baseline.py", description=__doc__,
                                     formatter_class=_formatter)
    commands = parser.add_subparsers(dest="command", required=True,
                                     metavar="{selftest,finite-support}")

    selftest = commands.add_parser(
        "selftest", formatter_class=_formatter,
        help="run the analytic cases, the planted experiment and the CSV pipeline on synthetic data",
        description=("Run the whole baseline on synthetic data with a known truth and write a real "
                     "JSON report (selftest_report.json) under --output-dir, together with the "
                     "synthetic input files and the frozen artifacts, so every number can be "
                     "re-derived. Exit status 0 only if every named check holds."))
    selftest.add_argument("--output-dir", required=True,
                          help="directory to write the report, inputs and artifacts into")
    selftest.add_argument("--seed", type=int, default=0,
                          help="master seed for every random stream (default: 0)")

    finite = commands.add_parser(
        "finite-support", formatter_class=_formatter,
        help="fit, freeze and evaluate on explicit CSV/JSON files",
        description=("Fit a ridge predictor and a DPO policy on the training outcomes, allocate "
                     "mass on the held-out candidates, write the hashed frozen artifact, and only "
                     "then open the evaluation outcomes to score it. See the module help for the "
                     "input contracts."))
    for flag, text in (("--candidates", "candidate CSV: candidate_id, sequence, group_id, split, "
                                        "prior_log_weight"),
                       ("--train-outcomes", "training outcome CSV: candidate_id, utility"),
                       ("--eval-outcomes", "evaluation outcome CSV (opened only after the frozen "
                                           "artifact is written): candidate_id, utility"),
                       ("--manifest", "manifest JSON (target, assay, utility, grouping, "
                                      "feature and prior provenance)"),
                       ("--output-dir", "directory for frozen_artifact.json and "
                                        "evaluation_report.json")):
        finite.add_argument(flag, required=True, help=text)
    finite.add_argument("--tau", type=float, required=True,
                        help="KL-regularisation temperature; every true-J comparison in the report "
                             "shares this one value (no default: it is a scientific choice)")
    finite.add_argument("--seed", type=int, default=_DEFAULTS.seed,
                        help="master seed for the label budget, preference pairs and SIR draws "
                             f"(default: {_DEFAULTS.seed})")
    finite.add_argument("--ridge-alpha", type=float, default=_DEFAULTS.ridge_alpha,
                        help=f"ridge penalty on the weights (default: {_DEFAULTS.ridge_alpha})")
    finite.add_argument("--label-budget", type=int, default=None,
                        help="acquire this many of the offered training labels: the first N of one "
                             "seeded permutation (nested across budgets for a fixed roster and "
                             "seed) that never sees evaluation outcomes "
                             "(default: all offered labels)")
    finite.add_argument("--declared-validation-labels", type=int,
                        default=_DEFAULTS.declared_validation_labels,
                        help="labels spent on a validation set used for tuning, declared as a "
                             "separate cost (their disjointness from the other labels is not "
                             "asserted without ids) (default: 0)")
    finite.add_argument("--dpo-beta", type=float, default=_DEFAULTS.dpo_beta,
                        help=f"DPO margin scale (default: {_DEFAULTS.dpo_beta})")
    finite.add_argument("--dpo-l2", type=float, default=_DEFAULTS.dpo_l2,
                        help="positive L2 on the DPO energy weights; the objective is the mean "
                             f"pair loss plus l2*||theta||^2/2 (default: {_DEFAULTS.dpo_l2})")
    finite.add_argument("--dpo-max-pairs", type=int, default=_DEFAULTS.dpo_max_pairs,
                        help="cap on expanded preference pairs; more are sampled with the seed "
                             f"(default: {_DEFAULTS.dpo_max_pairs})")
    finite.add_argument("--dpo-max-iterations", type=int, default=_DEFAULTS.dpo_max_iterations,
                        help=f"cap on Newton steps (default: {_DEFAULTS.dpo_max_iterations})")
    finite.add_argument("--dpo-tolerance", type=float, default=_DEFAULTS.dpo_tolerance,
                        help="gradient-norm convergence tolerance "
                             f"(default: {_DEFAULTS.dpo_tolerance})")
    finite.add_argument("--dpo-pressures", type=float, nargs="+",
                        default=list(_DEFAULTS.dpo_pressures),
                        help="multipliers c of the fitted energy for the pressure curve, "
                             "q = p*exp(c*theta.phi) normalised "
                             f"(default: {' '.join(str(c) for c in _DEFAULTS.dpo_pressures)})")
    finite.add_argument("--sir-proposal", choices=(fb.SIR_PRIOR, fb.SIR_UNIFORM),
                        default=_DEFAULTS.sir_proposal,
                        help="SIR proposal: the reference itself, or uniform over its positive "
                             f"support (default: {_DEFAULTS.sir_proposal})")
    finite.add_argument("--sir-proposal-draws", type=int, default=_DEFAULTS.sir_proposal_draws,
                        help="K proposal draws for SIR "
                             f"(default: {_DEFAULTS.sir_proposal_draws})")
    finite.add_argument("--final-draws", type=int, default=_DEFAULTS.final_draws,
                        help="M final draws: the SIR resample size and the draw count in the "
                             f"expected-distinct formulas (default: {_DEFAULTS.final_draws})")
    finite.add_argument("--overwrite", action="store_true",
                        help="replace an existing frozen artifact / report in --output-dir "
                             "(default: refuse). Never replaces an input: a path alias between an "
                             "input and an output is refused before anything is written")
    return parser


def _config(args):
    return fb.BaselineConfig(
        tau=args.tau, seed=args.seed, ridge_alpha=args.ridge_alpha, label_budget=args.label_budget,
        declared_validation_labels=args.declared_validation_labels, dpo_beta=args.dpo_beta,
        dpo_l2=args.dpo_l2, dpo_max_pairs=args.dpo_max_pairs,
        dpo_max_iterations=args.dpo_max_iterations, dpo_tolerance=args.dpo_tolerance,
        dpo_pressures=tuple(args.dpo_pressures), sir_proposal=args.sir_proposal,
        sir_proposal_draws=args.sir_proposal_draws, final_draws=args.final_draws)


def _run_selftest(args):
    report = fb.selftest(args.output_dir, seed=args.seed)
    for check in report["checks"]:
        print(f"  [{'ok' if check['holds'] else '!!'}] {check['name']}")
    print(f"selftest {'PASSED' if report['passed'] else 'FAILED'}: "
          f"{sum(1 for c in report['checks'] if c['holds'])}/{len(report['checks'])} checks; "
          f"report at {Path(args.output_dir) / 'selftest_report.json'}")
    print(report["note"])
    return 0 if report["passed"] else 1


def _run_finite_support(args):
    summary = fb.run_finite_support(
        candidates=args.candidates, train_outcomes=args.train_outcomes,
        eval_outcomes=args.eval_outcomes, manifest=args.manifest, output_dir=args.output_dir,
        config=_config(args), overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["all_identities_hold"] else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return _run_selftest(args)
        return _run_finite_support(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Is there ANY learnable ranking signal in the contrast labels?

A positive control for the HCDR3 contrast benchmark. It fits a ridge regression
on plain sequence features -- positional residue identity, composition, length --
directly against the measured affinities, with no transformer anywhere. Then it
scores the result with the SAME primary endpoint as the model
(`contrast_statistics.primary_endpoint`: equal-target, equal-group concordance,
floor 0.500), so the two numbers are directly comparable.

Why this is the deciding experiment. Four model-side explanations for the
checkpoint's null ranking are already eliminated: the architecture can express
antigen conditioning, no fusion gate is closed, the trained weights are not
stuck, and a pseudo-log-likelihood readout does not help. What remains is about
the DATA:

    (2) training never required antigen use for ranking, or
    (3) the labels carry no learnable ranking signal at all.

A linear model fit straight on the labels separates them. Above chance means
signal exists and the training objective is the problem. At chance means these
labels are not rankable from sequence at this resolution, and the benchmark
should be retired rather than optimised against.

Two arms:

    cv       group-level K-fold within each target. Fit on other backgrounds,
             predict held-out ones. This is the honest generalisation number.
    ceiling  fit and predict on the same variants -- deliberately in-sample.
             An UPPER BOUND, not a result. If even this sits at chance, the
             labels are not a function of the HCDR3 sequence and no model of
             any kind will rank them.

Targets are fit with y centred within each group, because the metric is a
within-background ordering and between-background offsets are nuisance.

Results are reported across a regularisation grid rather than at a single tuned
lambda: the question is binary (is anything above chance?), and reporting the
whole grid avoids quietly selecting the winner.

    python scripts/probe_linear_hcdr3_ranker.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.evaluation.contrast_statistics import (
    GroupStat, cluster_bootstrap, equal_group_by_target, primary_endpoint,
)
from smallAntibodyGen.evaluation.contrasts import concordance, load_json

DEFAULT_SCORES = ROOT / "outputs" / "hcdr3_contrasts" / "scores.json.gz"
DEFAULT_MANIFEST = ROOT / "outputs" / "hcdr3_contrasts" / "manifest.json.gz"
DEFAULT_OUTPUT = ROOT / "outputs" / "hcdr3_contrasts" / "linear_ranker.json"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def featurise(hcdr3s):
    """Positional residue one-hots (left- and right-anchored), composition, length."""
    width = max(len(h) for h in hcdr3s)
    anchor = min(width, 12)  # cap the positional block so features stay dense
    n_aa = len(AMINO_ACIDS)
    rows = []
    for sequence in hcdr3s:
        left = np.zeros((anchor, n_aa))
        right = np.zeros((anchor, n_aa))
        composition = np.zeros(n_aa)
        for position, residue in enumerate(sequence):
            index = AA_INDEX.get(residue)
            if index is None:
                continue
            composition[index] += 1
            if position < anchor:
                left[position, index] = 1.0
            back = len(sequence) - 1 - position
            if back < anchor:
                right[back, index] = 1.0
        rows.append(np.concatenate([left.ravel(), right.ravel(), composition,
                                    [len(sequence)]]))
    return np.asarray(rows, dtype=float)


def ridge_fit(X, y, lam):
    """Closed-form ridge with an unpenalised intercept."""
    design = np.hstack([X, np.ones((len(X), 1))])
    gram = design.T @ design
    penalty = lam * np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ y)


def ridge_predict(X, weights):
    return np.hstack([X, np.ones((len(X), 1))]) @ weights


def load_population(scores_path, manifest_path):
    """Per-target lists of (group_id, hcdr3s, oriented measurements)."""
    scores = load_json(scores_path)
    directions = {g["group_id"]: g.get("direction") for g in load_json(manifest_path)["groups"]}
    by_target = {}
    for group in scores["groups"]:
        direction = directions.get(group["group_id"])
        if direction not in ("higher", "lower"):
            raise ValueError("group {!r} has no direction".format(group["group_id"]))
        variants = group["variants"]
        if len(variants) < 2:
            continue
        sign = 1.0 if direction == "higher" else -1.0
        hcdr3s = [v["hcdr3"] for v in variants]
        values = np.array([sign * float(v["measurement"]) for v in variants])
        by_target.setdefault(group["target"], []).append(
            (group["group_id"], hcdr3s, values))
    return by_target


def group_stats_from_predictions(target, groups, predictions):
    """Score predicted values against oriented measurements, group by group."""
    stats = []
    for (group_id, _, values), predicted in zip(groups, predictions):
        counts = concordance(list(values), list(predicted), direction="higher")
        if not counts["comparable_pairs"]:
            continue
        stats.append(GroupStat(target=target, group_id=group_id,
                               concordant=counts["concordant"],
                               discordant=counts["discordant"],
                               score_ties=counts["score_ties"],
                               comparable_pairs=counts["comparable_pairs"],
                               n_variants=len(values)))
    return stats


def centred(values_list):
    """Within-group centring: the metric is a within-background ordering."""
    return np.concatenate([v - v.mean() for v in values_list])


def sequence_disjoint_folds(groups, folds):
    """Assign whole HCDR3-connected components to folds, so no sequence crosses.

    Group-level CV holds out backgrounds but not sequences: 66.6% of scored
    variants share an HCDR3 with another variant, so a sequence memorised in one
    fold can reappear in a held-out one. Linking groups that share any HCDR3 into
    components and splitting components is the same connected-component
    discipline the repo prescribes for leak-free corpus splits.
    """
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for index, (_, hcdr3s, _) in enumerate(groups):
        for sequence in hcdr3s:
            if sequence in seen:
                a, b = find(seen[sequence]), find(index)
                if a != b:
                    parent[a] = b
            else:
                seen[sequence] = index
    components = {}
    for index in range(len(groups)):
        components.setdefault(find(index), []).append(index)
    # Largest components first, round-robin, so folds stay balanced.
    ordered = sorted(components.values(), key=lambda c: (-len(c), groups[c[0]][0]))
    sizes = [0] * folds
    assignment = {}
    for component in ordered:
        target_fold = sizes.index(min(sizes))
        for index in component:
            assignment[index] = target_fold
        sizes[target_fold] += len(component)
    return assignment, len(ordered), max(len(c) for c in ordered) / len(groups)


def run_arm(by_target, lam, *, folds, arm):
    stats = []
    for target, groups in sorted(by_target.items()):
        order = sorted(range(len(groups)), key=lambda i: groups[i][0])  # deterministic
        if arm == "cv_seqdisjoint":
            assignment, _, _ = sequence_disjoint_folds(groups, folds)
        else:
            assignment = {order[i]: i % folds for i in range(len(order))}
        predictions = [None] * len(groups)
        splits = [None] if arm == "ceiling" else range(folds)
        for held in splits:
            if arm == "ceiling":
                fit_idx = list(range(len(groups)))
                predict_idx = fit_idx
            else:
                fit_idx = [i for i in range(len(groups)) if assignment[i] != held]
                predict_idx = [i for i in range(len(groups)) if assignment[i] == held]
            if not fit_idx or not predict_idx:
                continue
            X_fit = featurise([h for i in fit_idx for h in groups[i][1]])
            y_fit = centred([groups[i][2] for i in fit_idx])
            weights = ridge_fit(X_fit, y_fit, lam)
            for i in predict_idx:
                predictions[i] = ridge_predict(featurise(groups[i][1]), weights)
        usable = [(g, p) for g, p in zip(groups, predictions) if p is not None]
        stats.extend(group_stats_from_predictions(
            target, [g for g, _ in usable], [p for _, p in usable]))
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    by_target = load_population(args.scores, args.manifest)
    print("population: {} groups across {} targets".format(
        sum(len(v) for v in by_target.values()), len(by_target)), flush=True)

    report = {"floor": 0.5, "folds": args.folds, "lambda_grid": list(LAMBDA_GRID), "arms": {}}

    # Component structure decides whether a sequence-disjoint split is even
    # possible. A target whose largest component covers most of its groups
    # cannot be split without starving the fit.
    report["sequence_components"] = {}
    for target, groups in sorted(by_target.items()):
        _, components, largest = sequence_disjoint_folds(groups, args.folds)
        report["sequence_components"][target] = {
            "groups": len(groups), "components": components,
            "largest_component_group_share": largest}
        print("  {}: {} groups -> {} HCDR3-linked components, largest {:.1%}".format(
            target, len(groups), components, largest), flush=True)

    print("\n{:16}{:>10}{:>12}{:>12}   {}".format(
        "arm", "lambda", "primary", "vs floor", "per target"), flush=True)
    print("-" * 100)
    for arm in ("cv", "cv_seqdisjoint", "ceiling"):
        for lam in LAMBDA_GRID:
            stats = run_arm(by_target, lam, folds=args.folds, arm=arm)
            estimate = primary_endpoint(stats)
            per_target = equal_group_by_target(stats)
            report["arms"]["{}@{}".format(arm, lam)] = {
                "primary": estimate, "by_target": per_target, "groups": len(stats)}
            print("{:16}{:>10.1f}{:>12.4f}{:>+12.4f}   {}".format(
                arm, lam, estimate, estimate - 0.5,
                "  ".join("{}={:.3f}".format(t.split(":")[-1], v)
                          for t, v in per_target.items())), flush=True)
        print("-" * 100)

    for arm in ("cv", "cv_seqdisjoint"):
        prefix = arm + "@"
        best = max((k for k in report["arms"] if k.startswith(prefix)),
                   key=lambda k: report["arms"][k]["primary"])
        stats = run_arm(by_target, float(best.split("@")[1]), folds=args.folds, arm=arm)
        report["best_" + arm] = {
            "key": best,
            "bootstrap": cluster_bootstrap(stats, n_resamples=args.n_resamples, seed=args.seed)}
    report["comparison"] = {
        "checkpoint_primary_endpoint": 0.5089412572255549,
        "note": "Each best_* is the MAXIMUM over the lambda grid, so its interval is "
                "optimistic (winner's curse) and does not adjust for selection.",
        "cv_meaning": "held-out BACKGROUND only. Identical HCDR3 strings cross folds, so "
                      "this is an UPPER bound inflated by sequence memorisation.",
        "cv_seqdisjoint_meaning": "no HCDR3 string crosses a fold. On this population the "
                                  "HCDR3-linked components are so large that the held-out "
                                  "fold is predicted from a sliver of data, so this is a "
                                  "LOWER bound depressed by starved fits. The true "
                                  "novel-sequence value is not estimable here.",
    }
    for arm in ("cv", "cv_seqdisjoint"):
        boot = report["best_" + arm]["bootstrap"]
        print("\nbest {} {}: {:.4f}  95% CI [{:.4f}, {:.4f}]  (max over grid -> optimistic)".format(
            arm, report["best_" + arm]["key"], boot["estimate"], boot["low"], boot["high"]))
    print("checkpoint primary endpoint for comparison: 0.5089   chance floor: 0.5000")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

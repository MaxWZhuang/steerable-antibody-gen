"""Paired comparison of full-span and PLL scores on an identical frozen population.

This is an exploratory readout comparison on an already inspected validation
population. Higher PLL ranking alone does not establish antigen conditioning.
Substituted antigens remain unmeasured input ablations, not specificity labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.evaluation.contrasts import digest, file_sha256, load_json, save_json, verify_manifest
from smallAntibodyGen.evaluation.contrast_statistics import (
    _bootstrap_pairs, cluster_bootstrap, drop_largest_group, equal_group_by_target,
    group_stats, pair_pooled_by_target, paired_contrast,
)


def compare_scores(baseline, pll, manifest, *, n_resamples=20000, seed=42, alpha=.05):
    """Refuse population drift before computing paired group differences."""
    verify_manifest(manifest)
    if baseline.get("scoring_mode", "full_span") != "full_span" or pll.get("scoring_mode") != "pll":
        raise ValueError("comparison requires full_span baseline and pll scores")
    for document in (baseline, pll):
        if document["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("scores do not match the frozen manifest")
    if baseline["fold"] != pll["fold"]:
        raise ValueError("score folds differ")
    for key in ("checkpoint_sha256", "exposure_sha256", "reconstructed_train_config"):
        left, right = baseline["provenance"].get(key), pll["provenance"].get(key)
        if left is None or left != right:
            raise ValueError(f"score provenance differs or is missing: {key}")
    if digest(baseline["excluded"]) != digest(pll["excluded"]):
        raise ValueError("excluded populations differ")
    if baseline["antigen_truncated_contexts"] != pll["antigen_truncated_contexts"]:
        raise ValueError("antigen truncation differs")

    def population(document):
        groups = {}
        for group in document["groups"]:
            gid = group["group_id"]
            if gid in groups:
                raise ValueError("duplicate scored group")
            variants = {v["hcdr3"]: {k: x for k, x in v.items() if k != "scores"}
                        for v in group["variants"]}
            if len(variants) != len(group["variants"]):
                raise ValueError("duplicate scored variant")
            groups[gid] = (group["target"], group["fold"], variants)
        return groups

    if population(baseline) != population(pll):
        raise ValueError("scored groups, variants, or measurements differ")
    options = dict(n_resamples=n_resamples, seed=seed, alpha=alpha)
    native = [group_stats(doc, manifest) for doc in (baseline, pll)]
    substituted = [group_stats(doc, manifest, score_key="substituted_unmeasured", verify=False)
                   for doc in (baseline, pll)]
    # All four conditions must contribute exactly the same paired groups.
    identities = [{(s.target, s.group_id) for s in stats} for stats in native + substituted]
    if any(ids != identities[0] for ids in identities[1:]):
        raise ValueError("rankable groups differ between conditions")
    reduced = [drop_largest_group(stats) for stats in native]
    retains_targets = ({s.target for s in reduced[0]} == {s.target for s in native[0]})
    report = {
        "schema": "hcdr3-readout-comparison/1",
        "status": "exploratory comparison on previously inspected validation data",
        "endpoint": "equal-target, equal-group concordance; target-stratified group bootstrap",
        "alpha": alpha,
        "per_target_alpha": alpha / len({s.target for s in native[0]}),
        "population": {"groups": len(native[0]), "targets": len({s.target for s in native[0]}),
                       "variants": sum(s.n_variants for s in native[0]),
                       "comparable_pairs": sum(s.comparable_pairs for s in native[0])},
        "full_span": cluster_bootstrap(native[0], **options),
        "pll": cluster_bootstrap(native[1], **options),
        "pll_minus_full_span": paired_contrast(native[1], native[0], **options),
        "leave_largest_group_out_difference": (
            paired_contrast(reduced[1], reduced[0], **options) if retains_targets else
            {"estimate": None, "reason": "dropping largest groups removes an entire target"}),
        "by_target": {},
        "native_minus_substituted": {
            name: paired_contrast(a, b, **options)
            for name, a, b in zip(("full_span", "pll"), native, substituted)},
        "claims": {
            "historical_holdout_established": False,
            "antigen_conditioning_correctness_established": False,
            "interpretation": "PLL improvement supports a readout limitation for ranking; it does not by itself show antigen use. A null does not establish unlearnable labels.",
            "mask_distribution": "PLL leaves other candidate residues visible; the stage-4 checkpoint trained with full-span HCDR3 masking.",
        },
    }
    for target in sorted({s.target for s in native[0]}):
        a, b = [[s for s in stats if s.target == target] for stats in native]
        report["by_target"][target] = {
            "full_span": equal_group_by_target(a)[target], "pll": equal_group_by_target(b)[target],
            "full_span_pair_pooled": pair_pooled_by_target(a)[target],
            "pll_pair_pooled": pair_pooled_by_target(b)[target],
            "difference": paired_contrast(b, a, **dict(options, alpha=alpha / report["population"]["targets"])),
        }
    lookups = [{(s.target, s.group_id): s.concordance for s in stats}
               for stats in (native[0], native[1], substituted[0], substituted[1])]
    interaction = [(target, lookups[1][(target, gid)] - lookups[3][(target, gid)]
                    - lookups[0][(target, gid)] + lookups[2][(target, gid)])
                   for target, gid in sorted(identities[0])]
    report["change_in_native_minus_substituted"] = _bootstrap_pairs(interaction, **options)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pll", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/evaluation/hcdr3_contrasts_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    report = compare_scores(load_json(args.baseline), load_json(args.pll), load_json(args.manifest),
                            **{k: protocol[k] for k in ("n_resamples", "seed", "alpha")})
    report["provenance"] = {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (("baseline", args.baseline), ("pll", args.pll),
                           ("manifest", args.manifest), ("protocol", args.protocol),
                           ("comparison_implementation", Path(__file__)),
                           ("statistics_implementation", ROOT / "src/smallAntibodyGen/evaluation/contrast_statistics.py"))}
    save_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

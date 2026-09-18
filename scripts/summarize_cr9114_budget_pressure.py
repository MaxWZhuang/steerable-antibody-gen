#!/usr/bin/env python
"""Publish compact audited evidence and an honest report for the budget diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reference/evidence/cr9114-budget-pressure-2026-09-17.json"
REPORT = ROOT / "reference/cr9114-budget-pressure.md"


def number(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def finding(run):
    """Describe what the endpoint actually shows; never upgrade it to a claim."""
    endpoint = run["checkpoints"][str(run["config"]["steps"])]
    baseline = run["checkpoints"]["0"]
    exact = endpoint["exhaustive"]
    development = exact["conditional_affinity"]["development"]
    delta = exact["conditional_affinity_delta_from_sft"]["development"]
    ratio = (abs(delta["conditional_affinity_delta"]) / delta["assay_sem_proxy_for_delta"]
             if delta["conditional_affinity_delta"] is not None and delta["assay_sem_proxy_for_delta"] else None)
    return (f"After {run['config']['steps']:,} updates and "
            f"{run['config']['steps'] * run['config']['batch_size']:,} labelled exposures, the policy moved "
            f"{exact['total_variation_from_sft']:.4f} in total variation from SFT over the 65,536 legal identities, "
            f"with exact entropy {exact['entropy_nats']:.4f} nats (effective support "
            f"{exact['entropy_effective_support']:.1f}) against {baseline['exhaustive']['entropy_nats']:.4f} "
            f"nats at SFT. Probability assigned to development blocks changed from "
            f"{baseline['exhaustive']['split_mass']['development']:.4%} to "
            f"{exact['split_mass']['development']:.4%}. The final {endpoint['sampled_affinity']['sample_count']:,} "
            f"unconditional draws contained {endpoint['sampled_affinity']['split_counts'].get('development', 0):,} "
            f"development identities and {endpoint['diversity']['unique_genotypes']:,} distinct identities overall.\n\n"
            f"Ordinary top-16 affinity changed from {baseline['portfolios']['ordinary']['16']['mean_affinity']:.6f} "
            f"to {endpoint['portfolios']['ordinary']['16']['mean_affinity']:.6f}; top-32 changed from "
            f"{baseline['portfolios']['ordinary']['32']['mean_affinity']:.6f} to "
            f"{endpoint['portfolios']['ordinary']['32']['mean_affinity']:.6f}. "
            f"Conditional measured affinity on the 8,704 evaluated development identities moved "
            f"{number(delta['conditional_affinity_delta'])} "
            f"({'n/a' if ratio is None else f'{ratio:.1f}x'} its heuristic paired assay-SEM proxy), on "
            f"{number(development['measured_mass'], 6)} of the total probability mass with "
            f"{number(development['conditional_effective_genotypes'], 1)} effective genotypes. These are descriptive "
            f"quantities from one seeded continuation, not a significance test and not a promotion decision.")


def summarize(directory, test_log):
    run = json.loads((directory / "results.json").read_text())
    audit = json.loads((directory / "independent_audit.json").read_text())
    require(run["status"] == "completed" and audit["passes"], "Validation incomplete")
    require(not run["checkpoint_promoted"] and not run["reserved_test_labels_evaluated"]
            and not run["withheld_development_evaluated"], "Out-of-scope evaluation recorded")
    raw = test_log.read_bytes()
    log = raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8")
    match = re.search(r"(\d+) passed, (\d+) skipped(?:, (\d+) warnings)? in ([\d.]+)s", log)
    require(match is not None and "FAILED " not in log and "ERROR " not in log, "Full suite did not pass")
    steps = [0, *run["config"]["checkpoints"]]
    endpoints = [str(s) for s in run["config"]["exhaustive_checkpoints"]]
    evidence = {"schema_version": "cr9114-budget-pressure-evidence/1", "completed_date": "2026-09-17",
        "run_results_sha256": sha256(directory / "results.json"), "run": run, "independent_audit": audit,
        "validation": {"full_suite": {"passed": int(match[1]), "skipped": int(match[2]),
                                      "warnings": int(match[3] or 0), "seconds": float(match[4]),
                                      "log_sha256": sha256(test_log)}},
        "movement": {str(s): {"kl_to_sft_mc": run["checkpoints"][str(s)]["kl_to_sft_mc"],
                              "entropy_nats_mc": run["checkpoints"][str(s)]["diversity"]["entropy_nats_mc"],
                              "unique_draws": run["checkpoints"][str(s)]["diversity"]["unique_genotypes"],
                              "mean_pairwise_hamming_unbiased":
                                  run["checkpoints"][str(s)]["diversity"]["mean_pairwise_hamming_unbiased"],
                              "ordinary_top16_affinity":
                                  run["checkpoints"][str(s)]["portfolios"]["ordinary"]["16"]["mean_affinity"],
                              "ordinary_top32_affinity":
                                  run["checkpoints"][str(s)]["portfolios"]["ordinary"]["32"]["mean_affinity"],
                              "shared_with_sft_top16":
                                  run["checkpoints"][str(s)]["portfolios"]["ordinary"]["16"]["shared_with_sft"],
                              "pool_conditional_affinity": run["checkpoints"][str(s)]["pool_conditional_affinity"],
                              "pool_conditional_delta_from_sft":
                                  run["checkpoints"][str(s)]["pool_conditional_delta_from_sft"],
                              "sampled_affinity": run["checkpoints"][str(s)]["sampled_affinity"]}
                     for s in steps},
        "exact_endpoints": {s: run["checkpoints"][s]["exhaustive"] for s in endpoints}}
    save_json(EVIDENCE, evidence)

    lines = ["# Affinity-only training-budget stress test", "",
        f"2026-09-17. One predeclared continuation of {run['config']['steps']:,} updates completed; "
        "no checkpoint was selected or promoted.", "", finding(run), "",
        "## Policy movement by checkpoint", "",
        "The Monte Carlo columns come from 1,024 actual draws and carry sampling error. The",
        "two affinity columns are a top-K ranking of a fixed 2,048-identity development pool,",
        "not a property of the generated distribution; the continuous pool quantities and the",
        "affinity of the actual draws follow in the next two subsections. Exact distribution",
        "statistics exist only at the two exhaustively scored endpoints, further below.", "",
        "| Updates | Labelled exposures | KL to SFT (MC, nats) | MC SE | Entropy (MC, nats) | Unique / 1,024 | "
        "Top-16 affinity | Top-32 affinity | Top-16 shared with SFT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step in steps:
        cell = run["checkpoints"][str(step)]
        ordinary = cell["portfolios"]["ordinary"]
        lines.append(f"| {step:,} | {step * run['config']['batch_size']:,} | {cell['kl_to_sft_mc']['nats']:.4f} | "
                     f"{cell['kl_to_sft_mc']['standard_error']:.4f} | {cell['diversity']['entropy_nats_mc']:.4f} | "
                     f"{cell['diversity']['unique_genotypes']} | {ordinary['16']['mean_affinity']:.6f} | "
                     f"{ordinary['32']['mean_affinity']:.6f} | {ordinary['16']['shared_with_sft']}/16 |")
    lines += ["", "### Fixed-pool continuous affinity and mass", "",
        "Continuous quantities over that same fixed pool. Pool scores are constrained joint",
        "log probabilities over the full 65,536-point support, so the mass column is the true",
        "unconditional mass the pool's measured identities carry; the mean is conditional on",
        "them. Both are available at every checkpoint, not only at the endpoints.", "",
        "| Updates | Measured mass on the pool | Conditional mean | Effective genotypes | Delta vs SFT | "
        "Paired SEM proxy | Mass above threshold |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for step in steps:
        cell = run["checkpoints"][str(step)]["pool_conditional_affinity"]
        delta = run["checkpoints"][str(step)]["pool_conditional_delta_from_sft"]
        lines.append(f"| {step:,} | {number(cell['measured_mass'], 9)} | "
                     f"{number(cell['conditional_mean_affinity'])} | "
                     f"{number(cell['conditional_effective_genotypes'], 1)} | "
                     f"{number(delta['conditional_affinity_delta'])} | "
                     f"{number(delta['assay_sem_proxy_for_delta'])} | {number(cell['positive_mass'], 9)} |")
    lines += ["", "### Conditional affinity of the actual generated draws", "",
        "1,024 native draws per checkpoint, duplicates retained. Only draws that land on an",
        "identity carrying a usable measurement inside the evaluated whitelist enter a mean;",
        "the rest are counted as unscored rather than dropped. The sampling SE is the spread",
        "of the measured draws themselves and does not include assay uncertainty, which is",
        "the separate proxy column. A mean over very few draws is a noisy quantity, not a",
        "small effect.", "",
        "| Updates | Population | Measured draws | Conditional mean | Sampling SE | Assay-SEM proxy | "
        "Distinct genotypes | Unscored draws |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for step in steps:
        sampled = run["checkpoints"][str(step)]["sampled_affinity"]
        for name, cell in sampled["by_split"].items():
            lines.append(f"| {step:,} | {name} | {cell['measured_draw_count']}/{sampled['sample_count']:,} | "
                         f"{number(cell['conditional_mean_affinity'])} | "
                         f"{number(cell['sampling_sem_conditional_mean'])} | "
                         f"{number(cell['assay_sem_proxy'])} | {cell['distinct_measured_genotypes']} | "
                         f"{sampled['unscored_draw_count']}/{sampled['sample_count']:,} |")
    lines += ["", "## Exact distribution at the two exhaustively scored endpoints", "",
        "Every legal identity was scored without any affinity lookup. Mass by split is",
        "partly mechanical: split membership is a deterministic function of four of the",
        "sixteen editable sites, so movement there can reflect block drift rather than",
        "affinity learning. Per-site marginals and per-block mass are reported for that reason.", "",
        "| Endpoint | Support mass | Entropy, nats | Effective support | Collision | Max atom | KL to SFT | "
        "KL from SFT | Total variation | Train mass | Development mass | Test mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step in endpoints:
        exact = run["checkpoints"][step]["exhaustive"]
        lines.append(f"| {int(step):,} | {exact['normalization_mass_before_correction']:.9f} | "
                     f"{exact['entropy_nats']:.4f} | {exact['entropy_effective_support']:.1f} | "
                     f"{exact['collision_probability']:.3e} | {exact['maximum_genotype_probability']:.3e} | "
                     f"{exact['kl_to_sft_nats']:.4f} | {exact['kl_from_sft_nats']:.4f} | "
                     f"{exact['total_variation_from_sft']:.4f} | {exact['split_mass']['train']:.4f} | "
                     f"{exact['split_mass']['development']:.4f} | {exact['split_mass']['test']:.4f} |")
    lines += ["", "Allele-1 marginal per editable site. The four split-defining sites are marked.", "",
        "| Site | Split-defining | " + " | ".join(f"{int(s):,}" for s in endpoints) + " |",
        "|---|---|" + "---:|" * len(endpoints)]
    for site in range(16):
        marked = "yes" if site in run["config"]["split_defining_loci_0based"] else ""
        values = " | ".join(f"{run['checkpoints'][s]['exhaustive']['site_allele_one_marginals'][site]:.4f}"
                            for s in endpoints)
        lines.append(f"| {site} | {marked} | {values} |")
    blocks = run["checkpoints"][endpoints[0]]["exhaustive"]["block_mass"]
    lines += ["", "Exact mass per split-defining block. The block is the four split-defining digits,",
        "so this is the finest partition the train/development/test split is a function of;",
        "the three blocks the evaluated development pool draws from are marked.", "",
        "| Block | Development block | " + " | ".join(f"{int(s):,}" for s in endpoints) + " |",
        "|---|---|" + "---:|" * len(endpoints)]
    for key in sorted(blocks, key=int):
        marked = "yes" if int(key) in run["config"]["development_blocks"] else ""
        values = " | ".join(f"{run['checkpoints'][s]['exhaustive']['block_mass'][key]:.6f}" for s in endpoints)
        lines.append(f"| {key} | {marked} | {values} |")
    lines += ["", "## Conditional measured affinity", "",
        "These means are conditional on identities that carry a usable measurement and are",
        "inside the evaluated whitelist. They are not unconditional generated affinity. The",
        "assay-SEM proxy assumes independent genotype errors and rests on a shared pooled",
        "floor, so it is a heuristic scale, not a calibrated noise floor or a significance",
        "test. Shared identities cancel in the paired delta, which is the headline column.",
        "Training affinity is exposure, not generalization.", "",
        "| Endpoint | Population | Measured mass | Conditional mean | Effective genotypes | Distinct genotypes | "
        "Assay-SEM proxy | Delta vs SFT | Paired SEM proxy | Mass above threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step in endpoints:
        exact = run["checkpoints"][step]["exhaustive"]
        for name, cell in exact["conditional_affinity"].items():
            delta = exact["conditional_affinity_delta_from_sft"][name]
            lines.append(f"| {int(step):,} | {name} | {number(cell['measured_mass'])} | "
                         f"{number(cell['conditional_mean_affinity'])} | "
                         f"{number(cell['conditional_effective_genotypes'], 1)} | "
                         f"{cell['distinct_measured_genotypes']} | {number(cell['assay_sem_proxy'])} | "
                         f"{number(delta['conditional_affinity_delta'])} | "
                         f"{number(delta['assay_sem_proxy_for_delta'])} | {number(cell['positive_mass'])} |")
    lines += ["", "Probability mass by measurement availability, in six categories that partition the",
        "support, plus their total. Measured train and measured development stay separate.",
        "No missing outcome is imputed and no bias direction is asserted: eligibility mixes",
        "assay-floor censoring with under-replication, and the eligible table records",
        "survivors only. The conditional mean may therefore be biased in a direction and by a",
        "magnitude this study does not establish.", "",
        "| Endpoint | " + " | ".join(name.replace("_", " ") for name in
                                     run["checkpoints"][endpoints[0]]["exhaustive"]["category_mass"]) + " |",
        "|---|" + "---:|" * len(run["checkpoints"][endpoints[0]]["exhaustive"]["category_mass"])]
    for step in endpoints:
        table = run["checkpoints"][step]["exhaustive"]["category_mass"]
        lines.append(f"| {int(step):,} | " + " | ".join(f"{value:.6f}" for value in table.values()) + " |")
    lines += ["", "## Exposure overlap", "",
        "The original SFT run drew its own labelled batches. Overlap is reported against",
        "those identities, against this continuation's schedule prefix, and against the union.", "",
        "| Updates | SFT-exposed identities | Continuation-exposed identities | Union | Draws hitting the union |",
        "|---|---:|---:|---:|---:|"]
    for step in steps:
        counts = run["checkpoints"][str(step)]["exposure_counts"]
        overlap = run["checkpoints"][str(step)]["sampled_exposure_overlap"]["union"]
        lines.append(f"| {step:,} | {counts['sft_unique']} | {counts['continuation_unique']} | "
                     f"{counts['union_unique']} | {overlap['draw_count']}/1,024 |")
    lines += ["", "## What was fixed before fitting", "",
        f"The [protocol](../specs/cr9114_budget_pressure.md) and code were committed at "
        f"`{run['git_commit'][:12]}` before any update. Budget is the only intervention: the",
        "objective is the same unweighted NLL/16 over the same affinity-weighted training",
        "population, with no reference KL, entropy or embedding penalty, the same frozen",
        "encoder, the same seeded schedule, learning rate, weight decay and gradient clip.",
        "One AdamW object serves all updates; evaluation runs between steps without an",
        "optimizer and is asserted not to change decoder state, optimizer state or RNG.",
        f"The 256-update decoder reproduced the earlier affinity-only arm's state digest",
        f"`{run['expected_step_256_decoder_state_sha256'][:12]}` exactly, which is the",
        "reproduction claim; the checkpoint files differ because their metadata differs.",
        "Checkpoints are fixed at 256, 2,048, 8,192 and 16,384 updates and all are reported.", "",
        f"The weighted training target has mean affinity "
        f"{run['config']['training_population_weighted_expected_affinity']:.6f}. That is a benchmark, not a",
        "bound: it is the mean of the target distribution, and a policy concentrated on a",
        "strong measured identity exceeds it. It is not a convergence criterion or a gate.", "",
        f"The best measured training identity is {run['training_population_max_measured_affinity']:.6f}. That value *is* an upper",
        "bound, but only on a conditional average taken over these measured training",
        "identities, since such an average is a weighted mean of measured values and cannot",
        "exceed the largest of them. It is not a bound on conditional development affinity,",
        "on affinity outside the measured set, or on what the model could reach at an",
        "identity nobody has assayed.", "",
        "## Validation and limits", "",
        f"The [independent audit](evidence/cr9114-budget-pressure-2026-09-17.json) re-derived the",
        f"reported statistics with its own formulas over {audit['output_artifacts_checked']} saved artifacts: the schedule and",
        "the continuation exposures, the training population, the original SFT exposure",
        "replay and its generator state, the development whitelist and pool, the exact",
        "endpoint distributions and their normalization gate, the label-blind selections and",
        "portfolios, and the sampled diversity, KL, Hamming, conditional-affinity, split and",
        "exposure statistics with their standard errors. It does not re-derive recorded",
        "metadata: library versions and the strict-reload, optimizer-carried-forward,",
        "encoder-unchanged and RNG-invariance assertions were checked live during the run and",
        "are read back here; timing fields are only checked for internal consistency; and",
        "training itself is not replayed. Perturbing the 3,046 withheld development",
        "measurements changed no reported number, and a synthetic test-labelled row is",
        "rejected rather than filtered away.",
        f"The full test suite passed: {match[1]} passed, {match[2]} skipped.", "",
        "Monte Carlo KL and entropy remain estimates with sampling error at every",
        "checkpoint, including the exhaustively scored endpoints, where they estimate the",
        "exact value rather than equal it. The endpoint normalization check bounds float",
        "accumulation at the endpoints; it does not bound error at the intermediate",
        "checkpoints, which were never exhaustively scored. Intermediate pool statistics",
        "are true unconditional mass on a 2,048-identity subset, not a distribution summary.", "",
        "Only training measurements and the 8,704 previously evaluated eligible development",
        "identities entered evaluation. The remaining 3,046 eligible development",
        "measurements and every reserved test measurement stayed unused. This is a finite,",
        "previously assayed 16-site edit space around one antibody lineage on one antigen;",
        "nothing here is evidence about sequences outside that space, another lineage, or a",
        "new binding assay. No checkpoint was promoted.", "",
        "The model conditions on the [prepared 5CJQ context](cr9114-5cjq-context.md), which is",
        "a different construct from the one the affinities were measured on: an engineered",
        "stem trimer standing in for the assayed H1 ectodomain, and Fv geometry standing in",
        "for the assayed scFv, with two declared VH template differences and fragment",
        "separators where antigen coordinates are missing. No missing geometry was filled.",
        "Affinities remain the original experimental measurements. The construct mismatch",
        "limits conclusions about how well the structural conditioning transfers to the",
        "assay construct.", "",
        "[Compact evidence](evidence/cr9114-budget-pressure-2026-09-17.json) carries the full",
        "result document, the independent audit and every artifact hash.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(finding(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/cr9114_budget_pressure_20260917")
    parser.add_argument("--test-log", type=Path, default=ROOT / "outputs/cr9114_budget_pressure_final_tests.log")
    args = parser.parse_args()
    summarize(args.directory, args.test_log)

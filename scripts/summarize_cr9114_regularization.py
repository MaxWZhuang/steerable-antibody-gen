#!/usr/bin/env python
"""Publish compact, audited evidence for the fixed regularization comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory, test_log):
    run = json.loads((directory / "results.json").read_text())
    audit = json.loads((directory / "independent_audit.json").read_text())
    replay = json.loads((directory / "deterministic_replay.json").read_text())
    require(run["status"] == "completed" and audit["passes"] and replay["passes"], "Validation incomplete")
    raw = test_log.read_bytes()
    log = raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8")
    match = re.search(r"(\d+) passed, (\d+) skipped, (\d+) warnings in ([\d.]+)s", log)
    require(match is not None and "FAILED " not in log and "ERROR " not in log, "Full suite did not pass")
    paired = {}
    for name, result in run["arms"].items():
        control = run["arms"][f"seed_{result['seed']}_affinity"]
        paired[name] = {"versus_same_seed_affinity": {
            mode: {k: {"mean_affinity_delta": cell["mean_affinity"] - control["portfolios"][mode][k]["mean_affinity"],
                       "mean_hamming_delta": cell["diversity"]["mean_hamming"] - control["portfolios"][mode][k]["diversity"]["mean_hamming"],
                       "frozen_embedding_cosine_delta": cell["frozen_sft_embedding_cosine"] - control["portfolios"][mode][k]["frozen_sft_embedding_cosine"]}
                   for k, cell in cells.items()} for mode, cells in result["portfolios"].items()},
            "global_vs_control": {"entropy_delta": result["diversity"]["entropy_nats_mc"] - control["diversity"]["entropy_nats_mc"],
                                  "hamming_delta": result["diversity"]["mean_pairwise_hamming_unbiased"] - control["diversity"]["mean_pairwise_hamming_unbiased"],
                                  "frozen_embedding_cosine_delta": result["frozen_sft_embedding_cosine"] - control["frozen_sft_embedding_cosine"],
                                  "live_embedding_cosine_delta": result["live_embedding_cosine"] - control["live_embedding_cosine"]}}
    summary = {}
    for arm in run["config"]["arms"]:
        cells = [run["arms"][f"seed_{seed}_{arm['name']}"] for seed in run["config"]["seeds"]]
        summary[arm["name"]] = {
            k: {"ordinary_affinity_mean": float(np.mean([r["portfolios"]["ordinary"][k]["mean_affinity"] for r in cells])),
                "ordinary_affinity_seed_sd": float(np.std([r["portfolios"]["ordinary"][k]["mean_affinity"] for r in cells], ddof=1)),
                "ordinary_hamming_mean": float(np.mean([r["portfolios"]["ordinary"][k]["diversity"]["mean_hamming"] for r in cells])),
                "diverse_affinity_mean": float(np.mean([r["portfolios"]["diverse"][k]["mean_affinity"] for r in cells])),
                "diverse_hamming_mean": float(np.mean([r["portfolios"]["diverse"][k]["diversity"]["mean_hamming"] for r in cells]))}
            for k in ("16", "32")}
    evidence = {"schema_version": "cr9114-regularization-evidence/1", "completed_date": "2026-09-17",
        "run_results_sha256": sha256(directory / "results.json"), "run": run, "independent_audit": audit,
        "deterministic_replay": replay, "paired_comparisons": paired, "three_seed_summary": summary,
        "validation": {"full_suite": {"passed": int(match[1]), "skipped": int(match[2]), "warnings": int(match[3]),
                                      "seconds": float(match[4]), "log_sha256": sha256(test_log)}}}
    path = ROOT / "reference/evidence/cr9114-regularization-2026-09-17.json"
    save_json(path, evidence)
    lines = ["# Matched reference-KL and diversity comparison", "", "2026-09-17. All twelve predeclared runs completed; no checkpoint was promoted.", "",
        "## Measured results", "", "All affinities below use the same 2,048 development candidates and selected sets.",
        "Higher measured H1 affinity is better; Hamming counts differing editable sites.", "",
        "| Seed | Arm | Ordinary top-16 affinity | Ordinary top-32 affinity | Top-32 Hamming | Diverse top-32 affinity | Diverse top-32 Hamming |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name, r in [("SFT", run["sft"]), *run["arms"].items()]:
        ordinary, diverse = r["portfolios"]["ordinary"], r["portfolios"]["diverse"]["32"]
        lines.append(f"| {r.get('seed', '-')} | {r.get('arm', {}).get('name', 'SFT')} | {ordinary['16']['mean_affinity']:.6f} | {ordinary['32']['mean_affinity']:.6f} | {ordinary['32']['diversity']['mean_hamming']:.4f} | {diverse['mean_affinity']:.6f} | {diverse['diversity']['mean_hamming']:.4f} |")
    lines += ["", "## Sampling and representation diagnostics", "",
        "These concern unconditional policy samples, not just the selected high-score sets.",
        "Entropy and KL are Monte Carlo estimates; the evidence includes standard errors.",
        "A learned cosine improvement needs corroboration from fixed features and sequences.", "",
        "| Seed | Arm | Unique / 1,024 | Entropy, nats | Hamming | KL to SFT, nats | Frozen cosine | Live cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in [("SFT", run["sft"]), *run["arms"].items()]:
        d = r["diversity"]
        lines.append(f"| {r.get('seed', '-')} | {r.get('arm', {}).get('name', 'SFT')} | {d['unique_genotypes']} | {d['entropy_nats_mc']:.4f} | {d['mean_pairwise_hamming_unbiased']:.4f} | {r['kl_to_sft_mc']['nats']:.4f} | {r['frozen_sft_embedding_cosine']:.6f} | {r['live_embedding_cosine']:.6f} |")
    lines += ["", "## How the comparison isolates the mechanism", "",
        "The [protocol](../specs/cr9114_shortlist.md) was committed before the new shortlist",
        "development result. Each arm starts from the same SFT checkpoint, with a frozen",
        "encoder and fresh AdamW optimizer. Within each seed, the four arms receive the",
        "same 1,024 labelled sequence exposures over 256 updates. Only the regularizer",
        "changes. All use the existing training-only affinity-weighted likelihood.", "",
        "The objective is NLL/16, plus 0.1*KL(q||SFT)/16 when enabled. The entropy arm",
        "also subtracts 0.1*H(q)/16. The embedding arm instead adds 0.1 times mean",
        "off-diagonal cosine of normalized mean-pooled last decoder features. Thus the",
        "entropy coefficient here is 0.00625 per joint-sequence nat, smaller than the",
        "previous pilot's 0.03. These are fixed starting coefficients, not an optimized",
        "or equal-gradient-strength comparison.", "",
        "Eight fresh samples every four updates supply KL and entropy score-function",
        "gradients; leave-one-out baselines and duplicate samples are retained. The",
        "gradients are multiplied by four to account for update frequency. A separate",
        "frozen SFT decoder supplies reference probabilities. The embedding penalty",
        "differentiates the trainable representation of detached generated identities;",
        "it is not a frozen-feature reward and does not include a distribution-gradient",
        "term through the discrete sample. All 121 valid causal prediction contexts",
        "are pooled, including fixed scaffold positions.", "",
        "This is a [ProteinZero-inspired](https://arxiv.org/html/2506.07459v4) adaptation,",
        "not a reproduction of its full online reward/GRPO system. The affinity objective",
        "remains weighted SFT; it is not [ProteinDPO's](https://www.nature.com/articles/s41592-026-03137-3)",
        "scalar-label reference-relative DPO objective.", "",
        "## Validation and limits", "",
        f"The independent audit verified {audit['output_artifacts_checked']} output artifacts and all twelve",
        "checkpoint identities, independently recomputed schedules, selections, affinity,",
        "sequence metrics, reference KL and both embedding metrics. A separate-process",
        "replay of the first KL+embedding arm reproduced the final decoder bit for bit.",
        f"The full test suite passed: {match[1]} passed, {match[2]} skipped ({match[3]} warnings).",
        "Native-model tests check feature-path score/gradient parity, and an evaluation",
        "test confirms that changing development labels cannot change selected identities.", "",
        "This cohort was reused from the completed shortlist study, whose combined",
        "quality/diversity screen failed at K=32. The admission rule remains locked to",
        "the top 4*K model scores. Test measurements remain reserved, and no additional",
        "fresh development measurements were consumed by training comparison. No",
        "checkpoint, seed or coefficient was selected from these results. The three",
        "continuation seeds share one SFT initialization, one antibody lineage, and",
        "three development blocks; this is not evidence against all forms of overfitting.",
        "Frozen SFT features provide a fixed numerical diagnostic, not an independent",
        "biological assay. Uniqueness is near its sample-size ceiling and connected",
        "sequence clusters are not validated biological modes.", "",
        "[Compact evidence](evidence/cr9114-regularization-2026-09-17.json) includes",
        "per-seed results, selected identities, paired comparisons, hashes and audits.", ""]
    (ROOT / "reference/cr9114-regularization.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/cr9114_regularization_20260917")
    parser.add_argument("--test-log", type=Path, default=ROOT / "outputs/cr9114_regularization_final_tests.log")
    args = parser.parse_args()
    summarize(args.directory, args.test_log)

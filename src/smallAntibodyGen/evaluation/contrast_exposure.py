"""Separate source assignment, recorded eligibility, and ancestor sequence contact.

No-contact against a current snapshot is not proof of historical non-exposure.
Exact shared HCDR3/background contacts are reported as relatedness indicators;
fuzzy neighbourhood coverage is explicitly unmeasured in this first audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smallAntibodyGen.ancestor_quarantine import stage1_contributes_hcdr3
from smallAntibodyGen.data.affinity import infer_is_strong_binder
from .contrasts import digest, file_sha256, heavy_context, iter_jsonl, verify_manifest


def variant_identity(group: dict, variant: dict) -> str:
    return digest([group["group_id"], variant["hcdr3"]])


def stage_eligible(row: dict, cfg: dict) -> bool:
    stage = cfg["training_stage"]
    if stage in ("base", "paired_refine"):
        return True
    if stage == "antigen_real_label_refine":
        return row.get("binder_label") in (0, 1) or (
            cfg.get("include_strength_rows", False)
            and isinstance(row.get("affinity_strength_quantile"), (int, float))
            and not isinstance(row.get("affinity_strength_quantile"), bool))
    if stage == "antigen_hcdr3_infill_refine":
        # Match the reader and mlm_train.is_hcdr3_infill_record, including its
        # legacy coordinate fallback. Eligibility is deliberately distinct from
        # the benchmark's stricter sequence/coordinate consistency predicate.
        start, end, cdr3 = (row.get(k) for k in
                            ("cdr3_start_aa_heavy", "cdr3_end_aa_heavy", "cdr3_aa_heavy"))
        if start is None or end is None:
            start, end, cdr3 = (row.get(k) for k in ("cdr3_start_aa", "cdr3_end_aa", "cdr3_aa"))
        span_valid = (isinstance(start, int) and isinstance(end, int) and end > start
                      and isinstance(cdr3, str) and len(cdr3) == end - start)
        return bool(infer_is_strong_binder(row) and span_valid
                    and (row.get("sequence_heavy") or row.get("sequence") or "").strip()
                    and (row.get("sequence_antigen") or "").strip())
    raise ValueError(f"eligibility reconstruction is not supported for stage {stage!r}")


def checkpoint_chain(checkpoint: Path, root: Path, overrides: dict[str, str]) -> list[dict]:
    """Follow saved parents; missing parents fail rather than shorten the audit."""
    import torch
    chain, seen = [], set()
    while checkpoint:
        checkpoint = checkpoint.resolve()
        if checkpoint in seen:
            raise ValueError("checkpoint ancestry contains a cycle")
        seen.add(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = saved.get("train_config")
        if not isinstance(cfg, dict) or not cfg.get("data_path"):
            raise ValueError(f"checkpoint lacks a saved training configuration: {checkpoint}")
        stage = cfg.get("training_stage", "base")
        cfg = dict(cfg, training_stage=stage)
        corpus_text = overrides.get(stage, cfg["data_path"])
        corpus = Path(corpus_text)
        if not corpus.is_absolute():
            corpus = root / corpus
        if not corpus.is_file():
            raise FileNotFoundError(f"ancestor corpus is unavailable: {corpus}")
        chain.append({"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                      "epoch": saved.get("epoch"), "config": cfg, "corpus": str(corpus.resolve()),
                      "corpus_path_overridden": stage in overrides,
                      "training_snapshot_verified": False,
                      "snapshot_reason": "saved configuration does not authenticate the historical training-corpus bytes"})
        parent = cfg.get("init_checkpoint")
        del saved
        checkpoint = (Path(parent) if Path(parent).is_absolute() else root / parent) if parent else None
    return list(reversed(chain))


def audit_exposure(manifest: dict, stages: list[dict], *, progress=print) -> dict[str, Any]:
    verify_manifest(manifest)
    kinds = ("exact_heavy", "exact_antibody", "exact_biology", "exact_hcdr3", "exact_background")
    indices = {k: defaultdict(set) for k in kinds}
    variants = {}
    for g in manifest["groups"]:
        for v in g["variants"]:
            vid = variant_identity(g, v)
            h = g["prefix"] + v["hcdr3"] + g["suffix"]
            keys = (h, (h, g["light"]), (h, g["light"], g["antigen"]), v["hcdr3"],
                    (g["prefix"], g["suffix"], g["light"]))
            for kind, key in zip(kinds, keys):
                indices[kind][key].add(vid)
            variants[vid] = {"group_id": g["group_id"], "target": g["target"],
                             "record_id": v["record_id"], "source_split": v["source_split"],
                             "stages": {}}
    # Stages using the same corpus share one streaming pass.
    by_corpus = defaultdict(list)
    for i, stage in enumerate(stages):
        by_corpus[stage["corpus"]].append((i, stage))
    stage_counts = [Counter() for _ in stages]
    for corpus, selected in by_corpus.items():
        progress(f"audit: {corpus}")
        corpus_hash = file_sha256(corpus)
        for _, s in selected:
            s["corpus_sha256"] = corpus_hash
        for n, row in enumerate(iter_jsonl(corpus), 1):
            if n % 500000 == 0:
                progress(f"audit: scanned {n:,} rows")
            phase = {"train": "training", "val": "selection"}.get(row.get("split"))
            if phase is None:
                continue
            h = row.get("sequence_heavy") or row.get("sequence") or ""
            light = row.get("sequence_light") or ""
            antigen = row.get("sequence_antigen") or ""
            context = heavy_context(row)
            cdr3 = row.get("cdr3_aa_heavy") or row.get("cdr3_aa") or ""
            keys = (h, (h, light), (h, light, antigen), cdr3,
                    context[:3] if context else None)
            matches = {kind: indices[kind].get(key, ()) for kind, key in zip(kinds, keys)}
            for i, stage in selected:
                cfg = stage["config"]
                stage_counts[i]["assigned_" + phase] += 1
                eligible = stage_eligible(row, cfg)
                if eligible:
                    stage_counts[i]["eligible_" + phase] += 1
                # Assignment is exact triple membership, even if the row is ineligible.
                for vid in matches["exact_biology"]:
                    entry = variants[vid]["stages"].setdefault(str(i), set())
                    entry.add("assigned_" + phase)
                    if eligible:
                        entry.add("eligible_" + phase)
                if not eligible:
                    continue
                if cfg["training_stage"] == "base" and not stage1_contributes_hcdr3(row):
                    continue
                for kind, matched in matches.items():
                    for vid in matched:
                        variants[vid]["stages"].setdefault(str(i), set()).add(phase + ":" + kind)
        for i, _ in selected:
            stages[i]["population_counts"] = dict(stage_counts[i])
        if file_sha256(corpus) != corpus_hash:
            raise RuntimeError(f"ancestor corpus changed during audit: {corpus}")
    per_target = {}
    for target in sorted({v["target"] for v in variants.values()}):
        subset = [v for v in variants.values() if v["target"] == target]
        totals = Counter()
        for v in subset:
            flags = set().union(*v["stages"].values()) if v["stages"] else set()
            totals.update(flags)
        per_target[target] = {"variants": len(subset), "variants_with_contact": dict(sorted(totals.items()))}
    for v in variants.values():
        v["stages"] = {k: sorted(flags) for k, flags in sorted(v["stages"].items())}
    return {
        "schema": "hcdr3-contrast-exposure/1", "manifest_sha256": manifest["manifest_sha256"],
        "claims": {"historical_holdout_established": False,
                   "eligibility": "reconstructed from saved stage settings and supplied snapshot fields; not a per-example training trace",
                   "selection": "validation contact recorded because it could select best.pt",
                   "relatedness_measured": ["exact shared heavy", "exact shared HCDR3", "exact background with light chain"],
                   "fuzzy_hcdr3_neighborhood": "unmeasured; no neighbourhood-disjoint claim",
                   "absence": "no exact contact in supplied snapshots cannot authenticate historical non-exposure"},
        "stages": stages, "per_target": per_target, "variants": variants,
    }

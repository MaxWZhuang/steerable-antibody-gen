"""Deterministic fully masked / PLL scores and unmeasured antigen substitutions."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

import torch

from smallAntibodyGen.infill.hcdr3 import HCDR3Span, encode_masked_hcdr3_ids
from .contrasts import concordance, digest, verify_manifest


def group_record(group, variant):
    h = group["prefix"] + variant["hcdr3"] + group["suffix"]
    return SimpleNamespace(sequence_heavy=h, sequence=h, sequence_light=group["light"],
                           sequence_antigen=group["antigen"], heavy_locus=group["heavy_locus"],
                           light_locus=group["light_locus"], locus="PAIRED_ANTIGEN",
                           cdr3_aa_heavy=variant["hcdr3"], cdr3_start_aa_heavy=len(group["prefix"]),
                           cdr3_end_aa_heavy=len(group["prefix"]) + len(variant["hcdr3"]))


@torch.no_grad()
def score_manifest(manifest, infiller, *, fold="validation", batch_size=8, progress=print,
                   scoring_mode="full_span"):
    verify_manifest(manifest)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if scoring_mode not in ("full_span", "pll"):
        raise ValueError("scoring_mode must be full_span or pll")
    selected = [g for g in manifest["groups"] if fold == "all" or g["fold"] == fold]
    if not selected:
        raise ValueError(f"no contrast groups in fold {fold!r}")
    infiller.model.eval()
    # One forward per group/length/condition, not one per candidate sequence.
    tasks, excluded = [], []
    token_hashes = []
    for g in selected:
        lengths = defaultdict(list)
        for v in g["variants"]:
            lengths[len(v["hcdr3"])].append(v)
        temporary = []
        conditions = {"native": g["antigen"]}
        if g["antigen_control"]:
            conditions["substituted_unmeasured"] = g["antigen_control"]["sequence"]
        failure = None
        for length, variants in sorted(lengths.items()):
            record = group_record(g, variants[0])
            span = HCDR3Span.from_record(record)
            ids, positions, _, _ = encode_masked_hcdr3_ids(
                infiller.tokenizer, record, span, proposed_length=length)
            if len(ids) > infiller.max_length:
                failure = "complete_antibody_exceeds_checkpoint_context"
                break
            for condition, antigen in conditions.items():
                ag_ids = infiller.antigen_tokenizer.encode(antigen, infiller._antigen_encode_max_length)
                full_ag_ids = infiller.antigen_tokenizer.encode(antigen, len(antigen) + 8)
                item = {"group": g, "variants": variants, "condition": condition,
                        "ids": ids, "positions": positions, "antigen_ids": ag_ids,
                        "antigen_truncated": len(full_ag_ids) > len(ag_ids)}
                temporary.append(item)
        if failure:
            excluded.append({"group_id": g["group_id"], "target": g["target"], "reason": failure})
        else:
            tasks.extend(temporary)
    if not tasks:
        raise ValueError("no complete contrast group fits the checkpoint context")
    tasks.sort(key=lambda t: (len(t["ids"]), len(t["antigen_ids"]), t["group"]["group_id"], t["condition"]))
    truncated = Counter()
    for task in tasks:
        if task["antigen_truncated"]:
            truncated[task["group"]["target"] + ":" + task["condition"]] += 1
    if scoring_mode == "pll":
        expanded = []
        for task in tasks:
            for variant in task["variants"]:
                visible_ids = list(task["ids"])
                for position, aa in zip(task["positions"], variant["hcdr3"]):
                    visible_ids[position] = infiller.tokenizer.token_to_id[aa]
                for offset, position in enumerate(task["positions"]):
                    ids = list(visible_ids)
                    ids[position] = infiller.tokenizer.mask_id
                    expanded.append(dict(task, ids=ids, positions=[position],
                                         variants=[variant], residue_offset=offset))
        tasks = expanded
    scores = defaultdict(lambda: defaultdict(dict))
    pll_terms = defaultdict(list)
    canonical = infiller.canonical_token_ids
    aa_index = {infiller.tokenizer.id_to_token[token]: i for i, token in enumerate(canonical)}
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        ab_len = max(len(t["ids"]) for t in batch)
        ag_len = max(len(t["antigen_ids"]) for t in batch)
        def pad(field, width, pad_id):
            ids = torch.full((len(batch), width), pad_id,
                             dtype=torch.long, device=infiller.device)
            attention = torch.zeros_like(ids)
            for i, task in enumerate(batch):
                values = task[field]
                ids[i, :len(values)] = torch.tensor(values, device=infiller.device)
                attention[i, :len(values)] = 1
            return ids, attention
        ab, ab_mask = pad("ids", ab_len, infiller.tokenizer.pad_id)
        ag, ag_mask = pad("antigen_ids", ag_len, infiller.antigen_tokenizer.pad_id)
        logits, _ = infiller.model(antibody_input_ids=ab, antibody_attention_mask=ab_mask,
                                   antigen_input_ids=ag, antigen_attention_mask=ag_mask)
        for i, task in enumerate(batch):
            logp = torch.log_softmax(logits[i, task["positions"]][:, canonical].float(), dim=-1).double().cpu()
            if not torch.isfinite(logp).all():
                raise ValueError("model produced nonfinite HCDR3 scores")
            gid = task["group"]["group_id"]
            condition = task["condition"]
            for v in task["variants"]:
                residues = (v["hcdr3"][task["residue_offset"]]
                            if scoring_mode == "pll" else v["hcdr3"])
                labels = [aa_index[aa] for aa in residues]
                value = logp[torch.arange(len(labels)), labels].mean().item()
                if scoring_mode == "pll":
                    pll_terms[(gid, condition, v["hcdr3"])].append(value)
                else:
                    scores[gid][condition][v["hcdr3"]] = value
            token_hashes.append(digest({"group_id": gid, "condition": condition,
                                       "ids": task["ids"], "antigen_ids": task["antigen_ids"],
                                       "target_positions": task["positions"]}))
        if start % (batch_size * 50) == 0:
            progress(f"score: {min(start + batch_size, len(tasks)):,}/{len(tasks):,} context forwards")
    results, per_target = [], defaultdict(list)
    for (gid, condition, hcdr3), terms in pll_terms.items():
        if len(terms) != len(hcdr3):
            raise ValueError("PLL must score every HCDR3 residue exactly once")
        scores[gid][condition][hcdr3] = sum(terms) / len(terms)
    for g in selected:
        if g["group_id"] not in scores:
            continue
        conditions = scores[g["group_id"]]
        rows = [dict(v, scores={name: mapping[v["hcdr3"]] for name, mapping in conditions.items()})
                for v in g["variants"]]
        native = concordance([r["measurement"] for r in rows], [r["scores"]["native"] for r in rows],
                             direction=g["direction"])
        shifts = [r["scores"]["substituted_unmeasured"] - r["scores"]["native"] for r in rows
                  if "substituted_unmeasured" in r["scores"]]
        # No affinity-concordance label is assigned to an unmeasured substituted pair.
        sensitivity = None
        if shifts:
            swapped = [r["scores"]["substituted_unmeasured"] for r in rows]
            native_scores = [r["scores"]["native"] for r in rows]
            # Comparing two score orderings is label-free; neither ordering is
            # called correct for the unmeasured antigen substitution.
            order = concordance(native_scores, swapped, direction="higher")
            sensitivity = {"mean_absolute_score_shift": sum(abs(x) for x in shifts) / len(shifts),
                           "variant_dependent_shift_range": max(shifts) - min(shifts),
                           "strict_pair_order_reversals": order["discordant"],
                           "comparable_native_score_pairs": order["comparable_pairs"],
                           "correctness": "unmeasured"}
        result = {"group_id": g["group_id"], "target": g["target"], "fold": g["fold"],
                  "native_ranking": native, "antigen_sensitivity": sensitivity, "variants": rows}
        results.append(result)
        per_target[g["target"]].append(result)
    summaries = {}
    for target in sorted({g["target"] for g in selected}):
        rs = per_target[target]
        defined = [r["native_ranking"]["concordance"] for r in rs
                   if r["native_ranking"]["concordance"] is not None]
        sensitivities = [r["antigen_sensitivity"] for r in rs if r["antigen_sensitivity"]]
        summaries[target] = {
            "selected_groups": sum(g["target"] == target for g in selected), "scored_groups": len(rs),
            "excluded_groups": sum(r["target"] == target for r in excluded),
            "mean_group_concordance": sum(defined) / len(defined) if defined else None,
            "comparable_pairs": sum(r["native_ranking"]["comparable_pairs"] for r in rs),
            "sensitivity_only": {
                "mean_group_absolute_score_shift": sum(s["mean_absolute_score_shift"] for s in sensitivities) / len(sensitivities) if sensitivities else None,
                "groups_with_pair_order_reversal": sum(s["strict_pair_order_reversals"] > 0 for s in sensitivities),
                "strict_pair_order_reversals": sum(s["strict_pair_order_reversals"] for s in sensitivities),
                "comparable_native_score_pairs": sum(s["comparable_native_score_pairs"] for s in sensitivities),
                "correctness": "unmeasured",
            },
        }
    return {"schema": "hcdr3-contrast-scores/1", "manifest_sha256": manifest["manifest_sha256"],
            "score": ("mean log probability over 20 canonical residues with the complete HCDR3 masked; temperature 1; no sampling"
                      if scoring_mode == "full_span" else
                      "mean leave-one-residue-out HCDR3 pseudo-log-likelihood over 20 canonical residues; other candidate residues visible; temperature 1; no sampling"),
            "scoring_mode": scoring_mode,
            "fold": fold, "batch_size": batch_size,
            "encoded_inputs_sha256": digest(sorted(token_hashes)),
            "claims": {"historical_holdout_established": False, "antigen_correctness_established": False,
                       "interpretation": "exploratory native variant ranking and unmeasured antigen-input sensitivity",
                       "length": "per-residue normalization; candidate loop lengths remain observed inputs",
                       "exposure": "consult the separately linked exposure audit; a prospective fold does not certify historical holdout"},
            "antigen_truncated_contexts": dict(truncated), "per_target": summaries,
            "excluded": excluded, "groups": results}

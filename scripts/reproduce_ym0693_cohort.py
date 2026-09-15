#!/usr/bin/env python
"""
Reproduce the Open AlphaSeq YM_0693 paired cross-antigen cohort from the release.

Step 1 of the agreed order of work in ``docs/research-followup/MEMO.md`` §4e: verify the
mutation boundaries, the antigen construct sequences, the paired measurements, every
exclusion and its count, and the SIGN of the differential -- independently of the research
run that first proposed them.

The cohort is every same-length variant of the PP489 parental scFv whose only differences
from the parental fall inside the HCDR3 window, measured against both the human and the
mouse TIGIT construct.

Nothing here is model-specific. The output manifest is the input contract for the paired
evaluator, and carries the hashes and counts a later run needs to prove it used the same
data.

Usage
-----
    python scripts/reproduce_ym0693_cohort.py \
        --parquet data/raw/open_alphaseq/YM_0693.parquet \
        --out docs/research-followup/cohort/ym0693-manifest.json

Download (3.1 MB, revision pinned):
    curl -sSL -o data/raw/open_alphaseq/YM_0693.parquet \
      https://huggingface.co/datasets/aalphabio/open-alphaseq/resolve/3fb6b28b3ee758a7dfeaca5f14949dc505e90fda/data/YM_0693/data.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smallAntibodyGen.evaluation.contrasts import digest, file_sha256, save_json  # noqa: E402

SCHEMA = "ym0693-paired-cohort/1"

# Release identity. A different revision or hash is a different cohort and must not be
# silently accepted -- every count below is only meaningful against these exact bytes.
HF_REPO = "aalphabio/open-alphaseq"
HF_REVISION = "3fb6b28b3ee758a7dfeaca5f14949dc505e90fda"
EXPECTED_SHA256 = "6b82f9bb39c835ec160ca192ef0e1ba165a0d6651c520791081e800c485c907b"

HUMAN = "TIGIT_22-137_POI-AGA2"
MOUSE = "TIGIT_Mouse"

# 0-indexed, inclusive, in the 247-aa parental. Asserted against the parental subsequence
# below rather than trusted, and reported alongside the positions that actually vary.
HCDR3_LO, HCDR3_HI = 96, 113
HCDR3_PARENTAL = "ARSTYYYDSSGYDYYFDP"

# Counts the research run derived. Reproduced, not assumed: a mismatch is reported as a
# discrepancy rather than raised, so the manifest still records what this data does say.
CLAIMED = {"cohort": 935, "usable": 588, "z_gt_1p96": 273, "z_gt_3": 158}


def parental_sequence(frame: pd.DataFrame) -> str:
    """The wild-type scFv, taken from the wt_ replicate strains."""
    wt = frame[frame["mata_description"].str.startswith("wt_")]
    unique = wt["mata_sequence"].dropna().unique()
    if len(unique) != 1:
        raise ValueError(f"expected one parental sequence across wt_ strains, found {len(unique)}")
    return str(unique[0])


def varying_positions(frame: pd.DataFrame, parental: str) -> dict[str, Any]:
    """Where the same-length library actually differs from the parental.

    Reported because the HCDR3 window is a FILTER over a wider library, not a description
    of it. Treating the window as the library's extent would misdescribe the cohort.
    """
    same_length = frame[frame["mata_sequence"].str.len() == len(parental)]
    seqs = same_length["mata_sequence"].tolist()
    arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), len(parental))
    par = np.frombuffer(parental.encode(), dtype="S1")
    counts = (arr != par).sum(axis=0)
    varying = np.nonzero(counts)[0]
    return {
        "same_length_binders": int(len(same_length)),
        "first_varying_position": int(varying.min()),
        "last_varying_position": int(varying.max()),
        "n_varying_positions": int(len(varying)),
        "window_is_a_filter_not_the_library": bool(varying.min() < HCDR3_LO),
    }


def select_cohort(frame: pd.DataFrame, parental: str) -> pd.DataFrame:
    """Same-length variants differing from the parental only inside the HCDR3 window."""
    same_length = frame[frame["mata_sequence"].str.len() == len(parental)]

    def only_in_window(seq: str) -> bool:
        return all(a == b or HCDR3_LO <= i <= HCDR3_HI for i, (a, b) in enumerate(zip(seq, parental)))

    keep = same_length["mata_sequence"].map(only_in_window) & (same_length["mata_sequence"] != parental)
    return same_length[keep]


def paired_table(rows: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    used = rows[rows["mata_description"].isin(cohort["mata_description"])]
    pivot = lambda column: used.pivot_table(  # noqa: E731
        index="mata_description", columns="matalpha_description", values=column, aggfunc="first"
    )
    return {
        "affinity": pivot("alphaseq_affinity"),
        "lower": pivot("affinity_lower_bound"),
        "upper": pivot("affinity_upper_bound"),
        "above_background": pivot("above_background"),
        "replicates": pivot("sufficient_replicate_observations"),
    }


def half_width(lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Half the interval width, taken numerically.

    The release names these bounds in AFFINITY space, so ``affinity_lower_bound`` holds the
    NUMERICALLY LARGER log10 Kd (the weaker end). Subtracting them in the order the names
    suggest yields a negative width. Verified on every measured row in this release.
    """
    return (np.maximum(lower, upper) - np.minimum(lower, upper)) / 2.0


def build(parquet: Path) -> dict[str, Any]:
    observed_sha = file_sha256(parquet)
    frame = pd.read_parquet(parquet)
    with_sequence = frame.dropna(subset=["mata_sequence"])
    binders = with_sequence[["mata_description", "mata_sequence"]].drop_duplicates("mata_description")

    parental = parental_sequence(binders)
    window_observed = parental[HCDR3_LO : HCDR3_HI + 1]
    if window_observed != HCDR3_PARENTAL:
        raise ValueError(
            f"HCDR3 window {HCDR3_LO}..{HCDR3_HI} holds {window_observed!r}, expected {HCDR3_PARENTAL!r}"
        )

    antigens = (
        frame.dropna(subset=["matalpha_sequence"])[["matalpha_description", "matalpha_sequence"]]
        .drop_duplicates()
        .set_index("matalpha_description")["matalpha_sequence"]
        .to_dict()
    )
    for name in (HUMAN, MOUSE):
        if name not in antigens:
            raise ValueError(f"antigen {name!r} not present in the release")

    cohort = select_cohort(binders, parental)
    tables = paired_table(frame, cohort)
    affinity = tables["affinity"]

    measured_both = affinity[[HUMAN, MOUSE]].notna().all(axis=1)
    above_bg = tables["above_background"][HUMAN].fillna(False) & tables["above_background"][MOUSE].fillna(False)
    enough_reps = tables["replicates"][HUMAN].fillna(False) & tables["replicates"][MOUSE].fillna(False)
    usable = measured_both & above_bg & enough_reps

    delta = affinity.loc[usable, MOUSE] - affinity.loc[usable, HUMAN]
    standard_error = np.sqrt(
        (half_width(tables["lower"][HUMAN], tables["upper"][HUMAN]) / 1.96) ** 2
        + (half_width(tables["lower"][MOUSE], tables["upper"][MOUSE]) / 1.96) ** 2
    )[usable]
    # The target-wide shift is removed before asking whether a variant differs: nearly every
    # variant binds human more tightly than mouse, so the raw differential is dominated by an
    # offset that carries no variant-specific information.
    z = (delta - delta.mean()) / standard_error

    wt_rows = frame[frame["mata_description"].str.startswith("wt_")]
    wt_pivot = wt_rows.pivot_table(
        index="mata_description", columns="matalpha_description", values="alphaseq_affinity", aggfunc="first"
    )
    wt_delta = (wt_pivot[MOUSE] - wt_pivot[HUMAN]).dropna()

    counts = {
        "cohort": int(len(cohort)),
        "measured_on_both": int(measured_both.sum()),
        "and_above_background": int((measured_both & above_bg).sum()),
        "usable": int(usable.sum()),
        "z_gt_1p96": int((z.abs() > 1.96).sum()),
        "z_gt_3": int((z.abs() > 3.0).sum()),
    }
    discrepancies = {k: {"claimed": v, "reproduced": counts[k]} for k, v in CLAIMED.items() if counts[k] != v}

    variants = [
        {
            "variant": str(name),
            "hcdr3": str(binders.set_index("mata_description").loc[name, "mata_sequence"][HCDR3_LO : HCDR3_HI + 1]),
            "affinity_human": float(affinity.loc[name, HUMAN]),
            "affinity_mouse": float(affinity.loc[name, MOUSE]),
            "delta_mouse_minus_human": float(delta.loc[name]),
            "paired_standard_error": float(standard_error.loc[name]),
            "z_shift_removed": float(z.loc[name]),
            "exploratory_subset": bool(abs(z.loc[name]) > 1.96),
        }
        for name in delta.index
    ]

    return {
        "schema": SCHEMA,
        "source": {
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "path": "data/YM_0693/data.parquet",
            "sha256": observed_sha,
            "sha256_matches_expected": observed_sha == EXPECTED_SHA256,
            "license": "not stated in the dataset card API response; confirm before redistribution",
            "rows": int(len(frame)),
        },
        "parental": {
            "sequence": parental,
            "length": len(parental),
            "replicate_strains": int(wt_rows["mata_description"].nunique()),
        },
        "hcdr3_window": {
            "low": HCDR3_LO,
            "high": HCDR3_HI,
            "length": HCDR3_HI - HCDR3_LO + 1,
            "parental_subsequence": window_observed,
            "library_variation": varying_positions(binders, parental),
        },
        "antigens": {
            "human": {"name": HUMAN, "length": len(antigens[HUMAN]), "sequence": antigens[HUMAN]},
            "mouse": {"name": MOUSE, "length": len(antigens[MOUSE]), "sequence": antigens[MOUSE]},
            "construct_boundary_unresolved": (
                "The human construct is named POI-AGA2 and the display fusion boundary is an owner "
                "decision (MEMO R3). Sequences are recorded exactly as released."
            ),
        },
        "measurement": {
            "column": "alphaseq_affinity",
            "units": "log10 Kd (nM)",
            "direction": "lower is tighter",
            "direction_evidence": {
                "negative_control_median": float(
                    frame.loc[
                        frame["mata_description"].str.startswith("ANeg")
                        & (frame["matalpha_description"] == HUMAN),
                        "alphaseq_affinity",
                    ].median()
                ),
                "parental_human_median": float(wt_pivot[HUMAN].median()),
                "parental_mouse_median": float(wt_pivot[MOUSE].median()),
                "note": (
                    "Negative controls sit far above the parental, and this anti-human-TIGIT parental "
                    "binds human more tightly than mouse. Both agree with lower-is-tighter, so the "
                    "sign is established from the data rather than assumed from the column name."
                ),
            },
            "interval_columns_are_named_in_affinity_space": (
                "affinity_lower_bound holds the numerically LARGER log10 Kd on every measured row in "
                "this release. Take interval widths numerically, never by name order."
            ),
        },
        "exclusions": [
            {"step": "measured on both antigens", "remaining": counts["measured_on_both"]},
            {"step": "above_background on both", "remaining": counts["and_above_background"]},
            {"step": "sufficient_replicate_observations on both", "remaining": counts["usable"]},
        ],
        "counts": counts,
        "claimed_counts": CLAIMED,
        "discrepancies": discrepancies,
        "differential": {
            "definition": "delta(v) = affinity(v, mouse) - affinity(v, human)",
            "interpretation": "positive means weaker against mouse, i.e. the variant prefers human",
            "n": int(len(delta)),
            "mean_target_wide_shift": float(delta.mean()),
            "sd_across_variants": float(delta.std()),
            "median_paired_standard_error": float(standard_error.median()),
            "n_positive": int((delta > 0).sum()),
            "n_negative": int((delta < 0).sum()),
        },
        "assay_noise_floor": {
            "source": "the parental replicate strains, paired across the two antigens",
            "n_replicates": int(len(wt_delta)),
            "mean_delta": float(wt_delta.mean()),
            "sd_delta": float(wt_delta.std()),
            "ratio_variant_sd_to_replicate_sd": float(delta.std() / wt_delta.std()),
            "claim_limit": (
                "A noise floor on the differential for ONE repeated construct. It is not a ceiling on "
                "achievable correlation and must not be used as one. The release contains sparse "
                "exact-sequence repeats under distinct candidate IDs; those require a separate "
                "variant-level audit and do not establish a representative library-wide error model."
            ),
        },
        "primary_analysis_set": {
            "n": int(len(delta)),
            "description": "all usable variants; this is the primary set for agreement with measurement",
        },
        "exploratory_subset": {
            "n": counts["z_gt_1p96"],
            "rule": "|z| > 1.96 after removing the target-wide shift",
            "claim_limit": (
                "Selected on the outcome variable, so it inflates apparent agreement. Report second, "
                "never as the headline."
            ),
        },
        "variants": variants,
        "variants_digest": digest(variants),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, default=Path("data/raw/open_alphaseq/YM_0693.parquet"))
    parser.add_argument("--out", type=Path, default=Path("docs/research-followup/cohort/ym0693-manifest.json"))
    args = parser.parse_args()

    if not args.parquet.exists():
        parser.error(f"{args.parquet} not found; see the download command in this file's docstring")

    payload = build(args.parquet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, payload)

    counts, differential, noise = payload["counts"], payload["differential"], payload["assay_noise_floor"]
    print(f"wrote {args.out}")
    print(f"  source sha256 matches expected : {payload['source']['sha256_matches_expected']}")
    print(f"  cohort / usable                : {counts['cohort']} / {counts['usable']}")
    print(f"  |z|>1.96 / |z|>3 (exploratory)  : {counts['z_gt_1p96']} / {counts['z_gt_3']}")
    print(f"  target-wide shift              : {differential['mean_target_wide_shift']:+.4f} log10 units")
    print(f"  variant SD / replicate SD      : {differential['sd_across_variants']:.4f} / {noise['sd_delta']:.4f}"
          f"  (ratio {noise['ratio_variant_sd_to_replicate_sd']:.2f})")
    if payload["discrepancies"]:
        print("  DISCREPANCIES vs the claimed counts:")
        for key, value in payload["discrepancies"].items():
            print(f"    {key}: claimed {value['claimed']}, reproduced {value['reproduced']}")
    else:
        print("  all claimed counts reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

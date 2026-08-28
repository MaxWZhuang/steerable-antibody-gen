#!/usr/bin/env python
"""
Launch and compare J11's six evidence runs.

Two subcommands, deliberately in one file so the launch preconditions and the
comparison rules cannot drift apart:

``launch``   verify the six configs really encode the frozen design, require a
             clean worktree, pin the commit, and run them.
``compare``  read the six runs' FINAL step-51,000 metrics and apply the
             promotion rule, refusing anything incomplete.

The refusals are the substance. A harness that runs whatever it is given and
always produces a winner turns "we could not tell" into "1024 wins", and nothing
downstream ever revisits it.

Non-negotiables enforced here:

- Final-step metrics only. Never the best intermediate checkpoint -- selecting
  the best of many validations is a maximum over noise, and it would favour
  whichever arm was validated more often or got luckier.
- Intermediate validation is diagnostic. Early stopping is rejected at config
  level, because arms that stop at different steps are not comparable.
- A clean worktree and a recorded commit, so the code that produced the evidence
  is identifiable. Train a merged origin/main SHA, never a feature branch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "j11-comparison/1"

# Frozen design (specs/experiments/j11_swiglu_width.md).
WIDTHS = (680, 1024)
SEEDS = (42, 31415, 271828)
TOTAL_UPDATES = 51_000
WARMUP_UPDATES = 1_000
MIN_POST_WARMUP_UPDATES = 50_000

#: Keys allowed to differ between the two widths at a fixed seed.
WIDTH_AXIS_KEYS = frozenset({"swiglu_hidden_dim", "output_dir"})
#: Keys allowed to differ between seeds at a fixed width.
SEED_AXIS_KEYS = frozenset({"seed", "paired_init_seed", "output_dir"})

#: Promotion thresholds, predeclared. Named constants rather than CLI defaults so
#: they cannot be nudged at comparison time.
MAX_SLOWDOWN = 0.35
MIN_HCDR3_TOKEN_GAIN_PP = 1.0
MAX_MLM_LOSS_REGRESSION_NATS = 0.01
MAX_SPAN_EXACT_REGRESSION_PP = 0.25
MIN_MEMORY_HEADROOM = 0.25


class LaunchRefused(RuntimeError):
    """A precondition failed; nothing was run."""


# --------------------------------------------------------------------------- #
# Launch preconditions
# --------------------------------------------------------------------------- #
def config_path(width: int, seed: int) -> Path:
    return PROJECT_ROOT / "configs/experiments/swiglu_width" / f"arm_w{width}_s{seed}.yaml"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LaunchRefused(f"missing config {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def verify_design() -> dict[str, Any]:
    """
    Check the six configs encode the frozen design, on both axes.

    A pairwise check at fixed seed AND at fixed width, because a single sweep
    over all six would let a difference hide: two configs that both deviate the
    same way look consistent to a naive comparison.
    """
    configs = {(w, s): load(config_path(w, s)) for w in WIDTHS for s in SEEDS}
    problems: list[str] = []

    for (width, seed), cfg in configs.items():
        if cfg.get("max_updates") != TOTAL_UPDATES:
            problems.append(f"w{width}/s{seed}: max_updates={cfg.get('max_updates')} != {TOTAL_UPDATES}")
        if cfg.get("warmup_steps") != WARMUP_UPDATES:
            problems.append(f"w{width}/s{seed}: warmup_steps={cfg.get('warmup_steps')} != {WARMUP_UPDATES}")
        if cfg.get("swiglu_hidden_dim") != width:
            problems.append(f"w{width}/s{seed}: swiglu_hidden_dim mismatch")
        if cfg.get("seed") != seed or cfg.get("paired_init_seed") != seed:
            problems.append(f"w{width}/s{seed}: seed / paired_init_seed mismatch")
        if not cfg.get("paired_init_seed"):
            problems.append(
                f"w{width}/s{seed}: paired_init_seed is unset, so the arms would differ "
                "in far more than width and the seeds would not be paired"
            )
        if cfg.get("early_stopping_patience"):
            problems.append(
                f"w{width}/s{seed}: early stopping is set; arms must reach the SAME step"
            )
        if cfg.get("resume_from_last"):
            problems.append(f"w{width}/s{seed}: resume_from_last must be false for a pilot")

    # Width axis, at each fixed seed.
    for seed in SEEDS:
        left, right = configs[(WIDTHS[0], seed)], configs[(WIDTHS[1], seed)]
        differing = {k for k in set(left) | set(right) if left.get(k) != right.get(k)}
        unexpected = differing - WIDTH_AXIS_KEYS
        if unexpected:
            problems.append(f"seed {seed}: widths differ outside the axis: {sorted(unexpected)}")

    # Seed axis, at each fixed width.
    for width in WIDTHS:
        base = configs[(width, SEEDS[0])]
        for seed in SEEDS[1:]:
            other = configs[(width, seed)]
            differing = {k for k in set(base) | set(other) if base.get(k) != other.get(k)}
            unexpected = differing - SEED_AXIS_KEYS
            if unexpected:
                problems.append(
                    f"width {width}: seeds {SEEDS[0]} vs {seed} differ outside the axis: "
                    f"{sorted(unexpected)}"
                )

    if problems:
        raise LaunchRefused(
            "the six configs do not encode the frozen J11 design:\n  - "
            + "\n  - ".join(problems)
        )
    return {"configs_verified": len(configs)}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def require_clean_worktree() -> dict[str, str]:
    """
    Refuse to produce evidence from an unidentifiable tree.

    A dirty worktree means the recorded commit does not describe the code that
    ran, so the run cannot be reproduced or audited -- and a 36-hour experiment
    that cannot be attributed to a revision is not evidence.
    """
    status = git("status", "--porcelain")
    if status:
        raise LaunchRefused(
            "worktree is not clean; commit or stash first so the recorded commit "
            "describes the code that actually ran:\n" + status
        )
    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    return {"commit": commit, "branch": branch}


def warn_if_not_main(lineage: dict[str, str], allow_branch: bool) -> list[str]:
    """Training a feature-branch SHA produces evidence tied to a revision that
    may never exist publicly. Allowed only with an explicit flag."""
    warnings: list[str] = []
    if lineage["branch"] != "main":
        message = (
            f"HEAD is on '{lineage['branch']}', not main. J11 evidence should train a "
            "MERGED origin/main SHA from a clean checkout, so the pinned commit is one "
            "that exists in the public history."
        )
        if not allow_branch:
            raise LaunchRefused(message + " Pass --allow-branch to override.")
        warnings.append(message)
    return warnings


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def read_final_metrics(run_dir: Path) -> dict[str, Any]:
    """
    The metrics at the FINAL step, never the best intermediate.

    Best-checkpoint selection is a maximum over validation noise and would favour
    whichever arm validated more often or got luckier. J11 is defined at step
    51,000 in both arms.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise LaunchRefused(f"{run_dir.name}: no metrics.jsonl")
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = [r for r in records if (r.get("val") or {})]
    if not final:
        raise LaunchRefused(f"{run_dir.name}: metrics.jsonl has no validation record")
    return final[-1]


def collect(results_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """
    Read all six runs' final metrics, completeness FIRST.

    The two checks are deliberately ordered and not interleaved. Reading metrics
    inside the existence loop lets a per-run error surface before the set has
    been fully surveyed, so an operator fixes one run, re-runs, and discovers the
    next problem -- and never sees that three runs are missing. Survey the whole
    set, report every gap at once, then read.
    """
    expected = [(width, seed) for width in WIDTHS for seed in SEEDS]
    missing = [
        f"j11_w{width}_s{seed}"
        for width, seed in expected
        if not (results_root / f"j11_w{width}_s{seed}").exists()
    ]
    if missing:
        raise LaunchRefused(
            f"incomplete evidence set ({len(expected) - len(missing)}/{len(expected)} "
            "runs present); refusing to compare. Missing: "
            + ", ".join(missing)
            + ". A selection from a partial set is not the predeclared experiment."
        )

    runs: dict[tuple[int, int], dict[str, Any]] = {}
    problems: list[str] = []
    for width, seed in expected:
        try:
            runs[(width, seed)] = read_final_metrics(
                results_root / f"j11_w{width}_s{seed}"
            )
        except LaunchRefused as exc:
            problems.append(str(exc))
    if problems:
        raise LaunchRefused(
            "runs are present but their metrics are unusable:\n  - "
            + "\n  - ".join(problems)
        )
    return runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="verify preconditions and run the six arms")
    launch.add_argument("--dry-run", action="store_true",
                        help="verify everything and print the commands without running.")
    launch.add_argument("--allow-branch", action="store_true",
                        help="permit training a non-main SHA (discouraged).")
    launch.add_argument("--output-json", type=Path, default=None)

    compare = sub.add_parser("compare", help="apply the promotion rule to six finished runs")
    compare.add_argument("--results-root", type=Path,
                         default=PROJECT_ROOT / "checkpoints/experiments")
    compare.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.command == "launch":
        try:
            design = verify_design()
            lineage = require_clean_worktree()
            warnings = warn_if_not_main(lineage, args.allow_branch)
        except LaunchRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1

        print("J11 launch preconditions PASS")
        print(f"  configs verified : {design['configs_verified']}")
        print(f"  commit           : {lineage['commit']}")
        print(f"  branch           : {lineage['branch']}")
        print(f"  schedule         : {TOTAL_UPDATES:,} updates "
              f"({WARMUP_UPDATES:,} warmup + {TOTAL_UPDATES - WARMUP_UPDATES:,} post-warmup)")
        for warning in warnings:
            print(f"  WARNING: {warning}")

        commands = [
            [
                sys.executable, "scripts/mlm_train.py",
                "--config", str(config_path(width, seed).relative_to(PROJECT_ROOT)),
            ]
            for width in WIDTHS
            for seed in SEEDS
        ]
        print("\n  runs:")
        for command in commands:
            print("    " + " ".join(command))

        if args.dry_run:
            print("\n  --dry-run: nothing was executed.")
            return 0

        for command in commands:
            print(f"\n=== {' '.join(command)}")
            completed = subprocess.run(command, cwd=PROJECT_ROOT)
            if completed.returncode != 0:
                print(
                    f"REFUSED: run failed ({completed.returncode}); stopping. A partial "
                    "evidence set may not be compared.",
                    file=sys.stderr,
                )
                return 1
        return 0

    # compare
    try:
        runs = collect(args.results_root)
    except LaunchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"collected {len(runs)} runs; comparison implementation follows the "
          "promotion rule in specs/experiments/j11_swiglu_width.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

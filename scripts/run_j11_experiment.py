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
import hashlib
import json
import math
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

#: The shape criterion 1 governs. A stage-1 probe shows headroom the dual-stream
#: shape does not have, and on this box the absence of an OOM is not evidence a
#: config fits (the driver spills to system RAM instead of failing), so a probe
#: is accepted only when it says it measured THIS shape without spilling.
DUAL_STREAM_SHAPE = {"max_length": 288, "antigen_max_length": 1024, "batch_size": 16}

#: ...and WHICH MODEL, in WHAT numeric regime. Shape alone is insufficient: a
#: 288/1024 batch-16 probe of the single-stream model, of the legacy block, or
#: without AMP each measures a different thing and would answer criterion 1 with
#: a number that does not describe the arms under test.
DUAL_STREAM_PROBE_MODEL = {
    # `dual` is `gpu_memory_probe.py`'s own vocabulary (--model-kind dual), not a
    # name invented here. One vocabulary, owned by the writer: a reader that
    # renames its inputs is a reader nobody can produce a valid file for.
    "model_kind": "dual",
    "ffn_type": "swiglu",
    "norm_type": "rmsnorm",
    "position_encoding": "rope",
    "use_amp": True,
}
#: Every descriptor a probe row must carry AND match before it is read at all.
DUAL_STREAM_PROBE_REQUIREMENTS = {**DUAL_STREAM_SHAPE, **DUAL_STREAM_PROBE_MODEL}
#: Columns that are RESULTS of the probe, so they may differ between the arms.
#: Anything else differing means the probe moved a second axis (Rule 3, applied
#: to the measurement rather than to the training run).
PROBE_MEASURED_KEYS = frozenset({
    "peak_reserved_mib", "peak_allocated_mib", "parameters",
    "fits_in_device_memory", "loss_finite", "ok", "warning", "seed",
})
#: A shared fact about the card, not a per-arm result: headroom is a fraction of
#: a specific device, so two rows measured against different totals are not one
#: comparison.
PROBE_SHARED_KEYS = frozenset({"device_total_mib"})

#: Three verdicts, not two. `not_auditable` is the honest state of a criterion
#: whose measurement was never retained; collapsing it into `pass` would let an
#: unmeasured clause pay for a measured failure, and collapsing it into `fail`
#: would report evidence of a problem where there is only absence of evidence.
#: Promotion requires every criterion to `pass`, so it blocks promotion either
#: way -- but the report has to say which of the two happened.
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NOT_AUDITABLE = "not_auditable"

#: What a J11 result does NOT license, carried in the report itself because a
#: later reader meets the selected width without the context that produced it.
CLAIM_LIMIT = (
    "Measured at the frozen 51,000-update schedule, a fraction of one training "
    "epoch over the stage-1 corpus. This does NOT establish asymptotic "
    "equivalence after full training: it is a practical negative on paying for "
    "extra width at this budget, not a scientific negative on capacity."
)

#: What the experiment chose, and what it did not ask.
SCOPE = (
    "Selects the canonical SwiGLU width for the v5 lineage. It does not measure "
    "depth, d_model, the block recipe (adopted by owner decision), or any "
    "antigen-conditioning behaviour."
)


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


# --------------------------------------------------------------------------- #
# Evidence binding
# --------------------------------------------------------------------------- #
def canonical_sha256(payload: Any) -> str:
    """
    Hash a parsed JSON value by its canonical form.

    Sorting keys and fixing separators means the hash tracks the CONTENT, not
    the whitespace a writer happened to emit, so a report can be checked against
    a re-serialized record.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path | None) -> str | None:
    """
    Name a path without leaking where this checkout lives.

    `str(path)` embeds the absolute location and therefore the local username,
    which makes the report differ between two machines comparing identical
    evidence -- and leaks a home directory into a tracked artifact. Inside the
    repository a path becomes its repo-relative POSIX form; outside it, only the
    file name survives, because nothing outside the checkout can be named
    portably at all.
    """
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def read_provenance(run_dir: Path) -> dict[str, Any]:
    """
    Read the SHA the launcher wrote next to the weights.

    Raises:
        LaunchRefused: the file is absent. Weights whose commit is unknown are
            unattributable evidence -- the launcher writes this file beside the
            checkpoints precisely so they cannot be separated.
    """
    path = run_dir / "j11_provenance.json"
    if not path.exists():
        raise LaunchRefused(
            f"{run_dir.name}: no j11_provenance.json, so the commit that produced "
            "these weights is unknown. A comparison of unattributable runs is not "
            "the predeclared experiment."
        )
    return load(path)


def _valid_commit(value: Any) -> bool:
    """A 40-character hex SHA, and nothing else."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def bind_runs(results_root: Path, runs: dict[tuple[int, int], dict[str, Any]]) -> list[dict]:
    """
    Bind each run to its fingerprint, training commit, and final record.

    Every check here fails CLOSED. An earlier version accepted a missing
    fingerprint as `null`, six empty commit strings as "one shared commit" (they
    are all equal, so an equality test alone passes them), and never compared the
    provenance's width/seed/schedule against the run it sat beside. Each of those
    lets unattributable or mislabelled evidence reach a verdict while the report
    still looks fully bound -- the most expensive shape of wrong, because the
    artifact advertises the very property it lost.

    Raises:
        LaunchRefused: a fingerprint or provenance file is missing; a commit is
            absent, empty, or not a SHA; the provenance disagrees with its own
            directory or with the frozen schedule; the fingerprint's recorded
            commit disagrees with the provenance's; the run was trained from a
            dirty worktree; or the six runs do not share one commit.
    """
    bound: list[dict[str, Any]] = []
    commits: dict[str, list[str]] = {}
    for width in WIDTHS:
        for seed in SEEDS:
            run_dir = results_root / f"j11_w{width}_s{seed}"
            provenance = read_provenance(run_dir)

            fingerprint_path = run_dir / "run_fingerprint.json"
            if not fingerprint_path.exists():
                raise LaunchRefused(
                    f"{run_dir.name}: no run_fingerprint.json. A null hash in the "
                    "report is not a binding, it is a hole shaped like one."
                )
            fingerprint = load(fingerprint_path)

            commit = provenance.get("commit")
            if not _valid_commit(commit):
                raise LaunchRefused(
                    f"{run_dir.name}: j11_provenance.json has no usable commit "
                    f"({commit!r}). Six empty strings are all equal and would pass "
                    "as one shared commit; unattributable evidence stops here."
                )

            for field, expected in (("width", width), ("seed", seed)):
                if provenance.get(field) != expected:
                    raise LaunchRefused(
                        f"{run_dir.name}: provenance {field}="
                        f"{provenance.get(field)!r} disagrees with the run "
                        f"directory ({expected}). One of them is wrong and this "
                        "cannot tell which, so the arm may be mislabelled."
                    )

            for field, expected in (
                ("total_updates", TOTAL_UPDATES),
                ("warmup_updates", WARMUP_UPDATES),
            ):
                if provenance.get(field) != expected:
                    raise LaunchRefused(
                        f"{run_dir.name}: provenance {field}="
                        f"{provenance.get(field)!r}, not the frozen "
                        f"{expected:,}. A run that stopped at a different step is "
                        "not part of this experiment."
                    )

            recorded = (fingerprint.get("manifests", {}).get("source", {}) or {}).get("commit")
            if recorded != commit:
                raise LaunchRefused(
                    f"{run_dir.name}: the run fingerprint and j11_provenance.json "
                    f"disagree about the commit ({recorded!r} vs {commit!r}). They "
                    "are written independently, so disagreement means one of them "
                    "was copied from another run."
                )
            if fingerprint.get("worktree_dirty"):
                raise LaunchRefused(
                    f"{run_dir.name}: trained from a dirty worktree, so the "
                    f"recorded commit {commit[:7]} does not describe the code that "
                    "produced these weights."
                )

            commits.setdefault(commit, []).append(run_dir.name)
            bound.append(
                {
                    "width": width,
                    "seed": seed,
                    "run": run_dir.name,
                    "commit": commit,
                    "branch": provenance.get("branch"),
                    "run_fingerprint_sha256": file_sha256(fingerprint_path),
                    "run_hash": fingerprint.get("run_hash"),
                    "final_metrics_sha256": canonical_sha256(runs[(width, seed)]),
                }
            )

    if len(commits) > 1:
        detail = "; ".join(
            f"{commit}: {', '.join(sorted(names))}"
            for commit, names in sorted(commits.items())
        )
        raise LaunchRefused(
            "the six runs do not share one commit, so they are not one "
            f"experiment: {detail}"
        )
    return bound


# --------------------------------------------------------------------------- #
# The promotion rule, as arithmetic
# --------------------------------------------------------------------------- #
def _criterion(
    number: int,
    statement: str,
    verdict: str,
    *,
    reason: str = "",
    **fields: Any,
) -> dict[str, Any]:
    """One criterion's outcome, in the shape the report carries."""
    return {
        "number": number,
        "statement": statement,
        "verdict": verdict,
        "reason": reason,
        **fields,
    }


def paired_rows(runs: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    """
    One row per seed, pairing the two widths.

    The experimental unit is the paired seed, of which there are three -- not the
    ~772,000 validation residues, which are correlated within antibodies, within
    repeated biological families, and within a shared mask. Reporting all three
    deltas keeps the spread visible instead of hiding it behind a mean.
    """
    rows = []
    for seed in SEEDS:
        narrow = runs[(680, seed)]["val"]
        wide = runs[(1024, seed)]["val"]

        # ENFORCED, not merely quoted. The whole pairing argument is that both
        # widths saw the same validation mask at a seed; an earlier version
        # copied the 680 arm's denominators into the report without ever
        # comparing them to 1024, so an unpaired comparison would have produced a
        # confident verdict under a paired-looking table.
        for field in ("hcdr3_target_tokens", "hcdr3_valid_spans"):
            if narrow[field] != wide[field]:
                raise LaunchRefused(
                    f"seed {seed}: {field} differs between the arms "
                    f"({narrow[field]} at width 680 vs {wide[field]} at 1024), so "
                    "they did not see the same validation mask. The paired "
                    "comparison this experiment rests on does not hold."
                )

        rows.append(
            {
                "seed": seed,
                "hcdr3_token_acc_680": narrow["hcdr3_token_acc"],
                "hcdr3_token_acc_1024": wide["hcdr3_token_acc"],
                "hcdr3_token_acc_delta_pp": (
                    wide["hcdr3_token_acc"] - narrow["hcdr3_token_acc"]
                ) * 100.0,
                "mlm_loss_680": narrow["mlm_loss"],
                "mlm_loss_1024": wide["mlm_loss"],
                "mlm_loss_delta_nats": wide["mlm_loss"] - narrow["mlm_loss"],
                "hcdr3_span_exact_680": narrow["hcdr3_span_exact_match"],
                "hcdr3_span_exact_1024": wide["hcdr3_span_exact_match"],
                "hcdr3_span_exact_delta_pp": (
                    wide["hcdr3_span_exact_match"] - narrow["hcdr3_span_exact_match"]
                ) * 100.0,
                "hcdr3_target_tokens": narrow["hcdr3_target_tokens"],
                "hcdr3_valid_spans": narrow["hcdr3_valid_spans"],
                "favours_1024": wide["hcdr3_token_acc"] > narrow["hcdr3_token_acc"],
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def evaluate_memory(path: Path | None) -> dict[str, Any]:
    """
    Criterion 1: >=25% reserved headroom at the dual-stream shape, both arms.

    The measurement quoted in the spec was never written to an artifact, so with
    no probe file this is `not_auditable` rather than a pass on the prose. A
    probe at the wrong shape is also `not_auditable`: reading a stage-1 probe as
    if it answered this is how a memory claim silently becomes untrue.
    """
    statement = (
        f"Full {DUAL_STREAM_SHAPE['max_length']}/"
        f"{DUAL_STREAM_SHAPE['antigen_max_length']} dual-stream at batch "
        f"{DUAL_STREAM_SHAPE['batch_size']} retains >="
        f"{MIN_MEMORY_HEADROOM:.0%} reserved headroom with no driver spill"
    )
    if path is None or not path.exists():
        return _criterion(
            1, statement, VERDICT_NOT_AUDITABLE,
            reason=(
                "no dual-stream memory probe artifact; the figures in the "
                "protocol were measured but never retained, so nothing binds "
                "them to these runs"
            ),
            source=portable_path(path),
        )

    payload = load(path)
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    arms = {row.get("swiglu_hidden_dim"): row for row in rows}
    source = {"source": portable_path(path), "source_sha256": file_sha256(path)}

    def unauditable(reason: str) -> dict[str, Any]:
        return _criterion(1, statement, VERDICT_NOT_AUDITABLE, reason=reason, **source)

    missing = [width for width in WIDTHS if width not in arms]
    if missing:
        return unauditable(f"the probe has no row for width(s) {missing}")

    # 1. Each arm must SAY what it measured, and say the right thing. An absent
    #    descriptor is not an implicit match.
    mismatches = []
    for width in WIDTHS:
        for key, expected in DUAL_STREAM_PROBE_REQUIREMENTS.items():
            if key not in arms[width]:
                mismatches.append(f"width {width} does not record {key}")
            elif arms[width][key] != expected:
                mismatches.append(
                    f"width {width} has {key}={arms[width][key]!r}, expected {expected!r}"
                )
    if mismatches:
        return unauditable(
            "the probe did not measure the arms under test: " + "; ".join(mismatches)
        )

    # 2. The card is a shared fact, not a per-arm result.
    for key in sorted(PROBE_SHARED_KEYS):
        values = {arms[width].get(key) for width in WIDTHS}
        if len(values) != 1 or None in values:
            return unauditable(
                f"the arms report different {key} ({sorted(map(str, values))}); "
                "headroom is a fraction of one device, so these are not one "
                "comparison"
            )

    # 3. Rule 3 applied to the measurement: the rows may differ in the width under
    #    test and in what was measured, nothing else.
    off_axis = sorted(
        key
        for key in set(arms[680]) | set(arms[1024])
        if key not in DUAL_STREAM_PROBE_REQUIREMENTS
        and key not in PROBE_MEASURED_KEYS
        and key not in PROBE_SHARED_KEYS
        and key != "swiglu_hidden_dim"
        and arms[680].get(key) != arms[1024].get(key)
    )
    if off_axis:
        return unauditable(
            f"the two probe rows differ off-axis in {off_axis}, so any headroom "
            "gap is not attributable to width alone"
        )

    # 4. The wider arm is larger by construction; a probe that says otherwise
    #    measured something other than these two arms.
    params = {width: arms[width].get("parameters") for width in WIDTHS}
    if not all(isinstance(value, int) and value > 0 for value in params.values()):
        return unauditable(f"the probe does not record parameters ({params})")
    if params[1024] <= params[680]:
        return unauditable(
            f"width 1024 reports {params[1024]:,} parameters, not more than 680's "
            f"{params[680]:,}; the wider arm is larger by construction"
        )

    # Only now is the measurement itself read.
    headroom = {
        width: 1.0 - (arms[width]["peak_reserved_mib"] / arms[width]["device_total_mib"])
        for width in WIDTHS
    }
    spilled = [w for w in WIDTHS if not arms[w].get("fits_in_device_memory", False)]
    passed = not spilled and all(value >= MIN_MEMORY_HEADROOM for value in headroom.values())
    return _criterion(
        1, statement, VERDICT_PASS if passed else VERDICT_FAIL,
        reason="" if passed else (
            f"driver spill in {spilled}" if spilled
            else "reserved headroom below the floor: "
                 + ", ".join(f"{w}={headroom[w]:.1%}" for w in WIDTHS)
        ),
        measured={str(width): headroom[width] for width in WIDTHS},
        threshold=MIN_MEMORY_HEADROOM,
        parameters={str(width): params[width] for width in WIDTHS},
        device_total_mib=arms[680]["device_total_mib"],
        **source,
    )


def evaluate_step_time(path: Path | None) -> dict[str, Any]:
    """
    Criterion 2: width 1024 no more than 35% slower, from the pipeline preflight.
    """
    statement = f"Stage-1 median step time is <={MAX_SLOWDOWN:.0%} slower than 680"
    if path is None or not path.exists():
        return _criterion(
            2, statement, VERDICT_NOT_AUDITABLE,
            reason="no pipeline preflight artifact to read median step times from",
            source=portable_path(path),
        )

    payload = load(path)
    arms = {arm.get("swiglu_hidden_dim"): arm for arm in payload.get("arms", [])}
    missing = [width for width in WIDTHS if width not in arms]
    if missing:
        return _criterion(
            2, statement, VERDICT_NOT_AUDITABLE,
            reason=f"the preflight has no arm for width(s) {missing}",
            source=portable_path(path),
            source_sha256=file_sha256(path),
        )

    narrow = arms[680]["median_step_seconds"]
    wide = arms[1024]["median_step_seconds"]
    slowdown = (wide / narrow) - 1.0
    return _criterion(
        2, statement, VERDICT_PASS if slowdown <= MAX_SLOWDOWN else VERDICT_FAIL,
        reason="" if slowdown <= MAX_SLOWDOWN else (
            f"width 1024 is {slowdown:.1%} slower, past the {MAX_SLOWDOWN:.0%} ceiling"
        ),
        measured=slowdown,
        threshold=MAX_SLOWDOWN,
        median_step_seconds={"680": narrow, "1024": wide},
        source=portable_path(path),
        source_sha256=file_sha256(path),
    )


def _all_finite(payload: Any) -> bool:
    """True when every numeric leaf in a record is finite."""
    if isinstance(payload, bool):
        return True
    if isinstance(payload, (int, float)):
        return math.isfinite(payload)
    if isinstance(payload, dict):
        return all(_all_finite(value) for value in payload.values())
    if isinstance(payload, list):
        return all(_all_finite(value) for value in payload)
    return True


def evaluate_stability(runs: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    """
    Criterion 7: no NaNs, no unexplained AMP skips, no instability.

    Two claims, and only the first is checkable from what the runs retained.
    `UPDATE_COUNTER["amp_skips"]` is counted in-process but never written to
    `metrics.jsonl` or the checkpoint payload, and no J11 run log was kept -- so
    for runs that predate the counter being persisted the honest verdict is
    `not_auditable` WITH the NaN half reported, rather than one opaque shrug.
    """
    statement = "Neither arm shows NaNs, unexplained AMP skips, or instability"
    finite = all(_all_finite(record) for record in runs.values())
    skips = {
        f"w{width}_s{seed}": runs[(width, seed)].get("amp_skips")
        for width in WIDTHS
        for seed in SEEDS
    }
    unrecorded = sorted(name for name, value in skips.items() if value is None)

    if not finite:
        return _criterion(
            7, statement, VERDICT_FAIL,
            reason="a final metrics record contains a non-finite value",
            metrics_finite=False,
            amp_skips=skips,
        )
    if unrecorded:
        return _criterion(
            7, statement, VERDICT_NOT_AUDITABLE,
            reason=(
                "every retained metric is finite, but amp_skips was not persisted "
                f"for {len(unrecorded)} run(s) ({', '.join(unrecorded)}); the "
                "counter exists in the trainer and never reaches metrics.jsonl, "
                "the checkpoint, or any retained log"
            ),
            metrics_finite=True,
            amp_skips=skips,
        )
    offenders = sorted(name for name, value in skips.items() if value)
    return _criterion(
        7, statement, VERDICT_PASS if not offenders else VERDICT_FAIL,
        reason="" if not offenders else f"AMP skipped optimizer steps in {offenders}",
        metrics_finite=True,
        amp_skips=skips,
    )


def evaluate_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Criteria 3-6, all computed from the six final validation records."""
    token_gain = _mean([row["hcdr3_token_acc_delta_pp"] for row in rows])
    sweep = [row["seed"] for row in rows if row["favours_1024"]]
    loss_regression = _mean([row["mlm_loss_delta_nats"] for row in rows])
    span_regression = -_mean([row["hcdr3_span_exact_delta_pp"] for row in rows])

    return [
        _criterion(
            3,
            "Mean fixed-mask HCDR3 token recovery improves by "
            f">=+{MIN_HCDR3_TOKEN_GAIN_PP:.1f} absolute percentage points",
            VERDICT_PASS if token_gain >= MIN_HCDR3_TOKEN_GAIN_PP else VERDICT_FAIL,
            reason="" if token_gain >= MIN_HCDR3_TOKEN_GAIN_PP else (
                f"HCDR3 token recovery gain {token_gain:.3f} pp < required "
                f"{MIN_HCDR3_TOKEN_GAIN_PP:.1f} pp"
            ),
            measured=token_gain,
            threshold=MIN_HCDR3_TOKEN_GAIN_PP,
        ),
        _criterion(
            4,
            "All three paired seeds favour 1024 on HCDR3 token recovery",
            VERDICT_PASS if len(sweep) == len(SEEDS) else VERDICT_FAIL,
            reason="" if len(sweep) == len(SEEDS) else (
                f"only {len(sweep)}/{len(SEEDS)} paired seeds favour 1024"
            ),
            measured=len(sweep),
            threshold=len(SEEDS),
            seeds_favouring_1024=sweep,
        ),
        _criterion(
            5,
            "Mean validation MLM loss does not regress by more than "
            f"{MAX_MLM_LOSS_REGRESSION_NATS} nats",
            VERDICT_PASS
            if loss_regression <= MAX_MLM_LOSS_REGRESSION_NATS
            else VERDICT_FAIL,
            reason="" if loss_regression <= MAX_MLM_LOSS_REGRESSION_NATS else (
                f"MLM loss regressed by {loss_regression:.5f} nats, past the "
                f"{MAX_MLM_LOSS_REGRESSION_NATS} margin"
            ),
            measured=loss_regression,
            threshold=MAX_MLM_LOSS_REGRESSION_NATS,
        ),
        _criterion(
            6,
            "HCDR3 span-exact recovery does not regress by more than "
            f"{MAX_SPAN_EXACT_REGRESSION_PP} percentage points",
            VERDICT_PASS
            if span_regression <= MAX_SPAN_EXACT_REGRESSION_PP
            else VERDICT_FAIL,
            reason="" if span_regression <= MAX_SPAN_EXACT_REGRESSION_PP else (
                f"span-exact regressed by {span_regression:.3f} pp, past the "
                f"{MAX_SPAN_EXACT_REGRESSION_PP} pp margin"
            ),
            measured=span_regression,
            threshold=MAX_SPAN_EXACT_REGRESSION_PP,
            note=(
                "Reported for completeness: on ~"
                f"{rows[0]['hcdr3_valid_spans']:.0f} valid spans the unpaired "
                "binomial noise on this rate is wider than the margin itself, and "
                "the retained aggregates cannot support the paired analysis that "
                "would be appropriate. Treat this clause as under-instrumented."
            ),
        ),
    ]


def decide(criteria: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Apply the rule: 1024 is promoted only when EVERY criterion passes.

    680 is the predeclared tie and inconclusive fallback, so a `fail` and a
    `not_auditable` both select it -- but the reason distinguishes them, because
    "we measured this and it lost" and "we never measured this" are different
    facts about the experiment.
    """
    failed = [c for c in criteria if c["verdict"] == VERDICT_FAIL]
    unauditable = [c for c in criteria if c["verdict"] == VERDICT_NOT_AUDITABLE]
    promoted = not failed and not unauditable

    token_gain = _mean([row["hcdr3_token_acc_delta_pp"] for row in rows])
    sweep = sum(1 for row in rows if row["favours_1024"])

    if promoted:
        reason = (
            f"all {len(criteria)} criteria pass; mean HCDR3 token recovery gain "
            f"{token_gain:+.3f} pp at {sweep}/{len(rows)} paired seeds"
        )
    elif failed:
        reason = failed[0]["reason"] + f" (criterion {failed[0]['number']})"
    else:
        numbers = ", ".join(str(c["number"]) for c in unauditable)
        reason = (
            f"criteria {numbers} are not auditable and promotion requires every "
            "criterion; no measured clause failed"
        )

    if promoted:
        claim = (
            f"width 1024 earned promotion: {token_gain:+.3f} pp mean HCDR3 token "
            f"recovery across {sweep}/{len(rows)} paired seeds, clearing the "
            f"+{MIN_HCDR3_TOKEN_GAIN_PP:.1f} pp bar with no regression"
        )
    elif sweep == len(rows) and 0 < token_gain < MIN_HCDR3_TOKEN_GAIN_PP:
        claim = (
            f"width 1024 showed a consistent but practically insufficient gain "
            f"({token_gain:+.3f} pp mean HCDR3 token recovery, {sweep}/{len(rows)} "
            f"paired seeds, against a +{MIN_HCDR3_TOKEN_GAIN_PP:.1f} pp bar)"
        )
    else:
        claim = (
            f"width 1024 did not earn promotion: {token_gain:+.3f} pp mean HCDR3 "
            f"token recovery at {sweep}/{len(rows)} paired seeds"
        )

    return {
        "selected_width": 1024 if promoted else 680,
        "width_1024_promoted": promoted,
        "primary_reason": reason,
        "claim": claim,
    }


def compare(
    results_root: Path,
    preflight_path: Path | None,
    memory_probe_path: Path | None,
) -> dict[str, Any]:
    """
    Read six finished runs and apply the promotion rule.

    Args:
        results_root: directory holding the six ``j11_w{width}_s{seed}`` runs.
        preflight_path: pipeline-preflight artifact for criterion 2, or None.
        memory_probe_path: dual-stream memory probe for criterion 1, or None.

    Returns:
        The comparison report, ready to serialize. Deterministic: same evidence
        in, byte-identical report out, so a rerun that differs means the evidence
        moved rather than the comparator.

    Raises:
        LaunchRefused: the evidence set is incomplete, a run has no validation
            record, a run has no provenance, or the six runs span more than one
            commit.
    """
    runs = collect(results_root)
    bound = bind_runs(results_root, runs)
    rows = paired_rows(runs)

    criteria = [
        evaluate_memory(memory_probe_path),
        evaluate_step_time(preflight_path),
        *evaluate_quality(rows),
        evaluate_stability(runs),
    ]
    criteria.sort(key=lambda item: item["number"])
    unauditable = [c["number"] for c in criteria if c["verdict"] == VERDICT_NOT_AUDITABLE]

    report = {
        "schema_version": SCHEMA_VERSION,
        **decide(criteria, rows),
        "scope": SCOPE,
        "claim_limit": CLAIM_LIMIT,
        "criteria": criteria,
        "criteria_audited": len(criteria) - len(unauditable),
        "criteria_not_auditable": unauditable,
        "paired_seeds": list(SEEDS),
        # `paired_rows` refuses unequal masks, so reaching here means it held.
        # Recorded so the next reader does not have to re-derive it.
        "paired_masks_verified": True,
        "per_seed": rows,
        "means": {
            "hcdr3_token_acc_680": _mean([r["hcdr3_token_acc_680"] for r in rows]),
            "hcdr3_token_acc_1024": _mean([r["hcdr3_token_acc_1024"] for r in rows]),
            "hcdr3_token_acc_gain_pp": _mean(
                [r["hcdr3_token_acc_delta_pp"] for r in rows]
            ),
            "mlm_loss_680": _mean([r["mlm_loss_680"] for r in rows]),
            "mlm_loss_1024": _mean([r["mlm_loss_1024"] for r in rows]),
            "hcdr3_span_exact_680": _mean([r["hcdr3_span_exact_680"] for r in rows]),
            "hcdr3_span_exact_1024": _mean([r["hcdr3_span_exact_1024"] for r in rows]),
        },
        "schedule": {
            "total_updates": TOTAL_UPDATES,
            "warmup_updates": WARMUP_UPDATES,
            "post_warmup_updates": TOTAL_UPDATES - WARMUP_UPDATES,
        },
        "commit": bound[0]["commit"],
        "runs": bound,
    }
    return report


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON manifest deterministically (sorted keys, LF, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"  wrote {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="verify preconditions and run the six arms")
    launch.add_argument("--dry-run", action="store_true",
                        help="verify everything and print the commands without running.")
    launch.add_argument("--allow-branch", action="store_true",
                        help="permit training a non-main SHA (discouraged).")
    launch.add_argument("--output-json", type=Path, default=None)

    compare_parser = sub.add_parser(
        "compare", help="apply the promotion rule to six finished runs"
    )
    compare_parser.add_argument("--results-root", type=Path,
                                default=PROJECT_ROOT / "checkpoints/experiments")
    # Repo-anchored, never CWD-relative (Mirror BUG-18). Left as None so an
    # EXPLICIT path that is missing can be told apart from an absent default:
    # the first is a user error, the second is the honest state of criterion 1
    # and 2's evidence and downgrades the criterion instead of failing the run.
    compare_parser.add_argument("--preflight", type=Path, default=None,
                                help="pipeline preflight artifact (criterion 2). "
                                     "Default: outputs/j11-pipeline-preflight.json")
    compare_parser.add_argument("--memory-probe", type=Path, default=None,
                                help="dual-stream memory probe at 288/1024 batch 16 "
                                     "(criterion 1). Default: "
                                     "outputs/j11-dual-stream-memory.json")
    compare_parser.add_argument("--output-json", type=Path,
                                default=PROJECT_ROOT / "specs/evidence/j11-comparison.json")
    return parser


def resolve_evidence_path(
    explicit: Path | None, default: Path, label: str, parser: argparse.ArgumentParser
) -> Path:
    """
    Resolve an optional evidence artifact, failing loudly only when asked to.

    An explicitly named path that does not exist is a user error and stops the
    run. An absent default is not: the criterion it feeds reports
    `not_auditable`, which is the whole point of that verdict. Silently emitting
    a pass in either case is what Mirror BUG-18/21 did with
    `compatibility_score: null` -- exit 0, and nothing scored.
    """
    if explicit is not None:
        if not explicit.exists():
            parser.error(f"--{label} does not exist: {explicit}")
        return explicit
    if not default.exists():
        print(
            f"  note: no {label} artifact at {default}; the criterion it feeds "
            "will be reported as not_auditable.",
            file=sys.stderr,
        )
    return default


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

        manifest = {
            "schema_version": "j11-launch/1",
            "commit": lineage["commit"],
            "branch": lineage["branch"],
            "warnings": warnings,
            "schedule": {
                "total_updates": TOTAL_UPDATES,
                "warmup_updates": WARMUP_UPDATES,
                "post_warmup_updates": TOTAL_UPDATES - WARMUP_UPDATES,
                "widths": list(WIDTHS),
                "seeds": list(SEEDS),
            },
            "runs": [
                {
                    "width": width,
                    "seed": seed,
                    "config": str(config_path(width, seed).relative_to(PROJECT_ROOT)),
                    "output_dir": f"checkpoints/experiments/j11_w{width}_s{seed}",
                }
                for width in WIDTHS
                for seed in SEEDS
            ],
        }

        if args.dry_run:
            print("\n  --dry-run: nothing was executed.")
            if args.output_json is not None:
                write_manifest(args.output_json, manifest)
            return 0

        if args.output_json is not None:
            write_manifest(args.output_json, manifest)

        for command in commands:
            print(f"\n=== {' '.join(command)}")
            completed = subprocess.run(command, cwd=PROJECT_ROOT)
            if completed.returncode == 0:
                # The SHA goes NEXT TO the checkpoints, not only in the launch
                # manifest: a manifest can be moved or lost, and then the weights
                # are unattributable. This file cannot be separated from them.
                config_file = Path(command[-1])
                width = int(config_file.stem.split("_")[1][1:])
                seed = int(config_file.stem.split("_")[2][1:])
                run_dir = PROJECT_ROOT / "checkpoints/experiments" / f"j11_w{width}_s{seed}"
                if run_dir.exists():
                    write_manifest(
                        run_dir / "j11_provenance.json",
                        {
                            "schema_version": "j11-run-provenance/1",
                            "commit": lineage["commit"],
                            "branch": lineage["branch"],
                            "width": width,
                            "seed": seed,
                            "total_updates": TOTAL_UPDATES,
                            "warmup_updates": WARMUP_UPDATES,
                        },
                    )
            if completed.returncode != 0:
                print(
                    f"REFUSED: run failed ({completed.returncode}); stopping. A partial "
                    "evidence set may not be compared.",
                    file=sys.stderr,
                )
                return 1
        return 0

    # compare
    parser = build_arg_parser()
    preflight = resolve_evidence_path(
        args.preflight, PROJECT_ROOT / "outputs/j11-pipeline-preflight.json",
        "preflight", parser,
    )
    memory_probe = resolve_evidence_path(
        args.memory_probe, PROJECT_ROOT / "outputs/j11-dual-stream-memory.json",
        "memory-probe", parser,
    )

    try:
        report = compare(args.results_root, preflight, memory_probe)
    except LaunchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    print(f"J11 comparison ({report['schema_version']})")
    print(f"  commit          : {report['commit']}")
    print(f"  paired seeds    : {', '.join(str(s) for s in report['paired_seeds'])}")
    print()
    print(f"  {'seed':>8}  {'tok-acc 680':>12} {'1024':>10} {'delta pp':>10}"
          f" {'mlm delta':>11} {'span delta pp':>14}")
    for row in report["per_seed"]:
        print(f"  {row['seed']:>8}  {row['hcdr3_token_acc_680'] * 100:12.3f}"
              f" {row['hcdr3_token_acc_1024'] * 100:10.3f}"
              f" {row['hcdr3_token_acc_delta_pp']:+10.3f}"
              f" {row['mlm_loss_delta_nats']:+11.5f}"
              f" {row['hcdr3_span_exact_delta_pp']:+14.3f}")
    print()
    for criterion in report["criteria"]:
        mark = {
            VERDICT_PASS: "PASS", VERDICT_FAIL: "FAIL",
            VERDICT_NOT_AUDITABLE: "N/A ",
        }[criterion["verdict"]]
        print(f"  {mark}  {criterion['number']}. {criterion['statement']}")
        if criterion["reason"]:
            print(f"        {criterion['reason']}")
    print()
    print(f"  SELECTED WIDTH  : {report['selected_width']}")
    print(f"  reason          : {report['primary_reason']}")
    print(f"  audited         : {report['criteria_audited']}/{len(report['criteria'])}"
          f" criteria; not auditable: {report['criteria_not_auditable'] or 'none'}")
    print(f"  claim limit     : {report['claim_limit']}")

    if args.output_json is not None:
        write_manifest(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

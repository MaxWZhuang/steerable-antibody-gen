"""
J11 comparison: the promotion rule must be arithmetic, not prose.

`collect` already refuses an incomplete evidence set, but until now `compare`
stopped there and printed a sentence pointing at the spec. That is the failure
mode the whole harness exists to prevent, one level up: a predeclared rule that
nothing executes is a rule someone applies by hand, in a chat window, with the
numbers already visible. The verdict has to be a machine-emitted artifact bound
to the runs it was computed from.

Three properties carry the weight here:

- **Every criterion must pass for 1024 to win.** The rule is an AND across seven
  clauses with 680 as the explicit tie/inconclusive fallback, so any clause that
  is not a pass promotes 680.
- **`not_auditable` is never a pass, and it blocks promotion.** Absence of
  evidence would otherwise let an unmeasured clause pay for a measured one. The
  two clauses that are unauditable for the executed runs -- dual-stream memory
  (1) and AMP-skip instability (7) -- are unauditable because their measurements
  were never retained, not because the rule cannot express them. Both become
  auditable the moment the artifact exists, which is what keeps the comparator
  falsifiable rather than a function that always returns 680.
- **The report is bound to its evidence.** Run fingerprints, the training commit,
  and a hash of the exact final metrics record, so a later reader can tell
  whether the numbers in the report still describe the runs on disk.

The fixtures are synthetic. `checkpoints/` is git-ignored, so a test that read
the real runs would pass only on the machine that produced them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

WIDTHS = (680, 1024)
SEEDS = (42, 31415, 271828)
COMMIT = "561312b62f3af82013886ce9f1756c955128ef33"


@pytest.fixture
def runner(project_root: Path):
    script = project_root.parents[1] / "scripts" / "run_j11_experiment.py"
    spec = importlib.util.spec_from_file_location("run_j11_experiment", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _val_record(
    token_acc: float, mlm_loss: float, span_exact: float, amp_skips: int | None
) -> dict:
    """
    One final-step validation record, in the shape `metrics.jsonl` carries.

    `amp_skips` is written only when not None: the six executed runs predate the
    counter being persisted, and that absence is exactly what criterion 7 has to
    report honestly.
    """
    record = {
        "epoch": 1,
        "phase": "train_eval",
        "training_stage": "base",
        "val": {
            "hcdr3_span_exact_match": span_exact,
            "hcdr3_target_tokens": 772352.0,
            "hcdr3_token_acc": token_acc,
            "hcdr3_valid_spans": 496.0,
            "loss": mlm_loss,
            "mlm_acc": 0.82,
            "mlm_loss": mlm_loss,
            "pair_acc": None,
            "pair_loss": 0.0,
        },
    }
    if amp_skips is not None:
        record["amp_skips"] = amp_skips
    return record


@pytest.fixture
def make_results():
    """
    Build a six-run results tree.

    ``token_acc`` maps (width, seed) -> HCDR3 token recovery; the remaining
    metrics default to values that satisfy criteria 5 and 6, so a test can move
    one axis at a time.
    """

    def _build(
        root: Path,
        token_acc: dict,
        mlm_loss: dict | None = None,
        span_exact: dict | None = None,
        amp_skips: dict | None = None,
        commit: str = COMMIT,
    ) -> Path:
        for width in WIDTHS:
            for seed in SEEDS:
                run = root / f"j11_w{width}_s{seed}"
                run.mkdir(parents=True, exist_ok=True)
                record = _val_record(
                    token_acc[(width, seed)],
                    (mlm_loss or {}).get((width, seed), 0.68),
                    (span_exact or {}).get((width, seed), 0.013),
                    (amp_skips or {}).get((width, seed)),
                )
                (run / "metrics.jsonl").write_text(
                    json.dumps({"epoch": 0, "phase": "pretrain_eval"}) + "\n"
                    + json.dumps(record) + "\n",
                    encoding="utf-8",
                )
                (run / "run_fingerprint.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_hash": f"run-{width}-{seed}",
                            "worktree_dirty": False,
                            "manifests": {
                                "source": {
                                    "commit": commit,
                                    "dirty": False,
                                    "content_hash": f"src-{width}-{seed}",
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "j11_provenance.json").write_text(
                    json.dumps(
                        {
                            "branch": "main",
                            "commit": commit,
                            "seed": seed,
                            "width": width,
                            "total_updates": 51_000,
                            "warmup_updates": 1_000,
                        }
                    ),
                    encoding="utf-8",
                )
        return root

    return _build


@pytest.fixture
def preflight(tmp_path: Path):
    """A pipeline preflight carrying both arms' median step times."""

    def _write(median_680: float = 0.15158, median_1024: float = 0.17993) -> Path:
        path = tmp_path / "preflight.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "j11-pipeline-preflight/1",
                    "arms": [
                        {"swiglu_hidden_dim": 680, "median_step_seconds": median_680},
                        {"swiglu_hidden_dim": 1024, "median_step_seconds": median_1024},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    return _write


@pytest.fixture
def memory_probe(tmp_path: Path):
    """
    A dual-stream memory probe at the canonical 288/1024 batch-16 shape.

    The shape fields are not decoration: a stage-1 probe would show far more
    headroom than the shape criterion 1 actually governs, and on this box the
    absence of an OOM is not evidence a config fits (driver spill to system RAM).
    """

    def _write(
        reserved_680: float = 2392.0,
        reserved_1024: float = 2808.0,
        overrides: dict | None = None,
    ) -> Path:
        overrides = overrides or {}
        path = tmp_path / "memory.json"
        rows = []
        for width, reserved, params in (
            (680, reserved_680, 9_120_000),
            (1024, reserved_1024, 12_290_000),
        ):
            row = {
                # descriptors: what model, at what shape, in what numeric regime
                "swiglu_hidden_dim": width,
                "model_kind": "antibody_antigen",
                "ffn_type": "swiglu",
                "norm_type": "rmsnorm",
                "position_encoding": "rope",
                "use_amp": True,
                "max_length": 288,
                "antigen_max_length": 1024,
                "batch_size": 16,
                # measurements
                "device_total_mib": 4095.7,
                "peak_reserved_mib": reserved,
                "peak_allocated_mib": reserved - 290.0,
                "total_parameters": params,
                "fits_without_driver_spill": True,
            }
            row.update(overrides.get(width, {}))
            rows.append(row)
        path.write_text(
            json.dumps({"schema_version": "j11-memory-probe/1", "results": rows}),
            encoding="utf-8",
        )
        return path

    return _write


def _sweep(delta: float, base: float = 0.462) -> dict:
    """1024 ahead of 680 by `delta` (absolute, not pp) at every seed."""
    return {
        **{(680, seed): base for seed in SEEDS},
        **{(1024, seed): base + delta for seed in SEEDS},
    }


def _clean_skips() -> dict:
    """No AMP skips in any arm -- the state criterion 7 wants to confirm."""
    return {(width, seed): 0 for width in WIDTHS for seed in SEEDS}


def _criteria(report: dict) -> dict:
    return {c["number"]: c for c in report["criteria"]}


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #
def test_a_fully_audited_sweep_above_the_threshold_promotes_1024(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The rule is falsifiable. Given every artifact and a gain clearing +1.0 pp at
    all three paired seeds with no regression, 1024 is promoted -- otherwise a
    comparator that only ever returns 680 would be indistinguishable from one
    that works.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert report["selected_width"] == 1024
    assert report["width_1024_promoted"] is True
    assert report["criteria_audited"] == 7
    assert report["criteria_not_auditable"] == []


def test_a_sub_threshold_sweep_promotes_680(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The J11 outcome. A consistent but small gain fails criterion 3, and 680 is
    the predeclared fallback: the extra capacity must earn its place.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert report["selected_width"] == 680
    assert report["width_1024_promoted"] is False
    assert "0.306" in report["primary_reason"]
    assert "1.0" in report["primary_reason"]

    criteria = _criteria(report)
    assert criteria[3]["verdict"] == "fail"
    assert criteria[4]["verdict"] == "pass"  # all three seeds still favour 1024


def test_a_split_seed_sweep_promotes_680_even_when_the_mean_clears(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Criterion 4 is not redundant with criterion 3. One large seed can drag the
    mean over +1.0 pp while a second seed favours 680; that is a wider spread,
    not a reproducible gain.
    """
    token_acc = _sweep(0.02)
    token_acc[(1024, 271828)] = token_acc[(680, 271828)] - 0.001
    make_results(tmp_path, token_acc, amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    criteria = _criteria(report)
    assert criteria[3]["verdict"] == "pass"
    assert criteria[4]["verdict"] == "fail"
    assert report["selected_width"] == 680


def test_an_mlm_loss_regression_beyond_the_margin_promotes_680(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Criterion 5: a CDR3 gain may not be bought with a worse language model."""
    make_results(
        tmp_path,
        _sweep(0.015),
        mlm_loss={
            **{(680, seed): 0.68 for seed in SEEDS},
            **{(1024, seed): 0.70 for seed in SEEDS},
        },
        amp_skips=_clean_skips(),
    )
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert _criteria(report)[5]["verdict"] == "fail"
    assert report["selected_width"] == 680


def test_a_span_exact_regression_beyond_the_margin_promotes_680(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Criterion 6, in the direction that matters: 1024 worse than 680."""
    make_results(
        tmp_path,
        _sweep(0.015),
        span_exact={
            **{(680, seed): 0.02 for seed in SEEDS},
            **{(1024, seed): 0.01 for seed in SEEDS},
        },
        amp_skips=_clean_skips(),
    )
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert _criteria(report)[6]["verdict"] == "fail"
    assert report["selected_width"] == 680


def test_paired_deltas_are_reported_per_seed(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The experimental unit is the paired seed, of which there are three. The
    report carries all three deltas so a reader sees the spread rather than only
    a mean that hides it.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert report["paired_seeds"] == list(SEEDS)
    assert len(report["per_seed"]) == 3
    for row in report["per_seed"]:
        assert row["hcdr3_token_acc_delta_pp"] == pytest.approx(0.306, abs=1e-3)


# --------------------------------------------------------------------------- #
# `not_auditable` is never a pass, and it blocks promotion
# --------------------------------------------------------------------------- #
def test_an_unauditable_criterion_blocks_promotion(
    runner, tmp_path: Path, make_results, preflight
):
    """
    THE load-bearing test, and the actual state of the six executed runs. Every
    measurable criterion passes by a wide margin; criterion 1 has no retained
    dual-stream memory probe and criterion 7 no retained AMP-skip count. Absence
    of evidence is not a pass, so 680 is selected and the reason names the
    clauses that could not be checked rather than claiming a clean 680 win.
    """
    make_results(tmp_path, _sweep(0.015))  # no amp_skips, and no memory probe below
    report = runner.compare(tmp_path, preflight(), memory_probe_path=None)

    criteria = _criteria(report)
    assert criteria[1]["verdict"] == "not_auditable"
    assert criteria[7]["verdict"] == "not_auditable"
    for number in (2, 3, 4, 5, 6):
        assert criteria[number]["verdict"] == "pass"

    assert report["selected_width"] == 680
    assert report["width_1024_promoted"] is False
    assert "not auditable" in report["primary_reason"].lower()
    assert report["criteria_not_auditable"] == [1, 7]
    assert report["criteria_audited"] == 5


def test_a_not_auditable_verdict_is_never_rewritten_to_pass(
    runner, tmp_path: Path, make_results, preflight
):
    """The report cannot claim more verification than was performed."""
    make_results(tmp_path, _sweep(0.015))
    report = runner.compare(tmp_path, preflight(), memory_probe_path=None)

    verdicts = [c["verdict"] for c in report["criteria"]]
    assert verdicts.count("not_auditable") == 2
    assert len(report["criteria"]) == 7
    for criterion in report["criteria"]:
        if criterion["verdict"] == "not_auditable":
            assert criterion["reason"], "an unauditable clause must say why"


def test_criterion_7_reports_the_nan_half_it_can_actually_check(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Criterion 7 is two claims: no NaNs, and no unexplained AMP skips. The first
    is checkable from the retained records; only the second needs the counter.
    Saying so beats collapsing both into one opaque `not_auditable`.
    """
    make_results(tmp_path, _sweep(0.015))
    report = runner.compare(tmp_path, preflight(), memory_probe())

    criterion = _criteria(report)[7]
    assert criterion["verdict"] == "not_auditable"
    assert criterion["metrics_finite"] is True
    assert "amp_skips" in criterion["reason"]


def test_a_nonzero_amp_skip_count_fails_criterion_7(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """When the counter IS retained, it decides rather than being decoration."""
    skips = _clean_skips()
    skips[(1024, 31415)] = 37
    make_results(tmp_path, _sweep(0.015), amp_skips=skips)
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert _criteria(report)[7]["verdict"] == "fail"
    assert report["selected_width"] == 680


def test_a_nan_metric_fails_criterion_7(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """A NaN in a final record is instability, not a missing measurement."""
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = tmp_path / "j11_w1024_s42" / "metrics.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"mlm_acc": 0.82', '"mlm_acc": NaN'),
                    encoding="utf-8")
    report = runner.compare(tmp_path, preflight(), memory_probe())

    criterion = _criteria(report)[7]
    assert criterion["verdict"] == "fail"
    assert criterion["metrics_finite"] is False


def test_a_missing_preflight_makes_the_step_time_criterion_unauditable(
    runner, tmp_path: Path, make_results, memory_probe
):
    """
    Criterion 2 is computed from the pipeline preflight. With no artifact the
    honest verdict is `not_auditable`, never a silent pass on the spec's prose.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, tmp_path / "absent.json", memory_probe())

    assert _criteria(report)[2]["verdict"] == "not_auditable"
    assert report["criteria_not_auditable"] == [2]
    assert report["selected_width"] == 680


def test_the_step_time_criterion_reads_the_preflight_artifact(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """18.7% measured from both arms' median step times, against a 35% ceiling."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    criterion = _criteria(report)[2]
    assert criterion["verdict"] == "pass"
    assert criterion["measured"] == pytest.approx(0.187, abs=1e-3)


def test_a_slowdown_beyond_the_ceiling_fails_criterion_2(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(median_1024=0.30), memory_probe())

    assert _criteria(report)[2]["verdict"] == "fail"
    assert report["selected_width"] == 680


def test_insufficient_memory_headroom_fails_criterion_1(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """25% reserved headroom at the dual-stream shape, for BOTH arms."""
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe(reserved_1024=3600.0))

    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "fail"
    assert report["selected_width"] == 680


def test_a_memory_probe_at_the_wrong_shape_is_not_accepted(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    A stage-1 probe shows headroom the dual-stream shape does not have. Reading
    one as if it answered criterion 1 is how a memory claim silently becomes
    untrue -- so a shape mismatch is `not_auditable`, not a pass.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["results"]:
        row["antigen_max_length"] = 192
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = runner.compare(tmp_path, preflight(), path)
    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "not_auditable"
    assert "288" in criterion["reason"] or "1024" in criterion["reason"]


# --------------------------------------------------------------------------- #
# The report is bound to its evidence
# --------------------------------------------------------------------------- #
def test_the_report_binds_every_run_to_its_fingerprint_and_final_record(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Six runs, each with the fingerprint hash, the training commit, and a hash of
    the exact final record the verdict was computed from. Without this the
    report is a set of numbers with no way to tell which runs produced them.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert len(report["runs"]) == 6
    for row in report["runs"]:
        assert row["commit"] == COMMIT
        assert len(row["run_fingerprint_sha256"]) == 64
        assert len(row["final_metrics_sha256"]) == 64
        assert row["width"] in WIDTHS
        assert row["seed"] in SEEDS


def test_a_changed_final_record_changes_its_hash(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """The binding is real: perturb one arm's metric and its recorded hash moves."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    before = runner.compare(tmp_path, preflight(), memory_probe())

    make_results(tmp_path, _sweep(0.00307), amp_skips=_clean_skips())
    after = runner.compare(tmp_path, preflight(), memory_probe())

    was = {(r["width"], r["seed"]): r["final_metrics_sha256"] for r in before["runs"]}
    now = {(r["width"], r["seed"]): r["final_metrics_sha256"] for r in after["runs"]}
    assert was[(680, 42)] == now[(680, 42)]      # untouched arm
    assert was[(1024, 42)] != now[(1024, 42)]    # perturbed arm


def test_runs_trained_from_different_commits_are_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Six runs from two commits are not one experiment. The launcher pins a single
    SHA precisely so this cannot happen silently; the comparator must not undo
    that by averaging across it.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    stray = tmp_path / "j11_w1024_s271828" / "j11_provenance.json"
    payload = json.loads(stray.read_text(encoding="utf-8"))
    payload["commit"] = "0" * 40
    stray.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.LaunchRefused, match="commit"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_a_missing_provenance_file_is_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Weights whose commit is unknown are unattributable evidence."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    (tmp_path / "j11_w680_s42" / "j11_provenance.json").unlink()

    with pytest.raises(runner.LaunchRefused, match="provenance"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_the_report_carries_its_scope_and_claim_limit(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The J24 precedent: a selection report states what it does NOT license. J11
    ran a fraction of one training epoch, so it cannot speak to behaviour after
    full training -- and a later reader will meet the number without the context.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert "asymptotic" in report["claim_limit"].lower()
    assert report["schema_version"] == runner.SCHEMA_VERSION


def test_the_report_is_deterministic(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Same evidence in, byte-identical report out -- so a rerun that differs
    means the evidence moved, not the comparator."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    first = runner.compare(tmp_path, preflight(), memory_probe())
    second = runner.compare(tmp_path, preflight(), memory_probe())

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_compare_still_refuses_an_incomplete_evidence_set(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """The completeness gate survives the addition of the arithmetic."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    for stale in (tmp_path / "j11_w1024_s271828").iterdir():
        stale.unlink()
    (tmp_path / "j11_w1024_s271828").rmdir()

    with pytest.raises(runner.LaunchRefused, match="incomplete evidence set"):
        runner.compare(tmp_path, preflight(), memory_probe())


# --------------------------------------------------------------------------- #
# Provenance validation must fail CLOSED
#
# The first version of this comparator accepted a missing fingerprint as null,
# six empty commit strings as "one shared commit", and never checked the
# provenance's width/seed/schedule against the run it sat next to. Every one of
# those is a way for unattributable or mislabelled evidence to reach a verdict
# while the report still looks fully bound.
# --------------------------------------------------------------------------- #
def test_a_missing_run_fingerprint_is_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """A null hash in the report is not a binding; it is a hole shaped like one."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    (tmp_path / "j11_w680_s42" / "run_fingerprint.json").unlink()

    with pytest.raises(runner.LaunchRefused, match="run_fingerprint"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_an_empty_commit_is_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Six empty strings are all equal, so an equality check alone reads them as
    one shared commit. Unattributable evidence must stop the comparison.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips(), commit="")

    with pytest.raises(runner.LaunchRefused, match="commit"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_a_malformed_commit_is_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Not a 40-character hex SHA is not a commit."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips(), commit="HEAD")

    with pytest.raises(runner.LaunchRefused, match="commit"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_provenance_width_must_match_its_run_directory(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The directory name says which arm this is; so does the provenance. If they
    disagree, one of them is wrong and the comparator cannot know which -- so
    the 680 column may be reading a 1024 run.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w680_s42" / "j11_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["width"] = 1024
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.LaunchRefused, match="width"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_provenance_seed_must_match_its_run_directory(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Same argument as width: a mislabelled seed breaks the pairing silently."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w1024_s31415" / "j11_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed"] = 42
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.LaunchRefused, match="seed"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_provenance_schedule_must_match_the_frozen_design(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    A run that stopped at a different step is not part of this experiment. The
    launcher records the schedule beside the weights precisely so a shortened
    arm cannot be compared as if it were the frozen one.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w1024_s42" / "j11_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["total_updates"] = 40_000
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.LaunchRefused, match="51,000|total_updates"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_the_fingerprint_commit_must_match_the_provenance_commit(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Two independent records of the same fact. The launcher writes
    j11_provenance.json; the trainer writes the fingerprint from its own git
    read. Agreement is cheap to check and is the only thing that catches a
    provenance file copied from another run.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w680_s31415" / "run_fingerprint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifests"]["source"]["commit"] = "1" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.LaunchRefused, match="disagree|fingerprint"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_a_run_from_a_dirty_worktree_is_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The launcher refuses to START from a dirty tree. A fingerprint that records
    it ran dirty anyway means the recorded commit does not describe the code
    that produced the weights.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w1024_s271828" / "run_fingerprint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["worktree_dirty"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.LaunchRefused, match="dirty"):
        runner.compare(tmp_path, preflight(), memory_probe())


# --------------------------------------------------------------------------- #
# Paired masks are ENFORCED, not just reported
# --------------------------------------------------------------------------- #
def test_unequal_validation_masks_are_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The whole pairing argument is that both widths saw the SAME validation mask
    at a seed. The report quoted the 680 arm's counts without ever comparing
    them to 1024, so an unpaired comparison would have produced a confident
    verdict with a paired-looking table.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w1024_s42" / "metrics.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"hcdr3_target_tokens": 772352.0', '"hcdr3_target_tokens": 700000.0'
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.LaunchRefused, match="hcdr3_target_tokens"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_unequal_valid_span_counts_are_refused(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Same argument for the span denominator criterion 6 divides by."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    path = tmp_path / "j11_w680_s271828" / "metrics.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"hcdr3_valid_spans": 496.0', '"hcdr3_valid_spans": 512.0'
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.LaunchRefused, match="hcdr3_valid_spans"):
        runner.compare(tmp_path, preflight(), memory_probe())


def test_the_report_records_the_paired_mask_it_verified(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """Having checked it, say so -- otherwise the next reader re-derives it."""
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    assert report["paired_masks_verified"] is True


# --------------------------------------------------------------------------- #
# Cross-machine determinism
# --------------------------------------------------------------------------- #
def test_evidence_paths_are_repo_relative_and_carry_no_username(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    An absolute path embeds the local username and the checkout location, so two
    machines comparing the same evidence produce different reports and a
    committed artifact leaks a home directory.
    """
    make_results(tmp_path, _sweep(0.00306), amp_skips=_clean_skips())
    report = runner.compare(tmp_path, preflight(), memory_probe())

    blob = json.dumps(report)
    assert "\\\\" not in blob, "no Windows path separators anywhere in the report"
    for criterion in report["criteria"]:
        source = criterion.get("source")
        if source is None:
            continue
        assert not Path(source).is_absolute()
        assert ":" not in source, f"drive letter leaked into {source!r}"


def test_repo_paths_are_serialized_relative_to_the_repository_root(runner):
    """An artifact inside the repo is named by its repo-relative POSIX path."""
    inside = runner.PROJECT_ROOT / "outputs" / "j11-pipeline-preflight.json"
    assert runner.portable_path(inside) == "outputs/j11-pipeline-preflight.json"


def test_paths_outside_the_repository_keep_only_their_name(runner, tmp_path: Path):
    """Nothing outside the checkout can be named without leaking where it lives."""
    outside = tmp_path / "somewhere" / "probe.json"
    assert runner.portable_path(outside) == "probe.json"


def test_the_default_report_path_is_tracked_not_ignored(runner):
    """
    `outputs/` is git-ignored (.gitignore), so a report written there vanishes in
    a fresh clone -- taking the six fingerprint and metrics hashes with it, which
    is the entire point of emitting them. The default must land somewhere the
    repository keeps.
    """
    parser = runner.build_arg_parser()
    default = parser.parse_args(["compare"]).output_json
    relative = default.resolve().relative_to(runner.PROJECT_ROOT).as_posix()

    ignored = (runner.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert not relative.startswith("outputs/")
    assert relative.split("/")[0] + "/" not in ignored


# --------------------------------------------------------------------------- #
# Criterion 1's probe must prove WHAT it measured, not just the shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field, wrong",
    [
        ("model_kind", "antibody"),      # single-stream: no antigen tower at all
        ("ffn_type", "mlp"),             # legacy block, not the promoted one
        ("norm_type", "layernorm"),
        ("position_encoding", "learned"),
        ("use_amp", False),              # fp32 activations are a different memory regime
    ],
)
def test_a_probe_of_the_wrong_model_is_not_accepted(
    runner, tmp_path: Path, make_results, preflight, memory_probe, field, wrong
):
    """
    Shape alone is insufficient. A 288/1024 batch-16 probe of the single-stream
    model, or of the legacy block, or without AMP, measures a different thing and
    would answer criterion 1 with a number that does not describe the arms.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe(overrides={680: {field: wrong}, 1024: {field: wrong}})
    report = runner.compare(tmp_path, preflight(), path)

    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "not_auditable"
    assert field in criterion["reason"]


def test_a_probe_missing_a_required_descriptor_is_not_accepted(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """An absent descriptor is not an implicit match."""
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["results"]:
        del row["model_kind"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = runner.compare(tmp_path, preflight(), path)
    assert _criteria(report)[1]["verdict"] == "not_auditable"


def test_a_probe_whose_arms_differ_off_axis_is_not_accepted(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Rule 3 applied to the probe itself: the two rows may differ in the width
    under test and in what was measured, nothing else. Otherwise the headroom
    gap is attributable to whatever else moved.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe(overrides={1024: {"batch_size": 8}})
    report = runner.compare(tmp_path, preflight(), path)

    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "not_auditable"


def test_a_probe_from_a_different_device_is_not_accepted(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Headroom is a fraction of a specific card. Two rows measured against
    different device totals are not one comparison.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe(overrides={1024: {"device_total_mib": 8192.0}})
    report = runner.compare(tmp_path, preflight(), path)

    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "not_auditable"
    assert "device" in criterion["reason"].lower()


def test_a_probe_whose_wider_arm_is_not_larger_is_not_accepted(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    The 1024 arm has strictly more parameters by construction. A probe that says
    otherwise measured something other than the two arms.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe(overrides={1024: {"total_parameters": 9_120_000}})
    report = runner.compare(tmp_path, preflight(), path)

    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "not_auditable"
    assert "parameter" in criterion["reason"].lower()


def test_a_probe_reporting_driver_spill_fails_rather_than_downgrades(
    runner, tmp_path: Path, make_results, preflight, memory_probe
):
    """
    Spill is a measurement, not a missing one: on this box CUDA falls back to
    system RAM instead of raising, so a spilling config that "ran fine" is
    exactly the failure criterion 1 exists to catch.
    """
    make_results(tmp_path, _sweep(0.015), amp_skips=_clean_skips())
    path = memory_probe(overrides={1024: {"fits_without_driver_spill": False}})
    report = runner.compare(tmp_path, preflight(), path)

    criterion = _criteria(report)[1]
    assert criterion["verdict"] == "fail"
    assert report["selected_width"] == 680

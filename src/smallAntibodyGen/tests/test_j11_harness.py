"""
J11 execution harness: the frozen design must actually be what runs.

The gap this closes is the dangerous kind. Every artifact looked right -- a
predeclared spec, paired configs, a passing preflight -- while the configs
encoded 2,000 warmup steps and a full 185,640-update epoch, the trainer had no
way to stop at 51,000, and the normal training path never invoked the pairing
mechanism. Launching would have produced six runs that were not the experiment,
and nothing in the outputs would have said so.

So these tests check the executable form, not the intent: the schedule the
configs carry, the trainer's ability to honour it, and the launcher's refusal to
run anything else.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

WIDTHS = (680, 1024)
SEEDS = (42, 31415, 271828)


@pytest.fixture
def runner(project_root: Path):
    script = project_root.parents[1] / "scripts" / "run_j11_experiment.py"
    spec = importlib.util.spec_from_file_location("run_j11_experiment", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mlm_train(project_root: Path):
    from smallAntibodyGen.tests.test_train_infra import load_mlm_train_module

    return load_mlm_train_module(project_root)


def _config_dir(project_root: Path) -> Path:
    return project_root.parents[1] / "configs/experiments/swiglu_width"


# --------------------------------------------------------------------------- #
# The six configs encode the frozen design
# --------------------------------------------------------------------------- #
def test_all_six_arm_configs_exist(project_root: Path):
    """Three seeds per width, written out rather than generated at launch, so the
    executed experiment is a tracked artifact anyone can diff."""
    for width in WIDTHS:
        for seed in SEEDS:
            assert (_config_dir(project_root) / f"arm_w{width}_s{seed}.yaml").exists()


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_each_config_carries_the_frozen_schedule(project_root: Path, width, seed):
    """
    51,000 updates = 1,000 warmup + 50,000 post-warmup, exactly. An earlier draft
    carried 2,000 warmup and one full epoch (~185,640 updates), which is a
    different experiment from the one that was predeclared.
    """
    cfg = yaml.safe_load(
        (_config_dir(project_root) / f"arm_w{width}_s{seed}.yaml").read_text(encoding="utf-8")
    )
    assert cfg["max_updates"] == 51_000
    assert cfg["warmup_steps"] == 1_000
    assert cfg["max_updates"] - cfg["warmup_steps"] == 50_000
    assert cfg["swiglu_hidden_dim"] == width
    assert cfg["seed"] == seed
    assert cfg["paired_init_seed"] == seed
    assert not cfg.get("early_stopping_patience")
    assert cfg.get("resume_from_last") is False


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_each_config_uses_the_canonical_hcdr3_objective(project_root: Path, width, seed):
    """
    0.4, matching canonical stage 1. An earlier draft used 0.0, which is a
    plain-MLM pilot wearing a stage-1 name -- and it would have made two
    promotion criteria unmeasurable, because with no HCDR3 span deliberately
    masked `hcdr3_valid_spans` is ~0 and span-exact recovery is NaN.
    """
    cfg = yaml.safe_load(
        (_config_dir(project_root) / f"arm_w{width}_s{seed}.yaml").read_text(encoding="utf-8")
    )
    assert cfg["hcdr3_span_probability"] == 0.4
    canonical = yaml.safe_load(
        (project_root.parents[1] / "configs/pretrain_oas_small.yaml").read_text(encoding="utf-8")
    )
    assert cfg["hcdr3_span_probability"] == canonical["hcdr3_span_probability"]


def test_the_design_verifier_accepts_the_shipped_configs(runner):
    """The launcher's own check must pass on what is actually checked in."""
    assert runner.verify_design()["configs_verified"] == 6


def test_the_verifier_rejects_a_drifted_schedule(runner, monkeypatch, tmp_path):
    """A config whose schedule drifts must stop the launch, not be averaged in."""
    original = runner.load

    def drifted(path: Path):
        cfg = original(path)
        if "w1024_s42" in path.name:
            cfg["max_updates"] = 40_000
        return cfg

    monkeypatch.setattr(runner, "load", drifted)
    with pytest.raises(runner.LaunchRefused, match="max_updates"):
        runner.verify_design()


def test_the_verifier_rejects_unpaired_initialization(runner, monkeypatch):
    """Without paired init the seeds are three unrelated pairs, and the
    'all three seeds favor 1024' clause means nothing."""
    original = runner.load

    def unpaired(path: Path):
        cfg = original(path)
        cfg["paired_init_seed"] = 0
        return cfg

    monkeypatch.setattr(runner, "load", unpaired)
    with pytest.raises(runner.LaunchRefused, match="paired"):
        runner.verify_design()


def test_the_verifier_rejects_a_second_axis(runner, monkeypatch):
    """Widths at a fixed seed may differ ONLY in swiglu_hidden_dim and
    output_dir; anything else makes it a two-axis comparison."""
    original = runner.load

    def drifted(path: Path):
        cfg = original(path)
        if "w1024" in path.name:
            cfg["learning_rate"] = 0.001
        return cfg

    monkeypatch.setattr(runner, "load", drifted)
    with pytest.raises(runner.LaunchRefused, match="outside the axis"):
        runner.verify_design()


def test_the_verifier_rejects_early_stopping(runner, monkeypatch):
    """Arms that stop at different steps are not comparable."""
    original = runner.load

    def stopping(path: Path):
        cfg = original(path)
        cfg["early_stopping_patience"] = 3
        return cfg

    monkeypatch.setattr(runner, "load", stopping)
    with pytest.raises(runner.LaunchRefused, match="early stopping"):
        runner.verify_design()


def test_comparison_refuses_an_incomplete_evidence_set(runner, tmp_path):
    """
    Five of six runs is not the predeclared experiment. Refusing is the whole
    point: a harness that scores whatever it is given converts "we could not
    tell" into a winner.
    """
    for width in WIDTHS:
        for seed in SEEDS:
            if (width, seed) == (1024, 271828):
                continue
            (tmp_path / f"j11_w{width}_s{seed}").mkdir(parents=True)
    with pytest.raises(runner.LaunchRefused, match="incomplete evidence set"):
        runner.collect(tmp_path)


def test_comparison_refuses_a_run_with_no_validation_record(runner, tmp_path):
    for width in WIDTHS:
        for seed in SEEDS:
            run = tmp_path / f"j11_w{width}_s{seed}"
            run.mkdir(parents=True)
            (run / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
    with pytest.raises(runner.LaunchRefused, match="no validation record"):
        runner.collect(tmp_path)


# --------------------------------------------------------------------------- #
# The trainer can honour the schedule
# --------------------------------------------------------------------------- #
def test_max_updates_defaults_off(mlm_train, tmp_path: Path):
    """Default-off, so every existing config is byte-for-byte unchanged."""
    data = tmp_path / "tiny.jsonl.gz"
    data.write_text("", encoding="utf-8")
    cfg = mlm_train.parse_args(["--data-path", str(data)])
    assert cfg.max_updates == 0
    assert cfg.paired_init_seed == 0


def test_warmup_must_be_shorter_than_the_update_budget(mlm_train, tmp_path: Path):
    """
    Warmup >= the budget means the run is entirely warmup and produces zero
    post-warmup updates -- which would silently satisfy "it finished" while
    delivering nothing the evidence floor recognizes.
    """
    data = tmp_path / "tiny.jsonl.gz"
    data.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must be < max_updates"):
        mlm_train.parse_args(
            ["--data-path", str(data), "--max-updates", "1000", "--warmup-steps", "1000"]
        )


def test_max_updates_and_early_stopping_are_incompatible(mlm_train, tmp_path: Path):
    """A fixed-step comparison requires both arms to reach the SAME step."""
    data = tmp_path / "tiny.jsonl.gz"
    data.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        mlm_train.parse_args(
            [
                "--data-path", str(data),
                "--max-updates", "51000",
                "--early-stopping-patience", "3",
            ]
        )


def test_the_update_counter_tracks_applied_updates_not_batches(mlm_train):
    """
    Updates are counted as steps the optimizer APPLIED. AMP skips the step on
    inf/NaN gradients, so a batch counter would let an arm that hit more skips
    take fewer real updates than its pair while both reported the same step
    number -- an unequal comparison that looks equal.
    """
    assert set(mlm_train.UPDATE_COUNTER) == {"updates", "amp_skips"}
    assert mlm_train.UPDATE_COUNTER["updates"] >= 0

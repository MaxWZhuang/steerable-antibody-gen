"""
Contract tests for the Rule-4 shortcut-baseline FLOOR (`report_shortcut_baselines`).

The defect these are written against: the three baseline helpers
(`fit_group_majority_baselines`, `evaluate_group_majority_baselines`,
`format_baseline_summary`) have always been gated on `is_antigen_stage`, but the
call site in `main()` was gated on `cfg.training_stage == "antigen_refine"` -- the
synthetic shuffled-negative stage, which was the only antigen stage that existed
when the baselines were added (commit ae6e323) and which NO checked-in config
uses. So the floor that change-control Rule 4 requires beside every compatibility
claim printed for no production run at all.

The old code was untestable by construction: every assertion had to be written
against the helpers, which were already correct, while the broken decision lived
inline in `main()`. So the stage gate now lives inside `report_shortcut_baselines`
and these tests drive the real decision -- plus one test that drives the real
`main()`, because a helper nobody calls is exactly the failure being fixed.

The second contract here is RNG neutrality. The helpers iterate DataLoaders, and
creating a DataLoader iterator draws from the global torch RNG. The block runs
BEFORE `build_model`, so without a snapshot/restore, switching the baselines on
for a stage would shift every freshly-initialized parameter and the dropout
stream -- a read-only diagnostic silently changing training results.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HEAVY = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQGLEWMGWINAGNGNTKYSQKFQGRVTITRDT"
    "SASTAYMELSSLRSEDTAVYYCARDRSTFDYWGQGTLVTVSS"
)
ANTIGENS = [
    "MKTIIALSYIFCLVFADYKDDDDKGSHMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVID",
    "GETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLA",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ",
    "LMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACD",
]

ANTIGEN_STAGES = (
    "antigen_refine",
    "antigen_real_label_refine",
    "antigen_hcdr3_infill_refine",
)


def load_mlm_train_module(project_root: Path):
    """Import scripts/mlm_train.py as a module (same pattern as the sibling suites)."""
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("mlm_train", scripts_dir / "mlm_train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    record_id: str,
    *,
    split: str,
    binder_label: int | None,
    target_index: int,
    is_strong_binder: bool,
) -> dict:
    """
    One processed antibody-antigen row.

    `binder_label` is populated by the producer ONLY for `affinity_type == "bool"`
    rows, so a KD-derived strong binder legitimately carries `binder_label=None`
    while still passing the stage-4 filter. Both shapes are exercised below.
    """
    antigen = ANTIGENS[target_index % len(ANTIGENS)]
    return {
        "record_id": record_id,
        "sequence": HEAVY,
        "sequence_heavy": HEAVY,
        "sequence_light": None,
        "sequence_antigen": antigen,
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": split,
        "length": len(HEAVY),
        "target_key": f"uniprot:p{target_index:05d}",
        "canonical_target_id": f"target-{target_index}",
        "target_name": f"target_{target_index}",
        "target_pdb": f"{target_index}abc",
        "target_uniprot": f"P{target_index:05d}",
        "antigen_length": len(antigen),
        "dataset": "asd-test",
        "confidence": "very_high",
        "affinity_type": "bool" if binder_label is not None else "kd",
        "affinity_raw": None,
        "processed_measurement_raw": "1.0",
        "processed_measurement_float": 1.0,
        "binder_label": binder_label,
        "is_strong_binder": is_strong_binder,
        "is_nanobody": True,
        "scfv": False,
        "cdr3_aa_heavy": "ARDRSTFDY",
        "cdr3_start_aa_heavy": 98,
        "cdr3_end_aa_heavy": 107,
        "cdr3_aa_light": None,
        "cdr3_start_aa_light": None,
        "cdr3_end_aa_light": None,
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": False,
        "metadata": {},
        "source_file": "tiny_antigen.parquet",
    }


def _write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def _mixed_label_corpus(path: Path) -> Path:
    """
    A corpus every antigen stage can consume: both binder classes present, both
    splits populated, several distinct targets so the group-majority baselines
    have more than one group to vote in.
    """
    records = []
    for i in range(24):
        records.append(
            _record(
                f"train-{i}",
                split="train",
                binder_label=i % 2,
                target_index=i % 4,
                is_strong_binder=(i % 2 == 1),
            )
        )
    for i in range(8):
        records.append(
            _record(
                f"val-{i}",
                split="val",
                binder_label=i % 2,
                target_index=i % 4,
                is_strong_binder=(i % 2 == 1),
            )
        )
    return _write(path, records)


def _cfg(mlm_train, *, data_path: Path, stage: str, tmp_path: Path):
    init_ckpt = tmp_path / "init_placeholder.pt"
    if not init_ckpt.exists():
        init_ckpt.write_text("placeholder", encoding="utf-8")
    return mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            stage,
            "--init-checkpoint",
            str(init_ckpt),
            "--output-dir",
            str(tmp_path / f"out_{stage}"),
            "--batch-size",
            "4",
            "--eval-batch-size",
            "4",
            "--max-length",
            "192",
        ]
    )


# --------------------------------------------------------------------------- #
# The gate: which stages get a floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", ANTIGEN_STAGES)
def test_every_antigen_stage_gets_its_shortcut_baseline_floor(
    stage: str, tmp_path: Path, project_root: Path
):
    """
    THE defect. Pre-fix only `antigen_refine` reached the invocation, so the two
    production stages -- the only stages any checked-in config runs -- printed no
    floor at all.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / f"corpus_{stage}.jsonl.gz")
    cfg = _cfg(mlm_train, data_path=data_path, stage=stage, tmp_path=tmp_path)
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    # Fixture power: a stage whose filter emptied a split cannot prove anything
    # about the reporting gate.
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    lines = mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    )

    assert lines, f"stage {stage!r} produced no baseline report at all"
    joined = "\n".join(lines)
    assert "[compat-baseline-fit]" in joined
    assert "[compat-baseline] train_labeled=" in joined
    assert "val_always_pos_acc=" in joined, (
        "the always-positive floor -- the number Rule 4 names explicitly -- is missing"
    )
    assert "val_canonical_target_majority_acc=" in joined


def test_report_is_printed_not_merely_returned(tmp_path: Path, project_root: Path, capsys):
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_print.jsonl.gz")
    cfg = _cfg(
        mlm_train, data_path=data_path, stage="antigen_real_label_refine", tmp_path=tmp_path
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    lines = mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    )
    captured = capsys.readouterr().out

    assert lines
    for line in lines:
        assert line in captured


@pytest.mark.parametrize("stage", ("base", "paired_refine"))
def test_non_antigen_stages_report_nothing(stage: str, tmp_path: Path, project_root: Path):
    """
    A stage with no antigen has no compatibility task, so there is no floor to
    print -- and printing a "not computable" line for it would be noise, not
    honesty.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / f"corpus_{stage}.jsonl.gz")
    init_ckpt = tmp_path / "init_placeholder.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")
    argv = [
        "--data-path",
        str(data_path),
        "--training-stage",
        stage,
        "--output-dir",
        str(tmp_path / f"out_{stage}"),
        "--batch-size",
        "4",
        "--eval-batch-size",
        "4",
        "--max-length",
        "192",
    ]
    if stage == "paired_refine":
        argv += ["--init-checkpoint", str(init_ckpt)]
    cfg = mlm_train.parse_args(argv)
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    assert len(train_dataset) > 0  # fixture power: the stage does have data
    assert mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    ) == []


# --------------------------------------------------------------------------- #
# Additivity: the diagnostic must not move the training RNG
# --------------------------------------------------------------------------- #


def test_reporting_the_floor_does_not_advance_the_global_rng(
    tmp_path: Path, project_root: Path
):
    """
    `report_shortcut_baselines` runs BEFORE `build_model`. If it consumed global
    RNG, switching it on for a production stage would change that stage's model
    initialization and dropout stream -- i.e. the "additive" diagnostic would
    silently change every training result.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_rng.jsonl.gz")
    cfg = _cfg(
        mlm_train, data_path=data_path, stage="antigen_real_label_refine", tmp_path=tmp_path
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    # Fixture power. If the raw helpers happened NOT to touch the global RNG on
    # this fixture (too few rows to build a loader, say), then asserting the
    # report leaves it unchanged would prove nothing at all.
    mlm_train.set_seed(cfg.seed)
    torch_before_raw = torch.get_rng_state().clone()
    raw_fit = mlm_train.fit_group_majority_baselines(train_dataset, tokenizer, cfg)
    assert raw_fit, "powerless fixture: nothing to fit, so no loader was iterated"
    assert not torch.equal(torch_before_raw, torch.get_rng_state()), (
        "powerless fixture: the raw baseline helpers did not advance the global "
        "torch RNG on this data, so restoring it around the report proves nothing"
    )

    mlm_train.set_seed(cfg.seed)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    lines = mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    )

    assert lines  # the work really happened
    assert torch.equal(torch_before, torch.get_rng_state()), (
        "the baseline report advanced the global torch RNG; every parameter "
        "initialized after it would differ from a run without the report"
    )
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_the_next_random_draw_is_identical_with_and_without_the_report(
    tmp_path: Path, project_root: Path
):
    """
    The consequence form of the RNG contract, stated the way `build_model` sees
    it: the first draw after the preflight must not depend on whether the floor
    was reported.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_draw.jsonl.gz")
    cfg = _cfg(
        mlm_train, data_path=data_path, stage="antigen_real_label_refine", tmp_path=tmp_path
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    mlm_train.set_seed(cfg.seed)
    without = torch.randn(8)

    mlm_train.set_seed(cfg.seed)
    mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    )
    with_report = torch.randn(8)

    assert torch.equal(without, with_report)


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #


def test_no_binary_labels_reports_not_computable_instead_of_silence(
    tmp_path: Path, project_root: Path
):
    """
    A KD-derived strong binder carries `binder_label=None`, so a stage-4 corpus
    built only from those has NO row the compatibility baseline can be fit on.
    Pre-fix that printed nothing, which reads as "no shortcut exists"; it must
    say the floor is not computable.
    """
    mlm_train = load_mlm_train_module(project_root)
    records = [
        _record(
            f"train-{i}", split="train", binder_label=None, target_index=i % 4,
            is_strong_binder=True,
        )
        for i in range(12)
    ] + [
        _record(
            f"val-{i}", split="val", binder_label=None, target_index=i % 4,
            is_strong_binder=True,
        )
        for i in range(4)
    ]
    data_path = _write(tmp_path / "corpus_unlabeled.jsonl.gz", records)
    cfg = _cfg(
        mlm_train,
        data_path=data_path,
        stage="antigen_hcdr3_infill_refine",
        tmp_path=tmp_path,
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    # Fixture power: the stage filter must keep these rows (otherwise the test
    # is about an empty dataset, not about unlabeled rows), and the fit must
    # genuinely be impossible.
    assert len(train_dataset) == 12
    assert len(val_dataset) == 4
    assert mlm_train.fit_group_majority_baselines(train_dataset, tokenizer, cfg) == {}

    lines = mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    )

    assert len(lines) == 1
    assert "not computable" in lines[0]
    assert "antigen_hcdr3_infill_refine" in lines[0]
    # And it must not fabricate a number.
    assert "always_pos_acc" not in lines[0]


def test_single_class_population_is_flagged_as_vacuous(tmp_path: Path, project_root: Path):
    """
    The stage-4 filter keeps strong binders only, so every row that does carry a
    binary label carries a 1. `always_pos_acc=1.0000` next to a model number
    would read as a hard floor the model cleared; it is a vacuous number and the
    report has to say so.
    """
    mlm_train = load_mlm_train_module(project_root)
    records = [
        _record(
            f"train-{i}", split="train", binder_label=1, target_index=i % 4,
            is_strong_binder=True,
        )
        for i in range(12)
    ] + [
        _record(
            f"val-{i}", split="val", binder_label=1, target_index=i % 4,
            is_strong_binder=True,
        )
        for i in range(4)
    ]
    data_path = _write(tmp_path / "corpus_single_class.jsonl.gz", records)
    cfg = _cfg(
        mlm_train,
        data_path=data_path,
        stage="antigen_hcdr3_infill_refine",
        tmp_path=tmp_path,
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    fit = mlm_train.fit_group_majority_baselines(train_dataset, tokenizer, cfg)
    # Fixture power: the misleading number must actually be produced, otherwise
    # asserting that it is flagged proves nothing.
    assert fit and fit["positive_rate"] == 1.0
    metrics = mlm_train.evaluate_group_majority_baselines(
        val_dataset, tokenizer, cfg, fit
    )
    assert metrics["always_positive_acc"] == 1.0

    lines = mlm_train.report_shortcut_baselines(
        train_dataset, val_dataset, None, None, tokenizer, cfg
    )
    joined = "\n".join(lines)

    assert "val_always_pos_acc=1.0000" in joined  # the number is still shown
    assert "WARNING" in joined
    assert "single-class" in joined


def test_mixed_class_population_is_not_flagged(tmp_path: Path, project_root: Path):
    """Guard the other direction: the warning must not fire on a real two-class split."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_mixed.jsonl.gz")
    cfg = _cfg(
        mlm_train, data_path=data_path, stage="antigen_real_label_refine", tmp_path=tmp_path
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    fit = mlm_train.fit_group_majority_baselines(train_dataset, tokenizer, cfg)
    assert 0.0 < fit["positive_rate"] < 1.0  # fixture power

    joined = "\n".join(
        mlm_train.report_shortcut_baselines(
            train_dataset, val_dataset, None, None, tokenizer, cfg
        )
    )
    assert "WARNING" not in joined


def test_probe_splits_are_reported_when_present(tmp_path: Path, project_root: Path):
    """The probe arms keep their historical labels and their historical order."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_probes.jsonl.gz")
    cfg = _cfg(
        mlm_train, data_path=data_path, stage="antigen_real_label_refine", tmp_path=tmp_path
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)
    train_dataset, known_probe, row_probe = mlm_train.build_diagnostic_datasets(
        train_dataset, cfg
    )
    assert known_probe is not None and len(known_probe) > 0  # fixture power
    assert row_probe is not None and len(row_probe) > 0

    joined = "\n".join(
        mlm_train.report_shortcut_baselines(
            train_dataset, val_dataset, known_probe, row_probe, tokenizer, cfg
        )
    )

    assert joined.index("train_labeled=") < joined.index("known_target_probe_labeled=")
    assert joined.index("known_target_probe_labeled=") < joined.index("row_random_probe_labeled=")
    assert joined.index("row_random_probe_labeled=") < joined.index("val_labeled=")


# --------------------------------------------------------------------------- #
# The invocation itself
# --------------------------------------------------------------------------- #


def test_main_prints_the_floor_for_the_production_stage(
    tmp_path: Path, project_root: Path, monkeypatch, capsys
):
    """
    Drives the real `main()`. Without this, deleting the call from `main()`
    leaves every helper test above green -- which is exactly the shape of the
    bug being fixed.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_main.jsonl.gz")

    parent = tmp_path / "parent.pt"
    parent_cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="base",
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    tokenizer = mlm_train.build_tokenizer()
    parent_model = mlm_train.build_model(tokenizer, parent_cfg, torch.device("cpu"))
    mlm_train.save_checkpoint(
        parent,
        parent_model,
        mlm_train.build_optimizer(parent_model, parent_cfg),
        parent_cfg,
        epoch=1,
        val_loss=1.0,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlm_train",
            "--data-path", str(data_path),
            "--training-stage", "antigen_real_label_refine",
            "--init-checkpoint", str(parent),
            "--output-dir", str(tmp_path / "run"),
            "--device", "cpu",
            "--no-resume-from-last",
            "--no-progress",
            "--epochs", "1",
            "--batch-size", "4",
            "--eval-batch-size", "4",
            "--max-length", "192",
            "--d-model", "32",
            "--n-heads", "4",
            "--n-layers", "1",
            "--d-ff", "64",
            "--dropout", "0.0",
        ],
    )
    mlm_train.main()

    out = capsys.readouterr().out
    assert "[compat-baseline-fit]" in out, (
        "main() ran the production antigen stage without printing the Rule 4 floor"
    )
    assert "[compat-baseline] train_labeled=" in out
    assert "val_always_pos_acc=" in out


def _run_stage3(mlm_train, monkeypatch, data_path: Path, parent: Path, out_dir: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlm_train",
            "--data-path", str(data_path),
            "--training-stage", "antigen_real_label_refine",
            "--init-checkpoint", str(parent),
            "--output-dir", str(out_dir),
            "--device", "cpu",
            "--no-resume-from-last",
            "--no-progress",
            "--epochs", "1",
            "--batch-size", "4",
            "--eval-batch-size", "4",
            "--max-length", "192",
            "--d-model", "32",
            "--n-heads", "4",
            "--n-layers", "1",
            "--d-ff", "64",
            "--dropout", "0.1",
        ],
    )
    mlm_train.main()


def test_training_result_is_byte_identical_with_and_without_the_report(
    tmp_path: Path, project_root: Path, monkeypatch
):
    """
    ADDITIVITY PROOF, in the form the change has to satisfy: a stage that did not
    previously reach the baseline block must train to exactly the same weights
    now that it does.

    The second arm no-ops `report_shortcut_baselines`, which is precisely the
    pre-change behavior for `antigen_real_label_refine` (the old gate skipped the
    whole block). Same seed, same corpus, same parent checkpoint; the only
    difference is whether the floor was computed.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = _mixed_label_corpus(tmp_path / "corpus_additive.jsonl.gz")

    parent = tmp_path / "parent.pt"
    parent_cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="base",
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.1,
    )
    tokenizer = mlm_train.build_tokenizer()
    parent_model = mlm_train.build_model(tokenizer, parent_cfg, torch.device("cpu"))
    mlm_train.save_checkpoint(
        parent,
        parent_model,
        mlm_train.build_optimizer(parent_model, parent_cfg),
        parent_cfg,
        epoch=1,
        val_loss=1.0,
    )

    with_report_dir = tmp_path / "run_with_report"
    _run_stage3(mlm_train, monkeypatch, data_path, parent, with_report_dir)

    real_report = mlm_train.report_shortcut_baselines
    monkeypatch.setattr(mlm_train, "report_shortcut_baselines", lambda *a, **k: [])
    without_report_dir = tmp_path / "run_without_report"
    _run_stage3(mlm_train, monkeypatch, data_path, parent, without_report_dir)
    monkeypatch.setattr(mlm_train, "report_shortcut_baselines", real_report)

    a = torch.load(with_report_dir / "last.pt", map_location="cpu")
    b = torch.load(without_report_dir / "last.pt", map_location="cpu")

    # Fixture power: the run must actually have antigen-side parameters that are
    # initialized from the global RNG rather than copied from the parent, or an
    # RNG shift would be invisible here.
    antigen_side = [
        k for k in a["model_state_dict"]
        if k.startswith(("antigen_", "cross_", "fuse", "compat"))
    ]
    assert antigen_side, "no freshly-initialized antigen-side parameters to compare"

    assert a["model_state_dict"].keys() == b["model_state_dict"].keys()
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]), (
            f"parameter {key} differs: computing the baselines changed training"
        )
    assert a["val_loss"] == b["val_loss"]

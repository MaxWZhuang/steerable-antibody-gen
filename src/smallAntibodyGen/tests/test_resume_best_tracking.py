"""Regression test: a resume must not overwrite `best.pt` with a worse model.

`last.pt`'s `val_loss` is the LAST epoch's selection value, not the best. Seeding
best-tracking from it on resume inflates the threshold, so the first mediocre
post-resume epoch beats it and `best.pt` is rewritten with weights strictly worse
than the pre-interrupt best -- silently, and with every shipped config setting
`resume_from_last: true`.

The fix stores the running best in the checkpoint as its own `best_val_loss` field.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_mlm_train_module(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_dataset(write_processed_jsonl_gz, tmp_path):
    import random as _random

    rng = _random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    records = []
    for i in range(16):
        seq = "".join(rng.choice(aa) for _ in range(30))
        records.append({
            "sequence": seq, "locus": "IGH", "chain_group": "heavy",
            "split": "train" if i < 12 else "val", "length": 30,
        })
    return write_processed_jsonl_gz(tmp_path / "tiny.jsonl.gz", records)


def _cfg(mlm_train, data_path, out_dir, *, epochs, resume):
    return mlm_train.TrainConfig(
        data_path=str(data_path), training_stage="base", output_dir=str(out_dir),
        epochs=epochs, batch_size=4, eval_batch_size=4, max_length=64,
        d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0,
        hcdr3_span_probability=0.0, shuffle_pair_probability=0.0,
        pair_loss_weight=0.0, learning_rate=1e-3, device="cpu",
        show_progress=False, resume_from_last=resume,
    )


def _run(mlm_train, cfg, val_losses):
    """Drive main() with a scripted validation-loss sequence.

    Only the calls on the REAL validation split are scripted. main() also
    evaluates the two diagnostic probes each epoch (`train_probe`,
    `row_random_probe`), so counting raw calls would misalign the sequence -- the
    split name is the reliable discriminator.
    """
    calls = {"n": 0}
    real_evaluate = mlm_train.evaluate

    def fake_evaluate(**kwargs):
        metrics = real_evaluate(**kwargs)
        if getattr(kwargs["val_dataset"], "split", None) != "val":
            return metrics
        # The first val call is main()'s pre-train baseline; give it a large loss
        # so it never wins, then script the per-epoch values.
        if calls["n"] == 0:
            metrics["loss"] = 99.0
        else:
            idx = calls["n"] - 1
            metrics["loss"] = val_losses[idx] if idx < len(val_losses) else 99.0
        calls["n"] += 1
        return metrics

    original_parse = mlm_train.parse_args
    mlm_train.parse_args = lambda *a, **k: cfg
    mlm_train.evaluate = fake_evaluate
    try:
        mlm_train.main()
    finally:
        mlm_train.parse_args = original_parse
        mlm_train.evaluate = real_evaluate


def test_resume_does_not_clobber_best_with_a_worse_model(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz, capsys
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = _tiny_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"

    # Epochs 1..4 score 1.0, 0.5, 0.9, 0.95 -> best.pt holds epoch 2 at 0.5,
    # last.pt holds epoch 4 whose own val_loss is 0.95.
    _run(mlm_train, _cfg(mlm_train, data_path, out_dir, epochs=4, resume=False),
         [1.0, 0.5, 0.9, 0.95])
    capsys.readouterr()

    best = torch.load(out_dir / "best.pt", map_location="cpu")
    last = torch.load(out_dir / "last.pt", map_location="cpu")
    assert best["epoch"] == 2 and best["val_loss"] == pytest.approx(0.5)
    assert last["epoch"] == 4 and last["val_loss"] == pytest.approx(0.95)
    # The running best must be recorded separately from this epoch's score.
    assert last["best_val_loss"] == pytest.approx(0.5)

    # Resume for a 5th epoch scoring 0.8 -- worse than the 0.5 best, better than
    # the 0.95 last. Under the bug, 0.8 < 0.95 held and best.pt was rewritten.
    _run(mlm_train, _cfg(mlm_train, data_path, out_dir, epochs=5, resume=True),
         [0.8])
    capsys.readouterr()

    best_after = torch.load(out_dir / "best.pt", map_location="cpu")
    assert best_after["epoch"] == 2, (
        f"best.pt was overwritten by epoch {best_after['epoch']} "
        f"(val_loss {best_after['val_loss']}); the epoch-2 model at 0.5 was better"
    )
    assert best_after["val_loss"] == pytest.approx(0.5)


def test_resume_still_updates_best_on_a_genuine_improvement(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz, capsys
):
    """The guard must not freeze best.pt -- a real improvement still wins."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _tiny_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"

    _run(mlm_train, _cfg(mlm_train, data_path, out_dir, epochs=2, resume=False),
         [1.0, 0.5])
    capsys.readouterr()
    _run(mlm_train, _cfg(mlm_train, data_path, out_dir, epochs=3, resume=True),
         [0.2])
    capsys.readouterr()

    best = torch.load(out_dir / "best.pt", map_location="cpu")
    assert best["epoch"] == 3 and best["val_loss"] == pytest.approx(0.2)


def test_legacy_checkpoint_without_best_val_loss_warns(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz, capsys
):
    """A pre-fix last.pt carries no running best, so say the risk out loud."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _tiny_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"

    _run(mlm_train, _cfg(mlm_train, data_path, out_dir, epochs=2, resume=False),
         [1.0, 0.5])
    capsys.readouterr()

    # Strip the new field to simulate a checkpoint written by the old code.
    payload = torch.load(out_dir / "last.pt", map_location="cpu")
    payload.pop("best_val_loss")
    torch.save(payload, out_dir / "last.pt")

    _run(mlm_train, _cfg(mlm_train, data_path, out_dir, epochs=3, resume=True),
         [0.9])
    out = capsys.readouterr().out
    assert "predates best_val_loss tracking" in out

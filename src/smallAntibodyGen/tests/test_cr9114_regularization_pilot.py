"""Evaluation data flow: held-out labels cannot alter either portfolio selection."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_evaluation_label_changes_do_not_change_selected_identities(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("regularization_pilot_under_test", scripts / "run_cr9114_regularization.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    ids = [f"{i:016b}" for i in range(64)]
    frame = pd.DataFrame({"genotype": ids, "split": "development", "mean": np.linspace(8., 10., 64),
                          "effective_sem": .1, "block": 7})
    def fake_scores(decoder, bound, genotypes, name):
        return -1 - np.array([int(g, 2) for g in genotypes], dtype=float) / 100, np.tile([1., 0.], (len(genotypes), 1))
    samples = [ids[0], ids[0], ids[2], ids[3]]
    monkeypatch.setattr(runner, "score_and_embed", fake_scores)
    monkeypatch.setattr(runner, "sample_table", lambda *args: pd.DataFrame({"genotype": samples, "log_q": fake_scores(None, None, samples, "")[0]}))
    config = {"evaluation_samples": 4, "evaluation_sample_seed": 1}
    shortlist = {"calibration": {"selected_multiplier": 2, "quality_threshold": 9.}}
    frozen = np.tile([1., 0.], (len(frame), 1))
    results = []
    for i, records in enumerate((frame, frame.assign(mean=frame["mean"].iloc[::-1].to_numpy()))):
        directory = tmp_path / str(i)
        directory.mkdir()
        results.append(runner.evaluate(SimpleNamespace(decoder=None), None, None, records, frozen, config, shortlist, directory, "test"))
    for mode in ("ordinary", "diverse"):
        for k in ("16", "32"):
            first, second = (r["portfolios"][mode][k] for r in results)
            assert first["selected_genotypes"] == second["selected_genotypes"]
            assert first["mean_affinity"] != second["mean_affinity"]
    assert results[0]["diversity"]["unique_genotypes"] == 3  # duplicates retained

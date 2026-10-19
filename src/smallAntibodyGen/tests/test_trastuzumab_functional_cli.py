"""The functional arm through the real command line: planning, training, resume, inference (M2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import cli
from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol as fp
from smallAntibodyGen.experiments.trastuzumab_poc import inputs
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import read_json
from smallAntibodyGen.tests.test_trastuzumab_cli import mid_config, run, sha, write
from smallAntibodyGen.tests.test_trastuzumab_exports import _panel_document
from smallAntibodyGen.tests.test_trastuzumab_functional_data import KD, construct_row, group_row, heavy
from smallAntibodyGen.tests.test_trastuzumab_functional_runner import SMALL_VALUES, small_document


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def functional_section(**changes):
    section = {
        "config": {"gradient_mode": "shared", "weight": 1.0, "pairs_per_update": 2, "margin": 0.1,
                   "head_lr_scale": 1.0, "head_weight_decay": 0.0, "head_gradient_clip": 1.0,
                   "minimum_compared_fraction": 1.0},
        "constructs_path": "constructs.json", "measurements_path": "measurements.json",
        "budget_n": 2, "budget_seed": "functional-budget"}
    section.update(changes)
    return section


def cli_world(tmp_path, *, updates=3, singleton=False, forbidden_antigen=False, **changes):
    """The existing tiny midtraining configuration, plus a functional arm on disk.

    The measurements file is deliberately NOT written: tests create it from the plan's budget,
    as a curator would.
    """
    path, config = mid_config(tmp_path, updates=updates)
    declaration, records, population, _ = inputs.load_broad_context(tmp_path / "pairs.json")
    document = small_document(declaration)
    if singleton:
        document["constructs"].append(construct_row("s", heavy(5)))
        document["groups"].append(group_row("g-s", "ctx-a", KD, ["s"]))
    if forbidden_antigen:                       # "HIK" is the protected endpoint antigen
        document["constructs"].append(construct_row("z", heavy(3), Ag="HIK"))
        document["groups"].append(group_row("g-z", "ctx-a", KD, ["z"], Ag="HIK"))
    write(tmp_path / "constructs.json", document)
    section = functional_section(**changes)
    if forbidden_antigen:                       # keep z out of the budget: only the ordered pair is selected
        for index in range(300):
            section["budget_seed"] = f"seed-{index}"
            spec = {**section, "constructs_path": str(tmp_path / "constructs.json")}
            _, budget, _ = inputs.qualify_functional_inputs(spec, declaration, records, population)
            if budget.selected_set == {("g-x", "a"), ("g-x", "b")}:
                break
        else:
            raise AssertionError("no budget seed leaves the forbidden construct unselected")
    config["functional"] = section
    write(path, config)
    return path, config


def write_measurements(tmp_path, plan, world="ordered"):
    budget = plan["functional"]["budget"]
    rows = [{"group_id": g, "construct_id": c, "measurement": SMALL_VALUES[world][(g, c)]}
            for g, c in budget["selected"]]
    return write(tmp_path / "measurements.json", {"schema": fp.MEASUREMENTS_SCHEMA,
                                                  "selection_digest": budget["selection_digest"],
                                                  "measurements": rows})


def forbidden(*args, **kwargs):
    raise AssertionError("opened a measurement")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_reports_the_qualification_budget_coverage_and_certificate_without_labels_or_models(
        tmp_path, monkeypatch):
    path, _ = cli_world(tmp_path)
    monkeypatch.setattr(cli, "models", lambda *a: pytest.fail("planning constructed models"))
    for owner, name in ((inputs, "load_functional_measurements"), (fp, "load_restricted_measurements"),
                        (fp.RestrictedFunctionalData, "from_document")):
        monkeypatch.setattr(owner, name, forbidden)
    assert not (tmp_path / "measurements.json").exists()
    output = tmp_path / "plan.json"
    run("plan", "--config", path, "--output", output)
    plan = read_json(output)
    assert plan["schema"] == "trastuzumab-input-plan/2" and plan["training"]["total_updates"] == 3
    functional = plan["functional"]
    assert functional["qualified"]["by_partition"]["train"] == {"constructs": 2, "groups": 1, "observations": 2}
    assert functional["budget"]["budget_n"] == 2 and len(functional["budget"]["selected"]) == 2
    assert functional["exposure"]["schedule"]["compared_fraction"]["value"] == 1.0
    assert functional["exposure"]["schedule"]["pair_opportunities"] == 6
    assert functional["coverage_requirement_met"] is True
    assert functional["input"] == {"functional_constructs_sha256": sha(tmp_path / "constructs.json")}
    assert plan["transfer_certificate"]["functional"]["qualified_digest"] == functional["qualified"]["digest"]
    assert plan["coverage"]["coverage"]["unique_pairs"] == 1        # the denoising population is untouched


def test_a_run_without_the_arm_plans_and_trains_with_explicit_nulls(tmp_path):
    path, _ = mid_config(tmp_path, updates=1)
    run("plan", "--config", path, "--output", tmp_path / "plan.json")
    assert read_json(tmp_path / "plan.json")["functional"] is None
    run("train", "--config", path, "--output", tmp_path / "run")
    identity = read_json(tmp_path / "run" / "identity.json")
    assert identity["functional"] is None and identity["input"]["functional"] is None
    assert read_json(tmp_path / "run" / "result.json")["progress"]["functional"] is None
    assert not (tmp_path / "run" / "functional-exposure-plan.json").exists()


def test_the_coverage_gate_precedes_measurements_and_models(tmp_path, monkeypatch):
    # The singleton group's only member can never be compared, so full coverage is unreachable.
    path, _ = cli_world(tmp_path, singleton=True, budget_n=3)
    monkeypatch.setattr(cli, "models", lambda *a: pytest.fail("constructed models"))
    run("plan", "--config", path, "--output", tmp_path / "plan.json")
    assert read_json(tmp_path / "plan.json")["functional"]["coverage_requirement_met"] is False
    with pytest.raises(ValueError, match="declared horizon compares only"):
        run("train", "--config", path, "--output", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists() and not (tmp_path / "measurements.json").exists()


def test_a_forbidden_unselected_functional_construct_stops_planning_and_training(tmp_path, monkeypatch):
    path, _ = cli_world(tmp_path, forbidden_antigen=True)
    monkeypatch.setattr(cli, "models", lambda *a: pytest.fail("constructed models"))
    for command in (("plan", "--config", path, "--output", tmp_path / "plan.json"),
                    ("train", "--config", path, "--output", tmp_path / "rejected")):
        with pytest.raises(ValueError, match="downstream antigen sequence appears in functional training: z"):
            run(*command)
    assert not (tmp_path / "plan.json").exists() and not (tmp_path / "rejected").exists()
    assert not (tmp_path / "measurements.json").exists()


# ---------------------------------------------------------------------------
# Training, resume, inference
# ---------------------------------------------------------------------------

def test_train_resume_profile_and_inference_without_the_functional_inputs(tmp_path):
    path, _ = cli_world(tmp_path)
    run("plan", "--config", path, "--output", tmp_path / "plan.json")
    plan = read_json(tmp_path / "plan.json")
    write_measurements(tmp_path, plan)

    run("profile", "--config", path, "--output", tmp_path / "profile.json",
        "--warmup-updates", 1, "--measured-updates", 1)
    profile = read_json(tmp_path / "profile.json")
    assert profile["metrics"][0]["functional"]["pair_opportunities"] == 2

    partial, resumed, full = (tmp_path / name for name in ("partial", "resumed", "full"))
    run("train", "--config", path, "--output", partial, "--stop-after-updates", 1, "--checkpoint-every", 1)
    stopped = read_json(partial / "result.json")
    assert stopped["status"] == "stopped_at_boundary"
    assert stopped["progress"]["functional"]["counters"]["pair_opportunities"] == 2
    run("train", "--config", path, "--output", resumed, "--resume", partial / "checkpoint.pt",
        "--resume-sha256", stopped["checkpoint"]["sha256"])
    run("train", "--config", path, "--output", full)
    a, b = read_json(resumed / "result.json"), read_json(full / "result.json")
    assert a["status"] == b["status"] == "complete" and a["progress"] == b["progress"]
    assert b["progress"]["functional"]["counters"]["ordered_updates"] == 3
    logs = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in logs] == [2, 3] and all("functional" in row for row in logs)

    # The exposure plan is stored with the run and is the plan that was reported.
    assert read_json(full / "functional-exposure-plan.json") == plan["functional"]["exposure"]
    identity = read_json(full / "identity.json")
    assert identity["schema"] == "trastuzumab-training/2"
    assert identity["functional"]["selection_digest"] == plan["functional"]["budget"]["selection_digest"]
    assert identity["input"]["functional"]["budget"] == plan["functional"]["budget"]
    assert identity["input"]["functional_constructs_sha256"] == sha(tmp_path / "constructs.json")
    assert identity["input"]["functional_measurements_sha256"] == sha(tmp_path / "measurements.json")
    first = torch.load(resumed / "checkpoint.pt", weights_only=True)
    second = torch.load(full / "checkpoint.pt", weights_only=True)
    assert first["identity"] == second["identity"]
    assert torch.equal(first["functional"]["weights"]["weight"], second["functional"]["weights"]["weight"])

    # Inference and export open the model and the checkpoint only.
    for name in ("constructs.json", "measurements.json", "pairs.json"):
        (tmp_path / name).unlink()
    queries = write(tmp_path / "queries.json", _panel_document())
    common = ("--config", path, "--checkpoint", full / "checkpoint.pt", "--checkpoint-sha256", b["checkpoint"]["sha256"])
    run("score", *common, "--queries", queries, "--arm", "paired", "--scorer-id", "tiny-functional",
        "--orders", 1, "--order-seed", 3, "--output", tmp_path / "scores")
    assert (tmp_path / "scores" / "scores.json").exists()
    run("export-parent", *common, "--output", tmp_path / "parent.pt")
    parent = torch.load(tmp_path / "parent.pt", weights_only=True)
    assert "functional" not in parent
    assert parent["provenance"]["training"]["functional"]["selection_digest"] == identity["functional"]["selection_digest"]
    assert parent["provenance"]["progress"]["functional"]["counters"]["ordered_updates"] == 3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PROBLEMS = {
    "old_schema": lambda c: c.update(schema="trastuzumab-run/1"),
    "missing_section": lambda c: c.pop("functional"),
    "missing_path": lambda c: c["functional"].pop("measurements_path"),
    "extra_key": lambda c: c["functional"].update(extra=1),
    "partial_settings": lambda c: c["functional"]["config"].pop("margin"),
    "unknown_setting": lambda c: c["functional"]["config"].update(bias=1),
    "invalid_setting": lambda c: c["functional"]["config"].update(weight=-1.0),
    "boolean_budget": lambda c: c["functional"].update(budget_n=True),
    "zero_budget": lambda c: c["functional"].update(budget_n=0),
    "empty_seed": lambda c: c["functional"].update(budget_seed=""),
    "empty_path": lambda c: c["functional"].update(constructs_path=""),
    "post_stage": lambda c: c.update(training=dict(stage="post", total_updates=1, examples_per_update=1,
                                                   ordering_weight=1.0)),
}


@pytest.mark.parametrize("problem", sorted(CONFIG_PROBLEMS))
def test_functional_configuration_is_all_or_nothing_and_never_post(tmp_path, problem):
    path, config = cli_world(tmp_path)
    CONFIG_PROBLEMS[problem](config)
    write(path, config)
    with pytest.raises((ValueError, TypeError)):
        cli.load_config(path)


def test_the_example_configurations_load_under_the_current_schema():
    directory = Path(__file__).resolve().parents[3] / "configs" / "trastuzumab_lowresource"
    names = sorted(p.name for p in directory.glob("*.json"))
    expected = {"mid_650m_lora.json", "mid_650m_full.json", "post_650m.json",
                "functional_shared_650m_lora.json", "functional_detached_650m_lora.json",
                "functional_shared_650m_full.json", "functional_detached_650m_full.json"}
    assert set(names) == expected
    for name in names:
        config = cli.load_config(directory / name)
        assert config["schema"] == "trastuzumab-run/2"
        assert (config["functional"] is None) == name.startswith(("mid_", "post_"))
        if config["functional"] is not None:
            mode = "shared" if "shared" in name else "detached"
            assert config["functional"]["config"]["gradient_mode"] == mode
            assert config["fusion"]["update_mode"] == ("full" if name.endswith("full.json") else "lora")
            assert Path(config["functional"]["constructs_path"]).is_absolute()

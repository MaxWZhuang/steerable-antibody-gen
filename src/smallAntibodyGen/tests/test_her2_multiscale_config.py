"""The pilot config, the five roots, the fresh run root and the command line.

All of it runs without torch: the config and roots modules are the part the CLI uses before
``prepare_environment``. Nothing here reads the user's disk or builds a model.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smallAntibodyGen.experiments.her2_multiscale import config as config_lib
from smallAntibodyGen.experiments.her2_multiscale import roots as roots_lib

REPO = Path(__file__).resolve().parents[3]
REAL_PILOT = REPO / "configs" / "experiments" / "her2_multiscale_pilot.json"
BASE_SHA256 = "28ba8b6b3505b89514d94dc77c08ea9013074a8dd5e48ed66f2b96d66876f7f7"
SCRIPT = REPO / "scripts" / "her2_multiscale_pilot.py"


def document():
    return json.loads(REAL_PILOT.read_text(encoding="utf-8"))


def pilot_with(tmp_path, mutate=None, *, text=None):
    """Write a (mutated) copy of the shipped pilot config beside a copy of its base config."""
    code = tmp_path / "code"
    target = code / "configs" / "experiments"
    target.mkdir(parents=True, exist_ok=True)
    base = REPO / "configs" / "experiments" / "her2_generative_routes_dplm_train.json"
    (target / base.name).write_bytes(base.read_bytes())
    value = document()
    if mutate is not None:
        mutate(value)
    path = target / "pilot.json"
    path.write_text(text if text is not None else json.dumps(value), encoding="utf-8")
    return path, code


def load(tmp_path, mutate=None, **kwargs):
    path, code = pilot_with(tmp_path, mutate, **kwargs)
    return config_lib.load_pilot_config(path, code_root=code)


# --- the shipped config -----------------------------------------------------------------------

def test_the_shipped_config_binds_the_shipped_base_config_by_digest():
    pilot = config_lib.load_pilot_config(REAL_PILOT, code_root=REPO)
    assert pilot.base_sha256 == BASE_SHA256
    assert config_lib.file_sha256(pilot.base_path) == BASE_SHA256
    assert pilot.route == "dplm_150m_masked" and pilot.arm == "combined"
    assert pilot.updates == 1000 and pilot.reservoir_capacity == 8192
    assert pilot.expected_counts == {"candidates": 3619, "development": 100, "reserved": 100,
                                     "excluded": 1273, "retained_for_training": 2146}
    assert 3619 - 100 - 100 - 1273 == 2146


def test_the_four_arms_differ_in_masks_and_distances_only():
    designs = {arm: config_lib.ARM_DESIGN[arm] for arm in config_lib.ARMS}
    assert designs == {"baseline": ("suffix", "d1"), "masks": ("subset", "d1"),
                       "pairs": ("suffix", "d123"), "combined": ("subset", "d123")}
    pilot = config_lib.load_pilot_config(REAL_PILOT, code_root=REPO)
    for arm, (geometry, scheme) in designs.items():
        arm_config = pilot.with_arm(arm)
        assert arm_config.geometry == geometry and arm_config.distance_scheme == scheme
        weights = arm_config.distance_weights
        if scheme == "d1":
            assert weights == {1: 1.0}
        else:
            assert sorted(weights) == [1, 2, 3] and sum(weights.values()) == pytest.approx(1.0)
            assert weights[1] > weights[2] > weights[3]  # the declared 0.5 / 0.3 / 0.2
        assert arm_config.document["arm"] == arm
    with pytest.raises(ValueError, match="arm must be one of"):
        pilot.with_arm("nope")


@pytest.mark.parametrize("route", ["dplm_650m_lora_masked", "ablang2_masked_editor", ""])
def test_any_other_route_is_refused(tmp_path, route):
    assert config_lib.ROUTE == "dplm_150m_masked"
    with pytest.raises(ValueError, match="native DPLM 150M route only"):
        load(tmp_path, lambda value: value.update(route=route))


# --- a closed schema, strict numbers -------------------------------------------------------

@pytest.mark.parametrize("extra", [{"routes": {}}, {"optimizer": {"name": "adamw"}},
                                   {"learning_rate": 0.1}, {"microbatch": 8}, {"weights": {}},
                                   {"resume": True}])
def test_a_field_the_schema_does_not_list_is_refused(tmp_path, extra):
    with pytest.raises(ValueError, match="unrecognised field"):
        load(tmp_path, lambda value: value.update(extra))


@pytest.mark.parametrize("block, key", [("local", "momentum"), ("diagnostics", "bonus"),
                                        ("inputs", "pins"), ("base", "note")])
def test_a_nested_unknown_field_is_refused(tmp_path, block, key):
    with pytest.raises(ValueError, match="unrecognised field"):
        load(tmp_path, lambda value: value[block].update({key: 1}))


@pytest.mark.parametrize("key", ["pilot_id", "base", "arm", "updates", "seed", "local", "inputs",
                                 "diagnostics", "route"])
def test_a_missing_field_is_refused(tmp_path, key):
    with pytest.raises(ValueError, match="missing field"):
        load(tmp_path, lambda value: value.pop(key))


@pytest.mark.parametrize("field, value", [
    ("updates", 0), ("updates", -3), ("updates", 1000.0), ("updates", True), ("updates", "1000"),
    ("updates", 20001), ("updates", None), ("seed", -1), ("seed", 1.5), ("seed", False),
    ("seed", 2 ** 31), ("arm", "full"), ("arm", None), ("pilot_id", " "), ("schema_version", "x")])
def test_top_level_numbers_and_names_are_strict(tmp_path, field, value):
    with pytest.raises(ValueError):
        load(tmp_path, lambda doc: doc.update({field: value}))


@pytest.mark.parametrize("change", [
    {"replacement_probability": 1.5}, {"replacement_probability": -0.1},
    {"replacement_probability": True}, {"replacement_probability": "0.5"},
    {"reservoir_capacity": 0}, {"reservoir_capacity": 65537}, {"reservoir_capacity": 8192.0},
    {"reservoir_capacity": True}, {"distance_weights": {"1": 1.0, "2": 1.0}},
    {"distance_weights": {"1": 1.0, "2": 1.0, "3": 0.0}},
    {"distance_weights": {"1": 1.0, "2": -1.0, "3": 1.0}},
    {"distance_weights": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}},
    {"distance_weights": {"1": 1.0, "2": "x", "3": 1.0}},
    {"distance_weights": {"1": 1.0, "2": True, "3": 1.0}}])
def test_local_settings_are_strict(tmp_path, change):
    with pytest.raises(ValueError):
        load(tmp_path, lambda doc: doc["local"].update(change))


@pytest.mark.parametrize("change", [
    {"batch_size": 0}, {"batch_size": 65}, {"completion_rows": 0}, {"completion_rows": 201},
    {"completion_counts": []}, {"completion_counts": [3, 1]}, {"completion_counts": [1, 1]},
    {"completion_counts": [0, 2]}, {"completion_counts": [1, 11]}, {"completion_counts": [1.0]},
    {"order_rows": 0}, {"fit_pairs_per_distance": 513}, {"fit_pairs_per_distance": True}])
def test_diagnostic_bounds_are_strict(tmp_path, change):
    with pytest.raises(ValueError):
        load(tmp_path, lambda doc: doc["diagnostics"].update(change))


@pytest.mark.parametrize("key, value", [("candidates", -1), ("development", 1.5),
                                        ("reserved", True), ("excluded", "1273")])
def test_expected_split_counts_are_strict(tmp_path, key, value):
    with pytest.raises(ValueError):
        load(tmp_path, lambda doc: doc["inputs"]["expected_counts"].update({key: value}))


def test_the_bin_policy_is_a_declared_closed_choice():
    pilot = config_lib.load_pilot_config(REAL_PILOT, code_root=REPO)
    assert pilot.bin_policy == "current_exact_training"
    assert config_lib.BIN_POLICIES == ("current_exact_training",)
    assert document()["inputs"]["bin_policy"] == pilot.bin_policy
    assert pilot.with_arm("masks").bin_policy == pilot.bin_policy


def test_a_pilot_without_a_bin_policy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing field"):
        load(tmp_path, lambda doc: doc["inputs"].pop("bin_policy"))


@pytest.mark.parametrize("value", ["historical_replay", "frozen_historical", "", "Current_exact",
                                   None, 1, ["current_exact_training"]])
def test_any_other_bin_policy_is_refused(tmp_path, value):
    with pytest.raises(ValueError, match="bin_policy must be one of"):
        load(tmp_path, lambda doc: doc["inputs"].update(bin_policy=value))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_json_number_is_refused_at_parse_time(tmp_path, constant):
    text = REAL_PILOT.read_text(encoding="utf-8").replace('"replacement_probability": 0.5',
                                                          f'"replacement_probability": {constant}')
    with pytest.raises(ValueError, match="not a valid JSON number"):
        load(tmp_path, text=text)


def test_a_repeated_key_is_refused(tmp_path):
    text = REAL_PILOT.read_text(encoding="utf-8").replace(
        '"updates": 1000,', '"updates": 1000, "updates": 5,')
    with pytest.raises(ValueError, match="duplicate key"):
        load(tmp_path, text=text)


def test_the_base_config_must_hash_to_the_pinned_digest(tmp_path):
    path, code = pilot_with(tmp_path)
    base = code / "configs" / "experiments" / "her2_generative_routes_dplm_train.json"
    base.write_bytes(base.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hashes to"):
        config_lib.load_pilot_config(path, code_root=code)


@pytest.mark.parametrize("value", ["../elsewhere.json", "/etc/base.json", "missing.json"])
def test_the_base_config_must_be_a_file_inside_the_code_root(tmp_path, value):
    with pytest.raises(ValueError):
        load(tmp_path, lambda doc: doc["base"].update(config=value))


@pytest.mark.parametrize("value", ["abc", BASE_SHA256.upper(), BASE_SHA256[:-1], 5])
def test_a_malformed_digest_is_refused(tmp_path, value):
    with pytest.raises(ValueError, match="SHA-256"):
        load(tmp_path, lambda doc: doc["base"].update(sha256=value))


# --- the roots, the context, the fresh run root ---------------------------------------------

@pytest.fixture
def layout(tmp_path):
    names = ("code", "data", "artifacts", "frozen")
    for name in names:
        (tmp_path / name).mkdir()
    (tmp_path / "runs").mkdir()
    return tmp_path


BASE = REPO / "configs" / "experiments" / "her2_generative_routes_dplm_train.json"


def base_config():
    return json.loads(BASE.read_text(encoding="utf-8"))


def reserve(layout, run_root=None, **overrides):
    arguments = dict(code_root=layout / "code", data_repo_root=layout / "data",
                     artifact_root=layout / "artifacts", frozen_inputs=layout / "frozen")
    arguments.update(overrides)
    if "declared_inputs" not in arguments:
        arguments["declared_inputs"] = roots_lib.declared_input_roots(
            base_config(), data_repo_root=arguments["data_repo_root"])
    return roots_lib.reserve_run_root(run_root or layout / "runs" / "one", **arguments)


def test_reserving_creates_exactly_the_run_root_and_nothing_else(layout):
    roots = reserve(layout)
    assert roots.run_root == layout / "runs" / "one" and roots.run_root.is_dir()
    assert list(roots.run_root.iterdir()) == []
    assert roots.source_cache == roots.run_root / "source_cache"
    assert roots.runtime_root == roots.run_root / "runtime"
    assert not roots.source_cache.exists() and not roots.runtime_root.exists()
    assert sorted(path.name for path in (layout / "runs").iterdir()) == ["one"]
    for name in ("code", "data", "artifacts", "frozen"):
        assert list((layout / name).iterdir()) == []


def test_an_existing_run_root_is_refused_even_when_empty(layout):
    (layout / "runs" / "occupied").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        reserve(layout, layout / "runs" / "occupied")
    (layout / "runs" / "file").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        reserve(layout, layout / "runs" / "file")
    reserve(layout)
    with pytest.raises(ValueError, match="already exists"):
        reserve(layout)


def test_a_relative_run_root_or_a_missing_parent_is_refused(layout):
    with pytest.raises(ValueError, match="absolute"):
        reserve(layout, Path("relative") / "run")
    with pytest.raises(ValueError, match="does not exist"):
        reserve(layout, layout / "no" / "such" / "run")
    assert not (layout / "no").exists()


@pytest.mark.parametrize("name", ["code", "data", "artifacts", "frozen"])
def test_the_run_root_is_disjoint_from_every_input_root(layout, name):
    with pytest.raises(ValueError, match="overlaps protected"):
        reserve(layout, layout / name / "run")
    assert list((layout / name).iterdir()) == []


def test_input_roots_must_exist_and_be_absolute(layout):
    for key in ("code_root", "data_repo_root", "artifact_root", "frozen_inputs"):
        with pytest.raises(ValueError, match="existing directory"):
            reserve(layout, **{key: layout / "absent"})
        with pytest.raises(ValueError, match="absolute"):
            reserve(layout, **{key: Path("relative")})
    assert not (layout / "runs" / "one").exists()


def test_the_context_resolves_each_kind_of_path_under_its_own_root(layout):
    roots = reserve(layout)
    config = {"dplm": {"pins": "configs/experiments/dplm_pins.json"},
              "conditions": {"assays": ["a"], "bin_dropout": 0.1, "assay_dropout": 0.2}}
    context = roots_lib.PilotContext(config=config, config_path=layout / "code" / "base.json",
                                     roots=roots, revision="r", campaign_id="c")
    assert context.path("x.json") == roots.run_root / "x.json"
    assert context.artifact("models/m") == layout / "artifacts" / "models" / "m"
    assert context.repository("data/raw/her2") == layout / "data" / "data" / "raw" / "her2"
    assert context.repository("configs/experiments/dplm_pins.json") == (
        layout / "code" / "configs" / "experiments" / "dplm_pins.json")
    assert context.repository_root == layout / "data"
    assert context.artifact_root == layout / "artifacts" and context.run_root == roots.run_root
    assert context.condition_spec().assays == ("a",)
    for escape in ("../outside", "a/../../outside"):
        with pytest.raises(ValueError, match="traversing"):
            context.artifact(escape)
        with pytest.raises(ValueError, match="traversing"):
            context.repository(escape)


# --- declared historical and data inputs ------------------------------------------------------

def make_alias(link, target):
    """A directory symlink, or on Windows a junction, at ``link`` pointing at ``target``."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
    except (OSError, NotImplementedError):
        if os.name != "nt":
            pytest.skip("cannot create a directory symlink here")
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              capture_output=True, text=True)
        if made.returncode != 0:
            pytest.skip(f"cannot create a junction here: {made.stderr.strip()}")


def digest_tree(root):
    return {str(path.relative_to(root)): path.stat().st_size
            for path in sorted(Path(root).rglob("*"))}


@pytest.fixture
def aliased(layout):
    """The data repository's historical flight directories as junctions OUTSIDE every root.

    This is the development machine's shape: ``outputs/her2_next_flight_20260922/production_03`` and
    ``outputs/her2_parent_replay_20260920`` resolve to other directories on the same volume.
    """
    elsewhere = layout / "elsewhere"
    targets = {"next_flight": elsewhere / "flight", "parent_replay": elsewhere / "replay"}
    for target in targets.values():
        target.mkdir(parents=True)
        (target / "evidence.json").write_text("{}", encoding="utf-8")
    make_alias(layout / "data" / "outputs" / "her2_next_flight_20260922" / "production_03",
               targets["next_flight"])
    make_alias(layout / "data" / "outputs" / "her2_parent_replay_20260920",
               targets["parent_replay"])
    return targets


def test_the_declared_inputs_are_every_data_and_historical_location_of_the_base_config(layout):
    declared = roots_lib.declared_input_roots(base_config(), data_repo_root=layout / "data")
    base = layout / "data"
    assert declared["historical next_flight"] == (
        base / "outputs" / "her2_next_flight_20260922" / "production_03")
    assert declared["historical parent_replay"] == base / "outputs" / "her2_parent_replay_20260920"
    assert declared["historical posttrain"] == base / "outputs" / "her2_posttrain_20260918"
    for name in ("her2_raw", "paired_oas", "cr9114_raw"):
        assert f"data root {name}" in declared
    assert declared["data root her2_raw"] == base / "data" / "raw" / "her2_functional_20260918"
    assert not any("note" in name for name in declared)
    repository_sources = {name for name, entry in base_config()["sources"]["paths"].items()
                          if entry.get("root") == "repository"}
    assert repository_sources == {"cr9114", "cr9114_prepared", "cr9114_whitelist"}
    for name in repository_sources:
        assert f"data source {name}[0]" in declared
    assert all(location.is_relative_to(base) for location in declared.values())


@pytest.mark.parametrize("roots", [None, [], {"her2_raw": 3}, {"her2_raw": ["a"]},
                                   {"historical": ["a"]}, {"historical": {"x": 7}}])
def test_a_base_config_with_an_unreadable_roots_block_is_refused(layout, roots):
    base = {"roots": roots} if roots is not None else {}
    with pytest.raises(ValueError):
        roots_lib.declared_input_roots(base, data_repo_root=layout / "data")


def test_a_run_root_resolving_into_a_historical_flight_is_refused_before_mkdir(layout, aliased):
    """Lexically the run root is under none of the four roots; resolved, it is inside a flight."""
    before = digest_tree(layout / "elsewhere")
    for target in aliased.values():
        with pytest.raises(ValueError, match="overlaps protected declared inputs"):
            reserve(layout, target / "fresh_child")
        assert not (target / "fresh_child").exists()
    assert digest_tree(layout / "elsewhere") == before
    assert list((layout / "runs").iterdir()) == []


def test_without_the_declared_inputs_such_a_run_root_would_have_been_accepted(layout, aliased):
    """The reason for the protection: the four roots alone cannot see the junction's target."""
    roots = reserve(layout, aliased["next_flight"] / "fresh_child", declared_inputs={})
    assert roots.run_root.is_dir()


def test_the_alias_path_itself_is_refused_too(layout, aliased):
    alias = layout / "data" / "outputs" / "her2_next_flight_20260922" / "production_03"
    with pytest.raises(ValueError, match="overlaps protected"):
        reserve(layout, alias / "fresh_child")
    assert not (aliased["next_flight"] / "fresh_child").exists()


def test_a_legal_destination_outside_every_input_stays_valid(layout, aliased):
    destination = layout / "runs" / "legal"
    before = digest_tree(layout / "elsewhere")
    roots = reserve(layout, destination)
    assert roots.run_root == destination and destination.is_dir()
    assert digest_tree(layout / "elsewhere") == before


def test_a_run_root_that_would_contain_or_equal_a_declared_input_is_refused(layout):
    fresh = layout / "runs" / "fresh"
    inside = {"historical x": fresh / "flight" / "production"}
    with pytest.raises(ValueError, match="overlaps protected declared inputs"):
        reserve(layout, fresh, declared_inputs=inside)
    with pytest.raises(ValueError, match="overlaps protected declared inputs"):
        reserve(layout, fresh, declared_inputs={"historical x": fresh})
    assert not fresh.exists()


def test_a_declared_input_that_does_not_exist_yet_is_still_protected(layout):
    missing = layout / "historical-not-yet-there"
    assert not missing.exists()
    with pytest.raises(ValueError, match="overlaps protected declared inputs"):
        reserve(layout, missing, declared_inputs={"historical missing": missing})
    assert not missing.exists()
    parent = layout / "runs" / "later"
    with pytest.raises(ValueError, match="does not exist"):
        reserve(layout, parent / "child", declared_inputs={"historical missing": parent})


def test_the_declared_inputs_are_required_and_must_be_a_mapping(layout):
    with pytest.raises(TypeError):
        roots_lib.reserve_run_root(layout / "runs" / "one", code_root=layout / "code",
                                   data_repo_root=layout / "data",
                                   artifact_root=layout / "artifacts",
                                   frozen_inputs=layout / "frozen")
    for value in (None, [], [layout / "elsewhere"]):
        with pytest.raises(ValueError, match="declared_inputs must be"):
            reserve(layout, declared_inputs=value)
    assert not (layout / "runs" / "one").exists()


# --- the command line ------------------------------------------------------------------------

def load_script():
    spec = importlib.util.spec_from_file_location("her2_multiscale_pilot_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_creates_nothing_and_imports_no_scientific_module(tmp_path):
    completed = subprocess.run([sys.executable, "-I", str(SCRIPT), "--help"], cwd=tmp_path,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    for flag in ("--frozen-inputs", "--data-repo-root", "--run-root", "--artifact-root", "--arm"):
        assert flag in completed.stdout
    assert list(tmp_path.iterdir()) == []
    probe = (
        "import runpy, sys\n"
        f"sys.argv = [{str(SCRIPT)!r}, '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(sorted(name for name in ('torch', 'transformers', 'pandas') if name in sys.modules))")
    modules = subprocess.run([sys.executable, "-I", "-c", probe], cwd=tmp_path,
                             capture_output=True, text=True, timeout=120)
    assert modules.returncode == 0, modules.stderr
    assert modules.stdout.strip().splitlines()[-1] == "[]"
    assert list(tmp_path.iterdir()) == []


def arguments(layout, *extra, run_root=None):
    return ["--frozen-inputs", str(layout / "frozen"), "--data-repo-root", str(layout / "data"),
            "--artifact-root", str(layout / "artifacts"), "--run-root",
            str(run_root or layout / "runs" / "cli"), *extra]


def test_an_occupied_output_is_refused_before_the_environment_or_a_model(layout, monkeypatch):
    script = load_script()
    from smallAntibodyGen.experiments import her2_routes_env as env

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare_environment ran for a refused run root")
    monkeypatch.setattr(env, "prepare_environment", forbidden)
    occupied = layout / "runs" / "occupied"
    occupied.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        script.main(arguments(layout, run_root=occupied))
    assert list(occupied.iterdir()) == []


def test_a_run_root_inside_a_historical_flight_is_refused_before_the_environment(
        layout, aliased, monkeypatch):
    """The command line reads the base config's declared roots and protects their resolved targets."""
    script = load_script()
    from smallAntibodyGen.experiments import her2_routes_env as env

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare_environment ran for a refused run root")
    monkeypatch.setattr(env, "prepare_environment", forbidden)
    before = digest_tree(layout / "elsewhere")
    for target in aliased.values():
        with pytest.raises(ValueError, match="overlaps protected declared inputs"):
            script.main(arguments(layout, run_root=target / "child"))
        assert not (target / "child").exists()
    assert digest_tree(layout / "elsewhere") == before


def test_the_run_root_is_reserved_before_the_environment_is_prepared(layout, monkeypatch):
    script = load_script()
    from smallAntibodyGen.experiments import her2_routes_env as env
    seen = {}

    class Stop(Exception):
        pass

    def prepare(runtime_root, *, dependency_paths, offline):
        seen.update(runtime_root=Path(runtime_root), dependency_paths=list(dependency_paths),
                    offline=offline, reserved=Path(runtime_root).parent.is_dir())
        raise Stop
    monkeypatch.setattr(env, "prepare_environment", prepare)
    with pytest.raises(Stop):
        script.main(arguments(layout, "--arm", "masks"))
    run_root = layout / "runs" / "cli"
    assert seen["reserved"] is True and seen["runtime_root"] == run_root / "runtime"
    assert seen["dependency_paths"] == [layout / "artifacts" / "python_deps",
                                        layout / "artifacts" / "upstream" / "ablang2"]
    assert seen["offline"] is True


def test_a_second_launch_of_the_same_run_root_is_refused(layout, monkeypatch):
    script = load_script()
    from smallAntibodyGen.experiments import her2_routes_env as env

    class Stop(Exception):
        pass

    def prepare(*args, **kwargs):
        raise Stop
    monkeypatch.setattr(env, "prepare_environment", prepare)
    with pytest.raises(Stop):
        script.main(arguments(layout))
    with pytest.raises(ValueError, match="already exists"):
        script.main(arguments(layout))


def test_the_script_sets_the_bytecode_switch_before_any_project_import():
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    switch = next(node.lineno for node in tree.body if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == "sys.dont_write_bytecode")
    first_import = min(node.lineno for node in ast.walk(tree)
                       if isinstance(node, (ast.Import, ast.ImportFrom))
                       and "smallAntibodyGen" in ast.unparse(node))
    assert switch < first_import

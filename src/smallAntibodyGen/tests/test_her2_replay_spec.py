"""The computed source closure, the config, the capacity floor and the frozen fields.

The failure these guard against is a freeze that *passes* while something that
changes the computation is outside it: a lazily imported module, a package
``__init__``, an uncommitted new file, or a CLI flag that quietly overrides a
frozen setting at fit time.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_replay_spec as spec


REPOSITORY = Path(spec.__file__).resolve().parents[3]


def write_module(root, relative, text):
    """Write a source file with the bytes the caller asked for.

    ``newline="\\n"`` is not decoration: ``write_text`` without it translates ``\\n``
    to the platform default, so on Windows a fixture meant to hold LF would hold
    CRLF and an LF test would be testing nothing.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
    return relative


@pytest.fixture
def tiny_repo(tmp_path):
    """A miniature repository with a script, a package, lazy and relative imports."""
    write_module(tmp_path, "src/pkg/__init__.py", "")
    write_module(tmp_path, "src/pkg/sub/__init__.py", "")
    write_module(tmp_path, "src/pkg/leaf.py", "VALUE = 1\n")
    write_module(tmp_path, "src/pkg/sub/deep.py",
                 "from ..leaf import VALUE\n\n\ndef go():\n    import numpy\n    return numpy\n")
    write_module(tmp_path, "src/pkg/entry.py",
                 "from . import leaf\n\n\ndef later():\n"
                 "    from .sub import deep\n    return deep\n")
    write_module(tmp_path, "scripts/run.py",
                 "import json\nfrom pkg import entry\n\n\ndef main():\n"
                 "    import torch\n    return entry, torch, json\n")
    return tmp_path


# ---------------------------------------------------------------------------
# the computed closure
# ---------------------------------------------------------------------------

def test_the_closure_follows_relative_and_lazy_imports_and_includes_package_inits(tiny_repo):
    closure = spec.source_closure(tiny_repo, ["scripts/run.py"])
    assert set(closure["files"]) == {
        "scripts/run.py", "src/pkg/__init__.py", "src/pkg/entry.py", "src/pkg/leaf.py",
        "src/pkg/sub/__init__.py", "src/pkg/sub/deep.py"}
    assert closure["modules"]["src/pkg/sub/deep.py"] == "pkg.sub.deep"
    assert closure["modules"]["scripts/run.py"] is None


def test_a_function_level_import_is_in_the_closure_a_sys_modules_walk_would_miss(tiny_repo):
    """``from .sub import deep`` only happens when ``later()`` runs. The syntax says it can."""
    closure = spec.source_closure(tiny_repo, ["src/pkg/entry.py"])
    assert "src/pkg/sub/deep.py" in closure["files"]


def test_external_dependencies_are_recorded_and_the_virtualenv_is_not_walked(tiny_repo):
    closure = spec.source_closure(tiny_repo, ["scripts/run.py"])
    assert "torch" in " ".join(closure["external"])
    assert "json" in " ".join(closure["external"])
    versions = closure["external_versions"]
    assert versions["json"]["kind"] == "standard_library"
    assert versions["numpy"]["version"] is not None
    assert "virtualenv tree is never read" in closure["rule"]


def test_a_relative_import_inside_a_package_initializer_is_in_the_closure(tiny_repo):
    """``from .leaf import VALUE`` in ``pkg/__init__.py`` executes ``pkg.leaf``.

    The importer here *is* the package, so level 1 resolves to ``pkg`` and not to
    its parent. Resolving it by the ordinary module rule strips the package's own
    last component and yields the name ``leaf``, which matches nothing -- and the
    file the initializer runs on every import is then missing from the identity the
    freeze binds.
    """
    write_module(tiny_repo, "src/pkg/__init__.py", "from .leaf import VALUE\n")
    write_module(tiny_repo, "scripts/only_init.py", "import pkg\n")
    closure = spec.source_closure(tiny_repo, ["scripts/only_init.py"])
    assert set(closure["files"]) == {"scripts/only_init.py", "src/pkg/__init__.py",
                                     "src/pkg/leaf.py"}
    assert "leaf" not in closure["external"], (
        "a relative import must not be recorded as an external top-level dependency")


def test_a_deeper_relative_import_inside_a_package_initializer_resolves_upwards(tiny_repo):
    write_module(tiny_repo, "src/pkg/sub/__init__.py", "from ..leaf import VALUE\n")
    write_module(tiny_repo, "scripts/only_sub.py", "import pkg.sub\n")
    closure = spec.source_closure(tiny_repo, ["scripts/only_sub.py"])
    assert "src/pkg/leaf.py" in closure["files"]
    assert "src/pkg/sub/__init__.py" in closure["files"]


def test_the_current_initializers_contain_no_relative_imports_so_nothing_was_omitted():
    """The fix is a contract correction, not a repair of a demonstrated omission."""
    import ast
    for relative in spec.LEGACY_RAW_INITIALIZERS:
        tree = ast.parse((REPOSITORY / relative).read_text(encoding="utf-8"), filename=relative)
        assert not [node for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.level], (
            f"{relative} has gained a relative import; the closure now depends on the fix rather "
            "than merely being correct in advance of one")


def test_a_missing_closure_file_is_named(tiny_repo):
    with pytest.raises(ValueError, match="missing from the working tree"):
        spec.source_closure(tiny_repo, ["scripts/absent.py"])


def test_module_names_and_files_resolve_both_ways(tiny_repo):
    assert spec.module_name_for(tiny_repo, "src/pkg/leaf.py") == "pkg.leaf"
    assert spec.module_name_for(tiny_repo, "src/pkg/__init__.py") == "pkg"
    assert spec.module_relative(tiny_repo, "pkg.leaf") == "src/pkg/leaf.py"
    assert spec.module_relative(tiny_repo, "pkg") == "src/pkg/__init__.py"
    assert spec.module_relative(tiny_repo, "numpy") is None
    assert spec.package_initializers(tiny_repo, "pkg.sub.deep") == [
        "src/pkg/__init__.py", "src/pkg/sub/__init__.py"]


def test_the_real_entry_point_closure_covers_the_replay_modules_and_the_inherited_ones():
    closure = spec.source_closure(REPOSITORY, ["scripts/posttrain_her2_replay.py"])
    for expected in ("scripts/posttrain_her2_replay.py",
                     "src/smallAntibodyGen/experiments/her2_replay.py",
                     "src/smallAntibodyGen/experiments/her2_replay_campaign.py",
                     "src/smallAntibodyGen/experiments/her2_replay_streams.py",
                     "src/smallAntibodyGen/experiments/her2_policy.py",
                     "src/smallAntibodyGen/experiments/her2_preferences.py",
                     "src/smallAntibodyGen/experiments/her2_guard.py",
                     "src/smallAntibodyGen/experiments/her2_eval.py",
                     "src/smallAntibodyGen/experiments/__init__.py"):
        assert expected in closure["files"], f"{expected} is not in the computed closure"


def test_every_new_lf_pinned_file_is_in_the_real_closure():
    """The narrow .gitattributes rules and the closure must describe the same files."""
    closure = set(spec.source_closure(REPOSITORY, ["scripts/posttrain_her2_replay.py"])["files"])
    pinned = set(spec.REPLAY_LF_PINNED) - {"configs/experiments/her2_parent_replay.json",
                                           "scripts/rebuild_her2_support_report.py"}
    assert pinned <= closure
    attributes = (REPOSITORY / ".gitattributes").read_text(encoding="utf-8")
    for logical in spec.REPLAY_LF_PINNED:
        assert f"{logical} text eol=lf" in attributes, (
            f"{logical} is hashed by the freeze and has no narrow LF rule")


def test_the_runtime_closure_refuses_an_imported_module_that_was_not_frozen(tmp_path):
    closure = {"files": ["src/smallAntibodyGen/experiments/her2_replay.py"]}
    with pytest.raises(ValueError, match="not in the frozen source closure"):
        spec.require_runtime_closure(closure, repository_root=REPOSITORY, label="fit")


def test_the_runtime_closure_accepts_when_nothing_outside_it_is_imported(tmp_path):
    """Checked against a root no imported module lives under: the real fit process
    imports only its own entry point, while a pytest process has imported the whole
    repository and is deliberately not the population this guard is about."""
    closure = {"files": ["src/pkg/leaf.py"]}
    assert spec.require_runtime_closure(closure, repository_root=tmp_path) == ["src/pkg/leaf.py"]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_the_shipped_config_loads_and_carries_no_placeholder():
    document, digest, canonical = spec.load_config(
        REPOSITORY / "configs/experiments/her2_parent_replay.json")
    assert document["schema_version"] == spec.REPLAY_SCHEMA
    assert len(digest) == 64 and len(canonical) == 64
    assert document["screen"]["trajectories"] == 36
    assert document["screen"]["endpoint_updates"] == [1000, 2000, 3750]
    assert document["screen"]["endpoint_chosen_exposures"] == [64000, 128000, 240000]
    assert document["screen"]["chosen_per_update"] == 64


def test_the_shipped_config_declares_exposures_that_match_its_endpoints():
    document, _, _ = spec.load_config(REPOSITORY / "configs/experiments/her2_parent_replay.json")
    screen = document["screen"]
    assert [u * screen["chosen_per_update"] for u in screen["endpoint_updates"]] == \
        screen["endpoint_chosen_exposures"]
    assert (len(screen["tasks"]) * len(screen["replay_lambdas"])
            * len(screen["parent_seeds"])) == screen["trajectories"]
    assert (screen["trajectories"] * max(screen["endpoint_updates"])
            * screen["chosen_per_update"]) == screen["max_total_chosen_exposures"]


def test_a_placeholder_in_a_scientific_field_is_refused(tmp_path):
    document = json.loads((REPOSITORY / "configs/experiments/her2_parent_replay.json").read_text(
        encoding="utf-8"))
    document["optimization"]["schedule"] = "TODO"
    target = tmp_path / "config.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="launch-time placeholders"):
        spec.load_config(target)


def test_a_config_with_the_wrong_schema_is_refused(tmp_path):
    target = tmp_path / "config.json"
    target.write_text(json.dumps({"schema_version": "her2-support-audit/1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a her2-parent-replay/1 config"):
        spec.load_config(target)


def test_the_config_binds_the_audit_it_is_conditional_on():
    document, _, _ = spec.load_config(REPOSITORY / "configs/experiments/her2_parent_replay.json")
    audit = document["audit"]
    assert audit["required_outcome"] == "escalate"
    assert audit["source_freeze_commit"].startswith("be302f8")
    assert "ipo_tau0p1 crossed the tenfold criterion" in audit["escalation_detail"]
    assert "did not cross the hundredfold criterion" in audit["escalation_detail"]


def test_the_config_records_separate_gradient_and_score_tolerances():
    document, _, _ = spec.load_config(REPOSITORY / "configs/experiments/her2_parent_replay.json")
    tolerances = document["tolerances"]
    assert tolerances["sum_log_probability_atol"] == 5e-05
    assert tolerances["native_gradient_atol"] == 1e-4
    assert tolerances["post_adamw_parameter_atol"] == 3e-6
    assert tolerances["native_gradient_atol"] != tolerances["sum_log_probability_atol"]
    assert "independent of the sum-log-probability score tolerance" in \
        tolerances["native_gradient_justification"]


# ---------------------------------------------------------------------------
# frozen fields and capacity
# ---------------------------------------------------------------------------

def marker(**fields):
    return dict({"immutable_fields": ["device", "microbatch_rows"],
                 "screen": {"microbatch_rows": 16},
                 "config": {"resolved": {"inference": {"device": "cuda"}}}}, **fields)


def test_a_frozen_field_cannot_be_overridden_at_fit_time():
    assert spec.require_immutable_fields(marker(), {"device": "cuda"})
    assert spec.require_immutable_fields(marker(), {"device": None})
    with pytest.raises(ValueError, match="cannot be overridden"):
        spec.require_immutable_fields(marker(), {"device": "cpu"})
    with pytest.raises(ValueError, match="cannot be overridden"):
        spec.require_immutable_fields(marker(), {"microbatch_rows": 32})


def test_capacity_is_recorded_whether_or_not_it_passes(tmp_path):
    record = spec.capacity_record(tmp_path, minimum_bytes=2 ** 70, required=False)
    assert record["sufficient"] is False and record["free_bytes"] > 0
    assert record["required_free_gib"] > 0
    with pytest.raises(ValueError, match="GiB free and this campaign declares"):
        spec.capacity_record(tmp_path, minimum_bytes=2 ** 70, required=True)
    assert spec.capacity_record(tmp_path, minimum_bytes=1)["sufficient"] is True


def test_the_declared_minimum_capacity_is_the_frozen_one_not_todays_free_space():
    document, _, _ = spec.load_config(REPOSITORY / "configs/experiments/her2_parent_replay.json")
    assert document["storage"]["min_free_bytes"] == 20 * 1024 ** 3


# ---------------------------------------------------------------------------
# newline pinning
# ---------------------------------------------------------------------------

def test_crlf_is_refused_only_for_the_files_this_screen_introduces(tmp_path):
    new = write_module(tmp_path, "new.py", "x = 1\r\n")
    legacy = write_module(tmp_path, "legacy.py", "y = 2\r\n")
    report = spec.newline_report(tmp_path, [legacy], pinned=(new,))
    assert report["newlines"][legacy] == "crlf"
    with pytest.raises(ValueError, match="carry CRLF bytes"):
        spec.newline_report(tmp_path, [new, legacy], pinned=(new,))


def test_lf_files_are_recorded_as_lf(tmp_path):
    clean = write_module(tmp_path, "clean.py", "x = 1\n")
    assert (tmp_path / clean).read_bytes() == b"x = 1\n", (
        "the fixture itself must hold LF; a platform-translated write would make this test pass "
        "on a file that is not LF")
    report = spec.newline_report(tmp_path, [clean], pinned=(clean,))
    assert report["newlines"][clean] == "lf"
    assert clean in report["lf_pinned"]


def test_the_three_crlf_package_initializers_keep_their_bytes_and_are_declared(tmp_path):
    """They are in the closure, they hold CRLF, and the rule preserves those bytes."""
    attributes = (REPOSITORY / ".gitattributes").read_text(encoding="utf-8")
    for relative in spec.LEGACY_RAW_INITIALIZERS:
        assert f"{relative} -text whitespace=cr-at-eol" in attributes, (
            "the narrow rule that makes Git store these exact bytes is missing")
        assert relative not in spec.REPLAY_LF_PINNED, (
            "a file this screen did not introduce is never held to LF by it")
    report = spec.newline_report(REPOSITORY, list(spec.LEGACY_RAW_INITIALIZERS),
                                 pinned=spec.LEGACY_RAW_INITIALIZERS)
    assert set(report["newlines"]) == set(spec.LEGACY_RAW_INITIALIZERS), (
        "a declared legacy initializer is refused rather than reported")


# ---------------------------------------------------------------------------
# the audit this screen is conditional on
# ---------------------------------------------------------------------------

def audit_fixture(tmp_path, *, immutable=True, tamper=None):
    """A miniature published audit tree: manifest, decision, verification."""
    root = tmp_path
    published = root / "reference/evidence/audit/published"
    published.mkdir(parents=True)
    documents = {
        "decision.json": {"outcome": "escalate", "methods": {"ipo": {"outcome": "escalate"}}},
        "verification.json": {"immutable": immutable, "problems": [] if immutable else ["x"],
                              "shards_checked": 216},
        "coverage.json": {"rows": 3}}
    from smallAntibodyGen.experiments import her2_support_paths as paths
    files = {}
    for name, document in documents.items():
        paths.write_json(published / name, document)
        files[f"evidence/audit/published/{name}"] = paths.sha256_file(published / name)
    if tamper:
        (published / tamper).write_text("{}", encoding="utf-8")
    paths.write_json(published / "manifest.json",
                     {"record_kind": "publication_manifest", "root": "reference", "files": files})
    config = {"audit": {
        "publication_manifest": "reference/evidence/audit/published/manifest.json",
        "published": {name.split(".")[0]: f"reference/evidence/audit/published/{name}"
                      for name in documents},
        "required_outcome": "escalate", "source_freeze_commit": "be302f8"}}
    return root, config


def test_every_consulted_published_file_is_bound_to_the_publication_manifest(tmp_path):
    root, config = audit_fixture(tmp_path)
    block = spec.published_audit(root, config)
    assert block["decision_outcome"] == "escalate"
    assert block["published_verification"]["immutable"] is True
    assert block["publication_manifest_binding"]["file_count"] == 3
    for entry in block["published"].values():
        assert entry["sha256"] == entry["manifest_sha256"]


def test_a_published_file_that_does_not_match_its_manifest_is_refused(tmp_path):
    root, config = audit_fixture(tmp_path, tamper="coverage.json")
    with pytest.raises(ValueError, match="not bound to their publication manifest"):
        spec.published_audit(root, config)


def test_an_audit_whose_own_verification_failed_is_not_a_precondition(tmp_path):
    root, config = audit_fixture(tmp_path, immutable=False)
    with pytest.raises(ValueError, match="does not record immutable=true"):
        spec.published_audit(root, config)


def test_the_claimed_audit_freeze_is_tied_to_the_completion_marker(tmp_path):
    root, config = audit_fixture(tmp_path)
    audit = spec.published_audit(root, config)
    completion = {"record_kind": "audit_complete", "frozen_commit": "be302f8",
                  "decision_outcome": "escalate", "completed_at": "2026-09-19T12:00:00+00:00",
                  "verification": {"immutable": True, "shards_checked": 216},
                  "stages": {"report": "completed"}}
    assert spec.require_audit_completion(config, completion, audit=audit)["shards_checked"] == 216
    with pytest.raises(ValueError, match="is not the one the completed audit ran under"):
        spec.require_audit_completion(config, dict(completion, frozen_commit="deadbeef"),
                                      audit=audit)
    with pytest.raises(ValueError, match="conditional on"):
        spec.require_audit_completion(config, dict(completion, decision_outcome="continue"),
                                      audit=audit)
    with pytest.raises(ValueError, match="Two different completion claims"):
        spec.require_audit_completion(
            config, dict(completion, verification={"immutable": True, "shards_checked": 4}),
            audit=audit)


# ---------------------------------------------------------------------------
# the runtime contract
# ---------------------------------------------------------------------------

def runtime_marker(**fields):
    from smallAntibodyGen.experiments import her2_support as support
    environment = support.environment_record()
    return dict({"config": {"resolved": {"inference": {"device": "cuda",
                                                       "forward_dtype": "float32",
                                                       "reduction_dtype": "float64"}}},
                 "environment": environment,
                 "source": {"closure": {"external_versions": {}}}}, **fields)


def test_the_runtime_contract_refuses_a_device_the_freeze_did_not_declare():
    assert spec.require_runtime_contract(runtime_marker(), device="cuda")["device"] == "cuda"
    with pytest.raises(ValueError, match="does not match the frozen contract"):
        spec.require_runtime_contract(runtime_marker(), device="cpu")


def test_the_runtime_contract_refuses_a_changed_dependency_version():
    marker_document = runtime_marker()
    marker_document["environment"] = dict(marker_document["environment"], torch_version="0.0.1")
    with pytest.raises(ValueError, match="torch_version"):
        spec.require_runtime_contract(marker_document, device="cuda")


def test_production_never_bypasses_the_preflight():
    """The freeze stage takes the default, and the default refuses a missing preflight."""
    import inspect
    assert inspect.signature(spec.run_freeze).parameters[
        "allow_missing_preflight"].default is False
    script = (REPOSITORY / "scripts/posttrain_her2_replay.py").read_text(encoding="utf-8")
    body = script.split("def cmd_freeze")[1].split("\ndef ")[0]
    assert "allow_missing_preflight" not in body, (
        "the CLI freeze stage must take the mandatory path")


def test_the_config_accounting_for_the_teacher_caches_is_the_exact_product():
    document, _, _ = spec.load_config(REPOSITORY / "configs/experiments/her2_parent_replay.json")
    banks = document["banks"]
    per_array = int(banks["replay_rows"]) * 10 * 20 * 4
    assert banks["cache_bytes_per_parent"] == 2 * per_array == 160000000
    assert banks["monitor_cache_bytes_per_parent"] == 2 * int(banks["monitor_rows"]) * 10 * 20 * 4
    assert banks["cache_bytes_total"] == 3 * (banks["cache_bytes_per_parent"]
                                              + banks["monitor_cache_bytes_per_parent"])


def test_the_native_post_step_amendment_records_the_observation_it_rests_on():
    document, _, _ = spec.load_config(REPOSITORY / "configs/experiments/her2_parent_replay.json")
    tolerances = document["tolerances"]
    amendment = tolerances["native_post_step_amendment"]
    assert "1.392e-05" in amendment or "1.392e-5" in amendment, (
        "the measured failure is named, not paraphrased")
    assert "NOT widened" in amendment and "no smaller batch is retried" in amendment
    assert tolerances["post_adamw_parameter_atol"] == 3e-6, (
        "the amendment changes the criterion, not the tolerance")
    assert tolerances["post_adamw_oracle_atol"] < tolerances["post_adamw_parameter_atol"]

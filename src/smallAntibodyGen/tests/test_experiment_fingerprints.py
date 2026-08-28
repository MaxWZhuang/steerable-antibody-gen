"""Contract tests for run fingerprinting and checkpoint lineage (J03).

The point of `smallAntibodyGen.experiment` is that a resume against edited
inputs, an edited config, an edited tokenizer, or edited source code is
*impossible* rather than merely discouraged. These tests pin the pieces that
make that true:

- canonical serialization (sorted keys, no absolute paths, order-independent);
- six separate component fingerprints, so a mismatch says WHICH component moved;
- a source-revision hash computed from file CONTENT, so a dirty edit is
  identifiable by what changed rather than only by a `dirty: true` flag;
- an explicit, tested allowlist of generated artifacts -- and nothing else --
  excluded from that content hash;
- an explicit, tested constant naming the operational-only config fields, and
  nothing else, excluded from the objective fingerprint.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_repo(root: Path, *, source: str = "print('hi')\n", spec: str = "# contract\n") -> Path:
    """Build a minimal repo-shaped tree: the five hashed source roots."""
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_text(source, encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "train.py").write_text("# train\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "a.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (root / "specs").mkdir()
    (root / "specs" / "contract.md").write_text(spec, encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    # Files that must be ignored as generated artifacts.
    (root / "src" / "pkg" / "__pycache__").mkdir()
    (root / "src" / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00garbage")
    (root / "src" / "pkg.egg-info").mkdir()
    (root / "src" / "pkg.egg-info" / "PKG-INFO").write_text("Name: x\n", encoding="utf-8")
    (root / ".pytest_cache").mkdir()
    (root / "src" / ".DS_Store").write_bytes(b"\x00\x01")
    return root


def _tok():
    from smallAntibodyGen.tokenizer import AminoAcidTokenizer

    return AminoAcidTokenizer()


def _model_cfg(**overrides):
    from smallAntibodyGen.models.mlm import MLMConfig

    tok = _tok()
    params = dict(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    params.update(overrides)
    return MLMConfig(**params)


def _fingerprint(repo: Path, *, config=None, model_cfg=None, tokenizer=None,
                 data_paths=None, model_class="AntibodyMLM", parent=None):
    from smallAntibodyGen import experiment

    return experiment.compute_run_fingerprint(
        config=config if config is not None else {"data_path": "d.jsonl.gz", "seed": 42},
        model_config=model_cfg if model_cfg is not None else _model_cfg(),
        tokenizer=tokenizer if tokenizer is not None else _tok(),
        model_class=model_class,
        data_paths=data_paths or [],
        repo_root=repo,
        parent_checkpoint=parent,
    )


# --------------------------------------------------------------------------- #
# 1. Canonical serialization.
# --------------------------------------------------------------------------- #
def test_canonical_json_sorts_keys_and_is_insertion_order_independent():
    from smallAntibodyGen import experiment

    a = {"b": 1, "a": {"z": 2, "y": 3}}
    b = {"a": {"y": 3, "z": 2}, "b": 1}
    assert experiment.canonical_json(a) == experiment.canonical_json(b)
    assert experiment.canonical_json(a) == '{"a":{"y":3,"z":2},"b":1}'
    assert experiment.hash_payload(a) == experiment.hash_payload(b)


def test_hash_payload_is_stable_across_calls_and_has_no_timestamp():
    from smallAntibodyGen import experiment

    payload = {"x": 1}
    first = experiment.hash_payload(payload)
    assert first == experiment.hash_payload(payload)
    assert len(first) == 64


def test_absolute_paths_are_normalized_out_of_the_config_fingerprint(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    inside = repo / "data" / "processed" / "corpus.jsonl.gz"
    inside.parent.mkdir(parents=True)
    inside.write_text("x", encoding="utf-8")

    normalized = experiment.normalize_config_for_fingerprint(
        {"data_path": str(inside), "seed": 1}, repo_root=repo
    )
    assert normalized["data_path"] == "data/processed/corpus.jsonl.gz"
    assert str(repo) not in experiment.canonical_json(normalized)


def test_paths_outside_the_repo_collapse_to_a_basename(tmp_path: Path):
    """A tmp-dir corpus must fingerprint identically from any machine; only its
    content (hashed separately) distinguishes it."""
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    a = experiment.normalize_config_for_fingerprint(
        {"data_path": "/scratch/run-a/corpus.jsonl.gz"}, repo_root=repo
    )
    b = experiment.normalize_config_for_fingerprint(
        {"data_path": "/var/folders/xyz/corpus.jsonl.gz"}, repo_root=repo
    )
    assert a["data_path"] == b["data_path"] == "corpus.jsonl.gz"


# --------------------------------------------------------------------------- #
# 2. The operational-only exclusion list is a named, tested constant.
# --------------------------------------------------------------------------- #
def test_operational_only_exclusions_are_exactly_the_approved_set():
    from smallAntibodyGen import experiment

    assert experiment.OPERATIONAL_ONLY_CONFIG_FIELDS == frozenset({
        "output_dir",
        "tensorboard",
        "show_progress",
        "report_masked_fraction_bins",
        "resume_from_last",
    })


@pytest.mark.parametrize("field", sorted({
    "output_dir", "tensorboard", "show_progress",
    "report_masked_fraction_bins", "resume_from_last",
}))
def test_operational_fields_do_not_change_the_objective_fingerprint(field, tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    base = {"data_path": "d.jsonl.gz", field: "AAA" if field == "output_dir" else False}
    other = dict(base)
    other[field] = "BBB" if field == "output_dir" else True
    assert (
        experiment.objective_fingerprint(base, repo_root=repo)
        == experiment.objective_fingerprint(other, repo_root=repo)
    )


@pytest.mark.parametrize("field, value", [
    ("seed", 7),
    ("use_amp", True),
    ("device", "cuda"),
    ("train_num_workers", 4),
    ("eval_num_workers", 4),
    ("eval_batch_size", 64),
    ("checkpoint_every_steps", 100),
    ("smoke_test_only", True),
    ("mask_probability", 0.3),
    ("mask_rate_schedule", "uniform"),
    ("mask_replacement_strategy", "always_mask"),
    ("learning_rate", 1e-5),
    ("weight_decay", 0.5),
    ("conditional_denoising_eligibility", "binary_binders_only"),
])
def test_result_affecting_fields_do_change_the_objective_fingerprint(field, value, tmp_path: Path):
    """The owner's ruling: everything not on the exclusion list is result-affecting.

    `eval_batch_size` feeds `choose_probe_size` and probe rows are REMOVED from
    training; worker count moves the collator RNG stream through `seed_worker`;
    `use_amp` changes numerics and gates `scheduler.step()`.
    """
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    base = {"data_path": "d.jsonl.gz", field: None}
    other = dict(base)
    other[field] = value
    assert (
        experiment.objective_fingerprint(base, repo_root=repo)
        != experiment.objective_fingerprint(other, repo_root=repo)
    )


def test_every_train_config_field_is_either_excluded_or_fingerprinted(project_root: Path):
    """No TrainConfig field may fall through the classification unnoticed."""
    import importlib.util
    import sys

    from smallAntibodyGen import experiment

    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    cfg = module.TrainConfig(data_path="x")
    full = asdict(cfg)
    kept = experiment.normalize_config_for_fingerprint(full, repo_root=project_root.parents[1])
    dropped = set(full) - set(kept)
    assert dropped == set(experiment.OPERATIONAL_ONLY_CONFIG_FIELDS)


# --------------------------------------------------------------------------- #
# 3. Architecture fingerprint covers the MLMConfig-only fields.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field, value", [
    ("activation", "relu"),
    ("tie_weights", False),
    ("initializer_range", 0.05),
    ("scale_residual_init", False),
    ("norm_first", False),
    ("compat_readout", "mean"),
    ("d_model", 64),
    ("vocab_size", 36),
    ("pad_token_id", 1),
])
def test_architecture_fingerprint_covers_fields_unreachable_from_train_config(field, value):
    """`asdict(TrainConfig)` is NOT a complete architecture description.

    `activation`, `tie_weights`, `initializer_range` and `scale_residual_init`
    are hardcoded in `MLMConfig` and never reach `TrainConfig`, and
    `vocab_size`/`pad_token_id` come from the tokenizer. The architecture
    fingerprint is therefore computed from the CONSTRUCTED `MLMConfig`.
    """
    from smallAntibodyGen import experiment

    tok = _tok()
    a = experiment.architecture_fingerprint(_model_cfg(), tok, "AntibodyMLM")
    b = experiment.architecture_fingerprint(_model_cfg(**{field: value}), tok, "AntibodyMLM")
    assert a != b


def test_architecture_fingerprint_separates_the_two_model_classes():
    from smallAntibodyGen import experiment

    tok = _tok()
    cfg = _model_cfg()
    assert (
        experiment.architecture_fingerprint(cfg, tok, "AntibodyMLM")
        != experiment.architecture_fingerprint(cfg, tok, "AntibodyAntigenCrossAttention")
    )


# --------------------------------------------------------------------------- #
# 4. Tokenizer fingerprint hashes the real vocabulary, not the class name.
# --------------------------------------------------------------------------- #
def test_tokenizer_fingerprint_hashes_the_vocabulary_not_the_class_name():
    from smallAntibodyGen import experiment

    a = _tok()
    b = _tok()
    assert experiment.tokenizer_fingerprint(a) == experiment.tokenizer_fingerprint(b)

    mutated = _tok()
    mutated.vocab = list(mutated.vocab) + ["[NEW]"]
    mutated.token_to_id = {t: i for i, t in enumerate(mutated.vocab)}
    assert type(mutated).__name__ == type(a).__name__
    assert experiment.tokenizer_fingerprint(mutated) != experiment.tokenizer_fingerprint(a)


def test_tokenizer_fingerprint_is_sensitive_to_token_ORDER():
    """Vocabulary order IS the id assignment; a reorder silently remaps ids."""
    from smallAntibodyGen import experiment

    a = _tok()
    reordered = _tok()
    reordered.vocab = list(reordered.vocab)
    reordered.vocab[-1], reordered.vocab[-2] = reordered.vocab[-2], reordered.vocab[-1]
    reordered.token_to_id = {t: i for i, t in enumerate(reordered.vocab)}
    assert sorted(reordered.vocab) == sorted(a.vocab)
    assert experiment.tokenizer_fingerprint(reordered) != experiment.tokenizer_fingerprint(a)


def test_the_shipped_tokenizer_vocab_is_the_35_token_list():
    from smallAntibodyGen import experiment

    manifest = experiment.tokenizer_manifest(_tok())
    assert manifest["vocab_size"] == 35
    assert len(manifest["vocab"]) == 35


# --------------------------------------------------------------------------- #
# 5. Source revision: content hash, generated-artifact allowlist, dirty state.
# --------------------------------------------------------------------------- #
def test_generated_artifact_allowlist_is_explicit_and_covers_only_generated_files():
    from smallAntibodyGen import experiment

    assert experiment.GENERATED_ARTIFACT_DIR_NAMES >= frozenset({"__pycache__", ".pytest_cache"})
    assert ".pyc" in experiment.GENERATED_ARTIFACT_SUFFIXES
    assert ".DS_Store" in experiment.GENERATED_ARTIFACT_FILE_NAMES

    is_gen = experiment.is_generated_artifact
    assert is_gen(Path("src/pkg/__pycache__/mod.cpython-312.pyc"))
    assert is_gen(Path("src/pkg.egg-info/PKG-INFO"))
    assert is_gen(Path("src/.DS_Store"))
    assert is_gen(Path("scripts/x.pyc"))
    # Real source is never treated as generated.
    assert not is_gen(Path("src/smallAntibodyGen/experiment.py"))
    assert not is_gen(Path("configs/pretrain_oas_small.yaml"))
    assert not is_gen(Path("specs/decisions/0001-x.md"))
    assert not is_gen(Path("pyproject.toml"))


def test_source_files_are_sorted_and_exclude_only_generated_artifacts(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    rels = [p.as_posix() for p in experiment.iter_source_files(repo)]
    assert rels == sorted(rels)
    assert rels == [
        "configs/a.yaml",
        "pyproject.toml",
        "scripts/train.py",
        "specs/contract.md",
        "src/pkg/mod.py",
    ]


def test_untracked_source_files_are_included_in_the_content_hash(tmp_path: Path):
    """`specs/` is untracked in this repo today; the fingerprint must still see it."""
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    before = experiment.source_revision(repo)["content_hash"]
    (repo / "src" / "pkg" / "brand_new.py").write_text("x = 1\n", encoding="utf-8")
    after = experiment.source_revision(repo)["content_hash"]
    assert before != after


def test_a_dirty_edit_is_identifiable_by_content_not_only_by_a_flag(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    before = experiment.source_revision(repo)
    (repo / "src" / "pkg" / "mod.py").write_text("print('edited')\n", encoding="utf-8")
    after = experiment.source_revision(repo)

    assert before["content_hash"] != after["content_hash"]
    # The per-file map is what names the edited file, rather than a bare flag.
    assert before["files"]["src/pkg/mod.py"] != after["files"]["src/pkg/mod.py"]


def test_source_revision_works_without_git_and_records_the_commit_as_absent(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")  # not a git repo
    rev = experiment.source_revision(repo)
    assert rev["commit"] is None
    assert rev["git_available"] is False
    assert len(rev["content_hash"]) == 64


def test_dirty_worktree_is_detected_and_the_dirty_paths_recorded(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is unavailable")

    clean = experiment.source_revision(repo)
    assert clean["git_available"] is True
    assert clean["commit"] is not None and len(clean["commit"]) == 40
    assert clean["dirty"] is False
    assert clean["dirty_paths"] == []

    (repo / "src" / "pkg" / "mod.py").write_text("print('dirty')\n", encoding="utf-8")
    dirty = experiment.source_revision(repo)
    assert dirty["dirty"] is True
    assert "src/pkg/mod.py" in dirty["dirty_paths"]
    assert dirty["content_hash"] != clean["content_hash"]


def test_generated_artifacts_never_dirty_the_source_hash(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    before = experiment.source_revision(repo)["content_hash"]
    (repo / "src" / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x01different")
    (repo / "src" / "pkg.egg-info" / "PKG-INFO").write_text("Name: y\n", encoding="utf-8")
    (repo / "src" / ".DS_Store").write_bytes(b"\x02\x03")
    assert experiment.source_revision(repo)["content_hash"] == before


# --------------------------------------------------------------------------- #
# 6. Contracts fingerprint (D0/P0 and other approved specs).
# --------------------------------------------------------------------------- #
def test_contract_edit_changes_only_the_contracts_component(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    before = _fingerprint(repo)
    (repo / "specs" / "contract.md").write_text("# contract v2\n", encoding="utf-8")
    after = _fingerprint(repo)

    assert before["components"]["contracts"] != after["components"]["contracts"]
    # specs/ is inside the source roots too, so `source` moves as well; the
    # point is that the contract has its OWN component hash recorded.
    assert before["components"]["architecture"] == after["components"]["architecture"]
    assert before["components"]["objective"] == after["components"]["objective"]
    assert before["run_hash"] != after["run_hash"]


def test_contracts_manifest_records_every_spec_file_by_hash(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    (repo / "specs" / "decisions").mkdir()
    (repo / "specs" / "decisions" / "0001-x.md").write_text("d\n", encoding="utf-8")
    manifest = experiment.contracts_manifest(repo)
    assert sorted(manifest["files"]) == ["specs/contract.md", "specs/decisions/0001-x.md"]


# --------------------------------------------------------------------------- #
# 7. Run fingerprint: six components + combined hash + parent + dirty flag.
# --------------------------------------------------------------------------- #
def test_run_fingerprint_records_all_six_components_and_a_combined_hash(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    fp = _fingerprint(repo)
    assert set(fp["components"]) == {
        "architecture", "objective", "tokenizer", "data", "contracts", "source"
    }
    assert len(fp["run_hash"]) == 64
    assert "parent_checkpoint_hash" in fp
    assert fp["worktree_dirty"] in (True, False)
    assert fp["schema_version"] == experiment.FINGERPRINT_SCHEMA_VERSION


def test_run_fingerprint_is_deterministic_for_identical_inputs(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    first = _fingerprint(repo)
    second = _fingerprint(repo)
    assert first["run_hash"] == second["run_hash"]
    assert first["components"] == second["components"]
    # And nothing machine-specific leaked into it.
    assert str(repo) not in experiment.canonical_json(first)


def test_data_component_tracks_file_content_not_just_the_path(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    corpus = repo / "data" / "corpus.jsonl.gz"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("a\n", encoding="utf-8")

    before = _fingerprint(repo, data_paths=[corpus])
    corpus.write_text("b\n", encoding="utf-8")
    after = _fingerprint(repo, data_paths=[corpus])
    assert before["components"]["data"] != after["components"]["data"]
    assert before["run_hash"] != after["run_hash"]


def test_parent_checkpoint_hash_is_part_of_the_run_identity(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    a = _fingerprint(repo, parent={"path": "checkpoints/a/best.pt", "run_hash": "a" * 64,
                                   "file_sha256": "1" * 64})
    b = _fingerprint(repo, parent={"path": "checkpoints/b/best.pt", "run_hash": "b" * 64,
                                   "file_sha256": "2" * 64})
    assert a["parent_checkpoint_hash"] != b["parent_checkpoint_hash"]
    assert a["run_hash"] != b["run_hash"]


# --------------------------------------------------------------------------- #
# 8. Resume enforcement.
# --------------------------------------------------------------------------- #
def test_exact_match_resume_is_allowed(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    stored = _fingerprint(repo)
    current = _fingerprint(repo)
    experiment.check_resume_fingerprint(stored, current, Path("last.pt"))  # must not raise


def test_one_field_mismatch_names_the_field_and_both_values(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    stored = _fingerprint(repo, config={"data_path": "d.jsonl.gz", "mask_probability": 0.15})
    current = _fingerprint(repo, config={"data_path": "d.jsonl.gz", "mask_probability": 0.30})

    with pytest.raises(experiment.ResumeFingerprintMismatch) as excinfo:
        experiment.check_resume_fingerprint(stored, current, Path("last.pt"))

    message = str(excinfo.value)
    assert "mask_probability" in message
    assert "0.15" in message and "0.3" in message
    assert "objective" in message


def test_architecture_mismatch_rejects_resume_naming_the_field(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    stored = _fingerprint(repo, model_cfg=_model_cfg(d_model=32))
    current = _fingerprint(repo, model_cfg=_model_cfg(d_model=64))
    with pytest.raises(experiment.ResumeFingerprintMismatch, match="d_model"):
        experiment.check_resume_fingerprint(stored, current, Path("last.pt"))


def test_tokenizer_mismatch_rejects_resume(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    mutated = _tok()
    mutated.vocab = list(mutated.vocab) + ["[NEW]"]
    mutated.token_to_id = {t: i for i, t in enumerate(mutated.vocab)}

    stored = _fingerprint(repo)
    current = _fingerprint(repo, tokenizer=mutated)
    with pytest.raises(experiment.ResumeFingerprintMismatch, match="tokenizer"):
        experiment.check_resume_fingerprint(stored, current, Path("last.pt"))


def test_source_edit_rejects_resume_and_names_the_edited_file(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    stored = _fingerprint(repo)
    (repo / "scripts" / "train.py").write_text("# train v2\n", encoding="utf-8")
    current = _fingerprint(repo)
    with pytest.raises(experiment.ResumeFingerprintMismatch, match="scripts/train.py"):
        experiment.check_resume_fingerprint(stored, current, Path("last.pt"))


def test_changed_contract_rejects_resume(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    stored = _fingerprint(repo)
    (repo / "specs" / "contract.md").write_text("# contract v2\n", encoding="utf-8")
    current = _fingerprint(repo)
    with pytest.raises(experiment.ResumeFingerprintMismatch, match="contracts"):
        experiment.check_resume_fingerprint(stored, current, Path("last.pt"))


def test_mismatch_message_is_not_two_opaque_hashes(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    stored = _fingerprint(repo, config={"data_path": "d.jsonl.gz", "seed": 42})
    current = _fingerprint(repo, config={"data_path": "d.jsonl.gz", "seed": 7})
    with pytest.raises(experiment.ResumeFingerprintMismatch) as excinfo:
        experiment.check_resume_fingerprint(stored, current, Path("last.pt"))
    message = str(excinfo.value)
    assert "seed" in message
    assert "checkpoint=42" in message
    assert "run=7" in message


# --------------------------------------------------------------------------- #
# 9. Legacy checkpoints.
# --------------------------------------------------------------------------- #
def test_resume_against_a_fingerprintless_checkpoint_is_a_hard_error(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    current = _fingerprint(repo)
    with pytest.raises(experiment.LegacyCheckpointResumeError) as excinfo:
        experiment.check_resume_fingerprint(None, current, Path("checkpoints/mlm_small/last.pt"))
    message = str(excinfo.value)
    assert experiment.RUN_FINGERPRINT_KEY in message
    assert "checkpoints/mlm_small/last.pt" in message


def test_legacy_resume_error_is_raised_from_the_checkpoint_payload_too(tmp_path: Path):
    """The 6 checkpoints on disk have 5 top-level keys and no fingerprint."""
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    legacy_payload = {
        "epoch": 3, "model_state_dict": {}, "optimizer_state_dict": {},
        "train_config": {"d_model": 128}, "val_loss": 1.0,
    }
    assert experiment.read_fingerprint(legacy_payload) is None
    with pytest.raises(experiment.LegacyCheckpointResumeError):
        experiment.check_resume_fingerprint(
            experiment.read_fingerprint(legacy_payload), _fingerprint(repo), Path("last.pt")
        )


# --------------------------------------------------------------------------- #
# 10. Warm-start rules are SEPARATE from resume rules.
# --------------------------------------------------------------------------- #
def test_warm_start_allows_the_intended_objective_and_data_transition(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    parent = _fingerprint(
        repo,
        config={"data_path": "oas_paired.jsonl.gz", "training_stage": "paired_refine",
                "mask_probability": 0.15},
    )
    child = _fingerprint(
        repo,
        config={"data_path": "antigen.jsonl.gz", "training_stage": "antigen_real_label_refine",
                "mask_probability": 0.30},
    )
    assert parent["run_hash"] != child["run_hash"]
    # A warm start deliberately changes objective and data; that is legal.
    experiment.check_warm_start_fingerprint(parent, child, Path("parent.pt"))


def test_warm_start_rejects_an_architecture_change(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    parent = _fingerprint(repo, model_cfg=_model_cfg(activation="gelu"))
    child = _fingerprint(repo, model_cfg=_model_cfg(activation="relu"))
    with pytest.raises(experiment.WarmStartFingerprintMismatch, match="activation"):
        experiment.check_warm_start_fingerprint(parent, child, Path("parent.pt"))


def test_warm_start_rejects_a_tokenizer_change(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    mutated = _tok()
    mutated.vocab = list(mutated.vocab) + ["[NEW]"]
    mutated.token_to_id = {t: i for i, t in enumerate(mutated.vocab)}
    parent = _fingerprint(repo)
    child = _fingerprint(repo, tokenizer=mutated)
    with pytest.raises(experiment.WarmStartFingerprintMismatch, match="tokenizer"):
        experiment.check_warm_start_fingerprint(parent, child, Path("parent.pt"))


def test_warm_start_ignores_from_scratch_init_only_architecture_fields(tmp_path: Path):
    """`initializer_range` and `scale_residual_init` only shape a FROM-SCRATCH
    init; loaded weights overwrite it, so they must not block a warm start even
    though they are part of the run identity for a resume."""
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    parent = _fingerprint(repo, model_cfg=_model_cfg(initializer_range=0.02))
    child = _fingerprint(repo, model_cfg=_model_cfg(initializer_range=0.05,
                                                    scale_residual_init=False))
    assert parent["components"]["architecture"] != child["components"]["architecture"]
    experiment.check_warm_start_fingerprint(parent, child, Path("parent.pt"))  # must not raise


def test_warm_start_from_a_legacy_parent_is_allowed_but_reported(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    child = _fingerprint(repo)
    warning = experiment.warm_start_lineage_warning(None, Path("checkpoints/mlm_small/best.pt"))
    assert warning is not None
    assert "lineage" in warning.lower()
    assert "checkpoints/mlm_small/best.pt" in warning
    # And it does not raise.
    experiment.check_warm_start_fingerprint(None, child, Path("checkpoints/mlm_small/best.pt"))
    assert experiment.warm_start_lineage_warning(child, Path("p.pt")) is None


# --------------------------------------------------------------------------- #
# 11. Clean-worktree requirement for promoted canonical runs.
# --------------------------------------------------------------------------- #
def test_promoted_canonical_run_requires_a_clean_worktree(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    fp = _fingerprint(repo)
    fp["worktree_dirty"] = True
    fp["manifests"]["source"]["dirty"] = True
    fp["manifests"]["source"]["dirty_paths"] = ["src/pkg/mod.py"]
    fp["manifests"]["source"]["git_available"] = True
    with pytest.raises(experiment.DirtyWorktreeError, match="src/pkg/mod.py"):
        experiment.require_clean_worktree(fp)


def test_promotion_check_refuses_when_git_cannot_verify_the_worktree(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    fp = _fingerprint(repo)  # not a git repo
    with pytest.raises(experiment.DirtyWorktreeError, match="cannot be verified"):
        experiment.require_clean_worktree(fp)


def test_a_development_run_may_be_dirty_when_content_and_paths_are_recorded(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    fp = _fingerprint(repo)
    fp["worktree_dirty"] = True
    fp["manifests"]["source"]["dirty"] = True
    fp["manifests"]["source"]["dirty_paths"] = ["src/pkg/mod.py"]
    # No promotion requested -> the run proceeds, but the record is complete.
    assert len(fp["manifests"]["source"]["content_hash"]) == 64
    assert fp["manifests"]["source"]["files"]["src/pkg/mod.py"]
    assert fp["manifests"]["source"]["dirty_paths"] == ["src/pkg/mod.py"]


# --------------------------------------------------------------------------- #
# 12. The fingerprint payload survives a JSON round trip (it is written next to
#     the checkpoints as run_fingerprint.json, and stored inside every .pt).
# --------------------------------------------------------------------------- #
def test_fingerprint_payload_is_json_serializable(tmp_path: Path):
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    fp = _fingerprint(repo)
    assert json.loads(json.dumps(fp)) == fp


@pytest.mark.parametrize(
    "external",
    [
        "/scratch/run-a/corpus.jsonl.gz",       # POSIX absolute
        "/var/folders/xyz/corpus.jsonl.gz",     # POSIX absolute, different prefix
        "C:/scratch/run-a/corpus.jsonl.gz",     # Windows absolute, forward slashes
        r"C:\scratch\run-a\corpus.jsonl.gz",   # Windows absolute, backslashes
        r"\scratch\corpus.jsonl.gz",                   # Windows root-relative
    ],
)
def test_external_paths_collapse_regardless_of_the_host_os(tmp_path: Path, external: str):
    """
    Absoluteness must be a property of the STRING, not of the machine reading it.

    `os.path.isabs` / `Path.is_absolute` answer for the host only. On Windows
    `Path("/scratch/corpus.gz").is_absolute()` is False (no drive letter), so a
    POSIX scratch path was hashed into the objective fingerprint with its full
    machine-specific prefix; on POSIX the same happened to "C:/scratch/...".
    Two machines then fingerprint the same logical run differently and a resume
    is refused for a reason unrelated to the objective -- while the checkpoint
    also carries someone's absolute directory layout.

    Every spelling below names the same file and must collapse to its basename.
    """
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    normalized = experiment.normalize_config_for_fingerprint(
        {"data_path": external}, repo_root=repo
    )
    assert normalized["data_path"] == "corpus.jsonl.gz"


def test_repo_relative_paths_are_untouched_by_the_absoluteness_rule(tmp_path: Path):
    """The widened rule must not start collapsing ordinary relative paths."""
    from smallAntibodyGen import experiment

    repo = _make_repo(tmp_path / "repo")
    normalized = experiment.normalize_config_for_fingerprint(
        {"data_path": "data/processed/oas_5k/oas_all.jsonl.gz"}, repo_root=repo
    )
    assert normalized["data_path"] == "data/processed/oas_5k/oas_all.jsonl.gz"

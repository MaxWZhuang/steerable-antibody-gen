"""
Tests for `scripts/check_local_only_files.py`.

The guardrail used to live only in `.git/hooks/pre-commit`, which git does not
version and `git clone` does not copy -- so it protected exactly one working
copy and nothing verified it. These tests are the point of moving it into a
script: the policy is now executable and checkable from a fresh clone.

The most important case is the last one: the repository's own HEAD must be clean
under this policy. That turns the rule from a description into a standing check.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def guard(project_root: Path):
    script = project_root.parents[1] / "scripts" / "check_local_only_files.py"
    spec = importlib.util.spec_from_file_location("check_local_only_files", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "docs/ARCHITECTURE.md",
        "docs/BUGLOG.md",
        "docs/PLAN-steering-prerequisites.md",
        "docs/research/whatever.md",
        "CLAUDE.md",
        ".claude/settings.local.json",
        "outputs/J02-and-Ruling4-corpus-evidence.md",
        "outputs/gpu-memory-probe.json",
        "checkpoints/mlm_3m_unpaired_v5/best.pt",
        "data/raw/asd-antibody-antigen/part-00000.parquet",
        "data/processed/antibody_antigen/antibody_antigen.jsonl.gz",
        "wandb/run-123/logs",
        "logs/train.log",
        "something.tmp",
    ],
)
def test_local_only_paths_are_blocked(guard, path):
    assert guard.classify(path) is not None, f"{path} should be local-only"


def test_the_three_internal_docs_have_no_exemption(guard):
    """
    These were briefly tracked and then removed. The policy must have NO
    negation for them: in this repository tracking a file is a publishing
    decision, and the failure mode is silent -- nothing warns you at push time.
    """
    for path in (
        "docs/ARCHITECTURE.md",
        "docs/BUGLOG.md",
        "docs/PLAN-steering-prerequisites.md",
    ):
        reason = guard.classify(path)
        assert reason is not None
        assert "internal" in reason


@pytest.mark.parametrize(
    "path",
    [
        "src/smallAntibodyGen/models/mlm.py",
        "scripts/mlm_train.py",
        "configs/refine_antigen_real_label.yaml",
        "specs/conditional_denoising_eligibility.md",
        "specs/benchmarks/avida_hil6.json",
        "pyproject.toml",
        "README.md",
        ".gitignore",
    ],
)
def test_ordinary_repository_files_are_allowed(guard, path):
    """A guardrail that blocks real code would just get bypassed habitually."""
    assert guard.classify(path) is None, f"{path} should be committable"


def test_the_reason_is_reported_not_just_the_match(guard):
    """'Blocked' without 'why' invites someone to delete the rule."""
    offenders = guard.check(["docs/BUGLOG.md", "src/ok.py", "checkpoints/best.pt"])
    assert [p for p, _ in offenders] == ["checkpoints/best.pt", "docs/BUGLOG.md"]
    reasons = dict(offenders)
    assert "internal" in reasons["docs/BUGLOG.md"]
    assert "weights" in reasons["checkpoints/best.pt"]


def test_clean_input_returns_no_offenders(guard):
    assert guard.check(["src/a.py", "configs/b.yaml"]) == []


def test_main_exits_nonzero_on_a_local_only_path(guard, capsys):
    assert guard.main(["docs/BUGLOG.md"]) == 1
    err = capsys.readouterr().err
    assert "docs/BUGLOG.md" in err
    # The message must actively discourage the "just add an exemption" fix.
    assert "publishing decision" in err


def test_main_exits_zero_on_clean_paths(guard):
    assert guard.main(["src/a.py"]) == 0


def test_this_repository_head_is_clean_under_the_policy(guard, project_root: Path):
    """
    The standing check: nothing local-only may be present in the committed tree.

    This is what would have caught the three internal docs being tracked, and it
    runs on every clone rather than only where a hook happens to be installed.
    """
    import subprocess

    repo_root = project_root.parents[1]
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("git ls-tree unavailable")
    offenders = guard.check(
        [line.strip() for line in result.stdout.splitlines() if line.strip()]
    )
    assert offenders == [], (
        "local-only files are committed in HEAD: "
        + ", ".join(f"{p} [{why}]" for p, why in offenders)
    )

"""Authenticating local-only campaign evidence against committed digests.

The repository was narrowed to code alone, so `reference/evidence/` is no longer
tracked. Both campaign freezes used to ask "is every evidence file tracked, and
equal to HEAD?", which can only be satisfied by publishing the research material.
They now ask a committed manifest of digests instead, and these tests pin the
part that makes that substitution honest: a manifest only certifies evidence it
actually matches, and an untracked manifest certifies nothing at all.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from smallAntibodyGen.experiments import her2_support as support


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "reference" / "evidence" / "camp").mkdir(parents=True)
    # Mirror the real repository: reference/ is local-only, so `git add -A` must
    # not sweep the evidence into the commit. Without this the fixture would
    # track the evidence and quietly test the OLD contract.
    (repo / ".gitignore").write_text("reference/\n", encoding="utf-8", newline="\n")
    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"],
                       check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"],
                       check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is unavailable")
    return repo


def _commit(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, timeout=60)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "pin"], check=True, timeout=60)


def _evidence(repo: Path, **files: str) -> list[str]:
    logical = []
    for name, body in files.items():
        path = repo / "reference" / "evidence" / "camp" / name
        path.write_text(body, encoding="utf-8", newline="\n")
        logical.append(f"reference/evidence/camp/{name}")
    return sorted(logical)


def _write_manifest(repo: Path, logical: list[str], *, campaign="camp") -> Path:
    target = repo / support.EVIDENCE_MANIFEST_DIR / f"{campaign}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({
            "kind": "campaign_evidence_manifest",
            "campaign_id": campaign,
            "sha256": support.evidence_digests(repo, logical),
        }, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return target


def test_matching_evidence_is_authenticated_without_being_tracked(tmp_path: Path):
    """The whole point: the bytes stay local and the freeze still means something."""
    repo = _repo(tmp_path)
    logical = _evidence(repo, a="alpha\n", b="beta\n")
    _write_manifest(repo, logical)
    _commit(repo)

    # The evidence itself is deliberately NOT tracked.
    assert not support.git_tracked(repo, logical[0])

    record = support.require_evidence_matches_manifest(repo, logical, "camp")
    assert record["file_count"] == 2
    assert record["path"] == "configs/evidence_manifests/camp.json"


def test_changed_evidence_bytes_are_refused(tmp_path: Path):
    repo = _repo(tmp_path)
    logical = _evidence(repo, a="alpha\n")
    _write_manifest(repo, logical)
    _commit(repo)

    (repo / "reference" / "evidence" / "camp" / "a").write_text(
        "tampered\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="differ from their committed digests"):
        support.require_evidence_matches_manifest(repo, logical, "camp")


def test_an_extra_evidence_file_is_refused(tmp_path: Path):
    """A freeze over a grown evidence set would certify material nobody pinned."""
    repo = _repo(tmp_path)
    logical = _evidence(repo, a="alpha\n")
    _write_manifest(repo, logical)
    _commit(repo)

    logical = _evidence(repo, a="alpha\n", b="beta\n")
    with pytest.raises(ValueError, match="Not in the manifest"):
        support.require_evidence_matches_manifest(repo, logical, "camp")


def test_a_missing_manifest_refuses_rather_than_passing(tmp_path: Path):
    """Absent authentication must fail closed; silence here would certify anything."""
    repo = _repo(tmp_path)
    logical = _evidence(repo, a="alpha\n")
    _commit(repo)
    with pytest.raises(ValueError, match="is absent"):
        support.require_evidence_matches_manifest(repo, logical, "camp")


def test_an_untracked_manifest_certifies_nothing(tmp_path: Path):
    """
    The manifest is the committed half of the bargain. If it is not tracked,
    anyone can rewrite it beside the files it claims to certify, and the freeze
    would be checking a file against itself.
    """
    repo = _repo(tmp_path)
    logical = _evidence(repo, a="alpha\n")
    _commit(repo)
    _write_manifest(repo, logical)  # written AFTER the commit, so untracked

    with pytest.raises(ValueError, match="not tracked at HEAD"):
        support.require_evidence_matches_manifest(repo, logical, "camp")

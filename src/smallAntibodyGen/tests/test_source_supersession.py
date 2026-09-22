"""Reviewed source migrations for campaigns that were already frozen.

Narrowing the repository to code alone forced files inside both campaigns'
frozen source closures to change, which left every completed campaign unable to
re-verify: replay verification, the support-report rebuild and four gated stages
in ``audit_her2_support.py`` all failed on unchanged artifacts.

A freeze binds a campaign to exact source bytes, so the answer is not to relax
the check. A supersession records one migration explicitly. These tests pin the
ways it must REFUSE, because a mechanism that accepts anything would quietly
retire the guarantee it exists to preserve -- in particular, an unauthenticated
record lets an uncommitted source edit authorize itself, which is the exact
bypass the freeze is there to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from smallAntibodyGen.experiments import her2_support as support


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"],
                       check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"],
                       check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is unavailable")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8", newline="\n")
    _commit(repo)
    return repo


def _commit(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, timeout=60)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "pin"], check=True, timeout=60)


def _head(repo: Path) -> str:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True, timeout=60)
    return out.stdout.strip()


def _record(repo: Path, **over) -> dict:
    base = {"file": "src/a.py", "superseded_sha256": "old", "current_sha256": "new",
            "commit": _head(repo), "reason": "narrowing"}
    base.update(over)
    return base


def _write(repo: Path, *records: dict) -> None:
    target = repo / support.SOURCE_SUPERSESSIONS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"supersessions": list(records)}, indent=2) + "\n",
                      encoding="utf-8", newline="\n")


def test_an_exactly_recorded_committed_transition_is_accepted(tmp_path: Path):
    repo = _repo(tmp_path)
    _write(repo, _record(repo))
    _commit(repo)

    record = support.classify_source_drift(repo, "src/a.py", "old", "new")
    assert record is not None
    assert record["reason"] == "narrowing"


def test_an_uncommitted_record_cannot_authorize_itself(tmp_path: Path):
    """
    The bypass this check closes: edit a frozen source, add its new digest to the
    local record, and both identity verifiers would accept the change with no
    commit and no review.
    """
    repo = _repo(tmp_path)
    _write(repo, _record(repo))  # written, deliberately never committed

    with pytest.raises(ValueError, match="not tracked at HEAD"):
        support.classify_source_drift(repo, "src/a.py", "old", "new")


def test_a_locally_modified_record_is_refused(tmp_path: Path):
    """The reviewed version is the committed one, not whatever is on disk now."""
    repo = _repo(tmp_path)
    _write(repo, _record(repo))
    _commit(repo)
    _write(repo, _record(repo, current_sha256="something_else"))

    with pytest.raises(ValueError, match="differs from its committed bytes"):
        support.classify_source_drift(repo, "src/a.py", "old", "something_else")


def test_a_record_naming_no_real_commit_is_refused(tmp_path: Path):
    """A migration has to point at the change that made it, or it documents nothing."""
    repo = _repo(tmp_path)
    _write(repo, _record(repo, commit="0" * 40))
    _commit(repo)

    with pytest.raises(ValueError, match="not a commit in this repository"):
        support.classify_source_drift(repo, "src/a.py", "old", "new")


def test_a_record_without_a_reason_is_refused(tmp_path: Path):
    repo = _repo(tmp_path)
    _write(repo, _record(repo, reason=""))
    _commit(repo)

    with pytest.raises(ValueError, match="carries no reason"):
        support.classify_source_drift(repo, "src/a.py", "old", "new")


def test_a_record_does_not_authorize_whatever_is_on_disk_now(tmp_path: Path):
    """
    Both ends are pinned. A record naming the old digest but not the bytes
    actually present would turn one reviewed migration into a standing licence
    for that file to keep changing.
    """
    repo = _repo(tmp_path)
    _write(repo, _record(repo))
    _commit(repo)
    assert support.classify_source_drift(repo, "src/a.py", "old", "drifted_again") is None


def test_an_unrecorded_file_is_not_superseded(tmp_path: Path):
    repo = _repo(tmp_path)
    _write(repo, _record(repo))
    _commit(repo)
    assert support.classify_source_drift(repo, "src/b.py", "old", "new") is None


def test_absent_records_supersede_nothing(tmp_path: Path):
    repo = _repo(tmp_path)
    assert support.classify_source_drift(repo, "src/a.py", "old", "new") is None


def test_the_repository_records_only_the_narrowing_migration():
    """
    A guard on the real file: supersessions are meant to be rare and explained.
    If this list grows, somebody is using the mechanism to avoid re-freezing.
    """
    root = Path(__file__).resolve().parents[3]
    table = support.load_source_supersessions(root)
    assert len(table) <= 3, f"unexpected supersessions: {sorted(k[0] for k in table)}"
    for (logical, _), record in table.items():
        assert record["reason"], f"{logical} is superseded without a reason"
        assert record["commit"], f"{logical} is superseded without naming a commit"

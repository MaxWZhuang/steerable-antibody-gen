"""The atomic writers create their temporary EXCLUSIVELY, so a planted link is never written through.

Astra, confirming pass 2 (2026-10-08): ``paths.write_text`` opened a fixed ``<name>.tmp`` in ``"w"``
mode. A hard link (no privilege needed on NTFS) or symbolic link already sitting at that name made the
open truncate whatever it pointed at, so an admission-phase stage record could destroy a pinned origin
flight's ``best.pt``. The same fixed-name pattern was in ``paths.write_arrays`` and in
``her2_routes_checkpoints.save`` (the trainer's ``best.pt`` and ``last.pt``). Each writer now creates a
fresh random-named file with ``O_EXCL`` and renames it into place; the old temporary name is no
longer special.

``her2_runtime.save_json`` has the same fixed-name pattern and is deliberately NOT changed here: it is
pinned by the parent-replay lineage (``HER2_CODE_FILES``) and by the support audit's freeze, and no
routes code path calls it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.tests.test_her2_routes_checkpoints import TinyModel, identity, state_with

ORIGIN_BYTES = b"ORIGIN-CHECKPOINT" * 64


def _link(kind, link, target):
    """A hard link (works unprivileged on NTFS) or a symbolic link (skipped when not permitted)."""
    if kind == "hard":
        os.link(target, link)
        return
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"cannot create a symbolic link here: {error}")


def _write_text(path):
    paths.write_text(path, "hello\n")


def _write_json(path):
    paths.write_json(path, {"a": 1})


def _write_arrays(path):
    paths.write_arrays(path, {"x": np.arange(4)})


def _save_checkpoint(path):
    checkpoints.save(path, model=TinyModel(), heads=None, optimizer=None, scheduler=None,
                     state=state_with(), identity=identity())


#: (writer, target name, the fixed temporary name the OLD code opened for writing)
WRITERS = {
    "paths.write_text": (_write_text, "out.txt", "out.txt.tmp"),
    "paths.write_json": (_write_json, "out.json", "out.json.tmp"),
    "paths.write_arrays": (_write_arrays, "out.npz", "out.npz.tmp.npz"),
    "checkpoints.save": (_save_checkpoint, "last.pt", "last.pt.tmp"),
}


@pytest.mark.parametrize("kind", ["hard", "symbolic"])
@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_link_planted_at_the_old_temporary_name_is_never_written_through(tmp_path, writer, kind):
    call, name, legacy = WRITERS[writer]
    origin = tmp_path / "origin" / "best.pt"
    origin.parent.mkdir()
    origin.write_bytes(ORIGIN_BYTES)
    destination = tmp_path / "run" / "stages"
    destination.mkdir(parents=True)
    _link(kind, destination / legacy, origin)
    call(destination / name)
    assert origin.read_bytes() == ORIGIN_BYTES, f"{writer} wrote through a link at {legacy}"
    assert (destination / name).is_file() and (destination / name).stat().st_size > 0
    leftovers = sorted(p.name for p in destination.iterdir())
    assert leftovers == sorted([name, legacy]), (
        f"{writer} left something behind besides the planted link: {leftovers}")


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_writer_that_fails_before_the_replace_leaves_no_temporary(tmp_path, writer, monkeypatch):
    call, name, _legacy = WRITERS[writer]
    destination = tmp_path / "run"
    destination.mkdir()

    def refuse(*_args, **_kwargs):
        raise OSError("simulated failure at the rename")
    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(type(destination), "replace", refuse, raising=False)
    with pytest.raises(OSError, match="simulated"):
        call(destination / name)
    assert list(destination.iterdir()) == [], f"{writer} left a temporary after a failed replace"


def test_an_existing_target_is_replaced_not_edited_in_place(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old\n", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    os.link(target, alias)                     # the old inode, reachable by a second name
    paths.write_text(target, "new\n")
    assert target.read_text(encoding="utf-8") == "new\n"
    assert alias.read_text(encoding="utf-8") == "old\n", "the replace edited the shared inode"


def test_the_written_bytes_are_unchanged_by_the_new_temporary(tmp_path):
    paths.write_json(tmp_path / "a.json", {"b": 2, "a": 1})
    assert (tmp_path / "a.json").read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'
    digest = paths.write_text(tmp_path / "b.txt", "x\ny\n")
    assert (tmp_path / "b.txt").read_bytes() == b"x\ny\n"          # LF, never CRLF
    assert digest == paths.sha256_file(tmp_path / "b.txt")
    paths.write_arrays(tmp_path / "c.npz", {"v": np.arange(3)})
    assert paths.read_arrays(tmp_path / "c.npz")["v"].tolist() == [0, 1, 2]


def test_the_temporaries_are_exclusive_and_distinct(tmp_path):
    first = paths.exclusive_temporary(tmp_path / "x.json")
    second = paths.exclusive_temporary(tmp_path / "x.json")
    try:
        assert first[1] != second[1]
        assert first[1].parent == tmp_path and second[1].parent == tmp_path
        assert first[1].name.startswith("x.json.") and first[1].name.endswith(".tmp")
    finally:
        for descriptor, temporary in (first, second):
            os.close(descriptor)
            temporary.unlink()


def test_the_checkpoint_still_round_trips_through_the_new_writer(tmp_path):
    model = TinyModel()
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state_with(), identity=identity())
    other = TinyModel()
    restored, _record = checkpoints.restore(target, model=other, heads=None, optimizer=None,
                                            scheduler=None, state=state_with(step=0),
                                            identity=identity())
    assert restored.step == 5
    for key, value in model.state_dict().items():
        assert torch.equal(other.state_dict()[key], value)


# ---------------------------------------------------------------------------
# sol round 6: the raw descriptor must be closed when fdopen fails
# ---------------------------------------------------------------------------

def _fdopen_refuses(monkeypatch, created):
    real = paths.exclusive_temporary

    def recording(path, **kwargs):
        descriptor, temporary = real(path, **kwargs)
        created.append(descriptor)
        return descriptor, temporary
    monkeypatch.setattr(paths, "exclusive_temporary", recording)

    def refuse(*_args, **_kwargs):
        raise MemoryError("simulated failure while wrapping the descriptor")
    monkeypatch.setattr(os, "fdopen", refuse)


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_the_descriptor_and_the_temporary_are_released_when_fdopen_fails(tmp_path, writer, monkeypatch):
    call, name, _legacy = WRITERS[writer]
    destination = tmp_path / "run"
    destination.mkdir()
    created = []
    _fdopen_refuses(monkeypatch, created)
    with pytest.raises(MemoryError, match="simulated"):
        call(destination / name)
    assert len(created) == 1
    with pytest.raises(OSError):
        os.fstat(created[0])               # a closed descriptor: the raw handle did not leak
    assert list(destination.iterdir()) == [], f"{writer} left its temporary after fdopen failed"

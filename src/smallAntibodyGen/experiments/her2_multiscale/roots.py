"""The pilot's five roots, the context the services read through, and the fresh output directory.

The pilot reads from several places and writes to exactly one, and the existing
``RoutesContext`` assumes they are the same tree. They are not:

* ``code_root`` -- this worktree. The only thing read from it besides the code is the DPLM pins
  file (``config.dplm.pins``), which the services resolve through ``context.repository``.
* ``data_repo_root`` -- the existing data repository. Every other ``repository(logical)`` path the
  config declares (raw HER2, paired OAS, CR9114 files) resolves here.
* ``artifact_root`` -- the existing artifact volume: pinned weights, upstream packages, acquired
  source files and manifests. ``artifact(logical)`` resolves here, read-only.
* ``run_root`` -- a fresh directory this run creates and owns. ``path(...)`` resolves here. It is
  reserved exclusively BEFORE anything else happens: a directory that already exists is refused (an
  empty one included), as is one that is, contains or sits inside any other root or any data or
  historical location the base config declares (those can be junctions to volumes outside the four
  roots, so they are resolved before the comparison).
* ``run_root/source_cache`` and ``run_root/runtime`` -- the parsed-source caches and the
  ``prepare_environment`` caches and temporaries, so nothing is written under the artifact volume.

This module is torch-free so the command line can reserve the directory and then call
``prepare_environment`` ahead of the first torch import.

``training_identity`` of the old services hashes the config, the source snapshot and the external
marker of ONE root. With roots split like this it would still produce a number, and the number would
claim a provenance it does not have, so the pilot records its own identity (``runner``) and the old
one is not used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import her2_support_paths as paths
from ..her2_routes_conditions import ConditionSpec
from ..her2_runtime import require

#: ``<run root>/<name>``: where each kind of output lands.
SOURCE_CACHE = "source_cache"
RUNTIME = "runtime"


@dataclass(frozen=True)
class PilotRoots:
    code_root: Path
    data_repo_root: Path
    artifact_root: Path
    run_root: Path
    frozen_inputs: Path

    @property
    def source_cache(self):
        return self.run_root / SOURCE_CACHE

    @property
    def runtime_root(self):
        return self.run_root / RUNTIME

    def document(self):
        return {name: str(getattr(self, name)) for name in
                ("code_root", "data_repo_root", "artifact_root", "run_root", "frozen_inputs")}


def _existing_directory(value, name):
    path = Path(value)
    require(path.is_absolute(), f"{name} must be an absolute path, got {str(value)!r}")
    require(path.is_dir(), f"{name} {path} is not an existing directory")
    return path


def declared_input_roots(base, *, data_repo_root):
    """``{name: location}`` of every data and historical path the base config declares.

    Under the data repository, where ``repository(logical)`` resolves them: each ``roots`` entry
    (raw HER2, paired OAS, CR9114 raw, ...), each ``roots.historical`` flight directory (these are
    junctions to other volumes on the development machine, so a lexical comparison with the data
    repository would never see them), and every ``sources.paths`` entry whose root is the
    repository. Keys ending in ``_note`` are prose. A value that is not a path string is an error,
    not a skip: a new kind of root must be protected deliberately or refused.

    Stdlib plus the path helpers only, so the command line can call it before the environment is
    prepared.
    """
    base_root = Path(data_repo_root)
    declared = {}
    block = base.get("roots")
    require(isinstance(block, dict), "the base config declares no roots block")
    for name, value in sorted(block.items()):
        if name.endswith("_note"):
            continue
        if name == "historical":
            require(isinstance(value, dict), "roots.historical must be an object")
            for label, logical in sorted(value.items()):
                require(isinstance(logical, str), f"roots.historical.{label} must be a path string")
                declared[f"historical {label}"] = paths.resolve_under(base_root, logical)
        else:
            require(isinstance(value, str), f"roots.{name} must be a path string, got {value!r}")
            declared[f"data root {name}"] = paths.resolve_under(base_root, value)
    sources = (base.get("sources") or {}).get("paths") or {}
    for name, entry in sorted(sources.items()):
        if entry.get("root", "artifact") != "repository":
            continue
        logicals = entry.get("logicals") or ([entry["logical"]] if entry.get("logical") else [])
        for position, logical in enumerate(logicals):
            declared[f"data source {name}[{position}]"] = paths.resolve_under(base_root, logical)
    return declared


def reserve_run_root(run_root, *, code_root, data_repo_root, artifact_root, frozen_inputs,
                     declared_inputs):
    """Create ``run_root`` exclusively and return the :class:`PilotRoots` that owns it.

    Every input root must already exist; the run root must not -- not as an empty directory, not as
    a file. It must be absolute, its parent must exist (no tree of parents is conjured), and once
    junctions are resolved it must neither be, contain nor sit inside the code, data, artifact or
    frozen-input roots, nor any of ``declared_inputs`` (:func:`declared_input_roots`): the historical
    flights and data directories the base config names, which can resolve to roots outside those
    four. A declared input that does not exist yet is protected too -- a fresh child of one, or the
    path itself, is refused. The directory is made with ``exist_ok=False`` so two launches racing
    for one name cannot both succeed; everything is checked BEFORE it is made.
    """
    roots = {"the code root": _existing_directory(code_root, "the code root"),
             "the data repository": _existing_directory(data_repo_root, "the data repository"),
             "the artifact root": _existing_directory(artifact_root, "the artifact root"),
             "the frozen inputs": _existing_directory(frozen_inputs, "the frozen inputs")}
    target = Path(run_root)
    require(target.is_absolute(), f"the run root must be an absolute path, got {str(run_root)!r}")
    require(target.parent.is_dir(), f"the parent of the run root, {target.parent}, does not exist")
    require(not target.exists() and not target.is_symlink(),
            f"the run root {target} already exists. A pilot writes into a directory it created "
            "itself and never into an existing one, however empty")
    paths.require_disjoint_run_root(target, roots)
    require(isinstance(declared_inputs, dict),
            "declared_inputs must be the {name: location} mapping of declared_input_roots()")
    clashes = sorted(name for name, location in declared_inputs.items()
                     if paths.overlapping(target, location))
    require(not clashes,
            f"the run root {target.name!r} overlaps protected declared inputs {clashes} once "
            "junctions are resolved. Historical flights and data directories are read-only, and "
            "a fresh directory inside one is still inside it")
    try:
        target.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"the run root {target} was created by someone else first") from error
    return PilotRoots(code_root=roots["the code root"], data_repo_root=roots["the data repository"],
                      artifact_root=roots["the artifact root"], run_root=target,
                      frozen_inputs=roots["the frozen inputs"])


@dataclass(frozen=True)
class PilotContext:
    """The slice of ``RoutesContext`` the services touch, over the pilot's mixed roots.

    Interface: ``config``, ``config_path``, ``repository_root``, ``artifact_root``, ``run_root``,
    ``revision``, ``campaign_id``, ``path``, ``artifact``, ``repository``, ``condition_spec``.
    ``repository_root`` is the DATA repository, the root ``repository(logical)`` resolves declared
    data paths under; the one exception is ``config.dplm.pins``, which is a file of this worktree.
    """

    config: dict
    config_path: Path
    roots: PilotRoots
    revision: str
    campaign_id: str

    @property
    def repository_root(self):
        return self.roots.data_repo_root

    @property
    def artifact_root(self):
        return self.roots.artifact_root

    @property
    def run_root(self):
        return self.roots.run_root

    def path(self, *parts):
        """A path under the run root: the one place anything is written."""
        return self.roots.run_root.joinpath(*[str(part) for part in parts])

    def artifact(self, logical):
        return paths.resolve_under(self.roots.artifact_root, str(logical))

    def repository(self, logical):
        pins = (self.config.get("dplm") or {}).get("pins")
        root = self.roots.code_root if pins and str(logical) == str(pins) else \
            self.roots.data_repo_root
        return paths.resolve_under(root, str(logical))

    def condition_spec(self):
        block = self.config["conditions"]
        return ConditionSpec(assays=tuple(block["assays"]),
                             bin_dropout=float(block["bin_dropout"]),
                             assay_dropout=float(block["assay_dropout"]))

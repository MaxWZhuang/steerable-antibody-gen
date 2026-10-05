"""Which checkpoint each stage consumes, and what a frozen recipe can actually replay.

Torch-free and pure: path and stage-record logic only, so the campaign's orchestration
tests can exercise it without a model. Three defects this exists to close (audit
HR-01, HR-03, HR-07, HR-08):

* **A missing trained checkpoint used to become ``None``**, and every consumer read
  ``None`` as "score the frozen backbone". ``load_for_scoring`` refuses a *named but
  absent* checkpoint, but the call sites converted absent to ``None`` before the guard, so
  a fresh revision root ran the whole downstream on an untrained model and passed
  ``verify``, which checks completeness. :func:`require_selected_checkpoint` stops the
  stage instead; the backbone comparator is reached only by an explicit ``None``.
* **T3 and T4 optimised and nothing inherited the result.** :func:`resolve_lineage` makes
  the path explicit: priors starts from ``bakeoff/<route>/best.pt``; post-training starts
  from ``priors/<route>/distilled.pt`` when priors completed with one and otherwise from
  the bake-off, with the fallback **stated in the record**; decoding and the report read
  ``posttrain/<selected method>/last.pt``.
* **The confirmation recorded a recipe it did not apply.** :func:`recipe_replay_plan`
  classifies every element of a frozen recipe as replayed or not, and
  :func:`require_replayable` refuses (before any marker, opening or training file exists)
  when anything named would not be applied. This module does **not** implement replay of a
  qualified teacher or of a post-training method; it makes the gap fail closed.
"""
from __future__ import annotations

from pathlib import Path

from . import her2_support_paths as paths
from .her2_runtime import require


class LineageError(ValueError):
    """A required checkpoint or recorded selection is absent or ambiguous."""


class RecipeNotReplayable(LineageError):
    """The frozen recipe names something the confirmation training would not apply."""


SOURCE_BAKEOFF = "bakeoff_best"
SOURCE_PRIORS = "priors_distilled"
SOURCE_POSTTRAIN = "posttrain_selected"

#: Stages that read the bake-off winner.
BAKEOFF_STAGES = ("route_gate", "controls", "priors")
#: Stages that read the post-trained model.
FINAL_STAGES = ("decoding", "report")


def require_selected_checkpoint(path, *, stage, route, role="selected checkpoint"):
    """Return ``path`` or stop the stage. Never converts an absent file to ``None``."""
    target = Path(path)
    if not target.is_file():
        raise LineageError(
            f"{stage}: the {role} for route {route!r} is absent ({target}). This stage will not "
            "run on a fresh backbone under the name of a trained model; run the stage that "
            "produces it, or pass the frozen zero-shot comparator explicitly.")
    return target


def _record(source, path, *, stage, route, reason, fallback=False, considered=()):
    return {"stage": str(stage), "route": str(route), "source": source, "path": str(path),
            "sha256": paths.sha256_file(path), "reason": reason, "fallback": bool(fallback),
            "considered": [dict(item) for item in considered]}


def _lineage_require(condition, message):
    if not condition:
        raise LineageError(message)


def completed_document(document, *, stage, name, route):
    """The ``<name>.json`` of a completed stage, or a refusal. Never an empty stand-in."""
    _lineage_require(isinstance(document, dict),
                     f"{stage}: {name}.json is absent or malformed although its stage record says "
                     "completed; nothing is inferred from a missing document.")
    _lineage_require(document.get("route") in (None, route),
                     f"{stage}: {name}.json was produced for route {document.get('route')!r}, not "
                     f"{route!r}")
    return document


def distilled_checkpoints(priors_document, *, route, path_for):
    """Distilled checkpoints the (completed) priors document names for ``route``.

    Every named checkpoint must be the route's ``priors/<route>/distilled.pt`` and exist, and
    every qualified teacher must have been distilled and every distilled teacher must have been
    qualified; anything else raises instead of being filtered out and reported as "no teacher
    qualified". An absent ``distillations`` key means none; a present one must be a mapping.
    """
    document = priors_document if isinstance(priors_document, dict) else {}
    distillations = document.get("distillations", {})
    _lineage_require(isinstance(distillations, dict),
                     "priors.json 'distillations' is not a mapping")
    expected = Path(path_for("priors", route, "distilled.pt"))
    found = []
    for name, block in sorted(distillations.items()):
        checkpoint = block.get("checkpoint") if isinstance(block, dict) else None
        _lineage_require(bool(checkpoint),
                         f"priors.json distillation {name!r} names no checkpoint")
        _lineage_require(Path(checkpoint) == expected,
                         f"priors.json distillation {name!r} names {checkpoint}, not this run's "
                         f"route checkpoint {expected}")
        _lineage_require(expected.is_file(),
                         f"priors.json distillation {name!r} names {expected}, which is absent")
        found.append((name, expected))
    # a completed priors document must carry its teacher decisions; only a validated explicit
    # "no teacher qualified" outcome (a mapping whose every decision says so) may fall back
    teachers = document.get("teachers")
    _lineage_require(isinstance(teachers, dict),
                     "priors.json has no 'teachers' decision mapping although the stage record "
                     "says completed; 'no teacher qualified' is never inferred from its absence")
    for name, block in teachers.items():
        _lineage_require(isinstance(block, dict) and isinstance(block.get("qualified"), bool),
                         f"priors.json teacher decision {name!r} is malformed: it must be a "
                         "mapping with a boolean 'qualified'")
    qualified = sorted(name for name, block in teachers.items() if block["qualified"])
    undistilled = [name for name in qualified if name not in distillations]
    _lineage_require(not undistilled,
                     f"priors.json qualifies teacher(s) {undistilled} but records no "
                     "distillation for them")
    unqualified = sorted(str(name) for name in distillations
                         if name not in teachers or not teachers[name]["qualified"])
    _lineage_require(not unqualified,
                     f"priors.json records a distillation for teacher(s) {unqualified} that it "
                     "does not record as qualified (rejected or unrecorded); post-training will "
                     "not start from a checkpoint distilled from an unqualified teacher")
    return found


def selected_method(posttrain_document, posttrain_record=None):
    """The post-training method the development comparison selected, or a refusal."""
    document = dict(posttrain_document or {})
    best = (document.get("selection") or {}).get("best")
    if best is None and posttrain_record:
        best = posttrain_record.get("selected_method")
    if not best:
        raise LineageError(
            "posttrain.json names no selected method (selection.best is empty), so no "
            "post-trained checkpoint can be resolved for the stages that follow.")
    return str(best)


def resolve_lineage(stage, *, route, path_for, record_for, document_for):
    """The checkpoint ``stage`` consumes, resolved explicitly. A record, not a path.

    ``path_for(*parts)`` is the run-root path builder, ``record_for(stage)`` returns a
    stage record or ``None``, and ``document_for(name)`` returns the run-root
    ``<name>.json`` document or ``None``.
    """
    stage = str(stage)
    if stage in BAKEOFF_STAGES:
        path = require_selected_checkpoint(path_for("bakeoff", route, "best.pt"), stage=stage,
                                           route=route, role="bake-off best checkpoint")
        return _record(SOURCE_BAKEOFF, path, stage=stage, route=route,
                       reason="the bake-off winner is the start of the selection chain")
    if stage == "posttrain":
        considered = []
        priors_record = record_for("priors")
        if (priors_record or {}).get("status") == "completed":
            priors_document = completed_document(document_for("priors"), stage=stage,
                                                 name="priors", route=route)
            found = distilled_checkpoints(priors_document, route=route, path_for=path_for)
            considered.append({"source": SOURCE_PRIORS, "candidates": [name for name, _ in found]})
            if len(found) > 1:
                raise LineageError(
                    f"posttrain: priors distilled more than one teacher ({[n for n, _ in found]}) "
                    "into one checkpoint path; the start is ambiguous and is not guessed.")
            if found:
                return _record(SOURCE_PRIORS, found[0][1], stage=stage, route=route,
                               reason=f"priors completed and distilled teacher {found[0][0]!r}",
                               considered=considered)
            reason = ("priors completed without a distilled checkpoint (no teacher qualified), so "
                      "post-training starts from the bake-off best")
        else:
            considered.append({"source": SOURCE_PRIORS, "candidates": [],
                               "priors_status": (priors_record or {}).get("status", "not_run")})
            reason = ("the priors stage has not completed, so post-training starts from the "
                      "bake-off best")
        path = require_selected_checkpoint(path_for("bakeoff", route, "best.pt"), stage=stage,
                                           route=route, role="bake-off best checkpoint")
        return _record(SOURCE_BAKEOFF, path, stage=stage, route=route, reason=reason,
                       fallback=True, considered=considered)
    if stage in FINAL_STAGES:
        posttrain_record = record_for("posttrain")
        if (posttrain_record or {}).get("status") != "completed":
            raise LineageError(
                f"{stage}: the posttrain stage is {(posttrain_record or {}).get('status', 'not_run')!r}. "
                "Decoding and the report read the post-trained model; they do not fall back to "
                "the bake-off weights under its name.")
        document = completed_document(document_for("posttrain"), stage=stage, name="posttrain",
                                      route=route)
        method = selected_method(document, posttrain_record)
        path = require_selected_checkpoint(path_for("posttrain", method, "last.pt"), stage=stage,
                                           route=route, role=f"post-trained checkpoint ({method})")
        return _record(SOURCE_POSTTRAIN, path, stage=stage, route=route,
                       reason=f"the post-training method {method!r} selected on development",
                       considered=[{"source": SOURCE_POSTTRAIN, "method": method}])
    raise LineageError(f"No checkpoint lineage is defined for stage {stage!r}")


def teacher_decisions(priors_document):
    """``{teacher: qualified}`` from ``priors.json``. Refuses the stage record's name list.

    The priors STAGE RECORD carries ``teachers`` as a sorted list of names; the per-teacher
    documents with their ``qualified`` flags are in ``priors.json``. Reading the list as a
    mapping raised ``ValueError`` inside ``dict(...)`` and made the recipe unassemblable.
    """
    teachers = (priors_document or {}).get("teachers")
    require(isinstance(teachers, dict),
            f"priors.json 'teachers' must be a mapping of teacher name to its decision, got "
            f"{type(teachers).__name__}. The priors stage record holds only a list of names; the "
            "per-teacher qualification is in priors.json.")
    out = {}
    for name, block in sorted(teachers.items()):
        require(isinstance(block, dict),
                f"priors.json teacher {name!r} is a {type(block).__name__}, not a decision "
                "document")
        require(isinstance(block.get("qualified"), bool),
                f"priors.json teacher {name!r} has qualified={block.get('qualified')!r}; it must "
                "be an explicit boolean (a missing or non-boolean decision is not coerced)")
        out[str(name)] = block["qualified"]
    return out


def recipe_replay_plan(recipe):
    """Classify each element of a frozen recipe as replayed, not replayed, or not applicable.

    What the confirmation training (``services.run_confirmation``) applies today: the route
    and the base curriculum, trained from scratch for each seed. It does not replay a
    qualified teacher's distillation, a post-training method, or a decoding strategy --
    ``open_reserved`` scores the panel with the fixed declared-order functional and consumes
    no strategy. So a recipe naming any of them is not replayable, and this says so.
    """
    elements = {"route": "replayed", "base_curriculum": "replayed"}
    for name, qualified in sorted(dict(recipe.get("teachers") or {}).items()):
        elements[f"teacher::{name}"] = "not_replayed" if qualified else "not_applicable"
    method = recipe.get("posttrain_method")
    elements["posttrain_method"] = "not_replayed" if method else "not_applicable"
    strategy = recipe.get("decoding_strategy")
    elements["decoding_strategy"] = "not_replayed" if strategy else "not_applicable"
    unreplayed = sorted(name for name, state in elements.items() if state == "not_replayed")
    return {"elements": elements, "unreplayed": unreplayed, "replayable": not unreplayed,
            "named": {"posttrain_method": method, "decoding_strategy": strategy,
                      "qualified_teachers": sorted(name for name, q in dict(
                          recipe.get("teachers") or {}).items() if q)},
            "gap": ("Replay of a qualified teacher's distillation and of a post-training method "
                    "from the frozen recipe is not implemented, and the confirmation applies no "
                    "decoding strategy. A recipe that names one is refused rather than recorded "
                    "as though it were executed.")}


def require_replayable(recipe, *, where):
    """Raise before any side effect when the recipe names something that would not be applied."""
    plan = recipe_replay_plan(recipe)
    if not plan["replayable"]:
        raise RecipeNotReplayable(
            f"{where}: the frozen recipe names {plan['unreplayed']}, which the confirmation "
            "training does not apply. Nothing was frozen, opened or trained. Recorded gap: "
            f"{plan['gap']}")
    return plan


def consume_frozen_recipe(path, *, freeze_record=None):
    """Read an existing ``frozen_recipe.json`` and check its identity. Never re-assembles it."""
    target = Path(path)
    if not target.is_file():
        raise LineageError(
            f"{target} is absent. The confirmation consumes the recipe the confirmation_freeze "
            "stage froze; it is never silently re-assembled from the current stage records.")
    recipe = paths.read_json(target)
    recorded = recipe.get("sha256")
    recomputed = paths.digest_document({k: v for k, v in recipe.items() if k != "sha256"})
    require(recorded == recomputed,
            f"{target} records digest {recorded} and its content hashes to {recomputed}; the "
            "frozen recipe was edited after it was frozen.")
    if freeze_record is not None:
        require(freeze_record.get("recipe_sha256") == recorded,
                "The confirmation_freeze stage record names recipe "
                f"{freeze_record.get('recipe_sha256')} and the file on disk is {recorded}.")
    return recipe

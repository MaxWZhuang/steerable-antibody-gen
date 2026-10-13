"""Qualification runner: preflight, exact budget, run-root reservation, execution and the finite-JSON report.

The order of events is fixed and everything before the run root is reserved is side-effect free:

1. :func:`preflight` reads ONLY the explicit config and manifest, checks the split, withholds the sealed test
   partition, applies the shared joint-length cap, plans every masked diagnostic and prices all of it.
2. :func:`require_within_budget` rejects an over-budget plan. No backend factory has been called and no
   directory exists.
3. :func:`validate_run_root` proves the fresh output path is disjoint from the code checkout and from the
   input DIRECTORIES (config, manifest, records, upstream source, checkpoint) with junctions resolved; only
   then is the run root reserved with an atomic ``mkdir(exist_ok=False)``.
4. The backend factory is called (and only if a forward was planned), features are extracted one view at a
   time, the train-only ridge is fitted per arm, then the masked diagnostics run under ``no_grad``.
   :class:`CountingBackend` confirms the planned forward count.

``--admission-only`` stops after step 1 and prints the plan; it imports no torch and writes nothing.

:func:`run_with_backend` is the internal injection seam for software tests and its status says SOFTWARE TEST.
:func:`run_qualification` is the only production entry: explicit local pinned-MINT paths, ``float32`` with
``mmap``, and "released execution" is reported only after a forward pass on the pinned release completed.
Nothing here trains, distils, fuses, tunes a threshold or declares a scientific pass.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from smallAntibodyGen.experiments.interaction_qualification.arms import (
    ARM_NAMES,
    ArmFeatures,
    FullViewMemo,
    LengthAdmission,
    admit_by_joint_length,
    arm_report,
    assemble_matrix,
    common_ids,
    extract_arm_features,
    planned_feature_forwards,
)
from smallAntibodyGen.experiments.interaction_qualification.config import (
    QualificationConfig,
    SchemaError,
    decode_utf8,
    parse_config,
    read_file_bytes,
)
from smallAntibodyGen.experiments.interaction_qualification.diagnostics import (
    PHASE_DIAGNOSTICS,
    PHASE_FEATURES,
    BudgetMismatch,
    CountingBackend,
    DiagnosticPlan,
    plan_diagnostics,
    plan_document,
    require_within_budget,
    run_pair_tasks,
    run_singleton_tasks,
    summarize_pairs,
    summarize_singletons,
)
from smallAntibodyGen.experiments.interaction_qualification.probes import (
    evaluate,
    fit_ridge,
    predict_by_id,
    scored_rows,
    training_counts,
)
from smallAntibodyGen.experiments.interaction_qualification.records import (
    Panel,
    Record,
    load_manifest,
    support_counts,
    support_report,
)

REPORT_SCHEMA = "aa-qualification-report/1"
STATUS_SCHEMA = "aa-qualification-status/1"
ADMISSION_SCHEMA = "aa-admission/1"
SOFTWARE_TEST = "software_test"
PINNED_RELEASE = "pinned_release"
ORIGINS: Tuple[str, ...] = (SOFTWARE_TEST, PINNED_RELEASE)
RELEASE_LABEL = "pinned release"  # MintBackend.describe()["release"] for the pinned release only
RUN_FILES: Tuple[str, ...] = ("status.json", "plan.json", "report.json")

SCIENTIFIC_STATUS: Mapping[str, str] = MappingProxyType(
    {
        "antigen_specificity_of_joint_gap": "unestablished",
        "homology_leakage": "unverified",
        "upstream_pretraining_exclusion": "not_audited",
        "generative_success": "unestablished",
        "scientific_qualification": "not_assessed",
    }
)

LIMITATIONS: Tuple[str, ...] = (
    "the measured flag, sources, assays and split basis are DECLARED by the manifest author and are not independently verified",
    "split checks are exact-string and declared-group checks; homology separation is not audited",
    "exclusion of any record from the upstream MINT pretraining data is not audited",
    "the ridge probes are frozen-feature readouts with one fixed alpha for every arm; they are not a teacher, a distillation target or a conditioning input, and the arms differ in width",
    "joint-minus-antibody-only differences mix antigen information with the native token-dropout rescaling that changes with the unmasked antigen length; no antigen-specificity claim is made",
    "several mask counts, geometries and reveal orders of one molecule are repeated views, not independent molecules",
    "a local gap contrast is a sum over 1-3 changed sites between two exact measurements of one source and assay; it identifies no epistasis and says nothing about generative success",
    "pair screening is quadratic within (partition, source, assay, antigen) groups: bounded pilot qualification, not a scalable pipeline",
    "no thresholds, p-values or confidence intervals are computed and there is no pass/fail science flag",
    "nothing here trains, distils or fuses a model; the existing trainable multiscale DPLM pilot is untouched",
    "a software-test run exercises the code path only and says nothing about the released model",
)


# --- preflight --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Preflight:
    """Everything decided before a model exists. Immutable."""

    config: QualificationConfig
    config_path: Path
    config_file_sha256: str
    manifest_path: Path
    manifest_file_sha256: str
    records_path: Path
    panel: Panel
    sealed_test_records: int
    length: LengthAdmission
    plan: DiagnosticPlan

    @property
    def eligible(self) -> Tuple[Record, ...]:
        return self.length.eligible


def preflight(config_path: Union[str, Path], manifest_path: Union[str, Path]) -> Preflight:
    """Read the explicit config and manifest, audit the split, withhold sealed test, cap length, plan and price."""
    config_path, manifest_path = Path(config_path), Path(manifest_path)
    config_bytes = read_file_bytes(config_path, "config")
    config = parse_config(decode_utf8(config_bytes, "config"))
    before = hashlib.sha256(read_file_bytes(manifest_path, "manifest")).hexdigest()
    panel = load_manifest(manifest_path)
    after = hashlib.sha256(read_file_bytes(manifest_path, "manifest")).hexdigest()
    if before != after:
        raise SchemaError("the manifest changed while it was being read")
    records_path = Path(panel.manifest.records_file)
    if not records_path.is_absolute():
        records_path = manifest_path.parent / records_path
    active = [record for record in panel.records if config.evaluate_test or record.partition != "test"]
    length = admit_by_joint_length(active, config.max_tokens)
    plan = plan_diagnostics(
        length.eligible,
        config.diagnostics,
        planned_feature_forwards(length.eligible),
        allow_test=config.evaluate_test,
    )
    return Preflight(
        config=config,
        config_path=config_path,
        config_file_sha256=hashlib.sha256(config_bytes).hexdigest(),
        manifest_path=manifest_path,
        manifest_file_sha256=after,
        records_path=records_path,
        panel=panel,
        sealed_test_records=len(panel.records) - len(active),
        length=length,
        plan=plan,
    )


def require_budget(pf: Preflight) -> None:
    require_within_budget(pf.plan.budget)


# --- report pieces ------------------------------------------------------------------------------------


def canonical_text(document: Any) -> str:
    """Sorted-key, ASCII, finite (``allow_nan=False``) JSON with a single trailing LF."""
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> Dict[str, Any]:
    """Interpreter and library versions from package metadata; nothing is imported to read them."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "torch": _version("torch"),
    }


def _manifest_block(pf: Preflight) -> Dict[str, Any]:
    manifest = pf.panel.manifest
    return {
        "manifest_file_sha256": pf.manifest_file_sha256,
        "records_file_sha256": pf.panel.records_file_sha256,
        "panel_content_sha256": pf.panel.content_sha256,
        "split_regime": manifest.split_regime,
        "split_basis": manifest.split_basis,
        "split_basis_status": "declared",
        "sources": [
            {
                "source": declaration.source,
                "citation": declaration.citation,
                "assays": [{"assay": assay, "declaration": text} for assay, text in declaration.assays],
            }
            for declaration in manifest.sources
        ],
    }


def _plan_support(pf: Preflight) -> Dict[str, Any]:
    by_id = pf.panel.by_id()
    plan = pf.plan
    singleton = [by_id[record_id] for task in plan.singleton_tasks for record_id in task.record_ids]
    pair_ids = sorted(
        {rid for task in plan.pair_tasks for c in task.contrasts for rid in (c.record_id_a, c.record_id_b)}
    )
    return {
        "singleton": {
            "tasks": len(plan.singleton_tasks),
            "cells": plan.n_cells,
            "records": support_counts(singleton),
        },
        "pair": {
            "tasks": len(plan.pair_tasks),
            "contrasts": sum(len(task.contrasts) for task in plan.pair_tasks),
            "records": support_counts([by_id[rid] for rid in pair_ids]),
        },
    }


def _common_report(pf: Preflight) -> Dict[str, Any]:
    plan, config = pf.plan, pf.config
    document = plan_document(plan)
    return {
        "config": config.as_dict(),
        "config_content_sha256": config.content_sha256(),
        "config_file_sha256": pf.config_file_sha256,
        "manifest": _manifest_block(pf),
        "support_report": support_report(pf.panel),
        "sealed_test": {"evaluate_test": config.evaluate_test, "records_withheld": pf.sealed_test_records},
        "shared_length_admission": {
            "max_tokens": pf.length.max_tokens,
            "n_eligible": len(pf.length.eligible),
            "n_excluded": len(pf.length.excluded),
            "excluded": [{"record_id": rid, "joint_serialized_length": n} for rid, n in pf.length.excluded],
        },
        "active_support": support_counts(pf.eligible),
        "masked_support": _plan_support(pf),
        "forward_budget": plan.budget.to_dict(),
        "plan_summary": {
            "plan_sha256": hashlib.sha256(canonical_text(document).encode("ascii")).hexdigest(),
            "singleton_tasks": len(plan.singleton_tasks),
            "singleton_cells": plan.n_cells,
            "pair_tasks": len(plan.pair_tasks),
            "singleton_stats": dict(plan.singleton_stats),
            "pair_stats": dict(plan.pair_stats),
        },
        "scientific_status": dict(SCIENTIFIC_STATUS),
        "limitations": list(LIMITATIONS),
    }


def admission_report(pf: Preflight) -> Dict[str, Any]:
    """The ``--admission-only`` report: the plan and its price, with no backend, no torch and no file written."""
    report: Dict[str, Any] = {"schema": ADMISSION_SCHEMA, "mode": "admission_only"}
    report.update(_common_report(pf))
    report.update(
        files_written=0,
        backend="not constructed: admission reads the config and manifest only",
        runtime=runtime_provenance(),
    )
    return report


# --- run root ---------------------------------------------------------------------------------------


def protected_locations(
    pf: Preflight, code_root: Union[str, Path], extra: Optional[Mapping[str, Union[str, Path]]] = None
) -> Dict[str, Path]:
    protected = {
        "the code root": Path(code_root),
        "the config directory": pf.config_path.parent,
        "the manifest directory": pf.manifest_path.parent,
        "the records directory": pf.records_path.parent,
    }
    protected.update({name: Path(location) for name, location in (extra or {}).items()})
    return protected


def validate_run_root(run_root: Union[str, Path], protected: Mapping[str, Path]) -> Path:
    """Check a fresh output path without creating anything; raise ``ValueError`` on any problem.

    Every protected location must already exist (the disjointness helper skips a missing one, so a typo must
    not read as "no overlap"). The run root must be absolute, must not exist (not as an empty directory, not as
    a file or link), must have an existing parent, and once junctions are resolved must neither be, contain
    nor sit inside any protected DIRECTORY.
    """
    from smallAntibodyGen.experiments.her2_support_paths import require_disjoint_run_root

    target = Path(run_root)
    if not target.is_absolute():
        raise ValueError(f"the run root must be an absolute path, got {str(run_root)!r}")
    if not target.parent.is_dir():
        raise ValueError(f"the parent of the run root, {target.parent}, does not exist")
    if target.exists() or target.is_symlink():
        raise ValueError(f"the run root {target} already exists; a run writes only into a directory it created")
    missing = sorted(name for name, location in protected.items() if not Path(location).exists())
    if missing:
        raise ValueError(f"required input location(s) do not exist: {missing}")
    require_disjoint_run_root(target, {name: str(location) for name, location in protected.items()})
    return target


def reserve_run_root(target: Path) -> None:
    try:
        target.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise ValueError(f"the run root {target} was created by someone else first") from None


def write_json_lf(path: Path, document: Any) -> str:
    """Finite JSON with LF newlines, written to an exclusive temporary beside ``path`` and renamed over it."""
    data = canonical_text(document).encode("utf-8")  # raises on NaN/inf before any file exists
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "xb") as stream:
        stream.write(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


# --- probes -------------------------------------------------------------------------------------------


def run_probes(
    eligible: Sequence[Record], arms: Optional[Mapping[str, ArmFeatures]], config: QualificationConfig
) -> Dict[str, Any]:
    """Train-only fixed-alpha ridge per arm, scored on train, development and (only if unsealed) test."""
    evaluated = ("train", "development") + (("test",) if config.evaluate_test else ())
    train = [record for record in eligible if record.partition == "train"]
    out: Dict[str, Any] = {
        "ridge_alpha": config.ridge_alpha,
        "n_eligible_records": len(eligible),
        "n_train_records": len(train),
        "evaluated_partitions": list(evaluated),
        "train_metrics_note": "train rows are in-sample; development and (if unsealed) test rows are held out",
        "tuning": "none: no held-out tuning, no source features, one fixed alpha for every arm",
        "arms": {},
    }
    if arms is None:
        out.update(status="no_support", reason="no eligible records, so no features were extracted", arm_report=None)
        return out
    out["arm_report"] = arm_report(arms)
    if common_ids(arms) != tuple(sorted(record.record_id for record in eligible)):
        raise ValueError("the arms do not cover exactly the eligible record ids")
    if not train:
        out.update(status="no_training_support", reason="no eligible training records, so no probe was fitted")
        return out
    eval_records = [record for record in eligible if record.partition in evaluated]
    eval_ids = [record.record_id for record in eval_records]
    labels = np.array([record.pkd for record in train], dtype=np.float64)
    for name in ARM_NAMES:
        features = arms[name]
        fit = fit_ridge(assemble_matrix(features, [record.record_id for record in train]), labels, config.ridge_alpha)
        predictions = predict_by_id(fit, eval_ids, assemble_matrix(features, eval_ids))
        result = evaluate(
            scored_rows(eval_records, predictions), train_mean=fit.y_mean, train_counts=training_counts(train)
        )
        out["arms"][name] = {
            "width": features.dimension,
            "fit": fit.to_dict(include_arrays=True),
            "predictions": {record_id: predictions[record_id] for record_id in sorted(predictions)},
            "groups": [group.to_dict() for group in result.groups],
            "pooled_secondary": [group.to_dict() for group in result.pooled],
        }
    out["status"] = "fitted"
    return out


# --- execution ------------------------------------------------------------------------------------------


class _RunState:
    def __init__(self) -> None:
        self.counting: Optional[CountingBackend] = None
        self.plan_sha256: Optional[str] = None
        self.report_sha256: Optional[str] = None
        self.backend_constructed = False


def _describe(backend: Any) -> Dict[str, Any]:
    describe = getattr(backend, "describe", None)
    return dict(describe()) if callable(describe) else {"backend": type(backend).__name__}


def _execution_block(origin: str, state: _RunState, phase: str) -> Dict[str, Any]:
    forwards = 0 if state.counting is None else state.counting.total_calls
    if origin == PINNED_RELEASE:
        done = phase == "completed" and forwards > 0
        released = "executed" if done else ("incomplete" if phase == "error" and forwards > 0 else "not_run")
        if done:
            text = f"pinned MINT release: {forwards} forward passes completed on the explicit local source and checkpoint"
        elif phase == "completed":
            text = "pinned release path selected but no forward was planned, so no model was constructed or run"
        else:
            text = "pinned release path selected; no completed release forward is claimed"
    else:
        done = False
        released = "not_run"
        text = "SOFTWARE TEST: an injected software backend was used; the pinned release was NOT run"
    return {
        "origin": origin,
        "software_test": origin != PINNED_RELEASE,
        "backend_constructed": state.backend_constructed,
        "forward_calls": forwards,
        "released_weights_forward_completed": done,
        "released_execution": released,
        "description": text,
    }


def _status_document(pf: Preflight, origin: str, state: _RunState, phase: str, error: Optional[BaseException] = None) -> Dict[str, Any]:
    document: Dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "state": phase,
        "qualification_id": pf.config.qualification_id,
        "config_content_sha256": pf.config.content_sha256(),
        "panel_content_sha256": pf.panel.content_sha256,
        "execution": _execution_block(origin, state, phase),
        "forward_budget": pf.plan.budget.to_dict(),
        "scientific_status": dict(SCIENTIFIC_STATUS),
        "plan_sha256": state.plan_sha256,
        "report_sha256": state.report_sha256,
        "files": list(RUN_FILES),
    }
    if error is not None:
        document["error"] = {"type": type(error).__name__, "message": str(error)}
    return document


def _execute(pf: Preflight, factory: Callable[[], Any], origin: str, state: _RunState) -> Dict[str, Any]:
    plan, config = pf.plan, pf.config
    budget = plan.budget
    backend: Any = None
    backend_block: Dict[str, Any] = {"constructed": False, "reason": "no forward was planned"}
    if budget.total > 0:
        backend = factory()
        state.backend_constructed = True
        described = _describe(backend)
        is_release = described.get("release") == RELEASE_LABEL
        if origin == PINNED_RELEASE and not is_release:
            raise ValueError("the production loader did not return the pinned release")
        if origin == SOFTWARE_TEST and is_release:
            raise ValueError("run_with_backend is the software-test seam and must not be handed the pinned release")
        backend_block = {"constructed": True, "describe": described}
    counting = CountingBackend(backend, {PHASE_FEATURES: budget.feature, PHASE_DIAGNOSTICS: budget.diagnostics})
    state.counting = counting
    eligible = pf.eligible
    arms: Optional[Dict[str, ArmFeatures]] = None
    if eligible:
        with counting.phase(PHASE_FEATURES):
            memo = FullViewMemo(counting)
            arms = extract_arm_features(eligible, memo)
        if memo.forward_calls != budget.feature:
            raise BudgetMismatch(f"feature extraction ran {memo.forward_calls} forwards, planned {budget.feature}")
    probes = run_probes(eligible, arms, config)
    with counting.phase(PHASE_DIAGNOSTICS):
        cells = run_singleton_tasks(counting, plan.singleton_tasks)
        pair_results = run_pair_tasks(counting, plan.pair_tasks)
    counting.verify()
    by_id = pf.panel.by_id()
    report: Dict[str, Any] = {"schema": REPORT_SCHEMA, "mode": "full_run"}
    report.update(_common_report(pf))
    report.update(
        execution=_execution_block(origin, state, "completed"),
        backend=backend_block,
        runtime=runtime_provenance(),
        forward_budget_actual={
            "features": counting.calls[PHASE_FEATURES],
            "diagnostics": counting.calls[PHASE_DIAGNOSTICS],
            "total": counting.total_calls,
            "matches_plan": True,
        },
        probes=probes,
        diagnostics={
            "singleton": {
                "status": "computed" if plan.singleton_tasks else "no_support",
                "cells": cells,
                "summary": summarize_singletons(cells),
            },
            "pair": {
                "status": (
                    "disabled"
                    if not config.diagnostics.local_gap.enabled
                    else "computed" if plan.pair_tasks else "no_support"
                ),
                "tasks": pair_results,
                "summary": summarize_pairs(pair_results, plan.pair_tasks, by_id, plan.kappa),
            },
        },
    )
    return report


def run_with_backend(
    pf: Preflight,
    backend_factory: Callable[[], Any],
    *,
    run_root: Union[str, Path],
    code_root: Union[str, Path],
    origin: str = SOFTWARE_TEST,
    extra_protected: Optional[Mapping[str, Union[str, Path]]] = None,
) -> Dict[str, Any]:
    """Run the whole qualification on ``backend_factory()``; the internal seam software tests use.

    The budget and the run root are checked first, and the factory is called only after the run root is
    reserved and only if a forward was planned. ``origin`` defaults to ``software_test`` and the status then
    says so; only :func:`run_qualification` passes ``pinned_release``.
    """
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
    require_budget(pf)
    target = validate_run_root(run_root, protected_locations(pf, code_root, extra_protected))
    reserve_run_root(target)
    state = _RunState()
    try:
        write_json_lf(target / "status.json", _status_document(pf, origin, state, "running"))
        plan_doc = plan_document(pf.plan)
        state.plan_sha256 = write_json_lf(target / "plan.json", plan_doc)
        report = _execute(pf, backend_factory, origin, state)
        state.report_sha256 = write_json_lf(target / "report.json", report)
        status = _status_document(pf, origin, state, "completed")
        write_json_lf(target / "status.json", status)
    except BaseException as error:
        try:
            write_json_lf(target / "status.json", _status_document(pf, origin, state, "error", error))
        except (OSError, ValueError):
            pass
        raise
    return {"run_root": str(target), "status": status, "report": report}


def _release_factory(source: Path, checkpoint: Path, device: str) -> Callable[[], Any]:
    def factory() -> Any:
        import torch

        from smallAntibodyGen.models.antibody_antigen.mint import load_pinned_mint

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda was requested but CUDA is not available")
        return load_pinned_mint(source, checkpoint, device=device, dtype=torch.float32, mmap=True)

    return factory


def run_qualification(
    config_path: Union[str, Path],
    manifest_path: Union[str, Path],
    run_root: Union[str, Path],
    *,
    mint_source: Union[str, Path],
    mint_checkpoint: Union[str, Path],
    device: str = "cpu",
    code_root: Union[str, Path],
) -> Dict[str, Any]:
    """The production entry: explicit local pinned-MINT source and checkpoint, no discovery, no download."""
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    source, checkpoint = Path(mint_source), Path(mint_checkpoint)
    if not source.is_dir():
        raise ValueError(f"--mint-source must be an existing local directory, got {str(mint_source)!r}")
    if not checkpoint.is_file():
        raise ValueError(f"--mint-checkpoint must be an existing local file, got {str(mint_checkpoint)!r}")
    pf = preflight(config_path, manifest_path)
    return run_with_backend(
        pf,
        _release_factory(source, checkpoint, device),
        run_root=run_root,
        code_root=code_root,
        origin=PINNED_RELEASE,
        extra_protected={"the MINT source": source, "the MINT checkpoint directory": checkpoint.parent},
    )

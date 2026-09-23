"""The complete result matrix, the contrasts, the failure ledger, and verification.

Two rules shape everything here.

**Nothing is omitted.** Every declared row appears with a status: completed,
stopped by the gate, incomplete, failed, deferred, missing. A historical path
whose early checkpoints do not exist is marked *missing* and is never
reconstructed. A table shorter than the queue is a defect, and
:func:`verify` checks exactly that.

**Completion is not inferred from a launch.** A running worker, a held lock and
advancing journals say the flight is in progress. The flight is complete only
when every required stage has completed and every queued row has a terminal
status; those are separate fields and the report prints both.

The prose passes the inherited forbidden-claim guard, extended with this
flight's own refusals: no global winner, no affinity claim, no equivalence read
out of a nonsignificant three-seed interval, and no "always better" read out of
an absent yield crossover.
"""
from __future__ import annotations

import json

import numpy as np

from . import her2_nf_campaign as campaign
from . import her2_nf_contract as contract
from . import her2_nf_coupling as coupling
from . import her2_nf_metrics as metrics
from . import her2_nf_mixture as mixture_lib
from . import her2_nf_trajectory as trajectory_lib
from . import her2_replay_report as inherited_report
from . import her2_support_paths as paths
from .her2_runtime import require

REPORT_JSON = "report.json"
REPORT_MD = "report/her2-next-flight.md"
RUNS_JSONL = "runs.jsonl"
FAILURE_LEDGER = "failure_ledger.json"
ARTIFACT_MANIFEST = "artifact_manifest.json"

#: Added to the inherited list. These are the affirmative sentences somebody
#: would write later, not the nouns the report legitimately discusses.
EXTRA_FORBIDDEN_CLAIMS = (
    "proves equivalence", "establishes equivalence", "always better",
    "no crossover means", "the coupling causes", "functional interaction confirmed",
    "converged performance", "guaranteed preservation")

#: Words that make a following forbidden phrase a LIMITATION rather than a
#: claim. The report's own limits say "cannot establish converged performance",
#: and a substring filter rejected the report for saying exactly what it is
#: supposed to say. A behavioural check has to read the sentence.
NEGATION_WORDS = frozenset({"cannot", "can't", "not", "never", "no", "nor", "neither",
                            "without", "avoid", "avoids", "refuses", "refuse"})

#: How many words before the phrase are inspected. Wide enough for "cannot
#: establish X", narrow enough that an unrelated "no" earlier in the sentence
#: does not excuse a claim.
NEGATION_WINDOW_WORDS = 4


def claim_occurrences(text, phrase):
    """Every occurrence of ``phrase`` that is ASSERTED rather than denied.

    An occurrence is a claim unless one of the few words immediately before it
    negates it. ``"cannot establish converged performance"`` is a limitation;
    ``"establishes converged performance"`` is the claim this guard exists to
    stop, and the two differ only in that prefix.
    """
    lowered = str(text).lower()
    needle = str(phrase).lower()
    found, start = [], 0
    while True:
        position = lowered.find(needle, start)
        if position < 0:
            return found
        start = position + len(needle)
        preceding = lowered[max(0, position - 120):position].replace("-", " ").split()
        tail = [word.strip("'\"`,.;:()[]*_") for word in preceding[-NEGATION_WINDOW_WORDS:]]
        if not any(word in NEGATION_WORDS for word in tail):
            found.append({"phrase": phrase, "at": position,
                          "context": lowered[max(0, position - 60):start + 20]})
    return found


def require_no_forbidden_claim(text):
    inherited_report.require_no_forbidden_claim(text)
    found = [entry for phrase in EXTRA_FORBIDDEN_CLAIMS
             for entry in claim_occurrences(text, phrase)]
    require(not found,
            f"the rendered report ASSERTS {[entry['phrase'] for entry in found]}: "
            f"{[entry['context'] for entry in found][:3]}. This flight maps which objective/"
            "preservation combinations retain useful behaviour under stated constraints; it "
            "names no global winner, establishes no equivalence from a nonsignificant three-seed "
            "interval, and reads no 'always better' out of an absent crossover. A LIMITATION "
            "that names one of these phrases under a negation is not a claim and is not "
            "rejected.")
    return True


# ---------------------------------------------------------------------------
# the result matrix
# ---------------------------------------------------------------------------

def registry_entry(registry, entry, checkpoint):
    """The shared registry's block for one (arm, seed, checkpoint), if it has one."""
    entries = (registry or {}).get("entries") or {}
    if entry["block"] == "A":
        return entries.get(f"A_{entry['arm']}@u{int(checkpoint)}_seed{entry['seed']}")
    return entries.get(
        f"B_{entry['regime']}_{entry['arm']}@u{int(checkpoint)}_seed{entry['seed']}")


def checkpoint_matrix(context, state, *, registry=None):
    """One row per (entry, seed, checkpoint), including every non-outcome.

    Availability comes from the SHARED registry wherever that registry has an
    entry. A ``reuse_verified`` trajectory deliberately writes no local journal
    and no local checkpoint, so looking only in THIS run's trajectory directory
    reported every reused historical endpoint as missing while
    ``checkpoint_registry`` -- published in the same document, and the thing the
    audit and the contrasts actually read -- carried those same weights as
    present. Only a checkpoint the registry itself calls missing is reported
    missing, so the genuinely absent early ones keep their status.
    """
    rows = []
    for entry in state["rows"]:
        endpoints = list(entry["endpoints"]) or []
        reached = {int(value) for value in entry["endpoints_reached"]}
        for checkpoint in sorted(set(int(value) for value in entry["checkpoints"]) | set(endpoints)):
            directory = context.path("trajectories", entry["trajectory"])
            registered = registry_entry(registry, entry, checkpoint)
            available = (checkpoint in reached
                         or (directory / f"checkpoint_update{checkpoint}.pt").is_file()
                         or (checkpoint == 0 and (directory / "checkpoint_update0.json").is_file())
                         or (registered or {}).get("status") == "present")
            if checkpoint == 0 and entry["block"] == "A" and entry.get("reuse"):
                available = context.repository_root.joinpath(
                    context.config["block_a"]["parents"][str(entry["seed"])]["logical"]).is_file()
            rows.append({
                "kind": "preference_cell", "entry": entry["arm"], "block": entry["block"],
                "regime": entry["regime"], "trajectory": entry["trajectory"],
                "task": entry["task"], "preservation": entry["preservation"],
                "coefficients": entry["coefficients"], "seed": entry["seed"],
                "checkpoint_update": int(checkpoint),
                "status": entry["observed_status"],
                "checkpoint_status": ("reached" if available else
                                      "missing" if entry["observed_status"] in
                                      ("reuse_verified", "reuse_pending_parity") else
                                      "not_reached"),
                "stop_reason": (entry.get("terminal") or {}).get("stop_reason"),
                "journalled_updates": entry["journalled_updates"],
                "note": entry.get("note")})
    for comparator in campaign.comparator_rows():
        for checkpoint in comparator.get("endpoints", (None,)):
            rows.append({"kind": "comparator", "entry": comparator["id"],
                         "comparator_kind": comparator["kind"],
                         "checkpoint_update": checkpoint, "status": comparator["status"],
                         "checkpoint_status": "declared", "note": comparator["note"]})
    return rows


def paired_contrasts(records, *, contrasts):
    """Declared primary contrasts plus every requested pair, with raw seed differences."""
    out = []
    for entry in contrasts:
        left = {int(seed): value for seed, value in (records.get(entry["left"]) or {}).items()}
        right = {int(seed): value for seed, value in (records.get(entry["right"]) or {}).items()}
        seeds = sorted(set(left) & set(right))
        if len(seeds) < 2:
            out.append({**entry, "available": False, "paired_seeds": seeds,
                        "reason": ("fewer than two paired seeds carry this metric; a contrast is "
                                   "not computed from unpaired values")})
            continue
        differences = [float(left[seed]) - float(right[seed]) for seed in seeds]
        block = metrics.paired_t(np.asarray(differences))
        out.append({**entry, "available": True, "paired_seeds": seeds,
                    "left_values": [float(left[seed]) for seed in seeds],
                    "right_values": [float(right[seed]) for seed in seeds],
                    **block,
                    "reading": ("a nominal interval with df = {}. Failure to exclude zero is not "
                                "equivalence, and a difference between two methods is never "
                                "inferred from one method's significant parent contrast beside "
                                "another's nonsignificant one.".format(block[
                                    "degrees_of_freedom"]))})
    return out


def yield_section(curves, *, budgets=metrics.YIELD_BUDGETS, dense=None):
    """The unique-identity yield grid, the dense diagnostic grid, and every crossover."""
    dense = tuple(dense or metrics.dense_yield_grid())
    sections = []
    for name, block in sorted(dict(curves).items()):
        scores = np.asarray(block["log_probabilities"], dtype=np.float64)
        control = np.asarray(block["control_log_probabilities"], dtype=np.float64)
        grid = {str(int(n)): metrics.expected_distinct_yield(scores, int(n)) for n in budgets}
        differences = [metrics.expected_distinct_yield(scores, int(n))
                       - metrics.expected_distinct_yield(control, int(n)) for n in dense]
        brackets = metrics.crossing_brackets(list(dense), differences)
        refinements = [metrics.refine_bracket(
            entry, lambda n: (metrics.expected_distinct_yield(scores, int(n))
                              - metrics.expected_distinct_yield(control, int(n))))
            for entry in brackets["brackets"]]
        sections.append({"entry": name, "identities": int(scores.size), "curve": grid,
                         "dense_grid": [int(n) for n in dense],
                         "dense_differences": [float(value) for value in differences],
                         "crossovers": brackets, "refinements": refinements,
                         "control": block.get("control_label"),
                         "reading": ("initial sign, search domain and every sign change found on "
                                     "this grid are retained. 'No crossing found' is not 'always "
                                     "better', and configurations whose crossovers differ are not "
                                     "collapsed into one reported band.")})
    return sections


def coupling_section(blocks):
    """MI, total correlation, the decomposition, the hybrids and the classifiers."""
    return {"entries": list(blocks),
            "separation": ("the 45-pair MI sum and the actual total correlation are different "
                           "quantities and are reported apart. The parent-context Jensen gap "
                           "describes context dependence of distributional change; it is not "
                           "epistasis and it is not evidence that a network learned an "
                           "interaction."),
            "requirements": ["every MI row carries its bank size",
                             "every MI row carries the column-permutation floor",
                             "every TC row carries the independent-bank repeat",
                             "no bias-corrected dependence claim is emitted"],
            "classifier_note": ("the additive and pairwise fits are DISCRIMINATIVE comparators. "
                                "Their P(high) is a ranking score; it is never treated as a "
                                "generative probability and never enters a yield calculation."),
            "mixture_note": ("a mixture's dependence can come from the component indicator. Its "
                             "MI is reported separately and is not evidence that a single network "
                             "learned new interactions.")}


def decomposition_section(blocks):
    return {"entries": list(blocks),
            "identity": "s_Q - s_P = (g_Q - g_P) + (c_Q - c_P), exactly, for the estimated "
                        "marginals",
            "reported": ["full score", "marginal-only score", "dependency-bearing residual score",
                         "hybrid s_P + (g_Q - g_P)", "hybrid s_P + (c_Q - c_P)",
                         "within-class mean score decomposition"],
            "forbidden": ("AP differences are not added. AP is not linear in the score, so the "
                          "decomposition stays on class-mean scores and no field decomposes an "
                          "AP difference.")}


# ---------------------------------------------------------------------------
# the failure ledger and the artifact manifest
# ---------------------------------------------------------------------------

def failure_ledger(context, state):
    """Every stop, failure, deferral, missing checkpoint and blocked stage, in one place."""
    entries = []
    for row in state["rows"]:
        status = row["observed_status"]
        if status in ("completed",):
            continue
        entries.append({"kind": "trajectory", "trajectory": row["trajectory"],
                        "status": status,
                        "stop_reason": (row.get("terminal") or {}).get("stop_reason"),
                        "journalled_updates": row["journalled_updates"],
                        "classification": _classify(status)})
    for stage, block in campaign.stage_summary(context.run_root).items():
        if block["status"] in ("completed",):
            continue
        entries.append({"kind": "stage", "stage": stage, "status": block["status"],
                        "reason": block["reason"],
                        "classification": ("readiness gate" if block["status"] ==
                                           "blocked_readiness" else
                                           "scientific feasibility gate" if block["status"] ==
                                           "blocked_feasibility" else "not run")})
    calibration = context.path("calibration_ledger.json")
    if calibration.is_file():
        ledger = paths.read_json(calibration)
        for entry in ledger["entries"]:
            if entry["status"] in ("completed",):
                continue
            entries.append({"kind": "calibration", "entry_id": entry["entry_id"],
                            "status": entry["status"], "reason": entry.get("reason"),
                            "classification": _classify(entry["status"])})
    document = {"schema_version": contract.NF_SCHEMA, "record_kind": "failure_ledger",
                "entries": entries, "count": len(entries),
                "classification_rule": ("a stopped arm is an OUTCOME; a crash is a FAILURE needing "
                                        "diagnosis; a deferred optional arm is a declared "
                                        "omission; a missing historical checkpoint is missing. "
                                        "None of them is an omitted row.")}
    paths.write_json(context.path(FAILURE_LEDGER), document)
    return document


def _classify(status):
    return {"stopped_by_gate": "declared scientific outcome",
            "failed": "machinery failure needing diagnosis",
            "incomplete": "did not reach its declared endpoints",
            "interrupted": "in progress or killed; resumable",
            "queued": "not started",
            "deferred_optional": "declared optional work, explicitly deferred",
            "reuse_pending_parity": "historical reuse not yet established by parity",
            "reuse_verified": "a historical path reused after parity; early checkpoints missing",
            "coefficients_pending": "calibration has not frozen this arm's coefficient yet",
            "no_qualified_configuration": "bounded calibration produced no qualified configuration",
            "not_started": "never started; the bounded allowance was spent",
            }.get(status, status)


#: Paths the manifest never lists. A heartbeat, a lock, a log and a live progress
#: file all change while the manifest is being written, so a verification that
#: re-hashes them fails for reasons that say nothing about the science. The
#: report, the manifest and the verification are excluded for the same reason
#: with an extra twist: including them makes the manifest depend on its own
#: bytes, and a rerun then never reproduces it.
#:
#: ``queue.json`` and the report's and verification's own STAGE RECORDS belong
#: here for the same self-reference reason, one step removed: ``run_stage``
#: writes ``stages/<stage>.json`` AFTER its handler returns, and the supervisor
#: republishes ``queue.json`` on every pass. A second reporting run over a
#: partial flight therefore hashed three files that the reporting pipeline was
#: about to rewrite, and the verification that followed rejected artifacts it
#: had changed itself. Every other stage record stays in the manifest: those are
#: written once and are real evidence.
MUTABLE_ARTIFACTS = (
    "source_snapshot/", "heartbeat.json", "campaign.lock", "supervisor.log",
    "launch.json", "campaign_status.json", "run_all.json", ARTIFACT_MANIFEST,
    REPORT_JSON, REPORT_MD, "verification.json", "report/",
    "queue.json", "stages/report.json", "stages/verify.json",
)

#: Names inside a trajectory directory that are rolling rather than immutable.
MUTABLE_SUFFIXES = ("trajectory_progress.json", "resume_state.pt", "progress.json")


def is_immutable_artifact(relative):
    """Is this a scientific artifact that, once written, never changes again?"""
    text = str(relative)
    if any(text == name or text.startswith(name) for name in MUTABLE_ARTIFACTS):
        return False
    return not any(text.endswith(name) for name in MUTABLE_SUFFIXES)


def artifact_manifest(context):
    """Every IMMUTABLE artifact this run wrote, with its digest."""
    entries, excluded = [], []
    for path in sorted(context.run_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(context.run_root).as_posix()
        if not is_immutable_artifact(relative):
            excluded.append(relative)
            continue
        entries.append({"file": relative, "bytes": int(path.stat().st_size),
                        "sha256": paths.sha256_file(path)})
    document = {"schema_version": contract.NF_SCHEMA, "record_kind": "artifact_manifest",
                "run_root": context.config["run_root"], "files": entries, "count": len(entries),
                "excluded": excluded, "excluded_count": len(excluded),
                "exclusion_rule": ("mutable operational files -- the heartbeat, the lock, the "
                                   "supervisor log, the rolling resume state and progress files "
                                   "-- and the report, the manifest and the verification "
                                   "themselves. The last three would make the manifest depend on "
                                   "its own bytes, so a rerun could never reproduce it. The "
                                   "source_snapshot carries its own aggregate digest."),
                "storage_note": ("the run root is a portable logical name; its machine-local "
                                 "location may be a junction onto another volume and no "
                                 "scientific field carries a drive letter.")}
    paths.write_json(context.path(ARTIFACT_MANIFEST), document)
    return document


def runs_jsonl(context, state):
    """The per-run schema the specification asks for, one JSON object per line."""
    target = context.path(RUNS_JSONL)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in state["rows"]:
        terminal = row.get("terminal") or {}
        lines.append({
            "record_kind": "run", "run_id": row["trajectory"], "block": row["block"],
            "regime": row["regime"], "arm": row["arm"], "task": row["task"],
            "preservation": row["preservation"], "coefficients": row["coefficients"],
            "seed": row["seed"], "parent_id": row["parent_id"],
            "status": row["observed_status"], "stop_reason": terminal.get("stop_reason"),
            "final_update": row["journalled_updates"],
            "task_exposures": (terminal.get("exposures") or {}).get("chosen"),
            "replay_exposures": (terminal.get("exposures") or {}).get("replay"),
            "endpoints_reached": row["endpoints_reached"],
            "checkpoint_hashes": {str(key): (value.get("checkpoint") or {}).get("state_sha256")
                                  for key, value in
                                  (terminal.get("endpoints_reached") or {}).items()},
            "timing": terminal.get("cost"), "identity": terminal.get("identity")})
    # One object per line, so ``canonical_json`` (which indents) is the wrong
    # writer here: stripping its newlines would also strip newlines inside any
    # string value and silently corrupt a note field.
    text = "\n".join(json.dumps(line, sort_keys=True, allow_nan=False) for line in lines)
    paths.write_text(target, text + ("\n" if text else ""))
    return {"path": RUNS_JSONL, "rows": len(lines)}


# ---------------------------------------------------------------------------
# figures, with a data artifact either way
# ---------------------------------------------------------------------------

def render_curves(context, sections):
    """Write the curve data, then the figure if a backend exists. The data is not optional."""
    paths.write_json(context.path("curves.json"), {
        "schema_version": contract.NF_SCHEMA, "record_kind": "curves", "sections": sections})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:                                # pragma: no cover - env dependent
        return {"figures": [], "skipped": True,
                "reason": f"{type(error).__name__}: {error}",
                "data_artifact": "curves.json",
                "consequence": "the curve DATA is written regardless; only the rendering is "
                               "skipped"}
    written = []
    figure, axis = plt.subplots(figsize=(6, 4))
    for section in sections:
        axis.plot(section["dense_grid"], section["dense_differences"], label=section["entry"])
    axis.set_xscale("log")
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_xlabel("draws N")
    axis.set_ylabel("expected distinct high identities, minus control")
    axis.legend(fontsize=7)
    target = context.path("report", "her2-next-flight-yield.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    inherited_report.save_figure(figure, target, replace=True)
    written.append(str(target.relative_to(context.run_root).as_posix()))
    plt.close(figure)
    return {"figures": written, "skipped": False, "data_artifact": "curves.json"}


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def build_report(context, *, services=None, evidence=None):
    """Assemble every required section from the artifacts on disk."""
    queue = campaign.queue_for_reporting(context)
    state = campaign.campaign_state(context.run_root, queue)
    evidence = dict(evidence or {})
    matrix = checkpoint_matrix(context, state, registry=evidence.get("registry"))
    ledger = failure_ledger(context, state)
    runs = runs_jsonl(context, state)
    sections = yield_section(evidence.get("yield_curves") or {})
    figures = render_curves(context, sections)
    stages = campaign.stage_summary(context.run_root)
    unfinished = [row["trajectory"] for row in state["rows"]
                  if row["observed_status"] in campaign.NON_TERMINAL_STATUSES]
    # ``report`` is running right now and ``verify`` runs after it, so neither can
    # have a completed record at this moment. Requiring them here made completion
    # depend on itself: the report could never call the flight complete, however
    # complete it was. They are excluded and the verify stage checks the rest.
    pending_by_construction = ("report", "verify")
    complete = (all(block["status"] == "completed" for name, block in stages.items()
                    if name not in pending_by_construction)
                and not unfinished and bool(evidence.get("rankings"))
                and bool(evidence.get("yield_curves")) and bool(evidence.get("class_mass"))
                and bool(evidence.get("coupling")))
    document = {
        "schema_version": contract.NF_SCHEMA, "record_kind": "report",
        "campaign_id": context.campaign_id,
        "rows": len(matrix), "checkpoint_matrix": matrix,
        "queue_counts": state["counts"],
        "primary_contrasts": paired_contrasts(
            evidence.get("metric_records") or {},
            contrasts=context.config["analysis"]["primary_contrasts"]),
        "all_contrasts": paired_contrasts(
            evidence.get("metric_records") or {},
            contrasts=context.config["analysis"].get("secondary_contrasts") or []),
        "yield": sections, "figures": figures,
        "challenge": evidence.get("challenge"),
        "comparators": evidence.get("comparators"),
        "comparator_rankings": evidence.get("comparator_rankings"),
        "rankings": evidence.get("rankings"),
        "checkpoint_registry": evidence.get("registry"),
        "row_score_artifacts": evidence.get("row_scores"),
        "mixtures": evidence.get("mixtures"),
        "preservation_contrasts": evidence.get("preservation_contrasts"),
        "difference_in_differences": (evidence.get("challenge") or {}).get(
            "difference_in_differences"),
        "class_mass": evidence.get("class_mass"),
        "coupling": coupling_section(evidence.get("coupling") or []),
        "decomposition": decomposition_section(evidence.get("decomposition") or []),
        "mixture": _read_optional(context, "mixture_alpha_curve.json"),
        "preservation_audit": _read_optional(context, "preservation_audit.json"),
        "geometry": _read_optional(context, "geometry_feasibility.json"),
        "split_manifest": _read_optional(context, "split_manifest.json"),
        "neighbor_certificate": _read_optional(context, "neighbor_certificate.json"),
        "calibration_ledger": _read_optional(context, "calibration_ledger.json"),
        "runtime_forecast": _read_optional(context, campaign.FORECAST_JSON),
        "failure_ledger": ledger, "runs": runs,
        "stages": stages,
        "completion": {
            "complete": False,
            "ready_for_verification": bool(complete),
            "required_stages": list(campaign.REQUIRED_STAGES),
            "stages_completed": [name for name, block in stages.items()
                                 if block["status"] == "completed"],
            "rows_without_terminal_status": unfinished,
            "rule": ("complete means every required stage completed AND every queued row reached "
                     "a terminal status. A launched flight remains scientifically in progress "
                     "until then, and a successful launch never implies completion.")},
        "limits": LIMITS,
        "artifact_manifest": artifact_manifest(context)["files"]}
    # Coerced once, here: numpy scalars are JSON-invisible until ``allow_nan=False``
    # json.dumps meets one, and that happens at the very end of the flight.
    document = _jsonable(document)
    paths.write_json(context.path(REPORT_JSON), document)
    text = render_markdown(document)
    require_no_forbidden_claim(text)
    paths.write_text(context.path(REPORT_MD), text)
    return document


def finalize_verification(context, verification):
    """Publish completion only after the independent artifact checks have run."""
    target = context.path(REPORT_JSON)
    if not target.is_file():
        return
    document = paths.read_json(target)
    completion = document["completion"]
    completion["verification_passed"] = bool(verification["passed"])
    completion["complete"] = bool(completion.get("ready_for_verification") and verification["passed"])
    if verification["passed"] and "verify" not in completion["stages_completed"]:
        completion["stages_completed"].append("verify")
    paths.write_json(target, document)
    paths.write_text(context.path(REPORT_MD), render_markdown(document))


LIMITS = [
    "Three parent seeds give limited information about training variability, and one parent "
    "family supplied the calibration, which makes the production comparison exploratory.",
    "Historical validation and Block-A exposure carry into the challenge's data universe. E is "
    "newly separated at ADAPTATION; it was not globally unobserved by the research process.",
    "Class and WT-distance matching does not remove every distributional change the purge causes, "
    "and it does not randomize residue features or library provenance.",
    "The challenge's u1000 endpoint is a fixed-exposure question and cannot establish converged "
    "performance.",
    "A sequence mixture uses two checkpoints, may create mixture-induced dependence, and differs "
    "from a single policy in storage, scoring latency and operational simplicity.",
    "Rare-event preservation can improve while moderate probability loss worsens; tail and "
    "average measures are reported apart.",
    "A statistical dependency is not a functional interaction, a structural mechanism, or a "
    "useful unassayed molecule.",
    "Off-library functionality, affinity and clinical claims, complete pretraining-overlap "
    "certification and converged scratch-versus-pretrained performance remain outside this "
    "flight.",
    "Tail events here are INCLUSIVE (drop >= threshold); the historical post-hoc audit used the "
    "strict comparison. Both are published and neither substitutes for the other.",
]


def _read_optional(context, name):
    target = context.path(name)
    return paths.read_json(target) if target.is_file() else None


def _jsonable(node):
    """numpy scalars and arrays coerced to plain JSON types, recursively."""
    if isinstance(node, dict):
        return {str(key): _jsonable(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(value) for value in node]
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, np.ndarray):
        return node.tolist()
    return node


def _number(value, places=4):
    return "" if value is None else f"{float(value):.{places}f}"


def render_markdown(document):
    lines = [f"# HER2 next flight -- {document['campaign_id']}", "",
             "Status: **{}**.".format("complete" if document["completion"]["complete"]
                                      else "in progress or stopped"), "",
             "## Stage record", ""]
    lines.append("| stage | status | reason |")
    lines.append("|---|---|---|")
    for stage, block in document["stages"].items():
        lines.append(f"| {stage} | {block['status']} | {block.get('reason') or ''} |")
    lines += ["", "## Queue", "",
              "| status | rows |", "|---|---:|"]
    for status, count in sorted(document["queue_counts"].items()):
        lines.append(f"| {status} | {count} |")

    if document.get("rankings"):
        lines += ["", "## Ranking on original validation", "",
                  "| model | macro AP | pooled AP | pooled AUROC |",
                  "|---|---:|---:|---:|"]
        for name, block in sorted(document["rankings"].items()):
            pooled = block["pooled"]
            lines.append(f"| {name} | {_number(block['macro_average_precision'])} | "
                         f"{_number(pooled['average_precision'])} | "
                         f"{_number(pooled.get('auroc'))} |")

    lines += ["", "## Primary contrasts", "",
              "| contrast | mean difference | nominal 95% interval | paired seeds | raw values |",
              "|---|---:|---|---:|---|"]
    for entry in document["primary_contrasts"] + document["all_contrasts"]:
        if not entry.get("available"):
            lines.append(f"| {entry['name']} | | {entry.get('reason', 'not available')} | 0 | |")
            continue
        raw = ", ".join(f"{a:.4f}-{b:.4f}" for a, b in zip(entry["left_values"],
                                                           entry["right_values"]))
        lines.append(f"| {entry['name']} | {_number(entry.get('mean'))} | "
                     f"[{_number(entry.get('lower'))}, {_number(entry.get('upper'))}] | "
                     f"{len(entry['paired_seeds'])} | {raw} |")

    registry = document.get("checkpoint_registry") or {}
    if registry:
        lines += ["", "## Checkpoint coverage", "",
                  f"Expected {registry.get('expected')} checkpoints; "
                  f"{registry.get('present')} present, {len(registry.get('missing') or [])} "
                  "missing (listed with their reasons in the failure ledger)."]

    mass = document.get("class_mass") or {}
    if mass:
        lines += ["", "## Generated class mass on the assayed panel", "",
                  "| model | high | mid | low | panel total |", "|---|---:|---:|---:|---:|"]
        for name, block in sorted(mass.items()):
            panel = block.get("_panel") or {}
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    name, _number((block.get("high") or {}).get("mass"), 6),
                    _number((block.get("mid") or {}).get("mass"), 6),
                    _number((block.get("low") or {}).get("mass"), 6),
                    _number(panel.get("mass"), 6)))

    mixtures = (document.get("mixtures") or {}).get("curves") or []
    if mixtures:
        lines += ["", "## Sequence mixture, alpha grid", "",
                  "| model | alpha | mean sum log probability on high | E[Y@10k] |",
                  "|---|---:|---:|---:|"]
        for curve in mixtures:
            for point in curve["points"]:
                yields = point.get("expected_distinct_high") or {}
                lines.append(
                    f"| {curve['label']} | {point['alpha']:g} | "
                    f"{_number(point['mean_sum_log_probability_on_high'])} | "
                    f"{_number(yields.get('10000'), 2)} |")

    for section in document.get("yield") or []:
        lines += ["", f"### Yield, {section['entry']} versus {section.get('control')}", "",
                  "| N | expected distinct high |", "|---:|---:|"]
        for budget, value in sorted(section["curve"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {int(budget):,} | {_number(value, 2)} |")
        brackets = (section.get("crossovers") or {}).get("brackets") or []
        lines.append("")
        lines.append(f"Sign changes found on the declared grid: {len(brackets)}. "
                     "'No crossing found' is not 'always better'.")

    challenge = document.get("challenge") or {}
    if challenge.get("records"):
        lines += ["", "## Challenge panel (high-versus-rest AP on the fixed E)", "",
                  "| regime | arm | seed | update | AP | AUROC |",
                  "|---|---|---:|---:|---:|---:|"]
        for block in challenge["records"]:
            primary = block.get("primary") or {}
            lines.append(f"| {block['regime']} | {block['arm']} | {block['seed']} | "
                         f"{block['update']} | {_number(primary.get('average_precision'))} | "
                         f"{_number(primary.get('auroc'))} |")
        did = challenge.get("difference_in_differences") or {}
        if did.get("available"):
            lines += ["", f"Difference in differences: {_number(did.get('mean'))} "
                          f"[{_number(did.get('lower'))}, {_number(did.get('upper'))}] over "
                          f"{len(did['paired_seeds'])} paired seeds."]

    audit = document.get("preservation_contrasts") or {}
    if audit.get("models"):
        lines += ["", "## Preservation audit (fresh 50k banks)", "",
                  "| model | rows | mean log ratio | inclusive 10x rate | inclusive 100x rate |",
                  "|---|---:|---:|---:|---:|"]
        for block in audit["models"]:
            drop = block.get("drop") or {}
            events = ((drop.get("tails") or {}).get("events") or {})
            lines.append(f"| {block['model']} | {block['rows']} | "
                         f"{_number(drop.get('mean_log_ratio'))} | "
                         f"{_number((events.get('tenfold') or {}).get('rate'), 6)} | "
                         f"{_number((events.get('hundredfold') or {}).get('rate'), 6)} |")

    coupling_entries = (document.get("coupling") or {}).get("entries") or []
    if coupling_entries:
        lines += ["", "## Dependence", "",
                  "| model | rows | 45-pair MI sum | permutation floor | total correlation |",
                  "|---|---:|---:|---:|---:|"]
        for block in coupling_entries:
            for repeat in block.get("repeats") or []:
                lines.append(
                    f"| {block['model']} | {repeat['rows']} | "
                    f"{_number(repeat.get('pairwise_mi_sum'))} | "
                    f"{_number((repeat.get('permutation_floor') or {}).get('mean'))} | "
                    f"{_number((repeat.get('total_correlation') or {}).get('total_correlation'))}"
                    " |")

    lines += ["", f"Checkpoint matrix rows: {document['rows']}.",
              f"Failure-ledger entries: {document['failure_ledger']['count']}.",
              f"Row-score artifacts: {len(document.get('row_score_artifacts') or {})}.", "",
              "## What this cannot say", ""]
    lines += [f"- {entry}" for entry in document["limits"]]
    lines += ["", "## Completion", "",
              document["completion"]["rule"], ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(context):
    """Independent checks over what is actually on disk."""
    checks = {}
    queue = campaign.queue_for_reporting(context)
    state = campaign.campaign_state(context.run_root, queue)

    report = _read_optional(context, REPORT_JSON)
    checks["report_present"] = _check(report is not None, "the report document exists on disk")
    checks["every_queued_row_in_matrix"] = _check(
        report is not None and len({row["trajectory"] for row in report["checkpoint_matrix"]
                                    if row["kind"] == "preference_cell"})
        == len({row["trajectory"] for row in state["rows"]}),
        "every queued trajectory appears in the checkpoint matrix")
    # ``verify`` is excluded because it is the stage currently running: its own
    # record is written after this function returns, and demanding it here would
    # make the check unsatisfiable rather than strict.
    checks["stage_records_present"] = _check(
        all(campaign.read_stage_record(context.run_root, stage) is not None
            for stage in campaign.REQUIRED_STAGES if stage != "verify"),
        "each required stage other than verify itself wrote a record, even if that record is a "
        "blocked gate")

    manifest = _read_optional(context, ARTIFACT_MANIFEST)
    missing = []
    if manifest:
        for entry in manifest["files"]:
            target = context.path(entry["file"])
            if not target.is_file() or paths.sha256_file(target) != entry["sha256"]:
                missing.append(entry["file"])
    checks["artifacts_match_manifest"] = _check(manifest is not None and not missing,
                                                "every manifested artifact is present and hashed",
                                                detail=missing[:10])

    certificate = _read_optional(context, "neighbor_certificate.json")
    checks["neighbor_certificate_zero_violations"] = _check(
        certificate is not None and bool(certificate.get("zero_violations")),
        "the purge certificate reports zero prohibited cross-boundary pairs")

    calibration = _read_optional(context, "calibration_ledger.json")
    checks["calibration_ledger_complete"] = _check(
        calibration is not None and calibration["attempted"] >= calibration["completed"],
        "the calibration ledger lists attempts, not only survivors")

    snapshot = _read_optional(context, "source_snapshot.json")
    drift = []
    if snapshot:
        for logical, block in snapshot["files"].items():
            for where, target in (("worktree", context.repository(logical)),
                                  ("archive", paths.resolve_under(
                                      context.path("source_snapshot"), logical))):
                if not target.is_file() or paths.sha256_file(target) != block["sha256"]:
                    drift.append(f"{logical} ({where})")
    checks["source_snapshot_stable"] = _check(
        snapshot is not None and not drift,
        "every frozen source file is unchanged in the worktree AND in the archived snapshot",
        detail=drift[:10])

    # Absent evidence is not a failed check: a flight stopped at a readiness gate
    # never produced one, and reporting that as a verification failure would
    # confuse "the run stopped" with "the run wrote something inconsistent".
    registry_block = (_read_optional(context, "report_evidence.json") or {}).get("registry")
    checks["checkpoint_registry_complete"] = _check(
        registry_block is None
        or registry_block.get("expected") == (registry_block.get("present", 0)
                                              + len(registry_block.get("missing") or [])),
        "every expected checkpoint is either present or listed as missing with a reason",
        detail={"evidence": "absent" if registry_block is None else "present",
                "missing": (registry_block or {}).get("missing", [])[:10]})

    freeze = _read_optional(context, "finalist_freeze.json")
    audit = _read_optional(context, "preservation_audit.json")
    unaudited = []
    if freeze and audit:
        audited = {block["model"] for block in audit["models"]}
        unaudited = [entry["name"] for entry in freeze["named_checkpoints"]
                     if entry["name"] not in audited]
    checks["every_named_checkpoint_audited"] = _check(
        freeze is None or audit is None or not unaudited,
        "every checkpoint named in the finalist freeze was scored on a final audit bank",
        detail=unaudited[:10])

    banks_before_freeze = None
    if freeze and audit:
        banks_before_freeze = [block for block in audit["banks"]
                               if block.get("role") not in (None, "final_preservation")]
    checks["audit_banks_carry_the_audit_role"] = _check(
        audit is None or not banks_before_freeze,
        "every audit bank carries the final-preservation role, which refuses to inform a "
        "selection before the freeze record exists",
        detail=(banks_before_freeze or [])[:3])

    registry_streams = _read_optional(context, "stream_registry.json")
    seeds = [block["seed"] for block in (registry_streams or {}).get("streams", {}).values()]
    checks["spawned_streams_do_not_collide"] = _check(
        registry_streams is None or len(seeds) == len(set(seeds)),
        "no two spawned streams derived the same literal seed",
        detail={"streams": len(seeds), "distinct": len(set(seeds))})

    mismatched = []
    for row in state["rows"]:
        if row["observed_status"] != "completed":
            continue
        durable = trajectory_lib.durable_progress(context.path("trajectories",
                                                               row["trajectory"]))
        terminal = row.get("terminal") or {}
        if int(terminal.get("updates") or 0) != int(durable["updates"]):
            mismatched.append({"trajectory": row["trajectory"],
                               "terminal_updates": terminal.get("updates"),
                               "journalled_updates": durable["updates"],
                               "replayed_records": durable["replayed_update_records"]})
    checks["journal_dedup"] = _check(
        not mismatched,
        "each completed trajectory's terminal update count equals its de-duplicated journal "
        "count, so a resume's replayed records were not double-counted",
        detail=mismatched[:10])

    def stage_completed(name):
        return (campaign.read_stage_record(context.run_root, name) or {}).get("status") == "completed"

    if stage_completed("production"):
        checks["production_evidence_present"] = _check(
            registry_block is not None and registry_streams is not None and bool(seeds),
            "completed production requires a checkpoint registry and nonempty stream provenance")
        checks["production_has_no_pending_rows"] = _check(
            not any(row["observed_status"] in campaign.NON_TERMINAL_STATUSES for row in state["rows"]),
            "completed production cannot leave required cells pending")
    if stage_completed("audit"):
        checks["audit_evidence_present"] = _check(
            bool(freeze and audit and audit.get("models") and audit.get("banks")),
            "completed audit requires named checkpoints, actual model results and banks")
        checks["final_preservation_bank_sizes"] = _check(
            bool(audit and audit.get("banks")) and all(
                block.get("role") == "final_preservation" and int(block.get("rows", 0)) == 50000
                for block in audit["banks"]), "every final preservation bank has 50000 draws")
    if stage_completed("couple"):
        coupled = _read_optional(context, "coupling.json") or {}
        screens = coupled.get("screens") or []
        finalists = coupled.get("finalists") or []
        allowed_missing = ("stopped_by_gate", "incomplete", "no_qualified_configuration")
        missing_finalists = (freeze or {}).get("coupling_missing") or []
        expected_finalists = int(context.config["banks"]["finalist_models"])
        checks["declared_finalist_count"] = _check(
            len(finalists) + len(missing_finalists) == expected_finalists
            and all(str(entry.get("reason", "")).startswith(allowed_missing)
                    for entry in missing_finalists),
            "fifteen finalists, or each absent member has a declared scientific outcome",
            detail={"observed": len(finalists), "missing": missing_finalists})
        checks["coupling_coverage"] = _check(
            bool(freeze and screens and finalists)
            and {entry["name"] for entry in freeze["screen_models"]} == {entry["model"] for entry in screens}
            and {entry["name"] for entry in freeze["coupling_finalists"]} == {entry["model"] for entry in finalists},
            "every frozen screen and finalist has a coupling result")
        checks["coupling_bank_sizes"] = _check(
            bool(screens and finalists)
            and all(len(entry["banks"]) == 1 and int(entry["banks"][0]["rows"]) == 10000 for entry in screens)
            and all(len(entry["banks"]) == 2 and all(int(bank["rows"]) == 50000 for bank in entry["banks"])
                    for entry in finalists), "10000 per screen and two independent 50000 banks per finalist")
        checks["tc_bootstrap_present"] = _check(
            bool(finalists) and all(block.get("total_correlation_bootstrap")
                                    for entry in finalists for block in entry.get("repeats", [])),
            "actual TC has a whole-trajectory bootstrap on every finalist repeat")
        checks["primary_mixture_diagnostics"] = _check(
            len(coupled.get("mixture_diagnostics") or []) >= 3,
            "primary IPO mixture diagnostics cover all three original parents")
    if stage_completed("report"):
        checks["numeric_report_sections"] = _check(
            bool(report and report.get("rankings") and report.get("yield") and report.get("class_mass")
                 and (report.get("coupling") or {}).get("entries")),
            "completed report has ranking, yield, mass and coupling results")
        checks["six_primary_contrasts"] = _check(
            bool(report) and len(report.get("primary_contrasts") or []) == 6,
            "all six prespecified primary contrasts appear, including explicit unavailable results")
    if calibration is not None:
        checks["calibration_budget_respected"] = _check(
            float(calibration["measured_gpu_seconds"]) + float(calibration.get("uncertainty_debit_seconds", 0))
            <= float(calibration["gpu_hour_cap"]) * 3600,
            "measured GPU-work time plus separately labelled crash debits stays within the allowance")
    passed = all(block["passed"] for block in checks.values())
    document = {"schema_version": contract.NF_SCHEMA, "record_kind": "verification",
                "checks": checks, "passed": bool(passed),
                "claim": ("independent checks over the artifacts on disk. Passing verification "
                          "does not assert that the science is correct; it asserts that what was "
                          "written is complete, self-consistent and attributable.")}
    paths.write_json(context.path("verification.json"), document)
    return document


def _check(passed, description, detail=None):
    return {"passed": bool(passed), "description": description, "detail": detail}


def finalist_freeze(context, *, named_checkpoints, screen_models, tail_family, reason,
                    coupling_finalists=(), missing_required=(), coupling_missing=(),
                    provisional=False, production_outcome_sha256=None):
    """Freeze the finalist choice BEFORE any new audit bank is drawn.

    ``named_checkpoints`` is the PRESERVATION audit list -- Block A at u1000 and
    u3750, Block B at u1000, plus the parents and the reused controls.
    ``coupling_finalists`` is the smaller declared list that gets the two
    independent 50k generation banks. They are different requirements and the
    record keeps them apart; treating the coupling list as the audit list
    silently dropped every u3750 and every Block-B audit.
    """
    require(named_checkpoints, "a finalist freeze names at least one checkpoint")
    document = {"schema_version": contract.NF_SCHEMA, "record_kind": "finalist_freeze",
                "named_checkpoints": list(named_checkpoints),
                "coupling_finalists": list(coupling_finalists),
                "screen_models": list(screen_models),
                "missing_required": list(missing_required),
                "coupling_missing": list(coupling_missing),
                "coverage": {"audit_checkpoints": len(list(named_checkpoints)),
                             "coupling_finalists": len(list(coupling_finalists)),
                             "screens": len(list(screen_models)),
                             "missing": len(list(missing_required)),
                             "rule": ("a required checkpoint that does not exist is listed with "
                                      "its reason. It is never dropped from the coverage count "
                                      "and never reconstructed.")},
                "tail_family": dict(tail_family), "reason": str(reason),
                "provisional": bool(provisional),
                "production_outcome_sha256": production_outcome_sha256,
                "provisional_rule": ("a freeze taken while the production queue was still "
                                     "incomplete names only the checkpoints that existed then. "
                                     "It is retaken -- before any new bank is drawn -- as soon "
                                     "as production advances. Only a freeze taken over a "
                                     "terminal queue is final."),
                "rule": ("the tail family is chosen on three-seed DEVELOPMENT results: all-seed "
                         "point-rate feasibility first, then higher mean macro-AP, with a .001 AP "
                         "tie broken by Y@10k, then Y@1M, then IPO-tail. If neither tail family "
                         "is feasible, the family minimizing the maximum normalized tail "
                         "violation is audited under the same tie rules and that fact is "
                         "reported."),
                "retention": "the other family's 10k development result is retained and reported",
                "ordering": ("this record exists before any final-preservation or finalist "
                             "generation bank is drawn; those banks refuse to inform a selection "
                             "without it."),
                "frozen_at": paths.utc_now()}
    paths.write_json(context.path("finalist_freeze.json"), document)
    return document


def mixture_diagnostic(parent_log_conditionals, policy_log_conditionals, log_weights, *, label):
    """A mixture's own coupling diagnostic, from the TRUE mixture conditionals.

    Combined in log space under both component log posterior weights. Taking the
    logarithm of a probability-space mixture would lose every component
    conditional that underflowed, and a fixed alpha in place of the posterior
    weights would describe a different distribution entirely.
    """
    log_conditionals = mixture_lib.mixture_log_conditionals(
        parent_log_conditionals, policy_log_conditionals, log_weights)
    block = coupling.total_correlation(log_conditionals)
    return {"label": str(label),
            "mixture_total_correlation": {k: v for k, v in block.items()
                                          if k != "log_marginals"},
            "conditionals_used": ("posterior-weighted mixture conditionals in log space, not a "
                                  "fixed alpha and not log(mixture of probabilities)"),
            "caveat": ("component identity can itself induce dependence. This is reported "
                       "separately and is not evidence that a single network learned new "
                       "interactions.")}

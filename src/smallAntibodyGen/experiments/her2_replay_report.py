"""Reporting for the replay screen: every status, the tradeoff, and no winner.

The primary output is the **preservation-versus-ranking tradeoff**, not a
coefficient declared successful because it improved one average. Four refusals
are built into this module rather than left to the person writing the prose:

* **Nothing absent is filled in.** A trajectory that stopped, was interrupted or
  never started appears with that status in every table. A matched zero-replay
  delta exists only when *both* sides reached the same exposure endpoint at the
  same seed; otherwise it is reported unavailable, and no earlier endpoint, other
  seed, rolling last-passing state or historical run is substituted for it.
* **Pareto points are descriptive.** "Observed Pareto-efficient at these seeds and
  these endpoints" is a statement about the points measured; it is not statistical
  superiority and it is not a selection. No global winner and no post-hoc scalar
  combination of preservation and ranking is produced.
* **The claims stay separated.** Preservation is distributional, validation AP is a
  ranking observation on validation rows, diversity eligibility is an operational
  anti-collapse heuristic, and none of the three is affinity. The monitoring bank
  is used for diagnostics during the fit, so it is not an untouched final test and
  the report says so wherever it is cited.
* **A published figure is written once; a run-directory figure tracks the run.**
  The two are different artifacts that happen to both be PNGs. Inside the ignored
  run directory a figure describes a campaign that is still moving and is
  re-rendered as it moves (``replace=True``); a *published* deliverable somebody
  may have cited is compared by rendering to a temporary with an explicit format
  and is never replaced in place.
* **The per-stratum numbers are in the report, not only in the JSON.** The headline
  statistic is a macro average over three training-distance strata, and a macro
  average is exactly the number that can hide a stratum going the other way. Every
  stratum's AP and every stratum's matched zero-replay difference are printed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import her2_replay_campaign as campaign_lib
from . import her2_support_paths as paths
from .her2_guarded_eval import SELECTION_STRATA, macro_average_precision
from .her2_runtime import require

REPORT_SCHEMA = "her2-parent-replay/1"

#: Affirmative phrasings this screen may not write. Deliberately the *positive*
#: forms: the report legitimately contains "no global winner" and "not an affinity
#: measurement", so a guard that grepped for the bare nouns would reject the
#: sentences that exist to prevent the claim. These are the sentences somebody
#: would add later, and they are refused at render time.
FORBIDDEN_CLAIMS = (
    "independently confirms", "confirms affinity", "improves affinity",
    "establishes affinity", "proves preservation", "the global winner", "the winning arm",
    "best arm overall", "the best coefficient", "confirmatory evidence")


def attach_endpoint_records(run_root, state):
    """Read each trajectory's endpoint records off disk and attach them to the state.

    From the trajectory's own terminal status where there is one, and otherwise
    from its append-only endpoint journal -- so an arm that is still running, and
    one that was interrupted after reaching an endpoint, both keep the endpoints
    they actually reached instead of losing them to the absence of a terminal
    document.

    Nothing is inferred from the presence of a checkpoint file. A ``.pt`` whose
    endpoint record was never written is an interrupted save, and a report that
    counted it would be reporting an endpoint that has no evaluation.
    """
    run_root = Path(run_root)
    enriched = []
    for entry in state["trajectories"]:
        directory = run_root / "trajectories" / entry["trajectory"]
        terminal = campaign_lib.read_terminal_status(directory)
        source = ((terminal or {}).get("endpoints_reached")
                  if terminal is not None
                  else campaign_lib.durable_progress(directory)["endpoints_reached"])
        records = {str(key): value for key, value in (source or {}).items()
                   if isinstance(value, dict) and value.get("record_kind") == "exposure_endpoint"}
        enriched.append(dict(entry, endpoint_records=records,
                             endpoint_records_from=("the terminal status" if terminal is not None
                                                    else "the append-only endpoint journal")))
    return dict(state, trajectories=enriched)


def endpoint_rows(state, *, endpoint_updates, batch_rows):
    """One row per (trajectory, declared exposure endpoint), reached or not.

    Built from the declared grid crossed with the declared endpoints, so a missing
    combination is a row that says it is missing rather than a row that is absent
    from the table.
    """
    rows = []
    for entry in state["trajectories"]:
        reached = {str(key): value for key, value in
                   (entry.get("endpoint_records") or {}).items()}
        for update in sorted(int(value) for value in endpoint_updates):
            record = reached.get(str(update))
            rows.append({
                "trajectory": entry["trajectory"], "arm_id": entry["arm_id"],
                "task": entry["task"], "replay_lambda": float(entry["replay_lambda"]),
                "seed": int(entry["seed"]), "update": int(update),
                "chosen_exposures": int(update) * int(batch_rows),
                "trajectory_status": entry["status"],
                "stop_reason": entry.get("stop_reason"),
                "reached": record is not None,
                "macro_average_precision": (record or {}).get("macro_average_precision"),
                "worst_stratum_average_precision":
                    (record or {}).get("worst_stratum_average_precision"),
                "val_strata": (record or {}).get("val_strata"),
                "forward_kl": ((record or {}).get("preservation") or {}).get("forward_kl_mean"),
                "tenfold_fraction":
                    ((record or {}).get("preservation") or {}).get("tenfold_fraction"),
                "hundredfold_fraction":
                    ((record or {}).get("preservation") or {}).get("hundredfold_fraction"),
                "conditional_kl": ((record or {}).get("preservation") or {}).get(
                    "conditional_kl_mean"),
                "gate_passed": (record or {}).get("gate_passed"),
                "diversity_eligible": (record or {}).get("diversity_eligible"),
                "diversity_failed_gates": (record or {}).get("diversity_failed_gates"),
                "missing_reason": None if record is not None else (
                    f"the trajectory is {entry['status']}"
                    + (f" ({entry['stop_reason']})" if entry.get("stop_reason") else "")
                    + "; no endpoint exists at this exposure and none is substituted")})
    return rows


def stratum_average_precision(row, stratum):
    """One stratum's validation AP from an endpoint row, or ``None`` if it has no rows.

    A declared-but-empty stratum carries ``n: 0`` and no ``average_precision``. That
    is reported as unavailable rather than as a zero: an empty population is missing
    coverage, not a ranking result of nought.
    """
    block = ((row or {}).get("val_strata") or {}).get(str(stratum)) or {}
    value = block.get("average_precision")
    return None if value is None else float(value)


def matched_control_deltas(rows, *, control_lambda=0.0, strata=SELECTION_STRATA):
    """Differences against the matched zero-replay control. Same task, seed and endpoint.

    Descriptive. A positive delta is a validation-ranking observation, not evidence
    of improved binding, and the eligibility flags travel with every row so an
    ineligible endpoint cannot be read as a prespecified improvement.
    """
    control = {(row["task"], row["seed"], row["update"]): row for row in rows
               if float(row["replay_lambda"]) == float(control_lambda) and row["reached"]}
    matched, unavailable = [], []
    for row in rows:
        if float(row["replay_lambda"]) == float(control_lambda):
            continue
        key = (row["task"], row["seed"], row["update"])
        partner = control.get(key)
        if not row["reached"] or partner is None:
            unavailable.append({
                "trajectory": row["trajectory"], "task": row["task"], "seed": row["seed"],
                "update": row["update"], "replay_lambda": row["replay_lambda"],
                "arm_reached": bool(row["reached"]),
                "control_reached": partner is not None,
                "reason": ("no matched zero-replay endpoint at this seed and exposure"
                           if partner is None else row["missing_reason"])})
            continue
        matched.append({
            "trajectory": row["trajectory"], "task": row["task"], "seed": row["seed"],
            "update": row["update"], "replay_lambda": row["replay_lambda"],
            "macro_average_precision": row["macro_average_precision"],
            "control_macro_average_precision": partner["macro_average_precision"],
            # Per stratum as well as macro: the macro average is the number that can
            # move while one training-distance stratum moves the other way, and the
            # strata are the reason this statistic was chosen in the first place.
            "stratum_average_precision": {
                str(name): stratum_average_precision(row, name) for name in strata},
            "control_stratum_average_precision": {
                str(name): stratum_average_precision(partner, name) for name in strata},
            "delta_stratum_average_precision": {
                str(name): _difference(stratum_average_precision(row, name),
                                       stratum_average_precision(partner, name))
                for name in strata},
            "worst_stratum_average_precision": row["worst_stratum_average_precision"],
            "control_worst_stratum_average_precision":
                partner["worst_stratum_average_precision"],
            "delta_worst_stratum_average_precision": _difference(
                row["worst_stratum_average_precision"],
                partner["worst_stratum_average_precision"]),
            "delta_macro_average_precision": _difference(row["macro_average_precision"],
                                                          partner["macro_average_precision"]),
            "forward_kl": row["forward_kl"], "control_forward_kl": partner["forward_kl"],
            "delta_forward_kl": _difference(row["forward_kl"], partner["forward_kl"]),
            "tenfold_fraction": row["tenfold_fraction"],
            "control_tenfold_fraction": partner["tenfold_fraction"],
            "delta_tenfold_fraction": _difference(row["tenfold_fraction"],
                                                   partner["tenfold_fraction"]),
            "both_gate_eligible": bool(row["gate_passed"] and partner["gate_passed"]),
            "both_diversity_eligible": bool(row["diversity_eligible"]
                                            and partner["diversity_eligible"]),
            "control_trajectory": partner["trajectory"]})
    return {"matched": matched, "unavailable": unavailable,
            "strata": [str(name) for name in strata],
            "matched_on": "same task, same parent seed, same exact exposure endpoint",
            "interpretation": ("a descriptive validation-ranking and preservation difference at "
                               "matched exposure. It is not a confirmatory affinity claim, no "
                               "assay endpoint enters it, and an ineligible endpoint keeps its "
                               "flags rather than being dropped."),
            "substitution_policy": ("none. A missing control is reported unavailable; no earlier "
                                    "endpoint, other seed, rolling last-passing state or "
                                    "historical run stands in for one.")}


def _difference(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def pareto_front(rows, *, preservation="forward_kl", ranking="macro_average_precision"):
    """Observed Pareto-efficient points: lower preservation cost, higher ranking.

    A point is on the observed front when no other *reached* point at the same
    parent seed and the same exposure endpoint has both a smaller preservation
    statistic and a larger ranking statistic. That is a statement about the points
    measured here and nothing more: it is not statistical superiority, it is not a
    selection rule, and it does not identify a winner.
    """
    usable = [row for row in rows if row["reached"] and row.get(preservation) is not None
              and row.get(ranking) is not None]
    front = []
    for row in usable:
        dominated = any(
            other is not row
            and other["seed"] == row["seed"] and other["update"] == row["update"]
            and float(other[preservation]) <= float(row[preservation])
            and float(other[ranking]) >= float(row[ranking])
            and (float(other[preservation]) < float(row[preservation])
                 or float(other[ranking]) > float(row[ranking]))
            for other in usable)
        if not dominated:
            front.append({"trajectory": row["trajectory"], "task": row["task"],
                          "replay_lambda": row["replay_lambda"], "seed": row["seed"],
                          "update": row["update"],
                          preservation: row[preservation], ranking: row[ranking],
                          "gate_passed": row["gate_passed"],
                          "diversity_eligible": row["diversity_eligible"]})
    return {"axes": {"preservation": preservation, "ranking": ranking},
            "points": sorted(front, key=lambda entry: (entry["seed"], entry["update"],
                                                       entry["task"],
                                                       entry["replay_lambda"])),
            "grouped_within": "one parent seed and one exposure endpoint",
            "status": ("observed Pareto-efficient among the points that were actually reached. "
                       "Not statistically established superiority, not a selection, not a "
                       "winner.")}


def coverage_summary(state, rows):
    """How much of the declared grid produced a usable endpoint, stated as counts."""
    by_status = {}
    for entry in state["trajectories"]:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    reached = sum(1 for row in rows if row["reached"])
    return {"declared_trajectories": state["total"],
            "trajectory_status_counts": by_status,
            "declared_endpoint_rows": len(rows),
            "reached_endpoint_rows": reached,
            "missing_endpoint_rows": len(rows) - reached,
            "complete": bool(reached == len(rows)),
            "note": ("a missing endpoint is missing coverage. It never becomes a preservation "
                     "finding, a ranking result or an absence of effect.")}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

FIGURE_METADATA = {"Software": None, "Creation Time": None, "Date": None}


def save_figure(figure, path, *, dpi=150, replace=False):
    """Write one figure deterministically, with an explicit export format.

    The format is passed explicitly because the comparison path writes to a
    temporary name whose suffix is not the image format: matplotlib infers the
    format from the suffix, and a temporary called ``x.png.rerender`` has no
    inferable one.

    ``replace`` separates two different artifacts that happen to be PNGs. A figure
    in the ignored run directory describes a campaign that is still running, so it
    is re-rendered as the campaign progresses. A *published* figure is a
    deliverable somebody may have cited, so it is written once and a disagreement
    is reported rather than overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower().lstrip(".") or "png"
    keys = ("Date",) if suffix == "svg" else ("Software", "Creation Time")
    metadata = {key: FIGURE_METADATA[key] for key in keys}
    entry = {"name": path.name, "written": True}
    if path.is_file() and replace:
        figure.savefig(path, dpi=dpi, metadata=metadata, format=suffix)
        entry["sha256"] = paths.sha256_file(path)
        entry["replaced"] = "the run-directory figure tracks a campaign that is still running"
        return path.name, entry
    if path.is_file():
        temporary = path.with_name(f"{path.stem}.rerender.{suffix}")
        figure.savefig(temporary, dpi=dpi, metadata=metadata, format=suffix)
        rendered = paths.sha256_file(temporary)
        temporary.unlink()
        entry["sha256"] = paths.sha256_file(path)
        entry["rerendered_sha256"] = rendered
        if rendered != entry["sha256"]:
            entry["rerendered_bytes_differ"] = {
                "kept": "the original file",
                "reason": ("a completed figure is not replaced in place. The numbers behind it "
                           "are in the published tables either way.")}
        return path.name, entry
    figure.savefig(path, dpi=dpi, metadata=metadata, format=suffix)
    entry["sha256"] = paths.sha256_file(path)
    return path.name, entry


#: Stable per-task styles, so the same method is the same marker in every panel of
#: every figure and a reader does not have to re-learn the key three times.
TASK_STYLES = {"continued_sft": {"marker": "o", "color": "#1f77b4", "label": "continued SFT"},
               "ipo": {"marker": "s", "color": "#d62728", "label": "IPO tau=0.1"}}
OTHER_STYLE = {"marker": "^", "color": "#555555", "label": "other"}


def compact_lambda(value):
    """``λ=0``, ``λ=.01``, ``λ=100`` -- short enough to sit beside a point."""
    number = float(value)
    if number == 0:
        return "λ0"
    text = format(number, "g")
    return "λ" + (text[1:] if text.startswith("0.") else text)


def render_figures(directory, rows, *, prefix="her2-parent-replay-", replace=True):
    """Faceted tradeoff scatters: one panel per (exposure endpoint, parent seed).

    The declared grid puts 108 potential points on each axis pair. Drawing them in
    one 7.4x4.6 frame with a four-line label apiece produces a picture nobody can
    read, which is the failure mode this project has already published once. The
    panel grid is the declared design: exposures down, seeds across, twelve points
    per panel -- two tasks by six lambdas -- with compact lambda labels, one shared
    legend, and a stated count of the endpoints that are absent from each panel so a
    thin panel cannot be mistaken for a thin result.
    """
    figures = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {f"{prefix}tradeoff.png": {
            "written": False, "name": f"{prefix}tradeoff.png",
            "reason": f"matplotlib unavailable: {error}"}}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    updates = sorted({int(row["update"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    tasks = sorted({str(row["task"]) for row in rows})
    for axis, label, name in (
            ("forward_kl", "forward KL to the parent (nats / sequence)", "forward-kl"),
            ("tenfold_fraction", "fraction of parent draws with drop > ln 10", "tenfold"),
            ("hundredfold_fraction", "fraction of parent draws with drop > ln 100",
             "hundredfold")):
        rows_n, columns = max(1, len(updates)), max(1, len(seeds))
        figure, grid = plt.subplots(rows_n, columns, squeeze=False, sharey=True,
                                    figsize=(3.6 * columns, 2.9 * rows_n))
        for down, update in enumerate(updates or [None]):
            for across, seed in enumerate(seeds or [None]):
                panel = grid[down][across]
                cell = [row for row in rows
                        if int(row["update"]) == update and int(row["seed"]) == seed]
                plotted, missing = 0, 0
                for row in cell:
                    if (not row["reached"] or row.get(axis) is None
                            or row.get("macro_average_precision") is None):
                        missing += 1
                        continue
                    style = TASK_STYLES.get(row["task"], OTHER_STYLE)
                    eligible = bool(row["diversity_eligible"])
                    panel.scatter(float(row[axis]), float(row["macro_average_precision"]),
                                  marker=style["marker"], s=30,
                                  facecolors=style["color"] if eligible else "none",
                                  edgecolors=style["color"], linewidths=0.9)
                    panel.annotate(compact_lambda(row["replay_lambda"]),
                                   (float(row[axis]), float(row["macro_average_precision"])),
                                   fontsize=6.5, xytext=(3, 2), textcoords="offset points")
                    plotted += 1
                exposures = next((row["chosen_exposures"] for row in cell), None)
                panel.set_title(
                    f"seed {seed} · {format(int(exposures), ',') if exposures else '—'} exposures",
                    fontsize=8)
                panel.tick_params(labelsize=7)
                if missing:
                    # Stated, not implied by an empty corner of the panel.
                    panel.text(0.02, 0.02, f"{missing} of {len(cell)} not reached",
                               transform=panel.transAxes, fontsize=6.5, color="#777777")
                if plotted == 0:
                    panel.text(0.5, 0.5, "no endpoint reached", transform=panel.transAxes,
                               fontsize=7.5, ha="center", va="center", color="#777777")
                if across == 0:
                    panel.set_ylabel("validation macro-AP", fontsize=8)
                if down == rows_n - 1:
                    panel.set_xlabel(label, fontsize=8)
        handles = [plt.Line2D([], [], linestyle="none",
                              marker=TASK_STYLES.get(task, OTHER_STYLE)["marker"],
                              color=TASK_STYLES.get(task, OTHER_STYLE)["color"],
                              label=TASK_STYLES.get(task, OTHER_STYLE)["label"])
                   for task in tasks]
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                                  color="#777777", label="hollow: diversity-ineligible endpoint"))
        figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5,
                      frameon=False)
        figure.suptitle("Preservation versus ranking at each exposure endpoint and parent seed",
                        fontsize=10)
        figure.tight_layout(rect=(0, 0.05, 1, 0.96))
        figure_name, entry = save_figure(figure, directory / f"{prefix}{name}.png",
                                         replace=replace)
        plt.close(figure)
        figures[figure_name] = dict(
            entry, logical=f"report/figures/{figure_name}",
            panels={"rows": "exposure endpoint", "columns": "parent seed",
                    "points_per_panel": len(tasks) * len({row["replay_lambda"] for row in rows})},
            caption=("validation macro-AP against " + label + ", one panel per exposure endpoint "
                     "and parent seed. Marker shape and colour are the method, the label beside "
                     "each point is its replay lambda, and a hollow marker is an endpoint that "
                     "failed a diversity gate. Endpoints that were not reached are counted in each "
                     "panel and are never plotted as zeros."))
    return figures


# ---------------------------------------------------------------------------
# the narrative
# ---------------------------------------------------------------------------

def render_report(*, config, state, rows, deltas, pareto, coverage, banks, freeze, costs,
                  figures, publication=None):
    """The published narrative. All 36 trajectories appear, whatever they did."""
    screen = config["screen"]
    lines = [
        "# HER2 parent replay: exposure-matched screen", "",
        f"Protocol: [{config['protocol']}](../{config['protocol']}).",
        "",
        "A conditional, exposure-matched comparison of token-level parent distillation",
        "against no replay, at 64 chosen examples per optimizer update and the declared",
        "exposure endpoints. It reports preservation against validation ranking. It is not",
        "an affinity measurement, it is not evidence about newly generated designs, and it",
        "does not select a winner.", "",
        "## Status of the declared grid", "",
        f"- Declared trajectories: **{coverage['declared_trajectories']}** "
        f"({len(screen['tasks'])} tasks x {len(screen['replay_lambdas'])} lambda values x "
        f"{len(screen['parent_seeds'])} parent seeds).",
        f"- Declared endpoint rows: **{coverage['declared_endpoint_rows']}**; reached: "
        f"**{coverage['reached_endpoint_rows']}**; missing: "
        f"**{coverage['missing_endpoint_rows']}**.", ""]
    lines += ["| status | trajectories |", "| --- | --- |"]
    for status, count in sorted(coverage["trajectory_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines += ["", "`stopped_by_gate` is a declared scientific outcome: the parent-relative",
              "chosen-likelihood drop exceeded 1.0 nat per sequence and the trajectory stopped.",
              "`incomplete` is a trajectory that began and did not finish; it is never resumed",
              "and it is never a result at a lower exposure.", ""]

    lines += ["## Every trajectory", "",
              "| trajectory | task | lambda | seed | status | updates | chosen exposures | "
              "checks | endpoints |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in state["trajectories"]:
        exposures = (entry.get("exposures") or {}).get("chosen")
        lines.append(
            f"| `{entry['trajectory']}` | {entry['task']} | {entry['replay_lambda']:g} | "
            f"{entry['seed']} | {entry['status']}"
            + (f" ({entry['stop_reason']})" if entry.get("stop_reason") else "")
            + f" | {entry.get('updates', 0)} | "
            f"{'—' if exposures is None else format(int(exposures), ',')} | "
            f"{entry.get('checks', 0)} | "
            f"{', '.join(entry.get('endpoints_reached') or []) or '—'} |")

    lines += ["", "## Preservation versus ranking at each exposure endpoint", "",
              "| task | lambda | seed | exposures | macro-AP | forward KL | >ln10 | >ln100 | "
              "gate | diversity |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if not row["reached"]:
            lines.append(
                f"| {row['task']} | {row['replay_lambda']:g} | {row['seed']} | "
                f"{row['chosen_exposures']:,} | — | — | — | — | — | not reached |")
            continue
        lines.append(
            f"| {row['task']} | {row['replay_lambda']:g} | {row['seed']} | "
            f"{row['chosen_exposures']:,} | {_fmt(row['macro_average_precision'], 4)} | "
            f"{_fmt(row['forward_kl'], 4)} | {_fmt(row['tenfold_fraction'], 5)} | "
            f"{_fmt(row['hundredfold_fraction'], 5)} | "
            f"{'passed' if row['gate_passed'] else 'failed'} | "
            f"{'eligible' if row['diversity_eligible'] else 'ineligible'} |")
    missing = [row for row in rows if not row["reached"]]
    if missing:
        lines += ["", f"**{len(missing)}** declared endpoint rows were not reached:"]
        for row in missing:
            lines.append(f"- `{row['trajectory']}` at {row['chosen_exposures']:,} exposures — "
                         f"{row['missing_reason']}")

    strata = [str(name) for name in (deltas.get("strata") or SELECTION_STRATA)]
    lines += ["", "## Validation AP in each training-distance stratum", "",
              "The headline statistic is the macro average of these columns. A macro average is",
              "exactly the number that can improve while one stratum moves the other way, so the",
              "strata are printed rather than left in the JSON. Stratum 0 is reported by the",
              "evaluator and is never averaged into the macro.", "",
              "| task | lambda | seed | exposures | "
              + " | ".join(f"AP d={name}" for name in strata) + " | worst | macro |",
              "| --- | --- | --- | --- | " + " | ".join("---" for _ in strata) + " | --- | --- |"]
    for row in rows:
        if not row["reached"]:
            continue
        values = " | ".join(_fmt(stratum_average_precision(row, name), 4) for name in strata)
        lines.append(
            f"| {row['task']} | {row['replay_lambda']:g} | {row['seed']} | "
            f"{row['chosen_exposures']:,} | {values} | "
            f"{_fmt(row['worst_stratum_average_precision'], 4)} | "
            f"{_fmt(row['macro_average_precision'], 4)} |")
    if not any(row["reached"] for row in rows):
        lines.append("| — | — | — | — | " + " | ".join("—" for _ in strata)
                     + " | — | nothing has been reached yet |")

    lines += ["", "## Matched zero-replay differences", "",
              "Same task, same parent seed, same exact exposure endpoint. Descriptive.", "",
              "| task | lambda | seed | exposures | Δ macro-AP | "
              + " | ".join(f"Δ AP d={name}" for name in strata)
              + " | Δ worst | Δ forward KL | Δ >ln10 | both eligible |",
              "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in strata)
              + " | --- | --- | --- | --- |"]
    for row in deltas["matched"]:
        per_stratum = " | ".join(
            _fmt((row.get("delta_stratum_average_precision") or {}).get(name), 4)
            for name in strata)
        lines.append(
            f"| {row['task']} | {row['replay_lambda']:g} | {row['seed']} | "
            f"{row['update'] * int(screen['chosen_per_update']):,} | "
            f"{_fmt(row['delta_macro_average_precision'], 4)} | {per_stratum} | "
            f"{_fmt(row.get('delta_worst_stratum_average_precision'), 4)} | "
            f"{_fmt(row['delta_forward_kl'], 4)} | "
            f"{_fmt(row['delta_tenfold_fraction'], 5)} | "
            f"{'yes' if row['both_gate_eligible'] and row['both_diversity_eligible'] else 'no'} |")
    if not deltas["matched"]:
        lines.append("| — | — | — | — | — | " + " | ".join("—" for _ in strata)
                     + " | — | — | — | no matched pair exists yet |")
    if deltas["unavailable"]:
        lines += ["", "Comparisons that could not be formed, reported rather than skipped:"]
        for row in deltas["unavailable"]:
            lines.append(f"- `{row['trajectory']}` at update {row['update']} — {row['reason']}")

    lines += ["", "## Observed Pareto-efficient points", "",
              f"Axes: {pareto['axes']['ranking']} against {pareto['axes']['preservation']}, "
              f"within {pareto['grouped_within']}.", "",
              "| task | lambda | seed | exposures | ranking | preservation |",
              "| --- | --- | --- | --- | --- | --- |"]
    for point in pareto["points"]:
        lines.append(
            f"| {point['task']} | {point['replay_lambda']:g} | {point['seed']} | "
            f"{point['update'] * int(screen['chosen_per_update']):,} | "
            f"{_fmt(point.get(pareto['axes']['ranking']), 4)} | "
            f"{_fmt(point.get(pareto['axes']['preservation']), 4)} |")
    if not pareto["points"]:
        lines.append("| — | — | — | — | — | nothing has been reached yet |")
    lines += ["", pareto["status"], ""]

    lines += ["## Banks and exposure accounting", "",
              f"- Replay bank: **{banks.get('replay_rows', '—')}** independent temperature-1 "
              "parent draws per seed; duplicates retained.",
              f"- Monitoring bank: **{banks.get('monitor_rows', '—')}** freshly and independently "
              "sampled draws per seed, never trained on.",
              f"- Accidental identity overlap between the two, reported and retained: "
              f"{banks.get('overlap_summary', 'not measured')}.",
              f"- Maximum exposures if every arm reached the largest endpoint: "
              f"**{coverage['declared_trajectories'] * max(int(v) for v in screen['endpoint_updates']) * int(screen['chosen_per_update']):,}** "
              "chosen examples; early stops reduce the actual total.", "",
              "## Cost", "",
              "| category | seconds |", "| --- | --- |"]
    for name, value in sorted((costs or {}).items()):
        lines.append(f"| {name} | {_fmt(value, 1)} |")
    lines += ["", "Optimizer, monitoring, generation, evaluation and I/O are measured separately,",
              "and the teacher-cache build is its own category rather than part of bank",
              "inference: it is the cost the replay arms pay before any of them runs, and",
              "reporting it as zero while it happened would understate exactly that. Bank",
              "preparation is included above. Replay cost is measured, not assumed to be twice",
              "the baseline. GPU seconds are neither a matching variable nor a scientific",
              "endpoint in this screen.", ""]

    lines += ["## What this does not establish", "",
              "- Preservation is distributional. It is not biological quality and not affinity.",
              "- A higher validation macro-AP is a ranking observation on validation rows. It is",
              "  not proof of useful generation and no assay endpoint enters it.",
              "- The monitoring bank supplied diagnostics during the fit, so it is not an",
              "  untouched final test. A confirmatory preservation claim would need a new,",
              "  independently generated bank drawn after the candidates are frozen.",
              "- Unlabelled parent sequences may include poor binders. Distillation resists both",
              "  unwanted forgetting and some desired redistribution; the slope of that frontier",
              "  is an empirical result, not a target.",
              "- No global winner and no scalar combination of preservation and ranking is",
              "  produced by this screen.", ""]
    if freeze:
        lines += ["## Provenance", "",
                  f"- Training specification frozen at commit `{freeze.get('commit')}` "
                  f"({freeze.get('frozen_at')}).",
                  f"- Conditional on the published audit decision "
                  f"`{freeze.get('audit_decision_outcome')}`, audit source freeze "
                  f"`{freeze.get('audit_source_freeze_commit')}`.",
                  "- This training specification was written after the audit and is informed by",
                  "  it. Nothing in it was fixed before the audit ran.", ""]
    if figures:
        lines += ["## Figures", ""]
        for key, entry in sorted(figures.items()):
            name = entry.get("name", key)
            if entry.get("written"):
                lines += [f"![{entry.get('caption')}](figures/{name})", "",
                          f"*{name}* — {entry.get('caption')}", ""]
            else:
                lines.append(f"- `{name}` — not rendered: {entry.get('reason')}")
    if publication:
        lines += ["", "## Data behind this report", "",
                  "Every link below is relative to this file.", ""]
        for logical in sorted(publication["files"]):
            lines.append(f"- [{logical}]({logical})")
        lines.append("")
    lines += ["## Rebuilding this report", "",
              "```", f"python scripts/posttrain_her2_replay.py report --config "
                     f"{config.get('config_path', 'configs/experiments/her2_parent_replay.json')}",
              "```", ""]
    return "\n".join(lines) + "\n"


def _fmt(value, digits):
    if value is None:
        return "—"
    return format(float(value), f".{digits}f")


def require_no_forbidden_claim(text):
    """A last guard on the prose: the claims this screen may not make are grepped for."""
    lowered = str(text).lower()
    found = sorted(phrase for phrase in FORBIDDEN_CLAIMS if phrase in lowered)
    require(not found,
            f"The rendered report contains {found}. This screen establishes a distributional and "
            "a validation-ranking observation; it does not confirm affinity, it does not "
            "independently confirm a previously selected arm, and it names no global winner.")
    return True


def published_tables(*, rows, deltas, pareto, coverage, state, banks):
    """The small, linkable numeric tables the report cites. No per-row vectors."""
    return {"schema_version": REPORT_SCHEMA, "record_kind": "published_replay_tables",
            "coverage": coverage,
            "endpoints": rows,
            "matched_control_deltas": deltas,
            "observed_pareto": pareto,
            "banks": banks,
            "trajectory_statuses": {entry["trajectory"]: {
                "status": entry["status"], "stop_reason": entry.get("stop_reason"),
                "updates": entry.get("updates"), "checks": entry.get("checks"),
                "exposures": entry.get("exposures")}
                for entry in state["trajectories"]},
            "note": ("the per-update journals, the per-check score vectors and the checkpoints "
                     "stay in the ignored run directory; these are the tables the report cites")}


def endpoint_evaluation(*, scored_mean_log_probability, positives, cores, strata, categories,
                        k_values=(32,)):
    """Validation ranking for one endpoint: macro-AP over the fixed strata, plus each one.

    The statistic is the inherited one -- the macro average of per-stratum
    validation AP over training distances 1, 2 and >=3 -- computed through the same
    :mod:`her2_eval` functions every other scorer in this project goes through, so
    a difference here can never be a difference between two metric implementations.
    """
    from .her2_eval import rank_metrics, stratified_metrics
    values = np.asarray(scored_mean_log_probability, dtype=np.float64)
    overall = rank_metrics(values, positives, cores, k_values=tuple(k_values))
    per_stratum = stratified_metrics(values, positives, cores, strata, k_values=tuple(k_values),
                                     categories=tuple(categories))
    macro, worst, mask = macro_average_precision(per_stratum, strata=SELECTION_STRATA)
    return {"val_metrics": overall, "val_strata": per_stratum,
            "macro_average_precision": float(macro),
            "worst_stratum_average_precision": float(worst),
            "strata_used": list(mask),
            "statistic": ("mean of per-stratum validation average precision over the fixed "
                          "training-distance strata [1, 2, >=3]; stratum 0 is reported and never "
                          "averaged in"),
            "inputs": ["validation"],
            "note": "no reserved test label and no assay outcome enters this number"}


def trajectory_endpoint_record(*, row, update, exposures, checkpoint, gate, preservation,
                               evaluation, diversity):
    """The endpoint record, assembled only after its evaluation and bytes exist."""
    require(bool(gate.get("passed")),
            f"{row['trajectory']}: an endpoint record was requested at update {update} for a check "
            "that did not pass. A failed gate never produces a selectable nominal endpoint.")
    forward = (preservation or {}).get("forward_kl") or {}
    tails = ((preservation or {}).get("tails") or {}).get("counts") or {}
    return {
        "schema_version": campaign_lib.ENDPOINT_SCHEMA, "record_kind": "exposure_endpoint",
        "trajectory": row["trajectory"], "arm_id": row["arm_id"], "task": row["task"],
        "replay_lambda": float(row["replay_lambda"]), "seed": int(row["seed"]),
        "update": int(update), "chosen_exposures": int(exposures["chosen"]),
        "exposures": dict(exposures),
        "checkpoint": dict(checkpoint),
        "gate_passed": True, "gate_D": gate.get("D"),
        "macro_average_precision": evaluation["macro_average_precision"],
        "worst_stratum_average_precision": evaluation["worst_stratum_average_precision"],
        "val_metrics": evaluation["val_metrics"], "val_strata": evaluation["val_strata"],
        "diversity": diversity,
        "diversity_eligible": bool(diversity["eligible"]),
        "diversity_failed_gates": list(diversity.get("failed_gates") or []),
        "preservation": {
            "forward_kl_mean": forward.get("mean"),
            "forward_kl_standard_error": forward.get("standard_error"),
            "tenfold_fraction": (tails.get("tenfold") or {}).get("fraction"),
            "hundredfold_fraction": (tails.get("hundredfold") or {}).get("fraction"),
            "conditional_kl_mean": ((preservation or {}).get("conditional_kl") or {}).get("mean"),
            "bank": "fresh independent monitoring draws, never trained on",
            "status": "diagnostic; no preservation stopping rule was declared for this screen"},
        "eligibility_note": ("diversity ineligibility is a distinct observation from a likelihood "
                             "breach. This is still a reached, descriptive endpoint; it is not a "
                             "winner and it is not excluded from the tables."),
        "recorded_at": paths.utc_now()}

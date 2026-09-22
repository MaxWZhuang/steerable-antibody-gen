"""Verify and rebuild a COMPLETED HER2 support-audit report, without re-rendering anything.

Why this exists. The audit's production run completed: 216 shards verified, a
valid decision, a published report. A later full rerun of ``score``/``ches``/
``decide`` also succeeded, and only the ``report`` stage failed -- on a rendering
detail, not on a measurement. ``her2_support.save_figure`` compares an existing
figure by rendering to a temporary named ``<name>.png.rerender`` and calls
``savefig`` without ``format=``; matplotlib infers the format from the suffix and
``.rerender`` is not one. That module is a member of ``AUDIT_SOURCE_FILES``, so
editing it would change the frozen audit identity and invalidate a completed,
published measurement. It therefore stays exactly as it is, the defect stays
documented, and the rerun entry point for a *completed* report is this separate
tool.

What this tool does, and what it refuses:

* it **verifies** rather than regenerates: the frozen identity, the committed
  inventory, the completion manifest *and every shard it binds*, the audit
  completion marker, and every file the publication manifest names -- not only the
  report and the figures, because a corrupt published numeric JSON beside them
  would otherwise earn a "verified" status;
* it **writes nothing into the original ledger**. Recording this verifier as a
  completed artifact of the audit would mean a later tool had appended itself to
  the saved authority that 462 completed artifacts are checked against. The
  compatibility evidence is a separate standalone document, plus the one permitted
  operational progress status;
* it **reuses** the verified figure bytes and never calls ``save_figure``, so the
  legacy defect is not encountered and no figure can be replaced;
* it re-renders both Markdown variants through the audit's own *pure*
  ``render_report`` -- imported, never monkeypatched -- feeding in the **original
  recorded timings**, because that function embeds them and fresh timings would
  guarantee a byte mismatch;
* it compares the results **byte for byte** against the completed report in the
  run directory and the published report under ``reference/``;
* only after all of that does it record an operational "report verified"
  status and a separate compatibility-evidence document carrying this tool's own
  source hash beside the original audit freeze.

No hash check is relaxed, nothing in the audit's frozen sources is imported for
side effects, and no scientific artifact is rewritten.

Examples::

    python scripts/rebuild_her2_support_report.py --help
    python scripts/rebuild_her2_support_report.py --check-only
    python scripts/rebuild_her2_support_report.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_support as support         # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths     # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import require            # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_support_audit.json"
#: Where the compatibility evidence is written. A new file under the audit's
#: evidence tree; no existing evidence byte is touched.
EVIDENCE_RELATIVE = "review/report-compatibility.json"
#: The operational record in the (ignored) run directory.
RUN_RECORD = "report_compatibility.json"
PROGRESS_RECORD = "progress/report.json"
PROGRESS_ORIGINAL = "progress/report.original.json"

#: The captions ``her2_support.render_figures`` attaches to each figure. They are
#: needed because ``render_report`` embeds them in the Markdown and this tool
#: rebuilds that Markdown *without* running the figure renderer. They are not
#: taken on trust: a wrong caption changes the rendered bytes and the byte
#: comparison below fails, which is the check.
FIGURE_CAPTIONS = {
    "her2-support-vs-diversity.png": (
        "tenfold-drop fraction on parent draws against the historical uniqueness of policy "
        "draws; axes are different populations and the plot says so"),
    "her2-support-forward-kl.png": (
        "per-endpoint forward KL with paired-row bootstrap intervals"),
    "her2-support-ches.png": "displacement by parent-CHES decile, per endpoint",
}

#: The known legacy defect, recorded rather than fixed.
LEGACY_DEFECT = {
    "source": "src/smallAntibodyGen/experiments/her2_support.py",
    "function": "save_figure",
    "symptom": ("on a rerun it writes the comparison render to '<name>.png.rerender' and calls "
                "savefig without format=, so matplotlib cannot infer the image format and the "
                "report stage raises"),
    "not_fixed_because": ("her2_support.py is a member of AUDIT_SOURCE_FILES. Editing it changes "
                          "the frozen audit source identity and invalidates a completed, "
                          "published measurement whose freeze marker pins that file's bytes."),
    "consequence": ("the audit's original production of the report succeeded; the defect surfaces "
                    "on a re-render, whose recorded failure is kept. A completed report is "
                    "verified and rebuilt through this separate tool, which never calls "
                    "save_figure."),
}


def figure_entries(context, publication_manifest):
    """The verified figure block ``render_report`` needs, from bytes that already exist.

    Three refusals here, in order: a figure the publication manifest names but this
    tool does not know a caption for (an unexpected publication path); a figure
    whose run-directory bytes no longer match the completion manifest (a corrupt
    figure); and a published copy whose bytes differ from the run-directory
    original (a publication that is not a copy of what was measured).
    """
    settings = context.config["publication"]
    artifacts = (paths.read_completion_manifest(context.run.run_root).get("artifacts") or {})
    published = dict(publication_manifest.get("files") or {})
    named = sorted(name.split("/")[-1] for name in published
                   if name.startswith(f"{settings['figure_directory']}/"))
    unexpected = sorted(set(named) - set(FIGURE_CAPTIONS))
    require(not unexpected,
            f"The publication manifest names figures this tool has no recorded caption for: "
            f"{unexpected}. A figure whose caption is unknown cannot be re-rendered into the "
            "Markdown, and guessing one would produce a document that merely looks right.")
    figures = {}
    for name in named:
        logical = f"report/figures/{name}"
        source = context.run.path(logical)
        require(source.is_file(),
                f"{logical} is named by the publication manifest but is absent from the run "
                "directory; the report cannot be rebuilt from figures that are gone")
        observed = paths.sha256_file(source)
        entry = artifacts.get(context.run.logical(logical))
        require(entry is not None,
                f"{logical} exists but the run's completion manifest never recorded it. The "
                "manifest is the authority a rebuild is checked against; it is not re-derived "
                "from whatever is on disk now.")
        require(observed == entry["sha256"],
                f"{logical} hashes {observed} and the completion manifest recorded "
                f"{entry['sha256']}. This figure changed after it was completed; the rebuild "
                "stops rather than publishing a report that cites it.")
        destination = (context.repository_root / settings["root"]
                       / f"{settings['figure_directory']}/{name}")
        require(destination.is_file(), f"{destination} is missing from the published tree")
        require(destination.read_bytes() == source.read_bytes(),
                f"{destination} differs from the run-directory figure it was copied from")
        require(published[f"{settings['figure_directory']}/{name}"] == observed,
                f"The publication manifest records a different digest for {name}")
        figures[name] = {"name": name, "written": True, "sha256": observed,
                         "logical": logical, "caption": FIGURE_CAPTIONS[name]}
    require(figures, "The publication manifest names no figures; there is nothing to reuse")
    return figures


def published_files(context, publication_manifest, *, report_key):
    """Every file the publication manifest names, re-hashed against it.

    The narrow version of this check -- the report and the figures -- can pass while
    a published numeric deliverable beside them is corrupt, and the status it writes
    would still say the completed report was verified. Every named file is hashed
    here, and a manifest entry that resolves to nothing, or a path that escapes the
    published tree, is refused rather than skipped.
    """
    settings = context.config["publication"]
    root = context.repository_root / settings["root"]
    declared = dict(publication_manifest.get("files") or {})
    require(declared, "The publication manifest names no files")
    checked, problems = {}, []
    for logical, recorded in sorted(declared.items()):
        target = (root / logical).resolve()
        if root.resolve() != target and root.resolve() not in target.parents:
            problems.append(f"{logical} resolves outside the published tree")
            continue
        if not target.is_file():
            problems.append(f"{logical} is named by the publication manifest and is absent")
            continue
        observed = paths.sha256_file(target)
        checked[logical] = {"sha256": observed, "matches": observed == recorded,
                            "manifest_sha256": recorded, "bytes": int(target.stat().st_size),
                            "is_report": logical == report_key}
        if observed != recorded:
            problems.append(f"{logical} hashes {observed} and the publication manifest recorded "
                            f"{recorded}")
    require(not problems,
            "The published audit deliverables do not match their publication manifest: "
            + "; ".join(problems)
            + ". A rebuild that verified only the report and the figures would report a completed "
              "audit as verified while a published numeric file beside them was corrupt.")
    return {"files": checked, "file_count": len(checked),
            "manifest_sha256": paths.sha256_file(
                context.repository_root / settings["root"] / settings["manifest"]),
            "basis": ("every path in the publication manifest, hashed and compared to the digest "
                      "the manifest records for it; nothing is derived from a directory listing")}


def completion_ledger(context):
    """The original run's shard and completion-manifest verification, read-only.

    The audit's own :func:`her2_support.verify_outputs` is called rather than
    reimplemented: it re-reads every shard against its recorded dtypes, shapes,
    order and container hash, and every ledger entry against the bytes on disk. A
    rebuild that skipped it could reproduce a report byte for byte out of summaries
    whose underlying shards had been deleted.
    """
    verification = support.verify_outputs(context)
    require(verification["immutable"],
            "The completed run does not verify against its own completion manifest: "
            + "; ".join(str(problem) for problem in verification["problems"][:10])
            + ". The rebuild stops rather than recording a 'verified' status over it.")
    return {"shards_checked": verification["shards_checked"],
            "artifacts_checked": verification["completion_manifest"]["artifacts_checked"],
            "expected_output_count": verification["expected_output_count"],
            "immutable": True,
            "basis": ("the audit's own verify_outputs, unmodified and read-only: shards against "
                      "their recorded identity, ledger entries against the bytes on disk")}


def completed_documents(context):
    """Every input ``render_report`` reads, each verified against the completion manifest."""
    documents = support.completed_stage(context, {
        "checkpoints": support.CHECKPOINTS_JSON,
        "coverage": support.COVERAGE_JSON,
        "ches": support.CHES_JSON,
        "decision": support.DECISION_JSON,
        "inventory": support.INVENTORY_JSON})
    require(documents is not None,
            "This run has no completed score/ches/decide artifacts bound to its completion "
            "manifest. This tool verifies and rebuilds a COMPLETED report; it does not produce "
            "one.")
    complete_path = context.run.path(support.COMPLETE_JSON)
    require(complete_path.is_file(),
            f"{support.COMPLETE_JSON} is absent. The audit completion marker is written only "
            "after every declared stage completed and the immutability verification passed; "
            "without it there is no completed report to verify.")
    completion = paths.read_json(complete_path)
    require(completion.get("record_kind") == "audit_complete",
            f"{support.COMPLETE_JSON} is not an audit completion marker")
    return documents, completion


def rebuild(context, *, check_only=False):
    """Verify the completed report and prove both Markdown variants still reproduce."""
    started = time.perf_counter()
    marker = support.require_frozen_identity(context)
    documents, completion = completed_documents(context)
    inventory = documents["inventory"]
    support.require_committed_inventory(context, inventory)
    scored = documents["checkpoints"]
    coverage = documents["coverage"]["inventory_coverage"]
    ches = documents["ches"]
    decision = documents["decision"]

    publication_path = (context.repository_root / context.config["publication"]["root"]
                        / context.config["publication"]["manifest"])
    require(publication_path.is_file(),
            f"{publication_path} is absent; the published manifest is what a rebuilt report is "
            "checked against")
    publication_manifest = paths.read_json(publication_path)
    require(publication_manifest.get("record_kind") == "publication_manifest",
            f"{publication_path} is not a publication manifest")
    figures = figure_entries(context, publication_manifest)
    ledger = completion_ledger(context)

    # The ORIGINAL recorded timings. render_report embeds the cost table, so fresh
    # timings would guarantee a mismatch and the comparison would prove nothing.
    timings = support.stage_timings(context)
    require(timings, "No recorded stage timings; the report embeds them and cannot be rebuilt")

    def render(publication):
        return support.render_report(
            context, inventory=inventory, checkpoints=scored["checkpoints"], ches_summary=ches,
            decision=decision, coverage=coverage, timings=timings, figures=figures,
            paired=scored.get("paired_method_differences"), publication=publication)

    run_report_path = context.run.path(support.REPORT_MD)
    require(run_report_path.is_file(), f"{support.REPORT_MD} is absent from the run directory")
    run_bytes = run_report_path.read_bytes()
    run_text = render(None)
    comparisons = {
        "run_directory": _compare(run_text, run_bytes,
                                  label=context.run.logical(support.REPORT_MD))}
    require(completion.get("report", {}).get("sha256") == paths.sha256_bytes(run_bytes),
            "The completion marker records a different digest for the run-directory report than "
            "the bytes now on disk. The marker is the authority; the rebuild stops here.")

    plan = support.publication_plan(context, figures=figures, ches_document=ches)
    declared = sorted(name for name in (publication_manifest.get("files") or {})
                      if name != plan["report"])
    require(plan["files"] == declared,
            f"The publication plan rebuilt here is {plan['files']} and the published manifest "
            f"records {declared}. A rebuilt report must link exactly the files that were "
            "published beside it.")
    published_report_path = context.repository_root / context.config["published_report"]
    require(published_report_path.is_file(),
            f"{context.config['published_report']} is absent from the published tree")
    published_bytes = published_report_path.read_bytes()
    published_text = render(plan)
    comparisons["published"] = _compare(published_text, published_bytes,
                                        label=context.config["published_report"])
    require(publication_manifest["files"][plan["report"]] == paths.sha256_bytes(published_bytes),
            "The publication manifest records a different digest for the published report than "
            "the bytes now on disk")
    publication_files = published_files(context, publication_manifest, report_key=plan["report"])

    for name, block in sorted(comparisons.items()):
        require(block["matches"],
                f"The rebuilt {name} report differs from the bytes on disk "
                f"({block['rendered_sha256']} vs {block['recorded_sha256']}); first difference at "
                f"byte {block['first_difference']}. The original is kept and nothing is "
                "overwritten: a renderer that no longer reproduces a published deliverable is a "
                "finding, not something to write over it.")

    record = {
        "schema_version": paths.AUDIT_SCHEMA, "record_kind": "report_compatibility",
        "audit_id": context.config["audit_id"],
        "tool": {"path": "scripts/rebuild_her2_support_report.py",
                 "sha256": paths.sha256_file(ROOT / "scripts/rebuild_her2_support_report.py"),
                 "role": ("an additive verifier for a completed report. It imports the audit's "
                          "pure render_report and never monkeypatches, re-renders a figure or "
                          "relaxes a hash check.")},
        "audit_freeze": {"commit": marker["git"]["commit"], "frozen_at": marker["frozen_at"],
                         "marker_sha256": paths.sha256_file(
                             support.freeze_marker_path(context)),
                         "source_files": len(marker.get("source_sha256") or {}),
                         "inputs": marker.get("input_count")},
        "completion": {"completed_at": completion.get("completed_at"),
                       "decision_outcome": completion.get("decision_outcome"),
                       "shards_checked": (completion.get("verification") or {}).get(
                           "shards_checked"),
                       "report_sha256": (completion.get("report") or {}).get("sha256")},
        "figures": {name: {"sha256": entry["sha256"], "reused": True,
                           "rendered": False, "caption_source": "recorded in the rebuild tool"}
                    for name, entry in sorted(figures.items())},
        "comparisons": comparisons,
        "completion_ledger": ledger,
        "publication": {"manifest": context.config["publication"]["manifest"],
                        "manifest_sha256": paths.sha256_file(publication_path),
                        "files": sorted(publication_manifest.get("files") or {}),
                        "verified": publication_files},
        "timings_source": ("the originally recorded per-stage timing blocks in the run "
                           "directory; the report embeds them and they are not refreshed"),
        "legacy_defect": dict(LEGACY_DEFECT),
        "claim": ("the completed report reproduces byte for byte from the frozen sources, the "
                  "committed inventory, the completion-manifest-bound summaries and the reused "
                  "figure bytes. This is a compatibility verification of a finished measurement; "
                  "it is not a new audit, a new score or a new scientific claim."),
        # ``recorded_at`` and ``wall_seconds`` are operational keys that
        # ``scientific_projection`` strips, so a second verification run compares
        # equal and rewrites nothing.
        "recorded_at": paths.utc_now(),
        "wall_seconds": time.perf_counter() - started}
    if check_only:
        return record
    # Deliberately NOT ``context=``. Passing the context would call record_completion
    # and append this new verifier's artifact to the ORIGINAL audit's completion
    # manifest -- the saved authority that binds 462 completed artifacts and is the
    # thing a later verification is checked against. A tool that verifies a finished
    # measurement does not get to write into its ledger. The compatibility evidence
    # stands alone and carries its own digests.
    support.require_new_or_identical(context.run.path(RUN_RECORD), record,
                                     what="the report compatibility record")
    _mark_report_verified(context, record)
    evidence = (context.repository_root / context.config["evidence_root"] / EVIDENCE_RELATIVE)
    support.require_new_or_identical(evidence, paths.scrub_host_paths(record),
                                     what="the committed report compatibility evidence")
    return record


def _compare(text, recorded, *, label):
    rendered = text.encode("utf-8")
    first = next((position for position, (left, right)
                  in enumerate(zip(rendered, recorded)) if left != right),
                 min(len(rendered), len(recorded)) if len(rendered) != len(recorded) else None)
    return {"artifact": label, "rendered_bytes": len(rendered), "recorded_bytes": len(recorded),
            "rendered_sha256": paths.sha256_bytes(rendered),
            "recorded_sha256": paths.sha256_bytes(recorded),
            "matches": rendered == recorded,
            "first_difference": first,
            "comparison": "byte for byte, not a normalized or whitespace-insensitive diff"}


def _mark_report_verified(context, record):
    """Record the operational status, preserving the original stage record exactly once.

    The audit's original production of this report **succeeded**; what failed was
    the later re-render, on the legacy ``save_figure`` suffix defect, and that
    failure is evidence about the renderer rather than about the measurement. The
    stage record that carries it is preserved beside the new status rather than
    overwritten, and the new status says which tool established it and on what.
    """
    progress_path = context.run.path(PROGRESS_RECORD)
    original_path = context.run.path(PROGRESS_ORIGINAL)
    if progress_path.is_file() and not original_path.is_file():
        paths.write_json(original_path, dict(paths.read_json(progress_path),
                                             preserved_note=(
                                                 "the report stage record as it stood before this "
                                                 "verification, kept because the later re-render's "
                                                 "recorded failure is evidence about the legacy "
                                                 "renderer. The audit's original production of "
                                                 "this report succeeded.")))
    document = {
        "schema_version": paths.AUDIT_SCHEMA, "record_kind": "stage_progress",
        "stage": "report", "status": "completed", "total": 2, "completed": 2,
        "current": None, "error": None,
        "started_at": (paths.read_json(original_path).get("started_at")
                       if original_path.is_file() else None),
        "verified_by": "scripts/rebuild_her2_support_report.py",
        "verification": {"run_directory": record["comparisons"]["run_directory"]["matches"],
                         "published": record["comparisons"]["published"]["matches"],
                         "figures_reused": sorted(record["figures"]),
                         "tool_sha256": record["tool"]["sha256"]},
        "original_record": PROGRESS_ORIGINAL if original_path.is_file() else None,
        "note": ("the completed report was verified to reproduce byte for byte from the frozen "
                 "sources and the reused figure bytes. The original production of this report "
                 "succeeded; the later re-render's recorded failure is preserved beside this "
                 "record.")}
    if progress_path.is_file():
        existing = paths.read_json(progress_path)
        if paths.scientific_projection(existing) == paths.scientific_projection(document):
            return "unchanged"
    paths.write_json(progress_path, document)
    return "written"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rebuild_her2_support_report.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="audit configuration (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="run root override; defaults to the config's run_root")
    parser.add_argument("--guarded-root", default=None)
    parser.add_argument("--original-root", default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--check-only", action="store_true",
                        help="verify and report, writing nothing at all")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    context = support.resolve_context(
        ROOT, config_path=ROOT / args.config, run_root=args.output,
        root_overrides={"guarded": args.guarded_root, "original": args.original_root,
                        "raw": args.raw_root})
    record = rebuild(context, check_only=args.check_only)
    for name, block in sorted(record["comparisons"].items()):
        print(f"rebuild: {name} report reproduces byte for byte "
              f"({block['recorded_bytes']} bytes, {block['recorded_sha256'][:12]})", flush=True)
    print(f"rebuild: {len(record['figures'])} figures reused verbatim, none re-rendered",
          flush=True)
    if args.check_only:
        print("rebuild: --check-only, nothing written", flush=True)
    else:
        print(f"rebuild: compatibility evidence at "
              f"{context.config['evidence_root']}/{EVIDENCE_RELATIVE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

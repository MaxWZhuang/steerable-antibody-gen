"""Re-analyse a frozen contrast scoring run under the v2 statistical protocol.

Reads the cached artifacts only -- no model, no corpus, no training gate -- so a
full re-analysis costs seconds and cannot perturb the frozen inputs. The report
it writes is bound by hash to the exact scores/manifest it read.

    python scripts/hcdr3_contrast_statistics.py
    python scripts/hcdr3_contrast_statistics.py --effect 0.05 --fold validation

Knob precedence is protocol file < CLI flag, matching scripts/mlm_train.py: the
parser uses argparse.SUPPRESS so an unset flag never clobbers a protocol value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.evaluation.contrasts import file_sha256, load_json, save_json
from smallAntibodyGen.evaluation.contrast_statistics import analyze

# Repo-anchored, never CWD-relative, and pointing at the directory the scoring
# CLI actually writes (Mirror BUG-18/BUG-21: a drifting artifact default shipped
# nulls with exit 0).
DEFAULT_SCORES = ROOT / "outputs" / "hcdr3_contrasts" / "scores.json.gz"
DEFAULT_MANIFEST = ROOT / "outputs" / "hcdr3_contrasts" / "manifest.json.gz"
DEFAULT_PROTOCOL = ROOT / "configs" / "evaluation" / "hcdr3_contrasts_v2.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "hcdr3_contrasts" / "statistics_v2.json"

#: Protocol keys forwarded to `analyze`; anything else in the file is metadata.
ANALYSIS_KEYS = ("min_variants", "fold", "n_resamples", "n_permutations",
                 "n_simulations", "effect", "seed", "alpha")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument("--scores", type=Path, help="cached scores.json(.gz) from the scoring run")
    parser.add_argument("--manifest", type=Path, help="frozen manifest.json(.gz)")
    parser.add_argument("--protocol", type=Path, help="statistical protocol JSON")
    parser.add_argument("--output", type=Path, help="where to write the report")
    parser.add_argument("--fold", type=str, help="restrict to one fold, or 'all'")
    parser.add_argument("--min-variants", type=int, dest="min_variants")
    parser.add_argument("--effect", type=float, help="effect size for the power simulation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--n-resamples", type=int, dest="n_resamples")
    parser.add_argument("--n-permutations", type=int, dest="n_permutations")
    parser.add_argument("--n-simulations", type=int, dest="n_simulations")
    return parser


def resolve_path(args: argparse.Namespace, name: str, default: Path,
                 parser: argparse.ArgumentParser) -> Path:
    """An explicit-but-missing path is a user error; only a default may be absent."""
    supplied = name in vars(args)
    path = Path(vars(args)[name]) if supplied else default
    if not path.exists():
        if supplied:
            parser.error("--{} does not exist: {}".format(name, path))
        parser.error(
            "{} not found: {}\nRun scripts/hcdr3_contrast_benchmark.py score first, "
            "or pass --{} explicitly.".format(name, path, name))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    scores_path = resolve_path(args, "scores", DEFAULT_SCORES, parser)
    manifest_path = resolve_path(args, "manifest", DEFAULT_MANIFEST, parser)
    protocol_path = resolve_path(args, "protocol", DEFAULT_PROTOCOL, parser)
    output_path = Path(vars(args).get("output", DEFAULT_OUTPUT))

    protocol = load_json(protocol_path)
    settings = {k: protocol[k] for k in ANALYSIS_KEYS if k in protocol}
    settings.update({k: v for k, v in vars(args).items() if k in ANALYSIS_KEYS})
    if settings.get("fold") == "all":
        settings["fold"] = None

    report = analyze(load_json(scores_path), load_json(manifest_path), **settings)
    report["provenance"] = {
        "scores": str(scores_path), "scores_sha256": file_sha256(scores_path),
        "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "protocol": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
        "implementation_sha256": file_sha256(
            ROOT / "src" / "smallAntibodyGen" / "evaluation" / "contrast_statistics.py"),
    }
    report["protocol_notes"] = {k: v for k, v in protocol.items() if k not in ANALYSIS_KEYS}
    save_json(output_path, report)

    print(json.dumps({k: report[k] for k in
                      ("population", "primary", "by_target", "weight_concentration",
                       "leave_largest_group_out", "permutation_null", "power",
                       "native_minus_substituted")}, indent=2), flush=True)
    print("wrote {}".format(output_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

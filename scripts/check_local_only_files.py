#!/usr/bin/env python
"""
Refuse to commit local-only files. Versioned so it survives a clone.

The guardrail previously lived only in `.git/hooks/pre-commit`, which git does
not version and `git clone` does not copy. So the policy protected exactly one
working copy: a fresh clone, or a teammate, or CI, had no guardrail at all. This
script is the policy; the hook is a thin caller.

Categories are local-only for different reasons and the distinction matters:

- ``docs/`` is **internal**. The bug log, the architecture contract, the steering
  plans, and the research notes are the owner's working material, not the public
  repository's. There are no exemptions, and adding one is a publishing decision
  rather than a config change -- in this repository "tracked" and "public" are
  the same thing the moment anyone pushes.
- ``outputs/``, ``checkpoints/``, ``wandb/``, ``logs/`` and the large binary
  suffixes are run artifacts: machine-specific, large, or both.
- ``data/raw/`` and ``data/processed/`` are corpora, which are neither ours to
  redistribute nor small enough to want to.
- ``.claude/`` and ``CLAUDE.md`` are local tooling configuration.

Usage::

    python scripts/check_local_only_files.py            # check the git index
    python scripts/check_local_only_files.py --ref HEAD # check a commit's tree
    python scripts/check_local_only_files.py a.py b.md  # check explicit paths

Exit code 0 when clean, 1 when something local-only is present.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: One pattern per category, so a failure can say WHY a path is blocked rather
#: than only that it matched something.
LOCAL_ONLY_RULES: tuple[tuple[str, str], ...] = (
    (r"^docs/", "internal documents (bug log, architecture contract, plans, research notes)"),
    (r"^CLAUDE\.md$", "local tooling configuration"),
    (r"^\.claude/", "local tooling configuration"),
    (r"^outputs/", "run artifacts and local evidence"),
    (r"^checkpoints/", "trained weights"),
    (r"^wandb/", "run artifacts"),
    (r"^logs/", "run artifacts"),
    (r"\.log$", "run artifacts"),
    (r"^data/raw/", "corpora"),
    (r"^data/processed/", "corpora"),
    (r"\.(pt|pth|gz|zip|tar)$", "large binary artifacts"),
    (r"^\.tmp", "scratch files"),
    (r"\.tmp$", "scratch files"),
    (r"\.tmp\.json$", "scratch files"),
)

_COMPILED = tuple((re.compile(pattern), reason) for pattern, reason in LOCAL_ONLY_RULES)


def classify(path: str) -> str | None:
    """Return why ``path`` is local-only, or ``None`` if it may be committed."""
    for pattern, reason in _COMPILED:
        if pattern.search(path):
            return reason
    return None


def staged_paths() -> list[str]:
    """Paths added or modified in the index."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def tree_paths(ref: str) -> list[str]:
    """Every path in ``ref``'s tree -- used to audit a commit, not just an index."""
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def check(paths: list[str]) -> list[tuple[str, str]]:
    """Return ``(path, reason)`` for every local-only path, sorted."""
    found = [(p, reason) for p in paths if (reason := classify(p)) is not None]
    return sorted(found)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "paths",
        nargs="*",
        help="Explicit paths to check. Default: the git index.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Check every path in this ref's tree instead of the index (e.g. HEAD, origin/main).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.paths:
        paths, source = list(args.paths), "the given paths"
    elif args.ref:
        paths, source = tree_paths(args.ref), f"the tree of {args.ref}"
    else:
        paths, source = staged_paths(), "the git index"

    offenders = check(paths)
    if not offenders:
        return 0

    print(f"refusing: local-only files in {source}:", file=sys.stderr)
    for path, reason in offenders:
        print(f"  - {path}   [{reason}]", file=sys.stderr)
    print("", file=sys.stderr)
    print("These are not for the public repository.", file=sys.stderr)
    print("Unstage:  git restore --staged <file>", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Do NOT add an exemption to make this pass. Tracking a file that is "
        "currently local is a publishing decision, not a config change.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

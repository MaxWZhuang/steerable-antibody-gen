#!/usr/bin/env python
"""Write ``docs/INDEX.md``: a LOCAL, informational navigation index of docs/, specs/ and reference/.

Those three trees are git-ignored BY POLICY (``scripts/check_local_only_files.py``: they are the owner's
internal working material, and making them trackable is a publishing decision), so the index is local too.

What it is: a linked inventory with the evidence found for each file, shown with its uncertainty.
What it is not: an archive authority. It never says a file is unused; the weakest label is
``NO-REFERENCE-FOUND-IN-SCANNED-SOURCES`` and the page lists everything that was NOT scanned. It never infers
"current" or "obsolete" from a modification time, never moves or deletes anything, and writes exactly one
file, ``docs/INDEX.md``, only if that file is absent or already carries the generated marker.

How a label can be wrong, and the direction it errs in (two reviewers required this to be over-protective):

* evidence categories are kept apart: ``code`` (tracked sources), ``policy`` (``.gitignore``, the commit-policy
  script, this script and its test: they NAME the trees without consuming them, so they never count),
  ``notes`` (other files in the trees), ``loop`` (ignored loop records), ``outputs`` (the top level of each
  campaign directory under outputs/, where freeze markers and manifests live);
* a path counts if any ``docs``/``specs``/``reference`` component starts a literal that names the file, so
  ``../../docs/A.md`` and ``C:/work/repo/docs/A.md`` count (over-protection), ``mydocs/A.md`` does not;
* a tree NAMED ON ITS OWN ("keep docs/ tidy") says nothing about any one file and tags nothing, except as a
  QUOTED literal in a code or config source (``CONTRACT_ROOTS = ("specs",)``: ``experiment.py`` hashes every
  spec file), where it means the whole tree. Without this rule one sentence tags every file in the tree. For the
  same reason prose is not code: a comment, a docstring, a URL (``https://host/docs``), the text of a quoted shell
  string (``echo "read docs"``), a here-document, a PowerShell block comment or here-string, a YAML block scalar and
  the words of a plain config value (``note: Notes, docs``) never name a tree. Each language's quotes, escapes,
  comments and continuations are read by one small lexer (``python_logical_lines``, ``lex_line``, ``blank_blocks``,
  ``blank_urls``) so that every scanner sees code: two Python statements never merge, one string is never split;
* a path built with a placeholder, an f-string, a ``/ variable`` or a ``+ variable`` tail marks, as
  ``UNRESOLVED-POSSIBLE-REFERENCE``, the files below the literal prefix whose names could start with the literal
  head of the tail. A fully literal chain to a path that does not exist (a test fixture) names no existing file
  and marks nothing;
* a tree named in a config as a bare YAML/INI/TOML value (``root: reference``, ``- root: reference``) or as an UNQUOTED item
  INSIDE a flow collection or INI list (``roots: [docs, x]``, ``roots = docs, x``) means the whole tree, like a quoted
  literal; so does a tree named as an unquoted argument of a shell, PowerShell or batch command (``find docs``,
  ``Get-ChildItem docs -Recurse``). A complete config value is resolved as ONE path (a blank in it, or a JSON
  ``\\u00e9`` escape, does not split it) and a URL never is;
* a path is normalised before lookup (``docs/./a``, ``docs/b/../a``, ``docs/../specs/x``); ``[0-9]`` in a path is a glob
  class, so it is an unresolved component; Python's adjacent literals (``'a_' 'b'``) and literals added to each other
  (``'docs/' + 'A.md'``) are one string, a shell word glued to a quote (``reference/"run$id.json"``) is one argument, and
  a ``+`` or ``/`` that continues on the next line still counts as a tail. A candidate path spans at most twice the deepest
  inventoried path plus six components (``..`` can cancel the rest); longer ones are counted on the page, not followed;
* matching is case-insensitive, as the Windows filesystem resolves paths; every record keeps the real spelling;
* a source is decoded by its BOM (UTF-8, UTF-16, UTF-32; Windows PowerShell writes UTF-16 by default) or, without one,
  as UTF-16 when a quarter of its first 4 KB is NUL; a file that still holds NULs is counted on the page;
* ``import a.b, c`` marks every ``.py`` file sharing the stem of ANY name in the statement, because ``sys.path``
  insertions cannot be resolved; the statement is read from the LOGICAL line, so ``if TYPE_CHECKING: import x``, a
  parenthesised list with comments and a backslash continuation all count, and an ``import`` in a string does not;
* a notebook is read cell by cell (nbformat 3 and 4): a code cell is Python, a prose cell or an output is plain text
  where a tree named on its own means nothing; the evidence "line" is the cell number;
* nothing executes a path expression, imports a local module or follows a link. A tracked file below a junctioned
  ancestor is skipped and listed. Campaign directories under outputs/ that are junctions are NOT followed, so a file
  named only inside one can show a weaker tier than it has; the page says so.

Cost model, each point pinned by a test with a time limit: every source is read once (3 MiB cap) and scanned in time
linear in its size, with no pattern that backtracks across lines or blanks (the lexers jump from quote to quote with
``str.find``, never scan ahead for a newline per string, and count a run of escape characters once); the inventory holds O(depth) keys per
file; a source that marks a subtree costs one mark per file in it (S sources marking F files each: S x F marks, about
1 us each); the page lists at most 5 pieces of evidence per file and category, 8 hubs and 2,000 constructions, and
says how many it left out.

Usage::

    python -B scripts/workspace_index.py              # write docs/INDEX.md
    python -B scripts/workspace_index.py --check      # print it, write nothing
    python -B scripts/workspace_index.py --no-loop    # skip the (slow) loop-record scan

Exit codes: 0 ok; 1 the write failed; 2 refused (docs/ missing or linked, or docs/INDEX.md exists and is not
generated); 3 git failed (an empty source list is never silently accepted).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import bisect  # noqa: E402
import codecs  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple  # noqa: E402
from urllib.parse import quote  # noqa: E402

TREES = ("docs", "specs", "reference")
OUTPUT = "docs/INDEX.md"
MARKER = ("<!-- GENERATED by scripts/workspace_index.py - do not edit; "
          "regenerate with: python -B scripts/workspace_index.py -->")
MARKER_BYTES = MARKER.encode("utf-8")
SIZE_CAP = 3 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".ps1", ".sh", ".csv",
                 ".tsv", ".rst", ".ipynb", ".bat", ".cmd"}
TEXT_NAMES = {".gitignore", ".gitattributes", "Makefile", "LICENSE"}
SHELL_SUFFIXES = {".sh", ".ps1", ".bat", ".cmd"}                    # quoted and unquoted words glue into one path argument
SCRIPT_SUFFIXES = {".py"} | SHELL_SUFFIXES                          # can build a path out of literals and operators
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}  # hold quoted and unquoted path values
FLOW_SUFFIXES = {".yaml", ".yml", ".cfg", ".ini"}                       # ... where an UNQUOTED value can be a list item (JSON, TOML quote)
CODE_SUFFIXES = SCRIPT_SUFFIXES | CONFIG_SUFFIXES | {".ipynb"}       # a notebook's code cells are read as Python
PROSE_SUFFIXES = {".md", ".txt", ".rst"}
LOOP_DIR = ".claude-codex-loop/runs"
LOOP_SHALLOW_DEPTH = 3
LOOP_DEEP_SUFFIXES = {".json", ".md", ".txt", ".ps1", ".sh"}
OUTPUTS_DIR = "outputs"
POLICY_FILES = {".gitignore", ".gitattributes", "scripts/check_local_only_files.py", "scripts/workspace_index.py",
                "src/smallAntibodyGen/tests/test_workspace_index.py", "src/smallAntibodyGen/tests/test_workspace_index_fuzz.py"}
ENTRY_POINT_NAMES = ("README.md", "ARCHITECTURE.md", "CODE_MAP.md", "BUGLOG.md")
PROTECTED_RULES = (
    (lambda path: path == "docs/ARCHITECTURE.md", "AGENTS.md names it a living integration contract"),
    (lambda path: path.startswith("reference/evidence/"), "landed audit freezes and manifests hash these bytes"),
)

T_PATH = "PATH-REFERENCE-IN-TRACKED-SOURCES"
T_SUBTREE = "SUBTREE-REFERENCE-IN-TRACKED-SOURCES"
T_UNRESOLVED = "UNRESOLVED-POSSIBLE-REFERENCE"
T_IMPORT = "IMPORT-BY-STEM"
T_NAME = "NAME-MENTION-IN-TRACKED-SOURCES"
T_NOTES = "MENTIONED-BY-LOCAL-NOTES"
T_LOOP = "MENTIONED-IN-LOOP-RECORDS"
T_OUTPUTS = "NAMED-IN-OUTPUT-RECORDS"
T_POLICY = "POLICY-MENTION-ONLY"
T_NONE = "NO-REFERENCE-FOUND-IN-SCANNED-SOURCES"
REFERENCE_TAGS = {T_PATH, T_SUBTREE, T_UNRESOLVED, T_IMPORT, T_NAME, T_NOTES, T_LOOP, T_OUTPUTS}
EVIDENCE_CAP = 5
MAX_CONSTRUCTIONS = 2000
ODD_CONFIRMATIONS = 100_000
HUB_MIN_LINKS = 3

# Every pattern below is linear on a multi-megabyte line: a character class under a plain "+", or a match anchored
# at a known offset. The first real run took 269 s on quadratic patterns, so none of these may backtrack.
RUN = re.compile(r"[A-Za-z0-9_.\-{}*?%$~/\[\]]+")         # a maximal run of path characters (backslashes folded to "/")
WORD = re.compile(r"[A-Za-z0-9_.\-]+")
PLACEHOLDER = re.compile(r"[{}%*?$\[]")                    # "[" opens a glob character class: run[0-9].json
# an UNQUOTED command-line word that ENDS in a tree name, with any path in front of it (``../docs``, ``$PSScriptRoot\..\docs``).
# Quoted words are read as literals; a quote after the name would be the start of glued text (``reference/"run$id.json"``).
# The prefix class excludes "=" and "," so a token never extends over another delimiter and the scan stays linear.
ARG_ROOT = re.compile(r"(?:^|[\s;|&(=,])(?:[^\s;|&()\"'=,]*[/\\])?(docs|specs|reference)[/\\]?(?=$|[\s;|&),])", re.IGNORECASE)
BRACE_ALTERNATIVES = re.compile(r"\{([^{}\s]{1,200})\}")                  # a shell brace set: {docs,specs}/*.md
# an unknown base directory in front of a path: a Windows namespace prefix, a file: URI, or a variable followed by a separator
BASE_PREFIX = re.compile(r"(?://[?.]/|file:/{2,3}|(?:\$\{[^}/]*\}|\$\([^)/]*\)|\$[A-Za-z_]\w*|%~?[A-Za-z0-9_]+%?|\{[A-Za-z_]\w*\})(?=/|\.\.|$))")
TRANSFORM_TAIL = re.compile(r"[\s)]*\.\s*(with_suffix|with_name|with_stem)\s*\(")      # Path('a/b').with_suffix('.json')
MD_ANGLE = re.compile(r"\]\(<([^>\n]{1,500})>")                           # [text](<a destination with spaces>)
BLOCK_SCALAR = re.compile(r"[|>](?:[+-]\d*|\d+[+-]?)?")                    # a YAML block scalar header: >-  |  >2  |-2  |2-
LITERAL = re.compile(r"""(?<![A-Za-z0-9_])(?P<prefix>[rRbBuUfF]{0,2})(?:'(?P<single>[^'\n]*)'|"(?P<double>[^"\n]*)")""")
SLASH_BETWEEN = re.compile(r"[\s)]*/\s*")                  # ``Path('a/b') / 'c'``: the call closes before the operator
COMMA_BETWEEN = re.compile(r"\s*,\s*")
PLUS_BETWEEN = re.compile(r"\s*\+\s*")                     # ``'docs/' + 'A.md'``: two literals added are ONE string
SLASH_AFTER = re.compile(r"[\s)]*/")
VARIABLE_TAIL = re.compile(r"[\s)]*/\s*[A-Za-z_(\[]")
PLUS_TAIL = re.compile(r"[\s)]*\+")                                   # 'docs/x/result_' + name: the literal is only a HEAD
PATH_CALL = re.compile(r"(?:Path|PurePath|PurePosixPath|PureWindowsPath|join|joinpath)\([^()]*$")
# one ``import`` statement of a LOGICAL line (comments gone, strings blanked): ``import a.b, c as d``, ``from a.b import c, d``,
# ``from a import (b, c)``, and the same after ``if x:`` / ``try:`` / ``;``. Names run to the next ``;``.
IMPORT_STATEMENT = re.compile(
    r"(?:^|[;:])[ \t]*(?:from[ \t]+([A-Za-z_.][\w.]*)[ \t]+import[ \t]+([^;]*)|import[ \t]+([^;]*))")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
ANCHOR_OR_TAG = re.compile(r"^(?:[&!][^\s]*\s*)+")
# ``- root: value`` / ``- "root": value``: the (possibly quoted) key of a mapping in a sequence
SEQUENCE_KEY = re.compile(r"""[ \t]*(?:"[^"\n]*"|'[^'\n]*'|[^\s:#'"\[\]{},&*!|>](?:[^:]|:(?![ \t]|$))*):(?=\s|$)[ \t]*""")   # a colon ends a key only before a blank
LEADING_ANCHORS = re.compile(r"(?:[&!][^\s]*\s*)+")                       # ``&a``, ``!!str``: skipped with ``match(line, pos)``
SEQUENCE_DASH = re.compile(r"[ \t]*-(?:[ \t]+|$)")                         # a sequence dash and its blanks: ``- `` and the inner one of ``- - docs``
HEREDOC_PART = re.compile(r"""'([^']*)'|"((?:[^"\\]|\\.)*)"|\\(.)""", re.DOTALL)      # the quoting of a delimiter word, as a shell removes it
HEREDOC_DQ_ESCAPE = re.compile(r'\\(["\\$`])')                                          # inside "...": a backslash escapes only these
FLOW_DELIMITER = re.compile(r"[\[\]{},]")
FLOW_QUOTED_KEY = re.compile(r"[ \t]*\x01+[ \t]*:")                       # a blanked quoted key, then ``:`` with or without a blank
SECTION_HEADER = re.compile(r"[ \t]*\[\[?[^\[\]]*\]\]?[ \t]*$")           # a TOML / INI section: ``[docs]`` is a name, not a list
VALUE_MARK = re.compile(r"[:=]")
TOML_QUOTE = re.compile(r"""['"#]""")                                      # what can start a string or a comment on a TOML line
URL_MARK = "\x02"                                                         # what a blanked URL becomes; a literal that holds it is skipped
URL_STOP = re.compile(r"[ \t\r\n\"'`<>);|]")                             # where a URL ends in prose, a script or a comment
URL_STOP_PLAIN = re.compile(r"[ \t\r\n]")                                # an unquoted YAML / INI scalar outside a collection: to a blank
URL_STOP_FLOW = re.compile(r"[ \t\r\n,\]}]")                             # ... and an item of a flow collection: ``[https://x,docs]``
URL_STOP_DOUBLE = re.compile(r'[ \t\r\n"]')                              # inside "...": to the closing quote (``a(b)``, ``a;b``, ``o's``)
URL_STOP_SINGLE = re.compile(r"[ \t\r\n']")                              # inside '...': likewise (``say"hi"``)
URL_STOP_DOUBLE_F = re.compile(r'[ \t\r\n"{]')                           # inside f"...": a ``{`` starts a field, which is code
URL_STOP_SINGLE_F = re.compile(r"[ \t\r\n'{]")
QUOTE_CHAR = re.compile(r"""["']""")
FLOW_TOKEN = re.compile(r"""["'\[\]{}]""")                               # in YAML / INI: quotes and the brackets of flow collections
YAML_SEPARATOR = re.compile(r":(?=[ \t]|$)|=")                           # a mapping separator is ``: `` (``sphinx:root: x`` has key ``sphinx:root``)
NOT_NEWLINE = re.compile(r"[^\n]")
PS_BLOCK = re.compile(r"<#|@(['\"])[ \t]*\n")                            # a PowerShell block comment or here-string
HEREDOC = re.compile(r"""(?<!<)<<(-?)[ \t]*((?:[^\s|&;()<>'"\\]|'[^'\n]*'|"(?:[^"\\\n]|\\.)*"|\\\S)+)""")   # a here-document and its WHOLE word
CASE_SEPARATORS = ("|", ";;", ";&", ";;&")                                # what may precede a case pattern ``docs)`` on its line
CASE_TAIL = re.compile(r"(?:\|[^\s|()]*)*\)")                             # ``|specs|x)``: the rest of a case pattern list
BATCH_REM = re.compile(r"(?:^|[&|(@])[ \t]*(?:rem(?=[ \t]|$)|::)", re.IGNORECASE)   # a batch comment after a command separator
SUB_TOKEN_SH = re.compile(r"""\$\(|[()'"\\]""")                            # what changes the quoting or nesting state of a shell line
SUB_TOKEN_PS = re.compile(r"""\$\(|[()'"`]""")                             # ... in PowerShell, where a backtick is the escape
PARAMETER_DEFAULT = re.compile(r"""\$\{(?:[A-Za-z_]\w*|\d+):?[-=+?]([^}`\s]{1,200})\}""")      # ${DOCS:-docs}: the default word
BATCH_SET = re.compile(r'(?i)(?:^|[&|(@])[ \t]*set[ \t]+(?:/[a-z][ \t]+)*"[^="]+=([^"]*)"')      # set "DOCS=docs"
BATCH_FOR_COMMAND = re.compile(r"(?i)\bin[ \t]*\([ \t]*(?:'([^']*)'|`([^`]*)`)")                 # for /f %i in ('dir docs /b') do
FIELD_TOKEN = re.compile(r"""[(){}\[\]'":!]""")                          # what ends or nests the EXPRESSION of an f-string field
SPEC_TOKEN = re.compile(r"[{}]")                                          # ... and, in its format spec, the only syntax there is
F_STRING = re.compile(r"""(?<![A-Za-z0-9_])[rRbBuU]?[fF][rR]?['"]""")      # a quick look: does this line hold an f-string at all
F_PREFIX = re.compile(r"(?<![A-Za-z0-9_])[rRbBuU]?[fF][rR]?$")             # the prefix letters just before a quote make it an f-string
MAGIC_LINE = re.compile(r"!(?!=)|%%?[A-Za-z]")                             # a notebook shell line (``!ls``) or magic (``%cd``); not ``!=`` or ``% x``
ASSIGNED_ESCAPE = re.compile(r"[A-Za-z_][\w.]*(?:[ \t]*,[ \t]*[A-Za-z_][\w.]*)*[ \t]*=[ \t]*!(?!=)")     # ``files = !ls docs``: the shell output captured
SHELL_CELL_MAGICS = frozenset({"bash", "sh", "script", "shell"})            # ``%%bash``: the rest of the cell is a shell script
ESCAPED_TICK = re.compile(r"""(?<!`)((?:``)*)`(['"])""")                  # PowerShell: ``"`` inside a double-quoted string
SCHEME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+.-")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
HEADING_WINDOW = 300                                                    # a title is short; never match a 3 MB line
WINDOW = 96
HIGH = chr(0x10FFFF)                                   # sorts after every character a file name can hold
NAME_BEFORE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")   # a name does not start mid-word
NAME_AFTER = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")     # ... or end mid-word


class GitError(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


@dataclass
class Evidence:
    category: str
    source: str
    line: int
    form: str


@dataclass
class Construction:
    source: str
    line: int
    prefix: str
    head: str
    matched: int
    text: str


@dataclass
class Target:
    path: str
    tracked: bool
    title: str
    size: int
    mtime_ns: int
    tags: set = field(default_factory=set)
    evidence: dict = field(default_factory=lambda: defaultdict(list))          # at most EVIDENCE_CAP per category
    source_counts: dict = field(default_factory=lambda: defaultdict(int))      # distinct sources per category (a count, not a set)
    named_count: int = 0                                                        # distinct tracked sources naming it by path
    seen: set = field(default_factory=set)


@dataclass
class Model:
    head: str = ""
    targets: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)
    constructions: list = field(default_factory=list)
    hubs: list = field(default_factory=list)
    outgoing: dict = field(default_factory=lambda: defaultdict(set))
    policy_roots: dict = field(default_factory=lambda: defaultdict(set))


# ------------------------------------------------------------------------------------------------ helpers

def is_redirect(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def redirected_component(repo, relative):
    """The first component of ``relative`` (below ``repo``) that is a link or junction, else None."""
    here = Path(repo)
    for part in Path(relative).parts:
        here = here / part
        try:
            if is_redirect(os.lstat(here)):
                return here.relative_to(repo).as_posix()
        except OSError:
            return None
    return None


def make_link_guard(repo):
    """``blocked(relative)``: the first component of a path below ``repo`` that is a link or junction, else None.
    Checked for every ancestor, not only the file: a file under a junctioned directory is read THROUGH the junction."""
    clean = set()

    def blocked(relative):
        here = Path(repo)
        parts = Path(relative).parts
        for k, part in enumerate(parts):
            here = here / part
            if parts[:k + 1] in clean:
                continue
            try:
                if is_redirect(os.lstat(here)):
                    return here.relative_to(repo).as_posix()
            except OSError:
                return None                                  # missing or unreadable: the read reports it
            clean.add(parts[:k + 1])
        return None

    return blocked


def git(repo, *args):
    try:
        done = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    except OSError as error:
        raise GitError(f"git could not run: {error}") from error
    if done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}: {done.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return done.stdout


def note_unreadable(gaps, path):
    gaps["unreadable"] += 1
    if len(gaps["unreadable_paths"]) < 5:
        gaps["unreadable_paths"].append(str(path))


def read_text(path, gaps, scope):
    """The file's text with LF newlines, or None when it was skipped. Every skip is counted; nothing raises."""
    links = "links_skipped" if scope in ("tracked", "local") else f"{scope}_links_skipped"
    try:
        info = os.lstat(path)
        if is_redirect(info):
            gaps[links].append(str(path))
            return None
        if not stat.S_ISREG(info.st_mode):
            return None                                     # a directory named like a file, a device: not text to read
        if info.st_size > SIZE_CAP:
            gaps[f"{scope}_skipped_size"] += 1
            return None
        data = Path(path).read_bytes()
    except OSError:
        note_unreadable(gaps, path)
        return None
    text = decode_text(data)
    if "\ufffd" in text:
        gaps["decode_replaced"] += 1
    if "\x00" in text:
        gaps["nul_bytes_files"] += 1                       # still binary after decoding: only its readable parts can be scanned
    return text.replace("\r\n", "\n")


def decode_text(data):
    """Bytes to text. A BOM names the encoding (UTF-8, UTF-16, UTF-32: Windows PowerShell writes UTF-16 by default, and
    read as UTF-8 that is a NUL between every character, which no pattern can match); a BOM-less file with a quarter or
    more NUL bytes in its first 4 KB is wide text (UTF-16, endianness from where the NULs sit); everything else is UTF-8
    with replacement characters."""
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="replace")
    sample = data[:4096]
    if len(sample) >= 4 and sample.count(0) * 4 >= len(sample):
        encoding = "utf-16-le" if sample[1::2].count(0) >= sample[0::2].count(0) else "utf-16-be"
        return data.decode(encoding, errors="replace")
    return data.decode("utf-8", errors="replace")


def is_generated(text):
    """A file whose FIRST line is the marker, ignoring blanks (``decode_text`` has already dropped a BOM): a saved COPY of
    the index (an editor or a PowerShell redirect adds a BOM) must never become a source of references. A note that merely
    quotes the marker further down stays a source. Overwriting is stricter: see ``write_index``."""
    return text.split("\n", 1)[0].strip() == MARKER


# ------------------------------------------------------------------------------------------------ lexing
# What a comment, a docstring, a URL or a quote is decides what a tree name means, so each language's lexical rules are read
# here, once, and every scanner below sees code: two statements never merge, one string is never split, and prose is not code.

def blank_urls(text, flow=False, lists=False):
    """``text`` with every URL that has a network authority (``https://host/docs``) replaced by URL_MARK characters of the same
    length: a URL ending in ``/docs`` is not the local docs tree, and what a URL leaves behind (``"https://host" + "/docs"``)
    still carries the mark, so a literal that holds one is skipped whole. A ``file:`` URI and ``scheme:///path`` (no authority:
    ``sqlite:///docs/x.db``) name local paths and stay. A URL never spans a line, so offsets and line numbers hold.

    Where a URL ends depends on what surrounds it, tracked from the start of the line: INSIDE a quoted string it runs to that
    string's own closing quote or a blank (``"https://x.org/a(b);c/o's/docs"``; an escaped quote does not close it, except that
    a YAML single-quoted scalar has no backslash escape and a doubled quote, ``'o''s'``, is an apostrophe); in a
    YAML or INI file (``flow``) an UNQUOTED one is a plain scalar, which runs to a blank, and to a comma or closing bracket too
    inside a flow collection or an INI list (``lists``: ``roots = https://x.org,docs``) (``[https://x.org,docs]`` has two
    items; ``search(q='docs')`` is one URL); elsewhere it ends at
    ``) ; | < >`` or a quote. ``{`` always ends it: an f-string field or a template placeholder is not part of the URL. In a
    YAML or INI file a quote opens a string only where a scalar begins (``Don't`` opens none). An IPv6 authority keeps its
    brackets (``https://[::1]/docs``). Linear: every ``find`` resumes after the last URL, the scheme is looked for in at most
    32 characters, the authority in at most 64, and the quotes and brackets between two URLs are visited once."""
    at = text.find("://")
    if at < 0:
        return text
    chunks, last, size = [], 0, len(text)
    counted, active, depth, fstring = 0, None, 0, False       # quotes and brackets are tracked up to ``counted``; ``active``: the open quote
    track = FLOW_TOKEN if flow else QUOTE_CHAR
    while at >= 0:
        begin = at
        while begin > last and at - begin < 32 and text[begin - 1] in SCHEME_CHARS:
            begin -= 1
        while begin < at and not text[begin].isalpha():              # a scheme starts with a letter
            begin += 1
        scheme, after = text[begin:at], text[at + 3:at + 4]
        if len(scheme) >= 2 and scheme.lower() != "file" and after not in ("", "/"):
            resume = at + 3
            if after == "[":                                        # an IPv6 authority: its closing bracket is not a boundary
                bracket = text.find("]", resume, min(size, resume + 64))
                if bracket >= 0 and "\n" not in text[resume:bracket]:
                    resume = bracket + 1
            chunk = text[counted:begin]
            newline = chunk.rfind("\n")
            if newline >= 0:
                chunk, active, depth = chunk[newline + 1:], None, 0
            skip = -1
            for token in track.finditer(chunk):
                if token.start() <= skip:
                    continue
                char = token.group()
                if active is not None:
                    if char == active and not escaped(chunk, token.start(), 0, "" if flow and char == "'" else "\\"):
                        if flow and char == "'" and chunk.startswith("''", token.start()):
                            skip = token.start() + 1                  # YAML ``it''s``: a doubled quote is an apostrophe, not the end
                        else:
                            active = None
                elif char in "\"'":
                    if not flow or item_boundary(chunk, token.start()):      # YAML / INI: a quote inside a plain scalar is text
                        active = char
                        fstring = not flow and any(letter in "fF" for letter in chunk[max(0, token.start() - 2):token.start()])
                elif char in "[{":
                    depth += 1
                else:
                    depth = max(0, depth - 1)
            if active is not None:
                if active == '"':
                    stops = URL_STOP_DOUBLE_F if fstring else URL_STOP_DOUBLE
                else:
                    stops = URL_STOP_SINGLE_F if fstring else URL_STOP_SINGLE
            elif flow:
                stops = URL_STOP_FLOW if depth > 0 or lists else URL_STOP_PLAIN
            else:
                stops = URL_STOP
            stop = stops.search(text, resume)
            while stop is not None and active is not None and stop.group() == active:
                if escaped(text, stop.start(), resume, "" if flow and active == "'" else "\\"):
                    stop = stops.search(text, stop.end())            # ``say\"hi\"``: an escaped quote does not close the string
                elif flow and active == "'" and text.startswith("''", stop.start()):
                    stop = stops.search(text, stop.end() + 1)        # YAML ``it''s``: a doubled quote is an apostrophe
                else:
                    break
            end = size if stop is None else stop.start()
            chunks += [text[last:begin], URL_MARK * (end - begin)]
            last = counted = end
            at = text.find("://", end)
        else:
            at = text.find("://", at + 3)
    chunks.append(text[last:])
    return "".join(chunks)


def has_network_url(raw):
    """Does a decoded literal hold a URL with an authority (``https://host/docs``; not ``file:`` and not ``scheme:///path``)?
    ``find`` for ``://`` and a bounded look-back for the scheme, never a pattern that starts at every letter of a long run."""
    at = raw.find("://")
    while at >= 0:
        begin = at
        while begin > 0 and at - begin < 32 and raw[begin - 1] in SCHEME_CHARS:
            begin -= 1
        while begin < at and not raw[begin].isalpha():
            begin += 1
        scheme = raw[begin:at]
        if len(scheme) >= 2 and scheme.lower() != "file" and raw[at + 3:at + 4] not in ("", "/"):
            return True
        at = raw.find("://", at + 3)
    return False


def escaped(text, index, floor, mark):
    """True when ``text[index]`` follows an odd number of ``mark`` characters (those before ``floor`` are not counted)."""
    k = index
    while k > floor and text[k - 1] == mark:
        k -= 1
    return (index - k) % 2 == 1


def string_end(text, start, quote, mark="\\"):
    """The index just past the closing ``quote`` of the string opened at ``text[start]``. ``mark`` (a backslash; a backtick in
    PowerShell) protects the next character, so an escaped quote does not end the string and an escaped newline continues
    it. A string with no closing quote ends at the newline that ends its line (not consumed) or at the end of the text.
    Linear: each ``find`` resumes where the last stopped and is bounded by the candidate closer (a newline is looked for only
    BEFORE it: searching to the end of a 3 MB line for every string would be quadratic), and a run of ``mark`` characters is
    counted once."""
    pos, size = start + 1, len(text)
    close = text.find(quote, pos)
    while True:
        newline = text.find("\n", pos, size if close < 0 else close)
        if newline >= 0:
            if mark and escaped(text, newline, start + 1, mark):
                pos = newline + 1                                       # backslash-newline: the string goes on
                continue
            return newline
        if close < 0:
            return size
        if mark and escaped(text, close, start + 1, mark):
            pos = close + 1
            close = text.find(quote, pos)                               # searched again only once it has been passed
            continue
        return close + 1


ESCAPED_QUOTE = re.compile(r"""(?<!\\)((?:\\\\)*)\\(['"])""")


def rewrite_escaped_quotes(segment, mark="\\", fill=None):
    """One string segment with each escaped quote (``'owner\\'s notes'``) rewritten so no quote is left inside the string and
    the one-line literal pattern reads it whole: as ``\\x27`` / ``\\x22`` where ``decode_escapes`` restores the quote (Python,
    config), else as two ``fill`` characters (shells: the quote is only text). ``mark`` is the escape character (a backslash,
    or a backtick in PowerShell). Only an UNESCAPED run of marks is touched (``'a\\\\'`` ends at its quote)."""
    pattern = ESCAPED_QUOTE if mark == "\\" else ESCAPED_TICK
    if fill is None:
        return pattern.sub(lambda match: match.group(1) + ("\\x27" if match.group(2) == "'" else "\\x22"), segment)
    return pattern.sub(lambda match: match.group(1) + fill * 2, segment)


PY_SPECIAL = re.compile(r"""[#\n\\'"]""")
BRACKET = re.compile(r"[()\[\]{}]")
OPENERS, CLOSERS = "([{", ")]}"
MAX_JOINED_LINES = 40
ONE_LINE_TRIPLE = 300


def triple_end(text, pos, quote):
    """(index just past the closing triple quote, whether there was one) of a triple-quoted string whose body starts at ``pos``."""
    mark = quote * 3
    while True:
        close = text.find(mark, pos)
        if close < 0:
            return len(text), False
        if escaped(text, close, pos, "\\"):
            pos = close + 1
            continue
        return close + 3, True


def command_text(lines, first):
    """The command that begins on physical line ``first`` (1-based) of a notebook cell, as the shell reads it: stripped, with a trailing
    backslash joining the next line by a blank (at most MAX_JOINED_LINES lines). Python's comment rule does not apply to it."""
    parts, k = [], first - 1
    while 0 <= k < len(lines) and len(parts) < MAX_JOINED_LINES:
        text = lines[k].strip()
        if not text.endswith("\\"):
            parts.append(text)
            break
        parts.append(text[:-1].rstrip())
        k += 1
    return " ".join(parts)


def field_lines(line, expression, depth):
    """The logical lines of the expression of one replacement field of a multi-line f-string, all reported at ``line``. The field is
    code, so it is lexed like code: a triple-quoted string nested in it is prose, or an f-string with fields of its own. Nesting is
    followed to MAX_SUBSTITUTION_DEPTH and no further (a deeper field is dropped, not recursed into)."""
    if depth < MAX_SUBSTITUTION_DEPTH:
        for _, code, _ in python_logical_lines(expression, depth + 1):
            yield line, code, True


def python_logical_lines(text, depth=0):
    """(first physical line number, code, verbatim) per logical line of Python source, so that two statements never merge and
    one expression never splits: comments are dropped; the body of a multi-line triple-quoted string (a docstring: prose, not
    code) becomes an empty string while its newlines still count; physical lines are joined while a bracket is open or a
    backslash continues them (with nothing between the two halves of a string, a blank between tokens); an escaped quote is
    written ``\\x27`` / ``\\x22``, so no quote is left inside a string. ``verbatim`` is False when a string of the line was
    joined across lines, so no literal of it appears in the source as one token. Brackets are counted in the code BETWEEN
    strings with ``str.count``; at most MAX_JOINED_LINES physical lines are joined, so an unclosed bracket cannot swallow
    the file. The prose of a multi-line f-string goes with the rest of its body, but its replacement fields are code: each
    one follows the logical line it belongs to as a logical line of its own (``verbatim`` True)."""
    pos, size = 0, len(text)
    number = start = 1
    parts, plain, depth, joined, verbatim = [], [], 0, 1, True       # ``plain``: the code between strings, where brackets count
    fields = []                                                      # (line, expression) of the multi-line f-strings of this logical line
    while True:
        found = PY_SPECIAL.search(text, pos)
        stop = size if found is None else found.start()
        run = text[pos:stop]
        if run:
            parts.append(run)
            if BRACKET.search(run):                                     # only the runs that hold a bracket are counted later
                plain.append(run)
        if found is None:
            break
        mark, pos, ended = text[stop], stop + 1, False
        if mark == "#":
            end = text.find("\n", stop)
            pos = size if end < 0 else end
        elif mark == "\n":
            number += 1
            if plain:
                code = "".join(plain)
                depth = max(0, depth + sum(code.count(c) for c in OPENERS) - sum(code.count(c) for c in CLOSERS))
                plain = []
            if depth > 0 and joined < MAX_JOINED_LINES:
                parts.append(" ")
                joined += 1
            else:
                ended = True
        elif mark == "\\":
            if text.startswith("\n", pos):                              # a continuation BETWEEN tokens
                number += 1
                pos += 1
                if joined < MAX_JOINED_LINES:
                    parts.append(" ")
                    joined += 1
                else:
                    ended = True
            else:
                parts.append(text[stop:pos + 1])
                pos += 1
        elif text.startswith(mark * 3, stop):
            end, closed = triple_end(text, stop + 3, mark)
            body = text[stop + 3:end - 3] if closed else ""
            inside = text.count("\n", stop, end)
            if closed and not inside and len(body) <= ONE_LINE_TRIPLE and mark not in body and "\\" not in body:
                parts.append(mark + body + mark)                        # a one-line triple-quoted string is an ordinary string
            else:
                parts.append(mark + mark)
                if closed and F_PREFIX.search(text[max(0, stop - 3):stop]) is not None:
                    fields += [(number, expression) for expression in fstring_fields(body)]   # its prose goes, its fields are code
            number += inside
            pos = end
        else:
            end = string_end(text, stop, mark)
            segment = text[stop:end]
            if "\\" in segment:
                if "\\\n" in segment:                                   # a backslash-newline INSIDE a string: nothing is inserted
                    number += segment.count("\n")
                    segment = segment.replace("\\\n", "")
                    verbatim = False
                segment = rewrite_escaped_quotes(segment)
            parts.append(segment)
            pos = end
        if ended:
            code, first, exact, extra = "".join(parts), start, verbatim, fields
            parts, plain, depth, joined, start, verbatim, fields = [], [], 0, 1, number, True, []      # freed BEFORE the consumer runs
            if code.strip():
                yield first, code, exact
            for line, expression in extra:
                yield from field_lines(line, expression, depth)
    code = "".join(parts)
    parts = None
    if code.strip():
        yield start, code, verbatim
    for line, expression in fields:
        yield from field_lines(line, expression, depth)


class Profile(NamedTuple):
    """The lexical rules of one language: where its quotes and comments are."""
    special: object          # compiled pattern for the quote and comment characters, or None
    escaped: str             # quote characters whose strings ``mark`` protects (escapes)
    mark: str                # the escape character inside those strings
    plain: str               # quote characters whose strings have no escapes
    comments: str            # characters that start a comment
    after: object            # the characters a comment may follow (None: anywhere): ``a#b`` and ``$#`` are not comments
    outside: str             # the escape character OUTSIDE a string (``\"`` and backtick + ``"`` are no quotes), or ""
    filler: str              # what a blanked string becomes: a word character in shells (a glued word stays whole), else \x01
    decodable: bool          # an escaped quote is written \x27 / \x22 (decode_escapes restores it), else as two fillers
    arithmetic: bool         # ``((`` ... ``))`` is arithmetic, where ``<<`` is a shift and not a here-document (sh)
    backticks: bool          # a pair of backticks is a command substitution (sh)
    boundary: bool           # a quote opens a string only at a scalar boundary (YAML, INI): ``Don't`` and ``owner's`` have none


BLANKS = frozenset(" \t")
SHELL_BREAK = frozenset(" \t;|&(")                                 # a shell comment starts a word: after a blank or a separator


def make_profile(escaped_quotes, plain_quotes, comments, after, *, mark="\\", outside="", filler="\x01", decodable=True,
                 arithmetic=False, backticks=False, boundary=False):
    chars = escaped_quotes + plain_quotes + comments
    pattern = re.compile("[" + re.escape(chars) + "]") if chars else None
    return Profile(pattern, escaped_quotes, mark, plain_quotes, comments, after, outside, filler, decodable, arithmetic, backticks,
                   boundary)


_BATCH = make_profile("", '"', "", None, outside="^", filler="_", decodable=False)
_INI = make_profile("", "'\"", ";#", BLANKS, boundary=True)
_YAML = make_profile('"', "'", "#", BLANKS, boundary=True)
PROFILES = {
    ".py": make_profile("'\"", "", "#", None),
    ".sh": make_profile('"', "'", "#", SHELL_BREAK, outside="\\", filler="_", decodable=False, arithmetic=True, backticks=True),
    ".ps1": make_profile('"', "'", "#", SHELL_BREAK, mark="`", outside="`", filler="_", decodable=False),
    ".bat": _BATCH, ".cmd": _BATCH,
    ".yaml": _YAML, ".yml": _YAML,
    ".toml": make_profile('"', "'", "#", None),
    ".json": make_profile('"', "", "", None),
    ".ini": _INI, ".cfg": _INI,
}
MAX_SUBSTITUTION_DEPTH = 3


def lex_line(line, profile, blank=False):
    """``line`` without its comment (a comment marker inside a quoted string is not one, and an escaped quote outside a string
    is no quote), with every escaped quote INSIDE a string rewritten so none is left in it. With ``blank``, the text of quoted
    strings is replaced by ``profile.filler`` of the same length too (a word character in shells, so a quoted tail stays
    glued to the unquoted word before it); ``blank="plain"`` blanks only the strings a shell does not expand (single quotes).
    Linear: one jump per quote or comment character."""
    if profile.special is None or profile.special.search(line) is None:
        return line
    out, pos, size = [], 0, len(line)
    while True:
        found = profile.special.search(line, pos)
        if found is None:
            out.append(line[pos:])
            break
        i = found.start()
        char = line[i]
        if char in profile.comments:
            if profile.after is not None and i > 0 and line[i - 1] not in profile.after:
                out.append(line[pos:i + 1])                            # ``a#b`` and ``$#`` are not comments
                pos = i + 1
                continue
            out.append(line[pos:i])
            break
        if profile.outside and escaped(line, i, pos, profile.outside):
            out.append(line[pos:i + 1])                                # \" and `" outside a string are ordinary characters
            pos = i + 1
            continue
        if profile.boundary and i > 0 and not item_boundary(line, i):
            out.append(line[pos:i + 1])                                # ``Don't``, ``Use a " mark``: a quote inside a plain scalar
            pos = i + 1
            continue
        out.append(line[pos:i])
        if char in profile.escaped:
            close = line.find(char, i + 1)
            if close >= 0 and line.find(profile.mark, i + 1, close) < 0:
                end = close + 1                                        # nothing is escaped inside: the first quote closes it
            else:
                end = string_end(line, i, char, profile.mark)
        else:
            close = line.find(char, i + 1)
            end = size if close < 0 else close + 1
        segment = line[i:end]
        if blank is True or (blank == "plain" and char not in profile.escaped):
            out.append(profile.filler * len(segment))
        elif char in profile.escaped and profile.mark in segment:
            out.append(rewrite_escaped_quotes(segment, profile.mark, None if profile.decodable else profile.filler))
        else:
            out.append(segment)
        pos = end
    return "".join(out)


def lex_state(text, pos, upto, profile, arith=0):
    """(state, resume, arith): the lexical state at ``upto`` of text lexed from ``pos``, a point in code with ``arith`` shell
    arithmetic brackets open. ``state`` is 'code', or 'string' / 'comment' with ``resume`` the end of that string or comment.
    The same rules as ``lex_line``, over absolute offsets, so a caller walking a file lexes each line once."""
    search = profile.special.search if profile.special is not None else None
    size = len(text)
    while True:
        found = search(text, pos, upto) if search is not None else None
        stop = upto if found is None else found.start()
        if profile.arithmetic and stop > pos:
            run = text[pos:stop]
            arith = max(0, arith + run.count("((") - run.count("))"))
        if found is None:
            return "code", upto, arith
        char = text[stop]
        if char in profile.comments:
            if profile.after is not None and stop > 0 and text[stop - 1] != "\n" and text[stop - 1] not in profile.after:
                pos = stop + 1
                continue
            end = text.find("\n", stop)
            return "comment", (size if end < 0 else end), arith
        if profile.outside and escaped(text, stop, pos, profile.outside):
            pos = stop + 1
            continue
        if char in profile.escaped:
            end = string_end(text, stop, char, profile.mark)
        else:
            close = text.find(char, stop + 1)
            newline = text.find("\n", stop + 1, size if close < 0 else close)
            end = newline if newline >= 0 else (size if close < 0 else close + 1)
        if end > upto:
            return "string", end, arith
        pos = end


def heredoc_opener(match):
    """(strip leading tabs from the terminator, the delimiter word) of one here-document opener: the whole word, with its
    quotes and backslashes removed as the shell removes them (``E"OF"`` and ``\\EOF`` both delimit at ``EOF``; inside single
    quotes a backslash is a letter: ``'EOF\\END'`` delimits at ``EOF\\END``)."""
    def unquoted(part):
        if part.group(1) is not None:
            return part.group(1)
        if part.group(2) is not None:
            return HEREDOC_DQ_ESCAPE.sub(r"\1", part.group(2))
        return part.group(3)
    return match.group(1) == "-", HEREDOC_PART.sub(unquoted, match.group(2))


def heredoc_end(text, cursor, word, dash):
    """(start of the terminator line, start of the line after it) for a here-document whose body starts at ``cursor``: the
    terminator is EXACTLY the delimiter (``<<-`` also allows leading tabs); no terminator means the body runs to the end."""
    size = len(text)
    while cursor < size:
        eol = text.find("\n", cursor)
        stop = size if eol < 0 else eol
        line = text[cursor:stop]
        if (line.lstrip("\t") if dash else line) == word:
            return cursor, (size if eol < 0 else eol + 1)
        if eol < 0:
            break
        cursor = eol + 1
    return size, size


def blank_blocks(text, suffix):
    """``text`` with the multi-line text a shell reads as DATA replaced by blanks (newlines kept, so offsets and line numbers
    hold): PowerShell ``<# block comments #>`` and here-strings (``@" ... "@``), and shell here-documents (``<<EOF ... EOF``:
    every one on a command line, bodies in order; the opener's own line stays). An opener counts only in code: not inside a
    string or comment (``Write-Host "#" <# c #>`` is live, ``echo "<<EOF"`` is not) and, in sh, not inside arithmetic
    (``$((1 << SHIFT))``; an unquoted ``let x=1<<2`` IS a here-document, as bash parses the redirection first). Linear: each
    line is lexed once, forward, and the search resumes after a block."""
    if suffix == ".ps1":
        opener = PS_BLOCK
    elif suffix == ".sh":
        opener = HEREDOC
    else:
        return text
    found = opener.search(text)
    if found is None:
        return text
    profile = PROFILES[suffix]
    chunks, last, size = [], 0, len(text)
    lexed, arith = 0, 0                              # text[:lexed] is lexed, in code; ``arith``: arithmetic brackets open there
    while found is not None:
        begin = found.start()
        if begin < lexed or (suffix == ".sh" and escaped(text, begin, lexed, "\\")):
            found = opener.search(text, found.end())     # inside a string or comment already passed, or ``\<<EOF``: text
            continue
        state, resume, arith = lex_state(text, lexed, begin, profile, arith)       # across newlines: ``$((`` may span lines
        if state != "code" or arith:
            lexed = begin if state == "code" else resume
            found = opener.search(text, max(found.end(), lexed))
            continue
        lexed = begin
        if opener is PS_BLOCK:
            if found.group() == "<#":
                close = text.find("#>", found.end())
                end = size if close < 0 else close + 2
            else:
                close = text.find("\n" + found.group(1) + "@", found.end() - 1)
                end = size if close < 0 else close + 3
            chunks += [text[last:begin], NOT_NEWLINE.sub(" ", text[begin:end])]
            last, lexed, arith = end, end, 0
            found = opener.search(text, end)
            continue
        line_end = text.find("\n", found.end())
        if line_end < 0:
            break
        openers, cursor = [heredoc_opener(found)], found.end()
        while True:                                  # a command line may open several here-documents
            more = opener.search(text, cursor, line_end)
            if more is None:
                break
            if suffix == ".sh" and escaped(text, more.start(), cursor, "\\"):
                cursor = more.end()                  # ``cat <<A \<<B``: the second operator is text
                continue
            state, resume, arith = lex_state(text, cursor, more.start(), profile, arith)
            if state == "code" and not arith:
                openers.append(heredoc_opener(more))
            cursor = more.end() if state == "code" else max(more.end(), resume)
        position = line_end + 1
        for dash, word in openers:
            stop, after = heredoc_end(text, position, word, dash)
            chunks += [text[last:position], NOT_NEWLINE.sub(" ", text[position:stop])]
            last, position = stop, after
        lexed, arith = position, 0
        found = opener.search(text, position)
    chunks.append(text[last:])
    return "".join(chunks)


def batch_cut(line):
    """``line`` without a batch comment: ``rem ...`` or ``:: ...`` where a command starts (the line, or after ``&``, ``&&``,
    ``|``, ``(``, ``@``); text in quotes is ignored."""
    if "::" not in line and "rem" not in line.lower():
        return line
    view = lex_line(line, PROFILES[".bat"], blank=True)
    match = BATCH_REM.search(view)
    if match is None:
        return line
    return line[:match.start() + (1 if view[match.start()] in "&|(@" else 0)]


def substitution_bodies(code, profile, depth=0):
    """The bodies of the command substitutions on one shell line, nested ones too: ``$(...)`` in every shell and backtick pairs
    in sh. Each is matched by the shell's own quoting: a parenthesis inside single quotes (``"$(printf '('; find docs)"``) or
    double quotes is text, an escaped opener (``\\$(`` in sh, backtick-dollar in PowerShell) is text, and nothing inside single
    quotes expands. A stack of quote and bracket contexts, one pass per level of nesting; levels are capped."""
    if depth >= MAX_SUBSTITUTION_DEPTH or ("$(" not in code and "`" not in code):
        return
    escape = profile.outside
    token_pattern = SUB_TOKEN_PS if escape == "`" else SUB_TOKEN_SH
    stack, subs, i = [], 0, 0              # contexts: ("$", body start), ("(", 0) a bare parenthesis, ('"', 0) and ("'", 0) quotes
    while True:
        token = token_pattern.search(code, i)
        if token is None:
            break
        char, at = token.group(), token.start()
        i = token.end()
        top = stack[-1][0] if stack else ""
        if top == "'":
            if char == "'":
                stack.pop()
            continue
        if char == escape:                                          # the next character is not syntax
            i = at + 2
            continue
        if top == '"':
            if char == '"':
                stack.pop()
            elif char == "$(":
                stack.append(("$", i))
                subs += 1
            continue                                                # parentheses inside double quotes are text
        if char in "'\"":
            stack.append((char, 0))
        elif char == "$(":
            stack.append(("$", i))
            subs += 1
        elif char == "(":
            stack.append(("(", 0))
        elif stack:                                                 # ")"
            kind, start = stack.pop()
            if kind == "$":
                subs -= 1
                if subs == 0:                                       # the outermost body; its own substitutions come below
                    body = code[start:at]
                    yield body
                    yield from substitution_bodies(body, profile, depth + 1)
    if profile.backticks:
        masked = lex_line(code, profile, blank="plain")             # a backtick inside single quotes is text
        ticks = (match.start() for match in re.finditer("`", masked) if not escaped(code, match.start(), 0, escape))
        for opening in ticks:
            closing = next(ticks, None)
            if closing is None:
                break
            body = code[opening + 1:closing]
            yield body
            yield from substitution_bodies(body, profile, depth + 1)


def shell_units(text, suffix):
    """(line number, code) for every physical line of a shell, PowerShell or batch script AND for the body of every command
    substitution on it: block comments, here-strings and here-documents blanked, line comments removed, quoted strings left in
    place. The prose of a string is not code; the ``$(...)`` inside it is."""
    profile, batch = PROFILES[suffix], suffix in (".bat", ".cmd")
    for number, raw in enumerate(blank_blocks(text, suffix).split("\n"), start=1):
        code = lex_line(batch_cut(raw) if batch else raw, profile)
        yield number, code
        if batch:
            for command in batch_commands(code):
                yield number, command
        else:
            for body in substitution_bodies(code, profile):
                yield number, body
        for value in assigned_defaults(code, suffix, profile):
            yield number, '"' + value + '"'                       # a value the shell itself assigns is a literal path word


def assigned_defaults(code, suffix, profile):
    """The values a line assigns inside a quoted word: the default of a parameter expansion in sh (``"${DOCS:-docs}"``) and the
    value of batch ``set "DOCS=docs"``. Their strings are prose to the argument scan, but these words are paths."""
    if suffix == ".sh" and "${" in code:
        masked = lex_line(code, profile, blank="plain")                     # nothing inside single quotes is expanded
        for match in PARAMETER_DEFAULT.finditer(masked):
            word = code[match.start(1):match.end(1)].strip("'\"")           # ${D:-"docs"} and ${D:-'docs'}: the word without quotes
            if word and '"' not in word and "'" not in word:
                yield word
    elif suffix in (".bat", ".cmd") and "=" in code:
        for match in BATCH_SET.finditer(code):
            yield match.group(1)


def is_case_pattern(body, at):
    """Is the word at ``at`` (followed by ``)``) a ``case`` pattern: the first thing on its line, or right after ``|``, ``;;``,
    ``;&`` or the word ``in``? Only what directly precedes it is looked at, so a line of many matches costs a bounded step each."""
    end = at
    while end > 0 and body[end - 1] in " \t":
        end -= 1
    if end == 0 or body.endswith(CASE_SEPARATORS, max(0, end - 3), end):
        return True
    return body.endswith("in", max(0, end - 2), end) and (end == 2 or body[end - 3] in " \t;")


def batch_commands(code):
    """The command lines held in ``for /f ... in ('dir docs /b') do``: batch single quotes are not quotes, but cmd runs the text."""
    if "(" in code:
        for match in BATCH_FOR_COMMAND.finditer(code):
            yield match.group(1) if match.group(1) is not None else match.group(2)


def parse_field(body, start, depth=0):
    """(expression, expressions nested in its format spec, index after the closing brace) for the replacement field whose text
    starts at ``start`` (just after its ``{``), or None when it is never closed. The EXPRESSION part follows Python's own
    syntax: quotes open strings, brackets nest, and the first ``:`` or ``!`` (not ``!=``) at the top level ends it. Everything
    after that is FORMAT SPEC, where a quote is only a fill character (``{x:'>10}``) and ``{...}`` is a field of its own."""
    level, i = 0, start
    while True:
        token = FIELD_TOKEN.search(body, i)
        if token is None:
            return None
        char, i = token.group(), token.end()
        if char in "'\"":
            if body.startswith(char * 2, i):                  # '''...''' (or """...""") is ONE string; '' followed by a brace is not
                close = body.find(char * 3, i + 2)
                if close < 0:
                    return None
                i = close + 3
                continue
            close = body.find(char, i)
            if close < 0:
                return None
            i = close + 1
        elif char in "([{":
            level += 1
        elif char in ")]":
            level -= 1
        elif char == "}":
            if level <= 0:
                return trim_expression(body[start:token.start()]), [], i
            level -= 1
        elif level <= 0 and (char == ":" or (char == "!" and body[i:i + 1] != "=")):
            expression = body[start:token.start()]
            break
    nested = []
    while True:                                           # the format spec: nested fields, up to the closing brace
        token = SPEC_TOKEN.search(body, i)
        if token is None:
            return None
        if token.group() == "}":
            return trim_expression(expression), nested, token.end()
        if depth >= MAX_SUBSTITUTION_DEPTH:
            return None
        inner = parse_field(body, token.end(), depth + 1)
        if inner is None:
            return None
        if inner[0]:
            nested.append(inner[0])
        nested += inner[1]
        i = inner[2]


def trim_expression(expression):
    """The expression of a field without blanks and without the ``=`` of ``{x=}`` (an operator ``==``/``!=``/``<=`` stays)."""
    return re.sub(r"(?<![=!<>])=\s*$", "", expression.strip()).strip()


def fstring_fields(body):
    """The expression of every replacement field of an f-string body (``{{`` is a brace; ``{x!r:>{w}}`` has the expressions ``x``
    and ``w``). An unclosed field ends the scan: everything after it belongs to that field, so the cost stays linear."""
    fields, pos = [], 0
    while True:
        open_at = body.find("{", pos)
        if open_at < 0:
            return fields
        if body.startswith("{{", open_at):
            pos = open_at + 2
            continue
        parsed = parse_field(body, open_at + 1)
        if parsed is None:
            return fields
        expression, nested, pos = parsed
        if expression:
            fields.append(expression)
        fields += nested


def fstring_expressions(code, depth=0):
    """The expressions inside the replacement fields of every f-string literal on one logical line, nested f-strings included:
    ``f"{len(list(Path('docs').glob('*.md')))} files"`` executes ``Path('docs')``, which the opaque string hides."""
    if depth >= MAX_SUBSTITUTION_DEPTH or F_STRING.search(code) is None:
        return
    for match in LITERAL.finditer(code):
        if "f" in match.group("prefix").lower():
            body = match.group("double") if match.group("double") is not None else match.group("single")
            for expression in fstring_fields(body):
                yield expression
                yield from fstring_expressions(expression, depth + 1)


def import_stems(code):
    """The lower-cased identifier of every name in every ``import`` statement of one logical line (comments are gone,
    strings blanked): ``import os, a.b as c`` and ``from . import d, e`` all name modules, a dotted name names each of its
    parts, and the statement may follow ``if x:``, ``try:`` or ``;``."""
    if "import" not in code:
        return
    view = LITERAL.sub(lambda match: " " * len(match.group(0)), code)          # "import x" inside a string is no statement
    for match in IMPORT_STATEMENT.finditer(view):
        for identifier in IDENTIFIER.findall(" ".join(group for group in match.groups() if group)):
            if identifier not in ("as", "import", "from"):
                yield identifier.lower()


def value_start(line):
    """Where the value of a ``key: value``, ``key = value``, ``- value`` or ``- key: value`` config line begins (the line's
    comment already removed), or None when it has none."""
    stripped = line.lstrip()
    if stripped.startswith("- "):
        value_at = len(line) - len(stripped) + 2
        dash = SEQUENCE_DASH.match(line, value_at)
        while dash is not None:                                               # ``- - docs``: a sequence inside a sequence
            value_at = dash.end()
            dash = SEQUENCE_DASH.match(line, value_at)
        anchors = LEADING_ANCHORS.match(line, value_at)                       # ``- &a "root": reference``: the anchor is the key's
        key = SEQUENCE_KEY.match(line, value_at if anchors is None else anchors.end())   # ``- root: reference``
        return value_at if key is None else key.end()
    if stripped[:1] in ("'", '"'):                    # ``"sphinx:root": docs``: the separator follows the KEY, not a colon inside it
        quote, begin = stripped[0], len(line) - len(stripped)
        close = string_end(line, begin, quote, "\\" if quote == '"' else "")
        rest = line[close:].lstrip()
        if close <= len(line) and line[close - 1:close] == quote and rest[:1] in (":", "="):
            return len(line) - len(rest) + 1
        return None
    mark = YAML_SEPARATOR.search(line)
    return None if mark is None else mark.end()


def block_base(line):
    """The column a block scalar's body must be indented PAST. On ``- run: |`` that is the column of the KEY, not of the dash:
    the sibling key on the next line (``  working-directory: docs``) sits at the key's column, so it is no part of the body.
    On ``- |`` (no key) it is the column of the last dash, and on any other line the line's own indent."""
    at = len(line) - len(line.lstrip())
    if not line.lstrip().startswith("- "):
        return at
    column = at
    dash = SEQUENCE_DASH.match(line, at)
    while dash is not None:                                   # ``- - run: |``: the innermost dash and what follows it count
        text = dash.group()
        column = dash.start() + len(text) - len(text.lstrip(" \t"))
        at = dash.end()
        dash = SEQUENCE_DASH.match(line, at)
    anchors = LEADING_ANCHORS.match(line, at)
    key = SEQUENCE_KEY.match(line, at if anchors is None else anchors.end())
    return at if key is not None else column


def fold_triple_quotes(line):
    """A one-line TOML ``'''...'''`` or ``\"\"\"...\"\"\"`` string written as an ordinary one (``'...'``), or as an empty one when its body
    holds its own quote character: ``'''It's a 'docs' folder'''`` is ONE prose string, and read as pairs of quotes it would
    manufacture the literal ``'docs'`` (and with it, evidence for the whole tree). The line is read left to right, so a triple
    quote INSIDE an ordinary string (``["'''", 'docs', "'''"]``) is no delimiter and a comment ends the reading; a triple quote that
    is not closed on the line (a multi-line string) is left as it is. Linear: ``find`` jumps from quote to quote."""
    if "'''" not in line and '"""' not in line:
        return line
    out, pos = [], 0
    while True:
        found = TOML_QUOTE.search(line, pos)
        if found is None:
            break
        at, quote = found.start(), found.group()
        if quote == "#":
            break
        if line.startswith(quote * 3, at):
            close = line.find(quote * 3, at + 3)
            if close < 0:
                break
            body = line[at + 3:close]
            out += [line[pos:at], quote + body + quote if quote not in body else quote + quote]
            pos = close + 3
        else:
            end = string_end(line, at, quote, "\\" if quote == '"' else "")        # an ordinary string: skip it whole
            out.append(line[pos:end])
            pos = end
    out.append(line[pos:])
    return "".join(out)


def ini_value_start(line):
    """Where the value of an INI ``key = value`` or ``key:value`` line begins: after the FIRST ``=`` or ``:``. configparser needs no
    blank after a colon (``root:docs``); that is YAML's rule, and the one ``value_start`` applies."""
    mark = VALUE_MARK.search(line)
    return None if mark is None else mark.end()


UNSET = object()                                   # "not computed yet": a caller that already ran value_start passes the result


def unquoted_value(line, value_at=UNSET):
    """The unquoted value of a config line (see ``value_start``) as a Piece; None when there is no value, or the value is
    quoted or a flow collection (read elsewhere)."""
    if value_at is UNSET:
        value_at = value_start(line)
    if value_at is None:
        return None
    value = ANCHOR_OR_TAG.sub("", line[value_at:].strip())            # ``root: &anchor reference``, ``root: !!str reference``
    if not value or value[0] in "\"'[{":
        return None
    start = line.index(value, value_at)
    return Piece(value, start, start + len(value))


def flow_value(line, start, end):
    """The unquoted scalar of one item of a flow collection as a Piece (``contract: reference/flight specs`` gives
    ``reference/flight specs``; ``"contract":docs`` too: a quoted key needs no blank after its colon); None for a quoted item
    (read as a literal) or an empty one. ``line`` has its quoted strings blanked to \\x01."""
    text = line[start:end]
    compact = FLOW_QUOTED_KEY.match(text)
    if compact is not None:
        begin = compact.end()
    else:
        colon = text.find(": ")
        begin = colon + 2 if colon >= 0 else 0
    value = ANCHOR_OR_TAG.sub("", text[begin:].strip())
    if not value or value[0] in "\"'\x01":
        return None
    at = line.index(value, start + begin)
    return Piece(value, at, at + len(value))


def top_level_opener(line):
    """Index of the bracket when the whole YAML line IS a collection, after a document marker and properties
    (``--- [docs]``, ``&roots !!seq [docs]``, ``{root: docs}``), else None."""
    stripped = line.lstrip()
    top = len(line) - len(stripped)
    if stripped.startswith("---") and stripped[3:4] in ("", " ", "\t"):
        top += 3
    rest = line[top:]
    top += len(rest) - len(rest.lstrip())
    properties = LEADING_ANCHORS.match(line, top)
    if properties is not None:
        top = properties.end()
    return top if line[top:top + 1] in ("[", "{") else None


def item_boundary(code, at):
    """Does a quote at ``at`` BEGIN a scalar (after ``[``, ``{``, ``,``, ``:``, a sequence dash, or YAML properties that do),
    or does it sit inside a plain scalar (``[Keep "docs" tidy]``, ``Use a " character``, ``Don't``)? The text before it is
    walked back over blanks and properties in steps of at most 256 characters, so a line of many items costs a bounded step
    each."""
    end = at
    while True:
        while end > 0 and code[end - 1] in " \t":
            end -= 1
        if end == 0 or code[end - 1] in "[{,:":
            return True
        low = max(0, end - 256)
        begin = max(code.rfind(" ", low, end), code.rfind("\t", low, end)) + 1
        if begin == 0 and low > 0:
            return False                                   # a word longer than the look-back: not an item start
        token = code[begin:end]
        flows = len(token) - len(token.lstrip("[{,"))
        if flows:                                          # ``[&a``: the item starts inside this token, right after the bracket
            return token[flows:][:1] in ("&", "!")
        if token[:1] not in ("&", "!"):
            return False
        end = begin                                        # a property (``&a``, ``!!str``): the item starts before it


def scalar_limit(code, at=UNSET):
    """Where quoted text on a YAML or INI line stops being a literal, or None for no limit. The value of ``description: Keep
    "docs" tidy`` is ONE plain scalar: its quotes are prose, so everything from its start is text. After a quoted scalar
    (``root: "docs"``) nothing may follow, so the end of that string is the limit. A key before the value (``"docs": x``) is a
    literal. (A line that holds a flow collection is filtered per item: ``item_boundary``.)"""
    if at is UNSET:
        at = value_start(code)
    if at is None:
        at = len(code) - len(code.lstrip())
        if code[at:at + 3] == "---" and code[at + 3:at + 4] in ("", " ", "\t"):       # ``--- "docs"``: a document marker, then a scalar
            at += 3
    rest = code[at:]
    at += len(rest) - len(rest.lstrip())
    properties = LEADING_ANCHORS.match(code, at)
    if properties is not None:
        at = properties.end()
    first = code[at:at + 1]
    if first in ("", "[", "{"):
        return None
    if first in ("'", '"'):
        return string_end(code, at, first, "\\" if first == '"' else "")
    return at


def flow_start(line, depth, at=UNSET):
    """Where flow-collection parsing may begin on a (string-blanked) config line, or None. A collection continues from the
    previous line (``depth``), or is opened by a VALUE that begins with a bracket: ``roots: [docs]``, ``- {root: docs}``. The
    brackets of ``description: See [docs] for details`` are the text of a plain scalar."""
    if depth > 0:
        return 0
    opener = top_level_opener(line)
    if opener is not None:
        return opener
    if at is UNSET:
        at = value_start(line)
    if at is None:
        return None
    rest = line[at:]
    properties = LEADING_ANCHORS.match(line, at + len(rest) - len(rest.lstrip()))     # ``roots: &a !!seq [docs]``
    if properties is not None:
        at = properties.end()
    return at if line[at:].lstrip()[:1] in ("[", "{") else None


def flow_items(line, depth, start=0):
    """(Pieces, nesting depth after the line): the unquoted items INSIDE the flow collections (``[...]``, ``{...}``) of a line
    whose quoted strings are blanked and comment removed, from ``start`` on. ``roots: [docs, x]`` has two items; ``note: docs,
    x`` has none. A collection that is still open at the end of the line carries its depth to the next one."""
    if depth == 0 and "[" not in line and "{" not in line:
        return [], 0
    pieces, begin = [], start
    for match in FLOW_DELIMITER.finditer(line, start):
        i = match.start()
        if depth > 0 and i > begin:
            piece = flow_value(line, begin, i)
            if piece is not None:
                pieces.append(piece)
        if match.group() in "[{":
            depth += 1
        elif match.group() in "]}":
            depth = max(0, depth - 1)
        begin = i + 1
    if depth > 0 and begin < len(line):
        piece = flow_value(line, begin, len(line))
        if piece is not None:
            pieces.append(piece)
    return pieces, depth


def list_items(line, offset=None):
    """The unquoted comma-separated items of an INI-style value (``roots = docs, specs``) as Pieces. ``offset`` is where the
    value begins when the whole line is one (an indented continuation line); by default it follows the first ``=`` or ``:``."""
    if offset is None:
        mark = VALUE_MARK.search(line)
        if mark is None:
            return []
        offset = mark.end()
    if "," not in line:
        return []
    pieces = []
    for item in line[offset:].split(","):
        value = item.strip()
        if value and value[0] not in "\"'\x01":
            start = line.index(value, offset)
            pieces.append(Piece(value, start, start + len(value)))
        offset += len(item) + 1
    return pieces


def title_of(text, suffix):
    if suffix not in PROSE_SUFFIXES:
        return ""
    start = 0
    for _ in range(60):                                         # the first non-blank of the first 60 lines, never a full split
        end = text.find("\n", start)
        raw = text[start:] if end < 0 else text[start:end]
        if raw.strip():
            match = HEADING.match(raw[:HEADING_WINDOW])
            return (match.group(1) if match else raw.strip())[:100]
        if end < 0:
            break
        start = end + 1
    return ""


class Lines:
    """Offset -> 1-based line number, with the table of line starts built only if a hit needs it."""

    def __init__(self, text):
        self.text, self.starts = text, None

    def at(self, offset):
        if self.starts is None:
            self.starts = [0] + [m.end() for m in re.finditer("\n", self.text)]
        return bisect.bisect_right(self.starts, offset)


class FixedLine:
    """Stands in for ``Lines`` where every offset reports one number: a notebook cell."""

    def __init__(self, number):
        self.number = number

    def at(self, offset):
        return self.number


def cell_text(value):
    """(text, well formed) for a notebook cell's 'source' or 'input': a string or a list of strings."""
    if isinstance(value, str):
        return value, True
    if isinstance(value, list):
        return "".join(item if isinstance(item, str) else str(item) for item in value), all(isinstance(i, str) for i in value)
    return "", False


BINARY_MIME = ("image/", "audio/", "video/", "application/pdf", "application/octet-stream")   # a base64 payload names nothing
BINARY_V3 = ("png", "jpeg", "jpg", "gif", "pdf")                       # nbformat 3 keeps image data in fields of the output
SKIPPED_OUTPUT_FIELDS = ("output_type", "metadata", "execution_count", "prompt_number", "name", "stream")
OUTPUT_DEPTH = 40


def payload_strings(node, found, depth=0):
    """Every string and every key of an arbitrary JSON payload, each as an independent string (a file name can be a key).
    True when it nests deeper than OUTPUT_DEPTH. No MIME rule applies below this point: ``image/path`` is only a key."""
    if depth > OUTPUT_DEPTH:
        return True
    malformed = False
    if isinstance(node, str):
        found.append(node)
    elif isinstance(node, list):
        for item in node:
            malformed |= payload_strings(item, found, depth + 1)
    elif isinstance(node, dict):
        for name, value in node.items():
            if isinstance(name, str):
                found.append(name)
            malformed |= payload_strings(value, found, depth + 1)
    return malformed


def multiline_strings(value, found):
    """nbformat stores a multiline string as a string or as a LIST OF FRAGMENTS to be concatenated with no separator.
    True when the field held neither (its leaves are scanned anyway)."""
    if isinstance(value, str):
        found.append(value)
        return False
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        found.append("".join(value))
        return False
    payload_strings(value, found)
    return True


def output_strings(outputs, found):
    """Append every textual leaf of a cell's ``outputs`` to ``found``, following nbformat's own structure: ``text`` and
    every ``text/*`` entry of a MIME bundle are multiline strings, a traceback is a list of independent lines, binary
    bundle entries are skipped, and any other bundle entry is an arbitrary JSON payload. True when a field that must
    hold text held something else."""
    if not isinstance(outputs, list):
        return True
    malformed = False
    for output in outputs:
        if not isinstance(output, dict):
            malformed = True
            continue
        for name, value in output.items():
            if name in SKIPPED_OUTPUT_FIELDS or name in BINARY_V3:
                continue
            if name == "data":
                if not isinstance(value, dict):
                    malformed = True
                    continue
                for mime, entry in value.items():
                    if not isinstance(mime, str) or mime.startswith(BINARY_MIME):
                        continue
                    if mime.startswith("text/"):
                        malformed |= multiline_strings(entry, found)
                    else:
                        malformed |= payload_strings(entry, found)
            elif name in ("text", "source", "input"):
                malformed |= multiline_strings(value, found)
            elif name == "traceback":
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    found.append("\n".join(value))
                else:
                    malformed |= multiline_strings(value, found)
            else:                                                       # ename, evalue, an nbformat 3 html/latex/json field ...
                malformed |= payload_strings(value, found)
    return malformed


def notebook_cells(data):
    """(cells, malformed containers) for nbformat 4 (``cells``) and nbformat 3 (``worksheets[*].cells``), else None."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("cells"), list):
        return data["cells"], 0
    sheets = data.get("worksheets")
    if isinstance(sheets, list):
        cells, bad = [], 0
        for sheet in sheets:
            if isinstance(sheet, dict) and isinstance(sheet.get("cells"), list):
                cells += sheet["cells"]
            else:
                bad += 1
        return cells, bad
    return None


def normalise(components):
    """Drop empty and '.' components and resolve '..' against the component before it. A '..' with nothing before it is
    dropped: the path is read as relative to the repository root."""
    out = []
    for component in components:
        if component in ("", "."):
            continue
        if component == "..":
            if out:
                out.pop()
            continue
        out.append(component)
    return out


def trim_closers(run):
    """``run`` without the closing brackets that have no opener inside it: ``docs/A.md}`` ends a flow mapping
    (``{name: docs/A.md}``) and ``docs/A.md]}`` a sequence inside one, and neither closer belongs to the file name. A balanced
    pair stays (``docs/{a,b}.md`` is a template, ``x[0-9].md`` a glob). One count per run, then the trailing closers are
    peeled off one by one: linear."""
    surplus = {"}": run.count("}") - run.count("{"), "]": run.count("]") - run.count("[")}
    end = len(run)
    while end and surplus.get(run[end - 1], 0) > 0:
        surplus[run[end - 1]] -= 1
        end -= 1
    return run[:end]


def split_literal(raw):
    """(literal directory components, literal head of the first non-literal component, has a placeholder). '..' is kept:
    it only means something once ALL the pieces of a chain are known (``'reference/anchors' / '..' / name``)."""
    folded = raw.replace("\\", "/")
    base = BASE_PREFIX.match(folded)                    # ``$PSScriptRoot/..``, ``%~dp0..``, ``\\?\``, ``file:///``: where the path starts
    if base is not None:                                # is unknown, and a placeholder there must not hide the literal path after it
        folded = folded[base.end():]
    found = PLACEHOLDER.search(folded)
    if found is None:
        return [c for c in folded.split("/") if c not in ("", ".")], "", False
    head = folded[:found.start()]
    cut = head.rfind("/")
    return [c for c in head[:cut + 1].split("/") if c not in ("", ".")], head[cut + 1:], True


class Piece(NamedTuple):
    """One path-like string of a line: a quoted literal, several glued ones, or an unquoted config value.
    ``exact`` is True when ``raw`` appears in the source as ONE contiguous token (the path scan then read it already);
    a merged, glued or escape-decoded piece is not."""
    raw: str
    start: int
    end: int
    exact: bool = True


PATH_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-{}*?%$~/\\[]")
GLUE_LIMIT = 200


def glue_before(line, start):
    """(unquoted path text glued to the left of a quote, where it begins): ``reference/`` in ``reference/"run$id.json"``."""
    begin = start
    while begin > 0 and start - begin < GLUE_LIMIT and line[begin - 1] in PATH_CHARS:
        begin -= 1
    token = line[begin:start]
    return (token, begin) if token and token[-1] in "/\\" else ("", start)


def glue_after(line, end):
    """(unquoted path text glued to the right of a quote, where it ends): ``run1.json`` in ``"reference/"run1.json``."""
    finish = end
    while finish < len(line) and finish - end < GLUE_LIMIT and line[finish] in PATH_CHARS:
        finish += 1
    return line[end:finish], finish


def decode_escapes(raw):
    """The backslash escapes of a double-quoted string (``R\\u00e9sum\\u00e9``, ``my\\x20reports``) as the language that
    wrote it reads them: JSON first, then the superset Python, YAML and TOML share (``\\x``, ``\\U``); as written when invalid."""
    try:
        return json.loads('"' + raw + '"')
    except ValueError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                  # invalid escapes such as \\A are kept, with a DeprecationWarning
            return raw.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except (UnicodeError, ValueError):
        return raw


def literal_pieces(line, suffix, verbatim=True):
    """The quoted literals of one line as Pieces. Python: literals separated by blanks only are ONE string (implicit
    concatenation, ``'reference/run_' f'{n}.json'``). Shell and PowerShell: a quoted part glued to unquoted path text is
    ONE argument (``reference/"run$id.json"``), separate words are separate arguments. A double-quoted config string and
    a non-raw Python string have their backslash escapes decoded; ``''`` inside a single-quoted YAML or PowerShell string
    is one apostrophe."""
    parts, first, last, exact_run, quote_run = None, 0, 0, True, ""   # merge buffer: joined ONCE, never ``a + b`` in a loop
    for match in LITERAL.finditer(line):
        double = match.group("double")
        raw = double if double is not None else match.group("single")
        quote = '"' if double is not None else "'"
        start, end = match.start(), match.end()
        exact = verbatim and "\\x2" not in raw          # an escaped quote was written \x27 / \x22: not the source's own spelling
        if "\\" in raw and ((suffix in CONFIG_SUFFIXES and quote == '"')
                            or (suffix == ".py" and "r" not in match.group("prefix").lower())):
            decoded = decode_escapes(raw)
            if decoded != raw:
                exact, raw = False, decoded
        if suffix in SHELL_SUFFIXES:
            before, begin = glue_before(line, start)
            if before and (parts is None or begin >= last):
                raw, start, exact = before + raw, begin, False
            after, finish = glue_after(line, end)
            if after:
                raw, end, exact = raw + after, finish, False
        if parts is not None:
            between = line[last:start]
            if between == "" and quote == "'" and quote_run == "'" and suffix in (".yaml", ".yml", ".ps1"):
                parts.append("'" + raw)
                last, exact_run, quote_run = end, False, quote
                continue
            if (suffix == ".py" and (between == "" or between.isspace())) or (suffix in SHELL_SUFFIXES and between == ""):
                parts.append(raw)
                last, exact_run, quote_run = end, False, quote
                continue
            yield Piece("".join(parts), first, last, exact_run)
        parts, first, last, exact_run, quote_run = [raw], start, end, exact, quote
    if parts is not None:
        yield Piece("".join(parts), first, last, exact_run)


def joined_pieces(pieces, line, suffix):
    """Literals ADDED to each other are one string (Python, PowerShell): ``'docs/' + 'A.md'`` is ``docs/A.md``, so it is a path
    and not a head with a variable tail. A literal added to anything else stays a head (see PLUS_TAIL)."""
    if suffix not in (".py", ".ps1"):
        yield from pieces
        return
    pending = next(pieces, None)
    while pending is not None:
        following, parts, ends = next(pieces, None), None, pending.end
        while following is not None and PLUS_BETWEEN.fullmatch(line, ends, following.start):
            if parts is None:
                parts = [pending.raw]                     # a list joined once: ``a + b`` in a loop is quadratic
            parts.append(following.raw)
            ends = following.end
            following = next(pieces, None)
        yield pending if parts is None else Piece("".join(parts), pending.start, ends, False)
        pending = following


def tag_for(category, unresolved, form):
    if category == "policy":
        return T_POLICY
    if category == "notes":
        return T_NOTES
    if category == "loop":
        return T_LOOP
    if category == "outputs":
        return T_OUTPUTS
    if form == "import":
        return T_IMPORT
    if form == "name":
        return T_NAME
    if unresolved:
        return T_UNRESOLVED
    return T_PATH if form in ("path", "chain") else T_SUBTREE


TIERS = (("code-path", {T_PATH, T_IMPORT}), ("code-tree", {T_SUBTREE}), ("code-unresolved", {T_UNRESOLVED}),
         ("code-name", {T_NAME}), ("outputs", {T_OUTPUTS}), ("notes", {T_NOTES}), ("loop", {T_LOOP}))
TIER_NAMES = [name for name, _ in TIERS] + ["none"]
TIER_MEANING = {
    "code-path": "a tracked source names the file by path, or imports it by stem",
    "code-tree": "a tracked source names a directory (or a whole tree) that contains it",
    "code-unresolved": "a tracked source builds a path that could name it",
    "code-name": "a tracked source mentions its file name",
    "outputs": "only an output record (freeze markers, manifests) names it",
    "notes": "only other local notes name it",
    "loop": "only loop records name it",
    "none": "nothing scanned names it (see what was not scanned)",
}


def strongest(target):
    """The strongest KIND of evidence found for a file: a fact about the scan, not a verdict on whether it is used."""
    for name, wanted in TIERS:
        if target.tags & wanted:
            return name
    return "none"


def modified(mtime_ns):
    try:
        return datetime.datetime.fromtimestamp(mtime_ns / 1e9, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return "unknown"


# ------------------------------------------------------------------------------------------------ inventory

def inventory(repo, gaps):
    """Every regular file below the trees as {posix path: (size, mtime_ns)}. Links and junctions are listed, not followed."""
    found = {}
    for tree in TREES:
        base = repo / tree
        try:
            info = os.lstat(base)
        except OSError:
            gaps["missing_trees"].append(tree)
            continue
        if is_redirect(info) or not stat.S_ISDIR(info.st_mode):
            gaps["missing_trees"].append(tree)
            gaps["links_skipped"].append(tree)
            continue
        stack = [base]
        while stack:
            here = stack.pop()
            try:
                with os.scandir(here) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError:
                note_unreadable(gaps, here)
                continue
            for entry in entries:
                path = Path(entry.path)
                rel = path.relative_to(repo).as_posix()
                try:
                    info = os.lstat(path)
                except OSError:
                    note_unreadable(gaps, path)
                    continue
                if is_redirect(info):
                    gaps["links_skipped"].append(rel)
                elif stat.S_ISDIR(info.st_mode):
                    if entry.name != "__pycache__":
                        stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    found[rel] = (info.st_size, info.st_mtime_ns)
    return found


# ------------------------------------------------------------------------------------------------ the scanner

class Scanner:
    """Matching is case-insensitive, as the Windows filesystem resolves paths (measured on the real workspace:
    ``docs/architecture.md`` names ``docs/ARCHITECTURE.md``). Every lookup key is lower-cased; every record keeps the
    file's real spelling."""

    def __init__(self, model, files):
        self.model = model
        self.file_by_lower = defaultdict(list)
        self.dir_set = set(TREES)                  # lower-cased directories, the three roots included
        self.dir_actual = {tree: tree for tree in TREES}
        self.basenames = defaultdict(list)         # lower-cased file name -> real paths
        self.stems = defaultdict(list)             # lower-cased module name -> real .py paths
        below = defaultdict(list)                  # lower-cased directory -> (path relative to it, lower-cased; real path)
        reversed_files = []                        # (components read from the end, joined; real path), 3+ components
        self.max_depth = max((path.count("/") + 1 for path in files), default=1)    # no real path has more components
        self.reach = 2 * (self.max_depth + 1) + 6          # components a path candidate may span: room for '..' to cancel
        for path in files:
            parts = path.split("/")
            lowered = [p.lower() for p in parts]
            self.file_by_lower["/".join(lowered)].append(path)
            self.basenames[lowered[-1]].append(path)
            if lowered[-1].endswith(".py"):
                self.stems[lowered[-1][:-3]].append(path)
            if len(parts) >= 3:
                reversed_files.append(("/".join(reversed(lowered)), path))
            for k in range(1, len(parts)):
                directory = "/".join(lowered[:k])
                self.dir_set.add(directory)
                self.dir_actual.setdefault(directory, "/".join(parts[:k]))
                below[directory].append(("/".join(lowered[k:]), path))
        # sorted once, so "everything below d whose name starts with h" is a bisect range, not a scan of d's subtree
        self.below = {directory: sorted(entries) for directory, entries in below.items()}
        self.below_keys = {directory: [key for key, _ in entries] for directory, entries in self.below.items()}
        # and "everything whose path ENDS in a/b" is a bisect range over paths read from the end ("b/a/..."), not a bucket scan
        self.reversed_dirs = sorted(("/".join(reversed(d.split("/"))), d) for d in self.dir_set if d not in TREES)
        self.reversed_dir_keys = [key for key, _ in self.reversed_dirs]
        self.reversed_files = sorted(reversed_files)
        self.reversed_file_keys = [key for key, _ in self.reversed_files]
        self.word_names = {n for n in self.basenames if WORD.fullmatch(n)}
        # names with a space or another character the word pattern cannot hold: matched as a SEQUENCE of word tokens.
        # A token that is only dots ("a . b.md") is dropped here AND from the source sequence, so the two stay aligned.
        self.odd_names = {}
        for name in self.basenames:
            if name not in self.word_names:
                kept = [(m.start(), t) for m in WORD.finditer(name) if (t := m.group(0).rstrip(".").lower())]
                if kept:
                    self.odd_names[name] = ([t for _, t in kept], kept[0][0])
        self.flagged = set()
        self.touched = set()

    @staticmethod
    def ending_with(keys, entries, tail):
        """The entries whose path ends in the components of ``tail`` at a component boundary, as a bisect range over the
        paths read from the end: no scan of the entries that merely share a last component. (A tail never EQUALS a whole
        path: every path begins with a tree name, and a tail that does is resolved before this is asked.)"""
        reverse = "/".join(reversed(tail.split("/")))
        low = bisect.bisect_left(keys, reverse + "/")
        return [actual for _, actual in entries[low:bisect.bisect_left(keys, reverse + "/" + HIGH, low)]]

    # -- recording ---------------------------------------------------------------------------------
    def record(self, target_path, category, source, line, form, unresolved=False):
        if target_path == source:
            return
        target = self.model.targets[target_path]
        target.tags.add(tag_for(category, unresolved, form))
        mark = (target_path, category)                       # each source is scanned once, so "first touch in this scan"
        if mark not in self.touched:                         # is "one more distinct source": no per-target set of sources
            self.touched.add(mark)
            target.source_counts[category] += 1
        if category == "code" and not unresolved and form in ("path", "chain"):
            mark = (target_path, "named")
            if mark not in self.touched:
                self.touched.add(mark)
                target.named_count += 1
        if source in self.model.targets and not unresolved and form in ("path", "chain", "name"):
            self.model.outgoing[source].add(target_path)             # a document inside the trees linking another one
        key = (category, source, line, form)
        if key not in target.seen and len(target.evidence[category]) < EVIDENCE_CAP:
            target.seen.add(key)
            target.evidence[category].append(Evidence(category, source, line, form))

    def flag(self, prefix_parts, head, category, source, line, form, unresolved, root_ok, text=""):
        """Mark the files below a directory whose relative path starts with ``head``. A tree root with no head is a
        whole-tree statement, which only a quoted literal in a code or config source (``root_ok``) may make."""
        prefix = "/".join(prefix_parts).lower()
        if prefix not in self.dir_set:
            return 0
        if prefix in TREES and not head and not root_ok:
            if category == "policy":
                self.model.policy_roots[source].add(prefix)
            return 0
        start = head.lower()
        key = (prefix, start, form, unresolved, category)
        if key in self.flagged:
            return 0
        self.flagged.add(key)
        keys = self.below_keys.get(prefix, [])
        low = bisect.bisect_left(keys, start)                          # every key that starts with ``start`` lies in
        high = bisect.bisect_left(keys, start + HIGH, low) if start else len(keys)         # [start, start + U+10FFFF)
        matched = high - low
        for _, path in self.below.get(prefix, [])[low:high]:
            self.record(path, category, source, line, form, unresolved)
        if unresolved and category == "code":
            if len(self.model.constructions) < MAX_CONSTRUCTIONS:       # the LIST is capped; the marking above is not
                self.model.constructions.append(
                    Construction(source, line, self.dir_actual[prefix], head, matched, text.strip()[:160].replace("`", "'")))
            else:
                self.model.gaps["constructions_unlisted"] += 1
        return matched

    # -- one source text ---------------------------------------------------------------------------
    def scan(self, category, source, text):
        suffix = Path(source).suffix.lower()
        self.flagged = set()
        self.touched = set()
        root_ok = category == "code" and suffix in CODE_SUFFIXES
        if suffix == ".ipynb":
            if self.scan_notebook(category, source, text, root_ok):
                return
            self.model.gaps["notebooks_unparsed"] += 1              # not a notebook we can decode: read the raw text
        self.scan_body(category, source, text, suffix, root_ok, Lines(text), None)

    def scan_body(self, category, source, text, suffix, root_ok, lines, fixed_line):
        """One text read for paths, chains, configured values, imports and names. ``suffix`` says what kind of text it is;
        ``lines`` turns an offset into the number reported as evidence (a line, or a notebook cell)."""
        text = blank_urls(text, suffix in FLOW_SUFFIXES, suffix in (".ini", ".cfg"))   # a URL is no local path at any level
        folded = text.replace("\\", "/")                        # one character for one: offsets stay valid for line numbers
        self.scan_paths(category, source, folded, lines, root_ok)
        if suffix in PROSE_SUFFIXES:                            # [text](<a destination with spaces>) is one path
            for match in MD_ANGLE.finditer(text):
                destination = match.group(1)
                self.process_run(category, source, lines.at(match.start()), destination,
                                 [Piece(destination, 0, len(destination), False)], False, "")
        if suffix == ".py" or suffix == ".ipynb":               # a notebook read raw (unparsable) is scanned as Python
            for number, code, verbatim in python_logical_lines(text):
                number = fixed_line or number
                self.scan_chain_line(category, source, number, code, root_ok, ".py", verbatim)
                for expression in fstring_expressions(code):             # f"{Path('docs')}": the field is code, the rest is text
                    for _, inner, exact in python_logical_lines(expression):       # ... and code is lexed: ``'''It's 'docs''''`` is prose
                        self.scan_chain_line(category, source, number, inner, root_ok, ".py", exact)
                if suffix == ".py":
                    for stem in import_stems(code):
                        for path in self.stems.get(stem, ()):
                            self.record(path, category, source, number, "import")
        elif suffix in SHELL_SUFFIXES:
            profile = PROFILES[suffix]
            for number, code in shell_units(text, suffix):
                self.scan_chain_line(category, source, number, code, root_ok, suffix)
                if root_ok:
                    self.scan_command_line(category, source, number, code, profile)
        elif suffix in CONFIG_SUFFIXES:
            self.scan_config(category, source, text, suffix, root_ok, fixed_line)
        self.scan_names(category, source, folded, lines)

    def scan_notebook(self, category, source, text, root_ok):
        """Decode a notebook and read every cell as text of its own kind, reporting the 1-based CELL number: a code cell
        is Python (paths, chains, roots, imports), a prose cell or an output is text (paths and names only, and a tree
        named on its own there means nothing). False when the file is not a notebook we can decode."""
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):                         # not JSON, or nested past the parser's recursion limit
            return False
        found = notebook_cells(data)
        if found is None:
            return False
        cells, malformed_sheets = found
        gaps = self.model.gaps
        gaps["notebook_cells_malformed"] += malformed_sheets
        for number, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                gaps["notebook_cells_malformed"] += 1
                continue
            body, well_formed = cell_text(cell["source"] if "source" in cell else cell.get("input", ""))   # v4 source, v3 input
            code = cell.get("cell_type") == "code"
            self.scan_body(category, source, body, ".py" if code else ".md", root_ok and code, FixedLine(number), number)
            malformed = not well_formed
            if code:
                self.scan_magics(category, source, body, number, root_ok)
                shown = []
                malformed |= output_strings(cell.get("outputs", []), shown)
                if shown:
                    self.scan_body(category, source, "\n".join(shown), ".md", False, FixedLine(number), number)
            if malformed:
                gaps["notebook_cells_malformed"] += 1                # once per cell, whatever part of it was malformed
        return True

    def scan_magics(self, category, source, body, number, root_ok):
        """The commands of a notebook code cell that Python does not run: a ``!`` line (``!find docs``), an assigned one
        (``files = !ls docs``), a ``%`` line magic (``%cd docs``) and the body of a ``%%bash`` / ``%%sh`` / ``%%script`` cell
        (IPython skips blank lines before the header) are shell lines. A line INSIDE a Python string is text, so the lines are the
        logical lines of the cell (``python_logical_lines``); a URL is no path (``!wget https://host/docs``). The Python reading
        of the same text stays, and a ``!=`` or ``% (`` that continues an expression is not a command."""
        body = blank_urls(body)
        lines = body.split("\n")
        first = next((k for k, line in enumerate(lines) if line.strip()), len(lines))
        head = lines[first].strip().split(None, 1)[0] if first < len(lines) else ""
        script = []
        if head.startswith("%%") and head[2:] in SHELL_CELL_MAGICS:
            script = lines[first + 1:]
        else:
            for first, code, _ in python_logical_lines(body):         # the lexer says WHERE a command is; the shell reads its text
                stripped = code.strip()
                physical = command_text(lines, first) if 0 < first <= len(lines) else stripped      # ``build#1`` is no Python comment
                if MAGIC_LINE.match(stripped):
                    script.append((physical if MAGIC_LINE.match(physical) else stripped).lstrip("!%"))
                elif ASSIGNED_ESCAPE.match(stripped):
                    text = physical if ASSIGNED_ESCAPE.match(physical) else stripped
                    script.append(text[ASSIGNED_ESCAPE.match(text).end():])
        if script:
            profile = PROFILES[".sh"]
            for _, code in shell_units("\n".join(script), ".sh"):
                self.scan_chain_line(category, source, number, code, root_ok, ".sh")
                if root_ok:
                    self.scan_command_line(category, source, number, code, profile)

    def scan_paths(self, category, source, folded, lines, root_ok):
        for match in RUN.finditer(folded):
            run = match.group(0)
            if "/" not in run:
                continue
            run = self.without_closers(run)
            parts = [p for p in run.rstrip(".").split("/") if p]
            cancels = parts.count("..")
            for i, part in enumerate(parts):
                low = part.lower()
                root = low if low in TREES else part.lstrip("*_[").lower()   # **docs/A.md**, __docs/A.md__, [docs/A.md], Docs/a.md
                if root in TREES:
                    # A real path has at most max_depth components. Count the components that REMAIN after '..' ('.' and '..'
                    # cancel: ``docs/x/y/z/w/../../../../A.md`` is ``docs/A.md``): a candidate whose remaining depth exceeds
                    # max_depth even if every later '..' removed one can only match nothing, so it is cut there; and no
                    # candidate looks further than ``reach`` components, or ``docs/../docs/../...`` costs O(n) per start and
                    # O(n^2) in all. (An index loop: a slice per start would be O(n) again.)
                    rest, net, spent = [], 0, 0                         # '.' costs nothing: it cannot start a candidate
                    for j in range(i + 1, len(parts)):
                        rest.append(parts[j])
                        if parts[j] == ".":
                            continue
                        if parts[j] == "..":
                            net = max(0, net - 1)
                        else:
                            net += 1
                            if net - cancels > self.max_depth + 1:
                                break
                        spent += 1
                        if spent >= self.reach:                         # cut by the reach, not by the depth: too long to resolve
                            if j + 1 < len(parts):
                                self.model.gaps["paths_too_long"] += 1
                                rest = None
                            break
                    if rest is None:
                        continue
                    self.path_candidate([root] + rest, category, source, lines.at(match.start()), root_ok)

    def without_closers(self, run):
        """``run`` minus the closing brackets of the flow collection it stands in (``trim_closers``), unless the run, or the run
        minus fewer closers (at most three), names a file or directory of the inventory exactly: ``docs/reports]`` is a real
        directory and ``{k: docs/A.md]}`` ends in the name ``A.md]`` and the mapping's ``}``."""
        trimmed = trim_closers(run)
        for dropped in range(min(len(run) - len(trimmed), 3)):
            candidate = run[:len(run) - dropped]
            parts = [p for p in candidate.rstrip(".").split("/") if p]
            literal = self.rooted(normalise(parts))
            full = "/".join(literal).lower()
            if literal and (full in self.file_by_lower or full in self.dir_set):
                return candidate
        return trimmed

    @staticmethod
    def rooted(components):
        """The components from the first tree name on (a path may start with ``..`` or a directory outside the trees)."""
        for k, component in enumerate(components):
            if component.lower() in TREES:
                return components[k:]
        return []

    def path_candidate(self, parts, category, source, line, root_ok):
        literal = []
        for part in parts:
            found = PLACEHOLDER.search(part)
            if found is not None:
                literal = self.rooted(normalise(literal))
                if literal:
                    self.flag(literal, part[:found.start()], category, source, line, "template", True, root_ok, "/".join(parts))
                return
            literal.append(part)
        literal = self.rooted(normalise(literal))             # 'docs/./a' and 'docs/b/../a' are 'docs/a'; 'docs/../specs/x' is specs
        if not literal:
            return
        full = "/".join(literal).lower()
        if full not in self.file_by_lower and full not in self.dir_set:
            trimmed = literal[-1].rstrip("*_]")                              # emphasis or a bracket closing a path: __docs/A.md__
            if trimmed and trimmed != literal[-1]:
                literal[-1] = trimmed
                full = "/".join(literal).lower()
        if full in self.file_by_lower:
            for path in self.file_by_lower[full]:
                self.record(path, category, source, line, "path")
        elif len(literal) >= 2:
            self.flag(literal, "", category, source, line, "directory", False, root_ok)
        elif category == "policy":
            self.model.policy_roots[source].add(literal[0].lower())    # a tree named on its own tags no single file

    def scan_chain_line(self, category, source, number, line, root_ok, suffix, verbatim=True):
        """The quoted literals of ONE logical line (comments already gone), joined into path chains. ``verbatim`` is False
        when a string of the line was rebuilt from several physical lines."""
        if "'" not in line and '"' not in line:
            return
        pieces = joined_pieces(literal_pieces(line, suffix, verbatim), line, suffix)   # streamed: 3 MB of quotes is 1.5M matches
        pending = next(pieces, None)
        while pending is not None:
            run = [pending]
            pending = next(pieces, None)
            while pending is not None:
                a, b = run[-1].end, pending.start
                if SLASH_BETWEEN.fullmatch(line, a, b):
                    run.append(pending)
                elif COMMA_BETWEEN.fullmatch(line, a, b) and self.in_path_call(line, run[0].start):
                    run.append(pending)
                else:
                    break
                pending = next(pieces, None)
            end = run[-1].end
            self.process_run(category, source, number, line, run, root_ok, line[end:end + 300])

    def scan_command_line(self, category, source, number, code, profile):
        """A tree named as a bare command argument (``find docs``, ``Get-ChildItem docs -Recurse``, ``dir docs /s``) on one
        shell, PowerShell or batch line is a whole-tree reference like a quoted one. ``code`` is already free of comments,
        here-documents and here-strings; the text of quoted strings is blanked here (``echo "read docs now"``), so prose
        cannot name a tree. The ``$(...)`` inside a string arrives as a line of its own (``shell_units``)."""
        body = lex_line(code, profile, blank=True)
        for match in ARG_ROOT.finditer(body):
            after = body[match.end():match.end() + 1]
            if ((after == ")" or (after == "|" and CASE_TAIL.match(body, match.end(), match.end() + 200)))
                    and is_case_pattern(body, match.start(1))):
                continue                                              # ``docs) build ;;``, ``docs|specs) ...``: a case pattern
            self.flag([match.group(1)], "", category, source, number, "bare-root", False, True)
        for brace in BRACE_ALTERNATIVES.finditer(body):                  # cat {docs,specs}/*.md
            for alternative in brace.group(1).split(","):
                tree = alternative.strip("./\\").replace("\\", "/").split("/", 1)[0].lower()
                if tree in TREES:
                    self.flag([tree], "", category, source, number, "bare-root", False, True)

    def scan_config(self, category, source, text, suffix, root_ok, fixed_line):
        """Complete path values of a config file: quoted strings (escapes decoded, ``R\\u00e9sum\\u00e9``) and, outside JSON,
        unquoted YAML/INI/TOML values (``contract_root: reference/flight specs``, ``- root: reference``) and the unquoted
        items of flow collections (``roots: [docs, x]``) or INI lists (``roots = docs, x``). The path scan stops at a
        blank, so each value is resolved as ONE literal. Comments, the bodies of YAML block scalars (prose), the words
        of a plain scalar (``note: Notes, docs``) and a TOML / INI section header (``[docs]``) name nothing."""
        profile = PROFILES[suffix]
        yaml, ini = suffix in (".yaml", ".yml"), suffix in (".ini", ".cfg")
        lines_list = text.split("\n")
        block, depth, ini_open, ini_indent = None, 0, False, 0     # ``ini_open``: an INI option (at ``ini_indent``) whose value deeper lines continue
        for index, raw in enumerate(lines_list):
            if suffix == ".toml":
                raw = fold_triple_quotes(raw)
            if block is not None:
                if not raw.strip() or len(raw) - len(raw.lstrip()) > block:
                    continue                                          # the body of a block scalar: its folded value was read at the key
                block = None
            if not raw.strip():
                continue
            code = lex_line(raw, profile)
            if not code.strip():
                continue
            entries = [(piece, code) for piece in literal_pieces(code, suffix)] if ("'" in code or '"' in code) else []
            at = (ini_value_start(code) if ini else value_start(code)) if suffix != ".json" else None
            indent = len(raw) - len(raw.lstrip())
            continuation = ini and ini_open and indent > ini_indent         # configparser tests the indentation BEFORE it looks for a header
            if ini and not continuation:                                   # a new option (or a header, or junk) closes the old value
                ini_open, ini_indent = at is not None and SECTION_HEADER.fullmatch(code) is None, indent
            if continuation:                                               # a line indented DEEPER than its option is folded into the
                at = len(code) - len(code.lstrip())                        # value above: the whole line is the value
            if entries and (yaml or ini):                                  # quotes inside a plain scalar are prose
                opens = flow_start(code, depth, at) if yaml and (depth > 0 or "[" in code or "{" in code) else None
                if opens is not None:                                      # a collection: a quote counts where an item begins
                    entries = [(piece, line) for piece, line in entries if piece.start < opens or item_boundary(code, piece.start)]
                else:
                    limit = scalar_limit(code, at)
                    if limit is not None:
                        entries = [(piece, line) for piece, line in entries if piece.start < limit]
            if suffix != ".json":
                value = unquoted_value(code, at)
                if value is not None and yaml and BLOCK_SCALAR.fullmatch(value.raw):     # ``root: >-`` and the lines indented under it
                    block = block_base(raw)
                    folded = self.block_scalar(lines_list, index)
                    value = Piece(folded, 0, len(folded), False) if folded else None
                if value is not None:
                    entries.append((value, code))
                if not (SECTION_HEADER.fullmatch(code) and not yaml):
                    blanked = lex_line(raw, profile, blank=True)
                    if ini:
                        entries += [(piece, blanked) for piece in list_items(blanked, at if continuation else None)]
                    else:
                        start = flow_start(blanked, depth)
                        if start is not None:
                            items, depth = flow_items(blanked, depth, start)
                            entries += [(piece, blanked) for piece in items]
            for piece, line in entries:
                folded = piece.raw.replace("\\", "/")
                if "/" in folded or folded.strip("./").lower() in TREES:
                    self.process_run(category, source, fixed_line or index + 1, line, [piece], root_ok, "")

    @staticmethod
    def block_scalar(lines_list, index):
        """The text of the YAML block scalar that starts after line ``index``: the next lines indented deeper than it
        (at most 50), folded with blanks."""
        base = block_base(lines_list[index])
        parts = []
        for line in lines_list[index + 1:index + 51]:
            if not line.strip():
                continue
            if len(line) - len(line.lstrip()) <= base:
                break
            parts.append(line.strip())
        return " ".join(parts)

    @staticmethod
    def in_path_call(line, start):
        return PATH_CALL.search(line, max(0, start - WINDOW), start) is not None

    def path_context(self, line, run, raws, after_text):
        if len(run) > 1 or raws[0].startswith(("/", "\\")) or raws[0].endswith(("/", "\\")):
            return True
        start = run[0].start
        return (line[max(0, start - WINDOW):start].rstrip().endswith("/") or SLASH_AFTER.match(after_text) is not None
                or self.in_path_call(line, start))

    def process_run(self, category, source, number, line, run, root_ok, after_text):
        comps, head, placeholder, raws = [], "", False, []
        for piece in run:
            raws.append(piece.raw)
            parts, literal_head, has_placeholder = split_literal(piece.raw)
            comps += parts
            if has_placeholder:
                head, placeholder = literal_head, True
                break
        if any(URL_MARK in raw or has_network_url(raw) for raw in raws):
            return                                       # a literal that held a URL (blanked) or decodes to one (``https:\/\/host\/docs``)
        slash_tail = not placeholder and VARIABLE_TAIL.match(after_text) is not None
        plus_tail = not placeholder and not slash_tail and PLUS_TAIL.match(after_text) is not None
        transform = None if placeholder or slash_tail or plus_tail else TRANSFORM_TAIL.match(after_text)
        comps = normalise(comps)
        file_like = bool(comps) and not raws[-1].replace("\\", "/").endswith("/") and comps[-1].lower() not in TREES
        if plus_tail and file_like:
            comps, head = comps[:-1], comps[-1]          # 'docs/x/result_' + name: the last part is only a HEAD of a name
        elif transform is not None and file_like:
            last = comps[-1]                             # Path('a/report').with_suffix('.json'): same stem; with_name/with_stem: any name
            comps, head = comps[:-1], (last.rsplit(".", 1)[0] if transform.group(1) == "with_suffix" else "")
        variable_tail = slash_tail or plus_tail or transform is not None
        unresolved = placeholder or variable_tail
        if not comps:
            return
        # the excerpt shown beside a construction: a bounded window, never the whole line (a 3 MB line with 25,000
        # constructions cost O(line) each when the full line was stripped and copied per construction)
        excerpt = line[max(0, run[0].start - 40):run[0].start + 200] if unresolved else ""
        whole = raws[0].replace("\\", "/")
        # RUN cannot consume a space or a comma, so the path scan read this literal in full only if it is all path characters
        single_path = len(run) == 1 and run[0].exact and "/" in whole and RUN.fullmatch(whole) is not None
        first = next((k for k, c in enumerate(comps) if c.lower() in TREES), None)
        if first is not None:
            comps = comps[first:]
            if single_path and not variable_tail and (len(comps) >= 2 or placeholder):
                return                                           # one complete literal: the path scan read it
            full = "/".join(comps).lower()
            if len(comps) == 1:
                if unresolved:
                    self.flag(comps, head, category, source, number, "chain-directory", True, root_ok, excerpt)
                else:
                    self.flag(comps, "", category, source, number, "bare-root", False, root_ok)
            elif unresolved:
                self.flag(comps, head, category, source, number, "chain-directory", True, root_ok, excerpt)
            elif full in self.file_by_lower:
                for path in self.file_by_lower[full]:
                    self.record(path, category, source, number, "chain")
            else:
                self.flag(comps, "", category, source, number, "chain-directory", False, root_ok)
            return
        if not self.path_context(line, run, raws, after_text):
            return                                               # a lone literal with no path operator around it
        tail = "/".join(comps).lower()
        for directory in sorted(self.ending_with(self.reversed_dir_keys, self.reversed_dirs, tail)):
            self.flag(directory.split("/"), head, category, source, number, "chain-suffix", True, root_ok, excerpt)
        if len(comps) >= 2 and not unresolved:
            for path in sorted(self.ending_with(self.reversed_file_keys, self.reversed_files, tail)):
                self.record(path, category, source, number, "chain-suffix", True)

    def scan_names(self, category, source, folded, lines):
        words = set(WORD.findall(folded))
        hits = {w.rstrip(".").lower() for w in words} & self.word_names
        if hits:
            first = {}
            for match in WORD.finditer(folded):                    # ONE pass, stopping at the last name's first occurrence
                name = match.group(0).rstrip(".").lower()
                if name in hits and name not in first:
                    first[name] = match.start()
                    if len(first) == len(hits):
                        break
            for name in sorted(hits):
                for path in self.basenames[name]:
                    self.record(path, category, source, lines.at(first[name]), "name")
        if self.odd_names:
            stripped = {w.rstrip(".").lower() for w in words}
            candidates = sorted(name for name, (tokens, _) in self.odd_names.items() if stripped.issuperset(tokens))
            if candidates:
                self.scan_odd_names(category, source, folded, lines, candidates)

    def scan_odd_names(self, category, source, folded, lines, candidates):
        """Names a word cannot hold (a space, a plus ...) are located through the SEQUENCE of word tokens they contain:
        the occurrences of a name's rarest token give its candidate positions, and only those are confirmed against the
        text. Cost is one pass over the source plus the confirmations, which are BUDGETED per source (ODD_CONFIRMATIONS):
        an inventory of near-identical names against a source of decoys would otherwise cost names x occurrences. A name
        that runs out of budget is MARKED as mentioned, at the first occurrence of its rarest token, and counted on the
        page: over-marking is the safe direction."""
        sequence, offsets = [], []
        for match in WORD.finditer(folded):
            token = match.group(0).rstrip(".").lower()
            if token:                                    # "a . b.md": a lone dot is no token, here or in the name
                sequence.append(token)
                offsets.append(match.start())
        count = Counter(sequence)
        rarest = {name: min(self.odd_names[name][0], key=lambda token: (count[token], token)) for name in candidates}
        positions = defaultdict(list)
        wanted = set(rarest.values())
        for index, token in enumerate(sequence):
            if token in wanted:
                positions[token].append(index)
        budget = ODD_CONFIRMATIONS
        for name in candidates:
            tokens, lead = self.odd_names[name]
            size, rare = len(tokens), rarest[name]
            slots = [j for j, token in enumerate(tokens) if token == rare]
            found, exhausted = None, False
            for index in positions[rare]:
                for j in slots:
                    if budget <= 0:
                        exhausted = True
                        break
                    budget -= 1
                    begin = index - j
                    if begin < 0 or sequence[begin:begin + size] != tokens:
                        continue
                    start = offsets[begin] - lead
                    end = start + len(name)
                    if (start >= 0 and folded[start:end].lower() == name
                            and (start == 0 or folded[start - 1] not in NAME_BEFORE)
                            and (end >= len(folded) or folded[end] not in NAME_AFTER)):
                        found = start
                        break
                if found is not None or exhausted:
                    break
            if found is None and exhausted:
                found = max(0, offsets[positions[rare][0]] - lead)
                self.model.gaps["odd_names_overmarked"] += 1
            if found is not None:
                for path in self.basenames[name]:
                    self.record(path, category, source, lines.at(found), "name")


# ------------------------------------------------------------------------------------------------ the model

def tracked_files(repo):
    return [p.decode("utf-8", "replace") for p in git(repo, "ls-files", "-z").split(b"\0") if p]


def new_gaps():
    gaps = defaultdict(int)
    for key in ("missing_trees", "links_skipped", "loop_links_skipped", "outputs_links_skipped", "unreadable_paths"):
        gaps[key] = []
    gaps["loop_disabled"] = False
    return gaps


def build_model(repo, *, scan_loop=True):
    repo = Path(repo)
    model = Model()
    gaps = model.gaps = new_gaps()
    gaps["loop_disabled"] = not scan_loop
    tracked = tracked_files(repo)                              # a git failure raises: never an empty "code" category
    tracked_set = set(tracked)
    try:
        model.head = git(repo, "rev-parse", "--short", "HEAD").decode().strip()
    except GitError:
        model.head = "(no commits)"

    texts, excluded = {}, set()
    for rel, (size, mtime_ns) in sorted(inventory(repo, gaps).items()):
        if rel == OUTPUT:
            continue
        suffix = Path(rel).suffix.lower()
        text = None
        if suffix in TEXT_SUFFIXES or Path(rel).name in TEXT_NAMES:
            text = read_text(repo / rel, gaps, "local")
        if text is not None and is_generated(text):
            gaps["generated_marker_files"] += 1
            excluded.add(rel)
            continue
        model.targets[rel] = Target(rel, rel in tracked_set, title_of(text or "", suffix), size, mtime_ns)
        if text is not None:
            texts[rel] = text
    scanner = Scanner(model, list(model.targets))
    link_guard = make_link_guard(repo)

    for rel in sorted(tracked):                                # code and policy: tracked text files
        if rel == OUTPUT or rel in excluded:
            continue
        if Path(rel).suffix.lower() not in TEXT_SUFFIXES and Path(rel).name not in TEXT_NAMES:
            gaps["tracked_skipped_binary"] += 1
            continue
        if rel in model.targets:
            text = texts.get(rel)
        else:
            behind_link = link_guard(rel)
            if behind_link:
                gaps["links_skipped"].append(behind_link)             # never read a tracked file THROUGH a junction
                continue
            text = read_text(repo / rel, gaps, "tracked")
            if text is not None and is_generated(text):
                gaps["generated_marker_files"] += 1
                continue
        if text is not None:
            scanner.scan("policy" if rel in POLICY_FILES else "code", rel, text)

    for rel, text in sorted(texts.items()):                    # notes: the other files inside the trees
        if rel not in tracked_set and rel not in POLICY_FILES:
            scanner.scan("notes", rel, text)

    if scan_loop:
        scan_loop_records(repo, scanner, gaps)
    scan_output_records(repo, scanner, gaps)

    for rel, target in model.targets.items():
        if not target.tags & REFERENCE_TAGS:
            target.tags.add(T_NONE)
        for rule, reason in PROTECTED_RULES:
            if rule(rel):
                target.tags.add("PROTECTED:" + reason)
    gaps["missing_trees"] = sorted(gaps["missing_trees"])
    gaps["links_skipped"] = sorted(set(gaps["links_skipped"]))
    model.hubs = sorted(((src, len(out)) for src, out in model.outgoing.items()
                         if len(out) >= HUB_MIN_LINKS and Path(src).suffix.lower() in PROSE_SUFFIXES),
                        key=lambda item: (-item[1], item[0]))[:8]
    return model


def scan_loop_records(repo, scanner, gaps):
    blocked = redirected_component(repo, LOOP_DIR)
    if blocked:
        gaps["loop_links_skipped"].append(blocked)
        return
    base = repo / LOOP_DIR
    if not base.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(base, onerror=lambda error: note_unreadable(gaps, error.filename)):
        kept = []
        for name in sorted(dirnames):
            full = Path(dirpath) / name
            try:
                linked = is_redirect(os.lstat(full))
            except OSError:
                note_unreadable(gaps, full)
                continue
            if linked:
                gaps["loop_links_skipped"].append(full.relative_to(repo).as_posix())
            elif name != "__pycache__":
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            full = Path(dirpath) / name
            suffix = full.suffix.lower()
            if suffix == ".jsonl":
                gaps["loop_skipped_jsonl"] += 1
            elif suffix in TEXT_SUFFIXES:
                if len(full.relative_to(base).parts) > LOOP_SHALLOW_DEPTH and suffix not in LOOP_DEEP_SUFFIXES:
                    gaps["loop_skipped_deep_extension"] += 1
                    continue
                text = read_text(full, gaps, "loop")
                if text is None:
                    continue
                if is_generated(text):
                    gaps["generated_marker_files"] += 1
                    continue
                gaps["loop_files_scanned"] += 1
                scanner.scan("loop", full.relative_to(repo).as_posix(), text)


def scan_output_records(repo, scanner, gaps):
    blocked = redirected_component(repo, OUTPUTS_DIR)
    if blocked:
        gaps["outputs_links_skipped"].append(blocked)
        return
    base = repo / OUTPUTS_DIR
    if not base.is_dir():
        return
    try:
        with os.scandir(base) as it:
            campaigns = sorted(it, key=lambda e: e.name)
    except OSError:
        note_unreadable(gaps, base)
        return
    for campaign in campaigns:
        path = Path(campaign.path)
        try:
            info = os.lstat(path)
        except OSError:
            note_unreadable(gaps, path)
            continue
        if is_redirect(info):
            gaps["outputs_links_skipped"].append(path.relative_to(repo).as_posix())
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        gaps["outputs_campaigns_scanned"] += 1
        try:
            names = sorted(os.listdir(path))
        except OSError:
            note_unreadable(gaps, path)
            continue
        for name in names:
            full = path / name
            if full.suffix.lower() not in TEXT_SUFFIXES and name not in TEXT_NAMES:
                try:
                    info = os.lstat(full)
                    if stat.S_ISREG(info.st_mode) and not is_redirect(info):
                        gaps["outputs_skipped_extension"] += 1       # a record of a type that is not text: counted, not hidden
                except OSError:
                    note_unreadable(gaps, full)
                continue
            text = read_text(full, gaps, "outputs")
            if text is not None and not is_generated(text):
                scanner.scan("outputs", full.relative_to(repo).as_posix(), text)


# ------------------------------------------------------------------------------------------------ rendering

def link(path):
    relative = path[len("docs/"):] if path.startswith("docs/") else "../" + path
    return quote(relative, safe="/.-_~()")


def cell(text):
    return str(text).replace("\n", " ")


def describe(target):
    shown = []
    for category in ("code", "policy", "notes", "loop", "outputs"):
        for evidence in target.evidence.get(category, ())[:1]:
            others = target.source_counts[category] - 1
            shown.append(f"{category}: {evidence.source}:{evidence.line} ({evidence.form})" + (f" +{others} more" if others else ""))
    return "; ".join(shown) if shown else "-"


def render(model, today):
    g = model.gaps
    out = [MARKER, "", "# Workspace index: docs/, specs/, reference/", "",
           f"Generated {today} from git HEAD `{model.head}` by `python -B scripts/workspace_index.py`. "
           "This file is LOCAL: those trees are git-ignored by policy, and so is this index.", "",
           "It is a navigation aid with its uncertainty shown. It does not authorise moving, archiving or deleting "
           "anything, and it infers nothing about which document is current from a modification time. "
           f"`{T_NONE}` means only that the scanned sources (listed below) contained no match.", "",
           "This page is evidence, not proof: the ways a path can be spelled are open-ended, and the last section names the "
           "ones this scan does not decode. Before moving or archiving a file, search its name and stem across the repository "
           "and the local records, and validate every move with the test suite and the preservation baseline.", "",
           "## Start here", "", "### Hub documents (Markdown that links the most other documents in these trees)", ""]
    if model.hubs:
        for path, count in model.hubs:
            out.append(f"- [{Path(path).name}]({link(path)}) {cell(model.targets[path].title)} - links {count} documents")
    else:
        out.append("- none found")
    out += ["", "### Well-known entry points", ""]
    present = [p for p in sorted(model.targets) if Path(p).name in ENTRY_POINT_NAMES]
    out += [f"- [{Path(p).name}]({link(p)}) {cell(model.targets[p].title)} (`{p}`)" for p in present] or ["- none found"]
    named = sorted(((t.named_count, p) for p, t in model.targets.items()
                    if t.named_count and p.endswith(".md") and not p.startswith("reference/evidence/")),
                   key=lambda item: (-item[0], item[1]))[:10]
    out += ["", "### Documents most often named by tracked sources", ""]
    out += [f"- [{Path(p).name}]({link(p)}) {cell(model.targets[p].title)} - {n} tracked source(s)" for n, p in named] or ["- none found"]

    out += ["", "### Files by strongest evidence found (a fact about the scan, not a judgment about use)", ""]
    out += [f"- `{name}`: {TIER_MEANING[name]}" for name in TIER_NAMES] + [""]
    for tree in TREES:
        members = [t for p, t in model.targets.items() if p.split("/")[0] == tree]
        if members:
            counted = defaultdict(int)
            for target in members:
                counted[strongest(target)] += 1
            out.append(f"- {tree}/: " + ", ".join(f"{name} {counted[name]}" for name in TIER_NAMES))

    out += ["", "## What was and was not scanned", "",
            f"- trees missing: {', '.join(g['missing_trees']) or 'none'}",
            f"- text files over the size cap ({SIZE_CAP // (1024 * 1024)} MB) skipped: tracked {g['tracked_skipped_size']}, "
            f"in-tree {g['local_skipped_size']}, loop {g['loop_skipped_size']}, outputs {g['outputs_skipped_size']}",
            f"- tracked files skipped as non-text: {g['tracked_skipped_binary']}",
            f"- notebooks are read cell by cell (evidence 'line' = cell number); malformed cells coerced or skipped: "
            f"{g['notebook_cells_malformed']}; notebooks that could not be decoded (not JSON, or no cells), read as raw text: "
            f"{g['notebooks_unparsed']}",
            f"- files that contain U+FFFD after decoding (invalid bytes, or the character itself): {g['decode_replaced']}",
            f"- files that still hold NUL bytes after decoding (binary content; only their readable parts are scanned): "
            f"{g['nul_bytes_files']}",
            f"- unreadable entries: {g['unreadable']}" + (f" (first: {', '.join(g['unreadable_paths'])})" if g["unreadable_paths"] else ""),
            f"- links and junctions in or below the trees, listed and never followed: {len(g['links_skipped'])}"
            + (f" ({', '.join(g['links_skipped'][:5])})" if g["links_skipped"] else ""),
            f"- generated files excluded from every source category: {g['generated_marker_files']}",
            f"- file names with spaces or punctuation marked as mentioned WITHOUT confirmation (a source exhausted the "
            f"{ODD_CONFIRMATIONS:,}-confirmation budget; marking is the safe direction): {g['odd_names_overmarked']}",
            f"- path-like strings too long to resolve (more than twice the deepest inventoried path plus 6 components after a "
            f"tree name, as in `docs/../docs/../...`; they cost a bounded amount, so they are counted, not followed): "
            f"{g['paths_too_long']}"]
    if g["loop_disabled"]:
        out.append("- loop records (`.claude-codex-loop/runs`): NOT scanned (`--no-loop`)")
    else:
        out.append(f"- loop records: {g['loop_files_scanned']} files scanned; skipped {g['loop_skipped_jsonl']} `.jsonl`, "
                   f"{g['loop_skipped_deep_extension']} deeper than {LOOP_SHALLOW_DEPTH} levels with an extension other than "
                   f"`.json` `.md` `.txt` `.ps1` `.sh`, {len(g['loop_links_skipped'])} linked directories")
    out.append(f"- outputs records: only the top level of each campaign directory ({g['outputs_campaigns_scanned']} scanned; "
               f"{g['outputs_skipped_extension']} top-level records not text (by extension) and not read; "
               f"{len(g['outputs_links_skipped'])} linked campaign directories not followed"
               + (f": {', '.join(g['outputs_links_skipped'][:5])}" if g["outputs_links_skipped"] else "")
               + "; a file named only inside a linked campaign can therefore show a weaker tier than it has)")
    if model.policy_roots:
        out.append("- policy files naming a tree as a whole (no per-file tag): "
                   + "; ".join(f"`{src}` ({', '.join(sorted(trees))})" for src, trees in sorted(model.policy_roots.items())))

    out += ["", "## Known limits (what this scan cannot see by design)", "",
            "- a path read at run time from an environment variable, a command-line argument or a configuration value that is "
            "not in a scanned file;",
            "- files found by a walk (`os.walk`, `rglob`, `Get-ChildItem -Recurse`) over a directory whose location is held in a "
            "variable assigned elsewhere: only the literal parts of the path are matched, and the constructions listed below "
            "mark what they could name;",
            "- a module imported by a computed name (`importlib.import_module(name)`) or through a `sys.path` entry the scan "
            "cannot resolve: an `import stem` marks every `.py` file sharing the stem;",
            "- a path assembled across several statements (`x = \"docs\"` then `y = x + \"/a.md\"` as separate statements);",
            "- unquoted shell arguments with backslash-escaped spaces (`my\\ reports`) and PowerShell backtick escapes "
            "(``my` reports``), whose escaped blank splits the word;",
            "- percent-encoded paths (`%20`), Windows 8.3 short names (`DOCS~1`), and directory names with non-ASCII "
            "characters inside a Markdown link;",
            "- the line between code and data: a multi-line quoted string (a quoted YAML scalar or a shell string that continues "
            "on the next line) is read one line at a time, and so is a plain YAML scalar continued on the next line; a command "
            "held in a string (`bash -c '...'`, `Invoke-Expression`), in a "
            "here-document piped to a shell, or in a CI configuration (`run: |`, `script:` lists) is skipped as text, like "
            "comments, URLs, docstrings, here-strings and PowerShell block comments; and, the other way round, an UNQUOTED shell "
            "word that is only prose (`echo read docs`) or an INI value that lists single words (`description = notes, docs`) is "
            "read as the tree. A space-separated INI list (`testpaths = tests docs`) is not split. A one-line YAML flow collection "
            "whose plain item holds an odd number of one quote character, followed by an item quoted with that same character "
            "(`[a \" b, \"docs\"]`), pairs the two quote characters across the items. Only the replacement fields of "
            "an f-string that does not reuse its own quote character (before Python 3.12) are read as code, and a shell "
            "`$(...)` or backtick substitution to a depth of three;",
            "- `pathlib` or `os.path` chains beyond the literal forms read above (`/`, `join`, `+`, `with_suffix`, "
            "`with_name`, `with_stem`): `.parent`, `.parents[n]`, `.relative_to`, `.resolve()` results are not followed, and a "
            "`.joinpath('A.md')` chained on a call marks the tree and the name, not the file;",
            "- notebooks: a `!` line, an assigned one (`files = !ls docs`), a `%` line magic and a `%%bash` / `%%sh` / `%%script` "
            "cell are read as shell lines; any other cell magic is read as Python;",
            "- anything inside the linked directories that were not followed, a file type that is not text, and the sources "
            "listed above as skipped (over the size cap, deeper than the loop limit, `.jsonl`)."]

    out += ["", "## Inventory", ""]
    for tree in TREES:
        members = sorted(p for p in model.targets if p.split("/")[0] == tree)
        if not members:
            continue
        out += [f"<details><summary>{tree}/ - {len(members)} files</summary>", ""]
        for path in members:
            t = model.targets[path]
            out.append(f"- [{path}]({link(path)}) {cell(t.title)} ({t.size / 1024:.1f} KB, modified {modified(t.mtime_ns)}, "
                       f"strongest evidence: {strongest(t)}{', tracked' if t.tracked else ''})")
            out.append(f"  - tags: {', '.join(sorted(t.tags))}")
            out.append(f"  - evidence: {cell(describe(t))}")
        out += ["", "</details>", ""]

    out += ["## Constructions with a non-literal tail in tracked sources", "",
            "Each builds a path whose tail is a placeholder, an f-string or a variable. The files below the prefix whose names "
            f"could start with the head carry `{T_UNRESOLVED}`; `matched` is how many that is here.", ""]
    constructions = sorted(model.constructions, key=lambda c: (c.source, c.line, c.prefix, c.head))
    out += [f"- `{c.source}:{c.line}` -> `{c.prefix}/{c.head}...` matched {c.matched}: `{cell(c.text)}`" for c in constructions] or ["- none found"]
    if g["constructions_unlisted"]:
        out.append(f"- ... and {g['constructions_unlisted']} more constructions not listed (the list is capped at "
                   f"{MAX_CONSTRUCTIONS}; the files they could name ARE marked)")

    counts = defaultdict(int)
    for target in model.targets.values():
        for tag in target.tags:
            counts[tag.split(":")[0]] += 1
    out += ["", "## Tag counts", ""] + [f"- {tag}: {n}" for tag, n in sorted(counts.items())]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ the CLI

def write_index(repo, text):
    docs = repo / "docs"
    try:
        if is_redirect(os.lstat(docs)) or not docs.is_dir():
            raise Refused("docs/ is missing or is a link; nothing was written")
    except OSError as error:
        raise Refused("docs/ does not exist; nothing was written") from error
    target = repo / OUTPUT
    try:
        info = os.lstat(target)
    except FileNotFoundError:
        info = None
    if info is not None:
        if is_redirect(info) or not stat.S_ISREG(info.st_mode):
            raise Refused("docs/INDEX.md is a link or not a regular file; nothing was written")
        with open(target, "rb") as stream:
            first = stream.readline(len(MARKER_BYTES) + 2)      # the marker and at most a CRLF: a longer line is not ours
        if first not in (MARKER_BYTES, MARKER_BYTES + b"\n", MARKER_BYTES + b"\r\n"):    # EXACT: no padding, no BOM, no extra CRs
            raise Refused("docs/INDEX.md exists and is not a generated index; nothing was written")
    descriptor, name = tempfile.mkstemp(dir=str(docs), prefix="INDEX.md.", suffix=".tmp")
    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.write(text.encode("utf-8"))
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-loop", action="store_true", help="skip the loop-record scan")
    parser.add_argument("--check", action="store_true", help="print the index; write nothing")
    parser.add_argument("--today", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        model = build_model(repo, scan_loop=not args.no_loop)
    except GitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    text = render(model, args.today or datetime.date.today().isoformat())
    if args.check:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        sys.stdout.write(text)
        return 0
    try:
        write_index(repo, text)
    except Refused as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: could not write docs/INDEX.md: {error}", file=sys.stderr)
        return 1
    print(f"wrote {repo / OUTPUT} ({len(model.targets)} files indexed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""scripts/workspace_index.py against REAL PARSERS: differential fuzzing with fixed seeds.

Why this exists. Thirteen review rounds of hand-picked YAML / URL shapes never converged; a generator found six more defects in minutes and
the reviewers then answered NO BLOCKING FINDINGS. Each test below builds documents in which ONE value is labelled by construction (it
names the docs tree, names docs/A.md, or names nothing), serializes them in random styles, lets a real parser (PyYAML, ast, json,
tomllib, configparser) confirm that the text means what the structure says (a document it does not load back to the same structure is
discarded), and compares the index's tags for docs/A.md with the label. A labelled value that is not found is an UNDER-mark (the
defect: a file something depends on looks unused); an unlabelled document that marks is an OVER-mark. What is DELIBERATELY read as the
tree (a quoted key named docs) or documented as a limit (a URL whose "://" an escape hides, an apostrophe inside a triple-quoted URL,
a command held in a string, `Path("docs").joinpath("A.md")`) is not generated.

The seeds are fixed, so a failure reproduces and its message prints the offending text. To hunt, raise COUNT or change SEED. The teeth
tests at the end break the scanner on purpose and require these fuzzers to notice: a fuzzer that cannot fail proves nothing.
"""
from __future__ import annotations

import ast
import configparser
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "workspace_index.py"
SEED = 20261010
COUNT = {"yaml": 1500, "python": 1500, "json": 600, "toml": 600, "ini": 600}


def load():
    spec = importlib.util.spec_from_file_location("workspace_index_fuzz_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workspace_index_fuzz_under_test"] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


idx = load()


def tags_of(name, text):
    """The tags docs/A.md ends up with when ``text`` is the only source, read as a tracked ``name``."""
    model = idx.Model()
    for path in ("docs/A.md", "docs/B.md"):
        model.targets[path] = idx.Target(path, False, "", 0, 0)
    model.gaps = idx.new_gaps()
    idx.Scanner(model, list(model.targets)).scan("code", name, text)
    return model.targets["docs/A.md"].tags


def verdict(label, tags):
    path, tree = idx.T_PATH in tags, idx.T_SUBTREE in tags
    if label == "split":                                         # written in two pieces: any reference at all is enough
        return "ok" if tags & idx.REFERENCE_TAGS else "UNDER-split"
    if label == "file":
        return "ok" if path else "UNDER-file"
    if label == "tree":
        return "ok" if (tree or path) else "UNDER-tree"
    return "OVER" if (tree or path) else "ok"


def failures_of(cases, name):
    """[(verdict, text)] for every (label, text) whose tags disagree with the label."""
    found = []
    for label, text in cases:
        problem = verdict(label, tags_of(name, text))
        if problem != "ok":
            found.append((problem, text))
    return found


def report(found):
    shown = sorted(found, key=lambda item: len(item[1]))[:3]
    return f"{len(found)} documents disagree with their parser-confirmed labels; the shortest:\n" + "\n".join(
        f"--- {problem}\n{text}" for problem, text in shown)


# ------------------------------------------------------------------------------------------------ shared vocabulary
TREE = ["docs", "./docs", "docs/", "./docs/", "DOCS", ".\\docs"]
FILE = ["docs/A.md", "./docs/A.md", "Docs/a.md", "docs\\A.md", ".\\docs\\A.md"]
NONE = ["x", "y z", "see docs", "Keep docs tidy", "100%", "a/b", "n.md", "https://x.org/docs", "https://x.org/a(b)", "http://[::1]:8080/docs",
        "https://x.org/it's/docs/A.md", "ftp://h/docs/A.md", "docs.example.com", "mydocs", "docs-old/", "xdocs/A.md", "Don't", "owner's notes",
        "a#b", "a: b", "say \"hi\"", "[x]", "{y}", "https://x.org/a;b|c", "https://x.org/a,b/docs/A.md", "a,b", "1.5x", "",
        "'''", '"""', "a ''' b", "it's a \"\"\" test",                    # triple quotes INSIDE an ordinary string are text, not delimiters
        "https://graph.microsoft.com/v1.0/me/drive/root/search(q='docs')"]
SEGMENT_CHARS = "abcXYZ019-_.~%()[]{};|,'\"<>\\*+!$&=:@^`#?"


def weird_url(rng):
    """A URL whose path holds characters that end a URL in SOME context, and that ends in a tree path."""
    scheme = rng.choice(["http", "https", "ftp", "s3", "git+ssh", "ws"])
    host = rng.choice(["x.org", "localhost:8080", "[::1]", "[2001:db8::1]:443", "user:pw@h.example", "h"])
    segments = ["".join(rng.choice(SEGMENT_CHARS) for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(0, 3))]
    tail = rng.choice(["/docs/A.md", "/docs", "/docs/", "/x/docs/A.md", "/docs/A.md?q=1", "/docs/A.md#frag"])
    return f"{scheme}://{host}/" + "/".join(segments) + tail


def prose(rng):
    return weird_url(rng) if rng.random() < 0.35 else rng.choice(NONE)


def label_one(rng, container):
    """Replace ONE leaf of a nested dict / list of ``(text, label)`` leaves by a labelled one (70%): a hit cannot mask a miss."""
    leaves = []

    def walk(node):
        for key, value in (node.items() if isinstance(node, dict) else enumerate(node)):
            if isinstance(value, tuple):
                leaves.append((node, key))
            else:
                walk(value)

    walk(container)
    if leaves and rng.random() < 0.7:
        node, key = rng.choice(leaves)
        node[key] = (rng.choice(TREE), "tree") if rng.random() < 0.5 else (rng.choice(FILE), "file")


def bare(node):
    if isinstance(node, tuple):
        return node[0]
    return {k: bare(v) for k, v in node.items()} if isinstance(node, dict) else [bare(v) for v in node]


def label_of(node):
    found = set()

    def walk(value):
        if isinstance(value, tuple):
            found.add(value[1])
        else:
            for item in (value.values() if isinstance(value, dict) else value):
                walk(item)

    walk(node)
    return "file" if "file" in found else ("tree" if "tree" in found else None)


def nested(rng, keys, depth=0, limit=3):
    roll = rng.random()
    if depth >= limit or roll < 0.35:
        return (prose(rng), None)
    if roll < 0.70:
        return {rng.choice(keys) + (str(i) if i else ""): nested(rng, keys, depth + 1, limit) for i in range(rng.randint(1, 3))}
    return [nested(rng, keys, depth + 1, limit) for _ in range(rng.randint(1, 4))]


# ------------------------------------------------------------------------------------------------ YAML (oracle: PyYAML)
YAML_KEYS = ["root", "name", "path", "k", "a b", "sphinx:root", "x_1", "docs", "description", "url", "paths", "items"]
STYLE = {}                                         # per-document layout choices (spacing, trailing blanks, filler lines ...)
PLAIN_BAD_START = re.compile(r"^[-?:,\[\]{}#&*!|>'\"%@`\s]")


def plain_ok(text, flow):
    if not text or PLAIN_BAD_START.match(text) or text != text.strip() or ": " in text or " #" in text or text.endswith(":") or "\t" in text:
        return False
    # in a one-line flow collection a plain item with a quote character in it can pair across items with a later quoted item of
    # the same character: a documented limit, so the fuzz writes such an item quoted
    return not (flow and (re.search(r"[,\[\]{}]", text) or "'" in text or '"' in text))


def escaped_double(rng, text):
    out = []
    for char in text:
        roll = rng.random()
        if char == "/" and roll < 0.2:
            out.append("\\/")
        elif char.isascii() and char.isalpha() and roll < 0.1:
            out.append(rng.choice([f"\\x{ord(char):02x}", f"\\u{ord(char):04x}"]))
        elif char in '"\\':
            out.append("\\" + char)
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def yaml_scalar(rng, text, flow, block_ok=False):
    style = rng.choice(["plain", "single", "double"] + (["block"] if block_ok and text and text == text.strip() else []))
    if style == "plain" and not plain_ok(text, flow):
        style = rng.choice(["single", "double"])
    if style == "single":
        out = "'" + text.replace("'", "''") + "'"
    elif style == "double":
        out = escaped_double(rng, text) if rng.random() < 0.4 and "://" not in text else json.dumps(text, ensure_ascii=False)
    elif style == "block":
        return rng.choice(["|-", ">-", "|2-", ">2-", "|-2", ">-2"]), text
    else:
        out = text
    return rng.choice(["&a ", "!!str ", "&b !!str "]) + out if rng.random() < 0.12 else out


def yaml_comment(rng):
    tail = rng.choice(["", "", " ", "  ", "\t"]) if STYLE.get("trailing") else ""
    if rng.random() > 0.2:
        return tail
    return tail + "  # " + rng.choice(["docs", "'docs'", "https://x.org/docs", "it's [docs], {x}", "root: docs", "- docs", "docs/", "see docs"])


def yaml_key(rng, key):
    if key == "docs" or rng.random() < 0.7:
        return key if plain_ok(key, False) else json.dumps(key)
    return json.dumps(key) if rng.random() < 0.5 else "'" + key.replace("'", "''") + "'"


def yaml_flow(rng, node, multiline=False, indent=0):
    note = "  # docs 'x'" if STYLE.get("flow_comments") else ""
    sep = "," + (note if rng.random() < 0.3 else "") + "\n" + " " * (indent + 2) if multiline else ", "
    trailing = "," if STYLE.get("trailing_comma") and rng.random() < 0.5 else ""
    if isinstance(node, tuple):
        return yaml_scalar(rng, node[0], True)
    opener, closer = ("{", "}") if isinstance(node, dict) else ("[", "]")
    parts = ([f"{yaml_key(rng, k)}: {yaml_flow(rng, v, multiline, indent + 2)}" for k, v in node.items()] if isinstance(node, dict)
             else [yaml_flow(rng, v, multiline, indent + 2) for v in node])
    properties = rng.choice(["&a ", "&b ", "!!map " if isinstance(node, dict) else "!!seq "]) if rng.random() < 0.1 else ""
    if multiline:
        return properties + opener + "\n" + " " * (indent + 2) + sep.join(parts) + trailing + "\n" + " " * indent + closer
    return properties + opener + sep.join(parts) + trailing + closer


def yaml_colon(rng):
    return rng.choice([": ", ": ", ":  ", ":   ", ":\t"]) if STYLE.get("spacing") else ": "


def yaml_dash(rng):
    return rng.choice(["- ", "- ", "-  ", "-   ", "-\t"]) if STYLE.get("spacing") else "- "


def yaml_filler(rng, lines):
    if STYLE.get("fillers") and rng.random() < 0.12:
        lines.append(rng.choice(["", "   ", "# docs", "  # see 'docs' \"x\"", "#"]))


def yaml_emit(rng, node, indent, lines, in_item=False):
    pad = " " * indent
    if isinstance(node, dict):
        for number, (key, value) in enumerate(node.items()):
            lead = "" if (in_item and number == 0) else pad
            k = yaml_key(rng, key)
            yaml_filler(rng, lines)
            if isinstance(value, tuple):
                text = yaml_scalar(rng, value[0], False, block_ok=True)
                if isinstance(text, tuple):
                    lines += [f"{lead}{k}{yaml_colon(rng)}{text[0]}", f"{pad}  {text[1]}"]
                else:
                    lines.append(f"{lead}{k}{yaml_colon(rng)}{text}{yaml_comment(rng)}")
            elif rng.random() < 0.3:
                lines.append(f"{lead}{k}{yaml_colon(rng)}{yaml_flow(rng, value, rng.random() < 0.4, indent)}{yaml_comment(rng)}")
            else:
                lines.append(f"{lead}{k}:{yaml_comment(rng)}")
                deeper = indent if isinstance(value, list) and rng.random() < 0.3 else indent + rng.choice([2, 2, 4])
                yaml_emit(rng, value, deeper, lines)
        return
    for value in node:
        yaml_filler(rng, lines)
        if isinstance(value, tuple):
            text = yaml_scalar(rng, value[0], False, block_ok=True)
            if isinstance(text, tuple):
                lines += [f"{pad}{yaml_dash(rng)}{text[0]}", f"{pad}  {text[1]}"]
            else:
                lines.append(f"{pad}{yaml_dash(rng)}{text}{yaml_comment(rng)}")
        elif rng.random() < 0.3:
            lines.append(f"{pad}{yaml_dash(rng)}{yaml_flow(rng, value, rng.random() < 0.4, indent + 2)}{yaml_comment(rng)}")
        else:
            sub = []
            yaml_emit(rng, value, indent + 2, sub, in_item=True)
            sub[0] = pad + "- " + sub[0].lstrip()
            lines += sub


def yaml_cases(rng, count):
    import yaml
    cases = []
    while len(cases) < count:
        STYLE.clear()
        STYLE.update({key: rng.random() < 0.35 for key in ("spacing", "trailing", "fillers", "flow_comments", "trailing_comma")})
        node = nested(rng, YAML_KEYS, 0, 5 if rng.random() < 0.15 else 3)
        node = {"root": node} if isinstance(node, tuple) else node
        label_one(rng, node)
        lines, expected = [], [bare(node)]
        if rng.random() < 0.1:                                       # an unlabelled first document, then the labelled one
            first = nested(rng, YAML_KEYS)
            first = {"first": first} if isinstance(first, tuple) else first
            head = []
            yaml_emit(rng, first, 0, head)
            lines += ["---"] + head + ["..." if rng.random() < 0.3 else "---"]
            expected.insert(0, bare(first))
        elif rng.random() < 0.15:
            lines.append("---")
        if rng.random() < 0.2:
            lines.append("# " + rng.choice(["docs", "see 'docs' and \"docs\"", "./docs/"]))
        if rng.random() < 0.08:                                      # the whole document is ONE flow collection (maybe after ``--- ``)
            lines.append(rng.choice(["", "--- "]) + yaml_flow(rng, node, rng.random() < 0.4))
        else:
            yaml_emit(rng, node, 0, lines)
        text = "\n".join(lines) + "\n"
        try:
            if list(yaml.safe_load_all(text)) == expected:
                cases.append((label_of(node), text))
        except yaml.YAMLError:
            pass
    return cases


# ------------------------------------------------------------------------------------------------ Python (oracle: ast)
def py_literal(rng, s):
    """``s`` as a Python string literal in a random style (checked with ast.literal_eval)."""
    style = rng.choice(["double", "single", "double", "single", "triple", "raw", "unicode"])
    if style == "raw" and (s.endswith("\\") or '"' in s or "'" in s or "\n" in s):
        style = "double"
    if style == "triple" and ("\\" in s or '"' in s or "'" in s or "\n" in s):
        style = "single"
    if style == "double":
        out = json.dumps(s)
    elif style == "single":
        out = "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"
    elif style == "triple":
        quote = rng.choice(['"""', "'''"])
        out = quote + s + quote
    elif style == "raw":
        quote = rng.choice(["'", '"'])
        out = rng.choice(["r", "R"]) + quote + s + quote
    else:
        out = "u" + json.dumps(s)
    try:
        if ast.literal_eval(out) != s:
            raise ValueError(out)
    except (SyntaxError, ValueError):
        out = json.dumps(s)
    return out


PY_STRESS = [
    'a = "it\'s # not a comment"', "b = 'say \"hi\" # no'", 'c = """multi\n"quoted" # line\n"""', "d = '''x ''' + \"y\"", 'e = r"\\d+"',
    'f = f"{a!r:>{10}}"', "g = f'{d[\"k\"]}' if d else 1", "h = f\"{f'{a}'}\"", 'i = rb"\\x00" + Rb"\\xff"', 'j = fr"\\d{a}"',
    'k = ("(", ")", "[", "]", "{", "}", ";", "#")', "l = 1 if a else \\\n    2", 'm = lambda x: {x: "(" for x in []}',
    "n = [\n    # comment with ' and \" and (\n    1,\n]", 'o = "\u00fcn\u00efc\u00f6d\u00e9 \u2014 ok"', "p = 'tab\\there'", "q = {'a': {'b': [1, (2, 3)]}}",
    '@decorator("x")\ndef r():\n    pass', 'class S:\n    """Doc with \'quotes\' and "more"."""\n    x = 1', "s = 'a' 'b' \"c\"",
    't = b"bytes" + b\'more\'', 'u = "\\\\"', 'v = "\\""', "w = '\\''", 'x = "#" ; y = "docs.x"', "z = '''a'b\"c'''",
    'aa = "(" ; bb = ")"', 'cc = ["\\"", \'\\\'\']', "ee = '''\n''' ", 'ff = """\\\n"""',
    'gg = f"""\nprose about the docs folder {a}\n"""', 'hh = """\n{Path("docs")}\n"""', 'ii = 1 if"""\n{Path("docs")}\n""" else 2',
]
PY_FORMS = [
    "p = {e}", "open({e})", "open({e}).read()", "Path({e}).read_text()", "foo({e}, 1)", "foo(1, {e})", "foo(a={e})", "foo(a=1, b={e})",
    "xs = [{e}, 1]", "d = {{'k': {e}}}", "t = ({e},)", "d = {{{e}: 1}}", "y = x[{e}]", "print({e})", "{e}", "assert {e}",
    "z = lambda: {e}", "zs = [q for q in {e}]", "w = {e} if cond else 1", "v = 1 if cond else {e}", "for q in glob({e}):\n    pass",
    "with open({e}) as f:\n    pass", "if cond: p = {e}", "try:\n    p = {e}\nexcept OSError:\n    pass", "@app.route({e})\ndef handler():\n    pass",
    "def f(p={e}):\n    pass", "p: str = {e}", "(p := {e})", "a = 1; b = {e}; c = 2", "p = \\\n    {e}", "p = (  # note\n    {e}\n)",
    "class K:\n    path = {e}", "def g():\n    return {e}", "def g():\n    yield {e}", "async def g():\n    await foo({e})",
    "os.environ['X'] = {e}", "p = {e}.strip()", "p = str({e})", "return_value = [\n    {e},\n    2,\n]",
    "foo(\n    {e},\n    # comment ' \" (\n    key=1,\n)", "x = dict(path={e})", "x = {{'a': 1,\n     'b': {e}}}",
]


def py_reference(rng):
    """(expression, label)."""
    roll = rng.random()
    if roll < 0.07:
        inner, quote = rng.choice([("Path('docs') / 'A.md'", '"'), ("'docs/A.md'", '"'), ("os.path.join('docs', 'A.md')", '"'),
                                   ('Path("docs") / "A.md"', "'"), ('"docs/A.md"', "'")])
        return f"f{quote}x {{{inner}}} y{quote}", "file"
    if roll < 0.10:                                                    # a string continued with a backslash, and bytes
        return rng.choice(['"docs/\\\nA.md"', "'docs/\\\nA.md'", 'b"docs/A.md"', "b'docs/A.md'", '"docs\\\n/A.md"', "'./docs/A\\\n.md'"]), "file"
    if roll < 0.35:
        return py_literal(rng, rng.choice(FILE)), "file"
    if roll < 0.55:
        return rng.choice([
            f"Path({py_literal(rng, 'docs')}) / {py_literal(rng, 'A.md')}", f"Path({py_literal(rng, 'docs')}, {py_literal(rng, 'A.md')})",
            f"os.path.join({py_literal(rng, 'docs')}, {py_literal(rng, 'A.md')})", f"Path({py_literal(rng, 'docs/A.md')})",
            f"Path({py_literal(rng, '.')}) / {py_literal(rng, 'docs')} / {py_literal(rng, 'A.md')}"]), "file"
    if roll < 0.75:
        value = py_literal(rng, rng.choice(TREE))
        return rng.choice([value, f"Path({value})", f"os.listdir({value})"]), "tree"
    glue = rng.choice([" ", "", "\n    ", "  \\\n    "])
    return rng.choice([                                                # split in two: it only has to be found by SOME reference tag
        f"{py_literal(rng, 'docs/')} + {py_literal(rng, 'A.md')}", f"{py_literal(rng, 'docs')} + {py_literal(rng, '/A.md')}",
        "(" + py_literal(rng, "docs/") + glue + py_literal(rng, "A.md") + ")", f"{py_literal(rng, 'docs/{}')}.format({py_literal(rng, 'A.md')})",
        f"Path({py_literal(rng, 'docs')}).joinpath({py_literal(rng, 'A.md')})"]), "split"


def py_distractor(rng):
    roll = rng.random()
    if roll < 0.3:
        literal = py_literal(rng, weird_url(rng))
        return rng.choice([f"url = {literal}", f"u = ({literal})", f"foo({literal}, 1)", f"x = [{literal}, 2]", f"d = {{'k': {literal}}}",
                           f"y = {literal}  # docs", f"os.environ['U'] = {literal}"])
    if roll < 0.55:
        return rng.choice(PY_STRESS)
    if roll < 0.65:
        return "# " + rng.choice(["it's \"docs\" (see) [x] {y} '''", "docs", "TODO: docs", "'docs' \"docs\"", "https://x.org/docs"])
    if roll < 0.75:
        return rng.choice(["docs = 1", "docs_dir = None", "def docs():\n    pass", "class Docs:\n    pass", "x = docs if False else 0"])
    if roll < 0.9:
        return "msg = " + py_literal(rng, rng.choice(["read the docs", "docs.example.com", "see docs for help", "Docs are fun", "mydocs", "docs-old"]))
    return rng.choice(["import os", "from pathlib import Path", "x = 1  # docs", "y = [1, 2]  # 'docs'", "print('hello')"])


def python_cases(rng, count):
    cases = []
    while len(cases) < count:
        reference, label = py_reference(rng) if rng.random() < 0.75 else (None, None)
        parts = [py_distractor(rng) for _ in range(rng.randint(0, 3))]
        if reference is not None:
            statement = rng.choice(PY_FORMS).format(e=reference)
            continued = "\\\n" in reference                    # indenting would change the VALUE of a string continued on the next line
            if rng.random() < 0.3 and not continued:
                statement = "def run():\n" + "\n".join("    " + line for line in statement.split("\n"))
            elif rng.random() < 0.15 and not continued:
                statement = "if True:\n" + "\n".join("    " + line for line in statement.split("\n"))
            parts.insert(rng.randint(0, len(parts)), statement)
        text = "\n".join(parts) + "\n"
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        cases.append((label, text))
    return cases


# ------------------------------------------------------------------------------------------------ JSON / TOML (oracles: json, tomllib)
CONFIG_KEYS = ["root", "name", "path", "k", "a_b", "url", "paths", "items", "description"]       # never "docs": a quoted key is the tree


def json_text(rng, node):
    text = json.dumps(bare(node), indent=rng.choice([None, 1, 2, 4]), ensure_ascii=rng.random() < 0.5,
                      separators=rng.choice([None, (",", ":"), (", ", ": ")]))
    if rng.random() < 0.3:                                           # \/ for / and \u00XX for letters inside strings (not in URLs)
        def escape(match):
            body = match.group(0)
            if "://" in body:
                return body
            body = re.sub(r"(?<!\\)/", lambda m: "\\/" if rng.random() < 0.5 else "/", body)
            return re.sub(r"[a-zA-Z]", lambda m: f"\\u{ord(m.group()):04x}" if rng.random() < 0.1 else m.group(), body)
        text = re.sub(r'"(?:[^"\\]|\\.)*"', escape, text)
    return text + "\n"


def toml_string(rng, s):
    style = rng.choice(["basic", "literal", "basic", "ml_basic", "ml_literal"])
    if style == "literal" and ("'" in s or "\n" in s):
        style = "basic"
    if style in ("ml_literal", "ml_basic") and ("'" in s or '"' in s or "\n" in s):
        style = "basic"
    return {"basic": json.dumps(s, ensure_ascii=False), "literal": "'" + s + "'", "ml_basic": '"""' + s.replace("\\", "\\\\") + '"""',
            "ml_literal": "'''" + s + "'''"}[style]


def toml_key(rng, key):
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) and rng.random() < 0.8 else json.dumps(key)


def toml_value(rng, value):
    if isinstance(value, tuple):
        return toml_string(rng, value[0])
    if isinstance(value, list):
        items = [toml_value(rng, v) for v in value]
        return "[\n  " + ",\n  ".join(items) + ",\n]" if rng.random() < 0.4 else "[" + ", ".join(items) + "]"
    return "{" + ", ".join(f"{toml_key(rng, k)} = {toml_value(rng, v)}" for k, v in value.items()) + "}"


def toml_text(rng, node):
    lines = []

    def table(current, path):
        scalars = {k: v for k, v in current.items() if not isinstance(v, dict)}
        if path:
            lines.append("[" + ".".join(toml_key(rng, p) for p in path) + "]")
        for k, v in scalars.items():
            lines.append(f"{toml_key(rng, k)} = {toml_value(rng, v)}" + ("  # docs 'x'" if rng.random() < 0.15 else ""))
        for k, v in current.items():
            if isinstance(v, dict):
                table(v, path + [k])

    table(node, [])
    return "\n".join(lines) + "\n"


def config_cases(rng, count, kind):
    tomllib = pytest.importorskip("tomllib") if kind == "toml" else None
    cases = []
    while len(cases) < count:
        node = nested(rng, CONFIG_KEYS)
        node = node if isinstance(node, dict) else {"root": node}          # a TOML document is a table
        label_one(rng, node)
        text = json_text(rng, node) if kind == "json" else toml_text(rng, node)
        try:
            loaded = json.loads(text) if kind == "json" else tomllib.loads(text)
        except ValueError:
            continue
        if loaded == bare(node):
            cases.append((label_of(node), text))
    return cases


# ------------------------------------------------------------------------------------------------ INI (oracle: configparser)
INI_WORDS = ["x", "tests", "src", "a/b", "n.md", "see docs", "Keep docs tidy", "https://x.org/docs", "https://x.org/a(b)/docs/A.md", "docs.example.com",
             "mydocs", "docs-old/", "xdocs/A.md", "100%", "a b", "C:\\data", "key=value", "Programming Language :: Python :: 3",
             "http://[::1]:8080/docs", "https://x.org/a;b|c", "ftp://h/docs/A.md", "owner's", "say \"hi\""]
INI_KEYS = ["testpaths", "root", "name", "paths", "docs", "description", "url", "a_b", "path1"]
INI_SECTIONS = ["pytest", "paths", "metadata", "docs", "tool:pytest", "x y", "options.extras_require"]


def ini_cases(rng, count):
    cases = []
    while len(cases) < count:
        config = {rng.choice(INI_SECTIONS): {rng.choice(INI_KEYS) + str(n): [[rng.choice(INI_WORDS) for _ in range(rng.choice([1, 1, 1, 2, 3]))]
                                                                            for _ in range(rng.randint(1, 4))] for n in range(rng.randint(1, 3))}
                  for _ in range(rng.randint(1, 3))}
        slots = [(s, k, i, j) for s, options in config.items() for k, lines in options.items() for i, items in enumerate(lines) for j, _ in enumerate(items)]
        label = None
        if rng.random() < 0.7:
            s, k, i, j = rng.choice(slots)
            label, token = ("tree", rng.choice(TREE)) if rng.random() < 0.5 else ("file", rng.choice(FILE))
            config[s][k][i][j] = token
        out = []
        for section, options in config.items():
            out.append(f"[{section}]")
            for key, lines in options.items():
                base = rng.choice(["", "", "", "  ", "\t"])               # an option may itself be indented; its continuations go deeper
                separator, glue = rng.choice([" = ", " = ", "=", ": ", " : ", ":"]), (", " if rng.random() < 0.7 else ",")
                first, rest = lines[0], lines[1:]
                if rng.random() < 0.5 and len(lines) > 1:               # nothing on the option's own line
                    out.append(f"{base}{key}{separator.rstrip()}")
                    rest = lines
                else:
                    out.append(f"{base}{key}{separator}{glue.join(first)}")
                for line in rest:
                    deeper = base + rng.choice([" ", "  ", "    ", "\t"])
                    if rng.random() < 0.08:
                        out.append(deeper + "[unit]")                    # a deeper line that LOOKS like a header is still the value
                    out.append(deeper + glue.join(line))
                    if rng.random() < 0.08:
                        out.append(rng.choice(["", "   ", "# docs", "; docs"]))
        text = "\n".join(out) + "\n"
        parser = configparser.ConfigParser(interpolation=None, delimiters=("=", ":"), strict=False)
        try:
            parser.read_string(text)
        except configparser.Error:
            continue
        items = {piece.strip() for section in parser.sections() for key in parser[section] for line in parser[section][key].splitlines()
                 for piece in line.split(",")}
        if label is None or items & set(TREE + FILE):
            cases.append((label, text))
    return cases


# ------------------------------------------------------------------------------------------------ the tests
def test_a_yaml_document_means_to_the_index_what_pyyaml_says_it_means():
    pytest.importorskip("yaml")
    found = failures_of(yaml_cases(random.Random(SEED), COUNT["yaml"]), "configs/c.yaml")
    assert not found, report(found)


def test_a_python_program_means_to_the_index_what_ast_says_it_means():
    found = failures_of(python_cases(random.Random(SEED + 1), COUNT["python"]), "src/m.py")
    assert not found, report(found)


def test_a_json_document_means_to_the_index_what_json_says_it_means():
    found = failures_of(config_cases(random.Random(SEED + 2), COUNT["json"], "json"), "configs/c.json")
    assert not found, report(found)


def test_a_toml_document_means_to_the_index_what_tomllib_says_it_means():
    found = failures_of(config_cases(random.Random(SEED + 3), COUNT["toml"], "toml"), "configs/c.toml")
    assert not found, report(found)


def test_an_ini_file_means_to_the_index_what_configparser_says_it_means():
    found = failures_of(ini_cases(random.Random(SEED + 4), COUNT["ini"]), "configs/c.ini")
    assert not found, report(found)


# -- teeth: the fuzzers must notice a scanner that is broken on purpose --------------------------------------------------------
def test_the_python_fuzzer_notices_a_scanner_that_ignores_f_string_fields(monkeypatch):
    monkeypatch.setattr(idx, "fstring_expressions", lambda code, depth=0: [])
    assert any(problem == "UNDER-file" for problem, _ in failures_of(python_cases(random.Random(1), 300), "src/m.py"))


def test_the_python_fuzzer_notices_a_scanner_that_reads_a_url_as_a_path(monkeypatch):
    monkeypatch.setattr(idx, "blank_urls", lambda text, flow=False, lists=False: text)
    assert any(problem == "OVER" for problem, _ in failures_of(python_cases(random.Random(2), 300), "src/m.py"))


def test_the_yaml_fuzzer_notices_a_scanner_that_reads_a_url_as_a_path(monkeypatch):
    pytest.importorskip("yaml")
    monkeypatch.setattr(idx, "blank_urls", lambda text, flow=False, lists=False: text)
    assert any(problem == "OVER" for problem, _ in failures_of(yaml_cases(random.Random(3), 300), "configs/c.yaml"))


def test_the_yaml_fuzzer_notices_a_scanner_that_ignores_nested_sequences(monkeypatch):
    pytest.importorskip("yaml")
    monkeypatch.setattr(idx, "SEQUENCE_DASH", re.compile(r"(?!)"))
    assert any(problem.startswith("UNDER") for problem, _ in failures_of(yaml_cases(random.Random(4), 600), "configs/c.yaml"))


def test_the_ini_fuzzer_notices_a_scanner_that_cannot_see_continuation_lines(monkeypatch):
    real = idx.Scanner.scan_config

    def dedented(self, category, source, text, suffix, root_ok, fixed_line):
        return real(self, category, source, "\n".join(line.lstrip() for line in text.split("\n")), suffix, root_ok, fixed_line)

    monkeypatch.setattr(idx.Scanner, "scan_config", dedented)
    assert any(problem.startswith("UNDER") for problem, _ in failures_of(ini_cases(random.Random(5), 300), "configs/c.ini"))

"""scripts/workspace_index.py: a LOCAL, informational navigation index of docs/, specs/ and reference/.

The index must err toward OVER-protection (a file that something depends on must never read as safely unused)
and must show its uncertainty. Two reviewers (astra, sol) required these properties; each test below pins one:
policy files (``.gitignore``, the commit-policy script) name the trees without consuming them; the index never
manufactures references from itself; constructions with an unresolved tail attach to the narrowest subtree they
could name; a bare ``"specs"`` literal means the whole tree (``experiment.py`` hashes every spec file); an
import by module stem is ambiguous when two files share the stem; skipped input is counted, never hidden.

Fixtures are INVENTED: nothing from the real private trees is copied here (docs/, specs/ and reference/ are
local-only by policy). The repository is a throwaway ``git init`` with ``git add`` and no commit.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "workspace_index.py"


def load():
    spec = importlib.util.spec_from_file_location("workspace_index_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workspace_index_under_test"] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                   # importing the script must not leave a .pyc in scripts/
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


idx = load()
TODAY = "2026-10-10"


def make_repo(root, files, *, tracked=(), crlf=()):
    """Write ``files`` ({relative path: text}) under ``root``; ``git add`` the ``tracked`` paths (no commit)."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = text.replace("\n", "\r\n") if relative in crlf else text
        path.write_bytes(data.encode("utf-8"))
    if tracked:
        subprocess.run(["git", "add", "--", *tracked], cwd=root, check=True, capture_output=True)
    return root


def tags(model, path):
    return model.targets[path].tags


def cli(repo, *args):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(repo), "--today", TODAY, *args],
                          capture_output=True, text=True)


BASE = {"docs/A.md": "# Alpha\n", "docs/B.md": "# Beta\n"}


# ---------------------------------------------------------------------------------------- reference forms

def test_a_literal_path_in_tracked_code_is_a_path_reference(tmp_path):
    repo = make_repo(tmp_path / "r", {**BASE, "src/a.py": "open('docs/A.md')\n"}, tracked=["src/a.py"])
    model = idx.build_model(repo)
    assert idx.T_PATH in tags(model, "docs/A.md")
    assert idx.T_PATH not in tags(model, "docs/B.md")
    evidence = model.targets["docs/A.md"].evidence["code"][0]
    assert (evidence.source, evidence.line, evidence.form) == ("src/a.py", 1, "path")


@pytest.mark.parametrize("text", ["p = 'docs\\\\A.md'\n", 'p = "docs\\A.md"\n', '{"k": "docs\\\\A.md"}\n',
                                  "p = './docs/A.md'\n", "p = '../../docs/A.md'\n", "see C:/work/repo/docs/A.md here\n"])
def test_backslash_json_escaped_and_prefixed_paths_are_recognised(tmp_path, text):
    repo = make_repo(tmp_path / "r", {**BASE, "src/a.py": text}, tracked=["src/a.py"])
    assert idx.T_PATH in tags(idx.build_model(repo), "docs/A.md")


def test_a_path_ending_in_a_target_is_conservatively_a_reference_but_a_lookalike_directory_is_not(tmp_path):
    """Over-protection is the safe direction: 'other/docs/A.md' may be this repo seen from elsewhere, so it counts.
    'mydocs/A.md' names no tree root component, so it does not."""
    files = {**BASE, "src/a.py": "open('other/docs/A.md')\n", "src/b.py": "open('mydocs/B.md')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py", "src/b.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md")
    assert idx.T_PATH not in tags(model, "docs/B.md")


def test_a_literal_component_chain_is_a_path_reference(tmp_path):
    files = {"specs/sub/x.json": "{}", "specs/sub/y.json": "{}",
             "src/a.py": "p = ROOT / 'specs' / 'sub' / 'x.json'\nq = Path('specs', 'sub', 'y.json')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    for target in ("specs/sub/x.json", "specs/sub/y.json"):
        assert idx.T_PATH in tags(model, target), target
        assert model.targets[target].evidence["code"][0].form == "chain"


def test_a_chain_with_an_unresolved_tail_marks_only_the_subtree_it_could_name(tmp_path):
    files = {"specs/benchmarks/a.json": "{}", "specs/benchmarks/b.json": "{}", "specs/other/z.md": "# z\n",
             "src/t.py": "p = repository_root / 'specs' / 'benchmarks' / f'{stem}.json'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "specs/benchmarks/a.json")
    assert idx.T_UNRESOLVED in tags(model, "specs/benchmarks/b.json")
    assert idx.T_UNRESOLVED not in tags(model, "specs/other/z.md")
    assert [c.source for c in model.constructions] == ["src/t.py"]
    assert model.constructions[0].prefix == "specs/benchmarks"


def test_a_bare_tree_root_literal_in_code_means_the_whole_tree(tmp_path):
    files = {**BASE, "specs/s.md": "# s\n", "src/exp.py": "CONTRACT_ROOTS = ('specs',)\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/exp.py"]))
    assert idx.T_SUBTREE in tags(model, "specs/s.md")
    assert idx.T_SUBTREE not in tags(model, "docs/A.md")
    assert model.targets["specs/s.md"].evidence["code"][0].form == "bare-root"


def test_a_bare_root_word_in_prose_is_not_a_reference(tmp_path):
    files = {**BASE, "README.md": 'Open "docs" and read it.\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["README.md"]))
    assert idx.T_SUBTREE not in tags(model, "docs/A.md")


def test_a_directory_literal_marks_every_file_below_it(tmp_path):
    files = {"reference/evidence/x/e1.json": "{}", "reference/evidence/y/e2.json": "{}", "reference/n.md": "# n\n",
             "configs/c.json": '{"root": "reference/evidence"}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.json"]))
    assert idx.T_SUBTREE in tags(model, "reference/evidence/x/e1.json")
    assert idx.T_SUBTREE in tags(model, "reference/evidence/y/e2.json")
    assert idx.T_SUBTREE not in tags(model, "reference/n.md")


def test_a_template_with_a_placeholder_marks_the_directory_before_it(tmp_path):
    files = {"reference/pub/one.json": "{}", "reference/other/two.json": "{}",
             "src/t.py": "path = 'reference/pub/{name}.json'\nglob = 'reference/other/*.json'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "reference/pub/one.json")
    assert idx.T_UNRESOLVED in tags(model, "reference/other/two.json")


def test_a_chain_rooted_at_a_variable_still_reaches_the_directories_it_names(tmp_path):
    files = {"specs/benchmarks/a.json": "{}", "specs/other/b.json": "{}",
             "src/v.py": "p = base / 'benchmarks' / 'a.json'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/v.py"]))
    assert idx.T_UNRESOLVED in tags(model, "specs/benchmarks/a.json")
    assert idx.T_UNRESOLVED not in tags(model, "specs/other/b.json")


def test_a_basename_mention_is_weak_evidence_and_says_so(tmp_path):
    files = {"specs/x/FLIGHT_SPEC.md": "# spec\n", "src/n.py": "# see FLIGHT_SPEC.md for the rules\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/n.py"]))
    assert tags(model, "specs/x/FLIGHT_SPEC.md") >= {idx.T_NAME}
    assert idx.T_PATH not in tags(model, "specs/x/FLIGHT_SPEC.md")


def test_an_import_by_stem_marks_every_file_sharing_that_stem(tmp_path):
    files = {"specs/a/reference_math.py": "X = 1\n", "specs/b/reference_math.py": "X = 2\n",
             "src/test_i.py": "import reference_math\nfrom reference_math import X\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/test_i.py"]))
    assert idx.T_IMPORT in tags(model, "specs/a/reference_math.py")
    assert idx.T_IMPORT in tags(model, "specs/b/reference_math.py")


# ---------------------------------------------------------------------------------------- source categories

def test_policy_files_name_the_trees_but_are_not_consumer_evidence(tmp_path):
    """Root-only policy lines ('docs/') name a whole tree: they say nothing about one file, so they tag nothing.
    A SPECIFIC policy pattern does tag what it names, as a policy mention that never counts as a reference."""
    files = {**BASE, "specs/benchmarks/a.json": "{}", "specs/other.md": "# o\n",
             ".gitignore": "docs/\nreference/\nspecs/*\n!specs/benchmarks/*.json\n",
             "scripts/check_local_only_files.py": "PATTERNS = ['docs/', 'specs/', 'reference/']\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[".gitignore", "scripts/check_local_only_files.py"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}
    assert tags(model, "specs/other.md") == {idx.T_NONE}
    assert tags(model, "specs/benchmarks/a.json") == {idx.T_POLICY, idx.T_NONE}
    assert any("gitignore" in line for line in idx.render(model, TODAY).splitlines() if "whole" in line.lower())


def test_local_notes_loop_records_and_output_records_are_separate_categories(tmp_path):
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "docs/C.md": "# c\n", "docs/D.md": "# d\n",
             "docs/hub.md": "see docs/A.md\n",
             ".claude-codex-loop/runs/r1/launch.json": '{"f": "docs/B.md"}\n',
             "outputs/camp/audit_spec_frozen.json": '{"evidence": "docs/C.md"}\n',
             "outputs/camp/stages/freeze.json": '{"evidence": "docs/D.md"}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert tags(model, "docs/A.md") >= {idx.T_NOTES} and idx.T_NONE not in tags(model, "docs/A.md")
    assert tags(model, "docs/B.md") >= {idx.T_LOOP}
    assert tags(model, "docs/C.md") >= {idx.T_OUTPUTS}
    assert idx.T_OUTPUTS not in tags(model, "docs/D.md")                # nested below the top level: not scanned
    assert idx.T_NONE in tags(model, "docs/D.md")                       # ... and the limit is reported, not hidden
    assert "top level" in idx.render(model, TODAY)


def test_a_reference_found_nowhere_says_no_reference_found_not_unreferenced(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "src/a.py": "x = 1\n"}, tracked=["src/a.py"]))
    assert tags(model, "docs/B.md") == {idx.T_NONE}
    text = idx.render(model, TODAY)
    assert "NO-REFERENCE-FOUND-IN-SCANNED-SOURCES" in text and "unreferenced" not in text.lower()


def test_protected_files_are_protected_whatever_the_scan_finds(tmp_path):
    files = {"docs/ARCHITECTURE.md": "# arch\n", "reference/evidence/a/x.json": "{}", "docs/other.md": "# o\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert any(t.startswith("PROTECTED:") for t in tags(model, "docs/ARCHITECTURE.md"))
    assert any(t.startswith("PROTECTED:") for t in tags(model, "reference/evidence/a/x.json"))
    assert not any(t.startswith("PROTECTED:") for t in tags(model, "docs/other.md"))


def test_a_file_carrying_the_generated_marker_is_never_a_source(tmp_path):
    files = {**BASE, "docs/old_index.md": idx.MARKER + "\n- docs/A.md\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert idx.T_NOTES not in tags(model, "docs/A.md")
    assert model.gaps["generated_marker_files"] == 1


def test_a_second_run_over_its_own_output_is_identical(tmp_path):
    repo = make_repo(tmp_path / "r", {**BASE, "src/a.py": "open('docs/A.md')\n"}, tracked=["src/a.py"])
    first = cli(repo)
    assert first.returncode == 0, first.stderr
    written = (repo / "docs" / "INDEX.md").read_bytes()
    second = cli(repo)
    assert second.returncode == 0, second.stderr
    assert (repo / "docs" / "INDEX.md").read_bytes() == written
    assert b"\r" not in written and written.startswith(idx.MARKER.encode("utf-8"))


def test_the_index_is_not_listed_as_its_own_target_or_source(tmp_path):
    repo = make_repo(tmp_path / "r", dict(BASE))
    cli(repo)
    model = idx.build_model(repo)
    assert "docs/INDEX.md" not in model.targets
    assert idx.T_NOTES not in tags(model, "docs/A.md")


# ---------------------------------------------------------------------------------------- gaps and safety

def test_skipped_input_is_counted_not_hidden(tmp_path):
    big = "x" * (idx.SIZE_CAP + 10)
    files = {**BASE, "src/big.json": big, "src/blob.pt": "binary-ish", "src/bad.py": "x = 1\n",
             ".claude-codex-loop/runs/r1/a/b/c/deep.py": "open('docs/B.md')\n",
             ".claude-codex-loop/runs/r1/stream.jsonl": "docs/B.md\n"}
    repo = make_repo(tmp_path / "r", files, tracked=["src/big.json", "src/blob.pt", "src/bad.py"])
    (repo / "src" / "bad.py").write_bytes(b"x = '\xff\xfe docs/A.md'\n")
    model = idx.build_model(repo)
    gaps = model.gaps
    assert gaps["tracked_skipped_size"] == 1 and gaps["tracked_skipped_binary"] == 1
    assert gaps["decode_replaced"] >= 1
    assert gaps["loop_skipped_deep_extension"] >= 1 and gaps["loop_skipped_jsonl"] == 1
    assert idx.T_LOOP not in tags(model, "docs/B.md")
    assert idx.T_PATH in tags(model, "docs/A.md")                       # the undecodable file still yielded its reference
    rendered = idx.render(model, TODAY)
    for needle in ("skipped", "size cap"):
        assert needle in rendered.lower()


def test_the_loop_scan_can_be_disabled_and_the_index_says_so(tmp_path):
    files = {**BASE, ".claude-codex-loop/runs/r1/launch.json": '{"f": "docs/A.md"}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files), scan_loop=False)
    assert idx.T_LOOP not in tags(model, "docs/A.md")
    assert model.gaps["loop_disabled"] is True and "loop" in idx.render(model, TODAY).lower()


def test_a_missing_tree_is_noted_and_does_not_fail(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", dict(BASE)))
    assert model.gaps["missing_trees"] == ["reference", "specs"]


def test_crlf_files_are_handled_and_titles_are_clean(tmp_path):
    files = {"docs/A.md": "# Alpha title\nbody\n", "src/a.py": "open('docs/A.md')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"], crlf=("docs/A.md", "src/a.py")))
    assert model.targets["docs/A.md"].title == "Alpha title"
    assert idx.T_PATH in tags(model, "docs/A.md")


def test_titles_come_from_the_first_heading_else_the_first_line(tmp_path):
    files = {"docs/H.md": "\n\n## Heading two\ntext\n", "docs/P.md": "plain first line\n", "docs/E.md": "",
             "specs/s.json": '{"k": 1}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert model.targets["docs/H.md"].title == "Heading two"
    assert model.targets["docs/P.md"].title == "plain first line"
    assert model.targets["docs/E.md"].title == ""


def test_a_junction_inside_a_tree_is_recorded_and_never_followed(tmp_path):
    repo = make_repo(tmp_path / "r", dict(BASE))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# secret\n")
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(repo / "docs" / "jl"), str(outside)], capture_output=True, text=True)
    if done.returncode:
        pytest.skip("cannot create a junction here")
    try:
        model = idx.build_model(repo)
        assert not any(p.startswith("docs/jl") for p in model.targets)
        assert "docs/jl" in model.gaps["links_skipped"]
    finally:
        os.rmdir(repo / "docs" / "jl")


def test_hub_documents_are_found_by_their_outgoing_links(tmp_path):
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "docs/C.md": "# c\n", "docs/D.md": "# d\n",
             "docs/hub.md": "- docs/A.md\n- docs/B.md\n- docs/C.md\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert model.hubs[0][0] == "docs/hub.md" and model.hubs[0][1] == 3


def test_tracked_files_inside_the_trees_are_marked_tracked(tmp_path):
    files = {"specs/benchmarks/a.json": "{}", "specs/local.md": "# l\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["specs/benchmarks/a.json"]))
    assert model.targets["specs/benchmarks/a.json"].tracked is True
    assert model.targets["specs/local.md"].tracked is False


# ---------------------------------------------------------------------------------------- the CLI

def test_the_cli_writes_only_the_index_and_no_bytecode(tmp_path):
    repo = make_repo(tmp_path / "r", {**BASE, "src/a.py": "open('docs/A.md')\n"}, tracked=["src/a.py"])
    before = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts)
    pyc_before = set(SCRIPT.parent.rglob("__pycache__/workspace_index*"))
    done = cli(repo)
    assert done.returncode == 0, done.stderr
    after = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts)
    assert sorted(set(after) - set(before)) == [str(Path("docs") / "INDEX.md")]
    assert set(SCRIPT.parent.rglob("__pycache__/workspace_index*")) == pyc_before     # the run wrote no bytecode


def test_check_prints_and_writes_nothing(tmp_path):
    repo = make_repo(tmp_path / "r", dict(BASE))
    done = cli(repo, "--check")
    assert done.returncode == 0 and "docs/A.md" in done.stdout
    assert not (repo / "docs" / "INDEX.md").exists()


def test_a_foreign_index_file_is_never_overwritten(tmp_path):
    repo = make_repo(tmp_path / "r", {**BASE, "docs/INDEX.md": "# my own hand-written index\n"})
    done = cli(repo)
    assert done.returncode == 2 and "refus" in done.stderr.lower()
    assert (repo / "docs" / "INDEX.md").read_text() == "# my own hand-written index\n"


def test_outside_a_git_repository_it_fails_instead_of_scanning_nothing(tmp_path):
    plain = tmp_path / "plain"
    (plain / "docs").mkdir(parents=True)
    (plain / "docs" / "A.md").write_text("# a\n")
    done = cli(plain)
    assert done.returncode != 0 and "git" in done.stderr.lower()
    assert not (plain / "docs" / "INDEX.md").exists()


def test_output_is_deterministic_and_navigation_comes_first(tmp_path):
    files = {**BASE, "docs/CODE_MAP.md": "# Code map\n- docs/A.md\n- docs/B.md\n- docs/ARCHITECTURE.md\n",
             "docs/ARCHITECTURE.md": "# Architecture\n", "src/a.py": "open('docs/A.md')\n"}
    repo = make_repo(tmp_path / "r", files, tracked=["src/a.py"])
    model = idx.build_model(repo)
    one, two = idx.render(model, TODAY), idx.render(idx.build_model(repo), TODAY)
    assert one == two
    assert one.index("Start here") < one.index("Inventory")
    assert "[CODE_MAP.md](CODE_MAP.md)" in one
    assert one.count("\n") > 10 and "\r" not in one


# ---------------------------------------------------------------------------------------- found on the real workspace
# The first run against the real trees showed what the invented fixtures above did not: every file blanketed by one
# prose mention of "docs/", hub documents that "link" every file, fixture paths flagging real trees, a 269 s scan.

@pytest.mark.parametrize("category_files", [
    {"docs/hub.md": "keep everything in docs/ tidy\n"},
    {".claude-codex-loop/runs/r1/launch.json": '{"note": "docs/ was reviewed"}\n'},
    {"outputs/camp/review.json": '{"quote": "the docs/ tree"}\n'},
    {"README.md": "Open docs/ or `specs/` first.\n"},
    # the same statements as GLOBS: a template whose prefix is a whole tree is a whole-tree statement too
    {"docs/hub.md": "every docs/*.md file, and specs/* besides\n"},
    {".claude-codex-loop/runs/r1/launch.json": '{"glob": "docs/*", "also": "specs/{name}.md"}\n'},
    {"outputs/camp/review.json": '{"glob": "docs/**", "also": "specs/%s.md"}\n'},
    {"README.md": "Files match docs/*.md and specs/*.\n"},
])
def test_a_bare_tree_root_mention_in_prose_or_records_tags_no_file(tmp_path, category_files):
    files = {**BASE, "specs/spec_one.md": "# s\n", **category_files}
    tracked = [p for p in files if p == "README.md"]
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=tracked))
    assert tags(model, "docs/A.md") == {idx.T_NONE}
    assert tags(model, "specs/spec_one.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ["ROOTS = ('docs/', 'x')\n", "ROOTS = ('./docs', 'x')\n", 'ROOTS = ["docs"]\n'])
def test_a_quoted_tree_root_in_code_is_a_whole_tree_reference_whatever_its_slashes(tmp_path, text):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "src/a.py": text}, tracked=["src/a.py"]))
    assert idx.T_SUBTREE in tags(model, "docs/A.md") and idx.T_SUBTREE in tags(model, "docs/B.md")


@pytest.mark.parametrize("text", ['{"root": "reference"}\n', "{'root': 'reference/'}\n", 'root = "reference"\n'])
def test_a_quoted_tree_root_in_a_config_file_is_a_whole_tree_reference(tmp_path, text):
    suffix = ".toml" if text.startswith("root") else ".json"
    files = {"reference/n.md": "# n\n", "specs/s.md": "# s\n", f"configs/c{suffix}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"configs/c{suffix}"]))
    assert idx.T_SUBTREE in tags(model, "reference/n.md")
    assert idx.T_SUBTREE not in tags(model, "specs/s.md")


def test_a_longer_word_that_merely_contains_a_tree_name_is_not_a_reference(tmp_path):
    files = {**BASE, "src/a.py": "x = 'docs_dir'\ny = 'mydocs'\nz = '--docs'\n",
             "configs/c.json": '{"specs_dir": "x", "reference_model": "y"}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py", "configs/c.json"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_fully_literal_chain_to_a_file_that_does_not_exist_flags_nothing(tmp_path):
    """Tests build fixture repos: repo / 'specs' / 'contract.md' names a path that is not in THIS tree, so no
    existing file can be the one meant. Flagging the whole directory for it would bury every real signal."""
    files = {"specs/real.md": "# r\n", "src/t.py": "(repo / 'specs' / 'contract.md').write_text('x')\n"
                                                  "p = repo / 'specs' / 'decisions' / '0001-x.md'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert tags(model, "specs/real.md") == {idx.T_NONE}
    assert model.constructions == []


def test_an_unresolved_tail_below_a_directory_that_does_not_exist_flags_nothing(tmp_path):
    files = {"reference/pub/real.json": "{}", "src/t.py": "path = repo / 'reference' / 'pub' / 'camp' / name\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert tags(model, "reference/pub/real.json") == {idx.T_NONE}


def test_a_template_tail_flags_only_names_that_start_with_its_literal_head(tmp_path):
    files = {"reference/endpoint_update3.pt": "x", "reference/other.json": "{}",
             "src/t.py": "w = load(root / 'reference' / f'endpoint_update{n}.pt')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "reference/endpoint_update3.pt")
    assert tags(model, "reference/other.json") == {idx.T_NONE}
    assert [(c.source, c.prefix) for c in model.constructions] == [("src/t.py", "reference")]


def test_a_variable_tail_flags_everything_below_the_prefix(tmp_path):
    files = {"reference/a.json": "{}", "reference/sub/b.json": "{}", "docs/A.md": "# a\n",
             "src/t.py": "before = load(root / 'reference' / state_name)\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "reference/a.json") and idx.T_UNRESOLVED in tags(model, "reference/sub/b.json")
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_single_literal_with_slashes_after_a_variable_root_is_a_suffix_chain(tmp_path):
    files = {"specs/benchmarks/a.json": "{}", "specs/other/a.json": "{}", "src/v.py": "p = base / 'benchmarks/a.json'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/v.py"]))
    assert idx.T_UNRESOLVED in tags(model, "specs/benchmarks/a.json")
    assert idx.T_UNRESOLVED not in tags(model, "specs/other/a.json")


def test_os_path_join_with_a_leading_variable_still_reads_as_a_chain(tmp_path):
    files = {"specs/sub/x.json": "{}", "specs/sub/y.json": "{}",
             "src/v.py": "p = os.path.join(ROOT, 'specs', 'sub', 'x.json')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/v.py"]))
    assert idx.T_PATH in tags(model, "specs/sub/x.json") and idx.T_PATH not in tags(model, "specs/sub/y.json")


def test_an_absolute_prefix_in_one_literal_still_reaches_the_tree_chain(tmp_path):
    files = {**BASE, "src/a.py": "p = Path('C:/work/repo/docs') / 'A.md'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md") and idx.T_PATH not in tags(model, "docs/B.md")


def test_evidence_lines_are_exact_even_after_json_escaped_backslashes(tmp_path):
    """Collapsing '\\\\' to '/' before looking offsets up in the ORIGINAL text shifts every later match by one
    character per escape: with enough escapes above it, the reference would be reported on an earlier line."""
    escaped = "\\\\x" * 60
    text = f'{{\n  "a": "C:{escaped}",\n  "b": "C:{escaped}",\n  "c": "docs\\\\A.md"\n}}\n'
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "configs/c.json": text}, tracked=["configs/c.json"]))
    assert {e.line for e in model.targets["docs/A.md"].evidence["code"]} == {4}


def test_names_containing_spaces_are_still_found(tmp_path):
    files = {"docs/My Notes.md": "# notes\n", "src/a.py": "# see My Notes.md for details\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_NAME in tags(model, "docs/My Notes.md")


def test_hubs_count_only_resolved_links_from_markdown(tmp_path):
    """A prose mention of 'docs/' is not a link to every file, and a JSON manifest is not a navigation document."""
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "docs/C.md": "# c\n", "docs/D.md": "# d\n",
             "docs/prose.md": "everything in docs/ and specs/ is local\n",
             "docs/real.md": "- A.md\n- docs/B.md\n- docs/C.md\n",
             "docs/manifest.json": '["docs/A.md", "docs/B.md", "docs/C.md", "docs/D.md"]\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert [path for path, _ in model.hubs] == ["docs/real.md"] and model.hubs[0][1] == 3


def test_paths_are_matched_case_insensitively_as_the_windows_filesystem_resolves_them(tmp_path):
    """Measured on the real workspace: 'docs/architecture.md' names the real docs/ARCHITECTURE.md. A case-sensitive
    match would report a file as unreferenced that the filesystem resolves without complaint."""
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "docs/C.md": "# c\n", "reference/evidence/x/e1.json": "{}",
             "reference/endpoint_1.pt": "x", "specs/a/reference_math.py": "X = 1\n",
             "src/a.py": ("open('Docs/A.MD')\np = Path('DOCS') / 'b.MD'\nd = 'Reference/Evidence'\n"
                          "w = root / 'Reference' / f'Endpoint_{n}.pt'\nimport Reference_Math\n")}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md")
    assert idx.T_PATH in tags(model, "docs/B.md")
    assert idx.T_SUBTREE in tags(model, "reference/evidence/x/e1.json")
    assert idx.T_UNRESOLVED in tags(model, "reference/endpoint_1.pt")
    assert idx.T_IMPORT in tags(model, "specs/a/reference_math.py")
    assert tags(model, "docs/C.md") == {idx.T_NONE}


def test_a_name_mention_in_other_case_still_counts(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", {"docs/ARCHITECTURE.md": "# a\n", "docs/note.md": "see architecture.md\n"}))
    assert idx.T_NOTES in tags(model, "docs/ARCHITECTURE.md")


@pytest.mark.parametrize("name, text", [("c.yaml", "root: reference\n"), ("c.yml", "paths:\n  data: ./reference/\n"),
                                        ("c.yaml", "roots:\n  - specs\n  - reference  # both\n"), ("c.toml", "root = reference\n"),
                                        ("c.ini", "[paths]\nroot = reference\n"), ("c.cfg", "root: reference\n")])
def test_an_unquoted_tree_value_in_a_yaml_or_ini_config_is_a_whole_tree_reference(tmp_path, name, text):
    files = {"reference/n.md": "# n\n", "docs/A.md": "# a\n", f"configs/{name}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"configs/{name}"]))
    assert idx.T_SUBTREE in tags(model, "reference/n.md")
    assert tags(model, "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ["# root: reference\n", "note: the reference folder\n", "name: reference_model\n",
                                  "url: https://example.org/docs\n", "ref = referenced\n"])
def test_a_comment_or_a_longer_value_in_a_config_is_not_a_tree_reference(tmp_path, text):
    files = {"reference/n.md": "# n\n", "configs/c.yaml": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.yaml"]))
    assert idx.T_SUBTREE not in tags(model, "reference/n.md")


# ---------------------------------------------------------------------------------------- found by sol's checkpoint review

def test_a_concatenated_variable_tail_marks_the_names_that_could_start_with_the_literal_head(tmp_path):
    files = {"docs/benchmarks/result_a.json": "{}", "docs/benchmarks/other.json": "{}", "docs/A.md": "# a\n",
             "src/t.py": "p = root / 'docs/benchmarks/result_' + name + '.json'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "docs/benchmarks/result_a.json")
    assert idx.T_UNRESOLVED not in tags(model, "docs/benchmarks/other.json") and tags(model, "docs/A.md") == {idx.T_NONE}
    assert [c.prefix for c in model.constructions] == ["docs/benchmarks"] and model.constructions[0].head == "result_"


def test_a_directory_literal_followed_by_concatenation_marks_everything_below_it(tmp_path):
    files = {"docs/benchmarks/a.json": "{}", "docs/benchmarks/b.json": "{}", "docs/A.md": "# a\n",
             "src/t.py": "d = 'docs/benchmarks/' + name\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "docs/benchmarks/a.json") and idx.T_UNRESOLVED in tags(model, "docs/benchmarks/b.json")
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_tree_root_literal_followed_by_concatenation_marks_the_whole_tree(tmp_path):
    files = {"specs/a.json": "{}", "docs/A.md": "# a\n", "src/t.py": "base = 'specs' + os.sep + name\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "specs/a.json") and tags(model, "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [("c.yaml", "roots: [docs, benchmarks]\n"), ("c.yml", "paths: {root: docs, other: x}\n"),
                                        ("c.ini", "roots = docs, specs\n"), ("c.yaml", "roots: [ ./docs/ , x ]\n"),
                                        ("c.cfg", "roots = specs,docs\n")])
def test_a_tree_inside_a_flow_style_list_or_mapping_is_a_whole_tree_reference(tmp_path, name, text):
    files = {"docs/A.md": "# a\n", "reference/n.md": "# n\n", f"configs/{name}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"configs/{name}"]))
    assert idx.T_SUBTREE in tags(model, "docs/A.md")
    assert idx.T_SUBTREE not in tags(model, "reference/n.md")


@pytest.mark.parametrize("text", ["description: the docs folder\n", "# roots: [docs, specs]\n", "link: [site](docs-site)\n",
                                  "roots: [documents, spec]\n", "note: docs-only change, see below\n"])
def test_flow_style_look_alikes_and_comments_are_not_tree_references(tmp_path, text):
    model = idx.build_model(make_repo(tmp_path / "r", {"docs/A.md": "# a\n", "configs/c.yaml": text}, tracked=["configs/c.yaml"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_notebook_code_cell_is_read_as_code(tmp_path):
    import json
    cells = [{"cell_type": "code", "source": ["from pathlib import Path\n", "root = Path('specs')\n", "import reference_math\n",
                                              "p = Path(\"docs\") / \"A.md\"\n"]}]
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "specs/a/reference_math.py": "X = 1\n", "specs/s.json": "{}",
             "notebooks/n.ipynb": json.dumps({"cells": cells, "nbformat": 4}, indent=1)}     # Jupyter's own layout, \" escapes
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert idx.T_SUBTREE in tags(model, "specs/s.json")                               # Path('specs')
    assert idx.T_IMPORT in tags(model, "specs/a/reference_math.py")                   # import by stem inside a cell
    assert idx.T_PATH in tags(model, "docs/A.md") and idx.T_PATH not in tags(model, "docs/B.md")   # Path("docs") / "A.md"


# ---------------------------------------------------------------------------------------- found by sol's review round 1

def test_a_literal_whose_path_contains_a_space_is_read_from_the_parsed_literal(tmp_path):
    """The path regex cannot consume a space, so for such a literal the chain scan must not assume it was already read."""
    files = {"docs/my reports/result_1.json": "{}", "docs/my reports/other.json": "{}", "docs/A.md": "# a\n",
             "src/t.py": "open(f'docs/my reports/result_{n}.json')\nopen('docs/my reports/other.json')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "docs/my reports/result_1.json")
    assert idx.T_PATH in tags(model, "docs/my reports/other.json")
    assert idx.T_UNRESOLVED not in tags(model, "docs/my reports/other.json") and tags(model, "docs/A.md") == {idx.T_NONE}


def test_whitespace_only_inputs_stay_linear(tmp_path):
    """IMPORT's cross-line \\s* and CONFIG_BARE's overlapping repeats, and a title line of blanks, were quadratic."""
    # "# a" + blanks + "x": the lazy title group retries `\s*#*\s*$` at every blank, each time rescanning the blank run
    files = {**BASE, "docs/H.md": "# a" + " " * 400_000 + "x\n", "src/nl.py": "\n" * 800_000,
             "configs/c.yaml": " " * 400_000 + "\n"}
    repo = make_repo(tmp_path / "r", files, tracked=["src/nl.py", "configs/c.yaml"])
    done = subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(repo), "--today", TODAY, "--check", "--no-loop"],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr[-300:]


def _notebook(*cells):
    import json
    return json.dumps({"cells": list(cells), "nbformat": 4}, indent=1)


def test_notebook_evidence_uses_cell_numbers_and_prose_cells_are_not_code(tmp_path):
    notebook = _notebook(
        {"cell_type": "markdown", "source": ["Notes about docs/*.md and specs/* and docs/ in general.\n"]},
        {"cell_type": "code", "source": ["import os\n", "open('docs/A.md')\n"],
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["docs/C.md\n"]}]},
        {"cell_type": "code", "source": ["import glob\n", "glob.glob('specs/*.json')\n"], "outputs": []})
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "docs/C.md": "# c\n", "specs/s.json": "{}", "notebooks/n.ipynb": notebook}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert {e.line for e in model.targets["docs/A.md"].evidence["code"]} == {2}        # path AND name: the cell, not the JSON line
    assert idx.T_PATH in tags(model, "docs/C.md") and {e.line for e in model.targets["docs/C.md"].evidence["code"]} == {2}
    assert tags(model, "docs/B.md") == {idx.T_NONE}                                    # a prose cell naming the tree is not code
    assert idx.T_UNRESOLVED in tags(model, "specs/s.json")                              # a code cell's root glob is
    assert {e.line for e in model.targets["specs/s.json"].evidence["code"]} == {3}


def _run_check(repo, timeout):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(repo), "--today", TODAY, "--check", "--no-loop"],
                          capture_output=True, text=True, timeout=timeout)


def test_many_constructions_on_one_long_line_stay_linear(tmp_path):
    """Each construction used to strip() and copy the WHOLE source line for its excerpt."""
    line = " " * 2_000_000 + "".join(f"x = 'docs/p{i:06d}' + name; " for i in range(25_000))
    repo = make_repo(tmp_path / "r", {**BASE, "src/many.py": line + "\n"}, tracked=["src/many.py"])
    done = _run_check(repo, timeout=20)
    assert done.returncode == 0, done.stderr[-300:]
    assert "docs/p000001" in done.stdout                                    # the constructions are still listed


def test_a_source_naming_many_files_is_scanned_once_not_once_per_file(tmp_path):
    """One regex search over the whole source for every matched file name was O(size x names)."""
    names = [f"d{i:04d}.md" for i in range(1500)]
    files = {**{f"docs/{name}": "# d\n" for name in names}, "src/mentions.txt": " " * 2_000_000 + " ".join(names) + "\n"}
    repo = make_repo(tmp_path / "r", files, tracked=["src/mentions.txt"])
    done = _run_check(repo, timeout=20)
    assert done.returncode == 0, done.stderr[-300:]
    assert "src/mentions.txt:1 (name)" in done.stdout


# ---------------------------------------------------------------------------------------- found by sol's review round 3

def synthetic_scanner(paths):
    """A Scanner over an in-memory inventory: no files are created, so inventory-size effects can be measured."""
    model = idx.Model()
    for path in paths:
        model.targets[path] = idx.Target(path, False, "", 0, 0)
    model.gaps = idx.new_gaps()
    return model, idx.Scanner(model, list(model.targets))


def test_many_distinct_heads_against_a_large_inventory_stay_linear():
    """Every unmatched head used to scan every file below the prefix: heads x files."""
    import time
    model, scanner = synthetic_scanner(f"docs/q{i:06d}.md" for i in range(12_000))
    text = "".join(f"p = 'docs/p{i:06d}' + name\n" for i in range(12_000))
    started = time.perf_counter()
    scanner.scan("code", "src/t.py", text)
    assert time.perf_counter() - started < 8
    assert len(model.constructions) == idx.MAX_CONSTRUCTIONS and all(c.matched == 0 for c in model.constructions)
    assert model.gaps["constructions_unlisted"] == 12_000 - idx.MAX_CONSTRUCTIONS


def test_the_listing_of_constructions_is_capped_but_the_marking_is_not():
    count = idx.MAX_CONSTRUCTIONS + 500
    model, scanner = synthetic_scanner(["docs/q1.md", "docs/z9.md"])
    scanner.scan("code", "src/t.py", "".join(f"p = 'docs/q{i}' + name\n" for i in range(count)))
    assert len(model.constructions) == idx.MAX_CONSTRUCTIONS and model.gaps["constructions_unlisted"] == 500
    assert idx.T_UNRESOLVED in model.targets["docs/q1.md"].tags                      # the file is marked all the same
    assert "500 more constructions" in idx.render(model, TODAY)


def test_distinct_sources_are_counted_not_stored_per_file():
    """The page says '+N more' sources: a count, taken once per source however many times that source names the file."""
    model, scanner = synthetic_scanner(["docs/A.md", "docs/B.md"])
    scanner.scan("code", "src/one.py", "open('docs/A.md')\nopen('docs/A.md')\nx = 'docs/A.md'\n")
    scanner.scan("code", "src/two.py", "open('docs/A.md')\n")
    scanner.scan("notes", "docs/B.md", "see docs/A.md\n")
    target = model.targets["docs/A.md"]
    assert target.source_counts["code"] == 2 and target.source_counts["notes"] == 1 and target.named_count == 2
    assert "src/one.py:1 (path) +1 more" in idx.render(model, TODAY)


def test_a_head_still_matches_exactly_the_names_that_start_with_it():
    model, scanner = synthetic_scanner(["docs/q1.md", "docs/q10.md", "docs/Q2.md", "docs/r1.md", "docs/sub/q3.md"])
    scanner.scan("code", "src/t.py", "p = 'docs/q' + name\n")
    assert {p for p, t in model.targets.items() if idx.T_UNRESOLVED in t.tags} == {"docs/q1.md", "docs/q10.md", "docs/Q2.md"}


def test_odd_names_are_matched_by_token_sequence_not_by_one_search_each():
    import time
    model, scanner = synthetic_scanner([f"docs/odd name {i:04d}.md" for i in range(500)])
    # every word of every name is present and in sequence, but the comma breaks each complete name: only the located
    # candidates are confirmed against the text, and none confirms
    text = " " * 2_800_000 + "odd name, " + " ".join(f"{i:04d}.md" for i in range(500)) + "\n"
    started = time.perf_counter()
    scanner.scan("code", "src/t.txt", text)
    assert time.perf_counter() - started < 5
    assert all(idx.T_NAME not in t.tags for t in model.targets.values())       # no complete name occurs


def test_an_odd_name_followed_by_sentence_punctuation_is_still_found():
    model, scanner = synthetic_scanner(["docs/My Notes.md", "docs/Other Notes.md"])
    scanner.scan("code", "README.md", "intro\nSee My Notes.md.\n")
    assert idx.T_NAME in model.targets["docs/My Notes.md"].tags and idx.T_NAME not in model.targets["docs/Other Notes.md"].tags
    assert [e.line for e in model.targets["docs/My Notes.md"].evidence["code"]] == [2]


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 5
# Both reviewers independently reached most of these, each with a realistic tracked source that left a consumed file at `none`.

@pytest.mark.parametrize("encoding, mark, cjk", [("utf-16-le", "BOM_UTF16_LE", False), ("utf-16-be", "BOM_UTF16_BE", False),
                                                 ("utf-8", "BOM_UTF8", False), ("utf-32-le", "BOM_UTF32_LE", False),
                                                 ("utf-16-le", None, False), ("utf-16-be", None, False),
                                                 # mostly CJK: few NUL bytes, so only the BOM can say it is UTF-16
                                                 ("utf-16-le", "BOM_UTF16_LE", True), ("utf-16-be", "BOM_UTF16_BE", True),
                                                 ("utf-32-be", "BOM_UTF32_BE", False)])
def test_sources_in_other_encodings_are_decoded_before_scanning(tmp_path, encoding, mark, cjk):
    """Windows PowerShell writes UTF-16 by default; read as UTF-8 it is NULs between every character."""
    import codecs
    text = ("# " + "中" * 400 + "\r\n" if cjk else "") + 'Get-Content "reference/x.json"\r\n'
    repo = make_repo(tmp_path / "r", {"reference/x.json": "{}", "reference/y.json": "{}", "scripts/a.ps1": "placeholder"},
                     tracked=["scripts/a.ps1"])
    (repo / "scripts" / "a.ps1").write_bytes((getattr(codecs, mark) if mark else b"") + text.encode(encoding))
    model = idx.build_model(repo)
    assert idx.T_PATH in tags(model, "reference/x.json") and idx.T_PATH not in tags(model, "reference/y.json")


def test_text_that_stays_binary_after_decoding_is_counted_not_hidden(tmp_path):
    files = {**BASE, "src/blob.txt": "placeholder"}
    repo = make_repo(tmp_path / "r", files, tracked=["src/blob.txt"])
    (repo / "src" / "blob.txt").write_bytes(b"\x00\x01\x02\x03\x04" * 100 + b" docs/A.md ")      # 20% NUL: binary, not UTF-16
    model = idx.build_model(repo)
    assert model.gaps["nul_bytes_files"] == 1 and "nul" in idx.render(model, TODAY).lower()
    assert idx.T_PATH in tags(model, "docs/A.md")                                      # the readable part is still scanned


@pytest.mark.parametrize("text", ["import os, reference_math\n", "import reference.reference_math\n",
                                  "from reference import reference_math as rm\n", "import os, sys, reference_math as rm, json\n",
                                  "from . import reference_math\n", "    import reference_math  # indented\n"])
def test_every_module_named_by_an_import_statement_is_marked(tmp_path, text):
    files = {"reference/reference_math.py": "X = 1\n", "reference/other_mod.py": "X = 2\n", "src/a.py": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_IMPORT in tags(model, "reference/reference_math.py")
    assert idx.T_IMPORT not in tags(model, "reference/other_mod.py")


@pytest.mark.parametrize("tail", [b"\r" * 8 + b"Important handwritten text", b"\r\r", b"x", b" ", b"\r\rmore\n"])
def test_a_marker_line_that_continues_past_the_bounded_read_never_authorises_an_overwrite(tmp_path, tail):
    repo = make_repo(tmp_path / "r", dict(BASE))
    mine = idx.MARKER.encode("utf-8") + tail
    (repo / "docs" / "INDEX.md").write_bytes(mine)
    done = cli(repo)
    assert done.returncode == 2, done.stderr
    assert (repo / "docs" / "INDEX.md").read_bytes() == mine


def test_dot_and_parent_components_are_resolved_before_lookup(tmp_path):
    files = {"docs/reports/a.json": "{}", "reference/anchors/anchor.json": "{}", "reference/run1.json": "{}",
             "specs/s.json": "{}", "docs/other/z.md": "# z\n",
             "src/t.py": ("list(Path('docs/./reports').glob('*.json'))\nread(Path('docs/reports/../reports') / 'a.json')\n"
                          "load(Path('reference/anchors') / '..' / f'run{n}.json')\nopen('docs/../specs/s.json')\n")}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_SUBTREE in tags(model, "docs/reports/a.json") or idx.T_PATH in tags(model, "docs/reports/a.json")
    assert idx.T_UNRESOLVED in tags(model, "reference/run1.json")
    assert idx.T_PATH in tags(model, "specs/s.json")                                    # 'docs/..' leaves docs and names specs
    assert tags(model, "docs/other/z.md") == {idx.T_NONE}
    assert idx.T_UNRESOLVED not in tags(model, "reference/anchors/anchor.json")


def test_a_glob_character_class_is_an_unresolved_component(tmp_path):
    files = {"reference/run1.json": "{}", "reference/run2.json": "{}", "reference/other.json": "{}", "docs/docA.md": "# a\n",
             "docs/docB.md": "# b\n", "docs/other.md": "# o\n",
             "src/t.py": "a = glob.glob('reference/run[0-9].json')\nb = glob.glob(\"docs/doc[AB].md\")\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, "reference/run1.json") and idx.T_UNRESOLVED in tags(model, "reference/run2.json")
    assert idx.T_UNRESOLVED not in tags(model, "reference/other.json")
    assert idx.T_UNRESOLVED in tags(model, "docs/docA.md") and idx.T_UNRESOLVED in tags(model, "docs/docB.md")
    assert tags(model, "docs/other.md") == {idx.T_NONE}


def test_a_path_in_square_brackets_in_prose_is_still_found(tmp_path):
    files = {**BASE, "docs/note.md": "see [docs/A.md] and [**docs/B.md**](docs/B.md)\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert idx.T_NOTES in tags(model, "docs/A.md") and idx.T_NOTES in tags(model, "docs/B.md")


@pytest.mark.parametrize("code, wanted, form_tag", [
    ("a = Path('reference/flight_' 'manifest.json')\n", "reference/flight_manifest.json", "T_PATH"),
    ("a = Path('reference/run_' f'{n}.json')\n", "reference/run_1.json", "T_UNRESOLVED"),
    ('a = open("reference/re" "port.json")\n', "reference/report.json", "T_PATH"),
    ('a = open("reference/report_"\n         + name + ".json")\n', "reference/report_a.json", "T_UNRESOLVED"),
    ('a = open("reference/report_"  # trailing note\n         + name + ".json")\n', "reference/report_a.json", "T_UNRESOLVED"),
])
def test_python_string_composition_across_literals_and_lines_is_followed(tmp_path, code, wanted, form_tag):
    names = ["reference/flight_manifest.json", "reference/run_1.json", "reference/report.json", "reference/report_a.json",
             "reference/unrelated.json"]
    files = {**{n: "{}" for n in names}, "src/t.py": code}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert getattr(idx, form_tag) in tags(model, wanted)
    assert tags(model, "reference/unrelated.json") == {idx.T_NONE}


@pytest.mark.parametrize("suffix, name, glue", [(".py", "src/t.py", " "), (".py", "src/t.py", ""), (".sh", "scripts/t.sh", "")])
def test_a_very_long_run_of_adjacent_literals_is_merged_in_linear_time(suffix, name, glue):
    """Merging adjacent literals by repeated ``a + b`` copies an ever-growing string: 500,000 of them took 80 s."""
    import time
    model, scanner = synthetic_scanner(["docs/ab.md"])
    started = time.perf_counter()
    scanner.scan("code", name, "x = " + ("'ab'" + glue) * 300_000 + "\n")           # old: ~9 s each; linear: well under 1 s
    assert time.perf_counter() - started < 3.0


def test_adjacent_quoted_words_in_a_shell_script_are_separate_arguments(tmp_path):
    files = {"docs/ab.md": "# ab\n", "docs/a.md": "# a\n", "scripts/t.sh": 'echo "docs/a" "b.md"\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["scripts/t.sh"]))
    assert idx.T_UNRESOLVED not in tags(model, "docs/ab.md") and idx.T_PATH not in tags(model, "docs/ab.md")


@pytest.mark.parametrize("script, wanted, form_tag", [
    ('id=1; cat reference/"run${id}.json"\n', "reference/run1.json", "T_UNRESOLVED"),
    ('cat "reference/"run1.json\n', "reference/run1.json", "T_PATH"),
    ('Get-Content reference/"run$id.json"\n', "reference/run1.json", "T_UNRESOLVED"),
])
def test_an_unquoted_part_of_a_script_path_is_joined_to_its_adjacent_quoted_part(tmp_path, script, wanted, form_tag):
    suffix = ".ps1" if script.startswith("Get-") else ".sh"
    files = {"reference/run1.json": "{}", "reference/other.json": "{}", f"scripts/t{suffix}": script}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"scripts/t{suffix}"]))
    assert getattr(idx, form_tag) in tags(model, wanted)
    assert tags(model, "reference/other.json") == {idx.T_NONE}


@pytest.mark.parametrize("name, text, wanted", [
    ("c.json", '{"root": "docs/my reports", "x": 1}\n', "docs/my reports/a.json"),
    ("c.json", '{"file": "docs/R\\u00e9sum\\u00e9.md"}\n', "docs/Résumé.md"),
    ("c.yaml", 'contract_root: "reference/flight specs"\n', "reference/flight specs/contract.json"),
    ("c.yaml", "contract_root: 'reference/flight specs'\n", "reference/flight specs/contract.json"),
    ("c.toml", 'contract_root = "reference/flight specs"\n', "reference/flight specs/contract.json"),
    ("c.yaml", "contract_root: reference/flight specs\n", "reference/flight specs/contract.json"),
    ("c.ini", "contract_root = reference/flight specs\n", "reference/flight specs/contract.json"),
])
def test_complete_config_values_are_paths_even_with_spaces_and_escapes(tmp_path, name, text, wanted):
    files = {"docs/my reports/a.json": "{}", "docs/Résumé.md": "# r\n", "reference/flight specs/contract.json": "{}",
             "docs/A.md": "# a\n", f"configs/{name}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"configs/{name}"]))
    assert tags(model, wanted) & {idx.T_PATH, idx.T_SUBTREE}
    assert tags(model, "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [("c.yaml", "home: https://example.org/docs\n"), ("c.json", '{"u": "http://example.org/specs/"}\n'),
                                        ("c.yaml", 'home: "https://example.org/reference/"\n'), ("c.ini", "url = ftp://host/docs\n")])
def test_a_url_in_a_config_is_never_a_path(tmp_path, name, text):
    files = {**BASE, "specs/x/y.json": "{}", "reference/z.md": "# z\n", f"configs/{name}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"configs/{name}"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "reference/z.md") == {idx.T_NONE}
    assert tags(model, "specs/x/y.json") == {idx.T_NONE}


def test_a_glob_class_written_in_prose_marks_the_names_it_could_match(tmp_path):
    files = {"docs/docA.md": "# a\n", "docs/docB.md": "# b\n", "docs/other.md": "# o\n", "docs/note.md": "see docs/doc[AB].md for both\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert idx.T_NOTES in tags(model, "docs/docA.md") and idx.T_NOTES in tags(model, "docs/docB.md")
    assert tags(model, "docs/other.md") == {idx.T_NONE}


def test_a_config_sentence_that_merely_contains_a_tree_path_is_not_a_reference(tmp_path):
    files = {"reference/flight specs/contract.json": "{}", "configs/c.yaml": "note: see reference/flight specs for details\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.yaml"]))
    assert tags(model, "reference/flight specs/contract.json") & {idx.T_PATH, idx.T_SUBTREE} == set()


@pytest.mark.parametrize("name, line", [("a.ps1", "Get-ChildItem docs -Recurse -File | Get-FileHash"), ("a.sh", "find docs -name '*.md'"),
                                        ("a.bat", "dir docs /s"), ("a.ps1", "Get-ChildItem .\\docs\\ -Recurse"),
                                        ("a.sh", "cd ./docs/ && ls")])
def test_a_tree_named_as_a_command_argument_is_a_whole_tree_reference(tmp_path, name, line):
    files = {**BASE, "reference/n.md": "# n\n", f"scripts/{name}": line + "\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"scripts/{name}"]))
    assert idx.T_SUBTREE in tags(model, "docs/A.md") and idx.T_SUBTREE not in tags(model, "reference/n.md")


@pytest.mark.parametrize("name, line", [("a.ps1", "# keep docs tidy"), ("a.sh", "echo done  # docs are local"),
                                        ("a.bat", "REM docs here"), ("a.bat", ":: docs here"), ("a.sh", "echo mydocs docsx x.docs"),
                                        ("a.ps1", "Write-Host documents")])
def test_a_tree_word_in_a_script_comment_or_inside_another_word_is_not_a_reference(tmp_path, name, line):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, f"scripts/{name}": line + "\n"}, tracked=[f"scripts/{name}"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_bare_tree_name_in_python_prose_or_code_is_not_a_command_argument(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "src/a.py": "docs = 1\nprint(docs)\n# docs here\n"}, tracked=["src/a.py"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_output_records_of_any_text_type_are_scanned_and_the_rest_are_counted(tmp_path):
    files = {**BASE, "outputs/camp/freeze.yaml": "path: docs/A.md\n", "outputs/camp/blob.bin": "docs/B.md\n",
             "outputs/camp/notes.rst": "see docs/C.md\n", "docs/C.md": "# c\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert idx.T_OUTPUTS in tags(model, "docs/A.md") and idx.T_OUTPUTS in tags(model, "docs/C.md")
    assert idx.T_OUTPUTS not in tags(model, "docs/B.md") and model.gaps["outputs_skipped_extension"] == 1
    assert "not text" in idx.render(model, TODAY)


def test_the_page_lists_what_the_scan_cannot_see_by_design(tmp_path):
    text = idx.render(idx.build_model(make_repo(tmp_path / "r", dict(BASE))), TODAY)
    section = text.split("Known limits")[1].split("\n## ")[0].lower()
    for needle in ("environment", "command-line", "walk", "importlib"):
        assert needle in section, needle


def test_a_construction_beyond_the_listing_cap_still_marks_its_files():
    model, scanner = synthetic_scanner(["docs/q1.md", "docs/z9.md"])
    scanner.scan("code", "src/t.py", "".join(f"p = 'docs/q{i}' + name\n" for i in range(idx.MAX_CONSTRUCTIONS + 1))
                 + "late = 'docs/z' + name\n")
    assert model.gaps["constructions_unlisted"] == 2 and idx.T_UNRESOLVED in model.targets["docs/z9.md"].tags


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 6
# Spellings of an ordinary static dependency in the languages this repository uses. Each test is one reproduced under-marking.

@pytest.mark.parametrize("name, line", [
    ("a.sh", "find ../docs -type f"), ("a.sh", "find ../../docs/ -name x"), ("a.ps1", 'Get-ChildItem "$PSScriptRoot/../docs" -Recurse'),
    ("a.ps1", "Get-ChildItem $PSScriptRoot\\..\\docs -Recurse"), ("a.bat", "dir %~dp0..\\docs /s"), ("a.sh", 'ls "${ROOT}/docs"'),
    ("a.py", "list(Path(r'\\\\?\\C:\\repo\\docs').rglob('*'))"), ("a.py", "list(Path('C:/repo/docs').rglob('*'))"),
    ("c.json", '{"root": "file:///C:/repo/docs"}'), ("a.py", 'list(Path(f"{root}/docs").rglob("*"))'),
])
def test_a_tree_named_with_a_prefix_is_still_a_whole_tree_reference(tmp_path, name, line):
    folder = "configs" if name.endswith(".json") else "scripts"
    files = {**BASE, "reference/n.md": "# n\n", f"{folder}/{name}": line + "\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"{folder}/{name}"]))
    assert idx.T_SUBTREE in tags(model, "docs/A.md") and idx.T_SUBTREE not in tags(model, "reference/n.md")


@pytest.mark.parametrize("name, line", [("a.sh", "echo src/docsx ../mydocs ./x.docs"), ("a.ps1", 'Write-Host "https://example.org/docs"'),
                                        ("a.sh", "cat ../notdocs/zzz.txt")])
def test_a_prefix_does_not_turn_a_longer_name_or_a_url_into_a_tree(tmp_path, name, line):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, f"scripts/{name}": line + "\n"}, tracked=[f"scripts/{name}"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ["from reference import (\n    reference_math,\n    other_mod as om,\n)\n",
                                  "from reference import (  # the modules\n    # not unrelated, a comment\n    reference_math,\n)\n",
                                  "from reference import reference_math\n"])
def test_a_parenthesised_import_list_names_each_module(tmp_path, text):
    files = {"reference/reference_math.py": "X = 1\n", "reference/unrelated.py": "X = 2\n", "src/a.py": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_IMPORT in tags(model, "reference/reference_math.py") and idx.T_IMPORT not in tags(model, "reference/unrelated.py")


@pytest.mark.parametrize("code, wanted, form_tag", [
    ('p = (\n    "reference/flight_"\n    "manifest.json"\n)\n', "reference/flight_manifest.json", "T_PATH"),
    ('q = ("reference/run_"\n     f"{n}.json")\n', "reference/run_1.json", "T_UNRESOLVED"),
    ('r = open("reference/rep"  # the head\n         "ort.json")\n', "reference/report.json", "T_PATH"),
    ('s = [\n    "reference/report_",  # comma: two list items, not one string\n    "a.json",\n]\n', None, None),
    ('a = "reference/flight_"\nb = "manifest.json"\n', None, None),
])
def test_adjacent_python_literals_are_merged_inside_open_brackets_only(tmp_path, code, wanted, form_tag):
    names = ["reference/flight_manifest.json", "reference/run_1.json", "reference/report.json", "reference/report_a.json"]
    files = {**{n: "{}" for n in names}, "src/t.py": code}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    if wanted is None:
        assert tags(model, "reference/flight_manifest.json") == {idx.T_NONE}       # separate statements / list items are not one path
    else:
        assert getattr(idx, form_tag) in tags(model, wanted)


@pytest.mark.parametrize("code, wanted, unwanted", [
    ('open(Path("reference/report").with_suffix(".json"))\n', "reference/report.json", "reference/other.json"),
    ('Path("reference/sub/template.json").with_name("report.json").read_text()\n', "reference/sub/report.json", "reference/other.json"),
    ('Path("reference/sub/report.yaml").with_stem("x")\n', "reference/sub/x.yaml", "reference/other.json"),
])
def test_pathlib_name_transformations_mark_the_files_they_could_produce(tmp_path, code, wanted, unwanted):
    files = {"reference/report.json": "{}", "reference/sub/report.json": "{}", "reference/sub/x.yaml": "{}", "reference/other.json": "{}",
             "src/t.py": code}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.T_UNRESOLVED in tags(model, wanted)
    assert idx.T_UNRESOLVED not in tags(model, unwanted)


@pytest.mark.parametrize("name, text, wanted", [
    ("a.py", 'open("docs/R\\u00e9sum\\u00e9.md")\n', "docs/R\u00e9sum\u00e9.md"),
    ("a.py", "open('reference/my\\x20reports/contract.json')\n", "reference/my reports/contract.json"),
    ("c.yaml", 'root: "reference/my\\x20reports"\n', "reference/my reports/contract.json"),
    ("c.toml", 'root = "reference/my\\U000000e9"\n', "reference/my\u00e9/c.json"),
    ("c.yaml", "root: 'reference/owner''s notes'\n", "reference/owner's notes/a.json"),
    ("a.ps1", "Get-ChildItem 'reference/owner''s notes' -Recurse\n", "reference/owner's notes/a.json"),
])
def test_escapes_and_quote_doubling_are_decoded_by_the_language_that_wrote_them(tmp_path, name, text, wanted):
    folder = "scripts" if name.endswith((".py", ".ps1")) else "configs"
    files = {"docs/R\u00e9sum\u00e9.md": "# r\n", "reference/my reports/contract.json": "{}", "reference/my\u00e9/c.json": "{}",
             "reference/owner's notes/a.json": "{}", f"{folder}/{name}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"{folder}/{name}"]))
    assert tags(model, wanted) & {idx.T_PATH, idx.T_SUBTREE}


def test_a_raw_python_string_keeps_its_backslashes(tmp_path):
    files = {"reference/my reports/contract.json": "{}", "src/a.py": "open(r'reference/my\\x20reports/contract.json')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert tags(model, "reference/my reports/contract.json") & {idx.T_PATH, idx.T_SUBTREE} == set()


@pytest.mark.parametrize("text", ["root: &contract_root reference\n", "root: !!str reference\n", "root: >-\n  reference\n",
                                  "root: |\n  reference/\n", "root: >\n  reference/flight\n  specs\n"])
def test_yaml_anchors_tags_and_block_scalars_are_read_as_their_value(tmp_path, text):
    files = {"reference/flight specs/contract.json": "{}", "reference/n.md": "# n\n", "docs/A.md": "# a\n", "configs/c.yaml": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.yaml"]))
    wanted = "reference/flight specs/contract.json" if "flight" in text else "reference/n.md"
    assert idx.T_SUBTREE in tags(model, wanted) and tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_block_scalar_stops_at_the_next_key(tmp_path):
    files = {"reference/n.md": "# n\n", "docs/A.md": "# a\n", "configs/c.yaml": "note: >\n  see the text\nother: value\nroot: x\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.yaml"]))
    assert tags(model, "reference/n.md") == {idx.T_NONE}


@pytest.mark.parametrize("line, named", [("cat {docs,specs}/*.md", {"docs", "specs"}), ("ls {reference}", {"reference"}),
                                         ("tar cf x.tar {./docs,b}", {"docs"}), ("echo {a,b}/c {x}", set())])
def test_shell_brace_alternatives_that_name_a_tree_are_whole_tree_references(tmp_path, line, named):
    files = {**BASE, "specs/B.md": "# b\n", "reference/n.md": "# n\n", "scripts/a.sh": line + "\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["scripts/a.sh"]))
    got = {tree for tree, path in (("docs", "docs/A.md"), ("specs", "specs/B.md"), ("reference", "reference/n.md"))
           if idx.T_SUBTREE in tags(model, path)}
    assert got == named


def test_a_markdown_link_destination_in_angle_brackets_may_hold_spaces(tmp_path):
    files = {"reference/flight specs/contract.json": "{}", "docs/my reports/a.md": "# a\n", "docs/other.md": "# o\n",
             "docs/note.md": "[contract directory](<reference/flight specs/>) and [a](<docs/my reports/a.md> \"title\")\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert idx.T_NOTES in tags(model, "reference/flight specs/contract.json") and idx.T_NOTES in tags(model, "docs/my reports/a.md")
    assert tags(model, "docs/other.md") == {idx.T_NONE}


def test_redundant_dot_components_do_not_exhaust_the_depth_bound(tmp_path):
    files = {**BASE, "src/a.py": "open('docs/././././././././././././A.md')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md")


def test_the_page_says_it_is_evidence_not_proof_and_names_the_undecoded_families(tmp_path):
    text = idx.render(idx.build_model(make_repo(tmp_path / "r", dict(BASE))), TODAY)
    header = text.split("## Start here")[0].lower()
    assert "evidence" in header and "not proof" in header and "test suite" in header
    section = text.split("Known limits")[1].split("\n## ")[0].lower()
    for needle in ("backslash-escaped", "backtick", "percent-encoded", "8.3", "multi-line quoted"):
        assert needle in section, needle


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 7
# Round 6's own rules could blanket a tree: sol measured one comment turning the real page from "7 code-path, 20 notes, 9 loop"
# into "7 code-path, 29 code-tree". These tests pin that nothing ordinary does that, and the under-marking both models found.

@pytest.mark.parametrize("name, text", [
    ("a.py", '# Keep "docs" tidy\n'), ("a.py", "x = 1  # see 'docs' for more\n"), ("a.sh", '# cd "docs"\n'), ("a.ps1", "Get-Date  # \"docs\"\n"),
    ("c.yaml", '# root: "docs"\n'), ("c.yaml", 'name: x  # "docs"\n'), ("c.toml", 'a = 1  # "docs"\n'), ("c.ini", '; "docs"\n'),
    ("a.bat", 'REM "docs"\n'), ("a.py", '"""A docstring that says "docs" is fine."""\nx = 1\n'),
    ("a.py", "x = 'ok'  # keep \"docs\" tidy\n"), ("a.py", "x = (\n    'ok'  # keep \"docs\" tidy\n)\n"),
    ("a.sh", "echo 'ok'  # \"docs\"\n"), ("c.yaml", "name: 'ok'  # \"docs\"\n"), ("c.json", '{"a": "ok"}\n'),
])
def test_a_quoted_tree_name_inside_a_comment_is_not_a_whole_tree_reference(tmp_path, name, text):
    folder = "configs" if name.startswith("c.") else "scripts"
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, f"{folder}/{name}": text}, tracked=[f"{folder}/{name}"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_comment_can_still_name_a_specific_file(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "src/a.py": '# see "docs/A.md"\n'}, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md") and tags(model, "docs/B.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [
    ("a.py", 'url = "https://example.org/docs/{page}"\n'), ("a.py", "u = 'https://x.org/specs/'\n"),
    ("a.sh", "curl https://example.org/docs\n"), ("a.ps1", "Invoke-WebRequest https://example.org/docs -OutFile x\n"),
    ("a.sh", 'curl "https://example.org/docs/A.md"\n'), ("c.json", '{"u": "https://example.org/docs/A.md"}\n'),
    ("n.md", "see https://example.org/docs/A.md for details\n"),
])
def test_a_url_never_marks_a_tree_or_a_file(tmp_path, name, text):
    folder = "configs" if name.startswith("c.") else ("docs" if name == "n.md" else "scripts")
    files = {**BASE, f"{folder}/{name}": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"{folder}/{name}"] if folder != "docs" else []))
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "docs/B.md") == {idx.T_NONE}


def test_a_file_uri_is_still_a_local_path(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "src/a.py": "p = 'file:///C:/repo/docs/A.md'\n"}, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md")


@pytest.mark.parametrize("name, text", [
    ("a.sh", 'echo "Please read docs before running"\n'), ("a.ps1", 'Write-Host "read docs now"\n'),
    ("a.ps1", "<#\nKeep docs tidy\n#>\nGet-Date\n"), ("a.ps1", "<# docs #> Get-Date\n"), ("a.sh", "echo 'read docs now'\n"),
])
def test_a_tree_word_inside_quoted_prose_or_a_powershell_block_comment_is_not_an_argument(tmp_path, name, text):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, f"scripts/{name}": text}, tracked=[f"scripts/{name}"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [
    ("c.yaml", 'description: "The roots are [docs, specs]"\n'), ("c.yaml", "note: |\n  Roots: [docs, specs]\n"),
    ("c.yaml", "description: Notes, docs, examples\n"), ("c.json", '{"message": "Read \'docs\' for help"}\n'),
    ("c.yaml", "note: >\n  docs, specs\nother: x\n"), ("c.toml", 'help = "see [docs, specs]"\n'),
])
def test_prose_in_a_config_value_is_not_a_collection_of_roots(tmp_path, name, text):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "specs/s.md": "# s\n", f"configs/{name}": text}, tracked=[f"configs/{name}"]))
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "specs/s.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ["roots: [reference/flight specs]\n", "roots: [x, reference/flight specs, y]\n",
                                  "paths: {contract: reference/flight specs}\n", "- root: &folder reference/flight specs\n",
                                  "- root: reference/flight specs\n", "  - name: x\n    root: !!str reference/flight specs\n"])
def test_complete_unquoted_config_paths_in_collections_and_sequence_mappings_are_read(tmp_path, text):
    files = {"reference/flight specs/contract.json": "{}", "reference/n.md": "# n\n", "configs/c.yaml": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.yaml"]))
    assert idx.T_SUBTREE in tags(model, "reference/flight specs/contract.json")
    assert tags(model, "reference/n.md") == {idx.T_NONE}


def test_literal_additions_are_one_path_not_a_variable_tail(tmp_path):
    files = {**BASE, "src/a.py": 'p = "docs/" + "A.md"\nq = "docs/" + "missing.md"\nr = "docs/b" + ".md".lower()\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md")
    assert idx.T_UNRESOLVED not in tags(model, "docs/A.md") and idx.T_UNRESOLVED not in tags(model, "docs/B.md")


def test_an_addition_with_a_variable_is_still_a_tail(tmp_path):
    files = {**BASE, "src/a.py": 'p = "docs/" + name\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_UNRESOLVED in tags(model, "docs/A.md") and idx.T_UNRESOLVED in tags(model, "docs/B.md")


@pytest.mark.parametrize("text", ["if TYPE_CHECKING: import reference_math\n", "from reference import \\\n    reference_math\n",
                                  "from reference import (\n    helper,  # (utilities)\n    reference_math,\n)\n",
                                  "import os; import reference_math\n", "try:\n    import reference_math\nexcept ImportError:\n    pass\n",
                                  "import reference.\\\nreference_math\n"])
def test_static_imports_are_read_from_logical_lines(tmp_path, text):
    files = {"reference/reference_math.py": "X = 1\n", "reference/unrelated.py": "X = 2\n", "src/a.py": text}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_IMPORT in tags(model, "reference/reference_math.py") and idx.T_IMPORT not in tags(model, "reference/unrelated.py")


def test_an_import_word_inside_a_comment_or_a_longer_name_is_not_an_import(tmp_path):
    files = {"reference/reference_math.py": "X = 1\n", "src/a.py": "# import reference_math\nimportlib.import_module('x')\nimport_reference_math = 1\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_IMPORT not in tags(model, "reference/reference_math.py")


def test_brackets_inside_a_docstring_do_not_merge_statements_or_move_their_evidence(tmp_path):
    code = '"""Parse an opening parenthesis (\nThis is prose.\n"""\nROOT = "docs"\n"""Directory containing our contracts."""\n'
    files = {**BASE, "src/a.py": code, "src/b.py": '"""Example: (\n"""\nPath("docs") / "A.md"\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py", "src/b.py"]))
    assert idx.T_SUBTREE in tags(model, "docs/B.md")                                   # ROOT = "docs" is a whole-tree literal
    assert [e.line for e in model.targets["docs/A.md"].evidence["code"] if e.form == "chain"] == [3]


@pytest.mark.parametrize("code, wanted", [('p = "docs/\\\nA.md"\n', "docs/A.md"), ("p = Path('docs') / \\\n    'A.md'\n", "docs/A.md"),
                                          ('p = "docs/my \\\nreports/a.md"\n', "docs/my reports/a.md")])
def test_a_backslash_newline_inside_a_string_joins_without_a_blank_and_between_tokens_with_one(tmp_path, code, wanted):
    files = {**BASE, "docs/my reports/a.md": "# a\n", "src/a.py": code}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert tags(model, wanted) & {idx.T_PATH, idx.T_SUBTREE}


@pytest.mark.parametrize("code", ["open('reference/owner\\'s notes/c.json')\n", 'open("reference/owner\'s notes/c.json")\n',
                                  'open("reference/owner\\\'s notes/c.json")\n'])
def test_an_escaped_quote_does_not_end_a_python_string(tmp_path, code):
    files = {"reference/owner's notes/c.json": "{}", "reference/other.json": "{}", "src/a.py": code}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "reference/owner's notes/c.json") and idx.T_PATH not in tags(model, "reference/other.json")


def test_an_escaped_double_quote_does_not_end_a_json_or_python_string(tmp_path):
    """A double quote cannot be in a Windows file name, so the string is followed by a real path to prove it was read whole."""
    files = {**BASE, "configs/c.json": '{"note": "say \\"hi\\" now", "root": "docs/A.md"}\n',
             "src/a.py": 'x = "say \\"hi\\" now"; y = "docs/B.md"\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.json", "src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md") and idx.T_PATH in tags(model, "docs/B.md")


def test_an_escaped_quote_in_a_raw_string_does_not_swallow_the_rest_of_the_line(tmp_path):
    """A raw string keeps the backslash (its value is ``a\\'b``) but the quote still does not end it."""
    code = "x = r'a\\'b'; y = Path('docs') / 'A.md'\n"
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "src/a.py": code}, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md") and tags(model, "docs/B.md") == {idx.T_NONE}


def test_a_powershell_string_ending_in_a_backslash_is_not_an_escape(tmp_path):
    files = {**BASE, "reference/n.md": "# n\n", "scripts/a.ps1": 'Get-ChildItem "docs\\" -Recurse; Write-Host "x"\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["scripts/a.ps1"]))
    assert idx.T_SUBTREE in tags(model, "docs/A.md") and tags(model, "reference/n.md") == {idx.T_NONE}


@pytest.mark.parametrize("path", ["docs/x/y/z/w/../../../../A.md", "docs/sub/../sub/../sub/../sub/../sub/../A.md",
                                  "docs/./sub/./../././A.md"])
def test_parent_components_that_cancel_do_not_exhaust_the_depth_bound(tmp_path, path):
    files = {**BASE, "docs/sub/B2.md": "# b\n", "src/a.py": f"open('{path}')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/A.md") and tags(model, "docs/sub/B2.md") == {idx.T_NONE}


def test_a_hostile_run_of_cancelling_components_stays_linear():
    import time
    model, scanner = synthetic_scanner(["docs/A.md"])
    started = time.perf_counter()
    scanner.scan("code", "src/t.txt", "see " + "docs/../" * 200_000 + "A.md\n")
    assert time.perf_counter() - started < 3.0


# ---------------------------------------------------------------------------------------- round 7: the lexers themselves
# The fixes above rest on small lexers (Python logical lines, shell and config lines, URLs, here-documents). Each rule gets a
# test of its own, and the mutation harness removes each rule to prove one of these tests notices.

def one_file(tmp_path, folder, name, text, extra=None, tracked=True):
    files = {**BASE, **(extra or {}), f"{folder}/{name}": text}
    return idx.build_model(make_repo(tmp_path / "r", files, tracked=[f"{folder}/{name}"] if tracked else []))


@pytest.mark.parametrize("name, text", [
    ("a.sh", "cat <<EOF\nRead docs before running\nEOF\n"), ("a.sh", "cat <<'EOF'\nRead docs\nEOF\n"),
    ("a.sh", "cat <<-EOF\n\tRead docs\n\tEOF\n"), ("a.sh", 'cat <<"EOF"\nRead docs\nEOF\n'),
    ("a.ps1", '$text = @"\nRead docs before running\n"@\nGet-Date\n'), ("a.ps1", "$text = @'\nRead docs\n'@\n"),
    ("a.ps1", "<#\n.SYNOPSIS\nBuilds the docs\n#>\nGet-Date\n"),
    ("a.ps1", '$t = @"\nroot: \'docs\'\n"@\n'), ("a.sh", "cat <<EOF\nroot: 'docs'\nEOF\n"),
])
def test_here_documents_here_strings_and_block_comments_are_data(tmp_path, name, text):
    assert tags(one_file(tmp_path, "scripts", name, text), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [
    ("a.sh", "cat <<EOF\ntext\nEOF\nls docs\n"), ("a.ps1", "<# c #>\nGet-ChildItem docs\n"),
    ("a.ps1", '$t = @"\nx\n"@\nGet-ChildItem docs\n'), ("a.sh", "cat docs <<EOF\nbody\nEOF\n"),
    ("a.sh", "# cat <<EOF\nls docs\n"), ("a.sh", 'echo "a << b"\nls docs\n'), ("a.ps1", "# see <# here\nGet-ChildItem docs\n"),
    ("a.ps1", "Write-Host 'x'  <# c #>  ; Get-ChildItem docs\n"),
])
def test_code_after_a_block_or_around_a_false_opener_is_still_read(tmp_path, name, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")


@pytest.mark.parametrize("name, text, marked", [
    ("a.sh", 'echo "files: $(ls docs)"\n', True), ("a.sh", 'echo "`ls docs`"\n', True),
    ("a.ps1", 'Write-Host "$(Get-ChildItem docs)"\n', True), ("a.sh", "echo 'ls docs'\n", False),
    ("a.sh", 'echo "ls docs"\n', False), ("a.ps1", 'Write-Host "read docs`n"\n', False),
    ("a.sh", 'ls "$HOME"/docs\n', True), ("a.ps1", 'Get-ChildItem "$root"\\docs\n', True),
    ("a.sh", 'echo "x" docs\n', True),
    ("a.bat", "for /f %%i in ('dir docs /b') do echo %%i\n", True), ("a.bat", 'echo "read docs now"\n', False),
    ("a.bat", "dir docs /s\n", True), ("a.sh", "echo a#b docs\n", True), ("a.sh", "echo $# docs\n", True),
    ("a.sh", "echo a # docs\n", False),
])
def test_what_a_shell_runs_is_read_and_what_it_prints_is_not(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text", ["REM keep docs tidy\n", "rem docs\n", ":: docs\n", "@rem docs\n", "  REM\tdocs\n"])
def test_a_batch_comment_names_no_tree(tmp_path, text):
    assert tags(one_file(tmp_path, "scripts", "a.bat", text), "docs/A.md") == {idx.T_NONE}


def test_a_tree_deeper_than_any_real_path_is_cut_by_depth_and_is_not_counted_as_too_long(tmp_path):
    model = one_file(tmp_path, "src", "a.py", "open('" + "docs/" * 40 + "A.md')\n")
    assert model.gaps["paths_too_long"] == 0


def test_a_quoted_tail_stays_glued_to_the_word_before_it(tmp_path):
    model = one_file(tmp_path, "scripts", "a.sh", 'cat reference/"run1.json"\n', {"reference/run1.json": "{}", "reference/other.json": "{}"})
    assert tags(model, "reference/other.json") == {idx.T_NONE}


def test_a_yaml_block_scalar_body_is_prose_but_its_folded_value_is_a_path(tmp_path):
    extra = {"specs/s.md": "# s\n", "reference/flight specs/contract.json": "{}"}
    text = ("note: |\n  Roots: [docs, specs]\n  - docs\n  specs: specs\nother: >\n  reference: specs\n"
            "root: >-\n  reference/flight\n  specs\n")
    model = one_file(tmp_path, "configs", "c.yaml", text, extra)
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "specs/s.md") == {idx.T_NONE}
    assert idx.T_SUBTREE in tags(model, "reference/flight specs/contract.json")


@pytest.mark.parametrize("name, text", [("c.toml", "[docs]\nname = 'x'\n"), ("c.toml", "[[docs]]\n"), ("c.ini", "[docs]\nname = x\n"),
                                        ("c.ini", "[testenv:docs]\ncommands = x\n"), ("c.cfg", "[docs]\n")])
def test_a_section_header_is_a_name_and_not_a_list_of_roots(tmp_path, name, text):
    assert tags(one_file(tmp_path, "configs", name, text), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [
    ("c.ini", "[x]\nroots = docs, specs\n"), ("c.cfg", "[x]\nroots: specs,docs\n"), ("c.yaml", "roots: [\n  docs,\n  specs\n]\n"),
    ("c.yaml", "- [docs, specs]\n"), ("c.yaml", "- {root: docs, other: specs}\n"), ("c.yaml", "roots: [docs, specs] # both\n"),
    ("c.yaml", "root: docs # the tree\n--- \nroots: [specs]\n"),
])
def test_ini_lists_and_yaml_flow_collections_name_their_trees(tmp_path, name, text):
    model = one_file(tmp_path, "configs", name, text, {"specs/s.md": "# s\n"})
    assert idx.T_SUBTREE in tags(model, "docs/A.md") and idx.T_SUBTREE in tags(model, "specs/s.md")


def test_a_comment_hides_a_yaml_value_and_a_hash_inside_a_word_does_not(tmp_path):
    model = one_file(tmp_path, "configs", "c.yaml", "root: x # docs\nother: a#docs\nthird: 'docs'  # quoted\n", {"specs/s.md": "# s\n"})
    assert idx.T_SUBTREE in tags(model, "docs/A.md")               # only the quoted value names it
    only_comment = one_file(tmp_path / "again", "configs", "c.yaml", "root: x # docs\nother: a#docs\n", {"specs/s.md": "# s\n"})
    assert tags(only_comment, "docs/A.md") == {idx.T_NONE}


def test_brackets_inside_strings_do_not_join_statements(tmp_path):
    model = one_file(tmp_path, "src", "a.py", 'd = {"(": 1, "[": 2}\nPath("docs") / "A.md"\n')
    assert [e.line for e in model.targets["docs/A.md"].evidence["code"] if e.form == "chain"] == [2]


def test_an_unterminated_string_ends_at_its_line(tmp_path):
    model = one_file(tmp_path, "src", "a.py", "x = 'abc\nPath('docs') / 'A.md'\n")
    assert [e.line for e in model.targets["docs/A.md"].evidence["code"] if e.form == "chain"] == [2]


def test_a_one_line_triple_quoted_string_is_an_ordinary_string(tmp_path):
    model = one_file(tmp_path, "src", "a.py", 'p = """docs/A.md"""\nROOT = \'\'\'specs\'\'\'\n', {"specs/s.md": "# s\n"})
    assert idx.T_PATH in tags(model, "docs/A.md") and idx.T_SUBTREE in tags(model, "specs/s.md")


def test_a_multi_line_docstring_is_prose_even_when_it_quotes_a_tree(tmp_path):
    model = one_file(tmp_path, "src", "a.py", '"""\nSet ROOT = "docs" in your config.\nThen run it.\n"""\nx = 1\n')
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_a_backslash_continuation_between_tokens_joins_the_statement(tmp_path):
    model = one_file(tmp_path, "src", "a.py", "p = Path('docs') / \\\n    'A.md'\nq = 1\n")
    assert [e.line for e in model.targets["docs/A.md"].evidence["code"] if e.form == "chain"] == [1]


def test_an_unclosed_bracket_cannot_swallow_the_rest_of_the_file(tmp_path):
    """At most 40 physical lines are joined: the statement 60 lines later is its own line, with its own number."""
    code = "f(\n" + "x = 1\n" * 60 + "Path('docs') / 'A.md'\n"
    model = one_file(tmp_path, "src", "a.py", code)
    assert [e.line for e in model.targets["docs/A.md"].evidence["code"] if e.form == "chain"] == [62]


@pytest.mark.parametrize("text, marked", [
    ("p = 'sqlite:///docs/A.md'\n", True), ("p = 'file:///C:/repo/docs/A.md'\n", True), ("p = 'C://repo/docs/A.md'\n", True),
    ("p = 'http://host:8080/docs/A.md'\n", False), ("p = 'https://x.org/y' ; q = 'docs/A.md'\n", True),
    ("u = 'https://x.org/docs'\n", False), ("f('https://x.org/y','docs/A.md')\n", True), ("f(\"https://x.org/y\",\"docs/A.md\")\n", True),
    ("p = ('https://x.org/y')+('docs/A.md')\n", True), ("p = 'file://server/share/docs/A.md'\n", True),
])
def test_a_url_with_a_network_authority_is_not_a_path_and_local_forms_are(tmp_path, text, marked):
    found = bool(tags(one_file(tmp_path, "src", "a.py", text), "docs/A.md") & {idx.T_PATH, idx.T_SUBTREE, idx.T_UNRESOLVED})
    assert found is marked


def test_an_escaped_backslash_before_a_quote_still_ends_the_string(tmp_path):
    """``'a\\\\'`` is the string ``a\\`` (an escaped backslash), so its quote closes it: what follows is code again."""
    model = one_file(tmp_path, "src", "a.py", "x = 'a\\\\'; y = Path('docs') / 'A.md'\n")
    assert idx.T_PATH in tags(model, "docs/A.md") and tags(model, "docs/B.md") == {idx.T_NONE}
    commented = one_file(tmp_path / "again", "src", "a.py", "x = 'a\\\\'  # see \"docs\"\n")      # the comment is a comment
    assert tags(commented, "docs/A.md") == {idx.T_NONE}


def test_an_escaped_quote_in_a_flow_mapping_does_not_hide_the_value_after_it(tmp_path):
    model = one_file(tmp_path, "configs", "c.yaml", '{note: "say \\"hi\\"", root: "docs"}\n')
    assert idx.T_SUBTREE in tags(model, "docs/A.md")
    model = one_file(tmp_path / "again", "configs", "c.json", '{"note": "say \\"hi\\"", "root": "docs"}\n')
    assert idx.T_SUBTREE in tags(model, "docs/A.md")


def test_a_truncated_candidate_is_never_resolved_as_a_directory(tmp_path):
    """A path longer than the reach is counted, not cut short: its first components name ``docs/sub/x`` (a real directory
    here) and would mark everything in it, though the whole path is ``docs/A.md``."""
    path = "docs/sub/" + "x/../" * 30 + "../A.md"
    model = one_file(tmp_path, "src", "a.py", f"open('{path}')\n", {"docs/sub/x/F.md": "# f\n"})
    assert model.gaps["paths_too_long"] > 0 and tags(model, "docs/sub/x/F.md") == {idx.T_NONE}


def test_an_import_inside_a_string_is_not_an_import(tmp_path):
    files = {"reference/reference_math.py": "X = 1\n", "src/a.py": "s = 'import reference_math'\nd = {'k': 'x; import reference_math'}\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_IMPORT not in tags(model, "reference/reference_math.py")


@pytest.mark.parametrize("text", ["note: see [1], docs, [2]\n", "roots: [1], docs, [2]\n", "- [1] docs [2]\n"])
def test_words_outside_the_brackets_of_a_line_are_not_items_of_its_collection(tmp_path, text):
    model = one_file(tmp_path, "configs", "c.yaml", text)
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_literal_additions_in_powershell_are_one_string(tmp_path):
    model = one_file(tmp_path, "scripts", "a.ps1", '$p = "docs/" + "A.md"\n')
    assert idx.T_PATH in tags(model, "docs/A.md") and idx.T_UNRESOLVED not in tags(model, "docs/B.md")


def test_a_path_too_long_to_resolve_is_counted_and_the_page_says_so(tmp_path):
    model = one_file(tmp_path, "src", "a.py", "open('" + "docs/../" * 30 + "A.md')\n")
    assert model.gaps["paths_too_long"] > 0
    assert "too long to resolve" in idx.render(model, TODAY)


def test_a_path_within_the_reach_is_never_counted_as_too_long(tmp_path):
    model = one_file(tmp_path, "src", "a.py", "open('docs/x/y/z/w/../../../../A.md')\n")
    assert model.gaps["paths_too_long"] == 0


HOSTILE_SHAPES = {                                  # label -> (source name, builder): ~1-3 MB, quadratic code takes minutes
    "a string continued over a million lines": ("src/t.py", lambda: "x = '" + "a\\\n" * 1_000_000 + "'\n"),
    "escaped quotes in one string": ("src/t.py", lambda: "x = '" + "\\'" * 1_500_000 + "'\n"),
    "a backslash run before a quote": ("src/t.py", lambda: "x = '" + "\\" * 2_990_000 + "'\n"),
    "triple quotes": ("src/t.py", lambda: '"""' * 1_000_000),
    "short lines": ("src/t.py", lambda: "x\n" * 1_500_000),
    "open brackets": ("src/t.py", lambda: "f(\n" * 1_000_000),
    "block openers": ("scripts/t.ps1", lambda: "<#" * 1_500_000),
    "closed here-strings": ("scripts/t.ps1", lambda: '@"\nx\n"@\n' * 330_000),
    "heredoc openers": ("scripts/t.sh", lambda: "<<A\n" * 750_000),
    "closed heredocs": ("scripts/t.sh", lambda: "cat <<A\nx\nA\n" * 250_000),
    "URL schemes": ("src/t.md", lambda: "a://" * 750_000),
    "quotes in a shell script": ("scripts/t.sh", lambda: '"' * 2_000_000),
    "flow lines": ("configs/t.yaml", lambda: "k: [\n" + "  docs,\n" * 375_000 + "]\n"),
    "a block scalar": ("configs/t.yaml", lambda: "a: |\n" + "  docs\n" * 400_000),
    "INI list items": ("configs/t.ini", lambda: "k = " + "docs," * 600_000 + "\n"),
    "literals added in Python": ("src/t.py", lambda: "x = " + "'a'+" * 750_000 + "'z'\n", 5.0),       # linear: ~1.5 s, a copy per add: ~9 s
    "literals added in PowerShell": ("scripts/t.ps1", lambda: "$x = " + "'a'+" * 750_000 + "'z'\n", 5.0),
    "many strings on one line": ("src/t.py", lambda: "x = " + "'ab' " * 500_000 + "\n"),
    "cancelling components": ("src/t.txt", lambda: "docs/../" * 300_000 + "A.md\n"),
    "f-string fields": ("src/t.py", lambda: 'x = f"' + "{a}" * 700_000 + '"\n'),
    "f-string escaped braces": ("src/t.py", lambda: 'x = f"' + "{{" * 1_400_000 + '"\n'),
    "f-string unclosed field": ("src/t.py", lambda: 'x = f"{' + "a" * 2_900_000 + '"\n'),
    "substitutions in a string": ("scripts/t.sh", lambda: 'echo "' + "$(a)" * 600_000 + '"\n'),
    "unclosed substitutions": ("scripts/t.sh", lambda: 'echo "' + "$(" * 1_300_000 + "\n"),
    "backticks": ("scripts/t.sh", lambda: "`" * 2_900_000 + "\n"),
    "arithmetic shifts": ("scripts/t.sh", lambda: "$((1 << X)) " * 230_000 + "\n"),
    "quoted hashes then block comments": ("scripts/t.ps1", lambda: "'#' <# x #> " * 230_000 + "\n"),
    "anchor soup": ("configs/t.yaml", lambda: "- " + "&a " * 950_000 + "\n"),
    "quoted flow keys": ("configs/t.yaml", lambda: "k: {" + '"a":b, ' * 400_000 + "}\n"),
    "batch separators": ("scripts/t.bat", lambda: "echo a & rem b " * 190_000 + "\n"),
    "URLs after quotes": ("src/t.md", lambda: 'a "b\' https://x.org/a(b) ' * 120_000 + "\n"),
    "URLs inside strings": ("src/t.py", lambda: 'u = "https://x.org/a(b)/c" ' * 120_000 + "\n"),
    "flow items with prose": ("configs/t.yaml", lambda: "k: [" + 'Keep "docs" tidy, ' * 160_000 + "]\n"),
    "multi-line arithmetic": ("scripts/t.sh", lambda: "x=$((\n" + "1 << 2\n" * 400_000 + "))\n"),
    "delimiters with quotes": ("scripts/t.sh", lambda: 'cat <<E"OF"\nx\nEOF\n' * 150_000),
    "IPv6 URLs in Python": ("src/t.py", lambda: '"https://[::1]/docs" ' * 150_000 + "\n", 4.0),         # linear ~1.2 s, a line scan each ~7 s
    "IPv6 URLs in Markdown": ("src/t.md", lambda: "https://[::1] " * 210_000 + "\n", 4.0),
    "URLs after many quotes in YAML": ("configs/t.yaml", lambda: 'u: "x" https://x.org/a,b ' * 120_000 + "\n"),
    "many bare roots on one line": ("scripts/t.sh", lambda: "ls " + "docs " * 600_000 + "\n"),
    "many case-like roots on one line": ("scripts/t.sh", lambda: "docs) " * 500_000 + "\n"),
    "f-string format specs": ("src/t.py", lambda: 'x = f"' + "{a:'>10}" * 330_000 + '"\n'),
    "quoted parens in substitutions": ("scripts/t.sh", lambda: 'echo "' + "$(printf '(' )" * 200_000 + '"\n'),
    "escaped substitution openers": ("scripts/t.sh", lambda: 'echo "' + "\\$(a)" * 500_000 + '"\n'),
    "nested sequence dashes": ("configs/t.yaml", lambda: "- " * 1_400_000 + "docs\n"),
    "nested dashes then a block scalar": ("configs/t.yaml", lambda: "- " * 1_400_000 + "|\n  docs\n"),
    "closing braces after a path": ("configs/t.yaml", lambda: "k: {a: docs/" + "}" * 2_900_000 + "\n"),
    "alternating closers after a path": ("configs/t.yaml", lambda: "k: docs/" + "]}" * 1_400_000 + "\n"),
    "doubled quotes inside a URL": ("configs/t.yaml", lambda: "k: 'https://x.org/" + "''" * 1_400_000 + "'\n"),
    "doubled quotes before a URL": ("configs/t.yaml", lambda: "k: '" + "''" * 700_000 + " https://x.org/a'\n"),
    "block scalars on sequence lines": ("configs/t.yaml", lambda: "- run: |\n    x\n  k: docs\n" * 200_000),
    "TOML triple quotes": ("configs/t.toml", lambda: "k = " + "'''a''' " * 400_000 + "\n"),
    "unclosed TOML triple quotes": ("configs/t.toml", lambda: "k = " + "'''" * 900_000 + "\n"),
    "TOML triple quotes with inner quotes": ("configs/t.toml", lambda: "k = " + "'''It's''' " * 300_000 + "\n"),
    "notebook backslash-continued commands": ("notebooks/t.ipynb", lambda: __import__("json").dumps(
        {"nbformat": 4, "cells": [{"cell_type": "code", "source": "!ls \\\n" * 250_000}]})),
    "a plain key of many colons": ("configs/t.yaml", lambda: "- " + "a:" * 1_400_000 + "\n"),
    "a plain key of colons then a blank": ("configs/t.yaml", lambda: "- " + "a:" * 1_400_000 + " docs\n"),
    "multi-line f-string with nested triple quotes": ("src/t.py", lambda: 'x = f"""\n' + "{f'''{a}'''}\n" * 200_000 + '"""\n'),
    "multi-line f-string fields": ("src/t.py", lambda: 'x = f"""\n' + "{a}\n" * 600_000 + '"""\n'),
    "multi-line f-string with a long field": ("src/t.py", lambda: 'x = f"""\n{' + "a" * 2_900_000 + '}\n"""\n'),
    "INI continuation lines": ("configs/t.ini", lambda: "[x]\nk =\n" + "    docs\n" * 400_000),
    "INI continuation list items": ("configs/t.ini", lambda: "[x]\nk =\n    " + "docs," * 500_000 + "\n"),
    "notebook shell lines": ("notebooks/t.ipynb", lambda: __import__("json").dumps(
        {"nbformat": 4, "cells": [{"cell_type": "code", "source": "!ls docs\n" * 250_000}]})),
    "notebook bash cell": ("notebooks/t.ipynb", lambda: __import__("json").dumps(
        {"nbformat": 4, "cells": [{"cell_type": "code", "source": "%%bash\n" + "ls docs\n" * 250_000}]})),
    "names ending in closers": ("configs/t.yaml", lambda: "k: [" + "docs/a.md], " * 200_000 + "]\n"),
    "a run of closers after a path": ("configs/t.yaml", lambda: "k: docs/a.md" + "]}" * 1_400_000 + "\n"),
}


@pytest.mark.parametrize("label", list(HOSTILE_SHAPES))
def test_the_lexers_stay_linear_on_hostile_shapes(label):
    import time
    name, build, *limit = HOSTILE_SHAPES[label]
    text = build()
    model, scanner = synthetic_scanner(["docs/A.md", "specs/s.md"])
    started = time.perf_counter()
    scanner.scan("code", name, text)
    assert time.perf_counter() - started < (limit[0] if limit else 12.0), label


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 8
# Round 7's lexers still read three things as one: an interpolation inside a string (its expression is code, its prose is not),
# an escaped quote or comment next to a command separator, and the state at a heredoc / block-comment opener.

@pytest.mark.parametrize("name, text", [
    ("a.sh", 'echo "Generated $(date): see docs for details."\n'), ("a.ps1", 'Write-Host "Generated $(Get-Date): see docs"\n'),
    ("a.sh", 'echo "`date` - read docs"\n'),
])
def test_prose_around_a_substitution_in_a_string_is_still_prose(tmp_path, name, text):
    assert tags(one_file(tmp_path, "scripts", name, text), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text, marked", [
    ("a.sh", 'x="$(find \'docs\')"\n', True), ("a.sh", 'echo "$(find docs -type f)"\n', True), ("a.sh", 'echo "$(echo "docs")"\n', True),
    ("a.sh", "x=`ls docs`\n", True), ("a.sh", 'echo "files: `find docs`"\n', True), ("a.sh", 'echo "$(echo $(ls docs))"\n', True),
    ("a.ps1", "$x = \"$(Get-ChildItem 'docs')\"\n", True), ("a.ps1", 'Write-Host "n: $(Get-ChildItem docs | Measure-Object)"\n', True),
    ("a.sh", "echo '$(find docs)'\n", False), ("a.ps1", "Write-Host '$(Get-ChildItem docs)'\n", False),
    ("a.sh", "echo '`find docs`'\n", False), ("a.sh", "echo \"`find docs`\"\n", True),
    ("a.sh", 'echo "$(date)" docs\n', True), ("a.sh", 'echo "$(date)"\n', False),
])
def test_the_body_of_a_command_substitution_is_code(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("code, wanted", [
    ("open(f\"{Path('docs') / 'A.md'}\")\n", "A"), ("print(f\"{len(list(Path('docs').glob('*.md')))} files\")\n", "TREE"),
    ("x = f'{Path(\"docs\") / \"A.md\"!s:>10}'\n", "A"), ("x = rf'{Path(\"docs\")}/{name}'\n", "TREE"),
    ("x = f\"{{docs}} {name!r:>{width}}\"\n", "NONE"), ("x = f'{value:%Y-%m-%d}'\n", "NONE"), ("x = f'plain docs text'\n", "NONE"),
    ("x = f\"{{'docs'}} and }}'docs'{{\"\n", "NONE"),
    ("x = f\"{f'{Path(\\\"docs\\\") / \\\"A.md\\\"}'}\"\n", "NONE_OR_A"),
])
def test_the_replacement_fields_of_an_f_string_are_code(tmp_path, code, wanted):
    model = one_file(tmp_path, "src", "a.py", code)
    a, b = tags(model, "docs/A.md"), tags(model, "docs/B.md")
    if wanted == "A":
        assert idx.T_PATH in a and b == {idx.T_NONE}
    elif wanted == "TREE":
        assert idx.T_SUBTREE in a and idx.T_SUBTREE in b
    elif wanted == "NONE":
        assert a == {idx.T_NONE}
    else:
        assert idx.T_UNRESOLVED not in a                                  # a nested f-string must at least not crash or blanket


@pytest.mark.parametrize("name, text, marked", [
    ("a.ps1", 'Write-Host `"; Get-ChildItem docs\n', True), ("a.sh", 'echo \\"; find docs\n', True),
    ("a.bat", 'echo ^"& dir docs\n', True), ("a.sh", "echo \\' ; find docs\n", True),
    ("a.ps1", 'Write-Host "Example: `"Get-ChildItem \'docs\'`""\n', False), ("a.sh", 'echo "say \\"find \'docs\'\\" now"\n', False),
    ("a.ps1", 'Write-Host "a `"quoted`" docs"\n', False), ("a.sh", 'echo "\\\\" docs\n', True),
])
def test_an_escaped_quote_is_neither_an_opener_outside_strings_nor_a_closer_inside_them(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("name, text", [
    ("a.sh", 'true;# mention "docs"\n'), ("a.ps1", 'Get-Date;# "docs"\n'), ("a.sh", 'ls | # docs\n'), ("a.sh", 'a&# docs\n'),
    ("a.sh", '(# "docs"\n'), ("a.ps1", "Write-Host 'done';# keep docs tidy\n"),
])
def test_a_hash_after_a_command_separator_starts_a_comment(tmp_path, name, text):
    assert tags(one_file(tmp_path, "scripts", name, text), "docs/A.md") == {idx.T_NONE}


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 9
# Most of round 9 was my own round-8 rules meeting forms they did not expect: a `]` that ended an IPv6 URL, a bracket that was
# looked for before a YAML anchor, a parenthesis inside a quote inside a substitution, a quote that was a format-spec fill.

@pytest.mark.parametrize("name, text", [
    ("a.py", 'DOCS_URL = "https://[::1]/docs"\n'), ("a.py", 'u = "https://x.org/a[0]/docs"\n'), ("a.py", "u = 'http://[::1]:8080/docs'\n"),
    ("n.md", "see http://[::1]:8080/docs/A.md now\n"), ("a.sh", "curl https://[2001:db8::1]/docs\n"),
    ("c.yaml", "homepage: https://[::1]/docs\n"), ("c.yaml", "urls: [https://[::1]:8080/docs, https://x.org/a]\n"),
    ("c.ini", "[x]\nhomepage = https://[::1]/docs\n"), ("c.yaml", "urls: [https://[::1]/docs/A.md, x]\n"),
    ("c.ini", "[x]\nu = https://[::1]/docs/A.md,b\n"), ("c.yaml", "urls: [https://[2001:db8::1]:8080/docs/A.md]\n"),
])
def test_a_url_with_brackets_is_still_one_url(tmp_path, name, text):
    folder = "docs" if name == "n.md" else ("scripts" if name.endswith(".sh") else ("configs" if name.startswith("c.") else "src"))
    model = one_file(tmp_path, folder, name, text, tracked=folder != "docs")
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "docs/B.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ["roots: &roots [docs]\n", "roots: !!seq [docs]\n", "roots: &a !!seq [docs]\n", "- &a [docs]\n",
                                  "{root: docs}\n", '{"root":docs}\n', "[docs, specs]\n", "- {root: &x docs}\n", "roots:   &r   [ docs ]\n"])
def test_a_flow_collection_after_an_anchor_or_a_tag_or_at_the_top_of_a_document_is_read(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")


@pytest.mark.parametrize("name, text, marked", [
    ("c.yaml", '"sphinx:root": docs\n', True), ("c.yaml", "'a:b': docs\n", True), ("c.toml", '"a=b" = "docs"\n', True),
    ("c.toml", '"a.b" = docs\n', True), ("c.yaml", '"a:b": See docs\n', False), ("c.yaml", "key: \"a:b\"\n", False),
    ("c.yaml", '"a: b": docs\n', True), ("c.yaml", '"unterminated: docs\n', False),
])
def test_the_mapping_separator_is_found_outside_a_quoted_key(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "configs", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("name, text, marked", [
    ("a.sh", 'echo "$(printf \'(\'; find docs)"\n', True), ("a.sh", "x=\"$(printf ')'; find docs)\"\n", True),
    ("a.ps1", '$x = "$(Write-Output \')\'; Get-ChildItem docs)"\n', True), ("a.ps1", 'Write-Host "$(Write-Output \'(\'; Get-ChildItem docs)"\n', True),
    ("a.sh", 'echo "a $(echo "$(ls docs)")"\n', True), ("a.sh", 'x=$(echo $(ls docs))\n', True),
    ("a.sh", 'echo "Use \\$(find docs) later"\n', False), ("a.ps1", 'Write-Host "Use `$(Get-ChildItem docs) later"\n', False),
    ("a.sh", 'echo "\\\\$(find docs)"\n', True), ("a.sh", "echo 'a' \"$(ls docs)\" 'b'\n", True),
    ("a.sh", 'echo "(" "$(ls docs)" ")"\n', True), ("a.sh", 'echo "unbalanced ($(ls docs)"\n', True),
])
def test_a_substitution_is_matched_by_its_own_quoting_and_escapes(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("code, marked", [
    ("print(f\"{len(list(Path('docs').glob('*.md'))):'>10}\")\n", True), ("x = f\"{Path('docs')!r:'^20}\"\n", True),
    ("x = f\"{a:>{Path('docs').stat().st_size}}\"\n", True), ("x = f\"{value:'>10}\"\n", False), ("x = f\"{x:{'>'}10}\"\n", False),
    ("x = f\"{x!r:>10} docs {y:'<5}\"\n", False), ("x = f'{d[\"k\"]:\">5}'\n", False),
])
def test_quotes_in_an_f_string_format_spec_are_text(tmp_path, code, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "src", "a.py", code), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ('D=${D:-"docs"}\n', True), ("D=${D:-'docs'}\n", True), ('D="${D:-\'docs\'}"\n', True), ('D=${D:-"read docs now"}\n', False),
])
def test_the_default_word_of_a_parameter_expansion_is_read_without_its_quotes(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ('let x=1; cat <<EOF\nread "docs"\nEOF\n', False), ('let "x = 1 << 2"; find docs\n', True), ("let x=1<<SHIFT && find docs\n", True),
    ("echo a; let x=1<<SHIFT\nfind docs\n", False), ("let x=1<<EOF\nfind docs\nEOF\n", False),
    ("let x=1<<2 y=3<<4\nfind docs\n", False),
])
def test_an_unquoted_double_angle_after_let_is_a_here_document_like_any_other(tmp_path, text, marked):
    """Bash parses ``<<`` as a redirection BEFORE ``let`` sees its arguments, so ``let x=1<<EOF`` opens a here-document; only a
    quoted expression (``let "x = 1 << 2"``) or ``(( ))`` / ``$(( ))`` is arithmetic."""
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ('case "$mode" in docs) echo help ;; esac\n', False), ('case $x in docs|specs) echo ;; esac\n', False),
    ("for f in docs; do echo; done\n", True), ("for f in docs\ndo echo; done\n", True),
])
def test_the_first_case_pattern_after_in_is_a_pattern(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 10
# A quote inside a PLAIN scalar is text, a URL inside a QUOTED string is not cut at a comma, and a bash `let x=1<<2` is a heredoc.

@pytest.mark.parametrize("name, text", [
    ("c.yaml", 'description: Keep "docs" tidy\n'), ("c.yaml", "note: say 'docs' now\n"), ("c.yaml", '- see "docs" for details\n'),
    ("c.yaml", 'k: a "docs"\n'), ("c.yaml", 'description: Keep "docs" tidy # and "docs"\n'), ("c.yaml", 'k: &a see "docs"\n'),
    ("c.yaml", 'title: "x" and "docs"\n'), ("c.ini", '[x]\ndescription = Keep "docs" tidy\n'), ("c.cfg", "[x]\nnote = say 'docs'\n"),
])
def test_a_quoted_word_inside_a_plain_config_scalar_is_prose(tmp_path, name, text):
    assert tags(one_file(tmp_path, "configs", name, text, {"specs/s.md": "# s\n"}), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [
    ("c.yaml", 'root: "docs"\n'), ("c.yaml", "root: 'docs'\n"), ("c.yaml", 'roots: ["docs", x]\n'), ("c.yaml", '- "docs"\n'),
    ("c.yaml", 'root: &a "docs"\n'), ("c.yaml", '"docs": x\n'), ("c.yaml", '- {root: "docs"}\n'), ("c.yaml", 'roots: [\n  "docs",\n]\n'),
    ("c.yaml", '- &a "docs"\n'), ("c.ini", '[x]\nroot = "docs"\n'), ("c.toml", 'root = "docs"\n'), ("c.json", '{"root": "docs"}\n'),
    ("c.yaml", 'roots: [x,\n  "docs"]\n'), ("c.toml", 'description = "Keep \\"docs\\" tidy"\nroot = "docs"\n'),
])
def test_a_quoted_scalar_item_or_key_is_still_a_literal(tmp_path, name, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", name, text), "docs/A.md")


@pytest.mark.parametrize("name, text", [
    ("c.toml", 'url = "https://example.org/a,b/docs"\n'), ("c.yaml", 'url: "https://example.org/a,b/docs"\n'),
    ("c.yaml", "url: 'https://x.org/a,b/docs'\n"), ("c.ini", '[x]\nurl = "https://x.org/a,b/docs"\n'),
    ("c.yaml", 'urls: ["https://x.org/a,b/docs", "https://x.org/c]d/docs"]\n'), ("c.json", '{"u": "https://x.org/a,b/docs"}\n'),
    ("c.yaml", 'urls: ["https://x.org/a", "https://x.org/b,c/docs"]\n'),
])
def test_a_url_inside_a_quoted_config_string_is_not_cut_at_a_comma(tmp_path, name, text):
    assert tags(one_file(tmp_path, "configs", name, text), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ["--- [docs, specs]\n", "&roots !!seq [docs, specs]\n", "--- {root: docs, other: specs}\n",
                                  "--- &a [docs, specs]\n", "---   [docs, specs]\n"])
def test_a_document_marker_or_properties_before_a_top_level_collection_are_skipped(tmp_path, text):
    model = one_file(tmp_path, "configs", "c.yaml", text, {"specs/s.md": "# s\n"})
    assert idx.T_SUBTREE in tags(model, "docs/A.md") and idx.T_SUBTREE in tags(model, "specs/s.md")


@pytest.mark.parametrize("text", ['notes: [Keep "docs" tidy]\n', 'metadata: {description: Keep "docs" tidy}\n', '- [Keep "docs" tidy, x]\n',
                                  'k: [a, Keep "docs" tidy]\n', "{note: say 'docs' now}\n", 'roots: [\n  Keep "docs" tidy,\n]\n'])
def test_quoted_words_inside_a_plain_scalar_of_a_flow_collection_are_prose(tmp_path, text):
    assert tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("text", ['roots: ["docs", x]\n', 'roots: [x, "docs"]\n', '{root: "docs"}\n', 'roots: [&a "docs"]\n', '{"k":"docs"}\n',
                                  'roots: [\n  x,\n  "docs",\n]\n', 'roots: [ "docs" ]\n', "roots: ['docs']\n", '- ["docs"]\n',
                                  '"docs": [x]\n', 'roots: [!!str "docs"]\n'])
def test_a_quoted_item_of_a_flow_collection_is_a_literal(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")


@pytest.mark.parametrize("name, text", [
    ("a.py", 'URL = "https://example.org/a(b)/docs"\n'), ("a.py", "u = 'https://x.org/a;b/docs'\n"), ("a.py", 'u = "https://x.org/a|b/docs"\n'),
    ("c.yaml", 'url: "https://x.org/a(b)/docs"\n'), ("c.json", '{"u": "https://x.org/a(b);c/docs"}\n'),
    ("c.toml", 'url = "https://x.org/a(b)/docs"\n'), ("n.md", 'He said "see https://x.org/a(b)/docs now"\n'),
    ("a.py", "u = f'https://x.org/{a}(b)/docs'\n"), ("a.py", 'u = """https://x.org/a(b)/docs"""\n'),
])
def test_a_url_inside_a_quoted_string_runs_to_the_closing_quote(tmp_path, name, text):
    folder = "docs" if name == "n.md" else ("configs" if name.startswith("c.") else "src")
    model = one_file(tmp_path, folder, name, text, tracked=folder != "docs")
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "docs/B.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text, marked", [
    ("n.md", "[a](https://x.org/y)docs/A.md\n", True), ("n.md", "<https://x.org/y>docs/A.md\n", True),
    ("a.sh", "curl https://x.org/y;cat docs/A.md\n", True), ("a.sh", "curl https://x.org/y|cat docs/A.md\n", True),
    ("a.py", "# see https://x.org/a,b/docs/A.md for details\n", False), ("n.md", "see https://x.org/a,b/docs/A.md now\n", False),
    ("a.py", "# see https://x.org/a]b/docs/A.md for details\n", False), ("c.yaml", "url: https://x.org/a,b/docs/A.md\n", False),
    ("c.yaml", "urls: [https://x.org/a, b/docs/A.md]\n", True), ("c.ini", "[x]\nu = https://x.org/a,b/docs/A.md\n", True),
    ("a.py", "x = (https://x.org/y)\n# docs/A.md\n", True),
])
def test_an_unquoted_url_ends_at_the_boundary_of_its_file_type(tmp_path, name, text, marked):
    folder = "docs" if name == "n.md" else ("scripts" if name.endswith(".sh") else ("configs" if name.startswith("c.") else "src"))
    model = one_file(tmp_path, folder, name, text, tracked=folder != "docs")
    found = (idx.T_NOTES if name == "n.md" else idx.T_PATH) in tags(model, "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text", ['roots: ["owner\'s guide",https://example.org,docs]\n', 'roots: ["a\\"b",https://example.org,docs]\n',
                                  "roots: ['it''s',https://example.org,docs]\n", 'roots: ["x", \'y\', https://example.org,docs]\n'])
def test_a_quote_inside_another_kind_of_quote_does_not_hide_a_later_item(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")


@pytest.mark.parametrize("text, marked", [('--- "docs"\n', True), ("--- 'docs'\n", True), ('--- Keep "docs" tidy\n', False),
                                          ('---   "docs"\n', True), ('---\n', False)])
def test_a_document_marker_before_a_scalar_is_skipped(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ("cat <<EOF-END\nhello\nEOF-END\nfind docs\n", True), ('cat <<E"OF"\nhello\nEOF\nfind docs\n', True),
    ("cat <<'EOF-END'\nhello\nEOF-END\nfind docs\n", True), ('cat <<EOF-END\nread "docs" now\nEOF-END\n', False),
    ('cat <<E"OF"\nread "docs" now\nEOF\n', False), ("cat <<a.b\nread 'docs'\na.b\nfind docs\n", True), ("cat <<\\EOF\nread 'docs'\nEOF\n", False),
])
def test_a_here_document_delimiter_is_the_whole_word_with_its_quotes_removed(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ("x=$((\n1 << 2\n))\nfind docs\n", True), ("((\n x = 1 << 2\n))\nfind docs\n", True),
    ("x=$((\n1\n))\ncat <<EOF\nread docs\nEOF\n", False), ("x=$(( 1 +\n 2 << 3 ))\nfind docs\n", True),
])
def test_arithmetic_state_survives_a_newline(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 13
# URL blanking is a lexer too: it must know whether it stands in a string, in a plain scalar or in a flow item, and what it
# erases must stay erased (a blanked URL joined to "/docs" is still a URL; an f-string field next to a URL is still code).

@pytest.mark.parametrize("text, wanted", [
    ("roots: [owner's-notes,https://host,docs/A.md]\n", "A"), ("settings: {description: Use a \" character, root: docs}\n", "TREE"),
    ("roots: [it's \"x\",https://host,docs]\n", "TREE"), ("urls: [https://x.org/a(b), docs]\n", "TREE"),
    ("roots: [&a https://x.org/y, docs]\n", "TREE"), ("- \"docs\"\n", "TREE"), ("-  'docs'\n", "TREE"), ("- &a \"docs\"\n", "TREE"),
])
def test_url_blanking_knows_where_a_yaml_scalar_begins(tmp_path, text, wanted):
    model = one_file(tmp_path, "configs", "c.yaml", text)
    if wanted == "A":
        assert idx.T_PATH in tags(model, "docs/A.md")
    else:
        assert idx.T_SUBTREE in tags(model, "docs/A.md")


@pytest.mark.parametrize("name, text", [
    ("c.yaml", "url: https://graph.microsoft.com/v1.0/me/drive/root/search(q='docs')\n"), ("c.yaml", "url: https://x.org/a(b);c|d/docs\n"),
    ("c.yaml", "- https://x.org/a,b/docs\n"), ("c.yaml", "url: https://x.org/it's/docs\n"), ("c.ini", "[x]\nurl = https://x.org/a(b)/docs\n"),
    ("c.yaml", "url: https://x.org/a\"b/docs\n"),
    # a leftover of two components would be read as a local path: the URL must not stop before it
    ("c.yaml", "url: https://x.org/a,b/docs/A.md\n"), ("c.yaml", "- https://x.org/a]b/docs/A.md\n"),
    ("c.yaml", "url: https://x.org/it's/docs/A.md\n"), ("c.yaml", "url: https://x.org/a(b)/docs/A.md\n"),
    ("c.yaml", 'url: https://x.org/a"b/docs/A.md\n'), ("c.yaml", "url: https://x.org/{id}/docs/A.md\n"),
    ("c.yaml", "urls: [https://x.org/a(b)/docs/A.md, x]\n"), ("c.yaml", "urls: [https://x.org/it's/docs/A.md, x]\n"),
])
def test_an_unquoted_yaml_or_ini_url_runs_to_a_blank_in_block_context(tmp_path, name, text):
    assert tags(one_file(tmp_path, "configs", name, text), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("code", [
    'x = "a\\"b"; u = "https://x.org/a(b)/docs/A.md"\n', 'u = "https://x.org/o\'s/docs/A.md"\n', 'u = "https://x.org/a(b)/docs/A.md"\n',
    "u = 'https://x.org/say\"hi\"/docs/A.md'\n", 'u = "https://x.org/{id}/docs/A.md"\n', "u = 'https://x.org/a;b|c/docs/A.md'\n",
    'u = "https://x.org/a\\"b/docs/A.md"\n', "x = 'it\\'s'; u = 'https://x.org/a(b)/docs/A.md'\n",
    'u = "it\'s https://x.org/a(b)/docs/A.md"\n', "u = 'say \"hi\" https://x.org/a(b)/docs/A.md'\n",
])
def test_a_python_url_inside_a_string_runs_to_that_strings_closing_quote(tmp_path, code):
    assert tags(one_file(tmp_path, "src", "a.py", code), "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("code, marked", [
    ('url = f"https://example.org/{Path(\'docs/A.md\').read_text()}"\n', True), ('url = f\'https://x.org/{Path("docs/A.md").stat()}\'\n', True),
    ('url = rf"https://x.org/{Path(\'docs/A.md\').stat()}"\n', True), ('u = "https://x.org/{id}/docs/A.md"\n', False),
    ('u = f"https://x.org/{id}/y"\n', False),
])
def test_a_brace_ends_a_url_only_inside_an_f_string(tmp_path, code, marked):
    model = one_file(tmp_path, "src", "a.py", code)
    assert (idx.T_PATH in tags(model, "docs/A.md")) is marked


def test_a_path_with_a_blank_is_not_mistaken_for_a_blanked_url(tmp_path):
    files = {"docs/my reports/a.json": "{}", "src/a.py": "open('docs/my reports/a.json')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert idx.T_PATH in tags(model, "docs/my reports/a.json")


@pytest.mark.parametrize("code, wanted", [
    ('documentation_url = "https://example.org" + "/docs"\n', "NONE"), ('u = "https://x.org/" + "a" + "/docs"\n', "NONE"),
    ('u = "https://x.org" "/docs"\n', "NONE"), ('u = ("https://x.org"\n     + "/docs")\n', "NONE"),
    ('url = f"https://example.org/{Path(\'docs/A.md\').read_text().strip()}"\n', "A"), ('u = f"https://x.org/{a}/docs"\n', "NONE"),
    ('u = "https://x.org/{id}/docs"\n', "NONE"), ('p = "docs/" + "A.md"\n', "A"),
])
def test_a_blanked_url_stays_a_url_when_literals_are_joined_and_an_f_string_field_stays_code(tmp_path, code, wanted):
    model = one_file(tmp_path, "src", "a.py", code)
    if wanted == "A":
        assert idx.T_PATH in tags(model, "docs/A.md") and tags(model, "docs/B.md") == {idx.T_NONE}
    else:
        assert tags(model, "docs/A.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text, marked", [
    ("c.yaml", "sphinx:root: docs\n", True), ("c.yaml", "a:b: c\n", False), ("c.yaml", "key: value: docs\n", False),
    ("c.toml", 'a:b = "docs"\n', True), ("c.yaml", "a/b: docs\n", True), ("c.ini", "[x]\nroot = docs\n", True),
])
def test_a_yaml_mapping_separator_is_a_colon_followed_by_a_blank(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "configs", name, text), "docs/A.md")
    assert found is marked


def test_a_property_longer_than_the_old_window_still_begins_an_item(tmp_path):
    model = one_file(tmp_path, "configs", "c.yaml", "roots: [&" + "a" * 65 + ' "docs"]\n')
    assert idx.T_SUBTREE in tags(model, "docs/A.md")


@pytest.mark.parametrize("name, text", [
    ("a.py", 'URL = "https://host/say\\"hi\\"/docs"\n'), ("c.yaml", 'url: "https://host/say\\"hi\\"/docs"\n'),
    ("c.json", '{"u": "https://host/say\\"hi\\"/docs"}\n'),
])
def test_an_escaped_quote_does_not_end_a_url_in_a_string(tmp_path, name, text):
    folder = "configs" if name.startswith("c.") else "src"
    assert tags(one_file(tmp_path, folder, name, text), "docs/A.md") == {idx.T_NONE}


def test_an_escaped_second_here_document_operator_on_a_line_is_text(tmp_path):
    for text, marked in (("cat <<A \\<<B\nbody\nA\nfind docs\n", True), ("cat <<A <<B\nx\nA\ny\nB\nfind docs\n", True),
                         ("cat <<A \\<<B\nbody\nA\nread docs\n", True)):
        found = idx.T_SUBTREE in tags(one_file(tmp_path / str(hash(text)), "scripts", "a.sh", text), "docs/A.md")
        assert found is marked


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 12

@pytest.mark.parametrize("name, text", [
    ("a.py", 'URL = "https://example.org/owner\'s/docs"\n'), ("c.yaml", 'url: "https://x.org/owner\'s/docs"\n'),
    ("c.json", '{"u": "https://x.org/o\'s/docs"}\n'), ("c.toml", 'url = "https://x.org/o\'s/docs"\n'),
    ("a.py", "u = 'https://x.org/say\"hi\"/docs'\n"), ("c.ini", '[x]\nu = "https://x.org/o\'s/docs"\n'),
])
def test_a_url_runs_to_the_closing_quote_of_ITS_string_not_to_any_quote(tmp_path, name, text):
    folder = "configs" if name.startswith("c.") else "src"
    model = one_file(tmp_path, folder, name, text)
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "docs/B.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text, marked", [
    ("c.yaml", "roots: [owner's-notes, docs]\n", True), ("c.yaml", "settings: {description: Don't edit, root: docs}\n", True),
    ("c.yaml", 'note: Don\'t # "docs"\n', False), ("c.ini", "[x]\nroots = owner's, docs\n", True),
    ("c.yaml", "roots: ['quoted', docs]\n", True), ("c.yaml", 'roots: [it\'s "docs", x]\n', False),
    ("c.yaml", "- Don't edit docs\n", False), ("c.yaml", "roots: [\"a\",'b', docs]\n", True), ("c.yaml", "k: it's\nroot: docs\n", True),
    ("c.yaml", "k: [x, 'it''s', docs]\n", True),
])
def test_an_apostrophe_inside_a_plain_config_scalar_opens_no_string(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "configs", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ('notes: [Do not use !include "docs" here]\n', False), ('roots: [use &a "docs"]\n', False), ('roots: [x, &a "docs"]\n', True),
    ('roots: [!!str "docs"]\n', True), ('roots: [x, !!str &a "docs"]\n', True), ('{k: &a "docs"}\n', True),
])
def test_yaml_properties_count_only_at_the_start_of_a_flow_item(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("name, text, marked", [
    ("c.yaml", 'url: "https:\\/\\/example.org\\/docs"\n', False), ("c.json", '{"u": "https:\\/\\/example.org\\/docs"}\n', False),
    ("c.json", '{"u": "https\\u003a//example.org/docs"}\n', False), ("c.yaml", 'url: "https:\\x2f\\x2fexample.org/docs"\n', False),
    ("c.json", '{"u": "see https:\\/\\/example.org\\/docs now"}\n', False), ("c.json", '{"u": "sqlite:\\/\\/\\/docs\\/A.md"}\n', True),
    ("c.json", '{"u": "file:\\/\\/\\/C:\\/repo\\/docs\\/A.md"}\n', True),
])
def test_a_url_made_by_decoding_escapes_is_still_a_url(tmp_path, name, text, marked):
    model = one_file(tmp_path, "configs", name, text)
    assert bool(tags(model, "docs/A.md") & {idx.T_PATH, idx.T_SUBTREE}) is marked


@pytest.mark.parametrize("name, text", [
    ("a.sh", "echo ok\n# <<EOF\nfind docs\n"), ("a.ps1", "Write-Host hi\n# <# ignored\nGet-ChildItem docs\n"),
    ("a.sh", "echo ok\n  # <<EOF\nfind docs\n"),
])
def test_a_comment_at_the_start_of_a_line_hides_its_block_opener(tmp_path, name, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")


@pytest.mark.parametrize("text, marked", [
    ("echo \\<<EOF\nfind docs\n", True), ("echo \\\\<<EOF\nread docs\nEOF\n", False), ("echo a\\ <<EOF\nread docs\nEOF\n", False),
])
def test_an_escaped_here_document_operator_is_text(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text", [
    "cat <<'EOF\\END'\nread docs\nEOF\\END\nfind docs\n", 'cat <<"EOF\\"END"\nread docs\nEOF"END\nfind docs\n',
    "cat <<E\\OF\nread docs\nEOF\nfind docs\n", 'cat <<"E\\$OF"\nread docs\nE$OF\nfind docs\n',
])
def test_a_here_document_delimiter_is_unquoted_the_way_the_shell_does(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")


@pytest.mark.parametrize("text, marked", [
    ('echo "\\`read docs\\`"\n', False), ("x=`ls docs`\n", True), ('echo "\\\\`ls docs`"\n', True), ('echo "a \\` b `ls docs`"\n', True),
])
def test_an_escaped_backtick_is_not_a_substitution(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("name, text, marked", [
    ("a.sh", 'DOCS_DIR="${DOCS_DIR:-docs}"\n', True), ("a.sh", 'cd "${1:-docs}"\n', True), ("a.sh", 'x=${D-docs}\n', True),
    ("a.sh", 'D="${D:=./docs}"\n', True), ("a.sh", 'D="${D:-$PWD/docs}"\n', True),
    ("a.sh", "D='${D:-docs}'\n", False), ("a.sh", 'echo "${D:-nothing}"\n', False), ("a.sh", 'echo "${D:-read docs now}"\n', False),
    ("a.bat", 'set "DOCS=docs"\n', True), ("a.bat", 'set DOCS=docs\n', True), ("a.bat", 'set /p "X=read docs"\n', False),
    ("a.bat", 'echo & set "D=.\\docs"\n', True),
    ("a.bat", "for /f %%i in ('dir /b docs') do echo %%i\n", True), ("a.bat", "for /f \"usebackq\" %%i in (`dir docs`) do echo %%i\n", True),
    ("a.bat", 'for %%i in (a b c) do echo "%%i"\n', False),
])
def test_a_default_or_assigned_word_in_a_quoted_shell_assignment_is_a_path_word(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("name, text, marked", [
    ("a.sh", 'case "$1" in\n  docs) echo b ;;\n  *) ;;\nesac\n', False), ("a.sh", "case $x in\n  a|docs) echo ;;\nesac\n", False),
    ("a.sh", "case $x in\n  specs) ;; docs) echo ;;\nesac\n", False), ("a.sh", "case $x in a) ls docs ;; esac\n", True),
    ("a.sh", "x=$(echo docs)\n", True), ("a.ps1", "(Get-ChildItem docs)\n", True), ("a.sh", "(ls docs)\n", True),
])
def test_a_case_pattern_is_not_an_argument(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text", ["echo $# docs\n", "echo ${x#y} docs\n", "echo a#b docs\n"])
def test_a_hash_inside_a_word_is_still_not_a_comment(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")


@pytest.mark.parametrize("text, marked", [
    ("echo done & rem keep docs tidy\n", False), ("echo done && rem docs\n", False), ("echo a | rem docs\n", False),
    ("(rem docs\n)\n", False), ("@rem docs\n", False), ("echo \"a & rem b\" & dir docs\n", True),
    ("echo done & dir docs\n", True), ("echo remember docs\n", True),
])
def test_a_batch_rem_after_a_command_separator_is_a_comment(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.bat", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("name, text, marked", [
    ("a.ps1", 'Write-Host "#" <# "docs" #>\n', False), ("a.ps1", "Write-Host '#'; <# keep docs tidy #>\n", False),
    ("a.ps1", "Write-Host '#' ; @\"\nread docs\n\"@\n", False),
    ("a.sh", 'echo "don\'t"; cat <<EOF\nread docs now\nEOF\n', False), ("a.sh", "echo '#'; cat <<EOF\nread docs\nEOF\n", False),
    ("a.sh", "mask=$((1 << SHIFT))\nfind docs\n", True), ("a.sh", "(( x = 1 << SHIFT ))\nfind docs\n", True),
    ("a.sh", "echo $(( 2 << N )) ; find docs\n", True),
    ("a.sh", "cat <<EOF\nbody\nEOF\nfind docs\n", True),
    ("a.sh", 'echo \\"; cat <<EOF\nread docs\nEOF\n', False), ("a.ps1", 'Write-Host `"; @"\nread docs\n"@\n', False),
])
def test_a_block_opener_is_live_only_in_code_and_outside_arithmetic(tmp_path, name, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", name, text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ('cat <<EOF\n EOF\nread "docs" now\nEOF\n', False), ('cat <<EOF\nEOF \nread "docs" now\nEOF\n', False),
    ('cat <<-EOF\n  EOF\nread "docs" now\n\tEOF\n', False), ('cat <<-EOF\n\tread "docs"\n\tEOF\nls docs\n', True),
    ('cat <<EOF\nEOF\nls docs\n', True),
])
def test_a_heredoc_ends_at_its_exact_delimiter_line(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text, marked", [
    ('cat <<FIRST <<SECOND\nx\nFIRST\nread "docs" now\nSECOND\n', False),
    ('cat <<A <<B\nx\nA\ny\nB\nls docs\n', True), ('diff <(cat <<A) <(cat <<B)\nx\nA\nread "docs"\nB\n', False),
])
def test_every_heredoc_of_a_command_line_is_consumed_in_order(tmp_path, text, marked):
    found = idx.T_SUBTREE in tags(one_file(tmp_path, "scripts", "a.sh", text), "docs/A.md")
    assert found is marked


@pytest.mark.parametrize("text", ['- "root": docs\n', "- 'root': docs\n", '- "root": "docs"\n', '- &a "root": docs\n'])
def test_a_quoted_key_in_a_yaml_sequence_mapping_does_not_hide_its_value(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")


@pytest.mark.parametrize("text", ['roots: {"contract":docs}\n', 'roots: {"contract": docs}\n', "roots: {'contract':docs, x: y}\n"])
def test_a_compact_yaml_flow_mapping_separator_is_found(tmp_path, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")


@pytest.mark.parametrize("text", ["description: See [docs] for details\n", "note: a [docs] b\n", "- note: see [docs]\n",
                                  "description: Notes {docs}\n", "key: x [docs, specs] y\n"])
def test_brackets_inside_a_yaml_plain_scalar_are_not_a_flow_collection(tmp_path, text):
    model = one_file(tmp_path, "configs", "c.yaml", text, {"specs/s.md": "# s\n"})
    assert tags(model, "docs/A.md") == {idx.T_NONE} and tags(model, "specs/s.md") == {idx.T_NONE}


@pytest.mark.parametrize("name, text", [
    ("c.yaml", "roots: [https://example.com,docs]\n"), ("c.yaml", "roots: [https://example.com/a, docs]\n"),
    ("c.ini", "[x]\nroots = https://example.com,docs\n"), ("c.json", '{"a": "https://x.org/y", "b": "docs"}\n'),
    ("c.yaml", "roots: {a: https://example.com,b: docs}\n"),
])
def test_a_url_ends_at_the_boundary_of_its_collection_item(tmp_path, name, text):
    assert idx.T_SUBTREE in tags(one_file(tmp_path, "configs", name, text), "docs/A.md")


# ---------------------------------------------------------------------------------------- found by sol's review round 4

def test_standalone_dots_in_an_odd_name_do_not_break_the_token_sequence():
    model, scanner = synthetic_scanner(["docs/a . b.md", "docs/. Notes.md"])
    scanner.scan("code", "README.md", "See a . b.md.\nOpen . Notes.md now\n")
    assert idx.T_NAME in model.targets["docs/a . b.md"].tags and idx.T_NAME in model.targets["docs/. Notes.md"].tags
    assert [e.line for e in model.targets["docs/. Notes.md"].evidence["code"]] == [2]


def test_odd_name_confirmations_are_budgeted_and_exhaustion_errs_toward_marking():
    import time
    names = ["docs/a" + " " * i + "b.md" for i in range(1, 201)]
    model, scanner = synthetic_scanner(names)
    started = time.perf_counter()
    scanner.scan("code", "src/t.txt", "a,b.md\n" * 400_000)
    assert time.perf_counter() - started < 8
    assert all(idx.T_NAME in model.targets[name].tags for name in names)             # over-marked, never under-marked
    assert model.gaps["odd_names_overmarked"] == 200 and "without confirmation" in idx.render(model, TODAY).lower()


def test_below_the_budget_each_odd_name_is_still_confirmed_exactly():
    model, scanner = synthetic_scanner(["docs/a b.md", "docs/a  b.md", "docs/a   b.md"])
    scanner.scan("code", "src/t.txt", "see a  b.md here\n")
    assert [p for p, t in model.targets.items() if idx.T_NAME in t.tags] == ["docs/a  b.md"]
    assert model.gaps["odd_names_overmarked"] == 0


def test_mime_rules_apply_to_a_bundle_not_to_the_keys_of_a_json_payload(tmp_path):
    outputs = [{"output_type": "display_data", "metadata": {},
                "data": {"application/json": {"image/path": "docs/sub/A.md", "text/plain": ["docs/sub/B.md", "docs/sub/C.md"]},
                         "image/png": ["docs/sub/", "other.md"]}}]
    names = ["docs/sub/A.md", "docs/sub/B.md", "docs/sub/C.md", "docs/sub/other.md"]
    files = {**{n: "# x\n" for n in names}, "notebooks/n.ipynb": _notebook({"cell_type": "code", "source": ["1\n"], "outputs": outputs})}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert {n for n in names if idx.T_PATH in tags(model, n)} == set(names[:3])      # the png bundle entry is still skipped


def test_suffix_queries_do_not_scan_a_bucket_of_entries_sharing_a_last_component():
    import time
    model, scanner = synthetic_scanner(f"docs/q{i:06d}/common/x.md" for i in range(16_000))
    text = "".join(f"p = base / 'p{i:06d}/common' / name\n" for i in range(16_000))
    started = time.perf_counter()
    scanner.scan("code", "src/t.py", text)
    assert time.perf_counter() - started < 8
    assert not any(idx.T_UNRESOLVED in t.tags for t in model.targets.values())     # nothing ends in an absent suffix


def test_a_suffix_chain_marks_files_and_directories_that_end_in_it():
    model, scanner = synthetic_scanner(["specs/a/b/x.json", "specs/a/b/y.json", "docs/a/b/c/d.md", "docs/a/bb/c/e.md"])
    scanner.scan("code", "src/t.py", "p = base / 'a/b/x.json'\nq = base / 'b/c' / name\n")
    flagged = {p for p, t in model.targets.items() if idx.T_UNRESOLVED in t.tags}
    assert flagged == {"specs/a/b/x.json", "docs/a/b/c/d.md"}              # 'bb/c' is not 'b/c'; y.json was not named


def test_a_construction_excerpt_is_a_window_not_the_line():
    """With the listing capped, copying the whole line per listed construction costs 2,000 copies of a 3 MB line."""
    import time
    model, scanner = synthetic_scanner(["docs/q1.md"])
    text = " " * 3_000_000 + "".join(f"x = 'docs/p{i}' + name; " for i in range(idx.MAX_CONSTRUCTIONS)) + "\n"
    started = time.perf_counter()
    scanner.scan("code", "src/t.py", text)
    assert time.perf_counter() - started < 1.0                  # measured: 0.18 s as written, 2.5 s if the excerpt is the line
    assert len(model.constructions) == idx.MAX_CONSTRUCTIONS


def test_the_words_of_an_odd_name_split_by_other_punctuation_are_not_a_mention():
    model, scanner = synthetic_scanner(["docs/My Notes.md"])
    scanner.scan("code", "README.md", "Also (my, notes.md) and my-notes.md\n")
    assert idx.T_NAME not in model.targets["docs/My Notes.md"].tags


def test_an_odd_name_must_be_the_whole_name_in_the_text_not_just_its_words_in_order():
    """Same words, same order, same length, boundaries intact: only a different separator tells the text from the name."""
    model, scanner = synthetic_scanner(["docs/My Notes.md"])
    scanner.scan("code", "src/one.txt", "My,Notes.md\n")
    assert idx.T_NAME not in model.targets["docs/My Notes.md"].tags
    scanner.scan("code", "src/two.txt", "My Notes.md\n")
    assert [e.source for e in model.targets["docs/My Notes.md"].evidence["code"]] == ["src/two.txt"]


def test_an_odd_name_that_begins_with_punctuation_does_not_start_inside_a_longer_word():
    model, scanner = synthetic_scanner(["docs/+a b.md"])
    scanner.scan("code", "src/one.txt", "x+a b.md\n")                       # the '+' belongs to a longer word here
    assert idx.T_NAME not in model.targets["docs/+a b.md"].tags
    scanner.scan("code", "src/two.txt", "(+a b.md)\n")
    assert [e.source for e in model.targets["docs/+a b.md"].evidence["code"]] == ["src/two.txt"]


def test_an_odd_name_that_ends_with_punctuation_does_not_end_inside_a_longer_word():
    model, scanner = synthetic_scanner(["docs/a b+"])
    scanner.scan("code", "src/one.txt", "a b+c\n")                           # the name would continue into 'c'
    assert idx.T_NAME not in model.targets["docs/a b+"].tags
    scanner.scan("code", "src/two.txt", "a b+ c\n")
    assert [e.source for e in model.targets["docs/a b+"].evidence["code"]] == ["src/two.txt"]


def test_an_odd_name_is_found_case_insensitively_on_a_later_line_after_decoys():
    model, scanner = synthetic_scanner(["docs/My Notes.md"])
    scanner.scan("code", "README.md", "my notes.mdx is a decoy\nxmy notes.md is another\nthe file MY NOTES.MD is here\n")
    assert [e.line for e in model.targets["docs/My Notes.md"].evidence["code"]] == [3]


def test_suffix_tables_do_not_grow_with_the_square_of_the_directory_depth():
    import tracemalloc
    deep = "docs/" + "/".join(f"d{i:04d}" for i in range(400)) + "/x.md"
    tracemalloc.start()
    synthetic_scanner([deep, "docs/A.md"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, peak


def test_a_chain_suffix_still_finds_directories_and_files_by_their_trailing_components():
    model, scanner = synthetic_scanner(["specs/a/benchmarks/x.json", "specs/b/benchmarks/x.json", "specs/c/other/x.json",
                                        "specs/d/notbenchmarks/x.json", "specs/e/benchmarks2/x.json", "docs/benchmarks/y.md"])
    scanner.scan("code", "src/t.py", "p = base / 'benchmarks' / 'x.json'\nq = base / 'benchmarks'\n")
    unresolved = {p for p, t in model.targets.items() if idx.T_UNRESOLVED in t.tags}
    assert unresolved == {"specs/a/benchmarks/x.json", "specs/b/benchmarks/x.json", "docs/benchmarks/y.md"}


@pytest.mark.parametrize("output, expected, lines_together", [
    ({"output_type": "stream", "name": "stdout", "text": ["docs/sub/A.", "md"]}, ["docs/sub/A.md"], True),
    ({"output_type": "error", "traceback": ["docs/sub/B.md", "docs/sub/C.md"]}, ["docs/sub/B.md", "docs/sub/C.md"], True),
    ({"output_type": "display_data", "data": {"application/json": {"files": ["docs/sub/D.md", "docs/sub/E.md"]}}},
     ["docs/sub/D.md", "docs/sub/E.md"], True),
    ({"output_type": "display_data", "data": {"application/json": {"docs/sub/F.md": "abc123"}}}, ["docs/sub/F.md"], True),
    ({"output_type": "display_data", "data": {"text/plain": ["docs/sub/G", ".md"]}}, ["docs/sub/G.md"], True),
])
def test_notebook_output_strings_are_reassembled_the_way_nbformat_defines_them(tmp_path, output, expected, lines_together):
    names = {"docs/sub/A.md", "docs/sub/B.md", "docs/sub/C.md", "docs/sub/D.md", "docs/sub/E.md", "docs/sub/F.md", "docs/sub/G.md"}
    notebook = _notebook({"cell_type": "code", "source": ["1\n"], "outputs": [output]})
    files = {**{n: "# x\n" for n in names}, "notebooks/n.ipynb": notebook}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    found = {n for n in names if idx.T_PATH in tags(model, n)}
    assert found == set(expected), found
    assert model.gaps["notebook_cells_malformed"] == 0


def test_many_odd_names_cost_nothing_in_a_source_that_lacks_their_words(tmp_path):
    """Names the word pattern cannot hold (spaces) are searched for only where every word of the name occurs."""
    files = {f"docs/odd name {i:04d}.md": "# o\n" for i in range(1000)}
    files["src/blob.txt"] = "z" * 3_000_000
    repo = make_repo(tmp_path / "r", files, tracked=["src/blob.txt"])
    done = _run_check(repo, timeout=20)
    assert done.returncode == 0, done.stderr[-300:]


def test_a_version_3_notebook_is_decoded_not_scanned_as_a_script(tmp_path):
    import json
    notebook = json.dumps({"nbformat": 3, "worksheets": [{"cells": [
        {"cell_type": "code", "language": "python", "outputs": [],
         "input": ["from pathlib import Path\n", "root = Path('specs')\n", "import reference_math\n"]},
        {"cell_type": "markdown", "source": ["docs"]},
        {"cell_type": "heading", "level": 1, "source": ["specs"]}]}]}, indent=1)
    files = {"docs/A.md": "# a\n", "specs/s.json": "{}", "specs/a/reference_math.py": "X = 1\n", "notebooks/n.ipynb": notebook}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert idx.T_SUBTREE in tags(model, "specs/s.json") and idx.T_IMPORT in tags(model, "specs/a/reference_math.py")
    assert tags(model, "docs/A.md") == {idx.T_NONE}                                    # a prose cell naming a tree is not code
    assert model.gaps["notebooks_unparsed"] == 0


def test_every_textual_output_representation_is_scanned_and_binary_payloads_are_not(tmp_path):
    import json
    outputs = [{"output_type": "display_data", "metadata": {},
                "data": {"text/markdown": "see docs/sub/a.md", "text/html": ["<a href='docs/sub/b.md'>b</a>"],
                         "application/json": {"path": "docs/sub/c.md"}, "image/png": "docs/sub/other.md"}},
               {"output_type": "error", "ename": "E", "evalue": "no docs/sub/d.md", "traceback": ["Traceback docs/sub/e.md\n"]}]
    notebook = _notebook({"cell_type": "code", "source": ["1\n"], "outputs": outputs})
    files = {**{f"docs/sub/{n}.md": "# x\n" for n in "abcde"}, "docs/sub/other.md": "# o\n", "notebooks/n.ipynb": notebook}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    for name in "abcde":
        assert idx.T_PATH in tags(model, f"docs/sub/{name}.md"), name
        assert {e.line for e in model.targets[f"docs/sub/{name}.md"].evidence["code"]} == {1}
    assert tags(model, "docs/sub/other.md") == {idx.T_NONE}                           # an image/* payload is skipped


def test_a_deeply_nested_notebook_falls_back_to_raw_text_instead_of_crashing(tmp_path):
    nested = '{"cells": [], "metadata": {"x": ' + "[" * 1200 + "]" * 1200 + '}, "note": "docs/A.md"}'
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "notebooks/n.ipynb": nested}, tracked=["notebooks/n.ipynb"]))
    assert model.gaps["notebooks_unparsed"] == 1 and idx.T_PATH in tags(model, "docs/A.md")


@pytest.mark.parametrize("output", [{"output_type": "stream", "name": "stdout", "text": [17]},
                                    {"output_type": "stream", "name": "stdout", "text": {"a": 1}},
                                    {"output_type": "error", "traceback": [3, "x"]}])
def test_a_malformed_output_payload_is_counted_once_per_cell(tmp_path, output):
    notebook = _notebook({"cell_type": "code", "source": ["1\n"], "outputs": [output, dict(output)]})
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "notebooks/n.ipynb": notebook}, tracked=["notebooks/n.ipynb"]))
    assert model.gaps["notebook_cells_malformed"] == 1


def test_a_malformed_notebook_cell_is_counted_and_never_fatal(tmp_path):
    notebook = ('{"cells": [{"cell_type": "code", "source": [17]}, {"cell_type": "code", "source": ["open(\'docs/A.md\')\\n"]}, '
                '"junk", {"cell_type": "code", "source": {"a": 1}}]}')
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "notebooks/n.ipynb": notebook}, tracked=["notebooks/n.ipynb"]))
    assert idx.T_PATH in tags(model, "docs/A.md")                                      # the well-formed cell is still read
    assert model.gaps["notebook_cells_malformed"] == 3
    assert "malformed" in idx.render(model, TODAY)


def test_an_unparsable_notebook_is_scanned_as_raw_text_and_counted(tmp_path):
    files = {**BASE, "notebooks/n.ipynb": "{ not json  open('docs/A.md')"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert model.gaps["notebooks_unparsed"] == 1 and idx.T_PATH in tags(model, "docs/A.md")


def test_the_page_says_that_files_named_only_inside_linked_campaigns_may_look_weaker(tmp_path):
    text = idx.render(idx.build_model(make_repo(tmp_path / "r", dict(BASE))), TODAY)
    outputs_line = next(line for line in text.splitlines() if line.startswith("- outputs records"))
    assert "linked campaign directories not followed" in outputs_line and "weaker tier" in outputs_line


def test_a_tracked_file_below_a_junctioned_ancestor_is_never_read_through_it(tmp_path):
    repo = make_repo(tmp_path / "r", {**BASE, "src/alias/consumer.py": "x = 1\n"}, tracked=["src/alias/consumer.py"])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "consumer.py").write_text("open('docs/A.md')\n")
    (repo / "src" / "alias" / "consumer.py").unlink()
    os.rmdir(repo / "src" / "alias")
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(repo / "src" / "alias"), str(outside)], capture_output=True, text=True)
    if done.returncode:
        pytest.skip("cannot create a junction here")
    try:
        model = idx.build_model(repo)
        assert tags(model, "docs/A.md") == {idx.T_NONE}
        assert any("alias" in item for item in model.gaps["links_skipped"])
    finally:
        os.rmdir(repo / "src" / "alias")


BOM = chr(0xFEFF)                                   # a code point, never a literal: a BOM in source is invisible


@pytest.mark.parametrize("first_line", [" {M} ", "{M} ", "{B}{M}", "{M}x", "x{M}"])
def test_only_an_exactly_marked_file_is_ever_overwritten(tmp_path, first_line):
    mine = first_line.replace("{M}", idx.MARKER).replace("{B}", BOM) + "\nhand-written\n"
    repo = make_repo(tmp_path / "r", {**BASE, "docs/INDEX.md": mine})
    (repo / "docs" / "INDEX.md").write_bytes(mine.encode("utf-8"))
    done = cli(repo)
    assert done.returncode == 2, done.stderr
    assert (repo / "docs" / "INDEX.md").read_bytes() == mine.encode("utf-8")


def test_a_crlf_terminated_marker_line_is_still_our_file(tmp_path):
    repo = make_repo(tmp_path / "r", {**BASE})
    (repo / "docs" / "INDEX.md").write_bytes((idx.MARKER + "\r\nold\r\n").encode("utf-8"))
    done = cli(repo)
    assert done.returncode == 0, done.stderr
    assert (repo / "docs" / "INDEX.md").read_bytes().startswith(idx.MARKER.encode("utf-8") + b"\n")


def test_a_bom_prefixed_copy_of_the_index_is_still_excluded_as_a_source(tmp_path):
    """A copy saved by an editor or a PowerShell redirect gains a BOM: it must not become a source of references."""
    copy = BOM + idx.MARKER + "\n- docs/A.md\n"
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "docs/copy.md": copy}))
    assert model.gaps["generated_marker_files"] == 1 and "docs/copy.md" not in model.targets
    assert tags(model, "docs/A.md") == {idx.T_NONE}


def test_repeated_tree_components_in_one_token_stay_linear(tmp_path):
    """'docs/' * n is one token with n candidate starts; each used to re-slice and rescan the rest (quadratic)."""
    repo = make_repo(tmp_path / "r", {**BASE, "src/rep.py": "x = '" + "docs/" * 30_000 + "A.md'\n"}, tracked=["src/rep.py"])
    done = subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(repo), "--today", TODAY, "--check"],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr[-300:]
    assert "rep.py:1 (path)" in done.stdout                                        # the real docs/A.md suffix was still found


def test_a_tracked_markdown_document_inside_the_trees_can_be_a_hub(tmp_path):
    files = {"specs/A.md": "# a\n", "specs/B.md": "# b\n", "specs/C.md": "# c\n", "specs/hub.md": "- A.md\n- B.md\n- C.md\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["specs/hub.md"]))
    assert model.hubs == [("specs/hub.md", 3)]


def test_most_named_by_tracked_sources_lists_documents_not_evidence_manifests(tmp_path):
    files = {"docs/A.md": "# a\n", "reference/evidence/c/manifest.json": "{}",
             "src/a.py": "open('docs/A.md')\nopen('reference/evidence/c/manifest.json')\n"}
    text = idx.render(idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"])), TODAY)
    section = text.split("Documents most often named by tracked sources")[1].split("## ")[0]
    assert "A.md" in section and "manifest.json" not in section


# ---------------------------------------------------------------------------------------- found by differential fuzzing
# A generator serialized random nested structures with ONE labelled value (the docs tree, docs/A.md, or neither) in random YAML
# styles; PyYAML confirmed that each document loads back to the structure it came from, and the index was asked what it made of
# it. Every miss below is a style the hand-picked shapes of the review rounds had not covered.

def yaml_marks(tmp_path, text):
    return tags(one_file(tmp_path, "configs", "c.yaml", text), "docs/A.md")


@pytest.mark.parametrize("text", [
    "k: {name: docs/A.md}\n", "- {root: docs/A.md}\n", "k: {a: x, name: docs/A.md}\n", "k: {a: {b: docs/A.md}}\n",
    "k: [{name: docs/A.md}]\n", "k: {a: [docs/A.md]}\n", "k: {name: docs\\A.md}\n", "k: {name: ./docs/A.md}\n",
    "k: {name: &a docs/A.md}\n", "k: {\n  name: docs/A.md\n}\n", "k: {name: docs/A.md }\n", "k: {a: [x, docs/A.md]}\n",
    "k: [docs/A.md]\n", "k: {a: [docs/A.md]}}\n",
])
def test_an_unquoted_path_that_closes_a_flow_mapping_is_still_a_path(tmp_path, text):
    """``docs/A.md}`` was read as one run with a template character in it: the path became an UNRESOLVED possibility."""
    assert idx.T_PATH in yaml_marks(tmp_path, text)


@pytest.mark.parametrize("text, wanted", [
    ("k: {name: docs/{id}.md}\n", "NONE"), ("k: docs/{a,b}.md\n", "NONE"), ("k: {name: docs/x[0-9].md}\n", "NONE"),
    ("k: {name: docs/{id}}\n", "UNRESOLVED"), ("k: docs/{id}\n", "UNRESOLVED"),
])
def test_a_closing_bracket_with_an_opener_in_the_same_run_stays_part_of_the_path(tmp_path, text, wanted):
    """Only an UNBALANCED closer is dropped: ``docs/{id}`` is a template, and a glob class ``x[0-9]`` is a glob."""
    found = yaml_marks(tmp_path, text)
    if wanted == "NONE":
        assert idx.T_PATH not in found
    else:
        assert idx.T_UNRESOLVED in found and idx.T_PATH not in found


@pytest.mark.parametrize("header", ["|", ">", "|-", ">-", "|+", ">+", "|2", ">2", "|-2", ">-2", "|2-", ">2-", "|2+", ">2+", "|+2"])
def test_a_block_scalar_header_may_give_the_indent_and_the_chomp_in_either_order(tmp_path, header):
    """YAML 1.2 allows ``|2-`` as well as ``|-2``: the body of ``root: |2-`` is the value, so it can name the tree."""
    assert idx.T_SUBTREE in yaml_marks(tmp_path, f"root: {header}\n  docs\n")
    assert idx.T_SUBTREE in yaml_marks(tmp_path / "x", f"- {header}\n  docs\n")


@pytest.mark.parametrize("text, wanted", [
    ("- sphinx:root: docs\n", "TREE"), ("- sphinx:root: docs/A.md\n", "A"), ("- sphinx:root: [docs, x]\n", "TREE"),
    ("- sphinx:root: |-\n    docs\n", "TREE"), ("- a:b:c: ./docs/\n", "TREE"), ("- - sphinx:root: docs\n", "TREE"),
    ("k:\n  - x:y: docs\n    z: 1\n", "TREE"), ("- &a sphinx:root: docs\n", "TREE"), ("- 'a:b': docs\n", "TREE"),
    ("- sphinx:root: |-\n    ls docs\n  env: x\n", "NONE"), ("- sphinx:root: see docs\n", "NONE"), ("- note:docs\n", "NONE"),
    ("- a:b\n", "NONE"),
])
def test_a_plain_key_may_hold_a_colon_that_is_not_followed_by_a_blank(tmp_path, text, wanted):
    """``sphinx:root`` is ONE key: a colon ends a YAML key only when a blank follows. On a ``- `` line the key pattern stopped
    at the first colon, so the whole ``sphinx:root: |-`` was the value and nothing below it was read."""
    found = yaml_marks(tmp_path, text)
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


def test_a_balanced_template_or_glob_is_listed_with_its_closing_bracket(tmp_path):
    """The construction list quotes the run it read: trimming a closer that HAS its opener would show ``docs/{id`` there."""
    model = one_file(tmp_path, "configs", "c.yaml", "k: docs/{id}\nj: {name: docs/x[0-9]}\n")
    section = idx.render(model, TODAY).split("## Constructions with a non-literal tail")[1]
    assert "`docs/{id}`" in section and "`docs/x[0-9]`" in section


@pytest.mark.parametrize("text, wanted", [
    ("- - docs\n", "TREE"), ("- - 'docs'\n", "TREE"), ('- - "./docs/"\n', "TREE"), ("- - [docs]\n", "TREE"), ("- - docs/A.md\n", "A"),
    ("- - - docs\n", "TREE"), ("-   - docs\n", "TREE"), ("k:\n  - - docs\n    - x\n", "TREE"), ("- - key: docs\n", "TREE"),
    ("- - - key: docs/A.md\n", "A"), ("- - &a docs\n", "TREE"), ("- - !!str docs\n", "TREE"), ('- - ".\\\\docs"\n', "TREE"),
    ("- - {root: docs}\n", "TREE"), ("- - x\n  - docs\n", "TREE"),
    ("- - note: see docs\n", "NONE"), ('- - Keep "docs" tidy\n', "NONE"), ("- - x\n", "NONE"), ("- -\n", "NONE"),
    ("- - Don't docs\n", "NONE"),
])
def test_a_sequence_inside_a_sequence_is_read_like_any_other(tmp_path, text, wanted):
    """``- - docs`` was read as ONE plain scalar ``- docs``: its quotes were prose, and nothing named the tree."""
    found = yaml_marks(tmp_path, text)
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("text, wanted", [
    ("- run: |\n    make html\n  working-directory: docs\n", "TREE"), ("- name: >-\n    ./docs/\n  other: x\n", "TREE"),
    ("steps:\n  - run: |\n      echo hi\n    root: docs\n", "TREE"), ("- - run: >-\n      echo\n    root: docs\n", "TREE"),
    ("- k: |\n    text\n  paths:\n    - docs/A.md\n", "A"), ("- run: |\n    echo hi\n- root: docs\n", "TREE"),
    ("- |-\n  docs\n", "TREE"), ("- - |-\n    docs\n", "TREE"), ("k:\n- |\n  docs\n", "TREE"), ("k: >-\n  docs\n", "TREE"),
    ("- run: |\n    ls docs\n  env: x\n", "NONE"), ("- run: |\n    ls docs\n", "NONE"), ("- - run: |\n      ls docs\n    env: x\n", "NONE"),
    ("- run: |\n    echo\n  env:\n    NOTE: see docs\n", "NONE"),
])
def test_a_block_scalar_on_a_sequence_line_ends_where_its_key_s_column_ends_it(tmp_path, text, wanted):
    """The body of ``- run: |`` must be indented past the KEY (column 2), not past the dash (column 0): the sibling keys
    that follow were swallowed as body, so a ``working-directory: docs`` after a ``- run: |`` named nothing."""
    found = yaml_marks(tmp_path, text)
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("text", [
    "root: 'https://x.org/it''s/docs/A.md'\n", "- 'https://x.org/o''s/docs'\n", "k: ['https://x.org/it''s/docs/A.md', x]\n",
    "k: {u: 'https://x.org/it''s/docs/A.md'}\n", "k: 'say ''hi'' https://x.org/docs/A.md'\n",
    "k: 'a''b''c https://x.org/a''b/docs/A.md'\n",
    # after the doubled quote the string goes on, so the comma inside the URL is not an item boundary
    "k: ['it''s https://x.org/a,b/docs/A.md', x]\n", "k: ['a''b''c https://x.org/a,b/docs/A.md more', x]\n",
])
def test_a_doubled_quote_does_not_end_a_url_in_a_single_quoted_yaml_scalar(tmp_path, text):
    """In YAML ``it''s`` is ``it's``: the URL ran on, and what stood after the apostrophe was read as a local path."""
    assert yaml_marks(tmp_path, text) == {idx.T_NONE}


@pytest.mark.parametrize("text", [
    "k: 'https://x.org/a\\''/docs/A.md'\n", "k: ['C:\\data\\', 'https://x.org/a,b/docs/A.md']\n",
    "k: {a: 'C:\\x\\', b: 'https://x.org/a,b/docs/A.md'}\n", "k: 'x\\' 'https://x.org/a\\''/docs/A.md'\n",
])
def test_a_backslash_is_no_escape_in_a_single_quoted_yaml_scalar(tmp_path, text):
    """A YAML single-quoted scalar has no backslash escapes (``'C:\\data\\'`` is complete): the quote after it closes the string."""
    assert yaml_marks(tmp_path, text) == {idx.T_NONE}


def test_a_backslash_still_escapes_a_quote_in_a_double_quoted_yaml_scalar(tmp_path):
    assert yaml_marks(tmp_path, 'k: "https://x.org/say\\"hi\\"/docs/A.md"\n') == {idx.T_NONE}


def test_a_doubled_single_quote_is_still_two_strings_in_python(tmp_path):
    """In Python ``'a''b'`` closes one string and opens the next (adjacent literals), so the URL ends at the first quote and
    the path literal after it is still read: the YAML rule must not leak into code."""
    model = one_file(tmp_path, "src", "a.py", "u = 'https://x.org/a''docs/A.md'\n")
    assert idx.T_PATH in tags(model, "docs/A.md")


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 14
# Both reviewers returned NO BLOCKING FINDINGS after the differential fuzzing; the [P2]s below were reproduced as failing tests.

def test_a_closing_bracket_that_is_part_of_an_inventoried_name_is_kept(tmp_path):
    """``trim_closers`` drops the ``}`` of ``{k: docs/A.md}``; it must not drop a ``]`` or ``}`` that ends a real name."""
    files = {"docs/A.md]": "# a\n", "docs/reports]/B.md": "# b\n", "docs/C.md": "# c\n", "docs/E.md": "# e\n",
             "src/a.py": 'open("docs/A.md]")\nPath("docs/reports]")\n',
             "configs/c.yaml": "k: {name: docs/C.md}\nj: [docs/E.md]\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py", "configs/c.yaml"]))
    assert idx.T_PATH in tags(model, "docs/A.md]")
    assert idx.T_SUBTREE in tags(model, "docs/reports]/B.md")
    assert idx.T_PATH in tags(model, "docs/C.md") and idx.T_PATH in tags(model, "docs/E.md")      # a syntactic closer is still dropped


def test_a_name_ending_in_a_closer_inside_a_flow_collection_is_still_found(tmp_path):
    """``{k: docs/A.md]}``: the name ends in ``]`` and the mapping in ``}``: the longest prefix that names a file wins, so
    ``docs/A.md]`` gets the path and ``docs/A.md`` (a different file) only the weak name mention of its basename."""
    files = {"docs/A.md]": "# a\n", "docs/A.md": "# b\n", "configs/c.yaml": "k: {name: docs/A.md]}\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.yaml"]))
    assert idx.T_PATH in tags(model, "docs/A.md]") and idx.T_PATH not in tags(model, "docs/A.md")


@pytest.mark.parametrize("text, wanted", [
    ("[pytest]\ntestpaths =\n    docs\n", "TREE"), ("[paths]\nroot =\n    docs\n    specs\n", "TREE"), ("[x]\nroots =\n    a, docs\n", "TREE"),
    ("[x]\nk =\n    docs/A.md\n", "A"), ("[x]\nk =\n\n    docs\n", "TREE"), ("[x]\nk = a\n    docs\n", "TREE"),
    ("[x]\nk =\n    ./docs/\n", "TREE"), ("[x]\nk =\n  # note\n    docs\n", "TREE"), ("[x]\nk:\n\tdocs\n", "TREE"),
    ("[x]\nk =\n    note: see docs\n", "NONE"), ("[x]\nk =\n    Keep docs tidy\n", "NONE"), ("[x]\nk = 1\ndocs\n", "NONE"),
    ("[x]\nk =\n[y]\n    docs\n", "NONE"), ("[x]\nk =\n    a\nj = b\n    c\n", "NONE"), ("[x]\nk = 1\n\n\ndocs\n", "NONE"),
])
def test_an_ini_value_continued_on_indented_lines_is_read(tmp_path, text, wanted):
    """configparser folds the indented lines under ``testpaths =`` into the value; each used to be read alone, as nothing."""
    found = tags(one_file(tmp_path, "configs", "c.ini", text), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("source, marked", [
    (["!find docs -type f"], True), (["!ls docs"], True), (["!ls -la docs\n"], True), (["x = 1\n", "!rm -rf docs\n"], True),
    (["!echo hi && ls docs"], True), (["%cd docs"], True), (["%ls docs"], True), (["%%bash\n", "find docs -type f\n"], True),
    (["%%sh\n", "ls docs\n"], True), (["%%script bash\n", "ls docs\n"], True), (["!pip install -r requirements.txt\n", "!ls docs\n"], True),
    (["%env DATA=docs"], True),
    (["!echo \"read docs now\""], False), (["# !ls docs\n"], False), (["x = '!ls docs'\n"], False), (["x = a\n", "y = x != docs\n"], False),
    (["x = (a\n", "     != docs)\n"], False), (["%matplotlib inline"], False), (["m = ('%s'\n", "     % docs)\n"], False),
    (["m = ('%s %d'\n", "     % (docs, 1))\n"], False),
    (["%%time\n", "x = 1\n"], False), (["%%bash\n", "echo hello\n"], False), (["docs = 1\n"], False),
    (["%%capture\n", "y = docs\n"], False),                          # only %%bash / %%sh / %%script / %%shell cells are scripts
])
def test_a_notebook_shell_command_or_magic_is_read_as_a_shell_line(tmp_path, source, marked):
    """A code cell is Python, but ``!find docs`` and ``%cd docs`` are commands and a ``%%bash`` cell is a script."""
    files = {**BASE, "notebooks/n.ipynb": _notebook({"cell_type": "code", "source": source})}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert (idx.T_SUBTREE in tags(model, "docs/A.md")) is marked


@pytest.mark.parametrize("code, wanted", [
    ('report = f"""\n{len(list(Path(\'docs\').rglob(\'*\')))}\n"""\n', "TREE"), ("report = f'''\n{Path(\"docs\") / \"A.md\"}\n'''\n", "A"),
    ('x = rf"""\nprose {Path("docs/A.md").read_text()} more\nlines\n"""\n', "A"), ('x = (f"""a\n{os.listdir("docs")}\nb""")\n', "TREE"),
    ('x = f"""\nline {a}\nline {len(os.listdir(\'docs\'))}\n"""\n', "TREE"), ('x = F"""\n{f\'{Path("docs")}\'}\n"""\n', "TREE"),
    ('x = f"""\nSee the docs folder: {name}\n"""\n', "NONE"), ('x = """\n{Path("docs")}\n"""\n', "NONE"),
    ('x = f"""\n{{Path("docs")}}\n"""\n', "NONE"), ('x = f"""\n{a} {b}\n"""\ny = Path("docs") / "A.md"\n', "A"),
    ('def f():\n    """Doc: {Path("docs")}"""\n    return f"""\n{x}\n"""\n', "NONE"),
    ('x = 1 if"""\n{Path("docs")}\n""" else 2\n', "NONE"),                     # ``if`` ends in f, but it is no prefix
])
def test_the_fields_of_a_multi_line_f_string_are_code_and_its_prose_is_not(tmp_path, code, wanted):
    found = tags(one_file(tmp_path, "src", "a.py", code), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 15
# Round 15 (sol and astra: NO BLOCKING FINDINGS again) found edge cases in what round 14 added, ONE of them a regression of mine:
# an indented option that follows an indented option was swallowed as a "continuation", losing `root = docs`.

@pytest.mark.parametrize("text, wanted", [
    ("[paths]\n  other = x\n  root = docs\n", "TREE"), ("[x]\n  other = 1\n  root = docs/A.md\n", "A"), ("[x]\n\tk = 1\n\tj = docs\n", "TREE"),
    ("[x]\n\tk = 1\n\tj =\n\t\tdocs\n", "TREE"), ("[x]\n  k = a\n    docs\n  j = b\n", "TREE"), ("[x]\n  k =\n      docs\n  j = 1\n", "TREE"),
    ("[x]\n    k =\n  docs\n", "NONE"), ("[x]\n  k = a\n  docs\n", "NONE"), ("[x]\n  k = a\n  j = b\n    c\n", "NONE"),
])
def test_an_ini_continuation_is_indented_deeper_than_its_own_option(tmp_path, text, wanted):
    """configparser continues a value only with a line indented MORE than the option's own line; an equally indented line is
    the next option (or an error), never a continuation."""
    found = tags(one_file(tmp_path, "configs", "c.ini", text), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("source, marked", [
    (['help_text = """\n', '!ls docs\n', '"""\n'], False), (["help = '''\n", "%cd docs\n", "'''\n"], False),
    (['x = f"""\n', '!ls docs {a}\n', '"""\n'], False), (['x = ("a"\n', '     "!ls docs")\n'], False),
    (["files = !ls docs\n"], True), (["out = !find docs -type f | head\n"], True), (["x.out = !ls docs\n"], True),
    (["a = b != docs\n"], False), (["a = 1 if b else 2\n", "c = a != docs\n"], False),
    (["!wget https://example.org/docs\n"], False), (["%%bash\n", "wget https://example.org/docs\n"], False),
    (["!curl -sL https://example.org/a/docs -o out.html\n"], False), (["!ls docs https://example.org/x\n"], True),
    (["\n", "%%bash\n", "ls docs\n"], True), (["\n\n", "  \n", "%%sh\n", "ls docs\n"], True), (["x = 1\n", "\n", "%%bash\n", "ls docs\n"], False),
])
def test_notebook_commands_are_found_outside_python_strings_and_urls_are_not_paths(tmp_path, source, marked):
    files = {**BASE, "notebooks/n.ipynb": _notebook({"cell_type": "code", "source": source})}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert (idx.T_SUBTREE in tags(model, "docs/A.md")) is marked


@pytest.mark.parametrize("code, wanted", [
    ("s = f'''\n{f\"\"\"{len(os.listdir('docs'))} files\"\"\"}\n'''\n", "TREE"),
    ('x = f"""\n{len(\'\'\'it\'s\'\'\') + len(os.listdir(\'docs\'))}\n"""\n', "TREE"),
    ('x = f"""\n{render(\'\'\'\nSee the "docs" folder.\n\'\'\')}\n"""\n', "NONE"),
    ('x = f"""\n{render(\'\'\'\n{Path("docs")}\n\'\'\')}\n"""\n', "NONE"),
    ('x = f"""\n{render(\'\'\'a\'\'\', Path("docs"))}\n"""\n', "TREE"),
    ('x = f"""\n{f\'\'\'{Path("docs") / "A.md"}\'\'\'}\n"""\n', "A"),
])
def test_a_field_of_a_multi_line_f_string_is_lexed_like_any_other_code(tmp_path, code, wanted):
    """The field's text is code: a nested triple-quoted literal is one string (prose, or an f-string with fields of its own)."""
    import ast
    ast.parse(code)                                                  # the example is valid Python
    found = tags(one_file(tmp_path, "src", "a.py", code), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


# ---------------------------------------------------------------------------------------- found by sol AND astra, round 16
# Round 16: BOTH tiers gave a [P1] this time, and it was mine: the new fuzz module was not a policy file, so once tracked its
# fixture literals would have read as real dependencies and turned the real "7 code-path, 20 notes, 9 loop" into "7 code-path, 29
# code-tree" (the same defect class as the very first real run).

def test_every_test_module_of_the_index_is_a_policy_file():
    """Fixture literals name the trees on purpose; a module of this suite that is not listed would blanket every tree once tracked."""
    tests = Path(__file__).resolve().parent
    modules = sorted(tests.glob("test_workspace_index*.py"))
    assert modules                                                       # this module at least (a private copy holds only it)
    for module in modules:
        assert f"src/smallAntibodyGen/tests/{module.name}" in idx.POLICY_FILES, module.name


def test_the_suites_own_fixtures_do_not_blanket_the_trees_once_tracked(tmp_path):
    """The end-to-end form: the listed modules, TRACKED, with fixtures that name the trees, change no document's tier."""
    fixture = 'TREE = ["docs", "./docs", "specs"]\nFILE = "docs/A.md"\nPath("docs") / "A.md"\n'
    files = {**BASE, "specs/s.md": "# s\n", "src/smallAntibodyGen/tests/test_workspace_index.py": fixture,
             "src/smallAntibodyGen/tests/test_workspace_index_fuzz.py": fixture}
    tracked = ["src/smallAntibodyGen/tests/test_workspace_index.py", "src/smallAntibodyGen/tests/test_workspace_index_fuzz.py"]
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=tracked))
    for path in ("docs/A.md", "docs/B.md", "specs/s.md"):
        assert idx.strongest(model.targets[path]) == "none", (path, tags(model, path))      # at most a POLICY mention, never a code tier


@pytest.mark.parametrize("source, marked", [
    (["!echo build#1 && ls docs\n"], True), (["!ls reports#old docs\n"], True), (["files = !ls reports#old docs\n"], True),
    (["%cd reports#old\n", "!ls docs\n"], True), (["x = 1  # !ls docs\n"], False), (["!echo done # ls docs\n"], False),
    (["!echo 'a # b' docs\n"], True),
])
def test_a_notebook_shell_line_keeps_its_hash_the_shell_decides_what_is_a_comment(tmp_path, source, marked):
    """`build#1` is one shell word; a Python comment rule would cut the command there and lose the argument after it."""
    files = {**BASE, "notebooks/n.ipynb": _notebook({"cell_type": "code", "source": source})}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert (idx.T_SUBTREE in tags(model, "docs/A.md")) is marked


@pytest.mark.parametrize("text, wanted", [
    ("[paths]\nroot:docs\n", "TREE"), ("[x]\nk:docs/A.md\n", "A"), ("[x]\nk:\n    docs\n", "TREE"), ("[x]\nk:a, docs\n", "TREE"),
    ("[x]\nk=docs\n", "TREE"), ("[tool:pytest]\ntestpaths:docs\n", "TREE"), ("[x]\nurl = https://x.org/docs\n", "NONE"),
    ("[x]\npath:C:\\data\n", "NONE"), ("[tool:docs]\nk = 1\n", "NONE"), ("[a:docs]\nk = 1\n", "NONE"),
    ("[pytest]\ntestpaths =\n    [unit]\n    docs\n", "TREE"), ("[x]\n  k =\n    [y]\n    docs\n", "TREE"), ("[x]\nk =\n[y]\n    docs\n", "NONE"),
    ("[x]\n  k = a\n  [y]\n    docs\n", "NONE"),
])
def test_ini_delimiters_need_no_blank_and_a_deeper_header_line_is_still_the_value(tmp_path, text, wanted):
    """configparser splits at the first `=` or `:` (a blank is YAML's rule, not INI's) and checks the indentation BEFORE it looks for a
    section header, so a deeper `[unit]` belongs to the open value."""
    found = tags(one_file(tmp_path, "configs", "c.ini", text), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("source, marked", [
    (["!ls \\\n", "    docs\n"], True), (["files = !ls \\\n", "    docs\n"], True), (["!echo a#b \\\n", "    docs\n"], True),
    (["!ls \\\n", "  -la \\\n", "  docs\n"], True), (["%cd \\\n", "  docs\n"], True), (["!ls docs \\\n", "  -la\n"], True),
    (["!ls \\\n", "  -la\n", "docs = 1\n"], False), (["!echo \\\n", "  hi\n", "x = 'docs'\n"], True),
])
def test_a_notebook_shell_command_continued_with_a_backslash_is_one_command(tmp_path, source, marked):
    """The lexer joins the lines, the shell reads them as one command: `!ls \\` + `docs` lists the tree."""
    files = {**BASE, "notebooks/n.ipynb": _notebook({"cell_type": "code", "source": source})}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["notebooks/n.ipynb"]))
    assert (idx.T_SUBTREE in tags(model, "docs/A.md")) is marked


@pytest.mark.parametrize("text, wanted", [
    ("description = '''It's a 'docs' folder'''\n", "NONE"), ('description = """Say "docs" now"""\n', "NONE"), ("k = '''docs'''\n", "TREE"),
    ('k = """docs/A.md"""\n', "A"), ("k = ['''docs''', '''x''']\n", "TREE"), ("k = ['''It's''', 'docs']\n", "TREE"),
    ("a = '''It's 'docs' '''  # 'docs'\n", "NONE"), ("k = '''C:\\data'''\n", "NONE"), ('k = """./docs/"""\n', "TREE"),
    ("k = '''a'''\nj = 'docs'\n", "TREE"), ("k = ''''''\nj = ''\n", "NONE"),
    # triple quotes INSIDE ordinary strings are not delimiters: the strings around them stay separate
    ("paths = [\"'''\", 'docs', \"'''\"]\n", "TREE"), ("k = [\"'''\", \"it's\", \"docs\", \"'''\"]\n", "TREE"), ("k = ['\"\"\"', \"docs\", '\"\"\"']\n", "TREE"),
    ("k = \"'''\"  # '''docs'''\n", "NONE"), ("k = \"a '''\" \n", "NONE"), ("k = ['x', '''docs''', \"'''\"]\n", "TREE"),
    ("k = \"\\\"'''\"\nj = 'docs'\n", "TREE"), ("k = '''open\n", "NONE"),
])
def test_a_one_line_toml_triple_quoted_string_is_one_string(tmp_path, text, wanted):
    """`'''It's a 'docs' folder'''` is ONE prose string for tomllib; read as pairs of quotes it manufactured the literal 'docs'."""
    found = tags(one_file(tmp_path, "configs", "c.toml", text), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    elif wanted == "A":
        assert idx.T_PATH in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("code, wanted", [
    ('x = f"""\n{f"{len(\'\'\'It\'s a \'docs\' folder\'\'\')}"}\n"""\n', "NONE"), ('x = f"{len(\'\'\'It\'s a \'docs\' folder\'\'\')}"\n', "NONE"),
    ('x = f"""\n{f"{len(os.listdir(\'docs\'))}"}\n"""\n', "TREE"), ('x = f"{len(\'\'\'a\'\'\') + len(os.listdir(\'docs\'))}"\n', "TREE"),
    ('x = f"""{render(\'\'\'\nSee \'docs\'\n\'\'\')}"""\n', "NONE"),
])
def test_the_expression_of_a_field_is_lexed_before_its_strings_are_read(tmp_path, code, wanted):
    """A nested triple-quoted literal inside a field is ONE string: its prose must not be read as the literal 'docs'."""
    import ast
    ast.parse(code)
    found = tags(one_file(tmp_path, "src", "a.py", code), "docs/A.md")
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    else:
        assert found == {idx.T_NONE}


@pytest.mark.parametrize("text, wanted", [
    ("k: [Don't, 'docs']\n", "TREE"), ("k: {note: it's here, p: 'docs'}\n", "TREE"), ('- {a: Don\'t, b: ".\\\\docs"}\n', "TREE"),
    ("k: [x's, y's, 'docs']\n", "TREE"), ("- [a ' b, c ' d, 'docs']\n", "TREE"), ("k: [a '' b, 'docs']\n", "TREE"),
    ('k: [a " b, \'docs\']\n', "TREE"), ("k: [a ' b, \"docs\"]\n", "TREE"),
    ("k: [Don't, 'x']\n", "NONE"), ("k: Don't edit 'docs'\n", "NONE"), ("k: [Keep 'docs' tidy, x]\n", "NONE"), ('k: [a " b, c " d]\n', "NONE"),
    ("k: {note: it's 'docs' here}\n", "NONE"),
])
def test_a_stray_quote_in_a_plain_scalar_of_a_flow_collection_leaves_the_next_real_string_alone(tmp_path, text, wanted):
    """A quote inside a plain scalar opens no string. (A KNOWN LIMIT remains: an ODD number of one quote character in plain items of a
    one-line collection, then an item quoted with the same character, pairs across them; see the page's limits.)"""
    found = yaml_marks(tmp_path, text)
    if wanted == "TREE":
        assert idx.T_SUBTREE in found
    else:
        assert found == {idx.T_NONE}


def test_nested_multi_line_f_strings_are_bounded(tmp_path):
    """A field that holds a multi-line f-string that holds a field ...: lexed to a fixed depth, then dropped, never recursed to death."""
    depth = 400
    code = "x = " + 'f"""{' * 1 + "".join("f'''{" for _ in range(depth)) + "Path('docs')" + "}'''" * depth + '}"""\n'
    one_file(tmp_path, "src", "a.py", code)                          # must neither raise nor hang


# ---------------------------------------------------------------------------------------- speed and safety on hostile input

def test_pathological_inputs_are_scanned_in_linear_time(tmp_path):
    """The first real run took 269 s: chain scanning re-searched the whole line prefix for every string literal, and the
    path regex backtracked over long token runs. A multi-megabyte single line must stay fast."""
    long_word = "a" * (idx.SIZE_CAP - 1000)
    many_literals = "x = [" + "'a/b', " * 120_000 + "]\n"
    files = {**BASE, "src/blob.txt": long_word, "src/lits.py": many_literals,
             "outputs/camp/big.json": "[" + '"s/t", ' * 150_000 + '"docs/A.md"]\n'}
    repo = make_repo(tmp_path / "r", files, tracked=["src/blob.txt", "src/lits.py"])
    done = subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(repo), "--today", TODAY, "--check"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    assert "outputs/camp/big.json:1 (path)" in done.stdout          # the scan reached the END of the huge line


def test_a_junction_where_the_index_file_belongs_is_refused_and_not_written_through(tmp_path):
    repo = make_repo(tmp_path / "r", dict(BASE))
    outside = tmp_path / "outside"
    outside.mkdir()
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(repo / "docs" / "INDEX.md"), str(outside)],
                          capture_output=True, text=True)
    if done.returncode:
        pytest.skip("cannot create a junction here")
    try:
        ran = cli(repo)
        assert ran.returncode == 2 and "Traceback" not in ran.stderr, ran.stderr
        assert list(outside.iterdir()) == []
    finally:
        os.rmdir(repo / "docs" / "INDEX.md")


def test_a_note_that_merely_quotes_the_marker_is_still_a_source(tmp_path):
    files = {**BASE, "docs/quote.md": "# Notes\n" + idx.MARKER + "\nsee docs/A.md\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert idx.T_NOTES in tags(model, "docs/A.md") and model.gaps["generated_marker_files"] == 0


def test_in_tree_files_over_the_size_cap_are_listed_and_counted(tmp_path):
    files = {**BASE, "docs/huge.md": "x" * (idx.SIZE_CAP + 10)}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert "docs/huge.md" in model.targets and model.gaps["local_skipped_size"] == 1
    assert "in-tree" in idx.render(model, TODAY).lower()


def test_a_directory_with_a_text_extension_is_skipped_and_not_reported_as_unreadable(tmp_path):
    """The real outputs/ has a DIRECTORY named audit-02.json: read_bytes fails on it, and it is not a file we missed."""
    files = {**BASE, "outputs/camp/dir.json/inner.json": '{"f": "docs/A.md"}\n', "outputs/camp/ok.json": '{"f": "docs/B.md"}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files))
    assert model.gaps["unreadable"] == 0
    assert idx.T_OUTPUTS in tags(model, "docs/B.md") and idx.T_OUTPUTS not in tags(model, "docs/A.md")


def test_unreadable_files_are_named_in_the_gaps(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "r", {**BASE, "docs/locked.md": "# locked\n"})
    original = idx.Path.read_bytes

    def refuse(self):
        if self.name == "locked.md":
            raise PermissionError("denied")
        return original(self)

    monkeypatch.setattr(idx.Path, "read_bytes", refuse)
    model = idx.build_model(repo)
    assert model.gaps["unreadable"] == 1 and [Path(p).name for p in model.gaps["unreadable_paths"]] == ["locked.md"]
    assert "locked.md" in idx.render(model, TODAY)


@pytest.mark.parametrize("text", ["see **docs/A.md** now\n", "see *docs/A.md* now\n", "see __docs/A.md__ now\n",
                                  "see [**docs/A.md**](docs/A.md)\n"])
def test_markdown_emphasis_around_a_path_does_not_hide_it(tmp_path, text):
    model = idx.build_model(make_repo(tmp_path / "r", {**BASE, "docs/note.md": text}))
    assert idx.T_NOTES in tags(model, "docs/A.md") and tags(model, "docs/B.md") == {idx.T_NONE}


def test_the_strongest_evidence_found_orders_code_over_outputs_over_notes_over_loop(tmp_path):
    files = {"docs/A.md": "# a\n", "docs/B.md": "# b\n", "docs/C.md": "# c\n", "docs/D.md": "# d\n", "docs/E.md": "# e\n",
             "docs/F.md": "# f\n", "docs/note.md": "see docs/C.md\n",
             "src/a.py": "open('docs/A.md')\n# only the name E.md is mentioned here\n",
             "outputs/camp/m.json": '{"f": "docs/B.md"}\n',
             ".claude-codex-loop/runs/r1/launch.json": '{"f": "docs/D.md"}\n'}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/a.py"]))
    assert [idx.strongest(model.targets[f"docs/{n}.md"]) for n in "ABCD"] == ["code-path", "outputs", "notes", "loop"]
    assert idx.strongest(model.targets["docs/E.md"]) == "code-name"
    assert idx.strongest(model.targets["docs/F.md"]) == "none"
    text = idx.render(model, TODAY)
    assert "docs/: code-path 1" in text and "outputs 1" in text and "loop 1" in text
    assert "strongest evidence" in text.lower()


def test_a_named_directory_is_a_weaker_tier_than_a_named_file(tmp_path):
    files = {"reference/evidence/x/e1.json": "{}", "reference/n.md": "# n\n", "configs/c.json": '{"root": "reference/evidence"}\n',
             "src/a.py": "open('reference/n.md')\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["configs/c.json", "src/a.py"]))
    assert idx.strongest(model.targets["reference/n.md"]) == "code-path"
    assert idx.strongest(model.targets["reference/evidence/x/e1.json"]) == "code-tree"


def test_an_unresolved_construction_is_its_own_tier_below_exact_code_references(tmp_path):
    files = {"specs/benchmarks/a.json": "{}", "src/t.py": "p = root / 'specs' / 'benchmarks' / f'{stem}.json'\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["src/t.py"]))
    assert idx.strongest(model.targets["specs/benchmarks/a.json"]) == "code-unresolved"


def test_an_unrepresentable_modification_time_does_not_crash_the_render(tmp_path):
    model = idx.build_model(make_repo(tmp_path / "r", dict(BASE)))
    model.targets["docs/A.md"].mtime_ns = -(10 ** 19)
    assert "docs/A.md" in idx.render(model, TODAY)


def test_a_tracked_generated_file_inside_the_trees_is_excluded_once(tmp_path):
    files = {**BASE, "docs/gen.md": idx.MARKER + "\nsee docs/A.md\n"}
    model = idx.build_model(make_repo(tmp_path / "r", files, tracked=["docs/gen.md"]))
    assert model.gaps["generated_marker_files"] == 1 and "docs/gen.md" not in model.targets
    assert idx.T_PATH not in tags(model, "docs/A.md")


@pytest.mark.parametrize("where", [".claude-codex-loop/runs", "outputs/camp"])
def test_a_linked_directory_of_records_is_never_followed(tmp_path, where):
    repo = make_repo(tmp_path / "r", dict(BASE))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "rec.json").write_text('{"f": "docs/A.md"}\n')
    link_path = repo / where
    link_path.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(outside)], capture_output=True, text=True)
    if done.returncode:
        pytest.skip("cannot create a junction here")
    try:
        model = idx.build_model(repo)
        assert tags(model, "docs/A.md") == {idx.T_NONE}
        assert where in "".join(model.gaps["loop_links_skipped"] + model.gaps["outputs_links_skipped"])
    finally:
        os.rmdir(link_path)

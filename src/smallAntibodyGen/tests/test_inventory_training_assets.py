"""Tests for the Gate-0 asset inventory (`scripts/inventory_training_assets.py`).

Everything here runs on tiny temporary files. A real checkpoint fixture is
deliberately absent: the `.pt` files these tests read are hand-built zip
archives holding one pickled dict, which is exactly the on-disk shape
`torch.save` produces and needs neither torch nor 5 MB of weights to exercise
the metadata reader.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


class UnimportableMarker:
    """A class the metadata reader must stub out instead of importing."""


class StorageStandIn:
    """Stands in for a tensor storage, referenced by persistent id like torch's."""


def _load_inventory(project_root: Path):
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "inventory_training_assets", scripts_dir / "inventory_training_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inventory(project_root: Path):
    return _load_inventory(project_root)


@pytest.fixture
def inventory_script(project_root: Path) -> Path:
    return project_root.parents[1] / "scripts" / "inventory_training_assets.py"


def write_fake_checkpoint(path: Path, payload: dict, *, archive_root: str = "ckpt") -> Path:
    """
    Write a `.pt` in `torch.save`'s zip layout without importing torch.

    The metadata reader only ever opens `<root>/data.pkl`, so a fixture that
    contains just that member exercises the same code path a real checkpoint
    does, at a few hundred bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{archive_root}/data.pkl", pickle.dumps(payload))
        archive.writestr(f"{archive_root}/version", "3\n")
    return path


def write_checkpoint_with_storages(path: Path, payload: dict) -> Path:
    """
    Write a `.pt` whose tensors are persistent-id references, as torch's are.

    `StorageStandIn` values are pickled by reference into a `data/<key>` member
    exactly the way `torch.save` writes storages, so a reader that resolves
    persistent ids would have to go read those bytes. Here they are deliberate
    garbage: the test passes only if they are never touched.
    """

    class _StoragePickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, StorageStandIn):
                return ("storage", "FloatStorage", "0", "cpu", 64)
            return None

    buffer = io.BytesIO()
    _StoragePickler(buffer, protocol=2).dump(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ckpt/data.pkl", buffer.getvalue())
        archive.writestr("ckpt/data/0", b"\xff" * 64)
        archive.writestr("ckpt/version", "3\n")
    return path


def write_corpus(path: Path, text: str) -> Path:
    """
    Write ``text`` verbatim, with no platform newline translation.

    ``Path.write_text`` defaults to ``newline=None``, which rewrites
    ``\n`` as ``os.linesep``. On Windows that silently turns a fixture
    written as ``b'{"epoch": 1}\n'`` into ``b'{"epoch": 1}\r\n'``, so a
    test asserting on a hash of the intended bytes fails for a reason that
    has nothing to do with the hashing code under test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_file_chunks_are_actually_chunked(inventory, tmp_path: Path):
    """The streaming property, stated directly rather than inferred from a digest."""
    path = write_corpus(tmp_path / "corpus.jsonl", "abcdefghi")
    assert list(inventory.iter_file_chunks(path, 4)) == [b"abcd", b"efgh", b"i"]


def test_sha256_matches_a_whole_file_hash_at_every_chunk_size(inventory, tmp_path: Path):
    payload = b"x" * 5000 + b"y" * 137
    path = tmp_path / "corpus.jsonl"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    for chunk_bytes in (1, 7, 4096, 1 << 20):
        assert inventory.sha256_file(path, chunk_bytes) == expected


def test_zero_chunk_size_fails_loud(inventory, tmp_path: Path):
    path = write_corpus(tmp_path / "corpus.jsonl", "abc")
    with pytest.raises(ValueError, match="chunk_bytes"):
        list(inventory.iter_file_chunks(path, 0))


def test_hashing_never_reads_the_whole_file_at_once(inventory, tmp_path: Path, monkeypatch):
    """A corpus here is ~500 MB; a single `.read()` would be a resident copy of it."""
    path = tmp_path / "corpus.jsonl"
    path.write_bytes(b"z" * 10_000)
    sizes: list[int] = []

    class _RecordingHandle:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            sizes.append(size)
            return self._handle.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return self._handle.__exit__(*exc_info)

    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        return _RecordingHandle(real_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", recording_open)
    inventory.sha256_file(path, 1024)
    assert sizes
    assert all(size == 1024 for size in sizes)


# ---------------------------------------------------------------------------
# Deterministic walk order
# ---------------------------------------------------------------------------


def test_walk_order_is_sorted_and_independent_of_creation_order(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    for name in ("zulu", "alpha", "mike"):
        write_corpus(root / name / "oas_all.jsonl", name)
    walked = [inventory.logical_path(p, root) for p in inventory.iter_files(root)]
    assert walked == [
        "alpha/oas_all.jsonl",
        "mike/oas_all.jsonl",
        "zulu/oas_all.jsonl",
    ]


def test_corpora_are_sorted_by_logical_path(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    for name in ("oas_5k", "oas_1m", "oas_50k"):
        write_corpus(root / name / "oas_all.jsonl.gz", name)
    record = inventory.inventory_data_root("data/processed", root)
    paths = [item["logical_path"] for item in record["corpora"]]
    assert paths == sorted(paths)


def test_machine_noise_is_excluded(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    write_corpus(root / "oas_5k" / "oas_all.jsonl", "a")
    write_corpus(root / ".DS_Store", "finder")
    write_corpus(root / "oas_5k" / "oas_all.jsonl.gz.tmp", "half-written")
    walked = [inventory.logical_path(p, root) for p in inventory.iter_files(root)]
    assert walked == ["oas_5k/oas_all.jsonl"]


# ---------------------------------------------------------------------------
# Logical paths and root labels
# ---------------------------------------------------------------------------


def test_absolute_root_without_a_label_is_rejected(inventory):
    with pytest.raises(ValueError, match="LABEL=PATH"):
        inventory.parse_root_argument("/mnt/scratch/processed")


def test_absolute_root_with_a_label_keeps_the_label_only(inventory):
    label, path = inventory.parse_root_argument("corpora=/mnt/scratch/processed")
    assert label == "corpora"
    assert path == Path("/mnt/scratch/processed")


def test_relative_root_label_is_normalized(inventory):
    assert inventory.parse_root_argument("./data/processed/")[0] == "data/processed"
    assert inventory.parse_root_argument("data/processed")[0] == "data/processed"


def test_no_absolute_path_reaches_the_json(inventory, tmp_path: Path):
    """The same corpus at a different absolute path must inventory identically."""
    root = tmp_path / "deep" / "nested" / "processed"
    write_corpus(root / "oas_5k" / "oas_all.jsonl", "records")
    payload = inventory.build_inventory([("corpora", root)], [])
    text = inventory.render_json(payload)
    assert str(tmp_path) not in text
    assert "corpora" in text
    assert payload["data_roots"][0]["corpora"][0]["logical_path"] == "oas_5k/oas_all.jsonl"


def test_missing_root_is_recorded_as_absent_not_omitted(inventory, tmp_path: Path):
    payload = inventory.build_inventory(
        [("data/processed", tmp_path / "nope")], [("checkpoints", tmp_path / "also-nope")]
    )
    assert payload["data_roots"][0]["present"] is False
    assert payload["data_roots"][0]["corpora"] == []
    assert payload["checkpoint_roots"][0]["present"] is False
    assert payload["checkpoint_roots"][0]["runs"] == []


def test_zero_one_and_many_roots(inventory, tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_corpus(first / "oas_all.jsonl", "a")
    write_corpus(second / "oas_all.jsonl", "b")

    assert inventory.build_inventory([], [])["data_roots"] == []
    one = inventory.build_inventory([("a", first)], [])
    assert len(one["data_roots"]) == 1
    many = inventory.build_inventory([("a", first), ("b", second)], [])
    assert [root["label"] for root in many["data_roots"]] == ["a", "b"]


# ---------------------------------------------------------------------------
# Stats manifests
# ---------------------------------------------------------------------------


def test_sibling_of_directory_manifest_is_found(inventory, tmp_path: Path):
    """The layout this repo actually uses: `oas_5k/` beside `oas_5k.stats.json`."""
    root = tmp_path / "processed"
    corpus = write_corpus(root / "oas_5k" / "oas_all.jsonl.gz", "records")
    write_corpus(root / "oas_5k.stats.json", '{"records_kept": 5000}')
    record = inventory.describe_corpus(corpus, root)
    assert record["stats_manifest"]["logical_path"] == "oas_5k.stats.json"


def test_sibling_manifest_without_the_stats_infix_is_found(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    corpus = write_corpus(root / "oas_paired_all" / "oas_all.jsonl.gz", "records")
    write_corpus(root / "oas_paired_all.json", "{}")
    record = inventory.describe_corpus(corpus, root)
    assert record["stats_manifest"]["logical_path"] == "oas_paired_all.json"


def test_in_directory_manifest_is_found_and_wins(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    corpus = write_corpus(root / "oas_5k" / "oas_all.jsonl.gz", "records")
    write_corpus(root / "oas_5k" / "oas_all.stats.json", "{}")
    write_corpus(root / "oas_5k.stats.json", "{}")
    record = inventory.describe_corpus(corpus, root)
    assert record["stats_manifest"]["logical_path"] == "oas_5k/oas_all.stats.json"


def test_missing_manifest_is_null_not_omitted(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    corpus = write_corpus(root / "oas_5k" / "oas_all.jsonl.gz", "records")
    record = inventory.describe_corpus(corpus, root)
    assert "stats_manifest" in record
    assert record["stats_manifest"] is None


def test_manifest_outside_the_root_is_not_recorded(inventory, tmp_path: Path):
    """A manifest above the root has no logical path, so it must not be referenced."""
    root = tmp_path / "processed"
    corpus = write_corpus(root / "oas_all.jsonl.gz", "records")
    write_corpus(tmp_path / "processed.stats.json", "{}")
    assert inventory.describe_corpus(corpus, root)["stats_manifest"] is None


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------


def test_checkpoint_metadata_reads_stage_config_epoch_and_parent(inventory, tmp_path: Path):
    root = tmp_path / "checkpoints"
    write_fake_checkpoint(
        root / "paired" / "best.pt",
        {
            "epoch": 7,
            "val_loss": 0.5,
            "best_val_loss": 0.4,
            "train_config": {
                "training_stage": "paired_refine",
                "init_checkpoint": "checkpoints/mlm_50k/best.pt",
                "d_model": 256,
            },
        },
    )
    record = inventory.describe_checkpoint(root / "paired" / "best.pt", root)
    assert record["metadata_source"] == "zip_pickle"
    assert record["metadata_error"] is None
    assert record["epoch"] == 7
    assert record["training_stage"] == "paired_refine"
    assert record["training_stage_source"] == "saved_config"
    assert record["parent_checkpoint"] == "checkpoints/mlm_50k/best.pt"
    assert record["saved_config"]["d_model"] == 256
    assert record["size_bytes"] == (root / "paired" / "best.pt").stat().st_size


def test_missing_stage_field_is_distinguished_from_a_null_stage(inventory, tmp_path: Path):
    """The local checkpoints predate `training_stage`; that is a different fact."""
    root = tmp_path / "checkpoints"
    write_fake_checkpoint(root / "legacy" / "best.pt", {"epoch": 10, "train_config": {"seed": 42}})
    record = inventory.describe_checkpoint(root / "legacy" / "best.pt", root)
    assert record["training_stage"] is None
    assert record["training_stage_source"] == "absent_from_saved_config"
    assert record["parent_checkpoint"] is None


def test_checkpoint_metadata_never_materializes_tensors(inventory, tmp_path: Path):
    """
    The storage blobs must stay unread — that is the whole memory argument.

    The archive's `data/0` member is garbage; a reader that resolved persistent
    ids would try to turn it into a tensor and fail. Metadata still comes back.
    """
    root = tmp_path / "checkpoints"
    path = write_checkpoint_with_storages(
        root / "run" / "best.pt",
        {
            "epoch": 3,
            "model_state_dict": {"token_embedding.weight": StorageStandIn()},
            "train_config": {"training_stage": "base", "seed": 1},
        },
    )
    record = inventory.describe_checkpoint(path, root)
    assert record["metadata_source"] == "zip_pickle"
    assert record["metadata_error"] is None
    assert record["epoch"] == 3
    assert record["training_stage"] == "base"


def test_unknown_globals_become_opaque_instead_of_being_imported(inventory, tmp_path: Path):
    """An unknown class in the payload is recorded as unserializable, not imported."""
    root = tmp_path / "checkpoints"
    path = write_fake_checkpoint(
        root / "run" / "best.pt",
        {"epoch": 5, "train_config": {"marker": UnimportableMarker()}},
    )
    record = inventory.describe_checkpoint(path, root)
    assert record["epoch"] == 5
    marker = record["saved_config"]["marker"]
    assert marker["__unserializable__"].endswith(".UnimportableMarker")


def test_corrupt_checkpoint_produces_a_record_not_a_crash(inventory, tmp_path: Path):
    root = tmp_path / "checkpoints"
    path = root / "run" / "best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a torch checkpoint")
    record = inventory.describe_checkpoint(path, root)
    assert record["logical_path"] == "run/best.pt"
    assert record["sha256"] == hashlib.sha256(b"this is not a torch checkpoint").hexdigest()
    assert record["metadata_source"] is None
    assert record["metadata_error"] is not None
    assert record["saved_config"] is None
    assert record["training_stage_source"] == "unavailable"


def test_corrupt_checkpoint_error_leaks_no_path(inventory, tmp_path: Path):
    """Exception text embeds the absolute path it was handed; only class names survive."""
    root = tmp_path / "checkpoints"
    path = root / "run" / "best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"garbage")
    error = inventory.describe_checkpoint(path, root)["metadata_error"]
    assert str(tmp_path) not in error
    assert "zip_pickle:" in error and "torch_load:" in error


def test_corrupt_checkpoint_does_not_stop_the_rest_of_the_run(inventory, tmp_path: Path):
    root = tmp_path / "checkpoints"
    (root / "run").mkdir(parents=True)
    (root / "run" / "broken.pt").write_bytes(b"garbage")
    write_fake_checkpoint(root / "run" / "last.pt", {"epoch": 2, "train_config": {}})
    record = inventory.inventory_checkpoint_root("checkpoints", root)
    paths = [item["logical_path"] for item in record["runs"][0]["checkpoints"]]
    assert paths == ["run/broken.pt", "run/last.pt"]


def test_torch_load_is_only_a_fallback(inventory, tmp_path: Path, monkeypatch):
    path = write_fake_checkpoint(tmp_path / "best.pt", {"epoch": 1, "train_config": {}})

    def explode(_path):
        raise AssertionError("torch fallback must not run when the archive parses")

    monkeypatch.setattr(inventory, "read_torch_payload", explode)
    payload, source, error = inventory.read_checkpoint_payload(path)
    assert source == "zip_pickle"
    assert error is None
    assert payload["epoch"] == 1


def test_torch_fallback_is_used_when_the_archive_is_unparseable(
    inventory, tmp_path: Path, monkeypatch
):
    path = tmp_path / "legacy.pt"
    path.write_bytes(b"legacy non-zip format")

    monkeypatch.setattr(
        inventory, "read_torch_payload", lambda _path: {"epoch": 9, "train_config": {"seed": 1}}
    )
    payload, source, error = inventory.read_checkpoint_payload(path)
    assert source == "torch_load"
    assert error is None
    assert payload["epoch"] == 9


# ---------------------------------------------------------------------------
# metrics.jsonl
# ---------------------------------------------------------------------------


def test_missing_metrics_jsonl_is_recorded_as_null(inventory, tmp_path: Path):
    root = tmp_path / "checkpoints"
    write_fake_checkpoint(root / "mlm_small" / "best.pt", {"epoch": 1, "train_config": {}})
    record = inventory.inventory_checkpoint_root("checkpoints", root)
    run = record["runs"][0]
    assert "metrics_jsonl" in run
    assert run["metrics_jsonl"] is None


def test_present_metrics_jsonl_is_hashed(inventory, tmp_path: Path):
    root = tmp_path / "checkpoints"
    write_fake_checkpoint(root / "mlm_small" / "best.pt", {"epoch": 1, "train_config": {}})
    write_corpus(root / "mlm_small" / "metrics.jsonl", '{"epoch": 1}\n')
    run = inventory.inventory_checkpoint_root("checkpoints", root)["runs"][0]
    assert run["metrics_jsonl"]["logical_path"] == "mlm_small/metrics.jsonl"
    assert run["metrics_jsonl"]["sha256"] == hashlib.sha256(b'{"epoch": 1}\n').hexdigest()
    # ...and it is not double-counted as an ordinary file.
    assert [item["logical_path"] for item in run["files"]] == []


def test_empty_run_directory_still_reports_its_missing_metrics(inventory, tmp_path: Path):
    """`checkpoints/mlm/` is empty on this workstation; inferring runs from files would hide it."""
    root = tmp_path / "checkpoints"
    (root / "mlm").mkdir(parents=True)
    run = inventory.inventory_checkpoint_root("checkpoints", root)["runs"][0]
    assert run["logical_path"] == "mlm"
    assert run["checkpoints"] == []
    assert run["metrics_jsonl"] is None


# ---------------------------------------------------------------------------
# Strict JSON
# ---------------------------------------------------------------------------


def test_render_is_strict_json_with_sorted_keys(inventory, tmp_path: Path):
    root = tmp_path / "processed"
    write_corpus(root / "oas_5k" / "oas_all.jsonl", "records")
    text = inventory.render_json(inventory.build_inventory([("data/processed", root)], []))
    reparsed = json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    assert json.dumps(reparsed, indent=2, sort_keys=True) + "\n" == text


def test_non_finite_config_values_become_null(inventory, tmp_path: Path):
    root = tmp_path / "checkpoints"
    write_fake_checkpoint(
        root / "run" / "best.pt",
        {"epoch": 1, "val_loss": float("nan"), "train_config": {"lr": float("inf")}},
    )
    record = inventory.describe_checkpoint(root / "run" / "best.pt", root)
    assert record["val_loss"] is None
    assert record["saved_config"]["lr"] is None
    # The whole payload must still render under allow_nan=False.
    inventory.render_json(inventory.build_inventory([], [("checkpoints", root)]))


def test_render_refuses_a_non_finite_value_it_was_not_given_a_chance_to_sanitize(inventory):
    with pytest.raises(ValueError):
        inventory.render_json({"loss": float("nan")})


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_write_goes_through_os_replace_from_a_same_directory_temp_file(
    inventory, tmp_path: Path, monkeypatch
):
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(inventory.os, "replace", recording_replace)
    destination = tmp_path / "outputs" / "training-inventory.json"
    inventory.write_json_atomic({"schema_version": inventory.SCHEMA_VERSION}, destination)

    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == destination
    assert src.parent == destination.parent
    assert src.name.endswith(".tmp")
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"]


def test_a_failed_write_leaves_the_previous_inventory_intact(inventory, tmp_path: Path):
    destination = tmp_path / "training-inventory.json"
    inventory.write_json_atomic({"schema_version": "keep-me"}, destination)
    before = destination.read_bytes()

    with pytest.raises(ValueError):
        inventory.write_json_atomic({"loss": float("nan")}, destination)

    assert destination.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_no_temp_file_survives_a_successful_write(inventory, tmp_path: Path):
    destination = tmp_path / "training-inventory.json"
    inventory.write_json_atomic({"schema_version": "x"}, destination)
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# End to end, through the CLI
# ---------------------------------------------------------------------------


def _build_fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data" / "processed"
    write_corpus(data_root / "oas_5k" / "oas_all.jsonl.gz", "5k records")
    write_corpus(data_root / "oas_5k" / "oas_igh.jsonl.gz", "5k heavy")
    write_corpus(data_root / "oas_5k.stats.json", '{"records_kept": 5000}')
    write_corpus(data_root / "oas_1m" / "oas_all.jsonl.gz", "1m records")

    checkpoint_root = tmp_path / "checkpoints"
    write_fake_checkpoint(
        checkpoint_root / "mlm_small" / "best.pt",
        {"epoch": 10, "val_loss": 2.7, "train_config": {"training_stage": "base", "seed": 42}},
    )
    write_fake_checkpoint(
        checkpoint_root / "mlm_small" / "last.pt",
        {"epoch": 10, "val_loss": 2.8, "train_config": {"training_stage": "base", "seed": 42}},
    )
    write_corpus(checkpoint_root / "mlm_small" / "train_config.json", "{}")
    (checkpoint_root / "mlm").mkdir(parents=True)
    return data_root, checkpoint_root


def run_inventory_cli(script: Path, data_root: Path, checkpoint_root: Path, output: Path):
    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        f"corpora={data_root}",
        "--checkpoint-root",
        f"checkpoints={checkpoint_root}",
        "--output-json",
        str(output),
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_cli_output_is_byte_identical_on_rerun(inventory_script: Path, tmp_path: Path):
    """The whole point of the ticket: `diff` over two runs is a real integrity check."""
    data_root, checkpoint_root = _build_fixture_tree(tmp_path)
    first = tmp_path / "out" / "one.json"
    second = tmp_path / "out" / "two.json"
    run_inventory_cli(inventory_script, data_root, checkpoint_root, first)
    run_inventory_cli(inventory_script, data_root, checkpoint_root, second)
    assert first.read_bytes() == second.read_bytes()


def test_cli_records_the_full_shape(inventory_script: Path, tmp_path: Path):
    data_root, checkpoint_root = _build_fixture_tree(tmp_path)
    output = tmp_path / "out" / "inventory.json"
    run_inventory_cli(inventory_script, data_root, checkpoint_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "training-inventory/1"
    corpora = payload["data_roots"][0]["corpora"]
    assert [item["logical_path"] for item in corpora] == [
        "oas_1m/oas_all.jsonl.gz",
        "oas_5k/oas_all.jsonl.gz",
        "oas_5k/oas_igh.jsonl.gz",
    ]
    assert corpora[1]["stats_manifest"]["logical_path"] == "oas_5k.stats.json"
    assert corpora[0]["stats_manifest"] is None

    runs = {run["logical_path"]: run for run in payload["checkpoint_roots"][0]["runs"]}
    assert sorted(runs) == ["mlm", "mlm_small"]
    assert runs["mlm"]["checkpoints"] == []
    assert [item["logical_path"] for item in runs["mlm_small"]["checkpoints"]] == [
        "mlm_small/best.pt",
        "mlm_small/last.pt",
    ]
    assert runs["mlm_small"]["checkpoints"][0]["training_stage"] == "base"
    assert runs["mlm_small"]["metrics_jsonl"] is None
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_cli_requires_an_output_path(inventory_script: Path, tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(inventory_script), "--data-root", "data/processed"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--output-json" in result.stderr


def test_cli_rejects_an_unlabelled_absolute_root(inventory_script: Path, tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(inventory_script),
            "--data-root",
            str(tmp_path),
            "--output-json",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "LABEL=PATH" in result.stderr

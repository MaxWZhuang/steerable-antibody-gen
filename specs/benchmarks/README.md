# Benchmark source manifests

Raw benchmark downloads stay **outside Git**. What is committed here is the manifest: enough
metadata that another machine can retrieve the same release and prove, byte for byte, that it
prepared the same files.

One JSON file per source. The file stem is the `dataset_name`.

| File | Source |
|---|---|
| `cr9114_cr6261_landscape.json` | CR9114/CR6261 combinatorial landscapes |
| `avida_hil6.json` | AVIDa-hIL6 |
| `open_alphaseq.json` | Open AlphaSeq |

The schema and every validator live in `src/smallAntibodyGen/benchmarks/provenance.py`.
Tests live in `src/smallAntibodyGen/tests/test_benchmark_data.py` and never download anything.

## Status: all three manifests are unapproved

Every manifest in this directory is a **template**. The owner has not yet supplied release
versions, source URLs, licenses, retrieval dates, or hashes, and the raw downloads are not
present locally. Nothing here may be consumed as data.

That is enforced, not merely stated. Owner-supplied values carry the literal sentinel
`TODO(owner)`, and `validate_source_manifest` raises `UnsuppliedOwnerDecisionError` if a
sentinel survives anywhere in the document. There is no path by which an unfilled manifest is
mistaken for an approved one.

Two entry points, on purpose:

- `parse_manifest_document(doc)` — structural parse. Returns a `ManifestDocument` whose
  `is_approved` is false and whose `unsupplied_fields` lists the dotted path of every sentinel.
  Use this to *inspect* a template.
- `validate_source_manifest(doc)` — strict. Returns a `SourceManifest` or raises. Use this
  before touching data.

## Field reference

| Field | Meaning |
|---|---|
| `schema_version` | Must equal `provenance.SCHEMA_VERSION` (`"1"`). |
| `dataset_name` | Stable identifier; matches the filename stem. |
| `release_version` | The exact release, revision, or DOI version the owner pinned. |
| `source_url` | The authoritative download URL. Owner-supplied. |
| `license` | License of the retrieved artifacts. Owner-supplied. |
| `retrieval_date` | ISO `YYYY-MM-DD`. The date the files were actually retrieved. |
| `files` | One entry per prepared file: `relative_path` (relative to the raw root, forward slashes, no `..`), `size_bytes`, `sha256` (64 lowercase hex). An approved manifest needs at least one entry. |
| `candidate_source_url` | A URL taken from the plan's reference list and **not verified by anyone in this repository**. It is a lead, never a source. |
| `candidate_source_url_verified` | Must be `false`. An owner who verifies a URL promotes it to `source_url` rather than flipping this flag. |
| `plan_assertions` | Claims copied from `docs/PLAN-steering-prerequisites.md`, each with `verified: false`. They record what the plan says so a later reader can check it, and they are not treated as fact. |
| `owner_decisions` | The open questions blocking approval. Each has `key`, `question`, `status`. Any `unsupplied` entry blocks strict validation. |
| `notes` | Free text. |

The validator **rejects rather than repairs**: no trimming, no case folding, no defaults. A
hash with a leading space is an error, not something to clean up.

## Canonical form

Manifests are serialized through `provenance.dumps_manifest`: sorted keys, two-space indent,
trailing newline, no runtime timestamps. A manifest rewritten from its own parsed form is
byte-identical, and a test asserts it. Edit by hand only in that form, or regenerate.

## Filling one in

1. The owner supplies the release version, source URL, license, and answers to the
   `owner_decisions` entries.
2. Download to a raw root outside the repository.
3. Build one `FileEntry` per prepared file with `provenance.file_entry_for(root, path)`, which
   streams SHA-256 in chunks.
4. Replace every `TODO(owner)`, set each answered decision's `status` to `supplied`, and write
   the file with `dumps_manifest`.
5. Confirm `validate_source_manifest` now returns a `SourceManifest`, and confirm a second
   machine reproduces the same hashes with `verify_manifest_files(manifest, root)`.

A hash nobody computed is **absent**, never guessed. `files: []` is the honest state of an
unretrieved source.

## What is deliberately not here

J05a is provenance and schemas only. The CR9114 oracle (J05b), the AVIDa splitter (J05c), the
Open AlphaSeq audit (J05d), and the evaluator entry points are separate tickets, and none of
them exist yet.

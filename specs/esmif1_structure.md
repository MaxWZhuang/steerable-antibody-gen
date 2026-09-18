# Declared structural input for the ESM-IF1 editing policy

**Date:** 2026-09-15

**Status, 2026-09-18:** implemented as `smallAntibodyGen.structure`, with the CLI
`scripts/prepare_esmif1_structure.py`, pinned by
`src/smallAntibodyGen/tests/test_esmif1_structure.py`. This is **input
preparation only**. **Update, 2026-09-16:** the user selected
[5CJQ as the CR9114/H1 working template](../reference/5cjq-structural-template.md).
Its source files are hash-pinned, and the
[benchmark mapping audit](../reference/cr9114-5cjq-mapping.md) verifies all 16
sites and all 121 VH residues. The
[model input is prepared](../reference/cr9114-5cjq-context.md), and subsequent
[released-weight scoring and training](../reference/cr9114-5cjq-pilot.md) completed.
This document retains the structural-input contract; completed integration work
is summarized in [the policy specification](esmif1_policy.md#integration-status).

`specs/esmif1_policy.md` owns the probability contract. This document owns the
schema, the exact supported and rejected cases, and the limitations.

## What it is for

`ConstrainedEditPolicy.encode_geometry` takes an `(L, 3, 3)` N/CA/C array and a
confidence vector as plain arguments. Everything between "a PDB file someone
downloaded" and those two arrays is a sequence of choices that the parsers will
otherwise make silently:

| Silent choice | Where | What it changes |
|---|---|---|
| author vs. label residue numbering | `pdbx/convert.py:252` (`use_author_fields=True`) | what every `res_id` means |
| fallback to `label_*` when an `auth_*` column is absent | `pdbx/convert.py:455-473`, `warnings.warn` only | the same, without an error |
| first data block when none is named | `pdbx/convert.py:444-449` | which structure is read |
| `altloc="first"` | `pdbx/convert.py:250`, `pdb/file.py:338` | which conformation is encoded |
| `NaN` for an absent backbone atom | `esm/inverse_folding/util.py:102` | a missing residue becomes padding |
| model *ordinal*, not MODEL serial | `pdbx/convert.py:825-833`, `pdb/file.py:1164-1182` | which model is read |

Each one becomes a declared field that is checked, or a documented refusal.

## The manifest

One JSON document, one hash. The correspondence table and the edit-space sites
live **inside** the manifest, so `manifest_sha256` pins the source selection, the
residue mapping and the sites together and a site cannot drift away from the
mapping whose indices it names.

```json
{
  "schema_version": "esmif1-structure-manifest/1",
  "kind": "esmif1_structure_manifest",
  "source": {
    "relative_path": "toy_complex.pdb",
    "sha256": "<64 lowercase hex>",
    "size_bytes": 3400,
    "format": "pdb",
    "model_ordinal": 1,
    "residue_numbering": "author",
    "attribution": {"source": "...", "detail": "...", "license": "..."}
  },
  "chains": {
    "order": ["H", "A"],
    "decoded_chain": "H",
    "confidence": {"H": 1.0, "A": 1.0},
    "inter_chain_pad_length": 10
  },
  "conventions": {"altloc": "reject_any", "missing_backbone_atom": "reject"},
  "structural_relationship": {
    "kind": "synthetic_example",
    "description": "A generated straight-line backbone."
  },
  "decoded_sequence": "ACDEFGHIKL",
  "correspondence": [
    {"sequence_index": 0, "chain_id": "H", "res_id": 1, "ins_code": "",
     "expected_res_name": "ALA", "sequence_residue": "A", "mismatch": null}
  ],
  "edit_space": {
    "sites": [
      {"site_id": "site_b", "sequence_index": 5, "allowed_residues": ["G", "S"],
       "attribution": {"source": "...", "detail": "..."}},
      {"site_id": "site_a", "sequence_index": 2, "allowed_residues": ["D", "N"],
       "attribution": {"source": "...", "detail": "..."}}
    ]
  },
  "notes": ""
}
```

Every key is required at every level; an unknown key is refused rather than
ignored. JSON parsing itself is strict: a duplicate object key is rejected
(`json.loads` keeps the last one, so the digest would describe a document nobody
wrote) and the non-standard `NaN` / `Infinity` literals are rejected.

**A validated manifest owns its own copy.** `StructureManifest` keeps one
authoritative representation — the canonical JSON text as it validated — and
computes `digest` from that. The caller's dict is not retained, `.document`
returns a fresh parse on every read, and every nested mapping the parsed sections
expose (`chains.confidence`, both `attribution` blocks, a row's `mismatch`) is
read-only. Freezing a dataclass does not freeze the dictionaries inside it, so
without this a caller could edit the document it passed in — or the one it got
back — and silently relabel a manifest that had already been checked, digest
included.

**The snapshot is authoritative, and is re-checked against the parsed fields.**
`dataclasses.replace` builds a *new* `StructureManifest` from arbitrary parts
while carrying the old canonical text, so its parsed support can read `G/T` while
its digest still pins `G/S`. `revalidate_manifest(manifest)` re-parses the
snapshot and requires the rebuilt manifest to equal the one in hand, canonical
text included; a snapshot that is valid but not canonical is refused too. Every
public boundary that acts on a manifest calls it: `prepare_structure`,
`structure_path_for`, `prepared_to_document`, `check_prepared_state`,
`verify_against_source`, `build_report`, `build_edit_space` and `bind_policy`. It
costs one JSON parse and reads no arrays.

**Canonical text must be UTF-8 encodable.** A lone surrogate is a legal `str` and
a legal JSON string, so it is refused during validation rather than at the first
`.encode()` — the digest, the artifact and every file written are UTF-8 bytes of
this text.

**Closed enums.** `format` ∈ `{pdb, mmcif}` — **declared, never sniffed from the
extension**. `residue_numbering` ∈ `{author}`. `altloc` ∈ `{reject_any}`.
`missing_backbone_atom` ∈ `{reject}`. `structural_relationship.kind` ∈
`{same_construct, template_for_different_construct, synthetic_example}`.

**`model_ordinal` is a position, not a serial.** Both biotite readers slice by
ordinal. A file whose models are numbered 5 and 7 is addressed as ordinals 1 and
2, and the spec says so rather than implying a `MODEL` record lookup.

**Cross-checks, so no field is decorative.**

- `decoded_chain` must equal `chains.order[0]`: upstream always concatenates the
  target chain first (`esm/inverse_folding/multichain_util.py:68-77`).
- `correspondence` covers `0 .. len(decoded_sequence) - 1`, ascending, once each,
  with `chain_id` always the decoded chain and `(chain_id, res_id, ins_code)`
  unique — the insertion code is part of the identity, so 52, 52A and 52B are
  three residues.
- `sequence_residue` must equal `decoded_sequence[sequence_index]`.
- Mismatches are checked **both ways**: an undeclared mismatch is rejected, and a
  declared mismatch whose residues actually agree is rejected. Under
  `same_construct` any declared mismatch is rejected.
- Each site: unique `site_id`, unique in-range `sequence_index`, exactly two
  distinct canonical `allowed_residues`, and `decoded_sequence[index]` inside
  that support — the policy's own membership rule, caught early.

### Site order is carried, not preserved in place

`ConstrainedEditSpace` **sorts sites by ascending position**
(`esmif1_policy.py:368`), and every per-site vector the policy produces follows
that sorted order. So "preserve the caller's order" cannot be satisfied inside
the policy. The manifest carries it instead:

| Property | Meaning |
|---|---|
| `declared_site_ids` | the caller's order, as written |
| `policy_site_ids` | allele-vector order (ascending position) |
| `declared_to_policy` | where declared site *i* sits in policy order |
| `policy_to_declared` | where policy site *j* sits in declared order |

`BoundPolicy.alleles_by_site_id(alleles)` turns an allele vector into
`{site_id: residue}`, which is the reading that cannot be silently wrong. In the
example above, `site_b` is declared first but sits at position 5: a caller
reading `sample.alleles[0]` as `site_b` would be reading `site_a`, and nothing
would raise.

Each allele index must be an **actual integer** in `{0, 1}`. `True == 1` and
`0.0 == 0`, so a membership test alone accepts a bool (which then names an
allele) and a float (which then raises a bare `TypeError` from the tuple
subscript). Integral numpy and torch scalars are accepted via `operator.index`,
because those are what a sampled allele vector holds.

Allele order *within* a site is preserved exactly: `allowed_residues[0]` is
allele 0, because `EditableSite` takes allele indices from the caller's tuple
(`esmif1_policy.py:257-261`).

## Supported and rejected cases

**Supported.** Local `.pdb` and `.cif` files; one or more chains with the decoded
chain first; insertion codes; author numbering gaps; a model selected by ordinal;
a structural template whose residues deliberately differ from the edited
sequence, where the difference is declared per residue; non-backbone atoms
(`O`, side chains) in a selected chain, which are ignored.

**Rejected, always by name, never by filtering.** Within the selected model:

| Case | Why it is refused rather than handled |
|---|---|
| file size or SHA-256 differs from the declaration | a different file is a different structure |
| more than one mmCIF data block | biotite would silently take the first |
| a missing `auth_asym_id`/`auth_seq_id`/`auth_comp_id`/`auth_atom_id` column | the parser's `label_*` fallback only warns |
| an unresolved author chain identifier on **any** record in the selected model | chain selection would be ambiguous |
| an unresolved residue number, residue name or atom name on a **selected-chain** record | identity would be guessed |
| any alternate-location identifier on a selected chain | choosing one conformation silently changes the structure |
| a `HETATM` record on a selected chain | see the note below |
| a residue name outside the canonical 20 on a selected chain (`UNK`, `MSE`, ligands) | a modified residue is not a residue with a known identity |
| a mapped residue missing `N`, `CA` or `C` | the supported input contract requires complete mapped backbone atoms; missing context regions are represented as declared fragment breaks |
| two atoms with the same backbone name in one residue | ambiguous |
| a non-finite coordinate in a selected chain | `NaN` already means inter-chain padding in this packing |
| missing, non-finite, zero, negative, or greater-than-one atom occupancy on a selected chain | a coordinate row must not turn an unobserved atom into usable geometry; occupancy must be in `(0, 1]` |
| one residue's records split across two blocks, **including two blocks separated by another chain's records or by a `TER`** | ambiguous identity; biotite drops `TER` lines, so the two segments would otherwise become one residue |
| an mmCIF model serial that occupies more than one run of rows | a model is one contiguous block; an ordinal over interleaved serials selects a mixture of models |
| an mmCIF `pdbx_PDB_model_num` that is not an integer | parsed from the raw text, so a serial past `2**31 - 1` is not coerced either |
| an mmCIF `group_PDB` on a selected chain that is neither `ATOM` nor `HETATM` | `hetero` is derived as `group_PDB == "HETATM"`, so anything else reads as an ordinary atom |
| a correspondence that does not cover the entire observed decoded chain | v1 has no cropping; the rest would be encoded undeclared |
| a correspondence in a different order from the file | v1 has no reordering |
| a declared residue absent from the chain | nothing is substituted |
| a residue name that disagrees with `expected_res_name` | the declaration is the contract |
| a declared chain with no records | nothing is defaulted |
| a model ordinal beyond the file's model count | — |

**Solvent and ligands.** The `HETATM` rule is deliberately restrictive: a water
record that shares a *selected* chain's author identifier causes the whole
preparation to fail. There is therefore **no silent non-polymer exclusion path**
in v1 — nothing is quietly dropped, because dropping is how a bound ligand
becomes "not there". Records on unselected chains are never examined. A declared
allowance (`"solvent": "exclude"`, naming the residue names excluded) is the
future extension.

**Numbering gaps are an observation, not an inference.** A gap in author
numbering is reported in `findings.numbering_gaps` and the run still passes.
Author numbering is discontinuous by convention in many deposited structures, so
a gap here is evidence about the numbering and **nothing else** — in particular
it is not evidence of a spatial break or of a missing residue.

## Packing

Upstream's convention, reproduced exactly
(`esm/inverse_folding/multichain_util.py:68-77`): the decoded chain first, then,
for each further chain in declared order, `inter_chain_pad_length` rows of `NaN`
followed by that chain. `NaN` therefore appears in exactly one place, which is
upstream's own meaning for it. Confidence is the declared per-chain value on
chain rows and `1.0` on pad rows — upstream's `confidence=None` default — which
the converter's padding-mask arithmetic overwrites with `-1` regardless
(`esmif1_policy.py:930-941`), so the pad value is never read.

`test_the_packing_matches_upstream_concatenate_coords` compares the packed array
against the real `multichain_util._concatenate_coords`, so the convention is
validated against upstream rather than against a comment.

## The prepared artifact

`<name>.prepared.json`, schema `esmif1-prepared-structure/1`. It holds the
**inputs** to the encoder, never the encoding: `FixedGeometry` is device- and
dtype-bound and is explicitly not a portable artifact
(`esmif1_policy.py:523-541`).

| Section | Contents |
|---|---|
| `identity` | `structure_sha256`, `manifest_sha256`, `coordinates_sha256`, `confidence_sha256`, `content_sha256` |
| `manifest` | the validated manifest, embedded **once** |
| `packing` | convention name, pad length, total rows, chain spans, pad spans |
| `arrays` | `coordinates` and `confidence` as base64 little-endian C-order float32 |
| `site_permutations` | declared and policy site ids and both permutations |
| `findings` | numbering gaps, declared mismatches, residue counts |

There is no per-row echo of the correspondence: the rows live in the embedded
manifest, so there is no second copy for the first to drift away from. The format
is one canonical JSON document with the arrays base64-encoded as little-endian
C-order float32 — text, diffable, hashable by the same serializer as the
manifest, and byte-identical across two runs with the same inputs.

### What reloading does and does not establish

`load_prepared_structure` re-validates the **embedded content**:

- strict key sets and schema version;
- the embedded manifest through the same validator a fresh manifest passes;
- the packing convention, and the chain order against the manifest's;
- span layout, **walked in packing order rather than sorted**: chain *i* of
  `chains.order` must start where chain *i - 1* and its pad block ended, each pad
  span is the declared length, and the spans end exactly at `total_rows`. Sorting
  the union only asks whether the intervals tile the array, which every
  permutation of the chains does — two context chains with their row ranges
  exchanged tile it identically while claiming one chain's coordinates are the
  other's;
- array dtype, byte order, C ordering, declared shape and decoded length;
- `NaN` in every pad row and in no chain row; confidence finite, in `[0, 1]`,
  equal to the declared per-chain value on chain rows and to the pad convention
  (`1.0`) on pad rows;
- `identity.structure_sha256`: lowercase hex, and **equal to
  `manifest.source.sha256`** — preparation only ever records a digest it has just
  checked against the declaration, so the two cannot legitimately differ;
- the site permutations and **every finding, numbering gaps included**,
  re-derived from the embedded manifest;
- every recorded digest, recomputed.

This proves the artifact is internally consistent and unmodified since it was
written. **It is not evidence that the coordinates came from any particular
file.** A recomputed hash can only agree with the value it was computed from;
nothing in this path opens the structure — including the `structure_sha256` check
above, which compares two recorded values. `verify_against_source(prepared, path)`
is the separate, explicit re-read: it re-hashes the file, re-runs the whole
preparation, and returns one problem string per disagreement.

`check_prepared_state(prepared)` applies the same list to a **live** object,
which is what anything about to encode or report on one calls: the manifest
against its snapshot, `structure_sha256` against the digest the manifest pins,
each span as a real non-negative integer range, the arrays, and every finding
re-derived. A state that a reload would refuse is refused here too — `replace()`
and a writeable array can produce either.

Types are checked before values wherever an index or a count is compared, because
`True == 1` and `1.0 == 1`: a permutation of `[True, False]`, a residue count of
`4.0`, or a boolean `res_id` would otherwise compare equal to the derived one.

`findings.numbering_gaps` **is** re-derivable without the source, and is
re-derived: the correspondence declares every decoded residue's
`(res_id, ins_code)`, and preparation refuses unless those equal the observed
ones, so the gap list is a function of the embedded manifest alone. A fabricated
or dropped gap is a detectable disagreement, not an unverifiable claim.

## The report

`<name>.report.json`, schema `esmif1-structure-report/1`. Deterministic: no
timestamps are generated (the house rule from `benchmarks/provenance.py`), input
paths are recorded as given, and the artifact is recorded by **basename and
digest**, so two runs with the same arguments into different output directories
compare byte-identical.

`structural_checks` enumerates what was verified, filtered to the checks that
apply to the declared format, so a PDB report cannot claim that an mmCIF-only
check passed.

**The checks are re-run when the report is built, not assumed.** `build_report`
re-checks the prepared object's own invariants, re-reads and re-hashes the
manifest file it is about to name, re-reads the structure file and re-runs the
whole preparation (`verify_against_source`), and recomputes the artifact digest
it is given. Any disagreement raises instead of producing a report; there is no
path that writes `"status": "pass"` for a check that did not just run against the
files under `inputs`. That costs one extra parse of the structure file per
report, which is what the claim is worth: coordinates edited in memory with every
embedded digest recomputed are internally consistent, and only re-reading the
file can tell.

`model_integration_checks` is **`not_run`**, with a reason and four individually
named absent measurements:

```text
cached_geometry_matches_uncached_native_forward   not_run
sample_and_rescore_agree                          not_run
decoder_gradient_with_frozen_encoder              not_run
released_checkpoint_parity                        not_run
```

This tool supplies no model and loads no weights, so none of them can be
anything else.

## Writing

Two sequential writes make "a failure writes nothing" unachievable as an
absolute, so the contract is the achievable one: **inputs are never touched, an
existing output is never replaced, and any file the call created is removed again
if a later step fails.**

- `--name` must be a plain, portable file stem. A path separator, `..`, a drive
  letter, `:` (an NTFS alternate data stream), the characters Win32 forbids in a
  filename (`< > " | ? *`, the last two being wildcards), a control character, a
  trailing dot or space, and the Windows device names (`CON`, `NUL`, `COM1` …,
  the superscript `COM¹`/`LPT³` forms included, which stay devices however the
  extension reads) are all refused. The same rules apply to the manifest's
  `source.relative_path`, so a declaration cannot address a drive, a UNC share or
  a hidden stream; `structure_path_for` additionally resolves the result —
  symlinks included — and refuses anything outside the structure root.
- **Both documents are serialized and encoded to UTF-8 bytes before either file
  is created**, so the one failure that is not an OS error — a document that
  cannot be encoded — happens while nothing exists on disk, the output directory
  included. Serializing to text is not enough: a lone surrogate survives
  `json.dumps` and fails at the encode, which used to happen between the two
  writes. A JSON failure propagates unwrapped, as the `TypeError` the caller's
  object caused; a document that is not UTF-8 encodable is a
  `StructurePreparationError`.
- Both outputs are resolved and compared against every input path, so an output
  can never alias an input.
- Both outputs are preflighted for non-existence, then created with `open(..., "xb")`,
  which is atomic and refuses a dangling symlink as well as a real file.
- Only a path the call **created** is ever unlinked, and only after an OS error
  while writing; a pre-existing file is never removed.
- There is **no `--force`** and no overwrite path at all.
- Expected input and IO failures are named as this package's own errors rather
  than escaping raw: a manifest or artifact file that is unreadable or not UTF-8,
  an `OSError` while stat-ing or hashing the declared structure, a confidence with
  no float image (`10**400`), a malformed mmCIF model number. The CLI exits 2 with
  one line on stderr for those; a programming error is deliberately **not** caught,
  so it still prints its traceback.

## Using it

```python
from smallAntibodyGen.structure import load_manifest, prepare_structure
from smallAntibodyGen.structure.policy_adapter import bind_policy

manifest = load_manifest("toy.manifest.json")          # stdlib only; strict
prepared = prepare_structure(manifest, "toy_complex.pdb")   # biotite, imported lazily

bound = bind_policy(prepared, model, alphabet=alphabet)     # model is SUPPLIED
log_q = bound.policy.log_prob([candidate], bound.geometry)  # differentiable
sample = bound.policy.sample(bound.geometry, num_samples=8)
named = bound.alleles_by_site_id(sample.alleles[0])         # {site_id: residue}
```

Importing `smallAntibodyGen.structure` pulls in neither biotite nor torch;
`policy_adapter` is a separate module because it imports torch.

```powershell
.\.venv\Scripts\python.exe src\smallAntibodyGen\tests\fixtures_esmif1_structure.py outputs\scratch\toy
.\.venv\Scripts\python.exe scripts\prepare_esmif1_structure.py `
    --manifest outputs\scratch\toy\toy.manifest.json `
    --output-dir outputs\scratch\toy\prepared --name toy
```

The fixture writer generates a **clearly synthetic** complex: a straight line of
CA atoms 3.8 Å apart with no side chains and no chemistry. Nothing measured from
it is evidence about any protein.

## Obtaining reference inputs

The released model is `esm_if1_gvp4_t16_142M_UR50`, available from
[Meta's official model table](https://github.com/facebookresearch/esm#pre-trained-models).
Keep the checkpoint under the ignored `checkpoints/` directory and record its
SHA-256, source URL, and the upstream code revision used to load it.

[RCSB PDB](https://www.rcsb.org/docs/programmatic-access/file-download-services)
provides experimental coordinates and biological assembly files in mmCIF.
[SAbDab](https://sabdab.opig.stats.ox.ac.uk/) provides antibody annotations to help
identify the relevant chains. Pin the exact coordinate file; original author
numbering and renumbered antibody exports must not be mixed in one mapping.
Keep downloaded inputs under ignored `data/raw/`.

A small non-pathogen reference structure and the released weights are sufficient
to begin integration checks; a training corpus is not required. The declared
sequence, chain/model selection and residue correspondence must still be supplied.
Affinity evaluation additionally needs a separate assay dataset; coordinates
alone do not establish binding measurements. The v1 rejections below still apply.

## Limitations

- **Unit fixtures are synthetic.** The later
  [5CJQ preparation](../reference/cr9114-5cjq-context.md) supplies real-structure
  evidence for that declared context. It does not broaden the supported parser
  cases to arbitrary alternate locations, modified residues, or solvent chains.
- **No correspondence is derived.** The table is declared, never inferred. There
  is deliberately no alignment or scaffold-matching helper, because deriving a
  residue mapping is exactly the guessed mapping this layer exists to prevent.
- **Missing mapped coordinates are refused.** Upstream's two `NaN` branches
  (a `NaN` N becomes padding; a `NaN` CA only clears `coord_mask`) are outside
  this contract. Declared fragment breaks handle missing context regions.
- **One conformation only.** Alternate locations are refused, so a structure that
  carries any altloc on a selected chain cannot be prepared at all.
- **Canonical residues only** in a selected chain, which excludes selenomethionine
  and every modified residue.
- **Author numbering only.** `label` numbering is rejected rather than supported.
- **Unit tests do not load the released checkpoint.** The optional tests run a real
  `GVPTransformerModel` — a ~32-dimension one built from scratch with **random
  weights** — so they verify wiring. Released-checkpoint evidence is supplied by
  the subsequent pilot, not by these unit tests; random-weight outputs are not
  biological results.
- **`verify_against_source` is not automatic on load.** An ordinary
  `load_prepared_structure` validates embedded content only, and says so.
  `build_report` does run it, which is why a report costs a second parse.

## Integration status

Structure preparation, released-weight checks, device measurements, and SFT/DPO
integration are complete for the recorded CR9114 pilots; see the
[integration evidence](esmif1_policy.md#integration-status). The template's
relationship to the assay remains a stated proxy-context limitation. The
standalone p-IgGen/HER2 study has no structural-input dependency.

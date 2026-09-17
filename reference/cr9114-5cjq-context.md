# Prepared CR9114 / 5CJQ structural context

Prepared and source-verified on 2026-09-16. The model input is available locally
in `data/processed/cr9114_5cjq_context/` (git-ignored). Preparation itself loads no
weights. The subsequent [scoring and supervised pilot](cr9114-5cjq-pilot.md)
completed successfully; its result is separate from the original preparation report.

| Component | Retained context |
|---|---|
| Decoded heavy variable domain | 121 residues; all 16 verified binary sites |
| Partner light variable domain | 109 deposited residues |
| Antigen context | All three protomers from deposited biological assembly 1 |
| Missing antigen regions | Separate contiguous observed fragments, with 10 NaN rows between encoder chains |
| Total encoder input | 848 coordinate-bearing residues + 130 separator rows = 978 rows |

The identity-copy VH and VL retain the antibody variable-domain geometry while
excluding constant domains. All three antigen protomers retain the deposited
trimer context. The other two Fabs are omitted: this declares the context for
one antibody, without conditioning on additional symmetry-related antibodies.
This selection is an experimental modeling choice, not a demonstrated optimum.

Author residue numbers and insertion codes are preserved. Antigen fragment IDs
encode source chain, assembly operator, and observed run. The provenance file
records each fragment's original polymer-index range and the exact rigid
assembly operators. Independent assembly construction agreed within 0.00003
Angstrom. No loops were filled, coordinates minimized, or residues replaced.

Two fixed VH template differences (H24 and H46) are explicitly declared. The
decoded sequence uses the released somatic-16 benchmark sequence, with both
benchmark alleles available at each editable site in genotype order.

The light-chain check used the annotated translation of antibody accession
[JX213640.1](https://www.ebi.ac.uk/ena/browser/view/JX213640.1), found in both
released antibody supplements. Its 110-residue VL starts `SYV`; the 109-residue
template VL starts `SA`. The remaining 107 residues match. No ambiguous
residue-level alignment is imposed on those differing prefixes.

The engineered antigen stem remains a proxy for the assayed H1 ectodomain,
and Fv geometry is a proxy for the assayed scFv. Missing antigen geometry remains
absent; fragment separators introduce artificial ends. The 3.6-Angstrom template
retains the local quality limitations in the [mapping audit](cr9114-5cjq-mapping.md).
Confidence 1.0 is an input convention, not a measured confidence score.

## Files and reproduction

- `cr9114_5cjq_context.cif`: derived coordinates, with explicit fragment chains.
- `cr9114_5cjq.manifest.json`: source hash, correspondence, fixed mismatches, and edit space.
- `cr9114_5cjq.prepared.json`: portable packed coordinates and validated declaration.
- `cr9114_5cjq.report.json`: source verification and explicitly unrun model checks.
- `context-provenance.json`: parent source, light-chain audit, operators, selections, and output hashes.

The compact [preparation evidence](evidence/cr9114-5cjq-context-2026-09-16.json)
is retained with the repository. The original pinned raw files plus
`data/raw/cr9114_structure/JX213640.1.embl` are required to reproduce:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_cr9114_5cjq_context.py --output-dir outputs/cr9114_5cjq_context
```

The output directory must not already exist. The script reruns the mapping audit,
checks the light-chain reference hash, cross-checks assembly transforms, prepares
the artifact, re-verifies against source files, and validates the policy edit
space. A second run produced byte-identical coordinates, manifest, and prepared
artifact; report paths differ by output location.

Released-model scoring parity on this context, including cached versus uncached
geometry, is now recorded in the subsequent pilot. Preparation alone does not
establish scoring correctness or biological performance.

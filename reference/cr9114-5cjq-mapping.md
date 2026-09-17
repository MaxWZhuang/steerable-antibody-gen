# CR9114 benchmark mapping onto 5CJQ

**Verified, 2026-09-16:** all 16 benchmark sites map to observed 5CJQ heavy-chain
residues. All 121 residues of the released benchmark VH have N/CA/C coordinates.
This completes the sequence/site correspondence audit, not model preparation.

**Subsequent preparation:** the [structural context](cr9114-5cjq-context.md)
has now been prepared and source-verified. The remaining-input-work section below
records the handoff from this earlier mapping audit; scoring parity is still pending.

- [Site table (CSV)](evidence/cr9114-5cjq-sites-2026-09-16.csv)
- [Full mapping and local validation evidence](evidence/cr9114-5cjq-residue-mapping-2026-09-16.json)
- [Pinned input files, URLs, sizes and hashes](evidence/cr9114-5cjq-mapping-sources-2026-09-16.json)
- [Reproducible audit](../scripts/audit_cr9114_5cjq_mapping.py)

## What was verified

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8476123/) supplies
the two antibody plasmid files (supplements 4 and 5), antibody fragment labels
(supplement 2), and the archived analysis-code revision
`61c1673a101ea739d5b7e9b282f6bcfad41d7e90`.

The audit translates the VH region between the published scFv linker and the
annotated Myc tag in each antibody file. The 16 sequence differences agree with
the supplementary fragment amino-acid labels and with the translated binary
definitions in `CR9114/Seq_pipeline/nuc_to_aa.py`. Their order agrees with the
IMGT labels in `CR9114/Epistasis_linear_models/mutation_info.py`. Upstream
Python files are parsed as literal data, never executed.

All 65,536 released genotype strings are unique, retain their leading zeros,
and agree with `pos1..pos16`. State **0** is the released germline allele; state
**1** is the released somatic-16 allele. Binding measurements are not used to
choose or validate the mapping.

The 121-residue somatic-16 VH has one unique best ungapped match to the deposited
heavy-chain polymer sequence. The mmCIF polymer-sequence scheme supplies the
author residue identifiers, including insertion codes. These are distinct
from both the paper's IMGT positions and ordinary sequence indices.

| Genotype column | Paper IMGT | VH position (1-based) | 5CJQ author residue | Alleles 0 / 1 |
|---|---|---|---|---|
| pos1 | 30 | 29 | H29 | F / S |
| pos2 | 35 | 30 | H30 | S / N |
| pos3 | 36 | 31 | H31 | S / N |
| pos4 | 57 | 52 | H52 | I / S |
| pos5 | 64 | 57 | H56 | T / S |
| pos6 | 65 | 58 | H57 | A / T |
| pos7 | 66 | 59 | H58 | N / A |
| pos8 | 79 | 71 | H70 | T / S |
| pos9 | 82 | 74 | H73 | K / I |
| pos10 | 83 | 75 | H74 | S / F |
| pos11 | 84 | 76 | H75 | T / S |
| pos12 | 85 | 77 | H76 | S / N |
| pos13 | 92 | 84 | H82A | S / N |
| pos14 | 95 | 87 | H83 | R / T |
| pos15 | 103 | 95 | H91 | Y / F |
| pos16 | 113 | 106 | H100B | Y / S |

## Findings that preparation must preserve

**Two fixed template mismatches.** At sequential VH positions 24 and 46
(author H24 and H46), 5CJQ has S and D, while both benchmark endpoint sequences
have A and E. These positions are not among the 16 editable sites. A future
manifest must declare these sequence/geometry mismatches; copying the PDB
sequence as the benchmark reference would be wrong. This conclusion comes from
the released sequences, independently of numbering differences in the paper's
descriptions of excluded mutations.

**An unused upstream lookup is inconsistent.** The `aa_dict_list` entry at
pos12 labels ATT as N although its standard translation is I. The upstream
binary table instead uses AAT for state 1; its translation and the two antibody
sequences consistently establish N. The audit records the inconsistency and
does not rely on that unused amino-acid lookup.

**Local quality is mixed.** All 16 variable sites have favored Ramachandran
assignments and none has an RSRZ > 2 density-fit flag. The official validation
data flag clashes at H73, H74 and H100B; these include a backbone N at H74.
Within the rest of the VH, H61 has RSRZ 2.773. The pinned evidence preserves
the per-residue metrics. These findings support a usable mapping, not a claim
that the 3.60-Angstrom structure is uniformly precise or repaired.

**Missing residues are outside the benchmark VH, but remain relevant.** The
heavy-chain omissions are at polymer sequence positions 138-141 and 222-230.
HA1 and HA2 also contain internal missing regions, not just absent termini.
The adapter's check for backbone atoms on observed residues does not establish
that the complete antigen chain is present. Any treatment of these regions
must be explicit before scoring.

**The assay target has been identified.** The paper names
A/New Caledonia/20/1999 H1 ectodomain and provides its construct source in
supplement 8. Its exact sequence was not extracted in this antibody mapping
audit. The engineered stem #4900 in 5CJQ remains a different-construct proxy.
The benchmark used an scFv; the structural antibody is a Fab. Thus neither the
antigen nor the complete antibody context should be described as assay-matched.

## Remaining model-input work

Use the verified **121-residue VH** as the intended decoded sequence. The
current adapter requires the entire observed decoded chain (217 residues for
5CJQ H), so it cannot directly apply that extent to the untouched download.
A documented derived structure or an explicit selection extension is needed;
constant-domain residues must not silently become benchmark sequence positions.

Before creating that input, finish the light-chain correspondence and decide
the retained assembly/context and treatment of internal antigen gaps. Then
create the manifest, run preparation and source verification, and test scoring
parity with the released model. No model scoring or training was run in this
audit.

To reproduce with the pinned raw files present, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/audit_cr9114_5cjq_mapping.py --output-dir outputs/cr9114_mapping_audit
```

The script rejects changed input hashes, contradictory allele definitions,
ambiguous correspondence, missing VH backbone atoms, and existing outputs.

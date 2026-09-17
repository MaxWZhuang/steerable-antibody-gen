# 5CJQ working structural template

**Preparation update:** the [derived structural context](cr9114-5cjq-context.md)
is now prepared and source-verified. The selection-time checklist below is
historical; model scoring and exact assay-construct correspondence remain open.

**Decision: 2026-09-16.** The user selected **5CJQ** as the working template
for the CR9114/H1 fixed-target experiment using ESM-IF1. This selects the source
structure; it does not establish a prepared model input or close the structural
integration gate.

**Mapping update, 2026-09-16:** the [benchmark correspondence audit](cr9114-5cjq-mapping.md)
verified all 16 sites and all 121 VH residues, including two fixed differences
between the template and the benchmark. That audit also checks mmCIF coordinates
and local validation; the inventory below describes the earlier selection step.

## Biological relationship

[5CJQ](https://www.rcsb.org/structure/5CJQ) contains CR9114 bound to engineered
hemagglutinin stem construct **#4900**. The
[associated publication](https://doi.org/10.1126/science.aac7263) describes the
mini-HA design as H1-derived. Treat it as **proxy structural context** for the
H1 binding benchmark. A match to the exact assayed antigen has not been
established. The entry's "Mutation(s): No" annotation does not make this
synthetic construct native HA.

The published provenance and CR9114 complex support this working choice.
The 3.60-Angstrom resolution and reported R-free of 0.369 motivate local
validation before use; they are not a certificate of reliable coordinates at
every benchmark site. The
[wwPDB validation report](https://files.rcsb.org/validation/view/5cjq_full_validation.pdf)
is the source for subsequent local-quality assessment.

## Pinned downloads and coordinate inventory

The unmodified PDB and mmCIF files are saved under
`data/raw/cr9114_structure/5CJQ.{pdb,cif}` (git-ignored). Their download URLs,
byte counts, SHA-256 digests, and audit scope are recorded in the
[selection evidence](evidence/5cjq-structure-selection-2026-09-16.json).

The PDB audit uses author chain identifiers and includes insertion codes in
residue identity. These are asymmetric-unit coordinates; no biological-assembly
transforms, cropping, residue substitutions, or coordinate repair were applied.

| Author chain | Identity | Observed / deposited residues |
|---|---|---|
| H | CR9114 heavy chain | 217 / 230 |
| L | CR9114 light chain | 211 / 215 |
| A | Engineered HA1 | 52 / 66 |
| B | Engineered HA2 | 154 / 193 |

All 634 observed residues have exactly one N, CA, and C atom. These chains have
no alternate-location atom records, HETATM records, or nonfinite atom
coordinates. This does **not** account for the 70 deposited residues without
coordinates and does not verify local density support or the benchmark mapping.
The mmCIF file is hash-pinned but was not independently coordinate-audited.

## Before model preparation

1. Complete the exact assay-construct correspondence. The paper identifies
   A/New Caledonia/20/1999 H1 and supplies a construct source; #4900 is a proxy.
2. Carry the verified VH/site mapping and two fixed mismatches into preparation.
3. Complete light-chain correspondence and assess local validation and missing
   regions for the intended multichain context.
4. Declare the assembly, included chains, decoded sequence extent, and handling
   of unobserved residues. The current adapter does not support cropping.
5. Create the structural-input manifest and run the preparation CLI. This
   selection record is not a manifest accepted by that CLI.

No model preparation, weight loading, training, or sequence generation was run
as part of this selection. See the
[structural-input specification](../specs/esmif1_structure.md) for its contract.

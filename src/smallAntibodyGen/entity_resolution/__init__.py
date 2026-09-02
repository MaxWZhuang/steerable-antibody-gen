"""
Primitives for entity resolution and leakage-safe evaluation.

Four modules, each doing one job that the others do not:

- `alignment` — the metric. Local affine Smith-Waterman, reporting identity and
  coverage separately so that a truncation and a divergence stay distinguishable.
- `blocking` — candidate generation, with derived necessary conditions and a
  measured recall. A pair never proposed cannot be recovered downstream, so this
  is part of the correctness contract rather than an optimisation.
- `clustering` — bounded (complete-linkage) grouping, plus the single-linkage it
  replaces, exported so fault-injection tests can prove the criterion is doing
  something.
- `synthetic` — a planted corpus with known answers, implemented independently of
  everything above so it can measure them.
- `conformance` — the claim plane: controlled vocabularies, the Level-1 evidence
  artifacts, the sealed claim reducer, and a validator that fails.

`smallAntibodyGen.target_identity` is the domain layer built on top. The split
exists because the primitives here are about strings and graphs and are reusable
for any corpus, while the evidence-to-action table above them is about antigens
and curators specifically.

The normative requirements these implement are in
`specs/entity-resolution-conformance.md`, and the reasoning behind the operating
point is in `specs/decisions/0002-typed-target-identity.md`.
"""

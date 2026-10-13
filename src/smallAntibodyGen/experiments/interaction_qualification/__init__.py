"""Qualification data/probe slice: do frozen joint antibody-antigen MINT features carry more
than antibody-only or independently encoded features, on explicit exact-pKD observations?

``config`` and ``records`` are pure stdlib (plus ``models.antibody_antigen.views``): they parse
the closed config/manifest/record schemas and audit the panel splits. ``arms`` builds the three
frozen-feature arms from visible views only. ``probes`` fits the train-only fixed ridge and
scores it per partition/source/assay. Assay, source, group and record identifiers are audit,
split and metric metadata; they never reach a backend or a predictor.
"""

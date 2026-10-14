"""Explicit document loading; no dataset discovery or outcome filtering."""
from dataclasses import MISSING, asdict, fields
from pathlib import Path

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    digest, file_digest, read_json, require_keys,
)
from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    GroupingDeclaration, InputRecord, ManifestDeclaration, QualificationRule,
    SourceDeclaration, qualify_manifest,
)
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest, MeasuredEndpoint,
    RestrictedPostData, SelectionAllowance, select_budget,
)


def record(cls, value, converters=None):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    allowed = {f.name for f in fields(cls) if f.init}
    required = {f.name for f in fields(cls) if f.init and f.default is MISSING and f.default_factory is MISSING}
    if set(value) - allowed or required - set(value):
        raise ValueError(f"{cls.__name__}: unknown or missing fields")
    converted = dict(value)
    for key, convert in (converters or {}).items():
        if key in converted:
            converted[key] = convert(converted[key])
    return cls(**converted)


def sequence(value):
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a sequence")
    return tuple(value)


def optional_sequence(value):
    return None if value is None else sequence(value)


def coords(value):
    return None if value is None else tuple(sequence(c) for c in sequence(value))


def load_broad(path):
    document = read_json(path)
    require_keys(document, ("schema", "declaration", "records"), "paired data")
    if document["schema"] != "trastuzumab-pairs/1":
        raise ValueError("unsupported paired-data schema")
    declaration = record(ManifestDeclaration, document["declaration"], {
        "sources": lambda x: tuple(record(SourceDeclaration, s) for s in sequence(x)),
        "rules": lambda x: tuple(record(QualificationRule, r) for r in sequence(x)),
        "source_priority_order": sequence,
        "grouping": lambda x: record(GroupingDeclaration, x, {
            "separation_axes": sequence,
            "linked_groups": lambda g: None if g is None else tuple(frozenset(sequence(v)) for v in sequence(g)),
        }),
        **{name: sequence for name in ("excluded_record_ids", "excluded_content_keys",
                                      "excluded_antibody_families", "excluded_target_families")},
    })
    records = tuple(record(InputRecord, r, {"h_eligible": sequence,
                                           "l_eligible": optional_sequence,
                                           "aux_region": coords}) for r in sequence(document["records"]))
    population = qualify_manifest(declaration, records)
    identity = {"input_sha256": file_digest(path), "declaration": population.manifest_digest,
                "population": population.population_digest}
    return population, identity


def load_endpoint_manifest(path):
    document = read_json(path)
    require_keys(document, ("schema", "entries"), "endpoint manifest")
    if document["schema"] != "trastuzumab-endpoints/1":
        raise ValueError("unsupported endpoint schema")
    return EndpointManifest(tuple(record(EndpointEntry, row, {
        "construct": lambda x: record(EndpointConstruct, x),
    }) for row in sequence(document["entries"])))


def load_post(manifest_path, labels_path, *, budget_n, budget_seed):
    manifest = load_endpoint_manifest(manifest_path)
    budget = select_budget(manifest, budget_n, budget_seed)
    # Selection is complete before opening the explicitly restricted label file.
    document = read_json(labels_path)
    require_keys(document, ("schema", "selection_digest", "measurements"), "selected labels")
    if document["schema"] != "trastuzumab-selected-labels/1" or document["selection_digest"] != budget.selection_digest:
        raise ValueError("selected-label schema or selection digest mismatch")
    labels = tuple(record(MeasuredEndpoint, r) for r in sequence(document["measurements"]))
    restricted = RestrictedPostData(manifest=manifest, budget=budget, measured=labels,
                                    fitting_count=budget_n, development_labels=(),
                                    selection_allowance=SelectionAllowance(0, ()))
    identity = {"endpoint_manifest_sha256": file_digest(manifest_path),
                "selected_labels_sha256": file_digest(labels_path),
                "manifest": manifest.manifest_digest(), "selected_data": restricted.data_digest()}
    return restricted, identity


def coverage_report(population):
    return {"schema": "trastuzumab-coverage/1",
            "manifest_digest": population.manifest_digest,
            "population_digest": population.population_digest,
            "coverage": asdict(population.coverage),
            "excluded": [asdict(r) for r in population.excluded],
            "qualification": "caller-declared positive evidence and biological groups; not independently verified"}


def source_identity():
    package = Path(__file__).resolve().parents[2]
    paths = [*Path(__file__).parent.glob("*.py"),
             *(package / "models" / "antibody_antigen").glob("*.py"),
             package / "models" / "dplm_policy.py"]
    inventory = {p.relative_to(package).as_posix(): file_digest(p) for p in sorted(set(paths))}
    return {"schema": "trastuzumab-source/1", "files": inventory, "sha256": digest(inventory)}

#!/usr/bin/env python
"""Prepare a source-linked CR9114 antibody context; no fetching or model loading.

Retain one Fv and the deposited biological antigen trimer. Represent observed
antigen fragments as separate encoder chains rather than closing missing loops.
This is a geometry proxy, not a reconstructed assay construct.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_cr9114_5cjq_mapping import (  # noqa: E402
    audit, cif_rows, read_pinned_sources, require, snapgene_packets,
)

VL_SHA256 = "05f7a22f34aba4f465a9b5cf67f8373c7c49b32ae80e0b32f6c2a4ad25e5b085"

#: Runtime configuration, not a report. The manifest pins every source file by
#: URL, size and SHA-256, and preparation cannot run without it, so it lives
#: under configs/ where a fresh clone has it. It used to be read from
#: reference/evidence/, which is local-only research material -- a clean
#: checkout could not prepare the context at all.
DEFAULT_SOURCES = "configs/cr9114_5cjq_sources.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contiguous_runs(ids):
    """Split ordered unique polymer indices, never author numbering, at gaps."""
    require(bool(ids) and all(b > a for a, b in zip(ids, ids[1:])),
            "Polymer indices must be nonempty, unique and increasing")
    runs = [[ids[0]]]
    for value in ids[1:]:
        if value != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(value)
    return runs


def check_light_chain(reference, supplements, structure_sequence):
    """Use the annotated antibody VL translation to locate its assay sequence."""
    from biotite.sequence import NucleotideSequence

    require(hashlib.sha256(reference).hexdigest() == VL_SHA256,
            "Antibody light-variable reference hash changed")
    annotations = "".join(line[21:].strip() for line in reference.decode().splitlines()
                          if line.startswith("FT"))
    translations = re.findall(r'/translation="([A-Z]+)"', annotations)
    require(len(translations) == 1, "Ambiguous reference VL translation")
    vl = translations[0]
    require(len(vl) == 110, "Unexpected reference VL extent")
    for supplement in supplements:
        packets = list(snapgene_packets(supplement))
        sequences = [data[1:].decode("ascii").upper() for kind, data in packets if kind == 0]
        features = [ET.fromstring(data) for kind, data in packets if kind == 10]
        require(len(sequences) == len(features) == 1, "Ambiguous antibody supplement")
        bounds = {}
        for name in ("HA", "Myc"):
            matches = [f for f in features[0] if f.get("name") == name]
            require(len(matches) == 1 and matches[0].get("directionality") == "1",
                    f"Ambiguous {name} annotation")
            segments = matches[0].findall("Segment")
            require(len(segments) == 1, "Unexpected tag segmentation")
            bounds[name] = tuple(map(int, segments[0].get("range").split("-")))
        dna = sequences[0][bounds["HA"][1]:bounds["Myc"][0] - 1]
        protein = str(NucleotideSequence(dna).translate(complete=True))
        linker = "S" + "GGGGS" * 5
        require(protein.count(linker) == 1, "Ambiguous scFv linker")
        prefix = protein.split(linker)[0]
        require(prefix.count(vl) == 1 and prefix.endswith(vl),
                "Released antibody does not contain the reference VL before its linker")
    # Record the exact shared suffix; do not force an ambiguous N-terminal alignment.
    require(len(structure_sequence) == 109 and structure_sequence.startswith("SA")
            and vl.startswith("SYV") and structure_sequence[2:] == vl[3:],
            "Unexpected VL/template correspondence")
    return {"reference_accession": "JX213640.1", "protein_id": "AFP87547.1",
            "reference_sha256": VL_SHA256,
            "reference_url": "https://www.ebi.ac.uk/ena/browser/api/embl/JX213640.1?download=false",
            "reference_is_partial_cds": True,
            "assay_reference_vl_length": len(vl), "template_vl_length": len(structure_sequence),
            "both_antibody_supplements_contain_reference_vl": True,
            "shared_suffix_length": 107,
            "difference": "Template prefix SA versus assay/reference SYV; remaining 107 residues agree. No unique residue-level alignment is asserted for the altered prefix.",
            "interpretation": "Partner VL geometry is a template; it is not an exact assay light-chain match."}


def prepare(root, output_dir, sources_path=None):
    import numpy as np
    from biotite.sequence import ProteinSequence
    from biotite.structure.io import pdbx
    from smallAntibodyGen.structure import (
        build_report, load_manifest, prepare_structure, prepared_to_document, write_outputs,
    )
    from smallAntibodyGen.structure.policy_adapter import build_edit_space

    require(not output_dir.exists(), "Output directory already exists; choose a new one")
    sources_path = Path(sources_path) if sources_path else root / DEFAULT_SOURCES
    sources, contents = read_pinned_sources(root, sources_path)
    mapping = audit(root, sources_path)
    cif = pdbx.CIFFile.read(io.StringIO(contents["structure"].decode()))
    rows = cif_rows(cif.block["atom_site"])
    polymer = cif_rows(cif.block["pdbx_poly_seq_scheme"])
    light = sorted((r for r in polymer if r["pdb_strand_id"] == "L"),
                   key=lambda r: int(r["seq_id"]))
    light_sequence = "".join(ProteinSequence.convert_letter_3to1(r["mon_id"])
                             for r in light[:109])
    vl_report = check_light_chain(
        (root / "data/raw/cr9114_structure/JX213640.1.embl").read_bytes(),
        [contents["germline"], contents["somatic"]], light_sequence,
    )
    generators = cif_rows(cif.block["pdbx_struct_assembly_gen"])
    require(generators == [{"assembly_id": "1", "oper_expression": str(i),
                           "asym_id_list": "A,B,C,D"} for i in (1, 2, 3)],
            "Unexpected biological assembly specification")
    operators = {r["id"]: r for r in cif_rows(cif.block["pdbx_struct_oper_list"])}
    assembly = pdbx.get_assembly(cif, assembly_id="1", model=1, altloc="all",
                                extra_fields=["occupancy", "label_seq_id"])
    source_xyz = np.array([[float(r[k]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]
                          for r in rows])
    transformed = {}
    for op in ("1", "2", "3"):
        operator = operators[op]
        matrix = np.array([[float(operator[f"matrix[{i}][{j}]"]) for j in (1, 2, 3)]
                           for i in (1, 2, 3)])
        vector = np.array([float(operator[f"vector[{i}]"]) for i in (1, 2, 3)])
        transformed[op] = source_xyz @ matrix.T + vector
        native = assembly[assembly.sym_id == int(op) - 1]
        require(len(native) == len(rows), "Assembly copy size changed")
        require(np.allclose(transformed[op], native.coord, atol=3e-5, rtol=0),
                "Independent biological-assembly coordinate check failed")
    fragments = [("H", "H", "1", list(range(1, 122))),
                 ("L", "L", "1", list(range(1, 110)))]
    for op in ("1", "2", "3"):
        for chain in ("A", "B"):
            ids = sorted({int(r["label_seq_id"]) for r in rows if r["auth_asym_id"] == chain})
            for number, run in enumerate(contiguous_runs(ids), 1):
                fragments.append((f"{chain}{op}{number}", chain, op, run))
    derived_rows, fragment_report = [], []
    for name, chain, op, ids in fragments:
        indices = [i for i, r in enumerate(rows)
                   if r["auth_asym_id"] == chain and int(r["label_seq_id"]) in ids]
        observed_ids = sorted({int(rows[i]["label_seq_id"]) for i in indices})
        require(observed_ids == ids, f"Incomplete selected fragment {name}")
        for index in indices:
            atom = dict(rows[index])
            atom.update(id=str(len(derived_rows) + 1), auth_asym_id=name, label_asym_id=name)
            for axis, value in zip(("Cartn_x", "Cartn_y", "Cartn_z"), transformed[op][index]):
                atom[axis] = f"{value:.6f}"
            derived_rows.append(atom)
        fragment_report.append({"derived_chain": name, "source_author_chain": chain,
                                "assembly_operator": op, "label_seq_start": ids[0],
                                "label_seq_end": ids[-1], "residue_count": len(ids)})
    derived = pdbx.CIFFile()
    block = pdbx.CIFBlock()
    block["atom_site"] = pdbx.CIFCategory({key: [row[key] for row in derived_rows]
                                          for key in derived_rows[0]})
    derived["CR9114_5CJQ_context"] = block
    output_dir.mkdir(parents=True)
    structure_path = output_dir / "cr9114_5cjq_context.cif"
    derived.write(structure_path)
    chain_order = [r[0] for r in fragments]
    manifest = {
        "schema_version": "esmif1-structure-manifest/1", "kind": "esmif1_structure_manifest",
        "source": {"relative_path": structure_path.name, "sha256": digest(structure_path),
                   "size_bytes": structure_path.stat().st_size, "format": "mmcif",
                   "model_ordinal": 1, "residue_numbering": "author",
                   "attribution": {"source": "https://www.rcsb.org/structure/5CJQ",
                                   "detail": "Derived from biological assembly 1: one Fv plus antigen trimer; see context-provenance.json for parent hash, transforms, crops and fragment identities.",
                                   "license": "wwPDB archive data: CC0"}},
        "chains": {"order": chain_order, "decoded_chain": "H",
                   "confidence": {chain: 1.0 for chain in chain_order}, "inter_chain_pad_length": 10},
        "conventions": {"altloc": "reject_any", "missing_backbone_atom": "reject"},
        "structural_relationship": {"kind": "template_for_different_construct",
                                    "description": "Engineered H1 stem #4900/Fab geometry proxies an H1 ectodomain/scFv assay. Retain one Fv and all three antigen protomers; omit other Fabs and constant domains. VL N terminus differs. Split observed antigen fragments at missing polymer positions; no loop completion or coordinate repair."},
        "decoded_sequence": mapping["reference_sequences"]["somatic16_vh"],
        "correspondence": [{"sequence_index": r["vh_index_0based"], "chain_id": "H",
                            "res_id": r["author_res_id"], "ins_code": r["insertion_code"],
                            "expected_res_name": ProteinSequence.convert_letter_1to3(r["structure_residue"]),
                            "sequence_residue": r["benchmark_somatic_residue"],
                            "mismatch": {"reason": "Fixed benchmark residue differs from deposited template; preserve the benchmark sequence and unmodified template geometry."} if r["fixed_mismatch"] else None}
                           for r in mapping["vh_correspondence"]],
        "edit_space": {"sites": [{"site_id": s["site_id"], "sequence_index": s["vh_index_0based"],
                                   "allowed_residues": [s["allele_0"], s["allele_1"]],
                                   "attribution": {"source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8476123/",
                                                   "detail": "Verified released antibody endpoint sequences, binary genotype order and 5CJQ author mapping; see cr9114-5cjq-residue-mapping-2026-09-16.json."}}
                                  for s in mapping["sites"]]},
        "notes": "Confidence 1.0 is an explicit input convention, not experimental certainty or pLDDT. Antigen fragments use native ten-row NaN chain separators. Missing positions are not imputed. 3.6-Angstrom template with documented local clashes; model parity and scoring have not run.",
    }
    manifest_path = output_dir / "cr9114_5cjq.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    space = build_edit_space(prepared)
    require(len(space.sites) == 16, "Incorrect policy edit-space size")
    artifact = prepared_to_document(prepared)
    report = build_report(prepared, manifest_path=manifest_path, structure_path=structure_path,
                          artifact_filename="cr9114_5cjq.prepared.json",
                          artifact_content_sha256=artifact["identity"]["content_sha256"])
    write_outputs(artifact, report, output_dir=output_dir, name="cr9114_5cjq",
                  input_paths=(manifest_path, structure_path))
    provenance = {
        "kind": "cr9114_5cjq_prepared_context", "date": "2026-09-16",
        "source": sources["files"]["structure"], "light_chain_audit": vl_report,
        "assembly_id": "1", "operators": [operators[str(i)] for i in (1, 2, 3)],
        "selection": "Identity-copy VH and VL; all three antigen protomers; omit constant domains and the other two Fabs.",
        "fragment_mapping": fragment_report,
        "missing_source_residues": mapping["unobserved_residues"],
        "gap_convention": "Separate contiguous observed antigen runs into distinct encoder chains with 10 NaN separator rows. Artificial fragment ends; no imputed loop geometry or assertion of full-chain continuity.",
        "coordinate_changes": "Only deposited rigid assembly operators; author residue IDs, insertions, residue types and occupancy preserved. Coordinates serialized to six decimals. No minimization or residue replacement.",
        "assembly_coordinate_check": "All three complete transformed copies agree with independent biotite get_assembly within 0.00003 Angstrom absolute tolerance.",
        "decoded_residues": 121, "editable_sites": len(space.sites),
        "physical_coordinate_residues": sum(r["residue_count"] for r in fragment_report),
        "packed_rows": prepared.num_rows,
        "fixed_vh_template_mismatches": mapping["fixed_template_mismatches"],
        "limitations": ["Engineered antigen stem and Fv geometry are proxies for ectodomain/scFv assay.",
                        "Variable-chain crops remove constant domains; VL N terminus differs from assay.",
                        "Antigen missing regions remain absent; fragment convention has not been model-validated.",
                        "3.6-Angstrom coordinates retain the reported local clashes and density-fit limitations."],
        "structural_source_verification": "pass", "policy_edit_space_validation": "pass",
        "model_parity_run": False, "model_scoring_run": False, "training_run": False,
        "outputs": {p.name: {"sha256": digest(p), "size_bytes": p.stat().st_size}
                    for p in sorted(output_dir.iterdir()) if p.is_file()},
    }
    (output_dir / "context-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {prepared.num_rows} rows: 121 decoded residues, {len(space.sites)} editable sites.")
    print(f"Output: {output_dir}")
    print("Source checks and edit space passed; no model scoring or training run.")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=None,
                        help=f"pinned-source manifest (default: {DEFAULT_SOURCES})")
    args = parser.parse_args()
    prepare(args.root, args.output_dir, args.sources)


if __name__ == "__main__":
    main()

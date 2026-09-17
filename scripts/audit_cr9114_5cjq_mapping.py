#!/usr/bin/env python
"""Audit the published CR9114 benchmark correspondence to 5CJQ, without training.

Reads pinned local release artifacts. Upstream Python is parsed as data, never
executed. Outputs are evidence, not a structure-preparation manifest. Requires
biotite from the existing esm-if1 extra; no network access or model weights.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "61c1673a101ea739d5b7e9b282f6bcfad41d7e90"
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8476123/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned_sources(root, manifest):
    document = json.loads(manifest.read_text(encoding="utf-8"))
    contents = {}
    for name, source in document["files"].items():
        data = (root / source["relative_path"]).read_bytes()
        require(len(data) == source["size_bytes"], f"{name}: size changed")
        require(hashlib.sha256(data).hexdigest() == source["sha256"],
                f"{name}: SHA-256 changed")
        contents[name] = data
    return document, contents


def snapgene_packets(data):
    offset = 0
    while offset < len(data):
        require(offset + 5 <= len(data), "Truncated SnapGene packet header")
        kind, length = data[offset], struct.unpack(">I", data[offset + 1:offset + 5])[0]
        end = offset + 5 + length
        require(end <= len(data), "Truncated SnapGene packet body")
        yield kind, data[offset + 5:end]
        offset = end


def extract_vh(data):
    from biotite.sequence import NucleotideSequence

    packets = list(snapgene_packets(data))
    require(packets[0][0] == 9 and packets[0][1].startswith(b"SnapGene"),
            "Not a SnapGene file")
    sequences = [value[1:].decode("ascii").upper() for kind, value in packets if kind == 0]
    annotations = [ET.fromstring(value) for kind, value in packets if kind == 10]
    require(len(sequences) == len(annotations) == 1, "Ambiguous sequence/annotations")
    myc = [f for f in annotations[0] if f.get("name") == "Myc"]
    require(len(myc) == 1 and myc[0].get("directionality") == "1", "Ambiguous Myc anchor")
    segments = myc[0].findall("Segment")
    require(len(segments) == 1, "Unexpected Myc segmentation")
    start, end = map(int, segments[0].get("range").split("-"))
    dna = sequences[0]
    require(str(NucleotideSequence(dna[start - 1:end]).translate(complete=True)) == "EQKLISEEDL",
            "Myc annotation does not translate as expected")
    # A bounded region ending at the annotated Myc start; the published scFv
    # linker identifies the VH start independently of the structure alignment.
    window_start = start - 1 - 900
    require(window_start >= 0, "Insufficient sequence before Myc")
    translated = str(NucleotideSequence(dna[window_start:start - 1]).translate(complete=True))
    linker = "S" + "GGGGS" * 5
    require(translated.count(linker) == 1, "Ambiguous scFv linker")
    vh = translated.split(linker)[1]
    require(len(vh) == 121 and "*" not in vh, "Unexpected VH extent")
    return vh


def literal_assignment(data, name, numpy_array=False):
    tree = ast.parse(data.decode("utf-8"))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(matches) == 1, f"Ambiguous upstream assignment: {name}")
    value = matches[0]
    if numpy_array:
        require(isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "array" and len(value.args) == 1,
                f"Unexpected array declaration: {name}")
        value = value.args[0]
    return ast.literal_eval(value)


def fragment_mutation_labels(data):
    # Independent amino-acid position labels in the antibody supplement. This
    # audit does not assemble, export, or design nucleotide constructs.
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        strings = ["".join(node.itertext()) for node in
                   ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        labels = set()
        for row in sheet.findall(".//m:row", ns):
            for cell in row.findall("m:c", ns):
                if cell.get("r", "").startswith("C") and cell.get("t") == "s":
                    name = strings[int(cell.find("m:v", ns).text)]
                    labels.update((int(pos) - 1, before, after)
                                  for before, pos, after in re.findall(r"([A-Z])(\d+)([A-Z])", name))
    return labels


def verify_alleles(germline, somatic, binary, imgt, fragment_labels):
    from biotite.sequence import NucleotideSequence

    require(len(germline) == len(somatic), "VH lengths differ")
    indices = [i for i, (a, b) in enumerate(zip(germline, somatic)) if a != b]
    require(len(indices) == len(binary) == len(imgt) == 16, "Expected 16 sites")
    require(len(set(imgt)) == 16, "Duplicate IMGT site")
    require(fragment_labels == {(i, germline[i], somatic[i]) for i in indices},
            "Supplementary fragment labels disagree with VH differences")
    sites = []
    for ordinal, (index, codons, label) in enumerate(zip(indices, binary, imgt), 1):
        require(len(codons) == 2 and set(codons.values()) == {"0", "1"},
                "Invalid binary allele definition")
        alleles = {state: str(NucleotideSequence(codon).translate(complete=True))
                   for codon, state in codons.items()}
        require(alleles == {"0": germline[index], "1": somatic[index]},
                f"Site {ordinal}: binary direction or site order disagrees")
        sites.append({"site_id": f"pos{ordinal}", "genotype_index": ordinal - 1,
                      "imgt_position": label, "vh_index_0based": index,
                      "vh_position_1based": index + 1,
                      "allele_0": alleles["0"], "allele_1": alleles["1"]})
    return sites


def cif_rows(category):
    columns = {name: column.as_array(str).tolist() for name, column in category.items()}
    return [{name: values[i] for name, values in columns.items()}
            for i in range(category.row_count)]


def residue_key(row):
    insertion = row["pdbx_PDB_ins_code"]
    return (row["auth_asym_id"], int(row["auth_seq_id"]),
            "" if insertion in (".", "?") else insertion)


def map_vh(somatic, polymer, observed):
    from biotite.sequence import ProteinSequence

    heavy = [r for r in polymer if r["pdb_strand_id"] == "H"]
    heavy.sort(key=lambda r: int(r["seq_id"]))
    sequence = "".join(ProteinSequence.convert_letter_3to1(r["mon_id"]) for r in heavy)
    scores = [sum(a != b for a, b in zip(somatic, sequence[start:start + len(somatic)]))
              for start in range(len(sequence) - len(somatic) + 1)]
    require(scores and min(scores) == 2 and scores.count(2) == 1 and scores[0] == 2,
            "Expected unique ungapped VH match at heavy-chain N terminus")
    mismatch_indices = [i for i, a in enumerate(somatic) if a != sequence[i]]
    require(mismatch_indices == [23, 45] and somatic[23] == "A" and somatic[45] == "E"
            and sequence[23] == "S" and sequence[45] == "D",
            "Unexpected fixed-position template mismatches")
    correspondence = []
    for index, row in enumerate(heavy[:len(somatic)]):
        ins = "" if row["pdb_ins_code"] in (".", "?") else row["pdb_ins_code"]
        key = ("H", int(row["pdb_seq_num"]), ins)
        atoms = observed.get(key, [])
        require(all(sum(a["auth_atom_id"] == name for a in atoms) == 1
                    for name in ("N", "CA", "C")), f"Missing/duplicate backbone: {key}")
        require(all(a["auth_comp_id"] == row["mon_id"] for a in atoms), "Residue mismatch")
        correspondence.append({"vh_index_0based": index, "label_seq_id": int(row["seq_id"]),
                               "author_chain": "H", "author_res_id": key[1],
                               "insertion_code": ins, "structure_residue": sequence[index],
                               "benchmark_somatic_residue": somatic[index],
                               "fixed_mismatch": index in mismatch_indices,
                               "complete_backbone": True})
    return correspondence


def validation_metrics(data):
    root = ET.fromstring(gzip.decompress(data))
    result = {}
    for residue in root.iter("ModelledSubgroup"):
        if residue.get("model") != "1":
            continue
        key = (residue.get("chain"), int(residue.get("resnum")), residue.get("icode", "").strip())
        require(key not in result, f"Ambiguous validation residue: {key}")
        result[key] = {
            "rsrz": float(residue.get("rsrz")) if residue.get("rsrz") is not None else None,
            "rscc": float(residue.get("rscc")) if residue.get("rscc") is not None else None,
            "ramachandran": residue.get("rama"),
            "sidechain_rotamer": residue.get("rota"),
            "geometry_outlier_counts": {tag: len(residue.findall(tag)) for tag in
                                        ("bond-outlier", "angle-outlier", "clash", "plane-outlier")},
        }
    return result


def audit(root, sources_path):
    from biotite.structure.io.pdbx import CIFFile

    sources, contents = read_pinned_sources(root, sources_path)
    germline, somatic = (extract_vh(contents[name]) for name in ("germline", "somatic"))
    binary = literal_assignment(contents["binary_code"], "binary_dict_list")
    imgt = literal_assignment(contents["site_code"], "mut_names", numpy_array=True)
    sites = verify_alleles(germline, somatic, binary, imgt,
                          fragment_mutation_labels(contents["fragment_labels"]))
    cif = CIFFile.read(io.StringIO(contents["structure"].decode("utf-8"))).block
    observed = {}
    for atom in cif_rows(cif["atom_site"]):
        require(atom["pdbx_PDB_model_num"] == "1", "Unexpected extra coordinate model")
        require(atom["label_alt_id"] in (".", "?"), "Alternate coordinates present")
        require(atom["group_PDB"] == "ATOM", "Unexpected hetero atom")
        observed.setdefault(residue_key(atom), []).append(atom)
    correspondence = map_vh(somatic, cif_rows(cif["pdbx_poly_seq_scheme"]), observed)
    validation = validation_metrics(contents["validation"])
    vh_validation = []
    for row in correspondence:
        key = ("H", row["author_res_id"], row["insertion_code"])
        require(key in validation, f"No validation data at VH residue: {key}")
        vh_validation.append({"author_res_id": key[1], "insertion_code": key[2],
                              **validation[key]})
    for site in sites:
        row = correspondence[site["vh_index_0based"]]
        site.update({k: row[k] for k in ("label_seq_id", "author_chain", "author_res_id",
                                       "insertion_code", "structure_residue", "complete_backbone")})
        require(site["structure_residue"] == site["allele_1"], "Template differs at variable site")
        key = ("H", site["author_res_id"], site["insertion_code"])
        require(key in validation, f"No validation data at site: {key}")
        site["validation"] = validation[key]
    # Cross-check genotype order without fitting or reporting binding labels.
    seen = set()
    for row in csv.DictReader(io.StringIO(contents["landscape"].decode("utf-8-sig"))):
        genotype = row["genotype"]
        require(len(genotype) == 16 and set(genotype) <= {"0", "1"}, "Invalid genotype string")
        require(genotype == "".join(str(int(row[f"pos{i}"])) for i in range(1, 17)),
                "Genotype order differs from pos1..pos16")
        require(genotype not in seen, "Duplicate genotype")
        seen.add(genotype)
    require(len(seen) == 65536, "Incomplete genotype set")
    missing = cif_rows(cif["pdbx_unobs_or_zero_occ_residues"])
    missing_summary = []
    for chain in ("H", "L", "A", "B"):
        rows = [r for r in missing if r["auth_asym_id"] == chain]
        missing_summary.append({"author_chain": chain, "count": len(rows),
                                "label_seq_ids": [int(r["label_seq_id"]) for r in rows]})
    variable_indices = {s["vh_index_0based"] for s in sites}
    require(not variable_indices.intersection({23, 45}), "Fixed positions became editable")
    # The upstream unused aa_dict_list has a codon/translation discrepancy.
    # Report it; allele calls above use binary_dict_list + actual translation.
    from biotite.sequence import NucleotideSequence
    inconsistencies = []
    for ordinal, codons in enumerate(literal_assignment(contents["binary_code"], "aa_dict_list"), 1):
        for codon, declared in codons.items():
            actual = str(NucleotideSequence(codon).translate(complete=True))
            if actual != declared:
                inconsistencies.append({"site_id": f"pos{ordinal}", "codon": codon,
                                        "declared": declared, "standard_translation": actual})
    return {
        "kind": "cr9114_5cjq_residue_mapping_evidence", "schema_version": "1",
        "audit_date": sources["retrieval_date"], "status": "mapping_verified_model_input_not_prepared",
        "sources": sources, "paper": PAPER, "upstream_commit": COMMIT,
        "method": "Translate the two pinned antibody plasmid VH regions between the published scFv linker and annotated Myc tag. Check their 16 differences against translated binary_dict_list alleles, ordered IMGT mut_names, and supplementary fragment amino-acid labels. Uniquely align the somatic VH without gaps to the deposited heavy-chain polymer sequence; use mmCIF polymer scheme to map sequential indices to author identifiers including insertion codes; require backbone coordinates for all VH residues. This is not an IMGT renumbering of the entire antibody.",
        "reference_sequences": {"germline_vh": germline, "somatic16_vh": somatic},
        "vh_length": len(somatic), "verified_genotype_count": len(seen),
        "allele_convention": "0 = released germline, 1 = released somatic16; pos1 is leftmost genotype character",
        "sites": sites, "vh_correspondence": correspondence,
        "vh_validation": vh_validation,
        "local_quality_summary": {
            "variable_sites_with_RSRZ_gt_2": [s["site_id"] for s in sites
                                             if s["validation"]["rsrz"] is not None
                                             and s["validation"]["rsrz"] > 2],
            "variable_sites_with_clash_flags": [s["site_id"] for s in sites
                                               if s["validation"]["geometry_outlier_counts"]["clash"]],
            "vh_residues_with_RSRZ_gt_2": [r for r in vh_validation
                                          if r["rsrz"] is not None and r["rsrz"] > 2],
            "interpretation": "RSRZ > 2 is the wwPDB density-fit outlier flag. Its absence does not establish high coordinate accuracy. Clash flags are preserved, not repaired; counts at two residues can refer to the same clash.",
        },
        "fixed_template_mismatches": [r for r in correspondence if r["fixed_mismatch"]],
        "unobserved_residues": missing_summary,
        "upstream_unused_translation_table_inconsistencies": inconsistencies,
        "assay_target": {"identity": "Influenza A/New Caledonia/20/1999 H1 ectodomain",
                         "source": PAPER, "construct_source_available": True,
                         "construct_source": "Paper supplementary file 8; exact construct sequence not extracted in this audit",
                         "relationship_to_5cjq": "Engineered H1-derived stem #4900 is proxy structural context, not the same construct"},
        "pending": [
            "Choose and document the model's chain/domain extent; the current adapter requires the entire observed decoded chain, whereas the benchmark VH is 121 residues and observed H is 217 residues.",
            "Account explicitly for unobserved antigen regions and the scFv-versus-Fab context difference; local backbone completeness is not experimental coordinate accuracy.",
            "Review light-chain sequence correspondence and local-quality findings before selecting the final multichain context.",
            "Prepare a source-linked structural manifest, then verify released-model scoring parity."
        ],
        "model_input_prepared": False, "model_scoring_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sources", type=Path, default=ROOT / "reference/evidence/cr9114-5cjq-mapping-sources-2026-09-16.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "cr9114-5cjq-residue-mapping-2026-09-16.json"
    table_path = args.output_dir / "cr9114-5cjq-sites-2026-09-16.csv"
    require(not report_path.exists() and not table_path.exists(), "Output already exists")
    fields = [k for k in report["sites"][0] if k != "validation"]
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: site[k] for k in fields} for site in report["sites"])
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table_path.write_text(table.getvalue(), encoding="utf-8")
    print(f"Verified {len(report['sites'])} sites across {report['vh_length']} VH residues; "
          f"checked {report['verified_genotype_count']} genotypes.")
    print(f"Evidence: {report_path}\nTable: {table_path}")


if __name__ == "__main__":
    main()

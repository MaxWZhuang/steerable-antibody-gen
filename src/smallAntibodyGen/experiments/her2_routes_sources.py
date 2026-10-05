"""Source loaders and assay qualification for every enrolled measured landscape.

One record schema (:class:`Observation`) carries every source, and every field
that could be guessed is either derived from the file or recorded as missing.
The rules that cost the most to get wrong, and where they live:

* **A blank outcome is a missing number, not a negative.** Inherited verbatim from
  the AbSci audit rule in ``her2_data.classify_outcome``. Nothing here converts an
  unmeasured mutant, antigen swap or light-chain swap into a nonbinder.
* **``Pred_affinity`` is a measured column despite its name.** In the FLAb
  Engelhart/Li releases it is the assay-calibrated ``log10(KD/nM)`` AlphaSeq
  readout (Engelhart et al., PMC9606274), not a model prediction. The column name
  is the trap; :data:`SOURCES` records the primary citation beside it.
* **Koenig is enrichment.** The release carries ``KD (nM)``, ``KD (M)``,
  ``-ln(KD (nM))`` and ``-log( KD (M) )``, and the primary paper (PMC5278476)
  describes deep mutational **phage enrichment**, not 4,275 titrations. Those
  columns are enrolled as ``family=enrichment`` with
  ``measurement_kind=enrichment``, which trains a within-study ordering and never
  an interval likelihood on a KD scale. The internal consistency of the four
  columns is checked and the residual recorded, because agreeing derived columns
  are still derived.
* **Adams is doubly censored, and the format label is a curation artefact.** The
  Tite-Seq detection window is ``1e-9.5 .. 1e-5`` M; in ``pKD = -log10(KD/M)``
  that is right-censored at ``pKD >= 9.5`` and left-censored at ``pKD <= 5``.
  FLAb's ``format`` column says ``Fab``; the primary experiment
  (eLife 10.7554/eLife.23156) was **scFv**. Both are recorded and the
  qualification carries the disagreement rather than resolving it silently.
* **1000 is not a censoring limit.** FLAb KD values exceed it (2490 among them),
  so no numeric threshold is treated as a universal assay limit anywhere. Limits
  come only from a source's declared detection window.
* **Targets stay distinct.** S1, HA, HER2, VEGF and fluorescein are different
  antigens. Nothing pools them, and a MAGMA-seq row is not automatically H1.
* **CR9114's H1 columns cannot be read.** :data:`CR9114_RESERVED_COLUMNS`
  structurally mirrors ``her2_data.OUTCOME_COLUMNS``: the loader takes an explicit
  ``usecols`` and refuses a request that names any of them. Without that guard H1
  leaks by default, because the raw release is one CSV with ``h1_*``, ``h3_*`` and
  ``fluB_*`` side by side. The genotype is resolved through the **prepared
  manifest's** ``decoded_sequence`` and ``edit_space.sites``; there is no
  placeholder sequence and no invented edit coordinate, because a loader that
  falls back to ``heavy="A"`` collapses 65,536 distinct genotypes onto one
  construct and every downstream count then describes that one construct.
* **A re-publication is not a second study.** Engelhart 2022 and the first Li
  2023 release are the same AlphaSeq experiment under two names, so a dedup key
  containing the *publication alias* keeps both copies. :data:`CANONICAL_STUDY`
  maps aliases onto one study identity and
  :attr:`Observation.observation_identity` keys on that.
* **Enrichment whose direction is unresolved trains nothing ordered.** Koenig's
  KD-named columns are enrichment-derived and the enrichment-to-pseudo-KD mapping
  is not established, so the numbers are carried as quarantined raw fields with a
  count, and the *expression* enrichment ratio -- whose direction is declared by
  the source -- is what trains.
"""
from __future__ import annotations

import csv
import io
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .her2_routes_conditions import (
    FAMILY_AFFINITY,
    FAMILY_DISPLAY,
    FAMILY_ENRICHMENT,
)
from .her2_routes_serial import HEAVY, LIGHT, Construct, edit_map
from .her2_runtime import require, sha256

#: Measurement kinds, mirroring ``models.her2_routes_heads.KINDS``.
KIND_ORDINAL = "ordinal"
KIND_INTERVAL = "interval"
KIND_POINT = "point"
KIND_ENRICHMENT = "enrichment"

CENSOR_NONE = "none"
CENSOR_LEFT = "left"
CENSOR_RIGHT = "right"

#: Partition labels a loader may write into ``Observation.raw["partition"]``. The
#: names match ``her2_routes_data.PARTITIONS``; they live here too because the
#: loaders assign them and this module may not import the data layer.
PARTITION_TRAIN_LABEL = "train"
PARTITION_DEVELOPMENT_LABEL = "development"
PARTITION_PROTECTED_LABEL = "protected"

#: The declared analysis scale for every affinity source: ``pKD = -log10(KD/M)``.
#: Higher is stronger. ``log10(KD/nM) -> pKD`` is ``9 - value``; the derivation is
#: in :func:`pkd_from_log10_nm` so exactly one place can be wrong.
SCALE_PKD = "pKD = -log10(KD/M), higher is stronger"
SCALE_ENRICHMENT = "study-internal enrichment ratio, higher is more enriched"
SCALE_ORDINAL = "sorted display bin index, higher is stronger"
SCALE_EXPRESSION = "study-internal expression enrichment ratio"

#: CR9114 columns this campaign must never read. Mirrors ``her2_data.OUTCOME_COLUMNS``.
CR9114_RESERVED_COLUMNS = frozenset({"h1_repa", "h1_repb", "h1_repc", "h1_mean", "h1_sem"})
#: The reported pKD floor for the H3 and influenza-B partitions. A value at the
#: floor is LEFT-censored in pKD, not an exact weak-affinity measurement.
CR9114_H3_FLUB_FLOOR = 6.0

#: Adams Tite-Seq detection window, in molar, from the primary paper.
ADAMS_WINDOW_MOLAR = (10.0 ** -9.5, 10.0 ** -5.0)

#: Exactly how many preamble lines the second Li release carries before its header.
LI2_PREAMBLE_LINES = 6

#: Publication alias -> the ORIGINAL study whose measurements it publishes.
#: Engelhart 2022 and the first Li release are one AlphaSeq experiment; FLAb
#: republishes it under both names. De-duplicating on the alias keeps both copies
#: and doubles that study's weight in every mixture and every threshold fit.
CANONICAL_STUDY = {
    "engelhart2022_s1": "engelhart2022_alphaseq_s1",
    "li2023_s1_a": "engelhart2022_alphaseq_s1",
    "li2023_s1_b": "li2023_ml_designed_s1",
    "adams2017_fluorescein": "adams2017_titeseq",
    "adams2017_expression": "adams2017_titeseq",
    "adams2017_primary": "adams2017_titeseq",
    "koenig2017_g6": "koenig2017_g6_phage",
    "koenig2017_expression": "koenig2017_g6_phage",
    "petersen2024_ha": "petersen2024_magmaseq",
    "petersen2024_primary": "petersen2024_magmaseq",
    "alphabind_her2_validation": "alphabind_her2",
    "cr9114_h3": "phillips2021_cr9114",
    "cr9114_flub": "phillips2021_cr9114",
    "her2_display": "her2_display_20260918",
    "paired_oas": "oas_paired",
}


def canonical_study(source):
    """The original study a source name publishes. Unknown names map to themselves."""
    return CANONICAL_STUDY.get(str(source), str(source))


#: The exact Li/AlphaSeq family -> parent-control mapping, from the release's own
#: ``MIT_*`` parent-control rows. Four POI families over three distinct parents:
#: 49 and 50 share ``MIT_14``. A modal parent taken across the pooled release
#: would assign one background to all four and turn every 51/52 variant into a
#: multi-site "mutant" of the wrong molecule.
LI_POI_PARENTS = {"AAYL49": "MIT_14_HL_scFV", "AAYL50": "MIT_14_HL_scFV",
                  "AAYL51": "MIT_91_LH_scFV", "AAYL52": "MIT_95_HL_scFV"}
#: POI counts per family in the primary release, for an enrolment audit.
LI_POI_COUNTS = {"AAYL49": 26454, "AAYL50": 26224, "AAYL51": 22473, "AAYL52": 29818}
#: The four measured targets and the per-target row count of the primary release.
LI_TARGETS = ("MIT_Target", "AlphaNeg1", "AlphaNeg2", "AlphaNeg3")
LI_ROWS_PER_TARGET = 314925
LI_TOTAL_ROWS = 1259700


@dataclass(frozen=True)
class SourceSpec:
    """What one enrolled source is, where it came from, and what it can teach."""

    name: str
    family: str
    kind: str
    target: str
    scale: str
    primary: str
    columns: tuple
    note: str = ""
    enrolled: bool = True
    reason_not_enrolled: str = ""

    def document(self):
        return {"name": self.name, "family": self.family, "measurement_kind": self.kind,
                "target": self.target, "scale": self.scale, "primary": self.primary,
                "columns": list(self.columns), "note": self.note,
                "enrolled": bool(self.enrolled),
                "reason_not_enrolled": self.reason_not_enrolled or None}


SOURCES = {
    "alphabind_her2_validation": SourceSpec(
        name="alphabind_her2_validation", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="HER2", scale=SCALE_PKD,
        primary="A-Alpha-Bio/alphabind @15d2582, manuscript/data/alphaseq/validation",
        columns=("mata_description", "matalpha_description", "alphaseq_affinity",
                 "mata_sequence", "predicted_affinity", "edit_distance", "parental_mean_aff",
                 "parental_aff_margin", "avg_off_target_affinity", "model", "id"),
        note=("alphaseq_affinity = log10(KD/nM) and is the MEASURED column. "
              "predicted_affinity is a model output and is never used as an outcome.")),
    "alphabind_her2_full": SourceSpec(
        name="alphabind_her2_full", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="HER2", scale=SCALE_PKD,
        primary="the same release, full 7,540-row file",
        columns=(), enrolled=False,
        reason_not_enrolled=("88 length-compatible rows over 39 cores from a differently composed "
                             "subset. Merging them into the specified CDR population is exactly "
                             "the composition change that would invalidate a frozen split. The "
                             "exclusion closure covers them so a later amendment can enrol them "
                             "without re-freezing.")),
    "alphabind_aab_pp489": SourceSpec(
        name="alphabind_aab_pp489", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="TIGIT", scale=SCALE_PKD, primary="the same release, fine-tuning + validation",
        columns=(), note="non-HER2 AlphaSeq; orientation and parent derived from the descriptors"),
    "alphabind_pembrolizumab": SourceSpec(
        name="alphabind_pembrolizumab", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="PD-1", scale=SCALE_PKD, primary="the same release",
        columns=(), note="scFv orientation"),
    "alphabind_vhh72": SourceSpec(
        name="alphabind_vhh72", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="SARS-CoV-2 RBD", scale=SCALE_PKD, primary="the same release",
        columns=(), note=("a nanobody: heavy-only, no light chain is invented. Encoded with the "
                          "explicit heavy-only contract asserted in her2_routes_serial.")),
    "engelhart2022_s1": SourceSpec(
        name="engelhart2022_s1", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="SARS-CoV-2 spike HR2 (S2) peptide", scale=SCALE_PKD,
        primary="https://pmc.ncbi.nlm.nih.gov/articles/PMC9606274/ via Graylab/FLAb @e05e3f0",
        columns=("Pred_affinity", "heavy", "light", "fitness"),
        note="Pred_affinity is the measured assay-calibrated log10(KD/nM), not a prediction"),
    "li2023_s1_a": SourceSpec(
        name="li2023_s1_a", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="SARS-CoV-2 spike HR2 (S2) peptide", scale=SCALE_PKD, primary="Li et al. 2023 via FLAb",
        columns=("POI", "Sequence", "Target", "Assay", "Replicate", "Pred_affinity", "HC", "LC",
                 "CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"),
        note="target/assay/replicate retained; identical derived rows are not new experiments"),
    "li2023_s1_b": SourceSpec(
        name="li2023_s1_b", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="SARS-CoV-2 spike HR2 (S2) peptide", scale=SCALE_PKD, primary="Li et al. 2023 via FLAb",
        columns=("POI", "Sequence", "Target", "Assay", "Replicate", "Pred_affinity", "HC", "LC",
                 "CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"),
        note=f"same header after exactly {LI2_PREAMBLE_LINES} preamble lines, asserted before skip"),
    "adams2017_fluorescein": SourceSpec(
        name="adams2017_fluorescein", family=FAMILY_AFFINITY, kind=KIND_INTERVAL,
        target="fluorescein", scale=SCALE_PKD,
        primary="https://elifesciences.org/articles/23156 via FLAb",
        columns=("Kd (Tite-Seq) [M]", "heavy", "light", "format", "fitness"),
        note=("doubly censored at the 1e-9.5..1e-5 M detection window. FLAb's format=Fab is a "
              "curation label; the primary Tite-Seq experiment was scFv, and the disagreement is "
              "carried in the qualification rather than resolved.")),
    "adams2017_primary": SourceSpec(
        name="adams2017_primary", family=FAMILY_AFFINITY, kind=KIND_INTERVAL,
        target="fluorescein", scale=SCALE_PKD,
        primary="jbkinney/16_titeseq @54130c8, data/replicate_{1,2,3}.csv",
        columns=("CDR1H_AA", "CDR3H_AA", "fit_KD", "fit_KD_sigma", "expression", "unsorted"),
        note=("the three biological replicate tables behind the curated summary. Same canonical "
              "study, so the closure keeps one observation per (construct, assay, replicate) "
              "across both.")),
    "petersen2024_primary": SourceSpec(
        name="petersen2024_primary", family=FAMILY_AFFINITY, kind=KIND_INTERVAL,
        target="H1 HA", scale=SCALE_PKD,
        primary="41467_2024_48072_MOESM6_ESM.xlsx plus the author parent export",
        columns=("Variant", "Kd", "95% CI low", "95% CI high", "Replicate", "Ab"),
        note=("barcode-level MAGMA-seq rows resolved against the author parent export. Invalid "
              "confidence bounds are rejected for interval training and preserved raw.")),
    "adams2017_expression": SourceSpec(
        name="adams2017_expression", family=FAMILY_ENRICHMENT, kind=KIND_ENRICHMENT,
        target="fluorescein", scale=SCALE_EXPRESSION, primary="the same paper",
        columns=("Expression [ER]", "heavy", "light", "format", "fitness"),
        note="joined to the affinity table by complete sequence identity only"),
    "koenig2017_g6": SourceSpec(
        name="koenig2017_g6", family=FAMILY_ENRICHMENT, kind=KIND_ENRICHMENT,
        target="VEGF", scale=SCALE_ENRICHMENT,
        primary="https://pmc.ncbi.nlm.nih.gov/articles/PMC5278476/ via FLAb",
        columns=("heavy", "light", "-ln(KD (nM) )", "KD (nM)", "KD (M)", "-log( KD (M) )",
                 "fitness"),
        note=("the primary experiment is deep mutational phage ENRICHMENT. The KD-named columns "
              "are derived pseudo-KD and train ordering only; they are never given an interval "
              "likelihood on a KD scale.")),
    "koenig2017_expression": SourceSpec(
        name="koenig2017_expression", family=FAMILY_ENRICHMENT, kind=KIND_ENRICHMENT,
        target="VEGF", scale=SCALE_EXPRESSION, primary="the same paper",
        columns=("heavy", "light", "enrichment ratio", "fitness")),
    "petersen2024_ha": SourceSpec(
        name="petersen2024_ha", family=FAMILY_AFFINITY, kind=KIND_POINT,
        target="H1 HA", scale=SCALE_PKD,
        primary="https://www.nature.com/articles/s41467-024-48072-z",
        columns=("heavy", "light", "KD", "fitness"),
        note=("MAGMA-seq. The original supplements supply Kd in nM with confidence bounds; some "
              "bounds are zero, inconsistent, or do not enclose the estimate, and those are "
              "preserved raw and rejected for interval training rather than repaired.")),
    "cr9114_h3": SourceSpec(
        name="cr9114_h3", family=FAMILY_AFFINITY, kind=KIND_INTERVAL, target="H3 HA",
        scale=SCALE_PKD, primary="Phillips et al., eLife 2021 doi:10.7554/eLife.71393",
        columns=("genotype", "h3_repa", "h3_repb", "h3_repc", "h3_mean", "h3_sem"),
        note=f"reported pKD floor {CR9114_H3_FLUB_FLOOR}; a value at the floor is LEFT-censored"),
    "cr9114_flub": SourceSpec(
        name="cr9114_flub", family=FAMILY_AFFINITY, kind=KIND_INTERVAL, target="influenza B HA",
        scale=SCALE_PKD, primary="the same release",
        columns=("genotype", "fluB_repa", "fluB_repb", "fluB_repc", "fluB_mean", "fluB_sem"),
        note="same floor convention; a different antigen from H3 and never pooled with it"),
    "her2_display": SourceSpec(
        name="her2_display", family=FAMILY_DISPLAY, kind=KIND_ORDINAL, target="HER2",
        scale=SCALE_ORDINAL, primary="the pinned Buzz HER2 release",
        columns=("seq", "class", "label", "edit_distance"),
        note="sorted binding bins at 100 nM, not KD. The test split stays sealed."),
    "paired_oas": SourceSpec(
        name="paired_oas", family=FAMILY_DISPLAY, kind=KIND_ORDINAL, target="none",
        scale="no outcome", primary="data/processed/oas_paired_all",
        columns=("heavy", "light"),
        note=("no outcome at all: this source trains reconstruction and prior retention only. "
              "Call it PROXY replay unless p-IgGen's pretraining provenance is established.")),
}


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One qualified measurement of one construct, with its raw fields preserved."""

    construct: Construct
    source: str
    family: str
    kind: str
    target: str
    assay: str
    parent: str
    scale: str
    value: object = None
    interval: tuple = (None, None)
    censoring: str = CENSOR_NONE
    ordinal_class: object = None
    raw: dict = field(default_factory=dict)
    replicate: object = None
    uncertainty: object = None
    qualification: dict = field(default_factory=dict)

    @property
    def identity(self):
        return self.construct.identity()

    @property
    def study(self):
        """The ORIGINAL study, not the publication alias it arrived under."""
        return canonical_study(self.source)

    @property
    def assay_key(self):
        """``study:family:target[:sub-assay]`` -- the identity a condition names.

        Four components, each earning its place:

        * **study**, not the publication alias, so the curated Adams summary and
          the primary replicate tables are one assay rather than two;
        * **family**, so an affinity readout and an expression readout of the same
          study do not collide into one identity -- and one observation of each
          for the same construct does not look like a duplicate;
        * **target**, so CR9114's H3 and influenza-B partitions get separate
          observation scales and separate bin thresholds instead of a pooled one
          over two antigens;
        * the **sub-assay** where a source declares one (the Li releases' own
          ``Assay`` column, a Petersen barcode sheet), with the alias prefix the
          loader decorated it with stripped off.
        """
        text = str(self.assay)
        suffix = ""
        if text and text not in (self.source, self.study):
            prefix = f"{self.source}:"
            if text.startswith(prefix):
                text = text[len(prefix):]
            suffix = f":{text}"
        return f"{self.study}:{self.family}:{self.target}{suffix}"

    @property
    def population_key(self):
        """Construct + study + assay + target, WITHOUT the replicate.

        The key a quality bin is attached to. A bin is a property of a measured
        construct in an assay, not of one of its replicates, so keying it on the
        replicate would leave two replicates of one construct in different bins
        whenever they straddle a cut point.
        """
        return (self.identity, self.study, self.assay_key, self.target)

    @property
    def observation_identity(self):
        """Study + assay + target + replicate + construct: the SAME observation.

        Keyed on the canonical **study**, never on the publication alias: two FLAb
        releases that both carry an Engelhart row are one observation. Collapsing
        on the construct alone would merge genuine replicates; keying on the row,
        or on the alias, counts a re-publication as a new experiment.
        """
        return (self.identity, self.study, self.assay_key, self.target,
                "" if self.replicate is None else str(self.replicate),
                str(self.raw.get("measurement_id", "")))

    def is_trainable_interval(self):
        return self.kind == KIND_INTERVAL or (self.kind == KIND_POINT and self.value is not None)

    def document(self):
        return {"source": self.source, "family": self.family, "measurement_kind": self.kind,
                "target": self.target, "assay": self.assay, "parent": self.parent,
                "scale": self.scale,
                "value": None if self.value is None else float(self.value),
                "interval": [None if bound is None else float(bound) for bound in self.interval],
                "censoring": self.censoring, "ordinal_class": self.ordinal_class,
                "replicate": self.replicate,
                "uncertainty": None if self.uncertainty is None else float(self.uncertainty),
                "raw": dict(self.raw), "qualification": dict(self.qualification),
                "construct": self.construct.document()}


def pkd_from_log10_nm(value):
    """``pKD = 9 - log10(KD/nM)``. The one place the conversion may be written.

    Derivation, so it is checkable rather than believed: ``pKD = -log10(KD/M)`` and
    ``KD/M = (KD/nM) * 1e-9``, so ``pKD = -log10(KD/nM) + 9``. Higher pKD is
    stronger binding and smaller KD. This is a unit conversion and explicitly
    **not** a cross-assay calibration.
    """
    return 9.0 - float(value)


def pkd_from_molar(value):
    """``pKD = -log10(KD/M)``, refusing a non-positive KD rather than producing a NaN."""
    value = float(value)
    require(value > 0, f"A KD of {value} M is not a measurement; -log10 of it is not a number")
    return -math.log10(value)


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# zip and xlsx readers (standard library only; openpyxl is absent)
# ---------------------------------------------------------------------------

def zipped_csv_rows(path):
    """Rows of the single real CSV inside a FLAb ``.csv.zip``.

    macOS resource forks (``__MACOSX/`` and ``._*``) are filtered, and **exactly
    one** remaining CSV is required. A zip with two candidates is a finding, not a
    case for taking the first: the two could be different releases.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist()
                      if name.lower().endswith(".csv")
                      and not name.startswith("__MACOSX/")
                      and not Path(name).name.startswith("._")]
        require(len(candidates) == 1,
                f"{path.name} holds {len(candidates)} candidate CSV file(s) {candidates} after "
                "filtering macOS resource forks. Exactly one is required; picking the first of "
                "two would silently choose a release.")
        with archive.open(candidates[0]) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
            return candidates[0], list(csv.DictReader(text))


def csv_rows(path, *, skip=0, expect_header=None, forbid_columns=frozenset()):
    """Rows of a plain CSV, with the preamble asserted before it is skipped."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    require(len(lines) > int(skip), f"{path.name} has {len(lines)} lines and {skip} to skip")
    if skip:
        preamble = lines[:int(skip)]
        header_row = lines[int(skip)]
        require(expect_header is not None,
                "Skipping a preamble without an expected header would skip whatever is there")
        for position, line in enumerate(preamble):
            require(line.split(",")[0].strip() != expect_header[0],
                    f"{path.name}: preamble line {position} begins with the expected header field "
                    f"{expect_header[0]!r}, so the declared {skip}-line preamble is wrong and the "
                    "skip would discard real data")
        observed = [field.strip() for field in next(csv.reader([header_row]))]
        require(observed == list(expect_header),
                f"{path.name}: the row after the {skip}-line preamble is {observed}, not the "
                f"expected header {list(expect_header)}")
    rows = list(csv.DictReader(lines[int(skip):]))
    if forbid_columns and rows:
        present = sorted(set(rows[0]) & set(forbid_columns))
        require(not present,
                f"{path.name} carries reserved column(s) {present} and this loader refuses to "
                "materialise them")
    return rows


_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root:
        parts = []
        for child in item:
            if _tag(child) == "t":
                parts.append(child.text or "")
            elif _tag(child) == "r":
                parts.extend(grandchild.text or "" for grandchild in child
                             if _tag(grandchild) == "t")
        values.append("".join(parts))
    return values


def xlsx_sheet_names(path):
    """Sheet names in workbook order. ``her2_data.read_workbook`` reads only the first."""
    with zipfile.ZipFile(Path(path)) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find(f"{_MAIN_NS}sheets")
        require(sheets is not None and len(sheets), "Workbook declares no sheets")
        return tuple(sheet.get("name") for sheet in sheets)


def read_xlsx_sheet(path, sheet_name):
    """One named sheet as a list of dicts, standard library only.

    ``openpyxl`` is not installed and is not needed: an ``.xlsx`` is a ZIP of XML.
    Written here rather than reusing ``her2_data.read_workbook`` because that one
    is first-sheet-only and column-letter addressed, and the Petersen supplements
    carry six named sheets with real headers.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find(f"{_MAIN_NS}sheets")
        relationship = None
        for sheet in sheets:
            if sheet.get("name") == sheet_name:
                relationship = sheet.get(f"{_REL_NS}id")
        require(relationship is not None,
                f"{path.name} has no sheet named {sheet_name!r}; it has "
                f"{list(xlsx_sheet_names(path))}")
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for entry in rels:
            if entry.get("Id") == relationship:
                raw = entry.get("Target")
                target = "xl/" + raw if not raw.startswith("/") else raw.lstrip("/")
        require(target is not None, f"Sheet relationship {relationship!r} has no target")
        strings = _shared_strings(archive)
        grid = {}
        with archive.open(target) as stream:
            for _, element in ElementTree.iterparse(stream, events=("end",)):
                if _tag(element) != "c":
                    continue
                reference = element.get("r") or ""
                letters = "".join(ch for ch in reference if ch.isalpha())
                digits = "".join(ch for ch in reference if ch.isdigit())
                if not letters or not digits:
                    element.clear()
                    continue
                kind = element.get("t")
                if kind == "inlineStr":
                    node = element.find(f"{_MAIN_NS}is")
                    text = "".join(t.text or "" for t in node.iter(f"{_MAIN_NS}t")) \
                        if node is not None else ""
                else:
                    node = element.find(f"{_MAIN_NS}v")
                    text = "" if node is None or node.text is None else node.text
                    if kind == "s" and text != "":
                        text = strings[int(text)]
                grid.setdefault(int(digits), {})[letters] = text
                element.clear()
    require(grid, f"{path.name}:{sheet_name} holds no cells")
    first = min(grid)
    header = {letter: value for letter, value in grid[first].items() if str(value).strip()}
    records = []
    for number in sorted(grid):
        if number == first:
            continue
        cells = grid[number]
        record = {name: cells.get(letter, "") for letter, name in header.items()}
        if any(str(value).strip() for value in record.values()):
            record["_row"] = number
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# AlphaSeq / AlphaBind
# ---------------------------------------------------------------------------

#: The compatible trastuzumab scFv construct: VH(120) + linker(15) + VL(107) + R.
ALPHASEQ_CONSTRUCT_LENGTH = 243
ALPHASEQ_VH_LENGTH = 120
ALPHASEQ_LINKER_LENGTH = 15
ALPHASEQ_VL_START = 135
ALPHASEQ_VL_LENGTH = 107
ALPHASEQ_TERMINAL = "R"


def alphaseq_sequence_column():
    """The only column the pre-freeze split pass may read. Outcomes stay unopened."""
    return "mata_sequence"


def split_alphaseq_construct(sequence):
    """``(VH, linker, VL, terminal)`` for a length-243 scFv, or ``None``.

    The offsets are asserted constants derived from the release README's declared
    construct, exactly as ``her2_data`` treats the HER2 scaffold: a changed release
    fails here rather than shifting the editable window.
    """
    text = str(sequence).strip().upper()
    if len(text) != ALPHASEQ_CONSTRUCT_LENGTH:
        return None
    heavy = text[:ALPHASEQ_VH_LENGTH]
    linker = text[ALPHASEQ_VH_LENGTH:ALPHASEQ_VL_START]
    light = text[ALPHASEQ_VL_START:ALPHASEQ_VL_START + ALPHASEQ_VL_LENGTH]
    terminal = text[ALPHASEQ_VL_START + ALPHASEQ_VL_LENGTH:]
    require(len(linker) == ALPHASEQ_LINKER_LENGTH and len(light) == ALPHASEQ_VL_LENGTH,
            "The declared AlphaSeq construct offsets do not partition a 243-mer")
    if terminal != ALPHASEQ_TERMINAL:
        return None
    return heavy, linker, light, terminal


def derive_alphaseq_parent(sequences, *, minimum_share=0.9):
    """Derive the parental linker and VL from the release itself, by majority.

    The plan forbids inventing a light chain, and the release does not hand one
    over in a single labelled field. So the parent is *derived*: among length-243
    terminal-R rows, the modal 15-mer linker and modal 107-mer VL. Both must hold a
    declared majority share, which is what distinguishes "this release has one
    parental scaffold" from "I took whichever was first".
    """
    linkers, lights = {}, {}
    total = 0
    for sequence in sequences:
        parts = split_alphaseq_construct(sequence)
        if parts is None:
            continue
        total += 1
        linkers[parts[1]] = linkers.get(parts[1], 0) + 1
        lights[parts[2]] = lights.get(parts[2], 0) + 1
    require(total > 0, "No length-243 terminal-R AlphaSeq row to derive a parent from")
    linker, linker_count = max(sorted(linkers.items()), key=lambda item: item[1])
    light, light_count = max(sorted(lights.items()), key=lambda item: item[1])
    shares = {"linker": linker_count / total, "light": light_count / total}
    require(shares["linker"] >= float(minimum_share) and shares["light"] >= float(minimum_share),
            f"The derived parental linker/VL hold shares {shares} of {total} compatible rows, "
            f"below the declared {minimum_share}. That is not one parental scaffold, and choosing "
            "a modal value from a mixture would silently define the population.")
    return {"linker": linker, "light": light, "compatible_rows": total,
            "shares": shares, "distinct_linkers": len(linkers), "distinct_lights": len(lights),
            "derivation": ("modal 15-mer linker and modal 107-mer VL among length-243 terminal-R "
                           "rows, each required to hold a declared majority share. The parent is "
                           "derived from the release, never invented.")}


def alphaseq_compatibility(sequence, *, parent, scaffold, sites):
    """Is this row a member of the fixed-scaffold population? With the reason.

    Compatibility requires the complete VL, the complete linker, and every heavy
    position **outside** the edit map to match the parent. "The ten-residue window
    has the right length" is not compatibility: a row with a different framework is
    a different antibody, and enrolling it core-only would put a different molecule
    in the population the split was frozen over.
    """
    parts = split_alphaseq_construct(sequence)
    if parts is None:
        return False, "not a length-243 terminal-R construct"
    heavy, linker, light, _ = parts
    if linker != parent["linker"]:
        return False, "linker differs from the derived parental linker"
    if light != parent["light"]:
        return False, "VL differs from the derived parental VL"
    editable = {index for chain, index in sites if chain == HEAVY}
    for position in range(len(heavy)):
        if position in editable:
            continue
        if heavy[position] != scaffold.heavy[position]:
            return False, f"heavy framework position {position} differs from the parent"
    return True, "compatible"


#: Columns of the AlphaBind release that carry an outcome. Never materialised by
#: the pre-freeze pass, and named here so the refusal is structural.
ALPHASEQ_OUTCOME_COLUMNS = frozenset({"alphaseq_affinity", "predicted_affinity",
                                      "parental_mean_aff", "parental_aff_margin",
                                      "avg_off_target_affinity"})


def load_alphaseq_sequences(path, *, column=None, descriptor="mata_description"):
    """Sequence-only read. No outcome column is parsed into memory at all.

    Used by the split freeze, which must run before any outcome is opened. The
    earlier implementation read the whole file with ``read_text`` and a
    ``DictReader``, which materialises **every** column including
    ``alphaseq_affinity`` and then discards it; a comment saying the outcomes are
    unread does not make them unread. ``usecols`` pushes the restriction into the
    parser, so the outcome bytes are never turned into Python values.
    """
    import pandas as pd
    column = column or alphaseq_sequence_column()
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)
    require(column in available,
            f"{path.name} has no {column!r} column; it has {available}")
    wanted = [name for name in (column, descriptor) if name in available]
    forbidden = sorted(set(wanted) & ALPHASEQ_OUTCOME_COLUMNS)
    require(not forbidden,
            f"The sequence-only read asked for outcome column(s) {forbidden}")
    frame = pd.read_csv(path, usecols=wanted, dtype="string")
    sequences = [str(value).strip().upper() for value in frame[column].fillna("")]
    descriptions = ([str(value).strip() for value in frame[descriptor].fillna("")]
                    if descriptor in wanted else [""] * len(sequences))
    present = sorted(set(available) & ALPHASEQ_OUTCOME_COLUMNS)
    return {"sequences": tuple(sequences), "descriptions": tuple(descriptions),
            "rows": len(sequences), "sha256": sha256(path),
            "columns_parsed": wanted,
            "outcome_columns_present_but_unread": present,
            "rule": ("usecols restricts the PARSER, so the outcome columns are never converted "
                     "into values. Their presence in the file is recorded; their contents are "
                     "not read on this path.")}


class ReservedUnlock:
    """A single-use token that permits reading the reserved panel's outcomes.

    Minted once, by the confirmation stage, after every selection is frozen. It
    records who minted it and refuses a second use: the reserved test is opened
    once, and "once" has to be a property of the code rather than of the
    discipline of whoever runs it.
    """

    def __init__(self, *, reason, frozen_selection_sha256, minted_by):
        require(reason and frozen_selection_sha256 and minted_by,
                "A reserved unlock names its reason, the frozen selection it is bound to and "
                "who minted it. An anonymous unlock is not an audit trail.")
        self.reason = str(reason)
        self.frozen_selection_sha256 = str(frozen_selection_sha256)
        self.minted_by = str(minted_by)
        self._spent = False

    def spend(self, *, consumer):
        require(not self._spent,
                f"{consumer}: this reserved unlock has already been spent. The reserved panel is "
                "opened ONCE; a second opening is a second look at the same held-out data.")
        self._spent = True
        return {"consumer": str(consumer), "reason": self.reason,
                "frozen_selection_sha256": self.frozen_selection_sha256,
                "minted_by": self.minted_by}


def load_alphaseq_observations(path, *, source, scaffold, sites, parent=None, target="HER2",
                               assay=None, parent_id=None, allowed_windows=None,
                               window_partitions=None, reserved_unlock=None):
    """Full read, after the split is frozen. ``alphaseq_affinity`` is the outcome.

    ``allowed_windows`` is the frozen train+development window set. A row whose
    edit window is not in it is skipped **before** its outcome is converted, so
    the reserved measurements are not materialised as a by-product of loading. An
    unrestricted read requires a :class:`ReservedUnlock`, which is single-use.
    """
    spec = SOURCES[source]
    rows = csv_rows(path)
    require(rows, f"{Path(path).name} carried no data rows")
    if allowed_windows is None:
        require(reserved_unlock is not None,
                "Loading AlphaSeq outcomes without a frozen train/development window set reads "
                "the reserved panel too. Pass allowed_windows, or a ReservedUnlock if this IS "
                "the single declared opening.")
        unlock_record = reserved_unlock.spend(consumer=f"load_alphaseq_observations::{source}")
    else:
        allowed_windows = frozenset(str(value) for value in allowed_windows)
        unlock_record = None
    if parent is None:
        parent = derive_alphaseq_parent(row.get(alphaseq_sequence_column(), "") for row in rows)
    window_slice = (min(index for chain, index in sites if chain == HEAVY),
                    max(index for chain, index in sites if chain == HEAVY) + 1)
    partitions = {str(key): str(value) for key, value in dict(window_partitions or {}).items()}
    observations, excluded, by_partition = [], {}, {}
    for position, row in enumerate(rows):
        sequence = str(row.get(alphaseq_sequence_column(), "")).strip().upper()
        ok, reason = alphaseq_compatibility(sequence, parent=parent, scaffold=scaffold,
                                           sites=sites)
        if not ok:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        if allowed_windows is not None:
            window = sequence[window_slice[0]:window_slice[1]]
            if window not in allowed_windows:
                excluded["not in the frozen train/development window set"] = excluded.get(
                    "not in the frozen train/development window set", 0) + 1
                continue
        raw = finite(row.get("alphaseq_affinity"))
        heavy, _, light, _ = split_alphaseq_construct(sequence)
        construct = Construct(heavy=heavy, light=light, sites=edit_map(sites), source=source,
                              record_id=str(row.get("id") or position))
        window = sequence[window_slice[0]:window_slice[1]]
        partition = partitions.get(window, "train")
        qualification = {"compatibility": reason,
                         "predicted_affinity_is_not_measured": True,
                         "raw_column": "alphaseq_affinity",
                         "raw_units": "log10(KD/nM)"}
        if raw is None:
            excluded["non-finite alphaseq_affinity"] = \
                excluded.get("non-finite alphaseq_affinity", 0) + 1
            qualification["outcome"] = "missing"
            continue
        by_partition[partition] = by_partition.get(partition, 0) + 1
        observations.append(Observation(
            construct=construct, source=source, family=spec.family, kind=spec.kind,
            target=target, assay=assay or source, parent=parent_id or f"{source}::parental",
            scale=SCALE_PKD, value=pkd_from_log10_nm(raw),
            interval=(pkd_from_log10_nm(raw), pkd_from_log10_nm(raw)),
            censoring=CENSOR_NONE,
            raw={"alphaseq_affinity": raw,
                 "predicted_affinity": finite(row.get("predicted_affinity")),
                 "edit_distance": finite(row.get("edit_distance")),
                 "parental_mean_aff": finite(row.get("parental_mean_aff")),
                 "mata_description": row.get("mata_description"),
                 "matalpha_description": row.get("matalpha_description"),
                 "measurement_id": str(row.get("id") or position),
                 "window": window, "partition": partition,
                 "mutated_sites": [[chain, int(index)] for chain, index in sites
                                   if construct.residue_at((chain, index))
                                   != (scaffold.heavy[index] if chain == HEAVY
                                       else scaffold.light[index])],
                 "model": row.get("model")},
            qualification=qualification))
    return {"observations": tuple(observations), "parent": dict(parent),
            "excluded": dict(sorted(excluded.items())), "rows": len(rows),
            "by_partition": dict(sorted(by_partition.items())),
            "sha256": sha256(path),
            "window_gate": ("frozen train/development windows only"
                            if allowed_windows is not None else "UNRESTRICTED"),
            "reserved_unlock": unlock_record}


# ---------------------------------------------------------------------------
# FLAb: Engelhart, Li, Adams, Koenig, Petersen
# ---------------------------------------------------------------------------

def _paired_construct(heavy, light, *, source, record_id, sites=None):
    heavy = str(heavy).strip().upper()
    light = str(light).strip().upper()
    if sites is None:
        sites = ()
    return Construct(heavy=heavy, light=light, sites=tuple(sites), source=source,
                     record_id=str(record_id))


def derive_sites_against(construct_heavy, construct_light, parent_heavy, parent_light):
    """The edit map of one variant against its parent: the positions that differ.

    Refuses a length mismatch rather than aligning. An indel-carrying variant is a
    different construct and needs an alignment decision this campaign has not made;
    reporting it as unsupported is the honest outcome.
    """
    if len(construct_heavy) != len(parent_heavy) or len(construct_light) != len(parent_light):
        return None
    sites = [(HEAVY, i) for i, (a, b) in enumerate(zip(construct_heavy, parent_heavy)) if a != b]
    sites += [(LIGHT, j) for j, (a, b) in enumerate(zip(construct_light, parent_light)) if a != b]
    return tuple(sites)


def _majority_parent(pairs):
    """The modal (heavy, light) pair: the parental background of a mutational scan."""
    counts = {}
    for heavy, light in pairs:
        counts[(heavy, light)] = counts.get((heavy, light), 0) + 1
    require(counts, "No sequence pair to derive a parent from")
    (heavy, light), count = max(sorted(counts.items()), key=lambda item: item[1])
    return {"heavy": heavy, "light": light, "count": count, "distinct_pairs": len(counts)}


def single_mutant_consensus(pairs):
    """Infer the unmeasured reference of a single-substitution scan, with checks.

    A modal whole sequence is not the reference when every single mutant occurs
    once. Require an unambiguous column consensus and at most one substitution
    per observed pair; never create a measurement for the inferred reference.
    """
    from collections import Counter
    pairs = tuple(pairs)
    require(pairs, "No pairs from which to infer a single-mutant reference")
    require(len({(len(h), len(l)) for h, l in pairs}) == 1,
            "A single-mutant scan cannot contain length-changing variants")
    reference = []
    for chain in range(2):
        residues = []
        for column in zip(*(pair[chain] for pair in pairs)):
            counts = Counter(column).most_common()
            require(len(counts) == 1 or counts[0][1] > counts[1][1],
                    "The single-mutant reference is ambiguous at a residue")
            residues.append(counts[0][0])
        reference.append("".join(residues))
    heavy, light = reference
    distances = [sum(a != b for a, b in zip(h + l, heavy + light)) for h, l in pairs]
    require(max(distances) <= 1, "The inferred reference does not describe a single-mutant scan")
    return {"heavy": heavy, "light": light, "count": distances.count(0),
            "distinct_pairs": len(set(pairs)),
            "derivation": "unique column consensus, validated against every single mutant",
            "reference_measured": 0 in distances}


def common_region(observations, parent, *, maximum=None):
    """The positions that vary across a whole group: the group's SHARED support.

    This is the region every member of the group is scored over, and it is
    deliberately not each member's own mutated positions. A mutation-dependent
    region gives a single mutant a one-site window and a triple mutant a
    three-site window, so their window log probabilities are not comparable, a
    whole-window preference between them compares two different quantities, and
    the width of the window leaks how many edits the row carries.

    Computed with numpy over the packed group rather than by a Python scan, and
    capped at ``maximum`` positions by taking the most variable ones when a group
    is wider than the budget -- with the dropped count returned, because a
    silently narrowed region is a silently different task.
    """
    rows = list(observations)
    if not rows:
        return (), {"rows": 0, "varying": 0, "dropped": 0}
    heavy_length = len(parent["heavy"]) if parent and parent.get("heavy") else \
        len(rows[0].construct.heavy)
    light_length = len(parent.get("light") or "") if parent and parent.get("heavy") else \
        len(rows[0].construct.light)
    aligned = [record for record in rows
               if len(record.construct.heavy) == heavy_length
               and len(record.construct.light) == light_length]
    if not aligned:
        return (), {"rows": len(rows), "varying": 0, "dropped": 0}
    joined = "".join(record.construct.heavy + record.construct.light for record in aligned)
    width = heavy_length + light_length
    codes = np.frombuffer(joined.encode("ascii"), dtype=np.uint8).reshape(len(aligned), width)
    distinct = np.asarray([len(np.unique(codes[:, column])) for column in range(width)])
    columns = np.flatnonzero(distinct > 1)
    if parent is not None and parent.get("heavy"):
        reference = f"{parent['heavy']}{parent.get('light') or ''}"
        if len(reference) == width:
            row = np.frombuffer(reference.encode("ascii"), dtype=np.uint8)
            columns = np.union1d(columns, np.flatnonzero((codes != row).any(axis=0)))
    dropped = 0
    if maximum is not None and columns.size > int(maximum):
        order = np.argsort(-distinct[columns], kind="stable")[:int(maximum)]
        dropped = int(columns.size) - int(maximum)
        columns = np.sort(columns[order])
    sites = tuple((HEAVY, int(column)) if int(column) < heavy_length
                  else (LIGHT, int(column) - heavy_length) for column in columns)
    return sites, {"rows": len(rows), "aligned_rows": len(aligned),
                   "varying": int(columns.size) + dropped, "dropped": dropped,
                   "maximum": None if maximum is None else int(maximum)}


def _assign_sites(observations, parent, *, maximum=None):
    """Attach the group's SHARED region to every observation, plus its own edits.

    ``unalignable`` counts the rows whose chain lengths differ from the group's --
    an indel-carrying variant needs an alignment decision this campaign has not
    made. They are retained with an **empty** region and their own mutated sites
    unknown, which makes them trainable by the outcome heads (which read the whole
    molecule) and invisible to the window tasks (which need a shared support).
    Erasing them instead would delete the part of a non-HER2 release that carries
    insertions and deletions.
    """
    rows = list(observations)
    region, geometry = common_region(rows, parent, maximum=maximum)
    out, unalignable, head_only = [], 0, 0
    for record in rows:
        mutated = derive_sites_against(record.construct.heavy, record.construct.light,
                                       parent["heavy"], parent["light"]) \
            if parent is not None and parent.get("heavy") else None
        if mutated is None:
            unalignable += 1
        usable = region if mutated is not None else ()
        if not usable:
            head_only += 1
        construct = Construct(heavy=record.construct.heavy, light=record.construct.light,
                              sites=tuple(usable), source=record.construct.source,
                              record_id=record.construct.record_id)
        raw = dict(record.raw)
        raw["mutated_sites"] = [[chain, int(index)] for chain, index in (mutated or ())]
        raw["mutation_map_known"] = mutated is not None
        raw["region_sites"] = len(usable)
        out.append(Observation(**{**record.__dict__, "construct": construct, "raw": raw}))
    return tuple(out), unalignable, {**geometry, "head_only_rows": head_only,
                                     "unalignable_rows": unalignable}


#: The widest common region any source may declare. Beyond this the region is
#: trained as declared tiles; the number is the autoregressive route's query budget
#: minus the longest paired context it must fit beside, measured in
#: ``her2_routes_serial.serialised_length`` rather than chosen.
REGION_MAXIMUM = 96


def load_engelhart(path, *, source="engelhart2022_s1", region_maximum=REGION_MAXIMUM):
    """FLAb Engelhart AlphaSeq: ``Pred_affinity`` is measured log10(KD/nM)."""
    spec = SOURCES[source]
    name, rows = zipped_csv_rows(path)
    require(rows, f"{name} carried no rows")
    for column in ("Pred_affinity", "heavy", "light", "fitness"):
        require(column in rows[0], f"{name} has no {column!r} column; it has {sorted(rows[0])}")
    parent = _majority_parent([(str(r["heavy"]).strip().upper(),
                                str(r["light"]).strip().upper()) for r in rows])
    observations, missing = [], 0
    for position, row in enumerate(rows):
        raw = finite(row.get("Pred_affinity"))
        if raw is None:
            missing += 1
            continue
        observations.append(Observation(
            construct=_paired_construct(row["heavy"], row["light"], source=source,
                                        record_id=position),
            source=source, family=spec.family, kind=spec.kind, target=spec.target,
            assay=source, parent=f"{source}::parental", scale=SCALE_PKD,
            value=pkd_from_log10_nm(raw),
            interval=(pkd_from_log10_nm(raw), pkd_from_log10_nm(raw)),
            raw={"Pred_affinity": raw, "fitness": finite(row.get("fitness"))},
            qualification={"column_name_is_misleading": (
                "Pred_affinity is the measured assay-calibrated log10(KD/nM) readout, per "
                "PMC9606274. It is not a trained model's prediction."),
                "raw_units": "log10(KD/nM)"}))
    observations, unalignable, region = _assign_sites(observations, parent,
                                                      maximum=region_maximum)
    return {"observations": observations, "parent": parent, "rows": len(rows),
            "missing_affinity": missing, "unalignable": unalignable, "region": region,
            "member": name, "sha256": sha256(path)}


LI_HEADER = ("POI", "Sequence", "Target", "Assay", "Replicate", "Pred_affinity", "HC", "LC",
             "CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3")


#: The columns the streaming reader materialises. ``Sequence`` is the concatenation
#: of ``HC`` and ``LC`` and is the single largest column; leaving it to the parser
#: is most of what makes a 735 MB / 1,123 MB release readable here.
#:
#: The six CDR columns ARE read. An earlier revision dropped them as "derivable
#: from HC/LC", which is false: a CDR boundary is an annotation, and nothing in a
#: bare heavy chain says where CDRH2 ends. Without them the only available region
#: is each variant's own mutated positions against a guessed background, which is
#: the mutation-dependent map this campaign refuses. They are parsed once per POI,
#: not once per measurement row, and only to locate the spans.
LI_CDR_COLUMNS = ("CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3")
LI_STREAM_COLUMNS = ("POI", "Target", "Assay", "Replicate", "Pred_affinity", "HC", "LC",
                     *LI_CDR_COLUMNS)

#: The release's own parent-control names, per POI family prefix. Li1 carries the
#: three ``MIT_*`` control rows; the second release carries none, so its two
#: families are resolved through :data:`LI_CROSS_RELEASE_PARENTS` against Li1's
#: verified controls rather than left on a placeholder.
LI2_POI_PARENTS = {"14H": "MIT_14_HL_scFV", "14L": "MIT_14_HL_scFV"}
LI_CROSS_RELEASE_PARENTS = {"li2023_s1_b": LI2_POI_PARENTS}
#: POI counts per family in the second release, for an enrolment audit.
LI2_POI_COUNTS = {"14L": 44010, "14H": 35321}
LI2_TOTAL_ROWS = 1903921


def _li_member_text_head(path, *, lines):
    """The first ``lines`` physical lines of the release, without reading the rest."""
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(Path(path)) as archive:
            candidates = [name for name in archive.namelist()
                          if name.lower().endswith(".csv")
                          and not name.startswith("__MACOSX/")
                          and not Path(name).name.startswith("._")]
            require(len(candidates) == 1,
                    f"{Path(path).name} holds {len(candidates)} candidate CSV file(s) "
                    f"{candidates} after filtering macOS resource forks")
            member = candidates[0]
            with archive.open(member) as stream:
                text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
                head = [text.readline().rstrip("\r\n") for _ in range(int(lines))]
        return member, head
    member = Path(path).name
    head = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for _ in range(int(lines)):
            head.append(handle.readline().rstrip("\r\n"))
    return member, head


def _assert_li_preamble(path, *, preamble):
    """Assert the declared preamble against the real bytes, before any skip.

    Reads exactly ``preamble + 1`` lines. The earlier implementation parsed the
    whole release once with the wrong preamble and then re-read it, which is two
    full passes over a gigabyte to establish a fact about its first seven lines.
    """
    member, head = _li_member_text_head(path, lines=int(preamble) + 1)
    require(len(head) == int(preamble) + 1 and head[int(preamble)],
            f"{member} has too few lines for a {preamble}-line skip")
    for position, line in enumerate(head[:int(preamble)]):
        require(next(csv.reader([line]))[0].strip() != LI_HEADER[0],
                f"{member}: preamble line {position} begins with {LI_HEADER[0]!r}, so the "
                f"declared {preamble}-line preamble is wrong and the skip would discard real "
                "data")
    observed = [field.strip() for field in next(csv.reader([head[int(preamble)]]))]
    require(observed == list(LI_HEADER),
            f"{member}: the row after the {preamble}-line preamble is {observed}, not "
            f"{list(LI_HEADER)}")
    return member, observed


def locate_cdrs(heavy, light, annotations):
    """``(sites, signature)`` from the release's six CDR annotations, or ``None``.

    The annotations are the only thing that says where a CDR is. Each is located
    in its own chain with ``str.find`` and is required to occur **exactly once**:
    a CDR string that matches twice does not identify a span, and one that matches
    nowhere means the annotation and the sequence disagree. Both are counted and
    refused rather than resolved by taking the first hit.

    ``signature`` is the chain pair with the six spans replaced by ``*``: the
    scaffold family identity. The first Li release has exactly three of them over
    its four POI prefixes, which is what makes a per-family common region a real
    shared support rather than a per-row window.
    """
    spans, buffers = [], {HEAVY: list(heavy), LIGHT: list(light)}
    for name in LI_CDR_COLUMNS:
        text = str(annotations.get(name) or "").strip().upper()
        if not text:
            return None, "a CDR annotation is empty"
        chain = HEAVY if name.startswith("CDRH") else LIGHT
        sequence = heavy if chain == HEAVY else light
        first = sequence.find(text)
        if first < 0:
            return None, f"{name} does not occur in the {chain} chain"
        if sequence.find(text, first + 1) >= 0:
            return None, f"{name} occurs more than once in the {chain} chain"
        spans.append((chain, first, len(text)))
        for offset in range(len(text)):
            buffers[chain][first + offset] = "*"
    sites = []
    for chain, start, length in spans:
        sites.extend((chain, start + offset) for offset in range(length))
    signature = "".join(buffers[HEAVY]) + "|" + "".join(buffers[LIGHT])
    return edit_map(sites), signature


@dataclass
class LiTable:
    """A columnar AlphaSeq release: one sequence per POI, one row per measurement.

    The first release publishes 1,259,700 measurement rows over 104,972 POIs
    against four targets; the second publishes 1,903,921 rows over 79,331 POIs.
    Holding either as a million ``Observation`` objects is the shape that does not
    fit; holding the sequences once per POI and the measurements as arrays does.
    :func:`li_observations` materialises whichever slice is enrolled.

    ``families`` are the scaffold signatures derived from the six CDR annotations
    and ``family_sites`` is each family's **common region** -- the six annotated
    spans, identical for every member of a family by construction of the
    signature. That is the shared support the whole-window task needs.
    """

    source: str
    member: str
    sha256: str
    pois: tuple
    sequences: dict
    poi_index: object
    targets: tuple
    target_index: object
    assays: tuple
    assay_index: object
    replicate: object
    value: object
    rows: int
    missing: int
    preamble_lines: int = 0
    families: tuple = ()
    family_index: object = None
    family_sites: tuple = ()
    annotation_failures: dict = field(default_factory=dict)
    inconsistent_poi: int = 0

    def counts_by_target(self):
        import numpy as np
        counts = np.bincount(self.target_index, minlength=len(self.targets))
        return {name: int(value) for name, value in zip(self.targets, counts)}

    def counts_by_family(self):
        import numpy as np
        if self.family_index is None or not self.families:
            return {}
        counts = np.bincount(np.asarray(self.family_index, dtype=np.int64),
                             minlength=len(self.families))
        return {str(index): int(value) for index, value in enumerate(counts)}

    def counts_by_prefix(self):
        counts = {}
        for poi in self.pois:
            prefix = str(poi).split("_")[0]
            counts[prefix] = counts.get(prefix, 0) + 1
        return dict(sorted(counts.items()))

    def family_of(self, poi):
        """``(family index, sites)`` for one POI, or ``(None, ())``."""
        if self.family_index is None:
            return None, ()
        try:
            position = self.pois.index(poi)
        except ValueError:
            return None, ()
        index = int(self.family_index[position])
        if index < 0:
            return None, ()
        return index, self.family_sites[index]

    def parent_controls(self):
        """The ``MIT_*`` parent-control POIs the release carries explicitly."""
        return {poi: self.sequences[poi] for poi in self.pois if str(poi).startswith("MIT_")}

    def document(self):
        return {"source": self.source, "member": self.member, "sha256": self.sha256,
                "rows": int(self.rows), "measurements": int(len(self.value)),
                "missing_affinity": int(self.missing),
                "distinct_poi": len(self.pois),
                "targets": list(self.targets), "assays": list(self.assays),
                "rows_by_target": self.counts_by_target(),
                "poi_by_prefix": self.counts_by_prefix(),
                "scaffold_families": len(self.families),
                "poi_by_family": self.counts_by_family(),
                "family_region_sites": [len(sites) for sites in self.family_sites],
                "annotation_failures": dict(self.annotation_failures),
                "poi_with_inconsistent_sequences": int(self.inconsistent_poi),
                "parent_controls": sorted(self.parent_controls()),
                "preamble_lines": int(self.preamble_lines),
                "memory_rule": ("sequences and CDR annotations are parsed once per POI and the "
                                "measurements are stored as arrays. The release is read in chunks "
                                "and the full-Sequence column -- the concatenation of HC and LC, "
                                "and the single largest column -- is never parsed."),
                "annotation_rule": ("the six CDR columns are READ. A CDR boundary is an "
                                    "annotation and is not derivable from a bare chain; without "
                                    "them the only available region is each variant's own mutated "
                                    "positions against a guessed background.")}


def stream_li_table(path, *, source, preamble=0, chunk_rows=200000, cache=None):
    """Read an AlphaSeq release in chunks into a :class:`LiTable`.

    ``cache`` is a directory on the artifact volume. A compact re-read of the
    parsed columns is written there and reused when the source digest matches, so
    the gigabyte parse happens once per release rather than once per stage.
    """
    import numpy as np
    import pandas as pd
    digest = sha256(path)
    member, _ = _assert_li_preamble(path, preamble=preamble) if preamble else (
        _li_member_text_head(path, lines=1)[0], None)
    if cache is not None:
        cached = _read_li_cache(cache, source=source, digest=digest, preamble=int(preamble))
        if cached is not None:
            return cached
    poi_ids, poi_rows, sequences = {}, [], {}
    target_ids, assay_ids = {}, {}
    poi_index, target_index, assay_index, replicate, values = [], [], [], [], []
    family_ids, family_rows, family_sites, poi_family = {}, [], [], []
    annotation_failures, inconsistent = {}, 0
    total, missing = 0, 0

    def handle_chunk(chunk):
        nonlocal total, missing, inconsistent
        total += int(len(chunk))
        columns = [chunk[name] for name in LI_CDR_COLUMNS]
        for position, (poi, target, assay, rep, affinity, heavy, light) in enumerate(zip(
                chunk["POI"], chunk["Target"], chunk["Assay"], chunk["Replicate"],
                chunk["Pred_affinity"], chunk["HC"], chunk["LC"])):
            key = str(poi)
            chains = (str(heavy).strip().upper(), str(light).strip().upper())
            if key not in poi_ids:
                poi_ids[key] = len(poi_rows)
                poi_rows.append(key)
                sequences[key] = chains
                annotations = {name: columns[index].iloc[position]
                               for index, name in enumerate(LI_CDR_COLUMNS)}
                sites, signature = locate_cdrs(chains[0], chains[1], annotations)
                if sites is None:
                    annotation_failures[signature] = annotation_failures.get(signature, 0) + 1
                    poi_family.append(-1)
                else:
                    if signature not in family_ids:
                        family_ids[signature] = len(family_rows)
                        family_rows.append(signature)
                        family_sites.append(sites)
                    elif family_sites[family_ids[signature]] != sites:
                        # Two POIs share a masked framework but disagree about
                        # where their CDRs are. That is not one common region and
                        # it is counted rather than averaged over.
                        annotation_failures["span disagreement inside one signature"] = \
                            annotation_failures.get(
                                "span disagreement inside one signature", 0) + 1
                    poi_family.append(int(family_ids[signature]))
            elif sequences[key] != chains:
                # The same POI appearing with two different sequences would make
                # every derived map depend on which row was read first.
                inconsistent += 1
            number = finite(affinity)
            if number is None:
                missing += 1
                continue
            target_name = str(target).strip()
            assay_name = str(assay).strip() or "default"
            target_ids.setdefault(target_name, len(target_ids))
            assay_ids.setdefault(assay_name, len(assay_ids))
            poi_index.append(poi_ids[key])
            target_index.append(target_ids[target_name])
            assay_index.append(assay_ids[assay_name])
            # Replicate identifiers arrive as ``1`` and as ``1.0`` in the two
            # releases. ``str.isdigit`` is false for the second, and an earlier
            # revision therefore recorded every Li replicate as -1.
            replicate.append(_replicate_id(rep))
            values.append(float(number))

    reader_kwargs = {"usecols": list(LI_STREAM_COLUMNS), "chunksize": int(chunk_rows)}
    if preamble:
        reader_kwargs["skiprows"] = int(preamble)
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(Path(path)) as archive, archive.open(member) as stream:
            for chunk in pd.read_csv(io.TextIOWrapper(stream, encoding="utf-8-sig"),
                                     **reader_kwargs):
                handle_chunk(chunk)
    else:
        for chunk in pd.read_csv(Path(path), encoding="utf-8-sig", **reader_kwargs):
            handle_chunk(chunk)
    require(total, f"{member} carried no rows")
    table = LiTable(
        families=tuple(family_rows), family_sites=tuple(family_sites),
        family_index=np.asarray(poi_family, dtype=np.int16),
        annotation_failures=dict(sorted(annotation_failures.items())),
        inconsistent_poi=int(inconsistent),
        source=source, member=member, sha256=digest, pois=tuple(poi_rows), sequences=sequences,
        poi_index=np.asarray(poi_index, dtype=np.int32),
        targets=tuple(sorted(target_ids, key=target_ids.get)),
        target_index=np.asarray(target_index, dtype=np.int16),
        assays=tuple(sorted(assay_ids, key=assay_ids.get)),
        assay_index=np.asarray(assay_index, dtype=np.int16),
        replicate=np.asarray(replicate, dtype=np.int16),
        value=np.asarray(values, dtype=np.float32),
        rows=total, missing=missing, preamble_lines=int(preamble))
    if cache is not None:
        _write_li_cache(cache, table)
    return table


def _replicate_id(value):
    """``1``, ``1.0`` and ``"1"`` are all replicate 1. ``-1`` means "not recorded".

    The two AlphaSeq releases disagree on the type of this column, and
    ``str(value).strip().isdigit()`` is false for ``1.0`` -- so an earlier revision
    recorded every replicate in one release as unknown while the other looked fine.
    """
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return -1
    try:
        number = float(text)
    except ValueError:
        return -1
    if not math.isfinite(number) or number != int(number):
        return -1
    return int(number)


#: The cache layout revision. A parser change that alters the cached columns must
#: bump this, or a stale cache is silently reused under the new semantics.
LI_CACHE_REVISION = 3


def _li_cache_paths(cache, *, source):
    root = Path(cache) / f"li_table_{source}"
    return root, root / "index.json", root / "sequences.tsv", root / "measurements.npz"


def _write_li_cache(cache, table):
    """A compact re-readable copy of the parsed columns, on the artifact volume."""
    import json
    import numpy as np
    root, index_path, sequences_path, arrays_path = _li_cache_paths(cache, source=table.source)
    root.mkdir(parents=True, exist_ok=True)
    with sequences_path.open("w", encoding="utf-8", newline="\n") as handle:
        for poi in table.pois:
            heavy, light = table.sequences[poi]
            handle.write(f"{poi}\t{heavy}\t{light}\n")
    np.savez_compressed(arrays_path, poi_index=table.poi_index,
                        target_index=table.target_index, assay_index=table.assay_index,
                        replicate=table.replicate, value=table.value,
                        family_index=np.asarray(table.family_index, dtype=np.int16))
    index_path.write_text(json.dumps(
        {"source": table.source, "member": table.member, "sha256": table.sha256,
         "cache_revision": LI_CACHE_REVISION,
         "parser_columns": list(LI_STREAM_COLUMNS),
         "targets": list(table.targets), "assays": list(table.assays), "rows": int(table.rows),
         "missing": int(table.missing), "preamble_lines": int(table.preamble_lines),
         "families": list(table.families),
         "family_sites": [[[chain, int(index)] for chain, index in sites]
                          for sites in table.family_sites],
         "annotation_failures": dict(table.annotation_failures),
         "inconsistent_poi": int(table.inconsistent_poi)},
        indent=2, sort_keys=True), encoding="utf-8")
    return root


def _read_li_cache(cache, *, source, digest, preamble=None):
    """Reuse the compact cache only when the release, the parser and the skip agree.

    Three keys, not one. A cache keyed on the source digest alone is reused after
    a parser change that added a column or changed a replicate convention, and the
    reused arrays then carry the old semantics under the new code.
    """
    import json
    import numpy as np
    _, index_path, sequences_path, arrays_path = _li_cache_paths(cache, source=source)
    if not (index_path.is_file() and sequences_path.is_file() and arrays_path.is_file()):
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("sha256") != digest:
        return None          # a different release: re-parse rather than reuse
    if int(index.get("cache_revision") or 0) != LI_CACHE_REVISION:
        return None
    if list(index.get("parser_columns") or ()) != list(LI_STREAM_COLUMNS):
        return None
    if preamble is not None and int(index.get("preamble_lines", -1)) != int(preamble):
        return None
    pois, sequences = [], {}
    with sequences_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            poi, heavy, light = line.rstrip("\n").split("\t")
            pois.append(poi)
            sequences[poi] = (heavy, light)
    arrays = np.load(arrays_path)
    return LiTable(source=source, member=index["member"], sha256=digest, pois=tuple(pois),
                   sequences=sequences, poi_index=arrays["poi_index"],
                   targets=tuple(index["targets"]), target_index=arrays["target_index"],
                   assays=tuple(index["assays"]), assay_index=arrays["assay_index"],
                   replicate=arrays["replicate"], value=arrays["value"],
                   rows=int(index["rows"]), missing=int(index["missing"]),
                   preamble_lines=int(index["preamble_lines"]),
                   families=tuple(index.get("families") or ()),
                   family_index=arrays["family_index"],
                   family_sites=tuple(
                       tuple((str(chain), int(position)) for chain, position in sites)
                       for sites in (index.get("family_sites") or ())),
                   annotation_failures=dict(index.get("annotation_failures") or {}),
                   inconsistent_poi=int(index.get("inconsistent_poi") or 0))


def li_parent_of(poi, *, source=None):
    """The parent control a POI belongs to, from the release's own family prefix.

    Explicit, not modal. ``AAYL49`` and ``AAYL50`` share ``MIT_14``; ``AAYL51`` is
    ``MIT_91`` and ``AAYL52`` is ``MIT_95``. A modal parent taken across the
    pooled release assigns one background to all four families, which turns every
    variant of the other three into a many-site mutant of the wrong molecule and
    makes its derived edit map meaningless.

    The second release's ``14H`` and ``14L`` prefixes name the same MIT_14 parent
    and the release carries no control row of its own, so the mapping is declared
    per source in :data:`LI_CROSS_RELEASE_PARENTS` and the *sequence* is supplied
    from the first release's verified control.
    """
    text = str(poi)
    if text.startswith("MIT_"):
        return text
    table = dict(LI_CROSS_RELEASE_PARENTS.get(str(source) or "", {})) or LI_POI_PARENTS
    for family, parent in sorted(table.items(), key=lambda item: -len(item[0])):
        if text.startswith(family):
            return parent
    for family, parent in sorted(LI_POI_PARENTS.items(), key=lambda item: -len(item[0])):
        if text.startswith(family):
            return parent
    return None


def li_observations(table, *, source, targets=None, aggregate=True, spec=None,
                    parents=None):
    """Materialise the enrolled slice of a :class:`LiTable` as observations.

    ``aggregate`` keeps one observation per ``(POI, target, assay)`` carrying the
    replicate values and their spread, rather than one per replicate row. The
    replicate identities and the per-replicate values are preserved as fields;
    what is avoided is 1.26 million Python objects, not the replicate structure.

    The edit map is the POI's **scaffold family common region**: the six annotated
    CDR spans, shared by every member of the family. It is *not* the POI's own
    mutated positions against its parent, which is a different region for every
    row and leaks the edit count into the window width. The mutated positions are
    still derived and carried in ``raw`` for the local contrasts that need them.

    ``parents`` supplies parent-control sequences the release itself does not
    carry -- the second Li release has no control row, and its two families are
    the first release's MIT_14. A POI whose family cannot be resolved is counted
    as unassignable and enrolled with its family region and no parent, so it
    trains the window task and not the contextual contrasts.
    """
    import numpy as np
    spec = spec or SOURCES[source]
    wanted = (frozenset(str(name) for name in targets) if targets is not None
              else frozenset(table.targets))
    target_rows = np.isin(table.target_index,
                          [index for index, name in enumerate(table.targets)
                           if name in wanted])
    order = np.flatnonzero(target_rows)
    groups = {}
    for row in order:
        key = (int(table.poi_index[row]), int(table.target_index[row]),
               int(table.assay_index[row]))
        groups.setdefault(key, []).append(int(row))
    controls = dict(table.parent_controls())
    for name, sequences in dict(parents or {}).items():
        controls.setdefault(str(name), tuple(sequences))
    family_of_poi = {}
    if table.family_index is not None and len(table.families):
        for position, poi in enumerate(table.pois):
            index = int(table.family_index[position])
            family_of_poi[poi] = (index, table.family_sites[index] if index >= 0 else ())
    observations, unassignable, unalignable, unannotated = [], 0, 0, 0
    for (poi_row, target_row, assay_row), rows in sorted(groups.items()):
        poi = table.pois[poi_row]
        heavy, light = table.sequences[poi]
        parent_name = li_parent_of(poi, source=source)
        values = [float(table.value[row]) for row in rows]
        replicates = [int(table.replicate[row]) for row in rows]
        family_index, region = family_of_poi.get(poi, (None, ()))
        notes = []
        if not region:
            unannotated += 1
            notes.append("the six CDR annotations did not locate uniquely for this POI")
        mutated = ()
        if parent_name is None:
            unassignable += 1
            notes.append("no declared parent family for this POI prefix")
        elif parent_name in controls:
            parent_heavy, parent_light = controls[parent_name]
            derived = derive_sites_against(heavy, light, parent_heavy, parent_light)
            if derived is None:
                unalignable += 1
                notes.append("length mismatch against the declared parent control")
            else:
                mutated = derived
        else:
            notes.append(f"parent control {parent_name} is not available for this release")
        construct = Construct(heavy=heavy, light=light, sites=tuple(region),
                              source=source, record_id=str(poi))
        mean = float(np.mean(values))
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else None
        pkd = pkd_from_log10_nm(mean)
        rows_out = [{"replicate": rep, "log10_kd_nm": value}
                    for rep, value in zip(replicates, values)]
        observations.append(Observation(
            construct=construct, source=source, family=spec.family, kind=spec.kind,
            target=(spec.target if table.targets[target_row] == "MIT_Target"
                    else table.targets[target_row]), assay=f"{table.assays[assay_row]}",
            parent=parent_name or f"{source}::unassigned", scale=SCALE_PKD,
            value=pkd, interval=(pkd, pkd),
            replicate=None if aggregate else replicates[0],
            uncertainty=None if spread is None else spread,
            raw={"POI": poi, "Target": table.targets[target_row],
                 "Assay": table.assays[assay_row], "replicates": rows_out,
                 "measurement_id": str(poi),
                 "mutated_sites": [[chain, int(index)] for chain, index in mutated],
                 "region_sites": len(region),
                 "scaffold_family": family_index,
                 "log10_kd_nm_mean": mean},
            qualification={"raw_units": "log10(KD/nM)",
                           "replicate_support": len(values),
                           "replicate_values": [float(value) for value in values],
                           "parent_control": parent_name,
                           "parent_rule": ("the declared family parent, never a modal sequence "
                                           "across families. A release with no control row of its "
                                           "own takes the sequence from the release that has one, "
                                           "through an explicit cross-release table."),
                           "region_rule": ("the scaffold family's six annotated CDR spans, shared "
                                           "by every member of the family. Not the row's own "
                                           "mutated positions."),
                           "unresolved": "; ".join(notes) or None}))
    return {"observations": tuple(observations), "unassignable_poi": unassignable,
            "unalignable_against_parent": unalignable,
            "poi_without_annotation": unannotated,
            "parent_controls_available": sorted(controls),
            "targets_enrolled": sorted(wanted),
            "aggregate": bool(aggregate)}


def load_li(path, *, source, preamble=0, cache=None, targets=None, chunk_rows=200000,
            parents=None):
    """FLAb/primary Li AlphaSeq, streamed. The preamble is asserted before any skip.

    Target, assay and replicate are retained: the release carries four targets and
    collapsing them would pool a measured target with three negative controls.
    ``targets=None`` enrols every target present, which is what a small test file
    wants; production names the measured target explicitly and records the
    counter-target rows separately.

    ``parents`` carries parent-control sequences from another release of the same
    experiment. The second release publishes no control row, and its two families
    are the first release's MIT_14 -- supported by the framework signature AND by
    the primary design names -- so the sequence is copied through an explicit
    cross-release table rather than left on a placeholder.
    """
    spec = SOURCES[source]
    table = stream_li_table(path, source=source, preamble=int(preamble),
                            chunk_rows=int(chunk_rows), cache=cache)
    block = li_observations(table, source=source, targets=targets, spec=spec,
                            parents=parents)
    controls = table.parent_controls()
    return {"observations": block["observations"], "table": table,
            "parent": _parent_record(table, controls, parents),
            "rows": int(table.rows), "missing_affinity": int(table.missing),
            "unalignable": block["unalignable_against_parent"],
            "unassignable_poi": block["unassignable_poi"],
            "poi_without_annotation": block["poi_without_annotation"],
            "parent_controls_available": block["parent_controls_available"],
            "preamble_lines": int(preamble), "member": table.member,
            "sha256": table.sha256, "inventory": table.document()}


def li_parent_export(table):
    """``{parent name: (heavy, light)}`` for the controls one release carries.

    The cross-release handoff, as an explicit function rather than as a field read
    out of a dict at a call site: the first release's ``MIT_*`` rows are the only
    verified source of those sequences, and the second release needs MIT_14.
    """
    return {str(name): (str(heavy), str(light))
            for name, (heavy, light) in sorted(table.parent_controls().items())}


def _parent_record(table, controls, supplied):
    """What backgrounds this release resolved, and where each one came from."""
    imported = sorted(set(dict(supplied or {})) - set(controls))
    return {"controls": sorted(controls), "imported_controls": imported,
            "resolved": bool(controls or imported),
            "rule": ("declared family parents. A control present in this release is used from "
                     "it; one that is not is imported from the release of the same experiment "
                     "that carries it, named here. No modal sequence is substituted -- a modal "
                     "sequence looks like a parent and produces an edit map against a variant."),
            "unresolved_reason": (None if (controls or imported) else
                                  "this file carries no MIT_* parent-control row and none was "
                                  "supplied, so variants from it are enrolled with their family "
                                  "region and no background for local contrasts")}


#: The single-chain-Fv linker the AlphaSeq releases use. Present in the paired
#: constructs and absent from the nanobody one, which is how the orientation is
#: DERIVED rather than declared per file.
ALPHASEQ_LINKER = "GGGGSGGGGSGGGGS"

#: The non-HER2 AlphaSeq populations, their measured target and their orientation.
#: Orientation is verified against the sequences at load time; this table is what
#: the verification is checked against, not what it trusts.
ALPHABIND_NONHER2 = {
    "alphabind_aab_pp489": {"target": "TIGIT", "orientation": "paired_scfv",
                            "descriptor": "TIGIT_22-137_POI-AGA2"},
    "alphabind_pembrolizumab": {"target": "PD-1", "orientation": "paired_scfv",
                                "descriptor": "huPD-1_RE(rebuilt)"},
    "alphabind_vhh72": {"target": "SARS-CoV-2 RBD", "orientation": "heavy_only",
                        "descriptor": "SARS-CoV2_RBD_(6LZG)"},
}


def split_alphaseq_scfv(sequence, *, linker=ALPHASEQ_LINKER):
    """``(heavy, light, orientation)`` by locating the linker, or ``None``.

    The HER2 population is a length-243 construct at fixed offsets and is parsed
    by :func:`split_alphaseq_construct`. The non-HER2 populations are not: they
    carry insertions, deletions and full framework substitutions, so a fixed
    offset would silently reassign the chain boundary for every indel row. The
    linker is located instead, and a construct with no linker is heavy-only --
    which is what the nanobody release is, and no light chain is invented for it.
    """
    text = str(sequence).strip().upper()
    if not text:
        return None
    first = text.find(linker)
    if first < 0:
        return text, "", "heavy_only"
    if text.find(linker, first + 1) >= 0:
        return None       # two linkers: the chain boundary is not identified
    return text[:first], text[first + len(linker):], "paired_scfv"


def load_alphabind_nonher2(paths, *, source, target=None, orientation=None,
                           region_maximum=REGION_MAXIMUM):
    """A non-HER2 AlphaSeq population: fine-tuning and validation files together.

    Three things this loader does that the HER2 one does not need to:

    * **the chain boundary is located, not assumed.** These releases carry
      insertions, deletions and framework substitutions, so the length-243 offsets
      of the trastuzumab population would reassign the boundary on any indel row.
    * **rows whose length differs from the population's modal parent are kept.**
      They get no shared region -- an indel needs an alignment decision this
      campaign has not made -- so they train the outcome heads, which read the
      whole molecule, and not the window tasks, which need a shared support. The
      count is reported. Dropping them would delete the part of the release that
      carries the indels.
    * **``avg_off_target_affinity`` is preserved raw.** It is a measured
      counter-target readout *for these targets*, which is exactly what the HER2
      population does not have; E12 stays unsupported for HER2 and this column is
      what a non-HER2 specificity panel would be built from.
    """
    spec = SOURCES[source]
    declared = dict(ALPHABIND_NONHER2.get(source) or {})
    target = target or declared.get("target") or spec.target
    orientation = orientation or declared.get("orientation")
    rows, per_file = [], {}
    for entry in list(paths):
        path = Path(entry)
        require(path.is_file(), f"{path} is missing")
        block = csv_rows(path)
        require(block, f"{path.name} carried no data rows")
        for column in ("mata_sequence", "alphaseq_affinity", "mata_description",
                       "matalpha_description"):
            require(column in block[0],
                    f"{path.name} has no {column!r} column; it has {sorted(block[0])}")
        partition = "validation" if "validation" in path.name else "fine_tuning"
        for position, row in enumerate(block):
            rows.append((partition, path.name, position, row))
        per_file[path.name] = {"rows": len(block), "partition": partition,
                              "sha256": sha256(path),
                              "has_off_target": "avg_off_target_affinity" in block[0]}
    require(rows, f"No row was readable for {source}")
    parsed, rejected, descriptors, orientations = [], {}, {}, {}
    for partition, member, position, row in rows:
        split = split_alphaseq_scfv(row.get("mata_sequence"))
        if split is None:
            rejected["the linker does not identify one chain boundary"] = rejected.get(
                "the linker does not identify one chain boundary", 0) + 1
            continue
        heavy, light, observed = split
        orientations[observed] = orientations.get(observed, 0) + 1
        descriptors[str(row.get("matalpha_description"))] = descriptors.get(
            str(row.get("matalpha_description")), 0) + 1
        parsed.append((partition, member, position, row, heavy, light, observed))
    require(parsed, f"No {source} row produced a construct: {rejected}")
    dominant = max(sorted(orientations.items()), key=lambda item: item[1])[0]
    if orientation is not None:
        require(dominant == orientation,
                f"{source}: the sequences are dominantly {dominant!r} and the release table "
                f"declares {orientation!r}. The orientation is derived from the linker and the "
                "disagreement is a changed release, not a label to override.")
    controls = [(heavy, light) for partition, _m, _i, row, heavy, light, _o in parsed
                if partition == "fine_tuning" and finite(row.get("edit_distance")) == 0.0]
    if controls:
        require(len(set(controls)) == 1,
                f"{source}: zero-edit fine-tuning controls name multiple parents")
        parent = _majority_parent(controls)
        parent["qualification"] = "unique explicit zero-edit fine-tuning parent controls"
    else:
        parent = _majority_parent([(heavy, light) for _p, _m, _i, _r, heavy, light, _o in parsed])
        parent["qualification"] = "modal reference only; no explicit zero-edit control in input"
    observations = []
    for partition, member, position, row, heavy, light, observed in parsed:
        raw_value = finite(row.get("alphaseq_affinity"))
        if raw_value is None:
            rejected["non-finite alphaseq_affinity"] = rejected.get(
                "non-finite alphaseq_affinity", 0) + 1
            continue
        try:
            construct = Construct(heavy=heavy, light=light, sites=(), source=source,
                                  record_id=f"{member}:{row.get('id') or position}")
        except ValueError:
            rejected["non-canonical residue"] = rejected.get("non-canonical residue", 0) + 1
            continue
        pkd = pkd_from_log10_nm(raw_value)
        observations.append(Observation(
            construct=construct, source=source, family=spec.family, kind=KIND_POINT,
            target=str(target), assay=f"{source}:{partition}",
            parent=f"{source}::parental", scale=SCALE_PKD, value=pkd, interval=(pkd, pkd),
            censoring=CENSOR_NONE,
            raw={"alphaseq_affinity": raw_value, "edit_distance": finite(row.get("edit_distance")),
                 "parental_mean_aff": finite(row.get("parental_mean_aff")),
                 "parental_aff_margin": finite(row.get("parental_aff_margin")),
                 "avg_off_target_affinity": finite(row.get("avg_off_target_affinity")),
                 "predicted_affinity": finite(row.get("predicted_affinity")),
                 "mata_description": row.get("mata_description"),
                 "matalpha_description": row.get("matalpha_description"),
                 "model": row.get("model"), "member": member,
                 "measurement_id": f"{member}:{row.get('id') or position}",
                 "release_partition": partition},
            qualification={
                "raw_units": "log10(KD/nM)",
                "orientation": observed,
                "predicted_affinity_is_not_measured": True,
                "off_target_column": ("preserved raw. It is a measured counter-target readout for "
                                      "THIS target, which the HER2 population does not have; E12 "
                                      "stays unsupported for HER2 either way."),
                "indel_policy": ("a row whose chain lengths differ from the population's modal "
                                 "parent keeps its outcome and loses its shared region: it trains "
                                 "the outcome heads and not the window tasks, and the count is "
                                 "reported rather than the population truncated.")}))
    require(observations, f"No {source} observation survived: {rejected}")
    observations, unalignable, region = _assign_sites(observations, parent,
                                                     maximum=region_maximum)
    return {"observations": observations, "parent": parent,
            "rows": sum(block["rows"] for block in per_file.values()),
            "missing": sum(rejected.values()), "unalignable": unalignable,
            "region": region, "per_file": per_file,
            "excluded": dict(sorted(rejected.items())),
            "orientations": dict(sorted(orientations.items())),
            "antigen_descriptors": dict(sorted(descriptors.items())),
            "target": str(target),
            "sha256": sha256(Path(list(paths)[0]))}


def load_adams_affinity(path, *, source="adams2017_fluorescein",
                        window=ADAMS_WINDOW_MOLAR):
    """Adams Tite-Seq, doubly censored, with the format-label disagreement recorded.

    Sign convention, derived in one place: ``pKD = -log10(KD/M)``, so a KD **below**
    the window (tighter than 1e-9.5 M) gives a pKD **above** 9.5 and is
    right-censored; a KD above the window gives a pKD below 5 and is left-censored.
    """
    spec = SOURCES[source]
    name, rows = zipped_csv_rows(path) if str(path).endswith(".zip") else \
        (Path(path).name, csv_rows(path))
    require(rows, f"{name} carried no rows")
    column = "Kd (Tite-Seq) [M]"
    require(column in rows[0], f"{name} has no {column!r} column; it has {sorted(rows[0])}")
    low_pkd, high_pkd = pkd_from_molar(window[1]), pkd_from_molar(window[0])
    parent = _majority_parent([(str(r["heavy"]).strip().upper(),
                                str(r["light"]).strip().upper()) for r in rows])
    observations, missing, formats = [], 0, {}
    for position, row in enumerate(rows):
        molar = finite(row.get(column))
        formats[str(row.get("format"))] = formats.get(str(row.get("format")), 0) + 1
        if molar is None or molar <= 0:
            missing += 1
            continue
        pkd = pkd_from_molar(molar)
        if pkd >= high_pkd:
            interval, censoring = (high_pkd, None), CENSOR_RIGHT
        elif pkd <= low_pkd:
            interval, censoring = (None, low_pkd), CENSOR_LEFT
        else:
            interval, censoring = (pkd, pkd), CENSOR_NONE
        observations.append(Observation(
            construct=_paired_construct(row["heavy"], row["light"], source=source,
                                        record_id=position),
            source=source, family=spec.family, kind=KIND_INTERVAL, target=spec.target,
            assay=source, parent=f"{source}::parental", scale=SCALE_PKD,
            value=pkd if censoring == CENSOR_NONE else None, interval=interval,
            censoring=censoring,
            raw={"Kd (Tite-Seq) [M]": molar, "format": row.get("format"),
                 "fitness": finite(row.get("fitness")), "pkd": pkd},
            qualification={
                "detection_window_molar": [float(window[0]), float(window[1])],
                "detection_window_pkd": [low_pkd, high_pkd],
                "sign_convention": ("pKD = -log10(KD/M). A KD below the window is TIGHTER "
                                    "binding, hence right-censored in pKD; a KD above the window "
                                    "is left-censored."),
                "format_label": row.get("format"),
                "format_qualification": (
                    "FLAb's format column is a curation label. The primary Tite-Seq experiment "
                    "(eLife 10.7554/eLife.23156) was scFv, not Fab. The disagreement is carried, "
                    "not resolved."),
                "no_universal_1000_limit": (
                    "no numeric value is treated as a censoring limit; the limits above come from "
                    "the primary paper's declared detection window")}))
    observations, unalignable, region = _assign_sites(observations, parent,
                                                      maximum=REGION_MAXIMUM)
    return {"observations": observations, "parent": parent, "rows": len(rows),
            "missing_affinity": missing, "unalignable": unalignable, "region": region,
            "format_counts": dict(sorted(formats.items())), "member": name,
            "sha256": sha256(path)}


def load_expression(path, *, source, column):
    """An expression / enrichment-ratio table. Ordering only, never an affinity."""
    spec = SOURCES[source]
    name, rows = zipped_csv_rows(path) if str(path).endswith(".zip") else \
        (Path(path).name, csv_rows(path))
    require(rows, f"{name} carried no rows")
    require(column in rows[0], f"{name} has no {column!r} column; it has {sorted(rows[0])}")
    parent_fn = single_mutant_consensus if source == "koenig2017_expression" else _majority_parent
    parent = parent_fn([(str(r["heavy"]).strip().upper(),
                         str(r["light"]).strip().upper()) for r in rows])
    observations, missing = [], 0
    for position, row in enumerate(rows):
        value = finite(row.get(column))
        if value is None:
            missing += 1
            continue
        observations.append(Observation(
            construct=_paired_construct(row["heavy"], row["light"], source=source,
                                        record_id=position),
            source=source, family=FAMILY_ENRICHMENT, kind=KIND_ENRICHMENT, target=spec.target,
            assay=source, parent=f"{source}::parental", scale=spec.scale, value=value,
            raw={column: value, "format": row.get("format"),
                 "fitness": finite(row.get("fitness"))},
            qualification={"readout": column,
                           "is_not_an_affinity": (
                               "an expression or enrichment ratio is a different functional "
                               "readout. E16 compares affinity among expression-matched variants "
                               "and expression among affinity-matched variants; it never converts "
                               "one into the other.")}))
    observations, unalignable, region = _assign_sites(observations, parent,
                                                      maximum=REGION_MAXIMUM)
    return {"observations": observations, "parent": parent, "rows": len(rows),
            "missing": missing, "unalignable": unalignable, "region": region,
            "member": name, "sha256": sha256(path)}


def koenig_column_consistency(rows):
    """Check the four KD-named Koenig columns against each other. Residuals recorded.

    ``-ln(KD (nM))`` and ``-log( KD (M) )`` are two transforms of one number. If
    they agree, that establishes the release is internally consistent and nothing
    more: agreeing derived columns are still derived from enrichment. If they
    disagree, the mapping is unresolved and the rows train ordering only, which is
    what they do anyway.
    """
    worst_ln, worst_log, checked = 0.0, 0.0, 0
    for row in rows:
        nm = finite(row.get("KD (nM)"))
        molar = finite(row.get("KD (M)"))
        neg_ln = finite(row.get("-ln(KD (nM) )"))
        neg_log = finite(row.get("-log( KD (M) )"))
        if nm is None or nm <= 0:
            continue
        checked += 1
        if neg_ln is not None:
            worst_ln = max(worst_ln, abs(neg_ln - (-math.log(nm))))
        if molar is not None and molar > 0 and neg_log is not None:
            worst_log = max(worst_log, abs(neg_log - (-math.log10(molar))))
        if molar is not None and molar > 0:
            worst_log = max(worst_log, abs(math.log10(molar) - math.log10(nm * 1e-9)))
    return {"rows_checked": checked, "max_abs_error_neg_ln_nm": worst_ln,
            "max_abs_error_neg_log_molar": worst_log,
            "interpretation": ("internal consistency of four transforms of one number. It "
                              "establishes that the release is self-consistent and nothing about "
                              "whether the number is a titration. The primary paper describes "
                              "phage enrichment.")}


#: The Koenig binding evidence that is NOT sufficient to establish a direction,
#: recorded so the quarantine is a measurement rather than a caution.
KOENIG_DIRECTION_EVIDENCE = {
    "pkd_vs_qualified_expression_spearman": -0.499,
    "top_curated_pkd_mutants": ["H102F>N", "H102F>P", "H99F>D"],
    "reading": ("a -0.499 rank correlation between the pseudo-KD column and the qualified "
                "expression ratio, and a top-of-distribution occupied by substitutions at two "
                "contact positions, are consistent with EITHER direction: an enrichment ratio "
                "that rises with binding and an inverted pseudo-KD produce the same picture. "
                "This is caution, not proof, and it does not qualify an ordering."),
    "what_would_resolve_it": ("the primary release's own enrichment-to-KD calibration, or a "
                              "declared sign convention in the source. Neither is in the files "
                              "acquired here."),
}


def load_koenig(path, *, source="koenig2017_g6", quarantine_binding=True,
                 historical_path=None):
    """Qualify G6.31 binding enrichment against the pinned historical release.

    The current KD-named column matches historical enrichment ``fitness`` for
    every sequence. Qualified values use positive log enrichment, never pKD.
    Without that comparison the values remain quarantined.
    """
    spec = SOURCES[source]
    name, rows = zipped_csv_rows(path) if str(path).endswith(".zip") else \
        (Path(path).name, csv_rows(path))
    require(rows, f"{name} carried no rows")
    consistency = koenig_column_consistency(rows)
    historical_qualification = None
    if historical_path is not None:
        historical_rows = csv_rows(historical_path)
        historical = {(str(row["heavy"]).strip().upper(), str(row["light"]).strip().upper()):
                      float(row["fitness"]) for row in historical_rows}
        require(len(historical) == len(rows), "Koenig historical/current population changed")
        for row in rows:
            key = (str(row["heavy"]).strip().upper(), str(row["light"]).strip().upper())
            value = finite(row.get("KD (nM)"))
            require(key in historical and value is not None and value > 0
                    and math.isclose(value, historical[key], rel_tol=1e-8, abs_tol=1e-10),
                    "Koenig current KD-named value differs from its historical fitness")
        historical_qualification = {
            "sha256": sha256(historical_path), "matched_sequences": len(rows),
            "binding_direction": "higher enrichment is better; positive natural log",
            "primary": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5278476/",
            "historical_revision": "67738eea4841a1777b73609d56ddfa39de8d7360",
            "reanalysis_revision": "ce1b71c605f3d584ac24423633e51fda35efeb00"}
        quarantine_binding = False
    require(quarantine_binding or historical_qualification is not None,
            "Unquarantining Koenig requires the independently pinned historical comparison")
    parent = single_mutant_consensus([(str(r["heavy"]).strip().upper(),
                                str(r["light"]).strip().upper()) for r in rows])
    observations, missing, quarantined = [], 0, 0
    for position, row in enumerate(rows):
        pseudo = finite(row.get("-ln(KD (nM) )"))
        if pseudo is None:
            missing += 1
            continue
        if quarantine_binding:
            quarantined += 1
        observations.append(Observation(
            construct=_paired_construct(row["heavy"], row["light"], source=source,
                                        record_id=position),
            source=source, family=FAMILY_ENRICHMENT, kind=KIND_ENRICHMENT, target=spec.target,
            assay="phage_binding", parent=f"{source}::G6.31", scale=SCALE_ENRICHMENT,
            value=None if quarantine_binding else math.log(float(row["KD (nM)"])),
            raw={"-ln(KD (nM) )": pseudo, "KD (nM)": finite(row.get("KD (nM)")),
                 "KD (M)": finite(row.get("KD (M)")),
                 "-log( KD (M) )": finite(row.get("-log( KD (M) )")),
                 "fitness": finite(row.get("fitness"))},
            qualification={
                "curation_mapping": (
                    "the FLAb release names these columns KD. The primary paper (PMC5278476) "
                    "describes deep mutational PHAGE ENRICHMENT, not 4,275 physical titrations, "
                    "so the numbers are enrichment-derived pseudo-KD."),
                "quarantined": bool(quarantine_binding),
                "trains": ("nothing. The value is withheld until the mapping is qualified."
                           if quarantine_binding
                           else "within-study ordering only; never an interval likelihood"),
                "historical_qualification": historical_qualification,
                "unresolved": None if historical_qualification else (
                    "the enrichment-to-pseudo-KD mapping AND ITS DIRECTION are not established "
                    "by the acquired sources. No ordinal fallback is fabricated in its place, "
                    "and no ordering is trained from an unsigned quantity."),
                "direction_evidence": historical_qualification or dict(KOENIG_DIRECTION_EVIDENCE),
                "copying_a_curation_does_not_establish_units": (
                    "that another curation labelled these columns KD is evidence about that "
                    "curation, not about the experiment."),
                "column_consistency": dict(consistency)}))
    observations, unalignable, region = _assign_sites(observations, parent,
                                                      maximum=REGION_MAXIMUM)
    return {"observations": observations, "parent": parent, "rows": len(rows),
            "missing": missing, "unalignable": unalignable, "region": region,
            "consistency": consistency,
            "quarantined_binding_rows": quarantined,
            "quarantine": (historical_qualification or dict(KOENIG_DIRECTION_EVIDENCE)),
            "member": name, "sha256": sha256(path)}


def qualify_interval(estimate, low, high, *, minimum=0.0):
    """Accept a confidence interval only if it is real. Returns ``(interval, reason)``.

    Petersen's supplements carry bounds that are zero, inverted, or that do not
    enclose the estimate. Each is preserved raw and **rejected** for interval
    training with the reason recorded. Nothing is repaired: an invented valid
    interval would state a precision the experiment did not report.
    """
    estimate = finite(estimate)
    low, high = finite(low), finite(high)
    if estimate is None:
        return None, "no point estimate"
    if low is None or high is None:
        return None, "a bound is missing"
    if low <= float(minimum) or high <= float(minimum):
        return None, "a bound is at or below the physical floor"
    if low > high:
        return None, "bounds inverted"
    if not (low <= estimate <= high):
        return None, "bounds do not enclose the estimate"
    return (low, high), "qualified"


def load_petersen_flab(path, *, source="petersen2024_ha"):
    """The FLAb Petersen table: ``heavy, light, KD, fitness``. KD in nM."""
    spec = SOURCES[source]
    name, rows = zipped_csv_rows(path) if str(path).endswith(".zip") else \
        (Path(path).name, csv_rows(path))
    require(rows, f"{name} carried no rows")
    require("KD" in rows[0], f"{name} has no 'KD' column; it has {sorted(rows[0])}")
    parent = _majority_parent([(str(r["heavy"]).strip().upper(),
                                str(r["light"]).strip().upper()) for r in rows])
    observations, missing, above_thousand = [], 0, 0
    for position, row in enumerate(rows):
        nm = finite(row.get("KD"))
        if nm is None or nm <= 0:
            missing += 1
            continue
        if nm > 1000.0:
            above_thousand += 1
        pkd = pkd_from_log10_nm(math.log10(nm))
        observations.append(Observation(
            construct=_paired_construct(row["heavy"], row["light"], source=source,
                                        record_id=position),
            source=source, family=spec.family, kind=KIND_POINT, target=spec.target,
            assay=source, parent=f"{source}::parental", scale=SCALE_PKD, value=pkd,
            interval=(pkd, pkd),
            raw={"KD_nM": nm, "fitness": finite(row.get("fitness"))},
            qualification={"raw_units": "KD in nM",
                           "uncertainty": "absent in the FLAb table; the originals carry bounds",
                           "no_universal_1000_limit": (
                               "values above 1000 nM occur in this release, so 1000 is not a "
                               "censoring limit here or anywhere else in this campaign")}))
    observations, unalignable, region = _assign_sites(observations, parent,
                                                      maximum=REGION_MAXIMUM)
    return {"observations": observations, "parent": parent, "rows": len(rows),
            "missing": missing, "unalignable": unalignable, "region": region,
            "values_above_1000_nm": above_thousand, "member": name, "sha256": sha256(path)}


#: Adams primary: the wild-type heavy chain, and the two windows that vary.
#: Every other position of the 117-mer is constant across the library, which is
#: what makes filling two windows into a fixed backbone a reconstruction of the
#: real construct rather than a splice.
ADAMS_WT_HEAVY = ("EVKLDETGGGLVQPGRPMKLSCVASGFTFSDYWMNWVRQSPEKGLEWVAQIRNKPYNYETYYSDSVKG"
                  "RFTISRDDSKSSVYLQMNNLRVEDMGIYYCTGSYYGMDYWGQGTSVTVS")
ADAMS_CDR1_SLICE = (27, 37)
ADAMS_CDR3_SLICE = (99, 109)
ADAMS_WT_CDR1 = "TFSDYWMNWV"
ADAMS_WT_CDR3 = "GSYYGMDYWG"
ADAMS_LIGHT_LENGTH = 112
#: Censoring bounds in pKD, from the primary Tite-Seq detection window.
ADAMS_LEFT_CENSOR_PKD = 5.0
ADAMS_RIGHT_CENSOR_PKD = 9.5
#: ``KD_fit_log_poiss.py`` computes the fit variance in NATURAL-log KD, so the
#: reported sigma is divided by ln(10) to become a sigma in pKD. Using it as-is
#: overstates every Adams uncertainty by a factor of 2.303.
ADAMS_SIGMA_LOG_BASE = math.log(10.0)


def adams_wild_type_heavy():
    """The asserted wild-type heavy chain and the two variable windows."""
    require(len(ADAMS_WT_HEAVY) == 117,
            f"The Adams wild-type heavy chain is {len(ADAMS_WT_HEAVY)} residues, expected 117")
    require(ADAMS_WT_HEAVY[slice(*ADAMS_CDR1_SLICE)] == ADAMS_WT_CDR1,
            f"Heavy[{ADAMS_CDR1_SLICE}] is "
            f"{ADAMS_WT_HEAVY[slice(*ADAMS_CDR1_SLICE)]!r}, not the declared "
            f"{ADAMS_WT_CDR1!r}")
    require(ADAMS_WT_HEAVY[slice(*ADAMS_CDR3_SLICE)] == ADAMS_WT_CDR3,
            f"Heavy[{ADAMS_CDR3_SLICE}] is "
            f"{ADAMS_WT_HEAVY[slice(*ADAMS_CDR3_SLICE)]!r}, not the declared "
            f"{ADAMS_WT_CDR3!r}")
    return ADAMS_WT_HEAVY


def adams_construct_heavy(cdr1, cdr3):
    """The full 117-mer carrying one CDR1 and one CDR3. Lengths are asserted."""
    heavy = adams_wild_type_heavy()
    cdr1, cdr3 = str(cdr1).strip().upper(), str(cdr3).strip().upper()
    require(len(cdr1) == ADAMS_CDR1_SLICE[1] - ADAMS_CDR1_SLICE[0],
            f"CDR1 {cdr1!r} does not fill positions {ADAMS_CDR1_SLICE}")
    require(len(cdr3) == ADAMS_CDR3_SLICE[1] - ADAMS_CDR3_SLICE[0],
            f"CDR3 {cdr3!r} does not fill positions {ADAMS_CDR3_SLICE}")
    return (heavy[:ADAMS_CDR1_SLICE[0]] + cdr1 + heavy[ADAMS_CDR1_SLICE[1]:ADAMS_CDR3_SLICE[0]]
            + cdr3 + heavy[ADAMS_CDR3_SLICE[1]:])


def adams_edit_sites():
    """The twenty variable heavy coordinates, as an explicit edit map."""
    return edit_map([(HEAVY, index) for index in
                     list(range(*ADAMS_CDR1_SLICE)) + list(range(*ADAMS_CDR3_SLICE))])


def adams_expression_normaliser(rows, *, column="expression"):
    """The per-replicate wild-type-synonymous expression level.

    Every row's expression is reported on that replicate's own scale, so the
    across-replicate comparison needs a per-replicate reference. The reference is
    the set of rows whose translated CDR1 **and** CDR3 are both wild type -- the
    synonymous-DNA copies of the parent, which is the one construct present in
    every replicate by design. The median is used rather than the mean: the
    column is a ratio of sorted counts and its tail is long.
    """
    import numpy as np
    values = [finite(row.get(column)) for row in rows
              if str(row.get("CDR1H_AA", "")).strip().upper() == ADAMS_WT_CDR1
              and str(row.get("CDR3H_AA", "")).strip().upper() == ADAMS_WT_CDR3]
    values = [value for value in values if value is not None and value > 0]
    if not values:
        return {"normaliser": None, "wild_type_synonymous_rows": 0,
                "reason": ("no wild-type-synonymous row in this replicate, so its expression "
                           "column is reported raw and is NOT comparable across replicates")}
    return {"normaliser": float(np.median(values)),
            "wild_type_synonymous_rows": len(values),
            "rule": "median expression of the WT-synonymous rows of this replicate"}


def load_adams_primary(root, *, source="adams2017_primary", light=None,
                       replicates=(1, 2, 3)):
    """The Adams primary replicate tables: real biological replicates and uncertainty.

    What this adds over the curated FLAb summary, and why each matters:

    * **three biological replicates**, kept as separate observations of one
      construct rather than averaged into a point;
    * **``fit_KD_sigma`` converted to pKD** by dividing by ``ln(10)``, because
      ``KD_fit_log_poiss.py`` computes the variance in natural-log KD. Carrying
      it unconverted states an uncertainty 2.303x too wide;
    * **the full 117-mer construct**, rebuilt from the wild-type backbone and the
      two translated windows the tables carry, so the record is a paired antibody
      and not two CDR strings;
    * **per-replicate expression**, normalised by that replicate's
      wild-type-synonymous rows.

    The curated summary and these rows are the same experiment:
    :data:`CANONICAL_STUDY` maps both onto ``adams2017_titeseq`` so
    :func:`observation_closure` cannot count them twice.
    """
    spec = SOURCES["adams2017_fluorescein"]
    root = Path(root)
    require(light and len(str(light)) == ADAMS_LIGHT_LENGTH,
            f"The Adams light chain is a fixed {ADAMS_LIGHT_LENGTH}-mer and must be supplied "
            "from the curated table; it is not derivable from the replicate files and is not "
            "invented here.")
    light = str(light).strip().upper()
    sites = adams_edit_sites()
    observations, per_replicate, missing = [], {}, 0
    for number in replicates:
        path = root / "data" / f"replicate_{int(number)}.csv"
        require(path.is_file(), f"Adams primary replicate file {path} is missing")
        rows = csv_rows(path)
        require(rows, f"{path.name} carried no rows")
        for column in ("CDR1H_AA", "CDR3H_AA", "fit_KD", "fit_KD_sigma", "expression"):
            require(column in rows[0],
                    f"{path.name} has no {column!r} column; it has {sorted(rows[0])[:12]}...")
        normaliser = adams_expression_normaliser(rows)
        kept = 0
        for position, row in enumerate(rows):
            molar = finite(row.get("fit_KD"))
            cdr1 = str(row.get("CDR1H_AA", "")).strip().upper()
            cdr3 = str(row.get("CDR3H_AA", "")).strip().upper()
            if molar is None or molar <= 0 or len(cdr1) != 10 or len(cdr3) != 10:
                missing += 1
                continue
            try:
                heavy = adams_construct_heavy(cdr1, cdr3)
                construct = Construct(heavy=heavy, light=light, sites=sites, source=source,
                                      record_id=f"rep{number}:{position}")
            except ValueError:
                missing += 1
                continue
            pkd = pkd_from_molar(molar)
            sigma_natural = finite(row.get("fit_KD_sigma"))
            sigma_pkd = (None if sigma_natural is None
                         else abs(float(sigma_natural)) / ADAMS_SIGMA_LOG_BASE)
            expression = finite(row.get("expression"))
            normalised = (None if expression is None or not normaliser["normaliser"]
                          else expression / normaliser["normaliser"])
            if pkd >= ADAMS_RIGHT_CENSOR_PKD:
                interval, censoring, value = (ADAMS_RIGHT_CENSOR_PKD, None), CENSOR_RIGHT, None
            elif pkd <= ADAMS_LEFT_CENSOR_PKD:
                interval, censoring, value = (None, ADAMS_LEFT_CENSOR_PKD), CENSOR_LEFT, None
            else:
                interval, censoring, value = (pkd, pkd), CENSOR_NONE, pkd
            observations.append(Observation(
                construct=construct, source=source, family=spec.family, kind=KIND_INTERVAL,
                target=spec.target, assay="adams2017_titeseq", parent=f"{source}::wild_type",
                scale=SCALE_PKD, value=value, interval=interval, censoring=censoring,
                replicate=int(number), uncertainty=sigma_pkd,
                raw={"fit_KD_molar": molar, "pkd": pkd,
                     "measurement_id": f"{row.get('CDR1H', '')}|{row.get('CDR3H', '')}",
                     "biological_replicate": int(number),
                     "fit_KD_sigma_natural_log": sigma_natural,
                     "expression": expression, "expression_normalised": normalised,
                     "CDR1H": row.get("CDR1H"), "CDR3H": row.get("CDR3H"),
                     "CDR1H_AA": cdr1, "CDR3H_AA": cdr3,
                     "unsorted": finite(row.get("unsorted")),
                     "fit_fraction": finite(row.get("fit_fraction")),
                     "fit_saturation": finite(row.get("fit_saturation"))},
                qualification={
                    "replicate_file": path.name,
                    "sigma_units": ("fit_KD_sigma is a standard deviation in NATURAL-log KD "
                                    "(KD_fit_log_poiss.py lines 337-339); it is divided by "
                                    "ln(10) to become a sigma in pKD"),
                    "sigma_pkd": sigma_pkd,
                    "expression_normaliser": dict(normaliser),
                    "detection_window_pkd": [ADAMS_LEFT_CENSOR_PKD, ADAMS_RIGHT_CENSOR_PKD],
                    "construct_rule": ("the full 117-mer wild-type heavy chain with positions "
                                       f"{ADAMS_CDR1_SLICE} and {ADAMS_CDR3_SLICE} replaced by "
                                       "the translated windows this row carries"),
                    "double_count_guard": ("the curated FLAb summary of this experiment maps to "
                                           "the same canonical study, so the closure keeps one "
                                           "observation per (construct, assay, replicate)")}))
            kept += 1
        per_replicate[f"replicate_{int(number)}"] = {
            "rows": len(rows), "enrolled": kept, "normaliser": dict(normaliser),
            "sha256": sha256(path)}
    require(observations, f"No Adams primary observation survived under {root}")
    return {"observations": tuple(observations), "rows": sum(
        block["rows"] for block in per_replicate.values()),
        "missing": missing, "unalignable": 0, "per_replicate": per_replicate,
        "light_chain_length": len(light),
        "sites": [[chain, int(index)] for chain, index in sites],
        "sha256": sha256(root / "data" / f"replicate_{int(replicates[0])}.csv")}


def derive_adams_light(curated_observations):
    """The fixed Adams light chain, taken from the curated table and validated.

    The replicate files carry only the two heavy windows, so the light chain has
    to come from the curated summary of the same experiment. It is required to be
    unique across every curated row: a mixture would mean the curated table is
    not one construct family and the primary rows could not be attached to it.
    """
    lights = {record.construct.light for record in curated_observations}
    require(len(lights) == 1,
            f"The curated Adams table carries {len(lights)} distinct light chains. The primary "
            "replicate files declare none, so a unique curated light chain is the only way to "
            "build the paired construct; a mixture is reported rather than resolved.")
    light = next(iter(lights))
    require(len(light) == ADAMS_LIGHT_LENGTH,
            f"The curated Adams light chain is {len(light)} residues, expected "
            f"{ADAMS_LIGHT_LENGTH}")
    return light


#: Petersen supplementary Data 6 sheets, from the workbook's own README.
PETERSEN_DATA6_SHEETS = ("README", "all_barcodes_4A8_CC121", "4A8_CC121_combined",
                         "all_barcodes_222-1C06", "all_barcodes_319-345", "topbinHA_rep2")
#: The two barcode sheets that carry ``Variant`` plus a Kd and its bounds.
PETERSEN_BARCODE_SHEETS = ("all_barcodes_222-1C06", "all_barcodes_319-345")


def load_petersen_parents(path):
    """The author parent export: ``Antibody, Chain, Sequence``.

    Credited to the Whitehead group and verified against the sequence-derived
    consensus of the release. The variant specifications are 1-based protein
    indices against these sequences, so without them a ``Variant`` string cannot
    be resolved and the row has no construct -- which is why the earlier loader
    returned raw rows and nothing else.
    """
    rows = csv_rows(path)
    require(rows, f"{Path(path).name} carried no rows")
    for column in ("Antibody", "Chain", "Sequence"):
        require(column in rows[0], f"{Path(path).name} has no {column!r} column")
    parents, by_antibody = {}, {}
    for row in rows:
        name = str(row["Antibody"]).strip()
        chain = str(row["Chain"]).strip().upper()
        sequence = str(row["Sequence"]).strip().upper()
        require(chain in ("VH", "VL"), f"{name}: unexpected chain {chain!r}")
        parents[(name, chain)] = sequence
        by_antibody.setdefault(name, {})[chain] = len(sequence)
    paired = sorted(name for name, chains in by_antibody.items() if set(chains) == {"VH", "VL"})
    return {"parents": parents, "antibodies": sorted(by_antibody), "paired": paired,
            "lengths": by_antibody, "sha256": sha256(path),
            "provenance": ("author export, credited to the Whitehead group. Verified to match "
                           "the sequence-derived consensus of the release exactly.")}


def load_petersen_coding_parents(path):
    """Translate the author's complete frame-zero coding sequences for 4A8/CC121."""
    from itertools import product
    amino_acids = ("FFLLSSSSYY**CC*W" "LLLLPPPPHHQQRRRR"
                   "IIIMTTTTNNKKSSRR" "VVVVAAAADDEEGGGG")
    codons = dict(zip(("".join(c) for c in product("TCAG", repeat=3)), amino_acids))
    parents = {}
    for row in csv_rows(path):
        name, chain = str(row["gene"]).strip(), str(row["vh_or_vl"]).strip().upper()
        dna = str(row["seq"]).strip().upper()
        require(chain in ("VH", "VL") and len(dna) % 3 == 0 and set(dna) <= set("ACGT"),
                f"Invalid complete coding sequence for {name}:{chain}")
        protein = "".join(codons[dna[i:i + 3]] for i in range(0, len(dna), 3))
        require(protein and "*" not in protein, f"Unexpected stop in parent {name}:{chain}")
        require((name, chain) not in parents, f"Duplicate coding parent {name}:{chain}")
        parents[(name, chain)] = protein
    require(parents, "No coding parents were loaded")
    return {"parents": parents, "sha256": sha256(path),
            "provenance": "author WT/wt.csv, complete seq field translated in frame zero"}


def load_petersen_primary(path, *, parents, sheets=PETERSEN_BARCODE_SHEETS,
                          source="petersen2024_primary", target="H1 HA"):
    """MAGMA-seq barcode rows joined to their resolved parental backgrounds.

    Each row keeps its target study, its original barcode, its replicate and its
    raw units. An interval is enrolled only if :func:`qualify_interval` accepts
    it: bounds that are zero, inverted or that do not enclose the estimate are
    preserved raw and rejected, never repaired, and 1000 nM is not treated as a
    censoring limit here or anywhere else.
    """
    spec = SOURCES["petersen2024_ha"]
    observations, rejected, unresolved, per_sheet = [], {}, 0, {}
    sheet_rows = {sheet: read_xlsx_sheet(path, sheet) for sheet in sheets}
    # The query region is a property of the library design, shared by WT and all
    # its variants. A mutation-dependent query leaks the observed answer support.
    parent_sites = {}
    for rows in sheet_rows.values():
        for row in rows:
            resolved = resolve_petersen_variant(row.get("Variant"), parents)
            if resolved is not None:
                parent_name = str(row["Variant"]).split(">")[0].strip()
                parent_sites.setdefault(parent_name, set()).update(resolved[2])
    for sheet in sheets:
        rows = sheet_rows[sheet]
        kept = 0
        for row in rows:
            if row.get("Success") not in (None, "") and finite(row["Success"]) != 1.0:
                rejected["failed optimizer"] = rejected.get("failed optimizer", 0) + 1
                continue
            resolved = resolve_petersen_variant(row.get("Variant"), parents)
            if resolved is None:
                unresolved += 1
                continue
            heavy, light, sites = resolved
            nm = finite(row.get("Kd"))
            if nm is None or nm <= 0:
                rejected["no positive Kd"] = rejected.get("no positive Kd", 0) + 1
                continue
            interval_nm, reason = qualify_interval(nm, row.get("95% CI low"),
                                                   row.get("95% CI high"))
            if interval_nm is None:
                rejected[reason] = rejected.get(reason, 0) + 1
            pkd = pkd_from_log10_nm(math.log10(nm))
            bounds = ((pkd_from_log10_nm(math.log10(interval_nm[1])),
                       pkd_from_log10_nm(math.log10(interval_nm[0])))
                      if interval_nm is not None else (pkd, pkd))
            parent_name = str(row["Variant"]).split(">")[0].strip()
            require(parent_sites.get(parent_name),
                    f"No variable positions were resolved for Petersen parent {parent_name}")
            construct = Construct(heavy=heavy, light=light,
                                  sites=edit_map(sorted(parent_sites[parent_name])),
                                  source=source, record_id=str(row.get("Variant")))
            observations.append(Observation(
                construct=construct, source=source, family=spec.family, kind=KIND_POINT,
                target=str(row.get("Target") or target), assay=f"magmaseq:{sheet}",
                parent=parent_name,
                scale=SCALE_PKD, value=pkd, interval=(pkd, pkd), censoring=CENSOR_NONE,
                replicate=row.get("Replicate"),
                uncertainty=None if interval_nm is None else abs(bounds[1] - bounds[0]) / 3.92,
                raw={"Variant": row.get("Variant"), "Kd_nM": nm,
                     "ci_nM": [finite(row.get("95% CI low")), finite(row.get("95% CI high"))],
                     "mutated_sites": [[chain, int(index)] for chain, index in sites],
                     "barcode": row.get("Barcode"), "parent_antibody": row.get("Ab"),
                     "measurement_id": str(row.get("Barcode") or row.get("Variant")),
                     "replicate": row.get("Replicate"), "number_of_concentrations": row.get("n"),
                     "success": row.get("Success"), "sheet": sheet,
                     "mutation_map_known": True},
                qualification={"raw_units": "KD in nM", "interval_status": reason,
                               "uncertainty_enrolled": interval_nm is not None,
                               "uncertainty_rule": ("The 95% fit confidence interval is uncertainty "
                                                    "about the point estimate, not interval censoring. "
                                                    "Its log10 width / 3.92 approximates a standard error."),
                               "region_rule": "union of sequence-only variable positions for this parent",
                               "sheet": sheet,
                               "no_universal_1000_limit": (
                                   "no numeric value is treated as a censoring limit; the "
                                   "release carries KD values above 1000 nM"),
                               "barcode_is_not_a_sequence": (
                                   "a barcode or a replicate is not a unique construct; the "
                                   "exposure ledger counts them separately")}))
            kept += 1
        per_sheet[sheet] = {"rows": len(rows), "enrolled": kept}
    return {"observations": tuple(observations), "rows": sum(
        block["rows"] for block in per_sheet.values()),
        "missing": sum(rejected.values()), "unalignable": unresolved,
        "unresolved_variants": unresolved,
        "rejected_intervals": dict(sorted(rejected.items())),
        "per_sheet": per_sheet, "sha256": sha256(path),
        "resolved_parents": dict(sorted((name, len(sites))
                                        for name, sites in parent_sites.items())),
        "join_note": ("the number of curated pairs that join to a resolved variant is COMPUTED by "
                      "petersen_join_audit against the curated table, not asserted here. An "
                      "earlier revision carried a '545 of 547' figure from a join that had never "
                      "been run.")}


def petersen_join_audit(primary_observations, curated_observations):
    """How many curated Petersen pairs join to a resolved primary variant. Measured.

    Both counts are computed from the two loaded populations, and the unjoined
    identities are reported. The curated summary is a per-pair summary of the same
    barcode sheets, so a pair that does not join is either an unresolved variant
    specification or a curation difference -- and which it is cannot be decided by
    a sentence in a docstring.
    """
    primary = {record.identity for record in primary_observations}
    curated = {record.identity for record in curated_observations}
    joined = sorted(curated & primary)
    return {"curated_unique_pairs": len(curated), "primary_unique_pairs": len(primary),
            "joined": len(joined), "curated_not_joined": len(curated) - len(joined),
            "primary_not_in_curated": len(primary - curated),
            "examples_not_joined": sorted(curated - primary)[:5],
            "basis": ("exact (heavy, light) identity between the two loaded populations. No "
                      "count here is carried from a previous revision.")}


def load_petersen_original(path, *, sheet, source="petersen2024_ha"):
    """One sheet of a Petersen supplement, with uncertainty qualified per row.

    Returns raw rows plus the interval qualification; it does **not** build
    constructs, because the ``Variant`` field is a mutation specification
    (``parent>VH:mutations|parent>VL:mutations``, 1-based protein indices) and
    resolving it needs the parental sequences from Data 3. The resolver is
    :func:`resolve_petersen_variant`, called only when those parents are available.
    """
    rows = read_xlsx_sheet(path, sheet)
    qualified, rejected = 0, {}
    records = []
    for row in rows:
        interval, reason = qualify_interval(row.get("Kd"), row.get("95% CI low"),
                                            row.get("95% CI high"))
        if interval is None:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            qualified += 1
        records.append({"variant": row.get("Variant"), "kd_nm": finite(row.get("Kd")),
                        "interval_nm": interval, "interval_status": reason,
                        "replicate": row.get("Replicate") or row.get("n"),
                        "barcode": row.get("Ab") or row.get("Barcode"),
                        "success": row.get("Success"), "raw": dict(row)})
    return {"sheet": sheet, "rows": len(records), "records": tuple(records),
            "qualified_intervals": qualified, "rejected_intervals": dict(sorted(rejected.items())),
            "sha256": sha256(path),
            "per_barcode_note": ("a barcode or a replicate is not a unique sequence. The exposure "
                                 "ledger counts constructs and presentations separately."),
            "source": source}


def resolve_petersen_variant(specification, parents):
    """``parent>VH:muts|parent>VL:muts`` -> ``(heavy, light, sites)``, or ``None``.

    Protein indices are 1-based in the release; silent DNA-level changes are
    encoded in the same field and are ignored for the protein construct. Returns
    ``None`` when a named parent is unavailable -- the row is then reported as
    unresolved rather than attached to a guessed background.
    """
    import re
    text = str(specification or "").strip()
    if not text:
        return None
    heavy = light = None
    sites = []
    for chunk in text.split("|"):
        if ">" not in chunk or ":" not in chunk:
            continue
        head, mutations = chunk.split(":", 1)
        parent_name, chain = head.split(">", 1)
        chain = chain.strip().upper()
        sequence = parents.get((parent_name.strip(), chain))
        if sequence is None:
            return None
        buffer = list(sequence)
        for token in re.split(r"[;,\s]+", mutations.strip()):
            if token in ("", "WT"):
                continue
            match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])(?:-[ACGT]+)?", token)
            if match is None:
                return None
            original, position, replacement = match.group(1), int(match.group(2)), match.group(3)
            if not 1 <= position <= len(buffer) or buffer[position - 1] != original:
                return None       # the stated original residue does not match the parent
            buffer[position - 1] = replacement
            if replacement != original:
                sites.append(((HEAVY if chain.startswith("VH") else LIGHT), position - 1))
        if chain.startswith("VH"):
            heavy = "".join(buffer)
        else:
            light = "".join(buffer)
    if heavy is None:
        return None
    return heavy, (light or ""), tuple(sites)


# ---------------------------------------------------------------------------
# CR9114 / CR6261
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CR9114EditSpace:
    """The real 16-site edit space, read from the prepared manifest.

    ``decoded_sequence`` is the structural context, which may be mature. The
    ordered ``allowed_residues`` define genotype 0/1 at every editable position.

    There is no fallback. A loader that invents ``heavy="A"`` when the manifest is
    absent maps all 65,536 genotypes onto one construct, and every count, split
    and contrast computed afterwards describes that single construct while
    looking entirely normal.
    """

    germline: str
    sites: tuple

    @classmethod
    def from_prepared(cls, path):
        import json
        target = Path(path)
        require(target.is_file(),
                f"The CR9114 prepared manifest {target} is missing. The genotype -> sequence map "
                "comes from it; there is no placeholder sequence.")
        document = json.loads(target.read_text(encoding="utf-8"))
        manifest = document.get("manifest", document)
        germline = str(manifest.get("decoded_sequence") or "")
        require(germline, "The prepared manifest carries no decoded_sequence")
        space = manifest.get("edit_space") or document.get("edit_space") or {}
        rows = list(space.get("sites") or ())
        require(rows, "The prepared manifest's edit_space declares no sites")
        sites = []
        for entry in rows:
            index = int(entry["sequence_index"])
            allowed = [str(value).strip().upper() for value in entry["allowed_residues"]]
            require(len(allowed) == 2,
                    f"Site {entry.get('site_id')} declares {len(allowed)} allowed residues; a "
                    "binary genotype bit selects between exactly two")
            require(0 <= index < len(germline),
                    f"Site {entry.get('site_id')} names sequence_index {index} outside the "
                    f"{len(germline)}-residue decoded sequence")
            require(germline[index] in allowed,
                    f"Site {entry.get('site_id')}: the decoded sequence carries "
                    f"{germline[index]!r} at {index}, which is not among {allowed}")
            sites.append({"site_id": entry.get("site_id"), "sequence_index": index,
                          "allowed_residues": tuple(allowed),
                          "germline_residue": germline[index]})
        zero = list(germline)
        for site in sites:
            zero[site["sequence_index"]] = site["allowed_residues"][0]
            site["germline_residue"] = site["allowed_residues"][0]
        return cls(germline="".join(zero), sites=tuple(sites))

    @property
    def length(self):
        return len(self.sites)

    def sequence_for(self, genotype):
        """The heavy chain a 16-bit genotype names. Bit order is site order."""
        bits = [int(value) for value in str(genotype)]
        require(len(bits) == self.length,
                f"A {len(bits)}-bit genotype against a {self.length}-site edit space")
        buffer = list(self.germline)
        for bit, site in zip(bits, self.sites):
            require(bit in (0, 1), f"Genotype bit {bit!r} is not 0 or 1")
            buffer[site["sequence_index"]] = site["allowed_residues"][bit]
        return "".join(buffer)

    def edit_sites(self):
        return edit_map([(HEAVY, int(site["sequence_index"])) for site in self.sites])

    def document(self):
        return {"germline_length": len(self.germline), "sites": self.length,
                "sequence_indices": [int(site["sequence_index"]) for site in self.sites],
                "site_ids": [site["site_id"] for site in self.sites],
                "rule": ("bit i selects the somatic residue at site i. Derived from the prepared "
                         "manifest's decoded_sequence and edit_space.sites; nothing here invents "
                         "a coordinate or a residue.")}


def load_cr9114(path, *, antigen, space=None, floor=CR9114_H3_FLUB_FLOOR,
                allowed_genotypes=None, genotype_splits=None, light=""):
    """H3 or influenza-B replicates, with the H1 columns structurally unreadable.

    ``usecols`` is explicit and a request naming any of
    :data:`CR9114_RESERVED_COLUMNS` raises. The raw release is one CSV with the
    three antigens side by side, so a loader that read everything and filtered
    afterwards would have materialised the reserved partition first.

    The reported pKD floor is LEFT-censoring, not a weak-affinity target: a
    replicate at the floor says "no tighter than this", and a squared error against
    the floor value trains the model to believe a number the assay did not produce.

    ``genotype_splits`` maps each eligible genotype to ``train`` or ``development``
    and is written into every row's ``partition``. Without it every CR9114 row is
    a training row -- including the 11,750 development genotypes -- and the
    "train-only thresholds" and "held-out labels never enter an adaptation cache"
    claims are both false for the largest measured landscape in the campaign.
    """
    import pandas as pd
    require(antigen in ("h3", "fluB"), f"Unsupported antigen partition {antigen!r}")
    spec = SOURCES[f"cr9114_{'h3' if antigen == 'h3' else 'flub'}"]
    replicates = [f"{antigen}_repa", f"{antigen}_repb", f"{antigen}_repc"]
    columns = ["genotype", *replicates, f"{antigen}_mean", f"{antigen}_sem"]
    forbidden = sorted(set(columns) & CR9114_RESERVED_COLUMNS)
    require(not forbidden,
            f"The requested columns {forbidden} are reserved. The H1 partition is never read for "
            "development, teacher queries or replay.")
    frame = pd.read_csv(Path(path), usecols=columns, dtype={"genotype": "string"})
    require(list(frame.columns) == columns or set(frame.columns) == set(columns),
            f"The release no longer carries {columns}")
    require(not (set(frame.columns) & CR9114_RESERVED_COLUMNS),
            "A reserved H1 column reached the loaded frame")
    require(space is not None,
            "load_cr9114 needs a CR9114EditSpace. Without it the genotype cannot be turned into "
            "a sequence, and the earlier fallback -- heavy='A', one edit site at H0 -- collapsed "
            "every genotype onto one construct while every downstream count still looked normal.")
    require(isinstance(space, CR9114EditSpace),
            f"Expected a CR9114EditSpace, got {type(space).__name__}")
    values = frame[replicates].to_numpy(dtype=float)
    require(not np.isinf(values).any(), "Infinite assay replicate")
    counts = np.isfinite(values).sum(axis=1)
    sites = space.edit_sites()
    allowed = None if allowed_genotypes is None else frozenset(
        str(value) for value in allowed_genotypes)
    partitions = {str(key): str(value) for key, value in dict(genotype_splits or {}).items()}
    observations, excluded = [], {"no_replicates": 0, "not_in_allowed_genotypes": 0}
    by_partition = {}
    for position in range(len(frame)):
        genotype = str(frame.genotype.iloc[position])
        if allowed is not None and genotype not in allowed:
            excluded["not_in_allowed_genotypes"] += 1
            continue
        partition = partitions.get(genotype, "train" if partitions else "train")
        by_partition[partition] = by_partition.get(partition, 0) + 1
        row = values[position]
        finite_values = row[np.isfinite(row)]
        if finite_values.size == 0:
            excluded["no_replicates"] += 1
            continue
        heavy = space.sequence_for(genotype)
        construct = Construct(heavy=heavy, light=str(light), sites=sites,
                              source=spec.name, record_id=genotype)
        for replicate_index, value in enumerate(row):
            if not math.isfinite(value):
                continue
            at_floor = bool(value <= float(floor))
            observations.append(Observation(
                construct=construct, source=spec.name, family=FAMILY_AFFINITY,
                kind=KIND_INTERVAL, target=spec.target, assay=spec.name,
                parent="cr9114::germline", scale=SCALE_PKD,
                value=None if at_floor else float(value),
                interval=(None, float(floor)) if at_floor else (float(value), float(value)),
                censoring=CENSOR_LEFT if at_floor else CENSOR_NONE,
                replicate=replicates[replicate_index], uncertainty=None,
                raw={"genotype": genotype, "replicate_value": float(value),
                     "replicates": [None if not math.isfinite(v) else float(v) for v in row],
                     "measurement_id": f"{genotype}:{replicates[replicate_index]}",
                     "partition": partition,
                     "mutated_sites": [[HEAVY, int(site["sequence_index"])]
                                       for bit, site in zip(str(genotype), space.sites)
                                       if bit == "1"],
                     "reported_mean": finite(frame[f"{antigen}_mean"].iloc[position]),
                     "reported_sem": finite(frame[f"{antigen}_sem"].iloc[position])},
                qualification={
                    "floor_pkd": float(floor),
                    "floor_semantics": "Each floor replicate is LEFT-censored in pKD.",
                    "replicate_support": int(finite_values.size),
                    "mean_basis": ("no point aggregation: every finite replicate contributes its "
                                   "own censored or point likelihood; discarding the floor "
                                   "replicates would bias affinity upwards"),
                    "h1_never_read": True}))
    return {"observations": tuple(observations), "rows": int(len(frame)),
            "excluded": dict(excluded), "columns_read": list(columns),
            "edit_space": space.document(),
            "by_partition": dict(sorted(by_partition.items())),
            "partition_gate": ("the whitelist's own split column" if partitions
                               else "ABSENT: every row is labelled train"),
            "genotype_gate": ("the pinned eligibility whitelist" if allowed is not None
                              else "UNRESTRICTED"),
            "reserved_columns_refused": sorted(CR9114_RESERVED_COLUMNS),
            "replicate_counts": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(counts, return_counts=True))},
            "sha256": sha256(path)}


# ---------------------------------------------------------------------------
# paired OAS: reconstruction and prior retention, no outcome at all
# ---------------------------------------------------------------------------

#: The row keys the prepared paired OAS release actually carries. Written out
#: because an earlier revision read ``heavy``/``light``, which are absent, so the
#: loader returned zero rows and the reconstruction task trained on nothing while
#: every count in the inventory looked like a small file.
OAS_HEAVY_KEY = "sequence_heavy"
OAS_LIGHT_KEY = "sequence_light"
OAS_SPLIT_KEY = "split"
OAS_ID_KEY = "pair_id"
#: The prepared directory holds two populations, not two splits: ``oas_all`` and
#: ``oas_paired`` are the same pairs. Globbing both doubles every row.
OAS_MEMBER = "oas_paired.jsonl.gz"
OAS_CACHE_REVISION = 2


def _oas_study(record):
    """``author/run/subject`` -- the study identity a transfer claim needs."""
    metadata = dict(record.get("metadata") or {})
    parts = [str(metadata.get(name) or "") for name in ("author", "run", "subject")]
    return "/".join(part for part in parts if part) or "unknown"


def load_paired_oas(root, *, limit=200000, splits=("train",), cache=None, seed=20260927,
                    member=OAS_MEMBER):
    """Paired VH/VL records for prior retention. No outcome; one file; real keys.

    Four corrections over the earlier reader, each of which changed the result:

    * the row keys are ``sequence_heavy`` / ``sequence_light``. The earlier reader
      asked for ``heavy`` / ``light`` and enrolled **zero** rows, so the
      reconstruction task -- twenty per cent of the declared mixture -- trained on
      an empty pool.
    * ``oas_all.jsonl.gz`` and ``oas_paired.jsonl.gz`` are the same population.
      Globbing the directory reads both and every pair is presented twice.
    * the split is a **row field**. The earlier reader took the first 200,000 rows
      of whichever file sorted first, which is a biased sample of one author's
      run and leaves no held-out repertoire at all. Here each declared split is
      reservoir-sampled independently, so the cap is a uniform sample and the
      held-out split survives it.
    * ``metadata.author/run/subject`` is retained as a study identity, because a
      repertoire transfer claim is a claim about unseen studies.
    """
    import gzip
    import json
    root = Path(root)
    target = root / member if root.is_dir() else root
    require(target.is_file(),
            f"{target} is absent. This loader reads exactly one member, because the prepared "
            f"directory holds two copies of one population and globbing it doubles every row.")
    digest = sha256(target)
    wanted = tuple(str(name) for name in splits)
    if cache is not None:
        cached = _read_oas_cache(cache, digest=digest, limit=int(limit), splits=wanted,
                                 seed=int(seed))
        if cached is not None:
            return cached
    rng = np.random.default_rng([int(seed), 11])
    reservoirs = {name: [] for name in wanted}
    seen = {name: 0 for name in wanted}
    other_splits, rejected, studies = {}, {}, {}
    rows = 0
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        for line in stream:
            rows += 1
            try:
                record = json.loads(line)
            except ValueError:
                rejected["unparseable line"] = rejected.get("unparseable line", 0) + 1
                continue
            split = str(record.get(OAS_SPLIT_KEY) or "")
            if split not in reservoirs:
                other_splits[split] = other_splits.get(split, 0) + 1
                continue
            heavy = str(record.get(OAS_HEAVY_KEY) or "").strip().upper()
            light = str(record.get(OAS_LIGHT_KEY) or "").strip().upper()
            if not heavy or not light:
                rejected["a chain is absent"] = rejected.get("a chain is absent", 0) + 1
                continue
            study = _oas_study(record)
            entry = (str(record.get(OAS_ID_KEY) or rows), split, study, heavy, light)
            seen[split] += 1
            bucket = reservoirs[split]
            if len(bucket) < int(limit):
                bucket.append(entry)
            else:
                position = int(rng.integers(0, seen[split]))
                if position < int(limit):
                    bucket[position] = entry
    kept = [entry for name in wanted for entry in reservoirs[name]]
    require(kept,
            f"{target.name} yielded no paired row for splits {wanted}. Its keys are "
            f"{OAS_HEAVY_KEY}/{OAS_LIGHT_KEY}/{OAS_SPLIT_KEY}; a loader asking for other names "
            "returns nothing and looks like a small file.")
    constructs, by_split = [], {}
    for identifier, split, study, heavy, light in kept:
        try:
            construct = Construct(heavy=heavy, light=light, sites=(), source="paired_oas",
                                  record_id=str(identifier))
        except ValueError:
            rejected["non-canonical residue"] = rejected.get("non-canonical residue", 0) + 1
            continue
        studies[study] = studies.get(study, 0) + 1
        by_split.setdefault(split, []).append(construct)
        constructs.append(construct)
    record = {"constructs": tuple(constructs),
              "by_split": {name: tuple(values) for name, values in sorted(by_split.items())},
              "rows_in_file": rows, "rows_seen_by_split": dict(sorted(seen.items())),
              "other_splits_present": dict(sorted(other_splits.items())),
              "excluded": dict(sorted(rejected.items())),
              "studies": len(studies), "top_studies": dict(sorted(
                  studies.items(), key=lambda item: -item[1])[:10]),
              "limit_per_split": int(limit), "splits": list(wanted), "seed": int(seed),
              "member": member, "sha256": digest,
              "sampling": ("reservoir per split, so the cap is a uniform sample of that split "
                           "rather than its first n rows"),
              "outcome": "none. This source trains reconstruction and prior retention only."}
    if cache is not None:
        _write_oas_cache(cache, record, kept)
    return record


def _oas_cache_paths(cache):
    root = Path(cache) / "paired_oas"
    return root, root / "index.json", root / "rows.tsv"


def _write_oas_cache(cache, record, kept):
    import json
    root, index_path, rows_path = _oas_cache_paths(cache)
    root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, split, study, heavy, light in kept:
            handle.write(f"{identifier}\t{split}\t{study}\t{heavy}\t{light}\n")
    index = {key: value for key, value in record.items()
             if key not in ("constructs", "by_split")}
    index["cache_revision"] = OAS_CACHE_REVISION
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _read_oas_cache(cache, *, digest, limit, splits, seed):
    import json
    root, index_path, rows_path = _oas_cache_paths(cache)
    if not (index_path.is_file() and rows_path.is_file()):
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (index.get("sha256") != digest or int(index.get("cache_revision") or 0)
            != OAS_CACHE_REVISION or int(index.get("limit_per_split") or 0) != int(limit)
            or list(index.get("splits") or ()) != list(splits)
            or int(index.get("seed") or -1) != int(seed)):
        return None
    constructs, by_split = [], {}
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier, split, _study, heavy, light = line.rstrip("\n").split("\t")
            construct = Construct(heavy=heavy, light=light, sites=(), source="paired_oas",
                                  record_id=identifier)
            by_split.setdefault(split, []).append(construct)
            constructs.append(construct)
    record = dict(index)
    record.pop("cache_revision", None)
    record["constructs"] = tuple(constructs)
    record["by_split"] = {name: tuple(values) for name, values in sorted(by_split.items())}
    record["reused_cache"] = True
    return record


# ---------------------------------------------------------------------------
# identity closure across sources
# ---------------------------------------------------------------------------

def observation_closure(groups):
    """Collapse duplicate *observations* across sources; keep genuine replicates.

    Two FLAb releases that publish the same Engelhart row are one observation, not
    two experiments. Two replicates of one construct in one assay are two
    observations of one construct. The key that distinguishes them is
    :attr:`Observation.observation_identity`; the *sequence* identity is reported
    separately, because that is what the split closure needs.
    """
    seen, kept, duplicates = {}, [], []
    for name, observations in sorted(dict(groups).items()):
        for record in observations:
            key = record.observation_identity
            if key in seen:
                duplicates.append({"source": name, "duplicate_of": seen[key],
                                   "identity": record.identity[:32]})
                continue
            seen[key] = name
            kept.append(record)
    sequences = {}
    for record in kept:
        sequences.setdefault(record.identity, []).append(record.source)
    shared = {identity: sorted(set(sources)) for identity, sources in sequences.items()
              if len(set(sources)) > 1}
    return {"observations": tuple(kept), "kept": len(kept), "duplicates": len(duplicates),
            "duplicate_detail": duplicates[:50],
            "unique_constructs": len(sequences),
            "constructs_in_multiple_sources": len(shared),
            "shared_examples": {k: v for k, v in sorted(shared.items())[:20]},
            "rule": ("de-duplication is on (construct, source, assay, target, replicate). A "
                     "re-published identical row is not a new experiment; a genuine replicate "
                     "is not a new construct.")}


def apply_declared_heldouts(observations, *, withheld_parents=(), withheld_studies=(),
                            label=PARTITION_DEVELOPMENT_LABEL):
    """Move whole parents and whole studies into a held-out partition. E15's input.

    A functional-transfer claim is a claim about a background or a study the model
    never trained on, and the only way to have one is to declare it before the
    fit. The declaration is sequence-only -- a parent name and a study name -- so
    it can be frozen alongside the split and before any outcome is read.

    Returns the rewritten observations and the counts, so "unseen parent" is a
    number in the inventory rather than an assumption in a panel.
    """
    parents = frozenset(str(value) for value in withheld_parents)
    studies = frozenset(str(value) for value in withheld_studies)
    out, moved = [], {"parent": 0, "study": 0}
    for record in observations:
        reason = None
        if record.parent in parents:
            reason = "parent"
        elif record.study in studies or record.source in studies:
            reason = "study"
        if reason is None:
            out.append(record)
            continue
        moved[reason] += 1
        raw = dict(record.raw)
        raw["partition"] = str(label)
        raw["withheld_as"] = reason
        out.append(Observation(**{**record.__dict__, "raw": raw}))
    return tuple(out), {"withheld_parents": sorted(parents), "withheld_studies": sorted(studies),
                        "rows_moved": dict(moved), "partition_label": str(label),
                        "rule": ("declared before the fit and sequence-only. A transfer panel "
                                 "whose held-out background was chosen after the fit is a "
                                 "description of the fit.")}


def derive_assay_vocabulary(observations, *, minimum_rows=1,
                            additional_training_counts=None):
    """The frozen assay identities, taken from the ENROLLED training partitions.

    A static list in a config is a different thing from the assays the training
    stream actually presents: the Li releases carry a per-row ``Assay`` column, so
    ``li2023_s1_b`` alone contributes as many identities as that column has
    values. Conditioning on a name the stream never produces trains an embedding
    row on nothing, and producing a name the spec does not know is refused at
    ``ConditionSpec.encode`` -- mid-run.

    Returns the sorted identities plus their row counts, so an identity with one
    supporting row is visible rather than silently equal to one with 300,000.
    """
    counts = {}
    for record in observations:
        if str(record.raw.get("partition") or "train") != "train" or \
                record.qualification.get("quarantined", False):
            continue
        counts[record.assay_key] = counts.get(record.assay_key, 0) + 1
    for assay, count in dict(additional_training_counts or {}).items():
        require(int(count) >= 0, f"Negative training count for assay {assay!r}")
        if count:
            counts[str(assay)] = counts.get(str(assay), 0) + int(count)
    enrolled = sorted(name for name, rows in counts.items() if rows >= int(minimum_rows))
    thin = {name: rows for name, rows in sorted(counts.items()) if rows < int(minimum_rows)}
    return {"assays": tuple(enrolled), "counts": dict(sorted(counts.items())),
            "below_minimum": thin, "minimum_rows": int(minimum_rows),
            "rule": ("derived from the enrolled training observations, not declared in a config. "
                     "The vocabulary is frozen once here and the conditions module refuses any "
                     "identity outside it, so a mid-run unknown assay is impossible rather than "
                     "a runtime failure.")}


def source_inventory(loaded):
    """One table of what every source contributed, including what it could not."""
    rows = []
    for name, block in sorted(dict(loaded).items()):
        observations = block.get("observations") or ()
        spec = SOURCES.get(name)
        rows.append({
            "source": name, "enrolled": bool(spec.enrolled) if spec else None,
            "family": spec.family if spec else None,
            "target": spec.target if spec else None,
            "measurement_kind": spec.kind if spec else None,
            "rows_in_file": block.get("rows"),
            "observations": len(observations),
            "unique_constructs": len({record.identity for record in observations}),
            "missing_outcome": block.get("missing_affinity", block.get("missing")),
            "unalignable": block.get("unalignable"),
            "excluded": block.get("excluded"),
            "sha256": block.get("sha256")})
    return {"sources": rows,
            "totals": {"observations": sum(row["observations"] for row in rows),
                       "unique_constructs": sum(row["unique_constructs"] for row in rows)},
            "note": ("unique_constructs is summed per source and is therefore an upper bound on "
                     "the pooled count; observation_closure reports the pooled figure")}

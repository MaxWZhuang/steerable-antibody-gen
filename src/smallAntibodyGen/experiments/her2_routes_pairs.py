"""Measured one-site pairs for T4: qualification, context identity, conditions.

Pure functions over construct summaries and local contrasts. Nothing here touches a
model, a config file or the flight services, so every rule can be tested on synthetic
summaries and every refusal is a counted reason rather than a silent absence.

These rules are each the correction of something the first T4 pool got wrong:

* **Quantitative means exact, finite and in pKD -- not ``kind == point``.** ``kind`` is a
  source-level declaration. Adams publishes ``KIND_INTERVAL`` rows whose in-window values
  are uncensored degenerate intervals; a ``kind`` gate drops every one of them. The gate
  here reads the replicates: every one must be an uncensored finite point whose bounds,
  declared or implied, are that value, on ``SCALE_PKD``. A construct with one floor
  replicate and two points has an uncensored-replicate *mean* that no assay produced, so
  it is refused and counted, never differenced.
* **A pair carries its own assay, target, family and scale.** A pooled pool is a pool of
  pairs each of which names the measurement it came from. Nothing defaults to HER2 and an
  assay missing from the enrolled condition vocabulary is refused (``assay_not_enrolled``)
  rather than relabelled ``null``: the dropped-assay slot is a trained route for an unseen
  study, not a place to hide a pair whose assay nobody enrolled.
* **The background is the whole masked context.** The first pool keyed backgrounds on
  ``identity()[:24]``, which is the first 24 *heavy-chain residues of the rejected
  endpoint*: orientation-dependent, blind to the light chain, and a single stratum for
  every construct that shares a framework. The digest here covers the complete masked
  heavy and light chains plus assay, target, family, scale and the edit coordinate, and is
  built from the masked context, which both endpoints share by construction, so it cannot
  depend on which endpoint was chosen.
* **A pair's condition comes from the pair.** :func:`pair_condition` derives
  ``(family, bin, assay)`` from the pair's own metadata. An explicit condition that
  disagrees raises; a pair with no metadata raises. There is no HER2 default.
* **Identity and measured target are two digests.** ``pair_digest`` is the structural
  identity -- the masked context and the two unordered alternatives -- and does not move when
  the measurements do. ``target_digest`` is the *measured target*: the ordered endpoint
  values, the winner, the margin and each endpoint's study, source, assay, target and
  replicate identities. A pool whose values or winners changed has a different
  :func:`pair_targets_digest` although every ``pair_digest`` is the same, which is what lets a
  training identity distinguish "same pairs" from "same training targets".
* **A duplicate pair must agree.** Two contrasts that name the same structural pair are
  deduplicated and counted only if their target digests are equal. If they disagree in value,
  winner or provenance -- a measured tie against a measured preference included --
  :func:`build_measured_pairs` raises and names the disagreement; it never keeps whichever
  contrast was enumerated first.
* **T4 pairs are measured endpoints.** :func:`require_pair_metadata` holds an imported pair to
  ``margin == chosen_value - rejected_value`` and re-derives its context, pair and target
  digests from its endpoints. That equality is a property of T4's measured-endpoint pairs only:
  the background-permuted local contrasts elsewhere carry effect bundles whose margin
  intentionally differs from their raw endpoint values, and nothing here is applied to them.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

from . import her2_routes_data as data_lib
from .her2_routes_conditions import BIN_HIGH, FAMILIES, Condition
from .her2_routes_sources import SCALE_PKD
from .her2_runtime import require

#: Bumped when pair eligibility or the pair record's meaning changes. Recorded on every
#: pool, on E4 and in the post-training checkpoint identity, so a pool built under the
#: old semantics cannot be mistaken for this one.
PAIRS_VERSION = 2

#: A character that is not a residue, standing in for every edited coordinate.
MASK_CHARACTER = "#"

#: The keys a population filter may name. Anything else is refused, not ignored.
POPULATION_KEYS = ("assay", "target", "family")

#: Two measured values closer than this are a tie: the pair states no preference.
TIE_TOLERANCE = 1e-12

#: The two contrast classes that are a choice between two residues at one coordinate.
PAIR_CONTRAST_KINDS = (data_lib.CONTRAST_SINGLE, data_lib.CONTRAST_ALTERNATIVES)


# ---------------------------------------------------------------------------
# qualification
# ---------------------------------------------------------------------------

def quantitative_status(summary):
    """``(eligible, reason)`` for using a summary as an exact pKD measurement.

    Eligible only when every replicate is an exact finite point, the scale is pKD and
    the summary names its assay, target and family. The reason is one of
    ``exact_pkd``, ``scale_not_pkd``, ``missing_metadata`` or the summary's own
    :attr:`~her2_routes_data.ConstructSummary.exact_status` (``mixed_censoring``,
    ``censored``, ``nonfinite``, ``nondegenerate_bounds``, ``ordinal``, ``empty``).
    ``summary.kind`` is deliberately not consulted.
    """
    if getattr(summary, "scale", "") != SCALE_PKD:
        return False, "scale_not_pkd"
    status = summary.exact_status
    if status != "exact":
        return False, status
    for name in ("assay", "target", "family"):
        if not str(getattr(summary, name, "") or ""):
            return False, "missing_metadata"
    value = summary.exact_value
    if value is None or not math.isfinite(float(value)):
        return False, "nonfinite"
    return True, "exact_pkd"


# ---------------------------------------------------------------------------
# context identity
# ---------------------------------------------------------------------------

def masked_context(construct, sites):
    """``(heavy, light)`` with every edited coordinate replaced by :data:`MASK_CHARACTER`."""
    heavy, light = list(construct.heavy), list(construct.light)
    for chain, index in sites:
        chain_residues = heavy if chain == data_lib.HEAVY_CHAIN else light
        require(0 <= int(index) < len(chain_residues),
                f"Edit site {(chain, index)} is outside the {len(chain_residues)}-residue "
                f"{chain} chain")
        chain_residues[int(index)] = MASK_CHARACTER
    return "".join(heavy), "".join(light)


def _digest(document):
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def background_digest(*, heavy, light, assay, target, family, scale, sites):
    """SHA-256 of the complete masked context plus the measurement it was made under.

    ``heavy`` and ``light`` are the *masked* chains in full. Two pairs share a digest only
    if their whole visible context, their assay, target, family and scale and their edit
    coordinate agree -- a stratum of one identical masked input, not of one framework.
    """
    return _digest({"heavy": str(heavy), "light": str(light), "assay": str(assay),
                    "target": str(target), "family": str(family), "scale": str(scale),
                    "sites": sorted([str(chain), int(index)] for chain, index in sites)})


def pair_digest(*, background, chosen_residues, rejected_residues):
    """Identity of the UNORDERED pair: the context and the two alternatives, sorted."""
    return _digest({"background": str(background),
                    "alternatives": sorted([str(chosen_residues), str(rejected_residues)])})


def pair_pool_digest(pairs):
    """One STRUCTURAL digest over a pool, independent of enumeration order.

    It says which pairs are in the pool, not what was measured for them: change a value or
    flip a winner and it does not move. :func:`pair_targets_digest` is the one that does.
    """
    return _digest(sorted(str(pair["pair_digest"]) for pair in pairs))


def endpoint_record(row, sites):
    """The measured evidence one endpoint contributes to a pair's target, as plain data.

    The residues at the edited coordinate, the endpoint's exact value, the identities of the
    measurement (study, source, assay, target, parent) and every replicate's own
    ``(measurement_id, replicate, value)``. Two endpoints with the same residues and the same
    mean but different replicates or a different study are different evidence.
    """
    return {"residues": str(row.construct.residues(sites)), "value": float(row.exact_value),
            "study": str(row.study), "source": str(row.source), "assay": str(row.assay),
            "target": str(row.target), "parent": str(row.parent),
            "replicates": sorted([str(item.measurement_id), str(item.replicate),
                                  float(item.value)] for item in row.measurements)}


def target_digest(*, background, chosen, rejected, sites, tied=False):
    """SHA-256 of a pair's MEASURED TARGET, separate from its structural identity.

    Covers the context ``background`` digest, the ordered ``(chosen, rejected)`` endpoint
    records, the winner, the margin and the pair's assay, target, family and scale. ``chosen``
    is whichever endpoint measured higher, so the digest is as orientation independent as the
    pair digest is -- but unlike it, it changes when a value, a winner or a measurement
    identity changes. A ``tied`` evidence record (no winner) orders its endpoints by residues,
    so a measured tie can be compared with a measured preference for the same structural pair.
    """
    first, second = endpoint_record(chosen, sites), endpoint_record(rejected, sites)
    return _digest({"background": str(background), "tied": bool(tied),
                    "winner": None if tied else first["residues"],
                    "margin": first["value"] - second["value"],
                    "chosen": first, "rejected": second,
                    "assay": str(chosen.assay), "target": str(chosen.target),
                    "family": str(chosen.family), "scale": str(chosen.scale)})


def pair_targets_digest(pairs):
    """One digest over a pool's ``(pair_digest, target_digest)`` rows, independent of order.

    This is what a training identity binds: the same structural pool with a changed measured
    value, winner or provenance has the same :func:`pair_pool_digest` and a different one here.
    """
    return _digest(sorted([str(pair["pair_digest"]), str(pair["target_digest"])]
                          for pair in pairs))


# ---------------------------------------------------------------------------
# building the pool
# ---------------------------------------------------------------------------

def build_measured_pairs(contrasts, *, violates, enrolled_assays):
    """Qualified one-site pairs from local contrasts, with every refusal counted.

    ``violates(summary) -> bool`` is the exclusion-closure check at its radius; it is
    applied to **both** endpoints, because excluding only the chosen one would leave a
    reserved neighbour in the pool as the rejected half of every pair it appears in.
    ``enrolled_assays`` is the set of assay identities the condition table encodes.

    Both contrast classes survive: a measured mutant against its own background
    (``single_vs_background``) is a choice between two residues at one masked coordinate
    exactly as two competing substitutions are (``same_position_alternatives``). The
    chosen endpoint is always the one with the higher measured pKD, so no pair is
    enrolled as a preference for the background merely for being the background. Ties
    state no preference and are counted, not enrolled.
    """
    enrolled = frozenset(str(name) for name in enrolled_assays)
    require(enrolled, "build_measured_pairs needs the enrolled assay identities; an empty set "
                      "would refuse every pair, and omitting it would let any assay through")
    excluded, pairs, seen = {}, [], {}

    def refuse(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    seen_contrasts = 0
    for contrast in contrasts:
        seen_contrasts += 1
        if contrast.kind not in PAIR_CONTRAST_KINDS:
            refuse("not_a_pair_contrast")
            continue
        if len(contrast.sites) != 1:
            refuse("not_one_site")
            continue
        left, right = contrast.left, contrast.right
        reason = None
        for endpoint in (left, right):
            eligible, why = quantitative_status(endpoint)
            if not eligible:
                reason = why
                break
        if reason is not None:
            refuse(reason)
            continue
        metadata = {(row.assay, row.target, row.family, row.scale) for row in (left, right)}
        if len(metadata) != 1 or next(iter(metadata))[:2] != (contrast.assay, contrast.target):
            refuse("metadata_mismatch")
            continue
        assay, target, family, scale = next(iter(metadata))
        if not (left.is_training and right.is_training):
            refuse("not_training_partition")
            continue
        if violates(left) or violates(right):
            refuse("closure")
            continue
        if assay not in enrolled:
            refuse("assay_not_enrolled")
            continue
        sites = tuple(contrast.sites)
        context_left = masked_context(left.construct, sites)
        context_right = masked_context(right.construct, sites)
        if context_left != context_right:
            refuse("context_mismatch")
            continue
        residue_left = left.construct.residues(sites)
        residue_right = right.construct.residues(sites)
        if residue_left == residue_right:
            refuse("no_substitution")
            continue
        value_left, value_right = float(left.exact_value), float(right.exact_value)
        tied = abs(value_left - value_right) <= TIE_TOLERANCE
        if tied:
            # No winner. Order by residues so the evidence record cannot depend on orientation.
            chosen, rejected = sorted((left, right), key=lambda row: row.construct.residues(sites))
        else:
            chosen, rejected = (left, right) if value_left > value_right else (right, left)
        chosen_value, rejected_value = float(chosen.exact_value), float(rejected.exact_value)
        digest = background_digest(heavy=context_left[0], light=context_left[1], assay=assay,
                                   target=target, family=family, scale=scale, sites=sites)
        identity = pair_digest(background=digest, chosen_residues=chosen.construct.residues(sites),
                               rejected_residues=rejected.construct.residues(sites))
        evidence = target_digest(background=digest, chosen=chosen, rejected=rejected, sites=sites,
                                 tied=tied)
        site = sites[0]
        pair = None if tied else {
            "id": f"pair:{identity[:16]}", "pair_digest": identity, "target_digest": evidence,
            "version": PAIRS_VERSION, "measured": True, "source": chosen.source,
            "assay": assay, "target": target, "family": family, "scale": scale,
            "contrast_kind": contrast.kind,
            "chosen": chosen, "rejected": rejected, "sites": sites,
            "chosen_value": chosen_value, "rejected_value": rejected_value,
            "margin": chosen_value - rejected_value,
            "substitution": f"{site[0]}{site[1]}:{rejected.construct.residue_at(site)}->"
                            f"{chosen.construct.residue_at(site)}",
            "background": digest, "background_digest": digest}
        known = seen.get(identity)
        if known is not None:
            # The same structural pair reached again. Identical evidence is a counted
            # duplicate; different evidence is recorded and refused once the scan is done,
            # so the refusal and its message do not depend on which contrast came first.
            known["variants"].setdefault(evidence, _variant(chosen, rejected, sites, tied=tied))
            if evidence == known["evidence"]:
                if tied:
                    refuse("tie")
                else:
                    refuse("duplicate")
                    # Reached through both contrast classes: keep the representative that
                    # does not depend on enumeration order.
                    if contrast.kind < pairs[known["index"]]["contrast_kind"]:
                        pairs[known["index"]] = pair
            continue
        seen[identity] = {"evidence": evidence, "index": None if tied else len(pairs),
                          "variants": {evidence: _variant(chosen, rejected, sites, tied=tied)}}
        if tied:
            refuse("tie")
            continue
        pairs.append(pair)
    conflicts = {identity: known["variants"] for identity, known in seen.items()
                 if len(known["variants"]) > 1}
    if conflicts:
        raise ValueError(_conflict_message(conflicts))
    by_kind = {}
    for pair in pairs:
        by_kind[pair["contrast_kind"]] = by_kind.get(pair["contrast_kind"], 0) + 1
    return {"pairs": tuple(pairs), "version": PAIRS_VERSION,
            "counts": {"contrasts_seen": seen_contrasts, "eligible": len(pairs),
                       "excluded": dict(sorted(excluded.items())),
                       "ties": int(excluded.get("tie", 0)),
                       "by_contrast_kind": dict(sorted(by_kind.items())),
                       "by_assay": assay_exposure(pairs),
                       "pool_sha256": pair_pool_digest(pairs) if pairs else None,
                       "targets_sha256": pair_targets_digest(pairs) if pairs else None},
            "rule": ("exact finite uncensored pKD replicates only; both contrast classes; ties and "
                     "every other refusal counted; the background is the full masked-context "
                     "digest; identical duplicates are deduplicated and counted, conflicting "
                     "ones refuse the pool. summary.kind is not consulted.")}


def _variant(chosen, rejected, sites, *, tied):
    """A short description of one measured version of a structural pair, for refusals."""
    return {"tied": bool(tied),
            "winner": None if tied else str(chosen.construct.residues(sites)),
            "values": [float(chosen.exact_value), float(rejected.exact_value)],
            "studies": [str(chosen.study), str(rejected.study)]}


def _conflict_message(conflicts):
    """Name the conflicting pairs, independent of the order the contrasts were enumerated in."""
    def describe(item):
        what = "a measured tie" if item["tied"] else f"a preference for {item['winner']!r}"
        return f"{what} at values {item['values']} (studies {item['studies']})"
    shown = []
    for identity in sorted(conflicts)[:3]:
        versions = [describe(conflicts[identity][key]) for key in sorted(conflicts[identity])]
        shown.append(f"pair:{identity[:16]} is " + " and also ".join(versions))
    return (f"{len(conflicts)} structural pair(s) are measured in conflicting ways by different "
            f"contrasts: {'; '.join(shown)}. Which value, winner or provenance is the target "
            "cannot be decided by enumeration order, so the pool is refused rather than keeping "
            "the first one seen. Resolve the duplicated measurements upstream.")


#: What every pair handed to the T4 stage must carry. Nothing here is defaulted.
REQUIRED_PAIR_FIELDS = ("id", "pair_digest", "target_digest", "assay", "target", "family", "scale",
                        "chosen", "rejected", "sites", "chosen_value", "rejected_value", "margin",
                        "substitution", "background")

#: Relative and absolute slack for comparing two floats that should be the same measured number.
VALUE_TOLERANCE = 1e-9


def _finite_number(pair, key):
    """``pair[key]`` as a finite float; a bool, a string or NaN is refused, not coerced."""
    value = pair[key]
    require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"pair {pair.get('id')!r}: {key} is {value!r}; it must be a finite number")
    return float(value)


def require_pair_metadata(pairs):
    """Refuse a T4 pair that does not state, and cannot back up, the measurement it came from.

    Returns the count. A hand-built or legacy pair that omits its assay, target, family or
    scale is not completed with HER2 values here: it is rejected, because a default is exactly
    how a non-HER2 pair came to be scored under the HER2 condition. Beyond presence, an imported
    pair is held to what a pair from :func:`build_measured_pairs` satisfies by construction:

    * ``margin`` is a finite number equal to ``chosen_value - rejected_value`` -- T4's pairs are
      MEASURED endpoints, so the margin the likelihood and log-odds arms fit is exactly the
      measured difference. (The background-permuted local contrasts elsewhere carry effect
      bundles whose margin intentionally differs from their raw endpoint values; this rule is
      T4's and is not applied to them.)
    * each endpoint is a training-partition construct summary that is an exact finite pKD
      measurement, whose exact value is the pair's stated value, and whose assay, target, family
      and scale are the pair's own;
    * the two endpoints share one masked context, differ at the edited coordinate, and the
      pair's ``background``, ``pair_digest`` and ``target_digest`` are the digests those
      endpoints produce -- a digest copied from another pair, or left over from different
      measurements, is refused.
    """
    pairs = tuple(pairs)
    for pair in pairs:
        missing = [key for key in REQUIRED_PAIR_FIELDS
                   if pair.get(key) is None or (isinstance(pair.get(key), str) and not pair[key])]
        require(not missing,
                f"pair {pair.get('id')!r} lacks {missing}. Pairs carry their own assay, target, "
                "family, scale, context digest and measured-target digest; none of them is "
                "defaulted.")
        require(pair["scale"] == SCALE_PKD,
                f"pair {pair['id']!r} is on scale {pair['scale']!r}; T4 pairs are exact pKD "
                "measurements only.")
        require(pair["family"] in FAMILIES,
                f"pair {pair['id']!r} has unknown family {pair['family']!r}")
        require(len(tuple(pair["sites"])) == 1,
                f"pair {pair['id']!r} names {len(tuple(pair['sites']))} edit sites; T4 pairs are "
                "one-site conditionals")
        high = _finite_number(pair, "chosen_value")
        low = _finite_number(pair, "rejected_value")
        margin = _finite_number(pair, "margin")
        require(high > low,
                f"pair {pair['id']!r}: chosen {high} and rejected {low} must be finite with the "
                "chosen strictly higher; a tie states no preference")
        require(math.isclose(margin, high - low, rel_tol=VALUE_TOLERANCE, abs_tol=VALUE_TOLERANCE),
                f"pair {pair['id']!r}: margin {margin} is not chosen_value - rejected_value = "
                f"{high - low}. T4 pairs use the measured endpoints, so the margin IS their "
                "difference.")
        _require_pair_content(pair, high, low)
    return len(pairs)


def _require_pair_content(pair, high, low):
    """The endpoints behind a pair must produce the pair's values, context and digests."""
    label = f"pair {pair['id']!r}"
    sites = tuple(tuple(site) for site in pair["sites"])
    ends = (("chosen", pair["chosen"], high), ("rejected", pair["rejected"], low))
    for name, row, stated in ends:
        require(hasattr(row, "exact_status") and hasattr(row, "construct"),
                f"{label}: the {name} endpoint is not a measured construct summary, so its "
                "value, assay and context cannot be checked.")
        eligible, why = quantitative_status(row)
        require(eligible,
                f"{label}: the {name} endpoint is not an exact pKD measurement ({why}).")
        require(bool(getattr(row, "is_training", False)),
                f"{label}: the {name} endpoint is not a training-partition row.")
        require(math.isclose(float(row.exact_value), stated, rel_tol=VALUE_TOLERANCE,
                             abs_tol=VALUE_TOLERANCE),
                f"{label}: {name}_value {stated} is not the endpoint's exact measured value "
                f"{float(row.exact_value)}.")
        for field_name in ("assay", "target", "family", "scale"):
            require(str(getattr(row, field_name)) == str(pair[field_name]),
                    f"{label}: the {name} endpoint was measured as {field_name} "
                    f"{getattr(row, field_name)!r} but the pair states {pair[field_name]!r}.")
    chosen, rejected = pair["chosen"], pair["rejected"]
    heavy, light = masked_context(chosen.construct, sites)
    require(masked_context(rejected.construct, sites) == (heavy, light),
            f"{label}: the endpoints differ outside the edited coordinate, so they do not share "
            "one masked context.")
    require(chosen.construct.residues(sites) != rejected.construct.residues(sites),
            f"{label}: both endpoints carry the same residues at the edited coordinate.")
    background = background_digest(heavy=heavy, light=light, assay=pair["assay"],
                                   target=pair["target"], family=pair["family"],
                                   scale=pair["scale"], sites=sites)
    require(pair["background"] == background,
            f"{label}: its context digest is not the digest of its endpoints' masked context "
            "and measurement.")
    if pair.get("background_digest") is not None:
        require(pair["background_digest"] == background,
                f"{label}: background_digest disagrees with the context digest.")
    identity = pair_digest(background=background,
                           chosen_residues=chosen.construct.residues(sites),
                           rejected_residues=rejected.construct.residues(sites))
    require(pair["pair_digest"] == identity,
            f"{label}: its pair_digest is not the identity of its endpoints.")
    require(pair["target_digest"] == target_digest(background=background, chosen=chosen,
                                                   rejected=rejected, sites=sites),
            f"{label}: its target_digest is not the digest of its endpoints' measured values, "
            "winner and measurement identities.")


def assay_exposure(pairs):
    """``{assay: {pairs, backgrounds, rows}}`` for a pool or a set of drawn pairs."""
    out = {}
    for pair in pairs:
        block = out.setdefault(str(pair.get("assay")), {"pairs": 0, "backgrounds": set()})
        block["pairs"] += 1
        block["backgrounds"].add(pair.get("background"))
    return {assay: {"pairs": block["pairs"], "backgrounds": len(block["backgrounds"]),
                    "rows": 2 * block["pairs"]}
            for assay, block in sorted(out.items())}


# ---------------------------------------------------------------------------
# populations and samples
# ---------------------------------------------------------------------------

def select_population(pairs, population=None):
    """Restrict a pool to an EXPLICIT population, and count what fell outside it.

    ``None`` is the whole pool -- there is no implicit HER2 restriction. A population is a
    mapping over :data:`POPULATION_KEYS`; an unknown key is refused so a misspelt filter
    cannot quietly select everything.
    """
    pairs = tuple(pairs)
    if population is None:
        return {"pairs": pairs, "population": None, "available": len(pairs),
                "selected": len(pairs), "outside_population": 0}
    block = dict(population)
    unknown = sorted(set(block) - set(POPULATION_KEYS))
    require(not unknown, f"Unknown population key(s) {unknown}; expected a subset of "
                         f"{list(POPULATION_KEYS)}")
    require(block, "An empty population is not a filter; pass None for the whole pool")
    for key, value in block.items():
        require(isinstance(value, str) and value, f"population[{key!r}] must be a non-empty string")
    kept = tuple(pair for pair in pairs
                 if all(pair.get(key) == value for key, value in block.items()))
    return {"pairs": kept, "population": dict(sorted(block.items())), "available": len(pairs),
            "selected": len(kept), "outside_population": len(pairs) - len(kept)}


def sample_pairs(pairs, count):
    """A deterministic sample of ``count`` pairs, independent of enumeration order.

    Ordered by each pair's content digest and truncated: a content-addressed subsample, so
    the same pool yields the same pairs however the contrasts were enumerated, and no
    seed is needed to reproduce it.
    """
    pairs = tuple(pairs)
    for pair in pairs:
        require("pair_digest" in pair, f"pair {pair.get('id')!r} has no pair_digest; the "
                                       "sample is content-addressed")
    ordered = sorted(pairs, key=lambda pair: str(pair["pair_digest"]))
    chosen = tuple(ordered[:max(0, int(count))])
    return chosen, {"requested": int(count), "available": len(pairs), "drawn": len(chosen),
                    "rule": "ascending pair_digest, truncated; content-addressed, seedless"}


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

def pair_condition(pair, requested=None, *, bin=BIN_HIGH):
    """The :class:`Condition` a pair is scored under, derived from the pair itself.

    ``(family, bin, assay)`` come from the pair's metadata. ``requested`` may be an explicit
    :class:`Condition`, accepted only if its family and assay agree with the pair's own;
    anything else -- a named mode such as ``"prior_only"`` included -- is refused as
    unsupported rather than guessed at. A pair missing its assay or family raises: this
    function never falls back to HER2, to ``null`` or to a config value.
    """
    missing = [key for key in ("assay", "family") if not pair.get(key)]
    require(not missing,
            f"pair {pair.get('id')!r} carries no {missing}. A condition is derived from the "
            "pair's own measurement; it is never defaulted to HER2 or to the null assay.")
    derived = Condition(family=pair["family"], bin=bin, assay=pair["assay"])
    if requested is None:
        return derived
    require(isinstance(requested, Condition),
            f"Unsupported pair condition {requested!r}: only an explicit Condition is accepted. "
            "Named modes such as 'prior_only' are not implemented for T4.")
    require(requested.family == derived.family and requested.assay == derived.assay,
            f"The requested condition ({requested.family}, {requested.assay}) is incompatible "
            f"with pair {pair.get('id')!r}, which was measured as ({derived.family}, "
            f"{derived.assay}). A pair is scored under its own assay and family.")
    return requested

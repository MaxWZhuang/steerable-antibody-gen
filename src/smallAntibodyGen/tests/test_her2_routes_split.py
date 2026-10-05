"""I15 and the power arithmetic: the hard split, its closure, the reserved identities.

The closure test is the load-bearing one. A reserved identity must be refused by
**every** declared consumer, not only by the training rows, and an empty audit must not
read as a passing audit.
"""
from __future__ import annotations

import pytest

from smallAntibodyGen.experiments import her2_routes_split as split_lib


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def windows(count, *, seed=0):
    """``count`` distinct random 10-mers.

    Random draws over twenty letters sit about nine substitutions apart, which is the
    regime the real population is in: a fixture whose windows vary at only two or three
    positions makes every separation unsatisfiable and every test pass vacuously.
    """
    import random
    rng = random.Random(seed)
    out = set()
    while len(out) < count:
        out.add("".join(rng.choice(ALPHABET) for _ in range(10)))
    return sorted(out)


def near(window, distance=1):
    """A window at exactly ``distance`` substitutions from ``window``."""
    buffer = list(window)
    for position in range(int(distance)):
        buffer[position] = next(residue for residue in ALPHABET if residue != window[position])
    return "".join(buffer)


SETTINGS = {"seed": 20260927, "novel_separation": 3, "purge_radius": 2,
            "test_target": 10, "development_target": 10, "batch_size": 5,
            "batch_separation": 3, "minimum_reserved": 10, "assumed_rho": 0.5,
            "ambition_target": 0.05}


# --- power ----------------------------------------------------------------

def test_power_arithmetic_separates_the_standard_error_from_the_half_width():
    block = split_lib.spearman_half_width(0.5, 100)
    assert block["standard_error"] == pytest.approx(0.0754, abs=1e-3)
    assert block["half_width"] == pytest.approx(0.1478, abs=1e-3)
    # 0.0421 is the STANDARD ERROR at n = 322. The half-width is 1.96 times it, and
    # reading one as the other is what made an earlier amendment enlarge the panel.
    wider = split_lib.spearman_half_width(0.5, 322)
    assert wider["standard_error"] == pytest.approx(0.0419, abs=1e-3)
    assert wider["half_width"] == pytest.approx(0.0820, abs=1e-3)
    assert wider["half_width"] == pytest.approx(1.96 * wider["standard_error"])


def test_power_note_says_no_at_every_size_this_population_supports():
    note = split_lib.power_note(target=0.05, rho=0.5, sizes=[100, 152, 322])
    by_size = {row["n"]: row for row in note["rows"]}
    assert by_size[100]["resolvable"] is False
    assert by_size[152]["resolvable"] is False
    # 322 was the size the withdrawn amendment claimed resolved the target. It does
    # not: the half-width there is 0.082, still above +0.05.
    assert by_size[322]["resolvable"] is False
    assert note["any_size_resolvable"] is False
    assert "PAIRED" in note["primary_endpoint"]
    assert "does NOT bound it" in note["primary_endpoint"]
    assert "radius-2 purge" in note["retention"]


# --- the batched selection ------------------------------------------------

def test_selection_is_seed_dependent_and_order_independent():
    pool = windows(60)
    first = split_lib.select_batched(pool, target=10, batch_size=5, seed=1, separation=3)
    reversed_pool = split_lib.select_batched(list(reversed(pool)), target=10, batch_size=5,
                                            seed=1, separation=3)
    assert first.selected == reversed_pool.selected
    other = split_lib.select_batched(pool, target=10, batch_size=5, seed=2, separation=3)
    assert set(first.selected) != set(other.selected) or first.selected != other.selected


def test_batch_diameters_are_measured_and_reported():
    result = split_lib.select_batched(windows(60), target=10, batch_size=5, seed=1,
                                      separation=3)
    document = result.document()
    assert len(document["batch_diameters"]) == len(result.batches)
    assert document["max_batch_diameter"] is not None
    assert "not a biological family" in document["diameter_caveat"]


def test_selection_reports_exhaustion_rather_than_padding():
    result = split_lib.select_batched(windows(4), target=100, batch_size=5, seed=1,
                                      separation=3)
    assert result.exhausted
    assert len(result.selected) <= 4


def test_every_member_clears_the_separation_from_every_earlier_member():
    """Not just the seeds. Separating seeds alone lets batch three touch batch one.

    This is the property the earlier implementation did not have: it filtered the
    batch CENTRE against earlier centres and then took unfiltered nearest
    neighbours, so a later batch's rim could sit one substitution from an earlier
    batch's rim.
    """
    pool = windows(200, seed=7)
    result = split_lib.select_batches(pool, batches=4, batch_size=5, seed=3, separation=3)
    assert len(result.batches) == 4
    for index, batch in enumerate(result.batches):
        earlier = [member for block in result.batches[:index] for member in block]
        for member in batch:
            for other in earlier:
                assert split_lib.window_hamming(member, other) >= 3


def test_batch_roles_put_the_test_batches_first():
    pool = windows(200, seed=7)
    result = split_lib.select_batches(pool, batches=4, batch_size=5, seed=3, separation=3,
                                      roles=[split_lib.RESERVED] * 2
                                      + [split_lib.DEVELOPMENT] * 2)
    assert list(result.roles) == [split_lib.RESERVED, split_lib.RESERVED,
                                  split_lib.DEVELOPMENT, split_lib.DEVELOPMENT]


def test_vectorised_library_distance_agrees_with_the_naive_loop():
    """The fast path and the obvious path must agree, or the fast path is a guess."""
    pool = windows(60, seed=11)
    library = windows(40, seed=12)
    fast = split_lib.min_distance_to_library(pool, library, max_distance=2)
    for window, distance in zip(pool, fast):
        naive = min(split_lib.window_hamming(window, other) for other in library)
        assert int(distance) == (naive if naive <= 2 else -1)
    # An exact match is distance zero, not "not found".
    assert int(split_lib.min_distance_to_library([library[0]], library)[0]) == 0
    assert int(split_lib.min_distance_to_library([near(library[0], 2)], library)[0]) == 2


# --- the split ------------------------------------------------------------

def test_hard_split_delivers_exactly_the_declared_panel_sizes():
    pool = windows(300, seed=5)
    old = ["AAAAAAAAAA", "WWWWWWWWWW"]
    split = split_lib.build_hard_split(candidates=pool, old_library=old, settings=SETTINGS)
    # EXACTLY the declared sizes, in batch_size-sized batches, test first.
    assert split.counts["reserved"] == SETTINGS["test_target"]
    assert split.counts["development"] == SETTINGS["development_target"]
    assert split.counts["selection_complete"]
    assert split.counts["reserved_meets_minimum"]
    assert [len(batch) for batch in split.selection.batches] == [5, 5, 5, 5]
    assert list(split.selection.roles)[:2] == [split_lib.RESERVED, split_lib.RESERVED]
    for window in split.development:
        assert min(split_lib.window_hamming(window, other) for other in old) >= 3
    for window in split.reserved:
        assert min(split_lib.window_hamming(window, other)
                   for other in split.development) >= SETTINGS["batch_separation"]
    assert not set(split.development) & set(split.reserved)
    # The retention the panel costs is measured, not assumed, and it accounts for
    # every candidate: held out, purged, or trainable.
    assert split.counts["retained_for_training"] > 0
    assert (split.counts["retained_for_training"] + split.counts["purged_within_radius"]
            + 20 == split.counts["candidates"])


def test_novelty_gates_the_PANEL_and_does_not_remove_anything_from_training():
    """Two different rules, and conflating them costs most of the adaptation data.

    A candidate one substitution from the old library is not a hard panel
    candidate -- a parent has effectively seen it. It is still perfectly good
    training data, and excluding it would discard the majority of the population
    for no protection: the protection is the radius purge around the held-out
    cores, which is applied to the whole pool.
    """
    pool = windows(300, seed=5)
    intruder = near(pool[0], distance=1)
    split = split_lib.build_hard_split(candidates=pool + [intruder],
                                       old_library=[pool[0]], settings=SETTINGS)
    # Neither the exact match nor its one-substitution neighbour is a panel candidate.
    assert intruder not in split.reserved and intruder not in split.development
    assert pool[0] not in split.reserved and pool[0] not in split.development
    assert split.counts["not_panel_eligible"] == 2
    assert split.counts["panel_eligible"] == 299
    # Both are still trainable, unless the radius purge catches them.
    assert split.assignment[intruder] in (split_lib.TRAIN, split_lib.EXCLUDED)
    assert split.counts["not_panel_eligible_retained_for_training"] >= 1
    assert "training retention is the whole candidate pool" in split.counts["eligibility_rule"]


def test_purged_candidates_are_excluded_from_training_not_moved_into_a_panel():
    """The heldouts are frozen FIRST and the intruder is tested against them.

    The earlier version of this test added the intruder to the candidate pool and
    then assumed the selection was unchanged. It is not: the pool is part of the
    seeded selection's input, so a new candidate can change which cores are chosen
    and the test was asserting a coincidence.
    """
    pool = windows(300, seed=5)
    frozen = split_lib.build_hard_split(candidates=pool, old_library=["WWWWWWWWWW"],
                                        settings=SETTINGS)
    held_out = list(frozen.reserved) + list(frozen.development)
    intruder = near(held_out[0], distance=1)
    assert intruder not in pool
    distances = split_lib.min_distance_to_library([intruder], held_out, max_distance=2)
    assert int(distances[0]) == 1
    # And the closure built from the same frozen panel refuses it for every consumer.
    closure = split_lib.ExclusionClosure.build(
        reserved_windows=frozen.reserved, development_windows=frozen.development,
        radius=SETTINGS["purge_radius"])
    for consumer in split_lib.CONSUMERS:
        with pytest.raises(ValueError, match="inside the declared radius"):
            closure.require_allowed(consumer=consumer, window=intruder)


def test_split_certificate_reports_violations_and_power():
    split = split_lib.build_hard_split(candidates=windows(300, seed=5),
                                       old_library=["WWWWWWWWWW"], settings=SETTINGS)
    certificate = split_lib.split_certificate(split, rho=0.5, target=0.05)
    assert certificate["zero_violations"]
    assert certificate["violations"] == []
    assert certificate["power"]["rows"]
    assert certificate["power"]["any_size_resolvable"] is False
    assert "unopened BY US" in certificate["reserved_means"]


def test_a_certificate_reports_a_shortfall_rather_than_passing_it():
    thin = dict(SETTINGS, test_target=50, development_target=50, batch_size=25)
    split = split_lib.build_hard_split(candidates=windows(40, seed=9),
                                       old_library=["WWWWWWWWWW"], settings=thin)
    certificate = split_lib.split_certificate(split)
    assert not split.counts["selection_complete"]
    assert not certificate["zero_violations"]
    assert any("could not fill" in str(entry.get("problem"))
               for entry in certificate["violations"])


def test_an_impossible_separation_fails_rather_than_being_relaxed():
    with pytest.raises(ValueError, match="does not exist under these settings"):
        split_lib.build_hard_split(candidates=["AAAAAAAAAA"], old_library=["AAAAAAAAAA"],
                                    settings=SETTINGS)


def test_a_partial_batch_is_refused_rather_than_run():
    with pytest.raises(ValueError, match="a partial batch is a different algorithm"):
        split_lib.build_hard_split(candidates=windows(300, seed=5),
                                   old_library=["WWWWWWWWWW"],
                                   settings=dict(SETTINGS, test_target=7))


def test_freeze_refuses_to_run_after_outcomes_were_opened():
    split = split_lib.build_hard_split(candidates=windows(300, seed=5),
                                       old_library=["WWWWWWWWWW"], settings=SETTINGS)
    manifest = {"schema": "her2-routes-acquisition/1", "outcomes_opened": True,
                "retrieved_utc": "2026-09-27T10:30:55Z", "pins": {}}
    with pytest.raises(ValueError, match="not a reserved population"):
        split_lib.freeze_split(split, sequence_digest="a", algorithm_digest="b",
                               acquisition_manifest=manifest)
    manifest["outcomes_opened"] = False
    record = split_lib.freeze_split(split, sequence_digest="a", algorithm_digest="b",
                                   acquisition_manifest=manifest)
    assert record["record_kind"] == "hard_split_freeze"
    assert record["acquisition"]["outcomes_opened"] is False
    # Re-freezing the same panel is a reconfirmation; re-freezing a different one is not.
    again = split_lib.freeze_split(split, sequence_digest="a", algorithm_digest="b",
                                   acquisition_manifest=manifest, previous=record)
    assert again["reconfirmed"]
    with pytest.raises(ValueError, match="re-freezing silently"):
        split_lib.freeze_split(split, sequence_digest="different", algorithm_digest="b",
                               acquisition_manifest=manifest, previous=record)


def test_digests_are_order_independent_and_settings_sensitive():
    assert split_lib.sequence_digest(["b", "a"]) == split_lib.sequence_digest(["a", "b", "a"])
    assert (split_lib.algorithm_digest(SETTINGS)
            != split_lib.algorithm_digest({**SETTINGS, "seed": 1}))


# --- I15: the exclusion closure -------------------------------------------

def test_closure_refuses_every_declared_consumer_not_only_training_rows():
    closure = split_lib.ExclusionClosure.build(reserved_windows=["WGGDGFYAMD"],
                                               reserved_identities=["H|L"])
    for consumer in split_lib.CONSUMERS:
        with pytest.raises(ValueError, match="refused because"):
            closure.require_allowed(consumer=consumer, window="WGGDGFYAMD")
        with pytest.raises(ValueError, match="refused because"):
            closure.require_allowed(consumer=consumer, identity="H|L")
    # A window that is not reserved passes for every consumer.
    for consumer in split_lib.CONSUMERS:
        assert closure.require_allowed(consumer=consumer, window="AAAAAAAAAA")


def test_closure_refuses_an_undeclared_consumer():
    closure = split_lib.ExclusionClosure.build(reserved_windows=["WGGDGFYAMD"])
    with pytest.raises(ValueError, match="not a declared consumer"):
        closure.require_allowed(consumer="some_new_path", window="AAAAAAAAAA")


def test_closure_report_names_unaudited_consumers():
    closure = split_lib.ExclusionClosure.build(reserved_windows=["WGGDGFYAMD"])
    report = split_lib.closure_report(closure, [("replay_row", "AAAAAAAAAA", None)])
    assert report["passed"]
    assert report["consumers_covered"] == ["replay_row"]
    assert set(report["consumers_unaudited"]) == set(split_lib.CONSUMERS) - {"replay_row"}
    assert "An empty audit is not a passing audit" in report["unaudited_meaning"]
    violating = split_lib.closure_report(closure, [("replay_row", "WGGDGFYAMD", None)])
    assert not violating["passed"]
    assert violating["violations"][0]["consumer"] == "replay_row"


def test_a_shared_partial_masked_context_is_allowed():
    closure = split_lib.ExclusionClosure.build(reserved_windows=["WGGDGFYAMD"])
    document = closure.document()
    assert "fully masked edit window, is allowed" in document["allowed"]
    # At radius zero the exclusion is on identities, so a different window sharing a
    # masked context is not refused.
    assert closure.violates(window="WGGDGFYAMA") is None


def test_the_closure_radius_refuses_a_neighbour_of_a_held_out_core():
    """Identity-only exclusion is not the split's own rule.

    The split purges every training candidate within ``purge_radius`` of a held-out
    core. A closure that refused only exact identities would leave that same
    one-substitution neighbour available to every adaptation consumer -- preference
    endpoints, teacher queries, the generated refresh bank -- which is precisely
    what the purge exists to prevent.
    """
    identity_only = split_lib.ExclusionClosure.build(reserved_windows=["WGGDGFYAMD"])
    with_radius = split_lib.ExclusionClosure.build(
        reserved_windows=["WGGDGFYAMD"], development_windows=["AAAAAAAAAA"], radius=2)
    neighbour = "WGGDGFYAMA"                      # one substitution
    two_away = "WGGDGFYAAA"                       # two substitutions
    far = "CCCCCCCCCC"
    assert identity_only.violates(window=neighbour) is None
    assert "within 1 edit(s)" in with_radius.violates(window=neighbour)
    assert "within 2 edit(s)" in with_radius.violates(window=two_away)
    assert with_radius.violates(window=far) is None
    # A development core is guarded on the same footing as a reserved one.
    assert "within 1 edit(s)" in with_radius.violates(window="AAAAAAAAAW")
    for consumer in ("preference_chosen", "teacher_query", "posttrain_refresh",
                     "synthetic_candidate"):
        with pytest.raises(ValueError, match="inside the declared radius"):
            with_radius.require_allowed(consumer=consumer, window=neighbour)
    assert with_radius.document()["radius"] == 2
    assert "one-substitution neighbour" in with_radius.document()["radius_rule"]


# --- CR9114 reserved identities ------------------------------------------

def test_cr9114_reserved_genotypes_are_rederived_without_reading_h1():
    genotypes = [format(value, "016b") for value in range(256)]
    block = split_lib.cr9114_reserved_genotypes(genotypes, seed=20260916, loci_count=4,
                                                train_blocks=10, development_blocks=3)
    assert len(block["loci_0based"]) == 4
    assert set(block["counts"]) == {"train", "development", "reserved"}
    assert sum(block["counts"].values()) == len(genotypes)
    assert block["reserved"]
    assert "No H1 column is read" in block["derivation"]
    # Deterministic under the same seed, different under another.
    again = split_lib.cr9114_reserved_genotypes(genotypes, seed=20260916, loci_count=4,
                                                train_blocks=10, development_blocks=3)
    assert again["reserved"] == block["reserved"]
    other = split_lib.cr9114_reserved_genotypes(genotypes, seed=1, loci_count=4,
                                                train_blocks=10, development_blocks=3)
    assert other["loci_0based"] != block["loci_0based"] or other["reserved"] != block["reserved"]


def test_cr9114_block_rederivation_is_labelled_secondary():
    block = split_lib.cr9114_reserved_genotypes([format(v, "016b") for v in range(64)],
                                                seed=20260916, loci_count=4, train_blocks=10,
                                                development_blocks=3)
    complement = block["complement_of_eligible"]
    # The re-derivation cannot see the eligibility REJECTS, so it is not the
    # operative protection; the whitelist complement is.
    assert block["status"].startswith("SECONDARY")
    assert complement["resolved"] is True
    assert "cr9114_protected_genotypes" in complement["resolved_by"]
    assert "not positively eligible" in complement["policy"]


def test_cr9114_whitelist_is_read_for_two_columns_and_its_complement_is_protected(tmp_path):
    """The whitelist is the operative protection, and it never touches an H1 column."""
    path = tmp_path / "eligible_non_test_records.csv"
    genotypes = [format(value, "016b") for value in range(64)]
    eligible = genotypes[:40]
    lines = ["genotype,split,h1_mean"]
    for index, genotype in enumerate(eligible):
        lines.append(f"{genotype},{'train' if index < 30 else 'development'},9.5")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    whitelist = split_lib.cr9114_eligibility_whitelist(path, expected_total=40)
    assert whitelist["count"] == 40
    assert whitelist["by_split"] == {"development": 10, "train": 30}
    assert whitelist["columns_read"] == list(split_lib.CR9114_WHITELIST_COLUMNS)
    # The H1 column is in the file, is NAMED as present, and is not among the read ones.
    assert whitelist["h1_columns_present_but_unread"] == ["h1_mean"]
    assert "selected on H1 assay eligibility" in whitelist["selection_bias"]
    protected = split_lib.cr9114_protected_genotypes(whitelist, genotypes,
                                                     expected_protected=24)
    assert protected["count"] == 24
    assert set(protected["protected"]) == set(genotypes) - set(eligible)
    assert "complement of the eligibility whitelist" in protected["rule"].lower()
    with pytest.raises(ValueError, match="against a pinned"):
        split_lib.cr9114_protected_genotypes(whitelist, genotypes, expected_protected=1)
    with pytest.raises(ValueError, match="the pinned whitelist is"):
        split_lib.cr9114_eligibility_whitelist(path, expected_sha256="0" * 64)


def test_cr9114_refuses_a_non_genotype():
    with pytest.raises(ValueError, match="not a 16-bit"):
        split_lib.cr9114_reserved_genotypes(["012"], seed=1, loci_count=4, train_blocks=10,
                                            development_blocks=3)

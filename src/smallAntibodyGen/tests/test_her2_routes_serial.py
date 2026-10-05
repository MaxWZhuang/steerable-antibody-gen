"""I2, I3, I6, I7: serialization, alignment, mask eligibility, paired context.

Offline and synthetic: no weights, no F: access, no huge fixtures. These are the
checks that catch a coding error, and a tiny synthetic example cannot demonstrate
biological generalization -- which is why they are named after implementation checks
and not after panels.
"""
from __future__ import annotations

import pytest

from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments.her2_data import CANONICAL

VOCAB = {**{residue: index for index, residue in enumerate(
    "MRHKDESTNQCGPAVIFYWL", start=3)},
    "<PAD>": 0, "1": 23, "2": 24, "X": 25, "B": 1, "Z": 2}


@pytest.fixture(autouse=True)
def bound_vocabulary():
    serial.bind_piggen_vocabulary(VOCAB)
    yield
    serial.PIGGEN_ID_OF_RESIDUE.clear()


def construct(heavy="ACDEFGHIKL", light="MNPQRSTVWY", sites=((serial.HEAVY, 2),
                                                             (serial.HEAVY, 3))):
    return serial.Construct(heavy=heavy, light=light, sites=serial.edit_map(sites),
                            source="synthetic", record_id="t0")


# --- I2: residue and sentinel identity ------------------------------------

def test_i2_sentinels_and_residues_are_distinct():
    ids = set(serial.PIGGEN_ID_OF_RESIDUE.values())
    assert len(ids) == 20
    for sentinel in (serial.PIGGEN_MASK_ID, serial.PIGGEN_SEP_ID, serial.PIGGEN_START_ID,
                     serial.PIGGEN_END_ID, 0):
        assert sentinel not in ids


def test_i2_arginine_is_never_an_eos_event():
    # The released config sets eos_token_id: 2 and vocab id 2 is a residue in the
    # p-IgGen tokenizer. This campaign's decoders run a FIXED number of steps, so an
    # internal R cannot terminate one -- and the mask/separator ids are outside the
    # canonical support, so they cannot be emitted either.
    assert serial.PIGGEN_MASK_ID not in set(serial.PIGGEN_ID_OF_RESIDUE.values())
    assert serial.PIGGEN_SEP_ID not in set(serial.PIGGEN_ID_OF_RESIDUE.values())
    serialization = serial.causal_serialization(construct(), residues="AA")
    emitted = serialization.tokens[serialization.context_length + 1:]
    assert all(token in set(serial.PIGGEN_ID_OF_RESIDUE.values()) for token in emitted)


def test_i2_ablang_native_ids_match_the_pinned_vocabulary():
    assert serial.ABLANG_MASK_ID == 23
    assert serial.ABLANG_PAD_ID == 21
    assert (serial.ABLANG_START_ID, serial.ABLANG_END_ID, serial.ABLANG_SEP_ID) == (0, 22, 25)
    ids = serial.ablang_canonical_ids()
    assert len(set(ids)) == 20
    assert not set(ids) & {serial.ABLANG_MASK_ID, serial.ABLANG_PAD_ID, serial.ABLANG_START_ID,
                           serial.ABLANG_END_ID, serial.ABLANG_SEP_ID, serial.ABLANG_UNKNOWN_ID}
    # Ordering is CANONICAL order, which is what the log-prob gather assumes.
    assert ids[CANONICAL.index("A")] == serial.ABLANG_RESIDUE_IDS["A"]


# --- I3: prediction alignment ---------------------------------------------

def test_i3_query_slots_predict_the_next_token_not_the_same_one():
    thing = construct()
    order = tuple(thing.sites)
    serialization = serial.causal_serialization(thing, order=order,
                                                residues=thing.residues(order))
    # Slot t's NEXT token must be the answer at step t. A same-position (masked-style)
    # shift would make slot t hold the answer, so the structural claim is asserted too:
    # slot t is context_length + t, i.e. the separator at t = 0 and the PREVIOUS
    # emission afterwards.
    for step, slot in enumerate(serialization.query_slots):
        assert slot == serialization.context_length + step
        answer = serialization.tokens[slot + 1]
        assert answer == serial.PIGGEN_ID_OF_RESIDUE[CANONICAL[serialization.targets[step]]]
    assert serialization.tokens[serialization.context_length] == serial.PIGGEN_SEP_ID


def test_i3_targets_never_land_in_a_masked_input_slot():
    thing = construct()
    serialization = serial.causal_serialization(thing, residues=thing.residues(thing.sites))
    masked = {index for index, token in enumerate(serialization.tokens)
              if token == serial.PIGGEN_MASK_ID}
    assert masked
    assert not masked & {slot + 1 for slot in serialization.query_slots}


def test_i3_inference_serialization_exposes_one_slot_not_k_identical_ones():
    thing = construct()
    serialization = serial.causal_serialization(thing, residues=None)
    assert len(serialization.query_slots) == 1
    assert serialization.query_slots[0] == serialization.context_length
    # The pointers for every step are known in advance: they are coordinates only.
    assert len(serialization.pointers) == len(thing.sites)


# --- I5 / pointer leakage -------------------------------------------------

def test_pointers_depend_only_on_coordinates_not_on_residues():
    left = construct(heavy="ACDEFGHIKL")
    right = construct(heavy="ACWWFGHIKL")     # differs exactly at the two edit sites
    assert left.sites == right.sites
    assert serial.pointer_positions(left, left.sites) == \
        serial.pointer_positions(right, right.sites)
    # And the two alternatives' serialised CONTEXTS are byte-identical, because the
    # differing positions are masked.
    a = serial.causal_serialization(left, residues=left.residues(left.sites))
    b = serial.causal_serialization(right, residues=right.residues(right.sites))
    assert a.tokens[:a.context_length] == b.tokens[:b.context_length]
    assert a.pointers == b.pointers
    assert a.targets != b.targets


def test_pointer_positions_shift_with_the_chain_lengths():
    short = construct(heavy="ACDEFGHIKL", light="MN", sites=((serial.LIGHT, 1),))
    long = construct(heavy="ACDEFGHIKLM", light="MN", sites=((serial.LIGHT, 1),))
    assert serial.pointer_positions(short, short.sites)[0] == 1 + 10 + 1
    assert serial.pointer_positions(long, long.sites)[0] == 1 + 11 + 1


# --- I6: mask eligibility -------------------------------------------------

def test_i6_every_view_has_at_least_one_eligible_mask():
    with pytest.raises(ValueError, match="names no sites"):
        serial.edit_map([])


def test_i6_only_selected_positions_are_masked():
    thing = construct(sites=((serial.HEAVY, 0), (serial.LIGHT, 4)))
    serialization = serial.causal_serialization(thing, residues=thing.residues(thing.sites))
    masked = [index for index, token in enumerate(serialization.tokens[:serialization.context_length])
              if token == serial.PIGGEN_MASK_ID]
    positions = serial.causal_positions(thing)
    assert sorted(masked) == sorted(positions.values())


def test_i6_masked_editor_refuses_an_unmasked_query():
    thing = construct(sites=((serial.HEAVY, 2), (serial.HEAVY, 3)))
    with pytest.raises(ValueError, match="unmasked query reads its own answer"):
        serial.ablang_serialization(thing, masked_sites=((serial.HEAVY, 2),),
                                    order=((serial.HEAVY, 2), (serial.HEAVY, 3)))


def test_i6_position_map_and_masking_must_agree():
    thing = construct(sites=((serial.HEAVY, 2),))
    serialization = serial.ablang_serialization(thing)
    slot = serialization.query_slots[0]
    assert serialization.tokens[slot] == serial.ABLANG_MASK_ID


# --- I7: complete paired context ------------------------------------------

def test_i7_both_chains_and_their_boundaries_survive():
    thing = construct(heavy="ACDEFGHIKL", light="MNPQRSTVWY")
    serialization = serial.causal_serialization(thing, residues=thing.residues(thing.sites))
    assert serialization.context_length == 1 + 10 + 10 + 1
    assert serialization.tokens[0] == serial.PIGGEN_START_ID
    assert serialization.tokens[serialization.context_length - 1] == serial.PIGGEN_END_ID
    # The light chain is present, unmasked, in order.
    light_ids = serialization.tokens[1 + 10:1 + 20]
    assert list(light_ids) == [serial.PIGGEN_ID_OF_RESIDUE[r] for r in thing.light]


def test_i7_ablang_paired_length_and_offsets():
    thing = construct(heavy="A" * 120, light="C" * 107, sites=((serial.HEAVY, 98),))
    tokens = serial.ablang_token_ids(thing)
    assert len(tokens) == 120 + 107 + 5 == 232
    positions = serial.ablang_positions(thing)
    assert positions[(serial.HEAVY, 98)] == 99
    light = serial.Construct(heavy="A" * 120, light="C" * 107,
                             sites=serial.edit_map([(serial.LIGHT, 0)]), source="s")
    assert serial.ablang_positions(light)[(serial.LIGHT, 0)] == 124
    assert tokens[124] == serial.ABLANG_RESIDUE_IDS["C"]


def test_i7_heavy_only_strips_the_light_bracket_and_keeps_the_separator():
    nanobody = serial.Construct(heavy="ACDEFGHIKL", light="",
                                sites=serial.edit_map([(serial.HEAVY, 3)]), source="vhh")
    assert nanobody.is_heavy_only
    tokens = serial.ablang_token_ids(nanobody)
    # f"<{heavy}>|<{light}>".replace("<>", "") leaves a trailing separator and NO
    # closing light bracket. Asserting it means a symmetric guess cannot creep in.
    assert tokens[-1] == serial.ABLANG_SEP_ID
    assert len(tokens) == 1 + 10 + 1 + 1
    assert serial.ablang_positions(nanobody)[(serial.HEAVY, 3)] == 4


def test_i7_nothing_truncates_it_raises_with_the_measured_length():
    thing = construct(heavy="A" * 300, light="C" * 200, sites=((serial.HEAVY, 0),))
    with pytest.raises(ValueError, match="needs 504 positions"):
        serial.causal_serialization(thing, residues="A", max_positions=400)


# --- edit maps ------------------------------------------------------------

def test_edit_map_sorts_heavy_before_light_and_refuses_duplicates():
    mapping = serial.edit_map([(serial.LIGHT, 5), (serial.HEAVY, 9), (serial.HEAVY, 1)])
    assert mapping == ((serial.HEAVY, 1), (serial.HEAVY, 9), (serial.LIGHT, 5))
    with pytest.raises(ValueError, match="Duplicate edit site"):
        serial.edit_map([(serial.HEAVY, 1), (serial.HEAVY, 1)])
    with pytest.raises(ValueError, match="non-negative int"):
        serial.edit_map([(serial.HEAVY, True)])


def test_her2_core_edit_map_comes_from_the_asserted_constants():
    mapping = serial.her2_core_edit_map()
    assert len(mapping) == 10
    assert mapping[0] == (serial.HEAVY, 98)
    assert mapping[-1] == (serial.HEAVY, 107)


def test_construct_refuses_a_non_canonical_residue_rather_than_sanitising():
    with pytest.raises(ValueError, match="non-canonical"):
        serial.Construct(heavy="ACDXFGHIKL", light="MN",
                         sites=serial.edit_map([(serial.HEAVY, 0)]), source="s")


def test_with_residues_touches_only_the_named_sites():
    thing = construct()
    edited = thing.with_residues(((serial.HEAVY, 2),), "W")
    assert edited.heavy[2] == "W"
    assert edited.heavy[:2] == thing.heavy[:2]
    assert edited.heavy[3:] == thing.heavy[3:]
    assert edited.light == thing.light


def test_serialised_length_matches_the_real_serialisation():
    thing = construct()
    causal = serial.causal_serialization(thing, residues=thing.residues(thing.sites))
    assert serial.serialised_length(thing, route="causal") == causal.length
    masked = serial.ablang_serialization(thing)
    assert serial.serialised_length(thing, route="masked") == masked.length


def test_unbound_vocabulary_is_refused_rather_than_emitting_pad():
    serial.PIGGEN_ID_OF_RESIDUE.clear()
    with pytest.raises(ValueError, match="unbound"):
        serial.causal_serialization(construct(), residues="AA")

"""One serialization contract for all three generative routes.

This module exists so that "diffusion helped" can never mean "it saw the light
chain". Every route receives the same paired construct, the same edit map and
the same visible background; only the model and the objective differ.

Four properties are enforced here rather than left to each route:

* **An edit map is a tuple of ``(chain, index)`` coordinates**, not a slice.
  HER2's ``heavy[98:108]`` is one instance of :class:`EditMap`; CR9114 is
  sixteen heavy sites, and the FLAb/AlphaSeq sources carry light-chain and
  mixed-CDR edits. A route that only understands a contiguous heavy window
  cannot be compared against one that does not, so none of them gets that
  shortcut.
* **Serialised positions are derived, never searched.** ``her2_data`` already
  refuses to locate the editable window with ``str.index`` at generation time;
  the same discipline applies to the two tokenizations here. Both are pure
  functions of the chain lengths, and both are asserted against the pinned
  native conventions (p-IgGen ``1 + VH + VL + 2``; AbLang2 ``<VH>|<VL>``).
* **Nothing truncates.** A construct whose serialised length exceeds the
  model's position budget raises with the measured length in the message. The
  failure mode this prevents is a silently dropped light chain -- the model
  would still train, and the paired-context claim would be false.
* **The query pointer depends on coordinates only.** The autoregressive route
  needs to say *which* site it is emitting at step ``t``. It says so with a
  position pointer, not with a marker token: a pointer cannot be emitted by the
  output head, and it carries no information about the residue at the site, so
  the two alternatives of a masked contrast receive byte-identical pointers.
  :func:`pointer_positions` is the function a leakage test can call directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .her2_data import CANONICAL, CORE_LENGTH, CORE_START, HEAVY_LENGTH, LIGHT_LENGTH
from .her2_runtime import require

HEAVY = "H"
LIGHT = "L"
CHAINS = (HEAVY, LIGHT)

#: p-IgGen biological sentinels, from ``her2_policy.load_vocab``'s assertions.
PIGGEN_START_ID = 23
PIGGEN_END_ID = 24
#: The two tokens this campaign appends to p-IgGen's 26-entry vocabulary. Ids 26
#: and 27 are the next free slots; neither is ever in the sampling support, so
#: neither can be emitted, and ``eos_token_id: 2`` (arginine) still cannot
#: terminate a decode because decoding runs a fixed number of steps.
PIGGEN_MASK_ID = 26
PIGGEN_SEP_ID = 27
PIGGEN_VOCAB_SIZE = 26
PIGGEN_EXTENDED_VOCAB_SIZE = 28

#: AbLang2 native ids, transcribed from the pinned ``ablang2/models/ablang2/vocab.py``.
ABLANG_START_ID = 0
ABLANG_PAD_ID = 21
ABLANG_END_ID = 22
ABLANG_MASK_ID = 23
ABLANG_UNKNOWN_ID = 24
ABLANG_SEP_ID = 25
ABLANG_VOCAB_SIZE = 26
#: Residue -> id, transcribed from the same file. Deliberately written out rather
#: than imported from F:, so a changed upstream vocabulary fails the parity test
#: instead of silently re-indexing every weight row.
ABLANG_RESIDUE_IDS = {
    "M": 1, "R": 2, "H": 3, "K": 4, "D": 5, "E": 6, "S": 7, "T": 8, "N": 9,
    "Q": 10, "C": 11, "G": 12, "P": 13, "A": 14, "V": 15, "I": 16, "F": 17,
    "Y": 18, "W": 19, "L": 20}


def ablang_canonical_ids():
    """Native ids of the twenty canonical residues, in ``her2_data.CANONICAL`` order."""
    return tuple(ABLANG_RESIDUE_IDS[residue] for residue in CANONICAL)


def canonical_index(residue):
    position = CANONICAL.find(str(residue))
    require(position >= 0, f"{residue!r} is not one of the twenty canonical residues")
    return position


# ---------------------------------------------------------------------------
# edit maps
# ---------------------------------------------------------------------------

def edit_map(sites):
    """Validate and canonicalise a tuple of ``(chain, index)`` edit coordinates.

    Sorted heavy-then-light, ascending, duplicates refused. Sorting here is what
    makes an edit map a content-addressable identity: two callers that name the
    same sites in different orders produce the same map, and the declared
    emission order is a separate, explicit object (:func:`emission_order`).
    """
    cleaned = []
    for site in sites:
        require(isinstance(site, (tuple, list)) and len(site) == 2,
                f"An edit site is a (chain, index) pair, got {site!r}")
        chain, index = str(site[0]), site[1]
        require(chain in CHAINS, f"Unknown chain {chain!r}; expected one of {CHAINS}")
        require(isinstance(index, int) and not isinstance(index, bool) and index >= 0,
                f"Edit index must be a non-negative int, got {index!r}")
        cleaned.append((chain, int(index)))
    require(cleaned, "An edit map names no sites; there would be nothing to predict")
    require(len(set(cleaned)) == len(cleaned),
            f"Duplicate edit site in {sorted(cleaned)}; a site predicted twice would be "
            "counted twice in every exposure and loss")
    return tuple(sorted(cleaned, key=lambda site: (CHAINS.index(site[0]), site[1])))


def her2_core_edit_map():
    """The ten trastuzumab HCDR3 positions, derived from the asserted constants."""
    return edit_map([(HEAVY, CORE_START + offset) for offset in range(CORE_LENGTH)])


def emission_order(mapping, *, order="declared", rng=None):
    """The order the autoregressive route fills an edit map in.

    ``declared`` is the canonical (sorted) order and is what every reported
    whole-window log probability uses, because a fixed order defines one
    normalised joint over the window. ``random`` draws a permutation from
    ``rng`` and is the decoding ablation; it is never the reporting functional.
    """
    require(order in ("declared", "random"), f"Unknown emission order {order!r}")
    sites = list(mapping)
    if order == "random":
        require(rng is not None, "A random emission order needs an explicit generator")
        sites = [sites[i] for i in rng.permutation(len(sites))]
    return tuple(sites)


# ---------------------------------------------------------------------------
# constructs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Construct:
    """One full paired antibody plus the coordinates that may be edited.

    ``light`` may be empty for a nanobody or a heavy-only record. That is a
    declared contract, not a convenience: :func:`ablang_tokens` asserts what the
    native tokenizer actually does with an empty light chain rather than
    assuming a symmetric encoding.
    """

    heavy: str
    light: str
    sites: tuple
    source: str = "unknown"
    record_id: str = ""

    def __post_init__(self):
        require(isinstance(self.heavy, str) and self.heavy, "A construct needs a heavy chain")
        require(isinstance(self.light, str), "The light chain must be a string, possibly empty")
        for chain, text in ((HEAVY, self.heavy), (LIGHT, self.light)):
            bad = sorted({character for character in text if character not in CANONICAL})
            require(not bad,
                    f"{self.source}:{self.record_id}: non-canonical residue(s) {bad} in the "
                    f"{chain} chain. A construct is not silently sanitised -- an X or a gap is a "
                    "different sequence and it is reported.")
        for chain, index in self.sites:
            length = len(self.heavy) if chain == HEAVY else len(self.light)
            require(index < length,
                    f"{self.source}:{self.record_id}: edit site ({chain}, {index}) is outside the "
                    f"{chain} chain of length {length}")

    @property
    def is_heavy_only(self):
        return self.light == ""

    def residue_at(self, site):
        chain, index = site
        return (self.heavy if chain == HEAVY else self.light)[index]

    def residues(self, sites):
        return "".join(self.residue_at(site) for site in sites)

    def with_residues(self, sites, residues):
        """A copy carrying ``residues`` at ``sites``. The scaffold is untouched."""
        require(len(sites) == len(residues),
                f"{len(sites)} sites and {len(residues)} residues do not correspond")
        heavy, light = list(self.heavy), list(self.light)
        for site, residue in zip(sites, residues):
            require(residue in CANONICAL, f"{residue!r} is not canonical")
            chain, index = site
            (heavy if chain == HEAVY else light)[index] = residue
        return Construct(heavy="".join(heavy), light="".join(light), sites=self.sites,
                         source=self.source, record_id=self.record_id)

    def identity(self):
        """The exact-sequence identity used for split closure across every source."""
        return f"{self.heavy}|{self.light}"

    def document(self):
        return {"source": self.source, "record_id": self.record_id,
                "heavy_length": len(self.heavy), "light_length": len(self.light),
                "heavy_only": bool(self.is_heavy_only),
                "sites": [[chain, int(index)] for chain, index in self.sites]}


def her2_construct(scaffold, core, *, record_id=""):
    """The fixed HER2 scaffold carrying one ten-residue core.

    The offsets come from ``her2_data``'s asserted constants and from
    ``load_scaffold``'s own checks, so a changed release fails there.
    """
    require(len(core) == CORE_LENGTH, f"A HER2 core is {CORE_LENGTH} residues, got {len(core)}")
    require(len(scaffold.heavy) == HEAVY_LENGTH and len(scaffold.light) == LIGHT_LENGTH,
            "The fixed HER2 scaffold chain lengths changed")
    return Construct(heavy=scaffold.with_core(core), light=scaffold.light,
                     sites=her2_core_edit_map(), source="her2_display", record_id=record_id)


# ---------------------------------------------------------------------------
# route 2: the causal (p-IgGen) serialization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalSerialization:
    """Token ids, the query pointers and the targets for one AR training view.

    Layout, with lengths rather than magic numbers::

        [START] VH(masked at heavy sites) VL [END] [SEP] r_1 r_2 ... r_k

    The context block is ``1 + len(H) + len(L) + 1`` tokens. The emission block
    is ``1 + k``: the separator, then the residues already emitted. At query
    step ``t`` the model reads the residue emitted at ``t - 1`` (or the
    separator, at ``t = 0``) and the pointer of the site it must predict now,
    and its next-token logits are the conditional for that site.
    """

    tokens: tuple
    query_slots: tuple
    pointers: tuple
    targets: tuple
    order: tuple
    context_length: int

    @property
    def length(self):
        return len(self.tokens)

    def document(self):
        return {"tokens": self.length, "context_length": int(self.context_length),
                "queries": len(self.query_slots),
                "order": [[chain, int(index)] for chain, index in self.order]}


def causal_positions(construct):
    """``{site: serialised position}`` for the causal layout. Derived, not searched."""
    offsets = {HEAVY: 1, LIGHT: 1 + len(construct.heavy)}
    return {site: offsets[site[0]] + site[1] for site in construct.sites}


def pointer_positions(construct, order):
    """The pointer value for every query step: a function of coordinates only.

    This is the whole leakage argument for the pointer design, in one callable.
    It never touches ``construct.heavy``/``construct.light`` contents, only their
    lengths and the site coordinates, so the two alternatives of a masked
    contrast -- identical coordinates, different residues -- necessarily receive
    identical pointers. ``test_her2_routes_pointer`` asserts exactly that.
    """
    positions = causal_positions(construct)
    return tuple(positions[site] for site in order)


def causal_serialization(construct, *, order=None, residues=None, max_positions=400):
    """Serialise one construct for the AR route.

    ``residues`` supplies the teacher-forced answers; with ``None`` the emission
    block holds only the separator, which is the inference-time prefix.
    """
    require_bound_vocabulary()
    order = tuple(order) if order is not None else tuple(construct.sites)
    require(set(order) == set(construct.sites),
            f"The emission order {order} is not a permutation of the edit map "
            f"{construct.sites}")
    require(len(set(order)) == len(order), "The emission order repeats a site")
    positions = causal_positions(construct)
    masked_heavy = list(construct.heavy)
    masked_light = list(construct.light)
    for chain, index in construct.sites:
        (masked_heavy if chain == HEAVY else masked_light)[index] = None
    context = [PIGGEN_START_ID]
    for buffer in (masked_heavy, masked_light):
        for residue in buffer:
            context.append(PIGGEN_MASK_ID if residue is None
                           else PIGGEN_ID_OF_RESIDUE[residue])
    context.append(PIGGEN_END_ID)
    context_length = len(context)
    require(context_length == 1 + len(construct.heavy) + len(construct.light) + 1,
            "The causal context block is not 1 + VH + VL + 1 tokens")

    answers = None
    if residues is not None:
        require(len(residues) == len(order),
                f"{len(residues)} residues for {len(order)} query steps")
        answers = tuple(canonical_index(residue) for residue in residues)

    tokens = list(context) + [PIGGEN_SEP_ID]
    # Pointers are known for every step in advance: they are a function of the site
    # coordinates and the chain lengths alone. Query slots are the positions whose
    # NEXT-token logits are the conditionals -- the separator at step 0 and the
    # previously emitted residue afterwards, i.e. ``context_length + step``. With
    # the answers supplied, one causal forward yields all k conditionals at once,
    # and causal masking is what makes that exact rather than an approximation.
    pointers = [positions[site] for site in order]
    targets = list(answers) if answers is not None else []
    if answers is not None:
        tokens += [PIGGEN_ID_OF_RESIDUE[CANONICAL[value]] for value in answers]
        query_slots = [context_length + step for step in range(len(order))]
    else:
        # Inference: only the first conditional exists yet. The decoder extends the
        # sequence and advances the slot itself; fabricating k identical slots here
        # is how a whole window silently gets scored from one prefix.
        query_slots = [context_length]
    require(len(tokens) <= int(max_positions),
            f"The serialised construct needs {len(tokens)} positions and the model budget is "
            f"{max_positions}. Nothing is truncated here: a dropped light chain would leave the "
            "paired-context claim false while training continued normally.")
    return CausalSerialization(tokens=tuple(tokens), query_slots=tuple(query_slots),
                               pointers=tuple(pointers), targets=tuple(targets),
                               order=order, context_length=context_length)


#: p-IgGen is character-level over a 26-entry vocabulary whose residue ids are
#: read from the pinned ``tokenizer.json`` at load time. The serialiser needs the
#: mapping before a model exists (for offline tests and for the data stream), so
#: it is built once from :data:`CANONICAL` and *verified* against the released
#: tokenizer by :func:`bind_piggen_vocabulary`.
PIGGEN_ID_OF_RESIDUE = {}


def bind_piggen_vocabulary(vocab):
    """Bind residue -> id from the released tokenizer and check the sentinels.

    Called once by the route-2 model construction. The serialiser refuses to run
    against an unbound table, so an offline test has to bind a table explicitly
    rather than silently serialise against zeros.
    """
    require(vocab["1"] == PIGGEN_START_ID and vocab["2"] == PIGGEN_END_ID,
            "p-IgGen biological sentinel ids moved")
    require(max(vocab.values()) == PIGGEN_VOCAB_SIZE - 1,
            f"The released p-IgGen vocabulary is no longer {PIGGEN_VOCAB_SIZE} entries; ids "
            f"{PIGGEN_MASK_ID}/{PIGGEN_SEP_ID} may no longer be free")
    PIGGEN_ID_OF_RESIDUE.clear()
    for residue in CANONICAL:
        require(residue in vocab, f"Canonical residue {residue!r} is missing from the vocabulary")
        PIGGEN_ID_OF_RESIDUE[residue] = int(vocab[residue])
    require(PIGGEN_MASK_ID not in PIGGEN_ID_OF_RESIDUE.values()
            and PIGGEN_SEP_ID not in PIGGEN_ID_OF_RESIDUE.values(),
            "A canonical residue collides with the appended mask/separator ids")
    return dict(PIGGEN_ID_OF_RESIDUE)


def require_bound_vocabulary():
    require(PIGGEN_ID_OF_RESIDUE,
            "The p-IgGen residue table is unbound. Call bind_piggen_vocabulary with the released "
            "tokenizer vocabulary first; serialising against an empty table would emit zeros, "
            "which is <PAD>.")


# ---------------------------------------------------------------------------
# route 1: the AbLang2 masked-editor serialization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskedSerialization:
    """Native AbLang2 token ids plus the positions whose logits are the answers."""

    tokens: tuple
    query_slots: tuple
    targets: tuple
    order: tuple
    heavy_only: bool

    @property
    def length(self):
        return len(self.tokens)

    def document(self):
        return {"tokens": self.length, "queries": len(self.query_slots),
                "heavy_only": bool(self.heavy_only),
                "order": [[chain, int(index)] for chain, index in self.order]}


def ablang_positions(construct):
    """``{site: serialised position}`` for the native ``<VH>|<VL>`` layout.

    Derived from the tokenizer's own string construction,
    ``f"<{heavy}>|<{light}>".replace("<>", "")``:

    * paired -- ``<`` H... ``>`` ``|`` ``<`` L... ``>``, so heavy residue ``i``
      sits at ``1 + i`` and light residue ``j`` at ``len(H) + 4 + j``; the total
      is ``len(H) + len(L) + 5``;
    * heavy-only -- the empty light chain makes the tail ``<>``, which the
      tokenizer *deletes*, leaving ``<`` H... ``>`` ``|``: a trailing separator
      and no light bracket at all. That asymmetry is asserted rather than
      assumed, because a symmetric guess would shift every light-chain pointer.
    """
    heavy_offset = 1
    if construct.is_heavy_only:
        return {site: heavy_offset + site[1] for site in construct.sites}
    light_offset = len(construct.heavy) + 4
    return {site: (heavy_offset if site[0] == HEAVY else light_offset) + site[1]
            for site in construct.sites}


def ablang_token_ids(construct, *, masked_sites=()):
    """Native token ids for one construct, with ``masked_sites`` set to id 23."""
    masked = set(masked_sites)
    heavy = [ABLANG_MASK_ID if (HEAVY, i) in masked else ABLANG_RESIDUE_IDS[r]
             for i, r in enumerate(construct.heavy)]
    light = [ABLANG_MASK_ID if (LIGHT, j) in masked else ABLANG_RESIDUE_IDS[r]
             for j, r in enumerate(construct.light)]
    tokens = [ABLANG_START_ID, *heavy, ABLANG_END_ID, ABLANG_SEP_ID]
    if not construct.is_heavy_only:
        tokens += [ABLANG_START_ID, *light, ABLANG_END_ID]
    return tuple(tokens)


def ablang_serialization(construct, *, masked_sites=None, order=None, max_positions=None):
    """Serialise one construct for the masked-editor route.

    ``masked_sites`` defaults to the whole edit map, which is the whole-window
    task. A local contrast passes the union of changed positions so both
    measured alternatives see the identical visible background.
    """
    order = tuple(order) if order is not None else tuple(construct.sites)
    masked = tuple(masked_sites) if masked_sites is not None else tuple(construct.sites)
    require(set(order) <= set(masked),
            f"Every queried site must be masked in the input. Queried {sorted(set(order))}, "
            f"masked {sorted(set(masked))} -- an unmasked query reads its own answer.")
    tokens = ablang_token_ids(construct, masked_sites=masked)
    if max_positions is not None:
        require(len(tokens) <= int(max_positions),
                f"The serialised construct needs {len(tokens)} positions, budget "
                f"{max_positions}")
    positions = ablang_positions(construct)
    for site in order:
        require(tokens[positions[site]] == ABLANG_MASK_ID,
                f"Site {site} maps to serialised position {positions[site]}, which does not hold "
                "the native mask token. The position map and the masking disagree, which is the "
                "one error a reconstruction loss cannot show you.")
    targets = tuple(canonical_index(construct.residue_at(site)) for site in order)
    return MaskedSerialization(tokens=tokens, query_slots=tuple(positions[s] for s in order),
                               targets=targets, order=order,
                               heavy_only=bool(construct.is_heavy_only))


def serialised_length(construct, *, route):
    """The token count a route would need, for budget forecasting and for asserts."""
    if route == "causal":
        return 1 + len(construct.heavy) + len(construct.light) + 1 + 1 + len(construct.sites)
    if route == "masked":
        return len(ablang_token_ids(construct))
    raise ValueError(f"Unknown route {route!r}")

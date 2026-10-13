from __future__ import annotations

import ast
import dataclasses
import itertools
from pathlib import Path

import pytest

from smallAntibodyGen.models.antibody_antigen import views as views_module
from smallAntibodyGen.models.antibody_antigen.views import MaskedTargets, VisibleView, mask_complete

SEQUENCES = {"H": "EVQL", "L": "DIQ", "Ag": "KVF"}


def sample_view():
    return mask_complete(SEQUENCES, [("L", 2), ("H", 1)])


def test_views_module_imports_only_the_standard_library():
    tree = ast.parse(Path(views_module.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots <= {"__future__", "hashlib", "dataclasses", "typing"}


def test_mask_complete_returns_a_view_and_separate_targets():
    view, targets = sample_view()
    assert view.roles == ("H", "L", "Ag")
    assert view.chains == (("E", None, "Q", "L"), ("D", "I", None), ("K", "V", "F"))
    assert view.hidden_sites() == (("H", 1), ("L", 2))
    assert targets.items == ((("H", 1), "V"), (("L", 2), "Q"))
    assert len(view.coords()) == 10 and view.coords()[0] == ("H", 0) and view.coords()[-1] == ("Ag", 2)
    # The view has no slot for gold residues, ids, labels or records.
    assert {field.name for field in dataclasses.fields(view)} == {"roles", "chains"}


@pytest.mark.parametrize(
    "roles", [c for n in (1, 2, 3) for c in itertools.combinations(("H", "L", "Ag"), n)], ids="+".join
)
def test_every_non_empty_role_subset_is_a_valid_view(roles):
    masked = [(role, 0) for role in roles if role != "Ag"]
    view, targets = mask_complete({role: SEQUENCES[role] for role in roles}, masked)
    assert view.roles == roles
    assert len(targets) == len(masked)
    for role in roles:
        assert view.has_role(role) and len(view.chain(role)) == len(SEQUENCES[role])


def test_views_are_immutable_and_reveal_returns_a_new_view():
    view, _ = sample_view()
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.roles = ("H",)  # type: ignore[misc]
    assert type(view.chains) is tuple and all(type(chain) is tuple for chain in view.chains)
    snapshot = (view.roles, view.chains)
    revealed = view.reveal(("H", 1), "V")
    assert revealed is not view and revealed.token(("H", 1)) == "V"
    assert revealed.hidden_sites() == (("L", 2),)
    assert (view.roles, view.chains) == snapshot and view.token(("H", 1)) is None


def test_reveal_never_overwrites_a_known_residue_or_accepts_junk():
    view, _ = sample_view()
    for bad in (
        lambda: view.reveal(("H", 0), "A"),  # already visible
        lambda: view.reveal(("Ag", 0), "A"),
        lambda: view.reveal(("H", 1), "a"),
        lambda: view.reveal(("H", 1), "X"),
        lambda: view.reveal(("H", True), "A"),
        lambda: view.reveal(["H", 1], "A"),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(IndexError):
        view.reveal(("H", 9), "A")
    with pytest.raises(KeyError):
        mask_complete({"H": "EV"}, [("H", 0)])[0].reveal(("L", 0), "A")


@pytest.mark.parametrize(
    "build, error",
    [
        (lambda: VisibleView(("H",), ("EVQ",)), TypeError),  # a string mistaken for a token tuple
        (lambda: VisibleView(("H",), (["E", "V"],)), TypeError),  # a list is mutable
        (lambda: VisibleView(("H",), ((),)), ValueError),  # empty chain
        (lambda: VisibleView((), ()), ValueError),  # no roles
        (lambda: VisibleView(("H", "H"), (("E",), ("V",))), ValueError),  # duplicate role
        (lambda: VisibleView(("L", "H"), (("E",), ("V",))), ValueError),  # not canonical order
        (lambda: VisibleView(("X",), (("E",),)), ValueError),  # unknown role
        (lambda: VisibleView(("Ag",), ((None, "E"),)), ValueError),  # masked antigen
        (lambda: VisibleView(("H",), (("e",),)), ValueError),  # lowercase
        (lambda: VisibleView(("H",), (("X",),)), ValueError),  # non-canonical letter
        (lambda: VisibleView(("H",), (("EV",),)), ValueError),  # multi-character token
        (lambda: VisibleView(("H",), ((1,),)), ValueError),  # non-string token
        (lambda: VisibleView("H", (("E",),)), TypeError),  # roles must be a tuple
        (lambda: VisibleView(["H"], (("E",),)), TypeError),
        (lambda: VisibleView(("H", "L"), (("E",),)), ValueError),  # length mismatch
    ],
)
def test_malformed_views_are_rejected(build, error):
    with pytest.raises(error):
        build()


@pytest.mark.parametrize(
    "sequences, masked",
    [
        ({"H": ("E", "V")}, []),  # token tuple instead of a string
        ({"H": ""}, []),
        ({"H": "EvQ"}, []),
        ({"H": "EXQ"}, []),
        ({}, []),
        ({"Z": "EV"}, []),
        ({"H": "EV"}, [("H", True)]),  # bool index
        ({"H": "EV"}, [("H", 1.0)]),
        ({"H": "EV"}, [("H", -1)]),
        ({"H": "EV"}, [("H", 2)]),  # out of range
        ({"H": "EV"}, [("L", 0)]),  # absent role
        ({"H": "EV", "Ag": "KV"}, [("Ag", 0)]),  # antigen is never masked
        ({"H": "EV"}, [("H", 0), ("H", 0)]),  # duplicate site
        ({"H": "EV"}, [["H", 0]]),  # coordinates are tuples
        ({"H": "EV"}, [("Q", 0)]),
    ],
)
def test_mask_complete_rejects_bad_input(sequences, masked):
    with pytest.raises(ValueError):
        mask_complete(sequences, masked)


def test_position_and_token_follow_canonical_coordinate_order():
    view, _ = sample_view()
    for row, coord in enumerate(view.coords()):
        assert view.position(coord) == row
        assert view.token(coord) == view.chain(coord[0])[coord[1]]
    with pytest.raises(KeyError):
        mask_complete({"H": "EV"})[0].position(("Ag", 0))


def test_role_selection_keeps_canonical_order_and_never_returns_an_empty_view():
    view, _ = sample_view()
    assert view.select_roles(["Ag", "H"]).roles == ("H", "Ag")
    assert view.without_ag().roles == ("H", "L")
    assert view.ag_only().roles == ("Ag",) and view.ag_only().hidden_sites() == ()
    assert view.without_ag().chain("H") == view.chain("H")
    for bad in ("HL", [], ["H", "H"]):
        with pytest.raises((TypeError, ValueError)):
            view.select_roles(bad)
    with pytest.raises(KeyError):
        view.without_ag().select_roles(["Ag"])
    with pytest.raises(ValueError):
        view.ag_only().without_ag()


def test_content_digest_is_deterministic_and_depends_only_on_visible_content():
    view, targets = sample_view()
    again, _ = sample_view()
    assert view.content_digest() == again.content_digest()
    # Different hidden truth, same visible content, same digest.
    other, other_targets = mask_complete({"H": "EAQL", "L": "DIA", "Ag": "KVF"}, [("L", 2), ("H", 1)])
    assert other == view and other.content_digest() == view.content_digest()
    assert other_targets != targets
    digests = {
        view.content_digest(),
        view.reveal(("H", 1), "V").content_digest(),
        view.without_ag().content_digest(),
        mask_complete({"H": "EVQL", "L": "DIQ", "Ag": "KVY"}, [("L", 2), ("H", 1)])[0].content_digest(),
        mask_complete(SEQUENCES, [("L", 2), ("H", 1), ("H", 2)])[0].content_digest(),
    }
    assert len(digests) == 5


def test_masked_targets_are_canonically_ordered_and_validated():
    targets = MaskedTargets.from_mapping({("L", 2): "Q", ("H", 3): "L", ("H", 1): "V"})
    assert targets.coords() == (("H", 1), ("H", 3), ("L", 2))
    assert targets.residue(("H", 3)) == "L" and ("L", 2) in targets and ("L", 0) not in targets
    assert len(targets) == 3 and targets.as_dict() == {("H", 1): "V", ("H", 3): "L", ("L", 2): "Q"}
    with pytest.raises(KeyError):
        targets.residue(("L", 0))
    for bad in (
        lambda: MaskedTargets(((("H", 1), "V"), (("H", 1), "A"))),  # duplicate
        lambda: MaskedTargets(((("L", 0), "V"), (("H", 1), "A"))),  # not canonical order
        lambda: MaskedTargets(((("H", 1), "v"),)),  # lowercase
        lambda: MaskedTargets(((("H", True), "V"),)),
        lambda: MaskedTargets([(("H", 1), "V")]),  # must be a tuple
    ):
        with pytest.raises((TypeError, ValueError)):
            bad()

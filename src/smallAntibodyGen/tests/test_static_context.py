"""Tests for StaticContext: construction, query validation, persistence, and autograd."""
from __future__ import annotations

import os
import pytest
import torch
import torch.nn as nn

from smallAntibodyGen.models.antibody_antigen.views import VisibleView
from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(**overrides):
    base = {
        "source_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "extraction_layer": "layer_12",
        "precision": "float32",
        "tokenizer": "esm2_t33",
        "eval_mode": "true",
    }
    base.update(overrides)
    return base


def _h_view():
    # H-only: positions 0,1 visible; 2,3 masked (hidden union = {(H,2),(H,3)})
    return VisibleView(("H",), (("A", "C", None, None, "D"),))


def _hl_view():
    # H+L: H positions 1 masked; L position 2 masked
    return VisibleView(
        ("H", "L"),
        (("A", None, "C"), ("D", "E", None)),
    )


def _hlag_view():
    # H+L+Ag: same Ab masks as _hl_view, Ag fully visible
    return VisibleView(
        ("H", "L", "Ag"),
        (("A", None, "C"), ("D", "E", None), ("F", "G", "H")),
    )


def _make_ctx(view, D=4, identity=None, schema=None):
    n = len(view.coords())
    features = torch.randn(n, D)
    return StaticContext(view, features, identity or _identity(), schema)


def test_representation_layout_is_bound_to_key_and_integrity(tmp_path):
    view = _h_view()
    values = torch.zeros(len(view.coords()), 4, dtype=torch.float32)
    a = StaticContext(view, values, _identity())
    b = StaticContext(view, values.view(torch.float16), _identity())
    assert a.content_key != b.content_key
    path = tmp_path / "cache.pt"
    a.save(path)
    data = torch.load(path, weights_only=True)
    data["features"] = data["features"].view(torch.float16)
    torch.save(data, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        StaticContext.load(path)


@pytest.mark.parametrize("extra", [{"clean_sequence": "ACDE"}, {"assay_id": "high"}])
def test_identity_has_closed_fields(extra):
    with pytest.raises(ValueError, match="unexpected fields"):
        _make_ctx(_h_view(), identity=_identity(**extra))


def test_schema_cannot_carry_arbitrary_metadata():
    with pytest.raises(ValueError, match="derived feature"):
        _make_ctx(_h_view(), schema={"gold": "ACDE"})


def test_training_mode_and_invalid_source_hash_rejected():
    with pytest.raises(ValueError, match="evaluation mode"):
        _make_ctx(_h_view(), identity=_identity(eval_mode="false"))
    with pytest.raises(ValueError, match="SHA256"):
        _make_ctx(_h_view(), identity=_identity(source_sha256="unknown"))


def test_inference_tensor_input_can_train_projection():
    with torch.inference_mode():
        context = _make_ctx(_h_view())
    nn.Linear(4, 2)(context.ab_features()).sum().backward()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_h_only_construction():
    _make_ctx(_h_view())


def test_hl_construction():
    _make_ctx(_hl_view())


def test_hlag_construction():
    _make_ctx(_hlag_view())


def test_missing_h_raises():
    # L without H is forbidden
    view = VisibleView(("L",), (("A", None, "C"),))
    n = len(view.coords())
    with pytest.raises(ValueError, match="H is mandatory"):
        StaticContext(view, torch.randn(n, 4), _identity())


def test_no_masked_positions_raises():
    view = VisibleView(("H",), (("A", "C", "D"),))  # nothing masked
    with pytest.raises(ValueError, match="masked Ab position"):
        StaticContext(view, torch.randn(3, 4), _identity())


def test_wrong_feature_shape_raises():
    view = _h_view()
    n = len(view.coords())
    with pytest.raises(ValueError, match="shape"):
        StaticContext(view, torch.randn(n + 1, 4), _identity())


def test_zero_width_features_raises():
    view = _h_view()
    n = len(view.coords())
    with pytest.raises(ValueError, match="width D"):
        StaticContext(view, torch.zeros(n, 0), _identity())


def test_non_floating_features_raises():
    view = _h_view()
    n = len(view.coords())
    with pytest.raises(ValueError, match="floating"):
        StaticContext(view, torch.zeros(n, 4, dtype=torch.int32), _identity())


def test_non_finite_features_raises():
    view = _h_view()
    n = len(view.coords())
    f = torch.randn(n, 4)
    f[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        StaticContext(view, f, _identity())


def test_missing_identity_key_raises():
    view = _h_view()
    n = len(view.coords())
    bad = _identity()
    del bad["extraction_layer"]
    with pytest.raises(ValueError, match="missing required keys"):
        StaticContext(view, torch.randn(n, 4), bad)


def test_empty_identity_value_raises():
    view = _h_view()
    n = len(view.coords())
    bad = _identity(tokenizer="")
    with pytest.raises(ValueError, match="non-empty string"):
        StaticContext(view, torch.randn(n, 4), bad)


# ---------------------------------------------------------------------------
# Identity determinism: same view, different hidden-gold variants => same key
# ---------------------------------------------------------------------------

def test_same_identity_different_gold_variants():
    """Content key must not encode gold labels; same view => same key regardless of
    which residues happened to be hidden when the view was constructed."""
    view = _h_view()
    # The view already has None at H:2 and H:3 (hidden union).
    # Imagine two distinct gold assignments: gold_A=(H:2->'A', H:3->'C') and
    # gold_B=(H:2->'G', H:3->'W'). Both produce the same VisibleView.
    n = len(view.coords())
    feats = torch.randn(n, 4)
    ident = _identity()

    ctx_a = StaticContext(view, feats, ident)
    ctx_b = StaticContext(view, feats, ident)

    assert ctx_a.content_key == ctx_b.content_key
    assert torch.allclose(ctx_a.features, ctx_b.features)


def test_content_key_changes_on_different_view():
    # Different visible Ab scaffold => different content key
    view1 = VisibleView(("H",), (("A", None, "C"),))
    view2 = VisibleView(("H",), (("A", None, "D"),))  # C vs D at position 2
    n1, n2 = len(view1.coords()), len(view2.coords())
    ident = _identity()
    ctx1 = StaticContext(view1, torch.randn(n1, 4), ident)
    ctx2 = StaticContext(view2, torch.randn(n2, 4), ident)
    assert ctx1.content_key != ctx2.content_key


def test_content_key_changes_on_visible_ag():
    """Changing the visible Ag sequence must produce a different content key."""
    ag_a = VisibleView(
        ("H", "L", "Ag"),
        (("A", None, "C"), ("D", "E", None), ("F", "G", "H")),
    )
    ag_b = VisibleView(
        ("H", "L", "Ag"),
        (("A", None, "C"), ("D", "E", None), ("F", "G", "W")),  # H->W at Ag:2
    )
    n_a, n_b = len(ag_a.coords()), len(ag_b.coords())
    ident = _identity()
    ctx_a = StaticContext(ag_a, torch.randn(n_a, 4), ident)
    ctx_b = StaticContext(ag_b, torch.randn(n_b, 4), ident)
    assert ctx_a.content_key != ctx_b.content_key


def test_content_key_changes_on_encoder_identity():
    view = _h_view()
    n = len(view.coords())
    ctx1 = StaticContext(view, torch.randn(n, 4), _identity(extraction_layer="layer_6"))
    ctx2 = StaticContext(view, torch.randn(n, 4), _identity(extraction_layer="layer_12"))
    assert ctx1.content_key != ctx2.content_key


def test_different_targets_different_identities():
    """H-only vs H+L caches are different contexts."""
    ctx_h = _make_ctx(_h_view())
    ctx_hl = _make_ctx(_hl_view())
    assert ctx_h.content_key != ctx_hl.content_key


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------

def test_h_only_cache_h_only_query_ok():
    ctx = _make_ctx(_h_view())
    # Query: same H length, mask only position 2 (inside hidden union)
    query = VisibleView(("H",), (("A", "C", None, "A", "D"),))
    ctx.check_query(query)


def test_h_only_cache_fully_revealed_query_ok():
    # Query that reveals all masked positions is legal
    ctx = _make_ctx(_h_view())
    query = VisibleView(("H",), (("A", "C", "G", "W", "D"),))
    ctx.check_query(query)


def test_hl_cache_hl_noag_query_ok():
    """H+L query against H+L+Ag cache: Ab-only query is allowed."""
    ctx = _make_ctx(_hlag_view())
    # Query has H+L but no Ag
    query = VisibleView(
        ("H", "L"),
        (("A", None, "C"), ("D", "E", None)),
    )
    ctx.check_query(query)


def test_hl_cache_hlag_query_ok():
    """Query with correct Ag against H+L+Ag cache is allowed."""
    ctx = _make_ctx(_hlag_view())
    query = VisibleView(
        ("H", "L", "Ag"),
        (("A", None, "C"), ("D", "E", None), ("F", "G", "H")),
    )
    ctx.check_query(query)


def test_query_missing_h_raises():
    ctx = _make_ctx(_hl_view())
    # L-only query has no H
    query = VisibleView(("L",), (("D", "E", None),))
    with pytest.raises(ValueError, match="H"):
        ctx.check_query(query)


def test_query_ab_role_mismatch_raises():
    ctx = _make_ctx(_h_view())  # H-only cache
    # Query with L (Ab role mismatch) — "X" is not a canonical residue so use "A"
    query = VisibleView(("H", "L"), (("A", "C", None, "A", "D"), ("A", "C")))
    with pytest.raises(ValueError, match="Ab roles"):
        ctx.check_query(query)


def test_query_length_mismatch_raises():
    ctx = _make_ctx(_h_view())
    # Query H is one residue shorter
    query = VisibleView(("H",), (("A", "C", None, "D"),))  # len 4 vs 5
    with pytest.raises(ValueError, match="length mismatch"):
        ctx.check_query(query)


def test_query_scaffold_change_raises():
    ctx = _make_ctx(_h_view())
    # Position H:0 is visible as 'A' in cache; query has 'G' there
    query = VisibleView(("H",), (("G", "C", None, None, "D"),))
    with pytest.raises(ValueError, match="scaffold mismatch"):
        ctx.check_query(query)


def test_query_masks_immutable_position_raises():
    ctx = _make_ctx(_h_view())
    # H:0 is visible/immutable in the cache (not in hidden union); masking it is foreign
    query = VisibleView(("H",), ((None, "C", None, None, "D"),))
    with pytest.raises(ValueError, match="outside the cache hidden union"):
        ctx.check_query(query)


def test_query_foreign_mask_raises():
    ctx = _make_ctx(_h_view())
    # H:4 is visible ('D') in cache, outside the hidden union {H:2, H:3}
    query = VisibleView(("H",), (("A", "C", None, None, None),))
    with pytest.raises(ValueError, match="outside the cache hidden union"):
        ctx.check_query(query)


def test_query_with_ag_but_cache_has_none_raises():
    ctx = _make_ctx(_hl_view())  # no Ag in cache
    # Query adds Ag - should fail
    query = VisibleView(
        ("H", "L", "Ag"),
        (("A", None, "C"), ("D", "E", None), ("F", "G")),
    )
    with pytest.raises(ValueError, match="no antigen context"):
        ctx.check_query(query)


def test_query_ag_mismatch_raises():
    ctx = _make_ctx(_hlag_view())
    # Query Ag differs at position 2 (H->'W')
    query = VisibleView(
        ("H", "L", "Ag"),
        (("A", None, "C"), ("D", "E", None), ("F", "G", "W")),
    )
    with pytest.raises(ValueError, match="Ag mismatch"):
        ctx.check_query(query)


def test_revealed_position_inside_union_is_legal():
    """A query may reveal a position that was masked in the cache (inside union)."""
    ctx = _make_ctx(_hlag_view())
    # H:1 is masked in cache (inside hidden union); query reveals it as 'G' - legal
    query = VisibleView(
        ("H", "L"),
        (("A", "G", "C"), ("D", "E", None)),
    )
    ctx.check_query(query)  # must not raise


def test_query_ab_features_returns_ab_rows_only():
    view = _hlag_view()
    n = len(view.coords())
    D = 8
    feats = torch.arange(n * D, dtype=torch.float32).reshape(n, D)
    ctx = StaticContext(view, feats, _identity())

    # Ab-only query
    query = VisibleView(("H", "L"), (("A", None, "C"), ("D", "E", None)))
    ab_feats = ctx.query_ab_features(query)

    # view.coords() order: H:0,H:1,H:2, L:0,L:1,L:2, Ag:0,Ag:1,Ag:2
    # Ab rows are indices 0..5 (6 rows)
    assert ab_feats.shape == (6, D)
    expected = feats[:6].clone()
    assert torch.allclose(ab_feats, expected)


# ---------------------------------------------------------------------------
# Mutation safety
# ---------------------------------------------------------------------------

def test_constructor_input_mutation_does_not_affect_cache():
    view = _h_view()
    n = len(view.coords())
    feats = torch.randn(n, 4)
    ident = _identity()
    ctx = StaticContext(view, feats, ident)

    feats[0, 0] = 999.0
    ident["extraction_layer"] = "tampered"

    assert ctx.features[0, 0] != 999.0
    assert ctx.encoder_identity["extraction_layer"] == "layer_12"


def test_accessor_mutation_does_not_affect_cache():
    ctx = _make_ctx(_h_view())

    exposed_feats = ctx.features
    exposed_feats[0, 0] = 999.0
    assert ctx.features[0, 0] != 999.0

    exposed_ident = ctx.encoder_identity
    exposed_ident["extraction_layer"] = "tampered"
    assert ctx.encoder_identity["extraction_layer"] == "layer_12"

    exposed_schema = ctx.schema
    exposed_schema["extra"] = "tampered"
    assert "extra" not in ctx.schema


def test_ab_features_clone_independence():
    ctx = _make_ctx(_h_view())
    ab1 = ctx.ab_features()
    ab1[0, 0] = 999.0
    ab2 = ctx.ab_features()
    assert ab2[0, 0] != 999.0


# ---------------------------------------------------------------------------
# Disk roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    ctx = _make_ctx(_hlag_view(), D=6)
    path = str(tmp_path / "ctx.sc")
    ctx.save(path)

    loaded = StaticContext.load(path)
    assert loaded.content_key == ctx.content_key
    assert loaded.view == ctx.view
    assert loaded.hidden_union == ctx.hidden_union
    assert torch.allclose(loaded.features, ctx.features)
    assert loaded.encoder_identity == ctx.encoder_identity
    assert loaded.schema == ctx.schema


def test_save_refuses_overwrite_by_default(tmp_path):
    path = str(tmp_path / "ctx.sc")
    ctx = _make_ctx(_h_view())
    ctx.save(path)
    with pytest.raises(FileExistsError):
        ctx.save(path)


def test_save_overwrite_flag_allows_replace(tmp_path):
    path = str(tmp_path / "ctx.sc")
    ctx = _make_ctx(_h_view())
    ctx.save(path)
    ctx.save(path, overwrite=True)  # should not raise


def test_load_corrupt_features_raises(tmp_path):
    ctx = _make_ctx(_h_view())
    path = str(tmp_path / "ctx.sc")
    ctx.save(path)

    # Corrupt one feature value without updating the stored hash
    data = torch.load(path, weights_only=True)
    data["features"][0, 0] = data["features"][0, 0] + 1.0
    torch.save(data, path)

    with pytest.raises(ValueError, match="feature bytes hash"):
        StaticContext.load(path)


def test_load_corrupt_identity_raises(tmp_path):
    ctx = _make_ctx(_h_view())
    path = str(tmp_path / "ctx.sc")
    ctx.save(path)

    data = torch.load(path, weights_only=True)
    data["meta"]["encoder_identity"]["extraction_layer"] = "layer_99"
    torch.save(data, path)

    with pytest.raises(ValueError, match="content key"):
        StaticContext.load(path)


def test_load_extra_field_raises(tmp_path):
    ctx = _make_ctx(_h_view())
    path = str(tmp_path / "ctx.sc")
    ctx.save(path)

    data = torch.load(path, weights_only=True)
    data["meta"]["__unexpected__"] = "injected"
    torch.save(data, path)

    with pytest.raises(ValueError, match="unexpected meta keys"):
        StaticContext.load(path)


def test_load_wrong_format_tag_raises(tmp_path):
    ctx = _make_ctx(_h_view())
    path = str(tmp_path / "ctx.sc")
    ctx.save(path)

    data = torch.load(path, weights_only=True)
    data["meta"]["_format"] = "Unknown/99"
    torch.save(data, path)

    with pytest.raises(ValueError, match="format tag"):
        StaticContext.load(path)


# ---------------------------------------------------------------------------
# Trainable projection / autograd
# ---------------------------------------------------------------------------

def _run_backward(ctx: StaticContext) -> None:
    """Pass Ab features through a small linear, compute a scalar loss, backward."""
    feats = ctx.ab_features()            # [n_ab, D] — cloned, plain CPU float
    D = feats.shape[1]
    proj = nn.Linear(D, 1, bias=False)
    out = proj(feats)                    # autograd graph from proj.weight through feats
    loss = out.sum()
    loss.backward()
    assert proj.weight.grad is not None


def test_trainable_linear_backward_in_memory():
    _run_backward(_make_ctx(_hl_view(), D=8))


def test_trainable_linear_backward_after_load(tmp_path):
    ctx = _make_ctx(_hl_view(), D=8)
    path = str(tmp_path / "ctx.sc")
    ctx.save(path)
    loaded = StaticContext.load(path)
    _run_backward(loaded)


# ---------------------------------------------------------------------------
# Feature dtype acceptance
# ---------------------------------------------------------------------------

def test_float16_features_accepted():
    view = _h_view()
    n = len(view.coords())
    StaticContext(view, torch.randn(n, 4).to(torch.float16), _identity())


def test_float64_features_accepted():
    view = _h_view()
    n = len(view.coords())
    StaticContext(view, torch.randn(n, 4).to(torch.float64), _identity())


def test_integer_features_rejected():
    view = _h_view()
    n = len(view.coords())
    with pytest.raises(ValueError, match="floating"):
        StaticContext(view, torch.zeros(n, 4, dtype=torch.long), _identity())


def test_inf_features_rejected():
    view = _h_view()
    n = len(view.coords())
    f = torch.randn(n, 4)
    f[1, 2] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        StaticContext(view, f, _identity())

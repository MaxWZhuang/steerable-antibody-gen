"""Pins the ESM-IF1 compatibility layer.

`fair-esm` 2.0.0 is archived and cannot be imported on a modern stack without
three unrelated patches (see `smallAntibodyGen.esmif1_compat`). The tests that
matter most here are not "does it import" but the two ways this layer could be
silently wrong:

- the `scatter_add` substitute computing something *close to* upstream rather
  than equal to it, which would corrupt the frozen structure encoder without
  raising; and
- the layer shadowing a real upstream package on a machine where one is
  installed, so that a box with the genuine extension silently runs ours.

Both are pinned below. Weight-loading is deliberately NOT exercised: the
checkpoint is a 540 MiB download and belongs in the readiness record, not the
suite.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch

from smallAntibodyGen.esmif1_compat import (
    CompatReport,
    alias_biotite_renames,
    install,
    install_scatter_substitute,
    scatter,
    scatter_add,
)


BIOTITE_AVAILABLE = importlib.util.find_spec("biotite") is not None
PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None
ESM_AVAILABLE = importlib.util.find_spec("esm") is not None
ESM_IF1_STACK = BIOTITE_AVAILABLE and PYG_AVAILABLE and ESM_AVAILABLE


@pytest.fixture
def restore_compat_state():
    """Undo whatever a test installs, so ordering cannot leak between tests."""
    had_module = "torch_scatter" in sys.modules
    previous = sys.modules.get("torch_scatter")
    saved: list[tuple[object, str, object]] = []
    if BIOTITE_AVAILABLE:
        import biotite.structure as bs
        from biotite.structure.io import pdbx

        for owner, name in ((bs, "filter_backbone"), (pdbx, "PDBxFile")):
            saved.append((owner, name, getattr(owner, name, None)))

    yield

    if had_module:
        sys.modules["torch_scatter"] = previous
    else:
        sys.modules.pop("torch_scatter", None)
    for owner, name, value in saved:
        if value is None:
            if hasattr(owner, name):
                delattr(owner, name)
        else:
            setattr(owner, name, value)


# --------------------------------------------------------------------------
# scatter_add: the one upstream function we reimplement
# --------------------------------------------------------------------------

def test_scatter_add_matches_bincount_for_indegree_counts():
    """The only call site in all of ESM (`gvp_modules.py:452`) counts node
    in-degree. `bincount` is an independent specification of that result."""
    dst = torch.tensor([0, 2, 2, 5, 5, 5, 5])
    got = scatter_add(torch.ones_like(dst), dst, dim_size=8)

    assert torch.equal(got, torch.bincount(dst, minlength=8))


def test_scatter_add_matches_naive_reference_on_2d_source():
    """Generalise past the one call site: a row-wise accumulation must equal an
    explicit Python loop, so a future caller cannot get a plausible-but-wrong
    answer out of the substitute."""
    src = torch.randn(7, 4)
    index = torch.tensor([0, 0, 3, 1, 3, 1, 0])

    got = scatter_add(src, index, dim=0, dim_size=5)

    expected = torch.zeros(5, 4)
    for row, target in enumerate(index.tolist()):
        expected[target] += src[row]
    assert torch.allclose(got, expected, atol=1e-6)


def test_scatter_add_respects_dim_size_beyond_max_index():
    """Trailing nodes with no incoming edges must still produce rows. Inferring
    the size from `index.max()` instead would silently shorten the output and
    misalign every downstream node feature."""
    src = torch.ones(3)
    index = torch.tensor([0, 1, 1])

    got = scatter_add(src, index, dim_size=6)

    assert got.shape == (6,)
    assert torch.equal(got, torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))


def test_scatter_add_handles_empty_index():
    src = torch.empty(0)
    index = torch.empty(0, dtype=torch.long)

    got = scatter_add(src, index, dim_size=4)

    assert torch.equal(got, torch.zeros(4))


def test_scatter_add_gradient_routes_to_the_source_rows():
    """Gradient must reach the scattered source. For `w . scatter_add(src)` the
    exact derivative w.r.t. `src[i]` is `w[index[i]]`, so this is checked against
    a closed form rather than a tolerance on finite differences."""
    src = torch.randn(5, requires_grad=True)
    index = torch.tensor([2, 0, 2, 1, 0])
    weight = torch.tensor([10.0, 20.0, 30.0, 40.0])

    (scatter_add(src, index, dim_size=4) * weight).sum().backward()

    assert torch.equal(src.grad, weight[index])


def test_scatter_rejects_an_unsupported_reduction():
    """ESM never calls `scatter`, so anything but a sum is unimplemented. It must
    say so rather than quietly summing and returning a wrong answer."""
    with pytest.raises(NotImplementedError, match="reduce='mean'"):
        scatter(torch.ones(3), torch.tensor([0, 1, 1]), dim_size=2, reduce="mean")


# --------------------------------------------------------------------------
# install(): must be idempotent and must never shadow a genuine upstream
# --------------------------------------------------------------------------

def test_install_scatter_substitute_is_idempotent(restore_compat_state, monkeypatch):
    """Installing twice must reuse the first module rather than rebuilding it.

    Absence of upstream is established explicitly: on a machine that HAS the
    genuine extension the installer correctly skips, and asserting "installed"
    unconditionally would fail there for the right reason.
    """
    import smallAntibodyGen.esmif1_compat as compat

    sys.modules.pop("torch_scatter", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        compat.importlib.util,
        "find_spec",
        lambda name, *a, **k: (None if name == "torch_scatter"
                               else real_find_spec(name, *a, **k)),
    )

    first = install_scatter_substitute()
    module = sys.modules["torch_scatter"]
    second = install_scatter_substitute()

    assert first == "installed"
    assert second == "already installed"
    assert sys.modules["torch_scatter"] is module


def test_install_defers_to_a_real_torch_scatter(restore_compat_state):
    """On a box where the compiled extension exists, upstream wins. Ours is a
    substitute for an unavailable dependency, never a replacement for a present
    one -- silently preferring ours would make two machines disagree."""
    genuine = types.ModuleType("torch_scatter")
    genuine.__file__ = "/somewhere/real/torch_scatter/__init__.py"
    genuine.scatter_add = object()
    sys.modules["torch_scatter"] = genuine

    outcome = install_scatter_substitute()

    assert outcome == "skipped: upstream torch_scatter is present"
    assert sys.modules["torch_scatter"] is genuine


def test_install_defers_to_an_importable_but_unimported_torch_scatter(
    restore_compat_state, monkeypatch
):
    """The `sys.modules` check above only fires once upstream has been imported.
    On a machine where the compiled extension is installed but untouched, the
    spec lookup is what must keep us from pre-empting it."""
    import smallAntibodyGen.esmif1_compat as compat

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        compat.importlib.util,
        "find_spec",
        lambda name, *a, **k: (object() if name == "torch_scatter"
                               else real_find_spec(name, *a, **k)),
    )

    outcome = compat.install_scatter_substitute()

    assert outcome == "skipped: upstream torch_scatter is present"
    assert "torch_scatter" not in sys.modules


@pytest.mark.skipif(not BIOTITE_AVAILABLE, reason="biotite is not installed")
def test_alias_biotite_renames_reports_what_it_aliased(restore_compat_state):
    import biotite.structure as bs

    aliased = alias_biotite_renames()

    assert hasattr(bs, "filter_backbone")
    # biotite 1.x needs both aliases; on a 0.x install neither is needed.
    assert set(aliased) <= {"filter_backbone", "PDBxFile"}


@pytest.mark.skipif(not BIOTITE_AVAILABLE, reason="biotite is not installed")
def test_alias_biotite_renames_does_not_clobber_an_existing_name(restore_compat_state):
    """A biotite 0.x install already has the real `filter_backbone`. Overwriting
    it with the 1.x function would swap the atom selection underneath ESM."""
    import biotite.structure as bs

    sentinel = object()
    bs.filter_backbone = sentinel

    aliased = alias_biotite_renames()

    assert bs.filter_backbone is sentinel
    assert "filter_backbone" not in aliased


def test_install_raises_a_named_error_when_biotite_is_absent(
    restore_compat_state, monkeypatch
):
    """`install` documents that it raises rather than deferring the failure to an
    opaque ImportError later. Pinned by BLOCKING the import rather than by
    relying on the dependency being absent, so the contract is asserted on every
    box instead of only on ones that happen to lack biotite -- which is where CI
    (`pip install -e ".[dev]"`) actually sits.
    """
    class BiotiteBlocker:
        def find_spec(self, name, path=None, target=None):
            if name == "biotite" or name.startswith("biotite."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    for cached in [n for n in sys.modules
                   if n == "biotite" or n.startswith("biotite.")]:
        monkeypatch.delitem(sys.modules, cached)
    monkeypatch.setattr(sys, "meta_path", [BiotiteBlocker(), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError, match="biotite"):
        install()


@pytest.mark.skipif(not BIOTITE_AVAILABLE, reason="biotite is not installed")
def test_install_returns_a_report_of_all_three_patches(restore_compat_state):
    report = install()

    assert isinstance(report, CompatReport)
    assert report.scatter in ("installed", "already installed",
                              "skipped: upstream torch_scatter is present")
    assert isinstance(report.biotite_aliases, tuple)
    assert report.checkpoint_globals in ("allowed", "already allowed",
                                         "unsupported: torch has no add_safe_globals")


# --------------------------------------------------------------------------
# The payoff: the real upstream package imports and runs
# --------------------------------------------------------------------------

@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_install_makes_esm_inverse_folding_importable():
    install()

    import esm.inverse_folding  # noqa: F401  -- the import IS the assertion

    assert esm.inverse_folding.gvp_transformer.GVPTransformerModel is not None


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_gvp_conv_layer_runs_on_the_autoregressive_branch():
    """End-to-end through the real upstream layer, on the one branch that reaches
    the substituted call. Needs no checkpoint -- the layer is built from scratch."""
    install()
    from esm.inverse_folding.gvp_modules import GVPConvLayer

    torch.manual_seed(0)
    node_dims, edge_dims, n_nodes, n_edges = (16, 4), (8, 1), 12, 40
    layer = GVPConvLayer(node_dims, edge_dims, autoregressive=True).eval()
    x = (torch.randn(n_nodes, node_dims[0]), torch.randn(n_nodes, node_dims[1], 3))
    autoregressive_x = (torch.randn(n_nodes, node_dims[0]),
                        torch.randn(n_nodes, node_dims[1], 3))
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = (torch.randn(n_edges, edge_dims[0]),
                 torch.randn(n_edges, edge_dims[1], 3))

    (scalars, vectors), _ = layer(x, edge_index, edge_attr,
                                  autoregressive_x=autoregressive_x)

    assert scalars.shape == (n_nodes, node_dims[0])
    assert vectors.shape == (n_nodes, node_dims[1], 3)
    assert torch.isfinite(scalars).all() and torch.isfinite(vectors).all()

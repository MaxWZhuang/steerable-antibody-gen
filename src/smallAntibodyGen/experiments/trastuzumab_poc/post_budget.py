"""Canonical, outcome-free identity of the selected post-training fitting budget.

The record is built from the actual validated ``SelectedBudget`` (never from a caller
dictionary).  It keeps the selected IDs in canonical selection order, because
``SelectedBudget.selection_digest`` hashes the SORTED IDs and so cannot show that order.

Data-free validation proves what it can prove without the endpoint manifest: the shape of
the record, the ``(sha256(seed NUL id), id)`` order of the selected IDs and the selection
digest.  It does NOT prove that the IDs are the prefix of the manifest's post_pool order,
because the candidates that were left out are unknown here.  ``RestrictedPostData`` checked
that prefix against its init-only manifest when the run started.

No Torch dependency.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

from smallAntibodyGen.experiments.trastuzumab_poc.protocol import SelectedBudget

POST_BUDGET_SCHEMA = "trastuzumab-post-budget/1"
_KEYS = frozenset({"schema", "manifest_digest", "budget_seed", "budget_n", "selected_ids",
                   "selection_digest"})
_HEX = frozenset("0123456789abcdef")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{name} must be a lowercase SHA256")


def selection_key(budget_seed: str, endpoint_id: str) -> Tuple[str, str]:
    """The order ``protocol.select_budget`` sorts post_pool endpoints by."""
    return (hashlib.sha256(f"{budget_seed}\x00{endpoint_id}".encode("utf-8")).hexdigest(), endpoint_id)


def _selection_digest(manifest_digest: str, budget_seed: str, selected_ids) -> str:
    return hashlib.sha256(json.dumps(
        {"budget_seed": budget_seed, "selected": sorted(selected_ids), "manifest_digest": manifest_digest},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_post_budget_identity(identity) -> Tuple[str, ...]:
    """Data-free validation; returns the selected IDs in their canonical order."""
    if not isinstance(identity, dict) or set(identity) != _KEYS:
        raise ValueError(f"post budget identity: expected exactly {sorted(_KEYS)}")
    if identity["schema"] != POST_BUDGET_SCHEMA:
        raise ValueError("unsupported post budget identity schema")
    _sha(identity["manifest_digest"], "post budget manifest_digest")
    _sha(identity["selection_digest"], "post budget selection_digest")
    seed, n, ids = identity["budget_seed"], identity["budget_n"], identity["selected_ids"]
    if not isinstance(seed, str) or not seed:
        raise ValueError("post budget seed must be a nonempty string")
    if type(n) is not int or n < 1:
        raise ValueError("post budget size must be a positive integer")
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("post budget selected_ids must be a list of nonempty strings")
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("post budget selected_ids must be unique and match the budget size")
    order = [selection_key(seed, i) for i in ids]
    if any(left >= right for left, right in zip(order, order[1:])):
        raise ValueError("post budget selected_ids are not in the seeded selection order")
    if identity["selection_digest"] != _selection_digest(identity["manifest_digest"], seed, ids):
        raise ValueError("post budget selection digest does not bind the seed, IDs and manifest")
    return tuple(ids)


def post_budget_identity(budget: SelectedBudget) -> Dict[str, Any]:
    """Plain-JSON identity of the budget a post run actually fits; values are never part of it."""
    if not isinstance(budget, SelectedBudget):
        raise TypeError("budget must be a SelectedBudget")
    identity = {"schema": POST_BUDGET_SCHEMA, "manifest_digest": budget.manifest_digest,
                "budget_seed": budget.budget_seed, "budget_n": budget.budget_n,
                "selected_ids": list(budget.selected_ids), "selection_digest": budget.selection_digest}
    validate_post_budget_identity(identity)
    return identity

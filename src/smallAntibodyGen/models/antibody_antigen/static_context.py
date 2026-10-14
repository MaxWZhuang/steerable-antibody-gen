"""Static antigen-conditioned (or antibody-only) residue feature cache.

A :class:`StaticContext` holds per-residue encoder features for a fixed view whose
masked positions form the editable-Ab hidden union. A query :class:`VisibleView` may
borrow those features without a second encoder call once it passes structural
compatibility checks.

No gold residues, labels, assay IDs, or clean-sequence hashes are stored or accepted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch

from smallAntibodyGen.models.antibody_antigen.views import (
    Coord,
    VisibleView,
)

_REQUIRED_IDENTITY_KEYS = frozenset({
    "source_sha256",
    "checkpoint_sha256",
    "extraction_layer",
    "precision",
    "tokenizer",
    "eval_mode",
})

_SAVED_META_KEYS = frozenset({
    "_format",
    "content_key",
    "feature_bytes_hash",
    "view_roles",
    "view_chains",
    "encoder_identity",
    "schema",
})


def _validate_encoder_identity(identity: object) -> Dict[str, str]:
    if not isinstance(identity, dict):
        raise TypeError("encoder_identity must be a dict")
    missing = _REQUIRED_IDENTITY_KEYS - identity.keys()
    if missing:
        raise ValueError(f"encoder_identity missing required keys: {sorted(missing)}")
    if set(identity) != _REQUIRED_IDENTITY_KEYS:
        raise ValueError("encoder_identity has unexpected fields")
    result: Dict[str, str] = {}
    for key, val in identity.items():
        if not isinstance(val, str) or not val:
            raise ValueError(
                f"encoder_identity[{key!r}] must be a non-empty string, got {val!r}"
            )
        result[key] = val
    for key in ("source_sha256", "checkpoint_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", result[key]) is None:
            raise ValueError(f"{key} must be a lowercase SHA256 digest")
    if result["eval_mode"] != "true":
        raise ValueError("static features must be extracted in evaluation mode")
    if re.fullmatch(r"layer_[1-9][0-9]*", result["extraction_layer"]) is None:
        raise ValueError("extraction_layer must be layer_<positive integer>")
    if result["precision"] not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValueError("unknown encoder precision")
    return dict(sorted(result.items()))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _compute_content_key(
    view: VisibleView,
    hidden_union: Tuple[Coord, ...],
    encoder_identity: Dict[str, Any],
    schema: Dict[str, Any],
) -> str:
    payload = _canonical_json({
        "view": view.content_digest(),
        "hidden_union": [[role, idx] for role, idx in hidden_union],
        "encoder_identity": encoder_identity,
        "schema": schema,
    })
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _feature_bytes_hash(features: torch.Tensor) -> str:
    t = features.contiguous()
    if t.dtype == torch.bfloat16:
        # numpy does not support bfloat16; reinterpret as int16 (same width)
        t = t.view(torch.int16)
    descriptor = _canonical_json({"shape": list(features.shape), "dtype": str(features.dtype)})
    return hashlib.sha256(descriptor.encode("ascii") + b"\0" + t.numpy().tobytes()).hexdigest()


class StaticContext:
    """Frozen per-residue features for a fixed visible/hidden complex layout.

    H is mandatory; L and Ag are optional.  The caller supplies a :class:`VisibleView`
    where every editable Ab position is masked; those masked positions form the cache's
    *hidden union*.  Features are a ``[len(view.coords()), D]`` floating CPU tensor.

    Ag rows are present in the stored tensor when Ag appears in the view, but
    :meth:`ab_features` and :meth:`query_ab_features` always return only Ab rows,
    aligned by ``(role, index)``, never by tokeniser offsets.
    """

    def __init__(
        self,
        view: VisibleView,
        features: torch.Tensor,
        encoder_identity: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(view, VisibleView):
            raise TypeError("view must be a VisibleView")
        if not view.has_role("H"):
            raise ValueError("H is mandatory in a StaticContext")

        hidden = view.hidden_sites()
        if not hidden:
            raise ValueError("view must have at least one masked Ab position")

        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        n = len(view.coords())
        if features.dim() != 2 or features.shape[0] != n:
            raise ValueError(
                f"features must have shape [{n}, D], got {tuple(features.shape)}"
            )
        if features.shape[1] == 0:
            raise ValueError("feature width D must be > 0")
        if not features.is_floating_point():
            raise ValueError("features must be a floating-point tensor")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must all be finite")
        if features.device.type != "cpu":
            raise ValueError("features must reside on CPU")

        identity = _validate_encoder_identity(encoder_identity)

        derived_schema = {"version": "static-context/1", "feature_shape": list(features.shape),
                          "feature_dtype": str(features.dtype)}
        if schema is not None and schema != derived_schema:
            raise ValueError("schema must equal the derived feature representation")
        schema = derived_schema

        self._view = view
        self._hidden_union: Tuple[Coord, ...] = hidden
        with torch.inference_mode(False), torch.no_grad():
            self._features = features.clone().detach()
        self._encoder_identity = identity
        self._schema = schema
        self._content_key = _compute_content_key(view, hidden, identity, schema)
        self._feature_hash = _feature_bytes_hash(self._features)

    # --- read-only accessors ---

    @property
    def view(self) -> VisibleView:
        return self._view

    @property
    def hidden_union(self) -> Tuple[Coord, ...]:
        return self._hidden_union

    @property
    def encoder_identity(self) -> Dict[str, Any]:
        return copy.deepcopy(self._encoder_identity)

    @property
    def schema(self) -> Dict[str, Any]:
        return copy.deepcopy(self._schema)

    @property
    def content_key(self) -> str:
        return self._content_key

    @property
    def features(self) -> torch.Tensor:
        """Cloned copy of all feature rows; mutating it does not affect the cache."""
        return self._features.clone()

    # --- feature extraction ---

    def ab_features(self) -> torch.Tensor:
        """Rows for H (and L if present) in canonical ``(role, index)`` order.

        Antigen rows are excluded even when the cache includes Ag context.
        """
        ab_rows = [
            i for i, (role, _) in enumerate(self._view.coords()) if role != "Ag"
        ]
        return self._features[ab_rows].clone()

    # --- query validation ---

    def check_query(self, query: VisibleView) -> None:
        """Raise if ``query`` is structurally incompatible with this cache.

        Checks performed (all before any model call):
        - H present in query.
        - Ab roles (H, optional L) identical to cache.
        - H and L lengths identical to cache.
        - Positions immutable in the cache (visible) carry the same residue in query.
        - Query hidden positions are a subset of the cache hidden union.
        - If query has Ag: cache must have Ag and sequences must match exactly.
          If cache has no Ag: query must not introduce Ag.
        """
        if not isinstance(query, VisibleView):
            raise TypeError("query must be a VisibleView")
        if not query.has_role("H"):
            raise ValueError("query must include H")

        cache_ab_roles = tuple(r for r in self._view.roles if r != "Ag")
        query_ab_roles = tuple(r for r in query.roles if r != "Ag")
        if cache_ab_roles != query_ab_roles:
            raise ValueError(
                f"Ab roles mismatch: cache has {cache_ab_roles}, query has {query_ab_roles}"
            )

        for role in cache_ab_roles:
            cl = len(self._view.chain(role))
            ql = len(query.chain(role))
            if cl != ql:
                raise ValueError(
                    f"chain {role!r} length mismatch: cache {cl}, query {ql}"
                )

        cache_hidden_set = set(self._hidden_union)

        # Reject any query masked position that is not in the cache's hidden union.
        # This covers both "masking an immutable position" and "masking a position the
        # cache never saw" — both are the same category: outside the union.
        foreign = set(query.hidden_sites()) - cache_hidden_set
        if foreign:
            raise ValueError(
                f"query masks positions outside the cache hidden union: {sorted(foreign)}"
            )

        # Scaffold consistency: immutable positions must carry the same residue.
        # The foreign-mask check above guarantees that every non-hidden query token is
        # a residue (not None), so the != comparison is safe.
        for role in cache_ab_roles:
            for i, (ct, qt) in enumerate(
                zip(self._view.chain(role), query.chain(role))
            ):
                coord: Coord = (role, i)
                if coord not in cache_hidden_set and qt != ct:
                    raise ValueError(
                        f"scaffold mismatch at {coord!r}: cache {ct!r}, query {qt!r}"
                    )
                # Masked in cache: query may mask or reveal — both legal

        cache_has_ag = self._view.has_role("Ag")
        query_has_ag = query.has_role("Ag")
        if query_has_ag and not cache_has_ag:
            raise ValueError("query introduces Ag but the cache has no antigen context")
        if query_has_ag and cache_has_ag:
            cag, qag = self._view.chain("Ag"), query.chain("Ag")
            if len(cag) != len(qag):
                raise ValueError(
                    f"Ag length mismatch: cache {len(cag)}, query {len(qag)}"
                )
            for i, (ct, qt) in enumerate(zip(cag, qag)):
                if qt != ct:
                    raise ValueError(
                        f"Ag mismatch at (Ag, {i}): cache {ct!r}, query {qt!r}"
                    )

    def query_ab_features(self, query: VisibleView) -> torch.Tensor:
        """Features for Ab positions after compatibility check.

        Rows are aligned to ``(role, index)`` order.  Antigen rows are excluded.
        """
        self.check_query(query)
        return self.ab_features()

    # --- persistence ---

    def save(self, path: str, *, overwrite: bool = False) -> None:
        """Atomically write to ``path``.

        Raises :exc:`FileExistsError` when the target exists and ``overwrite=False``.
        """
        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite existing file: {path!r}")
        meta = {
            "_format": "StaticContext/1",
            "content_key": self._content_key,
            "feature_bytes_hash": self._feature_hash,
            "view_roles": list(self._view.roles),
            "view_chains": [
                ["__mask__" if t is None else t for t in chain]
                for chain in self._view.chains
            ],
            "encoder_identity": self._encoder_identity,
            "schema": self._schema,
        }
        dirpath = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".sc.tmp")
        try:
            os.close(fd)
            torch.save({"meta": meta, "features": self._features}, tmp)
            if overwrite:
                os.replace(tmp, path)
            else:
                os.link(tmp, path)
                os.unlink(tmp)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "StaticContext":
        """Load from ``path``, verify integrity, and return a :class:`StaticContext`.

        Raises :exc:`ValueError` on schema, content-key, or feature-bytes mismatch.
        """
        data = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(data, dict) or set(data.keys()) != {"meta", "features"}:
            raise ValueError(
                "file must contain exactly 'meta' and 'features' at the top level"
            )
        meta, features = data["meta"], data["features"]

        if not isinstance(meta, dict):
            raise ValueError("meta must be a dict")
        if set(meta.keys()) != _SAVED_META_KEYS:
            raise ValueError(f"unexpected meta keys in file: {set(meta.keys())}")
        if meta["_format"] != "StaticContext/1":
            raise ValueError(f"unknown format tag: {meta['_format']!r}")

        view = VisibleView(
            tuple(meta["view_roles"]),
            tuple(
                tuple(None if t == "__mask__" else t for t in ch)
                for ch in meta["view_chains"]
            ),
        )
        identity = _validate_encoder_identity(meta["encoder_identity"])
        schema = meta["schema"]
        if not isinstance(schema, dict):
            raise ValueError("schema in saved file must be a dict")

        hidden = view.hidden_sites()
        expected_key = _compute_content_key(view, hidden, identity, schema)
        if meta["content_key"] != expected_key:
            raise ValueError(
                "content key mismatch — view, identity or schema may be corrupted"
            )

        if not isinstance(features, torch.Tensor):
            raise ValueError("features must be a tensor")
        if not features.is_floating_point():
            raise ValueError("loaded features must be a floating-point tensor")
        if meta["feature_bytes_hash"] != _feature_bytes_hash(features):
            raise ValueError("feature bytes hash mismatch — features may be corrupted")

        # Construct via __init__ to re-run all validation on the loaded data.
        return cls(view, features, identity, schema)

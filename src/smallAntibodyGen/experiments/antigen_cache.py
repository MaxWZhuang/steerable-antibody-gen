"""
Cache frozen-ESM antigen encodings for J24's ESM arm.

Only the FROZEN backbone's output is cacheable. Everything downstream of it --
the projection, the cross-attention, the fusion, the heads -- trains, so its
output changes every step and caching it would silently freeze a trainable part
of the model. The cache therefore stores exactly the tensor
``ESMAntigenEncoder`` produces before its projection is applied, and nothing
further.

The corpus makes this worth doing: 828,315 antibody-antigen rows carry only ~3,176
distinct antigens, and the largest single antigen covers 63% of rows. The frozen
backbone would otherwise re-encode the same ~1000-token sequence millions of
times.

The dangerous failure is not a slow cache but a STALE one -- an entry computed
under a different model, truncation, or tokenizer, reused silently, producing a
number that looks like a result. So the key commits to every input that can
change the value, and a mismatch is a miss rather than a repair.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

CACHE_FORMAT_VERSION = "antigen-embedding-cache/1"


@dataclass(frozen=True)
class AntigenCacheKey:
    """
    Everything that can change a frozen antigen encoding.

    Each field is here because changing it alone changes the tensor:

    - ``esm_model_name``: different weights.
    - ``tokenizer_signature``: different token ids for the same residues.
    - ``token_budget`` and ``residue_budget``: different truncation, so a
      different sequence reaches the backbone. Both are recorded even though one
      usually determines the other, because J24 crops in residue space and the
      relationship between them is exactly what that cropping changes.
    - ``sequence_sha256``: the residues themselves.
    - ``dtype``: an fp16 encoding is not an fp32 one, and AMP makes that a real
      possibility rather than a hypothetical.

    ``format_version`` is separate from the content so that a change to how the
    cache is SERIALIZED invalidates every entry without pretending the underlying
    encoding changed.
    """

    esm_model_name: str
    tokenizer_signature: str
    token_budget: int
    residue_budget: int
    sequence_sha256: str
    dtype: str
    format_version: str = CACHE_FORMAT_VERSION

    def digest(self) -> str:
        """A filesystem-safe digest of the whole key."""
        payload = json.dumps(
            {
                "esm_model_name": self.esm_model_name,
                "tokenizer_signature": self.tokenizer_signature,
                "token_budget": self.token_budget,
                "residue_budget": self.residue_budget,
                "sequence_sha256": self.sequence_sha256,
                "dtype": self.dtype,
                "format_version": self.format_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequence_digest(sequence: str) -> str:
    """SHA-256 of the exact residue string that will be encoded."""
    return hashlib.sha256((sequence or "").encode("utf-8")).hexdigest()


class FrozenAntigenCache:
    """
    An in-memory cache of frozen antigen encodings, optionally backed by disk.

    Deliberately not an LRU: entries are keyed by antigen, the corpus has ~3,176
    of them, and silently evicting one would make throughput depend on iteration
    order. If memory ever becomes the constraint, cap the cache explicitly and
    report the cap -- a silent cap is the "no silent truncation" rule again.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def _disk_path(self, key: AntigenCacheKey) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / f"{key.digest()}.pt"

    def get(self, key: AntigenCacheKey) -> torch.Tensor | None:
        """Return a cached encoding, or ``None``. A key mismatch is a miss."""
        digest = key.digest()
        cached = self._memory.get(digest)
        if cached is not None:
            self.hits += 1
            return cached
        path = self._disk_path(key)
        if path is not None and path.exists():
            payload = torch.load(path, map_location="cpu")
            # The key is stored ALONGSIDE the tensor and re-checked. A digest
            # collision or a hand-copied file would otherwise be indistinguishable
            # from a hit, and this cache exists precisely to be trustworthy.
            if payload.get("key") == key.digest():
                tensor = payload["encoding"]
                self._memory[digest] = tensor
                self.hits += 1
                return tensor
        self.misses += 1
        return None

    def put(self, key: AntigenCacheKey, encoding: torch.Tensor) -> None:
        """Store an encoding, in memory and on disk when a directory is set."""
        digest = key.digest()
        detached = encoding.detach().cpu()
        self._memory[digest] = detached
        path = self._disk_path(key)
        if path is not None:
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"key": digest, "encoding": detached}, tmp)
            tmp.replace(path)  # atomic: a killed run must not leave a torn entry

    def get_or_compute(
        self,
        key: AntigenCacheKey,
        compute: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Cached encoding if present, otherwise compute, store, and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        encoding = compute()
        self.put(key, encoding)
        return encoding

    def stats(self) -> dict[str, Any]:
        """Hit/miss counts, for the J24 report's cache-cost column."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": total,
            "hit_rate": round(self.hits / total, 6) if total else 0.0,
            "entries": len(self._memory),
        }

"""Benchmark provenance, schemas, splitters, and metrics.

J05a populates only the provenance half: :mod:`smallAntibodyGen.benchmarks.provenance`
holds pure-stdlib schemas and validators for benchmark source manifests and assay
records. Nothing in this package downloads anything, and nothing here reads raw
data -- raw downloads stay outside Git and are described by the manifests tracked
under ``specs/benchmarks/``.

The oracle (J05b), the AVIDa splitter (J05c), the Open AlphaSeq audit (J05d), and
the evaluator entry points are deliberately absent.
"""
from __future__ import annotations

__all__ = ["provenance"]

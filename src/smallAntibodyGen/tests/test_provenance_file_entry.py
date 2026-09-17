"""Exact digest validation for generic local-file provenance."""

import pytest

from smallAntibodyGen.benchmarks.provenance import FileEntry, ManifestValidationError


@pytest.mark.parametrize("suffix", ["\n", "\r\n", " "])
def test_file_entry_rejects_digest_suffixes(suffix):
    digest = "ab" * 32
    assert FileEntry("toy.json", 0, digest).sha256 == digest
    with pytest.raises(ManifestValidationError, match="64 lowercase hex"):
        FileEntry("toy.json", 0, digest + suffix)

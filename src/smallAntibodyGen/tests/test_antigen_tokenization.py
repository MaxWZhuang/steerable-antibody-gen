from __future__ import annotations

import importlib.util

import pytest

from smallAntibodyGen.antigen_tokenization import (
    ESMAntigenTokenizer,
    ScratchAntigenTokenizer,
    build_antigen_tokenizer,
)
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDK"


def test_scratch_adapter_matches_inline_encode_sequence():
    """Routing the antigen through the scratch adapter must be a no-op vs the old
    inline call, so the from-scratch model is byte-for-byte unaffected."""
    tokenizer = AminoAcidTokenizer()
    adapter = ScratchAntigenTokenizer(tokenizer)

    for max_length in (8, 24, 64):
        assert adapter.encode(ANTIGEN, max_length) == tokenizer.encode_sequence(
            ANTIGEN, locus=None, max_length=max_length
        )


def test_scratch_adapter_pad_id_and_cls_position():
    tokenizer = AminoAcidTokenizer()
    adapter = ScratchAntigenTokenizer(tokenizer)

    ids = adapter.encode(ANTIGEN, max_length=64)
    assert adapter.pad_id == tokenizer.pad_id
    assert ids[0] == tokenizer.cls_id  # CLS-like summary token at index 0


def test_scratch_adapter_handles_empty_sequence():
    tokenizer = AminoAcidTokenizer()
    adapter = ScratchAntigenTokenizer(tokenizer)
    # [CLS] [OTHER_CHAIN] [EOS] with no residues.
    assert adapter.encode("", max_length=64) == tokenizer.encode_sequence(
        "", locus=None, max_length=64
    )


def test_build_antigen_tokenizer_selects_scratch():
    tokenizer = AminoAcidTokenizer()
    adapter = build_antigen_tokenizer("scratch", tokenizer, "facebook/esm2_t6_8M_UR50D")
    assert isinstance(adapter, ScratchAntigenTokenizer)


def test_build_antigen_tokenizer_rejects_unknown_type():
    tokenizer = AminoAcidTokenizer()
    with pytest.raises(ValueError, match="unknown antigen_encoder_type"):
        build_antigen_tokenizer("bogus", tokenizer, "facebook/esm2_t6_8M_UR50D")


@pytest.mark.skipif(
    TRANSFORMERS_AVAILABLE,
    reason="transformers installed; the missing-extra error path cannot be exercised",
)
def test_esm_adapter_without_transformers_raises_helpful_error():
    with pytest.raises(ImportError, match="esm"):
        ESMAntigenTokenizer("facebook/esm2_t6_8M_UR50D")


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="transformers not installed (optional 'esm' extra)",
)
def test_esm_adapter_encodes_with_cls_and_respects_max_length():
    try:
        adapter = build_antigen_tokenizer(
            "esm", AminoAcidTokenizer(), "facebook/esm2_t6_8M_UR50D"
        )
    except OSError:
        pytest.skip("ESM tokenizer weights unavailable (offline)")

    ids = adapter.encode(ANTIGEN, max_length=16)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert len(ids) <= 16
    assert isinstance(adapter.pad_id, int)

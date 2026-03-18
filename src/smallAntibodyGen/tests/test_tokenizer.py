import pytest

from smallAntibodyGen.tokenizer import AminoAcidTokenizer


def test_vocab_contains_special_and_chain_tokens():
    tok = AminoAcidTokenizer()

    assert "[PAD]" in tok.token_to_id
    assert "[CLS]" in tok.token_to_id
    assert "[EOS]" in tok.token_to_id
    assert "[MASK]" in tok.token_to_id
    assert "[UNK]" in tok.token_to_id

    assert "[IGH]" in tok.token_to_id
    assert "[IGK]" in tok.token_to_id
    assert "[IGL]" in tok.token_to_id
    assert "[OTHER_CHAIN]" in tok.token_to_id


def test_vocab_contains_canonical_amino_acids():
    tok = AminoAcidTokenizer()

    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert aa in tok.token_to_id


def test_chain_token_mapping():
    tok = AminoAcidTokenizer()

    assert tok.get_chain_token("IGH") == "[IGH]"
    assert tok.get_chain_token("IGK") == "[IGK]"
    assert tok.get_chain_token("IGL") == "[IGL]"
    assert tok.get_chain_token("weird") == "[OTHER_CHAIN]"
    assert tok.get_chain_token(None) == "[OTHER_CHAIN]"


def test_encode_sequence_adds_cls_chain_and_eos():
    tok = AminoAcidTokenizer()

    ids = tok.encode_sequence("CARDRST", locus="IGH")

    assert ids[0] == tok.cls_id
    assert ids[1] == tok.token_to_id["[IGH]"]
    assert ids[-1] == tok.eos_id


def test_encode_sequence_is_case_insensitive():
    tok = AminoAcidTokenizer()

    ids_upper = tok.encode_sequence("CARDRST", locus="IGH")
    ids_lower = tok.encode_sequence("cardrst", locus="IGH")

    assert ids_upper == ids_lower


def test_unknown_characters_become_unk():
    tok = AminoAcidTokenizer()

    ids = tok.encode_sequence("CAR*DR", locus="IGH")
    assert tok.unk_id in ids


def test_truncation_preserves_eos():
    tok = AminoAcidTokenizer()

    ids = tok.encode_sequence("ACDEFGHIK", locus="IGH", max_length=5)

    assert len(ids) == 5
    assert ids[0] == tok.cls_id
    assert ids[1] == tok.token_to_id["[IGH]"]
    assert ids[-1] == tok.eos_id


def test_roundtrip_decode_clean_sequence():
    tok = AminoAcidTokenizer()

    seq = "CARDRST"
    ids = tok.encode_sequence(seq, locus="IGH")
    decoded = tok.decode_ids(ids)

    assert decoded == seq


def test_save_and_reload_vocab(tmp_path):
    tok = AminoAcidTokenizer()

    vocab_path = tmp_path / "vocab.txt"
    tok.save_vocab(str(vocab_path))

    tok2 = AminoAcidTokenizer.from_vocab_file(str(vocab_path))

    assert tok.vocab == tok2.vocab
    assert tok.token_to_id == tok2.token_to_id
    assert tok.id_to_token == tok2.id_to_token
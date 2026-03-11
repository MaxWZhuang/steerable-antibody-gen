from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class AminoAcidTokenizer:
    """Simple fixed-vocabulary tokenizer for antibody amino-acid sequences.

    The vocabulary is deliberately small and explicit so you can understand
    exactly what is going into the model.
    """

    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    eos_token: str = "[EOS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    unk_token: str = "[UNK]"
    chain_tokens: List[str] = field(
        default_factory=lambda: ["[IGH]", "[IGK]", "[IGL]", "[OTHER_CHAIN]"]
    )

    def __post_init__(self) -> None:
        aa_tokens = list("ACDEFGHIKLMNPQRSTVWY") + ["X", "B", "Z", "U", "O"]
        special_tokens = [
            self.pad_token,
            self.cls_token,
            self.eos_token,
            self.sep_token,
            self.mask_token,
            self.unk_token,
            *self.chain_tokens,
        ]
        self.vocab: List[str] = special_tokens + aa_tokens
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[self.cls_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.sep_token]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.mask_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def special_ids(self) -> set[int]:
        return {
            self.pad_id,
            self.cls_id,
            self.eos_id,
            self.sep_id,
            self.mask_id,
            self.unk_id,
            *[self.token_to_id[t] for t in self.chain_tokens],
        }

    def get_chain_token(self, locus: str | None) -> str:
        locus = (locus or "").upper().strip()
        if locus == "IGH":
            return "[IGH]"
        if locus == "IGK":
            return "[IGK]"
        if locus == "IGL":
            return "[IGL]"
        return "[OTHER_CHAIN]"

    def encode_sequence(
        self,
        sequence: str,
        locus: str | None = None,
        max_length: int | None = None,
    ) -> List[int]:
        sequence = (sequence or "").upper().strip()
        tokens = [self.cls_token, self.get_chain_token(locus)]
        tokens.extend([aa if aa in self.token_to_id else self.unk_token for aa in sequence])
        tokens.append(self.eos_token)

        if max_length is not None:
            tokens = tokens[:max_length]
            if tokens[-1] != self.eos_token:
                tokens[-1] = self.eos_token

        return [self.token_to_id[token] for token in tokens]

    def decode_ids(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        if skip_special:
            tokens = [tok for tok in tokens if tok not in set(self.vocab[: len(self.special_ids)])]
        aa_tokens = [tok for tok in tokens if len(tok) == 1 and tok.isalpha()]
        return "".join(aa_tokens)

    def save_vocab(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(f"{token}\n")

    @classmethod
    def from_vocab_file(cls, path: str) -> "AminoAcidTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        obj = cls()
        obj.vocab = vocab
        obj.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        obj.id_to_token = {idx: tok for tok, idx in obj.token_to_id.items()}
        return obj
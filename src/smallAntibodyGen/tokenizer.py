from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class AminoAcidTokenizer:
    """Simple fixed-vocabulary tokenizer for antibody amino-acid sequences.

    The vocabulary is deliberately small and explicit to make best use of natural amino-acid meaning.
    """
    
    #special tokens below 
    
    pad_token: str = "[PAD]" #pads to local maximum
    cls_token: str = "[CLS]" #beginning/global summary token
    eos_token: str = "[EOS]" #end-of-sequence token
    sep_token: str = "[SEP]" # separator token, of significance when pairing vh/vl and antibody/antigen
    mask_token: str = "[MASK]" # for training the MLM
    unk_token: str = "[UNK]" # unknown for whatever isn't defined in the dictionary
    
    chain_tokens: List[str] = field(
        # class definition, default_factory to initalize the mutable state. If the locus token isn't informative enough, will change to heavy/light
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

        if max_length is not None and len(tokens) > max_length:
            dropped = len(tokens) - max_length
            warnings.warn(
                f"Tokenized sequence truncated by {dropped} token(s) to fit "
                f"max_length={max_length}; trailing residues (and any light chain "
                "past the cut) are dropped and the final token is forced to [EOS].",
                stacklevel=2,
            )
            tokens = tokens[:max_length]
            if tokens[-1] != self.eos_token:
                tokens[-1] = self.eos_token

        return [self.token_to_id[token] for token in tokens]

    def encode_paired_sequences(
        self,
        heavy_sequence: str,
        light_sequence: str,
        heavy_locus: str | None = "IGH",
        light_locus: str | None = "IGK",
        max_length: int | None = None,
    ) -> List[int]:
        """
        Encode a heavy/light pair into one token sequence.

        Format:
            [CLS] [IGH] HHH... [SEP] [IGK] LLL... [EOS]
        """
        heavy_sequence = (heavy_sequence or "").upper().strip()
        light_sequence = (light_sequence or "").upper().strip()

        tokens = [self.cls_token, self.get_chain_token(heavy_locus)]
        tokens.extend([aa if aa in self.token_to_id else self.unk_token for aa in heavy_sequence])
        tokens.append(self.sep_token)
        tokens.append(self.get_chain_token(light_locus))
        tokens.extend([aa if aa in self.token_to_id else self.unk_token for aa in light_sequence])
        tokens.append(self.eos_token)

        if max_length is not None and len(tokens) > max_length:
            dropped = len(tokens) - max_length
            warnings.warn(
                f"Tokenized sequence truncated by {dropped} token(s) to fit "
                f"max_length={max_length}; trailing residues (and any light chain "
                "past the cut) are dropped and the final token is forced to [EOS].",
                stacklevel=2,
            )
            tokens = tokens[:max_length]
            if tokens[-1] != self.eos_token:
                tokens[-1] = self.eos_token

        return [self.token_to_id[token] for token in tokens]

    def decode_ids(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        if skip_special:
            special_tokens = {
                self.pad_token,
                self.cls_token,
                self.eos_token,
                self.sep_token,
                self.mask_token,
                self.unk_token,
                *self.chain_tokens,
            }
            tokens = [tok for tok in tokens if tok not in special_tokens]
            tokens = [tok for tok in tokens if len(tok) == 1 and tok.isalpha()]
        return "".join(tokens)

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

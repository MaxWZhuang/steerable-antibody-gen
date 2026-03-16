from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset

from src.smallAntibodyGen.tokenizer import AminoAcidTokenizer

@dataclass
class OASRecord:
    sequence: str
    locus: str # IGH / IGK /I GL
    chain_group: str # heavy / light
    split: str # train / val
    length: int 
    cdr3_aa: str | None = None # expected to be either str or None, when no val, default is None
    cdr3_start_aa: int | None = None
    cdr3_end_aa: int | None = None
    v_call: str | None = None
    j_call: str | None = None
    
class OASSequenceDataset(Dataset[OASRecord]):
    def __init__(
        self, 
        data_path: str | Path, 
        split: str
    ) -> None:
        self.data_path = Path(data_path)
        self.split = split
        self.records = List(OASRecord) = []
        self._load()
        
    def _iter_jsonl(self) -> Iterator[Dict[str, object]]:
        opener = gzip.open if self.data_path.suffix == ".gz" else open
        with opener(self.data_path, "rt", encoding="utf-8") as f:
            for line in f: 
                if line.strip(): 
                    yield json.loads(line)
    
    def _load(self) -> None: 
        for record in self._iter_jsonl():
            # yields 1 line at a time, prefers over records
            if record.get("split") != self.split: 
                continue
            self.records.append(
                OASRecord(
                    sequence = str(record["sequence"]),
                    locus = str(record.get("locus", "")), # if record["locus"] does not exist, just use "". dictionary lookback w fallback
                    split = str(record.get("split", self.split)),
                    cdr_3_aa = record.get("cdr3_aa"),
                    v_call = record.get("v_call"),
                    j_call = record.get("j_call")
                    
                )
            )
    def __len__(self) -> int: 
        return len(self.records)
    def __getitem__(self, idx = int) -> OASRecord:
        return self.records[idx]
    
class MLMCollator:
    def __init__(
        self, 
        tokenizer: AminoAcidTokenizer,
        max_length: int, 
        mask_probability: float = 0.15, 
        rng_seed: int = 42
    ) -> None:
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.rng_seed = rng_seed
    
    def _mask_tokens(self, input_ids: torch.Tensor) -> tuple[torch.tensor, torch.tensor]:
        labels = input_ids.clone()
        masked_input = input_ids.clone()
        
        for i in range(input_ids.size[0]):
            for j in range(input_ids.size[1]):
                token_id = int(input_ids[i, j])
                if token_id in self.tokenizer.special_ids: 
                    labels[i, j] = -100
                    continue
                if self.rng.random() >= self.mask_probability: 
                    labels[i, j] = -100
                    continue
            
                dice = self.rng.random()
                if dice < 0.8:
                    #BERT masking recipe
                    #80% of the time, replace with [MASK], 10% of the time, replace with random token, 10% of the time, leave it unchanged
                    masked_input[i, j] = self.tokenizer.mask_id
                elif dice < 0.9: 
                    masked_input[i, j] = self.rng.randrange(self.tokenizer.vocab_size) # random token
        
        return masked_input, labels
    
    def __call__(self, batch: Sequence[OASRecord]) -> Dict[str, torch.Tensor]:
        encoded = [
            self.tokenizer.encode_sequence(item.sequence, locus = item.locus, max_length = self.max_length) for item in batch
        ] # list comprehension
        seq_lengths = [len(x) for x in encoded]
        max_len = min(max(seq_lengths), self.max_length)
        
        padded = []
        attention_masks = []
        for ids in encoded:
            ids = ids[:max_len] # only choose part of the sequence that fits the max_length
            pad_len = max_len - len(ids)
            padded.append(ids + [self.tokenizer.pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        input_ids = torch.tensor(padded, dtype = torch.long)
        attention_mask = torch.tensor(attention_masks, dtype = torch.long)
        masked_input_ids, labels = self._mask_tokens(input_ids)
        
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask, 
            "labels": labels
        }
        
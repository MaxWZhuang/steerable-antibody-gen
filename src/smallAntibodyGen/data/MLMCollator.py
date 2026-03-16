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
    locus: str
    split: str
    cdr3_aa: str | None = None # expected to be either str or None, when no val, default is None
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
                    locus = str(record.get("locus", "")), # if record["split"] does not exist, just use "". dictionary lookback w fallback
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
    
    
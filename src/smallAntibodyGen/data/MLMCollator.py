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
    """
    Masked language model (MLM) that batches for antibody sequences. 
    
    4 Components of the Collator:
        1. Tokenizes each sequence with the provided tokenizer
        2. Pads sequences in the batch to a common length
        3. Builds an attention mask so the model can ignore padding
        4. Applies a mixed MLM objective that can optionally focus on HCDR3 spans for heavy chains while still allowing
        standard random residue masking
    
    Intended as in-between PyTorch Dataset and mode. Dataset should return record-like objects with MINIMUM:
        - sequence
        - locus
        - chain_group
        - cdr3_start_aa
        - cdr3_end_aa
        
    Collator assumptions:
        - Tokenization prepends [CLS] and chain token
        - amino-acid coords are zero-based and relative to the cleaned amino-acid sequence
        - cdr3_end_aa is exclusive
    """
    def __init__(
        self, 
        tokenizer: AminoAcidTokenizer,
        max_length: int, 
        mask_probability: float = 0.15, 
        hcdr_3_span_probability: float = 0.5, 
        hcdr_3_span_min: int = 3, 
        hcdr3_span_max: int = 8, 
        rng_seed: int = 42
    ) -> None:
        """
        Stores tokenizer/configuration state and prepare RNG and list of residue-token IDs that are legal (not special) 
        random replacements during MLM corruption

        Args:
        
            tokenizer (AminoAcidTokenizer): Convert amino-acid sequences into token IDs. Must expose: 
                token_to_id, 
                special_ids, 
                mask_id, 
                encode_sequence
                
            max_length (int): Maximum tokenized sequence length allowed in a batch, INCLUDING special tokens (ex. CLS, chain token, EOS)
            
            mask_probability (float, optional): Fraction of eligible residue positions to turn into MLM targets. Defaults to 0.15.
            
            hcdr_3_span_probability (float, optional): Probability of attempting HCDR3 span masking for a heavy-chain example with 
                valid HCDR3 coordinates. If this does not trigger, example falls back to ordinary random MLM target selection. 
                Defaults to 0.5.
                
            hcdr_3_span_min (int, optional): Minimum number of residues to mask when sampling an HCDR3 span. Defaults to 3.
            hcdr3_span_max (int, optional): Maximum number of residues to mask when sampling an HCDR3 span. Defaults to 8.
            rng_seed (int, optional): seed for the Python random number generator used by this collator Defaults to 42.

        """
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.hcdr_3_span_probability = hcdr_3_span_probability
        self.hcdr3_span_min = hcdr_3_span_min
        self.hcdr3_span_max = hcdr3_span_max
        self.rng = random.Random(rng_seed)
        
        # sampling replacement tokens from actual residue tokens, not special tokens
        self.residue_token_ids = [
            idx 
            for tok, idx in self.tokenizer.token_to_id.items() 
            if len(tok) == 1 and tok.isalpha
        ]
    
    def _select_target_positions(
        self, 
        input_ids_row: torch.Tensor,
        record
    ) -> set[int]: 
        """
        Chose which token positions should become MLM targets. 
        
        General strategy:
        1. If heavy chain + valid HCDR3 coordinations + chosen to be sampled: 
        - potentially sample a span inside of HCDR3
        2. Then top up to the overall masking budget with random residue positions using ordinary random residue positions

        Args:
        
            input_ids_row (torch.Tensor): 1D tesnor of token IDs for a single, already-padded sequence. 
                Can include special tokens such as [CLS], chain token, [EOS], [PAD]
                
            record (dataset record): Dataset record for the same sequence as input_ids_rows. It is expected to expose: 
                - chain_group
                - cdr3_start_aa
                - cdr3_end_aa

        Returns:
            set[int]: Set of integer token positions that should become MLM targets. Positions are indices into "input_ids_row"
            
        """
        
        selected: set[int] = set()
        
        eligible_positions = [
            j
            for j, token_id in enumerate(input_ids_row.toList())
            if token_id not in self.tokenizer.special_ids
        ]
        
        if not eligible_positions:
            return selected
        
        target_budget = max(1, round(len(eligible_positions)) * self.mask_probability)
        
        if (
            record.chain_group == "heavy"
            and record.cdr3_start_aa is not None 
            and record.cdr3_end_aa is not None
            and self.rng.random() < self.hcdr3_span_probability
        ): 
            # Offset by 2, encode_seq() auto-prepends # [CLS], [CHAIN_TOKEN]
            cdr_3_start_token = 2 + record.cdr3_start_aa
            cdr_3_end_token = 2 + record.cdr3_end_aa # end-exclusive
            
            # clip to the available tokenized row length (if sampling good, should be a non-issue)
            cdr3_positions = [
                j
                for j in range(cdr_3_start_token, min(cdr_3_end_token, input_ids_row.size(0)))
                if int(input_ids_row[j]) not in self.tokenizer.special_ids
            ]
            
            if cdr3_positions: 
                span_max = min(self.hcdr3_span_max, len(cdr3_positions))
                span_min = min(self.hcdr3_span_min, span_max)
                span_len = self.rng.randint(span_min, span_max) 
                
                left_min = cdr3_positions[0]
                left_max = cdr3_positions[-1] - span_len + 1
                
                if left_max >= left_min:
                    span_left = self.rng.randint(left_min, left_max)
                    selected.update(range(span_left, span_left + span_len))
                    
        remaining = [j 
                    for j in eligible_positions 
                    if j not in selected]
        self.rng.shuffle(remaining)
        for j in remaining: 
            if len(selected) >= target_budget:
                break
            selected.add(j)
    
    
    def _mask_tokens(self, 
                    input_ids: torch.Tensor, 
                    batch_records: Sequence
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Build Masked Language Model corrupted input + sparse labels for one batch. 
        Given batch of token IDs and the correspodning dataset records, choose MLM target positions for each example, then apply standard
        BERT-style corruption: 
            - 80% replace target with [MASK]
            - 10% replace target with random residue token
            - 10% keep original token unchanged
            
        Labels set only at selected target positions; all other positions filled with -100 so as to be ignored by CrossEntropyLoss

        Args:
        
            input_ids (torch.Tensor): 2D tensor of shape [batch_size, seq_len] containing padded token IDs for the batch
            
            batch_records (Sequence): Original dataset recrods corresponding to each batch row. 
                Each record used to determine chain type and HCDR3 boundaries.

        Returns:
        
            tuple[torch.tensor, torch.tensor]: tuple (masked_input, labels) where: 
                - masked_input: A tensor of the same shape as "input_ids", containing corrupted version seen by the model
                - labels: A tensor of the sme shape as "input_ids", containg the original token IDs only at selected MLM target positions
                    and -100 everywhere else. 
            
        """
        
        labels = torch.full_like(input_ids, fill_value = -100)
        masked_input = input_ids.clone()
        
        for i, record in enumerate(batch_records):
            selected_positions = self._select_target_positions(input_ids[i], record)
            
            for j in selected_positions:
                labels[i, j] = input_ids[i, j]
                
                dice = self.rng.random()
                # standard BERT procedure, 
                # 80% of masked tokens actually masked, 
                # 10% of tokens replaced with rand. token, 
                # 10% of tokens will leave unchanged
                if dice < 0.8: 
                    masked_input[i, j] = self.tokenizer.mask_id
                elif dice < 0.9:
                    masked_input[i, j] = self.rng.choice(self.residue_token_ids)
                else: 
                    pass
        return masked_input, labels
    
    def __call__(self, batch: Sequence[OASRecord]) -> Dict[str, torch.Tensor]:
        encoded = [
            self.tokenizer.encode_sequence(
                item.sequence, 
                locus = item.locus, 
                max_length = self.max_length) 
            for item in batch
        ] 
        
        """
        
        Convert list of sequence records into one MLM training batch by tokenziing each example, padding all examples in the batch to the same length,
        building an attention mask, and creating MLM inputs/labels using "_mask_tokens"
        
        Args: 
            batch (Sequence[OASRecord]): Seequence of dataset record objects. Each record must have at least:
                - sequence
                - locus
                
                HCDR3 span masking also requires 
                - chain_group
                - cdr3_start_aa
                - cdr3_end_aa
                
        Returns:
            Dictionary["input_ids", "attention_mask", "labels"]: 
                - "input_ids": Tensor of shape [batch_size, seq_len] containing masked/corrupted token IDs for model input
                - "attention_mask": Tensor of shape [batch_size, seq_len] where 1 indicates a real token and 0 indicates padding
                - "labels": Tensor of shape [batch_size, seq_len] containing MLM targets at selected positions and -100 elsewhere
        
        """
        
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
        
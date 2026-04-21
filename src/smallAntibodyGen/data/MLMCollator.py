from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset

from smallAntibodyGen.tokenizer import AminoAcidTokenizer
from smallAntibodyGen.data.MLMSampler import ChainLengthBucketBatchSampler

@dataclass
class OASRecord:
    """In-memory representation of one processed OAS or antibody-antigen example."""

    sequence: str
    locus: str # IGH / IGK / IGL
    chain_group: str # heavy / light
    split: str # train / val
    length: int 
    cdr3_aa: str | None = None # expected to be either str or None, when no val, default is None
    cdr3_start_aa: int | None = None
    cdr3_end_aa: int | None = None
    v_call: str | None = None
    j_call: str | None = None
    token_ids: list[int] | None = None
    token_length: int | None = None
    sequence_heavy: str | None = None
    sequence_light: str | None = None
    heavy_locus: str | None = None
    light_locus: str | None = None
    is_paired: bool = False
    pair_id: str | None = None
    cdr3_aa_heavy: str | None = None
    cdr3_start_aa_heavy: int | None = None
    cdr3_end_aa_heavy: int | None = None
    cdr3_aa_light: str | None = None
    cdr3_start_aa_light: int | None = None
    cdr3_end_aa_light: int | None = None
    sequence_antigen: str | None = None
    target_key: str | None = None
    target_name: str | None = None
    target_pdb: str | None = None
    target_uniprot: str | None = None
    dataset_name: str | None = None
    confidence: str | None = None
    affinity_type: str | None = None
    affinity_raw: str | None = None
    processed_measurement_raw: str | None = None
    processed_measurement_float: float | None = None
    binder_label: int | None = None
    is_nanobody: bool = False
    scfv: bool = False
    
class OASSequenceDataset(Dataset[OASRecord]):
    """Dataset that reads processed single-chain or paired OAS JSONL records."""

    def __init__(
        self, 
        data_path: str | Path, 
        split: str
    ) -> None:
        self.data_path = Path(data_path)
        self.split = split
        self.records: list[OASRecord] = []
        self._load()
        
    def _iter_jsonl(self) -> Iterator[Dict[str, object]]:
        """Yield parsed JSON objects from a plain or gzipped JSONL file."""
        opener = gzip.open if self.data_path.suffix == ".gz" else open
        with opener(self.data_path, "rt", encoding="utf-8") as f:
            for line in f: 
                if line.strip(): 
                    yield json.loads(line)
    
    def _load(self) -> None: 
        """
        Load the requested split into memory as `OASRecord` objects.

        The processed schema now supports both classic single-chain records and
        native heavy/light paired examples. We preserve a single dataset class
        so the training code can switch between them based only on the contents
        of the processed JSONL file.
        """
        for record in self._iter_jsonl():
            # yields 1 line at a time, prefers over records
            if record.get("split") != self.split: 
                continue
            token_ids = record.get("token_ids")
            token_length = record.get("token_length")
            if token_length is None:
                if record.get("sequence_heavy") and record.get("sequence_light"):
                    token_length = len(str(record["sequence_heavy"])) + len(str(record["sequence_light"])) + 5
                elif isinstance(token_ids, list):
                    token_length = len(token_ids)
                else:
                    token_length = len(str(record["sequence"])) + 3
            self.records.append(
                OASRecord(
                    sequence=str(record["sequence"]),
                    locus=str(record.get("locus", "")),
                    chain_group=str(record.get("chain_group", "")),
                    split=str(record.get("split", self.split)),
                    length=int(record.get("length", len(str(record["sequence"])))),
                    cdr3_aa=record.get("cdr3_aa"),
                    cdr3_start_aa=record.get("cdr3_start_aa"),
                    cdr3_end_aa=record.get("cdr3_end_aa"),
                    v_call=record.get("v_call"),
                    j_call=record.get("j_call"),
                    token_ids=token_ids,
                    token_length=int(token_length),
                    sequence_heavy=record.get("sequence_heavy"),
                    sequence_light=record.get("sequence_light"),
                    heavy_locus=record.get("heavy_locus"),
                    light_locus=record.get("light_locus"),
                    is_paired=bool(record.get("is_paired")) or bool(record.get("sequence_heavy") and record.get("sequence_light")),
                    pair_id=record.get("pair_id"),
                    cdr3_aa_heavy=record.get("cdr3_aa_heavy"),
                    cdr3_start_aa_heavy=record.get("cdr3_start_aa_heavy"),
                    cdr3_end_aa_heavy=record.get("cdr3_end_aa_heavy"),
                    cdr3_aa_light=record.get("cdr3_aa_light"),
                    cdr3_start_aa_light=record.get("cdr3_start_aa_light"),
                    cdr3_end_aa_light=record.get("cdr3_end_aa_light"),
                    sequence_antigen=record.get("sequence_antigen"),
                    target_key=record.get("target_key"),
                    target_name=record.get("target_name"),
                    target_pdb=record.get("target_pdb"),
                    target_uniprot=record.get("target_uniprot"),
                    dataset_name=record.get("dataset"),
                    confidence=record.get("confidence"),
                    affinity_type=record.get("affinity_type"),
                    affinity_raw=record.get("affinity_raw"),
                    processed_measurement_raw=record.get("processed_measurement_raw"),
                    processed_measurement_float=record.get("processed_measurement_float"),
                    binder_label=record.get("binder_label"),
                    is_nanobody=bool(record.get("is_nanobody")),
                    scfv=bool(record.get("scfv")),
                )
            )
            
    def __len__(self) -> int: 
        return len(self.records)
    def __getitem__(self, idx: int) -> OASRecord:
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
        hcdr3_span_probability: float = 0.5, 
        hcdr3_span_min: int = 3, 
        hcdr3_span_max: int = 8, 
        shuffle_pair_probability: float = 0.5,
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
            
            hcdr3_span_probability (float, optional): Probability of attempting HCDR3 span masking for a heavy-chain example with 
                valid HCDR3 coordinates. If this does not trigger, example falls back to ordinary random MLM target selection. 
                Defaults to 0.5.
                
            hcdr3_span_min (int, optional): Minimum number of residues to mask when sampling an HCDR3 span. Defaults to 3.
            hcdr3_span_max (int, optional): Maximum number of residues to mask when sampling an HCDR3 span. Defaults to 8.
            rng_seed (int, optional): seed for the Python random number generator used by this collator Defaults to 42.

        """
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.hcdr3_span_probability = hcdr3_span_probability
        self.hcdr3_span_min = hcdr3_span_min
        self.hcdr3_span_max = hcdr3_span_max
        self.shuffle_pair_probability = shuffle_pair_probability
        self.rng = random.Random(rng_seed)
        
        # sampling replacement tokens from actual residue tokens, not special tokens
        self.residue_token_ids = [
            idx 
            for tok, idx in self.tokenizer.token_to_id.items() 
            if len(tok) == 1 and tok.isalpha()
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
            for j, token_id in enumerate(input_ids_row.tolist())
            if token_id not in self.tokenizer.special_ids
        ]
        
        if not eligible_positions:
            return selected
        
        target_budget = max(1, int(round(len(eligible_positions) * self.mask_probability)))
        
        if (
            self.rng.random() < self.hcdr3_span_probability
            and record.cdr3_start_aa is not None
            and record.cdr3_end_aa is not None
            and (record.chain_group == "heavy" or record.is_paired)
        ):
            # Offset by 2, encode_seq() auto-prepends # [CLS], [CHAIN_TOKEN]
            cdr3_start_token = 2 + record.cdr3_start_aa
            cdr3_end_token = 2 + record.cdr3_end_aa # end-exclusive
            
            # clip to the available tokenized row length (if sampling good, should be a non-issue)
            cdr3_positions = [
                j
                for j in range(cdr3_start_token, min(cdr3_end_token, input_ids_row.size(0)))
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
        
        return selected

    def _is_pairable_record(self, item: OASRecord) -> bool:
        """
        Decide whether one record can participate in pair shuffling.

        Args:
            item:
                Dataset record under consideration.

        Returns:
            True when the record already contains both heavy and light chain
            sequences and therefore supports the auxiliary compatibility task.
        """
        heavy_seq = (item.sequence_heavy or "").strip()
        light_seq = (item.sequence_light or "").strip()
        return bool(item.is_paired and heavy_seq and light_seq)

    def _build_pairing_batch(
        self,
        batch: Sequence[OASRecord],
    ) -> tuple[list[OASRecord], list[int], list[bool]]:
        """
        Materialize native or shuffled pairings for one batch.

        We keep preprocessing outputs strictly native/cognate. This helper
        synthesizes shuffled negatives on the fly by replacing the light chain
        for a subset of paired examples with a light chain drawn from another
        paired example in the same batch.

        Args:
            batch:
                Sequence of dataset records selected for this batch.

        Returns:
            Tuple `(effective_batch, pair_labels, pair_mask)` where:
                - `effective_batch` is the batch actually encoded
                - `pair_labels` contains 1 for native pairs and 0 for shuffled
                - `pair_mask` marks examples that participate in the pair loss
        """
        native_batch = [replace(item) for item in batch]
        effective_batch = [replace(item) for item in batch]
        pair_labels = [0] * len(batch)
        pair_mask = [False] * len(batch)

        pairable_indices = [idx for idx, item in enumerate(effective_batch) if self._is_pairable_record(item)]
        if len(pairable_indices) < 2:
            for idx in pairable_indices:
                pair_labels[idx] = 1
                pair_mask[idx] = True
            return effective_batch, pair_labels, pair_mask

        shuffled_indices = []
        for idx in pairable_indices:
            pair_labels[idx] = 1
            pair_mask[idx] = True
            if self.rng.random() < self.shuffle_pair_probability:
                shuffled_indices.append(idx)

        if len(shuffled_indices) == 1 and len(pairable_indices) > 1:
            # A lone negative cannot borrow from itself, so fall back to native.
            shuffled_indices = []

        for idx in shuffled_indices:
            donor_candidates = [candidate for candidate in pairable_indices if candidate != idx]
            donor_idx = self.rng.choice(donor_candidates)
            donor_record = native_batch[donor_idx]

            # We keep the heavy chain fixed and replace only the light chain so
            # the classifier learns whether the observed partner is cognate.
            effective_batch[idx] = replace(
                effective_batch[idx],
                sequence_light=donor_record.sequence_light,
                light_locus=donor_record.light_locus,
                cdr3_aa_light=donor_record.cdr3_aa_light,
                cdr3_start_aa_light=donor_record.cdr3_start_aa_light,
                cdr3_end_aa_light=donor_record.cdr3_end_aa_light,
            )
            pair_labels[idx] = 0

        return effective_batch, pair_labels, pair_mask
    
    
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
            if selected_positions is None:
                raise RuntimeError("_select_target_positions() returned None; it must return a set of positions")
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
    
    def _encode_record(self, item: OASRecord) -> list[int]:
        """
        Encode one single-chain or paired record into token IDs.

        Args:
            item:
                Dataset record to encode.

        Returns:
            List of integer token IDs suitable for batching.
        """
        heavy_seq = (item.sequence_heavy or "").strip() if item.sequence_heavy is not None else ""
        light_seq = (item.sequence_light or "").strip() if item.sequence_light is not None else ""
        if heavy_seq and light_seq:
            return self.tokenizer.encode_paired_sequences(
                heavy_sequence=heavy_seq,
                light_sequence=light_seq,
                heavy_locus=item.heavy_locus or "IGH",
                light_locus=item.light_locus or "IGK",
                max_length=self.max_length,
            )
        return self.tokenizer.encode_sequence(
            item.sequence,
            locus=item.locus,
            max_length=self.max_length,
        )

    def __call__(self, batch: Sequence[OASRecord]) -> Dict[str, torch.Tensor]:
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
            Dictionary["input_ids", "attention_mask", "labels", "pair_labels", "pair_mask"]: 
                - "input_ids": Tensor of shape [batch_size, seq_len] containing masked/corrupted token IDs for model input
                - "attention_mask": Tensor of shape [batch_size, seq_len] where 1 indicates a real token and 0 indicates padding
                - "labels": Tensor of shape [batch_size, seq_len] containing MLM targets at selected positions and -100 elsewhere
                - "pair_labels": Tensor of shape [batch_size] with 1 for native pairs and 0 for shuffled negatives
                - "pair_mask": Tensor of shape [batch_size] marking records that participate in the auxiliary pair loss
        
        """
        
        effective_batch, pair_labels, pair_mask = self._build_pairing_batch(batch)
        encoded = [self._encode_record(item) for item in effective_batch]
        
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
        
        masked_input_ids, labels = self._mask_tokens(input_ids, effective_batch)

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask, 
            "labels": labels,
            "pair_labels": torch.tensor(pair_labels, dtype=torch.long),
            "pair_mask": torch.tensor(pair_mask, dtype=torch.bool),
        }
        

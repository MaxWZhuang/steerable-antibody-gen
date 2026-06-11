#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    class _TqdmFallback:
        def __init__(self, iterable, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable, *args, **kwargs):
        return _TqdmFallback(iterable, *args, **kwargs)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.tokenizer import AminoAcidTokenizer
from smallAntibodyGen.data.MLMCollator import (
    AntibodyAntigenCollator,
    AntibodyAntigenRealLabelCollator,
    OASSequenceDataset,
    MLMCollator
)
from smallAntibodyGen.data.MLMSampler import ChainLengthBucketBatchSampler
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, AntibodyMLM, MLMConfig

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    yaml = None


HCDR3_INFILL_STAGE = "antigen_hcdr3_infill_refine"
ANTIGEN_STAGES = {"antigen_refine", "antigen_real_label_refine", HCDR3_INFILL_STAGE}


def is_antigen_stage(training_stage: str) -> bool:
    return training_stage in ANTIGEN_STAGES


def is_hcdr3_infill_stage(training_stage: str) -> bool:
    return training_stage == HCDR3_INFILL_STAGE


@dataclass
class TrainConfig:
    """
    Configuration object for MLM training.

    Attributes:
        data_path:
            Path to the processed JSONL(.gz) file written by prepare_oas.py.
        output_dir:
            Directory where checkpoints and logs will be written.
        max_length:
            Maximum tokenized sequence length seen by the collator/model.
        batch_size:
            Number of examples per training batch.
        eval_batch_size:
            Number of examples per evaluation batch.
        train_num_workers:
            Number of DataLoader workers for training.
        eval_num_workers:
            Number of DataLoader workers for evaluation.
        bucket_width:
            Size of sequence-length buckets used by the custom batch sampler.
        mask_probability:
            Fraction of eligible residue tokens selected as MLM targets.
        hcdr3_span_probability:
            Probability of using HCDR3 span masking on heavy-chain examples
            with valid HCDR3 coordinates.
        hcdr3_span_min:
            Minimum masked HCDR3 span length.
        hcdr3_span_max:
            Maximum masked HCDR3 span length.
        d_model:
            Transformer hidden dimension.
        n_heads:
            Number of attention heads.
        n_layers:
            Number of transformer encoder layers.
        d_ff:
            Feed-forward hidden size inside each transformer block.
        dropout:
            Dropout probability used in the model.
        learning_rate:
            AdamW learning rate.
        weight_decay:
            AdamW weight decay.
        grad_clip_norm:
            Maximum gradient norm.
        epochs:
            Number of full training epochs.
        seed:
            Random seed for reproducibility.
        use_amp:
            Whether to use automatic mixed precision on supported devices.
        smoke_test_only:
            If True, run one train step + one eval step and exit.
        device:
            Optional device override. If None, infer automatically.
    """

    data_path: str
    output_dir: str = "checkpoints/mlm"
    training_stage: str = "base"
    init_checkpoint: Optional[str] = None
    resume_from_last: bool = True
    max_length: int = 192

    batch_size: int = 32
    eval_batch_size: int = 32
    train_num_workers: int = 0
    eval_num_workers: int = 0
    bucket_width: int = 8

    mask_probability: float = 0.15
    hcdr3_span_probability: float = 0.0
    hcdr3_span_min: int = 3
    hcdr3_span_max: int = 8
    hcdr3_mask_mode: str = "sampled_span"
    mask_replacement_strategy: str = "bert"
    shuffle_pair_probability: float = 0.5
    shuffle_antigen_probability: float = 0.5

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    warmup_steps: int = 0
    pair_loss_weight: float = 1.0
    compatibility_loss_weight: float = 1.0
    epochs: int = 5
    seed: int = 42

    use_amp: bool = False
    smoke_test_only: bool = False
    show_progress: bool = True
    device: Optional[str] = None

    def validate(self) -> None:
        """
        Validate that the training configuration is internally consistent.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError:
                If any configuration value is invalid.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if self.bucket_width <= 0:
            raise ValueError("bucket_width must be > 0")
        if not (0.0 < self.mask_probability <= 1.0):
            raise ValueError("mask_probability must be in (0, 1]")
        if not (0.0 <= self.hcdr3_span_probability <= 1.0):
            raise ValueError("hcdr3_span_probability must be in [0, 1]")
        if not (0.0 <= self.shuffle_pair_probability <= 1.0):
            raise ValueError("shuffle_pair_probability must be in [0, 1]")
        if not (0.0 <= self.shuffle_antigen_probability <= 1.0):
            raise ValueError("shuffle_antigen_probability must be in [0, 1]")
        if self.hcdr3_span_min <= 0 or self.hcdr3_span_max <= 0:
            raise ValueError("HCDR3 span lengths must be > 0")
        if self.hcdr3_span_min > self.hcdr3_span_max:
            raise ValueError("hcdr3_span_min must be <= hcdr3_span_max")
        if self.hcdr3_mask_mode not in {"sampled_span", "full_span"}:
            raise ValueError("hcdr3_mask_mode must be one of: sampled_span, full_span")
        if self.mask_replacement_strategy not in {"bert", "always_mask"}:
            raise ValueError("mask_replacement_strategy must be one of: bert, always_mask")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.pair_loss_weight < 0:
            raise ValueError("pair_loss_weight must be >= 0")
        if self.compatibility_loss_weight < 0:
            raise ValueError("compatibility_loss_weight must be >= 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.train_num_workers < 0 or self.eval_num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.training_stage not in {"base", "paired_refine", *ANTIGEN_STAGES}:
            raise ValueError(
                "training_stage must be one of: base, paired_refine, "
                "antigen_refine, antigen_real_label_refine, "
                "antigen_hcdr3_infill_refine"
            )
        if self.training_stage == "paired_refine" and not self.init_checkpoint:
            raise ValueError(
                "paired_refine training requires `init_checkpoint` so refinement "
                "starts from a pretrained model."
            )
        if is_antigen_stage(self.training_stage) and not self.init_checkpoint:
            raise ValueError(
                f"{self.training_stage} training requires `init_checkpoint` so the "
                "dual-stream model starts from a paired-refine checkpoint."
            )


def _train_config_defaults() -> Dict[str, Any]:
    """
    Return default values for every optional TrainConfig field.

    We derive this from the dataclass instead of duplicating defaults in the
    CLI/config loader, which keeps the defaults in one authoritative place.

    Args:
        None.

    Returns:
        Dictionary of field name -> default value for optional config fields.
    """
    defaults: Dict[str, Any] = {}
    for field in fields(TrainConfig):
        if field.name == "data_path":
            continue
        if field.default is not MISSING:
            defaults[field.name] = field.default
    return defaults


def load_config_file(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a training config from JSON or YAML.

    Args:
        config_path:
            Path to a config file.

    Returns:
        Raw parsed config dictionary.

    Raises:
        ValueError:
            If the file extension is unsupported or the parsed payload is not
            a dictionary.
    """
    path = Path(config_path)
    suffixes = path.suffixes

    if suffixes[-2:] == [".jsonl", ".gz"]:
        raise ValueError("Config files must be JSON or YAML, not JSONL data files")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
    elif path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with `pip install pyyaml`."
            )
        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format for {path}. Use .json, .yaml, or .yml")

    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected top-level mapping in config file {path}")
    return raw_config


def normalize_config_data(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate config-file keys into the flat TrainConfig schema.

    The checked-in YAML uses a friendlier nested layout and a couple of older
    names (`num_workers`, `mixed_precision`). We normalize those here so the
    training loop can keep using one simple dataclass.

    Args:
        raw_config:
            Parsed JSON/YAML dictionary.

    Returns:
        Flat dictionary containing TrainConfig-compatible keys.

    Raises:
        ValueError:
            If a nested section is present but not a mapping.
    """
    normalized = dict(raw_config)

    mode = normalized.pop("mode", None)
    if mode is not None:
        normalized.setdefault("training_stage", mode)

    init_from_checkpoint = normalized.pop("init_from_checkpoint", None)
    if init_from_checkpoint is not None:
        normalized.setdefault("init_checkpoint", init_from_checkpoint)

    # Legacy/shared worker count: if the config specifies one worker value,
    # treat it as both train/eval unless a side-specific override exists.
    num_workers = normalized.pop("num_workers", None)
    if num_workers is not None:
        normalized.setdefault("train_num_workers", num_workers)
        normalized.setdefault("eval_num_workers", num_workers)

    # Keep the YAML key intuitive while mapping onto the runtime flag name.
    mixed_precision = normalized.pop("mixed_precision", None)
    if mixed_precision is not None:
        normalized.setdefault("use_amp", mixed_precision)

    model_config = normalized.pop("model", None)
    if model_config is not None:
        if not isinstance(model_config, dict):
            raise ValueError("The `model` config section must be a mapping")
        for key in ("d_model", "n_heads", "n_layers", "d_ff", "dropout"):
            if key in model_config:
                normalized.setdefault(key, model_config[key])

    optimizer_config = normalized.pop("optimizer", None)
    if optimizer_config is not None:
        if not isinstance(optimizer_config, dict):
            raise ValueError("The `optimizer` config section must be a mapping")
        # The current training loop only uses lr/weight decay. We accept the
        # section so existing configs keep working even though extra keys are
        # currently informational only.
        if "learning_rate" in optimizer_config:
            normalized.setdefault("learning_rate", optimizer_config["learning_rate"])
        if "weight_decay" in optimizer_config:
            normalized.setdefault("weight_decay", optimizer_config["weight_decay"])

    logging_config = normalized.pop("logging", None)
    if logging_config is not None and not isinstance(logging_config, dict):
        raise ValueError("The `logging` config section must be a mapping")

    # `warmup_steps` is a real TrainConfig field honored by build_lr_scheduler /
    # train_one_epoch, so it is passed through unchanged rather than dropped.

    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser used by the training entrypoint.

    Args:
        None.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Train an antibody MLM on processed OAS data.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--training-stage",
        type=str,
        choices=("base", "paired_refine", "antigen_refine", "antigen_real_label_refine", HCDR3_INFILL_STAGE),
    )
    parser.add_argument("--init-checkpoint", type=str)
    parser.add_argument("--resume-from-last", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-resume-from-last", dest="resume_from_last", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--max-length", type=int)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--train-num-workers", type=int)
    parser.add_argument("--eval-num-workers", type=int)
    parser.add_argument("--bucket-width", type=int)

    parser.add_argument("--mask-probability", type=float)
    parser.add_argument("--hcdr3-span-probability", type=float)
    parser.add_argument("--hcdr3-span-min", type=int)
    parser.add_argument("--hcdr3-span-max", type=int)
    parser.add_argument("--hcdr3-mask-mode", type=str, choices=("sampled_span", "full_span"))
    parser.add_argument("--mask-replacement-strategy", type=str, choices=("bert", "always_mask"))
    parser.add_argument("--shuffle-pair-probability", type=float)
    parser.add_argument("--shuffle-antigen-probability", type=float)

    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--pair-loss-weight", type=float)
    parser.add_argument("--compatibility-loss-weight", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--use-amp", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--smoke-test-only", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--show-progress", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    """
    Parse CLI arguments plus an optional config file into TrainConfig.

    Merge precedence is:
    1. TrainConfig dataclass defaults
    2. Values loaded from `--config`
    3. Explicit CLI flags

    This lets saved configs act as reusable presets while keeping the command
    line ergonomic for quick one-off overrides.

    Args:
        argv:
            Optional sequence of CLI arguments. If omitted, argparse reads from
            `sys.argv`.

    Returns:
        A validated TrainConfig object.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args_dict = vars(args)

    merged_config = _train_config_defaults()

    config_path = args_dict.pop("config", None)
    file_config: Dict[str, Any] = {}
    if config_path:
        file_config = normalize_config_data(load_config_file(config_path))
        merged_config.update(file_config)

    # CLI values always win over config-file values when both are provided.
    merged_config.update(args_dict)

    if merged_config.get("training_stage") == HCDR3_INFILL_STAGE:
        if "compatibility_loss_weight" not in file_config and "compatibility_loss_weight" not in args_dict:
            merged_config["compatibility_loss_weight"] = 0.0
        if "hcdr3_mask_mode" not in file_config and "hcdr3_mask_mode" not in args_dict:
            merged_config["hcdr3_mask_mode"] = "full_span"
        if "mask_replacement_strategy" not in file_config and "mask_replacement_strategy" not in args_dict:
            merged_config["mask_replacement_strategy"] = "always_mask"
        if "shuffle_antigen_probability" not in file_config and "shuffle_antigen_probability" not in args_dict:
            merged_config["shuffle_antigen_probability"] = 0.0

    output_dir_provided = ("output_dir" in file_config) or ("output_dir" in args_dict)
    if not output_dir_provided:
        if merged_config.get("training_stage") == "paired_refine":
            merged_config["output_dir"] = "checkpoints/mlm_paired_refine"
        elif merged_config.get("training_stage") == "antigen_refine":
            merged_config["output_dir"] = "checkpoints/mlm_antigen_refine"
        elif merged_config.get("training_stage") == "antigen_real_label_refine":
            merged_config["output_dir"] = "checkpoints/mlm_antigen_real_label_refine"
        elif merged_config.get("training_stage") == HCDR3_INFILL_STAGE:
            merged_config["output_dir"] = "checkpoints/mlm_antigen_hcdr3_infill_refine"

    if "data_path" not in merged_config or not merged_config["data_path"]:
        parser.error("--data-path is required unless provided via --config")

    cfg = TrainConfig(**merged_config)
    cfg.validate()
    return cfg


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed:
            Integer seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_override: Optional[str] = None) -> torch.device:
    """
    Choose a torch.device for training.

    Args:
        device_override:
            Optional user-specified device string, such as "cpu" or "cuda".

    Returns:
        A torch.device instance.
    """
    if device_override is not None:
        return torch.device(device_override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_cpu_runtime(device: torch.device) -> None:
    """
    Apply conservative CPU thread settings when running on CPU.

    This is a practical stability measure for some environments where large
    thread counts can make transformer training noisy or hard to debug.

    Args:
        device:
            The chosen training device.

    Returns:
        None.
    """
    if device.type == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


def seed_worker(worker_id: int) -> None:
    """
    Seed NumPy and Python RNGs inside each DataLoader worker.

    Args:
        worker_id:
            The integer worker ID assigned by PyTorch.

    Returns:
        None.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_tokenizer() -> AminoAcidTokenizer:
    """
    Build the tokenizer used by the MLM pipeline.

    Args:
        None.

    Returns:
        An initialized AminoAcidTokenizer.
    """
    return AminoAcidTokenizer()


def has_valid_heavy_hcdr3_span(record: Any) -> bool:
    """
    Return True when a record has a complete heavy-chain HCDR3 span.

    The HCDR3 infilling stage masks the entire known span, so it cannot use
    rows where the CDR3 coordinates are missing, malformed, or empty. We prefer
    heavy-specific span fields but accept the older generic fields as a
    fallback for heavy-chain records that were prepared before the
    antibody-antigen schema grew explicit ``*_heavy`` names.
    """
    start = getattr(record, "cdr3_start_aa_heavy", None)
    end = getattr(record, "cdr3_end_aa_heavy", None)
    cdr3 = getattr(record, "cdr3_aa_heavy", None)
    if start is None or end is None:
        start = getattr(record, "cdr3_start_aa", None)
        end = getattr(record, "cdr3_end_aa", None)
        cdr3 = getattr(record, "cdr3_aa", None)
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and end > start
        and isinstance(cdr3, str)
        and len(cdr3) == (end - start)
    )


def is_hcdr3_infill_record(record: Any) -> bool:
    """
    Decide whether one antibody-antigen record can train HCDR3 infilling.

    This stage is intentionally positive-only. It estimates a conditional
    residue model for observed binders, not a binder-vs-non-binder
    classifier. Compatibility scoring remains a separate ranking/filtering
    step after candidate generation.
    """
    heavy_sequence = (getattr(record, "sequence_heavy", None) or getattr(record, "sequence", "") or "").strip()
    antigen_sequence = (getattr(record, "sequence_antigen", None) or "").strip()
    return (
        getattr(record, "binder_label", None) == 1
        and bool(heavy_sequence)
        and bool(antigen_sequence)
        and has_valid_heavy_hcdr3_span(record)
    )


def build_datasets(cfg: TrainConfig) -> Tuple[OASSequenceDataset, OASSequenceDataset]:
    """
    Build train and validation datasets from the processed OAS file.

    Args:
        cfg:
            Training configuration.

    Returns:
        Tuple `(train_dataset, val_dataset)`.
    """
    train_dataset = OASSequenceDataset(cfg.data_path, split="train")
    val_dataset = OASSequenceDataset(cfg.data_path, split="val")
    if cfg.training_stage == "antigen_real_label_refine":
        train_dataset.records = [
            record for record in train_dataset.records if record.binder_label in (0, 1)
        ]
        val_dataset.records = [
            record for record in val_dataset.records if record.binder_label in (0, 1)
        ]
    elif is_hcdr3_infill_stage(cfg.training_stage):
        train_dataset.records = [
            record for record in train_dataset.records if is_hcdr3_infill_record(record)
        ]
        val_dataset.records = [
            record for record in val_dataset.records if is_hcdr3_infill_record(record)
        ]
    return train_dataset, val_dataset


class RecordSubsetDataset(Dataset):
    """
    Lightweight in-memory dataset view used for diagnostic probes.

    The training/eval loaders in this script expect a `.records` attribute, so
    this wrapper mirrors the shape of `OASSequenceDataset` closely enough to
    reuse the existing samplers and collators without changing the data format.
    """

    def __init__(self, records: Sequence[Any], split: str) -> None:
        self.records = list(records)
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Any:
        return self.records[idx]


def choose_probe_size(total_records: int, eval_batch_size: int) -> int:
    """
    Pick a small deterministic probe size for extra evaluations.
    """
    if total_records <= 1:
        return 0
    target = max(eval_batch_size * 32, 1024)
    return min(total_records - 1, target, 4096)


def choose_baseline_fit_size(total_records: int, eval_batch_size: int) -> int:
    """
    Pick a deterministic sample size for fitting lightweight diagnostic baselines.
    """
    if total_records <= 0:
        return 0
    target = max(eval_batch_size * 128, 4096)
    return min(total_records, target, 16384)


def build_diagnostic_datasets(
    train_dataset: OASSequenceDataset,
    cfg: TrainConfig,
) -> tuple[OASSequenceDataset | RecordSubsetDataset, RecordSubsetDataset | None, RecordSubsetDataset | None]:
    """
    Derive lightweight diagnostic datasets without changing the processed file.

    Returns:
        Tuple of:
        - training dataset used by the optimizer
        - known-target probe sampled from the retained training rows
        - row-random held-out probe removed from training rows
    """
    probe_size = choose_probe_size(len(train_dataset.records), cfg.eval_batch_size)
    if probe_size == 0:
        return train_dataset, None, None

    rng = random.Random(cfg.seed + 30_000)
    indices = list(range(len(train_dataset.records)))
    rng.shuffle(indices)

    row_random_probe_indices = set(indices[:probe_size])
    retained_records = [
        record
        for idx, record in enumerate(train_dataset.records)
        if idx not in row_random_probe_indices
    ]
    row_random_probe_records = [
        train_dataset.records[idx]
        for idx in indices[:probe_size]
    ]

    known_target_probe_size = min(probe_size, len(retained_records))
    known_target_probe_indices = list(range(len(retained_records)))
    rng.shuffle(known_target_probe_indices)
    known_target_probe_records = [
        retained_records[idx]
        for idx in known_target_probe_indices[:known_target_probe_size]
    ]

    return (
        RecordSubsetDataset(retained_records, split="train"),
        RecordSubsetDataset(known_target_probe_records, split="train_probe"),
        RecordSubsetDataset(row_random_probe_records, split="row_random_probe"),
    )


def summarize_target_overlap(
    train_dataset: OASSequenceDataset | RecordSubsetDataset,
    val_dataset: OASSequenceDataset | RecordSubsetDataset,
) -> dict[str, int]:
    """
    Summarize target-key overlap between two datasets.
    """
    train_targets = {record.target_key for record in train_dataset.records if record.target_key}
    val_targets = {record.target_key for record in val_dataset.records if record.target_key}
    return {
        "train_targets": len(train_targets),
        "val_targets": len(val_targets),
        "overlap": len(train_targets & val_targets),
    }


def format_metric_summary(
    metrics: Dict[str, float],
    cfg: TrainConfig,
    prefix: str,
) -> str:
    """
    Render one metric dictionary into a compact log line.
    """
    if is_antigen_stage(cfg.training_stage):
        aux_loss_name = "compatibility_loss"
        aux_acc_name = "compatibility_acc"
    else:
        aux_loss_name = "pair_loss"
        aux_acc_name = "pair_acc"
    summary = (
        f"{prefix}_loss={metrics['loss']:.4f} "
        f"{prefix}_mlm_loss={metrics['mlm_loss']:.4f} "
        f"{prefix}_{aux_loss_name}={metrics[aux_loss_name]:.4f} "
        f"{prefix}_mlm_acc={metrics['mlm_acc']:.4f} "
        f"{prefix}_{aux_acc_name}={metrics[aux_acc_name]:.4f}"
    )
    if is_antigen_stage(cfg.training_stage) and "compatibility_balanced_acc" in metrics:
        summary += (
            f" {prefix}_compat_labeled={int(metrics['compatibility_labeled_count'])}"
            f" {prefix}_compat_bal_acc={metrics['compatibility_balanced_acc']:.4f}"
            f" {prefix}_compat_mcc={metrics['compatibility_mcc']:.4f}"
            f" {prefix}_compat_auroc={metrics['compatibility_auroc']:.4f}"
            f" {prefix}_compat_auprc={metrics['compatibility_auprc']:.4f}"
        )
    if "hcdr3_token_acc" in metrics:
        summary += (
            f" {prefix}_hcdr3_tokens={int(metrics['hcdr3_target_tokens'])}"
            f" {prefix}_hcdr3_spans={int(metrics['hcdr3_valid_spans'])}"
            f" {prefix}_hcdr3_acc={metrics['hcdr3_token_acc']:.4f}"
            f" {prefix}_hcdr3_exact={metrics['hcdr3_span_exact_match']:.4f}"
        )
    return summary


def _json_safe(value: Any) -> Any:
    """
    Convert non-finite floats to null so metrics.jsonl is strict JSON.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def append_metrics_jsonl(
    output_dir: Path,
    record: Dict[str, Any],
) -> None:
    """
    Append one metrics record to the run's JSONL log.
    """
    with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def sample_records_for_diagnostics(
    dataset: OASSequenceDataset | RecordSubsetDataset,
    sample_size: int,
    seed: int,
    split_name: str,
) -> OASSequenceDataset | RecordSubsetDataset:
    """
    Return a deterministic in-memory subset for cheaper diagnostic passes.
    """
    if sample_size <= 0 or len(dataset.records) <= sample_size:
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset.records)))
    rng.shuffle(indices)
    sampled_records = [dataset.records[idx] for idx in indices[:sample_size]]
    return RecordSubsetDataset(sampled_records, split=split_name)


def _masked_metadata_values(batch: Dict[str, Any], key: str) -> list[str]:
    """
    Extract one metadata field for labeled compatibility rows only.
    """
    mask_tensor = batch["compatibility_mask"]
    masked_values: list[str] = []
    for idx, keep in enumerate(mask_tensor.tolist()):
        if not keep:
            continue
        value = batch[key][idx]
        if value is None or value == "":
            masked_values.append("missing")
        else:
            masked_values.append(str(value))
    return masked_values


def fit_group_majority_baselines(
    dataset: OASSequenceDataset | RecordSubsetDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """
    Fit simple group-majority baselines on a deterministic sampled training view.
    """
    if not is_antigen_stage(cfg.training_stage):
        return {}

    fit_size = choose_baseline_fit_size(len(dataset.records), cfg.eval_batch_size)
    fit_dataset = sample_records_for_diagnostics(
        dataset,
        sample_size=fit_size,
        seed=cfg.seed + 40_000,
        split_name="baseline_fit",
    )
    loader = build_eval_loader(fit_dataset, tokenizer, cfg)

    grouped_counts: dict[str, dict[str, Counter[int]]] = {
        "target_keys": defaultdict(Counter),
        "dataset_names": defaultdict(Counter),
        "antibody_format_groups": defaultdict(Counter),
        "antigen_length_buckets": defaultdict(Counter),
    }
    global_counts: Counter[int] = Counter()
    labeled_examples = 0

    for batch in loader:
        labels = batch["compatibility_labels"][batch["compatibility_mask"]].tolist()
        labeled_examples += len(labels)
        global_counts.update(labels)
        for group_name, counters in grouped_counts.items():
            values = _masked_metadata_values(batch, group_name)
            for value, label in zip(values, labels):
                counters[value][int(label)] += 1

    if labeled_examples == 0:
        return {}

    fallback_label = 1 if global_counts[1] >= global_counts[0] else 0
    majority_maps: dict[str, dict[str, int]] = {}
    for group_name, counters in grouped_counts.items():
        majority_maps[group_name] = {
            value: (1 if counts[1] >= counts[0] else 0)
            for value, counts in counters.items()
        }

    return {
        "fit_records": len(fit_dataset.records),
        "fit_labeled_examples": labeled_examples,
        "positive_rate": global_counts[1] / labeled_examples,
        "fallback_label": fallback_label,
        "majority_maps": majority_maps,
    }


def evaluate_group_majority_baselines(
    dataset: OASSequenceDataset | RecordSubsetDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    baselines: dict[str, Any],
) -> dict[str, float]:
    """
    Evaluate simple non-neural baselines on the synthetic compatibility task.
    """
    if not is_antigen_stage(cfg.training_stage) or not baselines:
        return {}

    loader = build_eval_loader(dataset, tokenizer, cfg)
    labeled_examples = 0
    positive_examples = 0
    always_positive_correct = 0
    group_correct = {
        "target_keys": 0,
        "dataset_names": 0,
        "antibody_format_groups": 0,
        "antigen_length_buckets": 0,
    }

    for batch in loader:
        labels = batch["compatibility_labels"][batch["compatibility_mask"]].tolist()
        labeled_examples += len(labels)
        positive_examples += sum(int(label) for label in labels)
        always_positive_correct += sum(int(label) for label in labels)

        for group_name in group_correct:
            values = _masked_metadata_values(batch, group_name)
            group_map = baselines["majority_maps"][group_name]
            fallback_label = baselines["fallback_label"]
            correct = 0
            for value, label in zip(values, labels):
                pred = group_map.get(value, fallback_label)
                if pred == int(label):
                    correct += 1
            group_correct[group_name] += correct

    if labeled_examples == 0:
        return {}

    return {
        "labeled_examples": float(labeled_examples),
        "positive_rate": positive_examples / labeled_examples,
        "always_positive_acc": always_positive_correct / labeled_examples,
        "target_key_majority_acc": group_correct["target_keys"] / labeled_examples,
        "dataset_majority_acc": group_correct["dataset_names"] / labeled_examples,
        "format_majority_acc": group_correct["antibody_format_groups"] / labeled_examples,
        "antigen_bucket_majority_acc": group_correct["antigen_length_buckets"] / labeled_examples,
    }


def format_baseline_summary(
    metrics: dict[str, float],
    prefix: str,
) -> str:
    """
    Render one baseline metrics dictionary into a compact log line.
    """
    return (
        f"{prefix}_labeled={int(metrics['labeled_examples'])} "
        f"{prefix}_pos_rate={metrics['positive_rate']:.4f} "
        f"{prefix}_always_pos_acc={metrics['always_positive_acc']:.4f} "
        f"{prefix}_target_majority_acc={metrics['target_key_majority_acc']:.4f} "
        f"{prefix}_dataset_majority_acc={metrics['dataset_majority_acc']:.4f} "
        f"{prefix}_format_majority_acc={metrics['format_majority_acc']:.4f} "
        f"{prefix}_antigen_bucket_majority_acc={metrics['antigen_bucket_majority_acc']:.4f}"
    )


def build_train_loader(
    dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    epoch: int = 0,
    device: torch.device | None = None,
) -> DataLoader:
    """
    Build the training DataLoader.

    This uses:
      - chain-aware, length-bucketed batch sampling
      - dynamic MLM masking in the collator

    Args:
        dataset:
            Training dataset.
        tokenizer:
            Tokenizer used by the collator.
        cfg:
            Training configuration.
        epoch:
            Current epoch index. This is used to reshuffle batch composition
            reproducibly across epochs.

    Returns:
        A DataLoader ready for one training epoch.
    """
    sampler = ChainLengthBucketBatchSampler(
        dataset=dataset,
        batch_size=cfg.batch_size,
        bucket_width=cfg.bucket_width,
        drop_last=False,
        seed=cfg.seed,
    )
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    if cfg.training_stage == "antigen_real_label_refine" or is_hcdr3_infill_stage(cfg.training_stage):
        collator = AntibodyAntigenRealLabelCollator(
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            mask_probability=cfg.mask_probability,
            hcdr3_span_probability=cfg.hcdr3_span_probability,
            hcdr3_span_min=cfg.hcdr3_span_min,
            hcdr3_span_max=cfg.hcdr3_span_max,
            hcdr3_mask_mode=cfg.hcdr3_mask_mode,
            mask_replacement_strategy=cfg.mask_replacement_strategy,
            shuffle_antigen_probability=0.0,
            rng_seed=cfg.seed + epoch,
        )
    elif cfg.training_stage == "antigen_refine":
        collator = AntibodyAntigenCollator(
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            mask_probability=cfg.mask_probability,
            hcdr3_span_probability=cfg.hcdr3_span_probability,
            hcdr3_span_min=cfg.hcdr3_span_min,
            hcdr3_span_max=cfg.hcdr3_span_max,
            hcdr3_mask_mode=cfg.hcdr3_mask_mode,
            mask_replacement_strategy=cfg.mask_replacement_strategy,
            shuffle_antigen_probability=cfg.shuffle_antigen_probability,
            rng_seed=cfg.seed + epoch,
        )
    else:
        collator = MLMCollator(
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            mask_probability=cfg.mask_probability,
            hcdr3_span_probability=cfg.hcdr3_span_probability,
            hcdr3_span_min=cfg.hcdr3_span_min,
            hcdr3_span_max=cfg.hcdr3_span_max,
            hcdr3_mask_mode=cfg.hcdr3_mask_mode,
            mask_replacement_strategy=cfg.mask_replacement_strategy,
            shuffle_pair_probability=cfg.shuffle_pair_probability,
            rng_seed=cfg.seed + epoch,
        )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.train_num_workers,
        pin_memory=((device.type == "cuda") if device is not None else torch.cuda.is_available()),
        worker_init_fn=seed_worker if cfg.train_num_workers > 0 else None,
    )
    return loader


def build_eval_loader(
    dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> DataLoader:
    """
    Build a deterministic-ish evaluation DataLoader.

    We rebuild the collator fresh for evaluation so the masking pattern starts
    from a fixed seed each time. That makes validation more stable.

    Args:
        dataset:
            Validation dataset.
        tokenizer:
            Tokenizer used by the collator.
        cfg:
            Training configuration.

    Returns:
        A DataLoader for validation.
    """
    sampler = ChainLengthBucketBatchSampler(
        dataset=dataset,
        batch_size=cfg.eval_batch_size,
        bucket_width=cfg.bucket_width,
        drop_last=False,
        seed=cfg.seed + 10_000,
    )

    if cfg.training_stage == "antigen_refine":
        collator = AntibodyAntigenCollator(
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            mask_probability=cfg.mask_probability,
            hcdr3_span_probability=cfg.hcdr3_span_probability,
            hcdr3_span_min=cfg.hcdr3_span_min,
            hcdr3_span_max=cfg.hcdr3_span_max,
            hcdr3_mask_mode=cfg.hcdr3_mask_mode,
            mask_replacement_strategy=cfg.mask_replacement_strategy,
            shuffle_antigen_probability=cfg.shuffle_antigen_probability,
            rng_seed=cfg.seed + 20_000,
        )
    elif cfg.training_stage == "antigen_real_label_refine" or is_hcdr3_infill_stage(cfg.training_stage):
        collator = AntibodyAntigenRealLabelCollator(
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            mask_probability=cfg.mask_probability,
            hcdr3_span_probability=cfg.hcdr3_span_probability,
            hcdr3_span_min=cfg.hcdr3_span_min,
            hcdr3_span_max=cfg.hcdr3_span_max,
            hcdr3_mask_mode=cfg.hcdr3_mask_mode,
            mask_replacement_strategy=cfg.mask_replacement_strategy,
            shuffle_antigen_probability=0.0,
            rng_seed=cfg.seed + 20_000,
        )
    else:
        collator = MLMCollator(
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            mask_probability=cfg.mask_probability,
            hcdr3_span_probability=cfg.hcdr3_span_probability,
            hcdr3_span_min=cfg.hcdr3_span_min,
            hcdr3_span_max=cfg.hcdr3_span_max,
            hcdr3_mask_mode=cfg.hcdr3_mask_mode,
            mask_replacement_strategy=cfg.mask_replacement_strategy,
            shuffle_pair_probability=cfg.shuffle_pair_probability,
            rng_seed=cfg.seed + 20_000,
        )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.eval_num_workers,
        pin_memory=((device.type == "cuda") if device is not None else torch.cuda.is_available()),
        worker_init_fn=seed_worker if cfg.eval_num_workers > 0 else None,
    )
    return loader


def build_model(
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build the MLM model and move it to the chosen device.

    Args:
        tokenizer:
            Tokenizer that defines vocabulary size and pad token ID.
        cfg:
            Training configuration.
        device:
            Target torch.device.

    Returns:
        An AntibodyMLM instance on the target device.
    """
    model_cfg = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=cfg.max_length,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    )
    if is_antigen_stage(cfg.training_stage):
        model = AntibodyAntigenCrossAttention(model_cfg).to(device)
    else:
        model = AntibodyMLM(model_cfg).to(device)
    return model


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> AdamW:
    """
    Build the optimizer used for MLM training.

    Args:
        model:
            The MLM model.
        cfg:
            Training configuration.

    Returns:
        An initialized AdamW optimizer.
    """
    # Weight decay should not be applied to biases or LayerNorm gains/biases
    # (1-D tensors); decaying them is a known training defect that pulls norm
    # gains toward zero. Split parameters into a decay and a no-decay group.
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )


def build_lr_scheduler(
    optimizer: AdamW, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """
    Build a linear LR warmup scheduler, or None when warmup is disabled.

    For optimizer step ``s`` (0-based), the LR multiplier is
    ``min(1.0, (s + 1) / warmup_steps)``; after ``warmup_steps`` updates the LR
    stays at ``cfg.learning_rate``. Returns ``None`` when ``warmup_steps == 0``
    so the constant-LR path is unchanged.
    """
    if cfg.warmup_steps <= 0:
        return None

    warmup = cfg.warmup_steps

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move all tensor values in a batch dictionary onto a device.

    Args:
        batch:
            Dictionary containing torch.Tensor values.
        device:
            Target device.

    Returns:
        New dictionary with all tensor values moved to `device`.
    """
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy only on masked-language-model target positions.

    Args:
        logits:
            Tensor of shape [batch_size, seq_len, vocab_size].
        labels:
            Tensor of shape [batch_size, seq_len] where target positions contain
            token IDs and non-target positions contain -100.

    Returns:
        Masked-token accuracy as a Python float.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()


def masked_accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """
    Return ``(correct, total)`` masked-token counts for token-level pooling.

    Unlike ``masked_accuracy`` (a per-batch mean), this returns raw sufficient
    statistics so accuracy can be pooled across batches of unequal masked-token
    counts. ``total`` is the number of non-``-100`` label positions; ``correct``
    is how many were predicted correctly. Returns ``(0, 0)`` for an empty batch.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    total = int(mask.sum().item())
    if total == 0:
        return 0, 0
    correct = int((preds[mask] == labels[mask]).sum().item())
    return correct, total


def hcdr3_metric_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hcdr3_target_mask: torch.Tensor,
    hcdr3_token_start: torch.Tensor,
    hcdr3_token_end: torch.Tensor,
    hcdr3_valid_mask: torch.Tensor,
) -> dict[str, int]:
    """
    Count HCDR3-specific infilling successes for one batch.

    ``hcdr3_target_mask`` marks the target residues inside the heavy-chain CDR3
    interval. Token accuracy is computed over those target residues only.
    Whole-span exact match is stricter: a row contributes to the denominator
    only when the encoded HCDR3 span is valid and every token in the span was
    targeted. This makes the metric meaningful for fixed-length full-span
    infilling while avoiding a misleading exact-match score during ordinary
    sampled-span MLM training.
    """
    preds = logits.argmax(dim=-1)
    target_mask = hcdr3_target_mask.bool() & (labels != -100)
    target_tokens = int(target_mask.sum().item())
    correct_tokens = int((preds[target_mask] == labels[target_mask]).sum().item()) if target_tokens else 0

    exact_matches = 0
    valid_spans = 0
    batch_size = labels.size(0)
    for idx in range(batch_size):
        if not bool(hcdr3_valid_mask[idx].item()):
            continue
        start = int(hcdr3_token_start[idx].item())
        end = int(hcdr3_token_end[idx].item())
        if start < 0 or end <= start or end > labels.size(1):
            continue
        span_target_mask = target_mask[idx, start:end]
        if int(span_target_mask.sum().item()) != (end - start):
            continue
        valid_spans += 1
        if torch.equal(preds[idx, start:end], labels[idx, start:end]):
            exact_matches += 1

    return {
        "hcdr3_correct_tokens": correct_tokens,
        "hcdr3_target_tokens": target_tokens,
        "hcdr3_exact_matches": exact_matches,
        "hcdr3_valid_spans": valid_spans,
    }


def finalize_hcdr3_metrics(counts: dict[str, int]) -> dict[str, float]:
    """
    Convert accumulated HCDR3 counts into stable scalar metrics.
    """
    target_tokens = counts.get("hcdr3_target_tokens", 0)
    valid_spans = counts.get("hcdr3_valid_spans", 0)
    return {
        "hcdr3_token_acc": (
            counts.get("hcdr3_correct_tokens", 0) / target_tokens
            if target_tokens > 0
            else float("nan")
        ),
        "hcdr3_span_exact_match": (
            counts.get("hcdr3_exact_matches", 0) / valid_spans
            if valid_spans > 0
            else float("nan")
        ),
        "hcdr3_target_tokens": float(target_tokens),
        "hcdr3_valid_spans": float(valid_spans),
    }


def pair_classification_accuracy(
    pair_logits: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
) -> float:
    """
    Compute pair-compatibility accuracy on valid paired examples only.

    Args:
        pair_logits:
            Tensor of shape [batch_size, 2] containing native-vs-shuffled logits.
        pair_labels:
            Tensor of shape [batch_size] containing integer class labels.
        pair_mask:
            Tensor of shape [batch_size] where True marks examples that
            represent actual paired records and therefore participate in the
            auxiliary objective.

    Returns:
        Classification accuracy as a Python float. Returns 0.0 if the batch has
        no paired examples.
    """
    if pair_mask.sum().item() == 0:
        return 0.0
    preds = pair_logits.argmax(dim=-1)
    return (preds[pair_mask] == pair_labels[pair_mask]).float().mean().item()


def compatibility_classification_accuracy(
    compatibility_logits: torch.Tensor,
    compatibility_labels: torch.Tensor,
    compatibility_mask: torch.Tensor,
) -> float:
    """
    Compute antibody-antigen compatibility accuracy on labeled rows only.
    """
    if compatibility_mask.sum().item() == 0:
        return 0.0
    preds = compatibility_logits.argmax(dim=-1)
    return (preds[compatibility_mask] == compatibility_labels[compatibility_mask]).float().mean().item()


def masked_classification_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[int, int]:
    """
    Count correct masked classifications and the number of labeled rows.

    This helper lets epoch metrics aggregate classification accuracy over all
    labeled examples instead of averaging per-batch accuracies, which can skew
    results when batches contain different numbers of supervised rows.
    """
    if mask.sum().item() == 0:
        return 0, 0
    preds = logits.argmax(dim=-1)
    correct = int((preds[mask] == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """
    Compute AUROC for binary labels using average ranks for tied scores.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(s)
    sorted_scores = s[order]
    ranks = np.empty_like(s, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)


def binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """
    Compute average precision / area under the precision-recall curve.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    pos_count = int((y == 1).sum())
    if pos_count == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    tp = np.cumsum(sorted_labels == 1)
    seen = np.arange(1, len(sorted_labels) + 1)
    precision = tp / seen
    return float(precision[sorted_labels == 1].sum() / pos_count)


def compatibility_binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    preds: Sequence[int],
) -> Dict[str, float]:
    """
    Compute compatibility metrics over all labeled rows in an epoch/eval pass.
    """
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    labeled = int(y.size)
    if labeled == 0:
        return {
            "compatibility_labeled_count": 0.0,
            "compatibility_positive_rate": float("nan"),
            "compatibility_precision": float("nan"),
            "compatibility_recall": float("nan"),
            "compatibility_specificity": float("nan"),
            "compatibility_balanced_acc": float("nan"),
            "compatibility_mcc": float("nan"),
            "compatibility_auroc": float("nan"),
            "compatibility_auprc": float("nan"),
            "compatibility_tp": 0.0,
            "compatibility_tn": 0.0,
            "compatibility_fp": 0.0,
            "compatibility_fn": 0.0,
        }

    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    if np.isfinite(recall) and np.isfinite(specificity):
        balanced_acc = (recall + specificity) / 2.0
    else:
        balanced_acc = float("nan")

    mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(mcc_denom) if mcc_denom > 0 else float("nan")

    return {
        "compatibility_labeled_count": float(labeled),
        "compatibility_positive_rate": float((y == 1).sum() / labeled),
        "compatibility_precision": float(precision),
        "compatibility_recall": float(recall),
        "compatibility_specificity": float(specificity),
        "compatibility_balanced_acc": float(balanced_acc),
        "compatibility_mcc": float(mcc),
        "compatibility_auroc": float(binary_auroc(labels, scores)),
        "compatibility_auprc": float(binary_average_precision(labels, scores)),
        "compatibility_tp": float(tp),
        "compatibility_tn": float(tn),
        "compatibility_fp": float(fp),
        "compatibility_fn": float(fn),
    }


def _make_progress_bar(
    iterable,
    *,
    total: int | None = None,
    desc: str,
    cfg: TrainConfig,
):
    """
    Wrap an iterable in a tqdm progress bar when interactive progress is enabled.
    """
    disable = (not cfg.show_progress) or (not sys.stderr.isatty())
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True, disable=disable)


def run_smoke_test(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    use_amp: bool,
    training_stage: str,
    grad_clip_norm: float = 1.0,
) -> None:
    """
    Run a minimal forward/backward/step proof of implementation.

    This is the fastest way to prove that:
    - the dataloader returns valid batches
    - the model forward pass works
    - loss computation works
    - gradients flow
    - optimizer.step() works

    Args:
        model:
            The MLM model.
        train_loader:
            Training DataLoader.
        optimizer:
            Optimizer.
        device:
            Target device.
        use_amp:
            Whether AMP should be used.

    Returns:
        None.
    """
    model.train()
    batch = next(iter(train_loader))
    batch = move_batch_to_device(batch, device)

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        if is_antigen_stage(training_stage):
            logits, compatibility_logits = model(
                antibody_input_ids=batch["antibody_input_ids"],
                antibody_attention_mask=batch["antibody_attention_mask"],
                antigen_input_ids=batch["antigen_input_ids"],
                antigen_attention_mask=batch["antigen_attention_mask"],
            )
            losses = model.compute_losses(
                mlm_logits=logits,
                labels=batch["antibody_labels"],
                compatibility_logits=compatibility_logits,
                compatibility_labels=batch["compatibility_labels"],
                compatibility_mask=batch["compatibility_mask"],
                compatibility_loss_weight=0.0 if is_hcdr3_infill_stage(training_stage) else 1.0,
            )
            print("smoke_test/antibody_input_ids:", tuple(batch["antibody_input_ids"].shape))
            print("smoke_test/antigen_input_ids:", tuple(batch["antigen_input_ids"].shape))
            print("smoke_test/logits:", tuple(logits.shape))
            print("smoke_test/compatibility_logits:", tuple(compatibility_logits.shape))
            compatibility_mask_count = int(batch["compatibility_mask"].sum().item())
            compatibility_positive_count = int(
                batch["compatibility_labels"][batch["compatibility_mask"]].sum().item()
            )
            print(
                "smoke_test/compatibility_batch:"
                f" labeled={compatibility_mask_count}/{batch['compatibility_mask'].numel()}"
                f" positives={compatibility_positive_count}"
            )
            print(
                "smoke_test/hcdr3_batch:"
                f" target_tokens={int(batch['hcdr3_target_mask'].sum().item())}"
                f" full_target_spans={int(batch['hcdr3_valid_mask'].sum().item())}"
            )
        else:
            logits, pair_logits = model.forward_with_pairing(batch["input_ids"], batch["attention_mask"])
            losses = model.compute_losses(
                mlm_logits=logits,
                labels=batch["labels"],
                pair_logits=pair_logits,
                pair_labels=batch["pair_labels"],
                pair_mask=batch["pair_mask"],
                pair_loss_weight=1.0,
            )
            print("smoke_test/input_ids:", tuple(batch["input_ids"].shape))
            print("smoke_test/logits:", tuple(logits.shape))
            print("smoke_test/pair_logits:", tuple(pair_logits.shape))
        loss = losses["loss"]

    print("smoke_test/loss:", float(loss.detach()))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()

    print("smoke_test/backward: ok")
    print("smoke_test/optimizer_step: ok")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run one full validation pass.

    Args:
        model:
            The MLM model.
        val_dataset:
            Validation dataset.
        tokenizer:
            Tokenizer used to rebuild the evaluation loader.
        cfg:
            Training configuration.
        device:
            Target device.

    Returns:
        Dictionary containing averaged validation metrics.
    """
    model.eval()
    val_loader = build_eval_loader(val_dataset, tokenizer, cfg, device=device)

    total_mlm_loss_weighted = 0.0
    total_mlm_correct = 0
    total_mlm_tokens = 0
    total_aux_loss_weighted = 0.0
    total_aux_correct = 0
    total_aux_labeled = 0
    total_batches = 0
    _aux_weight = (
        cfg.compatibility_loss_weight
        if is_antigen_stage(cfg.training_stage)
        else cfg.pair_loss_weight
    )
    compatibility_labels_all: list[int] = []
    compatibility_scores_all: list[float] = []
    compatibility_preds_all: list[int] = []
    hcdr3_counts = {
        "hcdr3_correct_tokens": 0,
        "hcdr3_target_tokens": 0,
        "hcdr3_exact_matches": 0,
        "hcdr3_valid_spans": 0,
    }

    progress = _make_progress_bar(
        val_loader,
        total=len(val_loader),
        desc="eval",
        cfg=cfg,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        if is_antigen_stage(cfg.training_stage):
            logits, compatibility_logits = model(
                antibody_input_ids=batch["antibody_input_ids"],
                antibody_attention_mask=batch["antibody_attention_mask"],
                antigen_input_ids=batch["antigen_input_ids"],
                antigen_attention_mask=batch["antigen_attention_mask"],
            )
            losses = model.compute_losses(
                mlm_logits=logits,
                labels=batch["antibody_labels"],
                compatibility_logits=compatibility_logits,
                compatibility_labels=batch["compatibility_labels"],
                compatibility_mask=batch["compatibility_mask"],
                compatibility_loss_weight=cfg.compatibility_loss_weight,
            )
            loss = losses["loss"]
            mlm_loss = losses["mlm_loss"]
            aux_loss = losses["compatibility_loss"]
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits, batch["antibody_labels"])
            aux_acc = compatibility_classification_accuracy(
                compatibility_logits,
                batch["compatibility_labels"],
                batch["compatibility_mask"],
            )
            aux_correct, aux_labeled = masked_classification_counts(
                compatibility_logits,
                batch["compatibility_labels"],
                batch["compatibility_mask"],
            )
            batch_hcdr3_counts = hcdr3_metric_counts(
                logits,
                batch["antibody_labels"],
                batch["hcdr3_target_mask"],
                batch["hcdr3_token_start"],
                batch["hcdr3_token_end"],
                batch["hcdr3_valid_mask"],
            )
            mask = batch["compatibility_mask"].bool()
            if mask.sum().item() > 0:
                compatibility_labels_all.extend(
                    batch["compatibility_labels"][mask].detach().cpu().tolist()
                )
                compatibility_scores_all.extend(
                    torch.softmax(compatibility_logits.detach(), dim=-1)[mask, 1].cpu().tolist()
                )
                compatibility_preds_all.extend(
                    compatibility_logits.detach().argmax(dim=-1)[mask].cpu().tolist()
                )
            aux_loss_name = "compatibility_loss"
            aux_acc_name = "compatibility_acc"
        else:
            logits, pair_logits = model.forward_with_pairing(batch["input_ids"], batch["attention_mask"])
            losses = model.compute_losses(
                mlm_logits=logits,
                labels=batch["labels"],
                pair_logits=pair_logits,
                pair_labels=batch["pair_labels"],
                pair_mask=batch["pair_mask"],
                pair_loss_weight=cfg.pair_loss_weight,
            )
            loss = losses["loss"]
            mlm_loss = losses["mlm_loss"]
            aux_loss = losses["pair_loss"]
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits, batch["labels"])
            aux_acc = pair_classification_accuracy(pair_logits, batch["pair_labels"], batch["pair_mask"])
            aux_correct, aux_labeled = masked_classification_counts(
                pair_logits,
                batch["pair_labels"],
                batch["pair_mask"],
            )
            batch_hcdr3_counts = hcdr3_metric_counts(
                logits,
                batch["labels"],
                batch["hcdr3_target_mask"],
                batch["hcdr3_token_start"],
                batch["hcdr3_token_end"],
                batch["hcdr3_valid_mask"],
            )
            aux_loss_name = "pair_loss"
            aux_acc_name = "pair_acc"

        total_mlm_loss_weighted += float(mlm_loss.item()) * mlm_tokens
        total_mlm_correct += mlm_correct
        total_mlm_tokens += mlm_tokens
        total_aux_loss_weighted += float(aux_loss.item()) * aux_labeled
        total_aux_correct += aux_correct
        total_aux_labeled += aux_labeled
        total_batches += 1
        for key, value in batch_hcdr3_counts.items():
            hcdr3_counts[key] += value
        running_aux_acc = (total_aux_correct / total_aux_labeled) if total_aux_labeled > 0 else 0.0
        running_mlm_loss = (total_mlm_loss_weighted / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
        running_mlm_acc = (total_mlm_correct / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
        running_aux_loss = (total_aux_loss_weighted / total_aux_labeled) if total_aux_labeled > 0 else 0.0
        running_hcdr3 = finalize_hcdr3_metrics(hcdr3_counts)
        progress.set_postfix(
            loss=f"{running_mlm_loss + _aux_weight * running_aux_loss:.4f}",
            mlm_loss=f"{running_mlm_loss:.4f}",
            **{aux_loss_name: f"{running_aux_loss:.4f}"},
            mlm_acc=f"{running_mlm_acc:.4f}",
            **{aux_acc_name: f"{running_aux_acc:.4f}"},
            hcdr3_acc=f"{running_hcdr3['hcdr3_token_acc']:.4f}",
        )

    progress.close()

    if total_batches == 0:
        metrics = {
            "loss": float("nan"),
            "mlm_loss": float("nan"),
            "mlm_acc": float("nan"),
        }
        if is_antigen_stage(cfg.training_stage):
            metrics["compatibility_loss"] = float("nan")
            metrics["compatibility_acc"] = float("nan")
            metrics.update(compatibility_binary_metrics([], [], []))
        else:
            metrics["pair_loss"] = float("nan")
            metrics["pair_acc"] = float("nan")
        metrics.update(finalize_hcdr3_metrics(hcdr3_counts))
        return metrics

    mlm_loss = (total_mlm_loss_weighted / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    mlm_acc = (total_mlm_correct / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    aux_loss = (total_aux_loss_weighted / total_aux_labeled) if total_aux_labeled > 0 else 0.0
    metrics = {
        "loss": mlm_loss + _aux_weight * aux_loss,
        "mlm_loss": mlm_loss,
        "mlm_acc": mlm_acc,
    }
    metrics.update(finalize_hcdr3_metrics(hcdr3_counts))
    if is_antigen_stage(cfg.training_stage):
        metrics["compatibility_loss"] = aux_loss
        metrics["compatibility_acc"] = (
            total_aux_correct / total_aux_labeled if total_aux_labeled > 0 else float("nan")
        )
        metrics.update(
            compatibility_binary_metrics(
                compatibility_labels_all,
                compatibility_scores_all,
                compatibility_preds_all,
            )
        )
    else:
        metrics["pair_loss"] = aux_loss
        metrics["pair_acc"] = (
            total_aux_correct / total_aux_labeled if total_aux_labeled > 0 else float("nan")
        )
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    train_dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model:
            The MLM model.
        train_dataset:
            Training dataset.
        tokenizer:
            Tokenizer used by the collator.
        optimizer:
            Optimizer.
        cfg:
            Training configuration.
        device:
            Target device.
        epoch:
            Zero-based epoch index.

    Returns:
        Dictionary containing averaged training metrics for the epoch.
    """
    model.train()
    train_loader = build_train_loader(train_dataset, tokenizer, cfg, epoch=epoch, device=device)
    # The GradScaler and warmup scheduler are created once in main() and reused
    # across epochs so the adaptive loss scale and warmup step count are not
    # reset at every epoch boundary (and survive --resume-from-last).

    total_mlm_loss_weighted = 0.0
    total_mlm_correct = 0
    total_mlm_tokens = 0
    total_aux_loss_weighted = 0.0
    total_aux_correct = 0
    total_aux_labeled = 0
    total_batches = 0
    _aux_weight = (
        cfg.compatibility_loss_weight
        if is_antigen_stage(cfg.training_stage)
        else cfg.pair_loss_weight
    )
    compatibility_labels_all: list[int] = []
    compatibility_scores_all: list[float] = []
    compatibility_preds_all: list[int] = []
    hcdr3_counts = {
        "hcdr3_correct_tokens": 0,
        "hcdr3_target_tokens": 0,
        "hcdr3_exact_matches": 0,
        "hcdr3_valid_spans": 0,
    }

    progress = _make_progress_bar(
        train_loader,
        total=len(train_loader),
        desc=f"train {epoch + 1}/{cfg.epochs}",
        cfg=cfg,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
            if is_antigen_stage(cfg.training_stage):
                logits, compatibility_logits = model(
                    antibody_input_ids=batch["antibody_input_ids"],
                    antibody_attention_mask=batch["antibody_attention_mask"],
                    antigen_input_ids=batch["antigen_input_ids"],
                    antigen_attention_mask=batch["antigen_attention_mask"],
                )
                losses = model.compute_losses(
                    mlm_logits=logits,
                    labels=batch["antibody_labels"],
                    compatibility_logits=compatibility_logits,
                    compatibility_labels=batch["compatibility_labels"],
                    compatibility_mask=batch["compatibility_mask"],
                    compatibility_loss_weight=cfg.compatibility_loss_weight,
                )
                loss = losses["loss"]
                mlm_loss = losses["mlm_loss"]
                aux_loss = losses["compatibility_loss"]
            else:
                logits, pair_logits = model.forward_with_pairing(batch["input_ids"], batch["attention_mask"])
                losses = model.compute_losses(
                    mlm_logits=logits,
                    labels=batch["labels"],
                    pair_logits=pair_logits,
                    pair_labels=batch["pair_labels"],
                    pair_mask=batch["pair_mask"],
                    pair_loss_weight=cfg.pair_loss_weight,
                )
                loss = losses["loss"]
                mlm_loss = losses["mlm_loss"]
                aux_loss = losses["pair_loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite training loss ({float(loss)}) before backward at "
                f"epoch {epoch + 1}; check for empty-target batches or divergence"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # Advance the LR warmup only when AMP did not skip the optimizer step
        # (it skips on inf/NaN grads), so warmup progress tracks real updates.
        if scheduler is not None and scaler.get_scale() >= scale_before:
            scheduler.step()

        if is_antigen_stage(cfg.training_stage):
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits.detach(), batch["antibody_labels"])
            aux_acc = compatibility_classification_accuracy(
                compatibility_logits.detach(),
                batch["compatibility_labels"],
                batch["compatibility_mask"],
            )
            aux_correct, aux_labeled = masked_classification_counts(
                compatibility_logits.detach(),
                batch["compatibility_labels"],
                batch["compatibility_mask"],
            )
            batch_hcdr3_counts = hcdr3_metric_counts(
                logits.detach(),
                batch["antibody_labels"],
                batch["hcdr3_target_mask"],
                batch["hcdr3_token_start"],
                batch["hcdr3_token_end"],
                batch["hcdr3_valid_mask"],
            )
            mask = batch["compatibility_mask"].bool()
            if mask.sum().item() > 0:
                compatibility_labels_all.extend(
                    batch["compatibility_labels"][mask].detach().cpu().tolist()
                )
                compatibility_scores_all.extend(
                    torch.softmax(compatibility_logits.detach(), dim=-1)[mask, 1].cpu().tolist()
                )
                compatibility_preds_all.extend(
                    compatibility_logits.detach().argmax(dim=-1)[mask].cpu().tolist()
                )
            aux_loss_name = "compatibility_loss"
            aux_acc_name = "compatibility_acc"
        else:
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits.detach(), batch["labels"])
            aux_acc = pair_classification_accuracy(
                pair_logits.detach(),
                batch["pair_labels"],
                batch["pair_mask"],
            )
            aux_correct, aux_labeled = masked_classification_counts(
                pair_logits.detach(),
                batch["pair_labels"],
                batch["pair_mask"],
            )
            batch_hcdr3_counts = hcdr3_metric_counts(
                logits.detach(),
                batch["labels"],
                batch["hcdr3_target_mask"],
                batch["hcdr3_token_start"],
                batch["hcdr3_token_end"],
                batch["hcdr3_valid_mask"],
            )
            aux_loss_name = "pair_loss"
            aux_acc_name = "pair_acc"

        total_mlm_loss_weighted += float(mlm_loss.item()) * mlm_tokens
        total_mlm_correct += mlm_correct
        total_mlm_tokens += mlm_tokens
        total_aux_loss_weighted += float(aux_loss.item()) * aux_labeled
        total_aux_correct += aux_correct
        total_aux_labeled += aux_labeled
        total_batches += 1
        for key, value in batch_hcdr3_counts.items():
            hcdr3_counts[key] += value
        running_aux_acc = (total_aux_correct / total_aux_labeled) if total_aux_labeled > 0 else 0.0
        running_mlm_loss = (total_mlm_loss_weighted / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
        running_mlm_acc = (total_mlm_correct / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
        running_aux_loss = (total_aux_loss_weighted / total_aux_labeled) if total_aux_labeled > 0 else 0.0
        running_hcdr3 = finalize_hcdr3_metrics(hcdr3_counts)
        progress.set_postfix(
            loss=f"{running_mlm_loss + _aux_weight * running_aux_loss:.4f}",
            mlm_loss=f"{running_mlm_loss:.4f}",
            **{aux_loss_name: f"{running_aux_loss:.4f}"},
            mlm_acc=f"{running_mlm_acc:.4f}",
            **{aux_acc_name: f"{running_aux_acc:.4f}"},
            hcdr3_acc=f"{running_hcdr3['hcdr3_token_acc']:.4f}",
        )

    progress.close()

    if total_batches == 0:
        metrics = {
            "loss": float("nan"),
            "mlm_loss": float("nan"),
            "mlm_acc": float("nan"),
        }
        if is_antigen_stage(cfg.training_stage):
            metrics["compatibility_loss"] = float("nan")
            metrics["compatibility_acc"] = float("nan")
            metrics.update(compatibility_binary_metrics([], [], []))
        else:
            metrics["pair_loss"] = float("nan")
            metrics["pair_acc"] = float("nan")
        metrics.update(finalize_hcdr3_metrics(hcdr3_counts))
        return metrics

    mlm_loss = (total_mlm_loss_weighted / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    mlm_acc = (total_mlm_correct / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    aux_loss = (total_aux_loss_weighted / total_aux_labeled) if total_aux_labeled > 0 else 0.0
    metrics = {
        "loss": mlm_loss + _aux_weight * aux_loss,
        "mlm_loss": mlm_loss,
        "mlm_acc": mlm_acc,
    }
    metrics.update(finalize_hcdr3_metrics(hcdr3_counts))
    if is_antigen_stage(cfg.training_stage):
        metrics["compatibility_loss"] = aux_loss
        metrics["compatibility_acc"] = (
            total_aux_correct / total_aux_labeled if total_aux_labeled > 0 else float("nan")
        )
        metrics.update(
            compatibility_binary_metrics(
                compatibility_labels_all,
                compatibility_scores_all,
                compatibility_preds_all,
            )
        )
    else:
        metrics["pair_loss"] = aux_loss
        metrics["pair_acc"] = (
            total_aux_correct / total_aux_labeled if total_aux_labeled > 0 else float("nan")
        )
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    cfg: TrainConfig,
    epoch: int,
    val_loss: float,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        path:
            Destination checkpoint path.
        model:
            The MLM model.
        optimizer:
            Optimizer.
        cfg:
            Training configuration.
        epoch:
            Epoch number being saved.
        val_loss:
            Validation loss associated with this checkpoint.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the numpy RNG state as plain primitives rather than the raw
    # ndarray it returns, so the checkpoint loads under torch.load's default
    # weights_only=True (PyTorch >= 2.6) instead of failing to unpickle.
    np_state = np.random.get_state()
    rng_state = {
        "python": random.getstate(),
        "numpy": {
            "name": np_state[0],
            "keys": [int(k) for k in np_state[1]],
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": rng_state,
            "val_loss": val_loss,
            "train_config": asdict(cfg),
        },
        path
    )
    
def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint into the model (and optionally optimizer).

    Args:
        path: Path to checkpoint file.
        model: Model to load into.
        optimizer: Optional optimizer to restore.
        map_location: Device mapping for torch.load.
        strict: Whether to require an exact key match for model weights.

    Returns:
        The full checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=map_location)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"[checkpoint] loaded <- {path}")
    if not strict:
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing:
            print(f"[checkpoint] init missing keys (left randomly initialized): {missing}")
        if unexpected:
            print(f"[checkpoint] ignored unexpected keys from checkpoint: {unexpected}")
    return checkpoint


def restore_rng_state(rng_state: dict | None) -> None:
    """
    Restore Python/NumPy/Torch (CPU+CUDA) RNG state captured by save_checkpoint.

    Restoring (rather than re-seeding from cfg.seed) is what makes a resumed run
    reproduce the dropout/sampling stream of an uninterrupted run. It is called
    only on --resume-from-last, after set_seed(cfg.seed), so it overrides the
    fresh seed with the interrupted run's exact RNG position.
    """
    if not rng_state:
        return
    python_state = rng_state["python"]
    # random.getstate() round-trips through torch.save as nested lists; rebuild
    # the (version, tuple-of-ints, gauss) shape setstate requires.
    if isinstance(python_state, list):
        python_state = (python_state[0], tuple(python_state[1]), python_state[2])
    random.setstate(python_state)
    np_blob = rng_state["numpy"]
    np.random.set_state(
        (
            np_blob["name"],
            np.array(np_blob["keys"], dtype=np.uint32),
            np_blob["pos"],
            np_blob["has_gauss"],
            np_blob["cached_gaussian"],
        )
    )
    torch_state = rng_state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.to("cpu", torch.uint8))
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.to("cpu", torch.uint8) for s in cuda_state])
        except Exception as exc:  # device-count mismatch, etc.
            print(f"[checkpoint] could not restore CUDA RNG state: {exc}")


def build_antigen_refine_init_state_dict(
    checkpoint_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Translate an antibody-only / paired-refine checkpoint into the subset of
    weights that can initialize the dual-stream antibody-antigen model.

    Initialization policy:
    - copy `sequence_encoder.*` into both `antibody_encoder.*` and `antigen_encoder.*`
    - copy `lm_head.*` directly
    - intentionally do not initialize cross-attention, fusion, or compatibility
      layers from the checkpoint
    - intentionally ignore `pair_head.*`
    """
    translated: Dict[str, torch.Tensor] = {}

    for key, value in checkpoint_state_dict.items():
        if key.startswith("sequence_encoder."):
            suffix = key[len("sequence_encoder."):]
            translated[f"antibody_encoder.{suffix}"] = value
            translated[f"antigen_encoder.{suffix}"] = value
        elif (
            key.startswith("token_embedding.")
            or key.startswith("position_embedding.")
            or key.startswith("encoder.")
            or key.startswith("final_norm.")
        ):
            translated[f"antibody_encoder.{key}"] = value
            translated[f"antigen_encoder.{key}"] = value
        elif key.startswith("lm_head."):
            translated[key] = value

    return translated


def initialize_antigen_refine_from_checkpoint(
    path: Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    Warm-start the dual-stream antigen model from a paired-refine checkpoint.

    This clones the pretrained antibody sequence encoder into both the
    antibody and antigen branches, reuses the MLM head, and leaves the new
    interaction/classification layers randomly initialized.
    """
    checkpoint = torch.load(path, map_location=map_location)
    checkpoint_state_dict = checkpoint["model_state_dict"]
    has_dual_stream_weights = any(
        key.startswith("antibody_encoder.") for key in checkpoint_state_dict
    )
    if has_dual_stream_weights:
        incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
        reused_message = "dual-stream checkpoint weights"
    else:
        translated_state_dict = build_antigen_refine_init_state_dict(checkpoint_state_dict)
        incompatible = model.load_state_dict(translated_state_dict, strict=False)
        reused_message = "antibody_encoder.*, antigen_encoder.*, lm_head.*"

    print(f"[checkpoint] antigen_refine init <- {path}")
    print(f"[checkpoint] antigen_refine reused components: {reused_message}")
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"[checkpoint] antigen_refine missing keys (left randomly initialized): {missing}")
    if unexpected:
        print(f"[checkpoint] antigen_refine unexpected translated keys: {unexpected}")

    return checkpoint


def validate_init_checkpoint_compatibility(
    cfg: TrainConfig,
    init_ckpt_path: Path | None,
) -> None:
    """
    Validate architecture compatibility between run config and init checkpoint.

    Args:
        cfg:
            Current run config.
        init_ckpt_path:
            Optional initialization checkpoint path.

    Returns:
        None.
    """
    if init_ckpt_path is None:
        return

    checkpoint = torch.load(init_ckpt_path, map_location="cpu")
    train_cfg = checkpoint.get("train_config")
    if not isinstance(train_cfg, dict):
        return

    keys_to_match = ("d_model", "n_heads", "n_layers", "d_ff", "dropout", "max_length")
    mismatches: list[str] = []
    for key in keys_to_match:
        ckpt_value = train_cfg.get(key)
        run_value = getattr(cfg, key, None)
        if ckpt_value is None:
            continue
        if run_value != ckpt_value:
            mismatches.append(f"{key}: checkpoint={ckpt_value}, run={run_value}")

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "init_checkpoint architecture mismatch. Use the same base-model "
            f"hyperparameters for refinement. Mismatches: {details}"
        )


def validate_checkpoint_plan(
    cfg: TrainConfig,
    output_dir: Path,
) -> Path | None:
    """
    Validate checkpoint initialization/resume settings for a run.

    Args:
        cfg:
            Training configuration.
        output_dir:
            Directory where this run writes checkpoints.

    Returns:
        Resolved initialization checkpoint path, or None when not used.
    """
    init_ckpt_path: Path | None = None
    if cfg.init_checkpoint:
        init_ckpt_path = Path(cfg.init_checkpoint).expanduser().resolve()
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"init_checkpoint does not exist: {init_ckpt_path}")

    output_dir_resolved = output_dir.resolve()
    if init_ckpt_path is not None and init_ckpt_path.parent == output_dir_resolved:
        raise ValueError(
            "init_checkpoint is inside output_dir. Choose a different output_dir "
            "to keep base pretraining and refinement checkpoints separated."
        )

    return init_ckpt_path

def main() -> None:
    """
    Main entrypoint for MLM training.

    Workflow:
    1. Parse config
    2. Set seeds and device
    3. Build tokenizer, datasets, model, optimizer
    4. Run an implementation smoke test if requested
    5. Train for multiple epochs
    6. Save best and last checkpoints

    Args:
        None.

    Returns:
        None.
    """
    
    cfg = parse_args()
    set_seed(cfg.seed)

    device = choose_device(cfg.device)
    configure_cpu_runtime(device)
    if cfg.use_amp and device.type != "cuda":
        print(
            f"[warn] use_amp/mixed_precision is set but device is {device.type!r}; "
            "AMP autocast and GradScaler only activate on CUDA, so this run is full precision."
        )

    output_dir = Path(cfg.output_dir)
    init_ckpt_path = validate_checkpoint_plan(cfg, output_dir)
    validate_init_checkpoint_compatibility(cfg, init_ckpt_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokenizer = build_tokenizer()
    train_dataset, val_dataset = build_datasets(cfg)
    train_dataset, train_known_target_probe, row_random_probe = build_diagnostic_datasets(
        train_dataset,
        cfg,
    )

    print(f"device: {device}")
    print(f"train examples: {len(train_dataset)}")
    print(f"val examples:   {len(val_dataset)}")
    print(f"vocab size:     {tokenizer.vocab_size}")
    if train_known_target_probe is not None:
        print(f"train_known_target_probe examples: {len(train_known_target_probe)}")
    if row_random_probe is not None:
        print(f"row_random_probe examples:         {len(row_random_probe)}")
    val_overlap = summarize_target_overlap(train_dataset, val_dataset)
    print(
        "[split] known_target_train_vs_val: "
        f"train_targets={val_overlap['train_targets']} "
        f"val_targets={val_overlap['val_targets']} "
        f"overlap={val_overlap['overlap']}"
    )
    if row_random_probe is not None:
        probe_overlap = summarize_target_overlap(train_dataset, row_random_probe)
        print(
            "[split] known_target_train_vs_row_random_probe: "
            f"train_targets={probe_overlap['train_targets']} "
            f"probe_targets={probe_overlap['val_targets']} "
            f"overlap={probe_overlap['overlap']}"
        )
    if cfg.training_stage == "antigen_refine":
        baseline_fit = fit_group_majority_baselines(train_dataset, tokenizer, cfg)
        if baseline_fit:
            print(
                "[compat-baseline-fit] "
                f"fit_records={baseline_fit['fit_records']} "
                f"fit_labeled={baseline_fit['fit_labeled_examples']} "
                f"fit_pos_rate={baseline_fit['positive_rate']:.4f} "
                f"fallback_label={baseline_fit['fallback_label']}"
            )
            baseline_parts = []
            train_baseline = evaluate_group_majority_baselines(train_dataset, tokenizer, cfg, baseline_fit)
            if train_baseline:
                baseline_parts.append(format_baseline_summary(train_baseline, "train"))
            if train_known_target_probe is not None and len(train_known_target_probe) > 0:
                known_target_baseline = evaluate_group_majority_baselines(
                    train_known_target_probe,
                    tokenizer,
                    cfg,
                    baseline_fit,
                )
                if known_target_baseline:
                    baseline_parts.append(format_baseline_summary(known_target_baseline, "known_target_probe"))
            if row_random_probe is not None and len(row_random_probe) > 0:
                row_random_baseline = evaluate_group_majority_baselines(
                    row_random_probe,
                    tokenizer,
                    cfg,
                    baseline_fit,
                )
                if row_random_baseline:
                    baseline_parts.append(format_baseline_summary(row_random_baseline, "row_random_probe"))
            val_baseline = evaluate_group_majority_baselines(val_dataset, tokenizer, cfg, baseline_fit)
            if val_baseline:
                baseline_parts.append(format_baseline_summary(val_baseline, "val"))
            if baseline_parts:
                print("[compat-baseline] " + " ".join(baseline_parts))

    model = build_model(tokenizer, cfg, device)
    optimizer = build_optimizer(model, cfg)
    # The GradScaler and warmup scheduler are owned by main() and reused across
    # epochs (and persisted/restored) so their adaptive state is not reset.
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))
    scheduler = build_lr_scheduler(optimizer, cfg)

    best_val_loss = float("inf")
    start_epoch = 0

    print(f"training_stage: {cfg.training_stage}")

    # Resume from last checkpoint if configured and available.
    last_ckpt_path = output_dir / "last.pt"
    if cfg.resume_from_last and last_ckpt_path.exists():
        checkpoint = load_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        # Restore RNG last so the resumed run reproduces the interrupted run's
        # dropout/sampling stream instead of the fresh set_seed(cfg.seed) stream.
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resuming from epoch {start_epoch}")
    elif init_ckpt_path is not None:
        if is_antigen_stage(cfg.training_stage):
            initialize_antigen_refine_from_checkpoint(
                path=init_ckpt_path,
                model=model,
                map_location=device,
            )
        else:
            load_checkpoint(
                path=init_ckpt_path,
                model=model,
                optimizer=None,
                map_location=device,
                strict=False,
            )
            print("[checkpoint] initialized model weights from init_checkpoint")
        if is_antigen_stage(cfg.training_stage):
            print(f"[checkpoint] initialized {cfg.training_stage} model weights from init_checkpoint")
        if last_ckpt_path.exists() and not cfg.resume_from_last:
            print("[checkpoint] ignored existing last.pt because resume_from_last=False")

    if cfg.smoke_test_only:
        smoke_loader = build_train_loader(train_dataset, tokenizer, cfg, epoch=0, device=device)
        run_smoke_test(
            model, smoke_loader, optimizer, device, cfg.use_amp, cfg.training_stage, cfg.grad_clip_norm
        )
        return

    pretrain_train_probe_metrics = None
    pretrain_row_random_metrics = None
    pretrain_val_metrics = evaluate(
        model=model,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
    )
    if train_known_target_probe is not None and len(train_known_target_probe) > 0:
        pretrain_train_probe_metrics = evaluate(
            model=model,
            val_dataset=train_known_target_probe,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
        )
    if row_random_probe is not None and len(row_random_probe) > 0:
        pretrain_row_random_metrics = evaluate(
            model=model,
            val_dataset=row_random_probe,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
        )

    pretrain_parts = []
    if pretrain_train_probe_metrics is not None:
        pretrain_parts.append(format_metric_summary(pretrain_train_probe_metrics, cfg, "pretrain_known_target"))
    if pretrain_row_random_metrics is not None:
        pretrain_parts.append(format_metric_summary(pretrain_row_random_metrics, cfg, "pretrain_row_random"))
    pretrain_parts.append(format_metric_summary(pretrain_val_metrics, cfg, "pretrain_val"))
    print("[epoch 0/0] " + " ".join(pretrain_parts))
    pretrain_metrics_record: Dict[str, Any] = {
        "epoch": 0,
        "phase": "pretrain_eval",
        "training_stage": cfg.training_stage,
        "pretrain_val": pretrain_val_metrics,
    }
    if pretrain_train_probe_metrics is not None:
        pretrain_metrics_record["pretrain_known_target"] = pretrain_train_probe_metrics
    if pretrain_row_random_metrics is not None:
        pretrain_metrics_record["pretrain_row_random"] = pretrain_row_random_metrics
    append_metrics_jsonl(output_dir, pretrain_metrics_record)

    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )

        train_known_target_metrics = None
        row_random_metrics = None
        if train_known_target_probe is not None and len(train_known_target_probe) > 0:
            train_known_target_metrics = evaluate(
                model=model,
                val_dataset=train_known_target_probe,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
            )
        if row_random_probe is not None and len(row_random_probe) > 0:
            row_random_metrics = evaluate(
                model=model,
                val_dataset=row_random_probe,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
            )

        val_metrics = evaluate(
            model=model,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
        )

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        if is_antigen_stage(cfg.training_stage):
            aux_loss_name = "compatibility_loss"
            aux_acc_name = "compatibility_acc"
        else:
            aux_loss_name = "pair_loss"
            aux_acc_name = "pair_acc"

        summary_parts = [format_metric_summary(train_metrics, cfg, "train")]
        if train_known_target_metrics is not None:
            summary_parts.append(format_metric_summary(train_known_target_metrics, cfg, "known_target_probe"))
        if row_random_metrics is not None:
            summary_parts.append(format_metric_summary(row_random_metrics, cfg, "row_random_probe"))
        summary_parts.append(format_metric_summary(val_metrics, cfg, "val"))
        print(f"[epoch {epoch+1}/{cfg.epochs}] " + " ".join(summary_parts))
        epoch_metrics_record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "phase": "train_eval",
            "training_stage": cfg.training_stage,
            "train": train_metrics,
            "val": val_metrics,
        }
        if train_known_target_metrics is not None:
            epoch_metrics_record["known_target_probe"] = train_known_target_metrics
        if row_random_metrics is not None:
            epoch_metrics_record["row_random_probe"] = row_random_metrics
        append_metrics_jsonl(output_dir, epoch_metrics_record)

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch + 1,
            val_loss=val_loss,
            scaler=scaler,
            scheduler=scheduler,
        )

        # Guard against NaN/Inf: `NaN < best` is always False, so a non-finite
        # val_loss would silently neither win nor warn. Skip and report instead.
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                val_loss=val_loss,
                scaler=scaler,
                scheduler=scheduler,
            )
        elif not math.isfinite(val_loss):
            print(f"[warn] val_loss is non-finite ({val_loss}); best.pt not updated this epoch")


if __name__ == "__main__":
    main()
    

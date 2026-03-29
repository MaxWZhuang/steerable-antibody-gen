#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.tokenizer import AminoAcidTokenizer
from smallAntibodyGen.data.MLMCollator import (
    OASSequenceDataset,
    MLMCollator
)
from smallAntibodyGen.data.MLMSampler import ChainLengthBucketBatchSampler
from smallAntibodyGen.models.mlm import AntibodyMLM, MLMConfig

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    yaml = None


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
    shuffle_pair_probability: float = 0.5

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    pair_loss_weight: float = 1.0
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
        if self.hcdr3_span_min <= 0 or self.hcdr3_span_max <= 0:
            raise ValueError("HCDR3 span lengths must be > 0")
        if self.hcdr3_span_min > self.hcdr3_span_max:
            raise ValueError("hcdr3_span_min must be <= hcdr3_span_max")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.pair_loss_weight < 0:
            raise ValueError("pair_loss_weight must be >= 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.train_num_workers < 0 or self.eval_num_workers < 0:
            raise ValueError("num_workers must be >= 0")


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

    # These keys are intentionally accepted but ignored for now so saved configs
    # from earlier experiments remain runnable instead of failing hard.
    normalized.pop("warmup_steps", None)

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
    parser.add_argument("--shuffle-pair-probability", type=float)

    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--pair-loss-weight", type=float)
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
    if config_path:
        file_config = normalize_config_data(load_config_file(config_path))
        merged_config.update(file_config)

    # CLI values always win over config-file values when both are provided.
    merged_config.update(args_dict)

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
    return train_dataset, val_dataset


def build_train_loader(
    dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    epoch: int = 0,
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

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        mask_probability=cfg.mask_probability,
        hcdr3_span_probability=cfg.hcdr3_span_probability,
        hcdr3_span_min=cfg.hcdr3_span_min,
        hcdr3_span_max=cfg.hcdr3_span_max,
        shuffle_pair_probability=cfg.shuffle_pair_probability,
        rng_seed=cfg.seed + epoch,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.train_num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if cfg.train_num_workers > 0 else None,
    )
    return loader


def build_eval_loader(
    dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
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

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        mask_probability=cfg.mask_probability,
        hcdr3_span_probability=cfg.hcdr3_span_probability,
        hcdr3_span_min=cfg.hcdr3_span_min,
        hcdr3_span_max=cfg.hcdr3_span_max,
        shuffle_pair_probability=cfg.shuffle_pair_probability,
        rng_seed=cfg.seed + 20_000,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.eval_num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if cfg.eval_num_workers > 0 else None,
    )
    return loader


def build_model(
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device,
) -> AntibodyMLM:
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
    model = AntibodyMLM(model_cfg).to(device)
    return model


def build_optimizer(model: AntibodyMLM, cfg: TrainConfig) -> AdamW:
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
    return AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
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
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


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
    model: AntibodyMLM,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    use_amp: bool,
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
        logits, pair_logits = model.forward_with_pairing(batch["input_ids"], batch["attention_mask"])
        losses = model.compute_losses(
            mlm_logits=logits,
            labels=batch["labels"],
            pair_logits=pair_logits,
            pair_labels=batch["pair_labels"],
            pair_mask=batch["pair_mask"],
            pair_loss_weight=1.0,
        )
        loss = losses["loss"]

    print("smoke_test/input_ids:", tuple(batch["input_ids"].shape))
    print("smoke_test/logits:", tuple(logits.shape))
    print("smoke_test/pair_logits:", tuple(pair_logits.shape))
    print("smoke_test/loss:", float(loss))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    print("smoke_test/backward: ok")
    print("smoke_test/optimizer_step: ok")


@torch.no_grad()
def evaluate(
    model: AntibodyMLM,
    val_dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[float, float]:
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
        Tuple `(mean_loss, mean_masked_accuracy)`.
    """
    model.eval()
    val_loader = build_eval_loader(val_dataset, tokenizer, cfg)

    total_loss = 0.0
    total_acc = 0.0
    total_pair_acc = 0.0
    total_batches = 0

    progress = _make_progress_bar(
        val_loader,
        total=len(val_loader),
        desc="eval",
        cfg=cfg,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
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
        acc = masked_accuracy(logits, batch["labels"])
        pair_acc = pair_classification_accuracy(pair_logits, batch["pair_labels"], batch["pair_mask"])

        total_loss += float(loss.item())
        total_acc += acc
        total_pair_acc += pair_acc
        total_batches += 1
        progress.set_postfix(
            loss=f"{total_loss / total_batches:.4f}",
            acc=f"{total_acc / total_batches:.4f}",
        )

    progress.close()

    if total_batches == 0:
        return float("nan"), float("nan")

    return total_loss / total_batches, total_acc / total_batches


def train_one_epoch(
    model: AntibodyMLM,
    train_dataset: OASSequenceDataset,
    tokenizer: AminoAcidTokenizer,
    optimizer: AdamW,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
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
        Tuple `(mean_train_loss, mean_train_masked_accuracy)`.
    """
    model.train()
    train_loader = build_train_loader(train_dataset, tokenizer, cfg, epoch=epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    total_loss = 0.0
    total_acc = 0.0
    total_pair_acc = 0.0
    total_batches = 0

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

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        acc = masked_accuracy(logits.detach(), batch["labels"])
        pair_acc = pair_classification_accuracy(
            pair_logits.detach(),
            batch["pair_labels"],
            batch["pair_mask"],
        )

        total_loss += float(loss.item())
        total_acc += acc
        total_pair_acc += pair_acc
        total_batches += 1
        progress.set_postfix(
            loss=f"{total_loss / total_batches:.4f}",
            acc=f"{total_acc / total_batches:.4f}",
            pair_acc=f"{total_pair_acc / total_batches:.4f}",
        )

    progress.close()

    return total_loss / total_batches, total_acc / total_batches


def save_checkpoint(
    path: Path,
    model: AntibodyMLM,
    optimizer: AdamW,
    cfg: TrainConfig,
    epoch: int,
    val_loss: float,
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
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "train_config": asdict(cfg),
        },
        path
    )
    
def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    Load a checkpoint into the model (and optionally optimizer).

    Args:
        path: Path to checkpoint file.
        model: Model to load into.
        optimizer: Optional optimizer to restore.
        map_location: Device mapping for torch.load.

    Returns:
        The full checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[checkpoint] loaded <- {path}")
    return checkpoint

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

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokenizer = build_tokenizer()
    train_dataset, val_dataset = build_datasets(cfg)

    print(f"device: {device}")
    print(f"train examples: {len(train_dataset)}")
    print(f"val examples:   {len(val_dataset)}")
    print(f"vocab size:     {tokenizer.vocab_size}")

    model = build_model(tokenizer, cfg, device)
    optimizer = build_optimizer(model, cfg)

    if cfg.smoke_test_only:
        smoke_loader = build_train_loader(train_dataset, tokenizer, cfg, epoch=0)
        run_smoke_test(model, smoke_loader, optimizer, device, cfg.use_amp)
        return

    best_val_loss = float("inf")
    start_epoch = 0

    # Resume from last checkpoint if it exists
    last_ckpt_path = output_dir / "last.pt"
    if last_ckpt_path.exists():
        checkpoint = load_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )

        val_loss, val_acc = evaluate(
            model=model,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
        )

        print(
            f"[epoch {epoch+1}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch + 1,
            val_loss=val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                val_loss=val_loss,
            )


if __name__ == "__main__":
    main()
    

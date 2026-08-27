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
import torch.nn.functional as F
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

from smallAntibodyGen import experiment
from smallAntibodyGen.tokenizer import AminoAcidTokenizer
from smallAntibodyGen.data.MLMCollator import (
    AntibodyAntigenCollator,
    AntibodyAntigenRealLabelCollator,
    CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES,
    MLM_IGNORE_INDEX,
    OASSequenceDataset,
    MLMCollator
)
from smallAntibodyGen.data.MLMSampler import ChainLengthBucketBatchSampler
# Re-exported, not re-implemented: the token-layout arithmetic now lives in
# data/lengths.py so scripts/context_length_census.py can share it without
# importing this training script. Callers that reach these through the
# `mlm_train` namespace keep working unchanged.
from smallAntibodyGen.data.lengths import (
    format_length_truncation_warning,
    summarize_length_truncation,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, AntibodyMLM, MLMConfig

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional 'tb' extra not installed
    SummaryWriter = None


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
    # Schedule-covering corruption (masked-diffusion direction). All three
    # defaults reproduce the historical behavior exactly: "fixed" consumes zero
    # extra collator RNG draws (byte-identical batches), "" inherits the train
    # schedule on the eval side, and False emits no new metrics keys -- so every
    # existing run and its metrics.jsonl are byte-for-byte unchanged.
    # "uniform" draws a per-row masking rate t ~ U(0, 1] and IGNORES
    # mask_probability (inert in the full_span HCDR3 mode).
    # eval_mask_rate_schedule exists so arms that differ in TRAIN schedule can
    # share one ARM-INDEPENDENT eval schedule (eval uniform for every arm,
    # binned post-hoc by realized masked fraction).
    mask_rate_schedule: str = "fixed"
    eval_mask_rate_schedule: str = ""
    report_masked_fraction_bins: bool = False
    shuffle_pair_probability: float = 0.5
    shuffle_antigen_probability: float = 0.5

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    # Pre-LN (True) vs post-LN (False) residual arrangement. Defaults to pre-LN,
    # matching every checked-in config and MLMConfig.norm_first; see the note
    # there for why the old False default was a footgun rather than a safeguard.
    # Changing this on an EXISTING chain is a from-scratch retrain, not a
    # migration: post-LN and pre-LN single-stream state dicts are
    # name/shape-identical, so strict loading cannot catch the mismatch. The
    # init-compat check below treats it as an architecture key for that reason,
    # and still reads a missing checkpoint key as post-LN.
    norm_first: bool = True

    # Antigen-stream encoder selection (Direction 1: hybrid PLM antigen encoder).
    # Defaults reproduce the original from-scratch dual-stream model, so setting
    # none of these keys leaves every existing stage byte-for-byte unchanged.
    antigen_encoder_type: str = "scratch"
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    antigen_max_length: int = 512
    antigen_encoder_finetune: str = "frozen"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    warmup_steps: int = 0
    # Opt-in LR decay + regularization + logging knobs. Every default below
    # reproduces the historical behavior exactly: warmup->constant LR, no early
    # stopping, no TensorBoard. So an unmodified config is byte-for-byte unchanged.
    lr_schedule: str = "constant"
    min_lr_ratio: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    tensorboard: bool = False
    checkpoint_every_steps: int = 0
    pair_loss_weight: float = 1.0
    compatibility_loss_weight: float = 1.0
    # Opt-in diagnostic knob. 1.0 reproduces the historical total exactly (a
    # multiply by exactly 1.0 is bit-exact); 0.0 removes MLM from optimization on
    # antigen stages so the compatibility term alone drives training. The
    # reported mlm_loss stays unweighted so curves remain comparable across arms.
    mlm_loss_weight: float = 1.0
    # Plumbed into MLMConfig. "cls" is the historical CLS-concat readout; "mean"
    # mask-aware mean-pools both fused streams before fusion_mlp.
    compat_readout: str = "cls"
    # Multiplies the LR of the modules the antigen-stage warm-start leaves
    # randomly initialized (the set is pinned in NEW_MODULE_LR_PREFIXES and
    # verified by a test that derives it from the real translation function).
    # 1.0 keeps the historical single-LR optimizer with its exact param-group
    # layout, so an unset run is byte-identical.
    new_module_lr_multiplier: float = 1.0
    # "val_compat_loss" keys best.pt selection, early stopping, and the
    # checkpoint's tracked val_loss payload on the compatibility term instead of
    # the MLM-dominated combined loss. Training gradients are unchanged either
    # way -- this is a selection knob, not an objective knob.
    best_checkpoint_metric: str = "val_loss"
    # Graded-affinity supervision. Both default to the historical behavior:
    # weight 0.0 builds NO strength head (no extra init-RNG) and adds a 0-scaled
    # differentiable-zero loss term, so every existing run is byte-for-byte
    # unchanged. `use_strength_head` on the model config is DERIVED from the
    # weight, so there is one source of truth rather than two flags that can
    # disagree.
    strength_loss_weight: float = 0.0
    # Whether the antigen-stage row filter also admits rows that carry a strength
    # quantile but NO binary label. Kept separate from the loss weight on purpose:
    # turning the head on and widening the training population at the same time
    # confounds "the head helps" with "more rows help".
    include_strength_rows: bool = False
    # Which rows contribute ANTIGEN-CONDITIONED MLM targets. Independent of
    # whether a row contributes compatibility or strength supervision, and it
    # never changes the corrupted input. See
    # specs/conditional_denoising_eligibility.md and
    # specs/decisions/0001-conditional-denoising-eligibility.md.
    #
    #   binary_binders_only : binder_label == 1. The Stage-3 default, because a
    #                         measured NONBINDER must not become a positive
    #                         reconstruction target under the antigen it is
    #                         labeled not to bind.
    #   all_filtered_rows   : every row the stage's dataset filter admitted. The
    #                         Stage-4 default, because that stage filters on
    #                         `is_strong_binder`, which is deliberately broader
    #                         than `binder_label`.
    #
    # WARNING: the two policies produce different training populations from
    # otherwise identical-looking configs. Stage-3 MLM loss is NOT comparable
    # across a change to this field -- re-baseline rather than compare.
    #
    # `None` means UNSET and is resolved per stage in `__post_init__`, so every
    # construction path -- CLI, YAML, a direct `TrainConfig(...)` in a test or in
    # `hcdr3_infill.config_from_checkpoint` -- gets the same answer. Resolving
    # this in `parse_args` alone would let a Stage-3 config built anywhere else
    # silently fall back to pre-fix behavior, and the contract is explicit that
    # "a default reaching production through omission is a defect".
    conditional_denoising_eligibility: str | None = None
    # Learned conditional length posterior. 0.0 (default) builds NO length head
    # and adds a 0-scaled differentiable-zero term, so existing runs are
    # byte-identical. `length_head_max` must cover the corpus's HCDR3 lengths;
    # rows longer than it are MASKED OUT of the length loss, never clamped, and
    # scripts/length_census.py is what tells you the right value.
    length_loss_weight: float = 0.0
    length_head_max: int = 32
    epochs: int = 5
    seed: int = 42

    use_amp: bool = False
    smoke_test_only: bool = False
    show_progress: bool = True
    # Promotion gate (J03 step 8). False (the default) is a DEVELOPMENT run: it
    # may run from a dirty worktree, but only because the complete source-content
    # hash and the dirty path list are recorded in the run fingerprint. True marks
    # a PROMOTED canonical run and refuses to start unless git reports a clean
    # worktree -- and refuses just as hard when git cannot verify it at all,
    # because "not known to be dirty" is not "known to be clean".
    #
    # This field is deliberately NOT on
    # `experiment.OPERATIONAL_ONLY_CONFIG_FIELDS`: that list is the owner's exact
    # approved set. Consequence: flipping the promotion gate mid-chain blocks a
    # resume, which fails closed and costs nothing in practice since a canonical
    # run keeps the flag set for its whole life.
    require_clean_worktree: bool = False
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.conditional_denoising_eligibility is None:
            # Stage 3 restricts conditional denoising to measured binders. Every
            # other stage keeps `all_filtered_rows`: Stage 4 filters on
            # `is_strong_binder`, which is deliberately broader than
            # `binder_label`, so `binder_label == 1` would silently drop the
            # large majority of its strong binders.
            self.conditional_denoising_eligibility = (
                "binary_binders_only"
                if self.training_stage == "antigen_real_label_refine"
                else "all_filtered_rows"
            )

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
        if self.hcdr3_mask_mode not in {"sampled_span", "full_span", "partial_span"}:
            raise ValueError(
                "hcdr3_mask_mode must be one of: sampled_span, full_span, partial_span"
            )
        if self.mask_replacement_strategy not in {"bert", "always_mask"}:
            raise ValueError("mask_replacement_strategy must be one of: bert, always_mask")
        if self.mask_rate_schedule not in {"fixed", "uniform"}:
            raise ValueError("mask_rate_schedule must be one of: fixed, uniform")
        if self.eval_mask_rate_schedule not in {"", "fixed", "uniform"}:
            raise ValueError(
                "eval_mask_rate_schedule must be one of: '' (inherit), fixed, uniform"
            )
        if not isinstance(self.report_masked_fraction_bins, bool):
            raise ValueError("report_masked_fraction_bins must be a bool")
        if not isinstance(self.require_clean_worktree, bool):
            raise ValueError("require_clean_worktree must be a bool")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be one of: constant, cosine")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps must be >= 0")
        if self.pair_loss_weight < 0:
            raise ValueError("pair_loss_weight must be >= 0")
        if self.compatibility_loss_weight < 0:
            raise ValueError("compatibility_loss_weight must be >= 0")
        if self.mlm_loss_weight < 0:
            raise ValueError("mlm_loss_weight must be >= 0")
        if self.mlm_loss_weight != 1.0 and not is_antigen_stage(self.training_stage):
            raise ValueError(
                "mlm_loss_weight != 1.0 is only supported for antigen stages; the "
                "antibody-only loss composition does not read it"
            )
        if self.compat_readout not in {"cls", "mean"}:
            raise ValueError("compat_readout must be either 'cls' or 'mean'")
        if self.new_module_lr_multiplier <= 0:
            raise ValueError("new_module_lr_multiplier must be > 0")
        if self.new_module_lr_multiplier != 1.0 and not is_antigen_stage(self.training_stage):
            raise ValueError(
                "new_module_lr_multiplier != 1.0 is only supported for antigen stages; "
                "the antibody-only model has none of the targeted modules"
            )
        if self.best_checkpoint_metric not in {"val_loss", "val_compat_loss"}:
            raise ValueError("best_checkpoint_metric must be one of: val_loss, val_compat_loss")
        if self.strength_loss_weight < 0:
            raise ValueError("strength_loss_weight must be >= 0")
        if self.strength_loss_weight > 0 and not is_antigen_stage(self.training_stage):
            raise ValueError(
                "strength_loss_weight > 0 is only supported for antigen stages; the "
                "antibody-only model has no joint representation to read strength off"
            )
        if not isinstance(self.include_strength_rows, bool):
            raise ValueError("include_strength_rows must be a bool")
        if self.conditional_denoising_eligibility not in CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES:
            raise ValueError(
                "conditional_denoising_eligibility must be one of: "
                + ", ".join(CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES)
            )
        if (
            self.conditional_denoising_eligibility == "binary_binders_only"
            and not is_antigen_stage(self.training_stage)
        ):
            raise ValueError(
                "conditional_denoising_eligibility='binary_binders_only' is only "
                "supported for antigen stages; a stage with no antigen has no "
                "antigen-conditioned denoising to restrict"
            )
        if self.length_loss_weight < 0:
            raise ValueError("length_loss_weight must be >= 0")
        if self.length_loss_weight > 0 and not is_antigen_stage(self.training_stage):
            raise ValueError(
                "length_loss_weight > 0 is only supported for antigen stages; the "
                "length posterior is conditioned on the antigen"
            )
        if self.length_head_max <= 0:
            raise ValueError("length_head_max must be > 0")
        if self.best_checkpoint_metric == "val_compat_loss" and not is_antigen_stage(
            self.training_stage
        ):
            raise ValueError(
                "best_checkpoint_metric='val_compat_loss' is only supported for antigen "
                "stages; other stages report no compatibility loss"
            )
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
        if self.antigen_encoder_type not in {"scratch", "esm"}:
            raise ValueError("antigen_encoder_type must be one of: scratch, esm")
        if self.antigen_encoder_finetune not in {"frozen", "lora"}:
            raise ValueError("antigen_encoder_finetune must be one of: frozen, lora")
        if not (0 < self.antigen_max_length <= 1024):
            raise ValueError("antigen_max_length must be in (0, 1024]")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be > 0")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be > 0")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.antigen_encoder_type == "esm" and not is_antigen_stage(self.training_stage):
            raise ValueError(
                "antigen_encoder_type='esm' only applies to antigen stages "
                "(antigen_refine, antigen_real_label_refine, antigen_hcdr3_infill_refine); "
                f"got training_stage='{self.training_stage}'"
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
        for key in ("d_model", "n_heads", "n_layers", "d_ff", "dropout", "norm_first"):
            if key in model_config:
                normalized.setdefault(key, model_config[key])

    antigen_encoder_config = normalized.pop("antigen_encoder", None)
    if antigen_encoder_config is not None:
        if not isinstance(antigen_encoder_config, dict):
            raise ValueError("The `antigen_encoder` config section must be a mapping")
        # Friendlier nested keys map onto the flat TrainConfig fields. `type` and
        # `finetune` are shortened in the section because the `antigen_encoder:`
        # prefix already scopes them.
        antigen_key_map = {
            "type": "antigen_encoder_type",
            "esm_model_name": "esm_model_name",
            "antigen_max_length": "antigen_max_length",
            "finetune": "antigen_encoder_finetune",
            "lora_r": "lora_r",
            "lora_alpha": "lora_alpha",
            "lora_dropout": "lora_dropout",
        }
        for section_key, flat_key in antigen_key_map.items():
            if section_key in antigen_encoder_config:
                normalized.setdefault(flat_key, antigen_encoder_config[section_key])

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
    parser.add_argument(
        "--hcdr3-mask-mode",
        type=str,
        choices=("sampled_span", "full_span", "partial_span"),
    )
    parser.add_argument("--mask-replacement-strategy", type=str, choices=("bert", "always_mask"))
    parser.add_argument("--mask-rate-schedule", type=str, choices=("fixed", "uniform"))
    parser.add_argument(
        "--eval-mask-rate-schedule", type=str, choices=("", "fixed", "uniform")
    )
    parser.add_argument(
        "--report-masked-fraction-bins", action="store_true", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no-report-masked-fraction-bins",
        dest="report_masked_fraction_bins",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--shuffle-pair-probability", type=float)
    parser.add_argument("--shuffle-antigen-probability", type=float)

    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--norm-first", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-norm-first", dest="norm_first", action="store_false", default=argparse.SUPPRESS)

    parser.add_argument("--antigen-encoder-type", type=str, choices=("scratch", "esm"))
    parser.add_argument("--esm-model-name", type=str)
    parser.add_argument("--antigen-max-length", type=int)
    parser.add_argument("--antigen-encoder-finetune", type=str, choices=("frozen", "lora"))
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--lr-schedule", type=str, choices=("constant", "cosine"))
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--tensorboard", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-every-steps", type=int)
    parser.add_argument("--pair-loss-weight", type=float)
    parser.add_argument("--compatibility-loss-weight", type=float)
    parser.add_argument("--mlm-loss-weight", type=float)
    parser.add_argument("--compat-readout", type=str, choices=("cls", "mean"))
    parser.add_argument("--new-module-lr-multiplier", type=float)
    parser.add_argument(
        "--best-checkpoint-metric", type=str, choices=("val_loss", "val_compat_loss")
    )
    parser.add_argument("--strength-loss-weight", type=float)
    parser.add_argument("--length-loss-weight", type=float)
    parser.add_argument("--length-head-max", type=int)
    parser.add_argument(
        "--include-strength-rows",
        dest="include_strength_rows",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-include-strength-rows",
        dest="include_strength_rows",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--conditional-denoising-eligibility",
        type=str,
        choices=CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES,
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--use-amp", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--smoke-test-only", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--show-progress", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument(
        "--require-clean-worktree", action="store_true", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no-require-clean-worktree",
        dest="require_clean_worktree",
        action="store_false",
        default=argparse.SUPPRESS,
    )
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

    ``torch.set_num_interop_threads`` is settable only ONCE per process, and only
    before any parallel work has started. That makes it fatal in exactly the
    situations where a thread hint matters least:

    - a notebook re-running the training cell (the second run dies before it
      builds anything),
    - a driver script that chains two stages in one process,
    - any interactive session that touched a torch op before training.

    So the call is best-effort. The failure means the interop pool is already
    sized -- which is the state we wanted anyway -- so aborting a training run
    over it would trade a whole run for a hint.

    Args:
        device:
            The chosen training device.

    Returns:
        None.
    """
    if device.type == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        # Repeatable; safe to call unconditionally.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already set, or parallel work has begun. Both are benign here.
            pass


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

    Positives are gated on ``is_strong_binder`` rather than ``binder_label == 1``
    so the stage uses the full observed-binder population -- explicit boolean
    positives plus the KD / -log KD / fuzzy strong binders that ``binder_label``
    (set only for ``affinity_type == "bool"`` rows) never covers. Restricting to
    ``binder_label == 1`` would silently drop the large majority of strong
    binders. ``EmpiricalHCDR3LengthPrior.fit`` is gated on the same flag so the
    length prior describes exactly the population the infiller trains on.
    """
    heavy_sequence = (getattr(record, "sequence_heavy", None) or getattr(record, "sequence", "") or "").strip()
    antigen_sequence = (getattr(record, "sequence_antigen", None) or "").strip()
    return (
        bool(getattr(record, "is_strong_binder", False))
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
        # `include_strength_rows` widens this filter to admit rows that carry a
        # graded strength quantile but NO binary label -- rows the binary head
        # cannot use and the strength head can. It is a SEPARATE knob from
        # `strength_loss_weight` on purpose: flipping both at once confounds
        # "the graded head helps" with "the larger population helps".
        def _eligible(record) -> bool:
            if record.binder_label in (0, 1):
                return True
            if not cfg.include_strength_rows:
                return False
            quantile = record.affinity_strength_quantile
            return isinstance(quantile, (int, float)) and not isinstance(quantile, bool)

        train_dataset.records = [r for r in train_dataset.records if _eligible(r)]
        val_dataset.records = [r for r in val_dataset.records if _eligible(r)]
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

    The probe rows are held out and REMOVED from training, so the size is capped
    at 20% of the training set. The old ``total_records - 1`` bound could hand the
    probe almost the entire dataset (e.g. 179 of 180 rows), silently leaving a
    single training record. Datasets under 5 records skip probing entirely. For
    real config-scale datasets this is byte-identical to the old cap, since
    ``total // 5`` far exceeds ``min(target, 4096)`` there.
    """
    if total_records < 5:
        return 0
    target = max(eval_batch_size * 32, 1024)
    return min(total_records // 5, target, 4096)


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
    Summarize target overlap between two datasets.

    Grouped on `canonical_target_id`, not on the legacy `target_key`. `target_key`
    picks the first of four mutually exclusive identifier branches with no
    reconciliation, so one biological antigen seen once with a UniProt accession
    and once with only a PDB code produces two different keys and draws two
    independent split assignments. Grouping the leakage report on that field
    would UNDER-REPORT exactly the alias overlap the report exists to catch.

    Falls back to `target_key` so corpora written before J02 stay readable; those
    corpora simply have no merges to recover and must be regenerated to get them.
    """

    def target_of(record):
        return getattr(record, "canonical_target_id", None) or record.target_key

    train_targets = {target_of(r) for r in train_dataset.records if target_of(r)}
    val_targets = {target_of(r) for r in val_dataset.records if target_of(r)}
    return {
        "train_targets": len(train_targets),
        "val_targets": len(val_targets),
        "overlap": len(train_targets & val_targets),
    }


def summarize_conditional_denoising_eligibility(
    dataset,
    conditional_denoising_eligibility: str,
) -> dict[str, int]:
    """
    Count how many rows of one split contribute antigen-conditioned MLM targets.

    Mirrors `AntibodyAntigenCollator._is_conditional_denoising_eligible` at the
    dataset level, before any model is built. The two must agree; a divergence
    would mean preflight blesses a population the collator then discards.

    Returns `total` and `eligible`. A zero-eligible split is reported by the
    caller, which owns the decision to fail -- an all-nonbinder *batch* is
    legitimate under `binary_binders_only`, but an all-nonbinder *corpus* means
    the stage would train its conditional policy on nothing while reporting a
    plausible loss curve, because MLM loss over an all-ignored batch is a finite
    differentiable zero rather than NaN.
    """
    total = 0
    eligible = 0
    for record in dataset.records:
        total += 1
        if conditional_denoising_eligibility == "binary_binders_only":
            if getattr(record, "binder_label", None) == 1:
                eligible += 1
        else:
            eligible += 1
    return {"total": total, "eligible": eligible}


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
        # Grouped on BOTH identities on purpose. `target_keys` is the legacy
        # first-available-identifier key, which splits one biological target
        # across several groups whenever it appears under different accessions.
        # That makes the target-family shortcut baseline ARTIFICIALLY WEAK --
        # fewer labeled rows per group, more fallback to the global majority --
        # so a model beating it looks better than it is. Gate 2A requires beating
        # the target-family baseline, so it must read the CANONICAL one.
        # The gap between the two accuracies measures how much aliasing was
        # inflating the old number.
        "canonical_target_ids": defaultdict(Counter),
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
        "canonical_target_ids": 0,
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
        # The Gate-2A target-family baseline. Read this one, not the legacy key.
        "canonical_target_majority_acc": (
            group_correct["canonical_target_ids"] / labeled_examples
        ),
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
        f"{prefix}_canonical_target_majority_acc="
        f"{metrics['canonical_target_majority_acc']:.4f} "
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
            antigen_encoder_type=cfg.antigen_encoder_type,
            esm_model_name=cfg.esm_model_name,
            antigen_max_length=cfg.antigen_max_length,
            rng_seed=cfg.seed + epoch,
            mask_rate_schedule=cfg.mask_rate_schedule,
            build_length_query=cfg.length_loss_weight > 0,
            length_head_max=cfg.length_head_max,
            conditional_denoising_eligibility=cfg.conditional_denoising_eligibility,
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
            antigen_encoder_type=cfg.antigen_encoder_type,
            esm_model_name=cfg.esm_model_name,
            antigen_max_length=cfg.antigen_max_length,
            rng_seed=cfg.seed + epoch,
            mask_rate_schedule=cfg.mask_rate_schedule,
            build_length_query=cfg.length_loss_weight > 0,
            length_head_max=cfg.length_head_max,
            conditional_denoising_eligibility=cfg.conditional_denoising_eligibility,
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
            mask_rate_schedule=cfg.mask_rate_schedule,
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
            antigen_encoder_type=cfg.antigen_encoder_type,
            esm_model_name=cfg.esm_model_name,
            antigen_max_length=cfg.antigen_max_length,
            rng_seed=cfg.seed + 20_000,
            mask_rate_schedule=(cfg.eval_mask_rate_schedule or cfg.mask_rate_schedule),
            build_length_query=cfg.length_loss_weight > 0,
            length_head_max=cfg.length_head_max,
            conditional_denoising_eligibility=cfg.conditional_denoising_eligibility,
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
            antigen_encoder_type=cfg.antigen_encoder_type,
            esm_model_name=cfg.esm_model_name,
            antigen_max_length=cfg.antigen_max_length,
            rng_seed=cfg.seed + 20_000,
            mask_rate_schedule=(cfg.eval_mask_rate_schedule or cfg.mask_rate_schedule),
            build_length_query=cfg.length_loss_weight > 0,
            length_head_max=cfg.length_head_max,
            conditional_denoising_eligibility=cfg.conditional_denoising_eligibility,
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
            mask_rate_schedule=(cfg.eval_mask_rate_schedule or cfg.mask_rate_schedule),
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


def model_class_for_stage(training_stage: str) -> str:
    """
    The model class a stage instantiates, by name.

    Two different architectures are built from one `TrainConfig`, so the class
    is part of a run's architecture identity and is fingerprinted alongside
    `MLMConfig`.
    """
    return (
        "AntibodyAntigenCrossAttention"
        if is_antigen_stage(training_stage)
        else "AntibodyMLM"
    )


def build_model_config(tokenizer: AminoAcidTokenizer, cfg: TrainConfig) -> MLMConfig:
    """
    Build the `MLMConfig` a run's model is constructed from.

    Split out of `build_model` because `MLMConfig` is the ONLY complete
    description of the architecture and was previously created inline and
    discarded. Four of its fields -- `activation`, `tie_weights`,
    `initializer_range`, `scale_residual_init` -- are hardcoded here and
    unreachable from `TrainConfig`, and `vocab_size`/`pad_token_id` come from
    the tokenizer. `asdict(cfg)` therefore cannot stand in for it when
    fingerprinting the architecture (J03).
    """
    return MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=cfg.max_length,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
        # Carried onto the model config so the antigen stream can branch on the
        # encoder choice in Stage A. Defaults ("scratch") keep today's model.
        antigen_encoder_type=cfg.antigen_encoder_type,
        esm_model_name=cfg.esm_model_name,
        antigen_max_length=cfg.antigen_max_length,
        antigen_encoder_finetune=cfg.antigen_encoder_finetune,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        compat_readout=cfg.compat_readout,
        # Single source of truth: the head exists iff its loss is weighted.
        use_strength_head=cfg.strength_loss_weight > 0,
        # Same single-source-of-truth contract as the strength head.
        use_length_head=cfg.length_loss_weight > 0,
        length_head_max=cfg.length_head_max,
    )


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
    model_cfg = build_model_config(tokenizer, cfg)
    if is_antigen_stage(cfg.training_stage):
        model = AntibodyAntigenCrossAttention(model_cfg).to(device)
    else:
        model = AntibodyMLM(model_cfg).to(device)
    return model


# The exact modules `build_antigen_refine_init_state_dict` leaves randomly
# initialized when warm-starting an antigen stage from an antibody-only or
# paired checkpoint. Derived empirically (not by eye) and pinned by
# `test_new_module_lr_prefixes_match_the_warm_start_missing_keys`, which
# recomputes the set from the real translation function in both norm modes.
#
# `antigen_encoder.` is deliberately absent: on the scratch path it is warm-started
# from the antibody encoder, and on the ESM path its backbone is pretrained. A
# fresh-init projection inside the ESM adapter is a separate concern.
NEW_MODULE_LR_PREFIXES = (
    "antibody_to_antigen.",
    "antigen_to_antibody.",
    "fusion_norm_antibody.",
    "fusion_norm_antigen.",
    "fusion_out_norm_antibody.",  # pre-LN only; absent in post-LN models
    "fusion_out_norm_antigen.",
    "fusion_mlp.",
    "compatibility_head.",
)


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
    if cfg.new_module_lr_multiplier == 1.0:
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        return AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
        )

    # multiplier != 1.0: give the randomly-initialized fusion/head modules their
    # own LR so a warm-started trunk is not dragged by a head still finding its
    # scale. Group ORDER is load-bearing for the optimizer state_dict layout; a
    # resume across a changed multiplier fails loudly on group-count mismatch,
    # which is the correct outcome (a changed config gets a fresh output_dir).
    base_decay: list = []
    base_no_decay: list = []
    new_decay: list = []
    new_no_decay: list = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_new = name.startswith(NEW_MODULE_LR_PREFIXES)
        decay = p.dim() >= 2
        if is_new:
            (new_decay if decay else new_no_decay).append(p)
        else:
            (base_decay if decay else base_no_decay).append(p)
    if not (new_decay or new_no_decay):
        raise ValueError(
            "new_module_lr_multiplier is set but the model has none of the targeted "
            f"modules ({', '.join(NEW_MODULE_LR_PREFIXES)})"
        )
    new_lr = cfg.learning_rate * cfg.new_module_lr_multiplier
    return AdamW(
        [
            {"params": base_decay, "weight_decay": cfg.weight_decay, "lr": cfg.learning_rate},
            {"params": base_no_decay, "weight_decay": 0.0, "lr": cfg.learning_rate},
            {"params": new_decay, "weight_decay": cfg.weight_decay, "lr": new_lr},
            {"params": new_no_decay, "weight_decay": 0.0, "lr": new_lr},
        ],
        lr=cfg.learning_rate,
    )


def build_lr_scheduler(
    optimizer: AdamW,
    cfg: TrainConfig,
    total_steps: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """
    Build the LR schedule: linear warmup, then a constant plateau or cosine decay.

    For optimizer step ``s`` (0-based) the LR multiplier is:

    - ``(s + 1) / warmup_steps`` while ``s < warmup_steps`` (linear warmup), then
    - ``1.0`` for ``lr_schedule == "constant"`` (historical behavior), or
    - a half-cosine from ``1.0`` down to ``min_lr_ratio`` over the remaining
      ``total_steps - warmup_steps`` updates for ``lr_schedule == "cosine"``.

    Returns ``None`` only for the original no-op case (``warmup_steps == 0`` and
    ``lr_schedule == "constant"``), so that path is byte-for-byte unchanged.
    Cosine decay needs a horizon: when ``total_steps`` is missing or does not
    extend past warmup, the schedule falls back to warmup-then-constant rather
    than dividing by zero.

    Args:
        optimizer:
            The optimizer whose LR is scheduled.
        cfg:
            Training configuration (reads ``warmup_steps``, ``lr_schedule``,
            ``min_lr_ratio``).
        total_steps:
            Total planned optimizer steps (``steps_per_epoch * epochs``). Only
            consulted for cosine decay; ignored for the constant schedule.

    Returns:
        A ``LambdaLR`` implementing the schedule, or ``None`` for the constant,
        no-warmup case.
    """
    warmup = cfg.warmup_steps
    use_cosine = cfg.lr_schedule == "cosine"

    if not use_cosine and warmup <= 0:
        return None

    # Cosine needs room to decay past warmup; without a valid horizon we degrade
    # gracefully to warmup-then-constant instead of dividing by zero.
    if use_cosine and (total_steps is None or total_steps <= warmup):
        use_cosine = False

    min_ratio = cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if not use_cosine:
            return 1.0
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        progress = min(1.0, max(0.0, progress))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def early_stopping_decision(
    val_loss: float,
    best_val_loss: float,
    epochs_without_improvement: int,
    patience: int,
    min_delta: float,
) -> tuple[int, bool]:
    """
    Decide whether training should stop early after an epoch's validation.

    ``best_val_loss`` is the best loss observed *before* this epoch. An epoch
    counts as an improvement only when ``val_loss`` is finite and beats that best
    by more than ``min_delta``. Patience counts consecutive non-improving epochs;
    a non-finite ``val_loss`` never counts as improvement.

    Args:
        val_loss:
            This epoch's validation loss.
        best_val_loss:
            Best validation loss before this epoch (``inf`` if none yet).
        epochs_without_improvement:
            Running count of consecutive non-improving epochs.
        patience:
            Max non-improving epochs to tolerate; ``<= 0`` disables early stop.
        min_delta:
            Minimum loss decrease that counts as an improvement.

    Returns:
        Tuple ``(new_epochs_without_improvement, should_stop)``. When
        ``patience <= 0`` this is always ``(0, False)``.
    """
    if patience <= 0:
        return 0, False
    improved = math.isfinite(val_loss) and val_loss < best_val_loss - min_delta
    if improved:
        return 0, False
    new_count = epochs_without_improvement + 1
    return new_count, new_count >= patience


def build_tensorboard_writer(cfg: TrainConfig, output_dir: Path):
    """
    Build a TensorBoard ``SummaryWriter`` when ``cfg.tensorboard`` is set.

    TensorBoard is an optional dependency (the ``tb`` extra). When requested but
    unavailable, we warn and return ``None`` so training still runs — mirroring
    the module's optional-``yaml``/``tqdm`` handling.

    Args:
        cfg:
            Training configuration.
        output_dir:
            Run output directory; logs are written under ``output_dir / "tb"``.

    Returns:
        A ``SummaryWriter`` writing to ``output_dir / "tb"``, or ``None`` when
        TensorBoard logging is disabled or the package is missing.
    """
    if not cfg.tensorboard:
        return None
    if SummaryWriter is None:
        print(
            "[warn] tensorboard=True but the 'tensorboard' package is not installed; "
            "skipping TensorBoard logging. Install it with `pip install -e \".[tb]\"`."
        )
        return None
    return SummaryWriter(log_dir=str(Path(output_dir) / "tb"))


def log_epoch_scalars(
    writer,
    step: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    learning_rate: float,
) -> None:
    """
    Write one epoch's scalar metrics to TensorBoard, if a writer is present.

    A ``None`` writer is a no-op so callers need not branch. Only a small, stable
    set of scalars is logged (train/val loss, val MLM accuracy, LR); the full
    metric record still lands in ``metrics.jsonl``.
    """
    if writer is None:
        return
    writer.add_scalar("loss/train", train_metrics["loss"], step)
    writer.add_scalar("loss/val", val_metrics["loss"], step)
    if "mlm_acc" in val_metrics:
        writer.add_scalar("mlm_acc/val", val_metrics["mlm_acc"], step)
    writer.add_scalar("lr", learning_rate, step)


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


# Masked-fraction bin edges for the corruption-coverage curve.
# Right-open bins except the last: [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1.0].
MASKED_FRACTION_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def masked_fraction_bin_key(lo: float, hi: float) -> str:
    """Metric-key suffix for one bin, e.g. ``(0.2, 0.4) -> 'frac_20_40'``."""
    return f"frac_{int(round(lo * 100))}_{int(round(hi * 100))}"


def masked_fraction_bin_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    special_ids: set[int],
) -> dict[str, list[int]]:
    """
    Per-bin ``[correct, total]`` masked-token counts, binned by each row's
    REALIZED masked fraction (the corruption-coverage curve; opt-in via
    ``report_masked_fraction_bins`` -- never computed otherwise).

    A row's masked fraction is ``#targets / #eligible`` where targets are the
    non-``-100`` label positions and eligible is the UNION of target positions
    and non-special input positions. The union is what makes the denominator
    corruption-invariant: under BERT corruption a target may hold ``[MASK]``
    (special) or a kept/replaced residue (non-special), and post-corruption
    ``input_ids`` alone cannot reconstruct the eligible count. Rows with zero
    eligible or zero target positions are skipped.
    """
    preds = logits.argmax(dim=-1)
    target_mask = labels != -100
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special |= input_ids == sid
    eligible_mask = target_mask | ~special

    counts: dict[str, list[int]] = {}
    for row in range(labels.shape[0]):
        n_targets = int(target_mask[row].sum().item())
        n_eligible = int(eligible_mask[row].sum().item())
        if n_eligible == 0 or n_targets == 0:
            continue
        # Bin index in EXACT integer arithmetic. The float form
        # `int((n_targets / n_eligible) / 0.2)` misbins frac == 3/5, because
        # 0.6 / 0.2 == 2.9999999999999996 -- every eligible-20 row with 12
        # targets landed one bin too low. `n_targets * 5 // n_eligible`
        # implements the documented right-open edges exactly, with the last bin
        # right-closed (frac == 1.0 -> 5 -> clamped to 4).
        bin_idx = min(n_targets * 5 // n_eligible, len(MASKED_FRACTION_BIN_EDGES) - 2)
        lo = MASKED_FRACTION_BIN_EDGES[bin_idx]
        hi = MASKED_FRACTION_BIN_EDGES[bin_idx + 1]
        key = masked_fraction_bin_key(lo, hi)
        row_correct = int(
            (preds[row][target_mask[row]] == labels[row][target_mask[row]]).sum().item()
        )
        entry = counts.setdefault(key, [0, 0])
        entry[0] += row_correct
        entry[1] += n_targets
    return counts


def finalize_masked_fraction_bins(counts: dict[str, list[int]]) -> dict[str, float]:
    """
    Turn accumulated per-bin counts into metric entries.

    EVERY bin is emitted (empty ones as NaN accuracy / 0 tokens) so arm-vs-arm
    comparisons never silently drop a key.
    """
    metrics: dict[str, float] = {}
    for lo, hi in zip(MASKED_FRACTION_BIN_EDGES[:-1], MASKED_FRACTION_BIN_EDGES[1:]):
        bin_key = masked_fraction_bin_key(lo, hi)
        bin_correct, bin_total = counts.get(bin_key, [0, 0])
        metrics[f"mlm_acc_{bin_key}"] = (
            (bin_correct / bin_total) if bin_total > 0 else float("nan")
        )
        metrics[f"mlm_tokens_{bin_key}"] = float(bin_total)
    return metrics


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


def tie_aware_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Spearman rank correlation with MID-RANKS for ties.

    Ties are the normal case here, not an edge case: a corpus with many rows at
    the same reported affinity produces long runs of equal values, and ordinal
    (competition) ranking would silently order them by list position -- turning
    an arbitrary input order into apparent signal. Mid-ranks make tied values
    genuinely interchangeable.

    Returns ``nan`` for fewer than two pairs or when either side is constant
    (an undefined correlation must not be reported as 0.0, which reads as
    "measured, no relationship").
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 2:
        return float("nan")

    def mid_ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx = mid_ranks(x)
    ry = mid_ranks(y)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


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

    Tied scores are collapsed into a single threshold group before precision and
    recall are read off, matching ``sklearn.average_precision_score`` and the
    area-under-PR definition. A naive "precision at each positive's row" sum is
    ambiguous under ties: for one positive and one negative sharing a score, it
    returns 1.0 or 0.5 purely from input order, whereas the threshold-grouped
    value is 0.5 either way. Exact ties are realistic here because a confident
    2-class head saturates ``softmax(...)[:, 1]`` to identical values across
    rows. For strictly distinct scores this reduces to the previous definition.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    pos_count = int((y == 1).sum())
    if pos_count == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_labels = y[order]
    tp_cum = np.cumsum(sorted_labels == 1)
    fp_cum = np.cumsum(sorted_labels == 0)

    # Keep only the last row of each tied-score run: these are the distinct
    # decision thresholds. tp/fp are read at the end of each run so a positive
    # and a negative at the same score share one precision.
    is_threshold = np.ones(len(sorted_scores), dtype=bool)
    is_threshold[:-1] = sorted_scores[1:] != sorted_scores[:-1]
    idx = np.nonzero(is_threshold)[0]

    tp_at = tp_cum[idx].astype(np.float64)
    fp_at = fp_cum[idx].astype(np.float64)
    precision = tp_at / (tp_at + fp_at)
    recall = tp_at / pos_count
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


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
    conditional_denoising_eligibility: str = "all_filtered_rows",
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
            # Under `binary_binders_only` an all-nonbinder first batch produces a
            # differentiable ZERO MLM loss and no MLM gradient, while the lines
            # below still print "backward: ok". Print the census so the smoke test
            # cannot silently stop proving the MLM path.
            eligible = batch.get("conditional_denoising_eligible")
            eligible_rows = "n/a" if eligible is None else int(eligible.sum())
            policy_rows = "n/a" if eligible is None else int(eligible.numel())
            print(
                "smoke_test/conditional_denoising:"
                f" policy={conditional_denoising_eligibility}"
                f" eligible_rows={eligible_rows}/{policy_rows}"
                f" mlm_target_tokens={int((batch['antibody_labels'] != MLM_IGNORE_INDEX).sum())}"
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
    # 1.0 on every non-antigen stage (validate() forbids anything else there), so
    # the reported/selection loss is unchanged unless the knob is set.
    _mlm_weight = cfg.mlm_loss_weight if is_antigen_stage(cfg.training_stage) else 1.0
    compatibility_labels_all: list[int] = []
    compatibility_scores_all: list[float] = []
    compatibility_preds_all: list[int] = []
    # Eligibility census, reported for the SAME reason as on the train side: under
    # `binary_binders_only` validation MLM loss measures a different population
    # than it did before, and nothing else in the val record would say so.
    # Reported, never raised -- an all-nonbinder split is a legitimate probe.
    conditional_eligible_rows: torch.Tensor | int = 0
    conditional_total_rows = 0
    conditional_eligible_tokens: torch.Tensor | int = 0
    hcdr3_counts = {
        "hcdr3_correct_tokens": 0,
        "hcdr3_target_tokens": 0,
        "hcdr3_exact_matches": 0,
        "hcdr3_valid_spans": 0,
    }
    # Corruption-coverage curve (opt-in; default False emits nothing, so
    # metrics.jsonl stays byte-identical for existing runs).
    want_fraction_bins = cfg.report_masked_fraction_bins
    fraction_bin_counts: dict[str, list[int]] = {}
    # Graded-strength supervision (opt-in). Off by default, in which case the
    # forward keeps its historical 2-tuple shape and no metric key is emitted.
    want_strength = cfg.strength_loss_weight > 0 and is_antigen_stage(cfg.training_stage)
    strength_pred_all: list[float] = []
    strength_target_all: list[float] = []
    # Learned length posterior (opt-in). The length query is a SEPARATE forward
    # on the collapsed-span encoding, so it runs only when the loss is active.
    want_length = cfg.length_loss_weight > 0 and is_antigen_stage(cfg.training_stage)
    length_correct = 0
    length_total = 0
    length_nll_sum = 0.0

    progress = _make_progress_bar(
        val_loader,
        total=len(val_loader),
        desc="eval",
        cfg=cfg,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        if is_antigen_stage(cfg.training_stage):
            # The 3-tuple forward is taken ONLY when the head exists, so the
            # default path keeps the exact historical 2-tuple call.
            if want_strength:
                logits, compatibility_logits, strength_predictions = model(
                    antibody_input_ids=batch["antibody_input_ids"],
                    antibody_attention_mask=batch["antibody_attention_mask"],
                    antigen_input_ids=batch["antigen_input_ids"],
                    antigen_attention_mask=batch["antigen_attention_mask"],
                    return_strength=True,
                )
            else:
                strength_predictions = None
                logits, compatibility_logits = model(
                    antibody_input_ids=batch["antibody_input_ids"],
                    antibody_attention_mask=batch["antibody_attention_mask"],
                    antigen_input_ids=batch["antigen_input_ids"],
                    antigen_attention_mask=batch["antigen_attention_mask"],
                )
            length_logits = None
            if want_length:
                # Separate forward: the length query uses the COLLAPSED-span antibody
                # stream, which is different tensors from the MLM forward above.
                length_logits = model.forward_length_query(
                    antibody_input_ids=batch["length_query_input_ids"],
                    antibody_attention_mask=batch["length_query_attention_mask"],
                    antigen_input_ids=batch["antigen_input_ids"],
                    antigen_attention_mask=batch["antigen_attention_mask"],
                )
            eligible_rows = batch.get("conditional_denoising_eligible")
            if eligible_rows is not None:
                conditional_eligible_rows += eligible_rows.sum()
                conditional_total_rows += eligible_rows.numel()
            conditional_eligible_tokens += (
                batch["antibody_labels"] != MLM_IGNORE_INDEX
            ).sum()
            losses = model.compute_losses(
                mlm_logits=logits,
                labels=batch["antibody_labels"],
                compatibility_logits=compatibility_logits,
                compatibility_labels=batch["compatibility_labels"],
                compatibility_mask=batch["compatibility_mask"],
                strength_predictions=strength_predictions if want_strength else None,
                strength_targets=batch.get("strength_targets") if want_strength else None,
                strength_mask=batch.get("strength_mask") if want_strength else None,
                length_logits=length_logits,
                length_labels=batch["length_labels"] if want_length else None,
                length_mask=batch["length_label_mask"] if want_length else None,
                mlm_loss_weight=cfg.mlm_loss_weight,
                compatibility_loss_weight=cfg.compatibility_loss_weight,
                strength_loss_weight=cfg.strength_loss_weight,
                length_loss_weight=cfg.length_loss_weight,
            )
            if want_length and length_logits is not None:
                _lm = batch["length_label_mask"]
                if bool(_lm.any()):
                    _labels = batch["length_labels"][_lm]
                    _lq = length_logits[_lm]
                    length_correct += int((_lq.argmax(dim=-1) == _labels).sum().item())
                    length_total += int(_lm.sum().item())
                    length_nll_sum += float(
                        F.cross_entropy(_lq, _labels, reduction="sum").item()
                    )
            if want_strength and strength_predictions is not None:
                _sm = batch["strength_mask"]
                if bool(_sm.any()):
                    strength_pred_all.extend(
                        strength_predictions.detach()[_sm].float().cpu().tolist()
                    )
                    strength_target_all.extend(
                        batch["strength_targets"][_sm].float().cpu().tolist()
                    )
            # NOTE: losses["loss"] (the optimized total, which DOES include the
            # weighted strength/length terms) is intentionally not used here. The
            # reported/selection loss is rebuilt below from the token- and
            # row-pooled MLM and aux terms so it stays comparable across batches
            # of unequal supervision counts -- which is also why it omits the
            # strength/length terms. See select_checkpoint_metric_value.
            mlm_loss = losses["mlm_loss"]
            aux_loss = losses["compatibility_loss"]
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits, batch["antibody_labels"])
            if want_fraction_bins:
                for _key, (_c, _t) in masked_fraction_bin_counts(
                    logits, batch["antibody_labels"], batch["antibody_input_ids"],
                    tokenizer.special_ids,
                ).items():
                    _entry = fraction_bin_counts.setdefault(_key, [0, 0])
                    _entry[0] += _c
                    _entry[1] += _t
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
            mlm_loss = losses["mlm_loss"]
            aux_loss = losses["pair_loss"]
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits, batch["labels"])
            if want_fraction_bins:
                for _key, (_c, _t) in masked_fraction_bin_counts(
                    logits, batch["labels"], batch["input_ids"],
                    tokenizer.special_ids,
                ).items():
                    _entry = fraction_bin_counts.setdefault(_key, [0, 0])
                    _entry[0] += _c
                    _entry[1] += _t
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
            loss=f"{_mlm_weight * running_mlm_loss + _aux_weight * running_aux_loss:.4f}",
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
            # Emitted even on an empty epoch so the metrics.jsonl key set does not
            # vary between epochs.
            metrics["conditional_denoising_policy_rows"] = 0.0
            metrics["conditional_denoising_eligible_rows"] = 0.0
            metrics["conditional_denoising_eligible_tokens"] = 0.0
        else:
            metrics["pair_loss"] = float("nan")
            metrics["pair_acc"] = float("nan")
        metrics.update(finalize_hcdr3_metrics(hcdr3_counts))
        return metrics

    mlm_loss = (total_mlm_loss_weighted / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    mlm_acc = (total_mlm_correct / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    aux_loss = (total_aux_loss_weighted / total_aux_labeled) if total_aux_labeled > 0 else 0.0
    metrics = {
        "loss": _mlm_weight * mlm_loss + _aux_weight * aux_loss,
        "mlm_loss": mlm_loss,
        "mlm_acc": mlm_acc,
    }
    if want_fraction_bins:
        metrics.update(finalize_masked_fraction_bins(fraction_bin_counts))
    if want_length:
        # Accuracy AND NLL: a categorical length head can look accurate while
        # being badly calibrated, and the proposal path SAMPLES from the posterior
        # rather than taking its argmax.
        metrics["length_eligible_rows"] = float(length_total)
        metrics["length_acc"] = (
            (length_correct / length_total) if length_total > 0 else float("nan")
        )
        metrics["length_nll"] = (
            (length_nll_sum / length_total) if length_total > 0 else float("nan")
        )
    if is_antigen_stage(cfg.training_stage):
        conditional_eligible_rows = int(conditional_eligible_rows)
        conditional_eligible_tokens = int(conditional_eligible_tokens)
        metrics["conditional_denoising_policy_rows"] = float(conditional_total_rows)
        metrics["conditional_denoising_eligible_rows"] = float(conditional_eligible_rows)
        metrics["conditional_denoising_eligible_tokens"] = float(conditional_eligible_tokens)
    if want_strength:
        # Reported next to compatibility_auroc so the graded head is judged on
        # ORDERING (what a ranking objective is for), not on MSE, whose scale is
        # not comparable across quantile populations. NaN with <2 eligible rows.
        metrics["strength_eligible_rows"] = float(len(strength_target_all))
        metrics["val_strength_spearman"] = float(
            tie_aware_spearman(strength_pred_all, strength_target_all)
        )
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
    output_dir: Optional[Path] = None,
    best_val_loss: float = float("inf"),
    run_fingerprint: Optional[Dict[str, Any]] = None,
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
        output_dir:
            Run output directory. When provided together with
            ``cfg.checkpoint_every_steps > 0``, ``last.pt`` is rewritten every N
            batches for crash recovery. ``None`` (the default) disables
            intra-epoch checkpointing, leaving the historical behavior unchanged.
        best_val_loss:
            Best validation loss observed so far. Written into any intra-epoch
            ``last.pt`` so that a resume restores best-tracking correctly instead
            of resetting it to this in-progress epoch's (unknown) loss.

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
    # 1.0 on every non-antigen stage (validate() forbids anything else there), so
    # the reported/selection loss is unchanged unless the knob is set.
    _mlm_weight = cfg.mlm_loss_weight if is_antigen_stage(cfg.training_stage) else 1.0
    compatibility_labels_all: list[int] = []
    compatibility_scores_all: list[float] = []
    compatibility_preds_all: list[int] = []
    hcdr3_counts = {
        "hcdr3_correct_tokens": 0,
        "hcdr3_target_tokens": 0,
        "hcdr3_exact_matches": 0,
        "hcdr3_valid_spans": 0,
    }
    # Corruption-coverage curve (opt-in; default False emits nothing, so
    # metrics.jsonl stays byte-identical for existing runs).
    want_fraction_bins = cfg.report_masked_fraction_bins
    fraction_bin_counts: dict[str, list[int]] = {}
    # Graded-strength supervision (opt-in). Off by default, in which case the
    # forward keeps its historical 2-tuple shape and no metric key is emitted.
    want_strength = cfg.strength_loss_weight > 0 and is_antigen_stage(cfg.training_stage)
    strength_pred_all: list[float] = []
    strength_target_all: list[float] = []
    # Learned length posterior (opt-in). The length query is a SEPARATE forward
    # on the collapsed-span encoding, so it runs only when the loss is active.
    want_length = cfg.length_loss_weight > 0 and is_antigen_stage(cfg.training_stage)
    length_correct = 0
    length_total = 0
    length_nll_sum = 0.0
    # Conditional-denoising eligibility census. Counted rather than asserted per
    # batch: an all-nonbinder batch is legitimate under `binary_binders_only`.
    # The whole-epoch total is what must not be zero.
    conditional_eligible_rows: torch.Tensor | int = 0
    conditional_total_rows = 0
    conditional_eligible_tokens: torch.Tensor | int = 0

    progress = _make_progress_bar(
        train_loader,
        total=len(train_loader),
        desc=f"train {epoch + 1}/{cfg.epochs}",
        cfg=cfg,
    )
    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
            if is_antigen_stage(cfg.training_stage):
                if want_strength:
                    logits, compatibility_logits, strength_predictions = model(
                        antibody_input_ids=batch["antibody_input_ids"],
                        antibody_attention_mask=batch["antibody_attention_mask"],
                        antigen_input_ids=batch["antigen_input_ids"],
                        antigen_attention_mask=batch["antigen_attention_mask"],
                        return_strength=True,
                    )
                else:
                    strength_predictions = None
                    logits, compatibility_logits = model(
                        antibody_input_ids=batch["antibody_input_ids"],
                        antibody_attention_mask=batch["antibody_attention_mask"],
                        antigen_input_ids=batch["antigen_input_ids"],
                        antigen_attention_mask=batch["antigen_attention_mask"],
                    )
                length_logits = None
                if want_length:
                    # Separate forward: the length query uses the COLLAPSED-span antibody
                    # stream, which is different tensors from the MLM forward above.
                    length_logits = model.forward_length_query(
                        antibody_input_ids=batch["length_query_input_ids"],
                        antibody_attention_mask=batch["length_query_attention_mask"],
                        antigen_input_ids=batch["antigen_input_ids"],
                        antigen_attention_mask=batch["antigen_attention_mask"],
                    )
                # Accumulated as device tensors and synced ONCE after the loop.
                # A `.item()` here would add two host syncs per step.
                eligible_rows = batch.get("conditional_denoising_eligible")
                if eligible_rows is not None:
                    conditional_eligible_rows += eligible_rows.sum()
                    conditional_total_rows += eligible_rows.numel()
                conditional_eligible_tokens += (
                    batch["antibody_labels"] != MLM_IGNORE_INDEX
                ).sum()
                losses = model.compute_losses(
                    mlm_logits=logits,
                    labels=batch["antibody_labels"],
                    compatibility_logits=compatibility_logits,
                    compatibility_labels=batch["compatibility_labels"],
                    compatibility_mask=batch["compatibility_mask"],
                    strength_predictions=strength_predictions if want_strength else None,
                    strength_targets=batch.get("strength_targets") if want_strength else None,
                    strength_mask=batch.get("strength_mask") if want_strength else None,
                    length_logits=length_logits,
                    length_labels=batch["length_labels"] if want_length else None,
                    length_mask=batch["length_label_mask"] if want_length else None,
                    mlm_loss_weight=cfg.mlm_loss_weight,
                    compatibility_loss_weight=cfg.compatibility_loss_weight,
                    strength_loss_weight=cfg.strength_loss_weight,
                    length_loss_weight=cfg.length_loss_weight,
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

        # Intra-epoch checkpoint (epoch-granular resume). Save the current
        # 0-based epoch index so a resumed run re-enters this epoch from batch 0
        # with these advanced weights; best_val_loss is carried through so
        # best-tracking survives the resume. Disabled unless both an output_dir
        # and cfg.checkpoint_every_steps > 0 are provided.
        if (
            output_dir is not None
            and cfg.checkpoint_every_steps > 0
            and (step + 1) % cfg.checkpoint_every_steps == 0
        ):
            save_checkpoint(
                path=output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                # This epoch is still in progress, so it has no score of its own;
                # the running best is the only meaningful value for both fields.
                val_loss=best_val_loss,
                scaler=scaler,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                # An intra-epoch last.pt is the checkpoint most likely to be
                # resumed from, so it needs the fingerprint most.
                run_fingerprint=run_fingerprint,
            )

        if is_antigen_stage(cfg.training_stage):
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits.detach(), batch["antibody_labels"])
            if want_fraction_bins:
                for _key, (_c, _t) in masked_fraction_bin_counts(
                    logits.detach(), batch["antibody_labels"], batch["antibody_input_ids"],
                    tokenizer.special_ids,
                ).items():
                    _entry = fraction_bin_counts.setdefault(_key, [0, 0])
                    _entry[0] += _c
                    _entry[1] += _t
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
            # Length / strength accumulation, mirroring `evaluate`. Without this
            # the accumulators declared above stayed at their initial values and
            # `train_length_acc` / `train_val_strength_spearman` were reported as
            # NaN over 0 rows on EVERY run with those heads enabled -- which reads
            # as "measured, no signal" rather than "never measured".
            if want_length and length_logits is not None:
                _lm = batch["length_label_mask"]
                if bool(_lm.any()):
                    _labels = batch["length_labels"][_lm]
                    _lq = length_logits.detach()[_lm]
                    length_correct += int((_lq.argmax(dim=-1) == _labels).sum().item())
                    length_total += int(_lm.sum().item())
                    length_nll_sum += float(
                        F.cross_entropy(_lq.float(), _labels, reduction="sum").item()
                    )
            if want_strength and strength_predictions is not None:
                _sm = batch["strength_mask"]
                if bool(_sm.any()):
                    strength_pred_all.extend(
                        strength_predictions.detach()[_sm].float().cpu().tolist()
                    )
                    strength_target_all.extend(
                        batch["strength_targets"][_sm].float().cpu().tolist()
                    )
            aux_loss_name = "compatibility_loss"
            aux_acc_name = "compatibility_acc"
        else:
            mlm_correct, mlm_tokens = masked_accuracy_counts(logits.detach(), batch["labels"])
            if want_fraction_bins:
                for _key, (_c, _t) in masked_fraction_bin_counts(
                    logits.detach(), batch["labels"], batch["input_ids"],
                    tokenizer.special_ids,
                ).items():
                    _entry = fraction_bin_counts.setdefault(_key, [0, 0])
                    _entry[0] += _c
                    _entry[1] += _t
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
            loss=f"{_mlm_weight * running_mlm_loss + _aux_weight * running_aux_loss:.4f}",
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
            # Emitted even on an empty epoch so the metrics.jsonl key set does not
            # vary between epochs.
            metrics["conditional_denoising_policy_rows"] = 0.0
            metrics["conditional_denoising_eligible_rows"] = 0.0
            metrics["conditional_denoising_eligible_tokens"] = 0.0
        else:
            metrics["pair_loss"] = float("nan")
            metrics["pair_acc"] = float("nan")
        metrics.update(finalize_hcdr3_metrics(hcdr3_counts))
        return metrics

    mlm_loss = (total_mlm_loss_weighted / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    mlm_acc = (total_mlm_correct / total_mlm_tokens) if total_mlm_tokens > 0 else float("nan")
    aux_loss = (total_aux_loss_weighted / total_aux_labeled) if total_aux_labeled > 0 else 0.0
    metrics = {
        "loss": _mlm_weight * mlm_loss + _aux_weight * aux_loss,
        "mlm_loss": mlm_loss,
        "mlm_acc": mlm_acc,
    }
    if want_fraction_bins:
        metrics.update(finalize_masked_fraction_bins(fraction_bin_counts))
    if want_length:
        # Accuracy AND NLL: a categorical length head can look accurate while
        # being badly calibrated, and the proposal path SAMPLES from the posterior
        # rather than taking its argmax.
        metrics["length_eligible_rows"] = float(length_total)
        metrics["length_acc"] = (
            (length_correct / length_total) if length_total > 0 else float("nan")
        )
        metrics["length_nll"] = (
            (length_nll_sum / length_total) if length_total > 0 else float("nan")
        )
    if want_strength:
        # Reported next to compatibility_auroc so the graded head is judged on
        # ORDERING (what a ranking objective is for), not on MSE, whose scale is
        # not comparable across quantile populations. NaN with <2 eligible rows.
        metrics["strength_eligible_rows"] = float(len(strength_target_all))
        metrics["val_strength_spearman"] = float(
            tie_aware_spearman(strength_pred_all, strength_target_all)
        )
    if is_antigen_stage(cfg.training_stage):
        conditional_eligible_rows = int(conditional_eligible_rows)
        conditional_eligible_tokens = int(conditional_eligible_tokens)
        metrics["conditional_denoising_policy_rows"] = float(conditional_total_rows)
        metrics["conditional_denoising_eligible_rows"] = float(conditional_eligible_rows)
        metrics["conditional_denoising_eligible_tokens"] = float(conditional_eligible_tokens)
        # A whole epoch with nothing to denoise is fatal under EITHER policy. Per
        # batch this is legitimate under `binary_binders_only`; per epoch it means
        # the conditional policy learned nothing, and MLM loss over an all-ignored
        # batch is a finite differentiable zero, so the loss curve would not show
        # it. See specs/conditional_denoising_eligibility.md.
        if conditional_total_rows > 0 and conditional_eligible_rows == 0:
            raise ValueError(
                f"epoch {epoch + 1}: conditional_denoising_eligibility="
                f"'{cfg.conditional_denoising_eligibility}' left 0 of "
                f"{conditional_total_rows} training rows eligible for antigen-conditioned "
                "MLM across the whole epoch."
            )
        if conditional_total_rows > 0 and conditional_eligible_tokens == 0:
            raise ValueError(
                f"epoch {epoch + 1}: {conditional_eligible_rows} rows were eligible for "
                "antigen-conditioned MLM but they produced 0 target tokens across the "
                "whole epoch, so the conditional policy received no gradient."
            )
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
    best_val_loss: float | None = None,
    run_fingerprint: Dict[str, Any] | None = None,
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
            The selection metric for THIS checkpoint's weights.
        best_val_loss:
            The best selection value seen so far in the run, which is what a
            resume must restore into best-tracking. It is stored separately from
            ``val_loss`` because the two are different quantities and conflating
            them destroys ``best.pt``:

            ``last.pt``'s ``val_loss`` is the LAST epoch's score, not the best.
            A resume that seeded best-tracking from it would start with an
            inflated threshold, so the first mediocre post-resume epoch beats it
            and overwrites ``best.pt`` with weights worse than the pre-interrupt
            best -- silently, and with every shipped config setting
            ``resume_from_last: true``.

            Defaults to ``val_loss`` so the two intra-epoch/legacy callers that
            already pass the running best in ``val_loss`` keep working.
        run_fingerprint:
            The run-provenance payload from
            ``experiment.compute_run_fingerprint`` -- component hashes for
            architecture / objective / tokenizer / data / contracts / source,
            the combined run hash, the parent checkpoint hash, and the
            dirty-worktree indicator. ``main`` computes it once per run (it
            hashes the corpus and the whole source tree) and hands the same
            object to every save. ``None`` writes a checkpoint with no
            fingerprint, which ``resume_from_last`` then refuses; that is the
            legacy shape and only direct callers (tests, tools) produce it.

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
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state,
        "val_loss": val_loss,
        # The running best, kept separate from this checkpoint's own val_loss so a
        # resume restores best-tracking rather than the last epoch's score.
        "best_val_loss": (val_loss if best_val_loss is None else best_val_loss),
        "train_config": asdict(cfg),
    }
    if run_fingerprint is not None:
        payload[experiment.RUN_FINGERPRINT_KEY] = run_fingerprint
    # Atomic write: serialize to a temp file on the same filesystem, then
    # os.replace() onto the final path. os.replace is atomic on both POSIX and
    # Windows, so a crash mid-write leaves the previous checkpoint intact rather
    # than truncating the only resume point. This matters most for frequent
    # intra-epoch saves (cfg.checkpoint_every_steps), where the odds of a crash
    # landing during a write are much higher than with once-per-epoch saves.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    
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
    # A DISABLED GradScaler serializes to `{}`, not to None, and
    # GradScaler.load_state_dict raises "The source state dict is empty" on it
    # (its `if not self._enabled: return` early-out keys off the LOADING scaler,
    # which is enabled here). So a run started with use_amp=false and resumed with
    # use_amp=true on CUDA aborted -- and refine_oas_paired.yaml literally invites
    # that flip ("Flip to true for speed if stable"). An empty dict carries no
    # scale to restore, so skipping it and starting at the default scale is correct.
    if scaler is not None and checkpoint.get("scaler_state_dict"):
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

    # When the antigen stream is an ESM encoder, the checkpoint's scratch
    # `antigen_encoder.*` weights are meaningless for it. Drop them so the ESM
    # backbone keeps its pretrained weights (and the projection its fresh init)
    # while the antibody encoder, cross-attention fusion, and heads are still
    # warm-started from the checkpoint.
    antigen_is_esm = (
        getattr(getattr(model, "config", None), "antigen_encoder_type", "scratch") == "esm"
    )

    def _drop_scratch_antigen(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not antigen_is_esm:
            return state
        return {k: v for k, v in state.items() if not k.startswith("antigen_encoder.")}

    has_dual_stream_weights = any(
        key.startswith("antibody_encoder.") for key in checkpoint_state_dict
    )
    if has_dual_stream_weights:
        incompatible = model.load_state_dict(
            _drop_scratch_antigen(checkpoint_state_dict), strict=False
        )
        reused_message = (
            "antibody_encoder.* + fusion/heads (ESM antigen kept at pretrained init)"
            if antigen_is_esm
            else "dual-stream checkpoint weights"
        )
    else:
        translated_state_dict = build_antigen_refine_init_state_dict(checkpoint_state_dict)
        incompatible = model.load_state_dict(
            _drop_scratch_antigen(translated_state_dict), strict=False
        )
        reused_message = (
            "antibody_encoder.*, lm_head.* (ESM antigen kept at pretrained init)"
            if antigen_is_esm
            else "antibody_encoder.*, antigen_encoder.*, lm_head.*"
        )

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
    tokenizer: AminoAcidTokenizer | None = None,
) -> None:
    """
    Validate architecture compatibility between run config and init checkpoint.

    Warm-start rules are deliberately SEPARATE from the resume rules. A warm
    start is meant to change the objective and the data -- that is what a stage
    transition is -- so neither is checked here. What it may not change is the
    architecture the weights were trained under or the tokenizer that assigned
    their token ids.

    Legacy parents (no ``run_fingerprint``) are ALLOWED through, still run every
    check below, and get a loud warning that their lineage is unverifiable.

    Args:
        cfg:
            Current run config.
        init_ckpt_path:
            Optional initialization checkpoint path.
        tokenizer:
            Tokenizer this run will use. Defaults to `build_tokenizer()`; the
            parameter exists so callers that already built one do not build a
            second.

    Returns:
        None.
    """
    if init_ckpt_path is None:
        return

    checkpoint = torch.load(init_ckpt_path, map_location="cpu")

    # Lineage first, and non-fatally: a fingerprinted parent gets its
    # architecture/tokenizer compared by content, a legacy parent gets a warning.
    parent_fingerprint = experiment.read_fingerprint(checkpoint)
    warning = experiment.warm_start_lineage_warning(parent_fingerprint, init_ckpt_path)
    if warning is not None:
        print(warning)
    run_tokenizer = tokenizer if tokenizer is not None else build_tokenizer()
    run_model_cfg = build_model_config(run_tokenizer, cfg)
    if parent_fingerprint is not None:
        current_fingerprint = experiment.compute_run_fingerprint(
            config=asdict(cfg),
            model_config=run_model_cfg,
            tokenizer=run_tokenizer,
            model_class=model_class_for_stage(cfg.training_stage),
            # A warm start is allowed to change the corpus, so the data
            # component is irrelevant here and is left uncomputed rather than
            # paying a full corpus hash for a value nothing reads.
            data_paths=(),
            repo_root=PROJECT_ROOT,
        )
        experiment.check_warm_start_fingerprint(
            parent_fingerprint, current_fingerprint, init_ckpt_path
        )

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

    # `norm_first` is checked separately because it must NOT take the
    # skip-when-absent path above. Checkpoints written before the knob existed
    # carry no `norm_first` key and were all post-LN, so a missing key means
    # False, not "unknown". Skipping would wave a pre-LN run through against a
    # post-LN checkpoint -- and for the single-stream model that mismatch is
    # invisible to `strict=True`, since post-LN and pre-LN encoder layers have
    # identical parameter names and shapes and differ only in forward order.
    ckpt_norm_first = bool(train_cfg.get("norm_first", False))
    if bool(cfg.norm_first) != ckpt_norm_first:
        mismatches.append(
            f"norm_first: checkpoint={ckpt_norm_first}, run={bool(cfg.norm_first)} "
            "(pre-LN and post-LN weights are not interchangeable; retrain from scratch)"
        )

    # `compat_readout` has exactly the same hazard as `norm_first`: the two modes
    # share a parameter set byte-for-byte (only the pooling that feeds
    # `fusion_mlp` differs), so `strict=True` cannot see the mismatch. Absent key
    # means "cls" -- the historical readout -- not "unknown".
    ckpt_compat_readout = str(train_cfg.get("compat_readout", "cls"))
    if str(cfg.compat_readout) != ckpt_compat_readout:
        mismatches.append(
            f"compat_readout: checkpoint={ckpt_compat_readout}, run={cfg.compat_readout} "
            "(the readouts share a parameter set, so a strict load cannot catch this; "
            "the fusion_mlp input distribution differs and the compatibility head "
            "would be read off-distribution)"
        )

    # `activation` and `tie_weights` live ONLY on `MLMConfig` -- they are
    # hardcoded in `build_model_config` and unreachable from `TrainConfig`, so
    # no checkpoint written before J03 records them at all. They get the same
    # absent-means-a-specific-legacy-value treatment as `norm_first`: every
    # checkpoint this repo has ever produced used "gelu" and tied weights.
    #
    # `activation` is the more dangerous of the two: swapping GELU for ReLU
    # changes no parameter name and no shape, so `strict=True` loads the
    # checkpoint happily and the model then computes something else. A
    # `tie_weights` flip does change the parameter set, so this only turns a
    # confusing strict-load error into a named one.
    #
    # `initializer_range` and `scale_residual_init` are deliberately NOT checked:
    # they shape a FROM-SCRATCH init only, and the loaded weights overwrite it.
    # They are still part of the architecture fingerprint, because a resume must
    # reproduce the run that produced the weights.
    ckpt_architecture = (
        (experiment.read_fingerprint(checkpoint) or {}).get("manifests", {}).get("architecture", {})
    )
    ckpt_model_cfg = ckpt_architecture.get("model_config", {}) if ckpt_architecture else {}
    for key, legacy_value in (("activation", "gelu"), ("tie_weights", True)):
        ckpt_value = ckpt_model_cfg.get(key, legacy_value)
        run_value = getattr(run_model_cfg, key)
        if run_value != ckpt_value:
            mismatches.append(
                f"{key}: checkpoint={ckpt_value}, run={run_value} "
                "(MLMConfig-only field; a legacy checkpoint with no recorded value is "
                f"read as the historical {legacy_value!r})"
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "init_checkpoint architecture mismatch. Use the same base-model "
            f"hyperparameters for refinement. Mismatches: {details}"
        )


def parent_checkpoint_descriptor(
    init_ckpt_path: Path | None,
) -> Dict[str, Any] | None:
    """
    Describe the checkpoint this run warm-starts from, for the lineage record.

    Records the parent's own ``run_hash`` when it has one, and always its file
    content hash, so a legacy parent (no fingerprint) still has a stable
    content identity. ``None`` for a from-scratch run.
    """
    if init_ckpt_path is None:
        return None
    payload = torch.load(init_ckpt_path, map_location="cpu")
    parent_fingerprint = experiment.read_fingerprint(payload)
    return {
        "path": experiment.normalize_path_value(init_ckpt_path, PROJECT_ROOT),
        "run_hash": parent_fingerprint["run_hash"] if parent_fingerprint else None,
        "file_sha256": experiment.hash_file(init_ckpt_path),
    }


def build_run_fingerprint(
    cfg: TrainConfig,
    tokenizer: AminoAcidTokenizer,
    init_ckpt_path: Path | None,
) -> Dict[str, Any]:
    """
    Compute this run's provenance fingerprint (J03).

    Called ONCE per run: it hashes the corpus and every source file, so it is
    not something to recompute per checkpoint save. The same payload is written
    into every checkpoint and into ``output_dir/run_fingerprint.json``.

    The architecture component is derived from the CONSTRUCTED `MLMConfig`
    rather than from `asdict(cfg)`, because `activation`, `tie_weights`,
    `initializer_range` and `scale_residual_init` exist only on `MLMConfig`.
    """
    return experiment.compute_run_fingerprint(
        config=asdict(cfg),
        model_config=build_model_config(tokenizer, cfg),
        tokenizer=tokenizer,
        model_class=model_class_for_stage(cfg.training_stage),
        data_paths=[cfg.data_path],
        repo_root=PROJECT_ROOT,
        parent_checkpoint=parent_checkpoint_descriptor(init_ckpt_path),
    )


def select_checkpoint_metric_value(cfg: TrainConfig, val_metrics: Dict[str, float]) -> float:
    """
    The scalar that drives best.pt selection, early stopping, and the
    checkpoint's tracked ``val_loss`` payload.

    Default ``"val_loss"`` returns the combined validation loss -- the historical
    behavior, byte-identical. ``"val_compat_loss"`` returns the validation
    compatibility term instead; the checkpoint's ``"val_loss"`` field then stores
    that selection value, which is exactly what the resume path restores into
    best-tracking (``main`` reads ``checkpoint["val_loss"]``), so resume stays
    coherent under either metric.

    Why the knob exists: on the antigen stages the combined loss is dominated by
    the MLM term, so "best" can advance on an epoch where the compatibility head
    got worse -- and the compatibility head is what downstream scoring and
    guidance actually read.
    """
    if cfg.best_checkpoint_metric == "val_compat_loss":
        return float(val_metrics["compatibility_loss"])
    return float(val_metrics["loss"])


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

    # Run provenance (J03). Computed once, before any data is loaded, so a
    # promotion refusal costs nothing; written next to the checkpoints and
    # embedded in every one of them.
    run_fingerprint = build_run_fingerprint(cfg, tokenizer, init_ckpt_path)
    with open(output_dir / "run_fingerprint.json", "w", encoding="utf-8") as f:
        json.dump(run_fingerprint, f, indent=2, sort_keys=True)
    print(f"[fingerprint] run_hash={run_fingerprint['run_hash']}")
    for _component, _digest in sorted(run_fingerprint["components"].items()):
        print(f"[fingerprint]   {_component}={_digest}")
    _source = run_fingerprint["manifests"]["source"]
    print(
        f"[fingerprint] source commit={_source['commit']} "
        f"dirty={_source['dirty']} files={len(_source['files'])}"
    )
    if cfg.require_clean_worktree:
        # Promoted canonical run: refuse a dirty OR unverifiable worktree.
        experiment.require_clean_worktree(run_fingerprint)
    elif _source["dirty"]:
        print(
            f"[warn] development run from a DIRTY worktree ({len(_source['dirty_paths'])} "
            "path(s) modified or untracked). The complete source-content hash and the "
            "dirty path list are recorded in run_fingerprint.json, so this run is "
            "identifiable -- but it is not a promoted canonical run. Re-run with "
            "--require-clean-worktree from a clean checkout to promote it."
        )

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
    # Preflight: how much of the corpus max_length silently deletes. Reported
    # before training starts, because the answer can invalidate the objective (see
    # summarize_length_truncation) and the tokenizer's own warning is deduplicated
    # away at corpus scale.
    for split_name, split_dataset in (("train", train_dataset), ("val", val_dataset)):
        truncation_warning = format_length_truncation_warning(
            summarize_length_truncation(split_dataset, cfg.max_length),
            cfg.max_length,
            split_name,
        )
        if truncation_warning:
            print(truncation_warning)

    # Preflight: does this stage actually have anything to denoise? Reported for
    # every antigen stage and FATAL when a split is empty of eligible rows, because
    # the failure is otherwise invisible -- an all-ignored MLM batch returns a
    # finite zero, so the run would train on almost nothing behind a healthy loss
    # curve. See specs/conditional_denoising_eligibility.md.
    if is_antigen_stage(cfg.training_stage):
        for split_name, split_dataset in (("train", train_dataset), ("val", val_dataset)):
            counts = summarize_conditional_denoising_eligibility(
                split_dataset, cfg.conditional_denoising_eligibility
            )
            share = (counts["eligible"] / counts["total"]) if counts["total"] else 0.0
            print(
                f"[conditional-denoising] {split_name}: "
                f"policy={cfg.conditional_denoising_eligibility} "
                f"eligible={counts['eligible']}/{counts['total']} ({share:.1%})"
            )
            if counts["total"] and counts["eligible"] == 0:
                raise ValueError(
                    f"conditional_denoising_eligibility="
                    f"'{cfg.conditional_denoising_eligibility}' leaves 0 of "
                    f"{counts['total']} {split_name} rows eligible for antigen-conditioned "
                    "MLM. The stage would train its conditional policy on nothing while "
                    "reporting a finite zero MLM loss. Check the stage's dataset filter "
                    "and the eligibility policy."
                )

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
    # Cosine decay needs the full training horizon; derive it from the epoch-0
    # loader length (constant across epochs) x epochs. Constant/warmup-only runs
    # ignore total_steps, so this is harmless there.
    steps_per_epoch = len(build_train_loader(train_dataset, tokenizer, cfg, epoch=0, device=device))
    total_training_steps = steps_per_epoch * cfg.epochs
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps=total_training_steps)
    tb_writer = build_tensorboard_writer(cfg, output_dir)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 0

    print(f"training_stage: {cfg.training_stage}")

    # Resume from last checkpoint if configured and available.
    last_ckpt_path = output_dir / "last.pt"
    if cfg.resume_from_last and last_ckpt_path.exists():
        # PROVENANCE GATE (J03 step 4). This runs BEFORE `load_checkpoint`, so
        # model/optimizer/scaler/scheduler are still pristine when it raises: a
        # rejected resume leaves nothing half-restored. `torch.load` here reads
        # only the fingerprint payload; no state is applied to anything.
        #
        # An exact run-fingerprint match is required. A checkpoint with no
        # fingerprint at all (every checkpoint written before J03) is a hard
        # error rather than a warning -- there is nothing to verify against, and
        # every shipped config sets `resume_from_last: true`, so a silent
        # legacy resume is exactly the accident this ticket exists to prevent.
        experiment.check_resume_fingerprint(
            experiment.read_fingerprint(torch.load(last_ckpt_path, map_location="cpu")),
            run_fingerprint,
            last_ckpt_path,
        )
        checkpoint = load_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = checkpoint["epoch"]
        # Prefer the explicitly-tracked running best. `val_loss` is THIS
        # checkpoint's own score, and for `last.pt` that is the LAST epoch's, not
        # the best -- seeding best-tracking from it inflates the threshold so the
        # first mediocre post-resume epoch overwrites `best.pt` with weights worse
        # than the pre-interrupt best. The `val_loss` fallback exists only for
        # checkpoints written before `best_val_loss` was stored; those carry no
        # better information, so a legacy resume keeps the old behavior and says so.
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]
        else:
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(
                "[warn] last.pt predates best_val_loss tracking; seeding best-tracking "
                f"from its val_loss ({best_val_loss}). If best.pt currently holds a "
                "better model, this resume may overwrite it -- back best.pt up first."
            )
        # Resume is EPOCH-granular: `start_epoch` is re-entered from batch 0, so an
        # intra-epoch `last.pt` (written by checkpoint_every_steps, which stores the
        # 0-based in-progress epoch) has a scheduler step count ahead of the epoch
        # boundary we are restarting from. Left alone, those steps are taken twice
        # and the LR curve runs ahead of real training progress -- with cosine that
        # pins the tail of the run at min_lr_ratio (0.0 by default) while training
        # continues, and every further interrupt shifts it more. Rewind to the
        # boundary so scheduler position and data position agree.
        if scheduler is not None:
            boundary_step = start_epoch * steps_per_epoch
            if scheduler.last_epoch > boundary_step:
                print(
                    f"[checkpoint] rewinding LR scheduler {scheduler.last_epoch} -> "
                    f"{boundary_step} to match the epoch-granular resume "
                    f"(epoch {start_epoch} restarts from batch 0)"
                )
                scheduler.last_epoch = boundary_step
                # Recompute each group's LR from its OWN base_lr and lambda:
                # new_module_lr_multiplier gives the fusion/head groups a different
                # base LR, so a single shared lambda/base would corrupt them.
                for group, base_lr, lr_lambda in zip(
                    optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
                ):
                    group["lr"] = base_lr * lr_lambda(boundary_step)
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
            model,
            smoke_loader,
            optimizer,
            device,
            cfg.use_amp,
            cfg.training_stage,
            cfg.grad_clip_norm,
            cfg.conditional_denoising_eligibility,
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
            output_dir=output_dir,
            best_val_loss=best_val_loss,
            run_fingerprint=run_fingerprint,
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

        # best.pt / early stopping / the checkpoint's tracked val_loss all key on
        # this; it IS `val_metrics["loss"]` unless `best_checkpoint_metric` is set.
        selection_value = select_checkpoint_metric_value(cfg, val_metrics)

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

        log_epoch_scalars(
            tb_writer,
            epoch + 1,
            train_metrics,
            val_metrics,
            optimizer.param_groups[0]["lr"],
        )

        # Early stopping keys off the best *before* this epoch's update, so
        # capture it before best_val_loss potentially advances below.
        prev_best_val_loss = best_val_loss

        # Guard against NaN/Inf: `NaN < best` is always False, so a non-finite
        # val_loss would silently neither win nor warn. Skip and report instead.
        improved = math.isfinite(selection_value) and selection_value < best_val_loss
        if improved:
            best_val_loss = selection_value
        elif not math.isfinite(selection_value):
            print(
                f"[warn] {cfg.best_checkpoint_metric} is non-finite ({selection_value}); "
                "best.pt not updated this epoch"
            )

        # best.pt is written BEFORE last.pt, and `best_val_loss` is advanced before
        # either. Both orderings matter for crash safety:
        #
        # - Advancing first means last.pt records the CURRENT running best, so a
        #   resume restores the right threshold instead of the last epoch's score.
        # - Writing best.pt first means a crash between the two leaves best.pt
        #   holding the better weights while last.pt still points at the previous
        #   epoch. The resume then re-runs that epoch and re-establishes the same
        #   best. The reverse order would leave last.pt CLAIMING a best that
        #   best.pt does not hold, stranding the good weights.
        if improved:
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                val_loss=selection_value,
                scaler=scaler,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                run_fingerprint=run_fingerprint,
            )

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch + 1,
            val_loss=selection_value,
            scaler=scaler,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            run_fingerprint=run_fingerprint,
        )

        epochs_without_improvement, should_stop = early_stopping_decision(
            val_loss=selection_value,
            best_val_loss=prev_best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
        )
        if should_stop:
            print(
                f"[early-stop] no {cfg.best_checkpoint_metric} improvement for {epochs_without_improvement} "
                f"epoch(s) (patience={cfg.early_stopping_patience}); stopping at epoch "
                f"{epoch + 1}, best val_loss={best_val_loss:.4f}"
            )
            break

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
    

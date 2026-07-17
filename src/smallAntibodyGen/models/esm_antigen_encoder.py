from __future__ import annotations

"""
ESM-2 backed antigen-stream encoder (Direction 1, Stage A/B).

Drops into ``AntibodyAntigenCrossAttention`` in place of the from-scratch
``TransformerSequenceEncoder``: it consumes ESM antigen token ids (produced by
``ESMAntigenTokenizer``) and returns per-residue hidden states projected to
``d_model`` together with the attention mask, matching the
``(hidden[B, L, d_model], attention_mask)`` contract. ESM ``<cls>`` sits at index 0,
so ``hidden[:, 0, :]`` is the antigen summary the fusion layer reads.

``transformers`` (and ``peft`` for LoRA) are imported lazily so the base install and
every from-scratch stage never require them.
"""

import torch
from torch import nn

from smallAntibodyGen.models.mlm import MLMConfig


class ESMAntigenEncoder(nn.Module):
    """
    Pretrained-ESM antigen encoder with a learned projection to ``d_model``.

    Finetune modes (``config.antigen_encoder_finetune``):
    - ``"frozen"``: ESM weights frozen; the ESM forward runs under ``no_grad`` and only
      the projection (and downstream cross-attention/heads) train. This is the Stage A
      ablation configuration.
    - ``"lora"``: LoRA adapters are attached to the ESM attention projections; the ESM
      base stays frozen while the adapters + projection train. This is Stage B.
    """

    def __init__(self, config: MLMConfig) -> None:
        super().__init__()
        self.config = config
        self.finetune = config.antigen_encoder_finetune

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - only without the 'esm' extra
            raise ImportError(
                "antigen_encoder_type='esm' requires the optional 'esm' extra. "
                "Install it with `pip install -e \".[esm]\"` (transformers + peft)."
            ) from exc

        esm = AutoModel.from_pretrained(config.esm_model_name)
        # Capture these before any LoRA wrapping, which can proxy `.config`.
        esm_hidden = int(esm.config.hidden_size)
        self._pad_token_id = int(getattr(esm.config, "pad_token_id", 1))

        for param in esm.parameters():
            param.requires_grad_(False)

        if self.finetune == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:  # pragma: no cover - only without the 'esm' extra
                raise ImportError(
                    "antigen_encoder_finetune='lora' requires 'peft' (the 'esm' extra)."
                ) from exc
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["query", "key", "value"],
                bias="none",
            )
            esm = get_peft_model(esm, lora_config)

        self.esm = esm
        self.projection = nn.Linear(esm_hidden, config.d_model)
        self.projection_norm = nn.LayerNorm(config.d_model)
        self.projection_dropout = nn.Dropout(config.dropout)

    def _run_esm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return ESM per-residue hidden states ``[B, L, esm_hidden]``."""
        return self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one antigen batch and return projected hidden states plus the mask.

        Args:
            input_ids: ``[B, L]`` ESM antigen token ids (from ``ESMAntigenTokenizer``).
            attention_mask: ``[B, L]`` with 1 for real tokens and 0 for padding. When
                omitted it is inferred from the ESM pad id.

        Returns:
            Tuple ``(hidden[B, L, d_model], attention_mask)``. When the ESM backbone is
            frozen its forward runs under ``no_grad`` while the projection still trains,
            because the projection weights carry gradient even on a no-grad input.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        if attention_mask is None:
            attention_mask = (input_ids != self._pad_token_id).long()
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        if self.finetune == "frozen":
            with torch.no_grad():
                esm_hidden = self._run_esm(input_ids, attention_mask)
        else:
            esm_hidden = self._run_esm(input_ids, attention_mask)

        hidden = self.projection(esm_hidden)
        hidden = self.projection_norm(hidden)
        hidden = self.projection_dropout(hidden)
        return hidden, attention_mask

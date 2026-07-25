from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F 
from torch import nn 

@dataclass
class MLMConfig: 
    """
    Configuration object for the antibody masked language model. 
    
    Stores all hyperparameters critical for the model. 
    
    Attributes:
        vocab_size (int):
            Number of tokens in the tokenizer vocabulary
        pad_token_id (int): 
            Integer ID used for the padding token
        max_length (int): 
            Maximum tokenized sequence length the model should support 
        d_model: (int) 
            Hidden dimension of embeddings and transformer states
        n_heads (int):
            Number of attention heads in each transformer layer
        n_layers (int):
            Number of transformer encoder layers
        d_ff (int):
            Hidden size of the feed-forward block inside each transformer layer
        dropout (float):
            Dropout probability aplied in embeddings and transformer blocks
        activation (str):
            Nonlinearity used in transformer feed-forward layers. 
            Options are "relu", Rectified Linear Unit, or "gelu", Gaussian Error Linear Unit
        tie_weights (bool):
            Whether to tie the output project weights to the input token embeddings
        norm_first (bool):
            Where LayerNorm sits relative to each residual sublayer.
            ``False`` (default) is post-LN — ``LayerNorm(x + sublayer(x))``, the
            original Vaswani-2017 arrangement and the PyTorch default. ``True``
            is pre-LN — ``x + sublayer(LayerNorm(x))`` — which keeps an
            unnormalized identity path from input to output and is what modern
            transformer stacks use, because gradients reach early layers without
            being rescaled at every depth.

            This flag governs the encoder stacks AND the cross-attention fusion
            block, so the two cannot drift apart.

            WARNING: post-LN and pre-LN produce *identical* parameter names and
            shapes. A checkpoint trained under one setting loads into the other
            under ``strict=True`` without complaint and then computes something
            different. Changing this value requires retraining from scratch;
            ``validate_init_checkpoint_compatibility`` in ``scripts/mlm_train.py``
            is what makes the mismatch fatal rather than silent.
        antigen_encoder_type (str):
            Which encoder backs the antigen stream of the dual-stream model.
            ``"scratch"`` (default) keeps the original in-repo
            ``TransformerSequenceEncoder``. ``"esm"`` selects a pretrained
            protein-language-model encoder (Direction 1: hybrid antigen encoder).
            Only consulted by ``AntibodyAntigenCrossAttention``; the antibody-only
            ``AntibodyMLM`` ignores it.
        esm_model_name (str):
            HuggingFace model id for the ESM backbone used when
            ``antigen_encoder_type == "esm"`` (default ESM-2 8M).
        antigen_max_length (int):
            Maximum antigen token length for the ESM antigen stream. Independent
            of ``max_length`` (which bounds the antibody stream) because the two
            encoders are separate and only interact through cross-attention.
        antigen_encoder_finetune (str):
            How the ESM antigen encoder is trained: ``"frozen"`` (features only,
            projection/cross-attention/heads trainable) or ``"lora"`` (LoRA
            adapters on the ESM backbone). Ignored for the scratch encoder.
        lora_r (int):
            LoRA rank used when ``antigen_encoder_finetune == "lora"``.
        lora_alpha (int):
            LoRA scaling factor used when ``antigen_encoder_finetune == "lora"``.
        lora_dropout (float):
            LoRA dropout used when ``antigen_encoder_finetune == "lora"``.
    """

    vocab_size: int
    pad_token_id: int
    max_length: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    tie_weights: bool = True
    # Post-LN by default so every pre-existing config and checkpoint is
    # unchanged. See the attribute docs above: flipping this is a retrain, not a
    # migration, because the state dict is silently compatible either way.
    norm_first: bool = False

    # Antigen-stream encoder selection (Direction 1: hybrid PLM antigen encoder).
    # Defaults preserve the original from-scratch dual-stream behavior, so these
    # fields are inert until AntibodyAntigenCrossAttention branches on them.
    antigen_encoder_type: str = "scratch"
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    antigen_max_length: int = 512
    antigen_encoder_finetune: str = "frozen"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Readout feeding fusion_mlp for the compatibility head. "cls" (default)
    # reproduces the historical CLS-concat byte-for-byte; "mean" mask-aware
    # mean-pools each fused stream over its non-pad positions ([CLS] included)
    # before concatenating. No parameters change in either mode, so a checkpoint
    # trained under one readout loads under the other — which is exactly why this
    # field is written into the checkpoint config and matched at warm-start.
    #
    # Why the knob exists: the CLS-concat readout collapses each stream to one
    # position, so the compatibility logit is only weakly sensitive to a single
    # residue substitution — and that logit is the guidance term
    # ``guided_infill`` steers with. "mean" is the cheapest readout that lets
    # every residue reach the head; it is a falsification arm, not a default.
    compat_readout: str = "cls"

    # Graded-affinity supervision. False (default) builds NO strength head, so a
    # default model draws ZERO extra init-RNG and every existing checkpoint and
    # run is byte-identical. When True a scalar regression head is added on top
    # of `joint_hidden`, trained against per-(dataset, affinity_type) strength
    # quantiles -- the corpus has far more graded measurements than clean
    # booleans, and the binary head throws all of that ordering away.
    use_strength_head: bool = False

    # Learned HCDR3-length posterior. Same conditional-construction contract as
    # the strength head: False (default) builds NO head and draws ZERO extra
    # init-RNG. When True, a categorical head over `length_head_max` classes sits
    # on `joint_hidden` and is queried on the COLLAPSED-SLOT encoding, so it can
    # never read the answer off the number of mask tokens.
    use_length_head: bool = False
    length_head_max: int = 32

    def validate(self) -> None:
        """
        Validate that the configuration is internally consistent.
        
        Raises ValueError if any hyperparameter is invalid or incompatible.
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if self.d_model <= 0: 
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0: 
            raise ValueError("n_heads must be > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.pad_token_id < 0 or self.pad_token_id >= self.vocab_size:
            raise ValueError("pad_token_id must be a valid token ID")
        if not isinstance(self.norm_first, bool):
            raise ValueError("norm_first must be a bool")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be either 'relu' (ReLU/Rectified Linear Unit) or 'gelu' (GELU/Gaussian Error Linear Unit)")
        if self.antigen_encoder_type not in {"scratch", "esm"}:
            raise ValueError("antigen_encoder_type must be either 'scratch' or 'esm'")
        if self.antigen_encoder_finetune not in {"frozen", "lora"}:
            raise ValueError("antigen_encoder_finetune must be either 'frozen' or 'lora'")
        if not (0 < self.antigen_max_length <= 1024):
            raise ValueError("antigen_max_length must be in (0, 1024]")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be > 0")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be > 0")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.compat_readout not in {"cls", "mean"}:
            raise ValueError("compat_readout must be either 'cls' or 'mean'")
        if not isinstance(self.use_strength_head, bool):
            raise ValueError("use_strength_head must be a bool")
        if not isinstance(self.use_length_head, bool):
            raise ValueError("use_length_head must be a bool")
        if self.length_head_max <= 0:
            raise ValueError("length_head_max must be > 0")

def length_to_class_index(length: int, length_head_max: int) -> int:
    """
    Map a 1-based HCDR3 length to a 0-based length-class index.

    This is the SINGLE definition of the length<->class mapping: a length ``L`` in
    ``1..length_head_max`` maps to class index ``L - 1``. It FAILS LOUD on an
    out-of-range length rather than silently clamping -- a clamp would assign a
    long HCDR3 the wrong class and corrupt the categorical head without any
    symptom. Callers that must never crash on a stray long span (the training
    collator) range-check first and mask the row out; callers that expect a valid
    length use this helper and get a hard error if ``length_head_max`` was
    mis-registered.
    """
    if not (1 <= length <= length_head_max):
        raise ValueError(
            f"HCDR3 length {length} is out of range for length_head_max="
            f"{length_head_max}; valid lengths are 1..{length_head_max}. "
            "Re-register length_head_max from scripts/length_census.py (no silent clamp)."
        )
    return length - 1


def class_index_to_length(class_index: int, length_head_max: int) -> int:
    """Inverse of ``length_to_class_index``: class index ``i`` -> length ``i + 1``."""
    if not (0 <= class_index < length_head_max):
        raise ValueError(
            f"class index {class_index} is out of range for length_head_max={length_head_max}"
        )
    return class_index + 1


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embedding layer.
    
    Uses an embedding table for token positions. Position index 0 is reserved for padding positions, and 
    real sequence positions start at 1. 
    
    As an example: 
        If attention_mask = [1, 1, 1, 0, 0]
        then position_ids = [1, 2, 3, 0, 0]

    """
    
    def __init__(self, max_length: int, d_model: int) -> None:
        """
        Initialize positional embedding table.

        Args:
            max_length (int): Maximum sequence length supported for real (non-pad) tokens.
            d_model (int): Embedding dimension
        """
        
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length + 1, d_model, padding_idx = 0)
    
    def forward(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask into learned positional embeddings.

        Args:
            attention_mask (torch.Tensor): Tensor of shape [batch_size, seq_len] with 1 for real teokens and 0 for paddding tokens

        Returns:
            torch.Tensor: Tensor of shape [batch_size, seq_len, d_model] containing positional embeddings.
            
        Raises ValueError if any effective position exceeds the configured max_length
        
        """
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch_size, seq_len]")
        
        # real tokens count upward from 1; pads remain 0
        position_ids = attention_mask.long().cumsum(dim = 1)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        
        max_pos = int(position_ids.max().item()) if position_ids.numel() > 0 else 0  # if the position_ids have more than 0 items
        if max_pos > self.max_length:
            raise ValueError(
                f"Sequence length {max_pos} exceeds configured max_length, which is equal to {self.max_length}"
            )
        return self.embedding(position_ids)


class TransformerSequenceEncoder(nn.Module):
    """
    Reusable transformer encoder stack for tokenized protein sequences.

    This keeps the antibody-only MLM path and the newer antibody-antigen path
    aligned on the same embedding / encoder implementation while letting the
    higher-level models decide how to fuse or decode the resulting features.
    """

    def __init__(self, config: MLMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = LearnedPositionalEmbedding(
            max_length=config.max_length,
            d_model=config.d_model,
        )
        self.embed_drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        # Required in pre-LN mode (the stack's output is un-normalized there);
        # near-redundant but harmless in post-LN mode, where the last layer
        # already ends in a LayerNorm. Kept unconditional so the parameter set
        # does not depend on norm_first.
        self.final_norm = nn.LayerNorm(config.d_model)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Validate model inputs and construct attention mask when omitted.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")

        _, seq_len = input_ids.shape
        if seq_len > self.config.max_length:
            raise ValueError(
                f"Input sequence length {seq_len} has to be less than the max length, equal to {self.config.max_length}"
            )

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        return attention_mask

    def _build_key_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a standard attention mask into a transformer key padding mask.
        """
        return attention_mask == 0

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the token + positional embeddings for the encoder stack.
        """
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(attention_mask)
        hidden = token_emb + pos_emb
        hidden = self.embed_drop(hidden)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one batch and return contextual hidden states plus the resolved mask.
        """
        attention_mask = self._validate_inputs(input_ids, attention_mask)
        hidden = self.embed(input_ids, attention_mask)
        key_padding_mask = self._build_key_padding_mask(attention_mask)
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.final_norm(hidden)
        return hidden, attention_mask


class AntibodyMLM(nn.Module):
    """
    Transformer-encoder MLM for antibody/nanobody sequences.
    
    Model expects tokenized sequences with optional chain tokens already inserted by the tokenizer/collator. 
    It returns per-position vocabulary logits (log-odds, to be converted to prob. distribution) suitable for masked language modelling. 
    
    Inputs:
        - input_ids: [batch_size, seq_len]
        - attention_mask: [batch_size, seq_len]
    
    Output:
        - logits: [batch_size, seq_len, vocab_size]
    """
    
    def __init__(self, config: MLMConfig) -> None: 
        super().__init__()
        config.validate()
        self.config = config
        self.sequence_encoder = TransformerSequenceEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False)
        self.pair_head = nn.Linear(config.d_model, 2)
        
        if config.tie_weights:
            self.lm_head.weight = self.sequence_encoder.token_embedding.weight
    
    def embed(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the input embeddings for the transformer

        Args:
            input_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing token IDs.
            
            attention_mask (torch.Tensor): Tensor of shape [batch_size, seq_len] indicating real vs pad tokens.

        Returns:
            torch.Tensor: Tensor of shape [batch_size, seq_len, d_model] containing the sum of token embeddings and positional embeddings, 
            followed by dropout.
        """
        return self.sequence_encoder.embed(input_ids, attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None  
    ) -> torch.Tensor:
        """
        Encode a batch of tokenized seqeunces into contextual hidden states.
        
        Args: 
            input_ids: Tensor of shape [batch_size, seq_len] containing token IDs.
            
            attention_masks: Optional tensor of shape [batch_size, seq_len]. If omitted, it is inferred from padding positions. 
            
        Returns: 
            torch.Tensor of shape [batch_size, seq_len, d_model] containing the contextual hidden states after the transformer encoder.
        """
        
        hidden, _ = self.sequence_encoder(input_ids, attention_mask)
        return hidden

    def pooled_cls(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: 
        """
        Return the contextual hidden state at the first token position. 
        
        Typically the [CLS] embedding if tokenizer prepends [CLS].

        Args:
            input_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing token IDs
            
            attention_mask (torch.Tensor | None, optional): Optional tensor of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Tensor of shape [batch_size, d_model] containing the first-token embedding for each sequence.
        """
        hidden = self.encode(input_ids, attention_mask)
        return hidden[:, 0, :]

    def predict_pairing(
        self,
        cls_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict whether each heavy/light combination is native or shuffled.

        Args:
            cls_hidden:
                Tensor of shape [batch_size, d_model] representing the final
                contextual [CLS] hidden state for each example.

        Returns:
            Tensor of shape [batch_size, 2] containing pair-classification
            logits, where class 1 corresponds to a native/cognate pair and
            class 0 corresponds to a shuffled negative.
        """
        if cls_hidden.dim() != 2:
            raise ValueError("cls_hidden must have shape [batch_size, d_model]")
        return self.pair_head(cls_hidden)


    def forward_with_pairing(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one forward pass that returns both MLM and pairing logits.

        This is the main forward helper for paired training. We expose both
        heads from one shared encoder pass so the training loop can optimize the
        residue-recovery objective and the native-vs-shuffled compatibility
        objective together.

        Args:
            input_ids:
                Tensor of shape [batch_size, seq_len] containing token IDs.
            attention_mask:
                Optional tensor of shape [batch_size, seq_len].

        Returns:
            Tuple `(mlm_logits, pair_logits)` where:
                - `mlm_logits` has shape [batch_size, seq_len, vocab_size]
                - `pair_logits` has shape [batch_size, 2]
        """
        hidden = self.encode(input_ids, attention_mask)
        mlm_logits = self.lm_head(hidden)
        pair_logits = self.predict_pairing(hidden[:, 0, :])
        return mlm_logits, pair_logits

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: 
        """
        Run a forward pass and return MLM logits. 

        Args:
            input_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing token IDs.
            
            attention_mask (torch.Tensor | None, optional): Optional tensor of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Tensor of shape [batch_size, seq_len, vocab_size] containing per-position token logits for MLM prediction.
        """
        logits, _ = self.forward_with_pairing(input_ids, attention_mask)
        return logits

    def compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute masked language model cross-entropy loss. 

        Args:
            logits (torch.Tensor): Tensor of shape [batch_size, seq_len, vocab_size]
            
            labels (torch.Tensor): Tensor of shape [batch_size, seq_len] containing target token IDs at MLM positions
                and 'ignore_index' elsewhere
            
            ignore_index (int): Label value to ignore when computing loss. Defaults to -100.

        Returns:
            torch.Tensor: Scalar tensor containing the MLM loss
            
        Raises ValueErorr if logits/labels do not have compatible shapes.
        """
        
        if logits.dim() != 3: 
            raise ValueError("logits must have shape [batch_size, seq_len, vocab_size]")
        if labels.dim() != 2: 
            raise ValueError("labels must have shape [batch_size, seq_len]")
        if logits.shape[:2] != labels.shape:
            raise ValueError("logits and labels must agree on [batch_size, seq_len]")

        # When a batch has no supervised tokens (every label == ignore_index),
        # F.cross_entropy(reduction="mean") computes 0/0 = NaN, which poisons
        # every weight through backward(). Return a differentiable zero instead,
        # mirroring the zero-target guard in compute_pair_loss.
        if (labels != ignore_index).sum() == 0:
            return logits.sum() * 0.0

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index = ignore_index
        )
        return loss

    def compute_pair_loss(
        self,
        pair_logits: torch.Tensor,
        pair_labels: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute native-vs-shuffled pair classification loss.

        Args:
            pair_logits:
                Tensor of shape [batch_size, 2] containing compatibility logits.
            pair_labels:
                Tensor of shape [batch_size] containing integer class labels.
            pair_mask:
                Optional boolean tensor of shape [batch_size] indicating which
                examples should contribute to the auxiliary loss. This lets the
                same code path handle batches that contain single-chain examples
                or mixed data where some rows are not true paired records.

        Returns:
            Scalar tensor containing the pair-classification loss. Returns a
            detached zero-like tensor when there are no valid paired examples.
        """
        if pair_logits.dim() != 2 or pair_logits.size(-1) != 2:
            raise ValueError("pair_logits must have shape [batch_size, 2]")
        if pair_labels.dim() != 1:
            raise ValueError("pair_labels must have shape [batch_size]")
        if pair_logits.size(0) != pair_labels.size(0):
            raise ValueError("pair_logits and pair_labels must agree on batch size")

        if pair_mask is None:
            pair_mask = torch.ones_like(pair_labels, dtype=torch.bool)
        if pair_mask.dim() != 1 or pair_mask.size(0) != pair_labels.size(0):
            raise ValueError("pair_mask must have shape [batch_size]")

        if pair_mask.sum().item() == 0:
            return pair_logits.sum() * 0.0

        return F.cross_entropy(pair_logits[pair_mask], pair_labels[pair_mask])

    def compute_losses(
        self,
        mlm_logits: torch.Tensor,
        labels: torch.Tensor,
        pair_logits: torch.Tensor | None = None,
        pair_labels: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        ignore_index: int = -100,
        pair_loss_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the joint training loss for MLM plus optional pairing.

        Args:
            mlm_logits:
                Tensor of shape [batch_size, seq_len, vocab_size].
            labels:
                Tensor of shape [batch_size, seq_len] containing MLM targets.
            pair_logits:
                Optional tensor of shape [batch_size, 2] with compatibility
                logits from the auxiliary pair head.
            pair_labels:
                Optional tensor of shape [batch_size] containing native-vs-
                shuffled labels.
            pair_mask:
                Optional boolean tensor of shape [batch_size] selecting examples
                that should participate in the pair loss.
            ignore_index:
                Ignore label for MLM cross-entropy.
            pair_loss_weight:
                Non-negative scalar multiplier applied to the pair loss.

        Returns:
            Dictionary containing:
                - `loss`: total scalar loss used for optimization
                - `mlm_loss`: scalar MLM loss
                - `pair_loss`: scalar pair-classification loss
        """
        mlm_loss = self.compute_loss(mlm_logits, labels, ignore_index=ignore_index)
        if pair_logits is None or pair_labels is None:
            pair_loss = mlm_loss.detach() * 0.0
        else:
            pair_loss = self.compute_pair_loss(pair_logits, pair_labels, pair_mask)

        total_loss = mlm_loss + (pair_loss_weight * pair_loss)
        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "pair_loss": pair_loss,
        }


class AntibodyAntigenCrossAttention(nn.Module):
    """
    Dual-encoder antibody-antigen model with cross-attention fusion.

    The antibody branch remains the only branch decoded with the MLM head so
    later HCDR3-focused masking can stay antibody-centric, while the joint
    compatibility decision is made from cross-attended antibody/antigen
    representations.
    """

    def __init__(self, config: MLMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.antibody_encoder = TransformerSequenceEncoder(config)
        # Direction 1: the antigen stream is either the original from-scratch encoder
        # or a pretrained ESM-2 encoder. Both honor the same
        # (hidden[B, L, d_model], mask) contract, so nothing downstream changes. The
        # ESM import is lazy so the scratch path never requires the 'esm' extra.
        if config.antigen_encoder_type == "esm":
            from smallAntibodyGen.models.esm_antigen_encoder import ESMAntigenEncoder

            self.antigen_encoder = ESMAntigenEncoder(config)
        else:
            self.antigen_encoder = TransformerSequenceEncoder(config)

        self.antibody_to_antigen = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.antigen_to_antibody = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.fusion_norm_antibody = nn.LayerNorm(config.d_model)
        self.fusion_norm_antigen = nn.LayerNorm(config.d_model)
        # Pre-LN leaves the fused residual stream un-normalized, so the heads
        # need a terminal norm the way a pre-LN stack needs one before its head.
        # Registered ONLY in pre-LN mode: the post-LN state dict stays exactly
        # what it was, and a cross-mode checkpoint now fails the strict load as
        # well as the init-compat check.
        if config.norm_first:
            self.fusion_out_norm_antibody = nn.LayerNorm(config.d_model)
            self.fusion_out_norm_antigen = nn.LayerNorm(config.d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.compatibility_head = nn.Linear(config.d_model, 2)
        # Conditional construction, not a zeroed head: building it unconditionally
        # would consume init RNG and shift every subsequent parameter draw, so a
        # "default-off" run would silently stop being byte-identical.
        if config.use_strength_head:
            self.strength_head = nn.Linear(config.d_model, 1)
        # Same conditional-construction contract as the strength head: a default
        # (and a strength-head-only) model draws ZERO length-head init-RNG.
        if config.use_length_head:
            self.length_head = nn.Linear(config.d_model, config.length_head_max)

        if config.tie_weights:
            self.lm_head.weight = self.antibody_encoder.token_embedding.weight

    def encode_antibody(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.antibody_encoder(input_ids, attention_mask)

    def encode_antigen(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.antigen_encoder(input_ids, attention_mask)

    def fuse(
        self,
        antibody_hidden: torch.Tensor,
        antibody_attention_mask: torch.Tensor,
        antigen_hidden: torch.Tensor,
        antigen_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply symmetric cross-attention between antibody and antigen branches.

        Norm placement follows ``config.norm_first`` so this block matches the
        encoder stacks feeding it:

        - post-LN (default): ``LayerNorm(x + crossattn(x))``
        - pre-LN: ``fusion_out_norm(x + crossattn(LayerNorm(x)))`` — the residual
          path itself carries no norm, and the terminal norm exists so the heads
          still read a normalized representation.

        Both branches are normalized from their own pre-attention states, so the
        two streams stay symmetric.
        """
        if self.config.norm_first:
            antibody_query = self.fusion_norm_antibody(antibody_hidden)
            antigen_query = self.fusion_norm_antigen(antigen_hidden)
        else:
            antibody_query = antibody_hidden
            antigen_query = antigen_hidden

        antibody_ctx, _ = self.antibody_to_antigen(
            query=antibody_query,
            key=antigen_query,
            value=antigen_query,
            key_padding_mask=(antigen_attention_mask == 0),
            need_weights=False,
        )
        antigen_ctx, _ = self.antigen_to_antibody(
            query=antigen_query,
            key=antibody_query,
            value=antibody_query,
            key_padding_mask=(antibody_attention_mask == 0),
            need_weights=False,
        )

        if self.config.norm_first:
            antibody_hidden = self.fusion_out_norm_antibody(antibody_hidden + antibody_ctx)
            antigen_hidden = self.fusion_out_norm_antigen(antigen_hidden + antigen_ctx)
        else:
            antibody_hidden = self.fusion_norm_antibody(antibody_hidden + antibody_ctx)
            antigen_hidden = self.fusion_norm_antigen(antigen_hidden + antigen_ctx)
        return antibody_hidden, antigen_hidden

    def joint_representation(
        self,
        antibody_hidden: torch.Tensor,
        antigen_hidden: torch.Tensor,
        antibody_attention_mask: torch.Tensor | None = None,
        antigen_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build one fused example-level embedding feeding ``fusion_mlp``.

        ``config.compat_readout == "cls"`` (default) concatenates the two [CLS]
        states and ignores the masks — byte-for-byte the historical readout.
        ``"mean"`` mask-aware mean-pools each fused stream over its non-pad
        positions ([CLS] included) then concatenates; the masks are required only
        for this readout, which is why they are optional arguments.
        """
        if self.config.compat_readout == "cls":
            joint = torch.cat([antibody_hidden[:, 0, :], antigen_hidden[:, 0, :]], dim=-1)
            return self.fusion_mlp(joint)
        if antibody_attention_mask is None or antigen_attention_mask is None:
            raise ValueError("compat_readout='mean' requires both attention masks")
        pooled = []
        for hidden, mask in (
            (antibody_hidden, antibody_attention_mask),
            (antigen_hidden, antigen_attention_mask),
        ):
            m = mask.to(hidden.dtype)  # [B, L], 1 = real position ([CLS] included)
            # clamp is defensive only (the collators never emit an all-pad row).
            pooled.append(
                (hidden * m.unsqueeze(-1)).sum(dim=1)
                / m.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            )
        joint = torch.cat([pooled[0], pooled[1]], dim=-1)
        return self.fusion_mlp(joint)

    def predict_strength(self, joint_hidden: torch.Tensor) -> torch.Tensor:
        """
        Read the graded-strength scalar off ``joint_hidden`` (shape [batch_size]).

        Only valid when the model was built with ``use_strength_head=True``.
        """
        if getattr(self, "strength_head", None) is None:
            raise RuntimeError(
                "predict_strength requires use_strength_head=True; no strength_head "
                "was constructed for this model."
            )
        return self.strength_head(joint_hidden).squeeze(-1)

    def predict_length_logits(self, joint_hidden: torch.Tensor) -> torch.Tensor:
        """
        Read categorical length-class logits off ``joint_hidden``
        (shape ``[batch_size, length_head_max]``).

        Only valid when the model was built with ``use_length_head=True``. Class
        index ``i`` corresponds to HCDR3 length ``i + 1`` (see
        ``length_to_class_index``).
        """
        if getattr(self, "length_head", None) is None:
            raise RuntimeError(
                "predict_length_logits requires use_length_head=True; no length_head "
                "was constructed for this model."
            )
        return self.length_head(joint_hidden)

    def forward_length_query(
        self,
        antibody_input_ids: torch.Tensor,
        antibody_attention_mask: torch.Tensor | None,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Run one dual-stream forward on the COLLAPSED-SPAN length query and return
        length-class logits (shape ``[batch_size, length_head_max]``).

        The antibody stream here is the collapsed-span encoding -- the whole HCDR3
        interval replaced by exactly ONE ``[MASK]`` -- so the head can never read
        the true length off the mask count. That is the entire reason this is a
        separate forward rather than a second head on the ordinary one: on the
        ordinary encoding the number of masks IS the answer.

        The trainer only calls this when the length loss is active, so default
        runs do zero extra work. It reuses the exact encode/fuse/joint sequence
        ``forward`` uses, so the length query sees an identical fusion path to the
        compatibility head.
        """
        antibody_hidden, antibody_attention_mask = self.encode_antibody(
            antibody_input_ids,
            antibody_attention_mask,
        )
        antigen_hidden, antigen_attention_mask = self.encode_antigen(
            antigen_input_ids,
            antigen_attention_mask,
        )
        fused_antibody, fused_antigen = self.fuse(
            antibody_hidden,
            antibody_attention_mask,
            antigen_hidden,
            antigen_attention_mask,
        )
        joint_hidden = self.joint_representation(
            fused_antibody,
            fused_antigen,
            antibody_attention_mask,
            antigen_attention_mask,
        )
        return self.predict_length_logits(joint_hidden)

    def forward(
        self,
        antibody_input_ids: torch.Tensor,
        antibody_attention_mask: torch.Tensor | None,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor | None,
        return_strength: bool = False,
        *,
        antigen_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Return antibody MLM logits plus antibody-antigen compatibility logits.

        ``antigen_state`` is a keyword-only opt-in whose default keeps this
        method byte-identical to the pre-change code and adds zero init RNG.
        When ``None`` (default) the antigen stream is encoded here exactly as
        before. When provided it must be the pre-fuse
        ``(antigen_hidden, antigen_attention_mask)`` pair EXACTLY as returned by
        ``encode_antigen`` on THIS SAME model -- the caller owns computing and
        caching it. ``encode_antigen`` is the antigen-only prefix that ends
        precisely where ``fuse`` (the first cross-stream contact) begins, so
        supplying its output pre-computed is exactly equivalent to recomputing it
        here (deterministic eval-mode encoder, no RNG). When given,
        ``antigen_input_ids`` / ``antigen_attention_mask`` are ignored for the
        antigen stream, and the cached pair's batch dim MUST match the antibody
        stream's.

        This matters because the antigen is CONSTANT across every step of guided
        decoding, and on the ESM path re-encoding it per step is the dominant
        cost of generation.
        """
        antibody_hidden, antibody_attention_mask = self.encode_antibody(
            antibody_input_ids,
            antibody_attention_mask,
        )
        if antigen_state is None:
            antigen_hidden, antigen_attention_mask = self.encode_antigen(
                antigen_input_ids,
                antigen_attention_mask,
            )
        else:
            # Caller-supplied pre-fuse encoding (see docstring): skip
            # encode_antigen and use the cached pair. Byte-identical to the None
            # path because the eval-mode encoder is a deterministic, RNG-free
            # function of its inputs.
            antigen_hidden, antigen_attention_mask = antigen_state
        fused_antibody, fused_antigen = self.fuse(
            antibody_hidden,
            antibody_attention_mask,
            antigen_hidden,
            antigen_attention_mask,
        )
        mlm_logits = self.lm_head(fused_antibody)
        joint_hidden = self.joint_representation(
            fused_antibody,
            fused_antigen,
            antibody_attention_mask,
            antigen_attention_mask,
        )
        compatibility_logits = self.compatibility_head(joint_hidden)
        if return_strength:
            # 3-tuple ONLY on explicit opt-in, so every historical caller keeps
            # unpacking exactly two values.
            strength_predictions = (
                self.predict_strength(joint_hidden)
                if getattr(self, "strength_head", None) is not None
                else None
            )
            return mlm_logits, compatibility_logits, strength_predictions
        return mlm_logits, compatibility_logits

    def compute_mlm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError("logits must have shape [batch_size, seq_len, vocab_size]")
        if labels.dim() != 2:
            raise ValueError("labels must have shape [batch_size, seq_len]")
        if logits.shape[:2] != labels.shape:
            raise ValueError("logits and labels must agree on [batch_size, seq_len]")

        # Guard the all-ignored batch (0/0 = NaN); see AntibodyMLM.compute_loss.
        if (labels != ignore_index).sum() == 0:
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )

    def compute_compatibility_loss(
        self,
        compatibility_logits: torch.Tensor,
        compatibility_labels: torch.Tensor,
        compatibility_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if compatibility_logits.dim() != 2 or compatibility_logits.size(-1) != 2:
            raise ValueError("compatibility_logits must have shape [batch_size, 2]")
        if compatibility_labels.dim() != 1:
            raise ValueError("compatibility_labels must have shape [batch_size]")
        if compatibility_logits.size(0) != compatibility_labels.size(0):
            raise ValueError("compatibility_logits and compatibility_labels must agree on batch size")

        if compatibility_mask is None:
            compatibility_mask = torch.ones_like(compatibility_labels, dtype=torch.bool)
        if compatibility_mask.dim() != 1 or compatibility_mask.size(0) != compatibility_labels.size(0):
            raise ValueError("compatibility_mask must have shape [batch_size]")
        if compatibility_mask.sum().item() == 0:
            return compatibility_logits.sum() * 0.0

        return F.cross_entropy(
            compatibility_logits[compatibility_mask],
            compatibility_labels[compatibility_mask],
        )

    def compute_strength_loss(
        self,
        strength_predictions: torch.Tensor,
        strength_targets: torch.Tensor,
        strength_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Masked MSE between predicted and target graded-strength quantiles.

        Mirrors ``compute_compatibility_loss``: an all-masked-out batch returns a
        *differentiable* zero (``strength_predictions.sum() * 0.0``) so the empty
        case does not produce 0/0 = NaN and does not detach the graph.
        """
        if strength_predictions.dim() != 1:
            raise ValueError("strength_predictions must have shape [batch_size]")
        if strength_targets.dim() != 1:
            raise ValueError("strength_targets must have shape [batch_size]")
        if strength_predictions.size(0) != strength_targets.size(0):
            raise ValueError(
                "strength_predictions and strength_targets must agree on batch size"
            )

        if strength_mask is None:
            strength_mask = torch.ones_like(strength_targets, dtype=torch.bool)
        if strength_mask.dim() != 1 or strength_mask.size(0) != strength_targets.size(0):
            raise ValueError("strength_mask must have shape [batch_size]")
        if strength_mask.sum().item() == 0:
            return strength_predictions.sum() * 0.0

        return F.mse_loss(
            strength_predictions[strength_mask],
            strength_targets[strength_mask],
        )

    def compute_length_loss(
        self,
        length_logits: torch.Tensor,
        length_labels: torch.Tensor,
        length_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Masked cross-entropy over length classes for the learned length head.

        ``length_logits`` are ``[batch_size, length_head_max]`` categorical logits
        from ``forward_length_query``; ``length_labels`` are ``[batch_size]``
        0-based class indices (length L -> index L-1, see
        ``length_to_class_index``); ``length_mask`` is ``[batch_size]`` marking the
        rows with a usable (valid-span, in-range, non-shuffled) length label.

        Mirrors ``compute_compatibility_loss``: an all-masked-out batch returns a
        *differentiable* zero so the empty case does not produce 0/0 = NaN and does
        not detach the graph.
        """
        if length_logits.dim() != 2:
            raise ValueError("length_logits must have shape [batch_size, num_length_classes]")
        if length_labels.dim() != 1:
            raise ValueError("length_labels must have shape [batch_size]")
        if length_logits.size(0) != length_labels.size(0):
            raise ValueError("length_logits and length_labels must agree on batch size")

        if length_mask is None:
            length_mask = torch.ones_like(length_labels, dtype=torch.bool)
        if length_mask.dim() != 1 or length_mask.size(0) != length_labels.size(0):
            raise ValueError("length_mask must have shape [batch_size]")
        if length_mask.sum().item() == 0:
            return length_logits.sum() * 0.0

        return F.cross_entropy(
            length_logits[length_mask],
            length_labels[length_mask],
        )

    def compute_losses(
        self,
        mlm_logits: torch.Tensor,
        labels: torch.Tensor,
        compatibility_logits: torch.Tensor | None = None,
        compatibility_labels: torch.Tensor | None = None,
        compatibility_mask: torch.Tensor | None = None,
        strength_predictions: torch.Tensor | None = None,
        strength_targets: torch.Tensor | None = None,
        strength_mask: torch.Tensor | None = None,
        length_logits: torch.Tensor | None = None,
        length_labels: torch.Tensor | None = None,
        length_mask: torch.Tensor | None = None,
        ignore_index: int = -100,
        mlm_loss_weight: float = 1.0,
        compatibility_loss_weight: float = 1.0,
        strength_loss_weight: float = 0.0,
        length_loss_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the antibody MLM loss plus optional compatibility loss.

        ``mlm_loss_weight`` defaults to 1.0 (multiplication by exactly 1.0 is
        bit-exact, so the historical total is unchanged); the returned
        ``"mlm_loss"`` entry is always the unweighted value so reported curves
        stay comparable across weights.
        """
        mlm_loss = self.compute_mlm_loss(mlm_logits, labels, ignore_index=ignore_index)
        if compatibility_logits is None or compatibility_labels is None:
            compatibility_loss = mlm_loss.detach() * 0.0
        else:
            compatibility_loss = self.compute_compatibility_loss(
                compatibility_logits,
                compatibility_labels,
                compatibility_mask,
            )

        if strength_predictions is None or strength_targets is None:
            # Differentiable zero, so the weighted term never detaches the graph.
            strength_loss = mlm_loss.detach() * 0.0
        else:
            strength_loss = self.compute_strength_loss(
                strength_predictions,
                strength_targets,
                strength_mask,
            )

        if length_logits is None or length_labels is None:
            length_loss = mlm_loss.detach() * 0.0
        else:
            length_loss = self.compute_length_loss(
                length_logits,
                length_labels,
                length_mask,
            )

        total_loss = (
            (mlm_loss_weight * mlm_loss)
            + (compatibility_loss_weight * compatibility_loss)
            + (strength_loss_weight * strength_loss)
            + (length_loss_weight * length_loss)
        )
        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "compatibility_loss": compatibility_loss,
            "strength_loss": strength_loss,
            "length_loss": length_loss,
        }

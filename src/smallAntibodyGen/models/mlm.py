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
            raise ValueError("d_ff must be in [0, 1)")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.pad_token_id < 0 or self.pad_token_id >= self.vocab_size:
            raise ValueError("pad_token_id must be a valid token ID")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be either 'relu' (ReLU/Rectified Linear Unit) or 'gelu' (GELU/Gaussian Error Linear Unit)")
    
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
        
        self.token_embedding = nn.Embedding(
            num_embeddings = config.vocab_size,
            embedding_dim = config.d_model,
            padding_idx = config.pad_token_id
        )
        
        self.position_embedding = LearnedPositionalEmbedding(
            max_length = config.max_length, 
            d_model = config.d_model
        )
        
        self.embed_drop = nn.Dropout(config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.d_model,
            nhead = config.n_heads,
            dim_feedforward = config.d_ff,
            dropout = config.dropout, 
            activation = config.activation,
            batch_first = True, 
            norm_first = False
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers = config.n_layers,
            enable_nested_tensor = False
        )
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False)
        self.pair_head = nn.Linear(config.d_model, 2)
        
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
    
    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None 
    ) -> torch.Tensor:
        
        """
        Validate model inputs and construct attention mask (if needed).

        Args:
            input_ids (torch.Tensor): Tensor of token IDs with shape [batch_size, seq_len]
            
            attention_mask (torch.Tensor | None): Optional tensor with shape [batch_size, seq_len]. If None, the mask is 
                inferred from input_ids != pad_token_id

        Returns:
            torch.Tensor: Valid attention_mask tensor with shape [batch_size, seq_len]
            
        Raises ValueError if shapes are wrong/sequence length exceeds max_length
        """
        if input_ids.dim() != 2: 
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_length:
            raise ValueError(
                f"Input sequence length {seq_len} has to be less than the max length, equal to {self.config.max_length}"
            )
        
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
        
        return attention_mask
        
    def _build_key_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a standard attention mask into a transformer key padding mask.

        Args:
            attention_mask (torch.Tensor): Tensor of shape [batch_size, seq_len] with 1 for real tokens and 0 for padding

        Returns:
            torch.Tensor: Boolean tensor of shape [batch_size, seq_len] where True marks padding positions to be ignored 
            by the transformer.
        """
        return attention_mask == 0

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
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(attention_mask)
        hidden = token_emb + pos_emb
        hidden = self.embed_drop(hidden)
        return hidden

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
        
        attention_mask = self._validate_inputs(input_ids, attention_mask)
        hidden = self.embed(input_ids, attention_mask)
        key_padding_mask = self._build_key_padding_mask(attention_mask)
        hidden = self.encoder(hidden, src_key_padding_mask = key_padding_mask)
        hidden = self.final_norm(hidden)
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

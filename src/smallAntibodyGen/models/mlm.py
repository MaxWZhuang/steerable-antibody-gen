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
        if self.d_model % self.heads != 0:
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
                dropout = config.dsropout, 
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
            
            if config.tie_weights:
                self.lm_head.weight = self.token_embedding.weight
"""Configuration class for the Transformer model."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration class to store the configuration of a Transformer model.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        max_seq_len (int): Maximum sequence length
        d_model (int): The dimension of the model (embedding dimension)
        num_heads (int): Number of attention heads
        num_layers (int): Number of layers in encoder and decoder
        d_ff (int): Dimension of the feed forward network
        dropout (float): Dropout rate
        pad_token_id (int): Token ID used for padding
        bos_token_id (int): Beginning of sequence token ID
        eos_token_id (int): End of sequence token ID
        label_smoothing (float): Label smoothing factor
        max_lr (float): Maximum learning rate for scheduler
        warmup_steps (int): Number of warmup steps for scheduler
    """
    
    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    label_smoothing: float = 0.1
    max_lr: float = 1e-3
    warmup_steps: int = 4000
    
    def __post_init__(self):
        """Verify the configuration is valid."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert self.label_smoothing >= 0 and self.label_smoothing < 1, \
            "label_smoothing must be between 0 and 1"
    
    @property
    def d_k(self) -> int:
        """Get the dimension of keys/queries in attention."""
        return self.d_model // self.num_heads
    
    @property
    def d_v(self) -> int:
        """Get the dimension of values in attention."""
        return self.d_model // self.num_heads 
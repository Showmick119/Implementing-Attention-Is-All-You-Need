"""Embedding layers for the Transformer model."""
import torch
import torch.nn as nn
import math
from .utils import positional_encoding


class TokenEmbedding(nn.Module):
    """Token embedding layer with scaled outputs."""
    
    def __init__(self, vocab_size: int, d_model: int):
        """Initialize the embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding layer.
        
        The embeddings are scaled by sqrt(d_model) as per the paper.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Positional encoding layer using sinusoidal functions."""
    
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """Initialize the positional encoding layer.
        
        Args:
            d_model: Dimension of the model
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant positional encoding matrix
        # (1, max_seq_len, d_model)
        self.pe = positional_encoding(max_seq_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the positional encoding layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x) 
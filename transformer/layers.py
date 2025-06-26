"""Core layers of the Transformer model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .utils import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config):
        """Initialize the multi-head attention layer.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        
        assert self.d_model % self.num_heads == 0
        
        self.depth = self.d_model // self.num_heads
        
        self.wq = nn.Linear(config.d_model, config.d_model)
        self.wk = nn.Linear(config.d_model, config.d_model)
        self.wv = nn.Linear(config.d_model, config.d_model)
        
        self.dense = nn.Linear(config.d_model, config.d_model)
        
    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split the last dimension into (num_heads, depth).
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # (batch_size, seq_len, num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # (batch_size, num_heads, seq_len, depth)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the multi-head attention layer.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len_q, d_model)
                - Attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)
        
        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # (batch_size, num_heads, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config):
        """Initialize the feed-forward network.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        
        self.ff1 = nn.Linear(config.d_model, config.d_ff)
        self.ff2 = nn.Linear(config.d_ff, config.d_model)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # (batch_size, seq_len, d_ff)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # (batch_size, seq_len, d_model)
        x = self.ff2(x)
        
        return x


class ResidualConnection(nn.Module):
    """Residual connection with layer normalization."""
    
    def __init__(self, config):
        """Initialize the residual connection.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual connection.
        
        Args:
            x: Input tensor
            sublayer_output: Output tensor from sublayer
            
        Returns:
            Output tensor after residual connection and normalization
        """
        return x + self.dropout(sublayer_output) 
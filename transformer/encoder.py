"""Encoder implementation for the Transformer model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .layers import MultiHeadAttention, FeedForwardNetwork, ResidualConnection


class EncoderBlock(nn.Module):
    """Single encoder block of the Transformer."""
    
    def __init__(self, config):
        """Initialize the encoder block.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        
        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)
        
        self.residual1 = ResidualConnection(config)
        self.residual2 = ResidualConnection(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Multi-head attention
        attn_output, attn_weights = self.mha(x, x, x, mask)
        # First residual connection
        out1 = self.residual1(x, attn_output)
        # Layer normalization 1
        out1 = self.norm1(out1)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        # Second residual connection
        out2 = self.residual2(out1, ffn_output)
        # Layer normalization 2
        out2 = self.norm2(out2)
        
        return out2, attn_weights


class Encoder(nn.Module):
    """Full encoder stack of the Transformer."""
    
    def __init__(self, config):
        """Initialize the encoder.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        
        self.num_layers = config.num_layers
        
        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(self.num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights from all layers of shape
                  (batch_size, num_layers, num_heads, seq_len, seq_len)
        """
        attention_weights = []
        
        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, mask)
            attention_weights.append(attn_weights)
        
        # Stack attention weights from all layers
        attention_weights = torch.stack(attention_weights, dim=1)
        
        return x, attention_weights 
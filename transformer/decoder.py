"""Decoder implementation for the Transformer model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .layers import MultiHeadAttention, FeedForwardNetwork, ResidualConnection


class DecoderBlock(nn.Module):
    """Single decoder block of the Transformer."""
    
    def __init__(self, config):
        """Initialize the decoder block.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        
        self.masked_mha = MultiHeadAttention(config)
        self.cross_mha = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)
        
        self.residual1 = ResidualConnection(config)
        self.residual2 = ResidualConnection(config)
        self.residual3 = ResidualConnection(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the decoder block.
        
        Args:
            x: Input tensor of shape (batch_size, target_seq_len, d_model)
            enc_output: Encoder output tensor of shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Optional mask for the target sequence
            padding_mask: Optional mask for the encoder outputs
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, target_seq_len, d_model)
                - Self attention weights of shape (batch_size, num_heads, target_seq_len, target_seq_len)
                - Cross attention weights of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # Masked self-attention
        attn1, attn_weights_block1 = self.masked_mha(x, x, x, look_ahead_mask)
        out1 = self.residual1(x, attn1)
        out1 = self.norm1(out1)
        
        # Cross-attention
        attn2, attn_weights_block2 = self.cross_mha(
            out1, enc_output, enc_output, padding_mask)
        out2 = self.residual2(out1, attn2)
        out2 = self.norm2(out2)
        
        # Feed-forward network
        ffn_output = self.ffn(out2)
        out3 = self.residual3(out2, ffn_output)
        out3 = self.norm3(out3)
        
        return out3, attn_weights_block1, attn_weights_block2


class Decoder(nn.Module):
    """Full decoder stack of the Transformer."""
    
    def __init__(self, config):
        """Initialize the decoder.
        
        Args:
            config: TransformerConfig instance
        """
        super().__init__()
        
        self.num_layers = config.num_layers
        
        self.dec_layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(self.num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass of the decoder.
        
        Args:
            x: Input tensor of shape (batch_size, target_seq_len, d_model)
            enc_output: Encoder output tensor of shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Optional mask for the target sequence
            padding_mask: Optional mask for the encoder outputs
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, target_seq_len, d_model)
                - List of self attention weights from each layer
                - List of cross attention weights from each layer
        """
        attention_weights = {
            'self_attention': [],
            'cross_attention': []
        }
        
        for i in range(self.num_layers):
            x, self_attn, cross_attn = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask)
            
            attention_weights['self_attention'].append(self_attn)
            attention_weights['cross_attention'].append(cross_attn)
        
        # Stack attention weights from all layers
        attention_weights['self_attention'] = torch.stack(
            attention_weights['self_attention'], dim=1)
        attention_weights['cross_attention'] = torch.stack(
            attention_weights['cross_attention'], dim=1)
        
        return x, attention_weights 
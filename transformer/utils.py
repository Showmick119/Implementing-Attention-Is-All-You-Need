"""Utility functions for the Transformer model."""
import torch
import numpy as np
from typing import Optional, Tuple


def create_padding_mask(seq: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create a padding mask for the input sequence.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_token_id: Token ID used for padding
        
    Returns:
        Padding mask of shape (batch_size, 1, 1, seq_len)
        1s indicate positions to mask, 0s indicate positions to attend to
    """
    # (batch_size, 1, 1, seq_len)
    return (seq == pad_token_id).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """Create a look-ahead mask to prevent attending to future tokens.
    
    Args:
        size: Size of the sequence
        
    Returns:
        Look-ahead mask of shape (1, 1, size, size)
        1s indicate positions to mask, 0s indicate positions to attend to
    """
    # (size, size)
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # (1, 1, size, size)
    return mask.unsqueeze(0).unsqueeze(1)


def get_angles(pos: torch.Tensor, i: torch.Tensor, d_model: int) -> torch.Tensor:
    """Calculate angles for positional encoding.
    
    Args:
        pos: Position indices
        i: Dimension indices
        d_model: Model dimension
        
    Returns:
        Angle rates for positional encoding
    """
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    return pos * angle_rates


def positional_encoding(max_position: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in the paper.
    
    Args:
        max_position: Maximum sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding of shape (1, max_position, d_model)
    """
    angle_rads = get_angles(
        pos=torch.arange(max_position).unsqueeze(1),
        i=torch.arange(d_model).unsqueeze(0),
        d_model=d_model
    )
    
    # Apply sin to even indices in the array
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads.unsqueeze(0)
    
    return pos_encoding


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate scaled dot product attention.
    
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
        
    Returns:
        Tuple of:
            - Output tensor of shape (..., seq_len_q, d_v)
            - Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    # (..., seq_len_q, seq_len_k)
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    
    # Scale matmul_qk
    d_k = q.size(-1)
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # Add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask, float('-inf'))
    
    # (..., seq_len_q, seq_len_k)
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    
    # (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights


def create_combined_mask(
    tgt: torch.Tensor,
    tgt_pad_token_id: int
) -> torch.Tensor:
    """Create combined mask for decoder (combines padding and look-ahead).
    
    Args:
        tgt: Target sequence tensor of shape (batch_size, tgt_len)
        tgt_pad_token_id: Token ID used for padding
        
    Returns:
        Combined mask of shape (batch_size, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt.size(1)
    
    # (batch_size, 1, 1, tgt_len)
    tgt_pad_mask = create_padding_mask(tgt, tgt_pad_token_id)
    # (1, 1, tgt_len, tgt_len)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len)
    
    # (batch_size, 1, tgt_len, tgt_len)
    combined_mask = torch.max(tgt_pad_mask, tgt_look_ahead_mask)
    return combined_mask 
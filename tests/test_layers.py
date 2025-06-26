"""Tests for the Transformer layers."""
import torch
import pytest
from transformer.config import TransformerConfig
from transformer.layers import MultiHeadAttention, FeedForwardNetwork
from transformer.utils import create_padding_mask, create_look_ahead_mask


@pytest.fixture
def config():
    """Create a test configuration."""
    return TransformerConfig(
        vocab_size=100,
        max_seq_len=10,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.1
    )


def test_multi_head_attention(config):
    """Test the multi-head attention layer."""
    batch_size = 2
    seq_len = 8
    
    # Create inputs
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Create attention layer
    mha = MultiHeadAttention(config)
    
    # Test self-attention
    output, attention = mha(x, x, x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.d_model)
    assert attention.shape == (batch_size, config.num_heads, seq_len, seq_len)
    
    # Test with mask
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    output_masked, attention_masked = mha(x, x, x, mask)
    
    assert output_masked.shape == (batch_size, seq_len, config.d_model)
    assert attention_masked.shape == (batch_size, config.num_heads, seq_len, seq_len)
    
    # Test that masked and unmasked outputs are different
    assert not torch.allclose(output, output_masked, atol=1e-6)


def test_feed_forward_network(config):
    """Test the feed-forward network layer."""
    batch_size = 2
    seq_len = 8
    
    # Create input
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Create FFN layer
    ffn = FeedForwardNetwork(config)
    
    # Test forward pass
    output = ffn(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_padding_mask():
    """Test the padding mask creation."""
    batch_size = 2
    seq_len = 8
    pad_token_id = 0
    
    # Create sequence with padding
    seq = torch.randint(1, 10, (batch_size, seq_len))
    seq[:, -2:] = pad_token_id
    
    # Create mask
    mask = create_padding_mask(seq, pad_token_id)
    
    # Check mask shape
    assert mask.shape == (batch_size, 1, 1, seq_len)
    
    # Check that padding tokens are masked
    assert torch.all(mask[:, :, :, -2:] == 1)
    assert torch.all(mask[:, :, :, :-2] == 0)


def test_look_ahead_mask():
    """Test the look-ahead mask creation."""
    size = 8
    
    # Create mask
    mask = create_look_ahead_mask(size)
    
    # Check mask shape
    assert mask.shape == (1, 1, size, size)
    
    # Check that future positions are masked
    for i in range(size):
        for j in range(size):
            if j > i:
                assert mask[0, 0, i, j] == 1
            else:
                assert mask[0, 0, i, j] == 0 
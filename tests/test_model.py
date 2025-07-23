"""
Tests for Transformer model components.
"""

import pytest
import torch
import torch.nn as nn
from transformer.model import (
    InputEmbeddings, PositionalEncoding, LayerNormalization,
    FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection,
    EncoderBlock, Encoder, DecoderBlock, Decoder, ProjectionLayer,
    Transformer, build_transformer
)


@pytest.fixture
def device():
    """Return available device for testing."""
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def vocab_size():
    return 100


def test_input_embeddings(batch_size, seq_len, d_model, vocab_size, device):
    """Test InputEmbeddings layer."""
    embed = InputEmbeddings(d_model, vocab_size)
    
    # Create random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embed(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


def test_positional_encoding(batch_size, seq_len, d_model, device):
    """Test PositionalEncoding layer."""
    dropout = 0.1
    pos_enc = PositionalEncoding(d_model, seq_len, dropout)
    
    # Create random input embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pos_enc(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


def test_layer_normalization(batch_size, seq_len, d_model, device):
    """Test LayerNormalization."""
    layer_norm = LayerNormalization(d_model)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer_norm(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    
    # Check normalization properties (approximately)
    mean = output.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


def test_feed_forward_block(batch_size, seq_len, d_model, device):
    """Test FeedForwardBlock."""
    d_ff = 256
    dropout = 0.1
    ff_block = FeedForwardBlock(d_model, d_ff, dropout)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = ff_block(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


def test_multi_head_attention_block(batch_size, seq_len, d_model, device):
    """Test MultiHeadAttentionBlock."""
    h = 8
    dropout = 0.1
    
    assert d_model % h == 0, "d_model must be divisible by h for this test"
    
    attention = MultiHeadAttentionBlock(d_model, h, dropout)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = attention(x, x, x, mask=None)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


def test_multi_head_attention_with_mask(batch_size, seq_len, d_model, device):
    """Test MultiHeadAttentionBlock with mask."""
    h = 8
    dropout = 0.1
    attention = MultiHeadAttentionBlock(d_model, h, dropout)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create a simple mask (mask out last half of sequence)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, :, seq_len//2:] = 0
    
    output = attention(x, x, x, mask=mask)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


def test_build_transformer():
    """Test building a complete transformer model."""
    src_vocab_size = 100
    tgt_vocab_size = 150
    src_seq_len = 20
    tgt_seq_len = 25
    d_model = 64
    N = 2
    h = 8
    
    transformer = build_transformer(
        src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
        d_model=d_model, N=N, h=h
    )
    
    assert isinstance(transformer, Transformer)
    
    # Test model parameters are initialized
    param_count = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    assert param_count > 0
    
    # Test forward pass shapes
    batch_size = 2
    src_len = 15
    tgt_len = 10
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    src_mask = torch.ones(batch_size, 1, 1, src_len)
    tgt_mask = torch.tril(torch.ones(batch_size, 1, tgt_len, tgt_len))
    
    # Test encoding
    encoder_output = transformer.encode(src, src_mask)
    assert encoder_output.shape == (batch_size, src_len, d_model)
    
    # Test decoding
    decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
    assert decoder_output.shape == (batch_size, tgt_len, d_model)
    
    # Test projection
    projection_output = transformer.project(decoder_output)
    assert projection_output.shape == (batch_size, tgt_len, tgt_vocab_size)


def test_transformer_components_integration():
    """Test that all components work together properly."""
    d_model = 64
    h = 8
    dropout = 0.1
    d_ff = 256
    
    # Test encoder block
    self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(d_model, self_attention, feed_forward, dropout)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    
    output = encoder_block(x, mask)
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test decoder block
    self_attention_dec = MultiHeadAttentionBlock(d_model, h, dropout)
    cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_dec = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(d_model, self_attention_dec, cross_attention, feed_forward_dec, dropout)
    
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    tgt_mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len))
    
    dec_output = decoder_block(x, encoder_output, mask, tgt_mask)
    assert dec_output.shape == (batch_size, seq_len, d_model)

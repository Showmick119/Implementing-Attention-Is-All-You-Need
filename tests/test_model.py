"""
Super simple tests for model components.
"""

import torch
from transformer.model import InputEmbeddings, PositionalEncoding, build_transformer


def test_input_embeddings():
    """Test that embeddings work."""
    embed = InputEmbeddings(d_model=64, vocab_size=100)
    x = torch.tensor([[1, 2, 3]])
    output = embed(x)
    
    assert output.shape == (1, 3, 64)


def test_positional_encoding():
    """Test that positional encoding works."""
    pos_enc = PositionalEncoding(d_model=64, seq_len=10, dropout=0.1)
    x = torch.randn(1, 5, 64)
    output = pos_enc(x)
    
    assert output.shape == (1, 5, 64)


def test_build_transformer():
    """Test that we can build a transformer."""
    transformer = build_transformer(
        src_vocab_size=100,
        tgt_vocab_size=100, 
        src_seq_len=10,
        tgt_seq_len=10,
        d_model=64,
        N=1,
        h=8
    )
    
    # Just check it exists and has parameters
    param_count = sum(p.numel() for p in transformer.parameters())
    assert param_count > 0
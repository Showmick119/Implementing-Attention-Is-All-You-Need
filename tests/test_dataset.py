"""
Super simple tests for dataset.
"""

import torch
from data.dataset import causal_mask


def test_causal_mask():
    """Test that causal mask works."""
    mask = causal_mask(3)
    
    assert mask.shape == (1, 3, 3)
    assert mask[0, 0, 0] == True   # Can see itself
    assert mask[0, 0, 1] == False  # Can't see future
    assert mask[0, 1, 0] == True   # Can see past
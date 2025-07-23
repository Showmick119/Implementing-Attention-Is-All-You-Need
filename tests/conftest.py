"""
Simple pytest configuration.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
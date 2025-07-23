"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cpu')  # Always use CPU for tests to ensure consistency


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        'batch_size': 2,
        'num_epochs': 1,
        'lr': 1e-4,
        'seq_len': 10,
        'd_model': 64,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'test_weights',
        'model_basename': 'test_model_',
        'preload': None,
        'tokenizer_file': 'test_tokenizer_{0}.json',
        'experiment_name': 'test_run'
    }

"""
Super simple tests for configuration module.
"""

from config.config import get_config


def test_get_config():
    """Test that get_config returns a dictionary."""
    config = get_config()
    
    assert isinstance(config, dict)
    assert 'batch_size' in config
    assert 'd_model' in config
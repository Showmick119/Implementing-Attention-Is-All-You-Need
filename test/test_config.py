"""
Tests for configuration module.
"""

import pytest
from pathlib import Path
from config.config import get_config, get_weights_file_path, latest_weights_file_path


def test_get_config():
    """Test that get_config returns a valid configuration dictionary."""
    config = get_config()
    
    # Check required keys exist
    required_keys = [
        'batch_size', 'num_epochs', 'lr', 'seq_len', 'd_model',
        'lang_src', 'lang_tgt', 'model_folder', 'model_basename'
    ]
    
    for key in required_keys:
        assert key in config, f"Required key '{key}' missing from config"
    
    # Check types and values
    assert isinstance(config['batch_size'], int)
    assert config['batch_size'] > 0
    
    assert isinstance(config['num_epochs'], int)
    assert config['num_epochs'] > 0
    
    assert isinstance(config['lr'], (int, float))
    assert config['lr'] > 0
    
    assert isinstance(config['seq_len'], int)
    assert config['seq_len'] > 0
    
    assert isinstance(config['d_model'], int)
    assert config['d_model'] > 0


def test_get_weights_file_path():
    """Test weights file path generation."""
    config = get_config()
    epoch = "05"
    
    file_path = get_weights_file_path(config, epoch)
    
    assert isinstance(file_path, str)
    assert epoch in file_path
    assert config['model_basename'] in file_path
    assert file_path.endswith('.pt')


def test_latest_weights_file_path_no_files(tmp_path):
    """Test latest_weights_file_path when no weight files exist."""
    config = get_config()
    config['model_folder'] = str(tmp_path)
    
    result = latest_weights_file_path(config)
    assert result is None


def test_latest_weights_file_path_with_files(tmp_path):
    """Test latest_weights_file_path when weight files exist."""
    config = get_config()
    config['model_folder'] = str(tmp_path)
    config['model_basename'] = 'test_model_'
    
    # Create some fake weight files
    (tmp_path / 'test_model_01.pt').touch()
    (tmp_path / 'test_model_03.pt').touch()
    (tmp_path / 'test_model_02.pt').touch()
    
    result = latest_weights_file_path(config)
    
    assert result is not None
    assert 'test_model_03.pt' in result

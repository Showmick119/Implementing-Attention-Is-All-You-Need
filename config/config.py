"""
Configuration module for the Transformer implementation.

This module contains configuration dictionaries and utility functions for managing
model hyperparameters, file paths, and training settings.
"""

from pathlib import Path
from typing import Union

def get_config() -> dict:
    """
    Get the default configuration dictionary for the Transformer model.
    
    Returns:
        dict: Configuration dictionary containing:
            - batch_size (int): Number of samples per batch
            - num_epochs (int): Number of training epochs
            - lr (float): Learning rate for optimizer
            - seq_len (int): Maximum sequence length
            - d_model (int): Model dimension (embedding size)
            - lang_src (str): Source language code
            - lang_tgt (str): Target language code
            - model_folder (str): Directory to save model weights
            - model_basename (str): Prefix for model checkpoint files
            - preload (str): Path to pretrained model (None for training from scratch)
            - tokenizer_file (str): Template for tokenizer file names
            - experiment_name (str): Name for experiment tracking
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str) -> str:
    """
    Generate the file path for saving model weights.
    
    Args:
        config (dict): Configuration dictionary
        epoch (str): Epoch number or identifier
        
    Returns:
        str: Complete file path for the model weights
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config) -> Union[str, None]:
    """
    Find the path to the most recent model weights file.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        str or None: Path to the latest weights file, or None if no weights found
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
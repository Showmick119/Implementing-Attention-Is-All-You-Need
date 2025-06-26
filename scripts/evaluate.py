"""Evaluation script for the Transformer model."""
import argparse
import torch
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from transformer.config import TransformerConfig
from transformer.model import Transformer


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Transformer, TransformerConfig]:
    """Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of:
            - Loaded model
            - Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TransformerConfig(**checkpoint['config'])
    
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def generate_translations(
    model: Transformer,
    src_texts: List[str],
    config: TransformerConfig,
    device: torch.device,
    max_length: int = None,
    temperature: float = 1.0
) -> List[str]:
    """Generate translations for a list of source texts.
    
    Args:
        model: Transformer model
        src_texts: List of source texts to translate
        config: Model configuration
        device: Device to run inference on
        max_length: Maximum length of generated sequences
        temperature: Sampling temperature
        
    Returns:
        List of translated texts
    """
    # TODO: Implement proper tokenization
    # This is a placeholder that treats each character as a token
    # You should implement proper tokenization using sentencepiece or similar
    
    translations = []
    
    for text in src_texts:
        # Convert text to tensor (placeholder implementation)
        src = torch.tensor(
            [ord(c) % config.vocab_size for c in text],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        
        # Generate translation
        output, _ = model.generate(src, max_length, temperature)
        
        # Convert output tokens to text (placeholder implementation)
        translation = ''.join(
            chr(token.item()) for token in output[0]
            if token.item() not in [config.pad_token_id, config.eos_token_id]
        )
        
        translations.append(translation)
    
    return translations


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Transformer model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Input text or path to input file")
    parser.add_argument("--output", type=str,
                       help="Path to output file (if not specified, print to stdout)")
    parser.add_argument("--max-length", type=int,
                       help="Maximum length of generated sequences")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device)
    model, config = load_checkpoint(args.model, device)
    
    # Get input texts
    if Path(args.input).is_file():
        with open(args.input) as f:
            src_texts = f.readlines()
    else:
        src_texts = [args.input]
    
    # Generate translations
    translations = generate_translations(
        model,
        src_texts,
        config,
        device,
        args.max_length,
        args.temperature
    )
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            for translation in translations:
                f.write(translation + '\n')
    else:
        for src, tgt in zip(src_texts, translations):
            print(f"Source: {src.strip()}")
            print(f"Translation: {tgt}\n") 
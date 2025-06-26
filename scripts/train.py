"""Training script for the Transformer model."""
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from transformer.config import TransformerConfig
from transformer.model import Transformer
from transformer.loss import masked_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path) as f:
        return json.load(f)


def get_optimizer(
    model: nn.Module,
    config: TransformerConfig
) -> optim.Optimizer:
    """Create an Adam optimizer with warmup."""
    return optim.Adam(
        model.parameters(),
        lr=config.max_lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )


def get_scheduler(
    optimizer: optim.Optimizer,
    config: TransformerConfig
) -> optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with warmup."""
    def lr_lambda(step: int) -> float:
        # Linear warmup followed by inverse square root decay
        if step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        return float(config.warmup_steps ** 0.5) / float(step ** 0.5)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    config: TransformerConfig,
    device: torch.device
) -> float:
    """Train the model for one epoch.
    
    Args:
        model: Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Model configuration
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Create target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        logits, _ = model(src, tgt_input)
        
        # Calculate loss
        loss = masked_loss(
            logits,
            tgt_output,
            config.pad_token_id,
            config.label_smoothing
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train the Transformer model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on (cuda/cpu)")
    args = parser.parse_args()
    
    # Load config and create model
    config_dict = load_config(args.config)
    config = TransformerConfig(**config_dict)
    
    device = torch.device(args.device)
    model = Transformer(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # TODO: Create proper dataset and dataloader
    # This is a placeholder for demonstration
    # You should implement your own dataset class
    dataloader = DataLoader(
        [(torch.randint(0, config.vocab_size, (10,)),
          torch.randint(0, config.vocab_size, (10,)))
         for _ in range(100)],
        batch_size=args.batch_size
    )
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, config, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config_dict,
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main() 
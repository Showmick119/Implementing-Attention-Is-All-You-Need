"""Loss functions for the Transformer model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingLoss(nn.Module):
    """Implements label smoothing loss function.
    
    Label smoothing prevents the model from becoming overconfident in its predictions.
    Instead of using one-hot targets, it uses a mixture of the correct label and a
    uniform distribution over all labels.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean'
    ):
        """Initialize the label smoothing loss.
        
        Args:
            smoothing: Label smoothing factor between 0 and 1
            ignore_index: Index to ignore in the loss calculation (e.g., padding)
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the label smoothing loss.
        
        Args:
            logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
            target: Target indices of shape (batch_size, seq_len)
            
        Returns:
            Label smoothing loss
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Create a mask for padding tokens if ignore_index is specified
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            n_tokens = mask.sum()
        else:
            mask = torch.ones_like(target, dtype=torch.float)
            n_tokens = batch_size * seq_len
        
        # Create smoothed targets
        target = target.view(-1)
        logits = logits.view(-1, vocab_size)
        
        # Convert targets to one-hot representation
        one_hot = torch.zeros_like(logits).scatter(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed_targets = one_hot * (1 - self.smoothing) + \
            self.smoothing / vocab_size
        
        # Calculate loss using KL divergence
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1)
        
        # Apply mask and reduction
        loss = loss.view(batch_size, seq_len) * mask
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # mean
            return loss.sum() / n_tokens


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_token_id: int,
    label_smoothing: float = 0.1
) -> torch.Tensor:
    """Calculate the masked loss for the transformer output.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        target: Target indices of shape (batch_size, seq_len)
        pad_token_id: Token ID used for padding
        label_smoothing: Label smoothing factor
        
    Returns:
        Masked loss value
    """
    loss_fn = LabelSmoothingLoss(
        smoothing=label_smoothing,
        ignore_index=pad_token_id,
        reduction='mean'
    )
    return loss_fn(logits, target) 
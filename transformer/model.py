"""Full Transformer model implementation."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple
from .config import TransformerConfig
from .embeddings import TokenEmbedding, PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
from .utils import create_padding_mask, create_combined_mask


class Transformer(nn.Module):
    """Implementation of the Transformer model from "Attention Is All You Need"."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize the Transformer model.
        
        Args:
            config: Configuration instance
        """
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.encoder_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.decoder_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout
        )
        
        # Encoder and Decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Final linear layer
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the source sequence.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            src_mask: Optional mask for the source sequence
            
        Returns:
            Tuple of:
                - Encoder output tensor of shape (batch_size, src_len, d_model)
                - Encoder attention weights
        """
        # (batch_size, src_len, d_model)
        src_emb = self.encoder_embedding(src)
        src_emb = self.positional_encoding(src_emb)
        
        return self.encoder(src_emb, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decode the target sequence.
        
        Args:
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            memory: Encoder output tensor of shape (batch_size, src_len, d_model)
            tgt_mask: Optional mask for the target sequence
            memory_mask: Optional mask for the encoder outputs
            
        Returns:
            Tuple of:
                - Decoder output tensor of shape (batch_size, tgt_len, d_model)
                - Dictionary containing attention weights
        """
        # (batch_size, tgt_len, d_model)
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        return self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            
        Returns:
            Tuple of:
                - Output logits of shape (batch_size, tgt_len, vocab_size)
                - Dictionary containing attention weights
        """
        # Create masks
        src_mask = create_padding_mask(src, self.config.pad_token_id)
        tgt_mask = create_combined_mask(tgt, self.config.pad_token_id)
        
        # Encoding
        enc_output, enc_attention = self.encode(src, src_mask)
        
        # Decoding
        dec_output, dec_attention = self.decode(
            tgt, enc_output, tgt_mask, src_mask)
        
        # Final linear layer
        logits = self.final_layer(dec_output)
        
        # Combine attention weights
        attention_weights = {
            'encoder_attention': enc_attention,
            'decoder_self_attention': dec_attention['self_attention'],
            'decoder_cross_attention': dec_attention['cross_attention']
        }
        
        return logits, attention_weights
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate a sequence using greedy decoding.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            max_length: Maximum length of the generated sequence
            temperature: Sampling temperature (1.0 means greedy)
            
        Returns:
            Tuple of:
                - Generated sequence tensor of shape (batch_size, max_length)
                - Dictionary containing attention weights
        """
        if max_length is None:
            max_length = self.config.max_seq_len
            
        batch_size = src.size(0)
        device = src.device
        
        # Initialize target sequence with start token
        tgt = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Encode source sequence
        enc_output, enc_attention = self.encode(
            src, create_padding_mask(src, self.config.pad_token_id))
        
        attention_weights = {
            'encoder_attention': enc_attention,
            'decoder_self_attention': [],
            'decoder_cross_attention': []
        }
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Get model predictions
            dec_output, dec_attention = self.decode(
                tgt,
                enc_output,
                create_combined_mask(tgt, self.config.pad_token_id),
                create_padding_mask(src, self.config.pad_token_id)
            )
            
            # Get the next token probabilities
            logits = self.final_layer(dec_output[:, -1:])
            if temperature != 1.0:
                logits = logits / temperature
            next_token = torch.argmax(logits, dim=-1)
            
            # Append the next token
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Store attention weights
            attention_weights['decoder_self_attention'].append(
                dec_attention['self_attention'])
            attention_weights['decoder_cross_attention'].append(
                dec_attention['cross_attention'])
            
            # Break if end token is generated
            if next_token.item() == self.config.eos_token_id:
                break
        
        # Stack decoder attention weights
        attention_weights['decoder_self_attention'] = torch.stack(
            attention_weights['decoder_self_attention'], dim=1)
        attention_weights['decoder_cross_attention'] = torch.stack(
            attention_weights['decoder_cross_attention'], dim=1)
        
        return tgt, attention_weights 
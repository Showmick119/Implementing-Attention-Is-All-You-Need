"""
Dataset module for bilingual translation tasks.

This module contains the BilingualDataset class for handling parallel text
and preparing data for the Transformer model. It includes tokenization, padding,
and mask generation for both encoder and decoder sequences.

The dataset handles English-Italian translation pairs with special token
handling (SOS, EOS, PAD) and creates the necessary attention masks for training.
"""

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """
    Dataset class for bilingual translation tasks.
    
    This class processes parallel text data for sequence-to-sequence translation,
    handling tokenization, padding, and mask generation for both source and target
    sequences.
    
    Args:
        ds: Raw dataset containing translation pairs
        tokenizer_src: Tokenizer for source language
        tokenizer_tgt: Tokenizer for target language  
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        seq_len (int): Maximum sequence length
    """
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # Convert special tokens to tensors for efficient processing
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - encoder_input: Tokenized source sequence with special tokens
                - decoder_input: Tokenized target sequence for decoder input
                - encoder_mask: Padding mask for encoder attention
                - decoder_mask: Combined padding and causal mask for decoder
                - label: Target sequence for loss calculation
                - src_text: Original source text
                - tgt_text: Original target text
                
        Raises:
            ValueError: If the sentence is too long for the specified seq_len
        """
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize the source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate padding requirements
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS

        # Check if sequences are too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long!')
        
        # Build encoder input: [SOS] + tokens + [EOS] + padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Build decoder input: [SOS] + tokens + padding (no EOS for input)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Build label: tokens + [EOS] + padding (for loss calculation)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Ensure all sequences have the correct length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text 
        }

def causal_mask(size):
    """
    Create a causal (look-ahead) mask for decoder self-attention.
    
    This mask prevents the decoder from attending to future positions during
    training, ensuring that predictions for position i can only depend on
    positions less than i.
    
    Args:
        size (int): Size of the sequence (sequence length)
        
    Returns:
        torch.Tensor: Boolean mask of shape (1, size, size) where True indicates
                     positions that can be attended to, False indicates masked positions
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
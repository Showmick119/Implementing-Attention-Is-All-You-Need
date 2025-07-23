"""
Tests for dataset module.
"""

import pytest
import torch
from data.dataset import BilingualDataset, causal_mask


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.token_to_id_map = {
            '[SOS]': 0,
            '[EOS]': 1,
            '[PAD]': 2,
            '[UNK]': 3
        }
        
    def encode(self, text):
        """Mock encode method."""
        class MockEncoding:
            def __init__(self, text_len):
                # Simple mock: each word becomes one token
                self.ids = list(range(4, min(4 + text_len, self.parent.vocab_size)))
                
        encoding = MockEncoding(len(text.split()))
        encoding.parent = self
        return encoding
        
    def token_to_id(self, token):
        """Mock token_to_id method."""
        return self.token_to_id_map.get(token, 3)  # Return UNK if not found


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            'translation': {
                'en': f'Hello world number {idx}',
                'it': f'Ciao mondo numero {idx}'
            }
        }


def test_causal_mask():
    """Test causal mask generation."""
    size = 5
    mask = causal_mask(size)
    
    assert mask.shape == (1, size, size)
    assert mask.dtype == torch.bool
    
    # Check that it's lower triangular (including diagonal)
    for i in range(size):
        for j in range(size):
            if j <= i:
                assert mask[0, i, j] == True, f"Position ({i}, {j}) should be True"
            else:
                assert mask[0, i, j] == False, f"Position ({i}, {j}) should be False"


def test_bilingual_dataset_initialization():
    """Test BilingualDataset initialization."""
    mock_ds = MockDataset(size=5)
    tokenizer_src = MockTokenizer()
    tokenizer_tgt = MockTokenizer()
    
    dataset = BilingualDataset(
        ds=mock_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang='en',
        tgt_lang='it',
        seq_len=20
    )
    
    assert len(dataset) == 5
    assert dataset.seq_len == 20
    assert dataset.src_lang == 'en'
    assert dataset.tgt_lang == 'it'


def test_bilingual_dataset_getitem():
    """Test BilingualDataset __getitem__ method."""
    mock_ds = MockDataset(size=5)
    tokenizer_src = MockTokenizer()
    tokenizer_tgt = MockTokenizer()
    seq_len = 20
    
    dataset = BilingualDataset(
        ds=mock_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang='en',
        tgt_lang='it',
        seq_len=seq_len
    )
    
    item = dataset[0]
    
    # Check all required keys are present
    required_keys = ['encoder_input', 'decoder_input', 'encoder_mask', 'decoder_mask', 'label', 'src_text', 'tgt_text']
    for key in required_keys:
        assert key in item, f"Key '{key}' missing from dataset item"
    
    # Check tensor shapes
    assert item['encoder_input'].shape == (seq_len,)
    assert item['decoder_input'].shape == (seq_len,)
    assert item['label'].shape == (seq_len,)
    assert item['encoder_mask'].shape == (1, 1, seq_len)
    assert item['decoder_mask'].shape == (1, seq_len, seq_len)
    
    # Check tensor types
    assert item['encoder_input'].dtype == torch.int64
    assert item['decoder_input'].dtype == torch.int64
    assert item['label'].dtype == torch.int64
    assert item['encoder_mask'].dtype == torch.int32
    assert item['decoder_mask'].dtype == torch.bool
    
    # Check text fields
    assert isinstance(item['src_text'], str)
    assert isinstance(item['tgt_text'], str)
    assert 'Hello world number 0' in item['src_text']
    assert 'Ciao mondo numero 0' in item['tgt_text']


def test_bilingual_dataset_special_tokens():
    """Test that special tokens are properly used in dataset."""
    mock_ds = MockDataset(size=1)
    tokenizer_src = MockTokenizer()
    tokenizer_tgt = MockTokenizer()
    seq_len = 20
    
    dataset = BilingualDataset(
        ds=mock_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang='en',
        tgt_lang='it',
        seq_len=seq_len
    )
    
    item = dataset[0]
    
    # Check that SOS token is at the beginning
    assert item['encoder_input'][0] == 0  # SOS token ID
    assert item['decoder_input'][0] == 0  # SOS token ID
    
    # Check that PAD tokens exist (assuming text is shorter than seq_len)
    pad_token_id = 2
    assert pad_token_id in item['encoder_input']
    assert pad_token_id in item['decoder_input']
    assert pad_token_id in item['label']


def test_bilingual_dataset_masks():
    """Test that masks are properly generated."""
    mock_ds = MockDataset(size=1)
    tokenizer_src = MockTokenizer()
    tokenizer_tgt = MockTokenizer()
    seq_len = 10
    
    dataset = BilingualDataset(
        ds=mock_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang='en',
        tgt_lang='it',
        seq_len=seq_len
    )
    
    item = dataset[0]
    
    # Encoder mask should mask padding tokens
    pad_token_id = 2
    encoder_input = item['encoder_input']
    encoder_mask = item['encoder_mask'][0, 0]  # Remove batch and head dimensions
    
    for i in range(seq_len):
        if encoder_input[i] == pad_token_id:
            assert encoder_mask[i] == 0, f"Padding position {i} should be masked"
        else:
            assert encoder_mask[i] == 1, f"Non-padding position {i} should not be masked"
    
    # Decoder mask should be causal and mask padding
    decoder_mask = item['decoder_mask'][0]  # Remove batch dimension
    
    # Check it's lower triangular
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert decoder_mask[i, j] == False, f"Future position ({i}, {j}) should be masked"

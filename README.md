# Transformer Implementation from Scratch

This repository contains a PyTorch implementation of the Transformer architecture as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The implementation follows the original paper closely while maintaining clean, modular, and well-documented code.

## Architecture Overview

The Transformer model consists of an encoder-decoder architecture that uses self-attention mechanisms to process sequential data. Here's a high-level overview of the key components:

```
Encoder:
├── Multi-Head Self-Attention
├── Feed-Forward Network
└── Layer Normalization & Residual Connections

Decoder:
├── Masked Multi-Head Self-Attention
├── Multi-Head Cross-Attention
├── Feed-Forward Network
└── Layer Normalization & Residual Connections
```

### Key Features

- Full implementation of the Transformer architecture
- Modular, class-based design using PyTorch
- Training with label smoothing and learning rate scheduler
- Proper masking implementation (padding + look-ahead)
- No external transformer libraries
- Colab-compatible training and testing notebooks
- Comprehensive test suite

## Project Structure

```
transformer-from-scratch/
├── notebooks/          # Jupyter notebooks for training and testing
├── transformer/        # Main transformer implementation
├── data/              # Training data and vocabulary
├── scripts/           # Training and evaluation scripts
└── tests/             # Unit tests
```

## Installation

```bash
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

## Usage

### Training

1. Open `notebooks/train.ipynb` in Google Colab
2. Follow the notebook instructions to train the model
3. Alternatively, use the CLI:
   ```bash
   python scripts/train.py --config config.json
   ```

### Testing

1. Open `notebooks/test.ipynb` in Google Colab
2. Load a trained model and generate translations
3. Or use the CLI:
   ```bash
   python scripts/evaluate.py --model path/to/model.pth --input "Source text"
   ```

## Model Configuration

The model can be configured through the `TransformerConfig` class in `config.py`. Key hyperparameters include:

- `vocab_size`: Size of the vocabulary
- `max_seq_len`: Maximum sequence length
- `d_model`: Model dimension (512 in paper)
- `num_heads`: Number of attention heads (8 in paper)
- `num_layers`: Number of encoder/decoder layers (6 in paper)
- `d_ff`: Feed-forward network dimension (2048 in paper)
- `dropout`: Dropout rate (0.1 in paper)

## Implementation Details

### Attention Mechanism

The multi-head attention mechanism is implemented as described in the paper:

```python
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

### Positional Encoding

Uses sinusoidal position embeddings:

```python
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

### Label Smoothing

Implements label smoothing as described in the paper to prevent overconfident predictions.

## Evaluation

The model can be evaluated using:
- BLEU score for translation tasks
- Perplexity for language modeling
- Custom metrics can be added in `utils.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

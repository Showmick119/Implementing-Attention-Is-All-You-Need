# Transformer Implementation from Scratch

A complete PyTorch implementation of the Transformer architecture from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. This implementation is designed for English-Italian translation tasks and includes comprehensive training and inference pipelines.

## 🏗️ Architecture Overview

This implementation includes all core components of the Transformer architecture:

- **Input Embeddings**: Token embeddings with scaling by √d_model
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Applied before each sub-layer (pre-norm)
- **Residual Connections**: Skip connections around each sub-layer
- **Encoder-Decoder Architecture**: Complete sequence-to-sequence model

## 📁 Project Structure

```
├── config/
│   └── config.py              # Model configuration and hyperparameters
├── data/
│   └── dataset.py             # Bilingual dataset class with tokenization
├── transformer/
│   └── model.py               # Complete Transformer implementation
├── scripts/
│   └── train.py               # Training script with WandB integration
│   └── translate.py           # Translation inference utilities
├── notebooks/
│   ├── transformer_train.ipynb      # Interactive training notebook
│   ├── transformer_inference.ipynb  # Inference and testing notebook
│   └── attention_visualization.ipynb # Attention pattern visualization
├── requirements.txt           # Project dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore patterns
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Showmick119/Implementing-Attention-Is-All-You-Need.git
cd Implementing-Attention-Is-All-You-Need
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

#### Option 1: Jupyter Notebooks (Recommended for Colab)
1. Open `notebooks/transformer_train.ipynb` in Google Colab
2. Follow the step-by-step training process
3. Monitor training progress with built-in visualizations

#### Option 2: Command Line
```bash
python scripts/train.py
```

### Inference

Use the `notebooks/transformer_inference.ipynb` notebook to:
- Load trained models
- Perform translation inference on custom inputs

### Attention Visualization

Use the `notebooks/attention_visualization.ipynb` notebook to:
- Load trained models
- Visualize attention patterns
- Analyze model behavior

## 🔧 Configuration

The model configuration is managed in `config/config.py`:

```python
{
    "batch_size": 8,           # Training batch size
    "num_epochs": 20,          # Number of training epochs
    "lr": 1e-4,               # Learning rate
    "seq_len": 350,           # Maximum sequence length
    "d_model": 512,           # Model dimension
    "lang_src": "en",         # Source language (English)
    "lang_tgt": "it",         # Target language (Italian)
    "model_folder": "weights", # Model checkpoint directory
    "preload": None,          # Path to pretrained model
    "experiment_name": "runs/tmodel"  # Experiment tracking name
}
```

## 📊 Dataset

The implementation uses the **OPUS Books** dataset for English-Italian translation:
- Automatically downloaded via HuggingFace datasets
- Includes proper tokenization with special tokens ([SOS], [EOS], [PAD])
- Handles variable-length sequences with padding
- Creates appropriate attention masks for training

## 🧠 Model Details

### Architecture Specifications
- **Model Dimension (d_model)**: 512
- **Feed-Forward Dimension**: 2048
- **Number of Heads**: 8
- **Number of Layers**: 6 (encoder) + 6 (decoder)
- **Vocabulary Size**: Dynamic (based on tokenizer)
- **Maximum Sequence Length**: 350 tokens

### Key Implementation Details
- **Attention Mechanism**: Scaled dot-product attention
- **Positional Encoding**: Sinusoidal functions (sin/cos)
- **Normalization**: Layer normalization (pre-norm configuration)
- **Dropout**: Applied throughout the model for regularization
- **Weight Initialization**: Xavier initialization

## 📈 Training Process

1. **Data Preprocessing**: Tokenization and sequence preparation
2. **Model Initialization**: Transformer model with specified configuration
3. **Training Loop**: Forward pass, loss calculation, backpropagation
4. **Validation**: BLEU score evaluation on validation set
5. **Checkpointing**: Model state saving for resuming training
6. **Monitoring**: Real-time metrics via Weights & Biases

## 🔍 Evaluation

The model is evaluated using:
- **BLEU Score**: Standard metric for translation quality
- **Attention Visualization**: Qualitative analysis of attention patterns

## 💻 Google Colab Support

The implementation is fully compatible with Google Colab:
- All notebooks run seamlessly in Colab environment
- Automatic GPU detection and utilization
- Pre-configured for easy experimentation
- No local setup required

## 🛠️ Customization

### For Different Language Pairs
1. Update `lang_src` and `lang_tgt` in configuration
2. Ensure dataset availability for the language pair
3. Adjust vocabulary size if needed

### For Different Datasets
1. Modify the dataset loading in `scripts/train.py`
2. Ensure data format compatibility with `BilingualDataset` class
3. Update tokenizer training if needed

### Model Architecture Changes
1. Adjust hyperparameters in `config/config.py`
2. Modify model architecture in `transformer/model.py`
3. Update training script accordingly

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Detailed implementation guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional features
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This implementation is based on the original Transformer paper. Special thanks to the PyTorch team and the open-source ML community for providing excellent tools and resources.
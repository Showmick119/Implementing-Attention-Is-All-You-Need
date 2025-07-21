"""
Transformer model implementation based on "Attention Is All You Need".

This module contains a complete implementation of the Transformer architecture including:
- Input embeddings with scaling
- Sinusoidal positional encoding
- Multi-head attention mechanism
- Feed-forward networks
- Layer normalization and residual connections
- Encoder and decoder stacks
- Complete transformer model with projection layer

The implementation follows the original paper closely and is designed for
sequence-to-sequence translation tasks.
"""

import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Input embedding layer that converts token IDs to dense vectors.
    
    This layer maps discrete token IDs to continuous embedding vectors and applies
    scaling by sqrt(d_model) as specified in the paper.
    
    Args:
        d_model (int): Dimension of the embedding vectors
        vocab_size (int): Size of the vocabulary
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # length of the embedding vectors for each token
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)  # will map Input IDs to their corresponding Embeddings
    
    def forward(self, x):
        """
        Forward pass through the embedding layer.
        
        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Scaled embeddings of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)  # scaling the embedding output as done in the paper


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer that adds position information to embeddings.
    
    Uses sinusoidal functions to encode position information as described in
    the paper. The encoding is added to the input embeddings to
    provide the model with information about token positions.
    
    Args:
        d_model (int): Dimension of the model
        seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # initializing a matrix of shape (seq_len, d_model) where we will store all our positional encodings
        pe = torch.zeros(seq_len, d_model)
        
        # create a position vector of shape (seq_len,) for applying the formula
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # converts tensor of shape (seq_len,) ---> (seq_len, 1)

        # create the division term for positional encoding formula, but with numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sine to the even positions
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))

        # apply cosine to the odd positions
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))

        # creating another dimension, such that we can have batches of sentence
        pe = pe.unsqueeze(0)  # (seq_len, d_model) ---> (1, seq_len, d_model)

        # registering the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Embeddings with positional encoding of shape (batch_size, seq_len, d_model)
        """
        x += (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    """
    Layer normalization module for stabilizing training.

    Normalizes inputs across the feature dimension and applies learnable
    scale and shift parameters.
    
    Args:
        features (int): The number of features for which the scale and shift parameters have to be learned
        eps (float): Small epsilon value for numerical stability. Also prevents division by zero. Default: 1e-6
    """
    
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # gets multiplied
        self.bias = nn.Parameter(torch.ones(features)) # gets added
    
    def forward(self, x):
        """
        Apply layer normalization to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Normalized tensor of shape (batch_size, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)  # (Batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)   # (Batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward network block.

    Consists of two linear transformations with ReLU activation and dropout.
    Applied to each position separately and identically.
    
    Args:
        d_model (int): Model dimension (input and output size)
        d_ff (int): Hidden dimension of the feed-forward network
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)  # W1 and B1, as defined in section 3.3 of the paper
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)  # W2 and B2

    def forward(self, x):
        """
        Apply position-wise feed-forward network.
        
        Applies two linear transformations with ReLU activation and dropout:
        FFN(x) = max(0, xW1 + b1)W2 + b2
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Linear 1: (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff)
        # Linear 2: (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention mechanism.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    
    Args:
        d_model (int): Model dimension
        h (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # making sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.
        
        Implements the core attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        Applies optional masking and dropout for regularization.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch, h, seq_len, d_k)
            key (torch.Tensor): Key tensor of shape (batch, h, seq_len, d_k)  
            value (torch.Tensor): Value tensor of shape (batch, h, seq_len, d_k)
            mask (torch.Tensor): Attention mask (None for no masking)
            dropout (nn.Dropout): Dropout layer (None for no dropout)
            
        Returns:
            tuple: (attention_output, attention_weights)
                - attention_output: Weighted values of shape (batch, h, seq_len, d_k)
                - attention_weights: Attention scores of shape (batch, h, seq_len, seq_len)
        """
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
             # writing a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len) # applying softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores  # will use the attention scores later on for visualizations

    def forward(self, q, k, v, mask):
        """
        Apply multi-head attention mechanism.
        
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model)
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model)
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model)
            mask (torch.Tensor): Attention mask (None for no masking)
            
        Returns:
            torch.Tensor: Multi-head attention output of shape (batch_size, seq_len_q, d_model)
        """
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # combining all the heads together

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)  # multiplying by Wo


class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization.
    
    Implements residual connections around each of the sub-layers, followed by
    layer normalization. Uses pre-norm architecture where normalization is
    applied before the sub-layer.
    
    Args:
        features (int): The number of features for which the scale and shift parameters have to be learned
        dropout (float): Dropout probability
    """
    
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        """
        Apply residual connection with layer normalization.
        
        Implements pre-norm architecture: x + dropout(sublayer(norm(x)))
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            sublayer (callable): Function to apply (attention or feed-forward)
            
        Returns:
            torch.Tensor: Output with residual connection of shape (batch_size, seq_len, d_model)
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Single encoder block of the Transformer.
    
    Contains a multi-head self-attention mechanism followed by a position-wise
    feed-forward network. Both sub-layers have residual connections and layer
    normalization.
    
    Args:
        features (int): The number of features for which the scale and shift parameters have to be learned
        self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
        dropout (float): Dropout probability for residual connections
    """
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        """
        Forward pass through encoder block.
        
        Applies self-attention followed by feed-forward network, each with
        residual connections and layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (torch.Tensor): Source padding mask
            
        Returns:
            torch.Tensor: Encoder block output of shape (batch_size, seq_len, d_model)
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Complete encoder stack of the Transformer.
    
    Composed of a stack of N identical encoder blocks. The output of each
    layer is the input to the next layer. Final layer normalization is
    applied to the output.
    
    Args:
        features (int): The number of features for which the scale and shift normalization parameters have to be learned
        layers (nn.ModuleList): List of encoder blocks
    """
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, mask):
        """
        Forward pass through encoder stack.
        
        Processes input through all encoder blocks sequentially and applies
        final layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor): Source padding mask
            
        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    """
    Single decoder block of the Transformer.
    
    Contains three sub-layers: masked multi-head self-attention, multi-head
    cross-attention over encoder output, and position-wise feed-forward network.
    All sub-layers have residual connections and layer normalization.
    
    Args:
        features (int): The number of features for which the scale and shift normalization parameters have to be learned
        self_attention_block (MultiHeadAttentionBlock): Masked self-attention mechanism
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention mechanism
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
        dropout (float): Dropout probability for residual connections
    """
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through decoder block.
        
        Applies masked self-attention, cross-attention, and feed-forward network,
        each with residual connections and layer normalization.
        
        Args:
            x (torch.Tensor): Target input of shape (batch_size, tgt_len, d_model)
            encoder_output (torch.Tensor): Encoder output of shape (batch_size, src_len, d_model)
            src_mask (torch.Tensor): Source padding mask
            tgt_mask (torch.Tensor): Target causal mask
            
        Returns:
            torch.Tensor: Decoder block output of shape (batch_size, tgt_len, d_model)
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Complete decoder stack of the Transformer.
    
    Composed of a stack of N identical decoder blocks. Each block processes
    the target sequence while attending to the encoder output. Final layer
    normalization is applied to the output.
    
    Args:
        features (int): The number of features for which the scale and shift normalization parameters have to be learned
        layers (nn.ModuleList): List of decoder blocks
    """
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through decoder stack.
        
        Processes target input through all decoder blocks sequentially while
        attending to encoder output, then applies final layer normalization.
        
        Args:
            x (torch.Tensor): Target input of shape (batch_size, tgt_len, d_model)
            encoder_output (torch.Tensor): Encoder output of shape (batch_size, src_len, d_model)
            src_mask (torch.Tensor): Source padding mask
            tgt_mask (torch.Tensor): Target causal mask
            
        Returns:
            torch.Tensor: Decoded representation of shape (batch_size, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Final projection layer that maps decoder output to vocabulary logits.
    
    Projects the decoder output from model dimension to vocabulary size and
    applies log softmax for probability distribution over the vocabulary.
    
    Args:
        d_model (int): Model dimension (input size)
        vocab_size (int): Size of the target vocabulary
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, x):
        """
        Project hidden states to vocabulary logits.
        
        Maps the decoder output to vocabulary size and applies log softmax
        to get log probabilities for each token in the vocabulary.
        
        Args:
            x (torch.Tensor): Decoder output of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, seq_len, vocab_size)
        """
        # (Batch, seq_len, d_model) ---> (Batch, seq_len, vocab_size) 
        # Mapping everything to the vocabulary
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.

    Combines encoder and decoder stacks with embeddings, positional encoding, and
    final projection layer for translation tasks.
    
    Args:
        encoder (Encoder): Encoder stack
        decoder (Decoder): Decoder stack  
        src_embed (InputEmbeddings): Source language embeddings
        tgt_embed (InputEmbeddings): Target language embeddings
        src_pos (PositionalEncoding): Source positional encoding
        tgt_pos (PositionalEncoding): Target positional encoding
        projection_layer (ProjectionLayer): Final vocabulary projection layer
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        """
        Encode source sequence through the encoder stack.
        
        Args:
            src (torch.Tensor): Source token IDs of shape (batch, src_len)
            src_mask (torch.Tensor): Source padding mask
            
        Returns:
            torch.Tensor: Encoded source representation of shape (batch, src_len, d_model)
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decode target sequence through the decoder stack.
        
        Args:
            encoder_output (torch.Tensor): Encoded source of shape (batch, src_len, d_model)
            src_mask (torch.Tensor): Source padding mask
            tgt (torch.Tensor): Target token IDs of shape (batch, tgt_len)
            tgt_mask (torch.Tensor): Target causal mask
            
        Returns:
            torch.Tensor: Decoded target representation of shape (batch, tgt_len, d_model)
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        Project decoder output to vocabulary logits.
        
        Args:
            x (torch.Tensor): Decoder output of shape (batch, tgt_len, d_model)
            
        Returns:
            torch.Tensor: Log probabilities over vocabulary of shape (batch, tgt_len, vocab_size)
        """
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Build a complete Transformer model with specified parameters.
    
    Creates and initializes all components of the Transformer architecture including
    embeddings, positional encodings, encoder/decoder stacks, and projection layer.
    Uses Xavier uniform initialization for all parameters.
    
    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary  
        src_seq_len (int): Maximum source sequence length
        tgt_seq_len (int): Maximum target sequence length
        d_model (int): Model dimension. Default: 512
        N (int): Number of encoder/decoder layers. Default: 6
        h (int): Number of attention heads. Default: 8
        dropout (float): Dropout probability. Default: 0.1
        d_ff (int): Feed-forward network hidden dimension. Default: 2048
        
    Returns:
        Transformer: Complete initialized Transformer model
    """
    # creating the embedding layers for both the source language and target language, for both the encoder and decoder
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # creating the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # creating the N number of encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # creating the N number of decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # creating the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # creates a projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameter's systematically instead of randomly
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
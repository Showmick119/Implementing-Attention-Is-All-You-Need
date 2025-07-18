import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # length of the embedding vectors for each token
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)  # will map Input IDs to their corresponding Embeddings
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # scaling the embedding output as done in the paper


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # initialize a matrix of shape (seq_len, d_model) where we will store all our positional encodings
        pe = torch.zeros(seq_len, d_model)
        
        # create a position vector of shape (seq_len,) for applying the formula
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # converts tensor of shape (seq_len,) ---> (seq_len, 1)

        # create the division term for positional encoding formula, but with numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sinusoidal encoding to the even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cosine encoding to the odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # creating another dimension, such that we can have batches of sentences. convert (seq_len, d_model) ---> (1, seq_len, d_model)

        '''
        Finally, register the tensor in the buffer of this module.

        - When you have a tensor you want to keep inside the module, not as a learned parameter, but you want it to be saved when you save the file of the model.
        - Then you should register it as a buffer, this way the tensor will be saved in the file, along with the state dicts of the model.
        '''
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        - These Positional Encodings are fixed. And we create Positional Encodings until the max seq_len.
        - These Positional Encodings are not learnt during training, hence we set requires_grad_( ) to False.
        '''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # gets multiplied
        self.bias = nn.Parameter(torch.ones(1)) # gets added
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)  # W1 and B1, as defined in section 3.3 of the paper
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)  # W2 and B2

    def forward(self, x):
        # Linear 1:
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff)

        # Linear 2:
        # (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

class MultiHeadAttention:
    pass
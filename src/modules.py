import numpy as np
from numpy.random import default_rng

def softmax(scores):
    """
    Apply softmax to normalize attention scores.

    Args:
        scores (np.ndarray): Attention scores.

    Returns:
        np.ndarray: Normalized attention scores.
    """
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # numerical stability, and normalizes across each row (i.e. across all key vectors for each query)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.seed = 47
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(self, x):
        return default_rng(self.seed).random((x.shape[1], self.d_model))

class SelfAttention:
    def __init__(self, d_model = 512):
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(1. / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(1. / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(1. / d_model)
    
    def forward(self, x):
        # x: N D
        q = np.dot(x, self.W_q)
        k = np.dot(x, self.W_k)
        v = np.dot(x, self.W_v)

        d_k = x.shape[-1]
        attn_scores = (q @ k.T) / np.sqrt(d_k)
        attn_weights = softmax(attn_scores)
        out = attn_weights @ v
        return out

class FFN:
    def __init__(self, d_model = 512, d_ff = 512 * 4):
        pass
    
    def forward(self, x):
        raise NotImplementedError()

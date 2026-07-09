import numpy as np
from numpy.random import default_rng
from backend import CPU

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.seed = 47
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(self, x):
        return default_rng(self.seed).random((x.shape[1], self.d_model))

class SelfAttention:
    def __init__(self, d_model = 512, backend = CPU):
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(1. / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(1. / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(1. / d_model)
        self.backend = backend
    
    def forward(self, x):
        # x: N D
        q = np.dot(x, self.W_q)
        k = np.dot(x, self.W_k)
        v = np.dot(x, self.W_v)

        attn_scores = self.backend.scale(self.backend.matmul(q, k.T), x.shape[-1])
        attn_weights = self.backend.softmax(attn_scores)
        out = self.backend.matmul(attn_weights, v)

        return out

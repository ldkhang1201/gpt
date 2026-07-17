import numpy as np
from backend import CPU

class MultiHeadAttention:
    def __init__(self, D = 512, n_heads = 4, backend = CPU):
        self.heads = [AttentionHead(D, D // n_heads, backend) for _ in range(n_heads)]
        self.W_q = np.random.randn(D, D)
        self.W_k = np.random.randn(D, D)
        self.W_v = np.random.randn(D, D)
    

class AttentionHead:
    def __init__(self, d_model = 512, k = 512 // 4,  backend = CPU):
        
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

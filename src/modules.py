import numpy as np
from backend import CPU

class AttentionHead:
    def __init__(self, d_model = 512, d_head = 128, backend = CPU()):
        self.backend = backend
        self.W_q = np.random.randn(d_model, d_head) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_head) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_head) / np.sqrt(d_model)

    def forward(self, x):
        # x: N D
        q = self.backend.matmul(x, self.W_q)
        k = self.backend.matmul(x, self.W_k)
        v = self.backend.matmul(x, self.W_v)

        attn_scores = self.backend.scale(self.backend.matmul(q, k.T), q.shape[-1])
        attn_weights = self.backend.softmax(attn_scores)
        return self.backend.matmul(attn_weights, v)


class MultiHeadAttention:
    def __init__(self, D = 512, n_heads = 4, backend = CPU()):
        self.heads = [AttentionHead(D, D // n_heads, backend) for _ in range(n_heads)]
        self.W_o = np.random.randn(D, D) / np.sqrt(D)
        self.backend = backend

    def forward(self, x):
        return self.backend.matmul(np.concatenate([h.forward(x) for h in self.heads], axis=-1), self.W_o)

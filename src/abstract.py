import torch
import torch.nn as nn

class TransformerBase(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, n_heads=4, d_ff=512 * 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        raise NotImplementedError()
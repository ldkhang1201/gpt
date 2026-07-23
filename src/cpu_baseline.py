import torch
from torch.profiler import record_function

from abstract import TransformerBase


class CpuPipeline(TransformerBase):
    """Reference implementation: attention in plain PyTorch on the CPU."""

    def attention(self, q, k, v):
        D = q.shape[-1]

        with record_function("2a_qk_matmul"):
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / D ** 0.5  # [B, N, N]

        with record_function("2b_softmax"):
            attn_weights = torch.softmax(attn_scores, dim=-1)

        with record_function("2c_value_weighted_sum"):
            return torch.matmul(attn_weights, v)  # [B, N, D]

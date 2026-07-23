import math

import torch

from abstract import TransformerBase


def _attend(q, k, v, scale):
    """Attention for one sequence as plain Python lists: [N][D] x3 -> [N][D].

    Both matmuls are the classic naive form — three nested loops, one output
    element per innermost accumulation — the baseline style behind published
    1000x+ GPU-vs-CPU speedups.
    """
    n, d = len(q), len(q[0])

    scores = [[0.0] * n for _ in range(n)]  # QK^T / sqrt(D)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for x in range(d):
                s += q[i][x] * k[j][x]
            scores[i][j] = s * scale

    weights = []  # row softmax, max-subtracted for stability
    for row in scores:
        m = max(row)
        exps = [math.exp(s - m) for s in row]
        total = sum(exps)
        weights.append([e / total for e in exps])

    out = [[0.0] * d for _ in range(n)]  # weights @ V
    for i in range(n):
        for x in range(d):
            s = 0.0
            for j in range(n):
                s += weights[i][j] * v[j][x]
            out[i][x] = s
    return out


class CpuNaivePipeline(TransformerBase):
    """Blog-post CPU baseline: attention as bare Python loops.

    Same algorithm as CpuPipeline with the library removed — no BLAS, no SIMD,
    no threads, no cache blocking. Exists to show what "2000x faster than the
    CPU" headlines divide by; O(N^2 D) interpreter ops, so small N only.
    """

    def attention(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        return torch.stack([
            torch.tensor(_attend(q[b].tolist(), k[b].tolist(), v[b].tolist(), scale))
            for b in range(q.shape[0])
        ])

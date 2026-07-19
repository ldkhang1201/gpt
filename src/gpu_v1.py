import math

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline


@cuda.jit
def _matmul(a, b, out):
    """out = a @ b — one thread per output element, all reads from global memory."""
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        acc = float32(0.)
        for kk in range(a.shape[1]):
            acc += a[i, kk] * b[kk, j]
        out[i, j] = acc


@cuda.jit
def _scale(x, num):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        x[i, j] /= num


@cuda.jit
def _softmax(x, out):
    """Row softmax, one thread per row: max pass, sum pass, write pass."""
    i = cuda.grid(1)
    if i < x.shape[0]:
        # row max for numerical stability (avoid exp overflow)
        m = x[i, 0]
        for j in range(1, x.shape[1]):
            if x[i, j] > m:
                m = x[i, j]
        denom = float32(0.)
        for j in range(x.shape[1]):
            denom += math.exp(x[i, j] - m)
        for j in range(x.shape[1]):
            out[i, j] = math.exp(x[i, j] - m) / denom


def _grid2d(rows, cols, tpb=(16, 16)):
    return (math.ceil(rows / tpb[0]), math.ceil(cols / tpb[1])), tpb


class GpuV1(GpuPipeline):
    """V1 — naive three-kernel attention: QK^T matmul, softmax, weighted sum.

    Every intermediate, including the full N x N score matrix, round-trips
    through global memory between kernels.
    """

    def _attend(self, q, k, v):
        N, D = q.shape
        scores = torch.empty((N, N), device=q.device)
        weights = torch.empty((N, N), device=q.device)
        out = torch.empty((N, D), device=q.device)

        bpg, tpb = _grid2d(N, N)
        _matmul[bpg, tpb](q, k.t().contiguous(), scores)
        # np.float32 keeps the kernel arithmetic in fp32 (a python float is typed float64)
        _scale[bpg, tpb](scores, np.float32(D ** 0.5))

        _softmax[math.ceil(N / 256), 256](scores, weights)

        bpg, tpb = _grid2d(N, D)
        _matmul[bpg, tpb](weights, v, out)
        return out

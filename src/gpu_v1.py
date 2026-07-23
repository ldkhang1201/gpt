import math

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline


@cuda.jit
def _matmul(a, b, out):
    """out = a @ b — one thread per output element, all reads from global memory.

    threadIdx.x walks the output column (the contiguous axis) so warp reads
    of b are coalesced; a[i, kk] is a broadcast within the warp.
    """
    j, i = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        acc = float32(0.)
        for kk in range(a.shape[1]):
            acc += a[i, kk] * b[kk, j]
        out[i, j] = acc


@cuda.jit
def _scale(x, num):
    j, i = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        x[i, j] /= num


TPB = 256  # threads cooperating on one row


@cuda.jit
def _softmax(x, out):
    """Row softmax, one block per row: max pass, sum pass, write pass.

    Each thread strides over its slice of the row; the block reduces the
    per-thread max and sum in shared memory. Still three reads of the row
    (naive), but the row is now processed cooperatively instead of by a
    single thread.
    """
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = x.shape[1]

    sm = cuda.shared.array(TPB, float32)
    sd = cuda.shared.array(TPB, float32)

    # row max for numerical stability (avoid exp overflow)
    m = x[row, tid] if tid < n else float32(-3.0e38)
    for j in range(tid + TPB, n, TPB):
        if x[row, j] > m:
            m = x[row, j]
    sm[tid] = m
    cuda.syncthreads()

    stride = TPB // 2
    while stride > 0:
        if tid < stride and sm[tid + stride] > sm[tid]:
            sm[tid] = sm[tid + stride]
        cuda.syncthreads()
        stride //= 2
    m = sm[0]

    d = float32(0.)
    for j in range(tid, n, TPB):
        d += math.exp(x[row, j] - m)
    sd[tid] = d
    cuda.syncthreads()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            sd[tid] += sd[tid + stride]
        cuda.syncthreads()
        stride //= 2
    denom = sd[0]

    for j in range(tid, n, TPB):
        out[row, j] = math.exp(x[row, j] - m) / denom


def _grid2d(rows, cols, tpb=(16, 16)):
    # grid x covers columns, grid y covers rows (matches j, i = cuda.grid(2))
    return (math.ceil(cols / tpb[0]), math.ceil(rows / tpb[1])), tpb


class GpuV1(GpuPipeline):
    """V1 — naive three-kernel attention: QK^T matmul, softmax, weighted sum.

    Every intermediate, including the full N x N score matrix, round-trips
    through global memory between kernels, and the softmax reads each row
    three times with one thread per row.
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

        _softmax[N, TPB](scores, weights)

        bpg, tpb = _grid2d(N, D)
        _matmul[bpg, tpb](weights, v, out)
        return out

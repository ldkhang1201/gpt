import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TILE_N = 8   # K/V rows staged in shared memory per step: 2 * TILE_N * D * 4B <= 48 KB for D <= 768
TILE_M = 64  # query rows (= threads) per block


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the fused kernel for a fixed model width D — numba needs
    compile-time-constant shared/local array shapes."""

    @cuda.jit
    def flash(q, k, v, scale, out):
        i = cuda.grid(1)  # this thread owns query row i
        tid = cuda.threadIdx.x
        N = q.shape[0]

        ks = cuda.shared.array((TILE_N, D), float32)
        vs = cuda.shared.array((TILE_N, D), float32)
        qi = cuda.local.array(D, float32)
        acc = cuda.local.array(D, float32)

        if i < N:
            for c in range(D):
                qi[c] = q[i, c]
                acc[c] = float32(0.)
        m = float32(-math.inf)
        l = float32(0.)

        for t in range(0, N, TILE_N):
            rows = min(TILE_N, N - t)

            # cooperative load of the K/V tile (all threads, flat index)
            for idx in range(tid, rows * D, TILE_M):
                r = idx // D
                c = idx - r * D
                ks[r, c] = k[t + r, c]
                vs[r, c] = v[t + r, c]
            cuda.syncthreads()

            if i < N:
                for r in range(rows):
                    s = float32(0.)
                    for c in range(D):
                        s += qi[c] * ks[r, c]
                    s *= scale

                    if s > m:  # new running max: rescale everything accumulated
                        corr = math.exp(m - s)
                        l *= corr
                        for c in range(D):
                            acc[c] *= corr
                        m = s

                    p = math.exp(s - m)
                    l += p
                    for c in range(D):
                        acc[c] += p * vs[r, c]
            cuda.syncthreads()

        if i < N:
            for c in range(D):
                out[i, c] = acc[c] / l

    return flash


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-style fused kernel.

    Tiles over Q, K and V simultaneously: the softmax runs online while the
    weighted sum accumulates, so the full N x N score matrix never exists in
    global memory (O(N) extra memory instead of O(N^2)).
    """

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_kernel(D)
        # np.float32 keeps the whole accumulator loop in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        # (32x slower fp64 FMAs on a T4)
        kernel[math.ceil(N / TILE_M), TILE_M](q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out

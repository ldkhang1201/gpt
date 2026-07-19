import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

WARP = 32
WARPS = 16   # query rows (= warps) per block; K/V tiles are reused WARPS times
TILE_N = 8   # K/V rows staged in shared memory per step: 2 * TILE_N * D * 4B <= 48 KB for D <= 768


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the fused kernel for a fixed model width D — numba needs
    compile-time-constant shared/local array shapes."""
    assert D % WARP == 0, "flash kernel assumes d_model is a multiple of 32"
    CH = D // WARP  # columns of the row owned by each lane (register-resident)

    @cuda.jit
    def flash(q, k, v, scale, out):
        lane = cuda.threadIdx.x                 # 0..31
        wid = cuda.threadIdx.y                  # warp id within the block
        i = cuda.blockIdx.x * WARPS + wid       # this warp owns query row i
        tid = wid * WARP + lane
        N = q.shape[0]

        ks = cuda.shared.array((TILE_N, D), float32)
        vs = cuda.shared.array((TILE_N, D), float32)
        qi = cuda.local.array(CH, float32)      # lane's slice of the query row
        acc = cuda.local.array(CH, float32)     # lane's slice of the output row

        if i < N:
            for c in range(CH):
                qi[c] = q[i, lane + c * WARP]
                acc[c] = float32(0.)
        m = float32(-math.inf)
        l = float32(0.)

        for t in range(0, N, TILE_N):
            rows = N - t  # builtin min() is unsupported in numba CUDA kernels
            if rows > TILE_N:
                rows = TILE_N

            # cooperative, coalesced load of the K/V tile (all threads, flat index)
            for idx in range(tid, rows * D, WARPS * WARP):
                r = idx // D
                c = idx - r * D
                ks[r, c] = k[t + r, c]
                vs[r, c] = v[t + r, c]
            cuda.syncthreads()

            if i < N:
                for r in range(rows):
                    part = float32(0.)
                    for c in range(CH):
                        part += qi[c] * ks[r, lane + c * WARP]
                    # butterfly reduction: every lane ends with the full dot product
                    part += cuda.shfl_xor_sync(0xffffffff, part, 16)
                    part += cuda.shfl_xor_sync(0xffffffff, part, 8)
                    part += cuda.shfl_xor_sync(0xffffffff, part, 4)
                    part += cuda.shfl_xor_sync(0xffffffff, part, 2)
                    part += cuda.shfl_xor_sync(0xffffffff, part, 1)
                    s = part * scale

                    if s > m:  # new running max: rescale everything accumulated
                        corr = math.exp(m - s)
                        l *= corr
                        for c in range(CH):
                            acc[c] *= corr
                        m = s

                    p = math.exp(s - m)
                    l += p
                    for c in range(CH):
                        acc[c] += p * vs[r, lane + c * WARP]
            cuda.syncthreads()

        if i < N:
            for c in range(CH):
                out[i, lane + c * WARP] = acc[c] / l

    return flash


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-style fused kernel, one warp per query row.

    Tiles over Q, K and V simultaneously: K/V tiles live in shared memory,
    the query row and the output accumulator are split across the 32 lanes'
    registers, and the softmax runs online (dot products reduced with warp
    shuffles) while the weighted sum accumulates — so the full N x N score
    matrix never exists in global memory (O(N) extra memory, not O(N^2)).
    """

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_kernel(D)
        # np.float32 keeps the whole accumulator loop in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        # (32x slower fp64 FMAs on a T4)
        kernel[math.ceil(N / WARPS), (WARP, WARPS)](q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out

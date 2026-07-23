import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

WARP = 32
WARPS = 8    # warps per block
ROWS = 4     # query rows per warp: one shared load feeds ROWS FMAs (register tiling)
TILE_N = 8   # K/V rows staged in shared memory per step: 2 * TILE_N * D * 4B <= 48 KB for D <= 768


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the fused kernel for a fixed model width D — numba needs
    compile-time-constant shared/local array shapes."""
    assert D % WARP == 0, "flash kernel assumes d_model is a multiple of 32"
    CH = D // WARP  # columns of a row owned by each lane (register-resident)

    @cuda.jit
    def flash(q, k, v, scale, out):
        lane = cuda.threadIdx.x                            # 0..31
        wid = cuda.threadIdx.y                             # warp id within the block
        base = (cuda.blockIdx.x * WARPS + wid) * ROWS      # first query row of this warp
        tid = wid * WARP + lane
        N = q.shape[0]

        ks = cuda.shared.array((TILE_N, D), float32)
        vs = cuda.shared.array((TILE_N, D), float32)
        qi = cuda.local.array((ROWS, CH), float32)   # lane's slices of the query rows
        acc = cuda.local.array((ROWS, CH), float32)  # lane's slices of the output rows
        mm = cuda.local.array(ROWS, float32)         # running max per row
        ll = cuda.local.array(ROWS, float32)         # running sum per row
        ss = cuda.local.array(ROWS, float32)         # scores of the current key

        for a in range(ROWS):
            for c in range(CH):
                qi[a, c] = q[base + a, lane + c * WARP] if base + a < N else float32(0.)
                acc[a, c] = float32(0.)
            mm[a] = float32(-math.inf)
            ll[a] = float32(0.)

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

            if base < N:
                for r in range(rows):
                    for a in range(ROWS):
                        ss[a] = float32(0.)
                    for c in range(CH):
                        kv = ks[r, lane + c * WARP]  # one shared load ...
                        for a in range(ROWS):
                            ss[a] += qi[a, c] * kv   # ... feeds ROWS FMAs
                    for a in range(ROWS):
                        # butterfly reduction: every lane ends with the full dot product
                        p = ss[a]
                        p += cuda.shfl_xor_sync(0xffffffff, p, 16)
                        p += cuda.shfl_xor_sync(0xffffffff, p, 8)
                        p += cuda.shfl_xor_sync(0xffffffff, p, 4)
                        p += cuda.shfl_xor_sync(0xffffffff, p, 2)
                        p += cuda.shfl_xor_sync(0xffffffff, p, 1)
                        s = p * scale

                        if s > mm[a]:  # new running max: rescale everything accumulated
                            corr = math.exp(mm[a] - s)
                            ll[a] *= corr
                            for c in range(CH):
                                acc[a, c] *= corr
                            mm[a] = s
                        ss[a] = math.exp(s - mm[a])
                        ll[a] += ss[a]
                    for c in range(CH):
                        kv = vs[r, lane + c * WARP]
                        for a in range(ROWS):
                            acc[a, c] += ss[a] * kv
            cuda.syncthreads()

        for a in range(ROWS):
            if base + a < N:
                for c in range(CH):
                    out[base + a, lane + c * WARP] = acc[a, c] / ll[a]

    return flash


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-style fused kernel, one warp per four query rows.

    Tiles over Q, K and V simultaneously: K/V tiles live in shared memory and
    are reused by all warps of the block; the query rows and the output
    accumulators are split across the 32 lanes' registers, with each warp
    register-tiled over ROWS query rows so every K/V element fetched from
    shared memory feeds ROWS FMAs. Dot products are reduced with warp
    shuffles and the softmax runs online while the weighted sum accumulates —
    the full N x N score matrix never exists in global memory (O(N) extra
    memory, not O(N^2)).
    """

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_kernel(D)
        # np.float32 keeps the whole accumulator loop in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        kernel[math.ceil(N / (WARPS * ROWS)), (WARP, WARPS)](
            q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out

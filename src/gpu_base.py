import torch
from numba import cuda

from abstract import TransformerBase


class GpuPipeline(TransformerBase):
    """Base for the GPU versions: handles CPU<->GPU transfer around `_attend`.

    The measured attention time deliberately includes the transfer cost —
    that is the true price of swapping the kernel into the CPU pipeline.
    """

    def attention(self, q, k, v):
        out = torch.empty_like(q)
        for b in range(q.shape[0]):
            res = self._attend(q[b].contiguous().cuda(),
                               k[b].contiguous().cuda(),
                               v[b].contiguous().cuda())
            cuda.synchronize()
            out[b] = res.cpu()
        return out

    def _attend(self, q, k, v):
        """Attention for one sequence, on device: [N, D] x3 -> [N, D]."""
        raise NotImplementedError()

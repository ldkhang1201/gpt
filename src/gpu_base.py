import torch
from numba import cuda

from abstract import TransformerBase


class GpuPipeline(TransformerBase):
    """Base for the GPU versions: handles CPU<->GPU transfer around `_attend`.

    The measured attention time deliberately includes the transfer cost —
    that is the true price of swapping the kernel into the CPU pipeline.
    """

    def attention(self, q, k, v):
        # detach: torch refuses __cuda_array_interface__ on tensors that
        # require grad, and this pipeline is inference-only anyway
        out = torch.empty_like(q)
        for b in range(q.shape[0]):
            res = self._attend(q[b].detach().contiguous().cuda(),
                               k[b].detach().contiguous().cuda(),
                               v[b].detach().contiguous().cuda())
            cuda.synchronize()
            out[b] = res.cpu()
        return out

    def _attend(self, q, k, v):
        """Attention for one sequence, on device: [N, D] x3 -> [N, D].

        Default: the unfused three-step pipeline, each step a separately
        timeable method. Fused versions override _attend wholesale.
        """
        N, D = q.shape
        scores = torch.empty((N, N), device=q.device)
        weights = torch.empty((N, N), device=q.device)
        out = torch.empty((N, D), device=q.device)
        self._step_qkt(q, k, scores)
        self._step_softmax(scores, weights)
        self._step_weighted_sum(weights, v, out)
        return out

    def _step_qkt(self, q, k, scores):
        """scores = (q @ k^T) / sqrt(D), on device."""
        raise NotImplementedError()

    def _step_softmax(self, scores, weights):
        """weights = row softmax of scores, on device."""
        raise NotImplementedError()

    def _step_weighted_sum(self, weights, v, out):
        """out = weights @ v, on device."""
        raise NotImplementedError()

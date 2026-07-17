import numpy as np
import pytest
import torch
import torch.nn.functional as F
from modules import MultiHeadAttention

def test_matches_torch_sdpa():
    mha = MultiHeadAttention(D=64, n_heads=4)
    x = np.random.randn(10, 64)

    out = mha.forward(x)

    # same weights, attention via torch (SDPA default scale = 1/sqrt(d_head) = your q.shape[-1])
    xt = torch.from_numpy(x)
    ref = torch.cat([
        F.scaled_dot_product_attention(
            xt @ torch.from_numpy(h.W_q),
            xt @ torch.from_numpy(h.W_k),
            xt @ torch.from_numpy(h.W_v),
        )
        for h in mha.heads
    ], dim=-1) @ torch.from_numpy(mha.W_o)

    torch.testing.assert_close(torch.from_numpy(out), ref)


def test_gpu_matmul_matches_numpy():
    pytest.importorskip("cupy")
    from backend import GPUv1

    a = np.random.randn(37, 53)  # odd shapes to catch indexing/bounds bugs
    b = np.random.randn(53, 29)

    out = GPUv1().matmul(a, b)

    # kernel computes in f32, numpy defaults to f64 — compare at f32 precision
    np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-6)

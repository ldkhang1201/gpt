import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

STEPS = ["1_embedding", "2a_qk_matmul", "2b_softmax", "2c_value_weighted_sum", "3_ffn"]
ATTN_STEPS = STEPS[1:4]


def _tokens(model, n, batch=1):
    return torch.randint(0, model.embedding.num_embeddings, (batch, n))


def _qkv(model, n):
    with torch.no_grad():
        h = model.embedding(_tokens(model, n))
        return model.q_proj(h), model.k_proj(h), model.v_proj(h)


def sdpa_reference(model, x):
    """Full forward pass with attention done by torch SDPA (correctness oracle)."""
    with torch.no_grad():
        h = model.embedding(x)
        attn = F.scaled_dot_product_attention(model.q_proj(h), model.k_proj(h), model.v_proj(h))
        h = model.norm1(h + attn)
        return model.norm2(h + model.ffn(h))


def step_percentages(model, n):
    """% of one forward pass spent in each labelled step (torch.profiler)."""
    x = _tokens(model, n)
    with torch.no_grad():
        model(x)  # warmup (first GPU call pays numba JIT compilation)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("0_total"):
                model(x)

    def t(key):  # a key can appear more than once -> take max over duplicates
        return max((e.cpu_time_total for e in prof.key_averages() if e.key == key), default=0.0)

    total = t("0_total")
    pct = {s: 100.0 * t(s) / total for s in STEPS}
    pct["other"] = 100.0 - sum(pct.values())
    return pct


def attention_ms(model, n, reps=5):
    """Average wall time (ms) of one attention step on precomputed q, k, v.

    GPU pipelines synchronize and copy back to CPU inside `attention`, so
    a wall clock around the call is correct and includes the transfer cost.
    """
    q, k, v = _qkv(model, n)
    with torch.no_grad():
        model.attention(q, k, v)  # warmup / JIT
        t0 = time.perf_counter()
        for _ in range(reps):
            model.attention(q, k, v)
    return (time.perf_counter() - t0) / reps * 1e3


def sdpa_gpu_ms(model, n, reps=5):
    """torch SDPA on GPU, same contract as `attention_ms`: CPU tensors in,
    CPU result out, with both transfers inside the timed region."""
    q, k, v = _qkv(model, n)
    with torch.no_grad():
        F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu()  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def bench_attention(classes, seq_lens, reps=5):
    """Attention-step time for each pipeline class at each sequence length."""
    times = {name: [] for name in classes}
    for n in seq_lens:
        for name, cls in classes.items():
            model = cls().eval()
            times[name].append(attention_ms(model, n, reps))
    return times

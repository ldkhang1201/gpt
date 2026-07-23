import time

import torch
import torch.nn.functional as F
from numba import cuda
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


def attention_ms(model, n, reps=5, warmup=True):
    """Average wall time (ms) of one attention step on precomputed q, k, v.

    GPU pipelines synchronize and copy back to CPU inside `attention`, so
    a wall clock around the call is correct and includes the transfer cost.
    Pass warmup=False for pure-Python models: nothing to JIT, and one extra
    pass costs seconds to minutes.
    """
    q, k, v = _qkv(model, n)
    with torch.no_grad():
        if warmup:
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


def gpu_specs():
    """Peak fp32 TFLOP/s and memory bandwidth, HW02 roofline style: device
    query for SMs/cc, pynvml for clocks, known-family tables for bus width
    and cores per SM — with a per-family fallback when pynvml is missing."""
    gpu = cuda.get_current_device()
    cc = tuple(gpu.compute_capability)
    sm = gpu.MULTIPROCESSOR_COUNT
    name = gpu.name.decode() if isinstance(gpu.name, bytes) else gpu.name
    cores_per_sm = {(7, 0): 64, (7, 5): 64, (8, 0): 64,
                    (8, 6): 128, (8, 9): 128, (9, 0): 128}.get(cc, 128)
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        sm_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        mem_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        try:
            bus_bits = pynvml.nvmlDeviceGetMemoryBusWidth(h)
        except pynvml.NVMLError:  # per-family fallback
            bus_bits = {(7, 0): 4096, (7, 5): 256, (8, 0): 5120,
                        (8, 6): 384, (8, 9): 384, (9, 0): 6144}.get(cc, 256)
        bw_gbs = mem_mhz * 1e6 * bus_bits / 8 * 2 / 1e9  # DDR: 2 transfers/clock
        pynvml.nvmlShutdown()
    except Exception:  # no pynvml: boost clock + bandwidth from spec sheets
        sm_mhz, bw_gbs = {(7, 5): (1590, 300.0), (8, 0): (1410, 1555.0),
                          (9, 0): (1785, 3900.0)}.get(cc, (1500, 500.0))
    tflops = sm * cores_per_sm * 2 * sm_mhz * 1e6 / 1e12
    return {"name": name, "sm": sm, "tflops": tflops, "bw_gbs": bw_gbs}


def attention_kernel_ms(model, n, reps=5):
    """Device-resident attention time via CUDA events (HW02's cuda_time_ms):
    q/k/v already on the GPU, no transfers in the timed region."""
    q, k, v = (t[0].detach().contiguous().cuda() for t in _qkv(model, n))
    model._attend(q, k, v)  # warmup / JIT
    cuda.synchronize()
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record()
    for _ in range(reps):
        model._attend(q, k, v)
    end.record()
    end.synchronize()
    return cuda.event_elapsed_time(start, end) / reps


def sdpa_kernel_ms(model, n, reps=5):
    """Device-resident torch SDPA time via CUDA events."""
    q, k, v = (t.detach().cuda() for t in _qkv(model, n))
    with torch.no_grad():
        F.scaled_dot_product_attention(q, k, v)  # warmup
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def bench_attention(classes, seq_lens, reps=5):
    """Attention-step time for each pipeline class at each sequence length."""
    times = {name: [] for name in classes}
    for n in seq_lens:
        for name, cls in classes.items():
            model = cls().eval()
            times[name].append(attention_ms(model, n, reps))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # drop cached N x N blocks between runs
    return times

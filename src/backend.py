from abc import ABC, abstractmethod
import numpy as np
try:
    import cupy as cp
except ImportError:  # ponytail: no CUDA locally, GPU backend only usable on remote
    cp = None

class Backend(ABC):
    @abstractmethod
    def matmul(self, a, b):
        pass
    
    @abstractmethod
    def softmax(self, a):
        pass

    @abstractmethod
    def scale(self, a):
        pass


class CPU(Backend):
    def matmul(self, a, b):
        return a @ b
    
    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    def scale(self, x, D):
        return x / np.sqrt(D)

class GPUv1(Backend):
    def __init__(self) -> None:
        self._matmul = cp.RawKernel(r'''
        extern "C" __global__
        void matmul(const float* x, const float* y, float* out, int m, int n, int p) {
            int i = blockIdx.y * blockDim.y + threadIdx.y;
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            float acc = 0.0f;
            if (i < m && j < p){
                for (int k = 0; k < n; ++k){
                    acc += x[i * n + k] * y[k * p + j];
                }
                out[i * p + j] = acc;
            }
        }
        ''', 'matmul')
    
    def matmul(self, a, b):
        a_gpu, b_gpu = cp.asarray(a, dtype=cp.float32), cp.asarray(b, dtype=cp.float32)
        m, n = a_gpu.shape
        n_b, p = b_gpu.shape
        out_gpu = cp.empty((m, p), dtype=cp.float32)

        assert(n == n_b)

        tpb = (16, 16)
        bpg = (
            (p + tpb[0] - 1) // tpb[0],
            (m + tpb[1] - 1) // tpb[1]
        )

        self._matmul(bpg, tpb, (a_gpu, b_gpu, out_gpu, cp.int32(m), cp.int32(n), cp.int32(p)))

        # ponytail: numpy in, numpy out — host round-trip per call, keep data on GPU if it gets slow
        return cp.asnumpy(out_gpu)

    def softmax(self, x):
        raise NotImplementedError  # ponytail: stub so GPUv1 is instantiable, matmul is the only GPU op so far

    def scale(self, x, D):
        raise NotImplementedError

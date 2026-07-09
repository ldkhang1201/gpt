from abc import ABC, abstractmethod
import numpy as np

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
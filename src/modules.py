import numpy as np
from numpy.random import default_rng

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.seed = 47
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def __call__(self, x):
        return default_rng(self.seed).random((x.shape[1], self.d_model))


class Linear:
    def __init__(self, in_features, out_features, bias=True) -> None:
        pass

    def __call__(self, x):
        raise NotImplementedError()


class Norm:
    def __init__(self, d_model) -> None:
        pass

    def __call__(self, x):
        raise NotImplementedError()


class ReLU:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        raise NotImplementedError()


class Sequential:
    def __init__(self, *modules) -> None:
        pass

    def __call__(self, x):
        raise NotImplementedError()

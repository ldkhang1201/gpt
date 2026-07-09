from modules import Embedding, Linear, Norm, ReLU, Sequential

## Single head attn, no gradient graph for now. Write entirely in n

class TransformerBase:
    def __init__(self, vocab_size=1000, d_model=512, d_ff=512 * 4):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)

        self.ffn = Sequential(
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model)
        )
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)

    def __call__(self, x):
        raise NotImplementedError()
from modules import Embedding, SelfAttention, FFN

## Single head attn, no gradient graph for now. Write entirely in n

class TransformerBase:
    def __init__(self, vocab_size=1000, d_model=512, d_ff=512 * 4):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.attn = SelfAttention(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        raise NotImplementedError()
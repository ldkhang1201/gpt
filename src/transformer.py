from modules import Embedding, SelfAttention

class Transformer:
    def __init__(self):
        self.embedding = Embedding()
        self.attn = SelfAttention()

    def forward(self, x):
        # x: [N]
        # ['heelo', 'im' , 'a', 'dog']
        x = self.embedding.forward(x) # [N, D]

        attn_out = self.attn.forward(x)

        return attn_out
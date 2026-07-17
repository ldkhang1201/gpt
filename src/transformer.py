from modules import Embedding, MultiHeadAttention

class Transformer:
    def __init__(self):
        self.attn = MultiHeadAttention()

    def forward(self, x):
        # x: [N]
        # ['heelo', 'im' , 'a', 'dog']

        return self.attn.forward(x)

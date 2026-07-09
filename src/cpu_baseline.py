from abstract import TransformerBase

class CpuPipeline(TransformerBase):
    def forward(self, x):
        # x: [N]
        # ['heelo', 'im' , 'a', 'dog']
        x = self.embedding.forward(x) # [N, D]

        attn_out = self.attn.forward(x)

        return self.ffn.forward(attn_out)
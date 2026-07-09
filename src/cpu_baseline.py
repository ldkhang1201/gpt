from abstract import TransformerBase
import numpy as np

class CpuPipeline(TransformerBase):
    def __call__(self, x):
        # x: [N]
        # ['heelo', 'im' , 'a', 'dog']
        x = self.embedding(x) # [N, D]
        print(x.shape)
        N, D = x.shape

        q = self.q_proj(x)  #  [x(N, D) . Wq(D, k)] -> [N, k]
        k = self.k_proj(x)  # [N, k] -> kT []
        v = self.v_proj(x)  # [N, k]

        attn_scores = np.matmul(q, k.transpose(-2, -1)) / D ** 0.5

        # x: N N
        attn_weights = np.softmax(attn_scores, dim=-1)
        attn_out = np.matmul(attn_weights, v)

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
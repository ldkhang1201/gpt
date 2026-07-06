from abstract import TransformerBase
from torch.profiler import record_function
import torch

class CpuPipeline(TransformerBase):
    def forward(self, x):
        # x: [N]
        # ['heelo', 'im' , 'a', 'dog']
        with record_function("1_embedding"):
            x = self.embedding(x) # [N, D]

        B, N, D = x.shape

        q = self.q_proj(x)  #  [x(N, D) . Wq(D, k)] -> [N, k]
        k = self.k_proj(x)  # [N, k] -> kT []
        v = self.v_proj(x)  # [N, k]

        with record_function("2_attention_total"):

            with record_function("2a_qk_matmul"):
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / D ** 0.5

            # x: N N

            with record_function("2b_softmax"):
                attn_weights = torch.softmax(attn_scores, dim=-1)

            with record_function("2c_value_weighted_sum"):
                attn_out = torch.matmul(attn_weights, v)

        x = self.norm1(x + attn_out)

        with record_function("3_ffn"):
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

        return x
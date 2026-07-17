import torch.nn.functional as F
from modules import SelfAttention

def test_matches_torch_sdpa():
    model = Transformer()
    x = torch.randint(0, 1000, (2, 16))  # [B, N] token ids

    with torch.no_grad():
        out = model(x)

        # same weights, attention via torch (SDPA scale = 1/sqrt(d_model) = your /D**0.5)
        h = model.embedding(x)
        attn = F.scaled_dot_product_attention(model.q_proj(h), model.k_proj(h), model.v_proj(h))
        h = model.norm1(h + attn)
        ref = model.norm2(h + model.ffn(h))

    torch.testing.assert_close(out, ref)

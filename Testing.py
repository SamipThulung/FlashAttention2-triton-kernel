import torch
from FlashAttention2_module import FlashAttention2 
import torch.nn.functional as F


# Testing to see if the output are correct
# Testing against the pytorch regular scaled_dot_product_attention
torch.manual_seed(0)

B = 2
N = 128
D = 64
tile_size = 32

Q = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
K = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
V = torch.randn(B, N, D, device="cuda", dtype=torch.float16)

O_triton = FlashAttention2.apply(Q, K, V, tile_size)


O_torch = F.scaled_dot_product_attention(
    Q, K, V,
    dropout_p=0.0,
    is_causal=True
)

print("Max absolute error:",
      (O_triton - O_torch).abs().max().item())

print("Mean absolute error:",
      (O_triton - O_torch).abs().mean().item())

print("Allclose:",
      torch.allclose(O_triton, O_torch.float(), atol=1e-2, rtol=1e-2))

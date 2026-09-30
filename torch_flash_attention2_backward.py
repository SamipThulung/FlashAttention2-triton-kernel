import torch

# A function for the backprop of FlashAttention2 using torch
# Main concept is Operator fusion and recomputation
@torch.compile
def flash_attention_bwd(
    Q, K, V, O, L,
    dO):
  Q, K, V, O, L = Q.float(), K.float(), V.float(), O.float(), L.float()
  
  batch, context_length, d = Q.shape
  scale = 1/(d ** 0.5)
  # precompute
  D = torch.sum(dO *O, axis=-1)
  # recompute score
  S = torch.matmul(Q, K.transpose(-1, -2)) * scale
  # mask
  causal_mask = torch.tril(
      torch.ones(
          (context_length, context_length),
          dtype=torch.bool,
          device=Q.device
      )
  )

  # Applying Mask
  S = S.masked_fill(~causal_mask, float("-inf"))

  # recompute P
  P = torch.exp(S - L.unsqueeze(-1))

  # calculate gradient
  dV = torch.matmul(P.transpose(-1, -2), dO)
  dP = torch.matmul(dO, V.transpose(-1, -2))
  dS = P * (dP - D.unsqueeze(-1))
  dQ = torch.matmul(dS, K) * scale
  dK = torch.matmul(dS.transpose(-1, -2), Q) * scale

  return dQ, dK, dV

import torch
from triton_flash_attention2_forward import flash_attention_fwd
from torch_flash_attention2_backward import flash_attention_bwd

# A torch autograd function that combines triton kernel (forward) and pytorch (backward)
class FlashAttention2(torch.autograd.Function):
  @staticmethod
  def forward(
        ctx,
        Q, K, V,
        tile_size_r
      ):

    B, context_length, d = Q.shape

    # Initializing the variables used to recursively update in the FlashAttention2 kernel
    LSE = torch.empty((B, context_length), dtype=torch.float32, device=Q.device)
    O = torch.empty_like(Q, dtype=torch.float32)

    # scale
    scale = 1 / (d ** 0.5)

    # total tiles
    num_tiles = triton.cdiv(context_length, tile_size_r)

    # grids and batch, 1 program == one q tile
    grid = (num_tiles, B)

    # Kernel launch
    flash_attention_fwd[grid](
        Q, K, V, O, LSE,
        # Strides
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1),
        # Shape
        context_length, d,
        # tensors
        tile_size_r,  tile_size_r,
        # scale
        scale, 
        # mask
        is_casual = True
    )

    # ctx saved for backward gradient calculation. 
    ctx.save_for_backward(Q, K, V, O, LSE)
    ctx.scale = scale

    return O
  
  @staticmethod
  def backward(ctx, dO):
    
    # get saved tensor
    Q, K, V, O, L = ctx.saved_tensors
    
    # run backward
    dQ, dK, dV = flash_attention_bwd(Q, K, V, O, L, dO)

    # gradient
    return dQ, dK, dV, None

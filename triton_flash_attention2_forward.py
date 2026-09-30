import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd(
      # 1. Poiters
      q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
      # Strides
      stride_qb, stride_qq, stride_qd,
      stride_kb, stride_kk, stride_kd,
      stride_vb, stride_vv, stride_vd,
      stride_ob, stride_oo, stride_od,
      stride_lb, stride_lr,
      # 2. Shape
      num_rows, D: tl.constexpr,
      # 3. tensors
      tile_size_qr: tl.constexpr,  tile_size_kvr: tl.constexpr,
      # 4. Scale
      scale,
      # 5. Mask
      is_casual: tl.constexpr
    ):
  # Index
  q_index = tl.program_id(0)
  batch_index = tl.program_id(1)

  # Calculating q offset for casual mask
  q_offset = q_index * tile_size_qr + tl.arange(0, tile_size_qr)

  # QKV ptrs
  q_block_ptr = tl.make_block_ptr(
      #Ptr
      q_ptr + batch_index * stride_qb,
      #Shape
      shape = (num_rows, D),
      #Strides
      strides = (stride_qq, stride_qd),
      #Offsets
      offsets = (q_index * tile_size_qr, 0),
      #Block size
      block_shape = (tile_size_qr, D),
      #order
      order = (1, 0)
  )

  k_block_ptr = tl.make_block_ptr(
      k_ptr + batch_index * stride_kb,
      shape = (num_rows, D),
      strides = (stride_kk, stride_kd),
      offsets = (0, 0),
      block_shape = (tile_size_kvr, D),
      order = (1, 0)
  )

  v_block_ptr = tl.make_block_ptr(
      v_ptr + batch_index * stride_vb,
      shape = (num_rows, D),
      strides = (stride_vv, stride_vd),
      offsets = (0,0),
      block_shape = (tile_size_kvr, D),
      order = (1, 0)
  )

  # Loading q tile
  q_tile = tl.load(q_block_ptr, boundary_check = (0, 1), padding_option="zero")

  # Variables recursively updating for Flash Attention
  # m = maximum score value
  m = tl.full((tile_size_qr,), value = float("-inf"), dtype = tl.float32) 
  # l = sum of softmax used for logsumexp for backprop
  l = tl.zeros((tile_size_qr,), dtype = tl.float32)
  # o = accumulated output
  o = tl.zeros((tile_size_qr, D), dtype = tl.float32)

  for i in range(0, num_rows, tile_size_kvr):
    # offset to calculated the boundary padding
    k_offset = i + tl.arange(0, tile_size_kvr)

    # loading KV tile
    k_tile = tl.load(k_block_ptr, boundary_check = (0, 1), padding_option="zero")
    v_tile = tl.load(v_block_ptr, boundary_check = (0, 1), padding_option="zero")

    # calculating QK.T
    score = tl.dot(q_tile, tl.trans(k_tile))
    score = score * scale

    # used to create a boolean tensor to remove invalid 0 padding that can contribute to score
    valid_k =  q_offset < num_rows

    if is_casual:
      # mask for the forward token
      casual_mask = (
          q_offset[:, None] >= k_offset [None, : ]
      )
      mask = (casual_mask & valid_k[None, :])

    else:
      mask = valid_k[None, :]

    score = tl.where(
        mask,
        score,
        float("-inf")
        )
    
    # calculate the max of the score and compare the max_old vs the new max
    max_ = tl.max(score, axis=-1)
    maximum = tl.maximum(max_, m)

    # stablize the score by substracting max value from the row and summing the value
    P = tl.exp(score-maximum[:, None])
    row_sum = tl.sum(P, axis=-1)

    # coffecient of the flashattention2 formula for updating the logexpsum 
    # alpha = e**(m_old - max)
    alpha = tl.exp(m - maximum)

    # recursively updating the l and o
    l = alpha * l + row_sum
    o = alpha[:, None] * o + tl.dot(P, v_tile.to(tl.float32))

    # setting the m and ptr for the next iteration
    m = maximum
    k_block_ptr = tl.advance(k_block_ptr, (tile_size_kvr, 0))
    v_block_ptr = tl.advance(v_block_ptr, (tile_size_kvr, 0))

  # Normalizing the output and calculating the logsumexp (used for backward)
  o = o / l[:, None]
  lse = m + tl.log(l)

  # intializing the output block and lse block
  out_block_ptr = tl.make_block_ptr(
      out_ptr + batch_index * stride_ob,
      shape = (num_rows, D),
      strides = (stride_oo, stride_od),
      offsets = (q_index * tile_size_qr, 0),
      block_shape = (tile_size_qr, D),
      order = (1, 0)
  )
  lse_block_ptr = tl.make_block_ptr(
      lse_ptr + batch_index * stride_lb,
      shape = (num_rows,),
      strides = (stride_lr,),
      offsets = (q_index * tile_size_qr,),
      block_shape = (tile_size_qr,),
      order = (0,),
  )

  # Storing for the ctx
  tl.store(out_block_ptr, o, boundary_check = (0,1))
  tl.store(lse_block_ptr, lse, boundary_check = (0,))



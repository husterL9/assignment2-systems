import math
from typing import Any
from einops import rearrange
from torch import Tensor
from torch.autograd.function import FunctionCtx
from typing import cast
import triton
import triton.language as tl
import torch
@triton.jit 
def flash_fwd_kernel( 
    Q_ptr, K_ptr, V_ptr, 
    O_ptr, L_ptr, 
    stride_qb, stride_qq, stride_qd, 
    stride_kb, stride_kk, stride_kd, 
    stride_vb, stride_vk, stride_vd, 
    stride_ob, stride_oq, stride_od, 
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, 
    scale, 
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices  
    query_tile_index = tl.program_id(0) 
    batch_index = tl.program_id(1)  

    # Offset each pointer with the corresponding batch index 
    # multiplied with the batch stride for each tensor 
    Q_block_ptr = tl.make_block_ptr( 
        Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),  
    )
    K_block_ptr=tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS,D),
        strides=(stride_kk,stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D), 
        order=(1,0),
    )
    V_block_ptr=tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS,D),
        strides=(stride_vk,stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D), 
        order=(1,0),
    )
    O_block_ptr=tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES,D),
        strides=(stride_oq,stride_od),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, D), 
        order=(1,0),
    )
    L_block_ptr=tl.make_block_ptr(
        L_ptr + batch_index *stride_lb ,
        shape=(N_KEYS,D),
        strides=(stride_lq,),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE,), 
        order=(1,0),
    )
    # Initialize a buffer to write to
    output_O = tl.empty((N_QUERIES,D), dtype=tl.float32)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the current block pointer
        # Since Q_TILE_SIZE might not divide N_QUERIES and K_TILE_SIZE might not divide N_KEYS,
        # we need boundary checks for both dimensions
        #(Q_TILE_SIZE,D)
        Q_tile=tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
        #(K_TILE_SIZE,D)
        K_tile=tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        V_tile=tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")

        # Compute the weighted sum of the row.
        #(batch,Bq,Bk) Compute tile of pre-softmax attention scores 
        S_tile=tl.matmul(Q_tile,tl.transpose(K_tile,-1,-2))*scale
        output += tl.sum(row * weight[None, :], axis=1)
        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # Move by D_TILE_SIZE

    return

class MyFlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
        return 
    
    @staticmethod 
    def backward(ctx: Any,*grad_outputs) -> Any:
        saved = cast(tuple[Tensor, Tensor, Tensor, Tensor, Tensor], ctx.saved_tensors)
        L, Q, K, V, O = saved

        raise NotImplementedError
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
        shape=(D,N_KEYS),
        strides=(stride_kd,stride_kk),
        offsets=(0,0),
        block_shape=(D, K_TILE_SIZE), 
        order=(0,1),
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
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), 
        order=(1,0),
    )
    L_block_ptr=tl.make_block_ptr(
        L_ptr + batch_index *stride_lb ,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE,), 
        order=(1,0),
    )
    # Initialize a buffer to write to
    output_O = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    l=tl.zeros((Q_TILE_SIZE,),dtype=tl.float32)
    m=tl.full((Q_TILE_SIZE,),float("-inf"),dtype=tl.float32)
    #(Q_TILE_SIZE,D)
    Q_tile=tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the current block pointer
        # Since Q_TILE_SIZE might not divide N_QUERIES and K_TILE_SIZE might not divide N_KEYS,
        # we need boundary checks for both dimensions

        #(K_TILE_SIZE,D)
        K_tile=tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        V_tile=tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")

        # Compute the weighted sum of the row.
        #(Bq,Bk) Compute tile of pre-softmax attention scores 
        S_tile=tl.dot(Q_tile,K_tile)*scale
        rowmax = tl.max(S_tile, axis=1)
        m_new = tl.maximum(m, rowmax)
        P_tile = tl.exp(S_tile - m_new[:, None])
        P_tile = P_tile.to(V_tile.dtype)
        # (Q_TILE_SIZE,)
        alpha = tl.exp(m - m_new)
        m = m_new  
        # (Q_TILE_SIZE,)                  
        l = alpha * l + tl.sum(P_tile, axis=1)      
        # (Q_TILE_SIZE, Dv)
        output_O = alpha[:, None] * output_O
        output_O = tl.dot(P_tile, V_tile, acc=output_O)
        # Move the pointers to the next tile.
        K_block_ptr = K_block_ptr.advance((0,K_TILE_SIZE))
        V_block_ptr=V_block_ptr.advance((K_TILE_SIZE,0))
    
    scale_O = 1.0 / l[:, None]   # (Q_TILE_SIZE, 1)
    output_O = output_O * scale_O      # (Q_TILE_SIZE, Dv)        
    logsumexp_l = tl.log(l)      # (Q_TILE_SIZE,)
    L = m + logsumexp_l        # (Q_TILE_SIZE,)
    output_O = output_O.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, output_O, boundary_check=(0,))
    tl.store(L_block_ptr,L,boundary_check=(0,))

class MyFlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
        Bq = 16   # query tile rows, >=16
        Bk = 16   # key tile cols, >=16
        Tq = (Q.shape[-2] + Bq - 1) // Bq
        q2 = rearrange(Q, "... n d -> (...) n d")   # (B_total, n_queries, D)
        k2 = rearrange(K, "... n d -> (...) n d")   # (B_total, n_keys, D)
        v2 = rearrange(V, "... n dv -> (...) n dv")   # (B_total, n_keys, D)
        dim_batch=q2.shape[0]
        dim_d=Q.shape[-1]
        dim_n_queries=Q.shape[-2]
        dim_k_keys=K.shape[-2]
        dim_dv=V.shape[-1]
        output_O_dims=(*Q.shape[:-1],dim_dv)
        output_L_dims=Q.shape[:-1]
        O_flatten= torch.empty((*q2.shape[:-1],dim_dv), device=Q.device,dtype=Q.dtype)
        L_flatten= torch.empty(q2.shape[:-1], device=Q.device,dtype=torch.float32)
        scale_attention = 1.0 / math.sqrt(dim_d)
        flash_fwd_kernel[(Tq,dim_batch)](
            q2,k2,v2,O_flatten,L_flatten,
            q2.stride(0),q2.stride(1),q2.stride(2),
            k2.stride(0),k2.stride(1),k2.stride(2),
            v2.stride(0),v2.stride(1),v2.stride(2),
            O_flatten.stride(0),O_flatten.stride(1),O_flatten.stride(2),
            L_flatten.stride(0),L_flatten.stride(1),
            dim_n_queries,dim_k_keys,
            scale_attention,
            dim_d,Bq,Bk # type: ignore[arg-type]
        )
        O=O_flatten.reshape(output_O_dims)
        L=L_flatten.reshape(output_L_dims)
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod 
    def backward(ctx: Any,*grad_outputs) -> Any:
        saved = cast(tuple[Tensor, Tensor, Tensor, Tensor, Tensor], ctx.saved_tensors)
        L, Q, K, V, O = saved

        raise NotImplementedError
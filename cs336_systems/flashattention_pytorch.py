import math
from typing import Any
from einops import rearrange
from torch import Tensor
from torch.autograd.function import FunctionCtx
from typing import cast

import torch
class MyFlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
        Bq = 16   # query tile rows, >=16
        Bk = 16   # key tile cols, >=16
        Tq = (Q.shape[-2] + Bq - 1) // Bq
        Tk = (K.shape[-2] + Bk - 1) // Bk
        # q: (..., n_queries, D)
        q2 = rearrange(Q, "... n d -> (...) n d")   # (B_total, n_queries, D)
        # k,v: (..., n_keys, D)
        k2 = rearrange(K, "... n d -> (...) n d")   # (B_total, n_keys, D)
        v2 = rearrange(V, "... n dv -> (...) n dv")   # (B_total, n_keys, D)
        dim_batch=q2.shape[0]
        dim_d=Q.shape[-1]
        # dim_n_queries=Q.shape[-2]
        dim_dv=V.shape[-1]
        output_O_dims=(*Q.shape[:-1],dim_dv)
        output_L_dims=Q.shape[:-1]
        O_flatten= torch.empty((*q2.shape[:-1],dim_dv), device=Q.device,dtype=Q.dtype)
        L_flatten= torch.empty(q2.shape[:-1], device=Q.device,dtype=Q.dtype)
        scale_attention = 1.0 / math.sqrt(dim_d)
        for i in range(0,Tq):
            # load Q_i:(batch, Tq*Bq:(Tq+1)*Bq, d)
            Q_i=q2[:,i*Bq:(i+1)*Bq,:]
            # initial O_i,l_i,m_i
            O_i=torch.zeros((dim_batch,Bq,dim_dv),device=Q_i.device)
            l_i=torch.zeros((dim_batch,Bq),device=Q_i.device)
            m_i = torch.full((dim_batch, Bq), float("-inf"), device=Q_i.device, dtype=Q_i.dtype)
            
            for j in range(0,Tk):
                # load K_j,V_j
                K_j=k2[:,j*Bk:(j+1)*Bk,:]
                V_j=v2[:,j*Bk:(j+1)*Bk,:]
                #(batch,Bq,Bk) Compute tile of pre-softmax attention scores 
                S_i=torch.matmul(Q_i,torch.transpose(K_j,-1,-2))*scale_attention
                # (batch,Bq)
                rowmax = torch.max(S_i, dim=-1).values
                # (batch, Bq)
                m_i_new = torch.maximum(m_i, rowmax)
                # (batch,Bq,Bk)
                P_i = torch.exp(S_i - m_i_new.unsqueeze(-1))
                # (batch,Bq)
                l_i = torch.exp(m_i - m_i_new) * l_i + P_i.sum(dim=-1)
                # (batch, Bq)
                alpha = torch.exp(m_i - m_i_new)
                # (B_total, Bq, 1)          
                alpha = alpha.unsqueeze(-1)               
                # (batch,Bq,dv)
                O_i=alpha * O_i+torch.matmul(P_i, V_j)
                m_i = m_i_new                
            scale_O_i=1/l_i.unsqueeze(-1)
            logsumexp_l_i=torch.log(l_i)
            O_i=O_i*scale_O_i
            L_i=m_i+logsumexp_l_i
            q_slice = slice(i * Bq, (i + 1) * Bq)
            O_flatten[:, q_slice, :] = O_i
            L_flatten[:, q_slice] = L_i
        O=O_flatten.reshape(output_O_dims)
        L=L_flatten.reshape(output_L_dims)
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod 
    def backward(ctx: Any,*grad_outputs) -> Any:
        saved = cast(tuple[Tensor, Tensor, Tensor, Tensor, Tensor], ctx.saved_tensors)
        L, Q, K, V, O = saved

        raise NotImplementedError
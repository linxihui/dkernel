import torch
from torch import Tensor
from typing import Tuple, Dict, Union
from s2attn.ops.sparse_attn_fwd import _forward
from s2attn.ops.sparse_attn_bwd import _backward


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                sm_scale: int,
                layout_csr: Tuple[Tensor, Tensor],
                layout_csc: Tuple[Tensor, Tensor],
                seq_dim: int,
                kwargs: Dict) -> Tensor:
        '''
        :param q, k, v: [batch, n_heads, seq_len, model_dim]. len of q is allowed to be different than k/v.
        TODO: 1. allow change seqlen in backward
              2. allow cross-attn, bi-directional
        '''
        # shape constraints
        ctx.layout_csr = layout_csr
        ctx.layout_csc = layout_csc
        return _forward(ctx, q, k, v, sm_scale, layout_csr, seq_dim=seq_dim, **kwargs)

    @staticmethod
    def backward(ctx, do: Tensor) -> Tuple[Union[Tensor, None], ...]:
        q, k, v, o, l, m = ctx.saved_tensors
        assert q.size(ctx.seq_dim) == k.size(ctx.seq_dim), \
            "Backward supports only when q/k/v have the same sequence length."
        return _backward(ctx, do)[:3] + (None, None, None, None, None)


__all__ = ["_sparse_attention"]
import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, Union
from dkernel.ops.sparse_attn_fwd import _forward
from dkernel.ops.sparse_attn_bwd import _backward
from dkernel.ops.sparse_attn_inference import padded_sparse_attn, varlen_sparse_attn


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                sm_scale: int,
                layout_csr: Tuple[Tensor, Tensor, int, int],
                layout_csc: Tuple[Tensor, Tensor, int, int],
                seq_dim: Optional[int],
                inf_kwargs: Dict,
                kwargs: Dict) -> Tensor:
        '''
        :param q, k, v: [batch, n_heads, seq_len, model_dim]. len of q is allowed to be different than k/v.
        TODO: 1. allow change seqlen in backward
              2. allow cross-attn, bi-directional
        '''
        # shape constraints
        ctx.layout_csr = layout_csr
        ctx.layout_csc = layout_csc
        seq_dim = 1 if seq_dim is None else seq_dim
        max_seqlen = kwargs.get("max_seqlen", None)
        # causal = kwargs.get('causal', True)

        lse = None
        assert q.dim() in (3, 4)
        # need_backwards = q.requires_grad() or k.requires_grad() or v.requires_grad()
        if q.dim() == 3:
            # assert seq_dim is None, f"cannot set seq_dim for variable length inputs"
            assert inf_kwargs.get("cu_seqlen_k", None) is not None
            ctx.support_backward = False
            ctx.message = "Currently does not suppoort variable length inputs. WIP."
            num_heads = q.size(1)
            num_kv_heads = k.size(1)
            # assert causal, "To be supported."
            output = varlen_sparse_attn(q, k, v,
                                      inf_kwargs["cu_seqlen_k"],
                                      inf_kwargs.get("cu_seqlen_q", None),
                                      sm_scale,
                                      layout_csr,
                                      max_seqlen=max_seqlen)
        else:
            left_paddings = inf_kwargs.get("left_paddings", None)
            seqlens =  inf_kwargs.get("seqlens", None)
            has_paddings = (left_paddings is not None) or (seqlens is not None)

            if has_paddings:
                ctx.support_backward = False
                ctx.message = "Currently does not support paddings in inputs. WIP."
                output = padded_sparse_attn(q, k, v,
                                          sm_scale,
                                          layout_csr,
                                          left_paddings=left_paddings,
                                          seqlens=seqlens,
                                          seq_dim=seq_dim,
                                          max_seqlen=max_seqlen)
            else:
                ctx.support_backward = True
                ctx.message = ""
                assert k.size(seq_dim) <= max_seqlen
                output, lse = _forward(ctx, q, k, v, sm_scale, layout_csr, seq_dim=seq_dim, **kwargs)
        return output, lse

    @staticmethod
    def backward(ctx, do: Tensor, dlse: Optional[Tensor]) -> Tuple[Union[Tensor, None], ...]:
        """
        Args:
            do: output gradient
            dlse: logsumexp gradient. It will be None if return_lse is False.
        """

        assert ctx.support_backward, ctx.message
        q, k, v, o, l, m = ctx.saved_tensors
        assert q.size(ctx.seq_dim) == k.size(ctx.seq_dim), \
            "Backward supports only when q/k/v have the same sequence length."
        return _backward(ctx, do, dlse)[:3] + (None, None, None, None, None, None)


__all__ = ["_sparse_attention"]
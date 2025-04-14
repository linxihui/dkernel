import math
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
                k: Optional[Tensor],
                v: Optional[Tensor],
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

        (q, k, v), (packed_shape, packed_mode, split_qkv) \
            = get_qkv_and_pack_mode(q, k, v, split_qkv=kwargs.get("split_qkv", None))

        ctx.layout_csr = layout_csr
        ctx.layout_csc = layout_csc
        ctx.packed_shape = packed_shape
        ctx.packed_mode = packed_mode
        ctx.split_qkv = split_qkv

        seq_dim = 1 if seq_dim is None else seq_dim
        max_seqlen = kwargs.get("max_seqlen", None)
        # causal = kwargs.get('causal', True)

        lse = None

        hdim = 3 - (seq_dim or 1) if q.dim() == 4 else 1
        layout_csr_crow = layout_csr[0]
        assert layout_csr_crow.size(0) in [1, k.size(hdim), q.size(hdim)], \
            (f"Input (q, k) have ({q.size(hdim)}, {k.size(hdim)}) heads"
             f"but the number of heads in the sparse pattern is {layout_csr_crow.size(0)}")

        sm_scale = sm_scale or 1. / math.sqrt(float(q.size(-1)))

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
            # check seqlen
            if q.dim() == 4:
                assert k.size(seq_dim) <= max_seqlen, \
                    (f"Input length {k.size(seq_dim)} is larger "
                    f"than the maximum length allowed ({max_seqlen})")
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
        q, k = ctx.saved_tensors[:2]
        # assert q.size(ctx.seq_dim) == k.size(ctx.seq_dim), \
        #     "Backward supports only when q/k/v have the same sequence length."

        dq, dk, dv = None, None, None
        if ctx.packed_mode == 'packed_qkv':
            dqkv = q.new_empty(ctx.packed_shape)
            dq, dk, dv = ctx.split_qkv(dqkv)

        elif ctx.packed_mode == 'packed_kv':
            dkv = k.new_empty(ctx.packed_shape)
            dk, dv = ctx.split_qkv(dkv)

        dq, dk, dv = _backward(ctx, do, dlse, dq=dq, dk=dk, dv=dv)[:3]

        if ctx.packed_mode == 'packed_qkv':
            grads = (dqkv, None, None)
        elif ctx.packed_mode == 'packed_kv':
            grads = (dq, dkv, None)
        else:
            grads = (dq, dk, dv)
        return grads + (None, ) * 6


def get_qkv_and_pack_mode(q: Tensor, k: Optional[Tensor], v: Optional[Tensor],
                          split_qkv: Optional[callable] = None
                          ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[tuple, str]]:
    """
    """
    qkv_pack_mode = 'split_qkv'
    packed_shape = None
    if v is None:
        # qkv packed or kv packed
        assert split_qkv is not None, "Callable `split_qkv` must be specified if qkv or kv are packed."
        # if split_qkv is None:
        #     split_qkv = lambda qkv: qkv.unbind(0)

        if k is None: # qkv packed
            packed_shape = q.shape
            q, k, v = split_qkv(q)
            qkv_pack_mode = 'packed_qkv'
        else: # kv packed
            packed_shape = k.shape
            k, v = split_qkv(k)
            qkv_pack_mode = 'packed_kv'
    return (q, k, v), (packed_shape, qkv_pack_mode, split_qkv)


__all__ = ["_sparse_attention"]
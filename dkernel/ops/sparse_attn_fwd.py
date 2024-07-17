import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple, Optional
from s2attn.utils import multiple_of, is_hip

import ipdb
_b = ipdb.set_trace



@triton.jit
def _fwd_one_kv_block(
    bid_n, micro_M,
    q, q2, acc, acc2, m_i, l_i, sm_scale,
    kt_ptrs, v_ptrs,
    offs_m, offs_n,
    stride_kn, stride_vn, stride_kd,
    PAST_LEN, N_CTX,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAST_N_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    ):

    start_n = bid_n * BLOCK_N
    if LAST_N_BLOCK:
        kt = tl.load(kt_ptrs + start_n * stride_kn, mask=offs_n[None, :] + start_n < N_CTX)
    else:
        kt = tl.load(kt_ptrs + start_n * stride_kn)

    qk = tl.dot(q, kt) # TODO: (H, M, D), (D, N) -> (H, M, N)

    if NUM_DBLOCKS >= 2:
        if LAST_N_BLOCK:
            kt = tl.load(kt_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd, mask=offs_n[None, :] + start_n < N_CTX)
        else:
            kt = tl.load(kt_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd)
        qk += tl.dot(q2, kt)
    qk *= sm_scale

    if MERGED_Q:
        qk += tl.where(tl.arange(0, BLOCK_M)[:, None] < micro_M, 0, -1e6)
    if CAUSAL & LAST_N_BLOCK:
        qk += tl.where(offs_m[:, None] + PAST_LEN >= (start_n + offs_n[None, :]), 0, -1e6)

    # flash-attn2
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])  # unormalized
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    # update m_i
    m_i = m_ij
    l_i = l_i * alpha + l_ij  # the normalizer of p above

    p = p.to(dtype)
    if LAST_N_BLOCK:
        v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < N_CTX)
    else:
        v = tl.load(v_ptrs + start_n * stride_vn)
    acc += tl.dot(p, v)

    if NUM_DBLOCKS >= 2:
        acc2 = acc2 * alpha[:, None]
        if LAST_N_BLOCK:
            v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_kd, mask=offs_n[:, None] + start_n < N_CTX)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_kd)
        acc2 += tl.dot(p, v)

    return acc, acc2, m_i, l_i


fwd_configs = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in ([1] if is_hip() else [1, 2, 3, 4, 5, 7])\
    for w in [4, 8] \
]


@triton.autotune(fwd_configs,
                 key=["N_CTX_FOR_AUTOTUNE", "BLOCK_DMODEL", "NUM_DBLOCKS",
                      "BLOCK_M", "BLOCK_N", "Q_ROUNDED_LEN"]
                )
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    PAST_LEN,
    Q_ROUNDED_LEN,
    N_CTX_FOR_AUTOTUNE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    INFERENCE: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
):
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    # initialize offsets

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    # off_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    # Initialize pointers to Q, K, V
    q_ptrs  = Q + off_q
    kt_ptrs = K + off_k
    v_ptrs  = V + off_v
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if NUM_DBLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:
        acc2 = 0

    sm_scale *= (
        1.44269504  # 1/log2 as we use base2 for exponential and logarithm
    )

    ## handle chunked prefix filling
    # if INFERENCE:
    #     if tl.program_id(0) == 0:
    #         past_left = PAST_LEN % BLOCK_M
    #         q_offs_m =  (tl.arange(0, BLOCK_M) - past_left)
    #         start_m = PAST_LEN // BLOCK_M
    #     else:
    #         start_m = PAST_LEN // BLOCK_M + tl.program_id(0)
    #         first_q_tokens = BLOCK_M - (PAST_LEN) % BLOCK_M
    #         q_offs_m = first_q_tokens + tl.arange(0, BLOCK_M)
    #     q_ptrs = Q + q_offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    #     q = tl.load(q_ptrs, mask=(q_offs_m >= 0) & (q_offs_m < Q_LEN))
    #     if NUM_DBLOCKS >= 2:
    #         q2 = tl.load(q_ptrs + BLOCK_DMODEL + stride_qd,
    #                      mask=(q_offs_m >= 0) & (q_offs_m < Q_LEN))

    # load q: it will stay in SRAM throughout
    q2 = 0
    if EVEN_M_BLOCK:
        q = tl.load(q_ptrs)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m[:, None] < Q_LEN)

    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    # start_l = tl.load(layout_ptr).to(tl.int32)
    # end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)
    start_l, end_l = tl.load(layout_ptr + tl.arange(0, 2) * layout_crow_stride_m).split()

    # loop over k, v and update accumulator
    non_diag_end = tl.maximum(end_l - NUM_DIAG_BLOCKS, start_l)
    for col_idx_idx in range(start_l, non_diag_end):
        # tl.device_print('# col_idx_idx:', col_idx_idx)
        if MERGED_Q:
            bid_n = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + 1).to(tl.int32)
        else:
            bid_n = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = 0
        acc, acc2, m_i, l_i = \
            _fwd_one_kv_block(bid_n, micro_M,
                q, q2, acc, acc2, m_i, l_i, sm_scale,
                kt_ptrs, v_ptrs,
                offs_m, offs_n,
                stride_kn, stride_vn, stride_kd,
                PAST_LEN, N_CTX,
                dtype=Q.dtype.element_ty,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                LAST_N_BLOCK=False,
                CAUSAL=True,
                BLOCK_DMODEL=BLOCK_DMODEL,
                NUM_DBLOCKS=NUM_DBLOCKS,
                MERGED_Q=MERGED_Q
                )

    # diag
    for col_idx_idx in range(non_diag_end, end_l):
        # tl.device_print('> col_idx_idx:', col_idx_idx)
        if MERGED_Q:
            bid_n = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + 1).to(tl.int32)
        else:
            bid_n = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = 0
        acc, acc2, m_i, l_i = \
            _fwd_one_kv_block(bid_n, micro_M,
                q, q2, acc, acc2, m_i, l_i, sm_scale,
                kt_ptrs, v_ptrs,
                offs_m, offs_n,
                stride_kn, stride_vn, stride_kd,
                PAST_LEN, N_CTX,
                dtype=Q.dtype.element_ty,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                LAST_N_BLOCK=True,
                CAUSAL=True,
                BLOCK_DMODEL=BLOCK_DMODEL,
                NUM_DBLOCKS=NUM_DBLOCKS,
                MERGED_Q=MERGED_Q
                )

    # flash-attn 2
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    if NUM_DBLOCKS >= 2:
        acc2 = acc2 / l_i[:, None]

    # write back l and m
    if not INFERENCE:
        l_ptrs = L + off_hz * Q_ROUNDED_LEN + offs_m
        m_ptrs = M + off_hz * Q_ROUNDED_LEN + offs_m
        tl.store(l_ptrs, l_i)
        tl.store(m_ptrs, m_i)
    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc,  mask=offs_m[:, None] < Q_LEN)
    if NUM_DBLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_DMODEL * stride_od, acc2,  mask=offs_m[:, None] < Q_LEN)


def _forward(ctx,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            sm_scale: int,
            layout_csr: Tuple[Tensor, Tensor],
            seq_dim: int=2,
            inference: int=None,
            out:Optional[Tensor]=None,
            d_splits: Optional[int]=None
            ) -> Tensor:
    """
    :param q, k, v: shape=(batch, n_heads, seq_len, head_size) if seq_len=2 else (batch, seq_len, n_heads, head_size).
        Length of q is allowed to be different than k/v for decoding.
        Note: head_size should be on of [64, 128, 256].
    :param sm_scale: softmax scale in attention.
    :param layout_csr: Tuple of (layout_crow_indices, layout_col_indices, int, int):
        same as CSR.crow_indices, and CSR.col_indices used to preresent a sparse tensor.
        Each element represent a block, i.e, all elements in a block to be attentdd, or not attended at all.
    :param seq_dim: which dimension of q/k/v is a dimenion of sequence length.
    :param inference: if at inference mode, i.e., do not save addition info needed in the backward pass.
    :param out: if provided, output will be saved to.
    :param d_splits: None, 1 or 2.  None=1 if head_dim=64 else 2.
    """
    assert seq_dim in [2, 1]
    hdim = 3 - seq_dim
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert q.shape[-1] in [64, 128, 256]
    assert k.shape[seq_dim] == v.shape[seq_dim]

    layout_crow_indices, layout_col_indices, block_m, block_n = layout_csr
    qlen, klen = q.size(seq_dim), k.size(seq_dim)
    if qlen != klen:
        # assert klen > qlen
        assert qlen == 1, \
            ("If q has differnt seq length that k and v, q should only have 1 token per batch for decoding,"
             f"but got q length = {qlen}.")
        layout_crow_indices = layout_crow_indices[..., (klen - qlen) // block_m:]

    # TODO: do I need to set o.requires_grad to True explicitely?
    o = out if out is not None else q.new_empty(q.shape).contiguous()

    merged_q = False
    if layout_col_indices.dim() == 3:
        assert layout_col_indices.size(2) == 2 and layout_col_indices.stride(2) == 1
        merged_q = True

    # NOTE: For inference with qlen=1, need to use the origin block_m, as lyaout_col_indices does not get trimmed.
    kwargs = {
        "MERGED_Q": merged_q,
        "NUM_DIAG_BLOCKS": max(1, block_m // block_n),
        }

    if inference is None:
        inference = (not q.requires_grad) and (not k.requires_grad)  and (not v.requires_grad)

    if inference and qlen < block_m:
        # change block_m:
        # For prefixing chunking TODO: smaller?  block_m = max(next_power_of_2(qlen) // 4, 16)? break 32 -> 2 blocks
        # Need to multiply layout_crow_indices, how to handle micro_M?
        block_m = max(triton.next_power_of_2(qlen), 16)

    grid = (triton.cdiv(q.shape[seq_dim], block_m), q.shape[0] * q.shape[hdim])
    q_rounded_len = grid[0] * block_m

    if inference:
        L = m = q.new_empty((1,))
    else:
        L = torch.zeros((q.shape[0] * q.shape[hdim], q_rounded_len), device=q.device, dtype=torch.float32)
        m = torch.zeros((q.shape[0] * q.shape[hdim], q_rounded_len), device=q.device, dtype=torch.float32)

    if layout_crow_indices.dim() == 1:
        layout_crow_indices = layout_crow_indices[None].expand(q.shape[hdim] , -1)
        layout_col_indices = layout_col_indices[None].expand((q.shape[hdim],) + layout_col_indices.shape)

    if d_splits is None:
        block_d = max(64, q.shape[-1] // 2) # allow larger num_stages to split along D
        d_splits = q.shape[-1] // block_d
    else:
        assert d_splits in [1, 2]
        block_d = q.shape[-1] // d_splits


    _fwd_kernel[grid](
        q, k, v, sm_scale,
        layout_crow_indices,
        layout_col_indices,
        layout_crow_indices.stride(0), layout_crow_indices.stride(1),
        layout_col_indices.stride(0), layout_col_indices.stride(1),
        L, m,
        o,
        q.stride(0), q.stride(hdim), q.stride(seq_dim), q.stride(3),
        k.stride(0), k.stride(hdim), k.stride(seq_dim), k.stride(3),
        v.stride(0), v.stride(hdim), v.stride(seq_dim), v.stride(3),
        o.stride(0), o.stride(hdim), o.stride(seq_dim), o.stride(3),
        q.shape[0], q.shape[hdim], k.shape[seq_dim],
        k.shape[seq_dim] - q.shape[seq_dim], # PAST_LEN
        q_rounded_len,
        multiple_of(qlen, 1024), # N_CTX_FOR_AUTOTUNE, TODO: inference??
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_d,
        EVEN_M_BLOCK=qlen % block_m == 0,
        EVEN_N_BLOCK=klen % block_n == 0 ,
        INFERENCE=inference,
        NUM_DBLOCKS=d_splits,
        **kwargs
    )
    if inference:
        L, m = None, None

    ctx.save_for_backward(q, k, v, o, L, m)
    ctx.block_d = block_d
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.merged_q = merged_q
    ctx.seq_dim = seq_dim
    ctx.hdim = hdim
    ctx.kwargs = kwargs

    return o

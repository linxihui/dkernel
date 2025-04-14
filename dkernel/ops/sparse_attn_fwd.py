import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple, Optional
from dkernel.utils import multiple_of, is_hip

# TODO: support different head_dim for k, v
# Idea: support BLOCK_VD, BLOCK_VD2, when NUM_VD_BLOCKS == 2

@triton.jit
def _load_with_1d_mask(ptrs,
                       left_d, right_d,
                       EVEN_D: tl.constexpr,
                       other=None,
                       ):
    if EVEN_D:
        mask = None
    else:
        mask = (left_d < right_d)
    return tl.load(ptrs, mask=mask, other=other)


@triton.jit
def _store_with_1d_mask(ptrs, values,
                       left_d, right_d,
                       EVEN_D: tl.constexpr,
                       ):
    if EVEN_D:
        mask = None
    else:
        mask = (left_d < right_d)
    tl.store(ptrs, values, mask=mask)


@triton.jit
def _load_with_2d_mask(ptrs,
                       left_d1, right_d1,
                       left_d2, right_d2,
                       EVEN_D1: tl.constexpr,
                       EVEN_D2: tl.constexpr,
                       other=None,
                       ):
    need_mask_1: tl.constexpr = ~EVEN_D1
    need_mask_2: tl.constexpr = ~EVEN_D2
    if need_mask_1:
        mask_1 = (left_d1 < right_d1)[:, None]
    if need_mask_2:
        mask_2 = (left_d2 < right_d2)[None, :]

    if need_mask_1:
        if need_mask_2:
            result = tl.load(ptrs, mask=mask_1 & mask_2, other=other)
        else:
            result = tl.load(ptrs, mask=mask_1, other=other)
    else:
        if need_mask_2:
            result = tl.load(ptrs, mask=mask_2, other=other)
        else:
            result = tl.load(ptrs, other=other)
    return result


@triton.jit
def _store_with_2d_mask(ptrs, values,
                       left_d1, right_d1,
                       left_d2, right_d2,
                       EVEN_D1: tl.constexpr,
                       EVEN_D2: tl.constexpr,
                       ):
    need_mask_1: tl.constexpr = ~EVEN_D1
    need_mask_2: tl.constexpr = ~EVEN_D2
    if need_mask_1:
        mask_1 = (left_d1 < right_d1)[:, None]
    if need_mask_2:
        mask_2 = (left_d2 < right_d2)[None, :]

    if need_mask_1:
        if need_mask_2:
            tl.store(ptrs, values, mask=mask_1 & mask_2)
        else:
            tl.store(ptrs, values, mask=mask_1)
    else:
        if need_mask_2:
            tl.store(ptrs, values, mask=mask_2)
        else:
            tl.store(ptrs, values)


@triton.jit
def _fwd_one_kv_block(
    bid_n, micro_M,
    q, q2, acc, acc2, m_i, l_i, sm_scale,
    kt_ptrs, kt_ptrs2, v_ptrs, v_ptrs2,
    offs_m, offs_n, offs_d, offs_d2,
    stride_kn, stride_vn, stride_kd,
    PAST_LEN, N_CTX, D,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAST_N_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    EVEN_D_BLOCK: tl.constexpr,
    MERGED_Q: tl.constexpr,
    ):


    EVEN_D1_BLOCK: tl.constexpr = EVEN_D_BLOCK | (NUM_DBLOCKS >= 2)
    EVEN_D2_BLOCK: tl.constexpr = EVEN_D_BLOCK

    start_n = bid_n * BLOCK_N
    kt = _load_with_2d_mask(kt_ptrs + start_n * stride_kn,
                            offs_d, D,
                            offs_n, N_CTX - start_n,
                            ~LAST_N_BLOCK,
                            EVEN_D1_BLOCK)
    qk = tl.dot(q, kt) # TODO: (H, M, D), (D, N) -> (H, M, N)

    if NUM_DBLOCKS >= 2:
        kt = _load_with_2d_mask(kt_ptrs2 + start_n * stride_kn,
                                offs_d2, D,
                                offs_n, N_CTX - start_n,
                                ~LAST_N_BLOCK,
                                EVEN_D2_BLOCK)
        qk += tl.dot(q2, kt)

    qk *= sm_scale

    if MERGED_Q:
        within_block_offset = PAST_LEN % BLOCK_M
        qk += tl.where(tl.arange(0, BLOCK_M)[:, None] < micro_M - within_block_offset, 0, -1e6)
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

    # v_ptrs = v_ptrs + start_n * stride_vn
    v = _load_with_2d_mask(v_ptrs + start_n * stride_vn,
                           offs_n, N_CTX - start_n,
                           offs_d, D,
                           ~LAST_N_BLOCK,
                           EVEN_D1_BLOCK)
    acc += tl.dot(p, v)

    if NUM_DBLOCKS >= 2:
        acc2 = acc2 * alpha[:, None]
        v = _load_with_2d_mask(v_ptrs2 + start_n * stride_vn,
                               offs_n, N_CTX - start_n,
                               offs_d2, D,
                               ~LAST_N_BLOCK,
                               EVEN_D2_BLOCK)
        acc2 += tl.dot(p, v)

    return acc, acc2, m_i, l_i


fwd_configs = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in ([1] if is_hip() else [1, 2, 3, 4, 5, 7])\
    for w in [4, 8] \
]


@triton.autotune(
        fwd_configs,
        key=["N_CTX_FOR_AUTOTUNE", "BLOCK_D", "BLOCK_D2", "NUM_DBLOCKS",
            "BLOCK_M", "BLOCK_N", "Q_ROUNDED_LEN", "CAUSAL"]
            )
@triton.jit
def _sparse_fwd_kernel(
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
    Z, H, Q_LEN, N_CTX, D,
    PAST_LEN,
    Q_ROUNDED_LEN,
    N_CTX_FOR_AUTOTUNE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    EVEN_D_BLOCK: tl.constexpr,
    INFERENCE: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Q_LEN = N_CTX - PAST_LEN
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
    offs_d = tl.arange(0, BLOCK_D)

    EVEN_D1_BLOCK: tl.constexpr = EVEN_D_BLOCK | (NUM_DBLOCKS >= 2)
    EVEN_D2_BLOCK: tl.constexpr = EVEN_D_BLOCK

    # Initialize pointers to Q, K, V
    q_ptrs  = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    kt_ptrs = K + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    v_ptrs  = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if NUM_DBLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D2], dtype=tl.float32)
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
    #         q2 = tl.load(q_ptrs + BLOCK_D + stride_qd,
    #                      mask=(q_offs_m >= 0) & (q_offs_m < Q_LEN))

    # load q: it will stay in SRAM throughout

    q = _load_with_2d_mask(q_ptrs,
                           offs_m, Q_LEN,
                           offs_d, D,
                           EVEN_M_BLOCK,
                           EVEN_D1_BLOCK,
                           other=0)

    if NUM_DBLOCKS >= 2:
        offs_d2 = BLOCK_D + tl.arange(0, BLOCK_D2)
        q_ptrs2  = Q + offs_m[:, None] * stride_qm + offs_d2[None, :] * stride_qd
        q2 = _load_with_2d_mask(q_ptrs2,
                        offs_m, Q_LEN,
                        offs_d2, D,
                        EVEN_M_BLOCK,
                        EVEN_D1_BLOCK,
                        other=0)

        kt_ptrs2 = K + offs_n[None, :] * stride_kn + offs_d2[:, None] * stride_kd
        v_ptrs2  = V + offs_n[:, None] * stride_vn + offs_d2[None, :] * stride_vd
    else:
        q2 = 0
        kt_ptrs2, v_ptrs2 = None, None
        offs_d2 = None

    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)
    # start_l, end_l = tl.load(layout_ptr + tl.arange(0, 2) * layout_crow_stride_m).split()

    # loop over k, v and update accumulator
    if CAUSAL:
        non_diag_end = tl.maximum(end_l - NUM_DIAG_BLOCKS, start_l)
    else:
        non_diag_end = end_l

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
                kt_ptrs, kt_ptrs2, v_ptrs, v_ptrs2,
                offs_m, offs_n, offs_d, offs_d2,
                stride_kn, stride_vn, stride_kd,
                PAST_LEN, N_CTX, D,
                dtype=Q.dtype.element_ty,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                LAST_N_BLOCK=False,
                CAUSAL=CAUSAL,
                NUM_DBLOCKS=NUM_DBLOCKS,
                EVEN_D_BLOCK=EVEN_D_BLOCK,
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
                kt_ptrs, kt_ptrs2, v_ptrs, v_ptrs2,
                offs_m, offs_n, offs_d, offs_d2,
                stride_kn, stride_vn, stride_kd,
                PAST_LEN, N_CTX, D,
                dtype=Q.dtype.element_ty,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                LAST_N_BLOCK=True,
                CAUSAL=CAUSAL,
                NUM_DBLOCKS=NUM_DBLOCKS,
                EVEN_D_BLOCK=EVEN_D_BLOCK,
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

    _store_with_2d_mask(out_ptrs, acc,
                        offs_m, Q_LEN,
                        offs_d, D,
                        EVEN_M_BLOCK,
                        EVEN_D1_BLOCK,
                        )
    if NUM_DBLOCKS >= 2:
        off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d2[None, :] * stride_od
        out_ptrs = Out + off_o
        _store_with_2d_mask(out_ptrs, acc2,
                            offs_m, Q_LEN,
                            offs_d2, D,
                            EVEN_M_BLOCK,
                            EVEN_D2_BLOCK,
                            )

def _forward(ctx,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            sm_scale: int,
            layout_csr: Tuple[Tensor, Tensor, int, int],
            seq_dim: int=1,
            inference: Optional[bool]=None,
            out: Optional[Tensor]=None,
            d_splits: Optional[int]=None,
            max_seqlen: Optional[int]=None,
            return_lse: bool=False,
            causal=True,
            split_qkv: Optional[bool]=None,
            ) -> Tensor:
    """
    :param q, k, v: shape=(batch, n_heads, seq_len, head_size) if seq_len=2 else (batch, seq_len, n_heads, head_size).
        Length of q is allowed to be different than k/v for decoding.
    :param sm_scale: softmax scale in attention.
    :param layout_csr: Tuple of (layout_crow_indices, layout_col_indices, int, int):
        same as CSR.crow_indices, and CSR.col_indices used to preresent a sparse tensor.
        Each element represent a block, i.e, all elements in a block to be attentdd, or not attended at all.
    :param seq_dim: which dimension of q/k/v is a dimenion of sequence length.
    :param inference: if at inference mode, i.e., do not save addition info needed in the backward pass.
    :param out: if provided, output will be saved to.
    :param d_splits: None, 1 or 2.  None=1 if head_dim=64 else 2.
    :param max_seqlen: used for checking input seqlen

    :Return:
        output: Tensor with same shape as q
        softmax_lse: logsumexp(logits), where logits are the softmax logits
    """
    assert seq_dim in [2, 1]
    hdim = 3 - seq_dim
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    # assert q.shape[-1] in [64, 128, 256]
    assert k.shape[seq_dim] == v.shape[seq_dim]

    layout_crow_indices, layout_col_indices, block_m, block_n = layout_csr
    qlen, klen = q.size(seq_dim), k.size(seq_dim)
    num_heads, num_kv_heads = q.size(hdim), k.size(hdim)

    if max_seqlen is not None:
        assert klen <= max_seqlen

    num_head_patterns = layout_crow_indices.size(0)
    if num_head_patterns > 1:
        assert num_head_patterns in [num_heads, num_kv_heads]

    # TODO: Remove the need to slicing layout_crow_indices.
    #       Compute the offset inside the kernel
    # if qlen != klen:
    if qlen == 1:  # decoding
        # assert klen > qlen
        # assert qlen == 1, \
        #     ("If q has differnt seq length that k and v, q should only have 1 token per batch for decoding,"
        #     f"but got q length = {qlen}.")'
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

    # support arbitrary head_dims
    d = q.shape[-1]
    d_p2 = triton.next_power_of_2(d)
    if d_splits is None:
        # e.g., if d < 32, will be padded to next_power_of_2(d)
        d_splits = 2 if d_p2 > 64 or (d > 32 and d_p2 != d) else 1

    if d_splits == 1:
        # d < 16, block_d = 16, block_d2 = 0, padding
        # d = 16, block_d = 16, block_d2 = 0
        # [17, 31), block_d = 32, block_d2 = 0, padding
        # d = 32, block_d = 32, block_d2 = 0
        # d = 64, block_d = 64, block_d2 = 0
        block_d = max(triton.next_power_of_2(d), 16)
        block_d2 = 0
    elif d_splits == 2:
        # e.g.,
        # d = 48, block_d = 32, block_d2 = 16
        # (48, 64) block_d = 32, blokc_d2 = 32, padding
        # d = 96, block_d = 64, block_d2 = 32
        # d = 128, block_d = 64, block_d2 = 64
        # d = 160, block_d = 128, block_d2 = 32
        # d = 190, block_d = 128, block_d2 = 64, padding
        # d = 192, block_d = 128, block_d2 = 64
        # d = 256, block_d = 128, block_d2 = 128
        block_d = triton.next_power_of_2(triton.cdiv(d, 2))  # >= 32
        block_d2 = max(triton.next_power_of_2(d - block_d), 16)
    else:
        ValueError(f"Invalid d_splits: {d_splits}. It must be either 1 or 2.")

    even_d = (block_d + block_d2) == d

    if d_splits is None:
        block_d = max(64, triton.next_power_of_2(q.shape[-1] // 2)) # allow larger num_stages to split along D
        d_splits = triton.cdiv(q.shape[-1],  block_d)
    else:
        assert d_splits in [1, 2]
        block_d = triton.next_power_of_2(q.shape[-1] // d_splits)

    past_len = k.shape[seq_dim] - q.shape[seq_dim] if causal else 0

    _sparse_fwd_kernel[grid](
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
        q.shape[0], q.shape[hdim], q.shape[seq_dim], k.shape[seq_dim], q.shape[-1],
        past_len, # PAST_LEN
        q_rounded_len,
        multiple_of(qlen, 1024), # N_CTX_FOR_AUTOTUNE, TODO: inference??
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_D2=block_d2,
        EVEN_M_BLOCK=qlen % block_m == 0,
        EVEN_N_BLOCK=klen % block_n == 0 ,
        EVEN_D_BLOCK=even_d,
        INFERENCE=inference,
        NUM_DBLOCKS=d_splits,
        CAUSAL=causal,
        **kwargs
    )
    if inference:
        L, m = None, None

    ctx.save_for_backward(q, k, v, o, L, m)
    ctx.block_d = block_d
    ctx.block_d2 = block_d2
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.merged_q = merged_q
    ctx.seq_dim = seq_dim
    ctx.hdim = hdim
    ctx.causal = causal
    ctx.d_splits = d_splits
    ctx.past_len = past_len
    ctx.kwargs = kwargs

    # m here is the log2sumexp ()
    if return_lse:
        softmax_lse = m.view(q.shape[0], num_heads, -1)[..., :qlen] * 0.69314718
        return o, softmax_lse
    return o, None


__all__ = ["_forward"]

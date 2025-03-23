import warnings
import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple, Optional
from dkernel.utils import multiple_of, is_hip


@triton.jit
def _load_multiple(ptrs, advance, num:tl.constexpr, mask=None):
    x1 = tl.load(ptrs, mask=mask)
    if num == 1:
        return x1
    ptrs += advance
    x2 = tl.load(ptrs, mask=mask)
    if num == 2:
        return x1, x2
    ptrs += advance
    x3 = tl.load(ptrs, mask=mask)
    if num == 3:
        return x1, x2, x3



@triton.jit
def _fwd_one_kv_block(
    bid_n, micro_M,
    q, q2, acc, acc2, m_i, l_i, sm_scale,
    kt_ptrs, v_ptrs,
    offs_m, offs_n,
    stride_kn, stride_vn, stride_kd,
    PAST_LEN, N_CTX,
    dtype: tl.constexpr,
    BLOCK_H: tl.constexpr,
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
        within_block_offset = PAST_LEN % BLOCK_M
        offs_m_loc = tl.arange(0, BLOCK_M * BLOCK_H) % BLOCK_M
        qk += tl.where(offs_m_loc[:, None] < micro_M - within_block_offset, 0, -1e6)
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
    for w in [2, 4, 8] \
]


@triton.autotune(
        fwd_configs,
        key=["N_CTX_FOR_AUTOTUNE", "BLOCK_DMODEL", "NUM_DBLOCKS",
            "BLOCK_H", "BLOCK_M", "BLOCK_N", "Q_ROUNDED_LEN"]
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
    Q_PER_K,
    PAST_LEN,
    Q_ROUNDED_LEN,
    N_CTX_FOR_AUTOTUNE,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    INFERENCE: tl.constexpr,
    HETERO_Q_GROUP: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
):
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # off_h = off_hz % H
    # off_z = off_hz // H
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # NUM_KV_HEADS = H // Q_PER_K
    NUM_H_BLOCK_PER_GROUP = (Q_PER_K + BLOCK_H - 1) // BLOCK_H
    off_h_for_kv = off_h // NUM_H_BLOCK_PER_GROUP
    h_block_within_group = off_h % NUM_H_BLOCK_PER_GROUP

    # LOAD_Q_PERK = 1, 2, 4, 8, ..
    # Q_PER_K: any
    off_h_for_q = off_h_for_kv * Q_PER_K + h_block_within_group * BLOCK_H
    # off_h_for_kv = (off_h * BLOCK_H) // Q_PER_K

    if HETERO_Q_GROUP:
        off_h_for_layout = off_h_for_q
    else:
        off_h_for_layout = off_h_for_kv

    Q += off_z * stride_qz # + off_h_for_q * stride_qh
    K += off_z * stride_kz + off_h_for_kv * stride_kh
    V += off_z * stride_vz + off_h_for_kv * stride_vh
    # initialize offsets

    # layout orderings: [H, M, N]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M * BLOCK_H) % BLOCK_M
    offs_hq = off_h_for_q + tl.arange(0, BLOCK_M * BLOCK_H) // BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd + offs_hq[:, None] * stride_qh
    # offs_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    offs_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    offs_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    # Initialize pointers to Q, K, V
    q_ptrs  = Q + offs_q
    kt_ptrs = K + offs_k
    v_ptrs  = V + offs_v
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M*BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M*BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M*BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
    if NUM_DBLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M*BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
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
    # TODO:
    # EVEN_H_BLOCK = (Q_PER_K % BLOCK_H == 0)
    IS_NO_PADDING_H_BLOCK = (Q_PER_K % BLOCK_H == 0) | (off_h_for_q + BLOCK_H <= Q_PER_K)
    # TODO: EVEN_M_BLOCK -> LAST_M_BLOCK.
    # Problem: if NUM_M_BLOCKS is constexpr, compiler will need to recompile the kernel for each value.
    #   => needs lots of kernels for inference. EVEN_M_BLOCK only has tow values, so it's fine.
    # For H_blocks, it is different as H is typically fixed for a model
    if EVEN_M_BLOCK & IS_NO_PADDING_H_BLOCK:
        q = tl.load(q_ptrs)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
        else:
            q2 = 0
    else:
        mask = (offs_m[:, None] < Q_LEN) & (offs_hq[:, None] < off_h_for_q + Q_PER_K)
        q = tl.load(q_ptrs, mask=mask)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=mask)
        else:
            q2 = 0

    layout_ptr = layout_crow_ptr + off_h_for_layout * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)
    # start_l, end_l = tl.load(layout_ptr + tl.arange(0, 2) * layout_crow_stride_m).split()

    layout_col_ptr += off_h_for_layout * layout_col_stride_h
    # loop over k, v and update accumulator
    non_diag_end = tl.maximum(end_l - NUM_DIAG_BLOCKS, start_l)
    for col_idx_idx in range(start_l, non_diag_end):
        # tl.device_print('# col_idx_idx:', col_idx_idx)
        if MERGED_Q:
            bid_n = tl.load(layout_col_ptr + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = tl.load(layout_col_ptr + col_idx_idx * layout_col_stride_m + 1).to(tl.int32)
        else:
            bid_n = tl.load(layout_col_ptr + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = 0
        acc, acc2, m_i, l_i = \
            _fwd_one_kv_block(bid_n, micro_M,
                q, q2, acc, acc2, m_i, l_i, sm_scale,
                kt_ptrs, v_ptrs,
                offs_m, offs_n,
                stride_kn, stride_vn, stride_kd,
                PAST_LEN, N_CTX,
                dtype=Q.dtype.element_ty,
                BLOCK_H=BLOCK_H,
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
            bid_n = tl.load(layout_col_ptr + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = tl.load(layout_col_ptr + col_idx_idx * layout_col_stride_m + 1).to(tl.int32)
        else:
            bid_n = tl.load(layout_col_ptr + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = 0
        acc, acc2, m_i, l_i = \
            _fwd_one_kv_block(bid_n, micro_M,
                q, q2, acc, acc2, m_i, l_i, sm_scale,
                kt_ptrs, v_ptrs,
                offs_m, offs_n,
                stride_kn, stride_vn, stride_kd,
                PAST_LEN, N_CTX,
                dtype=Q.dtype.element_ty,
                BLOCK_H=BLOCK_H,
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
        l_ptrs = L + off_z * H * Q_ROUNDED_LEN + offs_hq * Q_ROUNDED_LEN + offs_m
        m_ptrs = M + off_z * H * Q_ROUNDED_LEN + offs_hq * Q_ROUNDED_LEN + offs_m
        if IS_NO_PADDING_H_BLOCK:
            tl.store(l_ptrs, l_i)
            tl.store(m_ptrs, m_i)
        else:
            mask = offs_hq < off_h_for_q + Q_PER_K
            tl.store(l_ptrs, l_i, mask=mask)
            tl.store(m_ptrs, m_i, mask=mask)

    off_o = off_z * stride_oz + offs_hq[:, None] * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o

    if EVEN_M_BLOCK & IS_NO_PADDING_H_BLOCK:
        tl.store(out_ptrs, acc)
        if NUM_DBLOCKS >= 2:
            tl.store(out_ptrs + BLOCK_DMODEL * stride_od, acc2)
    else:
        # mask = (offs_m[:, None] < Q_LEN) & (offs_hq[:, None] < H)
        mask = (offs_m[:, None] < Q_LEN) & (offs_hq[:, None] < off_h_for_q + Q_PER_K)
        tl.store(out_ptrs, acc,  mask=mask)
        if NUM_DBLOCKS >= 2:
            tl.store(out_ptrs + BLOCK_DMODEL * stride_od, acc2,  mask=mask)


def get_block_h_and_m(q_per_k, qlen, klen, block_m):
    if qlen == 1:
        block_h = min(128, triton.next_power_of_2(q_per_k))
        block_m = triton.cdiv(16, block_h)
    else:
        # == 1 is the most common case
        # for short seq, block_h>=2 maynot be able to fully utilize all the SMs 
        if q_per_k == 1 or qlen <= 1024:
            block_h = 1
        elif q_per_k % 4 == 0:
            block_h = 4
        elif q_per_k >= 8:
            block_h = 2
        else:
            block_h = 1
    return block_h, block_m
    

def _forward(ctx,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            sm_scale: int,
            layout_csr: Tuple[Tensor, Tensor, int, int],
            seq_dim: int=1,
            inference: Optional[bool]=None,
            out:Optional[Tensor]=None,
            d_splits: Optional[int]=None,
            max_seqlen: Optional[int]=None,
            block_h: Optional[int]=None,
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
    :param max_seqlen: used for checking input seqlen.
    :param block_h: number of q heads per block.
    """
    assert seq_dim in [2, 1]
    hdim = 3 - seq_dim
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert q.shape[-1] in [64, 128, 256]
    assert k.shape[seq_dim] == v.shape[seq_dim]

    layout_crow_indices, layout_col_indices, block_m, block_n = layout_csr
    qlen, klen = q.size(seq_dim), k.size(seq_dim)
    num_heads, num_kv_heads = q.size(hdim), k.size(hdim)
    assert num_heads % num_kv_heads == 0
    q_per_k = num_heads // num_kv_heads

    
    if block_h is None:
        block_h, block_m = get_block_h_and_m(q_per_k, qlen, klen, block_m)
    assert block_h <= triton.next_power_of_2(q_per_k), "block_h should be less than or equal to next power of 2 of q_per_k, otherwise, too many unneccessary paddings."

    if max_seqlen is not None:
        assert klen <= max_seqlen

    num_head_patterns = layout_crow_indices.size(0)
    if num_head_patterns > 1:
        assert num_head_patterns in [num_heads, num_kv_heads]

    # TODO: Remove the need to slicing layout_crow_indices.
    #       Compute the offset inside the kernel
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

    # required by triton that all dimensions in matmul has to be power of 2
    assert triton.next_power_of_2(block_h * block_m) ==  block_h * block_m

    head_grid = k.shape[hdim] * triton.cdiv(q_per_k, block_h)
    grid = (triton.cdiv(q.shape[seq_dim], block_m), head_grid, q.shape[0])
    q_rounded_len = grid[0] * block_m

    if inference:
        L = m = q.new_empty((1,)).contiguous()
    else:
        L = torch.zeros((q.shape[0] * q.shape[hdim], q_rounded_len),
                        device=q.device,
                        dtype=torch.float32
                        ).contiguous()
        m = torch.zeros((q.shape[0] * q.shape[hdim], q_rounded_len),
                        device=q.device,
                        dtype=torch.float32
                        ).contiguous()

    if layout_crow_indices.dim() == 1:
        layout_crow_indices = layout_crow_indices[None].expand(num_kv_heads , -1)
        layout_col_indices = layout_col_indices[None].expand((num_kv_heads,) + layout_col_indices.shape)

    HETERO_Q_GROUP = q_per_k > 1 and layout_crow_indices.size(0) == num_heads
    if HETERO_Q_GROUP:
        # heads within a q-group are heterogeneous, process one q head at a time instead
        if block_h != 1:
            warnings.warn("For within q-group heterogeneous heads, block_h must be 1. Setting block_h=1.")
        block_h = 1

    if d_splits is None:
        block_d = max(64, q.shape[-1] // 2) # allow larger num_stages to split along D
        d_splits = q.shape[-1] // block_d
    else:
        assert d_splits in [1, 2]
        block_d = q.shape[-1] // d_splits

    # print(f'>> {grid=}, {block_h=}, {block_m=}')
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
        q_per_k, # Q_PER_K
        k.shape[seq_dim] - q.shape[seq_dim], # PAST_LEN
        q_rounded_len,
        multiple_of(qlen, 1024), # N_CTX_FOR_AUTOTUNE, TODO: inference??
        BLOCK_H=block_h,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_d,
        EVEN_M_BLOCK=qlen % block_m == 0,
        EVEN_N_BLOCK=klen % block_n == 0 ,
        INFERENCE=inference,
        HETERO_Q_GROUP=HETERO_Q_GROUP,
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
    ctx.q_per_k = q_per_k
    ctx.kwargs = kwargs

    return o


__all__ = ["_forward"]
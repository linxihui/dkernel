import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple, Optional
from dkernel.utils import is_hip



def padded_sparse_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sm_scale: float,
    sparse_layout: Tuple[Tensor, Tensor, int, int],
    *,
    left_paddings: Optional[Tensor] = None,
    seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int]= None,
    seq_dim: int = 1,
):
    """
    :param q, k, v: (batch, seqlen, num_heads/num_kv_heads, head_size) if seq_dim = 1 (default),
            else (batch, num_heads/num_kv_heads, seqlen, head_size)
    :param sm_scale:
    :param sparse_layout:
    :param left_paddings: (batch, ), number of left paddings for each sample.
    :param seqlens: real seqlen, can be optionally used when has right padding.
            No need to specify if left_paddings is used.
    :param max_seqlen:
    :param seq_dim:
    """

    assert seq_dim in (1, 2)
    hdim = 3 - seq_dim

    batches, _, _, head_size = q.shape
    q_len, k_len = q.size(seq_dim), k.size(hdim)
    num_heads, num_kv_heads = q.size(hdim), k.size(hdim)

    assert q.dim() == k.dim() == v.dim() == 4
    assert num_heads % num_kv_heads == 0
    assert q.size(0) == k.size(0) and q.size(3) == k.size(3)
    assert k.shape == v.shape # TODO: allow diff head_size for k, v
    assert q_len == 1 or q_len == k_len, \
        f'q length can only 1 for decoding for same as k length for prefilling.'
    q_k_ratio = num_heads // num_kv_heads

    if max_seqlen:
        assert k.size(seq_dim) <= max_seqlen, \
            f'k has seqlen {k.size(seq_dim)} while max sequence length is set to {max_seqlen}.'

    layout_crow_indices, layout_col_indices, block_m, block_n = sparse_layout

    # paddings always has zero output, a little slower than using empty
    out = q.new_zeros(q.shape)
                                                                                               
    block_d = triton.next_power_of_2(head_size)

    # k_batch_range = torch.zeros((batches, 2), dtype=torch.int32, device=q.device)
    # if left_paddings is not None:
    #     k_batch_range[:, 0].copy_(left_paddings)
    # if seqlens is not None:
    #     k_batch_range[:, 1].copy_(seqlens + k_batch_range[:, 0])
    # else:
    #     k_batch_range[:, ]

    if left_paddings is not None:
        assert left_paddings.shape == (batches,)
        k_batch_starts = left_paddings.to(q.device, dtype=torch.int32).contiguous()
    else:
        k_batch_starts = torch.zeros((batches,), dtype=torch.int32, device=q.device)

    if seqlens is not None:
        k_batch_ends = k_batch_starts + seqlens.type_as(k_batch_starts)
        assert k_batch_ends.max() <= k_len, f'seqlens (+left_paddings if any) exceeds seqlen.'
    else:
        k_batch_ends = torch.zeros_like(k_batch_starts) + k_len

    if q_len == 1:
        q_batch_starts = torch.zeros_like(k_batch_starts)
        q_batch_ends = q_batch_starts + 1
    else:
        q_batch_starts = k_batch_starts
        q_batch_ends = k_batch_ends

    # switch to use cpu to avoid too many kernel lauch when iterate over
    q_lens = (q_batch_ends - q_batch_starts).cpu()
    n_blocks = (q_lens + block_m - 1) // block_m

    q_batch_ids = torch.tensor([i for i, n in enumerate(n_blocks) for _ in range(n)],
                                dtype=q_batch_starts.dtype,
                                device=q_batch_starts.device)
    q_start_sids = torch.tensor([i * block_m for n in n_blocks for i in range(n)],
                               dtype=q_batch_starts.dtype,
                               device=q_batch_starts.device)

    MERGED_Q = False
    if layout_col_indices.dim() == 3:
        assert layout_col_indices.size(2) == 2 and layout_col_indices.stride(
            2) == 1
        MERGED_Q = True

    NUM_DIAG_BLOCKS = max(1, block_m // block_n)

    # block_d = 64 if head_size > 64 else triton.next_power_of_2(head_size)
    d_splits = 2 if head_size > 64 else 1
    block_d = triton.next_power_of_2(triton.cdiv(head_size, d_splits))
    even_d = head_size % (d_splits * block_d) == 0

    grid = (len(q_start_sids), num_heads)

    _fwd_kernel_batch_inference[grid](
    q, k, v, out,
    sm_scale,
    q_batch_starts,
    q_batch_ends,
    k_batch_starts,
    k_batch_ends,
    q_batch_ids,
    q_start_sids,

    *q.stride(),
    *k.stride(),
    *v.stride(),
    *out.stride(),

    layout_crow_indices,
    layout_col_indices,
    *layout_crow_indices.stride(),
    *layout_col_indices.stride(),

    q_k_ratio,
    MAX_K_LEN = (k_batch_ends - k_batch_starts).max().item(),
    HAS_BATCH_DIM = True,
    D_HEAD = head_size,
    BLOCK_M = block_n,
    BLOCK_N = block_n,
    BLOCK_D = block_d,
    NUM_DBLOCKS=d_splits,
    BLOCK_M_LOADING = 16 if q_len == 1 else block_n, # smaller for decoding
    EVEN_D = even_d,
    MERGED_Q=MERGED_Q,
    NUM_DIAG_BLOCKS=NUM_DIAG_BLOCKS,
    )

    return out
    

def varlen_pparse_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,  # (total_num_tokens, num_heads, head_size)
        cu_seqlens_k: Tensor,
        cu_seqlens_q: Optional[Tensor],
        sm_scale: int,
        sparse_layout: Tuple[Tensor, Tensor, int, int],
        *,
        max_seqlen: int=None):
    # split q to blocks

    assert isinstance(sparse_layout, (list, tuple))
    layout_crow_indices, layout_col_indices, block_m, block_n = sparse_layout

    _, num_heads, head_size = q.shape
    batch_size = cu_seqlens_k.size(0) - 1

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    # TODO(linxihui): allow k, v to have different head_size
    assert k.shape == v.shape
    assert cu_seqlens_k.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    if cu_seqlens_q is None:
        if q.size(0) == batch_size:  # decoding only
            cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=cu_seqlens_k.dtype,
                device=cu_seqlens_k.device,
            )
        elif q.size(0) == k.size(0):
            cu_seqlens_q = cu_seqlens_k
        else:
            raise ValueError("cu_seqlens_q must be specified\
                    if it mix of prefilling and decoding.")
    else:
        assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)

    # switch to use cpu to avoid too many kernel launches when iterated over
    cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(torch.int32)
    cu_seqlens_q_cpu = cu_seqlens_q.cpu()
    cu_seqlens_k_cpu = cu_seqlens_k.cpu()
    q_lens = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
    k_lens = cu_seqlens_k_cpu[1:] - cu_seqlens_k_cpu[:-1]

    max_k_len = k_lens.max().item()

    assert torch.logical_or(q_lens == 1, k_lens == q_lens).all(), (
        "length of q should either be 1 (decoding) or same as k (prefilling).")

    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    n_blocks = (q_lens + block_m - 1) // block_m

    q_batch_ids = torch.tensor(
        [i for i, n in enumerate(n_blocks) for _ in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )
    q_start_sids = torch.tensor(
        [i * block_m for n in n_blocks for i in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )

    out = q.new_empty(q.shape)

    decoding_only = (q_lens == 1).all().item()

    MERGED_Q = False
    if layout_col_indices.dim() == 3:
        assert layout_col_indices.size(2) == 2 and layout_col_indices.stride(
            2) == 1
        MERGED_Q = True

    NUM_DIAG_BLOCKS = max(1, block_m // block_n)

    # block_d = 64 if head_size > 64 else triton.next_power_of_2(head_size)
    d_splits = 2 if head_size > 64 else 1
    block_d = triton.next_power_of_2(triton.cdiv(head_size, d_splits))
    even_d = head_size % (d_splits * block_d) == 0

    grid = (len(q_start_sids), num_heads, 1)

    _fwd_kernel_batch_inference[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        cu_seqlens_q[:-1],
        cu_seqlens_q[1:],
        cu_seqlens_k[:-1],
        cu_seqlens_k[1:],
        q_batch_ids,
        q_start_sids,
        0,
        *q.stride(),
        0,
        *k.stride(),
        0,
        *v.stride(),
        0,
        *out.stride(),
        layout_crow_indices,
        layout_col_indices,
        *layout_crow_indices.stride(),
        *layout_col_indices.stride()[:2],
        q_k_ratio,
        MAX_K_LEN=max_k_len,
        HAS_BATCH_DIM=False,
        D_HEAD=head_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_DBLOCKS=d_splits,
        EVEN_D=even_d,
        BLOCK_M_LOADING=(16 if decoding_only else
                         block_m),  # smaller for decoding
        MERGED_Q=MERGED_Q,
        NUM_DIAG_BLOCKS=NUM_DIAG_BLOCKS,
        # num_warps=1 if decoding_only else 4,
        # num_stages=3
    )

    return out


@triton.jit
def _fwd_kernel_inner(
    acc,
    acc2,
    l_i,
    m_i,
    q,
    q2,
    Q,
    k_block_col_idx,
    layout_col_ptr,
    layout_col_stride_h,
    layout_col_stride_m,
    k_ptrs,
    v_ptrs,
    off_h,
    offs_m,
    offs_n,
    offs_d,
    stride_kt,
    stride_kd,
    stride_vt,
    stride_vd,
    sm_scale,
    k_seqlen,
    past_len,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr,
    MERGED_Q: tl.constexpr,
):

    if MERGED_Q:
        k_block_id = tl.load(layout_col_ptr + off_h * layout_col_stride_h +
                             k_block_col_idx * layout_col_stride_m).to(
                                 tl.int32)
        micro_M = tl.load(layout_col_ptr + off_h * layout_col_stride_h +
                          k_block_col_idx * layout_col_stride_m + 1).to(
                              tl.int32)

    else:
        k_block_id = tl.load(layout_col_ptr + off_h * layout_col_stride_h +
                             k_block_col_idx * layout_col_stride_m).to(
                                 tl.int32)
        micro_M = 0

    start_n = k_block_id * BLOCK_N
    k2 = 0
    if LAST_K_BLOCK:
        if EVEN_D:
            k = tl.load(
                k_ptrs + start_n * stride_kt,
                mask=offs_n[None, :] + start_n < k_seqlen,
            )
            if NUM_DBLOCKS > 1:
                k2 = tl.load(
                    k_ptrs + start_n * stride_kt + BLOCK_D * stride_kd,
                    mask=offs_n[None, :] + start_n < k_seqlen,
                )
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kt,
                mask=(offs_n[None, :] + start_n < k_seqlen) &
                (offs_d[:, None] < D_HEAD),
            )
            if NUM_DBLOCKS > 1:
                k2 = tl.load(
                    k_ptrs + start_n * stride_kt + BLOCK_D * stride_kd,
                    mask=(offs_n[None, :] + start_n < k_seqlen) &
                    (offs_d[:, None] + BLOCK_D < D_HEAD),
                )
    else:
        if EVEN_D:
            k = tl.load(k_ptrs + start_n * stride_kt)
            if NUM_DBLOCKS > 1:
                k2 = tl.load(k_ptrs + start_n * stride_kt +
                             BLOCK_D * stride_kd)
        else:
            k = tl.load(k_ptrs + start_n * stride_kt,
                        mask=offs_d[:, None] < D_HEAD)
            if NUM_DBLOCKS > 1:
                k2 = tl.load(k_ptrs + start_n * stride_kt +
                             BLOCK_D * stride_kd,
                             mask=offs_d[:, None] + BLOCK_D < D_HEAD)

    qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)

    if NUM_DBLOCKS > 1:
        qk += tl.dot(q2, k2)
    qk *= sm_scale

    if MERGED_Q:
        qk += tl.where(
            tl.arange(0, BLOCK_M_LOADING)[:, None] < micro_M, 0, float("-inf"))

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK | M_LT_N:
        qk += tl.where(
            offs_m[:, None] + past_len >= (start_n + offs_n[None, :]),
            0,
            float("-inf"),
        )

    # flash-attn2
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    # update m_i
    m_i = m_ij
    l_i = l_i * alpha + l_ij

    p = p.to(Q.dtype.element_ty)
    # update acc
    v2 = 0
    if LAST_K_BLOCK:
        if EVEN_D:
            v = tl.load(
                v_ptrs + start_n * stride_vt,
                mask=offs_n[:, None] + start_n < k_seqlen,
            )
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vt,
                mask=(offs_n[:, None] + start_n < k_seqlen) &
                (offs_d[None, :] < D_HEAD),
            )
    else:
        if EVEN_D:
            v = tl.load(v_ptrs + start_n * stride_vt)
        else:
            v = tl.load(v_ptrs + start_n * stride_vt,
                        mask=offs_d[None, :] < D_HEAD)

    acc += tl.dot(p, v)

    if NUM_DBLOCKS > 1:
        if LAST_K_BLOCK:
            if EVEN_D:
                v2 = tl.load(
                    v_ptrs + start_n * stride_vt + BLOCK_D * stride_vd,
                    mask=offs_n[:, None] + start_n < k_seqlen,
                )
            else:
                v2 = tl.load(
                    v_ptrs + start_n * stride_vt + BLOCK_D * stride_vd,
                    mask=(offs_n[:, None] + start_n < k_seqlen) &
                    (offs_d[None, :] + BLOCK_D < D_HEAD),
                )
        else:
            if EVEN_D:
                v2 = tl.load(v_ptrs + start_n * stride_vt +
                             BLOCK_D * stride_vd)
            else:
                v2 = tl.load(v_ptrs + start_n * stride_vt +
                             BLOCK_D * stride_vd,
                             mask=offs_d[None, :] + BLOCK_D < D_HEAD)
        acc2 += tl.dot(p, v2)

    return acc, acc2, l_i, m_i


fwd_configs = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in ([1] if is_hip() else [1, 2, 3, 5, 7]) \
    for w in [1, 2, 4, 8] \
]


@triton.heuristics({
    "M_LT_N":
    lambda kwargs: kwargs["BLOCK_M"] < kwargs["BLOCK_N"],
    # "NUM_DBLOCKS":lambda kwargs: triton.cdiv(kwargs["D_HEAD"], kwargs["BLOCK_D"]),
    # "EVEN_D": lambda kwargs: kwargs["D_HEAD"] % (triton.cdiv(kwargs["D_HEAD"], kwargs["BLOCK_D"]) * kwargs["BLOCK_D"]) == 0,
    "MAX_K_LEN_AUTOTUNE":
    lambda kwargs: triton.next_power_of_2(kwargs["MAX_K_LEN"])
})
@triton.autotune(
    fwd_configs,
    key=["MAX_K_LEN_AUTOTUNE", "HAS_BATCH_DIM", "BLOCK_D", "NUM_DBLOCKS", "BLOCK_M", "BLOCK_N"])
@triton.jit
def _fwd_kernel_batch_inference(
    Q,
    K,
    V,
    Out,
    sm_scale,
    q_batch_starts,
    q_batch_ends,
    k_batch_starts,
    k_batch_ends,
    q_batch_ids,
    q_start_sids,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    q_k_ratio,
    MAX_K_LEN,
    MAX_K_LEN_AUTOTUNE,
    HAS_BATCH_DIM: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  
    BLOCK_D: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
):
    """
    NOTATION:
    pid: position id
    sid: storage id
    sbid: storage block id
    pbid: position block id
    offs_m, offs_n: storage offsets of m-dim(q, row) and n-dim(k, col)

    TODO(linxihui):
    Optimize grouped-attn
    """
    off_zm = tl.program_id(0)
    off_h = tl.program_id(1)

    off_h_for_kv = off_h // q_k_ratio

    if HAS_BATCH_DIM:
        off_z = tl.program_id(2)
        Q += off_z * stride_qb
        K += off_z * stride_kb
        V += off_z * stride_vb
        Out += off_z * stride_ob
        start_m = off_zm
        q_start_sid = start_m * BLOCK_M  # always 0 for decoding
    else:
        off_z = tl.load(q_batch_ids + off_zm).to(tl.int32)  # [0, 0, 0, 1]
        q_start_sid = tl.load(q_start_sids + off_zm)
        start_m = q_start_sid // BLOCK_M  # q_sbid

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M_LOADING)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_cu_start = tl.load(q_batch_starts + off_z).to(tl.int32)
    q_seqlen = tl.load(q_batch_ends + off_z).to(tl.int32) - q_cu_start
    k_cu_start = tl.load(k_batch_starts + off_z).to(tl.int32)
    k_seqlen = tl.load(k_batch_ends + off_z).to(tl.int32) - k_cu_start
    past_len = k_seqlen - q_seqlen

    Q += q_cu_start * stride_qt + off_h * stride_qh
    K += k_cu_start * stride_kt + off_h_for_kv * stride_kh
    V += k_cu_start * stride_vt + off_h_for_kv * stride_vh
    Out += q_cu_start * stride_ot + off_h * stride_oh

    q_pbid = (past_len + q_start_sid) // BLOCK_M
    q_ptrs = Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd

    if EVEN_D:
        q = tl.load(
            q_ptrs,
            mask=offs_m[:, None] < q_seqlen,
        )
        q2 = 0
        if NUM_DBLOCKS > 1:
            q2 = tl.load(
                q_ptrs + BLOCK_D * stride_qd,
                mask=offs_m[:, None] < q_seqlen,
            )
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
            other=0,
        )
        q2 = 0
        if NUM_DBLOCKS > 1:
            q2 = tl.load(
                q_ptrs + BLOCK_D * stride_qd,
                mask=(offs_m[:, None] < q_seqlen) &
                (offs_d[None, :] + BLOCK_D) < D_HEAD,
            )

    sparse_crow_ptr = (layout_crow_ptr + off_h * layout_crow_stride_h +
                       q_pbid * layout_crow_stride_m)

    # TODO(linxihui): load at once, with any Triton version
    # that supports `tl.split`, e.g., Triton 3.0
    k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
    k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

    m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_LOADING, BLOCK_D], dtype=tl.float32)
    acc2 = 0
    if NUM_DBLOCKS:
        acc2 = tl.zeros([BLOCK_M_LOADING, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    sm_scale *= (
        1.44269504  # 1/log2 as we use base2 for exponential and logarithm
    )

    non_diag_end = tl.maximum(k_block_end - NUM_DIAG_BLOCKS, k_block_start)
    # for col_idx_idx in range(k_block_start, non_diag_end):

    for k_block_col_idx in range(k_block_start, non_diag_end):
        acc, acc2, l_i, m_i = _fwd_kernel_inner(
            acc, acc2, l_i, m_i, q, q2, Q, k_block_col_idx, layout_col_ptr,
            layout_col_stride_h, layout_col_stride_m, k_ptrs, v_ptrs, off_h,
            offs_m, offs_n, offs_d, stride_kt, stride_kd, stride_vt, stride_vd,
            sm_scale, k_seqlen, past_len, False, BLOCK_M_LOADING, BLOCK_N,
            BLOCK_D, D_HEAD, NUM_DBLOCKS, EVEN_D, M_LT_N, MERGED_Q)

    for k_block_col_idx in range(non_diag_end, k_block_end):
        acc, acc2, l_i, m_i = _fwd_kernel_inner(
            acc, acc2, l_i, m_i, q, q2, Q, k_block_col_idx, layout_col_ptr,
            layout_col_stride_h, layout_col_stride_m, k_ptrs, v_ptrs, off_h,
            offs_m, offs_n, offs_d, stride_kt, stride_kd, stride_vt, stride_vd,
            sm_scale, k_seqlen, past_len, True, BLOCK_M_LOADING, BLOCK_N,
            BLOCK_D, D_HEAD, NUM_DBLOCKS, EVEN_D, M_LT_N, MERGED_Q)

    # flash-attn 2
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # write output
    out_ptrs = Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    if EVEN_D:
        tl.store(
            out_ptrs,
            acc,
            mask=offs_m[:, None] < q_seqlen,
        )
    else:
        tl.store(
            out_ptrs,
            acc,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
        )

    if NUM_DBLOCKS > 1:
        acc2 = acc2 / l_i[:, None]
        if EVEN_D:
            tl.store(
                out_ptrs + BLOCK_D * stride_od,
                acc2,
                mask=offs_m[:, None] < q_seqlen,
            )
        else:
            tl.store(
                out_ptrs + BLOCK_D * stride_od,
                acc2,
                mask=(offs_m[:, None] < q_seqlen) &
                (offs_d[None, :] + BLOCK_D < D_HEAD),
            )

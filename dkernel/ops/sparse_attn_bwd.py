import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
from torch import Tensor
from dkernel.utils import multiple_of, is_hip
from dkernel.ops.sparse_attn_fwd import _load_with_1d_mask, _store_with_1d_mask
from dkernel.ops.sparse_attn_fwd import _load_with_2d_mask, _store_with_2d_mask


"""
@linxihui

## Limitation:
1. In bwd, assume contiguous: Out, Do has same layout
2. inference: very limit inference in the fwd training kernel.
    Maybe a better idea to separate inference and training.

"""


def get_non_trival_stride(x):
    return tuple(s if d > 1 else None for s, d in zip(x.stride(), x.size()))


## backward
@triton.heuristics(
    {
        'EVEN_M_BLOCK': lambda kwargs: kwargs['N_CTX'] % kwargs['BLOCK_M'] == 0,
        'ROUNDED_CTX': lambda kwargs: multiple_of(kwargs['N_CTX'], kwargs['BLOCK_M']),
        'HAS_DLSE': lambda kwargs: kwargs['DL'] is not None,
        'BLOCK_D': lambda kwargs: triton.next_power_of_2(kwargs['D']),
        'EVEN_D_BLOCK': lambda kwargs: triton.next_power_of_2(kwargs['D']) == kwargs['D'],
    }
)
@triton.jit
def _bwd_preprocess(
    Out, DO, DL, # assume Out, DO have same layout.
    Delta, # assume contiguous layout, rounded length ((N_CTX + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    stride_oz, stride_oh, stride_om, stride_od,
    stride_lz, stride_lh, stride_lm,
    H, N_CTX, D,
    ROUNDED_CTX,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    HAS_DLSE: tl.constexpr,
    EVEN_D_BLOCK: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    offs_d = tl.arange(0, BLOCK_D)

    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_oz + off_h * stride_oh
    Delta += off_hz * ROUNDED_CTX

    offs_block = offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    # load

    o = _load_with_2d_mask(Out + offs_block,
                           offs_m, N_CTX,
                           offs_d, D,
                           EVEN_M_BLOCK,
                           EVEN_D_BLOCK).to(tl.float32)
    do = _load_with_2d_mask(DO + offs_block,
                           offs_m, N_CTX,
                           offs_d, D,
                           EVEN_M_BLOCK,
                           EVEN_D_BLOCK).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    if HAS_DLSE:
        # Support grad of lse
        # delta = tl.sum(o * do, axis=1) - dlse
        DL += off_z * stride_lz + off_h * stride_lh
        dl = _load_with_1d_mask(DL + offs_m * stride_lm, offs_m, N_CTX, EVEN_M_BLOCK)
        delta -= dl
    _store_with_1d_mask(Delta + offs_m, delta,
                        offs_m, N_CTX,
                        EVEN_M_BLOCK)


@triton.jit
def _bwd_inner_dkdv(
    dk, dv, dk2, dv2,
    k, v, k2, v2,
    Q, DO, m_ptrs, D_ptrs,
    qk_scale, sm_scale,
    start_m, offs_m, offs_n, offs_d, offs_d2,
    stride_qm, stride_qd,
    stride_om, stride_od,
    Q_LEN, D, PAST_LEN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_D1_BLOCK: tl.constexpr,
    EVEN_D2_BLOCK: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    IS_DIAG_BLOCK: tl.constexpr,
):

    offs_m_curr = start_m + offs_m
    q_ptrs =   Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    # load q, k, v, do on-chip
    q = _load_with_2d_mask(q_ptrs,
                           offs_m_curr, Q_LEN,
                           offs_d, D,
                           EVEN_M_BLOCK,
                           EVEN_D1_BLOCK,
                           other=0)
    # re-compute p = softmax(qk, dim=-1).T
    # NOTE: `do` is pre-divided by `l`; no normalization here
    qk = tl.dot(q, tl.trans(k))

    if NUM_DBLOCKS >= 2:
        q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d2[None, :] * stride_qd)
        q2 = _load_with_2d_mask(q_ptrs,
                               offs_m_curr, Q_LEN,
                               offs_d2, D,
                               EVEN_M_BLOCK,
                               EVEN_D2_BLOCK,
                               other=0)
        qk += tl.dot(q2, tl.trans(k2))

    # TODO: causing "error: operation scheduled before its operands"
    if IS_DIAG_BLOCK: # row_idx_idx == start_l:
        qk += tl.where(offs_m_curr[:, None] + PAST_LEN >= (offs_n[None, :]), 0, float('-inf'))

    m = _load_with_1d_mask(m_ptrs + offs_m_curr,
                           offs_m_curr, Q_LEN,
                           EVEN_M_BLOCK)
    p = tl.math.exp2(qk * qk_scale - m[:, None])

    # compute dv
    do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
    do = _load_with_2d_mask(do_ptrs,
                            offs_m_curr, Q_LEN,
                            offs_d, D,
                            EVEN_M_BLOCK,
                            EVEN_D1_BLOCK)
    dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

    # compute dp = dot(v, do)
    Di = _load_with_1d_mask(D_ptrs + offs_m_curr,
                            offs_m_curr, Q_LEN,
                            EVEN_M_BLOCK)
    dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
    dp += tl.dot(do, tl.trans(v))

    if NUM_DBLOCKS >= 2:
        do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d2[None, :] * stride_od)
        do = _load_with_2d_mask(do_ptrs,
                                offs_m_curr, Q_LEN,
                                offs_d2, D,
                                EVEN_M_BLOCK,
                                EVEN_D2_BLOCK)
        dv2 += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        dp += tl.dot(do, tl.trans(v2))

    # compute ds = p * (dp - delta[:, None])
    ds = p * dp * sm_scale
    # compute dk = dot(ds.T, q)
    dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
    if NUM_DBLOCKS >= 2:
        dk2 += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q2)

    return dk, dv, dk2, dv2


@triton.jit
def _bwd_kernel_dkdv(
    start_n, off_hz,
    Q, K, V, sm_scale,
    layout_ccol_ptr,
    layout_row_ptr,
    layout_ccol_stride_h, layout_ccol_stride_m,
    layout_row_stride_h, layout_row_stride_m,
    Out, DO,  # assume contiguous: Out, Do, DQ, DK, DV, L, M, D, seq(q) == seq(k), with stride_oz, stride_oh, stride_om, stride_od,
    DQ, DK, DV,
    L, M,
    Delta,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    Z, H, Q_LEN, N_CTX, D, PAST_LEN, ROUNDED_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    EVEN_D_BLOCK: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # start_n = tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    # DQ += off_z * stride_oz + off_h * stride_oh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    # Look like this loop can be parallelled
    # for start_n in range(0, num_block):

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    # initialize pointers to value-like data
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # pointer to row-wise quantities in value-like data
    D_ptrs = Delta + off_hz * ROUNDED_CTX
    m_ptrs = M + off_hz * ROUNDED_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    # k and v stay in SRAM throughout

    EVEN_D1_BLOCK: tl.constexpr = EVEN_D_BLOCK | (NUM_DBLOCKS >= 2)
    EVEN_D2_BLOCK: tl.constexpr = EVEN_D_BLOCK

    k = _load_with_2d_mask(k_ptrs,
                           offs_n, N_CTX,
                           offs_d, D,
                           EVEN_N_BLOCK,
                           EVEN_D1_BLOCK,
                           other=0)

    v = _load_with_2d_mask(v_ptrs,
                           offs_n, N_CTX,
                           offs_d, D,
                           EVEN_N_BLOCK,
                           EVEN_D1_BLOCK,
                           other=0)

    if NUM_DBLOCKS >= 2:
        dv2 = tl.zeros([BLOCK_N, BLOCK_D2], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, BLOCK_D2], dtype=tl.float32)
        offs_d2 = BLOCK_D + tl.arange(0, BLOCK_D2)
        k_ptrs2 = K + (offs_n[:, None] * stride_kn + offs_d2[None, :] * stride_kd)
        v_ptrs2 = V + (offs_n[:, None] * stride_vn + offs_d2[None, :] * stride_vd)
        k2 = _load_with_2d_mask(k_ptrs2,
                            offs_n, N_CTX,
                            offs_d2, D,
                            EVEN_N_BLOCK,
                            EVEN_D2_BLOCK)

        v2 = _load_with_2d_mask(v_ptrs2,
                            offs_n, N_CTX,
                            offs_d2, D,
                            EVEN_N_BLOCK,
                            EVEN_D2_BLOCK,
                            other=0)
    else:
        k2, v2, dk2, dv2 = 0, 0, 0, 0
        offs_d2, k_ptrs2, v_ptrs2 = None, None, None

    # loop over rows

    layout_ptr = layout_ccol_ptr + off_h * layout_ccol_stride_h + start_n * layout_ccol_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_ccol_stride_m).to(tl.int32)
    qk_scale = sm_scale * 1.44269504

    # for seqlen < max_seqlen, need to truncat up to seqlen
    # can use  max_m_blocks, as this is 'KV cache frendly"

    if CAUSAL:
        max_m_blocks = (tl.minimum(Q_LEN, (N_CTX -  start_n * BLOCK_N)) + BLOCK_M - 1) // BLOCK_M
        end_l = tl.minimum(end_l, start_l + max_m_blocks)

    # TODO: split into two loops:
    # 1. for diag blocks
    # 2. for non-diag blocks

    # NOTE: This can be empty in some case, i.e, start and end are the same
    for row_idx_idx in range(start_l, end_l):
        row_idx = tl.load(layout_row_ptr + off_h * layout_row_stride_h + row_idx_idx * layout_row_stride_m).to(tl.int32)
        start_m = row_idx * BLOCK_M

        # if start_m < N_CTX:
        dk, dv, dk2, dv2 = _bwd_inner_dkdv(
            dk, dv, dk2, dv2,
            k, v, k2, v2,
            Q, DO, m_ptrs, D_ptrs,
            qk_scale, sm_scale,
            start_m, offs_m, offs_n, offs_d, offs_d2,
            stride_qm, stride_qd,
            stride_om, stride_od,
            Q_LEN, D, PAST_LEN,
            BLOCK_M,
            BLOCK_N,
            EVEN_M_BLOCK,
            EVEN_D1_BLOCK,
            EVEN_D2_BLOCK,
            NUM_DBLOCKS,
            CAUSAL,
        )

    # write-back

    # tl.device_print('start_n, num_blocks', start_n * 100000 + (end_l - start_l))
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    _store_with_2d_mask(dv_ptrs, dv, offs_n, N_CTX, offs_d, D, EVEN_N_BLOCK, EVEN_D1_BLOCK)
    _store_with_2d_mask(dk_ptrs, dk, offs_n, N_CTX, offs_d, D, EVEN_N_BLOCK, EVEN_D1_BLOCK)

    if NUM_DBLOCKS >= 2:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d2[None, :] * stride_dvd)
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d2[None, :] * stride_dkd)
        _store_with_2d_mask(dv_ptrs, dv2, offs_n, N_CTX, offs_d2, D, EVEN_N_BLOCK, EVEN_D2_BLOCK)
        _store_with_2d_mask(dk_ptrs, dk2, offs_n, N_CTX, offs_d2, D, EVEN_N_BLOCK, EVEN_D2_BLOCK)


@triton.jit
def _bwd_inner_dq(
    bid_n, micro_M,
    dq, dq2,
    q, q2, do, do2, m, Di,
    qk_scale, sm_scale,
    kt_ptrs, vt_ptrs, kt_ptrs2, vt_ptrs2,
    offs_m, offs_n, offs_d, offs_d2,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    PAST_LEN, N_CTX, D,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAST_N_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    EVEN_D1_BLOCK: tl.constexpr,
    EVEN_D2_BLOCK: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    ):

    start_n = bid_n * BLOCK_N
    offs_n_curr = offs_n + start_n

    # Load kt and vt for the first block
    kt = _load_with_2d_mask(
        kt_ptrs + start_n * stride_kn,
        offs_d, D,
        offs_n_curr, N_CTX,
        EVEN_D1_BLOCK,
        ~LAST_N_BLOCK,
        other=0
    )
    qk = tl.dot(q, kt) # TODO: (H, M, D), (D, N) -> (H, M, N)

    if NUM_DBLOCKS >= 2:
        kt2 = _load_with_2d_mask(
            kt_ptrs2 + start_n * stride_kn,
            offs_d2, D,
            offs_n_curr, N_CTX,
            EVEN_D2_BLOCK,
            ~LAST_N_BLOCK,
            other=0
        )
        qk += tl.dot(q2, kt2)
    qk *= qk_scale

    # if MERGED_Q:
    #     qk += tl.where(tl.arange(0, BLOCK_M)[:, None] < micro_M, 0, -1e6)
    # assume num_locals >= BLOCK_M // BLOCK_N
    if CAUSAL & LAST_N_BLOCK:
        qk += tl.where(offs_m[:, None] + PAST_LEN >= (start_n + offs_n[None, :]), 0, -1e6)

    p = tl.math.exp2(qk - m[:, None])

    # compute dp = dot(v, do)
    dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
    vt = _load_with_2d_mask(
        vt_ptrs + start_n * stride_vn,
        offs_d, D,
        offs_n_curr, N_CTX,
        EVEN_D1_BLOCK,
        ~LAST_N_BLOCK,
        other=0
    )
    dp += tl.dot(do, vt)

    if NUM_DBLOCKS >= 2:
        vt = _load_with_2d_mask(
            vt_ptrs2 + start_n * stride_vn,
            offs_d2, D,
            offs_n_curr, N_CTX,
            EVEN_D2_BLOCK,
            ~LAST_N_BLOCK,
            other=0
        )
        dp += tl.dot(do2, vt)

    ds = p * dp * sm_scale
    if MERGED_Q:
        # ds += tl.where(tl.arange(0, BLOCK_M)[:, None] < micro_M, 0, float('-inf'))
        ds *= (tl.arange(0, BLOCK_M)[:, None] < micro_M).to(ds.dtype)

    ds = ds.to(dtype)
    # # compute dq
    dq += tl.dot(ds, tl.trans(kt))
    if NUM_DBLOCKS >= 2:
        dq2 += tl.dot(ds, tl.trans(kt2))

    return dq, dq2


@triton.jit
def _bwd_kernel_dq(
    start_m, off_hz,
    Q, K, V, sm_scale,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,
    Out, DO,  # assume contiguous: Out, Do, DQ, DK, DV, L, M, D, seq(q) == seq(k), with stride_oz, stride_oh, stride_om, stride_od,
    DQ, DK, DV,
    L, M,
    Delta,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    # TODO: strides for DO
    Z, H, Q_LEN, N_CTX, D, PAST_LEN, ROUNDED_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    EVEN_D_BLOCK: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # start_m = tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q  += off_z * stride_qz + off_h * stride_qh
    K  += off_z * stride_kz + off_h * stride_kh
    V  += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_dqz + off_h * stride_dqh
    # DK += off_z * stride_oz + off_h * stride_oh
    # DV += off_z * stride_oz + off_h * stride_oh
    # Look like this loop can be parallelled
    # for start_n in range(0, num_block):

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    # initialize pointers to value-like data
    q_ptrs  = Q  + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    do_ptrs = DO + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)

    # pointer to row-wise quantities in value-like data
    d_ptrs = Delta + off_hz * ROUNDED_CTX + offs_m
    m_ptrs = M + off_hz * ROUNDED_CTX + offs_m

    kt_ptrs =  K + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
    vt_ptrs =  V + (offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vd)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    EVEN_D1_BLOCK: tl.constexpr = EVEN_D_BLOCK | (NUM_DBLOCKS >= 2)
    EVEN_D2_BLOCK: tl.constexpr = EVEN_D_BLOCK

    q  = _load_with_2d_mask( q_ptrs, offs_m, Q_LEN, offs_d, D, EVEN_M_BLOCK, EVEN_D1_BLOCK, other=0)
    do = _load_with_2d_mask(do_ptrs, offs_m, Q_LEN, offs_d, D, EVEN_M_BLOCK, EVEN_D1_BLOCK, other=0)
    m  = _load_with_1d_mask( m_ptrs, offs_m, Q_LEN, EVEN_M_BLOCK)
    Di = _load_with_1d_mask( d_ptrs, offs_m, Q_LEN, EVEN_M_BLOCK)

    if NUM_DBLOCKS >= 2:
        offs_d2 = BLOCK_D + tl.arange(0, BLOCK_D2)
        q_ptrs  = Q  + (offs_m[:, None] * stride_qm + offs_d2[None, :] * stride_qd)
        do_ptrs = DO + (offs_m[:, None] * stride_om + offs_d2[None, :] * stride_od)
        dq2 = tl.zeros([BLOCK_M, BLOCK_D2], dtype=tl.float32)
        q2  = _load_with_2d_mask( q_ptrs, offs_m, Q_LEN, offs_d2, D, EVEN_M_BLOCK, EVEN_D2_BLOCK, other=0)
        do2 = _load_with_2d_mask(do_ptrs, offs_m, Q_LEN, offs_d2, D, EVEN_M_BLOCK, EVEN_D2_BLOCK, other=0)
        kt_ptrs2 =  K + (offs_n[None, :] * stride_kn + offs_d2[:, None] * stride_kd)
        vt_ptrs2 =  V + (offs_n[None, :] * stride_vn + offs_d2[:, None] * stride_vd)
    else:
        q2, dq2, do2 = 0, 0, 0
        offs_d2, kt_ptrs2, vt_ptrs2 = None, None, None

    # loop over rows
    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    # se = tl.load(layout_ptr + tl.arange(0, 2) * layout_crow_stride_m)
    # start_l, end_l = tl.split(se)
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)
    if CAUSAL:
        non_diag_end = tl.maximum(end_l - NUM_DIAG_BLOCKS, start_l)
    else:
        non_diag_end = end_l

    qk_scale = sm_scale * 1.44269504

    for col_idx_idx in range(start_l, non_diag_end):
        layout_col_ptr_curr = layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m
        if MERGED_Q:
            # col_idx, micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + tl.arange(0, 2)).split()
            col_idx = tl.load(layout_col_ptr_curr).to(tl.int32)
            micro_M = tl.load(layout_col_ptr_curr + 1).to(tl.int32)
        else:
            col_idx = tl.load(layout_col_ptr_curr).to(tl.int32)
            micro_M = 0
        dq, dq2 = _bwd_inner_dq(
            col_idx, micro_M,
            dq, dq2,
            q, q2, do, do2, m, Di,
            qk_scale, sm_scale,
            kt_ptrs, vt_ptrs, kt_ptrs2, vt_ptrs2,
            offs_m, offs_n, offs_d, offs_d2,
            stride_kn, stride_kd,
            stride_vn, stride_vd,
            PAST_LEN, N_CTX, D,
            dtype=Q.dtype.element_ty,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            LAST_N_BLOCK=False,
            CAUSAL=CAUSAL,
            EVEN_D1_BLOCK=EVEN_D1_BLOCK,
            EVEN_D2_BLOCK=EVEN_D2_BLOCK,
            NUM_DBLOCKS=NUM_DBLOCKS,
            MERGED_Q=MERGED_Q
            )

    for col_idx_idx in range(non_diag_end, end_l):
        if MERGED_Q:
            # col_idx, micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + tl.arange(0, 2)).split()
            col_idx = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + 1).to(tl.int32)
            # tl.device_print('col_idx: ', col_idx_idx*100 + col_idx)
        else:
            col_idx = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = 0
        dq, dq2 = _bwd_inner_dq(
            col_idx, micro_M,
            dq, dq2,
            q, q2, do, do2, m, Di,
            qk_scale, sm_scale,
            kt_ptrs, vt_ptrs, kt_ptrs2, vt_ptrs2,
            offs_m, offs_n, offs_d, offs_d2,
            stride_kn, stride_kd,
            stride_vn, stride_vd,
            PAST_LEN, N_CTX, D,
            dtype =Q.dtype.element_ty,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            LAST_N_BLOCK=True,
            CAUSAL=CAUSAL,
            EVEN_D1_BLOCK=EVEN_D1_BLOCK,
            EVEN_D2_BLOCK=EVEN_D2_BLOCK,
            NUM_DBLOCKS=NUM_DBLOCKS,
            MERGED_Q=MERGED_Q
            )

    # write-back
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    _store_with_2d_mask(dq_ptrs, dq, offs_m, Q_LEN, offs_d, D, EVEN_M_BLOCK, EVEN_D1_BLOCK)

    if NUM_DBLOCKS >= 2:
        dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_d2[None, :] * stride_dqd)
        _store_with_2d_mask(dq_ptrs, dq2, offs_m, Q_LEN, offs_d2, D, EVEN_M_BLOCK, EVEN_D2_BLOCK)


# Does not suuport unequal seqlen(q) and seqlen(k)
configs_bwd = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in ([1] if is_hip() else [1, 2, 3, 5, 7]) \
    for w in [4, 8] \
]


@triton.heuristics(
    values={
        "EVEN_M_BLOCK" : lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_M" ] == 0,
        "EVEN_N_BLOCK" : lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_N" ] == 0,
        "EVEN_M2_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_M2"] == 0,
        "EVEN_N2_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_N2"] == 0,
        "EVEN_D_BLOCK": lambda kwargs: kwargs["BLOCK_D"] + kwargs["BLOCK_D2"] == kwargs["D"],
        "N_CTX_FOR_AUTOTUNE": lambda kwargs: triton.next_power_of_2(kwargs["N_CTX"]),
    })
@triton.autotune(
    configs_bwd,
    key=["N_CTX_FOR_AUTOTUNE", "BLOCK_D", "BLOCK_D2", "NUM_DBLOCKS",
        "BLOCK_M", "BLOCK_N", "BLOCK_M2", "BLOCK_N2", "CAUSAL"]
    )
@triton.jit
def _sparse_bwd_kernel(
    Q, K, V, sm_scale,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,

    layout_ccol_ptr,
    layout_row_ptr,
    layout_ccol_stride_h, layout_ccol_stride_m,
    layout_row_stride_h, layout_row_stride_m,

    Out, DO,  # assume contiguous: Out, Do, DQ, DK, DV, L, M, D, seq(q) == seq(k), with stride_oz, stride_oh, stride_om, stride_od,
    DQ, DK, DV,
    L, M,
    Delta,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    # TODO: strides for DO
    Z, H, Q_LEN, N_CTX, D, PAST_LEN,
    num_block,
    ROUNDED_CTX,
    NUM_BLOCKS_FOR_DQ: tl.constexpr,
    OFFSET_BLOCKS_FOR_DKDV: tl.constexpr,
    N_CTX_FOR_AUTOTUNE,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    EVEN_D_BLOCK: tl.constexpr,

    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    EVEN_M2_BLOCK: tl.constexpr,
    EVEN_N2_BLOCK: tl.constexpr,

    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    if tl.program_id(0) < NUM_BLOCKS_FOR_DQ:
        _bwd_kernel_dq(
            tl.program_id(0), tl.program_id(1),
            Q, K, V, sm_scale,
            layout_crow_ptr,
            layout_col_ptr,
            layout_crow_stride_h, layout_crow_stride_m,
            layout_col_stride_h, layout_col_stride_m,
            Out, DO,
            DQ, DK, DV,
            L, M,
            Delta,
            stride_qz, stride_qh, stride_qm, stride_qd,
            stride_kz, stride_kh, stride_kn, stride_kd,
            stride_vz, stride_vh, stride_vn, stride_vd,
            stride_oz, stride_oh, stride_om, stride_od,
            stride_dqz, stride_dqh, stride_dqm, stride_dqd,
            Z, H, Q_LEN, N_CTX, D, PAST_LEN, ROUNDED_CTX,
            num_block,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            BLOCK_D2,
            EVEN_M_BLOCK,
            EVEN_N_BLOCK,
            EVEN_D_BLOCK,
            NUM_DBLOCKS,
            MERGED_Q,
            NUM_DIAG_BLOCKS,
            CAUSAL,
        )

    if tl.program_id(0) >= OFFSET_BLOCKS_FOR_DKDV:
        _bwd_kernel_dkdv(
            tl.program_id(0) - OFFSET_BLOCKS_FOR_DKDV, tl.program_id(1),
            Q, K, V, sm_scale,
            layout_ccol_ptr,
            layout_row_ptr,
            layout_ccol_stride_h, layout_ccol_stride_m,
            layout_row_stride_h, layout_row_stride_m,
            Out, DO,
            DQ, DK, DV,
            L, M,
            Delta,
            stride_qz, stride_qh, stride_qm, stride_qd,
            stride_kz, stride_kh, stride_kn, stride_kd,
            stride_vz, stride_vh, stride_vn, stride_vd,
            stride_oz, stride_oh, stride_om, stride_od,
            stride_dkz, stride_dkh, stride_dkn, stride_dkd,
            stride_dvz, stride_dvh, stride_dvn, stride_dvd,
            Z, H, Q_LEN, N_CTX, D, PAST_LEN, ROUNDED_CTX,
            num_block,
            BLOCK_M2,
            BLOCK_N2,
            BLOCK_D,
            BLOCK_D2,
            EVEN_M2_BLOCK,
            EVEN_N2_BLOCK,
            EVEN_D_BLOCK,
            NUM_DBLOCKS,
            CAUSAL,
        )


def _backward(ctx,
              do: Tensor,
              dlse: Optional[Tensor],
              dq: Optional[Tensor]=None,
              dk: Optional[Tensor]=None,
              dv: Optional[Tensor]=None
              ) -> Tuple[Tensor, Tensor, Tensor]:
    """

    """
    q, k, v, o, l, m = ctx.saved_tensors

    layout_crow_indices, layout_col_indices, dq_block_m, dq_block_n = ctx.layout_csr
    layout_ccol_indices, layout_row_indices, dk_block_m, dk_block_n = ctx.layout_csc

    # assert len(set(get_non_trival_stride(x) for x in [o, do])) == 1, \
    #         f"strides incompatible: strides of o, do are {[x.stride() for x in [o, do]]}."

    if get_non_trival_stride(o) != get_non_trival_stride(do):
        o = o.contiguous()
        do = do.contiguous()

    if layout_ccol_indices.dim() == 1:
        layout_ccol_indices = layout_ccol_indices[None].expand(q.shape[hdim], -1)
        layout_row_indices = layout_row_indices[None].expand(q.shape[hdim], -1)

    # do = do.contiguous()

    def _check_or_new_tensor(x, shape):
        if x is not None:
            assert x.shape == shape
        else:
            x = q.new_empty(shape)
        return x

    dq = _check_or_new_tensor(dq, q.shape)
    dk = _check_or_new_tensor(dk, k.shape)
    dv = _check_or_new_tensor(dv, v.shape)

    # dq = dq if dq is not None else q.new_empty(q.shape).contiguous()
    # dk = dk if dk is not None else k.new_empty(k.shape).contiguous()
    # dv = dv if dv is not None else v.new_empty(v.shape).contiguous()
    # should be stride empty_like(o) instead of do. TODO: for fp8, format for bwd and fwd might be diff.
    # but dtype should be fp32/bf16? as do_scaled is processed.
    # do_scaled = torch.empty_like(do)
    delta = torch.empty_like(m).contiguous()
    hdim, seq_dim = ctx.hdim, ctx.seq_dim
    qlen = q.size(seq_dim)
    klen = k.size(seq_dim)

    if dlse is not None:
        dlse_stride = (dlse.stride(0), dlse.stride(hdim), dlse.stride(seq_dim))
    else:
        dlse_stride =(0, 0, 0)

    grid = (triton.cdiv(qlen, dq_block_m), q.shape[0] * q.shape[hdim])

    _bwd_preprocess[grid](
        o, do, dlse,
        delta,
        do.stride(0), do.stride(hdim), do.stride(seq_dim), do.stride(3),
        *dlse_stride,
        q.shape[hdim], qlen, q.shape[-1],
        BLOCK_M=dq_block_m,
    )

    if dq_block_m == dk_block_n and qlen == klen:
        # a pid process dq and dkdv sequentially, this naturally
        # balances the computation of each pid.
        num_seq_blocks = triton.cdiv(qlen, dq_block_m)
        num_blocks_for_dq = num_seq_blocks
        offset_blocks_for_dkdv = 0
    else:
        # dq and dkdv are computed in different pid
        num_blocks_for_dq = triton.cdiv(qlen, dq_block_m)
        offset_blocks_for_dkdv = num_blocks_for_dq
        num_seq_blocks = num_blocks_for_dq + triton.cdiv(klen, dk_block_n)

    grid = (num_seq_blocks, grid[1])
    # print(f'>> bwd: {num_blocks_for_dq=}, {offset_blocks_for_dkdv=}, {qlen=}, {klen=}, {grid=}')
    rounded_ctx = delta.size(-1)

    def _to_order_bhsd(x: tuple):
        return tuple([x[0], x[hdim], x[seq_dim]] + list(x[3:]))

    # TODO:  autotune BLOCK_M, BLOCK_N, layout_crow_indices can be heuristic with lamda function??

    _sparse_bwd_kernel[grid](
        q, k, v, ctx.sm_scale,

        layout_crow_indices,
        layout_col_indices,
        layout_crow_indices.stride(0), layout_crow_indices.stride(1),
        layout_col_indices.stride(0), layout_col_indices.stride(1),

        layout_ccol_indices,
        layout_row_indices,
        layout_ccol_indices.stride(0), layout_ccol_indices.stride(1),
        layout_row_indices.stride(0), layout_row_indices.stride(1),

        o, do,
        dq, dk, dv,
        l, m,
        delta,
        *_to_order_bhsd(q.stride()),
        *_to_order_bhsd(k.stride()),
        *_to_order_bhsd(v.stride()),
        *_to_order_bhsd(o.stride()), # shared between o, do
        *_to_order_bhsd(dq.stride()),
        *_to_order_bhsd(dk.stride()), # dk might have diff length to dq
        *_to_order_bhsd(dv.stride()), # dv mihgt have diff head_dim to dk
        q.shape[0], q.shape[hdim], q.shape[seq_dim], k.shape[seq_dim], q.shape[-1],
        ctx.past_len, # PAST_LEN
        ctx.grid[0], # num_blocks
        rounded_ctx,
        num_blocks_for_dq,
        offset_blocks_for_dkdv,
        BLOCK_M=dq_block_m,
        BLOCK_N=dq_block_n,
        BLOCK_D=ctx.block_d,
        BLOCK_M2=dk_block_m,
        BLOCK_N2=dk_block_n,
        BLOCK_D2=ctx.block_d2,
        NUM_DBLOCKS=ctx.d_splits,
        CAUSAL=ctx.causal,
        **ctx.kwargs
    )

    return dq, dk, dv


__all__ = ["_backward"]

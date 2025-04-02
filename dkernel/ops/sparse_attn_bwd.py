import torch
import triton
import os
import triton.language as tl
from typing import Tuple, Optional
from torch import Tensor
from dkernel.utils import multiple_of, is_hip


"""
@linxihui

## Limitation:
1. In bwd, assume contiguous: Out, Do, DQ, DK, DV, L, M, D, seq(q) == seq(k), with stride_oz, stride_oh, stride_om, stride_od,
2. inference: very limit inference in the fwd training kernel.
    Maybe a better idea to separate inference and training.

"""


## backward
@triton.heuristics(
    {
        'EVEN_M_BLOCK': lambda kwargs: kwargs['N_CTX'] % kwargs['BLOCK_M'] == 0,
        'ROUNDED_CTX': lambda kwargs: multiple_of(kwargs['N_CTX'], kwargs['BLOCK_M']),
        'HAS_DLSE': lambda kwargs: kwargs['DL'] is not None,
    }
)
@triton.jit
def _bwd_preprocess(
    Out, DO, DL, # assume Out, DO have same layout.
    Delta, # assume contiguous layout, rounded length ((N_CTX + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    stride_oz, stride_oh, stride_om, stride_od,
    stride_lz, stride_lh, stride_lm,
    H, N_CTX,
    ROUNDED_CTX,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    HAS_DLSE: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    off_d = tl.arange(0, D_HEAD)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_oz + off_h * stride_oh
    Delta += off_hz * ROUNDED_CTX

    offs_block = offs_m[:, None] * stride_om + off_d[None, :] * stride_od
    # load
    if EVEN_M_BLOCK:
        o  = tl.load(Out + offs_block).to(tl.float32)
        do = tl.load( DO + offs_block).to(tl.float32)
    else: # TODO: only need it on last block
        o  = tl.load(Out + offs_block, mask=offs_m[:, None] < N_CTX).to(tl.float32)
        do = tl.load( DO + offs_block, mask=offs_m[:, None] < N_CTX).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    if HAS_DLSE:
        DL += off_z * stride_lz + off_h * stride_lh
        if EVEN_M_BLOCK:
            dl = tl.load( DL + offs_m * stride_lm).to(tl.float32)
        else:
            dl = tl.load( DL + offs_m * stride_lm, mask=offs_m < N_CTX).to(tl.float32)
        delta -= dl
    tl.store(Delta + offs_m, delta)


# Support grad of lse
# delta = tl.sum(o * do, axis=1) - dlse

@triton.jit
def _bwd_inner_dkdv(
    dk, dv, dk2, dv2,
    k, v, k2, v2,
    Q, DO, m_ptrs, D_ptrs,
    qk_scale, sm_scale,
    start_m, offs_m, offs_n, offs_d,
    stride_qm, stride_qd,
    stride_om, stride_od,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    IS_DIAG_BLOCK: tl.constexpr,
):
    # offs_qm = start_m + tl.arange(0, BLOCK_M)
    offs_m_curr = start_m + offs_m
    q_ptrs =   Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)

    # load q, k, v, do on-chip
    if EVEN_M_BLOCK:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0)
    # re-compute p = softmax(qk, dim=-1).T
    # NOTE: `do` is pre-divided by `l`; no normalization here
    qk = tl.dot(q, tl.trans(k))

    if NUM_DBLOCKS >= 2:
        if EVEN_M_BLOCK:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
        else:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m_curr[:, None] < N_CTX)
        qk += tl.dot(q2, tl.trans(k2))

    # TODO: causing "error: operation scheduled before its operands"
    if IS_DIAG_BLOCK: # row_idx_idx == start_l:
        qk += tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), 0, float('-inf'))

    if EVEN_M_BLOCK:
        m = tl.load(m_ptrs + offs_m_curr)
    else:
        m = tl.load(m_ptrs + offs_m_curr, mask=offs_m_curr < N_CTX)
    p = tl.math.exp2(qk * qk_scale - m[:, None])

    # compute dv
    if EVEN_M_BLOCK:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0)

    dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

    # compute dp = dot(v, do)
    if EVEN_M_BLOCK:
        Di = tl.load(D_ptrs + offs_m_curr)
    else:
        Di = tl.load(D_ptrs + offs_m_curr, mask=offs_m_curr < N_CTX)
    dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
    dp += tl.dot(do, tl.trans(v))

    if NUM_DBLOCKS >= 2:
        if EVEN_M_BLOCK:
            do = tl.load(do_ptrs + BLOCK_DMODEL * stride_od)
        else:
            do = tl.load(do_ptrs + BLOCK_DMODEL * stride_od, mask=offs_m_curr[:, None] < N_CTX)
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
    D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    # stride_dz, stride_dh, stride_dm, stride_dd,
    Z, H, N_CTX, ROUNDED_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
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
    DK += off_z * stride_oz + off_h * stride_oh
    DV += off_z * stride_oz + off_h * stride_oh
    # Look like this loop can be parallelled
    # for start_n in range(0, num_block):

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers to value-like data
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * ROUNDED_CTX
    m_ptrs = M + off_hz * ROUNDED_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    if EVEN_N_BLOCK:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0)

    if NUM_DBLOCKS >= 2:
        dv2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        if EVEN_N_BLOCK:
            k2 = tl.load(k_ptrs + BLOCK_DMODEL * stride_kd)
            v2 = tl.load(v_ptrs + BLOCK_DMODEL * stride_vd)
        else:
            k2 = tl.load(k_ptrs + BLOCK_DMODEL * stride_kd, mask=offs_n[:, None] < N_CTX)
            v2 = tl.load(v_ptrs + BLOCK_DMODEL * stride_vd, mask=offs_n[:, None] < N_CTX, other=0)
    else:
        k2, v2, dk2, dv2 = 0, 0, 0, 0

    # loop over rows

    layout_ptr = layout_ccol_ptr + off_h * layout_ccol_stride_h + start_n * layout_ccol_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_ccol_stride_m).to(tl.int32)
    qk_scale = sm_scale * 1.44269504

    # for seqlen < max_seqlen, need to truncat up to seqlen
    # can use  max_m_blocks, as this is 'KV cache frendly"
    if CAUSAL:
        max_m_blocks = (N_CTX - start_n * BLOCK_N + BLOCK_M - 1) // BLOCK_M
        end_l = tl.minimum(end_l, start_l + max_m_blocks)

    # TODO: separate `start_l` (IS_DIAG_BLOCK=True). Others, False
    for row_idx_idx in range(start_l, end_l):
        row_idx = tl.load(layout_row_ptr + off_h * layout_row_stride_h + row_idx_idx * layout_row_stride_m).to(tl.int32)
        start_m = row_idx * BLOCK_M

        # if start_m < N_CTX:
        dk, dv, dk2, dv2 = _bwd_inner_dkdv(
            dk, dv, dk2, dv2,
            k, v, k2, v2,
            Q, DO, m_ptrs, D_ptrs,
            qk_scale, sm_scale,
            start_m, offs_m, offs_n, offs_d,
            stride_qm, stride_qd,
            stride_om, stride_od,
            N_CTX,
            BLOCK_M,
            BLOCK_N,
            EVEN_M_BLOCK,
            BLOCK_DMODEL,
            NUM_DBLOCKS,
            CAUSAL,
        )

    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    dk_ptrs = DK + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    if EVEN_N_BLOCK:
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
    else:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)

    if NUM_DBLOCKS >= 2:
        dv_ptrs2 = dv_ptrs + BLOCK_DMODEL * stride_od
        dk_ptrs2 = dk_ptrs + BLOCK_DMODEL * stride_od
        if EVEN_N_BLOCK:
            tl.store(dv_ptrs2, dv2)
            tl.store(dk_ptrs2, dk2)
        else:
            tl.store(dv_ptrs2, dv2, mask=offs_n[:, None] < N_CTX)
            tl.store(dk_ptrs2, dk2, mask=offs_n[:, None] < N_CTX)

@triton.jit
def _bwd_inner_dq(
    bid_n, micro_M,
    dq, dq2,
    q, q2, do, do2, m, Di,
    qk_scale, sm_scale,
    kt_ptrs, vt_ptrs,
    offs_m, offs_n,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
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
        vt = tl.load(vt_ptrs + start_n * stride_vn, mask=offs_n[None, :] + start_n < N_CTX)
    else:
        kt = tl.load(kt_ptrs + start_n * stride_kn)
        vt = tl.load(vt_ptrs + start_n * stride_vn)

    qk = tl.dot(q, kt) # TODO: (H, M, D), (D, N) -> (H, M, N)

    if NUM_DBLOCKS >= 2:
        if LAST_N_BLOCK:
            kt2 = tl.load(kt_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd, mask=offs_n[None, :] + start_n < N_CTX)
        else:
            kt2 = tl.load(kt_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd)
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
    dp += tl.dot(do, vt)

    if NUM_DBLOCKS >= 2:
        if LAST_N_BLOCK:
            vt = tl.load(vt_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd, mask=offs_n[None, :] + start_n < N_CTX)
        else:
            vt = tl.load(vt_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd)
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
    D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    # stride_dz, stride_dh, stride_dm, stride_dd,
    # TODO: strides for DO
    Z, H, N_CTX, ROUNDED_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
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
    DQ += off_z * stride_oz + off_h * stride_oh
    # DK += off_z * stride_oz + off_h * stride_oh
    # DV += off_z * stride_oz + off_h * stride_oh
    # Look like this loop can be parallelled
    # for start_n in range(0, num_block):

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers to value-like data
    q_ptrs  = Q  + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    do_ptrs = DO + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + off_hz * ROUNDED_CTX + offs_m
    m_ptrs = M + off_hz * ROUNDED_CTX + offs_m

    kt_ptrs =  K + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
    vt_ptrs =  V + (offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vd)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if EVEN_M_BLOCK:
        q =  tl.load( q_ptrs)
        do = tl.load(do_ptrs)
        m =  tl.load( m_ptrs)
        Di = tl.load( d_ptrs)
    else:
        q =  tl.load( q_ptrs, mask=offs_m[:, None] < N_CTX)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX)
        m =  tl.load( m_ptrs, mask=offs_m < N_CTX)
        Di = tl.load( d_ptrs, mask=offs_m < N_CTX)

    if NUM_DBLOCKS >= 2:
        dq2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        if EVEN_M_BLOCK:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
            do2 = tl.load(do_ptrs + BLOCK_DMODEL * stride_od)
        else:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m[:, None] < N_CTX)
            do2 = tl.load(do_ptrs + BLOCK_DMODEL * stride_od, mask=offs_m[:, None] < N_CTX)
    else:
        q2, dq2, do2 = 0, 0, 0

    # loop over rows
    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    se = tl.load(layout_ptr + tl.arange(0, 2) * layout_crow_stride_m)
    start_l, end_l = tl.split(se)
    # start_l = tl.load(layout_ptr).to(tl.int32)
    # end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)

    if CAUSAL:
        non_diag_end = tl.maximum(end_l - NUM_DIAG_BLOCKS, start_l)
    else:
        non_diag_end = end_l

    qk_scale = sm_scale * 1.44269504

    for col_idx_idx in range(start_l, non_diag_end):
        if MERGED_Q:
            # col_idx, micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + tl.arange(0, 2)).split()
            col_idx = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m + 1).to(tl.int32)
        else:
            col_idx = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
            micro_M = 0
        dq, dq2 = _bwd_inner_dq(
            col_idx, micro_M,
            dq, dq2,
            q, q2, do, do2, m, Di,
            qk_scale, sm_scale,
            kt_ptrs, vt_ptrs,
            offs_m, offs_n,
            stride_kn, stride_kd,
            stride_vn, stride_vd,
            0, N_CTX,
            dtype =Q.dtype.element_ty,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            LAST_N_BLOCK=False,
            CAUSAL=CAUSAL,
            BLOCK_DMODEL=BLOCK_DMODEL,
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
            kt_ptrs, vt_ptrs,
            offs_m, offs_n,
            stride_kn, stride_kd,
            stride_vn, stride_vd,
            0, N_CTX,
            dtype =Q.dtype.element_ty,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            LAST_N_BLOCK=True,
            CAUSAL=CAUSAL,
            BLOCK_DMODEL=BLOCK_DMODEL,
            NUM_DBLOCKS=NUM_DBLOCKS,
            MERGED_Q=MERGED_Q
            )

    # write-back
    dq_ptrs = DQ + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    if EVEN_M_BLOCK:
        tl.store(dq_ptrs, dq)
    else:
        tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)

    if NUM_DBLOCKS >= 2:
        dq_ptrs2 = dq_ptrs + BLOCK_DMODEL * stride_od
        if EVEN_M_BLOCK:
            tl.store(dq_ptrs2, dq2)
        else:
            tl.store(dq_ptrs2, dq2, mask=offs_m[:, None] < N_CTX)



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
        "N_CTX_FOR_AUTOTUNE": lambda kwargs: triton.next_power_of_2(kwargs["N_CTX"]),
    })
@triton.autotune(
    configs_bwd,
    key=["N_CTX_FOR_AUTOTUNE", "BLOCK_DMODEL", "NUM_DBLOCKS",
        "BLOCK_M", "BLOCK_N", "BLOCK_M2", "BLOCK_N2", "CAUSAL"]
    )
@triton.jit
def _bwd_kernel(
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
    D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    # stride_dz, stride_dh, stride_dm, stride_dd,
    # TODO: strides for DO
    Z, H, N_CTX,
    num_block,
    ROUNDED_CTX,
    NUM_M_BLOCK_FOR_DQ,
    N_CTX_FOR_AUTOTUNE,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,

    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    EVEN_M2_BLOCK: tl.constexpr,
    EVEN_N2_BLOCK: tl.constexpr,

    BLOCK_DMODEL: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
    MERGED_Q: tl.constexpr,
    NUM_DIAG_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    if (BLOCK_M == BLOCK_N2) | (tl.program_id(0) < NUM_M_BLOCK_FOR_DQ):
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
            D,
            stride_qz, stride_qh, stride_qm, stride_qd,
            stride_kz, stride_kh, stride_kn, stride_kd,
            stride_vz, stride_vh, stride_vn, stride_vd,
            stride_oz, stride_oh, stride_om, stride_od,
            Z, H, N_CTX, ROUNDED_CTX,
            num_block,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            EVEN_M_BLOCK,
            EVEN_N_BLOCK,
            NUM_DBLOCKS,
            MERGED_Q,
            NUM_DIAG_BLOCKS,
            CAUSAL,
        )

    if tl.program_id(0) >= NUM_M_BLOCK_FOR_DQ:
        _bwd_kernel_dkdv(
            tl.program_id(0) - NUM_M_BLOCK_FOR_DQ, tl.program_id(1),
            Q, K, V, sm_scale,
            layout_ccol_ptr,
            layout_row_ptr,
            layout_ccol_stride_h, layout_ccol_stride_m,
            layout_row_stride_h, layout_row_stride_m,
            Out, DO,
            DQ, DK, DV,
            L, M,
            D,
            stride_qz, stride_qh, stride_qm, stride_qd,
            stride_kz, stride_kh, stride_kn, stride_kd,
            stride_vz, stride_vh, stride_vn, stride_vd,
            stride_oz, stride_oh, stride_om, stride_od,
            Z, H, N_CTX, ROUNDED_CTX,
            num_block,
            BLOCK_M2,
            BLOCK_N2,
            BLOCK_DMODEL,
            EVEN_M2_BLOCK,
            EVEN_N2_BLOCK,
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

    if not do.is_contiguous():
        # TODO: is it necessary to have non-contiguous layout
        do = do.contiguous()

    if not o.is_contiguous():
        # TODO: currently only work with contiguous q/k/v.
        raise ValueError(f'--> output is not contiguous: {o.stride()=}. This is maybe caused by q/k/v not being contiguous.')

    if layout_ccol_indices.dim() == 1:
        layout_ccol_indices = layout_ccol_indices[None].expand(q.shape[hdim], -1)
        layout_row_indices = layout_row_indices[None].expand(q.shape[hdim], -1)

    # do = do.contiguous()

    def _check_or_new_tensor(x):
        if x is not None:
            assert x.shape == q.shape and x.is_contiguous()
        else:
            x = q.new_empty(q.shape).contiguous()
        return x

    dq = _check_or_new_tensor(dq)
    dk = _check_or_new_tensor(dk)
    dv = _check_or_new_tensor(dv)

    # dq = dq if dq is not None else q.new_empty(q.shape).contiguous()
    # dk = dk if dk is not None else k.new_empty(k.shape).contiguous()
    # dv = dv if dv is not None else v.new_empty(v.shape).contiguous()
    # should be stride empty_like(o) instead of do. TODO: for fp8, format for bwd and fwd might be diff.
    # but dtype should be fp32/bf16? as do_scaled is processed.
    # do_scaled = torch.empty_like(do)
    delta = torch.empty_like(m).contiguous()
    hdim, seq_dim = ctx.hdim, ctx.seq_dim
    qlen = q.size(seq_dim)

    # assert o.stride() == dq.stride() == dk.stride() == dv.stride() == do.stride(), \
    #     f'> {o.stride()=}\n  {dq.stride()=}\n  {dk.stride()=}\n  {dv.stride()=}\n  {do.stride()=}' # == do_scaled.stride()

    def get_non_trival_stride(x):
        return tuple(s if d > 1 else None for s, d in zip(x.stride(), x.size()))

    assert len(set(get_non_trival_stride(x) for x in [o, dq, dk, dv, do])) == 1, \
            f"strides incompatible: strides of o, dq, dk, dv, do are {[x.stride() for x in [o, dq, dk, dv, do]]}."

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
        q.shape[hdim], qlen,
        BLOCK_M=dq_block_m,
        D_HEAD=q.shape[-1],
    )

    if dq_block_m == dk_block_n:
        # a pid process dq and dkdv sequentially, this naturally
        # balances the computation of each pid.
        num_m_block_for_dq = 0
        num_seq_blocks = triton.cdiv(qlen, dq_block_m)
    else:
        # dq and dkdv are computed in different pid
        num_m_block_for_dq = triton.cdiv(qlen, dq_block_m)
        num_seq_blocks = num_m_block_for_dq + triton.cdiv(qlen, dk_block_m)

    grid = (num_seq_blocks, grid[1])
    rounded_ctx = delta.size(-1)

    _bwd_kernel[grid](
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
        q.stride(0), q.stride(hdim), q.stride(seq_dim), q.stride(3),
        k.stride(0), k.stride(hdim), k.stride(seq_dim), k.stride(3),
        v.stride(0), v.stride(hdim), v.stride(seq_dim), v.stride(3),
        o.stride(0), o.stride(hdim), o.stride(seq_dim), o.stride(3),
        q.shape[0], q.shape[hdim], q.shape[seq_dim],

        ctx.grid[0], # num_blocks
        rounded_ctx,
        num_m_block_for_dq,
        BLOCK_M=dq_block_m,
        BLOCK_N=dq_block_n,
        BLOCK_M2=dk_block_m,
        BLOCK_N2=dk_block_n,
        BLOCK_DMODEL=ctx.block_d,
        NUM_DBLOCKS=q.shape[-1] // ctx.block_d,
        CAUSAL=ctx.causal,
        **ctx.kwargs
    )

    return dq, dk, dv


__all__ = ["_backward"]
"""
Helper functions for 2D/3D sparse pattern.
These function are not optimized and very inefficient.
Avoid calling them too frequently.
"""

import torch
import triton
import numpy as np
from collections import namedtuple
from torch import Tensor
from typing import Tuple, Optional, Union
# from scipy import sparse


_CSR_Matrix = namedtuple('csr_matrix', ("indptr", 'indices', 'data'))


def to_csr(mat: np.ndarray) -> _CSR_Matrix:
    mat_bin =(mat != 0)
    num_elms = mat_bin.astype(np.int32).sum(-1)
    _, col_idx = np.where(mat_bin)
    crow = np.cumsum(np.concatenate([[0],  num_elms]))
    return _CSR_Matrix(crow.astype(np.int32), col_idx.astype(np.int32), mat[mat_bin])


def dense_to_crow_col(x: Tensor,
                      include_values: bool=False
                      ) -> Tuple[Tensor, Tensor]:
    ''' Turning a 2D/3D torch tensor (x) to CSR rows/cols indexing.

    TODO:
        1. improve efficiency, is it faster if done in CPU, or customize a cuda kernel for it?
    NOTE: col_indices padded -1
    '''
    device = x.device
    pad = -1
    dim = x.dim()
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x[None]

    # x = [xi.to_sparse_csr() for xi in x]
    # crows = torch.vstack([xi.crow_indices() for xi in x])
    # cols = [xi.col_indices() for xi in x]

    x = [to_csr(xi.to(torch.int32).cpu().numpy()) for xi in x]
    crows = torch.vstack([torch.from_numpy(xi.indptr) for xi in x])
    cols = [torch.from_numpy(xi.indices) for xi in x]

    max_cols = max(len(xi) for xi in cols)
    cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
    cols = torch.vstack(cols)

    if include_values:
        # vals = (xi.values() for xi in x)
        vals = (torch.from_numpy(xi.data) for xi in x)
        vals = [torch.cat([xi, xi.new_zeros(max_cols - xi.shape[0])]) for xi in vals]
        vals = torch.vstack(vals)
        assert vals.dtype in [torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64], f'> {vals.dtype}'
        cols = torch.stack([cols, vals.type_as(cols)], dim=-1)
    if dim == 2:
        crows = crows[0]
        cols = cols[0]
    return crows.to(torch.int32).to(device), cols.to(torch.int32).to(device)


def crow_col_to_dense(crows: Tensor,
                      cols: Tensor,
                      dtype: torch.dtype=torch.float16
                      ) -> Tensor:
    dim = crows.dim()
    if dim == 1:
        crows = crows[None]
        cols = cols[None]
    device = crows.device
    crows, cols = crows.cpu(), cols.cpu()  # faster in cpu
    shape = (crows.shape[0], crows.shape[1] - 1, cols.max() + 1)
    x = torch.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if cols.dim() == 2:
                x[i, j, cols[i, crows[i, j]:crows[i, j+1]]] = 1
            else:
                x[i, j, cols[i, crows[i, j]:crows[i, j+1], 0]] = cols[i, crows[i, j]:crows[i, j+1], 1]
    if dim == 1:
        x = x[0]
    return x.to(device)


def dense_to_ccol_row(x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    '''Similar, but to CSC format
    '''
    x = x.transpose(-2, -1)
    return dense_to_crow_col(x, **kwargs)


def ccol_row_to_dense(ccol: Tensor,
                      rows: Tensor,
                      dtype: torch.dtype=torch.float16
                      ) -> Tensor:
    return crow_col_to_dense(ccol, rows, dtype).permute(0, 2, 1).contiguous()


def get_sparse_attn_mask(num_heads: int,
                         seqlen: int,
                         block_size: int=64,
                         local_blocks: Union[int, Tuple[int, int]]=4,
                         vert_stride: int=4,
                         homo_head=False,
                         num_kv_heads=None,
                         past_len=None,
                         dtype=torch.bfloat16,
                         device=None,
                         return_dense=False,
                         head_sliding_step=None,
                         head_sliding_offset=0,
                         num_dense_heads=0,
                         q_start_position=None,
                         kv_start_position=None,
                         vert_position_type = None,
                         causal=True):
    """Create the crow sparse layout, thhe block mask and the dense mask.

    # TODO: with num_kv_heads, make the pattern size (num_kv_heads, m_blocks, n_blocks), instead of (num_heads, ..., )
    :param num_heads:
    :param seqlen:
    :param block_size:
    :param local_blocks: int for causal, or (int, int) for (left, right). This number includes the current blocks
    :param head_sliding_step: default to 1 if num_heads >= vert_stride else int(vert_stride / num_heads).
        Skipped if vert_position_type="relative".
    :param head_sliding_offsets:
    :param q_start_position, kv_start_position: for sequence paralllel/chunking
    :param vert_position_type: 'absolute' or 'relative'
        Default to None, i.e., absolute for causal to save kv cache and relative for bidirectional
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it is too big) if `return_dense==True`, otherwise, None
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    past_len = past_len or 0

    if homo_head:
        num_heads = 1
        # head_sliding_offset = 0
        assert num_dense_heads == 0, f"Cannot have dense head if stride pattern for heads is homogeneous."

    if vert_position_type is None:
        vert_position_type = 'absolute' if causal else 'relative'
    else:
        assert vert_position_type in ('absolute', 'relative')

    num_kv_heads = num_kv_heads or num_heads
    num_sparse_heads = num_kv_heads - num_dense_heads
    num_q_per_kv = num_heads // num_kv_heads

    q_start_block = 0
    kv_start_block = 0
    if q_start_position is not None:
        assert q_start_position % block_size == 0, "Sequence chunking must be at the block boundary"
        q_start_block = q_start_position // block_size
        assert past_len in [0, None], f"Cannot specify both past_len and q_start_position"
    if kv_start_position is not None:
        assert kv_start_position % block_size == 0, "Sequence chunking must be at the block boundary"
        kv_start_block = kv_start_position // block_size

    if causal:
        assert local_blocks >= 1
        local_blocks = (local_blocks, 1)  # 1 mean only the current on the right
    else:
        if isinstance(local_blocks, int):
            assert local_blocks >= 0
            local_blocks = (local_blocks, local_blocks)
        else:
            assert isinstance(local_blocks, tuple) and len(local_blocks) == 2
            assert all(x > 0 for x in local_blocks)

    with torch.no_grad():
        blocks = triton.cdiv(seqlen, block_size)
        q_pos = q_start_block + torch.arange(blocks)[None, :, None]
        k_pos = kv_start_block + torch.arange(blocks)[None, None]
        if head_sliding_step is None:
            head_sliding_step = max(1, int(vert_stride / num_heads))  # if vert_stride <= num_heads, rotating the heads
        if vert_position_type == 'absolute':
            # using absolute position for vertical
            mask_vert_strided = [(kv_start_block + torch.arange(blocks) + h * head_sliding_step + head_sliding_offset) % vert_stride == 0
                                if h < num_sparse_heads else torch.ones(blocks).bool()
                                for h in range(num_kv_heads) for _ in range(num_q_per_kv) ]
            mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
            local_mask = ((q_pos - k_pos) < local_blocks[0]).logical_and(k_pos - q_pos < local_blocks[1])
            block_mask = (mask_vert_strided | local_mask)
        else:
            left_mask = (q_pos - k_pos) - local_blocks[0] + 1
            left_mask = (left_mask) * (left_mask > 0).type_as(left_mask)
            right_mask = (k_pos - q_pos) - local_blocks[1] + 1
            right_mask = (right_mask) * (right_mask > 0).type_as(right_mask)
            mask_vert_strided = [(left_mask + right_mask) % vert_stride == 0
                                if h < num_sparse_heads else torch.ones(blocks).bool()
                                for h in range(num_kv_heads) for _ in range(num_q_per_kv) ]
            block_mask = torch.vstack(mask_vert_strided)

        if causal:
            # TODO: make it (num_kv_heads, ) instead of expand to (num_heads,)
            block_mask = (q_pos >= k_pos) & block_mask

        block_mask = block_mask.to(dtype).to(device)

        q_start_block = triton.cdiv(past_len, block_size)
        trimmed_block_mask = block_mask[:, q_start_block:]

    if return_dense:
        mask_dense = torch.kron(block_mask, block_mask.new_ones((block_size, block_size)))[..., :seqlen, :seqlen]
        if causal:
            causal_mask = torch.tril(torch.ones(seqlen, seqlen)).type_as(mask_dense)[past_len:]
            mask_dense = mask_dense[..., past_len:, :] * causal_mask[None]
        return dense_to_crow_col(trimmed_block_mask), trimmed_block_mask, mask_dense
    else:
        return dense_to_crow_col(trimmed_block_mask), trimmed_block_mask, None


def is_kv_cache_efficient(sparse_pattern: Tensor, block_m: int, block_n: int) -> bool:
    """ A sparse pattern is KV cache friendly, when the non-local KV used in generated later tokens
    are also used in generating earlier tokens.

    :param sparse_pattern: (heads, num_m_blocks, num_n_blocks), for homo, 3D expanded, i.e., heads=1
    :param block_m: kernel block_size on q tokens
    :param block_n: kernel block_size on k tokens
    """
    # start pos of block > end pos of block, then it is under the
    sparse_pattern = sparse_pattern.to(torch.int8) # signed
    non_diag = torch.arange(0, sparse_pattern.size(1))[:, None] * block_m >= \
                torch.arange(1, sparse_pattern.size(2) + 1) * block_n
    non_diag = non_diag.to(sparse_pattern.device)
    return (((sparse_pattern[:, 1:] - sparse_pattern[:, :-1]) <= 0) | (~non_diag[1:])).all()


def verify_sparse_pattern(sparse_pattern: Tensor) -> None:
    """Check sparse_pattern dimensions,
    """
    assert sparse_pattern.dim() in (2, 3)
    assert sparse_pattern.size(-2) == sparse_pattern.size(-1)


def is_causal(sparse_pattern: Tensor) -> bool:
    sparse_pattern = sparse_pattern.to(torch.int8)
    assert sparse_pattern.size(-2) == sparse_pattern.size(-1)
    sparse_pattern = sparse_pattern * (1 - torch.tril(torch.ones_like(sparse_pattern[0])))
    return not sparse_pattern.any()


def merge_split_fwd_kernel_blocks(block_size: int,
                                  sparse_pattern: Tensor,
                                  block_m: int,
                                  block_n: int
                                  ) -> Tuple[Tensor, Tensor]:
    """Merge q/m blocks and splits on q/m and/or k/n blocks
    :param block_size: the sparse block size
    :param sparse_pattern: shape=(num_heads, num_blocks, num_blocks).
         Here num_heads can be 1 if homogeneous head pattern used.
    :param block_m: kernel block_size on q tokens
    :param block_n: kernel block_size on k tokens

    :return: new sparse layout
    """
    if block_size == block_m and block_size == block_n:
        return dense_to_crow_col(sparse_pattern)

    # assert not getattr(self, "merged_or_splitted", False), "Already merged or splited"
    assert (block_m % block_size == 0) or (block_size % block_n == 0), \
        f"block_m must be mutliple or a factor of sparse block size for merging on Q blocks"
    assert block_size % block_n == 0, \
        f"block_n must be a factor of sparse block size for splitting on K blocks"

    device = sparse_pattern.device
    # split
    block_m_mult = block_size // block_m  if block_size > block_m else 1
    kron_factor = sparse_pattern.new_ones((block_m_mult, block_size // block_n))

    if kron_factor.numel() > 1:
        _block_m = block_size // kron_factor.size(0)
        _block_n = block_size // kron_factor.size(1)
        sparse_pattern = torch.kron(sparse_pattern, kron_factor)
        causal_mask = torch.arange(1, sparse_pattern.size(1)+1)[:, None] * _block_m \
                    > torch.arange(0, sparse_pattern.size(2)) * _block_n

        sparse_pattern = sparse_pattern * causal_mask.type_as(sparse_pattern)

    if block_size < block_m:
        # merge
        H, M, N = sparse_pattern.shape
        m_mult =  (block_m // block_size)
        assert M % m_mult == 0, f"> {M=}, {m_mult=}"

        # get the last nonzero index + 1,
        sparse_pattern = (sparse_pattern.to(torch.int32).view(H, M // m_mult, m_mult, N).flip([2]).cumsum(2) > 0).sum(2)
        sparse_pattern *= block_size
        return dense_to_crow_col(sparse_pattern, include_values=True)

    return dense_to_crow_col(sparse_pattern)


def zero_stride(x: Tensor, dim=0) -> Tensor:
    """Set the stride of x in dimenion `dim`
       to be 0 for broadcastable dimension `dim`.
    """
    assert x.size(dim) <= 1, f"Cannot set stride at {dim=} to 0 as the size is {x.size(dim)}."
    strides = list(x.stride())
    strides[dim] = 0
    return x.as_strided(x.shape, tuple(strides))


def multiple_of(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def is_hip() -> None:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except:
        return False


def get_q_block_ids(past_len, q_len, block_size):
    start = past_len // block_size
    end = triton.cdiv(past_len + q_len, block_size)
    return range(start, end)
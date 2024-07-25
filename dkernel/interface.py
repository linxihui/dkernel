import torch
import math
import warnings
import copy
from torch import Tensor
from functools import lru_cache
from typing import Tuple, Optional
from dkernel.utils import (get_sparse_attn_mask,
                          dense_to_ccol_row,
                          multiple_of,
                          is_kv_cache_efficient,
                          verify_sparse_pattern,
                          merge_split_fwd_kernel_blocks,
                          zero_stride)
from dkernel.ops import _sparse_attention


class SparseAttention(torch.nn.Module):
    """
    Arguments
    =========
    block_size: sparse_block_size. It has to be power of 2 and minimum of 16.
    sparse_pattern: 2D or 3D (per head) boolean/uint8 Tensor(squared). 1=used, 0=skipped
    seq_dim: Choices=[None, 1, 2]. Default to 1. Only applied to 4D inputs. Ignore if inputs have packed variable lengths.
    block_m, block_n:  The kernel block size for m blocks (q blocks) and n blocks (k/v blocks).
        Default to `block_size` and must be power of 2 with minimum of 16.
        Reduce them if you experience hardware resource constraint (shared memory).
        In that case, try first reduce `block_n`. If it is not enough, `block_m`.
        Latency-wise, reducing `block_n` likely has no impact, will reducing `block_m` may have a bigger impact.
    kwargs: kernel args, do not use unless you know what you're doing.
            "out": to specify an output tensor,
            "d_splits": None/1/2. Number of splits on the HEAD_DIM, Default to 2 if head_dim >=128.
            "inference": None, explicitely specify inference. Default depends on if any of q/k/v requres grad.

    Methods
    =======
    forward:
        :param q, k, v:
            Case 1: for training (requires backward):
                shape=(batch, seq, heads, head_dim) if self.seq_dim=1 or None (default)
                or shape=(batch, heads, seq, head_dim) if self.seq_dim=2.
                Cannot have left paddings
            Case 2: for inference (does not require backward)
                Can be either 4D like in Case 1, with `left_paddings` or right paddings `seqlens`,
                    or 3D with packed sequence in FlashAttn (total_num_tokens, heads, head_dim), where
                    total_num_tokens = sum(seqlens)
        :param sm_scale: softmax scale, default to `1/sqrt(q.size(-1))`.
        :param cu_seqlen_k: shape=(batch+1, ) (0, seqlen1, seqlen1 + seqlen2, ...., sum(seqlens))
            Can only be used at inference.
        :param cu_seqlen_q: shape=(batch+1, ), similar to above, but for q.
            Can only be used at inference.
        :param left_paddings: (batch, ), number of left paddings for each sample.
            Can only be used at inference.
        :param seqlens: real seqlen, can be optionally used when has right padding.
            No need to specify if left_paddings is used.
            Can only be used at inference.
    """
    def __init__(self,
                 block_size: int,
                 sparse_pattern: Tensor,
                 *,
                 seq_dim: Optional[int]=None,
                 block_m: Optional[int]=None,
                 block_n: Optional[int]=None,
                 **kwargs):
        super().__init__()

        def _check_block_size(block_size: int, name: str):
            if block_size is not None:
                assert block_size >=16 and math.log2(block_size) % 1 == 0, \
                    f"{name} must be power of 2 and at least 16, but {block_size} is given."

        _check_block_size(block_size, "block_size")
        _check_block_size(block_m, "block_m")
        _check_block_size(block_n, "block_n")
        assert block_size >=16 and math.log2(block_size) % 1 == 0, \
            f"block_size must be power of 2 and at least 16, but {block_size} is given."

        if block_m is None:
            # TODO: 128?
            block_m = max(block_size, 64)
        if block_n is None:
            block_n = block_size

        sparse_pattern = sparse_pattern.to(torch.uint8)
        if sparse_pattern.dim() == 2:
            sparse_pattern = zero_stride(sparse_pattern[None])

        verify_sparse_pattern(sparse_pattern)
        if not is_kv_cache_efficient(sparse_pattern, block_size, block_size):
            warnings.warn("The provided sparse_pattern is not friendly for KV cache, "
                          "i.e., KV cache needed for later tokens are not used in "
                          "earlier tokens. This may result in unexpected larger KV cache.")

        self.block_size = block_size
        self.block_m = block_m
        self.block_n = block_n
        self.seq_dim = seq_dim
        self.max_seqlen = sparse_pattern.size(1) * block_size
        self.kwargs = copy.copy(kwargs)
        self.kwargs["max_seqlen"] = self.max_seqlen

        # No needed
        # self.register_buffer("sparse_pattern",
        #                      sparse_pattern,
        #                      persistent=False)

        layout_csr = list(merge_split_fwd_kernel_blocks(
                                    block_size,
                                    sparse_pattern,
                                    self.block_m,
                                    self.block_n))
        # TODO: method to avoid this for inference?
        # TODO: need to split on backward for GPUs with less shared memory
        # if self.block_size >= 128:
        #     self.block_m2 = 32
        #     self.block_n2 = 128
        #     layout_csc = list(merge_split_fwd_kernel_blocks(
        #                                 block_size,
        #                                 sparse_pattern,
        #                                 self.block_m2,
        #                                 self.block_n2))
        # else:
        self.block_m2, self.block_n2 = self.block_size, self.block_size
        layout_csc = list(dense_to_ccol_row(sparse_pattern, include_values=False))

        if layout_csr[0].dim() == 1:
            layout_csr = [x[None] for x in layout_csr]
            layout_csc = [x[None] for x in layout_csc]
        if layout_csr[0].size(0) == 1:
            layout_csr = [zero_stride(x) for x in layout_csr]
            layout_csc = [zero_stride(x) for x in layout_csc]

        self.register_buffer("layout_csr_crow", layout_csr[0].to(torch.int32), persistent=False)
        self.register_buffer("layout_csr_col",  layout_csr[1].to(torch.int32), persistent=False)
        self.register_buffer("layout_csc_ccol", layout_csc[0].to(torch.int32), persistent=False)
        self.register_buffer("layout_csc_row",  layout_csc[1].to(torch.int32), persistent=False)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                sm_scale: Optional[int]=None,
                *,
                cu_seqlen_k: Optional[Tensor]=None,
                cu_seqlen_q: Optional[Tensor]=None,
                left_paddings: Optional[Tensor]=None,
                seqlens: Optional[Tensor]=None,
                ) -> Tensor:
        """
        :param q, k, v:
            Case 1: for training (requires backward):
                shape=(batch, seq, heads, head_dim) if self.seq_dim=1 or None (default)
                or shape=(batch, heads, seq, head_dim) if self.seq_dim=2.
                Cannot have left paddings
            Case 2: for inference (does not require backward)
                Can be either 4D like in Case 1, with `left_paddings` or right paddings `seqlens`,
                    or 3D with packed sequence in FlashAttn (total_num_tokens, heads, head_dim), where
                    total_num_tokens = sum(seqlens)
        :param sm_scale: softmax scale, default to `1/sqrt(q.size(-1))`.
        :param cu_seqlen_k: shape=(batch+1, ) (0, seqlen1, seqlen1 + seqlen2, ...., sum(seqlens))
            Can only be used at inference.
        :param cu_seqlen_q: shape=(batch+1, ), similar to above, but for q.
            Can only be used at inference.
        :param left_paddings: (batch, ), number of left paddings for each sample.
            Can only be used at inference.
        :param seqlens: real seqlen, can be optionally used when has right padding.
            No need to specify if left_paddings is used.
            Can only be used at inference.
        """

        # check heads
        hdim = 3 - self.seq_dim if q.dim() == 4 else 1
        assert self.layout_csr_crow.size(0) in [1, k.size(hdim), q.size(hdim)], \
            (f"Input (q, k) have ({q.size(hdim)}, {k.size(hdim)}) heads"
             f"but the number of heads in the sparse pattern is {self.layout_csr_crow.size(0)}")

        # check seqlen
        if q.dim() == 4:
            assert k.size(self.seq_dim) <= self.max_seqlen, \
                (f"Input length {k.size(self.seq_dim)} is larger "
                 f"than the maximum length allowed ({self.max_seqlen})")

        sm_scale = sm_scale or 1. / math.sqrt(float(q.size(-1)))
        inf_kwargs = {"cu_seqlen_k": cu_seqlen_k,
                      "cu_seqlen_q": cu_seqlen_q,
                      "left_paddings": left_paddings,
                      "seqlens": seqlens
                      }

        return _sparse_attention.apply(q, k, v,
                    sm_scale,
                    (self.layout_csr_crow, self.layout_csr_col, self.block_m,  self.block_n),
                    (self.layout_csc_ccol, self.layout_csc_row, self.block_m2, self.block_n2),
                    self.seq_dim,
                    inf_kwargs,
                    self.kwargs
                    )

@lru_cache(maxsize=8)
class LocalStrideSparseAttention(SparseAttention):
    """
    Arguments
    =========
    num_heads: number of q heads
    max_seq_len: max sequence length the model supports.
    block_size: sparse_block_size
    local_blocks: number of local blocks (sliding window)
    vert_stride: for non-local blocks, attention to 1 block per `vert_stride` blocks
    homo_head: the non-local blocks that a token attend to is the same across heads.
    num_dense_heads: numer of heads that are dense. If specified, dense heads are used
        in last heads.
    num_kv_heads: number of kv heads if different from q. If specified, the sparse pattern
        within the same group(shared the same k/v) will be the same for all q heads in the
        group. If this not intended, i.e., k/v within the group have differnt sparse pattern
        (which will results in less KV saving), repeeat the heads of k/v to be the same as q
        when passed to the `forward` method
    active_head_range: the start and end head index for this current partition. This is used
        in any head parallelization, such as Megatron Tensor slicing, DeepSpeed sequence
        parallellization (which turn sequence parallel to head parallel at attention).
    head_sliding_offsets: 0 means block-0 is always attended to at head-0.
        `n` means block-0 is always attended to at head-n.
    block_m, block_n:  The kernel block size for m blocks (q blocks) and n blocks (k/v blocks).
        Default to `block_size` and must be power of 2 with minimum of 16.
        Reduce them if you experience hardware resource constraint (shared memory).
        In that case, try first reduce `block_n`. If it is not enough, `block_m`.
        Latency-wise, reducing `block_n` likely has no impact, will reducing `block_m` may have a bigger impact.

    Methods
    =======
    forward:
        :param q, k, v: shape=(batch, heads, seq, head_dim) if self.seq_dim=2 (default)
              or shape=(batch, seq, heads, head_dim) if self.seq_dim=1.
        :param sm_scale: softmax scale, default to `1/sqrt(q.size(-1))`.

    """
    def __init__(self,
                 num_heads: int,
                 max_seq_len: int,
                 block_size: int,
                 local_blocks: int,
                 vert_stride: int,
                 *,
                 homo_head: bool=False,
                 num_dense_heads: int=0,
                 num_kv_heads: Optional[int]=None,
                 active_head_range: Optional[Tuple[int, int]]=None,
                 head_sliding_offset: int=0,
                 block_m: Optional[int]=None,
                 block_n: Optional[int]=None,
                 **kwargs
                 ):
        assert vert_stride >= 1, "Vertical stride should be position integer. Value 1 will collapse to dense attention."
        if vert_stride > 1:
            assert local_blocks >= 1, "Token in the first block will attend to nothing in some blocks."
        self.max_seq_len = multiple_of(max_seq_len, max([block_size, block_m or 1, block_n or 1, 64]))  # backward issue?
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.homo_head = homo_head
        self.num_dense_heads = num_dense_heads
        sparse_pattern = get_sparse_attn_mask(num_heads, self.max_seq_len,
                                              block_size=block_size,
                                              local_blocks=local_blocks,
                                              vert_stride=vert_stride,
                                              homo_head=homo_head,
                                              dtype=torch.uint8,
                                              num_kv_heads=num_kv_heads,
                                              head_sliding_offset=head_sliding_offset,
                                              num_dense_heads=num_dense_heads)[1]

        # no need to do this for homo heads, as patterns are the same across rangs
        if (not homo_head) and (active_head_range is not None):
            assert isinstance(active_head_range, tuple)
            assert len(active_head_range) == 2, '"active_head_range" should be a tuple of start/end index of the heads.'
            h_start, h_end = active_head_range
            sparse_pattern = sparse_pattern[h_start:h_end]

        super().__init__(block_size, sparse_pattern, block_m=block_m, block_n=block_n, **kwargs)


__all__ = [
    "SparseAttention",
    "LocalStrideSparseAttention",
    ]
import torch
import pytest
from dkernel.utils import get_sparse_attn_mask


def test_get_sparse_attn_mask_causal():
    num_heads = 2
    seqlen = 16
    (crows, ccols), block_mask, dense_mask = get_sparse_attn_mask(
        num_heads=num_heads,
        seqlen=seqlen,
        block_size=2,
        local_blocks=2,
        vert_stride=3,
        causal=True,
        homo_head=True,
        return_dense=True
    )
    # print(f'> {crows=}')
    # print(f'> {ccols=}')
    # print(f'> {block_mask=}')
    # print(f'> {dense_mask=}')

    true_block_mask = torch.tensor(
        [[[1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 0., 0., 0., 0., 0.],
         [1., 0., 1., 1., 0., 0., 0., 0.],
         [1., 0., 0., 1., 1., 0., 0., 0.],
         [1., 0., 0., 1., 1., 1., 0., 0.],
         [1., 0., 0., 1., 0., 1., 1., 0.],
         [1., 0., 0., 1., 0., 0., 1., 1.]]]).type_as(block_mask)

    assert (block_mask == true_block_mask).all()
    

def test_get_sparse_attn_mask_bidirectional():
    num_heads = 2
    seqlen = 16
    (crows, ccols), block_mask, dense_mask = get_sparse_attn_mask(
        num_heads=num_heads,
        seqlen=seqlen,
        block_size=2,
        local_blocks=2,
        vert_stride=3,
        causal=False,
        homo_head=True,
        return_dense=True
    )
    # print(f'> {crows=}')
    # print(f'> {ccols=}')
    # print(f'> {block_mask=}')
    # print(f'> {dense_mask=}')

    true_block_mask = torch.tensor(
        [[[1., 1., 0., 0., 1., 0., 0., 1.],
         [1., 1., 1., 0., 0., 1., 0., 0.],
         [0., 1., 1., 1., 0., 0., 1., 0.],
         [0., 0., 1., 1., 1., 0., 0., 1.],
         [1., 0., 0., 1., 1., 1., 0., 0.],
         [0., 1., 0., 0., 1., 1., 1., 0.],
         [0., 0., 1., 0., 0., 1., 1., 1.],
         [1., 0., 0., 1., 0., 0., 1., 1.]]]).type_as(block_mask)
    assert (block_mask == true_block_mask).all()

    
if __name__ == '__main__':
    test_get_sparse_attn_mask_causal()
    test_get_sparse_attn_mask_bidirectional()
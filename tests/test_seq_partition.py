
import torch
import pytest
import math
from dkernel import get_sparse_attn_mask, SparseAttention
from dkernel.interface import combine_attn_partitions


@pytest.mark.parametrize("b, h, seqlen, d, dtype, homo_head, block_size, num_partitions",
    [
     (1, 1, 512, 64, torch.float32, True, 16, 2),
     (1, 5, 1024, 64, torch.bfloat16, True, 64, 4),
     (2, 3, 2048, 64, torch.float16, False, 32, 8),
    ])
def test_seq_partition(b, h, seqlen, d,
            dtype,
            homo_head,
            block_size,
            num_partitions,
            backward=True,
            local_blocks=4,
            vert_stride=4,
            sm_scale=None,
            non_contiguous=False,
            seq_dim=2,
            num_kv_heads=None,
            causal=True):

    torch.manual_seed(20)
    qlen = seqlen
    assert seq_dim in [1, 2]

    if non_contiguous:
        h2 = h + 17
        seqlen2 = seqlen + 101
        qlen2 = qlen + 19
    else:
        h2, seqlen2, qlen2 = h, seqlen, qlen
    if seq_dim == 2:
        q = torch.empty((b, h2, qlen2  , d), dtype=dtype, device='cuda').normal_(mean=0, std=1)
        k = torch.empty((b, h2, seqlen2, d), dtype=dtype, device='cuda').normal_(mean=0, std=1)
        v = torch.empty((b, h2, seqlen2, d), dtype=dtype, device='cuda').normal_(mean=0, std=1)
        q = q[:, :h, :qlen]
        k = k[:, :h, :seqlen]
        v = v[:, :h, :seqlen]
        h_dim = 1
    else:
        q = torch.empty((b, qlen2  , h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=1)
        k = torch.empty((b, seqlen2, h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=1)
        v = torch.empty((b, seqlen2, h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=1)
        q = q[:, :qlen, :h]
        k = k[:, :seqlen,   :h]
        v = v[:, :seqlen,   :h]
        h_dim = 2

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(d)

    if backward:
        q.requires_grad_(), k.requires_grad_(), v.requires_grad_()
        dout = torch.randn_like(q).contiguous()


    chunk_size = seqlen // num_partitions
    block_masks = [[None for _ in range(num_partitions)] for _ in range(num_partitions)]
    outputs = [[None for _ in range(num_partitions)] for _ in range(num_partitions)]
    layers = [[None for _ in range(num_partitions)] for _ in range(num_partitions)]

    combined_outputs = [None for _ in range(num_partitions)]
    for qc in range(num_partitions):
        for kc in range(0, qc + 1):
            q_start_position = qc * chunk_size
            kv_start_position = kc * chunk_size
            _, _block_mask, _ = get_sparse_attn_mask(h,
                                    chunk_size,
                                    block_size,
                                    local_blocks,
                                    vert_stride,
                                    homo_head=homo_head,
                                    num_kv_heads=num_kv_heads,
                                    dtype=dtype,
                                    return_dense=False,
                                    q_start_position=q_start_position,
                                    kv_start_position=kv_start_position,
                                    causal=causal)

            block_masks[qc][kc] = _block_mask
            _q = q.chunk(num_partitions, dim=seq_dim)[qc]
            _k = k.chunk(num_partitions, dim=seq_dim)[kc]
            _v = v.chunk(num_partitions, dim=seq_dim)[kc]

            layer = SparseAttention(block_size,
                                    _block_mask,
                                    seq_dim=seq_dim,
                                    causal=causal and (qc == kc)
                                    )
            layers[qc][kc] = layer
            
            output = layer(_q, _k, _v, return_lse=True)
            outputs[qc][kc] = output
            
        # combine the outputs
        if qc == 0:
            combined_output = outputs[qc][0][0]
        else:
            combined_output = combine_attn_partitions(*outputs[qc][0:qc+1], hdim=h_dim)
        combined_outputs[qc] = combined_output
    
    output = torch.cat(combined_outputs, dim=seq_dim)
    
    if backward:
        output.backward(dout)
        dv, v.grad = v.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dq, q.grad = q.grad.clone(), None

    # === Baseline/Reference ==
    _, ref_block_mask, _ = get_sparse_attn_mask(h,
                            seqlen,
                            block_size,
                            local_blocks,
                            vert_stride,
                            homo_head=homo_head,
                            num_kv_heads=num_kv_heads,
                            dtype=dtype,
                            return_dense=False,
                            causal=causal)

    ref_layer = SparseAttention(block_size,
                                     ref_block_mask,
                                     seq_dim=seq_dim,
                                     causal=causal)
    
    ref_output = ref_layer(q, k, v, return_lse=False)

    assert torch.allclose(output, ref_output, rtol=1e-2, atol=1e-2), \
        f'> {(output - ref_output).abs().max()=}'
    
    print('=======')
    if backward:
        ref_output.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

        assert torch.allclose(dv, ref_dv, rtol=1e-2, atol=1e-2), f'> diff: {dv - ref_dv}'
        assert torch.allclose(dk, ref_dk, rtol=1e-2, atol=1e-2), f'> diff: {dk - ref_dk}'
        assert torch.allclose(dq, ref_dq, rtol=1e-2, atol=1e-2), f'> diff: {dq - ref_dq}'
        
    print('==== Done ===')


if __name__ == '__main__':
    test_seq_partition(1, 1, 512, 64, block_size=16,
                    dtype=torch.float32, homo_head=False,
                    num_partitions=2)
    test_seq_partition(1, 5, 1024, 64, block_size=64,
                    dtype=torch.bfloat16, homo_head=False,
                    num_partitions=4)
    test_seq_partition(2, 3, 2048, 64, block_size=32,
                    dtype=torch.float16, homo_head=False,
                    num_partitions=8)
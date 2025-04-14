
import torch
import pytest
import math
from dkernel import get_sparse_attn_mask, SparseAttention
from dkernel.interface import combine_attn_partitions


@pytest.mark.parametrize("b, h, seqlen, d, dtype, homo_head, block_size, num_partitions, mode",
    [
     (2, 3, 2048, 64, torch.float16, False, 64, 4, "one-kv"),
     (2, 3, 2048, 64, torch.float16, False, 32, 4, "two-kv"),
     (1, 1, 512, 64, torch.float32, True, 16, 2, "ring"),
     (1, 5, 1024, 64, torch.bfloat16, True, 64, 4, "ring"),
     (2, 3, 2048, 128, torch.float16, False, 32, 8, "ring"),
    ])
def test_seq_partition(b, h, seqlen, d,
            dtype,
            homo_head,
            block_size,
            num_partitions,
            mode, # choices: "ring", "one-kv", "two-kv"
            backward=True,
            local_blocks=4,
            vert_stride=4,
            sm_scale=None,
            non_contiguous=False,
            seq_dim=2,
            num_kv_heads=None,
            causal=True):

    assert causal, "only used for causal now."
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
    if mode == 'ring':
        num_kv_partitons = num_partitions
    elif mode == 'one-kv':
        num_kv_partitons = 1
    elif mode == 'two-kv':
        num_kv_partitons = 2
    else:
        raise ValueError(f'Invalid mode: {mode}')

    block_masks = [[] for _ in range(num_partitions)]
    outputs = [[] for _ in range(num_partitions)]
    layers = [[] for _ in range(num_partitions)]

    combined_outputs = [None for _ in range(num_partitions)]
    for qc in range(num_partitions):

        end_kv = (qc + 1) * chunk_size
        if mode == 'ring':
            start_ends = [(i * chunk_size, (i + 1) * chunk_size) for i in range(qc + 1)]
        elif mode == 'one-kv':
            start_ends = [(0, end_kv)]
        else:
            if qc == 0:
                start_ends = [(0, end_kv)]
            else:
                start_ends = [(0, end_kv - chunk_size), (end_kv - chunk_size, end_kv)]

        for kc, (kv_start_position, kv_end_position) in enumerate(start_ends):
            q_start_position = qc * chunk_size
            qc_end_position = q_start_position + chunk_size
            has_diag_block = kv_end_position == qc_end_position
            # kv_start_position = kc * chunk_size
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
                                    kv_seqlen=kv_end_position - kv_start_position,
                                    causal=causal)

            # print(f'>>>> {qc=}, {kc=}, {has_diag_block=}, {kv_start_position=}, {kv_end_position=}, {_block_mask.shape=}')
            block_masks[qc].append(_block_mask)
            _q = q.chunk(num_partitions, dim=seq_dim)[qc]
            # select q at seq_dim, from kv_start_position to kv_end_position
            if seq_dim == 1:
                _k, _v = [x[:, kv_start_position:kv_end_position] for x in [k, v]]
            else: # seq_dim=2
                _k, _v = [x[:, :, kv_start_position:kv_end_position] for x in [k, v]]

            layer = SparseAttention(block_size,
                                    _block_mask,
                                    seq_dim=seq_dim,
                                    causal=causal and has_diag_block,
                                    )
            layers[qc].append(layer)

            output = layer(_q, _k, _v, return_lse=True)
            outputs[qc].append(output)

        # combine the outputs
        if len(outputs[qc]) == 1:
            combined_output = outputs[qc][0][0]
        else:
            combined_output = combine_attn_partitions(*outputs[qc], hdim=h_dim)
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

    for qc in range(num_partitions):
        qc_block_mask = torch.cat(block_masks[qc], dim=-1)
        ref_qc_block_mask = ref_block_mask.chunk(num_partitions, dim=-2)[qc]
        if ref_qc_block_mask.dim() == 3:
            ref_qc_block_mask = ref_qc_block_mask[:, :, :qc_block_mask.size(2)]
        else:
            ref_qc_block_mask = ref_qc_block_mask[:, :qc_block_mask.size(1)]
        assert (qc_block_mask == ref_qc_block_mask).all(), \
            f'> {qc=}, {ref_qc_block_mask=}, {qc_block_mask=}'

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
        ref_dq, q.grad = q.grad.clone(), None
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None

        assert torch.allclose(dv, ref_dv, rtol=1e-2, atol=1e-2), f'> diff: {dv - ref_dv}'
        assert torch.allclose(dk, ref_dk, rtol=1e-2, atol=1e-2), f'> diff: {dk - ref_dk}'
        assert torch.allclose(dq, ref_dq, rtol=1e-2, atol=1e-2), f'> diff: {dq - ref_dq}'

    print('==== Done ===')


if __name__ == '__main__':
    test_seq_partition(2, 3, 2048, 64, block_size=64,
                    dtype=torch.float32, homo_head=True,
                    num_partitions=4, mode="one-kv")
    # test_seq_partition(1, 1, 512, 64, block_size=16,
    #                 dtype=torch.float32, homo_head=False,
    #                 num_partitions=2)
    # test_seq_partition(1, 5, 1024, 64, block_size=64,
    #                 dtype=torch.bfloat16, homo_head=False,
    #                 num_partitions=4)
    # test_seq_partition(2, 3, 2048, 64, block_size=32,
    #                 dtype=torch.float16, homo_head=False,
                    # num_partitions=8)
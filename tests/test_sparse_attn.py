import torch
import pytest
import math
from dkernel import get_sparse_attn_mask, LocalStrideSparseAttention


### all tests ###


def test_inference_padding_varlen():
    print('### Test inference prompting and decoding ###')
    # 1.0, left padding: prompt
    general_test_fn(2, 8, 4512, 128,  past_len=0, max_seqlen=8192,
            block_size=64, backward=False,
            dtype=torch.bfloat16, homo_head=False,
            has_left_paddings=True, seq_dim=1)
    # seq_dim=2, homo, float16, smaller block size
    general_test_fn(2, 8, 4096, 128,  past_len=0, max_seqlen=8192,
            block_size=16, backward=False,
            dtype=torch.float16, homo_head=True,
            has_left_paddings=True, seq_dim=2)

    # 1.1, left padding: decoding
    general_test_fn(2, 8, 1, 128,  past_len=64*21 + 12, max_seqlen=8192,
            block_size=64, backward=False,
            dtype=torch.bfloat16, homo_head=False,
            has_left_paddings=True, seq_dim=1)

    general_test_fn(2, 8, 1, 128,  past_len=64*5, max_seqlen=1024,
            block_size=32, backward=False,
            dtype=torch.bfloat16, homo_head=True,
            local_blocks=2,
            has_left_paddings=True, seq_dim=1)

    # 2.0, right padding: prompt
    general_test_fn(2, 8, 1421, 128,  past_len=0, max_seqlen=8192,
            block_size=64, backward=False,
            dtype=torch.bfloat16, homo_head=False,
            has_right_paddings=True, seq_dim=2)

    # 2.1, right padding: decoding
    general_test_fn(2, 8, 1, 128,  past_len=64*18 + 1, max_seqlen=8192,
            block_size=32, backward=False,
            dtype=torch.bfloat16, homo_head=False,
            has_left_paddings=True, seq_dim=1)

    # 3.0, varlen, prompt
    general_test_fn(2, 8, 2048, 128,  past_len=0, max_seqlen=8192,
            block_size=32, backward=False,
            dtype=torch.bfloat16, homo_head=False,
            varlen=True, seq_dim=1)
    # 3.1, varlen, decoding: not needed, bad KV cache design
    # 3.2, vllm blocktable


def test_dims_and_dtype():
    # diff dim, float types (float16, bfloat16, float32)
    print("### Test different head dims(64, 128, 256)" + \
            ", float types (float16, bfloat16, float32) ###")
    general_test_fn(1, 1, 1024, 64, block_size=64,
            dtype=torch.float16, homo_head=False)
    general_test_fn(2, 8, 1024, 64, block_size=64,
            dtype=torch.float32, homo_head=False)
    general_test_fn(2, 8, 1024, 128, block_size=64,
        dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 1024, 256, block_size=32,
        dtype=torch.bfloat16, homo_head=False)


def test_noncausal():
    "bidirectional attention"
    general_test_fn(1, 2, 2048, 64, local_blocks=4, vert_stride=8, block_size=64,
                    dtype=torch.float16, homo_head=False, causal=False)
    general_test_fn(1, 1, 2048, 64, local_blocks=4, vert_stride=10000, block_size=32,
                    dtype=torch.float16, homo_head=False, causal=False)


def test_dims_no_power_of_2():
    # head_dim not power of 2
    general_test_fn(2, 2, 2048, 48, block_size=64,
            dtype=torch.float32, homo_head=True)
    general_test_fn(2, 4, 2048, 96, block_size=64,
            dtype=torch.bfloat16, homo_head=True)
    general_test_fn(1, 4, 4096, 160, block_size=64,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(1, 5, 2048, 192, block_size=64,
            dtype=torch.float16, homo_head=False)


def test_seqlen():
    # diff seqlen
    print("### Test different sequence lengths ###")
    general_test_fn(2, 8, 15, 128,  block_size=64,
        dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 1028, 128,  block_size=32,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 5175, 128,  block_size=64,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            dtype=torch.bfloat16, homo_head=False)


def test_inference_no_padding():
    print('### Test inference prompting and decoding ###')
    # short prompt
    general_test_fn(2, 8, 15, 128,  past_len=0, max_seqlen=8192,
            block_size=64, backward=False,
            dtype=torch.bfloat16, homo_head=False)
    # decoding
    general_test_fn(2, 8, 1, 128,  past_len=64*5 + 12, max_seqlen=8192,
            block_size=64, backward=False,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 1, 128,  past_len=64*50 + 12, max_seqlen=8192,
            block_size=16, backward=False,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 1, 128,  past_len=64*50 + 12, max_seqlen=8192,
            block_size=16, backward=False, block_m=128,
            dtype=torch.bfloat16, homo_head=False)
    # # chunked prefilling, not supported yet
    # general_test_fn(2, 8, 12, 128,  past_len=2273, max_seqlen=8192,
    #         block_size=64, backward=False,
    #         dtype=torch.bfloat16, homo_head=False)
    # general_test_fn(1, 1, 65, 128,  past_len=120, max_seqlen=8192,
    #     block_size=64, backward=False,
    #     dtype=torch.bfloat16, homo_head=False)


def test_patterns():
    print("### Test difference sparsity, including dense heads, homo pattern ###")
    # diff local, vertical stride
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            local_blocks=10, vert_stride=3,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            local_blocks=1, vert_stride=8,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            local_blocks=8, vert_stride=1,
            dtype=torch.bfloat16, homo_head=False)
    # with dense heads
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            num_dense_heads=2,
            dtype=torch.bfloat16, homo_head=False)
    # homo head
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            dtype=torch.bfloat16, homo_head=True)


def test_sparse_block_sizes():
    # diff block size, with default block_m, block_n
    print("### Test different sparse block_size ###")
    general_test_fn(2, 8, 8192, 128,  block_size=16,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=32,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            dtype=torch.bfloat16, homo_head=False)
    # general_test_fn(2, 8, 8192, 128,  block_size=128,
    #         dtype=torch.bfloat16, homo_head=False,
    #         block_m=64, block_n=32)
    general_test_fn(2, 8, 8192, 64,  block_size=128,
        dtype=torch.bfloat16, homo_head=False)


def test_kernel_block_sizes():
    # diff block_m, block_n
    print("### Test different kernel block size ###")
    general_test_fn(2, 8, 8192, 128,  block_size=16,
            block_m=16, block_n=16,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=16,
            block_m=128, block_n=16,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=64,
            block_m=128, block_n=32,
            dtype=torch.bfloat16, homo_head=False)

    # # The following will suffer SRAM capacity in A600 GPU
    # general_test_fn(2, 8, 8192, 128,  block_size=128,
    #         block_m=64, block_n=64,
    #         dtype=torch.bfloat16, homo_head=False)
    # potential interact on diag blocks
    general_test_fn(2, 8, 8192, 128,  block_size=16,
            block_m=128, block_n=16, local_blocks=1,
            dtype=torch.bfloat16, homo_head=False)


def test_seq_dim_non_contiguous_qkv():
    # seq_dim=1
    print("### Test seq_dim=1 and non-contiguous qkv layout ###")
    general_test_fn(2, 8, 8192, 128,  block_size=64, seq_dim=1,
            dtype=torch.bfloat16, homo_head=False)
    general_test_fn(2, 8, 8192, 128,  block_size=64, seq_dim=1,
            non_contiguous=True,
            dtype=torch.bfloat16, homo_head=False)
    # non-contiguous q/k/v
    general_test_fn(2, 8, 8192, 128,  block_size=64, non_contiguous=True,
            dtype=torch.bfloat16, homo_head=False)


def test_d_splits():
    """
    By Default, d_splits=1 if head_dim <= 64 else 2.
    """
    # d_plits=1 for 128, 256. Default is 2
    print("Teset non default d_splits for different head size.")
    general_test_fn(2, 8, 8192, 128,  block_size=64, d_splits=1,
            dtype=torch.bfloat16, homo_head=False)
    # general_test_fn(2, 8, 8192, 256,  block_size=64, d_splits=1,
    #         dtype=torch.bfloat16, homo_head=False)
    # d_splits=2 for 64
    general_test_fn(2, 8, 8192, 64,  block_size=64, d_splits=2,
            dtype=torch.bfloat16, homo_head=False)


def test_qkv_packing():
    general_test_fn(2, 8, 8192, 128,  block_size=64, d_splits=None,
            dtype=torch.bfloat16, homo_head=False, qkv_pack_mode='packed_kv')
    general_test_fn(2, 8, 8192, 128,  block_size=64, d_splits=None,
            dtype=torch.bfloat16, homo_head=False, qkv_pack_mode='packed_qkv')


# @pytest.mark.parametrize('b, h, seqlen, d', [(2, 8, 2048, 128), (1, 4, 4096, 64)])
def general_test_fn(b, h, seqlen, d,
            past_len=None,
            dtype=torch.bfloat16,
            homo_head=False,
            block_size=64,
            backward=True,
            sparse_attention_fn=None,
            local_blocks=4,
            vert_stride=4,
            sm_scale=None,
            num_dense_heads=0,
            max_seqlen=None,
            block_m=None,
            block_n=None,
            non_contiguous=False,
            d_splits=None,
            seq_dim=2,
            head_sliding_offset=0,
            num_kv_heads=None,
            varlen=False,
            std=1,
            has_left_paddings=False,
            has_right_paddings=False,
            qkv_pack_mode='split_qkv',
            causal=True):

    qlen = seqlen
    past_len = past_len or 0
    seqlen = qlen + past_len

    max_seqlen = max_seqlen or seqlen
    torch.manual_seed(20)

    assert seq_dim in [1, 2]

    if non_contiguous:
        h2 = h + 17
        seqlen2 = seqlen + 101
        qlen2 = qlen + 19
    else:
        h2, seqlen2, qlen2 = h, seqlen, qlen
        
    assert qkv_pack_mode in ('split_qkv', 'packed_qkv', 'packed_kv')
    if seq_dim == 2:
        # if qkv_pack_mode == 'packed_qkv':
        #     assert qlen == seqlen
        #     qkv = torch.empty((b, 3 * h, seqlen, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #     q, k, v = qkv.chunk(3, dim=1)
        # else:
        #     q = torch.empty((b, h2, qlen2  , d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #     if qkv_pack_mode == 'packed_kv':
        #         kv = torch.empty((b, 2 * h, seqlen, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #         k, v = kv.chunk(2, dim=1)
        #     else:
        #         k = torch.empty((b, h2, seqlen2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #         v = torch.empty((b, h2, seqlen2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)

        q = torch.empty((b, h2, qlen2  , d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        k = torch.empty((b, h2, seqlen2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        v = torch.empty((b, h2, seqlen2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        q = q[:, :h, :qlen]
        k = k[:, :h, :seqlen]
        v = v[:, :h, :seqlen]
        h_dim = 1
    else:
        # if qkv_pack_mode == 'packed_qkv':
        #     assert qlen == seqlen
        #     qkv = torch.empty((b, seqlen, 3 * h, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #     q, k, v = qkv.chunk(3, dim=1)
        # else:
        #     q = torch.empty((b, qlen2 , h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #     if qkv_pack_mode == 'packed_kv':
        #         kv = torch.empty((b, seqlen, 2 * h, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #         k, v = kv.chunk(2, dim=1)
        #     else:
        #         k = torch.empty((b, seqlen2, h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        #         v = torch.empty((b, seqlen2, h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        q = torch.empty((b, qlen2 , h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        k = torch.empty((b, seqlen2, h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        v = torch.empty((b, seqlen2, h2, d), dtype=dtype, device='cuda').normal_(mean=0, std=std)
        q = q[:, :qlen, :h]
        k = k[:, :seqlen,   :h]
        v = v[:, :seqlen,   :h]
        h_dim = 2

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(d)

    sm_scale = 0.0078125
    if backward:
        q.requires_grad_(), k.requires_grad_(), v.requires_grad_()

    dout = torch.randn_like(q).contiguous()

    mask_csr, block_mask, mask_dense = get_sparse_attn_mask(k.size(h_dim), k.size(seq_dim),
                                                            block_size=block_size,
                                                            local_blocks=local_blocks,
                                                            vert_stride=vert_stride,
                                                            homo_head=homo_head,
                                                            num_kv_heads=num_kv_heads,
                                                            num_dense_heads=num_dense_heads,
                                                            head_sliding_offset=head_sliding_offset,
                                                            return_dense=True,
                                                            causal=causal)

    # reference implementation
    if has_left_paddings or has_right_paddings or varlen:
        backward = False
        # print(f'>> {mask_dense.shape=}')
        if past_len != 0:
            lens = torch.randint(qlen, k.size(seq_dim), size=(k.size(0),))
            if has_left_paddings:
                mask_dense0 = mask_dense[None].repeat((b, 1, 1, 1))
                mask_dense = mask_dense.new_zeros((b, h, q.size(seq_dim), k.size(seq_dim)))
                for i, l in enumerate(lens):
                    mask_dense[i, :, :qlen, :l] = mask_dense0[i, :, (l - qlen):l, :l]
        else:
            lens = torch.randint(1, k.size(seq_dim), size=(k.size(0),))
    else:
        mask_dense = mask_dense[..., past_len:seqlen, :seqlen]

    ref_out = torch_attention(q, k, v, mask_dense, sm_scale, seq_dim=seq_dim, causal=causal)
    ref_out0 = ref_out.clone()

    if backward:
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

        if qkv_pack_mode == 'packed_qkv':
            qkv = torch.cat([q, k, v], dim=h_dim).detach()
            qkv.requires_grad_()
            split_qkv = lambda x: x.chunk(3, dim=h_dim)
            q, k, v = split_qkv(qkv)
        elif qkv_pack_mode == 'packed_kv':
            kv = torch.cat([k, v], dim=h_dim).detach()
            kv.requires_grad_()
            split_qkv = lambda x: x.chunk(2, dim=h_dim)
            k, v = split_qkv(kv)
        else:
            split_qkv = None

    if sparse_attention_fn is None:
        sparse_attention_fn = LocalStrideSparseAttention(
                                                        h, max_seqlen,
                                                        block_size,
                                                        local_blocks,
                                                        vert_stride,
                                                        homo_head=homo_head,
                                                        num_kv_heads=num_kv_heads,
                                                        num_dense_heads=num_dense_heads,
                                                        block_m=block_m,
                                                        block_n=block_n,
                                                        d_splits=d_splits,
                                                        seq_dim=seq_dim,
                                                        head_sliding_offset=0,
                                                        causal=causal
                                                        )

        sparse_attention_fn.to(q.device).to(q.dtype)

    if varlen:
        # lens = torch.randint(0, k.size(seq_dim), size=(k.size(0),))
        cu_seqlen_k = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).to(torch.int32).to(q.device)
        assert seq_dim == 1
        k = torch.cat([x.narrow(0, 0, l) for l, x in zip(lens, k)], dim=0)
        v = torch.cat([x.narrow(0, 0, l) for l, x in zip(lens, v)], dim=0)
        if past_len == 0:
            q = torch.cat([x.narrow(0, 0, l) for l, x in zip(lens, q)], dim=0)
        else:
            assert q.size(1) == 1
            q = q.view(-1, q.size(2), q.size(3))

        tri_out = sparse_attention_fn(q, k, v, sm_scale, cu_seqlen_k=cu_seqlen_k)
        ref_out = torch.cat([x.narrow(0, 0, l) for l, x in zip(lens, ref_out)], dim=0)

    elif has_left_paddings or has_right_paddings:
        q0, k0, v0 = [x.clone() for x in [q, k, v]]
        # lens = torch.randint(0, k.size(seq_dim), size=(k.size(0),))
        # import ipdb; ipdb.set_trace()
        if has_right_paddings:
            tri_out = sparse_attention_fn(q, k, v, sm_scale=sm_scale, seqlens=lens)

            if past_len == 0:
                for i, l in enumerate(lens):
                    padding = k.size(seq_dim) - l
                    ref_out[i].narrow(seq_dim - 1, l, padding)[:] = 0
                    tri_out[i].narrow(seq_dim - 1, l, padding)[:] = 0
                    # if seq_dim == 1:
                    #     ref_out[i, l:] = 0
                    #     tri_out[i, l:] = 0
                    # else:
                    #     ref_out[i, :, l:] = 0
                    #     tri_out[i, :, l:] = 0
        elif has_left_paddings:
            left_paddings = k.size(seq_dim) - lens
            for i, len in enumerate(lens):
                left = k.size(seq_dim) - len
                # import ipdb; ipdb.set_trace()
                k[i].narrow(seq_dim - 1, left, len)[:] = k[i].narrow(seq_dim - 1, 0, len).clone()
                v[i].narrow(seq_dim - 1, left, len)[:] = v[i].narrow(seq_dim - 1, 0, len).clone()
                # k[i, left:] = k[i, :len]
                # v[i, left:] = v[i, :len]
                if past_len == 0:
                    q[i].narrow(seq_dim - 1, left, len)[:] = q[i].narrow(seq_dim - 1, 0, len).clone()
                    # q[i, left:] = q[i, :len]
            # import ipdb; ipdb.set_trace()
            tri_out = sparse_attention_fn(q, k, v, sm_scale=sm_scale, left_paddings=left_paddings)
            if past_len == 0:
                for i, len in enumerate(lens):
                    left = k.size(seq_dim) - len
                    tri_out[i].narrow(seq_dim - 1, 0, left)[:] = 0
                    ref_out[i].narrow(seq_dim - 1, left, len)[:] = ref_out[i].narrow(seq_dim - 1, 0, len).clone()
                    ref_out[i].narrow(seq_dim - 1, 0, left)[:] = 0
                    # tri_out[i, :left] = 0
                    # ref_out[i, left:] = ref_out[i, :len]
                    # ref_out[i, :left] = 0
    else:
        if qkv_pack_mode == 'split_qkv':
            tri_out = sparse_attention_fn(q, k, v, sm_scale=sm_scale)
        elif qkv_pack_mode == 'packed_kv':
            tri_out = sparse_attention_fn(q, kv, sm_scale=sm_scale, split_qkv=split_qkv)
        else:
            tri_out = sparse_attention_fn(qkv, sm_scale=sm_scale, split_qkv=split_qkv)

    # import ipdb; ipdb.set_trace()
    # decimal = 1 if dtype == torch.bfloat16 else 2
    assert torch.allclose(ref_out.cpu(), tri_out.cpu(), atol=1e-2, rtol=1e-2), \
        "---- out ---\n" + print_diff_stats(ref_out, tri_out)

    if backward:
        tri_out.backward(dout)
        if qkv_pack_mode == 'packed_qkv':
            tri_dqkv, qkv.grad= qkv.grad.clone(), None
            tri_dq, tri_dk, tri_dv = split_qkv(tri_dqkv)
        elif qkv_pack_mode == 'packed_kv':
            tri_dkv, kv.grad = kv.grad.clone(), None
            tri_dk, tri_dv = split_qkv(tri_dkv)
            tri_dq, q.grad = q.grad.clone(), None
        else:
            tri_dv, v.grad = v.grad.clone(), None
            tri_dk, k.grad = k.grad.clone(), None
            tri_dq, q.grad = q.grad.clone(), None

    if backward:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=1e-2), \
            "---- dv ---\n" + print_diff_stats(ref_dv, tri_dv)
        assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0), \
            "---- dk ---\n" + print_diff_stats(ref_dk, tri_dk)
        assert torch.allclose(ref_dq, tri_dq, atol=2e-2, rtol=0), \
            "---- dq ---\n" + print_diff_stats(ref_dq, tri_dq)

    print('-'*80,
            f'\nTest passed:'
            f'\n  {b=}, {h=}, {seqlen=}, {d=}, {past_len=}, {dtype=}',
            f'\n  {homo_head=}, {block_size=}, {local_blocks=}, {vert_stride=}'
            f'\n  {block_m=}, {block_n=}, {num_dense_heads=}, {d_splits=}',
            f'\n  {seq_dim=}, {non_contiguous=}'
            '\n' + f'-'*80)


def torch_attention(q, k, v,
                    attn_mask=None,
                    sm_scale=None,
                    block_attn_mask=None,
                    block_size=128,
                    do=None,
                    seq_dim=2,
                    causal=True):
    """
    :param q, k, v: shape=(batch, heads, seq, head_dim) if seq_dim=2 (default)
            or shape=(batch, seq, heads, head_dim) if seq_dim=1.
    """
    # for verification
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(float(q.size(-1)))

    assert seq_dim in [1, 2]
    if seq_dim == 1:
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
        if do is not None:
            do = do.transpose(1, 2)

    if block_attn_mask is not None:
        assert attn_mask is None
        outs = []
        for s in range(0, q.size(2), block_size):
            e = min(s + block_size, q.size(2))
            q_block = q[:, :, s:e]
            attn = torch.einsum('bhmd,bhnd->bhmn', q_block, k[:, :, :e]).float() * sm_scale
            mask = block_attn_mask[..., s // block_size, : (s // block_size + 1)]
            mask = torch.kron(mask, torch.ones(block_size, block_size, device=mask.device))
            if causal:
                mask[..., :, s:].masked_fill_(torch.arange(0, block_size)[:, None] <= torch.arange(0, block_size)[None, :], 0)
                attn = attn.masked_fill((1 - mask).bool(), float('-inf'))
            attn = attn.softmax(-1)
            out = torch.einsum('bhmn,bhnd->bhmd', attn.type_as(v), v[:, :, :e])
            outs.append(out)
        torch_output = torch.cat(outs, dim=2)
    else:
        attn = torch.einsum('bhmd,bhnd->bhmn', q, k).float() * sm_scale
        if attn_mask is not None:
            attn = attn.masked_fill((1 - attn_mask).bool(), float('-inf'))

        attn = attn.softmax(-1)
        if do is not None:
            dv = torch.einsum('bhqk,bhqd->bhkd', attn.type_as(do), do)
            print(f'> torch_attn computed dv: {dv=}')
        torch_output = torch.einsum('bhmn,bhnd->bhmd', attn.type_as(v), v)

    if seq_dim == 1:
        torch_output = torch_output.transpose(1, 2)
    return torch_output


def print_diff_stats(ref, tri):
    abs_v, abs_i = (ref - tri).abs().ravel().max(-1)
    tri2 = tri[ref.abs() > 1e-5]
    ref2 = ref[ref.abs() > 1e-5]

    sel_mask = torch.rand_like(tri2) > 0.9
    rel_v, rel_i = (tri2.abs() / ref2.abs() - 1).abs().ravel().max(0)
    rel_quantile_99 = (tri2.abs() / ref2.abs()).float()[sel_mask].quantile(0.99).item()
    coord = tuple(x[0].item() if len(x) else -1 for x in torch.where((ref - tri).abs() == abs_v))

    diff = (ref - tri).abs().float()
    sel_mask = torch.rand_like(diff) > 0.9
    diff = diff[sel_mask]

    msg = (f'>> rel_diff={rel_v.item()}, ref={ref2[rel_i]}, tri={tri2[rel_i]}, {rel_quantile_99=}\n'
            f'.. max_abs_diff={abs_v.item()}, {ref.ravel()[abs_i].item()=}, {tri.ravel()[abs_i].item()=}, {coord=}\n'
            f'.. {(ref - tri).abs().mean().item()=},\n'
            f'.. {diff.quantile(0.9).item()=},\n'
            f'.. {diff.quantile(0.99).item()=},\n'
            f'.. {diff.quantile(0.999).item()=},\n'
            f'.. {ref[sel_mask].abs().float().quantile(0.99).item()=},\n'
            f'.. {ref[sel_mask].abs().float().quantile(0.999).item()=},\n'
            f'.. {ref.abs().mean().item()=},\n'
            f'.. {(ref - tri).abs().ravel().topk(10).values=}')
    return (msg)


if __name__ == '__main__':
    test_qkv_packing()
    general_test_fn(1, 1, 1024, 128, vert_stride=4, local_blocks=2, block_size=16,
                    dtype=torch.bfloat16, homo_head=False,
                    block_m=128, block_n=16, std=2,
                    # bwd_block_sizes=(128, 16, 128, 16)
                    )
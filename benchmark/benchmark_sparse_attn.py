
import torch
import triton
import os
from dkernel import LocalStrideSparseAttention

try:
    import flash_attn
    from flash_attn import flash_attn_func
    print(f'>> {flash_attn.__version__=}')
    HAS_FLASH = True
except BaseException as e:
    print(f'> error to import flash_attn: {e=}')
    HAS_FLASH = False

try:
    from triton.ops.flash_attention import attention as triton_attention
    HAS_DENSE_TRITON_FLASH = False
except BaseException as e:
    print(f'> error to import triton flash_attn: {e=}')
    HAS_DENSE_TRITON_FLASH = False


def run_bench(b=4,
              h=16,
              d=128,
              local_tokens=512,
              vert=16,
              d_splits=None, block_size=64,
              block_m=None, block_n=None,
              support_backward:bool=True,
              save_path='.',
              log2_seqlen:int=16,
              num_dense_heads=0,
              homo_head=False,
            #   run_test=True,
              seq_dim=2,
             ):
    sparse_type = 'homo' if homo_head else 'hetero'
    dtype = torch.bfloat16
    # d_splits = 2

    BATCH, N_HEADS, D_HEAD = b, h, d
    assert local_tokens % block_size == 0
    local = local_tokens // block_size
    BLOCK_SIZE, LOCAl_BLOCKS, VERT_STRIDE = block_size, local, vert

    # if run_test:
    #     for seqlen in [1024, 1028]:
    #         test_op(b, h, seqlen, d,
    #                 block_size=block_size,
    #                 local_blocks=local,
    #                 vert_stride=vert,
    #                 dtype=dtype,
    #                 homo_head=homo_head,
    #                 backward=support_backward,
    #                 seq_dim=seq_dim,
    #                 block_m=block_m,
    #                 block_n=block_n)

    # os.environ["BS_DEBUG"] = "0"

    modes = ['fwd', 'bwd'] if support_backward else ['fwd']

    configs = [triton.testing.Benchmark(
        x_names=['SEQ_LEN'],
        x_vals=[2**i for i in range(10, log2_seqlen + 1)],
        line_arg='provider',
        line_vals=(['triton'] if HAS_DENSE_TRITON_FLASH else []) + \
                ['triton_sparse'] + (['flash'] if HAS_FLASH else []),
        line_names=(['Triton-Dense'] if HAS_DENSE_TRITON_FLASH else []) + \
                [f'Triton-S2 (V={VERT_STRIDE})']+ (['Flash2-Dense'] if HAS_FLASH else []),
        styles=[('red', '-'), ('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name=(f'fused-blocksparse-attention-v2-batch{BATCH}-head{N_HEADS}-d{D_HEAD}'
                    f'-sparse-local{LOCAl_BLOCKS}-vert{VERT_STRIDE}-{sparse_type}'
                    f'-dsplit-{d_splits}-{dtype}-{mode}'),
        args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': dtype, 'mode': mode}
    ) for mode in modes]


    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, SEQ_LEN, D_HEAD, mode, provider, dtype=torch.bfloat16, device='cuda'):
        assert mode in ['fwd', 'bwd']
        warmup = 25
        rep = 100
        sm_scale = 1.3

        q = torch.randn((BATCH, SEQ_LEN, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn((BATCH, SEQ_LEN, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn((BATCH, SEQ_LEN, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)

        if provider == 'triton':
            # q = torch.randn((BATCH, H, SEQ_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            # k = torch.randn((BATCH, H, SEQ_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            # v = torch.randn((BATCH, H, SEQ_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
            fn = lambda: triton_attention(q, k, v, True, sm_scale)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'triton_sparse':
            # q = torch.randn((BATCH, H, SEQ_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            # k = torch.randn((BATCH, H, SEQ_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            # v = torch.randn((BATCH, H, SEQ_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)

            if seq_dim == 2:
                q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
            sparse_attention_fn = LocalStrideSparseAttention(H, SEQ_LEN,
                                                            BLOCK_SIZE,
                                                            LOCAl_BLOCKS,
                                                            VERT_STRIDE,
                                                            d_splits=d_splits,
                                                            num_dense_heads=num_dense_heads,
                                                            block_m=block_m,
                                                            block_n=block_n,
                                                            seq_dim=seq_dim)
            sparse_attention_fn.to(q.device).to(q.dtype)
            fn = lambda: sparse_attention_fn(q, k, v, sm_scale)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'flash':
            fn = lambda: flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=True)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bench_flash_attention.run(save_path=save_path, print_data=True)


if __name__ == "__main__":
    import fire
    fire.Fire(run_bench)
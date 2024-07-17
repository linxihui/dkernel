import torch
from s2attn import LocalStrideSparseAttention

try:
    import flash_attn
    from flash_attn import flash_attn_func
    print(f'>> {flash_attn.__version__=}')
    HAS_FLASH = True
except BaseException as e:
    print(f'> error to import flash_attn: {e=}')
    HAS_FLASH = False


def run_profiling(
        b=4, h=16, d=128, seqlen=8192,
        local_tokens=512, vert=16,
        d_splits=None, block_size=64,
        block_m=None, block_n=None,
        save_path='./profiing_s2attn',
        num_dense_heads=0,
        homo_head=False,
        ):

    assert local_tokens % block_size == 0
    local_blocks = local_tokens // block_size
    block_m, block_n = 128, 32
    dtype = torch.bfloat16
    sm_scale = 0.1

    torch.manual_seed(20)
    q = torch.empty((b, h, seqlen, d), dtype=dtype, device='cuda').normal_(mean=0, std=1).requires_grad_()
    k = torch.empty((b, h, seqlen, d), dtype=dtype, device='cuda').normal_(mean=0, std=1).requires_grad_()
    v = torch.empty((b, h, seqlen, d), dtype=dtype, device='cuda').normal_(mean=0, std=1).requires_grad_()
    dout = torch.randn_like(q).contiguous()

    # warmup
    sparse_attention_fn = LocalStrideSparseAttention(h, seqlen,
                                                    block_size,
                                                    local_blocks,
                                                    vert,
                                                    num_kv_heads=h,
                                                    homo_head=homo_head,
                                                    d_splits=d_splits,
                                                    num_dense_heads=num_dense_heads,
                                                    block_m=block_m,
                                                    block_n=block_n)
    sparse_attention_fn.to(q.device)
    tri_out = sparse_attention_fn(q, k, v, sm_scale)
    tri_out.backward(dout)

    q1 = torch.randn((b, seqlen, h, d), dtype=dtype, device='cuda', requires_grad=True)
    k1 = torch.randn((b, seqlen, h, d), dtype=dtype, device='cuda', requires_grad=True)
    v1 = torch.randn((b, seqlen, h, d), dtype=dtype, device='cuda', requires_grad=True)
    dout1 = torch.randn_like(q1).contiguous()

    flash_out = flash_attn_func(q1, k1, v1, softmax_scale=sm_scale, causal=True)
    flash_out.backward(dout1)

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{save_path}/m{block_m}-n{block_n}'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for _ in range(10):
            prof.step()
            q.grad, k.grad, v.grade = None, None, None
            tri_out = sparse_attention_fn(q, k, v, sm_scale)
            tri_out.backward(dout)

            q1.grad, k1.grad, v1.grade = None, None, None
            flash_out = flash_attn_func(q1, k1, v1, softmax_scale=sm_scale, causal=True)
            flash_out.backward(dout1)


if __name__ == "__main__":
    import fire
    fire.Fire(run_profiling)
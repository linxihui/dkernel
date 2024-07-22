# Introduction

This repo contains customized CUDA kernel written in OpenAI Triton.
As of now, it contains the sparse attention kernel used in [phi-3-small models](https://huggingface.co/microsoft/Phi-3-small-8k-instruct).
This sparse attention is also supported in vLLM for fast inference.

Refernece:

# Install

```sh
pip install git+https://github.com/linxihui/dkernel
```

# Quick start

```python
import torch
from dkernel import SparseAttention, LocalStrideSparseAttention


# 1.) Using local-stride pattern

block_size = 64 # sparse block size, minimum 16
local_blocks = 32 # num local blocks, always attend to up to 64 * 16=1024 token
vert_stride = 8 # attend to 1 block per every 8 blocks after the local window above
max_seq_len = 8192 # model supports up to 8192 seqlen
num_heads = 32
device = "cuda"

q, k, v = [torch.rand(2, 8192, 32, 128, device=device).requires_grad_() for _ in range(3)]

attn = LocalStrideSparseAttention(
                 num_heads,
                 max_seq_len, 
                 block_size, 
                 local_blocks, 
                 vert_stride,
                 seq_dim=1, # q/k/v layout: (batch, seqlen, num_heads, head_dim)
                )
attn.to(device) # optional, attn default to current_device

# This first time, it needs to warmup, so could be slow.
attn(q, k, v)

# Now should be fast
ouput = attn(q, k, v)

# 2.) Using user defined arbitrary pattern

num_blocks = max_seq_len // block_size

# True/1 means attn to the blocks, 0 means not attend to.
block_sparse_pattern = torch.rand((num_heads, num_blocks, num_blocks)) > 0.8

# Ensure the diag blocks are always attended.
# Otherwise, tokens at block_0 have nothing to attend to
for head_sparse_pattern in block_sparse_pattern:
    head_sparse_pattern.diagonal()[:] = True

# Ensure it is causal
block_sparse_pattern *= torch.tril(torch.ones_like(block_sparse_pattern[0]))

# NOTE: You may get warning saying that pattern is not KV cache friendly, due to
# KV cache needed for later tokens are not used in earlier tokens.
# This may result in unexpected larger KV cache.
# So you may need to consider properly design the sparse pattern carefully.

attn = SparseAttention(block_size, block_sparse_pattern, seq_dim=1)
attn.to(device)

# similar, it needs to warmup the first time it runs
output = attn(q, k, v, sm_scale=0.008)

```


# Todo

- [x] *1*. allow change seqlen in backward: using start/end instead culen,
    culen vector can be modifed to 2 column start-end using as_strided.
    Done: inside bwd_dkdv kernel, check the max num q-blocks
- [ ] *2*. allow variable seqlen in a sequence for both forward/backward.
- [ ] *3*. step ahead of 2, unified with left padding (rope with left paddings?)
    i.e., using start/end instead cuseqlen like in 1.
- [ ] *4*. implement loading saving version for grouped attn, which also better
    utilizes tensorcore at inference. This could also potentially allow
    even smaller sparse block size, i.e. 16/q_per_kv
- [ ] 5. split sequence k-length: better GPU/SM utilizaiton for single seq with head parallel
    and inference, especially for very long seq in prefilling and decoding.
- [ ] 6. CPU/V100 version using libTorch.
- [ ] 7. merge N blocks using tl.cat, to reduce cuda core ops to reduce softmax.
    This also opens the possiblity to allow block_n < 16.
    Issue: tl.cat supports only `can_reorder = True`, i.e., at reduction dimention.
- [ ] 8. potentially higher latency with some head is dense (need to check)
- [ ] 9. bwd time increases significantly when sparse_block_size is smaller (fwd is less impacted). This is probably the same issue as 8: some threadblock is just too much, need to break them down., i.e, this is essentially the same as 5.
- [ ] 10. Autotune block_m
- [ ] 11. FP8
- [ ] 12. Inference-prefilling: batched chucked prefix filling
- [ ] 13. Inference-decoding: support different kv_cache size for diff batch, right now,
        only support same past_len as used in megatron for decoding
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention

from dkernel import get_sparse_attn_mask, LocalStrideSparseAttention
import numpy as np


BLOCK_M = 32
BLOCK_N = 32
SLEN = 1024
LOCALS = 4 
VERT = 1
HEADS = 32
BATCH = 4
DIM = 128
scale = np.sqrt(1/DIM)
dtype = torch.bfloat16

flex_attention = torch.compile(flex_attention)

def local_vert_stride(b, h, q_idx, kv_idx):
    q_bid = q_idx // BLOCK_M
    kv_bid = kv_idx // BLOCK_N
    locals = ((q_bid - kv_bid) < LOCALS)
    vert = (kv_bid) % VERT == 0
    return (q_idx >= kv_idx) & (locals | vert)

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idxq

# Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
# block_mask = create_block_mask(local_vert_stride, B=None, H=HEADS, Q_LEN=SLEN, KV_LEN=SLEN, device='cpu', BLOCK_SIZE=BLOCK_N)
# block_mask = create_block_mask(local_vert_stride, B=None, H=HEADS, Q_LEN=SLEN, KV_LEN=SLEN, device='cpu')
block_mask = create_block_mask(local_vert_stride, B=None, H=None, Q_LEN=SLEN, KV_LEN=SLEN, device='cpu')

print(repr(block_mask))
print(block_mask)


# In this case, we don't need a score_mod, so we won't pass any in.
# However, score_mod can still be combined with block_mask if you need the additional flexibility.

query = torch.empty((BATCH, HEADS, SLEN, DIM), dtype=dtype, device='cuda').normal_(mean=0, std=1) * 2
key = torch.empty((BATCH, HEADS, SLEN, DIM), dtype=dtype, device='cuda').normal_(mean=0, std=1) * 2
value = torch.empty((BATCH, HEADS, SLEN, DIM), dtype=dtype, device='cuda').normal_(mean=0, std=1) * 2


flex_out = flex_attention(query, key, value, block_mask=block_mask.to(0), scale=scale)



if VERT > 1:
    _, _, dense_mask = get_sparse_attn_mask(HEADS, SLEN, BLOCK_N, LOCALS, vert_stride=VERT, homo_head=True, return_dense=True)
    dense_mask = dense_mask[None].bool().to(0)
else:
    dense_mask = None

spda_out = scaled_dot_product_attention(query, key, value, attn_mask=dense_mask, scale=scale, is_causal=True)

sparse_attention_fn = LocalStrideSparseAttention(
                                            HEADS, SLEN,
                                            BLOCK_N,
                                            LOCALS,
                                            VERT,
                                            homo_head=True,
                                            block_m=max(64, BLOCK_M),
                                            block_n=min(64, BLOCK_N),
                                            seq_dim=2
                                            )

sparse_out = sparse_attention_fn(query, key, value, sm_scale=scale)

# print(f'> {spda_out.shape=}')
print(f'> {flex_out.shape=}')
print(f'> {sparse_out.shape=}')

# print(flex_out == sparse_out)

print('## forward:')
print(f'{(flex_out - sparse_out).abs().max()=}')
print(f'{(flex_out - spda_out).abs().max()=}')  # fwd: 0.1797
print(f'{(sparse_out - spda_out).abs().max()=}')  # fwd: 0.0312

# print(f'>>> {BLOCK_M=}, {repr(block_mask)}')

# Looks flex_attn has precision issue
# forward: 0.18


# backward
query.requires_grad_()
key.requires_grad_()
value.requires_grad_()

def get_grad_and_clear():
    grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())
    query.grad = None
    key.grad = None
    value.grad = None
    return grads


dout = torch.rand_like(spda_out)

flex_out = flex_attention(query, key, value, block_mask=block_mask.to(0), scale=scale)
flex_out.backward(dout)
flex_grads = get_grad_and_clear()

sparse_out = sparse_attention_fn(query, key, value, sm_scale=scale)
sparse_out.backward(dout)
sparse_grads = get_grad_and_clear()

spda_out = scaled_dot_product_attention(query, key, value, attn_mask=dense_mask, scale=scale, is_causal=True)
spda_out.backward(dout)
spda_grads = get_grad_and_clear()


print('## backward')
print('> flex grad vs spda grad')
for i, (fg, spda_g) in enumerate(zip(flex_grads, spda_grads)):
    print(f'  > {i=}, {(fg - spda_g).abs().max()=}')

print('> sparse grad vs spda grad')
for i, (sg, spda_g) in enumerate(zip(sparse_grads, spda_grads)):
    print(f'  > {i=}, {(sg - spda_g).abs().max()=}')


'''
# DENSE
slen=1024

> flex_out.shape=torch.Size([4, 32, 1024, 128])
> sparse_out.shape=torch.Size([4, 32, 1024, 128])
## forward:
(flex_out - sparse_out).abs().max()=tensor(0.1875, device='cuda:0', dtype=torch.bfloat16)
(flex_out - spda_out).abs().max()=tensor(0.1875, device='cuda:0', dtype=torch.bfloat16)
(sparse_out - spda_out).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)
## backward
> flex grad vs spda grad
  > i=0, (fg - spda_g).abs().max()=tensor(0.2188, device='cuda:0', dtype=torch.bfloat16)
  > i=1, (fg - spda_g).abs().max()=tensor(0.2812, device='cuda:0', dtype=torch.bfloat16)
  > i=2, (fg - spda_g).abs().max()=tensor(0.0625, device='cuda:0', dtype=torch.bfloat16)
> sparse grad vs spda grad
  > i=0, (sg - spda_g).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)
  > i=1, (sg - spda_g).abs().max()=tensor(0.0625, device='cuda:0', dtype=torch.bfloat16)
  > i=2, (sg - spda_g).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)


# sparse
BLOCK_M = 128
BLOCK_N = 128
SLEN = 1024
LOCALS = 4 
VERT = 4
HEADS = 32
BATCH = 4
DIM = 128

> flex_out.shape=torch.Size([4, 32, 1024, 128])
> sparse_out.shape=torch.Size([4, 32, 1024, 128])
## forward:
(flex_out - sparse_out).abs().max()=tensor(0.1875, device='cuda:0', dtype=torch.bfloat16)
(flex_out - spda_out).abs().max()=tensor(0.1875, device='cuda:0', dtype=torch.bfloat16)
(sparse_out - spda_out).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)
## backward
> flex grad vs spda grad
  > i=0, (fg - spda_g).abs().max()=tensor(0.2812, device='cuda:0', dtype=torch.bfloat16)
  > i=1, (fg - spda_g).abs().max()=tensor(0.2500, device='cuda:0', dtype=torch.bfloat16)
  > i=2, (fg - spda_g).abs().max()=tensor(0.0625, device='cuda:0', dtype=torch.bfloat16)
> sparse grad vs spda grad
  > i=0, (sg - spda_g).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)
  > i=1, (sg - spda_g).abs().max()=tensor(0.0625, device='cuda:0', dtype=torch.bfloat16)
  > i=2, (sg - spda_g).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)


# sparse block_size=32
BLOCK_M = 32
BLOCK_N = 32
SLEN = 1024
LOCALS = 4 
VERT = 4
HEADS = 32
BATCH = 4
DIM = 128

## forward:
(flex_out - sparse_out).abs().max()=tensor(0.1953, device='cuda:0', dtype=torch.bfloat16)
(flex_out - spda_out).abs().max()=tensor(0.1953, device='cuda:0', dtype=torch.bfloat16)
(sparse_out - spda_out).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)
## backward
> flex grad vs spda grad
  > i=0, (fg - spda_g).abs().max()=tensor(0.2500, device='cuda:0', dtype=torch.bfloat16)
  > i=1, (fg - spda_g).abs().max()=tensor(0.2812, device='cuda:0', dtype=torch.bfloat16)
  > i=2, (fg - spda_g).abs().max()=tensor(0.0625, device='cuda:0', dtype=torch.bfloat16)
> sparse grad vs spda grad
  > i=0, (sg - spda_g).abs().max()=tensor(0.0625, device='cuda:0', dtype=torch.bfloat16)
  > i=1, (sg - spda_g).abs().max()=tensor(0.1250, device='cuda:0', dtype=torch.bfloat16)
  > i=2, (sg - spda_g).abs().max()=tensor(0.0312, device='cuda:0', dtype=torch.bfloat16)

'''
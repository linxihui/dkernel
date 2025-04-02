import importlib.metadata
__version__ = importlib.metadata.version(__package__ or __name__)

from dkernel.interface import SparseAttention, LocalStrideSparseAttention, combine_attn_partitions
from dkernel.utils import get_sparse_attn_mask

__all__ = [
    "SparseAttention",
    "LocalStrideSparseAttention",
    "get_sparse_attn_mask",
    "combine_attn_partitions",
    ]
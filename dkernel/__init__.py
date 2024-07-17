import importlib.metadata
__version__ = importlib.metadata.version(__package__ or __name__)

from s2attn.interface import SparseAttention, LocalStrideSparseAttention
from s2attn.utils import get_sparse_attn_mask

__all__ = [
    "SparseAttention",
    "LocalStrideSparseAttention",
    "get_sparse_attn_mask",
    ]
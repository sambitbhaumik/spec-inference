"""Key-Value cache management for efficient transformer inference with past key/values reuse."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

try:
    import cuda_ops
except ImportError:  # pragma: no cover
    cuda_ops = None

# Type alias: each layer's (key_tensor, value_tensor) pair
PastKeyValues = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def _seq_dim(tensor: torch.Tensor) -> int:
    """
    Determine the sequence length dimension in a KV tensor.
    
    Args:
        tensor: Shape is either (batch, seq_len, heads, head_dim) [4D] 
                or (batch, heads, seq_len, head_dim) [3D]
    
    Returns:
        Dimension index containing sequence length (2 for 4D, 1 for 3D)
    """
    if tensor.dim() == 4:
        return 2
    if tensor.dim() == 3:
        return 1
    raise ValueError(f"Unsupported KV tensor rank: {tensor.dim()}")


def _seq_len_from_past(past_key_values: PastKeyValues) -> int:
    """Extract sequence length from first layer's key tensor."""
    if not past_key_values:
        return 0
    
    # Handle DynamicCache from transformers
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
        
    # Handle standard tuple/list of (key, value) pairs
    try:
        key = past_key_values[0][0]
    except (TypeError, IndexError):
        # Fallback for other structures if necessary
        return 0
        
    return key.size(_seq_dim(key))


def _trim_tensor(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Slice tensor along sequence dimension to specified length."""
    seq_dim = _seq_dim(tensor)
    slices = [slice(None)] * tensor.dim()
    slices[seq_dim] = slice(0, seq_len)
    return tensor[tuple(slices)]


@dataclass(frozen=True)
class KVCache:
    """Immutable container for transformer past_key_values and sequence position tracking."""
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    seq_len: int = 0  # Current sequence length in cache

    @classmethod
    def empty(cls) -> "KVCache":
        """Create an empty cache (for initial generation step)."""
        return cls(past_key_values=None, seq_len=0)

    @classmethod
    def from_past_key_values(cls, past_key_values: PastKeyValues) -> "KVCache":
        """Construct KVCache from model output past_key_values."""
        seq_len = _seq_len_from_past(past_key_values)
        
        # Convert DynamicCache or other structures to list of tuples for internal storage
        if hasattr(past_key_values, "to_legacy_cache"):
            # DynamicCache has to_legacy_cache() which returns List[Tuple[torch.Tensor, torch.Tensor]]
            past_key_values = past_key_values.to_legacy_cache()
        elif not isinstance(past_key_values, list):
            past_key_values = list(past_key_values)
            
        return cls(past_key_values=past_key_values, seq_len=seq_len)

    def trim(self, seq_len: int) -> "KVCache":
        """
        Truncate cache to specified sequence length.
        Used when draft predictions don't match verifier (need to reset to earlier point).
        
        Args:
            seq_len: Target sequence length to trim to
            
        Returns:
            New KVCache trimmed to seq_len
        """
        if self.past_key_values is None:
            return self
        if seq_len >= self.seq_len:
            return self

        trimmed: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in self.past_key_values:
            if cuda_ops is not None:
                trimmed_key = cuda_ops.kv_trim(key, seq_len)
                trimmed_value = cuda_ops.kv_trim(value, seq_len)
            else:
                trimmed_key = _trim_tensor(key, seq_len)
                trimmed_value = _trim_tensor(value, seq_len)
            trimmed.append((trimmed_key, trimmed_value))

        return KVCache(past_key_values=trimmed, seq_len=seq_len)

    def update(self, past_key_values: PastKeyValues) -> "KVCache":
        """Update cache with new past_key_values from model forward pass."""
        return KVCache.from_past_key_values(past_key_values)

    def device(self) -> torch.device:
        """Get device of cached tensors."""
        if self.past_key_values is None:
            return torch.device("cpu")
        return self.past_key_values[0][0].device

    def kv_update(
        self,
        layer_idx: int,
        key_update: torch.Tensor,
        value_update: torch.Tensor,
        start_pos: int,
    ) -> None:
        """
        In-place update cache at a specific layer and position.
        Used by optimized CUDA kernels for efficient cache updates.
        
        Args:
            layer_idx: Which transformer layer to update
            key_update: New key tokens, shape (batch, 1, heads, head_dim)
            value_update: New value tokens, same shape as key_update
            start_pos: Position to insert at in cache
        """
        if self.past_key_values is None:
            raise ValueError("KV cache is empty.")
        if cuda_ops is None:
            # Fallback: CPU update using narrow() slicing
            key, value = self.past_key_values[layer_idx]
            key.narrow(_seq_dim(key), start_pos, key_update.size(_seq_dim(key))).copy_(key_update)
            value.narrow(_seq_dim(value), start_pos, value_update.size(_seq_dim(value))).copy_(value_update)
            return

        # CUDA optimized update
        key, value = self.past_key_values[layer_idx]
        cuda_ops.kv_update(key, key_update, start_pos)
        cuda_ops.kv_update(value, value_update, start_pos)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

try:
    import cuda_ops
except ImportError:  # pragma: no cover
    cuda_ops = None

PastKeyValues = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def _seq_dim(tensor: torch.Tensor) -> int:
    if tensor.dim() == 4:
        return 2
    if tensor.dim() == 3:
        return 1
    raise ValueError(f"Unsupported KV tensor rank: {tensor.dim()}")


def _seq_len_from_past(past_key_values: PastKeyValues) -> int:
    if not past_key_values:
        return 0
    key = past_key_values[0][0]
    return key.size(_seq_dim(key))


def _trim_tensor(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    seq_dim = _seq_dim(tensor)
    slices = [slice(None)] * tensor.dim()
    slices[seq_dim] = slice(0, seq_len)
    return tensor[tuple(slices)]


@dataclass(frozen=True)
class KVCache:
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    seq_len: int = 0

    @classmethod
    def empty(cls) -> "KVCache":
        return cls(past_key_values=None, seq_len=0)

    @classmethod
    def from_past_key_values(cls, past_key_values: PastKeyValues) -> "KVCache":
        return cls(past_key_values=list(past_key_values), seq_len=_seq_len_from_past(past_key_values))

    def trim(self, seq_len: int) -> "KVCache":
        if self.past_key_values is None:
            return self
        if seq_len >= self.seq_len:
            return self

        trimmed: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in self.past_key_values:
            trimmed_key = _trim_tensor(key, seq_len)
            trimmed_value = _trim_tensor(value, seq_len)
            trimmed.append((trimmed_key, trimmed_value))

        return KVCache(past_key_values=trimmed, seq_len=seq_len)

    def update(self, past_key_values: PastKeyValues) -> "KVCache":
        return KVCache.from_past_key_values(past_key_values)

    def device(self) -> torch.device:
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
        if self.past_key_values is None:
            raise ValueError("KV cache is empty.")
        if cuda_ops is None:
            key, value = self.past_key_values[layer_idx]
            key.narrow(_seq_dim(key), start_pos, key_update.size(_seq_dim(key))).copy_(key_update)
            value.narrow(_seq_dim(value), start_pos, value_update.size(_seq_dim(value))).copy_(value_update)
            return

        key, value = self.past_key_values[layer_idx]
        cuda_ops.kv_update(key, key_update, start_pos)
        cuda_ops.kv_update(value, value_update, start_pos)

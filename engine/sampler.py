from dataclasses import dataclass
from typing import Optional

import torch

try:
    import cuda_ops
except ImportError:  # pragma: no cover
    cuda_ops = None


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits

    if cuda_ops is not None and logits.is_cuda:
        return cuda_ops.topk_filter(logits, top_k)

    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    if cuda_ops is not None and logits.is_cuda and logits.dim() == 2:
        return cuda_ops.topp_filter(logits, top_p)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 0] = False
    mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def filter_logits(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    logits = _apply_top_k(logits, config.top_k)
    logits = _apply_top_p(logits, config.top_p)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    if config.temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if cuda_ops is not None and logits.is_cuda:
        scaled_logits = cuda_ops.temperature_scale(logits, config.temperature)
    else:
        scaled_logits = logits / config.temperature
    filtered_logits = filter_logits(scaled_logits, config)
    probs = torch.softmax(filtered_logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token.squeeze(-1)


def sample_from_logits(
    logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    if logits.dim() == 2:
        return sample_next_token(logits, config)

    if logits.dim() != 3:
        raise ValueError("Logits must be 2D or 3D.")

    batch, seq_len, vocab = logits.shape
    flat_logits = logits.reshape(batch * seq_len, vocab)
    tokens = sample_next_token(flat_logits, config)
    return tokens.view(batch, seq_len)

"""Sampling utilities: temperature scaling, top-k, and top-p filtering."""
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import cuda_ops
except ImportError:  # pragma: no cover
    cuda_ops = None


@dataclass
class SamplingConfig:
    """Hyperparameters for next-token sampling."""
    temperature: float = 1.0  # Logit scaling: >1 increases diversity, <1 concentrates on top tokens
    top_k: int = 0            # Keep only top-k highest probability tokens (0 = disabled)
    top_p: float = 1.0        # Nucleus sampling: keep tokens until cumulative prob >= top_p


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k highest values, set others to -inf.
    
    Args:
        logits: Shape (batch, vocab_size)
        top_k: Number of top tokens to keep
        
    Returns:
        Filtered logits with same shape, -inf for filtered tokens
    """
    if top_k <= 0:
        return logits

    # Try CUDA kernel if available
    if cuda_ops is not None and logits.is_cuda:
        return cuda_ops.topk_filter(logits, top_k)

    # CPU implementation
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep tokens until cumulative probability exceeds top_p.
    
    Args:
        logits: Shape (batch, vocab_size)
        top_p: Cumulative probability threshold (0.0 to 1.0)
        
    Returns:
        Filtered logits, -inf for tokens outside nucleus
    """
    if top_p >= 1.0:
        return logits

    # Try CUDA kernel if available (only for 2D logits)
    if cuda_ops is not None and logits.is_cuda and logits.dim() == 2:
        return cuda_ops.topp_filter(logits, top_p)

    # CPU implementation: sort by logits, compute cumsum of softmax probs
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mark tokens beyond cumulative threshold
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 0] = False  # Always keep top token
    # Unsort mask back to original token positions
    mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def filter_logits(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Apply both top-k and top-p filtering sequentially."""
    logits = _apply_top_k(logits, config.top_k)
    logits = _apply_top_p(logits, config.top_p)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """
    Sample next token from logits using temperature scaling and filtering.
    
    Args:
        logits: Shape (batch, vocab_size) or (vocab_size,)
        config: SamplingConfig with temperature, top_k, top_p
        
    Returns:
        Sampled token IDs, shape (batch,) or (1,) depending on input
    """
    if config.temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Ensure 2D (batch, vocab)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Apply temperature scaling
    if cuda_ops is not None and logits.is_cuda:
        scaled_logits = cuda_ops.temperature_scale(logits, config.temperature)
    else:
        scaled_logits = logits / config.temperature
    
    # Apply sampling filters
    filtered_logits = filter_logits(scaled_logits, config)
    
    # Convert to probabilities via softmax and sample
    probs = torch.softmax(filtered_logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token.squeeze(-1)


def sample_from_logits(
    logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """
    Sample tokens from 2D or 3D logits.
    
    Args:
        logits: Shape (batch, vocab_size) [2D] or (batch, seq_len, vocab_size) [3D]
        config: SamplingConfig
        
    Returns:
        Sampled token IDs matching input batch/sequence dimensions
    """
    if logits.dim() == 2:
        return sample_next_token(logits, config)

    if logits.dim() != 3:
        raise ValueError("Logits must be 2D or 3D.")

    # Flatten sequence dimension and sample, then reshape back
    batch, seq_len, vocab = logits.shape
    flat_logits = logits.reshape(batch * seq_len, vocab)
    tokens = sample_next_token(flat_logits, config)
    return tokens.view(batch, seq_len)

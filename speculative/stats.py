"""Tracking statistics for speculative decoding: acceptance rates, speedup, token throughput."""
from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class SpeculativeStats:
    """Statistics tracker for speculative decoding performance monitoring."""
    total_proposed: int = 0  # Sum of all draft tokens proposed
    total_accepted: int = 0  # Sum of all draft tokens accepted by verifier
    total_emitted: int = 0   # Total tokens output (including mismatches)
    steps: int = 0            # Number of decode steps (verification rounds)
    last_acceptance: int = 0  # Tokens accepted in last step
    last_proposed: int = 0    # Tokens proposed in last step
    last_emitted: int = 0     # Tokens emitted in last step
    memory_mb: Dict[str, float] = field(default_factory=dict)  # GPU memory usage

    def update(self, proposed: int, accepted: int, emitted: int) -> None:
        """Update stats after one decode step.
        
        Args:
            proposed: Number of draft tokens proposed in this step (typically draft_k)
            accepted: Number of draft tokens accepted by verifier (0 to proposed)
            emitted: Total tokens emitted (accepted + replacement token on mismatch)
        """
        self.total_proposed += proposed
        self.total_accepted += accepted
        self.total_emitted += emitted
        self.steps += 1
        self.last_proposed = proposed
        self.last_acceptance = accepted
        self.last_emitted = emitted

    def update_memory(self) -> None:
        """Capture current GPU memory usage in MB."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            self.memory_mb = {"allocated": allocated, "reserved": reserved}

    @property
    def acceptance_rate(self) -> float:
        """Ratio of accepted tokens to proposed tokens (0.0 to 1.0)."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def avg_tokens_per_step(self) -> float:
        """Average tokens emitted per verification step."""
        if self.steps == 0:
            return 0.0
        return self.total_emitted / self.steps

    @property
    def speedup(self) -> float:
        """Effective speedup: tokens emitted per step (1.0 = same as auto-regressive)."""
        if self.steps == 0:
            return 0.0
        return self.total_emitted / self.steps

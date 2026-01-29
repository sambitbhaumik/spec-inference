from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class SpeculativeStats:
    total_proposed: int = 0
    total_accepted: int = 0
    total_emitted: int = 0
    steps: int = 0
    last_acceptance: int = 0
    last_proposed: int = 0
    memory_mb: Dict[str, float] = field(default_factory=dict)

    def update(self, proposed: int, accepted: int, emitted: int) -> None:
        self.total_proposed += proposed
        self.total_accepted += accepted
        self.total_emitted += emitted
        self.steps += 1
        self.last_proposed = proposed
        self.last_acceptance = accepted

    def update_memory(self) -> None:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            self.memory_mb = {"allocated": allocated, "reserved": reserved}

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def avg_tokens_per_step(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.total_emitted / self.steps

    @property
    def speedup(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.total_emitted / self.steps

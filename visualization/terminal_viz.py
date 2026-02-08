"""Live terminal visualization of speculative decoding generation and performance metrics."""
from __future__ import annotations

from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from speculative.stats import SpeculativeStats


class TerminalVisualizer:
    """Real-time terminal UI for generation progress with live stats updates."""
    
    def __init__(self, refresh_per_second: int = 8, max_text_chars: int = 1200) -> None:
        """
        Initialize terminal visualizer.
        
        Args:
            refresh_per_second: UI update frequency (8 = 125ms per frame)
            max_text_chars: Max chars to display in output buffer (older text truncated)
        """
        self.console = Console()
        self.refresh_per_second = refresh_per_second
        self.max_text_chars = max_text_chars
        self.live: Optional[Live] = None
        self.text_buffer = ""

    def __enter__(self) -> "TerminalVisualizer":
        """Context manager entry: start live rendering."""
        self.live = Live(
            self._render(SpeculativeStats(), ""),
            console=self.console,
            refresh_per_second=self.refresh_per_second
        )
        self.live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit: stop live rendering."""
        if self.live is not None:
            self.live.stop()
            self.live = None

    def update(self, stats: SpeculativeStats, new_text: str) -> None:
        """
        Update display with new stats and generated text.
        
        Args:
            stats: Current SpeculativeStats with acceptance rates, token counts
            new_text: Newly generated text tokens decoded to string
        """
        if new_text:
            self.text_buffer += new_text
            # Keep only recent output to avoid terminal lag
            if len(self.text_buffer) > self.max_text_chars:
                self.text_buffer = self.text_buffer[-self.max_text_chars :]

        if self.live is not None:
            self.live.update(self._render(stats, self.text_buffer))

    def _render(self, stats: SpeculativeStats, text: str):
        """Build the complete UI: acceptance bar, stats table, generated text."""
        stats.update_memory()
        bar = self._acceptance_bar(stats.acceptance_rate)

        # Build metrics table
        table = Table(box=box.ASCII, show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Steps", str(stats.steps))
        table.add_row("AcceptanceRate", f"{stats.acceptance_rate:.2f}")
        table.add_row("AvgTokensStep", f"{stats.avg_tokens_per_step:.2f}")
        table.add_row("Speedup", f"{stats.speedup:.2f}x")
        table.add_row("LastProposed", str(stats.last_proposed))
        table.add_row("LastAccepted", str(stats.last_acceptance))
        if stats.memory_mb:
            table.add_row("MemAllocatedMB", f"{stats.memory_mb['allocated']:.1f}")
            table.add_row("MemReservedMB", f"{stats.memory_mb['reserved']:.1f}")

        text_block = Text(text)
        panel = Panel(
            table,
            title="Speculative Decoding Stats",
            box=box.ASCII,
        )

        output_panel = Panel(
            text_block,
            title="Generated Text",
            box=box.ASCII,
        )

        group = Group(Text(bar), panel, output_panel)
        return Panel(group, box=box.ASCII)

    @staticmethod
    def _acceptance_bar(rate: float, width: int = 24) -> str:
        """
        Draw ASCII acceptance rate bar: [####-----] at given fill rate.
        
        Args:
            rate: Acceptance rate (0.0 to 1.0)
            width: Bar width in characters
            
        Returns:
            ASCII bar visualization string
        """
        rate = max(0.0, min(rate, 1.0))
        filled = int(rate * width)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

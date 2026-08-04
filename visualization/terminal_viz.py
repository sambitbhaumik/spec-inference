"""Live terminal visualization for speculative decoding."""

from __future__ import annotations

from typing import Optional

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from speculative.stats import SpeculativeStats


class TerminalVisualizer:
    """A compact view of the draft, verify, and emit loop."""

    def __init__(self, refresh_per_second: int = 8, max_text_chars: int = 1200) -> None:
        """Configure refresh frequency and the visible output history."""
        self.console = Console()
        self.refresh_per_second = refresh_per_second
        self.max_text_chars = max_text_chars
        self.live: Optional[Live] = None
        self.text_buffer = ""
        self._text_was_truncated = False

    def __enter__(self) -> "TerminalVisualizer":
        """Start live rendering with a clean output buffer."""
        self.text_buffer = ""
        self._text_was_truncated = False
        self.live = Live(
            self._render(SpeculativeStats(), ""),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
        )
        self.live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop live rendering."""
        if self.live is not None:
            self.live.stop()
            self.live = None

    def update(self, stats: SpeculativeStats, new_text: str) -> None:
        """Add newly decoded text and refresh the current engine state."""
        if new_text:
            self.text_buffer += new_text
            if len(self.text_buffer) > self.max_text_chars:
                self.text_buffer = self.text_buffer[-self.max_text_chars :]
                self._text_was_truncated = True

        if self.live is not None:
            self.live.update(self._render(stats, self.text_buffer))

    def _render(self, stats: SpeculativeStats, text: str) -> RenderableType:
        """Build the complete terminal view."""
        stats.update_memory()

        header = Text()
        header.append("SPECULATIVE DECODING", style="bold cyan")
        header.append("  draft → verify → emit", style="dim")

        round_panel = Panel(
            self._round_view(stats),
            title="[bold]CURRENT ROUND[/bold]",
            title_align="left",
            border_style="bright_black",
            box=box.ROUNDED,
            padding=(0, 1),
        )

        output = Text()
        if self._text_was_truncated:
            output.append("… ", style="dim")
        if text:
            output.append(text)
        else:
            output.append("Waiting for generated text…", style="dim italic")

        output_title = Text("OUTPUT", style="bold")
        output_title.append(f"  {stats.total_emitted} tokens", style="dim")
        output_panel = Panel(
            output,
            title=output_title,
            title_align="left",
            border_style="bright_black",
            box=box.ROUNDED,
            padding=(0, 1),
        )

        return Group(header, round_panel, self._metrics_view(stats), output_panel)

    @classmethod
    def _round_view(cls, stats: SpeculativeStats) -> RenderableType:
        """Show what happened to the most recent speculative window."""
        if stats.steps == 0:
            return Text("Waiting for the first draft…", style="dim italic")

        flow = Text()
        flow.append("DRAFT  ", style="bold")
        flow.append_text(cls._token_strip(stats.last_proposed, stats.last_acceptance))
        flow.append("   →   ", style="dim")
        flow.append("VERIFY  ", style="bold")
        flow.append(f"{stats.last_acceptance}/{stats.last_proposed}", style="bold cyan")
        flow.append(" accepted", style="dim")
        flow.append("   →   ", style="dim")
        flow.append("EMIT  ", style="bold")
        flow.append(str(stats.last_emitted), style="bold green")

        table = Table.grid(expand=True, padding=0)
        table.add_column()
        table.add_column(justify="right", no_wrap=True)
        table.add_row(flow, Text(f"round {stats.steps}", style="dim"))
        table.add_row(
            cls._acceptance_bar(stats.acceptance_rate),
            Text(f"{stats.acceptance_rate:.0%} overall", style="dim"),
        )
        return table

    @staticmethod
    def _token_strip(proposed: int, accepted: int) -> Text:
        """Render accepted, rejected, and unused draft positions."""
        strip = Text()
        for index in range(proposed):
            if index < accepted:
                strip.append("●", style="green")
            elif index == accepted:
                strip.append("×", style="red")
            else:
                strip.append("·", style="bright_black")
            if index < proposed - 1:
                strip.append(" ")
        return strip

    @staticmethod
    def _metrics_view(stats: SpeculativeStats) -> Table:
        """Render the few cumulative measures that explain engine efficiency."""
        metrics = [
            ("ACCEPTANCE", f"{stats.acceptance_rate:.0%}"),
            ("YIELD", f"{stats.avg_tokens_per_step:.2f} tok/round"),
            ("EMITTED", f"{stats.total_emitted} tokens"),
        ]
        if stats.memory_mb:
            metrics.append(("GPU", f"{stats.memory_mb['allocated']:.0f} MiB"))

        table = Table.grid(expand=True, padding=(0, 2))
        for _ in metrics:
            table.add_column(ratio=1)

        cells = []
        for label, value in metrics:
            cell = Text()
            cell.append(f"{label}\n", style="dim")
            cell.append(value, style="bold")
            cells.append(cell)
        table.add_row(*cells)
        return table

    @staticmethod
    def _acceptance_bar(rate: float, width: int = 24) -> Text:
        """Draw a restrained cumulative acceptance bar."""
        rate = max(0.0, min(rate, 1.0))
        filled = round(rate * width)
        bar = Text()
        bar.append("━" * filled, style="cyan")
        bar.append("━" * (width - filled), style="bright_black")
        return bar

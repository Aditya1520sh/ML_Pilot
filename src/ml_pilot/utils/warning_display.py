"""Rich helpers for rendering accumulated pipeline warnings.

Purpose:
    Present non-fatal issues collected on ``PipelineContext.warnings``.

Interactions:
    - Called by ``PipelineRunner`` after stages complete.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel


def render_warnings(console: Console, warnings: list[str]) -> None:
    """Display warnings in a Rich panel when any exist.

    Args:
        console: Rich console.
        warnings: Warning messages.
    """
    if not warnings:
        return
    body = "\n".join(f"- {message}" for message in warnings)
    console.print(Panel(body, title="Warnings", border_style="yellow"))

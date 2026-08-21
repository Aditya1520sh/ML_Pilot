"""Pipeline orchestrator: run stages with timing, logging, and error handling.

Purpose:
    Execute registered stages against a shared ``PipelineContext``, collect
    timings, surface Rich progress, and fail fast with clear errors.

Interactions:
    - Uses ``StageRegistry`` / ``build_default_registry``.
    - Instantiated by the Typer CLI and public Python API.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_pilot.config.resolver import load_config
from ml_pilot.config.schema import MLPilotConfig
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.defaults import (
    COMPARE_ONLY_STAGES,
    FULL_RUN_STAGES,
    build_default_registry,
)
from ml_pilot.core.registry import StageRegistry
from ml_pilot.utils.warning_display import render_warnings


class PipelineRunner:
    """Execute an ordered sequence of MLPilot stages."""

    def __init__(
        self,
        config: MLPilotConfig | None = None,
        registry: StageRegistry | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialise the runner.

        Args:
            config: Resolved configuration. Defaults are used when omitted.
            registry: Optional custom stage registry.
            console: Rich console for terminal output.
        """
        self.config = config or load_config()
        self.registry = registry or build_default_registry()
        self.console = console or Console(emoji=False, legacy_windows=False)

    def _resolve_stage_names(self) -> list[str]:
        """Pick which stages to run based on runtime flags."""
        if self.config.runtime.compare_only:
            return list(COMPARE_ONLY_STAGES)
        names = list(FULL_RUN_STAGES)
        if self.config.runtime.no_shap:
            names = [n for n in names if n not in {"select", "explain"}]
        if not self.config.tune.enable:
            names = [n for n in names if n != "tune"]
        if not self.config.select.enable or self.config.runtime.no_shap:
            names = [n for n in names if n != "select"]
        if not self.config.explain.enable or self.config.runtime.no_shap:
            names = [n for n in names if n != "explain"]
        return names

    def run(
        self,
        data_path: str | Path | None = None,
        target: str | None = None,
        stage_filter: Sequence[str] | None = None,
    ) -> PipelineContext:
        """Run the pipeline end-to-end.

        Args:
            data_path: Optional path override for the dataset.
            target: Optional target column override.
            stage_filter: Explicit stage name list; overrides runtime selection.

        Returns:
            Populated ``PipelineContext``.

        Raises:
            RuntimeError: When a stage fails.
        """
        overrides: dict[str, Any] = {}
        if data_path is not None:
            overrides.setdefault("data", {})["path"] = str(data_path)
        if target is not None:
            overrides.setdefault("data", {})["target"] = target
        if overrides:
            from ml_pilot.config.resolver import merge_overrides

            self.config = merge_overrides(self.config, overrides)

        context = PipelineContext(config=self.config)
        names = list(stage_filter) if stage_filter is not None else self._resolve_stage_names()
        stages = self.registry.ordered(only=names)

        self.console.print(
            Panel.fit(
                "[bold cyan]MLPilot[/] - Data In -> Insights Out",
                border_style="cyan",
            )
        )

        for stage in stages:
            if stage.should_skip(context):
                self.console.print(f"[yellow]Skipping[/] {stage.name}")
                continue

            self.console.print(f"[bold green]>[/] Running stage: [cyan]{stage.name}[/]")
            started = time.perf_counter()
            try:
                context = stage.run(context)
            except Exception as exc:  # noqa: BLE001 — surface stage failures cleanly
                elapsed = time.perf_counter() - started
                context.stage_timings[stage.name] = elapsed
                self.console.print(
                    f"[bold red]X Stage '{stage.name}' failed after {elapsed:.2f}s[/]"
                )
                self.console.print(f"[red]{exc}[/]")
                if self.config.runtime.verbose:
                    self.console.print(traceback.format_exc())
                raise RuntimeError(f"Stage '{stage.name}' failed: {exc}") from exc

            elapsed = time.perf_counter() - started
            context.stage_timings[stage.name] = elapsed
            self.console.print(f"[dim]  OK {stage.name} completed in {elapsed:.2f}s[/]")

        self._print_summary(context)
        render_warnings(self.console, context.warnings)
        return context

    def _print_summary(self, context: PipelineContext) -> None:
        """Render timing and metric summary tables."""
        timing = Table(title="Stage Timings", show_header=True, header_style="bold magenta")
        timing.add_column("Stage")
        timing.add_column("Seconds", justify="right")
        for name, seconds in context.stage_timings.items():
            timing.add_row(name, f"{seconds:.2f}")
        self.console.print(timing)

        if context.metrics:
            metrics = Table(title="Evaluation Metrics", show_header=True, header_style="bold blue")
            metrics.add_column("Metric")
            metrics.add_column("Value", justify="right")
            for key, value in context.metrics.items():
                if isinstance(value, float):
                    metrics.add_row(str(key), f"{value:.4f}")
                else:
                    metrics.add_row(str(key), str(value))
            self.console.print(metrics)

        if context.best_model_name:
            self.console.print(
                f"[bold]Best model:[/] [green]{context.best_model_name}[/]"
            )
        if context.artifacts:
            self.console.print("[bold]Artifacts:[/]")
            for key, path in context.artifacts.items():
                self.console.print(f"  - {key}: {path}")

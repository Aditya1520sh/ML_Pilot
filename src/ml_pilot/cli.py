"""Typer CLI entrypoint for MLPilot.

Purpose:
    Expose ``mlpilot run`` (and helpers) with Rich-powered terminal UX.

Interactions:
    - Builds ``MLPilotConfig`` via ``config.resolver``.
    - Delegates execution to ``PipelineRunner``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from rich.panel import Panel

import typer
from rich.console import Console

from ml_pilot import __version__
from ml_pilot.config.resolver import load_config
from ml_pilot.config.schema import TaskType
from ml_pilot.core.pipeline import PipelineRunner

app = typer.Typer(
    name="mlpilot",
    help="MLPilot - Data In -> Insights Out. Production AutoML for tabular data.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console(emoji=False, legacy_windows=False)


@app.callback()
def main() -> None:
    """MLPilot command group."""
    return None


@app.command("version")
def version() -> None:
    """Print the installed MLPilot version."""
    console.print(f"ml-pilot {__version__}")


@app.command("run")
def run(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV / Parquet / Excel dataset.",
    ),
    target: Optional[str] = typer.Option(
        None,
        "--target",
        "-t",
        help="Target column name. Auto-detected when omitted.",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Task type: auto | regression | classification.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory for artifacts (models, plots, metadata).",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Optional YAML configuration file.",
    ),
    no_shap: bool = typer.Option(
        False,
        "--no-shap",
        help="Disable SHAP-based select/explain stages.",
    ),
    compare_only: bool = typer.Option(
        False,
        "--compare-only",
        help="Stop after model comparison (plus save of leaderboard/bundle).",
    ),
    n_trials: Optional[int] = typer.Option(
        None,
        "--n-trials",
        help="Override Optuna trial count.",
    ),
) -> None:
    """Run the full MLPilot AutoML pipeline on a tabular dataset."""
    overrides: dict = {
        "data": {"path": str(data)},
        "runtime": {
            "no_shap": no_shap,
            "compare_only": compare_only,
        },
    }
    if target:
        overrides["data"]["target"] = target
    if task:
        try:
            TaskType(task.lower())
        except ValueError as exc:
            raise typer.BadParameter(
                "task must be one of: auto, regression, classification"
            ) from exc
        overrides["data"]["task"] = task.lower()
    if output:
        overrides["save"] = {"output_dir": str(output)}
    if n_trials is not None:
        overrides["tune"] = {"n_trials": n_trials}

    cfg = load_config(config_path=config, overrides=overrides)
    runner = PipelineRunner(config=cfg, console=console)
    try:
        ctx = runner.run()
    except Exception as exc:
        console.print(
            Panel.fit(
                f"[red]{exc}[/red]",
                title="Validation Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold green]Done.[/] Best model: [cyan]{ctx.best_model_name}[/] "
        f"| artifacts -> {cfg.save.output_dir}"
    )


if __name__ == "__main__":
    app()

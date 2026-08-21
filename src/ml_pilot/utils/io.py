"""Filesystem helpers for tabular data and JSON metadata.

Purpose:
    Centralise reading/writing so stages stay free of path boilerplate.

Interactions:
    - Used by Load, Save, Deploy, and config resolver consumers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".tsv"}


def read_tabular(path: str | Path) -> pd.DataFrame:
    """Load a tabular file with format auto-detection from the suffix.

    Args:
        path: Path to CSV, TSV, Parquet, or Excel file.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the suffix is unsupported.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(f"Unhandled suffix: {suffix}")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON document with UTF-8 encoding and indentation.

    Args:
        path: Destination file path.
        payload: JSON-serialisable mapping.

    Returns:
        Resolved path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return destination


def ensure_directory(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

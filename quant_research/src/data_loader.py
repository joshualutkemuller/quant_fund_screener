"""Data loading utilities for the quant research platform."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl


@dataclass
class DataLoader:
    """Load pricing and fundamental data using Polars."""

    data_dir: Path

    def _resolve_path(self, file_name: str) -> Path:
        """Resolve a file path within the configured data directory."""
        path = (self.data_dir / file_name).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path

    def load_prices(self, file_name: str, columns: Optional[Iterable[str]] = None) -> pl.DataFrame:
        """Load price data from a CSV file into a Polars DataFrame."""
        path = self._resolve_path(file_name)
        df = pl.read_csv(path, try_parse_dates=True)
        if columns:
            df = df.select(list(columns))
        return df

    def load_fundamentals(self, file_name: str) -> pl.DataFrame:
        """Load fundamental ratios from CSV."""
        path = self._resolve_path(file_name)
        return pl.read_csv(path, try_parse_dates=True)

    def load_benchmark(self, file_name: str) -> pl.DataFrame:
        """Load benchmark returns for risk calculations."""
        return self.load_prices(file_name)

    def clean_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardise column names."""
        return df.rename({col: col.strip().lower().replace(" ", "_") for col in df.columns})

    def merge_data(
        self,
        price_df: pl.DataFrame,
        fundamental_df: pl.DataFrame,
        on: str = "date",
    ) -> pl.DataFrame:
        """Merge price and fundamental data."""
        price_df = self.clean_column_names(price_df)
        fundamental_df = self.clean_column_names(fundamental_df)
        return price_df.join(fundamental_df, on=on, how="inner")

"""Unified scoring engine for combining technical, fundamental, and risk metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import polars as pl
import yaml


@dataclass
class ScoreConfig:
    """Configuration for the scoring engine."""

    weights: Dict[str, float]

    @classmethod
    def from_yaml(cls, path: Path) -> "ScoreConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(weights=data.get("weights", {}))


@dataclass
class ScoringEngine:
    """Combine z-scored metrics into a composite quant score."""

    config: ScoreConfig
    id_column: str = "fund_id"
    date_column: str = "date"

    def _validate_columns(self, df: pl.DataFrame) -> None:
        missing = [col for col in self.config.weights if col not in df.columns]
        if missing:
            raise KeyError(f"Missing metrics for scoring: {missing}")

    def compute_zscores(self, df: pl.DataFrame, metric_columns: Iterable[str]) -> pl.DataFrame:
        df = df.sort(self.date_column)
        for column in metric_columns:
            df = df.with_columns(
                (
                    (pl.col(column) - pl.col(column).mean().over(self.date_column))
                    / pl.col(column).std(ddof=1).over(self.date_column)
                ).alias(f"{column}_z")
            )
        return df

    def compute_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        self._validate_columns(df)
        z_df = self.compute_zscores(df, self.config.weights.keys())
        score_expr = sum(
            pl.col(f"{metric}_z") * weight for metric, weight in self.config.weights.items()
        )
        z_df = z_df.with_columns(score_expr.alias("quant_score"))
        z_df = z_df.with_columns(
            pl.col("quant_score").rank("ordinal").over(self.date_column).alias("quant_rank")
        )
        return z_df

    def top_bottom_deciles(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            (pl.col("quant_rank") / pl.col("quant_rank").max().over(self.date_column)).alias("quant_percentile")
        )
        return df.with_columns(
            pl.when(pl.col("quant_percentile") <= 0.1)
            .then("top_decile")
            .when(pl.col("quant_percentile") >= 0.9)
            .then("bottom_decile")
            .otherwise("middle")
            .alias("quant_bucket")
        )

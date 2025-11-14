"""Feature engineering utilities integrating technical and fundamental data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import polars as pl


@dataclass
class FundFeatureEngineer:
    """Compute technical indicators and merge with fundamental ratios."""

    id_column: str = "fund_id"
    date_column: str = "date"

    def compute_technical_indicators(
        self,
        price_df: pl.DataFrame,
        price_column: str = "price",
        return_column: str = "returns",
        windows: Sequence[int] = (21, 63, 126),
    ) -> pl.DataFrame:
        """Add returns, volatility, momentum, and drawdown features."""
        df = price_df.sort([self.id_column, self.date_column])
        df = df.with_columns(
            pl.col(price_column)
            .pct_change()
            .over(self.id_column)
            .alias(return_column)
        )
        df = df.with_columns(
            pl.col(return_column)
            .cumsum()
            .over(self.id_column)
            .alias("cumulative_return")
        )
        df = df.with_columns(
            (pl.col(price_column) / pl.col(price_column).cum_max().over(self.id_column) - 1).alias("drawdown")
        )
        for window in windows:
            df = df.with_columns(
                pl.col(return_column)
                .rolling_std(window)
                .over(self.id_column)
                .alias(f"volatility_{window}")
            )
            df = df.with_columns(
                pl.col(return_column)
                .rolling_sum(window)
                .over(self.id_column)
                .alias(f"momentum_{window}")
            )
            df = df.with_columns(
                pl.col(return_column)
                .rolling_mean(window)
                .over(self.id_column)
                .alias(f"avg_return_{window}")
            )
        return df

    def merge_with_fundamentals(
        self,
        technical_df: pl.DataFrame,
        fundamental_df: pl.DataFrame,
        fundamental_columns: Iterable[str],
    ) -> pl.DataFrame:
        """Merge engineered technical indicators with fundamental ratios."""
        cols = [self.id_column, self.date_column, *fundamental_columns]
        fundamentals = fundamental_df.select(cols)
        return technical_df.join(fundamentals, on=[self.id_column, self.date_column], how="left")

    def compute_rolling_correlations(
        self,
        df: pl.DataFrame,
        target_column: str,
        benchmark_column: str,
        window: int = 63,
    ) -> pl.DataFrame:
        """Compute rolling correlations with a benchmark."""
        df = df.sort([self.id_column, self.date_column])
        correlations = (
            df.lazy()
            .groupby(self.id_column)
            .agg(
                pl.pearson_corr(target_column, benchmark_column)
                .rolling_mean(window)
                .alias(f"rolling_corr_{window}")
            )
            .explode(f"rolling_corr_{window}")
            .collect()
        )
        return df.join(correlations, on=[self.id_column])

    def compute_valuation_zscores(
        self,
        df: pl.DataFrame,
        valuation_columns: Iterable[str],
    ) -> pl.DataFrame:
        """Compute cross-sectional z-scores for valuation metrics."""
        df = df.sort(self.date_column)
        for column in valuation_columns:
            df = df.with_columns(
                (
                    (pl.col(column) - pl.col(column).mean().over(self.date_column))
                    / pl.col(column).std(ddof=1).over(self.date_column)
                ).alias(f"{column}_zscore")
            )
        return df

    def compute_relative_ranks(self, df: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
        """Rank funds cross-sectionally for each date."""
        for column in columns:
            df = df.with_columns(
                pl.col(column)
                .rank("ordinal")
                .over(self.date_column)
                .alias(f"{column}_rank")
            )
        return df

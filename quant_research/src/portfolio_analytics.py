"""Portfolio analytics extension for combining fund scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import polars as pl


@dataclass
class PortfolioAnalytics:
    """Simulate portfolios of ranked funds and compute performance statistics."""

    id_column: str = "fund_id"
    date_column: str = "date"

    def _equal_weight(self, df: pl.DataFrame) -> pl.DataFrame:
        weights = (
            df.group_by([self.date_column])
            .agg(pl.len().alias("count"))
            .with_columns((1 / pl.col("count")).alias("weight"))
        )
        return df.join(weights.select(self.date_column, "weight"), on=self.date_column)

    def _compute_portfolio_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col("weight") * pl.col("returns")).alias("weighted_return"))

    def simulate_equal_weighted(self, df: pl.DataFrame) -> pl.DataFrame:
        weighted = self._equal_weight(df)
        weighted = self._compute_portfolio_returns(weighted)
        portfolio = (
            weighted.group_by(self.date_column)
            .agg(pl.col("weighted_return").sum().alias("portfolio_return"))
            .sort(self.date_column)
        )
        return portfolio

    def simulate_optimized(self, df: pl.DataFrame, risk_aversion: float = 10.0) -> pl.DataFrame:
        pandas_df = df.select([self.date_column, self.id_column, "returns"]).to_pandas()
        pivot = pandas_df.pivot(index=self.date_column, columns=self.id_column, values="returns").dropna()
        mean_returns = pivot.mean().values
        cov = pivot.cov().values
        inv = np.linalg.pinv(cov * risk_aversion)
        weights = inv @ mean_returns
        weights /= weights.sum()
        portfolio_returns = pivot.values @ weights
        result = pl.DataFrame({
            self.date_column: pivot.index,
            "portfolio_return": portfolio_returns,
        })
        return result

    def rolling_performance(self, returns_df: pl.DataFrame, window: int = 63) -> pl.DataFrame:
        df = returns_df.sort(self.date_column)
        df = df.with_columns(
            pl.col("portfolio_return").rolling_sum(window).alias("rolling_return"),
            pl.col("portfolio_return").rolling_std(window).alias("rolling_volatility"),
        )
        df = df.with_columns(
            (pl.col("rolling_return") - pl.col("rolling_volatility") * 1.65).alias("cvar_95")
        )
        return df

    def stress_test(self, df: pl.DataFrame, shock: float = -0.1) -> pl.DataFrame:
        return df.with_columns((pl.col("portfolio_return") + shock).alias("stress_scenario"))

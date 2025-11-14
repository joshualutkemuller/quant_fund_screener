"""Risk analytics utilities for quant research."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import polars as pl
from matplotlib import pyplot as plt


@dataclass
class RiskAnalytics:
    """Compute traditional and advanced risk metrics."""

    id_column: str = "fund_id"
    date_column: str = "date"

    def _to_pandas(self, df: pl.DataFrame):
        return df.to_pandas()

    def _returns(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column).pct_change().over(self.id_column).alias("returns")
        )

    def _long_format(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.melt(id_vars=[self.id_column, self.date_column], variable_name="metric", value_name="value")

    def compute_risk_metrics(
        self,
        df: pl.DataFrame,
        benchmark_returns: Optional[pl.Series] = None,
        risk_free_rate: float = 0.0,
    ) -> pl.DataFrame:
        """Compute volatility, skew, kurtosis, beta, tracking error, Sharpe, Sortino, and drawdown."""
        df = df.sort([self.id_column, self.date_column])
        returns = df.select(self.id_column, self.date_column, "returns")
        if "returns" not in df.columns:
            raise ValueError("Input DataFrame must include a 'returns' column.")
        group = returns.group_by(self.id_column)
        summary = group.agg(
            [
                pl.col(self.date_column).max().alias(self.date_column),
                pl.col("returns").std().alias("volatility"),
                pl.col("returns").skew().alias("skew"),
                pl.col("returns").kurtosis().alias("kurtosis"),
                pl.col("returns").mean().alias("avg_return"),
                pl.col("returns").count().alias("n_obs"),
                pl.col("returns").min().alias("min_return"),
            ]
        )
        sharpe = (
            summary
            .with_columns(
                ((pl.col("avg_return") - risk_free_rate) / pl.col("volatility")).alias("sharpe_ratio")
            )
        )
        downside = group.agg(
            pl.col("returns")
            .map_elements(lambda x: min(x - risk_free_rate, 0.0))
            .pow(2)
            .mean()
            .sqrt()
            .alias("downside_deviation")
        )
        sortino = sharpe.join(downside, on=self.id_column)
        sortino = sortino.with_columns(
            ((pl.col("avg_return") - risk_free_rate) / pl.col("downside_deviation")).alias("sortino_ratio")
        )
        result = sharpe.join(sortino.select(self.id_column, "downside_deviation", "sortino_ratio"), on=self.id_column)
        df = df.with_columns(
            pl.col("returns")
            .cum_sum()
            .over(self.id_column)
            .alias("cumulative_return")
        )
        df = df.with_columns(
            (pl.col("cumulative_return") - pl.col("cumulative_return").cum_max().over(self.id_column)).alias("drawdown")
        )
        max_dd = df.group_by(self.id_column).agg(pl.col("drawdown").min().alias("max_drawdown"))
        result = result.join(max_dd, on=self.id_column)
        if benchmark_returns is not None:
            beta = df.with_columns(
                pl.cov("returns", benchmark_returns).alias("covariance")
            )
            var_bench = benchmark_returns.var()
            beta_val = beta.group_by(self.id_column).agg(
                (pl.col("covariance") / var_bench).alias("beta")
            )
            tracking_error = df.with_columns(
                (pl.col("returns") - benchmark_returns).pow(2).alias("squared_diff")
            )
            tracking_error = tracking_error.group_by(self.id_column).agg(
                pl.col("squared_diff").mean().sqrt().alias("tracking_error")
            )
            result = result.join(beta_val, on=self.id_column).join(tracking_error, on=self.id_column)
        return result

    def compute_rolling_metrics(self, df: pl.DataFrame, windows: Sequence[int]) -> pl.DataFrame:
        """Calculate rolling risk metrics for different windows."""
        results = []
        for window in windows:
            rolling = (
                df.sort([self.id_column, self.date_column])
                .groupby_dynamic(self.date_column, every=1, period=window, by=self.id_column)
                .agg(
                    [
                        pl.col("returns").std().alias("volatility"),
                        pl.col("returns").mean().alias("avg_return"),
                        pl.col("returns").skew().alias("skew"),
                        pl.col("returns").kurtosis().alias("kurtosis"),
                    ]
                )
                .with_columns(pl.lit(window).alias("window"))
            )
            results.append(rolling)
        return pl.concat(results)

    def rolling_sharpe(self, df: pl.DataFrame, window: int, risk_free_rate: float = 0.0) -> pl.DataFrame:
        df = df.sort([self.id_column, self.date_column])
        return df.with_columns(
            (
                (pl.col("returns").rolling_mean(window) - risk_free_rate)
                / pl.col("returns").rolling_std(window)
            )
            .over(self.id_column)
            .alias("rolling_sharpe")
        )

    def tidy_long(self, df: pl.DataFrame, metric_columns: Iterable[str]) -> pl.DataFrame:
        return df.melt(id_vars=[self.id_column], value_vars=list(metric_columns), variable_name="metric", value_name="value")

    def risk_heatmap(self, df: pl.DataFrame, output_path: Optional[Path] = None) -> None:
        pivot = df.pivot(index=self.id_column, columns="metric", values="value")
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(pivot.to_pandas().values, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot[self.id_column])))
        ax.set_yticklabels(pivot[self.id_column])
        fig.colorbar(cax, ax=ax, label="Value")
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        plt.close(fig)

    def plot_rolling_sharpe(self, df: pl.DataFrame, output_path: Optional[Path] = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for fund_id, group in df.group_by(self.id_column):
            ax.plot(group[self.date_column], group["rolling_sharpe"], label=str(fund_id))
        ax.set_title("Rolling Sharpe Ratio")
        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe")
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        plt.close(fig)

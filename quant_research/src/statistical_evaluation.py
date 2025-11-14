"""Statistical evaluation tools for fund time series."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
from matplotlib import pyplot as plt
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class StatisticalEvaluator:
    """Run diagnostic statistical tests on fund series."""

    id_column: str = "fund_id"
    date_column: str = "date"

    def stationarity_tests(self, df: pl.DataFrame, value_column: str) -> pl.DataFrame:
        outputs = []
        for fund_id, group in df.group_by(self.id_column):
            series = group.sort(self.date_column)[value_column].to_numpy()
            result = adfuller(series, autolag="AIC")
            outputs.append({
                self.id_column: fund_id,
                "adf_stat": float(result[0]),
                "adf_pvalue": float(result[1]),
            })
        return pl.DataFrame(outputs)

    def autocorrelation_test(self, df: pl.DataFrame, residual_column: str) -> pl.DataFrame:
        metrics = []
        for fund_id, group in df.group_by(self.id_column):
            series = group[residual_column].to_numpy()
            metrics.append({
                self.id_column: fund_id,
                "durbin_watson": float(durbin_watson(series)),
            })
        return pl.DataFrame(metrics)

    def normality_test(self, df: pl.DataFrame, value_column: str) -> pl.DataFrame:
        metrics = []
        for fund_id, group in df.group_by(self.id_column):
            series = group[value_column].to_numpy()
            stat, pvalue, _, _ = jarque_bera(series)
            metrics.append({
                self.id_column: fund_id,
                "jarque_bera": float(stat),
                "jb_pvalue": float(pvalue),
            })
        return pl.DataFrame(metrics)

    def cointegration_test(self, df: pl.DataFrame, value_column: str) -> pl.DataFrame:
        funds = df.select(self.id_column).unique()[self.id_column].to_list()
        outputs = []
        for i, fund_a in enumerate(funds):
            for fund_b in funds[i + 1 :]:
                series_a = df.filter(pl.col(self.id_column) == fund_a)[value_column].to_numpy()
                series_b = df.filter(pl.col(self.id_column) == fund_b)[value_column].to_numpy()
                stat, pvalue, _ = coint(series_a, series_b)
                outputs.append({
                    "fund_a": fund_a,
                    "fund_b": fund_b,
                    "coint_stat": float(stat),
                    "coint_pvalue": float(pvalue),
                })
        return pl.DataFrame(outputs)

    def correlation_matrix(self, df: pl.DataFrame, value_column: str) -> pl.DataFrame:
        pivot = df.pivot(index=self.date_column, columns=self.id_column, values=value_column)
        corr = pivot.to_pandas().corr()
        return pl.DataFrame(corr.reset_index().rename(columns={"index": self.id_column}))

    def plot_rolling_r_squared(
        self,
        df: pl.DataFrame,
        value_column: str,
        window: int = 63,
        output_path: Optional[Path] = None,
    ) -> None:
        pivot = df.pivot(index=self.date_column, columns=self.id_column, values=value_column).to_pandas().dropna()
        rolling = pivot.rolling(window).apply(lambda x: x.corr().pow(2).mean().mean(), raw=False)
        plt.figure(figsize=(10, 4))
        plt.plot(rolling.index, rolling.values)
        plt.title("Rolling Average R-squared")
        plt.xlabel("Date")
        plt.ylabel("R-squared")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close()

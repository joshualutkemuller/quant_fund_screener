"""Visualization utilities for the quant research toolkit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
import plotly.express as px
from matplotlib import pyplot as plt


@dataclass
class Visualizer:
    """Plot technical, risk, and scoring outputs."""

    id_column: str = "fund_id"
    date_column: str = "date"

    def quant_score_bar(self, df: pl.DataFrame, output_path: Optional[Path] = None) -> None:
        latest = df.sort(self.date_column).group_by(self.id_column).tail(1)
        fig = px.bar(latest.to_pandas(), x=self.id_column, y="quant_score", title="Quant Score by Fund")
        if output_path:
            fig.write_image(output_path)
        else:
            fig.show()

    def decile_distribution(self, df: pl.DataFrame, output_path: Optional[Path] = None) -> None:
        counts = df.group_by([self.date_column, "quant_bucket"]).agg(pl.len().alias("count"))
        pivot = counts.pivot(index=self.date_column, columns="quant_bucket", values="count")
        pivot_pd = pivot.to_pandas().set_index(self.date_column)
        pivot_pd.plot.area(figsize=(10, 6))
        plt.title("Quant Score Decile Distribution")
        plt.xlabel("Date")
        plt.ylabel("Number of Funds")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close()

    def risk_heatmap(self, tidy_df: pl.DataFrame, output_path: Optional[Path] = None) -> None:
        pivot = tidy_df.pivot(index=self.id_column, columns="metric", values="value")
        fig = px.imshow(pivot.to_pandas(), labels=dict(color="Value"), title="Risk Factor Heatmap")
        if output_path:
            fig.write_image(output_path)
        else:
            fig.show()

    def rolling_metric_plot(
        self,
        df: pl.DataFrame,
        metric: str,
        output_path: Optional[Path] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for fund_id, group in df.group_by(self.id_column):
            ax.plot(group[self.date_column], group[metric], label=str(fund_id))
        ax.set_title(f"Rolling {metric}")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close()

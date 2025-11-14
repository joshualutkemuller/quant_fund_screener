"""Factor regression utilities for estimating alpha and factor exposures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl
import statsmodels.api as sm


@dataclass
class AlphaModel:
    """Run time-series and cross-sectional regressions to estimate alpha."""

    id_column: str = "fund_id"
    date_column: str = "date"

    def time_series_regression(
        self,
        df: pl.DataFrame,
        factors: Iterable[str],
        excess_return_column: str = "excess_return",
    ) -> pl.DataFrame:
        """Run separate regressions per fund to estimate alpha and betas."""
        results = []
        for fund_id, group in df.group_by(self.id_column):
            pandas_df = group.select([self.date_column, excess_return_column, *factors]).to_pandas()
            pandas_df = pandas_df.dropna()
            if pandas_df.empty:
                continue
            y = pandas_df[excess_return_column]
            X = sm.add_constant(pandas_df[list(factors)])
            model = sm.OLS(y, X).fit()
            res = {
                self.id_column: fund_id,
                "alpha": model.params.get("const", float("nan")),
                "alpha_t": model.tvalues.get("const", float("nan")),
                "r_squared": model.rsquared,
            }
            for factor in factors:
                res[f"beta_{factor}"] = model.params.get(factor, float("nan"))
                res[f"t_{factor}"] = model.tvalues.get(factor, float("nan"))
            results.append(res)
        return pl.DataFrame(results)

    def cross_sectional_regression(
        self,
        df: pl.DataFrame,
        factors: Iterable[str],
        excess_return_column: str = "excess_return",
    ) -> pl.DataFrame:
        """Run cross-sectional regression for each date across funds."""
        outputs = []
        for date, group in df.group_by(self.date_column):
            pandas_df = group.select([self.id_column, excess_return_column, *factors]).to_pandas().dropna()
            if pandas_df.empty or len(pandas_df) < len(list(factors)):
                continue
            y = pandas_df[excess_return_column]
            X = sm.add_constant(pandas_df[list(factors)])
            model = sm.OLS(y, X).fit()
            res = {
                self.date_column: date,
                "r_squared": model.rsquared,
            }
            for factor in factors:
                res[f"lambda_{factor}"] = model.params.get(factor, float("nan"))
                res[f"t_{factor}"] = model.tvalues.get(factor, float("nan"))
            outputs.append(res)
        return pl.DataFrame(outputs)

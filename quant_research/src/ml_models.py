"""Machine learning enhancement for predicting excess returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except ModuleNotFoundError:  # pragma: no cover
    XGBRegressor = None


@dataclass
class PredictiveModel:
    """Train predictive models using engineered features."""

    target_column: str = "excess_return"

    def _prepare_data(self, df: pl.DataFrame, feature_columns: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        pandas_df = df.select([*feature_columns, self.target_column]).to_pandas().dropna()
        X = pandas_df[feature_columns].values
        y = pandas_df[self.target_column].values
        return X, y

    def fit_regression(self, df: pl.DataFrame, feature_columns: Iterable[str], model_type: str = "lasso") -> dict:
        X, y = self._prepare_data(df, feature_columns)
        if model_type == "lasso":
            model = Lasso(alpha=0.01, max_iter=5000)
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "xgboost":
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed")
            model = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.fit(X, y)
        predictions = model.predict(X)
        rmse = mean_squared_error(y, predictions, squared=False)
        importance = getattr(model, "coef_", getattr(model, "feature_importances_", None))
        return {
            "model": model,
            "rmse": rmse,
            "feature_importance": importance,
        }

    def cross_validate(self, df: pl.DataFrame, feature_columns: Iterable[str], model_type: str = "lasso") -> float:
        X, y = self._prepare_data(df, feature_columns)
        if model_type == "lasso":
            model = Lasso(alpha=0.01, max_iter=5000)
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
        else:
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed")
            model = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05)
        cv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
        return float(np.mean(np.sqrt(-scores)))

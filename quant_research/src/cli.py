"""Command line interface for the quant research toolkit."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .alpha_model import AlphaModel
from .data_loader import DataLoader
from .feature_engineering import FundFeatureEngineer
from .portfolio_analytics import PortfolioAnalytics
from .risk_metrics import RiskAnalytics
from .scoring_model import ScoreConfig, ScoringEngine
from .statistical_evaluation import StatisticalEvaluator
from .visualization import Visualizer


def _resolve_paths(root: Path) -> dict:
    return {
        "data": root / "data",
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
        "config": root / "config" / "config.yaml",
    }


def load_data(args: argparse.Namespace) -> None:
    paths = _resolve_paths(Path(args.project_root))
    loader = DataLoader(paths["data"])
    prices = loader.load_prices("prices.csv")
    fundamentals = loader.load_fundamentals("fundamentals.csv")
    merged = loader.merge_data(prices, fundamentals)
    output_path = paths["reports"] / "merged_data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(output_path)
    print(f"Saved merged data to {output_path}")


def compute_scores(args: argparse.Namespace) -> None:
    paths = _resolve_paths(Path(args.project_root))
    loader = DataLoader(paths["data"])
    prices = loader.load_prices("prices.csv")
    fundamentals = loader.load_fundamentals("fundamentals.csv")
    engineer = FundFeatureEngineer()
    technical = engineer.compute_technical_indicators(prices)
    merged = engineer.merge_with_fundamentals(
        technical,
        fundamentals,
        fundamental_columns=["pe_ratio", "roe", "pb_ratio", "dividend_yield"],
    )
    merged = engineer.compute_valuation_zscores(merged, ["pe_ratio", "pb_ratio"])
    risk = RiskAnalytics()
    risk_df = risk.compute_risk_metrics(merged)
    latest = merged.sort(engineer.date_column).group_by(engineer.id_column).tail(1)
    risk_enriched = risk_df.join(
        latest.select(
            engineer.id_column,
            engineer.date_column,
            "momentum_63",
        ),
        on=[engineer.id_column, engineer.date_column],
        how="left",
    )
    config = ScoreConfig.from_yaml(paths["config"])
    scorer = ScoringEngine(config)
    scored = scorer.compute_scores(risk_enriched)
    scored = scorer.top_bottom_deciles(scored)
    output_path = paths["reports"] / "scored.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(output_path)
    print(f"Saved scores to {output_path}")


def visualize(args: argparse.Namespace) -> None:
    paths = _resolve_paths(Path(args.project_root))
    scored_path = paths["reports"] / "scored.parquet"
    df = pl.read_parquet(scored_path)
    visualizer = Visualizer()
    figures_dir = paths["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    metric = args.metric.lower()
    if metric == "sharpe_ratio":
        risk = RiskAnalytics()
        rolling = risk.rolling_sharpe(df, window=args.window)
        risk.plot_rolling_sharpe(rolling, output_path=figures_dir / "rolling_sharpe.png")
    elif metric == "quant_score":
        visualizer.quant_score_bar(df, output_path=figures_dir / "quant_scores.png")
    else:
        visualizer.rolling_metric_plot(df, metric, output_path=figures_dir / f"{metric}.png")
    print(f"Saved visualisation for {metric} to {figures_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-model", description="Quant research command line interface")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], help="Project root directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser("load-data", help="Load and merge raw data")
    load_parser.add_argument("--from", dest="data_source", default="local", help="Data source identifier")
    load_parser.set_defaults(func=load_data)

    compute_parser = subparsers.add_parser("compute-scores", help="Compute quant scores")
    compute_parser.add_argument("--window", type=int, default=252, help="Rolling window for metrics")
    compute_parser.set_defaults(func=compute_scores)

    viz_parser = subparsers.add_parser("visualize", help="Generate visualisations")
    viz_parser.add_argument("--metric", required=True, help="Metric to visualise")
    viz_parser.add_argument("--window", type=int, default=126, help="Rolling window for metrics")
    viz_parser.set_defaults(func=visualize)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

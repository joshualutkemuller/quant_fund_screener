# Quant Fund Screener

This repository contains a modular quant research toolkit that evaluates investment funds using technical, fundamental, risk, and statistical analytics. The project is organised under the `quant_research/` directory with dedicated modules for data ingestion, feature engineering, risk analytics, scoring, visualisation, portfolio construction, machine learning, and statistical evaluation.

## Project structure

```
quant_research/
├── src/
│   ├── alpha_model.py
│   ├── cli.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── ml_models.py
│   ├── portfolio_analytics.py
│   ├── risk_metrics.py
│   ├── scoring_model.py
│   ├── statistical_evaluation.py
│   └── visualization.py
├── config/
│   └── config.yaml
├── data/
├── reports/
└── tests/
```

## Getting started

1. Install dependencies (Polars, Statsmodels, scikit-learn, XGBoost, Plotly, Matplotlib, PyYAML).
2. Place raw fund price and fundamental data in `quant_research/data/` as `prices.csv` and `fundamentals.csv`.
3. Configure metric weights in `quant_research/config/config.yaml`.
4. Run the CLI:

```bash
python -m quant_research.src.cli load-data
python -m quant_research.src.cli compute-scores --window 252
python -m quant_research.src.cli visualize --metric sharpe_ratio
```

Generated artefacts are saved to `quant_research/reports/` and `quant_research/reports/figures/`.

## Testing

Run unit tests with:

```bash
pytest
```

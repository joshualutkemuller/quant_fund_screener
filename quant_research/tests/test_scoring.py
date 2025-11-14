"""Basic tests for the scoring engine."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pl = pytest.importorskip("polars")

from quant_research.src.scoring_model import ScoreConfig, ScoringEngine


def test_quant_score_ranking(tmp_path):
    data = pl.DataFrame(
        {
            "fund_id": ["A", "B", "C"],
            "date": ["2023-01-31"] * 3,
            "volatility": [0.1, 0.2, 0.3],
            "sharpe_ratio": [1.0, 0.5, 0.2],
            "sortino_ratio": [1.1, 0.7, 0.4],
            "max_drawdown": [-0.1, -0.2, -0.3],
            "momentum_63": [0.05, 0.02, -0.01],
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
weights:
  volatility: -0.5
  sharpe_ratio: 1.0
  sortino_ratio: 0.8
  max_drawdown: -0.7
  momentum_63: 0.6
"""
    )
    config = ScoreConfig.from_yaml(config_path)
    engine = ScoringEngine(config)
    scored = engine.compute_scores(data)
    assert scored.sort("quant_rank")["fund_id"][0] == "A"

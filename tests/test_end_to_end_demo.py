"""Offline end-to-end: demo CLI, reproducibility of fixed-input runs, protocol gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from nfl_origination.cli import app
from nfl_origination.config import load_config
from nfl_origination.errors import InvalidInputError
from nfl_origination.experiment import run_backtest

ROOT = Path(__file__).resolve().parents[1]
PROB_TOL = 1e-8
METRIC_TOL = 1e-6


def _small_config(tmp_path: Path, kind: str = "development", **model_overrides) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    raw["run"] = {
        "kind": kind,
        "label": "test",
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
    }
    raw["data"]["seasons"] = [2010, 2020]
    raw["evaluation"]["bootstrap_replicates"] = 200
    raw["model"].update(model_overrides)
    raw["market"]["odds_path"] = str(
        ROOT / "tests" / "fixtures" / "synthetic" / "odds_synthetic.csv"
    )
    path = tmp_path / f"{kind}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_fixed_inputs_reproduce_within_tolerance(tmp_path):
    cfg = load_config(_small_config(tmp_path))
    m1 = run_backtest(cfg, offline=True)
    m2 = run_backtest(cfg, offline=True)
    d1 = cfg.run.artifacts_dir / "runs" / m1.run_id
    d2 = cfg.run.artifacts_dir / "runs" / m2.run_id
    p1 = pd.read_parquet(d1 / "predictions.parquet").sort_values("game_id").reset_index(drop=True)
    p2 = pd.read_parquet(d2 / "predictions.parquet").sort_values("game_id").reset_index(drop=True)
    for col in (
        "p_home_win",
        "p_tie",
        "p_away_win",
        "fair_home_handicap_p_win",
        "fair_total_p_over",
    ):
        assert np.abs(p1[col] - p2[col]).max() <= PROB_TOL
    assert (p1["fair_home_handicap"] == p2["fair_home_handicap"]).all()
    met1 = json.loads((d1 / "metrics.json").read_text())
    met2 = json.loads((d2 / "metrics.json").read_text())
    for mid in met1["selected_model_ids"]:
        for k, v in met1["models"][mid]["pooled"].items():
            assert abs(v - met2["models"][mid]["pooled"][k]) <= METRIC_TOL, k
    assert (
        met1["decision_record"]["full"]["selected_alpha"]
        == met2["decision_record"]["full"]["selected_alpha"]
    )
    assert m1.config_hash == m2.config_hash and m1.run_id != m2.run_id


def test_holdout_gate_refuses_without_freeze(tmp_path):
    cfg = load_config(
        _small_config(tmp_path, "holdout", selected_alpha=100, ablation_selected_alpha=100)
    )
    with pytest.raises(InvalidInputError):
        run_backtest(cfg, offline=True)


def test_demo_cli_runs_offline(tmp_path):
    runner = CliRunner()
    cfg_path = _small_config(tmp_path, "demo")
    result = runner.invoke(app, ["demo", "--config", str(cfg_path), "--offline"])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    payload = json.loads(result.output[result.output.rindex("{\n") :])
    run_dir = Path(tmp_path / "artifacts" / "runs" / payload["run_id"])
    assert (run_dir / "report" / "report.md").exists()
    assert (run_dir / "paper_backtest.json").exists()
    report = (run_dir / "report" / "report.md").read_text()
    assert "SYNTHETIC" in report and "Limitations" in report
    for name in ("reliability.png", "interval_coverage.png", "key_numbers.png"):
        assert (run_dir / "report" / "charts" / name).exists()
    pb = json.loads((run_dir / "paper_backtest.json").read_text())
    assert pb["synthetic"] and pb["n_bets"] > 0 and "roi_bootstrap" in pb
    assert pb["wins"] + pb["losses"] + pb["pushes"] + pb["voids"] == pb["n_bets"]


def test_cli_exit_codes(tmp_path):
    runner = CliRunner()
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {kind: development}\nmodel: {nope: 1}\n")
    result = runner.invoke(app, ["backtest", "--config", str(bad)])
    assert result.exit_code == 2
    result = runner.invoke(
        app, ["report", "--run", "latest-nothing", "--config", str(_small_config(tmp_path))]
    )
    assert result.exit_code == 2

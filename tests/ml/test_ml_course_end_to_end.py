"""合成世界でコース適性モデルを端から端まで通す。catboost が無ければ飛ばす。

合成世界（tests/fixtures/ml_synthetic.py）にはコースの効果を入れていないので、ここでは
的中数ではなく、学習・検算・評価・予想の手順がコース適性モデルで通ることを確かめる。
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from keiba_ml.config import ModelConfig, SplitConfig, load_config
from keiba_ml.dataset import build
from keiba_ml.experiment import run_comparison
from tests.fixtures.ml_synthetic import build_world

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ml" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("world"), races_per_year=120)


@pytest.fixture(scope="module")
def dataset(world):
    config = load_config(model="course")
    df, ped, names, report = build.build(config, race_data_dir=world.race_data_dir,
                                         ml_data_dir=world.ml_data_dir,
                                         blood_data_dir=world.blood_data_dir)
    build.save(df, ped, names, report, config, ml_data_dir=world.ml_data_dir)
    return build.load(world.ml_data_dir)


def test_course_model_runs_end_to_end(world, dataset, capsys):
    pytest.importorskip("catboost")
    df, _, manifest = dataset
    assert "track_cd" in df.columns and manifest["schema_version"] == "5"
    config = load_config(model="course").with_overrides(
        split=SplitConfig(test_start=date(2021, 1, 1), test_end=date(2021, 12, 31),
                          valid_years=1, train_years=(5,)),
        model=ModelConfig(iterations=100, learning_rate=0.1, early_stopping_rounds=30,
                          thread_count=2, loss_function=load_config(model="course")
                          .model.loss_function))
    out_root = world.ml_data_dir / "runs" / "course"
    table = run_comparison(df, config, out_root=out_root, run_id="t", log_period=0,
                           dataset_id=manifest["dataset_id"])
    assert {"course_catboost", "course_rel_venue", "course_draw"} <= set(table["model"])
    checks = json.loads((out_root / "t" / "5y" / "checks.json").read_text(encoding="utf-8"))
    assert {c["name"]: c["status"] for c in checks} == {
        "feature_columns": "ok", "split": "ok", "course_asof_sample": "ok"}
    preds = pd.read_parquet(out_root / "t" / "5y" / "predictions_test.parquet")
    assert set(preds["model_name"]) == {"course_catboost"}

    model_dir = out_root / "t" / "5y"
    common = ["--model", str(model_dir), "--ml-data-dir", str(world.ml_data_dir)]
    assert _script("evaluate").main([*common, "--from", "2022-01-01", "--to", "2022-12-31"]) == 0
    result = json.loads((model_dir / "holdout_20220101_20221231.json").read_text(encoding="utf-8"))
    assert {c["name"]: c["status"] for c in result["checks"]} == {
        "feature_columns": "ok", "course_asof_sample": "ok"}
    capsys.readouterr()
    assert _script("predict").main(common) == 0
    out = capsys.readouterr().out
    assert world.entry_race_id in out and "コース適性モデル" in out and "この競馬場の出走" in out

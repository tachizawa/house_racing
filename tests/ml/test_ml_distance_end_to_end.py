"""合成世界で距離適性モデルを端から端まで通す。catboost が無ければ学習のテストは飛ばす。

合成世界の馬には得意距離があり、距離が離れるほど走りが落ちる（tests/fixtures/ml_synthetic.py）。
距離適性モデルは馬の強さを見ないので、偶然を上回る分は距離の合い方だけから来る。
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from keiba_ml.config import ExperimentConfig, MLConfig, ModelConfig, SplitConfig, load_config
from keiba_ml.dataset import build
from keiba_ml.experiment import IntegrityError, run_comparison
from keiba_ml.features.distance import build_features
from tests.fixtures.ml_synthetic import build_world

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ml" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distance_config(**split) -> MLConfig:
    """configs/ml/distance.toml を基に、テスト用に期間と学習回数を縮めた設定。"""
    config = load_config(model="distance")
    split_config = SplitConfig(**{"test_start": date(2021, 1, 1), "test_end": date(2021, 12, 31),
                                  "valid_years": 1, "train_years": (5,), **split})
    return config.with_overrides(
        split=split_config,
        model=ModelConfig(iterations=150, learning_rate=0.1, early_stopping_rounds=30,
                          thread_count=2))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("world"), races_per_year=120)


@pytest.fixture(scope="module")
def dataset(world):
    config = load_config(model="distance")
    df, ped, names, report = build.build(config, race_data_dir=world.race_data_dir,
                                         ml_data_dir=world.ml_data_dir,
                                         blood_data_dir=world.blood_data_dir)
    build.save(df, ped, names, report, config, ml_data_dir=world.ml_data_dir)
    return build.load(world.ml_data_dir)


def test_features_recover_the_true_preferred_distance(world, dataset):
    """距離と走りの傾きは真の得意距離と、今回の距離への当てはめは真の距離の合い方と同じ向き。"""
    df, _, _ = dataset
    frame = build_features(df, load_config(model="distance").distance).frame
    km = df["kyori_m"].astype(float)
    truth = df["ketto_num"].map(lambda k: world.horses[k]["pref_km"])
    runs_before = df.groupby("ketto_num").cumcount()
    late = (df["year"] >= 2018) & (runs_before >= 6)
    assert np.corrcoef(frame.loc[late, "own_slope_pf"], truth[late])[0, 1] > 0.2
    fit = -(km / 1000.0 - truth) ** 2
    assert np.corrcoef(frame.loc[late, "own_fit_pf"], fit[late])[0, 1] > 0.2


def test_distance_model_beats_chance_and_passes_the_checks(world, dataset):
    pytest.importorskip("catboost")
    df, _, manifest = dataset
    out_root = world.ml_data_dir / "runs" / "distance"
    table = run_comparison(df, _distance_config(), out_root=out_root, run_id="t", log_period=0,
                           dataset_id=manifest["dataset_id"])
    model_rows = table[table["model"] == "distance_catboost"]
    assert len(model_rows) == 1
    row = model_rows.iloc[0]
    assert row["hits_ci_low"] > row["hits_random"]
    assert row["brier"] < row["brier_base"]
    assert {"dist_rel_band", "dist_gap"} <= set(table["model"])
    checks = json.loads((out_root / "t" / "5y" / "checks.json").read_text(encoding="utf-8"))
    assert {c["name"]: c["status"] for c in checks} == {
        "feature_columns": "ok", "split": "ok", "distance_asof_sample": "ok"}
    config = json.loads((out_root / "t" / "config.json").read_text(encoding="utf-8"))
    assert config["experiment"]["model"] == "distance"
    meta = json.loads((out_root / "t" / "5y" / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == "distance_catboost"
    preds = pd.read_parquet(out_root / "t" / "5y" / "predictions_test.parquet")
    assert set(preds["model_name"]) == {"distance_catboost"}


def test_holdout_and_live_prediction_scripts_work_for_the_distance_model(world, dataset, capsys):
    pytest.importorskip("catboost")
    df, _, manifest = dataset
    out_root = world.ml_data_dir / "runs" / "distance"
    run_comparison(df, _distance_config(), out_root=out_root, run_id="hold", log_period=0,
                   dataset_id=manifest["dataset_id"])
    model_dir = out_root / "hold" / "5y"
    common = ["--model", str(model_dir), "--ml-data-dir", str(world.ml_data_dir)]

    evaluate = _script("evaluate")
    assert evaluate.main([*common, "--from", "2022-01-01", "--to", "2022-12-31"]) == 0
    result = json.loads((model_dir / "holdout_20220101_20221231.json").read_text(encoding="utf-8"))
    assert "distance_catboost-dist_gap" in result["differences"]
    assert {c["name"]: c["status"] for c in result["checks"]} == {
        "feature_columns": "ok", "distance_asof_sample": "ok"}
    # 血統用の入口は距離適性の記録を拒否する
    capsys.readouterr()
    assert _script("evaluate_pedigree").main([*common, "--from", "2022-02-01",
                                              "--to", "2022-12-31"]) == 2
    assert "距離適性" in capsys.readouterr().out

    assert _script("predict").main(common) == 0
    out = capsys.readouterr().out
    assert world.entry_race_id in out and "前走距離" in out and "距離適性モデル" in out
    assert _script("predict_pedigree").main(common) == 2


def test_train_script_writes_under_the_model_name_and_checks_the_config(world, dataset, tmp_path,
                                                                         capsys):
    train = _script("train")
    # 血統の設定ファイルを距離適性に渡すのは取り違え
    with pytest.raises(SystemExit):
        train.main(["--model", "distance", "--config", str(ROOT / "configs/ml/pedigree.toml"),
                    "--ml-data-dir", str(world.ml_data_dir)])
    assert "pedigree" in capsys.readouterr().err


def test_old_dataset_without_the_new_columns_is_refused(dataset, tmp_path):
    df, _, _ = dataset
    old = df.drop(columns=["result_margin_sec"])
    with pytest.raises(IntegrityError, match="build_dataset.py"):
        run_comparison(old, _distance_config(), out_root=tmp_path, run_id="x")
    assert not (tmp_path / "x").exists()        # 実行の記録を書く前に止める


def test_pedigree_model_still_runs_through_the_generic_path(world, dataset):
    pytest.importorskip("catboost")
    df, _, manifest = dataset
    config = _distance_config()
    config = load_config(model="pedigree").with_overrides(
        split=config.split, model=config.model)
    assert config.experiment == ExperimentConfig(model="pedigree")
    table = run_comparison(df, config, out_root=world.ml_data_dir / "runs" / "pedigree",
                           run_id="p", log_period=0, dataset_id=manifest["dataset_id"])
    assert {"pedigree_catboost", "sire_ae", "sire_cond_ae"} <= set(table["model"])


def test_scripts_refuse_bad_inputs_before_scoring(world, dataset, monkeypatch, capsys):
    """predict: --top-n < 1 と学習時と違う設定のデータセットを拒否。evaluate: FAIL のあるデータセットを拒否。"""
    pytest.importorskip("catboost")
    df, _, manifest = dataset
    out_root = world.ml_data_dir / "runs" / "distance"
    run_comparison(df, _distance_config(), out_root=out_root, run_id="refuse", log_period=0,
                   dataset_id=manifest["dataset_id"])
    model_dir = out_root / "refuse" / "5y"
    common = ["--model", str(model_dir), "--ml-data-dir", str(world.ml_data_dir)]
    predict = _script("predict")
    with pytest.raises(SystemExit):
        predict.main([*common, "--top-n", "0"])

    monkeypatch.setattr(build, "config_mismatches", lambda _m, _c: ["pedigree_generations"])
    capsys.readouterr()
    assert predict.main(common) == 2
    assert "pedigree_generations" in capsys.readouterr().out
    monkeypatch.undo()

    loaded = build.load(world.ml_data_dir)
    failed = (loaded[0], loaded[1], {**loaded[2], "checks_failed": ["duplicates"]})
    monkeypatch.setattr(build, "load", lambda *_a, **_k: failed)
    assert _script("evaluate").main([*common, "--from", "2023-01-01", "--to", "2023-12-31"]) == 2
    assert "FAIL" in capsys.readouterr().out
    assert not (model_dir / "holdout_20230101_20231231.json").exists()


def test_train_script_records_the_years_it_runs_and_keeps_old_runs(world, dataset, monkeypatch):
    train = _script("train")
    seen = {}

    def fake_run_comparison(_df, config, *, out_root, run_id, train_years, **_kw):
        seen["config"], seen["years"] = config, train_years
        (out_root / run_id).mkdir(parents=True)
        return pd.DataFrame()

    monkeypatch.setattr(train, "run_comparison", fake_run_comparison)
    common = ["--model", "distance", "--ml-data-dir", str(world.ml_data_dir), "--run-id", "years"]
    assert train.main([*common, "--years", "20"]) == 0
    assert seen["years"] == (20,) and seen["config"].split.train_years == (20,)
    # 同じ出力先には書かない（前の実行の記録と混ざる）
    seen.clear()
    assert train.main([*common, "--years", "10"]) == 2
    assert seen == {}

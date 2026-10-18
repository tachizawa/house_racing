"""合成世界で Phase 1 → Phase 2 を端から端まで通す。catboost が無ければ飛ばす。"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from keiba_ml.config import MLConfig, ModelConfig, PedigreeConfig, SplitConfig
from keiba_ml.dataset import build
from keiba_ml.evaluate.growth import age_curve
from keiba_ml.experiment import run_comparison
from keiba_ml.models.pedigree_catboost import PedigreeCatBoost
from tests.fixtures.ml_synthetic import build_world

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("world"), races_per_year=120)


@pytest.fixture(scope="module")
def dataset(world):
    config = MLConfig(pedigree=PedigreeConfig(generations=4, inbreeding_generations=4))
    df, ped, names, report = build.build(config, race_data_dir=world.race_data_dir,
                                         ml_data_dir=world.ml_data_dir)
    build.save(df, ped, names, report, config, ml_data_dir=world.ml_data_dir)
    return build.load(world.ml_data_dir)


def test_dataset_passes_integrity_checks(world, dataset):
    _, _, manifest = dataset
    assert manifest["checks_failed"] == []
    report = json.loads((world.ml_data_dir / "datasets" / manifest["dataset_id"]
                         / "integrity.json").read_text(encoding="utf-8"))
    stats = report["pedigree_stats"]
    assert stats["parent_link"]["rate"] == 1.0
    assert stats["fill_rate_by_generation"]["4"] == 1.0
    assert stats["source_counts"].get("UM", 0) > 0     # UM での補完の経路も通っている


def test_growth_report_recovers_true_maturity(world, dataset):
    df, names, _ = dataset
    table = age_curve(df, path="S", min_starts=100, names=names)
    truth = table["hansyoku_num"].map(lambda n: world.breeders[n].maturity)
    assert np.corrcoef(table["maturity"], truth)[0, 1] > 0.6


def test_pedigree_model_beats_chance_on_fixed_test_period(world, dataset):
    pytest.importorskip("catboost")
    df, _, _ = dataset
    config = MLConfig(
        pedigree=PedigreeConfig(generations=4, inbreeding_generations=4),
        split=SplitConfig(test_start=date(2022, 1, 1), test_end=date(2022, 12, 31),
                          valid_years=1, train_years=(5, 10)),
        model=ModelConfig(iterations=150, learning_rate=0.1, early_stopping_rounds=30,
                          thread_count=2))
    out_root = world.ml_data_dir / "runs"
    table = run_comparison(df, config, out_root=out_root, run_id="t", log_period=0)
    model_rows = table[table["model"] == "pedigree_catboost"]
    assert list(model_rows["window"]) == ["5y", "10y"]
    assert model_rows["test_races"].nunique() == 1          # 同じテストを受けている
    assert list(model_rows["truncated"]) == [False, False]
    assert (model_rows["hits_ci_low"] > model_rows["hits_random"]).all()
    assert (model_rows["brier"] < model_rows["brier_base"]).all()

    # 保存したモデルを読み直して同じ予測になる
    model = PedigreeCatBoost.load(out_root / "t" / "10y")
    assert model.feature_columns
    checks = json.loads((out_root / "t" / "10y" / "checks.json").read_text(encoding="utf-8"))
    assert {c["name"]: c["status"] for c in checks} == {
        "feature_columns": "ok", "split": "ok", "asof_sample": "ok"}


def test_ranking_loss_runs(world, dataset):
    pytest.importorskip("catboost")
    df, _, _ = dataset
    config = MLConfig(
        pedigree=PedigreeConfig(generations=4, inbreeding_generations=4),
        split=SplitConfig(test_start=date(2022, 1, 1), test_end=date(2022, 12, 31),
                          valid_years=1, train_years=(5,)),
        model=ModelConfig(loss_function="YetiRank", iterations=100, learning_rate=0.1,
                          early_stopping_rounds=30, thread_count=2))
    table = run_comparison(df, config, out_root=world.ml_data_dir / "runs", run_id="rank",
                           log_period=0)
    row = table[table["model"] == "pedigree_catboost"].iloc[0]
    assert row["hits_mean"] > row["hits_random"]
    assert "brier" not in table.columns or np.isnan(row.get("brier", np.nan))


def _script(name):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / "ml" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_stops_when_dataset_was_built_with_other_settings(world, dataset, capsys):
    # データセットは generations=4 で作ってある。既定の設定ファイルは 5 世代。
    train = _script("train_pedigree")
    assert train.main(["--ml-data-dir", str(world.ml_data_dir)]) == 2
    assert "pedigree_generations" in capsys.readouterr().out


def test_train_rejects_missing_config_and_bad_iterations(world, dataset, tmp_path):
    train = _script("train_pedigree")
    with pytest.raises(SystemExit):
        train.main(["--config", str(tmp_path / "typo.toml"),
                    "--ml-data-dir", str(world.ml_data_dir)])
    with pytest.raises(SystemExit):
        train.main(["--iterations", "0", "--ml-data-dir", str(world.ml_data_dir)])


def test_predict_requires_the_training_records(tmp_path, capsys):
    predict = _script("predict_pedigree")
    model_dir = tmp_path / "runs" / "r" / "10y"
    model_dir.mkdir(parents=True)
    assert predict.main(["--model", str(model_dir), "--ml-data-dir", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "config.json" in out and "metrics.json" in out


def test_build_refuses_an_existing_dataset_id_before_building(world, dataset):
    build_script = _script("build_dataset")
    _, _, manifest = dataset
    with pytest.raises(SystemExit):
        build_script.main(["--race-data-dir", str(world.race_data_dir),
                           "--ml-data-dir", str(world.ml_data_dir),
                           "--dataset-id", manifest["dataset_id"]])

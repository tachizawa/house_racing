"""合成世界で Phase 1 → Phase 2 を端から端まで通す。catboost が無ければ飛ばす。"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from keiba_ml.config import DataConfig, MLConfig, ModelConfig, PedigreeConfig, SplitConfig
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
    # UM での補完の経路も通すため、この層の血統の版（"ml"）から作る。
    config = MLConfig(data=DataConfig(pedigree_source="ml"),
                      pedigree=PedigreeConfig(generations=4, inbreeding_generations=4))
    df, ped, names, report = build.build(config, race_data_dir=world.race_data_dir,
                                         ml_data_dir=world.ml_data_dir)
    build.save(df, ped, names, report, config, ml_data_dir=world.ml_data_dir)
    return build.load(world.ml_data_dir)


def test_blood_version_gives_the_same_pedigree_as_the_ml_version(world):
    """既存の血統の版（BLDN の列名）から作った祖先表は、この層の版から作ったものと一致する。

    違うのは UM だけの馬（BLDN には UM の3代血統が無い）が欠けることだけ。
    """
    pedigree = PedigreeConfig(generations=4, inbreeding_generations=4)
    built = {}
    for source in ("ml", "blood"):
        config = MLConfig(data=DataConfig(pedigree_source=source), pedigree=pedigree)
        df, ped, _names, report = build.build(
            config, race_data_dir=world.race_data_dir, ml_data_dir=world.ml_data_dir,
            blood_data_dir=world.blood_data_dir)
        assert report["pedigree_source"] == source
        built[source] = (df, ped, report)
    ml_ped, blood_ped = built["ml"][1], built["blood"][1]
    from_sk = ml_ped.index[ml_ped["pedigree_source"] == "SK"]
    assert len(from_sk) > 0
    cols = [c for c in ml_ped.columns if c.startswith("ped_")] + ["birth_year", "inbreeding_f"]
    pd.testing.assert_frame_equal(blood_ped.loc[from_sk, cols], ml_ped.loc[from_sk, cols])
    um_only = ml_ped.index[ml_ped["pedigree_source"] == "UM"]
    assert len(um_only) > 0
    assert (blood_ped.loc[um_only, "pedigree_source"] == "").all()
    stats = built["blood"][2]["pedigree_stats"]
    assert stats["parent_link"]["rate"] == 1.0


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


def test_pedigree_model_beats_chance_on_fixed_test_period(world, dataset, capsys):
    pytest.importorskip("catboost")
    df, _, manifest = dataset
    config = MLConfig(
        pedigree=PedigreeConfig(generations=4, inbreeding_generations=4),
        split=SplitConfig(test_start=date(2022, 1, 1), test_end=date(2022, 12, 31),
                          valid_years=1, train_years=(5, 10)),
        model=ModelConfig(iterations=150, learning_rate=0.1, early_stopping_rounds=30,
                          thread_count=2))
    out_root = world.ml_data_dir / "runs"
    table = run_comparison(df, config, out_root=out_root, run_id="t", log_period=0,
                           dataset_id=manifest["dataset_id"])
    model_rows = table[table["model"] == "pedigree_catboost"]
    assert list(model_rows["window"]) == ["5y", "10y"]
    assert model_rows["test_races"].nunique() == 1          # 同じテストを受けている
    assert list(model_rows["truncated"]) == [False, False]
    assert (model_rows["hits_ci_low"] > model_rows["hits_random"]).all()
    assert (model_rows["brier"] < model_rows["brier_base"]).all()

    # 学習期間どうしの差（対応あり）は保存した予測から出し、学習し直さずに出し直せる
    diffs = pd.read_csv(out_root / "t" / "window_differences.csv")
    assert list(diffs["window"]) == ["10y"] and list(diffs["versus"]) == ["5y"]
    assert diffs["diff_ci_low"].iloc[0] <= diffs["hits_diff"].iloc[0] <= diffs["diff_ci_high"].iloc[0]
    compare = _script("compare_runs")
    common = ["--run", str(out_root / "t"), "--ml-data-dir", str(world.ml_data_dir)]
    assert compare.main(common) == 0              # 学習時のデータセット（dataset.json）を使う
    pd.testing.assert_frame_equal(pd.read_csv(out_root / "t" / "window_differences.csv"), diffs)
    capsys.readouterr()
    assert compare.main([*common, "--dataset-id", "other"]) == 2   # 違うデータセットとは突き合わせない
    assert "突き合わせません" in capsys.readouterr().out

    # 正解のレースが予測と違うデータセットとは突き合わせない
    shrunk = df[df["race_id"] != df.loc[df["year"] == 2022, "race_id"].iloc[0]]
    build.save(shrunk, pd.DataFrame(), {}, {"race_versions": [], "pedigree_versions": [],
                                         "checks": []},
               config, ml_data_dir=world.ml_data_dir, dataset_id="shrunk")
    (out_root / "t" / "dataset.json").rename(out_root / "t" / "dataset.json.bak")
    try:
        assert compare.main(common) == 2                            # 記録が無ければ ID が要る
        assert "--dataset-id" in capsys.readouterr().out
        assert compare.main([*common, "--dataset-id", "shrunk"]) == 2
        assert "正解にだけある 0 行" in capsys.readouterr().out     # 予測にだけあるレースがある
    finally:
        (out_root / "t" / "dataset.json.bak").rename(out_root / "t" / "dataset.json")
        (world.ml_data_dir / "datasets" / "CURRENT").write_text(manifest["dataset_id"],
                                                                 encoding="utf-8")

    # 保存したモデルを読み直して同じ予測になる
    model = PedigreeCatBoost.load(out_root / "t" / "10y")
    assert model.feature_columns
    checks = json.loads((out_root / "t" / "10y" / "checks.json").read_text(encoding="utf-8"))
    assert {c["name"]: c["status"] for c in checks} == {
        "feature_columns": "ok", "split": "ok", "asof_sample": "ok"}


def test_holdout_evaluation_uses_only_a_period_after_the_test_and_runs_once(world, dataset,
                                                                            capsys):
    pytest.importorskip("catboost")
    df, _, _ = dataset
    config = MLConfig(
        data=DataConfig(pedigree_source="ml"),      # データセットを作ったときと同じ設定
        pedigree=PedigreeConfig(generations=4, inbreeding_generations=4),
        split=SplitConfig(test_start=date(2021, 1, 1), test_end=date(2021, 12, 31),
                          valid_years=1, train_years=(5,)),
        model=ModelConfig(iterations=100, learning_rate=0.1, early_stopping_rounds=30,
                          thread_count=2))
    run_comparison(df, config, out_root=world.ml_data_dir / "runs", run_id="hold", log_period=0)
    model_dir = world.ml_data_dir / "runs" / "hold" / "5y"
    evaluate = _script("evaluate_pedigree")
    common = ["--model", str(model_dir), "--ml-data-dir", str(world.ml_data_dir)]

    assert evaluate.main([*common, "--from", "2021-06-01", "--to", "2022-12-31"]) == 2
    assert "テスト期間の終わり" in capsys.readouterr().out

    assert evaluate.main([*common, "--from", "2022-01-01", "--to", "2022-12-31"]) == 0
    result = json.loads((model_dir / "holdout_20220101_20221231.json").read_text(encoding="utf-8"))
    assert result["period"]["first_race"] >= "2022-01-01"
    hits = result["results"]["pedigree_catboost"]["ranking"]["hits"]
    assert hits["ci95"][0] > hits["random"]
    diff = result["differences"]["pedigree_catboost-sire_ae"]["hits"]
    assert diff["races_differing"] > 0 and diff["ci95"][0] <= diff["diff"] <= diff["ci95"][1]
    # 学習時と同じ未来情報の検査を通している
    assert {c["name"]: c["status"] for c in result["checks"]} == {
        "feature_columns": "ok", "asof_sample": "ok"}

    # 2回目は上書きしない（見てから評価し直すと、その期間も選択に使ったことになる）
    assert evaluate.main([*common, "--from", "2022-01-01", "--to", "2022-12-31"]) == 2
    assert "上書きしません" in capsys.readouterr().out

    # 学習時と違う設定で作られたデータセットでは評価しない
    run_config = model_dir.parent / "config.json"
    original = run_config.read_text(encoding="utf-8")
    changed = json.loads(original)
    changed["data"]["pedigree_source"] = "blood"
    run_config.write_text(json.dumps(changed), encoding="utf-8")
    try:
        assert evaluate.main([*common, "--from", "2022-02-01", "--to", "2022-12-31"]) == 2
        assert "data.pedigree_source" in capsys.readouterr().out
    finally:
        run_config.write_text(original, encoding="utf-8")


def test_compare_runs_keeps_the_trained_order():
    """--years 20 --years 10 で学習した実行の基準は 20y。名前で並べ替えると差の符号が逆になる。"""
    import tempfile
    from pathlib import Path
    compare = _script("compare_runs")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        pd.DataFrame({"window": ["20y", "20y", "10y", "10y", "5y"],
                      "model": ["pedigree_catboost", "sire_ae", "pedigree_catboost", "sire_ae",
                                "pedigree_catboost"]}).to_csv(run / "comparison.csv", index=False)
        assert compare.trained_order(run) == ["20y", "10y", "5y"]


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


@pytest.mark.parametrize("script", ["train_pedigree", "build_dataset"])
@pytest.mark.parametrize("body", ["[model]\niterations = true\n", "[model\n"])
def test_invalid_config_is_a_usage_error_not_a_traceback(tmp_path, capsys, script, body):
    """値の誤り・TOML の構文誤り（どちらも ValueError）も parser.error で止める。"""
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _script(script).main(["--config", str(path), "--ml-data-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "bad.toml" in capsys.readouterr().err


def test_predict_requires_the_training_records(tmp_path, capsys):
    predict = _script("predict_pedigree")
    model_dir = tmp_path / "runs" / "r" / "10y"
    model_dir.mkdir(parents=True)
    assert predict.main(["--model", str(model_dir), "--ml-data-dir", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "config.json" in out and "metrics.json" in out


def test_predict_drops_scratched_entries_and_recounts_field_size():
    """出馬表で取消・除外の馬は予想せず、特徴量の頭数（field_size）にも数えない。"""
    import pandas as pd
    predict = _script("predict_pedigree")
    df = pd.DataFrame({
        "race_id": ["old", "old", "live", "live", "live"],
        "umaban": ["01", "02", "01", "02", "03"],
        "has_result": [True, True, False, False, False],
        "result_ijyo_cd": ["0", "0", "0", "1", " "],
        "field_size": [2, 2, 3, 3, 3],
    })
    rows, targets = predict.live_rows(df)
    assert list(targets["umaban"]) == ["01", "03"]
    assert list(rows["field_size"]) == [2, 2, 2, 2]


def test_build_refuses_an_existing_dataset_id_before_building(world, dataset):
    build_script = _script("build_dataset")
    _, _, manifest = dataset
    with pytest.raises(SystemExit):
        build_script.main(["--race-data-dir", str(world.race_data_dir),
                           "--ml-data-dir", str(world.ml_data_dir),
                           "--dataset-id", manifest["dataset_id"]])

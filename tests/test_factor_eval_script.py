"""scripts/factor_eval.py のキャッシュの名前・作り直し・書き出し（作り物の小さな点数の表で）。Jev は呼ばない。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import factor_eval as script  # noqa: E402

from keiba.factors import table_factors as tf  # noqa: E402
from keiba.factors.registry import VERSIONS, compute_all, versions_tag  # noqa: E402
from tests.fixtures.factor_runs import make_factor_runs, to_runs_all  # noqa: E402

# 表の要素だけ v1 の版の組の印（v2 にする前の点数の表の名前に入っていた印）
V1_TABLE_TAG = versions_tag({**VERSIONS, **{name: "v1" for name in tf.TABLE_FACTORS}})


def test_cache_path_includes_data_blood_and_factor_versions(tmp_path):
    p = script.cache_path(tmp_path, "D1", "B1")
    assert p.parent == tmp_path
    assert p.name == f"factors_D1_B1_{versions_tag()}.pkl"
    assert script.cache_path(tmp_path, "D1", "B2") != p
    assert script.cache_path(tmp_path, "D1", "B1", {"jockey": "v1"}) != p


def test_git_head_is_a_commit_or_none(tmp_path):
    head = script.git_head()
    assert head is None or len(head) == 40
    assert script.git_head(tmp_path) is None            # リポジトリでない場所では None（止めない）


@pytest.fixture(scope="module")
def old_table(tmp_path_factory) -> Path:
    """表の要素を v1 で作った「古い点数の表」（年ごとの要素も入っている）。"""
    R = make_factor_runs(years=(2015, 2016, 2017))
    out = compute_all(R, to_runs_all(R), years=[2016, 2017], log=lambda *_: None)
    for name, fn in tf.TABLE_FACTORS.items():
        out[name] = fn(out, version="v1")
    path = tmp_path_factory.mktemp("old") / f"factors_D1_B1_{V1_TABLE_TAG}.pkl"
    out.to_pickle(path)
    return path


def _dirs(tmp_path: Path) -> list[str]:
    for name, cur in (("data", "D1"), ("blood", "B1")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "CURRENT").write_text(cur, encoding="utf-8")
    return ["--data-dir", str(tmp_path / "data"), "--blood-dir", str(tmp_path / "blood"),
            "--cache-dir", str(tmp_path / "cache"), "--n-boot", "20"]


def test_recompute_source_must_match_current_versions(tmp_path):
    ok = script.recompute_source_problem
    assert ok(tmp_path / f"factors_D1_B1_{V1_TABLE_TAG}.pkl", "D1", "B1") is None
    assert ok(tmp_path / f"factors_D1_B1_{versions_tag()}.pkl", "D1", "B1") is None    # 今の版の表
    assert ok(tmp_path / "factors_D1.pkl", "D1", "B1") is not None      # 古い名前は血統版を確かめられないので使わない
    assert "D0" in ok(tmp_path / f"factors_D0_B1_{V1_TABLE_TAG}.pkl", "D1", "B1")
    assert "B0" in ok(tmp_path / f"factors_D1_B0_{V1_TABLE_TAG}.pkl", "D1", "B1")
    assert ok(tmp_path / "old.pkl", "D1", "B1") is not None             # 版が読めない名前は使わない
    # 表の要素以外（引き継ぐ列）の版が今と違う表は使わない
    other = versions_tag({**VERSIONS, "time_figure": "v0"})
    assert "印" in ok(tmp_path / f"factors_D1_B1_{other}.pkl", "D1", "B1")
    assert ok(tmp_path / "factors_D1_B1_abcd1234.pkl", "D1", "B1") is not None


def test_recompute_from_other_data_version_is_refused(tmp_path, old_table, capsys):
    base = _dirs(tmp_path)
    (tmp_path / "data" / "CURRENT").write_text("D2", encoding="utf-8")
    rc = script.main(base + ["--out-dir", str(tmp_path / "out"), "--recompute-table-from", str(old_table)])
    assert rc == 2
    assert not script.cache_path(tmp_path / "cache", "D2", "B1").exists()
    assert "D2" in capsys.readouterr().out


def test_recompute_from_old_table_then_reuse_cache(tmp_path, old_table, capsys):
    base = _dirs(tmp_path)
    assert script.main(base + ["--out-dir", str(tmp_path / "out1"), "--recompute-table-from", str(old_table)]) == 0
    cache = script.cache_path(tmp_path / "cache", "D1", "B1")
    assert cache.exists() and not (tmp_path / "out1" / cache.name).exists()   # キャッシュは --out-dir に置かない
    R = pd.read_pickle(cache)
    old = pd.read_pickle(old_table)
    np.testing.assert_array_equal(R["jockey"].to_numpy(), tf.jockey(old, version="v2"))
    np.testing.assert_array_equal(R["time_figure"].to_numpy(), old["time_figure"].to_numpy())  # 年ごとの要素はそのまま
    m1 = json.loads((tmp_path / "out1" / "manifest.json").read_text(encoding="utf-8"))
    assert m1["cache"] == str(cache) and m1["cache_reused"] is False and m1["cache_source"] == str(old_table)
    assert "git_head" in m1 and m1["factors"]["jockey"] == "v2" and m1["factors"]["age"] == "v1"
    assert set(m1["excluded"]["by_reason"]) == {"dead_heat", "bad_odds", "small_field"}
    for f in ("report.md", "summary.csv", "by_year.csv", "summary_mixed.csv", "by_year_mixed.csv", "lambda.csv"):
        assert (tmp_path / "out1" / f).exists(), f
    assert not (tmp_path / "out1" / "summary_op.csv").exists()     # 作り物に OP のレースは無い
    by = pd.read_csv(tmp_path / "out1" / "by_year.csv")
    assert {"measured", "clipped_share", "base_nll"} <= set(by.columns)

    assert script.main(base + ["--out-dir", str(tmp_path / "out2"), "--compare-to", str(tmp_path / "out1")]) == 0
    m2 = json.loads((tmp_path / "out2" / "manifest.json").read_text(encoding="utf-8"))
    assert m2["cache_reused"] is True and m2["cache_source"] is None
    text = (tmp_path / "out2" / "report.md").read_text(encoding="utf-8")
    assert "## この版（v2）について" in text and "## 前の実行との判定の比較" in text
    assert "ブートストラップ（20回）" in text                       # --n-boot の回数をそのまま書く
    capsys.readouterr()


def test_recompute_takes_precedence_over_existing_cache(tmp_path, old_table, capsys):
    """--recompute-table-from を付けたら、同じ名前のキャッシュがあっても作り直して置き換える。"""
    base = _dirs(tmp_path)
    cache = script.cache_path(tmp_path / "cache", "D1", "B1")
    cache.parent.mkdir(parents=True)
    pd.read_pickle(old_table).assign(jockey=123.0).to_pickle(cache)          # 古い中身のキャッシュ
    assert script.main(base + ["--out-dir", str(tmp_path / "out"), "--recompute-table-from", str(old_table)]) == 0
    R = pd.read_pickle(cache)
    np.testing.assert_array_equal(R["jockey"].to_numpy(), tf.jockey(pd.read_pickle(old_table), version="v2"))
    m = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert m["cache_reused"] is False and m["cache_source"] == str(old_table)
    assert not list(cache.parent.glob("*.tmp"))
    capsys.readouterr()


def test_cache_missing_an_evaluation_year_is_refused(tmp_path, old_table, capsys):
    """キャッシュの名前に評価の年は入らないので、走があるのに年ごとの要素が無い年があれば止める。"""
    base = _dirs(tmp_path)
    cache = script.cache_path(tmp_path / "cache", "D1", "B1")
    cache.parent.mkdir(parents=True)
    R = pd.read_pickle(old_table)
    R.loc[R.year == 2017, list(script.YEARLY_FACTOR_NAMES)] = np.nan       # 2017年を計算していない表
    R.to_pickle(cache)
    assert script.main(base + ["--out-dir", str(tmp_path / "out")]) == 2
    assert "2017" in capsys.readouterr().out and not (tmp_path / "out").exists()
    assert script.missing_years(pd.read_pickle(old_table), [2016, 2017, 2030]) == []   # 走の無い年は数えない


def test_cache_is_written_atomically(tmp_path, old_table, monkeypatch):
    """書き込みが途中で失敗しても、キャッシュの名前のファイルは残らない（次の実行が壊れた表を読まない）。"""
    R = pd.read_pickle(old_table)
    target = tmp_path / "cache" / "factors_D1_B1_abcd1234.pkl"

    def broken(self, path, *a, **k):
        Path(path).write_bytes(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_pickle", broken)
    with pytest.raises(OSError):
        script.save_atomically(R, target)
    assert not target.exists() and not list(target.parent.iterdir())
    monkeypatch.undo()
    script.save_atomically(R, target)
    assert len(pd.read_pickle(target)) == len(R) and [p.name for p in target.parent.iterdir()] == [target.name]

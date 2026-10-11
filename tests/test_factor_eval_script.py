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
from keiba.factors.registry import compute_all, versions_tag  # noqa: E402
from tests.fixtures.factor_runs import make_factor_runs, to_runs_all  # noqa: E402


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
    path = tmp_path_factory.mktemp("old") / "factors_old.pkl"
    out.to_pickle(path)
    return path


def _dirs(tmp_path: Path) -> list[str]:
    for name, cur in (("data", "D1"), ("blood", "B1")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "CURRENT").write_text(cur, encoding="utf-8")
    return ["--data-dir", str(tmp_path / "data"), "--blood-dir", str(tmp_path / "blood"),
            "--cache-dir", str(tmp_path / "cache"), "--n-boot", "20"]


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
    capsys.readouterr()

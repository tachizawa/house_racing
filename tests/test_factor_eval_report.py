"""report.md の組み立てのテスト。"""

from __future__ import annotations

from keiba.evaluate.factor_eval import by_year, evaluate, summarize
from keiba.evaluate.factor_eval_report import render
from tests.test_factor_eval import synth_eval


def _render():
    R = synth_eval(n_per_year=40)
    res = evaluate(R, ["good", "noise"], log=lambda *_: None)
    pr = res["per_race"]
    s = summarize(pr, ["good", "noise"], n_boot=50)
    subsets = {"OP・重賞": summarize(pr[pr.op], ["good", "noise"], n_boot=50),
               "3歳以上の古馬混合戦": summarize(pr[pr.mixed], ["good", "noise"], n_boot=50)}
    return render(s, by_year(pr, ["good", "noise"]), subsets, res["excluded"], data_version="TEST",
                  years=list(range(2017, 2026)))


def test_report_sections():
    text = _render()
    for h in ("# 要素別の予想 — 過去での評価", "## 判定（市場への上乗せ）", "## 根拠: 的中", "## 根拠: 食い違い",
              "## 年ごと", "## 内訳: OP・重賞", "## 内訳: 3歳以上の古馬混合戦", "## 既知の弱点"):
        assert h in text


def test_report_shows_excluded_counts():
    text = _render()
    assert "評価から除いたレース" in text

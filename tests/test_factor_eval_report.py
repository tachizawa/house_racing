"""report.md の組み立てのテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.evaluate.factor_eval import by_year, evaluate, summarize
from keiba.evaluate.factor_eval_report import MEANING, render
from tests.test_factor_eval import _late_start, synth_eval

FACTORS = ["good", "noise", "late_start"]


def _parts():
    R = synth_eval(n_per_year=40)
    R = R.assign(late_start=_late_start(R))
    res = evaluate(R, FACTORS, log=lambda *_: None)
    pr = res["per_race"]
    s = summarize(pr, FACTORS, n_boot=50)
    subsets = {"OP・重賞": summarize(pr[pr.op], FACTORS, n_boot=50),
               "3歳以上の古馬混合戦": summarize(pr[pr.mixed], FACTORS, n_boot=50)}
    return s, by_year(pr, FACTORS), subsets, res["excluded"]


def _render(**kw):
    s, t, subsets, ex = _parts()
    return render(s, t, subsets, ex, data_version="TEST", years=list(range(2017, 2026)), **kw)


def _section(text: str, head: str) -> str:
    start = text.index(head)
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt < 0 else text[start:nxt]


def test_report_sections():
    text = _render()
    for h in ("# 要素別の予想 — 過去での評価", "## 判定（市場への上乗せ）", "## 根拠: 的中", "## 根拠: 食い違い",
              "## 年ごと", "## 内訳: OP・重賞", "## 内訳: 3歳以上の古馬混合戦", "## 既知の弱点"):
        assert h in text


def test_report_shows_excluded_counts_by_reason():
    text = _render()
    assert "評価から除いたレース" in text
    for label in ("1〜3着が一意でない（同着など）", "単勝オッズが無い・0以下の馬がいる", "4頭未満"):
        assert label in text


def test_unmeasured_years_print_dash_and_measured_count():
    """測れなかった年は年ごとの表で「—」、改善した年は「k/m（測れた年 m、全9年）」（final-findings #2）。"""
    text = _render()
    yearly = _section(text, "## 年ごと（Δ ×1000）")
    row2017 = next(line for line in yearly.splitlines() if line.startswith("| 2017 |"))
    cells = [c.strip() for c in row2017.strip("|").split("|")]
    assert cells[-1] == "—"                                   # late_start は 2017年を測れない
    assert "+0.00" not in row2017.split("|")[-2]
    verdict = _section(text, "## 判定（市場への上乗せ）")
    late = next(line for line in verdict.splitlines() if line.startswith("| late_start |"))
    assert "（測れた年 8、全9年）" in late
    good = next(line for line in verdict.splitlines() if line.startswith("| good |"))
    assert "/9 |" in good and "測れた年" not in good
    weak = _section(text, "## 既知の弱点")
    assert "測れなかった年がある要素: late_start（2017年）" in weak


def test_report_states_magnitude_meaning_and_bonferroni():
    """市場だけのモデルの −対数尤度・Δ の割合・意味の一文・的中に区間が無いこと・Bonferroni の列（#5）。"""
    text = _render()
    verdict = _section(text, "## 判定（市場への上乗せ）")
    assert "市場だけのモデルの −対数尤度は1レースあたり平均" in verdict
    assert "Bonferroni 15比較" in verdict and "99.67%" in verdict
    assert MEANING in text
    assert "市場に勝てる・的中が増えるという意味ではない" in MEANING
    assert "区間を付けていない" in _section(text, "## 根拠: 的中")
    good = next(line for line in verdict.splitlines() if line.startswith("| good |"))
    assert good.count("[") == 2 and "%" in good


def test_report_describes_z_clip_and_clip_table():
    text = _render()
    weak = _section(text, "## 既知の弱点")
    assert "±4 で切る" in weak and "1e-09 未満なら 0 とみなし" in weak
    assert "## 年ごと: z が ±4 で切られた評価の行の割合" in text


def test_subsets_have_hits_and_disagreement_tables():
    """内訳ごとに Δ の表・的中・食い違いを出す（仕様 §5.4、final-findings #4）。"""
    text = _render()
    for name in ("OP・重賞", "3歳以上の古馬混合戦"):
        sec = text[text.index(f"## 内訳: {name}"):]
        sec = sec[:sec.index("\n## ", 1)]
        assert f"### {name}: 的中" in sec and f"### {name}: 食い違い" in sec
        assert "判定 |" not in sec


def test_version_note_and_previous_comparison():
    lambdas = pd.DataFrame({"factor": ["jockey", "jockey"], "year": [2016, 2017],
                            "lambda_v1": [np.nan, 4760.0], "lambda_v2": [1383.0, 1358.0]})
    s, *_ = _parts()
    prev = s.copy()
    prev.loc["noise", "verdict"] = "上乗せあり"
    text = _render(lambdas=lambdas, previous=prev, previous_label="v1")
    note = _section(text, "## この版（v2）について")
    assert "4,760" in note and "1,358" in note and "| 騎手 | — | 1,383 |" in note
    assert "事後の判定" in note
    cmp_ = _section(text, "## 前の実行との判定の比較（v1）")
    assert "←変わった" in next(line for line in cmp_.splitlines() if line.startswith("| noise |"))

"""要素別の予想の過去での評価の report.md（仕様 §5.5）。計算はしない。"""

from __future__ import annotations

import numpy as np
import pandas as pd

JA = {"time_figure": "タイム指数", "opponents": "相手関係", "jockey": "騎手", "trainer": "調教師",
      "sire_surface": "父の芝ダ適性", "late_age": "晩成度×年齢", "age": "年齢", "cohort_level": "世代レベル",
      "distance_fit": "距離適性", "course_fit": "コース適性", "draw": "枠順", "weight_carried": "斤量",
      "layoff": "間隔", "body_weight": "馬体重", "travel": "遠征"}


def _ja(f: str) -> str:
    return JA.get(f, f)


def _x1000(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{1000 * v:+.2f}"


def _pct(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{100 * v:.1f}%"


def _verdict_table(s: pd.DataFrame, with_verdict: bool) -> list[str]:
    head = "| 要素 | Δ ×1000 [95%] | 改善した年 |" + (" 判定 |" if with_verdict else "")
    out = [head, "|---|---|---|" + ("---|" if with_verdict else "")]
    for f, r in s.sort_values("delta", ascending=False).iterrows():
        row = f"| {_ja(f)} | {_x1000(r.delta)} [{_x1000(r.lo)}, {_x1000(r.hi)}] | {int(r.years_improved)}/9 |"
        out.append(row + (f" **{r.verdict}** |" if with_verdict else ""))
    return out


def render(summary: pd.DataFrame, yearly: pd.DataFrame, subsets: dict, excluded: dict, *, data_version: str,
           years: list[int]) -> str:
    L = [f"# 要素別の予想 — 過去での評価（{years[0]}〜{years[-1]}年）", "",
         f"データ版 `{data_version}`。仕様 docs/superpowers/specs/2026-10-11-factor-predictions-design.md §5。Jev は使っていない。",
         f"評価年のレース {excluded['races']:,}、うち評価に使ったレース {excluded['valid']:,}。"
         f"評価から除いたレース {excluded['excluded']:,}（同着で1〜3着が一意でない・単勝オッズが無い・4頭未満）。", "",
         "## 判定（市場への上乗せ）", "",
         "Δ = 市場だけの着順モデルの −対数尤度（1〜3着の順番）− 市場＋要素の −対数尤度。**正なら要素で良くなった**。"
         "重みは評価年より前の年だけで推定。区間は開催日単位のブートストラップ（1,000回）。",
         "判定: 区間の下限 > 0 かつ 9年中7年以上で改善 → 上乗せあり。", ""]
    L += _verdict_table(summary, True)
    L += ["", "## 根拠: 的中（1レースあたりの3着内の頭数、上位3頭を選んだとき）", "",
          "| 要素 | 要素だけ | 市場（人気上位3頭） | 市場＋要素 | 市場だけのモデル |", "|---|---|---|---|---|"]
    for f, r in summary.iterrows():
        L.append(f"| {_ja(f)} | {r.solo:.3f} | {r.mkt_odds:.3f} | {r.comb:.3f} | {r.mkt_model:.3f} |")
    L += ["", "## 根拠: 食い違い（要素と市場の上位3頭が違った馬）", "",
          "「要素だけ」= 要素の上位3頭にいて人気上位3頭にいない馬、「市場だけ」= その逆。見込み = 人気から見込む3着内率。", "",
          "| 要素 | 要素だけ: 頭数 | 実際 | 見込み | 市場だけ: 頭数 | 実際 | 見込み |", "|---|---|---|---|---|---|---|"]
    for f, r in summary.iterrows():
        L.append(f"| {_ja(f)} | {r.fo_n:,.0f} | {_pct(r.fo_rate)} | {_pct(r.fo_e3)} | {r.mo_n:,.0f} | "
                 f"{_pct(r.mo_rate)} | {_pct(r.mo_e3)} |")
    L += ["", "## 年ごと（Δ ×1000）", "", "| 年 | " + " | ".join(_ja(f) for f in summary.index) + " |",
          "|---|" + "---|" * len(summary)]
    for y in sorted(set(yearly.index.get_level_values("year"))):
        L.append(f"| {y} | " + " | ".join(_x1000(yearly.loc[(y, f), "delta"]) for f in summary.index) + " |")
    for name, s in subsets.items():
        L += ["", f"## 内訳: {name}（参考。判定は付けない）", ""] + _verdict_table(s, False)
    L += ["", "## 既知の弱点", "",
          "- 過去の評価は確定オッズ（締切時点）。これからの記録は朝のオッズなので、市場の基準が弱くなる",
          "- 15要素を同時に調べているので、偶然の「上乗せあり」が混じりうる（年ごとの安定性の条件で和らげている）",
          "- 晩成度×年齢は 2016〜2017年に材料が無く、その年はレース平均（上乗せ 0）として扱われる",
          "- 小さな上乗せ（Δ ×1000 が 1 未満）は区間で 0 と区別できないことがある",
          "- 単独の的中・食い違いでは、点数のある馬が3頭未満のレース（新馬戦など、過去走を使う要素）では"
          "残りの枠が点数の無い馬に等分される（要素に不利な向きの薄め）", ""]
    return "\n".join(L)

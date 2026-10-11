"""要素別の予想の過去での評価の report.md（仕様 §5.5）。計算はしない。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.evaluate.factor_eval import ALPHA, MIN_YEARS_IMPROVED, N_COMPARE, Z_CLIP, ZERO_TOL

JA = {"time_figure": "タイム指数", "opponents": "相手関係", "jockey": "騎手", "trainer": "調教師",
      "sire_surface": "父の芝ダ適性", "late_age": "晩成度×年齢", "age": "年齢", "cohort_level": "世代レベル",
      "distance_fit": "距離適性", "course_fit": "コース適性", "draw": "枠順", "weight_carried": "斤量",
      "layoff": "間隔", "body_weight": "馬体重", "travel": "遠征"}
REASONS = {"dead_heat": "1〜3着が一意でない（同着など）", "bad_odds": "単勝オッズが無い・0以下の馬がいる",
           "small_field": "4頭未満"}
MEANING = ("「上乗せあり」＝市場だけのモデルに情報を足せる（統計的に検出できる）という意味で、"
           "市場に勝てる・的中が増えるという意味ではない")


def _ja(f: str) -> str:
    return JA.get(f, f)


def _x1000(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{1000 * v:+.2f}"


def _pct(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{100 * v:.1f}%"


def _years(r) -> str:
    """改善した年。全年を測れたら「k/9」、測れなかった年があれば「k/m（測れた年 m、全9年）」。"""
    k, m, total = int(r.years_improved), int(r.years_measured), int(r.years_total)
    return f"{k}/{total}" if m == total else f"{k}/{m}（測れた年 {m}、全{total}年）"


def _verdict_table(s: pd.DataFrame, with_verdict: bool) -> list[str]:
    conf = 100 * (1 - ALPHA / N_COMPARE)
    head = (f"| 要素 | Δ ×1000 [95%] | Δ / 市場の −対数尤度 | 参考: [{conf:.2f}%]（Bonferroni {N_COMPARE}比較） | 改善した年 |"
            + (" 判定 |" if with_verdict else ""))
    out = [head, "|---|---|---|---|---|" + ("---|" if with_verdict else "")]
    for f, r in s.sort_values("delta", ascending=False).iterrows():
        row = (f"| {_ja(f)} | {_x1000(r.delta)} [{_x1000(r.lo)}, {_x1000(r.hi)}] | {100 * r.delta_pct:+.3f}% | "
               f"[{_x1000(r.lo_bonf)}, {_x1000(r.hi_bonf)}] | {_years(r)} |")
        out.append(row + (f" **{r.verdict}** |" if with_verdict else ""))
    return out


def _hits_table(s: pd.DataFrame) -> list[str]:
    out = ["| 要素 | 要素だけ | 市場（人気上位3頭） | 市場＋要素 | 市場だけのモデル |", "|---|---|---|---|---|"]
    for f, r in s.iterrows():
        out.append(f"| {_ja(f)} | {r.solo:.3f} | {r.mkt_odds:.3f} | {r.comb:.3f} | {r.mkt_model:.3f} |")
    return out


def _disagree_table(s: pd.DataFrame) -> list[str]:
    out = ["| 要素 | 要素だけ: 頭数 | 実際 | 見込み | 市場だけ: 頭数 | 実際 | 見込み |", "|---|---|---|---|---|---|---|"]
    for f, r in s.iterrows():
        out.append(f"| {_ja(f)} | {r.fo_n:,.0f} | {_pct(r.fo_rate)} | {_pct(r.fo_e3)} | {r.mo_n:,.0f} | "
                   f"{_pct(r.mo_rate)} | {_pct(r.mo_e3)} |")
    return out


def _excluded_lines(excluded: dict) -> list[str]:
    line = (f"評価年のレース {excluded['races']:,}、うち評価に使ったレース {excluded['valid']:,}。"
            f"評価から除いたレース {excluded['excluded']:,}")
    by = excluded.get("by_reason")
    if by:
        line += "（理由別、重なりあり: " + "、".join(f"{REASONS.get(k, k)} {v:,}" for k, v in by.items()) + "）"
    return [line + "。"]


def _version_section(lambdas: pd.DataFrame) -> list[str]:
    table = list(lambdas["factor"].unique())
    L = ["## この版（v2）について", "",
         "v1 の実行（`data/factor_eval/20261011T140120/`）の後のレビューで、表の要素"
         f"（{'・'.join(_ja(n) for n in table)}）の縮小の強さ λ = σ²/τ² の推定が偏っていると分かった。"
         "A−E の散らばりは組ごとに大きく違う（人気馬に多く乗る大きな組ほど大きい）のに、v1 は組の中の散らばりを"
         "1つにまとめて組の数だけ引いていたので τ² が小さく出て、λ が大きすぎた（下の表。騎手 2017年 v1 "
         + _lam(lambdas, "jockey", 2017, "v1") + " → v2 " + _lam(lambdas, "jockey", 2017, "v2") + "）。"
         "推定できない年（τ² ≤ 0）はその年の点数が全部 0 になり（2016年の騎手・調教師・父の芝ダ）、"
         "年ごとに物差しがずれて評価の年の z が ±4 で切られる行が多かった。v2 は組ごとの散らばりを引く"
         "モーメント法で λ を推定する（`keiba.factors.cumulative.eb_variance_components`）。ほかの要素は v1 のまま。"
         "**v2 の判定は v1 の結果を見た後に推定を直したものなので、事後の判定として扱い、"
         "これからのレースでの記録（仕様 §6）で確かめる必要がある。** v1 の実行は記録として残している。", "",
         "λ（年 Y の点数に使う値。Y の前年末までで推定。— は推定できず点数が全部 0）:", "",
         "| 要素 | " + " | ".join(f"{y} v1 | {y} v2" for y in sorted(lambdas["year"].unique())) + " |",
         "|---|" + "---|---|" * lambdas["year"].nunique()]
    for name in table:
        cells = []
        for y in sorted(lambdas["year"].unique()):
            cells += [_lam(lambdas, name, y, "v1"), _lam(lambdas, name, y, "v2")]
        L.append(f"| {_ja(name)} | " + " | ".join(cells) + " |")
    return L + [""]


def _lam(lambdas: pd.DataFrame, name: str, year: int, ver: str) -> str:
    r = lambdas[(lambdas["factor"] == name) & (lambdas["year"] == year)]
    if r.empty or np.isnan(r.iloc[0][f"lambda_{ver}"]):
        return "—"
    return f"{r.iloc[0][f'lambda_{ver}']:,.0f}"


def _previous_section(summary: pd.DataFrame, previous: pd.DataFrame, label: str) -> list[str]:
    L = [f"## 前の実行との判定の比較（{label}）", "", "| 要素 | 前: Δ ×1000 [95%] | 前: 判定 | 今: Δ ×1000 [95%] | 今: 判定 |",
         "|---|---|---|---|---|"]
    for f, r in summary.sort_values("delta", ascending=False).iterrows():
        if f in previous.index:
            p = previous.loc[f]
            prev = f"{_x1000(p['delta'])} [{_x1000(p['lo'])}, {_x1000(p['hi'])}] | {p['verdict']}"
        else:
            prev = "— | —"
        mark = " ←変わった" if f in previous.index and previous.loc[f, "verdict"] != r.verdict else ""
        L.append(f"| {_ja(f)} | {prev} | {_x1000(r.delta)} [{_x1000(r.lo)}, {_x1000(r.hi)}] | {r.verdict}{mark} |")
    return L + [""]


def render(summary: pd.DataFrame, yearly: pd.DataFrame, subsets: dict, excluded: dict, *, data_version: str,
           years: list[int], lambdas: pd.DataFrame | None = None, versions: dict[str, str] | None = None,
           previous: pd.DataFrame | None = None, previous_label: str = "") -> str:
    """report.md の本文。

    lambdas（表の要素の λ の v1・v2 の比較）を渡すと「この版（v2）について」の節を、previous（前の実行の
    summary.csv）を渡すと判定の比較の節を足す。
    """
    total = len(years)
    base = float(summary["base_nll"].iloc[0])
    L = [f"# 要素別の予想 — 過去での評価（{years[0]}〜{years[-1]}年）", "",
         f"データ版 `{data_version}`。仕様 docs/superpowers/specs/2026-10-11-factor-predictions-design.md §5。Jev は使っていない。"]
    if versions:
        L.append("要素の版: " + "、".join(f"{_ja(k)} {v}" for k, v in versions.items()) + "。")
    L += _excluded_lines(excluded) + [""]
    if lambdas is not None:
        L += _version_section(lambdas)
    if previous is not None:
        L += _previous_section(summary, previous, previous_label)
    L += ["## 判定（市場への上乗せ）", "",
          "Δ = 市場だけの着順モデルの −対数尤度（1〜3着の順番）− 市場＋要素の −対数尤度。**正なら要素で良くなった**。"
          "重みは評価年より前の年だけで推定。区間は開催日単位のブートストラップ（1,000回）。",
          f"市場だけのモデルの −対数尤度は1レースあたり平均 **{base:.4f}**（{total}年まとめ）。"
          "「Δ / 市場の −対数尤度」はそれに対する Δ の割合で、上乗せの大きさの目安。",
          f"判定: 95% 区間の下限 > 0 かつ {total}年中{MIN_YEARS_IMPROVED}年以上で改善 → 上乗せあり（仕様 §5.2）。"
          "Bonferroni の区間は15要素を同時に調べていることの参考で、判定には使っていない。",
          "改善した年は、測れた年（学習・評価の年の点数がレース内で差を持つ年）のうち Δ > 0 の年の数。"
          f"測れなかった年がある要素は「k/m（測れた年 m、全{total}年）」と書く。測れなかった年は改善に数えない。",
          MEANING + "。", ""]
    L += _verdict_table(summary, True)
    L += ["", "## 根拠: 的中（1レースあたりの3着内の頭数、上位3頭を選んだとき）", "",
          "的中の列には区間を付けていない（差がごく小さく、判定には使わない）。", ""]
    L += _hits_table(summary)
    L += ["", "## 根拠: 食い違い（要素と市場の上位3頭が違った馬）", "",
          "「要素だけ」= 要素の上位3頭にいて人気上位3頭にいない馬、「市場だけ」= その逆。見込み = 人気から見込む3着内率。", ""]
    L += _disagree_table(summary)
    ys = sorted(set(yearly.index.get_level_values("year")))
    L += ["", "## 年ごと（Δ ×1000）", "", "— は測れなかった年（学習か評価の年の点数がレース内で全頭同じ・欠け）。", "",
          "| 年 | 市場の −対数尤度 | " + " | ".join(_ja(f) for f in summary.index) + " |",
          "|---|---|" + "---|" * len(summary)]
    for y in ys:
        cells = [(_x1000(yearly.loc[(y, f), "delta"]) if bool(yearly.loc[(y, f), "measured"]) else "—")
                 for f in summary.index]
        base_y = yearly.xs(y, level="year")["base_nll"].iloc[0]
        L.append(f"| {y} | {base_y:.4f} | " + " | ".join(cells) + " |")
    clipped = [f for f in summary.index if yearly.xs(f, level="factor")["clipped_share"].max() > 0]
    L += ["", f"## 年ごと: z が ±{Z_CLIP:g} で切られた評価の行の割合", "",
          "割合が大きい年の Δ は、要素の力より学習の年と評価の年の物差しのずれを映しうる。"
          + ("ここに無い要素はどの年も 0。" if 0 < len(clipped) < len(summary) else ""), ""]
    if clipped:
        L += ["| 年 | " + " | ".join(_ja(f) for f in clipped) + " |", "|---|" + "---|" * len(clipped)]
        for y in ys:
            L.append(f"| {y} | " + " | ".join(_pct(yearly.loc[(y, f), "clipped_share"]) for f in clipped) + " |")
    else:
        L.append("どの要素・どの年も 0。")
    for name, s in subsets.items():
        L += ["", f"## 内訳: {name}（参考。判定は付けない）", "",
              f"市場だけのモデルの −対数尤度は1レースあたり平均 {float(s['base_nll'].iloc[0]):.4f}。", ""]
        L += _verdict_table(s, False)
        L += ["", f"### {name}: 的中（区間なし）", ""] + _hits_table(s)
        L += ["", f"### {name}: 食い違い", ""] + _disagree_table(s)
    unmeasured = []
    for f in summary.index:
        miss = [str(y) for y in ys if not bool(yearly.loc[(y, f), "measured"])]
        if miss:
            unmeasured.append(f"{_ja(f)}（{'・'.join(miss)}年）")
    L += ["", "## 既知の弱点", "",
          "- 過去の評価は確定オッズ（締切時点）。これからの記録は朝のオッズなので、市場の基準が弱くなる",
          "- 15要素を同時に調べているので、偶然の「上乗せあり」が混じりうる（年ごとの安定性の条件で和らげている。"
          "Bonferroni の区間は参考）",
          f"- 要素の z は仕様 §4.2 から2点ずらしている: レース内で中心化した値の絶対値が {ZERO_TOL:g} 未満なら 0 とみなし"
          f"（丸めの残りで割る破綻を防ぐ）、z を ±{Z_CLIP:g} で切る（学習の年より評価の年の散らばりが大きい要素の破綻を防ぐ）。"
          "切られた割合は上の表と by_year.csv の clipped_share",
          "- 測れなかった年がある要素: " + ("、".join(unmeasured) if unmeasured else "なし")
          + "。その年は Δ = 0 として9年まとめの平均に入り、改善した年には数えない",
          "- 小さな上乗せ（Δ ×1000 が 1 未満）は区間で 0 と区別できないことがある",
          "- 単独の的中・食い違いでは、点数のある馬が3頭未満のレース（新馬戦など、過去走を使う要素）では"
          "残りの枠が点数の無い馬に等分される（要素に不利な向きの薄め）", ""]
    return "\n".join(L)

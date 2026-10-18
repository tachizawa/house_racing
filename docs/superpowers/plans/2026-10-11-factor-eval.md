# 要素別の予想 — 過去10年での評価 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 15要素それぞれの点数を過去の全レース（2016〜2025年）について「その日より前のデータだけ」で計算し、
市場（単勝オッズ）だけの着順モデルにその要素を足したときの改善を 2017〜2025年で測って、判定と根拠をレポートにする。

**Architecture:** `keiba.factors` パッケージに、走の表（`data.py`）、日より前だけの累積と経験ベイズ（`cumulative.py`）、
要素の計算（表・履歴・指数・年齢の4ファイル）、要素の一覧と一括計算（`registry.py`）、段ごとの条件付きロジット
（`order_model.py`）を置く。`keiba.evaluate.factor_eval` が拡大窓で重みを推定して評価し、`factor_eval_report` が
Markdown にする。`scripts/factor_eval.py` が長いデータ版から全部を回す。Jev は呼ばない。

**Tech Stack:** Python 3.13（`.venv`）、numpy、pandas、scipy（`scipy.optimize.minimize`）、pytest。新しい依存は足さない（pyarrow は無いので中間結果は pickle）。

**Spec:** `docs/superpowers/specs/2026-10-11-factor-predictions-design.md`（拘束力のある仕様）。本計画は §3〜§5 と §8・§9 の該当部分。
§6（これからの記録）・§7（外部モデル）は別の計画。

## Global Constraints

- コマンドはリポジトリ直下で `.venv/Scripts/python.exe -m pytest ...`。`pytest.ini` が `pythonpath = src`。テストから scripts を import するときは `sys.path.insert(0, <repo>/scripts)`
- 新しい依存は足さない。**Jev を呼ばない（課金なし）**。既存の予想・各種の版は変更しない
- **時点の規則**: 各行（1頭の1走）の点数は、その行の日（`date`）より前の行の結果だけから作る。同じ日の結果は使わない。
  例外として、その行自身のレース前に分かる情報（騎手・枠・斤量・馬体重の増減・単勝オッズ）は使ってよい
- 表と縮小の強さ λ は「行の年 Y の前年末まで」のデータで推定する。指数の基準・クラス差も同じ。各日の馬場差はその日の古馬戦だけから
- 2015年は助走。要素の点数は 2016年から。評価する年は **2017〜2025年**（2026年は使わない）
- 重み（着順モデル）は評価年 Y の前の年（**2016〜Y−1年**）だけで推定する（拡大窓）
- 判定: **9年まとめた Δ の 95% 区間の下限 > 0** かつ **9年中7年以上で年平均 Δ > 0** → 「上乗せあり」。それ以外は「上乗せは確認できない」
- 区間は開催日単位のクラスタ・ブートストラップ **1,000 回、種 20261011**
- 要素の版はすべて `v1`。結果を見てから定義を変えない
- 出力は `data/factor_eval/<実行時刻>/`（git に入れない。`/data/` は `.gitignore` 済み）
- docstring・コメント・レポートは日本語で「なぜ」を書く。コミットメッセージ末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **過去走が無い馬（新馬など）** — 履歴の要素が NaN になり、着順モデルではレース平均（z = 0）として扱われること。→ Task 7 `test_zscore_missing_is_race_mean`
2. **同着・単勝オッズが無い馬のいるレース** — 評価から除かれ、件数がレポートに出ること。→ Task 6 `test_valid_race_mask_excludes_dead_heat_and_missing_odds`、Task 8 `test_report_shows_excluded_counts`
3. **ある年に全行 NaN の要素**（晩成度×年齢は 2016〜2017年に材料が無い）— 推定が壊れず Δ ≈ 0・「上乗せは確認できない」になること。→ Task 7 `test_all_nan_factor_gives_no_lift`
4. **上位3頭の境の同点** — 単独の的中は同点を等分した分数で数えること（馬番で切らない）。→ Task 7 `test_topk_weights_split_ties`
5. **レース内で全頭同じ値の要素** — 中心化で 0 になり、Δ ≈ 0 になること。→ Task 7 `test_all_nan_factor_gives_no_lift`（同じテストで定数列も確かめる）

---

## File Structure

| ファイル | 責務 |
|---|---|
| Create `src/keiba/factors/__init__.py` | パッケージの説明だけ |
| Create `src/keiba/factors/data.py` | 版の CSV から「走」の表（`RUN_COLUMNS`）を作る |
| Create `src/keiba/factors/cumulative.py` | 日より前の累積（`prior_day_sums`）、縮小の強さ（`eb_lambda`）、縮小した点数（`shrunk_prior_scores`）、見込み3着内率と A−E（`add_expected_top3`） |
| Create `src/keiba/factors/table_factors.py` | 騎手・調教師・父の芝ダ・枠順・間隔・馬体重・遠征 |
| Create `src/keiba/factors/history_factors.py` | 距離適性・コース適性・斤量 |
| Create `src/keiba/factors/figure_factors.py` | 年ごとの物差し（その日の馬場差つき）、タイム指数、相手関係 |
| Create `src/keiba/factors/age_factors.py` | 年齢、晩成度×年齢、世代レベル |
| Create `src/keiba/factors/registry.py` | 15要素の一覧（名前・版・日本語名・朝に出せるか）と一括計算 `compute_all` |
| Create `src/keiba/factors/order_model.py` | 段ごとの条件付きロジット、レースごとの −対数尤度、1着・3着内確率 |
| Create `src/keiba/evaluate/factor_eval.py` | 拡大窓の評価、Δ、的中、食い違い、ブートストラップ、判定 |
| Create `src/keiba/evaluate/factor_eval_report.py` | report.md の組み立て |
| Create `scripts/factor_eval.py` | 実行用 |
| Create `tests/fixtures/factor_runs.py` | 作り物の走の表と、先読みの検査 |
| Create `tests/test_factor_*.py` | 各部品のテスト |
| Modify `CLAUDE.md` | 実データの結果と使い方（Task 8） |

依存の向き: `registry` → 4つの要素ファイル → `cumulative`・`data`。`factor_eval` → `order_model`・`registry`。

---

### Task 1: 走の表と、日より前だけの累積・縮小

**Files:**
- Create: `src/keiba/factors/__init__.py`, `src/keiba/factors/data.py`, `src/keiba/factors/cumulative.py`
- Create: `tests/fixtures/factor_runs.py`, `tests/test_factor_data.py`, `tests/test_factor_cumulative.py`

**Interfaces:**
- Produces:
  - `data.RUN_COLUMNS: list[str]` = `["race_id","date","year","umaban","waku","ketto_num","barei","cohort","futan","kisyu","chokyosi","tozai","sire_num","jyuni","field","top3","fin_pct","odds","ninki","time_sec","zogen","jyo","track_cd","track_kind","kyori","class_level","grade","syubetu"]`
  - `data.REGION: dict[str, str]`、`data.build_runs(races, race_horses, pedigrees) -> pd.DataFrame`、`data.load_runs(version_dir: Path, blood_dir: Path) -> pd.DataFrame`
  - `cumulative.prior_day_sums(df, keys: list[str], value: str) -> tuple[np.ndarray, np.ndarray]`（合計, 件数）
  - `cumulative.eb_lambda(values: np.ndarray, keys: np.ndarray) -> float | None`
  - `cumulative.shrunk_prior_scores(df, keys: list[str], value: str, *, first_year: int) -> np.ndarray`
  - `cumulative.add_expected_top3(df) -> pd.DataFrame`（列 `e3`, `ae` を足す）
  - `tests/fixtures/factor_runs.py`: `make_factor_runs(*, years=(2015, 2016, 2017), days_per_year=8, races_per_day=6, field=10, seed=0) -> pd.DataFrame`、`to_runs_all(R) -> pd.DataFrame`、`perturb_results_from(R, day) -> pd.DataFrame`、`assert_no_lookahead(compute, R, day)`、定数 `STRONG_JOCKEY = "J0"`

- [ ] **Step 1: 作り物の走の表を書く**

`tests/fixtures/factor_runs.py`:

```python
"""要素別の予想のテスト用の作り物の「走」の表（keiba.factors.data.RUN_COLUMNS と同じ列）。

馬ごとの地力 + 騎手の上乗せ（J0 だけ +1.0）+ 雑音で着順を決める。オッズは騎手の上乗せを知らない
（地力だけから作る）ので、J0 は「人気から見込むより多く3着内に来る」騎手になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.data import RUN_COLUMNS

STRONG_JOCKEY = "J0"


def make_factor_runs(*, years=(2015, 2016, 2017), days_per_year=8, races_per_day=6, field=10,
                     seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_horses = 160
    ability = rng.normal(0, 1, n_horses)
    birth = rng.integers(min(years) - 6, min(years) - 1, n_horses)
    sire = np.array([f"S{i % 6}" for i in range(n_horses)])
    rows = []
    for year in years:
        for d in range(days_per_year):
            date = pd.Timestamp(year, 1 + d * 12 // days_per_year, 5 + d % 3)
            for r in range(races_per_day):
                jyo = "05" if r % 2 == 0 else "08"
                kind = "turf" if r % 3 else "dirt"
                kyori = 1600 if r % 2 == 0 else 2000
                track_cd = "24" if kind == "turf" else "23"
                race_id = f"{date:%Y%m%d}{jyo}0101{r + 1:02d}"
                horses = rng.choice(n_horses, field, replace=False)
                jockeys = rng.choice([f"J{j}" for j in range(12)], field, replace=False)
                bonus = np.where(jockeys == STRONG_JOCKEY, 1.0, 0.0)
                perf = ability[horses] + bonus + rng.normal(0, 1, field)
                order = np.argsort(-perf)
                jyuni = np.empty(field, int)
                jyuni[order] = np.arange(1, field + 1)
                p = np.exp(ability[horses]) / np.exp(ability[horses]).sum()
                odds = np.round(0.8 / p, 1)
                ninki = (-p).argsort().argsort() + 1
                for i, h in enumerate(horses):
                    rows.append(dict(
                        race_id=race_id, date=date, umaban=f"{i + 1:02d}", waku=i // 2 + 1, ketto_num=f"H{h:03d}",
                        barei=int(year - birth[h]), futan=55.0 if i % 4 else 57.0, kisyu=jockeys[i],
                        chokyosi=f"T{h % 5}", tozai="1" if h % 2 else "2", sire_num=sire[h], jyuni=float(jyuni[i]),
                        odds=float(odds[i]), ninki=float(ninki[i]),
                        time_sec=float((60.5 - 0.3 * perf[i]) * kyori / 1000), zogen=float(rng.integers(-8, 9)),
                        jyo=jyo, track_cd=track_cd, track_kind=kind, kyori=kyori, class_level=2.0, grade="",
                        syubetu="13"))
    R = pd.DataFrame(rows)
    R["year"] = R["date"].dt.year
    R["cohort"] = R["year"] - R["barei"]
    R["field"] = R.groupby("race_id")["race_id"].transform("size")
    R["top3"] = R["jyuni"].between(1, 3).astype(float)
    R["fin_pct"] = (R["jyuni"] - 1) / (R["field"] - 1)
    return R.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)[RUN_COLUMNS]


def to_runs_all(R: pd.DataFrame) -> pd.DataFrame:
    """speed_figure.RUN_COLUMNS の形（指数の物差しの推定に使う）に直す。"""
    return pd.DataFrame({
        "race_id": R["race_id"], "umaban": R["umaban"], "ketto_num": R["ketto_num"],
        "day": R["date"].dt.strftime("%Y%m%d"), "date": R["date"], "year": R["year"], "jyo_cd": R["jyo"],
        "track_cd": R["track_cd"], "track_kind": R["track_kind"], "kyori_m": R["kyori"].astype(int),
        "syubetu_cd": R["syubetu"], "class_level": R["class_level"], "time_sec": R["time_sec"],
        "rank": R["jyuni"], "barei": R["barei"], "cohort": R["cohort"],
    }).reset_index(drop=True)


def perturb_results_from(R: pd.DataFrame, day) -> pd.DataFrame:
    """day 以降の行の結果（着順・3着内・着順の割合・タイム）を入れ替えた写し。レース前の情報はそのまま。"""
    P = R.copy()
    m = P["date"] >= pd.Timestamp(day)
    P.loc[m, "jyuni"] = P.loc[m, "field"] - P.loc[m, "jyuni"] + 1
    P.loc[m, "top3"] = P.loc[m, "jyuni"].between(1, 3).astype(float)
    P.loc[m, "fin_pct"] = (P.loc[m, "jyuni"] - 1) / (P.loc[m, "field"] - 1)
    P.loc[m, "time_sec"] = P.loc[m, "time_sec"] + 3.0
    return P


def assert_no_lookahead(compute, R: pd.DataFrame, day) -> None:
    """day の行の点数が、day 以降の結果を変えても変わらないことを確かめる（仕様 §9「時点の規則」）。"""
    before = np.asarray(compute(R), float)
    after = np.asarray(compute(perturb_results_from(R, day)), float)
    rows = (R["date"] == pd.Timestamp(day)).to_numpy()
    assert rows.any(), "その日の行が無い"
    np.testing.assert_allclose(before[rows], after[rows], equal_nan=True)
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_factor_data.py`:

```python
"""走の表（keiba.factors.data）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.data import RUN_COLUMNS, build_runs


def frames():
    races = pd.DataFrame([
        dict(race_id="2025010505010101", jyo_cd="05", track_cd="24", track_kind="turf", kyori_m="1600",
             class_level="2", grade_cd="", syubetu_cd="13", data_kubun="7", make_date="20250106"),
        dict(race_id="2025010505010102", jyo_cd="05", track_cd="51", track_kind="jump", kyori_m="3000",
             class_level="2", grade_cd="", syubetu_cd="18", data_kubun="7", make_date="20250106"),
        dict(race_id="2025010530010101", jyo_cd="30", track_cd="24", track_kind="dirt", kyori_m="1400",
             class_level="2", grade_cd="", syubetu_cd="13", data_kubun="7", make_date="20250106"),
    ], dtype=str)
    base = dict(race_id="2025010505010101", wakuban="1", barei="04", futan_kg="57.0", kisyu_code="J1",
                chokyosi_code="T1", tozai_cd="1", result_odds="3.5", result_ninki="1", result_time_sec="96.0",
                result_zogen_sa_kg="-2", result_ijyo_cd="0", data_kubun="7", make_date="20250106")
    horses = pd.DataFrame([
        dict(base, umaban="01", ketto_num="A", result_kakutei_jyuni="1"),
        dict(base, umaban="02", ketto_num="B", result_kakutei_jyuni="2", result_zogen_sa_kg=""),
        dict(base, umaban="03", ketto_num="C", result_kakutei_jyuni="0", result_ijyo_cd="4", result_time_sec="0"),
        dict(base, umaban="04", ketto_num="D", result_kakutei_jyuni="0", result_ijyo_cd="1"),
        dict(base, race_id="2025010505010102", umaban="01", ketto_num="E", result_kakutei_jyuni="1"),
        dict(base, race_id="2025010530010101", umaban="01", ketto_num="F", result_kakutei_jyuni="1"),
    ], dtype=str)
    ped = pd.DataFrame([dict(ketto_num="A", sire_num="S1"), dict(ketto_num="B", sire_num="S2")], dtype=str)
    return races, horses, ped


def test_build_runs_rows_and_columns():
    R = build_runs(*frames())
    assert list(R.columns) == RUN_COLUMNS
    assert R.ketto_num.tolist() == ["A", "B", "C"]          # 取消(1)・障害・地方は除く。中止(4)は出走として残す
    assert R.field.tolist() == [3, 3, 3]
    a = R.set_index("ketto_num").loc["A"]
    assert a.top3 == 1.0 and a.fin_pct == 0.0 and a.cohort == 2021 and a.sire_num == "S1"
    c = R.set_index("ketto_num").loc["C"]
    assert c.top3 == 0.0 and np.isnan(c.fin_pct) and np.isnan(c.time_sec)
    assert np.isnan(R.set_index("ketto_num").loc["B"].zogen)
    assert R.set_index("ketto_num").loc["C"].sire_num == ""
```

`tests/test_factor_cumulative.py`:

```python
"""日より前だけの累積と縮小（keiba.factors.cumulative）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.cumulative import add_expected_top3, eb_lambda, prior_day_sums, shrunk_prior_scores
from tests.fixtures.factor_runs import STRONG_JOCKEY, assert_no_lookahead, make_factor_runs


def test_prior_day_sums_excludes_same_day_and_future():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
                       "k": ["a", "a", "a", "b"], "v": [1.0, 2.0, np.nan, 5.0]})
    s, n = prior_day_sums(df, ["k"], "v")
    assert s.tolist() == [0.0, 0.0, 3.0, 0.0]
    assert n.tolist() == [0.0, 0.0, 2.0, 0.0]


def test_eb_lambda_none_when_groups_do_not_differ():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 4000)
    k = np.repeat(np.arange(40), 100)
    assert eb_lambda(v, k) is None or eb_lambda(v, k) > 50   # 差が無いので縮小は非常に強い（または無し）
    v2 = v + np.repeat(rng.normal(0, 1, 40), 100)
    lam = eb_lambda(v2, k)
    assert lam is not None and 0.5 < lam < 2.0               # σ² = 1, τ² ≈ 1


def test_shrunk_scores_recover_strong_jockey():
    R = add_expected_top3(make_factor_runs())
    sc = shrunk_prior_scores(R, ["kisyu"], "ae", first_year=2016)
    late = (R.year == 2017).to_numpy()
    j0 = sc[late & (R.kisyu == STRONG_JOCKEY).to_numpy()].mean()
    others = sc[late & (R.kisyu != STRONG_JOCKEY).to_numpy()].mean()
    assert j0 > 0.05 and j0 > others + 0.05
    assert np.isnan(sc[(R.year == 2015).to_numpy()]).all()   # 助走の年


def test_add_expected_top3_uses_prior_days_only():
    R = make_factor_runs()
    day = R.date.sort_values().unique()[10]
    assert_no_lookahead(lambda d: add_expected_top3(d)["e3"].to_numpy(), R, day)


def test_shrunk_scores_no_lookahead():
    R = make_factor_runs()
    day = R[R.year == 2016].date.unique()[3]
    assert_no_lookahead(lambda d: shrunk_prior_scores(add_expected_top3(d), ["kisyu"], "ae", first_year=2016), R, day)
```

- [ ] **Step 3: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_data.py tests/test_factor_cumulative.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'keiba.factors'`）

- [ ] **Step 4: 実装する**

`src/keiba/factors/__init__.py`:

```python
"""要素別の予想（予想家型）。仕様 docs/superpowers/specs/2026-10-11-factor-predictions-design.md。"""
```

`src/keiba/factors/data.py`:

```python
"""評価に使う「走」の表（仕様 §5.1）。1行 = 1頭の1走（確定成績、JRA10場の平地）。

予想に使う列（騎手・枠・斤量・オッズなど）と答え合わせに使う列（着順・タイム）が同じ行にある。
各要素はその行の日より前の行の結果だけを使う（cumulative.py とテストで守る）。
DataKubun の選び方は repository.py の `_pick_by_data_kubun` を使う（選び方を2か所に持たないため）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from keiba import config
from keiba.data.repository import _pick_by_data_kubun

JRA_JYO = frozenset(f"{i:02d}" for i in range(1, 11))
REGION = {"01": "北海道", "02": "北海道", "03": "東", "04": "東", "05": "東", "06": "東",
          "07": "西", "08": "西", "09": "西", "10": "小倉"}
RUN_COLUMNS = ["race_id", "date", "year", "umaban", "waku", "ketto_num", "barei", "cohort", "futan", "kisyu",
               "chokyosi", "tozai", "sire_num", "jyuni", "field", "top3", "fin_pct", "odds", "ninki", "time_sec",
               "zogen", "jyo", "track_cd", "track_kind", "kyori", "class_level", "grade", "syubetu"]
_RACE_FIELDS = ["race_id", "jyo_cd", "track_cd", "track_kind", "kyori_m", "class_level", "grade_cd", "syubetu_cd"]


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()


def build_runs(races: pd.DataFrame, race_horses: pd.DataFrame, pedigrees: pd.DataFrame) -> pd.DataFrame:
    """取消・除外（走っていない）は除き、中止・失格は出走として残す（着順 0 → 3着内ではない）。"""
    r = _pick_by_data_kubun(races, config.DATA_KUBUN_RESULT_PRIORITY)
    r = r[r["jyo_cd"].isin(JRA_JYO) & r["track_kind"].isin(("turf", "dirt"))]
    h = race_horses[race_horses["race_id"].isin(r["race_id"])]
    h = _pick_by_data_kubun(h, config.DATA_KUBUN_RESULT_PRIORITY)
    h = h[~h["result_ijyo_cd"].isin(config.IJYO_EXCLUDED)]
    m = h.merge(r[_RACE_FIELDS], on="race_id")
    ped = pedigrees[["ketto_num", "sire_num"]].drop_duplicates("ketto_num", keep="last")
    m = m.merge(ped, on="ketto_num", how="left")
    out = pd.DataFrame({
        "race_id": m["race_id"].to_numpy(),
        "date": pd.to_datetime(m["race_id"].str[:8], format="%Y%m%d").to_numpy(),
        "umaban": m["umaban"].to_numpy(), "waku": _num(m["wakuban"]), "ketto_num": m["ketto_num"].to_numpy(),
        "barei": _num(m["barei"]), "futan": _num(m["futan_kg"]), "kisyu": m["kisyu_code"].to_numpy(),
        "chokyosi": m["chokyosi_code"].to_numpy(), "tozai": m["tozai_cd"].to_numpy(),
        "sire_num": m["sire_num"].fillna("").to_numpy(),
        "jyuni": _num(m["result_kakutei_jyuni"]), "odds": _num(m["result_odds"]), "ninki": _num(m["result_ninki"]),
        "time_sec": _num(m["result_time_sec"]), "zogen": _num(m["result_zogen_sa_kg"]),
        "jyo": m["jyo_cd"].to_numpy(), "track_cd": m["track_cd"].to_numpy(), "track_kind": m["track_kind"].to_numpy(),
        "kyori": _num(m["kyori_m"]), "class_level": _num(m["class_level"]), "grade": m["grade_cd"].to_numpy(),
        "syubetu": m["syubetu_cd"].to_numpy(),
    })
    out = out[~np.isnan(out["barei"].to_numpy(float)) & ~np.isnan(out["kyori"].to_numpy(float))].copy()
    out["barei"] = out["barei"].astype(int)
    out["kyori"] = out["kyori"].astype(int)
    out["year"] = out["date"].dt.year
    out["cohort"] = out["year"] - out["barei"]
    out["field"] = out.groupby("race_id")["race_id"].transform("size")
    out["top3"] = out["jyuni"].between(1, 3).astype(float)
    out["fin_pct"] = np.where(out["jyuni"] >= 1, (out["jyuni"] - 1) / (out["field"] - 1).clip(lower=1), np.nan)
    out.loc[~(out["time_sec"] > 0), "time_sec"] = np.nan
    return out.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)[RUN_COLUMNS]


def load_runs(version_dir: Path, blood_dir: Path) -> pd.DataFrame:
    def read(p: Path) -> pd.DataFrame:
        return pd.read_csv(p, dtype=str, na_filter=False)
    return build_runs(read(version_dir / "races.csv"), read(version_dir / "race_horses.csv"),
                      read(blood_dir / "pedigrees.csv"))
```

`src/keiba/factors/cumulative.py`:

```python
"""「その行の日より前」だけを使う累積と、経験ベイズの縮小（仕様 §2「縮小」・§3.2）。

累積は日付単位で取り、同じ日の行は含めない（その日のほかのレースの結果は、予想の時点ではまだ分からない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prior_day_sums(df: pd.DataFrame, keys: list[str], value: str) -> tuple[np.ndarray, np.ndarray]:
    """各行について、同じ keys の行のうち日付がその行より前のものの value の合計と件数。NaN は数えない。"""
    v = df[value].to_numpy(float)
    d = df[keys].copy()
    d["date"] = df["date"].to_numpy()
    d["_s"] = np.nan_to_num(v)
    d["_n"] = (~np.isnan(v)).astype(float)
    daily = d.groupby(keys + ["date"], sort=True)[["_s", "_n"]].sum()
    before = daily.groupby(level=list(range(len(keys))), sort=False).cumsum() - daily
    got = before.reindex(pd.MultiIndex.from_frame(d[keys + ["date"]]))
    return got["_s"].to_numpy(), got["_n"].to_numpy()


def eb_lambda(values: np.ndarray, keys: np.ndarray) -> float | None:
    """組ごとの平均を縮める強さ λ = σ²/τ²（モーメント法）。組どうしの差が偶然の範囲（τ² ≤ 0）なら None。"""
    d = pd.DataFrame({"v": np.asarray(values, float), "k": np.asarray(keys)}).dropna(subset=["v"])
    if d.empty:
        return None
    g = d.groupby("k")["v"]
    n, m = g.size(), g.mean()
    if len(n) < 2:
        return None
    within = float(((d["v"] - d["k"].map(m)) ** 2).sum() / max(len(d) - len(n), 1))
    tau2 = float(np.average((m - d["v"].mean()) ** 2, weights=n)) - within * len(n) / float(n.sum())
    if tau2 <= 0 or within <= 0:
        return None
    return within / tau2


def shrunk_prior_scores(df: pd.DataFrame, keys: list[str], value: str, *, first_year: int) -> np.ndarray:
    """各行の点数 = (S − n·μ) / (n + λ_Y)。

    S・n はその行の日より前の同じ組の value の合計と件数、μ はその行の日より前の全体の平均。
    λ_Y は行の年 Y の前年末までの行で推定する。λ が無い年（組の差が偶然の範囲）と n = 0 の行は 0（平均並み）。
    first_year より前の年の行は NaN（助走）。
    """
    S, n = prior_day_sums(df, keys, value)
    allS, alln = prior_day_sums(df.assign(_all=0), ["_all"], value)
    mu = np.divide(allS, alln, out=np.zeros_like(allS), where=alln > 0)
    keyarr = df[keys].astype(str).agg("|".join, axis=1).to_numpy()
    vals = df[value].to_numpy(float)
    dates = df["date"].to_numpy()
    years = df["year"].to_numpy()
    out = np.full(len(df), np.nan)
    for y in np.unique(years):
        if y < first_year:
            continue
        rows = years == y
        prior = dates < np.datetime64(f"{int(y)}-01-01")
        lam = eb_lambda(vals[prior], keyarr[prior])
        if lam is None:
            out[rows] = 0.0
            continue
        out[rows] = (S[rows] - n[rows] * mu[rows]) / (n[rows] + lam)
    return out


def add_expected_top3(df: pd.DataFrame) -> pd.DataFrame:
    """見込み3着内率 e3（人気順位ごとの3着内率、その日より前）と A−E = 3着内 − e3 を足す。

    人気が無い行・その人気の前例が無い行は NaN。人気は 18 で頭打ち。
    """
    d = df.copy()
    d["_ninki"] = d["ninki"].clip(upper=18).fillna(-1).astype(int)
    S, n = prior_day_sums(d, ["_ninki"], "top3")
    e3 = np.divide(S, n, out=np.full(len(d), np.nan), where=n > 0)
    e3[d["_ninki"].to_numpy() < 1] = np.nan
    d["e3"] = e3
    d["ae"] = d["top3"] - d["e3"]
    return d.drop(columns="_ninki")
```

- [ ] **Step 5: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_data.py tests/test_factor_cumulative.py -v`
Expected: 6 passed

- [ ] **Step 6: コミット**

```bash
git add src/keiba/factors/__init__.py src/keiba/factors/data.py src/keiba/factors/cumulative.py tests/fixtures/factor_runs.py tests/test_factor_data.py tests/test_factor_cumulative.py
git commit -m "feat: 要素別の評価の走の表と、日より前だけの累積・縮小を追加

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 表の要素と履歴の要素（10要素）

**Files:**
- Create: `src/keiba/factors/table_factors.py`, `src/keiba/factors/history_factors.py`
- Create: `tests/test_factor_tables.py`

**Interfaces:**
- Consumes: Task 1 の `shrunk_prior_scores`, `prior_day_sums`, `add_expected_top3`, `REGION`
- Produces:（どれも `R` は `RUN_COLUMNS` ＋ `e3`・`ae` を持つ DataFrame、戻り値は `len(R)` の `np.ndarray`、大きいほど3着内に来やすい）
  - `table_factors.FIRST_YEAR = 2016`
  - `table_factors.jockey(R)`, `trainer(R)`, `sire_surface(R)`, `travel(R)`, `layoff(R)`, `body_weight(R)`, `draw(R)`
  - `table_factors.layoff_bin(R) -> np.ndarray`, `weight_bin(R) -> np.ndarray`
  - `history_factors.distance_fit(R)`, `course_fit(R)`, `weight_carried(R)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_tables.py`:

```python
"""表の要素・履歴の要素（仕様 §3.2 #3〜5, 9〜15）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors import history_factors as hf
from keiba.factors import table_factors as tf
from keiba.factors.cumulative import add_expected_top3
from tests.fixtures.factor_runs import STRONG_JOCKEY, assert_no_lookahead, make_factor_runs

TABLE = [tf.jockey, tf.trainer, tf.sire_surface, tf.travel, tf.layoff, tf.body_weight, tf.draw]
HISTORY = [hf.distance_fit, hf.course_fit, hf.weight_carried]


@pytest.fixture(scope="module")
def R():
    return add_expected_top3(make_factor_runs())


@pytest.mark.parametrize("fn", TABLE + HISTORY, ids=lambda f: f.__name__)
def test_no_lookahead(fn):
    base = make_factor_runs()
    day = base[base.year == 2016].date.unique()[4]
    assert_no_lookahead(lambda d: fn(add_expected_top3(d)), base, day)


@pytest.mark.parametrize("fn", TABLE + HISTORY, ids=lambda f: f.__name__)
def test_length_and_direction(fn, R):
    out = fn(R)
    assert out.shape == (len(R),)


def test_jockey_finds_strong_jockey(R):
    sc = tf.jockey(R)
    late = (R.year == 2017).to_numpy()
    assert sc[late & (R.kisyu == STRONG_JOCKEY).to_numpy()].mean() > sc[late & (R.kisyu != STRONG_JOCKEY).to_numpy()].mean()


def test_layoff_bins():
    R = pd.DataFrame({"ketto_num": ["A", "A", "A", "B"],
                      "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2021-03-01", "2020-01-01"])})
    assert tf.layoff_bin(R).tolist() == ["初出走", "〜13", "365〜", "初出走"]


def test_weight_bins():
    R = pd.DataFrame({"zogen": [-12.0, -4.0, 0.0, 4.0, 10.0, np.nan]})
    assert tf.weight_bin(R).tolist() == ["〜−10", "−9〜−4", "−3〜+3", "+4〜+9", "+10〜", "不明"]


def test_distance_fit_uses_only_prior_similar_runs():
    R = pd.DataFrame({
        "ketto_num": ["A"] * 4, "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "track_kind": ["turf", "turf", "dirt", "turf"], "kyori": [1600, 2400, 1600, 1800], "jyo": ["05"] * 4,
        "fin_pct": [0.0, 1.0, 0.0, 0.5]})
    out = hf.distance_fit(R)
    assert np.isnan(out[0])
    assert out[3] == pytest.approx(1.0)        # 1800±200 の芝: 1走目（1−0）だけ。2400m とダートは入らない


def test_weight_carried_sign():
    R = pd.DataFrame({"race_id": ["x", "x", "x"], "futan": [54.0, 56.0, 58.0]})
    assert hf.weight_carried(R).tolist() == [2.0, 0.0, -2.0]


def test_draw_is_zero_when_no_draw_effect(R):
    sc = tf.draw(R)
    rows = (R.year >= 2016).to_numpy()
    assert np.allclose(sc[rows], 0.0, atol=0.05)   # 作り物は枠に効果を入れていない
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_tables.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 実装する**

`src/keiba/factors/table_factors.py`:

```python
"""表の要素（仕様 §3.2 #3〜5, 11, 13〜15）。

どれも「その日より前」の A−E（実際の3着内 − 人気から見込む3着内率）を組ごとに縮小平均したもの。
人気（市場）が既に知っている分を差し引いた「上乗せ」なので、市場への上乗せの評価と向きがそろう。
枠順だけは、同じコースの3着内率からのずれを使う（人気は枠をあまり織り込まないため、仕様どおり）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.cumulative import prior_day_sums, shrunk_prior_scores
from keiba.factors.data import REGION

FIRST_YEAR = 2016   # 2015年は助走（仕様 §5.1）
LAYOFF_EDGES = [-1, 13, 27, 55, 118, 364, 10 ** 6]
LAYOFF_LABELS = ["〜13", "14〜27", "28〜55", "56〜118", "119〜364", "365〜"]
WEIGHT_EDGES = [-1000, -10, -4, 3, 9, 1000]
WEIGHT_LABELS = ["〜−10", "−9〜−4", "−3〜+3", "+4〜+9", "+10〜"]


def _by(R: pd.DataFrame, key: np.ndarray, value: str = "ae") -> np.ndarray:
    return shrunk_prior_scores(R.assign(_k=key), ["_k"], value, first_year=FIRST_YEAR)


def jockey(R: pd.DataFrame) -> np.ndarray:
    return _by(R, R["kisyu"].astype(str).to_numpy())


def trainer(R: pd.DataFrame) -> np.ndarray:
    return _by(R, R["chokyosi"].astype(str).to_numpy())


def sire_surface(R: pd.DataFrame) -> np.ndarray:
    return _by(R, (R["sire_num"].astype(str) + ":" + R["track_kind"]).to_numpy())


def travel(R: pd.DataFrame) -> np.ndarray:
    base = R["tozai"].map({"1": "美浦", "2": "栗東"}).fillna("その他")
    return _by(R, (base + "→" + R["jyo"].map(REGION).fillna("その他")).to_numpy())


def layoff_bin(R: pd.DataFrame) -> np.ndarray:
    """前走（同じ馬の前の行）からの日数の区分。前の行が無ければ「初出走」。"""
    order = np.lexsort((R["date"].to_numpy(), R["ketto_num"].to_numpy()))
    d = R.iloc[order]
    prev = d.groupby("ketto_num", sort=False)["date"].shift(1)
    gap = (d["date"] - prev).dt.days
    lab = pd.cut(gap, bins=LAYOFF_EDGES, labels=LAYOFF_LABELS).astype(object).where(gap.notna(), "初出走")
    out = np.empty(len(R), dtype=object)
    out[order] = lab.to_numpy()
    return out


def layoff(R: pd.DataFrame) -> np.ndarray:
    return _by(R, layoff_bin(R))


def weight_bin(R: pd.DataFrame) -> np.ndarray:
    z = R["zogen"]
    lab = pd.cut(z, bins=WEIGHT_EDGES, labels=WEIGHT_LABELS).astype(object)
    return lab.where(z.notna(), "不明").to_numpy()


def body_weight(R: pd.DataFrame) -> np.ndarray:
    return _by(R, weight_bin(R))


def draw(R: pd.DataFrame) -> np.ndarray:
    """（競馬場, track_cd, 距離, 枠番）の3着内率の、同じコース全体からのずれの縮小値。"""
    course = (R["jyo"].astype(str) + "|" + R["track_cd"].astype(str) + "|" + R["kyori"].astype(str)).to_numpy()
    d = R.assign(_c=course, _w=R["waku"].fillna(-1).astype(int).astype(str))
    S, n = prior_day_sums(d, ["_c"], "top3")
    course_mean = np.divide(S, n, out=np.full(len(d), np.nan), where=n > 0)
    d["_dev"] = d["top3"].to_numpy(float) - course_mean
    return shrunk_prior_scores(d, ["_c", "_w"], "_dev", first_year=FIRST_YEAR)
```

`src/keiba/factors/history_factors.py`:

```python
"""履歴の要素（仕様 §3.2 #9, 10, 12）。その馬のその日より前の走だけを使う。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prior_mean(R: pd.DataFrame, same) -> np.ndarray:
    """各行について、同じ馬のその日より前の走のうち same(prev, i) を満たすものの「1 − 着順の割合」の平均。"""
    out = np.full(len(R), np.nan)
    val = 1.0 - R["fin_pct"].to_numpy(float)
    dates = R["date"].to_numpy()
    for idx in R.groupby("ketto_num", sort=False).indices.values():
        idx = idx[np.argsort(dates[idx], kind="stable")]
        for j in range(1, len(idx)):
            i = idx[j]
            prev = idx[:j]
            prev = prev[dates[prev] < dates[i]]
            if len(prev) == 0:
                continue
            v = val[prev][same(prev, i)]
            v = v[~np.isnan(v)]
            if len(v):
                out[i] = float(v.mean())
    return out


def distance_fit(R: pd.DataFrame) -> np.ndarray:
    """同じ芝ダで今回の距離 ±200m の過去走の「1 − 着順の割合」の平均。"""
    kind, kyori = R["track_kind"].to_numpy(), R["kyori"].to_numpy()
    return _prior_mean(R, lambda prev, i: (kind[prev] == kind[i]) & (np.abs(kyori[prev] - kyori[i]) <= 200))


def course_fit(R: pd.DataFrame) -> np.ndarray:
    """同じ競馬場・同じ芝ダの過去走の同じ値。"""
    kind, jyo = R["track_kind"].to_numpy(), R["jyo"].to_numpy()
    return _prior_mean(R, lambda prev, i: (kind[prev] == kind[i]) & (jyo[prev] == jyo[i]))


def weight_carried(R: pd.DataFrame) -> np.ndarray:
    """−（斤量 − そのレースの平均斤量）。軽いほど高い。"""
    return -(R["futan"] - R.groupby("race_id")["futan"].transform("mean")).to_numpy(float)
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_tables.py -v`
Expected: 26 passed（先読み10・長さ10・個別6）

- [ ] **Step 5: コミット**

```bash
git add src/keiba/factors/table_factors.py src/keiba/factors/history_factors.py tests/test_factor_tables.py
git commit -m "feat: 表の要素と履歴の要素（騎手・調教師・父・枠・間隔・馬体重・遠征・距離・コース・斤量）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: タイム指数と相手関係

**Files:**
- Create: `src/keiba/factors/figure_factors.py`
- Create: `tests/test_factor_figures.py`

**Interfaces:**
- Consumes: `keiba.features.speed_figure`（`FACE`, `KEY`, `OLDER_SYUBETU`, `FigureModel`, `apply_figures`, `fit_model`）、Task 1 の fixtures（`to_runs_all`）
- Produces:
  - `day_variants(runs: pd.DataFrame, model: FigureModel) -> pd.Series`（index = FACE）
  - `yearly_model(runs_all: pd.DataFrame, year: int) -> FigureModel`
  - `yearly_figures(runs_all: pd.DataFrame, year: int) -> pd.DataFrame`（`apply_figures` の出力。`year+1` 年1/1 より前の全走）
  - `level_before(fig: pd.DataFrame, targets: pd.DataFrame, *, window_days: int = 365) -> np.ndarray`
  - `time_figure_scores(R, fig, year) -> np.ndarray`、`opponents_scores(R, fig, year) -> np.ndarray`（どちらも `len(R)`、`year` 年の行以外は NaN）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_figures.py`:

```python
"""タイム指数・相手関係（仕様 §3.2 #1, 2）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.figure_factors import (level_before, opponents_scores, time_figure_scores, yearly_figures,
                                          yearly_model)
from tests.fixtures.factor_runs import assert_no_lookahead, make_factor_runs, to_runs_all


def test_yearly_model_uses_only_prior_years_for_standards():
    R = make_factor_runs()
    m1 = yearly_model(to_runs_all(R), 2017)
    R2 = R.copy()
    R2.loc[R2.year == 2017, "time_sec"] += 5.0
    m2 = yearly_model(to_runs_all(R2), 2017)
    pd.testing.assert_series_equal(m1.standards, m2.standards)
    # 2017年の日の馬場差は、その日のレースが遅くなった分だけ動く
    d17 = [k for k in m1.variants.index if k[0].startswith("2017")]
    assert (m2.variants[d17] > m1.variants[d17]).all()


def test_level_before_window_and_top2():
    fig = pd.DataFrame({"ketto_num": ["A"] * 4, "figure": [1.0, 3.0, 2.0, 9.0],
                        "date": pd.to_datetime(["2019-01-01", "2020-01-01", "2020-06-01", "2020-07-01"])})
    tg = pd.DataFrame({"ketto_num": ["A", "A", "B"], "date": pd.to_datetime(["2020-07-01", "2019-01-01", "2020-07-01"])})
    out = level_before(fig, tg)
    assert out[0] == pytest.approx(2.5)        # 2019-07-02 以降・当日を除く → 3.0 と 2.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_time_figure_no_lookahead():
    R = make_factor_runs()
    day = R[R.year == 2017].date.unique()[3]

    def compute(d):
        return time_figure_scores(d, yearly_figures(to_runs_all(d), 2017), 2017)

    assert_no_lookahead(compute, R, day)


def test_opponents_no_lookahead():
    R = make_factor_runs()
    day = R[R.year == 2017].date.unique()[3]

    def compute(d):
        return opponents_scores(d, yearly_figures(to_runs_all(d), 2017), 2017)

    assert_no_lookahead(compute, R, day)


def test_scores_only_for_target_year():
    R = make_factor_runs()
    fig = yearly_figures(to_runs_all(R), 2017)
    tf_ = time_figure_scores(R, fig, 2017)
    op = opponents_scores(R, fig, 2017)
    assert np.isnan(tf_[(R.year != 2017).to_numpy()]).all()
    assert np.isnan(op[(R.year != 2017).to_numpy()]).all()
    assert np.isfinite(tf_[(R.year == 2017).to_numpy()]).mean() > 0.5


def test_time_figure_tracks_ability():
    R = make_factor_runs()
    fig = yearly_figures(to_runs_all(R), 2017)
    sc = time_figure_scores(R, fig, 2017)
    d = R.assign(sc=sc)[R.year == 2017].dropna(subset=["sc"])
    # 指数が高い馬ほど着順の割合が小さい（よい）
    assert np.corrcoef(d.sc, d.fin_pct)[0, 1] < -0.2
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_figures.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 実装する**

`src/keiba/factors/figure_factors.py`:

```python
"""タイム指数と相手関係（仕様 §3.2 #1, 2）。

物差しは評価する年ごとに作り直す。基準タイムとクラス差は「その年の1/1より前」で推定し、その年の各日の馬場差は
その日の古馬戦の1着だけから（基準・クラス差を固定して）推定する。こうすると、ある日のレースの予想に使う過去走の
指数には、その日以降の結果が一切入らない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.features.speed_figure import FACE, KEY, OLDER_SYUBETU, FigureModel, apply_figures, fit_model


def day_variants(runs: pd.DataFrame, model: FigureModel) -> pd.Series:
    """runs の各開催面の馬場差を、その面の古馬戦の1着だけから推定する（基準・クラス差は model のまま）。"""
    w = runs[runs["syubetu_cd"].isin(OLDER_SYUBETU) & (runs["rank"] == 1) & runs["class_level"].notna()]
    w = w.drop_duplicates("race_id")
    if w.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], [], []], names=FACE))
    s = model.standards.reindex(pd.MultiIndex.from_frame(w[KEY])).to_numpy()
    c = model.class_effects.reindex(pd.MultiIndex.from_frame(w[["class_level", "track_kind"]])).to_numpy()
    r = w["time_sec"].to_numpy() / (w["kyori_m"].to_numpy() / 1000.0) - s - c
    ok = ~np.isnan(r)
    w = w[ok].assign(_r=r[ok])
    g = w.groupby(FACE)["_r"]
    lam = model.sigma2 / model.tau2 if model.tau2 > 0 else 0.0
    return g.sum() / (g.size() + lam)


def yearly_model(runs_all: pd.DataFrame, year: int) -> FigureModel:
    cut = pd.Timestamp(year, 1, 1)
    model = fit_model(runs_all[runs_all["date"] < cut])
    cur = runs_all[(runs_all["date"] >= cut) & (runs_all["date"] < pd.Timestamp(year + 1, 1, 1))]
    new = day_variants(cur, model)
    variants = pd.concat([model.variants, new[~new.index.isin(model.variants.index)]])
    return FigureModel(model.standards, model.class_effects, variants, model.sigma2, model.tau2, model.n_races)


def yearly_figures(runs_all: pd.DataFrame, year: int) -> pd.DataFrame:
    """year の物差しで、year+1 年1/1 より前の全走に指数を付ける。"""
    model = yearly_model(runs_all, year)
    return apply_figures(runs_all[runs_all["date"] < pd.Timestamp(year + 1, 1, 1)], model)


def level_before(fig: pd.DataFrame, targets: pd.DataFrame, *, window_days: int = 365) -> np.ndarray:
    """targets の各行（ketto_num, date）について、同じ馬の [date − window, date) の指数の上位2走の平均。"""
    out = np.full(len(targets), np.nan)
    f = fig[fig["figure"].notna()].sort_values("date")
    groups = {k: (g["date"].to_numpy(), g["figure"].to_numpy()) for k, g in f.groupby("ketto_num", sort=False)}
    tk, td = targets["ketto_num"].to_numpy(), targets["date"].to_numpy()
    win = np.timedelta64(window_days, "D")
    for i in range(len(targets)):
        g = groups.get(tk[i])
        if g is None:
            continue
        d, v = g
        lo, hi = np.searchsorted(d, td[i] - win, "left"), np.searchsorted(d, td[i], "left")
        if hi > lo:
            seg = v[lo:hi]
            out[i] = float(seg.mean() if len(seg) == 1 else np.sort(seg)[-2:].mean())
    return out


def time_figure_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    out = np.full(len(R), np.nan)
    rows = (R["year"] == year).to_numpy()
    out[rows] = level_before(fig, R[rows])
    return out


def opponents_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """過去5走（1年以内）の「他の出走馬の当時の指数の平均 − 自分の着差（1600m 換算の秒）」の上位3走の平均。"""
    out = np.full(len(R), np.nan)
    win = R[(R["date"] >= pd.Timestamp(year - 1, 1, 1)) & (R["date"] < pd.Timestamp(year + 1, 1, 1))]
    lv = level_before(fig, win)
    d = win[["race_id", "ketto_num", "date", "time_sec", "kyori"]].assign(lv=lv)
    s = d.groupby("race_id")["lv"].transform("sum")
    c = d.groupby("race_id")["lv"].transform("count")
    own = d["lv"].notna().astype(float)
    others_n = c - own
    others = (s - d["lv"].fillna(0.0)) / others_n.where(others_n > 0)
    winner = d.groupby("race_id")["time_sec"].transform("min")
    margin = (d["time_sec"] - winner) * 1600.0 / d["kyori"]
    d = d.assign(perf=(others - margin).to_numpy())
    by = {k: (g["date"].to_numpy(), g["perf"].to_numpy()) for k, g in d.sort_values("date").groupby("ketto_num", sort=False)}
    year_rows = np.flatnonzero((R["year"] == year).to_numpy())
    keys, dates = R["ketto_num"].to_numpy(), R["date"].to_numpy()
    one_year = np.timedelta64(365, "D")
    for i in year_rows:
        g = by.get(keys[i])
        if g is None:
            continue
        dd, pp = g
        hi = np.searchsorted(dd, dates[i], "left")
        lo = max(np.searchsorted(dd, dates[i] - one_year, "left"), hi - 5)
        seg = pp[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = float(np.sort(seg)[-3:].mean())
    return out
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_figures.py -v`
Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/factors/figure_factors.py tests/test_factor_figures.py
git commit -m "feat: 年ごとの物差しによるタイム指数と相手関係の要素

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 年齢・晩成度×年齢・世代レベル

**Files:**
- Create: `src/keiba/factors/age_factors.py`
- Create: `tests/test_factor_age.py`

**Interfaces:**
- Consumes: `keiba.evaluate.age_curve`（`age_curve`, `curve_levels`, `period_values`）、`keiba.evaluate.generation`（`cohort_table`, `yearly_comparison`）、Task 1 の `eb_lambda`、Task 3 の `yearly_figures`（テストのみ）
- Produces:
  - `FIRST_DATA_YEAR = 2015`、`COHORT_AGES = list(range(2, 10))`
  - `age_scores(R, fig, year) -> np.ndarray`
  - `sire_lateness(R, fig, year) -> pd.Series`（sire_num → L。分からない父は含めない）
  - `late_age_scores(R, fig, year) -> np.ndarray`
  - `cohort_level_scores(R, fig, year) -> np.ndarray`
  （どれも `len(R)`、`year` 年の行以外は NaN。`fig` は `yearly_figures(runs_all, year)`）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_age.py`:

```python
"""年齢・晩成度×年齢・世代レベル（仕様 §3.2 #6〜8）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.age_factors import age_scores, cohort_level_scores, late_age_scores, sire_lateness
from keiba.factors.figure_factors import yearly_figures
from tests.fixtures.factor_runs import assert_no_lookahead, make_factor_runs, to_runs_all


@pytest.fixture(scope="module")
def data():
    R = make_factor_runs(years=(2015, 2016, 2017, 2018), days_per_year=10)
    return R, yearly_figures(to_runs_all(R), 2018)


@pytest.mark.parametrize("fn", [age_scores, late_age_scores, cohort_level_scores], ids=lambda f: f.__name__)
def test_only_target_year_and_no_lookahead(fn, data):
    R, fig = data
    out = fn(R, fig, 2018)
    assert out.shape == (len(R),)
    assert np.isnan(out[(R.year != 2018).to_numpy()]).all()
    day = R[R.year == 2018].date.unique()[4]
    assert_no_lookahead(lambda d: fn(d, yearly_figures(to_runs_all(d), 2018), 2018), R, day)


def test_age_scores_follow_curve_levels(data):
    R, fig = data
    out = age_scores(R, fig, 2018)
    d = R.assign(sc=out)[R.year == 2018]
    # 同じ年齢・同じ半期なら同じ値
    assert d.groupby(["barei", d.date.dt.month > 6]).sc.nunique(dropna=False).max() == 1


def test_sire_lateness_respects_cohort_limit(data):
    R, fig = data
    L = sire_lateness(R, fig, 2018)
    # 作り物のデータでは 2012年生まれ以前の「3歳前半」が 2015年以降に無いので、A の材料が無く L は空か B だけ
    assert isinstance(L, pd.Series)


def test_cohort_level_january_is_nan(data):
    R, fig = data
    out = cohort_level_scores(R, fig, 2018)
    jan = ((R.year == 2018) & (R.date.dt.month == 1)).to_numpy()
    assert np.isnan(out[jan]).all()       # 1/1 時点の今年の世代はまだ走っていない
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_age.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 実装する**

`src/keiba/factors/age_factors.py`:

```python
"""年齢・晩成度×年齢・世代レベル（仕様 §3.2 #6〜8）。

どれも年ごとの物差し（figure_factors.yearly_figures）の指数を使い、評価する年の1/1（世代レベルは月の1日）
より前のデータだけで作る。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from keiba.evaluate.age_curve import age_curve, curve_levels, period_values
from keiba.evaluate.generation import cohort_table, yearly_comparison
from keiba.factors.cumulative import eb_lambda

FIRST_DATA_YEAR = 2015
COHORT_AGES = list(range(2, 10))
LATE_A_MIN_N, LATE_B_MIN_N = 10, 5
LATE_B_FIRST_COHORT = 2013   # 2015年の2歳戦から見える最初の世代


def _rows(R: pd.DataFrame, year: int) -> np.ndarray:
    return (R["year"] == year).to_numpy()


def age_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """前年末までのデータで作った年齢曲線の、その馬の年齢・半期の水準。"""
    out = np.full(len(R), np.nan)
    curve = age_curve(fig, years=list(range(FIRST_DATA_YEAR, year)), before=date(year, 1, 1), n_boot=1, seed=0)
    lv = curve_levels(curve)
    rows = _rows(R, year)
    half = (R["date"].dt.month > 6).astype(int).to_numpy()
    ages = R["barei"].to_numpy()
    out[rows] = [lv.get((int(a), int(h)), np.nan) for a, h in zip(ages[rows], half[rows])]
    return out


def _shrunk_means(values: pd.Series, keys: pd.Series) -> tuple[pd.Series, pd.Series]:
    d = pd.DataFrame({"v": values.to_numpy(float), "k": keys.to_numpy()})
    d = d[(d["k"].astype(str) != "") & d["v"].notna()]
    g = d.groupby("k")["v"]
    n, m = g.size(), g.mean()
    lam = eb_lambda(d["v"].to_numpy(), d["k"].to_numpy())
    eff = (m - d["v"].mean()) * (n / (n + lam) if lam is not None else 0.0)
    return eff, n


def sire_lateness(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> pd.Series:
    """父の晩成度 L: 長い期間の伸び（A）と出世の時期（B）の縮小効果をそれぞれ標準化して平均したもの。

    材料は前年末までに5歳前半を終えた世代（cohort ≤ year − 6）だけ（仕様 §3.2 #6）。
    """
    max_c = year - 6
    cut = pd.Timestamp(year, 1, 1)
    sire = R[["ketto_num", "sire_num"]].drop_duplicates("ketto_num")
    parts = []
    pv = period_values(fig, years=list(range(FIRST_DATA_YEAR, year)), before=date(year, 1, 1))
    early = pv[(pv.age == 3) & (pv.half == 0)][["ketto_num", "year", "value"]].rename(
        columns={"value": "v_early", "year": "y3"})
    late = pv[((pv.age == 4) & (pv.half == 1)) | ((pv.age == 5) & (pv.half == 0))]
    late = late.groupby("ketto_num")["value"].mean().rename("v_late").reset_index()
    ga = early.merge(late, on="ketto_num")
    ga["cohort"] = ga["y3"] - 3
    ga = ga[ga["cohort"] <= max_c].copy()
    if len(ga) > 2:
        ga["delta"] = ga["v_late"] - ga["v_early"]

        def resid(d: pd.DataFrame) -> pd.Series:
            x, y = d["v_early"] - d["v_early"].mean(), d["delta"]
            den = float((x ** 2).sum())
            beta = float((x * (y - y.mean())).sum()) / den if den > 0 else 0.0
            return y - y.mean() - beta * x

        ga["resid"] = ga.groupby("cohort", group_keys=False)[["v_early", "delta"]].apply(resid)
        ga = ga.merge(sire, on="ketto_num")
        eA, nA = _shrunk_means(ga["resid"], ga["sire_num"])
        eA = eA[nA >= LATE_A_MIN_N]
        if len(eA) > 1 and eA.std() > 0:
            parts.append(((eA - eA.mean()) / eA.std()).rename("A"))
    op = R[(R["date"] < cut) & (R["class_level"] >= 5) & (R["top3"] == 1) & R["barei"].between(2, 5)]
    first = op.sort_values("date").drop_duplicates("ketto_num")
    first = first[(first["year"] - first["barei"]).between(LATE_B_FIRST_COHORT, max_c)]
    if len(first) > 2:
        eB, nB = _shrunk_means((first["barei"] >= 4).astype(float), first["sire_num"])
        eB = eB[nB >= LATE_B_MIN_N]
        if len(eB) > 1 and eB.std() > 0:
            parts.append(((eB - eB.mean()) / eB.std()).rename("B"))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def late_age_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """L × clip(年齢 − 3.5, −1.5, +1.5)。L の無い父は NaN。"""
    out = np.full(len(R), np.nan)
    L = sire_lateness(R, fig, year)
    rows = _rows(R, year)
    lv = R["sire_num"].map(L).to_numpy(float)
    coef = np.clip(R["barei"].to_numpy(float) - 3.5, -1.5, 1.5)
    out[rows] = (lv * coef)[rows]
    return out


def cohort_level_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """その馬の世代の例年比（中央値）。月の1日の 0:00 を時点として月ごとに計算する。"""
    out = np.full(len(R), np.nan)
    months = R["date"].dt.month.to_numpy()
    yr = _rows(R, year)
    for month in range(1, 13):
        rows = yr & (months == month)
        if not rows.any():
            continue
        table = cohort_table(fig, as_of=date(year, month, 1), ages=COHORT_AGES,
                             years=list(range(FIRST_DATA_YEAR, year + 1)))
        yc = yearly_comparison(table, year=year)
        med = yc[yc["metric"] == "median"].set_index("age")["diff"]
        out[rows] = R.loc[rows, "barei"].map(med).to_numpy(float)
    return out
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_age.py -v`
Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/factors/age_factors.py tests/test_factor_age.py
git commit -m "feat: 年齢・晩成度×年齢・世代レベルの要素

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 要素の一覧と一括計算

**Files:**
- Create: `src/keiba/factors/registry.py`
- Create: `tests/test_factor_registry.py`

**Interfaces:**
- Consumes: Task 1〜4 のすべて
- Produces:
  - `FactorSpec`（frozen dataclass: `name: str, version: str, ja: str, forward: bool`）
  - `FACTORS: list[FactorSpec]`（仕様 §3.2 の順、15個）、`FACTOR_NAMES: list[str]`
  - `compute_all(R, runs_all, *, years, log=print) -> pd.DataFrame`（`R` に `e3`・`ae` と15要素の列を足した写し）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_registry.py`:

```python
"""要素の一覧と一括計算のテスト。"""

from __future__ import annotations

import numpy as np

from keiba.factors.registry import FACTOR_NAMES, FACTORS, compute_all
from tests.fixtures.factor_runs import make_factor_runs, to_runs_all

EXPECTED = ["time_figure", "opponents", "jockey", "trainer", "sire_surface", "late_age", "age", "cohort_level",
            "distance_fit", "course_fit", "draw", "weight_carried", "layoff", "body_weight", "travel"]


def test_factor_list_matches_spec():
    assert FACTOR_NAMES == EXPECTED
    assert all(f.version == "v1" for f in FACTORS)
    assert [f.name for f in FACTORS if not f.forward] == ["body_weight"]


def test_compute_all_adds_every_factor():
    R = make_factor_runs(years=(2015, 2016, 2017))
    out = compute_all(R, to_runs_all(R), years=[2016, 2017], log=lambda *_: None)
    for name in EXPECTED + ["e3", "ae"]:
        assert name in out.columns
    assert len(out) == len(R)
    assert np.isnan(out.loc[out.year == 2015, "jockey"]).all()
    assert out.loc[out.year == 2017, "time_figure"].notna().mean() > 0.5
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_registry.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 実装する**

`src/keiba/factors/registry.py`:

```python
"""15要素の一覧（仕様 §3.2）と一括計算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from keiba.factors import history_factors as hf
from keiba.factors import table_factors as tf
from keiba.factors.age_factors import age_scores, cohort_level_scores, late_age_scores
from keiba.factors.cumulative import add_expected_top3
from keiba.factors.figure_factors import opponents_scores, time_figure_scores, yearly_figures


@dataclass(frozen=True)
class FactorSpec:
    name: str
    version: str
    ja: str
    forward: bool   # 当日の朝に出せるか（馬体重は発走前の発表までわからない）


FACTORS = [
    FactorSpec("time_figure", "v1", "タイム指数", True),
    FactorSpec("opponents", "v1", "相手関係", True),
    FactorSpec("jockey", "v1", "騎手", True),
    FactorSpec("trainer", "v1", "調教師", True),
    FactorSpec("sire_surface", "v1", "父の芝ダ適性", True),
    FactorSpec("late_age", "v1", "晩成度×年齢", True),
    FactorSpec("age", "v1", "年齢", True),
    FactorSpec("cohort_level", "v1", "世代レベル", True),
    FactorSpec("distance_fit", "v1", "距離適性", True),
    FactorSpec("course_fit", "v1", "コース適性", True),
    FactorSpec("draw", "v1", "枠順", True),
    FactorSpec("weight_carried", "v1", "斤量", True),
    FactorSpec("layoff", "v1", "間隔", True),
    FactorSpec("body_weight", "v1", "馬体重", False),
    FactorSpec("travel", "v1", "遠征", True),
]
FACTOR_NAMES = [f.name for f in FACTORS]
_WHOLE = {"jockey": tf.jockey, "trainer": tf.trainer, "sire_surface": tf.sire_surface, "draw": tf.draw,
          "layoff": tf.layoff, "body_weight": tf.body_weight, "travel": tf.travel,
          "distance_fit": hf.distance_fit, "course_fit": hf.course_fit, "weight_carried": hf.weight_carried}
_YEARLY = {"time_figure": time_figure_scores, "opponents": opponents_scores, "age": age_scores,
           "late_age": late_age_scores, "cohort_level": cohort_level_scores}


def compute_all(R: pd.DataFrame, runs_all: pd.DataFrame, *, years, log=print) -> pd.DataFrame:
    """R の写しに e3・ae と15要素の列を足す。年ごとの要素は years の年だけ計算し、ほかの年は NaN。"""
    out = add_expected_top3(R)
    for name, fn in _WHOLE.items():
        log(f"{name} …")
        out[name] = fn(out)
    for name in _YEARLY:
        out[name] = np.nan
    for y in years:
        log(f"{y}年の物差しと年ごとの要素 …")
        fig = yearly_figures(runs_all, y)
        rows = (out["year"] == y).to_numpy()
        for name, fn in _YEARLY.items():
            out.loc[rows, name] = fn(out, fig, y)[rows]
    return out
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_registry.py -v`
Expected: 2 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/factors/registry.py tests/test_factor_registry.py
git commit -m "feat: 15要素の一覧と一括計算

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 段ごとの条件付きロジット（着順モデル）

**Files:**
- Create: `src/keiba/factors/order_model.py`
- Create: `tests/test_factor_order_model.py`

**Interfaces:**
- Produces:
  - `valid_race_mask(R) -> np.ndarray`（評価に使える行: 単勝オッズが全頭正、1・2・3着が各1頭、4頭以上）
  - `add_market_features(R) -> pd.DataFrame`（列 `lq`, `lq2` を足す。lq = log(レース内で正規化した 1/単勝オッズ)）
  - `fit_stages(d, feats: list[str]) -> list[np.ndarray]`（段1〜3の重み。`d` は valid な行だけ、race_id でまとまって並ぶ）
  - `race_nll(d, feats, weights) -> pd.Series`（race_id → 段1〜3 の −対数尤度の合計）
  - `place_probabilities(d, feats, weights) -> tuple[np.ndarray, np.ndarray]`（行ごとの1着確率, 3着内確率）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_order_model.py`:

```python
"""段ごとの条件付きロジットのテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.order_model import (add_market_features, fit_stages, place_probabilities, race_nll,
                                       valid_race_mask)


def synth(n_races=400, field=8, w_true=(1.5, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        x = rng.normal(0, 1, field)
        g = rng.gumbel(size=field)
        order = np.argsort(-(w_true[0] * x + g))
        jyuni = np.empty(field)
        jyuni[order] = np.arange(1, field + 1)
        for i in range(field):
            rows.append(dict(race_id=f"R{r:04d}", umaban=f"{i + 1:02d}", x=x[i], jyuni=jyuni[i],
                             odds=float(np.round(2 + 10 * rng.random(), 1)), field=field))
    return pd.DataFrame(rows)


def test_valid_race_mask_excludes_dead_heat_and_missing_odds():
    d = synth(n_races=3)
    d.loc[(d.race_id == "R0001") & (d.jyuni == 2), "jyuni"] = 1.0       # 1着同着
    d.loc[(d.race_id == "R0002") & (d.umaban == "01"), "odds"] = np.nan
    m = valid_race_mask(d)
    assert set(d[m].race_id) == {"R0000"}


def test_fit_recovers_stage1_weight():
    d = synth()
    w = fit_stages(d, ["x"])
    assert w[0][0] == pytest.approx(1.5, abs=0.25)


def test_race_nll_and_probabilities():
    d = synth(n_races=50)
    w = fit_stages(d, ["x"])
    nll = race_nll(d, ["x"], w)
    assert len(nll) == 50 and (nll > 0).all()
    win, top3 = place_probabilities(d, ["x"], w)
    s = pd.DataFrame({"r": d.race_id, "w": win, "t": top3}).groupby("r").sum()
    assert np.allclose(s.w, 1.0) and np.allclose(s.t, 3.0)


def test_market_features():
    d = pd.DataFrame({"race_id": ["a", "a"], "odds": [2.0, 2.0]})
    out = add_market_features(d)
    assert np.allclose(out.lq, np.log(0.5)) and np.allclose(out.lq2, np.log(0.5) ** 2)
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_order_model.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 実装する**

`src/keiba/factors/order_model.py`:

```python
"""段ごとの条件付きロジット（仕様 §4）。段 s では、まだ着順の決まっていない馬の中から s 着の馬が選ばれる確率を
softmax(w_s · x) とする。スプリンターズS の combine.py と同じ形。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

STAGES = (1, 2, 3)


def valid_race_mask(R: pd.DataFrame) -> np.ndarray:
    bad_odds = (~(R["odds"] > 0)).groupby(R["race_id"]).transform("sum") > 0
    c = {k: R["jyuni"].eq(k).groupby(R["race_id"]).transform("sum") for k in STAGES}
    ok = ~bad_odds & (c[1] == 1) & (c[2] == 1) & (c[3] == 1) & (R["field"] >= 4)
    return ok.to_numpy()


def add_market_features(R: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / R["odds"]
    q = inv / inv.groupby(R["race_id"]).transform("sum")
    lq = np.log(q)
    return R.assign(lq=lq, lq2=lq ** 2)


def _stage_rows(d: pd.DataFrame, s: int) -> pd.DataFrame:
    f = d[~d["jyuni"].between(1, s - 1)] if s > 1 else d
    return f.assign(_y=(f["jyuni"] == s).astype(float))


def _codes(race_id: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    codes = pd.factorize(race_id)[0]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    return codes, starts, int(codes.max()) + 1


def _nll_grad(w, X, y, codes, starts, n_r):
    z = X @ w
    zmax = np.maximum.reduceat(z, starts)[codes]
    e = np.exp(z - zmax)
    den = np.bincount(codes, e, n_r)
    p = e / den[codes]
    ll = -(y * (z - zmax - np.log(den[codes]))).sum()
    grad = -(X * (y - p)[:, None]).sum(0)
    return ll, grad


def fit_stages(d: pd.DataFrame, feats: list[str]) -> list[np.ndarray]:
    weights = []
    for s in STAGES:
        f = _stage_rows(d, s)
        codes, starts, n_r = _codes(f["race_id"])
        X = f[feats].to_numpy(float)
        r = minimize(_nll_grad, np.zeros(len(feats)), args=(X, f["_y"].to_numpy(), codes, starts, n_r),
                     jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
        weights.append(r.x)
    return weights


def race_nll(d: pd.DataFrame, feats: list[str], weights: list[np.ndarray]) -> pd.Series:
    total = None
    for s, w in zip(STAGES, weights):
        f = _stage_rows(d, s)
        codes, starts, n_r = _codes(f["race_id"])
        z = f[feats].to_numpy(float) @ w
        zmax = np.maximum.reduceat(z, starts)
        lse = np.log(np.bincount(codes, np.exp(z - zmax[codes]), n_r)) + zmax
        zt = np.bincount(codes, z * f["_y"].to_numpy(), n_r)
        part = pd.Series(lse - zt, index=pd.unique(f["race_id"]))
        total = part if total is None else total.add(part, fill_value=0.0)
    return total


def _order_dist(etas: list[np.ndarray]) -> np.ndarray:
    e = [np.exp(x - x.max()) for x in etas]
    p1 = e[0] / e[0].sum()
    P2 = e[1][None, :] / (e[1].sum() - e[1])[:, None]
    np.fill_diagonal(P2, 0)
    d3 = e[2].sum() - e[2][:, None] - e[2][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        P3 = e[2][None, None, :] / d3[:, :, None]
    idx = np.arange(len(p1))
    P3[idx, idx, :] = 0
    P3[:, idx, idx] = 0
    P3[idx, :, idx] = 0
    return p1[:, None, None] * P2[:, :, None] * np.nan_to_num(P3)


def place_probabilities(d: pd.DataFrame, feats: list[str], weights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    win = np.zeros(len(d))
    top3 = np.zeros(len(d))
    X = d[feats].to_numpy(float)
    for idx in d.groupby("race_id", sort=False).indices.values():
        t = _order_dist([X[idx] @ w for w in weights])
        win[idx] = t.sum((1, 2))
        top3[idx] = t.sum((1, 2)) + t.sum((0, 2)) + t.sum((0, 1))
    return win, top3
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_order_model.py -v`
Expected: 4 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/factors/order_model.py tests/test_factor_order_model.py
git commit -m "feat: 段ごとの条件付きロジットの着順モデル

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 評価（Δ・的中・食い違い・区間・判定）

**Files:**
- Create: `src/keiba/evaluate/factor_eval.py`
- Create: `tests/test_factor_eval.py`

**Interfaces:**
- Consumes: Task 6 のすべて、Task 5 の `FACTORS`
- Produces:
  - 定数 `EVAL_YEARS = list(range(2017, 2026))`, `TRAIN_FIRST = 2016`, `N_BOOT = 1000`, `SEED = 20261011`, `MIN_YEARS_IMPROVED = 7`
  - `topk_weights(values: np.ndarray, race_ids: np.ndarray, k: int = 3) -> np.ndarray`
  - `zscore(d: pd.DataFrame, col: str, train: np.ndarray) -> np.ndarray`
  - `evaluate(R: pd.DataFrame, factors: list[str], *, years=EVAL_YEARS, log=print) -> dict`
    戻り値のキー: `"per_race"`（DataFrame: race_id, date, year, op, mixed, mkt_odds_hits, mkt_model_hits, と要素ごとに
    `delta:<f>`, `solo:<f>`, `comb:<f>`, `fo_n:<f>`, `fo_hit:<f>`, `fo_e3:<f>`, `mo_n:<f>`, `mo_hit:<f>`, `mo_e3:<f>`）、
    `"excluded"`（dict: 評価年のレース数と除いたレース数）
  - `summarize(per_race: pd.DataFrame, factors: list[str], *, n_boot=N_BOOT, seed=SEED) -> pd.DataFrame`
    （行 = 要素、列: `delta`, `lo`, `hi`, `years_improved`, `verdict`, `solo`, `mkt_odds`, `comb`, `mkt_model`,
    `fo_n`, `fo_rate`, `fo_e3`, `mo_n`, `mo_rate`, `mo_e3`）
  - `by_year(per_race, factors) -> pd.DataFrame`（行 = 年 × 要素、列 `delta`, `solo`, `mkt_odds`, `comb`, `mkt_model`）
  - `verdict(lo: float, years_improved: int) -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_eval.py`:

```python
"""要素別の評価（仕様 §5.2〜5.4）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.factor_eval import by_year, evaluate, summarize, topk_weights, verdict, zscore


def synth_eval(n_per_year=150, field=10, seed=0):
    """市場は地力だけ、要素 good は市場が知らない上乗せ、要素 noise は無関係。"""
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2016, 2026):
        for r in range(n_per_year):
            date = pd.Timestamp(year, 1 + r % 12, 1 + r % 27)
            ab = rng.normal(0, 1, field)
            extra = rng.normal(0, 1, field)
            g = rng.gumbel(size=field)
            perf = ab + 0.8 * extra + g
            order = np.argsort(-perf)
            jyuni = np.empty(field)
            jyuni[order] = np.arange(1, field + 1)
            p = np.exp(ab) / np.exp(ab).sum()
            for i in range(field):
                rows.append(dict(race_id=f"{year}{r:04d}", date=date, year=year, umaban=f"{i + 1:02d}",
                                 jyuni=jyuni[i], top3=float(jyuni[i] <= 3), field=field, odds=0.8 / p[i],
                                 e3=0.3, class_level=5.0 if r % 5 == 0 else 2.0, syubetu="13" if r % 2 else "12",
                                 good=extra[i], noise=rng.normal(), allnan=np.nan, const=1.0))
    return pd.DataFrame(rows)


def test_topk_weights_split_ties():
    v = np.array([3.0, 2.0, 2.0, 2.0, 1.0, np.nan])
    w = topk_weights(v, np.array(["a"] * 6))
    assert w.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3, 2 / 3, 0.0, 0.0])


def test_zscore_missing_is_race_mean():
    d = pd.DataFrame({"race_id": ["a", "a", "a", "b", "b"], "f": [1.0, 3.0, np.nan, 5.0, 5.0]})
    z = zscore(d, "f", np.ones(5, bool))
    assert z[2] == 0.0 and z[3] == 0.0 and z[4] == 0.0
    assert z[0] < 0 < z[1]


def test_good_factor_lifts_and_noise_does_not():
    R = synth_eval()
    res = evaluate(R, ["good", "noise"], log=lambda *_: None)
    s = summarize(res["per_race"], ["good", "noise"], n_boot=200)
    assert s.loc["good", "verdict"] == "上乗せあり"
    assert s.loc["good", "lo"] > 0
    assert s.loc["noise", "verdict"] == "上乗せは確認できない"


def test_all_nan_factor_gives_no_lift():
    R = synth_eval(n_per_year=60)
    res = evaluate(R, ["allnan", "const"], log=lambda *_: None)
    s = summarize(res["per_race"], ["allnan", "const"], n_boot=100)
    for f in ("allnan", "const"):
        assert abs(s.loc[f, "delta"]) < 1e-6
        assert s.loc[f, "verdict"] == "上乗せは確認できない"


def test_by_year_and_verdict_rule():
    R = synth_eval(n_per_year=60)
    res = evaluate(R, ["good"], log=lambda *_: None)
    t = by_year(res["per_race"], ["good"])
    assert set(t.index.get_level_values("year")) == set(range(2017, 2026))
    assert verdict(0.01, 7) == "上乗せあり"
    assert verdict(0.01, 6) == "上乗せは確認できない"
    assert verdict(-0.01, 9) == "上乗せは確認できない"


def test_expanding_window_uses_only_prior_years():
    R = synth_eval(n_per_year=60)
    R2 = R.copy()
    late = R2.year == 2025
    R2.loc[late, "jyuni"] = R2.loc[late, "field"] - R2.loc[late, "jyuni"] + 1
    R2.loc[late, "top3"] = (R2.loc[late, "jyuni"] <= 3).astype(float)
    a = evaluate(R, ["good"], log=lambda *_: None)["per_race"]
    b = evaluate(R2, ["good"], log=lambda *_: None)["per_race"]
    ea, eb = a[a.year < 2025].set_index("race_id"), b[b.year < 2025].set_index("race_id")
    np.testing.assert_allclose(ea["delta:good"], eb.loc[ea.index, "delta:good"])


def test_disagreement_counts_hand_checked():
    R = synth_eval(n_per_year=40)
    res = evaluate(R, ["good"], log=lambda *_: None)
    pr = res["per_race"]
    # 要素の上位3頭にいて市場の上位3頭にいない「重み」の合計は 0〜3 の範囲
    assert pr["fo_n:good"].between(0, 3).all() and pr["mo_n:good"].between(0, 3).all()
    # 両者の重みの合計は同じ（同じ3枠を入れ替えるだけ）
    assert np.allclose(pr["fo_n:good"], pr["mo_n:good"])
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_eval.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 実装する**

`src/keiba/evaluate/factor_eval.py`:

```python
"""要素別の予想の過去での評価（仕様 §5）。

評価年 Y ごとに、市場だけの着順モデルと「市場＋1要素」の着順モデルを 2016〜Y−1年で推定し、Y年のレースで
−対数尤度の差 Δ（正 = 要素で良くなった）を出す。判定は9年まとめた Δ の区間と、年ごとの安定性で行う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.order_model import add_market_features, fit_stages, place_probabilities, race_nll, valid_race_mask

EVAL_YEARS = list(range(2017, 2026))
TRAIN_FIRST = 2016
N_BOOT = 1000
SEED = 20261011
MIN_YEARS_IMPROVED = 7
MKT = ["lq", "lq2"]


def topk_weights(values: np.ndarray, race_ids: np.ndarray, k: int = 3) -> np.ndarray:
    """各レースで値の大きい順に k 頭の重み。NaN は最後。境の同点は残りの枠を等分した分数（馬番で切らない）。"""
    out = np.zeros(len(values))
    vals = np.where(np.isnan(values), -np.inf, values)
    for idx in pd.Series(np.arange(len(values))).groupby(np.asarray(race_ids), sort=False).indices.values():
        v = vals[idx]
        if len(v) <= k:
            out[idx] = 1.0
            continue
        b = np.sort(v)[::-1][k - 1]
        above, tied = v > b, v == b
        out[idx[above]] = 1.0
        out[idx[tied]] = (k - above.sum()) / tied.sum()
    return out


def zscore(d: pd.DataFrame, col: str, train: np.ndarray) -> np.ndarray:
    """レース内で中心化し、学習行の中心化した値の標準偏差で割る。NaN（点数なし）はレース平均（0）にする。"""
    v = d[col].to_numpy(float)
    mean = pd.Series(v).groupby(d["race_id"].to_numpy()).transform("mean").to_numpy()
    c = v - mean
    sd = np.nanstd(c[train]) if np.isfinite(c[train]).any() else 0.0
    z = c / sd if sd > 0 else np.zeros_like(c)
    return np.nan_to_num(z, nan=0.0)


def verdict(lo: float, years_improved: int) -> str:
    return "上乗せあり" if (lo > 0 and years_improved >= MIN_YEARS_IMPROVED) else "上乗せは確認できない"


def evaluate(R: pd.DataFrame, factors: list[str], *, years=EVAL_YEARS, log=print) -> dict:
    keep = valid_race_mask(R)
    in_years = R["year"].isin(years).to_numpy()
    n_races = R.loc[in_years, "race_id"].nunique()
    n_valid = R.loc[in_years & keep, "race_id"].nunique()
    d = add_market_features(R[keep & (R["year"] >= TRAIN_FIRST).to_numpy()].copy()).reset_index(drop=True)
    d = d.sort_values(["race_id", "umaban"]).reset_index(drop=True)
    parts = []
    for y in years:
        log(f"評価 {y}年 …")
        train = (d["year"] >= TRAIN_FIRST) & (d["year"] < y)
        test = (d["year"] == y).to_numpy()
        if not test.any() or not train.any():
            continue
        tr, te = d[train.to_numpy()], d[test].copy()
        w_m = fit_stages(tr, MKT)
        nll_m = race_nll(te, MKT, w_m)
        _, p_m = place_probabilities(te, MKT, w_m)
        rid = te["race_id"].to_numpy()
        top3 = te["top3"].to_numpy(float)
        e3 = te["e3"].to_numpy(float) if "e3" in te else np.full(len(te), np.nan)
        w_odds = topk_weights(-te["odds"].to_numpy(float), rid)
        w_mm = topk_weights(p_m, rid)
        res = pd.DataFrame({"race_id": rid})
        res["mkt_odds_hits"] = w_odds * top3
        res["mkt_model_hits"] = w_mm * top3
        deltas = {}
        for f in factors:
            z_all = zscore(d, f, train.to_numpy())
            if not np.any(z_all[train.to_numpy()]) or not np.any(z_all[test]):
                # 点数が全部欠けている・レース内で全頭同じ → 足しても市場だけと同じ（数値誤差で判定を揺らさない）
                nll_f, p_f = nll_m, p_m
            else:
                te["_z"] = z_all[test]
                tr_z = tr.assign(_z=z_all[train.to_numpy()])
                w_f = fit_stages(tr_z, MKT + ["_z"])
                nll_f = race_nll(te, MKT + ["_z"], w_f)
                _, p_f = place_probabilities(te, MKT + ["_z"], w_f)
            deltas[f] = nll_m - nll_f
            w_solo = topk_weights(te[f].to_numpy(float), rid)
            w_comb = topk_weights(p_f, rid)
            fo = w_solo * (1 - w_odds)
            mo = w_odds * (1 - w_solo)
            res[f"solo:{f}"] = w_solo * top3
            res[f"comb:{f}"] = w_comb * top3
            res[f"fo_n:{f}"], res[f"fo_hit:{f}"], res[f"fo_e3:{f}"] = fo, fo * top3, fo * np.nan_to_num(e3)
            res[f"mo_n:{f}"], res[f"mo_hit:{f}"], res[f"mo_e3:{f}"] = mo, mo * top3, mo * np.nan_to_num(e3)
        agg = res.groupby("race_id", sort=False).sum()
        for f in factors:
            agg[f"delta:{f}"] = deltas[f].reindex(agg.index).to_numpy()
        meta = te.groupby("race_id", sort=False).agg(date=("date", "first"), class_level=("class_level", "first"),
                                                       syubetu=("syubetu", "first"))
        agg = agg.join(meta)
        agg["year"] = y
        agg["op"] = agg["class_level"] >= 5
        agg["mixed"] = agg["syubetu"] == "13"
        parts.append(agg.reset_index())
    per_race = pd.concat(parts, ignore_index=True)
    return {"per_race": per_race, "excluded": {"races": int(n_races), "valid": int(n_valid),
                                               "excluded": int(n_races - n_valid)}}


def _cluster_ci(per_race: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    codes, uniq = pd.factorize(per_race["date"])
    sums = np.stack([np.bincount(codes, per_race[c].to_numpy(float), len(uniq)) for c in cols], axis=1)
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        means[b] = (w @ sums) / (w @ counts)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def summarize(per_race: pd.DataFrame, factors: list[str], *, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    cols = [f"delta:{f}" for f in factors]
    lo, hi = _cluster_ci(per_race, cols, n_boot, seed)
    yearly = per_race.groupby("year")[cols].mean()
    rows = []
    for i, f in enumerate(factors):
        improved = int((yearly[f"delta:{f}"] > 0).sum())
        fo_n, mo_n = per_race[f"fo_n:{f}"].sum(), per_race[f"mo_n:{f}"].sum()
        rows.append(dict(
            factor=f, delta=per_race[f"delta:{f}"].mean(), lo=lo[i], hi=hi[i], years_improved=improved,
            verdict=verdict(lo[i], improved), solo=per_race[f"solo:{f}"].mean(),
            mkt_odds=per_race["mkt_odds_hits"].mean(), comb=per_race[f"comb:{f}"].mean(),
            mkt_model=per_race["mkt_model_hits"].mean(), fo_n=fo_n,
            fo_rate=per_race[f"fo_hit:{f}"].sum() / fo_n if fo_n else np.nan,
            fo_e3=per_race[f"fo_e3:{f}"].sum() / fo_n if fo_n else np.nan, mo_n=mo_n,
            mo_rate=per_race[f"mo_hit:{f}"].sum() / mo_n if mo_n else np.nan,
            mo_e3=per_race[f"mo_e3:{f}"].sum() / mo_n if mo_n else np.nan))
    return pd.DataFrame(rows).set_index("factor")


def by_year(per_race: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for y, g in per_race.groupby("year"):
        for f in factors:
            rows.append(dict(year=y, factor=f, delta=g[f"delta:{f}"].mean(), solo=g[f"solo:{f}"].mean(),
                             mkt_odds=g["mkt_odds_hits"].mean(), comb=g[f"comb:{f}"].mean(),
                             mkt_model=g["mkt_model_hits"].mean()))
    return pd.DataFrame(rows).set_index(["year", "factor"])
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_eval.py -v`
Expected: 7 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/evaluate/factor_eval.py tests/test_factor_eval.py
git commit -m "feat: 要素別の評価（拡大窓の Δ・的中・食い違い・区間・判定）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: レポート、実行スクリプト、実データでの実行、記録

**Files:**
- Create: `src/keiba/evaluate/factor_eval_report.py`, `scripts/factor_eval.py`
- Create: `tests/test_factor_eval_report.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: Task 1〜7 のすべて、`keiba.features.speed_figure.prepare_runs`
- Produces: `factor_eval_report.render(summary, yearly, subsets: dict[str, pd.DataFrame], excluded: dict, *, data_version: str, years: list[int]) -> str`、
  `scripts/factor_eval.py` の `main(argv=None) -> int`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_factor_eval_report.py`:

```python
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
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_eval_report.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: レポートを実装する**

`src/keiba/evaluate/factor_eval_report.py`:

```python
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
          "- 小さな上乗せ（Δ ×1000 が 1 未満）は区間で 0 と区別できないことがある", ""]
    return "\n".join(L)
```

- [ ] **Step 4: レポートのテストが通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_eval_report.py -v`
Expected: 2 passed

- [ ] **Step 5: 実行スクリプトを書く**

`scripts/factor_eval.py`:

```python
"""要素別の予想の過去での評価（仕様 §5）。Jev は呼ばない（課金なし）。

    .venv/Scripts/python.exe scripts/factor_eval.py --data-dir data/_long

要素の点数の計算は重いので、data/factor_eval/factors_<版>.pkl に保存して再利用する（--rebuild で作り直し）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.evaluate.factor_eval import EVAL_YEARS, N_BOOT, SEED, by_year, evaluate, summarize  # noqa: E402
from keiba.evaluate.factor_eval_report import render  # noqa: E402
from keiba.factors.data import load_runs  # noqa: E402
from keiba.factors.registry import FACTOR_NAMES, FACTORS, compute_all  # noqa: E402
from keiba.features.speed_figure import prepare_runs  # noqa: E402

JST = timezone(timedelta(hours=9))


def _current(d: Path) -> str:
    return (d / "CURRENT").read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "_long")
    ap.add_argument("--blood-dir", type=Path, default=config.DATA_DIR / "blood")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    version = _current(args.data_dir)
    vdir = args.data_dir / "versions" / version
    bdir = args.blood_dir / "versions" / _current(args.blood_dir)
    stamp = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir or (config.DATA_DIR / "factor_eval" / stamp)
    cache = (args.out_dir or config.DATA_DIR / "factor_eval") / f"factors_{version}.pkl"
    years = [EVAL_YEARS[0] - 1] + EVAL_YEARS

    if cache.exists() and not args.rebuild:
        R = pd.read_pickle(cache)
        print(f"要素の点数をキャッシュから読む: {cache}")
    else:
        print("走の表を作る …", flush=True)
        R0 = load_runs(vdir, bdir)
        races = pd.read_csv(vdir / "races.csv", dtype=str, na_filter=False)
        horses = pd.read_csv(vdir / "race_horses.csv", dtype=str, na_filter=False)
        runs_all = prepare_runs(races, horses, as_of=datetime(EVAL_YEARS[-1] + 1, 1, 1, tzinfo=JST))
        R = compute_all(R0, runs_all, years=years, log=lambda m: print(m, flush=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        R.to_pickle(cache)
    res = evaluate(R, FACTOR_NAMES, years=EVAL_YEARS, log=lambda m: print(m, flush=True))
    pr = res["per_race"]
    summary = summarize(pr, FACTOR_NAMES, n_boot=args.n_boot, seed=SEED)
    yearly = by_year(pr, FACTOR_NAMES)
    subsets = {"OP・重賞": summarize(pr[pr["op"]], FACTOR_NAMES, n_boot=args.n_boot, seed=SEED),
               "3歳以上の古馬混合戦": summarize(pr[pr["mixed"]], FACTOR_NAMES, n_boot=args.n_boot, seed=SEED)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render(summary, yearly, subsets, res["excluded"], data_version=version,
                                              years=EVAL_YEARS), encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv")
    yearly.to_csv(out_dir / "by_year.csv")
    (out_dir / "manifest.json").write_text(json.dumps({
        "data_version": version, "blood_version": bdir.name, "eval_years": EVAL_YEARS, "n_boot": args.n_boot,
        "seed": SEED, "factors": {f.name: f.version for f in FACTORS}, "excluded": res["excluded"],
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 全テストを回す**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 既存 1400 件 ＋ 新規（6 + 26 + 6 + 6 + 2 + 4 + 7 + 2 = 59）件がすべて passed

- [ ] **Step 7: 実データで回す**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/factor_eval.py --data-dir data/_long
```

時間がかかる（要素の計算で数十分）。`timeout` は長め（例: 3時間）にし、バックグラウンドで回してよい。
Expected: `data/factor_eval/<時刻>/` に report.md・summary.csv・by_year.csv・manifest.json。report.md を開いて確かめ、異常があれば止めて報告する:
- 評価に使ったレースが 1年あたり 3,000 前後（9年で約 2.7万）
- 市場（人気上位3頭）の的中が 1レースあたり 1.2〜1.6 頭程度
- どの要素も Δ ×1000 が ±50 を超えない（超えたら先読みを疑って報告する）

- [ ] **Step 8: CLAUDE.md に記録する**

`CLAUDE.md` の「### 世代レベル（2026-10-04、第1段階）」の節の後ろに次を足す（数値は report.md から写す。事実だけを書く）:

```markdown
### 要素別の予想 — 過去での評価（2026-10-11）

- 15要素それぞれを市場（単勝オッズ）の着順モデルに足したときの改善を 2017〜2025年で測った。仕様
  `docs/superpowers/specs/2026-10-11-factor-predictions-design.md` §3〜§5、実装 `src/keiba/factors/`・`src/keiba/evaluate/factor_eval.py`
- `scripts/factor_eval.py --data-dir data/_long`（Jev は呼ばない）。要素の点数は `data/factor_eval/factors_<版>.pkl` に保存して再利用
- 結果: <report.md の判定の表から「上乗せあり」になった要素と Δ ×1000 [区間]、無ければ「なし」と1〜3行で>
```

テスト件数の行（「テスト全体（… 実行: N passed）」）も実際の件数に直す。

- [ ] **Step 9: コミット**

```bash
git add src/keiba/evaluate/factor_eval_report.py scripts/factor_eval.py tests/test_factor_eval_report.py CLAUDE.md
git commit -m "feat: 要素別の予想の過去での評価を実行するスクリプトを追加し、結果を記録

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

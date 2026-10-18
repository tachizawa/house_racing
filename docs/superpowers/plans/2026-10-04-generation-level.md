# 世代レベル（タイム指数）第1段階 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 走破タイムから全年齢共通のタイム指数を作り、世代（生まれ年）ごとの強さ・世代の歩み・年齢曲線を
レポートにする単体の指標を作る（予想への組み込みはしない）。

**Architecture:** `speed_figure.py` が古馬戦の1着だけから基準タイム・クラス差・馬場差（経験ベイズで縮小）を
推定し、全走に指数を付ける。`age_curve.py` が同じ馬の半年ごとの変化から年齢曲線を、`generation.py` が
世代の数値・例年比・歩み・見立ての照合・開催日クラスタのブートストラップを出す。`generation_report.py` が
Markdown を組み立て、`scripts/generation_level.py` が版を読んで全部を呼ぶ。Jev は呼ばない。

**Tech Stack:** Python 3.13（`.venv`）、numpy、pandas、pytest。新しい依存は足さない。

**Spec:** `docs/superpowers/specs/2026-10-04-generation-level-design.md`（拘束力のある仕様。実装者は必ず読む）

## Global Constraints

- コマンドはリポジトリ直下で `.venv/Scripts/python.exe -m pytest ...` のように `.venv` の python を明示する
- `pytest.ini` が `pythonpath = src` なので `from keiba...` で import できる。テストから scripts を import するときは
  `sys.path.insert(0, <repo>/scripts)`（`tests/test_update_data.py` と同じ方法）
- 新しい依存は足さない（numpy / pandas だけ。scipy も使わない）
- **Jev を呼ばない。課金が発生するコードを書かない**
- 既存の予想・Jev・版（`FEATURE_VERSION` / `PROMPT_VERSION` / `RECORD_SCHEMA_VERSION` / `DOSSIER_VERSION`）は変更しない
- 生のデータ（`data/` 以下）を書き換えない。出力は `data/generation/<as_of>/` だけ（`/data/` は `.gitignore` 済み）
- 生データを読む公開関数（`prepare_runs` / `compute_figures`）は `as_of` をキーワード専用・既定値なしで取る。
  `as_of` はタイムゾーン付き。発走が `as_of` **より前**（等号を含まない）のレースだけを使う
- 古馬戦 = `syubetu_cd` が `"13"`（3歳以上）か `"14"`（4歳以上）。2歳戦 `"11"`、3歳戦 `"12"`
- 指数の単位は 1600m 相当の秒（`(s + v − time_sec / (kyori_m/1000)) × 1.6`）。正ほど速い
- 世代の数値は `top10` / `top50` / `median`。見立ての組は (4,5)・(5,3)・(4,3)（左が強いと見立てた側）
- 2歳は「今いる世代の並び」（§5.3）に入れない
- 乱数の種の既定は 20261004、ブートストラップの既定は 200 回
- docstring・コメント・レポートは日本語。周りのコードと同じく「なぜ」を書く
- コミットメッセージは既存の形式（`feat: ...` / `docs: ...`、日本語）で、末尾に
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける

## Review Focus

1. **2歳戦にしか無いコース・距離（キー）** — 実データでは新馬・2歳戦だけの距離がある。指数が欠損になる走の件数と割合が
   レポートに必ず出ること（黙って捨てない）。→ Task 6 の `test_report_shows_missing_standard_count`
2. **時点がシーズン序盤で世代の頭数が足りない**（例: 2月の時点の2歳は0頭、3歳は10頭未満）— `top10` などが欠損になり、
   レポートは「—」を出して落ちないこと。区間も「—」。→ Task 6 の `test_report_renders_missing_values_as_dash`
3. **芝かダートの一方で年齢曲線が作れない**（データが少ない・期間がつながらない）— `age_curve` が空の表を返し、
   レポートは「データ不足」と書いて落ちないこと。→ Task 3 の `test_age_curve_empty_when_no_changes`、
   Task 6 の `test_report_handles_empty_curve`
4. **1着同着の古馬戦** — 同じレースの1着が2行あっても推定に二重に入らないこと。
   → Task 2 の `test_dead_heat_winner_counted_once`
5. **manifest に `fromdate` が無い版**（テスト用の版・古い版）— 比べてよい年をデータの最初の日から決めて動くこと。
   → Task 7 の `test_coverage_start_falls_back_to_first_run`

---

## File Structure

| ファイル | 責務 |
|---|---|
| Create `src/keiba/features/speed_figure.py` | §3。対象の走の抽出、物差しの推定（`FigureModel`）、指数の付与、再現性 |
| Create `src/keiba/evaluate/age_curve.py` | §6。期間の値、期間の変化、年齢曲線と区間、全盛期と落ち幅、半期の判定 |
| Create `src/keiba/evaluate/generation.py` | §4・§5。世代の数値、例年比、歩み、並び、見立ての照合、開催日ブートストラップ |
| Create `src/keiba/evaluate/generation_report.py` | §8 の `report.md` の組み立て（計算はしない） |
| Create `scripts/generation_level.py` | 実行用。版を読み、上を呼び、`data/generation/<as_of>/` に書く |
| Create `tests/fixtures/figure_runs.py` | 作り物の「走」の表（`prepare_runs` の出力と同じ形） |
| Create `tests/test_speed_figure.py` / `tests/test_age_curve.py` / `tests/test_generation.py` / `tests/test_generation_report.py` / `tests/test_generation_level_script.py` | 各部品のテスト |
| Modify `docs/superpowers/specs/2026-10-04-generation-level-design.md` | 計画を立てる中で分かった3点の訂正（Task 1） |
| Modify `CLAUDE.md` | 実データで回した結果と使い方（Task 7） |

依存の向き: `generation_report` → `generation` → (`age_curve`, `speed_figure`)。`age_curve` → なし（指数の表だけ受け取る）。

---

### Task 1: 仕様の訂正と、対象の走の抽出（`prepare_runs`）

計画を立てる中で分かった仕様の3点を先に直す。

1. 実データの manifest の `coverage.first_hasso_at` は **1989-04-01**（訂正再配信の古いレースが混じる）なので、
   「比べてよい年」をそれで決めると 2021年が入ってしまう。**manifest の `fromdate`**（取得開始日）で決め、
   無ければ対象の走の最初の日で決める
2. 照合の判定に「判定できない」（区間か差が欠損のとき）を足す
3. ブートストラップでは年齢曲線の水準を固定する（年齢分を差し引いた差の区間に曲線の誤差は入らない）

**Files:**
- Modify: `docs/superpowers/specs/2026-10-04-generation-level-design.md`
- Create: `src/keiba/features/speed_figure.py`
- Create: `tests/test_speed_figure.py`

**Interfaces:**
- Produces: `prepare_runs(races: pd.DataFrame, race_horses: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame`。
  列は `RUN_COLUMNS` = `race_id, umaban, ketto_num, day(str YYYYMMDD), date(datetime64), year(int), jyo_cd, track_cd,
  track_kind, kyori_m(int), syubetu_cd, class_level(float, 欠損は NaN), time_sec(float), rank(float), barei(int), cohort(int)`。
  定数 `JRA_JYO`, `FLAT`, `OLDER_SYUBETU`, `KEY`, `FACE`, `RUN_COLUMNS`

- [ ] **Step 1: 仕様を直す**

`docs/superpowers/specs/2026-10-04-generation-level-design.md` の §2 の表の「比べてよい年」の行を次に置き換える:

```markdown
| 比べてよい年 | 版の取得開始日（manifest の `fromdate`。無ければ対象の走の最初の日）がその年の 1/1 以前である年。今の5年版は `fromdate` が 2021-09-01 なので 2021年は使わない。manifest の `coverage.first_hasso_at` は訂正再配信の古いレース（1989年など）が混じるので使わない |
```

§5.3 の「照合:」の箇条を次に置き換える:

```markdown
- 照合: 組ごとに、区間が 0 を含まず見立ての向き →「見立てと合う」、区間が 0 を含まず逆向き →「見立てと逆」、
  区間が 0 を含む →「差があるとは言えない」、区間か差が欠損（頭数不足など）→「判定できない」。
  `top10` / `top50` / `median` で結論が割れたら、割れたことをそのまま書く
```

§5.5 の末尾に1文足す:

```markdown
年齢分を差し引いた差（§5.3）の区間では、年齢曲線の水準は点推定に固定する（曲線の誤差は区間に入らない）。
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_speed_figure.py` を作る:

```python
"""タイム指数（仕様 §3）のテスト。

最優先は2つ。as_of より後のレースが混ざらないこと（Task 1）と、2歳戦の遅さが馬場差に
吸収されないこと（Task 2、§3.4）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from keiba.features.speed_figure import RUN_COLUMNS, prepare_runs

JST = timezone(timedelta(hours=9))


def jst(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=JST)


def race_row(race_id, *, hasso_at="2025-01-05T15:00:00+09:00", jyo_cd="05", track_cd="24",
             track_kind="turf", kyori_m="1600", syubetu_cd="13", class_level="2",
             data_kubun="7", make_date="20250106") -> dict:
    return dict(race_id=race_id, hasso_at=hasso_at, jyo_cd=jyo_cd, track_cd=track_cd,
                track_kind=track_kind, kyori_m=kyori_m, syubetu_cd=syubetu_cd,
                class_level=class_level, data_kubun=data_kubun, make_date=make_date)


def horse_row(race_id, umaban="01", *, ketto_num="H1", barei="04", time="96.0", jyuni="1",
              ijyo="0", data_kubun="7", make_date="20250106") -> dict:
    return dict(race_id=race_id, umaban=umaban, ketto_num=ketto_num, barei=barei,
                result_time_sec=time, result_kakutei_jyuni=jyuni, result_ijyo_cd=ijyo,
                data_kubun=data_kubun, make_date=make_date)


def frames(races, horses):
    return pd.DataFrame(races, dtype=str), pd.DataFrame(horses, dtype=str)


def test_prepare_runs_columns_and_cohort():
    races, horses = frames([race_row("2025010505010101")], [horse_row("2025010505010101")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert list(runs.columns) == RUN_COLUMNS
    row = runs.iloc[0]
    assert row.day == "20250105" and row.year == 2025 and row.barei == 4
    assert row.cohort == 2021                     # 世代 = 年 − 馬齢
    assert row.kyori_m == 1600 and row.time_sec == 96.0 and row["class_level"] == 2.0


def test_prepare_runs_excludes_races_at_or_after_as_of():
    races, horses = frames(
        [race_row("2025010505010101", hasso_at="2025-01-05T15:00:00+09:00"),
         race_row("2025010505010102", hasso_at="2025-01-05T15:30:00+09:00")],
        [horse_row("2025010505010101"), horse_row("2025010505010102")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 1, 5, 15, 30))
    assert runs.race_id.tolist() == ["2025010505010101"]   # 同時刻は含めない


# race_id の先頭8桁は開催日として読むので、作り物でも本物と同じ形にする
OK, JUMP, NAR, ENTRY = "2025010505010101", "2025010505010102", "2025010530010103", "2025010505010104"


def test_prepare_runs_exclusions():
    races, horses = frames(
        [race_row(OK), race_row(JUMP, track_kind="jump"), race_row(NAR, jyo_cd="30"),
         race_row(ENTRY, data_kubun="2")],
        [horse_row(OK, "01"), horse_row(OK, "02", ijyo="1"), horse_row(OK, "03", ijyo="4"),
         horse_row(OK, "04", time=""), horse_row(OK, "05", time="0"),
         horse_row(JUMP), horse_row(NAR), horse_row(ENTRY, data_kubun="2")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert runs.race_id.tolist() == [OK] and runs.umaban.tolist() == ["01"]


def test_prepare_runs_prefers_result_over_entry_rows():
    races, horses = frames(
        [race_row(OK, data_kubun="2", make_date="20250103"), race_row(OK)],
        [horse_row(OK, time="", data_kubun="2", make_date="20250103"), horse_row(OK)])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert len(runs) == 1 and runs.iloc[0].time_sec == 96.0


def test_prepare_runs_requires_timezone():
    races, horses = frames([race_row(OK)], [horse_row(OK)])
    with pytest.raises(ValueError):
        prepare_runs(races, horses, as_of=datetime(2025, 2, 1))
```

- [ ] **Step 3: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_speed_figure.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'keiba.features.speed_figure'`）

- [ ] **Step 4: 実装する**

`src/keiba/features/speed_figure.py` を作る:

```python
"""タイム指数。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md §3。

1走ごとに「基準タイム + その日の馬場差 − 走破タイム」を 1600m 相当の秒にする（正ほど速い）。
基準タイム・クラス差・馬場差は**古馬戦（3歳以上・4歳以上）の1着だけ**から推定する。
2歳戦・3歳戦を混ぜると、世代の強弱が「その日の馬場が速かった／遅かった」として馬場差に
吸収され、世代差が消える（§3.4）。

DataKubun の選び方は repository.py の `_pick_by_data_kubun` をそのまま使う。選び方を
2か所に持つと、片方だけ直したときに「確定前の値を確定値として使う」事故が起きるため。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from keiba import config
from keiba.data.repository import _pick_by_data_kubun

JRA_JYO = frozenset(f"{i:02d}" for i in range(1, 11))
FLAT = ("turf", "dirt")
OLDER_SYUBETU = frozenset({"13", "14"})   # 3歳以上・4歳以上
KEY = ["jyo_cd", "track_cd", "kyori_m"]    # 基準タイムの単位（track_cd は芝ダと内外回りを含む）
FACE = ["day", "jyo_cd", "track_kind"]     # 馬場差の単位（開催面）
RUN_COLUMNS = ["race_id", "umaban", "ketto_num", "day", "date", "year", "jyo_cd", "track_cd",
               "track_kind", "kyori_m", "syubetu_cd", "class_level", "time_sec", "rank",
               "barei", "cohort"]
# 取消・除外（走っていない）と中止・失格（走破タイムが使えない）は指数の対象にしない
_EXCLUDED_IJYO = config.IJYO_EXCLUDED | config.IJYO_NO_TIME
_RACE_FIELDS = ["race_id", "jyo_cd", "track_cd", "track_kind", "kyori_m", "syubetu_cd", "class_level"]


def prepare_runs(races: pd.DataFrame, race_horses: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """指数の対象になる走を1行ずつ返す（§3.1）。発走が as_of より前の確定成績だけ。

    同時刻に発走したレースは含めない（Repository と同じく「その時点でまだ結果を知り得ない」）。
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of にはタイムゾーンが必要です（日本時間の 0:00 など）")
    r = _pick_by_data_kubun(races, config.DATA_KUBUN_RESULT_PRIORITY)
    r = r[r["jyo_cd"].isin(JRA_JYO) & r["track_kind"].isin(FLAT)]
    hasso = pd.to_datetime(r["hasso_at"], errors="coerce", utc=True)
    r = r[hasso < pd.Timestamp(as_of).tz_convert("UTC")]
    h = race_horses[race_horses["race_id"].isin(r["race_id"])]
    h = _pick_by_data_kubun(h, config.DATA_KUBUN_RESULT_PRIORITY)
    h = h[~h["result_ijyo_cd"].isin(_EXCLUDED_IJYO)]
    m = h.merge(r[_RACE_FIELDS], on="race_id", how="inner")
    m = m.assign(time_sec=pd.to_numeric(m["result_time_sec"], errors="coerce"),
                 kyori=pd.to_numeric(m["kyori_m"], errors="coerce"),
                 age=pd.to_numeric(m["barei"], errors="coerce"))
    m = m[(m["time_sec"] > 0) & m["kyori"].notna() & m["age"].notna()]   # NaN の比較は False なので落ちる
    day = m["race_id"].str[:8]
    year = day.str[:4].astype(int)
    out = pd.DataFrame({
        "race_id": m["race_id"], "umaban": m["umaban"], "ketto_num": m["ketto_num"],
        "day": day, "date": pd.to_datetime(day, format="%Y%m%d"), "year": year,
        "jyo_cd": m["jyo_cd"], "track_cd": m["track_cd"], "track_kind": m["track_kind"],
        "kyori_m": m["kyori"].astype(int), "syubetu_cd": m["syubetu_cd"],
        "class_level": pd.to_numeric(m["class_level"], errors="coerce"),
        "time_sec": m["time_sec"].astype(float),
        "rank": pd.to_numeric(m["result_kakutei_jyuni"], errors="coerce"),
        "barei": m["age"].astype(int), "cohort": year - m["age"].astype(int),
    })
    return out[RUN_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 5: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_speed_figure.py -v`
Expected: 5 passed

- [ ] **Step 6: コミット**

```bash
git add docs/superpowers/specs/2026-10-04-generation-level-design.md src/keiba/features/speed_figure.py tests/test_speed_figure.py
git commit -m "feat: タイム指数の対象の走を抽出する（世代レベル 第1段階）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 物差しの推定と指数（`fit_model` / `apply_figures` / `repeatability`）

**Files:**
- Modify: `src/keiba/features/speed_figure.py`
- Create: `tests/fixtures/figure_runs.py`
- Modify: `tests/test_speed_figure.py`

**Interfaces:**
- Consumes: Task 1 の `RUN_COLUMNS` 形の DataFrame
- Produces:
  - `FigureModel`（frozen dataclass）: `standards: pd.Series`（index = KEY の MultiIndex）、
    `class_effects: pd.Series`（index = (class_level, track_kind)）、`variants: pd.Series`（index = FACE）、
    `sigma2: float`、`tau2: float`、`n_races: int`
  - `fit_model(runs: pd.DataFrame, *, max_iter: int = 500, tol: float = 1e-10) -> FigureModel`
  - `apply_figures(runs: pd.DataFrame, model: FigureModel) -> pd.DataFrame`（`figure` / `has_standard` / `has_variant` を足す）
  - `compute_figures(races, race_horses, *, as_of) -> tuple[pd.DataFrame, FigureModel]`
  - `repeatability(fig: pd.DataFrame, *, max_gap_days: int = 120) -> tuple[float, int]`
  - `tests/fixtures/figure_runs.py`: `run_row(**kw) -> dict`、`older_runs() -> pd.DataFrame`、
    `young_runs(slow: float) -> pd.DataFrame`、定数 `S_TRUE`, `C_TRUE`, `V_TRUE`

- [ ] **Step 1: 作り物の走を書く**

`tests/fixtures/figure_runs.py`:

```python
"""タイム指数のテスト用の作り物の「走」。prepare_runs の出力と同じ列を持つ。

古馬戦は 1000m あたりの秒が s（距離ごと）+ c（クラス）+ v（日ごと）ちょうどになるよう作る。
雑音を入れないので、推定はほぼ厳密に真値を取り戻す。
"""

from __future__ import annotations

import pandas as pd

S_TRUE = {1600: 60.0, 2000: 61.0}       # 基準（1000m あたり秒）
C_TRUE = {2: 0.0, 3: -0.5}              # クラス差。2（1勝）が基準
V_TRUE = [-0.3, -0.1, 0.0, 0.1, 0.25, 0.05]   # 2025-01-01〜06 の馬場差


def run_row(**kw) -> dict:
    row = dict(race_id="R", umaban="01", ketto_num="H0", day="20250101",
               date=pd.Timestamp("2025-01-01"), year=2025, jyo_cd="05", track_cd="24",
               track_kind="turf", kyori_m=1600, syubetu_cd="13", class_level=2.0,
               time_sec=96.0, rank=1.0, barei=4, cohort=2021)
    row.update(kw)
    return row


def _day(i: int) -> str:
    return f"202501{i + 1:02d}"


def older_runs() -> pd.DataFrame:
    """6日 × 2距離 × 2クラスの古馬戦の1着。"""
    rows = []
    for i, v in enumerate(V_TRUE):
        day = _day(i)
        for kyori, s in S_TRUE.items():
            for cls, c in C_TRUE.items():
                rows.append(run_row(race_id=f"{day}-{kyori}-{cls}", day=day, date=pd.Timestamp(day),
                                    kyori_m=kyori, class_level=float(cls),
                                    time_sec=(s + c + v) * kyori / 1000))
    return pd.DataFrame(rows)


def young_runs(slow: float) -> pd.DataFrame:
    """各日に1つずつの2歳戦（1600m）。1000m あたり slow 秒だけ遅い。"""
    rows = []
    for i, v in enumerate(V_TRUE):
        day = _day(i)
        rows.append(run_row(race_id=f"{day}-Y", day=day, date=pd.Timestamp(day), syubetu_cd="11",
                            class_level=1.0, time_sec=(S_TRUE[1600] + v + slow) * 1.6,
                            barei=2, cohort=2023, ketto_num=f"Y{i}"))
    return pd.DataFrame(rows)
```

- [ ] **Step 2: 失敗するテストを足す**

`tests/test_speed_figure.py` の import を次にし、末尾にテストを足す:

```python
import numpy as np

from keiba.features.speed_figure import (RUN_COLUMNS, apply_figures, fit_model, prepare_runs,
                                         repeatability)
from tests.fixtures.figure_runs import C_TRUE, V_TRUE, older_runs, run_row, young_runs
```

```python
def test_fit_recovers_track_variants():
    model = fit_model(older_runs())
    est = model.variants.xs(("05", "turf"), level=("jyo_cd", "track_kind")).to_numpy()
    truth = np.array(V_TRUE)
    # 基準と馬場差は全体の定数を取り合うので、平均を引いた形で比べる
    assert np.allclose(est - est.mean(), truth - truth.mean(), atol=1e-6)
    assert model.n_races == 24


def test_fit_recovers_class_effects_and_standards():
    model = fit_model(older_runs())
    assert model.class_effects[(2.0, "turf")] == pytest.approx(0.0, abs=1e-9)
    assert model.class_effects[(3.0, "turf")] == pytest.approx(C_TRUE[3], abs=1e-6)
    s = model.standards
    assert s[("05", "24", 2000)] - s[("05", "24", 1600)] == pytest.approx(1.0, abs=1e-6)


def test_slow_two_year_olds_are_not_absorbed_into_variant():
    """§3.4 の核心。2歳戦が一律に遅くても馬場差は動かず、2歳の指数が下がる。"""
    base = fit_model(older_runs())
    runs = pd.concat([older_runs(), young_runs(slow=0.5)], ignore_index=True)
    model = fit_model(runs)
    pd.testing.assert_series_equal(model.variants, base.variants)
    fig = apply_figures(runs, model)
    young = fig[fig.syubetu_cd == "11"]
    assert np.allclose(young.figure, -0.5 * 1.6, atol=1e-6)
    older_cls3 = fig[(fig.syubetu_cd == "13") & (fig.class_level == 3.0)]
    assert np.allclose(older_cls3.figure, 0.5 * 1.6, atol=1e-6)   # 上のクラスほど速い


def test_non_winners_do_not_enter_the_fit():
    base = fit_model(older_runs())
    extra = pd.DataFrame([run_row(race_id="20250101-1600-2", umaban="02", rank=2.0, time_sec=200.0)])
    model = fit_model(pd.concat([older_runs(), extra], ignore_index=True))
    pd.testing.assert_series_equal(model.variants, base.variants)


def test_dead_heat_winner_counted_once():
    heat = pd.DataFrame([run_row(race_id="20250101-1600-2", umaban="09", ketto_num="H9",
                                 time_sec=older_runs().iloc[0].time_sec)])
    model = fit_model(pd.concat([older_runs(), heat], ignore_index=True))
    assert model.n_races == 24


def test_run_on_key_without_older_races_has_no_figure():
    odd = pd.DataFrame([run_row(race_id="X", kyori_m=1200, syubetu_cd="11", time_sec=72.0)])
    runs = pd.concat([older_runs(), odd], ignore_index=True)
    fig = apply_figures(runs, fit_model(runs))
    row = fig[fig.race_id == "X"].iloc[0]
    assert not row.has_standard and np.isnan(row.figure)


def test_run_on_day_without_older_races_uses_zero_variant():
    z = pd.DataFrame([run_row(race_id="Z", day="20250301", date=pd.Timestamp("2025-03-01"),
                              syubetu_cd="11", time_sec=60.0 * 1.6)])
    runs = pd.concat([older_runs(), z], ignore_index=True)
    model = fit_model(runs)
    row = apply_figures(runs, model).set_index("race_id").loc["Z"]
    assert not row.has_variant
    assert row.figure == pytest.approx((model.standards[("05", "24", 1600)] - 60.0) * 1.6)


def test_fit_without_older_races_raises():
    with pytest.raises(ValueError):
        fit_model(young_runs(slow=0.0))


def test_repeatability_pairs_consecutive_runs_within_gap():
    rows = []
    for h, base in enumerate([-1.0, 0.0, 1.0, 2.0]):
        for k, d in enumerate(["2025-01-01", "2025-02-01", "2025-12-01"]):
            rows.append(run_row(race_id=f"{h}{k}", ketto_num=f"H{h}", date=pd.Timestamp(d)))
    fig = pd.DataFrame(rows)
    fig["figure"] = [b + 0.01 * k for b in [-1.0, 0.0, 1.0, 2.0] for k in range(3)]
    r, n = repeatability(fig, max_gap_days=120)
    assert n == 4                       # 2月→12月は 120 日を超えるので数えない
    assert r == pytest.approx(1.0)
```

- [ ] **Step 3: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_speed_figure.py -v`
Expected: FAIL（`ImportError: cannot import name 'apply_figures'`）

- [ ] **Step 4: 実装する**

`src/keiba/features/speed_figure.py` の import に `from dataclasses import dataclass` と `import numpy as np` を足し、
定数の後ろと末尾に次を足す:

```python
REFERENCE_CLASS = 2            # 1勝クラス。クラス差の基準（c = 0）
PER_1000M_TO_FIGURE = 1.6      # 1000m あたり秒 → 1600m 相当の秒


@dataclass(frozen=True)
class FigureModel:
    """推定した物差し。値はすべて 1000m あたりの秒。"""

    standards: pd.Series       # KEY → 基準 s
    class_effects: pd.Series   # (class_level, track_kind) → クラス差 c。基準クラスは 0
    variants: pd.Series        # FACE → 馬場差 v（経験ベイズで縮小済み）
    sigma2: float              # 開催面の中の残差の分散
    tau2: float                # 開催面どうしの馬場差の分散
    n_races: int               # 推定に使った古馬戦の数
```

```python
def _older_winners(runs: pd.DataFrame) -> pd.DataFrame:
    """推定に使う行: 古馬戦の1着で、クラスが決まるもの。1着同着は1行にする（同じタイム）。"""
    w = runs[runs["syubetu_cd"].isin(OLDER_SYUBETU) & (runs["rank"] == 1) & runs["class_level"].notna()]
    w = w.drop_duplicates("race_id")
    return w.assign(y=w["time_sec"] / (w["kyori_m"] / 1000.0))


def _shrunk_face_effects(r: np.ndarray, fc: np.ndarray, nf: int) -> tuple[np.ndarray, float, float]:
    """開催面ごとの残差の平均を経験ベイズで 0 に縮める（§3.2）。

    σ²（面の中の分散）と τ²（面どうしの分散）はモーメント法。レースが少ない面ほど強く縮む。
    """
    n = np.bincount(fc, minlength=nf).astype(float)
    sums = np.bincount(fc, weights=r, minlength=nf)
    means = sums / n
    dof = len(r) - nf
    sigma2 = float(((r - means[fc]) ** 2).sum() / dof) if dof > 0 else 0.0
    tau2 = max(float(means.var()) - sigma2 * float((1.0 / n).mean()), 1e-12)
    return sums / (n + sigma2 / tau2), sigma2, tau2


def fit_model(runs: pd.DataFrame, *, max_iter: int = 500, tol: float = 1e-10) -> FigureModel:
    """古馬戦の1着だけから s・c・v を交互に推定する（§3.2）。s と c は縮めない。"""
    w = _older_winners(runs)
    if w.empty:
        raise ValueError("古馬戦の1着が1件もありません。基準タイムを推定できません")
    gk = w.groupby(KEY, sort=True)
    gc = w.groupby(["class_level", "track_kind"], sort=True)
    gf = w.groupby(FACE, sort=True)
    kc, cc, fc = gk.ngroup().to_numpy(), gc.ngroup().to_numpy(), gf.ngroup().to_numpy()
    key_index, class_index, face_index = gk.size().index, gc.size().index, gf.size().index
    nk, nc, nf = len(key_index), len(class_index), len(face_index)
    y = w["y"].to_numpy()
    n_k = np.bincount(kc, minlength=nk).astype(float)
    n_c = np.bincount(cc, minlength=nc).astype(float)
    key_kind = gk["track_kind"].first().to_numpy()
    class_kind = class_index.get_level_values("track_kind").to_numpy()
    class_lv = class_index.get_level_values("class_level").to_numpy()
    s = np.bincount(kc, weights=y, minlength=nk) / n_k
    c, v = np.zeros(nc), np.zeros(nf)
    sigma2 = tau2 = 0.0
    fitted = s[kc]
    for _ in range(max_iter):
        s = np.bincount(kc, weights=y - c[cc] - v[fc], minlength=nk) / n_k
        c = np.bincount(cc, weights=y - s[kc] - v[fc], minlength=nc) / n_c
        # 芝ダごとに基準クラス（1勝）の c を 0 にし、その分を s に移す（当てはまりは変わらない）
        for kind in np.unique(class_kind):
            ref_mask = (class_kind == kind) & (class_lv == REFERENCE_CLASS)
            ref = float(c[ref_mask][0]) if ref_mask.any() else 0.0
            c[class_kind == kind] -= ref
            s[key_kind == kind] += ref
        v, sigma2, tau2 = _shrunk_face_effects(y - s[kc] - c[cc], fc, nf)
        new = s[kc] + c[cc] + v[fc]
        done = float(np.max(np.abs(new - fitted))) < tol
        fitted = new
        if done:
            break
    return FigureModel(standards=pd.Series(s, index=key_index),
                       class_effects=pd.Series(c, index=class_index),
                       variants=pd.Series(v, index=face_index),
                       sigma2=sigma2, tau2=tau2, n_races=len(w))


def apply_figures(runs: pd.DataFrame, model: FigureModel) -> pd.DataFrame:
    """全走（2歳戦・3歳戦を含む）に指数を付ける（§3.3）。クラス差は足さない。

    基準が無いキーの走は指数を欠損にする。古馬戦の無い開催面は馬場差 0（縮小の事前の値）。
    どちらも has_standard / has_variant で数えられるようにする。
    """
    out = runs.copy()
    s = model.standards.reindex(pd.MultiIndex.from_frame(out[KEY])).to_numpy()
    v = model.variants.reindex(pd.MultiIndex.from_frame(out[FACE])).to_numpy()
    out["has_standard"] = ~np.isnan(s)
    out["has_variant"] = ~np.isnan(v)
    per_1000m = out["time_sec"].to_numpy() / (out["kyori_m"].to_numpy() / 1000.0)
    out["figure"] = (s + np.nan_to_num(v, nan=0.0) - per_1000m) * PER_1000M_TO_FIGURE
    return out


def compute_figures(races: pd.DataFrame, race_horses: pd.DataFrame, *,
                    as_of: datetime) -> tuple[pd.DataFrame, FigureModel]:
    """prepare_runs → fit_model → apply_figures をまとめて呼ぶ。"""
    runs = prepare_runs(races, race_horses, as_of=as_of)
    model = fit_model(runs)
    return apply_figures(runs, model), model


def repeatability(fig: pd.DataFrame, *, max_gap_days: int = 120) -> tuple[float, int]:
    """同じ馬の続けて2走（間隔 max_gap_days 日以内）の指数の相関係数と組の数（§3.6）。"""
    d = fig[fig["figure"].notna()].sort_values(["ketto_num", "date", "race_id"])
    horse = d["ketto_num"].to_numpy()
    gap = np.diff(d["date"].to_numpy()).astype("timedelta64[D]").astype(int)
    ok = (horse[1:] == horse[:-1]) & (gap > 0) & (gap <= max_gap_days)
    f = d["figure"].to_numpy()
    a, b = f[:-1][ok], f[1:][ok]
    if len(a) < 3:
        return float("nan"), int(len(a))
    return float(np.corrcoef(a, b)[0, 1]), int(len(a))
```

- [ ] **Step 5: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_speed_figure.py -v`
Expected: 14 passed

- [ ] **Step 6: コミット**

```bash
git add src/keiba/features/speed_figure.py tests/test_speed_figure.py tests/fixtures/figure_runs.py
git commit -m "feat: 古馬戦から基準タイム・馬場差を推定してタイム指数を付ける

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 年齢曲線（`age_curve.py`）

**Files:**
- Create: `src/keiba/evaluate/age_curve.py`
- Create: `tests/test_age_curve.py`

**Interfaces:**
- Consumes: 指数の表（列 `ketto_num, year, date, barei, track_kind, figure`。`apply_figures` の出力はこれを含む）
- Produces:
  - `ANCHOR = (3, 1)`（3歳後半）、`period_label(age: int, half: int) -> str`（例 `"3歳後半"`）、`half_of(d: date) -> int`（1〜6月 0、7〜12月 1）
  - `period_values(fig, *, years: list[int], before: date, surface: str | None = None) -> pd.DataFrame`（列 `ketto_num, year, half, age, value`）
  - `period_changes(pv: pd.DataFrame) -> pd.DataFrame`（列 `ketto_num, to_age, to_half, delta`）
  - `age_curve(fig, *, years, before, surface=None, n_boot=200, seed=20261004) -> pd.DataFrame`
    （列 `CURVE_COLUMNS` = `age, half, label, delta, n, delta_lo, delta_hi, level, level_lo, level_hi`）
  - `curve_levels(curve: pd.DataFrame) -> dict[tuple[int, int], float]`
  - `peak_summary(curve: pd.DataFrame) -> dict`（キー `peak`, `peak_level`, `drop_1y`）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_age_curve.py`:

```python
"""年齢曲線（仕様 §6）のテスト。同じ馬の変化で測れば、弱い馬の引退（生き残りの偏り）に強い。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.age_curve import (ANCHOR, CURVE_COLUMNS, age_curve, curve_levels, half_of,
                                      peak_summary, period_label, period_values)

# 真の年齢曲線（3歳後半 = 0）
L = {(2, 1): -2.0, (3, 0): -1.0, (3, 1): 0.0, (4, 0): 0.3, (4, 1): 0.4, (5, 0): 0.3, (5, 1): 0.1}
YEARS = [2022, 2023, 2024, 2025]


def synth_fig(*, n_horses=200, retire_weak_after=None, seed=0) -> pd.DataFrame:
    """馬ごとの地力 base + 年齢の水準 L。各期間に2走（同じ値）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_horses):
        base = float(rng.normal(0, 1))
        cohort = 2020 + i % 2
        for (age, half), lvl in L.items():
            if retire_weak_after and base < 0 and 2 * age + half > 2 * retire_weak_after[0] + retire_weak_after[1]:
                break
            year = cohort + age
            if year not in YEARS:
                continue
            for month in ((3, 5) if half == 0 else (9, 11)):
                rows.append(dict(ketto_num=f"H{i}", year=year, date=pd.Timestamp(year, month, 1),
                                 barei=age, track_kind="turf", figure=base + lvl))
    return pd.DataFrame(rows)


def test_labels_and_halves():
    assert period_label(3, 1) == "3歳後半" and period_label(4, 0) == "4歳前半"
    assert half_of(date(2026, 6, 30)) == 0 and half_of(date(2026, 7, 1)) == 1
    assert ANCHOR == (3, 1)


def test_curve_recovers_true_levels():
    curve = age_curve(synth_fig(), years=YEARS, before=date(2026, 1, 1), n_boot=20, seed=1)
    assert list(curve.columns) == CURVE_COLUMNS
    got = curve_levels(curve)
    for period, lvl in L.items():
        assert got[period] == pytest.approx(lvl, abs=1e-9)


def test_paired_changes_resist_survivorship_bias():
    """弱い馬が3歳後半で引退しても、同じ馬の変化なら真の曲線に近い。全体平均は高齢を強く見せる。"""
    fig = synth_fig(retire_weak_after=(3, 1))
    got = curve_levels(age_curve(fig, years=YEARS, before=date(2026, 1, 1), n_boot=20, seed=1))
    assert got[(4, 0)] == pytest.approx(L[(4, 0)], abs=1e-9)
    naive = fig.assign(half=(fig.date.dt.month > 6).astype(int)).groupby(["barei", "half"]).figure.mean()
    naive_level = naive[(4, 0)] - naive[(3, 1)]
    assert abs(naive_level - L[(4, 0)]) > 0.1


def test_single_run_periods_are_not_used():
    fig = pd.DataFrame([
        dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 3, 1), barei=3, track_kind="turf", figure=1.0),
        dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 9, 1), barei=3, track_kind="turf", figure=2.0),
        dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 10, 1), barei=3, track_kind="turf", figure=3.0),
    ])
    pv = period_values(fig, years=YEARS, before=date(2026, 1, 1))
    assert pv[["year", "half", "value"]].values.tolist() == [[2024, 1, 2.5]]   # 前半は1走なので使わない


def test_top_two_runs_are_averaged():
    rows = [dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 8, d), barei=3, track_kind="turf", figure=f)
            for d, f in ((1, 1.0), (2, 5.0), (3, 3.0))]
    pv = period_values(pd.DataFrame(rows), years=YEARS, before=date(2026, 1, 1))
    assert pv.value.tolist() == [4.0]


def test_incomplete_periods_are_not_used():
    pv = period_values(synth_fig(), years=YEARS, before=date(2025, 10, 4))
    assert not ((pv.year == 2025) & (pv.half == 1)).any()
    assert ((pv.year == 2025) & (pv.half == 0)).any()


def test_surface_filter():
    fig = synth_fig()
    fig.loc[fig.index % 2 == 0, "track_kind"] = "dirt"    # 各期間の1走目をダートに
    assert period_values(fig, years=YEARS, before=date(2026, 1, 1), surface="turf").empty   # 芝は1走ずつ


def test_peak_and_one_year_drop():
    curve = age_curve(synth_fig(), years=YEARS, before=date(2026, 1, 1), n_boot=10, seed=1)
    summary = peak_summary(curve)
    assert summary["peak"] == "4歳後半"
    assert summary["drop_1y"] == pytest.approx(0.4 - 0.1, abs=1e-9)


def test_bootstrap_is_reproducible():
    fig = synth_fig()
    fig["figure"] += np.random.default_rng(3).normal(0, 0.3, len(fig))
    a = age_curve(fig, years=YEARS, before=date(2026, 1, 1), n_boot=30, seed=7)
    b = age_curve(fig, years=YEARS, before=date(2026, 1, 1), n_boot=30, seed=7)
    pd.testing.assert_frame_equal(a, b)
    row = a.set_index("label").loc["4歳前半"]
    assert row.level_lo <= row.level <= row.level_hi


def test_age_curve_empty_when_no_changes():
    one = synth_fig(n_horses=1).iloc[:2]       # 1期間しかない
    curve = age_curve(one, years=YEARS, before=date(2026, 1, 1), n_boot=5, seed=1)
    assert curve.empty and list(curve.columns) == CURVE_COLUMNS
    assert peak_summary(curve)["peak"] is None
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_age_curve.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'keiba.evaluate.age_curve'`）

- [ ] **Step 3: 実装する**

`src/keiba/evaluate/age_curve.py`:

```python
"""年齢曲線（仕様 §6）。同じ馬の半年ごとの変化を積み上げ、3歳後半を 0 にする。

年齢ごとの全体の平均を並べると、弱い馬ほど早く引退するので高齢ほど強く見える（生き残りの偏り）。
同じ馬の隣り合う期間どうしの差だけを使うことで、それを避ける。ただし大きく衰えた馬はそのまま
引退して差が残らないので、衰えは小さめに出る（§6.3）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

ANCHOR = (3, 1)        # 3歳後半 = 0
MIN_RUNS = 2           # 期間の値を作るのに要る走の数
TOP_RUNS = 2           # 期間の値 = 上位2走の平均（最高値1つだと走った回数が多い期間ほど高く出る）
CURVE_COLUMNS = ["age", "half", "label", "delta", "n", "delta_lo", "delta_hi",
                 "level", "level_lo", "level_hi"]


def period_label(age: int, half: int) -> str:
    return f"{age}歳{'前半' if half == 0 else '後半'}"


def half_of(d: date) -> int:
    """1〜6月は 0（前半）、7〜12月は 1（後半）。"""
    return 0 if d.month <= 6 else 1


def _ordinal(age: int, half: int) -> int:
    return 2 * age + half


def period_values(fig: pd.DataFrame, *, years: list[int], before: date,
                  surface: str | None = None) -> pd.DataFrame:
    """馬 × 期間の値（上位2走の平均）。終わっている期間だけ、2走以上ある期間だけ。"""
    d = fig[fig["figure"].notna() & fig["year"].isin(years)]
    if surface is not None:
        d = d[d["track_kind"] == surface]
    d = d.assign(half=(d["date"].dt.month > 6).astype(int))
    end = pd.to_datetime(pd.DataFrame({"year": d["year"] + d["half"],
                                       "month": np.where(d["half"] == 0, 7, 1), "day": 1}))
    d = d[end <= pd.Timestamp(before)]
    keys = ["ketto_num", "year", "half", "barei"]
    d = d.sort_values("figure", ascending=False)
    d = d.assign(_k=d.groupby(keys).cumcount())
    count = d.groupby(keys).size()
    top = d[d["_k"] < TOP_RUNS].groupby(keys)["figure"].mean()
    top = top.loc[count[count >= MIN_RUNS].index]
    out = top.rename("value").reset_index().rename(columns={"barei": "age"})
    return out[["ketto_num", "year", "half", "age", "value"]].sort_values(
        ["ketto_num", "year", "half"]).reset_index(drop=True)


def period_changes(pv: pd.DataFrame) -> pd.DataFrame:
    """同じ馬の隣り合う期間（前半→後半、後半→翌年前半）の値の差。"""
    p = pv.assign(t=pv["year"] * 2 + pv["half"]).sort_values(["ketto_num", "t"])
    nxt = p.groupby("ketto_num")[["t", "age", "half", "value"]].shift(-1)
    ok = (nxt["t"] == p["t"] + 1).to_numpy()
    return pd.DataFrame({
        "ketto_num": p["ketto_num"].to_numpy()[ok],
        "to_age": nxt["age"].to_numpy()[ok].astype(int),
        "to_half": nxt["half"].to_numpy()[ok].astype(int),
        "delta": (nxt["value"] - p["value"]).to_numpy()[ok],
    })


def _levels(to_ord: list[int], deltas: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """期間 lo..hi（序数）の水準。ANCHOR を 0 にして隣の期間の変化を足し引きする。"""
    d = dict(zip(to_ord, deltas.tolist()))
    a = _ordinal(*ANCHOR)
    lv: dict[int, float] = {}
    if lo <= a <= hi:
        lv[a] = 0.0
        for o in range(a + 1, hi + 1):
            lv[o] = lv[o - 1] + d.get(o, np.nan)
        for o in range(a - 1, lo - 1, -1):
            lv[o] = lv[o + 1] - d.get(o + 1, np.nan)
    return np.array([lv.get(o, np.nan) for o in range(lo, hi + 1)])


def age_curve(fig: pd.DataFrame, *, years: list[int], before: date, surface: str | None = None,
              n_boot: int = 200, seed: int = 20261004) -> pd.DataFrame:
    """年齢曲線と区間（馬単位のブートストラップ）。変化が1つも無ければ空の表。"""
    ch = period_changes(period_values(fig, years=years, before=before, surface=surface))
    if ch.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    ords = (ch["to_age"] * 2 + ch["to_half"]).to_numpy()
    to_ord = sorted(set(ords.tolist()))
    tc = np.searchsorted(np.array(to_ord), ords)
    hc, horses = pd.factorize(ch["ketto_num"])
    delta = ch["delta"].to_numpy()
    T, H = len(to_ord), len(horses)

    def means(weights: np.ndarray) -> np.ndarray:
        num = np.bincount(tc, weights=delta * weights, minlength=T)
        den = np.bincount(tc, weights=weights, minlength=T)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    lo, hi = min(to_ord) - 1, max(to_ord)
    point = means(np.ones(len(ch)))
    rng = np.random.default_rng(seed)
    boot_d, boot_l = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, H, H), minlength=H)[hc].astype(float)
        m = means(w)
        boot_d.append(m)
        boot_l.append(_levels(to_ord, m, lo, hi))
    boot_d, boot_l = np.array(boot_d), np.array(boot_l)
    level = _levels(to_ord, point, lo, hi)
    n_by = np.bincount(tc, minlength=T)
    rows = []
    for i, o in enumerate(range(lo, hi + 1)):
        age, half = o // 2, o % 2
        j = to_ord.index(o) if o in to_ord else None
        q = lambda a: float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))  # noqa: E731
        d_lo, d_hi = q(boot_d[:, j]) if j is not None else (np.nan, np.nan)
        l_lo, l_hi = q(boot_l[:, i]) if not np.all(np.isnan(boot_l[:, i])) else (np.nan, np.nan)
        rows.append(dict(age=age, half=half, label=period_label(age, half),
                         delta=float(point[j]) if j is not None else np.nan,
                         n=int(n_by[j]) if j is not None else 0,
                         delta_lo=d_lo, delta_hi=d_hi, level=float(level[i]),
                         level_lo=l_lo, level_hi=l_hi))
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def curve_levels(curve: pd.DataFrame) -> dict[tuple[int, int], float]:
    return {(int(r.age), int(r.half)): float(r.level) for r in curve.itertuples()}


def peak_summary(curve: pd.DataFrame) -> dict:
    """全盛期（水準が最大の期間）と、その1年後（2期間後）までの落ち幅（正 = 落ちた）。"""
    c = curve.dropna(subset=["level"])
    if c.empty:
        return {"peak": None, "peak_level": float("nan"), "drop_1y": float("nan")}
    top = c.loc[c["level"].astype(float).idxmax()]
    later = curve[(curve["age"] * 2 + curve["half"]) == _ordinal(int(top.age), int(top.half)) + 2]
    drop = float(top.level - later["level"].iloc[0]) if len(later) else float("nan")
    return {"peak": top.label, "peak_level": float(top.level), "drop_1y": drop}
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_age_curve.py -v`
Expected: 11 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/evaluate/age_curve.py tests/test_age_curve.py
git commit -m "feat: 同じ馬の半年ごとの変化から年齢曲線を出す

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 世代の数値と比較（`generation.py` の点推定）

**Files:**
- Create: `src/keiba/evaluate/generation.py`
- Create: `tests/test_generation.py`

**Interfaces:**
- Consumes: 指数の表（列 `ketto_num, year, cohort, date, figure`）、Task 3 の `half_of`
- Produces:
  - 定数 `METRICS = ("top10", "top50", "median")`、`HYPOTHESIS_PAIRS = ((4, 5), (5, 3), (4, 3))`、`MIN_RANKING_AGE = 3`
  - `same_monthday(as_of: date, year: int) -> date`
  - `comparable_years(coverage_start: date, last_year: int) -> list[int]`
  - `cohort_values(fig, *, year: int, cutoff: date, cohort: int) -> dict`（キー `n, top10, top50, median`）
  - `cohort_table(fig, *, as_of: date, ages: list[int], years: list[int]) -> pd.DataFrame`（列 `year, age, cohort, n, top10, top50, median`）
  - `yearly_comparison(table, *, year: int) -> pd.DataFrame`（列 `age, cohort, metric, current, past_mean, diff, rank, n_years`）
  - `trajectory(table, *, year: int) -> pd.DataFrame`（列 `cohort, age_now, age, year, metric, value, deviation`）
  - `ranking_table(table, *, metric: str) -> pd.DataFrame`（index 年、列 年齢。3歳以上だけ）
  - `pair_differences(table, *, year: int, pairs, curve_levels: dict | None, half: int) -> pd.DataFrame`（列 `pair, stronger, weaker, metric, diff, age_gap, adjusted`）
  - `verdict(lo: float, hi: float) -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_generation.py`:

```python
"""世代の数値と比較（仕様 §4・§5）のテスト。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.generation import (HYPOTHESIS_PAIRS, cohort_table, cohort_values,
                                       comparable_years, pair_differences, ranking_table,
                                       same_monthday, trajectory, verdict, yearly_comparison)


def fig_rows(year, cohort, figures, *, month=5, prefix="H"):
    return [dict(ketto_num=f"{prefix}{cohort}-{i}", year=year, cohort=cohort,
                 date=pd.Timestamp(year, month, 1), figure=f) for i, f in enumerate(figures)]


def test_same_monthday_and_leap_day():
    assert same_monthday(date(2026, 10, 4), 2024) == date(2024, 10, 4)
    assert same_monthday(date(2024, 2, 29), 2023) == date(2023, 2, 28)
    assert same_monthday(date(2024, 2, 29), 2020) == date(2020, 2, 28)


def test_comparable_years_start_after_partial_year():
    assert comparable_years(date(2021, 9, 1), 2026) == [2022, 2023, 2024, 2025, 2026]
    assert comparable_years(date(2022, 1, 1), 2024) == [2022, 2023, 2024]


def test_cohort_values_best_per_horse_and_top_k():
    rows = fig_rows(2025, 2022, range(60))
    rows += [dict(ketto_num="H2022-0", year=2025, cohort=2022, date=pd.Timestamp(2025, 6, 1), figure=100.0)]
    v = cohort_values(pd.DataFrame(rows), year=2025, cutoff=date(2025, 10, 4), cohort=2022)
    best = sorted([100.0] + list(range(1, 60)), reverse=True)    # H2022-0 の最高値は 100
    assert v["n"] == 60
    assert v["top10"] == pytest.approx(np.mean(best[:10]))
    assert v["top50"] == pytest.approx(np.mean(best[:50]))
    assert v["median"] == pytest.approx(np.median(best))


def test_cohort_values_respects_cutoff_and_missing_top_k():
    rows = fig_rows(2025, 2022, [1.0, 2.0], month=5) + fig_rows(2025, 2022, [9.0], month=11, prefix="L")
    v = cohort_values(pd.DataFrame(rows), year=2025, cutoff=date(2025, 10, 4), cohort=2022)
    assert v["n"] == 2 and np.isnan(v["top10"]) and v["median"] == pytest.approx(1.5)


def test_cohort_values_empty_cohort():
    v = cohort_values(pd.DataFrame(fig_rows(2025, 2022, [1.0])), year=2025, cutoff=date(2025, 10, 4), cohort=2023)
    assert v["n"] == 0 and np.isnan(v["median"])


def _table():
    rows = []
    for year, shift in ((2023, 0.0), (2024, 1.0), (2025, -1.0)):
        for age, base in ((2, -3.0), (3, 0.0), (4, 1.0), (5, 0.5)):
            rows += fig_rows(year, year - age, np.full(60, base + shift + (age == 3) * (year == 2025) * -0.5))
    fig = pd.DataFrame(rows)
    return cohort_table(fig, as_of=date(2025, 10, 4), ages=[2, 3, 4, 5], years=[2023, 2024, 2025])


def test_cohort_table_shape():
    t = _table()
    assert list(t.columns) == ["year", "age", "cohort", "n", "top10", "top50", "median"]
    assert len(t) == 12
    assert t.set_index(["year", "age"]).at[(2024, 3), "cohort"] == 2021


def test_yearly_comparison_against_past_mean():
    yc = yearly_comparison(_table(), year=2025).set_index(["age", "metric"])
    row = yc.loc[(3, "top10")]
    assert row.current == pytest.approx(-1.5)          # 0 − 1（年の分）− 0.5
    assert row.past_mean == pytest.approx(0.5)          # 2023 の 0 と 2024 の 1
    assert row["diff"] == pytest.approx(-2.0)
    assert row["rank"] == 3 and row.n_years == 3


def test_trajectory_lines_up_each_cohort_by_age():
    tr = trajectory(_table(), year=2025)
    assert set(tr.cohort) == {2022, 2021, 2020}          # 2025年に3歳以上の世代だけ
    cell = tr[(tr.cohort == 2021) & (tr.age == 3) & (tr.metric == "top10")].iloc[0]
    assert cell.year == 2024
    assert cell.deviation == pytest.approx(1.0 - np.mean([0.0, -1.5]))   # 2024年の3歳 − 他の年の3歳


def test_ranking_table_excludes_two_year_olds():
    rt = ranking_table(_table(), metric="top10")
    assert 2 not in rt.columns and list(rt.columns) == [3, 4, 5]


def test_pair_differences_rejects_two_year_olds():
    with pytest.raises(ValueError):
        pair_differences(_table(), year=2025, pairs=[(3, 2)], curve_levels=None, half=1)


def test_pair_differences_age_adjustment():
    levels = {(3, 1): 0.0, (4, 1): 1.0, (5, 1): 0.5}
    pdf = pair_differences(_table(), year=2025, pairs=HYPOTHESIS_PAIRS, curve_levels=levels,
                           half=1).set_index(["pair", "metric"])
    row = pdf.loc[("4-3", "top10")]
    assert row["diff"] == pytest.approx(0.0 - (-1.5))
    assert row.age_gap == pytest.approx(1.0)
    assert row.adjusted == pytest.approx(1.5 - 1.0)
    assert np.isnan(pair_differences(_table(), year=2025, pairs=[(4, 5)], curve_levels=None,
                                     half=1).iloc[0].adjusted)


def test_verdict():
    assert verdict(0.1, 0.5) == "見立てと合う"
    assert verdict(-0.5, -0.1) == "見立てと逆"
    assert verdict(-0.1, 0.2) == "差があるとは言えない"
    assert verdict(float("nan"), 0.2) == "判定できない"
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'keiba.evaluate.generation'`）

- [ ] **Step 3: 実装する**

`src/keiba/evaluate/generation.py`:

```python
"""世代（生まれ年）の数値と比較。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md §4・§5。

世代の数値は「その年の 1/1 から時点の前日までの、馬ごとの指数の最高値」を高い順に並べて作る。
過去の年とは同じ月日の時点で比べる（年の途中の成長を揃えるため）。
2歳は「今いる世代の並び」に入れない。2歳と3歳以上は実力差が大きすぎ、並べても比較にならない（所有者の判断）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

METRICS = ("top10", "top50", "median")
TOP_K = {"top10": 10, "top50": 50}
HYPOTHESIS_PAIRS = ((4, 5), (5, 3), (4, 3))   # 所有者の見立て 4歳 > 5歳 > 3歳。左が強いと見立てた側
MIN_RANKING_AGE = 3
TABLE_COLUMNS = ["year", "age", "cohort", "n", *METRICS]


def same_monthday(as_of: date, year: int) -> date:
    """過去の年の同じ月日。2/29 は常に 2/28 にそろえる。"""
    if (as_of.month, as_of.day) == (2, 29):
        return date(year, 2, 28)
    return date(year, as_of.month, as_of.day)


def comparable_years(coverage_start: date, last_year: int) -> list[int]:
    """比べてよい年（データが 1/1 から揃う年）。coverage_start は版の取得開始日。"""
    first = coverage_start.year if (coverage_start.month, coverage_start.day) == (1, 1) else coverage_start.year + 1
    return list(range(first, last_year + 1))


def cohort_values(fig: pd.DataFrame, *, year: int, cutoff: date, cohort: int) -> dict:
    """year 年 1/1 から cutoff の前日までの、世代 cohort の数値（§4）。K 頭未満の topK は欠損。"""
    d = fig[(fig["year"] == year) & (fig["cohort"] == cohort)
            & (fig["date"] < pd.Timestamp(cutoff)) & fig["figure"].notna()]
    best = np.sort(d.groupby("ketto_num")["figure"].max().to_numpy())[::-1]
    n = len(best)
    out = {"n": n, "median": float(np.median(best)) if n else float("nan")}
    for name, k in TOP_K.items():
        out[name] = float(best[:k].mean()) if n >= k else float("nan")
    return out


def cohort_table(fig: pd.DataFrame, *, as_of: date, ages: list[int], years: list[int]) -> pd.DataFrame:
    """年 × 年齢 の世代の数値。各年の時点は as_of と同じ月日。"""
    fig = fig[fig["year"].isin(years)]
    rows = []
    for y in years:
        cutoff = same_monthday(as_of, y)
        for a in ages:
            rows.append({"year": y, "age": a, "cohort": y - a,
                         **cohort_values(fig, year=y, cutoff=cutoff, cohort=y - a)})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def yearly_comparison(table: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """仮説①: 今年の世代 − 過去の同じ年齢の世代の平均。順位は今年を含めた中での順位（1 が最強）。"""
    rows = []
    for age, g in table.groupby("age"):
        cur = g[g["year"] == year]
        if cur.empty:
            continue
        past = g[g["year"] != year]
        for m in METRICS:
            value = float(cur[m].iloc[0])
            vals = g[m].dropna()
            past_mean = float(past[m].mean())
            rows.append({"age": int(age), "cohort": int(cur["cohort"].iloc[0]), "metric": m,
                         "current": value, "past_mean": past_mean, "diff": value - past_mean,
                         "rank": int((vals > value).sum() + 1) if not np.isnan(value) else np.nan,
                         "n_years": int(vals.size)})
    return pd.DataFrame(rows, columns=["age", "cohort", "metric", "current", "past_mean", "diff",
                                       "rank", "n_years"])


def trajectory(table: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """仮説②: 今年3歳以上の各世代の、各年齢での「その年齢の他の世代の平均との差」。"""
    rows = []
    now = table[(table["year"] == year) & (table["age"] >= MIN_RANKING_AGE)]
    for cohort in sorted(now["cohort"].unique(), reverse=True):
        for r in table[table["cohort"] == cohort].sort_values("age").itertuples():
            others = table[(table["age"] == r.age) & (table["year"] != r.year)]
            for m in METRICS:
                value = float(getattr(r, m))
                rows.append({"cohort": int(cohort), "age_now": int(year - cohort), "age": int(r.age),
                             "year": int(r.year), "metric": m, "value": value,
                             "deviation": value - float(others[m].mean())})
    return pd.DataFrame(rows, columns=["cohort", "age_now", "age", "year", "metric", "value", "deviation"])


def ranking_table(table: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """年 × 年齢（3歳以上）の数値。今年の並びが例年と違うかを見る（§5.3）。"""
    t = table[table["age"] >= MIN_RANKING_AGE]
    return t.pivot(index="year", columns="age", values=metric)


def pair_differences(table: pd.DataFrame, *, year: int, pairs, curve_levels: dict | None,
                     half: int) -> pd.DataFrame:
    """見立ての照合（§5.3）。そのままの差と、年齢曲線で年齢分を差し引いた差。"""
    for a, b in pairs:
        if min(a, b) < MIN_RANKING_AGE:
            raise ValueError("2歳は今いる世代の並びに入れません（仕様 §5.3）")
    now = table[table["year"] == year].set_index("age")
    rows = []
    for a, b in pairs:
        if curve_levels is None:
            gap = float("nan")
        else:
            gap = curve_levels.get((a, half), float("nan")) - curve_levels.get((b, half), float("nan"))
        for m in METRICS:
            diff = float(now.at[a, m]) - float(now.at[b, m])
            rows.append({"pair": f"{a}-{b}", "stronger": a, "weaker": b, "metric": m,
                         "diff": diff, "age_gap": gap, "adjusted": diff - gap})
    return pd.DataFrame(rows, columns=["pair", "stronger", "weaker", "metric", "diff", "age_gap", "adjusted"])


def verdict(lo: float, hi: float) -> str:
    """区間から照合の判定（§5.3）。差は「見立てで強い側 − 弱い側」なので正が見立ての向き。"""
    if np.isnan(lo) or np.isnan(hi):
        return "判定できない"
    if lo > 0:
        return "見立てと合う"
    if hi < 0:
        return "見立てと逆"
    return "差があるとは言えない"
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation.py -v`
Expected: 12 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/evaluate/generation.py tests/test_generation.py
git commit -m "feat: 世代の数値・例年比・世代の歩み・見立ての照合を出す

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 開催日単位のブートストラップ（`replicate_stats` / `resample_days` / `bootstrap_intervals`）

**Files:**
- Modify: `src/keiba/evaluate/generation.py`
- Modify: `tests/fixtures/figure_runs.py`
- Modify: `tests/test_generation.py`

**Interfaces:**
- Consumes: Task 2 の `fit_model`, `apply_figures`、Task 3 の `half_of`、Task 4 の関数
- Produces:
  - `replicate_stats(fig, *, as_of: date, years: list[int], ages: list[int], curve_levels: dict | None) -> dict[str, float]`。
    キーの形: `value|<year>|<age>|<metric>`、`yearly|<age>|<metric>`、`pair_raw|<a>-<b>|<metric>`、
    `pair_adj|<a>-<b>|<metric>`、`traj|<cohort>|<age>|<metric>`
  - `resample_days(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame`
  - `bootstrap_intervals(runs, *, as_of, years, ages, curve_levels, n_boot: int, seed: int) -> pd.DataFrame`（index = 統計のキー、列 `lo`, `hi`）
  - `tests/fixtures/figure_runs.py`: `season_runs(*, years=(2024, 2025), days_per_year=4, field=12, seed=0) -> pd.DataFrame`

- [ ] **Step 1: 作り物の複数年の走を足す**

`tests/fixtures/figure_runs.py` の import に `import numpy as np` を足し、末尾に足す:

```python
def season_runs(*, years=(2024, 2025), days_per_year=4, field=12, seed=0) -> pd.DataFrame:
    """複数年の5月の開催日に、古馬戦（3〜5歳）と2歳戦（翌年分の世代）を並べる。

    年齢ごとに馬を field 頭ずつ作り、毎日すべての馬が走る。タイムは基準 + 馬場差 + 馬の地力 + 雑音。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        ability = {age: rng.normal(0, 0.3, field) for age in (2, 3, 4, 5)}
        for d in range(days_per_year):
            day = f"{year}05{d + 1:02d}"
            v = float(rng.normal(0, 0.2))
            for kyori, s in S_TRUE.items():
                race_id = f"{day}-{kyori}-O"
                runners = [(age, i) for age in (3, 4, 5) for i in range(field)]
                times = [(s - ability[age][i] + v + rng.normal(0, 0.05)) * kyori / 1000
                         for age, i in runners]
                order = np.argsort(times)
                for rank, k in enumerate(order, start=1):
                    age, i = runners[k]
                    rows.append(run_row(race_id=race_id, umaban=f"{k + 1:02d}", ketto_num=f"C{year - age}-{i}",
                                        day=day, date=pd.Timestamp(day), year=year, kyori_m=kyori,
                                        time_sec=times[k], rank=float(rank), barei=age, cohort=year - age))
            young_id = f"{day}-1600-Y"
            for i in range(field):
                rows.append(run_row(race_id=young_id, umaban=f"{i + 1:02d}", ketto_num=f"C{year - 2}-{i}",
                                    day=day, date=pd.Timestamp(day), year=year, syubetu_cd="11",
                                    class_level=1.0,
                                    time_sec=(S_TRUE[1600] + 1.0 - ability[2][i] + v) * 1.6,
                                    rank=float(i + 1), barei=2, cohort=year - 2))
    return pd.DataFrame(rows)
```

- [ ] **Step 2: 失敗するテストを足す**

`tests/test_generation.py` の import に次を足し、末尾にテストを足す:

```python
from keiba.evaluate.generation import bootstrap_intervals, replicate_stats, resample_days
from keiba.features.speed_figure import apply_figures, fit_model
from tests.fixtures.figure_runs import season_runs
```

```python
def _season_fig():
    runs = season_runs()
    return runs, apply_figures(runs, fit_model(runs))


def test_replicate_stats_keys():
    _, fig = _season_fig()
    stats = replicate_stats(fig, as_of=date(2025, 10, 4), years=[2024, 2025], ages=[2, 3, 4, 5],
                            curve_levels={(3, 1): 0.0, (4, 1): 0.2, (5, 1): 0.1})
    for key in ("value|2025|3|median", "yearly|2|median", "pair_raw|4-5|median",
                "pair_adj|4-3|median", "traj|2021|3|median"):
        assert key in stats
    assert not any(k.startswith("pair_") and ("-2|" in k or "|2-" in k) for k in stats)


def test_resample_days_keeps_each_day_whole():
    runs, _ = _season_fig()
    out = resample_days(runs, np.random.default_rng(0))
    assert out["day"].str.split("#").str[0].isin(runs["day"]).all()
    sizes = runs.groupby("day").size()
    for label, g in out.groupby("day"):
        assert len(g) == sizes[label.split("#")[0]]       # 選ばれた日は丸ごと入る
    assert out["day"].nunique() == runs["day"].nunique()   # 復元抽出の回数 = 日数（同じ日の写しは別名）


def test_bootstrap_is_reproducible_and_brackets_point():
    runs, fig = _season_fig()
    kw = dict(as_of=date(2025, 10, 4), years=[2024, 2025], ages=[2, 3, 4, 5], curve_levels=None)
    a = bootstrap_intervals(runs, n_boot=15, seed=11, **kw)
    b = bootstrap_intervals(runs, n_boot=15, seed=11, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["lo", "hi"]
    point = replicate_stats(fig, **kw)["value|2025|4|median"]
    lo, hi = a.loc["value|2025|4|median"]
    assert lo - 0.5 <= point <= hi + 0.5
```

- [ ] **Step 3: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation.py -v`
Expected: FAIL（`ImportError: cannot import name 'bootstrap_intervals'`）

- [ ] **Step 4: 実装する**

`src/keiba/evaluate/generation.py` の import に次を足す:

```python
from keiba.evaluate.age_curve import half_of
from keiba.features.speed_figure import apply_figures, fit_model
```

末尾に足す:

```python
def replicate_stats(fig: pd.DataFrame, *, as_of: date, years: list[int], ages: list[int],
                    curve_levels: dict | None) -> dict[str, float]:
    """1回の推定から出る統計を、ブートストラップで集めやすい「キー → 値」の形にする。"""
    table = cohort_table(fig, as_of=as_of, ages=ages, years=years)
    out: dict[str, float] = {}
    for r in table.itertuples():
        for m in METRICS:
            out[f"value|{r.year}|{r.age}|{m}"] = float(getattr(r, m))
    for r in yearly_comparison(table, year=as_of.year).itertuples():
        out[f"yearly|{r.age}|{r.metric}"] = float(r.diff)
    pairs = [p for p in HYPOTHESIS_PAIRS if set(p) <= set(ages)]
    if pairs:
        pdf = pair_differences(table, year=as_of.year, pairs=pairs, curve_levels=curve_levels,
                               half=half_of(as_of))
        for r in pdf.itertuples():
            out[f"pair_raw|{r.pair}|{r.metric}"] = float(r.diff)
            out[f"pair_adj|{r.pair}|{r.metric}"] = float(r.adjusted)
    for r in trajectory(table, year=as_of.year).itertuples():
        out[f"traj|{r.cohort}|{r.age}|{r.metric}"] = float(r.deviation)
    return out


def resample_days(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """開催日を復元抽出する（§5.5）。同じ日が2回選ばれたら別の日として扱う（day と race_id に #番号）。"""
    groups = runs.groupby("day", sort=True).indices
    days = np.array(sorted(groups))
    pick = rng.choice(days, size=len(days), replace=True)
    idx = np.concatenate([groups[d] for d in pick])
    copy_no = np.repeat(np.arange(len(pick)), [len(groups[d]) for d in pick])
    out = runs.iloc[idx].reset_index(drop=True)
    tag = pd.Series(copy_no).astype(str).radd("#")
    out["day"] = out["day"].astype(str) + tag
    out["race_id"] = out["race_id"].astype(str) + tag
    return out


def bootstrap_intervals(runs: pd.DataFrame, *, as_of: date, years: list[int], ages: list[int],
                        curve_levels: dict | None, n_boot: int, seed: int) -> pd.DataFrame:
    """開催日単位のクラスタ・ブートストラップで 95% 区間（§5.5）。物差しの推定からやり直す。

    年齢分を差し引いた差では curve_levels を固定する（曲線の誤差は区間に入らない）。
    """
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        rs = resample_days(runs, rng)
        fig = apply_figures(rs, fit_model(rs))
        reps.append(replicate_stats(fig, as_of=as_of, years=years, ages=ages, curve_levels=curve_levels))
    df = pd.DataFrame(reps)
    return pd.DataFrame({"lo": df.quantile(0.025), "hi": df.quantile(0.975)})
```

- [ ] **Step 5: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation.py tests/test_speed_figure.py tests/test_age_curve.py -v`
Expected: すべて passed（15 + 14 + 11 = 40）

- [ ] **Step 6: コミット**

```bash
git add src/keiba/evaluate/generation.py tests/test_generation.py tests/fixtures/figure_runs.py
git commit -m "feat: 世代の数値に開催日単位のブートストラップ区間を付ける

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: レポートの組み立て（`generation_report.py`）

**Files:**
- Create: `src/keiba/evaluate/generation_report.py`
- Create: `tests/test_generation_report.py`

**Interfaces:**
- Consumes: Task 2〜5 の出力
- Produces:
  - `ExampleHorse`（frozen dataclass: `umaban: str, bamei: str, age: int, cohort: int`）
  - `ReportInputs`（frozen dataclass。下のコードのとおり）
  - `render(inputs: ReportInputs) -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_generation_report.py`:

```python
"""report.md の組み立て（仕様 §8）。欠損・データ不足でも落ちずに「—」「データ不足」を出す。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from keiba.evaluate.age_curve import CURVE_COLUMNS, age_curve
from keiba.evaluate.generation import cohort_table, replicate_stats
from keiba.evaluate.generation_report import ExampleHorse, ReportInputs, render
from keiba.features.speed_figure import apply_figures, fit_model
from tests.fixtures.figure_runs import season_runs

AS_OF = date(2025, 10, 4)
YEARS = [2024, 2025]
AGES = [2, 3, 4, 5]


def _inputs(**over) -> ReportInputs:
    runs = season_runs()
    model = fit_model(runs)
    fig = apply_figures(runs, model)
    table = cohort_table(fig, as_of=AS_OF, ages=AGES, years=YEARS)
    point = replicate_stats(fig, as_of=AS_OF, years=YEARS, ages=AGES, curve_levels=None)
    ci = pd.DataFrame({"lo": {k: v - 0.1 for k, v in point.items()},
                       "hi": {k: v + 0.1 for k, v in point.items()}})
    curves = {"全体": age_curve(fig, years=YEARS, before=AS_OF, n_boot=5, seed=1),
              "芝": pd.DataFrame(columns=CURVE_COLUMNS), "ダート": pd.DataFrame(columns=CURVE_COLUMNS)}
    base = dict(as_of=AS_OF, version_id="TEST", years=YEARS, ages=AGES, model=model, fig=fig,
                table=table, point=point, ci=ci, curves=curves, repeat=(0.5, 100), n_boot=5, seed=1,
                examples={"2025100405040211": [ExampleHorse("01", "テストウマ", 3, 2022)]})
    base.update(over)
    return ReportInputs(**base)


def test_report_has_all_sections():
    text = render(_inputs())
    for heading in ("# 世代レベル", "## タイム指数", "## 仮説① 同じ年齢での世代比較",
                    "## 仮説② 世代の歩み", "## 見立て（4歳 > 5歳 > 3歳）の照合",
                    "## 仮説③ 年齢曲線", "## 例: 2025100405040211", "## 既知の偏り"):
        assert heading in text
    assert "精度の証拠としては扱わない" in text


def test_report_shows_missing_standard_count():
    inputs = _inputs()
    fig = inputs.fig.copy()
    fig.loc[fig.index[:3], "has_standard"] = False
    text = render(_inputs(fig=fig))
    assert "基準タイムが無い走: 3 件" in text


def test_report_renders_missing_values_as_dash():
    inputs = _inputs()
    point = {k: float("nan") for k in inputs.point}
    text = render(_inputs(point=point))
    assert "—" in text and "判定できない" in text


def test_report_handles_empty_curve():
    text = render(_inputs())
    assert "### 芝\n\nデータ不足" in text


def test_ranking_section_has_no_two_year_olds():
    text = render(_inputs())
    section = text.split("## 見立て（4歳 > 5歳 > 3歳）の照合")[1].split("## 仮説③")[0]
    assert "2歳" not in section
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation_report.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'keiba.evaluate.generation_report'`）

- [ ] **Step 3: 実装する**

`src/keiba/evaluate/generation_report.py`:

```python
"""世代レベルの report.md を組み立てる（仕様 §8）。計算はしない。並べて言葉を付けるだけ。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from keiba.evaluate.age_curve import peak_summary
from keiba.evaluate.generation import (HYPOTHESIS_PAIRS, METRICS, MIN_RANKING_AGE, ranking_table,
                                       verdict)
from keiba.features.speed_figure import FigureModel

METRIC_JA = {"top10": "上位10頭", "top50": "上位50頭", "median": "中央値"}


@dataclass(frozen=True)
class ExampleHorse:
    umaban: str
    bamei: str
    age: int
    cohort: int


@dataclass(frozen=True)
class ReportInputs:
    as_of: date
    version_id: str
    years: list[int]
    ages: list[int]
    model: FigureModel
    fig: pd.DataFrame
    table: pd.DataFrame
    point: dict[str, float]
    ci: pd.DataFrame                    # index = 統計のキー、列 lo / hi
    curves: dict[str, pd.DataFrame]     # "全体" / "芝" / "ダート"
    repeat: tuple[float, int]
    n_boot: int
    seed: int
    examples: dict[str, list[ExampleHorse]] = field(default_factory=dict)


def _num(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


def _plain(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def _ci(inp: ReportInputs, key: str) -> tuple[float, float]:
    if key not in inp.ci.index:
        return float("nan"), float("nan")
    return float(inp.ci.at[key, "lo"]), float(inp.ci.at[key, "hi"])


def _with_ci(inp: ReportInputs, key: str) -> str:
    lo, hi = _ci(inp, key)
    return f"{_num(inp.point.get(key, float('nan')))} [{_num(lo)}, {_num(hi)}]"


def _cohort_name(cohort: int, age: int, year_now: int, year: int) -> str:
    if year == year_now:
        return f"今年の{age}歳（{cohort}年生まれ）"
    age_now = year_now - cohort
    return f"{cohort}年生まれ（今の{age_now}歳）の{age}歳時"


def _section_figures(inp: ReportInputs) -> list[str]:
    fig, m = inp.fig, inp.model
    n = len(fig)
    no_std = int((~fig["has_standard"]).sum())
    no_var = int((fig["has_standard"] & ~fig["has_variant"]).sum())
    r, pairs = inp.repeat
    return [
        "## タイム指数", "",
        "- 指数 =（基準タイム + その日の馬場差 − 走破タイム）を 1600m 相当の秒にしたもの。正ほど速い",
        f"- 基準タイム・クラス差・馬場差は古馬戦の1着だけから推定（{m.n_races:,} レース）。"
        f"馬場差の縮小: σ² = {m.sigma2:.4f}、τ² = {m.tau2:.4f}",
        f"- 対象の走: {n:,} 件。基準タイムが無い走: {no_std} 件（{100 * no_std / max(n, 1):.1f}%、指数は欠損）。"
        f"馬場差が無い走（古馬戦の無い開催面、馬場差 0 で計算）: {no_var} 件",
        f"- 再現性（同じ馬の続けて2走、120 日以内）: 相関 {_plain(r)}（{pairs:,} 組）",
        f"- 比べた年: {', '.join(map(str, inp.years))}。区間は開催日単位のブートストラップ"
        f"（{inp.n_boot} 回、種 {inp.seed}）の 95%",
        "",
    ]


def _section_yearly(inp: ReportInputs) -> list[str]:
    out = ["## 仮説① 同じ年齢での世代比較（例年比）", "",
           f"各年の {inp.as_of.month}/{inp.as_of.day} 時点、その年の 1/1 からの馬ごとの最高値で比べる。", ""]
    year_now = inp.as_of.year
    for age in inp.ages:
        t = inp.table[inp.table["age"] == age].sort_values("year")
        out += [f"### {age}歳", "", "| 世代 | 年 | 上位10頭 | 上位50頭 | 中央値 | 頭数 |", "|---|---|---|---|---|---|"]
        for r in t.itertuples():
            out.append(f"| {_cohort_name(int(r.cohort), age, year_now, int(r.year))} | {r.year} | "
                       f"{_plain(r.top10)} | {_plain(r.top50)} | {_plain(r.median)} | {r.n} |")
        diffs = "、".join(f"{METRIC_JA[m]} {_with_ci(inp, f'yearly|{age}|{m}')}" for m in METRICS)
        out += ["", f"今年 − 過去の平均: {diffs}", ""]
    return out


def _section_trajectory(inp: ReportInputs) -> list[str]:
    year_now = inp.as_of.year
    ages = [a for a in inp.ages if a >= MIN_RANKING_AGE]
    max_age = max(inp.ages)
    out = ["## 仮説② 世代の歩み", "",
           "各世代が各年齢だったときの「その年齢の他の世代の平均との差」（上位10頭）。空欄は比べてよい年でない。", "",
           "| 世代 | " + " | ".join(f"{a}歳時" for a in range(2, max_age + 1)) + " |",
           "|---|" + "---|" * (max_age - 1)]
    for age_now in sorted(ages):
        cohort = year_now - age_now
        cells = []
        for a in range(2, max_age + 1):
            key = f"traj|{cohort}|{a}|top10"
            cells.append(_with_ci(inp, key) if key in inp.point else "")
        out.append(f"| {cohort}年生まれ（今の{age_now}歳） | " + " | ".join(cells) + " |")
    out += ["", "世代の数が少ないので、年齢を重ねても差が残るかは記述にとどめる。", ""]
    return out


def _section_ranking(inp: ReportInputs) -> list[str]:
    out = ["## 見立て（4歳 > 5歳 > 3歳）の照合", "",
           "差は「見立てで強い側 − 弱い側」。正なら見立ての向き。", "",
           "| 組 | 指標 | そのままの差 [95%] | 判定 | 年齢分を差し引いた差 [95%] | 判定 |",
           "|---|---|---|---|---|---|"]
    for a, b in HYPOTHESIS_PAIRS:
        if not {a, b} <= set(inp.ages):
            continue
        for m in METRICS:
            raw, adj = f"pair_raw|{a}-{b}|{m}", f"pair_adj|{a}-{b}|{m}"
            out.append(f"| {a}歳 − {b}歳 | {METRIC_JA[m]} | {_with_ci(inp, raw)} | {verdict(*_ci(inp, raw))} | "
                       f"{_with_ci(inp, adj)} | {verdict(*_ci(inp, adj))} |")
    rt = ranking_table(inp.table, metric="top10")
    out += ["", "例年の同じ月日の並び（上位10頭）:", "",
            "| 年 | " + " | ".join(f"{a}歳" for a in rt.columns) + " |", "|---|" + "---|" * len(rt.columns)]
    for year, row in rt.iterrows():
        out.append(f"| {year} | " + " | ".join(_plain(v) for v in row.to_numpy()) + " |")
    out += ["", "3歳は斤量の減量の分だけ有利に出る（斤量は補正していない）。", ""]
    return out


def _curve_table(curve: pd.DataFrame) -> list[str]:
    if curve.empty:
        return ["データ不足", ""]
    out = ["| 期間 | 前の期間からの変化 [95%] | 頭数 | 水準（3歳後半 = 0）[95%] |", "|---|---|---|---|"]
    for r in curve.itertuples():
        out.append(f"| {r.label} | {_num(r.delta)} [{_num(r.delta_lo)}, {_num(r.delta_hi)}] | {r.n} | "
                   f"{_num(r.level)} [{_num(r.level_lo)}, {_num(r.level_hi)}] |")
    s = peak_summary(curve)
    out += ["", f"全盛期: {s['peak'] or '—'}（水準 {_num(s['peak_level'])}）。"
                f"そこから1年後の落ち幅: {_plain(s['drop_1y'])} 秒", ""]
    return out


def _section_curve(inp: ReportInputs) -> list[str]:
    out = ["## 仮説③ 年齢曲線", "",
           "同じ馬の半年ごとの変化（期間の値 = 上位2走の平均、2走以上の期間だけ）を積み上げる。", ""]
    for name in ("全体", "芝", "ダート"):
        out += [f"### {name}", ""] + _curve_table(inp.curves.get(name, pd.DataFrame()))
    out += ["大きく衰えた馬はそのまま引退して変化が残らないので、衰えは小さめに出る。", ""]
    return out


def _section_examples(inp: ReportInputs) -> list[str]:
    out = []
    for race_id, horses in inp.examples.items():
        out += [f"## 例: {race_id}", "",
                "結果を見てから作った指標なので、**精度の証拠としては扱わない**。各馬の世代の例年比（上位10頭）。", "",
                "| 馬番 | 馬名 | 年齢 | 世代 | 世代の例年比 [95%] |", "|---|---|---|---|---|"]
        for h in horses:
            out.append(f"| {h.umaban} | {h.bamei} | {h.age} | {h.cohort}年生まれ | "
                       f"{_with_ci(inp, f'yearly|{h.age}|top10')} |")
        out.append("")
    return out


def render(inp: ReportInputs) -> str:
    lines = [f"# 世代レベル（{inp.as_of.isoformat()} 時点）", "",
             f"版 `{inp.version_id}`。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md。"
             "Jev は使っていない。", ""]
    lines += _section_figures(inp) + _section_yearly(inp) + _section_trajectory(inp)
    lines += _section_ranking(inp) + _section_curve(inp) + _section_examples(inp)
    lines += ["## 既知の偏り", "",
              "- スローペースのレースは時計が遅く出る（最高値・上位2走で和らげるが消えない）",
              "- 斤量は補正しない。3歳と古馬の並びでは3歳が減量の分だけ有利に出る",
              "- 比べられる過去の世代は少ない（例年比の「過去の平均」は粗い）",
              "- 古馬戦にも6月以降は3歳が混ざるので、3歳世代の強弱はわずかに馬場差へ漏れる",
              "- 年齢曲線は衰えを小さめに出す", ""]
    return "\n".join(lines)
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation_report.py -v`
Expected: 5 passed

- [ ] **Step 5: コミット**

```bash
git add src/keiba/evaluate/generation_report.py tests/test_generation_report.py
git commit -m "feat: 世代レベルのレポートを組み立てる

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 実行スクリプト、実データでの実行、記録

**Files:**
- Create: `scripts/generation_level.py`
- Create: `tests/test_generation_level_script.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: Task 1〜6 のすべて、`keiba.data.store.DataStore`、`keiba.ingest.store.read_manifest` / `version_dir`、`config.DATA_DIR`
- Produces: `main(argv: list[str] | None = None) -> int`、`coverage_start(manifest: dict, runs: pd.DataFrame) -> date`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_generation_level_script.py`:

```python
"""scripts/generation_level.py の端から端まで（小さな作り物の版で）。Jev は呼ばない。"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generation_level  # noqa: E402

from tests.fixtures import csv_data as fx  # noqa: E402
from tests.fixtures.figure_runs import season_runs  # noqa: E402


def write_version(tmp_path: Path) -> Path:
    runs = season_runs(years=(2023, 2024, 2025), days_per_year=3)
    races, horses = [], []
    for race_id, g in runs.groupby("race_id"):
        r = g.iloc[0]
        races.append(dict(race_id=race_id, hasso_at=f"{r.date:%Y-%m-%d}T15:00:00+09:00", jyo_cd=r.jyo_cd,
                          track_cd=r.track_cd, track_kind=r.track_kind, kyori_m=str(r.kyori_m),
                          syubetu_cd=r.syubetu_cd, class_level=str(int(r.class_level)),
                          data_kubun="7", make_date=f"{r.date:%Y%m%d}"))
        for h in g.itertuples():
            horses.append(dict(race_id=race_id, umaban=h.umaban, ketto_num=h.ketto_num, bamei=h.ketto_num,
                               barei=f"{h.barei:02d}", result_time_sec=f"{h.time_sec:.1f}",
                               result_kakutei_jyuni=str(int(h.rank)), result_ijyo_cd="0",
                               data_kubun="7", make_date=f"{h.date:%Y%m%d}"))
    return fx.write_dataset(tmp_path / "data", races, horses, [])


def test_script_writes_all_outputs(tmp_path):
    data_dir = write_version(tmp_path)
    out_dir = tmp_path / "out"
    code = generation_level.main(["--as-of", "20251004", "--data-dir", str(data_dir), "--out-dir", str(out_dir),
                                  "--bootstrap", "3", "--example-race", "20250501-1600-O"])
    assert code == 0
    for name in ("report.md", "cohorts.csv", "age_curve.csv", "figures.csv", "manifest.json"):
        assert (out_dir / name).exists(), name
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## 例: 20250501-1600-O" in report
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["as_of"] == "2025-10-04" and manifest["years"] == [2024, 2025]
    cohorts = pd.read_csv(out_dir / "cohorts.csv")
    assert {"year", "age", "cohort", "top10_lo", "median_hi"} <= set(cohorts.columns)


def test_script_runs_early_in_season(tmp_path):
    """Review Focus 2: 2月の時点（2歳は0頭、他も10頭未満）でも落ちない。"""
    data_dir = write_version(tmp_path)
    code = generation_level.main(["--as-of", "20250210", "--data-dir", str(data_dir),
                                  "--out-dir", str(tmp_path / "out"), "--bootstrap", "2"])
    assert code == 0


def test_coverage_start_uses_fromdate():
    runs = pd.DataFrame({"date": [pd.Timestamp("1989-04-01"), pd.Timestamp("2022-01-01")]})
    assert generation_level.coverage_start({"fromdate": "20210901000000"}, runs) == date(2021, 9, 1)


def test_coverage_start_falls_back_to_first_run():
    runs = pd.DataFrame({"date": [pd.Timestamp("2023-05-01"), pd.Timestamp("2024-05-01")]})
    assert generation_level.coverage_start({}, runs) == date(2023, 5, 1)
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation_level_script.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'generation_level'`）

- [ ] **Step 3: 実装する**

`scripts/generation_level.py`:

```python
"""世代レベル（タイム指数による世代の強さ）のレポートを書く。Jev は呼ばない（課金なし）。

    .venv/Scripts/python.exe scripts/generation_level.py --as-of 20261004 \\
        --data-dir tmp/keiba-report/oct04_2026/data --example-race 2026100405040211

仕様: docs/superpowers/specs/2026-10-04-generation-level-design.md
出力: data/generation/<as_of>/（report.md / cohorts.csv / age_curve.csv / figures.csv / manifest.json）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate.age_curve import age_curve, curve_levels  # noqa: E402
from keiba.evaluate.generation import (METRICS, bootstrap_intervals, cohort_table,  # noqa: E402
                                       comparable_years, replicate_stats)
from keiba.evaluate.generation_report import ExampleHorse, ReportInputs, render  # noqa: E402
from keiba.features.speed_figure import apply_figures, fit_model, prepare_runs, repeatability  # noqa: E402
from keiba.ingest.store import read_manifest, version_dir  # noqa: E402

JST = timezone(timedelta(hours=9))
SURFACES = (("全体", None), ("芝", "turf"), ("ダート", "dirt"))


def coverage_start(manifest: dict, runs: pd.DataFrame) -> date:
    """比べてよい年を決める取得開始日。manifest の fromdate、無ければ対象の走の最初の日。

    coverage.first_hasso_at は訂正再配信の古いレース（1989年など）が混じるので使わない。
    """
    fromdate = manifest.get("fromdate")
    if fromdate:
        return datetime.strptime(fromdate[:8], "%Y%m%d").date()
    return runs["date"].min().date()


def _examples(store: DataStore, race_ids: list[str]) -> dict[str, list[ExampleHorse]]:
    out = {}
    for race_id in race_ids:
        rows = store.race_horses[store.race_horses["race_id"] == race_id]
        rows = rows.sort_values("make_date").drop_duplicates("umaban", keep="last").sort_values("umaban")
        year = int(race_id[:4])
        out[race_id] = [ExampleHorse(umaban=r.umaban, bamei=r.bamei, age=int(r.barei),
                                     cohort=year - int(r.barei)) for r in rows.itertuples()]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", required=True, help="時点 YYYYMMDD（その日の 0:00 より前のレースだけ使う）")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="既定 data/generation/<as_of>")
    parser.add_argument("--ages", default="2,3,4,5")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--example-race", action="append", default=[], help="例として載せるレース（繰り返し可）")
    args = parser.parse_args(argv)

    as_of = datetime.strptime(args.as_of, "%Y%m%d").date()
    as_of_dt = datetime(as_of.year, as_of.month, as_of.day, tzinfo=JST)
    requested = sorted({int(a) for a in args.ages.split(",")})
    ages = list(range(2, max(requested) + 1))          # 世代の歩み（§5.4）は2歳時から並べる
    out_dir = args.out_dir or (config.DATA_DIR / "generation" / args.as_of)

    store = DataStore.open_current(args.data_dir)
    manifest = read_manifest(version_dir(args.data_dir, store.version_id))
    runs = prepare_runs(store.races, store.race_horses, as_of=as_of_dt)
    model = fit_model(runs)
    fig = apply_figures(runs, model)
    years = comparable_years(coverage_start(manifest, runs), as_of.year)
    print(f"対象の走 {len(runs):,} 件、古馬戦 {model.n_races:,} レース、比べる年 {years}", flush=True)

    curves = {name: age_curve(fig, years=years, before=as_of, surface=sf, n_boot=args.bootstrap, seed=args.seed)
              for name, sf in SURFACES}
    levels = curve_levels(curves["全体"])
    point = replicate_stats(fig, as_of=as_of, years=years, ages=ages, curve_levels=levels)
    print(f"ブートストラップ {args.bootstrap} 回……", flush=True)
    ci = bootstrap_intervals(runs, as_of=as_of, years=years, ages=ages, curve_levels=levels,
                             n_boot=args.bootstrap, seed=args.seed)
    table = cohort_table(fig, as_of=as_of, ages=ages, years=years)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = render(ReportInputs(as_of=as_of, version_id=store.version_id, years=years, ages=ages, model=model,
                                 fig=fig, table=table, point=point, ci=ci, curves=curves,
                                 repeat=repeatability(fig), n_boot=args.bootstrap, seed=args.seed,
                                 examples=_examples(store, args.example_race)))
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    cohorts = table.copy()
    for m in METRICS:
        keys = [f"value|{r.year}|{r.age}|{m}" for r in table.itertuples()]
        cohorts[f"{m}_lo"] = [ci.at[k, "lo"] if k in ci.index else float("nan") for k in keys]
        cohorts[f"{m}_hi"] = [ci.at[k, "hi"] if k in ci.index else float("nan") for k in keys]
    cohorts.to_csv(out_dir / "cohorts.csv", index=False)
    pd.concat([c.assign(surface=name) for name, c in curves.items()]).to_csv(out_dir / "age_curve.csv", index=False)
    fig[["race_id", "umaban", "ketto_num", "cohort", "barei", "figure", "jyo_cd", "track_cd", "kyori_m",
         "day", "track_kind", "has_standard", "has_variant"]].to_csv(out_dir / "figures.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "data_version": store.version_id, "as_of": as_of.isoformat(), "years": years, "ages": ages,
        "bootstrap": args.bootstrap, "seed": args.seed, "sigma2": model.sigma2, "tau2": model.tau2,
        "n_races": model.n_races, "examples": args.example_race,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation_level_script.py -v`
Expected: 4 passed

続けて全体: `.venv/Scripts/python.exe -m pytest`
Expected: 既存 1336 件 + 新規（14 + 11 + 15 + 5 + 4 = 49）件がすべて passed

- [ ] **Step 5: 実データで回す**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/generation_level.py --as-of 20261004 \
    --data-dir tmp/keiba-report/oct04_2026/data \
    --example-race 2026100405040211 --example-race 2026100408040211
```

Expected: `data/generation/20261004/` に5ファイル。`report.md` を開いて次を確かめ、異常があれば止めて報告する:
- 比べた年が `2022, 2023, 2024, 2025, 2026`（2021 が入っていない）
- 基準タイムが無い走の割合（大きければ、2歳戦にしか無いキーの多さとしてそのまま報告する）
- 再現性の相関が 0〜1 の範囲で、極端に低くない（0.3 未満なら報告する）
- 年齢曲線の「全体」の表が空でない

- [ ] **Step 6: CLAUDE.md に記録する**

`CLAUDE.md` の「### 払戻・オッズ（2026-09-26）」の節の後ろに次を足す（数値は Step 5 の report.md から写す）:

```markdown
### 世代レベル（2026-10-04、第1段階）

- 走破タイムから全年齢共通のタイム指数を作り、世代（生まれ年）の強さを測る単体の指標。**予想には組み込んでいない**
  （第2段階で別の仕様）。仕様 `docs/superpowers/specs/2026-10-04-generation-level-design.md`
- `scripts/generation_level.py --as-of YYYYMMDD [--data-dir DIR] [--example-race RACE_ID]`。Jev は呼ばない。
  出力は `data/generation/<as_of>/`
- 基準タイム・馬場差は**古馬戦の1着だけ**から推定する（2歳戦を混ぜると世代の弱さが馬場差に吸収される）
- 2026-10-04 時点の結果: <report.md の仮説①②③と見立ての照合の要点を1〜3行で>
```

- [ ] **Step 7: コミット**

```bash
git add scripts/generation_level.py tests/test_generation_level_script.py CLAUDE.md
git commit -m "feat: 世代レベルのレポートを書くスクリプトを追加し、10/4 時点の結果を記録

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

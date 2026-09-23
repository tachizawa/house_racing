# 3着以内予想（案A・案B）実装プラン

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 「実際の上位3着に入る馬」を選ぶために、(1) 1レース分の予想用資料を API を呼ばずに JSON / Markdown で出力し、(2) 各馬の「確定着順3着以内」確率を Jev に聞く案A（1回方式）と案B（展開予測→3着内の2段階）を実装し、(3) Python だけの基準・同じ state での勝率上位3頭と、3着内の評価指標で比較できるようにする。

**Architecture:** 既存の第1層/第2層パイプライン（勝率用）には手を入れず、並列に「3着内」系統を足す。資料（dossier）は既存の `build_context` の上に、コース識別・過去走の行・条件の組み合わせ集計・履歴の欠け方の区別を積んで作る。資料そのものを Jev の state にし、案Aは1リクエストに「馬ごとの Noul（3着以内か）」と「同じ state での winner Choice」をまとめる。案Bは先に pace Choice を1回聞き、その確率分布を state に足してから案Aと同じ質問を送る。評価はレース単位、開催日クラスタのブートストラップで差の区間を出す。

**Tech Stack:** Python 3.13（`.venv`）、pytest、pandas、`typesafe-sdk` 0.7.0（`Noul` / `NoulCriteria` / `Choice`）、既存の `keiba.*` パッケージ。

**Spec:** `tmp/keiba-report/report3-top3-prediction-design.md`（設計提案。Task 1 で `docs/superpowers/specs/2026-09-23-top3-prediction-design.md` に写す）。既存の拘束力のある仕様 `docs/superpowers/specs/2026-09-22-jev-prediction-core-design.md` の不変条件（`as_of` ゲート、`EntryRow`/`ResultRow` の分離、欠損を 0 や不適性に置き換えない）はこのプランでもすべて有効。

## Global Constraints

- コマンドは `.venv/Scripts/python.exe` を明示して使う（PowerShell では `.venv\Scripts\python.exe`）。
- **git はコミット0件で、最初のコミットを打つかは所有者の判断待ち**（CLAUDE.md）。各タスク末尾の「区切り」ステップは、所有者がコミットを許可していればそのメッセージでコミットし、許可が無ければ**コミットせずにテスト全体の合格だけを確認する**。
- `.gitignore` の `/data/` `/vendor/` `/tmp/` の先頭スラッシュを戻さない。
- 予想側のコードは `ResultRow` を受け取らない。対象レースの結果・`as_of` より後の情報を state / 資料に入れない。`Repository` の公開メソッドは必ず `as_of=` を渡す。
- 欠損は `None`（JSON では `null`）で残し、0 や「不適性」に置き換えない。未経験条件と欠損を区別する。
- 3着内確率は**全頭の合計を 1 に正規化しない**（spec §6）。
- 勝率用の重み（`data/weights/*`）と較正マップを3着内の系統に流用しない（spec §9）。
- **Jev に課金するコマンドは所有者の明示の承認を得てから実行する**（Task 14・Task 16）。`--dry-run` は課金しない。
- JRA-VAN のデータ・Jev への送信物は `data/` 配下にだけ置く（再配布不可）。
- `RaceConditions` にフィールドを足すときは、その列が発走前に確定することを確かめ、`tests/test_repository.py` の allow-list テストも更新する。
- state の大きさの上限: `STATE_MAX_CHARS = 48_000`（`json.dumps(state, ensure_ascii=False)` の文字数）。根拠は実測の 18頭 layer1 state = 31,868 文字 / 17,735 トークン（約 0.56 トークン/文字）で、48,000 文字 ≈ 27,000 トークン。32k の「state + 最長の質問」から質問ぶんの余裕を残す。Task 14 で実測の比から見直す。
- 版番号: `config.DOSSIER_VERSION = "1"`、`predict/top3_questions.TOP3_PROMPT_VERSION = "1"`、`config.TOP3_EVAL_RULES_VERSION = "1"`。state・質問・採点規則のいずれかを変えたら上げる。
- 表示候補数は `n`（既定 `config.DEFAULT_TOP_N = 3`）で持ち、3 をコードに直書きしない。**3頭固定は未確定**（spec §1）。ただし「3着以内」という事象の 3 は `config.TOP3_PLACE = 3` として別に持つ（表示頭数とは別物）。

## Review Focus

1. **同着・降着・失格・取消が絡むレース**（3着同着で実際の3着内が4頭、選んだ馬が取消）。評価から落とさず、spec §7 どおり固定した規則で採点し、件数を報告に出すこと → Task 8 のテスト。
2. **過去走を持つ馬が1頭もいないレース（新馬戦・取得期間外）**。資料は「新馬」「取得期間外の可能性」「不明」を区別して明示し、Python 基準は馬番順を「順位」と偽らず全頭同順位として報告すること → Task 4・Task 9 のテスト。
3. **フルゲート18頭で state が上限を超える**。黙って削らず、送る前に `StateTooLargeError` で止め、バックテストでは理由付きスキップとして数えること → Task 10・Task 13 のテスト。
4. **Jev が一部の馬の `top3_XX` を返さない応答**。0.0 で埋めずそのレースを失敗にすること（既存の第2層と同じ扱い）→ Task 11 のテスト。
5. **コースを特定できないレース**（`race_id` 無しの利用者入力、トラックコードと入力の芝・ダートが食い違う）。コース情報を推測で埋めず `identified: false` と理由を出し、コース単位の条件集計を「定義できない」にすること → Task 2・Task 5 のテスト。

---

## ファイル構成

| パス | 新規/変更 | 責務 |
|---|---|---|
| `docs/superpowers/specs/2026-09-23-top3-prediction-design.md` | 新規 | report3 の写し（設計の出典をリポジトリ内に置く） |
| `src/keiba/config.py` | 変更 | 版番号・`TOP3_PLACE`・`STATE_MAX_CHARS`・履歴判定の閾値 |
| `src/keiba/models.py` | 変更 | `RaceConditions` に `track_cd` / `course_kubun_cd` |
| `src/keiba/data/store.py` | 変更 | manifest の `coverage.first_hasso_at` を `DataStore.coverage_from` に |
| `src/keiba/data/repository.py` | 変更 | `race_conditions()` が2列増、`Repository.coverage_from` |
| `src/keiba/evaluate/coverage.py` | 新規 | データ版の充足度（先行出走を持つ馬の割合）を測る |
| `scripts/data_coverage.py` | 新規 | 上の CLI |
| `src/keiba/features/course.py` | 新規 | トラックコードの解読・コース識別・コース基本情報の参照 |
| `src/keiba/features/course_facts.json` | 新規 | JRA 公式ページから写したコース基本情報（出典・適用期間付き） |
| `src/keiba/features/history.py` | 新規 | 過去走の行・履歴の欠け方の区別・条件の組み合わせ集計 |
| `src/keiba/features/derive.py` | 変更 | `summarize_runs()` を公開（既存の `_build_condition_record` を包むだけ） |
| `src/keiba/features/dossier.py` | 新規 | 1レース分の資料（JSON-safe dict）と state 化・サイズ検査 |
| `src/keiba/features/dossier_markdown.py` | 新規 | 資料の人間向け Markdown |
| `scripts/dossier.py` | 新規 | 資料を JSON / Markdown で書き出す CLI（API を呼ばない） |
| `src/keiba/evaluate/top3_metrics.py` | 新規 | 3着内の正解・採点・集計・クラスタブートストラップ |
| `src/keiba/predict/top3_baselines.py` | 新規 | Python だけの基準（能力順位）と一様確率 |
| `src/keiba/predict/top3_questions.py` | 新規 | 3着内 Noul・同 state の winner・pace の質問文 |
| `src/keiba/predict/top3.py` | 新規 | 案A / 案B の Jev 呼び出しと応答の検査 |
| `src/keiba/store/top3_records.py` | 新規 | 3着内予想の記録（送った state・質問を含む）の保存・読み出し |
| `src/keiba/evaluate/top3_backtest.py` | 新規 | 過去レースでの比較実行と報告 |
| `scripts/backtest_top3.py` | 新規 | 上の CLI（`--dry-run` / 最終評価期間のガード） |
| `CLAUDE.md` | 変更 | 新しい系統の所在と使い方（Task 16） |

テストは各モジュールに対応する `tests/test_<module>.py` を新規に作る。

---

## Phase 0 — 前提: データの充足度

### Task 1: 設計の取り込みとデータ充足度の測定

**Files:**
- Create: `docs/superpowers/specs/2026-09-23-top3-prediction-design.md`（`tmp/keiba-report/report3-top3-prediction-design.md` の写し）
- Modify: `src/keiba/data/store.py`（`DataStore` に `coverage_from`）
- Modify: `src/keiba/data/repository.py`（`Repository.coverage_from`）
- Create: `src/keiba/evaluate/coverage.py`
- Create: `scripts/data_coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Produces: `DataStore.coverage_from: datetime | None`、`Repository.coverage_from -> datetime | None`、`coverage.measure(races: Sequence[RaceRow], repo: Repository, results: ResultRepository) -> CoverageReport`、`CoverageReport(n_races, n_horses, n_with_prior_run, n_with_3plus_prior_runs, share_with_prior_run, coverage_from)`

- [ ] **Step 1: 設計文書を写す**

```bash
cp tmp/keiba-report/report3-top3-prediction-design.md docs/superpowers/specs/2026-09-23-top3-prediction-design.md
```

写した文書の相対リンク（`../../src/...`、`report.md` など）は `tmp/` からの相対なので、`docs/superpowers/specs/` から辿れるよう `../../../src/...`・`../../../tmp/keiba-report/report.md` に直す。本文は変えない。

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_coverage.py`:

```python
"""データ版の充足度（予想対象レースの出走馬のうち、締切前の出走記録が版にある割合）。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.evaluate import coverage
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))


def _open(tmp_path, races, race_horses, *, first_hasso_at=None):
    data_dir = fx.write_dataset(tmp_path, races, race_horses, [])
    if first_hasso_at is not None:
        manifest_path = next((tmp_path / "versions").glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["coverage"] = {"first_hasso_at": first_hasso_at}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    store = DataStore.open_current(data_dir)
    return Repository(store), ResultRepository(store)


def test_coverage_from_reads_manifest(tmp_path):
    repo, _ = _open(tmp_path, [], [], first_hasso_at="2026-08-01T09:40:00+09:00")
    assert repo.coverage_from == datetime(2026, 8, 1, 9, 40, tzinfo=JST)


def test_coverage_from_is_none_when_manifest_has_no_coverage(tmp_path):
    repo, _ = _open(tmp_path, [], [])
    assert repo.coverage_from is None


def test_measure_counts_horses_with_prior_runs(tmp_path):
    races = [
        fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T12:00:00+09:00"),
        fx.make_race(race_id="2026032106011101", hasso_at="2026-03-21T15:25:00+09:00"),
    ]
    race_horses = [
        # 過去走あり（3/1 に走っている）
        fx.make_race_horse(race_id="2026030106010101", umaban="01", ketto_num="A"),
        fx.make_race_horse(race_id="2026032106011101", umaban="01", ketto_num="A"),
        # 過去走なし
        fx.make_race_horse(race_id="2026032106011101", umaban="02", ketto_num="B",
                           result_kakutei_jyuni="2"),
    ]
    repo, results = _open(tmp_path, races, race_horses)
    target = [results.race("2026032106011101")]

    report = coverage.measure(target, repo, results)

    assert report.n_races == 1
    assert report.n_horses == 2
    assert report.n_with_prior_run == 1
    assert report.n_with_3plus_prior_runs == 0
    assert report.share_with_prior_run == 0.5


def test_measure_on_empty_races_reports_nan_share(tmp_path):
    repo, results = _open(tmp_path, [], [])
    report = coverage.measure([], repo, results)
    assert report.n_horses == 0
    assert report.share_with_prior_run != report.share_with_prior_run  # NaN
```

- [ ] **Step 3: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coverage.py -v`
Expected: FAIL（`ImportError: cannot import name 'coverage'` / `AttributeError: 'Repository' object has no attribute 'coverage_from'`）

- [ ] **Step 4: 実装する**

`src/keiba/data/store.py` の `DataStore` に最後のフィールドとして足す（既定値付きなので既存の生成箇所は壊れない）:

```python
    #: manifest の coverage.first_hasso_at。データ版に入っている最も早い発走。
    #: 「この日より前の出走は版に無い」という境界で、過去走が見つからない理由を
    #: 「取得期間外」と「本当に無い」に分けるのに使う。古い manifest には無いので
    #: None になりうる（None は「境界が分からない」であって「境界が無い」ではない）。
    coverage_from: datetime | None = None
```

`open_version` の `cls(...)` に足す:

```python
            coverage_from=_coverage_from(manifest),
```

同じファイルのモジュール関数として:

```python
def _coverage_from(manifest: dict) -> datetime | None:
    value = (manifest.get("coverage") or {}).get("first_hasso_at")
    return datetime.fromisoformat(value) if value else None
```

`src/keiba/data/repository.py` の `Repository` に（`data_fetched_at` の直後）:

```python
    @property
    def coverage_from(self) -> datetime | None:
        """データ版に入っている最も早い発走（manifest 由来）。不明なら None。"""
        return self._store.coverage_from
```

`src/keiba/evaluate/coverage.py`:

```python
"""データ版の充足度。予想対象レースの出走馬のうち、予想締切より前の出走記録が
データ版にある馬の割合を測る。

CLAUDE.md の実測（延べ1,358頭のうち締切前の出走記録がある馬 29.0%）を、
データ版を取り直したあとに同じ定義で測り直すためのもの。Jev は呼ばない。
締切は evaluate/backtest.py と同じ定義（prediction_cutoff_for）を使う。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from keiba.data.repository import Repository, ResultRepository
from keiba.evaluate.backtest import prediction_cutoff_for
from keiba.models import RaceRow

__all__ = ["CoverageReport", "measure"]


@dataclass(frozen=True)
class CoverageReport:
    n_races: int
    n_horses: int
    n_with_prior_run: int
    n_with_3plus_prior_runs: int
    #: n_horses が 0 なら NaN（0.0 ではない。測っていない）。
    share_with_prior_run: float
    coverage_from: datetime | None


def measure(races: Sequence[RaceRow], repo: Repository,
            results: ResultRepository) -> CoverageReport:
    n_horses = n_prior = n_three = 0
    for race in races:
        as_of = prediction_cutoff_for(race)
        for result in results.results(race.race_id):
            n_horses += 1
            ketto_num = result.entry.ketto_num
            runs = repo.past_runs(ketto_num, as_of=as_of) if ketto_num else []
            if runs:
                n_prior += 1
            if len(runs) >= 3:
                n_three += 1
    share = n_prior / n_horses if n_horses else float("nan")
    return CoverageReport(
        n_races=len(races), n_horses=n_horses, n_with_prior_run=n_prior,
        n_with_3plus_prior_runs=n_three, share_with_prior_run=share,
        coverage_from=repo.coverage_from,
    )
```

`scripts/data_coverage.py`:

```python
"""データ版の充足度を表示する。Jev は呼ばない。

    .venv/Scripts/python.exe scripts/data_coverage.py --from 20260901 --to 20260922
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate import coverage  # noqa: E402

JST = timezone(timedelta(hours=9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD")
    parser.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args(argv)

    start = datetime.strptime(args.date_from, "%Y%m%d").replace(tzinfo=JST)
    end = datetime.strptime(args.date_to, "%Y%m%d").replace(
        hour=23, minute=59, second=59, tzinfo=JST)
    store = DataStore.open_current(args.data_dir)
    repo, results = Repository(store), ResultRepository(store)
    report = coverage.measure(results.races_between(start, end), repo, results)

    print(f"データ版: {store.version_id}  収録開始: {report.coverage_from}")
    print(f"対象レース: {report.n_races}  延べ出走馬: {report.n_horses}")
    print(f"締切前の出走記録あり: {report.n_with_prior_run}"
          f"（{report.share_with_prior_run:.1%}）")
    print(f"締切前の出走記録3走以上: {report.n_with_3plus_prior_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coverage.py -v`
Expected: 4 passed

- [ ] **Step 6: 現行データ版で測る（課金なし）**

Run: `.venv/Scripts/python.exe scripts/data_coverage.py --from 20260801 --to 20260922`
Expected: 「締切前の出走記録あり」がおよそ 29% 前後（CLAUDE.md の実測と同じ桁）。値を控える。

- [ ] **Step 7: 【所有者の操作】データ版を取り直す**

所有者に依頼する（JV-Link のダウンロード。Jev の課金は無いが時間がかかる）:

```bash
.venv/Scripts/python.exe scripts/update_data.py --dataspec RACE --fromdate 20230801000000
```

完了後、Step 6 と同じ期間で再測定し、両方の数字を `docs/superpowers/specs/2026-09-23-top3-prediction-design.md` の末尾に「データ充足度（実測）」節として追記する。**「締切前の出走記録あり」が 80% に届かない場合は、fromdate をさらに遡るかを所有者に相談してから Phase 3 に進む**（80% は判断の目安で、所有者が変えてよい）。

- [ ] **Step 8: 区切り**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全件 PASS。コミットが許可されていれば:

```bash
git add docs/superpowers/specs/2026-09-23-top3-prediction-design.md src/keiba/data/store.py src/keiba/data/repository.py src/keiba/evaluate/coverage.py scripts/data_coverage.py tests/test_coverage.py
git commit -m "feat: データ版の充足度を測る data_coverage を追加"
```

---

## Phase 1 — 予想用資料（API を呼ばない）

### Task 2: コース識別

**Files:**
- Modify: `src/keiba/models.py`（`RaceConditions`）
- Modify: `src/keiba/data/repository.py:392-422`（`race_conditions`）
- Modify: `tests/test_repository.py:305-319`（allow-list）
- Create: `src/keiba/features/course.py`
- Test: `tests/test_course.py`

**Interfaces:**
- Consumes: `Repository.race_conditions(race_id, *, as_of) -> RaceConditions | None`
- Produces:
  - `RaceConditions(class_level: int | None, track_cd: str | None = None, course_kubun_cd: str | None = None)`
  - `course.TrackCode(surface: str, direction: str | None, label: str)`、`course.decode_track_cd(track_cd: str) -> TrackCode | None`
  - `course.CourseIdentification(identified: bool, jyo_cd: str | None, track_cd: str | None, kyori_m: int, track: TrackCode | None, course_kubun_cd: str | None, reason: str | None)`
  - `course.identify(race_input: RaceInput, conditions: RaceConditions | None) -> CourseIdentification`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_course.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from keiba.features import course
from keiba.models import RaceConditions, RaceInput

JST = timezone(timedelta(hours=9))


def make_race_input(**kw) -> RaceInput:
    base = dict(
        date=date(2026, 8, 1), jyo="新潟", race_num=11, track_kind="turf",
        kyori_m=2000, entry_count=0,
        prediction_cutoff=datetime(2026, 7, 31, 20, 0, tzinfo=JST),
        horses=(), jyo_cd="04",
    )
    base.update(kw)
    return RaceInput(**base)


def test_decode_track_cd_right_outer_turf():
    code = course.decode_track_cd("18")
    assert code == course.TrackCode(surface="turf", direction="right", label="芝・右外")


def test_decode_track_cd_left_turf_has_no_layout_claim():
    # 11 は「芝・左」。内回りとは書かれていない（新潟内回りもこの値）ので、
    # 解読結果に「内回り」を作らない。内外の事実はコース基本情報表の仕事。
    code = course.decode_track_cd("11")
    assert code.label == "芝・左"
    assert code.direction == "left"


def test_decode_track_cd_straight():
    assert course.decode_track_cd("10").direction == "straight"
    assert course.decode_track_cd("29") == course.TrackCode("dirt", "straight", "ダート・直")


def test_decode_track_cd_unknown_or_jump_is_none():
    assert course.decode_track_cd("00") is None
    assert course.decode_track_cd("51") is None
    assert course.decode_track_cd("  ") is None


def test_identify_with_conditions():
    result = course.identify(
        make_race_input(),
        RaceConditions(class_level=5, track_cd="12", course_kubun_cd="A"))
    assert result.identified is True
    assert result.track_cd == "12"
    assert result.course_kubun_cd == "A"
    assert result.reason is None


def test_identify_without_race_id_is_not_identified():
    result = course.identify(make_race_input(), None)
    assert result.identified is False
    assert result.track is None
    assert "トラックコード" in result.reason


def test_identify_rejects_surface_mismatch():
    # 入力は芝なのにトラックコードはダート。どちらかが誤っているので推測しない。
    result = course.identify(
        make_race_input(track_kind="turf"),
        RaceConditions(class_level=5, track_cd="23", course_kubun_cd=""))
    assert result.identified is False
    assert "食い違" in result.reason


def test_identify_blank_course_kubun_is_none():
    result = course.identify(
        make_race_input(track_kind="dirt"),
        RaceConditions(class_level=5, track_cd="23", course_kubun_cd="  "))
    assert result.identified is True
    assert result.course_kubun_cd is None
```

`tests/test_repository.py` の allow-list テストを次の等式に書き換え、docstring の末尾に理由を1段落足す:

```python
        assert {f.name for f in dataclasses.fields(RaceConditions)} == {
            "class_level", "track_cd", "course_kubun_cd"}
```

足す段落: 「`track_cd`（使用するコース）と `course_kubun_cd`（仮柵の位置 A/B/C…）は番組と開催前の発表で決まり、発走前に確定する（report3 §4.2 の『使用コース』）。馬場状態（`baba_cd`）は当日確定なので入れない。」

`tests/test_repository.py` の `TestRaceConditions` に1件足す:

```python
    def test_race_conditions_returns_track_and_course_kubun(self, tmp_path):
        races = [dict(fx.make_race(race_id="R1", track_cd="18", data_kubun="7"),
                      course_kubun_cd="B")]
        repo = open_repository(tmp_path, races, [])
        conditions = repo.race_conditions("R1", as_of=jst(2026, 3, 20))
        assert conditions.track_cd == "18"
        assert conditions.course_kubun_cd == "B"
```

（`open_repository` / `jst` は同ファイルの既存ヘルパ。）

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_course.py tests/test_repository.py -v`
Expected: FAIL（`course` が無い、allow-list が不一致）

- [ ] **Step 3: 実装する**

`src/keiba/models.py` の `RaceConditions` にフィールドを足し、docstring の「現状は1列だけ」を「発走前に確定する3列」に直す:

```python
    class_level: int | None
    #: 2009 トラックコード。コース（芝・右外など）は番組で決まり発走前に確定する。
    track_cd: str | None = None
    #: コース区分（仮柵の位置 A/B/C…）。開催前に発表される。空白は None。
    course_kubun_cd: str | None = None
```

`src/keiba/data/repository.py` の `race_conditions` の return を:

```python
        row = picked.iloc[0]
        return RaceConditions(
            class_level=_int_or_none(row["class_level"]),
            track_cd=(row["track_cd"] or "").strip() or None,
            course_kubun_cd=(row.get("course_kubun_cd", "") or "").strip() or None,
        )
```

`src/keiba/features/course.py`:

```python
"""コースの識別。

「新潟・芝・2000m」には内回りと外回りがあり一意に決まらない（report3 §3.2）。
JV-Data のトラックコード（2009）と競馬場・距離の組でコースを特定する。

ここで決めるのは「どのコースか」だけ。直線の長さや坂などの事実は
course_facts.json（JRA 公式ページから写したもの）から引く（Task 3）。
「直線が長いので差し有利」のような予測はどこにも書かない（report3 §4.2）。

トラックコードの対応は vendor/jravan-sdk-500-64bit/ドキュメント/
JV-Data仕様書_4.9.0.1.xlsx「コード表」2009 から写した。略名もそのまま。
"""

from __future__ import annotations

from dataclasses import dataclass

from keiba import config
from keiba.models import RaceConditions, RaceInput

__all__ = ["CourseIdentification", "TrackCode", "decode_track_cd", "identify"]


@dataclass(frozen=True)
class TrackCode:
    surface: str            # "turf" / "dirt"（サンドは config.track_kind に従い dirt）
    direction: str | None   # "left" / "right" / "straight"
    label: str              # JV-Data の略名（例: "芝・右外"）


# 平地のみ。障害（51-59）は予想対象外なので None を返す。
_TRACK_CODES: dict[str, tuple[str, str]] = {
    "10": ("straight", "芝・直"),
    "11": ("left", "芝・左"),
    "12": ("left", "芝・左外"),
    "13": ("left", "芝・左内→外"),
    "14": ("left", "芝・左外→内"),
    "15": ("left", "芝・左内２周"),
    "16": ("left", "芝・左外２周"),
    "17": ("right", "芝・右"),
    "18": ("right", "芝・右外"),
    "19": ("right", "芝・右内→外"),
    "20": ("right", "芝・右外→内"),
    "21": ("right", "芝・右内２周"),
    "22": ("right", "芝・右外２周"),
    "23": ("left", "ダート・左"),
    "24": ("right", "ダート・右"),
    "25": ("left", "ダート・左内"),
    "26": ("right", "ダート・右外"),
    "27": ("left", "サンド・左"),
    "28": ("right", "サンド・右"),
    "29": ("straight", "ダート・直"),
}


def decode_track_cd(track_cd: str) -> TrackCode | None:
    code = (track_cd or "").strip()
    if code not in _TRACK_CODES:
        return None
    surface = config.track_kind(code)
    if surface not in ("turf", "dirt"):
        return None
    direction, label = _TRACK_CODES[code]
    return TrackCode(surface=surface, direction=direction, label=label)


@dataclass(frozen=True)
class CourseIdentification:
    identified: bool
    jyo_cd: str | None
    track_cd: str | None
    kyori_m: int
    track: TrackCode | None
    course_kubun_cd: str | None
    #: identified が False のときの理由。True なら None。
    reason: str | None


def identify(race_input: RaceInput, conditions: RaceConditions | None) -> CourseIdentification:
    def not_identified(reason: str) -> CourseIdentification:
        return CourseIdentification(
            identified=False, jyo_cd=race_input.jyo_cd, track_cd=None,
            kyori_m=race_input.kyori_m, track=None, course_kubun_cd=None, reason=reason)

    if conditions is None or not conditions.track_cd:
        return not_identified(
            "対象レースのトラックコードがデータ版に無いため、内外回りなどのコースを特定できない。")
    track = decode_track_cd(conditions.track_cd)
    if track is None:
        return not_identified(f"トラックコード {conditions.track_cd!r} は平地のコースとして解読できない。")
    if track.surface != race_input.track_kind:
        return not_identified(
            f"入力の芝・ダート（{race_input.track_kind}）とトラックコード"
            f"（{track.label}）が食い違うため、コースを特定しない。")
    return CourseIdentification(
        identified=True, jyo_cd=race_input.jyo_cd, track_cd=conditions.track_cd,
        kyori_m=race_input.kyori_m, track=track,
        course_kubun_cd=(conditions.course_kubun_cd or "").strip() or None, reason=None)
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_course.py tests/test_repository.py tests/test_context.py -v`
Expected: PASS（`test_context.py` は `RaceConditions(class_level=...)` を作る箇所があっても既定値で通る）

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: トラックコードからコースを識別する course.identify を追加"`（対象ファイルを `git add`）。

---

### Task 3: コース基本情報表

**Files:**
- Create: `src/keiba/features/course_facts.json`
- Modify: `src/keiba/features/course.py`（`load_facts` / `facts_for`）
- Test: `tests/test_course_facts.py`

**Interfaces:**
- Consumes: `CourseIdentification`（Task 2）
- Produces: `course.CourseFacts(facts: dict, start_note: str | None, source_url: str, retrieved_at: str, valid_from: str, valid_to: str | None)`、`course.facts_for(identification: CourseIdentification, on: date, *, table: list[dict] | None = None) -> CourseFacts | None`、`course.load_facts_table() -> list[dict]`

**事実の出どころについて（このタスクの要点）:** 直線の長さ・高低差・スタート位置などの数値は、**記憶や推測で書かない**。JRA 公式のコース紹介ページ（`https://www.jra.go.jp/facilities/race/<venue>/course/`。venue は sapporo / hakodate / fukushima / niigata / tokyo / nakayama / chukyo / kyoto / hanshin / kokura）を WebFetch で取得し、ページに書かれている値だけを写す。ページに無い項目はキーごと省く（`null` も入れない）。表のスキーマ:

```json
{
  "schema_version": "1",
  "courses": [
    {
      "jyo_cd": "04",
      "track_cd": "11",
      "valid_from": "YYYY-MM-DD",
      "valid_to": null,
      "facts": {
        "layout": "<ページの表記どおり。例: 内回り>",
        "straight_m": 0,
        "elevation_note": "<ページの記述をそのまま短く>"
      },
      "starts": {"2000": "<その距離のスタート位置の記述>"},
      "source": {"url": "https://www.jra.go.jp/facilities/race/niigata/course/", "retrieved_at": "2026-09-23"}
    }
  ]
}
```

`facts` に置いてよいキーは `layout` / `straight_m` / `circumference_m` / `width_m` / `elevation_note` / `corner_count` の6つだけ（テストで強制する）。`valid_from` はページに改修日の記載があればその日、無ければ `"2000-01-01"` とし、`source.retrieved_at` に取得日を書く。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_course_facts.py`:

```python
from __future__ import annotations

from datetime import date

import pytest

from keiba.features import course

ALLOWED_FACT_KEYS = {"layout", "straight_m", "circumference_m", "width_m",
                     "elevation_note", "corner_count"}


def _ident(jyo_cd="04", track_cd="11", kyori_m=2000, identified=True):
    return course.CourseIdentification(
        identified=identified, jyo_cd=jyo_cd, track_cd=track_cd if identified else None,
        kyori_m=kyori_m, track=course.decode_track_cd(track_cd) if identified else None,
        course_kubun_cd=None, reason=None if identified else "x")


TABLE = [
    {"jyo_cd": "04", "track_cd": "11", "valid_from": "2000-01-01", "valid_to": "2019-12-31",
     "facts": {"layout": "旧"}, "starts": {},
     "source": {"url": "https://www.jra.go.jp/a", "retrieved_at": "2026-09-23"}},
    {"jyo_cd": "04", "track_cd": "11", "valid_from": "2020-01-01", "valid_to": None,
     "facts": {"layout": "内回り", "straight_m": 359},
     "starts": {"2000": "スタンド前"},
     "source": {"url": "https://www.jra.go.jp/b", "retrieved_at": "2026-09-23"}},
]


def test_facts_for_picks_entry_valid_on_date():
    facts = course.facts_for(_ident(), date(2026, 8, 1), table=TABLE)
    assert facts.facts == {"layout": "内回り", "straight_m": 359}
    assert facts.start_note == "スタンド前"
    assert facts.source_url == "https://www.jra.go.jp/b"


def test_facts_for_respects_valid_to():
    facts = course.facts_for(_ident(), date(2019, 6, 1), table=TABLE)
    assert facts.facts == {"layout": "旧"}


def test_facts_for_unknown_distance_has_no_start_note():
    assert course.facts_for(_ident(kyori_m=1800), date(2026, 8, 1), table=TABLE).start_note is None


def test_facts_for_unidentified_course_is_none():
    assert course.facts_for(_ident(identified=False), date(2026, 8, 1), table=TABLE) is None


def test_facts_for_missing_entry_is_none():
    assert course.facts_for(_ident(jyo_cd="05"), date(2026, 8, 1), table=TABLE) is None


# --- 同梱の表そのものの検査 ------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    return course.load_facts_table()


def test_shipped_table_every_entry_has_official_source(shipped):
    assert shipped, "course_facts.json が空"
    for entry in shipped:
        assert entry["source"]["url"].startswith("https://www.jra.go.jp/"), entry
        date.fromisoformat(entry["source"]["retrieved_at"])
        date.fromisoformat(entry["valid_from"])


def test_shipped_table_uses_only_allowed_fact_keys(shipped):
    for entry in shipped:
        assert set(entry["facts"]) <= ALLOWED_FACT_KEYS, entry
        assert None not in entry["facts"].values(), "無い項目は null ではなく省く"


def test_shipped_table_has_no_overlapping_validity(shipped):
    by_key: dict[tuple, list] = {}
    for entry in shipped:
        by_key.setdefault((entry["jyo_cd"], entry["track_cd"]), []).append(entry)
    for key, entries in by_key.items():
        spans = sorted((date.fromisoformat(e["valid_from"]),
                        date.fromisoformat(e["valid_to"]) if e["valid_to"] else date.max)
                       for e in entries)
        for (a_from, a_to), (b_from, _) in zip(spans, spans[1:]):
            assert a_to < b_from, f"{key} の適用期間が重なっている"


def test_shipped_table_track_codes_are_decodable(shipped):
    for entry in shipped:
        assert course.decode_track_cd(entry["track_cd"]) is not None, entry
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_course_facts.py -v`
Expected: FAIL（`facts_for` / `load_facts_table` が無い）

- [ ] **Step 3: 参照コードを実装する**

`src/keiba/features/course.py` に足す（`__all__` にも `CourseFacts` / `facts_for` / `load_facts_table` を追加）:

```python
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

_FACTS_PATH = Path(__file__).with_name("course_facts.json")


@dataclass(frozen=True)
class CourseFacts:
    facts: dict
    start_note: str | None
    source_url: str
    retrieved_at: str
    valid_from: str
    valid_to: str | None


@lru_cache(maxsize=1)
def _load() -> tuple[dict, ...]:
    payload = json.loads(_FACTS_PATH.read_text(encoding="utf-8"))
    return tuple(payload["courses"])


def load_facts_table() -> list[dict]:
    return [dict(entry) for entry in _load()]


def facts_for(identification: CourseIdentification, on: date, *,
              table: list[dict] | None = None) -> CourseFacts | None:
    """識別済みコースの、``on`` の日に有効な基本情報。無ければ None（推測で埋めない）。"""
    if not identification.identified:
        return None
    entries = table if table is not None else _load()
    for entry in entries:
        if entry["jyo_cd"] != identification.jyo_cd or entry["track_cd"] != identification.track_cd:
            continue
        valid_from = date.fromisoformat(entry["valid_from"])
        valid_to = date.fromisoformat(entry["valid_to"]) if entry["valid_to"] else date.max
        if not valid_from <= on <= valid_to:
            continue
        return CourseFacts(
            facts=dict(entry["facts"]),
            start_note=entry.get("starts", {}).get(str(identification.kyori_m)),
            source_url=entry["source"]["url"],
            retrieved_at=entry["source"]["retrieved_at"],
            valid_from=entry["valid_from"],
            valid_to=entry["valid_to"],
        )
    return None
```

- [ ] **Step 4: 表を写す**

1. 実データに現れる `(jyo_cd, track_cd)` の組を列挙する（写す対象の一覧）:

```bash
.venv/Scripts/python.exe -c "
import pandas as pd, pathlib
v = pathlib.Path('data/CURRENT').read_text().strip()
r = pd.read_csv(f'data/versions/{v}/races.csv', dtype=str, keep_default_na=False)
r = r[r.track_kind.isin(['turf','dirt'])]
print(r.groupby(['jyo_cd','track_cd']).kyori_m.unique().to_string())
"
```

2. 各競馬場の JRA 公式コース紹介ページを WebFetch で取得し、上のスキーマで1組ずつ書く。**ページに書かれていない値は書かない。** 取得できなかったページは一覧に「未取得」として残し、表には入れない（`facts_for` が None を返し、資料に「コース基本情報なし」と出る）。
3. `.venv/Scripts/python.exe -m pytest tests/test_course_facts.py -v` で表の検査を通す。

- [ ] **Step 5: 【所有者の確認】**

写した表と、取得した各ページの URL を所有者に見せ、数値を公式ページと突き合わせてもらう。修正があれば反映してから次へ進む。

- [ ] **Step 6: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: JRA 公式ページから写したコース基本情報表を追加"`。

---

### Task 4: 過去走の行と、履歴の欠け方の区別

**Files:**
- Modify: `src/keiba/config.py`
- Create: `src/keiba/features/history.py`
- Modify: `tests/fixtures/runs.py`（`make_past_run` に `track_cd` / `race_id` / `jyuni_1c` 引数）
- Test: `tests/test_history.py`

**Interfaces:**
- Consumes: `PastRun`、`Repository.coverage_from`
- Produces:
  - `config.HISTORY_TRUNCATION_MARGIN_DAYS = 90`、`config.DOSSIER_VERSION = "1"`、`config.TOP3_PLACE = 3`
  - `history.run_line(run: PastRun, *, as_of: datetime) -> dict`（キーは下記 `RUN_LINE_KEYS`）
  - `history.recent_run_lines(runs: Sequence[PastRun], *, as_of: datetime, limit: int) -> list[dict]`
  - `history.HistoryStatus(status: str, note: str, runs_found: int, period_from: date | None, period_to: date | None)`
  - `history.history_status(runs: Sequence[PastRun], *, matched: bool, race_class_level: int | None, coverage_from: datetime | None) -> HistoryStatus`
  - status の値: `"found"` / `"found_may_be_truncated"` / `"debut"` / `"not_in_window"` / `"unknown"` / `"unmatched"`

- [ ] **Step 1: フィクスチャを広げる**

`tests/fixtures/runs.py` の `make_past_run` に引数を足す（既定値は現行と同じ振る舞い）:

```python
    track_cd: str | None = None,
    race_id: str = "2026010106010101",
    jyuni_1c: int | None = None,
```

本体で `track_cd=track_cd if track_cd is not None else _TRACK_CD.get(track_kind, "00")`、`race_id=race_id`、`jyuni_1c=jyuni_1c` を `RaceRow` / `ResultRow` に渡す（既存の固定値を置き換える）。

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_history.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from keiba.features import history
from tests.fixtures.runs import make_past_run

JST = timezone(timedelta(hours=9))
AS_OF = datetime(2026, 9, 20, 18, 0, tzinfo=JST)
COVERAGE_FROM = datetime(2026, 8, 1, 9, 40, tzinfo=JST)


def run_at(month, day, **kw):
    return make_past_run(hasso_at=datetime(2026, month, day, 15, 0, tzinfo=JST), **kw)


def test_run_line_has_fixed_keys_and_no_future_fields():
    line = history.run_line(run_at(9, 6, kakutei_jyuni=2, margin_sec=0.1,
                                   jyuni_4c=3, track_cd="18"), as_of=AS_OF)
    assert tuple(line) == history.RUN_LINE_KEYS
    assert line["date"] == "2026-09-06"
    assert line["days_before"] == 14
    assert line["track_cd"] == "18"
    assert line["track_label"] == "芝・右外"
    assert line["kakutei_jyuni"] == 2
    assert line["corner_positions"] == [None, None, None, 3]


def test_recent_run_lines_newest_first_and_limited():
    runs = [run_at(8, 2), run_at(9, 6), run_at(8, 23)]
    lines = history.recent_run_lines(runs, as_of=AS_OF, limit=2)
    assert [line["date"] for line in lines] == ["2026-09-06", "2026-08-23"]


def test_recent_run_lines_excludes_scratched_runs():
    runs = [run_at(9, 6, ijyo_cd="1"), run_at(8, 23)]
    assert [l["date"] for l in history.recent_run_lines(runs, as_of=AS_OF, limit=5)] == ["2026-08-23"]


def test_status_found_far_from_coverage_start():
    runs = [run_at(9, 6)]
    status = history.history_status(runs, matched=True, race_class_level=2,
                                    coverage_from=datetime(2023, 8, 1, tzinfo=JST))
    assert status.status == "found"
    assert status.runs_found == 1
    assert status.period_from == date(2026, 9, 6)


def test_status_may_be_truncated_when_first_run_near_coverage_start():
    runs = [run_at(8, 10), run_at(9, 6)]
    status = history.history_status(runs, matched=True, race_class_level=2,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "found_may_be_truncated"


def test_status_debut_for_newcomer_race():
    status = history.history_status([], matched=True, race_class_level=0,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "debut"


def test_status_not_in_window_for_winners_class():
    # 1勝クラス以上（class_level >= 2）は勝利経験が必要なので過去走は実在する。
    status = history.history_status([], matched=True, race_class_level=2,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "not_in_window"


def test_status_unknown_for_maiden_race_without_runs():
    # 未勝利（1）は未出走馬も出られるので、取得期間外か未出走かを区別できない。
    status = history.history_status([], matched=True, race_class_level=1,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "unknown"


def test_status_unknown_when_class_level_is_none():
    status = history.history_status([], matched=True, race_class_level=None,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "unknown"


def test_status_unmatched_when_name_not_matched():
    status = history.history_status([], matched=False, race_class_level=0,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "unmatched"


def test_status_found_without_coverage_info_does_not_claim_truncation():
    status = history.history_status([run_at(8, 10)], matched=True, race_class_level=2,
                                    coverage_from=None)
    assert status.status == "found"
```

- [ ] **Step 3: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py -v`
Expected: FAIL（`keiba.features.history` が無い）

- [ ] **Step 4: 実装する**

`src/keiba/config.py` の「既定値」節に:

```python
# 資料（features/dossier.py）の形の版。キーや中身の定義が変わったら上げる。
DOSSIER_VERSION = "1"
# 「3着以内」の 3。表示頭数（DEFAULT_TOP_N）とは別物なので分けて持つ。
TOP3_PLACE = 3
# 見つかった最古の過去走がデータ版の収録開始からこの日数以内なら、それより前の
# 出走が版の外にある可能性がある。中央の出走間隔はおおむね3〜8週なので、
# 90日以内なら「さらに前にも走っていた」ことを否定できない。
HISTORY_TRUNCATION_MARGIN_DAYS = 90
```

`src/keiba/features/history.py`:

```python
"""過去走の行と、過去走が無い・少ない理由の区別（report3 §4.3・§4.4）。

集計値だけでなく、根拠となる過去走そのものを数件添える。過去走が無いことは
「能力が低い」ではない。新馬で本当に無いのか、データ版の取得期間の外にあるのか、
区別できないのかを分けて書く。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from keiba import config
from keiba.features.course import decode_track_cd
from keiba.models import PastRun

__all__ = [
    "RUN_LINE_KEYS", "HistoryStatus", "history_status", "recent_run_lines", "run_line",
]

RUN_LINE_KEYS: tuple[str, ...] = (
    "date", "days_before", "jyo", "track_cd", "track_label", "track_kind", "kyori_m",
    "baba", "class_level", "syusso_tosu", "kakutei_jyuni", "margin_sec",
    "corner_positions", "last_3f_sec", "futan_kg",
)


def run_line(run: PastRun, *, as_of: datetime) -> dict:
    """過去走1件の行。締切より前の走だけが来る前提（Repository.past_runs の as_of）。"""
    track = decode_track_cd(run.race.track_cd)
    result = run.result
    return {
        "date": run.race.hasso_at.date().isoformat(),
        "days_before": (as_of - run.race.hasso_at).days,
        "jyo": run.race.jyo,
        "track_cd": run.race.track_cd,
        "track_label": track.label if track else None,
        "track_kind": run.race.track_kind,
        "kyori_m": run.race.kyori_m,
        "baba": run.race.baba,
        "class_level": run.race.class_level,
        "syusso_tosu": run.race.syusso_tosu,
        # 競走中止・失格は着順の数字が残っていても着順として読まない。
        "kakutei_jyuni": result.kakutei_jyuni if result.can_win else None,
        "margin_sec": result.margin_sec if run.has_time else None,
        "corner_positions": [result.jyuni_1c, result.jyuni_2c, result.jyuni_3c, result.jyuni_4c],
        "last_3f_sec": result.last_3f_sec,
        "futan_kg": result.entry.futan_kg,
    }


def recent_run_lines(runs: Sequence[PastRun], *, as_of: datetime, limit: int) -> list[dict]:
    """平地・出走集合に入る過去走の新しい順 ``limit`` 件。取消・除外は走っていないので除く。"""
    usable = [run for run in runs if run.race.is_flat and run.result.counts_as_starter]
    usable.sort(key=lambda run: run.race.hasso_at, reverse=True)
    return [run_line(run, as_of=as_of) for run in usable[:limit]]


@dataclass(frozen=True)
class HistoryStatus:
    status: str
    note: str
    runs_found: int
    period_from: date | None
    period_to: date | None


_NOTES = {
    "found": "締切前の出走記録がデータ版にある。",
    "found_may_be_truncated": (
        "締切前の出走記録はあるが、最古の記録がデータ版の収録開始に近く、"
        "それより前の出走が版の外にある可能性がある。"),
    "debut": "新馬戦のため、過去走は存在しない。",
    "not_in_window": (
        "1勝クラス以上のため過去走は実在するはずだが、データ版の取得期間内に"
        "見つからない。能力が低いことを意味しない。"),
    "unknown": (
        "データ版の取得期間内に過去走が見つからない。未出走か取得期間外かは区別できない。"
        "能力が低いことを意味しない。"),
    "unmatched": "馬名をデータ版の馬と照合できなかったため、過去走を引いていない。",
}


def history_status(runs: Sequence[PastRun], *, matched: bool, race_class_level: int | None,
                   coverage_from: datetime | None) -> HistoryStatus:
    def make(status: str, found: Sequence[PastRun] = ()) -> HistoryStatus:
        dates = [run.race.hasso_at.date() for run in found]
        return HistoryStatus(status=status, note=_NOTES[status], runs_found=len(found),
                             period_from=min(dates) if dates else None,
                             period_to=max(dates) if dates else None)

    if not matched:
        return make("unmatched")
    if runs:
        if coverage_from is not None:
            earliest = min(run.race.hasso_at for run in runs)
            if (earliest - coverage_from).days <= config.HISTORY_TRUNCATION_MARGIN_DAYS:
                return make("found_may_be_truncated", runs)
        return make("found", runs)
    if race_class_level == 0:
        return make("debut")
    if race_class_level is not None and race_class_level >= 2:
        return make("not_in_window")
    return make("unknown")
```

- [ ] **Step 5: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py tests/test_ability.py tests/test_derive.py -v`
Expected: PASS（フィクスチャ変更で既存テストが壊れていないこと）

- [ ] **Step 6: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 過去走の行と履歴の欠け方の区別を追加"`。

---

### Task 5: 条件の組み合わせ集計

**Files:**
- Modify: `src/keiba/features/derive.py`（`summarize_runs` を公開）
- Modify: `src/keiba/features/history.py`
- Test: `tests/test_history.py`（追記）

**Interfaces:**
- Consumes: `derive.ConditionRecord`、`derive.BABA_UNKNOWN_REASON`
- Produces:
  - `derive.summarize_runs(matched: list[PastRun]) -> ConditionRecord`
  - `history.TodayConditions(track_kind: str, kyori_m: int, jyo_cd: str | None, track_cd: str | None, baba_cd: str | None, class_level: int | None)`
  - `history.COMBINED_CONDITION_KEYS`（下記5つ、この順）
  - `history.combined_condition_records(runs: Sequence[PastRun], *, today: TodayConditions) -> dict[str, tuple[str, ConditionRecord]]`（値は `(定義文, 集計)`）

5区分と定義（`definition` の文言はこのまま state に出る）:

| キー | 一致の条件 | 定義できない場合 |
|---|---|---|
| `surface_exact_distance` | 芝ダ一致 × 距離一致 | — |
| `surface_near_distance` | 芝ダ一致 × 距離差 ±`SAME_DISTANCE_TOLERANCE_M` | — |
| `course_exact_distance` | 競馬場一致 × トラックコード一致 × 距離一致 | 今回のトラックコード不明 |
| `surface_same_going` | 芝ダ一致 × 馬場状態一致 | 今回の馬場状態が未入力（`BABA_UNKNOWN_REASON`） |
| `surface_near_distance_near_class` | 芝ダ一致 × 距離差 ±許容 × クラス差 ±1 | 今回のクラス不明 |

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_history.py` に追記）

```python
from keiba.features.derive import BABA_UNKNOWN_REASON


def today(**kw):
    base = dict(track_kind="turf", kyori_m=2000, jyo_cd="04", track_cd="11",
                baba_cd="1", class_level=3)
    base.update(kw)
    return history.TodayConditions(**base)


def test_combined_keys_are_fixed_and_ordered():
    records = history.combined_condition_records([], today=today())
    assert tuple(records) == history.COMBINED_CONDITION_KEYS


def test_surface_exact_distance_does_not_mix_dirt():
    runs = [run_at(9, 6, track_kind="turf", kyori_m=2000),
            run_at(8, 23, track_kind="dirt", kyori_m=2000)]
    definition, record = history.combined_condition_records(runs, today=today())["surface_exact_distance"]
    assert record.starts == 1
    assert "芝・ダート" in definition


def test_course_exact_distance_separates_inner_and_outer():
    runs = [run_at(9, 6, jyo_cd="04", track_cd="11", kyori_m=2000),
            run_at(8, 23, jyo_cd="04", track_cd="12", kyori_m=2000)]
    _, record = history.combined_condition_records(runs, today=today())["course_exact_distance"]
    assert record.starts == 1


def test_course_exact_distance_inapplicable_without_track_cd():
    _, record = history.combined_condition_records(
        [run_at(9, 6)], today=today(track_cd=None))["course_exact_distance"]
    assert record.starts is None
    assert record.inapplicable_reason is not None


def test_same_going_inapplicable_without_baba():
    _, record = history.combined_condition_records(
        [run_at(9, 6)], today=today(baba_cd=None))["surface_same_going"]
    assert record.starts is None
    assert record.inapplicable_reason == BABA_UNKNOWN_REASON


def test_near_class_window():
    runs = [run_at(9, 6, class_level=2), run_at(8, 30, class_level=4),
            run_at(8, 23, class_level=5)]
    _, record = history.combined_condition_records(
        runs, today=today(class_level=3))["surface_near_distance_near_class"]
    assert record.starts == 2


def test_near_class_inapplicable_without_class():
    _, record = history.combined_condition_records(
        [run_at(9, 6)], today=today(class_level=None))["surface_near_distance_near_class"]
    assert record.starts is None


def test_zero_matches_is_zero_not_none():
    _, record = history.combined_condition_records([], today=today())["surface_exact_distance"]
    assert record.starts == 0
    assert record.margin_sec_mean is None
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py -v`
Expected: FAIL（`TodayConditions` が無い）

- [ ] **Step 3: 実装する**

`src/keiba/features/derive.py`（`_build_condition_record` の直後。`__all__` に `"summarize_runs"` を追加）:

```python
def summarize_runs(matched: list[PastRun]) -> ConditionRecord:
    """条件で絞り終えた過去走を1つの ConditionRecord に集計する。

    features/history.py の組み合わせ集計が同じ集計規則（top3 は can_win の
    1〜3着、着差は時計のある走だけ）を使うための公開口。規則はここ1箇所に置く。
    """
    return _build_condition_record(matched)
```

`src/keiba/features/history.py` に追記（`__all__` に `TodayConditions` / `COMBINED_CONDITION_KEYS` / `combined_condition_records` を追加）:

```python
from collections.abc import Callable

from keiba.features.derive import BABA_UNKNOWN_REASON, ConditionRecord, summarize_runs


@dataclass(frozen=True)
class TodayConditions:
    """今回のレースの条件のうち、組み合わせ集計に要るもの。不明は None。"""

    track_kind: str
    kyori_m: int
    jyo_cd: str | None
    track_cd: str | None
    baba_cd: str | None
    class_level: int | None


_TOL = config.SAME_DISTANCE_TOLERANCE_M

# (キー, 定義文, 定義できない理由を返す関数, 一致判定)
_COMBINED: tuple[tuple[str, str,
                       Callable[[TodayConditions], str | None],
                       Callable[[PastRun, TodayConditions], bool]], ...] = (
    ("surface_exact_distance",
     "芝・ダートが今回と同じで、距離が今回と同じ過去走",
     lambda t: None,
     lambda r, t: r.race.track_kind == t.track_kind and r.race.kyori_m == t.kyori_m),
    ("surface_near_distance",
     f"芝・ダートが今回と同じで、距離が今回の±{_TOL}m以内の過去走",
     lambda t: None,
     lambda r, t: r.race.track_kind == t.track_kind and abs(r.race.kyori_m - t.kyori_m) <= _TOL),
    ("course_exact_distance",
     "競馬場・コース（トラックコード）・距離がすべて今回と同じ過去走",
     lambda t: None if t.track_cd else "今回のコース（トラックコード）が特定できないため比較できない",
     lambda r, t: (r.race.jyo_cd == t.jyo_cd and r.race.track_cd == t.track_cd
                   and r.race.kyori_m == t.kyori_m)),
    ("surface_same_going",
     "芝・ダートが今回と同じで、馬場状態が今回と同じ過去走",
     lambda t: None if t.baba_cd else BABA_UNKNOWN_REASON,
     lambda r, t: r.race.track_kind == t.track_kind and r.race.baba_cd == t.baba_cd),
    ("surface_near_distance_near_class",
     f"芝・ダートが今回と同じで、距離が±{_TOL}m以内、クラスが今回の±1段階以内の過去走",
     lambda t: None if t.class_level is not None else "今回のクラスが不明のため比較できない",
     lambda r, t: (r.race.track_kind == t.track_kind
                   and abs(r.race.kyori_m - t.kyori_m) <= _TOL
                   and r.race.class_level is not None
                   and abs(r.race.class_level - t.class_level) <= 1)),
)

COMBINED_CONDITION_KEYS: tuple[str, ...] = tuple(key for key, *_ in _COMBINED)


def combined_condition_records(runs: Sequence[PastRun], *,
                               today: TodayConditions) -> dict[str, tuple[str, ConditionRecord]]:
    """5区分すべてを常に返す。区分は排他ではない（同じ走が複数に入りうる）。"""
    eligible = [run for run in runs if run.race.is_flat and run.result.counts_as_starter]
    records: dict[str, tuple[str, ConditionRecord]] = {}
    for key, definition, why_undefined, matches in _COMBINED:
        reason = why_undefined(today)
        if reason is not None:
            records[key] = (definition, ConditionRecord(
                starts=None, wins=None, top3=None, margin_sec_mean=None,
                period_from=None, period_to=None, inapplicable_reason=reason))
            continue
        records[key] = (definition, summarize_runs([r for r in eligible if matches(r, today)]))
    return records
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py tests/test_derive.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 芝ダ・距離・コース・馬場・クラスを組み合わせた条件別集計を追加"`。

---

### Task 6: 1レース分の資料（dossier）と state 化

**Files:**
- Modify: `src/keiba/config.py`（`STATE_MAX_CHARS`、`DEFAULT_DOSSIER_RECENT_RUNS`）
- Create: `src/keiba/features/dossier.py`
- Test: `tests/test_dossier.py`

**Interfaces:**
- Consumes: `build_context`（`features/context.py`）、`summary._ability_dict` / `_running_style_dict` / `_race_dict` / `data_quality`、`course.identify` / `facts_for`、`history.*`、`Repository.race_conditions` / `past_runs` / `coverage_from`
- Produces:
  - `dossier.build_dossier(race_input: RaceInput, repo: Repository, *, max_recent_runs: int = config.DEFAULT_DOSSIER_RECENT_RUNS) -> dict`
  - `dossier.to_state(dossier: dict, *, pace_forecast: dict | None = None) -> dict`
  - `dossier.state_chars(state: dict) -> int`
  - `dossier.StateTooLargeError(chars: int, limit: int)`、`dossier.check_state_size(state: dict, *, limit: int = config.STATE_MAX_CHARS) -> int`
  - 資料のトップレベルキー（この順）: `dossier_version`, `prediction_cutoff`, `course`, `race`, `horses`, `field_summary`, `data_quality`
  - `horses[umaban]` のキー（この順）: `bamei`, `sex_age`, `futan_kg`, `waku`, `kisyu`, `history`, `recent_runs`, `vs_last_run`, `conditions`, `ability`, `running_style`

`build_context` の `summary.py` 内の private 関数を使うので、`summary.py` 側で `_ability_dict` / `_running_style_dict` / `_race_dict` を `ability_dict` / `running_style_dict` / `race_dict` という公開名でも参照できるよう、ファイル末尾に別名を置く（中身は変えない）:

```python
# features/dossier.py が同じ書式で出すための公開名。書式はここ1箇所で持つ。
ability_dict = _ability_dict
running_style_dict = _running_style_dict
race_dict = _race_dict
```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dossier.py`（実データの形を通すため CSV フィクスチャ + 本物の `Repository` を使う）:

```python
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from keiba import config
from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.features import dossier
from keiba.models import HorseEntry, RaceInput
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))
TARGET = "2026032106011101"


def _dataset(tmp_path):
    races = [
        fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T15:00:00+09:00",
                     track_cd="18", kyori_m="2000"),
        # 対象レース。結果の値に目印（time 999.9 / margin 9.87）を入れ、漏れを検出する。
        fx.make_race(race_id=TARGET, hasso_at="2026-03-21T15:25:00+09:00",
                     track_cd="18", kyori_m="2000", class_level="3"),
    ]
    # 馬名照合（find_horses_by_name）は make_date が締切以前の行しか見ない。
    # past_runs は発走時刻だけで絞る。A は過去走の行（make_date 3/2）で、
    # B は馬マスタ（UM, make_date 1/1）で照合させる——B は対象レースにしか
    # 出走記録が無いので、UM が無いと「照合できない（unmatched）」になる。
    race_horses = [
        fx.make_race_horse(race_id="2026030106010101", umaban="03", ketto_num="A",
                           bamei="カコアリ", result_kakutei_jyuni="2", result_margin_sec="0.2",
                           make_date="20260302"),
        fx.make_race_horse(race_id=TARGET, umaban="01", ketto_num="A", bamei="カコアリ",
                           result_time_sec="999.9", result_margin_sec="9.87"),
        fx.make_race_horse(race_id=TARGET, umaban="02", ketto_num="B", bamei="カコナシ",
                           result_kakutei_jyuni="2", result_time_sec="999.9"),
    ]
    horses = [fx.make_horse(ketto_num="B", bamei="カコナシ", bamei_kana="ｶｺﾅｼ",
                            make_date="20260101")]
    data_dir = fx.write_dataset(tmp_path, races, race_horses, horses)
    return Repository(DataStore.open_current(data_dir))


def _race_input(**kw):
    base = dict(
        date=date(2026, 3, 21), jyo="中山", race_num=11, track_kind="turf", kyori_m=2000,
        entry_count=2, prediction_cutoff=datetime(2026, 3, 20, 21, 25, tzinfo=JST),
        horses=(HorseEntry(umaban="01", bamei="カコアリ"),
                HorseEntry(umaban="02", bamei="カコナシ")),
        jyo_cd="06", race_id=TARGET)
    base.update(kw)
    return RaceInput(**base)


def test_top_level_keys(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert tuple(d) == ("dossier_version", "prediction_cutoff", "course", "race",
                        "horses", "field_summary", "data_quality")
    assert d["dossier_version"] == config.DOSSIER_VERSION


def test_every_horse_has_same_keys(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    key_sets = {tuple(h) for h in d["horses"].values()}
    assert key_sets == {("bamei", "sex_age", "futan_kg", "waku", "kisyu", "history",
                         "recent_runs", "vs_last_run", "conditions", "ability",
                         "running_style")}


def test_course_is_identified_from_race_conditions(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert d["course"]["identified"] is True
    assert d["course"]["track_label"] == "芝・右外"


def test_course_not_identified_without_race_id(tmp_path):
    d = dossier.build_dossier(_race_input(race_id=None), _dataset(tmp_path))
    assert d["course"]["identified"] is False
    assert d["course"]["facts"] is None
    assert d["horses"]["01"]["conditions"]["course_exact_distance"]["starts"] is None


def test_history_distinguishes_horses(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert d["horses"]["01"]["history"]["runs_found"] == 1
    assert d["horses"]["01"]["recent_runs"][0]["kakutei_jyuni"] == 2
    assert d["horses"]["02"]["history"]["status"] == "not_in_window"  # class_level 3
    assert d["horses"]["02"]["recent_runs"] == []


def test_target_race_result_never_leaks(tmp_path):
    text = json.dumps(dossier.build_dossier(_race_input(), _dataset(tmp_path)),
                      ensure_ascii=False)
    assert "999.9" not in text
    assert "9.87" not in text


def test_dossier_is_json_safe(tmp_path):
    json.dumps(dossier.build_dossier(_race_input(), _dataset(tmp_path)), ensure_ascii=False)


def test_field_summary_counts(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    fs = d["field_summary"]
    assert fs["n_horses"] == 2
    assert fs["n_with_history"] == 1
    assert fs["history_status_counts"] == {"found": 1, "not_in_window": 1}


def test_to_state_adds_pace_forecast_only_when_given(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert "pace_forecast" not in dossier.to_state(d)
    state = dossier.to_state(d, pace_forecast={"probabilities": {"fast": 0.2}})
    assert state["pace_forecast"] == {"probabilities": {"fast": 0.2}}
    assert "pace_forecast" not in d  # 元の資料を書き換えない


def test_check_state_size_raises_over_limit():
    with pytest.raises(dossier.StateTooLargeError) as info:
        dossier.check_state_size({"x": "あ" * 100}, limit=50)
    assert info.value.chars > 50


def test_check_state_size_returns_chars_within_limit():
    assert dossier.check_state_size({"x": "a"}, limit=1000) == dossier.state_chars({"x": "a"})
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dossier.py -v`
Expected: FAIL（`keiba.features.dossier` が無い）

- [ ] **Step 3: 実装する**

`src/keiba/config.py`（`DOSSIER_VERSION` の近く）:

```python
# 資料に載せる直近の過去走の件数。多いほど state が大きくなる（STATE_MAX_CHARS）。
DEFAULT_DOSSIER_RECENT_RUNS = 5
# state の上限（json.dumps(state, ensure_ascii=False) の文字数）。
# 実測: 18頭の第1層 state が 31,868 文字 / 17,735 トークン（約0.56トークン/文字）。
# 48,000 文字 ≈ 27,000 トークンで、Jev の「state + 最長の質問 ≤ 32k」に質問ぶんの
# 余裕を残す。案Aの実行（Task 14）で得た実測の比で見直すこと。
STATE_MAX_CHARS = 48_000
# field_summary で「先行型」と数える4角の相対位置（corner_pos_mean）の上限。
FRONT_RUNNER_MAX_CORNER_POS = 1 / 3
```

`src/keiba/features/dossier.py`:

```python
"""1レース分の予想用資料（report3 §4）。API は呼ばない。

資料は JSON-safe な dict で、そのまま Jev の state にもなる（to_state）。
人間が確認する Markdown は dossier_markdown.py が同じ dict から作る——
「人が見た物」と「Jev に送った物」を別々に組み立てると黙ってずれる。

事実（コースの基本情報、過去走）と予測（展開）を混ぜない。展開の予測は案Bで
pace_forecast として to_state の段階で足すだけで、資料そのものには入れない。
"""

from __future__ import annotations

import copy
import json
from collections import Counter

from keiba import config
from keiba.data.repository import Repository
from keiba.features import course as course_mod
from keiba.features import history
from keiba.features.context import build_context
from keiba.features.summary import ability_dict, data_quality, race_dict, running_style_dict
from keiba.models import RaceInput

__all__ = ["StateTooLargeError", "build_dossier", "check_state_size", "state_chars", "to_state"]


class StateTooLargeError(ValueError):
    def __init__(self, chars: int, limit: int) -> None:
        super().__init__(
            f"state が {chars:,} 文字で上限 {limit:,} 文字を超えています。"
            "--max-recent-runs を減らすか、上限を実測で見直してください。")
        self.chars = chars
        self.limit = limit


def _iso(value):
    return value.isoformat() if value is not None else None


def _course_dict(identification, facts) -> dict:
    return {
        "identified": identification.identified,
        "reason": identification.reason,
        "jyo_cd": identification.jyo_cd,
        "track_cd": identification.track_cd,
        "track_label": identification.track.label if identification.track else None,
        "direction": identification.track.direction if identification.track else None,
        "kyori_m": identification.kyori_m,
        "course_kubun_cd": identification.course_kubun_cd,
        "facts": facts.facts if facts else None,
        "start_note": facts.start_note if facts else None,
        "facts_source": ({"url": facts.source_url, "retrieved_at": facts.retrieved_at,
                          "valid_from": facts.valid_from, "valid_to": facts.valid_to}
                         if facts else None),
    }


def _conditions_dict(records) -> dict:
    return {
        key: {
            "definition": definition,
            "starts": record.starts,
            "wins": record.wins,
            "top3": record.top3,
            "margin_sec_mean": record.margin_sec_mean,
            "period": {"from": _iso(record.period_from), "to": _iso(record.period_to)},
            "inapplicable_reason": record.inapplicable_reason,
        }
        for key, (definition, record) in records.items()
    }


def build_dossier(race_input: RaceInput, repo: Repository, *,
                  max_recent_runs: int = config.DEFAULT_DOSSIER_RECENT_RUNS) -> dict:
    as_of = race_input.prediction_cutoff
    ctx = build_context(race_input, repo)
    conditions = (repo.race_conditions(race_input.race_id, as_of=as_of)
                  if race_input.race_id is not None else None)
    class_level = conditions.class_level if conditions is not None else None
    identification = course_mod.identify(race_input, conditions)
    facts = course_mod.facts_for(identification, race_input.date)
    today = history.TodayConditions(
        track_kind=race_input.track_kind, kyori_m=race_input.kyori_m,
        jyo_cd=race_input.jyo_cd, track_cd=identification.track_cd,
        baba_cd=race_input.baba_cd, class_level=class_level)

    horses: dict[str, dict] = {}
    statuses: Counter[str] = Counter()
    n_front = 0
    for hf in ctx.horses:
        runs = repo.past_runs(hf.ketto_num, as_of=as_of) if hf.ketto_num else []
        status = history.history_status(runs, matched=hf.ketto_num is not None,
                                        race_class_level=class_level,
                                        coverage_from=repo.coverage_from)
        statuses[status.status] += 1
        mean_pos = hf.running_style.corner_pos_mean
        if mean_pos is not None and mean_pos <= config.FRONT_RUNNER_MAX_CORNER_POS:
            n_front += 1
        horses[hf.umaban] = {
            "bamei": hf.entry.bamei,
            "sex_age": hf.entry.sex_age,
            "futan_kg": hf.entry.futan_kg,
            "waku": hf.entry.wakuban,
            "kisyu": hf.entry.kisyu,
            "history": {
                "status": status.status, "note": status.note,
                "runs_found": status.runs_found,
                "period": {"from": _iso(status.period_from), "to": _iso(status.period_to)},
            },
            "recent_runs": history.recent_run_lines(runs, as_of=as_of, limit=max_recent_runs),
            "vs_last_run": {
                "days_since_last_run": hf.recent.days_since_last_run,
                "distance_change_m": hf.recent.distance_change_m,
                "futan_change_kg": hf.recent.futan_change_kg,
                "class_level_change": hf.recent.class_level_change,
            },
            "conditions": _conditions_dict(history.combined_condition_records(runs, today=today)),
            "ability": ability_dict(hf.ability, hf.field_rank),
            "running_style": running_style_dict(hf.running_style),
        }

    race = dict(race_dict(race_input))
    race["class_level"] = class_level
    quality = dict(data_quality(ctx))
    quality["coverage_from"] = _iso(repo.coverage_from)
    return {
        "dossier_version": config.DOSSIER_VERSION,
        "prediction_cutoff": as_of.isoformat(),
        "course": _course_dict(identification, facts),
        "race": race,
        "horses": horses,
        "field_summary": {
            "n_horses": len(horses),
            "n_with_history": sum(1 for h in horses.values() if h["history"]["runs_found"] > 0),
            "history_status_counts": dict(sorted(statuses.items())),
            "n_front_runners": n_front,
            "front_runner_definition": (
                "近走の4角の相対位置（通過順÷頭数）の平均が "
                f"{config.FRONT_RUNNER_MAX_CORNER_POS:.2f} 以下の馬"),
        },
        "data_quality": quality,
    }


def to_state(dossier: dict, *, pace_forecast: dict | None = None) -> dict:
    """Jev に送る state。資料をコピーし、案Bなら pace_forecast を足す。"""
    state = copy.deepcopy(dossier)
    if pace_forecast is not None:
        state["pace_forecast"] = pace_forecast
    return state


def state_chars(state: dict) -> int:
    return len(json.dumps(state, ensure_ascii=False))


def check_state_size(state: dict, *, limit: int = config.STATE_MAX_CHARS) -> int:
    """上限以内なら文字数を返す。超えたら StateTooLargeError。黙って削らない。"""
    chars = state_chars(state)
    if chars > limit:
        raise StateTooLargeError(chars, limit)
    return chars
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dossier.py tests/test_summary.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 1レース分の予想用資料 build_dossier と state 化を追加"`。

---

### Task 7: 資料の Markdown と書き出し CLI

**Files:**
- Create: `src/keiba/features/dossier_markdown.py`
- Create: `scripts/dossier.py`
- Test: `tests/test_dossier_markdown.py`

**Interfaces:**
- Consumes: `build_dossier`（Task 6）、`evaluate.backtest.race_input_from_history`、`input.schema.load_race_input`
- Produces: `dossier_markdown.render(dossier: dict) -> str`、CLI `scripts/dossier.py (--race-id ID | --input FILE) [--out DIR] [--max-recent-runs N]` → `<out>/<race_id or input stem>.json` と `.md`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dossier_markdown.py`:

```python
from __future__ import annotations

from keiba.features import dossier_markdown


def _dossier():
    run = {"date": "2026-09-06", "days_before": 14, "jyo": "中山", "track_cd": "18",
           "track_label": "芝・右外", "track_kind": "turf", "kyori_m": 2000, "baba": "良",
           "class_level": 3, "syusso_tosu": 12, "kakutei_jyuni": 2, "margin_sec": 0.1,
           "corner_positions": [None, None, 3, 3], "last_3f_sec": 34.1, "futan_kg": 57.0}
    cond = {"definition": "定義", "starts": 1, "wins": 0, "top3": 1, "margin_sec_mean": 0.1,
            "period": {"from": "2026-09-06", "to": "2026-09-06"}, "inapplicable_reason": None}
    undefined = dict(cond, starts=None, wins=None, top3=None,
                     inapplicable_reason="今回の馬場状態が未入力のため比較できない")
    horse = {"bamei": "テストウマ", "sex_age": "牡4", "futan_kg": 57.0, "waku": "1",
             "kisyu": None,
             "history": {"status": "found", "note": "締切前の出走記録がデータ版にある。",
                         "runs_found": 1, "period": {"from": "2026-09-06", "to": "2026-09-06"}},
             "recent_runs": [run],
             "vs_last_run": {"days_since_last_run": 14, "distance_change_m": 0,
                             "futan_change_kg": 0.0, "class_level_change": 0},
             "conditions": {"surface_exact_distance": cond, "surface_same_going": undefined},
             "ability": {}, "running_style": {"corner_pos_mean": 0.25}}
    return {
        "dossier_version": "1", "prediction_cutoff": "2026-09-19T21:25:00+09:00",
        "course": {"identified": False, "reason": "トラックコードが無い", "facts": None,
                   "track_label": None, "kyori_m": 2000, "start_note": None,
                   "facts_source": None, "course_kubun_cd": None},
        "race": {"jyo": "中山", "track_kind": "turf", "kyori_m": 2000, "baba": None,
                 "class_name": None, "class_level": 3, "entry_count": 1},
        "horses": {"01": horse},
        "field_summary": {"n_horses": 1, "n_with_history": 1,
                          "history_status_counts": {"found": 1}, "n_front_runners": 1,
                          "front_runner_definition": "定義"},
        "data_quality": {"data_version": "v1", "coverage_from": None, "notes": [],
                         "horse_notes": {}},
    }


def test_render_has_sections_in_order():
    text = dossier_markdown.render(_dossier())
    order = ["## コース", "## 今回の条件", "## 出走馬", "## データの充足状況"]
    positions = [text.index(h) for h in order]
    assert positions == sorted(positions)


def test_render_says_course_unidentified_with_reason():
    assert "トラックコードが無い" in dossier_markdown.render(_dossier())


def test_render_shows_missing_as_dash_not_zero():
    text = dossier_markdown.render(_dossier())
    assert "今回の馬場状態が未入力のため比較できない" in text
    assert "| 馬場 | — |" in text or "馬場: —" in text


def test_render_lists_recent_run():
    text = dossier_markdown.render(_dossier())
    assert "2026-09-06" in text and "芝・右外" in text
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dossier_markdown.py -v`
Expected: FAIL（モジュールが無い）

- [ ] **Step 3: 実装する**

`src/keiba/features/dossier_markdown.py`:

```python
"""資料（features/dossier.py の dict）を人間が確認するための Markdown にする。

値を作らない・丸めない・補わない。欠損は「—」で出し、0 と見分けられるようにする。
"""

from __future__ import annotations

__all__ = ["render"]

_DASH = "—"


def _v(value) -> str:
    if value is None:
        return _DASH
    if isinstance(value, list):
        return "-".join(_v(x) for x in value)
    return str(value)


def _course(c: dict) -> list[str]:
    lines = ["## コース", ""]
    if not c["identified"]:
        lines.append(f"コースを特定していない: {c['reason']}")
        return lines + [""]
    lines.append(f"- コース: {c['track_label']} {c['kyori_m']}m"
                 f"（仮柵: {_v(c['course_kubun_cd'])}）")
    if c["facts"] is None:
        lines.append("- 基本情報: 表に無い（推測で補っていない）")
    else:
        for key, value in c["facts"].items():
            lines.append(f"- {key}: {value}")
        lines.append(f"- スタート: {_v(c['start_note'])}")
        src = c["facts_source"]
        lines.append(f"- 出典: {src['url']}（取得 {src['retrieved_at']}）")
    return lines + [""]


def _race(r: dict, cutoff: str) -> list[str]:
    return ["## 今回の条件", "",
            "| 項目 | 値 |", "|---|---|",
            f"| 予想締切 | {cutoff} |",
            f"| 競馬場 | {_v(r.get('jyo'))} |",
            f"| 芝・ダート | {_v(r.get('track_kind'))} |",
            f"| 距離 | {_v(r.get('kyori_m'))} |",
            f"| 馬場 | {_v(r.get('baba'))} |",
            f"| クラス | {_v(r.get('class_name'))}（順序値 {_v(r.get('class_level'))}） |",
            f"| 頭数 | {_v(r.get('entry_count'))} |", ""]


_RUN_COLS = ("date", "jyo", "track_label", "kyori_m", "baba", "class_level",
             "syusso_tosu", "kakutei_jyuni", "margin_sec", "corner_positions", "last_3f_sec")


def _horse(umaban: str, h: dict) -> list[str]:
    lines = [f"### {umaban} {h['bamei']}（{_v(h['sex_age'])} / {_v(h['futan_kg'])}kg / 枠{_v(h['waku'])}）",
             "", f"履歴: {h['history']['status']} — {h['history']['note']}"
             f"（{h['history']['runs_found']}走）", ""]
    if h["recent_runs"]:
        lines += ["| " + " | ".join(_RUN_COLS) + " |",
                  "|" + "---|" * len(_RUN_COLS)]
        for run in h["recent_runs"]:
            lines.append("| " + " | ".join(_v(run[k]) for k in _RUN_COLS) + " |")
        lines.append("")
    lines += ["| 条件 | 出走 | 勝 | 3着内 | 平均着差 | 備考 |", "|---|---|---|---|---|---|"]
    for key, c in h["conditions"].items():
        lines.append(f"| {key} | {_v(c['starts'])} | {_v(c['wins'])} | {_v(c['top3'])} | "
                     f"{_v(c['margin_sec_mean'])} | {_v(c['inapplicable_reason'])} |")
    return lines + [""]


def _quality(dq: dict, fs: dict) -> list[str]:
    lines = ["## データの充足状況", "",
             f"- データ版: {dq.get('data_version')}（収録開始 {_v(dq.get('coverage_from'))}）",
             f"- 過去走がある馬: {fs['n_with_history']} / {fs['n_horses']}",
             f"- 履歴の内訳: {fs['history_status_counts']}"]
    for note in dq.get("notes", []):
        lines.append(f"- 注記: {note}")
    for umaban, notes in dq.get("horse_notes", {}).items():
        for note in notes:
            lines.append(f"- {umaban}番: {note}")
    return lines + [""]


def render(dossier: dict) -> str:
    lines = [f"# 予想用資料（資料の版 {dossier['dossier_version']}）", ""]
    lines += _course(dossier["course"])
    lines += _race(dossier["race"], dossier["prediction_cutoff"])
    lines += ["## 出走馬", ""]
    for umaban, h in dossier["horses"].items():
        lines += _horse(umaban, h)
    lines += _quality(dossier["data_quality"], dossier["field_summary"])
    return "\n".join(lines)
```

`scripts/dossier.py`:

```python
"""1レース分の予想用資料を JSON と Markdown で書き出す。Jev は呼ばない。

    # データ版にある過去レースを、前日夜に予想した体で
    .venv/Scripts/python.exe scripts/dossier.py --race-id 2026080104020311
    # 利用者入力（scripts/predict.py と同じ形式）
    .venv/Scripts/python.exe scripts/dossier.py --input race.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate.backtest import race_input_from_history  # noqa: E402
from keiba.features import dossier as dossier_mod  # noqa: E402
from keiba.features import dossier_markdown  # noqa: E402
from keiba.input.schema import load_race_input  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--race-id")
    source.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/dossiers"))
    parser.add_argument("--max-recent-runs", type=int, default=config.DEFAULT_DOSSIER_RECENT_RUNS)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args(argv)

    store = DataStore.open_current(args.data_dir)
    repo = Repository(store)
    if args.race_id:
        results = ResultRepository(store)
        race = results.race(args.race_id)
        if race is None:
            print(f"race_id {args.race_id} はデータ版にありません。")
            return 1
        # 結果行の entry 部分だけを使う（EntryRow は結果の項目を持てない）。
        entries = [r.entry for r in results.results(args.race_id)]
        race_input = race_input_from_history(race, entries, top_n=config.DEFAULT_TOP_N)
        stem = args.race_id
    else:
        race_input = load_race_input(args.input)
        stem = args.input.stem

    d = dossier_mod.build_dossier(race_input, repo, max_recent_runs=args.max_recent_runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(dossier_markdown.render(d), encoding="utf-8")
    chars = dossier_mod.state_chars(dossier_mod.to_state(d))
    print(f"書き出し: {args.out / stem}.json / .md（state {chars:,} 文字、"
          f"上限 {config.STATE_MAX_CHARS:,}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dossier_markdown.py -v`
Expected: PASS

- [ ] **Step 5: 実データで1レース書き出す（課金なし）**

Run: `.venv/Scripts/python.exe scripts/dossier.py --race-id 2026080104020311`
Expected: `data/dossiers/2026080104020311.json` / `.md` ができ、state の文字数が表示される（新潟 18頭。48,000 以下であること。超えていたら `--max-recent-runs 3` で再実行し、両方の値を控える）。

- [ ] **Step 6: 【所有者の確認】（report3 §9-1）**

`.md` を所有者に見せ、(1) コースの特定と基本情報、(2) 過去走の行、(3) 条件別集計の定義、(4) 履歴の欠け方の表示が意図どおりかを確認してもらう。指摘は Task 4〜7 に戻して直し、形が変わったら `DOSSIER_VERSION` を上げる。

- [ ] **Step 7: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 予想用資料の Markdown と書き出し CLI を追加"`。

---

## Phase 2 — 3着内の評価基盤（API を呼ばない）

### Task 8: 3着内の採点と集計

**Files:**
- Modify: `src/keiba/config.py`（`TOP3_EVAL_RULES_VERSION`）
- Create: `src/keiba/evaluate/top3_metrics.py`
- Test: `tests/test_top3_metrics.py`

**Interfaces:**
- Consumes: `ResultRow`（`counts_as_starter` / `can_win` / `kakutei_jyuni`）
- Produces:
  - `Top3Outcome(race_id: str, top3: tuple[str, ...], starters: tuple[str, ...], excluded: tuple[str, ...])`
  - `outcome_from_results(race_id: str, results: Sequence[ResultRow]) -> Top3Outcome | None`
  - `score_race(selected: Sequence[str], probabilities: Mapping[str, float] | None, outcome: Top3Outcome) -> dict`（キー: `hits`, `all_top3_selected`, `top3_exceeds_n`, `selected_scratched`, `brier`, `log_loss`, `prob_sum`, `prob_sum_minus_actual`）
  - `aggregate(scored: Sequence[dict], *, n: int) -> dict`
  - `paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], *, cluster_of: Callable[[str], str], n_resamples: int = 2000, seed: int = 0) -> dict`（キー: `mean_diff`, `lo`, `hi`, `n_races`, `n_clusters`）
  - `race_date_cluster(race_id: str) -> str`（race_id 先頭8桁 = 開催日）

採点規則（`TOP3_EVAL_RULES_VERSION = "1"`。変えたら上げる。spec §7「評価前に固定」）:

1. 実際の3着内 = 出走集合（取消・発走除外・競走除外を除く）のうち `can_win` かつ `kakutei_jyuni ∈ {1,2,3}`。**3着同着なら4頭以上になる**。降着は確定着順をそのまま使う。競走中止・失格は入らない。
2. 実際の3着内が0頭なら採点しない（`None`）。
3. 選んだ馬が取消・除外だった場合は外れとして数え、`selected_scratched` に件数を残す（予想時点では取消を知り得ないので、評価から落とさない）。
4. 「3着内をすべて選べた」= 実際の3着内の集合 ⊆ 選んだ馬。実際の3着内が n 頭を超えるレースは必ず False になり、`top3_exceeds_n` で件数を別に出す。
5. Brier・log_loss は出走集合の馬についてだけ、各馬の二値事象（3着内なら 1）で計算し、レース内で平均する。確率が無い方法（Python 基準の順位）は `None`。
6. `prob_sum` は出走集合の確率の合計、`prob_sum_minus_actual` はそこから実際の3着内の頭数を引いた値（合計の偏りの診断。補正はしない）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_top3_metrics.py`:

```python
from __future__ import annotations

import math

import pytest

from keiba.evaluate import top3_metrics as tm
from keiba.models import EntryRow, ResultRow


def rr(umaban, jyuni, ijyo_cd="0"):
    entry = EntryRow(umaban=umaban, wakuban="1", ketto_num=umaban, bamei=umaban,
                     sex_cd="1", barei="04", futan_kg=57.0, kisyu_code="1",
                     tozai_cd="1", chokyosi_code="1")
    return ResultRow(entry=entry, ijyo_cd=ijyo_cd, kakutei_jyuni=jyuni, dochaku_kubun="0",
                     dochaku_tosu=None, time_sec=None, margin_sec=None, jyuni_1c=None,
                     jyuni_2c=None, jyuni_3c=None, jyuni_4c=None, last_3f_sec=None,
                     ba_taijyu_kg=None, zogen_sa_kg=None, odds=None, ninki=None)


def test_outcome_basic():
    o = tm.outcome_from_results("R", [rr("01", 1), rr("02", 2), rr("03", 3), rr("04", 4)])
    assert o.top3 == ("01", "02", "03")
    assert o.starters == ("01", "02", "03", "04")


def test_outcome_dead_heat_for_third_has_four():
    o = tm.outcome_from_results("R", [rr("01", 1), rr("02", 2), rr("03", 3), rr("04", 3), rr("05", 5)])
    assert o.top3 == ("01", "02", "03", "04")


def test_outcome_excludes_disqualified_and_scratched():
    o = tm.outcome_from_results("R", [rr("01", 1), rr("02", 2, ijyo_cd="5"), rr("03", 3),
                                      rr("04", None, ijyo_cd="1"), rr("05", 4)])
    assert o.top3 == ("01", "03")
    assert "02" in o.starters
    assert o.excluded == ("04",)


def test_outcome_none_without_placed_horses():
    assert tm.outcome_from_results("R", [rr("01", None, ijyo_cd="4"), rr("02", None, ijyo_cd="4")]) is None
    assert tm.outcome_from_results("R", []) is None


def _o(top3=("01", "02", "03"), starters=("01", "02", "03", "04", "05"), excluded=()):
    return tm.Top3Outcome("20260921R", top3, starters, excluded)


def test_score_hits_and_all_selected():
    s = tm.score_race(["01", "02", "04"], None, _o())
    assert s["hits"] == 2
    assert s["all_top3_selected"] is False
    assert s["brier"] is None


def test_score_scratched_selection_counts_as_miss():
    s = tm.score_race(["01", "02", "09"], None, _o(excluded=("09",)))
    assert s["hits"] == 2
    assert s["selected_scratched"] == 1


def test_score_top3_exceeds_n_cannot_be_all_selected():
    s = tm.score_race(["01", "02", "03"], None, _o(top3=("01", "02", "03", "04")))
    assert s["all_top3_selected"] is False
    assert s["top3_exceeds_n"] is True


def test_score_brier_over_starters_only():
    probs = {"01": 1.0, "02": 1.0, "03": 1.0, "04": 0.0, "05": 0.0, "09": 0.7}
    s = tm.score_race(["01", "02", "03"], probs, _o(excluded=("09",)))
    assert s["brier"] == 0.0
    assert s["prob_sum"] == 3.0
    assert s["prob_sum_minus_actual"] == 0.0


def test_score_requires_probability_for_every_starter():
    with pytest.raises(ValueError):
        tm.score_race(["01"], {"01": 0.5}, _o())


def test_aggregate_distribution_and_rates():
    scored = [tm.score_race(["01", "02", "03"], None, _o()),
              tm.score_race(["04", "05", "01"], None, _o())]
    agg = tm.aggregate(scored, n=3)
    assert agg["n_races"] == 2
    assert agg["mean_hits"] == 2.0
    assert agg["hit_distribution"] == {"0": 0, "1": 1, "2": 0, "3": 1}
    assert agg["all_top3_selected_rate"] == 0.5
    assert math.isnan(agg["brier"])


def test_aggregate_empty_is_nan_not_zero():
    agg = tm.aggregate([], n=3)
    assert agg["n_races"] == 0
    assert math.isnan(agg["mean_hits"])


def test_race_date_cluster():
    assert tm.race_date_cluster("2026092106040511") == "20260921"


def test_paired_bootstrap_zero_difference():
    a = {f"2026092{d}0604051{r}": 2.0 for d in range(1, 4) for r in range(1, 4)}
    result = tm.paired_bootstrap(a, dict(a), cluster_of=tm.race_date_cluster, n_resamples=200)
    assert result["mean_diff"] == 0.0
    assert result["lo"] == 0.0 and result["hi"] == 0.0
    assert result["n_clusters"] == 3


def test_paired_bootstrap_uses_only_common_races():
    a = {"2026092106040511": 3.0, "2026092206040511": 1.0}
    b = {"2026092106040511": 1.0}
    result = tm.paired_bootstrap(a, b, cluster_of=tm.race_date_cluster, n_resamples=50)
    assert result["n_races"] == 1
    assert result["mean_diff"] == 2.0
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_metrics.py -v`
Expected: FAIL（モジュールが無い）

- [ ] **Step 3: 実装する**

`src/keiba/config.py`（`FEATURE_VERSION` の近く）:

```python
# 3着内の採点規則（evaluate/top3_metrics.py の docstring）の版。同着・取消・
# 失格の扱いを変えたら上げる。版が違う報告どうしは比べない。
TOP3_EVAL_RULES_VERSION = "1"
```

`src/keiba/evaluate/top3_metrics.py`:

```python
"""3着以内の採点（report3 §7）。

採点規則はこのプランの Task 8 に列挙したもので、config.TOP3_EVAL_RULES_VERSION
で版を持つ。集計はレース単位（馬の行を独立標本として扱わない）。差の不確かさは
開催日クラスタで再標本化したブートストラップで出す。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from keiba import config
from keiba.models import ResultRow

__all__ = ["Top3Outcome", "aggregate", "outcome_from_results", "paired_bootstrap",
           "race_date_cluster", "score_race"]

_EPS = 1e-12


@dataclass(frozen=True)
class Top3Outcome:
    race_id: str
    top3: tuple[str, ...]
    starters: tuple[str, ...]
    excluded: tuple[str, ...]


def outcome_from_results(race_id: str, results: Sequence[ResultRow]) -> Top3Outcome | None:
    starters = tuple(r.entry.umaban for r in results if r.counts_as_starter)
    excluded = tuple(r.entry.umaban for r in results if not r.counts_as_starter)
    top3 = tuple(r.entry.umaban for r in results
                 if r.can_win and r.kakutei_jyuni is not None
                 and 1 <= r.kakutei_jyuni <= config.TOP3_PLACE)
    if not top3:
        return None
    return Top3Outcome(race_id=race_id, top3=top3, starters=starters, excluded=excluded)


def score_race(selected: Sequence[str], probabilities: Mapping[str, float] | None,
               outcome: Top3Outcome) -> dict:
    actual = set(outcome.top3)
    chosen = list(selected)
    result = {
        "hits": sum(1 for u in chosen if u in actual),
        "all_top3_selected": actual <= set(chosen),
        "top3_exceeds_n": len(actual) > len(chosen),
        "selected_scratched": sum(1 for u in chosen if u in outcome.excluded),
        "brier": None, "log_loss": None, "prob_sum": None, "prob_sum_minus_actual": None,
    }
    if probabilities is None:
        return result
    missing = [u for u in outcome.starters if u not in probabilities]
    if missing:
        raise ValueError(f"{outcome.race_id}: 出走馬の3着内確率がありません: {missing}")
    ps = [(probabilities[u], 1.0 if u in actual else 0.0) for u in outcome.starters]
    result["brier"] = sum((p - y) ** 2 for p, y in ps) / len(ps)
    result["log_loss"] = -sum(
        y * math.log(max(p, _EPS)) + (1 - y) * math.log(max(1 - p, _EPS)) for p, y in ps
    ) / len(ps)
    result["prob_sum"] = sum(p for p, _ in ps)
    result["prob_sum_minus_actual"] = result["prob_sum"] - len(actual)
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def aggregate(scored: Sequence[dict], *, n: int) -> dict:
    with_prob = [s for s in scored if s["brier"] is not None]
    distribution = {str(k): 0 for k in range(n + 1)}
    for s in scored:
        distribution[str(min(s["hits"], n))] += 1
    return {
        "n": n,
        "rules_version": config.TOP3_EVAL_RULES_VERSION,
        "n_races": len(scored),
        "mean_hits": _mean([s["hits"] for s in scored]),
        "hit_distribution": distribution,
        "all_top3_selected_rate": _mean([1.0 if s["all_top3_selected"] else 0.0 for s in scored]),
        "n_top3_exceeds_n": sum(1 for s in scored if s["top3_exceeds_n"]),
        "n_selected_scratched": sum(s["selected_scratched"] for s in scored),
        "n_with_probabilities": len(with_prob),
        "brier": _mean([s["brier"] for s in with_prob]),
        "log_loss": _mean([s["log_loss"] for s in with_prob]),
        "prob_sum_minus_actual_mean": _mean([s["prob_sum_minus_actual"] for s in with_prob]),
    }


def race_date_cluster(race_id: str) -> str:
    """race_id の先頭8桁（年4+月日4）= 開催日。同じ日のレースは馬場・天候を共有する。"""
    return race_id[:8]


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], *,
                     cluster_of: Callable[[str], str], n_resamples: int = 2000,
                     seed: int = 0) -> dict:
    """レースごとの値 a - b の平均と、クラスタ再標本化の95%区間。両方にあるレースだけ使う。"""
    common = sorted(set(a) & set(b))
    diffs = {race_id: a[race_id] - b[race_id] for race_id in common}
    clusters: dict[str, list[float]] = {}
    for race_id, d in diffs.items():
        clusters.setdefault(cluster_of(race_id), []).append(d)
    keys = sorted(clusters)
    mean_diff = _mean(list(diffs.values()))
    if not keys:
        return {"mean_diff": mean_diff, "lo": float("nan"), "hi": float("nan"),
                "n_races": 0, "n_clusters": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [d for key in (rng.choice(keys) for _ in keys) for d in clusters[key]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"mean_diff": mean_diff,
            "lo": means[int(0.025 * (n_resamples - 1))],
            "hi": means[int(0.975 * (n_resamples - 1))],
            "n_races": len(common), "n_clusters": len(keys)}
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_metrics.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 3着内の採点規則と集計・クラスタブートストラップを追加"`。

---

### Task 9: Python だけの基準

**Files:**
- Create: `src/keiba/predict/top3_baselines.py`
- Test: `tests/test_top3_baselines.py`

**Interfaces:**
- Consumes: 資料 dict（Task 6）の `horses[*].ability.recent_margin_sec_wavg` / `ability.field_rank`
- Produces:
  - `top3_baselines.BaselineRanking(ranking: tuple[str, ...], all_tied: bool, n_unranked: int)`
  - `top3_baselines.ability_ranking(dossier: dict) -> BaselineRanking` — `field_rank` の小さい順。`field_rank` が無い馬は後ろにまとめ馬番順。全頭無ければ `all_tied=True`
  - `top3_baselines.uniform_probabilities(dossier: dict) -> dict[str, float]` — 全頭に `min(TOP3_PLACE, 頭数) / 頭数`（確率の比較用の参照）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_top3_baselines.py`:

```python
from __future__ import annotations

from keiba.predict import top3_baselines as tb


def d(ranks: dict):
    return {"horses": {u: {"ability": {"field_rank": r}} for u, r in ranks.items()}}


def test_ranking_by_field_rank():
    r = tb.ability_ranking(d({"01": 3.0, "02": 1.0, "03": 2.0}))
    assert r.ranking == ("02", "03", "01")
    assert r.all_tied is False
    assert r.n_unranked == 0


def test_horses_without_rank_go_last_in_umaban_order():
    r = tb.ability_ranking(d({"03": None, "01": 2.0, "02": None}))
    assert r.ranking == ("01", "02", "03")
    assert r.n_unranked == 2


def test_all_missing_is_reported_as_tied():
    r = tb.ability_ranking(d({"01": None, "02": None}))
    assert r.all_tied is True


def test_uniform_probabilities_sum_to_three():
    p = tb.uniform_probabilities(d({f"{i:02d}": None for i in range(1, 11)}))
    assert abs(sum(p.values()) - 3.0) < 1e-9


def test_uniform_probabilities_small_field():
    p = tb.uniform_probabilities(d({"01": None, "02": None}))
    assert p == {"01": 1.0, "02": 1.0}
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_baselines.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`src/keiba/predict/top3_baselines.py`:

```python
"""Jev を使わない基準（report3 §7-3）。Jev による改善を測る物差し。

能力順位は資料の ability.field_rank（直近走の加重平均着差のレース内順位、
小さいほど良い）だけで並べる。欠けた馬は後ろにまとめるが、全頭欠けたときの
馬番順は「順位」ではないので all_tied で報告する（Review Focus 2）。
"""

from __future__ import annotations

from dataclasses import dataclass

from keiba import config

__all__ = ["BaselineRanking", "ability_ranking", "uniform_probabilities"]


@dataclass(frozen=True)
class BaselineRanking:
    ranking: tuple[str, ...]
    all_tied: bool
    n_unranked: int


def ability_ranking(dossier: dict) -> BaselineRanking:
    horses = dossier["horses"]
    ranked = sorted((h["ability"]["field_rank"], u) for u, h in horses.items()
                    if h["ability"]["field_rank"] is not None)
    unranked = sorted(u for u, h in horses.items() if h["ability"]["field_rank"] is None)
    return BaselineRanking(ranking=tuple(u for _, u in ranked) + tuple(unranked),
                           all_tied=not ranked, n_unranked=len(unranked))


def uniform_probabilities(dossier: dict) -> dict[str, float]:
    n = len(dossier["horses"])
    p = min(config.TOP3_PLACE, n) / n if n else 0.0
    return dict.fromkeys(dossier["horses"], p)
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_baselines.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 3着内の Python 基準（能力順位・一様確率）を追加"`。

---

## Phase 3 — 案A

### Task 10: 3着内の質問文

**Files:**
- Create: `src/keiba/predict/top3_questions.py`
- Test: `tests/test_top3_questions.py`

**Interfaces:**
- Consumes: 資料 dict（Task 6）、`questions.PACE_CHOICES` / `LAYER1_PACE_INSTRUCTIONS`
- Produces:
  - `TOP3_PROMPT_VERSION = "1"`
  - `question_name(umaban: str) -> str`（`"top3_" + umaban`）
  - `top3_questions(dossier: dict, *, with_pace_forecast: bool) -> dict[str, Noul | Choice]`（全頭の `top3_XX` と `winner`）
  - `pace_questions() -> dict[str, Choice]`（案Bの1回目。`{"pace": Choice}`）
  - 質問文は state に実在するキー（`course` / `race` / `horses` / `field_summary` / `data_quality` / 案Bのみ `pace_forecast`）だけを参照する

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_top3_questions.py`:

```python
from __future__ import annotations

import re

from typesafe_sdk import Choice, Noul

from keiba.predict import top3_questions as tq

DOSSIER = {"course": {}, "race": {}, "field_summary": {}, "data_quality": {},
           "horses": {"01": {"bamei": "イチ", "sex_age": "牡4", "futan_kg": 57.0},
                      "02": {"bamei": "ニ", "sex_age": "牝3", "futan_kg": 55.0}}}
STATE_KEYS = {"course", "race", "horses", "field_summary", "data_quality",
              "prediction_cutoff", "dossier_version"}


def test_one_noul_per_horse_plus_winner():
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=False)
    assert set(qs) == {"top3_01", "top3_02", "winner"}
    assert isinstance(qs["top3_01"], Noul)
    assert isinstance(qs["winner"], Choice)


def test_each_noul_names_its_target_horse():
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=False)
    assert "対象馬: 01" in qs["top3_01"].instructions
    assert "対象馬: 02" in qs["top3_02"].instructions
    assert "対象馬: 01" not in qs["top3_02"].instructions


def test_noul_criteria_defines_true_and_false():
    dumped = tq.top3_questions(DOSSIER, with_pace_forecast=False)["top3_01"].model_dump()
    assert "3着以内" in dumped["criteria"]["true"]
    assert "4着以下" in dumped["criteria"]["false"]


def test_pace_forecast_mentioned_only_when_present():
    without = tq.top3_questions(DOSSIER, with_pace_forecast=False)["top3_01"].instructions
    with_ = tq.top3_questions(DOSSIER, with_pace_forecast=True)["top3_01"].instructions
    assert "pace_forecast" not in without
    assert "pace_forecast" in with_


def test_instructions_reference_only_existing_state_keys():
    # 「state のキー」として名指しするのは ASCII の識別子。存在しないキーを名指ししない。
    for with_pace in (False, True):
        allowed = STATE_KEYS | ({"pace_forecast"} if with_pace else set())
        for q in tq.top3_questions(DOSSIER, with_pace_forecast=with_pace).values():
            named = set(re.findall(r"\b([a-z_]+)：", q.instructions))
            assert named <= allowed, named - allowed


def test_winner_criteria_are_umaban():
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=False)
    assert set(qs["winner"].model_dump()["criteria"]) == {"01", "02"}


def test_pace_questions_reuse_existing_bands():
    qs = tq.pace_questions()
    assert set(qs["pace"].model_dump()["criteria"]) == {"fast", "standard", "slow"}
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_questions.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`src/keiba/predict/top3_questions.py`:

```python
"""3着以内を聞く質問文（report3 §6 のたたき台を state のキー名に合わせたもの）。

Jev は同じリクエスト内の質問を独立・並列に評価する。質問を並べても前の答えは
後の質問に渡らない（report3 §3.4）。案Bの展開予測は Python が受け取って
state の pace_forecast に入れる。

文言を変えたら TOP3_PROMPT_VERSION を上げる。記録に必ず残る。
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, NoulCriteria

from keiba.predict.questions import LAYER1_PACE_INSTRUCTIONS, PACE_CHOICES

__all__ = ["TOP3_PROMPT_VERSION", "pace_questions", "question_name", "top3_questions"]

TOP3_PROMPT_VERSION = "1"

_MATERIALS = (
    "判断材料：\n"
    "course：今回のコースの基本情報（特定できない場合は identified が false）\n"
    "race：今回のレース条件\n"
    "horses：出走全頭の過去走（recent_runs）、条件別実績（conditions）、"
    "能力指標（ability）、脚質（running_style）、過去走の有無と理由（history）\n"
    "field_summary：出走全頭の概要\n"
    "data_quality：欠損とデータ版の情報\n"
)

_PACE_LINE = (
    "pace_forecast：前段で予測した展開（ペース区分の確率）。"
    "未確定の見立てとして考慮し、確定した事実として扱わないでください。\n"
)

_RULES = (
    "\n"
    "今回のコース条件、対象馬の過去走、出走全頭との能力比較を判断材料にしてください。\n"
    "条件別実績は出走数（starts）と比較条件（definition）を併せて参照してください。\n"
    "未経験の条件や欠損情報を、不適性の証拠として扱わないでください。"
    "history.status が debut / not_in_window / unknown / unmatched の馬は、"
    "過去走が無いことを能力が低い証拠として扱わないでください。\n"
    "馬名の知名度ではなく、入力された情報を根拠にしてください。"
)

_CRITERIA = NoulCriteria(
    true="対象馬が今回のレースで確定着順3着以内に入る。",
    false="対象馬が今回のレースで4着以下になる、または競走中止・失格などで3着以内に入らない。",
)


def question_name(umaban: str) -> str:
    return f"top3_{umaban}"


def _top3_instructions(umaban: str, *, with_pace_forecast: bool) -> str:
    return (
        f"対象馬: {umaban}\n"
        "予想締切までに得られた情報に基づき、対象馬（horses の上記馬番）が"
        "今回のレースで確定着順3着以内に入るかを予測してください。\n"
        "\n"
        + _MATERIALS
        + (_PACE_LINE if with_pace_forecast else "")
        + _RULES
    )


_WINNER_INSTRUCTIONS = (
    "予想締切時点の情報に基づき、今回のレースの1着馬を予測してください。\n"
    "\n"
    + _MATERIALS
    + _RULES
)


def _winner_criteria(dossier: dict) -> dict[str, str]:
    criteria = {}
    for umaban, h in dossier["horses"].items():
        parts = [h["bamei"]]
        if h.get("sex_age"):
            parts.append(h["sex_age"])
        if h.get("futan_kg") is not None:
            parts.append(f"{h['futan_kg']}kg")
        criteria[umaban] = " ".join(parts)
    return criteria


def top3_questions(dossier: dict, *, with_pace_forecast: bool) -> dict:
    """全頭の top3_XX（Noul）と、同じ state での winner（Choice）。

    winner は「同じ state で勝率上位を選んだら」の比較対象（report3 §7-2）。
    同じ state への質問なので同じリクエストにまとめる（増えるのは質問文のトークンだけ）。
    """
    questions: dict = {
        question_name(umaban): Noul(
            instructions=_top3_instructions(umaban, with_pace_forecast=with_pace_forecast),
            criteria=_CRITERIA)
        for umaban in dossier["horses"]
    }
    questions["winner"] = Choice(instructions=_WINNER_INSTRUCTIONS,
                                 criteria=_winner_criteria(dossier))
    return questions


def pace_questions() -> dict:
    """案Bの1回目。既存の pace 区分と文言を使う（state に pace_definition が要る）。"""
    return {"pace": Choice(instructions=LAYER1_PACE_INSTRUCTIONS, criteria=PACE_CHOICES)}
```

注意: `LAYER1_PACE_INSTRUCTIONS` は state の `horses[*].running_style` と `pace_definition` と `waku` を名指しする。資料の `horses[*]` は `running_style` と `waku` を持つ（Task 6）。`pace_definition` は Task 15 で案Bの1回目の state にだけ足す。

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_questions.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 3着内 Noul と同 state の winner 質問を追加"`。

---

### Task 11: 案Aの実行と応答の検査

**Files:**
- Create: `src/keiba/predict/top3.py`
- Test: `tests/test_top3.py`

**Interfaces:**
- Consumes: `jev_client.ask`、`dossier.to_state` / `check_state_size`、`top3_questions.top3_questions` / `question_name`
- Produces:
  - `Top3Answer(top3_probabilities: dict[str, float], winner_probabilities: dict[str, float], winner_missing: tuple[str, ...], pace_probabilities: dict[str, float] | None, model: str, input_tokens: int, requests: int, state_chars: int, sent: dict, seconds: float)`
    - `sent` = `{"requests": [{"state": ..., "questions": ...}, ...]}`（送った順。案Aは1件、案Bは2件）
  - `Top3IncompleteError(RuntimeError)`（`missing: list[str]`）
  - `run_plan_a(client, dossier: dict) -> Top3Answer`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_top3.py`:

```python
from __future__ import annotations

import types

import pytest
import typesafe_sdk as ts

from jev_client import JevRequestError
from keiba.features import dossier as dossier_mod
from keiba.predict import top3

DOSSIER = {"dossier_version": "1", "prediction_cutoff": "x", "course": {}, "race": {},
           "field_summary": {}, "data_quality": {},
           "horses": {"01": {"bamei": "イチ"}, "02": {"bamei": "ニ"}, "03": {"bamei": "サン"}}}


class Resp:
    def __init__(self, nouls=None, choices=None, tokens=500):
        self.nouls = nouls or {}
        self.choices = choices or {}
        self.model = "jev-1.13.0"
        self.usage = types.SimpleNamespace(input_tokens=tokens)


def noul(p):
    return types.SimpleNamespace(noul=p)


def winner(probs):
    return types.SimpleNamespace(choice=max(probs, key=probs.get), probabilities=probs,
                                 confidence=0.5)


class Client:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def system_one(self, state, questions, **kw):
        self.calls.append((state, questions))
        return self.handler(state, questions)


def full_handler(state, questions):
    return Resp(nouls={"top3_01": noul(0.7), "top3_02": noul(0.5), "top3_03": noul(0.2)},
                choices={"winner": winner({"01": 0.5, "02": 0.3, "03": 0.2})})


def test_plan_a_single_request_with_all_questions():
    client = Client(full_handler)
    answer = top3.run_plan_a(client, DOSSIER)
    assert len(client.calls) == 1
    assert set(client.calls[0][1]) == {"top3_01", "top3_02", "top3_03", "winner"}
    assert answer.top3_probabilities == {"01": 0.7, "02": 0.5, "03": 0.2}
    assert answer.requests == 1
    assert answer.input_tokens == 500


def test_plan_a_does_not_normalize():
    answer = top3.run_plan_a(Client(full_handler), DOSSIER)
    assert abs(sum(answer.top3_probabilities.values()) - 1.4) < 1e-9


def test_plan_a_keeps_what_it_sent():
    client = Client(full_handler)
    answer = top3.run_plan_a(client, DOSSIER)
    assert answer.sent["requests"][0]["state"] is client.calls[0][0]
    assert answer.sent["requests"][0]["questions"]["top3_01"]["type"] == "noul"


def test_missing_top3_answer_is_failure_not_zero():
    def handler(state, questions):
        return Resp(nouls={"top3_01": noul(0.7), "top3_02": noul(0.5)},
                    choices={"winner": winner({"01": 0.5, "02": 0.3, "03": 0.2})})
    with pytest.raises(top3.Top3IncompleteError) as info:
        top3.run_plan_a(Client(handler), DOSSIER)
    assert info.value.missing == ["top3_03"]


def test_missing_winner_entry_is_recorded_not_failure():
    def handler(state, questions):
        return Resp(nouls={f"top3_0{i}": noul(0.3) for i in (1, 2, 3)},
                    choices={"winner": winner({"01": 0.6, "02": 0.4})})
    answer = top3.run_plan_a(Client(handler), DOSSIER)
    assert answer.winner_probabilities["03"] == 0.0
    assert answer.winner_missing == ("03",)


def test_state_too_large_is_raised_before_sending(monkeypatch):
    client = Client(full_handler)
    monkeypatch.setattr(top3.config, "STATE_MAX_CHARS", 10)
    with pytest.raises(dossier_mod.StateTooLargeError):
        top3.run_plan_a(client, DOSSIER)
    assert client.calls == []


def test_api_error_is_wrapped():
    def handler(state, questions):
        raise ts.TypeSafeAPITimeoutError(10.0)
    with pytest.raises(JevRequestError):
        top3.run_plan_a(Client(handler), DOSSIER)
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`src/keiba/predict/top3.py`:

```python
"""案A / 案B の Jev 呼び出し（report3 §5）。

どちらも「全頭の 3着内確率（Noul）」と「同じ state での winner（Choice）」を
1リクエストで聞く。案Bはその前に展開（pace）を1回聞き、確率分布のまま
state の pace_forecast に入れる（最尤区分だけを渡すと確定した事実として
扱われるため）。

3着内確率は正規化しない。応答に欠けた top3_XX があれば 0.0 で埋めず
Top3IncompleteError にする（第2層と同じ扱い）。winner の欠けは 0.0 で埋めて
馬番を記録する（第1層と同じ扱い。比較対象として使うだけのため）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jev_client import ask
from keiba import config
from keiba.features.dossier import check_state_size, to_state
from keiba.predict import top3_questions as tq

__all__ = ["Top3Answer", "Top3IncompleteError", "run_plan_a", "run_plan_b"]


class Top3IncompleteError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"3着内の答えが欠けています: {missing}")
        self.missing = missing


@dataclass(frozen=True)
class Top3Answer:
    top3_probabilities: dict[str, float]
    winner_probabilities: dict[str, float]
    winner_missing: tuple[str, ...]
    pace_probabilities: dict[str, float] | None
    model: str
    input_tokens: int
    requests: int
    state_chars: int
    sent: dict
    seconds: float


def _dump(questions: dict) -> dict:
    return {name: q.model_dump() for name, q in questions.items()}


def _ask_top3(client: Any, dossier: dict, state: dict, *, with_pace_forecast: bool):
    chars = check_state_size(state, limit=config.STATE_MAX_CHARS)
    questions = tq.top3_questions(dossier, with_pace_forecast=with_pace_forecast)
    response = ask(client, state, questions)
    missing = [tq.question_name(u) for u in dossier["horses"]
               if tq.question_name(u) not in response.nouls]
    if missing:
        raise Top3IncompleteError(missing)
    top3 = {u: float(response.nouls[tq.question_name(u)].noul) for u in dossier["horses"]}
    winner_probs: dict[str, float] = {}
    winner_missing: list[str] = []
    raw = response.choices["winner"].probabilities if "winner" in response.choices else {}
    for u in dossier["horses"]:
        if u in raw:
            winner_probs[u] = float(raw[u])
        else:
            winner_probs[u] = 0.0
            winner_missing.append(u)
    return response, top3, winner_probs, tuple(winner_missing), chars, _dump(questions)


def run_plan_a(client: Any, dossier: dict) -> Top3Answer:
    started = time.perf_counter()
    state = to_state(dossier)
    response, top3, winner_probs, winner_missing, chars, sent_q = _ask_top3(
        client, dossier, state, with_pace_forecast=False)
    return Top3Answer(
        top3_probabilities=top3, winner_probabilities=winner_probs,
        winner_missing=winner_missing, pace_probabilities=None,
        model=response.model, input_tokens=response.usage.input_tokens or 0,
        requests=1, state_chars=chars,
        sent={"requests": [{"state": state, "questions": sent_q}]},
        seconds=time.perf_counter() - started)
```

（`run_plan_b` は Task 15 で足す。`__all__` にはこの時点では書かず、Task 15 で追加する。）

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 案A（3着内 Noul を1リクエストで聞く）を追加"`。

---

### Task 12: 3着内予想の記録

**Files:**
- Create: `src/keiba/store/top3_records.py`
- Test: `tests/test_top3_records.py`

**Interfaces:**
- Consumes: `Top3Answer`（Task 11）
- Produces:
  - `Top3Record(record_id: str, race_id: str | None, method: str, predicted_at: datetime, prediction_cutoff: datetime, versions: dict[str, str], top3_probabilities: dict[str, float] | None, winner_probabilities: dict[str, float] | None, winner_missing: list[str], pace_probabilities: dict[str, float] | None, ranking: list[str], usage: dict, state_chars: int | None, seconds: float, data_quality: dict)`
  - `versions_for(*, data_version: str, model: str | None) -> dict[str, str]`（キー: `data_version`, `model`, `dossier_version`, `top3_prompt_version`, `eval_rules_version`）
  - `save(record: Top3Record, *, sent: dict | None, data_dir: Path) -> Path`（記録は `data/top3/predictions/<YYYYMM>.jsonl` に追記、送った物は `data/top3/prompts/<YYYYMM>/<record_id>.json`）
  - `load_month(year_month: str, *, data_dir: Path) -> list[Top3Record]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_top3_records.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from keiba import config
from keiba.predict.top3_questions import TOP3_PROMPT_VERSION
from keiba.store import top3_records as store

JST = timezone(timedelta(hours=9))


def make_record(**kw):
    base = dict(
        record_id="abc", race_id="2026092106040511", method="plan_a",
        predicted_at=datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc),
        prediction_cutoff=datetime(2026, 9, 20, 21, 0, tzinfo=JST),
        versions=store.versions_for(data_version="v1", model="jev-1.13.0"),
        top3_probabilities={"01": 0.7, "02": 0.4}, winner_probabilities={"01": 0.6, "02": 0.4},
        winner_missing=[], pace_probabilities=None, ranking=["01", "02"],
        usage={"input_tokens": 500, "requests": 1}, state_chars=1234, seconds=1.5,
        data_quality={"notes": []})
    base.update(kw)
    return store.Top3Record(**base)


def test_versions_for_has_all_keys():
    v = store.versions_for(data_version="v1", model="jev-1.13.0")
    assert v == {"data_version": "v1", "model": "jev-1.13.0",
                 "dossier_version": config.DOSSIER_VERSION,
                 "top3_prompt_version": TOP3_PROMPT_VERSION,
                 "eval_rules_version": config.TOP3_EVAL_RULES_VERSION}


def test_round_trip(tmp_path):
    record = make_record()
    store.save(record, sent=None, data_dir=tmp_path)
    assert store.load_month("202609", data_dir=tmp_path) == [record]


def test_python_method_has_no_probabilities(tmp_path):
    record = make_record(method="python", top3_probabilities=None, winner_probabilities=None,
                         versions=store.versions_for(data_version="v1", model=None))
    store.save(record, sent=None, data_dir=tmp_path)
    loaded = store.load_month("202609", data_dir=tmp_path)[0]
    assert loaded.top3_probabilities is None
    assert loaded.versions["model"] is None


def test_sent_is_saved_separately(tmp_path):
    record = make_record()
    store.save(record, sent={"requests": [{"state": {"a": 1}, "questions": {}}]}, data_dir=tmp_path)
    path = tmp_path / "top3" / "prompts" / "202609" / "abc.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["record_id"] == "abc"
    assert payload["requests"][0]["state"] == {"a": 1}
    line = (tmp_path / "top3" / "predictions" / "202609.jsonl").read_text(encoding="utf-8")
    assert '"state"' not in line  # 記録本体には送った物を混ぜない


def test_load_missing_month_is_empty(tmp_path):
    assert store.load_month("209901", data_dir=tmp_path) == []
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_records.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`src/keiba/store/top3_records.py`:

```python
"""3着内予想の記録。勝率系の PredictionRecord とは別の型・別の木に置く。

勝率と3着内確率は意味が違うので、同じ記録形式・同じファイルに混ぜない
（report3 §9）。月ごとの jsonl に答えを、送った state と質問は1予想1ファイルで
別の木に置く（store/prompts.py と同じ理由: 18頭で数十万文字あり、答えを読む
たびに引きずらないため）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from keiba import config
from keiba.predict.top3_questions import TOP3_PROMPT_VERSION

__all__ = ["Top3Record", "load_month", "save", "versions_for"]


@dataclass(frozen=True)
class Top3Record:
    record_id: str
    race_id: str | None
    #: "python" / "plan_a" / "plan_b"。同じ Jev 応答から作る "winner_same_state" は
    #: 記録を分けず、plan_a / plan_b の記録の winner_probabilities から評価側が作る。
    method: str
    predicted_at: datetime
    prediction_cutoff: datetime
    versions: dict
    top3_probabilities: dict | None
    winner_probabilities: dict | None
    winner_missing: list
    pace_probabilities: dict | None
    ranking: list
    usage: dict
    state_chars: int | None
    seconds: float
    data_quality: dict


def versions_for(*, data_version: str, model: str | None) -> dict:
    return {
        "data_version": data_version,
        "model": model,
        "dossier_version": config.DOSSIER_VERSION,
        "top3_prompt_version": TOP3_PROMPT_VERSION,
        "eval_rules_version": config.TOP3_EVAL_RULES_VERSION,
    }


def _root(data_dir: Path) -> Path:
    return Path(data_dir) / "top3"


def _to_json(record: Top3Record) -> dict:
    payload = asdict(record)
    payload["predicted_at"] = record.predicted_at.isoformat()
    payload["prediction_cutoff"] = record.prediction_cutoff.isoformat()
    return payload


def _from_json(payload: dict) -> Top3Record:
    payload = dict(payload)
    payload["predicted_at"] = datetime.fromisoformat(payload["predicted_at"])
    payload["prediction_cutoff"] = datetime.fromisoformat(payload["prediction_cutoff"])
    return Top3Record(**payload)


def save(record: Top3Record, *, sent: dict | None, data_dir: Path) -> Path:
    month = record.predicted_at.strftime("%Y%m")
    path = _root(data_dir) / "predictions" / f"{month}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json(record), ensure_ascii=False) + "\n")
    if sent is not None:
        prompt_path = _root(data_dir) / "prompts" / month / f"{record.record_id}.json"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(json.dumps(
            {"record_id": record.record_id, "race_id": record.race_id,
             "versions": record.versions, **sent},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def load_month(year_month: str, *, data_dir: Path) -> list[Top3Record]:
    path = _root(data_dir) / "predictions" / f"{year_month}.jsonl"
    if not path.exists():
        return []
    return [_from_json(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_records.py -v`
Expected: PASS

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 3着内予想の記録形式を追加"`。

---

### Task 13: 比較バックテストと CLI（最終評価期間のガード付き）

**Files:**
- Create: `src/keiba/evaluate/top3_backtest.py`
- Create: `scripts/backtest_top3.py`
- Test: `tests/test_top3_backtest.py`

**Interfaces:**
- Consumes: `evaluate.backtest.race_input_from_history` / `prediction_cutoff_for`、`build_dossier`、`top3_baselines.*`、`top3.run_plan_a`（Task 15 で `run_plan_b`）、`top3_metrics.*`、`top3_records.*`、`input.match.AmbiguousMatchError`
- Produces:
  - `METHODS = ("python", "plan_a", "plan_b")`、評価上の系列 `SERIES = ("python", "plan_a", "winner_same_state_a", "plan_b", "winner_same_state_b")`
  - `Top3SkipReason = Literal["jump", "no_entries", "no_result", "match_ambiguous", "state_too_large", "jev_failed", "incomplete_answer", "final_period_locked"]`
  - `Top3Row(race_id: str, period: str, skipped: str | None, outcome: Top3Outcome | None, series: dict[str, dict], usage: dict[str, dict], python_all_tied: bool | None)` — `series[name] = {"ranking": [...], "probabilities": {...} | None}`、`usage[method] = {"input_tokens", "requests", "seconds", "state_chars"}`
  - `run(races, repo, results, client, *, methods: Sequence[str], n: int, tuning_until: date, allow_final: bool, max_recent_runs: int, save_records: bool, data_dir: Path, pace_definition_for: Callable[[RaceInput], dict | None] | None = None, on_row: Callable[[Top3Row], None] | None = None) -> list[Top3Row]`
  - `report(rows: Sequence[Top3Row], *, n: int, run_info: dict) -> dict`
  - `dry_run(races, repo, *, n: int, max_recent_runs: int) -> dict`（state の文字数の分布と、上限超えの race_id。Jev を呼ばない）

期間の区分: 予想締切の日付が `tuning_until` 以下なら `"tuning"`（プロンプト調整に使ってよい）、それより後は `"final"`（最後に1度だけ評価する）。`allow_final=False` の実行では final のレースを `final_period_locked` でスキップし、Jev を呼ばない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_top3_backtest.py`（CSV フィクスチャ + 偽クライアント。`tests/test_backtest.py` と同じ作り）:

```python
from __future__ import annotations

import types
from datetime import date

import pytest

from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.evaluate import top3_backtest as tbt
from tests.fixtures import csv_data as fx

TUNING_RACE = "2026030106010101"   # 締切 2026-02-28
FINAL_RACE = "2026032106011101"    # 締切 2026-03-20


def _repos(tmp_path):
    races = [fx.make_race(race_id=TUNING_RACE, hasso_at="2026-03-01T15:00:00+09:00"),
             fx.make_race(race_id=FINAL_RACE, hasso_at="2026-03-21T15:25:00+09:00")]
    race_horses = []
    for race_id in (TUNING_RACE, FINAL_RACE):
        for i in range(1, 6):
            race_horses.append(fx.make_race_horse(
                race_id=race_id, umaban=f"{i:02d}", ketto_num=f"{race_id}{i}",
                bamei=f"ウマ{race_id[-4:]}{i}", result_kakutei_jyuni=str(i)))
    store = DataStore.open_current(fx.write_dataset(tmp_path, races, race_horses, []))
    return Repository(store), ResultRepository(store)


class Client:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kw):
        self.calls += 1
        umabans = list(state["horses"])
        nouls = {f"top3_{u}": types.SimpleNamespace(noul=0.9 - 0.1 * i)
                 for i, u in enumerate(umabans)}
        probs = {u: 1 / len(umabans) for u in umabans}
        return types.SimpleNamespace(
            nouls=nouls, choices={"winner": types.SimpleNamespace(
                choice=umabans[0], probabilities=probs, confidence=0.5)},
            model="jev-1.13.0", usage=types.SimpleNamespace(input_tokens=100))


def _run(tmp_path, *, methods=("python", "plan_a"), allow_final=False, client=None):
    repo, results = _repos(tmp_path)
    races = [results.race(TUNING_RACE), results.race(FINAL_RACE)]
    return tbt.run(races, repo, results, client or Client(), methods=methods, n=3,
                   tuning_until=date(2026, 3, 10), allow_final=allow_final,
                   max_recent_runs=5, save_records=False, data_dir=tmp_path)


def test_final_period_is_locked_by_default(tmp_path):
    client = Client()
    rows = _run(tmp_path, client=client)
    by_id = {r.race_id: r for r in rows}
    assert by_id[TUNING_RACE].period == "tuning"
    assert by_id[FINAL_RACE].skipped == "final_period_locked"
    assert client.calls == 1  # final のレースには Jev を呼んでいない


def test_allow_final_runs_final_period(tmp_path):
    rows = _run(tmp_path, allow_final=True)
    assert {r.race_id: r.skipped for r in rows}[FINAL_RACE] is None


def test_series_include_winner_same_state(tmp_path):
    row = _run(tmp_path)[0]
    assert set(row.series) == {"python", "plan_a", "winner_same_state_a"}
    assert row.series["plan_a"]["ranking"][:3] == ["01", "02", "03"]
    assert row.series["python"]["probabilities"] is not None  # 一様確率


def test_python_only_does_not_call_jev(tmp_path):
    client = Client()
    _run(tmp_path, methods=("python",), allow_final=True, client=client)
    assert client.calls == 0


def test_state_too_large_is_skipped_with_reason(tmp_path, monkeypatch):
    from keiba import config
    monkeypatch.setattr(config, "STATE_MAX_CHARS", 10)
    rows = _run(tmp_path, methods=("plan_a",))
    assert rows[0].skipped == "state_too_large"


def test_incomplete_answer_is_skipped_with_reason(tmp_path):
    class Partial(Client):
        def system_one(self, state, questions, **kw):
            response = super().system_one(state, questions, **kw)
            response.nouls.pop(next(iter(response.nouls)))
            return response
    rows = _run(tmp_path, methods=("plan_a",), client=Partial())
    assert rows[0].skipped == "incomplete_answer"


def test_report_has_per_series_metrics_and_skips(tmp_path):
    rows = _run(tmp_path)
    rep = tbt.report(rows, n=3, run_info={"note": "test"})
    assert rep["skip_breakdown"]["final_period_locked"] == 1
    assert rep["by_period"]["tuning"]["series"]["plan_a"]["mean_hits"] == 3.0
    assert "plan_a_vs_python" in rep["by_period"]["tuning"]["comparisons"]
    assert rep["usage"]["plan_a"]["input_tokens"] == 100
    assert rep["versions"]["eval_rules_version"] == "1"


def test_dry_run_reports_state_sizes_without_client(tmp_path):
    repo, results = _repos(tmp_path)
    races = [results.race(TUNING_RACE)]
    out = tbt.dry_run(races, repo, n=3, max_recent_runs=5)
    assert out["n_races"] == 1
    assert out["max_state_chars"] > 0
    assert out["over_limit"] == []
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_backtest.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`src/keiba/evaluate/top3_backtest.py`:

```python
"""3着内予想の比較バックテスト（report3 §7 の比較手順）。

1レースの失敗では止めず、理由付きでスキップする（evaluate/backtest.py と同じ
方針）。プロンプト調整に使う期間（tuning）と、最後に評価する期間（final）を
分け、final は allow_final=True の実行でしか予想しない——結果を見ながら質問を
調整したレースは、重みを学習していなくても未使用の検証データではない
（report3 §8）。
"""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jev_client import JevRequestError
from keiba import config
from keiba.data.repository import Repository, ResultRepository
from keiba.evaluate import top3_metrics as tm
from keiba.evaluate.backtest import prediction_cutoff_for, race_input_from_history
from keiba.features import dossier as dossier_mod
from keiba.input.match import AmbiguousMatchError
from keiba.models import RaceInput, RaceRow
from keiba.predict import top3, top3_baselines
from keiba.store import top3_records

__all__ = ["METHODS", "SERIES", "Top3Row", "dry_run", "report", "run"]

METHODS: tuple[str, ...] = ("python", "plan_a", "plan_b")
SERIES: tuple[str, ...] = ("python", "plan_a", "winner_same_state_a",
                           "plan_b", "winner_same_state_b")
Top3SkipReason = Literal["jump", "no_entries", "no_result", "match_ambiguous",
                         "state_too_large", "jev_failed", "incomplete_answer",
                         "final_period_locked"]
_ALL_SKIPS: tuple[str, ...] = ("jump", "no_entries", "no_result", "match_ambiguous",
                               "state_too_large", "jev_failed", "incomplete_answer",
                               "final_period_locked")


@dataclass(frozen=True)
class Top3Row:
    race_id: str
    period: str
    skipped: str | None
    outcome: tm.Top3Outcome | None
    series: dict
    usage: dict
    python_all_tied: bool | None


def _period(race: RaceRow, tuning_until: date) -> str:
    return "tuning" if prediction_cutoff_for(race).date() <= tuning_until else "final"


def _rank(probabilities: dict[str, float]) -> list[str]:
    return [u for u, _ in sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))]


def _skip(race: RaceRow, period: str, reason: str) -> Top3Row:
    return Top3Row(race.race_id, period, reason, None, {}, {}, None)


def _record(race_input: RaceInput, dossier: dict, method: str, *, ranking, answer=None,
            probabilities=None) -> top3_records.Top3Record:
    return top3_records.Top3Record(
        record_id=uuid.uuid4().hex, race_id=race_input.race_id, method=method,
        predicted_at=datetime.now(timezone.utc),
        prediction_cutoff=race_input.prediction_cutoff,
        versions=top3_records.versions_for(
            data_version=dossier["data_quality"]["data_version"],
            model=answer.model if answer else None),
        top3_probabilities=answer.top3_probabilities if answer else probabilities,
        winner_probabilities=answer.winner_probabilities if answer else None,
        winner_missing=list(answer.winner_missing) if answer else [],
        pace_probabilities=answer.pace_probabilities if answer else None,
        ranking=list(ranking),
        usage=({"input_tokens": answer.input_tokens, "requests": answer.requests}
               if answer else {"input_tokens": 0, "requests": 0}),
        state_chars=answer.state_chars if answer else None,
        seconds=answer.seconds if answer else 0.0,
        data_quality={"history_status_counts": dossier["field_summary"]["history_status_counts"],
                      "course_identified": dossier["course"]["identified"]},
    )


def run(races: Sequence[RaceRow], repo: Repository, results: ResultRepository, client: Any, *,
        methods: Sequence[str], n: int, tuning_until: date, allow_final: bool,
        max_recent_runs: int, save_records: bool, data_dir: Path,
        pace_definition_for: Callable[[RaceInput], dict | None] | None = None,
        on_row: Callable[[Top3Row], None] | None = None) -> list[Top3Row]:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"未知の方式: {sorted(unknown)}")
    rows: list[Top3Row] = []

    def emit(row: Top3Row) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    for race in races:
        period = _period(race, tuning_until)
        if not race.is_flat:
            emit(_skip(race, period, "jump"))
            continue
        race_results = results.results(race.race_id)
        entries = [r.entry for r in race_results]
        if len(entries) < 2:
            emit(_skip(race, period, "no_entries"))
            continue
        outcome = tm.outcome_from_results(race.race_id, race_results)
        if outcome is None:
            emit(_skip(race, period, "no_result"))
            continue
        if period == "final" and not allow_final:
            emit(_skip(race, period, "final_period_locked"))
            continue

        race_input = race_input_from_history(race, entries, top_n=n)
        try:
            dossier = dossier_mod.build_dossier(race_input, repo, max_recent_runs=max_recent_runs)
        except AmbiguousMatchError:
            emit(_skip(race, period, "match_ambiguous"))
            continue

        series: dict[str, dict] = {}
        usage: dict[str, dict] = {}
        python_all_tied = None
        skipped = None
        for method in methods:
            try:
                if method == "python":
                    base = top3_baselines.ability_ranking(dossier)
                    probs = top3_baselines.uniform_probabilities(dossier)
                    python_all_tied = base.all_tied
                    series["python"] = {"ranking": list(base.ranking), "probabilities": probs}
                    record, sent = _record(race_input, dossier, "python",
                                           ranking=base.ranking, probabilities=probs), None
                else:
                    if method == "plan_a":
                        answer = top3.run_plan_a(client, dossier)
                    else:
                        pace_definition = (pace_definition_for(race_input)
                                           if pace_definition_for else None)
                        answer = top3.run_plan_b(client, dossier,
                                                 pace_definition=pace_definition)
                    suffix = method[-1]
                    ranking = _rank(answer.top3_probabilities)
                    series[method] = {"ranking": ranking,
                                      "probabilities": answer.top3_probabilities}
                    series[f"winner_same_state_{suffix}"] = {
                        "ranking": _rank(answer.winner_probabilities), "probabilities": None}
                    usage[method] = {"input_tokens": answer.input_tokens,
                                     "requests": answer.requests, "seconds": answer.seconds,
                                     "state_chars": answer.state_chars}
                    record, sent = _record(race_input, dossier, method, ranking=ranking,
                                           answer=answer), answer.sent
            except dossier_mod.StateTooLargeError:
                skipped = "state_too_large"
                break
            except top3.Top3IncompleteError:
                skipped = "incomplete_answer"
                break
            except JevRequestError:
                skipped = "jev_failed"
                break
            if save_records:
                top3_records.save(record, sent=sent, data_dir=data_dir)
        if skipped is not None:
            emit(Top3Row(race.race_id, period, skipped, None, {}, usage, None))
            continue
        emit(Top3Row(race.race_id, period, None, outcome, series, usage, python_all_tied))
    return rows


def _series_scores(rows: Sequence[Top3Row], name: str, n: int) -> dict[str, dict]:
    return {row.race_id: tm.score_race(row.series[name]["ranking"][:n],
                                       row.series[name]["probabilities"], row.outcome)
            for row in rows if name in row.series}


def _period_block(rows: Sequence[Top3Row], n: int) -> dict:
    scored_rows = [r for r in rows if r.skipped is None]
    scores = {name: _series_scores(scored_rows, name, n) for name in SERIES}
    series = {name: tm.aggregate(list(s.values()), n=n) for name, s in scores.items() if s}
    comparisons = {}
    for challenger in ("plan_a", "plan_b"):
        if challenger not in series:
            continue
        suffix = challenger[-1]
        for reference in ("python", f"winner_same_state_{suffix}"):
            if reference in series:
                comparisons[f"{challenger}_vs_{reference}"] = tm.paired_bootstrap(
                    {k: v["hits"] for k, v in scores[challenger].items()},
                    {k: v["hits"] for k, v in scores[reference].items()},
                    cluster_of=tm.race_date_cluster)
    if "plan_a" in series and "plan_b" in series:
        comparisons["plan_b_vs_plan_a"] = tm.paired_bootstrap(
            {k: v["hits"] for k, v in scores["plan_b"].items()},
            {k: v["hits"] for k, v in scores["plan_a"].items()},
            cluster_of=tm.race_date_cluster)
    return {
        "n_rows": len(rows),
        "n_scored": len(scored_rows),
        "series": series,
        "comparisons": comparisons,
        "n_python_all_tied": sum(1 for r in scored_rows if r.python_all_tied),
    }


def report(rows: Sequence[Top3Row], *, n: int, run_info: dict) -> dict:
    usage: dict[str, dict] = {}
    for row in rows:
        for method, u in row.usage.items():
            total = usage.setdefault(method, {"input_tokens": 0, "requests": 0,
                                              "seconds": 0.0, "races": 0})
            total["input_tokens"] += u["input_tokens"]
            total["requests"] += u["requests"]
            total["seconds"] += u["seconds"]
            total["races"] += 1
    for total in usage.values():
        total["usd"] = total["input_tokens"] / 1_000_000 * 0.042  # 入力 $0.042 / 1M（CLAUDE.md）
    return {
        "n": n,
        "versions": top3_records.versions_for(data_version=run_info.get("data_version", ""),
                                              model=None),
        "skip_breakdown": {reason: sum(1 for r in rows if r.skipped == reason)
                           for reason in _ALL_SKIPS},
        "by_period": {period: _period_block([r for r in rows if r.period == period], n)
                      for period in ("tuning", "final")},
        "usage": usage,
        "run": run_info,
    }


def dry_run(races: Sequence[RaceRow], repo: Repository, *, n: int, max_recent_runs: int) -> dict:
    """state の大きさだけを測る。Jev を呼ばない。結果（ResultRepository）も使わない。"""
    sizes: dict[str, int] = {}
    for race in races:
        if not race.is_flat:
            continue
        entries = repo.entries(race.race_id, as_of=prediction_cutoff_for(race))
        if len(entries) < 2:
            continue
        race_input = race_input_from_history(race, entries, top_n=n)
        try:
            d = dossier_mod.build_dossier(race_input, repo, max_recent_runs=max_recent_runs)
        except AmbiguousMatchError:
            continue
        sizes[race.race_id] = dossier_mod.state_chars(dossier_mod.to_state(d))
    values = list(sizes.values())
    return {
        "n_races": len(sizes),
        "max_state_chars": max(values) if values else 0,
        "median_state_chars": statistics.median(values) if values else 0,
        "limit": config.STATE_MAX_CHARS,
        "over_limit": sorted(r for r, c in sizes.items() if c > config.STATE_MAX_CHARS),
        # 実測の比（約0.56トークン/文字）での概算。Task 14 で見直す。
        "estimated_tokens_per_race_plan_a": int(statistics.mean(values) * 0.56) if values else 0,
    }
```

注意: `dry_run` は `repo.entries()`（`as_of` 付き、`EntryRow` しか返さない）を使い、評価用の結果に触れない。`run()` は採点のために `ResultRepository` を使うが、予想の入力は `race_input_from_history`（`EntryRow` だけ）を通る。

`scripts/backtest_top3.py`:

```python
"""3着内予想の比較バックテスト。

    # 課金なし: state の大きさを測る
    .venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260922 \
        --tuning-until 20260831 --dry-run
    # 課金あり（所有者の承認を得てから）: 調整期間を10レース
    .venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260831 \
        --tuning-until 20260831 --methods python,plan_a --limit 10

最終評価期間（--tuning-until より後）は --final を付けたときだけ予想する。
一度 --final を実行した期間を再び評価するには --final-rerun-reason が要り、
理由は data/top3/final_runs.jsonl に残る（report3 §8）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest import _sample_evenly  # noqa: E402  同じ間引き規則を使う（fix3 決定1）
from jev_client import JevConfigError, create_client  # noqa: E402
from keiba import config  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate import top3_backtest as tbt  # noqa: E402
from keiba.evaluate.backtest import hasso_window_for_cutoff_window  # noqa: E402
from keiba.features.pace import fit_pace_definition  # noqa: E402
from keiba.predict.top3_questions import TOP3_PROMPT_VERSION  # noqa: E402

JST = timezone(timedelta(hours=9))


def _day(value: str, *, end: bool = False) -> datetime:
    d = datetime.strptime(value, "%Y%m%d").replace(tzinfo=JST)
    return d.replace(hour=23, minute=59, second=59) if end else d


def _to_jsonable(value):
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def _check_final_guard(args, data_dir: Path) -> str | None:
    """--final の二度目以降は理由を要求する。問題があればメッセージを返す。"""
    log = data_dir / "top3" / "final_runs.jsonl"
    if not args.final:
        return None
    if log.exists() and log.read_text(encoding="utf-8").strip() and not args.final_rerun_reason:
        return ("最終評価期間はすでに評価済みです（data/top3/final_runs.jsonl）。"
                "再評価するなら --final-rerun-reason に理由を書いてください。")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "tuning_until": args.tuning_until, "to": args.date_to,
            "methods": args.methods, "top3_prompt_version": TOP3_PROMPT_VERSION,
            "dossier_version": config.DOSSIER_VERSION,
            "reason": args.final_rerun_reason}, ensure_ascii=False) + "\n")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD")
    p.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD")
    p.add_argument("--tuning-until", required=True, metavar="YYYYMMDD")
    p.add_argument("--methods", default="python,plan_a")
    p.add_argument("--n", type=int, default=config.DEFAULT_TOP_N)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-recent-runs", type=int, default=config.DEFAULT_DOSSIER_RECENT_RUNS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--final-rerun-reason", default=None)
    p.add_argument("--save-records", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=Path, default=Path("data/top3/backtests"))
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = p.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    store = DataStore.open_current(args.data_dir)
    repo, results = Repository(store), ResultRepository(store)
    start, end = hasso_window_for_cutoff_window(_day(args.date_from), _day(args.date_to, end=True))
    races = _sample_evenly(results.races_between(start, end), args.limit)
    print(f"対象レース: {len(races)}（データ版 {store.version_id}）")

    if args.dry_run:
        print(json.dumps(tbt.dry_run(races, repo, n=args.n, max_recent_runs=args.max_recent_runs),
                         ensure_ascii=False, indent=2))
        return 0

    problem = _check_final_guard(args, args.data_dir)
    if problem:
        print(problem)
        return 2

    all_races = results.races_between(_day("20000101"), end)

    def pace_definition_for(race_input):
        return fit_pace_definition(all_races, target=race_input, as_of=race_input.prediction_cutoff)

    needs_jev = any(m != "python" for m in methods)
    try:
        client = create_client() if needs_jev else None
    except JevConfigError as error:
        print(error)
        return 2

    tuning_until = _day(args.tuning_until).date()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    def write_row(row):
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")

    rows = tbt.run(races, repo, results, client, methods=methods, n=args.n,
                   tuning_until=tuning_until, allow_final=args.final,
                   max_recent_runs=args.max_recent_runs, save_records=args.save_records,
                   data_dir=args.data_dir, pace_definition_for=pace_definition_for,
                   on_row=write_row)
    rep = tbt.report(rows, n=args.n, run_info={
        "data_version": store.version_id, "from": args.date_from, "to": args.date_to,
        "tuning_until": args.tuning_until, "limit": args.limit, "methods": methods,
        "final": args.final, "max_recent_runs": args.max_recent_runs})
    (out_dir / "report.json").write_text(
        json.dumps(_to_jsonable(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_to_jsonable(rep["by_period"]), ensure_ascii=False, indent=2))
    print(f"スキップ: {rep['skip_breakdown']}")
    print(f"使用量: {rep['usage']}")
    print(f"保存: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3_backtest.py -v`
Expected: `plan_b` を使うテストは無いので PASS（`top3.run_plan_b` は Task 15 で足す。`run()` の `plan_b` 分岐は Task 15 のテストで通す）

- [ ] **Step 5: 課金なしの確認**

Run: `.venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260922 --tuning-until 20260831 --limit 100 --dry-run`
Expected: `over_limit` が空、`max_state_chars` ≤ 48,000。空でなければ `--max-recent-runs 3` で再測定し、既定値（`config.DEFAULT_DOSSIER_RECENT_RUNS`）を下げるかを所有者に相談する。

Run: `.venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260831 --tuning-until 20260831 --methods python --limit 100`
Expected: Jev を呼ばずに Python 基準の報告が出る（`usage` は空）。

- [ ] **Step 6: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 3着内の比較バックテストと CLI を追加"`。

---

### Task 14: 案Aの少数レース試行 【課金あり・所有者の承認が要る】

**Files:**
- Modify: `src/keiba/config.py`（`STATE_MAX_CHARS` の見直しが必要な場合のみ）
- Modify: `docs/superpowers/specs/2026-09-23-top3-prediction-design.md`（試行結果の節を追記）

- [ ] **Step 1: 費用の見積もりを所有者に示して承認を得る**

Task 13 Step 5 の `estimated_tokens_per_race_plan_a` × 10レースと、`$0.042 / 1M` で換算した額を示す（目安: 1レース 2万トークンなら 10レースで約 $0.008）。**承認が無ければここで止める。**

- [ ] **Step 2: 調整期間の10レースで案Aを実行する**

Run: `.venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260831 --tuning-until 20260831 --methods python,plan_a --limit 10`

- [ ] **Step 3: 記録を点検する**

1. `data/top3/prompts/<YYYYMM>/*.json` を1件開き、送った state に対象レースの結果が無いこと、質問が全頭分あることを目視で確認する。
2. `report.json` の `skip_breakdown` が想定内（`jev_failed` / `incomplete_answer` が0件）か。
3. `series.plan_a.prob_sum_minus_actual_mean`（確率合計の偏り）を控える。3 から大きく外れていても**このタスクでは補正しない**（spec §6）。
4. 記録の `usage.input_tokens` と `state_chars` からトークン/文字の比を計算し、0.56 から ±20% 以上ずれていれば `config.STATE_MAX_CHARS` のコメントと値を実測で書き直す（`27_000 / 比` を切り下げ）。

- [ ] **Step 4: 所見を仕様に残す**

`docs/superpowers/specs/2026-09-23-top3-prediction-design.md` の末尾に「案A 試行（10レース）」節として、実行日・データ版・版番号・費用・トークン/文字比・確率合計の偏り・気づいた点を書く。**10レースの的中数から優劣を結論しない**（標本が小さすぎる）。

- [ ] **Step 5: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "docs: 案A の少数レース試行の結果を記録"`。

---

## Phase 4 — 案B と比較

### Task 15: 案B（展開予測を先に行う2段階）

**Files:**
- Modify: `src/keiba/predict/top3.py`
- Test: `tests/test_top3.py`（追記）、`tests/test_top3_backtest.py`（追記）

**Interfaces:**
- Consumes: `top3_questions.pace_questions`、`fit_pace_definition` の返り値（`{"fast_max", "standard_max", "stratum", "race_count", "note"}`）
- Produces: `top3.run_plan_b(client, dossier: dict, *, pace_definition: dict | None) -> Top3Answer`
  - `pace_definition` が `None` なら `PaceUnavailableError`（`top3_backtest.run` は `"jev_failed"` ではなく新しい理由 `"pace_unavailable"` でスキップする）
  - 1回目の state = `to_state(dossier)` に `pace_definition` を足したもの。2回目の state = `to_state(dossier, pace_forecast={"probabilities": ..., "definition": pace_definition, "note": "前段の予測。確定した事実ではない。"})`
  - `requests == 2`、`input_tokens` は2回の合計、`sent["requests"]` は2件

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_top3.py` に追記）

```python
PACE_DEF = {"fast_max": 35.0, "standard_max": 36.0, "stratum": {"track_kind": "turf"},
            "race_count": 40, "note": "n"}


def plan_b_handler(state, questions):
    if "pace" in questions:
        assert "pace_definition" in state
        return Resp(choices={"pace": types.SimpleNamespace(
            choice="fast", probabilities={"fast": 0.5, "standard": 0.3, "slow": 0.2},
            confidence=0.4)}, tokens=300)
    assert state["pace_forecast"]["probabilities"] == {"fast": 0.5, "standard": 0.3, "slow": 0.2}
    assert "pace_definition" not in state
    return full_handler(state, questions)


def test_plan_b_two_requests_and_passes_distribution():
    client = Client(plan_b_handler)
    answer = top3.run_plan_b(client, DOSSIER, pace_definition=PACE_DEF)
    assert len(client.calls) == 2
    assert answer.requests == 2
    assert answer.input_tokens == 800
    assert answer.pace_probabilities == {"fast": 0.5, "standard": 0.3, "slow": 0.2}
    assert len(answer.sent["requests"]) == 2


def test_plan_b_second_request_mentions_pace_forecast():
    client = Client(plan_b_handler)
    top3.run_plan_b(client, DOSSIER, pace_definition=PACE_DEF)
    assert "pace_forecast" in client.calls[1][1]["top3_01"].instructions


def test_plan_b_without_pace_definition_does_not_send():
    client = Client(plan_b_handler)
    with pytest.raises(top3.PaceUnavailableError):
        top3.run_plan_b(client, DOSSIER, pace_definition=None)
    assert client.calls == []
```

`tests/test_top3_backtest.py` に追記:

```python
def test_plan_b_skips_when_pace_definition_unavailable(tmp_path):
    repo, results = _repos(tmp_path)
    rows = tbt.run([results.race(TUNING_RACE)], repo, results, Client(),
                   methods=("plan_b",), n=3, tuning_until=date(2026, 3, 10),
                   allow_final=False, max_recent_runs=5, save_records=False,
                   data_dir=tmp_path, pace_definition_for=lambda ri: None)
    assert rows[0].skipped == "pace_unavailable"
```

- [ ] **Step 2: 失敗を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3.py tests/test_top3_backtest.py -v`
Expected: FAIL（`run_plan_b` / `PaceUnavailableError` が無い）

- [ ] **Step 3: 実装する**

`src/keiba/predict/top3.py` に追記（`__all__` に `"PaceUnavailableError"`, `"run_plan_b"` を追加）:

```python
class PaceUnavailableError(RuntimeError):
    """ペース区分の定義が作れない。pace の質問文は state の pace_definition を
    名指しするので、定義が無いまま送ってはいけない（既存 spec §8.2 と同じ理由）。"""


_PACE_NOTE = "前段で予測した展開。確定した事実ではない。"


def run_plan_b(client: Any, dossier: dict, *, pace_definition: dict | None) -> Top3Answer:
    if pace_definition is None:
        raise PaceUnavailableError("ペース区分の定義（pace_definition）が作れないため案Bを実行しない。")
    started = time.perf_counter()

    pace_state = to_state(dossier)
    pace_state["pace_definition"] = pace_definition
    check_state_size(pace_state, limit=config.STATE_MAX_CHARS)
    pace_q = tq.pace_questions()
    pace_response = ask(client, pace_state, pace_q)
    if "pace" not in pace_response.choices:
        raise Top3IncompleteError(["pace"])
    pace_probs = {k: float(v) for k, v in pace_response.choices["pace"].probabilities.items()}

    # 最尤区分だけでなく分布のまま渡す（確定した事実として扱わせない）。
    state = to_state(dossier, pace_forecast={
        "probabilities": pace_probs, "definition": pace_definition, "note": _PACE_NOTE})
    response, top3, winner_probs, winner_missing, chars, sent_q = _ask_top3(
        client, dossier, state, with_pace_forecast=True)
    return Top3Answer(
        top3_probabilities=top3, winner_probabilities=winner_probs,
        winner_missing=winner_missing, pace_probabilities=pace_probs,
        model=response.model,
        input_tokens=(pace_response.usage.input_tokens or 0) + (response.usage.input_tokens or 0),
        requests=2, state_chars=chars,
        sent={"requests": [{"state": pace_state, "questions": _dump(pace_q)},
                           {"state": state, "questions": sent_q}]},
        seconds=time.perf_counter() - started)
```

`src/keiba/evaluate/top3_backtest.py`:
- `Top3SkipReason` と `_ALL_SKIPS` に `"pace_unavailable"` を足す。
- `run()` の `except` 群に `except top3.PaceUnavailableError: skipped = "pace_unavailable"; break` を `StateTooLargeError` の直後に足す。

- [ ] **Step 4: 展開予測の当たり方を報告に足す**（案Bが前段で何を当てたかの診断）

`tests/test_top3_backtest.py` に追記:

```python
def test_report_pace_accuracy_for_plan_b(tmp_path):
    from keiba.evaluate.top3_backtest import pace_accuracy
    rows_pace = [("fast", "fast"), ("slow", "fast"), ("standard", None)]
    result = pace_accuracy(rows_pace)
    assert result == {"n_scored": 2, "n_unscorable": 1, "argmax_accuracy": 0.5}
```

`src/keiba/evaluate/top3_backtest.py` に:

```python
def pace_accuracy(pairs: Sequence[tuple[str, str | None]]) -> dict:
    """(予測の最尤区分, 実際の区分) の組。実際が None（前半3Fが無い）は数えない。"""
    scored = [(p, a) for p, a in pairs if a is not None]
    return {"n_scored": len(scored), "n_unscorable": len(pairs) - len(scored),
            "argmax_accuracy": (sum(1 for p, a in scored if p == a) / len(scored)
                                if scored else float("nan"))}
```

`Top3Row` の最後にフィールドを足す（既定値付きなので既存の生成箇所は壊れない）:

```python
    #: 案Bの (予測の最尤区分, 実際の区分)。案Bを回していない行は None。
    pace_pair: tuple | None = None
```

`run()` の先頭で `from keiba.features.pace import classify_pace` を import し、レースごとのループの冒頭（`series = {}` の隣）で `pace_pair = None` を置く。`plan_b` の分岐で `answer` を得た直後に:

```python
                    if method == "plan_b":
                        # 前半3Fは結果なので評価側（ResultRepository）からだけ読む。
                        actual_race = results.race(race.race_id)
                        actual = classify_pace(
                            actual_race.first_3f_sec if actual_race else None, pace_definition)
                        predicted = max(answer.pace_probabilities,
                                        key=answer.pace_probabilities.get)
                        pace_pair = (predicted, actual)
```

成功行の生成を `Top3Row(race.race_id, period, None, outcome, series, usage, python_all_tied, pace_pair)` に変える。`report()` の戻り値に `"pace_accuracy": pace_accuracy([r.pace_pair for r in rows if r.pace_pair is not None])` を足す。

- [ ] **Step 5: 合格を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_top3.py tests/test_top3_backtest.py -v`
Expected: PASS

- [ ] **Step 6: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "feat: 案B（展開予測→3着内の2段階）を追加"`。

---

### Task 16: 調整期間での比較と最終評価 【課金あり・所有者の承認が要る】

**Files:**
- Modify: `docs/superpowers/specs/2026-09-23-top3-prediction-design.md`（結果の節）
- Modify: `CLAUDE.md`（新しい系統の所在と使い方、結果の要約）

- [ ] **Step 1: 期間を決めて所有者の承認を得る**

データ版を取り直した後（Task 1 Step 7）の期間から、**調整期間**（例: 古い側 2/3）と**最終評価期間**（新しい側 1/3）の境界 `--tuning-until` を決める。費用の見積もり（`--dry-run` の概算 × レース数 × 案Bは約2倍）を示して承認を得る。**最終評価期間はこの時点では予想しない。**

- [ ] **Step 2: 調整期間で3方式を比較する**

Run: `.venv/Scripts/python.exe scripts/backtest_top3.py --from <調整開始> --to <調整終了> --tuning-until <調整終了> --methods python,plan_a,plan_b --limit 100`

報告から控える: 各系列の `mean_hits` / `hit_distribution` / `all_top3_selected_rate` / `brier`、比較の `mean_diff` と95%区間（`plan_a_vs_python`・`plan_a_vs_winner_same_state_a`・`plan_b_vs_plan_a`）、`pace_accuracy`、費用・時間。

- [ ] **Step 3: プロンプトを調整するなら、調整期間の中だけで行う**

文言を変えたら `TOP3_PROMPT_VERSION` を上げ、Step 2 をやり直す。**最終評価期間の結果を見て文言を変えない。**

- [ ] **Step 4: 最終評価を1度だけ実行する**（所有者の承認後）

Run: `.venv/Scripts/python.exe scripts/backtest_top3.py --from <最終開始> --to <最終終了> --tuning-until <調整終了> --methods python,plan_a,plan_b --limit 100 --final`

- [ ] **Step 5: 結果を書き残す**

仕様の末尾に「調整期間の比較」「最終評価」の節を足す。各差について区間が 0 をまたぐなら「差は確認できなかった」と書き、的中数の差だけで優劣を結論しない。「判断材料不足が減った」と「予測が正しくなった」を分けて書く（spec §8）。

`CLAUDE.md` の「プロジェクトの現状」の表に次の行を足し、実行方法を「ビルド / テスト / 実行」節に足す:

| パス | 役割 |
|---|---|
| `src/keiba/features/{course,history,dossier,dossier_markdown}.py` | 3着内系統の予想用資料。API を呼ばない |
| `src/keiba/predict/{top3,top3_questions,top3_baselines}.py` | 案A・案B と Python 基準 |
| `src/keiba/evaluate/{top3_metrics,top3_backtest,coverage}.py` | 3着内の採点・比較・データ充足度 |
| `src/keiba/store/top3_records.py` | 3着内予想の記録（`data/top3/`） |

```bash
.venv/Scripts/python.exe scripts/dossier.py --race-id <race_id>          # 資料（課金なし）
.venv/Scripts/python.exe scripts/backtest_top3.py ... --dry-run          # state の大きさ（課金なし）
.venv/Scripts/python.exe scripts/backtest_top3.py ... --methods python,plan_a  # 比較（課金あり）
```

- [ ] **Step 6: 区切り**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS。コミット許可があれば `git commit -m "docs: 3着内予想の比較結果と使い方を記録"`。

---

## 自己点検の記録

- **spec との対応:** §3.1（目的と質問の不一致）→ Task 10・11。§3.2（コース識別）→ Task 2・3。§3.3（条件の組み合わせ）→ Task 5。§3.4（順番に推論したことにならない）→ Task 15 で Python が受け渡し。§4.1〜4.4（資料）→ Task 3〜7。§5 案A/案B → Task 11・15。§6（正規化しない、合計の診断）→ Task 8 の `prob_sum_minus_actual`・Task 11 のテスト。§7（評価指標・同着等の規則・比較手順1〜6・レース単位と開催日クラスタ）→ Task 1・8・9・13・16。§8（調整に使ったレースを検証データと同等に扱わない）→ Task 13 の期間ガード。§9（旧来の重み・較正を流用しない、記録形式を分ける）→ Global Constraints・Task 12。
- **このプランでやらないこと:** 3着内確率の補正（合計を3に合わせるなど。spec §6「必要な補正は別途検証」）、既存の5軸・学習処理の作り直し（spec §9）、利用者向けの予想 CLI（`scripts/predict.py` 相当）の3着内版——比較の結果を見てから決める。

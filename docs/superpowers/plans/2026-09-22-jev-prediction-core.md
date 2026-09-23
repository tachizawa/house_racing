# 予想コア・第2層・検証基盤 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** JRA-VAN の過去データから前日時点の特徴量を作り、Jev の第1層（レース全体の勝率分布）と第2層（馬ごとの5軸評価）を経て、条件付きロジットで統合した上位 n 頭を出し、過去レースで検証できるようにする。

**Architecture:** `src/jev_client.py`（既存の通信層）は触らず、予想は新設の `src/keiba/` に置く。未来情報の混入は `EntryRow` / `ResultRow` の型分離と `Repository` の `as_of` 必須引数で構造的に防ぐ。第3層はレース内 softmax の条件付きロジットにして、頭数が変わってもスケールが崩れないようにする。第1層のみの `baseline` を常に併走させ、第2層を足して改善しなければ採用しない。

**Tech Stack:** Python 3.13 / pytest / numpy / scipy（最尤推定）/ pandas（CSV 参照）/ scikit-learn（isotonic 較正）/ pywin32（JV-Link、Windows のみ）/ typesafe-sdk

**Spec:** [docs/superpowers/specs/2026-09-22-jev-prediction-core-design.md](../specs/2026-09-22-jev-prediction-core-design.md)

## Global Constraints

- **Python は `.venv` のものを明示して使う。** `.venv/Scripts/python.exe -m pytest`（PowerShell では `.venv\Scripts\python.exe`）。
- **`src/jev_client.py` への変更は Task 13 の1点のみ**（`ask_parallel` に `return_exceptions` を足す）。予想ロジックと重みをここに置かない。
- **`vendor/` を import しない。** オフセットは `src/keiba/ingest/records.py` に写し、由来をコメントに書く。`vendor/` は再配布不可で `.gitignore` 済み。
- **`Repository` の全公開メソッドは `as_of: datetime` をキーワード専用の必須引数に取る。** 既定値を与えない。評価用の参照は別クラス `ResultRepository` に置く。
- **欠損を 0 や不適性に置き換えない。** 値は `None` のままにし、`data_quality.missing` に列挙する。集計値には必ず `starts`（件数）を併記する。
- **CSV は UTF-8（BOM なし）、ID 列は文字列、欠損は空文字列。** pandas 読み込み時は `dtype=str` を明示する。先頭ゼロを数値化しない。
- **`UM`（馬マスタ）の累計集計列（`Ruikei*` / `ChakuKaisu*` / `Kyakusitu` / `RaceCount`）を CSV に書き出さない。** 過去時点の値を復元できないため。
- **テストは API に触らない。** Jev への実アクセスは `scripts/smoke_test.py` と `scripts/predict.py` の手動実行のみ。
- **版文字列の既定値**: `SCHEMA_VERSION = "1"` / `FEATURE_VERSION = "1"` / `INTEGRATION_VERSION = "1"` / `EXCEPTION_RULES_VERSION = "1"` / `PROMPT_VERSION = "1"`。
- **既定値**: `DEFAULT_TOP_N = 3` / `DEFAULT_RECENT_RUNS = 5` / `SAME_DISTANCE_TOLERANCE_M = 200` / `LONG_TRANSPORT_KM = 300` / `PACE_MIN_RACES = 30`。
- **git が未 `init` である。** 各タスクの末尾は「テスト全体を通す」で締める。コミットコマンドは参考として併記してあるので、`git init` 後はそのまま使える。
- **テストフィクスチャに実在の馬名・レース名・血統登録番号を入れない。** JRA-VAN のデータは再配布不可。

---

## ファイル構成

| ファイル | 責務 |
|---|---|
| `src/keiba/config.py` | 設定値とコード表の写像。他モジュールはここから読む |
| `src/keiba/models.py` | `EntryRow` / `ResultRow` / `RaceRow` / `PastRun` / `RaceInput` などの dataclass |
| `src/keiba/ingest/records.py` | `RA` / `SE` / `UM` の固定長 cp932 レコード → dict |
| `src/keiba/ingest/store.py` | 版付きディレクトリと manifest の書き込み・読み込み |
| `src/keiba/ingest/jvlink.py` | JV-Link の COM 呼び出し（Windows 専用） |
| `src/keiba/data/store.py` | CSV を pandas に読み込む `DataStore` |
| `src/keiba/data/repository.py` | `Repository`（as_of 必須）と `ResultRepository`（評価用） |
| `src/keiba/input/schema.py` | 利用者入力の JSON → `RaceInput` |
| `src/keiba/input/validate.py` | 入力検証 |
| `src/keiba/input/match.py` | 馬名正規化と馬マスタ照合 |
| `src/keiba/features/ability.py` | 能力指標 |
| `src/keiba/features/derive.py` | 条件別成績・近況・脚質・輸送 |
| `src/keiba/features/context.py` | `HorseFeatures` / `PredictionContext` の組み立て |
| `src/keiba/features/summary.py` | `state` の dict 化 |
| `src/keiba/predict/questions.py` | 質問定義とプロンプト版 |
| `src/keiba/predict/layers.py` | 第1層・第2層の実行 |
| `src/keiba/predict/integrate.py` | 第3層（baseline / model）と `Weights` |
| `src/keiba/predict/pipeline.py` | 層のオーケストレーションと記録の組み立て |
| `src/keiba/learn/conditional_logit.py` | 多項ロジットの最尤推定 |
| `src/keiba/learn/calibration.py` | isotonic 較正 |
| `src/keiba/evaluate/metrics.py` | 指標 |
| `src/keiba/evaluate/splits.py` | 時系列分割 |
| `src/keiba/evaluate/backtest.py` | バックテストの実行 |
| `src/keiba/store/records.py` | 予測記録の保存と `to_user_view` |

## フェーズ

| フェーズ | タスク | 完了時点で動くもの |
|---|---|---|
| 1. データ層 | 1–6 | JV-Link から CSV を作り、`as_of` で過去走を引ける |
| 2. 予想コア | 7–16 | 1レースの予想が出せる（`baseline`） |
| 3. 学習と統合 | 17–18 | 重みを学習して `model` が出せる |
| 4. 検証 | 19–21 | バックテストで `baseline` と `model` を比較できる |

---

# フェーズ1: データ層

## Task 1: 設定値とコード表

**Files:**
- Create: `src/keiba/__init__.py`
- Create: `src/keiba/config.py`
- Create: `tests/test_config.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: なし
- Produces: `SCHEMA_VERSION` `FEATURE_VERSION` `INTEGRATION_VERSION` `EXCEPTION_RULES_VERSION` `DEFAULT_TOP_N` `DEFAULT_RECENT_RUNS` `SAME_DISTANCE_TOLERANCE_M` `LONG_TRANSPORT_KM` `PACE_MIN_RACES` `DATA_DIR` `JYO_CD_TO_NAME: dict[str, str]` `TOZAI_CD_TO_BASE: dict[str, str]` `SEX_CD_TO_NAME: dict[str, str]` `BABA_CD_TO_NAME: dict[str, str]` `TRANSPORT_KM: dict[tuple[str, str], int]` `IJYO_EXCLUDED: frozenset[str]` `IJYO_NO_TIME: frozenset[str]` `DATA_KUBUN_RESULT_PRIORITY: tuple[str, ...]` `DATA_KUBUN_ENTRY_PRIORITY: tuple[str, ...]` `DATA_KUBUN_DROP: frozenset[str]` `track_kind(track_cd: str) -> str | None` `class_level(grade_cd: str, jyoken_cd_youngest: str) -> int | None`

- [ ] **Step 1: 依存を追加する**

`requirements.txt` の `# 開発用` の直前に追記する。

```
# 予想コア
numpy>=1.26
scipy>=1.11
pandas>=2.1
scikit-learn>=1.4
```

インストール: `.venv/Scripts/python.exe -m pip install -r requirements.txt`

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_config.py`:

```python
"""config のコード表。値は JV-Data仕様書_4.9.0.1.xlsx のコード表シート由来。"""

import pytest

from keiba import config


class TestTrackKind:
    @pytest.mark.parametrize(
        ("track_cd", "expected"),
        [
            ("10", "turf"),   # 平地 芝 直線
            ("11", "turf"),   # 平地 芝 左回り
            ("22", "turf"),   # 平地 芝 右回り 外2周
            ("23", "dirt"),   # 平地 ダート 左回り
            ("27", "dirt"),   # 平地 サンド 左回り
            ("29", "dirt"),   # 平地 ダート 直線
            ("51", "jump"),   # 障害 芝 襷
            ("59", "jump"),   # 障害 芝 外2周以上
            ("00", None),     # 未設定
            ("", None),
            ("99", None),
        ],
    )
    def test_maps_track_code(self, track_cd, expected):
        assert config.track_kind(track_cd) == expected


class TestClassLevel:
    @pytest.mark.parametrize(
        ("grade_cd", "jyoken", "expected"),
        [
            ("A", "999", 8),    # G1
            ("B", "999", 7),    # G2
            ("C", "999", 6),    # G3
            ("L", "999", 5),    # リステッド
            ("D", "999", 5),    # グレードのない重賞
            (" ", "701", 0),    # 新馬
            (" ", "702", 0),    # 未出走
            (" ", "703", 1),    # 未勝利
            (" ", "005", 2),    # 1勝クラス
            (" ", "010", 3),    # 2勝クラス
            (" ", "016", 4),    # 3勝クラス
            (" ", "999", 5),    # オープン
            ("E", "016", 4),    # 重賞以外の特別競走はクラスで決まる
            (" ", "001", 2),    # 旧100万下は1勝クラス相当
            (" ", "100", 5),    # 1600万超の条件は OP 相当
            (" ", "000", None), # 未設定は 0 にしない
            (" ", "", None),
        ],
    )
    def test_maps_class_level(self, grade_cd, jyoken, expected):
        assert config.class_level(grade_cd, jyoken) == expected


class TestCodeTables:
    def test_jyo_cd_covers_central_courses(self):
        assert config.JYO_CD_TO_NAME["01"] == "札幌"
        assert config.JYO_CD_TO_NAME["05"] == "東京"
        assert config.JYO_CD_TO_NAME["06"] == "中山"
        assert config.JYO_CD_TO_NAME["10"] == "小倉"
        assert len(config.JYO_CD_TO_NAME) == 10

    def test_tozai_cd_maps_to_training_centres(self):
        assert config.TOZAI_CD_TO_BASE["1"] == "美浦"
        assert config.TOZAI_CD_TO_BASE["2"] == "栗東"
        assert "0" not in config.TOZAI_CD_TO_BASE
        assert "3" not in config.TOZAI_CD_TO_BASE

    def test_transport_km_defined_for_every_base_and_course(self):
        for base in ("美浦", "栗東"):
            for jyo_cd in config.JYO_CD_TO_NAME:
                assert (base, jyo_cd) in config.TRANSPORT_KM

    def test_ijyo_groups_are_disjoint_and_keep_demotion_normal(self):
        # 6 落馬再騎乗 / 7 降着 は確定着順があるので通常扱い
        assert config.IJYO_EXCLUDED == frozenset({"1", "2", "3"})
        assert config.IJYO_NO_TIME == frozenset({"4", "5"})
        assert not (config.IJYO_EXCLUDED & config.IJYO_NO_TIME)
        assert "7" not in config.IJYO_EXCLUDED
        assert "7" not in config.IJYO_NO_TIME

    def test_data_kubun_priorities(self):
        assert config.DATA_KUBUN_RESULT_PRIORITY == ("7", "6", "5", "4", "3")
        assert config.DATA_KUBUN_ENTRY_PRIORITY == ("2", "1")
        assert config.DATA_KUBUN_DROP == frozenset({"0", "9", "A", "B"})
```

- [ ] **Step 3: テストが落ちることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keiba'`

- [ ] **Step 4: 実装する**

`src/keiba/__init__.py`:

```python
"""JRA-VAN のデータと Jev を使った競馬予想。

通信層は src/jev_client.py にある。このパッケージには予想の判断だけを置く。
"""
```

`src/keiba/config.py`:

```python
"""設定値とコード表。

コード表の値は vendor/jravan-sdk-500-64bit/ドキュメント/JV-Data仕様書_4.9.0.1.xlsx
の「コード表」シートから写したもの。vendor/ は再配布不可なので import せず、
必要な値だけをここに持つ。括弧内の番号は仕様書のコード表番号。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# --- 版。いずれかが変わった予想は別の記録として扱う -------------------------
SCHEMA_VERSION = "1"
FEATURE_VERSION = "1"
INTEGRATION_VERSION = "1"
EXCEPTION_RULES_VERSION = "1"

# --- 既定値 -----------------------------------------------------------------
DEFAULT_TOP_N = 3
DEFAULT_RECENT_RUNS = 5          # 直近何走を集計に使うか。仕様の初期案は 5〜10
SAME_DISTANCE_TOLERANCE_M = 200  # 「同距離」とみなす幅
LONG_TRANSPORT_KM = 300          # 「長距離輸送」とみなす閾値
PACE_MIN_RACES = 30              # ペース三分位を層別に計算するのに必要な最小レース数

# --- 2001. 競馬場コード（中央のみ。30番台以降は地方、A0 以降は海外） --------
JYO_CD_TO_NAME: dict[str, str] = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

# --- 2301. 東西所属コード ---------------------------------------------------
# 0:その他 1:関東(美浦) 2:関西(栗東) 3:地方招待 4:外国招待
# 輸送の起点になるのはトレセン所属の 1 と 2 だけ。
TOZAI_CD_TO_BASE: dict[str, str] = {"1": "美浦", "2": "栗東"}

# --- 2202. 性別コード -------------------------------------------------------
SEX_CD_TO_NAME: dict[str, str] = {"1": "牡", "2": "牝", "3": "セン"}

# --- 2010. 馬場状態コード ---------------------------------------------------
BABA_CD_TO_NAME: dict[str, str] = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}

# --- 2101. 異常区分コード ---------------------------------------------------
# 0:正常 1:出走取消 2:発走除外 3:競走除外 4:競走中止 5:失格 6:落馬再騎乗 7:降着
#
# 1/2/3 は走っていないので出走集合から抜く。
# 4/5 は走ったが着順がないので集合には残し、勝ち馬になり得ないものとして扱う。
# 6/7 は確定着順があるので通常扱いにする（ここで除外すると降着馬の着順を捨てる）。
IJYO_EXCLUDED = frozenset({"1", "2", "3"})
IJYO_NO_TIME = frozenset({"4", "5"})

# --- データ区分（レコードヘッダ）-------------------------------------------
# 1:出走馬名表(木) 2:出馬表(金土) 3-6:速報成績 7:成績(月) 9:中止 0:削除
# A:地方競馬 B:海外国際レース
DATA_KUBUN_RESULT_PRIORITY: tuple[str, ...] = ("7", "6", "5", "4", "3")
DATA_KUBUN_ENTRY_PRIORITY: tuple[str, ...] = ("2", "1")
DATA_KUBUN_DROP = frozenset({"0", "9", "A", "B"})

# --- 輸送距離（推定値）------------------------------------------------------
# トレセンから各競馬場までの代表的な陸路距離の概数。単位 km。
# 実際の出発地・経路・輸送日を示すものではない。state に渡すときも推定値と明示する。
TRANSPORT_KM: dict[tuple[str, str], int] = {
    ("美浦", "01"): 1100, ("栗東", "01"): 1500,   # 札幌
    ("美浦", "02"): 950,  ("栗東", "02"): 1350,   # 函館
    ("美浦", "03"): 250,  ("栗東", "03"): 650,    # 福島
    ("美浦", "04"): 330,  ("栗東", "04"): 550,    # 新潟
    ("美浦", "05"): 110,  ("栗東", "05"): 480,    # 東京
    ("美浦", "06"): 90,   ("栗東", "06"): 500,    # 中山
    ("美浦", "07"): 400,  ("栗東", "07"): 130,    # 中京
    ("美浦", "08"): 500,  ("栗東", "08"): 20,     # 京都
    ("美浦", "09"): 540,  ("栗東", "09"): 60,     # 阪神
    ("美浦", "10"): 1100, ("栗東", "10"): 620,    # 小倉
}


def track_kind(track_cd: str) -> str | None:
    """2009. トラックコード → turf / dirt / jump。

    10-22: 平地 芝 / 23-29: 平地 ダート・サンド / 51-59: 障害
    """
    if not track_cd.strip().isdigit():
        return None
    value = int(track_cd)
    if 10 <= value <= 22:
        return "turf"
    if 23 <= value <= 29:
        return "dirt"
    if 51 <= value <= 59:
        return "jump"
    return None


# 2003. グレードコード。D はグレードのない重賞なので OP 相当の 5。
# E（重賞以外の特別競走）と空白（一般競走）はクラスを表さないので競走条件へ送る。
_GRADE_TO_CLASS_LEVEL: dict[str, int] = {"A": 8, "B": 7, "C": 6, "L": 5, "D": 5}


def class_level(grade_cd: str, jyoken_cd_youngest: str) -> int | None:
    """クラスを 0-8 の順序値にする。決まらなければ None（0 にしない）。

    ``jyoken_cd_youngest`` は RA の競走条件コード5要素のうち「最若年条件」
    （出走可能な最も馬齢が若い馬に対する条件）。レースのクラスを代表する。

    0 新馬・未出走 / 1 未勝利 / 2 1勝 / 3 2勝 / 4 3勝 / 5 OP・L・重賞
    6 G3 / 7 G2 / 8 G1
    """
    if grade_cd in _GRADE_TO_CLASS_LEVEL:
        return _GRADE_TO_CLASS_LEVEL[grade_cd]

    code = (jyoken_cd_youngest or "").strip()
    if code in ("701", "702"):
        return 0
    if code == "703":
        return 1
    if code == "999":
        return 5
    if code.isdigit() and code != "000":
        # 001-100 は収得賞金による条件。005/010/016 が 1勝/2勝/3勝クラス。
        value = int(code)
        if value <= 5:
            return 2
        if value <= 10:
            return 3
        if value <= 16:
            return 4
        return 5
    return None
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: PASS（30件以上）

- [ ] **Step 6: テスト全体を通す**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 既存の16件と合わせて全件 PASS

参考（`git init` 後に使えるコミット）:

```bash
git add src/keiba/__init__.py src/keiba/config.py tests/test_config.py requirements.txt
git commit -m "feat(keiba): add config with JV-Data code tables"
```

---

## Task 2: データモデル（EntryRow と ResultRow の分離）

**Files:**
- Create: `src/keiba/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `keiba.config`
- Produces: `EntryRow` `ResultRow` `RaceRow` `PastRun` `HorseMaster` `HorseEntry` `RaceInput`。`RaceRow.is_flat` プロパティ、`ResultRow.counts_as_starter` プロパティ、`ResultRow.can_win` プロパティ、`PastRun.margin_sec` プロパティ。

**欠損を表す専用の型は作らない。** 値は `None` のままにし、欠けている項目は `data_quality.missing` にパスの一覧として持つ（spec §6.7）。センチネル型を作ると「欠損」の表し方が2通りになり、どちらを見るべきか決まらなくなる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_models.py`:

```python
"""EntryRow と ResultRow の分離が未来情報の混入を防ぐことを確認する。"""

from datetime import date, datetime

import pytest

from keiba.models import (
    EntryRow,
    HorseEntry,
    PastRun,
    RaceInput,
    RaceRow,
    ResultRow,
)


def make_entry(umaban: str = "1", **kw) -> EntryRow:
    base = dict(
        umaban=umaban,
        wakuban="1",
        ketto_num="0000000001",
        bamei="テスト馬",
        sex_cd="1",
        barei="4",
        futan_kg=57.0,
        kisyu_code="00001",
        tozai_cd="1",
        chokyosi_code="00001",
    )
    base.update(kw)
    return EntryRow(**base)


def make_race(**kw) -> RaceRow:
    base = dict(
        race_id="2026032106050311",
        hasso_at=datetime(2026, 3, 21, 15, 25),
        jyo_cd="06",
        kyori_m=2000,
        track_cd="17",
        track_kind="turf",
        baba_cd="1",
        class_level=4,
        grade_cd=" ",
        jyoken_cd_youngest="016",
        toroku_tosu=12,
        syusso_tosu=12,
        nyusen_tosu=12,
        first_3f_sec=35.2,
        race_name="テスト賞",
    )
    base.update(kw)
    return RaceRow(**base)


def make_result(**kw) -> ResultRow:
    base = dict(
        entry=make_entry(),
        ijyo_cd="0",
        kakutei_jyuni=1,
        dochaku_kubun="0",
        dochaku_tosu=None,
        time_sec=120.5,
        margin_sec=0.0,
        jyuni_1c=3,
        jyuni_2c=3,
        jyuni_3c=2,
        jyuni_4c=2,
        last_3f_sec=34.1,
        ba_taijyu_kg=486,
        zogen_sa_kg=-2,
        odds=3.4,
        ninki=1,
    )
    base.update(kw)
    return ResultRow(**base)


class TestEntryRow:
    def test_has_no_result_fields(self):
        """出馬表相当の型に結果の項目が生えていないこと。"""
        entry = make_entry()
        for forbidden in (
            "kakutei_jyuni", "time_sec", "margin_sec", "odds", "ninki",
            "jyuni_4c", "last_3f_sec", "ba_taijyu_kg", "ijyo_cd",
        ):
            assert not hasattr(entry, forbidden), forbidden

    def test_is_frozen(self):
        entry = make_entry()
        with pytest.raises(Exception):
            entry.umaban = "2"


class TestResultRow:
    @pytest.mark.parametrize(
        ("ijyo_cd", "starter", "can_win"),
        [
            ("0", True, True),    # 正常
            ("6", True, True),    # 落馬再騎乗
            ("7", True, True),    # 降着。確定着順がある
            ("1", False, False),  # 出走取消
            ("2", False, False),  # 発走除外
            ("3", False, False),  # 競走除外
            ("4", True, False),   # 競走中止。集合に残すが勝ち馬になり得ない
            ("5", True, False),   # 失格
        ],
    )
    def test_starter_and_winner_eligibility(self, ijyo_cd, starter, can_win):
        result = make_result(ijyo_cd=ijyo_cd)
        assert result.counts_as_starter is starter
        assert result.can_win is can_win


class TestRaceRow:
    def test_is_flat_excludes_jump(self):
        assert make_race(track_kind="turf").is_flat is True
        assert make_race(track_kind="dirt").is_flat is True
        assert make_race(track_kind="jump").is_flat is False
        assert make_race(track_kind=None).is_flat is False


class TestPastRun:
    def test_margin_sec_is_none_when_no_time(self):
        run = PastRun(race=make_race(), result=make_result(ijyo_cd="4", margin_sec=None))
        assert run.margin_sec is None

    def test_margin_sec_passes_through_for_normal_run(self):
        run = PastRun(race=make_race(), result=make_result(margin_sec=0.3))
        assert run.margin_sec == 0.3


class TestRaceInput:
    def test_defaults(self):
        race_input = RaceInput(
            date=date(2026, 3, 21),
            jyo="中山",
            race_num=11,
            track_kind="turf",
            kyori_m=2000,
            entry_count=2,
            prediction_cutoff=datetime(2026, 3, 20, 18, 0),
            horses=(HorseEntry(umaban="1", bamei="テスト馬"),
                    HorseEntry(umaban="2", bamei="サンプル馬")),
        )
        assert race_input.top_n == 3
        assert race_input.baba is None
        assert race_input.class_name is None
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keiba.models'`

- [ ] **Step 3: 実装する**

`src/keiba/models.py`:

```python
"""予想パイプラインが扱う型。

もっとも重要なのは EntryRow と ResultRow の分離。
予想側（features / predict）は EntryRow しか受け取れない。ResultRow を触るのは
評価側（evaluate）と、過去走を組み立てる data/repository.py だけ。
この分離が、対象レースの結果が state に混ざることを型で防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from keiba import config


@dataclass(frozen=True)
class EntryRow:
    """出馬表相当。予想パイプラインが受け取れるのはこれだけ。

    futan_before_kg / kisyu_code_before は変更があった場合のみ設定される。
    成績レコードから EntryRow を作ったとき、これらが入っていれば
    「前日時点の値と異なる可能性がある」の印になる。
    """

    umaban: str
    wakuban: str
    ketto_num: str
    bamei: str
    sex_cd: str
    barei: str
    futan_kg: float | None
    kisyu_code: str
    tozai_cd: str
    chokyosi_code: str
    futan_before_kg: float | None = None
    kisyu_code_before: str = ""

    @property
    def changed_after_entry(self) -> bool:
        """斤量か騎手が変更されているか。"""
        return self.futan_before_kg is not None or bool(self.kisyu_code_before.strip())

    @property
    def sex_age(self) -> str:
        return f"{config.SEX_CD_TO_NAME.get(self.sex_cd, '?')}{self.barei.lstrip('0') or '0'}"

    @property
    def base(self) -> str | None:
        """所属トレセン。美浦 / 栗東 / None（地方・外国招待など）。"""
        return config.TOZAI_CD_TO_BASE.get(self.tozai_cd)


@dataclass(frozen=True)
class ResultRow:
    """結果込み。評価側と過去走の組み立てだけが触る。"""

    entry: EntryRow
    ijyo_cd: str
    kakutei_jyuni: int | None
    dochaku_kubun: str
    dochaku_tosu: int | None
    time_sec: float | None
    margin_sec: float | None
    jyuni_1c: int | None
    jyuni_2c: int | None
    jyuni_3c: int | None
    jyuni_4c: int | None
    last_3f_sec: float | None
    ba_taijyu_kg: int | None
    zogen_sa_kg: int | None
    odds: float | None
    ninki: int | None

    @property
    def counts_as_starter(self) -> bool:
        """出走集合に含めるか。取消・発走除外・競走除外は走っていないので除く。"""
        return self.ijyo_cd not in config.IJYO_EXCLUDED

    @property
    def can_win(self) -> bool:
        """勝ち馬になり得るか。競走中止・失格は集合に残すが勝ち馬ではない。"""
        return self.counts_as_starter and self.ijyo_cd not in config.IJYO_NO_TIME

    @property
    def is_winner(self) -> bool:
        return self.can_win and self.kakutei_jyuni == 1


@dataclass(frozen=True)
class RaceRow:
    race_id: str
    hasso_at: datetime
    jyo_cd: str
    kyori_m: int
    track_cd: str
    track_kind: str | None
    baba_cd: str
    class_level: int | None
    grade_cd: str
    jyoken_cd_youngest: str
    toroku_tosu: int | None
    syusso_tosu: int | None
    nyusen_tosu: int | None
    first_3f_sec: float | None
    race_name: str

    @property
    def is_flat(self) -> bool:
        """平地か。初期版は障害を対象外にする。"""
        return self.track_kind in ("turf", "dirt")

    @property
    def jyo(self) -> str:
        return config.JYO_CD_TO_NAME.get(self.jyo_cd, self.jyo_cd)

    @property
    def baba(self) -> str | None:
        return config.BABA_CD_TO_NAME.get(self.baba_cd)


@dataclass(frozen=True)
class PastRun:
    """過去走1件。レース条件と、その馬のその走の結果。"""

    race: RaceRow
    result: ResultRow

    @property
    def margin_sec(self) -> float | None:
        return self.result.margin_sec

    @property
    def has_time(self) -> bool:
        return self.result.ijyo_cd not in config.IJYO_NO_TIME and self.margin_sec is not None


@dataclass(frozen=True)
class HorseMaster:
    """馬マスタ（UM）のうち時点に依存しない項目だけ。

    UM の累計成績（Ruikei* / ChakuKaisu* / Kyakusitu）は「現在までの累計」で
    過去時点の値を復元できないため、この型には含めない。
    """

    ketto_num: str
    bamei: str
    bamei_kana: str
    birth_date: date | None
    sex_cd: str


@dataclass(frozen=True)
class HorseEntry:
    """利用者が入力した1頭分。"""

    umaban: str
    bamei: str
    wakuban: str | None = None
    sex_age: str | None = None
    futan_kg: float | None = None
    kisyu: str | None = None


@dataclass(frozen=True)
class RaceInput:
    """利用者が入力したレース1件。

    entry_count は「出走予定頭数」。RaceRow.syusso_tosu（結果から算出される
    出走頭数）とは別物なので名前を分けている。
    """

    date: date
    jyo: str                     # 表示名。「中山」など
    race_num: int
    track_kind: str              # "turf" / "dirt"
    kyori_m: int
    entry_count: int
    prediction_cutoff: datetime
    horses: tuple[HorseEntry, ...]
    # コードに解決した値。過去走（jyo_cd / baba_cd を持つ）との突き合わせに使う。
    # 表示名のままでは条件別成績を集計できないので、入力を読む時点で解決する。
    jyo_cd: str | None = None
    baba_cd: str | None = None
    class_name: str | None = None
    baba: str | None = None      # 表示名。「良」など
    hasso_at: datetime | None = None
    top_n: int = config.DEFAULT_TOP_N
    race_id: str | None = None
```

`jyo_cd` と `baba_cd` は `input/schema.py`（Task 7）が表示名から解決して埋める。`RaceInput` を手で組むテストでは明示的に渡す。

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: テスト全体を通す**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全件 PASS

```bash
git add src/keiba/models.py tests/test_models.py
git commit -m "feat(keiba): split EntryRow and ResultRow to block leakage by type"
```

---

## Task 3: レコード分解（RA / SE / UM）

**Files:**
- Create: `src/keiba/ingest/__init__.py`, `src/keiba/ingest/records.py`
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/records.py`
- Create: `tests/test_records.py`

**Interfaces:**
- Consumes: `keiba.config.track_kind`, `keiba.config.class_level`
- Produces: `record_spec(raw: bytes) -> str` / `parse_ra(raw: bytes) -> dict` / `parse_se(raw: bytes) -> dict` / `parse_um(raw: bytes) -> dict` / `parse(raw: bytes) -> tuple[str, dict] | None` / `margin_sec(time_diff: str) -> float | None` / `race_time_sec(value: str) -> float | None` / `tenth_sec(value: str) -> float | None` / `to_int(value: str) -> int | None` / `RA_LENGTH = 1272` / `SE_LENGTH = 555`

オフセットは `vendor/.../JVData_Struct.py` の `SetDataB` と `ドキュメント/JV-Data仕様書_4.9.0.1.xlsx` の「フォーマット」シートから写す。**位置は 1 始まり**なので Python のスライスにするときは `-1` する。`vendor/` は import しない。

確定済みの位置（このタスクで使うものだけ）:

| レコード | 項目 | 位置 / 長さ |
|---|---|---|
| 共通 | レコード種別 / データ区分 / データ作成年月日 | 1/2, 3/1, 4/8 |
| 共通 | RACE_ID（年4・月日4・場2・回2・日目2・R2） | 12/4, 16/4, 20/2, 22/2, 24/2, 26/2 |
| RA | 競走名本題 | 33/60 |
| RA | グレードコード | 615/1 |
| RA | 競走種別 / 記号 / 重量種別 | 617/2, 619/3, 622/1 |
| RA | 競走条件コード5要素（最若年は5番目） | 623/3, 626/3, 629/3, 632/3, **635/3** |
| RA | 距離 / トラック / コース区分 | 698/4, 706/2, 710/2 |
| RA | 発走時刻 / 登録頭数 / 出走頭数 / 入線頭数 | 874/4, 882/2, 884/2, 886/2 |
| RA | 天候 / 芝馬場 / ダート馬場 | 888/1, 889/1, 890/1 |
| RA | ラップタイム（25繰返×3） | 891/3 × 25 |
| RA | **前3ハロン** / 前4 / 後3 / 後4 | **970/3**, 973/3, 976/3, 979/3 |
| SE | 枠番 / 馬番 / 血統登録番号 / 馬名 | 28/1, 29/2, 31/10, 41/36 |
| SE | 馬記号 / 性別 / 馬齢 / 東西所属 / 調教師 | 77/2, 79/1, 83/2, 85/1, 86/5 |
| SE | 負担重量 / 変更前 / ブリンカー | 289/3, 292/3, 295/1 |
| SE | 騎手 / 変更前騎手 | 297/5, 302/5 |
| SE | 馬体重 / 増減符号 / 増減差 | 325/3, 328/1, 329/3 |
| SE | 異常区分 / 確定着順 / 同着区分 / 同着頭数 | 332/1, 335/2, 337/1, 338/1 |
| SE | 走破タイム | 339/4 |
| SE | 1〜4コーナー順位 | 352/2, 354/2, 356/2, 358/2 |
| SE | 単勝オッズ / 人気 | 360/4, 364/2 |
| SE | 後4ハロン / 後3ハロン | 388/3, 391/3 |
| SE | **タイム差** | **532/4** |
| UM | 血統登録番号 / 抹消区分 / 登録日 / 抹消日 / 生年月日 | 12/10, 22/1, 23/8, 31/8, 39/8 |
| UM | 馬名 / 馬名カナ / 性別 / 品種 / 毛色 | 47/36, 83/36, 201/1, 202/1, 203/2 |

- [ ] **Step 1: フィクスチャを作る**

`tests/fixtures/__init__.py` は空ファイル。`tests/fixtures/records.py` に `make_ra()` / `make_se()` / `make_um()` を書く。実在の馬名・レース名・血統登録番号は入れない（JRA-VAN のデータは再配布不可）。

```python
"""合成レコード。バイト列の構造とオフセットだけを再現する。位置は 1 始まり。"""

from __future__ import annotations


def _blank(length: int) -> bytearray:
    return bytearray(b" " * length)


def _put(buf: bytearray, pos: int, value: str, length: int) -> None:
    encoded = value.encode("cp932")
    if len(encoded) > length:
        raise ValueError(f"{value!r} は {length} バイトに収まらない")
    buf[pos - 1 : pos - 1 + length] = encoded + b" " * (length - len(encoded))


def make_ra(*, data_kubun="7", year="2026", monthday="0321", jyo_cd="06",
            kaiji="01", nichiji="05", race_num="11", hondai="テスト特別",
            grade_cd=" ", jyoken_youngest="016", kyori="2000", track_cd="17",
            hasso_time="1525", toroku_tosu="12", syusso_tosu="12",
            nyusen_tosu="12", siba_baba="1", dirt_baba="0",
            first_3f="352", last_3f="341") -> bytes:
    buf = _blank(1272)
    _put(buf, 1, "RA", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, year, 4); _put(buf, 16, monthday, 4); _put(buf, 20, jyo_cd, 2)
    _put(buf, 22, kaiji, 2); _put(buf, 24, nichiji, 2); _put(buf, 26, race_num, 2)
    _put(buf, 33, hondai, 60)
    _put(buf, 615, grade_cd, 1)
    _put(buf, 617, "13", 2); _put(buf, 622, "4", 1)
    for i in range(5):                       # 5要素すべて同じ値を入れておく
        _put(buf, 623 + 3 * i, jyoken_youngest, 3)
    _put(buf, 698, kyori, 4); _put(buf, 706, track_cd, 2); _put(buf, 710, "A ", 2)
    _put(buf, 874, hasso_time, 4)
    _put(buf, 882, toroku_tosu, 2); _put(buf, 884, syusso_tosu, 2)
    _put(buf, 886, nyusen_tosu, 2)
    _put(buf, 888, "1", 1); _put(buf, 889, siba_baba, 1); _put(buf, 890, dirt_baba, 1)
    for i in range(10):                      # ラップ 12.3秒 × 10。合計は 36.9
        _put(buf, 891 + 3 * i, "123", 3)
    _put(buf, 970, first_3f, 3); _put(buf, 976, last_3f, 3)
    buf[1270] = ord("\r"); buf[1271] = ord("\n")
    return bytes(buf)


def make_se(*, data_kubun="7", year="2026", monthday="0321", jyo_cd="06",
            kaiji="01", nichiji="05", race_num="11", wakuban="1", umaban="01",
            ketto_num="0000000001", bamei="テストウマ", sex_cd="1", barei="04",
            tozai_cd="1", chokyosi_code="00001", futan="570", futan_before="   ",
            kisyu_code="00001", kisyu_code_before="     ", ba_taijyu="486",
            zogen_fugo="-", zogen_sa="002", ijyo_cd="0", kakutei_jyuni="01",
            dochaku_kubun="0", dochaku_tosu="0", time="1234", jyuni_1c="03",
            jyuni_2c="03", jyuni_3c="02", jyuni_4c="02", odds="0034", ninki="01",
            haron_l3="341", time_diff="-003") -> bytes:
    buf = _blank(555)
    _put(buf, 1, "SE", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, year, 4); _put(buf, 16, monthday, 4); _put(buf, 20, jyo_cd, 2)
    _put(buf, 22, kaiji, 2); _put(buf, 24, nichiji, 2); _put(buf, 26, race_num, 2)
    _put(buf, 28, wakuban, 1); _put(buf, 29, umaban, 2); _put(buf, 31, ketto_num, 10)
    _put(buf, 41, bamei, 36); _put(buf, 79, sex_cd, 1); _put(buf, 83, barei, 2)
    _put(buf, 85, tozai_cd, 1); _put(buf, 86, chokyosi_code, 5)
    _put(buf, 289, futan, 3); _put(buf, 292, futan_before, 3)
    _put(buf, 297, kisyu_code, 5); _put(buf, 302, kisyu_code_before, 5)
    _put(buf, 325, ba_taijyu, 3); _put(buf, 328, zogen_fugo, 1)
    _put(buf, 329, zogen_sa, 3); _put(buf, 332, ijyo_cd, 1)
    _put(buf, 335, kakutei_jyuni, 2); _put(buf, 337, dochaku_kubun, 1)
    _put(buf, 338, dochaku_tosu, 1); _put(buf, 339, time, 4)
    _put(buf, 352, jyuni_1c, 2); _put(buf, 354, jyuni_2c, 2)
    _put(buf, 356, jyuni_3c, 2); _put(buf, 358, jyuni_4c, 2)
    _put(buf, 360, odds, 4); _put(buf, 364, ninki, 2)
    _put(buf, 391, haron_l3, 3); _put(buf, 532, time_diff, 4)
    buf[553] = ord("\r"); buf[554] = ord("\n")
    return bytes(buf)


def make_um(*, data_kubun="0", ketto_num="0000000001", del_kubun="0",
            birth_date="20220315", bamei="テストウマ", bamei_kana="ﾃｽﾄｳﾏ",
            sex_cd="1") -> bytes:
    buf = _blank(1000)
    _put(buf, 1, "UM", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, ketto_num, 10); _put(buf, 22, del_kubun, 1)
    _put(buf, 23, "20240401", 8); _put(buf, 39, birth_date, 8)
    _put(buf, 47, bamei, 36); _put(buf, 83, bamei_kana, 36); _put(buf, 201, sex_cd, 1)
    return bytes(buf)
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_records.py`。TimeDiff の解釈がこのプロジェクトで最も壊れやすいので厚く書く。

```python
"""固定長 cp932 レコードの分解。

仕様書のフォーマットシート: TimeDiff は「符号(+または-)+99秒9 符号は1着:-、
2着以下:+」。1着馬の欄には「2着馬とのタイム差」が入るので margin は 0.0。
"""

import pytest

from keiba.ingest import records
from tests.fixtures import records as fx


def test_record_spec_reads_first_two_bytes():
    assert records.record_spec(fx.make_ra()) == "RA"
    assert records.record_spec(fx.make_se()) == "SE"
    assert records.record_spec(fx.make_um()) == "UM"


@pytest.mark.parametrize(
    ("time_diff", "expected"),
    [("-003", 0.0), ("-012", 0.0),        # 自身が1着。値は2着馬との差なので捨てる
     ("+003", 0.3), ("+000", 0.0), ("+125", 12.5),
     ("9999", None), ("    ", None), ("", None), ("+abc", None), ("?003", None)],
)
def test_margin_sec(time_diff, expected):
    assert records.margin_sec(time_diff) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1234", 83.4), ("0598", 59.8), ("2015", 121.5),
     ("0000", None), ("    ", None), ("9999", None)],
)
def test_race_time_sec(value, expected):
    assert records.race_time_sec(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("341", 34.1), ("352", 35.2), ("999", None), ("000", None), ("   ", None)],
)
def test_tenth_sec(value, expected):
    assert records.tenth_sec(value) == expected


class TestParseRa:
    def test_identity_and_conditions(self):
        row = records.parse_ra(fx.make_ra())
        assert row["race_id"] == "2026032106010511"
        assert row["jyo_cd"] == "06"
        assert row["race_num"] == "11"
        assert row["data_kubun"] == "7"
        assert row["race_name"] == "テスト特別"
        assert row["kyori_m"] == 2000
        assert row["track_kind"] == "turf"
        assert row["jyoken_cd_youngest"] == "016"
        assert row["class_level"] == 4
        assert row["toroku_tosu"] == 12
        assert row["hasso_at"] == "2026-03-21T15:25:00+09:00"

    def test_going_follows_track_kind(self):
        turf = records.parse_ra(fx.make_ra(track_cd="17", siba_baba="3", dirt_baba="0"))
        dirt = records.parse_ra(fx.make_ra(track_cd="23", siba_baba="0", dirt_baba="4"))
        assert turf["baba_cd"] == "3"
        assert dirt["track_kind"] == "dirt" and dirt["baba_cd"] == "4"

    def test_first_3f_comes_from_haron_s3_not_lap_sum(self):
        # ラップは 12.3 × 10 = 36.9 になるが、前3ハロン欄の 35.2 を使うこと
        assert records.parse_ra(fx.make_ra(first_3f="352"))["first_3f_sec"] == 35.2

    def test_grade_overrides_condition_code(self):
        row = records.parse_ra(fx.make_ra(grade_cd="A", jyoken_youngest="999"))
        assert row["class_level"] == 8


class TestParseSe:
    def test_entry_fields(self):
        row = records.parse_se(fx.make_se())
        assert row["race_id"] == "2026032106010511"
        assert row["umaban"] == "01"
        assert row["ketto_num"] == "0000000001"
        assert row["bamei"] == "テストウマ"
        assert row["tozai_cd"] == "1"
        assert row["futan_kg"] == 57.0
        assert row["kisyu_code"] == "00001"

    def test_change_fields_none_when_unset(self):
        row = records.parse_se(fx.make_se())
        assert row["futan_before_kg"] is None
        assert row["kisyu_code_before"] == ""

    def test_change_fields_kept_when_set(self):
        row = records.parse_se(fx.make_se(futan_before="565", kisyu_code_before="00002"))
        assert row["futan_before_kg"] == 56.5
        assert row["kisyu_code_before"] == "00002"

    def test_result_fields_are_prefixed(self):
        row = records.parse_se(fx.make_se())
        assert row["result_kakutei_jyuni"] == 1
        assert row["result_time_sec"] == 83.4
        assert row["result_margin_sec"] == 0.0        # 1着
        assert row["result_jyuni_4c"] == 2
        assert row["result_last_3f_sec"] == 34.1
        assert row["result_odds"] == 3.4
        assert row["result_ijyo_cd"] == "0"

    def test_weight_change_uses_sign_byte(self):
        minus = records.parse_se(fx.make_se(zogen_fugo="-", zogen_sa="014"))
        plus = records.parse_se(fx.make_se(zogen_fugo="+", zogen_sa="006"))
        assert minus["result_zogen_sa_kg"] == -14
        assert plus["result_zogen_sa_kg"] == 6

    @pytest.mark.parametrize("zogen_sa", ["999", "   "])
    def test_weight_change_none_when_unmeasurable(self, zogen_sa):
        assert records.parse_se(fx.make_se(zogen_sa=zogen_sa))["result_zogen_sa_kg"] is None

    def test_weight_change_zero_kept(self):
        row = records.parse_se(fx.make_se(zogen_fugo=" ", zogen_sa="000"))
        assert row["result_zogen_sa_kg"] == 0

    def test_scratched_horse_has_no_result_values(self):
        row = records.parse_se(
            fx.make_se(ijyo_cd="1", kakutei_jyuni="00", time="0000", time_diff="9999"))
        assert row["result_ijyo_cd"] == "1"
        assert row["result_kakutei_jyuni"] is None
        assert row["result_time_sec"] is None
        assert row["result_margin_sec"] is None


class TestParseUm:
    def test_time_independent_fields(self):
        row = records.parse_um(fx.make_um())
        assert row["ketto_num"] == "0000000001"
        assert row["bamei"] == "テストウマ"
        assert row["birth_date"] == "2022-03-15"
        assert row["sex_cd"] == "1"

    def test_no_cumulative_aggregates(self):
        """UM の累計成績は過去時点を復元できないので dict に載せない。"""
        row = records.parse_um(fx.make_um())
        for forbidden in ("ruikei", "chaku_kaisu", "kyakusitu", "race_count",
                          "chaku_sogo", "chaku_chuo"):
            assert not any(forbidden in key for key in row), forbidden


def test_parse_dispatches_and_ignores_other_specs():
    spec, row = records.parse(fx.make_se())
    assert spec == "SE" and row["umaban"] == "01"
    raw = bytearray(fx.make_se())
    raw[0:2] = b"O1"
    assert records.parse(bytes(raw)) is None
```

- [ ] **Step 3: テストが落ちることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keiba.ingest'`

- [ ] **Step 4: 実装する**

`src/keiba/ingest/__init__.py` は空ファイル。`src/keiba/ingest/records.py`:

```python
"""JV-Data の固定長 cp932 レコードを dict にする。

位置と長さは JVData_Struct.py の SetDataB と JV-Data仕様書のフォーマットシート
から写した。vendor/ は再配布不可なので import しない。位置は 1 始まり。
"""

from __future__ import annotations

from keiba import config

RA_LENGTH = 1272
SE_LENGTH = 555
_JST = "+09:00"          # JRA-VAN の日時はすべて日本時間


def _text(raw: bytes, pos: int, length: int) -> str:
    return raw[pos - 1 : pos - 1 + length].decode("cp932", errors="replace").strip()


def record_spec(raw: bytes) -> str:
    return _text(raw, 1, 2)


def to_int(value: str) -> int | None:
    """数字だけなら int。空白や非数字は None（0 にしない）。"""
    value = value.strip()
    return int(value) if value.isdigit() else None


def tenth_sec(value: str) -> float | None:
    """99.9秒形式（3桁、0.1秒単位）。000 と 999 は無効値。"""
    value = value.strip()
    if not value.isdigit() or value in ("000", "999"):
        return None
    return int(value) / 10.0


def race_time_sec(value: str) -> float | None:
    """走破タイム。9分99秒9 形式の4桁。'1234' → 1分23.4秒 = 83.4秒。"""
    value = value.strip()
    if len(value) != 4 or not value.isdigit() or value in ("0000", "9999"):
        return None
    return int(value[0]) * 60 + int(value[1:3]) + int(value[3]) / 10.0


def margin_sec(time_diff: str) -> float | None:
    """SE.TimeDiff から「1着馬とのタイム差（秒）」を復元する。

    符号は1着が '-'、2着以下が '+'。1着馬の欄には2着馬とのタイム差が入るが、
    1着馬の margin は定義上 0.0 なので符号が '-' なら値を捨てて 0.0 を返す。
    取消・除外・中止は '9999'。
    """
    value = (time_diff or "").strip()
    if not value or value == "9999" or len(value) != 4:
        return None
    sign, digits = value[0], value[1:]
    if not digits.isdigit():
        return None
    if sign == "-":
        return 0.0
    if sign == "+":
        return int(digits) / 10.0
    return None


def _race_id(raw: bytes) -> str:
    """RACE_ID（絶対位置 12 から 16 バイト）を連結した16桁のキー。"""
    return (_text(raw, 12, 4) + _text(raw, 16, 4) + _text(raw, 20, 2)
            + _text(raw, 22, 2) + _text(raw, 24, 2) + _text(raw, 26, 2))


def _hasso_at(year: str, monthday: str, hhmm: str) -> str | None:
    if not (len(year) == 4 and len(monthday) == 4 and len(hhmm) == 4):
        return None
    if not (year + monthday + hhmm).isdigit():
        return None
    return f"{year}-{monthday[:2]}-{monthday[2:]}T{hhmm[:2]}:{hhmm[2:]}:00{_JST}"


def parse_ra(raw: bytes) -> dict:
    year, monthday = _text(raw, 12, 4), _text(raw, 16, 4)
    track_cd = _text(raw, 706, 2)
    kind = config.track_kind(track_cd)
    grade_cd = raw[614:615].decode("cp932", errors="replace")   # 空白を潰さない
    jyoken_youngest = _text(raw, 635, 3)                        # 最若年条件
    siba_baba, dirt_baba = _text(raw, 889, 1), _text(raw, 890, 1)
    baba_cd = siba_baba if kind == "turf" else dirt_baba if kind == "dirt" else ""
    laps = [tenth_sec(_text(raw, 891 + 3 * i, 3)) for i in range(25)]
    return {
        "race_id": _race_id(raw), "year": year, "monthday": monthday,
        "jyo_cd": _text(raw, 20, 2), "kaiji": _text(raw, 22, 2),
        "nichiji": _text(raw, 24, 2), "race_num": _text(raw, 26, 2),
        "hasso_at": _hasso_at(year, monthday, _text(raw, 874, 4)),
        "kyori_m": to_int(_text(raw, 698, 4)),
        "track_cd": track_cd, "track_kind": kind,
        "jyoken_cd_youngest": jyoken_youngest,
        "jyoken_cd_all": ";".join(_text(raw, 623 + 3 * i, 3) for i in range(5)),
        "grade_cd": grade_cd.strip(),
        "class_level": config.class_level(grade_cd, jyoken_youngest),
        "syubetu_cd": _text(raw, 617, 2), "kigo_cd": _text(raw, 619, 3),
        "jyuryo_cd": _text(raw, 622, 1), "course_kubun_cd": _text(raw, 710, 2),
        "tenko_cd": _text(raw, 888, 1),
        "siba_baba_cd": siba_baba, "dirt_baba_cd": dirt_baba, "baba_cd": baba_cd,
        "toroku_tosu": to_int(_text(raw, 882, 2)),
        "syusso_tosu": to_int(_text(raw, 884, 2)),
        "nyusen_tosu": to_int(_text(raw, 886, 2)),
        "lap_times_sec": ";".join("" if v is None else f"{v:.1f}" for v in laps),
        # 前3ハロンは元データに合計値が入っている。ラップを自分で足さない。
        "first_3f_sec": tenth_sec(_text(raw, 970, 3)),
        "last_3f_sec": tenth_sec(_text(raw, 976, 3)),
        "race_name": _text(raw, 33, 60),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


def _zogen(raw: bytes) -> int | None:
    """馬体重の増減。符号1バイト(328) + 3桁(329)。999 計量不能 / 空白 初出走。"""
    digits = _text(raw, 329, 3)
    if not digits.isdigit() or digits == "999":
        return None
    sign = raw[327:328].decode("cp932", errors="replace")
    return -int(digits) if sign == "-" else int(digits)


def _futan(value: str) -> float | None:
    """負担重量。3桁、0.1kg 単位。'570' → 57.0。"""
    value = value.strip()
    return int(value) / 10.0 if value.isdigit() and value != "000" else None


def parse_se(raw: bytes) -> dict:
    """結果側の列には result_ 接頭辞を付ける。誤用しにくくするため。"""
    odds = to_int(_text(raw, 360, 4))
    return {
        "race_id": _race_id(raw),
        "umaban": _text(raw, 29, 2), "wakuban": _text(raw, 28, 1),
        "ketto_num": _text(raw, 31, 10), "bamei": _text(raw, 41, 36),
        "sex_cd": _text(raw, 79, 1), "barei": _text(raw, 83, 2),
        "futan_kg": _futan(_text(raw, 289, 3)),
        "futan_before_kg": _futan(_text(raw, 292, 3)),
        "kisyu_code": _text(raw, 297, 5),
        "kisyu_code_before": _text(raw, 302, 5),
        "tozai_cd": _text(raw, 85, 1), "chokyosi_code": _text(raw, 86, 5),
        "uma_kigo_cd": _text(raw, 77, 2), "blinker": _text(raw, 295, 1),
        "result_ijyo_cd": _text(raw, 332, 1),
        "result_kakutei_jyuni": to_int(_text(raw, 335, 2)),
        "result_dochaku_kubun": _text(raw, 337, 1),
        "result_dochaku_tosu": to_int(_text(raw, 338, 1)),
        "result_time_sec": race_time_sec(_text(raw, 339, 4)),
        "result_margin_sec": margin_sec(_text(raw, 532, 4)),
        "result_jyuni_1c": to_int(_text(raw, 352, 2)),
        "result_jyuni_2c": to_int(_text(raw, 354, 2)),
        "result_jyuni_3c": to_int(_text(raw, 356, 2)),
        "result_jyuni_4c": to_int(_text(raw, 358, 2)),
        "result_last_3f_sec": tenth_sec(_text(raw, 391, 3)),
        "result_haron_l4_sec": tenth_sec(_text(raw, 388, 3)),
        "result_ba_taijyu_kg": to_int(_text(raw, 325, 3)),
        "result_zogen_sa_kg": _zogen(raw),
        "result_odds": odds / 10.0 if odds else None,
        "result_ninki": to_int(_text(raw, 364, 2)),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


def _ymd(value: str) -> str | None:
    value = value.strip()
    if len(value) != 8 or not value.isdigit() or value == "00000000":
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def parse_um(raw: bytes) -> dict:
    """時点に依存しない項目だけを読む。

    Ruikei* / ChakuKaisu* / Kyakusitu / RaceCount は「現在までの累計」で
    過去時点の値を復元できないため、ここでは一切読まない。
    """
    return {
        "ketto_num": _text(raw, 12, 10), "del_kubun": _text(raw, 22, 1),
        "reg_date": _ymd(_text(raw, 23, 8)), "del_date": _ymd(_text(raw, 31, 8)),
        "birth_date": _ymd(_text(raw, 39, 8)),
        "bamei": _text(raw, 47, 36), "bamei_kana": _text(raw, 83, 36),
        "sex_cd": _text(raw, 201, 1), "hinsyu_cd": _text(raw, 202, 1),
        "keiro_cd": _text(raw, 203, 2),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


_PARSERS = {"RA": parse_ra, "SE": parse_se, "UM": parse_um}


def parse(raw: bytes) -> tuple[str, dict] | None:
    """レコード種別で振り分ける。扱わない種別（O1 など）は None。"""
    spec = record_spec(raw)
    parser = _PARSERS.get(spec)
    return (spec, parser(raw)) if parser else None
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_records.py -v`
Expected: PASS（50件以上）

- [ ] **Step 6: テスト全体を通す**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS

```bash
git add src/keiba/ingest tests/fixtures tests/test_records.py
git commit -m "feat(ingest): parse RA/SE/UM fixed-width records"
```

---

## Task 4: 版付き保存と manifest

**Files:**
- Create: `src/keiba/ingest/store.py`
- Create: `tests/test_ingest_store.py`

**Interfaces:**
- Consumes: `keiba.config.SCHEMA_VERSION`, `keiba.config.DATA_DIR`
- Produces: `RACE_COLUMNS: tuple[str, ...]` / `RACE_HORSE_COLUMNS: tuple[str, ...]` / `HORSE_COLUMNS: tuple[str, ...]` / `class VersionWriter` with `__init__(data_dir: Path, version_id: str, *, dataspec: str, fromdate: str, option: int)`, `add(spec: str, row: dict) -> None`, `commit() -> Path`, `abort() -> None` / `read_manifest(version_dir: Path) -> dict` / `current_version(data_dir: Path) -> str | None` / `version_dir(data_dir: Path, version_id: str) -> Path` / `new_version_id() -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_ingest_store.py`:

```python
"""版付き保存。更新途中の CSV を予想で読まないことが要件。"""

import json

import pytest

from keiba.ingest import store
from tests.fixtures import records as fx
from keiba.ingest import records


@pytest.fixture
def written(tmp_path):
    writer = store.VersionWriter(tmp_path, "20260322T010000Z",
                                 dataspec="RACE", fromdate="20260301000000", option=1)
    writer.add("RA", records.parse_ra(fx.make_ra()))
    writer.add("SE", records.parse_se(fx.make_se()))
    writer.add("UM", records.parse_um(fx.make_um()))
    path = writer.commit()
    return tmp_path, path


class TestVersionWriter:
    def test_writes_three_csv_files_and_manifest(self, written):
        _, version_path = written
        for name in ("races.csv", "race_horses.csv", "horses.csv", "manifest.json"):
            assert (version_path / name).exists(), name

    def test_csv_is_utf8_without_bom(self, written):
        _, version_path = written
        raw = (version_path / "races.csv").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        raw.decode("utf-8")

    def test_ids_keep_leading_zeros(self, written):
        _, version_path = written
        text = (version_path / "race_horses.csv").read_text(encoding="utf-8")
        assert "0000000001" in text          # ketto_num
        assert "2026032106010511" in text    # race_id

    def test_manifest_records_counts_and_checksums(self, written):
        _, version_path = written
        manifest = store.read_manifest(version_path)
        assert manifest["version_id"] == "20260322T010000Z"
        assert manifest["schema_version"] == "1"
        assert manifest["status"] == "complete"
        assert manifest["record_counts"] == {"RA": 1, "SE": 1, "UM": 1}
        assert manifest["dataspec"] == "RACE"
        names = {f["name"] for f in manifest["files"]}
        assert names == {"races.csv", "race_horses.csv", "horses.csv"}
        for entry in manifest["files"]:
            assert len(entry["sha256"]) == 64
            assert entry["rows"] == 1

    def test_coverage_spans_written_races(self, written):
        _, version_path = written
        manifest = store.read_manifest(version_path)
        assert manifest["coverage"]["race_count"] == 1
        assert manifest["coverage"]["first_hasso_at"].startswith("2026-03-21")

    def test_current_points_at_committed_version(self, written):
        data_dir, _ = written
        assert store.current_version(data_dir) == "20260322T010000Z"

    def test_partial_directory_is_gone_after_commit(self, written):
        data_dir, _ = written
        assert not (data_dir / "versions" / "20260322T010000Z.partial").exists()


class TestIncompleteVersion:
    def test_partial_directory_exists_before_commit(self, tmp_path):
        writer = store.VersionWriter(tmp_path, "20260322T020000Z",
                                     dataspec="RACE", fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        partial = tmp_path / "versions" / "20260322T020000Z.partial"
        assert partial.exists()
        assert not (tmp_path / "versions" / "20260322T020000Z").exists()
        assert store.current_version(tmp_path) is None
        writer.abort()
        assert not partial.exists()

    def test_read_manifest_rejects_incomplete_status(self, tmp_path):
        version_path = tmp_path / "versions" / "20260322T030000Z"
        version_path.mkdir(parents=True)
        (version_path / "manifest.json").write_text(
            json.dumps({"version_id": "x", "status": "incomplete"}), encoding="utf-8")
        with pytest.raises(store.DataVersionError, match="完了していません"):
            store.read_manifest(version_path)


def test_new_version_id_is_sortable_utc_timestamp():
    version_id = store.new_version_id()
    assert len(version_id) == 16
    assert version_id[8] == "T"
    assert version_id.endswith("Z")
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'store'`

- [ ] **Step 3: 実装する**

`src/keiba/ingest/store.py`:

```python
"""CSV の版付き保存。

更新途中の CSV を予想が読まないように、次の順で書く。
  1. <version_id>.partial/ に全ファイルを書く
  2. sha256 を計算して manifest を書き、status を complete にする
  3. ディレクトリを <version_id>/ にリネーム（同一 FS なので原子的）
  4. CURRENT を書き換える
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from keiba import config


class DataVersionError(RuntimeError):
    """データ版が読めない・壊れている。"""


RACE_COLUMNS: tuple[str, ...] = (
    "race_id", "year", "monthday", "jyo_cd", "kaiji", "nichiji", "race_num",
    "hasso_at", "kyori_m", "track_cd", "track_kind",
    "jyoken_cd_youngest", "jyoken_cd_all", "grade_cd", "class_level",
    "syubetu_cd", "kigo_cd", "jyuryo_cd", "course_kubun_cd",
    "tenko_cd", "siba_baba_cd", "dirt_baba_cd", "baba_cd",
    "toroku_tosu", "syusso_tosu", "nyusen_tosu",
    "lap_times_sec", "first_3f_sec", "last_3f_sec", "race_name",
    "data_kubun", "make_date", "fetched_at",
)

RACE_HORSE_COLUMNS: tuple[str, ...] = (
    "race_id", "umaban", "wakuban", "ketto_num", "bamei",
    "sex_cd", "barei", "futan_kg", "futan_before_kg",
    "kisyu_code", "kisyu_code_before", "tozai_cd", "chokyosi_code",
    "uma_kigo_cd", "blinker",
    "result_ijyo_cd", "result_kakutei_jyuni", "result_dochaku_kubun",
    "result_dochaku_tosu", "result_time_sec", "result_margin_sec",
    "result_jyuni_1c", "result_jyuni_2c", "result_jyuni_3c", "result_jyuni_4c",
    "result_last_3f_sec", "result_haron_l4_sec",
    "result_ba_taijyu_kg", "result_zogen_sa_kg", "result_odds", "result_ninki",
    "data_kubun", "make_date", "fetched_at",
)

# UM の累計成績列は入れない。過去時点の値を復元できないため。
HORSE_COLUMNS: tuple[str, ...] = (
    "ketto_num", "bamei", "bamei_kana", "birth_date", "sex_cd",
    "keiro_cd", "hinsyu_cd", "del_kubun", "reg_date", "del_date",
    "data_kubun", "make_date", "fetched_at",
)

_FILES = {
    "RA": ("races.csv", RACE_COLUMNS),
    "SE": ("race_horses.csv", RACE_HORSE_COLUMNS),
    "UM": ("horses.csv", HORSE_COLUMNS),
}


def new_version_id() -> str:
    """取得開始時刻の UTC タイムスタンプ。辞書順が時刻順になる。"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def version_dir(data_dir: Path, version_id: str) -> Path:
    return Path(data_dir) / "versions" / version_id


def current_version(data_dir: Path) -> str | None:
    """CURRENT が指す版。未設定なら None。"""
    marker = Path(data_dir) / "CURRENT"
    if not marker.exists():
        return None
    value = marker.read_text(encoding="utf-8").strip()
    return value or None


def read_manifest(version_path: Path) -> dict:
    """manifest を読む。読めない・完了していないものはすべて拒否する。

    「読めない理由」はすべて DataVersionError に寄せる。呼び出し側は
    「この版は使えるか」だけを知りたいので、JSON の壊れ方まで区別させない。
    manifest.json の書き込み中に異常終了すると JSON が途中で切れるため、
    JSONDecodeError は現実に起こる。
    """
    path = Path(version_path) / "manifest.json"
    if not path.exists():
        raise DataVersionError(f"manifest が見つかりません: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DataVersionError(
            f"manifest が壊れています: {path} / {error}") from error
    if not isinstance(manifest, dict):
        raise DataVersionError(f"manifest の形式が想定と違います: {path}")
    if manifest.get("status") != "complete":
        raise DataVersionError(
            f"データ版 {manifest.get('version_id')} は書き込みが完了していません。"
            " 更新をやり直してください。")
    return manifest


def _cell(value: Any) -> str:
    """CSV の1セル。欠損は空文字列。0 や - に置き換えない。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VersionWriter:
    """1つのデータ版を書く。commit() まで CURRENT は切り替わらない。"""

    def __init__(self, data_dir: Path, version_id: str, *,
                 dataspec: str, fromdate: str, option: int) -> None:
        self.data_dir = Path(data_dir)
        self.version_id = version_id
        self.dataspec = dataspec
        self.fromdate = fromdate
        self.option = option
        self.fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._partial = version_dir(self.data_dir, version_id + ".partial")
        self._partial.mkdir(parents=True, exist_ok=True)
        self._counts: dict[str, int] = {"RA": 0, "SE": 0, "UM": 0}
        self._hasso_at: list[str] = []
        self._race_ids: set[str] = set()
        self._handles: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        for spec, (name, columns) in _FILES.items():
            handle = (self._partial / name).open("w", encoding="utf-8", newline="")
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            self._handles[spec] = handle
            self._writers[spec] = writer

    def add(self, spec: str, row: dict) -> None:
        if spec not in _FILES:
            return
        _, columns = _FILES[spec]
        enriched = dict(row)
        enriched["fetched_at"] = self.fetched_at
        self._writers[spec].writerow([_cell(enriched.get(c)) for c in columns])
        self._counts[spec] += 1
        if spec == "RA":
            if row.get("hasso_at"):
                self._hasso_at.append(row["hasso_at"])
            self._race_ids.add(row["race_id"])

    def abort(self) -> None:
        """書きかけを捨てる。CURRENT は触らない。"""
        self._close()
        shutil.rmtree(self._partial, ignore_errors=True)

    def commit(self) -> Path:
        """manifest を書き、本番ディレクトリにリネームし、CURRENT を更新する。"""
        self._close()
        files = []
        for spec, (name, _) in _FILES.items():
            path = self._partial / name
            files.append({
                "name": name,
                "rows": self._counts[spec],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        manifest = {
            "version_id": self.version_id,
            "schema_version": config.SCHEMA_VERSION,
            "fetched_at": self.fetched_at,
            "dataspec": self.dataspec,
            "fromdate": self.fromdate,
            "option": self.option,
            "coverage": {
                "first_hasso_at": min(self._hasso_at) if self._hasso_at else None,
                "last_hasso_at": max(self._hasso_at) if self._hasso_at else None,
                "race_count": len(self._race_ids),
            },
            "files": files,
            "record_counts": dict(self._counts),
            "status": "complete",
        }
        (self._partial / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        final = version_dir(self.data_dir, self.version_id)
        if final.exists():
            # 完成済みの版は不変。黙って上書きすると、予想ジョブが固定したはずの
            # 版の中身が後から変わる。new_version_id() は秒精度なので、短時間の
            # 再実行で version_id が衝突しうる。
            try:
                read_manifest(final)
            except DataVersionError:
                shutil.rmtree(final)   # manifest が無い/未完成なら異常終了の残骸
            else:
                raise DataVersionError(
                    f"データ版 {self.version_id} は既に完成しています。"
                    " 上書きしません。新しい版として取り直してください。")
        self._partial.rename(final)
        (self.data_dir / "CURRENT").write_text(self.version_id, encoding="utf-8")
        return final

    def _close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest_store.py -v` → PASS

- [ ] **Step 5: テスト全体を通す**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS

```bash
git add src/keiba/ingest/store.py tests/test_ingest_store.py
git commit -m "feat(ingest): versioned CSV store with manifest and atomic commit"
```

---

> **Task 5 以降の書式について**
>
> Task 1–4 は各ステップのコードを全文載せた。Task 5 以降は、**インターフェース（関数名・引数・戻り値の型）とテストケースを完全に確定させたうえで、実装コードは非自明な箇所（アルゴリズム・境界条件・数値の扱い）だけを全文載せる**形式にする。機械的な詰め替えは列名とインターフェースが確定していれば一意に決まるため。
>
> 各タスクの手順は Task 1–4 と同じ: テストを書く → 落ちるのを確認 → 実装 → 通るのを確認 → テスト全体を通す。

---

## Task 5: JV-Link 呼び出しと update_data CLI

**Files:**
- Create: `src/keiba/ingest/jvlink.py`
- Create: `scripts/update_data.py`
- Create: `src/keiba/console.py`
- Create: `tests/test_jvlink.py`
- Create: `tests/test_console.py`

**Interfaces:**
- Consumes: `keiba.ingest.records.parse`, `keiba.ingest.store.VersionWriter`, `keiba.ingest.store.new_version_id`
- Produces:
  - `src/keiba/console.py`: `enable_utf8_output() -> None`（`scripts/smoke_test.py` の `_enable_utf8_output` と同じ処理を共通化。smoke_test.py 自体は変更しない）
  - `jvlink.JVLinkError(RuntimeError)`
  - `jvlink.open_link(sid: str = "UNKNOWN") -> Any`（`win32com.client.Dispatch("JVDTLab.JVLink")` して `JVInit`。非 Windows と pywin32 未導入は `JVLinkError`）
  - `jvlink.read_records(link: Any, dataspec: str, fromdate: str, option: int, *, buffer_size: int = 110000) -> Iterator[bytes]`（`JVOpen` → `JVStatus` でダウンロード待ち → `JVGets` でレコードを1件ずつ yield → `JVClose`）
  - `jvlink.describe_open_code(code: int) -> str`（`0` 正常 / `-1` 該当データなし / それ以外はエラー番号を添えた日本語）
  - `jvlink.verify_time_fields(rows: Sequence[dict]) -> list[dict]`（spec 15.2 節の突き合わせ。`SE` の dict 列を受け、レースごとに「2着馬の `result_time_sec` − 1着馬の `result_time_sec`」と「2着馬の `result_margin_sec`」を比べ、`{"race_id", "computed", "recorded", "matches"}` を返す）

- [ ] **Step 1: テストを書く**

`tests/test_console.py`:

| テスト | 内容 |
|---|---|
| `test_enable_utf8_output_is_idempotent` | 2回呼んでも例外を投げない |
| `test_enable_utf8_output_sets_stdout_encoding` | 呼んだあと `sys.stdout.encoding.lower()` が `utf-8` を含む（`reconfigure` を持たない stream でも落ちない） |

`tests/test_jvlink.py` は COM を叩かない。偽の link オブジェクトを使う。

| テスト | 内容 |
|---|---|
| `test_read_records_yields_each_record` | `JVGets` が `(len, memoryview, name)` を3回返し4回目に `0` を返す偽 link で、3件 yield される |
| `test_read_records_skips_file_switch` | `JVGets` が `-1`（ファイル切替）を返した回は yield しない |
| `test_read_records_raises_on_negative_code` | `JVGets` が `-2` を返したら `JVLinkError` |
| `test_read_records_raises_when_open_fails` | `JVOpen` が `4` を返したら `JVLinkError`。メッセージに `4` が入る |
| `test_read_records_returns_nothing_when_no_data` | `JVOpen` が `-1` を返したら例外にせず0件で終わる |
| `test_read_records_always_closes` | 途中で例外が出ても `JVClose` が呼ばれる |
| `test_read_records_waits_for_download` | `JVStatus` が `0,1,2` と返り `download_count=2` のとき、`JVStatus` が2回以上呼ばれる |
| `test_describe_open_code` | `0` → 「正常」／`-1` → 「該当データなし」／`4` → 数値を含む日本語 |
| `test_verify_time_fields_matches_when_consistent` | 1着 83.4 秒 / 2着 83.7 秒 / 2着の `margin_sec` 0.3 → `matches is True` |
| `test_verify_time_fields_flags_mismatch` | 2着の `margin_sec` を 0.9 にすると `matches is False`、`computed == 0.3` |
| `test_verify_time_fields_skips_races_without_two_finishers` | 1頭だけのレースは結果に出ない |

偽 link の雛形（テストに書く）:

```python
class FakeLink:
    def __init__(self, gets_returns, *, open_code=0, download_count=0, statuses=None):
        self._gets = list(gets_returns)
        self._open_code = open_code
        self._download_count = download_count
        self._statuses = list(statuses or [download_count])
        self.closed = False
        self.status_calls = 0

    def JVInit(self, sid):
        return 0

    def JVOpen(self, dataspec, fromdate, option, _a, _b, _c):
        # pywin32 は ByRef 出力引数を戻り値のタプルで返す
        return (self._open_code, 0, self._download_count, "")

    def JVStatus(self):
        self.status_calls += 1
        return self._statuses[min(self.status_calls - 1, len(self._statuses) - 1)]

    def JVGets(self, buf, size, fname):
        if not self._gets:
            return (0, memoryview(b""), "")
        return self._gets.pop(0)

    def JVClose(self):
        self.closed = True
```

- [ ] **Step 2: 実装する**

`read_records` の骨格（ここだけ非自明なので全文）:

```python
def read_records(link, dataspec, fromdate, option, *, buffer_size=110000):
    """JVOpen → ダウンロード待ち → JVGets の繰り返し → JVClose。

    JVGets の戻り値:
       >0  : memview.tobytes() にレコード1件
       -1  : ファイル切替（fname が対象ファイル名）
        0  : EOF
       <-1 : エラー
    レコードは cp932。デコードは records.py 側で行う。
    """
    open_code, _read_count, download_count, _lastfiletime = link.JVOpen(
        dataspec, fromdate, option, 0, 0, "")
    if open_code == -1:
        link.JVClose()
        return
    if open_code != 0:
        link.JVClose()
        raise JVLinkError(describe_open_code(open_code))
    try:
        while link.JVStatus() < download_count:
            time.sleep(0.2)
        while True:
            code, memview, _fname = link.JVGets(bytearray(buffer_size), buffer_size,
                                                bytearray())
            if code == 0:
                return
            if code == -1:          # ファイル切替。レコードではない
                continue
            if code < -1:
                raise JVLinkError(f"JVGets がエラーを返しました (code={code})。")
            yield memview.tobytes()
    finally:
        link.JVClose()
```

`verify_time_fields`（spec 15.2 節。仕様書の解釈が実データと合うかを機械で確認する）:

```python
def verify_time_fields(rows):
    """1着馬と2着馬の走破タイム差が、2着馬の margin_sec と一致するか確かめる。

    一致すれば records.margin_sec の解釈（符号1文字 + 0.1秒単位3桁）が正しい。
    手作業のメモにせず、update_data.py --verify-time-fields から再実行できる。
    """
    by_race: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("result_ijyo_cd") in ("1", "2", "3", "4", "5"):
            continue
        if row.get("result_time_sec") is None:
            continue
        by_race.setdefault(row["race_id"], []).append(row)

    report = []
    for race_id, entries in by_race.items():
        ranked = sorted(e for e in entries if e.get("result_kakutei_jyuni"))
        ranked = sorted(entries, key=lambda e: e["result_kakutei_jyuni"])
        first = next((e for e in ranked if e["result_kakutei_jyuni"] == 1), None)
        second = next((e for e in ranked if e["result_kakutei_jyuni"] == 2), None)
        if first is None or second is None:
            continue
        computed = round(second["result_time_sec"] - first["result_time_sec"], 1)
        recorded = second.get("result_margin_sec")
        report.append({
            "race_id": race_id,
            "computed": computed,
            "recorded": recorded,
            "matches": recorded is not None and abs(computed - recorded) < 0.05,
        })
    return report
```

`scripts/update_data.py` の引数:

```
--dataspec RACE            JVOpen の dataspec（既定 RACE）
--fromdate 20260801000000  取得開始日時（必須）
--option 1                 JVOpen の option（既定 1）
--data-dir data            保存先（既定 config.DATA_DIR）
--verify-time-fields       取得後に verify_time_fields を実行して不一致件数を表示
--dry-run                  取得はするが CURRENT を切り替えない（abort する）
```

出力: 読んだレコード数（種別ごと）、書いた version_id、manifest のパス、`--verify-time-fields` 指定時は照合件数と不一致件数。不一致が1件でもあれば exit 1 にして、黙って進まないようにする。

- [ ] **Step 3: 実データで1回だけ照合する（手動）**

Run: `.venv/Scripts/python.exe scripts/update_data.py --fromdate 20260801000000 --verify-time-fields`
Expected: 「不一致 0 件」。不一致が出た場合は spec 4.1.1 節の `margin_sec` の解釈を実態に合わせて直し、Task 3 のテストも直す。

- [ ] **Step 4: テスト全体を通す**

Run: `.venv/Scripts/python.exe -m pytest` → 全件 PASS

```bash
git add src/keiba/ingest/jvlink.py src/keiba/console.py scripts/update_data.py tests/test_jvlink.py tests/test_console.py
git commit -m "feat(ingest): read JV-Link records and add update_data CLI"
```

---

## Task 6: DataStore と Repository（as_of の強制）

**Files:**
- Create: `src/keiba/data/__init__.py`, `src/keiba/data/store.py`, `src/keiba/data/repository.py`
- Create: `tests/fixtures/csv_data.py`
- Create: `tests/test_repository.py`

**Interfaces:**
- Consumes: `keiba.models.*`, `keiba.ingest.store.read_manifest`, `keiba.ingest.store.current_version`, `keiba.config`
- Produces:
  - `store.DataStore`: `open_current(data_dir: Path) -> DataStore` / `open_version(data_dir: Path, version_id: str) -> DataStore`。属性 `version_id: str` / `fetched_at: datetime` / `races: pd.DataFrame` / `race_horses: pd.DataFrame` / `horses: pd.DataFrame`。**読み込みは `dtype=str` で行い、数値化は取り出すときに行う**（先頭ゼロを守る）
  - `repository.Repository(store: DataStore)`: 公開メソッドはすべて `*, as_of: datetime` を必須で取る
    - `past_runs(ketto_num: str, *, as_of: datetime, limit: int | None = None) -> list[PastRun]`（発走が新しい順）
    - `find_horses_by_name(normalized_name: str, *, as_of: datetime) -> list[HorseMaster]`
    - `entries(race_id: str, *, as_of: datetime) -> list[EntryRow]`
    - `race(race_id: str, *, as_of: datetime) -> RaceRow | None`
    - プロパティ `data_version: str` / `data_fetched_at: datetime`（`as_of` 不要なのでメソッドにしない）
  - `repository.ResultRepository(store: DataStore)`: 評価専用。`as_of` を取らない
    - `race(race_id: str) -> RaceRow | None`
    - `results(race_id: str) -> list[ResultRow]`
    - `winners(race_id: str) -> list[str]`（1着の `umaban`。同着は複数）
    - `races_between(start: datetime, end: datetime) -> list[RaceRow]`（`hasso_at` 昇順、平地のみ）

- [ ] **Step 1: CSV フィクスチャを作る**

`tests/fixtures/csv_data.py`: `write_dataset(tmp_path, races: list[dict], race_horses: list[dict], horses: list[dict], *, version_id="20260322T010000Z") -> Path`。`VersionWriter` を使わず直接 CSV を書く（Repository のテストを VersionWriter の挙動から切り離すため）。列は `ingest.store.RACE_COLUMNS` などを使い、指定のない列は空文字列。`manifest.json` は `status: "complete"` で書き、`CURRENT` も書く。

- [ ] **Step 2: テストを書く**

`tests/test_repository.py`。**未来情報の混入防止が最優先**なので、そこを最初に置く。

| テスト | 内容 |
|---|---|
| `test_every_public_method_requires_as_of` | `inspect.signature` で `Repository` の公開メソッド全部を検査。`as_of` が `KEYWORD_ONLY` で `default is Parameter.empty` であること |
| `test_result_repository_has_no_as_of` | `ResultRepository` の公開メソッドに `as_of` が無いこと（評価専用と分かる形） |
| `test_past_runs_excludes_races_after_as_of` | 3走（3/01, 3/08, 3/15）の馬で `as_of=3/10` なら 3/01 と 3/08 だけ返る |
| `test_past_runs_excludes_race_starting_exactly_at_as_of` | `hasso_at == as_of` の走は含まない（`<` で比較） |
| `test_past_runs_orders_newest_first` | 返り値の `race.hasso_at` が降順 |
| `test_past_runs_respects_limit` | `limit=2` で2件 |
| `test_past_runs_excludes_jump_races` | `track_kind == "jump"` の走は返らない |
| `test_past_runs_prefers_data_kubun_7` | 同じ `(race_id, umaban)` に `data_kubun` `3` と `7` があれば `7` の値が返る |
| `test_past_runs_falls_back_through_priority` | `7` が無く `5` と `3` があれば `5` |
| `test_past_runs_drops_deleted_and_cancelled` | `data_kubun` が `0` / `9` / `A` / `B` の行は返らない |
| `test_past_runs_uses_newest_make_date_on_duplicate` | 同一 `(race_id, umaban, data_kubun)` が2行なら `make_date` の新しい方 |
| `test_entries_prefers_data_kubun_2_then_1` | `2` があれば `2`、無ければ `1` |
| `test_entries_falls_back_to_result_rows_for_entry_fields_only` | `1`/`2` が無く `7` だけのとき `EntryRow` が返り、`ResultRow` は返らない |
| `test_entries_returns_entry_row_type` | 戻り値が `EntryRow` のインスタンスで、結果の属性を持たない |
| `test_results_include_scratched_horses` | `result_ijyo_cd="1"` の馬も `results()` には含まれる（除外は評価側の判断） |
| `test_winners_returns_all_dead_heat_horses` | `kakutei_jyuni=1` が2頭なら2件 |
| `test_winners_excludes_disqualified` | `ijyo_cd="5"`（失格）で `kakutei_jyuni=1` の馬は勝ち馬にしない |
| `test_races_between_is_inclusive_and_sorted` | 境界を含み `hasso_at` 昇順 |
| `test_races_between_excludes_jump` | 障害レースは返らない |
| `test_ids_are_strings_with_leading_zeros` | `ketto_num` が `"0000000001"` のまま（`1` にならない） |
| `test_open_current_rejects_incomplete_version` | `manifest.status != "complete"` なら `DataVersionError` |
| `test_open_current_raises_when_no_current` | `CURRENT` が無ければ `DataVersionError` |
| `test_data_version_and_fetched_at_exposed` | `repo.data_version` と `repo.data_fetched_at` が manifest の値と一致 |

- [ ] **Step 3: 実装する**

`Repository` の中核だけ全文。`as_of` フィルタと `data_kubun` の優先順位がこのプロジェクトの安全装置なので、ここは1か所に閉じ込める。

```python
class Repository:
    """予想用のデータ参照。

    全公開メソッドが as_of を必須のキーワード引数に取る。既定値を与えないのは、
    うっかり as_of なしで呼べる経路を作らないため。テストで signature を検査する。
    評価用（結果を見る参照）は ResultRepository に分けてある。
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    @property
    def data_version(self) -> str:
        return self._store.version_id

    @property
    def data_fetched_at(self) -> datetime:
        return self._store.fetched_at

    def past_runs(self, ketto_num: str, *, as_of: datetime,
                  limit: int | None = None) -> list[PastRun]:
        """as_of より前に発走した平地の走を、新しい順に返す。"""
        rows = self._store.race_horses
        rows = rows[rows["ketto_num"] == ketto_num]
        rows = _pick_by_data_kubun(rows, config.DATA_KUBUN_RESULT_PRIORITY)
        runs: list[PastRun] = []
        for _, row in rows.iterrows():
            race = self._race_row(row["race_id"])
            if race is None or not race.is_flat:
                continue
            if race.hasso_at >= as_of:       # 同時刻も含めない
                continue
            runs.append(PastRun(race=race, result=_to_result_row(row)))
        runs.sort(key=lambda run: run.race.hasso_at, reverse=True)
        return runs[:limit] if limit else runs
```

`_pick_by_data_kubun` は「行群を `(race_id, umaban)` ごとにまとめ、優先順位の最初に見つかった `data_kubun` の行を採り、同一区分が複数なら `make_date` の新しい方を採る」。`DATA_KUBUN_DROP` の行は先に落とす。

```python
def _pick_by_data_kubun(rows: pd.DataFrame, priority: tuple[str, ...]) -> pd.DataFrame:
    """同じ (race_id, umaban) の複数レコードから1行を選ぶ。

    JV-Data は木曜→金土→速報→月曜と同じレースを何度も配信するので、
    CSV には同じキーの行が複数入っている。優先順位で1行に絞る。
    """
    rows = rows[~rows["data_kubun"].isin(config.DATA_KUBUN_DROP)]
    if rows.empty:
        return rows
    rank = {kubun: i for i, kubun in enumerate(priority)}
    rows = rows[rows["data_kubun"].isin(rank)].copy()
    rows["_rank"] = rows["data_kubun"].map(rank)
    rows = rows.sort_values(["_rank", "make_date"], ascending=[True, False])
    keys = ["race_id", "umaban"] if "umaban" in rows.columns else ["race_id"]
    return rows.drop_duplicates(subset=keys, keep="first").drop(columns="_rank")
```

`ResultRepository.winners` は `ResultRow.is_winner`（= `can_win and kakutei_jyuni == 1`）を使う。失格・競走中止は `can_win` が `False` なので自動的に勝ち馬から外れる。

- [ ] **Step 4: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/data tests/fixtures/csv_data.py tests/test_repository.py
git commit -m "feat(data): repository with mandatory as_of and data_kubun selection"
```

---

## Task 7: 入力の型と検証

**Files:**
- Create: `src/keiba/input/__init__.py`, `src/keiba/input/schema.py`, `src/keiba/input/validate.py`
- Create: `tests/test_input.py`

**Interfaces:**
- Consumes: `keiba.models.RaceInput`, `keiba.models.HorseEntry`, `keiba.config`
- Produces:
  - `schema.InputError(ValueError)`
  - `schema.parse_race_input(payload: dict) -> RaceInput`（JSON 由来の dict から構築。日時は ISO 8601 文字列を受ける。**`jyo` と `baba` の表示名を `jyo_cd` / `baba_cd` に解決して埋める**）
  - `schema.load_race_input(path: Path) -> RaceInput`（`.json` を読む）
  - `schema.resolve_jyo_cd(value: str) -> str`（「中山」→ `"06"`。コードをそのまま渡してもよい。解決できなければ `InputError`）
  - `schema.resolve_baba_cd(value: str | None) -> str | None`（「良」→ `"1"`。`None` は `None`。解決できなければ `InputError`）
  - `validate.Problem`（`frozen dataclass`: `field: str`, `message: str`）
  - `validate.check(race_input: RaceInput) -> list[Problem]`（空なら予想を開始できる）

- [ ] **Step 1: テストを書く**

`tests/test_input.py`:

| テスト | 内容 |
|---|---|
| `test_parse_minimal_payload` | 必須項目だけの dict から `RaceInput` ができ、`top_n == 3` |
| `test_parse_reads_iso_datetimes` | `prediction_cutoff` / `hasso_at` が `datetime` になる |
| `test_parse_rejects_missing_required_field` | `jyo` が無ければ `InputError`。メッセージに `jyo` が入る |
| `test_parse_keeps_umaban_as_string_with_leading_zero` | `"01"` が `"01"` のまま |
| `test_parse_rejects_unknown_track_kind` | `track_kind="芝"` は `InputError`（`turf` / `dirt` のみ） |
| `test_parse_resolves_jyo_name_to_code` | `jyo="中山"` → `jyo_cd == "06"`、`jyo` は `"中山"` のまま |
| `test_parse_accepts_jyo_code_directly` | `jyo="06"` でも `jyo_cd == "06"` になり `jyo == "中山"` に整う |
| `test_parse_rejects_unknown_jyo` | `jyo="大井"`（地方）は `InputError`。中央10場のみ |
| `test_parse_resolves_baba_name_to_code` | `baba="稍重"` → `baba_cd == "2"` |
| `test_parse_keeps_baba_none` | `baba` 未指定で `baba_cd is None`（`"1"`（良）に寄せない） |
| `test_parse_rejects_unknown_baba` | `baba="やや重"` は `InputError`（表記を勝手に推測しない） |
| `test_check_passes_for_valid_input` | 問題なしで `[]` |
| `test_check_detects_duplicate_umaban` | 馬番重複を検出。`field == "horses"` |
| `test_check_detects_entry_count_mismatch` | `entry_count=12` で馬が3頭なら検出 |
| `test_check_detects_non_positive_distance` | `kyori_m=0` と `kyori_m=-1` |
| `test_check_detects_unrealistic_distance` | `kyori_m=100`（下限800m未満）と `kyori_m=10000`（上限5000m超） |
| `test_check_detects_cutoff_after_start` | `prediction_cutoff >= hasso_at` なら検出 |
| `test_check_allows_missing_optional_fields` | `baba` / `class_name` / `wakuban` / `futan_kg` が無くても `[]` |
| `test_check_detects_empty_horse_name` | `bamei=""` を検出 |
| `test_check_detects_top_n_out_of_range` | `top_n=0` と `top_n > 頭数` を検出 |
| `test_check_reports_all_problems_at_once` | 重複馬番と頭数不一致が同時にあれば2件返る（1件で止めない） |

- [ ] **Step 2: 実装する**

距離の現実的な範囲は `config.py` に出す: `MIN_DISTANCE_M = 800` / `MAX_DISTANCE_M = 5000`（中央の平地は 1000m〜3600m だが、障害や特殊条件の余地を残す）。

`check` は**最初の問題で止めず全部集めて返す**。利用者が1件ずつ直すのは無駄なため（spec 14.2 節）。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/input tests/test_input.py
git commit -m "feat(input): race input schema and validation"
```

---

## Task 8: 馬名照合

**Files:**
- Create: `src/keiba/input/match.py`
- Create: `tests/test_match.py`

**Interfaces:**
- Consumes: `keiba.data.repository.Repository`, `keiba.models.HorseMaster`, `keiba.models.HorseEntry`
- Produces:
  - `normalize_name(name: str) -> str`（NFKC 正規化 → 空白・中黒・ハイフン等の記号除去 → 大文字化。カタカナは変換しない）
  - `MatchResult`（`frozen dataclass`: `umaban: str`, `entry: HorseEntry`, `ketto_num: str | None`, `status: str`, `candidates: tuple[HorseMaster, ...]`）。`status` は `"matched"` / `"unregistered"` / `"ambiguous"`
  - `AmbiguousMatchError(ValueError)`（`results: tuple[MatchResult, ...]` を持つ）
  - `match_horses(race_input: RaceInput, repo: Repository, *, as_of: datetime) -> list[MatchResult]`。`ambiguous` が1件でもあれば `AmbiguousMatchError` を投げる

- [ ] **Step 1: テストを書く**

| テスト | 内容 |
|---|---|
| `test_normalize_collapses_width_and_symbols` | `"テストウマ"`（全角）と `"ﾃｽﾄｳﾏ"`、`"テスト ウマ"`、`"テスト・ウマ"` が同じ値になる |
| `test_normalize_keeps_katakana_as_katakana` | ひらがなに変換しない（`"テスト"` と `"てすと"` は別） |
| `test_match_resolves_unique_name` | 1件ヒットで `status == "matched"`、`ketto_num` が入る |
| `test_match_marks_unregistered_when_no_candidate` | 0件で `status == "unregistered"`、`ketto_num is None` |
| `test_match_narrows_by_sex_age` | 同名2頭を `sex_age="牡4"` で1頭に絞れる |
| `test_match_raises_on_ambiguous` | 絞っても2頭残れば `AmbiguousMatchError`。`candidates` に2件入る |
| `test_match_does_not_auto_pick_first_candidate` | 曖昧なとき先頭を勝手に採らない（上のテストで例外が出ることで担保） |
| `test_unregistered_is_not_ambiguous` | 未登録は例外にせず続行できる（spec 14.3 節「曖昧で確定できない」と「未登録で過去実績なし」を区別） |
| `test_match_does_not_borrow_another_horses_history` | `unregistered` の `ketto_num` が `None` なので `past_runs` を引けない |
| `test_match_passes_as_of_through` | `repo.find_horses_by_name` が `as_of` 付きで呼ばれる（モックで確認） |

- [ ] **Step 2: 実装する**

`normalize_name` だけ全文（境界が細かいので）:

```python
import re
import unicodedata

# 馬名の表記ゆれで意味を持たない記号。全角・半角の空白、中黒、ハイフン類。
_NOISE = re.compile(r"[\s　・･\-‐‑‒–—―ー]+")


def normalize_name(name: str) -> str:
    """馬名を照合用に正規化する。

    NFKC で全角・半角を揃えたあと、意味を持たない記号を落として大文字化する。
    カタカナはカタカナのまま残す（ひらがなに寄せると別の馬と衝突しうる）。

    注意: NFKC は長音符「ー」を変換しないので _NOISE で落としている。
    馬名の長音は表記ゆれが多く、照合のノイズになるため。
    """
    folded = unicodedata.normalize("NFKC", name)
    return _NOISE.sub("", folded).upper()
```

`match_horses` は「絞れない候補が1件でもあれば例外」。例外にする理由は、間違った馬の履歴で予想すると結果が静かに壊れるため（spec 14.3 節）。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/input/match.py tests/test_match.py
git commit -m "feat(input): horse name matching that refuses ambiguous candidates"
```

---

## Task 9: 能力指標

**Files:**
- Create: `src/keiba/features/__init__.py`, `src/keiba/features/ability.py`
- Create: `tests/fixtures/runs.py`
- Create: `tests/test_ability.py`

**Interfaces:**
- Consumes: `keiba.models.PastRun`, `keiba.config.DEFAULT_RECENT_RUNS`
- Produces:
  - `AbilitySummary`（`frozen dataclass`: `recent_margin_sec_wavg: float | None`, `runs_used: int`, `recent_class_level_wavg: float | None`, `best_class_level: int | None`, `period_from: date | None`, `period_to: date | None`）
  - `ability_summary(runs: Sequence[PastRun], *, recent_runs: int = config.DEFAULT_RECENT_RUNS) -> AbilitySummary`
  - `field_ranks(values: Mapping[str, float | None]) -> dict[str, float | None]`（昇順の順位。同値は平均順位。`None` は `None` のまま返し、最下位扱いにしない）
- `tests/fixtures/runs.py`: `make_past_run(*, hasso_at, kyori_m=2000, track_kind="turf", jyo_cd="06", baba_cd="1", class_level=4, margin_sec=0.3, kakutei_jyuni=3, ijyo_cd="0", jyuni_4c=5, syusso_tosu=12, zogen_sa_kg=-2, futan_kg=57.0) -> PastRun`

- [ ] **Step 1: テストを書く**

重みは「直近から 5,4,3,2,1」。加重平均の分母は**実際に使った走の重みの合計**（走数が5未満でも正しくなるように）。

| テスト | 内容 |
|---|---|
| `test_weighted_average_uses_recency_weights` | margin が新しい順に `[0.0, 1.0, 2.0]` なら `(0*5 + 1*4 + 2*3) / (5+4+3) = 10/12 ≈ 0.833` |
| `test_uses_only_recent_runs` | 7走あっても直近5走だけ使い `runs_used == 5` |
| `test_runs_used_reflects_fewer_runs` | 2走なら `runs_used == 2` |
| `test_returns_none_for_no_runs` | 0走で `recent_margin_sec_wavg is None`、`runs_used == 0`、`period_from is None` |
| `test_skips_runs_without_time` | `ijyo_cd="4"`（競走中止）の走は margin 集計から外れ、`runs_used` に数えない |
| `test_counts_demoted_run_as_normal` | `ijyo_cd="7"`（降着）は集計に入る |
| `test_skips_jump_runs` | `track_kind="jump"` は集計に入らない |
| `test_best_class_level_is_max_of_recent` | 直近5走のクラス順序値の最大 |
| `test_class_level_wavg_uses_same_weights` | クラス順序値も同じ重みで加重平均される |
| `test_class_level_none_when_all_missing` | 全走 `class_level=None` なら `recent_class_level_wavg is None` |
| `test_period_covers_used_runs_only` | `period_from` / `period_to` が実際に使った走の日付範囲 |
| `test_field_ranks_ascending_smaller_is_better` | `{"1": 0.1, "2": 0.5, "3": 0.3}` → `{"1": 1.0, "2": 3.0, "3": 2.0}` |
| `test_field_ranks_averages_ties` | `{"1": 0.3, "2": 0.3, "3": 0.5}` → `{"1": 1.5, "2": 1.5, "3": 3.0}` |
| `test_field_ranks_keeps_none_as_none` | `None` の馬は `None`。最下位に落とさない |
| `test_field_ranks_skips_none_in_numbering` | `{"1": 0.1, "2": None, "3": 0.3}` → `{"1": 1.0, "2": None, "3": 2.0}` |

- [ ] **Step 2: 実装する**

```python
def _weighted(values: Sequence[float | None], weights: Sequence[int]) -> float | None:
    """欠損を飛ばした加重平均。分母は実際に使った重みの合計。

    走数が 5 未満のときに分母を 15 に固定すると値が不当に小さくなるので、
    使った重みだけを足す。
    """
    total = weight_sum = 0.0
    for value, weight in zip(values, weights):
        if value is None:
            continue
        total += value * weight
        weight_sum += weight
    return total / weight_sum if weight_sum else None
```

重みは `range(recent_runs, 0, -1)` = 5,4,3,2,1。

`field_ranks` は `None` を除いた値だけで順位を付け、同値は平均順位にする。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/features/__init__.py src/keiba/features/ability.py tests/fixtures/runs.py tests/test_ability.py
git commit -m "feat(features): ability metric from recency-weighted margins"
```

---

## Task 10: 条件別成績・近況・脚質・輸送

**Files:**
- Create: `src/keiba/features/derive.py`
- Create: `tests/test_derive.py`

**Interfaces:**
- Consumes: `keiba.models.PastRun`, `keiba.models.RaceInput`, `keiba.config`
- Produces:
  - `ConditionRecord`（`frozen dataclass`: `starts: int`, `wins: int`, `top3: int`, `margin_sec_mean: float | None`, `period_from: date | None`, `period_to: date | None`）
  - `CONDITION_KEYS = ("exact_distance", "same_distance", "same_venue", "same_track_kind", "same_baba")`
  - `condition_records(runs: Sequence[PastRun], *, race: RaceInput) -> dict[str, ConditionRecord]`
  - `RecentForm`（`frozen dataclass`: `days_since_last_run: int | None`, `distance_change_m: int | None`, `futan_change_kg: float | None`, `class_level_change: int | None`, `last_run: dict | None`）
  - `recent_form(runs, *, as_of: datetime, race: RaceInput, futan_kg: float | None, class_level: int | None) -> RecentForm`
  - `RunningStyle`（`frozen dataclass`: `corner_pos_mean: float | None`, `corner_pos_stdev: float | None`, `corner_pos_starts: int`, `corner_source: str | None`）
  - `running_style(runs, *, recent_runs: int = config.DEFAULT_RECENT_RUNS) -> RunningStyle`
  - `Transport`（`frozen dataclass`: `base: str | None`, `estimated_transport_km: int | None`, `stay_status: str`, `past_long_transport_starts: int`, `past_long_transport_zogen_mean: float | None`, `past_long_transport_zogen_min: int | None`）
  - `transport(runs, *, tozai_cd: str, jyo_cd: str, stay_status: str = "unknown") -> Transport`

- [ ] **Step 1: テストを書く**

**条件別成績**

| テスト | 内容 |
|---|---|
| `test_exact_distance_matches_only_identical` | 2000m のレースで 2000m の走だけ |
| `test_same_distance_uses_200m_tolerance` | 1800m と 2200m は含み、1600m と 2400m は含まない（`±200` 以内、境界を含む） |
| `test_same_venue_matches_jyo_cd` | 競馬場コードで一致 |
| `test_same_track_kind_matches_turf_or_dirt` | 芝/ダートで一致 |
| `test_same_baba_requires_same_track_kind_too` | 馬場状態が同じでもトラック種別が違えば含まない |
| `test_zero_starts_reports_none_mean_not_zero` | 0件なら `starts=0`, `wins=0`, `top3=0`, `margin_sec_mean is None`（0.0 にしない） |
| `test_wins_counts_only_first_place` | `kakutei_jyuni=1` だけを `wins` に数える |
| `test_top3_counts_first_to_third` | 1〜3着を `top3` |
| `test_excludes_scratched_from_starts` | `ijyo_cd="1"` は `starts` に数えない |
| `test_includes_no_time_runs_in_starts_but_not_mean` | `ijyo_cd="4"` は `starts` に数えるが `margin_sec_mean` には入れない |
| `test_period_reflects_matched_runs` | `period_from` / `period_to` が一致した走の範囲 |
| `test_counts_are_always_present_with_rates` | どの区分でも `starts` が必ず入る（spec §4.3） |

**近況**

| テスト | 内容 |
|---|---|
| `test_days_since_last_run_from_as_of` | `as_of` − 直前走の `hasso_at` の日数 |
| `test_changes_are_signed_differences` | 距離 +200、斤量 +1.0、クラス 0 などの符号付き差分 |
| `test_all_none_when_no_past_runs` | 0走なら全項目 `None`、`last_run is None` |
| `test_class_change_none_when_either_side_missing` | どちらかの `class_level` が `None` なら差分も `None` |
| `test_last_run_includes_conditions_and_weight` | `last_run` に距離・トラック・馬場・クラス・頭数・着順・着差・斤量・馬体重が入る |
| `test_last_run_weight_is_previous_race_value` | 馬体重は前走の値（今回の値は当日発表なので使わない） |

**脚質**

| テスト | 内容 |
|---|---|
| `test_corner_position_is_relative` | `jyuni_4c=3`, `syusso_tosu=12` → `0.25` |
| `test_mean_and_stdev_over_recent_runs` | 手計算した値と一致 |
| `test_stdev_none_with_single_run` | 1走なら `corner_pos_stdev is None` |
| `test_source_is_jyuni_4c_when_available` | `corner_source == "jyuni_4c"` |
| `test_falls_back_to_jyuni_3c` | 4角が全走 `None` なら `jyuni_3c` を使い `corner_source == "jyuni_3c"` |
| `test_falls_back_to_jyuni_2c` | 3角も無ければ `jyuni_2c` |
| `test_all_none_when_no_corner_data` | どれも無ければ全項目 `None`、`corner_pos_starts == 0`、`corner_source is None` |
| `test_does_not_mix_sources` | 一度 `jyuni_4c` を選んだら、4角が無い走は集計から外す（3角で埋めない） |
| `test_skips_runs_without_field_size` | `syusso_tosu` が `None` か 0 の走は外す |

**輸送**

| テスト | 内容 |
|---|---|
| `test_base_from_tozai_cd` | `"1"` → 美浦、`"2"` → 栗東 |
| `test_base_none_for_non_trainee_centre` | `"0"` / `"3"` / `"4"` → `None`。`estimated_transport_km` も `None` |
| `test_estimated_km_from_table` | 美浦 → 中山 が `TRANSPORT_KM` の値と一致 |
| `test_stay_status_defaults_to_unknown` | 既定で `"unknown"`。遠隔開催でも推測しない |
| `test_stay_status_accepts_confirmed_value` | `stay_status="stay"` を渡せばそのまま |
| `test_past_long_transport_counts_only_over_threshold` | `LONG_TRANSPORT_KM=300` を超える過去走だけ集計 |
| `test_past_long_transport_reports_starts_even_when_zero` | 該当0件で `starts=0`, 平均と最小が `None` |
| `test_past_long_transport_uses_run_time_affiliation` | 過去走の `tozai_cd`（そのレース時点）で距離を出す。現在の所属で計算しない |
| `test_past_long_transport_skips_missing_weight_change` | `zogen_sa_kg is None` の走は平均に入れないが `starts` には数える |

- [ ] **Step 2: 実装する**

`running_style` の「情報源を混ぜない」部分だけ全文（ここが唯一非自明）:

```python
_CORNER_SOURCES = ("jyuni_4c", "jyuni_3c", "jyuni_2c")


def running_style(runs, *, recent_runs=config.DEFAULT_RECENT_RUNS) -> RunningStyle:
    """4角の相対位置から脚質傾向を出す。

    4角が取れないコース（コーナー数が少ない）のために 3角→2角へ落とすが、
    走ごとに別の情報源を混ぜない。一度 jyuni_4c を選んだら、4角が無い走は
    集計から外す。混ぜると「同じ指標」として比較できなくなるため。
    どの列を使ったかは corner_source に残す。
    """
    usable = [run for run in runs
              if run.race.is_flat and run.result.counts_as_starter
              and run.race.syusso_tosu]
    usable = usable[:recent_runs]

    for source in _CORNER_SOURCES:
        positions = [
            getattr(run.result, source) / run.race.syusso_tosu
            for run in usable
            if getattr(run.result, source)
        ]
        if positions:
            mean = sum(positions) / len(positions)
            stdev = statistics.stdev(positions) if len(positions) > 1 else None
            return RunningStyle(
                corner_pos_mean=mean,
                corner_pos_stdev=stdev,
                corner_pos_starts=len(positions),
                corner_source=source,
            )
    return RunningStyle(None, None, 0, None)
```

`transport` の過去走の輸送距離は、**その走の `tozai_cd`** と **その走の `jyo_cd`** から引く。現在の所属で過去の輸送を計算しない（転厩があるため）。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/features/derive.py tests/test_derive.py
git commit -m "feat(features): condition records, recent form, running style, transport"
```

---

## Task 11: PredictionContext と state 組み立て

**Files:**
- Create: `src/keiba/features/context.py`, `src/keiba/features/summary.py`
- Create: `tests/test_context.py`, `tests/test_summary.py`

**Interfaces:**
- Consumes: `keiba.features.ability`, `keiba.features.derive`, `keiba.input.match`, `keiba.data.repository.Repository`
- Produces:
  - `context.HorseFeatures`（`frozen dataclass`: `umaban`, `bamei`, `ketto_num: str | None`, `entry: HorseEntry`, `ability: AbilitySummary`, `field_rank: float | None`, `recent: RecentForm`, `by_condition: dict[str, ConditionRecord]`, `running_style: RunningStyle`, `transport: Transport`, `missing: tuple[str, ...]`, `notes: tuple[str, ...]`）
  - `context.PredictionContext`（`frozen dataclass`: `race: RaceInput`, `horses: tuple[HorseFeatures, ...]`, `data_version: str`, `data_fetched_at: datetime`, `notes: tuple[str, ...]`, `pace_definition: dict | None = None`）。メソッド `horse(umaban: str) -> HorseFeatures`
  - `context.build_context(race_input: RaceInput, repo: Repository, *, pace_definition: dict | None = None) -> PredictionContext`（`as_of` は `race_input.prediction_cutoff` を使う）
  - `summary.build_race_state(ctx: PredictionContext) -> dict`
  - `summary.build_horse_state(ctx: PredictionContext, umaban: str, *, race_context: dict | None = None) -> dict`
  - `summary.horse_criteria(ctx: PredictionContext) -> dict[str, str]`（第1層 `Choice` の `criteria`）
  - `summary.data_quality(ctx: PredictionContext, umabans: Sequence[str] | None = None) -> dict`

- [ ] **Step 1: テストを書く**

`tests/test_context.py`:

| テスト | 内容 |
|---|---|
| `test_uses_prediction_cutoff_as_as_of` | `repo.past_runs` が `as_of=race_input.prediction_cutoff` で呼ばれる |
| `test_unregistered_horse_has_empty_features` | `ketto_num is None` の馬は `ability.runs_used == 0`、`missing` に項目が入る |
| `test_unregistered_horse_notes_reason` | `notes` に「マスタ未登録」が入る |
| `test_field_rank_computed_across_all_horses` | 全馬の `recent_margin_sec_wavg` から順位が付く |
| `test_missing_lists_paths_not_values` | `missing` が `("ability", "by_condition")` のようなパス名 |
| `test_changed_entry_is_noted` | `EntryRow.changed_after_entry` が真の馬は `notes` に記録される |

`tests/test_summary.py` — **未来情報の混入防止テストをここに置く**。

| テスト | 内容 |
|---|---|
| `test_race_state_uses_spec_key_names` | トップレベルのキーが `prediction_cutoff` / `race` / `horses` / `data_quality`（spec §5.1） |
| `test_horse_state_uses_spec_key_names` | `prediction_cutoff` / `race` / `target_horse` / `opponents_summary` / `race_context` / `data_quality`（spec §6.1） |
| `test_all_horses_share_the_same_key_set` | `horses` の各値の `set(keys)` が全馬で同一（spec §5.1「同じ書式で入れる」） |
| `test_missing_values_stay_null_not_zero` | 未登録馬の `ability.recent_margin_sec_wavg` が `None`（`0` でない） |
| `test_aggregates_always_carry_starts` | `by_condition` の各区分に `starts` が必ずある |
| `test_target_race_results_never_appear_in_state` | 対象レースの確定着順・走破タイム・オッズ・人気の値を `json.dumps(state)` の文字列から検索して1件も出ないこと |
| `test_future_races_never_appear_in_state` | `as_of` より後のレース名・日付が `state` に出ない |
| `test_state_has_no_cumulative_um_aggregates` | `state` のキーに `ruikei` / `chaku_kaisu` / `kyakusitu` が現れない |
| `test_state_is_json_serialisable` | `json.dumps(state, ensure_ascii=False)` が通る（`datetime` は ISO 文字列にしてある） |
| `test_transport_note_marks_estimate` | `transport.estimated_note` に「推定」と「実際の経路・輸送日を示すものではない」相当の文言が入る（spec §4.2） |
| `test_transport_stay_status_unknown_by_default` | 遠隔開催でも `"unknown"` |
| `test_opponents_summary_includes_target_horse` | 相手要約に対象馬自身も含まれる（相手の中での位置が分かるように） |
| `test_opponents_summary_is_short` | 相手要約の各馬のキー数が `horses` の各馬より少ない（`ability` と `running_style` だけ） |
| `test_race_context_omitted_when_no_pace` | `race_context=None` のとき `state` にキーが無い |
| `test_race_context_keeps_all_three_pace_probabilities` | `pace` を渡すと3区分すべての確率が入り、最大のものに置き換わっていない（spec §5.3） |
| `test_horse_criteria_keys_are_umaban` | `criteria` のキーが馬番の文字列 |
| `test_horse_criteria_values_identify_the_horse` | 値に馬名・性齢・斤量が入る |
| `test_data_quality_lists_missing_per_horse` | `data_quality["missing"]["4"]` に項目名が入る |
| `test_data_quality_carries_data_version_and_fetched_at` | `data_version` と `data_fetched_at` が入る |

`test_target_race_results_never_appear_in_state` の書き方（このテストが本プロジェクトの要）:

```python
def test_target_race_results_never_appear_in_state(repo_with_finished_race):
    """対象レースの結果が state に混ざらないことを、文字列検索で確かめる。

    型で防いでいるが、dict を組み立てる過程で取り違える余地は残る。
    JSON にしてから結果の実値を検索し、1件も出ないことを確認する。
    """
    repo, race_input, sentinel = repo_with_finished_race
    ctx = build_context(race_input, repo)
    blob = json.dumps(build_race_state(ctx), ensure_ascii=False, default=str)
    for value in sentinel.values():      # 確定着順・走破タイム・オッズ・人気
        assert str(value) not in blob, f"{value} が state に混入している"
```

フィクスチャ `repo_with_finished_race` は、対象レースに**特徴的な値**（例: 走破タイム `999.9`、オッズ `987.6`、人気 `77`）を入れた結果行を持たせ、それらが出ないことを確認する。普通の値だと偶然一致してしまうため。

- [ ] **Step 2: 実装する**

`build_race_state` の `horses` は全馬で同一のキー集合にする。`None` でもキーを消さない。

`datetime` / `date` は ISO 8601 文字列にしてから dict に入れる（`json.dumps` が通るように）。

`transport` の説明文（`config.py` に定数として置く）:

```python
TRANSPORT_ESTIMATE_NOTE = (
    "トレセンから競馬場までの代表的な陸路距離の概数。"
    "実際の出発地・移動経路・輸送日・現地滞在を示すものではない。"
)
PAST_TRANSPORT_NOTE = (
    "過去に推定輸送距離が長かった走での馬体重増減。"
    "体重変化の原因が輸送であることを示すものではない。"
)
```

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/features/context.py src/keiba/features/summary.py tests/test_context.py tests/test_summary.py
git commit -m "feat(features): build prediction context and Jev state dicts"
```

---

# フェーズ2 続き: Jev への問い合わせと第3層

## Task 12: 質問定義とペース区分

**Files:**
- Create: `src/keiba/predict/__init__.py`, `src/keiba/predict/questions.py`
- Create: `src/keiba/features/pace.py`
- Create: `tests/test_questions.py`, `tests/test_pace.py`

**Interfaces:**
- Consumes: `typesafe_sdk.Choice`, `keiba.features.summary.horse_criteria`, `keiba.config.PACE_MIN_RACES`
- Produces（`features/pace.py`）:
  - `PACE_BANDS = ("fast", "standard", "slow")`
  - `STRATA = (("track_kind", "distance_band", "jyo_cd"), ("track_kind", "distance_band"), ("track_kind",))`（層を段階的に粗くする順）
  - `distance_band(kyori_m: int) -> int`（`kyori_m` を 400m 刻みのバンドに丸めたキー。`±200m` 幅の層になる）
  - `fit_pace_definition(races: Sequence[RaceRow], *, target: RaceInput, min_races: int = config.PACE_MIN_RACES) -> dict | None`
  - `classify_pace(first_3f_sec: float | None, definition: dict) -> str | None`（確定結果との照合用）
- Produces（`predict/questions.py`）:
  - `PROMPT_VERSION = "1"`
  - `LAYER1_WINNER_INSTRUCTIONS: str`（spec §8.1 の文面をそのまま）
  - `LAYER1_PACE_INSTRUCTIONS: str`（spec §8.2）
  - `PACE_CHOICES: dict[str, str]`（`slow` / `standard` / `fast` とその説明）
  - `AxisSpec`（`frozen dataclass`: `instructions: str`, `criteria: dict[str, str]`, `positive_key: str | None`, `negative_key: str | None`, `insufficient_key: str`）
  - `AXIS_SPECS: dict[str, AxisSpec]`（`distance_fit` / `course_going_fit` / `relative_ability` / `current_form` / `transport_concern` の5軸）
  - `AXIS_NAMES: tuple[str, ...]`
  - `layer1_questions(ctx: PredictionContext, *, with_pace: bool = False) -> dict[str, Choice]`
  - `layer2_questions() -> dict[str, Choice]`
  - `feature_names_for_axes() -> tuple[str, ...]`（`distance_fit_signed`, `distance_fit_insufficient`, … の10列。順序を固定する）

- [ ] **Step 1a: ペース区分のテストを書く**

`tests/test_pace.py`。`first_3f_sec` は**小さいほど速い**ので、下位1/3 が `fast`、上位1/3 が `slow`。

| テスト | 内容 |
|---|---|
| `test_thresholds_split_at_tertiles` | `first_3f_sec` が 30.0〜39.0 の 90 レースで、閾値が概ね 33.0 と 36.0 |
| `test_smaller_first_3f_is_fast` | 閾値 33.0 / 36.0 のとき `classify_pace(32.0, d) == "fast"`、`classify_pace(37.0, d) == "slow"` |
| `test_boundary_goes_to_faster_band` | ちょうど閾値の値は速い側（`<=` で判定）。区分が実行ごとに変わらないように決めておく |
| `test_uses_finest_stratum_when_enough_races` | 同じ `track_kind` × 距離帯 × 競馬場 が 30 レース以上なら、その層の閾値を使う |
| `test_falls_back_when_stratum_too_small` | 29 レースしかなければ競馬場を落とした層に落ち、`stratum` にそれが記録される |
| `test_falls_back_twice_when_still_too_small` | 距離帯も落として `track_kind` だけになる |
| `test_returns_none_when_no_stratum_has_enough` | どの層も足りなければ `None`（推測で閾値を作らない） |
| `test_definition_records_stratum_and_count` | `stratum` と `race_count` が入る（spec §8.2「どの粒度の閾値だったか後から分かるように」） |
| `test_excludes_jump_races` | 障害レースは `first_3f_sec` が設定されないので除外 |
| `test_excludes_races_without_first_3f` | `first_3f_sec is None` の行は数えない |
| `test_classify_returns_none_for_missing_value` | `classify_pace(None, d) is None` |
| `test_distance_band_groups_within_400m` | 1800m と 2000m と 2100m が同じバンド、1600m は別（`±200m` 幅） |

実装で判断が入るのは「境界をどちらに寄せるか」だけなので全文を載せる。

```python
def classify_pace(first_3f_sec, definition):
    """前半3ハロンの実測値をペース区分に落とす。確定結果との照合に使う。

    first_3f_sec は小さいほど速い。閾値ちょうどの値は速い側に寄せる（<=）。
    寄せ方を決めておかないと、同じデータで区分が変わりうる。
    """
    if first_3f_sec is None:
        return None
    if first_3f_sec <= definition["fast_max"]:
        return "fast"
    if first_3f_sec <= definition["standard_max"]:
        return "standard"
    return "slow"
```

`fit_pace_definition` が返す dict:

```python
{
    "fast_max": 33.0,          # これ以下が fast
    "standard_max": 36.0,      # これ以下が standard、超えたら slow
    "stratum": {"track_kind": "turf", "distance_band": 2000, "jyo_cd": "06"},
    "race_count": 48,
    "note": "学習期間のレースの前半3ハロン(RA.HaronTimeS3)の三分位。"
            "小さいほど速い。確定結果と照合できる定義。",
}
```

`stratum` には**実際に使った層のキーだけ**を入れる（落とした層のキーは入れない）。どの粒度だったか一目で分かるようにするため。

- [ ] **Step 1b: 質問定義のテストを書く**

| テスト | 内容 |
|---|---|
| `test_layer1_has_winner_question` | `layer1_questions(ctx)` のキーが `{"winner"}` |
| `test_layer1_adds_pace_when_requested` | `with_pace=True` でキーが `{"winner", "pace"}` |
| `test_winner_criteria_keys_are_umaban` | `criteria` のキーが出走馬の馬番と一致 |
| `test_winner_criteria_within_255_choices` | 選択肢が255以下（フルゲート18頭でも余裕。上限は SDK の制約） |
| `test_winner_instructions_mention_starts_count` | プロンプトに「出走数」が含まれる（spec §5.2「成績と出走数の両方を参照」） |
| `test_winner_instructions_forbid_treating_missing_as_signal` | プロンプトに「欠損」が含まれる |
| `test_pace_criteria_has_three_bands` | `slow` / `standard` / `fast` の3つ |
| `test_layer2_has_five_axes` | キーが5軸ぴったり |
| `test_every_axis_criteria_is_a_dict` | `Choice` なので `criteria` は dict（`Score` のリストではない） |
| `test_every_axis_has_insufficient_option` | 全軸に「判断材料不足」の選択肢がある |
| `test_insufficient_and_neutral_are_separate` | `distance_fit` に `neutral` と `insufficient` の両方がある（spec §6.2） |
| `test_transport_concern_has_three_options` | `concern` / `no_evidence` / `insufficient` の3つ（spec §6.2 の表通り） |
| `test_relative_ability_uses_superiority_keys` | `superior` / `comparable` / `inferior` / `insufficient` |
| `test_axis_criteria_descriptions_are_non_empty` | 全軸・全選択肢の説明が空でない |
| `test_axis_criteria_mention_experience_not_being_unfitness` | 各軸の `insufficient` の説明に「経験」か「材料」が含まれ、未経験を不適性と書いていない |
| `test_feature_names_are_stable_and_ten` | `feature_names_for_axes()` が10要素で、呼ぶたび同じ順序 |
| `test_prompt_version_is_a_string` | `PROMPT_VERSION` が文字列（記録に載せるため） |

- [ ] **Step 2: 実装する**

`AXIS_SPECS` の全文（spec §6.2 の表と説明例に従う）。共通ルールを全軸で守る: 質問文と選択肢の説明を一致させる／未経験を不適性と判定しない／「判断材料不足」と「中立」を分ける／出走数を確認できる形にする。

```python
_STANDARD_KEYS = ("positive", "neutral", "negative", "insufficient")

AXIS_SPECS: dict[str, AxisSpec] = {
    "distance_fit": AxisSpec(
        instructions=(
            "target_horse の過去の距離別成績と近走内容に照らして、"
            "今回の距離は能力発揮にとってどの区分の材料ですか。\n"
            "by_condition の exact_distance と same_distance は、"
            "成績と starts（出走数）の両方を見てください。\n"
            "その距離での出走経験がないことを、不適性の材料として扱わないでください。"
        ),
        criteria={
            "positive": "今回に近い距離で、比較可能な条件下の好成績が裏付けられている。",
            "neutral": "比較可能な材料はあるが、今回の距離を明確な有利・不利とする材料が乏しい。",
            "negative": "比較可能な実績に、今回の距離で能力発揮を妨げる懸念が認められる。",
            "insufficient": "比較に使える出走実績や判断材料が乏しい。",
        },
        positive_key="positive", negative_key="negative",
        insufficient_key="insufficient",
    ),
    "course_going_fit": AxisSpec(
        instructions=(
            "target_horse のコース別・馬場状態別の実績に照らして、"
            "今回の競馬場、芝・ダート、馬場状態はどの区分の材料ですか。\n"
            "by_condition の same_venue、same_track_kind、same_baba を、"
            "成績と starts（出走数）の両方で見てください。\n"
            "race.baba が入力されていない場合は、馬場状態を判断材料にしないでください。\n"
            "その条件での出走経験がないことを、不適性の材料として扱わないでください。"
        ),
        criteria={
            "positive": "今回と比較可能なコース・馬場状態での好成績が裏付けられている。",
            "neutral": "比較可能な材料はあるが、今回の条件を明確な有利・不利とする材料が乏しい。",
            "negative": "比較可能な実績に、今回のコースまたは馬場状態での懸念が認められる。",
            "insufficient": "比較に使える出走実績や判断材料が乏しい。馬場状態が未入力の場合も含む。",
        },
        positive_key="positive", negative_key="negative",
        insufficient_key="insufficient",
    ),
    "relative_ability": AxisSpec(
        instructions=(
            "target_horse の近走の対戦水準と内容を opponents_summary の全頭と比較して、"
            "実績上の能力はどの区分ですか。\n"
            "ability の recent_margin_sec_wavg（直近走の加重平均着差、小さいほど優秀）と "
            "recent_class_level_wavg（クラスの水準）を併せて見てください。\n"
            "着差だけ、クラスだけで判断しないでください。"
            "低いクラスでの小さい着差と、高いクラスでの大きい着差を区別してください。\n"
            "runs_used が少ない馬を、それだけで劣位としないでください。"
        ),
        criteria={
            "superior": "比較可能な実績で、相手の多くより高い能力が裏付けられている。",
            "comparable": "相手と同程度で、優劣を決める材料が乏しい。",
            "inferior": "比較可能な実績で、相手の多くより低い能力が示されている。",
            "insufficient": "比較に使える出走実績や判断材料が乏しい。",
        },
        positive_key="superior", negative_key="inferior",
        insufficient_key="insufficient",
    ),
    "current_form": AxisSpec(
        instructions=(
            "recent の days_since_last_run（前走からの日数）、近走の着差の推移、"
            "last_run の馬体重に照らして、今回の状態にはどの区分の材料がありますか。\n"
            "馬体重は前走時点の値です。今回の馬体重は予想締切時点で判明していません。\n"
            "日数が長いこと自体を不調の材料として扱わないでください。"
            "休養明けで好走する馬もいます。"
        ),
        criteria={
            "positive": "近走の内容または間隔に、状態が整っていることを示す材料がある。",
            "neutral": "材料はあるが、状態を明確な上向き・下降とする根拠が乏しい。",
            "negative": "近走の内容または間隔に、状態面の懸念を示す材料がある。",
            "insufficient": "状態を判断できる近走や確認済みの情報が乏しい。",
        },
        positive_key="positive", negative_key="negative",
        insufficient_key="insufficient",
    ),
    "transport_concern": AxisSpec(
        instructions=(
            "transport の確認済み情報から、今回の輸送についてどの区分ですか。\n"
            "estimated_transport_km は推定値で、実際の出発地・経路・輸送日を"
            "示すものではありません。\n"
            "stay_status が unknown の場合、当日輸送とも現地滞在とも決めつけないでください。\n"
            "past_long_transport の馬体重増減は starts（該当走数）と併せて見てください。"
            "体重変化の原因が輸送であるとは限りません。"
        ),
        criteria={
            "concern": "確認済みの輸送・滞在情報と過去の関連実績に、今回の輸送への懸念を示す材料がある。",
            "no_evidence": "確認できる範囲で、今回の輸送への懸念を示す材料がない。",
            "insufficient": "輸送・滞在の確認済み情報や関連する過去実績が乏しく、判断できない。",
        },
        # 2値+不足なので差を取らない。signed は P(concern) そのものになる。
        positive_key="concern", negative_key=None,
        insufficient_key="insufficient",
    ),
}

AXIS_NAMES: tuple[str, ...] = tuple(AXIS_SPECS)
```

第2層の特徴変換（Task 14）はこの `positive_key` / `negative_key` を見て `signed` を作る。`negative_key is None` の軸（`transport_concern`）は差を取らず `P(positive_key)` をそのまま使う。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/predict/__init__.py src/keiba/predict/questions.py tests/test_questions.py
git commit -m "feat(predict): layer 1 and layer 2 question definitions"
```

---

## Task 13: 第1層・第2層の実行（jev_client への追加を含む）

**Files:**
- Modify: `src/jev_client.py`（`ask_parallel` に `return_exceptions` を追加。**このプロジェクトで jev_client に触るのはここだけ**）
- Create: `src/keiba/predict/layers.py`
- Create: `tests/test_layers.py`
- Modify: `tests/test_jev_client.py`（`return_exceptions` のテストを追加）

**Interfaces:**
- Consumes: `jev_client.ask`, `jev_client.ask_parallel`, `jev_client.Job`, `jev_client.JevRequestError`, `keiba.predict.questions`, `keiba.features.summary`
- Produces:
  - `Layer1Result`（`frozen dataclass`: `win_probabilities: dict[str, float]`, `choice: str`, `confidence: float`, `pace_probabilities: dict[str, float] | None`, `model: str`, `input_tokens: int`）
  - `AxisAnswer`（`frozen dataclass`: `choice: str`, `probabilities: dict[str, float]`, `confidence: float`）
  - `Layer2Result`（`frozen dataclass`: `answers: dict[str, dict[str, AxisAnswer]]`, `failures: dict[str, str]`, `input_tokens: int`）。プロパティ `complete: bool`（`failures` が空）
  - `run_layer1(client, ctx: PredictionContext, *, with_pace: bool = False) -> Layer1Result`
  - `run_layer2(client, ctx: PredictionContext, layer1: Layer1Result, *, max_workers: int = 8, allow_partial: bool = False) -> Layer2Result`
  - `Layer2IncompleteError(RuntimeError)`（`allow_partial=False` で1頭でも失敗したときに投げる。`failures: dict[str, str]` を持つ）

- [ ] **Step 1: `jev_client` のテストを先に書く**

`tests/test_jev_client.py` に追記:

```python
class TestAskParallelReturnExceptions:
    def test_default_raises_on_first_failure(self):
        """既定の挙動を変えない。既存の呼び出し側が壊れないこと。"""
        client = _FailingClient(fail_on={"2"})
        with pytest.raises(jev_client.JevRequestError):
            jev_client.ask_parallel(client, _two_jobs())

    def test_return_exceptions_puts_error_in_result(self):
        client = _FailingClient(fail_on={"2"})
        results = jev_client.ask_parallel(client, _two_jobs(), return_exceptions=True)
        assert set(results) == {"1", "2"}
        assert not isinstance(results["1"], jev_client.JevRequestError)
        assert isinstance(results["2"], jev_client.JevRequestError)

    def test_return_exceptions_message_names_the_job(self):
        client = _FailingClient(fail_on={"2"})
        results = jev_client.ask_parallel(client, _two_jobs(), return_exceptions=True)
        assert "[2]" in str(results["2"])

    def test_return_exceptions_with_all_failing(self):
        client = _FailingClient(fail_on={"1", "2"})
        results = jev_client.ask_parallel(client, _two_jobs(), return_exceptions=True)
        assert all(isinstance(v, jev_client.JevRequestError) for v in results.values())
```

- [ ] **Step 2: `jev_client.ask_parallel` を直す**

`pool.map` は例外を伝播させるので、`return_exceptions=True` のときは各ジョブを個別に捕まえる。

```python
def ask_parallel(
    client: Any,
    jobs: Mapping[str, Job],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float | None = None,
    return_exceptions: bool = False,
) -> dict[str, Any]:
    """複数の state を並列に問い合わせ、ジョブ名をキーにした辞書で返す。

    第2層 (馬単位) 用。state が馬ごとに違うためリクエストを分けざるを得ないが、
    1頭分の中の質問はすべて1リクエストにまとめること。

    ``return_exceptions=True`` のとき、失敗したジョブの値は ``JevRequestError``
    のインスタンスそのものになる。1頭の失敗で全体を止めたくない用途
    (バックテスト) 向け。既定は False で、最初の失敗をそのまま投げる。
    """
    if not jobs:
        return {}

    def run(item: tuple[str, Job]) -> tuple[str, Any]:
        name, job = item
        try:
            return name, ask(client, job.state, job.questions, timeout=timeout)
        except JevRequestError as error:
            wrapped = JevRequestError(f"[{name}] {error}", error.original)
            if return_exceptions:
                return name, wrapped
            raise wrapped from error

    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        return dict(pool.map(run, jobs.items()))
```

`__all__` は変更不要（`ask_parallel` は既に入っている）。

- [ ] **Step 3: `layers.py` のテストを書く**

偽クライアントを使う。API に触らない。

```python
class FakeResponse:
    def __init__(self, choices=None, model="jev-1.13.0", input_tokens=100):
        self.choices = choices or {}
        self.model = model
        self.usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=0)


class FakeChoice:
    def __init__(self, choice, probabilities, confidence):
        self.choice = choice
        self.probabilities = probabilities
        self.confidence = confidence
```

| テスト | 内容 |
|---|---|
| `test_layer1_returns_probabilities_for_every_horse` | 全馬の馬番がキーに入る |
| `test_layer1_probabilities_sum_to_one` | 合計が 1.0（±1e-6） |
| `test_layer1_fills_missing_horse_with_zero` | Jev が一部の馬を返さなかった場合、欠けた馬を `0.0` で埋め、`notes` に記録する（合計は再正規化しない） |
| `test_layer1_pace_is_none_when_disabled` | `with_pace=False` で `pace_probabilities is None` |
| `test_layer1_pace_keeps_all_three_bands` | `with_pace=True` で3区分すべて入る |
| `test_layer1_sends_single_request` | `with_pace=True` でもリクエストは1回（spec §2.2） |
| `test_layer1_wraps_api_error` | `JevRequestError` がそのまま上がる |
| `test_layer2_sends_one_request_per_horse` | 3頭で3リクエスト |
| `test_layer2_bundles_all_axes_in_one_request` | 1頭あたりの `questions` に5軸すべてが入る（軸ごとに分けない） |
| `test_layer2_collects_all_axes_per_horse` | `answers["1"]` に5軸すべて |
| `test_layer2_passes_pace_into_race_context` | `pace_probabilities` があれば各馬の state の `race_context` に入る |
| `test_layer2_raises_on_partial_failure_by_default` | 1頭失敗で `Layer2IncompleteError`。`failures` に馬番とメッセージ |
| `test_layer2_allows_partial_when_requested` | `allow_partial=True` で成功分だけ返り `failures` に失敗分、`complete is False` |
| `test_layer2_sums_input_tokens` | 各レスポンスの `input_tokens` の合計 |
| `test_layer2_empty_when_no_horses` | 0頭でリクエストを投げない |

- [ ] **Step 4: 実装する**

`run_layer1` の確率の扱いだけ全文。**足りない馬を 0 で埋めるが再正規化しない**のが判断点。

```python
def run_layer1(client, ctx, *, with_pace=False) -> Layer1Result:
    """レース単位で1リクエスト。全頭の勝率分布が1回で返る。"""
    response = ask(client, build_race_state(ctx),
                   questions.layer1_questions(ctx, with_pace=with_pace))
    winner = response.choices["winner"]

    # Jev は渡した選択肢の中だけを返すが、確率辞書に全馬が載る保証はない。
    # 欠けた馬は 0.0 で埋める。再正規化はしない——合計が 1.0 から外れていること
    # 自体が「分布が取れていない」という情報なので、割り算で隠さない。
    probabilities = {
        horse.umaban: float(winner.probabilities.get(horse.umaban, 0.0))
        for horse in ctx.horses
    }

    pace = None
    if with_pace and "pace" in response.choices:
        pace = {k: float(v) for k, v in response.choices["pace"].probabilities.items()}

    return Layer1Result(
        win_probabilities=probabilities,
        choice=winner.choice,
        confidence=float(winner.confidence),
        pace_probabilities=pace,
        model=response.model,
        input_tokens=response.usage.input_tokens or 0,
    )
```

`run_layer2` は `Job(state=build_horse_state(ctx, umaban, race_context=...), questions=layer2_questions())` を馬ごとに作り、`ask_parallel(..., return_exceptions=allow_partial)` で投げる。`allow_partial=False` のときは `return_exceptions=False` で投げたうえで、`JevRequestError` を捕まえて `Layer2IncompleteError` に変換する（どの馬が失敗したか分かる形にするため）。

- [ ] **Step 5: テスト全体を通す** → 全件 PASS。既存の `test_jev_client.py` 16件も通ること

```bash
git add src/jev_client.py src/keiba/predict/layers.py tests/test_layers.py tests/test_jev_client.py
git commit -m "feat(predict): run layer 1 and layer 2; add return_exceptions to ask_parallel"
```

---

## Task 14: 第3層（baseline と条件付きロジット）

**Files:**
- Create: `src/keiba/predict/integrate.py`
- Create: `tests/test_integrate.py`

**Interfaces:**
- Consumes: `keiba.predict.questions.AXIS_SPECS`, `keiba.predict.layers.Layer1Result`, `keiba.predict.layers.Layer2Result`, `keiba.features.context.HorseFeatures`
- Produces:
  - `PYTHON_FEATURE_NAMES: tuple[str, ...]`（spec §10.2 の8列。順序固定）
  - `axis_features(answers: Mapping[str, AxisAnswer]) -> dict[str, float]`（10列）
  - `python_features(hf: HorseFeatures, race: RaceInput) -> dict[str, float | None]`（8列。欠損は `None` のまま）
  - `expand_missing(values: Mapping[str, float | None]) -> dict[str, float]`（`None` を `0.0` にし、`<name>_missing` を `1.0` にする。値があれば `_missing` は `0.0`）
  - `feature_names(*, with_layer2: bool = True) -> tuple[str, ...]`（`logit_layer1` + 第2層10列 + Python 8列 + `_missing` 8列）
  - `build_features(ctx, layer1, layer2) -> dict[str, dict[str, float]]`（馬番 → 特徴名 → 値）
  - `Weights`（`frozen dataclass`: `weights_id`, `feature_names: tuple[str, ...]`, `weights: dict[str, float]`, `standardization: dict[str, tuple[float, float]]`, `calibration: dict | None`, `versions: dict[str, str]`, `lambda_: float`, `dropped_features: tuple[str, ...]`, `splits: dict[str, str]`）。メソッド `load(path) -> Weights` / `save(path) -> None` / `check_compatible() -> None`（`versions` の `feature_version` / `prompt_version` / `integration_version` が現在のコードと違えば `WeightsIncompatibleError`）
  - `WeightsIncompatibleError(RuntimeError)`
  - `logit(p: float, eps: float = 1e-6) -> float`
  - `softmax_by_race(scores: Mapping[str, float]) -> dict[str, float]`
  - `baseline_ranking(layer1: Layer1Result) -> list[str]`
  - `model_scores(features, weights: Weights) -> dict[str, float]`
  - `score_breakdown(features, weights: Weights) -> dict[str, dict[str, float]]`
  - `standardize(features, standardization) -> dict[str, dict[str, float]]`

- [ ] **Step 1: テストを書く**

| テスト | 内容 |
|---|---|
| `test_axis_features_signed_is_positive_minus_negative` | `{positive: 0.6, neutral: 0.2, negative: 0.1, insufficient: 0.1}` → `signed == 0.5` |
| `test_axis_features_insufficient_is_kept_separately` | `insufficient == 0.1` |
| `test_axis_features_relative_ability_uses_superior_inferior` | `superior 0.5 / inferior 0.2` → `0.3` |
| `test_axis_features_transport_uses_concern_directly` | `concern 0.4` → `signed == 0.4`（差を取らない） |
| `test_axis_features_returns_ten_columns` | 10列 |
| `test_axis_features_ignores_confidence` | `confidence` を変えても特徴値が変わらない（spec §9） |
| `test_python_features_keeps_none` | `recent_margin_sec_wavg is None` なら `None` のまま（`0.0` にしない） |
| `test_expand_missing_sets_indicator` | `None` → 値 `0.0` + `_missing 1.0`。値あり → `_missing 0.0` |
| `test_expand_missing_distinguishes_zero_from_missing` | 値が `0.0` の馬と `None` の馬で `_missing` が違う |
| `test_logit_clips_at_zero_and_one` | `logit(0.0)` と `logit(1.0)` が有限値（`inf` でない） |
| `test_logit_is_monotonic` | `logit(0.1) < logit(0.5) < logit(0.9)` |
| `test_logit_of_half_is_zero` | `logit(0.5) == 0.0`（±1e-9） |
| `test_softmax_sums_to_one_for_8_horses` | 8頭で合計 1.0 |
| `test_softmax_sums_to_one_for_18_horses` | 18頭でも合計 1.0（頭数に依存しない。spec §13 最終項） |
| `test_softmax_handles_large_scores_without_overflow` | スコアに `1000.0` があっても `inf` / `nan` にならない |
| `test_softmax_preserves_order` | スコアの順序が確率の順序と一致 |
| `test_baseline_ranking_is_descending_probability` | 第1層の確率の降順 |
| `test_baseline_ranking_breaks_ties_by_umaban` | 同確率なら馬番の昇順（結果が実行ごとに変わらないように） |
| `test_model_scores_is_weighted_sum` | 手計算した値と一致 |
| `test_model_scores_ignores_features_not_in_weights` | `weights` に無い特徴は寄与しない |
| `test_score_breakdown_sums_to_score` | 内訳の合計が `model_scores` の値と一致（±1e-9） |
| `test_score_breakdown_has_one_entry_per_weight` | 内訳のキーが `weights.feature_names` と一致 |
| `test_standardize_applies_mean_and_std` | `(x - mean) / std` |
| `test_standardize_skips_logit_and_indicators` | `logit_layer1` と `*_missing` と `*_insufficient` は変換しない（spec §11.1.1） |
| `test_standardize_leaves_feature_when_std_is_zero` | 標準偏差 0 の特徴は変換せずそのまま（学習側で落とす） |
| `test_weights_roundtrip_through_json` | `save` → `load` で値が一致 |
| `test_weights_check_compatible_passes_for_current_versions` | 現在の版なら例外なし |
| `test_weights_check_compatible_rejects_old_prompt_version` | `prompt_version="0"` なら `WeightsIncompatibleError`。メッセージに `prompt_version` が入る |
| `test_weights_check_compatible_rejects_old_feature_version` | 同様 |
| `test_feature_names_without_layer2_excludes_axes` | `with_layer2=False` で第2層10列が入らない |
| `test_feature_names_is_stable` | 2回呼んで同じ順序 |

- [ ] **Step 2: 実装する**

`softmax_by_race` と `logit` は全文（数値安定性が判断点）。

```python
def logit(p: float, eps: float = 1e-6) -> float:
    """確率を logit に。p=0 / p=1 で -inf / +inf にならないよう clip する。"""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def softmax_by_race(scores: Mapping[str, float]) -> dict[str, float]:
    """レース内 softmax。頭数が何頭でも合計 1.0 を保つ。

    これが spec §13 の「第1層と第2層のスケールの揃え方」への答え。
    8頭でも18頭でも分母がそのレースの出走馬だけなので、
    重み付き和のように頭数で釣り合いが変わることがない。

    exp する前に最大値を引く（オーバーフロー対策）。
    """
    if not scores:
        return {}
    highest = max(scores.values())
    exponentials = {k: math.exp(v - highest) for k, v in scores.items()}
    total = sum(exponentials.values())
    return {k: v / total for k, v in exponentials.items()}
```

`expand_missing` の趣旨をコメントに書く:

```python
def expand_missing(values: Mapping[str, float | None]) -> dict[str, float]:
    """欠損を 0 埋めしつつ、欠損だったことを指示変数で残す。

    0 埋めだけでは「値が 0」と「欠損」が区別できず、spec §4.4 の
    「欠損値を0や不適性に置き換えない」に反する。値は 0 にするが
    <name>_missing を 1 にして、重みが別に付くようにする。
    """
```

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/predict/integrate.py tests/test_integrate.py
git commit -m "feat(predict): conditional-logit integration with race-wise softmax"
```

---

## Task 15: 予測記録の保存と利用者向けビュー

**Files:**
- Create: `src/keiba/store/__init__.py`, `src/keiba/store/records.py`
- Create: `tests/test_records_store.py`

**Interfaces:**
- Consumes: `keiba.config`, `keiba.predict.questions.PROMPT_VERSION`
- Produces:
  - `HorsePrediction`（`frozen dataclass`: `umaban`, `bamei`, `rank_baseline: int`, `rank_model: int | None`, `layer1_win_prob: float`, `model_score: float | None`, `model_win_prob: float | None`, `calibrated_win_prob: float | None`, `score_breakdown: dict[str, float]`, `layer2: dict[str, dict]`, `positives: tuple[str, ...]`, `negatives: tuple[str, ...]`, `insufficient: tuple[str, ...]`）
  - `PredictionRecord`（`frozen dataclass`: `record_id`, `race_id: str | None`, `predicted_at: datetime`, `prediction_cutoff: datetime`, `n: int`, `versions: dict[str, str]`, `horses: tuple[HorsePrediction, ...]`, `layer1: dict`, `data_quality: dict`, `usage: dict`, `status: str`, `race_input: dict`）
  - `current_versions(*, data_version: str, model: str, weights_id: str | None) -> dict[str, str]`
  - `save(record: PredictionRecord, *, data_dir: Path) -> Path`（`data/predictions/<YYYYMM>.jsonl` に追記）
  - `load_month(year_month: str, *, data_dir: Path) -> list[PredictionRecord]`
  - `to_user_view(record: PredictionRecord) -> dict`（上位 n 頭だけ。`state` 全体・生レスポンス・元 CSV 由来の生レコードを含めない）
  - `to_json(record) -> dict` / `from_json(payload) -> PredictionRecord`

- [ ] **Step 1: テストを書く**

| テスト | 内容 |
|---|---|
| `test_save_appends_one_line_per_record` | 2件保存でファイルが2行 |
| `test_save_uses_year_month_file` | `predicted_at` が 2026-03 なら `202603.jsonl` |
| `test_roundtrip_preserves_all_fields` | `save` → `load_month` で全項目一致 |
| `test_record_keeps_every_horse_not_just_top_n` | `n=3` でも18頭すべて保存される（spec §8「検証のため全頭の予測を保存」） |
| `test_versions_include_all_required_keys` | `data_version` / `schema_version` / `model` / `prompt_version` / `feature_version` / `integration_version` / `exception_rules_version` / `weights_id` |
| `test_calibrated_prob_is_none_until_verified` | 較正未検証なら `calibrated_win_prob is None`（spec §10.3） |
| `test_user_view_shows_only_top_n` | `n=3` で3頭 |
| `test_user_view_excludes_state` | 返り値の JSON 文字列に `"target_horse"` / `"opponents_summary"` / `"horses"` の state 構造が出ない |
| `test_user_view_excludes_raw_layer2_probabilities` | 生の確率分布を含めず、分類ラベルだけ（spec §8「評価理由は分類と独自集計に基づいて構成」） |
| `test_user_view_excludes_score_breakdown` | 内部の重みの内訳を返さない |
| `test_user_view_includes_prediction_time_and_data_update_time` | 予想時刻とデータ更新時刻が入る（spec §12.5） |
| `test_user_view_includes_insufficient_items` | 判断材料が不足している項目が入る（spec §8） |
| `test_user_view_labels_rank_as_likelihood_not_finish_order` | 順位のラベルに「着順」ではなく「勝ちやすさ」相当の語を使う（spec §8） |
| `test_record_id_is_unique_per_save` | 2回保存で `record_id` が異なる |
| `test_status_failed_is_saved` | `status="FAILED"` の記録も保存できる（spec §12.7） |

- [ ] **Step 2: 実装する**

`to_user_view` の除外リストを1か所に持つ。

```python
def to_user_view(record: PredictionRecord) -> dict:
    """利用者に返す形。内部記録とは別物にする。

    spec §8: 元CSV、過去走の生レコード、Jev へ送った state 全体、API の生レスポンスを
    利用者向けから取得できないようにする。評価理由は分類と独自集計に基づいて構成し、
    過去データの一覧をそのまま転載しない。

    ここで「含めるものを列挙する」方式を採る。除外リスト方式だと、
    項目が増えたときに漏れる。
    """
    top = sorted(record.horses, key=lambda h: h.rank_model or h.rank_baseline)[: record.n]
    return {
        "予想時刻": record.predicted_at.isoformat(timespec="seconds"),
        "予想締切": record.prediction_cutoff.isoformat(timespec="seconds"),
        "データ更新時刻": record.versions.get("data_fetched_at"),
        "上位n頭": record.n,
        "注意": "勝ちやすさの順位です。実際の着順の断定ではありません。",
        "予想": [
            {
                "予想順位": h.rank_model or h.rank_baseline,
                "馬番": h.umaban,
                "馬名": h.bamei,
                "プラス材料": list(h.positives),
                "マイナス材料": list(h.negatives),
                "判断材料が不足している項目": list(h.insufficient),
            }
            for h in top
        ],
    }
```

`positives` / `negatives` / `insufficient` は Task 16 のパイプラインで、第2層の分類と Python 側の集計からテンプレートで組む。Jev は自由文の理由を生成しないため（spec §8）。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/store tests/test_records_store.py
git commit -m "feat(store): prediction records and user-facing view"
```

---

## Task 16: パイプラインと predict CLI

**Files:**
- Create: `src/keiba/predict/pipeline.py`
- Create: `scripts/predict.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: すべての前タスク
- Produces:
  - `EvidenceBuilder`: `build(hf: HorseFeatures, answers: Mapping[str, AxisAnswer] | None) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]`（`positives` / `negatives` / `insufficient`）
  - `predict(race_input: RaceInput, repo: Repository, client, *, with_layer2: bool = True, with_pace: bool = False, weights: Weights | None = None, allow_partial: bool = False) -> PredictionRecord`
  - `PredictionFailed(RuntimeError)`（入力不備・照合失敗・第2層の部分失敗）

- [ ] **Step 1: テストを書く**

偽クライアントと CSV フィクスチャで、API に触らず全体を回す。

| テスト | 内容 |
|---|---|
| `test_produces_record_with_every_horse` | 出走馬全頭が `record.horses` に入る |
| `test_baseline_ranking_present_without_weights` | `weights=None` で `rank_baseline` が入り `rank_model is None` |
| `test_model_ranking_present_with_weights` | `weights` を渡すと `rank_model` と `model_win_prob` が入る |
| `test_model_and_baseline_both_recorded` | 両方が同じ記録に入る（spec §10.1） |
| `test_skips_layer2_when_disabled` | `with_layer2=False` で第2層のリクエストが飛ばない |
| `test_fails_when_input_invalid` | `validate.check` が問題を返したら `PredictionFailed`。メッセージに問題が列挙される |
| `test_fails_when_name_match_ambiguous` | `AmbiguousMatchError` が `PredictionFailed` になる |
| `test_fails_on_partial_layer2_by_default` | 1頭失敗で `PredictionFailed`、記録を作らない（spec §12.7「不完全な評価を正常完了として返さない」） |
| `test_allows_partial_and_marks_status` | `allow_partial=True` で `status == "PARTIAL"`、失敗した馬が `data_quality` に入る |
| `test_evidence_includes_condition_counts` | `positives` の文言に出走数が入る（例「同距離 8戦3勝」） |
| `test_evidence_marks_insufficient_axes` | `insufficient` の確率が最大の軸が `insufficient` に入る |
| `test_evidence_marks_zero_start_aggregates` | `starts == 0` の集計が `insufficient` に入る |
| `test_evidence_is_built_from_classes_not_free_text` | `positives` の各要素が定型文（テンプレート由来）であること |
| `test_record_versions_include_weights_id` | `weights` を渡したとき `versions["weights_id"]` が入る |
| `test_rejects_incompatible_weights` | `prompt_version` が違う `weights` で `PredictionFailed` |
| `test_usage_sums_tokens_from_both_layers` | 第1層と第2層の `input_tokens` の合計 |
| `test_unregistered_horse_does_not_break_pipeline` | 未登録馬が混ざっても最後まで通り、`data_quality` に記録される |

- [ ] **Step 2: 実装する**

`predict` の流れ:

1. `validate.check(race_input)` → 問題があれば `PredictionFailed`
2. `match_horses(race_input, repo, as_of=race_input.prediction_cutoff)` → `AmbiguousMatchError` は `PredictionFailed` に変換
3. `weights.check_compatible()`（渡されていれば）
4. `build_context(race_input, repo, pace_definition=...)`
5. `run_layer1(client, ctx, with_pace=with_pace)`
6. `with_layer2` なら `run_layer2(client, ctx, layer1, allow_partial=allow_partial)`
7. `build_features` → `weights` があれば `standardize` → `model_scores` → `softmax_by_race` → `score_breakdown`
8. `baseline_ranking` と `model` の順位を両方付ける
9. `EvidenceBuilder` で `positives` / `negatives` / `insufficient` を組む
10. `PredictionRecord` を組んで返す（保存は呼び出し側＝CLI）

`EvidenceBuilder` の文言テンプレート例:

```python
# Jev は自由文の理由を生成しない（spec §8）。分類と自前集計から定型文で組む。
_CONDITION_LABELS = {
    "exact_distance": "同距離", "same_distance": "同距離帯",
    "same_venue": "同競馬場", "same_track_kind": "同トラック", "same_baba": "同馬場状態",
}

def _condition_phrase(key: str, record: ConditionRecord) -> str:
    """「同距離 8戦3勝(3着内5)」。必ず出走数を添える（spec §4.3）。"""
    return (f"{_CONDITION_LABELS[key]} {record.starts}戦{record.wins}勝"
            f"(3着内{record.top3})")
```

`scripts/predict.py` の引数:

```
--input race.json          利用者入力（必須）
--n 3                      上位何頭（既定 config.DEFAULT_TOP_N。入力側の top_n を上書き）
--layer2 / --no-layer2     第2層を使うか（既定 使う）
--pace / --no-pace         展開予測を第1層に含めるか（既定 含めない）
--weights <weights_id>     統合に使う重み（省略時は baseline のみ）
--allow-partial            第2層の部分失敗を許して PARTIAL で返す（既定 許さない）
--data-dir data            データ版の場所
--save / --no-save         予測記録を保存するか（既定 保存する）
```

出力は `to_user_view` の内容を人が読める形で表示し、`--save` のときは保存先パスも出す。失敗時は理由を日本語で出して exit 1。

- [ ] **Step 3: 実データで手動確認する**

Run: `.venv/Scripts/python.exe scripts/predict.py --input <用意した race.json> --no-layer2`
Expected: 上位3頭が出て exit 0。次に `--layer2` を付けて 5軸が付くことを確認する。

- [ ] **Step 4: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/predict/pipeline.py scripts/predict.py tests/test_pipeline.py
git commit -m "feat(predict): end-to-end pipeline and predict CLI"
```

---

# フェーズ3: 学習と較正

## Task 17: 条件付きロジットの最尤推定

**Files:**
- Create: `src/keiba/learn/__init__.py`, `src/keiba/learn/conditional_logit.py`
- Create: `tests/test_conditional_logit.py`

**Interfaces:**
- Consumes: `numpy`, `scipy.optimize`
- Produces:
  - `FitResult`（`frozen dataclass`: `weights: np.ndarray`, `feature_names: tuple[str, ...]`, `standardization: dict[str, tuple[float, float]]`, `dropped_features: tuple[str, ...]`, `lambda_: float`, `log_likelihood: float`, `n_races: int`, `n_rows: int`, `converged: bool`）
  - `neg_log_likelihood(w: np.ndarray, X: np.ndarray, group_starts: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]`（値と勾配）
  - `fit(X: np.ndarray, groups: np.ndarray, y: np.ndarray, *, feature_names: Sequence[str], l2: float = 0.1, standardize_mask: Sequence[bool] | None = None) -> FitResult`
  - `fit_standardization(X, feature_names, mask) -> tuple[dict[str, tuple[float, float]], tuple[str, ...]]`

- [ ] **Step 1: テストを書く**

**真の重みが既知の合成データで復元できること**が要（spec §11.1）。

| テスト | 内容 |
|---|---|
| `test_recovers_known_weights_from_synthetic_data` | 真の重み `[1.5, -0.8]` で 3000 レース（各8頭）を生成し、`l2=1e-6` で推定して各成分が ±0.15 以内 |
| `test_recovers_single_feature` | 1特徴でも復元できる |
| `test_gradient_matches_numerical_derivative` | `scipy.optimize.approx_fprime` との差が 1e-5 以内 |
| `test_likelihood_decreases_with_regularisation` | `l2` を上げると重みのノルムが小さくなる |
| `test_handles_variable_field_size` | 8頭と18頭が混在しても推定できる |
| `test_handles_dead_heat_as_shared_outcome` | 1着が2頭のレース（`y` が `0.5` ずつ）を受け付け、log-likelihood が有限 |
| `test_raises_on_race_without_winner` | どの行も `y > 0` でないレースがあれば `ValueError` |
| `test_raises_on_mismatched_lengths` | `X` と `y` の行数が違えば `ValueError` |
| `test_standardization_skips_masked_features` | `standardize_mask` が `False` の列は平均0分散1にしない |
| `test_drops_constant_features` | 分散 0 の列が `dropped_features` に入り、重みが 0 になる |
| `test_converged_flag_reflects_optimiser` | 収束したら `True` |
| `test_empty_input_raises` | 0行で `ValueError` |

合成データの作り方（テストに書く）:

```python
def make_synthetic(rng, *, n_races, field_size, true_w):
    """条件付きロジットの生成モデルそのままでデータを作る。

    各馬に特徴 x を引き、score = x·w、レース内 softmax の確率で勝ち馬を1頭選ぶ。
    この作り方なら最尤推定は真の w を復元できるはず。復元できなければ実装が誤り。
    """
    X, groups, y = [], [], []
    for race in range(n_races):
        x = rng.normal(size=(field_size, len(true_w)))
        scores = x @ true_w
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        winner = rng.choice(field_size, p=probabilities)
        for i in range(field_size):
            X.append(x[i])
            groups.append(race)
            y.append(1.0 if i == winner else 0.0)
    return np.array(X), np.array(groups), np.array(y)
```

- [ ] **Step 2: 実装する**

対数尤度と勾配を全文（ここが最も間違いやすい）。

```python
def neg_log_likelihood(w, X, group_starts, y, l2):
    """条件付きロジットの負の対数尤度と勾配。

        L(w) = Σ_races Σ_i y_i * (score_i - log Σ_j exp(score_j))

    y は勝ち馬に 1（同着なら 1/同着頭数）。レースごとに Σ y_i == 1 になる。
    勾配は Σ_races Σ_i (y_i - p_i) * x_i。p はレース内 softmax。
    L2 項は目的関数に + l2 * ||w||^2、勾配に + 2 * l2 * w。
    """
    scores = X @ w
    total = 0.0
    gradient = np.zeros_like(w)
    for start, end in zip(group_starts[:-1], group_starts[1:]):
        s = scores[start:end]
        shifted = s - s.max()                  # オーバーフロー対策
        exponentials = np.exp(shifted)
        log_denominator = np.log(exponentials.sum()) + s.max()
        yy = y[start:end]
        total += float((yy * (s - log_denominator)).sum())
        probabilities = exponentials / exponentials.sum()
        gradient += (yy - probabilities) @ X[start:end]
    return -total + l2 * float(w @ w), -gradient + 2.0 * l2 * w
```

`fit` は `groups` を昇順に並べ替えて `group_starts`（各レースの開始行インデックス + 末尾）を作り、`scipy.optimize.minimize(..., jac=True, method="L-BFGS-B")` を呼ぶ。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/learn/__init__.py src/keiba/learn/conditional_logit.py tests/test_conditional_logit.py
git commit -m "feat(learn): conditional logit MLE with analytic gradient"
```

---

## Task 18: 較正と fit_weights CLI

**Files:**
- Create: `src/keiba/learn/calibration.py`
- Create: `scripts/fit_weights.py`
- Create: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `sklearn.isotonic.IsotonicRegression`
- Produces:
  - `fit_calibration(probabilities: Sequence[float], outcomes: Sequence[float]) -> dict`（`{"method": "isotonic", "x": [...], "y": [...]}`）
  - `apply_calibration(calibration: dict, probabilities: Mapping[str, float]) -> tuple[dict[str, float], float]`（較正後の確率と、再正規化前の合計。合計が 1.0 から離れていれば較正が信頼できない兆候なので返して記録する）
  - `interpolate(calibration: dict, p: float) -> float`
- `scripts/fit_weights.py`: train / dev / calib / test に分けて学習・選択・較正し、`data/weights/<weights_id>.json` を書く

- [ ] **Step 1: テストを書く**

| テスト | 内容 |
|---|---|
| `test_fit_returns_monotonic_mapping` | `x` が昇順、`y` が非減少 |
| `test_fit_maps_overconfident_to_lower` | 予測 0.8 で実測 0.4 のデータを与えると `interpolate(cal, 0.8) < 0.8` |
| `test_interpolate_clamps_outside_range` | 学習範囲の外は端の値に張り付き、`[0, 1]` に収まる |
| `test_apply_returns_sum_before_renormalisation` | 較正後の生の合計を返す |
| `test_apply_renormalises_to_one` | 返る確率の合計が 1.0（±1e-9） |
| `test_apply_handles_all_zero_after_calibration` | 全部 0 になったら均等割りにして `nan` を出さない |
| `test_apply_preserves_order` | 較正は単調なので順序が変わらない |
| `test_fit_raises_on_length_mismatch` | 長さ違いで `ValueError` |
| `test_fit_raises_on_empty` | 0件で `ValueError` |

- [ ] **Step 2: `fit_weights.py` を実装する**

引数:

```
--data-dir data
--from 20250101 --to 20260901     対象期間
--train-until 20251201            train / dev の境界（必須）
--dev-until 20260401              dev / calib の境界（必須）
--calib-until 20260701            calib / test の境界（必須）
--lambda 0.1                      固定する場合。省略時は dev で選ぶ
--lambda-grid 0.001,0.01,0.1,1,10 探索するグリッド
--with-python-features / --without-python-features
                                  Python 側8列を入れる／入れない（両方試して比較する）
--predictions-from <dir>          保存済みの予測記録を使う（Jev を呼び直さない）
--out data/weights
```

**重要**: 学習には Jev の回答が必要なので、`scripts/backtest.py`（Task 21）で保存した予測記録を入力にする。`fit_weights.py` が Jev を呼び直さない。呼び直すと同じレースに二重課金するうえ、記録した回答と学習に使った回答がずれる。

出力: 選んだ `lambda`、採用した特徴、`dev` の指標、`data/weights/<weights_id>.json` のパス。`--with-python-features` と `--without-python-features` の両方の `dev` 指標を並べて表示し、どちらが良いかを人が判断できるようにする。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/learn/calibration.py scripts/fit_weights.py tests/test_calibration.py
git commit -m "feat(learn): isotonic calibration and fit_weights CLI"
```

---

# フェーズ4: 検証

## Task 19: 指標

**Files:**
- Create: `src/keiba/evaluate/__init__.py`, `src/keiba/evaluate/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `RaceOutcome`（`frozen dataclass`: `race_id: str`, `winners: tuple[str, ...]`, `starters: tuple[str, ...]`, `excluded: tuple[str, ...]`）
  - `OutcomeError(ValueError)`
  - `outcome_from_results(race_id: str, results: Sequence[ResultRow]) -> RaceOutcome | None`（勝ち馬不在・1頭立ては `None`）
  - `top_n_hit(ranking: Sequence[str], outcome: RaceOutcome, n: int) -> bool`
  - `top1_win(ranking, outcome) -> bool`
  - `brier(probabilities: Mapping[str, float], outcome) -> float`
  - `log_loss(probabilities, outcome) -> float`
  - `expected_top_n(probabilities, ranking, n) -> float`
  - `calibration_bins(pairs: Sequence[tuple[float, float]], *, bins: int = 10) -> list[dict]`
  - `MetricsReport`（`frozen dataclass`: `n_races`, `n_skipped`, `top_n_hit_rate`, `top1_win_rate`, `brier`, `log_loss`, `expected_top_n_mean`, `calibration: list[dict]`, `n: int`）
  - `aggregate(rows: Sequence[dict], *, n: int) -> MetricsReport`
  - `compare(baseline: MetricsReport, model: MetricsReport) -> dict`

- [ ] **Step 1: テストを書く**

手計算した小さな例と一致させる。

| テスト | 内容 |
|---|---|
| `test_top_n_hit_when_winner_in_top_n` | `ranking=["3","1","2"]`, 勝ち馬 `"1"`, `n=2` → `True` |
| `test_top_n_miss_when_winner_below` | 同じで `n=1` → `False` |
| `test_top_n_hit_with_dead_heat_any_winner` | 勝ち馬 `("1","5")` で `"1"` が上位nにあれば `True` |
| `test_top1_win_only_for_first_pick` | `ranking[0]` が勝ち馬のときだけ `True` |
| `test_brier_hand_computed` | 3頭で `p=(0.5,0.3,0.2)`、勝ち馬が1番目 → `(0.5-1)² + 0.3² + 0.2² = 0.38` |
| `test_brier_dead_heat_splits_outcome` | 勝ち馬2頭なら各 `y=0.5`（spec §12.2） |
| `test_log_loss_hand_computed` | 勝ち馬の `p=0.25` → `-log(0.25) ≈ 1.3863` |
| `test_log_loss_dead_heat_sums_probabilities` | 同着2頭の `p` が `0.2` と `0.3` なら `-log(0.5)`（spec §12.2） |
| `test_log_loss_clips_zero_probability` | 勝ち馬の `p=0.0` でも `inf` にならない（`eps` で clip） |
| `test_expected_top_n_sums_top_probabilities` | 上位n頭の確率の合計 |
| `test_outcome_excludes_scratched_from_starters` | `ijyo_cd` `1`/`2`/`3` は `starters` に入らず `excluded` に入る |
| `test_outcome_keeps_retired_in_starters` | `ijyo_cd` `4`/`5` は `starters` に残るが `winners` に入らない（spec §12.2） |
| `test_outcome_keeps_demoted_as_possible_winner` | `ijyo_cd="7"` で `kakutei_jyuni=1` なら勝ち馬 |
| `test_outcome_none_when_no_winner` | 勝ち馬不在で `None`（レース不成立） |
| `test_outcome_none_when_single_starter` | 1頭立てで `None` |
| `test_calibration_bins_group_by_predicted_probability` | 10ビンに分かれ、各ビンに `count` / `predicted_mean` / `actual_rate` |
| `test_calibration_bins_skip_empty_bins` | 空のビンは返らない |
| `test_aggregate_counts_skipped_races` | `skipped` フラグの行が `n_skipped` に数えられ、指標の分母に入らない |
| `test_aggregate_records_n_used` | レポートに使った `n` が入る（spec §12.1「レポートには使った n を必ず書く」） |
| `test_compare_reports_differences` | `model - baseline` の差が出る |

- [ ] **Step 2: 実装する**

`log_loss` の同着の扱いだけ全文（判断が入る箇所）。

```python
def log_loss(probabilities, outcome, *, eps=1e-12):
    """勝ち馬に割り当てた確率の負の対数。

    同着のときは同着馬の確率を合計する（spec §12.2）。
    「いずれかが勝つ」事象に対する確率だと解釈するため。
    p=0 で -inf にならないよう eps で clip する。
    """
    total = sum(probabilities.get(umaban, 0.0) for umaban in outcome.winners)
    return -math.log(max(total, eps))
```

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/evaluate/__init__.py src/keiba/evaluate/metrics.py tests/test_metrics.py
git commit -m "feat(evaluate): metrics with dead-heat and exclusion rules"
```

---

## Task 20: 時系列分割

**Files:**
- Create: `src/keiba/evaluate/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `Splits`（`frozen dataclass`: `train_until: date`, `dev_until: date`, `calib_until: date`）。メソッド `as_dict() -> dict[str, str]`
  - `SPLIT_NAMES = ("train", "dev", "calib", "test")`
  - `assign(hasso_at: datetime, splits: Splits) -> str`
  - `by_fraction(dates: Sequence[date], *, fractions: tuple[float, float, float, float] = (0.5, 0.2, 0.15, 0.15)) -> Splits`
  - `group_by_split(races: Sequence[RaceRow], splits: Splits) -> dict[str, list[RaceRow]]`
  - `SplitError(ValueError)`

- [ ] **Step 1: テストを書く**

| テスト | 内容 |
|---|---|
| `test_assign_puts_early_dates_in_train` | `train_until` 以前は `train`（境界を含む） |
| `test_assign_boundaries_are_inclusive_on_the_left` | `hasso_at.date() == train_until` は `train` |
| `test_assign_puts_late_dates_in_test` | `calib_until` より後は `test` |
| `test_assign_covers_all_four_splits` | 4区分すべてに振り分けられる |
| `test_by_fraction_respects_chronological_order` | 境界が昇順（`train_until <= dev_until <= calib_until`） |
| `test_by_fraction_uses_given_fractions` | 100日分で `(0.5,0.2,0.15,0.15)` なら概ね 50/20/15/15 |
| `test_by_fraction_raises_when_fractions_do_not_sum_to_one` | `SplitError` |
| `test_by_fraction_raises_on_empty_dates` | `SplitError` |
| `test_group_by_split_never_splits_a_race` | 同一 `race_id` が2つの区分に現れない（spec §9.2） |
| `test_group_by_split_returns_all_four_keys` | 空でも4キーある |
| `test_as_dict_is_json_serialisable` | 日付が ISO 文字列 |

- [ ] **Step 2: 実装する**

区分はレース単位で決まるので「同じレースの馬を学習側と評価側に分割しない」（spec §9.2）は自動的に満たす。それをテストで固定する。

- [ ] **Step 3: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/evaluate/splits.py tests/test_splits.py
git commit -m "feat(evaluate): chronological splits that never divide a race"
```

---

## Task 21: バックテストと backtest CLI

**Files:**
- Create: `src/keiba/evaluate/backtest.py`
- Create: `scripts/backtest.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `keiba.predict.pipeline.predict`, `keiba.evaluate.metrics`, `keiba.evaluate.splits`, `keiba.data.repository.ResultRepository`
- Produces:
  - `SkipReason`: `"no_winner"` / `"single_starter"` / `"jump"` / `"jev_failed"` / `"no_entries"` / `"match_ambiguous"` / `"input_invalid"`
  - `BacktestRow`（`frozen dataclass`: `race_id`, `hasso_at`, `split`, `skipped: SkipReason | None`, `baseline_ranking: tuple[str, ...]`, `baseline_probabilities: dict[str, float]`, `model_ranking: tuple[str, ...] | None`, `model_probabilities: dict[str, float] | None`, `outcome: RaceOutcome | None`, `record_id: str | None`, `entry_changed_count: int`）
  - `race_input_from_history(race: RaceRow, entries: Sequence[EntryRow], *, cutoff_hours: int = 18, top_n: int) -> RaceInput`（過去レースから前日時点の入力を復元する。`prediction_cutoff` は発走の `cutoff_hours` 時間前）
  - `run(races: Sequence[RaceRow], repo: Repository, results: ResultRepository, client, *, splits: Splits, n: int, with_layer2: bool, weights: Weights | None, save_records: bool, data_dir: Path) -> list[BacktestRow]`
  - `report(rows: Sequence[BacktestRow], *, n: int) -> dict`（`baseline` と `model` の `MetricsReport`、`compare`、スキップの内訳、区分別の内訳、`entry_changed` の件数）

- [ ] **Step 1: テストを書く**

| テスト | 内容 |
|---|---|
| `test_race_input_cutoff_is_before_start` | `prediction_cutoff < race.hasso_at` |
| `test_race_input_cutoff_is_previous_day_by_default` | 既定 18 時間前なので前日になる |
| `test_race_input_uses_toroku_tosu_not_syusso_tosu` | `entry_count` に `syusso_tosu` を使わない（spec §4.1.4） |
| `test_race_input_excludes_scratched_horses` | `ijyo_cd` `1`/`2`/`3` の馬は出走馬リストに入れない |
| `test_race_input_counts_changed_entries` | `EntryRow.changed_after_entry` の件数を数える |
| `test_run_skips_race_without_winner` | `skipped == "no_winner"`、指標の分母に入らない |
| `test_run_skips_single_starter_race` | `skipped == "single_starter"` |
| `test_run_skips_jump_race` | `skipped == "jump"` |
| `test_run_continues_after_jev_failure` | 1レースで Jev が落ちても残りが処理され、そのレースは `skipped == "jev_failed"` |
| `test_run_does_not_stop_on_ambiguous_match` | `skipped == "match_ambiguous"` で続行 |
| `test_run_assigns_split_per_race` | 各行に `train` / `dev` / `calib` / `test` が入る |
| `test_run_saves_records_when_requested` | `save_records=True` で予測記録が保存される（`fit_weights.py` の入力になる） |
| `test_report_includes_skip_breakdown` | スキップの理由別件数が入る（spec §12.3「スキップ件数と理由を必ず出す」） |
| `test_report_never_hides_skips` | スキップが1件でもあればレポートの必須キーに出る |
| `test_report_compares_baseline_and_model` | 両方の指標と差分が入る |
| `test_report_breaks_down_by_split` | 区分ごとの指標が入る |
| `test_report_includes_n_used` | 使った `n` がレポートに入る |
| `test_report_reports_entry_changed_count` | 斤量・騎手の変更があった馬の件数（spec §4.1.2 の注意） |
| `test_report_with_no_usable_races` | 全件スキップでもレポートが返り、`n_races == 0` |

- [ ] **Step 2: 実装する**

**部分失敗の扱いが予想1件とバックテストで違う**ことをコードに明記する（spec §12.3）。

```python
def run(races, repo, results, client, *, splits, n, with_layer2, weights,
        save_records, data_dir):
    """過去レースで予想を作り、確定結果と照合する。

    予想1件（scripts/predict.py）は第2層が1頭でも失敗したら FAILED にするが、
    バックテストは1頭の失敗で数百レースを止めると使えないので、そのレースを
    skipped にして続行する。スキップは report() が必ず件数と理由を出す。
    黙って落とすとスキップの偏りで指標が歪む。
    """
    rows = []
    for race in races:
        if not race.is_flat:
            rows.append(_skipped(race, splits, "jump"))
            continue
        outcome = metrics.outcome_from_results(race.race_id,
                                              results.results(race.race_id))
        if outcome is None:
            # 勝ち馬不在か1頭立て。どちらかを理由として記録する
            rows.append(_skipped(race, splits, _why_no_outcome(results, race)))
            continue

        entries = [r.entry for r in results.results(race.race_id)
                   if r.counts_as_starter]
        if len(entries) < 2:
            rows.append(_skipped(race, splits, "no_entries"))
            continue

        race_input = race_input_from_history(race, entries, top_n=n)
        try:
            record = predict(race_input, repo, client, with_layer2=with_layer2,
                             weights=weights, allow_partial=False)
        except PredictionFailed as error:
            rows.append(_skipped(race, splits, _classify_failure(error)))
            continue

        if save_records:
            store_records.save(record, data_dir=data_dir)
        rows.append(_row_from_record(race, splits, record, outcome, entries))
    return rows
```

`_classify_failure` は `PredictionFailed` の原因を `"jev_failed"` / `"match_ambiguous"` / `"input_invalid"` に振り分ける。理由が分からないスキップを作らない（偏りが見えなくなるため）。

`scripts/backtest.py` の引数:

```
--data-dir data
--from 20250101 --to 20260901
--train-until / --dev-until / --calib-until   省略時は by_fraction で自動
--n 3
--layer2 / --no-layer2
--weights <weights_id>
--save-records / --no-save-records            既定 保存する（fit_weights の入力になる）
--limit 100                                   レース数の上限（費用の歯止め）
--out data/backtests
```

出力: `data/backtests/<run_id>/report.json` と `rows.jsonl`、標準出力に `baseline` / `model` の指標とスキップ内訳の表。`--limit` を既定で付けるのは、Jev の費用が出走頭数×レース数で効くため。

- [ ] **Step 3: 小規模に実データで確認する**

Run: `.venv/Scripts/python.exe scripts/backtest.py --from 20260801 --to 20260810 --limit 5 --no-layer2`
Expected: 5レース分の `baseline` 指標とスキップ内訳が出て exit 0

- [ ] **Step 4: テスト全体を通す** → 全件 PASS

```bash
git add src/keiba/evaluate/backtest.py scripts/backtest.py tests/test_backtest.py
git commit -m "feat(evaluate): backtest with explicit skip reporting"
```

---

# 完了後の確認

すべてのタスクが終わったら次を確認する。

- [ ] `.venv/Scripts/python.exe -m pytest` が全件 PASS
- [ ] `Repository` の公開メソッド全部に `as_of` がある（`test_repository.py::test_every_public_method_requires_as_of`）
- [ ] 対象レースの結果が `state` に現れない（`test_summary.py::test_target_race_results_never_appear_in_state`）
- [ ] `to_user_view` が `state` と生レスポンスを含まない（`test_records_store.py`）
- [ ] `UM` の累計集計が CSV にも `state` にも無い
- [ ] `scripts/smoke_test.py` が変更されておらず、まだ動く
- [ ] `src/jev_client.py` の変更が `ask_parallel` の `return_exceptions` だけ
- [ ] `scripts/update_data.py --verify-time-fields` で不一致 0 件
- [ ] `scripts/predict.py` が実データで上位 n 頭を出せる
- [ ] `scripts/backtest.py` が小規模で回り、スキップ内訳が出る

## 仕様との対応

| spec の節 | 実装するタスク |
|---|---|
| §4.1 レコード分解 / §4.1.1 数値形式 | Task 3 |
| §4.1.2 DataKubun の選択 | Task 6 |
| §4.1.3 コード表 / §4.1.4 頭数 | Task 1, 3 |
| §4.2 CSV の列定義 | Task 4 |
| §4.3 版付き保存と manifest | Task 4 |
| §5 jev_client への変更 | Task 13 |
| §6.1 能力指標 / §6.2 クラス順序値 | Task 9, Task 1 |
| §6.3 条件別成績 / §6.4 近況 / §6.5 脚質 / §6.6 輸送 / §6.7 欠損 | Task 10 |
| §7 state の形式 | Task 11 |
| §8 質問定義 | Task 12 |
| §8.2 ペース区分の閾値（三分位と層の段階的な粗化） | Task 12（`features/pace.py`） |
| §9 第2層 → 数値特徴 | Task 14 |
| §10 第3層の統合 | Task 14 |
| §11 学習と較正 | Task 17, 18 |
| §12 検証 | Task 19, 20, 21 |
| §13 予測記録 | Task 15 |
| §14 入力 | Task 7, 8 |
| §15 テスト戦略 | 全タスク |
| §16 CLI | Task 5, 16, 18, 21 |

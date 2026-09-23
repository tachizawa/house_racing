# 予想コア・第2層・検証基盤 設計

作成日: 2026-09-22
元仕様: [tmp/jev_prediction_spec.md](../../../tmp/jev_prediction_spec.md)
対象範囲: 元仕様のうち A（予想コア）+ B最小（JV-Link → CSV 版付き保存）+ E（検証）+ F（第2層）

## 1. この文書の範囲

元仕様は独立した7つのサブシステムを含む。本設計はそのうち4つを扱う。

| | サブシステム | 元仕様の節 | 本設計 |
|---|---|---|---|
| A | 予想コア（入力 → 特徴量 → 第1層 → 第3層 → 記録） | §4〜§8, §10-4/5/6 | 含む |
| B | データ取り込み（JV-Link → CSV → 版付き保存） | §12.4, §10-1 | **最小部分のみ**（下記1.1） |
| C | AWS 基盤（Cognito / API GW / Lambda / SQS / DynamoDB） | §12.2/12.3/12.7, §10-2/5 | 含まない |
| D | Web フロント | §12.5, §10-3/6 | 含まない |
| E | 検証・バックテスト | §9, §10-7 | 含む |
| F | 第2層（適性・状態評価） | §6, §10「次に追加」 | 含む |
| G | スクショ入力（OCR） | §12.6, §10「次に追加」 | 含まない |

### 1.1 B の「最小部分」の定義

含む: JV-Link から `RA` / `SE` / `UM` を読み、CSV（UTF-8・ID は文字列・列定義固定・欠損表現固定）に変換し、**ローカルの版付きディレクトリ + manifest** に保存する。

含まない: S3 アップロード、Parquet 読み取り層、分割・索引・集計済みデータの事前生成、差分更新。元仕様 §12.4 のうち保存先が S3 であることと読み取り最適化だけが後回しになる。

### 1.2 確定した前提

| 項目 | 決定 | 根拠 |
|---|---|---|
| 予想時点 | **前日**（出馬表確定後・発走前日） | 未来情報混入（§9.3）を防ぎやすく、バックテストで過去時点を再現しやすい |
| 統合方法 | **条件付きロジット（Luce モデル）+ ベースライン併走** | §13 で未確定。レース内 softmax が §13 最終項「第1層と第2層のスケールの揃え方」を構造的に解消する |
| `n` の既定値 | **3**（設定可） | §13 で未確定。設定値として扱うと §1 が定めている |
| `state` の形式 | **dict** | §5 節参照 |
| 第2層の質問種別 | **`Choice`**（`Noul` ではない） | §6.2 の表が4区分の分類を指定している |
| 展開予測 `pace` | 既定 OFF（実装はする） | §5.3 は任意 |
| 展開別3着内予測 | **実装しない** | §6.3 が「初期版ではこの評価を省略してよい」と明記 |
| 実着順の組合せ予測 | **実装しない** | §11 が「最小構成には含めず」と明記 |
| 障害レース | 対象外（平地のみ） | 能力指標の比較対象が揃わない |

## 2. パッケージ構成

`src/jev_client.py` は通信層として維持する。CLAUDE.md の「予想ロジックと重みは置かない」「通信・認証・例外処理と予想処理を分離する」（元仕様 §7.2）を守る。

```
src/
  jev_client.py            既存。5章の1点だけ変更（return_exceptions の追加）
  keiba/
    __init__.py
    config.py              設定値。n / 対象走数 / 軸のON・OFF / 版文字列 / 例外ルール
    models.py              dataclass 群。EntryRow と ResultRow の分離（3章）
    ingest/
      __init__.py
      records.py           RA / SE / UM の固定長レコード(cp932) → dict
      jvlink.py            JV-Link 呼び出し。Windows 専用（pywin32）
      store.py             版付きディレクトリと manifest の読み書き
    data/
      __init__.py
      repository.py        CSV 参照。全メソッドが as_of を必須引数に取る
    input/
      __init__.py
      schema.py            利用者入力 → RaceInput
      validate.py          §12.5 の入力検証
      match.py             馬名正規化と馬マスタ照合
    features/
      __init__.py
      ability.py           能力指標（定義固定）
      derive.py            §4.3 の派生値（近況 / 条件別成績 / 脚質 / 輸送）
      summary.py           第1層・第2層の state 組み立て
    predict/
      __init__.py
      questions.py         第1層・第2層の質問定義とプロンプト版
      layers.py            第1層・第2層の実行
      integrate.py         第3層（baseline / model）
      pipeline.py          層のオーケストレーションと予測記録の組み立て
    learn/
      __init__.py
      conditional_logit.py 重みの最尤推定
      calibration.py       確率の較正
    evaluate/
      __init__.py
      metrics.py           §9.1 の指標
      splits.py            時系列分割（§9.2）
      backtest.py          過去レースで予想を作り照合
    store/
      __init__.py
      records.py           予測記録の保存と読み出し（JSONL）

scripts/
  smoke_test.py            既存。触らない
  update_data.py           B: JV-Link → CSV 版付き保存
  predict.py               A: 1レース予想
  fit_weights.py           学習 + 較正
  backtest.py              E: 期間指定バックテスト

tests/
  test_jev_client.py       既存
  fixtures/                実バイト列のレコードフィクスチャ、小さな CSV 一式
  test_*.py                各モジュールに対応
```

追加依存（`requirements.txt`）:

```
numpy>=1.26
scipy>=1.11          # 条件付きロジットの最適化
pandas>=2.1          # CSV 参照
scikit-learn>=1.4    # isotonic 較正
```

## 3. 未来情報の混入防止を型で担保する

元仕様 §9.3 をコメントや注意書きではなく構造で守る。これが本設計で最も重要な決定。

### 3.1 EntryRow と ResultRow の分離

```python
@dataclass(frozen=True)
class EntryRow:
    """出馬表相当。予想パイプラインが受け取れるのはこれだけ。"""
    umaban: str
    wakuban: str
    ketto_num: str
    bamei: str
    sex_cd: str
    barei: str
    futan_kg: float | None
    kisyu_code: str
    tozai_cd: str        # そのレース時点の所属。美浦 / 栗東 の判定に使う
    chokyosi_code: str

@dataclass(frozen=True)
class ResultRow:
    """結果込み。照合側だけが触る。"""
    entry: EntryRow
    ijyo_cd: str                    # 異常区分。"0" 以外は出走扱いにしない
    kakutei_jyuni: int | None
    dochaku_kubun: str
    time_sec: float | None
    time_diff_sec: float | None
    jyuni_1c: int | None
    jyuni_2c: int | None
    jyuni_3c: int | None
    jyuni_4c: int | None
    haron_time_l3_sec: float | None
    ba_taijyu_kg: int | None
    zogen_sa_kg: int | None
    odds: float | None              # 確定後の値。予想入力には使わない
    ninki: int | None               # 同上
```

`predict/pipeline.py` と `features/` の型注釈は `EntryRow` のみ。`ResultRow` は `evaluate/` と `data/repository.py` の過去走参照だけが扱う。

### 3.2 as_of の強制

`Repository` の全公開メソッドは `as_of: datetime` を必須引数に取る。既定値を与えず、`as_of` なしで呼べる経路を作らない。

過去走の採用条件: そのレースの発走日時 < `as_of`。発走日時は `RA` の開催年月日 + `HassoTime` から組む。

### 3.2.1 `RaceRow` は時点で分割しない。代わりにゲートを置く

`EntryRow` / `ResultRow` と違い、`RaceRow` はレース前の条件（距離・トラック・クラス）と結果（`syusso_tosu` / `nyusen_tosu` / `first_3f_sec` / `last_3f_sec`）を1つの型に持つ。分割しない理由は、**過去走の `RaceRow` では結果側が正当に必要だから**である。

- 脚質の4角相対位置（6.5 節）は分母に `syusso_tosu` を使う
- ペース区分の閾値（8.2 節）は `first_3f_sec` を使う

どちらも `as_of` より前のレースなので、結果が判明しているのは当然。ここで型を割ると、過去走のたびに2つの型を結合し直すことになる。

危険なのは**対象レース自身**の `RaceRow` だけ。そこで `Repository.race()` にゲートを置く。

```
判定に使う hasso_at = 取得できたすべての版の hasso_at の最大値
  >= as_of のとき: data_kubun 1（出走馬名表）/ 2（出馬表）の行だけを読む。
                   成績行（3〜7）には落ちない。該当が無ければ None を返す。
  <  as_of のとき: 通常どおり DATA_KUBUN_RESULT_PRIORITY で選ぶ。
```

**判定は必ず最大値で行う。** `RA.HassoTime`（位置874）は発走時刻が変更されると後の版で書き換わり、変更前の値は `変更前発走時刻`（位置878、データ区分3〜7で設定）に移る。つまり出馬表（区分2）の `hasso_at` は**予定**時刻で、成績（区分7）の `hasso_at` が実際の時刻になる。出馬表側の早い時刻で判定すると、発走が遅れたレースについて「もう終わっている」と誤判定して結果を漏らす。

どの版を「正しい」とするかで迷わないために最大値を採る。過小に隠すと静かな誤りになるが、過大に隠しても呼び出し側には `None` が見えるだけで気づける。

同じ規則を `past_runs` の `as_of` 判定にも使う。両方が「そのレースについて分かっているすべての版の最大 `hasso_at`」という1つのヘルパを通ることで、片方だけが単一版を信じる不整合をなくす。

### 3.2.2 残存する制約: 新しい版が存在しない場合

最大値を採っても、**後の版がそもそも取得できていない**場合は守れない。`SE` が成績（区分7）なのに `RA` が出馬表（区分2）しか無ければ、分かる `hasso_at` は予定時刻だけで、実際の発走が `as_of` より後だったかどうかは知りようがない。

これを塞ぐ手段は2つとも代償が大きい。

- 「`SE` が成績なのに `RA` が出馬表なら、その過去走を使わない」→ データの揃い方によっては正当な過去走を大量に落とし、能力指標が空になる
- 何らかの推定で補う → 根拠のない規則を増やす

**初期版では塞がない。** 理由は、この経路が噛むのは `as_of` がレースの予定発走時刻の「遅延幅の内側」にある場合に限られるため。本仕様の予想時点は**前日**（§1.2）で、`as_of` は対象レースの前日夕方、過去走はさらに前の週末にある。予定時刻と `as_of` の差は数十時間あり、数分から数十分の発走遅延がその境界をまたぐことはない。

**当日予想（発走直前）を追加する場合は、この制約を先に解消すること。** そこでは `as_of` がレースの発走時刻の近傍に来るため、この経路が現実に噛む。

これを守る防御は2重になっている。

1. このゲート（`Repository.race()`）
2. 組み立てた `state` に対象レースの結果値が現れないことを文字列検索で確かめるテスト（15.1 節）

加えて、予想パイプライン（11 章の `build_context`）は**対象レースの条件を利用者入力の `RaceInput` から取り**、`Repository.race()` からは取らない。`Repository.race()` が対象レースに対して返すのは、内部データとの照合用の補助情報に留める。

### 3.3 UM の累計集計を使わない

`UM`（馬マスタ）の `RuikeiHonsyo*` / `ChakuKaisu*`（総合・中央・馬場別・馬場状態別・距離別）/ `Kyakusitu` / `RaceCount` は**現在までの累計**であり、過去時点の値を復元できない。元仕様 §9.3 の「将来の成績を含めて集計した馬・騎手・調教師の成績」に該当するため入力に使わない。

`UM` から使うのは時点に依存しない項目だけ: `KettoNum` / `Bamei` / `BameiKana` / `BirthDate` / `SexCD` / `KeiroCD` / `HinsyuCD`。所属（`TozaiCD`）と調教師は `UM` ではなく `SE`（そのレース時点の値）から取る。

条件別成績はすべて `SE` を `as_of` で絞って自前集計する。

### 3.4 これを守るテスト

- `as_of` より後のレースが過去走に混ざらない
- 対象レースの `ResultRow` 由来の値が、組み立てた `state` の中に一度も現れない（state を JSON 文字列化して結果値を文字列検索する）
- `Repository` の全公開メソッドのシグネチャに `as_of` があり、既定値を持たない（`inspect.signature` で検査）
- `UM` の累計集計列が `state` に現れない

## 4. データ層（B 最小）

### 4.0 取得する dataspec は1つでは足りない

**`JVOpen("RACE", ...)` は馬マスタ（`UM`）を返さない。** 仕様書の「データ種別一覧」より:

| dataspec | `UM` を含むか | 備考 |
|---|---|---|
| `RACE` | **含まない** | レース情報（`RA` / `SE` / `HR` / `O1`–`O6` / `H1` / `H6` / `JG` / `WF`） |
| `DIFF` / `DIFN` | 含む | 蓄積系。**前回提供分から変更・追加のあった** 競走馬マスタ。月曜14:00頃、1年間 |
| `TCOV` / `RCOV` | 含む | 非蓄積系。全出走馬（登録馬／出走予定馬）の競走馬マスタ |

SDK の `ドキュメント/蓄積系提供データ一覧.xls` から独立に裏付けが取れる。提供ファイル名の3文字目が提供区分を表しており、`UMFWyyyymmddyyyymmddhhmmss.jvd`（`F` = `DIFF`）が存在する。

| 3文字目 | dataspec |
|---|---|
| `F` | `DIFF` |
| `O` | `RCOV` |
| `P` | `SNAP` |
| `T` | `TCOV` |
| `B` | `BLOD` |
| `X` + 末尾 `M` | 月次 |

同じ一覧で `RA` と `SE` には `F` のファイルが無い。つまり **`DIFF` はレース系を返さない**ので、`RACE` と併せて取得しても版の中で行が二重にならない。

したがって `update_data.py` は**複数の dataspec を1つのデータ版にまとめて取得する**。既定は `RACE` と `DIFF`。

manifest のキーは `dataspec`（文字列）から **`dataspecs`（一覧）へ改名する**。型だけ変えて名前を据え置いてはいけない——`manifest["dataspec"]` が `"RACE"` のままだと `for s in manifest["dataspec"]` が1文字ずつ回って黙って通る。改名すれば旧形式を前提にしたコードは `KeyError` で落ちる。同じ理由で `VersionWriter` は生の文字列を渡されたら `TypeError` にする。

`config.SCHEMA_VERSION` は上げない。この定数は CSV の列構成を指しており、列は変わらない。かつ同じ定数が予想レコード（`store/records.py`）にも書かれるため、上げると「予想レコードの形も変わった」という誤った含意が出る。

manifest の `record_counts` は平坦な `{"RA":…, "SE":…, "UM":…}` のまま保つ。

#### 4.0.2 `DIFF` は初回にセットアップ取得が必要（実機で確認）

`JVOpen` の戻り値を直接測った結果:

| dataspec | option | fromdate | `open_code` |
|---|---|---|---|
| `DIFF` | 1 | 20260901000000 | **-1**（該当データなし） |
| `DIFF` | 2 | 20260901000000 | **-111**（`DIFF` と option=2 は不正な組み合わせ） |
| `DIFF` | 1 | 20260101000000 | **-1**（8ヶ月遡っても同じ） |
| `RACE` | 1 | 20260901000000 | 0（正常、62ファイル） |

蓄積系差分は「前回提供分から変更・追加のあったもの」なので、JV-Link が前回提供分を
持っていない状態では差分が作れない。**馬マスタを取り込むには一度だけ
`--dataspec DIFF --option 4`（セットアップ、ダイアログなし）を走らせる必要がある。**

これは §4.0.1 で予期した「運用の前提が増える」がそのまま実測で確定したもの。
**だから照合は馬マスタの完全性に依存させない。** `DIFF` が空でも予想は成立する。

CLI は `open_code == -1` を `(0件)` と表示してはいけない。「読んだが0件だった」と
「開けなかった」は別のことで、運用者が区別できなければ欠陥に気づけない。
`-1` のときは該当データなしであることと、`DIFF` / `DIFN` に限ってセットアップ取得の
手順を表示する。
これは保存された CSV の行数を表すもので、CSV は dataspec をまたいで1本にまとまるから。dataspec ごとの内訳は CLI の表示に置く。

### 4.0.1 それでも馬マスタの完全性に依存しない

`DIFF` は**差分**なので、取得期間の中でマスタ行が更新されなかった馬は含まれない。初回の完全な取り込みには `JVOpen` の option をセットアップ用（3 または 4）にする必要があり、運用の前提が増える。

**馬名照合が馬マスタの完全性に依存してはいけない。** `SE`（馬毎レース情報）は出走した全馬について `bamei` と `ketto_num` を持っており、それは「このデータ版の中に過去走がある馬」の集合とちょうど一致する。過去走が無い馬は、マスタにあってもこのシステムでは使える履歴がない。

そこで **14.3 節の正規化済み索引は `horses.csv` と `race_horses.csv` の両方から作る。**

| 索引の出典 | 得られるもの |
|---|---|
| `horses.csv`（`UM`） | `bamei` / `bamei_kana` / `birth_date` / `sex_cd` |
| `race_horses.csv`（`SE`） | `bamei` / `sex_cd`（`bamei_kana` と `birth_date` は `None`） |

`SE` 由来の馬は `bamei_kana` を持たないので半角カナでの照合ができない。その制約は受け入れる——照合がレース情報だけで成立することのほうが重要で、`DIFF` の取得状況に予想の成否が左右されるほうが悪い。

### 4.1 レコード分解

`ingest/records.py` は `vendor/jravan-sdk-500-64bit/JV-Data構造体/Python版/JVData_Struct.py` のオフセット定義に従って `RA` / `SE` / `UM` を dict に分解する。

- レコードは cp932。`memview.tobytes().decode("cp932")` でデコードする
- レコード種別は先頭2バイト
- `JVRead` ではなく `JVGets` を使う

`vendor/` を直接 import しない。オフセットは `ingest/records.py` に写し、由来を各定数のコメントに書く。`vendor/` は再配布不可で `.gitignore` 済みのため、それに依存すると環境を選ぶ。

### 4.1.1 数値項目の形式（`JV-Data仕様書_4.9.0.1.xlsx` の「フォーマット」シートで確定）

| 項目 | 位置・長さ | 形式 |
|---|---|---|
| `SE.Time`（走破タイム） | 339 / 4 | `9分99秒9`。`"1234"` → 1分23.4秒 = 83.4秒 |
| `SE.TimeDiff`（タイム差） | 532 / 4 | **符号1文字 + `99秒9`（3桁、0.1秒単位）。符号は1着が `-`、2着以下が `+`。1着馬には「2着馬とのタイム差」が入る。** 出走取消・競走除外・発走除外・競走中止は `"9999"` |
| `SE.HaronTimeL3`（後3ハロン） | 391 / 3 | `99.9秒`。取消・除外・中止・タイムオーバーは `"999"` |
| `SE.ZogenSa`（増減差） | 329 / 3 | 単位 kg。`001`–`998` が有効、`999` 計量不能、`000` 前差なし、スペース 初出走 |
| `SE.ZogenFugo`（増減符号） | 328 / 1 | `+` / `-` |
| `RA.HaronTimeS3`（前3ハロン） | 970 / 3 | `99.9秒`。**ラップタイム前半3ハロンの合計が既に入っている。平地競走のみ** |
| `RA.HassoTime`（発走時刻） | 874 / 4 | `hhmm` |
| `RA.LapTime` | 891 / 3×25 | `99.9秒`、1ハロン(200m)ごと。平地のみ |

`margin_sec` の復元（6.1 節で使う）:

```python
def margin_sec(time_diff: str) -> float | None:
    """SE.TimeDiff から「1着馬とのタイム差（秒）」を復元する。

    1着馬の TimeDiff には「2着馬とのタイム差」が符号 '-' で入っている。
    1着馬の margin は定義上 0.0 なので、符号が '-' なら値を捨てて 0.0 を返す。
    """
    if time_diff == "9999" or not time_diff.strip():
        return None                      # 取消・除外・中止
    sign, digits = time_diff[0], time_diff[1:]
    if not digits.isdigit():
        return None
    if sign == "-":
        return 0.0                       # 自身が1着
    if sign == "+":
        return int(digits) / 10.0
    return None
```

`RA.HaronTimeS3` が既に前半3ハロンの合計なので、**`LapTime` を自分で合計しない**（8.2 節の `first_3f_sec` はこれを使う）。`LapTime` は CSV に残すが `pace` の判定には使わない。

### 4.1.2 データ区分（`DataKubun`）による行の選択 ← 未来情報混入に直結する

同じ `race_id`（および `umaban`）に対して、**更新段階ごとに複数のレコードが存在する。**

| `DataKubun` | 内容 |
|---|---|
| `1` | 出走馬名表（木曜） |
| `2` | 出馬表（金・土曜） |
| `3` | 速報成績（3着まで確定） |
| `4` | 速報成績（5着まで確定） |
| `5` | 速報成績（全馬着順確定） |
| `6` | 速報成績（全馬着順＋コーナー通過順） |
| `7` | 成績（月曜） |
| `A` | 地方競馬 |
| `B` | 海外国際レース |
| `9` | レース中止 |
| `0` | 該当レコード削除（提供ミスなど） |

CSV は重複行を持つ。`Repository` が用途別に1行へ絞る。

| 用途 | 選択規則 |
|---|---|
| 過去走の集計（`ResultRow`） | `data_kubun == "7"` を優先。無ければ `"6"` → `"5"` → `"4"` → `"3"` の順 |
| 予想対象レースの出馬表（`EntryRow`） | `data_kubun == "2"` を優先、次に `"1"`。無ければ成績行（上の順）から **`EntryRow` 相当の列だけ**を抽出 |
| 常に除外 | `"0"`（削除）、`"9"`（中止）、`"A"`（地方）、`"B"`（海外） |

同一 `(race_id, umaban, data_kubun)` が重複した場合は `make_date` が新しい行を採用する。

### 4.1.2.1 `UM` のデータ区分は `RA`/`SE` と意味が違う

上の表は `RA` と `SE` のもの。**`UM`（競走馬マスタ）は同じ位置に別の意味のコードを持つ。**

| `UM` の `DataKubun` | 内容 |
|---|---|
| `1` | 新規馬名登録 |
| `2` | 馬名変更 |
| `3` | 再登録（抹消後の再登録） |
| `4` | その他更新 |
| `9` | **抹消** |
| `0` | 該当レコード削除（提供ミスなどの理由による） |

したがって **`UM` には `RA`/`SE` の除外集合を使ってはいけない。** `UM` で除外するのは `"0"`（提供ミスによる削除）だけ。

`"9"`（抹消）を除外すると**引退馬がマスタから消える**。引退馬は実在の馬で完全な戦績を持ち、しかもバックテスト（12 章）で予想する過去レースの出走馬は、現在の馬マスタではほぼ全頭が抹消済みである。除外すると過去レースの馬名照合がほぼ全滅する。

`A` / `B` は `UM` では定義されていないので、除外集合に入れる意味がない。

`config.py` は2つを別の定数で持つ。

```python
DATA_KUBUN_DROP = frozenset({"0", "9", "A", "B"})        # RA / SE 用
DATA_KUBUN_DROP_HORSE = frozenset({"0"})                 # UM 用。9 は抹消＝実在の馬
```

**成績行から `EntryRow` を作った場合の注意**: `Futan`（負担重量）と `KisyuCode`（騎手）は変更後の値になっている可能性がある。`SE` には `FutanBefore` / `KisyuCodeBefore` がある。これらが設定されている馬は「前日時点の値と異なる可能性がある」と `data_quality.horse_notes` に記録し、バックテストのレポートで件数を報告する。元仕様 §12.4 の「過去時点の情報を再現できない項目は、その時点で利用可能だったと仮定しない」に該当するため、黙って使わない。

> **訂正（2026-09-23）**: 当初この節は「`FutanBefore` / `KisyuCodeBefore` は**変更があった場合のみ設定される**」と書いていたが、**`KisyuCodeBefore` については虚偽だった。** JV-Data は「変更なし」を空白ではなくゼロ埋め（`"00000"`）で常に入れてくる。実データの `race_horses.csv` では 7,455 行中 7,389 行が `"00000"` で、この不変条件を信じた `EntryRow.changed_after_entry` が `bool("00000".strip())` を読み、**1,358 頭全頭に注記を付けてそのうち 1,346 頭（99.1%）が偽**だった。真の変更は 12 頭。`FutanBefore` は `"000"` が `None` に落ちるので当初の記述どおりで正しい。判定は `EntryRow.changed_after_entry` に閉じ込め、`ingest/records.py` でも取り込み時にゼロ埋めを空文字列へ正規化する（正規化はこれから作る版にしか効かないので、読む側にも同じ判定を置く）。

### 4.1.3 コード表の実値（`JV-Data仕様書_4.9.0.1.xlsx` の「コード表」シート）

**グレードコード（2003）** — 1バイト

| 値 | 内容 |
|---|---|
| `A` | G1（平地） |
| `B` | G2（平地） |
| `C` | G3（平地） |
| `D` | グレードのない重賞 |
| `E` | 重賞以外の特別競走 |
| `F` `G` `H` | J・G1 / J・G2 / J・G3（障害） |
| `L` | リステッド |
| スペース | 一般競走または未設定 |

**競走条件コード（2007）** — 3バイト。`RA` には **5要素**ある（2歳条件 / 3歳条件 / 4歳条件 / 5歳以上条件 / **最若年条件**）。

`class_level` の判定には **最若年条件（`JyokenCD[4]`、位置 635）** を使う。「出走可能な最も馬齢が若い馬に対する条件」で、レースのクラスを代表する。

| 値 | 内容 |
|---|---|
| `000` | 未設定 |
| `001`–`100` | 収得賞金による条件（`005` = 1勝クラス、`010` = 2勝クラス、`016` = 3勝クラス） |
| `701` | 新馬 |
| `702` | 未出走 |
| `703` | 未勝利 |
| `999` | オープン |

**トラックコード（2009）** — 2バイト

| 範囲 | `track_kind` |
|---|---|
| `10`–`22` | `turf`（平地 芝） |
| `23`–`29` | `dirt`（平地 ダート・サンド） |
| `51`–`59` | `jump`（障害） |
| `00` その他 | `null` |

**馬場状態コード（2010）**: `0` 未設定 / `1` 良 / `2` 稍重 / `3` 重 / `4` 不良

**異常区分コード（2101）**: `0` 下記以外（正常） / `1` 出走取消 / `2` 発走除外 / `3` 競走除外 / `4` 競走中止 / `5` 失格 / `6` 落馬再騎乗 / `7` 降着

**性別コード（2202）**: `1` 牡 / `2` 牝 / `3` セン

**東西所属コード（2301）**: `0` その他 / `1` 関東（美浦） / `2` 関西（栗東） / `3` 地方招待 / `4` 外国招待

**着差コード（2102）は馬身単位のコード**（`_12` = 1/2馬身、`A__` = アタマ など）で**秒ではない**。能力指標には使わず、`TimeDiff` を使う。

### 4.1.4 頭数の扱い

| 項目 | 内容 | 予想時点で使えるか |
|---|---|---|
| `TorokuTosu`（登録頭数） | 出馬表発表時の登録頭数 | **使える** |
| `SyussoTosu`（出走頭数） | 登録頭数から出走取消・競走除外・発走除外を除いた頭数 | **結果情報。使わない** |
| `NyusenTosu`（入線頭数） | 出走頭数から競走中止を除いた頭数 | 結果情報。評価側だけ |

脚質の 4角相対位置（6.5 節）の分母は過去走の値なので `SyussoTosu` を使う（過去走は確定済み）。予想対象レースの頭数は利用者入力と `TorokuTosu` を使う。

### 4.2 CSV の列定義

元仕様 §12.4 の要求に従う。

共通規則:

- 文字コードは UTF-8（BOM なし）、改行は `\n`
- ID 列（`ketto_num` / `kisyu_code` / `chokyosi_code` / `jyo_cd` / 各種コード）は**文字列**。先頭ゼロを数値化しない。pandas 読み込み時に `dtype=str` を明示する
- 欠損は**空文字列**で表す。`0` や `-` を使わない
- 時刻列は ISO 8601（`2026-09-21T10:30:00+09:00`）

ファイルは3つ。

**`races.csv`** — `RA` 由来。1行1レース。

| 列 | 型 | 説明 |
|---|---|---|
| `race_id` | str | `{year}{monthday}{jyo_cd}{kaiji}{nichiji}{race_num}` の16桁 |
| `year` `monthday` `jyo_cd` `kaiji` `nichiji` `race_num` | str | `RACE_ID` の各要素 |
| `hasso_at` | str | 開催日 + `HassoTime` から組んだ発走日時（JST） |
| `kyori_m` | int | `Kyori` |
| `track_cd` | str | `TrackCD` |
| `track_kind` | str | `track_cd` から導出。`turf` / `dirt` / `jump` |
| `jyoken_cd_youngest` | str | `JyokenCD[4]`（最若年条件）。`class_level` の判定に使う |
| `jyoken_cd_all` | str | `JyokenCD` の5要素を `;` 連結（参照用） |
| `grade_cd` | str | `GradeCD`。スペースは空文字列にしない（`" "` のまま保持し、CSV では `_` に置換せず空文字列で書く） |
| `class_level` | int | 0–8 の順序値（6.2 節） |
| `syubetu_cd` `kigo_cd` `jyuryo_cd` | str | `RACE_JYOKEN` の各要素 |
| `course_kubun_cd` | str | `CourseKubunCD`（`"A "`–`"E "`） |
| `tenko_cd` `siba_baba_cd` `dirt_baba_cd` | str | `TENKO_BABA_INFO` |
| `baba_cd` | str | `track_kind` に応じて芝／ダートから選んだ馬場状態 |
| `toroku_tosu` | int | 登録頭数。**予想時点で使える** |
| `syusso_tosu` `nyusen_tosu` | int | 出走・入線頭数。**結果情報**（4.1.4 節） |
| `lap_times_sec` | str | `LapTime` を `;` 連結（秒）。参照用 |
| `first_3f_sec` | float | **`RA.HaronTimeS3` をそのまま秒にした値。`LapTime` を合計しない** |
| `last_3f_sec` | float | `RA.HaronTimeL3` |
| `race_name` | str | `RaceInfo.Hondai` |
| `data_kubun` | str | `RECORD_ID.DataKubun`（4.1.2 節） |
| `make_date` | str | `RECORD_ID.MakeDate` |
| `fetched_at` | str | 取得時刻 |

**`race_horses.csv`** — `SE` 由来。1行1頭×1レース。`EntryRow` 相当の列と `ResultRow` 相当の列を**列名の接頭辞で分ける**（`result_` を付ける）。読み取り側が結果列を誤って使いにくくする。

| 列 | 型 | 分類 |
|---|---|---|
| `race_id` | str | キー |
| `umaban` `wakuban` `ketto_num` `bamei` | str | Entry |
| `sex_cd` `barei` `futan_kg` `kisyu_code` `tozai_cd` `chokyosi_code` `uma_kigo_cd` `blinker` | str/float | Entry |
| `futan_before_kg` `kisyu_code_before` | float/str | Entry。`futan_before_kg` は変更時のみ設定。**`kisyu_code_before` は変更が無くてもゼロ埋めで入る**ので、値の有無で変更を判定しない（4.1.2 節の訂正） |
| `result_ijyo_cd` | str | Result |
| `result_kakutei_jyuni` `result_dochaku_kubun` `result_dochaku_tosu` | str/int | Result |
| `result_time_sec` `result_time_diff_sec` | float | Result |
| `result_jyuni_1c` … `result_jyuni_4c` | int | Result |
| `result_haron_l3_sec` `result_haron_l4_sec` | float | Result |
| `result_ba_taijyu_kg` `result_zogen_sa_kg` | int | Result |
| `result_odds` `result_ninki` | float/int | Result |
| `data_kubun` `make_date` `fetched_at` | str | メタ |

**`horses.csv`** — `UM` 由来。3.3 節の通り時点非依存の項目だけ。

| 列 | 型 |
|---|---|
| `ketto_num` `bamei` `bamei_kana` `birth_date` `sex_cd` `keiro_cd` `hinsyu_cd` | str |
| `del_kubun` `reg_date` `del_date` | str |
| `data_kubun` `make_date` `fetched_at` | str |

`UM` の累計成績列は CSV に**書き出さない**。書き出さなければ誤用できない。

### 4.3 版付き保存と manifest

`data/` 全体の配置（`.gitignore` 済み。JRA-VAN のデータは再配布不可）:

```
data/
  CURRENT                        使用中の version_id を1行で持つ
  versions/<version_id>/
    races.csv
    race_horses.csv
    horses.csv
    manifest.json
  weights/<weights_id>.json      学習済みの重みと較正（11.1.2 節）
  predictions/<YYYYMM>.jsonl     予測記録（13 章）
  backtests/<run_id>/            バックテストのレポートと生の予測（12 章）
```

`version_id` は取得開始時刻の `YYYYMMDDTHHMMSSZ`。

`manifest.json`:

```json
{
  "version_id": "20260922T013000Z",
  "schema_version": "1",
  "fetched_at": "2026-09-22T10:30:00+09:00",
  "dataspec": "RACE",
  "fromdate": "20260801000000",
  "option": 1,
  "coverage": {"first_hasso_at": "...", "last_hasso_at": "...", "race_count": 552},
  "files": [
    {"name": "races.csv", "rows": 552, "bytes": 123456, "sha256": "..."}
  ],
  "record_counts": {"RA": 552, "SE": 7294, "UM": 0},
  "status": "complete"
}
```

書き込み手順（元仕様 §12.4「更新途中の CSV を予想で読まない」）:

1. `data/versions/<version_id>.partial/` に書く
2. 全ファイルを書き終えたら sha256 を計算して manifest を書き、`status: "complete"` にする
3. ディレクトリを `<version_id>/` にリネーム（同一ファイルシステム内なので原子的）。**同じ `version_id` の完成済みディレクトリが既にある場合は上書きせず拒否する**（下記）
4. `data/CURRENT` を書き換える

**完成済みの版は不変として扱う。** `commit()` は最終ディレクトリの有無を次のように場合分けする。

| 最終ディレクトリの状態 | 扱い |
|---|---|
| 存在しない | そのままリネームする |
| 存在し、manifest が `status: "complete"` | **`DataVersionError` を投げて拒否する。** 削除も上書きもしない |
| 存在するが manifest が無いか `complete` でない | 過去の異常終了が残した残骸なので削除して進む |

完成済みを黙って上書きすると、予想ジョブが固定したはずの版の中身が後から変わる（§12.4「予想ジョブごとに参照データ版を固定する」が成立しなくなる）。`new_version_id()` は秒精度なので、短時間の再実行で `version_id` が衝突しうる。衝突時は新しい版として取り直すか、運用者が手でディレクトリを消す。

この規則により、危険な窓は「リネーム成功後・`CURRENT` 書き込み前」だけになる。そこで異常終了しても `CURRENT` は**以前の版を指したまま**で、新しい版は参照されずに残るだけなので、読み取り側は壊れない。

`Repository` は `CURRENT` が指す版の manifest を読み、`status != "complete"` なら読み込みを拒否する。`.partial` サフィックスの付いたディレクトリは無視する。

予想ジョブは開始時に version_id を固定して記録する（元仕様 §12.4「予想ジョブごとに参照データ版を固定する」）。

## 5. jev_client への唯一の変更

`ask_parallel` に `return_exceptions: bool = False` を追加する。

現状の `ask_parallel` は `pool.map` を使っているため、1頭の失敗で全体が例外になる。バックテストで1頭の失敗が数百レースを止めると使えない。

```python
def ask_parallel(
    client, jobs, *, max_workers=DEFAULT_MAX_WORKERS, timeout=None,
    return_exceptions: bool = False,
) -> dict[str, Any]:
    """...

    ``return_exceptions=True`` のとき、失敗したジョブの値は
    ``JevRequestError`` インスタンスそのものになる。呼び出し側が
    ``isinstance(value, JevRequestError)`` で判定する。
    """
```

既定値 `False` なので既存の呼び出し（`smoke_test.py`）と既存テストの挙動は変わらない。

並列実行はこのモジュールの責務（docstring の3番「馬単位の問い合わせ (第2層) のための並列実行」）なので置き場所として正しく、予想ロジックも重みも入らない。

## 6. 特徴量

### 6.1 能力指標（定義固定）

元仕様 §4.3 は「定義を固定した能力指標」を要求し、§13 は計算方法を未確定としている。**根拠のない換算係数を発明せずに固定する**方針を採る。

対象は平地の過去走のみ。`result_ijyo_cd` が `1` / `2` / `3` / `4` / `5` の走は集計から除外する（12.2 節）。`6`（落馬再騎乗）と `7`（降着）は含める。

| 項目 | 定義 |
|---|---|
| `margin_sec` | その走の1着馬とのタイム差（秒）。1着馬は `0.0`。復元手順は 4.1.1 節の `margin_sec()` |
| `recent_margin_wavg` | 直近5走の `margin_sec` を新しい順に 5,4,3,2,1 の重みで加重平均。**使った走数を併記** |
| `recent_class_level_wavg` | 同じ重みでクラス順序値を加重平均 |
| `best_class_level` | 直近5走のクラス順序値の最大 |
| `field_rank` | 今回の出走馬内での `recent_margin_wavg` 昇順の順位。同値は平均順位。`null` の馬は最下位扱いにせず `null` |

`k = 5` は `config.py` の設定値。元仕様 §4.1 が「直近5〜10走を初期案とする」としているため、5 を既定に 10 まで変えられる形にする。

**クラス補正を秒に換算しない。** 換算係数には根拠がなく、元仕様 §4.3 が「補正方法と比較対象をコードで定義してから追加する」と定めている。代わりに着差とクラス順序値を別項目として並べて渡し（§4.1「各走の条件をセットにする」）、比較は Jev に任せ、重みは第3層の学習で決める。

**走破時計の馬場差補正は初期版に入れない。** 同じ理由。

### 6.2 クラス順序値

`jyoken_cd_youngest`（最若年条件）と `grade_cd` からの機械的写像。写像表は `config.py` に置く。

| 順序値 | クラス |
|---|---|
| 0 | 新馬・未出走 |
| 1 | 未勝利 |
| 2 | 1勝クラス（500万下） |
| 3 | 2勝クラス（1000万下） |
| 4 | 3勝クラス（1600万下） |
| 5 | オープン・リステッド・グレードのない重賞 |
| 6 | G3 |
| 7 | G2 |
| 8 | G1 |

判定順（上から評価し、最初に一致したものを採る）:

```python
def class_level(grade_cd: str, jyoken_cd_youngest: str) -> int | None:
    # 1. グレードが付いていればそれが最優先
    #    D（グレードのない重賞）は 5、E（重賞以外の特別競走）はクラスを表さないので後段へ
    #    F/G/H は障害なので対象外（呼び出し側で除外済み）
    by_grade = {"A": 8, "B": 7, "C": 6, "L": 5, "D": 5}
    if grade_cd in by_grade:
        return by_grade[grade_cd]

    # 2. 競走条件コード
    code = jyoken_cd_youngest
    if code == "701" or code == "702":   # 新馬・未出走
        return 0
    if code == "703":                    # 未勝利
        return 1
    if code == "999":                    # オープン
        return 5
    if code.isdigit() and code != "000":
        # 001-100 は収得賞金による条件。005/010/016 が 1勝/2勝/3勝クラス。
        # 旧 100万下～400万下 などの下位コードは 1勝クラス相当に寄せる。
        v = int(code)
        if v <= 5:
            return 2
        if v <= 10:
            return 3
        if v <= 16:
            return 4
        return 5                         # 1600万超の条件戦は OP 相当

    return None                          # 決まらない場合は 0 にしない
```

`GradeCD` が `E`（重賞以外の特別競走）やスペース（一般競走）の場合は競走条件コードで決まる。どちらからも決まらない場合は `null`（0 にしない）。

### 6.3 条件別成績

`as_of` より前の平地の走を対象に、次の区分ごとに集計する。

| 区分 | 定義 |
|---|---|
| `same_distance` | 今回の距離 ±200m 以内 |
| `exact_distance` | 今回の距離と同一 |
| `same_venue` | 今回と同じ `jyo_cd` |
| `same_track_kind` | 今回と同じ `turf` / `dirt` |
| `same_baba` | 今回と同じ `baba_cd`（かつ同じ `track_kind`） |

#### 6.3.1 馬場状態が未入力のとき `same_baba` は 0戦 ではない

`baba` は任意入力で（§14.1）、**予想時点を前日に固定した以上、馬場状態はほぼ常に未確定**。
つまりこれは例外経路ではなく既定の経路であり、この系が出すすべての予想が通る。

`0戦0勝` は「この馬はこの馬場状態で走ったことがない」という**馬についての主張**。
馬場状態が未入力のときに言えるのは「今回の馬場状態が分からない」だけで、馬については何も言えていない。
§14.1 の「未入力は欠損として明示」を満たすには、0 と「定義できない」を型で分ける必要がある。

`ConditionRecord` は `inapplicable_reason: str | None` を持つ。`None` でないとき、
`starts` / `wins` / `top3` は `0` ではなく `None`——区別せずに数を読むコードが黙って 0 を読むのではなく落ちるように。

| 状況 | `inapplicable_reason` | `starts` |
|---|---|---|
| 馬場状態が未入力 | `"今回の馬場状態が未入力のため比較できない"` | `None` |
| 馬場状態が入力済み、該当する走が無い | `None` | `0`（本物の0。従来どおり件数で表示する） |
| 馬場状態が入力済み、該当する走がある | `None` | 件数 |

`by_condition` のキーは**常に5つ**。`same_baba` を省いてはいけない——§8.2 の
「プロンプトは state に存在するキーしか参照してよい」に反する（`course_going_fit` の指示文が名指ししている）。
`state` では6キーすべてを出し、値を `null` にする。

表示は件数の形を作らない。`同馬場状態 判定不能（今回の馬場状態が未入力のため比較できない）` を
判断材料不足の束に入れる。レース単位の注記も1件足す。

実データでの確認（2026-09-06 札幌ダート1700m、実際の馬場は良）: この規則が無いと
ファイアリービートとクルビデンスは実際には同馬場で 1戦1勝なのに `0戦0勝(3着内0)` と表示された。
この2頭は出力の1位と2位。

各区分で出す値:

```json
{
  "starts": 20,
  "wins": 10,
  "top3": 14,
  "margin_sec_mean": 0.42,
  "period": {"from": "2024-03-10", "to": "2026-09-14"},
  "inapplicable_reason": null
}
```

条件が定義できないときは同じ6キーで、件数が `null` になる（§6.3.1）:

```json
{
  "starts": null,
  "wins": null,
  "top3": null,
  "margin_sec_mean": null,
  "period": {"from": null, "to": null},
  "inapplicable_reason": "今回の馬場状態が未入力のため比較できない"
}
```

元仕様 §4.3 の「集計値だけではなく件数と比較対象を一緒に渡す」を満たすため、`starts` を必ず併記する。`starts` が 0 のときは率を計算せず、`wins` / `top3` を 0、`margin_sec_mean` を `null` にする。

対象期間は全期間（`as_of` まで）。§4.1 の「それぞれの対象期間を明示する」に従い、各集計に実際の `period` を入れる。

### 6.4 近況

| 項目 | 定義 |
|---|---|
| `days_since_last_run` | `as_of` − 直前走の `hasso_at`（日） |
| `last_run` | 直前走の条件一式（距離・トラック・馬場・クラス順序値・頭数・着順・着差・斤量・馬体重） |
| `distance_change_m` | 今回の距離 − 直前走の距離 |
| `futan_change_kg` | 今回の斤量 − 直前走の斤量 |
| `class_level_change` | 今回のクラス順序値 − 直前走のクラス順序値 |

**馬体重は直前走の値のみ。** 前日予想なので今回の馬体重（当日発表）は使わない。

### 6.5 脚質

各過去走の 4角相対位置 = `result_jyuni_4c / syusso_tosu`（0 に近いほど前）。

| 項目 | 定義 |
|---|---|
| `corner_pos_mean` | 直近5走の相対位置の平均 |
| `corner_pos_stdev` | 同・標準偏差（走数2未満なら `null`） |
| `corner_pos_starts` | 集計に使えた走数 |
| `corner_source` | 実際に使った列（`jyuni_4c` / `jyuni_3c` / `jyuni_2c`） |

4角が空のとき（コーナー数が少ないコース）は `jyuni_3c` → `jyuni_2c` の順に代替し、**どの列を使ったかを `corner_source` に記録する**。どれも無ければ全項目 `null`。

### 6.6 輸送

元仕様 §4.2 に従う。

- 起点は `SE.tozai_cd`（そのレース時点の所属）。`1` → 美浦、`2` → 栗東。それ以外（地方・海外）は `null`
- `estimated_transport_km`: トレセン → 競馬場の固定テーブル。**名前に `estimated_` を付けて推定値であることを明示する**。実際の出発地・経路・輸送日を証明しないと元仕様 §4.2 が言っているため、`state` 上の説明文にもそう書く
- `stay_status`: 既定 `"unknown"`。遠隔開催でも `"likely_stay"` のような推測値を入れない（§4.2「現地滞在が不明なら、当日輸送と決めつけない」）。確認済み情報を利用者が入力できる欄だけ用意し、入力があれば `"stay"` / `"same_day"` を入れる
- `past_long_transport`: 過去走のうち `estimated_transport_km` が閾値（既定 300km、設定値）以上だった走について、`starts` と `zogen_sa_kg` の平均・最小を出す。**件数を必ず併記**し、`state` 上の説明文で体重変化の原因を輸送に断定しない（§4.2）

固定テーブルは `config.py` に置き、各値に「代表的な陸路距離の概数」という注記を付ける。

### 6.7 欠損の扱い

- 値は `None` のまま。0 や不適性に置き換えない（§4.4）
- `data_quality.missing` に「どの馬のどの項目が欠けているか」をパス形式で列挙する
- 集計が 0 件のときは率を計算せず `null` にし、`starts: 0` を残す

## 7. state の形式

**dict を採用する。** CLAUDE.md の未確定事項「dict で構造化するか、整形済みテキストにするか」への答え。

理由:

1. 元仕様 §5.2 / §6.3 のプロンプト案がすでに `race` / `horses` / `target_horse` / `opponents_summary` といったキー名を参照して書かれている。dict ならキー名がそのまま判断材料の名前になり、質問文と一致する
2. §4.4 の「欠損値を0や不適性に置き換えない」を守りやすい。テキスト整形では欠損が空欄に潰れて 0 と区別できない。dict なら `null` のまま送れる
3. §4.3 の「集計値だけではなく件数と比較対象を一緒に渡す」が `{"starts": 20, "wins": 10, "period": {...}}` という形で自然に書ける

§5.1 の「同じ書式で入れる」に従い、`horses` は**全頭で同一のキー集合**を持つ。値が `null` でもキーを残す。

### 7.1 第1層の state

キー名は元仕様 §5.1 のものをそのまま使う。

```json
{
  "prediction_cutoff": "2026-09-20T18:00:00+09:00",
  "race": {
    "jyo": "中山", "track_kind": "turf", "kyori_m": 2000,
    "baba": "良", "class_level": 4, "class_name": "3勝クラス",
    "syusso_tosu": 12, "hasso_at": "2026-09-21T15:25:00+09:00"
  },
  "horses": {
    "1": {
      "bamei": "...", "sex_age": "牡4", "futan_kg": 57.0,
      "waku": "1", "kisyu": "...",
      "ability": {"recent_margin_sec_wavg": 0.35, "runs_used": 5,
                  "recent_class_level_wavg": 3.6, "best_class_level": 4,
                  "field_rank": 2},
      "recent": {"days_since_last_run": 35, "distance_change_m": 200,
                 "futan_change_kg": 1.0, "class_level_change": 0,
                 "last_run": {...}},
      "by_condition": {"same_distance": {...}, "same_venue": {...}, ...},
      "running_style": {"corner_pos_mean": 0.31, "corner_pos_stdev": 0.08,
                        "corner_pos_starts": 5, "corner_source": "jyuni_4c"},
      "transport": {"base": "美浦", "estimated_transport_km": 40,
                    "estimated_note": "代表的な陸路距離の概数。実際の経路・輸送日を示すものではない",
                    "stay_status": "unknown", "past_long_transport": {...}}
    },
    "2": { "...同一のキー集合..." }
  },
  "pace_definition": { "...任意..." },
  "data_quality": {
    "as_of": "2026-09-20T18:00:00+09:00",
    "data_version": "20260922T013000Z",
    "data_fetched_at": "2026-09-22T10:30:00+09:00",
    "missing": {"4": ["ability", "by_condition", "running_style"]},
    "notes": ["4番は馬マスタ未登録。過去実績なしとして扱った"]
  }
}
```

### 7.2 第2層の state

キー名は元仕様 §6.1 のものをそのまま使う。

```json
{
  "prediction_cutoff": "...",
  "race": { "...第1層と同じ..." },
  "target_horse": { "...第1層 horses の1頭分をそのまま..." },
  "opponents_summary": {
    "2": {"bamei": "...", "ability": {...簡略...}, "running_style": {...簡略...}},
    "3": { "..." }
  },
  "race_context": {
    "pace_probabilities": {"slow": 0.2, "standard": 0.5, "fast": 0.3},
    "note": "第1層で得た予測分布。確定した事実ではない"
  },
  "data_quality": { "...対象馬と相手の欠損・件数..." }
}
```

`race_context` は `pace` が OFF のとき省略する。ON のときは**3区分すべての確率**を渡す（§5.3「最大確率の展開を、確定した事実として扱わない」）。

`opponents_summary` は §6.1 の「相手全頭の短い比較情報」なので、能力指標と脚質だけの簡略版にする。対象馬自身も含める（相手の中での自分の位置が分かるように）。

## 8. 質問定義

`predict/questions.py` にプロンプト文を集約し、`PROMPT_VERSION` を同じファイルで管理する。元仕様 §8 / §12.7 が「プロンプトが変わった予想は別の記録とする」と定めているため。

### 8.1 第1層 `winner`

`Choice`。`criteria` は馬番 → 短い識別情報（馬名・性齢・斤量・騎手・能力指標の順位）。

プロンプトは元仕様 §5.2 の案をそのまま使う。

```
予想締切時点の情報に基づき、今回のレースの1着馬を予測してください。

判断材料：
race：今回のレース条件
horses：出走全頭の能力指標、近走内容、条件別実績、今回の出走条件

今回の出走馬同士を比較してください。
過去の着順だけでなく、クラス、着差、条件の違いを考慮してください。
条件別実績は、成績と出走数の両方を参照してください。
欠損している情報を、良い実績や悪い実績として扱わないでください。
馬名の知名度ではなく、入力された情報を根拠にしてください。
```

取得: `choice` / `probabilities`（全頭） / `confidence`。

### 8.2 第1層 `pace`（任意・既定 OFF）

`Choice(slow / standard / fast)`。

プロンプトは元仕様 §5.3 の案を土台にするが、**参照するキー名は本仕様 §7.1 の state に合わせて直す**。元の案は `horses.running_style_summary` を参照していたが、実際に組み立てる state のキーは `horses[<馬番>].running_style` である。存在しないキーを参照させると、モデルはその指示を無視するか自分で材料を選ぶ。どちらも静かに起こり、出力はもっともらしいままになる。

```text
race と、horses の各馬の running_style を参照し、
今回のレース前半のペース区分を予測してください。

区分の意味は pace_definition に従ってください。
出走全頭の先行傾向、枠順（waku）、距離、コース条件を判断材料にしてください。
```

**プロンプトが参照するキーは、必ず state に実在するものだけにする。** 元仕様の草案から引き写すときは、§7.1 の state のキー集合と突き合わせること。

`pace_definition` はコード側で定義する。学習期間のレースについて `races.csv` の `first_3f_sec`（= `RA.HaronTimeS3`、前半3ハロンの合計が元データに入っている）の三分位を取り、**値が小さいほど速いので下位1/3 を `fast`、上位1/3 を `slow`**、中間を `standard` とする。閾値は学習期間だけから計算し、使った層と件数を含めて `pace_definition` として state に入れる。

`first_3f_sec` は平地競走のみ設定されるため、`track_kind == "jump"` のレースは対象外（初期版は平地のみなので問題にならない）。

層別化は段階的に粗くする。`track_kind` × 距離帯（±200m）× `jyo_cd` で**30レース以上**あればその層の三分位を使い、足りなければ `track_kind` × 距離帯（`jyo_cd` を落とす）、それでも足りなければ `track_kind` のみ、に落とす。取得初期は数百レース規模なので最も細かい層は成立しない前提で書く。実際に使った層と件数を `pace_definition.stratum` と `pace_definition.race_count` に記録し、どの粒度の閾値だったか後から分かるようにする。

この定義なら確定結果（実際の `first_3f_sec`）と照合できるので、§5.3 の「結果との照合に使える定義を固定してから採用する」を満たす。

`winner` と同じリクエストにまとめる（§5.3、および §2.2「同じ state に対する複数の質問は1リクエストにまとめる」）。同一リクエスト内なので `winner` は `pace` の回答を参照しない（§2.2）。

### 8.3 第2層の5軸

すべて `Choice`。1頭1リクエストに5問をまとめる。プロンプトと選択肢は元仕様 §6.2 の表の通り。

| 質問ID | 選択肢キー |
|---|---|
| `distance_fit` | `positive` / `neutral` / `negative` / `insufficient` |
| `course_going_fit` | `positive` / `neutral` / `negative` / `insufficient` |
| `relative_ability` | `superior` / `comparable` / `inferior` / `insufficient` |
| `current_form` | `positive` / `neutral` / `negative` / `insufficient` |
| `transport_concern` | `concern` / `no_evidence` / `insufficient` |

`criteria` の説明文は §6.2 の距離適性の例に倣って全軸に書く。共通ルール（§6.2）を各説明文で守る:

- 質問文と選択肢の説明を一致させる
- 未経験を不適性と判定しない
- 「判断材料不足」と「中立」を分ける
- 出走数と比較する過去走の条件を確認できる形にする

`transport_concern` の選択肢が3つなのは、§6.2 の表が「懸念あり／懸念を示す材料なし／判断材料不足」の3区分を指定しているため。

### 8.4 実装しない質問

`top3_if_slow` / `top3_if_standard` / `top3_if_fast`（§6.3）は実装しない。§6.3 が「初期版ではこの評価を省略してよい」と明記しており、かつ「実際に起きなかった展開について、そのレースの結果だけで直接検証することはできない」ため検証の設計が別途必要。

## 9. 第2層の回答 → 数値特徴

「判断材料不足」を「中立」と混ぜないため（§6.2）、**1軸から2つの特徴**を作る。

| 軸 | 特徴1（`*_signed`） | 特徴2（`*_insufficient`） |
|---|---|---|
| `distance_fit` | `P(positive) − P(negative)` | `P(insufficient)` |
| `course_going_fit` | `P(positive) − P(negative)` | `P(insufficient)` |
| `relative_ability` | `P(superior) − P(inferior)` | `P(insufficient)` |
| `current_form` | `P(positive) − P(negative)` | `P(insufficient)` |
| `transport_concern` | `P(concern)`（2値なので差にしない） | `P(insufficient)` |

`confidence` は特徴に使わない（§2.1「`confidence` は…そのまま的中率として扱わない」）。記録には残す。

`choice`（最大確率の選択肢）も特徴に使わない。§2.2 の「前段で得た確率分布は、原則として最大確率の選択肢だけに置き換えずに残す」に従い、分布全体を使う。

## 10. 第3層の統合

### 10.1 2つの出力を常に両方作る

| 名前 | 定義 |
|---|---|
| `baseline` | 第1層の `probabilities` の降順。第2層を使わない |
| `model` | 条件付きロジット（10.2） |

両方を毎回計算して**両方を記録する**。元仕様 §9.1 の「比較基準との差」がそのまま出るため、`model` が `baseline` を上回らなければ第2層を採用しないという判断ができる（§10「各段階で基準を上回るかを確認する」）。

### 10.2 条件付きロジット

```
score_i = w0 * logit(clip(p_i, eps, 1-eps)) + Σ_k w_k * f_ik
P(i が勝つ) = exp(score_i) / Σ_j exp(score_j)
```

- `p_i` は第1層の勝率、`eps = 1e-6`
- `softmax` の分母はそのレースの出走馬のみ。**頭数が 8 でも 18 でも合計 1.0 を保つ**ため、元仕様 §13 最終項「第1層と第2層のスケールの揃え方」と §7.1「頭数によってスケールの釣り合いが変わる」が構造的に解消する
- スコアの内訳（各項の寄与 `w_k * f_ik`）を記録に残す。CLAUDE.md の「最終スコアの内訳が常に見える」を満たす

`f_ik` に入れる特徴を確定させる。列挙しない特徴は学習に使わない（`state` には入っていても第3層には入らない）。

第2層由来（9章、10列）:

```
distance_fit_signed        distance_fit_insufficient
course_going_fit_signed    course_going_fit_insufficient
relative_ability_signed    relative_ability_insufficient
current_form_signed        current_form_insufficient
transport_concern          transport_concern_insufficient
```

Python 側（6章、8列）:

```
recent_margin_wavg          直近5走の加重平均着差（秒）
recent_margin_runs_used     集計に使えた走数
recent_class_level_wavg     直近5走の加重平均クラス順序値
class_level_change          今回 − 前走のクラス順序値
days_since_last_run         前走からの日数
distance_change_m           今回 − 前走の距離
corner_pos_mean             4角相対位置の平均
estimated_transport_km      推定輸送距離
```

Python 側特徴を入れる理由は、これらが第1層の `state` にも入っているため二重評価（§7.1）になり得る一方で、Jev が数値比較を苦手とする可能性（元仕様 §14 の model-jaggedness）もあり、どちらが効くかは事前に決められないこと。dev 期間で**Python 側特徴を入れた場合と入れない場合を両方学習して比較し、改善しない特徴は落とす**（§7.2 の 3「各評価が順位の補正に役立つか調べる」/ 5「役立つ評価を採用する」）。

欠損（`null`）の扱い: 値を 0 に置き換えず、**その特徴を 0 にしたうえで対応する `*_missing` 指示変数を 1 にする**。0 埋めだけでは「値が 0」と「欠損」が区別できず §4.4 に反する。`*_missing` は上記 Python 側8列それぞれに付く（第2層特徴は必ず値が返るので不要）。

数値安定性: `softmax` は `score − max(score)` を取ってから `exp` する。

### 10.3 勝率として表示する条件

`model` の出力を勝率として表示するのは、較正が**別期間で検証できたときだけ**（§7.1「合計スコアを、そのまま勝率として表示しない」「合計を1に正規化するだけでは、確率の妥当性を保証できない」）。

未検証のあいだは `model` の値を「順位スコア」として扱い、**利用者向けの出力（`to_user_view`）に較正後の勝率を出さない**（§8 の「較正後の推定勝率（検証済みの場合のみ）」）。

> **改訂（2026-09-23）**: 当初この節は「記録の `calibrated_win_prob` は `null` にする」と書いていた。だが §11.2 は較正の「再正規化の前後とずれ」を記録せよと要求しており、両立しない——記録が `null` なら「前後」の後が無い。実際この矛盾のせいで較正は**本番経路に配線されないまま**になっていた（`apply_calibration` を呼ぶ口がどこにも無かった）。
>
> 改訂後の規則は2つに分ける。**記録には残す**（`calibrated_win_prob` と `data_quality.calibration`）。**表示には出さない**（`to_user_view` は較正後の確率を列挙しない。`tests/test_records_store.py::test_user_view_never_exposes_the_calibrated_probability` が守る）。**順位にも使わない**——isotonic は単調なので理屈の上では順位不変だが、`numpy.interp` は定義域の外で外挿せず端の値に張り付き、節点の少ない較正マップでは同値が量産されて順位が潰れる。張り付いた馬番は `data_quality.calibration.clipped_umaban` に名指しで残す。

## 11. 学習と較正

### 11.1 重みの学習

多項ロジットの最尤推定。レースごとに正解は勝ち馬1頭。

```
L(w) = Σ_races [ score_winner − log Σ_j exp(score_j) ] − λ * ||w||²
```

`scipy.optimize.minimize`（L-BFGS-B）で `-L` を最小化する。`λ` は調整期間で選ぶ。

sklearn に条件付きロジットが無いため自前実装する。**真の重みが既知の合成データで復元できることをテストで確認する。**

### 11.1.1 標準化

`f_ik` は学習前に標準化する（train 期間の平均・標準偏差）。ただし次は標準化しない。

- `logit(p_i)`（`w0` の対象）。すでに意味のあるスケールで、標準化すると `baseline` との比較で解釈しにくくなる
- `*_missing` 指示変数と `*_insufficient`。0/1 と 0–1 の確率なのでそのまま使う

標準偏差が 0 の特徴（train 期間で定数）は学習から落とし、落としたことを記録する。

### 11.1.2 重みの保存

学習結果は `data/weights/<weights_id>.json` に保存する。`weights_id` は学習実行時刻の `YYYYMMDDTHHMMSSZ`。

```json
{
  "weights_id": "20260922T020000Z",
  "fitted_at": "2026-09-22T11:00:00+09:00",
  "integration_version": "1",
  "feature_version": "1",
  "prompt_version": "1",
  "data_version": "20260922T013000Z",
  "splits": {"train_until": "...", "dev_until": "...", "calib_until": "..."},
  "lambda": 0.1,
  "feature_names": ["logit_layer1", "distance_fit_signed", "..."],
  "weights": {"logit_layer1": 1.02, "distance_fit_signed": 0.31, "...": 0.0},
  "standardization": {"distance_fit_signed": {"mean": 0.05, "std": 0.41}},
  "dropped_features": ["..."],
  "dev_metrics": {"...": 0.0},
  "calibration": {"method": "isotonic", "x": [...], "y": [...]}
}
```

`scripts/predict.py` は `--weights <weights_id>` で指定された重みを読む。指定がなければ `baseline` のみを出す。重みの `feature_version` / `prompt_version` が現在のコードと一致しない場合は**予想を拒否する**。プロンプトが変わった重みを黙って使わない（§12.7）。

### 11.2 較正

`model` の出力に isotonic regression を**較正専用期間**で当てる。

較正後はレース内合計が 1.0 から外れるので再正規化する。再正規化の前後を両方記録し、ずれの大きさも記録する（ずれが大きければ較正が信頼できない兆候）。

### 11.3 期間分割

元仕様 §9.2 に従い時系列順に4分割する。

| 期間 | 用途 |
|---|---|
| train | 重みの最尤推定 |
| dev | `λ` と採用する特徴の選択 |
| calib | isotonic の当てはめ |
| test | 最終評価。重み・特徴・較正の調整に**使わない** |

レース単位で分割するので「同じレースの馬を学習側と評価側に分割しない」（§9.2）は自動的に満たす。分割の境界日を記録に残す。

## 12. 検証（E）

### 12.1 指標

元仕様 §9.1 の5つすべてと、§9.1 が求める整合性チェック。

| 指標 | 定義 |
|---|---|
| `top_n_hit_rate` | 上位 n 頭に勝ち馬が含まれたレースの割合 |
| `top1_win_rate` | 最上位に選んだ馬が勝った割合 |
| `brier` | レースごとに `Σ_i (p_i − y_i)²`、全レース平均 |
| `log_loss` | レースごとに `−log p_winner`、全レース平均 |
| `calibration` | 予測確率をビン分割し、ビンごとの予測平均と実測勝率 |
| `expected_vs_actual_top_n` | 上位 n 頭の予測確率の合計の平均と、実測 `top_n_hit_rate` の比較（§9.1） |

すべて `baseline` と `model` の両方について出し、差を並べる（§9.1「比較基準との差」）。

`n` と対象条件を固定して比較する。**`n` を増やして包含率が上がることを改善として報告しない**（§9.1）。レポートには使った `n` を必ず書く。

### 12.2 例外ルール（ここで定義する）

元仕様 §9.1 が「同着・不成立などの例外の評価ルールは、実装前に定義する」と求めている。定義を `config.py` に置き、予測記録に適用したルールの版を残す。

異常区分コード（`result_ijyo_cd`）は 3 つに分けて扱う。「`"0"` 以外はすべて除外」では**降着（`7`）と落馬再騎乗（`6`）の確定着順を捨ててしまう**ため、細分する。

| `result_ijyo_cd` | 内容 | 扱い |
|---|---|---|
| `0` | 正常 | 通常 |
| `6` | 落馬再騎乗 | **通常**（完走していれば確定着順がある） |
| `7` | 降着 | **通常**（降着後の確定着順を使う） |
| `1` `2` `3` | 出走取消・発走除外・競走除外 | **出走集合から除外**（走っていない） |
| `4` `5` | 競走中止・失格 | **出走集合に含めるが勝ち馬になり得ない**。過去走の `margin_sec` 集計からは除外（タイムがない） |

`4` / `5` を出走集合に含める理由は、予想時点では出走する予定だった馬であり、分母から抜くと予想の対象集合が後から変わってしまうため。一方 `1` / `2` / `3` は出走自体がないので分母から抜く。

| 事象 | ルール |
|---|---|
| `result_ijyo_cd` が `1` / `2` / `3` | その馬を出走集合から除外し、`data_quality` に記録。予想前に判明していれば入力から除く |
| `result_ijyo_cd` が `4` / `5` | 出走集合に残し、勝ち馬ではないものとして評価。過去走としての `margin_sec` は `null` |
| 1着同着 | 同着馬すべてを勝ち馬として扱う。`log_loss` は同着馬の確率**合計**を使う。`top_n_hit_rate` は「いずれかが上位 n に入れば的中」。`brier` は同着馬の `y_i` を `1/同着頭数` にする |
| レース不成立（`nyusen_tosu` が 0、または勝ち馬不在） | そのレースを評価対象から除外。除外件数と理由を必ず報告 |
| 障害レース（`track_kind == "jump"`） | 対象外 |
| 出走馬が1頭のみ | 評価対象から除外（比較にならない） |

### 12.3 バックテストの部分失敗

元仕様 §12.7 は「Jev の一部呼び出しに失敗した場合、不完全な評価を正常完了として返さない」と定めるが、バックテストで1頭の失敗が数百レースを止めると使えない。**用途で分ける。**

| 用途 | 部分失敗時の扱い |
|---|---|
| 予想1件（`scripts/predict.py`） | 第2層の1頭でも失敗したら FAILED。フォールバックしない（§12.7 の初期版ルール） |
| バックテスト（`scripts/backtest.py`） | そのレースを `skipped` として記録し続行。**スキップ件数と理由を最終レポートに必ず出す** |

バックテストのレポートでスキップを黙って落とさないことが重要。スキップが偏っていれば指標が歪む。

## 13. 予測記録

元仕様 §8 の項目をすべて保持する。JSONL（1レース1行）で `data/predictions/` に保存する。

```json
{
  "record_id": "...",
  "race_id": "2026092106050311",
  "predicted_at": "2026-09-20T18:05:00+09:00",
  "prediction_cutoff": "2026-09-20T18:00:00+09:00",
  "n": 3,
  "versions": {
    "data_version": "20260922T013000Z",
    "schema_version": "1",
    "model": "jev-1.13.0",
    "prompt_version": "1",
    "feature_version": "1",
    "integration_version": "1",
    "exception_rules_version": "1",
    "weights_id": "..."
  },
  "horses": [
    {
      "umaban": "1", "bamei": "...",
      "rank_baseline": 2, "rank_model": 1,
      "layer1_win_prob": 0.18,
      "model_score": 1.42,
      "model_win_prob": 0.24,
      "calibrated_win_prob": null,
      "score_breakdown": {"layer1_logit": 0.85, "distance_fit_signed": 0.31, "...": 0.0},
      "layer2": {
        "distance_fit": {"choice": "positive",
                         "probabilities": {"positive": 0.62, "neutral": 0.2,
                                           "negative": 0.1, "insufficient": 0.08},
                         "confidence": 0.62},
        "...": {}
      },
      "positives": ["同距離 8戦3勝", "..."],
      "negatives": ["..."],
      "insufficient": ["running_style（集計対象0走）"]
    }
  ],
  "layer1": {"confidence": 0.31, "pace_probabilities": null},
  "data_quality": {...},
  "usage": {"input_tokens": 12345, "requests": 13},
  "status": "SUCCEEDED"
}
```

- **全頭を保存する。表示は上位 n 頭**（§8）
- `positives` / `negatives` / `insufficient` は、取得した分類と Python 側の集計から構成する。Jev は自由文の理由を生成しないため（§8）、テンプレートで組む
- `versions` のいずれかが変わった予想は別の記録にする（§12.7）
- 元仕様 §8 の「内部保存用の予測記録と、利用者へ返すレスポンスを分ける」は、利用者向け API が C/D のスコープなので本設計では記録側だけを作る。ただし記録から表示用に絞る関数 `to_user_view(record)` を用意し、`state` 全体・生レスポンス・元 CSV 由来の生レコードを含めないことをテストで固定する

## 14. 入力（§12.5 のうちコアが担う部分）

Web 画面（D）は作らないが、**構造化入力の型と検証と照合はコアの責務**なので実装する。CLI が JSON / YAML ファイルを受け取る。

### 14.1 RaceInput

| 分類 | 項目 | 必須 |
|---|---|---|
| レース識別 | `date` / `jyo` / `race_num` | 必須 |
| 条件 | `track_kind` / `kyori_m` / `syusso_tosu` | 必須 |
| 追加条件 | `class_name` / `baba` / `hasso_at` | 任意（未入力は欠損として明示） |
| 出走馬 | 全頭の `umaban` / `bamei` | 必須 |
| 出走条件 | `wakuban` / `sex_age` / `futan_kg` / `kisyu` | 任意 |
| 予想条件 | `prediction_cutoff` / `n` | `prediction_cutoff` 必須、`n` は既定 3 |

当日情報（馬体重・取消・騎手変更・オッズ）は前日予想なので初期版の入力項目に含めない。§12.5 の表にはあるが、予想時点を前日に固定した以上使えない。取消だけは「予想前に判明していれば入力から除く」形で扱う（12.2 節）。

### 14.2 検証（§12.5）

予想開始前に修正を求めるもの:

- `umaban` の重複
- `syusso_tosu` と出走馬リストの件数の不一致
- `kyori_m` が正でない、現実的な範囲外
- 必須項目の欠落
- `prediction_cutoff` が `hasso_at` 以降

任意項目の欠損は許容し、`data_quality` に反映する。

### 14.3 馬名照合（§12.5）

1. 馬名を正規化する（NFKC、全角→半角の統一、記号・空白・長音符の除去）
2. `horses.csv` の `bamei` / `bamei_kana` の**正規化済みインデックス**と照合

正規化の関数は `src/keiba/names.py` に置く。`data/repository.py`（データ層）と `input/match.py`（入力層）の両方が使うので、どちらかに置くと層が逆向きに依存する。

**インデックスは `DataStore` が一度だけ作る。** 照合のたびに馬マスタ全行を正規化すると、1レース18頭 × マスタ十万行で毎回数百万回の正規化になる。`DataStore` が遅延構築して保持し、同じ版への問い合わせで使い回す。

**`bamei` と `bamei_kana` の両方を索引に入れる。** 利用者が半角カナで入力することがあり、`bamei` だけでは拾えない。

長音符を落とすのは表記ゆれ対策だが、**落とさないと実データが一致しない**。JRA の馬名には長音符が頻出する（`オルフェーヴル` / `ゴールドシップ` など）。正規化した名前を素の `bamei` と比較すると、そうした馬はすべて「未登録」になる。
3. 複数候補があれば `sex_age`（性齢）で絞る。**馬齢を先に、性別を後に**当てる
4. 馬齢に合う候補が1頭も無ければ「過去実績なし」（`unregistered`）として続行し、注記に申告馬齢と同名候補の生年を残す
5. 性別で絞って0件になっても、馬齢で絞った候補が1頭なら `matched` として続行し、食い違いを注記に残す
6. それ以外で1件に決まらなければ自動確定しない。候補一覧を返してエラーにする（§12.5「曖昧な候補を自動確定しない」）
7. 候補ゼロなら「過去実績なし」として扱い、`data_quality.notes` に記録する。**別の馬の履歴で補完しない**（§12.5）

#### 14.3.0 馬齢で絞る（当初の判断を実測で覆した）

**この節は実装中に改訂した。** 当初は「絞るのは性別の文字だけで、馬齢は使わない。JRA の馬齢は
暦年ベースなので変換を誤ると正しい馬を候補から消す」と定めていた。実データで3つのことが分かって覆した。

| 測ったこと | 結果 |
|---|---|
| `馬齢 = レースの年 - 生年` の成立 | **950行中950行（100.00%）**。別途40レース204頭でも **204/204** |
| マスタ全体で正規化名が複数の馬を指すもの | 7,371件 / 190,148名（3.9%） |
| 同名グループ2,000件のうち生年が重なるもの | **0件** |
| 性別だけで絞ったときの中止率 | **480レース中151件（31.5%）** |
| 馬齢を足したあとの中止率 | **0件（0.0%）** |

「変換を誤ると候補を消す」という懸念は正しかったが、**変換規則が実データで例外なく成立する**ことが
確かめられた。そして性別だけに頼ると、同名馬（マスタは1936年まで遡る）で 31.5% のレースが止まる。

馬齢を性別より**先**に当てるのは、馬齢のほうが分離力が強い（生年の重複が0件）ため。
`birth_year` が不明な候補は馬齢で落とさない。

#### 14.3.0.1 馬齢に合う候補が無ければ `unregistered`

2024年生まれの2歳馬に対して候補が1989年生と1996年生だけ、という状況が実データで31件あった。
**明らかに別の馬**なので、候補を残して `ambiguous` にするとレース全体が止まる。
馬齢の規則が100%成立する以上、「馬齢に合う候補が無い」は「この馬はデータに居ない」を意味する。

打ち間違いへの備えは注記で足りる（申告馬齢と、同名候補の生年を並べる）。

#### 14.3.0.2 性別だけの食い違いは止めない

マスタの性別は古びる。実測で**出走馬の1.6%（82頭）**が、`SE`（レース時点）では セン なのに
`UM`（マスタ）では 牡 だった。マスタ行の更新が去勢より前で止まっているため。
したがって `sex_cd` は `as_of` 以前の最新 `SE` 行から採り、マスタは `SE` に無い欄だけを供給する
（`HorseMaster.sex_cd_source` にどちらから採ったかを残す）。

そのうえで、名前と馬齢で1頭に特定できているなら識別は成功している。性別は識別の**裏付け**であって
識別子ではない。レース全体を捨てるのは不均衡なので `matched` + 注記にする。
当初 `conflict` を設けた意図（「性別の打ち間違いが黙って `unregistered` になる」のを防ぐ）は
注記で満たされる——黙って続行しない。

### 14.3.1 照合の結果は4通りある

| 状態 | 意味 | `ketto_num` | 続行できるか |
|---|---|---|---|
| `matched` | 候補がちょうど1頭 | その馬の値 | はい |
| `unregistered` | 名前が一致する馬がマスタに**いない** | `None` | はい（過去実績なしとして扱う） |
| `ambiguous` | 名前が一致する馬が2頭以上あり、馬齢でも性別でも絞れない | `None` | **いいえ**（利用者が解消する） |
| `conflict` | 名前は一致したが、入力された性別がどの候補とも合わず、かつ候補が2頭以上 | `None` | **いいえ**（利用者が解消する） |

**候補が1頭のときの性別の食い違いは `conflict` にしない**（§14.3.0.2）。
**馬齢に合う候補が1頭も無い場合は `unregistered`**（§14.3.0.1）。

`conflict` を `unregistered` と区別する理由: 名前がマスタにある馬について「未登録」と記録するのは**データについて偽を述べること**になる。性別の1文字を打ち間違えただけで、その馬の過去実績が黙って空になり、`data_quality` は「マスタ未登録」と報告する。入力とマスタが食い違っているなら、どちらが誤りかはコードには決められないので止める。

`ambiguous` と `conflict` はどちらも `AmbiguousMatchError`（「1頭に確定できなかった馬がある」という意味）で送出し、例外はレース全頭の `MatchResult` を持つ。1頭ずつ往復させず、利用者が全部の問題を一度に見られるようにする。

## 15. テスト戦略

TDD で進める。各モジュールに対応するテストを実装より先に書く。

### 15.1 未来情報の混入防止（最優先）

- `as_of` より後のレースが過去走に混ざらない
- 対象レースの結果値が `state` の JSON 文字列に一度も現れない
- `Repository` の全公開メソッドが `as_of` を必須引数に持つ（`inspect.signature` で検査）
- `UM` の累計集計列が CSV にも `state` にも現れない
- `to_user_view(record)` が `state` 全体・生レスポンス・生レコードを含まない
- **`RaceInput` に写してよい列の許可リストを固定する**（下記）

#### 15.1.1 sentinel 方式では捕まらない向きがある

「対象レースの結果値が `state` に現れない」テストは、`Repository` から特徴量層へ向かう漏洩を
捕まえる。**呼び出し側が `RaceInput` に結果由来の値を詰めて渡す向きは捕まえられない。**
テスト自身が `RaceInput` を組み立てるので、その経路の値は sentinel になりえない。

このプロジェクトで実際に見つかった漏洩は2件とも**こちら向き**だった。
バックテストが (1) レース後に確定した馬場状態を渡していた、(2) 取消馬を除いた出走集合
（= `syusso_tosu` と定義上同一）を渡していた。どちらも per-task のレビューからは見えなかった。

したがって `evaluate/backtest.race_input_from_history` に対して、
**`RaceRow` / `ResultRow` から `RaceInput` に写してよいフィールドの許可リストを固定するテスト**を置く。
新しいフィールドを写した瞬間に落ちる形にすること。sentinel 集合を増やす方式ではこの向きは永久に捕まらない。

### 15.2 レコード分解

数値項目の形式は 4.1.1 節で `JV-Data仕様書_4.9.0.1.xlsx` の「フォーマット」シートから確定済みなので、合成フィクスチャで回帰テストする。

**2026-09-22 に実データで照合済み。** `scripts/update_data.py --fromdate 20260801000000 --dry-run --verify-time-fields` を実 JV-Link に対して実行し、`RA` 564件 / `SE` 7455件 を読み込んで **479レース中、不一致 0件**。4.1.1 節の `margin_sec()` の解釈（符号1文字 + 0.1秒単位3桁、1着は符号 `-` で値は2着馬との差なので margin は `0.0`）が実データと一致することを確認した。

ただし**仕様書の記述とデータの実態がずれる可能性は残る**ため、実データでの確認を1回だけ挟む。

1. **実レコードでの突き合わせ（1回、`scripts/update_data.py` の開発時）**: 取得した `SE` から数十件を取り出し、同一レース内で「`KakuteiJyuni == 1` の馬の `Time`」と「2着馬の `Time` − 1着馬の `Time`」を計算して、2着馬の `TimeDiff` の値と一致するかを確認する。一致すれば 4.1.1 節の解釈が正しい。差があれば 4.1.1 節を実態に合わせて直す。
2. **合成フィクスチャでの回帰テスト**: 確定した解釈に基づく合成レコードを `tests/fixtures/` に置き、以降の変更で解釈が壊れないことを守る。

この突き合わせは `scripts/update_data.py --verify-time-fields` として実装し、いつでも再実行できるようにする。手作業のメモにしない。

JRA-VAN のデータは再配布不可なので、コミットするフィクスチャに実在の馬名・レース名・血統登録番号を含めない。バイト列の構造とオフセットだけを再現した合成レコードにする。`vendor/` と `data/` は `.gitignore` 済みなのでそのまま使えない。

### 15.3 その他

| 対象 | テスト |
|---|---|
| 条件付きロジット | 真の重みが既知の合成データで復元できる |
| `softmax` | 頭数が変わっても合計 1.0、`score` に大きな値があっても overflow しない |
| 指標 | 手計算した小さな例と一致 |
| 例外ルール | 12.2 節の各行 |
| 入力検証 | 14.2 節の各行 |
| 馬名照合 | 同名馬の曖昧照合を自動確定しない／未登録は続行する |
| 能力指標 | 走数が 5 未満のとき、0 件のとき、全て `null` のとき |
| 条件別成績 | `starts: 0` で率を計算しない |
| 脚質 | 4角が空のとき `jyuni_3c` に落ち、`corner_source` に記録される |
| `DataKubun` の選択 | `"7"` があれば `"3"`–`"6"` より優先される／`"0"` `"9"` `"A"` `"B"` を除外する／同一 `(race_id, umaban, data_kubun)` は `make_date` の新しい行を採る |
| `margin_sec` の復元 | `"-005"` → `0.0`（1着）／`"+003"` → `0.3`／`"9999"` → `None`／スペース → `None` |
| `class_level` | `grade_cd="A"` → 8／`grade_cd="E"` + `jyoken="016"` → 4／`jyoken="703"` → 1／`jyoken="000"` → `None` |
| `track_kind` | `"11"` → `turf`／`"23"` → `dirt`／`"54"` → `jump`／`"00"` → `None` |
| 頭数 | 予想時点で `syusso_tosu` を参照しない（`toroku_tosu` と利用者入力だけを使う） |
| 版付き保存 | `.partial` を読まない／`status != "complete"` を拒否する |
| `ask_parallel` | `return_exceptions=True` で部分失敗が値として返る／既定 `False` の挙動が変わらない |

### 15.4 Jev への実アクセスを含むテスト

単体テストは API に触らない。実アクセスは既存の `scripts/smoke_test.py` と、新設する `scripts/predict.py` の手動実行で確認する。CI で API キーを使わない。

## 16. CLI

| スクリプト | 役割 |
|---|---|
| `scripts/update_data.py` | JV-Link から取得して CSV 版付き保存。`--dataspec` `--fromdate` `--option`。`--verify-time-fields` で 15.2 節の突き合わせを実行 |
| `scripts/predict.py` | 1レース予想。`--input race.json` `--n 3` `--layer2/--no-layer2` `--weights <id>` |
| `scripts/fit_weights.py` | 期間指定で重み学習 + 較正。`--train-until` `--dev-until` `--calib-until` |
| `scripts/backtest.py` | 期間指定でバックテスト。`--from` `--to` `--n` `--layer2/--no-layer2` |

`scripts/smoke_test.py` は変更しない。

Windows コンソールの文字化け対策（`smoke_test.py` の `_enable_utf8_output`）は共通化して各 CLI から使う。

## 17. スコープ外（明示）

| 項目 | 理由 |
|---|---|
| AWS 一式（Cognito / API GW / Lambda / SQS / DynamoDB / S3 / CloudFront） | 別サブシステム C |
| Web 画面・予想履歴画面 | 別サブシステム D |
| スクショ入力・OCR | 別サブシステム G |
| 展開別3着内予測（`top3_if_*`） | §6.3 が初期版で省略可と明記 |
| 実着順の組合せ予測 | §11 が最小構成外と明記 |
| 補正時計・馬場差補正 | §4.3 が定義を固定してから追加と明記 |
| 調教・血統・騎手の条件別成績 | §4.1 が「後から追加」に分類 |
| 当日情報による更新 | 予想時点を前日に固定した |
| Parquet 読み取り層・差分更新・索引 | B の残り |

## 18. 未確定のまま残るもの

本設計で決めなかったもの。実装後の検証で決める。

- `k`（対象走数）の最適値。既定 5、設定で 10 まで
- `λ`（正則化強度）。dev 期間で選ぶ
- 採用する特徴の組み合わせ。dev 期間で選ぶ
- `n` の運用値。既定 3
- `pace` を採用するか。実装はするが既定 OFF、検証で判断
- 較正を表示に使えるか。test 期間で判断
- 輸送距離の固定テーブルの値の精度

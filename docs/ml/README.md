# 機械学習系（血統モデル ほか）

既存の Jev 統計モデル（`src/keiba/`）と**独立に**、CatBoost を中心とした個別モデルを
`src/keiba_ml/` に置く。本書は Phase 1（データ基盤）と Phase 2（血統モデル）の
使い方と設計判断をまとめる。

> **実データでの動作は未確認。** 開発環境に JV-Link と SDK（`vendor/`）が無いため、
> 合成データ（`tests/fixtures/ml_synthetic.py`）でのみ端から端まで検証した。
> 自分の PC で最初に「[1. まず確かめること](#1-まず確かめること)」を実行すること。

## 0. 置き場所

| パス | 役割 |
|---|---|
| `src/keiba_ml/ingest/` | 血統レコード SK / HN / UM(3代血統) の分解と版付き保存 |
| `src/keiba_ml/dataset/` | **Phase 1**。長期データの読み出し・コード体系の統一・血統の階層化・整合性検査・時系列分割 |
| `src/keiba_ml/features/` | 特徴量生成（前処理と分離。as-of で未来情報を使わない） |
| `src/keiba_ml/models/` | 血統 CatBoost と、全モデル共通の予測出力形式（`models/base.py`） |
| `src/keiba_ml/evaluate/` | 上位3頭の指標・較正・早熟/晩成の集計 |
| `src/keiba_ml/experiment.py` | 学習期間ごとの実験と比較表 |
| `scripts/ml/` | CLI（下記） |
| `configs/ml/pedigree.toml` | 設定。学習期間・テスト期間・世代数・目的関数など |
| `tests/ml/` | テスト（109件）。catboost が無ければ結合テストだけ飛ぶ |
| `data/ml/` | 出力（`.gitignore` 済み。JRA-VAN のデータは再配布不可） |

既存モデルとの関係:

- `keiba_ml` → `keiba` の import は「JV-Link の呼び出し層」「既存データ版の場所と読み方」
  「コード表」だけ。予想ロジック・重み・記録には触れない。
- `keiba` → `keiba_ml` の import は作らない。既存のデータ版（`data/versions/`）にも書かない。
  既存のテスト 1109 件は catboost の有無に関係なく通る。

## 1. まず確かめること

```powershell
# 依存（初回のみ）
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ml.txt

# (a) 血統レコードの位置を SDK の構造体定義と照合する。NG があれば
#     src/keiba_ml/ingest/pedigree_records.py の定数を SDK に合わせて直す
.venv\Scripts\python.exe scripts\ml\check_offsets.py
.venv\Scripts\python.exe scripts\ml\check_offsets.py --show    # SDK 側で拾えた項目を全部見る

# (b) テスト
.venv\Scripts\python.exe -m pytest tests\ml
```

位置は JV-Data仕様書 4.9 から写したもの（SK 208バイト・HN 251バイト・UM 1609バイト）。
取得時に長さが合わないレコードがあると `update_pedigree.py` が警告し、manifest の
`length_mismatch` に件数が残る。

## 2. 実行手順

```powershell
# Phase 1-1. レースデータ（既存の取得スクリプト）。30〜40年に広げるなら fromdate を戻す。
#            期間を分けて取った版は configs/ml/pedigree.toml の data.race_versions に並べれば束ねられる
.venv\Scripts\python.exe scripts\update_data.py --dataspec RACE --fromdate 19860101000000

# Phase 1-2. 血統（BLOD = SK 産駒マスタ / HN 繁殖馬マスタ）。初回はセットアップ取得が要ることがある
.venv\Scripts\python.exe scripts\ml\update_pedigree.py --fromdate 19860101000000 --option 4
#            UM の3代血統でも補うなら: --dataspec BLOD --dataspec DIFN
#            （UM が DIFN と DIFF のどちらで来るかは契約・取得状況で違う。未確認。0件なら DIFF を試す）

# Phase 1-3. データセット作成 + 整合性検査（FAIL があると終了コード 1）
.venv\Scripts\python.exe scripts\ml\build_dataset.py

# Phase 2-1. 10年で基準モデル
.venv\Scripts\python.exe scripts\ml\train_pedigree.py --years 10

# Phase 2-2. 学習期間の比較（10/20/30/40年。テスト期間は固定）
.venv\Scripts\python.exe scripts\ml\train_pedigree.py --compare

# 目的関数の比較（未決定事項）
.venv\Scripts\python.exe scripts\ml\train_pedigree.py --years 10 --loss YetiRank

# 早熟・晩成の集計（父ごとの年齢別 A/E。検証期間より前だけで集計）
.venv\Scripts\python.exe scripts\ml\growth_report.py --path S

# 出馬表のレースを予想（人が判断するための表。父・母父も並べる）
.venv\Scripts\python.exe scripts\ml\predict_pedigree.py --model data\ml\runs\<run_id>\10y
```

**テスト期間は `configs/ml/pedigree.toml` の `[split]` で決める**（既定は 2025年の1年間、
その直前1年が検証期間）。手元のデータが学習期間に足りないと、比較表の `truncated` が
True になり `actual_train_start` に実際の開始日が出る。40年と書いて実は10年で学習した、
ということが表から分かるようにしてある。

目安（合成データで計測）: 特徴量生成は 10年・37万行で約1分。40年なら1期間あたり
4〜5分。CatBoost の学習時間は行数と iterations 次第。

## 3. 出力

```
data/ml/
  pedigree/versions/<id>/          sk.csv hn.csv um_ketto.csv manifest.json
  datasets/<id>/                   runners.parquet pedigree.parquet ancestor_names.json
                                   integrity.json manifest.json
  runs/<run_id>/config.json        実行時の設定（そのまま再現できる）
  runs/<run_id>/comparison.md|csv  学習期間 × モデル の比較表
  runs/<run_id>/<10y|20y...>/      model.cbm model_meta.json metrics.json checks.json
                                   predictions_valid.parquet predictions_test.parquet
                                   feature_importance.csv
  reports/growth_*.csv             早熟・晩成の集計
```

予測は全モデル共通の形式（`models/base.py` の `PREDICTION_COLUMNS`）で保存する。
Phase 4 の統合モデルはこの表だけを受け取ればよい。

## 4. 設計

### 4.1 Phase 1: データ基盤

| 開発方針の項目 | 実装 |
|---|---|
| コード体系の統一 | `dataset/codes.py`。馬齢（2000年以前は数え年）・競走条件（年代で違う収得賞金条件 → 段階 0〜5）・未設定値（空白/ゼロ埋め → 欠損）。**所有者が確認済みの変更は `JYOKEN_TIER_OVERRIDES` などに足すこと** |
| 欠損値処理 | 欠損は欠損のまま持つ（0 で埋めない）。カテゴリは `"NA"`、数値は NaN で CatBoost に渡す |
| ID の関連付け | 馬 = 血統登録番号、祖先 = 繁殖登録番号、レース = race_id |
| 血統の階層化（三世代以上） | `dataset/pedigree.py`。3世代14枠は SK を優先・UM で補完。4世代目以降は HN の父母リンクを再帰的に辿る（既定5世代） |
| 未来情報の除外 | 年齢は「レースの年 − 生年」。産駒成績は全て as-of（下記） |
| 時系列分割 | `dataset/split.py`。学習 → 検証 → テスト（固定） |
| 整合性チェック | `dataset/integrity.py`。下表 |

整合性検査（`build_dataset.py` が実行し `integrity.json` に残す）:

| 検査 | 内容 |
|---|---|
| `unique_runner` | 同じレース・同じ馬番／同じ馬の重複が無い |
| `targets` | 結果のあるレースで3着以内が3頭未満のものが少ない（取り込み漏れの検出） |
| `age` / `birth_before_race` | 年齢が妥当、生年がレースより前 |
| `age_convention` | 表記の馬齢を変換した値と「レースの年 − 生年」の**年別の一致率**。数え年の境目（`codes.kazoe_age_until_year`）の確認 |
| `pedigree_coverage` | 父が分かる出走の割合（年別） |
| `pedigree_parent_link` | SK/UM の「父父」と HN から辿った「父の父」などの一致率。**位置がずれていればここが崩れる** |

学習直前にも検査する（`checks.json`）: 特徴量の列名に結果・オッズ・目的変数が無いこと、
分割が日付で重ならないこと、as-of 産駒成績を無作為100行で総当たりに計算し直して一致すること。

### 4.2 Phase 2: 血統モデル

特徴量（`features/pedigree.py`）:

1. **祖先のカテゴリ** `ped_<位置>`（3世代14枠の繁殖登録番号）。CatBoost の
   ordered target statistics で数値化する。学習データは日付順に並べて `has_time=True`。
2. **祖先の産駒成績（as-of）** `st_<位置>_...`（既定の位置: 父 S・母 D・母父 DS・父父 SS）。
   産駒の3着以内数 ÷ 偶然の期待3着以内数（出走 n 頭なら min(3,n)/n の和）= A/E を、
   縮小推定（`shrinkage`）して出す。全体・芝ダ・距離区分・芝ダ×距離・馬場（良/道悪）・
   年齢区分・競馬場の条件別。
   - **予想対象のレース日より前**の結果だけを使う（同じ日も含めない）
   - **本人の過去走は除く**（`exclude_self`）。除かないと母の産駒成績 ≒ 本人の成績になり、
     血統モデルが過去成績モデルになってしまう
   - `stats_history = "window"` なら学習期間の開始日以降だけで集計する。学習期間の比較で
     「データ量の効果」を切り分けるため
3. **早熟・晩成** `st_<位置>_ae_age2 / _age3 / _age4p` と `st_<位置>_maturity`
   （= log(4歳以上の A/E ÷ 2歳の A/E)。正なら晩成寄り）。定義が未決定なので連続量のまま渡す。
4. **条件**: 芝ダ・距離・距離区分・馬場・競馬場・年齢・性・頭数・月。
   **馬自身の過去成績・騎手・オッズは入れない**（別の独立モデルの担当。オッズは設計原則5で分離）。
5. **近交**: 近交係数（Wright の近似）・クロス数・最も近いクロス（例 3×4 → 7）。

比較対象（学習しない基準）も同じテスト期間で評価する: 偶然（解析的な期待値）、
父の A/E だけで並べた順位（`sire_ae`）、父×芝ダ×距離の A/E（`sire_cond_ae`）。
CatBoost がこれらを超えないなら、学習の意味がない。

### 4.3 評価指標（`evaluate/metrics.py`）

| 指標 | 意味 |
|---|---|
| `hits_mean` | 予測上位3頭のうち3着以内に入った頭数の平均（0〜3）。95% 区間はレース単位のブートストラップ |
| `all3_rate` | 予測上位3頭がすべて3着以内（三連複ボックス3頭）の割合 |
| `winner_in_top3` / `top1_win` | 1着馬を上位3頭に含む率 / 1位予測の勝率 |
| `ndcg3` | 1着=3・2着=2・3着=1 の関連度での NDCG@3 |
| `brier` / `log_loss` / `ece` | Logloss で学習したときの3着以内確率の較正 |
| `*_random` | 同じレース構成で無作為に3頭選んだときの期待値。**必ずこれとの差で読む** |

同着は実際の3着以内を4頭以上にする（`all3` は予測3頭がその集合に含まれれば的中）。

## 5. 未決定事項の扱い

| 未決定事項（開発方針 §9） | このコードでの扱い |
|---|---|
| CatBoost の目的関数 | `model.loss_function` で切替。Logloss（既定）と YetiRank などを同じ入口で比較できる |
| 血統のエンコーディング | CatBoost のカテゴリ（ordered TS）＋ as-of A/E の両方を入れてある。どちらが効くかは `feature_importance.csv` と、`pedigree.stat_paths` / `categorical_generations` を変えた比較で判断 |
| 階層ベイズの必要性 | 未実装。A/E の縮小推定（経験ベイズの最も軽い形）だけ入れてある |
| 晩成型・早熟型の定義 | ラベルは付けず連続量（`maturity`）。`growth_report.py` が定義を決める材料を出す |
| 三連複確率への変換・統合方式・購入基準 | Phase 4 以降。予測は共通形式で保存してあるので、統合側から読める |

## 6. 既知の限界

- **実データ未検証。** レコードの位置（§1）、`kazoe_age_until_year`、競走条件コードの
  年代別の対応は、実データで `check_offsets.py` と整合性レポートを見て確かめること。
- 4世代目以降は HN（繁殖馬マスタ）に父母が登録されている馬までしか辿れない。
  輸入種牡馬の祖先は欠けやすい（`fill_rate_by_generation` で確認）。
- `exclude_self` は「本人」だけを除く。全兄弟（同じ父母）の成績は母の産駒成績に入る。
  これは血統の情報として正当だが、兄弟が多い母では本人の能力と区別しにくい。
- 統合モデル用の未学習データ予測（out-of-fold）はまだ作っていない（Phase 4）。
  検証期間の予測（`predictions_valid.parquet`）は early stopping に使っているので、
  統合モデルの学習には使わないこと。

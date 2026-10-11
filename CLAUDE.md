# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクトの現状

2026-09-22 時点。**データ取得から予想・学習・検証まで端から端まで動く。** 実データで
100レースのバックテストを実行済み（テスト 1074 件）。

設計は `docs/superpowers/specs/2026-09-22-jev-prediction-core-design.md`（**拘束力のある仕様**）。

| パス | 役割 |
|---|---|
| `src/jev_client.py` | Jev への薄いアクセス層。キー解決・例外の日本語化・並列実行のみ。**予想ロジックと重みは置かない** |
| `src/keiba/ingest/` | JV-Link からの取得、レコード分解、版付き CSV 保存（レース・血統・払戻オッズの3系統）、生レコードの保存（`rawdump`） |
| `src/keiba/data/` | 版の読み出しと `Repository`。**全公開メソッドが `as_of` を必須キーワード引数に取る** |
| `src/keiba/input/` | 利用者入力の解析・検証・馬名照合 |
| `src/keiba/features/` | 数値化。`EntryRow` しか受け取らない（結果は型で入らない） |
| `src/keiba/predict/` | 第1層・第2層・統合・予想の組み立て |
| `src/keiba/learn/` | 条件付きロジットの最尤推定と isotonic 較正 |
| `src/keiba/evaluate/` | 指標・時系列分割・バックテスト |
| `src/keiba/store/` | 予想記録の保存と利用者向けビュー（**列挙方式**で生データを漏らさない） |
| `src/keiba/features/{course,history,dossier,dossier_markdown}.py` | 3着内系統の予想用資料。API を呼ばない |
| `src/keiba/predict/{top3,top3_questions,top3_baselines}.py` | 案A・案B と Python 基準 |
| `src/keiba/evaluate/{top3_metrics,top3_backtest,coverage}.py` | 3着内の採点・比較・データ充足度 |
| `src/keiba/store/top3_records.py` | 3着内予想の記録（`data/top3/`） |
| `scripts/` | `update_data` / `predict` / `fit_weights` / `backtest` / `smoke_test` / `dossier` / `backtest_top3` / `data_coverage` |
| `vendor/jravan-sdk-500-64bit/` | JRA-VAN SDK 5.0.0（64bit）の展開物。再配布不可のため `.gitignore` 済み |
| `data/` | 取得したデータ版・予想記録・重み。再配布不可のため `.gitignore` 済み |
| `.env` | `TYPESAFE_API_KEY`。コミットしない（`.env.example` が雛形） |

- git リポジトリは **`init` 済み・コミット0件**（branch は `master`）。
  最初のコミットを打つかは未決（方針決定なので所有者の判断を仰ぐ）。
  `.gitignore` の先頭スラッシュ漏れは 2026-09-23 に修正済み（`/data/` `/vendor/` `/tmp/`）。
  **元に戻さないこと** —— 先頭スラッシュが無いと任意の深さの `data/` に一致し、
  `src/keiba/data/` のソース3本が黙って除外される。
- ディレクトリ名は `house_racing` だが、対象は **horse racing（競馬）**。

### 版の現況（2026-09-23）

`PROMPT_VERSION` / `FEATURE_VERSION` / `RECORD_SCHEMA_VERSION` はいずれも **"3"**。
**出荷重み `20260922T135009Z`（どちらも "2"）は `check_compatible()` に弾かれる。**
次の課金実行で作る記録から学習し直す前提。経緯は
`tmp/keiba-report/report2-prompts-and-fix-plan.md` §9。

### 実データで測った性能（2026-09-22、100レース）

| 指標 | baseline（第1層のみ） | model（27特徴＋学習した重み＋較正） | 一様分布 |
|---|---|---|---|
| 上位3頭（held-out 11レース） | 45.5% | 45.5% | 23.9% |
| brier（held-out） | 1.0070 | **0.8891** | 0.9204 |
| log_loss（held-out） | 12.9761 | **2.2612** | 2.5591 |

**較正は効いており held-out でも成立する。順位の改善は示せていない。**
上位n頭の的中率は境界の同着に支配される（100レース中69件）ので、数千レース規模になるまでは
`brier` / `log_loss` で判断すること。詳細は `.superpowers/sdd/.../progress.md`。

順位については正確二項検定（ポアソン二項分布）で片側 p 値を出してある。baseline は重みを
使わないので100レース全部が held-out と等価で、30.0% 対 偶然 23.4%、**p = 0.077**——
信号はありそうだが確定しない。model の 39.0%（p = 0.0003）は89レースが学習に使われている
ので**有意として扱ってはいけない**。model と baseline の差（5pp）を見るには約770レース要る。

### プロンプトが効いているかは、学習を通さない検定の方が明確に出る

**この規模では重みの学習を挟まない方が結論が出る。** 学習に89レースを食われると
held-out が11レースしか残らないが、Jev の生の答えを直接検定すれば100レース全部使える。
レース内の並べ替え検定（勝ち馬を無作為に置き換えた分布と比べる）で:

| 対象 | 片側 p | 判定 |
|---|---:|---|
| 第1層（勝ち馬に乗った確率の合計） | **0.0088** | H0 を棄却。**効いている** |
| ↳ 過去成績を持つ馬がいる61レースだけ | **0.0006** | **材料があれば偶然の約2.6倍** |
| ↳ 過去成績が1頭も無い39レースだけ | 0.9975 | **信号ゼロ**。馬名から順位を作ってはいない |
| 第2層 `relative_ability` | **0.0003** | 交絡を除いても棄却。**効いている** |
| 第2層 `current_form` | 0.0336 | Bonferroni（α=0.01）では残らない |
| 第2層 `distance_fit` / `course_going_fit` / `transport_concern` | — | **標本 6 / 2 / 0 件で検定不能** |

第2層は必ず**2段階**で検定すること。素朴に全頭でやると4軸すべてが p<0.001 になるが、
これは交絡（Jev が答えを返すのは過去成績がある馬＝勝ちやすい馬）を拾っているだけ。
**答えが返った馬だけに限った比較**が本題。実装は
`tmp/keiba-report/verify_nullhypothesis.py` と `verify_confound.py`。

### 第2層の答えの 90.8% は「判断材料不足」だった（最優先の課題）

第2層の質問 32,645 問のうち **29,649 問（90.8%）が `insufficient`** で返った。
**記録されたトークンの 66%（$1.31）が「分からない」という答えの代金**になっている。
輸送負担の軸は 6,529 回聞いて実質的な答えは 2 回。

**原因は「データ版の取得期間が短いこと」で、馬マスタ未取得ではない。**
データ版は `fromdate=20260801000000` の7週間ぶん（564レース）しかないので、
予想対象レースより前の出走記録がほとんど存在しない。延べ1,358頭のうち締切前の出走記録が
データ版にある馬は **394頭（29.0%）だけ**。100レース全件で「データ版にある先行出走記録の
数」と「記録上の過去成績ありの数」が**完全に一致**する
（`tmp/keiba-report/verify_rootcause.py` で実測）。

**`horses.csv`（UM）が0行なのは別問題で、より軽い。** UM の列は
`ketto_num, bamei, bamei_kana, birth_date, sex_cd, keiro_cd, hinsyu_cd, del_kubun,
reg_date, del_date` で、**過去成績は1列も入っていない**。過去走は `race_horses.csv`（SE）の
より前のレースの行から引く。UM が効くのは馬名照合と性別・生年の解決だけ。

```bash
# バックテストを回す前にこれをやる。--option 4 ではない
scripts/update_data.py --dataspec RACE --fromdate 20230801000000
```

何年ぶん遡れば足りるかは**未実測**。これをやらずに測った「Jev の価値」は Jev の実力ではない。

### Jev への実支払い

記録に残っている 491 件で **47,012,778 トークン / $1.97**。`--no-save-records` の実行を
含めた推定は **約 76,430,000 トークン / $3.21**（記録ぶんの 1.63 倍）。
**`--no-save-records` でも課金は発生する**（`predict()` が保存判定より前に走る）。
同じ 238 レースに平均 3.34 回投げている。内訳と根拠は `tmp/keiba-report/`。

### 3着以内予想（2026-09-23）

上位n頭（勝率）とは別に、**実際の上位3着に入る馬**を選ぶ系統（案A・案B、`src/keiba/predict/top3*.py`
ほか）を実装し、調整期間・最終評価期間で比較した。設計と結果は
`docs/superpowers/specs/2026-09-23-top3-prediction-design.md` §11〜14（**拘束力のある仕様の続き**）。

- 上の比較結果を出したデータ版は `20260923T115107Z`（約1年ぶん）。`--fromdate` を古くしても
  **option 1 では約1年ぶんしか返らない**。
- **2026-09-24 に5年ぶんのデータ版 `20260924T115403Z` を作った（現在の CURRENT）。**
  `update_data.py --dataspec RACE --fromdate 20210901000000 --option 4`。JRA は
  2021-09〜2026-09-27（2022〜2025年は各3,456レース）。セットアップ取得は
  **スタートキットのダイアログに所有者が「持っていない」→ OK を押す必要がある**。利用キーが
  「試験運用中」の間は -301 が断続的に出ていた。
  - 地方（data_kubun A）・海外（B）のレースも届く（SE の38%）が、`config.DATA_KUBUN_DROP` で
    読み込み時に捨てるので予想には入らない。1989〜2020年の JRA の訂正再配信も約450レース混じる
  - 5年ぶんにすると過去走が揃い、資料の大きさが増える。2026年1〜6月の1,661レースの
    ドライランで **29件（1.7%）が `STATE_MAX_CHARS` を超えた**（最大 52,664 文字）。対処は未決
- 版: `DOSSIER_VERSION` 2 / `TOP3_PROMPT_VERSION` 2 / `TOP3_EVAL_RULES_VERSION` 1。
  `n = 3`（`config.DEFAULT_TOP_N`）。
- **最終評価期間 2026-07-01〜09-22 は使用済み**（境界は `data/top3/final_runs.jsonl` に
  固定。最初の `--final` 実行 2026-09-23T13:10:16Z）。同じ境界より後の日付を指定する
  再評価には `--final-rerun-reason` が要り、理由が同ファイルに残る。
- 最終評価100レースの結果（`mean_hits`、的中数の差の95%区間はレース単位・開催日クラスタの
  ブートストラップ）: plan_a − python +0.08 [−0.02, 0.18]（**差は確認できなかった**）、
  **plan_a − winner_same_state_a +0.14 [+0.04, +0.24]**（区間が0を含まない。馬ごとに
  「3着以内か」を聞く方式が、同じリクエストのwinner確率から選ぶより的中数が多かった）、
  plan_b − plan_a −0.07 [−0.15, 0.00]（**差は確認できなかった**、案Bはplan_aの約1.7倍の
  費用）。案B（展開予測を先に聞く）の1段目の的中率は0.30で偶然（≈0.33）と区別できず、
  **展開予測に検出可能な信号は無い**。plan_aのBrierは一様分布の基準を下回る（較正前の値、
  確率の合計は入着頭数より過大）。多重比較の補正はしていない5比較の1つなので示唆にとどまる。
  詳細・調整期間の数値は spec §13〜14。
- **「判断材料不足が減った」と「予測が正しくなった」は別**。今回測ったのは的中数・Brierなど
  後者だけで、`insufficient` 回答の割合そのものはこの評価の対象にしていない。
- コース事実テーブル（`src/keiba/features/course_facts.json`）は 2026-09-24 に
  **JRA10場・26エントリ**になり、データ版の平地 3,384 レースすべてに該当する。
  JV-Data のコース情報（CS レコード、dataspec `COMM`）は option 1 で `-1`、
  option 4 で `-301` となり取れないため、JRA 公式のコース紹介ページから写している。
  残り: 距離ごとのスタート地点（`starts`）は全エントリ空、数値は A コースのものだけ、
  幅員が範囲表記（「25〜27m」など）の場合は省いている。
- 所有者に残っている項目: 案Bの展開予測ステージは上記のとおり信号なし——使うかどうかの判断が残る。

### 血統データ（2026-09-25）

- `data/blood/versions/<id>/` に**レースとは別の系統**で置く（`CURRENT` も別）。
  `update_data.py --kind blood --option 4 --fromdate 19860101000000`（ダイアログ操作が要る）。
  `pedigrees.csv`（SK。3代血統14頭を `sire_num` `dam_sire_num` … の列で）、
  `breeding_horses.csv`（HN）、`lineages.csv`（BT。系統説明の自由文は捨てる）
- **dataspec は `BLDN`。`BLOD` を使わないこと。** BLOD は JV-Data 4.9.0（2023-08）より前の
  形式（繁殖登録番号8桁）のまま 2023-07 作成分で止まっており、仕様書どおりに読むと位置が全部ずれる。
  パーサはレコード長で旧形式を弾く。race 側既定の `DIFF` も同じ問題を抱えている可能性がある（未確認）
- 現在の版 `20260925T043102Z`: SK 427,630 / HN 162,061 / BT 92。2021〜2025年の JRA 出走馬
  27,574頭の **99.92%** で父・母父の名前が引ける。BT は系統の起点の馬にしか付かない（父に直接
  付くのは3.35%）ので、系統は父系をさかのぼって最初に BT がある馬で決める
- 中山・芝中距離の記述統計は `tmp/keiba-report/stats1/report.md`（血統なし。市場の期待との差で
  ほぼ全変数の差が消え、遠征だけが全場で −0.9pp ±0.6 残った）

### 払戻・オッズ（2026-09-26）

- `data/market/versions/<id>/` に**レース・血統とは別の系統**で置く（`CURRENT` も別）。
  `update_data.py --kind market`（dataspec は RACE。HR と O1〜O6 だけを保存）
- ファイル: `payout_races.csv` / `payouts.csv`（HR。組番ごと、100円あたり）、`odds_pools.csv`（賭式ごとの
  発売状況・票数合計）、`odds_tansyo.csv` / `odds_fukusyo.csv` / `odds_wakuren.csv`、
  `odds_umaren.csv.gz` / `odds_wide.csv.gz` / `odds_umatan.csv.gz` / `odds_sanrenpuku.csv.gz` /
  `odds_sanrentan.csv.gz`（大きい表は gzip）。組番は馬番2桁を `-` でつなぐ。馬単・3連単は着順どおり、他は昇順
- **データ区分の意味が RA/SE と違う。** HR は 2（月曜）> 1、O は 5（確定・月曜）> 4（確定）。O の 1〜3 は
  確定前（`config.DATA_KUBUN_PAYOUT_PRIORITY` / `DATA_KUBUN_ODDS_PRIORITY`）。セットアップ取得では HR は全件 2、
  O は 5（中止の 9 が33件）だった
- **確定オッズは締切後にしか分からない。** 期待値で買い目を選ぶ検証に使うと、買う時点で知りえない値で選ぶことになる
- 2026-09-26 に生レコード `data/raw/20260925T172902Z/`（RACE, fromdate 20210901, option 4。`keiba.ingest.rawdump`
  の形式）から `--from-raw` で作った。生レコードで照合済み: 全レコード長が仕様どおり・パースエラー0、O1 の単勝オッズと
  SE の確定単勝オッズが 239,960 頭で全一致、3連単の払戻の組番と SE の1-2-3着が同着なしの 17,416 レースで全一致、
  単勝の払戻とオッズ×100 のずれは1着同着の46件だけ
- `--from-raw DIR` は JV-Link を呼ばずに生レコードから版を作る。ダイアログ操作が要るセットアップ取得を、
  パーサの実装より先に済ませたいときに使う

### 世代レベル（2026-10-04、第1段階）

- 走破タイムから全年齢共通のタイム指数を作り、世代（生まれ年）の強さを測る単体の指標。**予想には組み込んでいない**
  （第2段階で別の仕様）。仕様 `docs/superpowers/specs/2026-10-04-generation-level-design.md`
- `scripts/generation_level.py --as-of YYYYMMDD [--data-dir DIR] [--example-race RACE_ID]`。Jev は呼ばない。
  出力は `data/generation/<as_of>/`
- 基準タイム・馬場差は**古馬戦の1着だけ**から推定する（2歳戦を混ぜると世代の弱さが馬場差に吸収される）
- 2026-10-04 時点の結果（比べた年 2022〜2026、過去の世代は4つだけなので示唆にとどまる）: 今年の3歳の例年比は上位10頭
  +0.40 [+0.08, +0.69]、上位50頭 +0.19 [+0.02, +0.37]、中央値 +0.09 [+0.03, +0.16] で、過去より強い側に出た（区間は0を含まない）。
  見立て（4歳>5歳>3歳）は「例年の差と比べた差」で判定し、3組×3指標の9通りすべて差があるとは言えない（そのままの差
  4歳−3歳 中央値 +1.25 は例年の +1.26 とほぼ同じ＝毎年出る生き残り・斤量の偏り）。年齢曲線（全体）は 4歳前半〜5歳前半でほぼ横ばい、
  最大は 4歳前半 +0.09。区間は物差しを日で推定し直し馬を世代ごとに抽出する手順（点推定は60区間すべての内側）

### 要素別の予想 — 過去での評価（2026-10-11）

- 15要素それぞれを市場（単勝オッズ）の着順モデルに足したときの改善を 2017〜2025年で測った。仕様
  `docs/superpowers/specs/2026-10-11-factor-predictions-design.md` §3〜§5、実装 `src/keiba/factors/`・`src/keiba/evaluate/factor_eval.py`
- `scripts/factor_eval.py --data-dir data/_long`（Jev は呼ばない）。要素の点数は
  `data/factor_eval/factors_<データ版>_<血統版>_<要素の版の印>.pkl` に保存して再利用（`--cache-dir`、`--rebuild`）。
  `--recompute-table-from <古い pkl>` で表の要素7つだけ作り直せる（年ごとの要素の30分以上を省く）
- **現在の結果は v2 の実行 `data/factor_eval/20261011T143147/`**（データ版 `20261010T155433Z`、評価に使ったレース 29,789、
  除いた 175 は全部「1〜3着が一意でない」）。v1 の実行 `20261011T140120/` は記録として残す。最初の実行
  `20261011T134627/` は年齢・父の芝ダ適性の Δ が壊れていたので使わない
- **v2 にした理由**: 表の要素（騎手・調教師・父の芝ダ・枠順・間隔・馬体重・遠征）の縮小の強さ λ を、v1 は組の中の散らばりを
  1つにまとめたモーメント法で推定していた。A−E の散らばりは組ごとに大きく違うので τ² が小さく出て λ が大きすぎた
  （騎手 2017年 4,760 → v2 1,358、調教師 13,430 → 2,744。2016年の騎手・調教師・父の芝ダは推定できず点数が全部 0）。
  v2 は組ごとの散らばりを引く推定（`cumulative.eb_variance_components`）。ほかの8要素は v1 のままで点数も同じ。
  **v2 は v1 の結果を見た後の変更なので、判定は事後のもの**。これからのレースの記録（仕様 §6）で確かめる
- v2 の結果（Δ ×1000 [95%]、判定は区間の下限 > 0 かつ 9年中7年以上で改善）: 上乗せあり 9要素 —— タイム指数 +6.57 [+5.00, +8.24]、
  相手関係 +5.04 [+3.93, +6.19]、間隔 +2.66 [+1.81, +3.55]、遠征 +0.98 [+0.33, +1.63]、馬体重 +0.95 [+0.40, +1.52]（測れた8年すべて改善）、
  距離適性 +0.87 [+0.31, +1.39]、斤量 +0.74 [+0.31, +1.16]、枠順 +0.63 [+0.02, +1.27]、コース適性 +0.52 [+0.14, +0.92]。
  確認できない 6要素: 父の芝ダ適性 +1.29 [+0.28, +2.33]（区間は0を含まないが改善は 6/9年）・世代レベル +0.15・調教師 −0.07・
  騎手 −0.17・晩成度×年齢 −0.18・年齢 −0.34。**判定は v1 と15要素すべて同じ**（父の芝ダは +0.06 → +1.29、改善 4 → 6年）
- 大きさ: 市場だけのモデルの −対数尤度は1レースあたり 6.0525。最大のタイム指数でも Δ はその 0.108%。15比較の Bonferroni
  （99.67%）の参考区間では枠順・コース適性・父の芝ダの下限が 0 を下回る（判定には使っていない）。
  「上乗せあり」は市場モデルに情報を足せる（検出できる）という意味で、的中が増える・市場に勝てるという意味ではない
- 的中は市場の上位3頭が 1.579 頭/レースで、市場＋要素でも最大 1.581（タイム指数・相手関係）と差はごく小さい（的中には区間なし）
- 測れなかった年（学習か評価の年の z が全部 0。Δ = 0 として平均に入るが改善には数えない。レポートでは「—」）:
  晩成度×年齢 2017・2018年、年齢 2017年、馬体重 2017年（v2 でも 2016年は λ が推定できず点数が全部 0）
- 要素の z は中心化した値の |c| < 1e−9 を 0 とみなし、±4 で切っている（仕様 §4.2 からの逸脱。`factor_eval.zscore`）。
  v2 でも切られる評価の行が多い年が残る: **父の芝ダ 2017年 49.7%、調教師 2017年 23.0%**。原因は、1頭だけの組（n_k < 2）の
  ばらつきを全体の値で埋めている（`cumulative.eb_variance_components` の `fillna(pooled)`）ため 2016年の λ が大きくなること
  （父の芝ダ 8,379。0 で埋めると 461）。**父の芝ダの判定はこの埋め方で入れ替わる**（全体で埋める: +1.29、6/9、確認できない ／
  0 で埋める: +1.89、7/9、上乗せあり）。結果を見た後に選ぶと後付けになるので、今の判定のまま、これからのレースで確かめる。
  v1 で多かった父の芝ダ 2019年（66%）は 2.7%、調教師 2018年（36%）は 5.1% に下がった。
  年 × 要素の割合は `by_year.csv` の `clipped_share`
- 15要素を同時に調べているので偶然の「上乗せあり」が混じりうる（多重比較の補正はしていない）

## 何を作るのか

TypeSafe AI の **Jev** を使った競馬予想システム。**実装言語は Python。**

パイプライン:

```
データ取得 → 数値化（特徴量化）→ state 組み立て → Jev へリクエスト → 統合 → 上位 n 頭
```

最終出力は **上位 n 頭**。次に行われるレースを対象とする。

## Jev とは（ここを外すと設計を間違える）

Jev は TypeSafe AI の **System One モデル**。**文章を生成しない。** 「型付きの答え + 較正済み確率」だけを返す。

- 入力: `state`（判断材料。文字列または dict）と `questions`（質問の辞書）
- 出力: 質問ごとに、あらかじめ型で定義した値と確率
- 返せる答えは渡した選択肢の中だけ。ハルシネーションも型エラーも原理的に起きない
- 1リクエスト内の全質問を **並列評価** する。質問を増やしてもレイテンシはほぼ変わらず、増える費用は質問文のトークン分だけ

### 質問プリミティブ

| 種類 | 用途 | 返り値 |
|---|---|---|
| `Noul` | Yes/No。**確率そのものが欲しいとき**。軸ごとの評価はこれ | `noul`（0–1） |
| `Choice` | 順序のない選択肢から1つ選ぶ。選択肢は最大 255 | `choice`, `probabilities`, `confidence` |
| `Score` | 順序のある段階で評価する（0–4 など） | `score`, `legend`, `probabilities`, `confidence` |

`Score` の `score` は選んだ段階そのものではなく **各段階の確率で重み付けした平均**。

`Choice` の `probabilities` は **選択肢名をキーにした合計 1.0 の辞書**。`confidence` は確率分布の集中度（1つに集中すれば 1.0、散らばるほど低い）。

```python
Choice(
    instructions="Which team should handle this?",
    criteria={               # 選択肢名 -> その説明
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
    },
)
```

**`criteria` の型は質問の種類ごとに違う。ここを間違えると通らない。**

| 種類 | `criteria` の型 | 必須か |
|---|---|---|
| `Noul` | `NoulCriteria(true=..., false=...)` または省略 | 任意 |
| `Choice` | **dict**（選択肢名 -> 説明。説明不要なら値に `None`） | 必須 |
| `Score` | **順序付きのリスト**（0 から順に1段階ずつ説明を並べる） | 必須 |

`Score` は dict ではない。0–4 の5段階にしたいなら **要素5つのリスト**を渡す。

```python
Score(
    instructions="How frustrated the customer appears",
    criteria=[               # [0], [1], [2] ... の順
        "Calm, just stating facts",
        "Frustrated but civil",
        "Very angry, strong language",
    ],
)
```

### 呼び出し方

```bash
pip install typesafe-sdk
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()  # 環境変数 TYPESAFE_API_KEY を自動で読む
response = client.system_one(
    state={"race": ...},
    questions={"urgency": Noul(instructions="Is this urgent?")},
)
```

答えは `response.answers[質問名]` のほか、**種類ごとの辞書**から取り出せる:

```python
response.nouls["urgency"].noul            # float 0-1
response.choices["winner"].choice         # 選ばれた選択肢名
response.choices["winner"].probabilities  # {選択肢名: 確率}、合計 1.0
response.choices["winner"].confidence
response.scores["level"].score            # 確率で重み付けした平均
response.usage.input_tokens               # None のことがある
```

`state` は文字列でも dict でもよい（dict はそのまま JSON として送られる）。
`TypeSafeClient` は `with` 文で使える。

- エンドポイント: `POST https://api.typesafe.ai/v1/systemone`
- 認証: 環境変数 `TYPESAFE_API_KEY`（キーは https://console.typesafe.ai/ のダッシュボードで取得）
- モデル: `jev-latest` → `jev-1.13.0`

### 制約

- コンテキスト: 1リクエスト合計 **64k トークン**、うち `state` + 最長の質問で **32k トークン**
- 入力は **テキストのみ**（画像・音声・動画は不可）
- レート: 1,200 req/min、250,000 tokens/sec（予告なく変動しうる）
- 料金: 入力 $0.042 / 1M トークン、**出力は無料**
- **SDK の既定タイムアウトは 10 秒**。第2層で全頭を並列に投げると足りないことがある。
  `src/jev_client.py` では 30 秒にしている
- `RetryPolicy.timeout` は「再試行を含めた総予算」（既定 30 秒）。1回分の timeout より
  短いと最初の試行の途中で打ち切られるので、両方を同時に調整すること

### エラー

SDK はエラーを例外で返す。すべて `TypeSafeError` の派生。

| 例外 | 意味 | 持っている情報 |
|---|---|---|
| `TypeSafeAuthenticationError` | 401。キーが誤り/失効 | `status`, `body` |
| `TypeSafePermissionDeniedError` | 403 | 同上 |
| `TypeSafeRateLimitError` | 429 | `retry_after_ms` |
| `TypeSafeBadRequestError` | 400。state/questions の作り方 | 同上 |
| `TypeSafeUnprocessableEntityError` | 422 | 同上 |
| `TypeSafeInternalServerError` | 5xx | 同上 |
| `TypeSafeAPITimeoutError` | タイムアウト | `timeout` |
| `TypeSafeAPIConnectionError` | 接続不能 | — |
| `TypeSafeAPIResponseValidationError` | 応答が想定の形式でない | `field_path` |

**`TypeSafeAPITimeoutError` は `TypeSafeAPIConnectionError` の派生。**
`except` や分岐の順番を間違えるとタイムアウトを「接続できない」と誤報する。
日本語化は `src/jev_client.py` の `describe_error()` にまとめてある。

## データ取得（JRA-VAN Data Lab）

一次データソースは **JRA-VAN Data Lab**。スクレイピングは JRA-VAN で埋まらない項目が出た場合の **最終手段**であり、既定の経路ではない。

- 利用には データラボ会員登録が必要（月額 2,090円 税込）。初月のみ仮キーで試用できる
- **JV-Link** は COM（ActiveX）コンポーネント。**Windows 専用**。Python からは `pywin32` 経由で叩く

  ```python
  import win32com.client
  jvlink = win32com.client.Dispatch("JVDTLab.JVLink")
  ```

### 現状: 取得できる状態になっている（2026-09-21 実機検証済み）

**64bit Python から JRA-VAN のデータを取得できることを確認済み。** 32bit 回避策は一切不要。

- インストール済み: **JV-Link 5.0.0（64bit）** — `C:\Windows\System32\JVDTLAB\JVDTLab.dll`
  （`C:\Program Files\JRA-VAN\Data Lab`）
- 32bit 版（`C:\Windows\SysWOW64\JVDTLAB\JVDTLab.dll`）も TARGET frontier JV 用に併存。両方動く
- 利用キーは 64bit 側でも有効（実ダウンロードまで確認済み）

検証内容（`JVOpen("RACE", "20260801000000", 1, ...)`）:

| 項目 | 結果 |
|---|---|
| `JVInit` / `JVOpen` | ともに `0` |
| ダウンロード | 151ファイル / 約4秒 |
| 読み出し | 19,439レコード |
| `RA` レース詳細 | 552 — **第1層の state 用** |
| `SE` 馬毎レース情報 | 7,294 — **第2層の state 用** |
| `HR` / `O1`–`O6` / `H1` / `H6` | 各480 |
| `JG` / `WF` | 7,259 / 14（種別の内容は未確認） |

### JV-Link の呼び出し方（SDK の sample1_Python3.14 準拠）

pywin32 は ByRef 出力引数を戻り値のタプルで返す。プレースホルダを渡して受け取る。

```python
import win32com.client

jvlink = win32com.client.Dispatch("JVDTLab.JVLink")
jvlink.JVInit("UNKNOWN")

# out 引数には 0, 0, "" をプレースホルダとして渡し、戻りのタプルで受ける
open_code, read_count, download_count, lastfiletime = jvlink.JVOpen(
    dataspec, fromdate, option, 0, 0, ""
)
#   0 : 正常 /  -1 : 該当データなし / 他 : エラー

while jvlink.JVStatus() < download_count:   # ダウンロード進捗
    ...

ret, text, _size, fname = jvlink.JVRead("", 110000, "")
#  >0  : text にレコード1件（ret は cp932 のバイト長。_size は渡した値のまま）
#  -1  : ファイル切替
#   0  : EOF
#  <-1 : エラー
raw = text.encode("cp932")   # len(raw) == ret を確かめる

jvlink.JVClose()
```

- **`JVGets` は使わない（2026-09-24 実測）。** 公式サンプルは `JVGets` だが、pywin32 経由では
  呼ぶたびに約110KB（バッファ長ぶん）漏れ、`JVClose` しても戻らない。5年分のセットアップ取得
  （数十万件）では黙ってメモリを使い切って落ちる。`JVRead` なら漏れない
- **`JVRead` は読んだ時点のコンソール出力コードページで文字列にする。** 65001（UTF-8）だと
  日本語が欧文として化ける。`jvlink.read_open_dataspec()` が読む間だけ 932 に切り替えている。
  cp932 に戻したバイト列は `JVGets` の結果（末尾の NUL を除く）と全件一致することを確認済み
- **レコードは cp932。** 固定長の位置はバイト単位なので、デコードは `records.py` 側で行う
- **option 3/4（セットアップ）では「スタートキット（CD/DVD-ROM）」のダイアログが出る。**
  option 4 でも出る。**「持っていない」を選んで OK** を所有者に押してもらう必要がある
  （「持っている」のままだと -501、放置・異常終了の直後は -301 がしばらく続く）
- レコード種別は先頭2バイト（`RA` / `SE` / `HR` / `O1` …）
- `JVOpen` の `option` は型ライブラリ上 **`VT_UINT`**。このため **VBScript からは呼べない**
  （「型が一致しません」で落ちる）。Python なら問題ない

### SDK の中身（`vendor/jravan-sdk-500-64bit/`）

`https://jra-van.jp/dlb/sdv/sdk/JVDTLABSDK500_64bit.zip` からログイン不要で取得できた（27MB）。

| 用途 | パス（`vendor/jravan-sdk-500-64bit/` 以下） |
|---|---|
| JV-Data 構造体（Python） | `JV-Data構造体/Python版/JVData_Struct.py` |
| 公式 Python サンプル | `サンプルプログラム/sample1_Python3.14/` |
| JV-Data 仕様書 | `ドキュメント/JV-Data仕様書_4.9.0.1.pdf` |
| JV-Link IF 仕様書 | `ドキュメント/JV-Linkインターフェース仕様書_4.9.0.1(Win).pdf` |
| 64bit インストーラ | `JV-Link/JV-Link.exe` |

### 参考

- SDK 配布とドキュメント: https://developer.jra-van.jp/t/topic/45
- 開発者コミュニティ（エラーコードの実例が集まっている）: https://developer.jra-van.jp/
- Data Lab. 概要: https://jra-van.jp/dlb/

### スクレイピングを使う場合（最終手段）

- 対象サイトの利用規約を必ず先に確認する。競馬情報サイトは自動取得を禁じているところが多い
- JRA-VAN から取得したデータは再配布不可

## 予想アーキテクチャ

### リクエストを分ける基準は「軸」ではなく「state の粒度」

Jev は1リクエスト内の全質問を並列評価するので、**同じ state に対する質問はまとめたほうが安くて速い**。
公式の検証では 13問を1リクエストにまとめて **12.2倍安く・10倍速く**、しかも答えは個別に送った場合と同一だった。state を送り直す分が丸ごと無駄になるため。

したがってリクエストは「距離の質問」「会場の質問」のように **軸で割らない**。**state が変わるところで割る。**

### 第1層: レース単位（1レースにつき1リクエスト）

`state` = 出走全頭の数値サマリ

```python
questions={
    "winner": Choice(
        instructions="このレースで最も勝ちやすいのはどの馬か",
        criteria={"1": "<1番の特徴>", "2": "<2番の特徴>", ...},  # 馬番 -> 特徴
    )
}
```

`probabilities` が合計 1.0 の辞書で返るため、**この1回で全頭の勝率分布が手に入る**。上位 n 頭という目的に最短で効く。選択肢は最大 255 なのでフルゲート18頭でも余裕。

### 第2層: 馬単位（出走頭数分。並列実行する）

`state` = その馬1頭分の詳細な数値

```python
questions={
    "distance_fit":  Noul(instructions="この距離で好走できるか"),
    "venue_travel":  Noul(instructions="所属トレセンから開催会場への輸送負担が不利に働くか"),
    # 軸を足すときはここに1行足す
}
```

**軸は全部ここにまとめる。軸ごとにリクエストを分けない。**
馬ごとには別リクエストになるので、`ThreadPoolExecutor` などで並列に投げる（公式の re-ranking クックブックも同じやり方）。

#### 軸「輸送負担」の扱い（決定済み）

見たいのは **輸送が長いと馬に負担がかかる**という効果。したがって起点は牧場ではなく、**実際に輸送が発生する所属トレセン（美浦 / 栗東）** とする。

実装時に検討しておくこと:

- **単純な距離だけでは負担を表しきれない。** 北海道開催（札幌・函館）などは現地に事前滞在するのが一般的で、
  当日輸送とは負担の質が違う。「距離」と「当日輸送か滞在か」を分けて `state` に入れるか、質問文で区別する
- **負担の結果は馬体重の増減に現れやすい。** JRA-VAN に馬体重があるので、過去の長距離輸送時に体重を大きく
  減らした馬かどうかを `state` に入れる価値がある

### 第3層: 統合（Jev は使わない。ただの Python）

1. 各軸を 0–1 に正規化する（`Noul` はそのまま 0–1。`Score` を 0–4 で使ったならコード側で 4 で割る）
2. 重み付き和で馬ごとの合計スコアを出す
3. 第1層の勝率と合成する
4. 降順に並べて **上位 n 頭**

```python
total = (w_race * race_prob[horse]) + (w_dist * distance_fit) + (w_venue * venue_travel)
```

重みは Jev ではなく **自分のコードで持つ**。最終スコアの内訳が常に見えるので、予想が外れたときモデルの作り直しではなく重みの調整で対応できる。
**軸の追加が「第2層に質問を1行、重みを1つ」で済む構造を保つこと。**

## ビルド / テスト / 実行

venv は `.venv`（Python 3.13.3 / 64bit）。コマンドは `.venv` の python を明示して使う。

```bash
# 依存（初回のみ）
.venv/Scripts/python.exe -m pip install -r requirements.txt

# テスト全体（2026-10-11 実行: 1493 passed）
.venv/Scripts/python.exe -m pytest

# テスト1件だけ
.venv/Scripts/python.exe -m pytest tests/test_jev_client.py::<テスト関数名>

# Jev への疎通確認（.env に TYPESAFE_API_KEY が必要）
.venv/Scripts/python.exe scripts/smoke_test.py

# 3着内予想: 1レース分の資料を書き出す（課金なし。API を呼ばない）
.venv/Scripts/python.exe scripts/dossier.py --race-id <race_id>

# 3着内予想: state の大きさを測るだけ（課金なし）
.venv/Scripts/python.exe scripts/backtest_top3.py --from <YYYYMMDD> --to <YYYYMMDD> \
    --tuning-until <YYYYMMDD> --dry-run

# 3着内予想: 課金あり（所有者の承認を得てから）。--methods は明示する
.venv/Scripts/python.exe scripts/backtest_top3.py --from <YYYYMMDD> --to <YYYYMMDD> \
    --tuning-until <YYYYMMDD> --methods python,plan_a,plan_b --limit 100
```

`--methods` を省略すると既定は `python`（Jev を使わない基準）のみで、**課金は発生しない**。
案A・案Bを走らせるときは `--methods` に `plan_a` / `plan_b` を明示すること。

PowerShell ではパス区切りが `.venv\Scripts\python.exe`。

`pytest.ini` が `pythonpath = src` / `testpaths = tests` を設定しているので、`src/` は
インストールせずに import できる。

`requirements.txt` には JRA-VAN 用の `pywin32`（`sys_platform == "win32"` マーカー付き）も含まれる。
`.venv` の python から JV-Link を `Dispatch` してデータを読めることは 2026-09-21 に確認済み。

## 参考リンク

- ドキュメント: https://docs.typesafe.ai/ （全ページ索引: https://docs.typesafe.ai/llms.txt）
- Composite scoring: https://docs.typesafe.ai/patterns/composite-scoring.md
- Parallel questions: https://docs.typesafe.ai/cookbooks/parallel_questions.md
- Re-ranking: https://docs.typesafe.ai/cookbooks/rerank_typesafe.md
- Confidence: https://docs.typesafe.ai/confidence.md
- モデルの苦手分野: https://docs.typesafe.ai/model-jaggedness/jev-1.13.md （未読。実装前に目を通すこと）
- Claude Code 用の agent skill: `claude plugin marketplace add typesafe-ai/skills` → `claude plugin install typesafe@typesafe-ai`

> 上記の Jev 仕様は 2026-09-21 に `typesafe-sdk` 0.7.0 の実物と照合済み。
> 照合で判明した当初の誤りは本ファイルに反映してある（`Score` の `criteria` はリスト、
> 答えは `response.nouls` / `response.choices` / `response.scores` から取り出す、
> 例外は `TypeSafeError` 系）。**実 API への接続も 2026-09-21 に確認済み**
> （`scripts/smoke_test.py` が `jev-1.13.0` から応答を得て exit 0）。

## 決まったこと（当初の未確定事項）

- ~~**数値化の方法**~~ — dict の `state`。形式は仕様 §7
- ~~**軸の一覧**~~ — 5軸（距離適性 / コース・馬場適性 / 相対能力 / 現在の状態 / 輸送負担）。仕様 §8.3
- ~~**重みの決め方**~~ — 条件付きロジットの最尤推定（`scripts/fit_weights.py`）。過去レースで最適化する
- ~~**n の値**~~ — 既定3（`config.DEFAULT_TOP_N`）
- ~~**`TYPESAFE_API_KEY`**~~ — 取得済み。`.env` に置く（`.env.example` 参照）
- ~~**第1層と第2層のスケールの揃え方**~~ — **レース内 softmax（条件付きロジット）で解決した**。
  分母がそのレースの出走馬だけなので、8頭でも18頭でも釣り合いが変わらない。仕様 §10

## 未解決のまま残っていること

- **第1層の確率が 0.01 刻みで返る。** 13〜18頭立てでは大半の馬が 0.00 になり、
  `logit_layer1` は clip 後に同値になるのでそれらの馬を区別しない。較正で確率の質は
  救えるが、第1層の分解能そのものは上がらない
- **順位の技量が偶然と区別できない。** 100レースでは検出力が足りない（上位3頭の的中率の
  標準誤差が約4.7pp）。判断するなら数千レース規模が要る
- **`config.TIE_BAND` は第1層の尺度でしか測っていない。** model 尺度で同程度を判定するには
  その尺度で測り直すこと（現状は重みがあるとき同程度の判定を行わない）
- ~~**較正の適用が未配線**~~ — 2026-09-23 に配線した。較正マップを持つ重みで model の
  順位を出す実行では `calibrated_win_prob` が埋まり、`data_quality.calibration` に
  再正規化前の合計・1.0 からのずれ・定義域・端に張り付いた馬番が残る（仕様 §11.2）。
  **順位には使わない。利用者にも見せない**（仕様 §10.3 を改訂）
- **出荷する重み `20260922T135009Z` の性能が未測定。** 上の性能表は退避済みの
  `20260922T130614Z`（`prompt_version=1`、現行コードの `check_compatible()` が拒否する）で
  測ったもの。出荷重みでバックテストを走らせた実行が1本も無い
- **`recent_margin_runs_used_missing` は全 6,529 行で 0 の定数列**なのに
  `dropped_features` が空。「落ちた列 0」を美点として読むと誤る
- ~~**Jev に送った state と質問文が記録に無い**~~ — 2026-09-23 に配線した。
  `data/prompts/<YYYYMM>/<record_id>.json` に**送った物そのもの**（再生成ではない）を
  保存する。`predict.py` / `backtest.py` の `--save-prompts`（既定 ON）。
  18頭のレースで 244KB、100レースの実行で 15〜25MB 程度
- ~~**`EntryRow.changed_after_entry` が `"00000"` を「変更あり」と読む**~~ — 2026-09-23 に
  修正した。`models.py`（既存の版を救う）と `ingest/records.py`（将来の版を救う）の
  両方を直し、docstring と仕様 §4.1.2 の虚偽の不変条件も訂正した。
  実測で注記は **1,358 → 12 頭**（偽 1,346 = 99.1% が消えた）。送る文字数は
  **577,850 文字（4.09%）減**——偽の注記は第2層の state にも入るので、
  1レースあたり (1 + 頭数) 回ぶん送られていた
- **`stay_status` を渡す本番コードが存在しない。** `derive.transport()` の既定値
  `"unknown"` が全頭に入る（渡しているのは `tests/test_derive.py` だけ）。
  `transport_concern` が 99.97% `insufficient` な要因の1つ。ただし P(concern) は
  過去の輸送実績と減量歴に単調に反応しているので、これ単独が律速ではない

# house_racing

Jev（TypeSafe AI の System One モデル）を使った競馬予想システム。
JRA-VAN Data Lab からのデータ取得、数値化、Jev への質問（勝率の上位 n 頭と、3着以内に入る馬の選択）、
重みの学習と較正、バックテストまで端から端まで動く。ほかに Jev を使わない分析として、世代レベルと
要素別の予想の過去での評価がある。

設計方針と現況（実測した性能を含む）は [CLAUDE.md](CLAUDE.md) と `docs/superpowers/specs/` を参照。

## 動かす

```bash
# 1. 依存を入れる（初回のみ）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. API キーを設定する
cp .env.example .env
#    .env を開いて TYPESAFE_API_KEY= の右辺を実際のキーに書き換える
#    キーの取得: https://console.typesafe.ai/keys

# 3. 疎通確認
.venv/Scripts/python.exe scripts/smoke_test.py
```

PowerShell の場合、手順1・3の `.venv/Scripts/python.exe` は `.venv\Scripts\python.exe`。

`smoke_test.py` はダミーの3頭立てレースで第1層→第2層→第3層を一通り実行し、
勝率分布・軸ごとの評価・合成スコア・トークン使用量を表示する。
キーが未設定なら、どこに何を書けばよいかを表示して終了する。

## テスト

```bash
.venv/Scripts/python.exe -m pytest
```

Jev・JV-Link の実通信はしない。Jev の実通信の確認は `smoke_test.py` の役目（JV-Link には接続しない）。

## 構成

| パス | 役割 |
|---|---|
| `src/jev_client.py` | Jev への薄いアクセス層。キー解決・例外の日本語化・並列実行だけを持つ |
| `src/keiba/` | データ取得・読み出し・数値化・予想・学習・評価・記録（詳細は CLAUDE.md の表） |
| `scripts/` | `update_data` / `predict` / `fit_weights` / `backtest` / `dossier` / `backtest_top3` / `data_coverage` / `generation_level` / `factor_eval` / `smoke_test` |
| `scripts/smoke_test.py` | 疎通確認。CLAUDE.md の3層構成をダミーデータで実演する |
| `tests/` | 実通信をしないテスト |
| `data/` | 取得したデータ版・予想記録・重み。JRA-VAN のデータは再配布不可のため**コミットしない** |
| `.env` | API キー。**コミットしない**（`.gitignore` 済み） |

`jev_client.py` に予想ロジックと重みは置かない。あそこは通信の都合だけを閉じ込める場所。

## 軸を足すとき

本番の質問は勝率予想が `src/keiba/predict/questions.py`（変えたら `PROMPT_VERSION` を上げる）、3着内予想が `src/keiba/predict/top3_questions.py`（変えたら `TOP3_PROMPT_VERSION` を上げる）。以下は疎通確認の実演の場合。

`scripts/smoke_test.py` の `AXIS_QUESTIONS` に質問を1行、`WEIGHTS` に重みを1つ足す。
「値が大きいほど不利」を問う軸は `INVERTED_AXES` にも名前を入れる（第3層で `1 - x` に反転される）。
第2層のリクエストは馬ごとに1本のまま変わらない — 軸ごとにリクエストを分けないこと。

## 次にやること

未解決の項目は CLAUDE.md の「未解決のまま残っていること」を参照。

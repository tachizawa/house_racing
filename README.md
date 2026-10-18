# house_racing

Jev（TypeSafe AI の System One モデル）を使った競馬予想システム。
現在入っているのは **Jev へのアクセス層と疎通確認** まで。データ取得（JRA-VAN）と
予想ロジック本体はこれから。

設計方針は [CLAUDE.md](CLAUDE.md) を参照。

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

API に触れない部分（キーの解決、例外の日本語化、並列実行）だけを検証する。
実通信の確認は `smoke_test.py` の役目。

## 構成

| パス | 役割 |
|---|---|
| `src/jev_client.py` | Jev への薄いアクセス層。キー解決・例外の日本語化・並列実行だけを持つ |
| `scripts/smoke_test.py` | 疎通確認。CLAUDE.md の3層構成をダミーデータで実演する |
| `tests/` | `jev_client` のうち API に触れない部分のテスト |
| `.env` | API キー。**コミットしない**（`.gitignore` 済み） |

`jev_client.py` に予想ロジックと重みは置かない。あそこは通信の都合だけを閉じ込める場所。

## 機械学習系（血統モデル）

CatBoost の血統モデルは `src/keiba_ml/` に既存モデルと分けて置いてある。
手順と設計は [docs/ml/README.md](docs/ml/README.md)。

## 軸を足すとき

`scripts/smoke_test.py` の `AXIS_QUESTIONS` に質問を1行、`WEIGHTS` に重みを1つ足す。
「値が大きいほど不利」を問う軸は `INVERTED_AXES` にも名前を入れる（第3層で `1 - x` に反転される）。
第2層のリクエストは馬ごとに1本のまま変わらない — 軸ごとにリクエストを分けないこと。

## 次にやること

- JRA-VAN Data Lab からのデータ取得（`build_race_state` / `build_horse_state` の中身を差し替える）
- 軸の確定と重みの決め方
- 第1層の確率（全頭で合計 1.0）と Noul（1頭ごとに 0–1）のスケールの揃え方

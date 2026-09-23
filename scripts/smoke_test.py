"""Jev への疎通確認。CLAUDE.md の3層構成をダミーレースで一通り動かす。

    python scripts/smoke_test.py

第1層 (レース単位1リクエスト) → 第2層 (馬単位を並列) → 第3層 (Python だけで合成)
の順に実行し、確率・confidence・トークン使用量を表示する。

ここで使うのはダミーデータ。JRA-VAN からの取得が入ったら、
build_race_state / build_horse_state の中身を差し替える。
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _enable_utf8_output() -> None:
    """Windows コンソールで日本語が化けないようにする。

    Python は既定でコンソールの文字コード (日本語 Windows なら cp932) に合わせて
    出力するが、Git Bash や Windows Terminal は UTF-8 を期待するため化ける。
    コンソール側のコードページを UTF-8 に切り替えたうえで stdout も UTF-8 に固定する。
    パイプ経由のときは前者が無視されるだけで害はない。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # コンソールが無い環境ではそもそも不要
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_enable_utf8_output()

from typesafe_sdk import Choice, Noul  # noqa: E402

from jev_client import (  # noqa: E402
    Job,
    JevConfigError,
    JevRequestError,
    ask,
    ask_parallel,
    create_client,
)

TOP_N = 2

# ---------------------------------------------------------------------------
# 軸の定義。軸を足すときはここに1行、WEIGHTS に1行。それだけで済む形を保つこと。
# ---------------------------------------------------------------------------

AXIS_QUESTIONS = {
    "distance_fit": Noul(
        instructions="この馬は本レースの距離で好走できるか",
    ),
    "venue_travel": Noul(
        instructions=(
            "所属トレセンから開催会場への輸送負担が、この馬にとって不利に働くか。"
            "当日輸送か現地滞在かの違いと、過去の長距離輸送時の馬体重の増減を考慮すること"
        ),
    ),
}

#: 値が大きいほど「不利」を意味する軸。第3層で 1 - x に反転して足す。
INVERTED_AXES = {"venue_travel"}

WEIGHTS = {
    "race_win": 0.5,
    "distance_fit": 0.3,
    "venue_travel": 0.2,
}


# ---------------------------------------------------------------------------
# ダミーデータ
# ---------------------------------------------------------------------------

RACE = {
    "会場": "中山",
    "コース": "芝",
    "距離m": 2000,
    "馬場状態": "良",
    "クラス": "3勝クラス",
}

HORSES = {
    "1": {
        "馬名": "テストランナー",
        "性齢": "牡4",
        "斤量kg": 57.0,
        "所属": "美浦",
        "輸送区分": "当日輸送",
        "トレセンから会場までのkm": 40,
        "近走着順": [1, 3, 2],
        "同距離での成績": {"出走": 5, "1着": 2, "連対率": 0.6},
        "同会場での成績": {"出走": 4, "1着": 2},
        "前走馬体重kg": 486,
        "長距離輸送時の馬体重増減kg": -2,
    },
    "2": {
        "馬名": "サンプルゴースト",
        "性齢": "牝5",
        "斤量kg": 55.0,
        "所属": "栗東",
        "輸送区分": "当日輸送",
        "トレセンから会場までのkm": 520,
        "近走着順": [2, 1, 5],
        "同距離での成績": {"出走": 8, "1着": 3, "連対率": 0.5},
        "同会場での成績": {"出走": 1, "1着": 0},
        "前走馬体重kg": 452,
        "長距離輸送時の馬体重増減kg": -14,
    },
    "3": {
        "馬名": "ダミーブレイブ",
        "性齢": "牡6",
        "斤量kg": 58.0,
        "所属": "美浦",
        "輸送区分": "当日輸送",
        "トレセンから会場までのkm": 40,
        "近走着順": [7, 4, 6],
        "同距離での成績": {"出走": 3, "1着": 0, "連対率": 0.0},
        "同会場での成績": {"出走": 6, "1着": 1},
        "前走馬体重kg": 510,
        "長距離輸送時の馬体重増減kg": 0,
    },
}


def pad(text: str, width: int) -> str:
    """全角を2桁と数えて右側を空白で埋める。日本語の馬名で桁がずれないように。"""
    display = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - display)


def name_of(number: str) -> str:
    return HORSES[number]["馬名"]


def build_race_state() -> dict:
    """第1層の state = 出走全頭の数値サマリ。"""
    return {"レース": RACE, "出走馬": HORSES}


def build_horse_state(number: str) -> dict:
    """第2層の state = その馬1頭分の詳細。"""
    return {"レース": RACE, "この馬": HORSES[number], "馬番": number}


def summarize(horse: dict) -> str:
    """Choice の criteria に載せる、1頭あたりの短い特徴。"""
    distance = horse["同距離での成績"]
    return (
        f"{horse['馬名']} / {horse['性齢']} / 斤量{horse['斤量kg']}kg / "
        f"{horse['所属']}所属 / 近走{horse['近走着順']} / "
        f"同距離 {distance['出走']}戦{distance['1着']}勝"
    )


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Jev 疎通確認 (ダミーレース: 中山 芝2000m / 3頭立て)")
    print("=" * 72)

    try:
        client = create_client()
    except JevConfigError as error:
        print("\n[設定エラー]\n")
        print(error)
        return 1

    total_input = 0
    total_output = 0

    with client:
        # --- 第1層: レース単位。1リクエストで全頭の勝率分布が返る -------------
        print("\n[第1層] レース単位で勝率分布を取得中...")
        try:
            race_result = ask(
                client,
                build_race_state(),
                {
                    "winner": Choice(
                        instructions="このレースで最も勝ちやすいのはどの馬か",
                        criteria={n: summarize(h) for n, h in HORSES.items()},
                    )
                },
            )
        except JevRequestError as error:
            print("\n[API エラー]\n")
            print(error)
            return 1

        winner = race_result.choices["winner"]
        total_input += race_result.usage.input_tokens or 0
        total_output += race_result.usage.output_tokens or 0

        print(f"  model      : {race_result.model}")
        print(f"  最有力     : {winner.choice}番 ({name_of(winner.choice)})")
        print(f"  confidence : {winner.confidence:.3f}")
        print("  勝率分布   :")
        for number, probability in sorted(
            winner.probabilities.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"    {number}番 {pad(name_of(number), 18)} {probability:6.1%}")

        # --- 第2層: 馬単位。state が馬ごとに違うのでリクエストを分け、並列に投げる ---
        print(f"\n[第2層] {len(HORSES)}頭ぶんの軸評価を並列取得中"
              f" (軸: {', '.join(AXIS_QUESTIONS)})...")
        jobs = {
            number: Job(state=build_horse_state(number), questions=AXIS_QUESTIONS)
            for number in HORSES
        }
        try:
            horse_results = ask_parallel(client, jobs)
        except JevRequestError as error:
            print("\n[API エラー]\n")
            print(error)
            return 1

        axis_scores: dict[str, dict[str, float]] = {}
        for number, result in horse_results.items():
            axis_scores[number] = {name: result.nouls[name].noul for name in AXIS_QUESTIONS}
            total_input += result.usage.input_tokens or 0
            total_output += result.usage.output_tokens or 0

        print("  馬番 " + pad("馬名", 19) + " ".join(f"{name:>14}" for name in AXIS_QUESTIONS))
        for number in HORSES:
            values = " ".join(f"{axis_scores[number][name]:>14.3f}" for name in AXIS_QUESTIONS)
            print(f"  {number:>3}  {pad(name_of(number), 18)}{values}")

    # --- 第3層: Jev は使わない。重み付き和を取るだけ -------------------------
    # 注意: 第1層の確率は全頭で合計 1.0 になるスケール (3頭なら平均 0.33)、
    # 一方 Noul は1頭ごとに 0-1。頭数が変わると両者の釣り合いが変わるため、
    # 実運用では正規化の方法を決めること。ここでは CLAUDE.md の式をそのまま使う。
    print("\n[第3層] 重み付き和で合成 (重み: "
          + ", ".join(f"{k}={v}" for k, v in WEIGHTS.items()) + ")")

    totals: dict[str, float] = {}
    for number in HORSES:
        total = WEIGHTS["race_win"] * winner.probabilities[number]
        for name in AXIS_QUESTIONS:
            value = axis_scores[number][name]
            if name in INVERTED_AXES:
                value = 1.0 - value  # 「不利か」を問うた軸は反転して足す
            total += WEIGHTS[name] * value
        totals[number] = total

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n  --- 上位 {TOP_N} 頭 ---")
    for rank, (number, score) in enumerate(ranked, start=1):
        if rank == TOP_N + 1:
            print("  --- 以下 ---")
        print(f"  {rank}. {number}番 {pad(name_of(number), 18)} score={score:.4f}")

    print(f"\nトークン使用量: 入力 {total_input:,} / 出力 {total_output:,}"
          f" (概算コスト: ${total_input * 0.042 / 1_000_000:.6f} / 出力は無料)")
    print("\n疎通確認 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

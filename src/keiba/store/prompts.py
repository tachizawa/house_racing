"""Jev へ実際に送った state と質問文の保存。

予想記録 (``store/records.py``) が持つのは**答えだけ**だった。外れた予想を
見返しても「材料が悪かったのか、聞き方が悪かったのか」を切り分けられず、
何を送ったかはコードから再生成するしかなかった——だがコードは変わるので、
再生成は「いま送るとしたらこうなる」であって「あのとき送った物」ではない。
ここは後者を残す (report2 §6 の #7)。

予想記録本体 (``predictions/<YYYYMM>.jsonl``) には混ぜない。18頭のレースで
実測 244KB あるため、1行1レコードの jsonl に入れると ``load_month`` が読むたびに
全部を引きずることになる。同じ ``record_id`` で引けるよう、別の木に1レース
1ファイルで置く。

組み立てはしない——何を送ったかを知っているのは ``predict/layers.py`` と
``predict/pipeline.py`` で、ここは受け取った物をそのまま JSON にするだけ。
"""

from __future__ import annotations

import json
from pathlib import Path

from keiba.store.records import PredictionRecord

__all__ = ["save", "load", "path_for"]


def path_for(record: PredictionRecord, *, data_dir: Path) -> Path:
    """この記録に対応するファイルの場所。

    月で束ねるのは ``records.save`` と同じ理由（検証フェーズが月単位で読む）。
    月は ``predicted_at``（予想を出した時刻）から決める——レース施行日では
    なく、予想記録そのものが発生した時刻。同じ規則にそろえておかないと、
    記録とその送信内容が別の月のファイルに落ちる。
    """
    return (Path(data_dir) / "prompts" / record.predicted_at.strftime("%Y%m")
            / f"{record.record_id}.json")


def save(record: PredictionRecord, sent_prompts: dict, *, data_dir: Path) -> Path:
    """1レース1ファイルで書き出す。

    ``sent_prompts`` は ``predict()`` が埋めた辞書をそのまま渡す
    （``{"layer1": {"state", "questions"}, "layer2": {"questions", "states"} | None}``）。
    版（``record.versions``）を一緒に書くので、このファイル1つで「どの世代の
    文言・特徴量で作った state か」が分かる。
    """
    payload = {
        "record_id": record.record_id,
        "race_id": record.race_id,
        "predicted_at": record.predicted_at.isoformat(),
        "versions": dict(record.versions),
        "layer1": sent_prompts.get("layer1"),
        "layer2": sent_prompts.get("layer2"),
    }
    path = path_for(record, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 整形しない。第2層は馬ごとに state を1つずつ送り、そのどれもが
    # data_quality と opponents_summary を丸ごと含むので、1レースぶんが大きい
    # （18頭のレースで実測 215,000 文字）。indent=2 を付けると 454KB になり、
    # 100レースの実行で 45MB を超える。読むときに整形すればよい。
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def load(record: PredictionRecord, *, data_dir: Path) -> dict | None:
    """保存されていなければ ``None``。

    ``--no-save-prompts`` の実行や、この機能より前に作った記録には対応する
    ファイルが無い。それは壊れた状態ではないので例外にしない——「送った物が
    残っていない」ことを呼び出し側が区別できるよう、空辞書ではなく None を
    返す。
    """
    path = path_for(record, data_dir=data_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

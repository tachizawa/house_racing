"""Jev へ実際に送った state と質問文の保存 (keiba.store.prompts)。

予想記録 (PredictionRecord) には**答えしか残っていなかった**——何を送ったかは
記録から復元できず、外れた予想を見返しても「材料が悪かったのか、聞き方が
悪かったのか」を切り分けられなかった (report2 §6 の #7)。ここは「送った物を
そのまま、記録と同じ record_id で引ける場所に置く」ことだけを持つ。

予想記録本体 (predictions/*.jsonl) には混ぜない。18頭のレースで実測 244KB
(第2層は馬ごとに state を1つずつ送り、そのどれもが data_quality と
opponents_summary を丸ごと含むため) あるので、1行1レコードの jsonl に入れると
load_month が毎回それを読むことになる。
"""

from __future__ import annotations

import json
from datetime import datetime

from keiba.store import prompts, records


def _make_record(
    *,
    record_id: str = "rid-0",
    race_id: str | None = "2026032106010511",
    predicted_at: datetime | None = None,
) -> records.PredictionRecord:
    predicted_at = predicted_at or datetime(2026, 3, 21, 10, 0, 0)
    return records.PredictionRecord(
        record_id=record_id,
        race_id=race_id,
        predicted_at=predicted_at,
        prediction_cutoff=predicted_at,
        n=3,
        versions=records.current_versions(
            data_version="20260320T000000Z", model="jev-1.13.0", weights_id="w1"),
        horses=(),
        layer1={},
        data_quality={"notes": [], "horse_notes": {}, "tie_band_scale": "layer1_win_prob"},
        usage={},
        status="SUCCEEDED",
        race_input={"race_id": race_id},
    )


def _sent_prompts() -> dict:
    return {
        "layer1": {
            "state": {"race": {"jyo": "中山"}, "horses": {"01": {"bamei": "テストウマ"}}},
            "questions": {"winner": {"type": "choice", "instructions": "誰が勝つか",
                                      "criteria": {"01": "テストウマ"}}},
        },
        "layer2": {
            "questions": {"distance_fit": {"type": "choice", "instructions": "距離は",
                                            "criteria": {"positive": "良い"}}},
            "states": {"01": {"target_horse": {"bamei": "テストウマ"}}},
        },
    }


class TestSaveAndLoad:
    def test_round_trip_returns_what_was_sent(self, tmp_path):
        record = _make_record()
        sent = _sent_prompts()

        prompts.save(record, sent, data_dir=tmp_path)

        loaded = prompts.load(record, data_dir=tmp_path)
        assert loaded is not None
        assert loaded["layer1"] == sent["layer1"]
        assert loaded["layer2"] == sent["layer2"]

    def test_saved_under_month_and_record_id(self, tmp_path):
        record = _make_record(record_id="abc123",
                               predicted_at=datetime(2026, 3, 21, 10, 0, 0))

        path = prompts.save(record, _sent_prompts(), data_dir=tmp_path)

        assert path == tmp_path / "prompts" / "202603" / "abc123.json"
        assert path.exists()

    def test_load_returns_none_when_nothing_was_saved(self, tmp_path):
        """保存していない実行 (--no-save-prompts) は「壊れている」ではない。"""
        assert prompts.load(_make_record(), data_dir=tmp_path) is None

    def test_payload_carries_the_versions_of_the_record(self, tmp_path):
        """どの版の文言・特徴量で作った state なのかが、送った物の側にも残る。

        質問文は PROMPT_VERSION ごとに違う。版を書かずに state だけ残すと、
        後から読んだとき「この文言はいつの世代か」を予想記録と突き合わせない
        と分からない。1ファイルで完結させる。
        """
        record = _make_record()

        prompts.save(record, _sent_prompts(), data_dir=tmp_path)

        payload = json.loads(
            (tmp_path / "prompts" / "202603" / "rid-0.json").read_text(encoding="utf-8"))
        assert payload["record_id"] == record.record_id
        assert payload["race_id"] == record.race_id
        assert payload["predicted_at"] == record.predicted_at.isoformat()
        assert payload["versions"] == record.versions

    def test_written_compactly(self, tmp_path):
        """整形用の空白を入れない。

        1レースぶんが大きい——第2層は馬ごとに state を1つずつ送り、その
        どれもが data_quality と opponents_summary を丸ごと含むので、18頭の
        レースで送る量は第1層の state 1つの約7倍になる。実測で1レース
        215,000 文字、indent=2 で整形すると 465,137 バイト（454 KB）。
        100レースの実行で 45MB を超える。読むときに整形すればよいので、
        保存は詰めた形にする。
        """
        record = _make_record()

        path = prompts.save(record, _sent_prompts(), data_dir=tmp_path)

        text = path.read_text(encoding="utf-8")
        assert "\n  " not in text
        # 詰めても読み戻せることは round_trip のテストが押さえている。
        assert len(text) < len(json.dumps(
            json.loads(text), ensure_ascii=False, indent=2))

    def test_layer2_may_be_absent(self, tmp_path):
        """--no-layer2 の実行では第2層を送っていない。空の辞書を作らない。"""
        record = _make_record()
        sent = {"layer1": _sent_prompts()["layer1"], "layer2": None}

        prompts.save(record, sent, data_dir=tmp_path)

        loaded = prompts.load(record, data_dir=tmp_path)
        assert loaded["layer2"] is None

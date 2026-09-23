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

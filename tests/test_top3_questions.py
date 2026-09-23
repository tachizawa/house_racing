from __future__ import annotations

import re

from typesafe_sdk import Choice, Noul

from keiba.predict import top3_questions as tq

DOSSIER = {"course": {}, "race": {}, "field_summary": {}, "data_quality": {},
           "horses": {"01": {"bamei": "イチ", "sex_age": "牡4", "futan_kg": 57.0},
                      "02": {"bamei": "ニ", "sex_age": "牝3", "futan_kg": 55.0}}}
STATE_KEYS = {"course", "race", "horses", "field_summary", "data_quality",
              "prediction_cutoff", "dossier_version"}


def test_one_noul_per_horse_plus_winner():
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=False)
    assert set(qs) == {"top3_01", "top3_02", "winner"}
    assert isinstance(qs["top3_01"], Noul)
    assert isinstance(qs["winner"], Choice)


def test_each_noul_names_its_target_horse():
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=False)
    assert "対象馬: 01" in qs["top3_01"].instructions
    assert "対象馬: 02" in qs["top3_02"].instructions
    assert "対象馬: 01" not in qs["top3_02"].instructions


def test_noul_criteria_defines_true_and_false():
    dumped = tq.top3_questions(DOSSIER, with_pace_forecast=False)["top3_01"].model_dump()
    assert "3着以内" in dumped["criteria"]["true"]
    assert "4着以下" in dumped["criteria"]["false"]


def test_pace_forecast_mentioned_only_when_present():
    without = tq.top3_questions(DOSSIER, with_pace_forecast=False)["top3_01"].instructions
    with_ = tq.top3_questions(DOSSIER, with_pace_forecast=True)["top3_01"].instructions
    assert "pace_forecast" not in without
    assert "pace_forecast" in with_


def test_instructions_reference_only_existing_state_keys():
    # 「state のキー」として名指しするのは ASCII の識別子。存在しないキーを名指ししない。
    for with_pace in (False, True):
        allowed = STATE_KEYS | ({"pace_forecast"} if with_pace else set())
        for q in tq.top3_questions(DOSSIER, with_pace_forecast=with_pace).values():
            named = set(re.findall(r"\b([a-z_]+)：", q.instructions))
            assert named <= allowed, named - allowed


def test_winner_pace_forecast_mentioned_only_when_present():
    without = tq.top3_questions(DOSSIER, with_pace_forecast=False)["winner"].instructions
    with_ = tq.top3_questions(DOSSIER, with_pace_forecast=True)["winner"].instructions
    assert "pace_forecast" not in without
    assert "pace_forecast" in with_


def test_winner_criteria_are_umaban():
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=False)
    assert set(qs["winner"].model_dump()["criteria"]) == {"01", "02"}


def test_pace_questions_reuse_existing_bands():
    qs = tq.pace_questions()
    assert set(qs["pace"].model_dump()["criteria"]) == {"fast", "standard", "slow"}


def test_rules_warn_that_truncated_history_may_be_partial():
    # found_may_be_truncated は取得期間が短いと頻出する（final review I-3a）。
    for q in tq.top3_questions(DOSSIER, with_pace_forecast=False).values():
        assert "found_may_be_truncated" in q.instructions
        assert "一部だけ" in q.instructions


def test_winner_rules_do_not_mention_a_target_horse():
    # winner には対象馬が無い（final review I-3b）。top3 は対象馬のまま。
    qs = tq.top3_questions(DOSSIER, with_pace_forecast=True)
    assert "対象馬" not in qs["winner"].instructions
    assert "各馬の過去走" in qs["winner"].instructions
    assert "対象馬の過去走" in qs["top3_01"].instructions
    assert "pace_forecast" in qs["winner"].instructions

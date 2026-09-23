from __future__ import annotations

import types

import pytest
import typesafe_sdk as ts

from jev_client import JevRequestError
from keiba.features import dossier as dossier_mod
from keiba.predict import top3

DOSSIER = {"dossier_version": "1", "prediction_cutoff": "x", "course": {}, "race": {},
           "field_summary": {}, "data_quality": {},
           "horses": {"01": {"bamei": "イチ"}, "02": {"bamei": "ニ"}, "03": {"bamei": "サン"}}}


class Resp:
    def __init__(self, nouls=None, choices=None, tokens=500):
        self.nouls = nouls or {}
        self.choices = choices or {}
        self.model = "jev-1.13.0"
        self.usage = types.SimpleNamespace(input_tokens=tokens)


def noul(p):
    return types.SimpleNamespace(noul=p)


def winner(probs):
    return types.SimpleNamespace(choice=max(probs, key=probs.get), probabilities=probs,
                                 confidence=0.5)


class Client:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def system_one(self, state, questions, **kw):
        self.calls.append((state, questions))
        return self.handler(state, questions)


def full_handler(state, questions):
    return Resp(nouls={"top3_01": noul(0.7), "top3_02": noul(0.5), "top3_03": noul(0.2)},
                choices={"winner": winner({"01": 0.5, "02": 0.3, "03": 0.2})})


def test_plan_a_single_request_with_all_questions():
    client = Client(full_handler)
    answer = top3.run_plan_a(client, DOSSIER)
    assert len(client.calls) == 1
    assert set(client.calls[0][1]) == {"top3_01", "top3_02", "top3_03", "winner"}
    assert answer.top3_probabilities == {"01": 0.7, "02": 0.5, "03": 0.2}
    assert answer.requests == 1
    assert answer.input_tokens == 500


def test_plan_a_does_not_normalize():
    answer = top3.run_plan_a(Client(full_handler), DOSSIER)
    assert abs(sum(answer.top3_probabilities.values()) - 1.4) < 1e-9


def test_plan_a_keeps_what_it_sent():
    client = Client(full_handler)
    answer = top3.run_plan_a(client, DOSSIER)
    assert answer.sent["requests"][0]["state"] is client.calls[0][0]
    assert answer.sent["requests"][0]["questions"]["top3_01"]["type"] == "noul"


def test_missing_top3_answer_is_failure_not_zero():
    def handler(state, questions):
        return Resp(nouls={"top3_01": noul(0.7), "top3_02": noul(0.5)},
                    choices={"winner": winner({"01": 0.5, "02": 0.3, "03": 0.2})})
    with pytest.raises(top3.Top3IncompleteError) as info:
        top3.run_plan_a(Client(handler), DOSSIER)
    assert info.value.missing == ["top3_03"]


def test_missing_winner_entry_is_recorded_not_failure():
    def handler(state, questions):
        return Resp(nouls={f"top3_0{i}": noul(0.3) for i in (1, 2, 3)},
                    choices={"winner": winner({"01": 0.6, "02": 0.4})})
    answer = top3.run_plan_a(Client(handler), DOSSIER)
    assert answer.winner_probabilities["03"] == 0.0
    assert answer.winner_missing == ("03",)


def test_state_too_large_is_raised_before_sending(monkeypatch):
    client = Client(full_handler)
    monkeypatch.setattr(top3.config, "STATE_MAX_CHARS", 10)
    with pytest.raises(dossier_mod.StateTooLargeError):
        top3.run_plan_a(client, DOSSIER)
    assert client.calls == []


def test_api_error_is_wrapped():
    def handler(state, questions):
        raise ts.TypeSafeAPITimeoutError(10.0)
    with pytest.raises(JevRequestError):
        top3.run_plan_a(Client(handler), DOSSIER)


PACE_DEF = {"fast_max": 35.0, "standard_max": 36.0, "stratum": {"track_kind": "turf"},
            "race_count": 40, "note": "n"}


def plan_b_handler(state, questions):
    if "pace" in questions:
        assert "pace_definition" in state
        return Resp(choices={"pace": types.SimpleNamespace(
            choice="fast", probabilities={"fast": 0.5, "standard": 0.3, "slow": 0.2},
            confidence=0.4)}, tokens=300)
    assert state["pace_forecast"]["probabilities"] == {"fast": 0.5, "standard": 0.3, "slow": 0.2}
    assert "pace_definition" not in state
    return full_handler(state, questions)


def test_plan_b_two_requests_and_passes_distribution():
    client = Client(plan_b_handler)
    answer = top3.run_plan_b(client, DOSSIER, pace_definition=PACE_DEF)
    assert len(client.calls) == 2
    assert answer.requests == 2
    assert answer.input_tokens == 800
    assert answer.pace_probabilities == {"fast": 0.5, "standard": 0.3, "slow": 0.2}
    assert len(answer.sent["requests"]) == 2


def test_plan_b_second_request_mentions_pace_forecast():
    client = Client(plan_b_handler)
    top3.run_plan_b(client, DOSSIER, pace_definition=PACE_DEF)
    assert "pace_forecast" in client.calls[1][1]["top3_01"].instructions


def test_plan_b_without_pace_definition_does_not_send():
    client = Client(plan_b_handler)
    with pytest.raises(top3.PaceUnavailableError):
        top3.run_plan_b(client, DOSSIER, pace_definition=None)
    assert client.calls == []


def test_plan_b_second_step_incomplete_keeps_pace_tokens():
    # pace は課金済み。2回目の答えが欠けても pace のトークンを失わない（final review I-4b）。
    def handler(state, questions):
        if "pace" in questions:
            return plan_b_handler(state, questions)
        return Resp(nouls={"top3_01": noul(0.7)},
                    choices={"winner": winner({"01": 0.5, "02": 0.3, "03": 0.2})})
    client = Client(handler)
    with pytest.raises(top3.PlanBPartialFailure) as info:
        top3.run_plan_b(client, DOSSIER, pace_definition=PACE_DEF)
    assert isinstance(info.value.cause, top3.Top3IncompleteError)
    assert info.value.input_tokens == 300
    assert info.value.requests_sent == 2
    assert len(client.calls) == 2


def test_plan_b_second_step_api_error_keeps_pace_tokens():
    def handler(state, questions):
        if "pace" in questions:
            return plan_b_handler(state, questions)
        raise ts.TypeSafeAPITimeoutError(10.0)
    with pytest.raises(top3.PlanBPartialFailure) as info:
        top3.run_plan_b(Client(handler), DOSSIER, pace_definition=PACE_DEF)
    assert isinstance(info.value.cause, JevRequestError)
    assert info.value.input_tokens == 300
    assert info.value.requests_sent == 2


def test_plan_b_too_large_second_state_sends_nothing(monkeypatch):
    # pace の state は上限内だが、pace_forecast を足した2回目が上限を超える。
    # pace を聞いてから気付くと pace の代金が無駄になる（final review I-4b）。
    pace_state = dossier_mod.to_state(DOSSIER)
    pace_state["pace_definition"] = PACE_DEF
    limit = dossier_mod.state_chars(pace_state)
    monkeypatch.setattr(top3.config, "STATE_MAX_CHARS", limit)
    client = Client(plan_b_handler)
    with pytest.raises(dossier_mod.StateTooLargeError):
        top3.run_plan_b(client, DOSSIER, pace_definition=PACE_DEF)
    assert client.calls == []


def test_plan_b_size_check_is_worst_case_for_the_real_forecast():
    # 事前検査に使う仮の pace_forecast は、実際に送る物より短くない。
    ugly = {"fast": 1 / 3, "standard": 1e-300 / 7, "slow": 0.123456789012345678}
    real = dossier_mod.to_state(DOSSIER, pace_forecast={
        "probabilities": ugly, "definition": PACE_DEF, "note": top3._PACE_NOTE})
    worst = dossier_mod.to_state(DOSSIER, pace_forecast=top3._worst_case_pace_forecast(PACE_DEF))
    assert dossier_mod.state_chars(worst) >= dossier_mod.state_chars(real)

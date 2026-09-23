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

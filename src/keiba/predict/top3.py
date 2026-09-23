"""案A / 案B の Jev 呼び出し（report3 §5）。

どちらも「全頭の 3着内確率（Noul）」と「同じ state での winner（Choice）」を
1リクエストで聞く。案Bはその前に展開（pace）を1回聞き、確率分布のまま
state の pace_forecast に入れる（最尤区分だけを渡すと確定した事実として
扱われるため）。

3着内確率は正規化しない。応答に欠けた top3_XX があれば 0.0 で埋めず
Top3IncompleteError にする（第2層と同じ扱い）。winner の欠けは 0.0 で埋めて
馬番を記録する（第1層と同じ扱い。比較対象として使うだけのため）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jev_client import ask
from keiba import config
from keiba.features.dossier import check_state_size, to_state
from keiba.predict import top3_questions as tq

__all__ = ["PaceUnavailableError", "Top3Answer", "Top3IncompleteError",
           "run_plan_a", "run_plan_b"]


class Top3IncompleteError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"3着内の答えが欠けています: {missing}")
        self.missing = missing


class PaceUnavailableError(RuntimeError):
    """ペース区分の定義が作れない。pace の質問文は state の pace_definition を
    名指しするので、定義が無いまま送ってはいけない（既存 spec §8.2 と同じ理由）。"""


@dataclass(frozen=True)
class Top3Answer:
    top3_probabilities: dict[str, float]
    winner_probabilities: dict[str, float]
    winner_missing: tuple[str, ...]
    pace_probabilities: dict[str, float] | None
    model: str
    input_tokens: int
    requests: int
    state_chars: int
    sent: dict
    seconds: float


def _dump(questions: dict) -> dict:
    return {name: q.model_dump() for name, q in questions.items()}


def _ask_top3(client: Any, dossier: dict, state: dict, *, with_pace_forecast: bool):
    chars = check_state_size(state, limit=config.STATE_MAX_CHARS)
    questions = tq.top3_questions(dossier, with_pace_forecast=with_pace_forecast)
    response = ask(client, state, questions)
    missing = [tq.question_name(u) for u in dossier["horses"]
               if tq.question_name(u) not in response.nouls]
    if missing:
        raise Top3IncompleteError(missing)
    top3 = {u: float(response.nouls[tq.question_name(u)].noul) for u in dossier["horses"]}
    winner_probs: dict[str, float] = {}
    winner_missing: list[str] = []
    raw = response.choices["winner"].probabilities if "winner" in response.choices else {}
    for u in dossier["horses"]:
        if u in raw:
            winner_probs[u] = float(raw[u])
        else:
            winner_probs[u] = 0.0
            winner_missing.append(u)
    return response, top3, winner_probs, tuple(winner_missing), chars, _dump(questions)


def run_plan_a(client: Any, dossier: dict) -> Top3Answer:
    started = time.perf_counter()
    state = to_state(dossier)
    response, top3, winner_probs, winner_missing, chars, sent_q = _ask_top3(
        client, dossier, state, with_pace_forecast=False)
    return Top3Answer(
        top3_probabilities=top3, winner_probabilities=winner_probs,
        winner_missing=winner_missing, pace_probabilities=None,
        model=response.model, input_tokens=response.usage.input_tokens or 0,
        requests=1, state_chars=chars,
        sent={"requests": [{"state": state, "questions": sent_q}]},
        seconds=time.perf_counter() - started)


_PACE_NOTE = "前段で予測した展開。確定した事実ではない。"


def run_plan_b(client: Any, dossier: dict, *, pace_definition: dict | None) -> Top3Answer:
    if pace_definition is None:
        raise PaceUnavailableError("ペース区分の定義（pace_definition）が作れないため案Bを実行しない。")
    started = time.perf_counter()

    pace_state = to_state(dossier)
    pace_state["pace_definition"] = pace_definition
    check_state_size(pace_state, limit=config.STATE_MAX_CHARS)
    pace_q = tq.pace_questions()
    pace_response = ask(client, pace_state, pace_q)
    if "pace" not in pace_response.choices:
        raise Top3IncompleteError(["pace"])
    pace_probs = {k: float(v) for k, v in pace_response.choices["pace"].probabilities.items()}

    # 最尤区分だけでなく分布のまま渡す（確定した事実として扱わせない）。
    state = to_state(dossier, pace_forecast={
        "probabilities": pace_probs, "definition": pace_definition, "note": _PACE_NOTE})
    response, top3, winner_probs, winner_missing, chars, sent_q = _ask_top3(
        client, dossier, state, with_pace_forecast=True)
    return Top3Answer(
        top3_probabilities=top3, winner_probabilities=winner_probs,
        winner_missing=winner_missing, pace_probabilities=pace_probs,
        model=response.model,
        input_tokens=(pace_response.usage.input_tokens or 0) + (response.usage.input_tokens or 0),
        requests=2, state_chars=chars,
        sent={"requests": [{"state": pace_state, "questions": _dump(pace_q)},
                           {"state": state, "questions": sent_q}]},
        seconds=time.perf_counter() - started)

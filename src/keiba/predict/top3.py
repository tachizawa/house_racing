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

import sys
import time
from dataclasses import dataclass
from typing import Any

from jev_client import JevRequestError, ask
from keiba import config
from keiba.features.dossier import StateTooLargeError, check_state_size, to_state
from keiba.predict import top3_questions as tq
from keiba.predict.questions import PACE_CHOICES

__all__ = ["PaceUnavailableError", "PlanBPartialFailure", "Top3Answer",
           "Top3IncompleteError", "run_plan_a", "run_plan_b"]


class Top3IncompleteError(RuntimeError):
    """応答は返ったが、必要な答えが欠けていた。

    応答が返った時点で課金は確定しているので、その応答の利用量
    （``input_tokens`` / ``requests``）を持って上へ投げる。呼び出し側は
    これを usage に足す（足さないと使用量と費用を過少報告する）。
    """

    def __init__(self, missing: list[str], *, input_tokens: int = 0,
                 requests: int = 0) -> None:
        super().__init__(f"3着内の答えが欠けています: {missing}")
        self.missing = missing
        self.input_tokens = input_tokens
        self.requests = requests


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
        raise Top3IncompleteError(missing, input_tokens=response.usage.input_tokens or 0,
                                  requests=1)
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

# JSON にしたとき最も長くなる正の float（repr が23文字）。事前の大きさ検査で
# 確率の値をこれに置き換えれば、実際に返ってくるどの確率より短くならない。
_LONGEST_FLOAT = sys.float_info.min


class PlanBPartialFailure(RuntimeError):
    """案Bで、展開（pace）のリクエストが成功した**後**に失敗した。

    pace のリクエストは課金済みなので、そのトークン数を失わないよう持って
    上へ投げる（final review I-4b）。``cause`` が元の例外
    （Top3IncompleteError / JevRequestError / StateTooLargeError）。
    ``requests_sent`` は2回目を送る前に止めたなら 1、送ったなら 2。
    ``input_tokens`` は利用量が読めた応答の合計（pace と、``cause`` が
    Top3IncompleteError なら2回目の応答のぶんも含む）。
    """

    def __init__(self, cause: Exception, *, input_tokens: int, requests_sent: int,
                 seconds: float) -> None:
        super().__init__(f"案Bの展開予測の後で失敗しました: {cause}")
        self.cause = cause
        self.input_tokens = input_tokens
        self.requests_sent = requests_sent
        self.seconds = seconds


def _pace_forecast(probabilities: dict[str, float], pace_definition: dict) -> dict:
    return {"probabilities": probabilities, "definition": pace_definition, "note": _PACE_NOTE}


def _worst_case_pace_forecast(pace_definition: dict) -> dict:
    """2回目の state の大きさを pace を聞く前に測るための仮の pace_forecast。

    キーは実物と同じ。確率は全区分ぶん、最も長い表記になる値で埋める
    （Jev は渡した選択肢の中からしか答えないので、区分がこれより増えることはない）。
    """
    return _pace_forecast({band: _LONGEST_FLOAT for band in PACE_CHOICES}, pace_definition)


def run_plan_b(client: Any, dossier: dict, *, pace_definition: dict | None) -> Top3Answer:
    if pace_definition is None:
        raise PaceUnavailableError("ペース区分の定義（pace_definition）が作れないため案Bを実行しない。")
    started = time.perf_counter()

    pace_state = to_state(dossier)
    pace_state["pace_definition"] = pace_definition
    check_state_size(pace_state, limit=config.STATE_MAX_CHARS)
    # 2回目の state が上限を超えるなら、pace を聞く前に止める（pace の代金を無駄にしない）。
    check_state_size(to_state(dossier, pace_forecast=_worst_case_pace_forecast(pace_definition)),
                     limit=config.STATE_MAX_CHARS)
    pace_q = tq.pace_questions()
    pace_response = ask(client, pace_state, pace_q)
    pace_tokens = pace_response.usage.input_tokens or 0
    if "pace" not in pace_response.choices:
        raise Top3IncompleteError(["pace"], input_tokens=pace_tokens, requests=1)
    pace_probs = {k: float(v) for k, v in pace_response.choices["pace"].probabilities.items()}

    # 最尤区分だけでなく分布のまま渡す（確定した事実として扱わせない）。
    state = to_state(dossier, pace_forecast=_pace_forecast(pace_probs, pace_definition))
    try:
        response, top3, winner_probs, winner_missing, chars, sent_q = _ask_top3(
            client, dossier, state, with_pace_forecast=True)
    except (Top3IncompleteError, JevRequestError, StateTooLargeError) as error:
        # StateTooLargeError は送る前に弾いている（上の事前検査があるので通常は来ない）。
        # Top3IncompleteError は2回目の応答が返っているので、その利用量も足す。
        second_tokens = error.input_tokens if isinstance(error, Top3IncompleteError) else 0
        raise PlanBPartialFailure(
            error, input_tokens=pace_tokens + second_tokens,
            requests_sent=1 if isinstance(error, StateTooLargeError) else 2,
            seconds=time.perf_counter() - started) from error
    return Top3Answer(
        top3_probabilities=top3, winner_probabilities=winner_probs,
        winner_missing=winner_missing, pace_probabilities=pace_probs,
        model=response.model,
        input_tokens=pace_tokens + (response.usage.input_tokens or 0),
        requests=2, state_chars=chars,
        sent={"requests": [{"state": pace_state, "questions": _dump(pace_q)},
                           {"state": state, "questions": sent_q}]},
        seconds=time.perf_counter() - started)

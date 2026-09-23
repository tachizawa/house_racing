"""第1層 (レース単位) と第2層 (馬単位) の Jev 呼び出しをオーケストレーションする。

state の組み立ては ``features/summary.py``、質問文は ``predict/questions.py``
の仕事。ここでは両者を ``jev_client.Job`` にまとめて送り、返ってきた答えを
そのままの形 (確率分布・confidence) で束ねるだけ。数値特徴への変換や重み付け
はしない——それは第3層 (Task 14 以降) の仕事。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jev_client import Job, JevRequestError, ask, ask_parallel
from keiba.features.context import PredictionContext
from keiba.features.summary import build_horse_state, build_race_state
from keiba.predict import questions

__all__ = [
    "AxisAnswer",
    "Layer1IncompleteError",
    "Layer1Result",
    "Layer2IncompleteError",
    "Layer2Result",
    "run_layer1",
    "run_layer2",
]


@dataclass(frozen=True)
class Layer1Result:
    """第1層 (レース単位、1リクエスト) の答え。"""

    win_probabilities: dict[str, float]
    choice: str
    confidence: float
    pace_probabilities: dict[str, float] | None
    model: str
    input_tokens: int
    # モデルが確率を返さなかった馬番。0.0 で埋めた馬がここに入る。
    # 0.0 には「勝てないと判断した」と「言及されなかった」の2つの意味があり、
    # 値だけでは区別できない。合計が 1.0 に足りないことも手がかりにはなるが、
    # どの馬が欠けたかは復元できないし、浮動小数の誤差とも紛らわしい。
    missing_probability_umaban: tuple[str, ...]
    #: 実際に送った state と質問（JSON-safe な形）。**送った物そのもの**を
    #: 持ち帰る——あとで build_race_state を呼び直して再生成する方式だと、
    #: それは「いまのコードが送るとしたらどうなるか」であって、あのとき
    #: 送った物ではない（report2 §6 の #7）。既定値を持つのは、この2欄を
    #: 知らない既存の組み立て（テストの偽 Layer1Result など）を壊さないため。
    sent_state: dict = field(default_factory=dict)
    sent_questions: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class AxisAnswer:
    """第2層の1軸ぶんの答え。

    confidence はどの特徴でも使わないが、予想記録として後から見返せるように
    そのまま保持する (数値特徴への変換は Task 14 の仕事で、ここではしない)。
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class Layer2Result:
    """第2層 (馬単位、並列) の答え。"""

    answers: dict[str, dict[str, AxisAnswer]]
    failures: dict[str, str]
    input_tokens: int
    #: 馬番 → その馬に実際に送った state。失敗した馬の分も残す（リクエストは
    #: 成立して課金されているので、何を送って失敗したかは実在する事実）。
    sent_states: dict[str, dict] = field(default_factory=dict)
    #: 5軸の質問文。**全頭で同じなので1回だけ持つ**（18頭ぶん18回書かない）。
    sent_questions: dict[str, dict] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.failures


class Layer2IncompleteError(RuntimeError):
    """``allow_partial=False`` で1頭でも失敗したときに投げる。

    ``ask_parallel`` の既定 (``return_exceptions=False``) は最初の失敗だけを
    伝えて残りを打ち切るので、それでは何頭失敗したか分からない。ここでは常に
    ``return_exceptions=True`` で全頭を走らせ切ってから、失敗をまとめて
    1つの例外にする。``failures`` にどの馬番がどう失敗したかが残る。
    """

    def __init__(self, failures: dict[str, str]) -> None:
        detail = "; ".join(f"{umaban}: {reason}" for umaban, reason in failures.items())
        super().__init__(f"第2層が {len(failures)} 頭ぶん失敗しました: {detail}")
        self.failures = failures


class Layer1IncompleteError(RuntimeError):
    """第1層の応答に ``winner`` が無い（質問したのに答えが返っていない）。

    ``response.choices["winner"]`` を無検証で引くと生の ``KeyError`` が
    ``predict()`` の外まで漏れ、利用者にトレースバックが出る。第1層は
    「確率辞書に全馬が載る保証はない」と明記して欠落を記録までしているのに、
    答えそのものの存在は仮定していた（fix5 決定2）。``JevRequestError``
    とは別のクラスにする——通信は成功しており、失敗したのは応答の形なので、
    「API が落ちた」と報告させない。
    """


def _questions_to_json(asked: Mapping[str, Any]) -> dict[str, dict]:
    """送った質問を JSON-safe な形にする。

    ``Choice`` は pydantic のモデルなので ``model_dump()`` が
    ``{"type", "instructions", "criteria"}`` をそのまま返す。ここで
    ``instructions`` だけを抜き出したりしない——選択肢（criteria）は答えの
    意味そのもので、後から「何を選ばせたのか」を復元するのに要る。
    """
    return {name: question.model_dump() for name, question in asked.items()}


def run_layer1(client: Any, ctx: PredictionContext, *, with_pace: bool = False) -> Layer1Result:
    """レース単位で1リクエスト。全頭の勝率分布が1回で返る。"""
    # state と questions をローカルに束ねてから送る。送信と記録で別々に
    # 組み立てると、片方だけが直されたときに「記録にある物」と「送った物」が
    # 黙ってずれる。
    state = build_race_state(ctx)
    asked = questions.layer1_questions(ctx, with_pace=with_pace)
    response = ask(client, state, asked)
    if "winner" not in response.choices:
        raise Layer1IncompleteError(
            "第1層の応答に winner がありません（返ってきた質問: "
            f"{sorted(response.choices)}）。")
    winner = response.choices["winner"]

    # Jev は渡した選択肢の中だけを返すが、確率辞書に全馬が載る保証はない。
    # 欠けた馬は 0.0 で埋め、どの馬が欠けたかを missing_probability_umaban に
    # 名指しで残す——0.0 には「勝てないと判断した」と「言及されなかった」の
    # 2つの意味があり、値だけでは区別できないため。再正規化はしない——合計が
    # 1.0 から外れていること自体が「分布が取れていない」という情報なので、
    # 割り算で隠さない。順序は ctx.horses の順を保つ (再現性のため)。
    probabilities: dict[str, float] = {}
    missing_umaban: list[str] = []
    for horse in ctx.horses:
        if horse.umaban in winner.probabilities:
            probabilities[horse.umaban] = float(winner.probabilities[horse.umaban])
        else:
            probabilities[horse.umaban] = 0.0
            missing_umaban.append(horse.umaban)

    pace = None
    if with_pace and "pace" in response.choices:
        pace = {k: float(v) for k, v in response.choices["pace"].probabilities.items()}

    return Layer1Result(
        win_probabilities=probabilities,
        choice=winner.choice,
        confidence=float(winner.confidence),
        pace_probabilities=pace,
        model=response.model,
        input_tokens=response.usage.input_tokens or 0,
        missing_probability_umaban=tuple(missing_umaban),
        sent_state=state,
        sent_questions=_questions_to_json(asked),
    )


def _incomplete_reason(answers: dict[str, AxisAnswer]) -> str | None:
    """1頭分の第2層の答えが不完全なら理由を、完全なら None を返す。

    見るのは2つ。(1) ``AXIS_SPECS`` の5軸すべてがあるか、(2) 各軸の確率辞書が
    その軸の全選択肢（``AxisSpec.criteria`` のキー）を覆っているか。

    第1層は「確率辞書に全馬が載る保証はない」と明記して欠落を記録している
    のに、第2層は同じ SDK の同じ形（``choices`` と ``probabilities``）に対して
    完全性を無検証で仮定していた。その結果、1軸が欠ける応答で
    ``KeyError: 'current_form'``、選択肢が1つ欠ける応答で
    ``KeyError: 'negative'`` が ``predict()`` の外まで生で漏れ、利用者には
    トレースバックが、課金済みのバックテストには1頭の応答での途中終了
    （``report.json`` も書かれない）が起きていた（fix5 決定2）。

    欠けた分を 0.0 で埋めることはしない（spec §4.4「欠損値を0や不適性に
    置き換えない」）。この馬を失敗として扱い、既存の per-horse failure の
    経路（``allow_partial``）に載せる。
    """
    missing_axes = [name for name in questions.AXIS_SPECS if name not in answers]
    if missing_axes:
        return f"第2層の応答に軸が欠けています: {missing_axes}"
    for name, spec in questions.AXIS_SPECS.items():
        missing_keys = [key for key in spec.criteria
                        if key not in answers[name].probabilities]
        if missing_keys:
            return f"第2層の応答の {name} に選択肢の確率が欠けています: {missing_keys}"
    return None


def run_layer2(
    client: Any,
    ctx: PredictionContext,
    layer1: Layer1Result,
    *,
    max_workers: int = 8,
    allow_partial: bool = False,
) -> Layer2Result:
    """馬単位で並列に1リクエストずつ。1頭の state に5軸すべての質問をまとめる。

    応答は ``_incomplete_reason`` で完全性を検査する。5軸のどれかが欠ける、
    あるいはどれかの軸の確率キーが足りない応答は、その馬を失敗として
    ``failures`` に入れる（0.0 で埋めない）。これで既存の ``allow_partial``
    の分岐にそのまま乗る（``predict`` は FAILED、``backtest`` は skip）。
    """
    asked = questions.layer2_questions()
    if not ctx.horses:
        return Layer2Result(answers={}, failures={}, input_tokens=0,
                            sent_states={}, sent_questions=_questions_to_json(asked))

    race_context = None
    if layer1.pace_probabilities is not None:
        # 最尤区分だけを渡すと「そのペースに決まった」前提で評価されてしまう
        # (spec §5.3「最大確率の展開を、確定した事実として扱わない」)。
        # 3区分すべてを確率のまま渡す。
        race_context = {"pace_probabilities": layer1.pace_probabilities}

    # state は馬ごとに1回だけ組み立て、それを送信にも記録にも使う（第1層と
    # 同じ理由——2回組み立てると片方だけ直されたときに黙ってずれる）。
    sent_states = {
        horse.umaban: build_horse_state(ctx, horse.umaban, race_context=race_context)
        for horse in ctx.horses
    }
    jobs = {
        umaban: Job(state=state, questions=asked)
        for umaban, state in sent_states.items()
    }

    # allow_partial の値によらず return_exceptions=True で全頭を走らせ切る。
    # ask_parallel の既定 (最初の失敗で打ち切り) に任せると、allow_partial=False
    # で例外に変換する際にも「2頭目以降が失敗したか」が分からなくなる。
    raw_results = ask_parallel(client, jobs, max_workers=max_workers, return_exceptions=True)

    answers: dict[str, dict[str, AxisAnswer]] = {}
    failures: dict[str, str] = {}
    input_tokens = 0
    for umaban, result in raw_results.items():
        if isinstance(result, JevRequestError):
            failures[umaban] = str(result)
            continue
        horse_answers = {
            axis: AxisAnswer(
                choice=axis_result.choice,
                probabilities=dict(axis_result.probabilities),
                confidence=float(axis_result.confidence),
            )
            for axis, axis_result in result.choices.items()
        }
        # トークンは応答が不完全でも数える（リクエストは成立して課金されて
        # いるので、使用量としては実在する）。
        input_tokens += result.usage.input_tokens or 0
        reason = _incomplete_reason(horse_answers)
        if reason is not None:
            failures[umaban] = reason
            continue
        answers[umaban] = horse_answers

    if failures and not allow_partial:
        raise Layer2IncompleteError(failures)

    return Layer2Result(
        answers=answers, failures=failures, input_tokens=input_tokens,
        sent_states=sent_states, sent_questions=_questions_to_json(asked),
    )

"""keiba.predict.layers のテスト。

第1層・第2層のオーケストレーションだけを見る。Jev の実応答は使わず、
system_one を差し替えた偽クライアントで検証する。特徴量の数値化や重み付けは
別の層 (Task 14) の責任なので、ここでは answers/probabilities をそのまま
受け取れているかだけを見る。
"""

from __future__ import annotations

import types
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pytest
import typesafe_sdk as ts

from jev_client import JevRequestError
from keiba.data.repository import Repository
from keiba.features.context import PredictionContext, build_context
from keiba.models import HorseEntry, RaceInput
from keiba.predict import layers
from keiba.predict.questions import AXIS_NAMES, AXIS_SPECS

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


def make_entry(*, umaban="01", bamei="テストウマ", **kw) -> HorseEntry:
    return HorseEntry(umaban=umaban, bamei=bamei, **kw)


def make_race_input(horses, **kw) -> RaceInput:
    defaults = dict(
        date=date(2026, 3, 21),
        jyo="中山",
        race_num=11,
        track_kind="turf",
        kyori_m=2000,
        entry_count=len(horses),
        prediction_cutoff=jst(2026, 3, 20, 20, 0, 0),
        horses=tuple(horses),
        jyo_cd="06",
        baba_cd="1",
    )
    defaults.update(kw)
    return RaceInput(**defaults)


def make_context(entries) -> PredictionContext:
    """馬名照合を全頭「未登録」に固定した最小の PredictionContext。

    layers.py が見るのは build_race_state / build_horse_state の出力と
    ctx.horses の馬番集合だけなので、test_questions.py と同じ最小構成でよい
    (本物の Repository/CSV は経由しない)。
    """
    mock_repo = mock.create_autospec(Repository, instance=True)
    mock_repo.find_horses_by_name.return_value = []
    race_input = make_race_input(entries)
    return build_context(race_input, mock_repo)


@pytest.fixture
def ctx_three_horses():
    entries = [
        make_entry(umaban="01", bamei="イチバンウマ"),
        make_entry(umaban="02", bamei="ニバンウマ"),
        make_entry(umaban="03", bamei="サンバンウマ"),
    ]
    return make_context(entries)


@pytest.fixture
def ctx_no_horses():
    return make_context([])


def make_layer1_result(*, pace_probabilities=None) -> layers.Layer1Result:
    """run_layer2 のテストで使う、値そのものは重要でない Layer1Result。"""
    return layers.Layer1Result(
        win_probabilities={},
        choice="01",
        confidence=0.5,
        pace_probabilities=pace_probabilities,
        model="jev-1.13.0",
        input_tokens=0,
        missing_probability_umaban=(),
    )


# --- 偽クライアント ------------------------------------------------------------

class FakeResponse:
    def __init__(self, choices=None, model="jev-1.13.0", input_tokens=100):
        self.choices = choices or {}
        self.model = model
        self.usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=0)


class FakeChoice:
    def __init__(self, choice, probabilities, confidence):
        self.choice = choice
        self.probabilities = probabilities
        self.confidence = confidence


class _FakeClient:
    """system_one だけを持つ差し替え用クライアント (tests/test_jev_client.py と同じ形)。"""

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self._handler(state, questions)


class _ExplodingClient:
    """呼ばれたら即座に失敗する。0頭のときにリクエストが飛んでいないことの検出用。"""

    def system_one(self, state, questions, **kwargs):
        raise AssertionError("system_one が呼ばれた: リクエストを送ってはいけない")


def _axis_choices() -> dict[str, FakeChoice]:
    """5軸すべてを、各軸の実際の criteria キーで埋めた FakeChoice の辞書。

    確率のキーは軸ごとに違う（relative_ability は superior/comparable/inferior/
    insufficient、transport_concern は concern/no_evidence/insufficient）。
    run_layer2 は「各軸の確率キーが AxisSpec の全選択肢を覆う」ことを検査する
    ので（fix5 決定2）、偽の応答も本物と同じキー集合にする——ここを手書きの
    固定キーにすると、軸を足したり選択肢を変えたときにテストだけが通る。
    """
    return {
        name: FakeChoice(
            choice=next(iter(spec.criteria)),
            probabilities=dict.fromkeys(spec.criteria, 1.0 / len(spec.criteria)),
            confidence=0.7,
        )
        for name, spec in AXIS_SPECS.items()
    }


# --- 送った物の記録 -------------------------------------------------------------
#
# 「あとで再生成すればよい」では足りない。再生成は**いまのコードが送るとしたら
# どうなるか**であって、あのとき送った物ではない。文言も特徴量も版が上がる
# （実際 PROMPT_VERSION は3世代目）ので、答えの隣に送った物を置いておかないと、
# 外れた予想を「材料が悪かったのか聞き方が悪かったのか」で切り分けられない
# （report2 §6 の #7）。

def test_layer1_keeps_the_very_state_object_it_sent(ctx_three_horses):
    """``sent_state`` が**クライアントに渡った物そのもの**であること。

    ``is`` で見るのが眼目——``build_race_state(ctx)`` を呼び直して等しい辞書を
    作る実装だと、この表明は通らない。2回組み立てる実装は、片方だけ直された
    ときに黙ってずれる。
    """
    winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    sent_state, _sent_questions, _kwargs = client.calls[0]
    assert result.sent_state is sent_state


def test_layer1_records_questions_as_plain_json(ctx_three_horses):
    """質問は ``Choice`` のまま持たない（JSON にならない）。本文と選択肢を残す。"""
    winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    _state, sent_questions, _kwargs = client.calls[0]
    assert set(result.sent_questions) == set(sent_questions)
    assert result.sent_questions["winner"]["instructions"] == (
        sent_questions["winner"].instructions)
    assert result.sent_questions["winner"]["criteria"] == sent_questions["winner"].criteria


def test_layer2_keeps_the_state_object_it_sent_for_each_horse(ctx_three_horses):
    client = _FakeClient(lambda state, questions: FakeResponse(choices=_axis_choices()))

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert set(result.sent_states) == {"01", "02", "03"}
    sent_by_umaban = {state["target_horse"]["bamei"]: state for state, _q, _k in client.calls}
    for umaban, state in result.sent_states.items():
        assert state is sent_by_umaban[state["target_horse"]["bamei"]]
        assert state["target_horse"]["bamei"] == (
            next(h.bamei for h in ctx_three_horses.horses if h.umaban == umaban))


def test_layer2_records_questions_once_not_per_horse(ctx_three_horses):
    """5軸の質問文は全頭で同じ。1頭ぶんだけ残す（18頭で18回書かない）。"""
    client = _FakeClient(lambda state, questions: FakeResponse(choices=_axis_choices()))

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert set(result.sent_questions) == set(AXIS_SPECS)
    for name, spec in AXIS_SPECS.items():
        assert result.sent_questions[name]["instructions"] == spec.instructions
        assert result.sent_questions[name]["criteria"] == spec.criteria


# --- 第1層 ---------------------------------------------------------------------

def test_layer1_returns_probabilities_for_every_horse(ctx_three_horses):
    winner = FakeChoice(
        choice="01",
        probabilities={"01": 0.5, "02": 0.3, "03": 0.2},
        confidence=0.6,
    )
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    assert set(result.win_probabilities) == {"01", "02", "03"}


def test_layer1_probabilities_sum_to_one(ctx_three_horses):
    winner = FakeChoice(
        choice="01",
        probabilities={"01": 0.5, "02": 0.3, "03": 0.2},
        confidence=0.6,
    )
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    assert sum(result.win_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_layer1_fills_missing_horse_with_zero(ctx_three_horses):
    """Jev が一部の馬 (03番) を返さなかった場合、0.0 で埋め、
    missing_probability_umaban に名指しで残し、合計は再正規化しない。

    0.0 には「勝てないと判断した」と「言及されなかった」の2つの意味があり、
    値だけでは区別できない。合計の不足は手がかりになるが、どの馬が欠けたかは
    そこからは復元できない。
    """
    winner = FakeChoice(
        choice="01",
        probabilities={"01": 0.5, "02": 0.3},  # 03番が欠けている
        confidence=0.6,
    )
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    assert result.win_probabilities["03"] == 0.0
    assert result.missing_probability_umaban == ("03",)
    # 再正規化していれば 1.0 になるはず。欠落がそのまま見える値であること。
    assert sum(result.win_probabilities.values()) < 1.0


def test_layer1_missing_probability_umaban_empty_when_every_horse_is_covered(ctx_three_horses):
    """全頭ぶん返ってきたなら空。"""
    winner = FakeChoice(
        choice="01",
        probabilities={"01": 0.5, "02": 0.3, "03": 0.2},
        confidence=0.6,
    )
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    assert result.missing_probability_umaban == ()


def test_layer1_missing_probability_umaban_follows_ctx_horses_order(ctx_three_horses):
    """欠けた馬が複数のとき、順序は ctx.horses の順 (01, 02, 03) を保つ。

    winner.probabilities に残ったキーの順や集合の反復順に引きずられると
    実行のたびに順序が変わりうる。再現性のため ctx.horses 側の順を使う。
    """
    winner = FakeChoice(
        choice="02",
        probabilities={"02": 1.0},  # 01番・03番の両方が欠けている
        confidence=0.6,
    )
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses)

    assert result.missing_probability_umaban == ("01", "03")


def test_layer1_pace_is_none_when_disabled(ctx_three_horses):
    winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
    client = _FakeClient(lambda state, questions: FakeResponse(choices={"winner": winner}))

    result = layers.run_layer1(client, ctx_three_horses, with_pace=False)

    assert result.pace_probabilities is None


def test_layer1_pace_keeps_all_three_bands(ctx_three_horses):
    winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
    pace = FakeChoice(
        choice="standard",
        probabilities={"fast": 0.2, "standard": 0.5, "slow": 0.3},
        confidence=0.4,
    )
    client = _FakeClient(
        lambda state, questions: FakeResponse(choices={"winner": winner, "pace": pace})
    )

    result = layers.run_layer1(client, ctx_three_horses, with_pace=True)

    assert result.pace_probabilities == {"fast": 0.2, "standard": 0.5, "slow": 0.3}


def test_layer1_sends_single_request(ctx_three_horses):
    winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
    pace = FakeChoice(
        choice="standard",
        probabilities={"fast": 0.2, "standard": 0.5, "slow": 0.3},
        confidence=0.4,
    )
    client = _FakeClient(
        lambda state, questions: FakeResponse(choices={"winner": winner, "pace": pace})
    )

    layers.run_layer1(client, ctx_three_horses, with_pace=True)

    assert len(client.calls) == 1


def test_layer1_wraps_api_error(ctx_three_horses):
    def fail(state, questions):
        raise ts.TypeSafeAPITimeoutError(10.0)

    client = _FakeClient(fail)

    with pytest.raises(JevRequestError):
        layers.run_layer1(client, ctx_three_horses)


# --- 第2層 ---------------------------------------------------------------------

def test_layer2_sends_one_request_per_horse(ctx_three_horses):
    client = _FakeClient(lambda state, questions: FakeResponse(choices=_axis_choices()))

    layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert len(client.calls) == 3


def test_layer2_bundles_all_axes_in_one_request(ctx_three_horses):
    client = _FakeClient(lambda state, questions: FakeResponse(choices=_axis_choices()))

    layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    for _state, questions, _kwargs in client.calls:
        assert set(questions) == set(AXIS_NAMES)


def test_layer2_collects_all_axes_per_horse(ctx_three_horses):
    client = _FakeClient(lambda state, questions: FakeResponse(choices=_axis_choices()))

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert set(result.answers["01"]) == set(AXIS_NAMES)
    axis_answer = result.answers["01"]["distance_fit"]
    spec = AXIS_SPECS["distance_fit"]
    assert axis_answer.choice == next(iter(spec.criteria))
    # 確率は軸の全選択肢を覆ったまま、そのままの形で運ばれる（数値化は第3層）。
    assert axis_answer.probabilities == dict.fromkeys(spec.criteria, 1.0 / len(spec.criteria))
    assert axis_answer.confidence == 0.7


def test_layer2_passes_pace_into_race_context(ctx_three_horses):
    client = _FakeClient(lambda state, questions: FakeResponse(choices=_axis_choices()))
    layer1 = make_layer1_result(
        pace_probabilities={"fast": 0.2, "standard": 0.5, "slow": 0.3}
    )

    layers.run_layer2(client, ctx_three_horses, layer1)

    for state, _questions, _kwargs in client.calls:
        assert state["race_context"]["pace_probabilities"] == {
            "fast": 0.2, "standard": 0.5, "slow": 0.3,
        }


def test_layer2_raises_on_partial_failure_by_default(ctx_three_horses):
    fail_bamei = {"ニバンウマ", "サンバンウマ"}

    def handler(state, questions):
        if state["target_horse"]["bamei"] in fail_bamei:
            raise ts.TypeSafeAPITimeoutError(10.0)
        return FakeResponse(choices=_axis_choices())

    client = _FakeClient(handler)

    with pytest.raises(layers.Layer2IncompleteError) as excinfo:
        layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert set(excinfo.value.failures) == {"02", "03"}
    message = str(excinfo.value)
    assert "02" in message
    assert "03" in message


def test_layer2_allows_partial_when_requested(ctx_three_horses):
    fail_bamei = {"ニバンウマ", "サンバンウマ"}

    def handler(state, questions):
        if state["target_horse"]["bamei"] in fail_bamei:
            raise ts.TypeSafeAPITimeoutError(10.0)
        return FakeResponse(choices=_axis_choices())

    client = _FakeClient(handler)

    result = layers.run_layer2(
        client, ctx_three_horses, make_layer1_result(), allow_partial=True
    )

    assert result.complete is False
    assert set(result.failures) == {"02", "03"}
    assert set(result.answers) == {"01"}


def test_layer2_sums_input_tokens(ctx_three_horses):
    tokens_by_bamei = {"イチバンウマ": 100, "ニバンウマ": 150, "サンバンウマ": None}

    def handler(state, questions):
        bamei = state["target_horse"]["bamei"]
        return FakeResponse(choices=_axis_choices(), input_tokens=tokens_by_bamei[bamei])

    client = _FakeClient(handler)

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert result.input_tokens == 250


def test_layer2_empty_when_no_horses(ctx_no_horses):
    client = _ExplodingClient()

    result = layers.run_layer2(client, ctx_no_horses, make_layer1_result())

    assert result.answers == {}
    assert result.failures == {}
    assert result.input_tokens == 0
    assert result.complete is True


# --- fix5 決定2: 応答の完全性 ---------------------------------------------------

def _drop_axis_choices(axis: str) -> dict[str, FakeChoice]:
    choices = _axis_choices()
    del choices[axis]
    return choices


def _drop_choice_key(axis: str, key: str) -> dict[str, FakeChoice]:
    choices = _axis_choices()
    original = choices[axis]
    choices[axis] = FakeChoice(
        choice=original.choice,
        probabilities={k: v for k, v in original.probabilities.items() if k != key},
        confidence=original.confidence,
    )
    return choices


def test_layer2_treats_a_missing_axis_as_a_failure(ctx_three_horses):
    """1軸が欠けた応答は、その馬の失敗として failures に入る（0.0 で埋めない）。"""
    client = _FakeClient(
        lambda state, questions: FakeResponse(choices=_drop_axis_choices("current_form")))

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result(),
                               allow_partial=True)

    assert set(result.failures) == {"01", "02", "03"}
    assert result.answers == {}
    assert "current_form" in result.failures["01"]


def test_layer2_treats_a_missing_choice_key_as_a_failure(ctx_three_horses):
    """軸の選択肢が1つ欠けた応答も失敗として扱う。"""
    client = _FakeClient(
        lambda state, questions: FakeResponse(
            choices=_drop_choice_key("distance_fit", "negative")))

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result(),
                               allow_partial=True)

    assert set(result.failures) == {"01", "02", "03"}
    assert "negative" in result.failures["01"]


def test_layer2_incomplete_answer_raises_without_allow_partial(ctx_three_horses):
    client = _FakeClient(
        lambda state, questions: FakeResponse(choices=_drop_axis_choices("current_form")))

    with pytest.raises(layers.Layer2IncompleteError) as excinfo:
        layers.run_layer2(client, ctx_three_horses, make_layer1_result())

    assert set(excinfo.value.failures) == {"01", "02", "03"}


def test_layer2_incomplete_answer_fails_only_that_horse(ctx_three_horses):
    """不完全なのが1頭だけなら、その馬だけが落ちて残りは答えを持つ。"""
    def handler(state, questions):
        if state["target_horse"]["bamei"] == "ニバンウマ":
            return FakeResponse(choices=_drop_axis_choices("relative_ability"))
        return FakeResponse(choices=_axis_choices())

    client = _FakeClient(handler)

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result(),
                               allow_partial=True)

    assert set(result.failures) == {"02"}
    assert set(result.answers) == {"01", "03"}


def test_layer2_counts_tokens_of_an_incomplete_answer(ctx_three_horses):
    """不完全な応答でもリクエストは成立して課金されているので、トークンは数える。"""
    client = _FakeClient(
        lambda state, questions: FakeResponse(
            choices=_drop_axis_choices("current_form"), input_tokens=40))

    result = layers.run_layer2(client, ctx_three_horses, make_layer1_result(),
                               allow_partial=True)

    assert result.input_tokens == 120


def test_layer1_raises_when_winner_is_missing(ctx_three_horses):
    """第1層の応答に winner が無ければ Layer1IncompleteError（生の KeyError にしない）。

    通信は成功しているので JevRequestError ではない——「API が落ちた」と
    報告させないために別のクラスにしてある。
    """
    client = _FakeClient(lambda state, questions: FakeResponse(choices={}))

    with pytest.raises(layers.Layer1IncompleteError):
        layers.run_layer1(client, ctx_three_horses)

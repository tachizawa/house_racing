"""keiba.predict.pipeline のテスト。

``predict`` は Task 6〜15 を配線するだけの層なので、ここでは「正しい順で
呼ばれるか」「途中で何が起きたら PredictionFailed になるか」「答えが記録の
どこに反映されるか」を見る。API には一切触れない――偽クライアント
(tests/test_layers.py と同じ形) と CSV フィクスチャ (tests/fixtures/csv_data.py)
だけで全体を回す。

``EvidenceBuilder`` は独立した関数なので、まず HorseFeatures / AxisAnswer を
手で組み立てて単体で検証し (テンプレートの正しさに集中できる)、そのあと
``predict`` 経由の配線をレース単位で見る。
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import typesafe_sdk as ts

from jev_client import JevConfigError
from keiba import config
from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.features.ability import AbilitySummary
from keiba.features.context import HorseFeatures
from keiba.features.derive import ConditionRecord, RecentForm, RunningStyle, Transport
from keiba.models import HorseEntry, RaceInput
from keiba.predict import integrate as ig
from keiba.predict.layers import AxisAnswer
from keiba.predict.pipeline import EvidenceBuilder, PredictionFailed, predict
from keiba.predict.questions import AXIS_SPECS, PROMPT_VERSION
from keiba.store.records import to_user_view
from tests.fixtures import csv_data as fx

# scripts/ は keiba パッケージの外にあり pytest.ini の pythonpath にも入って
# いないため、tests/test_update_data.py と同じ方法 (sys.path への追加) で
# import する。名前を predict_cli に別名しているのは、この関数 predict
# (keiba.predict.pipeline.predict) とバッティングさせないため。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import predict as predict_cli  # noqa: E402

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


def open_repository(tmp_path, races=None, race_horses=None, horses=None, **kw) -> Repository:
    data_dir = fx.write_dataset(tmp_path, races or [], race_horses or [], horses or [], **kw)
    return Repository(DataStore.open_current(data_dir))


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
        # top_n の既定は config.DEFAULT_TOP_N (3) だが、テストでは頭数が
        # それより少ないことが多い。既定のまま validate.check に引っかかる
        # (top_n が出走頭数を超える) のを避けるため、頭数に合わせる。
        top_n=len(horses),
    )
    defaults.update(kw)
    return RaceInput(**defaults)


def make_horse_features(*, umaban="01", by_condition: dict | None = None) -> HorseFeatures:
    """EvidenceBuilder のテスト用。by_condition だけを差し替えられる、値その
    ものは重要でない最小の HorseFeatures。5区分すべてを既定で starts=0 にして
    おき、テストが見たいキーだけ上書きする (condition_records が実際に返す
    「5区分すべて常に存在する」形を模す)。
    """
    ability = AbilitySummary(
        recent_margin_sec_wavg=1.0, runs_used=3, recent_class_level_wavg=3.0,
        best_class_level=4, period_from=None, period_to=None,
    )
    recent = RecentForm(days_since_last_run=20, distance_change_m=0, futan_change_kg=0.0,
                         class_level_change=0, last_run=None)
    style = RunningStyle(corner_pos_mean=0.4, corner_pos_stdev=None,
                          corner_pos_starts=3, corner_source="jyuni_4c")
    transport_features = Transport(
        base="美浦", estimated_transport_km=100, stay_status="unknown",
        past_long_transport_starts=0, past_long_transport_zogen_mean=None,
        past_long_transport_zogen_min=None,
    )
    default_by_condition = {
        key: ConditionRecord(starts=0, wins=0, top3=0, margin_sec_mean=None,
                              period_from=None, period_to=None)
        for key in ("exact_distance", "same_distance", "same_venue",
                    "same_track_kind", "same_baba")
    }
    if by_condition:
        default_by_condition.update(by_condition)
    return HorseFeatures(
        umaban=umaban, bamei="テストウマ", ketto_num="0000000001",
        entry=make_entry(umaban=umaban), ability=ability, field_rank=1.0,
        recent=recent, by_condition=default_by_condition, running_style=style,
        transport=transport_features, transport_base_source="entry",
        missing=(), notes=(),
    )


def make_axis_answer(choice: str, probabilities: dict | None = None, *,
                      confidence: float = 0.8) -> AxisAnswer:
    probabilities = probabilities or {choice: 1.0}
    return AxisAnswer(choice=choice, probabilities=probabilities, confidence=confidence)


#: 軸ごとの「中立」相当の choice。AXIS_SPECS.criteria の辞書順は先頭が
#: 「良い」側 (positive/superior/concern) なので、`next(iter(...))` で既定を
#: 選ぶと意図せず全軸が「良い」判定になってしまう。テストの既定はここで
#: 明示した中立の choice にする (transport_concern に本当の中立は無いので
#: 「懸念を示す材料がない」no_evidence を使う)。
_NEUTRAL_CHOICE: dict[str, str] = {
    "distance_fit": "neutral",
    "course_going_fit": "neutral",
    "relative_ability": "comparable",
    "current_form": "neutral",
    "transport_concern": "no_evidence",
}


def _uniform_answers(overrides: dict[str, str] | None = None) -> dict[str, AxisAnswer]:
    """5軸すべてに答えを持つ辞書。overrides で軸ごとに choice だけ差し替える。"""
    overrides = overrides or {}
    answers = {}
    for name in AXIS_SPECS:
        choice = overrides.get(name, _NEUTRAL_CHOICE[name])
        answers[name] = make_axis_answer(choice)
    return answers


def make_weights(*, weights_id: str = "test-weights", overrides: dict | None = None,
                  calibration: dict | None = None) -> ig.Weights:
    """全27特徴に対応した、そのまま predict() に通せる最小の Weights。

    logit_layer1 の重みだけ 1.0、残りは 0.0 にしてある――model_score が
    layer1 の確率の単調変換 (logit) だけで決まるようにし、model のランキング
    が何で決まっているか読みやすくするため。
    """
    names = ig.feature_names(with_layer2=True)
    standardization = {
        name: (0.0, 1.0)
        for name in names
        if name != "logit_layer1" and not name.endswith("_missing")
        and not name.endswith("_insufficient")
    }
    weights = dict.fromkeys(names, 0.0)
    weights["logit_layer1"] = 1.0
    if overrides:
        weights.update(overrides)
    return ig.Weights(
        weights_id=weights_id,
        feature_names=names,
        weights=weights,
        standardization=standardization,
        calibration=calibration,
        versions={
            "feature_version": config.FEATURE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "integration_version": config.INTEGRATION_VERSION,
        },
        lambda_=0.0,
        dropped_features=(),
        splits={},
        fitted_at="2026-09-22T02:00:00+00:00",
        dev_metrics={"dev_log_likelihood_per_race": -1.0},
    )


def make_weights_without_layer2(*, weights_id: str = "baseline-only-weights") -> ig.Weights:
    """第2層抜き (17特徴: logit_layer1 + Python8 + missing8) で学習した想定の
    Weights。with_layer2=False でも使える組み合わせ (concern 3) の確認用。
    """
    names = ig.feature_names(with_layer2=False)
    standardization = {
        name: (0.0, 1.0)
        for name in names
        if name != "logit_layer1" and not name.endswith("_missing")
    }
    weights = dict.fromkeys(names, 0.0)
    weights["logit_layer1"] = 1.0
    return ig.Weights(
        weights_id=weights_id,
        feature_names=names,
        weights=weights,
        standardization=standardization,
        calibration=None,
        versions={
            "feature_version": config.FEATURE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "integration_version": config.INTEGRATION_VERSION,
        },
        lambda_=0.0,
        dropped_features=(),
        splits={},
        fitted_at="2026-09-22T02:00:00+00:00",
        dev_metrics={"dev_log_likelihood_per_race": -1.0},
    )


# --- 偽クライアント (tests/test_layers.py と同じ形) ---------------------------

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
    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self._handler(state, questions)


class _ExplodingClient:
    """呼ばれたら即座に失敗する。早期失敗パスで API に触っていないことの検出用。"""

    def system_one(self, state, questions, **kwargs):
        raise AssertionError("system_one が呼ばれた: この経路でリクエストを送ってはいけない")


class _RaisingClient:
    """system_one を呼ぶと、与えられた例外をそのまま送出する (fix1)。

    第1層 (winner) が最初に呼ばれるので、渡した例外を第1層の失敗として
    再現できる。
    """

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def system_one(self, state, questions, **kwargs):
        raise self._error


def _axis_choices() -> dict[str, FakeChoice]:
    """5軸すべてを、各軸の実際の criteria キーで埋めた FakeChoice の辞書。"""
    return {
        name: FakeChoice(
            choice=next(iter(spec.criteria)),
            probabilities=dict.fromkeys(spec.criteria, 1.0 / len(spec.criteria)),
            confidence=0.7,
        )
        for name, spec in AXIS_SPECS.items()
    }


def _make_handler(winner_probabilities, *, fail_bamei=None,
                   layer1_tokens=100, layer2_tokens=50):
    """winner (第1層) と target_horse (第2層) の質問を見分けて応答する。"""
    fail_bamei = fail_bamei or set()

    def handler(state, questions):
        if "winner" in questions:
            winner_choice = max(winner_probabilities, key=winner_probabilities.get)
            winner = FakeChoice(choice=winner_choice,
                                 probabilities=winner_probabilities, confidence=0.6)
            choices = {"winner": winner}
            if "pace" in questions:
                choices["pace"] = FakeChoice(
                    choice="standard",
                    probabilities={"fast": 0.3, "standard": 0.4, "slow": 0.3},
                    confidence=0.5,
                )
            return FakeResponse(choices=choices, input_tokens=layer1_tokens)
        bamei = state["target_horse"]["bamei"]
        if bamei in fail_bamei:
            raise ts.TypeSafeAPITimeoutError(10.0)
        return FakeResponse(choices=_axis_choices(), input_tokens=layer2_tokens)

    return handler


# --- EvidenceBuilder ----------------------------------------------------------

def test_evidence_includes_condition_counts():
    """positives の文言に出走数が入る（例「同距離 8戦3勝」）。"""
    hf = make_horse_features(by_condition={
        "exact_distance": ConditionRecord(starts=8, wins=3, top3=5, margin_sec_mean=0.5,
                                           period_from=None, period_to=None),
    })
    answers = _uniform_answers({"distance_fit": "positive"})

    positives, _negatives, _insufficient = EvidenceBuilder.build(hf, answers)

    assert "同距離 8戦3勝(3着内5)" in positives


def test_evidence_marks_insufficient_axes():
    """最尤の答えが「判断材料不足」だった軸は insufficient に入る。"""
    hf = make_horse_features(by_condition={
        "exact_distance": ConditionRecord(starts=5, wins=1, top3=2, margin_sec_mean=0.5,
                                           period_from=None, period_to=None),
        "same_distance": ConditionRecord(starts=5, wins=1, top3=2, margin_sec_mean=0.5,
                                          period_from=None, period_to=None),
    })
    answers = _uniform_answers({"distance_fit": "insufficient", "relative_ability": "insufficient"})

    _positives, _negatives, insufficient = EvidenceBuilder.build(hf, answers)

    assert "距離適性: 判断材料不足" in insufficient
    assert "相対能力: 判断材料不足" in insufficient
    # starts > 0 でも choice が insufficient なら positives には積まない。
    assert "同距離 5戦1勝(3着内2)" not in insufficient  # 個別の条件フレーズは出ない


def test_evidence_marks_zero_start_aggregates():
    """starts == 0 の集計は、軸の分類（positive）に関わらず insufficient に入る。"""
    hf = make_horse_features()  # by_condition は全区分 starts=0 のまま
    answers = _uniform_answers({"distance_fit": "positive", "course_going_fit": "positive"})

    positives, _negatives, insufficient = EvidenceBuilder.build(hf, answers)

    assert "同距離 0戦0勝(3着内0)" in insufficient
    assert "同距離帯 0戦0勝(3着内0)" in insufficient
    assert "同競馬場 0戦0勝(3着内0)" in insufficient
    assert "同トラック 0戦0勝(3着内0)" in insufficient
    assert "同馬場状態 0戦0勝(3着内0)" in insufficient
    assert positives == ()  # starts=0 なので positive 分類でも何も積まれない


def test_evidence_is_built_from_classes_not_free_text():
    """positives の各要素が定型文（テンプレート由来）であること。

    同じ入力を2回渡しても常に同じ文字列になる (モデルの自由文ではなく、
    分類と集計値だけから決まるテンプレートで組まれていることの確認)。
    """
    hf = make_horse_features(by_condition={
        "exact_distance": ConditionRecord(starts=4, wins=2, top3=3, margin_sec_mean=0.2,
                                           period_from=None, period_to=None),
    })
    answers = _uniform_answers({"distance_fit": "positive", "relative_ability": "superior"})

    positives_1, _, _ = EvidenceBuilder.build(hf, answers)
    positives_2, _, _ = EvidenceBuilder.build(hf, answers)

    assert positives_1 == positives_2
    assert positives_1 == ("同距離 4戦2勝(3着内3)", "相対能力")
    for text in positives_1:
        assert isinstance(text, str) and text


def test_evidence_returns_empty_tuples_without_layer2_answers():
    """answers=None (with_layer2=False) では、軸レベルの証拠は作れないが、
    starts==0 の集計は Python 側だけで分かるので insufficient に残る。
    """
    hf = make_horse_features()

    positives, negatives, insufficient = EvidenceBuilder.build(hf, None)

    assert positives == ()
    assert negatives == ()
    assert len(insufficient) == 5  # 5区分すべて starts=0


# --- predict: 全頭が記録に入る -------------------------------------------------

def test_produces_record_with_every_horse(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ"),
               make_entry(umaban="03", bamei="サンバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.5, "02": 0.3, "03": 0.2}))

    record = predict(race_input, repo, client)

    assert {h.umaban for h in record.horses} == {"01", "02", "03"}


# --- ランキング ----------------------------------------------------------------

def test_baseline_ranking_present_without_weights(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.7, "02": 0.3}))

    record = predict(race_input, repo, client)

    assert all(h.rank_model is None for h in record.horses)
    assert all(h.model_win_prob is None for h in record.horses)
    by_umaban = {h.umaban: h for h in record.horses}
    assert by_umaban["01"].rank_baseline == 1
    assert by_umaban["02"].rank_baseline == 2


def test_model_ranking_present_with_weights(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02"), make_entry(umaban="03")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.3, "03": 0.1}))
    weights = make_weights()

    record = predict(race_input, repo, client, weights=weights)

    assert all(h.rank_model is not None for h in record.horses)
    assert all(h.model_win_prob is not None for h in record.horses)
    assert sorted(h.rank_model for h in record.horses) == [1, 2, 3]


def test_model_and_baseline_both_recorded(tmp_path):
    """spec §10.1: baseline と model の両方が同じ記録に入る。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))
    weights = make_weights()

    record = predict(race_input, repo, client, weights=weights)

    for h in record.horses:
        assert h.rank_baseline is not None
        assert h.rank_model is not None


def test_baseline_and_model_use_the_same_tie_break(tmp_path):
    """同確率のときの崩し方が baseline と model で食い違わない。

    どちらも predict/integrate.py の rank_by_score を通すことで、
    「馬番の数値昇順」という同じ規則を1箇所だけで決めている。将来どちらか
    だけが直されて乖離したら、このテストが落ちる。
    """
    repo = open_repository(tmp_path)
    # ゼロ詰めされていない馬番をわざと混ぜる。文字列順なら "1","10","11","2"
    # になってしまうところを、数値順の "1","2","10","11" で崩れることを見る。
    entries = [make_entry(umaban=str(n)) for n in (1, 2, 10, 11)]
    race_input = make_race_input(entries)
    probabilities = {str(n): 0.25 for n in (1, 2, 10, 11)}  # 全頭同確率
    client = _FakeClient(_make_handler(probabilities))
    weights = make_weights()  # 全頭 model_score も同値になる (logit は単調)

    record = predict(race_input, repo, client, weights=weights)

    expected_order = ["1", "2", "10", "11"]
    by_umaban = {h.umaban: h for h in record.horses}
    baseline_order = sorted(by_umaban, key=lambda u: by_umaban[u].rank_baseline)
    model_order = sorted(by_umaban, key=lambda u: by_umaban[u].rank_model)
    assert baseline_order == expected_order
    assert model_order == expected_order


# --- with_layer2 ----------------------------------------------------------------

def test_skips_layer2_when_disabled(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)

    def handler(state, questions):
        assert "winner" in questions, "第2層のリクエストが飛んではいけない"
        return FakeResponse(choices={
            "winner": FakeChoice(choice="01", probabilities={"01": 0.6, "02": 0.4},
                                  confidence=0.6),
        })

    client = _FakeClient(handler)

    record = predict(race_input, repo, client, with_layer2=False)

    assert len(client.calls) == 1
    assert record.status == "SUCCEEDED"
    assert all(h.layer2 == {} for h in record.horses)


# --- features (Task 18 supplement: 学習入力を組み立てるための生の特徴量) ---------

def test_record_features_match_the_integration_input(tmp_path):
    """with_layer2=True で第2層が完全なとき、HorsePrediction.features の
    キー集合は integrate.feature_names(with_layer2=True)（27列）と一致する。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client)

    expected = set(ig.feature_names(with_layer2=True))
    for h in record.horses:
        assert set(h.features.keys()) == expected


def test_record_features_are_narrower_without_layer2(tmp_path):
    """with_layer2=False の実行では、HorsePrediction.features のキー集合が
    integrate.feature_names(with_layer2=False)（17列）と一致する。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)

    def handler(state, questions):
        assert "winner" in questions, "第2層のリクエストが飛んではいけない"
        return FakeResponse(choices={
            "winner": FakeChoice(choice="01", probabilities={"01": 0.6, "02": 0.4},
                                  confidence=0.6),
        })

    client = _FakeClient(handler)

    record = predict(race_input, repo, client, with_layer2=False)

    expected = set(ig.feature_names(with_layer2=False))
    for h in record.horses:
        assert set(h.features.keys()) == expected


# --- 失敗パス ------------------------------------------------------------------

def test_fails_when_input_invalid(tmp_path):
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry], entry_count=5, top_n=9)
    client = _ExplodingClient()

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client)

    message = str(excinfo.value)
    assert "entry_count" in message
    assert "top_n" in message


def test_fails_when_name_match_ambiguous(tmp_path):
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="1"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(umaban="01", bamei="ドウメイウマ")
    race_input = make_race_input([entry])
    client = _ExplodingClient()

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client)

    message = str(excinfo.value)
    assert "01" in message
    assert "0000000001" in message
    assert "0000000002" in message


# --- 送った物の持ち出し (sent_prompts) ------------------------------------------

def test_sent_prompts_carries_what_the_client_received(tmp_path):
    """``sent_prompts`` に渡した辞書が、実際に送った state と質問で埋まる。

    ``predict`` は保存しない（呼び出し側の仕事）ので、送った物も「返す」
    のではなく呼び出し側の器に入れる。器を渡さない既存の呼び出しは
    そのまま動く。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))
    sent: dict = {}

    predict(race_input, repo, client, sent_prompts=sent)

    layer1_state = next(state for state, questions, _k in client.calls if "winner" in questions)
    assert sent["layer1"]["state"] is layer1_state
    assert set(sent["layer1"]["questions"]) == {"winner"}
    assert set(sent["layer2"]["states"]) == {"01", "02"}
    assert set(sent["layer2"]["questions"]) == set(AXIS_SPECS)


def test_sent_prompts_is_json_serializable(tmp_path):
    """そのまま JSON にできること（Choice や datetime を持ち帰らない）。"""
    repo = open_repository(tmp_path)
    race_input = make_race_input([make_entry(umaban="01")])
    client = _FakeClient(_make_handler({"01": 1.0}))
    sent: dict = {}

    predict(race_input, repo, client, sent_prompts=sent)

    json.dumps(sent, ensure_ascii=False)


def test_sent_prompts_layer2_is_none_when_layer2_is_off(tmp_path):
    """第2層を回していない実行で、空の第2層を作らない。"""
    repo = open_repository(tmp_path)
    race_input = make_race_input([make_entry(umaban="01")])
    client = _FakeClient(_make_handler({"01": 1.0}))
    sent: dict = {}

    predict(race_input, repo, client, with_layer2=False, sent_prompts=sent)

    assert sent["layer2"] is None


def test_predict_works_without_a_sent_prompts_container(tmp_path):
    """器を渡さない既存の呼び出しは、これまでどおり動く。"""
    repo = open_repository(tmp_path)
    race_input = make_race_input([make_entry(umaban="01")])
    client = _FakeClient(_make_handler({"01": 1.0}))

    record = predict(race_input, repo, client)

    assert record.status == "SUCCEEDED"


def test_layer1_request_failure_becomes_prediction_failed(tmp_path):
    """第1層の Jev 呼び出しが失敗しても PredictionFailed になる (fix1 Critical)。

    修正前は run_layer1 の呼び出しが try/except で包まれておらず、
    JevRequestError (レート制限・タイムアウト・5xx というもっとも普通に
    起きる失敗) がそのまま predict() の外へ漏れていた。scripts/predict.py
    は JevRequestError を捕まえないので、利用者に生のトレースバックが出る
    欠陥だった。
    """
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _RaisingClient(ts.TypeSafeAPITimeoutError(10.0))

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client)

    assert "第1層" in str(excinfo.value)


def test_layer1_config_error_is_not_swallowed(tmp_path):
    """JevConfigError (キー未設定など) は PredictionFailed に変換せず、
    そのまま外へ通す。scripts/predict.py がこれを専用に扱って案内を出す
    契約になっているため、ここで PredictionFailed に化けさせると
    案内が出なくなる。
    """
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _RaisingClient(JevConfigError("APIキーが設定されていません"))

    with pytest.raises(JevConfigError):
        predict(race_input, repo, client)


def test_layer2_request_failure_becomes_prediction_failed(tmp_path):
    """第2層で JevRequestError が起きても (allow_partial=False の既定経路で)
    Layer2IncompleteError 経由で PredictionFailed になる。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"ニバンウマ"}))

    with pytest.raises(PredictionFailed):
        predict(race_input, repo, client)


def test_fails_on_partial_layer2_by_default(tmp_path):
    """1頭失敗で PredictionFailed になり、記録は作らない (spec §12.7)。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"ニバンウマ"}))

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client)

    assert "02" in str(excinfo.value)


def test_allows_partial_and_marks_status(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"ニバンウマ"}))

    record = predict(race_input, repo, client, allow_partial=True)

    assert record.status == "PARTIAL"
    assert "02" in record.data_quality["layer2_failures"]
    by_umaban = {h.umaban: h for h in record.horses}
    assert by_umaban["02"].layer2 == {}
    assert by_umaban["01"].layer2 != {}


def test_partial_layer2_yields_no_model_ranking(tmp_path):
    """層2が欠けたら model の順位は出さない。baseline だけを返す。

    生き残った馬だけで softmax を取ると、欠けた馬の確率をその馬たちが
    吸収して過大になり、順位からは出走馬が消える。出走馬が消えた順位は
    順位の形をしているので、spec §12.7 の「不完全な評価を正常完了として
    返さない」に反する。層1は全頭を覆っているので baseline は完全。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"ニバンウマ"}))
    weights = make_weights()

    record = predict(race_input, repo, client, allow_partial=True, weights=weights)

    assert record.status == "PARTIAL"
    assert all(h.rank_model is None for h in record.horses)
    assert all(h.model_win_prob is None for h in record.horses)
    assert all(h.model_score is None for h in record.horses)
    assert all(h.rank_baseline is not None for h in record.horses)


def test_rejects_incompatible_weights(tmp_path):
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _ExplodingClient()
    weights = make_weights()
    weights = dataclasses.replace(weights, versions={**weights.versions, "prompt_version": "0"})

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client, weights=weights)

    assert "prompt_version" in str(excinfo.value)


def test_rejects_weights_needing_layer2_when_disabled(tmp_path):
    """with_layer2 特徴 (軸由来の10列) を必要とする重みを with_layer2=False で
    使おうとすると、足りない列を具体的に挙げて PredictionFailed になる。

    「weights と with_layer2 の組み合わせを丸ごと禁じる」のではなく、実際に
    その重みが要求する特徴と、この実行が作れる特徴の差分だけを見る
    (concern 3: 第2層抜きで学習した重みは with_layer2=False でも使えるべき)。
    """
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _ExplodingClient()
    weights = make_weights()  # with_layer2=True 相当の27列 (軸特徴込み)

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client, weights=weights, with_layer2=False)

    message = str(excinfo.value)
    assert "distance_fit_signed" in message


def test_accepts_weights_without_layer2_when_disabled(tmp_path):
    """第2層抜きで学習した重み (17列) は with_layer2=False でもそのまま使える。

    build_features (predict/integrate.py) は軸特徴量を無条件に読むため
    そのままでは使えない――pipeline.py 側が with_layer2 に応じて特徴量の
    組み立て方を切り替えていることの確認 (concern 3 の「今まで動いていた
    はずの正当な組み合わせ」)。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)

    def handler(state, questions):
        assert "winner" in questions, "第2層のリクエストが飛んではいけない"
        return FakeResponse(choices={
            "winner": FakeChoice(choice="01", probabilities={"01": 0.6, "02": 0.4},
                                  confidence=0.6),
        })

    client = _FakeClient(handler)
    weights = make_weights_without_layer2()

    record = predict(race_input, repo, client, weights=weights, with_layer2=False)

    assert len(client.calls) == 1  # 第2層は飛ばない
    assert all(h.rank_model is not None for h in record.horses)
    assert all(h.model_win_prob is not None for h in record.horses)


def test_rejects_baseline_weights_when_layer2_is_enabled(tmp_path):
    """層2なしで学習した重みを、層2ありの実行 (CLI の既定) に使わせない。

    fix round 2 で見つかった欠陥の再現テスト。required (17列) ⊆ available
    (27列) は成り立つので、片方向だけの部分集合チェックでは通ってしまう。
    しかし層2の10列を黙って無視することになり、しかも馬ごとに層2を呼んで
    課金したうえでその答えを捨てることになる。完全一致を要求することで、
    ここで PredictionFailed にする (以前は standardize() が
    WeightsIncompatibleError を投げ、それが predict() の外へ生で漏れて
    CLI がトレースバックで落ちていた)。
    """
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _ExplodingClient()
    weights = make_weights_without_layer2()  # with_layer2=False 相当の17列

    with pytest.raises(PredictionFailed, match="重みが知らない列") as excinfo:
        predict(race_input, repo, client, weights=weights, with_layer2=True)

    message = str(excinfo.value)
    assert "distance_fit_signed" in message


def test_feature_mismatch_failure_chains_weights_incompatible_error(tmp_path):
    """特徴量集合の不一致で上がる PredictionFailed の __cause__ が
    WeightsIncompatibleError であること (task-19-21 fix Critical 1)。

    修正前はこの raise に `from error` が無く __cause__ が None だった。
    keiba.evaluate.backtest.run() は
    isinstance(error.__cause__, WeightsIncompatibleError) でしか
    「全レースで同じように起きる設定の誤り」を横取りできないため、
    この枝が受け皿 (jev_failed) に落ち、「API が N 回落ちた」と誤報していた
    (レビュアー再現: 3レース・Jev 呼び出し0回で
    skip_breakdown = {'jev_failed': 3, ...})。
    """
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _ExplodingClient()
    weights = make_weights_without_layer2()  # with_layer2=False 相当の17列

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client, weights=weights, with_layer2=True)

    assert isinstance(excinfo.value.__cause__, ig.WeightsIncompatibleError)


# --- 較正の適用 (spec §11.2) ----------------------------------------------------
#
# 較正マップは fit_weights が作って重みファイルに入れていたが、本番経路に
# それを使う口が無く、calibrated_win_prob は常に None のままだった
# （report2 §6 の #8）。順位には使わない——isotonic は単調なので理屈の上では
# 順位不変だが、定義域外が端の値に張り付くと同値が量産されて順位が潰れる。
# 較正マップは節点10個・calib 16レースしかない。

def _identity_calibration() -> dict:
    """確率をそのまま返す較正マップ（[0,1] を線形に写す）。"""
    return {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}


def _two_horse_setup(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.8, "02": 0.2}))
    return repo, race_input, client


def test_calibration_fills_calibrated_win_prob(tmp_path):
    repo, race_input, client = _two_horse_setup(tmp_path)
    weights = make_weights(calibration=_identity_calibration())

    record = predict(race_input, repo, client, weights=weights)

    assert all(h.calibrated_win_prob is not None for h in record.horses)
    assert sum(h.calibrated_win_prob for h in record.horses) == pytest.approx(1.0)


def test_calibration_does_not_change_the_ranking(tmp_path):
    """順位は model_win_prob のまま。較正後の値で付け直さない。

    ``y`` を定数にすると較正後は全頭が同値になる。順位を較正後の確率で
    付けていれば、ここで順序が入れ替わるか潰れる。
    """
    repo, race_input, client = _two_horse_setup(tmp_path)
    flat = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.5, 0.5]}
    weights = make_weights(calibration=flat)

    record = predict(race_input, repo, client, weights=weights)

    by_umaban = {h.umaban: h for h in record.horses}
    assert by_umaban["01"].rank_model == 1
    assert by_umaban["02"].rank_model == 2
    assert by_umaban["01"].model_win_prob > by_umaban["02"].model_win_prob
    # None == None で通ってしまわないよう、値があることを先に押さえる。
    assert by_umaban["01"].calibrated_win_prob is not None
    assert by_umaban["01"].calibrated_win_prob == by_umaban["02"].calibrated_win_prob


def test_calibration_records_the_sum_before_renormalization(tmp_path):
    """spec §11.2: 再正規化の前後とずれを記録する。

    較正は馬ごとに独立に効くので、適用後の合計は 1.0 から離れうる。その
    ずれの大きさは「この較正が今回の顔ぶれに対してどれだけ信頼できるか」の
    兆候なので、割り算で隠さず残す。
    """
    repo, race_input, client = _two_horse_setup(tmp_path)
    doubling = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}
    weights = make_weights(calibration=doubling)

    record = predict(race_input, repo, client, weights=weights)

    calibration = record.data_quality["calibration"]
    assert calibration["applied"] is True
    assert calibration["sum_before_renormalization"] == pytest.approx(1.0, abs=1e-6)
    assert calibration["deviation_from_one"] == pytest.approx(0.0, abs=1e-6)


def test_calibration_names_the_horses_clipped_to_the_edge(tmp_path):
    """較正マップの定義域の外に出た馬を名指しで残す。

    ``numpy.interp`` は外挿せず端の値に張り付く。節点10個の較正マップでは
    ふつうに起きるが、値だけを見ても起きたことが分からない——張り付いた
    馬どうしは同じ確率になり、較正が「そう答えた」ように見える。
    """
    repo, race_input, client = _two_horse_setup(tmp_path)
    narrow = {"method": "isotonic", "x": [0.45, 0.55], "y": [0.45, 0.55]}
    weights = make_weights(calibration=narrow)

    record = predict(race_input, repo, client, weights=weights)

    calibration = record.data_quality["calibration"]
    assert calibration["domain"] == [0.45, 0.55]
    assert set(calibration["clipped_umaban"]) == {"01", "02"}


def test_calibration_says_why_it_was_not_applied(tmp_path):
    """較正マップを持たない重みでは、適用しなかったことと理由を残す。

    キーごと無いと「適用した結果 None だった」と読めてしまう。
    """
    repo, race_input, client = _two_horse_setup(tmp_path)

    record = predict(race_input, repo, client, weights=make_weights())

    calibration = record.data_quality["calibration"]
    assert calibration["applied"] is False
    assert "較正マップ" in calibration["reason"]
    assert all(h.calibrated_win_prob is None for h in record.horses)


def test_calibration_is_not_applied_without_weights(tmp_path):
    """baseline だけの実行（重み無し）には較正する対象の確率が無い。"""
    repo, race_input, client = _two_horse_setup(tmp_path)

    record = predict(race_input, repo, client)

    assert record.data_quality["calibration"]["applied"] is False
    assert all(h.calibrated_win_prob is None for h in record.horses)


def test_cli_reports_a_weights_mismatch_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """CLI は重みの不一致でトレースバックを出さず、メッセージを出して 1 を返す。

    scripts/predict.py の既定 (--layer2 が既定 ON) のまま、第2層抜きで学習した
    重み (17列) を --weights で渡す――利用者が --no-layer2 を付け忘れるだけで
    踏む、ごくふつうの組み合わせ。generate_client は本物の Jev クライアントを
    作らない偽物に差し替える (predict() は特徴量不整合を検出した時点
    (ステップ3) で失敗するので、system_one は一度も呼ばれないはずだが、
    万一呼ばれたら即座に分かるよう例外を投げる作りにしてある)。
    """
    data_dir = fx.write_dataset(tmp_path / "data", [], [], [])
    input_path = tmp_path / "race.json"
    input_path.write_text(json.dumps({
        "date": "2026-03-21",
        "jyo": "中山",
        "race_num": 11,
        "track_kind": "turf",
        "kyori_m": 2000,
        "entry_count": 1,
        "prediction_cutoff": "2026-03-20T20:00:00+09:00",
        "horses": [{"umaban": "01", "bamei": "テストウマ"}],
        "top_n": 1,
    }, ensure_ascii=False), encoding="utf-8")

    weights_dir = data_dir / "weights"
    weights_dir.mkdir(parents=True)
    make_weights_without_layer2().save(weights_dir / "mismatch.json")

    class _DummyClient:
        """本物の TypeSafeClient は一切作らない。with 文と system_one の
        形だけを持つ (predict() が特徴量不整合で先に失敗する想定なので、
        system_one が実際に呼ばれたらテストを失敗させる)。
        """

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def system_one(self, *args, **kwargs):
            raise AssertionError("system_one が呼ばれた: リクエストを送ってはいけない")

    monkeypatch.setattr(predict_cli, "create_client", lambda **kwargs: _DummyClient())

    exit_code = predict_cli.main([
        "--input", str(input_path),
        "--data-dir", str(data_dir),
        "--weights", "mismatch",
        "--no-save",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "重みが知らない列" in captured.out
    assert "Traceback" not in captured.err


# --- versions / usage / data_quality --------------------------------------------

def test_record_versions_include_weights_id(tmp_path):
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _FakeClient(_make_handler({"01": 1.0}))
    weights = make_weights(weights_id="w-abc")

    record = predict(race_input, repo, client, weights=weights)

    assert record.versions["weights_id"] == "w-abc"


def test_record_versions_omit_weights_id_without_weights(tmp_path):
    repo = open_repository(tmp_path)
    entry = make_entry(umaban="01")
    race_input = make_race_input([entry])
    client = _FakeClient(_make_handler({"01": 1.0}))

    record = predict(race_input, repo, client)

    assert record.versions["weights_id"] is None


def test_usage_sums_tokens_from_both_layers(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler(
        {"01": 0.6, "02": 0.4}, layer1_tokens=100, layer2_tokens=50))

    record = predict(race_input, repo, client)

    assert record.usage["layer1_input_tokens"] == 100
    assert record.usage["layer2_input_tokens"] == 100  # 2頭 x 50
    assert record.usage["total_input_tokens"] == 200


def test_user_view_shows_the_race_note_once(tmp_path):
    """baba_cd=None で予想したとき、to_user_view の「注記」に馬場状態の
    注記がちょうど1件だけ入り、どの馬の3束（プラス材料／マイナス材料／
    判断材料が不足している項目）にも「判定不能」という文言は現れない
    （fix3 決定1: レース単位の事実は馬ごとに繰り返さず、レース単位に1回だけ
    出す。修正1の end-to-end 確認）。

    horses.csv に両馬とも登録し、races/race_horses は与えない（過去走なし）
    ――未登録・所属推定などの馬単位の notes が混ざると、注記が1件だけという
    アサートが意図しない理由で崩れるため。
    """
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="イチバンウマ"),
        fx.make_horse(ketto_num="0000000002", bamei="ニバンウマ"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries, baba_cd=None)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client)
    view = to_user_view(record)

    assert len(view["注記"]) == 1
    assert "馬場状態" in view["注記"][0]
    for horse in view["予想"]:
        for key in ("プラス材料", "マイナス材料", "判断材料が不足している項目"):
            assert not any("判定不能" in item for item in horse[key])


def test_horse_prediction_carries_the_horses_notes(tmp_path):
    """predict() の結果で、注記のある馬の HorsePrediction.notes にその文が
    入っている（HorseFeatures.notes をそのまま運ぶ。fix4 決定2）。notes が
    無い馬（01番）は空タプルのまま。
    """
    horses = [fx.make_horse(ketto_num="0000000001", bamei="イチバンウマ")]
    repo = open_repository(tmp_path, horses=horses)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)  # "02" は未登録 → notes が付く
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client)

    by_umaban = {h.umaban: h for h in record.horses}
    assert by_umaban["02"].notes != ()
    assert "馬マスタにも取得済みの出走記録にも" in "".join(by_umaban["02"].notes)
    assert by_umaban["01"].notes == ()


def test_unregistered_horse_does_not_break_pipeline(tmp_path):
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, horses=horses)
    entries = [make_entry(umaban="01", bamei="トウロクウマ"),
               make_entry(umaban="02", bamei="ミトウロクウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client)

    assert {h.umaban for h in record.horses} == {"01", "02"}
    assert "02" in record.data_quality["missing"]
    # fix4: 馬単位の注記は data_quality["notes"]（レース単位専用）ではなく
    # data_quality["horse_notes"] に馬番ごとに入る。
    joined_notes = "".join(record.data_quality["horse_notes"]["02"])
    assert "馬マスタにも取得済みの出走記録にも" in joined_notes


# --- tie_group / tie_band_scale (fix2) -------------------------------------------

def test_tie_band_scale_is_not_measured_when_weights_are_given(tmp_path):
    """weights があり第2層も完全な実行では、同程度の判定そのものを行わない。

    順位は model_win_prob で決まるが、config.TIE_BAND はその尺度では測って
    いない（第1層の勝率で測った値）。測っていない尺度で組を作ると、測って
    いないばらつきを主張することになる（fix5 決定1）。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02"), make_entry(umaban="03")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.3, "03": 0.1}))
    weights = make_weights()

    record = predict(race_input, repo, client, weights=weights)

    assert record.data_quality["tie_band_scale"] == "not_measured"


def test_tie_groups_use_layer1_without_weights(tmp_path):
    """weights が無い (baseline のみ) 実行では、第1層の勝率の尺度で計算する。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client)

    assert record.data_quality["tie_band_scale"] == "layer1_win_prob"


def test_tie_groups_use_layer1_when_layer2_is_partial(tmp_path):
    """weights があっても第2層が部分失敗 (allow_partial=True) なら model の
    順位そのものを出さないので、同程度の組も第1層の尺度で計算する
    (「model_win_prob が無い」分岐そのもの)。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"ニバンウマ"}))
    weights = make_weights()

    record = predict(race_input, repo, client, allow_partial=True, weights=weights)

    assert record.status == "PARTIAL"
    assert record.data_quality["tie_band_scale"] == "layer1_win_prob"


def test_horses_within_the_band_share_a_group(tmp_path):
    """第1層の勝率を band 以内に作った偽クライアントで、該当馬の tie_group が
    同じになる (weights なし=第1層の尺度で判定される)。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02"), make_entry(umaban="03")]
    race_input = make_race_input(entries)
    close_gap = config.TIE_BAND / 2  # band 以内
    far_gap = config.TIE_BAND * 5    # band よりはっきり離す
    probabilities = {"01": 0.5, "02": 0.5 - close_gap, "03": 0.5 - far_gap}
    client = _FakeClient(_make_handler(probabilities))

    record = predict(race_input, repo, client)

    by_umaban = {h.umaban: h for h in record.horses}
    assert by_umaban["01"].tie_group == by_umaban["02"].tie_group
    assert by_umaban["03"].tie_group != by_umaban["01"].tie_group
    assert record.status == "SUCCEEDED"


# --- fix5 決定1: 測っていない尺度では同程度の組を作らない ------------------------

def test_no_tie_groups_when_weights_are_used(tmp_path):
    """重みがある実行では tie_group が全馬で一意になり、尺度は "not_measured"。

    config.TIE_BAND は第1層の勝率の run 間ばらつきとして実測した値で、
    model_score / model_win_prob の上では測っていない。測っていない尺度で
    組を作ると、スコアの幅が広い run では全馬が1組になり、利用者向けの
    「予想順位」を持つ馬が0頭になる（最終出力そのものが消える）。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02"), make_entry(umaban="03")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.34, "02": 0.33, "03": 0.33}))
    weights = make_weights()

    record = predict(race_input, repo, client, weights=weights)

    groups = [h.tie_group for h in record.horses]
    assert sorted(groups) == [1, 2, 3]
    assert record.data_quality["tie_band_scale"] == "not_measured"


def test_top_n_is_never_empty_with_weights(tmp_path):
    """18頭・model 尺度の差が TIE_BAND 未満でも、上位 n 頭は n 頭出る（要）。

    レビュアー再現: model_score の幅が狭い（λ が大きい／学習レースが少ない
    ときに十分ありうる）と、TIE_BAND=0.03 を model_win_prob に当てた結果
    全18頭が1組になり、to_user_view は18頭すべてを返して「予想順位」を持つ
    馬が0頭になった。第1層の勝率を一様（1/18 ≒ 0.056）にすると
    model_win_prob もほぼ一様になり、どの2頭の差も 0.03 未満になる。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban=f"{i:02d}") for i in range(1, 19)]
    race_input = make_race_input(entries, top_n=3)
    probabilities = {f"{i:02d}": 1.0 / 18 for i in range(1, 19)}
    client = _FakeClient(_make_handler(probabilities))
    weights = make_weights()

    record = predict(race_input, repo, client, weights=weights)
    view = to_user_view(record)

    ranked = [h for h in view["予想"] if "予想順位" in h]
    assert len(ranked) == 3, view["予想"]
    assert [h["予想順位"] for h in ranked] == [1, 2, 3]


# --- fix5 決定2: 第2層の応答の完全性 --------------------------------------------

def _incomplete_layer2_handler(winner_probabilities, *, drop_axis=None,
                                drop_choice=None, only_bamei=None):
    """第2層の応答から軸そのもの、または軸の選択肢1つを落とす偽クライアント。

    ``only_bamei`` を渡すとその馬の応答だけを欠けさせる（1頭だけ失敗させて
    残りは成立させる、部分失敗の経路を試すため）。
    """
    def handler(state, questions):
        if "winner" in questions:
            winner_choice = max(winner_probabilities, key=winner_probabilities.get)
            winner = FakeChoice(choice=winner_choice,
                                 probabilities=winner_probabilities, confidence=0.6)
            return FakeResponse(choices={"winner": winner})
        choices = _axis_choices()
        bamei = state["target_horse"]["bamei"]
        if only_bamei is None or bamei == only_bamei:
            if drop_axis is not None:
                del choices[drop_axis]
            if drop_choice is not None:
                axis, key = drop_choice
                original = choices[axis]
                choices[axis] = FakeChoice(
                    choice=original.choice,
                    probabilities={k: v for k, v in original.probabilities.items()
                                    if k != key},
                    confidence=original.confidence,
                )
        return FakeResponse(choices=choices)

    return handler


def test_layer2_missing_axis_is_a_prediction_failure(tmp_path):
    """1軸が欠けた応答は PredictionFailed（生の KeyError を外に出さない）。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_incomplete_layer2_handler(
        {"01": 0.6, "02": 0.4}, drop_axis="current_form"))

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client)

    assert "current_form" in str(excinfo.value)


def test_layer2_missing_choice_key_is_a_prediction_failure(tmp_path):
    """軸の選択肢が1つ欠けた応答も PredictionFailed。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_incomplete_layer2_handler(
        {"01": 0.6, "02": 0.4}, drop_choice=("distance_fit", "negative")))

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client)

    assert "negative" in str(excinfo.value)


def test_layer2_incomplete_answer_fails_only_that_horse(tmp_path):
    """allow_partial=True なら不完全な応答の馬だけが失敗になる（0.0 で埋めない）。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_incomplete_layer2_handler(
        {"01": 0.6, "02": 0.4}, drop_axis="current_form", only_bamei="ニバンウマ"))

    record = predict(race_input, repo, client, allow_partial=True)

    assert record.status == "PARTIAL"
    assert set(record.data_quality["layer2_failures"]) == {"02"}
    by_umaban = {h.umaban: h for h in record.horses}
    # 0.0 埋めをしていないこと: 軸の特徴量そのものが記録に無い（17列にそろう）。
    assert "current_form_signed" not in by_umaban["01"].features
    assert "current_form_signed" not in by_umaban["02"].features


def test_layer1_missing_winner_is_a_prediction_failure(tmp_path):
    """第1層の応答に winner が無ければ PredictionFailed（生の KeyError にしない）。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(lambda state, questions: FakeResponse(choices={}))

    with pytest.raises(PredictionFailed) as excinfo:
        predict(race_input, repo, client, with_layer2=False)

    assert "winner" in str(excinfo.value)


# --- fix5 決定7: pace_definition が無いなら pace 質問を送らない -------------------

def test_cli_does_not_send_the_pace_question_without_a_pace_definition(
        tmp_path, monkeypatch, capsys):
    """--pace を付けても、層別化に必要なレース数が足りず pace_definition が
    None なら pace 質問を送らない（要）。

    送ると、プロンプトが state に無い pace_definition を名指しすることになり
    仕様 §8.2 に正面から反する。§8.2 自身が「取得初期は成立しない前提」と
    書いている状況なので、ふつうに踏む経路である。
    """
    data_dir = fx.write_dataset(tmp_path / "data", [], [], [])
    input_path = tmp_path / "race.json"
    input_path.write_text(json.dumps({
        "date": "2026-03-21",
        "jyo": "中山",
        "race_num": 11,
        "track_kind": "turf",
        "kyori_m": 2000,
        "entry_count": 1,
        "prediction_cutoff": "2026-03-20T20:00:00+09:00",
        "horses": [{"umaban": "01", "bamei": "テストウマ"}],
        "top_n": 1,
    }, ensure_ascii=False), encoding="utf-8")

    sent: list[dict] = []

    class _CapturingClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def system_one(self, state, questions, **kwargs):
            sent.append(questions)
            winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
            return FakeResponse(choices={"winner": winner})

    monkeypatch.setattr(predict_cli, "create_client", lambda **kwargs: _CapturingClient())

    exit_code = predict_cli.main([
        "--input", str(input_path),
        "--data-dir", str(data_dir),
        "--pace",
        "--no-layer2",
        "--no-save",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert sent, "第1層のリクエストが飛んでいない"
    assert all("pace" not in questions for questions in sent), sent
    assert "ペース質問は送りません" in captured.out


def _write_minimal_input(tmp_path) -> Path:
    input_path = tmp_path / "race.json"
    input_path.write_text(json.dumps({
        "date": "2026-03-21",
        "jyo": "中山",
        "race_num": 11,
        "track_kind": "turf",
        "kyori_m": 2000,
        "entry_count": 1,
        "prediction_cutoff": "2026-03-20T20:00:00+09:00",
        "horses": [{"umaban": "01", "bamei": "テストウマ"}],
        "top_n": 1,
    }, ensure_ascii=False), encoding="utf-8")
    return input_path


class _WinnerOnlyClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def system_one(self, state, questions, **kwargs):
        winner = FakeChoice(choice="01", probabilities={"01": 1.0}, confidence=0.6)
        return FakeResponse(choices={"winner": winner})


def test_cli_saves_the_prompts_it_sent(tmp_path, monkeypatch, capsys):
    """既定で、送った state と質問文が data/prompts/ に残る。

    次に課金する実行を後から検証できるようにするための欄（report2 §6 の #7）。
    予想記録と同じ record_id で引ける。
    """
    data_dir = fx.write_dataset(tmp_path / "data", [], [], [])
    monkeypatch.setattr(predict_cli, "create_client", lambda **kwargs: _WinnerOnlyClient())

    exit_code = predict_cli.main([
        "--input", str(_write_minimal_input(tmp_path)),
        "--data-dir", str(data_dir),
        "--no-layer2",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    saved = list((data_dir / "prompts").rglob("*.json"))
    assert len(saved) == 1, captured.out
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["layer1"]["state"]["race"]["jyo"] == "中山"
    assert "winner" in payload["layer1"]["questions"]
    assert payload["layer2"] is None


def test_cli_can_turn_off_saving_the_prompts(tmp_path, monkeypatch, capsys):
    data_dir = fx.write_dataset(tmp_path / "data", [], [], [])
    monkeypatch.setattr(predict_cli, "create_client", lambda **kwargs: _WinnerOnlyClient())

    exit_code = predict_cli.main([
        "--input", str(_write_minimal_input(tmp_path)),
        "--data-dir", str(data_dir),
        "--no-layer2",
        "--no-save-prompts",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert not (data_dir / "prompts").exists()


def test_cli_prints_that_ties_were_not_judged(capsys, tmp_path):
    """--weights を使った実行では、CLI も「同程度の判定を行っていない」旨を出す。

    ここは to_user_view の値をそのまま表示するだけ（同じ日本語文を2箇所に
    書かない）。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02"), make_entry(umaban="03")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.5, "02": 0.3, "03": 0.2}))
    record = predict(race_input, repo, client, weights=make_weights())

    predict_cli._print_view(to_user_view(record))

    captured = capsys.readouterr().out
    assert "モデル尺度のばらつきは測っていないため" in captured


# --- fix5 決定10: usage にリクエスト件数を残す（仕様 §13）-------------------------

def test_usage_counts_requests(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client)

    # 第1層1回 + 馬2頭ぶん。
    assert record.usage["layer1_requests"] == 1
    assert record.usage["layer2_requests"] == 2
    assert record.usage["requests"] == 3


def test_usage_counts_only_layer1_without_layer2(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client, with_layer2=False)

    assert record.usage["layer2_requests"] == 0
    assert record.usage["requests"] == 1


def test_usage_counts_failed_layer2_requests_too(tmp_path):
    """失敗した馬のリクエストも送っている（課金されている）ので数える。"""
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"ニバンウマ"}))

    record = predict(race_input, repo, client, allow_partial=True)

    assert record.usage["layer2_requests"] == 2
    assert record.usage["requests"] == 3


def test_cli_reports_an_old_weights_file_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """fitted_at / dev_metrics の無い（決定6より前の）重みファイルで、CLI は
    トレースバックではなくメッセージを出して 1 を返す。
    """
    data_dir = fx.write_dataset(tmp_path / "data", [], [], [])
    input_path = tmp_path / "race.json"
    input_path.write_text(json.dumps({
        "date": "2026-03-21",
        "jyo": "中山",
        "race_num": 11,
        "track_kind": "turf",
        "kyori_m": 2000,
        "entry_count": 1,
        "prediction_cutoff": "2026-03-20T20:00:00+09:00",
        "horses": [{"umaban": "01", "bamei": "テストウマ"}],
        "top_n": 1,
    }, ensure_ascii=False), encoding="utf-8")

    weights_dir = data_dir / "weights"
    weights_dir.mkdir(parents=True)
    weights_path = weights_dir / "old.json"
    make_weights().save(weights_path)
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    del payload["fitted_at"]
    del payload["dev_metrics"]
    weights_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(predict_cli, "create_client",
                         lambda **kwargs: _ExplodingClient())

    exit_code = predict_cli.main([
        "--input", str(input_path),
        "--data-dir", str(data_dir),
        "--weights", "old",
        "--no-save",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "学習し直してください" in captured.out
    assert "Traceback" not in captured.err


# --- fix6 ⑤-1: pace の規則は predict() 自身が守る --------------------------------

def test_predict_does_not_send_the_pace_question_without_a_pace_definition(tmp_path):
    """``with_pace=True`` と ``pace_definition=None`` でも pace 質問を送らない。

    規則が CLI 側にしか無いと、``predict()`` を直接呼ぶ経路（将来のバックテスト
    など）が state に無いキーを名指しするプロンプトを送れてしまう。例外にならず
    「もっともらしい出力」として出る種類の欠陥なので、層の境界で検査する
    （第2層で直したのと同じ形）。
    """
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))

    record = predict(race_input, repo, client, with_layer2=False,
                      with_pace=True, pace_definition=None)

    layer1_questions = [q for _state, q, _kw in client.calls if "winner" in q]
    assert layer1_questions, "第1層のリクエストが飛んでいない"
    assert all("pace" not in questions for questions in layer1_questions)
    # 送らなかったことと理由を記録に残す（黙って落とすと「pace を要求しなかった
    # 実行」と区別できない）。
    assert any("ペース定義が作れなかった" in note
                for note in record.data_quality["notes"])
    assert record.layer1["pace_probabilities"] is None


def test_predict_still_sends_the_pace_question_with_a_definition(tmp_path):
    repo = open_repository(tmp_path)
    entries = [make_entry(umaban="01"), make_entry(umaban="02")]
    race_input = make_race_input(entries)
    client = _FakeClient(_make_handler({"01": 0.6, "02": 0.4}))
    pace_definition = {"fast_max": 33.0, "standard_max": 36.0,
                        "stratum": {"track_kind": "turf"}, "race_count": 40,
                        "note": "テスト用の定義。"}

    record = predict(race_input, repo, client, with_layer2=False,
                      with_pace=True, pace_definition=pace_definition)

    layer1_questions = [q for _state, q, _kw in client.calls if "winner" in q]
    assert any("pace" in questions for questions in layer1_questions)
    assert not any("ペース定義が作れなかった" in note
                    for note in record.data_quality["notes"])

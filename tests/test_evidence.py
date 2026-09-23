"""keiba.predict.evidence のテスト。

``EvidenceBuilder`` は独立した関数なので、HorseFeatures / AxisAnswer を手で
組み立てて単体で確かめる（tests/test_pipeline.py の EvidenceBuilder テストと
同じやり方。テンプレート組み立てのロジックに集中できる）。

fix2/fix3: same_baba が「定義できない」（``inapplicable_reason`` が None
でない）ときの扱いを確認する。fix3 の決定で、この分岐は3つの束の
どれにも入らず ``continue`` するだけになった（レース単位の事実を馬ごとに
繰り返さないため）。守るべき不変条件は、この分岐を answer による分岐より
先に置くことだけ——``record.starts == 0`` との前後は結果を変えない
（どちらも continue するだけのガードなので）。ガードを answer 側の分岐より
下に動かすと、``starts`` が ``None`` の inapplicable レコードが
``_condition_phrase`` まで渡って ``ValueError`` を投げ、予想全体が落ちる。
"""

from __future__ import annotations

import pytest

from keiba.features.ability import AbilitySummary
from keiba.features.context import HorseFeatures
from keiba.features.derive import (
    BABA_UNKNOWN_REASON,
    ConditionRecord,
    RecentForm,
    RunningStyle,
    Transport,
)
from keiba.models import HorseEntry
from keiba.predict.evidence import EvidenceBuilder, _condition_phrase
from keiba.predict.layers import AxisAnswer

_INAPPLICABLE_SAME_BABA = ConditionRecord(
    starts=None, wins=None, top3=None, margin_sec_mean=None,
    period_from=None, period_to=None,
    inapplicable_reason=BABA_UNKNOWN_REASON,
)


def make_entry(*, umaban="01", bamei="テストウマ", **kw) -> HorseEntry:
    return HorseEntry(umaban=umaban, bamei=bamei, **kw)


def make_horse_features(*, umaban="01", by_condition: dict | None = None) -> HorseFeatures:
    """EvidenceBuilder のテスト用。by_condition だけを差し替えられる、値その
    ものは重要でない最小の HorseFeatures。5区分すべてを既定で starts=0 にして
    おき、テストが見たいキーだけ上書きする（condition_records が実際に返す
    「5区分すべて常に存在する」形を模す。tests/test_pipeline.py と同じ構成）。
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
                      confidence: float = 0.9) -> AxisAnswer:
    probabilities = probabilities or {choice: 1.0}
    return AxisAnswer(choice=choice, probabilities=probabilities, confidence=confidence)


def test_inapplicable_condition_appears_in_no_bundle():
    """馬場状態が未入力で same_baba が定義できない (inapplicable_reason が
    None でない) とき、positives / negatives / insufficient のどの束にも
    その記録は現れない（fix3 決定1）。

    これはレース単位の事実（今回の馬場状態が未入力）であって、馬についての
    事実ではない。ここで文言化すると同じ一文が出走馬の数だけ繰り返されて
    しまう。レース単位の事実は features/context.py が ctx.notes に1回だけ
    出す。情報は失われない——軸レベルのループが
    「コース・馬場適性: 判断材料不足」を出すので、利用者は軸が判断材料
    不足であることを知れる。
    """
    hf = make_horse_features(by_condition={"same_baba": _INAPPLICABLE_SAME_BABA})
    answers = {"course_going_fit": make_axis_answer("positive")}

    positives, negatives, insufficient = EvidenceBuilder.build(hf, answers)

    for bundle in (positives, negatives, insufficient):
        assert not any("同馬場状態" in item for item in bundle)
        assert not any(BABA_UNKNOWN_REASON in item for item in bundle)


def test_inapplicable_condition_never_shows_a_count():
    """same_baba が定義できない場合でも、他の区分に本物の0を混ぜなければ、
    3つの束のどの要素にも「0戦」という部分文字列が現れない
    （inapplicable は件数の形を持たないため）。"""
    hf = make_horse_features(by_condition={
        "exact_distance": ConditionRecord(starts=5, wins=2, top3=3, margin_sec_mean=0.4,
                                           period_from=None, period_to=None),
        "same_distance": ConditionRecord(starts=6, wins=1, top3=2, margin_sec_mean=0.4,
                                          period_from=None, period_to=None),
        "same_venue": ConditionRecord(starts=7, wins=2, top3=4, margin_sec_mean=0.3,
                                       period_from=None, period_to=None),
        "same_track_kind": ConditionRecord(starts=8, wins=3, top3=5, margin_sec_mean=0.3,
                                            period_from=None, period_to=None),
        "same_baba": _INAPPLICABLE_SAME_BABA,
    })
    answers = {
        "distance_fit": make_axis_answer("positive"),
        "course_going_fit": make_axis_answer("negative"),
        "relative_ability": make_axis_answer("superior"),
        "current_form": make_axis_answer("negative"),
        "transport_concern": make_axis_answer("concern"),
    }

    positives, negatives, insufficient = EvidenceBuilder.build(hf, answers)

    for bundle in (positives, negatives, insufficient):
        for text in bundle:
            assert "0戦" not in text


def test_condition_phrase_refuses_an_inapplicable_record():
    """_condition_phrase は inapplicable なレコードを渡されると ValueError。
    件数の無いものに黙って「None戦」の形を与えないための防御。"""
    with pytest.raises(ValueError):
        _condition_phrase("same_baba", _INAPPLICABLE_SAME_BABA)


def test_real_zero_starts_still_shows_the_count():
    """本物の0（inapplicable_reason=None の starts=0）は今まで通り件数の
    文言で出す。今回区別するのは「条件が決まっていない」ケースだけ。"""
    record = ConditionRecord(starts=0, wins=0, top3=0, margin_sec_mean=None,
                              period_from=None, period_to=None)

    assert _condition_phrase("same_baba", record) == "同馬場状態 0戦0勝(3着内0)"


def test_real_zero_starts_is_still_classified_as_insufficient():
    """本物の0（inapplicable_reason=None, starts=0）は、分類ループを通しても
    insufficient に「0戦0勝(3着内0)」として入る。

    fix3 で inapplicable_reason が None でない記録を continue するだけに
    変えても、「本物の0」との区別（starts == 0 のガード）は別のまま保たれる
    ことを、_condition_phrase を直接呼ぶのではなく
    EvidenceBuilder.build 経由（分類ループ）で確かめる。_condition_phrase を
    直接呼ぶだけでは、分類ループのガードの順序や continue の扱いを変異させ
    ても検出できない。
    """
    hf = make_horse_features()  # by_condition は5区分すべて starts=0 のまま
    answers = {"course_going_fit": make_axis_answer("positive")}

    _positives, _negatives, insufficient = EvidenceBuilder.build(hf, answers)

    assert "同馬場状態 0戦0勝(3着内0)" in insufficient


def test_classification_does_not_raise_when_baba_unknown_and_answers_present():
    """馬場状態が未入力（same_baba が inapplicable）でも、layer2 の答えが
    揃っていれば分類ループは例外を投げない。

    fix3 で明文化した不変条件（inapplicable_reason のチェックは answer に
    よる分岐より先に置く）を守れているかの確認: このガードを answer 側の
    分岐より後に置くと、starts=None のレコードは ``starts == 0`` が False
    のまま answer 側の分岐まで落ち、``_condition_phrase`` が ValueError を
    投げて予想全体が止まる。ここでは course_going_fit の答えが実際に存在
    するケースで、それが起きないことを確かめる（表の他のテストは分類結果
    だけを見ており、この経路はカバーしていない）。
    """
    hf = make_horse_features(by_condition={"same_baba": _INAPPLICABLE_SAME_BABA})
    answers = {"course_going_fit": make_axis_answer("positive")}

    # 例外を投げずに完了することそのものが確認したい事実。
    EvidenceBuilder.build(hf, answers)

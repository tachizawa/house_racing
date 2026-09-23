"""keiba.predict.integrate のテスト。

第3層は Jev を呼ばない純粋な計算なので、ここでは本物の Repository も
system_one も経由しない。Layer1Result / Layer2Result / HorseFeatures を
手で組み立てて、logit・softmax・条件付きロジットの重み付け・標準化・
Weights の JSON 往復だけを見る。

とくに手厚く見るべきは softmax_by_race の「頭数に依存しない」性質
（CLAUDE.md §13。第1層は全頭で合計1.0、Noul は1頭ごとに0-1という
スケールの違いを吸収するのが softmax の役目）と、expand_missing が
「値が0」と「欠損」を区別できているか（0埋めだけでは区別できない、が
このプロジェクトが Task 3 以来ずっと避けてきた失敗）。
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from keiba import config
from keiba.features.ability import AbilitySummary
from keiba.features.context import HorseFeatures, PredictionContext
from keiba.features.derive import RecentForm, RunningStyle, Transport
from keiba.models import HorseEntry, RaceInput
from keiba.predict import integrate as ig
from keiba.predict import questions as q
from keiba.predict.layers import AxisAnswer, Layer1Result, Layer2Result

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


def make_horse_features(
    *,
    umaban="01",
    recent_margin_sec_wavg: float | None = 1.2,
    runs_used=4,
    recent_class_level_wavg: float | None = 3.0,
    class_level_change: float | None = 1,
    days_since_last_run: float | None = 20,
    distance_change_m: float | None = 100,
    corner_pos_mean: float | None = 0.4,
    estimated_transport_km: float | None = 250,
) -> HorseFeatures:
    """python_features が読む8項目だけを差し替えられる最小の HorseFeatures。

    build_context は経由しない (Repository を用意する必要がなく、python_features
    が実際に読むフィールドだけに絞れるため)。
    """
    ability = AbilitySummary(
        recent_margin_sec_wavg=recent_margin_sec_wavg,
        runs_used=runs_used,
        recent_class_level_wavg=recent_class_level_wavg,
        best_class_level=5,
        period_from=None,
        period_to=None,
    )
    recent = RecentForm(
        days_since_last_run=days_since_last_run,
        distance_change_m=distance_change_m,
        futan_change_kg=0.0,
        class_level_change=class_level_change,
        last_run=None,
    )
    style = RunningStyle(
        corner_pos_mean=corner_pos_mean,
        corner_pos_stdev=None,
        corner_pos_starts=3,
        corner_source="jyuni_4c",
    )
    transport_features = Transport(
        base="美浦",
        estimated_transport_km=estimated_transport_km,
        stay_status="unknown",
        past_long_transport_starts=0,
        past_long_transport_zogen_mean=None,
        past_long_transport_zogen_min=None,
    )
    return HorseFeatures(
        umaban=umaban,
        bamei="テストウマ",
        ketto_num="0000000001",
        entry=make_entry(umaban=umaban),
        ability=ability,
        field_rank=1.0,
        recent=recent,
        by_condition={},
        running_style=style,
        transport=transport_features,
        transport_base_source="entry",
        missing=(),
        notes=(),
    )


def _uniform_axis_answers() -> dict[str, AxisAnswer]:
    """5軸すべてにダミーの一様分布を持つ answers。

    axis_features は AXIS_SPECS の全軸を読むので、1軸だけのテストでも
    残り4軸ぶんは埋めておく必要がある。ここで一度に作って、個々のテストは
    _with_axis で見たい軸だけ差し替える。
    """
    answers: dict[str, AxisAnswer] = {}
    for name, spec in q.AXIS_SPECS.items():
        keys = list(spec.criteria)
        p = 1.0 / len(keys)
        answers[name] = AxisAnswer(
            choice=keys[0], probabilities=dict.fromkeys(keys, p), confidence=0.5)
    return answers


def _with_axis(axis_name: str, probabilities: dict[str, float], *,
               confidence: float = 0.9) -> dict[str, AxisAnswer]:
    answers = _uniform_axis_answers()
    choice = max(probabilities, key=probabilities.get)
    answers[axis_name] = AxisAnswer(
        choice=choice, probabilities=probabilities, confidence=confidence)
    return answers


def make_layer1(win_probabilities: dict[str, float], **kw) -> Layer1Result:
    defaults = dict(
        choice=next(iter(win_probabilities)),
        confidence=0.9,
        pace_probabilities=None,
        model="jev-test",
        input_tokens=0,
        missing_probability_umaban=(),
    )
    defaults.update(kw)
    return Layer1Result(win_probabilities=win_probabilities, **defaults)


def make_weights(feature_names, weights, **kw) -> ig.Weights:
    defaults = dict(
        weights_id="test-weights",
        standardization={},
        calibration=None,
        versions={
            "feature_version": config.FEATURE_VERSION,
            "prompt_version": q.PROMPT_VERSION,
            "integration_version": config.INTEGRATION_VERSION,
        },
        lambda_=0.0,
        dropped_features=(),
        splits={},
        # fix5 決定6: 学習時刻と dev 指標は必須の欄（既定値を持たせない）。
        fitted_at="2026-09-22T02:00:00+00:00",
        dev_metrics={},
    )
    defaults.update(kw)
    return ig.Weights(feature_names=tuple(feature_names), weights=dict(weights), **defaults)


# --- axis_features -----------------------------------------------------------

def test_axis_features_signed_is_positive_minus_negative():
    answers = _with_axis("distance_fit", {
        "positive": 0.6, "neutral": 0.2, "negative": 0.1, "insufficient": 0.1,
    })
    result = ig.axis_features(answers)
    assert result["distance_fit_signed"] == pytest.approx(0.5)


def test_axis_features_insufficient_is_kept_separately():
    answers = _with_axis("distance_fit", {
        "positive": 0.6, "neutral": 0.2, "negative": 0.1, "insufficient": 0.1,
    })
    result = ig.axis_features(answers)
    assert result["distance_fit_insufficient"] == pytest.approx(0.1)


def test_axis_features_relative_ability_uses_superior_inferior():
    answers = _with_axis("relative_ability", {
        "superior": 0.5, "comparable": 0.2, "inferior": 0.2, "insufficient": 0.1,
    })
    result = ig.axis_features(answers)
    assert result["relative_ability_signed"] == pytest.approx(0.3)


def test_axis_features_transport_uses_concern_directly():
    """negative_key が None の軸は差を取らず、positive_key の確率そのもの。"""
    answers = _with_axis("transport_concern", {
        "concern": 0.4, "no_evidence": 0.5, "insufficient": 0.1,
    })
    result = ig.axis_features(answers)
    assert result["transport_concern_signed"] == pytest.approx(0.4)


def test_axis_features_returns_ten_columns():
    result = ig.axis_features(_uniform_axis_answers())
    assert len(result) == 10


def test_axis_features_ignores_confidence():
    """spec §2.1: confidence はどの特徴にも使わない。"""
    probs = {"positive": 0.5, "neutral": 0.3, "negative": 0.1, "insufficient": 0.1}
    low_confidence = _with_axis("distance_fit", probs, confidence=0.1)
    high_confidence = _with_axis("distance_fit", probs, confidence=0.99)
    assert ig.axis_features(low_confidence) == ig.axis_features(high_confidence)


# --- python_features -----------------------------------------------------------

def test_python_features_keeps_none():
    hf = make_horse_features(recent_margin_sec_wavg=None)
    result = ig.python_features(hf)
    assert result["recent_margin_wavg"] is None


def test_python_features_maps_runs_used_to_recent_margin_runs_used():
    hf = make_horse_features(runs_used=7)
    result = ig.python_features(hf)
    assert result["recent_margin_runs_used"] == 7.0


# --- expand_missing --------------------------------------------------------------

def test_expand_missing_sets_indicator():
    result = ig.expand_missing({"a": None, "b": 2.0})
    assert result == {"a": 0.0, "a_missing": 1.0, "b": 2.0, "b_missing": 0.0}


def test_expand_missing_distinguishes_zero_from_missing():
    """値が本物の0.0の馬と、値が無い(None)の馬とで、展開後の特徴が違う
    ことを確認する。0埋めだけでは両者が同じ 0.0 になって区別できなくなる
    ——このプロジェクトが Task 3 以来避けてきた失敗そのもの。
    """
    zero_horse = make_horse_features(distance_change_m=0.0)
    missing_horse = make_horse_features(distance_change_m=None)

    zero_expanded = ig.expand_missing(ig.python_features(zero_horse))
    missing_expanded = ig.expand_missing(ig.python_features(missing_horse))

    assert zero_expanded["distance_change_m"] == 0.0
    assert zero_expanded["distance_change_m_missing"] == 0.0
    assert missing_expanded["distance_change_m"] == 0.0
    assert missing_expanded["distance_change_m_missing"] == 1.0
    assert zero_expanded != missing_expanded


# --- is_missing_indicator / python_feature_block (fix round 2 Minor 4) -----------------


def test_is_missing_indicator_matches_python_feature_names():
    for name in ig.PYTHON_FEATURE_NAMES:
        assert ig.is_missing_indicator(f"{name}_missing") is True


def test_is_missing_indicator_rejects_non_missing_names():
    assert ig.is_missing_indicator("logit_layer1") is False
    assert ig.is_missing_indicator("distance_fit_insufficient") is False
    for name in ig.PYTHON_FEATURE_NAMES:
        assert ig.is_missing_indicator(name) is False  # 生の値そのものは対象外


def test_python_feature_block_has_16_names():
    block = ig.python_feature_block()
    assert len(block) == 16
    for name in ig.PYTHON_FEATURE_NAMES:
        assert name in block
        assert f"{name}_missing" in block


def test_python_feature_block_matches_feature_names_layout():
    """feature_names(with_layer2=True) の末尾16列（python8+missing8）と
    python_feature_block() が同じ集合になる。"""
    names = ig.feature_names(with_layer2=True)
    assert set(names[11:]) == ig.python_feature_block()


# --- logit -----------------------------------------------------------------------

def test_logit_clips_at_zero_and_one():
    assert math.isfinite(ig.logit(0.0))
    assert math.isfinite(ig.logit(1.0))


def test_logit_is_monotonic():
    assert ig.logit(0.1) < ig.logit(0.5) < ig.logit(0.9)


def test_logit_of_half_is_zero():
    assert ig.logit(0.5) == pytest.approx(0.0, abs=1e-9)


# --- softmax_by_race ---------------------------------------------------------------

def test_softmax_sums_to_one_for_8_horses():
    scores = {str(i): float(i) * 0.3 for i in range(8)}
    result = ig.softmax_by_race(scores)
    assert sum(result.values()) == pytest.approx(1.0)


def test_softmax_sums_to_one_for_18_horses():
    scores = {str(i): float(i) * 0.3 for i in range(18)}
    result = ig.softmax_by_race(scores)
    assert sum(result.values()) == pytest.approx(1.0)


def test_softmax_handles_large_scores_without_overflow():
    scores = {"01": 1000.0, "02": 0.0, "03": -5.0}
    result = ig.softmax_by_race(scores)
    for value in result.values():
        assert math.isfinite(value)
    assert sum(result.values()) == pytest.approx(1.0)


def test_softmax_preserves_order():
    scores = {"01": 0.5, "02": 2.0, "03": -1.0}
    result = ig.softmax_by_race(scores)
    ranked_by_score = sorted(scores, key=lambda k: scores[k], reverse=True)
    ranked_by_prob = sorted(result, key=lambda k: result[k], reverse=True)
    assert ranked_by_score == ranked_by_prob


def test_softmax_relative_order_is_field_size_independent():
    """8頭と18頭で、同じ相対スコア差なら順位が変わらないことを確認する。

    第1層の確率スケールが頭数で変わる問題を softmax が吸収しているなら、
    同じ重み・同じ特徴値から作ったスコア差は頭数によらず同じ順位になるはず。
    """
    base_scores = {"01": 1.0, "02": 0.5, "03": -0.2}
    padded_scores = dict(base_scores)
    for i in range(4, 19):
        padded_scores[f"{i:02d}"] = -3.0  # 残りは大きく劣るスコアで埋める

    small = ig.softmax_by_race(base_scores)
    large = ig.softmax_by_race(padded_scores)
    order_small = sorted(("01", "02", "03"), key=lambda k: small[k], reverse=True)
    order_large = sorted(("01", "02", "03"), key=lambda k: large[k], reverse=True)
    assert order_small == order_large == ["01", "02", "03"]


# --- baseline_ranking -------------------------------------------------------------

def test_baseline_ranking_is_descending_probability():
    layer1 = make_layer1({"01": 0.2, "02": 0.5, "03": 0.3})
    assert ig.baseline_ranking(layer1) == ["02", "03", "01"]


def test_baseline_ranking_breaks_ties_by_umaban():
    layer1 = make_layer1({"03": 0.3, "01": 0.3, "02": 0.4})
    assert ig.baseline_ranking(layer1) == ["02", "01", "03"]


def test_baseline_ranking_breaks_ties_numerically_not_lexicographically():
    """ゼロ詰めされていない馬番でも数値順に並ぶ。

    文字列比較だと "10" と "11" が "2" より前に来る。利用者入力の馬番は
    ゼロ詰めが保証されていない（input/validate.py は正の整数として読めることだけ
    を確かめる）ので、ゼロ詰めの有無で順位が変わってしまう。同着が出るのは
    例外的な状況ではない――run_layer1 はモデルが返さなかった馬を 0.0 で埋める
    ので、11頭立てで1頭欠ければこの形になる。
    """
    probabilities = {str(i): 0.0 for i in range(1, 12)}
    probabilities["5"] = 0.9
    layer1 = make_layer1(probabilities)
    assert ig.baseline_ranking(layer1) == [
        "5", "1", "2", "3", "4", "6", "7", "8", "9", "10", "11",
    ]


# --- tie_groups (fix2) --------------------------------------------------------------
#
# キーは馬番なので、rank_by_score (tie_groups が内部で呼ぶ) の int(umaban) 変換に
# 通る数値文字列にする ("a"/"b" のような英字はここでは使えない――
# input/validate.py が保証するのは「正の整数として読めること」だけ)。
#
# config.TIE_BAND (実測: 実レース8件・各5回・101頭ぶんの run 間の幅の95%点) の
# 意味を確かめるテストなので、band には固定の 0.03 を渡す (config.TIE_BAND を
# そのまま使うと、値が将来測り直されたときにテストの意図――「95%点=0.03 の
# ときにこう動く」――が読めなくなる)。

def test_tie_groups_splits_where_the_gap_exceeds_the_band():
    values = {"1": 0.50, "2": 0.20, "3": 0.18}
    assert ig.tie_groups(values, band=0.03) == {"1": 1, "2": 2, "3": 2}


def test_tie_groups_does_not_chain_transitively():
    """隣り合う2頭の差だけで見ると 1-2(0.02)・2-3(0.02)・3-4(0.02) が鎖状に
    つながり、全員が同じ組になってしまう。だが「組の中の最大差」で見ると、
    1番と4番の差は 0.06 で band(0.03) のちょうど2倍あるので、1/2番の組と
    3/4番の組は別でなければならない。
    """
    values = {"1": 0.20, "2": 0.18, "3": 0.16, "4": 0.14}
    groups = ig.tie_groups(values, band=0.03)
    assert groups["1"] == groups["2"]
    assert groups["3"] == groups["4"]
    assert groups["1"] != groups["3"]


def test_tie_groups_puts_everything_in_one_group_when_all_are_close():
    values = {"1": 0.250, "2": 0.245, "3": 0.242, "4": 0.241}
    groups = ig.tie_groups(values, band=0.03)
    assert len(set(groups.values())) == 1


def test_tie_groups_gives_each_horse_its_own_group_when_all_are_far():
    values = {"1": 0.90, "2": 0.50, "3": 0.10, "4": 0.01}
    assert ig.tie_groups(values, band=0.03) == {"1": 1, "2": 2, "3": 3, "4": 4}


def test_tie_groups_orders_like_rank_by_score():
    """組の番号は rank_by_score と同じ順序に整合し、同値は馬番の数値順で
    (ゼロ詰めされていない "10" と "2" で確かめる)。
    """
    values = {"10": 0.3, "2": 0.3, "1": 0.1}
    order = ig.rank_by_score(values)
    groups = ig.tie_groups(values, band=0.03)
    ordered_group_numbers = [groups[u] for u in order]
    assert ordered_group_numbers == sorted(ordered_group_numbers)
    assert groups["2"] == groups["10"]  # 同値の2頭は同じ組
    assert order.index("2") < order.index("1")  # 数値順の確認 (rank_by_score 側)


def test_tie_groups_handles_a_single_horse():
    assert ig.tie_groups({"1": 0.5}, band=0.03) == {"1": 1}


def test_tie_groups_rejects_an_empty_mapping():
    with pytest.raises(ValueError):
        ig.tie_groups({}, band=0.03)


def test_tie_groups_accepts_a_band_of_zero():
    """band=0.0 は「差が無いものだけ同組」という実在の指定。falsy な値として
    既定の band に化けてはいけない (このプロジェクトが繰り返し踏んできた
    「偽値と欠損の混同」と同じ形の欠陥になる)。
    """
    values = {"1": 0.30, "2": 0.30, "3": 0.10}
    groups = ig.tie_groups(values, band=0.0)
    assert groups["1"] == groups["2"]
    assert groups["3"] != groups["1"]


# --- feature_names -----------------------------------------------------------------

def test_feature_names_without_layer2_excludes_axes():
    names = ig.feature_names(with_layer2=False)
    for axis in q.AXIS_SPECS:
        assert f"{axis}_signed" not in names
        assert f"{axis}_insufficient" not in names
    assert "logit_layer1" in names
    for name in ig.PYTHON_FEATURE_NAMES:
        assert name in names
        assert f"{name}_missing" in names


def test_feature_names_is_stable():
    assert ig.feature_names() == ig.feature_names()
    assert ig.feature_names(with_layer2=False) == ig.feature_names(with_layer2=False)


def test_feature_names_layout():
    """logit_layer1 + 軸10列 + python 8列 + missing 8列、の順に並ぶ。"""
    names = ig.feature_names(with_layer2=True)
    assert names[0] == "logit_layer1"
    assert names[1:11] == q.feature_names_for_axes()
    assert names[11:19] == ig.PYTHON_FEATURE_NAMES
    assert names[19:] == tuple(f"{n}_missing" for n in ig.PYTHON_FEATURE_NAMES)
    assert len(names) == 27


# --- build_features (wiring) --------------------------------------------------------

def test_build_features_combines_layer1_layer2_and_python_features():
    hf = make_horse_features(umaban="01")
    race = make_race_input([make_entry(umaban="01")])
    ctx = PredictionContext(
        race=race,
        horses=(hf,),
        data_version="v1",
        data_fetched_at=jst(2026, 3, 20),
        notes=(),
        pace_definition=None,
    )
    layer1 = make_layer1({"01": 0.4})
    layer2 = Layer2Result(answers={"01": _uniform_axis_answers()}, failures={}, input_tokens=0)

    features = ig.build_features(ctx, layer1, layer2)

    assert set(features["01"]) == set(ig.feature_names(with_layer2=True))
    assert features["01"]["logit_layer1"] == pytest.approx(ig.logit(0.4))


# --- model_scores / score_breakdown ------------------------------------------------

def test_model_scores_is_weighted_sum():
    weights = make_weights(("a", "b"), {"a": 2.0, "b": -1.0})
    features = {"01": {"a": 3.0, "b": 4.0}}
    result = ig.model_scores(features, weights)
    assert result["01"] == pytest.approx(2.0 * 3.0 + (-1.0) * 4.0)


def test_model_scores_ignores_features_not_in_weights():
    weights = make_weights(("a",), {"a": 2.0})
    features = {"01": {"a": 3.0, "unrelated": 999.0}}
    result = ig.model_scores(features, weights)
    assert result["01"] == pytest.approx(6.0)


def test_score_breakdown_sums_to_score():
    weights = make_weights(("a", "b"), {"a": 1.5, "b": -0.5})
    features = {"01": {"a": 2.0, "b": 4.0}, "02": {"a": -1.0, "b": 0.5}}
    scores = ig.model_scores(features, weights)
    breakdown = ig.score_breakdown(features, weights)
    for umaban in features:
        assert sum(breakdown[umaban].values()) == pytest.approx(scores[umaban], abs=1e-9)


def test_score_breakdown_has_one_entry_per_weight():
    weights = make_weights(("a", "b", "c"), {"a": 1.0, "b": 1.0, "c": 1.0})
    features = {"01": {"a": 1.0, "b": 1.0, "c": 1.0}}
    breakdown = ig.score_breakdown(features, weights)
    assert set(breakdown["01"]) == set(weights.feature_names)


# --- standardize -------------------------------------------------------------------

def test_standardize_applies_mean_and_std():
    features = {"01": {"distance_change_m": 200.0}}
    standardization = {"distance_change_m": (100.0, 50.0)}
    result = ig.standardize(features, standardization)
    assert result["01"]["distance_change_m"] == pytest.approx(2.0)


def test_standardize_skips_logit_and_indicators():
    """素通し列は、恒等の標準化パラメータ (0.0, 1.0) があっても変換されない。

    fit_standardization は mask が False の列に (0.0, 1.0) を書くので、正しく
    学習された重みファイルは必ずこの形になる。
    """
    features = {"01": {
        "logit_layer1": 1.23,
        "distance_fit_insufficient": 0.4,
        "recent_margin_wavg_missing": 1.0,
    }}
    standardization = {
        "logit_layer1": (0.0, 1.0),
        "distance_fit_insufficient": (0.0, 1.0),
        "recent_margin_wavg_missing": (0.0, 1.0),
    }
    result = ig.standardize(features, standardization)
    assert result["01"]["logit_layer1"] == 1.23
    assert result["01"]["distance_fit_insufficient"] == 0.4
    assert result["01"]["recent_margin_wavg_missing"] == 1.0


@pytest.mark.parametrize(
    "name", ["logit_layer1", "distance_fit_insufficient", "recent_margin_wavg_missing"],
)
def test_standardize_rejects_non_identity_parameters_for_passthrough_columns(name):
    """素通し列に恒等でない標準化パラメータが付いていたら例外にする（fix5 決定9-3）。

    その組み合わせは「学習側はこの列を標準化していた」という意味で、黙って
    素通しするとその列の寄与だけ学習時と推論時で std 倍ずれる（fix1
    Critical 2 と同型の欠陥が、mask の渡し間違いではなく重みファイル側から
    入る経路）。値を無視して通すのではなく、食い違いとして落とす。
    """
    with pytest.raises(ig.WeightsIncompatibleError, match=name):
        ig.standardize({"01": {name: 0.5}}, standardization={name: (0.0, 2.0)})


def test_standardize_leaves_feature_when_std_is_zero():
    features = {"01": {"x": 5.0}}
    standardization = {"x": (5.0, 0.0)}
    result = ig.standardize(features, standardization)
    assert result["01"]["x"] == 5.0


def test_standardize_raises_when_a_standardized_feature_has_no_parameters():
    """標準化すべき特徴量のパラメータが無ければ、黙って素通ししない。

    素通しすると、その特徴量だけ学習時と違う尺度のまま重みに掛かる。
    スコアは狂うのに他のテストは全部通ってしまう――重みファイルと今の特徴量
    集合が食い違っているという意味なので、check_compatible() と同じ
    WeightsIncompatibleError にする（KeyError だと「重みが古い/違う」という
    原因が伝わらず、呼び出し側が反射的に broad except で握りつぶしやすい）。
    """
    with pytest.raises(ig.WeightsIncompatibleError, match="recent_margin_wavg"):
        ig.standardize({"01": {"recent_margin_wavg": 0.5}}, standardization={})


@pytest.mark.parametrize(
    "name", ["logit_layer1", "distance_fit_insufficient", "corner_pos_mean_missing"],
)
def test_standardize_skip_list_features_need_no_parameters(name):
    """logit と 0/1・確率の列は標準化の対象外なので、standardization に
    パラメータが無くても通る――マッピングを引く前に短絡するので、
    KeyError にも WeightsIncompatibleError にもならない。
    """
    result = ig.standardize({"01": {name: 0.5}}, standardization={})
    assert result["01"][name] == 0.5


# --- standardize_mask_for (Task 17 fix1) --------------------------------------------

def test_standardize_mask_for_marks_passthrough_columns_false():
    """logit_layer1 / *_missing / *_insufficient は False、それ以外は True。

    keiba.learn.conditional_logit.fit の standardize_mask はキーワード専用の
    必須引数（fix1 決定2）。この mask を渡し間違えると、学習時に std で割り
    推論時に割らない列が生まれ、寄与が std 倍ずれる――実際に最大10.5倍の
    ずれとして踏まれた不具合（fix1-brief Critical 2）なので、その既定の
    作り方そのものをここで固定する。
    """
    names = ig.feature_names(with_layer2=True)
    mask = ig.standardize_mask_for(names)
    assert len(mask) == len(names)
    for name, should_standardize in zip(names, mask):
        if name == "logit_layer1" or name.endswith("_missing") or name.endswith("_insufficient"):
            assert should_standardize is False, name
        else:
            assert should_standardize is True, name


def test_standardize_mask_for_agrees_with_standardize():
    """standardize_mask_for が True/False とした列を、standardize が実際に
    変換する/しないと一致させる。

    規則を2箇所（standardize の中と standardize_mask_for）に書くと、片方だけ
    直されたときにここが検出できなくなる。standardization には
    mask が True とした列のパラメータしか用意しない――もし standardize が
    実際には別の列集合を素通ししていたら、True の列なのに変換されず
    (raw のまま) 失敗するか、False の列なのに standardization を引こうとして
    WeightsIncompatibleError になる。どちらの食い違いもこのテストで拾える。
    """
    names = ig.feature_names(with_layer2=True)
    mask = ig.standardize_mask_for(names)
    standardization = {
        name: (1.0, 2.0) for name, should_standardize in zip(names, mask) if should_standardize
    }
    features = {"01": {name: 10.0 for name in names}}

    result = ig.standardize(features, standardization)

    for name, should_standardize in zip(names, mask):
        if should_standardize:
            assert result["01"][name] == pytest.approx((10.0 - 1.0) / 2.0), name
        else:
            assert result["01"][name] == 10.0, name


# --- Weights -------------------------------------------------------------------------

def test_weights_roundtrip_through_json(tmp_path):
    weights = make_weights(
        ("logit_layer1", "distance_fit_signed"),
        {"logit_layer1": 1.0, "distance_fit_signed": 0.5},
        standardization={"distance_fit_signed": (0.1, 0.2)},
        calibration={"method": "platt", "a": 1.0, "b": 0.0},
        dropped_features=("current_form_insufficient",),
        splits={"train": "2020-01-01..2024-12-31", "test": "2025-01-01..2025-12-31"},
    )
    path = tmp_path / "weights.json"
    weights.save(path)
    loaded = ig.Weights.load(path)
    assert loaded == weights


def test_weights_check_compatible_passes_for_current_versions():
    weights = make_weights(("a",), {"a": 1.0})
    weights.check_compatible()  # 例外が出なければ OK


def test_weights_check_compatible_rejects_old_prompt_version():
    weights = make_weights(("a",), {"a": 1.0})
    weights = dataclasses.replace(
        weights, versions={**weights.versions, "prompt_version": "0"})
    with pytest.raises(ig.WeightsIncompatibleError, match="prompt_version"):
        weights.check_compatible()


def test_weights_check_compatible_rejects_old_feature_version():
    weights = make_weights(("a",), {"a": 1.0})
    weights = dataclasses.replace(
        weights, versions={**weights.versions, "feature_version": "0"})
    with pytest.raises(ig.WeightsIncompatibleError, match="feature_version"):
        weights.check_compatible()


# --- fix5 決定6: 重みファイルの来歴 ---------------------------------------------

def test_weights_roundtrip_keeps_fitted_at_and_dev_metrics(tmp_path):
    """fitted_at と dev_metrics が save/load を往復する（仕様 §11.1.2）。

    この2欄が無いと、重みファイルから「いつ作ったのか」「2つのうちどちらが
    新しいのか」「なぜこの λ なのか」が分からない。
    """
    weights = make_weights(
        ("logit_layer1",), {"logit_layer1": 1.0},
        fitted_at="2026-09-22T02:34:56+00:00",
        dev_metrics={"lambda": 0.1, "dev_log_likelihood_per_race": -2.5,
                      "configurations": {"with_python_features": {"n_features": 27}}},
    )
    path = tmp_path / "weights.json"
    weights.save(path)

    loaded = ig.Weights.load(path)

    assert loaded.fitted_at == "2026-09-22T02:34:56+00:00"
    assert loaded.dev_metrics["dev_log_likelihood_per_race"] == -2.5
    assert loaded == weights


def test_weights_save_writes_both_new_fields(tmp_path):
    """JSON のキーとしても出る（仕様 §11.1.2 の形に合わせる）。"""
    weights = make_weights(
        ("logit_layer1",), {"logit_layer1": 1.0},
        fitted_at="2026-09-22T02:34:56+00:00", dev_metrics={"lambda": 0.1})
    path = tmp_path / "weights.json"
    weights.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["fitted_at"] == "2026-09-22T02:34:56+00:00"
    assert payload["dev_metrics"] == {"lambda": 0.1}


def test_weights_load_rejects_a_file_without_the_provenance_fields(tmp_path):
    """2欄が無い重みファイルは黙って空を主張させない（.get で埋めない）。"""
    weights = make_weights(("logit_layer1",), {"logit_layer1": 1.0})
    path = tmp_path / "weights.json"
    weights.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["fitted_at"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(KeyError):
        ig.Weights.load(path)

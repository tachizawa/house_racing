"""keiba.learn.calibration のテスト。

isotonic regression そのもの（scikit-learn）の正しさは検証しない。ここで
見るのは (1) 返す辞書が ``interpolate`` の契約（昇順 x・非減少 y）を満たす形
であること、(2) 過信したモデルを実測に合わせて下げる方向に働くこと、
(3) 範囲外・再正規化・全頭0という周辺ケースを ``nan`` を出さずに扱うこと。
"""

from __future__ import annotations

import math

import pytest

from keiba.learn import calibration as cal


# --- fit_calibration ------------------------------------------------------------


def test_fit_returns_monotonic_mapping():
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    outcomes = [0, 0, 0, 1, 0, 1, 1, 1, 1]

    result = cal.fit_calibration(probabilities, outcomes)

    assert result["method"] == "isotonic"
    xs = result["x"]
    ys = result["y"]
    assert len(xs) == len(ys)
    assert xs == sorted(xs)
    assert all(y2 >= y1 for y1, y2 in zip(ys, ys[1:]))


def test_fit_maps_overconfident_to_lower():
    """モデルが 0.8 と答えた馬が実際には 4割しか勝っていないデータを与えると、
    その付近の較正後の値は 0.8 より下がる。"""
    # 0.8 付近の予測は繰り返し外れ (outcome 0 が多い) にして過信を再現する。
    probabilities = [0.1, 0.2, 0.75, 0.8, 0.8, 0.8, 0.8, 0.85, 0.9, 0.95]
    outcomes =       [0,   0,   0,    0,   1,   0,   0,   1,    1,   1]

    result = cal.fit_calibration(probabilities, outcomes)

    assert cal.interpolate(result, 0.8) < 0.8


def test_fit_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        cal.fit_calibration([0.1, 0.2], [1.0])


def test_fit_raises_on_empty():
    with pytest.raises(ValueError):
        cal.fit_calibration([], [])


# --- interpolate ------------------------------------------------------------------


def test_interpolate_clamps_outside_range():
    calibration = {"method": "isotonic", "x": [0.2, 0.5, 0.8], "y": [0.1, 0.4, 0.9]}

    below = cal.interpolate(calibration, 0.0)
    above = cal.interpolate(calibration, 1.0)

    assert below == 0.1  # 学習範囲の下端の y に張り付く
    assert above == 0.9  # 学習範囲の上端の y に張り付く
    assert 0.0 <= below <= 1.0
    assert 0.0 <= above <= 1.0


def test_interpolate_is_linear_inside_range():
    calibration = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}
    assert cal.interpolate(calibration, 0.5) == pytest.approx(0.5)


# --- apply_calibration ------------------------------------------------------------


def test_apply_returns_sum_before_renormalisation():
    # y=x の恒等較正なので、較正後の生の合計はそのまま入力の合計になる。
    calibration = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}
    probabilities = {"1": 0.5, "2": 0.3, "3": 0.1}  # 合計 0.9 (< 1.0)

    _normalized, raw_sum = cal.apply_calibration(calibration, probabilities)

    assert raw_sum == pytest.approx(0.9)


def test_apply_renormalises_to_one():
    calibration = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}
    probabilities = {"1": 0.5, "2": 0.3, "3": 0.1}

    normalized, _raw_sum = cal.apply_calibration(calibration, probabilities)

    assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_raw_sum_is_a_float_even_when_empty():
    """probabilities が空のとき、sum(()) は int 0 を返す。宣言された戻り値の
    型 (float) を裏切らない (fix round 2 Minor 3)。"""
    calibration = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}

    normalized, raw_sum = cal.apply_calibration(calibration, {})

    assert normalized == {}
    assert isinstance(raw_sum, float)


def test_apply_handles_all_zero_after_calibration():
    """較正が全頭 0 を返す退化ケースでも、nan を出さず均等割りにする。"""
    calibration = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 0.0]}
    probabilities = {"1": 0.9, "2": 0.05, "3": 0.05}

    normalized, raw_sum = cal.apply_calibration(calibration, probabilities)

    assert raw_sum == 0.0
    assert normalized == {"1": pytest.approx(1 / 3), "2": pytest.approx(1 / 3),
                           "3": pytest.approx(1 / 3)}
    assert all(not math.isnan(v) for v in normalized.values())


def test_apply_preserves_order():
    """較正は単調なので、確率の大小関係（順位）は較正前後で変わらない。"""
    calibration = {"method": "isotonic", "x": [0.0, 0.3, 0.6, 1.0],
                    "y": [0.05, 0.2, 0.55, 0.95]}
    probabilities = {"1": 0.7, "2": 0.5, "3": 0.2}

    normalized, _raw_sum = cal.apply_calibration(calibration, probabilities)

    order_before = sorted(probabilities, key=lambda k: -probabilities[k])
    order_after = sorted(normalized, key=lambda k: -normalized[k])
    assert order_before == order_after

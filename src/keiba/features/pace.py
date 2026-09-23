"""前半3ハロン(``RA.HaronTimeS3``)からペース区分を定義する。

ペースを「速い／標準／遅い」の3区分で扱うのは、確定結果（実際の前半3ハロン）
と照合できる定義を先に固定してから使うため（spec §5.3「結果との照合に使える
定義を固定してから採用する」）。固定の秒数を決め打ちにせず、学習期間の分布
から三分位を取るのは、コースや距離によって前半3ハロンの相場が大きく違う
（同じ「速い」でも短距離と長距離では意味が違う）ため。

層別化・as_of ゲート・境界の寄せ方は spec §8.2 / §9.3 で決まっているので、
ここでは判断を増やさない。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from keiba import config
from keiba.models import RaceInput, RaceRow

PACE_BANDS: tuple[str, ...] = ("fast", "standard", "slow")

# 層別化の順序。この並びそのものが「どこまで妥協するか」の仕様なので、
# 定数として固定する（spec §8.2）。
STRATA: tuple[tuple[str, ...], ...] = (
    ("track_kind", "distance_band", "jyo_cd"),
    ("track_kind", "distance_band"),
    ("track_kind",),
)


def distance_band(kyori_m: int) -> int:
    """距離を400m刻みのバンドキーに丸める。

    200 を基準点にして400m幅で区切ることで、±200m の近傍（例: 1800/2000/2100）
    が同じバンドに入る。
    """
    return ((kyori_m - 200) // 400) * 400 + 200


def _usable_races(races: Sequence[RaceRow], *, as_of: datetime) -> list[RaceRow]:
    """学習に使える過去走だけを残す。

    平地限定（障害は first_3f_sec が設定されない）・値が実在すること・
    as_of より前の発走であること（対象レース自身や未来のレースから
    閾値を作らない）の3条件。
    """
    return [
        race for race in races
        if race.is_flat and race.first_3f_sec is not None and race.hasso_at < as_of
    ]


def _stratum_value(race: RaceRow, key: str) -> object:
    if key == "distance_band":
        return distance_band(race.kyori_m)
    return getattr(race, key)


def _tertile_bounds(values: Sequence[float]) -> tuple[float, float]:
    """昇順に並べた三分位の境界値。

    first_3f_sec は小さいほど速いので、下位1/3の最大値が fast_max、
    中位1/3の最大値が standard_max になる。

    三分位は補間せず順序統計量で取る。補間すると、どのレースも実際には
    走っていない時計を閾値として作り出すことになる。30走以上あれば
    補間との差は1走分未満で、実質的な違いはない。
    """
    ordered = sorted(values)
    n = len(ordered)
    fast_max = ordered[n // 3 - 1]
    standard_max = ordered[(2 * n) // 3 - 1]
    return fast_max, standard_max


def fit_pace_definition(
    races: Sequence[RaceRow],
    *,
    target: RaceInput,
    as_of: datetime,
    min_races: int = config.PACE_MIN_RACES,
) -> dict | None:
    """学習期間のレースからペース区分の閾値を作る。

    ``as_of`` は必須のキーワード専用引数にしてある。対象レース自身や
    予想時点より後のレースの走破時計から閾値を作ると、それ自体が未来情報
    になる——「閾値でしかない」という理由では免除されない。閾値の形が
    答えの形を決めるため（spec §9.3）。

    層は ``STRATA`` の順に試し、``min_races`` 件に届く最初の層を使う。
    どの層も届かなければ推測で埋めず ``None`` を返す。
    """
    usable = _usable_races(races, as_of=as_of)
    target_values = {
        "track_kind": target.track_kind,
        "distance_band": distance_band(target.kyori_m),
        "jyo_cd": target.jyo_cd,
    }

    for keys in STRATA:
        matched = [
            race for race in usable
            if all(_stratum_value(race, key) == target_values[key] for key in keys)
        ]
        if len(matched) >= min_races:
            fast_max, standard_max = _tertile_bounds(
                [race.first_3f_sec for race in matched])
            return {
                "fast_max": fast_max,
                "standard_max": standard_max,
                "stratum": {key: target_values[key] for key in keys},
                "race_count": len(matched),
                "note": (
                    "学習期間のレースの前半3ハロン(RA.HaronTimeS3)の三分位。"
                    "小さいほど速い。確定結果と照合できる定義。"
                ),
            }
    return None


def classify_pace(first_3f_sec: float | None, definition: dict) -> str | None:
    """前半3ハロンの実測値をペース区分に落とす。確定結果との照合に使う。

    first_3f_sec は小さいほど速い。閾値ちょうどの値は速い側に寄せる（<=）。
    寄せ方を決めておかないと、同じデータで区分が変わりうる。
    """
    if first_3f_sec is None:
        return None
    if first_3f_sec <= definition["fast_max"]:
        return "fast"
    if first_3f_sec <= definition["standard_max"]:
        return "standard"
    return "slow"

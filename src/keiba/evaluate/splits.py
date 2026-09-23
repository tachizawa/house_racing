"""時系列分割（spec §9.2）。

学習・過学習監視・較正・最終評価の4区分に、時系列の早い順で振り分ける。

**区分は予想締切（``prediction_cutoff``）の日付で決める。発走日時を渡しては
いけない——境界の翌開催日が丸ごと1区分ずれる（fix5 §4）。** 発走日で振り分けた
``keiba.evaluate.backtest`` と、記録の ``prediction_cutoff`` で振り分けた
``scripts/fit_weights.py`` が食い違い、backtest が「held-out。学習に使って
いません」と銘打ったレースの一部が fit_weights の calib に入っていた。基準は
``keiba.evaluate.backtest.SPLIT_BASIS``（``"prediction_cutoff"``）にまとめてあり、
``report()["run"]["split_basis"]`` としてレポートにも出る。

境界はレース単位で決まるので、同じレースの馬を学習側と評価側にまたがって分割
することは構造的に起きない――assign() は1レースにつき1回しか呼ばれず、そのレース
の全頭が同じ答えを受け取る。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from keiba.models import RaceRow

__all__ = [
    "SPLIT_NAMES",
    "Splits",
    "SplitError",
    "assign",
    "by_fraction",
    "group_by_split",
]

#: 区分名。境界の早い順。
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "calib", "test")


class SplitError(ValueError):
    """分割の指定（fractions・dates）が不正なときに送出する。"""


@dataclass(frozen=True)
class Splits:
    """4区分の境界日。境界日はその区分に含まれる最後の日（両端を含む）。"""

    train_until: date
    dev_until: date
    calib_until: date

    def as_dict(self) -> dict[str, str]:
        """JSON にそのまま乗せられる形（日付は ISO 文字列）。"""

        return {
            "train_until": self.train_until.isoformat(),
            "dev_until": self.dev_until.isoformat(),
            "calib_until": self.calib_until.isoformat(),
        }


def assign(at: datetime, splits: Splits) -> str:
    """``at`` の日付がどの区分に入るかを返す。境界日は早い側の区分に含める。

    **``at`` に渡すのは予想締切（``prediction_cutoff``）。発走日時
    （``RaceRow.hasso_at``）を渡してはいけない。** 発走で判定すると、記録側
    （``PredictionRecord.prediction_cutoff`` で判定する
    ``scripts/fit_weights.py``）と各境界の翌開催日が丸ごと1区分ずれる
    （fix5 §4 の欠陥そのもの）。引数名を ``hasso_at`` から ``at`` に変えたのは、
    素直に発走日時を渡す経路を名前の側から塞ぐため（fix6 ④）。
    バックテストは ``keiba.evaluate.backtest.prediction_cutoff_for(race)`` を
    通した値を渡す。
    """

    d = at.date()
    if d <= splits.train_until:
        return "train"
    if d <= splits.dev_until:
        return "dev"
    if d <= splits.calib_until:
        return "calib"
    return "test"


def _clamp_index(idx: int, lo: int, hi: int) -> int:
    return min(max(idx, lo), hi)


def by_fraction(
    dates: Sequence[date],
    *,
    fractions: tuple[float, float, float, float] = (0.5, 0.2, 0.15, 0.15),
) -> Splits:
    """日付の並びを fractions の比率（train, dev, calib, test）で4分割する。

    境界は dates を昇順に並べたときの位置で決める。fractions は合計が
    1.0 でなければならない。
    """

    if not dates:
        raise SplitError("dates が空です。境界を決められません。")

    total = sum(fractions)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise SplitError(f"fractions の合計が1.0ではありません: {total}")

    ordered = sorted(dates)
    n = len(ordered)
    train_frac, dev_frac, calib_frac, _test_frac = fractions

    train_idx = _clamp_index(round(train_frac * n), 1, n)
    dev_idx = _clamp_index(round((train_frac + dev_frac) * n), train_idx, n)
    calib_idx = _clamp_index(round((train_frac + dev_frac + calib_frac) * n), dev_idx, n)

    return Splits(
        train_until=ordered[train_idx - 1],
        dev_until=ordered[dev_idx - 1],
        calib_until=ordered[calib_idx - 1],
    )


def group_by_split(races: Sequence[RaceRow], splits: Splits, *,
                    at: Callable[[RaceRow], datetime]) -> dict[str, list[RaceRow]]:
    """レースを4区分に振り分ける。

    ``at`` はレースから**判定に使う時刻**を取り出す関数で、キーワード専用の
    必須引数。既定値を置かない――既定にできる唯一の候補は ``race.hasso_at``
    だが、それは ``assign`` に渡してはいけない値そのものである（fix5 §4 /
    fix6 ④）。バックテストと同じ基準にするなら
    ``keiba.evaluate.backtest.prediction_cutoff_for`` を渡す（この関数から
    その定義を import しないのは、締切の規則が評価側にあり、ここが
    ``backtest`` を import すると循環するため）。

    RaceRow はレース単位（1レース1行）なので、同じ race_id が2つの区分に
    またがって現れることは無い。空の区分もキーとしては必ず4つとも返す。
    """

    groups: dict[str, list[RaceRow]] = {name: [] for name in SPLIT_NAMES}
    for race in races:
        groups[assign(at(race), splits)].append(race)
    return groups

"""能力指標。直近走の着差とクラスを、合成せず別々の数値のまま返す。

クラスを秒に換算して着差と足し合わせる案は設計段階で外した。換算係数に
根拠がなく、測定値のふりをした当て推量になるため（詳細は docs のタスク仕様）。
ここでは着差の加重平均とクラスの加重平均・最大値を別項目として出し、
重み付けは呼び出し側（第3層 or 将来のフィット）に委ねる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from keiba import config
from keiba.models import PastRun


@dataclass(frozen=True)
class AbilitySummary:
    """1頭分の能力指標。値が欠けている項目は None（0.0 で埋めない）。"""

    recent_margin_sec_wavg: float | None
    runs_used: int
    recent_class_level_wavg: float | None
    best_class_level: int | None
    period_from: date | None
    period_to: date | None


def _weighted(values: Sequence[float | None], weights: Sequence[int]) -> float | None:
    """欠損を飛ばした加重平均。分母は実際に使った重みの合計。

    走数が recent_runs 未満のときに分母を固定すると値が不当に小さくなるので、
    使った重みだけを足す。
    """
    total = weight_sum = 0.0
    for value, weight in zip(values, weights):
        if value is None:
            continue
        total += value * weight
        weight_sum += weight
    return total / weight_sum if weight_sum else None


def ability_summary(runs: Sequence[PastRun], *,
                     recent_runs: int = config.DEFAULT_RECENT_RUNS) -> AbilitySummary:
    """直近 recent_runs 走から能力指標をまとめる。

    対象は平地・出走集合に入る走（race.is_flat かつ result.counts_as_starter）。
    呼び出し側の並び順は信用せず、ここで race.hasso_at の新しい順に並べ直す
    （Repository.past_runs は新しい順だが、それに依存する関数は次の呼び出し側で
    壊れる罠になる）。

    窓は「直近 recent_runs 走の対象走」であって「直近 recent_runs 走の
    margin が取れた走」ではない。競走中止（時計なし）の走も対象走として窓の
    1枠を消費する — 実際に出走した走であり、runs_used が窓の大きさより
    小さいこと自体が「見た目より比較材料が少ない」という信号になるため。
    """
    eligible = [run for run in runs if run.race.is_flat and run.result.counts_as_starter]
    eligible.sort(key=lambda run: run.race.hasso_at, reverse=True)
    window = eligible[:recent_runs]
    weights = list(range(recent_runs, 0, -1))

    # margin は時計が無い走（競走中止・失格・margin_sec 欠損）を追加で除く。
    margins = [run.margin_sec if run.has_time else None for run in window]
    runs_used = sum(1 for margin in margins if margin is not None)
    recent_margin_sec_wavg = _weighted(margins, weights)

    class_levels = [run.race.class_level for run in window]
    recent_class_level_wavg = _weighted(class_levels, weights)
    known_class_levels = [level for level in class_levels if level is not None]
    best_class_level = max(known_class_levels) if known_class_levels else None

    used_dates = [
        run.race.hasso_at.date()
        for run, margin in zip(window, margins)
        if margin is not None
    ]
    period_from = min(used_dates) if used_dates else None
    period_to = max(used_dates) if used_dates else None

    return AbilitySummary(
        recent_margin_sec_wavg=recent_margin_sec_wavg,
        runs_used=runs_used,
        recent_class_level_wavg=recent_class_level_wavg,
        best_class_level=best_class_level,
        period_from=period_from,
        period_to=period_to,
    )


def field_ranks(values: Mapping[str, float | None]) -> dict[str, float | None]:
    """昇順の順位（値が小さいほど良い）を付ける。同値は平均順位。None は None のまま。

    ability_summary は呼ばない。すでに計算済みの値の上でだけ働く純粋関数にして、
    「順位付け」と「指標の計算」を別の関心事として保つ。
    """
    known = [(key, value) for key, value in values.items() if value is not None]
    known.sort(key=lambda item: item[1])

    ranks: dict[str, float | None] = dict.fromkeys(values)
    i = 0
    while i < len(known):
        j = i
        while j < len(known) and known[j][1] == known[i][1]:
            j += 1
        # i, j は 0 始まりの半開区間。順位は 1 始まりなので i+1..j の平均。
        average_rank = (i + 1 + j) / 2
        for key, _ in known[i:j]:
            ranks[key] = average_rank
        i = j
    return ranks

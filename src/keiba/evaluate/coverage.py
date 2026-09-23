"""データ版の充足度。予想対象レースの出走馬のうち、予想締切より前の出走記録が
データ版にある馬の割合を測る。

CLAUDE.md の実測（延べ1,358頭のうち締切前の出走記録がある馬 29.0%）を、
データ版を取り直したあとに同じ定義で測り直すためのもの。Jev は呼ばない。
締切は evaluate/backtest.py と同じ定義（prediction_cutoff_for）を使う。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from keiba.data.repository import Repository, ResultRepository
from keiba.evaluate.backtest import prediction_cutoff_for
from keiba.models import RaceRow

__all__ = ["CoverageReport", "measure"]


@dataclass(frozen=True)
class CoverageReport:
    n_races: int
    n_horses: int
    n_with_prior_run: int
    n_with_3plus_prior_runs: int
    #: n_horses が 0 なら NaN（0.0 ではない。測っていない）。
    share_with_prior_run: float
    coverage_from: datetime | None


def measure(races: Sequence[RaceRow], repo: Repository,
            results: ResultRepository) -> CoverageReport:
    n_horses = n_prior = n_three = 0
    for race in races:
        as_of = prediction_cutoff_for(race)
        for result in results.results(race.race_id):
            n_horses += 1
            ketto_num = result.entry.ketto_num
            runs = repo.past_runs(ketto_num, as_of=as_of) if ketto_num else []
            if runs:
                n_prior += 1
            if len(runs) >= 3:
                n_three += 1
    share = n_prior / n_horses if n_horses else float("nan")
    return CoverageReport(
        n_races=len(races), n_horses=n_horses, n_with_prior_run=n_prior,
        n_with_3plus_prior_runs=n_three, share_with_prior_run=share,
        coverage_from=repo.coverage_from,
    )

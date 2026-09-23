"""過去走の行と、過去走が無い・少ない理由の区別（report3 §4.3・§4.4）。

集計値だけでなく、根拠となる過去走そのものを数件添える。過去走が無いことは
「能力が低い」ではない。新馬で本当に無いのか、データ版の取得期間の外にあるのか、
区別できないのかを分けて書く。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from keiba import config
from keiba.features.course import decode_track_cd
from keiba.models import PastRun

__all__ = [
    "RUN_LINE_KEYS", "HistoryStatus", "history_status", "recent_run_lines", "run_line",
]

RUN_LINE_KEYS: tuple[str, ...] = (
    "date", "days_before", "jyo", "track_cd", "track_label", "track_kind", "kyori_m",
    "baba", "class_level", "syusso_tosu", "kakutei_jyuni", "margin_sec",
    "corner_positions", "last_3f_sec", "futan_kg",
)


def run_line(run: PastRun, *, as_of: datetime) -> dict:
    """過去走1件の行。締切より前の走だけが来る前提（Repository.past_runs の as_of）。"""
    track = decode_track_cd(run.race.track_cd)
    result = run.result
    return {
        "date": run.race.hasso_at.date().isoformat(),
        "days_before": (as_of - run.race.hasso_at).days,
        "jyo": run.race.jyo,
        "track_cd": run.race.track_cd,
        "track_label": track.label if track else None,
        "track_kind": run.race.track_kind,
        "kyori_m": run.race.kyori_m,
        "baba": run.race.baba,
        "class_level": run.race.class_level,
        "syusso_tosu": run.race.syusso_tosu,
        # 競走中止・失格は着順の数字が残っていても着順として読まない。
        "kakutei_jyuni": result.kakutei_jyuni if result.can_win else None,
        "margin_sec": result.margin_sec if run.has_time else None,
        "corner_positions": [result.jyuni_1c, result.jyuni_2c, result.jyuni_3c, result.jyuni_4c],
        "last_3f_sec": result.last_3f_sec,
        "futan_kg": result.entry.futan_kg,
    }


def recent_run_lines(runs: Sequence[PastRun], *, as_of: datetime, limit: int) -> list[dict]:
    """平地・出走集合に入る過去走の新しい順 ``limit`` 件。取消・除外は走っていないので除く。"""
    usable = [run for run in runs if run.race.is_flat and run.result.counts_as_starter]
    usable.sort(key=lambda run: run.race.hasso_at, reverse=True)
    return [run_line(run, as_of=as_of) for run in usable[:limit]]


@dataclass(frozen=True)
class HistoryStatus:
    status: str
    note: str
    runs_found: int
    period_from: date | None
    period_to: date | None


_NOTES = {
    "found": "締切前の出走記録がデータ版にある。",
    "found_may_be_truncated": (
        "締切前の出走記録はあるが、最古の記録がデータ版の収録開始に近く、"
        "それより前の出走が版の外にある可能性がある。"),
    "debut": "新馬戦のため、過去走は存在しない。",
    "not_in_window": (
        "1勝クラス以上のため過去走は実在するはずだが、データ版の取得期間内に"
        "見つからない。能力が低いことを意味しない。"),
    "unknown": (
        "データ版の取得期間内に過去走が見つからない。未出走か取得期間外かは区別できない。"
        "能力が低いことを意味しない。"),
    "unmatched": "馬名をデータ版の馬と照合できなかったため、過去走を引いていない。",
}


def history_status(runs: Sequence[PastRun], *, matched: bool, race_class_level: int | None,
                   coverage_from: datetime | None) -> HistoryStatus:
    def make(status: str, found: Sequence[PastRun] = ()) -> HistoryStatus:
        dates = [run.race.hasso_at.date() for run in found]
        return HistoryStatus(status=status, note=_NOTES[status], runs_found=len(found),
                             period_from=min(dates) if dates else None,
                             period_to=max(dates) if dates else None)

    if not matched:
        return make("unmatched")
    if runs:
        if coverage_from is not None:
            earliest = min(run.race.hasso_at for run in runs)
            if (earliest - coverage_from).days <= config.HISTORY_TRUNCATION_MARGIN_DAYS:
                return make("found_may_be_truncated", runs)
        return make("found", runs)
    if race_class_level == 0:
        return make("debut")
    if race_class_level is not None and race_class_level >= 2:
        return make("not_in_window")
    return make("unknown")

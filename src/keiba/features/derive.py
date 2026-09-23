"""条件別成績・近況・脚質・輸送。Task 9（ability.py）が作った能力指標に対して、
「どんな条件で走ってきたか」「直前の状態」「レースの運び方」「輸送負担」を
別の指標として積む。ここでも合成はしない — 各項目は独立した数値のまま返し、
重み付けは第3層（呼び出し側）に委ねる。

4つの機能群は互いに独立している。「平地・出走集合に入る過去走だけを見る」
という Task 9 由来の前提（_eligible_runs）は条件別成績・近況の2つで共有する
（どちらも今回の平地レースとの比較なので、障害走を除く理由がそのまま効く）。
脚質（running_style）は頭数0頭の走を追加で除く必要があるため独自のフィルタを
持ち（仕様側の指定どおり）、輸送（transport）は逆に障害走を除いてはいけない
ため別のフィルタ（_transported_runs）を持つ。「隣の関数と同じにする」ことを
フィルタの根拠にしない — 根拠は常に「その数値が何を測っているか」。
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from keiba import config
from keiba.models import PastRun, RaceInput

__all__ = [
    "BABA_UNKNOWN_REASON",
    "CONDITION_KEYS",
    "ConditionRecord",
    "RecentForm",
    "RunningStyle",
    "Transport",
    "condition_records",
    "recent_form",
    "running_style",
    "summarize_runs",
    "transport",
]


def _eligible_runs(runs: Sequence[PastRun]) -> list[PastRun]:
    """平地・出走集合に入る過去走だけ（ability_summary と同じ基準）。

    条件別成績・近況はこの集合を土台にする。どちらも今回の平地レースとの
    比較なので、障害走を除く理由（着差が比較できない）がそのまま当てはまる。
    輸送（transport）はこれを使わない — 輸送距離・馬体重の増減は平地でも
    障害でも同じように発生するので、障害走を除く理由がないため
    （_transported_runs を参照）。
    """
    return [run for run in runs if run.race.is_flat and run.result.counts_as_starter]


# === 条件別成績 ==============================================================


@dataclass(frozen=True)
class ConditionRecord:
    """条件を絞った過去走の集計。starts が常に添う（spec §4.3）。

    inapplicable_reason が None でないときは、その条件が**今回のレースについて
    定義できない**ことを表す。このとき starts / wins / top3 は 0 ではなく None:
    0 は「該当する走が無かった」という馬についての主張であり、「条件そのものが
    決まっていない」とは別のことだから。None にしておけば、区別せずに数を読む
    コードは黙って 0 を読むのではなく落ちる（spec §14.1「未入力は欠損として明示」）。

    inapplicable_reason が None のときは starts / wins / top3 は必ず int。
    starts == 0 のときも wins/top3 は 0（該当なしなので自然な0）だが、
    margin_sec_mean・period_from・period_to は「計算していない」ことを表す
    None にする（0.0 で埋めない）。
    """

    starts: int | None
    wins: int | None
    top3: int | None
    margin_sec_mean: float | None
    period_from: date | None
    period_to: date | None
    inapplicable_reason: str | None = None


#: 今回のレースの馬場状態が未入力のとき、same_baba に載せる理由。
#: 表示（predict/evidence.py）と state（features/summary.py）が同じ文言を
#: 使うように、文字列はここ1箇所に置く。
BABA_UNKNOWN_REASON = "今回の馬場状態が未入力のため比較できない"


def _matches_exact_distance(run: PastRun, race: RaceInput) -> bool:
    return run.race.kyori_m == race.kyori_m


def _matches_same_distance(run: PastRun, race: RaceInput) -> bool:
    return abs(run.race.kyori_m - race.kyori_m) <= config.SAME_DISTANCE_TOLERANCE_M


def _matches_same_venue(run: PastRun, race: RaceInput) -> bool:
    return run.race.jyo_cd == race.jyo_cd


def _matches_same_track_kind(run: PastRun, race: RaceInput) -> bool:
    return run.race.track_kind == race.track_kind


def _matches_same_baba(run: PastRun, race: RaceInput) -> bool:
    """馬場状態が一致し、かつトラック種別も一致する走だけ。

    馬場状態コード（良/稍重/重/不良）は芝とダートで別基準で付けられるので、
    track_kind が違えば「同じ馬場」とは言えない。

    race.baba_cd が None（利用者が馬場状態を入力しなかった）の場合はこの関数を
    呼ばない前提（condition_records が same_baba を「定義できない」として
    別扱いする）。ここに防御的な None 分岐を残すと「未入力でも 0戦として
    集計される」経路が生き残ってしまう。
    """
    return run.race.baba_cd == race.baba_cd and run.race.track_kind == race.track_kind


# CONDITION_KEYS はこの辞書の挿入順そのもの。インターフェース仕様の順と揃えてある。
_CONDITION_MATCHERS: dict[str, Callable[[PastRun, RaceInput], bool]] = {
    "exact_distance": _matches_exact_distance,
    "same_distance": _matches_same_distance,
    "same_venue": _matches_same_venue,
    "same_track_kind": _matches_same_track_kind,
    "same_baba": _matches_same_baba,
}

CONDITION_KEYS: tuple[str, ...] = tuple(_CONDITION_MATCHERS)


def _build_condition_record(matched: list[PastRun]) -> ConditionRecord:
    starts = len(matched)
    if starts == 0:
        return ConditionRecord(starts=0, wins=0, top3=0, margin_sec_mean=None,
                                period_from=None, period_to=None)

    wins = sum(1 for run in matched if run.result.is_winner)
    # top3 は「勝ち馬になり得た（competes_as_starter かつ時計あり）」走のうち
    # 1〜3着。競走中止・失格は can_win が False なので自動的に除かれる。
    top3 = sum(
        1 for run in matched
        if run.result.can_win and run.result.kakutei_jyuni in (1, 2, 3)
    )
    margins = [run.margin_sec for run in matched if run.has_time]
    margin_sec_mean = sum(margins) / len(margins) if margins else None
    dates = [run.race.hasso_at.date() for run in matched]
    return ConditionRecord(
        starts=starts,
        wins=wins,
        top3=top3,
        margin_sec_mean=margin_sec_mean,
        period_from=min(dates),
        period_to=max(dates),
    )


def summarize_runs(matched: list[PastRun]) -> ConditionRecord:
    """条件で絞り終えた過去走を1つの ConditionRecord に集計する。

    features/history.py の組み合わせ集計が同じ集計規則（top3 は can_win の
    1〜3着、着差は時計のある走だけ）を使うための公開口。規則はここ1箇所に置く。
    """
    return _build_condition_record(matched)


def condition_records(runs: Sequence[PastRun], *, race: RaceInput) -> dict[str, ConditionRecord]:
    """今回のレース条件との近さ別に過去走を束ねる。5区分すべてを常に返す。

    区分は互いに排他ではない（同じ走が exact_distance にも same_distance にも
    入りうる）。呼び出し側が知りたい軸ごとに別の質問として使うための束ね方。

    race.baba_cd が None（利用者が馬場状態を入力しなかった）のときは、
    same_baba だけ「定義できない」レコードにする。0戦として集計すると
    「この馬はこの馬場状態で走ったことがない」という馬についての主張に
    なってしまうが、実際には今回の馬場状態が分からないだけで、馬については
    何も言えていないため（他の4区分はこの影響を受けない）。
    """
    eligible = _eligible_runs(runs)
    records: dict[str, ConditionRecord] = {}
    for key, matcher in _CONDITION_MATCHERS.items():
        if key == "same_baba" and race.baba_cd is None:
            records[key] = ConditionRecord(
                starts=None, wins=None, top3=None, margin_sec_mean=None,
                period_from=None, period_to=None,
                inapplicable_reason=BABA_UNKNOWN_REASON,
            )
            continue
        records[key] = _build_condition_record([run for run in eligible if matcher(run, race)])
    return records


# === 近況 =====================================================================


@dataclass(frozen=True)
class RecentForm:
    """直前走からの変化。前走が無ければ全項目 None。"""

    days_since_last_run: int | None
    distance_change_m: int | None
    futan_change_kg: float | None
    class_level_change: int | None
    last_run: dict | None


def _diff(today: float | int | None, previous: float | int | None) -> float | int | None:
    """今回 - 前走。片方でも欠けていたら None（0 とみなさない）。"""
    if today is None or previous is None:
        return None
    return today - previous


def _last_run_dict(run: PastRun) -> dict:
    """前走の条件と結果をそのまま辞書にする。

    ba_taijyu_kg は前走時点の馬体重である。今回の馬体重は当日発表されるもので
    予想時点（前日〜前夜）では分からないため、ここに入るのは必ず前走の値。
    呼び出し側が「今回の馬体重」と取り違えないよう、この関数の外でも
    docstring に明記しておく。
    """
    return {
        "kyori_m": run.race.kyori_m,
        "track_kind": run.race.track_kind,
        "baba_cd": run.race.baba_cd,
        "class_level": run.race.class_level,
        "syusso_tosu": run.race.syusso_tosu,
        "kakutei_jyuni": run.result.kakutei_jyuni,
        "margin_sec": run.result.margin_sec,
        "futan_kg": run.result.entry.futan_kg,
        "ba_taijyu_kg": run.result.ba_taijyu_kg,  # 前走時点の馬体重（今回の値ではない）
    }


def recent_form(runs: Sequence[PastRun], *, as_of: datetime, race: RaceInput,
                 futan_kg: float | None, class_level: int | None) -> RecentForm:
    """直近の出走からの変化をまとめる。futan_kg / class_level は今回側の値を
    呼び出し側から受け取る（RaceInput は馬ごとの斤量やクラスの数値を持たないため）。
    """
    eligible = _eligible_runs(runs)
    if not eligible:
        return RecentForm(None, None, None, None, None)

    last = max(eligible, key=lambda run: run.race.hasso_at)
    days_since_last_run = (as_of - last.race.hasso_at).days

    return RecentForm(
        days_since_last_run=days_since_last_run,
        distance_change_m=_diff(race.kyori_m, last.race.kyori_m),
        futan_change_kg=_diff(futan_kg, last.result.entry.futan_kg),
        class_level_change=_diff(class_level, last.race.class_level),
        last_run=_last_run_dict(last),
    )


# === 脚質 =====================================================================


@dataclass(frozen=True)
class RunningStyle:
    corner_pos_mean: float | None
    corner_pos_stdev: float | None
    corner_pos_starts: int
    corner_source: str | None


_CORNER_SOURCES = ("jyuni_4c", "jyuni_3c", "jyuni_2c")


def running_style(runs, *, recent_runs=config.DEFAULT_RECENT_RUNS) -> RunningStyle:
    """4角の相対位置から脚質傾向を出す。

    4角が取れないコース（コーナー数が少ない）のために 3角→2角へ落とすが、
    走ごとに別の情報源を混ぜない。一度 jyuni_4c を選んだら、4角が無い走は
    集計から外す。混ぜると「同じ指標」として比較できなくなるため。
    どの列を使ったかは corner_source に残す。

    呼び出し側の並び順は信用せず、ここで hasso_at の新しい順に並べ直してから
    直近 recent_runs 走を切り出す（ability_summary と同じ理由）。
    """
    usable = [run for run in runs
              if run.race.is_flat and run.result.counts_as_starter
              and run.race.syusso_tosu]
    # 呼び出し側の並び順を信用しない。ability_summary と同じ理由で、
    # 入力の順序に依存する関数は次の呼び出し側で静かに壊れる。
    usable.sort(key=lambda run: run.race.hasso_at, reverse=True)
    usable = usable[:recent_runs]

    for source in _CORNER_SOURCES:
        positions = [
            getattr(run.result, source) / run.race.syusso_tosu
            for run in usable
            if getattr(run.result, source)
        ]
        if positions:
            mean = sum(positions) / len(positions)
            stdev = statistics.stdev(positions) if len(positions) > 1 else None
            return RunningStyle(
                corner_pos_mean=mean,
                corner_pos_stdev=stdev,
                corner_pos_starts=len(positions),
                corner_source=source,
            )
    return RunningStyle(None, None, 0, None)


# === 輸送 =====================================================================


@dataclass(frozen=True)
class Transport:
    """今回の輸送見込みと、過去の長距離輸送時の馬体重変化。"""

    base: str | None
    estimated_transport_km: int | None
    stay_status: str
    past_long_transport_starts: int
    past_long_transport_zogen_mean: float | None
    past_long_transport_zogen_min: int | None


def _run_base(run: PastRun) -> str | None:
    """その走の時点の所属トレセン。

    今回の所属（呼び出し側が渡す tozai_cd）ではなく、その走の EntryRow から読む。
    転厩がありうるため、今回の所属で過去の輸送距離を計算すると別のトレセンから
    走っていた頃の輸送を今の所属のものとして数えてしまう。
    """
    return config.TOZAI_CD_TO_BASE.get(run.result.entry.tozai_cd)


def _run_transport_km(run: PastRun) -> int | None:
    base = _run_base(run)
    if base is None:
        return None
    return config.TRANSPORT_KM.get((base, run.race.jyo_cd))


def _transported_runs(runs: Sequence[PastRun]) -> list[PastRun]:
    """輸送の集計に使う走。_eligible_runs とは違い、障害走を除かない。

    能力指標（ability.py）が障害走を外すのは着差が比較できないからで、
    その理由はここには当てはまらない。栗東から札幌への輸送は、それが
    平地でも障害でも同じ距離を運ばれたということ。体重の増減も同じように出る。
    ここで _eligible_runs を再利用すると、実際にあった長距離輸送の証拠を
    「平地の比較に使えないから」という無関係な理由で捨ててしまう。

    出走取消・発走除外・競走除外（counts_as_starter が False）は外す。
    輸送されたかどうか自体が不確かなうえ、計量がないので増減差が空になり、
    starts だけを膨らませて平均を薄めてしまうため。
    """
    return [run for run in runs if run.result.counts_as_starter]


def transport(runs: Sequence[PastRun], *, tozai_cd: str, jyo_cd: str,
              stay_status: str = "unknown") -> Transport:
    """今回の輸送距離見込みと、過去の長距離輸送で馬体重をどう減らしてきたか。

    stay_status は呼び出し側が明示的に渡した値をそのまま通すだけで、
    ここでは一切推測しない。開催地が遠いからといって「現地滞在したはず」と
    決めつけるのは、この関数がやってはいけないことそのもの（spec §4.2）。
    """
    base = config.TOZAI_CD_TO_BASE.get(tozai_cd)
    estimated_transport_km = (
        config.TRANSPORT_KM.get((base, jyo_cd)) if base is not None else None
    )

    # 「長距離輸送」は閾値**以上**（>=）。仕様 §6.6 が「`estimated_transport_km`
    # が閾値（既定 300km、設定値）以上だった走」と定めているので、境界の
    # 300km ちょうどは長距離側に含める（以前は厳密な超過で判定していて、
    # 仕様と1件ぶんずれていた。fix5 決定10）。
    long_runs = []
    for run in _transported_runs(runs):
        km = _run_transport_km(run)
        if km is not None and km >= config.LONG_TRANSPORT_KM:
            long_runs.append(run)

    starts = len(long_runs)
    zogens = [run.result.zogen_sa_kg for run in long_runs if run.result.zogen_sa_kg is not None]

    return Transport(
        base=base,
        estimated_transport_km=estimated_transport_km,
        stay_status=stay_status,
        past_long_transport_starts=starts,
        past_long_transport_zogen_mean=sum(zogens) / len(zogens) if zogens else None,
        past_long_transport_zogen_min=min(zogens) if zogens else None,
    )

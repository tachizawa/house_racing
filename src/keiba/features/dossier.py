"""1レース分の予想用資料（report3 §4）。API は呼ばない。

資料は JSON-safe な dict で、そのまま Jev の state にもなる（to_state）。
人間が確認する Markdown は dossier_markdown.py が同じ dict から作る——
「人が見た物」と「Jev に送った物」を別々に組み立てると黙ってずれる。

事実（コースの基本情報、過去走）と予測（展開）を混ぜない。展開の予測は案Bで
pace_forecast として to_state の段階で足すだけで、資料そのものには入れない。
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from fractions import Fraction

from keiba import config
from keiba.data.repository import Repository
from keiba.features import course as course_mod
from keiba.features import history
from keiba.features.context import build_context
from keiba.features.derive import eligible_runs
from keiba.features.summary import ability_dict, data_quality, race_dict, running_style_dict
from keiba.models import RaceInput

__all__ = ["StateTooLargeError", "build_dossier", "check_state_size", "state_chars", "to_state"]


class StateTooLargeError(ValueError):
    def __init__(self, chars: int, limit: int) -> None:
        super().__init__(
            f"state が {chars:,} 文字で上限 {limit:,} 文字を超えています。"
            "--max-recent-runs を減らすか、上限を実測で見直してください。")
        self.chars = chars
        self.limit = limit


def _iso(value):
    return value.isoformat() if value is not None else None


def _course_dict(identification, facts) -> dict:
    return {
        "identified": identification.identified,
        "reason": identification.reason,
        "jyo_cd": identification.jyo_cd,
        "track_cd": identification.track_cd,
        "track_label": identification.track.label if identification.track else None,
        "direction": identification.track.direction if identification.track else None,
        "kyori_m": identification.kyori_m,
        "course_kubun_cd": identification.course_kubun_cd,
        "facts": facts.facts if facts else None,
        "start_note": facts.start_note if facts else None,
        "facts_source": ({"url": facts.source_url, "retrieved_at": facts.retrieved_at,
                          "valid_from": facts.valid_from, "valid_to": facts.valid_to,
                          "rail_basis": facts.rail_basis}
                         if facts else None),
    }


def _conditions_dict(records) -> dict:
    """definition は含めない——5区分ぶん、馬ごとに同じ文が繰り返され state を
    膨らませるため、top-level の condition_definitions に1回だけ持つ
    （statesize-fix-brief.md）。"""
    return {
        key: {
            "starts": record.starts,
            "wins": record.wins,
            "top3": record.top3,
            "margin_sec_mean": record.margin_sec_mean,
            "period": {"from": _iso(record.period_from), "to": _iso(record.period_to)},
            "inapplicable_reason": record.inapplicable_reason,
        }
        for key, (definition, record) in records.items()
    }


def _condition_definitions(today: history.TodayConditions) -> dict[str, str]:
    """5区分の比較条件の定義文。過去走（runs）には依らず今回の条件だけで決まるので、
    レースにつき1回だけ作る（馬ごとに同じ文を持たせない）。"""
    records = history.combined_condition_records((), today=today)
    return {key: definition for key, (definition, _) in records.items()}


def _threshold_text(value: float) -> str:
    """判定に使う閾値を、文言でも同じ値として読めるように書く（final review I-3c）。

    有限小数で書けない分数（1/3 など）は「1/3」と分数で書く。それ以外は
    float の repr（round-trip で同じ値に戻る最短の表記）。``:.2f`` のように
    丸めると「0.33 以下」と書いて 1/3 で判定し、文言と判定がずれる。
    """
    fraction = Fraction(value).limit_denominator(12)
    denominator = fraction.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if float(fraction) == value and denominator != 1:
        return f"{fraction.numerator}/{fraction.denominator}"
    return repr(float(value))


def build_dossier(race_input: RaceInput, repo: Repository, *,
                  max_recent_runs: int = config.DEFAULT_DOSSIER_RECENT_RUNS) -> dict:
    as_of = race_input.prediction_cutoff
    ctx = build_context(race_input, repo)
    conditions = (repo.race_conditions(race_input.race_id, as_of=as_of)
                  if race_input.race_id is not None else None)
    class_level = conditions.class_level if conditions is not None else None
    identification = course_mod.identify(race_input, conditions)
    facts = course_mod.facts_for(identification, race_input.date)
    today = history.TodayConditions(
        track_kind=race_input.track_kind, kyori_m=race_input.kyori_m,
        jyo_cd=race_input.jyo_cd, track_cd=identification.track_cd,
        baba_cd=race_input.baba_cd, class_level=class_level)

    horses: dict[str, dict] = {}
    statuses: Counter[str] = Counter()
    n_front = 0
    for hf in ctx.horses:
        runs = repo.past_runs(hf.ketto_num, as_of=as_of) if hf.ketto_num else []
        # 取消・除外（と障害走）は走っていない／比較できないので、recent_runs・
        # conditions と同じ出走集合で判定する。生の past_runs を渡すと、取消だけの
        # 馬が found / runs_found 1 になり recent_runs は空という矛盾が出る
        # （final review I-2）。
        status = history.history_status(eligible_runs(runs),
                                        matched=hf.ketto_num is not None,
                                        race_class_level=class_level,
                                        coverage_from=repo.coverage_from)
        statuses[status.status] += 1
        mean_pos = hf.running_style.corner_pos_mean
        if mean_pos is not None and mean_pos <= config.FRONT_RUNNER_MAX_CORNER_POS:
            n_front += 1
        horses[hf.umaban] = {
            "bamei": hf.entry.bamei,
            "sex_age": hf.entry.sex_age,
            "futan_kg": hf.entry.futan_kg,
            "waku": hf.entry.wakuban,
            "kisyu": hf.entry.kisyu,
            "history": {
                # note はここに置かない。出現した status ぶんだけ top-level の
                # history_status_notes に集約する（statesize-fix-brief.md）。
                "status": status.status,
                "runs_found": status.runs_found,
                "period": {"from": _iso(status.period_from), "to": _iso(status.period_to)},
            },
            "recent_runs": history.recent_run_lines(runs, as_of=as_of, limit=max_recent_runs),
            "vs_last_run": {
                "days_since_last_run": hf.recent.days_since_last_run,
                "distance_change_m": hf.recent.distance_change_m,
                "futan_change_kg": hf.recent.futan_change_kg,
                "class_level_change": hf.recent.class_level_change,
            },
            "conditions": _conditions_dict(history.combined_condition_records(runs, today=today)),
            "ability": ability_dict(hf.ability, hf.field_rank),
            "running_style": running_style_dict(hf.running_style),
        }

    race = dict(race_dict(race_input))
    race["class_level"] = class_level
    quality = dict(data_quality(ctx))
    quality["coverage_from"] = _iso(repo.coverage_from)
    return {
        "dossier_version": config.DOSSIER_VERSION,
        "prediction_cutoff": as_of.isoformat(),
        "course": _course_dict(identification, facts),
        "race": race,
        "horses": horses,
        "condition_definitions": _condition_definitions(today),
        "history_status_notes": {status: history.STATUS_NOTES[status]
                                 for status in sorted(statuses)},
        "field_summary": {
            "n_horses": len(horses),
            "n_with_history": sum(1 for h in horses.values() if h["history"]["runs_found"] > 0),
            "history_status_counts": dict(sorted(statuses.items())),
            "n_front_runners": n_front,
            "front_runner_definition": (
                "近走の4角の相対位置（通過順÷頭数）の平均が "
                f"{_threshold_text(config.FRONT_RUNNER_MAX_CORNER_POS)} 以下の馬"),
        },
        "data_quality": quality,
    }


def to_state(dossier: dict, *, pace_forecast: dict | None = None) -> dict:
    """Jev に送る state。資料をコピーし、案Bなら pace_forecast を足す。"""
    state = copy.deepcopy(dossier)
    if pace_forecast is not None:
        state["pace_forecast"] = pace_forecast
    return state


def state_chars(state: dict) -> int:
    return len(json.dumps(state, ensure_ascii=False))


def check_state_size(state: dict, *, limit: int = config.STATE_MAX_CHARS) -> int:
    """上限以内なら文字数を返す。超えたら StateTooLargeError。黙って削らない。"""
    chars = state_chars(state)
    if chars > limit:
        raise StateTooLargeError(chars, limit)
    return chars

"""``PredictionContext`` を Jev へ送る JSON-safe な dict（state）に変換する。

ここでは I/O も特徴量の計算もしない。計算はすべて ``context.py`` 側で終わって
いる前提で、辞書の形に整えるだけ。``datetime`` / ``date`` は文字列に変換し、
``json.dumps(state, ensure_ascii=False)`` がそのまま通る形にする（``default=``
は使わない——想定していない型が紛れ込んだら、隠さずに落ちてほしい）。

このモジュールは2つの不変条件を最後まで守る場所でもある。

1. ``horses`` の各値が全馬で同じキー集合を持つこと（spec §5.1「同じ書式で
   入れる」）。``_horse_full_dict`` が常に同じ形の辞書を返すことで保証する。
2. 対象レースの結果や、``as_of`` より後のレースの情報が一切混ざらないこと。
   型（``EntryRow`` / ``ResultRow`` の分離）と ``Repository`` の ``as_of``
   ゲートが最初の防御線で、ここでの文字列検索テスト（test_summary.py）が
   最後の防御線になる。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from keiba.features.ability import AbilitySummary
from keiba.features.context import HorseFeatures, PredictionContext
from keiba.features.derive import ConditionRecord, RecentForm, RunningStyle, Transport
from keiba.models import RaceInput

# 輸送距離は「代表的な陸路距離の概数」であって、実際の出発地・経路・輸送日・
# 現地滞在を示す測定値ではない。state 自身にこの注記を持たせないと、モデルが
# 概数を実測値として読んでしまう（元仕様 §4.2）。config.py にはまだこの注記が
# 無いため、ここに定数として置く。
TRANSPORT_ESTIMATE_NOTE = (
    "トレセンから競馬場までの代表的な陸路距離の概数。"
    "実際の出発地・移動経路・輸送日・現地滞在を示すものではない。"
)
PAST_TRANSPORT_NOTE = (
    "過去に推定輸送距離が長かった走での馬体重増減。"
    "体重変化の原因が輸送であることを示すものではない。"
)


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _period_dict(period_from: date | None, period_to: date | None) -> dict:
    return {"from": _iso(period_from), "to": _iso(period_to)}


def _ability_dict(ability: AbilitySummary, field_rank: float | None) -> dict:
    return {
        "recent_margin_sec_wavg": ability.recent_margin_sec_wavg,
        "runs_used": ability.runs_used,
        "recent_class_level_wavg": ability.recent_class_level_wavg,
        "best_class_level": ability.best_class_level,
        "field_rank": field_rank,
        # margin_period は着差の加重平均に寄与した走の範囲だけを表す
        # （Task 9: AbilitySummary.period_from/to の定義）。
        # recent_class_level_wavg / best_class_level は同じ窓でも「対象走に
        # なった」走全体（時計の有無を問わない）が範囲なので、これより広い
        # ことがある。単に period_from/to と名付けると両者を混同するので、
        # margin（着差）に限定した集計であることをキー名に明示する。
        "margin_period": _period_dict(ability.period_from, ability.period_to),
    }


def _condition_record_dict(record: ConditionRecord) -> dict:
    """6キーを常に出す。``inapplicable_reason`` が文字列のときは starts/wins/
    top3/period も ``None``（その条件が今回のレースについて定義できないため。
    derive.ConditionRecord の docstring 参照）。キーの有無では分岐させない
    ——プロンプトは state に存在するキーしか参照してはいけないので、
    ``same_baba`` を含む5区分すべてで同じ6キーの形を保つ（spec §8.2）。
    """
    return {
        "starts": record.starts,
        "wins": record.wins,
        "top3": record.top3,
        "margin_sec_mean": record.margin_sec_mean,
        "period": _period_dict(record.period_from, record.period_to),
        "inapplicable_reason": record.inapplicable_reason,
    }


def _by_condition_dict(by_condition: dict[str, ConditionRecord]) -> dict:
    return {key: _condition_record_dict(record) for key, record in by_condition.items()}


def _recent_dict(recent: RecentForm) -> dict:
    return {
        "days_since_last_run": recent.days_since_last_run,
        "distance_change_m": recent.distance_change_m,
        "futan_change_kg": recent.futan_change_kg,
        "class_level_change": recent.class_level_change,
        "last_run": recent.last_run,
    }


def _running_style_dict(style: RunningStyle) -> dict:
    return {
        "corner_pos_mean": style.corner_pos_mean,
        "corner_pos_stdev": style.corner_pos_stdev,
        "corner_pos_starts": style.corner_pos_starts,
        "corner_source": style.corner_source,
    }


def _transport_dict(transport_: Transport, base_source: str | None) -> dict:
    return {
        "base": transport_.base,
        # base がどこから来たか（"entry": 今回の出馬表 / "last_run": 直近走から
        # の推定 / None: どちらも無い）。"last_run" は確認済みの情報ではなく
        # 推定なので、state 上でも見分けられるようにする（spec §4.2）。
        "base_source": base_source,
        "estimated_transport_km": transport_.estimated_transport_km,
        "estimated_note": TRANSPORT_ESTIMATE_NOTE,
        "stay_status": transport_.stay_status,
        "past_long_transport": {
            "starts": transport_.past_long_transport_starts,
            "zogen_mean": transport_.past_long_transport_zogen_mean,
            "zogen_min": transport_.past_long_transport_zogen_min,
            "note": PAST_TRANSPORT_NOTE,
        },
    }


def _horse_full_dict(features: HorseFeatures) -> dict:
    """第1層 ``horses[umaban]`` / 第2層 ``target_horse`` の形。

    値が ``None`` でもキーは必ず入れる。馬ごとに持っているキーが違うと、
    比較を頼んでいるモデルには「別の書式」に見えてしまう（spec §5.1）。
    """
    entry = features.entry
    return {
        "bamei": entry.bamei,
        "sex_age": entry.sex_age,
        "futan_kg": entry.futan_kg,
        "waku": entry.wakuban,
        "kisyu": entry.kisyu,
        "ability": _ability_dict(features.ability, features.field_rank),
        "recent": _recent_dict(features.recent),
        "by_condition": _by_condition_dict(features.by_condition),
        "running_style": _running_style_dict(features.running_style),
        "transport": _transport_dict(features.transport, features.transport_base_source),
    }


def _horse_short_dict(features: HorseFeatures) -> dict:
    """``opponents_summary`` の1頭分。比較材料を能力指標と脚質だけに絞った
    簡略版（spec §6.1「相手全頭の短い比較情報」）。"""
    return {
        "bamei": features.entry.bamei,
        "ability": _ability_dict(features.ability, features.field_rank),
        "running_style": _running_style_dict(features.running_style),
    }


def _race_dict(race: RaceInput) -> dict:
    return {
        "jyo": race.jyo,
        "track_kind": race.track_kind,
        "kyori_m": race.kyori_m,
        "baba": race.baba,
        "class_name": race.class_name,
        # RaceRow.syusso_tosu（結果から算出される出走頭数）とは別物で、対象
        # レースはまだ走っていないので出せない。利用者が申告した出走予定頭数
        # をそのまま渡す（models.py の RaceInput.entry_count のコメント参照）。
        "entry_count": race.entry_count,
        "hasso_at": _iso(race.hasso_at),
    }


def data_quality(ctx: PredictionContext, umabans: Sequence[str] | None = None) -> dict:
    """欠損・版情報をまとめる。``umabans`` を渡すと、その馬だけに絞って集計する。

    ``notes`` はレース単位の注記だけ、``horse_notes`` は馬単位の注記を
    馬番ごとに持つ。平坦に連結すると注記が誰についてのものか分からなくなり、
    利用者向けの表示で「表示する馬だけに絞る」ができなくなる
    （fix4 で見つかった欠陥。前ラウンドは3頭分の重複を12頭分の重複に
    付け替えただけだった）。``missing`` と同じ約束で、注記の無い馬は
    ``horse_notes`` にキーとして現れない（「0件」と「不在」は違う値だが、
    「言うことが無い」を空リストではなくキー不在で表す既存の約束を踏襲する）。
    """
    horses = ctx.horses if umabans is None else tuple(ctx.horse(u) for u in umabans)
    missing = {h.umaban: list(h.missing) for h in horses if h.missing}
    horse_notes = {h.umaban: list(h.notes) for h in horses if h.notes}
    return {
        "as_of": ctx.race.prediction_cutoff.isoformat(),
        "data_version": ctx.data_version,
        "data_fetched_at": ctx.data_fetched_at.isoformat(),
        "missing": missing,
        "notes": list(ctx.notes),
        "horse_notes": horse_notes,
    }


def horse_criteria(ctx: PredictionContext) -> dict[str, str]:
    """第1層 ``winner``（``Choice``）の ``criteria``。馬番 → 短い識別情報。"""
    criteria: dict[str, str] = {}
    for features in ctx.horses:
        entry = features.entry
        parts = [entry.bamei]
        if entry.sex_age:
            parts.append(entry.sex_age)
        if entry.futan_kg is not None:
            parts.append(f"{entry.futan_kg}kg")
        criteria[features.umaban] = " ".join(parts)
    return criteria


def build_race_state(ctx: PredictionContext) -> dict:
    """第1層の state（spec §5.1）。必須キーは ``prediction_cutoff`` / ``race`` /
    ``horses`` / ``data_quality`` の4つ。

    ``pace_definition`` は pace 質問を評価するときだけ足す（``race_context`` と
    同じ理由でキーごと省く。§8.2 の pace プロンプトは「区分の意味は
    pace_definition に従ってください」と指示するので、質問を出す以上は
    state 側にも定義を載せないと、モデルが自分で閾値を作ってしまう）。
    """
    state = {
        "prediction_cutoff": ctx.race.prediction_cutoff.isoformat(),
        "race": _race_dict(ctx.race),
        "horses": {features.umaban: _horse_full_dict(features) for features in ctx.horses},
        "data_quality": data_quality(ctx),
    }
    if ctx.pace_definition is not None:
        state["pace_definition"] = ctx.pace_definition
    return state


def build_horse_state(ctx: PredictionContext, umaban: str, *,
                       race_context: dict | None = None) -> dict:
    """第2層の state（spec §6.1）。

    ``race_context`` は pace 質問が OFF（未評価）のとき丸ごと省く。キーが無い
    ことで「評価していない」を表す——``None`` を入れると、無いはずのペース
    情報についてモデルが推論してしまう。
    """
    target = ctx.horse(umaban)
    state = {
        "prediction_cutoff": ctx.race.prediction_cutoff.isoformat(),
        "race": _race_dict(ctx.race),
        "target_horse": _horse_full_dict(target),
        # 対象馬自身も含める。相手の中での自分の位置が分かるように。
        "opponents_summary": {
            features.umaban: _horse_short_dict(features) for features in ctx.horses
        },
        "data_quality": data_quality(ctx),
    }
    if race_context is not None:
        state["race_context"] = race_context
    return state


# features/dossier.py が同じ書式で出すための公開名。書式はここ1箇所で持つ。
ability_dict = _ability_dict
running_style_dict = _running_style_dict
race_dict = _race_dict

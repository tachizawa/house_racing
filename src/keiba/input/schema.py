"""JSON 由来の入力を RaceInput / HorseEntry に組み立てる。

ここが引き受けるのは「理解できるか」だけ。必須項目の欠落、知らない開催場、
ISO 8601 でない日時など、理解できない入力は ``InputError`` で止める。

値の意味的な妥当性（距離が非現実的、馬番が重複している、など）は
``validate.check()`` の仕事であり、ここでは扱わない。理解はできたが中身が
おかしい、という状態を区別するため。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from keiba import config
from keiba.models import HorseEntry, RaceInput

__all__ = [
    "InputError",
    "load_race_input",
    "parse_race_input",
    "resolve_baba_cd",
    "resolve_jyo_cd",
]

# 表示名 → コードの逆引き。config.JYO_CD_TO_NAME / BABA_CD_TO_NAME を単一の正とし、
# 逆引き辞書はここで都度組み立てる（config.py に足すと2つの表が食い違いうる）。
_JYO_NAME_TO_CD: dict[str, str] = {name: cd for cd, name in config.JYO_CD_TO_NAME.items()}
_BABA_NAME_TO_CD: dict[str, str] = {name: cd for cd, name in config.BABA_CD_TO_NAME.items()}

# 初期版は平地のみを対象にする（RaceRow.is_flat と同じ判断）。障害は対象外。
_VALID_TRACK_KINDS = ("turf", "dirt")

_REQUIRED_RACE_FIELDS = (
    "date",
    "jyo",
    "race_num",
    "track_kind",
    "kyori_m",
    "entry_count",
    "prediction_cutoff",
    "horses",
)
_REQUIRED_HORSE_FIELDS = ("umaban", "bamei")


class InputError(ValueError):
    """入力を理解できなかったときに送出する。"""


def resolve_jyo_cd(value: str) -> str:
    """開催場の表示名またはコードをコードに解決する。中央10場のみ扱う。

    近い表記を推測で補うことはしない。解決できなければ ``InputError``。
    """
    if value in config.JYO_CD_TO_NAME:
        return value
    if value in _JYO_NAME_TO_CD:
        return _JYO_NAME_TO_CD[value]
    raise InputError(f"開催場を解決できません: {value!r}（中央10場の名称またはコードのみ受け付けます）")


def resolve_baba_cd(value: str | None) -> str | None:
    """馬場状態の表示名またはコードをコードに解決する。未指定は ``None`` のまま通す。

    「やや重」のような近い表記を「稍重」とみなすことはしない。解決できなければ
    ``InputError``。良馬場などの既定値に寄せることもしない。
    """
    if value is None:
        return None
    if value in config.BABA_CD_TO_NAME:
        return value
    if value in _BABA_NAME_TO_CD:
        return _BABA_NAME_TO_CD[value]
    raise InputError(f"馬場状態を解決できません: {value!r}（表記を推測して補うことはしません）")


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise InputError(f"{field} は ISO 8601 の文字列である必要があります: {value!r}")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise InputError(f"{field} を日時として解釈できません: {value!r} ({error})") from error


# --- 型境界: 量とラベルは別の扱いをする ---------------------------------------
#
# 量（kyori_m / entry_count / race_num / top_n / futan_kg）は「見た目が違っても
# 意味が一意なら理解する」。"2000" と書かれた距離は 2000 以外に読みようがないので
# 変換する。一方、識別子・ラベル（umaban / bamei / wakuban / kisyu / jyo /
# class_name / track_kind）は他の同種の値と比較・突き合わせに使われるため、
# 見た目を揃えて理解してあげることはしない。1 と "1" は別物として扱う。


def _coerce_int(value: Any, *, field: str) -> int:
    """量の項目を int にする。曖昧でない表現だけを受け付ける。

    bool は int のサブクラスなので isinstance だけでは弾けない
    （isinstance(True, int) is True）。先に type で除外する。
    非整数の float は切り捨てず拒否する。切り捨ては値を捨てる推測であり、
    ここでは行わない。
    """
    if type(value) is bool:
        raise InputError(f"{field} は数値である必要があります（真偽値は不可）: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise InputError(
                f"{field} は整数である必要があります（小数点以下を切り捨てて解決することはしません）: {value!r}"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise InputError(f"{field} を整数として解釈できません: {value!r} ({error})") from error
    raise InputError(f"{field} の型が不正です（整数または整数を表す文字列のみ受け付けます）: {value!r}")


def _coerce_float(value: Any, *, field: str) -> float:
    """量の項目を float にする。``_coerce_int`` と同じ方針。"""
    if type(value) is bool:
        raise InputError(f"{field} は数値である必要があります（真偽値は不可）: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise InputError(f"{field} を数値として解釈できません: {value!r} ({error})") from error
    raise InputError(f"{field} の型が不正です（数値または数値を表す文字列のみ受け付けます）: {value!r}")


def _require_str(value: Any, *, field: str) -> str:
    """識別子・ラベルの項目。文字列以外は理由を示して拒否する（黙って変換しない）。"""
    if not isinstance(value, str):
        raise InputError(
            f"{field} は文字列である必要があります: {value!r}"
            f"（数値に見えても引用符で囲んで文字列にしてください）"
        )
    return value


def _parse_horse(payload: Any) -> HorseEntry:
    if not isinstance(payload, dict):
        raise InputError(f"horses の要素は dict である必要があります: {payload!r}")
    for field in _REQUIRED_HORSE_FIELDS:
        if field not in payload or payload[field] is None:
            raise InputError(f"馬の必須項目が指定されていません: {field}")

    umaban = _require_str(payload["umaban"], field="umaban")
    bamei = _require_str(payload["bamei"], field="bamei")

    wakuban_raw = payload.get("wakuban")
    wakuban = _require_str(wakuban_raw, field="wakuban") if wakuban_raw is not None else None

    kisyu_raw = payload.get("kisyu")
    kisyu = _require_str(kisyu_raw, field="kisyu") if kisyu_raw is not None else None

    futan_kg_raw = payload.get("futan_kg")
    futan_kg = _coerce_float(futan_kg_raw, field="futan_kg") if futan_kg_raw is not None else None

    return HorseEntry(
        umaban=umaban,
        bamei=bamei,
        wakuban=wakuban,
        sex_age=payload.get("sex_age"),
        futan_kg=futan_kg,
        kisyu=kisyu,
    )


def parse_race_input(payload: dict) -> RaceInput:
    """JSON 由来の dict から ``RaceInput`` を組み立てる。

    ``jyo`` / ``baba`` は表示名・コードのどちらで渡されても解決し、
    表示名 (``jyo`` / ``baba``) とコード (``jyo_cd`` / ``baba_cd``) の
    両方を埋める。呼び出し側がどちらの形式で入力したかを気にしなくてよいように。
    """
    if not isinstance(payload, dict):
        raise InputError(f"payload は dict である必要があります: {type(payload).__name__}")

    for field in _REQUIRED_RACE_FIELDS:
        if field not in payload or payload[field] is None:
            raise InputError(f"必須項目が指定されていません: {field}")

    track_kind = _require_str(payload["track_kind"], field="track_kind")
    if track_kind not in _VALID_TRACK_KINDS:
        raise InputError(
            f"track_kind が不正です: {track_kind!r}（{'/'.join(_VALID_TRACK_KINDS)} のみ受け付けます）"
        )

    jyo_raw = _require_str(payload["jyo"], field="jyo")
    jyo_cd = resolve_jyo_cd(jyo_raw)

    baba_raw = payload.get("baba")
    if baba_raw is not None:
        baba_raw = _require_str(baba_raw, field="baba")
    baba_cd = resolve_baba_cd(baba_raw)

    class_name_raw = payload.get("class_name")
    class_name = _require_str(class_name_raw, field="class_name") if class_name_raw is not None else None

    if not isinstance(payload["date"], str):
        raise InputError(f"date は ISO 8601 の文字列である必要があります: {payload['date']!r}")
    try:
        race_date = date.fromisoformat(payload["date"])
    except ValueError as error:
        raise InputError(f"date を日付として解釈できません: {payload['date']!r} ({error})") from error

    prediction_cutoff = _parse_datetime(payload["prediction_cutoff"], field="prediction_cutoff")
    hasso_at_raw = payload.get("hasso_at")
    hasso_at = _parse_datetime(hasso_at_raw, field="hasso_at") if hasso_at_raw is not None else None

    kyori_m = _coerce_int(payload["kyori_m"], field="kyori_m")
    entry_count = _coerce_int(payload["entry_count"], field="entry_count")
    race_num = _coerce_int(payload["race_num"], field="race_num")
    top_n = _coerce_int(payload.get("top_n", config.DEFAULT_TOP_N), field="top_n")

    horses = tuple(_parse_horse(h) for h in payload["horses"])

    return RaceInput(
        date=race_date,
        jyo=config.JYO_CD_TO_NAME[jyo_cd],
        race_num=race_num,
        track_kind=track_kind,
        kyori_m=kyori_m,
        entry_count=entry_count,
        prediction_cutoff=prediction_cutoff,
        horses=horses,
        jyo_cd=jyo_cd,
        baba_cd=baba_cd,
        class_name=class_name,
        baba=config.BABA_CD_TO_NAME[baba_cd] if baba_cd is not None else None,
        hasso_at=hasso_at,
        top_n=top_n,
        race_id=payload.get("race_id"),
    )


def load_race_input(path: Path) -> RaceInput:
    """``.json`` ファイルを読んで ``RaceInput`` を組み立てる。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise InputError(f"入力ファイルを読めません: {path} ({error})") from error
    except UnicodeDecodeError as error:
        raise InputError(f"入力ファイルが UTF-8 ではありません: {path} ({error})") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise InputError(f"JSON として解釈できません: {path} ({error})") from error
    return parse_race_input(payload)

"""コースの識別。

「新潟・芝・2000m」には内回りと外回りがあり一意に決まらない（report3 §3.2）。
JV-Data のトラックコード（2009）と競馬場・距離の組でコースを特定する。

ここで決めるのは「どのコースか」だけ。直線の長さや坂などの事実は
course_facts.json（JRA 公式ページから写したもの）から引く（Task 3）。
「直線が長いので差し有利」のような予測はどこにも書かない（report3 §4.2）。

トラックコードの対応は vendor/jravan-sdk-500-64bit/ドキュメント/
JV-Data仕様書_4.9.0.1.xlsx「コード表」2009 から写した。略名もそのまま。
"""

from __future__ import annotations

from dataclasses import dataclass

from keiba import config
from keiba.models import RaceConditions, RaceInput

__all__ = ["CourseIdentification", "TrackCode", "decode_track_cd", "identify"]


@dataclass(frozen=True)
class TrackCode:
    surface: str            # "turf" / "dirt"（サンドは config.track_kind に従い dirt）
    direction: str | None   # "left" / "right" / "straight"
    label: str              # JV-Data の略名（例: "芝・右外"）


# 平地のみ。障害（51-59）は予想対象外なので None を返す。
_TRACK_CODES: dict[str, tuple[str, str]] = {
    "10": ("straight", "芝・直"),
    "11": ("left", "芝・左"),
    "12": ("left", "芝・左外"),
    "13": ("left", "芝・左内→外"),
    "14": ("left", "芝・左外→内"),
    "15": ("left", "芝・左内２周"),
    "16": ("left", "芝・左外２周"),
    "17": ("right", "芝・右"),
    "18": ("right", "芝・右外"),
    "19": ("right", "芝・右内→外"),
    "20": ("right", "芝・右外→内"),
    "21": ("right", "芝・右内２周"),
    "22": ("right", "芝・右外２周"),
    "23": ("left", "ダート・左"),
    "24": ("right", "ダート・右"),
    "25": ("left", "ダート・左内"),
    "26": ("right", "ダート・右外"),
    "27": ("left", "サンド・左"),
    "28": ("right", "サンド・右"),
    "29": ("straight", "ダート・直"),
}


def decode_track_cd(track_cd: str) -> TrackCode | None:
    code = (track_cd or "").strip()
    if code not in _TRACK_CODES:
        return None
    surface = config.track_kind(code)
    if surface not in ("turf", "dirt"):
        return None
    direction, label = _TRACK_CODES[code]
    return TrackCode(surface=surface, direction=direction, label=label)


@dataclass(frozen=True)
class CourseIdentification:
    identified: bool
    jyo_cd: str | None
    track_cd: str | None
    kyori_m: int
    track: TrackCode | None
    course_kubun_cd: str | None
    #: identified が False のときの理由。True なら None。
    reason: str | None


def identify(race_input: RaceInput, conditions: RaceConditions | None) -> CourseIdentification:
    def not_identified(reason: str) -> CourseIdentification:
        return CourseIdentification(
            identified=False, jyo_cd=race_input.jyo_cd, track_cd=None,
            kyori_m=race_input.kyori_m, track=None, course_kubun_cd=None, reason=reason)

    if conditions is None or not conditions.track_cd:
        return not_identified(
            "対象レースのトラックコードがデータ版に無いため、内外回りなどのコースを特定できない。")
    track = decode_track_cd(conditions.track_cd)
    if track is None:
        return not_identified(f"トラックコード {conditions.track_cd!r} は平地のコースとして解読できない。")
    if track.surface != race_input.track_kind:
        return not_identified(
            f"入力の芝・ダート（{race_input.track_kind}）とトラックコード"
            f"（{track.label}）が食い違うため、コースを特定しない。")
    return CourseIdentification(
        identified=True, jyo_cd=race_input.jyo_cd, track_cd=conditions.track_cd,
        kyori_m=race_input.kyori_m, track=track,
        course_kubun_cd=(conditions.course_kubun_cd or "").strip() or None, reason=None)

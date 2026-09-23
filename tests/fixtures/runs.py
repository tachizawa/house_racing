"""Task 9（能力指標）と Task 10（条件別成績・近況・脚質・輸送）のテスト用に、
本物の PastRun を組み立てる。

ability.py が見る項目に加えて、Task 10 が必要とする tozai_cd（輸送の起点）・
jyuni_3c / jyuni_2c（脚質のフォールバック元）・ba_taijyu_kg（前走の馬体重）も
パラメータ化した。型を通すために RaceRow / ResultRow / EntryRow の全項目を
埋める。計算に関係ない項目は固定のダミー値にしてある。
"""

from __future__ import annotations

from datetime import datetime

from keiba.models import EntryRow, PastRun, RaceRow, ResultRow

_TRACK_CD = {"turf": "17", "dirt": "24", "jump": "51"}


def make_past_run(
    *,
    hasso_at: datetime,
    kyori_m: int = 2000,
    track_kind: str | None = "turf",
    jyo_cd: str = "06",
    baba_cd: str = "1",
    class_level: int | None = 4,
    margin_sec: float | None = 0.3,
    kakutei_jyuni: int | None = 3,
    ijyo_cd: str = "0",
    jyuni_2c: int | None = None,
    jyuni_3c: int | None = None,
    jyuni_4c: int | None = 5,
    syusso_tosu: int | None = 12,
    zogen_sa_kg: int | None = -2,
    futan_kg: float | None = 57.0,
    tozai_cd: str = "1",
    ba_taijyu_kg: int | None = None,
) -> PastRun:
    entry = EntryRow(
        umaban="1",
        wakuban="1",
        ketto_num="0000000001",
        bamei="テストウマ",
        sex_cd="1",
        barei="04",
        futan_kg=futan_kg,
        kisyu_code="00001",
        tozai_cd=tozai_cd,
        chokyosi_code="00001",
    )
    race = RaceRow(
        race_id=f"{hasso_at:%Y%m%d%H%M}{jyo_cd}",
        hasso_at=hasso_at,
        jyo_cd=jyo_cd,
        kyori_m=kyori_m,
        track_cd=_TRACK_CD.get(track_kind or "", "17"),
        track_kind=track_kind,
        baba_cd=baba_cd,
        class_level=class_level,
        grade_cd=" ",
        jyoken_cd_youngest="016",
        toroku_tosu=syusso_tosu,
        syusso_tosu=syusso_tosu,
        nyusen_tosu=syusso_tosu,
        first_3f_sec=35.2,
        race_name="テストレース",
    )
    result = ResultRow(
        entry=entry,
        ijyo_cd=ijyo_cd,
        kakutei_jyuni=kakutei_jyuni,
        dochaku_kubun="0",
        dochaku_tosu=None,
        time_sec=None,
        margin_sec=margin_sec,
        jyuni_1c=None,
        jyuni_2c=jyuni_2c,
        jyuni_3c=jyuni_3c,
        jyuni_4c=jyuni_4c,
        last_3f_sec=None,
        ba_taijyu_kg=ba_taijyu_kg,
        zogen_sa_kg=zogen_sa_kg,
        odds=None,
        ninki=None,
    )
    return PastRun(race=race, result=result)

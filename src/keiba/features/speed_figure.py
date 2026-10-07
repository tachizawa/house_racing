"""タイム指数。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md §3。

1走ごとに「基準タイム + その日の馬場差 − 走破タイム」を 1600m 相当の秒にする（正ほど速い）。
基準タイム・クラス差・馬場差は**古馬戦（3歳以上・4歳以上）の1着だけ**から推定する。
2歳戦・3歳戦を混ぜると、世代の強弱が「その日の馬場が速かった／遅かった」として馬場差に
吸収され、世代差が消える（§3.4）。

DataKubun の選び方は repository.py の `_pick_by_data_kubun` をそのまま使う。選び方を
2か所に持つと、片方だけ直したときに「確定前の値を確定値として使う」事故が起きるため。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from keiba import config
from keiba.data.repository import _pick_by_data_kubun

JRA_JYO = frozenset(f"{i:02d}" for i in range(1, 11))
FLAT = ("turf", "dirt")
OLDER_SYUBETU = frozenset({"13", "14"})   # 3歳以上・4歳以上
KEY = ["jyo_cd", "track_cd", "kyori_m"]    # 基準タイムの単位（track_cd は芝ダと内外回りを含む）
FACE = ["day", "jyo_cd", "track_kind"]     # 馬場差の単位（開催面）
RUN_COLUMNS = ["race_id", "umaban", "ketto_num", "day", "date", "year", "jyo_cd", "track_cd",
               "track_kind", "kyori_m", "syubetu_cd", "class_level", "time_sec", "rank",
               "barei", "cohort"]
# 取消・除外（走っていない）と中止・失格（走破タイムが使えない）は指数の対象にしない
_EXCLUDED_IJYO = config.IJYO_EXCLUDED | config.IJYO_NO_TIME
_RACE_FIELDS = ["race_id", "jyo_cd", "track_cd", "track_kind", "kyori_m", "syubetu_cd", "class_level"]


def prepare_runs(races: pd.DataFrame, race_horses: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """指数の対象になる走を1行ずつ返す（§3.1）。発走が as_of より前の確定成績だけ。

    同時刻に発走したレースは含めない（Repository と同じく「その時点でまだ結果を知り得ない」）。
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of にはタイムゾーンが必要です（日本時間の 0:00 など）")
    r = _pick_by_data_kubun(races, config.DATA_KUBUN_RESULT_PRIORITY)
    r = r[r["jyo_cd"].isin(JRA_JYO) & r["track_kind"].isin(FLAT)]
    hasso = pd.to_datetime(r["hasso_at"], errors="coerce", utc=True)
    r = r[hasso < pd.Timestamp(as_of).tz_convert("UTC")]
    h = race_horses[race_horses["race_id"].isin(r["race_id"])]
    h = _pick_by_data_kubun(h, config.DATA_KUBUN_RESULT_PRIORITY)
    h = h[~h["result_ijyo_cd"].isin(_EXCLUDED_IJYO)]
    m = h.merge(r[_RACE_FIELDS], on="race_id", how="inner")
    m = m.assign(time_sec=pd.to_numeric(m["result_time_sec"], errors="coerce"),
                 kyori=pd.to_numeric(m["kyori_m"], errors="coerce"),
                 age=pd.to_numeric(m["barei"], errors="coerce"))
    m = m[(m["time_sec"] > 0) & m["kyori"].notna() & m["age"].notna()]   # NaN の比較は False なので落ちる
    day = m["race_id"].str[:8]
    year = day.str[:4].astype(int)
    out = pd.DataFrame({
        "race_id": m["race_id"], "umaban": m["umaban"], "ketto_num": m["ketto_num"],
        "day": day, "date": pd.to_datetime(day, format="%Y%m%d"), "year": year,
        "jyo_cd": m["jyo_cd"], "track_cd": m["track_cd"], "track_kind": m["track_kind"],
        "kyori_m": m["kyori"].astype(int), "syubetu_cd": m["syubetu_cd"],
        "class_level": pd.to_numeric(m["class_level"], errors="coerce"),
        "time_sec": m["time_sec"].astype(float),
        "rank": pd.to_numeric(m["result_kakutei_jyuni"], errors="coerce"),
        "barei": m["age"].astype(int), "cohort": year - m["age"].astype(int),
    })
    return out[RUN_COLUMNS].reset_index(drop=True)

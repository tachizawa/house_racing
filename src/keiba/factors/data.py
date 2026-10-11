"""評価に使う「走」の表（仕様 §5.1）。1行 = 1頭の1走（確定成績、JRA10場の平地）。

予想に使う列（騎手・枠・斤量・オッズなど）と答え合わせに使う列（着順・タイム）が同じ行にある。
各要素はその行の日より前の行の結果だけを使う（cumulative.py とテストで守る）。
DataKubun の選び方は repository.py の `_pick_by_data_kubun` を使う（選び方を2か所に持たないため）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from keiba import config
from keiba.data.repository import _pick_by_data_kubun

JRA_JYO = frozenset(f"{i:02d}" for i in range(1, 11))
REGION = {"01": "北海道", "02": "北海道", "03": "東", "04": "東", "05": "東", "06": "東",
          "07": "西", "08": "西", "09": "西", "10": "小倉"}
RUN_COLUMNS = ["race_id", "date", "year", "umaban", "waku", "ketto_num", "barei", "cohort", "futan", "kisyu",
               "chokyosi", "tozai", "sire_num", "jyuni", "field", "top3", "fin_pct", "odds", "ninki", "time_sec",
               "zogen", "jyo", "track_cd", "track_kind", "kyori", "class_level", "grade", "syubetu"]
_RACE_FIELDS = ["race_id", "jyo_cd", "track_cd", "track_kind", "kyori_m", "class_level", "grade_cd", "syubetu_cd"]


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()


def build_runs(races: pd.DataFrame, race_horses: pd.DataFrame, pedigrees: pd.DataFrame) -> pd.DataFrame:
    """取消・除外（走っていない）は除き、中止・失格は出走として残す（着順 0 → 3着内ではない）。"""
    r = _pick_by_data_kubun(races, config.DATA_KUBUN_RESULT_PRIORITY)
    r = r[r["jyo_cd"].isin(JRA_JYO) & r["track_kind"].isin(("turf", "dirt"))]
    h = race_horses[race_horses["race_id"].isin(r["race_id"])]
    h = _pick_by_data_kubun(h, config.DATA_KUBUN_RESULT_PRIORITY)
    h = h[~h["result_ijyo_cd"].isin(config.IJYO_EXCLUDED)]
    m = h.merge(r[_RACE_FIELDS], on="race_id")
    ped = pedigrees[["ketto_num", "sire_num"]].drop_duplicates("ketto_num", keep="last")
    m = m.merge(ped, on="ketto_num", how="left")
    out = pd.DataFrame({
        "race_id": m["race_id"].to_numpy(),
        "date": pd.to_datetime(m["race_id"].str[:8], format="%Y%m%d").to_numpy(),
        "umaban": m["umaban"].to_numpy(), "waku": _num(m["wakuban"]), "ketto_num": m["ketto_num"].to_numpy(),
        "barei": _num(m["barei"]), "futan": _num(m["futan_kg"]), "kisyu": m["kisyu_code"].to_numpy(),
        "chokyosi": m["chokyosi_code"].to_numpy(), "tozai": m["tozai_cd"].to_numpy(),
        "sire_num": m["sire_num"].fillna("").to_numpy(),
        "jyuni": _num(m["result_kakutei_jyuni"]), "odds": _num(m["result_odds"]), "ninki": _num(m["result_ninki"]),
        "time_sec": _num(m["result_time_sec"]), "zogen": _num(m["result_zogen_sa_kg"]),
        "jyo": m["jyo_cd"].to_numpy(), "track_cd": m["track_cd"].to_numpy(), "track_kind": m["track_kind"].to_numpy(),
        "kyori": _num(m["kyori_m"]), "class_level": _num(m["class_level"]), "grade": m["grade_cd"].to_numpy(),
        "syubetu": m["syubetu_cd"].to_numpy(),
    })
    out = out[~np.isnan(out["barei"].to_numpy(float)) & ~np.isnan(out["kyori"].to_numpy(float))].copy()
    out["barei"] = out["barei"].astype(int)
    out["kyori"] = out["kyori"].astype(int)
    out["year"] = out["date"].dt.year
    out["cohort"] = out["year"] - out["barei"]
    out["field"] = out.groupby("race_id")["race_id"].transform("size")
    out["top3"] = out["jyuni"].between(1, 3).astype(float)
    out["fin_pct"] = np.where(out["jyuni"] >= 1, (out["jyuni"] - 1) / (out["field"] - 1).clip(lower=1), np.nan)
    out.loc[~(out["time_sec"] > 0), "time_sec"] = np.nan
    return out.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)[RUN_COLUMNS]


def load_runs(version_dir: Path, blood_dir: Path) -> pd.DataFrame:
    def read(p: Path) -> pd.DataFrame:
        return pd.read_csv(p, dtype=str, na_filter=False)
    return build_runs(read(version_dir / "races.csv"), read(version_dir / "race_horses.csv"),
                      read(blood_dir / "pedigrees.csv"))

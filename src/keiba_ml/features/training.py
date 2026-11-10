"""総合モデルの調教の特徴量（``t_``。2026-11-09）。JV-Data の坂路（HC）・ウッドチップ（WC）から作る。

**調教はレースの前に行われる情報だが、使うのは予想するレースの日より前（前日まで）の記録だけ**にする
（当日の朝の調教は前日の予想では分からない）。結果の列は読まない。

坂路とウッドは時計の水準がまったく違うので、**同じ日・同じトレセン・同じ種類（ウッドはコースも）の
調教の中での z 値**にそろえる（負ほど速い）。その日の馬場の重さもここで消える。

- 量: 直近 7 / 14 / 30 / 60 日の本数、14 日の坂路・ウッドの本数、最後の調教からの日数
- 追い切り: 直近 7 日（無ければ 14 日）で全体の時計がいちばん速い調教の z 値・ラスト1ハロン・加速・
  レースまでの日数・種類・時計そのもの
- 14 日の最速・平均、15〜120 日前の最速との差（本人の普段より動けているか）
- 調教師の普段の追い切りとの差（その日より前の管理馬の平均）
- レース内の順位・平均との差・3番手との差
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from keiba.config import TRAINING_DATA_DIR
from keiba_ml.config import ML_DATA_DIR

#: 同じ日・場所の調教がこれより少ないと z 値を作らない（平均と散らばりが当てにならない）。
MIN_DAY_GROUP = 8
#: 調教を探す最長の日数（本人の普段の時計に使う）。
LONG_LOOKBACK_DAYS = 120
#: 全体の時計（4ハロン）が欠損のときは 3ハロンの z 値を使う。
_SLOPE_COLUMNS = ["tresen_kubun", "chokyo_date", "ketto_num", "time_4f", "time_3f", "lap_2",
                  "lap_1", "data_kubun"]
_WOOD_COLUMNS = ["tresen_kubun", "chokyo_date", "ketto_num", "course", "time_4f", "time_3f",
                 "lap_2", "lap_1", "data_kubun"]
_RACE_RELATIVE = ("t_oi_z", "t_oi_zl1", "t_best14_z", "t_mean14_z", "t_oi_vs_self",
                  "t_oi_vs_trainer", "t_n_14", "t_oi_za", "t_best14_l1",
                  "t_slope_best14_l1", "t_wood_best14_l1", "t_slope_best14_t4",
                  "t_wood_best14_t4", "t_load14")


def _current_version(root: Path) -> str:
    return (root / "CURRENT").read_text(encoding="utf-8").strip()


def _day_z(values: pd.Series, keys: list[pd.Series]) -> pd.Series:
    grouped = values.groupby(keys, sort=False)
    n = grouped.transform("count")
    z = (values - grouped.transform("mean")) / grouped.transform("std").where(lambda s: s > 1e-6)
    return z.where(n >= MIN_DAY_GROUP).clip(-4.0, 4.0)


def normalize_workouts(slope: pd.DataFrame, wood: pd.DataFrame) -> pd.DataFrame:
    """2種類の調教を1つの表にして z 値を足す。列: ketto_num, day(datetime), kind(0坂路/1ウッド), ..."""
    frames = []
    for kind, raw in ((0, slope), (1, wood)):
        d = raw[raw["data_kubun"].astype(str) != "0"].copy()
        course = d["course"].astype(str) if kind else pd.Series("S", index=d.index)
        day = pd.to_datetime(d["chokyo_date"])
        tresen = d["tresen_kubun"].astype(str)
        keys = [tresen, course, day]
        out = pd.DataFrame({"ketto_num": d["ketto_num"].astype(str), "day": day,
                            "kind": np.int8(kind)})
        for col in ("time_4f", "time_3f", "lap_2", "lap_1"):
            out[col] = pd.to_numeric(d[col], errors="coerce").astype("float32")
        z4 = _day_z(out["time_4f"], keys)
        z3 = _day_z(out["time_3f"], keys)
        out["z"] = z4.fillna(z3).astype("float32")
        out["zl1"] = _day_z(out["lap_1"], keys).astype("float32")
        # 同じ世代どうしの z 値（2歳・3歳・4歳以上。若い馬は時計が出にくい）
        birth = pd.to_numeric(out["ketto_num"].str[:4], errors="coerce")
        age = (day.dt.year - birth).clip(lower=2, upper=4).astype("Int64").astype(str)
        za4 = _day_z(out["time_4f"], [*keys, age])
        out["za"] = za4.fillna(_day_z(out["time_3f"], [*keys, age])).astype("float32")
        out["accel"] = (out["lap_2"] - out["lap_1"]).astype("float32")
        frames.append(out.drop(columns=["lap_2"]))
    table = pd.concat(frames, ignore_index=True)
    return table.dropna(subset=["z"]).sort_values(["ketto_num", "day", "z"], kind="mergesort") \
        .reset_index(drop=True)


def load_workouts(version: str = "", *, data_dir: Path = TRAINING_DATA_DIR,
                  cache_dir: Path | None = None) -> pd.DataFrame:
    """調教の版を読む。作り直しは数分かかるので data/ml/cache に版ごとに残す。"""
    version = version or _current_version(data_dir)
    cache_dir = cache_dir or ML_DATA_DIR / "cache"
    cached = cache_dir / f"workouts_v2_{version}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    folder = data_dir / "versions" / version
    slope = pd.read_csv(folder / "slope.csv.gz", dtype=str, usecols=_SLOPE_COLUMNS)
    wood = pd.read_csv(folder / "wood.csv.gz", dtype=str, usecols=_WOOD_COLUMNS)
    table = normalize_workouts(slope, wood)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cached, index=False)
    return table


def _epoch_days(values) -> np.ndarray:
    return pd.to_datetime(values).to_numpy().astype("datetime64[D]").astype(np.int64)


def _expand(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各行の [lo, hi) を縦に展開する。戻りは (行の番号, 調教の番号)。"""
    n = np.maximum(hi - lo, 0)
    rows = np.repeat(np.arange(len(lo)), n)
    starts = np.repeat(lo, n)
    offset = np.arange(n.sum()) - np.repeat(np.r_[0, np.cumsum(n)[:-1]], n)
    return rows, starts + offset


def _trainer_table(trainer: np.ndarray, day: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """調教師ごとの、その日より前の値の合計と件数（同じ日は入れない）を各行に付けた表。"""
    daily = pd.DataFrame({"trainer": trainer, "day": day, "v": values}).dropna(subset=["v"])
    daily = daily.groupby(["trainer", "day"], sort=True)["v"].agg(["sum", "count"]).reset_index()
    by = daily.groupby("trainer", sort=False)
    last = pd.DataFrame({"trainer": trainer, "day": day, "_i": np.arange(len(trainer))}) \
        .sort_values("day", kind="mergesort")
    # 値の無い日の行にも、それ以前の累積が入るように「その日以前の最後」を引く
    daily["s_all"], daily["c_all"] = by["sum"].cumsum(), by["count"].cumsum()
    merged = pd.merge_asof(last, daily[["trainer", "day", "s_all", "c_all"]].sort_values("day"),
                           on="day", by="trainer", allow_exact_matches=False).sort_values("_i")
    return merged.fillna({"s_all": 0.0, "c_all": 0.0})


def _trainer_prior_mean(trainer, day, values) -> np.ndarray:
    t = _trainer_table(trainer, day, values)
    return t["s_all"].to_numpy(dtype=float) / t["c_all"].where(t["c_all"] > 0).to_numpy(dtype=float)


def training_features(df: pd.DataFrame, workouts: pd.DataFrame) -> pd.DataFrame:
    """df（race_id, race_date, ketto_num, chokyosi_code）の各行の t_ 列。index は df と同じ。"""
    horses = pd.Index(workouts["ketto_num"].unique())
    w_code = horses.get_indexer(workouts["ketto_num"])
    w_day = _epoch_days(workouts["day"])
    q_code = horses.get_indexer(df["ketto_num"].astype(str))
    q_day = _epoch_days(df["race_date"])
    base = np.int64(1) << 20
    keys = w_code.astype(np.int64) * base + w_day
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    known = q_code >= 0
    q_key = q_code.astype(np.int64) * base + q_day
    # hi は「レース当日の 0 件目」= 前日までの調教だけ
    hi = np.where(known, np.searchsorted(keys, q_key, side="left"), 0)
    lo = np.where(known, np.searchsorted(keys, q_key - LONG_LOOKBACK_DAYS, side="left"), 0)
    rows, picks = _expand(lo, hi)
    idx = order[picks]
    ago = q_day[rows] - w_day[idx]
    e = pd.DataFrame({
        "row": rows, "ago": ago.astype(np.int32),
        "kind": workouts["kind"].to_numpy()[idx],
        "z": workouts["z"].to_numpy()[idx], "zl1": workouts["zl1"].to_numpy()[idx],
        "accel": workouts["accel"].to_numpy()[idx],
        "t4": workouts["time_4f"].to_numpy()[idx], "t3": workouts["time_3f"].to_numpy()[idx],
        "l1": workouts["lap_1"].to_numpy()[idx], "za": workouts["za"].to_numpy()[idx],
    })
    out = pd.DataFrame(index=np.arange(len(df)))
    for days in (7, 14, 30, 60):
        out[f"t_n_{days}"] = e[e["ago"] <= days].groupby("row").size()
    recent = e[e["ago"] <= 14]
    for kind, name in ((0, "slope"), (1, "wood")):
        out[f"t_n_14_{name}"] = recent[recent["kind"] == kind].groupby("row").size()
    count_cols = [c for c in out.columns if c.startswith("t_n_")]
    # 調教の記録が1本でもある馬だけ 0 を入れる（馬が記録に無い = 欠損のまま）
    filled = out[count_cols].fillna(0.0).to_numpy(dtype=float)
    filled[~known] = np.nan
    out[count_cols] = filled
    out["t_days_last"] = e.groupby("row")["ago"].min()
    grouped14 = recent.groupby("row")
    out["t_best14_z"] = grouped14["z"].min()
    out["t_mean14_z"] = grouped14["z"].mean()
    out["t_best14_zl1"] = grouped14["zl1"].min()
    out["t_best14_l1"] = grouped14["l1"].min()
    # 追い切り = 直近 7 日の最速。無ければ 14 日の最速
    ranked = recent.assign(_near=(recent["ago"] > 7).astype(np.int8)) \
        .sort_values(["row", "_near", "z"], kind="mergesort").drop_duplicates("row") \
        .set_index("row")
    out["t_oi_z"] = ranked["z"]
    out["t_oi_zl1"] = ranked["zl1"]
    out["t_oi_accel"] = ranked["accel"]
    out["t_oi_days"] = ranked["ago"]
    out["t_oi_kind"] = ranked["kind"]
    out["t_oi_t4_slope"] = ranked["t4"].where(ranked["kind"] == 0)
    out["t_oi_t4_wood"] = ranked["t4"].where(ranked["kind"] == 1)
    out["t_oi_l1"] = ranked["l1"]
    out["t_oi_za"] = ranked["za"]
    out["t_best14_za"] = grouped14["za"].min()
    out["t_load14"] = (-recent["z"]).clip(lower=0).groupby(recent["row"]).sum()
    last = e.sort_values(["row", "ago"], kind="mergesort").drop_duplicates("row").set_index("row")
    out["t_last_z"] = last["z"]
    out["t_last_l1"] = last["l1"]
    for kind, name in ((0, "slope"), (1, "wood")):
        grouped = recent[recent["kind"] == kind].groupby("row")
        out[f"t_{name}_best14_z"] = grouped["z"].min()
        out[f"t_{name}_best14_l1"] = grouped["l1"].min()
        out[f"t_{name}_best14_t4"] = grouped["t4"].min()
        out[f"t_{name}_best14_t3"] = grouped["t3"].min()
    out["t_best_prior_z"] = e[e["ago"] > 14].groupby("row")["z"].min()
    out["t_oi_vs_self"] = out["t_oi_z"] - out["t_best_prior_z"]
    out = out.set_axis(df.index).astype(np.float32)

    # 調教師の普段との差（その日より前に出走した管理馬の平均。同じ日の馬は入れない）
    trainer = df["chokyosi_code"].astype(str).to_numpy()
    values = out["t_oi_z"].to_numpy(dtype=float)
    out["t_oi_vs_trainer"] = (values - _trainer_prior_mean(trainer, q_day, values)).astype(np.float32)

    # 前走のときの調教との差（同じ馬の1つ前の出走。df の中の日付順）
    order_df = pd.DataFrame({"k": df["ketto_num"].astype(str).to_numpy(), "d": q_day,
                             "i": np.arange(len(df))}).sort_values(["k", "d"], kind="mergesort")
    for col, new in (("t_oi_z", "t_oi_z_vs_prev"), ("t_n_14", "t_n_14_vs_prev"),
                     ("t_best14_l1", "t_best14_l1_vs_prev")):
        values = out[col].to_numpy(dtype=float)[order_df["i"].to_numpy()]
        prev = pd.Series(values).groupby(order_df["k"].to_numpy(), sort=False).shift(1).to_numpy()
        delta = np.empty(len(df), dtype=np.float32)
        delta[order_df["i"].to_numpy()] = values - prev
        out[new] = delta

    race = df["race_id"]
    for col in _RACE_RELATIVE:
        values = out[col].astype(float)
        grouped = values.groupby(race)
        ascending = col not in ("t_n_14", "t_load14")     # 時計は小さいほど速い。量は多いほど上
        rank = grouped.rank(ascending=ascending, method="first")
        third = values.where(rank == 3).groupby(race).transform("max")
        out[f"{col}_rpct"] = grouped.rank(ascending=ascending, pct=True)
        out[f"{col}_rdiff"] = values - grouped.transform("mean")
        out[f"{col}_gap3"] = values - third
    return out.astype(np.float32)

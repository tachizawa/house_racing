"""as-of 集計: 各行について「キーが同じで、レース日が**厳密に前**の行」だけを集計する。

血統モデルの ``features.pedigree._asof_sums`` を、合計以外（最大・最小・直前の値）と
任意の列に広げたもの。手順は同じ:

1. 履歴に入れる行（``include``）を (keys, 日付) で集計し、keys 内で日付順に累積する
   （= その日までの値。その日を含む）。
2. 各行に、keys が同じで日付が**厳密に前**の最新の累積を merge_asof で付ける
   （allow_exact_matches=False）。同じ日のレースの結果は入らない（前日予想の前提）。

keys のいずれかが欠損（MISSING / 空 / NaN）の行と、前の履歴が無い行は、合計なら 0、
それ以外（最大・最小・直前）なら NaN を返す。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from keiba_ml.features.base import MISSING

#: 集計の種類 → (同じ日の行のまとめ方, 日付をまたぐ累積のしかた)
_HOW = {
    "sum": ("sum", "cumsum"),
    "max": ("max", "cummax"),
    "min": ("min", "cummin"),
    "last": ("last", "ffill"),
}


def valid_keys(df: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    valid = np.ones(len(df), dtype=bool)
    for key in keys:
        valid &= ((df[key] != "") & (df[key] != MISSING) & df[key].notna()).to_numpy()
    return valid


def asof(df: pd.DataFrame, keys: Sequence[str], columns: Mapping[str, tuple[str, str]], *,
         include: pd.Series | np.ndarray) -> pd.DataFrame:
    """columns = {出力名: (df の列名, "sum" | "max" | "min" | "last")}。

    include は履歴に入れる行（結果のある行など）。集計される側の値の NaN は、合計では
    0 として、それ以外では無視して扱う（"last" は NaN でない最後の値）。
    """
    keys = list(keys)
    names = list(columns)
    # 履歴が空でも誤りは誤りとして返す（黙って NaN / 0 を返さない）。
    bad = sorted({how for _col, how in columns.values() if how not in _HOW})
    if bad:
        raise ValueError(f"集計の種類が違います: {bad}")
    sums = [n for n in names if columns[n][1] == "sum"]
    result = pd.DataFrame(np.nan, index=df.index, columns=names)
    result[sums] = 0.0
    valid = valid_keys(df, keys)
    hist = df[valid & np.asarray(include, dtype=bool)]
    if hist.empty:
        return result
    source = hist[keys + ["race_date"]].copy()
    for name in names:
        col, _how = columns[name]
        source[name] = pd.to_numeric(hist[col], errors="coerce").astype(float)
    agg = (source.groupby(keys + ["race_date"], sort=False)
           .agg(**{n: (n, _HOW[columns[n][1]][0]) for n in names})
           .reset_index()
           .sort_values(keys + ["race_date"], kind="mergesort"))
    agg[sums] = agg[sums].fillna(0.0)
    grouped = agg.groupby(keys, sort=False)
    for name in names:
        cumulative = _HOW[columns[name][1]][1]
        agg[name] = getattr(grouped[name], cumulative)()
        if cumulative in ("cummax", "cummin"):
            # 累積の最大・最小は NaN の日を NaN のまま返す。その日の行に前の日までの値を残す。
            agg[name] = agg.groupby(keys, sort=False)[name].ffill()
    agg = agg.sort_values("race_date", kind="mergesort")

    left = df.loc[valid, keys + ["race_date"]].copy()
    # pandas のバージョンで groupby 後のキーの dtype（object / string）が変わる。
    # merge_asof は by の dtype の一致を要求するので両側をそろえる。
    for key in keys:
        left[key] = left[key].astype(str)
        agg[key] = agg[key].astype(str)
    left["_row"] = left.index
    left = left.sort_values("race_date", kind="mergesort")
    merged = pd.merge_asof(left, agg, on="race_date", by=keys,
                           allow_exact_matches=False, direction="backward")
    merged = merged.set_index("_row")
    result.loc[merged.index, names] = merged[names].to_numpy()
    result[sums] = result[sums].fillna(0.0)
    return result

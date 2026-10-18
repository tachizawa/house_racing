"""早熟・晩成の集計（開発方針「特に調査する内容: 早熟型・晩成型の傾向、年齢による成長特性」）。

祖先（既定は父）ごとに、産駒の年齢別 A/E（3着以内数 / 偶然の期待3着以内数）を出す。
これは**記述統計**で、学習の入力ではない（学習の入力は features.pedigree の as-of 版）。
集計範囲は呼び出し側が渡した期間だけ。テスト期間を含めて集計しないこと。

「早熟型・晩成型」の定義は未決定（開発方針 §9）なので、ラベルは付けずに
``maturity = log(A/E 4歳以上 / A/E 2歳)`` という連続量と、各年齢の標本数を並べる。
定義を決めるときの材料にする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_ml.features.pedigree import expected_top3

AGES: tuple[int, ...] = (2, 3, 4, 5, 6)


def age_curve(df: pd.DataFrame, *, path: str = "S", min_starts: int = 200,
              shrinkage: float = 10.0, names: dict[str, str] | None = None) -> pd.DataFrame:
    res = df[df["has_result"] & (df[f"ped_{path}"] != "") & df["age"].notna()].copy()
    res["e3"] = expected_top3(res["field_size"])
    res["age_g"] = res["age"].astype(int).clip(upper=AGES[-1])
    total = res.groupby(f"ped_{path}").agg(starts=("e3", "size"), horses=("ketto_num", "nunique"),
                                            top3=("target_top3", "sum"), e3=("e3", "sum"))
    total = total[total["starts"] >= min_starts]
    by_age = res.groupby([f"ped_{path}", "age_g"]).agg(
        n=("e3", "size"), k=("target_top3", "sum"), e=("e3", "sum")).reset_index()
    by_age["ae"] = (by_age["k"] + shrinkage) / (by_age["e"] + shrinkage)
    ae = by_age.pivot(index=f"ped_{path}", columns="age_g", values="ae")
    n = by_age.pivot(index=f"ped_{path}", columns="age_g", values="n")
    out = total.copy()
    out["ae_all"] = (out["top3"] + shrinkage) / (out["e3"] + shrinkage)
    for age in AGES:
        label = f"{age}p" if age == AGES[-1] else str(age)
        out[f"ae_age{label}"] = ae.get(age)
        out[f"n_age{label}"] = n.get(age).fillna(0).astype(int) if age in n else 0
    older = res[res["age_g"] >= 4].groupby(f"ped_{path}").agg(k=("target_top3", "sum"),
                                                             e=("e3", "sum"))
    older_ae = (older["k"] + shrinkage) / (older["e"] + shrinkage)
    out["ae_age4plus"] = older_ae
    out["maturity"] = np.log(out["ae_age4plus"] / out["ae_age2"])
    out.index.name = "hansyoku_num"
    out = out.reset_index()
    # names が空でも列は作る（growth_report.py が "name" 列を選ぶ）。
    out.insert(1, "name", out["hansyoku_num"].map(names or {}).fillna(""))
    return out.sort_values("maturity", ascending=False).reset_index(drop=True)

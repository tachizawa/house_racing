"""学習時点の情報整合性チェック（Phase 1）と、学習直前の未来情報チェック。

各検査は ``{"name", "status": ok|warn|fail, "detail": {...}, "message"}`` を返す。
fail は「このまま学習すると結果が信用できない」、warn は「人が見て判断する」。
build_dataset.py は fail があっても保存はする（原因を調べるのに表が要るため）が、
学習スクリプトは既定で fail のあるデータセットを拒否する。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from keiba_ml.config import FORBIDDEN_FEATURE_NAMES, FORBIDDEN_FEATURE_PREFIXES, MLConfig


def _result(name: str, status: str, message: str, **detail) -> dict:
    return {"name": name, "status": status, "message": message, "detail": detail}


def check_unique_runner(df: pd.DataFrame) -> dict:
    dup = int(df.duplicated(subset=["race_id", "umaban"]).sum())
    dup_horse = int(df.duplicated(subset=["race_id", "ketto_num"]).sum())
    status = "ok" if dup == 0 and dup_horse == 0 else "fail"
    return _result("unique_runner", status,
                   "同じレース・同じ馬番（または同じ馬）の行が重複していないか",
                   duplicated_umaban=dup, duplicated_horse=dup_horse)


def check_targets(df: pd.DataFrame) -> dict:
    """結果のあるレースで、3着以内が3頭前後になっているか。

    同着で4頭以上になるのはありうる。0〜2頭は結果の取り込み漏れを疑う。
    """
    res = df[df["has_result"]]
    if res.empty:
        return _result("targets", "warn", "結果のある行がありません。")
    per_race = res.groupby("race_id").agg(top3=("target_top3", "sum"),
                                          wins=("target_win", "sum"),
                                          n=("umaban", "size"))
    short = per_race[(per_race["top3"] < 3) & (per_race["n"] >= 3)]
    no_winner = per_race[per_race["wins"] < 1]
    rate = len(short) / len(per_race)
    status = "ok" if rate < 0.01 else "warn" if rate < 0.05 else "fail"
    return _result("targets", status,
                   "3着以内が3頭未満のレース（出走3頭以上）の割合",
                   races=int(len(per_race)), short_races=int(len(short)),
                   short_rate=rate, races_without_winner=int(len(no_winner)),
                   examples=list(short.index[:5]))


def check_age(df: pd.DataFrame) -> dict:
    age = df["age"]
    missing = int(age.isna().sum())
    out = int(((age < 2) | (age > 15)).fillna(False).sum())
    status = "ok" if out == 0 else "warn"
    if len(df) and missing / len(df) > 0.05:
        status = "warn"
    return _result("age", status, "年齢が 2〜15 歳の範囲にあり、欠損が少ないか",
                   missing=missing, out_of_range=out,
                   source_counts={k: int(v) for k, v in df["age_source"].value_counts().items()})


def check_age_convention(df: pd.DataFrame, kazoe_until_year: int) -> dict:
    """表記の馬齢を満年齢に直した値と「レースの年 - 生年」の一致率を年代別に出す。

    数え年の時代の境目（CodeConfig.kazoe_age_until_year）が合っていれば、
    両方の年代で一致率がほぼ 100% になる。片方だけ崩れていれば境目が違う。
    """
    both = df[df["birth_year"].notna() & df["age_from_barei"].notna()]
    if both.empty:
        return _result("age_convention", "warn",
                       "生年と表記の馬齢の両方がある行が無く、検証できません。")
    by_birth = both["year"] - both["birth_year"].astype(int)
    agree = (by_birth == both["age_from_barei"].astype(int))
    era = np.where(both["year"] <= kazoe_until_year, "kazoe", "man")
    rates = {}
    for name in ("kazoe", "man"):
        mask = era == name
        if mask.any():
            rates[name] = {"rows": int(mask.sum()), "agree_rate": float(agree[mask].mean())}
    yearly = (pd.DataFrame({"year": both["year"], "agree": agree})
              .groupby("year")["agree"].mean())
    bad_years = [int(y) for y, r in yearly.items() if r < 0.95]
    status = "ok" if not bad_years else "fail"
    return _result("age_convention", status,
                   "数え年/満年齢の変換が年代ごとに正しいか（生年との一致率）",
                   kazoe_until_year=kazoe_until_year, by_era=rates,
                   years_below_95pct=bad_years)


def check_pedigree_coverage(df: pd.DataFrame) -> dict:
    has = df["ped_S"] != ""
    yearly = has.groupby(df["year"]).mean()
    overall = float(has.mean()) if len(df) else 0.0
    low = {int(y): float(r) for y, r in yearly.items() if r < 0.9}
    status = "ok" if overall >= 0.95 and not low else "warn" if overall >= 0.5 else "fail"
    return _result("pedigree_coverage", status,
                   "父が分かる出走の割合（年別に 90% 未満の年を列挙）",
                   overall=overall, years_below_90pct=low)


def check_parent_link(ped_stats: dict) -> dict:
    link = ped_stats.get("parent_link", {})
    checked, rate = link.get("checked", 0), link.get("rate")
    if not checked:
        return _result("pedigree_parent_link", "warn",
                       "SK/UM の祖先と HN の父母リンクを突き合わせられる組がありません"
                       "（HN が未取得なら4世代目以降は埋まりません）。", **link)
    status = "ok" if rate >= 0.98 else "warn" if rate >= 0.9 else "fail"
    return _result("pedigree_parent_link", status,
                   "SK/UM の『父父』等と、HN から辿った『父の父』等の一致率。"
                   "低いならレコードの位置（オフセット）がずれている", **link)


def check_birth_before_race(df: pd.DataFrame) -> dict:
    bad = df["birth_year"].notna() & (df["birth_year"].astype("Float64") >= df["year"])
    count = int(bad.fillna(False).sum())
    return _result("birth_before_race", "ok" if count == 0 else "fail",
                   "生年がレースの年より前か", violations=count)


def check_dataset(df: pd.DataFrame, ped_stats: dict, config: MLConfig) -> list[dict]:
    return [
        check_unique_runner(df),
        check_targets(df),
        check_age(df),
        check_age_convention(df, config.codes.kazoe_age_until_year),
        check_pedigree_coverage(df),
        check_parent_link(ped_stats),
        check_birth_before_race(df),
    ]


# --- 学習直前の検査 ---------------------------------------------------------

def check_feature_columns(columns: Iterable[str]) -> dict:
    """特徴量の列名に結果・オッズ・目的変数が混ざっていないか。"""
    bad = [c for c in columns
           if c.startswith(FORBIDDEN_FEATURE_PREFIXES) or c in FORBIDDEN_FEATURE_NAMES]
    return _result("feature_columns", "ok" if not bad else "fail",
                   "特徴量に結果・オッズ・目的変数の列が入っていないか", forbidden=bad)


def check_split(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    """学習 < 検証 < テスト が日付で重ならず、同じレースが2か所に無いか。"""
    problems = []
    parts = [("train", train), ("valid", valid), ("test", test)]
    for (a_name, a), (b_name, b) in zip(parts, parts[1:]):
        if len(a) and len(b) and a["race_date"].max() >= b["race_date"].min():
            problems.append(f"{a_name} の最終日 >= {b_name} の初日")
    ids = [set(p["race_id"]) for _, p in parts]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        problems.append("同じ race_id が複数の区間にある")
    return _result("split", "ok" if not problems else "fail",
                   "時系列分割が重なっていないか", problems=problems,
                   ranges={name: [str(p["race_date"].min().date()) if len(p) else None,
                                  str(p["race_date"].max().date()) if len(p) else None]
                           for name, p in parts})


def check_asof_sample(history: pd.DataFrame, feature_frame: pd.DataFrame, *, path: str,
                      shrinkage: float, exclude_self: bool,
                      sample: int = 200, seed: int = 0, only: pd.Index | None = None) -> dict:
    """as-of 産駒成績を、無作為に選んだ行で総当たりに計算し直して一致を確かめる。

    features.pedigree の累積計算（groupby + cumsum + merge_asof）が「その日より前」
    だけを使っていることを、遅いが自明に正しい方法で検算する。history は特徴量を
    作ったのと同じ表（同じ index）を渡す。only を渡すと、検算する行をその index の中から
    選ぶ（評価する期間の行だけを確かめたいとき。履歴は history 全体を使う）。
    """
    from keiba_ml.features.pedigree import expected_top3

    col = f"st_{path}_ae_top3"
    if col not in feature_frame.columns:
        return _result("asof_sample", "warn", f"{col} がありません。")
    hist = history[history["has_result"]]
    candidates = history.index[history[f"ped_{path}"] != ""]
    if only is not None:
        candidates = candidates.intersection(only)
    if len(candidates) == 0:
        return _result("asof_sample", "warn", "検算できる行がありません。")
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
    mismatches = []
    for idx in picked:
        row = history.loc[idx]
        prior = hist[(hist[f"ped_{path}"] == row[f"ped_{path}"])
                     & (hist["race_date"] < row["race_date"])]
        if exclude_self:
            prior = prior[prior["ketto_num"] != row["ketto_num"]]
        actual = prior["target_top3"].sum()
        expected = expected_top3(prior["field_size"]).sum()
        value = (actual + shrinkage) / (expected + shrinkage)
        feature = feature_frame.at[idx, col]
        if not np.isclose(value, feature, rtol=1e-5, atol=1e-6):   # 特徴量は float32
            mismatches.append({"index": int(idx), "recomputed": float(value),
                               "feature": float(feature)})
    return _result("asof_sample", "ok" if not mismatches else "fail",
                   f"{col} を総当たりで検算して一致するか",
                   checked=int(len(picked)), mismatches=mismatches[:10])


def check_asof_samples(history: pd.DataFrame, feature_frame: pd.DataFrame, *,
                       paths: Sequence[str], shrinkage: float, exclude_self: bool,
                       sample: int = 200, seed: int = 0, only: pd.Index | None = None) -> dict:
    """check_asof_sample を産駒成績を出す全ての位置で行い、1つの結果にまとめる。

    先頭の位置だけでは、他の位置（母・母父 …）の未来情報の混入を見逃す。
    """
    by_path = {path: check_asof_sample(history, feature_frame, path=path, shrinkage=shrinkage,
                                       exclude_self=exclude_self, sample=sample, seed=seed,
                                       only=only)
               for path in paths}
    order = {"ok": 0, "warn": 1, "fail": 2}
    status = max((r["status"] for r in by_path.values()), key=order.__getitem__, default="warn")
    return _result("asof_sample", status,
                   f"産駒成績の A/E（{', '.join(paths)}）を総当たりで検算して一致するか",
                   by_path={p: {"status": r["status"], "message": r["message"], **r["detail"]}
                            for p, r in by_path.items()})


def summarize(checks: Sequence[dict]) -> str:
    lines = []
    for check in checks:
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[check["status"]]
        lines.append(f"[{mark}] {check['name']}: {check['message']}")
        if check["status"] != "ok":
            for key, value in check["detail"].items():
                lines.append(f"         {key}: {value}")
    return "\n".join(lines)

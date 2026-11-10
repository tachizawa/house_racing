"""学習時点の情報整合性チェック（Phase 1）と、学習直前の未来情報チェック。

各検査は ``{"name", "status": ok|warn|fail, "detail": {...}, "message"}`` を返す。
fail は「このまま学習すると結果が信用できない」、warn は「人が見て判断する」。
build_dataset.py は fail があっても保存はする（原因を調べるのに表が要るため）が、
学習スクリプトは既定で fail のあるデータセットを拒否する。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

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


def check_breeder_code(ped_stats: dict) -> dict:
    """生産者コードが埋まっているか。

    SK に生産者コードが無いと build_pedigree は全頭 "" にする。列はあるので breeder_features は
    止まらず、brd_* が全頭 1.0 の定数になって黙って効かなくなる。warn で知らせる（fail にしないのは、
    生産者を使わない設定でも同じデータセットを使うため）。
    """
    rate = ped_stats.get("breeder_code_fill_rate")
    if rate is None:
        return _result("breeder_code", "warn", "生産者コードの埋まり具合が統計にありません。")
    return _result("breeder_code", "ok" if rate > 0 else "warn",
                   "生産者コード（breeder_code）が埋まっている馬の割合。0 なら pedigree.breeder_stats の"
                   " brd_* が全頭 1.0 の定数になる", rate=rate)


def check_dataset(df: pd.DataFrame, ped_stats: dict, config: MLConfig) -> list[dict]:
    return [
        check_unique_runner(df),
        check_targets(df),
        check_age(df),
        check_age_convention(df, config.codes.kazoe_age_until_year),
        check_pedigree_coverage(df),
        check_parent_link(ped_stats),
        check_birth_before_race(df),
        check_breeder_code(ped_stats),
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


def check_history_start(history: pd.DataFrame, train_start: date, stats_history: str) -> dict:
    """履歴が設定どおりの日（"lead:N" なら学習開始日の N 年前）から始まっているか。

    総合モデルのタイム指数は履歴の最初の speed_standard_years 年の1着から基準タイムを作る。
    データが学習開始日の N 年前まで無いと（40年の窓など）、基準タイムが学習期間の結果から作られ、
    学習の行自身の結果が特徴量に入る。黙って短い履歴で続けず fail にする。"window" / "all" は
    見ない（学習期間の切り詰めは split.Window.truncated が持つ）。
    """
    from keiba_ml.dataset.split import TRUNCATION_TOLERANCE_DAYS, history_start

    start = history_start(train_start, stats_history)
    if start is None or not stats_history.startswith("lead:"):
        return _result("history_start", "ok", "履歴の開始日の確認は lead:N のときだけ",
                       stats_history=stats_history)
    first = history["race_date"].min().date() if len(history) else None
    late = first is None or (first - start).days > TRUNCATION_TOLERANCE_DAYS
    return _result("history_start", "fail" if late else "ok",
                   "履歴が学習開始日の lead 年前から始まっているか"
                   "（タイム指数の基準タイムを学習期間の結果で作らないため）",
                   expected_start=str(start), actual_start=str(first), stats_history=stats_history)


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


def check_extended_asof_sample(history: pd.DataFrame, feature_frame: pd.DataFrame, *,
                               paths: Sequence[str], shrinkage: float, exclude_self: bool,
                               sample: int = 100, seed: int = 0,
                               only: pd.Index | None = None) -> dict:
    """拡張の産駒成績（features.pedigree.extended_stat_features）を総当たりで検算する。

    無作為の行で、祖先ごとの産駒の走りの質の平均（``stx_<位置>_pf``）と出走した格の平均
    （``stx_<位置>_tier_mean``）を「その日より前の行だけ」から計算し直して比べる。
    縮小の先の全体の平均も、その日より前の行だけから出す（exclude_self なら本人の過去走を除く）。
    """
    from keiba_ml.features.pedigree import beat_fraction

    names = [f"stx_{p}_{k}" for p in paths for k in ("pf", "tier_mean")]
    absent = [c for c in names if c not in feature_frame.columns]
    if absent:
        return _result("extended_asof_sample", "warn", f"{absent} がありません。")
    hist = history[history["has_result"]].copy()
    hist["_pf"] = beat_fraction(hist)
    hist["_tier"] = pd.to_numeric(hist["grade_tier"], errors="coerce").astype(float)
    candidates = history.index if only is None else history.index.intersection(only)
    if len(candidates) == 0:
        return _result("extended_asof_sample", "warn", "検算できる行がありません。")
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
    mismatches = []
    kinds = (("pf", "_pf"), ("tier_mean", "_tier"))
    for idx in picked:
        row = history.loc[idx]
        before = hist[hist["race_date"] < row["race_date"]]
        # 縮小の先の全体の平均は祖先の位置によらないので、行ごとに1回だけ出す
        pool = before[before["ketto_num"] != row["ketto_num"]] if exclude_self else before
        base = {}
        for kind, col in kinds:
            overall = pool[col].dropna()
            base[kind] = float(overall.mean()) if len(overall) else 0.0
        for path in paths:
            key = row[f"ped_{path}"]
            prior = before[before[f"ped_{path}"] == key] if key not in ("", "NA") \
                else before.iloc[0:0]
            if exclude_self:
                prior = prior[prior["ketto_num"] != row["ketto_num"]]
            for kind, col in kinds:
                values = prior[col].dropna()
                value = (values.sum() + shrinkage * base[kind]) / (len(values) + shrinkage)
                feature = feature_frame.at[idx, f"stx_{path}_{kind}"]
                if not np.isclose(value, feature, rtol=1e-4, atol=1e-5):   # 特徴量は float32
                    mismatches.append({"index": int(idx), "column": f"stx_{path}_{kind}",
                                       "recomputed": float(value), "feature": float(feature)})
    return _result("extended_asof_sample", "ok" if not mismatches else "fail",
                   f"拡張の産駒成績（{', '.join(names)}）を総当たりで検算して一致するか",
                   checked=int(len(picked)), mismatches=mismatches[:10])


def check_form_asof_sample(history: pd.DataFrame, feature_frame: pd.DataFrame, *,
                           people_shrinkage: float, sample: int = 100, seed: int = 0,
                           only: pd.Index | None = None) -> dict:
    """総合モデルの本人の過去走・騎手の特徴量（features.form）を総当たりで検算する。

    無作為の行で、前走の走りの質（``f_pf_last1``）・本人の3着以内 A/E（``f_ae_own``）・
    騎手の A/E（``h_jockey_ae``）を「その日より前の行だけ」から計算し直して比べる。
    """
    from keiba_ml.features.base import MISSING
    from keiba_ml.features.form import OWN_SHRINKAGE
    from keiba_ml.features.pedigree import beat_fraction, expected_top3

    names = ["f_pf_last1", "f_ae_own", "h_jockey_ae"]
    absent = [c for c in names if c not in feature_frame.columns]
    if absent:
        return _result("form_asof_sample", "warn", f"{absent} がありません。")
    hist = history[history["has_result"]].copy()
    hist["_pf"] = beat_fraction(hist)
    hist["_e3"] = expected_top3(hist["field_size"])
    jockey = hist["kisyu_code"].fillna("").astype(str).str.strip()
    # 前走の値は own_history_features と同じく「結果の有無を問わず直前の行」から取る（結果の無い行
    # が間に挟まれば欠損になる）。結果のある行だけから取ると、その場合に検算が食い違う
    every = history.assign(_pf=beat_fraction(history))
    candidates = history.index if only is None else history.index.intersection(only)
    if len(candidates) == 0:
        return _result("form_asof_sample", "warn", "検算できる行がありません。")
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
    mismatches = []
    for idx in picked:
        row = history.loc[idx]
        before = hist["race_date"] < row["race_date"]
        own = hist[before & (hist["ketto_num"] == row["ketto_num"])]
        prev = every[(every["race_date"] < row["race_date"])
                     & (every["ketto_num"] == row["ketto_num"])].sort_values(
            "race_date", kind="mergesort")
        last = prev["_pf"].iloc[-1] if len(prev) else np.nan
        ae = (own["target_top3"].sum() + OWN_SHRINKAGE) / (own["_e3"].sum() + OWN_SHRINKAGE)
        # 欠損（pd.NA）と MISSING の騎手コードは people_features と同じく「騎手なし」として扱う
        code = "" if pd.isna(row["kisyu_code"]) else str(row["kisyu_code"]).strip()
        jk = hist[before & (jockey == code)] if code not in ("", MISSING) else hist.iloc[0:0]
        jae = (jk["target_top3"].sum() + people_shrinkage) / (jk["_e3"].sum() + people_shrinkage)
        for name, value in (("f_pf_last1", last), ("f_ae_own", ae), ("h_jockey_ae", jae)):
            got = float(feature_frame.at[idx, name])
            same = (np.isnan(value) and np.isnan(got)) or \
                np.isclose(value, got, rtol=1e-4, atol=1e-5)   # 特徴量は float32
            if not same:
                mismatches.append({"index": int(idx), "column": name,
                                   "recomputed": float(value), "feature": got})
    return _result("form_asof_sample", "ok" if not mismatches else "fail",
                   f"本人の過去走・騎手の特徴量（{', '.join(names)}）を総当たりで検算して一致するか",
                   checked=int(len(picked)), mismatches=mismatches[:10])


def _distance_m(values) -> pd.Series:
    """距離（m）。features.distance.prepare と同じく 0 以下は欠損とみなす。"""
    m = pd.to_numeric(values, errors="coerce").astype(float)
    return m.where(m > 0)


def _beat_fraction(rows: pd.DataFrame) -> pd.Series:
    """走りの質（features.distance の pf）を素朴に計算する。走りの質に入らない行は落とす。"""
    km = _distance_m(rows["kyori_m"])
    rank = pd.to_numeric(rows["result_rank"], errors="coerce").astype(float)
    n = rows["field_size"].astype(float)
    ok = rows["result_margin_sec"].notna() & rank.notna() & (n >= 2) & km.notna()
    return ((n - rank) / (n - 1)).clip(0.0, 1.0)[ok]


def check_distance_asof_sample(history: pd.DataFrame, feature_frame: pd.DataFrame, *,
                               config, sample: int = 100, seed: int = 0,
                               only: pd.Index | None = None) -> dict:
    """距離適性の特徴量を、無作為に選んだ行で「その日より前の行だけ」から計算し直して比べる。

    features.distance の累積計算（groupby + 累積 + merge_asof）が同じ日・後の日の結果を
    使っていないことを、遅いが自明に正しい方法で検算する。config は DistanceConfig。
    確かめる列: 本人の今回の距離帯での出走数・平均出走距離との差・前走距離との差・
    芝ダ×距離帯の相対的な走り、祖先ごとの産駒の平均距離・3着以内の平均距離との差。
    """
    hist = history[history["has_result"]]
    candidates = history.index if only is None else history.index.intersection(only)
    if len(candidates) == 0:
        return _result("distance_asof_sample", "warn", "検算できる行がありません。")
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
    m = config.own_shrinkage
    mismatches = []
    for idx in picked:
        row = history.loc[idx]
        km = float(_distance_m(pd.Series([row["kyori_m"]], dtype=object)).iloc[0])
        prior = hist[(hist["ketto_num"] == row["ketto_num"])
                     & (hist["race_date"] < row["race_date"])]
        # 距離の無い走りは数えない（features.distance の "last" も欠損でない最後の値）。
        dist = _distance_m(prior.sort_values("race_date", kind="mergesort")["kyori_m"]).dropna()
        same_band = _distance_m(prior.loc[prior["distance_band"] == row["distance_band"],
                                          "kyori_m"]).dropna()
        expected = {
            "own_log_runs_dband": np.log1p(len(same_band)),
            "dist_minus_mean": km - dist.mean() if len(dist) else np.nan,
            "dist_minus_last": km - float(dist.iloc[-1]) if len(dist) else np.nan,
        }
        surface = prior[prior["track_kind"] == row["track_kind"]]
        pf_surface = _beat_fraction(surface)
        pf_band = _beat_fraction(surface[surface["distance_band"] == row["distance_band"]])
        if len(pf_surface):
            base = pf_surface.mean()
            expected["own_rel_pf_surface_dband"] = (
                (pf_band.sum() + m * base) / (len(pf_band) + m) - base)
        else:
            expected["own_rel_pf_surface_dband"] = np.nan
        for path in config.pedigree_paths:
            col = f"ped_{path}"
            if row[col] in ("", None) or pd.isna(row[col]):
                continue
            kin = hist[(hist[col] == row[col]) & (hist["race_date"] < row["race_date"])]
            if config.exclude_self:
                kin = kin[kin["ketto_num"] != row["ketto_num"]]
            top3 = _distance_m(kin.loc[kin["target_top3"] == 1, "kyori_m"]).dropna()
            ran = _distance_m(kin.loc[_beat_fraction(kin).index, "kyori_m"])
            mean = ran.mean() if len(ran) else np.nan
            expected[f"ped_{path}_dist_minus_mean"] = km - mean
            expected[f"ped_{path}_dist_minus_top3_mean"] = km - (top3.mean() if len(top3) else mean)
        for name, value in expected.items():
            feature = float(feature_frame.at[idx, name])
            if not np.isclose(value, feature, rtol=1e-5, atol=1e-5, equal_nan=True):
                mismatches.append({"index": int(idx), "feature": name,
                                   "recomputed": float(value), "value": feature})
    return _result("distance_asof_sample", "ok" if not mismatches else "fail",
                   "距離適性の特徴量（本人の距離の経験・相対的な走り・産駒の距離の傾向）を"
                   "総当たりで検算して一致するか",
                   checked=int(len(picked)), mismatches=mismatches[:10])


def check_course_asof_sample(history: pd.DataFrame, feature_frame: pd.DataFrame, *,
                             config, sample: int = 100, seed: int = 0,
                             only: pd.Index | None = None) -> dict:
    """コース適性の特徴量を、無作為に選んだ行で「その日より前の行だけ」から計算し直して比べる。

    コースの区分のキー（競馬場 × 芝ダなど）と1走ごとの補助列は features.course.prepare_course で作る
    （区分の付け方は日付と関係ない）。集計は素朴に、その行より厳密に前の日の行だけで行う。config は CourseConfig。
    特徴量表の列のうち条件（context）と馬番の位置（umaban_frac）以外は**すべて**検算する。
    検算の仕方を知らない列が表にあれば fail（新しい列を足したら、ここにも検算を足すこと）。
    """
    from keiba_ml.features import course as cf

    candidates = history.index if only is None else history.index.intersection(only)
    if len(candidates) == 0:
        return _result("course_asof_sample", "warn", "検算できる行がありません。")
    prepared = cf.prepare_course(history, config)
    hist = prepared[prepared["has_result"]]
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
    m, mp, sb = config.own_shrinkage, config.pedigree_shrinkage, config.bias_shrinkage

    own_groups = [*cf._without_going(cf.OWN_GROUPS, config),
                  *((g, cf.EXTRA_GROUPS[g]) for g in config.extra_groups)]
    count_groups = [g for g in cf.COUNT_GROUPS if config.use_going or g != "going"]
    count_groups += list(config.extra_groups)
    ped_groups = [*cf._without_going(cf.PEDIGREE_GROUPS, config),
                  *((g, cf.PEDIGREE_EXTRA_GROUPS[g]) for g in config.pedigree_extra_groups)]

    def weights(rows: pd.DataFrame, day, decay: float) -> pd.Series:
        """古い走りほど軽くする重み exp(−経過日数 / decay)。decay が 0 なら全部 1。"""
        if decay <= 0:
            return pd.Series(1.0, index=rows.index)
        return np.exp(-(day - rows["race_date"]).dt.days.astype(float) / decay)

    def rel(prior: pd.DataFrame, key: str, base_key: str, row: pd.Series, quality: str,
            shrink: float, decay: float = 0.0) -> float:
        """区分 key の走りの平均を、相手の区分 base_key の平均に向けて縮めた値 − 相手の平均。"""
        if row[base_key] == "NA":
            return np.nan
        base = prior[(prior[base_key] == row[base_key]) & (prior["_perf"] > 0)]
        if base.empty:
            return np.nan
        w = weights(base, row["race_date"], decay)
        # 区分が分からない行は「その区分の走り 0 走」= 相手と同じ（0）。
        inside = (base[key] == row[key]) if row[key] != "NA" else pd.Series(False, index=base.index)
        value = w * base[f"_{quality}"]
        mean = value.sum() / w.sum()
        return (value[inside].sum() + shrink * mean) / (w[inside].sum() + shrink) - mean

    def same(rows: pd.DataFrame, key: str, row: pd.Series) -> pd.DataFrame:
        """区分 key がその行と同じ行。区分が分からなければ 0 行。"""
        return rows[rows[key] == row[key]] if row[key] != "NA" else rows.iloc[0:0]

    def ae(rows: pd.DataFrame, k: str, e: str, shrink: float) -> float:
        """3着以内の A/E を 1 に向けて縮めたもの。"""
        return (rows[k].sum() + shrink) / (rows[e].sum() + shrink)

    def own_mean(prior: pd.DataFrame, n: str, s: str, w: pd.Series) -> float:
        """本人の過去走の重み付き平均（位置取り・末脚）。0走なら NaN。"""
        count = (w * prior[n]).sum()
        return float((w * prior[s]).sum() / count) if count > 0 else np.nan

    mismatches = []
    checked_names: set[str] = set()
    for idx in picked:
        row = prepared.loc[idx]
        day = row["race_date"]
        earlier = hist[hist["race_date"] < day]
        prior = earlier[earlier["ketto_num"] == row["ketto_num"]]
        expected: dict[str, float] = {}
        for group, base_key in own_groups:
            for q in config.qualities:
                expected[f"own_rel_{q}_{group}"] = rel(prior, f"k_{group}", base_key, row, q, m,
                                                       config.decay_days)
        for group in count_groups:
            key = f"k_{group}"
            expected[f"own_log_runs_{group}"] = np.log1p(
                int(((prior[key] == row[key]) & (prior["_run"] > 0)).sum())) \
                if row[key] != "NA" else 0.0
        for path in config.pedigree_paths:
            col = f"ped_{path}"
            if row[col] in ("", None, "NA") or pd.isna(row[col]):
                kin = earlier.iloc[0:0]
            else:
                kin = earlier[earlier[col] == row[col]]
                if config.exclude_self:
                    kin = kin[kin["ketto_num"] != row["ketto_num"]]
            for group, base_key in ped_groups:
                expected[f"ped_{path}_rel_pf_{group}"] = rel(kin, f"k_{group}", base_key, row,
                                                             "pf", mp)
        if config.draw_bias:
            expected["bias_draw_ae"] = ae(same(earlier, "k_draw", row), "_k3", "_e3", sb)
            if config.bias_by_going:
                expected["bias_draw_going_ae"] = ae(same(earlier, "k_draw_going", row),
                                                    "_k3", "_e3", sb)
        own_style = np.nan
        if config.style:
            front = ae(same(earlier, "k_course_dband", row), "_fr_k3", "_fr_e3", sb)
            expected["bias_front_ae"] = front
            if config.bias_by_going:
                expected["bias_front_going_ae"] = ae(same(earlier, "k_course_dband_going", row),
                                                     "_fr_k3", "_fr_e3", sb)
            decay = config.decay_days if config.decay_style else 0.0
            own_style = own_mean(prior, "_c4n", "_c4", weights(prior, day, decay))
            expected["own_style"] = own_style
            expected["style_fit"] = (front - 1.0) * (0.5 - own_style)
            if config.front_relative:
                every = ae(same(earlier, "k_surface_dband", row), "_fr_k3", "_fr_e3", sb)
                expected["bias_front_rel"] = front / every
                expected["style_fit_rel"] = (front / every - 1.0) * (0.5 - own_style)
        if config.kick:
            here = ae(same(earlier, "k_course_dband", row), "_fk_k3", "_fk_e3", sb)
            every = ae(same(earlier, "k_surface_dband", row), "_fk_k3", "_fk_e3", sb)
            own_kick = own_mean(prior, "_kn", "_kick", pd.Series(1.0, index=prior.index))
            expected["bias_kick_rel"] = here / every
            expected["own_kick"] = own_kick
            expected["kick_fit"] = (here / every - 1.0) * (0.5 - own_kick)
        if config.meeting_days > 0:
            # 予想する日の meeting_days 日前から前日まで（同じ日は入らない）
            recent = earlier[earlier["race_date"] >= day - pd.Timedelta(days=config.meeting_days)]
            ms = config.meeting_shrinkage
            expected["meet_draw_ae"] = ae(same(recent, "k_venue_zone", row), "_k3", "_e3", ms)
            front_rel = ae(same(recent, "k_venue", row), "_fr_k3", "_fr_e3", ms) / ae(
                same(earlier, "k_venue", row), "_fr_k3", "_fr_e3", ms)
            expected["meet_front_rel"] = front_rel
            if config.style:
                expected["meet_style_fit"] = (front_rel - 1.0) * (0.5 - own_style)
        if config.race_relative:
            # レース内の差は、同じレースの全頭の元の列（上で検算する列）の平均との差
            members = prepared.index[prepared["race_id"] == row["race_id"]]
            for src in cf.RELATIVE_SOURCES:
                if src in feature_frame:
                    values = feature_frame.loc[members, src].astype(float)
                    expected[f"rr_{src}"] = float(feature_frame.at[idx, src]) - values.mean()
        for name, value in expected.items():
            if name not in feature_frame:
                mismatches.append({"index": int(idx), "feature": name,
                                   "recomputed": float(value), "value": None})
                continue
            checked_names.add(name)
            feature = float(feature_frame.at[idx, name])
            if not np.isclose(value, feature, rtol=1e-4, atol=1e-4, equal_nan=True):
                mismatches.append({"index": int(idx), "feature": name,
                                   "recomputed": float(value), "value": feature})
    exempt = set(config.context) | {"umaban_frac"}
    unchecked = sorted(c for c in feature_frame.columns if c not in exempt and c not in checked_names)
    status = "ok" if not mismatches and not unchecked else "fail"
    return _result("course_asof_sample", status,
                   "コース適性の特徴量（条件と馬番の位置を除く全列）を総当たりで検算して一致するか",
                   checked=int(len(picked)), mismatches=mismatches[:10], unchecked=unchecked)


#: 切り詰め検査で、その日の結果として消す列（発走前に分からない値）。
RESULT_COLUMNS_TEXT: tuple[str, ...] = (
    "result_time_sec", "result_jyuni_4c", "result_last_3f_sec", "result_ijyo_cd")


def blank_results(history: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """mask の行を「結果がまだ無い」状態にした写し（馬体重は発走前に発表されるので残す）。"""
    out = history.copy()
    out.loc[mask, "has_result"] = False
    for col in RESULT_COLUMNS_TEXT:
        if col in out:
            out.loc[mask, col] = ""
    out.loc[mask, "result_rank"] = 0
    for col in ("result_margin_sec", "target_top3", "target_win"):
        out.loc[mask, col] = np.nan
    return out


def check_truncation_sample(history: pd.DataFrame, feature_frame: pd.DataFrame, *, name: str,
                            columns: Sequence[str], builder, days: int = 2, seed: int = 0,
                            only: pd.Index | None = None) -> dict:
    """切り詰め検査: 無作為の日 D について、D より後の行を捨て D の結果を消した履歴で
    ``builder`` を作り直し、D の行の ``columns`` が feature_frame と一致するか。

    一致すれば、その列は D より後の情報も D の結果も使っていない（行ごとの再計算を書かずに
    列の全体をまとめて確かめられる）。builder は履歴 → columns を持つ表。
    """
    absent = [c for c in columns if c not in feature_frame.columns]
    if absent:
        return _result(name, "fail", f"{absent} が特徴量にありません。")
    rows = history[history["has_result"].astype(bool)]
    if only is not None:
        rows = rows[rows.index.isin(only)]
    dates = np.sort(rows["race_date"].unique())
    if len(dates) == 0:
        return _result(name, "warn", "検算できる日がありません。")
    rng = np.random.default_rng(seed)
    picked = rng.choice(dates, size=min(days, len(dates)), replace=False)
    mismatches, checked = [], 0
    for day in picked:
        part = history[history["race_date"] <= day]
        part = blank_results(part, part["race_date"] == day)
        rebuilt = builder(part)
        index = part.index[part["race_date"] == day]
        checked += len(index)
        for col in columns:
            a = feature_frame.loc[index, col].to_numpy(dtype=float)
            b = rebuilt.loc[index, col].to_numpy(dtype=float)
            same = np.isclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True)
            if not same.all():
                mismatches.append({"date": str(pd.Timestamp(day).date()), "column": col,
                                   "rows": int((~same).sum())})
    return _result(name, "ok" if not mismatches else "fail",
                   f"{len(columns)} 列を、その日より後の行とその日の結果を消した履歴で作り直して一致するか",
                   days=[str(pd.Timestamp(d).date()) for d in picked], checked=checked,
                   mismatches=mismatches[:10])


def summarize(checks: Sequence[dict]) -> str:
    lines = []
    for check in checks:
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[check["status"]]
        lines.append(f"[{mark}] {check['name']}: {check['message']}")
        if check["status"] != "ok":
            for key, value in check["detail"].items():
                lines.append(f"         {key}: {value}")
    return "\n".join(lines)

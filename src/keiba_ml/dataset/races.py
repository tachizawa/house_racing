"""既存のデータ版（races.csv / race_horses.csv）から、学習用の出走表を作る。

既存モデルの ``keiba.data.repository`` は「1レースを as_of 付きで引く」ための層で、
数十万行を一括で扱う用途には向かない。ここでは同じ規則（data_kubun の優先順位・
異常区分の扱い）を**表全体に一度に**当てる。規則の定数は ``keiba.config`` のものを
そのまま使い、二重定義しない。

出力は1行 = 1頭 × 1レース。結果の列は ``target_`` / ``result_`` 接頭辞を付け、
特徴量の列と名前で区別できるようにする（integrity.check_feature_columns が検査）。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from keiba import config as keiba_config
from keiba.ingest.store import current_version, read_manifest, version_dir
from keiba_ml.config import CodeConfig, DataConfig
from keiba_ml.dataset import codes

_RACE_USECOLS = (
    "race_id", "hasso_at", "jyo_cd", "kyori_m", "track_cd", "baba_cd",
    "grade_cd", "jyoken_cd_youngest", "syusso_tosu", "data_kubun", "make_date",
)
_RUNNER_USECOLS = (
    "race_id", "umaban", "ketto_num", "bamei", "sex_cd", "barei",
    "result_ijyo_cd", "result_kakutei_jyuni", "data_kubun", "make_date",
)

_RESULT_KUBUN = keiba_config.DATA_KUBUN_RESULT_PRIORITY
_PRIORITY = keiba_config.DATA_KUBUN_RESULT_PRIORITY + keiba_config.DATA_KUBUN_ENTRY_PRIORITY


def resolve_race_versions(data_dir: Path, versions: Sequence[str]) -> list[str]:
    if versions:
        return list(versions)
    current = current_version(data_dir)
    if current is None:
        raise FileNotFoundError(
            f"{data_dir} にデータ版がありません。scripts/update_data.py を先に実行してください。")
    return [current]


def _read(path: Path, usecols: Sequence[str]) -> pd.DataFrame:
    header = pd.read_csv(path, dtype=str, nrows=0).columns
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"{path} に必要な列がありません: {missing}")
    return pd.read_csv(path, dtype=str, na_filter=False, usecols=list(usecols))


def load_raw(data_dir: Path, versions: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """指定した版の races / race_horses を縦に連結する（重複はまだ残る）。"""
    races, runners = [], []
    for version_id in versions:
        path = version_dir(data_dir, version_id)
        read_manifest(path)       # 未完成・壊れた版は DataVersionError で止める
        races.append(_read(path / "races.csv", _RACE_USECOLS))
        runners.append(_read(path / "race_horses.csv", _RUNNER_USECOLS))
    return pd.concat(races, ignore_index=True), pd.concat(runners, ignore_index=True)


def pick_by_data_kubun(rows: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """同じキーの複数行から、結果側を優先・make_date の新しいものを1行選ぶ。

    keiba.data.repository._pick_by_data_kubun と同じ規則を表全体に当てる。
    """
    rows = rows[~rows["data_kubun"].isin(keiba_config.DATA_KUBUN_DROP)]
    rows = rows[rows["data_kubun"].isin(_PRIORITY)].copy()
    rank = {kubun: i for i, kubun in enumerate(_PRIORITY)}
    rows["_rank"] = rows["data_kubun"].map(rank)
    rows = rows.sort_values(["_rank", "make_date"], ascending=[True, False], kind="mergesort")
    rows = rows.drop_duplicates(subset=list(keys), keep="first")
    return rows.drop(columns="_rank")


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce").astype("Int64")


def build_runners(races: pd.DataFrame, runners: pd.DataFrame, *,
                  data: DataConfig, codes_config: CodeConfig) -> pd.DataFrame:
    """重複を除き、コードを正規化した出走表を返す。"""
    races = pick_by_data_kubun(races, ["race_id"])
    runners = pick_by_data_kubun(runners, ["race_id", "umaban"])

    races = races.copy()
    races["race_date"] = pd.to_datetime(races["race_id"].str[:8], format="%Y%m%d",
                                        errors="coerce")
    races = races[races["race_date"].notna()]
    races["track_kind"] = races["track_cd"].map(codes.track_kind)
    if data.jra_only:
        races = races[races["jyo_cd"].isin(keiba_config.JYO_CD_TO_NAME)]
    kinds = {"turf", "dirt"} | ({"jump"} if data.include_jump else set())
    races = races[races["track_kind"].isin(kinds)]
    races["kyori_m"] = _to_int(races["kyori_m"])
    races["baba_cd"] = races["baba_cd"].map(codes.baba_cd)
    races["grade_tier"] = [codes.grade_tier(g, j) for g, j in
                           zip(races["grade_cd"], races["jyoken_cd_youngest"])]
    races["distance_band"] = races["kyori_m"].map(
        lambda v: codes.distance_band(None if pd.isna(v) else int(v)))
    races = races.rename(columns={"data_kubun": "race_data_kubun"})
    race_cols = ["race_id", "race_date", "hasso_at", "jyo_cd", "kyori_m", "track_kind",
                 "distance_band", "baba_cd", "grade_tier", "race_data_kubun"]

    df = runners.merge(races[race_cols], on="race_id", how="inner")
    df["ketto_num"] = df["ketto_num"].str.strip()
    df = df[df["ketto_num"].str.strip("0") != ""]
    df["has_result"] = df["data_kubun"].isin(_RESULT_KUBUN)
    ijyo = df["result_ijyo_cd"].str.strip()
    # 出走取消・発走除外・競走除外（1/2/3）は走っていないので出走集合から抜く。
    # 競走中止・失格（4/5）は走ったので残し、3着以内ではないものとして扱う。
    df = df[~(df["has_result"] & ijyo.isin(keiba_config.IJYO_EXCLUDED))].copy()
    df["result_rank"] = _to_int(df["result_kakutei_jyuni"])
    df.loc[~df["has_result"], "result_rank"] = pd.NA

    df["year"] = df["race_date"].dt.year.astype(int)
    df["month"] = df["race_date"].dt.month.astype(int)
    df["sex_cd"] = df["sex_cd"].map(codes.sex_cd)
    df["barei_raw"] = df["barei"]
    df["age_from_barei"] = [
        codes.age_from_barei(b, y, kazoe_until_year=codes_config.kazoe_age_until_year)
        for b, y in zip(df["barei"], df["year"])
    ]
    df["field_size"] = df.groupby("race_id")["umaban"].transform("size").astype(int)

    has = df["has_result"]
    rank = df["result_rank"]
    # 競走中止・失格（4/5）は着順の欄に値があっても3着以内・勝ちにしない
    # （keiba.models の is_winner と同じ扱い）。
    can_finish = ~df["result_ijyo_cd"].str.strip().isin(keiba_config.IJYO_NO_TIME)
    df["target_top3"] = np.where(has, (can_finish & (rank.fillna(99) <= 3)).astype(float), np.nan)
    df["target_win"] = np.where(has, (can_finish & (rank.fillna(99) == 1)).astype(float), np.nan)
    keep = ["race_id", "race_date", "hasso_at", "year", "month", "jyo_cd", "kyori_m",
            "track_kind", "distance_band", "baba_cd", "grade_tier", "field_size",
            "umaban", "ketto_num", "bamei", "sex_cd", "barei_raw", "age_from_barei",
            "has_result", "result_ijyo_cd", "result_rank", "target_top3", "target_win",
            "data_kubun", "race_data_kubun"]
    df = df[keep].rename(columns={"data_kubun": "runner_data_kubun"})
    return df.sort_values(["race_date", "race_id", "umaban"], kind="mergesort").reset_index(drop=True)


def load_runners(data_dir: Path, *, data: DataConfig, codes_config: CodeConfig) -> pd.DataFrame:
    versions = resolve_race_versions(data_dir, data.race_versions)
    races, runners = load_raw(data_dir, versions)
    df = build_runners(races, runners, data=data, codes_config=codes_config)
    df.attrs["race_versions"] = versions
    return df


def relevance(rows: pd.DataFrame) -> np.ndarray:
    """順位学習と nDCG の関連度（1着 3 / 2着 2 / 3着 1 / それ以外 0）。

    target_top3 と同じ規則にするため、競走中止・失格（target_top3 が 0）は着順の欄に
    値があっても 0。結果の無い行（target_top3 が NaN）も 0。
    """
    rank = rows["result_rank"].astype("Float64").fillna(99).to_numpy(dtype=float)
    return np.where(rows["target_top3"].to_numpy(dtype=float) == 1, 4.0 - rank, 0.0)

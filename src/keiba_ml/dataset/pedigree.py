"""血統の階層化（Phase 1「血統データの階層化」）。

入力は次のどちらか（``DataConfig.pedigree_source``）。
  - "blood": 既存の血統の版（data/blood/versions/。pedigrees.csv / breeding_horses.csv。
    dataspec BLDN）。列名をこの層の名前（anc_<位置> / father_num / mother_num）に
    読み替える。UM の3代血統は持たないので補完は無い
  - "ml": ingest/pedigree_store が書いた版（sk.csv / hn.csv / um_ketto.csv）
出力は「血統登録番号（馬）→ 祖先の繁殖登録番号」の横持ちの表で、列は
``ped_<位置>``（位置は S=父 / D=母 を左から辿った文字列。``ped_DS`` = 母父）。

埋め方:
  1. 3世代（14枠）は SK（産駒マスタ）を優先し、空いた枠を UM（競走馬マスタ）で補う。
  2. まだ空いている枠と 4世代目以降は、繁殖馬マスタ HN の「父・母の繁殖登録番号」を
     親の枠から1世代ずつ辿って埋める。
  3. SK/UM が埋めた 2〜3世代の枠を、HN から辿った値と突き合わせ、一致率を
     ``stats["parent_link"]`` に残す。位置（オフセット）を間違えていればここが崩れる。

血統は時点に依存しない事実なので as_of は取らない。ただし「その馬が生まれる前に
繁殖登録されていた」ことは自明なので、未来情報の混入にはならない。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from keiba.ingest import records as keiba_records
from keiba.ingest import store as keiba_store
from keiba_ml.config import KETTO3_PATHS, all_paths
from keiba_ml.ingest.pedigree_records import UM_KETTO_COLUMNS
from keiba_ml.ingest.pedigree_store import (
    PedigreeVersionError, complete_versions, read_manifest, version_dir,
)

#: SK/HN/UM の「削除」データ区分。最新の行がこれならそのキーは無かったことにする。
DELETED_KUBUN = "0"

#: 既存の血統の版の列 → この層の列。SK の祖先の並びはどちらも仕様書の
#: HansyokuNum[0..13] の順（テストで固定）。
BLOOD_SK_RENAME: dict[str, str] = {
    f"{name}_num": f"anc_{path}"
    for name, path in zip(keiba_records.SK_ANCESTOR_NAMES, KETTO3_PATHS, strict=True)
}
BLOOD_HN_RENAME: dict[str, str] = {"sire_num": "father_num", "dam_num": "mother_num"}


def _read(path: Path) -> pd.DataFrame:
    # PedigreeVersionWriter は0行でも3つの CSV を必ず書く。無いのは版が壊れているので、
    # 空として続けず止める（欠けた血統のままデータセットを作らない）。
    if not path.exists():
        raise PedigreeVersionError(f"血統の版のファイルがありません: {path}")
    return pd.read_csv(path, dtype=str, na_filter=False)


def _latest(frames: list[pd.DataFrame], key: str) -> pd.DataFrame:
    """版の古い順に積んだ行から、キーごとに最後（新しい版・新しい make_date）を残す。"""
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df[key] != ""]
    df = df.sort_values(["_version_order", "make_date"], kind="mergesort")
    df = df.drop_duplicates(subset=[key], keep="last")
    df = df[df["data_kubun"] != DELETED_KUBUN]
    return df.drop(columns="_version_order").reset_index(drop=True)


def load_sources(ml_data_dir: Path, versions: Sequence[str] = ()) -> dict[str, pd.DataFrame]:
    """血統の版を読み、SK / HN / UM をキーごとに最新1行へ絞って返す。"""
    versions = list(versions) or complete_versions(ml_data_dir)
    if not versions:
        raise FileNotFoundError(
            "血統の版がありません。scripts/ml/update_pedigree.py を先に実行してください。")
    sk, hn, um = [], [], []
    for order, version_id in enumerate(versions):
        path = version_dir(ml_data_dir, version_id)
        read_manifest(path)
        for bucket, name in ((sk, "sk.csv"), (hn, "hn.csv"), (um, "um_ketto.csv")):
            frame = _read(path / name)
            if not frame.empty:
                frame["_version_order"] = order
            bucket.append(frame)
    sources = {
        "SK": _latest(sk, "ketto_num"),
        "HN": _latest(hn, "hansyoku_num"),
        "UM": _latest(um, "ketto_num"),
    }
    sources["versions"] = versions  # type: ignore[assignment]
    return sources


def complete_blood_versions(blood_data_dir: Path) -> list[str]:
    """既存の血統の版のうち完成済みのものを古い順に。.partial と壊れた版は含めない。"""
    root = Path(blood_data_dir) / "versions"
    if not root.exists():
        return []
    result = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        try:
            keiba_store.read_manifest(path)
        except keiba_store.DataVersionError:
            continue
        result.append(path.name)
    return result


def load_blood_sources(blood_data_dir: Path, versions: Sequence[str] = ()
                       ) -> dict[str, pd.DataFrame]:
    """既存の血統の版（BLDN）を読み、load_sources と同じ形で返す。

    versions が空なら完成済みの版を全部、古い順に重ねて新しい行を勝たせる。
    CURRENT の1版だけにしないのは、セットアップ取得の後に option 1 で取った版が
    差分しか持たず、CURRENT がそれを指すことがあるため（重ねないと SK/HN の大半が落ちる）。
    UM は空（既存の版は UM の3代血統を持たない）。
    """
    versions = list(versions) or complete_blood_versions(blood_data_dir)
    if not versions:
        raise FileNotFoundError(
            f"{blood_data_dir} に血統の版がありません。"
            " scripts/update_data.py --kind blood を先に実行してください。")
    sk, hn = [], []
    for order, version_id in enumerate(versions):
        path = keiba_store.version_dir(blood_data_dir, version_id)
        keiba_store.read_manifest(path)   # 未完成・壊れた版は DataVersionError で止める
        for bucket, name, rename in ((sk, "pedigrees.csv", BLOOD_SK_RENAME),
                                     (hn, "breeding_horses.csv", BLOOD_HN_RENAME)):
            frame = _read(path / name)
            missing = sorted(set(rename) - set(frame.columns))
            if missing:
                raise PedigreeVersionError(f"{path / name} に必要な列がありません: {missing}")
            frame = frame.rename(columns=rename)
            if not frame.empty:
                frame["_version_order"] = order
            bucket.append(frame)
    sources = {
        "SK": _latest(sk, "ketto_num"),
        "HN": _latest(hn, "hansyoku_num"),
        "UM": pd.DataFrame(columns=list(UM_KETTO_COLUMNS), dtype=str),
    }
    sources["versions"] = versions  # type: ignore[assignment]
    return sources


def _year_from_date(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str[:4].replace("", np.nan), errors="coerce").astype("Int64")


def build_pedigree(sk: pd.DataFrame, hn: pd.DataFrame, um: pd.DataFrame, *,
                   generations: int, inbreeding_generations: int,
                   horses: Sequence[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """馬ごとの祖先表と、埋まり具合・整合性の統計を返す。

    horses を渡すとその馬だけを対象にする（出走表に出てくる馬だけ作れば足りる）。
    """
    paths = all_paths(generations)
    ketto3 = [f"anc_{p}" for p in KETTO3_PATHS]
    for frame, cols in ((sk, ["ketto_num", "birth_date", *ketto3]),
                        (um, ["ketto_num", "birth_date", *ketto3])):
        for col in cols:
            if col not in frame.columns:
                frame[col] = pd.Series(dtype=str)

    keys = pd.Index(sorted(set(sk["ketto_num"]) | set(um["ketto_num"])), name="ketto_num")
    if horses is not None:
        keys = pd.Index(sorted(set(horses)), name="ketto_num")
    sk_i = sk.set_index("ketto_num").reindex(keys)
    um_i = um.set_index("ketto_num").reindex(keys)

    ped = pd.DataFrame(index=keys)
    from_sk = pd.Series(False, index=keys)
    from_um = pd.Series(False, index=keys)
    for path in KETTO3_PATHS:
        if path not in paths:
            continue
        a = sk_i[f"anc_{path}"].fillna("")
        b = um_i[f"anc_{path}"].fillna("")
        ped[f"ped_{path}"] = a.where(a != "", b)
        from_sk |= a != ""
        from_um |= (a == "") & (b != "")
    source = np.where(from_sk & from_um, "SK+UM",
                      np.where(from_sk, "SK", np.where(from_um, "UM", "")))
    ped["pedigree_source"] = source

    birth = sk_i["birth_date"].fillna("")
    birth = birth.where(birth != "", um_i["birth_date"].fillna(""))
    ped["birth_date"] = birth
    ped["birth_year"] = _year_from_date(birth)

    # --- HN の父母リンクで埋める・突き合わせる --------------------------------
    father = dict(zip(hn.get("hansyoku_num", []), hn.get("father_num", [])))
    mother = dict(zip(hn.get("hansyoku_num", []), hn.get("mother_num", [])))
    link_checked = link_agreed = filled_from_hn = 0
    for path in paths:
        col = f"ped_{path}"
        if len(path) == 1:
            if col not in ped:
                ped[col] = ""
            continue
        parent = ped[f"ped_{path[:-1]}"]
        lookup = father if path[-1] == "S" else mother
        derived = parent.map(lambda h: lookup.get(h, "") if h else "")
        if col in ped:
            current = ped[col].fillna("")
            both = (current != "") & (derived != "")
            link_checked += int(both.sum())
            link_agreed += int((both & (current == derived)).sum())
            fill = (current == "") & (derived != "")
            filled_from_hn += int(fill.sum())
            ped[col] = current.where(~fill, derived)
        else:
            ped[col] = derived
            filled_from_hn += int((derived != "").sum())

    ordered = [f"ped_{p}" for p in paths]
    ped[ordered] = ped[ordered].fillna("")
    inbreeding = inbreeding_table(ped, inbreeding_generations)
    ped = ped.join(inbreeding)
    ped = ped[[*ordered, "pedigree_source", "birth_date", "birth_year",
               "inbreeding_f", "n_cross", "closest_cross"]]

    stats = {
        "horses": int(len(ped)),
        "with_any_pedigree": int((ped["pedigree_source"] != "").sum()),
        "source_counts": {k: int(v) for k, v in
                          ped["pedigree_source"].replace("", "none").value_counts().items()},
        "fill_rate_by_generation": {
            str(g): float((ped[[f"ped_{p}" for p in paths if len(p) == g]] != "")
                          .to_numpy().mean()) if len(ped) else 0.0
            for g in range(1, generations + 1)
        },
        "parent_link": {
            "checked": link_checked, "agreed": link_agreed,
            "rate": (link_agreed / link_checked) if link_checked else None,
        },
        "filled_from_hn": filled_from_hn,
    }
    return ped, stats


def inbreeding_table(ped: pd.DataFrame, generations: int) -> pd.DataFrame:
    """父系と母系の両方に現れる祖先（クロス）から、近交係数を Wright の式で近似する。

        F ≈ Σ_{共通祖先 A} Σ_{父側の出現 i, 母側の出現 j} (1/2)^(i + j - 1)

    i, j は馬から見た世代（父=1、父父=2 ...）。共通祖先自身の近交（1+F_A）と、
    経路の重なり（同じ個体を2回通る経路）の除外は省く近似。closest_cross は
    最も近いクロスの i + j（例: 3×4 なら 7）。クロスが無ければ F=0、n_cross=0。
    """
    rows = []
    for path in all_paths(generations):
        col = f"ped_{path}"
        if col not in ped:
            continue
        series = ped[col]
        series = series[series != ""]
        rows.append(pd.DataFrame({"ketto_num": series.index, "anc": series.to_numpy(),
                                  "side": path[0], "depth": len(path)}))
    empty = pd.DataFrame({"inbreeding_f": 0.0, "n_cross": 0, "closest_cross": np.nan},
                         index=ped.index)
    if not rows:
        return empty
    long = pd.concat(rows, ignore_index=True)
    sire = long[long["side"] == "S"][["ketto_num", "anc", "depth"]]
    dam = long[long["side"] == "D"][["ketto_num", "anc", "depth"]]
    pairs = sire.merge(dam, on=["ketto_num", "anc"], suffixes=("_s", "_d"))
    if pairs.empty:
        return empty
    pairs["contrib"] = 0.5 ** (pairs["depth_s"] + pairs["depth_d"] - 1)
    pairs["cross"] = pairs["depth_s"] + pairs["depth_d"]
    grouped = pairs.groupby("ketto_num").agg(
        inbreeding_f=("contrib", "sum"), n_cross=("anc", "nunique"),
        closest_cross=("cross", "min"))
    result = empty.copy()
    result.loc[grouped.index, ["inbreeding_f", "n_cross", "closest_cross"]] = \
        grouped[["inbreeding_f", "n_cross", "closest_cross"]].to_numpy()
    result["n_cross"] = result["n_cross"].astype(int)
    return result


def ancestor_names(hn: pd.DataFrame, um: pd.DataFrame) -> dict[str, str]:
    """繁殖登録番号 → 馬名。HN を優先し、無ければ UM の3代血統の馬名で補う。"""
    names: dict[str, str] = {}
    for path in KETTO3_PATHS:
        num, name = f"anc_{path}", f"name_{path}"
        if num in um.columns and name in um.columns:
            for h, n in zip(um[num], um[name]):
                if h and n:
                    names.setdefault(h, n)
    if "hansyoku_num" in hn.columns:
        for h, n in zip(hn["hansyoku_num"], hn["bamei"]):
            if h and n:
                names[h] = n
    return names

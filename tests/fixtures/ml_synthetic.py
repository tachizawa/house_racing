"""血統モデルの結合テスト用の合成世界。

父の「能力」「成長型（晩成ほど年齢とともに伸びる）」「芝ダートの向き」と、
母父の能力が子の着順に効くように作る。血統モデルがこれを拾えれば、
偶然より明らかに上位3頭が当たる。

生成物:
- 既存形式のデータ版（data/versions/<id>/races.csv ほか）… tests.fixtures.csv_data で書く
- 血統の版（data/ml/pedigree/versions/<id>/）… 合成バイト列 → pedigree_records → 保存
  （分解の経路も通す）
- 既存形式の血統の版（data/blood/versions/<id>/）… 同じ合成バイト列 → keiba.ingest.records
  → keiba.ingest.store。BLDN は UM を持たないので、UM だけの馬はこちらに入らない
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from keiba.ingest import records as keiba_records
from keiba.ingest import store as keiba_store
from keiba_ml.ingest import pedigree_records, pedigree_store
from tests.fixtures import csv_data
from tests.fixtures import ml_records as mr


@dataclass
class Breeder:
    num: str
    sex: str
    father: str = ""
    mother: str = ""
    quality: float = 0.0
    maturity: float = 0.0
    turf: float = 0.0


@dataclass
class World:
    race_data_dir: Path
    ml_data_dir: Path
    blood_data_dir: Path
    breeders: dict[str, Breeder] = field(default_factory=dict)
    horses: dict[str, dict] = field(default_factory=dict)
    sires: list[str] = field(default_factory=list)
    entry_race_id: str = ""


def _ancestors(breeders: dict[str, Breeder], sire: str, dam: str) -> list[str]:
    """3代14枠を JV-Data の並び（父, 母, 父父, 父母, 母父, 母母, 父父父 ...）で返す。"""
    def parents(num: str) -> tuple[str, str]:
        b = breeders.get(num)
        return (b.father, b.mother) if b else ("", "")
    gen1 = [sire, dam]
    gen2 = [p for n in gen1 for p in parents(n)]
    gen3 = [p for n in gen2 for p in parents(n)]
    return gen1 + gen2 + gen3


def build_world(tmp_path: Path, *, first_year: int = 2011, last_year: int = 2022,
                races_per_year: int = 160, seed: int = 7,
                um_only_rate: float = 0.05) -> World:
    rng = np.random.default_rng(seed)
    breeders: dict[str, Breeder] = {}
    counter = iter(range(1, 10**6))

    def new(sex: str, father: str = "", mother: str = "", **kw) -> str:
        num = f"{1100000000 + next(counter)}"
        breeders[num] = Breeder(num=num, sex=sex, father=father, mother=mother, **kw)
        return num

    def layer(n_stallions: int, n_mares: int, prev: tuple[list[str], list[str]] | None):
        stallions, mares = [], []
        for sex, n, bucket in (("1", n_stallions, stallions), ("2", n_mares, mares)):
            for _ in range(n):
                if prev:
                    f, m = rng.choice(prev[0]), rng.choice(prev[1])
                    bucket.append(new(sex, str(f), str(m)))
                else:
                    bucket.append(new(sex))
        return stallions, mares

    l4 = layer(4, 12, None)
    l3 = layer(8, 30, l4)
    l2 = layer(12, 60, l3)
    for num in l2[0]:
        breeders[num].quality = float(rng.normal(0, 0.5))      # 母父としての効き
    l1_stallions, l1_mares = layer(12, 260, l2)
    for num in l1_stallions:
        b = breeders[num]
        b.quality = float(rng.normal(0, 0.6))
        b.maturity = float(rng.normal(0, 0.6))
        b.turf = float(rng.normal(0, 0.6))
    for num in l1_mares:
        b = breeders[num]
        b.quality = 0.6 * breeders[b.father].quality + float(rng.normal(0, 0.3))

    horses: dict[str, dict] = {}
    for birth_year in range(first_year - 6, last_year - 1):
        for _ in range(int(races_per_year * 10 / 3.5 / 2.5)):
            sire, dam = str(rng.choice(l1_stallions)), str(rng.choice(l1_mares))
            ketto = f"{birth_year}{len(horses) + 1:06d}"
            ability = (breeders[sire].quality + 0.5 * breeders[dam].quality
                       + float(rng.normal(0, 0.5)))
            horses[ketto] = {"sire": sire, "dam": dam, "birth_year": birth_year,
                             "ability": ability, "sex": str(rng.choice(["1", "2"])),
                             "bamei": f"ウマ{len(horses) + 1}"}

    races, runners = [], []
    by_year: dict[int, list[str]] = {}
    for ketto, h in horses.items():
        by_year.setdefault(h["birth_year"], []).append(ketto)
    for year in range(first_year, last_year + 1):
        pool = [k for age in range(2, 7) for k in by_year.get(year - age, [])]
        start = date(year, 1, 5)
        for r in range(races_per_year):
            day = start + timedelta(days=int(r * 350 / races_per_year))
            race_num = r % 12 + 1
            jyo = f"{(r // 12) % 10 + 1:02d}"
            race_id = f"{day:%Y%m%d}{jyo}0101{race_num:02d}"
            turf = bool(rng.random() < 0.55)
            n = int(rng.integers(8, 15))
            field_horses = rng.choice(pool, size=n, replace=False)
            perf = []
            for k in field_horses:
                h, s = horses[k], breeders[horses[k]["sire"]]
                age = year - h["birth_year"]
                p = (h["ability"] + 0.4 * s.maturity * (age - 3)
                     + 0.5 * s.turf * (1 if turf else -1) + float(rng.gumbel(0, 1.0)))
                perf.append(p)
            order = np.argsort(-np.array(perf))
            ranks = np.empty(n, dtype=int)
            ranks[order] = np.arange(1, n + 1)
            races.append(csv_data.make_race(
                race_id=race_id, hasso_at=f"{day:%Y-%m-%d}T{10 + race_num % 6:02d}:00:00+09:00",
                jyo_cd=jyo, kyori_m=str(int(rng.choice([1200, 1600, 2000, 2400]))),
                track_cd="17" if turf else "24", track_kind="turf" if turf else "dirt",
                baba_cd=str(int(rng.choice([1, 1, 1, 2, 3]))), grade_cd="",
                jyoken_cd_youngest="005", toroku_tosu=str(n), syusso_tosu=str(n),
                nyusen_tosu=str(n), data_kubun="7", make_date=f"{day + timedelta(days=1):%Y%m%d}"))
            for i, (k, rank) in enumerate(zip(field_horses, ranks)):
                h = horses[k]
                runners.append(csv_data.make_race_horse(
                    race_id=race_id, umaban=f"{i + 1:02d}", ketto_num=k, bamei=h["bamei"],
                    sex_cd=h["sex"], barei=f"{year - h['birth_year']:02d}",
                    result_kakutei_jyuni=str(int(rank)), data_kubun="7",
                    make_date=f"{day + timedelta(days=1):%Y%m%d}"))

    # 結果の出ていない出馬表のレース（予想スクリプト用）
    day = date(last_year, 12, 28)
    entry_race_id = f"{day:%Y%m%d}05010111"
    pool = [k for age in range(3, 6) for k in by_year.get(last_year - age, [])]
    entry = rng.choice(pool, size=10, replace=False)
    races.append(csv_data.make_race(race_id=entry_race_id, hasso_at=f"{day:%Y-%m-%d}T15:40:00+09:00",
                                    jyo_cd="05", kyori_m="2400", track_cd="17", track_kind="turf",
                                    baba_cd="1", data_kubun="2", make_date=f"{day:%Y%m%d}",
                                    syusso_tosu="10"))
    for i, k in enumerate(entry):
        h = horses[k]
        runners.append(csv_data.make_race_horse(
            race_id=entry_race_id, umaban=f"{i + 1:02d}", ketto_num=k, bamei=h["bamei"],
            sex_cd=h["sex"], barei=f"{last_year - h['birth_year']:02d}",
            result_kakutei_jyuni="", result_ijyo_cd="", data_kubun="2",
            make_date=f"{day:%Y%m%d}"))

    race_data_dir = tmp_path / "data"
    race_data_dir.mkdir()
    csv_data.write_dataset(race_data_dir, races, runners, [])

    ml_data_dir = race_data_dir / "ml"
    writer = pedigree_store.PedigreeVersionWriter(
        ml_data_dir, "20260101T000000Z", dataspecs=["BLOD"], fromdate="19860101000000",
        option=4)
    blood_data_dir = race_data_dir / "blood"
    blood = keiba_store.VersionWriter(
        blood_data_dir, "20260101T000000Z", dataspecs=["BLDN"], fromdate="19860101000000",
        option=4, files=keiba_store.BLOOD_FILES)
    for num, b in breeders.items():
        raw = mr.make_hn(hansyoku_num=num, bamei=f"シュボバ{num[-4:]}", sex_cd=b.sex,
                         father=b.father, mother=b.mother)
        writer.add(*pedigree_records.parse(raw))
        blood.add("HN", keiba_records.parse_hn(raw))
    for ketto, h in horses.items():
        ancestors = _ancestors(breeders, h["sire"], h["dam"])
        birth = f"{h['birth_year']}0401"
        if rng.random() < um_only_rate:
            writer.add(*pedigree_records.parse(mr.make_um(
                ketto_num=ketto, bamei=h["bamei"], birth_date=birth, sex_cd=h["sex"],
                ancestors=[(a, f"シュボバ{a[-4:]}") for a in ancestors])))
        else:
            raw = mr.make_sk(ketto_num=ketto, birth_date=birth, sex_cd=h["sex"],
                             ancestors=ancestors)
            writer.add(*pedigree_records.parse(raw))
            blood.add("SK", keiba_records.parse_sk(raw))
    writer.commit()
    blood.commit()
    return World(race_data_dir=race_data_dir, ml_data_dir=ml_data_dir,
                 blood_data_dir=blood_data_dir, breeders=breeders,
                 horses=horses, sires=l1_stallions, entry_race_id=entry_race_id)

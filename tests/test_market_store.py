"""払戻・オッズの保存。別の系統（data/market/）に、大きい表は gzip で書く。"""

import gzip

import pytest

from keiba.ingest import records, store
from tests.fixtures import records as fx


@pytest.fixture
def market(tmp_path):
    writer = store.VersionWriter(tmp_path, "20260926T010000Z", dataspecs=["RACE"],
                                 fromdate="20210901000000", option=4,
                                 files=store.MARKET_FILES,
                                 source={"raw_dir": "data/raw/20260925T172902Z"})
    for raw in (fx.make_hr(), fx.make_o1(),
                fx.make_odds("O6", slots=[(0, "01020300001500001")])):
        for key, row in records.parse_rows(raw):
            writer.add(key, row)
    writer.add("RA", records.parse_ra(fx.make_ra()))   # 組に無い種別は捨てる
    return writer.commit()


def test_every_table_key_of_hr_and_odds_has_a_file():
    keys = {k for spec in ("HR", "O1", "O2", "O3", "O4", "O5", "O6")
            for k in records.table_keys(spec)}
    assert keys == set(store.MARKET_FILES)


def test_large_odds_tables_are_gzipped(market):
    names = sorted(p.name for p in market.iterdir())
    assert "odds_sanrentan.csv.gz" in names
    assert "payouts.csv" in names and "races.csv" not in names


def test_gzipped_table_reads_back_as_csv(market):
    with gzip.open(market / "odds_sanrentan.csv.gz", "rt", encoding="utf-8") as handle:
        header, row = handle.read().splitlines()
    values = dict(zip(header.split(","), row.split(",")))
    assert values["combination"] == "01-02-03"
    assert values["odds"] == "15.0"
    assert values["popularity"] == "1"
    assert values["fetched_at"]


def test_manifest_hashes_the_compressed_bytes(market):
    manifest = store.read_manifest(market)
    entry = next(f for f in manifest["files"] if f["name"] == "odds_sanrentan.csv.gz")
    assert entry["rows"] == 1
    assert entry["sha256"] == store._sha256(market / "odds_sanrentan.csv.gz")


def test_manifest_keeps_the_source(market):
    manifest = store.read_manifest(market)
    assert manifest["source"] == {"raw_dir": "data/raw/20260925T172902Z"}
    assert manifest["record_counts"]["payout_races"] == 1


def test_race_manifest_has_no_source_key(tmp_path):
    writer = store.VersionWriter(tmp_path, "20260926T020000Z", dataspecs=["RACE"],
                                 fromdate="20260301000000", option=1)
    assert "source" not in store.read_manifest(writer.commit())

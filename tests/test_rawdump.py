"""生レコードの保存と読み戻し。JV-Link を叩き直さずにパーサを通し直すため。"""

import gzip
import json

import pytest

from keiba.ingest import rawdump
from tests.fixtures import records as fx


def _dump(path, raws):
    with rawdump.RawDumpWriter(path, dataspec="RACE", fromdate="20210901000000",
                               option=4) as writer:
        for raw in raws:
            writer.write(raw)


def test_round_trip_keeps_bytes_and_groups_by_spec(tmp_path):
    ra, se, hr = fx.make_ra(), fx.make_se(), fx.make_hr()
    _dump(tmp_path, [ra, se, hr, se])
    assert sorted(p.name for p in tmp_path.glob("*.bin.gz")) == [
        "HR.bin.gz", "RA.bin.gz", "SE.bin.gz"]
    assert list(rawdump.iter_records(tmp_path, ["SE"])) == [se, se]
    assert sorted(rawdump.iter_records(tmp_path)) == sorted([ra, se, hr, se])


def test_manifest_records_counts_and_completion(tmp_path):
    _dump(tmp_path, [fx.make_ra()])
    manifest = rawdump.read_manifest(tmp_path)
    assert manifest["completed"] is True
    assert manifest["counts"] == {"RA": 1}
    assert (manifest["dataspec"], manifest["fromdate"], manifest["option"]) == (
        "RACE", "20210901000000", 4)


def test_failure_marks_the_dump_incomplete(tmp_path):
    with pytest.raises(RuntimeError):
        with rawdump.RawDumpWriter(tmp_path, dataspec="RACE", fromdate="20210901000000",
                                   option=4) as writer:
            writer.write(fx.make_ra())
            raise RuntimeError("切断")
    with pytest.raises(rawdump.RawDumpError, match="完了していません"):
        rawdump.read_manifest(tmp_path)


def test_reads_the_format_written_by_the_first_dump(tmp_path):
    """2026-09-26 の取得（tmp/fetch_raw/dump_raw.py）と同じ形式・同じ manifest のキー。"""
    raw = fx.make_hr()
    with gzip.open(tmp_path / "HR.bin.gz", "wb") as handle:
        handle.write(len(raw).to_bytes(4, "big") + raw)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataspec": "RACE", "fromdate": "20210901000000", "option": 4,
        "completed": True, "counts": {"HR": 1}}), encoding="utf-8")
    assert rawdump.read_manifest(tmp_path)["dataspec"] == "RACE"
    assert list(rawdump.iter_records(tmp_path)) == [raw]


def test_truncated_file_is_an_error(tmp_path):
    raw = fx.make_ra()
    with gzip.open(tmp_path / "RA.bin.gz", "wb") as handle:
        handle.write(len(raw).to_bytes(4, "big") + raw[:-10])
    with pytest.raises(rawdump.RawDumpError, match="途中で切れて"):
        list(rawdump.iter_records(tmp_path))

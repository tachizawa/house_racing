"""調教（HC 坂路・WC ウッドチップ）の取り込み。COM は叩かない。"""

import gzip

import pandas as pd
import pytest

from keiba.ingest import rawdump, records, store
from tests.test_update_data import update_data


def make_hc(*, kubun="1", tresen="1", date="20261020", time="0834", ketto="2016100126",
            t4="0596", l4="166", t3="0430", l3="154", t2="0276", l2="144", l1="132") -> bytes:
    raw = f"HC{kubun}20261020{tresen}{date}{time}{ketto}{t4}{l4}{t3}{l3}{t2}{l2}{l1}\r\n"
    data = raw.encode("cp932")
    assert len(data) == records.HC_LENGTH
    return data


def make_wc(*, kubun="1", tresen="0", date="20261020", time="0634", ketto="2018100664",
            course="2", around="1", furlongs=None, l1="165") -> bytes:
    # 10〜2 ハロンの (合計, ラップ)。既定は 6ハロンから計測（それより長い区間は 0000 / 000）
    furlongs = furlongs or {6: ("0831", "131"), 5: ("0700", "130"), 4: ("0570", "128"),
                            3: ("0442", "135"), 2: ("0307", "142")}
    body = "".join(furlongs.get(f, ("0000", "000"))[0] + furlongs.get(f, ("0000", "000"))[1]
                   for f in range(10, 1, -1))
    raw = f"WC{kubun}20261020{tresen}{date}{time}{ketto}{course}{around}0{body}{l1}\r\n"
    data = raw.encode("cp932")
    assert len(data) == records.WC_LENGTH
    return data


class TestParse:
    def test_hc_fields_and_units(self):
        row = records.parse_hc(make_hc())
        assert row["tresen_kubun"] == "1" and row["chokyo_date"] == "2026-10-20"
        assert row["chokyo_time"] == "0834" and row["ketto_num"] == "2016100126"
        assert row["time_4f"] == 59.6 and row["lap_4"] == 16.6
        assert row["time_3f"] == 43.0 and row["lap_3"] == 15.4
        assert row["time_2f"] == 27.6 and row["lap_2"] == 14.4 and row["lap_1"] == 13.2
        # 4ハロン = 最初のラップ + 3ハロン
        assert row["time_4f"] == pytest.approx(row["lap_4"] + row["time_3f"])

    def test_hc_bad_measurement_is_missing(self):
        """600m 計測（2004-11 以前の美浦）や測定不良は 0000 / 000。0 秒にしない。"""
        row = records.parse_hc(make_hc(t4="0000", l4="000"))
        assert row["time_4f"] is None and row["lap_4"] is None
        assert row["time_3f"] == 43.0

    def test_wc_fields_units_and_caps(self):
        row = records.parse_wc(make_wc())
        assert row["course"] == "2" and row["baba_around"] == "1"
        assert row["time_10f"] is None and row["lap_10"] is None
        assert row["time_6f"] == 83.1 and row["lap_6"] == 13.1
        assert row["time_2f"] == 30.7 and row["lap_2"] == 14.2 and row["lap_1"] == 16.5
        capped = records.parse_wc(make_wc(furlongs={10: ("9999", "999")}))
        assert capped["time_10f"] is None and capped["lap_10"] is None

    def test_columns_match_store(self):
        assert tuple(records.parse_hc(make_hc())) == store.SLOPE_COLUMNS
        assert tuple(records.parse_wc(make_wc())) == store.WOOD_COLUMNS

    @pytest.mark.parametrize("maker", [make_hc, make_wc])
    def test_wrong_length_is_refused(self, maker):
        with pytest.raises(ValueError, match="長さ"):
            records.parse_rows(maker()[:-3]).__next__()


def _raw(path, dataspec, items, fromdate="20030101000000"):
    with rawdump.RawDumpWriter(path, dataspec=dataspec, fromdate=fromdate, option=4) as writer:
        for raw in items:
            writer.write(raw)
    return path


class TestTrainingKind:
    def test_defaults(self):
        args = update_data._parse_args(["--kind", "training", "--fromdate", "20030101000000"])
        assert update_data._resolve_dataspecs(args) == ("SLOP", "WOOD")
        assert update_data._resolve_data_dir(args) == update_data.config.TRAINING_DATA_DIR

    def test_two_raw_dumps_make_one_version(self, tmp_path, monkeypatch):
        def refuse(*a, **k):
            raise AssertionError("JV-Link を呼んではいけない")
        monkeypatch.setattr(update_data.jvlink, "open_link", refuse)
        slop = _raw(tmp_path / "slop", "SLOP", [make_hc(), make_hc(kubun="0")])
        wood = _raw(tmp_path / "wood", "WOOD", [make_wc()])
        out = tmp_path / "training"
        assert update_data.main(["--kind", "training", "--from-raw", str(slop),
                                 "--from-raw", str(wood), "--data-dir", str(out)]) == 0
        version = out / "versions" / store.current_version(out)
        manifest = store.read_manifest(version)
        assert manifest["dataspecs"] == ["SLOP", "WOOD"]
        assert manifest["record_counts"] == {"HC": 2, "WC": 1}
        assert [r["dataspec"] for r in manifest["source"]["raw"]] == ["SLOP", "WOOD"]
        with gzip.open(version / "slope.csv.gz", "rt", encoding="utf-8") as handle:
            slope = pd.read_csv(handle, dtype=str)
        # 削除（データ区分 0）も保存はする（捨てるのは読む側）
        assert list(slope["data_kubun"]) == ["1", "0"]

    def test_mismatched_fromdate_is_refused(self, tmp_path):
        slop = _raw(tmp_path / "slop", "SLOP", [make_hc()])
        wood = _raw(tmp_path / "wood", "WOOD", [make_wc()], fromdate="20210701000000")
        with pytest.raises(SystemExit, match="fromdate"):
            update_data.main(["--kind", "training", "--from-raw", str(slop),
                              "--from-raw", str(wood), "--data-dir", str(tmp_path / "t")])

    def test_same_dataspec_twice_is_refused(self, tmp_path):
        a = _raw(tmp_path / "a", "SLOP", [make_hc()])
        b = _raw(tmp_path / "b", "SLOP", [make_hc()])
        with pytest.raises(SystemExit, match="同じ dataspec"):
            update_data.main(["--kind", "training", "--from-raw", str(a),
                              "--from-raw", str(b), "--data-dir", str(tmp_path / "t")])

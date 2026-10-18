"""update_data の --kind market と --from-raw。COM は叩かない。"""

import json

import pytest

from keiba.ingest import rawdump, store
from tests.fixtures import records as fx
from tests.test_update_data import DataspecAwareLink, update_data


def _manifest(data_dir):
    return store.read_manifest(data_dir / "versions" / store.current_version(data_dir))


class TestMarketKind:
    """--kind market は払戻・オッズを、レースとは別の系統に保存する。"""

    def test_defaults_to_race_dataspec_and_its_own_dir(self):
        args = update_data._parse_args(["--kind", "market", "--fromdate", "20210901000000"])
        assert update_data._resolve_dataspecs(args) == ("RACE",)
        assert update_data._resolve_data_dir(args) == update_data.config.MARKET_DATA_DIR

    def test_writes_payouts_and_odds_from_jvlink(self, tmp_path, monkeypatch, capsys):
        link = DataspecAwareLink(
            open_codes={"RACE": 0},
            records_by_dataspec={"RACE": [fx.make_ra(), fx.make_hr(),
                                              fx.make_o1(tansyo=[(0, "01003401")])]})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        exit_code = update_data.main([
            "--kind", "market", "--fromdate", "20210901000000", "--option", "4",
            "--data-dir", str(tmp_path)])

        assert exit_code == 0
        manifest = _manifest(tmp_path)
        assert manifest["record_counts"]["payout_races"] == 1
        assert manifest["record_counts"]["odds_tansyo"] > 0
        out = capsys.readouterr().out
        assert "RA: 1（保存対象外）" in out
        assert "HR: 1\n" in out and "O1: 1\n" in out

    def test_race_kind_still_ignores_payouts(self, tmp_path, monkeypatch, capsys):
        link = DataspecAwareLink(open_codes={"RACE": 0},
                                 records_by_dataspec={"RACE": [fx.make_ra(), fx.make_hr()]})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)
        update_data.main(["--dataspec", "RACE", "--fromdate", "20260801000000",
                          "--data-dir", str(tmp_path)])
        assert _manifest(tmp_path)["record_counts"] == {"RA": 1, "SE": 0, "UM": 0}
        assert "HR: 1（保存対象外）" in capsys.readouterr().out

    def test_race_kind_does_not_parse_records_it_does_not_save(self, tmp_path, monkeypatch, capsys):
        """保存しない種別はパースしない。長さが仕様と違う O1 でも race の版の取り込みは止まらない。"""
        broken_o1 = fx.make_o1()[:-10]
        link = DataspecAwareLink(open_codes={"RACE": 0},
                                 records_by_dataspec={"RACE": [fx.make_ra(), broken_o1]})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)
        assert update_data.main(["--dataspec", "RACE", "--fromdate", "20260801000000",
                                 "--data-dir", str(tmp_path)]) == 0
        assert _manifest(tmp_path)["record_counts"] == {"RA": 1, "SE": 0, "UM": 0}
        assert "O1: 1（保存対象外）" in capsys.readouterr().out

    def test_verify_time_fields_checks_se_even_though_market_does_not_save_it(
            self, tmp_path, monkeypatch, capsys):
        """market の版は SE を保存しないが、--verify-time-fields の照合には SE を通す。"""
        link = DataspecAwareLink(open_codes={"RACE": 0},
                                 records_by_dataspec={"RACE": [fx.make_ra(), *_mismatched_se()]})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        exit_code = update_data.main([
            "--kind", "market", "--fromdate", "20210901000000", "--option", "4",
            "--data-dir", str(tmp_path), "--verify-time-fields"])

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "SE: 2（保存対象外）" in out
        assert "SE" not in _manifest(tmp_path)["record_counts"]


def _mismatched_se():
    """1着と2着の SE。実際のタイム差は 0.3 秒だが、記録上の margin は 0.9 秒にしてずらす。"""
    return [fx.make_se(kakutei_jyuni="01", time="1234", time_diff="-003", umaban="01"),
            fx.make_se(kakutei_jyuni="02", time="1237", time_diff="+009", umaban="02")]


class TestFromRaw:
    """--from-raw は保存した生レコードから版を作る。JV-Link は呼ばない。"""

    @pytest.fixture
    def raw_dir(self, tmp_path):
        path = tmp_path / "raw"
        with rawdump.RawDumpWriter(path, dataspec="RACE", fromdate="20210901000000",
                                   option=4) as writer:
            for raw in (fx.make_ra(), fx.make_hr(), fx.make_o1()):
                writer.write(raw)
        return path

    def test_builds_a_market_version_without_jvlink(self, tmp_path, raw_dir, monkeypatch):
        def refuse(*a, **k):
            raise AssertionError("JV-Link を呼んではいけない")
        monkeypatch.setattr(update_data.jvlink, "open_link", refuse)
        out_dir = tmp_path / "market"

        exit_code = update_data.main(["--kind", "market", "--from-raw", str(raw_dir),
                                      "--data-dir", str(out_dir)])

        assert exit_code == 0
        manifest = _manifest(out_dir)
        assert manifest["dataspecs"] == ["RACE"]
        assert (manifest["fromdate"], manifest["option"]) == ("20210901000000", 4)
        assert manifest["source"]["raw_dir"] == str(raw_dir)
        assert manifest["record_counts"]["payout_races"] == 1

    def test_verify_time_fields_runs_on_raw(self, tmp_path, capsys):
        """--from-raw でも --verify-time-fields を黙って無視せず、不一致なら exit 1。"""
        path = tmp_path / "raw_se"
        with rawdump.RawDumpWriter(path, dataspec="RACE", fromdate="20210901000000",
                                   option=4) as writer:
            writer.write(fx.make_ra())
            writer.write(fx.make_se(kakutei_jyuni="01", time="1234", time_diff="-003", umaban="01"))
            # 実際のタイム差は 0.3 秒だが、記録上の margin は 0.9 秒にしてずらす。
            writer.write(fx.make_se(kakutei_jyuni="02", time="1237", time_diff="+009", umaban="02"))

        exit_code = update_data.main(["--from-raw", str(path), "--data-dir", str(tmp_path / "race"),
                                      "--verify-time-fields"])

        assert exit_code == 1
        assert "走破タイムの照合" in capsys.readouterr().out

    def test_verify_time_fields_on_raw_reads_se_for_market(self, tmp_path, capsys):
        """market の版でも、照合するときは生レコードの SE を読む（保存はしない）。"""
        path = tmp_path / "raw_se"
        with rawdump.RawDumpWriter(path, dataspec="RACE", fromdate="20210901000000",
                                   option=4) as writer:
            for raw in (fx.make_ra(), fx.make_hr(), *_mismatched_se()):
                writer.write(raw)

        exit_code = update_data.main(["--kind", "market", "--from-raw", str(path),
                                      "--data-dir", str(tmp_path / "market"),
                                      "--verify-time-fields"])

        assert exit_code == 1
        assert "SE: 2（保存対象外）" in capsys.readouterr().out

    def test_missing_spec_file_is_refused(self, tmp_path, raw_dir):
        """manifest に載っている種別のファイルが無ければ、欠けた版を作らない。"""
        (raw_dir / "HR.bin.gz").unlink()
        out_dir = tmp_path / "market"
        with pytest.raises(SystemExit, match="HR.bin.gz"):
            update_data.main(["--kind", "market", "--from-raw", str(raw_dir),
                              "--data-dir", str(out_dir)])
        assert store.current_version(out_dir) is None

    def test_count_mismatch_with_manifest_is_refused(self, tmp_path, raw_dir):
        path = raw_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["counts"]["HR"] = 2
        path.write_text(json.dumps(manifest), encoding="utf-8")
        out_dir = tmp_path / "market"
        with pytest.raises(SystemExit, match="件数が manifest と合いません"):
            update_data.main(["--kind", "market", "--from-raw", str(raw_dir),
                              "--data-dir", str(out_dir)])
        assert store.current_version(out_dir) is None

    def test_race_kind_without_ra_is_refused(self, tmp_path):
        """DIFF だけの生レコードは RA を運ばない。レースの無い版に CURRENT を向けない。"""
        path = tmp_path / "raw_diff"
        with rawdump.RawDumpWriter(path, dataspec="DIFF", fromdate="20210901000000",
                                   option=4) as writer:
            writer.write(fx.make_um())
        out_dir = tmp_path / "race"
        with pytest.raises(SystemExit, match="RA が1件もありません"):
            update_data.main(["--from-raw", str(path), "--data-dir", str(out_dir)])
        assert store.current_version(out_dir) is None

    def test_fromdate_is_required_without_raw(self):
        with pytest.raises(SystemExit):
            update_data._parse_args(["--kind", "market"])

    def test_raw_dataspec_must_match_the_request(self, tmp_path, raw_dir):
        with pytest.raises(SystemExit):
            update_data.main(["--kind", "blood", "--from-raw", str(raw_dir),
                              "--data-dir", str(tmp_path / "blood")])

    def test_fromdate_with_raw_is_rejected(self, raw_dir):
        with pytest.raises(SystemExit):
            update_data._parse_args(["--kind", "market", "--from-raw", str(raw_dir),
                                     "--fromdate", "20210901000000"])

"""版付き保存。更新途中の CSV を予想で読まないことが要件。"""

import json
from pathlib import Path

import pytest

from keiba.ingest import store
from tests.fixtures import records as fx
from keiba.ingest import records


@pytest.fixture
def written(tmp_path):
    writer = store.VersionWriter(tmp_path, "20260322T010000Z",
                                 dataspecs=["RACE"], fromdate="20260301000000", option=1)
    writer.add("RA", records.parse_ra(fx.make_ra()))
    writer.add("SE", records.parse_se(fx.make_se()))
    writer.add("UM", records.parse_um(fx.make_um()))
    path = writer.commit()
    return tmp_path, path


class TestVersionWriter:
    def test_writes_three_csv_files_and_manifest(self, written):
        _, version_path = written
        for name in ("races.csv", "race_horses.csv", "horses.csv", "manifest.json"):
            assert (version_path / name).exists(), name

    def test_csv_is_utf8_without_bom(self, written):
        _, version_path = written
        raw = (version_path / "races.csv").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        raw.decode("utf-8")

    def test_ids_keep_leading_zeros(self, written):
        _, version_path = written
        text = (version_path / "race_horses.csv").read_text(encoding="utf-8")
        assert "0000000001" in text          # ketto_num
        assert "2026032106010511" in text    # race_id

    def test_manifest_records_counts_and_checksums(self, written):
        _, version_path = written
        manifest = store.read_manifest(version_path)
        assert manifest["version_id"] == "20260322T010000Z"
        assert manifest["schema_version"] == "1"
        assert manifest["status"] == "complete"
        assert manifest["record_counts"] == {"RA": 1, "SE": 1, "UM": 1}
        assert manifest["dataspecs"] == ["RACE"]
        names = {f["name"] for f in manifest["files"]}
        assert names == {"races.csv", "race_horses.csv", "horses.csv"}
        for entry in manifest["files"]:
            assert len(entry["sha256"]) == 64
            assert entry["rows"] == 1

    def test_checksums_match_the_written_files(self, written):
        """manifest の sha256 が実際のファイル内容と一致すること。"""
        import hashlib
        _, version_path = written
        for entry in store.read_manifest(version_path)["files"]:
            digest = hashlib.sha256((version_path / entry["name"]).read_bytes()).hexdigest()
            assert digest == entry["sha256"], entry["name"]

    def test_coverage_spans_written_races(self, written):
        _, version_path = written
        manifest = store.read_manifest(version_path)
        assert manifest["coverage"]["race_count"] == 1
        assert manifest["coverage"]["first_hasso_at"].startswith("2026-03-21")

    def test_current_points_at_committed_version(self, written):
        data_dir, _ = written
        assert store.current_version(data_dir) == "20260322T010000Z"

    def test_partial_directory_is_gone_after_commit(self, written):
        data_dir, _ = written
        assert not (data_dir / "versions" / "20260322T010000Z.partial").exists()


class TestIncompleteVersion:
    def test_partial_directory_exists_before_commit(self, tmp_path):
        writer = store.VersionWriter(tmp_path, "20260322T020000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        partial = tmp_path / "versions" / "20260322T020000Z.partial"
        assert partial.exists()
        assert not (tmp_path / "versions" / "20260322T020000Z").exists()
        assert store.current_version(tmp_path) is None
        writer.abort()
        assert not partial.exists()

    def test_read_manifest_rejects_incomplete_status(self, tmp_path):
        version_path = tmp_path / "versions" / "20260322T030000Z"
        version_path.mkdir(parents=True)
        (version_path / "manifest.json").write_text(
            json.dumps({"version_id": "x", "status": "incomplete"}), encoding="utf-8")
        with pytest.raises(store.DataVersionError, match="完了していません"):
            store.read_manifest(version_path)

    def test_abort_leaves_an_existing_current_alone(self, tmp_path):
        """書きかけを捨てても、使用中の版と CURRENT は動かさない。"""
        good = store.VersionWriter(tmp_path, "20260322T070000Z",
                                   dataspecs=["RACE"], fromdate="x", option=1)
        good.add("RA", records.parse_ra(fx.make_ra()))
        good.commit()

        doomed = store.VersionWriter(tmp_path, "20260322T080000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        doomed.add("RA", records.parse_ra(fx.make_ra()))
        doomed.abort()

        assert store.current_version(tmp_path) == "20260322T070000Z"
        assert not store.version_dir(tmp_path, "20260322T080000Z.partial").exists()
        assert not store.version_dir(tmp_path, "20260322T080000Z").exists()


class TestImmutableVersions:
    def test_refuses_to_overwrite_a_completed_version(self, tmp_path):
        """完成済みの版を上書きしない。固定した版の中身が変わってはいけない。"""
        first = store.VersionWriter(tmp_path, "20260322T040000Z",
                                    dataspecs=["RACE"], fromdate="x", option=1)
        first.add("RA", records.parse_ra(fx.make_ra()))
        first.add("SE", records.parse_se(fx.make_se()))
        first.commit()

        second = store.VersionWriter(tmp_path, "20260322T040000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        second.add("RA", records.parse_ra(fx.make_ra()))
        with pytest.raises(store.DataVersionError, match="既に完成"):
            second.commit()

    def test_original_version_survives_the_refusal(self, tmp_path):
        """拒否したあとも元の版が無傷で残り、CURRENT もそれを指したまま。"""
        first = store.VersionWriter(tmp_path, "20260322T050000Z",
                                    dataspecs=["RACE"], fromdate="x", option=1)
        first.add("RA", records.parse_ra(fx.make_ra()))
        first.add("SE", records.parse_se(fx.make_se()))
        first.add("UM", records.parse_um(fx.make_um()))
        version_path = first.commit()

        second = store.VersionWriter(tmp_path, "20260322T050000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        second.add("RA", records.parse_ra(fx.make_ra()))
        with pytest.raises(store.DataVersionError):
            second.commit()

        manifest = store.read_manifest(version_path)
        assert manifest["record_counts"] == {"RA": 1, "SE": 1, "UM": 1}
        assert store.current_version(tmp_path) == "20260322T050000Z"

    def test_removes_debris_without_a_manifest(self, tmp_path):
        """manifest の無い残骸（異常終了の跡）は消して進む。"""
        debris = store.version_dir(tmp_path, "20260322T060000Z")
        debris.mkdir(parents=True)
        (debris / "races.csv").write_text("stale", encoding="utf-8")

        writer = store.VersionWriter(tmp_path, "20260322T060000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        version_path = writer.commit()

        assert store.read_manifest(version_path)["status"] == "complete"
        assert "stale" not in (version_path / "races.csv").read_text(encoding="utf-8")


class TestCorruptManifest:
    def test_truncated_json_is_rejected_as_a_data_version_error(self, tmp_path):
        """manifest の書き込み中に落ちると JSON が途中で切れる。JSONDecodeError を漏らさない。"""
        version_path = store.version_dir(tmp_path, "20260322T090000Z")
        version_path.mkdir(parents=True)
        (version_path / "manifest.json").write_text(
            '{"version_id": "20260322T090000Z", "status": "comp',
            encoding="utf-8")
        with pytest.raises(store.DataVersionError, match="壊れています"):
            store.read_manifest(version_path)

    def test_non_dict_json_is_rejected(self, tmp_path):
        version_path = store.version_dir(tmp_path, "20260322T100000Z")
        version_path.mkdir(parents=True)
        (version_path / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(store.DataVersionError, match="形式"):
            store.read_manifest(version_path)

    def test_commit_recovers_over_a_corrupt_manifest(self, tmp_path):
        """壊れた manifest を持つディレクトリは残骸として片付けてから書ける。

        fix round 1 の except 節が本来扱うべきだった、まさにこの場合。
        """
        debris = store.version_dir(tmp_path, "20260322T110000Z")
        debris.mkdir(parents=True)
        (debris / "manifest.json").write_text('{"status": "comp', encoding="utf-8")
        (debris / "races.csv").write_text("stale", encoding="utf-8")

        writer = store.VersionWriter(tmp_path, "20260322T110000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        version_path = writer.commit()

        assert store.read_manifest(version_path)["status"] == "complete"
        assert "stale" not in (version_path / "races.csv").read_text(encoding="utf-8")
        assert store.current_version(tmp_path) == "20260322T110000Z"


class TestMultipleDataspecs:
    """RACE だけでは UM（競走馬マスタ）が取れない。1つの版に複数 dataspec をまとめる。

    fix round 2: JVOpen("RACE", ...) は UM を運ばない。UM を運ぶ DIFF も同じ版に
    含められるようにした。manifest のキーは dataspec (str) から dataspecs (list)
    へ改名してあるので、旧形式を読むコードは KeyError で気づける。
    """

    def test_manifest_records_every_dataspec_in_order(self, tmp_path):
        writer = store.VersionWriter(tmp_path, "20260322T120000Z",
                                     dataspecs=["RACE", "DIFF"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        version_path = writer.commit()
        manifest = store.read_manifest(version_path)
        assert manifest["dataspecs"] == ["RACE", "DIFF"]

    def test_manifest_has_no_singular_dataspec_key(self, tmp_path):
        writer = store.VersionWriter(tmp_path, "20260322T120100Z",
                                     dataspecs=["RACE", "DIFF"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        version_path = writer.commit()
        manifest = store.read_manifest(version_path)
        assert "dataspec" not in manifest

    def test_rejects_an_empty_dataspec_list(self, tmp_path):
        with pytest.raises(ValueError):
            store.VersionWriter(tmp_path, "20260322T120200Z",
                                dataspecs=[], fromdate="x", option=1)

    def test_rejects_duplicate_dataspecs(self, tmp_path):
        with pytest.raises(ValueError):
            store.VersionWriter(tmp_path, "20260322T120300Z",
                                dataspecs=["RACE", "RACE"], fromdate="x", option=1)

    def test_rejects_a_bare_string(self, tmp_path):
        with pytest.raises(TypeError):
            store.VersionWriter(tmp_path, "20260322T120400Z",
                                dataspecs="RACE", fromdate="x", option=1)

    def test_schema_version_is_unchanged(self, tmp_path):
        writer = store.VersionWriter(tmp_path, "20260322T120500Z",
                                     dataspecs=["RACE", "DIFF"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        version_path = writer.commit()
        assert store.read_manifest(version_path)["schema_version"] == "1"

    def test_rows_from_two_dataspecs_land_in_one_version(self, tmp_path):
        """RACE からの RA と DIFF からの UM が、同じ版の別々の CSV に収まる。"""
        writer = store.VersionWriter(tmp_path, "20260322T120600Z",
                                     dataspecs=["RACE", "DIFF"], fromdate="x", option=1)
        writer.add("RA", records.parse_ra(fx.make_ra()))
        writer.add("UM", records.parse_um(fx.make_um()))
        version_path = writer.commit()

        races_lines = (version_path / "races.csv").read_text(encoding="utf-8").strip().splitlines()
        horses_lines = (version_path / "horses.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(races_lines) - 1 == 1     # ヘッダを除いて1行
        assert len(horses_lines) - 1 == 1    # ヘッダを除いて1行

        manifest = store.read_manifest(version_path)
        assert manifest["record_counts"] == {"RA": 1, "SE": 0, "UM": 1}


class TestConstructionFailure:
    """fix round 3: __init__ が CSV を開く途中で失敗すると .partial が残っていた。

    RA -> SE -> UM の順に3つ開くうち、2つ目（race_horses.csv）で OSError が
    起きると、"writer = VersionWriter(...)" の代入自体が終わらないので、
    呼び出し側は writer.abort() を呼びようがない。__init__ 自身が、自分が
    作った .partial と自分が開いたハンドルを片付けてから例外を投げ直す。
    """

    def _flaky_open(self, monkeypatch, *, fail_on_call):
        """pathlib.Path.open を、指定回数目の呼び出しでだけ OSError にする。

        monkeypatch は Path.open 全体に効くので、失敗させたい呼び出しだけを
        数えられるよう、呼び出し回数のカウンタを持つラッパーを当てる。
        opened には実際に開けたハンドルを記録し、後始末の検証に使う。
        """
        original_open = Path.open
        calls = {"n": 0}
        opened: list = []

        def wrapper(path_self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == fail_on_call:
                raise OSError("ディスク満杯")
            handle = original_open(path_self, *args, **kwargs)
            opened.append(handle)
            return handle

        monkeypatch.setattr(Path, "open", wrapper)
        return opened

    def test_partial_directory_is_removed_when_construction_fails(self, tmp_path, monkeypatch):
        self._flaky_open(monkeypatch, fail_on_call=2)

        with pytest.raises(OSError):
            store.VersionWriter(tmp_path, "20260322T130000Z",
                                dataspecs=["RACE"], fromdate="x", option=1)

        versions = tmp_path / "versions"
        leftover = list(versions.glob("*.partial")) if versions.exists() else []
        assert leftover == []

    def test_construction_failure_closes_handles_it_opened(self, tmp_path, monkeypatch):
        opened = self._flaky_open(monkeypatch, fail_on_call=2)

        with pytest.raises(OSError):
            store.VersionWriter(tmp_path, "20260322T130100Z",
                                dataspecs=["RACE"], fromdate="x", option=1)

        assert len(opened) == 1          # RA (races.csv) だけが開けた
        assert opened[0].closed is True

    def test_validation_failure_creates_nothing_on_disk(self, tmp_path):
        with pytest.raises(ValueError):
            store.VersionWriter(tmp_path, "20260322T130200Z",
                                dataspecs=[], fromdate="x", option=1)

        versions = tmp_path / "versions"
        assert not versions.exists() or list(versions.iterdir()) == []


def test_new_version_id_is_sortable_utc_timestamp():
    version_id = store.new_version_id()
    assert len(version_id) == 16
    assert version_id[8] == "T"
    assert version_id.endswith("Z")


class TestConcurrentPartial:
    """fix5 決定9-5: 同じ .partial を2つの書き手が共有させない。"""

    def test_refuses_an_existing_partial_directory(self, tmp_path):
        """new_version_id() は秒精度なので、同一秒に2プロセスが走ると同じ
        .partial を開き、互いの CSV を truncate（"w" で開き直す）したまま
        両方が commit を目指す。2つめは作らせない。
        """
        first = store.VersionWriter(tmp_path, "20260322T090000Z",
                                    dataspecs=["RACE"], fromdate="x", option=1)
        first.add("RA", records.parse_ra(fx.make_ra()))

        with pytest.raises(store.DataVersionError, match="書きかけ"):
            store.VersionWriter(tmp_path, "20260322T090000Z",
                                 dataspecs=["RACE"], fromdate="x", option=1)

        # 1つめは無傷のまま commit できる（2つめは何も壊していない）。
        version_path = first.commit()
        assert (version_path / "races.csv").exists()

    def test_a_different_version_id_is_unaffected(self, tmp_path):
        first = store.VersionWriter(tmp_path, "20260322T100000Z",
                                    dataspecs=["RACE"], fromdate="x", option=1)
        second = store.VersionWriter(tmp_path, "20260322T110000Z",
                                     dataspecs=["RACE"], fromdate="x", option=1)
        first.abort()
        second.abort()

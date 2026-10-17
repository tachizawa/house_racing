"""中断しても続きから読めるセットアップ取得（scripts/dump_raw.py）。COM は叩かない。

JV-Link の仕様書（インターフェース仕様書 4.9.0.1 p.19）の「セットアップデータの中断・再開」に
従い、読み終えたファイル名を残し、再開時は同じ条件で JVOpen して済んだファイルを JVSkip する。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import dump_raw  # noqa: E402

from keiba.ingest import rawdump  # noqa: E402
from tests.fixtures import records as fx  # noqa: E402


class FilesLink:
    """jvd ファイルの並びを持つ偽の JV-Link。JVSkip は仕様書 p.29〜30 のとおり動く。

    files: [(ファイル名, [レコード, ...]), ...]。fail_after を渡すと、その件数のレコードを
    返した次の JVRead でエラー（-502）を返す（取得の途中で止まった状況）。
    """

    def __init__(self, files, *, fail_after=None, open_code=0):
        self._events = []
        for name, raws in files:
            for raw in raws:
                self._events.append((name, raw))
            self._events.append((None, None))     # ファイルの終わりで -1
        self._pos = 0
        self._fail_after = fail_after
        self._open_code = open_code
        self.records_returned = 0
        self.skips = 0
        self.closed = False
        self.opened_with = None

    def JVInit(self, sid):
        return 0

    def JVOpen(self, dataspec, fromdate, option, _a, _b, _c):
        self.opened_with = (dataspec, fromdate, option)
        return (self._open_code, 0, 0, "")

    def JVStatus(self):
        return 0

    def JVRead(self, buf, size, fname):
        if self._fail_after is not None and self.records_returned >= self._fail_after:
            return (-502, "", size, "")
        if self._pos >= len(self._events):
            return (0, "", size, "")
        name, raw = self._events[self._pos]
        self._pos += 1
        if raw is None:
            return (-1, "", size, "")
        self.records_returned += 1
        return (len(raw), raw.decode("cp932"), size, name)

    def JVSkip(self):
        # 現在のファイルの残りを飛ばし、次のファイルの先頭へ（ファイル末の -1 も越える）
        while self._pos < len(self._events) and self._events[self._pos][1] is not None:
            self._pos += 1
        if self._pos < len(self._events):
            self._pos += 1
        self.skips += 1

    def JVClose(self):
        self.closed = True


def _files():
    ra, se, hr = fx.make_ra(), fx.make_se(), fx.make_hr()
    return [
        ("RAVM1986.jvd", [ra, ra]),
        ("SEVM1986.jvd", [se, se, se]),
        ("HRVM1986.jvd", [hr]),
        ("RAVM1987.jvd", [ra]),
    ]


def _expected():
    return sorted(raw for _, raws in _files() for raw in raws)


def _run(tmp_path, link, monkeypatch, fromdate="19860101000000"):
    monkeypatch.setattr(dump_raw.jvlink, "open_link", lambda *a, **k: link)
    return dump_raw.main(["--dataspec", "RACE", "--fromdate", fromdate, "--option", "4",
                          "--out", str(tmp_path / "dump")])


def test_a_full_run_writes_a_complete_dump(tmp_path, monkeypatch):
    link = FilesLink(_files())
    assert _run(tmp_path, link, monkeypatch) == 0

    dump = tmp_path / "dump"
    manifest = rawdump.read_manifest(dump)
    assert manifest["counts"] == {"HR": 1, "RA": 3, "SE": 3}
    assert (manifest["dataspec"], manifest["fromdate"], manifest["option"]) == (
        "RACE", "19860101000000", 4)
    assert sorted(rawdump.iter_records(dump)) == _expected()
    assert link.closed


def test_resume_skips_finished_files_and_does_not_duplicate(tmp_path, monkeypatch):
    # 6件目（HRVM の1件）を返した次で止まる: RAVM・SEVM は済み、HRVM は書きかけ
    first = FilesLink(_files(), fail_after=6)
    assert _run(tmp_path, first, monkeypatch) == 1
    assert first.closed
    dump = tmp_path / "dump"
    with pytest.raises(rawdump.RawDumpError):
        rawdump.read_manifest(dump)     # 途中のものは版の材料にさせない
    state = json.loads((dump / "resume.json").read_text(encoding="utf-8"))
    assert state["done_files"] == ["RAVM1986.jvd", "SEVM1986.jvd"]

    second = FilesLink(_files())
    assert _run(tmp_path, second, monkeypatch) == 0
    assert second.opened_with == ("RACE", "19860101000000", 4)
    assert second.skips == 2            # 済んだ2ファイルは読み飛ばした
    assert sorted(rawdump.iter_records(dump)) == _expected()
    assert rawdump.read_manifest(dump)["counts"] == {"HR": 1, "RA": 3, "SE": 3}


def test_resume_drops_a_half_written_member(tmp_path, monkeypatch):
    """gz に書いた後・resume.json を書く前に落ちた分は、記録した大きさまで切り詰めて消す。"""
    _run(tmp_path, FilesLink(_files(), fail_after=6), monkeypatch)
    dump = tmp_path / "dump"
    se_path = dump / "SE.bin.gz"
    se_path.write_bytes(se_path.read_bytes() + b"\x1f\x8b garbage of an unfinished member")
    (dump / "HR.bin.gz").write_bytes(b"\x1f\x8b a file that resume.json never recorded")

    assert _run(tmp_path, FilesLink(_files()), monkeypatch) == 0
    assert sorted(rawdump.iter_records(dump)) == _expected()


def test_resume_refuses_different_parameters(tmp_path, monkeypatch):
    _run(tmp_path, FilesLink(_files(), fail_after=3), monkeypatch)
    with pytest.raises(SystemExit):
        _run(tmp_path, FilesLink(_files()), monkeypatch, fromdate="20000101000000")


def test_refuses_a_non_empty_directory_without_resume_state(tmp_path, monkeypatch):
    (tmp_path / "dump").mkdir()
    (tmp_path / "dump" / "SE.bin.gz").write_bytes(b"old")
    with pytest.raises(SystemExit):
        _run(tmp_path, FilesLink(_files()), monkeypatch)


def test_stops_when_jvread_gives_no_file_name(tmp_path, monkeypatch):
    """ファイル名が取れないと再開できない。何十時間も読んだ後で気づかないよう最初に止める。"""
    link = FilesLink([("", [fx.make_ra()])])
    assert _run(tmp_path, link, monkeypatch) == 1
    assert link.closed


def test_a_finished_dump_is_not_reopened(tmp_path, monkeypatch):
    _run(tmp_path, FilesLink(_files()), monkeypatch)
    link = FilesLink(_files())
    assert _run(tmp_path, link, monkeypatch) == 0
    assert link.opened_with is None     # 完了済みなら JV-Link を開かない

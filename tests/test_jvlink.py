"""JV-Link 呼び出しのラップ。COM は一切叩かない。偽の link オブジェクトを使う。"""

import sys

import pytest

from keiba.ingest import jvlink


class FakeLink:
    """JV-Link の COM オブジェクトの偽物。JVOpen/JVGets の型だけ real に合わせる。"""

    def __init__(self, gets_returns, *, open_code=0, download_count=0, statuses=None,
                 read_count=0, lastfiletime=""):
        self._gets = list(gets_returns)
        self._open_code = open_code
        self._download_count = download_count
        self._statuses = list(statuses or [download_count])
        self._read_count = read_count
        self._lastfiletime = lastfiletime
        self.closed = False
        self.status_calls = 0

    def JVInit(self, sid):
        return 0

    def JVOpen(self, dataspec, fromdate, option, _a, _b, _c):
        # pywin32 は ByRef 出力引数を戻り値のタプルで返す
        return (self._open_code, self._read_count, self._download_count, self._lastfiletime)

    def JVStatus(self):
        self.status_calls += 1
        return self._statuses[min(self.status_calls - 1, len(self._statuses) - 1)]

    def JVGets(self, buf, size, fname):
        if not self._gets:
            return (0, memoryview(b""), "")
        return self._gets.pop(0)

    def JVClose(self):
        self.closed = True


def _rec(payload: bytes):
    return (len(payload), memoryview(payload), "")


class TestReadRecords:
    def test_read_records_yields_each_record(self):
        link = FakeLink([_rec(b"RA1"), _rec(b"RA2"), _rec(b"RA3")])
        rows = list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert rows == [b"RA1", b"RA2", b"RA3"]

    def test_read_records_skips_file_switch(self):
        link = FakeLink([_rec(b"RA1"), (-1, memoryview(b""), "20260801.dat"), _rec(b"RA2")])
        rows = list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert rows == [b"RA1", b"RA2"]

    def test_read_records_raises_on_negative_code(self):
        link = FakeLink([_rec(b"RA1"), (-2, memoryview(b""), "")])
        with pytest.raises(jvlink.JVLinkError):
            list(jvlink.read_records(link, "RACE", "20260801000000", 1))

    def test_read_records_raises_when_open_fails(self):
        link = FakeLink([], open_code=4)
        with pytest.raises(jvlink.JVLinkError, match="4"):
            list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert link.closed is True

    def test_read_records_returns_nothing_when_no_data(self):
        link = FakeLink([], open_code=-1)
        rows = list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert rows == []
        assert link.closed is True

    def test_read_records_always_closes(self):
        link = FakeLink([_rec(b"RA1"), (-2, memoryview(b""), "")])
        with pytest.raises(jvlink.JVLinkError):
            list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert link.closed is True

    def test_read_records_waits_for_download(self, monkeypatch):
        monkeypatch.setattr(jvlink.time, "sleep", lambda _seconds: None)
        link = FakeLink([_rec(b"RA1")], download_count=2, statuses=[0, 1, 2])
        list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert link.status_calls >= 2

    def test_closing_the_generator_early_still_closes_the_link(self):
        """途中で読むのをやめても、close() すれば JVClose() が呼ばれる。

        read_records はジェネレータなので、for を最後まで回しきらない呼び出し方
        では JVClose() の実行がガベージコレクション任せになる。close() を
        明示すればその場で finally が走ることを確認する（docstring の注意書きの
        裏付け）。
        """
        link = FakeLink([_rec(b"RA1"), _rec(b"RA2"), _rec(b"RA3")], download_count=0)
        stream = jvlink.read_records(link, "RACE", "20260801000000", 1)
        next(stream)              # 1件だけ読む
        stream.close()            # 明示的に閉じる
        assert link.closed is True

    def test_read_records_still_yields_nothing_on_minus_one(self):
        """fix round 4: open_dataspec() を使う形に組み替えても挙動は変わらない。"""
        link = FakeLink([], open_code=-1)
        rows = list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert rows == []
        assert link.closed is True

    def test_read_records_still_raises_on_other_errors(self):
        link = FakeLink([], open_code=-111)
        with pytest.raises(jvlink.JVLinkError):
            list(jvlink.read_records(link, "RACE", "20260801000000", 1))
        assert link.closed is True


class TestOpenDataspec:
    """fix round 4: open_code を呼び出し側（update_data.py）に渡すための下請け。

    「該当データなし」(-1) と「開けなかった」を CLI が区別できるようにする。
    JVClose はしない——呼び出し側が open → 表示 → 読み出し → JVClose を
    自分で回すための部品なので、閉じる判断まで持ってはいけない。
    """

    def test_open_dataspec_returns_the_code_and_download_count(self):
        link = FakeLink([], open_code=0, download_count=3, read_count=5, lastfiletime="f")
        assert jvlink.open_dataspec(link, "RACE", "20260801000000", 1) == (0, 3)

    def test_open_dataspec_returns_minus_one_without_raising(self):
        link = FakeLink([], open_code=-1)
        assert jvlink.open_dataspec(link, "RACE", "20260801000000", 1) == (-1, 0)

    def test_open_dataspec_does_not_close(self):
        link = FakeLink([], open_code=-1)
        jvlink.open_dataspec(link, "RACE", "20260801000000", 1)
        assert link.closed is False


class TestDescribeOpenCode:
    def test_describe_open_code(self):
        assert "正常" in jvlink.describe_open_code(0)
        assert "該当データなし" in jvlink.describe_open_code(-1)
        assert "4" in jvlink.describe_open_code(4)

    def test_describe_open_code_includes_the_dataspec_when_given(self):
        message = jvlink.describe_open_code(-111, "DIFF")
        assert "DIFF" in message
        assert "-111" in message

    def test_describe_open_code_is_unchanged_without_a_dataspec(self):
        assert jvlink.describe_open_code(-111) == "JVOpen がエラーを返しました (code=-111)。"


class TestVerifyTimeFields:
    def _rows(self, *, second_margin):
        return [
            {
                "race_id": "R1", "result_ijyo_cd": "0",
                "result_kakutei_jyuni": 1, "result_time_sec": 83.4,
                "result_margin_sec": 0.0,
            },
            {
                "race_id": "R1", "result_ijyo_cd": "0",
                "result_kakutei_jyuni": 2, "result_time_sec": 83.7,
                "result_margin_sec": second_margin,
            },
        ]

    def test_verify_time_fields_matches_when_consistent(self):
        report = jvlink.verify_time_fields(self._rows(second_margin=0.3))
        assert len(report) == 1
        assert report[0]["matches"] is True

    def test_verify_time_fields_flags_mismatch(self):
        report = jvlink.verify_time_fields(self._rows(second_margin=0.9))
        assert report[0]["matches"] is False
        assert report[0]["computed"] == 0.3

    def test_verify_time_fields_skips_races_without_two_finishers(self):
        rows = [
            {
                "race_id": "R2", "result_ijyo_cd": "0",
                "result_kakutei_jyuni": 1, "result_time_sec": 83.4,
                "result_margin_sec": 0.0,
            },
        ]
        report = jvlink.verify_time_fields(rows)
        assert report == []


class TestOpenLink:
    """open_link() は non-Windows と pywin32 未導入を、原因が分かる形で失敗させる。

    どちらも実機を持ち出さずに monkeypatch で再現できる。エラーメッセージが
    唯一の価値なので、「Windows 専用」と「pywin32 が無い」を混同しないこと。
    """

    def test_open_link_rejects_non_windows(self, monkeypatch):
        monkeypatch.setattr(jvlink.sys, "platform", "linux")
        with pytest.raises(jvlink.JVLinkError, match="Windows"):
            jvlink.open_link()

    def test_open_link_reports_missing_pywin32(self, monkeypatch):
        monkeypatch.setattr(jvlink.sys, "platform", "win32")
        # win32com.client を import しようとして ImportError になる状況を、
        # 実際にアンインストールせずに再現する。値が None のキャッシュは
        # 「import しようとして失敗した」ことを import 機構自身が表す。
        monkeypatch.setitem(sys.modules, "win32com.client", None)
        with pytest.raises(jvlink.JVLinkError, match="pywin32"):
            jvlink.open_link()

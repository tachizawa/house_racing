"""scripts/ml/update_pedigree.py の後始末。COM は叩かず、偽の link を使う。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from keiba_ml.ingest import pedigree_store as ps

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("update_pedigree",
                                               ROOT / "scripts/ml/update_pedigree.py")
update_pedigree = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_pedigree)


class FakeLink:
    def __init__(self, *, open_code=-1, open_error: Exception | None = None):
        self.open_code = open_code
        self.open_error = open_error
        self.close_calls = 0

    def JVOpen(self, dataspec, fromdate, option, _a, _b, _c):
        if self.open_error is not None:
            raise self.open_error
        return (self.open_code, 0, 0, "")

    def JVStatus(self):
        return 0

    def JVGets(self, buf, size, fname):
        return (0, memoryview(b""), "")

    def JVClose(self):
        self.close_calls += 1


def _run(tmp_path, monkeypatch, link, *extra):
    monkeypatch.setattr(update_pedigree.jvlink, "open_link", lambda *a, **k: link)
    return update_pedigree.main(["--fromdate", "19860101000000", "--ml-data-dir", str(tmp_path),
                                 *extra])


def test_jvopen_exception_still_closes_the_link(tmp_path, monkeypatch):
    link = FakeLink(open_error=RuntimeError("COM が落ちた"))
    with pytest.raises(RuntimeError, match="COM"):
        _run(tmp_path, monkeypatch, link)
    assert link.close_calls == 1
    assert list((tmp_path / "pedigree" / "versions").glob("*")) == []


def test_no_records_does_not_commit_an_empty_version(tmp_path, monkeypatch, capsys):
    link = FakeLink(open_code=-1)
    assert _run(tmp_path, monkeypatch, link) == 1
    assert link.close_calls == 1
    assert ps.complete_versions(tmp_path) == []
    assert not (tmp_path / "pedigree" / "CURRENT").exists()
    assert "版を作りませんでした" in capsys.readouterr().out


def test_duplicate_dataspec_is_rejected_before_opening_jvlink(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("JV-Link を開いてはいけない")
    monkeypatch.setattr(update_pedigree.jvlink, "open_link", fail)
    with pytest.raises(ValueError, match="重複"):
        update_pedigree.main(["--fromdate", "19860101000000", "--ml-data-dir", str(tmp_path),
                              "--dataspec", "BLOD", "--dataspec", "BLOD"])

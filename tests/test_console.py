"""コンソールの文字コード設定。COM にも実ファイルにも触れない。"""

import sys

from keiba import console


class _StreamWithReconfigure:
    """reconfigure を持つ stream の偽物。実際の TextIOWrapper の代わり。"""

    def __init__(self) -> None:
        self.encoding = "cp932"

    def reconfigure(self, *, encoding=None, errors=None) -> None:
        self.encoding = encoding


class _StreamWithoutReconfigure:
    """reconfigure を持たない stream の偽物。古い/差し替えられた stdout を模す。"""

    def __init__(self) -> None:
        self.encoding = "cp932"


def test_enable_utf8_output_is_idempotent(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _StreamWithReconfigure())
    monkeypatch.setattr(sys, "stderr", _StreamWithReconfigure())

    console.enable_utf8_output()
    console.enable_utf8_output()  # 2回目も例外を投げない


def test_enable_utf8_output_sets_stdout_encoding(monkeypatch):
    fake_stdout = _StreamWithReconfigure()
    fake_stderr = _StreamWithoutReconfigure()  # reconfigure が無くても落ちないことを確認する
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    console.enable_utf8_output()

    assert "utf-8" in sys.stdout.encoding.lower()

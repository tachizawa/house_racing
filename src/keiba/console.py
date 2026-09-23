"""コンソールの文字コード設定。

``scripts/smoke_test.py`` の ``_enable_utf8_output`` と同じ処理をここに寄せた
もの。smoke_test.py 自体は単体で動く必要があるため変更しないが、update_data.py
など今後増える CLI から日本語出力の設定を共有できるようにする。
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    """Windows コンソールで日本語が化けないようにする。

    Python は既定でコンソールの文字コード (日本語 Windows なら cp932) に合わせて
    出力するが、Git Bash や Windows Terminal は UTF-8 を期待するため化ける。
    コンソール側のコードページを UTF-8 に切り替えたうえで stdout/stderr も UTF-8
    に固定する。パイプ経由のときは前者が無視されるだけで害はない。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # コンソールが無い環境ではそもそも不要
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

"""scripts/dossier.py の失敗の扱い。Jev は呼ばない。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import dossier as dossier_cli  # noqa: E402

from tests.fixtures import csv_data as fx  # noqa: E402


def test_missing_data_version_is_reported_without_traceback(tmp_path, capsys):
    rc = dossier_cli.main(["--race-id", "2026032106011101", "--data-dir", str(tmp_path / "none"),
                           "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "データ版を開けません" in capsys.readouterr().out


def test_unreadable_input_is_reported_without_traceback(tmp_path, capsys):
    data_dir = fx.write_dataset(tmp_path / "data", [fx.make_race()], [], [])
    bad = tmp_path / "race.json"
    bad.write_text("{", encoding="utf-8")
    rc = dossier_cli.main(["--input", str(bad), "--data-dir", str(data_dir),
                           "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "入力を読み込めません" in capsys.readouterr().out


def test_negative_max_recent_runs_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        dossier_cli.main(["--race-id", "2026032106011101", "--data-dir", str(tmp_path),
                          "--max-recent-runs", "-1"])
    assert exc.value.code == 2

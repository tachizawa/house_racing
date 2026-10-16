"""scripts/data_coverage.py の配線。Jev は呼ばない。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import data_coverage  # noqa: E402

from tests.fixtures import csv_data as fx  # noqa: E402

RACE_ID = "2026030106010101"   # 発走 2026-03-01 15:00、締切 2026-02-28 21:00


def _data_dir(tmp_path):
    races = [fx.make_race(race_id=RACE_ID, hasso_at="2026-03-01T15:00:00+09:00")]
    race_horses = [fx.make_race_horse(race_id=RACE_ID, umaban=f"{i:02d}", ketto_num=f"{i:010d}",
                                      result_kakutei_jyuni=str(i)) for i in range(1, 4)]
    return fx.write_dataset(tmp_path, races, race_horses, [])


def test_range_is_by_prediction_cutoff_like_the_backtests(tmp_path, capsys):
    """--from / --to はバックテストと同じく予想締切の日付。発走日では絞らない。"""
    data_dir = _data_dir(tmp_path)

    assert data_coverage.main(["--from", "20260228", "--to", "20260228",
                               "--data-dir", str(data_dir)]) == 0
    assert "対象レース: 1" in capsys.readouterr().out

    # 発走日（3/1）だけを指定すると、締切（2/28）が範囲の外なので数えない。
    assert data_coverage.main(["--from", "20260301", "--to", "20260301",
                               "--data-dir", str(data_dir)]) == 0
    assert "対象レース: 0" in capsys.readouterr().out


def test_enables_utf8_output(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data_coverage.console, "enable_utf8_output", lambda: calls.append(1))
    data_coverage.main(["--from", "20260228", "--to", "20260228",
                        "--data-dir", str(_data_dir(tmp_path))])
    assert calls == [1]

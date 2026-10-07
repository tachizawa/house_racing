"""scripts/generation_level.py の端から端まで（小さな作り物の版で）。Jev は呼ばない。"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generation_level  # noqa: E402

from tests.fixtures import csv_data as fx  # noqa: E402
from tests.fixtures.figure_runs import season_runs  # noqa: E402


def write_version(tmp_path: Path) -> Path:
    runs = season_runs(years=(2023, 2024, 2025), days_per_year=3)
    races, horses = [], []
    for race_id, g in runs.groupby("race_id"):
        r = g.iloc[0]
        races.append(dict(race_id=race_id, hasso_at=f"{r.date:%Y-%m-%d}T15:00:00+09:00", jyo_cd=r.jyo_cd,
                          track_cd=r.track_cd, track_kind=r.track_kind, kyori_m=str(r.kyori_m),
                          syubetu_cd=r.syubetu_cd, class_level=str(int(r.class_level)),
                          race_name="テスト記念" if race_id == "20250501-1600-O" else "",
                          data_kubun="7", make_date=f"{r.date:%Y%m%d}"))
        for h in g.itertuples():
            horses.append(dict(race_id=race_id, umaban=h.umaban, ketto_num=h.ketto_num, bamei=h.ketto_num,
                               barei=f"{h.barei:02d}", result_time_sec=f"{h.time_sec:.1f}",
                               result_kakutei_jyuni=str(int(h.rank)), result_ijyo_cd="0",
                               data_kubun="7", make_date=f"{h.date:%Y%m%d}"))
    return fx.write_dataset(tmp_path / "data", races, horses, [])


def test_script_writes_all_outputs(tmp_path):
    data_dir = write_version(tmp_path)
    out_dir = tmp_path / "out"
    code = generation_level.main(["--as-of", "20251004", "--data-dir", str(data_dir), "--out-dir", str(out_dir),
                                  "--bootstrap", "3", "--example-race", "20250501-1600-O"])
    assert code == 0
    for name in ("report.md", "cohorts.csv", "age_curve.csv", "figures.csv", "manifest.json"):
        assert (out_dir / name).exists(), name
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## 例: 20250501-1600-O（テスト記念）" in report
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["as_of"] == "2025-10-04" and manifest["years"] == [2024, 2025]
    assert manifest["data_dir"] == str(data_dir)
    assert manifest["coverage_start"] == "2023-05-01"          # fromdate が無い版なので最初の走の日
    cohorts = pd.read_csv(out_dir / "cohorts.csv")
    assert {"year", "age", "cohort", "top10_lo", "median_hi"} <= set(cohorts.columns)


def test_script_runs_early_in_season(tmp_path):
    """Review Focus 2: 2月の時点（2歳は0頭、他も10頭未満）でも落ちない。"""
    data_dir = write_version(tmp_path)
    code = generation_level.main(["--as-of", "20250210", "--data-dir", str(data_dir),
                                  "--out-dir", str(tmp_path / "out"), "--bootstrap", "2"])
    assert code == 0


def test_coverage_start_uses_fromdate():
    runs = pd.DataFrame({"date": [pd.Timestamp("1989-04-01"), pd.Timestamp("2022-01-01")]})
    assert generation_level.coverage_start({"fromdate": "20210901000000"}, runs) == date(2021, 9, 1)


def test_coverage_start_falls_back_to_first_run():
    runs = pd.DataFrame({"date": [pd.Timestamp("2023-05-01"), pd.Timestamp("2024-05-01")]})
    assert generation_level.coverage_start({}, runs) == date(2023, 5, 1)


def test_script_rejects_bootstrap_below_one(tmp_path, capsys):
    data_dir = write_version(tmp_path)
    with pytest.raises(SystemExit) as exc:
        generation_level.main(["--as-of", "20251004", "--data-dir", str(data_dir), "--bootstrap", "0"])
    assert exc.value.code == 2 and "--bootstrap" in capsys.readouterr().err

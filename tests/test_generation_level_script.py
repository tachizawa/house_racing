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


def test_coverage_start_uses_first_race_when_fromdate_is_older():
    # option 1 の版: fromdate は 2023-08 でも実データは 2025-09 から
    runs = pd.DataFrame({"date": [pd.Timestamp("2025-09-27")]})
    manifest = {"fromdate": "20230801000000",
                "coverage": {"first_hasso_at": "2025-09-27T09:45:00+09:00"}}
    assert generation_level.coverage_start(manifest, runs) == date(2025, 9, 27)


def test_coverage_start_ignores_correction_redeliveries():
    runs = pd.DataFrame({"date": [pd.Timestamp("1989-04-01"), pd.Timestamp("2022-01-01")]})
    manifest = {"fromdate": "20210901000000",
                "coverage": {"first_hasso_at": "1989-04-01T10:00:00+09:00"}}
    assert generation_level.coverage_start(manifest, runs) == date(2021, 9, 1)


def test_coverage_start_falls_back_to_first_run():
    runs = pd.DataFrame({"date": [pd.Timestamp("2023-05-01"), pd.Timestamp("2024-05-01")]})
    assert generation_level.coverage_start({}, runs) == date(2023, 5, 1)


def test_script_rejects_bootstrap_below_one(tmp_path, capsys):
    data_dir = write_version(tmp_path)
    with pytest.raises(SystemExit) as exc:
        generation_level.main(["--as-of", "20251004", "--data-dir", str(data_dir), "--bootstrap", "0"])
    assert exc.value.code == 2 and "--bootstrap" in capsys.readouterr().err


@pytest.mark.parametrize("ages", ["", ",", "1", "0,1", "x", "2,a"])
def test_script_rejects_invalid_ages(tmp_path, capsys, ages):
    """--ages が空・2 未満・整数でないなら、ブートストラップを回す前に止める。"""
    with pytest.raises(SystemExit) as exc:
        generation_level.main(["--as-of", "20251004", "--data-dir", str(tmp_path / "none"),
                               "--ages", ages])
    assert exc.value.code == 2 and "--ages" in capsys.readouterr().err


class _Store:
    def __init__(self, rows):
        self.race_horses = pd.DataFrame(rows)


def _horse(umaban, *, barei="03", data_kubun="7", make_date="20251005", bamei=None):
    return dict(race_id="2025100506010101", umaban=umaban, bamei=bamei or f"ウマ{umaban}",
                barei=barei, data_kubun=data_kubun, make_date=make_date)


def test_examples_ignore_deleted_and_cancelled_records():
    """作成日が新しくても、削除（0）・中止（9）の行で名前と馬齢を上書きしない。"""
    store = _Store([_horse("01", make_date="20251005"),
                    _horse("01", data_kubun="0", make_date="20251010", bamei="消えた", barei="09"),
                    _horse("02", make_date="20251005"),
                    _horse("02", data_kubun="9", make_date="20251010", bamei="中止", barei="09")])
    out = generation_level._examples(store, ["2025100506010101"])["2025100506010101"]
    assert [(h.umaban, h.bamei, h.age, h.cohort) for h in out] == [
        ("01", "ウマ01", 3, 2022), ("02", "ウマ02", 3, 2022)]


def test_examples_skip_horses_without_a_positive_age():
    """barei が空・"00" の馬は載せない（"00" を当年生まれにしない、空で止まらない）。"""
    store = _Store([_horse("01"), _horse("02", barei=""), _horse("03", barei="00")])
    out = generation_level._examples(store, ["2025100506010101"])["2025100506010101"]
    assert [h.umaban for h in out] == ["01"]

"""データ版の充足度（予想対象レースの出走馬のうち、締切前の出走記録が版にある割合）。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.evaluate import coverage
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))


def _open(tmp_path, races, race_horses, *, first_hasso_at=None):
    data_dir = fx.write_dataset(tmp_path, races, race_horses, [])
    if first_hasso_at is not None:
        manifest_path = next((tmp_path / "versions").glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["coverage"] = {"first_hasso_at": first_hasso_at}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    store = DataStore.open_current(data_dir)
    return Repository(store), ResultRepository(store)


def test_coverage_from_reads_manifest(tmp_path):
    repo, _ = _open(tmp_path, [], [], first_hasso_at="2026-08-01T09:40:00+09:00")
    assert repo.coverage_from == datetime(2026, 8, 1, 9, 40, tzinfo=JST)


def test_coverage_from_is_none_when_manifest_has_no_coverage(tmp_path):
    repo, _ = _open(tmp_path, [], [])
    assert repo.coverage_from is None


def test_measure_counts_horses_with_prior_runs(tmp_path):
    races = [
        fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T12:00:00+09:00"),
        fx.make_race(race_id="2026032106011101", hasso_at="2026-03-21T15:25:00+09:00"),
    ]
    race_horses = [
        # 過去走あり（3/1 に走っている）
        fx.make_race_horse(race_id="2026030106010101", umaban="01", ketto_num="A"),
        fx.make_race_horse(race_id="2026032106011101", umaban="01", ketto_num="A"),
        # 過去走なし
        fx.make_race_horse(race_id="2026032106011101", umaban="02", ketto_num="B",
                           result_kakutei_jyuni="2"),
    ]
    repo, results = _open(tmp_path, races, race_horses)
    target = [results.race("2026032106011101")]

    report = coverage.measure(target, repo, results)

    assert report.n_races == 1
    assert report.n_horses == 2
    assert report.n_with_prior_run == 1
    assert report.n_with_3plus_prior_runs == 0
    assert report.share_with_prior_run == 0.5


def test_measure_on_empty_races_reports_nan_share(tmp_path):
    repo, results = _open(tmp_path, [], [])
    report = coverage.measure([], repo, results)
    assert report.n_horses == 0
    assert report.share_with_prior_run != report.share_with_prior_run  # NaN


def test_measure_excludes_scratch_only_horses(tmp_path):
    """過去の出走が取消だけの馬は「過去走あり」に数えない（final review I-2。
    資料の history.status と同じ出走集合 derive.eligible_runs で数える）。"""
    races = [
        fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T12:00:00+09:00"),
        fx.make_race(race_id="2026030806010101", hasso_at="2026-03-08T12:00:00+09:00"),
        fx.make_race(race_id="2026031506010101", hasso_at="2026-03-15T12:00:00+09:00"),
        fx.make_race(race_id="2026032106011101", hasso_at="2026-03-21T15:25:00+09:00"),
    ]
    race_horses = [
        # A: 取消だけ
        fx.make_race_horse(race_id="2026030106010101", umaban="01", ketto_num="A",
                           result_ijyo_cd="1", result_kakutei_jyuni="0"),
        fx.make_race_horse(race_id="2026032106011101", umaban="01", ketto_num="A"),
        # B: 2走＋取消1回（3回出てくるが走ったのは2回）
        fx.make_race_horse(race_id="2026030106010101", umaban="02", ketto_num="B",
                           result_kakutei_jyuni="2"),
        fx.make_race_horse(race_id="2026030806010101", umaban="02", ketto_num="B"),
        fx.make_race_horse(race_id="2026031506010101", umaban="02", ketto_num="B",
                           result_ijyo_cd="1", result_kakutei_jyuni="0"),
        fx.make_race_horse(race_id="2026032106011101", umaban="02", ketto_num="B",
                           result_kakutei_jyuni="2"),
    ]
    repo, results = _open(tmp_path, races, race_horses)
    report = coverage.measure([results.race("2026032106011101")], repo, results)
    assert report.n_horses == 2
    assert report.n_with_prior_run == 1
    assert report.n_with_3plus_prior_runs == 0

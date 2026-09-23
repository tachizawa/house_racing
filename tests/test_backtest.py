"""keiba.evaluate.backtest のテスト。

過去レースを「前日夜に予想したら」で再現する部分（race_input_from_history・
prediction_cutoff_for・count_changed_entries）は RaceRow/EntryRow/ResultRow を直接
組み立てて単体で見る。run()/report() は CSV フィクスチャ（tests/fixtures/
csv_data.py）と偽クライアント（tests/test_pipeline.py と同じ形）で end-to-end
に見る。API・JV-Link には一切触れない。
"""

from __future__ import annotations

import json
import sys
import types
import typing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import typesafe_sdk as ts

from keiba import config as keiba_config
from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.evaluate import backtest as bt
from keiba.evaluate import metrics
from keiba.evaluate.splits import Splits
from keiba.models import EntryRow, RaceRow, ResultRow
from keiba.predict.integrate import Weights, WeightsIncompatibleError, feature_names
from keiba.predict.questions import AXIS_SPECS, PROMPT_VERSION
from keiba.store.records import load_month
from tests.fixtures import csv_data as fx

# scripts/ は keiba パッケージの外にあり pytest.ini の pythonpath にも入って
# いないため、tests/test_pipeline.py と同じ方法 (sys.path への追加) で
# import する。名前を backtest_cli に別名しているのは、keiba.evaluate.backtest
# （bt としてすでに import 済み）とバッティングさせないため。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backtest as backtest_cli  # noqa: E402
import fit_weights as fit_weights_cli  # noqa: E402

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


# --- 素の値でレコードを組み立てるヘルパ（race_input 系） ----


def make_race_row(*, race_id="2026032106011101", hasso_at=None, jyo_cd="06",
                   kyori_m=2000, track_cd="17", track_kind="turf", baba_cd="1",
                   class_level=4, grade_cd="", jyoken_cd_youngest="016",
                   toroku_tosu=2, syusso_tosu=2, nyusen_tosu=2,
                   first_3f_sec=35.2, race_name="テスト特別") -> RaceRow:
    return RaceRow(
        race_id=race_id, hasso_at=hasso_at or jst(2026, 3, 21, 15, 25, 0),
        jyo_cd=jyo_cd, kyori_m=kyori_m, track_cd=track_cd, track_kind=track_kind,
        baba_cd=baba_cd, class_level=class_level, grade_cd=grade_cd,
        jyoken_cd_youngest=jyoken_cd_youngest, toroku_tosu=toroku_tosu,
        syusso_tosu=syusso_tosu, nyusen_tosu=nyusen_tosu,
        first_3f_sec=first_3f_sec, race_name=race_name,
    )


def make_entry_row(*, umaban="01", wakuban="1", ketto_num="0000000001",
                    bamei="テストウマ", sex_cd="1", barei="04", futan_kg=57.0,
                    kisyu_code="00001", tozai_cd="1", chokyosi_code="00001",
                    futan_before_kg=None, kisyu_code_before="") -> EntryRow:
    return EntryRow(
        umaban=umaban, wakuban=wakuban, ketto_num=ketto_num, bamei=bamei,
        sex_cd=sex_cd, barei=barei, futan_kg=futan_kg, kisyu_code=kisyu_code,
        tozai_cd=tozai_cd, chokyosi_code=chokyosi_code,
        futan_before_kg=futan_before_kg, kisyu_code_before=kisyu_code_before,
    )


def make_result_row(*, entry=None, ijyo_cd="0", kakutei_jyuni=1, **kw) -> ResultRow:
    entry = entry or make_entry_row()
    base = dict(
        entry=entry, ijyo_cd=ijyo_cd, kakutei_jyuni=kakutei_jyuni,
        dochaku_kubun="0", dochaku_tosu=None, time_sec=120.5, margin_sec=0.0,
        jyuni_1c=3, jyuni_2c=3, jyuni_3c=2, jyuni_4c=2, last_3f_sec=34.1,
        ba_taijyu_kg=486, zogen_sa_kg=-2, odds=3.4, ninki=1,
    )
    base.update(kw)
    return ResultRow(**base)


def make_bad_weights() -> Weights:
    """versions が現在のコードと絶対に合わない Weights（fix1 修正2用）。

    ``Weights.check_compatible()`` は versions（feature_version /
    prompt_version / integration_version）だけを見るので、feature_names /
    weights / standardization の中身は空でよい（check_compatible() が
    それより先に落ちるので、そこまで到達しない）。
    """
    return Weights(
        weights_id="incompatible", feature_names=(), weights={},
        standardization={}, calibration=None,
        versions={"feature_version": "0", "prompt_version": "0",
                   "integration_version": "0"},
        lambda_=0.0, dropped_features=(), splits={},
        fitted_at="2026-09-22T02:00:00+00:00", dev_metrics={},
    )


def make_feature_mismatched_weights() -> Weights:
    """versions は現在のコードと合っているが、feature_names が
    with_layer2=False の実行と食い違う（27列 vs 17列）Weights。

    task-19-21 fix Critical 1 の再現用。check_compatible() は通るので、
    predict() は「required != available」の分岐（with_layer2 の
    取り違え検査。Task 16 fix round 2 が追加）で PredictionFailed になる。
    この経路は修正前 `from error` が無く、run() の
    isinstance(error.__cause__, WeightsIncompatibleError) が横取りできず
    jev_failed に落ちていた。
    """
    names = feature_names(with_layer2=True)  # 27列（with_layer2=False では17列）
    return Weights(
        weights_id="mismatched-features", feature_names=names,
        weights=dict.fromkeys(names, 0.0), standardization={},
        calibration=None,
        versions={"feature_version": keiba_config.FEATURE_VERSION,
                   "prompt_version": PROMPT_VERSION,
                   "integration_version": keiba_config.INTEGRATION_VERSION},
        lambda_=0.0, dropped_features=(), splits={},
        fitted_at="2026-09-22T02:00:00+00:00", dev_metrics={},
    )


def make_outcome(**kw) -> metrics.RaceOutcome:
    base = dict(race_id="race1", winners=("01",), starters=("01", "02"), excluded=())
    base.update(kw)
    return metrics.RaceOutcome(**base)


def make_row(*, race_id="race1", split="test", skipped=None,
             baseline_ranking=("01", "02"),
             baseline_probabilities=None, model_ranking=None,
             model_probabilities=None, outcome=None, record_id="rec1",
             entry_changed_count=0,
             tie_band_scale="layer1_win_prob", hasso_at=None,
             excluded_probability_mass=0.0, scratched_count=0) -> bt.BacktestRow:
    return bt.BacktestRow(
        race_id=race_id, hasso_at=hasso_at or jst(2026, 3, 21, 15, 25, 0),
        split=split, skipped=skipped, baseline_ranking=baseline_ranking,
        baseline_probabilities=baseline_probabilities or {"01": 0.6, "02": 0.4},
        model_ranking=model_ranking, model_probabilities=model_probabilities,
        outcome=outcome if outcome is not None else (make_outcome() if skipped is None else None),
        record_id=record_id, entry_changed_count=entry_changed_count,
        tie_band_scale=tie_band_scale,
        excluded_probability_mass=excluded_probability_mass,
        scratched_count=scratched_count,
    )


def make_metrics_report(**kw) -> metrics.MetricsReport:
    base = dict(
        n_races=5, n_skipped=0, top_n_hit_rate=0.6, top1_win_rate=0.4,
        brier=0.3, log_loss=1.0, expected_top_n_mean=0.55, calibration=[], n=3,
        n_boundary_tied=0, n_excluded_with_probability=0,
        n_winner_probability_clipped=0,
    )
    base.update(kw)
    return metrics.MetricsReport(**base)


# --- CSV フィクスチャ + Repository / ResultRepository -------------------------


def open_repos(tmp_path, races=None, race_horses=None, horses=None):
    data_dir = fx.write_dataset(tmp_path, races or [], race_horses or [], horses or [])
    store = DataStore.open_current(data_dir)
    return Repository(store), ResultRepository(store)


# --- 偽クライアント（tests/test_pipeline.py と同じ形） -------------------------


class FakeResponse:
    def __init__(self, choices=None, model="jev-1.13.0", input_tokens=100):
        self.choices = choices or {}
        self.model = model
        self.usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=0)


class FakeChoice:
    def __init__(self, choice, probabilities, confidence):
        self.choice = choice
        self.probabilities = probabilities
        self.confidence = confidence


class FakeClient:
    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self._handler(state, questions)


class ExplodingClient:
    """呼ばれたら即座に失敗する。スキップ経路で Jev に触れていないことの検出用。"""

    def system_one(self, state, questions, **kwargs):
        raise AssertionError("system_one が呼ばれた: この経路で Jev を呼んではいけない")


def _axis_choices() -> dict[str, FakeChoice]:
    return {
        name: FakeChoice(
            choice=next(iter(spec.criteria)),
            probabilities=dict.fromkeys(spec.criteria, 1.0 / len(spec.criteria)),
            confidence=0.7,
        )
        for name, spec in AXIS_SPECS.items()
    }


def make_handler(winner_probabilities, *, fail_bamei=None):
    """winner（第1層）と target_horse（第2層）の質問を見分けて応答する。"""
    fail_bamei = fail_bamei or set()

    def handler(state, questions):
        if "winner" in questions:
            winner_choice = max(winner_probabilities, key=winner_probabilities.get)
            winner = FakeChoice(choice=winner_choice,
                                 probabilities=winner_probabilities, confidence=0.6)
            return FakeResponse(choices={"winner": winner})
        bamei = state["target_horse"]["bamei"]
        if bamei in fail_bamei:
            raise ts.TypeSafeAPITimeoutError(10.0)
        return FakeResponse(choices=_axis_choices())

    return handler


SPLITS = Splits(train_until=date(2020, 1, 1), dev_until=date(2020, 1, 2),
                 calib_until=date(2020, 1, 3))


# ===============================================================================
# race_input_from_history
# ===============================================================================


def test_race_input_cutoff_is_before_start():
    race = make_race_row(hasso_at=jst(2026, 3, 21, 15, 25, 0))
    entries = [make_entry_row(umaban="01"), make_entry_row(umaban="02")]

    race_input = bt.race_input_from_history(race, entries, top_n=2)

    assert race_input.prediction_cutoff < race.hasso_at


def test_race_input_cutoff_is_previous_day_by_default():
    race = make_race_row(hasso_at=jst(2026, 3, 21, 15, 25, 0))
    entries = [make_entry_row(umaban="01"), make_entry_row(umaban="02")]

    race_input = bt.race_input_from_history(race, entries, top_n=2)

    assert race_input.prediction_cutoff.date() == date(2026, 3, 20)
    assert race_input.prediction_cutoff.date() != race.hasso_at.date()


def test_race_input_count_matches_the_horse_list():
    """entry_count は渡された馬の数（fix3 欠陥1）。

    toroku_tosu（登録頭数）を使うと、entries が取消馬を除いたもの
    （test_race_input_excludes_scratched_horses）である以上、取消馬が
    1頭でもいるレースで entry_count != len(horses) になり、
    validate.check() の「entry_count と馬の数が一致しない」検査に必ず
    引っかかる。ここでは toroku_tosu と len(entries) をわざと食い違う値に
    しておき、それでも entry_count が entries の数と一致することを見る。
    """
    race = make_race_row(toroku_tosu=99, syusso_tosu=99)
    entries = [make_entry_row(umaban="01"), make_entry_row(umaban="02"),
               make_entry_row(umaban="03")]

    race_input = bt.race_input_from_history(race, entries, top_n=2)

    assert race_input.entry_count == len(entries) == 3
    assert race_input.entry_count == len(race_input.horses)


def test_race_input_does_not_filter_the_entries_it_is_given():
    """``race_input_from_history`` は渡された馬をそのまま使う（絞らない）。

    「どの馬を対象にするか」は呼び出し側の判断で、``run()`` は結果行を持つ
    全馬（取消馬も含む）を渡す（fix5 決定5）。
    """
    race = make_race_row()
    results = [
        make_result_row(entry=make_entry_row(umaban="01", bamei="A"), ijyo_cd="0"),
        # ijyo_cd=1: 出走取消。本番の予想時点では知り得ないので入力に残す。
        make_result_row(entry=make_entry_row(umaban="02", bamei="B"), ijyo_cd="1",
                         kakutei_jyuni=None),
        make_result_row(entry=make_entry_row(umaban="03", bamei="C"), ijyo_cd="0",
                         kakutei_jyuni=2),
    ]

    entries = [r.entry for r in results]
    race_input = bt.race_input_from_history(race, entries, top_n=2)

    assert {h.umaban for h in race_input.horses} == {"01", "02", "03"}
    assert race_input.entry_count == 3


def test_race_input_counts_changed_entries():
    entries = [
        make_entry_row(umaban="01"),
        make_entry_row(umaban="02", futan_before_kg=58.0),
        make_entry_row(umaban="03", kisyu_code_before="00099"),
    ]

    assert bt.count_changed_entries(entries) == 2


def test_race_input_does_not_carry_the_going():
    """馬場状態は渡さない（task-19-21 fix High 2: 漏洩）。

    race（ResultRepository.races_between が確定後の RA 行から作った
    RaceRow）が baba_cd を持っていても、締切時点では未確定なので
    RaceInput には渡さない。
    """
    race = make_race_row(baba_cd="1")  # 確定後の馬場状態（良）
    entries = [make_entry_row(umaban="01"), make_entry_row(umaban="02")]

    race_input = bt.race_input_from_history(race, entries, top_n=2)

    assert race_input.baba_cd is None
    assert race_input.baba is None


def test_same_baba_is_inapplicable_in_a_backtest():
    """race_input_from_history が組んだ RaceInput では same_baba が判定不能
    になる（本番の「馬場未入力」と同じ状態。features/derive.py は今回の
    修正対象外なので、その純関数をそのまま呼んで確認する）。
    """
    from keiba.features.derive import BABA_UNKNOWN_REASON, condition_records

    race = make_race_row(baba_cd="1")
    entries = [make_entry_row(umaban="01"), make_entry_row(umaban="02")]
    race_input = bt.race_input_from_history(race, entries, top_n=2)

    records = condition_records([], race=race_input)

    assert records["same_baba"].inapplicable_reason == BABA_UNKNOWN_REASON
    assert records["same_baba"].starts is None


# ===============================================================================
# run()
# ===============================================================================


def test_run_skips_race_without_winner(tmp_path):
    """2頭とも失格で勝ち馬が確定しない。的中率の分母に入らない。"""
    race = make_race_row(race_id="2026032106011101")
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_ijyo_cd="5", result_kakutei_jyuni=""),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_ijyo_cd="5", result_kakutei_jyuni=""),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)

    rows = bt.run([race], repo, results, ExplodingClient(),
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert len(rows) == 1
    assert rows[0].skipped == "no_winner"
    assert rows[0].outcome is None


def test_run_skips_single_starter_race(tmp_path):
    """1頭は正常着順だが、もう1頭は出走取消で単走扱いになる。"""
    race = make_race_row(race_id="2026032106011101")
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_ijyo_cd="0", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_ijyo_cd="1", result_kakutei_jyuni=""),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)

    rows = bt.run([race], repo, results, ExplodingClient(),
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].skipped == "single_starter"


def test_a_race_with_a_scratch_is_not_skipped(tmp_path):
    """取消馬を含むレースが input_invalid にならず、指標の分母に入る
    （fix3 欠陥1）。

entry_count に toroku_tosu（登録頭数）を使っていたときは、取消馬が
    1頭でもいると entry_count != len(entries) になり、validate.check() の
    「entry_count と馬の数が一致しない」検査に必ず引っかかっていた（実測:
    100レース中9件が input_invalid で消えていた）。entry_count は渡した馬の
    数そのものなので、toroku_tosu と実際の頭数をわざと食い違わせても通る。
    """
    race = make_race_row(race_id="2026032106011101", toroku_tosu=3, syusso_tosu=2)
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_ijyo_cd="0", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_ijyo_cd="0", result_kakutei_jyuni="2"),
        # 出走取消（ijyo_cd=1）。予想の入力には残る（fix5 決定5）。
        fx.make_race_horse(race_id=race.race_id, umaban="03", bamei="ウマサン",
                            result_ijyo_cd="1", result_kakutei_jyuni=""),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    rows = bt.run([race], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].skipped is None
    # 取消馬（03番）も予想の入力に残る（fix5 決定5）。
    assert set(rows[0].baseline_ranking) == {"01", "02", "03"}
    assert rows[0].scratched_count == 1


def test_run_skips_jump_race(tmp_path):
    race = make_race_row(race_id="2026032106011101", track_kind="jump")

    rows = bt.run([race], None, None, ExplodingClient(),
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].skipped == "jump"


def test_run_continues_after_jev_failure(tmp_path):
    """1レースで Jev（第2層）が落ちても残りは処理される。"""
    race_a = make_race_row(race_id="2026032106011101", hasso_at=jst(2026, 3, 21, 15, 25, 0))
    race_b = make_race_row(race_id="2026032206011101", hasso_at=jst(2026, 3, 22, 15, 25, 0))
    race_horses = [
        fx.make_race_horse(race_id=race_a.race_id, umaban="01", bamei="ハイルウマ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_a.race_id, umaban="02", bamei="シッパイウマ",
                            result_kakutei_jyuni="2"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="01", bamei="セイコウイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="02", bamei="セイコウニ",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}, fail_bamei={"シッパイウマ"}))

    rows = bt.run([race_a, race_b], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=True, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0].skipped == "jev_failed"
    assert rows[1].skipped is None
    assert set(rows[1].baseline_ranking) == {"01", "02"}


def test_run_does_not_stop_on_ambiguous_match(tmp_path):
    """曖昧な馬名解決で1レースが失敗しても、残りは処理される。"""
    race_a = make_race_row(race_id="2026032106011101", hasso_at=jst(2026, 3, 21, 15, 25, 0))
    race_b = make_race_row(race_id="2026032206011101", hasso_at=jst(2026, 3, 22, 15, 25, 0))
    race_horses = [
        # 同名2頭（同馬齢・同性別）がマスタに居るため、"ドウメイウマ" は
        # 名前だけでは1頭に決まらない。
        fx.make_race_horse(race_id=race_a.race_id, umaban="01", bamei="ドウメイウマ",
                            sex_cd="1", barei="04", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_a.race_id, umaban="02", bamei="コトナルウマ",
                            sex_cd="2", barei="05", result_kakutei_jyuni="2"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="01", bamei="セイコウイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="02", bamei="セイコウニ",
                            result_kakutei_jyuni="2"),
    ]
    horses = [
        fx.make_horse(ketto_num="0000000101", bamei="ドウメイウマ", sex_cd="1",
                       birth_date="2022-04-01", make_date="20200101"),
        fx.make_horse(ketto_num="0000000102", bamei="ドウメイウマ", sex_cd="1",
                       birth_date="2022-04-01", make_date="20200101"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses, horses=horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    rows = bt.run([race_a, race_b], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0].skipped == "match_ambiguous"
    assert rows[1].skipped is None


def test_weights_incompatibility_stops_the_run(tmp_path):
    """重みの不整合は全レースで同じように起きる設定の誤りなので、スキップに
    せず run() から投げ直して全体を止める（fix1 修正2）。ExplodingClient を
    使い、1レース目でも Jev を一切呼ばずに落ちること（check_compatible() が
    第1層より前で落ちること）と、2レース目が試みられないこと（例外で
    ループそのものが止まること）の両方を確認する。
    """
    race_a = make_race_row(race_id="2026032106011101", hasso_at=jst(2026, 3, 21, 15, 25, 0))
    race_b = make_race_row(race_id="2026032206011101", hasso_at=jst(2026, 3, 22, 15, 25, 0))
    race_horses = [
        fx.make_race_horse(race_id=race_a.race_id, umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_a.race_id, umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="01", bamei="ウマサン",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="02", bamei="ウマヨン",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)

    with pytest.raises(WeightsIncompatibleError):
        bt.run([race_a, race_b], repo, results, ExplodingClient(),
               splits=SPLITS, n=2, with_layer2=False, weights=make_bad_weights(),
               save_records=False, data_dir=tmp_path)


def test_run_stops_on_feature_mismatch_not_just_version_mismatch(tmp_path):
    """check_compatible（版違い）経由だけでなく、特徴量集合の不一致
    （--no-layer2 と27列重みの取り違えなど、実際に起きやすい方）でも
    run() 全体が止まる（task-19-21 fix Critical 1）。

    前回の修正は check_compatible() 由来の WeightsIncompatibleError しか
    見ておらず、この経路（pipeline.py の「重みとこの実行の特徴量が
    一致しません」）は __cause__ が None のまま jev_failed に落ちていた。
    """
    race_a = make_race_row(race_id="2026032106011101", hasso_at=jst(2026, 3, 21, 15, 25, 0))
    race_b = make_race_row(race_id="2026032206011101", hasso_at=jst(2026, 3, 22, 15, 25, 0))
    race_horses = [
        fx.make_race_horse(race_id=race_a.race_id, umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_a.race_id, umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="01", bamei="ウマサン",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_b.race_id, umaban="02", bamei="ウマヨン",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)

    with pytest.raises(WeightsIncompatibleError):
        bt.run([race_a, race_b], repo, results, ExplodingClient(),
               splits=SPLITS, n=2, with_layer2=False,
               weights=make_feature_mismatched_weights(),
               save_records=False, data_dir=tmp_path)


def test_weights_incompatibility_is_not_a_skip_reason():
    """SkipReason に重み関連の値が無い（スキップではなく即座に止めるため）。"""
    reasons = set(typing.get_args(bt.SkipReason))
    assert reasons == {
        "no_winner", "single_starter", "jump", "jev_failed",
        "no_entries", "match_ambiguous", "input_invalid",
    }
    assert not any("weight" in reason for reason in reasons)


def test_run_assigns_split_per_race(tmp_path):
    """区分の境界を跨ぐ5レース（jump にしてスキップさせ、split 判定だけを見る）。

    判定は**予想締切**（発走の18時間前）で行う（fix5 決定4）。3件目は発走が
    3/1 15:00 でも締切は 2/28 21:00 なので dev に入る——発走日で判定して
    いたときは calib で、fit_weights（記録の prediction_cutoff で判定）と
    食い違っていた。
    """
    splits = Splits(train_until=date(2026, 1, 31), dev_until=date(2026, 2, 28),
                     calib_until=date(2026, 3, 15))
    races = [
        make_race_row(race_id="2026011506010101", hasso_at=jst(2026, 1, 15, 15, 0, 0),
                       track_kind="jump"),
        make_race_row(race_id="2026021006010101", hasso_at=jst(2026, 2, 10, 15, 0, 0),
                       track_kind="jump"),
        # 発走は 3/1（calib 側）だが締切は 2/28（dev の最終日）。
        make_race_row(race_id="2026030106010101", hasso_at=jst(2026, 3, 1, 15, 0, 0),
                       track_kind="jump"),
        make_race_row(race_id="2026031006010101", hasso_at=jst(2026, 3, 10, 15, 0, 0),
                       track_kind="jump"),
        make_race_row(race_id="2026032006010101", hasso_at=jst(2026, 3, 20, 15, 0, 0),
                       track_kind="jump"),
    ]

    rows = bt.run(races, None, None, ExplodingClient(),
                   splits=splits, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert [row.split for row in rows] == ["train", "dev", "dev", "calib", "test"]


def test_run_saves_records_when_requested(tmp_path):
    race = make_race_row(race_id="2026032106011101")
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    rows = bt.run([race], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=True, data_dir=tmp_path)

    assert rows[0].skipped is None
    month = datetime.now(timezone.utc).strftime("%Y%m")
    loaded = load_month(month, data_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].race_id == race.race_id


def test_run_saves_prompts_when_requested(tmp_path):
    """課金した実行で「何を送ったか」が後から確かめられること（report2 §6 の #7）。

    バックテストは Jev への支払いが発生する唯一の経路なので、ここで送った物を
    残せないと、同じレースにもう一度課金しないと検証できない。
    """
    race = make_race_row(race_id="2026032106011101")
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    bt.run([race], repo, results, client,
           splits=SPLITS, n=2, with_layer2=False, weights=None,
           save_records=True, save_prompts=True, data_dir=tmp_path)

    month = datetime.now(timezone.utc).strftime("%Y%m")
    record = load_month(month, data_dir=tmp_path)[0]
    payload = json.loads(
        (tmp_path / "prompts" / month / f"{record.record_id}.json").read_text(encoding="utf-8"))
    assert set(payload["layer1"]["state"]["horses"]) == {"01", "02"}
    assert "winner" in payload["layer1"]["questions"]


def test_cli_passes_the_save_prompts_flag_through_to_run(tmp_path, monkeypatch, capsys):
    """CLI の --save-prompts / --no-save-prompts が bt.run まで届くこと。

    ``bt.run`` 側の既定は False なので、配線を1本落とすと**課金する実行だけ**が
    黙って記録なしになる——一番残したい実行で一番静かに失われる。
    """
    race_horses = [
        fx.make_race_horse(race_id="2026032106011101", umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id="2026032106011101", umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
    ]
    races = [fx.make_race(race_id="2026032106011101",
                           hasso_at="2026-03-21T15:25:00+09:00")]
    data_dir = fx.write_dataset(tmp_path / "data", races, race_horses, [])
    seen: list[bool] = []

    def _fake_run(*args, **kwargs):
        seen.append(kwargs["save_prompts"])
        return []

    monkeypatch.setattr(backtest_cli.bt, "run", _fake_run)
    class _NoopClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(backtest_cli, "create_client", lambda **kw: _NoopClient())

    common = ["--data-dir", str(data_dir), "--from", "20260320", "--to", "20260321",
              "--out", str(tmp_path / "out"), "--no-layer2", "--no-save-records"]
    backtest_cli.main([*common])
    backtest_cli.main([*common, "--no-save-prompts"])

    assert seen == [True, False], capsys.readouterr().out


def test_run_does_not_save_prompts_by_default(tmp_path):
    """既定を勝手に ON にしない。保存するかどうかは呼び出し側が決める。"""
    race = make_race_row(race_id="2026032106011101")
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    bt.run([race], repo, results, client,
           splits=SPLITS, n=2, with_layer2=False, weights=None,
           save_records=False, data_dir=tmp_path)

    assert not (tmp_path / "prompts").exists()


def test_a_race_with_no_result_rows_is_no_entries(tmp_path):
    """SE 行が1件も無いレースは single_starter に化けない
    （task-19-21 fix 決定7）。

    outcome_from_results は空の results を None にするが、その None を
    そのまま _why_no_outcome に渡すと starters=[] で single_starter に
    分類されてしまい、データ欠損（結果データがそもそも無い）が「1頭立て」
    という正当なスキップに見えてしまっていた。
    """
    race = make_race_row(race_id="2026032106011101")
    repo, results = open_repos(tmp_path)  # race_horses を一切書かない

    rows = bt.run([race], repo, results, ExplodingClient(),
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].skipped == "no_entries"


def test_rows_are_written_as_they_are_produced(tmp_path):
    """on_row コールバックが行ができるたびに（run() が返る前に）呼ばれる。
    3レース目で想定外の例外が起きても、1〜2レース目ぶんはすでに
    コールバックへ渡っている（task-19-21 fix 決定6: 90レース目で例外が
    出ると Jev に払った分ごと全行が失われる欠陥の修正）。
    """
    race_a = make_race_row(race_id="A", track_kind="jump")
    race_b = make_race_row(race_id="B", track_kind="jump")
    race_c = make_race_row(race_id="C")  # 平地。results.results("C") で例外にする

    class _BrokenResultRepo:
        def results(self, race_id):
            if race_id == "C":
                raise RuntimeError("想定外の例外")
            return []

    received: list[bt.BacktestRow] = []

    with pytest.raises(RuntimeError):
        bt.run([race_a, race_b, race_c], None, _BrokenResultRepo(), None,
               splits=SPLITS, n=2, with_layer2=False, weights=None,
               save_records=False, data_dir=tmp_path, on_row=received.append)

    assert [row.race_id for row in received] == ["A", "B"]


def test_skipped_rows_carry_no_tie_information(tmp_path):
    race = make_race_row(race_id="2026032106011101", track_kind="jump")

    rows = bt.run([race], None, None, ExplodingClient(),
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].tie_band_scale == ""


# ===============================================================================
# report()
# ===============================================================================


def test_report_includes_skip_breakdown():
    rows = [
        make_row(race_id="r1", skipped="jump"),
        make_row(race_id="r2", skipped="no_winner"),
        make_row(race_id="r3", skipped=None),
    ]

    result = bt.report(rows, n=2)

    assert result["skip_breakdown"]["jump"] == 1
    assert result["skip_breakdown"]["no_winner"] == 1
    # 起きていない理由も 0 件として必ず出る。
    for reason in ("single_starter", "jev_failed", "no_entries",
                    "match_ambiguous", "input_invalid"):
        assert result["skip_breakdown"][reason] == 0


def test_report_never_hides_skips():
    rows = [make_row(race_id="r1", skipped="jev_failed")]

    result = bt.report(rows, n=2)

    assert "skip_breakdown" in result
    assert result["skip_breakdown"]["jev_failed"] == 1
    assert result["n_races"] == 0


def test_report_compares_baseline_and_model():
    outcome = make_outcome(winners=("01",), starters=("01", "02"))
    rows = [
        make_row(race_id="r1", baseline_ranking=("02", "01"),
                  baseline_probabilities={"01": 0.3, "02": 0.7},
                  model_ranking=("01", "02"),
                  model_probabilities={"01": 0.7, "02": 0.3},
                  outcome=outcome),
    ]

    result = bt.report(rows, n=1)

    # baseline は1位に外れた馬を出す。model は1位に勝ち馬を出す。
    assert result["baseline"].top1_win_rate == 0.0
    assert result["model"].top1_win_rate == 1.0
    assert result["compare"]["top1_win_rate"] == pytest.approx(1.0)
    assert "n_boundary_tied_baseline" in result["compare"]
    assert "n_boundary_tied_model" in result["compare"]


def test_model_boundary_tied_is_not_measured():
    """[baseline] は自分の確率分布で数え、[model] は None（測っていない）。

    model 尺度（model_win_prob）の run 間ばらつきは測っていないので、その
    尺度では同程度の組を作らない（fix5 決定1）。**0 にはしない**——
    「0件だった」と「測っていない」は別のことを主張する。
    """
    outcome = make_outcome(winners=("01",), starters=("01", "02", "03"))
    row = make_row(
        race_id="r1",
        baseline_ranking=("01", "02", "03"),
        # 01/02 の差 0.01 <= TIE_BAND(0.03) → 境界（n=1）は同程度。
        baseline_probabilities={"01": 0.5, "02": 0.49, "03": 0.01},
        model_ranking=("01", "02", "03"),
        # 01/02 の差 0.85 > TIE_BAND → 境界（n=1）は同程度ではない。
        model_probabilities={"01": 0.9, "02": 0.05, "03": 0.05},
        outcome=outcome,
    )

    result = bt.report([row], n=1)

    assert result["baseline"].n_boundary_tied == 1
    assert result["model"].n_boundary_tied is None


def test_boundary_tied_uses_reports_own_n_not_records_n():
    """boundary_tied は report(rows, n=...) の n に対して計算し直す
    （run() 時点の record.n には固定しない。task-19-21 fix 決定5）。
    """
    outcome = make_outcome(winners=("01",), starters=("01", "02", "03"))
    row = make_row(
        race_id="r1",
        baseline_ranking=("01", "02", "03"),
        # 01/02 の差 0.4 > band（境界タイなし）。02/03 の差 0.01 <= band
        # （境界タイあり）。n を変えるとどちらの境界を見るかが変わる。
        baseline_probabilities={"01": 0.9, "02": 0.5, "03": 0.49},
        outcome=outcome,
    )

    assert bt.report([row], n=1)["baseline"].n_boundary_tied == 0
    assert bt.report([row], n=2)["baseline"].n_boundary_tied == 1


def test_report_breaks_down_by_split():
    rows = [
        make_row(race_id="r1", split="train"),
        make_row(race_id="r2", split="test"),
        make_row(race_id="r3", split="test", skipped="jump"),
    ]

    result = bt.report(rows, n=2)

    assert result["by_split"]["train"]["baseline"].n_races == 1
    assert result["by_split"]["test"]["baseline"].n_races == 1
    assert result["by_split"]["test"]["skip_breakdown"]["jump"] == 1
    assert result["by_split"]["dev"]["baseline"].n_races == 0
    assert result["by_split"]["calib"]["baseline"].n_races == 0


def test_report_includes_n_used():
    result = bt.report([make_row(race_id="r1")], n=5)

    assert result["n"] == 5


def test_report_records_its_own_provenance():
    """report() の 'run' 節に来歴が入る（task-19-21 fix 決定3）。

    by_split の数字は間引き後のレース列と区分の境界（--limit で動く）に
    依存するので、この節が無いと再現も解釈もできない。
    """
    run_info = {
        "date_from": "20260801", "date_to": "20260921", "limit": 100,
        "races_in_range": 480, "sampled_from": "2026-08-01", "sampled_to": "2026-09-21",
        "with_layer2": True, "weights_id": "w1",
        "splits": {"train_until": "2026-08-01", "dev_until": "2026-08-15",
                   "calib_until": "2026-09-01"},
    }

    result = bt.report([make_row(race_id="r1")], n=2, run_info=run_info)

    # split_basis は report() 自身が書く欄（利用者が渡すものではない。決定4）。
    assert result["run"] == {**run_info, "split_basis": "prediction_cutoff"}


def test_report_run_provenance_defaults_to_none_without_run_info():
    """run_info を渡さない既存の呼び出しでも 'run' 節そのものは必ず出る
    （キーが無いのではなく、値が None）。
    """
    result = bt.report([make_row(race_id="r1")], n=2)

    assert set(result["run"]) == {
        "date_from", "date_to", "limit", "races_in_range",
        "sampled_from", "sampled_to", "with_layer2", "weights_id", "splits",
        "split_basis",
    }
    # 利用者の指定に由来する欄は None、split_basis はこの report が実際に
    # 使った基準なので常に値が入る（決定4）。
    assert result["run"]["split_basis"] == "prediction_cutoff"
    assert all(value is None for key, value in result["run"].items()
                if key != "split_basis")


def test_report_rejects_incomplete_run_info():
    with pytest.raises(ValueError):
        bt.report([make_row(race_id="r1")], n=2, run_info={"date_from": "20260801"})


def test_report_reports_entry_changed_count():
    rows = [
        make_row(race_id="r1", entry_changed_count=2),
        make_row(race_id="r2", entry_changed_count=1),
        make_row(race_id="r3", skipped="jump", entry_changed_count=0),
    ]

    result = bt.report(rows, n=2)

    assert result["entry_changed_count"] == 3


def test_report_with_no_usable_races():
    rows = [
        make_row(race_id="r1", skipped="no_winner"),
        make_row(race_id="r2", skipped="jump"),
    ]

    result = bt.report(rows, n=3)

    assert result["n_races"] == 0
    assert result["baseline"].n_races == 0
    assert result["model"].n_races == 0
    assert result["skip_breakdown"]["no_winner"] == 1
    assert result["skip_breakdown"]["jump"] == 1


def test_report_includes_the_tie_band_scale_breakdown():
    rows = [
        make_row(race_id="r1", tie_band_scale="layer1_win_prob"),
        make_row(race_id="r2", tie_band_scale="layer1_win_prob"),
        make_row(race_id="r3", tie_band_scale="model_win_prob"),
        make_row(race_id="r4", skipped="jump"),
    ]

    result = bt.report(rows, n=2)

    assert result["tie_band_scale_breakdown"] == {"layer1_win_prob": 2, "model_win_prob": 1}


# ===============================================================================
# scripts/backtest.py の表示 (fix2)
# ===============================================================================


def test_model_section_says_no_weights_instead_of_nan(capsys):
    """重みを指定していない実行 (model_ranking が1件も無い) の標準出力には
    "nan" が一切現れず、代わりに「重みを指定していない」旨が出る。
    """
    rows = [make_row(race_id="r1"), make_row(race_id="r2"), make_row(race_id="r3")]
    result = bt.report(rows, n=2)

    backtest_cli._print_report(result, has_model_ranking=False, warnings=[])

    captured = capsys.readouterr().out
    assert "nan" not in captured.lower()
    assert "重みを指定していないのでモデル順位はありません" in captured


def test_rows_without_a_model_ranking_are_not_counted_as_skipped():
    """「スキップ」は SkipReason が付いた行だけを指す（fix2 の表示対応、
    task-19-21 fix 決定8 で backtest.py 自体も直した）。

    重みなしの実行では3行とも skipped=None（SkipReason は1件も無い）。
    以前は keiba.evaluate.backtest._metrics_row が model_ranking の無い行を
    「skipped=True」として model 側の集計に渡していたため、
    MetricsReport.n_skipped が model 側だけ 3 になり、report.json を読む人が
    skip_breakdown（全部0件）と食い違う数字を見ていた（決定8）。今は
    そのような行を model 側の集計そのものから除外するので、
    model.n_skipped も 0（skip_breakdown と一致）になり、件数は
    n_without_model_ranking で別に見える。
    """
    rows = [make_row(race_id="r1"), make_row(race_id="r2"), make_row(race_id="r3")]
    result = bt.report(rows, n=2)

    assert sum(result["skip_breakdown"].values()) == 0
    assert backtest_cli._skip_count(result) == 0
    assert result["model"].n_skipped == 0
    assert result["n_without_model_ranking"] == 3


def test_the_tie_caveat_is_omitted_when_the_count_is_zero(capsys):
    m = make_metrics_report(n_boundary_tied=0)

    backtest_cli._print_metrics_block("baseline", m, skip_count=0)

    captured = capsys.readouterr().out
    assert "うち n位の境界が同程度の組の中だった: 0 件" in captured
    assert "※" not in captured
    # 取消・除外馬の確率についての行も同じ扱い（決定3 + fix2）。
    assert "うち取消・除外馬に確率が乗っていたレース: 0 件" in captured
    assert "その分だけ Brier は悪く出ます" not in captured


def test_the_tie_caveat_is_shown_when_the_count_is_positive(capsys):
    m = make_metrics_report(n_boundary_tied=2, n_excluded_with_probability=1)

    backtest_cli._print_metrics_block("baseline", m, skip_count=0)

    captured = capsys.readouterr().out
    assert "うち n位の境界が同程度の組の中だった: 2 件" in captured
    assert "※この2件は順位が再現しないので" in captured
    assert "うち取消・除外馬に確率が乗っていたレース: 1 件（その分だけ Brier は悪く出ます）" in captured


def test_log_loss_clip_caveat_omitted_when_zero(capsys):
    m = make_metrics_report(n_winner_probability_clipped=0)

    backtest_cli._print_metrics_block("baseline", m, skip_count=0)

    captured = capsys.readouterr().out
    assert "下限に張り付いています" not in captured


def test_log_loss_clip_caveat_shown_when_positive(capsys):
    """log_loss が clip 定数に支配されている件数を、件数が1件以上のときだけ
    注意書きとして出す（task-19-21 fix 決定9）。
    """
    m = make_metrics_report(n_winner_probability_clipped=44)

    backtest_cli._print_metrics_block("baseline", m, skip_count=0)

    captured = capsys.readouterr().out
    assert "うち 44 件は勝ち馬の確率が 0 で下限に張り付いています" in captured
    assert "絶対値を他と比べないこと" in captured


def test_test_split_is_shown_first_and_labeled_held_out(capsys):
    """test 区分を最初に、held-out と明記して表示する
    （task-19-21 fix 決定4）。
    """
    rows = [
        make_row(race_id="r1", split="train"),
        make_row(race_id="r2", split="test"),
    ]
    result = bt.report(rows, n=2)

    backtest_cli._print_report(result, has_model_ranking=False, warnings=[])

    captured = capsys.readouterr().out
    assert "held-out" in captured
    test_pos = captured.index("test 区分")
    overall_pos = captured.index("全行平均")
    assert test_pos < overall_pos


def test_report_json_has_no_nan_literals():
    """report.json に NaN リテラルを書かない（task-19-21 fix 決定3）。

    n_races == 0 のときの各指標は NaN になる（_mean([]) の既定）。
    そのまま json.dumps すると非標準の NaN リテラルになり、jq・JS の
    JSON.parse が拒否する。_to_jsonable は None（JSON の null）に変える
    ――0.0 にはしない（「測っていない」と「0だった」は別物）。
    """
    rows = [make_row(race_id="r1", skipped="jump")]  # n_races == 0
    result = bt.report(rows, n=3)

    jsonable = backtest_cli._to_jsonable(result)
    text = json.dumps(jsonable, ensure_ascii=False)

    assert "NaN" not in text
    reloaded = json.loads(text)  # 標準 JSON として読める
    assert reloaded["baseline"]["top_n_hit_rate"] is None
    assert reloaded["baseline"]["brier"] is None


def test_report_sums_scratched_count():
    """report()['scratched_count'] は行ごとの BacktestRow.scratched_count の
    合計（fix2/fix3: 取消・除外で対象から外した頭数。本番の予想時点では
    誰が取消になるか分からないので、この頭数ぶんバックテストの方が出走
    頭数が少なく、的中率が少し楽に出る差を見えるようにする）。スキップ行の
    scratched_count は0（数えていない）なので、含めても合計は変わらない。
    """
    rows = [
        make_row(race_id="r1", scratched_count=2),
        make_row(race_id="r2", scratched_count=1),
        make_row(race_id="r3", skipped="jump", scratched_count=0),
    ]

    result = bt.report(rows, n=2)

    assert result["scratched_count"] == 3


# ===============================================================================
# scripts/backtest.py: --limit の均等間引きと空区分の警告 (fix3)
# ===============================================================================


def test_limit_samples_across_the_whole_range():
    """480件相当から100件選ぶとき、選ばれた発走日の最小・最大が元の
    最小・最大に一致する（先頭から取ると対象期間の先頭に固まってしまう。
    実測: 7週間指定で --limit 100 が最初の3日分に収まっていた）。
    """
    races = [make_race_row(race_id=f"r{i}", hasso_at=jst(2026, 1, 1) + timedelta(days=i))
              for i in range(480)]

    sampled = backtest_cli._sample_evenly(races, 100)

    assert len(sampled) == 100
    assert sampled[0].hasso_at == races[0].hasso_at
    assert sampled[-1].hasso_at == races[-1].hasso_at


def test_limit_keeps_chronological_order():
    races = [make_race_row(race_id=f"r{i}", hasso_at=jst(2026, 1, 1) + timedelta(hours=i))
              for i in range(50)]

    sampled = backtest_cli._sample_evenly(races, 10)

    hasso_ats = [race.hasso_at for race in sampled]
    assert hasso_ats == sorted(hasso_ats)
    assert len(set(hasso_ats)) == len(hasso_ats)  # 重複して選んでいない


def test_limit_larger_than_the_pool_takes_everything():
    races = [make_race_row(race_id=f"r{i}", hasso_at=jst(2026, 1, 1) + timedelta(days=i))
              for i in range(5)]

    assert backtest_cli._sample_evenly(races, 100) == races


def test_the_chosen_range_is_printed(capsys):
    races_in_range = [make_race_row(race_id=f"r{i}", hasso_at=jst(2026, 1, 1) + timedelta(days=i))
                        for i in range(10)]
    sampled = backtest_cli._sample_evenly(races_in_range, 3)

    backtest_cli._print_sample_summary(races_in_range, sampled)

    captured = capsys.readouterr().out
    assert str(sampled[0].hasso_at.date()) in captured
    assert str(sampled[-1].hasso_at.date()) in captured
    assert "3" in captured


def test_an_empty_split_is_warned_about(capsys):
    """dev が0件のとき、標準出力と（report.json に入る）warnings リストの
    両方に警告が出る。
    """
    rows = [make_row(race_id="r1", split="train"), make_row(race_id="r2", split="calib")]
    result = bt.report(rows, n=2)

    warnings = backtest_cli._split_warnings(result)

    assert any("dev 区分が 0 件" in w for w in warnings)
    assert any("test 区分が 0 件" in w for w in warnings)
    assert not any("train 区分が 0 件" in w for w in warnings)
    assert not any("calib 区分が 0 件" in w for w in warnings)

    backtest_cli._print_report(result, has_model_ranking=False, warnings=warnings)
    captured = capsys.readouterr().out
    assert "dev 区分が 0 件" in captured


def test_no_warning_when_every_split_has_races():
    rows = [
        make_row(race_id="r1", split="train"),
        make_row(race_id="r2", split="dev"),
        make_row(race_id="r3", split="calib"),
        make_row(race_id="r4", split="test"),
    ]
    result = bt.report(rows, n=2)

    assert backtest_cli._split_warnings(result) == []


# ===============================================================================
# fix5 決定3: 結果行だけのレースでも class_level が取れる
# ===============================================================================


def test_class_level_change_is_present_in_a_backtest(tmp_path):
    """結果行だけのレース（実データの 480 レースがこれ）でも
    recent.class_level_change が None にならない（要）。

    実データ確認: races.csv の data_kubun は {"2": 84, "7": 480} で、結果行を
    持つ480レースのうち出馬表行を併せ持つものは0件。Repository.race() は
    発走が as_of 以降なら結果行を使わないので 466/466 で None を返し、
    class_level_change が全馬 None → 0.0 埋め + missing=1.0 になっていた
    （27列のうち2列が常に無効で、本番とは特徴分布が系統的に違う）。
    """
    target_id = "2026032106011101"
    past_id = "2026011006010111"
    target = make_race_row(race_id=target_id, hasso_at=jst(2026, 3, 21, 15, 25, 0))
    races = [
        # 対象レース: 結果行（data_kubun=7）だけ。出馬表行は無い。
        fx.make_race(race_id=target_id, hasso_at="2026-03-21T15:25:00+09:00",
                      class_level="5", data_kubun="7", make_date="20260322"),
        # 同じ2頭の前走。1つ下のクラス。
        fx.make_race(race_id=past_id, hasso_at="2026-01-10T15:25:00+09:00",
                      class_level="3", data_kubun="7", make_date="20260111"),
    ]
    race_horses = [
        fx.make_race_horse(race_id=target_id, umaban="01", ketto_num="0000000001",
                            bamei="イチバンウマ", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=target_id, umaban="02", ketto_num="0000000002",
                            bamei="ニバンウマ", result_kakutei_jyuni="2"),
        fx.make_race_horse(race_id=past_id, umaban="01", ketto_num="0000000001",
                            bamei="イチバンウマ", result_kakutei_jyuni="1",
                            make_date="20260111"),
        fx.make_race_horse(race_id=past_id, umaban="02", ketto_num="0000000002",
                            bamei="ニバンウマ", result_kakutei_jyuni="2",
                            make_date="20260111"),
    ]
    repo, results = open_repos(tmp_path, races=races, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    rows = bt.run([target], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=True, data_dir=tmp_path)

    assert rows[0].skipped is None
    [record] = load_month(datetime.now(timezone.utc).strftime("%Y%m"), data_dir=tmp_path)
    for horse in record.horses:
        assert horse.features["class_level_change_missing"] == 0.0, horse.umaban
        assert horse.features["class_level_change"] == 2.0, horse.umaban


# ===============================================================================
# fix5 決定4: 区分の基準時刻は prediction_cutoff
# ===============================================================================


BOUNDARY_SPLITS = Splits(train_until=date(2026, 1, 31), dev_until=date(2026, 2, 20),
                          calib_until=date(2026, 3, 5))


def test_backtest_and_fit_weights_agree_on_the_split(tmp_path):
    """同じ境界日を渡したとき、同じ race_id が同じ区分に入る（要）。

    backtest は hasso_at.date()、fit_weights は prediction_cutoff.date() で
    判定していたため、各境界の翌開催日が丸ごと1区分ずれていた。backtest が
    「held-out。学習に使っていません」と銘打ったレースの一部が fit_weights
    では calib に使われる（レポートの主張が偽になる）。
    """
    race_id = "2026020106010111"
    # 発走は 2/1 10:00。予想締切は前日 1/31（train の最終日）。
    race = make_race_row(race_id=race_id, hasso_at=jst(2026, 2, 1, 10, 0, 0))
    races = [fx.make_race(race_id=race_id, hasso_at="2026-02-01T10:00:00+09:00",
                           data_kubun="7", make_date="20260202")]
    race_horses = [
        fx.make_race_horse(race_id=race_id, umaban="01", ketto_num="0000000001",
                            bamei="イチバンウマ", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_id, umaban="02", ketto_num="0000000002",
                            bamei="ニバンウマ", result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, races=races, race_horses=race_horses)
    client = FakeClient(make_handler({"01": 0.6, "02": 0.4}))

    rows = bt.run([race], repo, results, client,
                   splits=BOUNDARY_SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=True, data_dir=tmp_path)

    [record] = load_month(datetime.now(timezone.utc).strftime("%Y%m"), data_dir=tmp_path)
    fit_split = fit_weights_cli._assign_split(
        record,
        train_until=BOUNDARY_SPLITS.train_until,
        dev_until=BOUNDARY_SPLITS.dev_until,
        calib_until=BOUNDARY_SPLITS.calib_until,
    )

    assert rows[0].split == fit_split
    assert rows[0].split == "train"


def test_report_states_the_split_basis():
    result = bt.report([make_row()], n=2)
    assert result["run"]["split_basis"] == "prediction_cutoff"


# ===============================================================================
# fix5 決定5: 取消馬も入力に残す
# ===============================================================================


def _race_with_one_scratched_horse(tmp_path):
    race_id = "2026032106011101"
    race = make_race_row(race_id=race_id)
    race_horses = [
        fx.make_race_horse(race_id=race_id, umaban="01", ketto_num="0000000001",
                            bamei="イチバンウマ", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_id, umaban="02", ketto_num="0000000002",
                            bamei="ニバンウマ", result_kakutei_jyuni="2"),
        # 出走取消。本番の予想時点（前日夜）では誰が取消になるか分からない。
        fx.make_race_horse(race_id=race_id, umaban="03", ketto_num="0000000003",
                            bamei="サンバンウマ", result_ijyo_cd="1",
                            result_kakutei_jyuni=""),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)
    return race, repo, results


def test_race_input_keeps_scratched_horses(tmp_path):
    """取消馬を予想の入力から外さない。

    外すと entry_count が「結果から算出される出走頭数（SyussoTosu）」と
    定義上同一になり、仕様 §4.1.4 が「結果情報。使わない」とした数を
    バックテストだけが見ることになる。しかも外された馬は「絶対に勝てない馬」
    なので的中率が系統的に楽に出る。
    """
    race, repo, results = _race_with_one_scratched_horse(tmp_path)
    client = FakeClient(make_handler({"01": 0.5, "02": 0.3, "03": 0.2}))

    rows = bt.run([race], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=True, data_dir=tmp_path)

    assert rows[0].skipped is None
    [record] = load_month(datetime.now(timezone.utc).strftime("%Y%m"), data_dir=tmp_path)
    assert {h["umaban"] for h in record.race_input["horses"]} == {"01", "02", "03"}
    assert record.race_input["entry_count"] == 3
    # 採点は走った馬だけ（outcome.starters は取消馬を含まない）。
    assert rows[0].outcome.starters == ("01", "02")
    assert rows[0].outcome.excluded == ("03",)


def test_excluded_probability_mass_is_positive_when_a_scratched_horse_had_probability(tmp_path):
    """取消馬に乗った確率が実際に測れる（要）。

    取消馬を入力から外していたので、そのバイアスを測るために作った
    excluded_probability_mass と n_excluded_with_probability は構造的に
    常に 0.0 / 0 だった（2つのタスクの決定が互いを打ち消していた）。
    """
    race, repo, results = _race_with_one_scratched_horse(tmp_path)
    client = FakeClient(make_handler({"01": 0.5, "02": 0.3, "03": 0.2}))

    rows = bt.run([race], repo, results, client,
                   splits=SPLITS, n=2, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].excluded_probability_mass == pytest.approx(0.2)
    result = bt.report(rows, n=2)
    assert result["baseline"].n_excluded_with_probability == 1


# ===============================================================================
# fix5 決定10: top_n が出走頭数を超えるレースの分類
# ===============================================================================


def test_a_race_with_fewer_horses_than_n_is_single_starter(tmp_path):
    """n 頭に足りないレースは input_invalid ではなく single_starter に数える。

    入力の書式が壊れているわけではなく、「n 頭を選べるだけの出走馬がいない」
    だけなので、1頭立てと同じ理由で採点が成立しない。input_invalid に数えると
    「利用者入力の不備」が起きたように読める。
    """
    race = make_race_row(race_id="2026032106011101")
    race_horses = [
        fx.make_race_horse(race_id=race.race_id, umaban="01", bamei="ウマイチ",
                            result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race.race_id, umaban="02", bamei="ウマニ",
                            result_kakutei_jyuni="2"),
    ]
    repo, results = open_repos(tmp_path, race_horses=race_horses)

    rows = bt.run([race], repo, results, ExplodingClient(),
                   splits=SPLITS, n=3, with_layer2=False, weights=None,
                   save_records=False, data_dir=tmp_path)

    assert rows[0].skipped == "single_starter"
    result = bt.report(rows, n=3)
    assert result["skip_breakdown"]["input_invalid"] == 0
    assert result["skip_breakdown"]["single_starter"] == 1


def test_the_tie_line_says_not_measured_for_the_model(capsys):
    """n_boundary_tied が None のとき「測っていません」と出す（0 件と書かない）。"""
    m = make_metrics_report(n_boundary_tied=None)

    backtest_cli._print_metrics_block("model", m, skip_count=0)

    captured = capsys.readouterr().out
    assert "測っていません" in captured
    # 「0 件」と書かない（測っていないことと0件は別）。
    assert "同程度の組の中だった: 0 件" not in captured
    assert "※" not in captured


def test_the_scratched_line_says_the_horses_stay_in_the_input(capsys):
    """取消馬の頭数の注記が「入力に残している」に変わっている（決定5）。"""
    rows = [make_row(race_id="r1", scratched_count=2)]
    result = bt.report(rows, n=2)

    backtest_cli._print_report(result, has_model_ranking=False, warnings=[])

    captured = capsys.readouterr().out
    assert "入力に残しています" in captured
    assert "対象から外した頭数" not in captured


# ===============================================================================
# fix5 決定4: --from / --to も予想締切で絞る
# ===============================================================================


def test_cutoff_window_translates_to_a_hasso_window():
    """予想締切の範囲 → 発走時刻の範囲（締切 = 発走 - 18時間）。"""
    start = jst(2026, 8, 1, 0, 0, 0)
    end = jst(2026, 8, 31, 23, 59, 59)

    hasso_start, hasso_end = bt.hasso_window_for_cutoff_window(start, end)

    assert hasso_start == start + timedelta(hours=bt.PREDICTION_CUTOFF_HOURS)
    assert hasso_end == end + timedelta(hours=bt.PREDICTION_CUTOFF_HOURS)
    # 両端の発走レースは、締切がちょうど指定範囲の端に来る。
    assert bt.prediction_cutoff_for(make_race_row(hasso_at=hasso_start)) == start
    assert bt.prediction_cutoff_for(make_race_row(hasso_at=hasso_end)) == end


def _write_one_race_dataset(tmp_path, *, race_id, hasso_at):
    races = [fx.make_race(race_id=race_id, hasso_at=hasso_at,
                           data_kubun="7", make_date="20260202")]
    race_horses = [
        fx.make_race_horse(race_id=race_id, umaban="01", ketto_num="0000000001",
                            bamei="イチバンウマ", result_kakutei_jyuni="1"),
        fx.make_race_horse(race_id=race_id, umaban="02", ketto_num="0000000002",
                            bamei="ニバンウマ", result_kakutei_jyuni="2"),
    ]
    return fx.write_dataset(tmp_path / "data", races, race_horses, [])


class _CliClient:
    """with 文と system_one だけを持つ偽クライアント（CLI 経路用）。"""

    def __init__(self):
        self._handler = make_handler({"01": 0.6, "02": 0.4})

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def system_one(self, state, questions, **kwargs):
        return self._handler(state, questions)


def test_cli_selects_races_by_prediction_cutoff(tmp_path, monkeypatch, capsys):
    """--from / --to は予想締切で絞る（発走日ではない）。

    発走 2026-02-01 10:00 のレースの締切は前日 1/31。--from/--to に 1/31 を
    渡したときに対象になり、発走日（2/1）を渡したときは対象にならない。
    """
    data_dir = _write_one_race_dataset(
        tmp_path, race_id="2026020106010111", hasso_at="2026-02-01T10:00:00+09:00")
    monkeypatch.setattr(backtest_cli, "create_client", lambda **kwargs: _CliClient())

    exit_code = backtest_cli.main([
        "--data-dir", str(data_dir), "--from", "20260131", "--to", "20260131",
        "--n", "2", "--no-layer2", "--no-save-records",
        "--out", str(tmp_path / "out1"),
    ])
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "対象 1 件" in captured.out

    exit_code = backtest_cli.main([
        "--data-dir", str(data_dir), "--from", "20260201", "--to", "20260201",
        "--n", "2", "--no-layer2", "--no-save-records",
        "--out", str(tmp_path / "out2"),
    ])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "対象期間にレースがありません" in captured.out


def test_cli_report_states_the_split_basis(tmp_path, monkeypatch, capsys):
    data_dir = _write_one_race_dataset(
        tmp_path, race_id="2026020106010111", hasso_at="2026-02-01T10:00:00+09:00")
    monkeypatch.setattr(backtest_cli, "create_client", lambda **kwargs: _CliClient())
    out_dir = tmp_path / "out"

    exit_code = backtest_cli.main([
        "--data-dir", str(data_dir), "--from", "20260131", "--to", "20260131",
        "--n", "2", "--no-layer2", "--no-save-records", "--out", str(out_dir),
    ])
    assert exit_code == 0, capsys.readouterr().out

    [report_path] = list(out_dir.glob("*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run"]["split_basis"] == "prediction_cutoff"


def test_the_public_helper_that_judged_ties_on_the_model_scale_is_gone():
    """``boundary_tied`` と ``_display_rank`` を消したこと（fix6 ⑤-2）。

    単に未使用だったからではない。``_display_rank`` は rank_model を優先する
    ので、あの関数は model の順位に対して呼べてしまい、返り値が ``bool`` な
    ので「測っていない」を表現できず ``False`` と断言する——§1 が禁じた規則を
    体現した公開ヘルパだった。指標側の判定は
    ``_probability_boundary_tied``（baseline 尺度のみ、model は ``None``）に
    1つだけある。
    """
    assert not hasattr(bt, "boundary_tied")
    assert not hasattr(bt, "_display_rank")
    assert "boundary_tied" not in bt.__all__

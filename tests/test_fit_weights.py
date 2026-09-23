"""scripts/fit_weights.py のテスト。

Jev には一切触れない（このスクリプトの契約そのもの）。ここで見るのは
(1) レース選定の除外理由（非有限な特徴量・重複レースの畳み込みを含む）が
正しく数えられ、範囲外に落ちた件数もどこかに必ず出ること、
(2) 「使える記録に絞ってから畳み、畳んでから split する」という順序が
守られていて、同じレースが train と dev の両方に入る漏洩が無いこと
(fix round 2 Critical 3)、(3) 学習に使った記録の versions（data_version・
feature_version・prompt_version・integration_version・model）が実際に使った
値のまま記録され、check_compatible が正しく効くこと (fix round 2 Critical 1)、
(4) predict() が絶対に読めない特徴量集合の重みは書き出さないこと
(fix round 2 Critical 2)、(5) CLI が実際に data/weights/<weights_id>.json を
書き出すところまでの end-to-end 配線。

scripts/ は keiba パッケージの外にあるので、tests/test_update_data.py と
同じ方法 (sys.path への追加) で import する。
"""

from __future__ import annotations

import dataclasses
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fit_weights  # noqa: E402

from keiba.learn import conditional_logit as cl  # noqa: E402
from keiba.predict import integrate as ig  # noqa: E402
from keiba.predict.integrate import Weights  # noqa: E402
from keiba.predict.questions import PROMPT_VERSION  # noqa: E402
from keiba.store import records as records_store  # noqa: E402
from tests.fixtures import csv_data as fx  # noqa: E402


# --- 組み立てヘルパー --------------------------------------------------------


def _features(win_prob: float) -> dict[str, float]:
    """27列の生の特徴量。logit_layer1 だけ win_prob から作り、残りは 0.0。"""
    return {
        name: (ig.logit(win_prob) if name == "logit_layer1" else 0.0)
        for name in ig.feature_names(with_layer2=True)
    }


def _horse(umaban: str, win_prob: float) -> records_store.HorsePrediction:
    return records_store.HorsePrediction(
        umaban=umaban, bamei=f"ウマ{umaban}", rank_baseline=1, rank_model=None,
        tie_group=1, layer1_win_prob=win_prob, model_score=None, model_win_prob=None,
        calibrated_win_prob=None, score_breakdown={}, features=_features(win_prob),
        layer2={}, positives=(), negatives=(), insufficient=(), notes=(),
    )


def _horse_with_corner(umaban: str, win_prob: float, corner_pos_mean: float,
                        ) -> records_store.HorsePrediction:
    """corner_pos_mean (Python側特徴の1つ) だけ差し替えた馬。

    削減構成 (without_python_features) は corner_pos_mean を見ないので、
    train と dev でこの列と結果の相関関係を逆転させれば、全列見る構成
    (with_python_features) だけを狂わせて dev で不利にできる
    (test_reduced_configuration_win_is_reported_loudly 用)。
    """
    features = _features(win_prob)
    features["corner_pos_mean"] = corner_pos_mean
    return dataclasses.replace(_horse(umaban, win_prob), features=features)


def _record(
    *, race_id: str | None, horses: tuple[records_store.HorsePrediction, ...],
    prediction_cutoff: datetime, predicted_at: datetime | None = None,
    status: str = "SUCCEEDED", model: str = "jev-1.13.0",
    data_version: str = "20260101T000000Z",
    feature_version: str | None = None, prompt_version: str | None = None,
    integration_version: str | None = None,
) -> records_store.PredictionRecord:
    versions = dict(records_store.current_versions(
        data_version=data_version, model=model, weights_id=None))
    if feature_version is not None:
        versions["feature_version"] = feature_version
    if prompt_version is not None:
        versions["prompt_version"] = prompt_version
    if integration_version is not None:
        versions["integration_version"] = integration_version
    versions["data_fetched_at"] = "2026-01-01T00:00:00+09:00"
    return records_store.PredictionRecord(
        record_id=uuid.uuid4().hex, race_id=race_id,
        predicted_at=predicted_at or datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc),
        prediction_cutoff=prediction_cutoff,
        n=1, versions=versions, horses=horses,
        layer1={"choice": horses[0].umaban, "confidence": 0.5},
        data_quality={"notes": [], "horse_notes": {}},
        usage={"input_tokens": 100},
        status=status,
        race_input={"race_id": race_id},
    )


class _StubResultRepo:
    """result_repo.winners(race_id) だけを持つ最小の偽物。"""

    def __init__(self, winners_by_race: dict[str, list[str]]):
        self._winners_by_race = winners_by_race

    def winners(self, race_id):
        return self._winners_by_race.get(race_id, [])


def _cutoff(day: int) -> datetime:
    return datetime(2025, 1, day, 20, 0, 0, tzinfo=timezone.utc)


# --- _filter_usable: 除外理由 --------------------------------------------------


def test_filter_usable_excludes_non_succeeded_status():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1),
                      status="PARTIAL")
    repo = _StubResultRepo({"r1": ["01"]})

    usable, counts, _mismatch = fit_weights._filter_usable([record], set(ig.feature_names()), repo)

    assert usable == []
    assert counts["status_not_succeeded"] == 1
    assert counts["no_winner"] == 0


def test_filter_usable_excludes_when_no_winner_recorded():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1))
    repo = _StubResultRepo({})  # winners(race_id) は空

    usable, counts, _mismatch = fit_weights._filter_usable([record], set(ig.feature_names()), repo)

    assert usable == []
    assert counts["no_winner"] == 1


def test_filter_usable_excludes_when_winner_not_among_horses():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1))
    repo = _StubResultRepo({"r1": ["99"]})  # 記録の馬に99番がいない

    usable, counts, _mismatch = fit_weights._filter_usable([record], set(ig.feature_names()), repo)

    assert usable == []
    assert counts["winner_not_in_horses"] == 1


def test_filter_usable_excludes_on_feature_key_mismatch():
    horse_bad = dataclasses.replace(_horse("01", 0.7), features={"only_one_key": 1.0})
    horses = (horse_bad, _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1))
    repo = _StubResultRepo({"r1": ["01"]})

    usable, counts, _mismatch = fit_weights._filter_usable([record], set(ig.feature_names()), repo)

    assert usable == []
    assert counts["feature_mismatch"] == 1


def test_filter_usable_excludes_non_finite_features():
    """NaN/inf が1つ混じったレースは学習集合から除外し、件数を出す
    (fix round 2 Minor 1)。除外しないと conditional_logit.fit の検証が
    未捕捉の ValueError で落ち、並べ替え後の行番号しか分からずどのレース・
    どの馬かが追えない。
    """
    horse_bad = dataclasses.replace(
        _horse("01", 0.7), features={**_features(0.7), "corner_pos_mean": float("nan")})
    horses = (horse_bad, _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1))
    repo = _StubResultRepo({"r1": ["01"]})

    usable, counts, _mismatch = fit_weights._filter_usable([record], set(ig.feature_names()), repo)

    assert usable == []
    assert counts["non_finite_feature"] == 1


def test_filter_usable_keeps_valid_race():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1))
    repo = _StubResultRepo({"r1": ["01"]})

    usable, counts, _mismatch = fit_weights._filter_usable([record], set(ig.feature_names()), repo)

    assert len(usable) == 1
    assert usable[0].winners == ("01",)
    assert sum(counts.values()) == 0


def test_filter_usable_reports_every_reason_even_when_zero():
    """理由ごとの件数を毎回すべて出す (0件の理由も省かない)。"""
    _usable, counts, _mismatch = fit_weights._filter_usable([], set(ig.feature_names()),
                                                   _StubResultRepo({}))
    assert set(counts.keys()) == set(fit_weights._EXCLUSION_REASONS)
    assert all(v == 0 for v in counts.values())


# --- _filter_in_range (fix round 2 Important 1) ---------------------------------


def test_filter_in_range_counts_out_of_range_records():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    in_range = _record(race_id="in", horses=horses, prediction_cutoff=_cutoff(5))
    too_early = _record(race_id="early", horses=horses,
                         prediction_cutoff=datetime(2024, 1, 1, tzinfo=timezone.utc))
    too_late = _record(race_id="late", horses=horses,
                        prediction_cutoff=datetime(2026, 1, 1, tzinfo=timezone.utc))

    kept, out_of_range = fit_weights._filter_in_range(
        [in_range, too_early, too_late],
        date_from=date(2025, 1, 1), date_to=date(2025, 12, 31))

    assert [r.race_id for r in kept] == ["in"]
    assert out_of_range == 2


# --- _collapse_duplicate_races (fix round 1 / fix round 2 Minor 2) ---------------


def test_duplicate_records_for_one_race_are_collapsed():
    """同じ race_id の記録が2件（predicted_at が違う）あっても、学習に使われる
    レース数は1になる。record_id でグルーピングすると2つの独立した観測として
    尤度に入り、1つのレースを2回数えて標準誤差を過小評価する (fix round 1)。
    """
    horses_a = (_horse("01", 0.7), _horse("02", 0.3))
    horses_b = (_horse("01", 0.6), _horse("02", 0.4))
    older = _record(race_id="dup-race", horses=horses_a, prediction_cutoff=_cutoff(1),
                     predicted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _record(race_id="dup-race", horses=horses_b, prediction_cutoff=_cutoff(1),
                     predicted_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    repo = _StubResultRepo({"dup-race": ["01"]})
    usable, _counts, _mismatch = fit_weights._filter_usable([older, newer], set(ig.feature_names()), repo)

    deduped, _duplicate_count = fit_weights._collapse_duplicate_races(usable)

    assert len(deduped) == 1


def test_the_newest_record_wins():
    """重複を畳むとき残るのは predicted_at が最も新しい記録。"""
    older_horses = (_horse("01", 0.5), _horse("02", 0.5))
    newer_horses = (_horse("01", 0.9), _horse("02", 0.1))
    older = _record(race_id="dup-race", horses=older_horses, prediction_cutoff=_cutoff(1),
                     predicted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _record(race_id="dup-race", horses=newer_horses, prediction_cutoff=_cutoff(1),
                     predicted_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    repo = _StubResultRepo({"dup-race": ["01"]})
    usable, _counts, _mismatch = fit_weights._filter_usable([older, newer], set(ig.feature_names()), repo)

    deduped, _duplicate_count = fit_weights._collapse_duplicate_races(usable)

    assert len(deduped) == 1
    kept_horses = {h.umaban: h for h in deduped[0].record.horses}
    assert kept_horses["01"].features["logit_layer1"] == pytest.approx(ig.logit(0.9))


def test_duplicates_are_counted_in_the_exclusion_report():
    """畳んで落とした件数が除外の内訳に duplicate_race として出る
    （黙って落とさない。理由ごとの件数を必ず表示する既存の方針と同じ）。
    """
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    older = _record(race_id="dup-race", horses=horses, prediction_cutoff=_cutoff(1),
                     predicted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _record(race_id="dup-race", horses=horses, prediction_cutoff=_cutoff(1),
                     predicted_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    repo = _StubResultRepo({"dup-race": ["01"]})
    usable, _counts, _mismatch = fit_weights._filter_usable([older, newer], set(ig.feature_names()), repo)

    _deduped, duplicate_count = fit_weights._collapse_duplicate_races(usable)

    assert duplicate_count == 1


def test_usable_old_record_not_overwritten_by_unusable_new_record():
    """規則は「最新を残す」ではなく「使える記録のうち最新を残す」
    (fix round 2 Minor 2)。畳み込みより先に使える記録へ絞ることで、
    「再予想したら層2が一部失敗した」ケースで使えた古い記録が
    使えない新しい記録に理由も分からず潰されることを防ぐ。
    """
    usable_old = _record(
        race_id="dup-race", horses=(_horse("01", 0.7), _horse("02", 0.3)),
        prediction_cutoff=_cutoff(1), predicted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    unusable_new = _record(
        race_id="dup-race", horses=(_horse("01", 0.6), _horse("02", 0.4)),
        prediction_cutoff=_cutoff(1), predicted_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        status="PARTIAL")
    repo = _StubResultRepo({"dup-race": ["01"]})

    usable, counts, _mismatch = fit_weights._filter_usable(
        [usable_old, unusable_new], set(ig.feature_names()), repo)
    deduped, duplicate_count = fit_weights._collapse_duplicate_races(usable)

    assert counts["status_not_succeeded"] == 1
    assert duplicate_count == 0  # 畳み込みの対象はそもそも1件しか残っていない
    assert len(deduped) == 1
    assert deduped[0].record.predicted_at == usable_old.predicted_at


# --- _split_races / 漏洩 (fix round 2 Critical 3) --------------------------------


def test_a_race_never_appears_in_two_splits():
    """同じレースを別の cutoff で予想し直すと、境界をまたいで train と dev
    両方に入ってしまう漏洩を防ぐ。畳み込みは split の前に、全レコードに
    対して1回だけ行う（要）。
    """
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    train_side = _record(
        race_id="same-race", horses=horses,
        prediction_cutoff=datetime(2025, 3, 1, 20, 0, 0, tzinfo=timezone.utc),
        predicted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    dev_side = _record(
        race_id="same-race", horses=horses,
        prediction_cutoff=datetime(2025, 7, 1, 20, 0, 0, tzinfo=timezone.utc),
        predicted_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    repo = _StubResultRepo({"same-race": ["01"]})

    usable, _counts, _mismatch = fit_weights._filter_usable(
        [train_side, dev_side], set(ig.feature_names()), repo)
    deduped, _duplicate_count = fit_weights._collapse_duplicate_races(usable)
    buckets = fit_weights._split_races(
        deduped, train_until=date(2025, 6, 1), dev_until=date(2025, 8, 1),
        calib_until=date(2025, 10, 1))

    train_race_ids = {r.record.race_id for r in buckets["train"]}
    dev_race_ids = {r.record.race_id for r in buckets["dev"]}
    assert not (train_race_ids & dev_race_ids)
    # 全体で1レースぶんしか残らない (新しい方 = dev 側の記録が残る)。
    assert sum(len(v) for v in buckets.values()) == 1


def test_duplicates_across_splits_are_counted():
    """境界をまたぐ重複も duplicate_race に数えられる（畳んだ後に split する
    ので、旧実装のように split ごとに 0 と出て見えなくなることがない）。
    """
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    train_side = _record(
        race_id="same-race", horses=horses,
        prediction_cutoff=datetime(2025, 3, 1, 20, 0, 0, tzinfo=timezone.utc),
        predicted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    dev_side = _record(
        race_id="same-race", horses=horses,
        prediction_cutoff=datetime(2025, 7, 1, 20, 0, 0, tzinfo=timezone.utc),
        predicted_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    repo = _StubResultRepo({"same-race": ["01"]})

    usable, _counts, _mismatch = fit_weights._filter_usable(
        [train_side, dev_side], set(ig.feature_names()), repo)
    _deduped, duplicate_count = fit_weights._collapse_duplicate_races(usable)

    assert duplicate_count == 1


# --- build_training_matrix: race_id でグルーピングする ---------------------------


def test_groups_use_race_id():
    """(重複が既に畳まれている前提で) build_training_matrix のグルーピング鍵は
    race_id である。"""
    horses_a = (_horse("01", 0.7), _horse("02", 0.3))
    horses_b = (_horse("01", 0.6), _horse("02", 0.4))
    record_a = _record(race_id="race-a", horses=horses_a, prediction_cutoff=_cutoff(1))
    record_b = _record(race_id="race-b", horses=horses_b, prediction_cutoff=_cutoff(2))

    races = [
        fit_weights._UsableRace(record=record_a, winners=("01",)),
        fit_weights._UsableRace(record=record_b, winners=("01",)),
    ]

    X, groups, y = fit_weights.build_training_matrix(races, ig.feature_names(with_layer2=True))

    assert X.shape == (4, len(ig.feature_names(with_layer2=True)))
    assert set(groups) == {"race-a", "race-b"}
    for race_id in ("race-a", "race-b"):
        mask = groups == race_id
        assert y[mask].sum() == pytest.approx(1.0)


# --- combine_data_versions (fix round 1) -----------------------------------------


def test_data_version_records_the_single_version_used():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1),
                      data_version="20260901T000000Z")
    race = fit_weights._UsableRace(record=record, winners=("01",))

    assert fit_weights.combine_data_versions([race]) == "20260901T000000Z"


def test_data_version_records_every_version_used():
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    r1 = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1),
                 data_version="20260922T070053Z")
    r2 = _record(race_id="r2", horses=horses, prediction_cutoff=_cutoff(2),
                 data_version="20260901T000000Z")
    races = [fit_weights._UsableRace(record=r1, winners=("01",)),
             fit_weights._UsableRace(record=r2, winners=("01",))]

    assert fit_weights.combine_data_versions(races) == "20260901T000000Z,20260922T070053Z"


# --- require_consistent_versions (fix round 2 Critical 1) -----------------------


def test_versions_come_from_the_training_records():
    """記録の prompt_version が現在の値と違っても、記録側の値が書かれる。"""
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    record = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1),
                      prompt_version="OLD-PROMPT-v0")
    race = fit_weights._UsableRace(record=record, winners=("01",))

    versions = fit_weights.require_consistent_versions([race])

    assert versions["prompt_version"] == "OLD-PROMPT-v0"


def test_mixed_prompt_version_is_an_error():
    """記録間で prompt_version が混在していたらエラー終了、メッセージに
    キー名が出る。"""
    horses = (_horse("01", 0.7), _horse("02", 0.3))
    r1 = _record(race_id="r1", horses=horses, prediction_cutoff=_cutoff(1),
                  prompt_version="v1")
    r2 = _record(race_id="r2", horses=horses, prediction_cutoff=_cutoff(2),
                  prompt_version="v2")
    races = [fit_weights._UsableRace(record=r1, winners=("01",)),
             fit_weights._UsableRace(record=r2, winners=("01",))]

    with pytest.raises(ValueError, match="prompt_version"):
        fit_weights.require_consistent_versions(races)


# --- _is_loadable_by_predict (fix round 2 Critical 2) ----------------------------


def test_full_27_column_set_is_loadable():
    assert fit_weights._is_loadable_by_predict(ig.feature_names(with_layer2=True)) is True


def test_baseline_17_column_set_is_loadable():
    assert fit_weights._is_loadable_by_predict(ig.feature_names(with_layer2=False)) is True


def test_reduced_11_column_set_is_not_loadable():
    """logit_layer1 + 軸10列 (Python側を除いた構成) は with_layer2=True/False
    のどちらとも一致しないので、predict() は読めない。"""
    reduced = tuple(
        n for n in ig.feature_names(with_layer2=True) if n not in ig.python_feature_block())
    assert fit_weights._is_loadable_by_predict(reduced) is False


# --- end-to-end: CLI が weights ファイルを書き出す -----------------------------


def _write_result_data(tmp_path: Path, race_ids: list[str], *,
                        version_id: str = "20260322T010000Z") -> Path:
    """race_ids それぞれについて「01番が勝ち、02番が2着」の結果 CSV を書く。"""
    races = [fx.make_race(race_id=rid) for rid in race_ids]
    race_horses = []
    for rid in race_ids:
        race_horses.append(fx.make_race_horse(
            race_id=rid, umaban="01", ketto_num="0000000001", bamei="イチバンウマ",
            result_kakutei_jyuni="1", result_ninki="1"))
        race_horses.append(fx.make_race_horse(
            race_id=rid, umaban="02", ketto_num="0000000002", bamei="ニバンウマ",
            result_kakutei_jyuni="2", result_ninki="2"))
    return fx.write_dataset(tmp_path, races, race_horses, [], version_id=version_id)


def _build_end_to_end_dataset(
    tmp_path: Path, *, data_version: str = "20260101T000000Z",
    store_version_id: str = "20260322T010000Z",
    prompt_version: str | None = None, feature_version: str | None = None,
    integration_version: str | None = None,
    horse_factory=None,
) -> tuple[Path, Path]:
    """train 4件・dev 2件・calib 2件の記録と、それに対応する結果 CSV を書く。

    ``data_version``（記録側の versions["data_version"]）と ``store_version_id``
    （--data-dir が指す DataStore の版）を別の値にできる――fix round 1 の
    「versions["data_version"] は現在開いている版ではなく実際に使った記録の版」
    を確かめるため。``prompt_version``/``feature_version``/``integration_version``
    は fix round 2 Critical 1 の確認用。``horse_factory(umaban, win_prob, *, is_train)``
    を渡すと馬の組み立て方を差し替えられる（Critical 2 の削減構成勝利テスト用）。
    """
    data_dir = tmp_path / "data"
    predictions_dir = tmp_path / "predictions_src"

    train_ids = [f"train{i:02d}" for i in range(4)]
    dev_ids = [f"dev{i:02d}" for i in range(2)]
    calib_ids = [f"calib{i:02d}" for i in range(2)]
    _write_result_data(data_dir, train_ids + dev_ids + calib_ids, version_id=store_version_id)

    def make_horses(is_train: bool) -> tuple[records_store.HorsePrediction, ...]:
        if horse_factory is None:
            return (_horse("01", 0.7), _horse("02", 0.3))
        return horse_factory(is_train=is_train)

    def save_pair(race_id: str, cutoff_day_offset: int, month: int, *, is_train: bool) -> None:
        record = _record(
            race_id=race_id, horses=make_horses(is_train),
            prediction_cutoff=datetime(2025, month, 1 + cutoff_day_offset, 20, 0, 0,
                                        tzinfo=timezone.utc),
            # predicted_at をレースごとに変え、月をまたがせる
            # (_load_all_records が複数の月ファイルを正しくまとめて読むことの確認)。
            predicted_at=datetime(2026, 1 + (cutoff_day_offset % 2), 1, 0, 0, 0,
                                   tzinfo=timezone.utc),
            data_version=data_version, prompt_version=prompt_version,
            feature_version=feature_version, integration_version=integration_version,
        )
        records_store.save(record, data_dir=predictions_dir)

    for i, rid in enumerate(train_ids):
        save_pair(rid, i, month=3, is_train=True)  # 2025-03: train 期間
    for i, rid in enumerate(dev_ids):
        save_pair(rid, i, month=7, is_train=False)  # 2025-07: dev 期間
    for i, rid in enumerate(calib_ids):
        save_pair(rid, i, month=9, is_train=False)  # 2025-09: calib 期間

    return data_dir, predictions_dir


_STANDARD_CLI_ARGS = [
    "--from", "20250101", "--to", "20251231",
    "--train-until", "20250601",
    "--dev-until", "20250801",
    "--calib-until", "20251001",
]


def test_cli_writes_weights_file_end_to_end(tmp_path, capsys):
    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "除外[no_winner]: 0 件" in captured.out
    assert "除外[duplicate_race]: 0 件" in captured.out
    assert "除外[non_finite_feature]: 0 件" in captured.out
    assert "除外[out_of_range]: 0 件" in captured.out
    assert "書き出し先" in captured.out

    written = list((tmp_path / "weights_out").glob("*.json"))
    assert len(written) == 1
    weights = Weights.load(written[0])
    assert weights.calibration is not None
    assert weights.calibration["method"] == "isotonic"
    assert weights.splits == {
        "train_until": "20250601", "dev_until": "20250801", "calib_until": "20251001",
    }
    assert weights.versions["feature_version"] is not None
    # 全記録が同じ data_version (既定 "20260101T000000Z") で保存されているので、
    # 単一版のまま (カンマ連結にならない)。
    assert weights.versions["data_version"] == "20260101T000000Z"


def test_written_weights_are_loadable_by_predict(tmp_path):
    """書かれた重みの feature_names が feature_names(with_layer2=True) と
    完全一致する（要）。旧アサーション ``set(...) in (27列, 11列)`` は
    採用構成が入れ替わっても検出できなかった。
    """
    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])
    assert exit_code == 0

    [weights_path] = list((tmp_path / "weights_out").glob("*.json"))
    weights = Weights.load(weights_path)
    assert weights.feature_names == ig.feature_names(with_layer2=True)


def test_data_version_is_not_the_currently_open_version(tmp_path):
    """記録側の data_version と、現在 --data-dir が指す DataStore の version_id
    が違うとき、Weights.versions["data_version"] には記録側の値が入る
    （現在開いている版を書くと「この重みはこの版で学習した」という事実で
    ない主張が残る。fix round 1）。
    """
    data_dir, predictions_dir = _build_end_to_end_dataset(
        tmp_path, data_version="20260101T000000Z", store_version_id="99999999T999999Z")

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])
    assert exit_code == 0

    [weights_path] = list((tmp_path / "weights_out").glob("*.json"))
    weights = Weights.load(weights_path)
    assert weights.versions["data_version"] == "20260101T000000Z"
    assert weights.versions["data_version"] != "99999999T999999Z"


def test_check_compatible_rejects_weights_from_an_old_prompt(tmp_path):
    """古い記録（prompt_version が現在のコードと違う）から作った重みを
    check_compatible() すると WeightsIncompatibleError になる（要）。
    """
    data_dir, predictions_dir = _build_end_to_end_dataset(
        tmp_path, prompt_version="OLD-PROMPT-v0", feature_version="OLD-FEATURE-v0")

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])
    assert exit_code == 0

    [weights_path] = list((tmp_path / "weights_out").glob("*.json"))
    weights = Weights.load(weights_path)
    assert weights.versions["prompt_version"] == "OLD-PROMPT-v0"
    assert weights.versions["feature_version"] == "OLD-FEATURE-v0"
    with pytest.raises(ig.WeightsIncompatibleError):
        weights.check_compatible()


def test_reduced_configuration_is_compared_but_not_written(tmp_path, capsys):
    """--without-python-features を選んでも、比較表には両構成の dev 指標が
    出る。predict() が読めない構成なので重みファイルは書かれない。
    """
    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--without-python-features",
        "--out", str(tmp_path / "weights_out"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "with_python_features" in captured.out
    assert "without_python_features" in captured.out
    assert "書き出しません" in captured.out

    written = list((tmp_path / "weights_out").glob("*.json"))
    assert written == []


def test_reduced_configuration_win_is_reported_loudly(tmp_path, capsys):
    """削減構成 (without_python_features) の dev 指標が採用構成を上回ったとき、
    その旨と「推論側の変更が必要」が表示される。

    train では corner_pos_mean が勝ち馬と完全に一致し (winner=1.0, loser=0.0)、
    dev ではその相関を逆転させる (winner=0.0, loser=1.0)。win_prob 自体は
    弱い手がかり (0.55/0.45) のまま train/dev で一貫させる。全27列を見る
    構成は corner_pos_mean に強く依存する重みを学習して dev で自滅するが、
    それを見ない削減構成は弱い win_prob の手がかりだけで dev もそこそこ
    当てるので、削減構成が dev 対数尤度で上回る。
    """
    def horse_factory(*, is_train: bool) -> tuple[records_store.HorsePrediction, ...]:
        winner_corner, loser_corner = (1.0, 0.0) if is_train else (0.0, 1.0)
        return (
            _horse_with_corner("01", 0.55, winner_corner),
            _horse_with_corner("02", 0.45, loser_corner),
        )

    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path, horse_factory=horse_factory)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--lambda", "0.001",  # 削減構成が dev で上回るよう正則化を弱く固定する
        "--out", str(tmp_path / "weights_out"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "上回りました" in captured.out
    assert "列部分集合モード" in captured.out


def test_cli_rejects_out_of_order_boundaries(tmp_path, capsys):
    exit_code = fit_weights.main([
        "--data-dir", str(tmp_path / "data"),
        "--predictions-from", str(tmp_path / "predictions_src"),
        "--from", "20250101", "--to", "20251231",
        "--train-until", "20250901",  # dev-until より後ろ (順序違反)
        "--dev-until", "20250601",
        "--calib-until", "20251001",
    ])

    assert exit_code == 1
    assert "順序どおりではありません" in capsys.readouterr().out


def test_cli_never_imports_jev_client():
    """fit_weights.py は Jev の API クライアントを一切構築しない
    (jev_client / typesafe_sdk への参照が無い)。"""
    source = Path(fit_weights.__file__).read_text(encoding="utf-8")
    assert "jev_client" not in source
    assert "typesafe_sdk" not in source
    assert "TypeSafeClient" not in source


# --- --without-python-features (fix round 2 Important 2) ------------------------


def test_without_python_features_flag_is_accepted():
    """--without-python-features が解釈され、構成が切り替わる。"""
    args = fit_weights._parse_args([
        "--predictions-from", "x", *_STANDARD_CLI_ARGS, "--without-python-features",
    ])
    assert args.with_python_features is False


def test_with_python_features_is_the_default():
    """どちらも指定しなければ True。"""
    args = fit_weights._parse_args(["--predictions-from", "x", *_STANDARD_CLI_ARGS])
    assert args.with_python_features is True


def test_printed_flag_names_are_real():
    """比較表の見出しに出るフラグ名の文字列が、_parse_args が実際に受け付ける
    フラグ名と一致する（解析器に実際に通して確かめる。要）。
    """
    header = "--with-python-features / --without-python-features"
    flag_names = [token.strip() for token in header.split("/")]
    assert flag_names == ["--with-python-features", "--without-python-features"]
    for flag in flag_names:
        # SystemExit (exit code 2) を起こさず解析できることが「実在する」の意味。
        args = fit_weights._parse_args(["--predictions-from", "x", *_STANDARD_CLI_ARGS, flag])
        assert isinstance(args.with_python_features, bool)


# --- --lambda-grid の値域 (fix round 2 Minor 5) ----------------------------------


def test_lambda_grid_rejects_empty_string():
    with pytest.raises(ValueError, match="空"):
        fit_weights._parse_lambda_grid("")


def test_lambda_grid_rejects_negative_values():
    with pytest.raises(ValueError, match="負"):
        fit_weights._parse_lambda_grid("-1,0.1")


def test_lambda_grid_accepts_zero():
    """l2 == 0.0 は実在の値なので通る（グリッドの境界値）。"""
    assert fit_weights._parse_lambda_grid("0,0.1") == (0.0, 0.1)


# --- fix5 決定6: weights_id / fitted_at / dev_metrics ----------------------------


def test_weights_id_is_the_fit_timestamp(tmp_path):
    """weights_id は学習実行時刻の YYYYMMDDTHHMMSSZ（仕様 §11.1.2）。

    uuid4 だと「いつ作ったのか」「2つのうちどちらが新しいのか」がファイル名
    からも中身からも分からない。
    """
    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])
    assert exit_code == 0

    [weights_path] = list((tmp_path / "weights_out").glob("*.json"))
    weights = Weights.load(weights_path)

    assert re.fullmatch(r"\d{8}T\d{6}Z", weights.weights_id), weights.weights_id
    assert weights_path.stem == weights.weights_id
    # fitted_at は同じ瞬間を指す（now() を2回呼んでずれていない）。
    assert weights.fitted_at.startswith(
        f"{weights.weights_id[:4]}-{weights.weights_id[4:6]}-{weights.weights_id[6:8]}")


def test_weights_record_the_dev_metrics_that_chose_lambda(tmp_path):
    """dev_metrics に、選んだ λ の dev 対数尤度と両構成の比較値が入る。"""
    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])
    assert exit_code == 0

    [weights_path] = list((tmp_path / "weights_out").glob("*.json"))
    weights = Weights.load(weights_path)

    assert weights.dev_metrics["chosen_configuration"] == "with_python_features"
    assert weights.dev_metrics["lambda"] == weights.lambda_
    assert isinstance(weights.dev_metrics["dev_log_likelihood_per_race"], float)
    configurations = weights.dev_metrics["configurations"]
    assert set(configurations) == {"with_python_features", "without_python_features"}
    for values in configurations.values():
        assert "dev_log_likelihood_per_race" in values
        assert "lambda" in values


def test_feature_mismatch_reports_the_column_counts(tmp_path, capsys):
    """feature_mismatch の件数に原因（要求列数と記録側の列数）を1行足す（決定9-6）。"""
    record = _record(race_id="r1", horses=(_horse("01", 0.6), _horse("02", 0.4)),
                      prediction_cutoff=datetime(2025, 3, 1, 20, 0, 0, tzinfo=timezone.utc))
    # 17列（層2抜き）の記録を、27列を要求する実行に混ぜる。
    trimmed = tuple(
        dataclasses.replace(horse, features={
            name: value for name, value in horse.features.items()
            if name in set(ig.feature_names(with_layer2=False))
        })
        for horse in record.horses
    )
    predictions_dir = tmp_path / "predictions_src"
    records_store.save(dataclasses.replace(record, horses=trimmed), data_dir=predictions_dir)
    data_dir = _write_result_data(tmp_path / "data", ["r1"])

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1, captured.out          # train が0件になる
    assert "除外[feature_mismatch]: 1 件" in captured.out
    assert "27列" in captured.out
    assert "17列" in captured.out


def test_float64_indistinguishable_columns_are_warned_about(tmp_path, capsys, monkeypatch):
    """準定数列があれば警告を出す（落とさない。決定9-2）。

    実データの記録で狙って作るのは難しいので、fit の結果に名前が載っていれば
    警告が出ることを、FitResult を差し替えて確かめる。
    """
    data_dir, predictions_dir = _build_end_to_end_dataset(tmp_path)
    real_fit = cl.fit

    def fake_fit(*args, **kwargs):
        result = real_fit(*args, **kwargs)
        return dataclasses.replace(
            result, float64_indistinguishable_features=("corner_pos_mean",))

    monkeypatch.setattr(fit_weights.cl, "fit", fake_fit)

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "corner_pos_mean" in captured.out
    assert "標準偏差が数値的に区別できない" in captured.out
    assert "落としていません" in captured.out
    # 警告した列の名前は重みファイルにも残る（あとから解釈するときの手がかり）。
    [weights_path] = list((tmp_path / "weights_out").glob("*.json"))
    weights = Weights.load(weights_path)
    assert weights.dev_metrics["float64_indistinguishable_features"] == ["corner_pos_mean"]


# --- fix6 ①: 旧版の記録と新版の記録を混ぜて学習させない -------------------------


def test_mixed_prompt_version_is_refused(tmp_path, capsys):
    """旧版（prompt_version="1"）の記録と現行版の記録が混ざったらエラー終了する（要）。

    require_consistent_versions() は「文言の違う記録を混ぜて学習させない」ために
    作られた関数だが、第1層の本文を直したのに PROMPT_VERSION を上げなかった間は、
    文言の違う2世代が同じ "1" を名乗っていて無警告で通っていた（実データで
    291件 + 100件の混在）。版を上げることで、この混在が機構から見えるようになる。
    """
    race_ids = ["train00", "train01", "dev00", "calib00"]
    data_dir = _write_result_data(tmp_path / "data", race_ids)
    predictions_dir = tmp_path / "predictions_src"

    def save(race_id: str, cutoff: datetime, prompt_version: str | None = None) -> None:
        records_store.save(
            _record(race_id=race_id, horses=(_horse("01", 0.7), _horse("02", 0.3)),
                     prediction_cutoff=cutoff, prompt_version=prompt_version),
            data_dir=predictions_dir)

    # 旧版の記録（プロンプトが ASCII "." 終わりだった世代）。
    save("train00", datetime(2025, 3, 1, 20, 0, 0, tzinfo=timezone.utc), prompt_version="1")
    # 現行版の記録（版番号は records.current_versions が現在のコードから埋める）。
    save("train01", datetime(2025, 3, 2, 20, 0, 0, tzinfo=timezone.utc))
    save("dev00", datetime(2025, 7, 1, 20, 0, 0, tzinfo=timezone.utc))
    save("calib00", datetime(2025, 9, 1, 20, 0, 0, tzinfo=timezone.utc))

    exit_code = fit_weights.main([
        "--data-dir", str(data_dir),
        "--predictions-from", str(predictions_dir),
        *_STANDARD_CLI_ARGS,
        "--out", str(tmp_path / "weights_out"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1, captured.out
    assert "prompt_version" in captured.out
    # 両方の版番号がメッセージに出る（どちらとどちらが混ざったのか分かる形）。
    # 現行版はリテラルで書かない——版を上げるたびにこのテストが
    # 「混在を拒否できたか」ではなく「版番号がいくつか」を測り始める。
    assert "'1'" in captured.out
    assert f"'{PROMPT_VERSION}'" in captured.out
    assert not list((tmp_path / "weights_out").glob("*.json"))

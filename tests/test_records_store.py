"""予測記録の保存 (save/load_month) と利用者向けビュー (to_user_view)。

保存側 (PredictionRecord) は検証フェーズが読み戻す完全な記録、利用者向け
(to_user_view) は spec §8 / §12.8 が定める狭い範囲。この2つの形を混同しない
ことがこのテストファイルの主眼。
"""

from __future__ import annotations

import dataclasses
import json
import types
import uuid
from datetime import datetime

import pytest

from keiba import config
from keiba.predict.questions import PROMPT_VERSION
from keiba.store import records


# --- フィクスチャ組み立て ------------------------------------------------------


def _make_horse(
    umaban: str,
    *,
    rank_baseline: int,
    rank_model: int | None = None,
    tie_group: int = 1,
    layer1_win_prob: float = 0.1,
    model_score: float | None = None,
    model_win_prob: float | None = None,
    calibrated_win_prob: float | None = None,
    score_breakdown: dict | None = None,
    features: dict | None = None,
    layer2: dict | None = None,
    positives: tuple[str, ...] = (),
    negatives: tuple[str, ...] = (),
    insufficient: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> records.HorsePrediction:
    return records.HorsePrediction(
        umaban=umaban,
        bamei=f"テストウマ{umaban}",
        rank_baseline=rank_baseline,
        rank_model=rank_model,
        tie_group=tie_group,
        layer1_win_prob=layer1_win_prob,
        model_score=model_score,
        model_win_prob=model_win_prob,
        calibrated_win_prob=calibrated_win_prob,
        score_breakdown=score_breakdown or {},
        features=features or {},
        layer2=layer2 or {},
        positives=tuple(positives),
        negatives=tuple(negatives),
        insufficient=tuple(insufficient),
        notes=tuple(notes),
    )


def _make_record(
    *,
    n: int = 3,
    num_horses: int = 5,
    predicted_at: datetime | None = None,
    prediction_cutoff: datetime | None = None,
    status: str = "SUCCEEDED",
    record_id: str = "rid-0",
    race_id: str | None = "2026032106010511",
    data_fetched_at: str | None = None,
    tie_band_scale: str = "layer1_win_prob",
) -> records.PredictionRecord:
    predicted_at = predicted_at or datetime(2026, 3, 21, 10, 0, 0)
    prediction_cutoff = prediction_cutoff or predicted_at
    versions = records.current_versions(
        data_version="20260320T000000Z", model="jev-1.13.0", weights_id="w1")
    if data_fetched_at is not None:
        versions = dict(versions)
        versions["data_fetched_at"] = data_fetched_at
    horses = tuple(
        _make_horse(
            str(i + 1),
            rank_baseline=i + 1,
            rank_model=i + 1,
            # 既定では1頭ずつ別の組にする (同着を前提にしない既存のテストの
            # 挙動を保つ)。同程度の組そのものを確かめるテストは、この既定を
            # 明示的に上書きする。
            tie_group=i + 1,
            layer1_win_prob=1.0 / num_horses if num_horses else 0.0,
        )
        for i in range(num_horses)
    )
    return records.PredictionRecord(
        record_id=record_id,
        race_id=race_id,
        predicted_at=predicted_at,
        prediction_cutoff=prediction_cutoff,
        n=n,
        versions=versions,
        horses=horses,
        layer1={"choice": "1", "confidence": 0.5},
        # data_quality()（features/summary.py）は "notes"（レース単位）と
        # "horse_notes"（馬単位、馬番キー）を必ず作る。to_user_view は
        # "notes" を .get ではなく直接引くので、フィクスチャ側も実物の形に
        # 合わせておく（fix3 決定1 / fix4 決定1）。
        # predict() は tie_band_scale も必ず書く（同程度の判定をどの尺度で
        # 行ったか、あるいは行っていないか）。to_user_view は .get ではなく
        # 直接引くので、フィクスチャ側も実物の形に合わせる（fix5 決定1）。
        data_quality={"notes": [], "horse_notes": {},
                       "tie_band_scale": tie_band_scale},
        usage={"input_tokens": 100},
        status=status,
        race_input={"race_id": race_id},
    )


# --- 較正後の確率は記録にだけ置く（spec §10.3）--------------------------------

def test_user_view_never_exposes_the_calibrated_probability():
    """較正後の確率を利用者に見せない。

    spec §10.3 は「``model`` の出力を勝率として表示するのは、較正が**別期間で
    検証できたときだけ**」としている。較正マップを本番経路に配線した
    （report2 §6 の #8）ことで ``calibrated_win_prob`` には値が入るように
    なったが、**検証はまだ済んでいない**——出荷している較正マップは節点10個・
    calib 16レースで作ったものでしかない。記録として残すこと（§11.2 が要求）と
    利用者に勝率として見せること（§10.3 が禁じる）は別なので、ここで後者を
    塞いでおく。``to_user_view`` は削るのではなく出す項目を列挙する作りなので、
    このテストは「うっかり列挙に足した」ときに落ちる。
    """
    record = _make_record(num_horses=2)
    record = dataclasses.replace(record, horses=tuple(
        dataclasses.replace(h, calibrated_win_prob=0.123456789, model_win_prob=0.5)
        for h in record.horses))

    view = records.to_user_view(record)

    assert "0.123456789" not in json.dumps(view, ensure_ascii=False)
    assert "較正" not in json.dumps(view, ensure_ascii=False)
    for horse in view["予想"]:
        assert "calibrated_win_prob" not in horse


# --- save / load_month --------------------------------------------------------


def test_save_appends_one_line_per_record(tmp_path):
    records.save(_make_record(record_id="r1"), data_dir=tmp_path)
    records.save(_make_record(record_id="r2"), data_dir=tmp_path)
    path = tmp_path / "predictions" / "202603.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["record_id"] == "r1"
    assert json.loads(lines[1])["record_id"] == "r2"


def test_save_uses_year_month_file(tmp_path):
    record = _make_record(predicted_at=datetime(2026, 3, 5, 12, 0, 0))
    path = records.save(record, data_dir=tmp_path)
    assert path == tmp_path / "predictions" / "202603.jsonl"
    assert path.exists()


def test_roundtrip_preserves_all_fields(tmp_path):
    """save → load_month で全項目一致する (dict/tuple の中身を含む)。

    layer1/data_quality/usage/race_input と layer2 の中身は JSON 上 list に
    しかならないので、ここでも list で組む――tuple を混ぜると保存側と
    読み出し側で型が食い違い、この比較が失敗する。往復で tuple のまま保つ
    のは HorsePrediction が明示的に tuple 型で宣言しているフィールド
    (positives/negatives/insufficient/notes と horses そのもの) だけ。
    """
    horse_a = _make_horse(
        "07",
        rank_baseline=2,
        rank_model=1,
        layer1_win_prob=0.234,
        model_score=1.23,
        model_win_prob=0.31,
        calibrated_win_prob=None,
        score_breakdown={"logit_layer1": 0.5, "distance_fit_signed": 0.1},
        layer2={
            "distance_fit": {
                "choice": "positive",
                "probabilities": {
                    "positive": 0.7, "neutral": 0.2, "negative": 0.05, "insufficient": 0.05,
                },
                "confidence": 0.8,
            },
        },
        positives=("同距離 8戦3勝(3着内5)",),
        negatives=(),
        insufficient=("輸送",),
    )
    horse_b = _make_horse("03", rank_baseline=1, rank_model=2, layer1_win_prob=0.3)
    record = records.PredictionRecord(
        record_id="rid-1",
        race_id="2026032106010511",
        predicted_at=datetime(2026, 3, 21, 9, 0, 0),
        prediction_cutoff=datetime(2026, 3, 21, 9, 30, 0),
        n=1,
        versions=records.current_versions(
            data_version="20260320T000000Z", model="jev-1.13.0", weights_id="w1"),
        horses=(horse_a, horse_b),
        layer1={"choice": "07", "confidence": 0.6, "missing_probability_umaban": ["03"]},
        data_quality={"unregistered": ["99"]},
        usage={"layer1_input_tokens": 500, "layer2_input_tokens": 1200},
        status="SUCCEEDED",
        race_input={"race_id": "2026032106010511", "entries": ["07", "03"]},
    )
    record2 = _make_record(record_id="rid-2", predicted_at=datetime(2026, 3, 21, 9, 1, 0))

    records.save(record, data_dir=tmp_path)
    records.save(record2, data_dir=tmp_path)
    loaded = records.load_month("202603", data_dir=tmp_path)

    assert len(loaded) == 2
    assert loaded[0] == record
    assert loaded[1] == record2
    # 個別フィールドでも突き合わせる (dataclass の == に頼り切らない)
    assert loaded[0].horses[0].layer2 == horse_a.layer2
    assert loaded[0].horses[0].score_breakdown == horse_a.score_breakdown
    assert loaded[0].horses[0].positives == ("同距離 8戦3勝(3着内5)",)
    assert isinstance(loaded[0].horses[0].positives, tuple)
    assert loaded[0].data_quality == {"unregistered": ["99"]}
    assert loaded[0].predicted_at == record.predicted_at
    assert loaded[0].prediction_cutoff == record.prediction_cutoff


def test_record_keeps_every_horse_not_just_top_n(tmp_path):
    record = _make_record(num_horses=18, n=3)
    records.save(record, data_dir=tmp_path)
    [loaded] = records.load_month("202603", data_dir=tmp_path)
    assert len(loaded.horses) == 18
    view = records.to_user_view(loaded)
    assert len(view["予想"]) == 3


def test_versions_include_all_required_keys():
    versions = records.current_versions(
        data_version="20260320T000000Z", model="jev-1.13.0", weights_id="w1")
    required = {
        "data_version", "schema_version", "model", "prompt_version",
        "feature_version", "integration_version", "exception_rules_version", "weights_id",
    }
    assert required <= versions.keys()
    assert versions["schema_version"] == config.RECORD_SCHEMA_VERSION
    assert versions["feature_version"] == config.FEATURE_VERSION
    assert versions["integration_version"] == config.INTEGRATION_VERSION
    assert versions["exception_rules_version"] == config.EXCEPTION_RULES_VERSION
    assert versions["prompt_version"] == PROMPT_VERSION
    assert versions["model"] == "jev-1.13.0"
    assert versions["weights_id"] == "w1"


def test_versions_allow_weights_id_none():
    """baseline のみ (重み未適用) の予想では weights_id が None のまま持てる。"""
    versions = records.current_versions(
        data_version="20260320T000000Z", model="jev-1.13.0", weights_id=None)
    assert versions["weights_id"] is None


def test_calibrated_prob_is_none_until_verified(tmp_path):
    record = _make_record(num_horses=3, n=3)
    assert all(h.calibrated_win_prob is None for h in record.horses)
    records.save(record, data_dir=tmp_path)
    [loaded] = records.load_month("202603", data_dir=tmp_path)
    assert all(h.calibrated_win_prob is None for h in loaded.horses)


# --- notes（fix4: 持ち主ごとに持つ）--------------------------------------------


def test_notes_round_trip_as_a_tuple(tmp_path):
    """HorsePrediction.notes は他の tuple フィールドと同じく、
    save → load_month の往復で tuple のまま戻る。"""
    horse = _make_horse("1", rank_baseline=1, rank_model=1,
                         notes=("01番は今回の出馬表がまだ無いため、推定した。",))
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))

    records.save(record, data_dir=tmp_path)
    [loaded] = records.load_month("202603", data_dir=tmp_path)

    assert loaded.horses[0].notes == ("01番は今回の出馬表がまだ無いため、推定した。",)
    assert isinstance(loaded.horses[0].notes, tuple)


def test_from_dict_rejects_a_payload_without_notes():
    """保存済みの予想記録はまだ1件も無いので、``notes`` キーの無い古い形式の
    payload が来たら、``.get`` で黙って空にするのではなく KeyError で落ちる
    べき（ブリーフの明示的な指示）。"""
    horse = _make_horse("1", rank_baseline=1, rank_model=1)
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))
    payload = records.to_json(record)
    del payload["horses"][0]["notes"]

    with pytest.raises(KeyError):
        records.from_json(payload)


# --- features (Task 18 supplement: 学習入力を組み立てるための生の特徴量) ---------


def test_features_round_trip(tmp_path):
    """HorsePrediction.features は save → load_month の往復で全キー・全値が
    保たれる (score_breakdown と同じ自由形式 dict だが、tuple ではない)。"""
    horse = _make_horse(
        "07", rank_baseline=1, rank_model=1,
        features={"logit_layer1": 0.234, "distance_fit_signed": -0.1, "days_since_last_run": 20.0},
    )
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))

    records.save(record, data_dir=tmp_path)
    [loaded] = records.load_month("202603", data_dir=tmp_path)

    assert loaded.horses[0].features == {
        "logit_layer1": 0.234, "distance_fit_signed": -0.1, "days_since_last_run": 20.0,
    }


def test_from_dict_rejects_a_payload_without_features():
    """保存済みの予想記録はまだ1件も無いので、``features`` キーの無い古い
    形式の payload が来たら、``.get`` で黙って既定値を埋めるのではなく
    KeyError で落ちるべき（notes/tie_group と同じ規則）。"""
    horse = _make_horse("1", rank_baseline=1, rank_model=1)
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))
    payload = records.to_json(record)
    del payload["horses"][0]["features"]

    with pytest.raises(KeyError):
        records.from_json(payload)


def test_user_view_does_not_expose_features():
    """to_user_view の出力に特徴量の列名・値が1つも現れない (spec §8 の
    「Jev へ送った state 全体を利用者向けから取得できないようにする」と
    同じ性質の要求。生の特徴量も利用者向けの項目ではない)。"""
    horse = _make_horse(
        "1", rank_baseline=1, rank_model=1,
        features={"logit_layer1": 0.9876543, "distance_fit_signed": -0.333, "corner_pos_mean": 0.42},
    )
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))

    view = records.to_user_view(record)
    dumped = json.dumps(view, ensure_ascii=False)

    assert "features" not in dumped
    assert "logit_layer1" not in dumped
    assert "distance_fit_signed" not in dumped
    assert "corner_pos_mean" not in dumped
    assert "0.9876543" not in dumped
    assert "-0.333" not in dumped


# --- tie_group (fix2) -----------------------------------------------------------


def test_tie_group_round_trips(tmp_path):
    """HorsePrediction.tie_group は他の欄と同じく、save → load_month の
    往復で保たれる。"""
    horse = _make_horse("1", rank_baseline=1, rank_model=1, tie_group=2)
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))

    records.save(record, data_dir=tmp_path)
    [loaded] = records.load_month("202603", data_dir=tmp_path)

    assert loaded.horses[0].tie_group == 2


def test_from_dict_rejects_a_payload_without_tie_group():
    """保存済みの予想記録はまだ1件も無いので、``tie_group`` キーの無い古い
    形式の payload が来たら、``.get`` で黙って既定値を埋めるのではなく
    KeyError で落ちるべき（notes と同じ規則。ブリーフの明示的な指示）。"""
    horse = _make_horse("1", rank_baseline=1, rank_model=1)
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))
    payload = records.to_json(record)
    del payload["horses"][0]["tie_group"]

    with pytest.raises(KeyError):
        records.from_json(payload)


# --- to_user_view --------------------------------------------------------------


def test_user_view_shows_only_top_n():
    record = _make_record(num_horses=8, n=3)
    view = records.to_user_view(record)
    assert len(view["予想"]) == 3
    assert view["上位n頭"] == 3
    assert [h["予想順位"] for h in view["予想"]] == [1, 2, 3]


# --- to_user_view: 同程度の組 (fix2 決定4) ---------------------------------------


def _make_horses_with_groups(groups: list[int]) -> tuple[records.HorsePrediction, ...]:
    """groups[i] を i+1 番目の馬 (rank_baseline=rank_model=i+1) の tie_group に
    する。並び順そのものが順位なので、groups は「組の降順の並び」を表す。
    """
    return tuple(
        _make_horse(str(i + 1), rank_baseline=i + 1, rank_model=i + 1, tie_group=group)
        for i, group in enumerate(groups)
    )


def test_user_view_extends_past_n_when_the_boundary_is_tied():
    """n=3 で3位の属する組が5位まで続くとき、「予想」は5件になる
    （決定4の1〜2: 組の最後まで表示対象にする。n を超えてよい）。
    """
    horses = _make_horses_with_groups([1, 2, 3, 3, 3, 4])
    record = dataclasses.replace(_make_record(num_horses=6, n=3), horses=horses)

    view = records.to_user_view(record)

    assert len(view["予想"]) == 5
    assert [h["馬番"] for h in view["予想"]] == ["1", "2", "3", "4", "5"]


def test_user_view_omits_the_rank_inside_a_tied_group():
    """複数頭の組に属する馬には「予想順位」キーが無い（.get で既定を埋めず、
    キーの有無で「順位をつけられない」を表す。決定4）。
    """
    horses = _make_horses_with_groups([1, 2, 3, 3, 3])
    record = dataclasses.replace(_make_record(num_horses=5, n=3), horses=horses)

    view = records.to_user_view(record)

    tied = [h for h in view["予想"] if h["馬番"] in ("3", "4", "5")]
    assert len(tied) == 3
    for h in tied:
        assert "予想順位" not in h


def test_user_view_keeps_the_rank_for_a_single_horse_group():
    """1頭だけの組には、これまでどおり「予想順位」が入る。"""
    horses = _make_horses_with_groups([1, 2, 3, 3, 3])
    record = dataclasses.replace(_make_record(num_horses=5, n=3), horses=horses)

    view = records.to_user_view(record)

    by_umaban = {h["馬番"]: h for h in view["予想"]}
    assert by_umaban["1"]["予想順位"] == 1
    assert by_umaban["2"]["予想順位"] == 2


def test_user_view_explains_only_multi_horse_groups():
    """「同程度の組の説明」には1頭だけの組の番号が入らない。"""
    horses = _make_horses_with_groups([1, 2, 3, 3, 3])
    record = dataclasses.replace(_make_record(num_horses=5, n=3), horses=horses)

    view = records.to_user_view(record)

    assert "1" not in view["同程度の組の説明"]
    assert "2" not in view["同程度の組の説明"]
    assert "3" in view["同程度の組の説明"]


def test_user_view_quotes_the_band_from_config(monkeypatch):
    """説明文の band は config.TIE_BAND から組む。文字列に焼き込んでいないこと
    を、``TIE_BAND`` を一時的に別の値にして確かめる（説明文もそれに応じて
    変わらなければならない）。
    """
    horses = _make_horses_with_groups([1, 2, 3, 3, 3])
    record = dataclasses.replace(_make_record(num_horses=5, n=3), horses=horses)

    monkeypatch.setattr(config, "TIE_BAND", 0.03)
    view_before = records.to_user_view(record)
    assert str(config.TIE_BAND) in view_before["同程度の組の説明"]["3"]

    monkeypatch.setattr(config, "TIE_BAND", 0.12345)
    view_after = records.to_user_view(record)
    assert str(config.TIE_BAND) in view_after["同程度の組の説明"]["3"]
    assert view_before["同程度の組の説明"]["3"] != view_after["同程度の組の説明"]["3"]


def test_display_rank_falls_back_only_when_rank_model_is_none():
    """rank_model が 0 のときに baseline へ落ちない。

    順位は 1 始まりなので 0 は本来出ないが、``or`` を使うと「0」と
    「未設定」が同じ扱いになる。このプロジェクトで繰り返し出ている
    「偽値と欠損を区別する」規則をここでも守る。

    horse_zero (rank_model=0) と horse_none (rank_model=None) を混ぜる。
    ``or`` 方式なら両方 rank_baseline で並び替わるので、baseline=1 の
    horse_none が baseline=5 の horse_zero より先に出る。``is not None``
    方式なら horse_zero が rank_model=0 で最優先になり、先に出る――
    2つの読み方で順序が変わるように仕組んである。
    """
    # tie_group を別々にする (この2頭が「同程度」に丸められてしまうと
    # 「予想順位」キー自体が消え、このテストが見たい ``is not None`` の
    # 分岐そのものを確かめられなくなる)。
    horse_zero = _make_horse("9", rank_baseline=5, rank_model=0, tie_group=1)
    horse_none = _make_horse("2", rank_baseline=1, rank_model=None, tie_group=2)
    record = dataclasses.replace(
        _make_record(num_horses=2, n=2), horses=(horse_zero, horse_none))

    view = records.to_user_view(record)

    umaban_order = [h["馬番"] for h in view["予想"]]
    assert umaban_order == ["9", "2"]
    assert view["予想"][0]["予想順位"] == 0


def test_user_view_excludes_state():
    """Jev へ送る state と同じ形 (target_horse/opponents_summary/horses) の
    データを record.race_input に埋め込んでも、to_user_view の出力に出ない
    こと (features/summary.py が実際に使うキー名で確かめる)。
    """
    state_like = {
        "target_horse": {"umaban": "1", "ability": {"recent_margin_sec_wavg": 0.3}},
        "opponents_summary": {"2": {"umaban": "2"}},
        "horses": {"1": {"umaban": "1"}},
    }
    record = _make_record()
    record = dataclasses.replace(
        record, race_input={**record.race_input, "state_for_debug": state_like})
    view = records.to_user_view(record)
    dumped = json.dumps(view, ensure_ascii=False)
    assert "target_horse" not in dumped
    assert "opponents_summary" not in dumped
    assert "state_for_debug" not in dumped


def test_user_view_excludes_raw_layer2_probabilities():
    horse = _make_horse(
        "1", rank_baseline=1, rank_model=1,
        layer2={
            "distance_fit": {
                "choice": "positive",
                "probabilities": {
                    "positive": 0.91, "neutral": 0.05, "negative": 0.03, "insufficient": 0.01,
                },
                "confidence": 0.83,
            },
        },
    )
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))
    view = records.to_user_view(record)
    dumped = json.dumps(view, ensure_ascii=False)
    assert "0.91" not in dumped
    assert "probabilities" not in dumped
    assert "distance_fit" not in dumped
    assert "confidence" not in dumped


def test_user_view_excludes_score_breakdown():
    horse = _make_horse(
        "1", rank_baseline=1, rank_model=1,
        score_breakdown={"logit_layer1": 0.42, "distance_fit_signed": 0.1234})
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))
    view = records.to_user_view(record)
    dumped = json.dumps(view, ensure_ascii=False)
    assert "score_breakdown" not in dumped
    assert "logit_layer1" not in dumped
    assert "0.42" not in dumped


def test_user_view_includes_prediction_time_and_data_update_time():
    record = _make_record(
        predicted_at=datetime(2026, 3, 21, 10, 30, 0),
        prediction_cutoff=datetime(2026, 3, 21, 10, 25, 0),
        data_fetched_at="2026-03-20T09:00:00+09:00",
    )
    view = records.to_user_view(record)
    assert view["予想時刻"] == "2026-03-21T10:30:00"
    assert view["予想締切"] == "2026-03-21T10:25:00"
    assert view["データ更新時刻"] == "2026-03-20T09:00:00+09:00"


def test_user_view_includes_insufficient_items():
    horse = _make_horse("1", rank_baseline=1, rank_model=1, insufficient=("輸送", "距離"))
    record = dataclasses.replace(_make_record(num_horses=1, n=1), horses=(horse,))
    view = records.to_user_view(record)
    assert view["予想"][0]["判断材料が不足している項目"] == ["輸送", "距離"]


def test_user_view_labels_rank_as_likelihood_not_finish_order():
    record = _make_record()
    view = records.to_user_view(record)
    assert "勝ちやすさ" in view["注意"]
    assert "予想順位" in view["予想"][0]
    assert "着順" not in view["予想"][0].keys()


def test_user_view_ignores_fields_not_in_the_inclusion_list():
    """列挙方式であることの確認: record に無い属性 (raw_api_response) が後から
    足されても、frozen dataclass ではない代役オブジェクトを介して to_user_view
    に渡し、それが出力に出ないことを確かめる。除外リスト方式なら、この種の
    新規フィールドは気づかれずに漏れる。
    """
    base = _make_record()
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    fields["raw_api_response"] = {"secret_token": "leak-me-please"}
    fake_record = types.SimpleNamespace(**fields)

    view = records.to_user_view(fake_record)
    dumped = json.dumps(view, ensure_ascii=False)
    assert "raw_api_response" not in dumped
    assert "secret_token" not in dumped
    assert "leak-me-please" not in dumped


def test_user_view_shows_only_the_shown_horses_notes():
    """n=1 で3頭ぶんの注記を持つ記録を渡すと、to_user_view の出力には
    表示した馬（1頭）の注記しか現れない（fix4 決定3。前ラウンドの欠陥
    ――「3頭ぶんの繰り返し」を「12頭ぶんの繰り返し」に付け替えただけ――
    の再発防止）。
    """
    # tie_group を別々にする――この3頭が「同程度」に丸められると、n=1 でも
    # 組の最後まで表示対象が広がり (決定4)、このテストが見たい「表示しない
    # 馬の注記が漏れないこと」そのものを確かめられなくなる。
    horse_a = _make_horse("1", rank_baseline=1, rank_model=1, tie_group=1,
                           notes=("01番の注記",))
    horse_b = _make_horse("2", rank_baseline=2, rank_model=2, tie_group=2,
                           notes=("02番の注記",))
    horse_c = _make_horse("3", rank_baseline=3, rank_model=3, tie_group=3,
                           notes=("03番の注記",))
    record = dataclasses.replace(
        _make_record(num_horses=3, n=1), horses=(horse_a, horse_b, horse_c))

    view = records.to_user_view(record)

    assert len(view["予想"]) == 1
    assert view["予想"][0]["馬番"] == "1"
    assert view["予想"][0]["注記"] == ["01番の注記"]
    dumped = json.dumps(view, ensure_ascii=False)
    assert "02番の注記" not in dumped
    assert "03番の注記" not in dumped


def test_user_view_keeps_race_notes_at_the_top():
    """レース単位の注記は top-level の "注記" に出る
    （data_quality["notes"] はレース単位専用。決定1）。馬に注記が無くても
    top-level の注記は消えない。"""
    record = dataclasses.replace(
        _make_record(num_horses=1, n=1),
        data_quality={"notes": ["同馬場状態の集計は行っていない（...）。"],
                       "horse_notes": {}, "tie_band_scale": "layer1_win_prob"},
    )

    view = records.to_user_view(record)

    assert view["注記"] == ["同馬場状態の集計は行っていない（...）。"]
    assert view["予想"][0]["注記"] == []


# --- record_id / status --------------------------------------------------------


def test_record_id_is_unique_per_save(tmp_path):
    r1 = _make_record(record_id=uuid.uuid4().hex, predicted_at=datetime(2026, 3, 21, 10, 0, 0))
    r2 = _make_record(record_id=uuid.uuid4().hex, predicted_at=datetime(2026, 3, 21, 11, 0, 0))
    assert r1.record_id != r2.record_id

    records.save(r1, data_dir=tmp_path)
    records.save(r2, data_dir=tmp_path)
    loaded = records.load_month("202603", data_dir=tmp_path)

    assert len(loaded) == 2
    loaded_ids = {r.record_id for r in loaded}
    assert loaded_ids == {r1.record_id, r2.record_id}


def test_status_failed_is_saved(tmp_path):
    """失敗した予想も痕跡を残せること (spec §12.7)。horses が空でも保存できる。"""
    record = dataclasses.replace(
        _make_record(status="FAILED", num_horses=0, n=0),
        horses=(),
        data_quality={"failure_reason": "第2層が3頭ぶん失敗しました"},
    )
    records.save(record, data_dir=tmp_path)
    [loaded] = records.load_month("202603", data_dir=tmp_path)
    assert loaded.status == "FAILED"
    assert loaded.horses == ()
    assert loaded.data_quality == {"failure_reason": "第2層が3頭ぶん失敗しました"}


# --- fix5 決定1: 同程度の判定を行っていない実行 ----------------------------------


def test_user_view_says_when_ties_were_not_judged():
    """tie_band_scale が "not_measured" のとき、その1行を出す。

    重みを使った実行では model 尺度のばらつきを測っていないので同程度の
    判定を行わない。利用者に「組が1つも無い」理由が伝わらないと、
    同程度の説明が消えたことを精度の主張と読み違える。
    """
    record = _make_record(num_horses=5, n=3, tie_band_scale="not_measured")

    view = records.to_user_view(record)

    assert view["同程度の判定"] == records.TIE_NOT_MEASURED_NOTE
    assert "測っていない" in view["同程度の判定"]


def test_user_view_omits_the_tie_note_when_ties_were_judged():
    """第1層の尺度で判定した実行では、その行そのものを出さない（キーが無い）。"""
    record = _make_record(num_horses=5, n=3, tie_band_scale="layer1_win_prob")

    view = records.to_user_view(record)

    assert "同程度の判定" not in view

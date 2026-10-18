from __future__ import annotations

from keiba.features import dossier_markdown


def _dossier():
    run = {"date": "2026-09-06", "days_before": 14, "jyo": "中山", "track_cd": "18",
           "track_label": "芝・右外", "track_kind": "turf", "kyori_m": 2000, "baba": "良",
           "class_level": 3, "syusso_tosu": 12, "kakutei_jyuni": 2, "margin_sec": 0.1,
           "corner_positions": [None, None, 3, 3], "last_3f_sec": 34.1, "futan_kg": 57.0}
    cond = {"starts": 1, "wins": 0, "top3": 1, "margin_sec_mean": 0.1,
            "period": {"from": "2026-09-06", "to": "2026-09-06"}, "inapplicable_reason": None}
    undefined = dict(cond, starts=None, wins=None, top3=None, margin_sec_mean=None,
                     period={"from": None, "to": None},
                     inapplicable_reason="今回の馬場状態が未入力のため比較できない")
    horse = {"bamei": "テストウマ", "sex_age": "牡4", "futan_kg": 57.0, "waku": "1",
             "kisyu": None,
             "history": {"status": "found",
                         "runs_found": 1, "period": {"from": "2026-09-06", "to": "2026-09-06"}},
             "recent_runs": [run],
             "vs_last_run": {"days_since_last_run": 14, "distance_change_m": 0,
                             "futan_change_kg": 0.0, "class_level_change": 0},
             "conditions": {"surface_exact_distance": cond, "surface_same_going": undefined},
             "ability": {}, "running_style": {"corner_pos_mean": 0.25}}
    return {
        "dossier_version": "2", "prediction_cutoff": "2026-09-19T21:25:00+09:00",
        "course": {"identified": False, "reason": "トラックコードが無い", "facts": None,
                   "track_label": None, "kyori_m": 2000, "start_note": None,
                   "facts_source": None, "course_kubun_cd": None},
        "race": {"jyo": "中山", "track_kind": "turf", "kyori_m": 2000, "baba": None,
                 "class_name": None, "class_level": 3, "entry_count": 1},
        "horses": {"01": horse},
        "condition_definitions": {"surface_exact_distance": "芝・ダートと距離が今回と同じ過去走",
                                  "surface_same_going": "芝・ダートと馬場状態が今回と同じ過去走"},
        "history_status_notes": {"found": "締切前の出走記録がデータ版にある。"},
        "field_summary": {"n_horses": 1, "n_with_history": 1,
                          "history_status_counts": {"found": 1}, "n_front_runners": 1,
                          "front_runner_definition": "定義"},
        "data_quality": {"data_version": "v1", "coverage_from": None, "notes": [],
                         "horse_notes": {}},
    }


def test_render_has_sections_in_order():
    text = dossier_markdown.render(_dossier())
    order = ["## コース", "## 今回の条件", "## 出走馬", "## データの充足状況"]
    positions = [text.index(h) for h in order]
    assert positions == sorted(positions)


def test_render_says_course_unidentified_with_reason():
    assert "トラックコードが無い" in dossier_markdown.render(_dossier())


def test_render_shows_missing_as_dash_not_zero():
    text = dossier_markdown.render(_dossier())
    assert "今回の馬場状態が未入力のため比較できない" in text
    assert "| 馬場 | — |" in text or "馬場: —" in text
    # 定義できない条件の行は、出走・勝・3着内・平均着差がすべて「—」（0 ではない）。
    assert ("| surface_same_going | — | — | — | — | "
            "今回の馬場状態が未入力のため比較できない |") in text.splitlines()
    # 0戦は 0 のまま出す（— と区別できる）。
    d = _dossier()
    d["horses"]["01"]["conditions"]["surface_exact_distance"] = dict(
        d["horses"]["01"]["conditions"]["surface_exact_distance"],
        starts=0, wins=0, top3=0, margin_sec_mean=None)
    assert "| surface_exact_distance | 0 | 0 | 0 | — | — |" in dossier_markdown.render(d).splitlines()


def test_render_uses_japanese_labels_for_course_facts():
    # course_facts.json の実際のキー（英字）を生のまま出さない。
    d = _dossier()
    d["course"] = {
        "identified": True, "reason": None, "jyo_cd": "01", "track_cd": "17",
        "track_label": "芝・右", "direction": "right", "kyori_m": 1800,
        "course_kubun_cd": None,
        "facts": {"straight_m": 266.1, "circumference_m": 1640.9, "width_m": 25,
                  "elevation_note": "高低差0.7m", "layout": "内回り"},
        "start_note": None,
        "facts_source": {"url": "https://example.com", "retrieved_at": "2026-09-23",
                         "valid_from": "2000-01-01", "valid_to": None, "rail_basis": None},
    }
    text = dossier_markdown.render(d)
    assert "- 直線距離: 266.1m" in text
    assert "- 1周の距離: 1640.9m" in text
    assert "- 幅員: 25m" in text
    assert "- 高低差: 高低差0.7m" in text
    assert "- コースの形: 内回り" in text
    for raw in ("straight_m", "circumference_m", "width_m", "elevation_note", "layout"):
        assert raw not in text


def test_render_recent_runs_header_is_japanese():
    text = dossier_markdown.render(_dossier())
    assert "| 日付 | 競馬場 | コース | 距離(m) |" in text
    assert "track_label" not in text
    assert "last_3f_sec" not in text


def test_render_history_breakdown_is_not_a_dict_repr():
    d = _dossier()
    d["field_summary"]["history_status_counts"] = {"found": 1, "unknown": 2}
    text = dossier_markdown.render(d)
    assert "- 履歴の内訳: found 1、unknown 2" in text
    assert "{" not in text


def test_render_shows_what_jev_receives_per_horse():
    # 前走との比較・能力・脚質・計算できなかった項目は Jev に送っている。
    # 人が見る .md からも読めるようにする。
    d = _dossier()
    d["horses"]["01"]["vs_last_run"] = {"days_since_last_run": 14, "distance_change_m": -200,
                                        "futan_change_kg": 1.0, "class_level_change": None}
    d["horses"]["01"]["ability"] = {"recent_margin_sec_wavg": 0.35, "runs_used": 3,
                                    "recent_class_level_wavg": 3.5, "best_class_level": 4,
                                    "field_rank": 2.0,
                                    "margin_period": {"from": "2026-08-01", "to": "2026-09-06"}}
    d["horses"]["01"]["running_style"] = {"corner_pos_mean": 0.25, "corner_pos_stdev": None,
                                          "corner_pos_starts": 3, "corner_source": "jyuni_3c"}
    d["data_quality"]["missing"] = {"01": ["transport"]}
    text = dossier_markdown.render(d)
    assert "前走からの日数 14" in text
    assert "距離の変化(m) -200" in text
    assert "クラス順序値の変化 —" in text
    assert "近走の着差の加重平均(秒) 0.35" in text
    assert "着差の集計期間 2026-08-01〜2026-09-06" in text
    assert "使ったコーナー jyuni_3c" in text
    assert "- 計算できなかった項目: transport" in text


def test_render_lists_recent_run():
    text = dossier_markdown.render(_dossier())
    assert "2026-09-06" in text and "芝・右外" in text


def test_render_shows_condition_definitions_once():
    text = dossier_markdown.render(_dossier())
    assert text.count("芝・ダートと距離が今回と同じ過去走") == 1
    assert text.count("芝・ダートと馬場状態が今回と同じ過去走") == 1


def test_render_shows_history_note_per_horse_from_top_level_map():
    text = dossier_markdown.render(_dossier())
    assert "締切前の出走記録がデータ版にある。" in text


def test_render_shows_rail_basis_in_source_when_present():
    d = _dossier()
    d["course"] = {
        "identified": True, "reason": None, "jyo_cd": "04", "track_cd": "18",
        "track_label": "芝・右外", "direction": "右", "kyori_m": 2000,
        "course_kubun_cd": "B",
        "facts": {"直線距離": "353m", "高低差": "2.2m"},
        "start_note": "スタンド前発走",
        "facts_source": {"url": "https://example.com/course", "retrieved_at": "2026-09-01",
                         "valid_from": "2026-01-01", "valid_to": None, "rail_basis": "Aコース"},
    }
    text = dossier_markdown.render(d)
    assert "Aコース" in text

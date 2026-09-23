from __future__ import annotations

from keiba.features import dossier_markdown


def _dossier():
    run = {"date": "2026-09-06", "days_before": 14, "jyo": "中山", "track_cd": "18",
           "track_label": "芝・右外", "track_kind": "turf", "kyori_m": 2000, "baba": "良",
           "class_level": 3, "syusso_tosu": 12, "kakutei_jyuni": 2, "margin_sec": 0.1,
           "corner_positions": [None, None, 3, 3], "last_3f_sec": 34.1, "futan_kg": 57.0}
    cond = {"definition": "定義", "starts": 1, "wins": 0, "top3": 1, "margin_sec_mean": 0.1,
            "period": {"from": "2026-09-06", "to": "2026-09-06"}, "inapplicable_reason": None}
    undefined = dict(cond, starts=None, wins=None, top3=None,
                     inapplicable_reason="今回の馬場状態が未入力のため比較できない")
    horse = {"bamei": "テストウマ", "sex_age": "牡4", "futan_kg": 57.0, "waku": "1",
             "kisyu": None,
             "history": {"status": "found", "note": "締切前の出走記録がデータ版にある。",
                         "runs_found": 1, "period": {"from": "2026-09-06", "to": "2026-09-06"}},
             "recent_runs": [run],
             "vs_last_run": {"days_since_last_run": 14, "distance_change_m": 0,
                             "futan_change_kg": 0.0, "class_level_change": 0},
             "conditions": {"surface_exact_distance": cond, "surface_same_going": undefined},
             "ability": {}, "running_style": {"corner_pos_mean": 0.25}}
    return {
        "dossier_version": "1", "prediction_cutoff": "2026-09-19T21:25:00+09:00",
        "course": {"identified": False, "reason": "トラックコードが無い", "facts": None,
                   "track_label": None, "kyori_m": 2000, "start_note": None,
                   "facts_source": None, "course_kubun_cd": None},
        "race": {"jyo": "中山", "track_kind": "turf", "kyori_m": 2000, "baba": None,
                 "class_name": None, "class_level": 3, "entry_count": 1},
        "horses": {"01": horse},
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


def test_render_lists_recent_run():
    text = dossier_markdown.render(_dossier())
    assert "2026-09-06" in text and "芝・右外" in text


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

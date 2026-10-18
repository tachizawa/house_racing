from __future__ import annotations

from datetime import date

import pytest

from keiba.features import course

ALLOWED_FACT_KEYS = {"layout", "straight_m", "circumference_m", "width_m",
                     "elevation_note", "corner_count"}


def _ident(jyo_cd="04", track_cd="11", kyori_m=2000, identified=True):
    return course.CourseIdentification(
        identified=identified, jyo_cd=jyo_cd, track_cd=track_cd if identified else None,
        kyori_m=kyori_m, track=course.decode_track_cd(track_cd) if identified else None,
        course_kubun_cd=None, reason=None if identified else "x")


TABLE = [
    {"jyo_cd": "04", "track_cd": "11", "valid_from": "2000-01-01", "valid_to": "2019-12-31",
     "facts": {"layout": "旧"}, "starts": {},
     "source": {"url": "https://www.jra.go.jp/a", "retrieved_at": "2026-09-23"}},
    {"jyo_cd": "04", "track_cd": "11", "valid_from": "2020-01-01", "valid_to": None,
     "facts": {"layout": "内回り", "straight_m": 359},
     "starts": {"2000": "スタンド前"},
     "source": {"url": "https://www.jra.go.jp/b", "retrieved_at": "2026-09-23"}},
    {"jyo_cd": "09", "track_cd": "17", "valid_from": "2000-01-01", "valid_to": None,
     "rail_basis": "Aコース", "facts": {"straight_m": 356.5}, "starts": {},
     "source": {"url": "https://www.jra.go.jp/c", "retrieved_at": "2026-09-23"}},
]


def test_facts_for_picks_entry_valid_on_date():
    facts = course.facts_for(_ident(), date(2026, 8, 1), table=TABLE)
    assert facts.facts == {"layout": "内回り", "straight_m": 359}
    assert facts.start_note == "スタンド前"
    assert facts.source_url == "https://www.jra.go.jp/b"


def test_facts_for_respects_valid_to():
    facts = course.facts_for(_ident(), date(2019, 6, 1), table=TABLE)
    assert facts.facts == {"layout": "旧"}


def test_facts_for_unknown_distance_has_no_start_note():
    assert course.facts_for(_ident(kyori_m=1800), date(2026, 8, 1), table=TABLE).start_note is None


def test_facts_for_unidentified_course_is_none():
    assert course.facts_for(_ident(identified=False), date(2026, 8, 1), table=TABLE) is None


def test_facts_for_missing_entry_is_none():
    assert course.facts_for(_ident(jyo_cd="05"), date(2026, 8, 1), table=TABLE) is None


def test_facts_for_reports_rail_basis_when_present():
    facts = course.facts_for(_ident(jyo_cd="09", track_cd="17"), date(2026, 8, 1), table=TABLE)
    assert facts.rail_basis == "Aコース"


def test_facts_for_rail_basis_is_none_when_absent():
    facts = course.facts_for(_ident(), date(2026, 8, 1), table=TABLE)
    assert facts.rail_basis is None


# --- 同梱の表そのものの検査 ------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    return course.load_facts_table()


def test_shipped_table_every_entry_has_official_source(shipped):
    assert shipped, "course_facts.json が空"
    for entry in shipped:
        assert entry["source"]["url"].startswith("https://www.jra.go.jp/"), entry
        date.fromisoformat(entry["source"]["retrieved_at"])
        date.fromisoformat(entry["valid_from"])


def test_shipped_table_uses_only_allowed_fact_keys(shipped):
    for entry in shipped:
        assert set(entry["facts"]) <= ALLOWED_FACT_KEYS, entry
        assert None not in entry["facts"].values(), "無い項目は null ではなく省く"


def test_shipped_table_has_no_overlapping_validity(shipped):
    by_key: dict[tuple, list] = {}
    for entry in shipped:
        by_key.setdefault((entry["jyo_cd"], entry["track_cd"]), []).append(entry)
    for key, entries in by_key.items():
        spans = sorted((date.fromisoformat(e["valid_from"]),
                        date.fromisoformat(e["valid_to"]) if e["valid_to"] else date.max)
                       for e in entries)
        for (a_from, a_to), (b_from, _) in zip(spans, spans[1:]):
            assert a_to < b_from, f"{key} の適用期間が重なっている"


def test_shipped_table_track_codes_are_decodable(shipped):
    for entry in shipped:
        assert course.decode_track_cd(entry["track_cd"]) is not None, entry

"""schema.parse_race_input / validate.check の振る舞いを確認する。

parse は「理解できるか」（理解できなければ InputError）、
check は「意味的に妥当か」（妥当でなければ Problem を集めて返す、raise しない）
の分担を確認する。
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from keiba import config
from keiba.input import schema, validate
from keiba.models import HorseEntry, RaceInput


def make_payload(**overrides) -> dict:
    """必須項目だけを持つ最小の payload。"""
    payload = {
        "date": "2026-03-21",
        "jyo": "中山",
        "race_num": 11,
        "track_kind": "turf",
        "kyori_m": 2000,
        "entry_count": 2,
        "prediction_cutoff": "2026-03-20T18:00:00",
        "horses": [
            {"umaban": "1", "bamei": "テスト馬"},
            {"umaban": "2", "bamei": "サンプル馬"},
        ],
    }
    payload.update(overrides)
    return payload


def make_race_input(**overrides) -> RaceInput:
    """check() 向けの、それ自体は妥当な RaceInput。top_n は頭数(2)に合わせてある。"""
    base = dict(
        date=date(2026, 3, 21),
        jyo="中山",
        race_num=11,
        track_kind="turf",
        kyori_m=2000,
        entry_count=2,
        prediction_cutoff=datetime(2026, 3, 20, 18, 0),
        horses=(
            HorseEntry(umaban="1", bamei="テスト馬"),
            HorseEntry(umaban="2", bamei="サンプル馬"),
        ),
        top_n=2,
    )
    base.update(overrides)
    return RaceInput(**base)


# --- schema.parse_race_input -------------------------------------------------

class TestParseRaceInput:
    def test_parse_minimal_payload(self):
        race_input = schema.parse_race_input(make_payload())
        assert isinstance(race_input, RaceInput)
        assert race_input.top_n == config.DEFAULT_TOP_N

    def test_parse_reads_iso_datetimes(self):
        race_input = schema.parse_race_input(make_payload(hasso_at="2026-03-21T15:25:00"))
        assert race_input.prediction_cutoff == datetime(2026, 3, 20, 18, 0)
        assert isinstance(race_input.prediction_cutoff, datetime)
        assert race_input.hasso_at == datetime(2026, 3, 21, 15, 25)
        assert isinstance(race_input.hasso_at, datetime)

    def test_parse_rejects_missing_required_field(self):
        payload = make_payload()
        del payload["jyo"]
        with pytest.raises(schema.InputError, match="jyo"):
            schema.parse_race_input(payload)

    def test_parse_keeps_umaban_as_string_with_leading_zero(self):
        payload = make_payload(
            horses=[
                {"umaban": "01", "bamei": "テスト馬"},
                {"umaban": "02", "bamei": "サンプル馬"},
            ]
        )
        race_input = schema.parse_race_input(payload)
        assert race_input.horses[0].umaban == "01"
        assert race_input.horses[1].umaban == "02"

    def test_parse_rejects_unknown_track_kind(self):
        with pytest.raises(schema.InputError):
            schema.parse_race_input(make_payload(track_kind="芝"))

    def test_parse_resolves_jyo_name_to_code(self):
        race_input = schema.parse_race_input(make_payload(jyo="中山"))
        assert race_input.jyo_cd == "06"
        assert race_input.jyo == "中山"

    def test_parse_accepts_jyo_code_directly(self):
        race_input = schema.parse_race_input(make_payload(jyo="06"))
        assert race_input.jyo_cd == "06"
        assert race_input.jyo == "中山"

    def test_parse_rejects_unknown_jyo(self):
        with pytest.raises(schema.InputError):
            schema.parse_race_input(make_payload(jyo="大井"))

    def test_parse_resolves_baba_name_to_code(self):
        race_input = schema.parse_race_input(make_payload(baba="稍重"))
        assert race_input.baba_cd == "2"
        assert race_input.baba == "稍重"

    def test_parse_keeps_baba_none(self):
        race_input = schema.parse_race_input(make_payload())
        assert race_input.baba_cd is None
        assert race_input.baba is None

    def test_parse_rejects_unknown_baba(self):
        with pytest.raises(schema.InputError):
            schema.parse_race_input(make_payload(baba="やや重"))


class TestLoadRaceInput:
    def test_load_reads_json_file(self, tmp_path: Path):
        path = tmp_path / "race.json"
        path.write_text(json.dumps(make_payload(), ensure_ascii=False), encoding="utf-8")
        race_input = schema.load_race_input(path)
        assert race_input.jyo == "中山"
        assert race_input.jyo_cd == "06"

    def test_load_rejects_invalid_json(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(schema.InputError):
            schema.load_race_input(path)


class TestResolveJyoCd:
    def test_resolves_name_to_code(self):
        assert schema.resolve_jyo_cd("中山") == "06"

    def test_accepts_code_directly(self):
        assert schema.resolve_jyo_cd("06") == "06"

    def test_rejects_local_course(self):
        with pytest.raises(schema.InputError):
            schema.resolve_jyo_cd("大井")


class TestResolveBabaCd:
    def test_resolves_name_to_code(self):
        assert schema.resolve_baba_cd("稍重") == "2"

    def test_passes_none_through(self):
        assert schema.resolve_baba_cd(None) is None

    def test_rejects_near_miss(self):
        with pytest.raises(schema.InputError):
            schema.resolve_baba_cd("やや重")


# --- validate.Problem ---------------------------------------------------------

class TestProblem:
    def test_is_frozen(self):
        problem = validate.Problem(field="jyo", message="test")
        with pytest.raises(Exception):
            problem.field = "other"


# --- validate.check ------------------------------------------------------------

class TestCheck:
    def test_check_passes_for_valid_input(self):
        assert validate.check(make_race_input()) == []

    def test_check_detects_duplicate_umaban(self):
        race_input = make_race_input(
            horses=(
                HorseEntry(umaban="1", bamei="テスト馬"),
                HorseEntry(umaban="1", bamei="サンプル馬"),
            ),
        )
        problems = validate.check(race_input)
        assert any(p.field == "horses" for p in problems)

    def test_check_detects_entry_count_mismatch(self):
        race_input = make_race_input(entry_count=12)
        problems = validate.check(race_input)
        assert any(p.field == "entry_count" for p in problems)

    @pytest.mark.parametrize("kyori_m", [0, -1])
    def test_check_detects_non_positive_distance(self, kyori_m):
        race_input = make_race_input(kyori_m=kyori_m)
        problems = validate.check(race_input)
        assert any(p.field == "kyori_m" for p in problems)

    @pytest.mark.parametrize("kyori_m", [100, 10000])
    def test_check_detects_unrealistic_distance(self, kyori_m):
        race_input = make_race_input(kyori_m=kyori_m)
        problems = validate.check(race_input)
        assert any(p.field == "kyori_m" for p in problems)

    def test_check_detects_cutoff_after_start(self):
        moment = datetime(2026, 3, 21, 15, 25)
        race_input = make_race_input(prediction_cutoff=moment, hasso_at=moment)
        problems = validate.check(race_input)
        assert any(p.field == "prediction_cutoff" for p in problems)

    def test_check_allows_missing_optional_fields(self):
        race_input = make_race_input(
            baba=None,
            class_name=None,
            horses=(
                HorseEntry(umaban="1", bamei="テスト馬", wakuban=None, futan_kg=None),
                HorseEntry(umaban="2", bamei="サンプル馬", wakuban=None, futan_kg=None),
            ),
        )
        assert validate.check(race_input) == []

    def test_check_detects_empty_horse_name(self):
        race_input = make_race_input(
            horses=(
                HorseEntry(umaban="1", bamei=""),
                HorseEntry(umaban="2", bamei="サンプル馬"),
            ),
        )
        problems = validate.check(race_input)
        assert any(p.field == "horses" for p in problems)

    @pytest.mark.parametrize("top_n", [0, 3])
    def test_check_detects_top_n_out_of_range(self, top_n):
        race_input = make_race_input(top_n=top_n)
        problems = validate.check(race_input)
        assert any(p.field == "top_n" for p in problems)

    def test_check_reports_all_problems_at_once(self):
        race_input = make_race_input(
            entry_count=99,
            horses=(
                HorseEntry(umaban="1", bamei="テスト馬"),
                HorseEntry(umaban="1", bamei="サンプル馬"),
            ),
        )
        problems = validate.check(race_input)
        assert len(problems) == 2
        fields = {p.field for p in problems}
        assert fields == {"entry_count", "horses"}

    @pytest.mark.parametrize("umaban", ["0", "-1", "abc"])
    def test_check_detects_invalid_umaban(self, umaban):
        race_input = make_race_input(
            horses=(
                HorseEntry(umaban=umaban, bamei="テスト馬"),
                HorseEntry(umaban="2", bamei="サンプル馬"),
            ),
        )
        problems = validate.check(race_input)
        assert any(p.field == "horses" for p in problems)

    def test_check_allows_leading_zero_umaban(self):
        race_input = make_race_input(
            horses=(
                HorseEntry(umaban="01", bamei="テスト馬"),
                HorseEntry(umaban="02", bamei="サンプル馬"),
            ),
        )
        assert validate.check(race_input) == []

    def test_check_detects_blank_jyo(self):
        race_input = make_race_input(jyo="")
        problems = validate.check(race_input)
        assert any(p.field == "jyo" for p in problems)

    def test_check_detects_blank_track_kind(self):
        race_input = make_race_input(track_kind="")
        problems = validate.check(race_input)
        assert any(p.field == "track_kind" for p in problems)

    def test_check_detects_no_horses(self):
        race_input = make_race_input(horses=(), entry_count=0, top_n=1)
        problems = validate.check(race_input)
        assert any(p.field == "horses" for p in problems)


# --- 型境界: 量は引用符付きの数値なら受け入れ、識別子は文字列のみ受け入れる -----

class TestNumericFieldTypes:
    @pytest.mark.parametrize("field", ["kyori_m", "entry_count", "race_num", "top_n"])
    def test_accepts_a_quoted_number(self, field):
        """数値を引用符付きで書いた JSON は意味が一意なので受け入れる。"""
        value = "2000" if field == "kyori_m" else "2"
        race_input = schema.parse_race_input(make_payload(**{field: value}))
        assert isinstance(getattr(race_input, field), int)
        assert getattr(race_input, field) == int(value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("kyori_m", "にせんめーとる"),
            ("kyori_m", 2000.5),
            ("kyori_m", True),
            ("kyori_m", None),
            ("top_n", "さん"),
            ("top_n", True),
            ("entry_count", []),
            ("race_num", "11R"),
        ],
    )
    def test_rejects_an_ambiguous_or_wrong_type(self, field, value):
        """切り捨ても bool の 1 扱いもしない。分からないものは InputError。"""
        with pytest.raises(schema.InputError, match=field):
            schema.parse_race_input(make_payload(**{field: value}))

    def test_rejects_a_bare_number_for_umaban(self):
        """馬番は識別子。1 を "1" に黙って直すと、他の全部と挙動が変わる。"""
        payload = make_payload(
            horses=[
                {"umaban": 1, "bamei": "テスト馬"},
                {"umaban": "2", "bamei": "サンプル馬"},
            ]
        )
        with pytest.raises(schema.InputError, match="umaban"):
            schema.parse_race_input(payload)

    def test_rejects_a_bare_number_for_bamei(self):
        """馬名も識別子。文字列以外は黙って変換しない。"""
        payload = make_payload(
            horses=[
                {"umaban": "1", "bamei": 123},
                {"umaban": "2", "bamei": "サンプル馬"},
            ]
        )
        with pytest.raises(schema.InputError, match="bamei"):
            schema.parse_race_input(payload)

    def test_rejects_a_bare_number_for_jyo(self):
        """jyo も識別子。コード "06" のつもりで 6 と書いても解決しない。"""
        payload = make_payload(jyo=6)
        with pytest.raises(schema.InputError, match="jyo"):
            schema.parse_race_input(payload)

    def test_rejects_a_bare_number_for_baba(self):
        """baba も jyo と同じ構造の識別子。1 を "1" と解釈してあげない。"""
        payload = make_payload(baba=1)
        with pytest.raises(schema.InputError, match="baba"):
            schema.parse_race_input(payload)

    def test_baba_none_still_passes_through(self):
        """baba は任意項目。None はそのまま None（文字列必須の対象外）。"""
        race_input = schema.parse_race_input(make_payload())
        assert race_input.baba is None
        assert race_input.baba_cd is None

    def test_accepts_futan_kg_as_quoted_number(self):
        payload = make_payload(
            horses=[
                {"umaban": "1", "bamei": "テスト馬", "futan_kg": "57.0"},
                {"umaban": "2", "bamei": "サンプル馬"},
            ]
        )
        race_input = schema.parse_race_input(payload)
        assert race_input.horses[0].futan_kg == 57.0
        assert isinstance(race_input.horses[0].futan_kg, float)

    def test_rejects_bool_for_futan_kg(self):
        payload = make_payload(
            horses=[
                {"umaban": "1", "bamei": "テスト馬", "futan_kg": True},
                {"umaban": "2", "bamei": "サンプル馬"},
            ]
        )
        with pytest.raises(schema.InputError, match="futan_kg"):
            schema.parse_race_input(payload)


class TestCheckNeverRaises:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("kyori_m", "2000"),
            ("top_n", "3"),
            ("entry_count", "2"),
            ("kyori_m", None),
            ("top_n", None),
        ],
    )
    def test_check_returns_problems_rather_than_raising(self, field, value):
        """schema を通さず RaceInput を直接組んだ場合でも check() は落ちない。

        check() の契約は「問題を並べて返す」。呼び出し側が例外を捕まえずに
        済むことが前提なので、どんな RaceInput を渡されても raise しない。
        """
        race_input = dataclasses.replace(make_race_input(), **{field: value})
        problems = validate.check(race_input)  # raise しないこと自体が検証
        assert any(p.field == field for p in problems)


# --- check() の契約は「どんな RaceInput でも問題を並べて返す」こと -----------

_HOSTILE_VALUES = (None, "文字列", 0, -1, 1.5, True, [], {}, object())


class TestCheckIsTotal:
    """check() は「どんな RaceInput でも問題を並べて返す」契約。

    型ごとにガードを足していく方式では、項目が増えたときに漏れる。
    全項目 × 敵対的な値を機械的に回して、raise しないことそのものを検証する。
    RaceInput / HorseEntry にフィールドが増えれば、このテストは自動的に
    そのフィールドも対象に含める。
    """

    @pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(RaceInput)])
    @pytest.mark.parametrize("value", _HOSTILE_VALUES)
    def test_never_raises_for_any_race_input_field(self, field, value):
        race_input = dataclasses.replace(make_race_input(), **{field: value})
        problems = validate.check(race_input)
        assert isinstance(problems, list)
        assert all(isinstance(p, validate.Problem) for p in problems)

    @pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(HorseEntry)])
    @pytest.mark.parametrize("value", _HOSTILE_VALUES)
    def test_never_raises_for_any_horse_entry_field(self, field, value):
        base = make_race_input()
        broken = dataclasses.replace(base.horses[0], **{field: value})
        race_input = dataclasses.replace(base, horses=(broken,) + base.horses[1:])
        problems = validate.check(race_input)
        assert isinstance(problems, list)
        assert all(isinstance(p, validate.Problem) for p in problems)

"""update_data CLI の配線。COM は叩かず、jvlink.open_link を偽の link に差し替える。

scripts/ は keiba パッケージの外にあり pytest.ini の pythonpath にも入っていないため、
update_data.py 自身が行っているのと同じ方法（sys.path への追加）で import する。

fix round 4 で main() は jvlink.read_records() を呼ばなくなり、代わりに
jvlink.open_dataspec() / jvlink.read_open_dataspec() / link.JVClose() を
dataspec ごとに自分で呼ぶようになった。そのため偽物は jvlink 側の関数では
なく open_link() が返す link オブジェクトそのものを差し替える。
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import update_data  # noqa: E402

from keiba.ingest import store  # noqa: E402
from tests.fixtures import records as fx  # noqa: E402


class DataspecAwareLink:
    """dataspec ごとに JVOpen の結果と JVRead で返すレコードを変えられる偽 link。

    update_data.py は dataspec ごとに open_dataspec() / read_open_dataspec() /
    JVClose() を呼ぶので、単一の open_code しか持てない偽物では「DIFF は -1、
    RACE は 0」のような組み合わせを表現できない。COM には一切触れない。
    """

    def __init__(self, *, open_codes: dict, records_by_dataspec: dict | None = None,
                 download_counts: dict | None = None):
        self._open_codes = open_codes
        self._records = {k: list(v) for k, v in (records_by_dataspec or {}).items()}
        self._download_counts = download_counts or {}
        self._pending: list = []
        self._current_download_count = 0
        self.close_calls = 0

    def JVOpen(self, dataspec, fromdate, option, _a, _b, _c):
        code = self._open_codes.get(dataspec, 0)
        self._current_download_count = self._download_counts.get(dataspec, 0)
        self._pending = list(self._records.get(dataspec, []))
        return (code, 0, self._current_download_count, "")

    def JVStatus(self):
        return self._current_download_count

    def JVRead(self, buf, size, fname):
        if not self._pending:
            return (0, "", size, "")
        payload = self._pending.pop(0)
        return (len(payload), payload.decode("cp932"), size, "")

    def JVClose(self):
        self.close_calls += 1


def test_aborts_the_writer_when_reading_fails(tmp_path, monkeypatch):
    """読み取りが途中で落ちたら .partial を残さず、CURRENT も動かさない。

    store.commit() の後始末は同じ version_id で書き直したときしか効かない。
    時刻違いの version_id を持つ .partial は誰にも片付けられずに溜まり続けるので、
    読み取りが例外を投げたらここで abort() し、開いたファイルハンドルと
    ディレクトリの両方を後始末する。
    """
    class ExplodingLink(DataspecAwareLink):
        def JVRead(self, buf, size, fname):
            if self._pending:
                return super().JVRead(buf, size, fname)
            raise RuntimeError("ダウンロード中に切断")

    link = ExplodingLink(
        open_codes={"RACE": 0},
        records_by_dataspec={"RACE": [fx.make_ra()]},
    )
    monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

    with pytest.raises(RuntimeError, match="切断"):
        update_data.main([
            "--dataspec", "RACE", "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        ])

    versions = tmp_path / "versions"
    assert not versions.exists() or list(versions.glob("*.partial")) == []
    assert store.current_version(tmp_path) is None


def test_aborts_the_writer_when_commit_fails(tmp_path, monkeypatch):
    """commit() が落ちても .partial を残さない。

    読み取りループだけを守っても、commit() 自身が失敗する経路（manifest 書き込み
    中のディスクフル、rename の失敗、DataVersionError など）で同じリークが起きる。
    読み取りは正常に数件返す偽物にし、commit() の方を壊す。
    """
    link = DataspecAwareLink(
        open_codes={"RACE": 0},
        records_by_dataspec={"RACE": [fx.make_ra(), fx.make_se(), fx.make_um()]},
    )
    monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)
    monkeypatch.setattr(store.VersionWriter, "commit",
                         lambda self: (_ for _ in ()).throw(OSError("ディスク満杯")))

    with pytest.raises(OSError, match="ディスク満杯"):
        update_data.main([
            "--dataspec", "RACE", "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        ])

    versions = tmp_path / "versions"
    assert not versions.exists() or list(versions.glob("*.partial")) == []
    assert store.current_version(tmp_path) is None


def test_dry_run_does_not_touch_current(tmp_path, monkeypatch):
    """成功時の --dry-run は abort() され、CURRENT は書かれない。"""
    link = DataspecAwareLink(
        open_codes={"RACE": 0},
        records_by_dataspec={"RACE": [fx.make_ra(), fx.make_se(), fx.make_um()]},
    )
    monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

    exit_code = update_data.main([
        "--dataspec", "RACE", "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        "--dry-run",
    ])

    assert exit_code == 0
    assert store.current_version(tmp_path) is None
    versions = tmp_path / "versions"
    assert not versions.exists() or list(versions.iterdir()) == []


def test_verify_time_fields_exits_nonzero_on_mismatch(tmp_path, monkeypatch):
    """margin_sec の解釈が実データとずれていたら、黙って成功せず exit 1 にする。"""
    link = DataspecAwareLink(
        open_codes={"RACE": 0},
        records_by_dataspec={"RACE": [
            fx.make_se(kakutei_jyuni="01", time="1234", time_diff="-003", umaban="01"),
            # 実際のタイム差は 0.3 秒だが、記録上の margin は 0.9 秒にしてずらす。
            fx.make_se(kakutei_jyuni="02", time="1237", time_diff="+009", umaban="02"),
        ]},
    )
    monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

    exit_code = update_data.main([
        "--dataspec", "RACE", "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        "--verify-time-fields",
    ])

    assert exit_code == 1


class _RecordingWriter:
    def __init__(self):
        self.rows = []

    def add(self, key, row):
        self.rows.append(key)


def test_consume_without_se_rows_still_writes_se():
    """走破タイムを照合しないときは SE の行を溜めない（40年ぶんでは数GBになる）。"""
    from collections import Counter

    writer = _RecordingWriter()
    counts = Counter()
    update_data._consume(fx.make_se(), counts, writer, None)

    assert writer.rows == ["SE"]
    assert counts["SE"] == 1


def test_consume_collects_se_rows_when_given_a_list():
    from collections import Counter

    se_rows = []
    update_data._consume(fx.make_se(), Counter(), _RecordingWriter(), se_rows)

    assert len(se_rows) == 1


class TestDataspecArgument:
    """RACE だけでは UM が取れない。--dataspec は繰り返せて、既定は RACE + DIFF。

    COM には触れないので、jvlink を差し替えず _parse_args/_resolve_dataspecs を
    直接呼ぶだけで確かめられる。
    """

    def test_defaults_to_race_and_diff(self):
        args = update_data._parse_args(["--fromdate", "20260801000000"])
        assert update_data._resolve_dataspecs(args) == ("RACE", "DIFF")

    def test_repeated_flag_replaces_the_default(self):
        args = update_data._parse_args(
            ["--dataspec", "RACE", "--fromdate", "20260801000000"])
        assert update_data._resolve_dataspecs(args) == ["RACE"]

    def test_repeated_flag_accumulates(self):
        args = update_data._parse_args([
            "--dataspec", "RACE", "--dataspec", "TCOV", "--fromdate", "20260801000000",
        ])
        assert update_data._resolve_dataspecs(args) == ["RACE", "TCOV"]

    def test_an_explicit_empty_list_is_not_replaced_by_the_default(self):
        """argparse では作れない入力（[]）を直接渡して is not None の意味を固定する。

        action="append" は一度も指定されなければ必ず None を返し、[] を返す
        経路が無いので、argparse 経由のテストだけでは is not None を or に
        書き換えても全部通ってしまう（このガード自体が未検証になる）。
        空リストは「何も取らない」という明示的な指定なので、黙って既定に
        差し替えてはいけない。差し替えると VersionWriter の ValueError
        （「何も取らない版は版ではない」）に届かなくなる。
        """
        args = argparse.Namespace(dataspecs=[])
        assert update_data._resolve_dataspecs(args) == []

    def test_none_is_replaced_by_the_default(self):
        args = argparse.Namespace(dataspecs=None)
        assert update_data._resolve_dataspecs(args) == update_data.config.DEFAULT_DATASPECS


class TestOpenCodeReporting:
    """fix round 4: 「該当データなし」(-1) と「読んだが0件」を CLI が混同しないこと。

    蓄積系差分 (DIFF) は初回にセットアップ取得（--option 4）を走らせるまで
    空で、JVOpen は -1 を返す。これまでは (0件) としか表示されず、運用者が
    「今週は更新が無かった」のか「そもそも取得できていない」のか区別できな
    かった。
    """

    def test_reports_no_data_separately_from_zero_records(self, tmp_path, monkeypatch, capsys):
        link = DataspecAwareLink(open_codes={"RACE": 0, "DIFF": -1})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        exit_code = update_data.main([
            "--dataspec", "RACE", "--dataspec", "DIFF",
            "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        ])

        out = capsys.readouterr().out
        assert exit_code == 0
        race_section, diff_section = out.split("RACE:")[1].split("DIFF:")
        assert "該当データなし" in diff_section
        assert "(0件)" not in diff_section
        assert "(0件)" in race_section

    def test_hints_at_setup_only_for_the_differential_dataspec(self, tmp_path, monkeypatch, capsys):
        """fix round 5: ヒントは DIFF/DIFN のときだけ出て、かつ実測に基づく。

        実機検証の結果、--option 4 のセットアップ取得だけでは現役馬の大半を
        含む完全な馬マスタにはならない（実測: 生年は2021年までに偏り、収録馬
        の96%が抹消済み馬）ことが分かったので、ヒントは「これをやれば馬マス
        タが揃う」と誤読されない文言でなければならない。同時に「馬マスタが
        無くても・不完全でも予想は成立する」ことも伝える必要がある。
        """
        link = DataspecAwareLink(open_codes={"RACE": -1, "DIFF": -1})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        update_data.main([
            "--dataspec", "RACE", "--dataspec", "DIFF",
            "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        ])

        out = capsys.readouterr().out
        race_section, diff_section = out.split("RACE:")[1].split("DIFF:")

        # DIFF/DIFN のときだけ出る、という条件は変えない。
        assert "--option 4" not in race_section
        assert "--option 4" in diff_section

        # 数値が実測であることが分かる（将来変わりうる断定にしない）。
        assert "実測" in diff_section
        # 「揃う」という誤読を残さない: 大半が欠けること・抹消馬中心であることを言う。
        assert "抹消" in diff_section
        # 馬マスタが無くても・不完全でも予想が成立することを言う。
        assert "予想は成立" in diff_section

    def test_closes_the_link_for_every_dataspec(self, tmp_path, monkeypatch):
        link = DataspecAwareLink(open_codes={"RACE": 0, "DIFF": -1})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        update_data.main([
            "--dataspec", "RACE", "--dataspec", "DIFF",
            "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
        ])

        assert link.close_calls == 2

    def test_closes_the_link_when_reading_raises(self, tmp_path, monkeypatch):
        class ExplodingLink(DataspecAwareLink):
            def JVRead(self, buf, size, fname):
                raise RuntimeError("ダウンロード中に切断")

        link = ExplodingLink(open_codes={"RACE": 0})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        with pytest.raises(RuntimeError, match="切断"):
            update_data.main([
                "--dataspec", "RACE",
                "--fromdate", "20260801000000", "--data-dir", str(tmp_path),
            ])

        assert link.close_calls == 1
        assert store.current_version(tmp_path) is None


class TestBloodKind:
    """--kind blood は BLDN を読み、レースの版とは別の系統に保存する。"""

    def test_defaults_to_bldn(self):
        args = update_data._parse_args(["--kind", "blood", "--fromdate", "19860101000000"])
        assert update_data._resolve_dataspecs(args) == ("BLDN",)

    def test_default_data_dir_is_separate_from_races(self):
        race = update_data._parse_args(["--fromdate", "20260801000000"])
        blood = update_data._parse_args(["--kind", "blood", "--fromdate", "19860101000000"])
        assert update_data._resolve_data_dir(race) == update_data.config.DATA_DIR
        assert update_data._resolve_data_dir(blood) == update_data.config.BLOOD_DATA_DIR

    def test_writes_blood_files_without_touching_race_current(self, tmp_path, monkeypatch,
                                                              capsys):
        race_dir = tmp_path / "race"
        race_dir.mkdir()
        (race_dir / "CURRENT").write_text("20260924T115403Z", encoding="utf-8")
        blood_dir = tmp_path / "blood"
        link = DataspecAwareLink(
            open_codes={"BLDN": 0},
            records_by_dataspec={"BLDN": [fx.make_sk(), fx.make_hn(), fx.make_bt()]},
        )
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)

        exit_code = update_data.main([
            "--kind", "blood", "--fromdate", "19860101000000", "--option", "4",
            "--data-dir", str(blood_dir),
        ])

        assert exit_code == 0
        version = store.current_version(blood_dir)
        manifest = store.read_manifest(blood_dir / "versions" / version)
        assert manifest["record_counts"] == {"SK": 1, "HN": 1, "BT": 1}
        assert manifest["dataspecs"] == ["BLDN"]
        assert (race_dir / "CURRENT").read_text(encoding="utf-8") == "20260924T115403Z"
        out = capsys.readouterr().out
        assert "SK: 1" in out and "保存対象外" not in out

    def test_race_kind_marks_blood_records_as_not_saved(self, tmp_path, monkeypatch, capsys):
        link = DataspecAwareLink(open_codes={"DIFF": 0},
                                 records_by_dataspec={"DIFF": [fx.make_hn()]})
        monkeypatch.setattr(update_data.jvlink, "open_link", lambda *a, **k: link)
        update_data.main(["--dataspec", "DIFF", "--fromdate", "20260801000000",
                          "--data-dir", str(tmp_path)])
        assert "HN: 1（保存対象外）" in capsys.readouterr().out

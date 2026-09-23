"""JV-Link からレコードを取得し、版付き CSV として保存する CLI。

配線だけをここに置く。COM の呼び出しは jvlink.py、レコードのデコードは
records.py、CSV への保存は store.py に任せる。

    .venv/Scripts/python.exe scripts/update_data.py --fromdate 20260801000000

.gitignore 済みの vendor/ や data/ には触れず、versioned な保存先は
keiba.config.DATA_DIR（既定）。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config, console  # noqa: E402
from keiba.ingest import jvlink, records, store  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JV-Link から取得して版付き CSV（RA/SE/UM）を保存する。")
    # action="append" は default と併用すると既定値に追記されてしまうため、
    # default はここでは与えない。未指定 (None) のときの既定は
    # _resolve_dataspecs() で config.DEFAULT_DATASPECS を当てる。
    parser.add_argument("--dataspec", action="append", dest="dataspecs", metavar="SPEC",
                         help="JVOpen の dataspec。繰り返し指定できる"
                              "（既定: RACE DIFF）")
    parser.add_argument("--fromdate", required=True,
                         help="取得開始日時。例: 20260801000000")
    parser.add_argument("--option", type=int, default=1,
                         help="JVOpen の option（既定 %(default)s）")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                         help="保存先（既定 %(default)s）")
    parser.add_argument("--verify-time-fields", action="store_true",
                         help="取得後に走破タイムの整合性 (margin_sec の解釈) を確認する")
    parser.add_argument("--dry-run", action="store_true",
                         help="取得はするが CURRENT を切り替えない（保存内容は破棄する）")
    return parser.parse_args(argv)


def _resolve_dataspecs(args: argparse.Namespace) -> Sequence[str]:
    """--dataspec が一度も指定されなければ既定の (RACE, DIFF) を使う。

    RACE だけでは競走馬マスタ UM が取れない（JVOpen("RACE", ...) は運ばない）。
    UM を運ぶのは蓄積系差分の DIFF なので、既定で両方を読む。
    """
    return args.dataspecs if args.dataspecs is not None else config.DEFAULT_DATASPECS


#: 蓄積系差分は初回にセットアップ取得（--option 4）を走らせないと空になる。
#: -1（該当データなし）が出たときにヒントを出すのはこれらの dataspec だけ。
_SETUP_REQUIRED_DATASPECS = ("DIFF", "DIFN")


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    args = _parse_args(argv)
    dataspecs = _resolve_dataspecs(args)

    link = jvlink.open_link()
    version_id = store.new_version_id()
    writer = store.VersionWriter(
        args.data_dir, version_id,
        dataspecs=dataspecs, fromdate=args.fromdate, option=args.option,
    )

    # dataspec ごと・spec ごとの件数。RA/SE/UM 以外は store.VersionWriter.add() が
    # 黙って捨てるので、何が捨てられたかをここで operator に見せる。
    # JVOpen("RACE", ...) は競走馬マスタ UM を運ばない（運ぶのは DIFF）ため、
    # dataspec ごとの内訳を見せないと「UM が0件」に気づけない。
    counts_by_dataspec: dict[str, Counter[str]] = {}
    open_codes: dict[str, int] = {}
    se_rows: list[dict] = []

    # VersionWriter は「構築したら必ず commit() か abort() のどちらかで終わる」
    # という不変条件を保つ。読み取りループの中だけでなく commit() 自身が失敗
    # する経路（manifest 書き込み中のディスクフル、rename の失敗、既に完成した
    # 版がある場合の DataVersionError）でも同じリークが起きるため、commit()
    # の呼び出しもこの try の中に含める。abort() は rmtree に ignore_errors=True
    # を使っており、閉じ済みハンドルに対しても、rename 後で .partial が既に
    # 存在しない場合でも安全に呼べる（詳細は task-5-report.md）。
    # KeyboardInterrupt も含めて後始末したいので BaseException で受ける。
    # 半端な版を作らないよう、全 dataspec を読み切ってから1回だけ commit する。
    try:
        for dataspec in dataspecs:
            # jvlink.read_records() をそのまま使うと open_code が見えず、
            # 「該当データなし (-1)」と「読んだが0件」を区別できない。
            # open_dataspec() で JVOpen の結果を自分で見てから読み出しに進み、
            # JVClose も自分で持つ（read_records の finally に頼らない）。
            open_code, download_count = jvlink.open_dataspec(
                link, dataspec, args.fromdate, args.option)
            open_codes[dataspec] = open_code
            try:
                if open_code == 0:
                    counts = counts_by_dataspec.setdefault(dataspec, Counter())
                    for raw in jvlink.read_open_dataspec(link, download_count):
                        spec = records.record_spec(raw)
                        counts[spec] += 1
                        parsed = records.parse(raw)
                        if parsed is None:
                            continue
                        parsed_spec, row = parsed
                        writer.add(parsed_spec, row)
                        if parsed_spec == "SE":
                            se_rows.append(row)
                elif open_code != -1:
                    # -1 (該当データなし) はエラーではないので読まずに次へ進む。
                    # それ以外は本当のエラーなので版全体を捨てる。
                    raise jvlink.JVLinkError(
                        jvlink.describe_open_code(open_code, dataspec))
            finally:
                link.JVClose()

        print("読み込んだレコード数:")
        for dataspec in dataspecs:
            if open_codes[dataspec] == 0:
                print(f"  {dataspec}: 正常")
                counts = counts_by_dataspec.get(dataspec, Counter())
                if counts:
                    for spec in sorted(counts):
                        note = "" if spec in ("RA", "SE", "UM") else "（保存対象外）"
                        print(f"    {spec}: {counts[spec]}{note}")
                else:
                    print("    (0件)")
            else:
                # ここに来る時点で open_codes[dataspec] は -1 しかありえない
                # （それ以外は上のループで JVLinkError にして版ごと捨てている）。
                print(f"  {dataspec}: 該当データなし（JVOpen が -1 を返しました）")
                if dataspec in _SETUP_REQUIRED_DATASPECS:
                    print("    蓄積系差分は --option 4 のセットアップ取得で馬マスタ UM を"
                          "取り込めます")
                    print("    （実測 2026-01-01 時点で 198,212 件。値は実測であり、"
                          "データの提供状況が変われば変わりえます）。")
                    print("    ただし実測では生年が 1936〜2021 年に偏り、収録馬の96%が"
                          "抹消済み馬でした。")
                    print("    現役の若い馬の大半はマスタに含まれないとみてください。")
                    print("    馬マスタが無くても・不完全でも予想は成立します"
                          "（馬齢は出走記録から計算し、馬名の索引も出走記録から作るため）。")
                    print("    それでも補いたい場合は一度だけ次を実行してください:")
                    print(f"      scripts/update_data.py --dataspec {dataspec}"
                          f" --option 4 --fromdate <取得開始日時>")

        if args.dry_run:
            writer.abort()
            print("\n--dry-run のため CURRENT は変更していません（保存内容は破棄しました）。")
        else:
            version_path = writer.commit()
            print(f"\nversion_id: {version_id}")
            print(f"manifest  : {version_path / 'manifest.json'}")
    except BaseException:
        writer.abort()      # 握りつぶさない。後始末だけしてそのまま投げ直す
        raise

    exit_code = 0
    if args.verify_time_fields:
        report = jvlink.verify_time_fields(se_rows)
        mismatches = [entry for entry in report if not entry["matches"]]
        print(f"\n走破タイムの照合: {len(report)} 件中、不一致 {len(mismatches)} 件")
        for entry in mismatches:
            print(f"  race_id={entry['race_id']}"
                  f" computed={entry['computed']} recorded={entry['recorded']}")
        if mismatches:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

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
from keiba.ingest import jvlink, rawdump, records, store  # noqa: E402


#: 取り込む系統ごとの既定。blood を race と同じ保存先に書くと、commit() が
#: CURRENT をレースの無い版に切り替えて予想を壊す。
#: required は --from-raw で1件も無ければ版を作らない種別（今は race の RA だけ）。
_KINDS: dict[str, dict] = {
    "race": {"dataspecs": config.DEFAULT_DATASPECS, "data_dir": config.DATA_DIR,
             "files": store.RACE_FILES, "required": "RA"},
    "blood": {"dataspecs": config.DEFAULT_BLOOD_DATASPECS,
              "data_dir": config.BLOOD_DATA_DIR, "files": store.BLOOD_FILES},
    # 払戻・オッズも RACE が運ぶが、レースの版とは別の系統に置く（config.MARKET_DATA_DIR）。
    "market": {"dataspecs": config.DEFAULT_MARKET_DATASPECS,
               "data_dir": config.MARKET_DATA_DIR, "files": store.MARKET_FILES},
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JV-Link から取得して版付き CSV を保存する"
                    "（--kind race: RA/SE/UM、--kind blood: SK/HN/BT（BLDN）、"
                    "--kind market: HR/O1〜O6（払戻・オッズ））。")
    parser.add_argument("--kind", choices=_KINDS, default="race",
                         help="取り込む系統（既定 %(default)s）。blood は血統を"
                              " レースとは別の保存先に置く")
    # action="append" は default と併用すると既定値に追記されてしまうため、
    # default はここでは与えない。未指定 (None) のときの既定は
    # _resolve_dataspecs() で config.DEFAULT_DATASPECS を当てる。
    parser.add_argument("--dataspec", action="append", dest="dataspecs", metavar="SPEC",
                         help="JVOpen の dataspec。繰り返し指定できる"
                              "（既定: race は RACE DIFF、blood は BLDN）")
    parser.add_argument("--fromdate",
                         help="取得開始日時。例: 20260801000000（--from-raw 以外では必須）")
    parser.add_argument("--from-raw", type=Path, default=None, metavar="DIR",
                         help="JV-Link を呼ばず、保存した生レコード（keiba.ingest.rawdump の形式）"
                              "から版を作る。dataspec・fromdate・option はその manifest から取る")
    # 既定は None にして、--from-raw と一緒に明示されたかを見分ける（既定の 1 は下で当てる）。
    parser.add_argument("--option", type=int, default=None,
                         help="JVOpen の option（既定 1）")
    parser.add_argument("--data-dir", type=Path, default=None,
                         help=f"保存先（既定: race は {config.DATA_DIR}、"
                              f"blood は {config.BLOOD_DATA_DIR}、market は {config.MARKET_DATA_DIR}）")
    parser.add_argument("--verify-time-fields", action="store_true",
                         help="取得後に走破タイムの整合性 (margin_sec の解釈) を確認する")
    parser.add_argument("--dry-run", action="store_true",
                         help="取得はするが CURRENT を切り替えない（保存内容は破棄する）")
    args = parser.parse_args(argv)
    if args.from_raw is None:
        if args.fromdate is None:
            parser.error("--fromdate は必須です（--from-raw を使うときを除く）。")
        if args.option is None:
            args.option = 1
    elif args.fromdate is not None or args.option is not None:
        # 生レコードは取得したときの fromdate・option のものしか入っていない。
        # 別の値を manifest に書くと、版がどこからどこまでを持つかを偽ることになる。
        parser.error("--from-raw では --fromdate / --option を指定できません"
                     "（生レコードの manifest の値を使います）。")
    return args


def _resolve_dataspecs(args: argparse.Namespace) -> Sequence[str]:
    """--dataspec が一度も指定されなければ系統ごとの既定を使う。

    race の既定は (RACE, DIFF)。RACE だけでは競走馬マスタ UM が取れない
    （JVOpen("RACE", ...) は運ばない）。UM を運ぶのは蓄積系差分の DIFF なので、
    既定で両方を読む。blood の既定は (BLDN,)。BLOD は旧形式で 2023-07 に止まっている。
    """
    if args.dataspecs is not None:
        return args.dataspecs
    return _KINDS[getattr(args, "kind", "race")]["dataspecs"]


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    """--data-dir が無ければ系統ごとの既定。血統をレースの版に混ぜないため。"""
    if args.data_dir is not None:
        return args.data_dir
    return _KINDS[args.kind]["data_dir"]


#: 蓄積系差分は初回にセットアップ取得（--option 4）を走らせないと空になる。
#: -1（該当データなし）が出たときにヒントを出すのはこれらの dataspec だけ。
_SETUP_REQUIRED_DATASPECS = ("DIFF", "DIFN")


def _is_saved(spec: str, files: store.FileSet) -> bool:
    """レコード種別の行が、この版のファイルの組のどれかに書かれるか。"""
    return any(key in files for key in records.table_keys(spec))


def _consume(raw: bytes, counts: Counter, writer: store.VersionWriter,
             se_rows: list[dict] | None, files: store.FileSet) -> None:
    """1レコードを数えて、表の行に分けて書く。

    files の組に書かれない種別は数えるだけでパースしない（race の版に来る O1〜O6 など）。
    使わない種別のパースに時間を使わず、その種別の形式が変わっていても取り込みを止めない。

    se_rows は --verify-time-fields のときだけ渡す（None なら溜めない）。5年ぶんの
    セットアップ取得では SE が数十万行あり、使わないのに全部をメモリに持つことになる。
    """
    spec = records.record_spec(raw)
    counts[spec] += 1
    if not _is_saved(spec, files):
        return
    for key, row in records.parse_rows(raw):
        writer.add(key, row)
        if key == "SE" and se_rows is not None:
            se_rows.append(row)


def _verify_time_fields(se_rows: list[dict]) -> int:
    """走破タイムの照合を出力し、不一致があれば 1、なければ 0 を返す。"""
    report = jvlink.verify_time_fields(se_rows)
    mismatches = [entry for entry in report if not entry["matches"]]
    print(f"\n走破タイムの照合: {len(report)} 件中、不一致 {len(mismatches)} 件")
    for entry in mismatches:
        print(f"  race_id={entry['race_id']}"
              f" computed={entry['computed']} recorded={entry['recorded']}")
    return 1 if mismatches else 0


def _from_raw(args: argparse.Namespace) -> int:
    """保存した生レコードから版を作る。JV-Link は呼ばない。"""
    try:
        raw_manifest = rawdump.read_manifest(args.from_raw)
    except rawdump.RawDumpError as error:
        raise SystemExit(f"生レコードを使えません: {error}") from error
    dataspec = raw_manifest["dataspec"]
    wanted = _resolve_dataspecs(args)
    if dataspec not in wanted:
        raise SystemExit(f"生レコードの dataspec は {dataspec} ですが、--kind {args.kind} は"
                         f" {'/'.join(wanted)} を読みます。取り違えていないか確かめてください。")
    files = _KINDS[args.kind]["files"]
    version_id = store.new_version_id()
    writer = store.VersionWriter(
        _resolve_data_dir(args), version_id, dataspecs=[dataspec],
        fromdate=raw_manifest["fromdate"], option=raw_manifest["option"], files=files,
        source={"raw_dir": str(args.from_raw), "raw_started_at": raw_manifest.get("started_at"),
                "raw_counts": raw_manifest.get("counts")},
    )
    counts: Counter[str] = Counter()
    se_rows: list[dict] | None = [] if args.verify_time_fields else None
    try:
        # 表に書かれない種別（JG など）はパースもしないので、組に関係する種別だけ読む。
        specs = [spec for spec in raw_manifest.get("counts", {}) if _is_saved(spec, files)]
        try:
            for raw in rawdump.iter_records(args.from_raw, specs):
                _consume(raw, counts, writer, se_rows, files)
        except rawdump.RawDumpError as error:
            raise SystemExit(f"生レコードを使えません: {error}") from error
        print(f"生レコード {args.from_raw} から読んだレコード数:")
        for spec in sorted(raw_manifest.get("counts", {})):
            note = "" if spec in counts else "（保存対象外）"
            print(f"  {spec}: {counts.get(spec, raw_manifest['counts'][spec])}{note}")
        # ファイルが途中で終わっていても長さの欄の切れ目なら iter_records は気づけない。
        # manifest の件数と合わなければ、欠けたまま版を作らない。
        short = {spec: (counts.get(spec, 0), raw_manifest["counts"][spec]) for spec in specs
                 if counts.get(spec, 0) != raw_manifest["counts"][spec]}
        if short:
            raise SystemExit("生レコードの件数が manifest と合いません（読んだ件数, manifest）: "
                             f"{short}。取り直してください。")
        required = _KINDS[args.kind].get("required")
        if required is not None and not counts.get(required):
            # DIFF だけの生レコードなどは RA を運ばない。レースの無い版に CURRENT を
            # 切り替えると予想が壊れる。
            raise SystemExit(f"生レコードに {required} が1件もありません（dataspec {dataspec}）。"
                             f" --kind {args.kind} の版にはできません。")
        if args.dry_run:
            writer.abort()
            print("\n--dry-run のため CURRENT は変更していません（保存内容は破棄しました）。")
        else:
            version_path = writer.commit()
            print(f"\nversion_id: {version_id}")
            print(f"manifest  : {version_path / 'manifest.json'}")
    except BaseException:
        writer.abort()
        raise
    return _verify_time_fields(se_rows) if se_rows is not None else 0


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    args = _parse_args(argv)
    if args.from_raw is not None:
        return _from_raw(args)
    dataspecs = _resolve_dataspecs(args)
    files = _KINDS[args.kind]["files"]

    link = jvlink.open_link()
    version_id = store.new_version_id()
    writer = store.VersionWriter(
        _resolve_data_dir(args), version_id,
        dataspecs=dataspecs, fromdate=args.fromdate, option=args.option, files=files,
    )

    # dataspec ごと・spec ごとの件数。files に無い種別は store.VersionWriter.add() が
    # 黙って捨てるので、何が捨てられたかをここで operator に見せる。
    # JVOpen("RACE", ...) は競走馬マスタ UM を運ばない（運ぶのは DIFF）ため、
    # dataspec ごとの内訳を見せないと「UM が0件」に気づけない。
    counts_by_dataspec: dict[str, Counter[str]] = {}
    open_codes: dict[str, int] = {}
    se_rows: list[dict] | None = [] if args.verify_time_fields else None

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
                        _consume(raw, counts, writer, se_rows, files)
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
                        note = "" if _is_saved(spec, files) else "（保存対象外）"
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

    return _verify_time_fields(se_rows) if se_rows is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())

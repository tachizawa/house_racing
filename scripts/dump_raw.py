"""JV-Link の蓄積系データを加工せずに保存する。中断しても同じコマンドで続きから読める。

セットアップ取得（option 3/4）は所有者のダイアログ操作が要り、40年ぶん（1986年〜）だと
JVRead だけで30時間ほどかかる。途中で止まっても（PC の再起動・Ctrl+C・エラー）、
同じコマンドをもう一度実行すれば、読み終えた jvd ファイルを JVSkip で飛ばして続きから読む
（インターフェース仕様書 p.19「セットアップデータの中断・再開」）。

実行:
  .venv/Scripts/python.exe scripts/dump_raw.py --fromdate 19860101000000 --out data/raw/race_19860101

出力（keiba.ingest.rawdump の形式。update_data.py --from-raw でそのまま版にできる）:
  <out>/<種別>.bin.gz   … 4バイトのビッグエンディアン長 + cp932 のバイト列（jvd ファイルごとに gzip の member）
  <out>/resume.json     … 読み終えたファイル名・各 gz の大きさ・件数（再開に使う）
  <out>/manifest.json   … 全部読み終えたときだけ書く
  <out>/progress.log    … 進み具合
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba.ingest import jvlink, rawdump  # noqa: E402

#: 進み具合を書く間隔（レコード数）
_PROGRESS_EVERY = 50_000


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataspec", default="RACE")
    parser.add_argument("--fromdate", required=True, help="例: 19860101000000")
    parser.add_argument("--option", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True,
                        help="保存先。再開の記録があれば続きから読む")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        dump = rawdump.ResumableRawDump(args.out, dataspec=args.dataspec,
                                        fromdate=args.fromdate, option=args.option)
    except rawdump.RawDumpError as error:
        raise SystemExit(f"保存先を使えません: {error}") from error
    if dump.finished:
        print(f"already complete: {args.out} (manifest.json)")
        return 0

    plog = open(args.out / "progress.log", "a", encoding="utf-8", buffering=1)

    def log(message: str, console: str) -> None:
        # コンソールには ASCII だけ出す（読んでいる間はコードページを 932 にしているため）
        plog.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}\n")
        print(console, flush=True)

    resumed = dump.done_file_count
    log(f"JVOpen {args.dataspec} {args.fromdate} option={args.option}（済み {resumed} ファイル・"
        f"{dump.record_count:,} 件から再開。ダイアログが出たら「持っていない」→ OK）",
        f"JVOpen {args.dataspec} {args.fromdate} option={args.option}"
        f" (resume: {resumed} files done, {dump.record_count} records)")
    link = jvlink.open_link()
    t0 = time.time()
    read = skipped = 0
    try:
        open_code, download_count = jvlink.open_dataspec(link, args.dataspec, args.fromdate,
                                                         args.option)
        log(f"JVOpen の戻り {open_code}、ダウンロード {download_count} ファイル",
            f"JVOpen returned {open_code}, downloading {download_count} files")
        if open_code != 0:
            log(f"中止: {jvlink.describe_open_code(open_code, args.dataspec)}",
                f"aborted: JVOpen code {open_code}")
            return 1
        for fname, raw in jvlink.read_open_dataspec_files(link, download_count):
            if raw is None:                     # ファイル切替
                dump.end_file()
                continue
            if not fname:
                raise jvlink.JVLinkError("JVRead がファイル名を返しません。再開できないので中止します。")
            if dump.is_done(fname):
                link.JVSkip()                   # 読み終えたファイルの残りを飛ばす
                skipped += 1
                continue
            dump.add(fname, raw)
            read += 1
            if read % _PROGRESS_EVERY == 0:
                rate = read / max(time.time() - t0, 1)
                log(f"今回 {read:,} 件（{rate:.0f} 件/秒）・累計 {dump.record_count:,} 件確定・"
                    f"済み {dump.done_file_count} ファイル（飛ばした {skipped}）・読み中 {fname}",
                    f"{read} records this run ({rate:.0f}/s), {dump.done_file_count} files done,"
                    f" {skipped} skipped")
        dump.complete()
    except KeyboardInterrupt:
        dump.discard_pending()
        log("中断しました（Ctrl+C）。同じコマンドで続きから読めます。",
            "interrupted. run the same command again to resume.")
        return 130
    except Exception as error:  # noqa: BLE001  確定した分は残し、理由と再開の仕方を出す
        dump.discard_pending()
        log(f"エラー: {error!r}。同じコマンドで続きから読めます。",
            f"error: {type(error).__name__}. run the same command again to resume.")
        return 1
    finally:
        link.JVClose()
        plog.close()
    print(f"complete: {dump.record_count} records, {dump.done_file_count} files"
          f" ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

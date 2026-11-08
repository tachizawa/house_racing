"""JV-Link のレコードを加工せずに保存する（keiba.ingest.rawdump の形式）。

セットアップ取得（option 3/4）は所有者のダイアログ操作（スタートキットは「持っていない」→ OK）が
要るので、パーサより先に取っておき、版は後から作る:

    .venv/Scripts/python.exe scripts/dump_raw.py --dataspec SLOP --fromdate 20030101000000 --option 4
    .venv/Scripts/python.exe scripts/update_data.py --kind training --from-raw data/raw/<dir>

出力は data/raw/<UTC時刻>_<dataspec>/（<種別>.bin.gz と manifest.json、進み具合は progress.log）。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from keiba import config, console  # noqa: E402
from keiba.ingest import jvlink  # noqa: E402
from keiba.ingest.rawdump import RawDumpWriter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataspec", required=True, help="JVOpen の dataspec（例: SLOP / WOOD）")
    parser.add_argument("--fromdate", required=True, help="例: 20030101000000")
    parser.add_argument("--option", type=int, default=1,
                        help="JVOpen の option（セットアップは 4。ダイアログが出る）")
    parser.add_argument("--out-root", type=Path, default=config.DATA_DIR / "raw")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_root / f"{stamp}_{args.dataspec}"
    log_path = out.parent / f"{out.name}.progress.log"
    out.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if args.option in (3, 4):
        log("セットアップ取得です。スタートキットのダイアログが出たら「持っていない」を選んで OK を押してください。")
    link = jvlink.open_link()
    started = time.time()
    open_code, download_count = jvlink.open_dataspec(link, args.dataspec, args.fromdate,
                                                     args.option)
    log(f"JVOpen {args.dataspec} {args.fromdate} option={args.option} → {open_code}、"
        f"ダウンロード {download_count} ファイル（{time.time() - started:.0f}秒）")
    if open_code != 0:
        link.JVClose()
        log("中止: " + jvlink.describe_open_code(open_code, args.dataspec))
        return 1
    count = 0
    try:
        with RawDumpWriter(out, dataspec=args.dataspec, fromdate=args.fromdate,
                           option=args.option) as writer:
            for raw in jvlink.read_open_dataspec(link, download_count):
                writer.write(raw)
                count += 1
                if count % 200_000 == 0:
                    log(f"{count:,} 件（{time.time() - started:.0f}秒）")
    finally:
        link.JVClose()
    log(f"完了: {count:,} 件 → {out}（{time.time() - started:.0f}秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

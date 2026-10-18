"""JV-Link から血統系レコード（SK / HN / UM の3代血統）を取得し、版付き CSV に保存する。

    .venv/Scripts/python.exe scripts/ml/update_pedigree.py --fromdate 19860101000000 --option 4

- **SK/HN だけなら、このスクリプトは要らない。** 既存の ``scripts/update_data.py --kind blood``
  の版（data/blood/）を build_dataset.py がそのまま読む（data.pedigree_source = "blood"）。
- 既定の dataspec は BLDN（SK 産駒マスタ・HN 繁殖馬マスタ・BT 系統情報を運ぶ）。
  BLOD は旧形式（繁殖登録番号8桁）のまま 2023-07 で止まっているので使わない。
  UM（競走馬マスタ）の3代血統で補いたいときは ``--dataspec BLDN --dataspec DIFN``
  のように足す（DIFN / DIFF のどちらで UM が来るかは契約・取得状況で違う。未確認）。
- 蓄積系は初回にセットアップ取得（--option 3 か 4）が要ることがある。-1（該当データ
  なし）が返ったらまずそれを疑う。
- 保存先は data/ml/pedigree/versions/<version_id>/。既存モデルの data/versions/ には
  書かない。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba.ingest import jvlink  # noqa: E402
from keiba_ml.config import ML_DATA_DIR  # noqa: E402
from keiba_ml.ingest import pedigree_records, pedigree_store  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="血統系レコードを取得して保存する。")
    parser.add_argument("--dataspec", action="append", dest="dataspecs", metavar="SPEC",
                        help="JVOpen の dataspec。繰り返し指定できる（既定: BLDN）")
    parser.add_argument("--fromdate", required=True, help="取得開始日時。例: 19860101000000")
    parser.add_argument("--option", type=int, default=1,
                        help="JVOpen の option（1 通常 / 3・4 セットアップ。既定 %(default)s）")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="保存せずに件数だけ見る")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    args = _parse_args(argv)
    dataspecs = args.dataspecs or ["BLDN"]
    # dataspecs の重複・版 id の衝突は writer が拒否する。JV-Link を開く前に確かめる。
    writer = pedigree_store.PedigreeVersionWriter(
        args.ml_data_dir, pedigree_store.new_version_id(),
        dataspecs=dataspecs, fromdate=args.fromdate, option=args.option)
    seen: dict[str, Counter[str]] = {}
    try:
        link = jvlink.open_link()
        for dataspec in dataspecs:
            try:
                # JVOpen 自体が例外を投げても JVClose する（finally の内側で開く）。
                code, download_count = jvlink.open_dataspec(
                    link, dataspec, args.fromdate, args.option)
                if code == -1:
                    print(f"{dataspec}: 該当データなし（-1）。初回ならセットアップ取得"
                          f"（--option 4）を試してください。")
                    continue
                if code != 0:
                    raise jvlink.JVLinkError(jvlink.describe_open_code(code, dataspec))
                counts = seen.setdefault(dataspec, Counter())
                for raw in jvlink.read_open_dataspec(link, download_count):
                    counts[pedigree_records.record_spec(raw)] += 1
                    parsed = pedigree_records.parse(raw)
                    if parsed is not None:
                        writer.add(*parsed)
            finally:
                link.JVClose()
        for dataspec, counts in seen.items():
            print(f"{dataspec}:")
            for spec, count in sorted(counts.items()):
                note = "" if spec in pedigree_store.FILES else "（保存対象外）"
                print(f"  {spec}: {count}{note}")
        mismatch = {k: v for k, v in writer.length_mismatch.items() if v}
        if mismatch:
            print(f"\n警告: 仕様の長さと違うレコードがあります: {mismatch}"
                  f"（期待値 {pedigree_records.EXPECTED_LENGTHS}）。"
                  " scripts/ml/check_offsets.py で位置を確かめてください。")
        if args.dry_run:
            writer.abort()
            print("\n--dry-run のため保存していません。")
        elif not any(writer.counts.values()):
            # 全 dataspec が -1 などで1件も取れなかった。空の版を CURRENT にしない。
            writer.abort()
            print("\n保存対象のレコードが1件も取得できなかったので、版を作りませんでした。")
            return 1
        else:
            path = writer.commit()
            print(f"\n保存しました: {path}")
    except BaseException:
        writer.abort()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

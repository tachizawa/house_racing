"""早熟・晩成の集計（父ごとの産駒の年齢別 A/E）を CSV に出す。

    .venv/Scripts/python.exe scripts/ml/growth_report.py --until 2023-12-31 --path S

--until より後のレースは集計しない。テスト期間を含めて眺めると、その知見で
特徴量や定義を選んだ時点でテストが汚れるので、既定は設定の検証期間の開始日の前日。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.config import DEFAULT_CONFIG_PATH, ML_DATA_DIR, load_config, path_label  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.dataset.split import make_window  # noqa: E402
from keiba_ml.evaluate.growth import age_curve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None,
                        help=f"設定ファイル（省略時 {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--path", default="S", help="祖先の位置（S=父, DS=母父 ...）")
    parser.add_argument("--until", type=date.fromisoformat, default=None)
    parser.add_argument("--min-starts", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as error:
        parser.error(str(error))
    df, names, manifest = build.load(args.ml_data_dir, args.dataset_id)
    cutoff = make_window(config.split, 1, date.min).valid_start - timedelta(days=1)
    until = args.until or cutoff
    if until > cutoff:
        print(f"注意: --until {until} は検証期間の開始日の前日 {cutoff} より後です。"
              " 検証・テスト期間の結果を見て特徴量や定義を選ぶと、その評価は信用できなくなります。")
    df = df[df["race_date"].dt.date <= until]
    table = age_curve(df, path=args.path, min_starts=args.min_starts,
                      shrinkage=config.pedigree.shrinkage, names=names)
    out = args.out or (args.ml_data_dir / "reports" /
                       f"growth_{args.path}_{manifest['dataset_id']}_until{until}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{path_label(args.path)}ごとの年齢別 A/E（{until} まで、出走 {args.min_starts} 以上）: {out}")
    cols = ["name", "starts", "ae_all", "ae_age2", "ae_age3", "ae_age4plus", "maturity"]
    print("\n晩成寄り（maturity 上位10）:")
    print(table[cols].head(10).to_string(index=False))
    print("\n早熟寄り（maturity 下位10）:")
    print(table[cols].dropna().tail(10).iloc[::-1].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

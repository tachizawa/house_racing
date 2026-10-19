"""学習期間の比較（train_pedigree.py --compare）の差を、保存済みの予測から出し直す。

    .venv/Scripts/python.exe scripts/ml/compare_runs.py --run data/ml/runs/<run_id>

先頭の学習期間（既定 10y）に対する各期間の的中数の差を、同じテストレースでの対応あり
ブートストラップ（evaluate.metrics.paired_difference。回数は設定の evaluate.bootstrap）で出し、
<run_dir>/window_differences.md|csv に書く。学習はやり直さない。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.config import ML_DATA_DIR, config_from_dict  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.dataset.split import Window, split_frame  # noqa: E402
from keiba_ml.experiment import to_markdown, write_window_differences  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="data/ml/runs/<run_id>")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args(argv)

    run_config = args.run / "config.json"
    if not run_config.exists():
        print(f"学習時の設定がありません: {run_config}")
        return 2
    config = config_from_dict(json.loads(run_config.read_text(encoding="utf-8")),
                              source=str(run_config))
    labels = sorted((p.name for p in args.run.iterdir()
                     if (p / "predictions_test.parquet").exists() and p.name.endswith("y")),
                    key=lambda name: int(name[:-1]))
    if len(labels) < 2:
        print(f"比べる学習期間が2つ以上ありません: {labels}")
        return 2
    saved = json.loads((args.run / labels[0] / "metrics.json").read_text(encoding="utf-8"))["window"]
    window = Window(train_years=saved["train_years"],
                    **{k: date.fromisoformat(str(saved[k])) for k in
                       ("train_start", "actual_train_start", "valid_start", "test_start", "test_end")})
    df, _names, _ = build.load(args.ml_data_dir, args.dataset_id)
    _, _, test = split_frame(df, window)
    table = write_window_differences(args.run, labels, test, config)
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

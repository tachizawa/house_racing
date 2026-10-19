"""Phase 2: 血統モデル（CatBoost）を学習し、固定のテスト期間で評価する。

scripts/ml/train.py --model pedigree と同じ（以前からの入口として残す）。

    # 10年で基準モデル
    .venv/Scripts/python.exe scripts/ml/train_pedigree.py --years 10
    # 学習期間の比較（設定ファイルの split.train_years。既定 10/20/30/40）
    .venv/Scripts/python.exe scripts/ml/train_pedigree.py --compare
    # 目的関数を変えて比べる
    .venv/Scripts/python.exe scripts/ml/train_pedigree.py --years 10 --loss YetiRank

結果は data/ml/runs/pedigree/<run_id>/ に出る（2026-10-20 より前の実行は data/ml/runs/<run_id>/）。
comparison.md が期間別の比較表。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _train():
    path = Path(__file__).resolve().parent / "train.py"
    spec = importlib.util.spec_from_file_location("ml_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return _train().main(argv, model="pedigree")


if __name__ == "__main__":
    raise SystemExit(main())

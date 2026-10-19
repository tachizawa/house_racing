"""学習済みの血統モデルで、結果の出ていないレース（出馬表）の上位3頭を出す。

scripts/ml/predict.py と同じ（以前からの入口として残す。血統モデル以外の記録は拒否する）。

    .venv/Scripts/python.exe scripts/ml/predict_pedigree.py --model data/ml/runs/<run_id>/10y
    .venv/Scripts/python.exe scripts/ml/predict_pedigree.py --model ... --race-id 2026101805041101
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _predict():
    path = Path(__file__).resolve().parent / "predict.py"
    spec = importlib.util.spec_from_file_location("ml_predict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _predict()
live_rows = _module.live_rows


def main(argv: list[str] | None = None) -> int:
    return _module.main(argv, expected_model="pedigree")


if __name__ == "__main__":
    raise SystemExit(main())

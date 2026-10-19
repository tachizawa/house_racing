"""学習済みの血統モデルを、まだ何の判断にも使っていない期間で1回だけ評価する。

scripts/ml/evaluate.py と同じ（以前からの入口として残す。血統モデル以外の記録は拒否する）。

    .venv/Scripts/python.exe scripts/ml/evaluate_pedigree.py --model data/ml/runs/<run_id>/10y \\
        --from 2026-01-01 --to 2026-10-18
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _evaluate():
    path = Path(__file__).resolve().parent / "evaluate.py"
    spec = importlib.util.spec_from_file_location("ml_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return _evaluate().main(argv, expected_model="pedigree")


if __name__ == "__main__":
    raise SystemExit(main())

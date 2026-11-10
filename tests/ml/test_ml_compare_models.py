"""scripts/ml/compare_models.py の --blank の展開。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("compare_models",
                                               ROOT / "scripts/ml/compare_models.py")
compare_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_models)


def test_blanking_any_weight_column_blanks_all_weight_columns():
    # x_weight_vs_mean は当日の馬体重から作るので、c_weight だけ消すと前日予想の条件を甘く測る
    assert compare_models._expand_blank(["c_weight", "c_zogen"]) == [
        "c_weight", "c_zogen", "x_weight_vs_mean"]
    assert compare_models._expand_blank(["c_zogen"]) == ["c_zogen", "c_weight", "x_weight_vs_mean"]


def test_blank_without_weight_columns_is_unchanged():
    assert compare_models._expand_blank([]) == []
    assert compare_models._expand_blank(["t_slope_best", "t_slope_best"]) == ["t_slope_best"]

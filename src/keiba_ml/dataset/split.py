"""時系列分割。ランダム分割はしない。

    |<------ 学習 N 年 ------>|<- 検証 V 年 ->|<-- テスト（固定）-->|
    train_start          valid_start      test_start         test_end

テスト期間は学習期間を変えても固定する（学習期間の比較で「同じ試験」を受けさせる）。
手元のデータが N 年ぶん遡れないときは ``truncated=True`` にして実際の開始日を残す。
黙って短い期間で学習し「40年で学習した」と報告しないため。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from keiba_ml.config import SplitConfig

#: 実際のデータ開始日が設定の開始日よりこれ以上遅ければ「切り詰め」とみなす。
#: 年初の数日は開催が無いので、厳密に比べると常に切り詰め扱いになってしまう。
#: 1か月ぶん（1月が丸ごと無い、など）を見逃さないよう1週間にとどめる。
TRUNCATION_TOLERANCE_DAYS = 7


@dataclass(frozen=True)
class Window:
    train_years: int
    train_start: date          # 設定どおりの開始日
    actual_train_start: date   # データがある最初の日（train_start より後なら切り詰め）
    valid_start: date
    test_start: date
    test_end: date

    @property
    def truncated(self) -> bool:
        return (self.actual_train_start - self.train_start).days > TRUNCATION_TOLERANCE_DAYS

    @property
    def label(self) -> str:
        return f"{self.train_years}y"


def _minus_years(day: date, years: int) -> date:
    return (pd.Timestamp(day) - pd.DateOffset(years=years)).date()


def make_window(split: SplitConfig, train_years: int, first_data_date: date) -> Window:
    valid_start = _minus_years(split.test_start, split.valid_years)
    train_start = _minus_years(valid_start, train_years)
    return Window(
        train_years=train_years, train_start=train_start,
        actual_train_start=max(train_start, first_data_date),
        valid_start=valid_start, test_start=split.test_start, test_end=split.test_end,
    )


def split_frame(df: pd.DataFrame, window: Window
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """結果のある行だけを3つに分ける。"""
    res = df[df["has_result"]]
    day = res["race_date"].dt.date
    train = res[(day >= window.train_start) & (day < window.valid_start)]
    valid = res[(day >= window.valid_start) & (day < window.test_start)]
    test = res[(day >= window.test_start) & (day <= window.test_end)]
    return train, valid, test


def history_frame(df: pd.DataFrame, window: Window, stats_history: str) -> pd.DataFrame:
    """産駒成績の集計に使ってよい範囲（テスト終了日まで。window なら学習開始日から）。"""
    day = df["race_date"].dt.date
    mask = day <= window.test_end
    if stats_history == "window":
        mask &= day >= window.train_start
    elif stats_history != "all":
        raise ValueError(f'stats_history は "window" か "all" です: {stats_history!r}')
    return df[mask]

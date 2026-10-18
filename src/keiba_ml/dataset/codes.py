"""年代で異なるコード体系を、学習で使う1つの体系にそろえる。

ここに置くのは「生のコード → 正規化した値」の変換だけ。どの列を特徴量に
するかは features/ が決める（設計原則 2: 前処理と特徴量生成を分離する）。

**所有者が確認済みのコード変更をここに足すこと。** 現在入れてあるのは、
JV-Data の仕様として広く知られているものだけ:

1. 馬齢表記: 2000年以前は数え年、2001年から満年齢（CodeConfig.kazoe_age_until_year）。
   生年が血統（SK/UM）から分かる馬は ``レースの年 - 生年`` で計算し、表記の
   違いを経由しない。生年が分からない馬だけ表記の変換を使う。
2. 競走条件コード（収得賞金の条件）: 時代によって 400万/500万/700万/800万/900万/
   1000万/1200万/1500万/1600万 などの区分があり、2019年夏に 1勝/2勝/3勝クラスへ
   名称変更された（コード 005/010/016 はそのまま）。順序の段階（tier）に寄せる。
   ``JYOKEN_TIER_OVERRIDES`` に確認済みの対応を足すと、閾値による推定より優先する。
3. 未設定の表現: 空白・ゼロ埋め（"0", "00", "0000000000"）を欠損に寄せる。
"""

from __future__ import annotations

import math

from keiba import config as keiba_config
from keiba_ml.config import DISTANCE_BANDS

#: 生のまま特徴量にすると「未設定」が1つのカテゴリとして学習される値。
_BLANK_CODES = frozenset({"", "0", "00", "000"})

#: 競走条件コード → 段階（0 新馬 / 1 未勝利 / 2 1勝相当 / 3 2勝相当 / 4 3勝相当 /
#: 5 オープン）。ここに書いたコードは閾値による推定より優先する。
#: 所有者が確認した年代別の対応があればここに足す（例: "009": 3）。
JYOKEN_TIER_OVERRIDES: dict[str, int] = {
    "701": 0,  # 新馬
    "702": 0,  # 未出走
    "703": 1,  # 未勝利
    "999": 5,  # オープン
}

#: 収得賞金条件（001〜100、単位 100万円）を段階に寄せるときの上限。
#: 上から順に「この値以下ならこの段階」。2019年以前の 500万下=1勝、1000万下=2勝、
#: 1600万下=3勝 に合わせた。より古い時代の区分（400万/700万/800万/900万/1200万/
#: 1500万 等）は、近い現行区分に寄せる仮置き。
JYOKEN_PRIZE_TIERS: tuple[tuple[int, int], ...] = ((5, 2), (10, 3), (16, 4))


def blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return None if value in _BLANK_CODES or not value.strip("0") else value


def jyoken_tier(code: str | None) -> int | None:
    """競走条件コードを段階（0〜5）にする。決まらなければ None。"""
    code = blank_to_none(code)    # ゼロ埋め（"0", "00", "000" ...）は未設定
    if code is None:
        return None
    if code in JYOKEN_TIER_OVERRIDES:
        return JYOKEN_TIER_OVERRIDES[code]
    if code.isdigit():
        value = int(code)
        if value <= 100:
            for upper, tier in JYOKEN_PRIZE_TIERS:
                if value <= upper:
                    return tier
            return 5
    return None


def grade_tier(grade_cd: str | None, jyoken_cd: str | None) -> int | None:
    """グレード（G1/G2/G3/L/重賞）を優先し、無ければ競走条件の段階。

    既存の keiba.config.class_level と同じ 0〜8 の尺度に合わせる。
    """
    grade = (grade_cd or "").strip()
    level = {"A": 8, "B": 7, "C": 6, "L": 5, "D": 5}.get(grade)
    if level is not None:
        return level
    return jyoken_tier(jyoken_cd)


def track_kind(track_cd: str | None) -> str | None:
    return keiba_config.track_kind(track_cd or "")


def distance_band(kyori_m: float | int | None) -> str | None:
    if kyori_m is None or (isinstance(kyori_m, float) and math.isnan(kyori_m)):
        return None
    band = None
    for lower, name in DISTANCE_BANDS:
        if kyori_m >= lower:
            band = name
    return band


def baba_cd(value: str | None) -> str | None:
    """馬場状態コード。1 良 / 2 稍重 / 3 重 / 4 不良。0 や空白は欠損。"""
    value = (value or "").strip()
    return value if value in keiba_config.BABA_CD_TO_NAME else None


def sex_cd(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if value in keiba_config.SEX_CD_TO_NAME else None


def age_from_barei(barei: str | int | None, race_year: int, *,
                   kazoe_until_year: int) -> int | None:
    """表記上の馬齢を満年齢にする。数え年の時代は 1 を引く。

    0 や負は「不明」で、年齢として扱わない（既存 repository の判断と同じ）。
    """
    try:
        value = int(str(barei).strip())
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    age = value - 1 if race_year <= kazoe_until_year else value
    return age if age >= 1 else None

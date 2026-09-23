"""parse 済みの ``RaceInput`` の妥当性を確認する。

schema.py が「理解できるか」を見るのに対し、こちらは「理解はできたが中身が
おかしくないか」を見る。DB を必要とする照合（馬名のマスタ突き合わせなど）は
ここでは行わない。``RaceInput`` 自身から分かることだけを見る。それ以上は
Task 8（マスタ突き合わせ）の仕事。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from keiba import config
from keiba.models import RaceInput

__all__ = ["TOP_N_EXCEEDS_ENTRIES", "Problem", "check"]

#: ``top_n`` が出走頭数を超えているという問題の目印。メッセージの先頭に置く。
#:
#: ``keiba.evaluate.backtest`` がこの文字列でスキップ理由を分類する
#: （「n 頭に足りないレース」は入力の書式が壊れているわけではないので、
#: ``input_invalid`` ではなく ``single_starter`` 相当に数える。fix5 決定10）。
#: 同じ日本語文を2箇所に書かないため、判定側もこの定数を参照する。
TOP_N_EXCEEDS_ENTRIES = "top_n が出走頭数を超えています"


@dataclass(frozen=True)
class Problem:
    """``check()`` が見つけた1件の問題。"""

    field: str
    message: str


def _is_int(value: object) -> bool:
    """量として使える int か。bool は int のサブクラスなので明示的に除く。"""
    return isinstance(value, int) and not isinstance(value, bool)


def check(race_input: RaceInput) -> list[Problem]:
    """``race_input`` の問題をすべて集めて返す。空なら予想を開始できる。

    利用者が1件直しては再提出する往復を無駄にしないため、最初に見つかった
    問題で止めず、最後まで見て集めたものを返す。

    契約は「どんな RaceInput でも問題を並べて返す（raise しない）」。
    ``race_input`` は ``schema.parse_race_input`` を経由したものとは限らない
    （テストや将来のタスクが ``RaceInput`` を直接組み立てることもある）ため、
    型が想定と違う値が来ても比較・イテレート・集合操作で例外を出さず、
    その項目についての Problem に変えて報告する。個々の比較の前に型を
    guard する方式を徹底し、``check()`` 全体を ``try/except`` で覆うことは
    しない（クラッシュを空リストに変えてしまい、本物のバグまで隠すため）。

    ここで検査しているのは「意味的な妥当性のルールがある項目」だけ。
    ルールの無い項目（``date`` / ``race_num`` / ``jyo_cd`` / ``baba_cd`` /
    ``class_name`` / ``baba`` / ``race_id`` と、馬側の ``wakuban`` /
    ``sex_age`` / ``futan_kg`` / ``kisyu``）は、もとから何も見ていない
    （見ていないので、どんな値が来ても検査自体が発生せず、それ自体で
    安全）。
    """
    problems: list[Problem] = []

    if not isinstance(race_input.jyo, str):
        problems.append(Problem("jyo", f"jyo は文字列である必要があります: {race_input.jyo!r}"))
    elif not race_input.jyo.strip():
        problems.append(Problem("jyo", "開催場が指定されていません。"))

    if not isinstance(race_input.track_kind, str):
        problems.append(
            Problem("track_kind", f"track_kind は文字列である必要があります: {race_input.track_kind!r}")
        )
    elif not race_input.track_kind.strip():
        problems.append(Problem("track_kind", "track_kind が指定されていません。"))

    # horses 自体の型を最初に確定させる。tuple/list でなければ以降の
    # len()・イテレートがどれも例外になりうるので、空の () に落として
    # 以降の処理を安全にする（race_input.horses は元の値のまま参照しない）。
    horses = race_input.horses
    if not isinstance(horses, (tuple, list)):
        problems.append(Problem("horses", f"horses は配列である必要があります: {horses!r}"))
        horses = ()
    elif not horses:
        problems.append(Problem("horses", "出走馬が1頭もいません。"))

    if not _is_int(race_input.kyori_m):
        problems.append(Problem("kyori_m", f"kyori_m は整数である必要があります: {race_input.kyori_m!r}"))
    elif race_input.kyori_m <= 0:
        problems.append(Problem("kyori_m", f"距離は正の値である必要があります: {race_input.kyori_m}"))
    elif not (config.MIN_DISTANCE_M <= race_input.kyori_m <= config.MAX_DISTANCE_M):
        problems.append(
            Problem(
                "kyori_m",
                f"距離が現実的な範囲外です: {race_input.kyori_m}"
                f"（{config.MIN_DISTANCE_M}〜{config.MAX_DISTANCE_M}）",
            )
        )

    if not _is_int(race_input.entry_count):
        problems.append(
            Problem("entry_count", f"entry_count は整数である必要があります: {race_input.entry_count!r}")
        )
    elif race_input.entry_count != len(horses):
        problems.append(
            Problem(
                "entry_count",
                f"entry_count ({race_input.entry_count}) と馬の数 "
                f"({len(horses)}) が一致しません。",
            )
        )

    # prediction_cutoff の唯一のルールは「hasso_at より前」であることなので、
    # hasso_at が None（＝未指定。それ自体は妥当）のときは何も比較しない。
    # hasso_at が指定されているときだけ、比較の前に両方が datetime かを
    # 確認する。文字列や数値を datetime と比較すると TypeError になるため。
    if race_input.hasso_at is not None:
        if not isinstance(race_input.hasso_at, datetime):
            problems.append(
                Problem("hasso_at", f"hasso_at は日時である必要があります: {race_input.hasso_at!r}")
            )
        elif not isinstance(race_input.prediction_cutoff, datetime):
            problems.append(
                Problem(
                    "prediction_cutoff",
                    f"prediction_cutoff は日時である必要があります: {race_input.prediction_cutoff!r}",
                )
            )
        elif race_input.prediction_cutoff >= race_input.hasso_at:
            problems.append(
                Problem("prediction_cutoff", "prediction_cutoff は hasso_at より前である必要があります。")
            )

    seen_umaban: set[str] = set()
    for horse in horses:
        # umaban は文字列でなければ以降の set 操作（重複判定）が
        # unhashable な型（list / dict）でクラッシュしうるので、
        # 文字列であることを確認してから重複判定・整数変換に進む。
        if not isinstance(horse.umaban, str):
            problems.append(Problem("horses", f"umaban は文字列である必要があります: {horse.umaban!r}"))
        else:
            if horse.umaban in seen_umaban:
                problems.append(Problem("horses", f"umaban が重複しています: {horse.umaban!r}"))
            seen_umaban.add(horse.umaban)

            try:
                umaban_value = int(horse.umaban)
            except ValueError:
                problems.append(Problem("horses", f"umaban を整数として読めません: {horse.umaban!r}"))
            else:
                if umaban_value <= 0:
                    problems.append(
                        Problem("horses", f"umaban は正の整数である必要があります: {horse.umaban!r}")
                    )

        if not isinstance(horse.bamei, str):
            problems.append(Problem("horses", f"馬名が文字列ではありません: umaban={horse.umaban!r}"))
        elif not horse.bamei.strip():
            problems.append(Problem("horses", f"馬名が空です: umaban={horse.umaban!r}"))

    if not _is_int(race_input.top_n):
        problems.append(Problem("top_n", f"top_n は整数である必要があります: {race_input.top_n!r}"))
    elif race_input.top_n < 1:
        problems.append(Problem("top_n", f"top_n は1以上である必要があります: {race_input.top_n}"))
    elif race_input.top_n > len(horses):
        problems.append(
            Problem(
                "top_n",
                f"{TOP_N_EXCEEDS_ENTRIES}: top_n={race_input.top_n}"
                f" 出走頭数={len(horses)}。",
            )
        )

    return problems

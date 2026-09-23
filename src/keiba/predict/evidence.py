"""第2層の分類 (``AxisAnswer.choice``) と、features 側の自前集計
(``HorseFeatures.by_condition`` など) から、定型文の証拠を組む。

Jev は自由文の理由を生成しない (spec §8)。ここで作る文言はすべて固定
テンプレート + Python 側の集計値であり、モデルの choice/probabilities は
「どのテンプレートをどちらの束 (positives/negatives/insufficient) に入れるか」
を決めるための分類としてだけ使う。「表示文言の組み立て」という単一の関心事
なので、``predict`` のオーケストレーション (predict/pipeline.py) とはファイルを
分けてある。
"""

from __future__ import annotations

from collections.abc import Mapping

from keiba.features.context import HorseFeatures
from keiba.features.derive import ConditionRecord
from keiba.predict.layers import AxisAnswer
from keiba.predict.questions import AXIS_SPECS

__all__ = ["EvidenceBuilder"]

#: by_condition のキー → 表示ラベル。derive.CONDITION_KEYS と同じ5種。
_CONDITION_LABELS: dict[str, str] = {
    "exact_distance": "同距離",
    "same_distance": "同距離帯",
    "same_venue": "同競馬場",
    "same_track_kind": "同トラック",
    "same_baba": "同馬場状態",
}

#: distance_fit / course_going_fit の指示文 (questions.py) が明示的に参照する
#: by_condition のキー。この2軸だけが「条件別成績」を根拠に持つ。
_AXIS_CONDITION_KEYS: dict[str, tuple[str, ...]] = {
    "distance_fit": ("exact_distance", "same_distance"),
    "course_going_fit": ("same_venue", "same_track_kind", "same_baba"),
}

#: 軸レベルの evidence (条件別成績を持たない3軸、および insufficient) に使う
#: 表示ラベル。
_AXIS_LABELS: dict[str, str] = {
    "distance_fit": "距離適性",
    "course_going_fit": "コース・馬場適性",
    "relative_ability": "相対能力",
    "current_form": "現在の状態",
    "transport_concern": "輸送負担",
}

# choice の値が「良い」「悪い」のどちらを意味するか。
#
# AXIS_SPECS.positive_key / negative_key は Task 14 が signed 特徴
# (P(positive) - P(negative)) を作るための命名であり、evidence の良し悪しとは
# 別の軸。transport_concern がその典型で、positive_key="concern" は実際には
# 悪材料を指す (「懸念の確率をそのまま signed にする」ための命名)。ここでは
# evidence 用に「良い/悪い」を choice の値で明示的に持つ。transport_concern に
# 「良い」choice が無いのは、no_evidence (懸念を示す材料が無い) は「懸念が
# 無いというだけ」で好材料の断定にはならないため (positives には積まない)。
_GOOD_CHOICE: dict[str, str] = {
    "distance_fit": "positive",
    "course_going_fit": "positive",
    "relative_ability": "superior",
    "current_form": "positive",
}
_BAD_CHOICE: dict[str, str] = {
    "distance_fit": "negative",
    "course_going_fit": "negative",
    "relative_ability": "inferior",
    "current_form": "negative",
    "transport_concern": "concern",
}


def _condition_phrase(key: str, record: ConditionRecord) -> str:
    """「同距離 8戦3勝(3着内5)」。必ず出走数を添える (spec §4.3: 率だけでは
    出走数を伴わない別の主張になる)。

    定義できない条件には使えない。件数の無いものに件数の形を与えないため、
    黙って None を埋め込むのではなく落とす。
    """
    if record.inapplicable_reason is not None:
        raise ValueError(
            f"{key} は集計が成立していないので件数の文言は作れない: "
            f"{record.inapplicable_reason}")
    return (f"{_CONDITION_LABELS[key]} {record.starts}戦{record.wins}勝"
            f"(3着内{record.top3})")


class EvidenceBuilder:
    """第2層の分類と、features 側の自前集計から、定型文の証拠を組む。"""

    @staticmethod
    def build(
        hf: HorseFeatures,
        answers: Mapping[str, AxisAnswer] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        positives: list[str] = []
        negatives: list[str] = []
        insufficient: list[str] = []

        # 条件別成績 (by_condition): starts == 0 の集計は、軸の分類に関わらず
        # 判断材料不足として扱う——「今回の条件と比較できる実績が無い」こと
        # 自体が理由で、Jev の分類結果とは独立に決まる事実だから。
        #
        # 守るべき不変条件は「この2つのガードを answer による分岐 (下の
        # `if answer is None` 以降) より先に置くこと」だけ。inapplicable_reason
        # と starts == 0 のどちらを先に見るかは結果を変えない——どちらも
        # continue するだけのガードなので、`None == 0` が False でも隣の
        # ガードがそのまま拾う。危険なのはガードを answer 側の分岐より下に
        # 動かすことで、そのときだけ starts が None の inapplicable な記録が
        # `_condition_phrase` まで渡って ValueError を投げる。
        #
        # inapplicable_reason が None でない記録はレース単位の事実（例:
        # 今回の馬場状態が未入力）であって、馬についての事実ではない。
        # ここで文言化すると同じ一文が出走馬の数だけ繰り返されてしまうので、
        # 3つの束のどれにも入れず continue するだけにする（fix3 決定1）。
        # レース単位の事実は features/context.py が ctx.notes に1回だけ出す。
        # 情報は失われない——下の軸レベルのループが「コース・馬場適性:
        # 判断材料不足」を出すので、利用者は軸が判断材料不足であることを知れる。
        for axis_name, condition_keys in _AXIS_CONDITION_KEYS.items():
            answer = answers.get(axis_name) if answers is not None else None
            for key in condition_keys:
                record = hf.by_condition.get(key)
                if record is None:
                    continue
                if record.inapplicable_reason is not None:
                    continue
                if record.starts == 0:
                    insufficient.append(_condition_phrase(key, record))
                    continue
                if answer is None:
                    continue
                if answer.choice == _GOOD_CHOICE.get(axis_name):
                    positives.append(_condition_phrase(key, record))
                elif answer.choice == _BAD_CHOICE.get(axis_name):
                    negatives.append(_condition_phrase(key, record))

        # 軸レベル: 「判断材料不足」が最尤だった軸 (5軸すべて対象)。
        # 条件別成績を持たない3軸は、良い/悪いの分類もここで一緒に処理する。
        if answers is not None:
            for axis_name, spec in AXIS_SPECS.items():
                answer = answers.get(axis_name)
                if answer is None:
                    continue
                if answer.choice == spec.insufficient_key:
                    insufficient.append(f"{_AXIS_LABELS[axis_name]}: 判断材料不足")
                    continue
                if axis_name in _AXIS_CONDITION_KEYS:
                    continue  # 上のループで扱い済み (条件別成績の言い回しを使う)
                if answer.choice == _GOOD_CHOICE.get(axis_name):
                    positives.append(_AXIS_LABELS[axis_name])
                elif answer.choice == _BAD_CHOICE.get(axis_name):
                    negatives.append(_AXIS_LABELS[axis_name])

        return tuple(positives), tuple(negatives), tuple(insufficient)

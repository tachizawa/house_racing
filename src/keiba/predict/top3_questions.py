"""3着以内を聞く質問文（report3 §6 のたたき台を state のキー名に合わせたもの）。

Jev は同じリクエスト内の質問を独立・並列に評価する。質問を並べても前の答えは
後の質問に渡らない（report3 §3.4）。案Bの展開予測は Python が受け取って
state の pace_forecast に入れる。

文言を変えたら TOP3_PROMPT_VERSION を上げる。記録に必ず残る。
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, NoulCriteria

from keiba.predict.questions import LAYER1_PACE_INSTRUCTIONS, PACE_CHOICES

__all__ = ["TOP3_PROMPT_VERSION", "pace_questions", "question_name", "top3_questions"]

TOP3_PROMPT_VERSION = "1"

_MATERIALS = (
    "判断材料：\n"
    "course：今回のコースの基本情報（特定できない場合は identified が false）\n"
    "race：今回のレース条件\n"
    "horses：出走全頭の過去走（recent_runs）、条件別実績（conditions）、"
    "能力指標（ability）、脚質（running_style）、過去走の有無と理由（history）\n"
    "field_summary：出走全頭の概要\n"
    "data_quality：欠損とデータ版の情報\n"
)

_PACE_LINE = (
    "pace_forecast：前段で予測した展開（ペース区分の確率）。"
    "未確定の見立てとして考慮し、確定した事実として扱わないでください。\n"
)

def _rules(whose_runs: str) -> str:
    """判断の規則。winner には対象馬が無いので、過去走の主語だけ差し替える
    （final review I-3b）。"""
    return (
        "\n"
        f"今回のコース条件、{whose_runs}の過去走、出走全頭との能力比較を判断材料にしてください。\n"
        "条件別実績は出走数（starts）と比較条件（definition）を併せて参照してください。\n"
        "未経験の条件や欠損情報を、不適性の証拠として扱わないでください。"
        "history.status が debut / not_in_window / unknown / unmatched の馬は、"
        "過去走が無いことを能力が低い証拠として扱わないでください。"
        "found_may_be_truncated の馬は、表示されている過去走が一部だけの可能性があります。\n"
        "馬名の知名度ではなく、入力された情報を根拠にしてください。"
    )


_TOP3_RULES = _rules("対象馬")
_WINNER_RULES = _rules("各馬")

_CRITERIA = NoulCriteria(
    true="対象馬が今回のレースで確定着順3着以内に入る。",
    false="対象馬が今回のレースで4着以下になる、または競走中止・失格などで3着以内に入らない。",
)


def question_name(umaban: str) -> str:
    return f"top3_{umaban}"


def _top3_instructions(umaban: str, *, with_pace_forecast: bool) -> str:
    return (
        f"対象馬: {umaban}\n"
        "予想締切までに得られた情報に基づき、対象馬（horses の上記馬番）が"
        "今回のレースで確定着順3着以内に入るかを予測してください。\n"
        "\n"
        + _MATERIALS
        + (_PACE_LINE if with_pace_forecast else "")
        + _TOP3_RULES
    )


def _winner_instructions(*, with_pace_forecast: bool) -> str:
    return (
        "予想締切時点の情報に基づき、今回のレースの1着馬を予測してください。\n"
        "\n"
        + _MATERIALS
        + (_PACE_LINE if with_pace_forecast else "")
        + _WINNER_RULES
    )


def _winner_criteria(dossier: dict) -> dict[str, str]:
    criteria = {}
    for umaban, h in dossier["horses"].items():
        parts = [h["bamei"]]
        if h.get("sex_age"):
            parts.append(h["sex_age"])
        if h.get("futan_kg") is not None:
            parts.append(f"{h['futan_kg']}kg")
        criteria[umaban] = " ".join(parts)
    return criteria


def top3_questions(dossier: dict, *, with_pace_forecast: bool) -> dict:
    """全頭の top3_XX（Noul）と、同じ state での winner（Choice）。

    winner は「同じ state で勝率上位を選んだら」の比較対象（report3 §7-2）。
    同じ state への質問なので同じリクエストにまとめる（増えるのは質問文のトークンだけ）。
    """
    questions: dict = {
        question_name(umaban): Noul(
            instructions=_top3_instructions(umaban, with_pace_forecast=with_pace_forecast),
            criteria=_CRITERIA)
        for umaban in dossier["horses"]
    }
    questions["winner"] = Choice(
        instructions=_winner_instructions(with_pace_forecast=with_pace_forecast),
        criteria=_winner_criteria(dossier))
    return questions


def pace_questions() -> dict:
    """案Bの1回目。既存の pace 区分と文言を使う（state に pace_definition が要る）。"""
    return {"pace": Choice(instructions=LAYER1_PACE_INSTRUCTIONS, criteria=PACE_CHOICES)}

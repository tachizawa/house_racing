"""Jev へ送る質問文（第1層・第2層）とペース区分の選択肢定義。

ここに書く文言は長い設計議論の結果そのもので、実装時に言い換えない。
わずかな言い回しの違いでも、型は保ったまま答えだけがもっともらしく
ずれうる——構造化出力なのでハルシネーションは起きないが、質問文の質は
コードでは検証できない。

I/O・特徴量計算はしない（それぞれ jev_client.py / features/*.py の仕事）。
ここは Choice の組み立てとプロンプト文字列だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass

from typesafe_sdk import Choice

from keiba.features.context import PredictionContext
from keiba.features.summary import horse_criteria

#: この文言のもとで得た予想と、別の文言・別の版で得た予想を混同しないための
#: 版番号。予想記録に必ず残す（CLAUDE.md「プロンプトが変わった予想は別の
#: 記録として扱う」、元仕様 §8 / §12.7）。
#:
#: "2": 第1層の2つの文言（LAYER1_WINNER_INSTRUCTIONS / LAYER1_PACE_INSTRUCTIONS）
#:      の末尾を ASCII の "." から仕様どおりの "。" に直した（fix5 決定10）。
#:      **"1" の記録は実在する**——data/predictions に旧文言で得た記録が
#:      保存済みで、新文言の記録と同じファイルに並んでいる（実測 291件 / 100件）。
#:      文言が違えば同じ質問名でも答えの意味が同じとは言えないので、版を分けて
#:      機構に見せる: scripts/fit_weights.py の require_consistent_versions() が
#:      2世代の混在を拒否し、Weights.check_compatible() が旧版の重みを拒否する。
#:      **元の "." に戻して版を据え置く選択は採らない**——仕様に合わせた字面を、
#:      版番号の偽の主張を維持するために巻き戻すのは順序が逆（fix6 ①）。
#:
#: "3": course_going_fit の insufficient の説明から「馬場状態が未入力の場合も
#:      含む。」を外した。**"2" の記録も実在する**（data/predictions の 391件）。
#:      句点の直しと違い、これは**選択肢の意味そのものの変更**——"2" の
#:      insufficient は「この馬の実績が乏しい」と「今回のレースの馬場が未入力」
#:      の2つを吸っていて、"3" は前者だけを指す。同じ
#:      course_going_fit_insufficient という列名でも中身が違うので、版を分けて
#:      混在を機構に拒否させる（require_consistent_versions / check_compatible）。
PROMPT_VERSION = "3"

# --- 第1層 --------------------------------------------------------------------

#: 元仕様 §5.2 の文面をそのまま使う。
LAYER1_WINNER_INSTRUCTIONS = (
    "予想締切時点の情報に基づき、今回のレースの1着馬を予測してください。\n"
    "\n"
    "判断材料：\n"
    "race：今回のレース条件\n"
    "horses：出走全頭の能力指標、近走内容、条件別実績、今回の出走条件\n"
    "\n"
    "今回の出走馬同士を比較してください。\n"
    "過去の着順だけでなく、クラス、着差、条件の違いを考慮してください。\n"
    "条件別実績は、成績と出走数の両方を参照してください。\n"
    "欠損している情報を、良い実績や悪い実績として扱わないでください。\n"
    "馬名の知名度ではなく、入力された情報を根拠にしてください。"
)

# design doc §8.2（2026-09-22 修正版）の文面。元仕様 §5.3 の草稿は
# 「horses.running_style_summary」という、state に存在しないキーを参照していた
# （実際のキーは horses[umaban].running_style）。草稿を後から確定した state の
# キー名（features/summary.py）と突き合わせていなかったための不整合で、
# 検証済みキー名に差し替えてある。verbatim の原則より、state に実在しない
# キーを参照させない正しさを優先する（task-12 コーディネーターの裁定）。
LAYER1_PACE_INSTRUCTIONS = (
    "race と、horses の各馬の running_style を参照し、\n"
    "今回のレース前半のペース区分を予測してください。\n"
    "\n"
    "区分の意味は pace_definition に従ってください。\n"
    "出走全頭の先行傾向、枠順（waku）、距離、コース条件を判断材料にしてください。"
)

#: pace 質問の選択肢。区分の意味そのものは pace_definition（features/pace.py）
#: が計算する fast_max / standard_max に従う——ここで固定の秒数を書かない。
PACE_CHOICES: dict[str, str] = {
    "fast": "前半3ハロンが速く入る区分。pace_definition の fast_max 以下に相当する。",
    "standard": (
        "前半3ハロンが標準的な区分。"
        "pace_definition の fast_max を超え standard_max 以下に相当する。"
    ),
    "slow": "前半3ハロンが遅く入る区分。pace_definition の standard_max を超える区分。",
}


def layer1_questions(ctx: PredictionContext, *, with_pace: bool = False) -> dict[str, Choice]:
    """第1層の質問。winner は常に含み、pace は要求されたときだけ足す。

    winner と pace は同じ state（レース単位のサマリ）に対する質問なので、
    別リクエストにせず同じ辞書にまとめる——state を送り直す分がそのまま
    無駄になる（CLAUDE.md「リクエストは state が変わるところで割る」）。
    """
    result: dict[str, Choice] = {
        "winner": Choice(
            instructions=LAYER1_WINNER_INSTRUCTIONS,
            criteria=horse_criteria(ctx),
        ),
    }
    if with_pace:
        result["pace"] = Choice(
            instructions=LAYER1_PACE_INSTRUCTIONS,
            criteria=PACE_CHOICES,
        )
    return result


# --- 第2層 --------------------------------------------------------------------


@dataclass(frozen=True)
class AxisSpec:
    """第2層の1軸ぶんの質問定義。

    positive_key / negative_key は Task 14 が signed 特徴（P(positive) -
    P(negative)）を作るときに使う。negative_key が None の軸（2値+不足の
    transport_concern）は差を取らず、positive_key の確率をそのまま使う。
    """

    instructions: str
    criteria: dict[str, str]
    positive_key: str | None
    negative_key: str | None
    insufficient_key: str


AXIS_SPECS: dict[str, AxisSpec] = {
    "distance_fit": AxisSpec(
        instructions=(
            "target_horse の過去の距離別成績と近走内容に照らして、"
            "今回の距離は能力発揮にとってどの区分の材料ですか。\n"
            "by_condition の exact_distance と same_distance は、"
            "成績と starts（出走数）の両方を見てください。\n"
            "その距離での出走経験がないことを、不適性の材料として扱わないでください。"
        ),
        criteria={
            "positive": "今回に近い距離で、比較可能な条件下の好成績が裏付けられている。",
            "neutral": "比較可能な材料はあるが、今回の距離を明確な有利・不利とする材料が乏しい。",
            "negative": "比較可能な実績に、今回の距離で能力発揮を妨げる懸念が認められる。",
            "insufficient": "比較に使える出走実績や判断材料が乏しい。",
        },
        positive_key="positive", negative_key="negative",
        insufficient_key="insufficient",
    ),
    "course_going_fit": AxisSpec(
        instructions=(
            "target_horse のコース別・馬場状態別の実績に照らして、"
            "今回の競馬場、芝・ダート、馬場状態はどの区分の材料ですか。\n"
            "by_condition の same_venue、same_track_kind、same_baba を、"
            "成績と starts（出走数）の両方で見てください。\n"
            "race.baba が入力されていない場合は、馬場状態を判断材料にしないでください。\n"
            "その条件での出走経験がないことを、不適性の材料として扱わないでください。"
        ),
        criteria={
            "positive": "今回と比較可能なコース・馬場状態での好成績が裏付けられている。",
            "neutral": "比較可能な材料はあるが、今回の条件を明確な有利・不利とする材料が乏しい。",
            "negative": "比較可能な実績に、今回のコースまたは馬場状態での懸念が認められる。",
            # 馬場状態が未入力であることをここに吸わせない。それはレース単位の
            # 事実で、この馬の実績が乏しいことではない（report2 §4）。未入力の
            # 馬場を判断材料にしないという指示は instructions 側に残してある。
            "insufficient": "比較に使える出走実績や判断材料が乏しい。",
        },
        positive_key="positive", negative_key="negative",
        insufficient_key="insufficient",
    ),
    "relative_ability": AxisSpec(
        instructions=(
            "target_horse の近走の対戦水準と内容を opponents_summary の全頭と比較して、"
            "実績上の能力はどの区分ですか。\n"
            "ability の recent_margin_sec_wavg（直近走の加重平均着差、小さいほど優秀）と "
            "recent_class_level_wavg（クラスの水準）を併せて見てください。\n"
            "着差だけ、クラスだけで判断しないでください。"
            "低いクラスでの小さい着差と、高いクラスでの大きい着差を区別してください。\n"
            "runs_used が少ない馬を、それだけで劣位としないでください。"
        ),
        criteria={
            "superior": "比較可能な実績で、相手の多くより高い能力が裏付けられている。",
            "comparable": "相手と同程度で、優劣を決める材料が乏しい。",
            "inferior": "比較可能な実績で、相手の多くより低い能力が示されている。",
            "insufficient": "比較に使える出走実績や判断材料が乏しい。",
        },
        positive_key="superior", negative_key="inferior",
        insufficient_key="insufficient",
    ),
    "current_form": AxisSpec(
        instructions=(
            "recent の days_since_last_run（前走からの日数）、近走の着差の推移、"
            "last_run の馬体重に照らして、今回の状態にはどの区分の材料がありますか。\n"
            "馬体重は前走時点の値です。今回の馬体重は予想締切時点で判明していません。\n"
            "日数が長いこと自体を不調の材料として扱わないでください。"
            "休養明けで好走する馬もいます。"
        ),
        criteria={
            "positive": "近走の内容または間隔に、状態が整っていることを示す材料がある。",
            "neutral": "材料はあるが、状態を明確な上向き・下降とする根拠が乏しい。",
            "negative": "近走の内容または間隔に、状態面の懸念を示す材料がある。",
            "insufficient": "状態を判断できる近走や確認済みの情報が乏しい。",
        },
        positive_key="positive", negative_key="negative",
        insufficient_key="insufficient",
    ),
    "transport_concern": AxisSpec(
        instructions=(
            "transport の確認済み情報から、今回の輸送についてどの区分ですか。\n"
            "estimated_transport_km は推定値で、実際の出発地・経路・輸送日を"
            "示すものではありません。\n"
            "stay_status が unknown の場合、当日輸送とも現地滞在とも決めつけないでください。\n"
            "past_long_transport の馬体重増減は starts（該当走数）と併せて見てください。"
            "体重変化の原因が輸送であるとは限りません。"
        ),
        criteria={
            "concern": "確認済みの輸送・滞在情報と過去の関連実績に、今回の輸送への懸念を示す材料がある。",
            "no_evidence": "確認できる範囲で、今回の輸送への懸念を示す材料がない。",
            "insufficient": "輸送・滞在の確認済み情報や関連する過去実績が乏しく、判断できない。",
        },
        # 2値+不足なので差を取らない。signed は P(concern) そのものになる。
        positive_key="concern", negative_key=None,
        insufficient_key="insufficient",
    ),
}

AXIS_NAMES: tuple[str, ...] = tuple(AXIS_SPECS)


def layer2_questions() -> dict[str, Choice]:
    """第2層の5軸。1頭1リクエストに5問すべてをまとめる。

    軸ごとにリクエストを分けない——1頭に対する state は同じなので、まとめた
    ほうが安くて速い（CLAUDE.md、公式の並列質問クックブック）。馬ごとの
    リクエスト分割（state が変わる境界）は呼び出し側が担う。
    """
    return {
        name: Choice(instructions=spec.instructions, criteria=spec.criteria)
        for name, spec in AXIS_SPECS.items()
    }


def feature_names_for_axes() -> tuple[str, ...]:
    """第2層の回答を数値特徴にするときの列名。10列、常に同じ順序。

    「判断材料不足」を「中立」と混ぜないため、1軸から signed / insufficient の
    2特徴を作る（元仕様 §9）。順序は AXIS_SPECS の定義順で固定し、Task 14 の
    重みベクトルとの対応がぶれないようにする。
    """
    names: list[str] = []
    for name in AXIS_SPECS:
        names.append(f"{name}_signed")
        names.append(f"{name}_insufficient")
    return tuple(names)

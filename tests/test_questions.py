"""keiba.predict.questions のテスト。

ここでのプロンプト文字列そのものは長い議論を経て確定したものなので、
verbatim であることは self-review の文字diffで別途確認する。テストが
見るのは「型として守るべき制約」（choice/dict/文言に含むべき語）と、
「軸を追加しても崩れない構造」（feature_names_for_axes の安定した順序、
insufficient と neutral の分離）。
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
from typesafe_sdk import Choice

from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.features.context import build_context
from keiba.features.summary import build_horse_state, build_race_state
from keiba.models import HorseEntry, RaceInput
from keiba.predict import questions as q
from tests.fixtures import csv_data as fx

# scripts/ は keiba パッケージの外にあり pytest.ini の pythonpath にも入って
# いないため、tests/test_pipeline.py と同じ方法（sys.path への追加）で import
# する。「送られる質問」を決める規則（pace_definition が無ければ pace 質問を
# 送らない）は CLI 側が持つので、その判定そのものをここで呼ぶ——テスト側に
# 同じ規則を書き写すと、片方だけ直されたときに検出できない。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import predict as predict_cli  # noqa: E402

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


def make_entry(*, umaban="01", bamei="テストウマ", **kw) -> HorseEntry:
    return HorseEntry(umaban=umaban, bamei=bamei, **kw)


def make_race_input(horses, **kw) -> RaceInput:
    defaults = dict(
        date=date(2026, 3, 21),
        jyo="中山",
        race_num=11,
        track_kind="turf",
        kyori_m=2000,
        entry_count=len(horses),
        prediction_cutoff=jst(2026, 3, 20, 20, 0, 0),
        horses=tuple(horses),
        jyo_cd="06",
        baba_cd="1",
    )
    defaults.update(kw)
    return RaceInput(**defaults)


def make_context(entries):
    """馬名照合を全頭「未登録」に固定した最小の PredictionContext。

    layer1_questions / horse_criteria が見るのは entry（馬名・性齢・斤量）と
    umaban の集合だけなので、過去走データは不要——本物の Repository/CSV を
    経由せずモックで済ませ、質問定義側のテストを feature 計算のテストから
    切り離す（test_context.py の手法と同じ）。
    """
    mock_repo = mock.create_autospec(Repository, instance=True)
    mock_repo.find_horses_by_name.return_value = []
    race_input = make_race_input(entries)
    return build_context(race_input, mock_repo)


@pytest.fixture
def ctx_three_horses():
    entries = [
        make_entry(umaban="01", bamei="イチバンウマ", sex_age="牡4", futan_kg=57.0),
        make_entry(umaban="02", bamei="ニバンウマ", sex_age="牝5", futan_kg=54.0),
        make_entry(umaban="03", bamei="サンバンウマ", sex_age="セン6", futan_kg=58.0),
    ]
    return make_context(entries)


# --- 第1層 ---------------------------------------------------------------------

def test_layer1_has_winner_question(ctx_three_horses):
    result = q.layer1_questions(ctx_three_horses)
    assert set(result) == {"winner"}


def test_layer1_adds_pace_when_requested(ctx_three_horses):
    result = q.layer1_questions(ctx_three_horses, with_pace=True)
    assert set(result) == {"winner", "pace"}


def test_winner_criteria_keys_are_umaban(ctx_three_horses):
    result = q.layer1_questions(ctx_three_horses)
    assert set(result["winner"].criteria) == {"01", "02", "03"}


def test_winner_criteria_within_255_choices(ctx_three_horses):
    result = q.layer1_questions(ctx_three_horses)
    assert len(result["winner"].criteria) <= 255


def test_winner_instructions_mention_starts_count():
    assert "出走数" in q.LAYER1_WINNER_INSTRUCTIONS


def test_winner_instructions_forbid_treating_missing_as_signal():
    assert "欠損" in q.LAYER1_WINNER_INSTRUCTIONS


def test_pace_criteria_has_three_bands():
    assert set(q.PACE_CHOICES) == {"slow", "standard", "fast"}


def test_pace_prompt_only_references_keys_that_exist_in_the_state(ctx_three_horses):
    """プロンプトが参照するキーは state に実在するものだけにする。

    存在しないキーを参照させると、モデルはその指示を無視するか自分で材料を
    選ぶ。どちらも例外を出さず出力はもっともらしいままなので、プロンプト文と
    実際の state のキー集合を突き合わせるテストでしか捕まらない
    （LAYER1_PACE_INSTRUCTIONS が実在しない running_style_summary を参照して
    いた実例があったため、同じ種類の回帰を防ぐ）。
    """
    state = build_race_state(ctx_three_horses)
    horse_keys = set(next(iter(state["horses"].values())))
    for key in ("running_style", "waku", "ability", "recent", "by_condition", "transport"):
        assert key in horse_keys

    assert "running_style" in q.LAYER1_PACE_INSTRUCTIONS
    assert "running_style_summary" not in q.LAYER1_PACE_INSTRUCTIONS
    for name in ("race", "horses", "pace_definition"):
        assert name in q.LAYER1_PACE_INSTRUCTIONS


# --- 全プロンプトの横断チェック --------------------------------------------------
#
# 上の test_pace_prompt_only_references_keys_that_exist_in_the_state は pace の
# 1問だけを手で確認したもの。だが今回の不具合（running_style_summary）が示した
# 教訓は「この種の欠陥は無音で、通常のテストでは捕まらない」ことそのもの。
# 手書きの「プロンプト→参照するキー」対応表を別途持つのも同じ問題を抱える
# ——プロンプト本文を直して表を直し忘れた瞬間にずれる。そこで、プロンプト
# 本文からキーらしいトークンを正規表現で抽出し、実際の state のキー集合と
# 機械的に突き合わせる。プロンプト側の書き換えも state 側の改名も、どちらも
# このテストが自動的に捕まえる。

#: state のキーはすべて ASCII の snake_case。プロンプト本文は日本語なので、
#: この正規表現はキー名らしいトークンだけを拾う。
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{2,}")

#: キーではないが本文に出てくる ASCII 語。値や英単語はここに挙げ、
#: 「キーでない理由」を必ずコメントで添える。増えていくようなら、
#: それは正規表現が緩すぎてテストが骨抜きになっている兆候。
_NOT_A_KEY = frozenset({
    "unknown",  # stay_status の既定値（キー名ではなく値。derive.py の
                # transport(stay_status: str = "unknown") を参照）
})


def _state_key_names(obj) -> set[str]:
    """dict/list を再帰的にたどり、あらゆる階層のキー名を集める。"""
    names: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            names.add(str(key))
            names |= _state_key_names(value)
    elif isinstance(obj, list):
        for item in obj:
            names |= _state_key_names(item)
    return names


#: maximal_ctx が渡すペース定義（features/pace.fit_pace_definition の戻り値と
#: 同じ形）。層別化に必要なレース数が足りないとき、これは None になる。
_PACE_DEFINITION = {
    "fast_max": 33.0, "standard_max": 36.0,
    "stratum": {"track_kind": "turf"}, "race_count": 40,
    "note": "テスト用の定義。",
}


def _build_maximal_ctx(tmp_path, *, pace_definition):
    """state のあらゆる任意キーが埋まった PredictionContext。

    ``pace_definition=None`` は「層別化の30レースに届かなかった」状態
    （§8.2 自身が「取得初期は成立しない前提」と書いている状況）。この場合
    state に "pace_definition" キーは存在しない。

    登録済みで過去走のある馬を1頭作る（ability / by_condition / recent /
    running_style / transport が「未計算」ではなく実際の値で埋まる）。
    pace_definition も渡す。build_horse_state の race_context は state の
    キーではあるが構築時の任意引数なので、テスト側で別途渡す。

    ここを薄い（未登録馬だけの）context にすると、本当は存在するキーまで
    「state に無い」と誤判定してテストが的外れに失敗する——このフィクス
    チャは常に「最大構成」を保つこと。
    """
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(
        race_id="R1", umaban="01", ketto_num="0000000001", bamei="トウロクウマ",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    data_dir = fx.write_dataset(tmp_path, races, race_horses, horses)
    repo = Repository(DataStore.open_current(data_dir))

    # sex_age の馬齢は fixture の生年（2022-03-15）と 2026年のレースに合わせる。
    # 合っていないと馬齢の段階で「この馬ではない」＝未登録になり、この
    # フィクスチャが「最大構成」であるという前提が静かに崩れる。
    entry = make_entry(umaban="01", bamei="トウロクウマ", sex_age="牡4", futan_kg=57.0)
    race_input = make_race_input([entry])
    return build_context(race_input, repo, pace_definition=pace_definition)


@pytest.fixture
def maximal_ctx(tmp_path):
    return _build_maximal_ctx(tmp_path, pace_definition=_PACE_DEFINITION)


@pytest.mark.parametrize("pace_definition", [_PACE_DEFINITION, None],
                          ids=["with_pace_definition", "without_pace_definition"])
def test_every_prompt_only_names_keys_that_exist_in_the_state(tmp_path, pace_definition):
    """実際に送られる全プロンプトについて、名指ししたキーが state に実在する
    ことを確かめる。

    存在しないキーを参照させると、モデルはその指示を無視するか自分で材料を
    選ぶ。どちらも例外を出さず、出力はもっともらしいままなので、プロンプト
    本文と state のキー集合を機械的に突き合わせる以外に捕まえ方がない
    （実例: LAYER1_PACE_INSTRUCTIONS が実在しない running_style_summary を
    参照していた）。

    手書きの対応表にしないのは、プロンプトを直して表を直し忘れた瞬間に
    ずれるため。本文から抽出すれば、プロンプト側の変更も state 側の改名も
    どちらも捕まる。

    ``pace_definition=None``（層別化に必要なレース数が足りない）でも
    ``--pace`` を要求した場合を含めて見る（fix5 決定7）。この組み合わせでは
    state に "pace_definition" キーが無いので、pace 質問を送ると §8.2 に
    正面から反する——「送る質問」を決める規則（``resolve_with_pace``）を
    通したうえで、送られるものだけを検査する。
    """
    ctx = _build_maximal_ctx(tmp_path, pace_definition=pace_definition)
    race_state = build_race_state(ctx)
    horse_state = build_horse_state(
        ctx, "01",
        race_context={
            "pace_probabilities": {"slow": 0.2, "standard": 0.5, "fast": 0.3},
            "note": "第1層で得た予測分布。確定した事実ではない",
        },
    )
    available = _state_key_names(race_state) | _state_key_names(horse_state)

    with_pace = predict_cli.resolve_with_pace(
        requested=True, pace_definition=ctx.pace_definition)
    prompts = {
        name: question.instructions
        for name, question in q.layer1_questions(ctx, with_pace=with_pace).items()
    }
    prompts.update({name: spec.instructions for name, spec in q.AXIS_SPECS.items()})
    for prompt_name, text in prompts.items():
        named = set(_IDENTIFIER.findall(text)) - _NOT_A_KEY
        missing = named - available
        assert not missing, f"{prompt_name} が state に無いキーを参照している: {sorted(missing)}"


def test_layer1_questions_are_choice_instances(ctx_three_horses):
    result = q.layer1_questions(ctx_three_horses, with_pace=True)
    assert isinstance(result["winner"], Choice)
    assert isinstance(result["pace"], Choice)


# --- 第2層 ---------------------------------------------------------------------

def test_layer2_has_five_axes():
    result = q.layer2_questions()
    assert set(result) == set(q.AXIS_SPECS)
    assert len(result) == 5


def test_every_axis_criteria_is_a_dict():
    """Choice なので criteria は dict（Score のリストとは違う）。"""
    for question in q.layer2_questions().values():
        assert isinstance(question, Choice)
        assert isinstance(question.criteria, dict)


def test_every_axis_has_insufficient_option():
    for name, spec in q.AXIS_SPECS.items():
        assert spec.insufficient_key in spec.criteria, name


def test_insufficient_and_neutral_are_separate():
    """spec §6.2:「判断材料不足」と「中立」は別の選択肢。"""
    criteria = q.AXIS_SPECS["distance_fit"].criteria
    assert "neutral" in criteria
    assert "insufficient" in criteria
    assert criteria["neutral"] != criteria["insufficient"]


def test_transport_concern_has_three_options():
    assert set(q.AXIS_SPECS["transport_concern"].criteria) == {
        "concern", "no_evidence", "insufficient"}
    assert q.AXIS_SPECS["transport_concern"].negative_key is None


def test_course_going_insufficient_does_not_absorb_missing_baba():
    """`insufficient` の説明で「馬場状態が未入力の場合も含む」と言わない。

    馬場状態が未入力なのはレース単位の事実で、**その馬の実績が乏しいこと
    ではない。** 同じ選択肢に両方を吸わせると、コース別・トラック別の実績を
    持っている馬まで `insufficient` に落ちる。実測でも、過去成績を持つ馬に
    限ると baba の有無で insufficient 率が 93.18% → 47.46% と倍近く動いて
    いた（report2 §4）。

    馬場状態を判断材料にしないという**指示のほう**は残す——そちらは
    「未入力の値を推測するな」という正しい制約で、選択肢の意味づけとは別。
    """
    spec = q.AXIS_SPECS["course_going_fit"]
    assert "馬場状態" not in spec.criteria["insufficient"]
    assert "race.baba が入力されていない場合" in spec.instructions


def test_relative_ability_uses_superiority_keys():
    assert set(q.AXIS_SPECS["relative_ability"].criteria) == {
        "superior", "comparable", "inferior", "insufficient"}


def test_axis_criteria_descriptions_are_non_empty():
    for spec in q.AXIS_SPECS.values():
        assert spec.instructions.strip()
        for description in spec.criteria.values():
            assert description and description.strip()


def test_axis_criteria_mention_experience_not_being_unfitness():
    """各軸の全文（instructions + criteria）に「経験」か「材料」が含まれ、
    未経験に触れている軸ではそれを不適性として扱わない旨が明記されている。
    """
    for name, spec in q.AXIS_SPECS.items():
        full_text = spec.instructions + "".join(spec.criteria.values())
        assert ("経験" in full_text) or ("材料" in full_text), name

    for name in ("distance_fit", "course_going_fit"):
        instructions = q.AXIS_SPECS[name].instructions
        assert "経験がないことを" in instructions
        assert "扱わないでください" in instructions


def test_feature_names_are_stable_and_ten():
    names_a = q.feature_names_for_axes()
    names_b = q.feature_names_for_axes()
    assert names_a == names_b
    assert len(names_a) == 10
    for axis in q.AXIS_SPECS:
        assert f"{axis}_signed" in names_a
        assert f"{axis}_insufficient" in names_a


def test_feature_names_follow_axis_specs_order():
    expected = []
    for axis in q.AXIS_SPECS:
        expected.append(f"{axis}_signed")
        expected.append(f"{axis}_insufficient")
    assert q.feature_names_for_axes() == tuple(expected)


# --- 版番号 ---------------------------------------------------------------------

def test_prompt_version_is_a_string():
    assert isinstance(q.PROMPT_VERSION, str)


def test_layer1_prompts_end_with_a_japanese_full_stop():
    """プロンプト末尾の句点は "。"（仕様どおり）。ASCII の "." にしない。

    PROMPT_VERSION は上げない——句点の誤記を仕様の字面に戻すだけで、質問の
    意味も指示の内容も変わらない（fix5 決定10）。
    """
    for name, text in (("winner", q.LAYER1_WINNER_INSTRUCTIONS),
                        ("pace", q.LAYER1_PACE_INSTRUCTIONS)):
        assert text.endswith("。"), name
        assert not text.endswith("."), name


def test_every_axis_prompt_ends_with_a_japanese_full_stop():
    for name, spec in q.AXIS_SPECS.items():
        assert spec.instructions.endswith("。"), name


def test_prompt_version_is_three():
    """`course_going_fit` の選択肢の意味が変わったので、プロンプトの版は "3"。

    "1"（ASCII "." 終わり）と "2"（"。" 終わり）の記録はどちらも
    data/predictions に実在する。"3" は `insufficient` の説明から
    「馬場状態が未入力の場合も含む」を外した世代で、**同じ
    `course_going_fit_insufficient` でも意味が違う**——同じ版を名乗ったまま
    混ぜると、fit_weights.require_consistent_versions() が
    「文言の違う記録を混ぜて学習させない」という役目を果たせない。
    """
    assert q.PROMPT_VERSION == "3"

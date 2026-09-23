"""Task 6〜15 を1レースぶん配線する。ここは Jev への state 組み立ても重みの
学習も特徴量の計算もしない――それぞれ features/*.py・learn/*.py・
predict/integrate.py の仕事。証拠文言の組み立ても持たない (predict/evidence.py
に分けてある)。ここが持つのは「どの順で呼ぶか」「途中で何が起きたら
PredictionFailed にするか」「答えをどう1つの記録にまとめるか」の3つだけ。

保存はしない (``predict`` は ``PredictionRecord`` を返すだけ)。CLI
(``scripts/predict.py``) や将来のバックテスト (Task 21) が保存するかどうか・
いつ保存するかを決める。ここで保存まで済ませると、バックテストが1レースごとに
ファイル I/O を抱えることになる。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from jev_client import JevRequestError
from keiba import config
from keiba.data.repository import Repository
from keiba.features.context import PredictionContext, build_context
from keiba.features.summary import data_quality as _build_data_quality
from keiba.learn.calibration import apply_calibration
from keiba.input import validate
from keiba.input.match import AmbiguousMatchError, match_horses
from keiba.models import RaceInput
from keiba.predict.evidence import EvidenceBuilder
from keiba.predict.integrate import (
    Weights,
    WeightsIncompatibleError,
    baseline_ranking,
    build_features,
    expand_missing,
    feature_names,
    logit,
    model_scores,
    python_features,
    rank_by_score,
    score_breakdown,
    softmax_by_race,
    standardize,
    tie_groups,
)
from keiba.predict.layers import (
    Layer1IncompleteError,
    Layer1Result,
    Layer2IncompleteError,
    Layer2Result,
    run_layer1,
    run_layer2,
)
from keiba.store.records import HorsePrediction, PredictionRecord, current_versions

__all__ = ["EvidenceBuilder", "PredictionFailed", "predict"]


class PredictionFailed(RuntimeError):
    """入力不備・照合失敗・重みの不整合・第2層の部分失敗など、``predict`` が
    完走できなかったときに送出する。

    記録 (``PredictionRecord``) は作らない――spec §12.7「不完全な評価を
    正常完了として返さない」。何が起きたかはメッセージに具体的に書く
    (利用者がそれを見て入力なり操作なりを直せる粒度で)。
    """


def _ambiguous_match_message(error: AmbiguousMatchError) -> str:
    """AmbiguousMatchError から、どの馬が・なぜ・どの候補と競合したかを
    人が読める形にする (候補の血統登録番号・馬名・性別まで含める)。
    """
    unresolved = [r for r in error.results if r.status in ("ambiguous", "conflict")]
    parts: list[str] = []
    for r in unresolved:
        candidates = "、".join(
            f"{c.ketto_num}:{c.bamei}"
            f"({config.SEX_CD_TO_NAME.get(c.sex_cd, c.sex_cd)})"
            for c in r.candidates
        )
        parts.append(f"{r.umaban}番 {r.entry.bamei}（{r.status}）候補: {candidates or 'なし'}")
    return "馬名を一意に決定できません: " + " / ".join(parts)


def _build_features_without_layer2(
    ctx: PredictionContext, layer1: Layer1Result,
) -> dict[str, dict[str, float]]:
    """with_layer2=False のときの特徴量 (``integrate.feature_names(with_layer2=False)``
    と同じ17列: logit_layer1 + Python8 + missing8)。

    ``integrate.build_features`` は軸特徴量 (``layer2.answers[umaban]``) を
    無条件に読むため、第2層を実施していない (``layer2.answers`` が空の) ここ
    では使えない。predict/integrate.py はこの1本のためには変更しない
    (Task 16 fix で許可された変更は rank_by_score の追加のみ) ので、
    build_features が内部で呼んでいる個々の純関数 (logit・python_features・
    expand_missing) をここでそのまま呼び、軸の部分を持たない行を組む。
    """
    features: dict[str, dict[str, float]] = {}
    for hf in ctx.horses:
        row: dict[str, float] = {"logit_layer1": logit(layer1.win_probabilities[hf.umaban])}
        row.update(expand_missing(python_features(hf)))
        features[hf.umaban] = row
    return features


def _race_input_to_dict(race_input: RaceInput) -> dict:
    """記録に埋め込む利用者入力の JSON-safe な形。tuple は list にする
    (store/records.py の free-form フィールドは JSON を経由すると list に
    戻るので、保存前から list で持つ)。
    """
    return {
        "date": race_input.date.isoformat(),
        "jyo": race_input.jyo,
        "race_num": race_input.race_num,
        "track_kind": race_input.track_kind,
        "kyori_m": race_input.kyori_m,
        "entry_count": race_input.entry_count,
        "prediction_cutoff": race_input.prediction_cutoff.isoformat(),
        "horses": [
            {
                "umaban": h.umaban,
                "bamei": h.bamei,
                "wakuban": h.wakuban,
                "sex_age": h.sex_age,
                "futan_kg": h.futan_kg,
                "kisyu": h.kisyu,
            }
            for h in race_input.horses
        ],
        "jyo_cd": race_input.jyo_cd,
        "baba_cd": race_input.baba_cd,
        "class_name": race_input.class_name,
        "baba": race_input.baba,
        "hasso_at": race_input.hasso_at.isoformat() if race_input.hasso_at is not None else None,
        "top_n": race_input.top_n,
        "race_id": race_input.race_id,
    }


def predict(
    race_input: RaceInput,
    repo: Repository,
    client: Any,
    *,
    with_layer2: bool = True,
    with_pace: bool = False,
    weights: Weights | None = None,
    allow_partial: bool = False,
    pace_definition: dict | None = None,
    sent_prompts: dict | None = None,
) -> PredictionRecord:
    """1レースぶんを予想する。保存はしない (呼び出し側の仕事)。

    ``sent_prompts`` に辞書を渡すと、**実際に送った** state と質問文がその場で
    入る（``{"layer1": {"state", "questions"}, "layer2": {"questions", "states"}
    | None}``）。返り値に足さず器で受け渡すのは、保存するかどうかを決めるのが
    呼び出し側だから——``predict`` は I/O をしない。あとで
    ``build_race_state`` を呼び直して再生成する方式は採らない。再生成は
    「いまのコードが送るとしたらどうなるか」であって、あのとき送った物では
    ない（report2 §6 の #7）。

    ``as_of`` はここで一度だけ決める (``race_input.prediction_cutoff``)。
    以降このレース内で as_of を必要とするすべての呼び出し (``match_horses``・
    ``build_context`` 内部の ``as_of``) はこの1つの値をそのまま使う――
    CLAUDE.md の制約どおり、別の時刻を作る経路を持たない。

    ``pace_definition`` は CLI 側 (``scripts/predict.py``) が
    ``fit_pace_definition`` で計算したものをそのまま受け取り、
    ``build_context`` へ渡すだけ。ここでは計算しない――``repo``
    (``Repository``) だけでは学習期間のレース集合 (``ResultRepository``
    が持つ) に手が届かないため (進捗台帳 所見D の裁定どおり、呼ぶのは CLI)。
    ただし ``with_pace=True`` と ``pace_definition=None`` の組み合わせは
    **ここで落とす** (ステップ0。spec §8.2「state に無いキーを名指しする
    プロンプトを送らない」を、呼び出し側の作法に依存せず守る)。
    """
    # 0. ペース質問を送れるかどうか。``pace_definition`` が無いなら
    #    ``with_pace`` を落とす (fix6 ⑤-1)。pace のプロンプトは state の
    #    ``pace_definition`` を名指しして「区分の意味は pace_definition に
    #    従ってください」と指示するが、``pace_definition`` が None のとき state に
    #    そのキーは無い (features/summary.py)。存在しないキーを名指しする
    #    プロンプトを送ることは spec §8.2 が禁じており、しかも例外にならず
    #    「もっともらしい出力」として出る——第2層で直したのと同じ形
    #    (ある層が別の層の保証を検査せずに仮定する) なので、ここでも検査する。
    #    CLI (scripts/predict.py) にも同じ判定があり、そちらは利用者向けの
    #    表示を担う。ここは「送らない」ことだけを担保する。
    pace_skipped = with_pace and pace_definition is None
    with_pace = with_pace and pace_definition is not None

    # 1. 入力の妥当性。
    problems = validate.check(race_input)
    if problems:
        detail = "; ".join(f"{p.field}: {p.message}" for p in problems)
        raise PredictionFailed(f"入力に問題があります: {detail}")

    as_of = race_input.prediction_cutoff

    # 2. 馬名照合。1頭でも決めきれなければ AmbiguousMatchError → PredictionFailed。
    #    ここでは「早期に失敗させる」ためだけに呼ぶ。実際に使う照合結果は
    #    build_context が内部でもう一度 match_horses を呼んで作る
    #    (同じ repo・同じ as_of なので結果は同一)。
    try:
        match_horses(race_input, repo, as_of=as_of)
    except AmbiguousMatchError as error:
        raise PredictionFailed(_ambiguous_match_message(error)) from error

    # 3. 重みの整合性。バージョンの整合性 (check_compatible) と、この実行が
    #    実際に作る特徴量集合が重みの要求する列とちょうど一致するか
    #    (feature_names) の2つを見る。片方向 (重みが要求するが作られない列)
    #    だけでなく、逆方向 (作られるのに重みが知らない列) も見る――
    #    with_layer2=True の実行に with_layer2=False で学習した17列の重みを
    #    渡すと片方向チェックは通ってしまうが、層2の10列を黙って無視する
    #    ことになる (fix round 2 で見つかった欠陥)。しかも馬ごとに層2を
    #    呼んで課金したうえで、その答えを使わず捨てることになるので、
    #    「使われない列がある」状態そのものを許さない。
    if weights is not None:
        try:
            weights.check_compatible()
        except WeightsIncompatibleError as error:
            raise PredictionFailed(str(error)) from error
        available = set(feature_names(with_layer2=with_layer2))
        required = set(weights.feature_names)
        if required != available:
            missing = sorted(required - available)
            extra = sorted(available - required)
            details = []
            if missing:
                details.append(f"重みが必要とするが作られない列: {missing}")
            if extra:
                details.append(f"作られるが重みが知らない列: {extra}")
            # WeightsIncompatibleError を明示的に作って from で繋ぐ (task-19-21
            # fix Critical 1)。ここは except ブロックではなく素の if なので、
            # 繋ぐ元の例外が自然には存在しない。それでも __cause__ に
            # WeightsIncompatibleError を残さないと、backtest.py の
            # isinstance(error.__cause__, WeightsIncompatibleError) がこの
            # 枝を横取りできず、「全レースで同じように起きる設定の誤り」が
            # 200回スキップされて「API が N 回落ちた」と誤報する
            # (レビュアー再現: --no-layer2 と27列重みの取り違えで
            # skip_breakdown = {'jev_failed': 3, ...})。
            error = WeightsIncompatibleError(
                f"重みとこの実行の特徴量が一致しません（with_layer2={with_layer2}）。"
                + " / ".join(details)
                + "。重みと同じ構成で実行するか（--no-layer2 など）、学習し直してください。"
            )
            raise PredictionFailed(str(error)) from error

    # 4. state の組み立て。
    ctx = build_context(race_input, repo, pace_definition=pace_definition)

    # 5. 第1層 (レース単位、1リクエスト)。
    #
    #    ここだけ try/except で包まれていない状態が残っていた (fix1 Critical)。
    #    レート制限 (429)・タイムアウト・5xx はもっとも普通に起きる失敗だが、
    #    それらは JevRequestError になって run_layer1 の外まで生で漏れ、
    #    scripts/predict.py は InputError/DataVersionError/OSError/
    #    JevConfigError/PredictionFailed しか捕まえないため、利用者に生の
    #    トレースバックが出ていた。第2層 (Layer2IncompleteError) と
    #    照合・入力検証は元から包まれていたのに、ここだけ漏れていた。
    #
    #    JevConfigError (キー未設定など、通信する前に分かる設定の誤り) は
    #    ここでは包まない――scripts/predict.py が専用の案内を別に出す契約
    #    なので、そのまま外へ通す。JevRequestError と JevConfigError は共に
    #    JevError の子だが、ここでは JevRequestError だけを狙って捕まえる。
    #    応答の形が足りない場合 (winner が返っていない) は Layer1IncompleteError
    #    になる。これも PredictionFailed に変換する——生の KeyError を外に
    #    出さない (fix5 決定2)。通信の失敗 (JevRequestError) と応答の不足は
    #    別のクラスで来るので、メッセージも分ける。
    try:
        layer1 = run_layer1(client, ctx, with_pace=with_pace)
    except JevRequestError as error:
        raise PredictionFailed(f"第1層のリクエストが失敗しました: {error}") from error
    except Layer1IncompleteError as error:
        raise PredictionFailed(f"第1層の応答が不完全です: {error}") from error

    # 6. 第2層 (馬単位、並列)。allow_partial=False で1頭でも失敗すれば
    #    run_layer2 自身が Layer2IncompleteError を投げる (第2層で全員を
    #    走らせ切ったうえでの失敗、途中で打ち切ってはいない)。
    if with_layer2:
        try:
            layer2 = run_layer2(client, ctx, layer1, allow_partial=allow_partial)
        except Layer2IncompleteError as error:
            detail = "; ".join(f"{umaban}: {reason}" for umaban, reason in error.failures.items())
            raise PredictionFailed(f"第2層が一部の馬で失敗しました: {detail}") from error
    else:
        layer2 = Layer2Result(answers={}, failures={}, input_tokens=0)

    status = "PARTIAL" if layer2.failures else "SUCCEEDED"

    # 7. 特徴量 → (weights があれば) 標準化 → モデルスコア → softmax → 内訳。
    #
    #    第2層が一部の馬で失敗した (layer2.failures が非空、つまり
    #    allow_partial=True で status == "PARTIAL") ときは、モデルランキングを
    #    一切出さない (全頭 None のまま)。生き残った馬だけで softmax を取ると
    #    欠けた馬の確率をその馬たちが吸収して過大になるうえ、ランキングから
    #    出走馬が消える――順位の形をした不完全な結果になり、spec §12.7
    #    「不完全な評価を正常完了として返さない」に反する。第1層は全頭を
    #    覆っている (run_layer1 は欠けた馬を 0.0 で埋める) ので baseline は
    #    常に完全であり、そちらだけを返す。
    model_scores_by_umaban: dict[str, float] = {}
    model_win_probs: dict[str, float] = {}
    breakdown_by_umaban: dict[str, dict[str, float]] = {}
    model_rank_by_umaban: dict[str, int] = {}

    # raw_features は HorsePrediction.features (学習入力。Task 18 supplement)
    # と、weights があるときの standardize() の両方が使う。1箇所で計算し、
    # 両方がこれを参照する――同じものを2回作ると片方だけ直される。
    #
    # layer2 が一部失敗している (layer2.failures が非空) ときは、失敗した馬の
    # 軸特徴が無く build_features が KeyError になるので、その実行全体を
    # _build_features_without_layer2 (17列) にそろえる。この記録は
    # status="PARTIAL" になり、scripts/fit_weights.py が学習集合から除外する
    # ので、記録によって27列/17列が混在しても学習には影響しない。
    raw_features = (
        build_features(ctx, layer1, layer2) if with_layer2 and not layer2.failures
        else _build_features_without_layer2(ctx, layer1)
    )

    # has_model_ranking はこのブロックとステップ8の同程度の組の両方が使う
    # 「model のランキングを出すかどうか」の1つの判定。片方だけ直して2つの
    # 判定が食い違う (model_win_prob は無いのに model 尺度で組を計算する、など)
    # のを避けるため、ここで1回だけ決める。
    has_model_ranking = weights is not None and not layer2.failures
    if has_model_ranking:
        # ステップ3の完全一致チェックにより、ここで WeightsIncompatibleError が
        # 出ることは無いはずだが、それでも try で囲む。「無いはず」の例外が
        # predict() の外へ生で漏れると、CLI はメッセージの代わりにトレース
        # バックを出す――重み/特徴量の不整合が想定外の場所で表面化するのは
        # これで2度目なので、届かないはずの経路にも同じ費用 (try 1つ) を払う。
        try:
            features = standardize(raw_features, weights.standardization)
        except WeightsIncompatibleError as error:
            raise PredictionFailed(f"重みがこの実行と整合しません: {error}") from error
        model_scores_by_umaban = model_scores(features, weights)
        model_win_probs = softmax_by_race(model_scores_by_umaban)
        breakdown_by_umaban = score_breakdown(features, weights)
        model_order = rank_by_score(model_win_probs)
        model_rank_by_umaban = {umaban: i + 1 for i, umaban in enumerate(model_order)}

    # 7b. 較正 (spec §11.2)。**順位には使わない**——isotonic は単調なので
    #     理屈の上では順位不変だが、定義域の外は端の値に張り付く
    #     (numpy.interp は外挿しない)。出荷している較正マップは節点10個・
    #     calib 16レースしかないので張り付きはふつうに起きる。そこで同値が
    #     量産されると、順位の形をした「較正がそう答えた」に見えるものが
    #     できあがる。較正後の確率は記録して、順位は model_win_prob のまま。
    calibration_map = weights.calibration if weights is not None else None
    if has_model_ranking and calibration_map is not None:
        calibrated_win_probs, raw_sum = apply_calibration(calibration_map, model_win_probs)
        # 定義域の外に出た馬。値だけを見ても張り付きは分からない——張り付いた
        # 馬どうしは同じ確率になり、較正がそう判断したように読める。
        domain_lo, domain_hi = calibration_map["x"][0], calibration_map["x"][-1]
        clipped = sorted(
            umaban for umaban, p in model_win_probs.items()
            if p < domain_lo or p > domain_hi
        )
        calibration_info = {
            "applied": True,
            # 再正規化の前後とずれ。1.0 から大きく離れていれば、この較正が
            # 今回の顔ぶれに対してどれだけ信頼できるかの兆候になる。
            "sum_before_renormalization": raw_sum,
            "deviation_from_one": raw_sum - 1.0,
            "domain": [domain_lo, domain_hi],
            "clipped_umaban": clipped,
            "used_for_ranking": False,
        }
    else:
        calibrated_win_probs = {}
        if weights is None:
            reason = "重みを使わない実行なので、較正する対象の確率 (model_win_prob) が無い。"
        elif calibration_map is None:
            reason = "この重みは較正マップを持っていない。"
        else:
            reason = "第2層が一部の馬で失敗しており、model のランキングを出していない。"
        calibration_info = {"applied": False, "reason": reason, "used_for_ranking": False}

    # 8. 順位。baseline は常に付ける。model は weights があり、かつ第2層が
    #    完全 (allow_partial で欠けが出ていない) ときだけ。
    baseline_order = baseline_ranking(layer1)
    baseline_rank_by_umaban = {umaban: i + 1 for i, umaban in enumerate(baseline_order)}

    # 8b. 同程度の組。**測っていない尺度では組を作らない** (fix5 決定1)。
    #     config.TIE_BAND は第1層の勝率の run 間ばらつきとして実測した値で、
    #     model_score / model_win_prob の上では測っていない。model の順位を
    #     出す実行 (has_model_ranking) でこの値を model_win_prob に当てると、
    #     スコアの幅が狭い run では全馬が1つの組になり、to_user_view が
    #     「予想順位」を持つ馬を1頭も返さない——製品の最終出力 (上位n頭) が
    #     消えたうえ、測っていない尺度のばらつきを主張することになる。
    #     ここでは全馬を別々の組にし (model の順位がそのまま組番号)、
    #     data_quality["tie_band_scale"] に "not_measured" と残す。
    #     baseline の順位 (第1層の勝率) を出す実行は、まさにその尺度で測った
    #     値なので従来どおり tie_groups を適用する。
    if has_model_ranking:
        tie_band_scale = "not_measured"
        tie_group_by_umaban = dict(model_rank_by_umaban)
    else:
        tie_band_scale = "layer1_win_prob"
        tie_group_by_umaban = tie_groups(layer1.win_probabilities, band=config.TIE_BAND)

    # 9. 馬ごとの証拠 (positives/negatives/insufficient) と HorsePrediction。
    horses: list[HorsePrediction] = []
    for hf in ctx.horses:
        umaban = hf.umaban
        # None: 第2層未実施 (with_layer2=False) か、この馬だけ失敗した
        # (allow_partial=True)。「回答が無い」ことを表すので、falsy な {} と
        # 区別して明示的に is None で見る。
        axis_answers = layer2.answers.get(umaban)
        layer2_dict = {
            axis: {
                "choice": ans.choice,
                "probabilities": dict(ans.probabilities),
                "confidence": ans.confidence,
            }
            for axis, ans in (axis_answers if axis_answers is not None else {}).items()
        }
        positives, negatives, insufficient = EvidenceBuilder.build(
            hf, axis_answers if with_layer2 else None)

        horses.append(HorsePrediction(
            umaban=umaban,
            bamei=hf.bamei,
            rank_baseline=baseline_rank_by_umaban[umaban],
            rank_model=model_rank_by_umaban.get(umaban),
            tie_group=tie_group_by_umaban[umaban],
            layer1_win_prob=layer1.win_probabilities[umaban],
            model_score=model_scores_by_umaban.get(umaban),
            model_win_prob=model_win_probs.get(umaban),
            calibrated_win_prob=calibrated_win_probs.get(umaban),
            score_breakdown=breakdown_by_umaban.get(umaban, {}),
            # raw_features は ctx.horses と同じ馬番集合を必ず持つ (上で同じ
            # ctx.horses から作っている)。.get で既定を埋めず直接引く――
            # 無ければ計算のどこかが崩れているので、黙って {} にせず落ちる
            # べき。
            features=raw_features[umaban],
            layer2=layer2_dict,
            positives=positives,
            negatives=negatives,
            insufficient=insufficient,
            # HorseFeatures.notes をそのまま運ぶ (data_quality 経由ではなく
            # ctx.horses から直接)。持ち主 (この馬) に紐づけたまま保存し、
            # to_user_view が表示する馬だけに絞れるようにする (fix4 決定2)。
            notes=hf.notes,
        ))

    # 10. 記録を組み立てて返す (保存は呼び出し側)。
    versions = dict(current_versions(
        data_version=ctx.data_version,
        model=layer1.model,
        weights_id=weights.weights_id if weights is not None else None,
    ))
    versions["data_fetched_at"] = ctx.data_fetched_at.isoformat()

    data_quality_dict = dict(_build_data_quality(ctx))
    if pace_skipped:
        # ペース質問を要求されたが送らなかったこと (と理由) を記録に残す
        # (fix6 ⑤-1)。黙って落とすと、記録からは「pace を要求しなかった実行」と
        # 区別できない。notes はレース単位の注記なので利用者にも届く。
        data_quality_dict["notes"] = [
            *data_quality_dict["notes"],
            "ペース定義が作れなかったので展開（pace）の質問は送っていない。",
        ]
    if with_layer2:
        data_quality_dict["layer2_failures"] = dict(layer2.failures)
    # 同程度の組 (ステップ8b) をどちらの尺度で判定したか、あるいは判定して
    # いないか ("not_measured")。後から見て解釈できるように常に残す
    # (with_layer2 の有無に関わらず、この実行が付けた tie_group 全部に対応する)。
    data_quality_dict["tie_band_scale"] = tie_band_scale
    # 較正を適用したか、していないならなぜか (ステップ7b)。キーごと省くと
    # 「適用した結果 None だった」と区別できないので、常に残す。
    data_quality_dict["calibration"] = calibration_info

    layer2_input_tokens = layer2.input_tokens if with_layer2 else 0
    # リクエスト件数も残す (spec §13 の usage は "requests" を持つ)。第1層は
    # レース単位で常に1回、第2層は馬ごとに1回ずつで、失敗した馬の分も送って
    # いる (課金されている) ので answers と failures の両方を数える。
    layer2_requests = len(layer2.answers) + len(layer2.failures) if with_layer2 else 0
    usage = {
        "layer1_input_tokens": layer1.input_tokens,
        "layer2_input_tokens": layer2_input_tokens,
        "total_input_tokens": layer1.input_tokens + layer2_input_tokens,
        "layer1_requests": 1,
        "layer2_requests": layer2_requests,
        "requests": 1 + layer2_requests,
    }

    # 送った物を呼び出し側の器へ。第2層を回していない実行では "layer2" を
    # None にする——空の辞書を入れると「送ったが空だった」と読めてしまう。
    if sent_prompts is not None:
        sent_prompts["layer1"] = {
            "state": layer1.sent_state,
            "questions": layer1.sent_questions,
        }
        sent_prompts["layer2"] = {
            "questions": layer2.sent_questions,
            "states": layer2.sent_states,
        } if with_layer2 else None

    layer1_dict = {
        "choice": layer1.choice,
        "confidence": layer1.confidence,
        "model": layer1.model,
        "missing_probability_umaban": list(layer1.missing_probability_umaban),
        "pace_probabilities": (
            dict(layer1.pace_probabilities) if layer1.pace_probabilities is not None else None
        ),
    }

    return PredictionRecord(
        record_id=uuid.uuid4().hex,
        race_id=race_input.race_id,
        predicted_at=datetime.now(timezone.utc),
        prediction_cutoff=as_of,
        n=race_input.top_n,
        versions=versions,
        horses=tuple(horses),
        layer1=layer1_dict,
        data_quality=data_quality_dict,
        usage=usage,
        status=status,
        race_input=_race_input_to_dict(race_input),
    )

"""予測記録の保存と、利用者向けビューへの絞り込み。

保存する記録 (``PredictionRecord``) と、利用者に見せる形 (``to_user_view`` の
返り値) は意図して別の型にしてある。前者は検証フェーズ (Task 19-21) が
全頭・全分布・全版番号を読み戻すための完全な記録、後者は spec §8 / §12.8 が
定める「利用者に見せてよい範囲」。両者を同じ辞書から「削る」方式で作ると、
記録側に項目が増えるたびに利用者向けへ静かに漏れる。``to_user_view`` は削る
のではなく、出す項目を列挙して作る。

state の組み立て・第2層の分類・重み付けはここではしない。ここは保存形式と
JSON への往復、そして「利用者に何を見せるか」の決定だけを持つ。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from keiba import config
from keiba.predict.questions import PROMPT_VERSION

__all__ = [
    "HorsePrediction",
    "PredictionRecord",
    "current_versions",
    "save",
    "load_month",
    "to_user_view",
    "to_json",
    "from_json",
]


@dataclass(frozen=True)
class HorsePrediction:
    """1頭ぶんの予測。第1層・第2層・第3層の答えをすべて保持する。

    ``calibrated_win_prob`` は ``predict/pipeline.py`` が較正マップを適用した
    結果（重みが較正マップを持ち、model のランキングを出す実行のときだけ。
    それ以外は ``None``）。**この層のどの関数もここに数値を書き込まない**——
    書くのは pipeline、ここは運ぶだけ。

    値が入っていても**利用者には見せない**。spec §10.3 が「勝率として表示
    するのは較正が別期間で検証できたときだけ」としており、その検証はまだ
    済んでいない。``to_user_view`` は削るのではなく出す項目を列挙する作りなので、
    列挙に足さない限り漏れない。
    """

    umaban: str
    bamei: str
    rank_baseline: int
    rank_model: int | None
    #: 同程度の組の番号（1始まり）。同じ番号の馬は、勝率の差がモデルの run 間の
    #: ばらつき（config.TIE_BAND）より小さく、前後関係が再現しない。
    tie_group: int
    layer1_win_prob: float
    model_score: float | None
    model_win_prob: float | None
    calibrated_win_prob: float | None
    score_breakdown: dict[str, float]
    #: 統合に使った生の特徴量（integrate.build_features の1頭分）。キーは
    #: integrate.feature_names(with_layer2=...) と同じ (with_layer2 の有無で
    #: 27列/17列。層2が一部失敗した実行では常に17列にそろえる――
    #: predict/pipeline.py 参照)。
    #:
    #: score_breakdown（weight * value、寄与）とは別に生の値を持つ。重みを
    #: 学習する前の記録には重みが無いので score_breakdown は空 ({}) で、
    #: そこからは学習入力を作れない。生の値を記録しておけば、学習
    #: (scripts/fit_weights.py) は Jev を呼び直さずに済み（同じレースへの
    #: 二重課金を避けられ）、かつ「記録した回答」と「学習に使った入力」が
    #: 原理的にずれない (Task 18 supplement)。
    features: dict[str, float]
    layer2: dict[str, dict]
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    insufficient: tuple[str, ...]
    #: この馬についての注記（HorseFeatures.notes をそのまま運ぶ）。レース
    #: 単位の注記は PredictionRecord.data_quality["notes"] にある——注記を
    #: 平坦に連結すると持ち主が分からなくなり、利用者向けの表示で「表示する
    #: 馬だけに絞る」ができなくなる（fix4 決定2）。
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PredictionRecord:
    """1レースぶんの予測記録。

    出走全頭を ``horses`` に保持する――``n`` は表示だけを絞る値で、記録その
    ものを絞る値ではない。検証フェーズは全頭ぶんの答えを必要とするため
    (spec §8)。
    """

    record_id: str
    race_id: str | None
    predicted_at: datetime
    prediction_cutoff: datetime
    n: int
    versions: dict[str, str]
    horses: tuple[HorsePrediction, ...]
    layer1: dict
    data_quality: dict
    #: 使用量。入力トークン（layer1/layer2/合計）と**リクエスト件数**
    #: （``layer1_requests`` / ``layer2_requests`` / ``requests``）を持つ
    #: （spec §13 の ``usage`` は ``input_tokens`` と ``requests`` の両方を
    #: 要求している）。件数はトークンから復元できない――第2層は馬ごとに
    #: 1リクエストなので、頭数が分からないと「何回投げたか」が残らない。
    #: 組み立てるのは ``predict/pipeline.py``（ここは形式だけを持つ）。
    usage: dict
    status: str
    race_input: dict


def current_versions(*, data_version: str, model: str, weights_id: str | None) -> dict[str, str]:
    """比較可能性を決める8キーを1箇所にまとめる (spec §12.7)。

    このうち1つでも違う記録は比較できない。呼び出し側がキーを個別に組み立て
    ると1つ書き漏らしうるので、ここに集約する。``weights_id`` は baseline の
    みの予想 (重み未適用) では ``None`` になる。
    """
    return {
        "data_version": data_version,
        "schema_version": config.RECORD_SCHEMA_VERSION,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "feature_version": config.FEATURE_VERSION,
        "integration_version": config.INTEGRATION_VERSION,
        "exception_rules_version": config.EXCEPTION_RULES_VERSION,
        "weights_id": weights_id,
    }


# --- JSON との往復 -----------------------------------------------------------
#
# datetime は isoformat() の文字列で持つ。json.dumps(default=...) は使わない
# ――default はどんな型でも黙って呑み込んでしまい、想定していない値が紛れ
# 込んだときにテストが気づけなくなる。datetime はここで明示的に変換する。
# tuple はここで list に変わる (JSON に tuple 型は無い) ので、読み戻すとき
# ``HorsePrediction`` の tuple フィールド (positives/negatives/insufficient/
# notes) は明示的に tuple() で戻す。それ以外の自由形式の dict (layer1/
# data_quality/usage/race_input/layer2) の中身は list のまま持ち回るだけで、
# tuple には戻さない――どの値が元 tuple だったかは JSON からは分からないため。


def _horse_to_json(horse: HorsePrediction) -> dict:
    return {
        "umaban": horse.umaban,
        "bamei": horse.bamei,
        "rank_baseline": horse.rank_baseline,
        "rank_model": horse.rank_model,
        "tie_group": horse.tie_group,
        "layer1_win_prob": horse.layer1_win_prob,
        "model_score": horse.model_score,
        "model_win_prob": horse.model_win_prob,
        "calibrated_win_prob": horse.calibrated_win_prob,
        "score_breakdown": dict(horse.score_breakdown),
        "features": dict(horse.features),
        "layer2": horse.layer2,
        "positives": list(horse.positives),
        "negatives": list(horse.negatives),
        "insufficient": list(horse.insufficient),
        "notes": list(horse.notes),
    }


def _horse_from_json(payload: dict) -> HorsePrediction:
    return HorsePrediction(
        umaban=payload["umaban"],
        bamei=payload["bamei"],
        rank_baseline=payload["rank_baseline"],
        rank_model=payload["rank_model"],
        # 古い形式を黙って救わない。.get で既定を埋めると、tie_group キーの
        # 無い payload が来ても気づけない。**旧形式の記録は実在する**が、
        # どの形式かは versions["schema_version"]（config.RECORD_SCHEMA_VERSION）
        # で判別できるので、読めない記録を推測で埋める必要はない——欠けている
        # のはこちらが決めるべき値ではなく、その記録に無い事実である。
        tie_group=payload["tie_group"],
        layer1_win_prob=payload["layer1_win_prob"],
        model_score=payload["model_score"],
        model_win_prob=payload["model_win_prob"],
        calibrated_win_prob=payload["calibrated_win_prob"],
        score_breakdown=dict(payload["score_breakdown"]),
        # notes/tie_group と同じ規則で、古い形式を黙って救わない。.get で既定を
        # 埋めると、features キーの無い payload が来ても気づけない (Task 18
        # supplement の明示的な指示)。形式の違いは schema_version で判別する。
        features=dict(payload["features"]),
        layer2=payload["layer2"],
        positives=tuple(payload["positives"]),
        negatives=tuple(payload["negatives"]),
        insufficient=tuple(payload["insufficient"]),
        # 古い形式を黙って救わない。.get で既定を埋めると、notes キーの無い
        # payload が来ても気づけない（「注記が無かった」と「注記の欄そのものが
        # 無い形式だった」は別のこと）。形式の違いは schema_version で判別する。
        notes=tuple(payload["notes"]),
    )


def to_json(record: PredictionRecord) -> dict:
    """保存用の完全な形。``save`` → ``load_month`` の往復で元の record と
    一致することがここの契約――欠けたキーがあると検証フェーズ (Task 19-21)
    の手が届かなくなる。
    """
    return {
        "record_id": record.record_id,
        "race_id": record.race_id,
        "predicted_at": record.predicted_at.isoformat(),
        "prediction_cutoff": record.prediction_cutoff.isoformat(),
        "n": record.n,
        "versions": dict(record.versions),
        "horses": [_horse_to_json(h) for h in record.horses],
        "layer1": record.layer1,
        "data_quality": record.data_quality,
        "usage": record.usage,
        "status": record.status,
        "race_input": record.race_input,
    }


def from_json(payload: dict) -> PredictionRecord:
    return PredictionRecord(
        record_id=payload["record_id"],
        race_id=payload["race_id"],
        predicted_at=datetime.fromisoformat(payload["predicted_at"]),
        prediction_cutoff=datetime.fromisoformat(payload["prediction_cutoff"]),
        n=payload["n"],
        versions=dict(payload["versions"]),
        horses=tuple(_horse_from_json(h) for h in payload["horses"]),
        layer1=payload["layer1"],
        data_quality=payload["data_quality"],
        usage=payload["usage"],
        status=payload["status"],
        race_input=payload["race_input"],
    )


# --- 保存・読み出し -----------------------------------------------------------


def _month_path(data_dir: Path, year_month: str) -> Path:
    return Path(data_dir) / "predictions" / f"{year_month}.jsonl"


def save(record: PredictionRecord, *, data_dir: Path) -> Path:
    """1行1レコードの JSON Lines として追記する。

    月ごとにファイルを分けるのは、検証フェーズが「その月の記録だけ読む」を
    ふつうに行うため。月は ``predicted_at`` (予想を出した時刻) から決める
    ――レース施行日ではなく、予想記録そのものが発生した時刻で束ねる。
    """
    path = _month_path(data_dir, record.predicted_at.strftime("%Y%m"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_json(record), ensure_ascii=False))
        handle.write("\n")
    return path


def load_month(year_month: str, *, data_dir: Path) -> list[PredictionRecord]:
    """該当月のファイルが無ければ空リストを返す (「まだ何も予想していない」
    は壊れた状態ではない)。
    """
    path = _month_path(data_dir, year_month)
    if not path.exists():
        return []
    loaded: list[PredictionRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                loaded.append(from_json(json.loads(line)))
    return loaded


# --- 利用者向けビュー ---------------------------------------------------------


def _display_rank(horse: HorsePrediction) -> int:
    """表示に使う順位。``rank_model`` が無ければ ``rank_baseline`` に落ちる。

    ``rank_model or rank_baseline`` だと ``rank_model == 0`` のときも
    baseline に落ちてしまう――0 と「未設定 (None)」を区別できない。順位は
    1始まりなので今のところ 0 は出ないが、この関数の値を決めるのは
    Task 16 であり、ここで暗黙のルールを決め打ちしない。``None`` かどうかを
    明示的に見る。
    """
    return horse.rank_model if horse.rank_model is not None else horse.rank_baseline


def _extend_to_tie_group_boundary(
    sorted_horses: Sequence[HorsePrediction], n: int,
) -> list[HorsePrediction]:
    """順位順に並んだ全頭から、上位 n 頭の表示対象を決める (Task 16 fix2 決定4)。

    n 番目の馬が属する組を見つけ、その組の最後の馬まで表示対象にする――
    n を超えてよい。同程度の馬を n で機械的に切ると、境界の前後関係が
    run ごとに入れ替わる馬だけを不自然に選んでしまう (spec が明示的に避けた
    やり方)。tie_group は pipeline.py が順位と同じ尺度で計算しているので、
    ここでは並び (sorted_horses) に沿って「境界の組の tie_group 以下は
    すべて含める」だけでよい――同じ尺度・同じ順序で作られた組は、この並びの
    中で必ず連続したまとまりになる。
    """
    if n <= 0 or not sorted_horses:
        return []
    boundary_index = min(n, len(sorted_horses)) - 1
    boundary_group = sorted_horses[boundary_index].tie_group
    return [h for h in sorted_horses if h.tie_group <= boundary_group]


#: 同程度の判定を行っていない実行（``data_quality["tie_band_scale"]`` が
#: ``"not_measured"``）のときに1行出す説明。config.TIE_BAND は第1層の勝率で
#: 測った値で、モデル尺度の上では測っていないため、その尺度では組を作らない
#: （fix5 決定1）。利用者向けの表示（``to_user_view`` と CLI の
#: ``_print_view``）が同じ文言を2箇所に書かないよう、ここに1つだけ持つ。
TIE_NOT_MEASURED_NOTE = (
    "モデル尺度のばらつきは測っていないため、同程度の判定は行っていません")


def to_user_view(record: PredictionRecord) -> dict:
    """利用者に返す形。内部記録とは別物にする。

    spec §8: 元CSV、過去走の生レコード、Jev へ送った state 全体、API の生
    レスポンスを利用者向けから取得できないようにする。評価理由は分類と
    独自集計に基づいて構成し、過去データの一覧をそのまま転載しない。

    ここで「含めるものを列挙する」方式を採る。除外リスト方式だと、
    項目が増えたときに漏れる。

    Task 16 fix2 決定4: 上位 n 頭の境界が同程度の組にかかっているときは、
    その組の最後まで表示する (n を超えてよい)。組の要素が複数頭のときは、
    その馬たちに個別の「予想順位」を出さず (キー自体を持たせない)、代わりに
    「同程度の組の説明」で組の範囲をまとめて示す。

    fix5 決定1: 重みを使った実行では同程度の判定そのものを行わない
    (model 尺度のばらつきを測っていないため)。その場合は「同程度の判定」に
    ``TIE_NOT_MEASURED_NOTE`` を1行入れる。
    """
    sorted_horses = sorted(record.horses, key=_display_rank)
    top = _extend_to_tie_group_boundary(sorted_horses, record.n)

    group_sizes: dict[int, int] = {}
    for h in top:
        group_sizes[h.tie_group] = group_sizes.get(h.tie_group, 0) + 1

    explanations: dict[str, str] = {}
    for h in top:
        size = group_sizes[h.tie_group]
        key = str(h.tie_group)
        if size <= 1 or key in explanations:
            continue
        ranks = sorted(_display_rank(m) for m in top if m.tie_group == h.tie_group)
        explanations[key] = (
            f"{ranks[0]}位〜{ranks[-1]}位。勝率の差がモデルのばらつき"
            f"（{config.TIE_BAND}）より小さく、この{size}頭の前後関係は"
            "繰り返すと変わります。"
        )

    def _horse_view(h: HorsePrediction) -> dict:
        view: dict = {"同程度の組": h.tie_group}
        # 組の要素が1頭のときだけ「予想順位」を出す。.get で既定を埋めず、
        # キーの有無そのもので「順位をつけられない」を表す (決定4)。
        if group_sizes[h.tie_group] == 1:
            view["予想順位"] = _display_rank(h)
        view.update({
            "馬番": h.umaban,
            "馬名": h.bamei,
            "プラス材料": list(h.positives),
            "マイナス材料": list(h.negatives),
            "判断材料が不足している項目": list(h.insufficient),
            "注記": list(h.notes),
        })
        return view

    view: dict = {
        "予想時刻": record.predicted_at.isoformat(timespec="seconds"),
        "予想締切": record.prediction_cutoff.isoformat(timespec="seconds"),
        "データ更新時刻": record.versions.get("data_fetched_at"),
        "上位n頭": record.n,
        "注意": "勝ちやすさの順位です。実際の着順の断定ではありません。",
        # data_quality()（features/summary.py）が必ず "notes" キーを作るので
        # .get ではなく直接引く。無ければ黙って空になるのではなく落ちるべき。
        # ここにはレース単位の注記だけが入る（fix4 決定1）。馬単位の注記は
        # data_quality["horse_notes"] ではなく、持ち主である h.notes から
        # 「予想」の各馬に付ける――data_quality["horse_notes"] を経由すると、
        # 表示しない馬（top に入らない馬）の注記まで持ち主の情報を失って
        # 混ざってしまう（前ラウンドの欠陥の再発防止）。
        "注記": list(record.data_quality["notes"]),
        "予想": [_horse_view(h) for h in top],
        # 組が複数頭のものだけキーを持つ。0.03 のような数字を文字列に
        # 焼き込まず、config.TIE_BAND から組む (決定4)。
        "同程度の組の説明": explanations,
    }
    # 同程度の判定を行っていない実行では、その事実を1行出す (fix5 決定1)。
    # data_quality["tie_band_scale"] は predict() が常に書くので .get で
    # 既定を埋めない——無ければ predict() 側の欠陥なので落ちるべき。
    if record.data_quality["tie_band_scale"] == "not_measured":
        view["同程度の判定"] = TIE_NOT_MEASURED_NOTE
    return view

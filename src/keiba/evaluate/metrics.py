"""予想の答え合わせ指標。

このモジュールが持たないもの: Jev への通信、state の組み立て、順位そのものの計算
（ranking はどこか別の場所で既に決まっている前提で受け取る）。

同着の扱い（spec §12.2）:
- 勝ち馬は複数になりうる（デッドヒート）。`RaceOutcome.winners` はタプルで持つ。
- brier では同着馬に事象を均等に割る（2頭同着なら各 y=0.5）。
- log_loss では「同着馬のいずれかが勝つ」という1つの事象として扱うため、
  同着馬の確率を合計する。

出走集合の扱い（spec §12.2）:
- 取消・発走除外・競走除外（ResultRow.counts_as_starter が False）は starters に
  含めない。走っていないので「外れた」にも「当たった」にもできない。
- 競走中止・失格は starters に残るが勝ち馬にはなれない
  （ResultRow.can_win / is_winner が既にその規則を持っている。ここで
  ijyo_cd を直接見ない）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from keiba.models import ResultRow

__all__ = [
    "MetricsReport",
    "OutcomeError",
    "RaceOutcome",
    "aggregate",
    "brier",
    "calibration_bins",
    "compare",
    "expected_top_n",
    "log_loss",
    "log_loss_is_clipped",
    "outcome_from_results",
    "top1_win",
    "top_n_hit",
]


class OutcomeError(ValueError):
    """「採点できない」ではなく「データが矛盾している」ときに送出する（fix1）。

    使い分け（``outcome_from_results`` の docstring も参照）:

    - ``None`` を返す: 勝ち馬が居ない／1頭立て／``results`` が空。
      どれも「このレースは採点できない」という同じ状況で、競走が成立
      しなかった・そもそも結果データが無いというだけであり、渡された
      データそのものは矛盾していない。
    - ``OutcomeError`` を投げる: **矛盾したデータ**。勝ち馬が ``starters``
      に含まれていない、``umaban`` が重複している（``outcome_from_results``）、
      あるいは ``starters`` の馬の確率が ``probabilities`` に無い
      （``brier`` / ``log_loss``）。渡された値どうしが食い違っており、
      既定値で埋めて黙って進めると見たい欠陥を見失う。
    """


@dataclass(frozen=True)
class RaceOutcome:
    """1レース分の答え合わせ用の正解データ。"""

    race_id: str
    #: 勝ち馬の馬番。デッドヒートなら複数。
    winners: tuple[str, ...]
    #: 出走集合の馬番（取消・発走除外・競走除外を除く）。
    starters: tuple[str, ...]
    #: starters から除かれた馬番（取消・発走除外・競走除外）。
    excluded: tuple[str, ...]


def outcome_from_results(race_id: str, results: Sequence[ResultRow]) -> RaceOutcome | None:
    """ResultRow の並びからそのレースの正解を組み立てる。

    None を返す場合（fix1: どれも「採点できない」という同じ状況）:
    - 勝ち馬が1頭もいない
    - 出走が1頭以下（レースとして成立しない）
    - results が空（そもそも結果データが無い）

    OutcomeError を投げる場合（矛盾したデータ）:
    - umaban が重複している
    - 勝ち馬が starters に含まれていない
    """

    if not results:
        return None

    umabans = [r.entry.umaban for r in results]
    duplicates = sorted({u for u in umabans if umabans.count(u) > 1})
    if duplicates:
        raise OutcomeError(
            f"race_id={race_id!r} で umaban が重複しています: {duplicates}")

    starters = tuple(r.entry.umaban for r in results if r.counts_as_starter)
    excluded = tuple(r.entry.umaban for r in results if not r.counts_as_starter)
    winners = tuple(r.entry.umaban for r in results if r.is_winner)

    if not winners or len(starters) <= 1:
        return None

    missing_from_starters = [w for w in winners if w not in starters]
    if missing_from_starters:
        raise OutcomeError(
            f"race_id={race_id!r} で勝ち馬が starters に含まれていません: "
            f"{missing_from_starters}")

    return RaceOutcome(race_id=race_id, winners=winners, starters=starters, excluded=excluded)


def top_n_hit(ranking: Sequence[str], outcome: RaceOutcome, n: int) -> bool:
    """予想の上位n頭に、勝ち馬（同着ならそのいずれか）が入っているか。"""

    top = set(ranking[:n])
    return any(umaban in top for umaban in outcome.winners)


def top1_win(ranking: Sequence[str], outcome: RaceOutcome) -> bool:
    """予想の1位が勝ち馬（同着ならそのいずれか）と一致するか。"""

    if not ranking:
        return False
    return ranking[0] in outcome.winners


def _require_starter_probabilities(probabilities: Mapping[str, float], outcome: RaceOutcome) -> None:
    """``outcome.starters`` の全頭が ``probabilities`` に居ることを確認する（fix1）。

    予想が馬を1頭でも落としていたら、それは 0.0 として静かに採点してよい
    「言及されなかった」ではなく見たい欠陥なので、OutcomeError にする。
    ``probabilities`` の方が ``starters`` より多いのは正常（予想は前日に
    作るので、そのあと取消・除外になった馬の確率が残っていることがある）
    なので、キー集合の完全一致は要求しない――starters 側が欠けていないか
    だけを見る。
    """
    missing = [umaban for umaban in outcome.starters if umaban not in probabilities]
    if missing:
        raise OutcomeError(
            f"race_id={outcome.race_id!r} で starters の確率が probabilities に"
            f"ありません: {missing}")


def brier(probabilities: Mapping[str, float], outcome: RaceOutcome) -> float:
    """出走馬全頭についての Brier score（二乗誤差の和）。

    正解 y は勝ち馬なら 1/len(winners)、それ以外は 0（spec §12.2 の同着の
    割り方）。``outcome.starters`` の馬が1頭でも ``probabilities`` に無ければ
    OutcomeError（fix1: 0.0 で静かに採点しない。予想が馬を落としていたら
    見たい欠陥のため）。再正規化はしない――``probabilities`` の合計が
    1.0 未満（取消・除外馬に確率が乗っている分）でも、モデルが出した数字を
    そのまま使う。
    """

    _require_starter_probabilities(probabilities, outcome)
    y_win = 1.0 / len(outcome.winners)
    total = 0.0
    for umaban in outcome.starters:
        p = probabilities[umaban]
        y = y_win if umaban in outcome.winners else 0.0
        total += (p - y) ** 2
    return total


def log_loss(probabilities: Mapping[str, float], outcome: RaceOutcome, *, eps: float = 1e-12) -> float:
    """勝ち馬に割り当てた確率の負の対数。

    同着のときは同着馬の確率を合計する（spec §12.2）。
    「いずれかが勝つ」事象に対する確率だと解釈するため。
    p=0 で -inf にならないよう eps で clip する。

    ``brier`` と同じ理由・同じ判定で、``outcome.starters`` の馬が1頭でも
    ``probabilities`` に無ければ OutcomeError にする（fix1）。実際に和を取る
    のは winners だけだが、非勝ち馬の欠落を brier だけが検出し log_loss は
    素通りする、という非対称を避けるため starters 全体を確認する。
    """

    _require_starter_probabilities(probabilities, outcome)
    total = sum(probabilities[umaban] for umaban in outcome.winners)
    return -math.log(max(total, eps))


def log_loss_is_clipped(
    probabilities: Mapping[str, float], outcome: RaceOutcome, *, eps: float = 1e-12,
) -> bool:
    """``log_loss`` が eps 側で clip されたか（勝ち馬の確率の合計が eps 以下）。

    clip されたレースの ``log_loss`` の値は ``-log(eps)`` という**定数**
    そのもの（eps=1e-12 なら 27.63）で、実際の確率の大小を反映していない。
    100レース中44件がこれだった実測（task-19-21 fix 決定9）があり、
    headline の ``log_loss`` の値は clip 定数の選び方で動く
    （eps を 1e-9 にすると 4.4 になる）。``log_loss`` 自体の計算式は変えず
    （§15 で既に固定されている挙動）、件数だけを別途数えられるようにする。

    ``brier``/``log_loss`` と同じ判定（``_require_starter_probabilities``）
    を通す――starters の確率が欠けている行をここだけ静かに素通しすると
    「clip されていない」という誤った answer になる。
    """
    _require_starter_probabilities(probabilities, outcome)
    total = sum(probabilities[umaban] for umaban in outcome.winners)
    return total <= eps


def expected_top_n(probabilities: Mapping[str, float], ranking: Sequence[str], n: int) -> float:
    """予想の上位n頭に割り当てた確率の合計（run に依存しない期待的中数）。"""

    return sum(probabilities.get(umaban, 0.0) for umaban in ranking[:n])


def calibration_bins(pairs: Sequence[tuple[float, float]], *, bins: int = 10) -> list[dict]:
    """(予測確率, 実際の結果) の組を予測確率でビン分けし、ビンごとに集計する。

    各ビンは幅 1/bins の半開区間（最後のビンだけ両端を含む）。要素が1つも
    無いビンは返さない（test_calibration_bins_skip_empty_bins）。

    ``bins=10`` の出典: 仕様 §12.1 は「予測確率をビン分割し、ビンごとの予測
    平均と実測勝率」とだけ定めており**ビン数を指定していない**。10 は較正曲線
    （reliability diagram）で慣例的に使われる幅 0.1 のビンで、この関数の外に
    出る数字（各ビンの count / predicted_mean / actual_rate）はビン数で変わる
    ので、run どうしを比べるときは同じ値で揃えること（fix5 決定10）。
    """

    buckets: dict[int, list[tuple[float, float]]] = {}
    for predicted, actual in pairs:
        idx = min(max(int(predicted * bins), 0), bins - 1)
        buckets.setdefault(idx, []).append((predicted, actual))

    result = []
    for idx in sorted(buckets):
        items = buckets[idx]
        count = len(items)
        result.append(
            {
                "bin": idx,
                "lo": idx / bins,
                "hi": (idx + 1) / bins,
                "count": count,
                "predicted_mean": sum(p for p, _ in items) / count,
                "actual_rate": sum(a for _, a in items) / count,
            }
        )
    return result


@dataclass(frozen=True)
class MetricsReport:
    """複数レース分の指標をまとめた報告。spec §12.1「レポートには使った n を必ず書く」。"""

    #: 指標の計算に使ったレース数（skipped を除く）。
    n_races: int
    #: skipped フラグが立っていて分母から除いたレース数。
    n_skipped: int
    top_n_hit_rate: float
    top1_win_rate: float
    brier: float
    log_loss: float
    expected_top_n_mean: float
    calibration: list[dict]
    #: 上位何頭で評価したか。
    n: int
    #: n位の境界が同程度の組の中を通っていたレース数（task-19-21-supplement 決定1）。
    #: その組の中では順位が run ごとに動くので、この数が多いほど top_n_hit_rate は
    #: 揺れる。expected_top_n_mean は確率から計算するのでこの影響を受けない。
    #:
    #: ``None`` は「**測っていない**」。model 尺度（model_win_prob）の run 間
    #: ばらつきは測っていないので、その尺度では同程度の組を作らない
    #: （config.TIE_BAND は第1層の勝率で測った値。fix5 決定1）。行が1件も無い
    #: 集計（baseline のみの実行の model 側など）も ``None``。**0 にはしない**
    #: ——「0件だった」と「測っていない」は別のことを主張する（fix6 ③）。
    n_boundary_tied: int | None
    #: 取消・除外された馬に非ゼロの確率が乗っていたレース数（fix1 修正3）。
    #: この数が多いほど、その分だけ Brier は悪く出る（取消馬に配った確率は
    #: どのみち勝てないので、starters 側の確率がその分不足したまま採点される）。
    #: 再正規化はしていないので、この欄はその読み方の手がかりとして持つ。
    n_excluded_with_probability: int
    #: log_loss が eps で clip されたレース数（task-19-21 fix 決定9）。
    #: この件数が多いほど、headline の log_loss は clip 定数（-log(eps)）に
    #: 支配されており、絶対値を他の run と比べる意味が薄い。
    n_winner_probability_clipped: int


def _mean(values: Sequence[float]) -> float:
    """空なら NaN（未定義であって 0.0 ではない。falsy を absent と混同しない）。"""

    if not values:
        return float("nan")
    return sum(values) / len(values)


def aggregate(rows: Sequence[dict], *, n: int) -> MetricsReport:
    """行の並びから MetricsReport を組み立てる。

    各行は次のキーを持つ dict:
        - "skipped": bool（True ならこの行は分母に入れない）
        - "boundary_tied": bool（n位の境界が同程度の組の中だったか）。
          ``None`` は「その尺度では判定していない」で、1行でも ``None`` が
          あれば ``MetricsReport.n_boundary_tied`` は ``None``（測っていない）
          になる——判定していない行を「境界は同組でなかった」と数えない
          （fix5 決定1）
        - "excluded_probability_mass": float（取消・除外された馬に乗っていた
          確率の合計。該当馬が無ければ 0.0。fix1 修正3）
        - skipped が False のときのみ: "ranking", "probabilities", "outcome"

    "boundary_tied" / "excluded_probability_mass" が無い行は KeyError にする
    （task-19-21-supplement 決定1、fix1 修正3）。既定値で埋めると、古い行が
    混ざったときに「境界のくじ引きは無かった」「取消馬に確率は乗っていなかった」
    と黙って主張してしまう。
    """

    n_skipped = 0
    n_boundary_tied = 0
    # 行が1件も無ければ何も測っていない（0 件ではない）。baseline のみの実行では
    # model 側の行が1件も入らないので、True で初期化すると「境界が同程度だった
    # レースは0件だった」＝測ったうえでの0件を主張してしまう（fix6 ③）。
    boundary_tied_measured = len(rows) > 0
    n_excluded_with_probability = 0
    n_winner_probability_clipped = 0
    hits: list[bool] = []
    top1s: list[bool] = []
    briers: list[float] = []
    log_losses: list[float] = []
    expecteds: list[float] = []
    pairs: list[tuple[float, float]] = []

    for row in rows:
        # None は「この尺度では判定していない」。0 件に数えず、報告全体を
        # 「測っていない」にする（fix5 決定1）。
        if row["boundary_tied"] is None:
            boundary_tied_measured = False
        elif row["boundary_tied"]:
            n_boundary_tied += 1
        if row["excluded_probability_mass"] > 0.0:
            n_excluded_with_probability += 1

        if row["skipped"]:
            n_skipped += 1
            continue

        ranking = row["ranking"]
        probabilities = row["probabilities"]
        outcome = row["outcome"]

        hits.append(top_n_hit(ranking, outcome, n))
        top1s.append(top1_win(ranking, outcome))
        briers.append(brier(probabilities, outcome))
        log_losses.append(log_loss(probabilities, outcome))
        if log_loss_is_clipped(probabilities, outcome):
            n_winner_probability_clipped += 1
        expecteds.append(expected_top_n(probabilities, ranking, n))

        y_win = 1.0 / len(outcome.winners)
        for umaban in outcome.starters:
            # brier() が既に starters 全頭の存在を確認済みなので、ここでは
            # 直接引ける（.get で握りつぶさない）。
            predicted = probabilities[umaban]
            actual = y_win if umaban in outcome.winners else 0.0
            pairs.append((predicted, actual))

    return MetricsReport(
        n_races=len(hits),
        n_skipped=n_skipped,
        top_n_hit_rate=_mean(hits),
        top1_win_rate=_mean(top1s),
        brier=_mean(briers),
        log_loss=_mean(log_losses),
        expected_top_n_mean=_mean(expecteds),
        calibration=calibration_bins(pairs) if pairs else [],
        n=n,
        n_boundary_tied=n_boundary_tied if boundary_tied_measured else None,
        n_excluded_with_probability=n_excluded_with_probability,
        n_winner_probability_clipped=n_winner_probability_clipped,
    )


def compare(baseline: MetricsReport, model: MetricsReport) -> dict:
    """model - baseline の差分。境界のくじ引き件数は差ではなく両方の値をそのまま。"""

    return {
        "top_n_hit_rate": model.top_n_hit_rate - baseline.top_n_hit_rate,
        "top1_win_rate": model.top1_win_rate - baseline.top1_win_rate,
        "brier": model.brier - baseline.brier,
        "log_loss": model.log_loss - baseline.log_loss,
        "expected_top_n_mean": model.expected_top_n_mean - baseline.expected_top_n_mean,
        "n_boundary_tied_baseline": baseline.n_boundary_tied,
        "n_boundary_tied_model": model.n_boundary_tied,
    }

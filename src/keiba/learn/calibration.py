"""確率較正（isotonic regression）。

条件付きロジットが返す ``model_win_prob`` は「そのレース内で相対的にどちらが
勝ちやすいか」を softmax で正規化した値であって、絶対的な的中率と一致する
保証は無い（例えば「0.8 と答えた馬が実際に 8 割勝つ」とは限らない・過信/過小
評価になりうる）。isotonic regression は「予測確率の順序は信頼するが、その
値自体は実測の的中率に合わせて単調に補正する」――``scikit-learn`` の
``IsotonicRegression`` をそのまま使い、ここでは独自の較正アルゴリズムを書かない。

このモジュールが持たないもの:
- 重みの学習（``keiba.learn.conditional_logit`` の仕事）
- calib 用データの組み立て（``scripts/fit_weights.py`` の仕事。ここは
  ``probabilities`` / ``outcomes`` を受け取るだけで、どのレース・どの split
  から来たかを知らない）
- Jev への通信（ここでは一切行わない）

``fit_calibration`` が返す ``dict`` は ``IsotonicRegression`` オブジェクトその
ものではなく、``{"method": "isotonic", "x": [...], "y": [...]}`` という JSON に
そのまま乗る形にしてある。``integrate.Weights.calibration`` はこの dict を
そのまま保持し、``save``/``load`` で JSON と往復する（sklearn オブジェクトは
pickle 経由でしか永続化できず、``Weights`` の「JSON だけで完結する」という
契約 (Task 17/18) を壊す）。``interpolate`` は sklearn を経由せず、この
``x``/``y`` 配列に対する区分線形補間 (``numpy.interp``) だけで答えを出す
――``IsotonicRegression(out_of_bounds="clip").predict`` と同じ結果になる
（区分線形補間 + 範囲外は端の値に張り付く、という ``out_of_bounds="clip"``
の実装そのものが ``X_thresholds_``/``y_thresholds_`` 上の線形補間のため）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = ["apply_calibration", "fit_calibration", "interpolate"]


def fit_calibration(probabilities: Sequence[float], outcomes: Sequence[float]) -> dict:
    """isotonic regression を当てはめ、JSON に乗る形の較正マッピングを返す。

    ``probabilities`` はモデルの生の勝率、``outcomes`` は実際に勝ったか
    （0/1。同着なら学習時の y と同じ規則で 1/同着頭数）。長さが揃っていない
    ・0件はどちらも呼び出し側の組み立てミスなので ``ValueError``（黙って
    切り詰めたり空のマッピングを返したりしない）。

    ``y_min=0.0, y_max=1.0`` は「確率は 0〜1」という定義そのものであって、
    チューニングする閾値ではない――``criteria`` の並びのような外から与える
    べき定数ではないので、ここに固定してよい。
    """
    if len(probabilities) != len(outcomes):
        raise ValueError(
            "probabilities と outcomes の長さが一致しません: "
            f"probabilities={len(probabilities)}, outcomes={len(outcomes)}"
        )
    if len(probabilities) == 0:
        raise ValueError("probabilities/outcomes が空です。較正には1件以上必要です。")

    regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    regressor.fit(np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float))

    return {
        "method": "isotonic",
        "x": regressor.X_thresholds_.tolist(),
        "y": regressor.y_thresholds_.tolist(),
    }


def interpolate(calibration: dict, p: float) -> float:
    """較正マッピングを1つの確率に適用する。

    ``calibration["x"]``（昇順）と ``["y"]``（非減少）の間を区分線形補間する。
    学習範囲の外側は ``numpy.interp`` の既定の挙動どおり、端の値に張り付く
    （外挿しない）。``fit_calibration`` は ``y_min=0.0, y_max=1.0`` で当てて
    いるので通常は範囲内に収まるが、浮動小数の丸めで僅かに外へ出る場合に
    備えて明示的に ``[0, 1]`` へ clip する。
    """
    result = float(np.interp(p, calibration["x"], calibration["y"]))
    return min(max(result, 0.0), 1.0)


def apply_calibration(
    calibration: dict, probabilities: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    """レース1件ぶんの確率（馬番→確率）に較正を適用し、再正規化する。

    較正は馬ごとに独立に効くので、適用後の合計は 1.0 から離れうる
    （較正前は softmax でレース内合計 1.0 だったものが、isotonic の
    区分線形写像を通ると一般には保たれない）。再正規化する前の合計を
    2つ目の値として返す――1.0 から大きく離れていれば、この較正が今回の
    顔ぶれに対してどれだけ信頼できるかの兆候になるので、記録して後から
    確認できるようにする（黙って握りつぶさない）。

    再正規化後に全頭 0 になる場合（較正後の合計がちょうど 0.0。全馬の
    較正後確率が 0 になった、退化したケース）は等分に割り、0/0 の
    ``nan`` を出さない。``0.0`` は「まだ答えていない」ではなく「較正が
    そう返した」という実在の値なので、``if raw_sum:`` のような falsy 判定
    ではなく明示的に ``== 0.0`` で分岐する。
    """
    calibrated = {umaban: interpolate(calibration, p) for umaban, p in probabilities.items()}
    # sum() は空 (probabilities が {}) のとき int 0 を返す。戻り値の型は
    # tuple[dict[str, float], float] なので float に包む（fix round 2 Minor 3）。
    raw_sum = float(sum(calibrated.values()))

    if not calibrated:
        return {}, raw_sum
    if raw_sum == 0.0:
        uniform = 1.0 / len(calibrated)
        return {umaban: uniform for umaban in calibrated}, raw_sum

    normalized = {umaban: value / raw_sum for umaban, value in calibrated.items()}
    return normalized, raw_sum

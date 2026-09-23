"""条件付きロジット（Luce モデル）の最尤推定。

`keiba.predict.integrate` の forward 方向（`softmax_by_race` / `model_scores` /
`standardize`）を学習する側。レース単位の softmax の分母がそのレースの出走馬
だけであることが Luce モデルの核心で、これにより頭数が違うレースを同じ尺度で
扱える（CLAUDE.md §13）。ここで学ぶ `w` は `score_i = w · x_i` の `w` そのもので、
`integrate.model_scores` が計算する score と同じ量になるよう、学習も推論と同じ
「標準化してから内積」という手順で行う。

このモジュールが持たないもの:
- Jev への通信（jev_client.py / predict/layers.py の仕事）
- state の組み立て（features/*.py の仕事）
- `Weights` の構築・JSON 往復（Task 18。ここは `FitResult` を返すだけ）
- `standardize_mask` を決めること（`integrate.standardize_mask_for` の仕事。
  ここは `integrate` を import しない――どの列を標準化するかは呼び出し側の
  責任で、`fit` は渡された mask をそのまま使うだけ。Task 18 が配線する）

fix1（レビュー対応）で変わった契約:
- `fit_standardization` が返す `standardization` は **`feature_names` の全列を覆う**
  （分散0で落とした列も `(mean, 0.0)` で載る）。そうしないと
  `integrate.standardize` が「重みファイルと特徴集合が食い違っている」と
  誤認して `WeightsIncompatibleError` を投げる（学習も収束も成功したモデルが
  そのまま予測に使えなくなる、という実際に踏まれた不具合）。
- `standardize_mask` はキーワード専用の必須引数。既定値も `None` も認めない
  （`integrate.standardize` が14列を無条件に素通しするのに `fit` が既定で
  全列を標準化すると、その列の寄与が学習時と推論時で std 倍ずれる）。

fix2（再レビュー対応）で変わった契約:
- `fit_standardization` の `mask` もキーワード専用の必須引数にした（`None`
  も拒否）。`fit` の `standardize_mask` を必須にしても、`fit_standardization`
  を直接 `mask=None` で呼べば同じ形の不具合が1段下に残るため。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

__all__ = ["FitResult", "fit", "fit_standardization", "neg_log_likelihood"]


@dataclass(frozen=True)
class FitResult:
    """`fit` の結果一式。`Weights`（Task 18）の材料になる。

    `weights` / `feature_names` は同じ長さ・同じ並びで対応する（`feature_names`
    は `fit` に渡されたものをそのまま保持する――標準化できずに落とした特徴も
    含む全体集合。落とした特徴は `weights` 側で 0.0 になり、`dropped_features`
    に名前が載る。呼び出し側が `integrate.feature_names()` の全体集合と
    突き合わせられるようにするため）。

    `standardization` は `feature_names` の全列を覆う（fix1 決定1。分散0で
    落とした列も `(mean, 0.0)` で載る）。`weights` は frozen dataclass の外側
    からの再代入は防げても中身は可変な `np.ndarray` なので、書き換え不可に
    してある（`weights.flags.writeable is False`）。
    """

    weights: np.ndarray
    feature_names: tuple[str, ...]
    standardization: dict[str, tuple[float, float]]
    dropped_features: tuple[str, ...]
    lambda_: float
    log_likelihood: float
    n_races: int
    n_rows: int
    converged: bool
    #: 分散0ではないが、列のスケールに対して std が **float64 では区別できない**
    #: ほど小さい列。**落とさない**——判定を変えずに名前だけ残し、
    #: ``scripts/fit_weights.py`` が警告を出す（fix5 決定9-2）。標準化すると
    #: (value - mean)/std が極端に増幅され、その列の重みが解釈不能になる。
    #:
    #: 名前どおり、これは**丸め誤差で std がぴったり 0 にならなかっただけの列**
    #: の検出器である（閾値は sqrt(machine epsilon)）。統計的に情報の無い列
    #: （相対 std が 1e-4 級の「実質定数」）は捕まえない——そこに線を引ける
    #: 出典が無いため（fix6 ⑤-3 で「準定数（near_constant）」から改名した。
    #: 前の名前は捕まえる範囲を実際より広く主張していた）。
    float64_indistinguishable_features: tuple[str, ...]


def neg_log_likelihood(
    w: np.ndarray,
    X: np.ndarray,
    group_starts: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    """条件付きロジットの負の対数尤度と勾配。

        L(w) = Σ_races Σ_i y_i * (score_i - log Σ_j exp(score_j))

    y は勝ち馬に 1（同着なら 1/同着頭数）。レースごとに Σ y_i == 1 になる。
    勾配は Σ_races Σ_i (y_i - p_i) * x_i。p はレース内 softmax。
    L2 項は目的関数に + l2 * ||w||^2、勾配に + 2 * l2 * w。

    この式は Σ_race y_i == 1 のときだけ正しい（`fit` がその前提を検証してから
    ここへ渡す。0.5/0.5 の同着はレビュアーが手で導いた一般形と厳密に一致する
    ―― Σy=1 を満たす限り、同着かどうかで式は変わらない）。

    `X` / `y` は `group_starts` の境界どおりに、あらかじめレース単位で
    連続に並んでいる前提（`fit` が並べ替えてから渡す）。
    """
    scores = X @ w
    total = 0.0
    gradient = np.zeros_like(w)
    for start, end in zip(group_starts[:-1], group_starts[1:]):
        s = scores[start:end]
        shifted = s - s.max()  # オーバーフロー対策
        exponentials = np.exp(shifted)
        log_denominator = np.log(exponentials.sum()) + s.max()
        yy = y[start:end]
        total += float((yy * (s - log_denominator)).sum())
        probabilities = exponentials / exponentials.sum()
        gradient += (yy - probabilities) @ X[start:end]
    return -total + l2 * float(w @ w), -gradient + 2.0 * l2 * w


#: 「列のスケールに対して std が float64 では区別できない」とみなす相対許容差。
#:
#: 出典は数値精度のみ: float64 の機械epsilon の平方根
#: （``sqrt(np.finfo(float).eps)`` ≒ 1.49e-8。scipy の有限差分の既定ステップと
#: 同じ尺度で、「この桁より下は浮動小数では区別できない」の慣用値）。
#: **「実質定数だから落とす」と言える閾値ではない**——それには出典が無いので、
#: この値は警告を出す目印としてしか使わない（fix5 決定9-2）。この閾値は
#: 丸め誤差レベルの検出であり、統計的に情報の無い列（相対 std 1e-4 級）は
#: 捕まえない（fix6 ⑤-3）。
_FLOAT64_INDISTINGUISHABLE_TOLERANCE = math.sqrt(np.finfo(float).eps)


def fit_standardization(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    mask: Sequence[bool],
) -> tuple[dict[str, tuple[float, float]], tuple[str, ...], tuple[str, ...]]:
    """列ごとの (mean, std) を求める。

    `standardization` は必ず `feature_names` の全列を覆う（fix1 決定1）。
    `integrate.standardize` は `standardization` に無い列を
    `WeightsIncompatibleError` にするので、標準化できない列（分散0）も
    `(mean, 0.0)` で載せておく――std==0 は `integrate.standardize` 側の
    「値をそのまま通す」合図でもある（`test_standardize_leaves_feature_when_std_is_zero`
    が既に固定している契約）。分散0の列は `dropped` にも名前を積む
    （`fit` 側でその列の重みを 0 に固定し、最適化の対象から外す）。

    `mask` はキーワード専用の必須引数で `None` も拒否する（fix2 決定5）。
    `fit` 側で `standardize_mask` を必須にしても、この関数を直接
    `mask=None` で呼べば「全列を標準化する」という Critical 2 と同じ形の
    不具合が1段下に残ってしまう――呼び出し元は `fit` だけだが、それでも
    穴を塞ぐ。`mask` が `False` の列は実データの平均・分散を計算せず、
    恒等変換 `(0.0, 1.0)` を割り当てる（`logit_layer1` や `*_missing` の
    ような、既に意味のある尺度に乗っている列。`integrate.standardize` が
    無条件に素通しする列を学習側で標準化すると、その列の寄与が学習時と
    推論時で std 倍ずれる――fix1 決定2）。

    返り値は ``(standardization, dropped, float64_indistinguishable)`` の3つ組。
    3つめは分散0ではないが列のスケールに対して std が float64 では区別できない
    ほど小さい列の名前（``_FLOAT64_INDISTINGUISHABLE_TOLERANCE``）。
    **落とす対象は変えない**（分散0の列だけ）——相対閾値に出典が無いので、
    名前を呼び出し側に渡して警告させるだけにする（fix5 決定9-2）。丸め誤差
    レベルの検出であって、統計的に情報の無い列の検出ではない（fix6 ⑤-3）。
    """
    if mask is None:
        raise ValueError(
            "mask は省略・None を許しません。どの列を標準化するか明示して"
            "ください（integrate.standardize_mask_for が既定の規則を提供します）。"
        )
    standardization: dict[str, tuple[float, float]] = {}
    dropped: list[str] = []
    indistinguishable: list[str] = []
    for j, name in enumerate(feature_names):
        if not mask[j]:
            standardization[name] = (0.0, 1.0)
            continue
        column = X[:, j]
        mean = float(column.mean())
        std = float(column.std())
        if std == 0.0:
            dropped.append(name)
        else:
            # 列のスケールは平均の大きさと値域の広さのうち大きいほうで見る
            # （平均0の列で相対比較が意味を失わないように）。
            scale = max(abs(mean), float(np.abs(column).max()))
            if scale > 0.0 and std <= scale * _FLOAT64_INDISTINGUISHABLE_TOLERANCE:
                indistinguishable.append(name)
        standardization[name] = (mean, std)
    return standardization, tuple(dropped), tuple(indistinguishable)


def fit(
    X: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2: float = 0.1,
    standardize_mask: Sequence[bool],
) -> FitResult:
    """条件付きロジットの重みを最尤推定する。

    `standardize_mask` はキーワード専用の必須引数（fix1 決定2）。`None` も
    省略も認めない――どの列を標準化するかは `integrate.standardize` が実際に
    素通しする列（`logit_layer1` / `*_missing` / `*_insufficient`）と一致して
    いなければならず、既定値を置くと「黙って壊れる側」になってしまう
    （`integrate.standardize_mask_for` が正しい mask を作る。Task 18 が配線する
    ――このモジュールは `integrate` を import しない）。

    手順:
    1. 入力を検証する（形状・重複列名・`standardize_mask` の長さ・非負・有限）。
    2. `groups` の昇順に行を並べ替え、レースごとの開始行インデックス
       (`group_starts`) を作る。レースごとに `Σy` が 1 であることを検証する
       （`math.isclose` の既定の相対許容差を使う。3頭同着の 1/3+1/3+1/3 が
       浮動小数で `0.9999999999999998` になっても通す一方、`Σy=2` のような
       記録ミスは弾く）。
    3. 学習データから標準化パラメータを求める（`fit_standardization`）。
       分散 0 の列はここで学習対象から外す。
    4. 標準化した特徴で `neg_log_likelihood` を `scipy.optimize.minimize`
       （`L-BFGS-B`、解析的勾配）にかける。
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    feature_names = tuple(feature_names)

    if X.ndim != 2:
        raise ValueError(f"X は2次元配列である必要があります: shape={X.shape}")
    n_rows, n_features = X.shape
    if n_rows == 0:
        raise ValueError("学習データが空です。")
    # l2 == 0.0 は「正則化しない」という実在の指定なので通す（truthiness で
    # 判定すると 0.0 が False 扱いになり弾かれてしまう）。負の値は目的関数
    # -||w||^2 の項を作って非凸にするので拒否する（fix round 2 Minor 5）。
    if l2 < 0:
        raise ValueError(f"l2 は 0 以上である必要があります: l2={l2!r}")
    if len(groups) != n_rows or len(y) != n_rows:
        raise ValueError(
            "X・groups・y の行数が一致しません: "
            f"X={n_rows}, groups={len(groups)}, y={len(y)}"
        )
    if n_features != len(feature_names):
        raise ValueError(
            f"X の列数 ({n_features}) と feature_names の長さ "
            f"({len(feature_names)}) が一致しません。"
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"feature_names に重複があります: {feature_names}")
    if standardize_mask is None:
        raise ValueError(
            "standardize_mask は省略できません。どの列を標準化するか明示して"
            "ください（integrate.standardize_mask_for が既定の規則を提供します）。"
        )
    if len(standardize_mask) != n_features:
        raise ValueError(
            f"standardize_mask の長さ ({len(standardize_mask)}) が"
            f" X の列数 ({n_features}) と一致しません。"
        )
    if not np.isfinite(X).all():
        bad_row, bad_col = (int(i) for i in np.argwhere(~np.isfinite(X))[0])
        raise ValueError(
            f"X に有限でない値があります: row={bad_row}, col={bad_col}"
            f" (value={X[bad_row, bad_col]!r})"
        )
    if not np.isfinite(y).all():
        bad_row = int(np.argwhere(~np.isfinite(y))[0][0])
        raise ValueError(f"y に有限でない値があります: row={bad_row} (value={y[bad_row]!r})")
    if np.any(y < 0):
        raise ValueError("y に負の値があります。")

    order = np.argsort(groups, kind="stable")
    X_sorted = X[order]
    y_sorted = y[order]
    groups_sorted = groups[order]

    unique_groups, start_indices = np.unique(groups_sorted, return_index=True)
    group_starts = np.append(start_indices, n_rows)
    n_races = len(unique_groups)

    for race_index, (start, end) in enumerate(zip(group_starts[:-1], group_starts[1:])):
        total = float(y_sorted[start:end].sum())
        if not math.isclose(total, 1.0):
            raise ValueError(
                f"レース (group={unique_groups[race_index]!r}) の y の合計が 1"
                f" ではありません: 合計={total}。単独勝ちなら1、同着ならその"
                "頭数で均等に割った値を使ってください。"
            )

    standardization, dropped_features, indistinguishable_features = fit_standardization(
        X_sorted, feature_names, mask=standardize_mask
    )

    dropped_set = set(dropped_features)
    keep_idx = [i for i, name in enumerate(feature_names) if name not in dropped_set]

    X_std = X_sorted.copy()
    for i, name in enumerate(feature_names):
        if name in dropped_set:
            continue
        mean, std = standardization[name]
        X_std[:, i] = (X_sorted[:, i] - mean) / std
    X_reduced = X_std[:, keep_idx]

    w0 = np.zeros(len(keep_idx))
    result = optimize.minimize(
        neg_log_likelihood,
        w0,
        args=(X_reduced, group_starts, y_sorted, l2),
        jac=True,
        method="L-BFGS-B",
    )

    full_weights = np.zeros(len(feature_names))
    for reduced_i, orig_i in enumerate(keep_idx):
        full_weights[orig_i] = result.x[reduced_i]
    full_weights.flags.writeable = False

    pure_nll, _ = neg_log_likelihood(result.x, X_reduced, group_starts, y_sorted, 0.0)

    return FitResult(
        weights=full_weights,
        feature_names=feature_names,
        standardization=standardization,
        dropped_features=dropped_features,
        lambda_=l2,
        log_likelihood=-pure_nll,
        n_races=n_races,
        n_rows=n_rows,
        converged=bool(result.success),
        float64_indistinguishable_features=indistinguishable_features,
    )

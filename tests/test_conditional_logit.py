"""keiba.learn.conditional_logit のテスト。

条件付きロジット（Luce モデル）の最尤推定が本題。要は「モデル自身が生成した
合成データから、真の重みを復元できるか」（spec §11.1）。復元できなければ
勾配か尤度の式そのものが間違っている――許容誤差を緩めて通すのではなく、
実装を疑う。

fix1（レビュー対応）で `make_synthetic` を「列ごとに異なるスケール・非ゼロ平均を
持つ」形に直した。標準正規のままだと std≈1・平均≈0 になり、`fit` が内部で行う
標準化の工程を通しても通さなくても答えがほぼ変わらず、Critical 2（標準化しては
いけない列を標準化してしまう不具合）をどのテストも検出できていなかった。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize

import keiba.learn.conditional_logit as cl
from keiba.learn.conditional_logit import fit, fit_standardization, neg_log_likelihood

#: scipy.optimize.approx_fprime の既定ステップ幅と同じ値を、出典の無い数値
#: リテラルを書かずに導く。
_FPRIME_EPS = np.sqrt(np.finfo(float).eps)


def make_synthetic(rng, *, n_races, field_size, true_w, scales, offsets):
    """条件付きロジットの生成モデルそのままでデータを作る。

    列ごとに scale と offset を変える。標準正規のままだと std≈1・平均≈0 に
    なり、標準化の工程を通しても通さなくても同じ答えになってしまう
    （実際にそれでテストが空振りしていた）。`scales`/`offsets` は呼び出し側が
    明示する（この関数の中に魔法の既定値は置かない）。

    真の重みは raw の特徴（このデータそのもの。scale・offset 適用後）に
    対するものなので、標準化した行列で `fit` した重みとは一致しない
    （offset は同一レース内の全頭に同じだけ加わる定数なので softmax の中で
    打ち消えるが、scale は打ち消えない――標準化空間での重みは
    `true_w * scales` になる）。
    """
    X, groups, y = [], [], []
    scales_arr = np.asarray(scales, dtype=float)
    offsets_arr = np.asarray(offsets, dtype=float)
    for race in range(n_races):
        raw = rng.normal(size=(field_size, len(true_w)))
        x = raw * scales_arr + offsets_arr
        scores = x @ true_w
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        winner = rng.choice(field_size, p=probabilities)
        for i in range(field_size):
            X.append(x[i])
            groups.append(race)
            y.append(1.0 if i == winner else 0.0)
    return np.array(X), np.array(groups), np.array(y)


def make_group_starts(n_races, field_size):
    return np.arange(0, (n_races + 1) * field_size, field_size)


class TestRecoversKnownWeights:
    def test_recovers_known_weights_from_synthetic_data(self):
        """看板テスト。真の重みは大きくしない（決定: ±0.15 は絶対誤差なので
        大きい真値には自動的に通用しない。レビュアー実測で w=8.0 なら誤差0.23）。
        """
        rng = np.random.default_rng(20260922)
        true_w = np.array([1.5, -0.8])
        X, groups, y = make_synthetic(
            rng, n_races=3000, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=1e-6,
            standardize_mask=[True, True],
        )

        assert result.converged
        assert np.max(np.abs(result.weights - true_w)) <= 0.15, result.weights

    def test_recovers_single_feature(self):
        rng = np.random.default_rng(7)
        true_w = np.array([1.2])
        X, groups, y = make_synthetic(
            rng, n_races=3000, field_size=8, true_w=true_w,
            scales=[1.0], offsets=[0.0],
        )

        result = fit(
            X, groups, y, feature_names=("f0",), l2=1e-6,
            standardize_mask=[True],
        )

        assert result.converged
        assert abs(result.weights[0] - true_w[0]) <= 0.15, result.weights


class TestGradient:
    def test_gradient_matches_numerical_derivative(self):
        rng = np.random.default_rng(3)
        n_races, field_size, n_features = 40, 5, 3
        X = rng.normal(size=(n_races * field_size, n_features))
        group_starts = make_group_starts(n_races, field_size)
        y = np.zeros(n_races * field_size)
        for race in range(n_races):
            winner = rng.integers(field_size)
            y[race * field_size + winner] = 1.0
        w = rng.normal(size=n_features)
        l2 = 0.1

        def objective(weights):
            value, _ = neg_log_likelihood(weights, X, group_starts, y, l2)
            return value

        analytic = neg_log_likelihood(w, X, group_starts, y, l2)[1]
        numerical = optimize.approx_fprime(w, objective, _FPRIME_EPS)

        assert np.max(np.abs(analytic - numerical)) <= 1e-5, (analytic, numerical)

    def test_dead_heat_gradient_matches_the_numerical_derivative(self):
        """有限性だけでは Σy=2 のような不正データも通ってしまう（実際に空振り
        していた変異: `yy = y[start:end]` → `(y[start:end] > 0)`）。0.5/0.5・
        1/3ずつの同着を含むデータで、解析勾配が数値微分と一致することまで見る。
        """
        rng = np.random.default_rng(15)
        n_features = 3
        X = rng.normal(size=(12, n_features)) * np.array([10.0, 0.5, 1.0]) + np.array(
            [100.0, -3.0, 0.0]
        )
        group_starts = np.array([0, 4, 8, 12])
        y = np.zeros(12)
        y[0] = 1.0  # race0: 単独勝ち
        y[4] = 0.5  # race1: 2頭同着
        y[5] = 0.5
        y[8] = 1.0 / 3.0  # race2: 3頭同着
        y[9] = 1.0 / 3.0
        y[10] = 1.0 / 3.0
        w = rng.normal(size=n_features)
        l2 = 0.1

        def objective(weights):
            value, _ = neg_log_likelihood(weights, X, group_starts, y, l2)
            return value

        analytic = neg_log_likelihood(w, X, group_starts, y, l2)[1]
        numerical = optimize.approx_fprime(w, objective, _FPRIME_EPS)

        assert np.max(np.abs(analytic - numerical)) <= 1e-5, (analytic, numerical)


class TestRegularisation:
    def test_likelihood_decreases_with_regularisation(self):
        rng = np.random.default_rng(11)
        true_w = np.array([1.5, -0.8])
        X, groups, y = make_synthetic(
            rng, n_races=1000, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        weak = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=1e-6,
            standardize_mask=[True, True],
        )
        strong = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=10.0,
            standardize_mask=[True, True],
        )

        assert np.linalg.norm(strong.weights) < np.linalg.norm(weak.weights)

    def test_l2_zero_is_not_the_default(self):
        """l2=0.0（正則化なし、という実在の値）が `l2 or 0.1` のようなフォール
        バックで既定値に化けていないことを固定する。この変異は Important 5 の
        表で「l2=0.0 を使うテストが1本も無い」と指摘された空振り箇所。
        """
        rng = np.random.default_rng(25)
        true_w = np.array([1.5, -0.8])
        X, groups, y = make_synthetic(
            rng, n_races=1000, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        zero = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.0,
            standardize_mask=[True, True],
        )
        regularised = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )

        assert zero.lambda_ == 0.0
        assert not np.allclose(zero.weights, regularised.weights)
        assert np.linalg.norm(zero.weights) > np.linalg.norm(regularised.weights)

    def test_l2_negative_raises(self):
        """l2 < 0 は目的関数の -l2*||w||^2 項を作り非凸にする (fix round 2
        Minor 5)。l2 == 0.0 は実在の値なので通ることも合わせて確認する
        （truthiness で判定していないことの固定）。
        """
        rng = np.random.default_rng(7)
        true_w = np.array([1.0, -0.5])
        X, groups, y = make_synthetic(
            rng, n_races=100, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        with pytest.raises(ValueError, match="l2"):
            fit(X, groups, y, feature_names=("f0", "f1"), l2=-0.1,
                standardize_mask=[True, True])

        # l2 == 0.0 は落ちない (境界値)。
        fit(X, groups, y, feature_names=("f0", "f1"), l2=0.0,
            standardize_mask=[True, True])


class TestFieldSize:
    def test_handles_variable_field_size(self):
        rng = np.random.default_rng(13)
        true_w = np.array([1.0, -0.5])
        X8, groups8, y8 = make_synthetic(
            rng, n_races=200, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )
        X18, groups18, y18 = make_synthetic(
            rng, n_races=200, field_size=18, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )
        groups18 = groups18 + groups8.max() + 1  # レース ID が衝突しないようにずらす

        X = np.concatenate([X8, X18])
        groups = np.concatenate([groups8, groups18])
        y = np.concatenate([y8, y18])

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=1e-6,
            standardize_mask=[True, True],
        )

        assert result.converged
        assert result.n_races == 400
        assert result.n_rows == 200 * 8 + 200 * 18
        assert np.all(np.isfinite(result.weights))

    def test_rows_of_the_same_race_may_be_shuffled(self):
        """`np.argsort(groups, kind="stable")` を `np.arange(n_rows)` に
        差し替える変異が空振りしていた箇所。同一レースの行が隣接していない
        入力でも、あらかじめソートされた入力と同じ重みが返らなければならない。
        """
        rng = np.random.default_rng(12)
        true_w = np.array([1.0, -0.5])
        X, groups, y = make_synthetic(
            rng, n_races=300, field_size=6, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        permutation = np.random.default_rng(99).permutation(len(groups))
        X_shuffled, groups_shuffled, y_shuffled = (
            X[permutation], groups[permutation], y[permutation],
        )

        sorted_result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )
        shuffled_result = fit(
            X_shuffled, groups_shuffled, y_shuffled, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )

        assert np.allclose(sorted_result.weights, shuffled_result.weights, atol=1e-6)


class TestDeadHeat:
    def test_handles_dead_heat_as_shared_outcome(self):
        # 3レース、各4頭。うち1レースだけ1着同着（y が 0.5 ずつ）。
        rng = np.random.default_rng(5)
        X = rng.normal(size=(12, 2))
        groups = np.repeat([0, 1, 2], 4)
        y = np.zeros(12)
        y[0] = 1.0  # race 0: 単独勝ち
        y[4] = 0.5  # race 1: 同着
        y[5] = 0.5
        y[8] = 1.0  # race 2: 単独勝ち

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )

        assert np.isfinite(result.log_likelihood)

    def test_three_way_dead_heat_is_accepted(self):
        """3頭同着（1/3 ずつ）は現実的なケースとして残す。現物の `np.sum` は
        この頭数ではたまたま厳密に 1.0 になる（fix2: `k=3` は
        `test_math_isclose_is_needed_for_seven_way_ties` の表に無い）。
        """
        rng = np.random.default_rng(10)
        X = rng.normal(size=(4, 2))
        groups = np.zeros(4, dtype=int)
        y = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )

        assert np.isfinite(result.log_likelihood)

    def test_math_isclose_is_needed_for_seven_way_ties(self):
        """fix1 report の「1/3 ずつ3頭の同着は np.sum で 0.9999999999999998 に
        なる」という記述は誤りだった（レビュアーが検算: `np.sum` が厳密に
        1.0 にならないのは k=6, 7, 13, 14, 15, 19〜23 で、k=3 は含まれない。
        Python の組み込み `sum` はこの範囲（k=2〜24）でずれない）。

        7頭同着（1/7 ずつ、k=7）の `np.sum([1/7]*7)` は実測で
        `0.9999999999999998` になり、厳密比較 `total != 1.0` に変えると
        このテストが落ちる。`math.isclose` の既定の相対許容差が効いている
        ことをここで固定する。
        """
        total = float(np.sum(np.full(7, 1.0 / 7.0)))
        assert total != 1.0  # 厳密には1.0にならないことの前提を固定する
        assert math.isclose(total, 1.0)

        rng = np.random.default_rng(18)
        X = rng.normal(size=(7, 2))
        groups = np.zeros(7, dtype=int)
        y = np.full(7, 1.0 / 7.0)

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )

        assert np.isfinite(result.log_likelihood)

    def test_raises_when_a_race_does_not_sum_to_one(self):
        """正しく記録された同着（Σy=1）ではなく、記録ミスで Σy≠1 になった
        レースだけを弾く。メッセージにどのレースか（group ラベル）と実際の
        合計値が両方入っていることを `match=` で固定する。
        """
        rng = np.random.default_rng(8)
        X = rng.normal(size=(8, 2))
        groups = np.repeat([0, 1], 4)
        y = np.zeros(8)
        y[0] = 1.0
        y[1] = 1.0  # race 0 (group=0): 合計2.0（誤り。同着なら0.5ずつのはず）
        y[4] = 1.0  # race 1: 正常

        with pytest.raises(ValueError, match=r"group=.*0.*合計=2\.0"):
            fit(
                X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True, True],
            )


class TestValidation:
    def test_raises_on_race_without_winner(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(8, 2))
        groups = np.repeat([0, 1], 4)
        y = np.zeros(8)
        y[0] = 1.0  # race 0 には勝ち馬がいるが、race 1 には誰もいない（Σy=0）

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True, True])

    def test_raises_on_mismatched_lengths(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(8, 2))
        groups = np.repeat([0, 1], 4)
        y = np.zeros(7)  # X より1行少ない
        y[0] = 1.0

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True, True])

    def test_empty_input_raises(self):
        X = np.empty((0, 2))
        groups = np.array([])
        y = np.array([])

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True, True])

    def test_rejects_negative_y(self):
        """負値は「同着でも記録ミスでもない」明確な入力誤り。Σy=1 の
        チェックだけに頼ると `[1.5, -0.5, 0, 0]`（合計1.0）のような
        辻褄合わせを見逃すので、独立に検証する。
        """
        rng = np.random.default_rng(6)
        X = rng.normal(size=(4, 2))
        groups = np.zeros(4, dtype=int)
        y = np.array([1.5, -0.5, 0.0, 0.0])  # 合計は1.0だが負値を含む

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True, True])

    def test_raises_on_non_finite_values(self):
        """`match="有限"` で、Σy 検証（NaN は `math.isclose` でも False になり
        ValueError を出しうる）ではなく、意図どおり有限性チェックの方で
        落ちていることを区別する。行・列の情報がメッセージに入っていることも
        `row=`/`col=` の値まで固定する。
        """
        rng = np.random.default_rng(14)
        base_X = rng.normal(size=(8, 2))
        groups = np.repeat([0, 1], 4)
        base_y = np.zeros(8)
        base_y[0] = 1.0
        base_y[4] = 1.0

        X_nan = base_X.copy()
        X_nan[0, 0] = np.nan
        X_inf = base_X.copy()
        X_inf[1, 1] = np.inf
        y_nan = base_y.copy()
        y_nan[2] = np.nan

        cases = [
            (X_nan, base_y, r"有限.*row=0, col=0"),
            (X_inf, base_y, r"有限.*row=1, col=1"),
            (base_X, y_nan, r"有限.*row=2"),
        ]
        for X_bad, y_bad, expected in cases:
            with pytest.raises(ValueError, match=expected):
                fit(
                    X_bad, groups, y_bad, feature_names=("f0", "f1"),
                    standardize_mask=[True, True],
                )

    def test_standardize_mask_is_required(self):
        rng = np.random.default_rng(16)
        X, groups, y = make_synthetic(
            rng, n_races=5, field_size=4, true_w=np.array([1.0]),
            scales=[1.0], offsets=[0.0],
        )

        with pytest.raises(TypeError):
            fit(X, groups, y, feature_names=("f0",))  # standardize_mask を渡さない

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0",), standardize_mask=None)  # 明示的な None も不可

    def test_rejects_a_mask_of_the_wrong_length(self):
        rng = np.random.default_rng(2)
        X, groups, y = make_synthetic(
            rng, n_races=5, field_size=4, true_w=np.array([1.0, 1.0]),
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True])
        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("f0", "f1"), standardize_mask=[True, True, True])

    def test_rejects_duplicate_feature_names(self):
        rng = np.random.default_rng(4)
        X, groups, y = make_synthetic(
            rng, n_races=5, field_size=4, true_w=np.array([1.0, 1.0]),
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        with pytest.raises(ValueError):
            fit(X, groups, y, feature_names=("a", "a"), standardize_mask=[True, True])


class TestStandardization:
    def test_standardization_skips_masked_features(self):
        X = np.column_stack(
            [
                np.array([1.0, 2.0, 3.0, 4.0]),  # 標準化対象。mean=2.5, std!=1
                np.array([0.0, 1.0, 0.0, 1.0]),  # マスクで除外する指示変数もどき
            ]
        )
        standardization, dropped, _indistinguishable = fit_standardization(
            X, ("scaled", "raw_flag"), mask=[True, False]
        )

        assert dropped == ()
        mean, std = standardization["scaled"]
        assert mean == pytest.approx(X[:, 0].mean())
        assert std == pytest.approx(X[:, 0].std())
        # マスクが False の列は「平均0・分散1にしない」= 恒等変換のパラメータ。
        assert standardization["raw_flag"] == (0.0, 1.0)

    def test_drops_constant_features(self):
        rng = np.random.default_rng(9)
        true_w = np.array([1.0])
        X_varying, groups, y = make_synthetic(
            rng, n_races=500, field_size=8, true_w=true_w,
            scales=[1.0], offsets=[0.0],
        )
        X = np.column_stack([X_varying[:, 0], np.full(X_varying.shape[0], 7.0)])

        result = fit(
            X, groups, y, feature_names=("varying", "constant"), l2=1e-6,
            standardize_mask=[True, True],
        )

        assert "constant" in result.dropped_features
        # fix1 決定1: 落とした列も standardization には (mean, 0.0) で載る
        # （integrate.standardize が「無い列」を重み不整合と誤認しないように）。
        assert "constant" in result.standardization
        assert result.standardization["constant"] == (7.0, 0.0)
        constant_index = result.feature_names.index("constant")
        assert result.weights[constant_index] == 0.0

    def test_zero_variance_column_is_still_in_standardization(self):
        X = np.column_stack([np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 9.0)])

        standardization, dropped, _indistinguishable = fit_standardization(
            X, ("varying", "constant"), mask=[True, True]
        )

        assert "constant" in dropped
        assert standardization["constant"] == (9.0, 0.0)

    def test_standardization_covers_every_feature_name(self):
        X = np.column_stack(
            [
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.full(4, 9.0),
                np.array([0.0, 1.0, 0.0, 1.0]),
            ]
        )
        feature_names = ("varying", "constant", "masked")

        standardization, _dropped, _indistinguishable = fit_standardization(
            X, feature_names, mask=[True, True, False])

        assert set(standardization) == set(feature_names)

    def test_standardization_is_applied_to_the_fitted_matrix(self):
        """標準化を学習行列に適用しない／std で割らない、という2つの変異が
        空振りしていた箇所（Important 5）。列ごとに違うスケール・非ゼロ平均を
        持つデータで、標準化した重みが `true_w * scales` に一致し、標準化を
        外す（mask 全 False）と明確に違う値になることを固定する。

        scales=[10.0, 0.5] は brief の再現例と同じ値。true_w_raw は
        `true_w_raw * scales == [1.5, -0.8]`（看板テストで誤差0.15以内に収まる
        ことが分かっている組み合わせ）になるように選んだ。
        """
        rng = np.random.default_rng(23)
        scales = [10.0, 0.5]
        offsets = [100.0, -3.0]
        true_w_raw = np.array([0.15, -1.6])
        X, groups, y = make_synthetic(
            rng, n_races=3000, field_size=8, true_w=true_w_raw,
            scales=scales, offsets=offsets,
        )

        standardized = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=1e-6,
            standardize_mask=[True, True],
        )
        not_standardized = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=1e-6,
            standardize_mask=[False, False],
        )

        target = true_w_raw * np.array(scales)
        assert np.max(np.abs(standardized.weights - target)) <= 0.15, standardized.weights
        # 標準化を外すと明確に違う（raw スケールの、より小さい）値が返る。
        assert np.max(np.abs(not_standardized.weights - standardized.weights)) > 1.0

    def test_log_likelihood_excludes_the_l2_penalty(self):
        """`log_likelihood` に L2 罰則を含める、という変異が空振りしていた
        箇所。`FitResult.standardization` を使って学習行列を手元で再現し、
        `neg_log_likelihood(..., l2=0.0)` から求めた純粋な対数尤度と
        `result.log_likelihood` が（l2 をいくつに変えても）一致することを見る。
        """
        rng = np.random.default_rng(31)
        true_w = np.array([1.0, -0.6])
        X, groups, y = make_synthetic(
            rng, n_races=500, field_size=6, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )
        feature_names = ("f0", "f1")

        for l2 in (0.0, 0.1, 5.0):
            result = fit(
                X, groups, y, feature_names=feature_names, l2=l2,
                standardize_mask=[True, True],
            )

            order = np.argsort(groups, kind="stable")
            X_sorted, y_sorted, groups_sorted = X[order], y[order], groups[order]
            _, start_indices = np.unique(groups_sorted, return_index=True)
            group_starts = np.append(start_indices, len(groups_sorted))

            X_std = X_sorted.copy()
            for j, name in enumerate(feature_names):
                mean, std = result.standardization[name]
                if std != 0.0:
                    X_std[:, j] = (X_sorted[:, j] - mean) / std

            pure_nll, _ = neg_log_likelihood(result.weights, X_std, group_starts, y_sorted, 0.0)
            assert result.log_likelihood == pytest.approx(-pure_nll, abs=1e-8), l2


class TestConvergence:
    def test_converged_flag_reflects_optimiser(self):
        rng = np.random.default_rng(17)
        true_w = np.array([1.0, -1.0])
        X, groups, y = make_synthetic(
            rng, n_races=500, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=0.1,
            standardize_mask=[True, True],
        )

        assert result.converged is True

    def test_converged_is_false_when_the_optimiser_fails(self, monkeypatch):
        """`converged=bool(result.success)` → `converged=True`（固定）という
        変異が空振りしていた箇所。実データで収束を失敗させるのは狙って作ると
        壊れやすいので、`scipy.optimize.minimize` を成功しない結果を返す
        スタブに差し替えて、`fit` がその `success` をそのまま反映することを
        直接確かめる。
        """
        rng = np.random.default_rng(21)
        X, groups, y = make_synthetic(
            rng, n_races=10, field_size=4, true_w=np.array([1.0]),
            scales=[1.0], offsets=[0.0],
        )

        class _FailedResult:
            success = False
            x = np.zeros(1)

        monkeypatch.setattr(cl.optimize, "minimize", lambda *args, **kwargs: _FailedResult())

        result = cl.fit(
            X, groups, y, feature_names=("f0",), l2=0.1, standardize_mask=[True],
        )

        assert result.converged is False


class TestFitResultImmutability:
    def test_weights_array_is_not_writeable(self):
        """`FitResult` は frozen dataclass だが、`weights`（`np.ndarray`）は
        中身だけなら再代入せずに書き換えられてしまう。`flags.writeable = False`
        にしてあることを、実際に代入を試みて確認する。
        """
        rng = np.random.default_rng(19)
        X, groups, y = make_synthetic(
            rng, n_races=10, field_size=4, true_w=np.array([1.0]),
            scales=[1.0], offsets=[0.0],
        )

        result = fit(
            X, groups, y, feature_names=("f0",), l2=0.1, standardize_mask=[True],
        )

        assert result.weights.flags.writeable is False
        with pytest.raises(ValueError):
            result.weights[0] = 999.0


class TestFitStandardizationRequiresMask:
    def test_fit_standardization_rejects_missing_mask(self):
        X = np.column_stack([np.array([1.0, 2.0, 3.0, 4.0])])

        with pytest.raises(TypeError):
            fit_standardization(X, ("f0",))  # mask を渡さない

    def test_fit_standardization_rejects_mask_of_none(self):
        """`fit` は `standardize_mask=None` を拒否するが、`fit_standardization`
        を直接 `mask=None` で呼ぶ経路が残っていると、同じ形（全列を黙って
        標準化する）の不具合が1段下で再発する。fix2 決定5。
        """
        X = np.column_stack([np.array([1.0, 2.0, 3.0, 4.0])])

        with pytest.raises(ValueError):
            fit_standardization(X, ("f0",), mask=None)


class TestAcceptsArrayMask:
    def test_accepts_ndarray_standardize_mask(self):
        """`standardize_mask` に `list`/`tuple` だけでなく `np.ndarray` も
        渡せることを確認する。`mask is not None and not mask[j]` を
        `mask and not mask[j]` に弱める変異は、非空の `list`/`tuple` では
        Python の truthiness上区別がつかない（非空なら常に真）が、
        `np.ndarray`（要素2個以上）では `bool(mask)` が
        `ValueError: truth value of an array...` になるため、その変異が
        あればこの呼び出しそのものが（意図しない例外で）失敗する。
        """
        rng = np.random.default_rng(20)
        true_w = np.array([1.5, -0.8])
        X, groups, y = make_synthetic(
            rng, n_races=1000, field_size=8, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0],
        )

        result = fit(
            X, groups, y, feature_names=("f0", "f1"), l2=1e-6,
            standardize_mask=np.array([True, True]),
        )

        assert result.converged
        assert np.all(np.isfinite(result.weights))


class TestFloat64IndistinguishableColumns:
    """fix5 決定9-2: 準定数列は落とさず、名前だけ残して警告させる。"""

    def _matrix_with_an_indistinguishable_column(self):
        # 2列目は平均 1000 に対して std がほぼ 0（float64 では区別できない幅）。
        varying = np.array([1.0, 2.0, 3.0, 4.0])
        indistinguishable = np.array([1000.0, 1000.0, 1000.0, 1000.0 + 1e-9])
        return np.column_stack([varying, indistinguishable])

    def test_indistinguishable_column_is_reported(self):
        X = self._matrix_with_an_indistinguishable_column()

        standardization, dropped, indistinguishable = fit_standardization(
            X, ("varying", "almost_constant"), mask=[True, True])

        assert indistinguishable == ("almost_constant",)
        # 落とさない（分散0ではないので学習対象のまま）。相対閾値に出典が
        # 無いため、判定は変えずに名前だけ残す。
        assert dropped == ()
        assert standardization["almost_constant"][1] > 0.0

    def test_a_normal_column_is_not_reported(self):
        X = self._matrix_with_an_indistinguishable_column()

        _standardization, _dropped, indistinguishable = fit_standardization(
            X, ("varying", "almost_constant"), mask=[True, True])

        assert "varying" not in indistinguishable

    def test_masked_columns_are_never_reported(self):
        """素通し列（mask=False）は実データの分散を見ないので対象外。"""
        X = self._matrix_with_an_indistinguishable_column()

        _standardization, _dropped, indistinguishable = fit_standardization(
            X, ("varying", "almost_constant"), mask=[True, False])

        assert indistinguishable == ()

    def test_fit_carries_the_names(self):
        rng = np.random.default_rng(31)
        true_w = np.array([1.0, 0.0])
        X, groups, y = make_synthetic(
            rng, n_races=50, field_size=4, true_w=true_w,
            scales=[1.0, 1.0], offsets=[0.0, 0.0])
        # 2列目を準定数にする（平均 500、幅 1e-9）。
        X[:, 1] = 500.0
        X[0, 1] = 500.0 + 1e-9

        result = fit(X, groups, y, feature_names=("f0", "f1"), l2=0.1,
                      standardize_mask=[True, True])

        assert result.float64_indistinguishable_features == ("f1",)
        assert "f1" not in result.dropped_features

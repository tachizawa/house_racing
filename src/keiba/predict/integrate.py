"""第3層: 第1層・第2層の答えを1つのランキングに統合する。Jev は呼ばない。

CLAUDE.md が「未確定」として残していた層。元の設計案（重み付き和で終わり）は
壊れている――第1層の確率は出走全頭で合計 1.0 なので、8頭立てなら平均 0.125、
18頭立てなら平均 0.056 になり、頭数が変わるたびに重みの意味が変わってしまう。
ここでは条件付きロジットの形にする:

    score_i = w0 * logit(p_i) + Σ_k w_k * f_ik
    P(i が勝つ) = softmax_by_race(score)

softmax の分母はそのレースの出走馬だけなので、頭数によらず合計は常に 1.0。
重みの意味が頭数で変わることがない。**最終的な統合を単純な重み付き和のまま
終わらせない**――score はあくまで softmax に渡す前の中間値。

このモジュールが持たないもの:
- Jev への通信（jev_client.py / predict/layers.py の仕事）
- state の組み立て（features/*.py の仕事）
- 重みの学習（Task 17。ここは既にある Weights を適用するだけ）
- confidence を特徴として使うこと（layer2 の確信度をその馬の証拠として読むのは
  spec §2.1 が禁じる。AxisAnswer.confidence は記録のためだけに存在する）
- choice（argmax の選択肢）を特徴として使うこと（分布を最尤の1点に潰さない。
  spec §2.2）
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from keiba import config
from keiba.features.context import HorseFeatures, PredictionContext
from keiba.predict.layers import AxisAnswer, Layer1Result, Layer2Result
from keiba.predict.questions import AXIS_SPECS, PROMPT_VERSION, feature_names_for_axes

__all__ = [
    "PYTHON_FEATURE_NAMES",
    "Weights",
    "WeightsIncompatibleError",
    "axis_features",
    "baseline_ranking",
    "build_features",
    "expand_missing",
    "feature_names",
    "is_missing_indicator",
    "logit",
    "model_scores",
    "python_feature_block",
    "python_features",
    "rank_by_score",
    "score_breakdown",
    "softmax_by_race",
    "standardize",
    "standardize_mask_for",
    "tie_groups",
]

#: Python 側（Jev を介さない）8特徴。順序固定――学習済みの重みベクトルは
#: この並びに揃っているので、並びを変えると重みと特徴が静かに入れ替わる。
PYTHON_FEATURE_NAMES: tuple[str, ...] = (
    "recent_margin_wavg",
    "recent_margin_runs_used",
    "recent_class_level_wavg",
    "class_level_change",
    "days_since_last_run",
    "distance_change_m",
    "corner_pos_mean",
    "estimated_transport_km",
)

#: 欠損指示変数の接尾辞。``expand_missing``（作る側）・``_is_passthrough``
#: （検知する側）・``scripts/fit_weights.py``（``--with-python-features`` が
#: Python側の生の値+欠損指示変数16列をまとめて選ぶ側）が同じ文字列リテラルを
#: それぞれ独立に書いていた（fix round 2 Minor 4。Task 17 で10.5倍のずれを
#: 起こした「規則を2箇所に書く」と同型の欠陥）。ここに1箇所だけ持つ。
_MISSING_SUFFIX = "_missing"


def is_missing_indicator(name: str) -> bool:
    """``PYTHON_FEATURE_NAMES`` の欠損指示変数（``expand_missing`` が作る
    ``<name>_missing``）かどうか。``_is_passthrough`` が使う。

    ``expand_missing`` 自体はどんなキー名にも ``_missing`` を付けられる汎用
    関数（``test_expand_missing`` が ``PYTHON_FEATURE_NAMES`` に無いキーでも
    確認している）なので、この述語は使わない――判定はしていないところに
    判定を混ぜない。
    """
    return name in {f"{base}{_MISSING_SUFFIX}" for base in PYTHON_FEATURE_NAMES}


def python_feature_block() -> frozenset[str]:
    """Python側特徴（``PYTHON_FEATURE_NAMES``）の生の値＋欠損指示変数、
    16列をまとめて返す。

    ``scripts/fit_weights.py`` の ``--with-python-features`` /
    ``--without-python-features`` が「このブロックを含めるかどうか」を
    切り替えるときに使う（fix round 2 Minor 4）。
    """
    return frozenset(PYTHON_FEATURE_NAMES) | frozenset(
        f"{name}{_MISSING_SUFFIX}" for name in PYTHON_FEATURE_NAMES)


def logit(p: float, eps: float = 1e-6) -> float:
    """確率を logit に。p=0 / p=1 で -inf / +inf にならないよう clip する。"""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def softmax_by_race(scores: Mapping[str, float]) -> dict[str, float]:
    """レース内 softmax。頭数が何頭でも合計 1.0 を保つ。

    これが spec §13 の「第1層と第2層のスケールの揃え方」への答え。
    8頭でも18頭でも分母がそのレースの出走馬だけなので、
    重み付き和のように頭数で釣り合いが変わることがない。

    exp する前に最大値を引く（オーバーフロー対策）。
    """
    if not scores:
        return {}
    highest = max(scores.values())
    exponentials = {k: math.exp(v - highest) for k, v in scores.items()}
    total = sum(exponentials.values())
    return {k: v / total for k, v in exponentials.items()}


def axis_features(answers: Mapping[str, AxisAnswer]) -> dict[str, float]:
    """第2層の1頭分の答え（軸名→AxisAnswer）を10列の数値特徴にする。

    choice も confidence も使わない――確率分布をそのまま使う
    （choice は spec §2.2、confidence は spec §2.1 がそれぞれ特徴として
    読むことを禁じている）。negative_key が None の軸（transport_concern）は
    差を取らず positive_key の確率そのものを signed にする
    （questions.AxisSpec の docstring と同じ理由）。
    """
    features: dict[str, float] = {}
    for axis_name, spec in AXIS_SPECS.items():
        probabilities = answers[axis_name].probabilities
        positive_p = probabilities[spec.positive_key]
        if spec.negative_key is None:
            signed = positive_p
        else:
            signed = positive_p - probabilities[spec.negative_key]
        features[f"{axis_name}_signed"] = signed
        features[f"{axis_name}_insufficient"] = probabilities[spec.insufficient_key]
    return features


def python_features(hf: HorseFeatures) -> dict[str, float | None]:
    """Python 側（Jev を介さない）8特徴。欠損は None のまま返す
    （0埋めは expand_missing の仕事。ここで丸めない）。

    8列はすべて hf 側の値でまかなえる（RaceInput 側の値は features/derive.py が
    recent.distance_change_m などに既に反映済み）ので、race は引数に取らない。
    使わない引数を持たせると、次に読む人が「意味がある」と思って何かを渡そうと
    してしまう――将来レース条件そのものを使う特徴を足すときに、理由付きで
    一行足せばよい。
    """
    ability = hf.ability
    recent = hf.recent
    return {
        "recent_margin_wavg": ability.recent_margin_sec_wavg,
        # 特徴名 recent_margin_runs_used ⇔ AbilitySummary.runs_used。
        # spec が特徴に付けた名前と dataclass 側のフィールド名が違うだけの対応。
        "recent_margin_runs_used": float(ability.runs_used),
        "recent_class_level_wavg": ability.recent_class_level_wavg,
        "class_level_change": (
            float(recent.class_level_change) if recent.class_level_change is not None else None
        ),
        "days_since_last_run": (
            float(recent.days_since_last_run) if recent.days_since_last_run is not None else None
        ),
        "distance_change_m": (
            float(recent.distance_change_m) if recent.distance_change_m is not None else None
        ),
        "corner_pos_mean": hf.running_style.corner_pos_mean,
        "estimated_transport_km": (
            float(hf.transport.estimated_transport_km)
            if hf.transport.estimated_transport_km is not None else None
        ),
    }


def expand_missing(values: Mapping[str, float | None]) -> dict[str, float]:
    """欠損を 0 埋めしつつ、欠損だったことを指示変数で残す。

    0 埋めだけでは「値が 0」と「欠損」が区別できず、spec §4.4 の
    「欠損値を0や不適性に置き換えない」に反する。値は 0 にするが
    <name>_missing を 1 にして、重みが別に付くようにする。
    """
    result: dict[str, float] = {}
    for name, value in values.items():
        if value is None:
            result[name] = 0.0
            result[f"{name}{_MISSING_SUFFIX}"] = 1.0
        else:
            result[name] = float(value)
            result[f"{name}{_MISSING_SUFFIX}"] = 0.0
    return result


def feature_names(*, with_layer2: bool = True) -> tuple[str, ...]:
    """全特徴の列名を固定順で返す。学習済みの重みベクトルはこの順に並んでいる
    ので、呼ぶたびに同じ結果になることが前提（ソートや集合経由では作らない）。
    """
    names: list[str] = ["logit_layer1"]
    if with_layer2:
        names.extend(feature_names_for_axes())
    names.extend(PYTHON_FEATURE_NAMES)
    names.extend(f"{name}{_MISSING_SUFFIX}" for name in PYTHON_FEATURE_NAMES)
    return tuple(names)


def build_features(
    ctx: PredictionContext,
    layer1: Layer1Result,
    layer2: Layer2Result,
) -> dict[str, dict[str, float]]:
    """馬番ごとに、特徴名→値の辞書を組み立てる（feature_names(with_layer2=True)
    と同じキー集合になる）。
    """
    features: dict[str, dict[str, float]] = {}
    for hf in ctx.horses:
        umaban = hf.umaban
        row: dict[str, float] = {"logit_layer1": logit(layer1.win_probabilities[umaban])}
        row.update(axis_features(layer2.answers[umaban]))
        row.update(expand_missing(python_features(hf)))
        features[umaban] = row
    return features


def rank_by_score(values: Mapping[str, float]) -> list[str]:
    """値の降順、同値は馬番の数値順。順位付けはこの1箇所だけで決める。

    文字列比較だと "10" が "2" より前に来る。利用者入力の馬番はゼロ詰めが
    保証されていない（input/validate.py は正の整数として読めることだけを
    確かめる）。同着順位の目的は再現性なので、利用者がゼロ詰めしたかどうかで
    順位が変わってはいけない。baseline・model のどちらのランキングもここを
    通す――片方だけ直されて崩れ方が食い違うことを防ぐため（Task 16 fix:
    元は predict/pipeline.py に同じ規則がもう1つあった）。
    """
    return sorted(values, key=lambda umaban: (-values[umaban], int(umaban)))


def tie_groups(values: Mapping[str, float], *, band: float) -> dict[str, int]:
    """馬番 → 同程度の組の番号（1始まり）。順位の降順に並べて組にする。

    組の作り方は「組の中の最大差が band 以下」。隣り合う2頭の差だけで見ると、
    1頭ずつ band 以内の鎖がつながって、端どうしの差が band の何倍にもなった
    集まりを「同程度」と呼んでしまう（Task 16 fix2 決定2）。

    順序は rank_by_score と同じ（値の降順、同値は馬番の数値順）――順位付けは
    1箇所だけで決めるという既存の約束を守り、rank_by_score を再実装しない。

    値の降順に並べたとき先頭（＝各組の最初の馬）が組の中で最大なので、
    「組の最大値」は組を開始した馬の値のまま比較すればよく、別途最大値を
    追跡し直す必要はない。band=0.0 は「差が無いものだけ同組」という実在の
    指定であり、falsy として何かの既定値に化けさせてはいけない
    （比較は `>` で行い、`band` の真偽値は一切見ない）。
    """
    if not values:
        raise ValueError("values が空です。同程度の組は1頭以上のときだけ定義できます。")

    order = rank_by_score(values)
    groups: dict[str, int] = {}
    group_no = 1
    group_max = values[order[0]]
    for umaban in order:
        value = values[umaban]
        if group_max - value > band:
            group_no += 1
            group_max = value
        groups[umaban] = group_no
    return groups


def baseline_ranking(layer1: Layer1Result) -> list[str]:
    """第1層の勝率だけで降順に並べたランキング。同確率は馬番の昇順で崩す
    （実行のたびに順序が変わらないように）。
    """
    return rank_by_score(layer1.win_probabilities)


def score_breakdown(
    features: Mapping[str, Mapping[str, float]],
    weights: Weights,
) -> dict[str, dict[str, float]]:
    """馬ごとに、特徴量1列ずつの寄与（weight * value）を返す。

    合計すると model_scores の値と一致する――内訳が常に見えることが、
    予想が外れたときモデルの作り直しではなく重みの調整で対応できるという
    CLAUDE.md の設計意図そのもの。キーは weights.feature_names と揃える
    （features 側に無いキーは weights.feature_names に含めない――Task 17 が
    そう作る前提）。
    """
    return {
        umaban: {
            name: weights.weights[name] * horse_features[name]
            for name in weights.feature_names
        }
        for umaban, horse_features in features.items()
    }


def model_scores(
    features: Mapping[str, Mapping[str, float]],
    weights: Weights,
) -> dict[str, float]:
    """馬ごとの score（softmax に渡す前の中間値）。score_breakdown の内訳の
    合計として求める――「内訳を合計すると score に一致する」ことを二重に
    計算せず、定義そのもので保証する。

    weights に載っていない特徴（weights.feature_names に無いもの）は
    score_breakdown 側で最初から見ないので、ここでも寄与しない。
    """
    return {
        umaban: sum(contributions.values())
        for umaban, contributions in score_breakdown(features, weights).items()
    }


def _is_passthrough(name: str) -> bool:
    """`standardize` が標準化せず値をそのまま通す列名か。

    判定はここ1箇所だけに置く。`standardize` と `standardize_mask_for`
    （Task 17 fix1: `keiba.learn.conditional_logit.fit` の `standardize_mask`
    を作るための関数）の両方がこの述語を呼ぶ――規則を2箇所に書くと片方だけ
    直されて学習時と推論時の標準化が食い違う（実際に踏まれた不具合。
    fix1-brief Critical 2）。``_missing`` 部分の判定は `is_missing_indicator`
    に寄せてある（fix round 2 Minor 4。文字列リテラルの独立した重複を防ぐ）。
    """
    return name == "logit_layer1" or is_missing_indicator(name) or name.endswith("_insufficient")


def standardize_mask_for(names: Sequence[str]) -> tuple[bool, ...]:
    """標準化してよい列を True にした mask。`standardize` と同じ規則で判定する。

    `standardize` が素通しする列（`logit_layer1` と 0/1 の指示列）を学習側で
    標準化してしまうと、学習時に std で割り推論時に割らないので、その列の
    寄与が std 倍ずれる。規則を2箇所に書くと片方だけ直されるので、
    `standardize` が使う判定（`_is_passthrough`）をここから呼ぶ形にして
    出どころを1つにする。`keiba.learn.conditional_logit.fit` の
    `standardize_mask`（キーワード専用の必須引数）にそのまま渡す想定。
    """
    return tuple(not _is_passthrough(name) for name in names)


def standardize(
    features: Mapping[str, Mapping[str, float]],
    standardization: Mapping[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """保存済みの平均・標準偏差を適用する（ここでは計算しない――学習期間
    だけから求めるのは Task 17 の仕事。spec §11.1.1）。

    logit_layer1 と *_missing / *_insufficient は対象外。logit は既に
    意味のある尺度に乗っており、指示変数は 0/1（insufficient は確率 0-1）
    なので、標準化すると値の意味そのものが壊れる。この3種類はマッピングを
    引く前に短絡させる――Task 17 はそもそもこれらのパラメータを学習しないので、
    standardization に無くて当然であり、無いことを確かめる必要すらない。
    標準偏差 0 の特徴は 0 除算を避けてそのまま通す（学習側で落とす前提の値）。

    それ以外の特徴が standardization に無いのは、上の3種類とは違い「無くて
    当然」ではない――重みファイルが学習した特徴集合と、今渡された特徴集合が
    食い違っているという意味。値をそのまま素通しすると、その特徴だけ学習時と
    違う尺度のまま重みに掛かってスコアが狂うのに、テストも例外も出ない。
    ここでは黙って隠さず WeightsIncompatibleError にする
    （check_compatible() が版のずれを検知するのと同じ「重みと今のコードが
    食い違っている」という種類の失敗なので、同じ例外クラスに載せる）。
    """
    result: dict[str, dict[str, float]] = {}
    for umaban, horse_features in features.items():
        row: dict[str, float] = {}
        for name, value in horse_features.items():
            if _is_passthrough(name):
                # 素通し列に恒等でないパラメータが付いていたら、学習側がこの列を
                # 標準化していた合図。黙って素通しすると、その列の寄与だけ学習時と
                # 推論時で std 倍ずれる（fix1 Critical 2 と同型の欠陥が、mask の
                # 渡し間違いではなく重みファイル側から入る経路）。
                # fit_standardization は mask が False の列に (0.0, 1.0) を書くので、
                # 正しく学習された重みは必ずこれを満たす。
                if tuple(standardization.get(name, (0.0, 1.0))) != (0.0, 1.0):
                    raise WeightsIncompatibleError(
                        f"素通しする列 {name} に恒等でない標準化パラメータ"
                        f" {tuple(standardization[name])} が付いています。"
                        " 学習時に標準化された列を推論時に素通しすると寄与が"
                        " std 倍ずれます。学習し直してください。"
                    )
                row[name] = value
                continue
            if name not in standardization:
                raise WeightsIncompatibleError(
                    f"標準化パラメータに {name} がありません。"
                    " 重みファイルと現在の特徴量の集合が食い違っています。"
                    " 学習し直してください。"
                )
            mean, std = standardization[name]
            row[name] = value if std == 0 else (value - mean) / std
        result[umaban] = row
    return result


class WeightsIncompatibleError(RuntimeError):
    """保存済みの重みが、現在のコードの版と違う。

    プロンプト・特徴量・統合ロジックのいずれかが変わったあとに古い重みを
    そのまま使うと、数値は返るが何を測っているのか一致していない
    ――CLAUDE.md 「プロンプトが変わった予想は別の記録として扱う」と同じ理由で、
    ここでは黙って使わせず例外にする。
    """


@dataclass(frozen=True)
class Weights:
    """学習済みの重みベクトルと、それが何の上で学習されたかの記録一式。

    JSON との往復（save/load）だけをここで持つ。フィット（Task 17）はしない。
    """

    weights_id: str
    feature_names: tuple[str, ...]
    weights: dict[str, float]
    standardization: dict[str, tuple[float, float]]
    calibration: dict | None
    versions: dict[str, str]
    lambda_: float
    dropped_features: tuple[str, ...]
    splits: dict[str, str]
    #: 学習を実行した時刻（UTC の ISO 8601）。spec §11.1.2 が要求する欄。
    #: これが無いと「いつ作った重みか」「2つのうちどちらが新しいか」が
    #: ファイルから分からない（weights_id が uuid4 だったときはそうだった）。
    #: 既定値は持たない——来歴の欄が既定値を持つと、渡し忘れた組み立てが
    #: 黙って空の来歴を主張する。
    fitted_at: str
    #: λ を選んだ根拠。少なくとも採用した λ の dev 対数尤度と、両特徴量構成の
    #: 比較値を入れる（spec §11.1.2）。標準出力にしか出さないと、あとから
    #: 「なぜこの λ なのか」を重みファイルから確かめられない。
    dev_metrics: dict

    def save(self, path: str | Path) -> None:
        # "lambda" は Python の予約語なので、フィールド名は lambda_ にして
        # JSON 上のキーだけ "lambda" に戻す。
        data = {
            "weights_id": self.weights_id,
            "fitted_at": self.fitted_at,
            "feature_names": list(self.feature_names),
            "weights": self.weights,
            "standardization": {k: list(v) for k, v in self.standardization.items()},
            "calibration": self.calibration,
            "versions": self.versions,
            "lambda": self.lambda_,
            "dropped_features": list(self.dropped_features),
            "splits": self.splits,
            "dev_metrics": self.dev_metrics,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Weights:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights_id=data["weights_id"],
            feature_names=tuple(data["feature_names"]),
            weights=dict(data["weights"]),
            standardization={k: tuple(v) for k, v in data["standardization"].items()},
            calibration=data["calibration"],
            versions=dict(data["versions"]),
            lambda_=data["lambda"],
            dropped_features=tuple(data["dropped_features"]),
            splits=dict(data["splits"]),
            # .get で既定を埋めない。この2欄が無い重みファイルは spec §11.1.2 を
            # 満たしていない（いつ学習したのか・λ を選んだ根拠が失われている）
            # ので、黙って空を主張させず KeyError で落とす。
            fitted_at=data["fitted_at"],
            dev_metrics=dict(data["dev_metrics"]),
        )

    def check_compatible(self) -> None:
        """versions の feature_version / prompt_version / integration_version が
        現在のコードと違えば WeightsIncompatibleError。

        違う版で学習した重みをそのまま使うと、数値は返るが何を測っているか
        一致していない（プロンプト文言が変われば同じ「distance_fit_signed」
        でも意味が変わりうるし、特徴の並びが変わればどの重みがどの列に
        掛かるかがずれる）。
        """
        current = {
            "feature_version": config.FEATURE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "integration_version": config.INTEGRATION_VERSION,
        }
        mismatches = [
            f"{key}: 保存値={self.versions.get(key)!r} 現在値={current_value!r}"
            for key, current_value in current.items()
            if self.versions.get(key) != current_value
        ]
        if mismatches:
            raise WeightsIncompatibleError(
                "重みの版が現在のコードと一致しません: " + "; ".join(mismatches)
            )

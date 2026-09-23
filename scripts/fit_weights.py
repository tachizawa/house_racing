"""保存済みの予測記録から重みを学習し、較正して data/weights/<weights_id>.json を書く CLI。

    .venv/Scripts/python.exe scripts/fit_weights.py \\
        --from 20250101 --to 20260901 \\
        --train-until 20251201 --dev-until 20260401 --calib-until 20260701 \\
        --predictions-from data --out data/weights

**Jev は一切呼ばない。** 学習の入力は ``--predictions-from`` が指すディレクトリに
既に保存されている予測記録（``keiba.store.records.save`` が書いたもの）だけで、
ここから ``HorsePrediction.features``（Task 18 supplement で追加された生の特徴量）
と実際の勝ち馬を突き合わせて train/dev/calib の3つの用途に使う。Jev を呼び直すと
同じレースに二重課金するうえ、記録した回答と学習に使った回答がずれる
(``task-18-brief.md``)。

train / dev / calib / test の境界（``--train-until`` / ``--dev-until`` /
``--calib-until``）はレースの時系列（``PredictionRecord.prediction_cutoff`` の
日付）で切る――``predicted_at``（このスクリプトやバックテストを実行した時刻）
ではない。バックテストは通常「今日」まとめて過去レースぶんの記録を保存するので、
``predicted_at`` の月で束ねると学習期間の意味を失う。

記録 → 学習入力の組み立ては次の順で行う（fix round 2 Critical 3 / Minor 2。
順序を誤ると同じレースが train と dev の両方に入る漏洩や、使える古い記録が
使えない新しい記録に潰される欠陥になる）:

    1. 全記録を読む (``_load_all_records``)
    2. ``--from``/``--to`` の範囲で絞る。落ちた件数を ``out_of_range`` として数える
    3. 使える記録だけに絞る (``_filter_usable``。status・勝ち馬・特徴量集合の
       一致・有限性)
    4. **使える記録のうち** ``race_id`` ごとに1件に畳む (``_collapse_duplicate_races``。
       ``predicted_at`` が最新)
    5. split に振り分ける (``_split_races``)
    6. 件数を表示する

標準化 mask は ``keiba.predict.integrate.standardize_mask_for`` から作る
（自分で規則を書かない――学習時と推論時でずれると Task 17 で実測 10.5倍の
ずれが出た欠陥の再発になる）。確率の計算（標準化→モデルスコア→softmax）は
``keiba.predict.integrate`` の関数をそのまま呼ぶ――``keiba.predict.pipeline``
が推論時に通る経路と同じ関数を通すことで、学習側だけ別の式を使ってしまう
ことを防ぐ。

test split はここでは一切読まない（将来の検証タスクのために予約するだけ）。

出走取消・競走除外馬は学習の ``X`` に1行として残り、Luce の分母に入る。実際の
予想時点では誰が取消になるか分からないので全登録馬が入るのが正しく、学習は
記録された全馬をそのまま使う。バックテスト側も取消馬を入力に残すようになった
（fix5 決定5）ので、**学習・推論・バックテストの3者で出走集合の定義が一致して
いる**。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config, console  # noqa: E402
from keiba.data.repository import ResultRepository  # noqa: E402
from keiba.data.store import DataStore, DataVersionError  # noqa: E402
from keiba.learn import calibration as cal  # noqa: E402
from keiba.learn import conditional_logit as cl  # noqa: E402
from keiba.predict import integrate as ig  # noqa: E402
from keiba.store import records as records_store  # noqa: E402
from keiba.store.records import PredictionRecord  # noqa: E402

#: log(0) を避ける clip。keiba.evaluate.metrics（Task 19 brief）の log_loss が
#: 同じ目的で使う既定値 (eps=1e-12) と揃える――ここだけ別の値を決め打ちしない。
_LOG_EPS = 1e-12

#: 「使える記録に絞る」(``_filter_usable``) で数える除外理由と、そのあと
#: ``_collapse_duplicate_races`` が数える重複の理由。呼び出し側 (``main``) が
#: 1つの dict にまとめて全件を表示する（黙って落とすと学習レース数のずれに
#: 気づけない）。
_EXCLUSION_REASONS: tuple[str, ...] = (
    "status_not_succeeded", "no_winner", "winner_not_in_horses", "feature_mismatch",
    "non_finite_feature", "duplicate_race",
)


@dataclass(frozen=True)
class _UsableRace:
    """学習・評価に使える1レースぶん。勝ち馬 (umaban) を record と一緒に運ぶ
    ――呼び出しのたびに ResultRepository.winners を引き直さない。
    """

    record: PredictionRecord
    winners: tuple[str, ...]


# --- 引数 ------------------------------------------------------------------


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _parse_lambda_grid(value: str) -> tuple[float, ...]:
    """カンマ区切りの lambda グリッドを解析する。

    空（``""`` や空白だけの要素しかない）なら分かるメッセージで
    ``ValueError``。負の値が1つでもあれば ``ValueError``（``fit`` 自身も
    ``l2 < 0`` を拒否するが、グリッド探索の1候補目まで進んでから落ちるより
    早く分かるメッセージを出す。fix round 2 Minor 5）。
    """
    items = [x for x in value.split(",") if x.strip() != ""]
    if not items:
        raise ValueError(f"--lambda-grid が空です: {value!r}")
    grid = tuple(float(x) for x in items)
    negatives = [x for x in grid if x < 0]
    if negatives:
        raise ValueError(f"--lambda-grid に負の値があります: {negatives}")
    return grid


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="保存済みの予測記録から重みを学習・較正し、"
                     "data/weights/<weights_id>.json を書く。Jev は呼ばない。"
                     "日付（--from / --to / --*-until）はすべて予想締切"
                     "（発走の前日）で判定します。")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                         help="データ版の場所。勝ち馬の参照 (ResultRepository) に使う"
                              " (既定 %(default)s)")
    parser.add_argument("--predictions-from", type=Path, required=True,
                         help="保存済みの予測記録のディレクトリ"
                              " (keiba.store.records.save の data_dir)")
    parser.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD",
                         help="対象期間の開始。日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD",
                         help="対象期間の終了。日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--train-until", required=True, metavar="YYYYMMDD",
                         help="train / dev の境界 (必須)。"
                              "日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--dev-until", required=True, metavar="YYYYMMDD",
                         help="dev / calib の境界 (必須)。"
                              "日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--calib-until", required=True, metavar="YYYYMMDD",
                         help="calib / test の境界 (必須)。test はこのスクリプトでは"
                              "使わない。日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None,
                         help="L2係数を固定する場合の値。省略時は --lambda-grid の中から"
                              " dev の対数尤度が最大のものを選ぶ")
    parser.add_argument("--lambda-grid", default="0.001,0.01,0.1,1,10",
                         help="--lambda 省略時に探索するグリッド (カンマ区切り、既定 %(default)s)")
    # BooleanOptionalAction は --with-python-features から --no-with-python-features
    # という名前しか生えない。ブリーフが指定した --without-python-features は
    # 別物として解釈されず exit 2 になり、それなのに比較表の見出しは
    # --without-python-features と印字していた（実在しないフラグを運用者に
    # 打たせる欠陥。fix round 2 Important 2）。dest を共有する2つの独立した
    # フラグとして定義し、実際に印字する名前と一致させる。
    parser.add_argument("--with-python-features", dest="with_python_features",
                         action="store_true", default=True,
                         help="Python側8特徴 (+その欠損指示変数8列) を、実際に書き出す"
                              " 重みに含める (既定)")
    parser.add_argument("--without-python-features", dest="with_python_features",
                         action="store_false",
                         help="Python側8特徴 (+その欠損指示変数8列) を除いた構成で"
                              " 比較・採用する。両方の dev 指標は常に並べて表示するので、"
                              " 判断はその比較を見て行う")
    parser.add_argument("--out", type=Path, default=config.DATA_DIR / "weights",
                         help="重みの出力先ディレクトリ (既定 %(default)s)")
    return parser.parse_args(argv)


# --- 記録の読み込みと期間での絞り込み ------------------------------------------


def _load_all_records(predictions_dir: Path) -> list[PredictionRecord]:
    """``<predictions_dir>/predictions/*.jsonl`` を月をまたいで全部読む。

    ``records.load_month`` は1か月分しか読まないが、バックテストで作った記録は
    ``predicted_at``（実行した「今日」）で1つの月ファイルにまとまる一方、対象と
    するレースは何年も前まで散らばりうる。どの月ファイルに何年分のレースが
    入っているか事前には分からないので、存在する月ファイルを全部読んでから
    ``prediction_cutoff`` の日付で選別する。
    """
    month_dir = predictions_dir / "predictions"
    if not month_dir.exists():
        return []
    year_months = sorted(p.stem for p in month_dir.glob("*.jsonl"))
    all_records: list[PredictionRecord] = []
    for year_month in year_months:
        all_records.extend(records_store.load_month(year_month, data_dir=predictions_dir))
    return all_records


def _filter_in_range(
    records_list: list[PredictionRecord], *, date_from: date, date_to: date,
) -> tuple[list[PredictionRecord], int]:
    """``prediction_cutoff`` の日付が ``[date_from, date_to]`` の記録だけを残し、
    範囲外だった件数を返す。

    range外に落ちた件数をどこにも出さないと、ディスク上の記録数と表示された
    件数の合計が食い違い、消えた記録に気づけない (fix round 2 Important 1)。
    """
    in_range: list[PredictionRecord] = []
    out_of_range = 0
    for record in records_list:
        d = record.prediction_cutoff.date()
        if date_from <= d <= date_to:
            in_range.append(record)
        else:
            out_of_range += 1
    return in_range, out_of_range


# --- 記録 → 使えるレース --------------------------------------------------------


def _filter_usable(
    records_subset: list[PredictionRecord], required_keys: set[str], result_repo: ResultRepository,
) -> tuple[list[_UsableRace], dict[str, int], dict[int, int]]:
    """使える記録だけを ``_UsableRace`` にして残し、除外理由ごとの件数を返す。

    ここではまだ重複（同じレースの複数記録）を畳まない――畳むのはこのあと
    ``_collapse_duplicate_races``。1本混じると ``conditional_logit.fit`` の
    Σy==1 検証が全体を ValueError で止めるので、ここで確実に弾く。

    3つめの返り値は ``feature_mismatch`` で落ちた記録の「1頭あたりの特徴量の
    列数 → 件数」。``main()`` がこれを使って「17列の記録に27列を要求した」と
    原因まで表示する――件数だけでは、運用者が何を直せばよいか分からない
    （fix5 決定9-6）。
    """
    counts: dict[str, int] = dict.fromkeys(_EXCLUSION_REASONS, 0)
    mismatch_sizes: dict[int, int] = {}
    usable: list[_UsableRace] = []
    for record in records_subset:
        # PredictionRecord.status は pipeline.predict() が返す実際の値
        # ("SUCCEEDED"/"PARTIAL") を基準にする。supplement の文言は "OK" と
        # 書いているが、コードが実際に出す値は "SUCCEEDED" ―― 層2が一部でも
        # 失敗した記録 ("PARTIAL") を含め、それ以外は全部除外する。
        if record.status != "SUCCEEDED":
            counts["status_not_succeeded"] += 1
            continue
        # race_id が無い記録は勝ち馬を引けない (ResultRepository.winners は
        # 該当レースが無ければ空リストを返すので、None もここで自然に
        # "no_winner" に落ちる)。
        winners = result_repo.winners(record.race_id) if record.race_id is not None else []
        if not winners:
            counts["no_winner"] += 1
            continue
        horse_umaban = {h.umaban for h in record.horses}
        if not all(w in horse_umaban for w in winners):
            counts["winner_not_in_horses"] += 1
            continue
        if any(set(h.features.keys()) != required_keys for h in record.horses):
            counts["feature_mismatch"] += 1
            for horse in record.horses:
                if set(horse.features.keys()) != required_keys:
                    size = len(horse.features)
                    mismatch_sizes[size] = mismatch_sizes.get(size, 0) + 1
            continue
        # NaN/inf が1つ混じると conditional_logit.fit の検証が未捕捉の
        # ValueError で落ち、しかも並べ替え後の行番号しか分からずどのレース・
        # どの馬かが追えない。実パイプラインからは出ない値だが、レース単位で
        # ここで弾いて件数を出す (fix round 2 Minor 1)。
        if any(
            not math.isfinite(value)
            for horse in record.horses for value in horse.features.values()
        ):
            counts["non_finite_feature"] += 1
            continue
        usable.append(_UsableRace(record=record, winners=tuple(winners)))
    return usable, counts, mismatch_sizes


def _collapse_duplicate_races(races: list[_UsableRace]) -> tuple[list[_UsableRace], int]:
    """使える記録のうち、``race_id`` ごとに ``predicted_at`` が最も新しい
    ものだけを残す。

    同じレースを複数回予想し直すのは異常ではない（重みを変えて再予想、
    プロンプトを直して取り直す、など）。しかし畳まずに学習へ渡すと、同じ
    レースが2つの独立した観測として尤度に入り、標準誤差を過小評価して
    そのレースの主張に重みが偏る（fix round 1）。

    **呼び出し順が重要**: ここは ``_filter_usable`` の後に呼ぶ前提
    （fix round 2 Critical 3 / Minor 2）。規則は「最新を残す」ではなく
    「使える記録のうち最新を残す」――先に畳んでから使える記録を選ぶと、
    「再予想したら層2が一部失敗した」ようなケースで、使えた古い記録が
    使えない新しい記録に理由も分からず潰される。
    """
    kept: dict[str, _UsableRace] = {}
    duplicate_count = 0
    for race in races:
        race_id = race.record.race_id
        existing = kept.get(race_id)
        if existing is None:
            kept[race_id] = race
        elif race.record.predicted_at > existing.record.predicted_at:
            kept[race_id] = race
            duplicate_count += 1
        else:
            duplicate_count += 1
    return list(kept.values()), duplicate_count


def _assign_split(
    record: PredictionRecord, *, train_until: date, dev_until: date, calib_until: date,
) -> str:
    d = record.prediction_cutoff.date()
    if d <= train_until:
        return "train"
    if d <= dev_until:
        return "dev"
    if d <= calib_until:
        return "calib"
    return "test"


def _split_races(
    races: list[_UsableRace], *, train_until: date, dev_until: date, calib_until: date,
) -> dict[str, list[_UsableRace]]:
    """使える・畳み込み済みのレースを ``prediction_cutoff`` の日付で
    train/dev/calib/test に振り分ける。

    ``_collapse_duplicate_races`` の後に呼ぶ前提――この時点で同じ ``race_id``
    は1件しか残っていないので、同じレースが2つの split にまたがることはない
    （fix round 2 Critical 3: 畳み込みを split の中で個別に行っていた旧実装は、
    同じレースを別の cutoff で予想し直すと境界をまたいで train/dev 両方に
    入り、しかも duplicate_race が両方 0 と出るので運用者から見えなかった）。
    """
    buckets: dict[str, list[_UsableRace]] = {"train": [], "dev": [], "calib": [], "test": []}
    for race in races:
        split_name = _assign_split(
            race.record, train_until=train_until, dev_until=dev_until, calib_until=calib_until)
        buckets[split_name].append(race)
    return buckets


def build_training_matrix(
    races: list[_UsableRace], feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``conditional_logit.fit`` に渡す (X, groups, y) を組み立てる。

    ``groups`` は ``record.race_id`` を使う。呼び出し側が既に
    ``_collapse_duplicate_races`` で同じレースの重複記録を1件に畳んでいる
    前提――``record_id``（記録ごとに一意）でグルーピングすると、重複が
    残っていた場合に同じレースが2つの独立した観測として尤度に入り、標準誤差を
    過小評価してそのレースに重みが偏る（fix round 1。record_id グルーピングは
    「``fit`` が Σy=2 で落ちる」という正しい警告への誤った対処だった）。
    """
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    group_rows: list[str] = []
    for race in races:
        weight = 1.0 / len(race.winners)
        for horse in race.record.horses:
            X_rows.append([horse.features[name] for name in feature_names])
            y_rows.append(weight if horse.umaban in race.winners else 0.0)
            group_rows.append(race.record.race_id)
    X = np.array(X_rows, dtype=float) if X_rows else np.empty((0, len(feature_names)))
    y = np.array(y_rows, dtype=float)
    groups = np.array(group_rows)
    return X, groups, y


def combine_data_versions(races: list[_UsableRace]) -> str:
    """学習に使った記録の ``data_version`` を集め、重複を除き昇順・カンマ連結で返す。

    ``DataStore.open_current`` で現在開いている版ではなく、実際に学習に使った
    記録の版を記録する――現在の版を書くと「この重みはこの版のデータで学習
    した」という事実でない主張が記録に残る（fix round 1）。
    ``Weights.check_compatible`` は ``feature_version``/``prompt_version``/
    ``integration_version`` の3つだけを照合に使い ``data_version`` は見ないので、
    複数の版が混在していても互換性チェックは壊れない（純粋な来歴の記録）。
    """
    versions_used = sorted({race.record.versions["data_version"] for race in races})
    return ",".join(versions_used)


def require_consistent_versions(races: list[_UsableRace]) -> dict[str, str]:
    """学習に使った記録のうち、``model`` と ``Weights.check_compatible`` が
    照合する3キー (``feature_version``/``prompt_version``/``integration_version``)
    が単一であることを確認し、その値を返す。

    ``records_store.current_versions(...)`` をそのまま使うと、これらの値が
    「実行時のコードの値」になり、``check_compatible`` は常に無条件で通って
    しまう（fix round 2 Critical 1: レビュアーが記録側を古い版
    ``prompt_version="OLD-PROMPT-v0"`` に書き換えて学習し、exit 0・警告なし・
    ``check_compatible()`` が例外を投げないことを再現した）。
    ``check_compatible`` が見るのはこの3つだけなので、ガードが存在する理由
    そのものが無効化される。ここでは学習に使った記録側の実際の値を書く。
    単一でなければ ``ValueError``（``model`` と同じ規律。答えの前提が違う
    レースを1つの重みにまとめることになるため）。fix round1 supplement の
    「``versions`` は ``records.current_versions`` と揃える」は、この3キーに
    関しては自己矛盾だった――現在のコードの値を書くと、まさに「学習した版と
    今使う版が一致しているかの確認」が原理的に不可能になる。
    """
    mismatches: list[str] = []
    result: dict[str, str] = {}
    for key in ("model", "feature_version", "prompt_version", "integration_version"):
        values_used = sorted({race.record.versions[key] for race in races})
        if len(values_used) != 1:
            mismatches.append(f"{key}={values_used}")
        else:
            result[key] = values_used[0]
    if mismatches:
        raise ValueError(
            "学習に使った記録の版が揃っていません: " + " / ".join(mismatches))
    return result


# --- 予測確率（推論と同じ経路）------------------------------------------------


def _temporary_weights(fit_result: cl.FitResult, feature_names: tuple[str, ...]) -> ig.Weights:
    """``integrate.standardize`` / ``model_scores`` を呼ぶためだけの器。

    保存はしない（``weights_id`` はダミー）。確率の計算そのものは
    ``keiba.predict.integrate`` の関数を素通りさせるだけで、ここで
    標準化やスコアの式を書き直さない――推論 (``predict/pipeline.py``) と
    同じ式を通すことが目的。
    """
    return ig.Weights(
        weights_id="__fit_weights_tmp__",
        feature_names=feature_names,
        weights=dict(zip(feature_names, fit_result.weights.tolist())),
        standardization=dict(fit_result.standardization),
        calibration=None,
        versions={},
        lambda_=fit_result.lambda_,
        dropped_features=fit_result.dropped_features,
        splits={},
        # 保存しない器なので来歴は空にする（保存する重みは main() が本物の
        # 値を入れる）。ダミーだと分かる値を明示的に置き、欄そのものを
        # 省略できないようにしてある。
        fitted_at="",
        dev_metrics={},
    )


def race_probabilities(
    race: _UsableRace, feature_names: tuple[str, ...], weights: ig.Weights,
) -> dict[str, float]:
    """1レースぶんの予測勝率 (softmax) を、推論と同じ関数で計算する。"""
    raw = {
        horse.umaban: {name: horse.features[name] for name in feature_names}
        for horse in race.record.horses
    }
    standardized = ig.standardize(raw, weights.standardization)
    scores = ig.model_scores(standardized, weights)
    return ig.softmax_by_race(scores)


def mean_log_likelihood(
    races: list[_UsableRace], feature_names: tuple[str, ...], weights: ig.Weights,
) -> float:
    """レースあたり平均の対数尤度 (Σ y_i log p_i)。

    ``conditional_logit.fit`` が学習で最大化する量 (l2=0 のときの
    ``FitResult.log_likelihood``) と同じ式を、held-out の split に対して
    ``integrate`` の実際の推論経路 (standardize → model_scores → softmax)
    で評価する。dev split での lambda 選択も、この値が最大のものを選ぶ。

    ``probabilities[umaban]`` は直接インデックスする（``.get(umaban, 0.0)``
    にしない）。勝ち馬は ``_filter_usable`` の "winner_not_in_horses" 判定で
    既に record.horses の中にいることを確認済みなので、
    ``race_probabilities`` が返す辞書に必ずキーがある――``.get`` の既定値は
    理論上到達不能だが、到達すれば ``log(1e-12)=-27.6`` が静かに混ざって
    バグを隠す（fix round 2 Minor 3）。
    """
    total = 0.0
    for race in races:
        probabilities = race_probabilities(race, feature_names, weights)
        weight = 1.0 / len(race.winners)
        total += sum(
            weight * math.log(max(probabilities[umaban], _LOG_EPS))
            for umaban in race.winners
        )
    return total / len(races)


def collect_calibration_pairs(
    races: list[_UsableRace], feature_names: tuple[str, ...], weights: ig.Weights,
) -> tuple[list[float], list[float]]:
    """calib split から (予測確率, 実際の結果) の対を集める。

    outcome は学習の y と同じ規則 (単独勝ちなら 1.0、同着なら 1/同着頭数、
    それ以外は 0.0)。isotonic regression は連続値の y も扱えるので、
    同着を 0/1 に丸めない。
    """
    probabilities: list[float] = []
    outcomes: list[float] = []
    for race in races:
        race_probs = race_probabilities(race, feature_names, weights)
        weight = 1.0 / len(race.winners)
        for umaban, p in race_probs.items():
            probabilities.append(p)
            outcomes.append(weight if umaban in race.winners else 0.0)
    return probabilities, outcomes


# --- lambda 選択 --------------------------------------------------------------


def select_lambda(
    train_races: list[_UsableRace], dev_races: list[_UsableRace],
    feature_names: tuple[str, ...], *, fixed_lambda: float | None, grid: tuple[float, ...],
) -> dict:
    """train で学習し、dev の対数尤度が最大の lambda を選ぶ。

    ``fixed_lambda`` が指定されていればそれだけで学習し、dev 指標は参考値として
    計算する（グリッド探索はしない）。``grid`` は ``_parse_lambda_grid`` が
    渡す順序のままここで走査する。dev 指標が同点になったときにどちらが
    残るかは、この走査順と ``>`` 比較（同点では更新しない）で決まる――
    利用者が順序を保証されていると期待すべき契約ではない（fix round 2
    Minor 5: 旧 docstring は「grid の並び順 (昇順) のまま」という、利用者側が
    頼ってよいかのような前提を書いていたが、それは実装の副作用であって契約
    ではない）。
    """
    mask = ig.standardize_mask_for(feature_names)
    X_train, groups_train, y_train = build_training_matrix(train_races, feature_names)
    candidates = (fixed_lambda,) if fixed_lambda is not None else grid

    best: dict | None = None
    for l2 in candidates:
        fit_result = cl.fit(X_train, groups_train, y_train,
                             feature_names=feature_names, l2=l2, standardize_mask=mask)
        weights = _temporary_weights(fit_result, feature_names)
        dev_metric = mean_log_likelihood(dev_races, feature_names, weights)
        if best is None or dev_metric > best["dev_metric"]:
            best = {"lambda": l2, "fit_result": fit_result, "dev_metric": dev_metric}
    assert best is not None  # candidates は常に1件以上
    return best


def _is_loadable_by_predict(feature_names: tuple[str, ...]) -> bool:
    """``pipeline.predict`` のステップ3（特徴量集合の双方向完全一致）を
    通れる特徴量集合か。

    双方向の完全一致は緩めない――層2の列を黙って無視して、馬ごとに課金
    した答えを捨てることを防ぐガード（Task 16 fix round 2）。判定はここで
    新しく書かず、``integrate.feature_names`` が実際に作る2つの集合
    (``with_layer2=True``/``False``) とだけ比較する（fix round 2 Critical 2:
    削減構成 (11列: logit_layer1+軸10) はどちらとも一致せず、``predict()`` は
    ``with_layer2`` の真偽どちらでも "PredictionFailed" になる。実際に
    ``predict()`` を呼んで確かめずに済むよう、判定基準を
    ``integrate.feature_names`` 側とだけ揃える）。
    """
    names = frozenset(feature_names)
    return (names == frozenset(ig.feature_names(with_layer2=True))
            or names == frozenset(ig.feature_names(with_layer2=False)))


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    args = _parse_args(argv)

    date_from = _parse_date(args.date_from)
    date_to = _parse_date(args.date_to)
    train_until = _parse_date(args.train_until)
    dev_until = _parse_date(args.dev_until)
    calib_until = _parse_date(args.calib_until)

    if not (date_from <= train_until <= dev_until <= calib_until <= date_to):
        print("期間の境界が順序どおりではありません: "
              f"--from={args.date_from} <= --train-until={args.train_until} <= "
              f"--dev-until={args.dev_until} <= --calib-until={args.calib_until} <= "
              f"--to={args.date_to} である必要があります。")
        return 1

    try:
        lambda_grid = _parse_lambda_grid(args.lambda_grid)
    except ValueError as error:
        print(f"\n{error}")
        return 1
    if args.lambda_ is not None and args.lambda_ < 0:
        print(f"\n--lambda は0以上である必要があります: {args.lambda_}")
        return 1

    try:
        store = DataStore.open_current(args.data_dir)
    except DataVersionError as error:
        print(f"データ版を開けません: {error}")
        return 1
    result_repo = ResultRepository(store)

    all_records = _load_all_records(args.predictions_from)
    in_range_records, out_of_range = _filter_in_range(
        all_records, date_from=date_from, date_to=date_to)

    feature_full = ig.feature_names(with_layer2=True)
    required_keys = set(feature_full)

    usable_races, counts, mismatch_sizes = _filter_usable(
        in_range_records, required_keys, result_repo)
    deduped_races, duplicate_count = _collapse_duplicate_races(usable_races)
    counts["duplicate_race"] = duplicate_count
    buckets = _split_races(
        deduped_races, train_until=train_until, dev_until=dev_until, calib_until=calib_until)

    print(f"--- レース選定 ({args.date_from} 〜 {args.date_to}) ---")
    print(f"読み込み総数(ディスク): {len(all_records)} 件")
    print(f"    除外[out_of_range]: {out_of_range} 件")
    for reason in _EXCLUSION_REASONS:
        print(f"    除外[{reason}]: {counts[reason]} 件")
        # 件数だけでは何を直せばよいか分からないので、原因を1行足す
        # （fix5 決定9-6）。
        if reason == "feature_mismatch" and counts[reason]:
            observed = ", ".join(f"{size}列 {n}頭" for size, n in sorted(mismatch_sizes.items()))
            print(f"      原因: この実行が要求するのは {len(required_keys)}列"
                  f"（with_layer2=True）ですが、記録側は {observed} でした。"
                  " 記録を取り直すか、同じ構成の記録だけを --predictions-from に"
                  " 置いてください。")
    print(f"train: {len(buckets['train'])} 件")
    print(f"dev: {len(buckets['dev'])} 件")
    print(f"calib: {len(buckets['calib'])} 件")
    print(f"test: {len(buckets['test'])} 件（このスクリプトでは使用しません）")

    if not buckets["train"]:
        print("\ntrain レースが0件です。学習できません。")
        return 1
    if not buckets["dev"]:
        print("\ndev レースが0件です。lambda を選べません。")
        return 1
    if not buckets["calib"]:
        print("\ncalib レースが0件です。較正を作成できません。")
        return 1

    # 学習に使った記録の版が揃っているか確認する。2構成 × グリッド分の fit を
    # 回してから落とすと無駄なので、fit の前に検査する（fix round 2 Minor 6）。
    used_races = buckets["train"] + buckets["dev"] + buckets["calib"]
    try:
        consistent_versions = require_consistent_versions(used_races)
    except ValueError as error:
        print(f"\n{error}")
        return 1

    python_block = ig.python_feature_block()
    feature_configs: dict[str, tuple[str, ...]] = {
        "with_python_features": tuple(feature_full),
        "without_python_features": tuple(n for n in feature_full if n not in python_block),
    }

    results: dict[str, dict] = {}
    for config_name, names in feature_configs.items():
        results[config_name] = select_lambda(
            buckets["train"], buckets["dev"], names,
            fixed_lambda=args.lambda_, grid=lambda_grid)

    print("\n--- dev 指標の比較 (--with-python-features / --without-python-features) ---")
    for config_name, names in feature_configs.items():
        r = results[config_name]
        print(f"{config_name} ({len(names)}列): lambda={r['lambda']}"
              f" dev対数尤度(平均/レース)={r['dev_metric']:.4f}"
              f" converged={r['fit_result'].converged}")

    # 削減構成 (without_python_features) は predict() のステップ3 (双方向の
    # 完全一致) を構造的に通らない (11列は with_layer2=True/False どちらの
    # 27列/17列とも一致しない)。それでも dev で上回ることはありうるので、
    # 採用構成がどちらであっても目立つ形で表示する (fix round 2 Critical 2)。
    if (not _is_loadable_by_predict(feature_configs["without_python_features"])
            and results["without_python_features"]["dev_metric"]
            > results["with_python_features"]["dev_metric"]):
        print(
            "\n注意: 削減構成 (without_python_features) の dev 対数尤度"
            f" ({results['without_python_features']['dev_metric']:.4f}) が"
            " 採用可能な構成 (with_python_features,"
            f" {results['with_python_features']['dev_metric']:.4f}) を上回りました。"
            " ただしこの構成は predict() が読める特徴量集合と一致しないため"
            "（with_layer2=True/False どちらの列集合とも不一致）、"
            " この構成を使うには推論側 (keiba.predict.pipeline) に"
            " 列部分集合モードを足す判断が必要です。"
        )

    chosen_name = "with_python_features" if args.with_python_features else "without_python_features"
    chosen = results[chosen_name]
    chosen_feature_names = feature_configs[chosen_name]

    if not _is_loadable_by_predict(chosen_feature_names):
        print(
            f"\n採用しようとした構成 ({chosen_name}, {len(chosen_feature_names)}列) は"
            " predict() が読める特徴量集合と一致しません"
            "（with_layer2=True/False どちらの列集合とも不一致）。"
            " この構成を使うには推論側 (keiba.predict.pipeline) に"
            " 列部分集合モードを足す判断が必要です。重みファイルは書き出しません。"
        )
        return 0

    # std が float64 で区別できない列の警告（fix5 決定9-2）。落とす対象は変えない
    # ――「この std 以下なら実質定数」と言える相対閾値に出典が無いため、警告だけ
    # 出して判断は運用者に委ねる。標準化すると (value - mean)/std が極端に増幅され、
    # その列の重みが実質的に解釈不能になる。
    indistinguishable = chosen["fit_result"].float64_indistinguishable_features
    if indistinguishable:
        print(f"\n警告: 列のスケールに対して標準偏差が数値的に区別できないほど小さい"
              f"列があります（{len(indistinguishable)}列）: {list(indistinguishable)}。"
              " 標準化で値が極端に増幅されるため、これらの重みは解釈しないこと。"
              " 落とすかどうかの閾値には出典が無いので、ここでは落としていません。"
              " なお**これは丸め誤差レベルの検出であり、統計的に情報の無い列"
              "（相対的な標準偏差が 1e-4 級の列）は捕まえません**（fix6 ⑤-3）。")

    chosen_weights_for_calibration = _temporary_weights(chosen["fit_result"], chosen_feature_names)
    probabilities, outcomes = collect_calibration_pairs(
        buckets["calib"], chosen_feature_names, chosen_weights_for_calibration)
    calibration_map = cal.fit_calibration(probabilities, outcomes)

    # weights_id は学習実行時刻の YYYYMMDDTHHMMSSZ（仕様 §11.1.2）。uuid4 だと
    # 「いつ作ったのか」「2つのうちどちらが新しいのか」がファイル名からも
    # 中身からも分からない（fix5 決定6）。fitted_at は同じ時刻の ISO 8601 で、
    # weights_id と同じ瞬間を指す（2回 now() を呼ばず1つの値から作る）。
    fitted_at_dt = datetime.now(timezone.utc)
    weights_id = fitted_at_dt.strftime("%Y%m%dT%H%M%SZ")
    fitted_at = fitted_at_dt.isoformat(timespec="seconds")
    data_version_used = combine_data_versions(used_races)
    versions = dict(records_store.current_versions(
        data_version=data_version_used,
        model=consistent_versions["model"],
        weights_id=weights_id))
    # fix round 2 Critical 1: current_versions() は feature_version/
    # prompt_version/integration_version を「実行時のコードの値」で埋める。
    # check_compatible が実際に照合するのはこの3つなので、学習に使った記録側
    # の値で上書きする（data_version は照合に使われないので current_versions()
    # が組んだ複数版カンマ連結のままでよい）。
    versions["feature_version"] = consistent_versions["feature_version"]
    versions["prompt_version"] = consistent_versions["prompt_version"]
    versions["integration_version"] = consistent_versions["integration_version"]

    final_weights = ig.Weights(
        weights_id=weights_id,
        feature_names=chosen_feature_names,
        weights=dict(zip(chosen_feature_names, chosen["fit_result"].weights.tolist())),
        standardization=dict(chosen["fit_result"].standardization),
        calibration=calibration_map,
        versions=versions,
        lambda_=chosen["lambda"],
        dropped_features=chosen["fit_result"].dropped_features,
        splits={
            "train_until": args.train_until,
            "dev_until": args.dev_until,
            "calib_until": args.calib_until,
        },
        fitted_at=fitted_at,
        # λ を選んだ根拠を重みファイルに残す（仕様 §11.1.2）。採用した構成の
        # dev 対数尤度と、比較した両特徴量構成の値の両方を入れる――標準出力に
        # しか出さないと、あとから「なぜこの λ なのか」を確かめられない。
        dev_metrics={
            "chosen_configuration": chosen_name,
            "lambda": chosen["lambda"],
            "dev_log_likelihood_per_race": chosen["dev_metric"],
            "converged": chosen["fit_result"].converged,
            "n_train_races": len(buckets["train"]),
            "n_dev_races": len(buckets["dev"]),
            "n_calib_races": len(buckets["calib"]),
            "float64_indistinguishable_features": list(indistinguishable),
            "configurations": {
                name: {
                    "n_features": len(feature_configs[name]),
                    "lambda": results[name]["lambda"],
                    "dev_log_likelihood_per_race": results[name]["dev_metric"],
                    "converged": results[name]["fit_result"].converged,
                }
                for name in feature_configs
            },
        },
    )

    args.out.mkdir(parents=True, exist_ok=True)
    weights_path = args.out / f"{weights_id}.json"
    final_weights.save(weights_path)

    print("\n--- 採用したモデル ---")
    print(f"採用した特徴量セット: {chosen_name}")
    print(f"採用した特徴量 ({len(chosen_feature_names)}列): {list(chosen_feature_names)}")
    # 空 tuple (1列も落ちなかった) は「落ちた列の情報が無い」とは違う実在の
    # 値なので、``if ...:`` で隠さず常に表示する (fix round 2 Minor 3)。
    print(f"分散0のため学習対象から外れた特徴量"
          f" ({len(chosen['fit_result'].dropped_features)}列):"
          f" {list(chosen['fit_result'].dropped_features)}")
    print(f"選んだ lambda: {chosen['lambda']}")
    print(f"dev 対数尤度(平均/レース): {chosen['dev_metric']:.4f}")
    print(f"書き出し先: {weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

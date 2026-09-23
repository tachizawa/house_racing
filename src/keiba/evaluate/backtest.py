"""バックテスト: 過去レースを「前日夜に予想したら」として再現し、確定結果と照合する。

``scripts/predict.py``（予想1件）と違い、ここは数百レースをまとめて回す。
**部分失敗の扱いが違う**: predict() は第2層が1頭でも失敗すれば
``allow_partial=False`` で PredictionFailed にするが、それをそのままバック
テストの外へ伝播させると1レースの失敗で残り全部が止まってしまい、
「~100レースで検証する」という目的そのものが果たせない。ここでは1レースの
失敗を ``BacktestRow.skipped`` に変え、残りを続ける。スキップは
``report()`` が理由別件数を必ず出す（黙って落とすと指標がスキップの偏りで
歪む）。

Jev（``predict.pipeline.predict`` 経由）と ``keiba.data`` 以外はこのモジュール
だけで完結する。state の組み立て・重み付け・指標の定義はここでは持たない
（それぞれ features/*.py・predict/integrate.py・evaluate/metrics.py の仕事）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from jev_client import JevRequestError
from keiba import config
from keiba.data.repository import Repository, ResultRepository
from keiba.evaluate import metrics
from keiba.evaluate.splits import SPLIT_NAMES, Splits, assign
from keiba.input import validate
from keiba.input.match import AmbiguousMatchError
from keiba.models import EntryRow, HorseEntry, RaceInput, RaceRow, ResultRow
from keiba.predict.integrate import Weights, WeightsIncompatibleError, rank_by_score, tie_groups
from keiba.predict.pipeline import PredictionFailed, predict
from keiba.store import prompts as store_prompts
from keiba.store import records as store_records
from keiba.store.records import PredictionRecord

__all__ = [
    "PREDICTION_CUTOFF_HOURS",
    "SPLIT_BASIS",
    "SkipReason",
    "BacktestRow",
    "hasso_window_for_cutoff_window",
    "prediction_cutoff_for",
    "race_input_from_history",
    "count_changed_entries",
    "run",
    "report",
]

#: 予想締切を発走の何時間前に置くか。
#:
#: 出典: 仕様 §1.2「予想時点は**前日**（出馬表確定後・発走前日）」。中央の
#: 発走時刻はおよそ 9:50〜16:40 なので、18時間前にすると全レースの締切が
#: 必ず前日（15:50〜22:40）に収まる。仕様 §13 の記録例も締切を前日18:00 に
#: 置いている（fix5 決定10）。
PREDICTION_CUTOFF_HOURS = 18

#: 区分（train/dev/calib/test）の振り分けに使う時刻の基準。
#:
#: ``prediction_cutoff``（発走の前日）で判定する。``hasso_at`` で判定していた
#: ときは ``scripts/fit_weights.py``（``PredictionRecord.prediction_cutoff``
#: の日付で判定する）と各境界の翌開催日が丸ごと1区分ずれ、backtest が
#: 「held-out。学習に使っていません」と銘打ったレースの一部が fit_weights の
#: calib に入っていた（fix5 決定4）。``report()`` がこの値をそのまま
#: ``["run"]["split_basis"]`` に書く。
SPLIT_BASIS = "prediction_cutoff"

#: スキップの理由。理由が分からないスキップを作らない（偏りが見えなくなる）。
SkipReason = Literal[
    "no_winner",
    "single_starter",
    "jump",
    "jev_failed",
    "no_entries",
    "match_ambiguous",
    "input_invalid",
]

#: report() のスキップ内訳で必ず出す全種類。1件も起きていない理由も 0 件として
#: 出す（CLAUDE.md「スキップの件数と理由を必ず出す」。0件も実在する報告）。
_ALL_SKIP_REASONS: tuple[SkipReason, ...] = (
    "no_winner", "single_starter", "jump", "jev_failed",
    "no_entries", "match_ambiguous", "input_invalid",
)


@dataclass(frozen=True)
class BacktestRow:
    """1レースぶんのバックテスト結果。スキップした行も同じ形で持つ
    （skipped が None かどうかで分岐する。指標の集計時にキーが無くて
    落ちることがないように、スキップ行も同じフィールドを埋める）。
    """

    race_id: str
    hasso_at: datetime
    split: str
    skipped: SkipReason | None
    baseline_ranking: tuple[str, ...]
    baseline_probabilities: dict[str, float]
    model_ranking: tuple[str, ...] | None
    model_probabilities: dict[str, float] | None
    outcome: metrics.RaceOutcome | None
    record_id: str | None
    #: 登録後に斤量・騎手が変更された馬の数（spec §4.1.2 の注意）。
    entry_changed_count: int
    #: 同程度の判定に使った尺度（"layer1_win_prob" / "not_measured"）。
    #: スキップ行は "" （そもそも予想していないので、どちらでもない）。
    tie_band_scale: str
    #: 取消・除外された馬（outcome.excluded）に乗っていた確率の合計
    #: （baseline_probabilities 側。fix1 修正3）。該当馬が無ければ 0.0。
    #: baseline を基準にするのは、model が無い run（weights=None）でも
    #: 常に計算できる唯一の確率分布だから。再正規化はしていないので、
    #: 合計が 1.0 未満になっている分の手がかりとしてこの値を使う。
    excluded_probability_mass: float
    #: 取消・除外（outcome.excluded）だった頭数。
    #: **この頭数は本番の予想時点（前日夜）では取消を知り得ないので入力に
    #: 残している**（fix5 決定5）。採点は走った馬だけ（outcome.starters）で
    #: 行うため、取消馬に配られた確率は excluded_probability_mass に残る。
    #: スキップ行は 0（そもそも数えていない）。
    scratched_count: int


# --- 予想締切 ------------------------------------------------------------------


def prediction_cutoff_for(race: RaceRow, *,
                           cutoff_hours: int = PREDICTION_CUTOFF_HOURS) -> datetime:
    """そのレースを「前日夜に予想した」ときの予想締切。

    ``race_input_from_history`` が ``RaceInput.prediction_cutoff`` に入れる値と、
    ``run()`` が区分（train/dev/calib/test）の振り分けに使う値は**同じもので
    なければならない**（fix5 決定4）。両方がこの1つの関数を通る。
    """
    return race.hasso_at - timedelta(hours=cutoff_hours)


def hasso_window_for_cutoff_window(
    start: datetime, end: datetime, *, cutoff_hours: int = PREDICTION_CUTOFF_HOURS,
) -> tuple[datetime, datetime]:
    """予想締切が ``[start, end]`` に入るレースの、発走時刻の範囲を返す。

    締切 = 発走 - cutoff_hours なので、``start <= 締切 <= end`` は
    ``start + cutoff_hours <= 発走 <= end + cutoff_hours`` と同値。
    ``scripts/backtest.py`` の ``--from`` / ``--to``（予想締切で絞る）を
    ``ResultRepository.races_between``（発走時刻で絞る）へ翻訳するために使う。
    ここに置くのは、締切の定義（``prediction_cutoff_for``）と同じ場所に
    置いておかないと、片方だけ直されたときに範囲がずれるため。
    """
    offset = timedelta(hours=cutoff_hours)
    return start + offset, end + offset


# --- race_input の復元 --------------------------------------------------------


def race_input_from_history(
    race: RaceRow,
    entries: Sequence[EntryRow],
    *,
    cutoff_hours: int = PREDICTION_CUTOFF_HOURS,
    top_n: int,
) -> RaceInput:
    """過去のレース1件から、予想締切（発走の ``cutoff_hours`` 時間前）時点で
    利用者が入力していたはずの ``RaceInput`` を復元する。

    ``entries`` は呼び出し側が用意したものをそのまま ``horses`` にする――
    「どの馬を対象にするか」（出走取消などをどう扱うか）は呼び出し側の判断で、
    ここでは絞らない。``run()`` は**結果行を持つ全馬**を渡す（取消馬も残す。
    本番の予想時点では誰が取消になるか知り得ないため。fix5 決定5）。

    ``entry_count`` は渡された馬の数（``len(entries)``）。``race.syusso_tosu``
    ではない――結果から算出される実際の出走頭数は、対象レースがまだ走って
    いない（前日時点）という体で予想する以上、存在しない情報を使うことに
    なってしまう。``validate.check()`` が要求する「entry_count と馬の数の一致」
    も、この定義でしか満たせない。``run()`` が取消馬も残すようになったので、
    この値は登録全頭の数になる（``toroku_tosu`` と一致するのが自然な状態）。

    **馬場状態（``baba_cd``/``baba``）は渡さない（常に ``None``）。**
    ``race`` は ``ResultRepository.races_between`` が確定後の ``RA`` 行から
    作った ``RaceRow`` なので、その ``baba_cd`` は**レース後に確定した馬場
    状態**である。馬場は発走18時間前（予想締切）には確定していない
    （当日発表・天候で変わる）。ここでそのまま渡すと state の ``"baba"`` に
    確定値が入り、``same_baba`` の条件別成績集計が有効化されてしまう――
    本番（``scripts/predict.py``）は利用者申告で、未入力なら「同馬場状態の
    集計は行っていない」（判定不能）になる。バックテストだけが確定後の
    馬場を使えると、本番より有利な条件で評価することになり、指標が実際より
    良く出る（task-19-21 fix High 2: 漏洩）。引数で馬場を渡せる口も意図的に
    作っていない――渡せる口があれば、いずれ「実は分かっている」何かから
    埋めたくなり、同じ漏洩が再発する。
    """
    horses = tuple(
        HorseEntry(
            umaban=entry.umaban,
            bamei=entry.bamei,
            wakuban=entry.wakuban,
            sex_age=entry.sex_age,
            futan_kg=entry.futan_kg,
            # 騎手名は持たない。EntryRow は騎手コード（kisyu_code）しか持たず、
            # コード→名前の対応表がこのプロジェクトに無い。コードをそのまま
            # 名前欄に入れると本物の名前のふりをすることになるので、既定の
            # None のままにする（HorseEntry.kisyu は state の "kisyu" に
            # そのまま出るので、ここで嘘の値を作らないことが重要）。
        )
        for entry in entries
    )
    return RaceInput(
        date=race.hasso_at.date(),
        jyo=race.jyo,
        # race_id は 年4 + 月日4 + 場2 + 回2 + 日目2 + R2 の16桁固定
        # （ingest/records.py の _race_id と同じ構成）。RaceRow は race_num を
        # 別項目として持たないので、race_id の末尾2桁から読む。
        race_num=int(race.race_id[-2:]),
        track_kind=race.track_kind,
        kyori_m=race.kyori_m,
        entry_count=len(entries),
        prediction_cutoff=prediction_cutoff_for(race, cutoff_hours=cutoff_hours),
        horses=horses,
        jyo_cd=race.jyo_cd,
        # baba_cd/baba は渡さない（None）。締切時点では未確定の情報を
        # 渡すと same_baba の集計が漏洩する（このメソッドの docstring 参照。
        # task-19-21 fix High 2）。
        baba_cd=None,
        # class_name は表示用の自由文字列で、class_level (int) からの対応表を
        # このプロジェクトは持たない。build_context は class_name を読まず
        # repo.race().class_level を直接読むので、predict() の正しさには
        # 影響しない（features/context.py 参照）。
        class_name=None,
        baba=None,
        hasso_at=race.hasso_at,
        top_n=top_n,
        race_id=race.race_id,
    )


def count_changed_entries(entries: Sequence[EntryRow]) -> int:
    """登録後に斤量・騎手が変更された馬の数（``EntryRow.changed_after_entry``）。"""
    return sum(1 for entry in entries if entry.changed_after_entry)


# --- run -----------------------------------------------------------------------
#
# 同程度の組の境界を1件の記録から判定する公開ヘルパ（``boundary_tied``）と、その
# ための ``_display_rank`` は fix6 ⑤-2 で削除した。単に未使用だったからではなく、
# **§1 が禁じた規則（model 尺度に組の判定を当てる）を体現していた**ため:
# ``_display_rank`` は rank_model を優先するので model の順位に対して呼べてしまい、
# 返り値が ``bool`` なので「測っていない」を表現できず ``False`` と断言する。
# 指標側の判定は ``_probability_boundary_tied``（baseline 尺度のみ、model は
# ``None``）に1つだけある。


def _why_no_outcome(results: Sequence[ResultRow]) -> SkipReason:
    """勝ち馬が確定しない（``outcome_from_results`` が None を返した）理由を
    1つに決める。0頭・1頭立ては答え合わせそのものが成立しないので、勝ち馬の
    有無に関わらず single_starter を優先する。
    """
    starters = [r for r in results if r.counts_as_starter]
    if len(starters) <= 1:
        return "single_starter"
    return "no_winner"


def _classify_failure(error: PredictionFailed) -> SkipReason:
    """``PredictionFailed`` の原因を3種類に振り分ける。

    ``WeightsIncompatibleError`` から連鎖したものはここに来る前に ``run()``
    が横取りして投げ直す（fix1 修正2。全レースで同じように起きる設定の誤りを
    スキップとして200回数えるのは費用と時間の無駄で、レポートが「API が
    落ちた」ように読めてしまう）ので、ここでは扱わない。

    - ``AmbiguousMatchError`` から連鎖したもの → match_ambiguous
    - ``top_n`` が出走頭数を超えていた（``validate.TOP_N_EXCEEDS_ENTRIES``）
      → single_starter。入力の書式が壊れているわけではなく、「n 頭を選べる
      だけの出走馬がいないレース」であり、1頭立てと同じ理由で採点が成立
      しないだけなので input_invalid には数えない（fix5 決定10）
    - その他の ``入力.validate.check()`` の問題（"入力に問題があります" で始まる
      メッセージ。predict/pipeline.py がこの文言でしか使わない） → input_invalid
    - それ以外（第2層の部分失敗、第1層の Jev リクエスト失敗など、state を
      組み立てたあとで Jev の答えを使うところまで進んでから失敗したもの）
      → jev_failed（理由不明のままスキップを作らない、という制約を守るための
      既定の受け皿）
    """
    if isinstance(error.__cause__, AmbiguousMatchError):
        return "match_ambiguous"
    if str(error).startswith("入力に問題があります"):
        # 文言は validate 側の定数を参照する（同じ日本語文をここに書き写すと、
        # 片方だけ直されたときに分類が静かに input_invalid へ戻る）。
        if validate.TOP_N_EXCEEDS_ENTRIES in str(error):
            return "single_starter"
        return "input_invalid"
    return "jev_failed"


def _skipped_row(race: RaceRow, splits: Splits, reason: SkipReason) -> BacktestRow:
    return BacktestRow(
        race_id=race.race_id,
        hasso_at=race.hasso_at,
        # 区分は予想締切で判定する（fix5 決定4）。hasso_at ではない――
        # fit_weights は記録の prediction_cutoff で判定するので、そちらと
        # 食い違うと同じレースが別の区分に入る。
        split=assign(prediction_cutoff_for(race), splits),
        skipped=reason,
        baseline_ranking=(),
        baseline_probabilities={},
        model_ranking=None,
        model_probabilities=None,
        outcome=None,
        record_id=None,
        entry_changed_count=0,
        tie_band_scale="",
        excluded_probability_mass=0.0,
        scratched_count=0,
    )


def _row_from_record(
    race: RaceRow, splits: Splits, record: PredictionRecord,
    outcome: metrics.RaceOutcome, entries: Sequence[EntryRow],
) -> BacktestRow:
    baseline_order = sorted(record.horses, key=lambda h: h.rank_baseline)
    baseline_ranking = tuple(h.umaban for h in baseline_order)
    baseline_probabilities = {h.umaban: h.layer1_win_prob for h in record.horses}

    has_model = bool(record.horses) and record.horses[0].rank_model is not None
    model_ranking = None
    model_probabilities = None
    if has_model:
        model_order = sorted(record.horses, key=lambda h: h.rank_model)
        model_ranking = tuple(h.umaban for h in model_order)
        model_probabilities = {h.umaban: h.model_win_prob for h in record.horses}

    return BacktestRow(
        race_id=race.race_id,
        hasso_at=race.hasso_at,
        # 区分は予想締切で判定する（fix5 決定4。_skipped_row と同じ規則）。
        split=assign(prediction_cutoff_for(race), splits),
        skipped=None,
        baseline_ranking=baseline_ranking,
        baseline_probabilities=baseline_probabilities,
        model_ranking=model_ranking,
        model_probabilities=model_probabilities,
        outcome=outcome,
        record_id=record.record_id,
        entry_changed_count=count_changed_entries(entries),
        # data_quality["tie_band_scale"] は with_layer2 の有無に関わらず
        # predict() が常に書く（pipeline.py 参照）。.get で既定を埋めない
        # ――キーが無ければそれ自体が predict() 側の欠陥なので落ちるべき。
        tie_band_scale=record.data_quality["tie_band_scale"],
        excluded_probability_mass=sum(
            baseline_probabilities.get(umaban, 0.0) for umaban in outcome.excluded),
        scratched_count=len(outcome.excluded),
    )


def run(
    races: Sequence[RaceRow],
    repo: Repository,
    results: ResultRepository,
    client: Any,
    *,
    splits: Splits,
    n: int,
    with_layer2: bool,
    weights: Weights | None,
    save_records: bool,
    data_dir: Path,
    save_prompts: bool = False,
    on_row: Callable[[BacktestRow], None] | None = None,
) -> list[BacktestRow]:
    """過去レースで予想を作り、確定結果と照合する。

    ``on_row``（任意）は行ができるたびに（スキップ行も含めて）その場で
    呼ばれる。``rows``（戻り値のリスト）は ``run()`` が正常に返るまで
    呼び出し側からは見えないため、``on_row`` 無しで数百レースを回すと、
    途中で想定外の例外（``OSError``・``KeyError`` など）が起きたときに
    ここまで Jev へ払った分ごと全行が失われる（task-19-21 fix 決定6）。
    ``scripts/backtest.py`` はこれを使って行ができるたびに ``rows.jsonl``
    へ追記し、``run()`` が例外で終わってもそれまでの行をディスクに残す。

    **1レースの失敗では止めないが、全レースで同じように起きる設定の誤りは
    即座に止める。両者の違いが設計の要点。**

    予想1件（scripts/predict.py）は第2層が1頭でも失敗したら（allow_partial=
    False のまま）PredictionFailed にしてそこで終わるが、バックテストは1頭の
    失敗で数百レースを止めると使えないので、そのレースを skipped にして続行
    する。スキップは report() が必ず件数と理由を出す。黙って落とすとスキップ
    の偏りで指標が歪む。

    ただし ``WeightsIncompatibleError`` （渡された ``weights`` が現在のコード
    の版・特徴量集合と合っていない）は例外にする。これは「たまたまこの
    レースで起きた失敗」ではなく、この run() 呼び出し全体で同じように起きる
    設定の誤りである。1レース目で分かることを何百回も繰り返して
    ``jev_failed: 200`` のように報告するのは費用と時間の無駄なうえ、レポート
    が実態（API 障害ではなく重みの設定間違い）を誤って伝える（fix1 修正2）。
    ここではスキップにせず、その場で ``WeightsIncompatibleError`` を投げ直し
    て run() 全体を止める。呼び出し側（scripts/backtest.py）がこれを捕まえて
    利用者に伝える。

    ``predict()`` は通常 ``PredictionFailed`` しか外へ投げない契約だが、
    第1層（レース単位、1リクエスト）の Jev 呼び出しは以前 ``run_layer1`` が
    ``PredictionFailed`` に包まずそのまま伝播させる経路を持っていた欠陥が
    あった（fix1 修正1で ``predict/pipeline.py`` 側を直したので、現在は
    ``JevRequestError`` も ``PredictionFailed`` に変換されて出てくる）。
    ここでも1レースを止めないために ``JevRequestError`` を同じく
    skipped=jev_failed として扱う経路は、発生源が直った後の多重の安全網
    として残す。
    """
    rows: list[BacktestRow] = []

    def _emit(row: BacktestRow) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    for race in races:
        if not race.is_flat:
            _emit(_skipped_row(race, splits, "jump"))
            continue

        race_results = results.results(race.race_id)
        if not race_results:
            # 結果データが無い（SE 行ゼロ）。outcome_from_results はこれを
            # None にする（None を _why_no_outcome にそのまま渡すと
            # starters=[] で single_starter に分類されてしまい、データ欠損が
            # 「1頭立て」という正当なスキップに化ける。task-19-21 fix 決定7）。
            _emit(_skipped_row(race, splits, "no_entries"))
            continue

        # OutcomeError（矛盾したデータ: umaban 重複・勝ち馬が starters に
        # 含まれない）はここでは捕まえない。「たまたまこのレースで起きた
        # 失敗」ではなく、データそのものの問題として run() 全体を止める
        # （WeightsIncompatibleError と同じ扱い。task-19-21 fix 決定7）。
        # 現実には到達不能（重複馬番は上流で排除、勝ち馬非 starter も
        # 起きない）だが、将来矛盾が増えても黙って化けさせないための
        # 安全網として残す。
        outcome = metrics.outcome_from_results(race.race_id, race_results)

        if outcome is None:
            _emit(_skipped_row(race, splits, _why_no_outcome(race_results)))
            continue

        # 結果行を持つ全馬を予想の入力にする（取消・除外馬も残す。fix5 決定5）。
        # 本番の予想時点（前日夜）では誰が取消になるか知り得ないので、外すと
        # (1) entry_count が仕様 §4.1.4 が「結果情報。使わない」とした
        # SyussoTosu と定義上同一の数になり、(2) 「絶対に勝てない馬」が消えて
        # 的中率が系統的に楽に出る。採点は走った馬だけ（outcome.starters）。
        entries = [result.entry for result in race_results]
        if len(entries) < 2:
            _emit(_skipped_row(race, splits, "no_entries"))
            continue

        race_input = race_input_from_history(race, entries, top_n=n)
        # 送った物の器。バックテストは Jev への支払いが発生する唯一の経路
        # なので、ここで送った state と質問文を残せないと、検証のたびに同じ
        # レースへもう一度課金することになる（report2 §6 の #7）。
        sent_prompts: dict | None = {} if save_prompts else None
        try:
            record = predict(
                race_input, repo, client,
                with_layer2=with_layer2, weights=weights, allow_partial=False,
                sent_prompts=sent_prompts,
            )
        except PredictionFailed as error:
            if isinstance(error.__cause__, WeightsIncompatibleError):
                # 全レースで同じように起きる設定の誤り。スキップにせず
                # その場で投げ直して run() 全体を止める（fix1 修正2）。
                raise error.__cause__ from error
            _emit(_skipped_row(race, splits, _classify_failure(error)))
            continue
        except JevRequestError:
            # fix1 修正1で発生源（predict/pipeline.py）を直した後の多重の
            # 安全網。現行の predict() はここに来る前に PredictionFailed に
            # 変換するはずだが、経路が増えたときのために残す。
            _emit(_skipped_row(race, splits, "jev_failed"))
            continue

        if save_records:
            store_records.save(record, data_dir=data_dir)
        if sent_prompts is not None:
            store_prompts.save(record, sent_prompts, data_dir=data_dir)
        _emit(_row_from_record(race, splits, record, outcome, entries))
    return rows


# --- report ---------------------------------------------------------------------


def _probability_boundary_tied(probabilities: Mapping[str, float] | None, *, n: int,
                                measured: bool) -> bool | None:
    """1つの確率分布（baseline なら layer1）について、順位 n と n+1 が同じ
    同程度の組にいるか（task-19-21 fix 決定5）。

    ``measured=False``（model 尺度）のときは ``None`` を返す。``config.TIE_BAND``
    は第1層の勝率の run 間ばらつきとして測った値で、model_win_prob の上では
    測っていないので、その尺度では組を作らない（``predict/pipeline.py`` と
    同じ規則。fix5 決定1）。``False``（境界は同組でなかった）や ``0`` 件に
    しないこと——「測っていない」と「0件」は別のことを主張する。

    ``report(rows, n=...)`` の ``n`` に対して毎回ここで計算し直す――
    ``run()`` 時点の ``record.n`` には固定しない。``BacktestRow.
    boundary_tied``（run() 時点の record.n で固定され、かつ「表示に使った
    1つの尺度」しか持たない旧フィールド）は使わない。以前は baseline・model
    の両方がこの1つの値をコピーしていたため、重みを入れた run では
    ``n_boundary_tied_baseline`` と ``_model`` が構造的に常に同値になって
    いた（``boundary_tied()`` が ``_display_rank``＝rank_model 優先で
    順序を決めるため）。ここでは baseline なら
    ``baseline_probabilities``、model なら ``model_probabilities`` 自身の
    分布から、それぞれ独立に ``tie_groups`` を計算する。
    """
    if not measured:
        return None
    if probabilities is None or len(probabilities) <= n:
        return False
    order = rank_by_score(probabilities)
    groups = tie_groups(probabilities, band=config.TIE_BAND)
    return groups[order[n - 1]] == groups[order[n]]


def _metrics_row(row: BacktestRow, *, model: bool, n: int) -> dict | None:
    """``metrics.aggregate`` に渡す1行分の dict にする。

    model=True で「重みを渡さなかったので model_ranking が無い」行
    （``row.skipped is None`` かつ ``row.model_ranking is None``）は
    ``None`` を返す。呼び出し側（``report()``）はこれをリストから除外
    する――``SkipReason`` が付いていないので「スキップ」ではなく、
    ``MetricsReport.n_skipped`` に混ぜると ``skip_breakdown``（SkipReason
    だけを数える）と食い違う（task-19-21 fix 決定8。実測:
    ``model.n_skipped=91`` に対し ``skip_breakdown`` の合計は 0）。

    ``boundary_tied`` は ``_probability_boundary_tied`` で、この行・この
    指標（baseline/model）自身の確率分布から ``n`` に対して計算し直す
    （決定5）。model 側は ``None``（測っていない）になる——model 尺度の
    ばらつきを測っていないので、その尺度で組を作らない（fix5 決定1）。
    """
    if model:
        if row.skipped is None and row.model_ranking is None:
            return None
        ranking, probabilities = row.model_ranking, row.model_probabilities
    else:
        ranking, probabilities = row.baseline_ranking, row.baseline_probabilities

    skipped = row.skipped is not None
    if skipped and not model:
        # スキップ行は判定していないので False（境界が無かった、の側に倒す）。
        # 「測っていない尺度」の None とは別の意味なので混ぜない。
        boundary_tied_value: bool | None = False
    else:
        # 「measured=False（model 尺度）なら None」の規則は
        # _probability_boundary_tied の中に1つだけ置く。
        boundary_tied_value = _probability_boundary_tied(
            probabilities, n=n, measured=not model)
    result: dict = {
        "skipped": skipped,
        "boundary_tied": boundary_tied_value,
        "excluded_probability_mass": row.excluded_probability_mass if not skipped else 0.0,
    }
    if not skipped:
        result["ranking"] = ranking
        result["probabilities"] = probabilities
        result["outcome"] = row.outcome
    return result


#: report() の "run" 節が必ず持つキー（task-19-21 fix 決定3）。run_info 無し
#: で report() を呼んだテスト・用途のために、無ければ全部 None で埋める――
#: キー自体は常に出す。by_split の数字は間引き後のレース列と区分の境界に
#: 依存する（--limit を変えると動く）ので、この節が無いと再現も解釈もできない。
_RUN_INFO_KEYS: tuple[str, ...] = (
    "date_from", "date_to", "limit", "races_in_range",
    "sampled_from", "sampled_to", "with_layer2", "weights_id", "splits",
)


def _run_provenance(run_info: Mapping[str, object] | None) -> dict:
    if run_info is None:
        provenance: dict = dict.fromkeys(_RUN_INFO_KEYS)
    else:
        missing = [key for key in _RUN_INFO_KEYS if key not in run_info]
        if missing:
            raise ValueError(f"run_info に必須キーがありません: {missing}")
        provenance = {key: run_info[key] for key in _RUN_INFO_KEYS}
    # 区分をどの時刻で判定したか。呼び出し側からは渡させない――この report が
    # 実際に使った基準を書く欄なので、run_info（利用者の指定）とは出どころが
    # 違う（fix5 決定4）。
    provenance["split_basis"] = SPLIT_BASIS
    return provenance


def _model_rows_for_aggregate(rows: Sequence[BacktestRow], *, n: int) -> list[dict]:
    """model 側の metrics.aggregate 入力。model_ranking が無い（weights 無し）
    行は「スキップ」ではないのでリストから除く（決定8。_metrics_row 参照）。
    """
    return [r for r in (_metrics_row(row, model=True, n=n) for row in rows) if r is not None]


def _count_without_model_ranking(rows: Sequence[BacktestRow]) -> int:
    return sum(1 for row in rows if row.skipped is None and row.model_ranking is None)


def report(
    rows: Sequence[BacktestRow], *, n: int, run_info: Mapping[str, object] | None = None,
) -> dict:
    """複数レースぶんの ``BacktestRow`` から報告をまとめる。

    スキップの内訳（全7種、起きていない理由も0件として出す）と、区分別
    （train/dev/calib/test）の内訳を必ず含める――どちらも黙って省くと、
    見えない偏りが指標にそのまま乗る（CLAUDE.md、task-19-21-supplement 決定3）。

    ``run_info`` は ``scripts/backtest.py`` の ``main()`` だけが知っている
    来歴（``--from``/``--to``/``--limit``・母集団件数・間引き後の実期間・
    ``with_layer2``・``weights_id``・区分の境界）。この report() 単体では
    再構成できないので、呼び出し側から渡してもらう（task-19-21 fix 決定3）。
    """
    baseline_report = metrics.aggregate(
        [_metrics_row(row, model=False, n=n) for row in rows], n=n)
    model_report = metrics.aggregate(_model_rows_for_aggregate(rows, n=n), n=n)

    skip_breakdown = {
        reason: sum(1 for row in rows if row.skipped == reason)
        for reason in _ALL_SKIP_REASONS
    }

    by_split: dict[str, dict] = {}
    for split_name in SPLIT_NAMES:
        split_rows = [row for row in rows if row.split == split_name]
        by_split[split_name] = {
            "baseline": metrics.aggregate(
                [_metrics_row(row, model=False, n=n) for row in split_rows], n=n),
            "model": metrics.aggregate(_model_rows_for_aggregate(split_rows, n=n), n=n),
            "skip_breakdown": {
                reason: sum(1 for row in split_rows if row.skipped == reason)
                for reason in _ALL_SKIP_REASONS
            },
            "n_without_model_ranking": _count_without_model_ranking(split_rows),
        }

    # 尺度の内訳（決定3）。スキップ行の "" は数えない――判定していないので、
    # どちらの尺度でもない。
    tie_band_scale_breakdown: dict[str, int] = {}
    for row in rows:
        if row.skipped is not None or not row.tie_band_scale:
            continue
        tie_band_scale_breakdown[row.tie_band_scale] = (
            tie_band_scale_breakdown.get(row.tie_band_scale, 0) + 1)

    return {
        "n": n,
        "n_races": baseline_report.n_races,
        "baseline": baseline_report,
        "model": model_report,
        "compare": metrics.compare(baseline_report, model_report),
        "skip_breakdown": skip_breakdown,
        "by_split": by_split,
        "entry_changed_count": sum(row.entry_changed_count for row in rows),
        "tie_band_scale_breakdown": tie_band_scale_breakdown,
        "scratched_count": sum(row.scratched_count for row in rows),
        # model_ranking が無い（weights 無し）行の数。n_skipped には混ぜない
        # （決定8）ので、これで別途見えるようにする。
        "n_without_model_ranking": _count_without_model_ranking(rows),
        "run": _run_provenance(run_info),
    }

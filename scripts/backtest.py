"""過去レースをまとめて予想し、確定結果と照合する CLI。

    .venv/Scripts/python.exe scripts/backtest.py --from 20250101 --to 20260901 \\
        --n 3 --limit 100 --no-layer2

配線だけをここに置く。1レースぶんの再現は ``keiba.evaluate.backtest``、
指標の定義は ``keiba.evaluate.metrics``、時系列分割は ``keiba.evaluate.splits``
に任せる。

``--limit`` は既定で 100 を付ける。Jev の費用は出走頭数×レース数で効くため、
歯止め無しでは事故がそのまま課金額になる。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config, console  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore, DataVersionError  # noqa: E402
from keiba.evaluate import backtest as bt  # noqa: E402
from keiba.evaluate.metrics import MetricsReport  # noqa: E402
from keiba.evaluate.splits import SPLIT_NAMES, Splits, SplitError, by_fraction  # noqa: E402
from keiba.predict.integrate import Weights, WeightsIncompatibleError  # noqa: E402

from jev_client import JevConfigError, create_client  # noqa: E402

JST = timezone(timedelta(hours=9))


def _parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _sample_evenly(races: list, limit: int | None) -> list:
    """発走日時の昇順に並んだ ``races`` から、位置が均等になるように
    ``limit`` 件を間引いて選ぶ（fix3 決定1）。

    先頭から ``limit`` 件を取ると対象期間の先頭に固まる（実測: 7週間を
    指定したのに --limit 100 の結果が最初の3日分に収まり、train/dev/calib/
    test の時系列分割で dev と test が0件になった）。ここでは元の列の順序
    （時系列順）を保ったまま等間隔にインデックスを選ぶので、選ばれた
    レースは引き続き発走日時の昇順で、対象期間全体に広がる。
    """
    n = len(races)
    if limit is None or n <= limit:
        return list(races)
    if limit <= 0:
        return []
    if limit == 1:
        return [races[0]]
    step = (n - 1) / (limit - 1)
    indices = [min(round(i * step), n - 1) for i in range(limit)]
    return [races[i] for i in indices]


def _print_sample_summary(races_in_range: list, races: list) -> None:
    """実際に選ばれたレースの件数と期間を必ず表示する（fix3 決定1）。

    ``--from``/``--to`` は利用者が「指定した」期間で、``races`` の実際の
    発走日の範囲とは別物になりうる（間引きが起きなければ一致するが、
    間引きが起きたときこそ実際の範囲を示す必要がある）。
    """
    if len(races) < len(races_in_range):
        print(f"対象 {len(races_in_range)} 件から {len(races)} 件を均等に間引きました"
              f"（{races[0].hasso_at.date()} 〜 {races[-1].hasso_at.date()}）")
    else:
        print(f"対象 {len(races)} 件をすべて使用します"
              f"（{races[0].hasso_at.date()} 〜 {races[-1].hasso_at.date()}）")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="過去レースをまとめて予想し、確定結果と照合する。"
                     "日付（--from / --to / --*-until）はすべて予想締切"
                     "（発走の前日）で判定します。")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                         help="データ版の場所 (既定 %(default)s)")
    parser.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD",
                         help="対象期間の開始（両端を含む）。"
                              "日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD",
                         help="対象期間の終了（両端を含む）。"
                              "日付は予想締切（発走の前日）で判定します")
    parser.add_argument("--train-until", metavar="YYYYMMDD", default=None,
                         help="train 区分の最終日。日付は予想締切（発走の前日）で"
                              "判定します。省略時は3つとも省略し by_fraction で自動")
    parser.add_argument("--dev-until", metavar="YYYYMMDD", default=None)
    parser.add_argument("--calib-until", metavar="YYYYMMDD", default=None)
    parser.add_argument("--n", type=int, default=config.DEFAULT_TOP_N,
                         help="上位何頭で指標を測るか (既定 %(default)s)")
    parser.add_argument("--layer2", action=argparse.BooleanOptionalAction, default=True,
                         help="第2層 (馬単位・5軸) を使うか (既定: 使う)")
    parser.add_argument("--weights", default=None, metavar="WEIGHTS_ID",
                         help="統合に使う重み (data/weights/<WEIGHTS_ID>.json)。"
                              "省略時は baseline (第1層の勝率のみ) で順位を出す")
    parser.add_argument("--save-records", action=argparse.BooleanOptionalAction, default=True,
                         help="予測記録を保存するか (既定: 保存する。fit_weights.py の入力になる)")
    parser.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=True,
                         help="Jev へ実際に送った state と質問文を data/prompts/ に"
                              "保存するか (既定: 保存する。18頭のレースで約244KB、"
                              "100レースで 15〜25MB)")
    parser.add_argument("--limit", type=int, default=100,
                         help="対象レース数の上限。Jev の費用は出走頭数×レース数で"
                              "効くための歯止め (既定 %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("data/backtests"),
                         help="report.json / rows.jsonl の出力先 (既定 %(default)s)")
    return parser.parse_args(argv)


def _cutoff_dates(races) -> list[date]:
    """区分の境界を自動で決めるための日付列（予想締切の日付）。

    ``bt.run`` が区分を予想締切で振り分けるので、境界を決める側も同じ時刻で
    見る（fix5 決定4）。発走日で境界を決めて締切で振り分けると、境界の翌
    開催日が1区分ずれる。
    """
    return [bt.prediction_cutoff_for(race).date() for race in races]


def _build_splits(args: argparse.Namespace, races) -> Splits:
    given = (args.train_until, args.dev_until, args.calib_until)
    if all(v is not None for v in given):
        return Splits(
            train_until=_parse_yyyymmdd(args.train_until),
            dev_until=_parse_yyyymmdd(args.dev_until),
            calib_until=_parse_yyyymmdd(args.calib_until),
        )
    if any(v is not None for v in given):
        raise SplitError(
            "--train-until / --dev-until / --calib-until は3つとも指定するか、"
            "3つとも省略してください（一部だけの指定はできません）。")
    return by_fraction(_cutoff_dates(races))


def _row_to_json(row: bt.BacktestRow) -> dict:
    return {
        "race_id": row.race_id,
        "hasso_at": row.hasso_at.isoformat(),
        "split": row.split,
        "skipped": row.skipped,
        "baseline_ranking": list(row.baseline_ranking),
        "baseline_probabilities": row.baseline_probabilities,
        "model_ranking": list(row.model_ranking) if row.model_ranking is not None else None,
        "model_probabilities": row.model_probabilities,
        "outcome": dataclasses.asdict(row.outcome) if row.outcome is not None else None,
        "record_id": row.record_id,
        "entry_changed_count": row.entry_changed_count,
        "tie_band_scale": row.tie_band_scale,
        "excluded_probability_mass": row.excluded_probability_mass,
        "scratched_count": row.scratched_count,
    }


def _to_jsonable(value: Any) -> Any:
    """report() の戻り値（MetricsReport を含む dict）を JSON へ変換する。

    NaN は書かない（task-19-21 fix 決定3）。``json.dumps`` は既定で NaN を
    非標準のリテラル ``NaN`` として書いてしまい（``json.loads`` は読めても
    jq・JS の ``JSON.parse`` は拒否する）、実測で典型的な run につき40箇所
    出ていた。``n_races == 0`` のときの各指標（``_mean([])`` が返す NaN）は
    「測っていない」であって「0だった」ではないので、``0.0`` にはせず
    ``null``（Python 側は ``None``）にする。
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _skip_count(result: dict) -> int:
    """「スキップ」件数。``SkipReason`` が付いた行だけを指す（fix2）。

    ``MetricsReport.n_skipped`` は baseline/model それぞれの分母から除いた
    行数で、model 側は「``weights`` を渡さなかったので model_ranking が
    無い」行もここに数えてしまう（``keiba.evaluate.backtest._metrics_row``
    の仕様。``evaluate/backtest.py`` はこのラウンドで変更できない）。
    その値をそのまま「スキップ」として表示すると、SkipReason が1件も
    無いのに「n_skipped: 5」と出て ``skip_breakdown``（全部0件）と食い違う。
    ここでは表示用に ``skip_breakdown`` の合計（＝ SkipReason が付いた行数、
    常に正しい）を「スキップ」の唯一の定義として使う。
    """
    return sum(result["skip_breakdown"].values())


_SPLIT_EMPTY_CONSEQUENCE: dict[str, str] = {
    "train": "train が空だと学習できません。",
    "dev": "fit_weights.py は dev が空だと lambda を選べません。",
    "calib": "calib が空だと較正できません。",
    "test": "test が空だと held-out の評価ができません。",
}


def _split_warnings(result: dict) -> list[str]:
    """区分（train/dev/calib/test）のどれかが0件なら警告を1件作る（fix3 決定2）。

    --limit が期間の先頭に偏ってレースを取っていた欠陥（fix3 欠陥2）や、
    対象期間そのものが狭いなどの理由で特定の区分が0件になりうる。
    ``nan%`` が並ぶだけでは原因が分からないので、レポートの先頭で理由付きの
    警告を出す。
    """
    warnings: list[str] = []
    for split_name in SPLIT_NAMES:
        if result["by_split"][split_name]["baseline"].n_races == 0:
            warnings.append(
                f"警告: {split_name} 区分が 0 件です。--train-until / --dev-until / "
                "--calib-until を明示するか、対象期間を広げてください。"
                f"{_SPLIT_EMPTY_CONSEQUENCE[split_name]}"
            )
    return warnings


def _print_metrics_block(label: str, m: MetricsReport, *, skip_count: int) -> None:
    """decision3: 的中率と併せて境界の件数を必ず出す。的中率だけを単独で
    出さない（境界の件数を隠すと、精度があるように見える数字だけが残る）。

    件数の行は0件でも必ず出す（decision3）が、「※」の注意書きは1件以上の
    ときだけ出す（fix2）。0件のときに「この0件は順位が再現しない」と
    書いても意味を成さない。
    """
    hits = round(m.top_n_hit_rate * m.n_races) if m.n_races else 0
    print(f"\n[{label}] 上位{m.n}頭の的中率: {m.top_n_hit_rate:.1%}（{hits}/{m.n_races}）")
    # n_boundary_tied は None（測っていない）になりうる。0 件と書き分ける
    # ――model 尺度の run 間ばらつきは測っていないので、その尺度では同程度の
    # 判定そのものを行わない（fix5 決定1）。
    if m.n_boundary_tied is None:
        print("  n位の境界が同程度の組の中だったか: 測っていません"
              "（モデル尺度のばらつきは未測定のため、同程度の判定を行っていません）")
    else:
        print(f"  うち n位の境界が同程度の組の中だった: {m.n_boundary_tied} 件")
        if m.n_boundary_tied > 0:
            print(f"  ※この{m.n_boundary_tied}件は順位が再現しないので、"
                  "的中率はこの分だけ run ごとに動きます")
    print(f"  期待的中数から計算した値（確率ベース、run に依存しない）: "
          f"{m.expected_top_n_mean:.3f}")
    print(f"  1着的中率: {m.top1_win_rate:.1%}")
    print(f"  brier: {m.brier:.4f}  log_loss: {m.log_loss:.4f}")
    if m.n_winner_probability_clipped > 0:
        # 件数が1件以上のときだけ注意書きを出す（decision3 と同じ方針。
        # task-19-21 fix 決定9）。log_loss の値そのものは変えない――
        # clip されたレースの寄与は -log(eps) という定数であり、この件数が
        # 多いほど headline の log_loss は eps の選び方で動く（実測: 100
        # レース中44件が clip され、eps を 1e-9 にすると 27.63 が 4.4 に
        # なるほど headline が変わる）。
        print(f"  うち {m.n_winner_probability_clipped} 件は勝ち馬の確率が 0 で"
              "下限に張り付いています。この値は下限の定数で動くので、"
              "絶対値を他と比べないこと")
    caveat = "（その分だけ Brier は悪く出ます）" if m.n_excluded_with_probability > 0 else ""
    print(f"  うち取消・除外馬に確率が乗っていたレース: "
          f"{m.n_excluded_with_probability} 件{caveat}")
    # n_skipped は m.n_skipped ではなく skip_count（SkipReason ベース）を使う
    # （fix2。理由は _skip_count の docstring 参照）。
    print(f"  n_races: {m.n_races}  n_skipped: {skip_count}")


def _print_report(result: dict, *, has_model_ranking: bool, warnings: list[str]) -> None:
    skip_count = _skip_count(result)

    # 警告はレポートの先頭に出す（fix3 決定2）。nan% が並ぶ前に、なぜ区分が
    # 空なのかが分かる必要がある。
    if warnings:
        print("\n--- 警告 ---")
        for warning in warnings:
            print(warning)

    print(f"\n対象レース数（分母, n={result['n']}）: {result['n_races']}")

    print("\n--- スキップ内訳 ---")
    for reason, count in result["skip_breakdown"].items():
        print(f"  {reason}: {count} 件")

    print("\n--- 同程度の判定に使った尺度の内訳 ---")
    if not result["tie_band_scale_breakdown"]:
        print("  (該当レースなし)")
    for scale, count in result["tie_band_scale_breakdown"].items():
        print(f"  {scale}: {count} 件")

    print(f"\n登録後に斤量・騎手が変更された馬の数（分母に入ったレースのみ）: "
          f"{result['entry_changed_count']} 頭")
    print(f"取消・除外だった頭数（分母に入ったレースのみ。この頭数は本番の"
          f"予想時点では取消を知り得ないので入力に残しています）: "
          f"{result['scratched_count']} 頭")

    # test（held-out）区分を最初に、明記して出す（task-19-21 fix 決定4）。
    # 重みを入れた瞬間、一番目立つ「全行平均」には学習に使った train 区分の
    # レースも含まれ、held-out の数字が一番見えにくい構図になっていた
    # （実測でそうなった）。
    print("\n=== test 区分（held-out。学習に使っていません）===")
    test_split = result["by_split"]["test"]
    test_skip_count = sum(test_split["skip_breakdown"].values())
    if test_split["baseline"].n_races == 0:
        print("  該当レースなし")
    else:
        _print_metrics_block("baseline (test)", test_split["baseline"],
                              skip_count=test_skip_count)
        if has_model_ranking and test_split["model"].n_races > 0:
            _print_metrics_block("model (test)", test_split["model"],
                                  skip_count=test_skip_count)

    overall_heading = "\n=== 全行平均 ==="
    if has_model_ranking:
        overall_heading = ("\n=== 全行平均（学習に使ったレースを含みます。"
                            "held-out の数字は上の test 区分を見てください）===")
    print(overall_heading)
    _print_metrics_block("baseline", result["baseline"], skip_count=skip_count)
    if has_model_ranking:
        _print_metrics_block("model", result["model"], skip_count=skip_count)
    else:
        print("\n[model] 重みを指定していないのでモデル順位はありません"
              "（--weights で指定します）")

    print("\n--- 差分 (model - baseline) ---")
    if has_model_ranking:
        for key, value in result["compare"].items():
            print(f"  {key}: {value}")
    else:
        print("  (model が無いので差分はありません)")

    print("\n--- 区分別（test を最初に表示。上の MetricsReport 全体は各区分の"
          "節に対応） ---")
    for split_name in ("test", "train", "dev", "calib"):
        sub = result["by_split"][split_name]
        baseline_n = sub["baseline"].n_races
        if baseline_n == 0:
            # 0件の区分は nan% を出さない（above の警告で理由は既に出ている）。
            print(f"  {split_name}: 該当レースなし")
            continue
        line = (f"  {split_name}: baseline n={baseline_n} "
                f"top_n_hit_rate={sub['baseline'].top_n_hit_rate:.1%} "
                f"brier={sub['baseline'].brier:.4f} log_loss={sub['baseline'].log_loss:.4f}")
        model_n = sub["model"].n_races
        if has_model_ranking and model_n > 0:
            line += (f" / model n={model_n} "
                      f"top_n_hit_rate={sub['model'].top_n_hit_rate:.1%} "
                      f"brier={sub['model'].brier:.4f} log_loss={sub['model'].log_loss:.4f}")
        print(line)


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    args = _parse_args(argv)

    try:
        store = DataStore.open_current(args.data_dir)
    except DataVersionError as error:
        print(f"データ版を開けません: {error}")
        return 1

    repo = Repository(store)
    result_repo = ResultRepository(store)

    # --from / --to は予想締切（発走の前日）で判定する（fix5 決定4）。
    # races_between は発走時刻で絞るので、締切の範囲を発走の範囲へ翻訳する
    # （翻訳の規則は keiba.evaluate.backtest 側に1つだけ置いてある）。
    cutoff_start = datetime.combine(
        _parse_yyyymmdd(args.date_from), datetime.min.time(), tzinfo=JST)
    cutoff_end = datetime.combine(
        _parse_yyyymmdd(args.date_to), datetime.max.time(), tzinfo=JST)
    start, end = bt.hasso_window_for_cutoff_window(cutoff_start, cutoff_end)
    races_in_range = result_repo.races_between(start, end)
    if not races_in_range:
        print(f"対象期間にレースがありません: {args.date_from} 〜 {args.date_to}")
        return 1

    # 先頭から --limit 件を取ると対象期間の先頭に固まり、時系列分割の
    # dev/test が0件になりうる（fix3 決定1）。期間全体から均等に間引く。
    races = _sample_evenly(races_in_range, args.limit)
    if not races:
        print(f"--limit の指定によりレースが0件になりました: --limit {args.limit}")
        return 1
    _print_sample_summary(races_in_range, races)

    try:
        splits = _build_splits(args, races)
    except SplitError as error:
        print(f"区分の指定に問題があります: {error}")
        return 1

    weights = None
    if args.weights is not None:
        weights_path = args.data_dir / "weights" / f"{args.weights}.json"
        try:
            weights = Weights.load(weights_path)
        except OSError as error:
            print(f"重みファイルを読み込めません: {weights_path} ({error})")
            return 1
        except KeyError as error:
            # fix5 決定6 で fitted_at / dev_metrics を必須にした。それより前に
            # 書かれた重みファイルにはこの欄が無い。生のトレースバックを
            # 出さず、何をすればよいかを伝える（scripts/predict.py と同じ）。
            print(f"重みファイルに必要な欄がありません（{error}）: {weights_path}。"
                  " fit_weights.py で学習し直してください。")
            return 1

    try:
        client = create_client()
    except JevConfigError as error:
        print(error)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # report() 単体では再構成できない来歴（task-19-21 fix 決定3）。
    # by_split の数字は間引き後のレース列と区分の境界（--limit を変えると
    # 動く）に依存するので、これが無いと再現も解釈もできない。
    run_info = {
        "date_from": args.date_from,
        "date_to": args.date_to,
        "limit": args.limit,
        "races_in_range": len(races_in_range),
        "sampled_from": races[0].hasso_at.date().isoformat(),
        "sampled_to": races[-1].hasso_at.date().isoformat(),
        "with_layer2": args.layer2,
        "weights_id": weights.weights_id if weights is not None else None,
        "splits": splits.as_dict(),
    }

    rows_path = out_dir / "rows.jsonl"
    rows: list[bt.BacktestRow] = []
    # rows.jsonl は run() が返る前に1行ずつ追記する（task-19-21 fix 決定6）。
    # rows はローカル変数でしかなく、途中（例えば90レース目）で OSError や
    # KeyError のような想定外の例外が起きると run() は戻らない。返り値を
    # 待ってから書くと、そこまで Jev に払った分（層2ありなら1レースあたり
    # 数十リクエスト）ごと全行が失われる。
    with rows_path.open("w", encoding="utf-8") as handle:
        def _write_row(row: bt.BacktestRow) -> None:
            handle.write(json.dumps(_row_to_json(row), ensure_ascii=False))
            handle.write("\n")
            handle.flush()

        try:
            with client:
                rows = bt.run(
                    races, repo, result_repo, client,
                    splits=splits, n=args.n, with_layer2=args.layer2, weights=weights,
                    save_records=args.save_records, save_prompts=args.save_prompts,
                    data_dir=args.data_dir,
                    on_row=_write_row,
                )
        except WeightsIncompatibleError as error:
            # run() は「全レースで同じように起きる設定の誤り」をスキップに
            # せずここまで投げ直す（fix1 修正2）。重みの互換性チェックは
            # predict() 内で Jev を呼ぶより前（state を組み立てる前）に
            # 行われるので、1レース目で判明した時点で Jev への課金は
            # 発生していない。rows.jsonl はここまでの行（多くは空）を
            # 保持したまま残る。
            print(f"重みと現在のコードが一致しません: {error}")
            return 1

    # report.json は最後にまとめて書く（部分的なレポートは誤解を招くので
    # 書かない。決定6）。
    result = bt.report(rows, n=args.n, run_info=run_info)
    has_model_ranking = any(row.model_ranking is not None for row in rows)
    warnings = _split_warnings(result)

    report_dict = _to_jsonable(result)
    report_dict["warnings"] = warnings
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"出力先: {out_dir}")
    _print_report(result, has_model_ranking=has_model_ranking, warnings=warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

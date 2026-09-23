"""利用者入力の JSON から、次に行われる1レースを予想する CLI。

    .venv/Scripts/python.exe scripts/predict.py --input race.json --no-layer2
    .venv/Scripts/python.exe scripts/predict.py --input race.json --layer2 --weights w1

配線だけをここに置く。入力の読み込みは input/schema.py、照合・特徴量・Jev
呼び出し・統合は predict/pipeline.py、保存と表示用ビューは store/records.py に
任せる。ここが持つのは引数の解釈と、失敗したときに理由を日本語で出すことだけ。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config, console  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore, DataVersionError  # noqa: E402
from keiba.features.pace import fit_pace_definition  # noqa: E402
from keiba.input.schema import InputError, load_race_input  # noqa: E402
from keiba.predict.integrate import Weights  # noqa: E402
from keiba.predict.pipeline import PredictionFailed, predict  # noqa: E402
from keiba.store.prompts import save as save_prompts  # noqa: E402
from keiba.store.records import save, to_user_view  # noqa: E402

from jev_client import JevConfigError, create_client  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jev を使って次のレース1件を予想し、上位n頭を表示する。")
    parser.add_argument("--input", required=True, type=Path,
                         help="利用者入力の JSON (必須)")
    parser.add_argument("--n", type=int, default=None,
                         help="上位何頭 (既定: 入力の top_n。指定すると入力側を上書きする)")
    parser.add_argument("--layer2", action=argparse.BooleanOptionalAction, default=True,
                         help="第2層 (馬単位・5軸) を使うか (既定: 使う)")
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=False,
                         help="展開予測を第1層に含めるか (既定: 含めない)")
    parser.add_argument("--weights", default=None, metavar="WEIGHTS_ID",
                         help="統合に使う重み (data/weights/<WEIGHTS_ID>.json)。"
                              "省略時は baseline (第1層の勝率のみ) で順位を出す")
    parser.add_argument("--allow-partial", action="store_true",
                         help="第2層の一部の馬が失敗しても PARTIAL として結果を返す"
                              " (既定: 1頭でも失敗したら予想を打ち切る)")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                         help="データ版の場所 (既定 %(default)s)")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                         help="予測記録を保存するか (既定: 保存する)")
    parser.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=True,
                         help="Jev へ実際に送った state と質問文を "
                              "data/prompts/ に保存するか (既定: 保存する)")
    return parser.parse_args(argv)


def resolve_with_pace(*, requested: bool, pace_definition: dict | None) -> bool:
    """実際に pace 質問を送るかどうか。

    ``--pace`` を指定しても ``pace_definition`` が作れなかった（層別化に必要な
    レース数が学習期間に足りない。仕様 §8.2 自身が「取得初期は成立しない前提」
    と書いている状況）ときは ``False`` にする。

    pace のプロンプトは state の ``pace_definition`` を名指しして「区分の意味は
    pace_definition に従ってください」と指示するが、``pace_definition`` が
    ``None`` のとき state にそのキーは存在しない（``features/summary.py``）。
    存在しないキーを名指しするプロンプトを送ることは仕様 §8.2 が禁じている
    ――モデルは指示を無視するか自分で材料を選び、どちらも例外を出さず出力は
    もっともらしいままなので、送ってしまうと気づけない（fix5 決定7）。

    判定をこの1つの関数に置くのは、「送る質問」の規則をテスト側に書き写さない
    ため（``tests/test_questions.py`` の「プロンプトが名指しするキーは state に
    実在する」検査がこの関数を呼ぶ）。

    **同じ判定は ``predict()`` の冒頭にもある**（fix6 ⑤-1）。あちらが最後の砦で、
    ``predict()`` を直接呼ぶ経路でもプロンプトを送らせない。こちらは「利用者に
    何を表示するか」を決めるために使う——意図した二重化で、どちらも
    ``pace_definition is None`` という同じ1つの条件しか見ない。
    """
    return requested and pace_definition is not None


def _print_view(view: dict) -> None:
    """to_user_view の内容を人が読める形で表示する。

    見出しの「上位n頭」は record.n (利用者が求めた頭数) のまま――同程度の
    組で表示が n を超えても (store/records.py 決定4)、それが分かるのは
    「同程度」の見出しからであって、上位見出しの数字を変えるからではない。
    """
    print(f"予想時刻    : {view['予想時刻']}")
    print(f"予想締切    : {view['予想締切']}")
    print(f"データ更新時刻: {view['データ更新時刻']}")
    print(f"注意        : {view['注意']}")
    if view["注記"]:
        print("\n--- 注記 ---")
        for note in view["注記"]:
            print(f"  {note}")
    print(f"\n--- 上位{view['上位n頭']}頭 ---")
    # 同程度の判定を行っていない実行では、その1行を出す (fix5 決定1)。
    # 文言は to_user_view と同じ値をそのまま表示する (同じ日本語文を2箇所に
    # 書かない)。キーが無い = 判定した実行。
    if "同程度の判定" in view:
        print(f"（{view['同程度の判定']}）")
    explanations = view["同程度の組の説明"]
    printed_group: int | None = None
    for horse in view["予想"]:
        if "予想順位" in horse:
            print(f"\n{horse['予想順位']}位  {horse['馬番']}番 {horse['馬名']}")
            printed_group = None
        else:
            # 組の要素が複数頭 (「予想順位」キーが無い)。組が変わるたびに
            # 見出しと説明を1回だけ出す (同じ組の2頭目以降は出さない)。
            group_no = horse["同程度の組"]
            if group_no != printed_group:
                print("\n--- 同程度（順位つけられず）---")
                print(f"  {explanations[str(group_no)]}")
                printed_group = group_no
            print(f"\n     {horse['馬番']}番 {horse['馬名']}")
        for mark, key in (
            ("+", "プラス材料"), ("-", "マイナス材料"), ("?", "判断材料が不足している項目"),
            ("*", "注記"),
        ):
            items = horse[key]
            if items:
                print(f"    {mark} " + " / ".join(items))


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    args = _parse_args(argv)

    try:
        race_input = load_race_input(args.input)
    except InputError as error:
        print(f"入力を読み込めません: {error}")
        return 1

    if args.n is not None:
        race_input = dataclasses.replace(race_input, top_n=args.n)

    try:
        store = DataStore.open_current(args.data_dir)
    except DataVersionError as error:
        print(f"データ版を開けません: {error}")
        return 1

    repo = Repository(store)

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
            # 書かれた重みファイルにはこの欄が無く、load が KeyError になる。
            # 生のトレースバックを出さず、何をすればよいかを伝える。
            print(f"重みファイルに必要な欄がありません（{error}）: {weights_path}。"
                  " fit_weights.py で学習し直してください"
                  "（いつ学習したのか・λ を選んだ根拠を残すために必須の欄です）。")
            return 1

    # --pace は as_of=race_input.prediction_cutoff を必須で渡す。対象レース
    # 自身やそれ以降のレースの走破時計から閾値を作ると、それ自体が未来情報に
    # なる (進捗台帳 所見D の裁定、spec §9.3)。pace_definition の計算は
    # predict() の外 (ここ) で行う――Repository だけでは学習期間のレース集合
    # (ResultRepository.races_between) に手が届かないため。
    pace_definition = None
    if args.pace:
        as_of = race_input.prediction_cutoff
        result_repo = ResultRepository(store)
        # 下限は「取得済みデータの最初から」を素直に表す datetime.min。
        # fit_pace_definition は as_of より前のレースしか使わないので、これで
        # 「取得済み全期間」になる――特定の年を決め打ちにする根拠が無いので
        # 定数を作らない。as_of と同じ tzinfo に揃え、時差をまたいだ比較で
        # タイムゾーンの取り違えが起きないようにする。
        races = result_repo.races_between(
            datetime.min.replace(tzinfo=as_of.tzinfo), as_of)
        pace_definition = fit_pace_definition(races, target=race_input, as_of=as_of)

    # pace_definition が無いなら pace 質問は送らない（fix5 決定7）。
    # 送ると state に無い pace_definition を名指しするプロンプトになる。
    with_pace = resolve_with_pace(requested=args.pace, pace_definition=pace_definition)
    if args.pace and not with_pace:
        print("展開予測: 層別化に必要なレース数が学習期間に足りず"
              "ペース定義が作れなかったのでペース質問は送りません。")

    try:
        client = create_client()
    except JevConfigError as error:
        print(error)
        return 1

    # 送った物の器。--no-save-prompts でも predict() には渡してよいが、
    # 使わない辞書を持ち回る意味は無いので、保存するときだけ渡す。
    sent_prompts: dict | None = {} if args.save_prompts else None

    with client:
        try:
            record = predict(
                race_input, repo, client,
                with_layer2=args.layer2, with_pace=with_pace,
                weights=weights, allow_partial=args.allow_partial,
                pace_definition=pace_definition,
                sent_prompts=sent_prompts,
            )
        except PredictionFailed as error:
            print(f"予想に失敗しました: {error}")
            return 1

    if args.save:
        path = save(record, data_dir=args.data_dir)
        print(f"保存先: {path}")
    if sent_prompts is not None:
        prompts_path = save_prompts(record, sent_prompts, data_dir=args.data_dir)
        print(f"送った state と質問文: {prompts_path}")

    print()
    _print_view(to_user_view(record))
    if record.status != "SUCCEEDED":
        print(f"\n状態: {record.status}"
              "（第2層の一部が失敗しています。data_quality を確認してください）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

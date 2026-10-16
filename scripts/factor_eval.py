"""要素別の予想の過去での評価（仕様 §5）。Jev は呼ばない（課金なし）。

    .venv/Scripts/python.exe scripts/factor_eval.py --data-dir data/_long

要素の点数の計算は重い（年ごとの物差しで30分以上）ので、点数の表を
<cache-dir>/factors_<データ版>_<血統版>_<要素の版の印>.pkl に保存して再利用する（--rebuild で作り直し）。
要素の版の印は registry.versions_tag()。版を上げるとファイル名が変わるので、古い版の点数を取り違えない。

--recompute-table-from PATH: 古い版の点数の表 PATH を読み、表の要素（縮小の要素7つ）だけを今の版で作り直して
新しい名前で保存する。表の要素は走の表と e3・ae だけから決まるので、年ごとの要素を作り直す必要がない。

--compare-to DIR: 前の実行の summary.csv と判定を比べる節をレポートに足す。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from keiba import config  # noqa: E402
from keiba.evaluate.factor_eval import EVAL_YEARS, N_BOOT, SEED, by_year, evaluate, summarize  # noqa: E402
from keiba.evaluate.factor_eval_report import render  # noqa: E402
from keiba.factors import table_factors as tf  # noqa: E402
from keiba.factors.data import load_runs  # noqa: E402
from keiba.factors.registry import FACTOR_NAMES, VERSIONS, compute_all, recompute_table_factors, versions_tag  # noqa: E402
from keiba.features.speed_figure import prepare_runs  # noqa: E402
from keiba.ingest.store import current_version, read_manifest, version_dir  # noqa: E402

JST = timezone(timedelta(hours=9))
SUBSETS = {"op": ("OP・重賞", "op"), "mixed": ("3歳以上の古馬混合戦", "mixed")}
LAMBDA_REPORT_YEARS = (2016, 2017, 2018, 2019, 2025)   # レポートに出す年（全年は lambda.csv）


# factors_<データ版>.pkl（血統版・要素の版を名前に入れる前の古い名前）と factors_<データ版>_<血統版>_<印>.pkl
_CACHE_NAME = re.compile(r"factors_(?P<data>[^_]+)(?:_(?P<blood>[^_]+)_(?P<tag>[^_]+))?\.pkl")


def _current(d: Path) -> str:
    version = current_version(d)
    if version is None:
        raise SystemExit(f"CURRENT がありません: {d}")
    return version


def recompute_source_problem(path: Path, data_version: str, blood_version: str) -> str | None:
    """--recompute-table-from の表が今のデータ版・血統版から作ったものか。違えば理由を返す。

    作り直した表は今の版の名前で保存するので、別の版の表を渡すと中身と名前が食い違ったまま
    以後の実行で再利用される。版はファイル名からしか分からない（表の中には入っていない）。
    """
    m = _CACHE_NAME.fullmatch(path.name)
    if m is None:
        return (f"{path.name} から版を読み取れません。factors_<データ版>_<血統版>_<印>.pkl の名前の"
                "表を渡してください。")
    if m["data"] != data_version:
        return f"{path.name} はデータ版 {m['data']} の表です。今のデータ版は {data_version} です。"
    if m["blood"] is not None and m["blood"] != blood_version:
        return f"{path.name} は血統版 {m['blood']} の表です。今の血統版は {blood_version} です。"
    return None


def cache_path(cache_dir: Path, data_version: str, blood_version: str, versions: dict[str, str] | None = None) -> Path:
    """点数の表のキャッシュの場所。データ版・血統版・要素の版のどれかが違えば別のファイルになる。"""
    return cache_dir / f"factors_{data_version}_{blood_version}_{versions_tag(versions)}.pkl"


def git_head(repo: Path = ROOT) -> str | None:
    """実行したコードの版（git の HEAD）。git が無い・リポジトリでないときは None（記録に残せないだけで止めない）。"""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "_long")
    ap.add_argument("--blood-dir", type=Path, default=config.DATA_DIR / "blood")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=config.DATA_DIR / "factor_eval",
                    help="点数の表のキャッシュを置く場所（--out-dir とは別。既定 data/factor_eval/）")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--recompute-table-from", type=Path, default=None)
    ap.add_argument("--compare-to", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    version = _current(args.data_dir)
    vdir = version_dir(args.data_dir, version)
    blood = _current(args.blood_dir)
    bdir = version_dir(args.blood_dir, blood)
    stamp = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir or (config.DATA_DIR / "factor_eval" / stamp)
    cache = cache_path(args.cache_dir, version, blood)
    years = [EVAL_YEARS[0] - 1] + EVAL_YEARS
    log = lambda m: print(m, flush=True)  # noqa: E731

    reused, source = False, None
    if args.recompute_table_from is not None:
        problem = recompute_source_problem(args.recompute_table_from, version, blood)
        if problem:
            print(problem)
            return 2
    if cache.exists() and not args.rebuild:
        R = pd.read_pickle(cache)
        reused = True
        print(f"要素の点数をキャッシュから読む: {cache}")
        if args.recompute_table_from is not None:
            print("  （作り直した後の表がすでにあるので --recompute-table-from は使っていません。"
                  "作り直すなら --rebuild も付けてください）")
    elif args.recompute_table_from is not None:
        source = str(args.recompute_table_from)
        print(f"古い点数の表から表の要素だけ作り直す: {source}", flush=True)
        R = recompute_table_factors(pd.read_pickle(args.recompute_table_from), log=log)
    else:
        print("走の表を作る …", flush=True)
        read_manifest(vdir)          # 書きかけ・壊れた版はここで止める（DataVersionError）
        read_manifest(bdir)
        R0 = load_runs(vdir, bdir)
        races = pd.read_csv(vdir / "races.csv", dtype=str, na_filter=False)
        horses = pd.read_csv(vdir / "race_horses.csv", dtype=str, na_filter=False)
        runs_all = prepare_runs(races, horses, as_of=datetime(EVAL_YEARS[-1] + 1, 1, 1, tzinfo=JST))
        R = compute_all(R0, runs_all, years=years, log=log)
    if not reused:
        cache.parent.mkdir(parents=True, exist_ok=True)
        R.to_pickle(cache)
        print(f"点数の表を保存: {cache}")

    res = evaluate(R, FACTOR_NAMES, years=EVAL_YEARS, log=log)
    pr = res["per_race"]
    summary = summarize(pr, FACTOR_NAMES, n_boot=args.n_boot, seed=SEED)
    yearly = by_year(pr, FACTOR_NAMES)
    subsets, subset_years = {}, {}
    for key, (label, col) in SUBSETS.items():
        part = pr[pr[col]]
        if part.empty:          # 作り物などで該当レースが無いときは出さない
            continue
        subsets[key] = (label, summarize(part, FACTOR_NAMES, n_boot=args.n_boot, seed=SEED))
        subset_years[key] = by_year(part, FACTOR_NAMES)
    lambdas = tf.lambda_table(R, [y for y in years if y >= tf.FIRST_YEAR])
    previous = pd.read_csv(args.compare_to / "summary.csv", index_col="factor") if args.compare_to else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render(
        summary, yearly, {label: s for label, s in subsets.values()}, res["excluded"], data_version=version,
        years=EVAL_YEARS, lambdas=lambdas[lambdas["year"].isin(LAMBDA_REPORT_YEARS)], versions=VERSIONS,
        previous=previous, previous_label=str(args.compare_to) if args.compare_to else ""), encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv")
    yearly.to_csv(out_dir / "by_year.csv")
    for key, (_, s) in subsets.items():
        s.to_csv(out_dir / f"summary_{key}.csv")
        subset_years[key].to_csv(out_dir / f"by_year_{key}.csv")
    lambdas.to_csv(out_dir / "lambda.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "data_version": version, "blood_version": blood, "eval_years": EVAL_YEARS, "n_boot": args.n_boot,
        "seed": SEED, "factors": VERSIONS, "versions_tag": versions_tag(), "excluded": res["excluded"],
        "git_head": git_head(), "cache": str(cache), "cache_reused": reused, "cache_source": source,
        "compared_to": str(args.compare_to) if args.compare_to else None,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
